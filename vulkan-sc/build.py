#!/usr/bin/env python3
"""Build and validate Vulkan compute shaders and NVIDIA Vulkan SC caches."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
FIXTURES = {
    "07_regex": ("regex.py", "07_regex.c"),
    "08_json_parser": ("json_inputs.py", "08_json_parser.cpp"),
    "09_bytecode_vm": ("bytecode.c", "09_bytecode_vm.c"),
    "10_template_madness": ("templates.cpp", "10_template_madness.cpp"),
}


def pipeline_json(shader, entry):
    """Describe the single storage-buffer binding shared by the shaders."""
    def info(structure, **fields):
        return dict(sType=f"VK_STRUCTURE_TYPE_{structure}_CREATE_INFO", pNext="NULL", flags="0", **fields)

    binding = dict(binding=0, descriptorType="VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
                   descriptorCount=1, stageFlags="VK_SHADER_STAGE_COMPUTE_BIT", pImmutableSamplers="NULL")
    descriptor_layout = info("DESCRIPTOR_SET_LAYOUT", bindingCount=1, pBindings=[binding])
    layout = info("PIPELINE_LAYOUT", setLayoutCount=1, pSetLayouts=["pSetLayouts_0"],
                  pushConstantRangeCount=0, pPushConstantRanges="NULL")
    stage = info("PIPELINE_SHADER_STAGE", stage="VK_SHADER_STAGE_COMPUTE_BIT",
                 module="", pName=entry, pSpecializationInfo="NULL")
    compute = info("COMPUTE_PIPELINE", stage=stage, layout="", basePipelineHandle="", basePipelineIndex=0)
    state = dict(YcbcrSamplers=[], ImmutableSamplers=[], DescriptorSetLayouts=[{"pSetLayouts_0": descriptor_layout}],
                 PipelineLayout=layout, ComputePipeline=compute,
                 ShaderFileNames=[dict(stage="VK_SHADER_STAGE_COMPUTE_BIT", filename=f"{shader}.spv")])
    identifier = hashlib.sha256(f"nv-isa-corpus/vulkan-sc/{shader}".encode()).digest()[:16]
    return dict(ComputePipelineState=state, EnabledExtensions=[], PipelineUUID=list(identifier))


def run(command, log, timeout=180):
    try:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            log.write(f"Exit status {result.returncode}.\n".encode())
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log.write(b"Compiler timed out.\n")
        return False


def compile_shader(source, args):
    name = source.stem
    entry = "main" if source.suffix == ".comp" else "program_entry"
    output = ROOT / "spirv" / f"{name}.spv"
    log_file = ROOT / "logs" / f"{name}.log"
    pipeline = ROOT / "pipelines" / f"{name}.json"
    expected = json.dumps(pipeline_json(name, entry), indent=2) + "\n"
    fixture = FIXTURES.get(name)
    generator = source.parent.parent / "fixtures" / fixture[0] if fixture else None
    original = WORKSPACE / "programs" / fixture[1] if fixture else None
    modified = max(path.stat().st_mtime_ns for path in (source, generator, original) if path)
    if (args.reuse and output.is_file() and pipeline.is_file() and pipeline.read_text() == expected
            and output.stat().st_mtime_ns >= modified):
        with tempfile.TemporaryFile() as log:
            if run([args.spirv_val, "--target-env", "vulkan1.2", output], log, 10):
                return name, True
    if (args.reuse and not output.exists() and not pipeline.exists() and log_file.is_file()
            and log_file.stat().st_size and log_file.stat().st_mtime_ns >= modified):
        return name, False
    with tempfile.TemporaryDirectory() as temp, log_file.open("wb") as log:
        raw, stripped = (Path(temp) / f"{stem}.spv" for stem in ("raw", "stripped"))
        ready = True
        if generator:
            if generator.suffix == ".py":
                generate = [sys.executable, generator, original]
            else:
                executable = Path(temp) / "fixture"
                compiler = ([args.cxx, "-std=c++20"] if generator.suffix == ".cpp" else [args.cc])
                ready = run([*compiler, "-O2", generator, "-o", executable], log, 30)
                generate = [executable]
            if ready:
                with (Path(temp) / "fixture.glsl").open("wb") as include:
                    try:
                        ready = subprocess.run(generate, stdout=include, stderr=log, timeout=30).returncode == 0
                    except subprocess.TimeoutExpired:
                        log.write(b"Fixture generator timed out.\n")
                        ready = False
                if not ready:
                    log.write(b"Fixture generation failed.\n")
        if source.suffix == ".comp":
            command = [args.glslc, "--target-env=vulkan1.2", f"-I{temp}", "-o", raw, source]
        else:
            command = [args.clspv, "-cl-std=CL2.0", "-w", "-O1", "-inline-entry-points",
                       "-spv-version=1.5", "-int8", "-o", raw, source]
        commands = [command, [args.spirv_opt, "--strip-nonsemantic", raw, "-o", stripped],
                    [args.spirv_val, "--target-env", "vulkan1.2", stripped]]
        if not ready or not all(run(command, log, 45) for command in commands):
            output.unlink(missing_ok=True)
            pipeline.unlink(missing_ok=True)
            return name, False
        output.write_bytes(stripped.read_bytes())
        pipeline.write_text(expected)
    return name, True


def valid_cache(data, identifier):
    if len(data) < 56:
        return False
    header_size, version, vendor = struct.unpack_from("<III", data)
    index_count, stride, offset = struct.unpack_from("<IIQ", data, 40)
    return (56 <= header_size <= offset and version == 1000298001 and vendor == 0x10DE
            and index_count == 1 and stride >= 16 and offset + stride <= len(data)
            and data[offset:offset + 16] == identifier)


def compile_cache(name, chip, args):
    shader = ROOT / "spirv" / f"{name}.spv"
    pipeline = ROOT / "pipelines" / f"{name}.json"
    cache = ROOT / "cache" / chip / f"{name}.bin"
    diagnostic = ROOT / "logs" / chip / f"{name}.log"
    cache.parent.mkdir(exist_ok=True)
    diagnostic.parent.mkdir(exist_ok=True)
    description = json.loads(pipeline.read_text())
    identifier = bytes(description["PipelineUUID"])
    override = getattr(args, "pcc_overrides", {}).get(chip)
    compiler = override or args.pcc
    marker = b"PCC override: "
    compiler_matches = not override
    if diagnostic.is_file():
        with diagnostic.open("rb") as log:
            first_line = log.readline()
        compiler_matches = (first_line == f"PCC override: {override}\n".encode()
                            if override else not first_line.startswith(marker))
    if args.reuse and (not override or compiler_matches) and cache.is_file() and cache.stat().st_mtime_ns >= max(
            shader.stat().st_mtime_ns, pipeline.stat().st_mtime_ns):
        if valid_cache(cache.read_bytes(), identifier):
            return name, chip, True
    if (args.reuse and compiler_matches and not cache.exists() and diagnostic.is_file() and diagnostic.stat().st_size
            and diagnostic.stat().st_mtime_ns >= max(shader.stat().st_mtime_ns, pipeline.stat().st_mtime_ns)):
        return name, chip, False
    with tempfile.TemporaryDirectory() as folder, diagnostic.open("wb") as log:
        if override:
            log.write(f"PCC override: {override}\n".encode())
            log.flush()
        scratch = Path(folder)
        # PCC treats input filenames containing "json" as pipeline descriptions.
        (scratch / "input.spv").symlink_to(shader)
        description["ComputePipelineState"]["ShaderFileNames"][0]["filename"] = "input.spv"
        (scratch / "input.json").write_text(json.dumps(description))
        result = scratch / "cache.bin"
        passed = run([compiler, "-path", scratch, "-prefix", "input", "-chip", chip, "-j", "1", "-out", result],
                     log, 180 if override else 40)
        if passed and result.is_file():
            data = result.read_bytes()
            if valid_cache(data, identifier):
                cache.write_bytes(data)
                return name, chip, True
            log.write(b"Invalid Vulkan SC cache header or pipeline identifier.\n")
        cache.unlink(missing_ok=True)
    return name, chip, False


def unported_reason(source):
    name = source.stem
    if name.startswith("syscall_") or name == "asm_syscall_raw":
        return "POSIX kernel interfaces require a host process."
    if name.startswith("asm_") or name == "min_nop_sled":
        return "The workload exercises CPU instructions or a CPU calling context."
    if name == "int_woma_keygen":
        return "Generates activation credentials for third-party software."
    if name in {"03_linked_list", "mem_flexible_arena_rewriter"}:
        return "Requires a GPU-specific bounded arena and pointer representation."
    if name in {"cf_setjmp", "cf_longjmp_threaded_vm", "cf_computed_goto"}:
        return "Requires an explicit shader stack or state machine for host control transfers."
    if name == "int_atomic_orders":
        return "Exercises C atomics on 8-bit and 16-bit storage as well as wider values."
    if name == "call_abi_stress_matrix":
        return "Exercises CPU variadic and aggregate calling conventions."
    if source.suffix == ".cpp":
        return "Requires a dedicated port of its C++ runtime or library features."
    return "Requires porting host library calls, I/O, or dynamic data structures."


def write_table(path, header, rows):
    with path.open("w", newline="") as output:
        table = csv.writer(output, delimiter="\t", lineterminator="\n")
        table.writerow(header)
        table.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clspv", default="clspv")
    parser.add_argument("--glslc", default="glslc")
    parser.add_argument("--cc", default="cc", help="host C compiler for bytecode fixtures")
    parser.add_argument("--cxx", default="c++", help="host C++ compiler for template fixtures")
    parser.add_argument("--spirv-opt", default="spirv-opt")
    parser.add_argument("--spirv-val", default="spirv-val")
    parser.add_argument("--pcc", default="nvidia-pcc")
    parser.add_argument("--pcc-for-chip", metavar="CHIP=EXECUTABLE", action="append", default=[],
                        help="use a different PCC executable for a particular chip")
    parser.add_argument("--chip", nargs="+", default=["ad102"], help="PCC chip names or all")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--reuse", action="store_true", help="reuse current build results and validate successful outputs")
    args = parser.parse_args()
    args.pcc_overrides = {}
    for override in args.pcc_for_chip:
        chip, separator, compiler = override.partition("=")
        if not chip or not separator or not compiler:
            parser.error("--pcc-for-chip requires CHIP=EXECUTABLE")
        args.pcc_overrides[chip] = compiler
    chips = args.chip
    if chips == ["all"]:
        supported = subprocess.check_output([args.pcc, "-chips"], text=True)
        chips = re.findall(r"(?m)^\[INFO\]\s+([a-z]+\d+[a-z]?)(?: .*)?$", supported)
        if not chips:
            parser.error("could not discover NVIDIA PCC chip names")
        chips.extend(chip for chip in args.pcc_overrides if chip not in chips)

    for kind in ("logs", "spirv", "pipelines", "cache"):
        (ROOT / kind).mkdir(exist_ok=True)
    sources = {path.stem: path for path in sorted((ROOT / "opencl").glob("*.cl"))}
    sources.update({path.stem: path for path in sorted((ROOT / "workloads").glob("*.cl"))})
    sources.update({path.stem: path for path in sorted((ROOT / "glsl").glob("*.comp"))})
    with ThreadPoolExecutor(args.jobs) as pool:
        shaders = dict(pool.map(lambda source: compile_shader(source, args), sources.values()))
        buildable = [name for name, success in shaders.items() if success]
        jobs = [(name, chip) for chip in chips for name in buildable]
        caches = list(pool.map(lambda job: compile_cache(*job, args), jobs))
    rows = []
    for source in sorted((WORKSPACE / "programs").iterdir()):
        if source.stem not in shaders:
            rows.append((source.name, "not ported", unported_reason(source)))
        elif shaders[source.stem]:
            rows.append((source.name, "ok", f"spirv/{source.stem}.spv"))
        else:
            rows.append((source.name, "failed", f"logs/{source.stem}.log"))
    write_table(ROOT / "shader-status.tsv", ["program", "spirv_status", "details"], rows)
    write_table(ROOT / "cache-status.tsv", ["program", "chip", "cache_status", "path"],
                ((name, chip, "ok" if ok else "failed",
                  f"cache/{chip}/{name}.bin" if ok else f"logs/{chip}/{name}.log")
                 for name, chip, ok in caches))
    print(f"{len(buildable)} validated shaders, {sum(ok for _, _, ok in caches)} caches across {len(chips)} requested chips")
    return 0 if buildable and all(shaders.values()) and all(ok for _, _, ok in caches) else 1


if __name__ == "__main__":
    raise SystemExit(main())
