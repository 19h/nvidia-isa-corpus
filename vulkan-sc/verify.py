#!/usr/bin/env python3
"""Run representative offline Vulkan SC caches and compare against native C."""

import argparse
import csv
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent

CASES = {
    "call_nested": (("sink", 1, "q"),),
    "call_struct_return": (("sink", 1, "q"),),
    "cf_indirect_call": (("sink", 1, "q"),),
    "cf_duff_device_parser": (("cf_duff_device_parser_sink", 1, "Q"),),
    "cf_irreducible_state_maze": (("cf_irreducible_sink", 1, "Q"),),
    "fp_piecewise_chaos": (("sink", 1, "Q"),),
    "fp_mandelbrot": (("sink", 1, "I"),),
    "int_crc32": (("sink", 1, "I"),),
    "mem_sequential": (("sink", 1, "q"),),
    "mem_strided": (("sink", 1, "q"),),
    "call_tail_recursive": (("sink", 1, "q"),),
    "call_many_int_args": (("sink", 1, "q"),),
    "cf_switch_dense": (("sink", 1, "q"),),
    "cf_switch_sparse": (("sink", 1, "q"),),
    "fp_fft": (("sink", 1, "d"),),
    "int_bigint": (("sink", 1, "I"),),
    "int_bitfields": (("sink", 1, "Q"),),
    "int_galois": (("sink", 1, "B"),),
    "int_int128_ops": (("sink128", 1, "2Q"), ("sink", 17, "Q")),
    "int_popcount": (("sink", 1, "I"),),
    "int_sha256": (("sink", 1, "I"),),
    "mem_tls": (("sink", 1, "Q"),),
    "min_add_chain": (("sink", 1, "i"),),
    "min_bitwise": (("sink", 1, "I"),),
    "min_branch_chain": (("sink", 1, "i"),),
    "min_load_chain": (("sink", 1, "i"),),
    "min_store_chain": (("sink", 1, "i"),),
}


def native_result(compiler, program, fields):
    source = ROOT.parent / "programs" / f"{program}.c"
    outputs = [f"fwrite((const void *)&{sink}, 1, sizeof({sink}), out);" for sink, _, _ in fields]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        wrapper = path / "reference.c"
        executable = path / "reference"
        wrapper.write_text(
            f'#include <stdio.h>\n#define main corpus_main\n#include "{source}"\n'
            '#undef main\nint main(int argc, char **argv) {\n'
            '    if (argc != 2 || corpus_main()) return 1;\n'
            '    FILE *out = fopen(argv[1], "wb");\n    if (!out) return 1;\n    ' +
            '\n    '.join(outputs) + '\n    return fclose(out);\n}\n'
        )
        subprocess.run([compiler, "-O2", "-ffp-contract=off", "-fwrapv", wrapper, "-o", executable],
                       check=True, timeout=60, capture_output=True)
        output = path / "sink.bin"
        subprocess.run([executable, output], check=True, timeout=30, capture_output=True)
        return output.read_bytes()


def dispatch(runner, chip, program, *inputs):
    pipeline = json.loads((ROOT / "pipelines" / f"{program}.json").read_text())
    entry = pipeline["ComputePipelineState"]["ComputePipeline"]["stage"]["pName"]
    uuid = bytes(pipeline["PipelineUUID"]).hex()
    cache = ROOT / "cache" / chip / f"{program}.bin"
    result = subprocess.check_output([runner, cache, uuid, entry, *map(str, inputs)], timeout=30, text=True)
    return [int(word, 16) for word in result.split()]


def check(runner, compiler, chip, program):
    if program == "01_hello_world":
        words = dispatch(runner, chip, program)
        reference = b"Hello, World!\n"
        actual = bytes(words[1:15])
        return words[0] == 0 and actual == reference, actual.hex(), reference.hex()
    if program == "cpp_consteval_tuple_automaton":
        source = ROOT.parent / "programs" / f"{program}.cpp"
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "reference"
            subprocess.run(["c++", "-O2", "-std=c++20", source, "-o", executable], check=True, timeout=60)
            reference = tuple(map(int, subprocess.check_output([executable], text=True, timeout=30).split()[1:]))
        words = dispatch(runner, chip, program)
        actual = (int.from_bytes(bytes(words[1:9]), "little"), words[17])
        return words[0] == 0 and actual == reference, str(actual), str(reference)
    if program == "02_fibonacci":
        words = dispatch(runner, chip, program, 20)
        invalid = dispatch(runner, chip, program, 93)
        memo = int.from_bytes(bytes(words[1:9]), "little")
        naive = int.from_bytes(bytes(words[17:25]), "little")
        expected = 6765
        return (words[0] == 0 and words[33] == 1 and memo == naive == expected and invalid[0] == 1,
                f"{memo}/{naive}, invalid={invalid[0]}", f"{expected}/{expected}, invalid=1")

    fields = CASES[program]
    words = dispatch(runner, chip, program)
    reference = native_result(compiler, program, fields)
    seen, expected, advance = [], [], 0
    passed = words[0] == 0
    for _, index, fmt in fields:
        size = struct.calcsize("<" + fmt)
        cpu = struct.unpack_from("<" + fmt, reference, advance)
        gpu = struct.unpack("<" + fmt, bytes(words[index:index + size]))
        match = math.isclose(cpu[0], gpu[0], rel_tol=1e-12) if fmt == "d" else cpu == gpu
        passed &= match
        seen.extend(gpu)
        expected.extend(cpu)
        advance += size
    return passed, ", ".join(map(str, seen)), ", ".join(map(str, expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", type=Path)
    parser.add_argument("--chip", default="ad102")
    parser.add_argument("--cc", default="cc")
    args = parser.parse_args()
    cases = ["01_hello_world", "02_fibonacci", "cpp_consteval_tuple_automaton", *CASES]
    rows = []
    for program in cases:
        if not (ROOT / "cache" / args.chip / f"{program}.bin").is_file():
            continue
        try:
            passed, actual, reference = check(args.runner.resolve(), args.cc, args.chip, program)
            rows.append((program, args.chip, "pass" if passed else "mismatch", actual, reference))
        except (subprocess.SubprocessError, ValueError, OSError, struct.error) as error:
            rows.append((program, args.chip, "error", str(error), ""))

    with (ROOT / "smoke-test.tsv").open("w", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(("program", "chip", "status", "device", "reference"))
        writer.writerows(rows)
    print(f"{sum(row[2] == 'pass' for row in rows)}/{len(rows)} Vulkan SC smoke tests passed")
    return 0 if rows and all(row[2] == "pass" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
