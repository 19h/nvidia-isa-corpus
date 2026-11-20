#!/usr/bin/env python3
"""Run representative offline Vulkan SC caches and compare against native C."""

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import re
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent

CASES = {
    "cf_exception": (("sink", 1, "q"),),
    "cpp_lambda": (("sink", 1, "q"),),
    "cpp_virtual": (("sink", 1, "q"),),
    "asm_atomic_ops": (("sink", 1, "q"),),
    "asm_barriers": (("sink", 1, "q"),),
    "call_abi_stress_matrix": (("call_abi_sink", 1, "Q"),),
    "call_many_float_args": (("sink_d", 1, "d"), ("sink_f", 17, "f")),
    "call_mixed_aggregates": (("sink", 1, "Q"),),
    "call_mixed_args": (("sink", 1, "d"),),
    "call_variadic": (("sink", 1, "q"),),
    "cf_computed_goto": (("sink", 1, "q"),),
    "cf_longjmp_threaded_vm": (("cf_longjmp_threaded_vm_sink", 1, "Q"),),
    "cf_setjmp": (("sink", 1, "q"),),
    "int_atomic_orders": (("sink", 1, "Q"),),
    "mem_random": (("sink", 1, "q"),),
    "call_nested": (("sink", 1, "q"),),
    "call_struct_return": (("sink", 1, "q"),),
    "cf_indirect_call": (("sink", 1, "q"),),
    "cf_duff_device_parser": (("cf_duff_device_parser_sink", 1, "Q"),),
    "cf_irreducible_state_maze": (("cf_irreducible_sink", 1, "Q"),),
    "fp_piecewise_chaos": (("sink", 1, "Q"),),
    "fp_dsp_filter": (("sink", 1, "f"),),
    "fp_neural_layer": (("sink", 1, "f"),),
    "fp_physics_sim": (("sink", 1, "d"),),
    "fp_mandelbrot": (("sink", 1, "I"),),
    "int_crc32": (("sink", 1, "I"),),
    "int_carryless_crc_mesh": (("sink", 1, "Q"),),
    "int_divrem_wide_lattice": (("sink", 1, "Q"),),
    "mem_sequential": (("sink", 1, "q"),),
    "mem_strided": (("sink", 1, "q"),),
    "mem_stack_realign": (("sink", 1, "Q"),),
    "mem_unaligned_packed": (("sink", 1, "Q"),),
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
    "min_div_chain": (("sink", 1, "i"),),
    "min_float_add": (("sink_d", 1, "d"), ("sink_f", 17, "f")),
    "min_float_mul": (("sink_d", 1, "d"), ("sink_f", 17, "f")),
    "min_load_chain": (("sink", 1, "i"),),
    "min_store_chain": (("sink", 1, "i"),),
    "min_mul_chain": (("sink", 1, "i"),),
    "min_shift_chain": (("sink", 1, "I"),),
    "simd_aes": (("sink", 1, "B"),),
    "simd_dot_product": (("sink", 1, "f"),),
    "simd_matrix_mul": (("sink", 1, "f"),),
    "simd_portable_bitplane": (("sink", 1, "Q"),),
    "simd_string_search": (("sink", 1, "Q"),),
}


PRINTED_CASES = {
    "call_struct_byval": (r"^(\d+)", ((1, 8, 10),)),
    "cf_musttail": (r"^(\d+)", ((1, 8, 10),)),
    "cpp_type_erased_callmaze": (r"cpp_type_erased_callmaze (\d+)", ((1, 8, 10),)),
    "cpp_allocator_polymorph_zoo": (r"cpp_allocator_polymorph_zoo (\d+) (\d+) (\d+)",
                                    ((1, 8, 10), (17, 1, 10), (33, 1, 10))),
    "cpp_constexpr_variant_graph": (r"cpp_constexpr_variant_graph (\d+)", ((1, 8, 10),)),
    "cpp_exception_variant_unwind": (r"cpp_exception_variant_unwind (\d+)", ((1, 8, 10),)),
    "cpp_coroutine_lifetime": (r"cpp_coroutine_lifetime (\d+) (\d+)", ((1, 8, 10), (17, 1, 10))),
    "fp_denormal_rounding_lab": (r"checksum=([0-9a-f]+) branches=(\d+)", ((1, 8, 16), (17, 8, 10))),
    "fp_fixedpoint_hybrid_solver": (r"checksum=([0-9a-f]+) branches=(\d+)", ((1, 8, 16), (17, 8, 10))),
    "int_crypto_mix_network": (r": ([0-9a-f]+) ([0-9a-f]+)", ((17, 4, 16), (1, 8, 16))),
    "mem_provenance_launder_lab": (r": ([0-9a-f]+) ([0-9a-f]+) ([0-9a-f]+)",
                                   ((17, 4, 16), (33, 4, 16), (1, 8, 16))),
    "mem_alias_lifetime_graph": (r": ([0-9a-f]+) ([0-9a-f]+)", ((17, 4, 16), (1, 8, 16))),
    "mem_flexible_arena_rewriter": (r": ([0-9a-f]+) ([0-9a-f]+) ([0-9a-f]+)",
                                    ((17, 4, 16), (33, 4, 16), (1, 8, 16))),
}


def native_stdout(compiler, program, probes=()):
    source = ROOT.parent / "programs" / f"{program}.c"
    command = [compiler]
    if not source.exists():
        source = source.with_suffix(".cpp")
        command = ["c++", "-std=c++20"]
    if program == "05_thread_pool":
        command.append("-pthread")
    if program == "10_template_madness":
        command.extend(("-include", "algorithm"))
    with tempfile.TemporaryDirectory() as tmp:
        executable = Path(tmp) / "reference"
        if probes:
            code = source.read_text()
            for anchor, label, expression in probes:
                if code.count(anchor) != 1:
                    raise ValueError(f"could not insert native probe {label}")
                code = code.replace(anchor, f'{anchor}\n        std::cout << "PROBE {label} "'
                                            f' << ({expression}) << "\\n";', 1)
            source = Path(tmp) / source.name
            source.write_text(code)
        subprocess.run([*command, "-O2", "-ffp-contract=off", "-fwrapv", source, "-o", executable],
                       check=True, timeout=60, capture_output=True)
        return subprocess.check_output([executable], text=True, timeout=30)


def native_result(compiler, program, fields):
    source = ROOT.parent / "programs" / f"{program}.c"
    is_cpp = not source.exists()
    if is_cpp:
        source = source.with_suffix(".cpp")
    outputs = [f"fwrite((const void *)&{sink}, 1, sizeof({sink}), out);" for sink, _, _ in fields]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        wrapper = path / ("reference.cpp" if is_cpp else "reference.c")
        executable = path / "reference"
        wrapper.write_text(
            f'#include <stdio.h>\n#define main corpus_main\n#include "{source}"\n'
            '#undef main\nint main(int argc, char **argv) {\n'
            '    if (argc != 2 || corpus_main()) return 1;\n'
            '    FILE *out = fopen(argv[1], "wb");\n    if (!out) return 1;\n    ' +
            '\n    '.join(outputs) + '\n    return fclose(out);\n}\n'
        )
        command = ["c++", "-std=c++20"] if is_cpp else [compiler]
        subprocess.run([*command, "-O2", "-ffp-contract=off", "-fwrapv", wrapper, "-o", executable],
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


def check_bst(runner, compiler, chip):
    printed = native_stdout(compiler, "04_bst")
    fruit = "apple banana cherry grape kiwi mango orange pineapple strawberry watermelon zebra".split()
    traversals = re.findall(r"(?:Inorder|Preorder|Postorder|Level-order|Sorted vector|"
                           r"Even numbers \(preorder\)|Transformed \(level-order\)): ([^\n]+)", printed)
    if not traversals or "All assertions PASSED!" not in printed:
        raise ValueError("could not read native BST traversal and assertion results")
    digest = 14695981039346656037
    for traversal in traversals:
        values = [fruit.index(key) if key in fruit else int(key.removeprefix("node_"))
                  for key in traversal.replace(" -> ", " ").split()]
        for value in [*values, 0xffffffff]:
            digest = ((digest ^ value) * 1099511628211) & ((1 << 64) - 1)

    fields = re.search(
        r"(?ms)^Size: (\d+).*?^Height: (\d+).*?^Is balanced: (yes|no).*?"
        r"^Min value: (\d+).*?^Max value: (\d+).*?^Contains 40: (yes|no).*?"
        r"^Contains 45: (yes|no).*?^Find 60: (60|not found).*?^Final size: (\d+).*?"
        r"^Original after move .*?size = (\d+).*?^Empty tree - size: \d+, height: (-?\d+).*?"
        r"^Empty tree - isBalanced: (yes|no).*?^Single node - size: \d+, height: (\d+).*?"
        r"^Single node - isBalanced: (yes|no).*?^Height: (\d+).*?^Is balanced: (yes|no).*?"
        r"^Size: (\d+).*?^Min: (\w+).*?^Max: (\w+).*?^Contains 'banana': (yes|no).*?"
        r"^Contains 'cherry': (yes|no).*?^Sum of all elements: (\d+)", printed)
    removals = re.findall(r"(?m)^Removing .*?: (success|failed)", printed)
    if fields is None or len(removals) != 4:
        raise ValueError("could not read native BST properties")
    (size, height, balanced, minimum, maximum, find40, find45, find60, final_size, moved_size,
     empty_height, empty_balanced, single_height, single_balanced, skew_height, skew_balanced,
     distinct, first_fruit, last_fruit, find_banana, find_cherry, callback_sum) = fields.groups()
    found = (find40 == "yes") | ((find45 == "yes") << 1) | ((find60 == "60") << 2)
    removed = sum((result == "success") << i for i, result in enumerate(removals))
    fruit_found = (find_banana == "yes") | ((find_cherry == "yes") << 1)
    reference = (digest, len(traversals), int(size), int(height), int(balanced == "yes"),
                 int(minimum), int(maximum), found, removed, int(final_size), int(moved_size),
                 int(empty_height) & 0xffffffff, int(empty_balanced == "yes"), int(single_height),
                 int(single_balanced == "yes"), int(skew_height), int(skew_balanced == "yes"),
                 int(distinct), fruit.index(first_fruit), fruit.index(last_fruit), fruit_found,
                 int(callback_sum))
    words = dispatch(runner, chip, "04_bst")
    actual = (words[1] | (words[2] << 32), *words[3:24])
    return words[0] == 0 and actual == reference, str(actual), str(reference)


def check_thread_pool(runner, compiler, chip):
    printed = native_stdout(compiler, "05_thread_pool")
    fields = re.search(
        r"(?s)Parallel result:\s+(\d+).*?Sequential result:\s+(\d+).*?"
        r"Lambda \(10 \+ 20\): (\d+).*?Function pointer .*?: (\d+).*?"
        r"Functor .*?: (\d+).*?std::function .*?: (\d+).*?Capturing lambda .*?: (\d+).*?"
        r"Successful task .*?: (\d+).*?Total result: (\d+).*?Stats: (\d+)/\d+ tasks completed.*?"
        r"Counter after .*?: (\d+).*?Submitted (\d+) tasks \(each takes", printed)
    if fields is None or "Graceful shutdown test completed" not in printed:
        raise ValueError("could not read native thread-pool results")
    parallel, sequential, *values = map(int, fields.groups())
    if parallel != sequential:
        raise ValueError("native parallel and sequential results disagree")
    exception = int("Exception caught as expected: Negative value" in printed)
    stopped = 2 * int("Exception caught as expected:" in printed.split("Test 7:", 1)[1])
    reference = (parallel, *values[:6], exception, *values[6:], stopped)
    words = dispatch(runner, chip, "05_thread_pool")
    actual = (words[1] | (words[2] << 32), *words[3:15])
    return words[0] == 0 and actual == reference, str(actual), str(reference)


def check_templates(runner, compiler, chip):
    printed = native_stdout(compiler, "10_template_madness")

    def capture(pattern):
        match = re.search(pattern, printed, re.M)
        if match is None:
            raise ValueError(f"could not read native template result for {pattern}")
        return match.groups()

    def digest(values, seed=2166136261):
        for value in values:
            seed = ((seed ^ value) * 16777619) & 0xffffffff
        return seed

    constants = [int(value) for value in re.findall(r"(?m)^\w+\([^)]+\) = (\d+)", printed)[:5]]
    sequences = [constants]
    for label in ("Primes up to 50", "Unsorted", "Sorted (at compile time!)",
                  "Sorted (at compile time!)", "First 10 factorials", "First 20 Fibonacci numbers"):
        raw, = capture(rf"^{re.escape(label)}: ([^\n]+)")
        sequences.append(list(map(int, re.findall(r"\d+", raw))))
    constant_hash = digest(value for sequence in sequences for value in [*sequence, 0xffffffff])
    expression, = capture(r"^2\*v1 \+ v2\*3 - v1 = \[([^]]+)\]")
    dot_product, = capture(r"^Dot product v1.v2 = ([\d.]+)")
    tuple_int, tuple_double, tuple_text = capture(r'^Tuple.*? = \((\d+), ([\d.]+), "(.*)"\)')
    shapes = re.findall(r"Shape with area: ([\d.]+), perimeter: ([\d.]+)", printed)
    if len(shapes) != 2:
        raise ValueError("could not read the two native shape results")
    people, = capture(r"^Person instances created: (\d+)")
    compare, = capture(r"^Alice .*: (true|false)")
    total, = capture(r"^sum\([^)]+\) = (\d+)")
    product, = capture(r"^product\([^)]+\) = (\d+)")
    any_int, = capture(r"^Any holds int: (\d+)")
    any_text, = capture(r"^Any holds string: (.*)")
    any_double, = capture(r"^Any holds double: ([\d.]+)")
    query, = capture(r"^Generated query: (.*)")

    words = dispatch(runner, chip, "10_template_madness")

    def real(word):
        return struct.unpack("<d", struct.pack("<2I", *words[word:word + 2]))[0]

    actual = (words[1], *(real(i) for i in range(2, 12, 2)), words[12], real(13), words[15],
              *(float(format(real(i), ".6g")) for i in range(16, 24, 2)), *words[24:30], real(30), words[32])
    reference = (constant_hash, *map(float, expression.split(", ")), float(dot_product), int(tuple_int),
                 float(tuple_double), digest(tuple_text.encode()), *(float(v) for shape in shapes for v in shape),
                 int(people), int(compare == "true"), int(total), int(product), int(any_int),
                 digest(any_text.encode()), float(any_double), digest(query.encode()))
    return words[0] == 0 and actual == reference, str(actual), str(reference)


def check_json(runner, compiler, chip):
    probes = (
        ("JsonValue doc = json::parse(complex_json);", "original", "json::compact(doc)"),
        ("std::string compact = json::compact(doc);", "modified", "compact"),
        ("std::cout << json::stringify(config) << '\\n';", "config", "json::compact(config)"),
        ('JsonValue mixed = json::array({42, "hello", true, nullptr, json::object({{"x", 1}})});',
         "mixed", "json::compact(mixed)"),
        ('std::cout << "  Chinese: " << unicode_test["chinese"].get_string() << \'\\n\';',
         "unicode", "json::compact(unicode_test)"),
    )
    printed = native_stdout(compiler, "08_json_parser", probes)
    documents = {label: json.loads(raw) for label, raw in re.findall(r"(?m)^PROBE (\w+) (.+)", printed)}
    if len(documents) != len(probes):
        raise ValueError("could not read the native JSON trees")

    def summarize(value):
        digest = 2166136261
        nodes = 0

        def add(word):
            nonlocal digest
            digest = ((digest ^ word) * 16777619) & 0xffffffff

        def text(value):
            encoded = value.encode("utf-8")
            add(len(encoded))
            for byte in encoded:
                add(byte)

        def visit(value):
            nonlocal nodes
            nodes += 1
            if type(value) is list:
                add(1)
                for child in value:
                    visit(child)
                add(0xffffffff)
            elif type(value) is dict:
                add(2)
                for key in sorted(value):
                    text(key)
                    visit(value[key])
                add(0xffffffff)
            elif type(value) in (int, float):
                add(3)
                add(struct.unpack("<I", struct.pack("<f", value))[0])
            elif type(value) is str:
                add(4)
                text(value)
            else:
                add(7 if value is None else 5 if value else 6)

        visit(value)
        return digest, nodes

    compact, pretty = map(int, re.findall(r"(?m)^  (?:Total|Pretty) length: (\d+) characters", printed))
    errors = printed.split("7. Error handling demonstrations:", 1)[1].split("8. Type checking", 1)[0]
    labels = ("Missing value", "Missing colon", "Trailing comma", "Unterminated string",
              "Invalid escape", "Invalid number", "Invalid literal")
    rejected = sum((f"  {label}:\n" in errors) << i for i, label in enumerate(labels))
    defaults = [int(value) if value.isdigit() else int(value == "true")
                for value in re.findall(r"(?m)^  (?:timeout|retries|verbose|debug): (\S+)", printed)]
    modified, nodes = summarize(documents["modified"])
    reference = (summarize(documents["original"])[0], modified, nodes, pretty - compact,
                 summarize(documents["config"])[0], summarize(documents["mixed"])[0],
                 rejected, summarize(documents["unicode"])[0], *defaults)
    words = dispatch(runner, chip, "08_json_parser")
    actual = (*words[1:4], words[5] - words[4], *words[6:14])
    return words[0] == 0 and actual == reference, str(actual), str(reference)


def check_validators(runner, compiler, chip, program):
    source = ROOT.parent / "programs" / f"{program}.c"
    if program == "cf_license_crackme":
        candidates = ("", "not a key", "2" * 24, "z" * 24, "2" * 23, "I" * 24)
        verdict = "k2(argv[i])"
    else:
        candidates = ("", "not a sigil", "G" * 24, "Z" * 24, "h" * 24, "G" * 23)
        verdict = 'verify("GPU Corpus", sizeof("GPU Corpus") - 1, argv[i])'

    with tempfile.TemporaryDirectory() as tmp:
        wrapper = Path(tmp) / "reference.c"
        executable = Path(tmp) / "reference"
        wrapper.write_text(
            f'#define main corpus_main\n#include "{source}"\n#undef main\n'
            'int main(int argc, char **argv) {\n'
            f'    for (int i = 1; i < argc; ++i) printf("%u\\n", {verdict});\n'
            '    return 0;\n}\n'
        )
        subprocess.run([compiler, "-O2", wrapper, "-o", executable], check=True,
                       timeout=60, capture_output=True)
        reference = tuple(map(int, subprocess.check_output(
            [executable, *candidates], timeout=30, text=True).split()))

    actual = []
    passed = True
    for candidate in candidates:
        inputs = struct.unpack("<6I", candidate.encode("ascii").ljust(24, b"\0"))
        words = dispatch(runner, chip, program, *inputs)
        actual.append(words[1])
        passed &= words[0] == 0
    actual = tuple(actual)
    return passed and actual == reference, str(actual), str(reference)


def check(runner, compiler, chip, program):
    if program == "01_hello_world":
        words = dispatch(runner, chip, program)
        reference = b"Hello, World!\n"
        actual = bytes(words[1:15])
        return words[0] == 0 and actual == reference, actual.hex(), reference.hex()
    if program == "03_linked_list":
        printed = native_stdout(compiler, program)
        snapshots = [line for line in printed.splitlines()
                     if line.startswith("List:") or line == "List is empty (NULL)"]
        expected_hash = 14695981039346656037
        for line in snapshots:
            values = [int(value) for value in re.findall(r"\[(-?\d+)\]", line)]
            for value in [*values, 0xffffffff]:
                expected_hash = ((expected_hash ^ (value & 0xffffffff)) * 1099511628211) & ((1 << 64) - 1)
        expected_frees = sum(map(int, re.findall(r"Freed (\d+) node", printed))) + 5
        words = dispatch(runner, chip, program)
        actual = (int.from_bytes(bytes(words[1:9]), "little"), *words[17:22])
        reference = (expected_hash, len(snapshots), 3, 7, 0, expected_frees)
        return words[0] == 0 and actual == reference, str(actual), str(reference)
    if program == "04_bst":
        return check_bst(runner, compiler, chip)
    if program == "05_thread_pool":
        return check_thread_pool(runner, compiler, chip)
    if program == "10_template_madness":
        return check_templates(runner, compiler, chip)
    if program in {"cf_license_crackme", "cf_rorog"}:
        return check_validators(runner, compiler, chip, program)
    if program == "simd_image_filter":
        arch = platform.machine().lower()
        mode = 1 if arch in {"x86_64", "amd64"} else 2 if arch in {"aarch64", "arm64"} else 0
        reference, = native_result(compiler, program, (("sink", 1, "B"),))
        words = dispatch(runner, chip, program, mode)
        return words[0] == 0 and words[1] == reference, str(words[1]), str(reference)
    if program == "06_allocator":
        printed = native_stdout(compiler, program)
        stats, heap = printed.split("Final Statistics:", 1)[1].split("=== Final Heap State ===")
        required = ("Total allocations performed", "Total frees performed", "Peak memory usage",
                    "Coalescing operations", "Block splits")
        reference = tuple(int(re.search(rf"{label}:\s+(\d+)", stats)[1]) for label in required)
        reference += tuple(int(re.search(rf"{label}:\s+(\d+)", heap)[1])
                           for label in ("Memory used", "Total blocks"))
        allocations = re.findall(r"Result: .* \((success!?|failed[^)]*)\)", printed)
        if len(allocations) != 2 or "YES - all zeros!" not in printed:
            raise ValueError("could not read allocator stress results")
        reference += (sum(1 << i for i, value in enumerate(allocations) if value.startswith("success")),)
        words = dispatch(runner, chip, program)
        actual = tuple(words[1:9])
        return words[0] == 0 and actual == reference, str(actual), str(reference)
    if program == "07_regex":
        printed = native_stdout(compiler, program)
        results = re.findall(r"(?m)^(PASS|FAIL):", printed)
        if not results:
            raise ValueError("could not read native regex results")
        masks = [sum((result == "PASS") << i for i, result in enumerate(results[offset:offset + 32]))
                 for offset in range(0, len(results), 32)]
        reference = (len(results), results.count("PASS"), *masks)
        words = dispatch(runner, chip, program)
        actual = tuple(words[1:1 + len(reference)])
        return words[0] == 0 and actual == reference, str(actual), str(reference)
    if program == "08_json_parser":
        return check_json(runner, compiler, chip)
    if program == "09_bytecode_vm":
        printed = native_stdout(compiler, program)
        runs = re.findall(r"=== Execution ===\n(.*?)\n=== Statistics ===\n"
                          r"Instructions executed: (\d+)\nFinal stack pointer: (\d+)", printed, re.S)
        if len(runs) != 4:
            raise ValueError("could not read four native VM demos")
        actual, reference = [], []
        passed = True
        for i, (stdout, instructions, stack_pointer) in enumerate(runs):
            output = stdout.encode("ascii")
            digest = 14695981039346656037
            for byte in output:
                digest = ((digest ^ byte) * 1099511628211) & ((1 << 64) - 1)
            reference.append((int(instructions), int(stack_pointer), len(output), digest))
            words = dispatch(runner, chip, program, i)
            actual.append((*words[1:4], words[4] | (words[5] << 32)))
            passed &= words[0] == 0
        invalid = dispatch(runner, chip, program, len(runs))[0]
        return passed and actual == reference and invalid == 9, f"{actual}, invalid={invalid}", f"{reference}, invalid=9"
    if program in PRINTED_CASES:
        pattern, fields = PRINTED_CASES[program]
        match = re.search(pattern, native_stdout(compiler, program))
        if match is None:
            raise ValueError(f"could not read native result of {program}")
        words = dispatch(runner, chip, program)
        actual = tuple(int.from_bytes(bytes(words[index:index+size]), "little")
                       for index, size, _ in fields)
        reference = tuple(int(value, radix) for value, (_, _, radix) in zip(match.groups(), fields))
        return words[0] == 0 and actual == reference, str(actual), str(reference)
    if program == "cpp_rtti":
        source = ROOT.parent / "programs/cpp_rtti.cpp"
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = Path(tmp) / "rtti.cpp"
            executable = Path(tmp) / "reference"
            types = ("Base", "DerivedA", "DerivedB", "DerivedAA", "DerivedAB", "MultiDerived")
            values = ", ".join("unsigned(typeid(" + name + ").hash_code() & 255)" for name in types)
            wrapper.write_text(
                f'#include <cstdio>\n#define main corpus_main\n#include "{source}"\n#undef main\n'
                'int main() { if (corpus_main()) return 1; '
                f'std::printf("%lld %u %u %u %u %u %u\\n", (long long)sink, {values}); }}\n'
            )
            subprocess.run(["c++", "-std=c++20", "-O2", wrapper, "-o", executable], check=True,
                           timeout=60, capture_output=True)
            reference, *tags = map(int, subprocess.check_output([executable], text=True, timeout=30).split())
        words = dispatch(runner, chip, program, *tags)
        actual = int.from_bytes(bytes(words[1:9]), "little", signed=True)
        return words[0] == 0 and actual == reference, str(actual), str(reference)
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
    cases = ["01_hello_world", "02_fibonacci", "03_linked_list", "04_bst", "05_thread_pool", "06_allocator", "07_regex",
             "08_json_parser", "09_bytecode_vm", "10_template_madness", "cpp_consteval_tuple_automaton", "cpp_rtti",
             "cf_license_crackme", "cf_rorog", "simd_image_filter", *PRINTED_CASES, *CASES]
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
