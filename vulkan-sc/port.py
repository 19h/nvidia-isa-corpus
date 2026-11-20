#!/usr/bin/env python3
"""Translate the portable C subset of the corpus into single-invocation shaders."""

import json
import re
import subprocess
import tempfile

BUFFERED_STATE = {
    "fp_dsp_filter", "fp_fft", "fp_mandelbrot", "fp_neural_layer", "int_crc32",
    "int_galois", "int_sha256", "mem_sequential", "mem_strided", "min_load_chain",
    "min_store_chain", "simd_dot_product", "simd_image_filter", "simd_portable_bitplane",
}

PRELUDE = """\
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;
typedef long int64_t;
typedef ulong uint64_t;
#define INT32_C(x) x
#define INT64_C(x) x##L
#define UINT64_C(x) x##UL
#define UINT32_C(x) x##U
#define UINT8_MAX 255u
#define UINT16_MAX 65535u
#define UINT32_MAX 4294967295u
#define UINT64_MAX 18446744073709551615UL
#define INT32_MAX 2147483647
#define INT32_MIN (-2147483647 - 1)
#define INT64_MAX 9223372036854775807L
#define INT64_MIN (-9223372036854775807L - 1L)
#define NULL ((void *)0)
#define offsetof(type, member) __builtin_offsetof(type, member)
#define printf(...) ((void)0)
#define memcpy __builtin_memcpy
#define memset __builtin_memset
"""


def children(node):
    for child in node.get("inner", ()):
        yield child
        yield from children(child)


def position(location):
    return location.get("expansionLoc", location)


def offsets(node):
    begin, end = (position(node["range"][key]) for key in ("begin", "end"))
    return begin["offset"], end["offset"] + end["tokLen"]


def reference_offsets(node, data, token):
    start, end = offsets(node)
    if data[start:end] == token:
        return start, end
    begin = node["range"]["begin"].get("spellingLoc", {})
    end = node["range"]["end"].get("spellingLoc", {})
    if "offset" in begin and "offset" in end and not begin.get("includedFrom"):
        start, stop = begin["offset"], end["offset"] + end["tokLen"]
        if data[start:stop] == token:
            return start, stop
    raise ValueError(f"cannot locate source token {token.decode()}")


def declared_here(node):
    location = position(node.get("loc", {}))
    return "offset" in location and "includedFrom" not in location and not node.get("isImplicit")


def portable_preprocessing(name, data):
    if name.startswith("simd_"):
        data = re.sub(
            rb"(?m)^(#if )defined\(__x86_64__\) \|\| defined\(__i386__\)",
            rb"\g<1>0",
            data,
        )
    if name == "simd_portable_bitplane":
        data = data.replace(b"defined(__clang__) || defined(__GNUC__)", b"0")
        data = data.replace(b"    memcpy(work, planes, sizeof(work));",
                            b"""    for (int p = 0; p < PLANES; ++p)
        for (int w = 0; w < WORDS; ++w)
            work[p][w] = planes[p][w];""")
    if name in ("fp_physics_sim", "fp_fft"):
        data = re.sub(rb"\bhalf\b", b"half_count", data)
    if name == "call_nested":
        data = re.sub(rb"\blocal\b", b"local_value", data)
    if name == "fp_fft":
        data = data.replace(b"unsigned int result = 0;", b"volatile unsigned int result = 0;")
    if name == "int_crc32":
        declaration = b"static uint32_t crc32c_table[256];"
        data = data.replace(declaration, b"")
        data = data.replace(b"static uint32_t crc_table[256];", declaration + b"\nstatic uint32_t crc_table[256];")
    if name == "min_load_chain":
        data = data.replace(b"struct node *next;", b"int next;").replace(b".next = &nodes[next_idx]", b".next = next_idx")
        data = data.replace(b"struct node *ptr = &nodes[0];", b"int cursor = 0;")
        data = data.replace(b"ptr->value", b"nodes[cursor].value").replace(b"ptr = ptr->next", b"cursor = nodes[cursor].next")
    if name == "mem_stack_realign":
        data = data.replace(b"vla[len]", b"vla[64]")
        data = re.sub(rb"    uint8_t \*raw = [^\n]+\n    uintptr_t aligned_addr = [^\n]+\n    uint8_t \*aligned = [^\n]+",
                      b"    _Alignas(32) uint8_t aligned[64 * 3 + 17];", data)
    if name in ("call_many_int_args", "call_many_float_args"):
        data = re.sub(rb"    if \(depth <= 0\) \{\s*(return [^\n]+)\s*\}\s*return recursive_sum(?:_f)?\(depth - 1,[^;]+;",
                      lambda match: b"    for (; depth > 0; --depth) {\n        " +
                      b" ".join(letter + b" += " + (b"0.1" if name == "call_many_float_args" else b"1") + b";"
                                for letter in (b"a", b"b", b"c", b"d", b"e", b"f", b"g", b"h")) +
                      b"\n    }\n    " + match[1], data)
    if name == "mem_sequential":
        data = data.replace(b"        result += (int64_t)sequential_dot(array_d, array_e, ARRAY_SIZE);",
                            b"""        double dot = sequential_dot(array_d, array_e, ARRAY_SIZE);
        // Match the native x86 reference for overflowing float-to-int conversion.
        result += (dot >= 0x1p63 || dot < -0x1p63) ? INT64_MIN : (int64_t)dot;""")
    if name == "cf_irreducible_state_maze":
        begin = data.index(b"#define MAZE_FUEL")
        end = data.index(b"    return mix32(acc ^ latch", begin)
        text = data[begin:end]
        blocks = re.split(rb"(?m)^(\w+):\n", text)
        labels = blocks[1:-2:2]
        transition = rb"{ node = NODE_\1; continue; }"
        body = [b"    enum { " + b", ".join(b"NODE_" + label for label in labels) + b" } node;",
                b"    node = ((mix32(seed) ^ seed ^ cf_irreducible_fence) & 3u) == 0u ? NODE_Cross : NODE_North;",
                b"    while (++steps < max_steps) {",
                b"        switch (node) {"]
        for label, operations in zip(labels, blocks[2:-2:2]):
            operations = operations.replace(b"    MAZE_FUEL();\n", b"").rstrip()
            operations = re.sub(rb"goto (\w+);", transition, operations)
            body.append(b"            case NODE_" + label + b": {")
            body.extend(b"            " + line if line else b"" for line in operations.splitlines())
            body.append(b"            }")
        data = data[:begin] + b"\n".join(body + [b"        }", b"    }\n\n"]) + data[end:]
    if name == "call_nested":
        start = data.index(b"int64_t deep_with_locals(")
        end = data.index(b"\nint main(", start)
        data = data[:start] + b"""int64_t deep_with_locals(int depth, int val) {
    // The test calls this function at a depth of eight.
    int values[9] = {val};
    int depths[9] = {depth};
    int count = 1;
    int64_t total = 0;
    while (count) {
        int at = --count;
        int value = values[at];
        int remaining = depths[at];
        total += (value ^ 0xff) + (value & 0xf0);
        if (remaining <= 0) {
            total += (value + 1) + value * 2;
        } else {
            values[count] = value * 2;
            depths[count++] = remaining - 1;
            values[count] = value + 1;
            depths[count++] = remaining - 1;
        }
    }
    return total;
}

static int64_t mutual_walk(int depth, int value, bool side_b) {
    int64_t total = 0;
    for (; depth > 0; --depth) {
        total += value;
        value = side_b ? value * 2 : value + 1;
        side_b = !side_b;
    }
    return total + (side_b ? value * 2 : value);
}

int64_t mutual_a(int depth, int val) {
    return mutual_walk(depth, val, false);
}

int64_t mutual_b(int depth, int val) {
    return mutual_walk(depth, val, true);
}
""" + data[end:]
        data = b"#include <stdbool.h>\n" + data
    print_fields = {
        "int_crypto_mix_network": ("summary",),
        "mem_alias_lifetime_graph": ("live_total",),
        "mem_provenance_launder_lab": ("visit_total", "live_total"),
    }
    if name in print_fields:
        fields = print_fields[name]
        declarations = b"\nvolatile uint64_t sink_checksum;\n"
        assignments = b"    sink_checksum = checksum;\n"
        for field in fields:
            value = field.encode()
            declarations += b"volatile uint32_t sink_" + value + b";\n"
            assignments += b"    sink_" + value + b" = " + value + b";\n"
        last_header = list(re.finditer(rb"(?m)^#include [^\n]+\n", data))[-1].end()
        data = data[:last_header] + declarations + data[last_header:]
        data = data.replace(b'    printf("' + name.encode(), assignments + b'    printf("' + name.encode())
    if name == "cf_indirect_call":
        data = data.replace(b"void *self", b"Object *self").replace(b"(Object *)self", b"self")
        data = data.replace(b"void *context", b"int64_t *context").replace(b"(int64_t *)context", b"context")
        data = data.replace(b"    Object *objects[2] = {&obj_a, &obj_b};\n", b"")
        data = data.replace(b"objects[i % 2]", b"(i % 2 == 0 ? &obj_a : &obj_b)")
        # Vulkan shaders use finite selectors for the original indirect dispatch.
        for kind, operations in {
            "Binary": ("op_add", "op_sub", "op_mul", "op_div", "op_mod",
                       "op_and", "op_or", "op_xor", "op_min", "op_max"),
            "Unary": ("unary_neg", "unary_not", "unary_abs", "unary_square", "unary_double", "unary_half"),
        }.items():
            family = kind.lower().encode()
            alias = kind.encode() + b"Op"
            data = re.sub(rb"typedef int \(\*" + alias + rb"\)\([^;]+;", b"typedef int " + alias + b";", data)
            table = re.search(rb"static " + alias + rb" " + family + rb"_ops\[\] = \{[^}]+\};", data)
            aliases = {op: "SELECT_" + op.upper() for op in operations}
            data = data[:table.start()] + b"enum { " + ", ".join(aliases.values()).encode() + b" };" + data[table.end():]
            data = re.sub(rb"(#define NUM_" + family.upper() + rb"_OPS ).*", rb"\g<1>" + str(len(operations)).encode(), data)
            args = b"a, b" if kind == "Binary" else b"a"
            call = b"return " + family + b"_ops[op](" + args + b");"
            cases = [
                b"            case " + aliases[op].encode() + b": return " + op.encode() + b"(" + args + b");"
                for op in operations
            ]
            data = data.replace(call, b"switch (op) {\n" + b"\n".join(cases) + b"\n        }")
            for op, selector in aliases.items():
                data = re.sub(rb"\b" + op.encode() + rb"\b(?=\s*[,?:])", selector.encode(), data)

        data = re.sub(rb"typedef struct \{\n    int \(\*get_value\)[^}]+\} VTable;\n", b"", data)
        data = re.sub(rb"static const VTable vtable_[ab] = [^\n]+", b"", data)
        data = data.replace(b"const VTable *vtable;", b"int variant;")
        data = data.replace(b"{&vtable_a, 100}", b"{0, 100}").replace(b"{&vtable_b, 100}", b"{1, 100}")
        data = data.replace(b"obj->vtable->get_value(obj)", b"(obj->variant ? impl_b_get(obj) : impl_a_get(obj))")
        data = data.replace(b"obj->vtable->set_value(obj, i);",
                            b"if (obj->variant) impl_b_set(obj, i); else impl_a_set(obj, i);")
        data = data.replace(b"obj->vtable->process(obj, i % 100)",
                            b"(obj->variant ? impl_b_process(obj, i % 100) : impl_a_process(obj, i % 100))")
        data = data.replace(b"typedef void (*Callback)(int value, int64_t *context);", b"typedef int Callback;")
        data = data.replace(b"        cb(arr[i], context);",
                            b"        if (cb) callback_multiply(arr[i], context); else callback_accumulate(arr[i], context);")
        data = data.replace(b"? callback_accumulate : callback_multiply,", b"? 0 : 1,").replace(b", callback_multiply, &acc", b", 1, &acc")
        data = data.replace(b"return f(f(x));", b"return dispatch_unary(f, dispatch_unary(f, x));")
        data = data.replace(b"return f(g(x));", b"return dispatch_unary(f, dispatch_unary(g, x));")
        data = data.replace(b"acc = f(acc, arr[i]);", b"acc = dispatch_binary(f, acc, arr[i]);")
        data = data.replace(b"typedef void (*StateHandler)(StateMachine *sm, int input);",
                            b"typedef enum { IDLE, RUNNING, PAUSED } StateHandler;")
        for state_name in ("idle", "running", "paused"):
            old = b"state_" + state_name.encode()
            new = state_name.upper().encode()
            data = data.replace(b"current_state = " + old, b"current_state = " + new)
            data = data.replace(b"{" + old + b",", b"{" + new + b",")
        data = data.replace(b"    sm->current_state(sm, input);",
                            b"""    switch (sm->current_state) {
        case IDLE: state_idle(sm, input); break;
        case RUNNING: state_running(sm, input); break;
        case PAUSED: state_paused(sm, input); break;
    }""")
    if name == "cf_duff_device_parser":
        data = data.replace(b"    const unsigned char *base;\n    const unsigned char *cursor;\n    const unsigned char *end;",
                            b"    size_t cursor;\n    size_t length;")
        data = data.replace(b"static void parse_one(Parser *parser)", b"static void parse_one(Parser *parser, const unsigned char *text)")
        data = data.replace(b"(uint32_t)(parser->cursor - parser->base)", b"(uint32_t)parser->cursor")
        data = data.replace(b"*parser->cursor++", b"text[parser->cursor++]")
        data = data.replace(b"    parser.base = text;\n    parser.cursor = text;\n    parser.end = text + len;",
                            b"    parser.cursor = 0;\n    parser.length = len;")
        data = data.replace(b"parser.cursor == parser.end", b"parser.cursor == parser.length")
        data = data.replace(b"parse_one(&parser);", b"parse_one(&parser, text);")
        data = data.replace(b"static const unsigned char script", b"const unsigned char script")
        data = re.sub(rb"(?m)^typedef uint32_t \(\*ParserAction\).*;\n", b"", data)
        data = re.sub(rb"static ParserAction action_table\[4\] = \{[^}]+\};", b"", data)
        data = data.replace(b"ParserAction action = action_table[(cls ^ old_state ^ (ch >> 4u)) & 3u];",
                            b"uint32_t action = (cls ^ old_state ^ (ch >> 4u)) & 3u;")
        data = data.replace(b"    parser->hash ^= action(parser, ch, cls);",
                            b"""    switch (action) {
        case 0: parser->hash ^= action_lace(parser, ch, cls); break;
        case 1: parser->hash ^= action_stride(parser, ch, cls); break;
        case 2: parser->hash ^= action_gate(parser, ch, cls); break;
        case 3: parser->hash ^= action_echo(parser, ch, cls); break;
    }""")
    if name in ("cf_musttail", "call_struct_byval"):
        data = data.replace(b"<cstdint>", b"<stdint.h>").replace(b"<cstdio>", b"<stdio.h>")
        data = data.replace(b"std::", b"").replace(b"[[clang::musttail]] ", b"")
        data = data.replace(b"int main()", b"volatile int64_t sink;\n\nint main(void)")
        data = re.sub(rb'    printf\("%ld\\n", (\w+)\);', rb'    sink = \1;', data)
        data = data.replace(b"struct Big {", b"typedef struct {").replace(b"long a[8];\n};", b"long a[8];\n} Big;")
        data = data.replace(b"    if (n == 0)\n        return acc;\n    return countdown(n - 1, acc + n);",
                            b"    while (n != 0)\n        acc += n--;\n    return acc;")
        # The checksum variable precedes the helper functions as in the C workloads.
        data = data.replace(b"volatile int64_t sink;\n\nint main", b"int main")
        at = data.index(b"#include <stdio.h>") + len(b"#include <stdio.h>")
        data = data[:at] + b"\n\nvolatile int64_t sink;" + data[at:]
    return data


def port(source, clang):
    data = portable_preprocessing(source.stem, source.read_bytes())
    with tempfile.NamedTemporaryFile(suffix=".c") as temporary:
        temporary.write(data)
        temporary.flush()
        ast = subprocess.run(
            [clang, "-std=gnu11", "-Wno-everything", "-Xclang", "-ast-dump=json",
             "-fsyntax-only", temporary.name], capture_output=True, check=True,
        )

    nodes = [node for node in json.loads(ast.stdout)["inner"] if declared_here(node)]
    functions = [node for node in nodes if node["kind"] == "FunctionDecl"]
    variables = [node for node in nodes if node["kind"] == "VarDecl"]
    main = next((node for node in functions if node["name"] == "main"), None)
    if not main or main["type"]["qualType"] != "int (void)":
        raise ValueError("expected an int main(void) workload")

    edits = {}
    mutable = {}
    sink_types = {}
    initializers = []
    members = []
    last_global = 0
    for node in variables:
        name = node["name"]
        start, end = offsets(node)
        declaration = data[start:end]
        if name in mutable.values() or not re.match(rb"\s*(?:;|,)", data[end:]):
            raise ValueError(f"unsupported global declaration {name}")
        if re.match(rb"\s*,", data[end:]):
            raise ValueError(f"multiple declarations in global {name}")
        if re.match(rb"^(?:static\s+)?const\s+(?!.*\*)", declaration):
            edits[start, start] = b"__constant "
            continue
        mutable[node["id"]] = name
        sink_types[name] = node["type"].get("desugaredQualType", node["type"]["qualType"])
        last_global = max(last_global, data.index(b";", end) + 1)
        field, separator, expression = declaration.partition(b"=")
        field = re.sub(rb"\b(?:static|_Thread_local)\s+", b"", field).strip()
        if name.startswith("sink") and source.stem in BUFFERED_STATE:
            # The sink is read into the result buffer, so it remains observable.
            field = re.sub(rb"\bvolatile\s+", b"", field)
        members.append(field)
        edits[start, data.index(b";", end) + 1] = b""
        if separator:
            initializers.append(b"." + name.encode() + b" = " + expression.strip())

    if any(offsets(function)[0] < last_global for function in functions) and members:
        raise ValueError("mutable global declarations interleaved with functions")

    buffered = bool(members) and source.stem in BUFFERED_STATE
    address_space = b"__global" if buffered else b"__private"
    if members:
        state = b"typedef struct {\n    " + b";\n    ".join(members) + b";\n} ProgramState;\n\n"
        first_function = min(offsets(node)[0] for node in functions)
        edits[first_function, first_function] = state

    function_ids = {node["id"]: node["name"] for node in functions}
    spans = set()
    for node in functions:
        at = position(node["loc"])["offset"]
        name = node["name"].encode()
        if node is main:
            edits[at, at + len(name)] = b"program_main"
        if members:
            opening = data.index(b"(", at + len(name)) + 1
            prefix = re.match(rb"\s*(void\s*)?(?=\))", data[opening:])
            if prefix:
                edits[opening, opening + prefix.end()] = address_space + b" ProgramState *program_state"
            else:
                edits[opening, opening] = address_space + b" ProgramState *program_state, "
        for descendant in children(node):
            if descendant.get("kind") == "DeclRefExpr":
                referenced = descendant.get("referencedDecl", {}).get("id")
                if referenced in mutable:
                    name = mutable[referenced].encode()
                    begin, end = reference_offsets(descendant, data, name)
                    edits[begin, end] = b"program_state->" + name
            if descendant.get("kind") != "CallExpr" or not members:
                continue
            operands = descendant.get("inner", ())
            if not operands:
                continue
            callee = next((item for item in children(operands[0]) if item.get("kind") == "DeclRefExpr"), operands[0])
            reference = callee.get("referencedDecl", {}).get("id")
            if reference not in function_ids:
                continue
            _, end = reference_offsets(callee, data, function_ids[reference].encode())
            opening = data.index(b"(", end) + 1
            spans.add((opening, b"program_state, " if len(operands) > 1 else b"program_state"))
    for opening, state_arg in spans:
        edits[opening, opening] = state_arg

    for (start, end), replacement in sorted(edits.items(), reverse=True):
        data = data[:start] + replacement + data[end:]
    # Drop host headers after AST generation. Shader implementations are supplied above.
    data = re.sub(rb"(?m)^#\s*include\s+.*$", b"", data)
    data = re.sub(rb"\bregister\s+", b"", data)
    data = re.sub(rb"__attribute__\s*\(\(noinline\)\)", b"", data)

    if source.stem == "int_sha256":
        data = data.replace(b"uint32_t state[8],", b"__private uint32_t state[8],")
        data = data.replace(b"SHA256_CTX *", b"__private SHA256_CTX *")
    if source.stem == "fp_neural_layer":
        data = data.replace(b"const float w[OUTPUT_SIZE][INPUT_SIZE]", b"const __global float w[OUTPUT_SIZE][INPUT_SIZE]")
    if source.stem == "mem_strided":
        data = data.replace(b"const int32_t mat[MATRIX_DIM][MATRIX_DIM]", b"const __global int32_t mat[MATRIX_DIM][MATRIX_DIM]")
        data = data.replace(b"const int32_t src[MATRIX_DIM][MATRIX_DIM]", b"const __global int32_t src[MATRIX_DIM][MATRIX_DIM]")
    if source.stem == "simd_portable_bitplane":
        data = data.replace(b"(const uint32_t (*)[WORDS])", b"(const __global uint32_t (*)[WORDS])")
        for array in (b"dst", b"src", b"state"):
            name = array + b"[PLANES][WORDS]"
            data = data.replace(b"uint32_t " + name, b"__global uint32_t " + name)

    # Reserve word zero for the original main's status, followed by sink bytes.
    entry = [""]
    if buffered:
        entry.append("typedef struct { uint results[64]; ProgramState state; } Storage;")
        argument = "__global Storage *buffer"
    else:
        argument = "__global uint *results"
    entry += ["__attribute__((reqd_work_group_size(1, 1, 1)))",
              f"__kernel void program_entry({argument}) {{"]
    if buffered:
        entry.extend(["    __global uint *results = buffer->results;",
                      "    __global ProgramState *storage = &buffer->state;"])
        for initializer in initializers:
            entry.append(f"    storage->{initializer[1:].decode()};")
    elif members:
        values = b", ".join(initializers).decode() if initializers else "0"
        entry.append(f"    ProgramState storage = {{{values}}};")
    arguments = "storage" if buffered else "&storage" if members else ""
    entry.append(f"    results[0] = (uint)program_main({arguments});")
    index = 1
    for name in mutable.values():
        if not (name.startswith("sink") or name.endswith("_sink")):
            continue
        typ = sink_types[name].replace("volatile ", "")
        value = f"storage->{name}" if buffered else f"storage.{name}"
        if typ == "double":
            encoded = f"as_ulong({value})"
        elif typ == "float":
            encoded = f"as_uint({value})"
        elif typ in ("long", "unsigned long", "long long", "unsigned long long", "int64_t", "uint64_t"):
            encoded = f"(ulong){value}"
        else:
            encoded = f"(uint){value}"
        entry.append(f"    for (uint i = 0; i < sizeof({value}); ++i)")
        entry.append(f"        results[{index} + i] = (uint)(({encoded} >> (8u * i)) & 255u);")
        index += 16
    entry.append("}")
    return PRELUDE + data.decode() + "\n".join(entry) + "\n"


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--clang", default="clang")
    args = parser.parse_args()
    output = Path(__file__).resolve().parent / "opencl"
    output.mkdir(exist_ok=True)
    for source in args.sources:
        (output / (source.stem + ".cl")).write_text(port(source, args.clang))


if __name__ == "__main__":
    main()
