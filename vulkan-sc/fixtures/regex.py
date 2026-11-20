"""Generate GLSL fixture data from the original C regex test calls."""

import ast
from pathlib import Path
import re
import sys


def generate(source):
    main = source.read_text().split("int main(void) {", 1)[1]
    string = r'"(?:[^"\\]|\\.)*"'
    call = re.compile(
        rf'(?m)^\s*test_(match|full_match|search_pos|compile_error)\('
        rf'({string})(?:, ({string}), (-?\d+)(?:, (\d+))?)?, {string}\);$'
    )
    tests = call.findall(main)
    if not tests or len(tests) != len(re.findall(r"(?m)^\s*test_\w+\(", main)):
        raise ValueError("could not read all native regex test calls")
    if len(tests) > 61 * 32:
        raise ValueError("regex results exceed the shader's output bitset")

    strings = {}
    blob = bytearray()

    def store(literal):
        value = ast.literal_eval(literal).encode("ascii") if literal else b""
        if value not in strings:
            strings[value] = len(blob)
            blob.extend(value + b"\0")
        return strings[value]

    modes = {"match": 0, "full_match": 1, "search_pos": 2, "compile_error": 3}
    rows = [f"    ivec4({store(pattern)}, {store(subject)}, {modes[mode]}, {expected or 0})"
            for mode, pattern, subject, expected, _ in tests]
    print("const ivec4 cases[] = ivec4[](\n" + ",\n".join(rows) + "\n);")
    print("const int lengths[] = int[](" + ", ".join(length or "0" for *_, length in tests) + ");")
    # Pack four fixture bytes into each constant word.
    blob.extend(b"\0" * (-len(blob) % 4))
    words = [int.from_bytes(blob[i:i + 4], "little") for i in range(0, len(blob), 4)]
    print("const uint fixture[] = uint[](\n" + ",\n".join(
        "    " + ", ".join(f"0x{value:08x}u" for value in words[i:i + 8])
        for i in range(0, len(words), 8)) + "\n);")


if __name__ == "__main__":
    generate(Path(sys.argv[1]))
