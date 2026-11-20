"""Embed the source JSON demos without decoding the JSON on the build host."""

from pathlib import Path
import re
import sys


def generate(source):
    source = source.read_text()
    complex_demo = source.split("std::string complex_json = ", 1)[1].split(")JSON\";", 1)[0]
    malformed = source.split("error_tests = {", 1)[1].split("\n        };", 1)[0]
    unicode_demo = source.split('JsonValue unicode_test = json::parse(R"(', 1)[1].split(')");', 1)[0]
    inputs = {
        "COMPLEX": "".join(re.findall(r'R"JSON\((.*?)(?:\)JSON"|$)', complex_demo, re.S)),
        "UNICODE": unicode_demo,
        **{f"MALFORMED_{i}": value for i, value in enumerate(re.findall(r'R"\((.*?)\)"', malformed))},
    }
    if not inputs["COMPLEX"] or len(inputs) != 9:
        raise ValueError("could not read all native JSON fixtures")

    keys_and_values = {
        "NEW_FIELD": "newField", "ADDED_TEXT": "Added programmatically", "TIMESTAMP": "timestamp",
        "NUMBERS": "numbers", "ADDED": "added", "VERSION": "version", "ENABLED": "enabled",
        "ARRAYS": "arrays", "STRINGS": "strings", "DATE": "date", "ELDERBERRY": "elderberry",
        "NEW_ARRAY": "newArray", "NEW_OBJECT": "newObject", "KEY1": "key1", "VALUE1": "value1",
        "KEY2": "key2", "KEY3": "key3", "METADATA": "metadata", "APPLICATION": "application",
        "APP": "MyApp", "SETTINGS": "settings", "THEME": "theme", "DARK": "dark", "FONT_SIZE": "fontSize",
        "AUTO_SAVE": "autoSave", "FILES": "recentFiles", "FILE1": "/path/to/file1.txt",
        "FILE2": "/path/to/file2.txt", "USERS": "users", "NAME": "name", "ALICE": "Alice", "BOB": "Bob",
        "ROLE": "role", "ADMIN": "admin", "USER": "user", "HELLO": "hello", "X": "x",
        "TIMEOUT": "timeout", "RETRIES": "retries", "VERBOSE": "verbose", "DEBUG_KEY": "debug",
    }
    blob = bytearray()
    for name, value in {**inputs, **keys_and_values}.items():
        encoded = value.encode("utf-8")
        print(f"const ivec2 {name} = ivec2({len(blob)}, {len(encoded)});")
        blob.extend(encoded)
    print("const ivec2 malformed[] = ivec2[](" +
          ", ".join(f"MALFORMED_{i}" for i in range(7)) + ");")
    blob.extend(b"\0" * (-len(blob) % 4))
    words = [int.from_bytes(blob[i:i + 4], "little") for i in range(0, len(blob), 4)]
    print("const uint fixture[] = uint[](\n" + ",\n".join(
        "    " + ", ".join(f"0x{value:08x}u" for value in words[i:i + 8])
        for i in range(0, len(words), 8)) + "\n);")


if __name__ == "__main__":
    generate(Path(sys.argv[1]))
