/* Assemble the original VM programs and emit their bytecode for the shader. */
#include <stdio.h>

#define main corpus_main
#define printf(...) fprintf(stderr, __VA_ARGS__)
#include "../../programs/09_bytecode_vm.c"
#undef printf
#undef main

int main(void) {
    const char *sources[] = {
        arithmetic_source, factorial_source, fibonacci_source, nested_calls_source
    };
    enum { DEMOS = sizeof(sources) / sizeof(*sources) };
    uint8_t bytes[DEMOS * CODE_SIZE + 3] = {0};
    uint32_t total = 0;

    for (int opcode = 0; opcode < OP_COUNT; ++opcode)
        printf("const uint OP_%s = %du;\n", opcode_info[opcode].name, opcode);
    printf("const int STACK_SIZE = %d, CALL_STACK_SIZE = %d, LOCALS_SIZE = %d, GLOBALS_SIZE = %d;\n",
           STACK_SIZE, CALL_STACK_SIZE, LOCALS_SIZE, GLOBALS_SIZE);
    puts("const uvec2 demos[] = uvec2[](");
    for (int demo = 0; demo < DEMOS; ++demo) {
        uint32_t size;
        if (!assemble(sources[demo], bytes + total, &size))
            return 1;
        printf("    uvec2(%u, %u)%s\n", total, size, demo + 1 < DEMOS ? "," : "");
        total += size;
    }
    puts(");");
    puts("const uint bytecode[] = uint[](");
    for (uint32_t i = 0; i < total; i += 4) {
        uint32_t word = (uint32_t)bytes[i] | ((uint32_t)bytes[i + 1] << 8)
                      | ((uint32_t)bytes[i + 2] << 16) | ((uint32_t)bytes[i + 3] << 24);
        printf("    0x%08xu%s\n", word, i + 4 < total ? "," : "");
    }
    puts(");");
    return 0;
}
