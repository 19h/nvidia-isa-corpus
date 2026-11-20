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
/*
 * Minimal Memory Store Chain
 *
 * Generates sequences dominated by memory store instructions.
 * Sequential stores to memory with various sizes and patterns.
 *
 * Expected dominant instructions by architecture:
 *   x86:    mov (to memory)
 *   ARM:    str, strb, strh
 *   RISC-V: sw, sd, sb, sh
 *   MIPS:   sw, sd, sb, sh
 *   PPC:    stw, std, stb, sth
 */

#define ARRAY_SIZE 4096







typedef struct {
    int sink;
    int data[ARRAY_SIZE];
    char bytes[ARRAY_SIZE];
    short shorts[ARRAY_SIZE];
} ProgramState;

int program_main(__global ProgramState *program_state) {
    int val = 42;

    for (int iter = 0; iter < 1000; iter++) {
        /* Sequential word stores */
        for (int i = 0; i < ARRAY_SIZE; i += 8) {
            program_state->data[i]     = val;
            program_state->data[i + 1] = val + 1;
            program_state->data[i + 2] = val + 2;
            program_state->data[i + 3] = val + 3;
            program_state->data[i + 4] = val + 4;
            program_state->data[i + 5] = val + 5;
            program_state->data[i + 6] = val + 6;
            program_state->data[i + 7] = val + 7;
            val += 8;
        }

        /* Byte stores */
        for (int i = 0; i < ARRAY_SIZE; i += 8) {
            program_state->bytes[i]     = (char)(val & 0xFF);
            program_state->bytes[i + 1] = (char)((val + 1) & 0xFF);
            program_state->bytes[i + 2] = (char)((val + 2) & 0xFF);
            program_state->bytes[i + 3] = (char)((val + 3) & 0xFF);
            program_state->bytes[i + 4] = (char)((val + 4) & 0xFF);
            program_state->bytes[i + 5] = (char)((val + 5) & 0xFF);
            program_state->bytes[i + 6] = (char)((val + 6) & 0xFF);
            program_state->bytes[i + 7] = (char)((val + 7) & 0xFF);
            val += 8;
        }

        /* Half-word stores */
        for (int i = 0; i < ARRAY_SIZE; i += 8) {
            program_state->shorts[i]     = (short)(val & 0xFFFF);
            program_state->shorts[i + 1] = (short)((val + 1) & 0xFFFF);
            program_state->shorts[i + 2] = (short)((val + 2) & 0xFFFF);
            program_state->shorts[i + 3] = (short)((val + 3) & 0xFFFF);
            program_state->shorts[i + 4] = (short)((val + 4) & 0xFFFF);
            program_state->shorts[i + 5] = (short)((val + 5) & 0xFFFF);
            program_state->shorts[i + 6] = (short)((val + 6) & 0xFFFF);
            program_state->shorts[i + 7] = (short)((val + 7) & 0xFFFF);
            val += 8;
        }
    }

    /* Prevent dead code elimination */
    program_state->sink = program_state->data[0] + program_state->bytes[0] + program_state->shorts[0];
    return 0;
}

typedef struct { uint results[64]; ProgramState state; } Storage;
__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global Storage *buffer) {
    __global uint *results = buffer->results;
    __global ProgramState *storage = &buffer->state;
    results[0] = (uint)program_main(storage);
    for (uint i = 0; i < sizeof(storage->sink); ++i)
        results[1 + i] = (uint)(((uint)storage->sink >> (8u * i)) & 255u);
}
