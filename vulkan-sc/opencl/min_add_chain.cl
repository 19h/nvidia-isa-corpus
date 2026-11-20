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
 * Minimal Integer Addition Chain
 *
 * Generates sequences dominated by integer ADD instructions.
 * Useful for ISA detection training - creates distinctive ADD patterns.
 *
 * Expected dominant instructions by architecture:
 *   x86:    add, lea
 *   ARM:    add, adds
 *   RISC-V: add, addi
 *   MIPS:   addu, addiu
 *   PPC:    add, addi
 */



typedef struct {
    volatile int sink;
} ProgramState;

int program_main(__private ProgramState *program_state) {
    int a = 1, b = 2, c = 3, d = 4;
    int e = 5, f = 6, g = 7, h = 8;

    for (int i = 0; i < 100000; i++) {
        /* Chain of dependent additions */
        a = a + b;
        b = b + c;
        c = c + d;
        d = d + a;

        e = e + f;
        f = f + g;
        g = g + h;
        h = h + e;

        /* Cross-chain additions */
        a = a + e;
        b = b + f;
        c = c + g;
        d = d + h;

        /* More additions to increase density */
        e = e + a + 1;
        f = f + b + 2;
        g = g + c + 3;
        h = h + d + 4;
    }

    program_state->sink = a + b + c + d + e + f + g + h;
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)(((uint)storage.sink >> (8u * i)) & 255u);
}
