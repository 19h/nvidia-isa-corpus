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
 * Minimal Conditional Branch Chain
 *
 * Generates sequences dominated by compare and branch instructions.
 * Creates many conditional branches with varying outcomes.
 *
 * Expected dominant instructions by architecture:
 *   x86:    cmp, jcc (je, jne, jl, jg, etc.)
 *   ARM:    cmp, b.cc (b.eq, b.ne, b.lt, b.gt, etc.)
 *   RISC-V: beq, bne, blt, bge
 *   MIPS:   beq, bne, slt, slti
 *   PPC:    cmpw, bc
 */



typedef struct {
    volatile int sink;
} ProgramState;

int program_main(__private ProgramState *program_state) {
    int count = 0;
    int a = 0, b = 1, c = 2;

    for (int i = 0; i < 100000; i++) {
        /* Many conditional branches */
        if (a < b) count++;
        if (b > c) count--;
        if (a == 0) count += 2;
        if (c != 0) count -= 1;

        if (a <= b) count++;
        if (b >= c) count--;
        if (a < 100) count += 1;
        if (c > -100) count -= 1;

        /* Update values to vary branch outcomes */
        a = (a + 7) % 200 - 100;
        b = (b + 13) % 200 - 100;
        c = (c + 19) % 200 - 100;

        /* More branches with computed conditions */
        int x = a + b;
        int y = b - c;
        int z = a * c;

        if (x > 0) count++;
        if (y < 0) count--;
        if (z == 0) count += 3;
        if (z != 0) count -= 2;

        if (x > y) count++;
        if (y > z) count--;
        if (x < z) count++;

        /* Nested conditions */
        if (a > 0) {
            if (b > 0) count++;
            else count--;
        } else {
            if (c > 0) count++;
            else count--;
        }
    }

    program_state->sink = count;
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)(((uint)storage.sink >> (8u * i)) & 255u);
}
