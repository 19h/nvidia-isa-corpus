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
 * Minimal Integer Multiplication Chain
 *
 * Generates sequences dominated by integer MUL instructions.
 * Useful for ISA detection - multiplication encodings vary significantly.
 *
 * Expected dominant instructions by architecture:
 *   x86:    imul
 *   ARM:    mul, mla
 *   RISC-V: mul
 *   MIPS:   mult, mul
 *   PPC:    mullw, mulld
 */



typedef struct {
    volatile int sink;
} ProgramState;

int program_main(__private ProgramState *program_state) {
    int a = 3, b = 5, c = 7, d = 11;
    int e = 13, f = 17, g = 19, h = 23;

    for (int i = 0; i < 100000; i++) {
        /* Multiplication chain */
        a = a * b;
        b = b * c;
        c = c * d;
        d = d * 3;

        e = e * f;
        f = f * g;
        g = g * h;
        h = h * 5;

        /* Cross multiplications */
        a = (a * e) & 0xFFFF;  /* Mask to prevent overflow */
        b = (b * f) & 0xFFFF;
        c = (c * g) & 0xFFFF;
        d = (d * h) & 0xFFFF;

        /* Multiply-add patterns */
        e = (e * 7) + 1;
        f = (f * 11) + 2;
        g = (g * 13) + 3;
        h = (h * 17) + 4;

        /* Reset to prevent overflow */
        if (i % 100 == 0) {
            a = 3; b = 5; c = 7; d = 11;
            e = 13; f = 17; g = 19; h = 23;
        }
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
