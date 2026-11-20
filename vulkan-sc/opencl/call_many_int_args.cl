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
 * Many Integer Arguments
 *
 * Functions with many integer arguments to exercise register/stack passing.
 * Tests the calling convention's allocation and stack spillage.
 *
 * Features exercised:
 *   - Register arguments (first N args)
 *   - Stack arguments (remaining args)
 *   - Argument ordering
 */





/* Prevent inlining to preserve call semantics */
typedef struct {
    volatile int64_t sink;
} ProgramState;


int64_t sum_8_args(__private ProgramState *program_state, int a, int b, int c, int d, int e, int f, int g, int h) {
    return (int64_t)a + b + c + d + e + f + g + h;
}


int64_t sum_12_args(__private ProgramState *program_state, int a, int b, int c, int d, int e, int f,
                    int g, int h, int i, int j, int k, int l) {
    return (int64_t)a + b + c + d + e + f + g + h + i + j + k + l;
}


int64_t sum_16_args(__private ProgramState *program_state, int a, int b, int c, int d, int e, int f, int g, int h,
                    int i, int j, int k, int l, int m, int n, int o, int p) {
    return (int64_t)a + b + c + d + e + f + g + h +
           i + j + k + l + m + n + o + p;
}


int64_t weighted_sum(__private ProgramState *program_state, int a, int b, int c, int d, int e, int f, int g, int h,
                     int w1, int w2, int w3, int w4, int w5, int w6, int w7, int w8) {
    return (int64_t)a * w1 + (int64_t)b * w2 + (int64_t)c * w3 + (int64_t)d * w4 +
           (int64_t)e * w5 + (int64_t)f * w6 + (int64_t)g * w7 + (int64_t)h * w8;
}

/* 64-bit arguments on 32-bit platforms require pairs */

int64_t sum_64bit_args(__private ProgramState *program_state, int64_t a, int64_t b, int64_t c, int64_t d,
                       int64_t e, int64_t f, int64_t g, int64_t h) {
    return a + b + c + d + e + f + g + h;
}

/* Mixed sizes */

int64_t mixed_sizes(__private ProgramState *program_state, int8_t a, int16_t b, int32_t c, int64_t d,
                    uint8_t e, uint16_t f, uint32_t g, uint64_t h) {
    return (int64_t)a + b + c + d + e + f + g + (int64_t)h;
}

/* Recursive with many args */

int64_t recursive_sum(__private ProgramState *program_state, int depth, int a, int b, int c, int d,
                      int e, int f, int g, int h) {
    for (; depth > 0; --depth) {
        a += 1; b += 1; c += 1; d += 1; e += 1; f += 1; g += 1; h += 1;
    }
    return (int64_t)a + b + c + d + e + f + g + h;
}

int program_main(__private ProgramState *program_state) {
    int64_t result = 0;

    for (int i = 0; i < 100000; i++) {
        result += sum_8_args(program_state, i, i+1, i+2, i+3, i+4, i+5, i+6, i+7);

        result += sum_12_args(program_state, i, i+1, i+2, i+3, i+4, i+5,
                             i+6, i+7, i+8, i+9, i+10, i+11);

        result += sum_16_args(program_state, i, i+1, i+2, i+3, i+4, i+5, i+6, i+7,
                             i+8, i+9, i+10, i+11, i+12, i+13, i+14, i+15);

        result += weighted_sum(program_state, i, i+1, i+2, i+3, i+4, i+5, i+6, i+7,
                              1, 2, 3, 4, 5, 6, 7, 8);

        result += sum_64bit_args(program_state, i, i*2, i*3, i*4, i*5, i*6, i*7, i*8);

        result += mixed_sizes(program_state, (int8_t)i, (int16_t)i, i, (int64_t)i,
                             (uint8_t)i, (uint16_t)i, (uint32_t)i, (uint64_t)i);

        if (i % 1000 == 0) {
            result += recursive_sum(program_state, 5, i, i+1, i+2, i+3, i+4, i+5, i+6, i+7);
        }
    }

    program_state->sink = result;
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)(((ulong)storage.sink >> (8u * i)) & 255u);
}
