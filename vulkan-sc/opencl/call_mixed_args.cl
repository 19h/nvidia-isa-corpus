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
 * Mixed Integer and Floating Point Arguments
 *
 * Tests interleaved int/float arguments.
 * Most ABIs use separate sets for int/float.
 *
 * Features exercised:
 *   - Interleaved argument types
 *   - Register allocation across types
 *   - Type promotion/demotion
 */





typedef struct {
    volatile double sink;
} ProgramState;


double mixed_8(__private ProgramState *program_state, int a, double b, int c, double d,
               int e, double f, int g, double h) {
    return (double)a + b + (double)c + d + (double)e + f + (double)g + h;
}


double mixed_16(__private ProgramState *program_state, int a1, double b1, int a2, double b2,
                int a3, double b3, int a4, double b4,
                int a5, double b5, int a6, double b6,
                int a7, double b7, int a8, double b8) {
    return (double)(a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8) +
           b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8;
}


double float_int_float(__private ProgramState *program_state, float a, int b, float c, int d,
                       float e, int f, float g, int h) {
    return (double)a + (double)b + (double)c + (double)d +
           (double)e + (double)f + (double)g + (double)h;
}


double pointer_and_values(__private ProgramState *program_state, int *p1, double v1, int *p2, double v2,
                          int *p3, double v3, int *p4, double v4) {
    return (double)*p1 + v1 + (double)*p2 + v2 +
           (double)*p3 + v3 + (double)*p4 + v4;
}


double many_mixed(__private ProgramState *program_state, int8_t a, float b, int16_t c, double d,
                  int32_t e, float f, int64_t g, double h,
                  uint8_t i, float j, uint16_t k, double l) {
    return (double)a + b + (double)c + d + (double)e + f +
           (double)g + h + (double)i + j + (double)k + l;
}

int program_main(__private ProgramState *program_state) {
    double result = 0.0;
    int vals[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    for (int i = 0; i < 100000; i++) {
        double d = (double)i * 0.001;

        result += mixed_8(program_state, i, d, i+1, d+0.1, i+2, d+0.2, i+3, d+0.3);

        result += mixed_16(program_state, i, d, i+1, d+0.1, i+2, d+0.2, i+3, d+0.3,
                          i+4, d+0.4, i+5, d+0.5, i+6, d+0.6, i+7, d+0.7);

        result += float_int_float(program_state, (float)d, i, (float)(d+0.1), i+1,
                                 (float)(d+0.2), i+2, (float)(d+0.3), i+3);

        result += pointer_and_values(program_state, &vals[0], d, &vals[1], d+0.1,
                                    &vals[2], d+0.2, &vals[3], d+0.3);

        result += many_mixed(program_state, (int8_t)i, (float)d, (int16_t)i, d,
                            i, (float)d, (int64_t)i, d,
                            (uint8_t)i, (float)d, (uint16_t)i, d);

        vals[i % 8] = i;
    }

    program_state->sink = result;
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)((as_ulong(storage.sink) >> (8u * i)) & 255u);
}
