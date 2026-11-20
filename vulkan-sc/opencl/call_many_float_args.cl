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
 * Many Floating Point Arguments
 *
 * Functions with many FP arguments to exercise FP passing.
 * Most ABIs have separate FP registers for arguments.
 *
 * Features exercised:
 *   - FP arguments (XMM, NEON, etc.)
 *   - Stack spillage for excess FP args
 *   - Single/double precision handling
 */




typedef struct {
    volatile double sink_d;
    volatile float sink_f;
} ProgramState;


double sum_8_doubles(__private ProgramState *program_state, double a, double b, double c, double d,
                     double e, double f, double g, double h) {
    return a + b + c + d + e + f + g + h;
}


double sum_12_doubles(__private ProgramState *program_state, double a, double b, double c, double d,
                      double e, double f, double g, double h,
                      double i, double j, double k, double l) {
    return a + b + c + d + e + f + g + h + i + j + k + l;
}


float sum_8_floats(__private ProgramState *program_state, float a, float b, float c, float d,
                   float e, float f, float g, float h) {
    return a + b + c + d + e + f + g + h;
}


float sum_16_floats(__private ProgramState *program_state, float a, float b, float c, float d,
                    float e, float f, float g, float h,
                    float i, float j, float k, float l,
                    float m, float n, float o, float p) {
    return a + b + c + d + e + f + g + h +
           i + j + k + l + m + n + o + p;
}


double weighted_double(__private ProgramState *program_state, double a, double b, double c, double d,
                       double w1, double w2, double w3, double w4) {
    return a * w1 + b * w2 + c * w3 + d * w4;
}


double dot_product_8(__private ProgramState *program_state, double a1, double a2, double a3, double a4,
                     double b1, double b2, double b3, double b4) {
    return a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4;
}

/* Recursive with many float args */

double recursive_sum_f(__private ProgramState *program_state, int depth, double a, double b, double c, double d,
                       double e, double f, double g, double h) {
    for (; depth > 0; --depth) {
        a += 0.1; b += 0.1; c += 0.1; d += 0.1; e += 0.1; f += 0.1; g += 0.1; h += 0.1;
    }
    return a + b + c + d + e + f + g + h;
}

int program_main(__private ProgramState *program_state) {
    double result_d = 0.0;
    float result_f = 0.0f;

    for (int i = 0; i < 100000; i++) {
        double d = (double)i * 0.001;
        float f = (float)i * 0.001f;

        result_d += sum_8_doubles(program_state, d, d+0.1, d+0.2, d+0.3,
                                  d+0.4, d+0.5, d+0.6, d+0.7);

        result_d += sum_12_doubles(program_state, d, d+0.1, d+0.2, d+0.3, d+0.4, d+0.5,
                                   d+0.6, d+0.7, d+0.8, d+0.9, d+1.0, d+1.1);

        result_f += sum_8_floats(program_state, f, f+0.1f, f+0.2f, f+0.3f,
                                f+0.4f, f+0.5f, f+0.6f, f+0.7f);

        result_f += sum_16_floats(program_state, f, f+0.1f, f+0.2f, f+0.3f, f+0.4f, f+0.5f, f+0.6f, f+0.7f,
                                 f+0.8f, f+0.9f, f+1.0f, f+1.1f, f+1.2f, f+1.3f, f+1.4f, f+1.5f);

        result_d += weighted_double(program_state, d, d+1, d+2, d+3, 1.0, 2.0, 3.0, 4.0);

        result_d += dot_product_8(program_state, d, d+1, d+2, d+3, 0.5, 0.5, 0.5, 0.5);

        if (i % 1000 == 0) {
            result_d += recursive_sum_f(program_state, 5, d, d+0.1, d+0.2, d+0.3,
                                        d+0.4, d+0.5, d+0.6, d+0.7);
        }
    }

    program_state->sink_d = result_d;
    program_state->sink_f = result_f;
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink_d); ++i)
        results[1 + i] = (uint)((as_ulong(storage.sink_d) >> (8u * i)) & 255u);
    for (uint i = 0; i < sizeof(storage.sink_f); ++i)
        results[17 + i] = (uint)((as_uint(storage.sink_f) >> (8u * i)) & 255u);
}
