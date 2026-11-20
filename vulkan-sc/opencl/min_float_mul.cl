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
 * Minimal Floating Point Multiplication Chain
 *
 * Generates sequences dominated by FP multiply instructions.
 * Also includes FMA (fused multiply-add) patterns when available.
 *
 * Expected dominant instructions by architecture:
 *   x86:    mulss, mulsd (SSE), vmulss, vmulsd, vfmadd (AVX/FMA)
 *   ARM:    fmul, fmadd
 *   RISC-V: fmul.s, fmul.d, fmadd.s, fmadd.d
 *   MIPS:   mul.s, mul.d, madd.s, madd.d
 *   PPC:    fmuls, fmul, fmadd, fmsub
 */




typedef struct {
    volatile double sink_d;
    volatile float sink_f;
} ProgramState;

int program_main(__private ProgramState *program_state) {
    /* Single precision */
    float fa = 1.01f, fb = 1.02f, fc = 1.03f, fd = 1.04f;
    float fe = 0.99f, ff = 0.98f, fg = 0.97f, fh = 0.96f;

    /* Double precision */
    double da = 1.001, db = 1.002, dc = 1.003, dd = 1.004;
    double de = 0.999, df = 0.998, dg = 0.997, dh = 0.996;

    for (int i = 0; i < 100000; i++) {
        /* Single precision multiplications */
        fa = fa * fb;
        fb = fb * fc;
        fc = fc * fd;
        fd = fd * fe;

        fe = fe * ff;
        ff = ff * fg;
        fg = fg * fh;
        fh = fh * fa;

        /* FMA patterns: a * b + c */
        fa = fa * fe + 0.001f;
        fb = fb * ff + 0.001f;
        fc = fc * fg + 0.001f;
        fd = fd * fh + 0.001f;

        /* Double precision multiplications */
        da = da * db;
        db = db * dc;
        dc = dc * dd;
        dd = dd * de;

        de = de * df;
        df = df * dg;
        dg = dg * dh;
        dh = dh * da;

        /* FMA patterns */
        da = da * de + 0.0001;
        db = db * df + 0.0001;
        dc = dc * dg + 0.0001;
        dd = dd * dh + 0.0001;

        /* Normalize periodically to stay in reasonable range */
        if (i % 1000 == 0) {
            fa = 1.01f; fb = 1.02f; fc = 1.03f; fd = 1.04f;
            fe = 0.99f; ff = 0.98f; fg = 0.97f; fh = 0.96f;
            da = 1.001; db = 1.002; dc = 1.003; dd = 1.004;
            de = 0.999; df = 0.998; dg = 0.997; dh = 0.996;
        }
    }

    program_state->sink_f = fa * fb * fc * fd * fe * ff * fg * fh;
    program_state->sink_d = da * db * dc * dd * de * df * dg * dh;
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
