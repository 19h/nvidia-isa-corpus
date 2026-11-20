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
 * Minimal Floating Point Addition Chain
 *
 * Generates sequences dominated by FP add instructions.
 * Exercises the floating point unit's addition pipeline.
 *
 * Expected dominant instructions by architecture:
 *   x86:    addss, addsd, addps, addpd (SSE), vaddss, vaddsd (AVX)
 *   ARM:    fadd (scalar/vector)
 *   RISC-V: fadd.s, fadd.d
 *   MIPS:   add.s, add.d
 *   PPC:    fadds, fadd
 */




typedef struct {
    volatile double sink_d;
    volatile float sink_f;
} ProgramState;

int program_main(__private ProgramState *program_state) {
    /* Single precision chain */
    float fa = 1.0f, fb = 0.1f, fc = 0.01f, fd = 0.001f;
    float fe = 0.5f, ff = 0.05f, fg = 0.005f, fh = 0.0005f;

    /* Double precision chain */
    double da = 1.0, db = 0.1, dc = 0.01, dd = 0.001;
    double de = 0.5, df = 0.05, dg = 0.005, dh = 0.0005;

    for (int i = 0; i < 100000; i++) {
        /* Single precision additions */
        fa = fa + fb;
        fb = fb + fc;
        fc = fc + fd;
        fd = fd + fa;

        fe = fe + ff;
        ff = ff + fg;
        fg = fg + fh;
        fh = fh + fe;

        fa = fa + fe;
        fb = fb + ff;
        fc = fc + fg;
        fd = fd + fh;

        /* Double precision additions */
        da = da + db;
        db = db + dc;
        dc = dc + dd;
        dd = dd + da;

        de = de + df;
        df = df + dg;
        dg = dg + dh;
        dh = dh + de;

        da = da + de;
        db = db + df;
        dc = dc + dg;
        dd = dd + dh;

        /* Reset periodically to prevent overflow */
        if (i % 10000 == 0) {
            fa = 1.0f; fb = 0.1f; fc = 0.01f; fd = 0.001f;
            fe = 0.5f; ff = 0.05f; fg = 0.005f; fh = 0.0005f;
            da = 1.0; db = 0.1; dc = 0.01; dd = 0.001;
            de = 0.5; df = 0.05; dg = 0.005; dh = 0.0005;
        }
    }

    program_state->sink_f = fa + fb + fc + fd + fe + ff + fg + fh;
    program_state->sink_d = da + db + dc + dd + de + df + dg + dh;
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
