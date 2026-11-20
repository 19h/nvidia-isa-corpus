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
 * Floating-Point Piecewise Chaos
 *
 * Iterates several bounded piecewise maps and folds their output into an
 * integer checksum. The workload stresses floating-point arithmetic, branch
 * selection on FP ranges, loop-carried dependencies, and FP-to-integer
 * quantization without requiring libm.
 *
 * Features exercised:
 *   - Double and float arithmetic
 *   - Piecewise FP branches
 *   - Quantized checksum accumulation
 *   - Bounded chaotic recurrence
 */





#define ORBITS 32
#define STEPS 384

typedef struct {
    double x;
    double y;
    double z;
    double carry;
} Orbit;





typedef struct {
    volatile uint64_t sink;
    Orbit orbits[ORBITS];
} ProgramState;

static uint32_t lcg_next(__private ProgramState *program_state, uint32_t *state) {
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

static double unit_from_u32(__private ProgramState *program_state, uint32_t x) {
    return ((double)(x >> 8) / 16777216.0) * 2.0 - 1.0;
}

static double fold_signed(__private ProgramState *program_state, double v) {
    if (v > 1.0) {
        v = 2.0 - v;
    } else if (v < -1.0) {
        v = -2.0 - v;
    }

    if (v > 1.0) {
        v = 1.0;
    } else if (v < -1.0) {
        v = -1.0;
    }

    return v;
}

static uint64_t mix64(__private ProgramState *program_state, uint64_t h, uint64_t v) {
    h ^= v + UINT64_C(0x9e3779b97f4a7c15) + (h << 6) + (h >> 2);
    h ^= h >> 33;
    h *= UINT64_C(0xff51afd7ed558ccd);
    h ^= h >> 29;
    return h;
}

static uint32_t quantize(__private ProgramState *program_state, double v) {
    if (v > 1.25) {
        v = 1.25;
    } else if (v < -1.25) {
        v = -1.25;
    }

    int32_t q;
    if (v >= 0.0) {
        q = (int32_t)(v * 1000000.0 + 0.5);
    } else {
        q = (int32_t)(v * 1000000.0 - 0.5);
    }
    return (uint32_t)q;
}

static void init_orbits(__private ProgramState *program_state) {
    uint32_t seed = UINT32_C(0x4d6f726f);

    for (int i = 0; i < ORBITS; i++) {
        double sx = unit_from_u32(program_state, lcg_next(program_state, &seed));
        double sy = unit_from_u32(program_state, lcg_next(program_state, &seed));
        double sz = unit_from_u32(program_state, lcg_next(program_state, &seed));

        program_state->orbits[i].x = 0.73 * sx;
        program_state->orbits[i].y = 0.67 * sy;
        program_state->orbits[i].z = 0.59 * sz;
        program_state->orbits[i].carry = 0.125 + 0.00390625 * (double)(i + 1);
    }
}

static uint32_t step_orbit(__private ProgramState *program_state, Orbit *o, int step, int index) {
    double x = o->x;
    double y = o->y;
    double z = o->z;
    double drive = 0.001953125 * (double)((step * 17 + index * 29) & 63) - 0.0615234375;

    double nx;
    if (x < -0.42) {
        nx = -0.71 * x + 0.29 * y * y - 0.17 * z + drive;
    } else if (x < 0.11) {
        nx = x * (1.28 - 0.43 * y) + 0.24 * z * z - 0.35 * drive;
    } else if (x < 0.58) {
        nx = 0.61 - x * x + 0.21 * y - 0.12 * z * drive;
    } else {
        nx = 0.39 * x - 0.56 * y + 0.18 * x * z + 0.07 * drive;
    }

    double ny;
    double gate = y + 0.25 * x - 0.125 * z;
    if (gate < -0.55) {
        ny = 0.64 * y + 0.19 * x * z + 0.03125 * (double)(index & 7);
    } else if (gate < -0.05) {
        ny = -0.82 * y + 0.33 * x - 0.14 * z * z;
    } else if (gate < 0.47) {
        ny = y + 0.27 * (x - y * z) + drive;
    } else {
        ny = -0.37 * x + 0.72 * z - 0.22 * y * y;
    }

    double nz;
    if (z < -0.25) {
        nz = 0.48 * z + 0.31 * x - 0.26 * y + o->carry;
    } else if (z < 0.33) {
        nz = z * (0.93 - 0.25 * x) + 0.18 * x * y - drive;
    } else {
        nz = -0.58 * z + 0.34 * y + 0.16 * x * x - o->carry;
    }

    o->x = fold_signed(program_state, nx);
    o->y = fold_signed(program_state, ny);
    o->z = fold_signed(program_state, nz);

    float fx = (float)o->x;
    float fy = (float)o->y;
    float fz = (float)o->z;
    float fc = (float)o->carry;
    o->carry = 0.5 * o->carry + (double)(0.125f * fx + 0.0625f * fy - 0.03125f * fz + 0.015625f * fc);
    o->carry = fold_signed(program_state, o->carry);

    return quantize(program_state, o->x + 0.5 * o->y - 0.25 * o->z + 0.125 * o->carry);
}

int program_main(__private ProgramState *program_state) {
    init_orbits(program_state);

    uint64_t checksum = UINT64_C(0x6d6f726f5f667031);
    uint64_t branch_score = 0;

    for (int step = 0; step < STEPS; step++) {
        for (int i = 0; i < ORBITS; i++) {
            uint32_t q = step_orbit(program_state, &program_state->orbits[i], step, i);
            checksum = mix64(program_state, checksum, (uint64_t)q ^ ((uint64_t)(uint32_t)step << 32) ^ (uint32_t)i);

            if (program_state->orbits[i].x < -0.33) {
                branch_score += 3u;
            } else if (program_state->orbits[i].x < 0.33) {
                branch_score += 5u;
            } else {
                branch_score += 7u;
            }
        }
    }

    program_state->sink = checksum ^ (branch_score << 1);
    printf("fp_piecewise_chaos checksum=%016" PRIx64 " branches=%" PRIu64 "\n", program_state->sink, branch_score);
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)(((ulong)storage.sink >> (8u * i)) & 255u);
}
