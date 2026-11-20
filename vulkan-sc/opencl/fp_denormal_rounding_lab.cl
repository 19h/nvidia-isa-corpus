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
// SPDX-License-Identifier: MIT
/*
 * Floating-Point Denormal Rounding Lab
 *
 * Exercises tiny finite FP values, scaling back into normal ranges,
 * cancellation against larger anchors, branch decisions, and integer checksum
 * quantization. The output is a compact integer checksum instead of relying on
 * exact decimal formatting of floating-point values.
 */





#define CELLS 18
#define STEPS 224
#define D_TINY_COUNT 12
#define F_TINY_COUNT 10

typedef struct {
    double sum;
    double residue;
    float drift;
    uint32_t flags;
} LabCell;





__constant static const double d_scale[8] = {
    0x1p1022,
    0x1p1018,
    0x1p1008,
    0x1p996,
    0x1p984,
    0x1p972,
    0x1p960,
    0x1p948
};

__constant static const float f_scale[8] = {
    0x1p126f,
    0x1p123f,
    0x1p119f,
    0x1p115f,
    0x1p111f,
    0x1p107f,
    0x1p103f,
    0x1p99f
};




typedef struct {
    volatile double tiny_d[D_TINY_COUNT];
    volatile float tiny_f[F_TINY_COUNT];
    LabCell cells[CELLS];
    volatile uint64_t sink;
} ProgramState;

static uint64_t rotl64(__private ProgramState *program_state, uint64_t x, unsigned n) {
    n &= 63u;
    return n == 0u ? x : ((x << n) | (x >> (64u - n)));
}

static uint64_t mix64(__private ProgramState *program_state, uint64_t h, uint64_t v) {
    h ^= v + UINT64_C(0x9e3779b97f4a7c15) + (h << 6) + (h >> 2);
    h ^= h >> 31;
    h *= UINT64_C(0x7fb5d329728ea185);
    h ^= h >> 27;
    h *= UINT64_C(0x81dadef4bc2dd44d);
    return h ^ (h >> 33);
}

static double fold_unit(__private ProgramState *program_state, double v) {
    if (v > 2.0) {
        v = 4.0 - v;
    } else if (v < -2.0) {
        v = -4.0 - v;
    }

    if (v > 2.0) {
        v = 2.0;
    } else if (v < -2.0) {
        v = -2.0;
    }

    return v;
}

static uint32_t quantize_double(__private ProgramState *program_state, double v) {
    if (v > 8.0) {
        v = 8.0;
    } else if (v < -8.0) {
        v = -8.0;
    }

    int32_t q;
    if (v >= 0.0) {
        q = (int32_t)(v * 1048576.0 + 0.5);
    } else {
        q = (int32_t)(v * 1048576.0 - 0.5);
    }
    return (uint32_t)q;
}

static void init_cells(__private ProgramState *program_state) {
    for (int i = 0; i < CELLS; i++) {
        double td = program_state->tiny_d[i % D_TINY_COUNT] * d_scale[i & 7];
        float tf = program_state->tiny_f[(i * 3) % F_TINY_COUNT] * f_scale[(i + 2) & 7];

        program_state->cells[i].sum = 0.03125 * (double)(i - 9) + td;
        program_state->cells[i].residue = 0.0009765625 * (double)((i * 5 + 3) & 15) - td * 0.25;
        program_state->cells[i].drift = 0.015625f * (float)(i - 4) + tf;
        program_state->cells[i].flags = (uint32_t)(i * 17 + 5);
    }
}

static uint32_t step_cell(__private ProgramState *program_state, LabCell *cell, int step, int index, uint64_t *branches) {
    double raw = program_state->tiny_d[(step + index * 5) % D_TINY_COUNT];
    double scaled = raw * d_scale[(step + index) & 7];
    double anchor = ((step + index) & 1) != 0 ? 1.0 : -1.0;
    double cancelled = (anchor + scaled) - anchor;
    double recovered = scaled - cancelled;
    double skew = program_state->tiny_d[(step * 3 + index) % D_TINY_COUNT] * d_scale[(step + 3) & 7];

    if (cancelled == 0.0) {
        *branches += 11u;
        cell->flags ^= UINT32_C(0x00010001) + (uint32_t)step;
    } else if ((cancelled > 0.0) == (scaled > 0.0)) {
        *branches += 17u;
        cell->flags += (uint32_t)(index * 29 + step);
    } else {
        *branches += 23u;
        cell->flags = (cell->flags << 3) ^ (cell->flags >> 5) ^ UINT32_C(0xa5a5);
    }

    float fraw = program_state->tiny_f[(step * 7 + index * 2) % F_TINY_COUNT];
    float fscaled = fraw * f_scale[(step + index * 3) & 7];
    float fanchor = ((step + index) & 2) != 0 ? 0.5f : -0.5f;
    float fcancelled = (fanchor + fscaled) - fanchor;

    if (fcancelled == 0.0f) {
        *branches += 3u;
        cell->drift *= 0.875f;
    } else if (fcancelled > 0.0f) {
        *branches += 5u;
        cell->drift = cell->drift * 0.8125f + fcancelled * 48.0f;
    } else {
        *branches += 7u;
        cell->drift = cell->drift * 0.8125f - fcancelled * 32.0f;
    }

    double drift = (double)cell->drift;
    double next = cell->sum * 0.78125 + cancelled * 64.0 - recovered * 0.03125;
    next += skew * 3.0 + drift * 0.0625 + cell->residue * 0.125;

    cell->residue = fold_unit(program_state, cell->residue * -0.625 + recovered * 512.0 + (double)fcancelled);
    cell->sum = fold_unit(program_state, next);

    double probe = cell->sum + cell->residue * 0.5 + (double)(int32_t)(cell->flags & 255u) * 0.0001220703125;
    return quantize_double(program_state, probe);
}

int program_main(__private ProgramState *program_state) {
    init_cells(program_state);

    uint64_t checksum = UINT64_C(0x66705f64656e6f72);
    uint64_t branches = 0;

    for (int step = 0; step < STEPS; step++) {
        for (int i = 0; i < CELLS; i++) {
            uint32_t q = step_cell(program_state, &program_state->cells[i], step, i, &branches);
            uint64_t tag = ((uint64_t)(uint32_t)step << 32) ^ (uint32_t)i;
            tag ^= (uint64_t)q ^ ((uint64_t)program_state->cells[i].flags << ((i & 1) ? 17 : 5));
            checksum = mix64(program_state, rotl64(program_state, checksum, (unsigned)(i + 7)), tag);
        }
    }

    program_state->sink = checksum ^ rotl64(program_state, branches, 9);
    printf("fp_denormal_rounding_lab checksum=%016" PRIx64 " branches=%" PRIu64 "\n", program_state->sink, branches);
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {.tiny_d = {
    0x1p-1074,
    -0x1p-1074,
    0x1p-1070,
    -0x1.8p-1072,
    0x1p-1060,
    -0x1.4p-1055,
    0x1p-1040,
    -0x1.2p-1033,
    0x1p-1022,
    -0x1.8p-1022,
    0x1.4p-1020,
    -0x1.1p-1018
}, .tiny_f = {
    0x1p-149f,
    -0x1p-149f,
    0x1p-145f,
    -0x1.8p-144f,
    0x1p-140f,
    -0x1.2p-136f,
    0x1p-130f,
    -0x1.8p-128f,
    0x1p-126f,
    -0x1.4p-124f
}};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)(((ulong)storage.sink >> (8u * i)) & 255u);
}
