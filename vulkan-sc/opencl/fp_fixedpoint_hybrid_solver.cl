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
 * Floating-Point Fixed-Point Hybrid Solver
 *
 * Runs a small deterministic relaxation solver that mixes Q16.16 fixed-point
 * integer math with float and double iterative updates. The checksum folds
 * quantized state so the result line is stable and compact.
 */





#define NODES 20
#define ITERS 180
#define Q_SHIFT 16
#define Q_ONE (INT32_C(1) << Q_SHIFT)
#define Q_LIMIT (INT32_C(10) << Q_SHIFT)

typedef struct {
    int32_t xq;
    int32_t yq;
    int32_t biasq;
    float vf;
    double vd;
    uint32_t tag;
} SolverNode;

__constant static const int32_t coeff_q[16] = {
    39322, -14746, 8192, -22938,
    11469, 32768, -6554, 27853,
    -9830, 16384, 4915, -19661,
    24576, -12288, 20480, -4096
};

__constant static const float relax_f[12] = {
    0.109375f, 0.078125f, 0.140625f, 0.0625f,
    0.09375f, 0.15625f, 0.046875f, 0.125f,
    0.0703125f, 0.1171875f, 0.0859375f, 0.1328125f
};

__constant static const double weight_d[14] = {
    0.3125, -0.1875, 0.140625, -0.09375,
    0.21875, 0.0625, -0.15625, 0.1015625,
    -0.0546875, 0.171875, -0.125, 0.0390625,
    0.265625, -0.078125
};




typedef struct {
    SolverNode nodes[NODES];
    volatile uint64_t sink;
} ProgramState;

static uint64_t rotl64(__private ProgramState *program_state, uint64_t x, unsigned n) {
    n &= 63u;
    return n == 0u ? x : ((x << n) | (x >> (64u - n)));
}

static uint64_t mix64(__private ProgramState *program_state, uint64_t h, uint64_t v) {
    h ^= v + UINT64_C(0x9e3779b97f4a7c15) + (h << 6) + (h >> 2);
    h ^= h >> 30;
    h *= UINT64_C(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h *= UINT64_C(0x94d049bb133111eb);
    return h ^ (h >> 31);
}

static int32_t clamp_q(__private ProgramState *program_state, int64_t v) {
    if (v > (int64_t)Q_LIMIT) {
        return Q_LIMIT;
    }
    if (v < -(int64_t)Q_LIMIT) {
        return -Q_LIMIT;
    }
    return (int32_t)v;
}

static int32_t add_q(__private ProgramState *program_state, int32_t a, int32_t b) {
    return clamp_q(program_state, (int64_t)a + (int64_t)b);
}

static int32_t mul_q(__private ProgramState *program_state, int32_t a, int32_t b) {
    int64_t prod = (int64_t)a * (int64_t)b;
    if (prod >= 0) {
        prod += INT64_C(1) << (Q_SHIFT - 1);
    } else {
        prod -= INT64_C(1) << (Q_SHIFT - 1);
    }
    return clamp_q(program_state, prod / Q_ONE);
}

static double q_to_double(__private ProgramState *program_state, int32_t q) {
    return (double)q / (double)Q_ONE;
}

static int32_t q_from_double(__private ProgramState *program_state, double v) {
    if (v > 9.75) {
        v = 9.75;
    } else if (v < -9.75) {
        v = -9.75;
    }

    if (v >= 0.0) {
        return (int32_t)(v * (double)Q_ONE + 0.5);
    }
    return (int32_t)(v * (double)Q_ONE - 0.5);
}

static float clamp_float(__private ProgramState *program_state, float v) {
    if (v > 8.0f) {
        return 8.0f;
    }
    if (v < -8.0f) {
        return -8.0f;
    }
    return v;
}

static double clamp_double(__private ProgramState *program_state, double v) {
    if (v > 8.0) {
        return 8.0;
    }
    if (v < -8.0) {
        return -8.0;
    }
    return v;
}

static uint32_t splitmix32(__private ProgramState *program_state, uint32_t x) {
    x += UINT32_C(0x9e3779b9);
    x = (x ^ (x >> 16)) * UINT32_C(0x85ebca6b);
    x = (x ^ (x >> 13)) * UINT32_C(0xc2b2ae35);
    return x ^ (x >> 16);
}

static uint32_t quantize_double(__private ProgramState *program_state, double v) {
    if (v > 16.0) {
        v = 16.0;
    } else if (v < -16.0) {
        v = -16.0;
    }

    int32_t q;
    if (v >= 0.0) {
        q = (int32_t)(v * 262144.0 + 0.5);
    } else {
        q = (int32_t)(v * 262144.0 - 0.5);
    }
    return (uint32_t)q;
}

static void init_nodes(__private ProgramState *program_state) {
    uint32_t seed = UINT32_C(0x6d6f726f);

    for (int i = 0; i < NODES; i++) {
        seed = splitmix32(program_state, seed + (uint32_t)i * UINT32_C(97));
        int32_t sx = (int32_t)(seed & UINT32_C(0x3ffff)) - INT32_C(0x20000);

        seed = splitmix32(program_state, seed ^ UINT32_C(0xa5a55a5a));
        int32_t sy = (int32_t)(seed & UINT32_C(0x3ffff)) - INT32_C(0x20000);

        seed = splitmix32(program_state, seed + UINT32_C(0x13579bdf));
        int32_t sb = (int32_t)(seed & UINT32_C(0x1ffff)) - INT32_C(0x10000);

        program_state->nodes[i].xq = sx;
        program_state->nodes[i].yq = sy;
        program_state->nodes[i].biasq = sb / 3;
        program_state->nodes[i].vf = (float)(sx - sy) / (float)(Q_ONE * 4);
        program_state->nodes[i].vd = (double)(sx + sy + sb) / (double)(Q_ONE * 5);
        program_state->nodes[i].tag = seed ^ (uint32_t)i * UINT32_C(0x45d9f3b);
    }
}

static int32_t signed_magnitude_u(__private ProgramState *program_state, uint32_t x) {
    uint32_t m = x & UINT32_C(0x7fff);
    return (x & UINT32_C(0x8000)) != 0u ? -(int32_t)m : (int32_t)m;
}

static uint32_t solve_node(__private ProgramState *program_state, int i, int iter, uint64_t *branch_score) {
    SolverNode *n = &program_state->nodes[i];
    SolverNode *left = &program_state->nodes[(i + NODES - 1) % NODES];
    SolverNode *right = &program_state->nodes[(i * 7 + iter + 3) % NODES];

    int32_t ax = coeff_q[(i + iter) & 15];
    int32_t ay = coeff_q[(i * 3 + iter + 5) & 15];
    int32_t cross = mul_q(program_state, left->xq, ay) - mul_q(program_state, right->yq, ax);
    int32_t residual_q = add_q(program_state, n->biasq, cross / 2);
    residual_q = add_q(program_state, residual_q, -mul_q(program_state, n->xq, coeff_q[(iter + 9) & 15]) / 3);

    double residual_d = q_to_double(program_state, residual_q);
    double neighbor_d = q_to_double(program_state, left->yq - right->xq);
    double curve = n->vd * weight_d[(i + iter) % 14] + (double)n->vf * 0.375;
    double correction = residual_d * weight_d[(iter + 4) % 14] - neighbor_d * 0.0625 + curve;
    float fcorrection = (float)residual_d * relax_f[(i + iter) % 12] + n->vf * 0.25f;

    int32_t delta_q = q_from_double(program_state, correction * 0.5 + (double)fcorrection * 0.25);
    int32_t jitter = signed_magnitude_u(program_state, n->tag >> ((iter + i) & 7)) * 2;

    if (residual_q > (Q_ONE / 3)) {
        *branch_score += 19u;
        delta_q = add_q(program_state, delta_q, jitter);
        n->tag ^= (uint32_t)residual_q + UINT32_C(0x10001);
    } else if (residual_q < -(Q_ONE / 3)) {
        *branch_score += 23u;
        delta_q = add_q(program_state, delta_q, -jitter);
        n->tag += (uint32_t)(-residual_q) ^ UINT32_C(0x9e37);
    } else if (correction > (double)fcorrection) {
        *branch_score += 29u;
        delta_q = add_q(program_state, delta_q / 2, coeff_q[(i + 11) & 15] / 8);
        n->tag = (n->tag << 5) ^ (n->tag >> 3) ^ (uint32_t)iter;
    } else {
        *branch_score += 31u;
        delta_q = add_q(program_state, delta_q / 2, -coeff_q[(i + 2) & 15] / 9);
        n->tag = (n->tag >> 7) ^ (n->tag << 11) ^ UINT32_C(0x5bd1e995);
    }

    n->xq = add_q(program_state, n->xq, delta_q / 3);
    n->yq = add_q(program_state, n->yq, -delta_q / 4 + mul_q(program_state, left->xq, coeff_q[(iter + i + 6) & 15]) / 8);
    n->vf = clamp_float(program_state, n->vf * 0.84375f + fcorrection * 0.5f + (float)q_to_double(program_state, n->xq) * 0.03125f);
    n->vd = clamp_double(program_state, n->vd * 0.8125 + correction * 0.4375 + q_to_double(program_state, n->yq) * 0.046875);

    double probe = q_to_double(program_state, n->xq) * 0.75 + q_to_double(program_state, n->yq) * -0.5;
    probe += (double)n->vf * 0.25 + n->vd * 0.125;
    return quantize_double(program_state, probe);
}

int program_main(__private ProgramState *program_state) {
    init_nodes(program_state);

    uint64_t checksum = UINT64_C(0x66705f71736f6c76);
    uint64_t branch_score = 0;

    for (int iter = 0; iter < ITERS; iter++) {
        for (int i = 0; i < NODES; i++) {
            uint32_t q = solve_node(program_state, i, iter, &branch_score);
            uint64_t tag = ((uint64_t)(uint32_t)iter << 32) ^ (uint32_t)i;
            tag ^= (uint64_t)q ^ ((uint64_t)program_state->nodes[i].tag << ((i & 3) + 9));
            tag ^= (uint64_t)(uint32_t)program_state->nodes[i].xq << 17;
            checksum = mix64(program_state, rotl64(program_state, checksum, (unsigned)(i + 11)), tag);
        }
    }

    program_state->sink = checksum ^ rotl64(program_state, branch_score, 13);
    printf("fp_fixedpoint_hybrid_solver checksum=%016" PRIx64 " branches=%" PRIu64 "\n", program_state->sink, branch_score);
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)(((ulong)storage.sink >> (8u * i)) & 255u);
}
