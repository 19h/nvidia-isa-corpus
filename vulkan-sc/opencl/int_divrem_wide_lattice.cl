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
 * int_divrem_wide_lattice.c
 *
 * Fringe integer division/remainder lattice with mixed signed and unsigned
 * cells. Operands are generated inside conservative bounds to avoid signed
 * overflow, division by zero, and INT_MIN / -1.
 */





#define ROWS 17u
#define COLS 19u
#define ROUNDS 41u

typedef struct {
    int64_t signed_lane[4];
    uint64_t unsigned_lane[4];
    uint64_t tag;
} LatticeCell;






typedef struct {
    LatticeCell lattice[ROWS][COLS];
    uint64_t divisor_table[64];
    volatile uint64_t sink;
} ProgramState;

static uint64_t rotl64(__private ProgramState *program_state, uint64_t x, unsigned bits) {
    bits &= 63u;
    return bits == 0u ? x : ((x << bits) | (x >> (64u - bits)));
}

static uint64_t rotr64(__private ProgramState *program_state, uint64_t x, unsigned bits) {
    bits &= 63u;
    return bits == 0u ? x : ((x >> bits) | (x << (64u - bits)));
}

static uint64_t mix64(__private ProgramState *program_state, uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static int64_t signed_numerator(__private ProgramState *program_state, uint64_t x) {
    uint64_t bounded = x & ((UINT64_C(1) << 51) - 1u);
    return (int64_t)bounded - (INT64_C(1) << 50);
}

static int64_t signed_denominator(__private ProgramState *program_state, uint64_t x) {
    int64_t mag = (int64_t)((x & UINT64_C(0x1ffff)) + 3u);
    return (x & UINT64_C(0x20000)) != 0u ? -mag : mag;
}

static uint64_t unsigned_denominator(__private ProgramState *program_state, uint64_t x) {
    uint64_t d = ((x >> 7) ^ (x >> 29) ^ UINT64_C(0x9e3779b97f4a7c15));
    return ((d & UINT64_C(0x0000ffffffffffff)) | UINT64_C(1));
}

static uint64_t fold_signed_result(__private ProgramState *program_state, int64_t q, int64_t r, int64_t d, uint64_t salt) {
    int64_t abs_r = r < 0 ? -r : r;
    uint64_t uq = (uint64_t)q;
    uint64_t ur = (uint64_t)abs_r;
    uint64_t ud = (uint64_t)(d < 0 ? -d : d);

    return rotl64(program_state, uq ^ (ur * UINT64_C(0x100000001b3)), (unsigned)(salt & 31u) + 1u)
         ^ rotr64(program_state, ud + salt, (unsigned)((salt >> 9) & 31u) + 1u);
}


static uint64_t signed_divrem_cell(__private ProgramState *program_state, LatticeCell *cell, uint64_t seed, unsigned round) {
    uint64_t acc = cell->tag ^ seed;

    for (unsigned lane = 0; lane < 4u; lane++) {
        uint64_t basis = mix64(program_state, acc + program_state->divisor_table[(round + lane * 7u) & 63u]);
        int64_t n = signed_numerator(program_state, basis ^ (uint64_t)cell->signed_lane[lane]);
        int64_t d = signed_denominator(program_state, basis >> 11);
        int64_t q = n / d;
        int64_t r = n % d;
        int64_t rebuilt = q * d + r;

        if (rebuilt == n) {
            acc ^= fold_signed_result(program_state, q, r, d, basis);
            cell->signed_lane[lane] = q + (r ^ (int64_t)(round + lane));
        } else {
            acc ^= UINT64_C(0xdeaddeaddeaddead);
            cell->signed_lane[lane] = r;
        }

        if (((uint64_t)r & 3u) == (lane & 3u)) {
            acc += rotl64(program_state, (uint64_t)(q ^ r), lane + round + 1u);
        } else {
            acc ^= rotr64(program_state, (uint64_t)(q - r), lane * 5u + round + 3u);
        }
    }

    cell->tag = rotl64(program_state, cell->tag ^ acc ^ seed, (round % 23u) + 1u);
    return acc;
}


static uint64_t unsigned_divrem_cell(__private ProgramState *program_state, LatticeCell *cell, uint64_t seed, unsigned round) {
    uint64_t acc = seed ^ rotr64(program_state, cell->tag, round + 5u);

    for (unsigned lane = 0; lane < 4u; lane++) {
        uint64_t n = mix64(program_state, cell->unsigned_lane[lane] + acc + program_state->divisor_table[(lane * 13u + round) & 63u]);
        uint64_t d = unsigned_denominator(program_state, n ^ seed ^ ((uint64_t)lane << 32));
        uint64_t q = n / d;
        uint64_t r = n % d;
        uint64_t rebuilt = q * d + r;

        if (rebuilt == n) {
            acc += rotl64(program_state, q ^ program_state->divisor_table[(r + lane) & 63u], (unsigned)(r & 31u) + 1u);
            acc ^= rotr64(program_state, r + UINT64_C(0xa5a5a5a55a5a5a5a), lane + 9u);
        } else {
            acc ^= UINT64_C(0xfeedfacecafef00d);
        }

        if ((q ^ r ^ acc) & UINT64_C(0x80)) {
            cell->unsigned_lane[lane] = q + rotl64(program_state, r, lane + 1u);
        } else {
            cell->unsigned_lane[lane] = r ^ rotr64(program_state, q + acc, lane + 3u);
        }
    }

    cell->tag ^= rotl64(program_state, acc + seed + (uint64_t)round, (round & 15u) + 1u);
    return acc;
}

static void init_tables(__private ProgramState *program_state) {
    uint64_t x = UINT64_C(0x6a09e667f3bcc909);

    for (unsigned i = 0; i < 64u; i++) {
        x = mix64(program_state, x + (uint64_t)i * UINT64_C(0x9e3779b97f4a7c15));
        program_state->divisor_table[i] = (x | UINT64_C(1)) ^ ((uint64_t)(i + 3u) << 48);
    }
}

static void init_lattice(__private ProgramState *program_state) {
    for (unsigned r = 0; r < ROWS; r++) {
        for (unsigned c = 0; c < COLS; c++) {
            uint64_t seed = mix64(program_state, UINT64_C(0x123456789abcdef0) ^ ((uint64_t)r << 32) ^ c);
            LatticeCell *cell = &program_state->lattice[r][c];

            cell->tag = seed ^ program_state->divisor_table[(r * 11u + c) & 63u];
            for (unsigned lane = 0; lane < 4u; lane++) {
                uint64_t v = mix64(program_state, seed + (uint64_t)lane * UINT64_C(0x45d9f3b));
                cell->signed_lane[lane] = signed_numerator(program_state, v);
                cell->unsigned_lane[lane] = v ^ program_state->divisor_table[(c * 5u + lane) & 63u];
            }
        }
    }
}


static uint64_t lattice_round(__private ProgramState *program_state, unsigned round) {
    uint64_t wave = mix64(program_state, UINT64_C(0x3141592653589793) + round);

    for (unsigned r = 0; r < ROWS; r++) {
        for (unsigned c = 0; c < COLS; c++) {
            LatticeCell *cell = &program_state->lattice[r][c];
            uint64_t neighbor = program_state->lattice[(r + ROWS - 1u) % ROWS][(c + 3u) % COLS].tag;
            uint64_t seed = wave ^ neighbor ^ ((uint64_t)r << 40) ^ ((uint64_t)c << 8);

            if (((r + c + round) & 1u) == 0u) {
                wave ^= signed_divrem_cell(program_state, cell, seed, round);
                wave += unsigned_divrem_cell(program_state, cell, seed ^ wave, round + c);
            } else {
                wave += unsigned_divrem_cell(program_state, cell, seed, round);
                wave ^= signed_divrem_cell(program_state, cell, seed + wave, round + r);
            }

            if ((cell->tag & UINT64_C(0x100000000)) != 0u) {
                cell->tag += rotl64(program_state, wave, (r + c + round) & 31u);
            } else {
                cell->tag ^= rotr64(program_state, wave + program_state->divisor_table[(r + c) & 63u], (round & 31u) + 1u);
            }
        }
    }

    return wave;
}

static uint64_t checksum_lattice(__private ProgramState *program_state) {
    uint64_t h = UINT64_C(0xcbf29ce484222325);

    for (unsigned r = 0; r < ROWS; r++) {
        for (unsigned c = 0; c < COLS; c++) {
            const LatticeCell *cell = &program_state->lattice[r][c];

            h ^= cell->tag + ((uint64_t)r << 16) + c;
            h *= UINT64_C(0x100000001b3);
            for (unsigned lane = 0; lane < 4u; lane++) {
                h ^= (uint64_t)cell->signed_lane[lane] ^ rotl64(program_state, cell->unsigned_lane[lane], lane + 1u);
                h = rotl64(program_state, h, 11u) * UINT64_C(0xd6e8feb86659fd93);
            }
        }
    }

    return h;
}

int program_main(__private ProgramState *program_state) {
    uint64_t trace = UINT64_C(0x6d6f726f6b646976);

    init_tables(program_state);
    init_lattice(program_state);

    for (unsigned round = 0; round < ROUNDS; round++) {
        trace ^= lattice_round(program_state, round);
        trace = rotl64(program_state, trace + program_state->divisor_table[round & 63u], (round % 37u) + 1u);
    }

    program_state->sink = checksum_lattice(program_state) ^ trace;
    printf("int_divrem_wide_lattice checksum=%016" PRIx64 " trace=%016" PRIx64 "\n", program_state->sink, trace);
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)(((ulong)storage.sink >> (8u * i)) & 255u);
}
