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
 * int_carryless_crc_mesh.c
 *
 * Carryless GF(2)-style multiplication, CRC-like folding, and a bit-matrix
 * mesh. This is deterministic, bounded, and uses only integer operations.
 */





#define MESH_N 16u
#define ROUNDS 96u
#define GF_POLY UINT64_C(0x1b)

typedef struct {
    uint64_t lo;
    uint64_t hi;
} ClProduct;

typedef struct {
    uint64_t row[MESH_N];
} BitMatrix;







typedef struct {
    uint64_t crc_table[256];
    uint64_t mesh[MESH_N][MESH_N];
    BitMatrix matrix_bank[4];
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
    x ^= x >> 33;
    x *= UINT64_C(0xff51afd7ed558ccd);
    x ^= x >> 29;
    x *= UINT64_C(0xc4ceb9fe1a85ec53);
    return x ^ (x >> 32);
}

static uint64_t parity64(__private ProgramState *program_state, uint64_t x) {
    x ^= x >> 32;
    x ^= x >> 16;
    x ^= x >> 8;
    x ^= x >> 4;
    return (UINT64_C(0x6996) >> (x & 15u)) & 1u;
}

static ClProduct clmul64(__private ProgramState *program_state, uint64_t a, uint64_t b) {
    ClProduct p = { 0u, 0u };

    for (unsigned bit = 0; bit < 64u; bit++) {
        if (((b >> bit) & 1u) != 0u) {
            if (bit == 0u) {
                p.lo ^= a;
            } else {
                p.lo ^= a << bit;
                p.hi ^= a >> (64u - bit);
            }
        }
    }

    return p;
}

static void toggle_product_bit(__private ProgramState *program_state, ClProduct *p, unsigned bit) {
    if (bit < 64u) {
        p->lo ^= UINT64_C(1) << bit;
    } else {
        p->hi ^= UINT64_C(1) << (bit - 64u);
    }
}

static uint64_t gf_reduce64(__private ProgramState *program_state, ClProduct p) {
    for (int bit = 127; bit >= 64; bit--) {
        unsigned high_bit = (unsigned)bit - 64u;

        if (((p.hi >> high_bit) & 1u) != 0u) {
            unsigned base = (unsigned)bit - 64u;

            toggle_product_bit(program_state, &p, (unsigned)bit);
            toggle_product_bit(program_state, &p, base);
            toggle_product_bit(program_state, &p, base + 1u);
            toggle_product_bit(program_state, &p, base + 3u);
            toggle_product_bit(program_state, &p, base + 4u);
        }
    }

    return p.lo;
}


static uint64_t gf_mul64(__private ProgramState *program_state, uint64_t a, uint64_t b) {
    return gf_reduce64(program_state, clmul64(program_state, a, b));
}

static uint64_t crc64_fold(__private ProgramState *program_state, uint64_t crc, uint64_t word) {
    crc ^= word;
    for (unsigned i = 0; i < 8u; i++) {
        uint8_t idx = (uint8_t)crc;
        crc = (crc >> 8) ^ program_state->crc_table[idx] ^ rotl64(program_state, word, i * 7u + 3u);
        word = rotr64(program_state, word ^ crc, 11u);
    }
    return crc;
}

static uint64_t matrix_apply(__private ProgramState *program_state, const BitMatrix *matrix, uint64_t v) {
    uint64_t out = 0u;

    for (unsigned row = 0; row < MESH_N; row++) {
        uint64_t bit = parity64(program_state, matrix->row[row] & v);
        out |= bit << row;
    }

    return out | (v & UINT64_C(0xffffffffffff0000));
}

static void init_crc_table(__private ProgramState *program_state) {
    for (unsigned i = 0; i < 256u; i++) {
        uint64_t crc = i;

        for (unsigned bit = 0; bit < 8u; bit++) {
            uint64_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (UINT64_C(0xc96c5795d7870f42) & mask);
        }

        program_state->crc_table[i] = crc;
    }
}

static void init_matrices(__private ProgramState *program_state) {
    uint64_t seed = UINT64_C(0x243f6a8885a308d3);

    for (unsigned bank = 0; bank < 4u; bank++) {
        for (unsigned row = 0; row < MESH_N; row++) {
            seed = mix64(program_state, seed + ((uint64_t)bank << 24) + row);
            program_state->matrix_bank[bank].row[row] = (seed | (UINT64_C(1) << row)) & UINT64_C(0xffff);
        }
    }
}

static void init_mesh(__private ProgramState *program_state) {
    uint64_t seed = UINT64_C(0x13198a2e03707344);

    for (unsigned r = 0; r < MESH_N; r++) {
        for (unsigned c = 0; c < MESH_N; c++) {
            seed = mix64(program_state, seed ^ ((uint64_t)r << 32) ^ c);
            program_state->mesh[r][c] = seed ^ rotl64(program_state, UINT64_C(0x9e3779b97f4a7c15), r + c + 1u);
        }
    }
}


static uint64_t mesh_round(__private ProgramState *program_state, unsigned round, uint64_t carry) {
    uint64_t diag = carry ^ program_state->crc_table[(round * 29u) & 255u];

    for (unsigned r = 0; r < MESH_N; r++) {
        uint64_t left = program_state->mesh[r][(round + r) & (MESH_N - 1u)];
        uint64_t above = program_state->mesh[(r + MESH_N - 1u) & (MESH_N - 1u)][r];
        uint64_t factor = matrix_apply(program_state, &program_state->matrix_bank[(round + r) & 3u], left ^ above ^ diag);
        uint64_t product = gf_mul64(program_state, left ^ carry, factor | UINT64_C(1));
        uint64_t folded = crc64_fold(program_state, diag ^ product, above + ((uint64_t)r << 48));

        for (unsigned c = 0; c < MESH_N; c++) {
            uint64_t old = program_state->mesh[r][c];
            uint64_t neighbor = program_state->mesh[(r + c + 3u) & (MESH_N - 1u)][(c + 5u) & (MESH_N - 1u)];
            uint64_t gate = gf_mul64(program_state, old ^ folded, neighbor | UINT64_C(1));

            if (((gate >> ((c + r) & 31u)) & 1u) != 0u) {
                program_state->mesh[r][c] = crc64_fold(program_state, gate ^ product, old) ^ rotl64(program_state, factor, c + 1u);
            } else if ((folded & (UINT64_C(1) << (c & 15u))) != 0u) {
                program_state->mesh[r][c] = gf_mul64(program_state, old ^ diag, folded | UINT64_C(1)) + rotr64(program_state, neighbor, r + 7u);
            } else {
                program_state->mesh[r][c] = old ^ gate ^ rotl64(program_state, neighbor + carry, r + c + 3u);
            }

            diag ^= matrix_apply(program_state, &program_state->matrix_bank[c & 3u], program_state->mesh[r][c]);
            diag = crc64_fold(program_state, diag, program_state->mesh[r][c] ^ ((uint64_t)c << 56));
        }
    }

    return diag;
}

static uint64_t checksum_mesh(__private ProgramState *program_state) {
    uint64_t h = UINT64_C(0xcbf29ce484222325);

    for (unsigned r = 0; r < MESH_N; r++) {
        for (unsigned c = 0; c < MESH_N; c++) {
            uint64_t v = program_state->mesh[r][c] ^ ((uint64_t)r << 40) ^ ((uint64_t)c << 8);
            h = crc64_fold(program_state, h, v);
            h ^= gf_mul64(program_state, v | UINT64_C(1), h ^ UINT64_C(0xd6e8feb86659fd93));
        }
    }

    return h;
}

int program_main(__private ProgramState *program_state) {
    uint64_t trace = UINT64_C(0x6d6f726f6b676632);

    init_crc_table(program_state);
    init_matrices(program_state);
    init_mesh(program_state);

    for (unsigned round = 0; round < ROUNDS; round++) {
        trace ^= mesh_round(program_state, round, trace + program_state->crc_table[round & 255u]);
        trace = rotl64(program_state, trace ^ UINT64_C(0xa5a5a5a55a5a5a5a), (round & 31u) + 1u);
    }

    program_state->sink = checksum_mesh(program_state) ^ trace;
    printf("int_carryless_crc_mesh checksum=%016" PRIx64 " trace=%016" PRIx64 "\n", program_state->sink, trace);
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)(((ulong)storage.sink >> (8u * i)) & 255u);
}
