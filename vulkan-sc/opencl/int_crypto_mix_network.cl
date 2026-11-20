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
 * int_crypto_mix_network.c
 *
 * Deterministic integer mixing network with generated tables, rotates,
 * checksums, branches, memcpy type movement, and call-heavy round functions.
 */









#define TABLE_SIZE 256u
#define BLOCK_COUNT 24u
#define STATE_WORDS 8u
#define ROUNDS 96u

typedef struct {
    uint32_t lane[STATE_WORDS];
    uint8_t bytes[32];
    uint64_t tag;
} MixBlock;

typedef struct {
    uint32_t round_key[ROUNDS];
    uint32_t sbox[TABLE_SIZE];
    uint8_t perm[TABLE_SIZE];
    uint64_t tweak[16];
} NetworkTables;




typedef struct {
    volatile uint64_t sink_checksum;
    volatile uint32_t sink_summary;
    NetworkTables tables;
    MixBlock blocks[BLOCK_COUNT];
} ProgramState;

static uint32_t rotl32(__private ProgramState *program_state, uint32_t x, unsigned n) {
    n &= 31u;
    return n == 0u ? x : ((x << n) | (x >> (32u - n)));
}

static uint32_t rotr32(__private ProgramState *program_state, uint32_t x, unsigned n) {
    n &= 31u;
    return n == 0u ? x : ((x >> n) | (x << (32u - n)));
}

static uint64_t rotl64(__private ProgramState *program_state, uint64_t x, unsigned n) {
    n &= 63u;
    return n == 0u ? x : ((x << n) | (x >> (64u - n)));
}

static uint32_t load32_le(__private ProgramState *program_state, const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static void store32_le(__private ProgramState *program_state, uint8_t *p, uint32_t v) {
    memcpy(p, &v, sizeof(v));
}

static uint32_t splitmix32(__private ProgramState *program_state, uint32_t x) {
    x += 0x9e3779b9u;
    x = (x ^ (x >> 16)) * 0x85ebca6bu;
    x = (x ^ (x >> 13)) * 0xc2b2ae35u;
    return x ^ (x >> 16);
}

static void init_tables(__private ProgramState *program_state) {
    uint32_t x = 0x243f6a88u;

    for (uint32_t i = 0; i < TABLE_SIZE; i++) {
        x = splitmix32(program_state, x + i * 0x45d9f3bu);
        program_state->tables.sbox[i] = rotl32(program_state, x ^ (i * 0x01010101u), (i % 29u) + 1u);
        program_state->tables.perm[i] = (uint8_t)((i * 73u + (x >> 24) + (x >> 11)) & 255u);
    }

    for (uint32_t i = 0; i < ROUNDS; i++) {
        x = splitmix32(program_state, x ^ i ^ program_state->tables.sbox[(i * 17u) & 255u]);
        program_state->tables.round_key[i] = x ^ rotl32(program_state, program_state->tables.sbox[(x >> 24) & 255u], i);
    }

    for (uint32_t i = 0; i < 16u; i++) {
        uint64_t hi = splitmix32(program_state, x + i * 0x1020304u);
        uint64_t lo = splitmix32(program_state, (uint32_t)hi ^ program_state->tables.round_key[(i * 5u) % ROUNDS]);
        program_state->tables.tweak[i] = (hi << 32) | lo;
    }
}

static uint32_t table_mix(__private ProgramState *program_state, uint32_t x, uint32_t key, uint32_t round) {
    uint8_t a = (uint8_t)x;
    uint8_t b = (uint8_t)(x >> 8);
    uint8_t c = (uint8_t)(x >> 16);
    uint8_t d = (uint8_t)(x >> 24);
    uint32_t y = program_state->tables.sbox[a] ^ rotl32(program_state, program_state->tables.sbox[b], 7);

    y += rotr32(program_state, program_state->tables.sbox[c], 11) ^ key;
    y ^= rotl32(program_state, program_state->tables.sbox[program_state->tables.perm[d]], (round % 17u) + 3u);
    y += (uint32_t)program_state->tables.perm[(a + c + round) & 255u] * 0x01010101u;
    return y ^ rotl32(program_state, x + key, (d & 15u) + 1u);
}


static uint32_t branch_scramble(__private ProgramState *program_state, uint32_t x, uint32_t y, uint32_t key, uint32_t round) {
    uint32_t r = table_mix(program_state, x ^ y, key, round);

    if ((r & 0x80000000u) != 0u) {
        r = rotl32(program_state, r + y, 9) ^ (x * 0x7feb352du);
    } else if ((r & 0x00010000u) != 0u) {
        r = rotr32(program_state, r ^ x, 5) + (y * 0x846ca68bu);
    } else {
        r ^= rotl32(program_state, x + y + key, (round & 15u) + 1u);
        r += 0x6a09e667u ^ round;
    }

    return r ^ (r >> 16);
}


static void bytes_to_lanes(__private ProgramState *program_state, MixBlock *block) {
    for (uint32_t i = 0; i < STATE_WORDS; i++) {
        block->lane[i] ^= load32_le(program_state, block->bytes + i * 4u);
        block->lane[i] = rotl32(program_state, block->lane[i], (i * 3u + 5u) & 31u);
    }
}


static void lanes_to_bytes(__private ProgramState *program_state, MixBlock *block, uint32_t salt) {
    for (uint32_t i = 0; i < STATE_WORDS; i++) {
        uint32_t v = block->lane[(i + salt) % STATE_WORDS] ^ program_state->tables.round_key[(salt + i) % ROUNDS];
        store32_le(program_state, block->bytes + i * 4u, v);
    }
}


static uint64_t block_checksum(__private ProgramState *program_state, const MixBlock *block, uint64_t seed) {
    uint64_t h = seed ^ block->tag;

    for (uint32_t i = 0; i < STATE_WORDS; i++) {
        h ^= (uint64_t)block->lane[i] << ((i & 1u) ? 17u : 3u);
        h = rotl64(program_state, h, (i * 9u + 7u) & 63u) * 0x100000001b3ull;
    }

    for (uint32_t i = 0; i < sizeof(block->bytes); i++) {
        h += block->bytes[i] ^ (uint8_t)i;
        h ^= h >> 23;
        h = rotl64(program_state, h, 13);
    }

    return h;
}

static MixBlock make_block(__private ProgramState *program_state, uint32_t idx) {
    MixBlock block;
    memset(&block, 0, sizeof(block));

    for (uint32_t i = 0; i < STATE_WORDS; i++) {
        block.lane[i] = splitmix32(program_state, 0x13198a2eu + idx * 97u + i * 131u);
    }

    for (uint32_t i = 0; i < sizeof(block.bytes); i++) {
        uint32_t v = splitmix32(program_state, idx * 0x9e37u + i * 0x45d9u);
        block.bytes[i] = (uint8_t)(v ^ (v >> 11) ^ (idx * 19u));
    }

    block.tag = ((uint64_t)block.lane[0] << 32) | block.lane[STATE_WORDS - 1u];
    block.tag ^= program_state->tables.tweak[idx & 15u];
    return block;
}


static void network_round(__private ProgramState *program_state, MixBlock *block, uint32_t round) {
    uint32_t carry = program_state->tables.round_key[round];
    uint32_t old[STATE_WORDS];

    memcpy(old, block->lane, sizeof(old));

    for (uint32_t i = 0; i < STATE_WORDS; i++) {
        uint32_t left = old[i];
        uint32_t right = old[(i + 1u) % STATE_WORDS];
        uint32_t far = old[(i + 5u) % STATE_WORDS];
        uint32_t mixed = branch_scramble(program_state, left + carry, right ^ far, program_state->tables.round_key[(round + i) % ROUNDS], round + i);

        block->lane[i] = old[(i + 3u) % STATE_WORDS] ^ mixed ^ rotl32(program_state, carry, i + 1u);
        carry += mixed ^ program_state->tables.sbox[(block->lane[i] >> 24) & 255u];
    }

    if ((carry & 3u) == 0u) {
        lanes_to_bytes(program_state, block, round & 7u);
    } else if ((carry & 3u) == 1u) {
        bytes_to_lanes(program_state, block);
    } else {
        uint32_t pos = (carry >> 5) & 31u;
        block->bytes[pos] ^= (uint8_t)(carry + round);
        block->bytes[(pos + 11u) & 31u] += (uint8_t)(carry >> 16);
    }

    block->tag ^= ((uint64_t)carry << 32) | block->lane[round % STATE_WORDS];
    block->tag = rotl64(program_state, block->tag + program_state->tables.tweak[round & 15u], (round % 31u) + 1u);
}


static uint64_t process_pair(__private ProgramState *program_state, MixBlock *a, MixBlock *b, uint32_t outer) {
    uint8_t image[sizeof(MixBlock) * 2u + 5u];
    uint64_t acc;

    memset(image, 0, sizeof(image));
    memcpy(image + 2u, a, sizeof(*a));
    memcpy(image + 5u + sizeof(*a), b, sizeof(*b));

    for (uint32_t i = 0; i < 9u; i++) {
        uint32_t off = (i * 13u + outer) % (uint32_t)(sizeof(MixBlock) - sizeof(uint32_t));
        uint32_t v = load32_le(program_state, image + 2u + off);
        v ^= table_mix(program_state, v + outer, program_state->tables.round_key[(outer + i) % ROUNDS], outer + i);
        store32_le(program_state, image + 2u + off, v);
    }

    memcpy(a, image + 2u, sizeof(*a));
    memcpy(b, image + 5u + sizeof(*a), sizeof(*b));

    for (uint32_t r = 0; r < ROUNDS; r++) {
        network_round(program_state, a, r);
        if ((r & 7u) == 3u) {
            network_round(program_state, b, (r + outer) % ROUNDS);
            a->lane[(r + outer) % STATE_WORDS] ^= b->lane[(r + 3u) % STATE_WORDS];
        }
    }

    acc = block_checksum(program_state, a, 0xcbf29ce484222325ull ^ outer);
    acc ^= block_checksum(program_state, b, acc + 0x9e3779b97f4a7c15ull);
    return acc;
}

int program_main(__private ProgramState *program_state) {
    uint64_t checksum = 0x6a09e667f3bcc908ull;
    uint32_t summary = 0x510e527fu;

    init_tables(program_state);

    for (uint32_t i = 0; i < BLOCK_COUNT; i++) {
        program_state->blocks[i] = make_block(program_state, i);
    }

    for (uint32_t outer = 0; outer < 37u; outer++) {
        uint32_t a = (outer * 7u + summary) % BLOCK_COUNT;
        uint32_t b = (outer * 11u + (uint32_t)(checksum >> 32)) % BLOCK_COUNT;

        checksum ^= process_pair(program_state, &program_state->blocks[a], (a == b) ? &program_state->blocks[a] : &program_state->blocks[b], outer);
        checksum = rotl64(program_state, checksum + program_state->tables.tweak[outer & 15u], (outer % 19u) + 5u);
        summary ^= (uint32_t)checksum ^ program_state->blocks[a].lane[outer % STATE_WORDS];
        summary = rotl32(program_state, summary + program_state->tables.round_key[outer % ROUNDS], (outer & 15u) + 1u);
    }

    for (uint32_t i = 0; i < BLOCK_COUNT; i++) {
        checksum ^= block_checksum(program_state, &program_state->blocks[i], (uint64_t)i * 0x94d049bb133111ebull);
        summary += program_state->blocks[i].lane[i % STATE_WORDS] ^ (uint32_t)program_state->blocks[i].tag;
    }

    program_state->sink_checksum = checksum;
    program_state->sink_summary = summary;
    printf("int_crypto_mix_network: %08x %016llx\n",
           (unsigned)summary, (unsigned long long)checksum);
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink_checksum); ++i)
        results[1 + i] = (uint)(((ulong)storage.sink_checksum >> (8u * i)) & 255u);
    for (uint i = 0; i < sizeof(storage.sink_summary); ++i)
        results[17 + i] = (uint)(((uint)storage.sink_summary >> (8u * i)) & 255u);
}
