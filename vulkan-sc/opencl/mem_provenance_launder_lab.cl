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
 * mem_provenance_launder_lab.c
 *
 * Standards-conscious pointer provenance and alias laundering stress.
 * Pointer values move through memcpy-backed capsules and legal object/array
 * navigation; uintptr_t is used only for relative opaque decisions/checksums.
 */










#define CELL_COUNT 40u
#define WALK_COUNT 240u
#define BOX_COUNT 17u

typedef struct {
    uint32_t lane[4];
    uint8_t bytes[23];
    uint16_t next[3];
    uint8_t tag;
    uint8_t live;
    uint64_t stamp;
} Cell;

typedef struct {
    uint8_t raw[sizeof(Cell *)];
    uint32_t salt;
} PointerBox;

typedef struct {
    Cell cells[CELL_COUNT];
    PointerBox boxes[BOX_COUNT];
    uint8_t scratch[sizeof(Cell) * 2u + 13u];
    uint32_t visits[CELL_COUNT];
} Lab;

typedef struct {
    volatile uint64_t sink_checksum;
    volatile uint32_t sink_visit_total;
    volatile uint32_t sink_live_total;
} ProgramState;

static uint32_t rotl32(__private ProgramState *program_state, uint32_t x, unsigned n) {
    n &= 31u;
    return n == 0u ? x : ((x << n) | (x >> (32u - n)));
}

static uint64_t rotl64(__private ProgramState *program_state, uint64_t x, unsigned n) {
    n &= 63u;
    return n == 0u ? x : ((x << n) | (x >> (64u - n)));
}

static uint32_t mix32(__private ProgramState *program_state, uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

static uint32_t load32_bytes(__private ProgramState *program_state, const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static void store32_bytes(__private ProgramState *program_state, uint8_t *p, uint32_t v) {
    memcpy(p, &v, sizeof(v));
}

static void box_store(__private ProgramState *program_state, PointerBox *box, Cell *ptr, uint32_t salt) {
    memcpy(box->raw, &ptr, sizeof(ptr));
    box->salt = salt;
}

static Cell *box_load(__private ProgramState *program_state, const PointerBox *box) {
    Cell *ptr;
    memcpy(&ptr, box->raw, sizeof(ptr));
    return ptr;
}

static uintptr_t cell_token(__private ProgramState *program_state, const Lab *lab, const Cell *ptr) {
    ptrdiff_t index = ptr - &lab->cells[0];
    return (uintptr_t)(size_t)index;
}

static uint64_t checksum_bytes(__private ProgramState *program_state, const void *data, size_t len, uint64_t seed) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = seed ^ 0x6a09e667f3bcc908ull;

    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t)p[i] + ((uint64_t)i << 21);
        h = rotl64(program_state, h, 9) * 0x100000001b3ull;
        h ^= h >> 27;
    }

    return h;
}

static void init_cell(__private ProgramState *program_state, Cell *cell, uint32_t idx, uint32_t seed) {
    memset(cell, 0, sizeof(*cell));

    for (uint32_t i = 0; i < 4u; i++) {
        cell->lane[i] = mix32(program_state, seed + idx * 97u + i * 0x45d9f3bu);
    }

    for (uint32_t i = 0; i < sizeof(cell->bytes); i++) {
        uint32_t v = mix32(program_state, seed ^ (idx * 131u) ^ (i * 23u));
        cell->bytes[i] = (uint8_t)(v ^ (v >> 11));
    }

    for (uint32_t i = 0; i < 3u; i++) {
        cell->next[i] = (uint16_t)((idx * (i + 5u) + seed + i * 7u) % CELL_COUNT);
    }

    cell->tag = (uint8_t)((seed + idx) & 3u);
    cell->live = (uint8_t)((idx % 7u) != 0u);
    cell->stamp = ((uint64_t)cell->lane[0] << 32) ^ cell->lane[3] ^ seed;
}


static uint64_t alias_fold(__private ProgramState *program_state, Cell *a, Cell *b, uint32_t step, uintptr_t token) {
    uint64_t acc = ((uint64_t)a->lane[0] << 19) ^ b->stamp ^ (uint64_t)token;
    uint32_t lane = step & 3u;

    a->lane[lane] += b->lane[(lane + 1u) & 3u] ^ step;
    b->lane[(lane + 2u) & 3u] ^= rotl32(program_state, a->lane[lane], (step % 17u) + 1u);
    a->stamp = rotl64(program_state, a->stamp + b->lane[lane] + step, (step % 29u) + 1u);
    b->stamp ^= rotl64(program_state, a->stamp ^ token, (step % 23u) + 3u);

    if (((uint32_t)token ^ a->tag ^ step) & 1u) {
        uint32_t off = (step * 5u + a->tag) % (uint32_t)(sizeof(a->bytes) - sizeof(uint32_t));
        uint32_t v = load32_bytes(program_state, a->bytes + off);
        v ^= rotl32(program_state, b->lane[(step + 1u) & 3u] + (uint32_t)token, step);
        store32_bytes(program_state, a->bytes + off, v);
        acc ^= v;
    } else {
        uint8_t carry = (uint8_t)(step + b->tag + (uint32_t)token);
        for (size_t i = 0; i < sizeof(b->bytes); i++) {
            uint8_t old = b->bytes[i];
            b->bytes[i] = (uint8_t)(old + carry + (uint8_t)i);
            carry ^= old;
        }
        acc += carry;
    }

    a->next[(step + a->tag) % 3u] = (uint16_t)((a->next[step % 3u] + b->lane[0] + step) % CELL_COUNT);
    b->live = (uint8_t)((b->live ^ (uint8_t)(acc >> 7)) & 1u);
    return acc ^ checksum_bytes(program_state, a, sizeof(*a), b->stamp + step);
}


static uint64_t launder_pair(__private ProgramState *program_state, Lab *lab, uint32_t box_index, uint32_t step) {
    PointerBox *box = &lab->boxes[box_index % BOX_COUNT];
    Cell *first = box_load(program_state, box);
    uintptr_t first_token = cell_token(program_state, lab, first);
    uint32_t next_index = first->next[(box->salt + step + (uint32_t)first_token) % 3u] % CELL_COUNT;
    Cell *second = &lab->cells[next_index];
    uintptr_t second_token = cell_token(program_state, lab, second);
    Cell local_a;
    Cell local_b;
    uint64_t acc;

    memcpy(&local_a, first, sizeof(local_a));
    memcpy(&local_b, second, sizeof(local_b));

    memset(lab->scratch, 0, sizeof(lab->scratch));
    memcpy(lab->scratch + 3u, &local_a, sizeof(local_a));
    memcpy(lab->scratch + 11u + sizeof(local_a), &local_b, sizeof(local_b));

    for (size_t off = 3u; off + sizeof(uint32_t) <= 3u + sizeof(local_a); off += 7u) {
        uint32_t v = load32_bytes(program_state, lab->scratch + off);
        v ^= mix32(program_state, v + (uint32_t)off + step + (uint32_t)first_token);
        store32_bytes(program_state, lab->scratch + off, v);
    }

    memcpy(&local_a, lab->scratch + 3u, sizeof(local_a));
    memcpy(&local_b, lab->scratch + 11u + sizeof(local_a), sizeof(local_b));
    local_a.tag &= 3u;
    local_b.tag &= 3u;
    local_a.live &= 1u;
    local_b.live &= 1u;

    if (((first_token ^ second_token ^ (uintptr_t)step) & 3u) == 0u) {
        acc = alias_fold(program_state, &local_a, &local_a, step, first_token);
    } else {
        acc = alias_fold(program_state, &local_a, &local_b, step, second_token);
    }

    memcpy(first, &local_a, sizeof(*first));
    if (first != second) {
        memcpy(second, &local_b, sizeof(*second));
    }

    box_store(program_state, box, &lab->cells[(next_index + (uint32_t)acc + step) % CELL_COUNT], box->salt + step + 13u);
    return acc ^ ((uint64_t)first_token << 32) ^ second_token;
}

static uint64_t walk_lab(__private ProgramState *program_state, Lab *lab) {
    uint64_t result = 0x243f6a8885a308d3ull;

    for (uint32_t step = 0; step < WALK_COUNT; step++) {
        uint32_t box = (step * 9u + (uint32_t)result) % BOX_COUNT;
        Cell *ptr = box_load(program_state, &lab->boxes[box]);
        uintptr_t token = cell_token(program_state, lab, ptr);

        lab->visits[(uint32_t)token % CELL_COUNT]++;
        result ^= launder_pair(program_state, lab, box, step);
        result = rotl64(program_state, result + lab->visits[(uint32_t)token] + token, (step % 31u) + 1u);

        if (((uint32_t)result ^ (uint32_t)token ^ step) & 4u) {
            uint32_t idx = (uint32_t)(token + ptr->next[(step + ptr->tag) % 3u]) % CELL_COUNT;
            box_store(program_state, &lab->boxes[(box + 5u) % BOX_COUNT], &lab->cells[idx], (uint32_t)result ^ step);
        }
    }

    return result;
}

int program_main(__private ProgramState *program_state) {
    Lab lab;
    uint64_t checksum;
    uint32_t visit_total = 0;
    uint32_t live_total = 0;

    memset(&lab, 0, sizeof(lab));

    for (uint32_t i = 0; i < CELL_COUNT; i++) {
        init_cell(program_state, &lab.cells[i], i, 0x9e3779b9u + i * 0x1020304u);
    }

    for (uint32_t i = 0; i < BOX_COUNT; i++) {
        uint32_t idx = (i * 11u + 3u) % CELL_COUNT;
        box_store(program_state, &lab.boxes[i], &lab.cells[idx], mix32(program_state, i + 0x31415926u));
    }

    checksum = walk_lab(program_state, &lab);

    for (uint32_t i = 0; i < CELL_COUNT; i++) {
        visit_total += lab.visits[i];
        live_total += lab.cells[i].live;
        checksum ^= checksum_bytes(program_state, &lab.cells[i], sizeof(lab.cells[i]), (uint64_t)i * 0x94d049bb133111ebull);
    }

    program_state->sink_checksum = checksum;
    program_state->sink_visit_total = visit_total;
    program_state->sink_live_total = live_total;
    printf("mem_provenance_launder_lab: %08x %08x %016llx\n",
           (unsigned)visit_total, (unsigned)live_total, (unsigned long long)checksum);
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink_checksum); ++i)
        results[1 + i] = (uint)(((ulong)storage.sink_checksum >> (8u * i)) & 255u);
    for (uint i = 0; i < sizeof(storage.sink_visit_total); ++i)
        results[17 + i] = (uint)(((uint)storage.sink_visit_total >> (8u * i)) & 255u);
    for (uint i = 0; i < sizeof(storage.sink_live_total); ++i)
        results[33 + i] = (uint)(((uint)storage.sink_live_total >> (8u * i)) & 255u);
}
