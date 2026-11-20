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
 * mem_unaligned_packed.c
 *
 * Packed/unaligned memory access patterns.
 * Stresses unaligned loads/stores and packed layouts.
 *
 * Features exercised:
 *   - Packed structs with misaligned fields
 *   - Unaligned 64-bit accesses via packed fields
 *   - Volatile packed loads/stores
 *   - Mixed-size field access
 */





typedef struct __attribute__((packed)) {
    uint8_t tag;
    uint16_t len;
    uint32_t id;
    uint64_t payload;
} PackedRecord;

typedef struct __attribute__((packed, aligned(1))) {
    uint8_t pad;
    PackedRecord recs[8];
    uint16_t tail;
} PackedBlock;

typedef struct __attribute__((packed)) {
    uint8_t head;
    uint8_t bytes[7];
    uint64_t tail;
} PackedTail;

typedef struct {
    volatile uint64_t sink;
} ProgramState;


void init_block(__private ProgramState *program_state, volatile PackedBlock *block, uint32_t seed) {
    block->pad = (uint8_t)seed;
    block->tail = (uint16_t)(seed ^ 0x5a5au);

    for (int i = 0; i < 8; i++) {
        block->recs[i].tag = (uint8_t)(seed + i);
        block->recs[i].len = (uint16_t)((seed >> 1) + i * 3);
        block->recs[i].id = seed ^ (uint32_t)(i * 0x9e37u);
        block->recs[i].payload = ((uint64_t)block->recs[i].id << 32) |
                                 (uint64_t)(block->recs[i].len | (block->recs[i].tag << 16));
    }
}


uint64_t sum_packed(__private ProgramState *program_state, const volatile PackedBlock *block) {
    uint64_t sum = 0;

    sum += block->pad;
    sum += block->tail;

    for (int i = 0; i < 8; i++) {
        sum += block->recs[i].tag;
        sum += block->recs[i].len;
        sum += block->recs[i].id;
        sum ^= block->recs[i].payload;
        sum += block->recs[i].payload >> 32;
    }

    return sum;
}


uint64_t mix_tail(__private ProgramState *program_state, volatile PackedTail *tail, uint32_t seed) {
    tail->head = (uint8_t)(seed ^ 0xa5u);
    for (int i = 0; i < 7; i++) {
        tail->bytes[i] = (uint8_t)(seed + i * 11);
    }
    tail->tail = ((uint64_t)seed << 48) | ((uint64_t)seed << 16) | (uint64_t)seed;

    uint64_t sum = tail->head;
    for (int i = 0; i < 7; i++) {
        sum += tail->bytes[i];
    }
    sum ^= tail->tail;
    sum += tail->tail >> 17;
    return sum;
}

int program_main(__private ProgramState *program_state) {
    uint64_t result = 0;
    volatile PackedBlock block;
    volatile PackedTail tail;

    for (uint32_t i = 0; i < 1000; i++) {
        init_block(program_state, &block, i * 3u + 1u);
        result += sum_packed(program_state, &block);
        result += mix_tail(program_state, &tail, i ^ 0x1234u);
    }

    program_state->sink = result;
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)(((ulong)storage.sink >> (8u * i)) & 255u);
}
