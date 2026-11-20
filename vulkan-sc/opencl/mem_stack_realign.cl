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
 * mem_stack_realign.c
 *
 * Stack alignment and dynamic allocation patterns.
 * Stresses VLAs, alloca, and over-aligned locals.
 *
 * Features exercised:
 *   - Variable-length arrays
 *   - __builtin_alloca with manual alignment
 *   - Over-aligned local buffers
 *   - Mixed stack frame sizes
 */





typedef struct {
    volatile uint64_t sink;
} ProgramState;


uint64_t stack_mix(__private ProgramState *program_state, uint32_t seed, uint32_t n) {
    _Alignas(32) uint8_t fixed[64];
    for (uint32_t i = 0; i < 64; i++) {
        fixed[i] = (uint8_t)(seed + i * 3u);
    }

    uint32_t len = (n & 0x3fu) + 1u;
    uint32_t vla[64];
    for (uint32_t i = 0; i < len; i++) {
        vla[i] = seed ^ (uint32_t)(i * 0x9e37u);
    }

    uint32_t dyn_size = len * 3u + 17u;
    _Alignas(32) uint8_t aligned[64 * 3 + 17];

    for (uint32_t i = 0; i < dyn_size; i++) {
        aligned[i] = (uint8_t)(fixed[i & 63u] ^ (uint8_t)vla[i % len]);
    }

    uint64_t sum = 0;
    for (uint32_t i = 0; i < len; i++) {
        sum += vla[i];
    }
    for (uint32_t i = 0; i < dyn_size; i += 5u) {
        sum += aligned[i];
    }
    sum ^= (uint64_t)aligned[0] << 56;

    return sum;
}

int program_main(__private ProgramState *program_state) {
    uint64_t result = 0;

    for (uint32_t i = 1; i < 2000; i++) {
        result += stack_mix(program_state, i * 7u + 3u, i);
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
