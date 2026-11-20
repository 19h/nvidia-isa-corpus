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
 * Population Count and Bit Manipulation
 *
 * Counts set bits, leading/trailing zeros.
 * Uses both software and hardware (if available) implementations.
 *
 * Features exercised:
 *   - POPCNT instruction (x86, ARM)
 *   - CLZ/CTZ instructions
 *   - Bit manipulation patterns
 */





/* Software popcount - naive */
typedef struct {
    volatile uint32_t sink;
} ProgramState;

static inline int popcount_naive(__private ProgramState *program_state, uint32_t x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

/* Software popcount - Brian Kernighan's algorithm */
static inline int popcount_kernighan(__private ProgramState *program_state, uint32_t x) {
    int count = 0;
    while (x) {
        x &= x - 1;  /* Clear least significant set bit */
        count++;
    }
    return count;
}

/* Software popcount - parallel */
static inline int popcount_parallel(__private ProgramState *program_state, uint32_t x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    x = x + (x >> 8);
    x = x + (x >> 16);
    return x & 0x3F;
}

/* Software popcount - lookup table */
__constant static const uint8_t popcount_table[256] = {
    0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8
};

static inline int popcount_lookup(__private ProgramState *program_state, uint32_t x) {
    return popcount_table[x & 0xFF]
         + popcount_table[(x >> 8) & 0xFF]
         + popcount_table[(x >> 16) & 0xFF]
         + popcount_table[(x >> 24) & 0xFF];
}

/* 64-bit popcount */
static inline int popcount64_parallel(__private ProgramState *program_state, uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    x = x + (x >> 8);
    x = x + (x >> 16);
    x = x + (x >> 32);
    return (int)(x & 0x7F);
}

/* Count leading zeros (software) */
static inline int clz_soft(__private ProgramState *program_state, uint32_t x) {
    if (x == 0) return 32;

    int n = 0;
    if ((x & 0xFFFF0000) == 0) { n += 16; x <<= 16; }
    if ((x & 0xFF000000) == 0) { n += 8; x <<= 8; }
    if ((x & 0xF0000000) == 0) { n += 4; x <<= 4; }
    if ((x & 0xC0000000) == 0) { n += 2; x <<= 2; }
    if ((x & 0x80000000) == 0) { n += 1; }
    return n;
}

/* Count trailing zeros (software) */
static inline int ctz_soft(__private ProgramState *program_state, uint32_t x) {
    if (x == 0) return 32;

    int n = 0;
    if ((x & 0x0000FFFF) == 0) { n += 16; x >>= 16; }
    if ((x & 0x000000FF) == 0) { n += 8; x >>= 8; }
    if ((x & 0x0000000F) == 0) { n += 4; x >>= 4; }
    if ((x & 0x00000003) == 0) { n += 2; x >>= 2; }
    if ((x & 0x00000001) == 0) { n += 1; }
    return n;
}

/* Find first set bit (1-indexed, 0 if none) */
static inline int ffs_soft(__private ProgramState *program_state, uint32_t x) {
    if (x == 0) return 0;
    return ctz_soft(program_state, x) + 1;
}

/* Bit reversal */
static inline uint32_t bit_reverse(__private ProgramState *program_state, uint32_t x) {
    x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1);
    x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2);
    x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4);
    x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8);
    x = (x << 16) | (x >> 16);
    return x;
}

/* Byte swap */
static inline uint32_t byte_swap(__private ProgramState *program_state, uint32_t x) {
    return ((x >> 24) & 0x000000FF)
         | ((x >> 8)  & 0x0000FF00)
         | ((x << 8)  & 0x00FF0000)
         | ((x << 24) & 0xFF000000);
}

/* Parity (1 if odd number of bits set) */
static inline int parity(__private ProgramState *program_state, uint32_t x) {
    x ^= x >> 16;
    x ^= x >> 8;
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    return x & 1;
}

/* Using compiler builtins when available */
#if defined(__GNUC__) || defined(__clang__)
#define BUILTIN_POPCOUNT(x) __builtin_popcount(x)
#define BUILTIN_CLZ(x) ((x) ? __builtin_clz(x) : 32)
#define BUILTIN_CTZ(x) ((x) ? __builtin_ctz(x) : 32)
#define BUILTIN_FFS(x) __builtin_ffs(x)
#define BUILTIN_PARITY(x) __builtin_parity(x)
#else
#define BUILTIN_POPCOUNT(x) popcount_parallel(x)
#define BUILTIN_CLZ(x) clz_soft(x)
#define BUILTIN_CTZ(x) ctz_soft(x)
#define BUILTIN_FFS(x) ffs_soft(x)
#define BUILTIN_PARITY(x) parity(x)
#endif

int program_main(__private ProgramState *program_state) {
    uint32_t checksum = 0;

    for (uint32_t iter = 0; iter < 100000; iter++) {
        /* Generate test value */
        uint32_t x = iter * 0x9E3779B9;  /* Golden ratio hash */
        uint64_t x64 = ((uint64_t)x << 32) | (x ^ 0xDEADBEEF);

        /* Test various popcount implementations */
        int pc1 = popcount_naive(program_state, x);
        int pc2 = popcount_kernighan(program_state, x);
        int pc3 = popcount_parallel(program_state, x);
        int pc4 = popcount_lookup(program_state, x);
        int pc5 = BUILTIN_POPCOUNT(x);
        int pc6 = popcount64_parallel(program_state, x64);

        checksum += (uint32_t)(pc1 + pc2 + pc3 + pc4 + pc5 + pc6);

        /* Test CLZ/CTZ */
        int lz1 = clz_soft(program_state, x);
        int lz2 = BUILTIN_CLZ(x);
        int tz1 = ctz_soft(program_state, x);
        int tz2 = BUILTIN_CTZ(x);
        int ff = BUILTIN_FFS(x);

        checksum += (uint32_t)(lz1 + lz2 + tz1 + tz2 + ff);

        /* Test other operations */
        uint32_t rev = bit_reverse(program_state, x);
        uint32_t swp = byte_swap(program_state, x);
        int par = parity(program_state, x);
        int par2 = BUILTIN_PARITY(x);

        checksum ^= rev;
        checksum ^= swp;
        checksum += (uint32_t)(par + par2);
    }

    program_state->sink = checksum;
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)(((uint)storage.sink >> (8u * i)) & 255u);
}
