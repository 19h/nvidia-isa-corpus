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
 * int_int128_ops.c
 *
 * 128-bit arithmetic and conversions.
 * Exercises wide integer operations, shifts, and division helpers.
 *
 * Features exercised:
 *   - 128-bit add/mul/shift/xor
 *   - 128-bit division and modulo
 *   - 64 <-> 128 conversions
 *   - Wide rotate/bit-mix patterns
 */



typedef unsigned __int128 u128;
typedef __int128 s128;




typedef struct {
    volatile u128 sink128;
    volatile uint64_t sink;
} ProgramState;

static inline u128 make_u128(__private ProgramState *program_state, uint64_t hi, uint64_t lo) {
    return ((u128)hi << 64) | (u128)lo;
}


u128 rotl_u128(__private ProgramState *program_state, u128 v, unsigned shift) {
    shift &= 127u;
    if (shift == 0) {
        return v;
    }
    return (v << shift) | (v >> (128u - shift));
}


u128 mix_u128(__private ProgramState *program_state, u128 a, u128 b, unsigned shift) {
    u128 x = a + (b << 7);
    u128 y = a * (b | 1);
    u128 z = x ^ rotl_u128(program_state, y, shift);
    z += (z >> 11) ^ (a << 3);
    return z;
}


s128 divmod_u128(__private ProgramState *program_state, s128 numer, s128 denom, s128 *rem) {
    s128 q = numer / denom;
    *rem = numer % denom;
    return q;
}


uint64_t fold_u128(__private ProgramState *program_state, u128 v) {
    return (uint64_t)v ^ (uint64_t)(v >> 64);
}

int program_main(__private ProgramState *program_state) {
    uint64_t result = 0;

    for (uint64_t i = 1; i < 8000; i++) {
        uint64_t hi = (i * 0x9e3779b97f4a7c15ull) & 0x7fffffffffffffffull;
        uint64_t lo = i * 0xbf58476d1ce4e5b9ull + 0x94d049bb133111ebull;
        u128 a = make_u128(program_state, hi, lo);
        u128 b = make_u128(program_state, hi ^ 0x13579bdf2468ace0ull,
                           lo ^ 0x0f0e0d0c0b0a0908ull) | 1u;

        u128 mixed = mix_u128(program_state, a, b, (unsigned)(i & 127u));

        s128 rem = 0;
        s128 q = divmod_u128(program_state, (s128)mixed, (s128)(b | 1u), &rem);

        program_state->sink128 = (u128)q ^ (u128)rem ^ mixed;
        result += fold_u128(program_state, (u128)q) ^ fold_u128(program_state, (u128)rem);
    }

    program_state->sink = result;
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink128); ++i)
        results[1 + i] = (uint)(((uint)storage.sink128 >> (8u * i)) & 255u);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[17 + i] = (uint)(((ulong)storage.sink >> (8u * i)) & 255u);
}
