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
 * int_bitfields.c
 *
 * Complex bitfield packing and extraction patterns.
 * Stresses sign extension, packed storage units, and zero-width alignment.
 *
 * Features exercised:
 *   - Packed bitfields across storage units
 *   - Signed/unsigned bitfield extraction and sign extension
 *   - Zero-width bitfields to force alignment
 *   - Volatile bitfield access
 */





typedef struct __attribute__((packed)) {
    unsigned int a:3;
    unsigned int b:5;
    unsigned int c:1;
    unsigned int :0;
    unsigned int d:17;
    signed int e:11;
    unsigned long long f:33;
    unsigned long long g:31;
} BitFieldA;

typedef struct __attribute__((packed)) {
    signed int s1:4;
    signed int s2:7;
    unsigned int u1:5;
    unsigned int u2:9;
    unsigned int :0;
    signed long long s3:20;
    unsigned long long u3:12;
} BitFieldB;

typedef struct {
    volatile uint64_t sink;
} ProgramState;


uint64_t pack_fields(__private ProgramState *program_state, volatile BitFieldA *bf, uint32_t seed) {
    bf->a = seed & 0x7u;
    bf->b = (seed >> 3) & 0x1fu;
    bf->c = (seed >> 8) & 0x1u;
    bf->d = (seed * 3u) & 0x1ffffu;
    bf->e = (int)(seed & 0x7ffu) - 512;
    bf->f = ((unsigned long long)seed << 13) ^ 0x1ffffffffull;
    bf->g = (bf->f >> 1) & 0x7fffffffu;

    return (((uint64_t)bf->a) |
            ((uint64_t)bf->b << 3) |
            ((uint64_t)bf->c << 8) |
            ((uint64_t)bf->d << 9)) ^
           (uint64_t)bf->f;
}


int64_t sign_fields(__private ProgramState *program_state, volatile BitFieldB *bf, int32_t seed) {
    uint32_t useed = (uint32_t)seed;
    int64_t s3 = ((int64_t)seed * 17);

    bf->s1 = (int)((useed & 0x7u) - 4u);
    bf->s2 = (int)(((useed >> 3) & 0x3fu) - 32u);
    bf->u1 = (unsigned int)((useed >> 1) & 0x1fu);
    bf->u2 = (unsigned int)((useed * 5u) & 0x1ffu);
    bf->s3 = (int)((s3 & 0xfffff) - 0x80000);
    bf->u3 = (unsigned int)((useed >> 2) & 0xfffu);

    return (int64_t)bf->s1 + (int64_t)bf->s2 +
           (int64_t)bf->u1 + (int64_t)bf->u2 +
           (int64_t)bf->s3 - (int64_t)bf->u3;
}

int program_main(__private ProgramState *program_state) {
    uint64_t result = 0;
    volatile BitFieldA a;
    volatile BitFieldB b;

    for (uint32_t i = 0; i < 50000; i++) {
        result += pack_fields(program_state, &a, i * 11u + 7u);
        result += (uint64_t)sign_fields(program_state, &b, (int32_t)i - 25000);
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
