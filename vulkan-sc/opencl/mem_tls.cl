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
 * mem_tls.c
 *
 * Thread-local storage access patterns.
 * Exercises TLS data access and updates.
 *
 * Features exercised:
 *   - _Thread_local variables of multiple sizes
 *   - TLS access in loops and helper calls
 *   - Mixed TLS reads/writes and address taking
 */









typedef struct {
    uint32_t a;
    uint16_t b;
    uint8_t c;
    uint8_t d;
} TlsBlock;




typedef struct {
    volatile uint64_t sink;
    uint32_t tls_counter;
    uint64_t tls_accum;
    uint8_t tls_bytes[13];
    TlsBlock tls_block;
    uint64_t tls_array[4];
} ProgramState;


uint64_t tls_update(__private ProgramState *program_state, uint32_t seed) {
    uint32_t *ctr_ptr = &program_state->tls_counter;
    *ctr_ptr += seed + 1u;
    program_state->tls_accum ^= ((uint64_t)seed << 32) | (uint64_t)(*ctr_ptr);

    for (int i = 0; i < 13; i++) {
        program_state->tls_bytes[i] = (uint8_t)(program_state->tls_bytes[i] + seed + (uint32_t)i);
    }

    program_state->tls_block.a += seed ^ 0x5a5au;
    program_state->tls_block.b ^= (uint16_t)seed;
    program_state->tls_block.c = (uint8_t)(program_state->tls_block.c + 1u);
    program_state->tls_block.d = (uint8_t)(program_state->tls_block.d ^ (seed >> 3));

    program_state->tls_array[seed & 3u] ^= program_state->tls_accum;

    return program_state->tls_accum + program_state->tls_block.a + program_state->tls_block.b + program_state->tls_block.c + program_state->tls_block.d;
}


uint64_t tls_fold(__private ProgramState *program_state) {
    uint64_t sum = program_state->tls_counter + program_state->tls_accum;
    for (int i = 0; i < 13; i++) {
        sum += program_state->tls_bytes[i];
    }
    for (int i = 0; i < 4; i++) {
        sum ^= program_state->tls_array[i];
    }
    sum += program_state->tls_block.a + program_state->tls_block.b + program_state->tls_block.c + program_state->tls_block.d;
    return sum;
}

int program_main(__private ProgramState *program_state) {
    uint64_t result = 0;

    for (uint32_t i = 0; i < 5000; i++) {
        result += tls_update(program_state, i * 7u + 3u);
        if ((i & 31u) == 0u) {
            result ^= tls_fold(program_state);
        }
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
