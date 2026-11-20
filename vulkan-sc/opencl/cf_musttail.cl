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
// Guaranteed tail calls (`[[clang::musttail]]`).
//
// A `musttail` call must be the last instruction before its `ret`, and the
// `ret` must forward the call's result.  Obfuscation passes that split the
// return block, insert before the return, or rewrite the return operand must
// leave these sequences alone or they produce IR that fails the verifier.
// Exercises that resilience across integer-returning trampolines.






typedef struct {
    volatile int64_t sink;
} ProgramState;

 int leaf(__private ProgramState *program_state, int x) { return x * 3 + 1; }

 int trampoline(__private ProgramState *program_state, int x) {
    return leaf(program_state, x + 1);
}

 int64_t leaf64(__private ProgramState *program_state, int64_t x) {
    return (x ^ 0x5555) + 7;
}

 int64_t trampoline64(__private ProgramState *program_state, int64_t x) {
    if (x < 0)
        return leaf64(program_state, -x);
    return leaf64(program_state, x);
}

// A longer chain so the mutual-recursion style tail forwarding is present.
 unsigned countdown(__private ProgramState *program_state, unsigned n, unsigned acc) {
    while (n != 0)
        acc += n--;
    return acc;
}

int program_main(__private ProgramState *program_state) {
    long sum = 0;
    for (int i = 0; i < 1000; ++i)
        sum += trampoline(program_state, i);
    for (int64_t i = -500; i < 500; ++i)
        sum += trampoline64(program_state, i);
    sum += countdown(program_state, 100, 0);
    program_state->sink = sum;
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)(((ulong)storage.sink >> (8u * i)) & 255u);
}
