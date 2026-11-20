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
// Large struct by-value parameters and returns (byval / sret in IR).
//
// These calls carry ABI parameter attributes (byval, sret).  Passes that route
// calls through forwarder proxies must preserve those attributes end-to-end or
// the caller and callee disagree on argument layout — a silent ABI miscompile
// the verifier does not catch.  Deterministic checksum output makes any such
// corruption observable by a clean-vs-obfuscated diff.






typedef struct {
    long a[8];
} Big;

typedef struct {
    volatile int64_t sink;
} ProgramState;

 Big make(__private ProgramState *program_state, long base) {
    Big b;
    for (int i = 0; i < 8; ++i)
        b.a[i] = base + (long)i * i;
    return b; // sret
}

 long consume(__private ProgramState *program_state, Big x) { // byval
    long s = 0;
    for (int i = 0; i < 8; ++i)
        s += x.a[i] * (i + 1);
    return s;
}

 Big transform(__private ProgramState *program_state, Big x, long k) { // byval + sret
    Big r;
    for (int i = 0; i < 8; ++i)
        r.a[i] = x.a[i] ^ k;
    return r;
}

int program_main(__private ProgramState *program_state) {
    long total = 0;
    for (long i = 0; i < 20000; ++i) {
        Big b = make(program_state, i);
        Big t = transform(program_state, b, i & 0x3f);
        total += consume(program_state, t);
    }
    program_state->sink = total;
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)(((ulong)storage.sink >> (8u * i)) & 255u);
}
