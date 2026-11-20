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
 * call_mixed_aggregates.c
 *
 * Mixed aggregate calling conventions.
 * Stresses struct passing, return values, and alignment.
 *
 * Features exercised:
 *   - Small/large struct passing and returns
 *   - Mixed integer/float/double fields
 *   - Over-aligned aggregates
 *   - By-value aggregate arrays
 */





typedef struct {
    int8_t a;
    int16_t b;
    int32_t c;
} SmallA;

typedef struct {
    double d;
    int32_t x;
    uint16_t y;
    uint8_t z;
    uint8_t pad;
} MixedA;

typedef struct {
    float f[3];
    uint32_t tag;
} MixedB;

typedef struct {
    uint64_t a;
    uint64_t b;
    uint32_t c;
    uint32_t d;
} BigA;

typedef struct __attribute__((aligned(16))) {
    uint8_t bytes[24];
} AlignedA;

typedef struct {
    volatile uint64_t sink;
} ProgramState;


SmallA make_small(__private ProgramState *program_state, int32_t seed) {
    SmallA s;
    s.a = (int8_t)seed;
    s.b = (int16_t)(seed >> 3);
    s.c = seed ^ 0x12345678;
    return s;
}


MixedA tweak_mixed(__private ProgramState *program_state, MixedA m, SmallA s, int flag) {
    if (flag) {
        m.d += (double)s.c * 0.5;
        m.x ^= s.c;
    } else {
        m.d -= (double)s.b * 1.25;
        m.x += s.c;
    }
    m.y += (uint16_t)(s.a * 3);
    m.z ^= (uint8_t)(flag * 7);
    return m;
}


MixedB scale_mixedb(__private ProgramState *program_state, MixedB b, float scale) {
    for (int i = 0; i < 3; i++) {
        b.f[i] = b.f[i] * scale + (float)i;
    }
    b.tag ^= (uint32_t)(scale * 100.0f);
    return b;
}


BigA merge_big(__private ProgramState *program_state, BigA a, BigA b) {
    BigA out;
    out.a = a.a + b.b;
    out.b = a.b ^ b.a;
    out.c = a.c + b.c;
    out.d = a.d ^ b.d;
    return out;
}


AlignedA shuffle_aligned(__private ProgramState *program_state, AlignedA in, uint32_t seed) {
    AlignedA out = in;
    for (int i = 0; i < 24; i++) {
        uint32_t idx = (uint32_t)(i + seed) % 24u;
        out.bytes[i] = (uint8_t)(in.bytes[idx] ^ (uint8_t)(seed + i));
    }
    return out;
}


uint64_t consume_all(__private ProgramState *program_state, SmallA s, MixedA m, MixedB b, BigA big, AlignedA al) {
    float acc = b.f[0] + b.f[1] * 2.0f - b.f[2] * 0.25f;
    uint64_t sum = 0;

    sum += (uint64_t)s.a + (uint64_t)s.b + (uint64_t)s.c;
    sum += (uint64_t)m.x + (uint64_t)m.y + (uint64_t)m.z;
    sum += (uint64_t)(m.d + (double)acc);
    sum += big.a + big.b + big.c + big.d;

    for (int i = 0; i < 24; i++) {
        sum += al.bytes[i];
    }

    return sum;
}

int program_main(__private ProgramState *program_state) {
    uint64_t result = 0;

    for (uint32_t i = 0; i < 5000; i++) {
        SmallA s = make_small(program_state, (int32_t)i * 7);
        MixedA m = { (double)i * 1.5, (int32_t)i ^ 0x55aa55aa, (uint16_t)i, (uint8_t)(i ^ 0x5a), 0 };
        MixedB b = { { (float)i, (float)(i * 2), (float)(i * 3) }, i };
        BigA big = { (uint64_t)i << 32, (uint64_t)i * 0x1234u, i ^ 0xa5a5u, i + 1 };
        AlignedA al = { {0} };

        for (int j = 0; j < 24; j++) {
            al.bytes[j] = (uint8_t)(i + j * 3);
        }

        m = tweak_mixed(program_state, m, s, (i & 1u) != 0u);
        b = scale_mixedb(program_state, b, (float)((i & 7u) + 1u) * 0.5f);
        big = merge_big(program_state, big, (BigA){ big.b, big.a, big.d, big.c });
        al = shuffle_aligned(program_state, al, i);

        result += consume_all(program_state, s, m, b, big, al);
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
