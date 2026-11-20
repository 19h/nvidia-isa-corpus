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
 * Big Integer Arithmetic
 *
 * Arbitrary precision integer operations.
 * Common in cryptography (RSA, ECC).
 *
 * Features exercised:
 *   - Multi-word arithmetic
 *   - Carry propagation
 *   - 64-bit multiply producing 128-bit result
 */




#define BIGINT_WORDS 16  /* 512 bits */



typedef struct {
    uint32_t words[BIGINT_WORDS];
} BigInt;

/* Set from 32-bit value */
typedef struct {
    volatile uint32_t sink;
} ProgramState;

static void bigint_from_u32(__private ProgramState *program_state, BigInt *a, uint32_t val) {
    memset(a->words, 0, sizeof(a->words));
    a->words[0] = val;
}

/* Copy */
static void bigint_copy(__private ProgramState *program_state, BigInt *dst, const BigInt *src) {
    memcpy(dst->words, src->words, sizeof(dst->words));
}

/* Compare: returns -1, 0, or 1 */
static int bigint_cmp(__private ProgramState *program_state, const BigInt *a, const BigInt *b) {
    for (int i = BIGINT_WORDS - 1; i >= 0; i--) {
        if (a->words[i] > b->words[i]) return 1;
        if (a->words[i] < b->words[i]) return -1;
    }
    return 0;
}

/* Add: c = a + b, returns carry */
static uint32_t bigint_add(__private ProgramState *program_state, BigInt *c, const BigInt *a, const BigInt *b) {
    uint64_t carry = 0;

    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t sum = (uint64_t)a->words[i] + b->words[i] + carry;
        c->words[i] = (uint32_t)sum;
        carry = sum >> 32;
    }

    return (uint32_t)carry;
}

/* Subtract: c = a - b, returns borrow */
static uint32_t bigint_sub(__private ProgramState *program_state, BigInt *c, const BigInt *a, const BigInt *b) {
    int64_t borrow = 0;

    for (int i = 0; i < BIGINT_WORDS; i++) {
        int64_t diff = (int64_t)a->words[i] - b->words[i] - borrow;
        if (diff < 0) {
            diff += (int64_t)1 << 32;
            borrow = 1;
        } else {
            borrow = 0;
        }
        c->words[i] = (uint32_t)diff;
    }

    return (uint32_t)borrow;
}

/* Multiply: c = a * b (lower half only) */
static void bigint_mul(__private ProgramState *program_state, BigInt *c, const BigInt *a, const BigInt *b) {
    uint32_t result[BIGINT_WORDS * 2];
    memset(result, 0, sizeof(result));

    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < BIGINT_WORDS; j++) {
            uint64_t prod = (uint64_t)a->words[i] * b->words[j]
                          + result[i + j] + carry;
            result[i + j] = (uint32_t)prod;
            carry = prod >> 32;
        }
        if (i + BIGINT_WORDS < BIGINT_WORDS * 2) {
            result[i + BIGINT_WORDS] = (uint32_t)carry;
        }
    }

    /* Copy lower half */
    memcpy(c->words, result, sizeof(c->words));
}

/* Multiply by single word */
static uint32_t bigint_mul_word(__private ProgramState *program_state, BigInt *c, const BigInt *a, uint32_t b) {
    uint64_t carry = 0;

    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t prod = (uint64_t)a->words[i] * b + carry;
        c->words[i] = (uint32_t)prod;
        carry = prod >> 32;
    }

    return (uint32_t)carry;
}

/* Left shift by words */
static void bigint_shl_words(__private ProgramState *program_state, BigInt *a, int n) {
    if (n >= BIGINT_WORDS) {
        memset(a->words, 0, sizeof(a->words));
        return;
    }

    for (int i = BIGINT_WORDS - 1; i >= n; i--) {
        a->words[i] = a->words[i - n];
    }
    for (int i = 0; i < n; i++) {
        a->words[i] = 0;
    }
}

/* Left shift by bits */
static void bigint_shl(__private ProgramState *program_state, BigInt *a, int bits) {
    if (bits >= 32) {
        bigint_shl_words(program_state, a, bits / 32);
        bits %= 32;
    }

    if (bits == 0) return;

    uint32_t carry = 0;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint32_t new_carry = a->words[i] >> (32 - bits);
        a->words[i] = (a->words[i] << bits) | carry;
        carry = new_carry;
    }
}

/* Right shift by bits */
static void bigint_shr(__private ProgramState *program_state, BigInt *a, int bits) {
    int word_shift = bits / 32;
    int bit_shift = bits % 32;

    if (word_shift >= BIGINT_WORDS) {
        memset(a->words, 0, sizeof(a->words));
        return;
    }

    for (int i = 0; i < BIGINT_WORDS - word_shift; i++) {
        a->words[i] = a->words[i + word_shift];
    }
    for (int i = BIGINT_WORDS - word_shift; i < BIGINT_WORDS; i++) {
        a->words[i] = 0;
    }

    if (bit_shift > 0) {
        uint32_t carry = 0;
        for (int i = BIGINT_WORDS - 1; i >= 0; i--) {
            uint32_t new_carry = a->words[i] << (32 - bit_shift);
            a->words[i] = (a->words[i] >> bit_shift) | carry;
            carry = new_carry;
        }
    }
}

static int bigint_is_zero(__private ProgramState *program_state, const BigInt *a) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        if (a->words[i] != 0) {
            return 0;
        }
    }
    return 1;
}

static uint32_t bigint_get_bit(__private ProgramState *program_state, const BigInt *a, int bit) {
    return (a->words[bit / 32] >> (bit % 32)) & 1u;
}

static void bigint_shl1(__private ProgramState *program_state, BigInt *a) {
    uint32_t carry = 0;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint32_t next_carry = a->words[i] >> 31;
        a->words[i] = (a->words[i] << 1) | carry;
        carry = next_carry;
    }
}

/* Modular reduction: a = a mod m, bounded by the operand bit width. */
static void bigint_mod(__private ProgramState *program_state, BigInt *a, const BigInt *m) {
    if (bigint_is_zero(program_state, m)) {
        return;
    }

    BigInt remainder;
    memset(remainder.words, 0, sizeof(remainder.words));

    for (int bit = BIGINT_WORDS * 32 - 1; bit >= 0; bit--) {
        bigint_shl1(program_state, &remainder);
        remainder.words[0] |= bigint_get_bit(program_state, a, bit);
        if (bigint_cmp(program_state, &remainder, m) >= 0) {
            bigint_sub(program_state, &remainder, &remainder, m);
        }
    }

    bigint_copy(program_state, a, &remainder);
}

/* Modular multiplication: c = (a * b) mod m */
static void bigint_mulmod(__private ProgramState *program_state, BigInt *c, const BigInt *a, const BigInt *b, const BigInt *m) {
    bigint_mul(program_state, c, a, b);
    bigint_mod(program_state, c, m);
}

/* Modular exponentiation: c = a^e mod m (square-and-multiply) */
static void bigint_powmod(__private ProgramState *program_state, BigInt *c, const BigInt *base, const BigInt *exp, const BigInt *m) {
    BigInt result, b, e;

    bigint_from_u32(program_state, &result, 1);
    bigint_copy(program_state, &b, base);
    bigint_copy(program_state, &e, exp);

    /* Square and multiply */
    for (int i = 0; i < BIGINT_WORDS * 32; i++) {
        if (e.words[0] & 1) {
            bigint_mulmod(program_state, &result, &result, &b, m);
        }
        bigint_mulmod(program_state, &b, &b, &b, m);
        bigint_shr(program_state, &e, 1);

        /* Check if exp is zero */
        int is_zero = 1;
        for (int j = 0; j < BIGINT_WORDS; j++) {
            if (e.words[j] != 0) {
                is_zero = 0;
                break;
            }
        }
        if (is_zero) break;
    }

    bigint_copy(program_state, c, &result);
}

int program_main(__private ProgramState *program_state) {
    BigInt a, b, c, m;
    uint32_t checksum = 0;

    /* Initialize test values */
    bigint_from_u32(program_state, &m, 0);
    m.words[0] = 0xFFFFFFFF;
    m.words[1] = 0x0000FFFF;  /* Small modulus for testing */

    for (int iter = 0; iter < 1000; iter++) {
        /* Set up operands */
        bigint_from_u32(program_state, &a, iter * 17 + 31);
        bigint_from_u32(program_state, &b, iter * 13 + 47);

        /* Shift to make them bigger */
        bigint_shl(program_state, &a, iter % 32);
        bigint_shl(program_state, &b, (iter * 7) % 32);

        /* Addition */
        bigint_add(program_state, &c, &a, &b);
        checksum ^= c.words[0];

        /* Subtraction */
        if (bigint_cmp(program_state, &a, &b) >= 0) {
            bigint_sub(program_state, &c, &a, &b);
        } else {
            bigint_sub(program_state, &c, &b, &a);
        }
        checksum ^= c.words[0];

        /* Multiplication */
        bigint_mul(program_state, &c, &a, &b);
        checksum ^= c.words[0];

        /* Multiply by word */
        bigint_mul_word(program_state, &c, &a, iter + 1);
        checksum ^= c.words[0];

        /* Modular operations (every 10th iteration to save time) */
        if (iter % 10 == 0) {
            bigint_from_u32(program_state, &a, iter + 2);
            bigint_from_u32(program_state, &b, (iter % 16) + 2);
            bigint_powmod(program_state, &c, &a, &b, &m);
            checksum ^= c.words[0];
        }
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
