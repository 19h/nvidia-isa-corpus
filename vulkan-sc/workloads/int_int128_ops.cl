// Arithmetic port of programs/int_int128_ops.c using two 64-bit limbs.
typedef struct { ulong lo, hi; } U128;

static U128 add(U128 a, U128 b) {
    ulong lo = a.lo + b.lo;
    return (U128){lo, a.hi + b.hi + (lo < a.lo)};
}

static U128 subtract(U128 a, U128 b) {
    return (U128){a.lo - b.lo, a.hi - b.hi - (a.lo < b.lo)};
}

static U128 shift_left(U128 a, uint count) {
    if (!count) return a;
    if (count >= 64) return (U128){0, a.lo << (count - 64)};
    return (U128){a.lo << count, (a.hi << count) | (a.lo >> (64 - count))};
}

static U128 shift_right(U128 a, uint count) {
    if (!count) return a;
    if (count >= 64) return (U128){a.hi >> (count - 64), 0};
    return (U128){(a.lo >> count) | (a.hi << (64 - count)), a.hi >> count};
}

static U128 xor_bits(U128 a, U128 b) {
    return (U128){a.lo ^ b.lo, a.hi ^ b.hi};
}

static U128 multiply(U128 a, U128 b) {
    return (U128){a.lo * b.lo, mul_hi(a.lo, b.lo) + a.lo * b.hi + a.hi * b.lo};
}

static U128 rotate_left(U128 a, uint count) {
    if (!count) return a;
    return xor_bits(shift_left(a, count), shift_right(a, 128 - count));
}

static U128 mix(U128 a, U128 b, uint count) {
    U128 z = xor_bits(add(a, shift_left(b, 7)), rotate_left(multiply(a, b), count));
    return add(z, xor_bits(shift_right(z, 11), shift_left(a, 3)));
}

static bool less(U128 a, U128 b) {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}

static U128 divmod(U128 numerator, U128 denominator, U128 *remainder) {
    const U128 zero = {0, 0};
    bool negative = (numerator.hi >> 63) != 0;
    if (negative) numerator = subtract(zero, numerator);
    U128 quotient = zero;
    U128 r = zero;
    for (int bit = 127; bit >= 0; --bit) {
        r = shift_left(r, 1);
        r.lo |= shift_right(numerator, (uint)bit).lo & 1UL;
        quotient = shift_left(quotient, 1);
        if (!less(r, denominator)) {
            r = subtract(r, denominator);
            quotient.lo |= 1UL;
        }
    }
    *remainder = negative ? subtract(zero, r) : r;
    return negative ? subtract(zero, quotient) : quotient;
}

static void output_word(__global uint *results, uint index, ulong value) {
    for (uint byte = 0; byte < 8; ++byte)
        results[index + byte] = (uint)((value >> (byte * 8)) & 255UL);
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ulong total = 0;
    U128 sink128 = {0, 0};
    for (ulong i = 1; i < 8000; ++i) {
        ulong hi = (i * 0x9e3779b97f4a7c15UL) & 0x7fffffffffffffffUL;
        ulong lo = i * 0xbf58476d1ce4e5b9UL + 0x94d049bb133111ebUL;
        U128 a = {lo, hi};
        U128 b = {(lo ^ 0x0f0e0d0c0b0a0908UL) | 1UL, hi ^ 0x13579bdf2468ace0UL};
        U128 mixed = mix(a, b, (uint)(i & 127UL));
        U128 remainder;
        U128 quotient = divmod(mixed, b, &remainder);
        sink128 = xor_bits(xor_bits(quotient, remainder), mixed);
        total += quotient.lo ^ quotient.hi ^ remainder.lo ^ remainder.hi;
    }
    results[0] = 0;
    output_word(results, 1, sink128.lo);
    output_word(results, 9, sink128.hi);
    output_word(results, 17, total);
}
