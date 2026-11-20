// Validator from programs/cf_license_crackme.c. The host presents the verdict.
typedef uint uint32_t;

static uint32_t r0(uint32_t x, unsigned n) {
    n &= 31u;
    return (x << n) | (x >> ((32u - n) & 31u));
}

static uint32_t r1(uint32_t x, unsigned n) {
    n &= 31u;
    return (x >> n) | (x << ((32u - n) & 31u));
}

static uint32_t m0(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static uint32_t m1(uint32_t x) {
    x += 0x9e3779b9u;
    x = (x ^ (x >> 13)) * 0xc2b2ae35u;
    x = (x ^ (x >> 16)) * 0x27d4eb2du;
    return x ^ (x >> 15);
}

static uint32_t q0(uint32_t a, uint32_t b) {
    a ^= r0(b + 0xa511e9b3u, (a ^ b) & 15u);
    a += (b ^ 0x63d83595u) * 0x45d9f3bu;
    return m0(a ^ r1(b, 7));
}

static uint32_t q1(uint32_t a, uint32_t b) {
    b += r1(a ^ 0x91e10da5u, (b >> 3) & 15u);
    b ^= (a + 0x7f4a7c15u) * 0x119de1f3u;
    return m1(a ^ b ^ r0(a, 11));
}

static uint32_t q2(uint32_t a, uint32_t b) {
    uint32_t c = (a & b) + ((a | b) ^ 0xbead6b61u);
    c ^= r0(a + b + 0x6c8e9cf5u, (c >> 27) & 31u);
    return m0(c + r1(b, 5));
}

static uint32_t q3(uint32_t a, uint32_t b) {
    uint32_t c = (a ^ 0xd1b54a35u) + r0(b, 9);
    c ^= (b + 0x27d4eb2du) * 0x165667b1u;
    return m1(c ^ r1(a, 13));
}

static uint32_t q4(uint32_t a, uint32_t b) {
    uint32_t c = (a + (b ^ 0x85ebca6bu)) ^ r1(a + b, 3);
    c += ((a << 1) | 1u) * ((b >> 1) | 1u);
    return m0(c ^ 0xc2b2ae35u);
}

static uint32_t q5(uint32_t a, uint32_t b) {
    uint32_t c = (a - r0(b, 17)) ^ ((b + 0x94d049bbu) * 0x9e3779b9u);
    c += (a ^ (b >> 3)) + 0x632be59bu;
    return m1(c);
}

static uint32_t q6(uint32_t a, uint32_t b) {
    uint32_t c = (a ^ r1(b, 21)) + ((a & 0xffffu) * 0x1003fu);
    c ^= (b | 0x13579bdfu) + r0(a, 3);
    return m0(c);
}

static uint32_t q7(uint32_t a, uint32_t b) {
    uint32_t c = ((a | 1u) * 0x2c1b3c6du) ^ (b + r1(a, 19));
    c += r0(c ^ b, 7) ^ 0x297a2d39u;
    return m1(c);
}

static uint32_t dispatch_q(unsigned index, uint32_t a, uint32_t b) {
    switch (index) {
        case 0u: return q0(a, b);
        case 1u: return q1(a, b);
        case 2u: return q2(a, b);
        case 3u: return q3(a, b);
        case 4u: return q4(a, b);
        case 5u: return q5(a, b);
        case 6u: return q6(a, b);
        default: return q7(a, b);
    }
}

static uint32_t l0(const unsigned char *b, size_t n, unsigned t) {
    uint32_t a = 0x4f1bbcdcu ^ ((uint32_t)t * 0x1f123bb5u);
    uint32_t c = 0xa0761d65u + ((uint32_t)n * 0x7f4a7c15u);
    for (size_t i = 0; i < n; ++i) {
        uint32_t w = (uint32_t)b[i] + ((uint32_t)i << ((i & 3u) + 1u)) + r0(c, (unsigned)((i + t) & 15u));
        unsigned s = (unsigned)((a ^ c ^ w ^ (uint32_t)i ^ ((uint32_t)t << 2)) & 7u);
        a = dispatch_q(s, a, w ^ c);
        if (((i + t) & 3u) == 1u) {
            c ^= dispatch_q((s + 3u) & 7u, c + 0x9e3779b9u, a ^ ((uint32_t)i * 0x45d9f3bu));
        } else {
            c += r1(a ^ w, (unsigned)((i * 5u + t) & 31u)) ^ 0x6d2b79f5u;
        }
    }
    return m0(a ^ r0(c, 9) ^ (uint32_t)(n * 0x27d4eb2du));
}

static uint32_t l1(const unsigned char *b, unsigned t) {
    uint32_t s = 0x243f6a88u ^ ((uint32_t)t * 0x9e3779b9u);
    for (unsigned i = 0; i < 11u; ++i) {
        uint32_t k = (uint32_t)b[(i * 7u + 3u) % 11u] + ((uint32_t)b[i] << ((i & 3u) + 4u));
        switch ((s ^ k ^ i ^ t) & 7u) {
            case 0u: s = q3(s, k + 0x13198a2eu); break;
            case 1u: s = q6(k ^ s, r0(s, 5)); break;
            case 2u: s ^= q1(s + k, 0xa4093822u ^ i); break;
            case 3u: s += q5(s ^ 0x299f31d0u, k); break;
            case 4u: s = r0(s + q2(k, s), (i + 11u) & 31u); break;
            case 5u: s = q7(s - k, k ^ 0x082efa98u); break;
            case 6u: s ^= r1(q0(s, k), (t + i) & 31u); break;
            default: s += q4(k + 0xec4e6c89u, s); break;
        }
    }
    return m1(s ^ r1(s, 17) ^ 0x452821e6u);
}

static uint32_t u0(const unsigned char *p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint32_t u2(uint32_t a, uint32_t b, unsigned y, unsigned t) {
    uint32_t x = a + r0(b ^ 0xbead6b61u, t + 9u);
    x ^= (uint32_t)y * 0x1f1f1f1fu;
    x += ((uint32_t)t + 1u) * 0x7f4a7c15u;
    return m0(x ^ r1(a, 11) ^ 0x5bf03635u);
}

static unsigned char u3(uint32_t a, uint32_t b, unsigned y, unsigned z, unsigned t) {
    uint32_t x = a ^ r0(b, t + 5u) ^ ((uint32_t)y << 13) ^ ((uint32_t)(z & 0xfcu) << 21) ^ 0x91e10da5u;
    uint32_t k = m1(x) ^ m0(b + ((uint32_t)t * 0x45d9f3bu) + (a >> 3));
    return (unsigned char)((k >> 11) ^ (k >> 19) ^ ((uint32_t)t * 0x53u) ^ ((uint32_t)y * 13u));
}

static unsigned g0(int ch) {
    unsigned c = (unsigned char)ch;
    if (c >= (unsigned)'a' && c <= (unsigned)'z') {
        c -= 32u;
    }
    if (c >= (unsigned)'2' && c <= (unsigned)'9') {
        return c - (unsigned)'2';
    }
    if (c >= (unsigned)'A' && c <= (unsigned)'H') {
        return 8u + c - (unsigned)'A';
    }
    if (c >= (unsigned)'J' && c <= (unsigned)'N') {
        return 8u + c - (unsigned)'A' - 1u;
    }
    if (c >= (unsigned)'P' && c <= (unsigned)'Z') {
        return 8u + c - (unsigned)'A' - 2u;
    }
    return 255u;
}

static int g1(const char *in, unsigned char out[24]) {
    size_t n = 0;
    for (size_t i = 0; in[i] != 0; ++i) {
        unsigned c = (unsigned char)in[i];
        if (c == (unsigned)'-' || c == (unsigned)' ' || c == (unsigned)'\t' ||
            c == (unsigned)'\n' || c == (unsigned)'\r') {
            continue;
        }
        if (g0((int)c) >= 32u) {
            return -1;
        }
        if (c >= (unsigned)'a' && c <= (unsigned)'z') {
            c -= 32u;
        }
        if (n >= 24u) {
            return -1;
        }
        out[n++] = (unsigned char)c;
    }
    return n == 24u ? 0 : -1;
}

static int g2(const unsigned char in[24], unsigned char out[15]) {
    uint32_t acc = 0;
    unsigned bits = 0;
    unsigned j = 0;
    for (unsigned i = 0; i < 24u; ++i) {
        unsigned v = g0((int)in[i]);
        if (v >= 32u) {
            return 0;
        }
        acc = (acc << 5) | v;
        bits += 5u;
        if (bits >= 8u) {
            bits -= 8u;
            if (j >= 15u) {
                return 0;
            }
            out[j++] = (unsigned char)((acc >> bits) & 0xffu);
        }
    }
    return j == 15u && bits == 0u;
}

struct k1 {
    unsigned ok;
    unsigned shape;
    unsigned tier;
    uint32_t mark;
};

/* Returns a packed verdict (bit0=ok, bit1=shape, bits2-3=tier) instead of a
 * by-value struct: a scalar return is virtualizable, an aggregate return is not.
 * The internal struct is kept so the decision logic is byte-for-byte unchanged. */
static unsigned k2(const char *s) {
    struct k1 v = { 0u, 0u, 0u, 0u };
    unsigned char a[24];
    unsigned char b[15];
    if (g1(s, a) == 0) {
        v.shape = 1u;
        if (g2(a, b)) {
            uint32_t x = u0(b);
            uint32_t y = u0(b + 4);
            unsigned cloak = (((x >> 7) + (y >> 11) + 0x2du) & 3u);
            unsigned tier = ((unsigned)b[9] & 3u) ^ cloak;
            if (tier < 4u) {
                uint32_t p = l1(b, tier);
                switch ((p ^ x ^ (uint32_t)b[8] ^ tier) & 7u) {
                    case 0u:
                    case 3u:
                    case 5u:
                        p ^= q2(p, y + b[10]);
                        break;
                    case 1u:
                    case 6u:
                        p += q5(x ^ p, y ^ b[9]);
                        break;
                    case 2u:
                        p = r0(p ^ q7(y, x), (tier + 9u) & 31u);
                        break;
                    default:
                        p ^= q4(y + p, x ^ 0xd1b54a35u);
                        break;
                }
                if ((((unsigned)b[9] & 3u) ^ cloak) == tier) {
                    unsigned char want = u3(x, y, b[8], b[9], tier);
                    if (b[10] == want) {
                        uint32_t tag = l0(b, 11u, tier) ^ l1(b, tier) ^ u2(x, y, b[8], tier);
                        uint32_t got = u0(b + 11);
                        if (got == tag) {
                            uint32_t last = p ^ tag ^ q6(x + b[8], y + b[10]);
                            volatile uint32_t drag = last ^ p ^ got;
                            if (drag != 0u || b[10] == want) {
                                v.ok = 1u;
                                v.tier = tier;
                                v.mark = last;
                            }
                        }
                    }
                }
            }
        }
    }
    return (v.ok & 1u) | ((v.shape & 1u) << 1) | ((v.tier & 3u) << 2);
}


__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *result) {
    char candidate[160];
    unsigned length = result[64] != 0u ? 159u : 24u;
    for (unsigned i = 0; i < length; ++i) {
        unsigned word = length == 24u ? 63u - i / 4u : 65u + i / 4u;
        candidate[i] = (char)(result[word] >> ((i & 3u) * 8u));
    }
    candidate[length] = 0;
    result[0] = 0u;
    result[1] = k2(candidate);
}
