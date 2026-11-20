// Validator from programs/cf_rorog.c. The host handles the terminal interface.
typedef uint uint32_t;
typedef ushort uint16_t;

#define R_ROUNDS 16u
#define RS 0x9A17C0DEu
#define MAGIC_X 0xF12EB19Du
#define MAGIC_Y 0x2A606C63u
#define TIER_STEP 0x51A3E7D1u
__constant char RUNE[16] = {'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P',
                            'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Z'};

static uint32_t rotl32(uint32_t x, unsigned r) {
    r &= 31u;
    return (x << r) | (x >> ((32u - r) & 31u));
}
static uint32_t rotr32(uint32_t x, unsigned r) {
    r &= 31u;
    return (x >> r) | (x << ((32u - r) & 31u));
}
static void speck(uint32_t *x, uint32_t *y, uint32_t k) {
    uint32_t a = *x, b = *y;
    a = (rotr32(a, 8) + b);
    a ^= k;
    b = rotl32(b, 3) ^ a;
    *x = a;
    *y = b;
}
static void schedule(unsigned tier, uint32_t k[R_ROUNDS]) {
    k[0] = RS;
    for (unsigned i = 1; i < R_ROUNDS; ++i) {
        uint32_t t = k[i - 1] ^ (i * 0x9E3779B9u + tier * 0x85EBCA6Bu);
        k[i] = rotl32(t, 5) + 0x27D4EB2Du;
    }
}
static uint32_t fnv1a(const unsigned char *p, size_t n) {
    uint32_t h = 0x811c9dc5u;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x01000193u;
    }
    return h;
}
static uint16_t crc16(const unsigned char *p, size_t n) {
    uint16_t c = 0xFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c ^= (uint16_t)p[i] << 8;
        for (int b = 0; b < 8; b++)
            c = (c & 0x8000u) ? (uint16_t)((c << 1) ^ 0x1021u) : (uint16_t)(c << 1);
    }
    return c;
}
static void norm(const char *s, size_t len, unsigned char *out, size_t *on) {
    size_t j = 0;
    for (size_t i = 0; i < len; ++i) {
        unsigned c = (unsigned char)s[i];
        if (c == '-' || c == '_' || c == ' ')
            continue;
        if (c >= 'A' && c <= 'Z')
            c += 32u;
        out[j++] = (unsigned char)c;
    }
    *on = j;
}

static void derive(const char *name, size_t nlen, unsigned tier, unsigned char sig[12]) {
    unsigned char nb[256];
    size_t nn = 0;
    if (nlen > sizeof nb)
        nlen = sizeof nb;
    norm(name, nlen, nb, &nn);
    uint32_t oh = fnv1a(nb, nn);
    uint32_t ks[R_ROUNDS];
    schedule(tier, ks);
    uint32_t x = oh ^ MAGIC_X, y = MAGIC_Y + tier * TIER_STEP;
    for (unsigned r = 0; r < R_ROUNDS; ++r)
        speck(&x, &y, ks[r]);
    uint32_t A = x, N = y;
    unsigned cloak = ((A >> 5) ^ (N >> 11) ^ (A * N)) & 3u;
    sig[0] = (unsigned char)A;
    sig[1] = (unsigned char)(A >> 8);
    sig[2] = (unsigned char)(A >> 16);
    sig[3] = (unsigned char)(A >> 24);
    sig[4] = (unsigned char)N;
    sig[5] = (unsigned char)(N >> 8);
    sig[6] = (unsigned char)(N >> 16);
    sig[7] = (unsigned char)(N >> 24);
    unsigned char mb[9];
    for (unsigned i = 0; i < 8; ++i)
        mb[i] = sig[i];
    mb[8] = (unsigned char)tier;
    uint32_t m = fnv1a(mb, 9);
    sig[8] = (unsigned char)((m ^ (m >> 8) ^ (m >> 16) ^ (m >> 24)) & 0xffu);
    sig[9] = (unsigned char)(((m >> 3) & 0xFCu) | ((tier ^ cloak) & 3u));
    uint16_t tag = crc16(sig, 10);
    sig[10] = (unsigned char)tag;
    sig[11] = (unsigned char)(tag >> 8);
}

static int rune_val(unsigned c) {
    if (c >= 'a' && c <= 'z')
        c -= 32u;
    for (int i = 0; i < 16; ++i)
        if ((unsigned char)RUNE[i] == c)
            return i;
    return -1;
}
static int decode_sigil(const char *s, unsigned char out[12]) {
    unsigned nib[24];
    unsigned k = 0;
    for (size_t i = 0; s[i]; ++i) {
        unsigned c = (unsigned char)s[i];
        if (c == '-' || c == ' ' || c == '\t' || c == '\r' || c == '\n')
            continue;
        int v = rune_val(c);
        if (v < 0)
            return -1;
        if (k >= 24u)
            return -1;
        nib[k++] = (unsigned)v;
    }
    if (k != 24u)
        return -1;
    for (unsigned i = 0; i < 12u; ++i)
        out[i] = (unsigned char)((nib[2 * i] << 4) | nib[2 * i + 1]);
    return 0;
}
/* ==== analysis-only distractions (never affect the live verdict) ==== */
static int opq(uint32_t x) {
    return (int)((x * x + x) & 1u);
} /* x*x+x always even -> always 0 */
__constant unsigned char SB[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};
static uint32_t sbnoise(uint32_t x) {
    unsigned char v = SB[x & 0xffu];
    return (uint32_t)(v ^ v);
} /* == 0 */
static unsigned rarog_shadow(const unsigned char sig[12], const char *name, size_t nlen) {
    unsigned char probe[12];
    derive(name, nlen, (unsigned)(sig[9] & 3u), probe);
    uint32_t acc = fnv1a(probe, 12) ^ fnv1a(sig, 12);
    acc += sbnoise(acc);
    return (acc == 0x1CEDF00Du) ? (1u | (2u << 1)) : 0u;
}

/* ==== real verdict: bit0=granted, bit1=well-shaped, bits2-3=tier ==== */
static unsigned verify(const char *name, size_t nlen, const char *sigil) {
    unsigned char sig[12];
    if (decode_sigil(sigil, sig) != 0)
        return 0u;
    unsigned shaped = 2u;
    uint32_t A = (uint32_t)sig[0] | ((uint32_t)sig[1] << 8) | ((uint32_t)sig[2] << 16) |
                 ((uint32_t)sig[3] << 24);
    uint32_t N = (uint32_t)sig[4] | ((uint32_t)sig[5] << 8) | ((uint32_t)sig[6] << 16) |
                 ((uint32_t)sig[7] << 24);
    unsigned cloak = ((A >> 5) ^ (N >> 11) ^ (A * N)) & 3u;
    unsigned tier = ((unsigned)sig[9] & 3u) ^ cloak;
    if (opq(A ^ N))
        return rarog_shadow(sig, name, nlen) | shaped; /* dead path */
    unsigned char want[12];
    derive(name, nlen, tier, want);
    unsigned mismatch = 0u;
    for (unsigned i = 0; i < 12u; ++i)
        mismatch |= want[i] ^ sig[i];
    int ok = mismatch == 0u;
    return (ok ? 1u : 0u) | shaped | ((tier & 3u) << 2);
}

__attribute__((reqd_work_group_size(1, 1, 1))) __kernel void program_entry(__global uint *result) {
    char name[64], candidate[64];
    if (result[64] != 0u) {
        for (unsigned i = 0; i < 63u; ++i) {
            name[i] = (char)(result[65u + i / 4u] >> ((i & 3u) * 8u));
            candidate[i] = (char)(result[81u + i / 4u] >> ((i & 3u) * 8u));
        }
    } else {
        const char fixture[] = "GPU Corpus";
        for (unsigned i = 0; i < sizeof fixture; ++i)
            name[i] = fixture[i];
        for (unsigned i = 0; i < 24u; ++i)
            candidate[i] = (char)(result[63u - i / 4u] >> ((i & 3u) * 8u));
        candidate[24] = 0;
    }
    name[63] = 0;
    candidate[63] = 0;
    unsigned length = 0;
    while (name[length] != 0)
        ++length;
    result[0] = 0u;
    result[1] = verify(name, length, candidate);
}
