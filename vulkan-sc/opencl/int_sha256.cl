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
 * SHA-256 Hash Implementation
 *
 * Standard SHA-256 cryptographic hash function.
 * Heavy use of bitwise rotations, XOR, addition.
 *
 * Features exercised:
 *   - Bit rotation (ROR)
 *   - XOR, AND operations
 *   - 32-bit addition with overflow
 */





#define BLOCK_SIZE 64
#define HASH_SIZE 32



/* SHA-256 constants */
__constant static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

/* Initial hash values */
__constant static const uint32_t H_INIT[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

/* Rotate right */
typedef struct {
    uint32_t sink;
} ProgramState;

static inline uint32_t rotr(__global ProgramState *program_state, uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

/* SHA-256 functions */
static inline uint32_t Ch(__global ProgramState *program_state, uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

static inline uint32_t Maj(__global ProgramState *program_state, uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

static inline uint32_t Sigma0(__global ProgramState *program_state, uint32_t x) {
    return rotr(program_state, x, 2) ^ rotr(program_state, x, 13) ^ rotr(program_state, x, 22);
}

static inline uint32_t Sigma1(__global ProgramState *program_state, uint32_t x) {
    return rotr(program_state, x, 6) ^ rotr(program_state, x, 11) ^ rotr(program_state, x, 25);
}

static inline uint32_t sigma0(__global ProgramState *program_state, uint32_t x) {
    return rotr(program_state, x, 7) ^ rotr(program_state, x, 18) ^ (x >> 3);
}

static inline uint32_t sigma1(__global ProgramState *program_state, uint32_t x) {
    return rotr(program_state, x, 17) ^ rotr(program_state, x, 19) ^ (x >> 10);
}

/* Process a single 64-byte block */
static void sha256_transform(__global ProgramState *program_state, __private uint32_t state[8], const uint8_t block[64]) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;

    /* Prepare message schedule */
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint32_t)block[i * 4 + 0] << 24)
             | ((uint32_t)block[i * 4 + 1] << 16)
             | ((uint32_t)block[i * 4 + 2] << 8)
             | ((uint32_t)block[i * 4 + 3]);
    }

    for (int i = 16; i < 64; i++) {
        W[i] = sigma1(program_state, W[i - 2]) + W[i - 7] + sigma0(program_state, W[i - 15]) + W[i - 16];
    }

    /* Initialize working variables */
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    /* Main loop */
    for (int i = 0; i < 64; i++) {
        uint32_t T1 = h + Sigma1(program_state, e) + Ch(program_state, e, f, g) + K[i] + W[i];
        uint32_t T2 = Sigma0(program_state, a) + Maj(program_state, a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    /* Update state */
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

/* SHA-256 context */
typedef struct {
    uint32_t state[8];
    uint8_t buffer[64];
    size_t buffer_len;
    uint64_t total_len;
} SHA256_CTX;

static void sha256_init(__global ProgramState *program_state, __private SHA256_CTX *ctx) {
    memcpy(ctx->state, H_INIT, sizeof(H_INIT));
    ctx->buffer_len = 0;
    ctx->total_len = 0;
}

static void sha256_update(__global ProgramState *program_state, __private SHA256_CTX *ctx, const uint8_t *data, size_t len) {
    ctx->total_len += len;

    /* Fill buffer and process complete blocks */
    while (len > 0) {
        size_t to_copy = 64 - ctx->buffer_len;
        if (to_copy > len) to_copy = len;

        memcpy(ctx->buffer + ctx->buffer_len, data, to_copy);
        ctx->buffer_len += to_copy;
        data += to_copy;
        len -= to_copy;

        if (ctx->buffer_len == 64) {
            sha256_transform(program_state, ctx->state, ctx->buffer);
            ctx->buffer_len = 0;
        }
    }
}

static void sha256_final(__global ProgramState *program_state, __private SHA256_CTX *ctx, uint8_t hash[32]) {
    /* Padding */
    size_t pad_len = (ctx->buffer_len < 56) ? (56 - ctx->buffer_len) : (120 - ctx->buffer_len);
    uint8_t padding[128];
    memset(padding, 0, sizeof(padding));
    padding[0] = 0x80;

    /* Append length in bits */
    uint64_t bit_len = ctx->total_len * 8;
    padding[pad_len + 0] = (uint8_t)(bit_len >> 56);
    padding[pad_len + 1] = (uint8_t)(bit_len >> 48);
    padding[pad_len + 2] = (uint8_t)(bit_len >> 40);
    padding[pad_len + 3] = (uint8_t)(bit_len >> 32);
    padding[pad_len + 4] = (uint8_t)(bit_len >> 24);
    padding[pad_len + 5] = (uint8_t)(bit_len >> 16);
    padding[pad_len + 6] = (uint8_t)(bit_len >> 8);
    padding[pad_len + 7] = (uint8_t)(bit_len);

    sha256_update(program_state, ctx, padding, pad_len + 8);

    /* Output hash */
    for (int i = 0; i < 8; i++) {
        hash[i * 4 + 0] = (uint8_t)(ctx->state[i] >> 24);
        hash[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
        hash[i * 4 + 2] = (uint8_t)(ctx->state[i] >> 8);
        hash[i * 4 + 3] = (uint8_t)(ctx->state[i]);
    }
}

int program_main(__global ProgramState *program_state) {
    uint8_t data[256];
    uint8_t hash[32];
    uint32_t checksum = 0;

    /* Initialize test data */
    for (int i = 0; i < 256; i++) {
        data[i] = (uint8_t)(i & 0xFF);
    }

    for (int iter = 0; iter < 10000; iter++) {
        SHA256_CTX ctx;
        sha256_init(program_state, &ctx);

        /* Hash varying lengths */
        size_t len = (iter % 256) + 1;
        sha256_update(program_state, &ctx, data, len);

        /* Add iteration-dependent data */
        uint8_t extra[4];
        extra[0] = (uint8_t)(iter >> 24);
        extra[1] = (uint8_t)(iter >> 16);
        extra[2] = (uint8_t)(iter >> 8);
        extra[3] = (uint8_t)(iter);
        sha256_update(program_state, &ctx, extra, 4);

        sha256_final(program_state, &ctx, hash);

        /* Accumulate hash bytes */
        for (int i = 0; i < 32; i++) {
            checksum += hash[i];
        }

        /* Use hash to modify input for next iteration */
        data[iter % 256] ^= hash[0];
    }

    program_state->sink = checksum;
    return 0;
}

typedef struct { uint results[64]; ProgramState state; } Storage;
__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global Storage *buffer) {
    __global uint *results = buffer->results;
    __global ProgramState *storage = &buffer->state;
    results[0] = (uint)program_main(storage);
    for (uint i = 0; i < sizeof(storage->sink); ++i)
        results[1 + i] = (uint)(((uint)storage->sink >> (8u * i)) & 255u);
}
