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
 * CRC32 Implementation
 *
 * Multiple CRC32 implementations:
 * - Table-based (standard)
 * - Byte-at-a-time
 * - Slice-by-4 (faster)
 *
 * Features exercised:
 *   - XOR operations
 *   - Table lookups
 *   - Shift operations
 */




#define DATA_SIZE 4096





/* CRC32 polynomial (IEEE 802.3) */
#define CRC32_POLY 0xEDB88320

/* Pre-computed CRC table */





/* Generate CRC lookup table */
typedef struct {
    uint32_t sink;
    uint8_t data[DATA_SIZE];
    uint32_t crc32c_table[256];
    uint32_t crc_table[256];
    uint32_t crc_table_slice[4][256];
    int tables_initialized;
} ProgramState;

static void init_crc_table(__global ProgramState *program_state) {
    if (program_state->tables_initialized) return;

    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ CRC32_POLY;
            } else {
                crc = crc >> 1;
            }
        }
        program_state->crc_table[i] = crc;
    }

    /* Generate slice-by-4 tables */
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = program_state->crc_table[i];
        program_state->crc_table_slice[0][i] = crc;
        for (int j = 1; j < 4; j++) {
            crc = (crc >> 8) ^ program_state->crc_table[crc & 0xFF];
            program_state->crc_table_slice[j][i] = crc;
        }
    }

    program_state->tables_initialized = 1;
}

/* Bit-by-bit CRC32 (slow but simple) */
static uint32_t crc32_bitwise(__global ProgramState *program_state, const uint8_t *buf, size_t len) {
    uint32_t crc = 0xFFFFFFFF;

    for (size_t i = 0; i < len; i++) {
        crc ^= buf[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ CRC32_POLY;
            } else {
                crc = crc >> 1;
            }
        }
    }

    return crc ^ 0xFFFFFFFF;
}

/* Table-based CRC32 (standard) */
static uint32_t crc32_table(__global ProgramState *program_state, const uint8_t *buf, size_t len) {
    uint32_t crc = 0xFFFFFFFF;

    for (size_t i = 0; i < len; i++) {
        uint8_t index = (crc ^ buf[i]) & 0xFF;
        crc = (crc >> 8) ^ program_state->crc_table[index];
    }

    return crc ^ 0xFFFFFFFF;
}

/* Slice-by-4 CRC32 (faster) */
static uint32_t crc32_slice4(__global ProgramState *program_state, const uint8_t *buf, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    const uint8_t *p = buf;

    /* Process 4 bytes at a time */
    while (len >= 4) {
        crc ^= ((uint32_t)p[0])
             | ((uint32_t)p[1] << 8)
             | ((uint32_t)p[2] << 16)
             | ((uint32_t)p[3] << 24);

        crc = program_state->crc_table_slice[3][crc & 0xFF]
            ^ program_state->crc_table_slice[2][(crc >> 8) & 0xFF]
            ^ program_state->crc_table_slice[1][(crc >> 16) & 0xFF]
            ^ program_state->crc_table_slice[0][(crc >> 24) & 0xFF];

        p += 4;
        len -= 4;
    }

    /* Process remaining bytes */
    while (len--) {
        crc = (crc >> 8) ^ program_state->crc_table[(crc ^ *p++) & 0xFF];
    }

    return crc ^ 0xFFFFFFFF;
}

/* CRC32C polynomial (used in iSCSI, ext4) */
#define CRC32C_POLY 0x82F63B78



static void init_crc32c_table(__global ProgramState *program_state) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ CRC32C_POLY;
            } else {
                crc = crc >> 1;
            }
        }
        program_state->crc32c_table[i] = crc;
    }
}

static uint32_t crc32c(__global ProgramState *program_state, const uint8_t *buf, size_t len) {
    uint32_t crc = 0xFFFFFFFF;

    for (size_t i = 0; i < len; i++) {
        crc = (crc >> 8) ^ program_state->crc32c_table[(crc ^ buf[i]) & 0xFF];
    }

    return crc ^ 0xFFFFFFFF;
}

/* Combined CRC (for testing) */
static uint32_t combined_crc(__global ProgramState *program_state, const uint8_t *buf, size_t len) {
    uint32_t crc1 = crc32_table(program_state, buf, len);
    uint32_t crc2 = crc32c(program_state, buf, len);
    return crc1 ^ crc2;
}

int program_main(__global ProgramState *program_state) {
    /* Initialize tables */
    init_crc_table(program_state);
    init_crc32c_table(program_state);

    /* Initialize test data */
    for (int i = 0; i < DATA_SIZE; i++) {
        program_state->data[i] = (uint8_t)((i * 17 + 31) % 256);
    }

    uint32_t checksum = 0;

    for (int iter = 0; iter < 1000; iter++) {
        size_t len = (iter % DATA_SIZE) + 1;

        /* Test different implementations */
        uint32_t crc1, crc2, crc3, crc4;

        /* Table-based */
        crc1 = crc32_table(program_state, program_state->data, len);
        checksum ^= crc1;

        /* Slice-by-4 */
        crc2 = crc32_slice4(program_state, program_state->data, len);
        checksum ^= crc2;

        /* CRC32C variant */
        crc3 = crc32c(program_state, program_state->data, len);
        checksum ^= crc3;

        /* Combined */
        crc4 = combined_crc(program_state, program_state->data, len);
        checksum ^= crc4;

        /* Bitwise (only on small data for speed) */
        if (len < 256) {
            uint32_t crc5 = crc32_bitwise(program_state, program_state->data, len);
            checksum ^= crc5;
        }

        /* Modify data based on CRC */
        program_state->data[iter % DATA_SIZE] ^= (uint8_t)(crc1 & 0xFF);
    }

    program_state->sink = checksum;
    return 0;
}

typedef struct { uint results[64]; ProgramState state; } Storage;
__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global Storage *buffer) {
    __global uint *results = buffer->results;
    __global ProgramState *storage = &buffer->state;
    storage->tables_initialized = 0;
    results[0] = (uint)program_main(storage);
    for (uint i = 0; i < sizeof(storage->sink); ++i)
        results[1 + i] = (uint)(((uint)storage->sink >> (8u * i)) & 255u);
}
