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
 * Minimal Shift and Rotate Chain
 *
 * Generates sequences dominated by shift and rotate instructions.
 * Important for detecting ISAs - shift encodings vary significantly.
 *
 * Expected dominant instructions by architecture:
 *   x86:    shl, shr, sar, rol, ror
 *   ARM:    lsl, lsr, asr, ror
 *   RISC-V: sll, srl, sra, (no rotate in base ISA)
 *   MIPS:   sll, srl, sra, rotr
 *   PPC:    slw, srw, sraw, rlwnm
 */



/* Rotate left (portable) */
typedef struct {
    volatile unsigned int sink;
} ProgramState;

static inline unsigned int rotl(__private ProgramState *program_state, unsigned int x, int n) {
    return (x << n) | (x >> (32 - n));
}

/* Rotate right (portable) */
static inline unsigned int rotr(__private ProgramState *program_state, unsigned int x, int n) {
    return (x >> n) | (x << (32 - n));
}

int program_main(__private ProgramState *program_state) {
    unsigned int a = 0xDEADBEEF;
    unsigned int b = 0xCAFEBABE;
    unsigned int c = 0x12345678;
    unsigned int d = 0x87654321;

    for (int i = 0; i < 100000; i++) {
        /* Left shifts */
        a = a << 1;
        b = b << 3;
        c = c << 5;
        d = d << 7;

        /* Right shifts (logical) */
        a = a >> 2;
        b = b >> 4;
        c = c >> 6;
        d = d >> 1;

        /* Variable shifts */
        int shift_amt = (i & 15) + 1;
        a = a << shift_amt;
        b = b >> shift_amt;

        /* Arithmetic right shifts (signed) */
        int sa = (int)a;
        int sb = (int)b;
        sa = sa >> 3;
        sb = sb >> 5;
        a = (unsigned int)sa;
        b = (unsigned int)sb;

        /* Rotates */
        c = rotl(program_state, c, 7);
        d = rotr(program_state, d, 11);
        a = rotl(program_state, a, shift_amt);
        b = rotr(program_state, b, shift_amt);

        /* Combined patterns */
        a = (a << 4) | (b >> 28);
        c = (c >> 8) | (d << 24);

        /* Reset periodically */
        if (i % 10000 == 0) {
            a = 0xDEADBEEF;
            b = 0xCAFEBABE;
            c = 0x12345678;
            d = 0x87654321;
        }
    }

    program_state->sink = a ^ b ^ c ^ d;
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)(((uint)storage.sink >> (8u * i)) & 255u);
}
