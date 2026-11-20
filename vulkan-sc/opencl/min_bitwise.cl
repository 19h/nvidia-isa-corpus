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
 * Minimal Bitwise Operation Chain
 *
 * Generates sequences dominated by AND, OR, XOR, NOT instructions.
 * Bitwise operations are fundamental and present in all ISAs.
 *
 * Expected dominant instructions by architecture:
 *   x86:    and, or, xor, not
 *   ARM:    and, orr, eor, mvn, bic
 *   RISC-V: and, or, xor, (not via xori -1)
 *   MIPS:   and, or, xor, nor
 *   PPC:    and, or, xor, nor, andc
 */



typedef struct {
    volatile unsigned int sink;
} ProgramState;

int program_main(__private ProgramState *program_state) {
    unsigned int a = 0xAAAAAAAA;
    unsigned int b = 0x55555555;
    unsigned int c = 0xFF00FF00;
    unsigned int d = 0x00FF00FF;
    unsigned int e = 0xF0F0F0F0;
    unsigned int f = 0x0F0F0F0F;
    unsigned int g = 0xCCCCCCCC;
    unsigned int h = 0x33333333;

    for (int i = 0; i < 100000; i++) {
        /* AND operations */
        a = a & b;
        c = c & d;
        e = e & f;
        g = g & h;

        /* OR operations */
        a = a | c;
        b = b | d;
        e = e | g;
        f = f | h;

        /* XOR operations */
        a = a ^ e;
        b = b ^ f;
        c = c ^ g;
        d = d ^ h;

        /* NOT operations */
        e = ~a;
        f = ~b;
        g = ~c;
        h = ~d;

        /* Combined operations */
        a = (a & 0xFF) | (b & 0xFF00);
        b = (c ^ d) & 0xFFFF;
        c = (e | f) ^ (g & h);
        d = ~(a ^ b) & (c | d);

        /* Bit clear patterns (AND with complement) */
        e = a & ~0xFF;
        f = b & ~0xFF00;
        g = c & ~0xFF0000;
        h = d & ~0xFF000000;

        /* Reset periodically */
        if (i % 10000 == 0) {
            a = 0xAAAAAAAA;
            b = 0x55555555;
            c = 0xFF00FF00;
            d = 0x00FF00FF;
            e = 0xF0F0F0F0;
            f = 0x0F0F0F0F;
            g = 0xCCCCCCCC;
            h = 0x33333333;
        }
    }

    program_state->sink = a ^ b ^ c ^ d ^ e ^ f ^ g ^ h;
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)(((uint)storage.sink >> (8u * i)) & 255u);
}
