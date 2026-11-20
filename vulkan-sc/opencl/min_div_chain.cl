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
 * Minimal Integer Division Chain
 *
 * Generates sequences dominated by integer division and modulo.
 * Division encoding varies significantly across ISAs.
 *
 * Expected dominant instructions by architecture:
 *   x86:    div, idiv
 *   ARM:    udiv, sdiv
 *   RISC-V: div, divu, rem, remu (M extension)
 *   MIPS:   div, divu
 *   PPC:    divw, divwu
 */



typedef struct {
    volatile int sink;
} ProgramState;

int program_main(__private ProgramState *program_state) {
    unsigned int ua = 1000000;
    unsigned int ub = 7;
    unsigned int uc = 13;
    unsigned int ud = 17;

    int sa = 1000000;
    int sb = -7;
    int sc = 13;
    int sd = -17;

    int sum = 0;

    for (int i = 0; i < 50000; i++) {
        /* Unsigned division */
        unsigned int q1 = ua / ub;
        unsigned int r1 = ua % ub;
        unsigned int q2 = uc / ud;
        unsigned int r2 = uc % ud;

        sum += (int)(q1 + r1 + q2 + r2);

        /* Signed division */
        int q3 = sa / sb;
        int r3 = sa % sb;
        int q4 = sc / sd;
        int r4 = sc % sd;

        sum += q3 + r3 + q4 + r4;

        /* Division by varying values */
        int divisor = (i % 15) + 2;  /* Avoid div by 0/1 */
        sum += (int)ua / divisor;
        sum += sa / divisor;

        /* Update operands */
        ua = (ua * 7 + 11) % 10000000 + 1000;
        ub = (ub * 3 + 5) % 100 + 2;
        uc = (uc * 11 + 7) % 10000000 + 1000;
        ud = (ud * 13 + 3) % 100 + 2;

        sa = ((sa * 7 + 11) % 10000000) - 5000000;
        sb = ((int)((i * 3) % 98) + 2) * ((i & 1) ? 1 : -1);
        sc = ((sc * 11 + 7) % 10000000) - 5000000;
        sd = ((int)((i * 7) % 98) + 2) * ((i & 2) ? 1 : -1);

        /* Ensure non-zero divisors */
        if (sb == 0) sb = 1;
        if (sd == 0) sd = 1;
    }

    program_state->sink = sum;
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)(((uint)storage.sink >> (8u * i)) & 255u);
}
