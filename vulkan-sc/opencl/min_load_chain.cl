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
 * Minimal Memory Load Chain
 *
 * Generates sequences dominated by memory load instructions.
 * Creates pointer-chasing pattern that forces sequential loads.
 *
 * Expected dominant instructions by architecture:
 *   x86:    mov (memory operand), movzx, movsx
 *   ARM:    ldr, ldrb, ldrh
 *   RISC-V: lw, ld, lb, lh
 *   MIPS:   lw, ld, lb, lh
 *   PPC:    lwz, ld, lbz, lhz
 */

#define ARRAY_SIZE 1024



/* Linked list node for pointer chasing */
struct node {
    int next;
    int value;
};




typedef struct {
    int sink;
    struct node nodes[ARRAY_SIZE];
    int data[ARRAY_SIZE];
} ProgramState;

int program_main(__global ProgramState *program_state) {
    /* Initialize data array */
    for (int i = 0; i < ARRAY_SIZE; i++) {
        program_state->data[i] = i * 7 + 3;
    }

    /* Initialize linked list with pseudo-random order */
    for (int i = 0; i < ARRAY_SIZE; i++) {
        int next_idx = (i * 17 + 31) % ARRAY_SIZE;
        program_state->nodes[i].next = next_idx;
        program_state->nodes[i].value = i;
    }

    int sum = 0;
    int cursor = 0;

    /* Pointer chasing - forces sequential loads */
    for (int i = 0; i < 100000; i++) {
        /* Load from linked list (pointer chase) */
        sum += program_state->nodes[cursor].value;
        cursor = program_state->nodes[cursor].next;

        sum += program_state->nodes[cursor].value;
        cursor = program_state->nodes[cursor].next;

        sum += program_state->nodes[cursor].value;
        cursor = program_state->nodes[cursor].next;

        sum += program_state->nodes[cursor].value;
        cursor = program_state->nodes[cursor].next;

        /* Array loads with varying offsets */
        int idx = (i * 13) % ARRAY_SIZE;
        sum += program_state->data[idx];
        sum += program_state->data[(idx + 1) % ARRAY_SIZE];
        sum += program_state->data[(idx + 2) % ARRAY_SIZE];
        sum += program_state->data[(idx + 3) % ARRAY_SIZE];
    }

    program_state->sink = sum;
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
