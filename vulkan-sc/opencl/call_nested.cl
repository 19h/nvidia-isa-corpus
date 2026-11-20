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
 * Deeply Nested Function Calls
 *
 * Tests stack frame allocation and unwinding.
 * Exercises prologue/epilogue code generation.
 *
 * Features exercised:
 *   - Deep call stacks
 *   - Stack frame setup/teardown
 *   - Callee-saved preservation
 */





typedef struct {
    volatile int64_t sink;
} ProgramState;


int64_t level_10(__private ProgramState *program_state, int a, int b, int c, int d) {
    return (int64_t)a + b + c + d;
}


int64_t level_9(__private ProgramState *program_state, int a, int b, int c, int d) {
    int local_value = a + b;
    return level_10(program_state, local_value, b + c, c + d, d + a) + local_value;
}


int64_t level_8(__private ProgramState *program_state, int a, int b, int c, int d) {
    int local_value = a * 2;
    return level_9(program_state, local_value, b, c, d) + local_value;
}


int64_t level_7(__private ProgramState *program_state, int a, int b, int c, int d) {
    int local_value = a + 1;
    return level_8(program_state, local_value, b + 1, c + 1, d + 1) + local_value;
}


int64_t level_6(__private ProgramState *program_state, int a, int b, int c, int d) {
    int local1 = a, local2 = b;
    return level_7(program_state, local1, local2, c, d) + local1 + local2;
}


int64_t level_5(__private ProgramState *program_state, int a, int b, int c, int d) {
    int arr[4] = {a, b, c, d};
    return level_6(program_state, arr[0], arr[1], arr[2], arr[3]) + arr[0];
}


int64_t level_4(__private ProgramState *program_state, int a, int b, int c, int d) {
    int local_value = a ^ b ^ c ^ d;
    return level_5(program_state, a + local_value, b, c, d) + local_value;
}


int64_t level_3(__private ProgramState *program_state, int a, int b, int c, int d) {
    return level_4(program_state, a, b, c, d) + level_4(program_state, b, c, d, a);
}


int64_t level_2(__private ProgramState *program_state, int a, int b, int c, int d) {
    int sum = 0;
    for (int i = 0; i < 2; i++) {
        sum += (int)level_3(program_state, a + i, b, c, d);
    }
    return sum;
}


int64_t level_1(__private ProgramState *program_state, int a, int b, int c, int d) {
    return level_2(program_state, a, b, c, d) + a + b + c + d;
}

/* Alternative deep nesting with more local_value variables */

int64_t deep_with_locals(__private ProgramState *program_state, int depth, int val) {
    // The test calls this function at a depth of eight.
    int values[9] = {val};
    int depths[9] = {depth};
    int count = 1;
    int64_t total = 0;
    while (count) {
        int at = --count;
        int value = values[at];
        int remaining = depths[at];
        total += (value ^ 0xff) + (value & 0xf0);
        if (remaining <= 0) {
            total += (value + 1) + value * 2;
        } else {
            values[count] = value * 2;
            depths[count++] = remaining - 1;
            values[count] = value + 1;
            depths[count++] = remaining - 1;
        }
    }
    return total;
}

static int64_t mutual_walk(__private ProgramState *program_state, int depth, int value, bool side_b) {
    int64_t total = 0;
    for (; depth > 0; --depth) {
        total += value;
        value = side_b ? value * 2 : value + 1;
        side_b = !side_b;
    }
    return total + (side_b ? value * 2 : value);
}

int64_t mutual_a(__private ProgramState *program_state, int depth, int val) {
    return mutual_walk(program_state, depth, val, false);
}

int64_t mutual_b(__private ProgramState *program_state, int depth, int val) {
    return mutual_walk(program_state, depth, val, true);
}

int program_main(__private ProgramState *program_state) {
    int64_t result = 0;

    for (int i = 0; i < 10000; i++) {
        result += level_1(program_state, i, i + 1, i + 2, i + 3);

        if (i % 100 == 0) {
            result += deep_with_locals(program_state, 8, i);
        }

        result += mutual_a(program_state, 10, i % 100);
    }

    program_state->sink = result;
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)(((ulong)storage.sink >> (8u * i)) & 255u);
}
