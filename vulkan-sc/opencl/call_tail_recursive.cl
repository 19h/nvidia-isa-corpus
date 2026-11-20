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
 * Tail Recursive Functions
 *
 * Functions that can be optimized via tail call optimization (TCO).
 * When TCO is applied, no stack growth occurs.
 *
 * Features exercised:
 *   - Tail call optimization detection
 *   - Jump vs call instruction selection
 *   - Accumulator pattern
 */





/* Classic factorial with accumulator - tail recursive */
typedef struct {
    volatile int64_t sink;
} ProgramState;


int64_t factorial_tail(__private ProgramState *program_state, int n, int64_t acc) {
    if (n <= 1) return acc;
    return factorial_tail(program_state, n - 1, acc * n);
}


int64_t factorial(__private ProgramState *program_state, int n) {
    return factorial_tail(program_state, n, 1);
}

/* Fibonacci with accumulators - tail recursive */

int64_t fib_tail(__private ProgramState *program_state, int n, int64_t a, int64_t b) {
    if (n == 0) return a;
    if (n == 1) return b;
    return fib_tail(program_state, n - 1, b, a + b);
}


int64_t fibonacci(__private ProgramState *program_state, int n) {
    return fib_tail(program_state, n, 0, 1);
}

/* Sum with accumulator */

int64_t sum_tail(__private ProgramState *program_state, int n, int64_t acc) {
    if (n <= 0) return acc;
    return sum_tail(program_state, n - 1, acc + n);
}

/* GCD - naturally tail recursive */

int64_t gcd(__private ProgramState *program_state, int64_t a, int64_t b) {
    if (b == 0) return a;
    return gcd(program_state, b, a % b);
}

/* List-like iteration with accumulator */

int64_t iterate_tail(__private ProgramState *program_state, const int *arr, int len, int64_t acc) {
    if (len <= 0) return acc;
    return iterate_tail(program_state, arr + 1, len - 1, acc + arr[0]);
}

/* Power with accumulator */

int64_t power_tail(__private ProgramState *program_state, int64_t base, int exp, int64_t acc) {
    if (exp == 0) return acc;
    if (exp % 2 == 0) {
        return power_tail(program_state, base * base, exp / 2, acc);
    }
    return power_tail(program_state, base, exp - 1, acc * base);
}

/* Even/odd mutual recursion - can be TCO'd */
 int is_even_tail(__private ProgramState *program_state, int n);
 int is_odd_tail(__private ProgramState *program_state, int n);

int is_even_tail(__private ProgramState *program_state, int n) {
    if (n == 0) return 1;
    return is_odd_tail(program_state, n - 1);
}

int is_odd_tail(__private ProgramState *program_state, int n) {
    if (n == 0) return 0;
    return is_even_tail(program_state, n - 1);
}

/* State machine as tail recursion */
typedef enum { STATE_A, STATE_B, STATE_C, STATE_DONE } State;


int64_t state_machine(__private ProgramState *program_state, State state, int counter, int64_t acc) {
    switch (state) {
        case STATE_A:
            if (counter <= 0) return state_machine(program_state, STATE_DONE, 0, acc);
            return state_machine(program_state, STATE_B, counter, acc + 1);
        case STATE_B:
            return state_machine(program_state, STATE_C, counter, acc * 2);
        case STATE_C:
            return state_machine(program_state, STATE_A, counter - 1, acc + 3);
        case STATE_DONE:
        default:
            return acc;
    }
}

int program_main(__private ProgramState *program_state) {
    int64_t result = 0;
    int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    for (int i = 0; i < 10000; i++) {
        result += factorial(program_state, 12);

        result += fibonacci(program_state, 30);

        result += sum_tail(program_state, 100, 0);

        result += gcd(program_state, i + 1000, (i % 100) + 1);

        result += iterate_tail(program_state, arr, 10, 0);

        result += power_tail(program_state, 2, i % 20, 1);

        result += is_even_tail(program_state, i % 100);
        result += is_odd_tail(program_state, i % 100);

        if (i % 100 == 0) {
            result += state_machine(program_state, STATE_A, 10, 0);
        }
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
