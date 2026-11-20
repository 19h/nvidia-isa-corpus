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
 * Variadic Functions
 *
 * Functions with variable number of arguments.
 * Tests va_list implementation across architectures.
 *
 * Features exercised:
 *   - va_start, va_arg, va_end
 *   - Type promotion in variadic calls
 *   - va_list passing and copying
 */






typedef struct {
    volatile int64_t sink;
} ProgramState;


int sum_ints(__private ProgramState *program_state, int count, ...) {
    va_list args;
    va_start(args, count);

    int sum = 0;
    for (int i = 0; i < count; i++) {
        sum += va_arg(args, int);
    }

    va_end(args);
    return sum;
}


double sum_doubles(__private ProgramState *program_state, int count, ...) {
    va_list args;
    va_start(args, count);

    double sum = 0.0;
    for (int i = 0; i < count; i++) {
        sum += va_arg(args, double);
    }

    va_end(args);
    return sum;
}


int64_t sum_mixed(__private ProgramState *program_state, const char *format, ...) {
    va_list args;
    va_start(args, format);

    int64_t sum = 0;
    while (*format) {
        switch (*format) {
            case 'i':
                sum += va_arg(args, int);
                break;
            case 'd':
                sum += (int64_t)va_arg(args, double);
                break;
            case 'l':
                sum += va_arg(args, int64_t);
                break;
            case 'p':
                sum += (int64_t)(intptr_t)va_arg(args, void *);
                break;
        }
        format++;
    }

    va_end(args);
    return sum;
}

/* va_list as parameter */

int sum_from_va_list(__private ProgramState *program_state, int count, va_list args) {
    int sum = 0;
    for (int i = 0; i < count; i++) {
        sum += va_arg(args, int);
    }
    return sum;
}


int wrapper_sum(__private ProgramState *program_state, int count, ...) {
    va_list args;
    va_start(args, count);
    int result = sum_from_va_list(program_state, count, args);
    va_end(args);
    return result;
}

/* va_copy */

int double_sum(__private ProgramState *program_state, int count, ...) {
    va_list args, args_copy;
    va_start(args, count);
    va_copy(args_copy, args);

    int sum1 = 0, sum2 = 0;
    for (int i = 0; i < count; i++) {
        sum1 += va_arg(args, int);
    }
    for (int i = 0; i < count; i++) {
        sum2 += va_arg(args_copy, int);
    }

    va_end(args);
    va_end(args_copy);
    return sum1 + sum2;
}

/* Variadic with struct return */
typedef struct { int a, b; } Pair;


Pair min_max(__private ProgramState *program_state, int count, ...) {
    va_list args;
    va_start(args, count);

    int min_val = va_arg(args, int);
    int max_val = min_val;

    for (int i = 1; i < count; i++) {
        int v = va_arg(args, int);
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    va_end(args);
    Pair p = {min_val, max_val};
    return p;
}

int program_main(__private ProgramState *program_state) {
    int64_t result = 0;

    for (int i = 0; i < 50000; i++) {
        result += sum_ints(program_state, 4, i, i+1, i+2, i+3);
        result += sum_ints(program_state, 8, i, i+1, i+2, i+3, i+4, i+5, i+6, i+7);

        double d = (double)i;
        result += (int64_t)sum_doubles(program_state, 4, d, d+0.1, d+0.2, d+0.3);

        result += sum_mixed(program_state, "iidl", i, i+1, d, (int64_t)i*1000);

        result += wrapper_sum(program_state, 6, i, i+1, i+2, i+3, i+4, i+5);

        result += double_sum(program_state, 4, i, i+1, i+2, i+3);

        Pair p = min_max(program_state, 5, i, i+10, i-5, i+3, i+7);
        result += p.a + p.b;
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
