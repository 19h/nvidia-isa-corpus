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
 * Sequential Memory Access Patterns
 *
 * Tests linear memory traversal patterns.
 * Exercises prefetching and cache line utilization.
 *
 * Features exercised:
 *   - Sequential reads and writes
 *   - Stream processing
 *   - Cache-friendly access patterns
 *   - Memory bandwidth utilization
 */




#define ARRAY_SIZE 4096









/* Sequential read */
typedef struct {
    int64_t sink;
    int32_t array_a[ARRAY_SIZE];
    int32_t array_b[ARRAY_SIZE];
    int32_t array_c[ARRAY_SIZE];
    double array_d[ARRAY_SIZE];
    double array_e[ARRAY_SIZE];
} ProgramState;


int64_t sequential_read(__global ProgramState *program_state, const int32_t *arr, size_t len) {
    int64_t sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum += arr[i];
    }
    return sum;
}

/* Sequential write */

void sequential_write(__global ProgramState *program_state, int32_t *arr, size_t len, int32_t value) {
    for (size_t i = 0; i < len; i++) {
        arr[i] = value + (int32_t)i;
    }
}

/* Sequential copy */

void sequential_copy(__global ProgramState *program_state, int32_t *dst, const int32_t *src, size_t len) {
    for (size_t i = 0; i < len; i++) {
        dst[i] = src[i];
    }
}

/* Sequential add (a = b + c) */

void sequential_add(__global ProgramState *program_state, int32_t *a, const int32_t *b, const int32_t *c, size_t len) {
    for (size_t i = 0; i < len; i++) {
        a[i] = b[i] + c[i];
    }
}

/* Sequential scale (a = b * scalar) */

void sequential_scale(__global ProgramState *program_state, int32_t *a, const int32_t *b, int32_t scalar, size_t len) {
    for (size_t i = 0; i < len; i++) {
        a[i] = b[i] * scalar;
    }
}

/* Sequential FMA (a = b * c + d) */

void sequential_fma(__global ProgramState *program_state, double *a, const double *b, const double *c, const double *d, size_t len) {
    for (size_t i = 0; i < len; i++) {
        a[i] = b[i] * c[i] + d[i];
    }
}

/* Triad (a = b + scalar * c) - classic bandwidth test */

void sequential_triad(__global ProgramState *program_state, double *a, const double *b, const double *c, double scalar, size_t len) {
    for (size_t i = 0; i < len; i++) {
        a[i] = b[i] + scalar * c[i];
    }
}

/* Sequential reduction */

int32_t sequential_reduce_min(__global ProgramState *program_state, const int32_t *arr, size_t len) {
    int32_t min_val = arr[0];
    for (size_t i = 1; i < len; i++) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
    }
    return min_val;
}


int32_t sequential_reduce_max(__global ProgramState *program_state, const int32_t *arr, size_t len) {
    int32_t max_val = arr[0];
    for (size_t i = 1; i < len; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return max_val;
}

/* Sequential search */

int sequential_find(__global ProgramState *program_state, const int32_t *arr, size_t len, int32_t target) {
    for (size_t i = 0; i < len; i++) {
        if (arr[i] == target) {
            return (int)i;
        }
    }
    return -1;
}

/* Sequential count */

size_t sequential_count(__global ProgramState *program_state, const int32_t *arr, size_t len, int32_t value) {
    size_t count = 0;
    for (size_t i = 0; i < len; i++) {
        if (arr[i] == value) {
            count++;
        }
    }
    return count;
}

/* Sequential filter (copy matching elements) */

size_t sequential_filter(__global ProgramState *program_state, int32_t *dst, const int32_t *src, size_t len, int32_t threshold) {
    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        if (src[i] > threshold) {
            dst[j++] = src[i];
        }
    }
    return j;
}

/* Sequential transform */

void sequential_transform(__global ProgramState *program_state, int32_t *arr, size_t len) {
    for (size_t i = 0; i < len; i++) {
        int32_t x = arr[i];
        arr[i] = ((x * 3) >> 1) + (x & 0xFF);
    }
}

/* Sequential dot product */

double sequential_dot(__global ProgramState *program_state, const double *a, const double *b, size_t len) {
    double sum = 0.0;
    for (size_t i = 0; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/* Sequential accumulate with dependency */

int64_t sequential_dependent(__global ProgramState *program_state, const int32_t *arr, size_t len) {
    int64_t acc = 0;
    for (size_t i = 0; i < len; i++) {
        acc = acc * 3 + arr[i];
    }
    return acc;
}

/* Multiple sequential passes */

void multi_pass(__global ProgramState *program_state, int32_t *arr, size_t len, int passes) {
    for (int p = 0; p < passes; p++) {
        for (size_t i = 0; i < len; i++) {
            arr[i] = arr[i] + p;
        }
    }
}

/* Forward and backward sequential access */

int64_t forward_backward(__global ProgramState *program_state, const int32_t *arr, size_t len) {
    int64_t sum = 0;

    /* Forward pass */
    for (size_t i = 0; i < len; i++) {
        sum += arr[i];
    }

    /* Backward pass */
    for (size_t i = len; i > 0; i--) {
        sum += arr[i - 1];
    }

    return sum;
}

int program_main(__global ProgramState *program_state) {
    int64_t result = 0;

    /* Initialize arrays */
    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        program_state->array_a[i] = (int32_t)i;
        program_state->array_b[i] = (int32_t)(i * 2);
        program_state->array_c[i] = (int32_t)(i % 100);
        program_state->array_d[i] = (double)i * 0.1;
        program_state->array_e[i] = (double)i * 0.2;
    }

    for (int iter = 0; iter < 1000; iter++) {
        /* Sequential read */
        result += sequential_read(program_state, program_state->array_a, ARRAY_SIZE);

        /* Sequential write */
        sequential_write(program_state, program_state->array_b, ARRAY_SIZE, iter);

        /* Sequential copy */
        sequential_copy(program_state, program_state->array_c, program_state->array_a, ARRAY_SIZE);

        /* Sequential add */
        sequential_add(program_state, program_state->array_a, program_state->array_b, program_state->array_c, ARRAY_SIZE);

        /* Sequential scale */
        sequential_scale(program_state, program_state->array_c, program_state->array_b, 3, ARRAY_SIZE);

        /* Sequential FMA */
        sequential_fma(program_state, program_state->array_d, program_state->array_d, program_state->array_e, program_state->array_e, ARRAY_SIZE);

        /* Sequential triad */
        sequential_triad(program_state, program_state->array_e, program_state->array_d, program_state->array_e, 2.5, ARRAY_SIZE);

        /* Reductions */
        result += sequential_reduce_min(program_state, program_state->array_a, ARRAY_SIZE);
        result += sequential_reduce_max(program_state, program_state->array_a, ARRAY_SIZE);

        /* Search */
        result += sequential_find(program_state, program_state->array_a, ARRAY_SIZE, iter % ARRAY_SIZE);

        /* Count */
        result += (int64_t)sequential_count(program_state, program_state->array_c, ARRAY_SIZE, iter % 100);

        /* Filter */
        result += (int64_t)sequential_filter(program_state, program_state->array_b, program_state->array_a, ARRAY_SIZE, ARRAY_SIZE / 2);

        /* Transform */
        sequential_transform(program_state, program_state->array_c, ARRAY_SIZE);

        /* Dot product */
        double dot = sequential_dot(program_state, program_state->array_d, program_state->array_e, ARRAY_SIZE);
        // Match the native x86 reference for overflowing float-to-int conversion.
        result += (dot >= 0x1p63 || dot < -0x1p63) ? INT64_MIN : (int64_t)dot;

        /* Dependent accumulate */
        result += sequential_dependent(program_state, program_state->array_c, 256);

        /* Multi-pass */
        if (iter % 100 == 0) {
            multi_pass(program_state, program_state->array_b, ARRAY_SIZE, 3);
        }

        /* Forward/backward */
        result += forward_backward(program_state, program_state->array_a, 512);
    }

    program_state->sink = result;
    return 0;
}

typedef struct { uint results[64]; ProgramState state; } Storage;
__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global Storage *buffer) {
    __global uint *results = buffer->results;
    __global ProgramState *storage = &buffer->state;
    results[0] = (uint)program_main(storage);
    for (uint i = 0; i < sizeof(storage->sink); ++i)
        results[1 + i] = (uint)(((ulong)storage->sink >> (8u * i)) & 255u);
}
