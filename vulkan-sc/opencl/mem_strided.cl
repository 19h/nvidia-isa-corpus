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
 * Strided Memory Access Patterns
 *
 * Tests non-contiguous memory access with regular strides.
 * Exercises cache behavior with various stride patterns.
 *
 * Features exercised:
 *   - Power-of-2 strides
 *   - Non-power-of-2 strides
 *   - Column-major access
 *   - Struct member access
 */




#define ARRAY_SIZE 4096
#define MATRIX_DIM 64






/* Structure with padding - accessing single field has stride */
typedef struct {
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t w;
} Vec4;

typedef struct {
    int32_t value;
    char padding[60]; /* Force 64-byte stride */
} PaddedInt;




/* Fixed stride access */
typedef struct {
    int64_t sink;
    int32_t linear_array[ARRAY_SIZE];
    int32_t matrix[MATRIX_DIM][MATRIX_DIM];
    Vec4 vec_array[ARRAY_SIZE / 4];
    PaddedInt padded_array[1024];
} ProgramState;


int64_t stride_2(__global ProgramState *program_state, const int32_t *arr, size_t len) {
    int64_t sum = 0;
    for (size_t i = 0; i < len; i += 2) {
        sum += arr[i];
    }
    return sum;
}


int64_t stride_4(__global ProgramState *program_state, const int32_t *arr, size_t len) {
    int64_t sum = 0;
    for (size_t i = 0; i < len; i += 4) {
        sum += arr[i];
    }
    return sum;
}


int64_t stride_8(__global ProgramState *program_state, const int32_t *arr, size_t len) {
    int64_t sum = 0;
    for (size_t i = 0; i < len; i += 8) {
        sum += arr[i];
    }
    return sum;
}


int64_t stride_16(__global ProgramState *program_state, const int32_t *arr, size_t len) {
    int64_t sum = 0;
    for (size_t i = 0; i < len; i += 16) {
        sum += arr[i];
    }
    return sum;
}


int64_t stride_64(__global ProgramState *program_state, const int32_t *arr, size_t len) {
    int64_t sum = 0;
    for (size_t i = 0; i < len; i += 64) {
        sum += arr[i];
    }
    return sum;
}

/* Variable stride */

int64_t stride_n(__global ProgramState *program_state, const int32_t *arr, size_t len, size_t stride) {
    int64_t sum = 0;
    for (size_t i = 0; i < len; i += stride) {
        sum += arr[i];
    }
    return sum;
}

/* Non-power-of-2 strides */

int64_t stride_3(__global ProgramState *program_state, const int32_t *arr, size_t len) {
    int64_t sum = 0;
    for (size_t i = 0; i < len; i += 3) {
        sum += arr[i];
    }
    return sum;
}


int64_t stride_5(__global ProgramState *program_state, const int32_t *arr, size_t len) {
    int64_t sum = 0;
    for (size_t i = 0; i < len; i += 5) {
        sum += arr[i];
    }
    return sum;
}


int64_t stride_7(__global ProgramState *program_state, const int32_t *arr, size_t len) {
    int64_t sum = 0;
    for (size_t i = 0; i < len; i += 7) {
        sum += arr[i];
    }
    return sum;
}

/* Column-major access (stride = row size) */

int64_t column_access(__global ProgramState *program_state, const __global int32_t mat[MATRIX_DIM][MATRIX_DIM], size_t col) {
    int64_t sum = 0;
    for (size_t row = 0; row < MATRIX_DIM; row++) {
        sum += mat[row][col];
    }
    return sum;
}

/* Diagonal access */

int64_t diagonal_access(__global ProgramState *program_state, const __global int32_t mat[MATRIX_DIM][MATRIX_DIM]) {
    int64_t sum = 0;
    for (size_t i = 0; i < MATRIX_DIM; i++) {
        sum += mat[i][i];
    }
    return sum;
}

/* Anti-diagonal access */

int64_t anti_diagonal_access(__global ProgramState *program_state, const __global int32_t mat[MATRIX_DIM][MATRIX_DIM]) {
    int64_t sum = 0;
    for (size_t i = 0; i < MATRIX_DIM; i++) {
        sum += mat[i][MATRIX_DIM - 1 - i];
    }
    return sum;
}

/* Struct field access - accessing x gives stride of sizeof(Vec4) */

int64_t struct_field_x(__global ProgramState *program_state, const Vec4 *arr, size_t len) {
    int64_t sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum += arr[i].x;
    }
    return sum;
}


int64_t struct_field_y(__global ProgramState *program_state, const Vec4 *arr, size_t len) {
    int64_t sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum += arr[i].y;
    }
    return sum;
}


int64_t struct_field_z(__global ProgramState *program_state, const Vec4 *arr, size_t len) {
    int64_t sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum += arr[i].z;
    }
    return sum;
}

/* Padded struct access - large stride */

int64_t padded_access(__global ProgramState *program_state, const PaddedInt *arr, size_t len) {
    int64_t sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum += arr[i].value;
    }
    return sum;
}

/* Interleaved write with stride */

void strided_write(__global ProgramState *program_state, int32_t *arr, size_t len, size_t stride, int32_t value) {
    for (size_t i = 0; i < len; i += stride) {
        arr[i] = value + (int32_t)(i / stride);
    }
}

/* Gather operation - indirect strided access */

int64_t gather_strided(__global ProgramState *program_state, const int32_t *arr, const size_t *indices, size_t count) {
    int64_t sum = 0;
    for (size_t i = 0; i < count; i++) {
        sum += arr[indices[i]];
    }
    return sum;
}

/* Scatter operation */

void scatter_strided(__global ProgramState *program_state, int32_t *arr, const size_t *indices, const int32_t *values, size_t count) {
    for (size_t i = 0; i < count; i++) {
        arr[indices[i]] = values[i];
    }
}

/* Matrix transpose (strided read, sequential write per column) */

void transpose_matrix(__global ProgramState *program_state, int32_t dst[MATRIX_DIM][MATRIX_DIM],
                      const __global int32_t src[MATRIX_DIM][MATRIX_DIM]) {
    for (size_t i = 0; i < MATRIX_DIM; i++) {
        for (size_t j = 0; j < MATRIX_DIM; j++) {
            dst[j][i] = src[i][j];
        }
    }
}

/* Blocked/tiled strided access */

int64_t blocked_stride(__global ProgramState *program_state, const int32_t *arr, size_t len, size_t block_size, size_t stride) {
    int64_t sum = 0;
    for (size_t block = 0; block < len; block += block_size * stride) {
        for (size_t i = 0; i < block_size && block + i * stride < len; i++) {
            sum += arr[block + i * stride];
        }
    }
    return sum;
}

/* Bidirectional strided access */

int64_t bidirectional_stride(__global ProgramState *program_state, const int32_t *arr, size_t len, size_t stride) {
    int64_t sum = 0;

    /* Forward */
    for (size_t i = 0; i < len; i += stride) {
        sum += arr[i];
    }

    /* Backward */
    for (size_t i = ((len - 1) / stride) * stride; ; i -= stride) {
        sum += arr[i];
        if (i < stride) break;
    }

    return sum;
}

int program_main(__global ProgramState *program_state) {
    int64_t result = 0;

    /* Initialize arrays */
    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        program_state->linear_array[i] = (int32_t)i;
    }

    for (size_t i = 0; i < MATRIX_DIM; i++) {
        for (size_t j = 0; j < MATRIX_DIM; j++) {
            program_state->matrix[i][j] = (int32_t)(i * MATRIX_DIM + j);
        }
    }

    for (size_t i = 0; i < ARRAY_SIZE / 4; i++) {
        program_state->vec_array[i].x = (int32_t)i;
        program_state->vec_array[i].y = (int32_t)(i * 2);
        program_state->vec_array[i].z = (int32_t)(i * 3);
        program_state->vec_array[i].w = (int32_t)(i * 4);
    }

    for (size_t i = 0; i < 1024; i++) {
        program_state->padded_array[i].value = (int32_t)i;
    }

    /* Prepare indices for gather/scatter */
    size_t indices[256];
    int32_t values[256];
    for (size_t i = 0; i < 256; i++) {
        indices[i] = i * 16;
        values[i] = (int32_t)(i * 100);
    }

    int32_t transposed[MATRIX_DIM][MATRIX_DIM];

    for (int iter = 0; iter < 2000; iter++) {
        /* Power-of-2 strides */
        result += stride_2(program_state, program_state->linear_array, ARRAY_SIZE);
        result += stride_4(program_state, program_state->linear_array, ARRAY_SIZE);
        result += stride_8(program_state, program_state->linear_array, ARRAY_SIZE);
        result += stride_16(program_state, program_state->linear_array, ARRAY_SIZE);
        result += stride_64(program_state, program_state->linear_array, ARRAY_SIZE);

        /* Variable stride */
        result += stride_n(program_state, program_state->linear_array, ARRAY_SIZE, (iter % 8) + 1);

        /* Non-power-of-2 strides */
        result += stride_3(program_state, program_state->linear_array, ARRAY_SIZE);
        result += stride_5(program_state, program_state->linear_array, ARRAY_SIZE);
        result += stride_7(program_state, program_state->linear_array, ARRAY_SIZE);

        /* Matrix column access */
        result += column_access(program_state, program_state->matrix, iter % MATRIX_DIM);

        /* Diagonal accesses */
        result += diagonal_access(program_state, program_state->matrix);
        result += anti_diagonal_access(program_state, program_state->matrix);

        /* Struct field access */
        result += struct_field_x(program_state, program_state->vec_array, ARRAY_SIZE / 4);
        result += struct_field_y(program_state, program_state->vec_array, ARRAY_SIZE / 4);
        result += struct_field_z(program_state, program_state->vec_array, ARRAY_SIZE / 4);

        /* Padded access */
        result += padded_access(program_state, program_state->padded_array, 1024);

        /* Strided write */
        strided_write(program_state, program_state->linear_array, ARRAY_SIZE, 4, iter);

        /* Gather/scatter */
        result += gather_strided(program_state, program_state->linear_array, indices, 256);
        scatter_strided(program_state, program_state->linear_array, indices, values, 256);

        /* Matrix transpose */
        if (iter % 100 == 0) {
            transpose_matrix(program_state, transposed, program_state->matrix);
            result += transposed[0][0] + transposed[MATRIX_DIM-1][MATRIX_DIM-1];
        }

        /* Blocked stride */
        result += blocked_stride(program_state, program_state->linear_array, ARRAY_SIZE, 8, 4);

        /* Bidirectional */
        result += bidirectional_stride(program_state, program_state->linear_array, ARRAY_SIZE, 8);
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
