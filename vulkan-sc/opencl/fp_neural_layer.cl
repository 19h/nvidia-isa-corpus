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
 * Neural Network Dense Layer
 *
 * Computes y = activation(W * x + b) for a fully-connected layer.
 * Heavy matrix-vector multiply with activation function.
 *
 * Features exercised:
 *   - FP multiply-accumulate (FMA)
 *   - Vectorizable loops
 *   - Non-linear activation (ReLU, tanh approximation)
 */



#define INPUT_SIZE 256
#define OUTPUT_SIZE 128
#define BATCH_SIZE 32








/* ReLU activation */
typedef struct {
    float sink;
    float weights[OUTPUT_SIZE][INPUT_SIZE];
    float bias[OUTPUT_SIZE];
    float input[BATCH_SIZE][INPUT_SIZE];
    float output[BATCH_SIZE][OUTPUT_SIZE];
} ProgramState;

static inline float relu(__global ProgramState *program_state, float x) {
    return x > 0.0f ? x : 0.0f;
}

/* Approximate tanh using rational approximation */
static inline float approx_tanh(__global ProgramState *program_state, float x) {
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;

    float x2 = x * x;
    float num = x * (27.0f + x2);
    float den = 27.0f + 9.0f * x2;
    return num / den;
}

/* Sigmoid approximation */
static inline float approx_sigmoid(__global ProgramState *program_state, float x) {
    return 0.5f * (approx_tanh(program_state, x * 0.5f) + 1.0f);
}

/* Dense layer forward pass with ReLU */
static void dense_relu(__global ProgramState *program_state, const float *x, float *y,
                       const __global float w[OUTPUT_SIZE][INPUT_SIZE],
                       const float *b) {
    for (size_t i = 0; i < OUTPUT_SIZE; i++) {
        float sum = b[i];
        for (size_t j = 0; j < INPUT_SIZE; j++) {
            sum += w[i][j] * x[j];
        }
        y[i] = relu(program_state, sum);
    }
}

/* Dense layer forward pass with tanh */
static void dense_tanh(__global ProgramState *program_state, const float *x, float *y,
                       const __global float w[OUTPUT_SIZE][INPUT_SIZE],
                       const float *b) {
    for (size_t i = 0; i < OUTPUT_SIZE; i++) {
        float sum = b[i];
        for (size_t j = 0; j < INPUT_SIZE; j++) {
            sum += w[i][j] * x[j];
        }
        y[i] = approx_tanh(program_state, sum);
    }
}

/* Batch forward pass */
static void batch_forward_relu(__global ProgramState *program_state) {
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        dense_relu(program_state, program_state->input[batch], program_state->output[batch], program_state->weights, program_state->bias);
    }
}

static void batch_forward_tanh(__global ProgramState *program_state) {
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        dense_tanh(program_state, program_state->input[batch], program_state->output[batch], program_state->weights, program_state->bias);
    }
}

/* Initialize with pseudo-random values */
static void init_weights(__global ProgramState *program_state) {
    /* Xavier initialization approximation */
    float scale = 0.1f;
    unsigned int seed = 12345;

    for (size_t i = 0; i < OUTPUT_SIZE; i++) {
        for (size_t j = 0; j < INPUT_SIZE; j++) {
            seed = seed * 1103515245 + 12345;
            int rand_val = (int)((seed >> 16) & 0x7FFF) - 16384;
            program_state->weights[i][j] = scale * (float)rand_val / 16384.0f;
        }
    }

    for (size_t i = 0; i < OUTPUT_SIZE; i++) {
        seed = seed * 1103515245 + 12345;
        int rand_val = (int)((seed >> 16) & 0x7FFF) - 16384;
        program_state->bias[i] = 0.01f * (float)rand_val / 16384.0f;
    }
}

static void init_input(__global ProgramState *program_state, int iter) {
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        for (size_t i = 0; i < INPUT_SIZE; i++) {
            /* Varying input based on iteration */
            program_state->input[batch][i] = 0.1f * (float)((batch * INPUT_SIZE + i + iter) % 100);
        }
    }
}

int program_main(__global ProgramState *program_state) {
    init_weights(program_state);

    float result_sum = 0.0f;

    for (int iter = 0; iter < 1000; iter++) {
        init_input(program_state, iter);

        /* Alternate between activation functions */
        if (iter % 2 == 0) {
            batch_forward_relu(program_state);
        } else {
            batch_forward_tanh(program_state);
        }

        /* Accumulate outputs */
        for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
            for (size_t i = 0; i < OUTPUT_SIZE; i++) {
                result_sum += program_state->output[batch][i];
            }
        }
    }

    program_state->sink = result_sum;
    return 0;
}

typedef struct { uint results[64]; ProgramState state; } Storage;
__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global Storage *buffer) {
    __global uint *results = buffer->results;
    __global ProgramState *storage = &buffer->state;
    results[0] = (uint)program_main(storage);
    for (uint i = 0; i < sizeof(storage->sink); ++i)
        results[1 + i] = (uint)((as_uint(storage->sink) >> (8u * i)) & 255u);
}
