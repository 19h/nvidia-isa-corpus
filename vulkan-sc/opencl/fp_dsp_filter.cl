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
 * Digital Signal Processing Filter
 *
 * Implements IIR and FIR filters for audio/signal processing.
 * Heavy use of multiply-accumulate operations.
 *
 * Features exercised:
 *   - FP multiply-accumulate (FMA)
 *   - Delay line / circular buffer operations
 *   - Loop unrolling friendly patterns
 */



#define SIGNAL_LENGTH 4096
#define FIR_TAPS 64
#define IIR_ORDER 4






/* FIR filter coefficients (low-pass) */



/* IIR filter coefficients (biquad cascade) */
typedef struct {
    float b0, b1, b2;  /* Feedforward coefficients */
    float a1, a2;      /* Feedback coefficients */
    float z1, z2;      /* State variables */
} Biquad;



/* Initialize FIR filter with sinc-based low-pass */
typedef struct {
    float sink;
    float input_signal[SIGNAL_LENGTH];
    float output_signal[SIGNAL_LENGTH];
    float fir_coeffs[FIR_TAPS];
    float fir_delay[FIR_TAPS];
    Biquad iir_sections[IIR_ORDER];
} ProgramState;

static void init_fir(__global ProgramState *program_state, float cutoff) {
    /* Generate windowed sinc coefficients */
    int center = FIR_TAPS / 2;
    float sum = 0.0f;

    for (int i = 0; i < FIR_TAPS; i++) {
        int n = i - center;
        if (n == 0) {
            program_state->fir_coeffs[i] = cutoff;
        } else {
            /* sinc(n * cutoff) */
            float x = 3.14159f * (float)n * cutoff;
            float sinc;
            /* Taylor series for sin(x)/x */
            float x2 = x * x;
            sinc = 1.0f - x2 / 6.0f + x2 * x2 / 120.0f;
            program_state->fir_coeffs[i] = cutoff * sinc;
        }

        /* Hamming window */
        float window = 0.54f - 0.46f * (float)(1.0f - 2.0f * 3.14159f * i / (FIR_TAPS - 1));
        /* Simplified: just use a triangular window approximation */
        window = 1.0f - (float)(i - center) * (float)(i - center) / (float)(center * center);
        if (window < 0.0f) window = 0.0f;

        program_state->fir_coeffs[i] *= window;
        sum += program_state->fir_coeffs[i];
    }

    /* Normalize */
    for (int i = 0; i < FIR_TAPS; i++) {
        program_state->fir_coeffs[i] /= sum;
        program_state->fir_delay[i] = 0.0f;
    }
}

/* Initialize IIR biquad cascade (Butterworth-like) */
static void init_iir(__global ProgramState *program_state) {
    /* Simple low-pass biquad coefficients */
    for (int i = 0; i < IIR_ORDER; i++) {
        float w0 = 0.1f + 0.05f * i;  /* Varying cutoff per section */

        /* Approximate Butterworth coefficients */
        program_state->iir_sections[i].b0 = w0 * w0 / 4.0f;
        program_state->iir_sections[i].b1 = 2.0f * program_state->iir_sections[i].b0;
        program_state->iir_sections[i].b2 = program_state->iir_sections[i].b0;
        program_state->iir_sections[i].a1 = -2.0f * (1.0f - w0);
        program_state->iir_sections[i].a2 = (1.0f - w0) * (1.0f - w0);

        program_state->iir_sections[i].z1 = 0.0f;
        program_state->iir_sections[i].z2 = 0.0f;
    }
}

/* FIR filter - direct form */
static void fir_filter(__global ProgramState *program_state, const float *input, float *output, size_t length) {
    for (size_t n = 0; n < length; n++) {
        /* Shift delay line */
        for (int i = FIR_TAPS - 1; i > 0; i--) {
            program_state->fir_delay[i] = program_state->fir_delay[i - 1];
        }
        program_state->fir_delay[0] = input[n];

        /* Convolve */
        float sum = 0.0f;
        for (int i = 0; i < FIR_TAPS; i++) {
            sum += program_state->fir_coeffs[i] * program_state->fir_delay[i];
        }
        output[n] = sum;
    }
}

/* FIR filter - optimized with loop unrolling */
static void fir_filter_unrolled(__global ProgramState *program_state, const float *input, float *output, size_t length) {
    for (size_t n = 0; n < length; n++) {
        /* Shift delay line */
        for (int i = FIR_TAPS - 1; i > 0; i--) {
            program_state->fir_delay[i] = program_state->fir_delay[i - 1];
        }
        program_state->fir_delay[0] = input[n];

        /* Convolve with 4-way unrolling */
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int i = 0;
        for (; i + 4 <= FIR_TAPS; i += 4) {
            sum0 += program_state->fir_coeffs[i + 0] * program_state->fir_delay[i + 0];
            sum1 += program_state->fir_coeffs[i + 1] * program_state->fir_delay[i + 1];
            sum2 += program_state->fir_coeffs[i + 2] * program_state->fir_delay[i + 2];
            sum3 += program_state->fir_coeffs[i + 3] * program_state->fir_delay[i + 3];
        }
        for (; i < FIR_TAPS; i++) {
            sum0 += program_state->fir_coeffs[i] * program_state->fir_delay[i];
        }
        output[n] = sum0 + sum1 + sum2 + sum3;
    }
}

/* IIR biquad filter - transposed direct form II */
static float biquad_process(__global ProgramState *program_state, Biquad *bq, float input) {
    float output = bq->b0 * input + bq->z1;
    bq->z1 = bq->b1 * input - bq->a1 * output + bq->z2;
    bq->z2 = bq->b2 * input - bq->a2 * output;
    return output;
}

/* Cascaded IIR filter */
static void iir_filter(__global ProgramState *program_state, const float *input, float *output, size_t length) {
    for (size_t n = 0; n < length; n++) {
        float sample = input[n];

        /* Process through each biquad section */
        for (int i = 0; i < IIR_ORDER; i++) {
            sample = biquad_process(program_state, &program_state->iir_sections[i], sample);
        }

        output[n] = sample;
    }
}

/* Initialize test signal */
static void init_signal(__global ProgramState *program_state, int seed) {
    for (size_t i = 0; i < SIGNAL_LENGTH; i++) {
        /* Mix of frequencies */
        float t = (float)i / SIGNAL_LENGTH;
        float signal = 0.0f;

        /* Low frequency component */
        signal += 0.5f * ((float)((i * 17 + seed) % 1000) / 1000.0f - 0.5f);

        /* High frequency noise */
        signal += 0.3f * ((float)((i * 31 + seed * 7) % 100) / 100.0f - 0.5f);

        program_state->input_signal[i] = signal;
    }
}

int program_main(__global ProgramState *program_state) {
    init_fir(program_state, 0.3f);
    init_iir(program_state);

    float result_sum = 0.0f;

    for (int iter = 0; iter < 100; iter++) {
        init_signal(program_state, iter);

        /* Apply FIR filter */
        if (iter % 3 == 0) {
            fir_filter(program_state, program_state->input_signal, program_state->output_signal, SIGNAL_LENGTH);
        } else if (iter % 3 == 1) {
            fir_filter_unrolled(program_state, program_state->input_signal, program_state->output_signal, SIGNAL_LENGTH);
        } else {
            /* Apply IIR filter */
            /* Reset IIR state */
            for (int i = 0; i < IIR_ORDER; i++) {
                program_state->iir_sections[i].z1 = 0.0f;
                program_state->iir_sections[i].z2 = 0.0f;
            }
            iir_filter(program_state, program_state->input_signal, program_state->output_signal, SIGNAL_LENGTH);
        }

        /* Accumulate output energy */
        for (size_t i = 0; i < SIGNAL_LENGTH; i += 16) {
            result_sum += program_state->output_signal[i] * program_state->output_signal[i];
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
