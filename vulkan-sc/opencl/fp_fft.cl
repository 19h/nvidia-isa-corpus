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
 * Fast Fourier Transform (Radix-2 Cooley-Tukey)
 *
 * Computes FFT of complex signal.
 * Heavy use of trig functions, complex arithmetic.
 *
 * Features exercised:
 *   - FP sin/cos (or table lookup)
 *   - Complex multiply/add
 *   - Bit reversal permutation
 */



/* Simple inline math since we avoid libm dependency */
#define PI 3.14159265358979323846

#define FFT_SIZE 1024



/* Complex number */
typedef struct {
    double re;
    double im;
} Complex;




/* Approximate sine using Taylor series */
typedef struct {
    volatile double sink;
    Complex data[FFT_SIZE];
    Complex twiddle[FFT_SIZE / 2];
} ProgramState;

static double approx_sin(__global ProgramState *program_state, double x) {
    /* Reduce to [-pi, pi] */
    while (x > PI) x -= 2 * PI;
    while (x < -PI) x += 2 * PI;

    double x2 = x * x;
    double x3 = x2 * x;
    double x5 = x3 * x2;
    double x7 = x5 * x2;
    double x9 = x7 * x2;

    return x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0 + x9 / 362880.0;
}

/* Approximate cosine using Taylor series */
static double approx_cos(__global ProgramState *program_state, double x) {
    /* Reduce to [-pi, pi] */
    while (x > PI) x -= 2 * PI;
    while (x < -PI) x += 2 * PI;

    double x2 = x * x;
    double x4 = x2 * x2;
    double x6 = x4 * x2;
    double x8 = x6 * x2;

    return 1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0 + x8 / 40320.0;
}

/* Bit reversal */
static unsigned int bit_reverse(__global ProgramState *program_state, unsigned int x, int bits) {
    volatile unsigned int result = 0;
    for (int i = 0; i < bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

/* Precompute twiddle factors */
static void init_twiddle(__global ProgramState *program_state) {
    for (size_t k = 0; k < FFT_SIZE / 2; k++) {
        double angle = -2.0 * PI * k / FFT_SIZE;
        program_state->twiddle[k].re = approx_cos(program_state, angle);
        program_state->twiddle[k].im = approx_sin(program_state, angle);
    }
}

/* In-place FFT */
static void fft(__global ProgramState *program_state, Complex *x, size_t n) {
    int bits = 0;
    size_t temp = n;
    while (temp > 1) {
        temp >>= 1;
        bits++;
    }

    /* Bit-reversal permutation */
    for (size_t i = 0; i < n; i++) {
        size_t j = bit_reverse(program_state, (unsigned int)i, bits);
        if (i < j) {
            Complex t = x[i];
            x[i] = x[j];
            x[j] = t;
        }
    }

    /* Cooley-Tukey iterative FFT */
    for (size_t len = 2; len <= n; len *= 2) {
        size_t half_count = len / 2;
        size_t step = n / len;

        for (size_t i = 0; i < n; i += len) {
            for (size_t j = 0; j < half_count; j++) {
                Complex w = program_state->twiddle[j * step];
                Complex *a = &x[i + j];
                Complex *b = &x[i + j + half_count];

                /* Butterfly operation */
                Complex t;
                t.re = w.re * b->re - w.im * b->im;
                t.im = w.re * b->im + w.im * b->re;

                Complex u = *a;
                a->re = u.re + t.re;
                a->im = u.im + t.im;
                b->re = u.re - t.re;
                b->im = u.im - t.im;
            }
        }
    }
}

/* Inverse FFT */
static void ifft(__global ProgramState *program_state, Complex *x, size_t n) {
    /* Conjugate */
    for (size_t i = 0; i < n; i++) {
        x[i].im = -x[i].im;
    }

    fft(program_state, x, n);

    /* Conjugate and scale */
    double scale = 1.0 / n;
    for (size_t i = 0; i < n; i++) {
        x[i].re *= scale;
        x[i].im = -x[i].im * scale;
    }
}

int program_main(__global ProgramState *program_state) {
    init_twiddle(program_state);

    double result_sum = 0.0;

    for (int iter = 0; iter < 1000; iter++) {
        /* Initialize with test signal */
        for (size_t i = 0; i < FFT_SIZE; i++) {
            /* Mix of frequencies */
            double t = (double)i / FFT_SIZE;
            program_state->data[i].re = approx_sin(program_state, 2 * PI * 10 * t)
                       + 0.5 * approx_sin(program_state, 2 * PI * 50 * t)
                       + 0.25 * approx_sin(program_state, 2 * PI * 100 * t);
            program_state->data[i].im = 0.0;
        }

        /* Forward FFT */
        fft(program_state, program_state->data, FFT_SIZE);

        /* Accumulate magnitude at some frequencies */
        for (size_t i = 0; i < FFT_SIZE / 2; i += 10) {
            double mag = program_state->data[i].re * program_state->data[i].re + program_state->data[i].im * program_state->data[i].im;
            result_sum += mag;
        }

        /* Inverse FFT */
        ifft(program_state, program_state->data, FFT_SIZE);

        /* Verify reconstruction */
        for (size_t i = 0; i < FFT_SIZE; i += 100) {
            result_sum += program_state->data[i].re;
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
        results[1 + i] = (uint)((as_ulong(storage->sink) >> (8u * i)) & 255u);
}
