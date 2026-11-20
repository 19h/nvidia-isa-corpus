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
 * Mandelbrot Set Computation
 *
 * Computes Mandelbrot set escape times.
 * Heavy use of floating point multiplication, addition, comparison.
 *
 * Features exercised:
 *   - FP multiply, add, compare
 *   - FP to integer conversion
 *   - Loop-carried dependencies
 */



#define WIDTH 256
#define HEIGHT 256
#define MAX_ITER 100





/* Compute escape time for a single point */
typedef struct {
    uint32_t sink;
    uint8_t image[HEIGHT][WIDTH];
} ProgramState;

static int mandelbrot_point(__global ProgramState *program_state, double cr, double ci) {
    double zr = 0.0;
    double zi = 0.0;

    for (int i = 0; i < MAX_ITER; i++) {
        double zr2 = zr * zr;
        double zi2 = zi * zi;

        if (zr2 + zi2 > 4.0) {
            return i;
        }

        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }

    return MAX_ITER;
}

/* Single precision version */
static int mandelbrot_point_f(__global ProgramState *program_state, float cr, float ci) {
    float zr = 0.0f;
    float zi = 0.0f;

    for (int i = 0; i < MAX_ITER; i++) {
        float zr2 = zr * zr;
        float zi2 = zi * zi;

        if (zr2 + zi2 > 4.0f) {
            return i;
        }

        zi = 2.0f * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }

    return MAX_ITER;
}

int program_main(__global ProgramState *program_state) {
    /* Region: [-2.5, 1] x [-1, 1] */
    double x_min = -2.5;
    double x_max = 1.0;
    double y_min = -1.0;
    double y_max = 1.0;

    double x_step = (x_max - x_min) / WIDTH;
    double y_step = (y_max - y_min) / HEIGHT;

    uint32_t checksum = 0;

    /* Multiple iterations for timing */
    for (int iter = 0; iter < 10; iter++) {
        /* Double precision computation */
        for (int py = 0; py < HEIGHT; py++) {
            double ci = y_min + py * y_step;
            for (int px = 0; px < WIDTH; px++) {
                double cr = x_min + px * x_step;
                int escape = mandelbrot_point(program_state, cr, ci);
                program_state->image[py][px] = (uint8_t)(escape % 256);
            }
        }

        /* Accumulate checksum */
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                checksum += program_state->image[y][x];
            }
        }

        /* Single precision computation */
        float fx_min = -2.5f;
        float fx_step = 3.5f / WIDTH;
        float fy_min = -1.0f;
        float fy_step = 2.0f / HEIGHT;

        for (int py = 0; py < HEIGHT; py++) {
            float ci = fy_min + py * fy_step;
            for (int px = 0; px < WIDTH; px++) {
                float cr = fx_min + px * fx_step;
                int escape = mandelbrot_point_f(program_state, cr, ci);
                program_state->image[py][px] = (uint8_t)(escape % 256);
            }
        }

        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                checksum += program_state->image[y][x];
            }
        }
    }

    program_state->sink = checksum;
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
