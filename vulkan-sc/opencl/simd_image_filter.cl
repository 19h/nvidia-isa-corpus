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
 * SIMD Image Box Blur Filter
 *
 * Applies 3x3 box blur using SIMD instructions.
 * Common pattern in image processing pipelines.
 *
 * Features exercised:
 *   x86:    SSE/AVX packed byte operations, unpack, average
 *   ARM:    NEON vld, vst, vhadd
 */




#if 0

#define HAS_X86_SIMD 1
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)

#define HAS_NEON 1
#endif

#define WIDTH 128
#define HEIGHT 128






/* Scalar 3x3 box blur */
typedef struct {
    uint8_t sink;
    uint8_t input[HEIGHT][WIDTH];
    uint8_t output[HEIGHT][WIDTH];
} ProgramState;

void box_blur_scalar(__global ProgramState *program_state) {
    for (int y = 1; y < HEIGHT - 1; y++) {
        for (int x = 1; x < WIDTH - 1; x++) {
            int sum = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    sum += program_state->input[y + dy][x + dx];
                }
            }
            program_state->output[y][x] = (uint8_t)(sum / 9);
        }
    }
}

#if defined(HAS_X86_SIMD) && defined(__SSE2__)
void box_blur_sse2(void) {
    for (int y = 1; y < HEIGHT - 1; y++) {
        for (int x = 1; x < WIDTH - 1 - 16; x += 16) {
            __m128i sum = _mm_setzero_si128();

            /* Process 3x3 neighborhood */
            for (int dy = -1; dy <= 1; dy++) {
                /* Load 16 bytes from each row position */
                __m128i row_left = _mm_loadu_si128((__m128i *)&input[y + dy][x - 1]);
                __m128i row_center = _mm_loadu_si128((__m128i *)&input[y + dy][x]);
                __m128i row_right = _mm_loadu_si128((__m128i *)&input[y + dy][x + 1]);

                /* Unpack to 16-bit for accumulation */
                __m128i lo_l = _mm_unpacklo_epi8(row_left, _mm_setzero_si128());
                __m128i lo_c = _mm_unpacklo_epi8(row_center, _mm_setzero_si128());
                __m128i lo_r = _mm_unpacklo_epi8(row_right, _mm_setzero_si128());

                sum = _mm_add_epi16(sum, lo_l);
                sum = _mm_add_epi16(sum, lo_c);
                sum = _mm_add_epi16(sum, lo_r);
            }

            /* Divide by 9 (approximate with multiply and shift) */
            /* 9 ~= 1/9 * 65536 = 7282 */
            __m128i divisor = _mm_set1_epi16(7282);
            sum = _mm_mulhi_epu16(sum, divisor);

            /* Pack back to bytes */
            __m128i result = _mm_packus_epi16(sum, _mm_setzero_si128());
            _mm_storel_epi64((__m128i *)&output[y][x], result);
        }
    }
}
#endif

#if defined(HAS_NEON)
void box_blur_neon(void) {
    for (int y = 1; y < HEIGHT - 1; y++) {
        for (int x = 1; x < WIDTH - 1 - 8; x += 8) {
            uint16x8_t sum = vdupq_n_u16(0);

            /* Process 3x3 neighborhood */
            for (int dy = -1; dy <= 1; dy++) {
                uint8x8_t row_left = vld1_u8(&input[y + dy][x - 1]);
                uint8x8_t row_center = vld1_u8(&input[y + dy][x]);
                uint8x8_t row_right = vld1_u8(&input[y + dy][x + 1]);

                /* Widen and accumulate */
                sum = vaddw_u8(sum, row_left);
                sum = vaddw_u8(sum, row_center);
                sum = vaddw_u8(sum, row_right);
            }

            /* Divide by 9: multiply by 7282 and take high 16 bits */
            uint16x8_t divisor = vdupq_n_u16(7282);
            uint32x4_t prod_lo = vmull_u16(vget_low_u16(sum), vget_low_u16(divisor));
            uint32x4_t prod_hi = vmull_u16(vget_high_u16(sum), vget_high_u16(divisor));
            uint16x4_t result_lo = vshrn_n_u32(prod_lo, 16);
            uint16x4_t result_hi = vshrn_n_u32(prod_hi, 16);
            uint16x8_t result16 = vcombine_u16(result_lo, result_hi);

            /* Narrow to 8-bit */
            uint8x8_t result = vmovn_u16(result16);
            vst1_u8(&output[y][x], result);
        }
    }
}
#endif

int program_main(__global ProgramState *program_state) {
    /* Initialize input image with gradient pattern */
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            program_state->input[y][x] = (uint8_t)((x + y) & 0xFF);
        }
    }

    for (int iter = 0; iter < 1000; iter++) {
        /* Use best available SIMD implementation */
#if defined(HAS_X86_SIMD) && defined(__SSE2__)
        box_blur_sse2();
#elif defined(HAS_NEON)
        box_blur_neon();
#else
        box_blur_scalar(program_state);
#endif

        /* Swap buffers */
        for (int y = 1; y < HEIGHT - 1; y++) {
            for (int x = 1; x < WIDTH - 1; x++) {
                program_state->input[y][x] = program_state->output[y][x];
            }
        }
    }

    /* Prevent optimization */
    uint8_t sum = 0;
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            sum ^= program_state->output[i][j];
        }
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
