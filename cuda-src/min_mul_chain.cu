/*
 * Minimal Integer Multiplication Chain
 *
 * Generates sequences dominated by integer MUL instructions.
 * Useful for ISA detection - multiplication encodings vary significantly.
 *
 * Expected dominant instructions by architecture:
 *   x86:    imul
 *   ARM:    mul, mla
 *   RISC-V: mul
 *   MIPS:   mult, mul
 *   PPC:    mullw, mulld
 */

__device__ volatile int sink;

__device__ int program_main(void) {
    int a = 3, b = 5, c = 7, d = 11;
    int e = 13, f = 17, g = 19, h = 23;

    for (int i = 0; i < 100000; i++) {
        /* Multiplication chain */
        a = a * b;
        b = b * c;
        c = c * d;
        d = d * 3;

        e = e * f;
        f = f * g;
        g = g * h;
        h = h * 5;

        /* Cross multiplications */
        a = (a * e) & 0xFFFF;  /* Mask to prevent overflow */
        b = (b * f) & 0xFFFF;
        c = (c * g) & 0xFFFF;
        d = (d * h) & 0xFFFF;

        /* Multiply-add patterns */
        e = (e * 7) + 1;
        f = (f * 11) + 2;
        g = (g * 13) + 3;
        h = (h * 17) + 4;

        /* Reset to prevent overflow */
        if (i % 100 == 0) {
            a = 3; b = 5; c = 7; d = 11;
            e = 13; f = 17; g = 19; h = 23;
        }
    }

    sink = a + b + c + d + e + f + g + h;
    return 0;
}

extern "C" __global__ void program_entry(int *status) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *status = program_main();
}
