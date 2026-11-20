/*
 * 01_hello_world.c
 * 
 * The simplest possible C program - prints "Hello, World!" to stdout.
 * This is a trivial example demonstrating basic program structure.
 */

#include <stdio.h>

__device__ int program_main(void) {
    printf("Hello, World!\n");
    return 0;
}

extern "C" __global__ void program_entry(int *status) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *status = program_main();
}
