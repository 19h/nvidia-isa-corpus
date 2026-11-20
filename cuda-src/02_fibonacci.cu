/*
 * Fibonacci Sequence with Memoization
 * 
 * This program demonstrates the dramatic performance difference between
 * naive recursive Fibonacci (O(2^n) time complexity) and memoized
 * Fibonacci (O(n) time complexity).
 *
 * Complexity Analysis:
 * --------------------
 * Naive Recursive: O(2^n) time, O(n) space (call stack)
 *   - Each call branches into two recursive calls
 *   - Results in exponential number of redundant calculations
 *   - fib(40) makes over 300 million function calls
 *
 * Memoized: O(n) time, O(n) space (cache + call stack)
 *   - Each unique fib(k) is computed exactly once
 *   - Subsequent calls retrieve cached result in O(1)
 *   - fib(40) makes only 79 function calls (40 unique + 39 cache hits)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* Maximum Fibonacci index we can compute (limited by int64_t overflow) */
#define MAX_FIB_INDEX 92

/* Memoization cache - static array initialized to 0 */
__device__ static int64_t memo_cache[MAX_FIB_INDEX + 1] = {0};

/* Flag to track which cache entries are valid (0 is a valid fib result) */
__device__ static int cache_initialized[MAX_FIB_INDEX + 1] = {0};

/*
 * Naive recursive Fibonacci
 * 
 * Time Complexity: O(2^n)
 * Space Complexity: O(n) for the call stack
 *
 * This implementation recalculates the same subproblems repeatedly.
 * For fib(n), it makes approximately 2^n function calls because each
 * call (except base cases) spawns two more calls, forming a binary tree
 * of recursive calls.
 *
 * Example call tree for fib(5):
 *                    fib(5)
 *                   /      \
 *              fib(4)      fib(3)
 *             /     \      /    \
 *         fib(3)  fib(2) fib(2) fib(1)
 *         /   \
 *     fib(2) fib(1)
 *
 * Notice fib(2) is computed 3 times, fib(3) is computed 2 times.
 * This redundancy grows exponentially with n.
 */
__device__ int64_t fib_naive(int n) {
    /* Error handling for negative inputs */
    if (n < 0) {
        printf("Error: fib_naive() called with negative index %d\n", n);
        return -1;
    }
    
    /* Base cases */
    if (n == 0) return 0;
    if (n == 1) return 1;
    
    /* Recursive case: sum of two preceding numbers */
    return fib_naive(n - 1) + fib_naive(n - 2);
}

/*
 * Memoized recursive Fibonacci
 *
 * Time Complexity: O(n) - each subproblem solved exactly once
 * Space Complexity: O(n) for cache + O(n) for call stack
 *
 * Memoization transforms the exponential algorithm into a linear one
 * by caching previously computed results. When fib_memo(k) is called:
 *   1. If k is in cache, return cached value immediately (O(1))
 *   2. Otherwise, compute it, store in cache, then return
 *
 * This ensures each fib(0), fib(1), ..., fib(n) is computed at most once.
 * The first call to fib_memo(n) will compute and cache all values from
 * fib(2) to fib(n), making subsequent calls O(1) lookups.
 */
__device__ int64_t fib_memo(int n) {
    /* Error handling for negative inputs */
    if (n < 0) {
        printf("Error: fib_memo() called with negative index %d\n", n);
        return -1;
    }
    
    /* Bounds checking to prevent buffer overflow */
    if (n > MAX_FIB_INDEX) {
        printf("Error: fib_memo() index %d exceeds maximum %d\n", 
                n, MAX_FIB_INDEX);
        return -1;
    }
    
    /* Base cases */
    if (n == 0) return 0;
    if (n == 1) return 1;
    
    /* Check if result is already cached */
    if (cache_initialized[n]) {
        return memo_cache[n];
    }
    
    /* Compute, cache, and return the result */
    memo_cache[n] = fib_memo(n - 1) + fib_memo(n - 2);
    cache_initialized[n] = 1;
    
    return memo_cache[n];
}

/*
 * Reset the memoization cache
 * Useful for timing comparisons to ensure fair benchmarking
 */
__device__ void reset_memo_cache(void) {
    for (int i = 0; i <= MAX_FIB_INDEX; i++) {
        memo_cache[i] = 0;
        cache_initialized[i] = 0;
    }
}

// The caller supplies n and the GPU's nominal clock frequency in kHz.
__device__ int program_main(int n, unsigned int clock_khz) {
    if (n < 0 || n > MAX_FIB_INDEX || clock_khz == 0) {
        printf("Expected a Fibonacci index from 0 to %d and a nonzero clock frequency.\n", MAX_FIB_INDEX);
        return EXIT_FAILURE;
    }
    const double cycles_per_second = 1000.0 * clock_khz;

    printf("Computing Fibonacci(%d)\n", n);
    printf("========================================\n\n");
    
    /*
     * Test error handling with negative input
     */
    printf("Testing error handling with negative input:\n");
    int64_t result_neg = fib_memo(-5);
    if (result_neg == -1) {
        printf("  Correctly returned -1 for negative input\n\n");
    }
    
    /*
     * Memoized version timing
     * Run this FIRST to avoid the long wait from naive version
     */
    printf("Memoized Fibonacci (O(n) time complexity):\n");
    reset_memo_cache();
    
    unsigned long long start_memo = clock64();
    int64_t result_memo = fib_memo(n);
    unsigned long long end_memo = clock64();
    double time_memo = (end_memo - start_memo) / cycles_per_second;
    
    printf("  fib(%d) = %lld\n", n, (long long)result_memo);
    printf("  Time: %.9f seconds\n\n", time_memo);
    
    /*
     * Naive version timing
     * WARNING: This is VERY slow for n > 40
     */
    printf("Naive Recursive Fibonacci (O(2^n) time complexity):\n");
    
    if (n > 45) {
        printf("  SKIPPED: n=%d would take too long (exponential time)\n", n);
    } else {
        unsigned long long start_naive = clock64();
        int64_t result_naive = fib_naive(n);
        unsigned long long end_naive = clock64();
        double time_naive = (end_naive - start_naive) / cycles_per_second;
        
        printf("  fib(%d) = %lld\n", n, (long long)result_naive);
        printf("  Time: %.9f seconds\n\n", time_naive);
        
        /*
         * Performance comparison
         */
        printf("Performance Comparison:\n");
        printf("========================================\n");
        
        if (time_memo > 0) {
            double speedup = time_naive / time_memo;
            printf("  Memoized version is %.1fx faster\n", speedup);
        } else {
            printf("  Memoized version completed in < 1 nanosecond\n");
        }
        
        printf("\n");
        printf("Theoretical Analysis for n=%d:\n", n);
        printf("  Naive: ~2^%d = ~%.2e function calls\n", n, (double)(1ULL << n));
        printf("  Memoized: %d unique computations + %d cache hits\n", n - 1, n - 1);
        
        /* Verify both methods produce the same result */
        if (result_naive != result_memo) {
            printf("\nERROR: Results don't match! naive=%lld, memo=%lld\n",
                    (long long)result_naive, (long long)result_memo);
            return EXIT_FAILURE;
        }
        printf("\n  Results verified: both methods return the same value\n");
    }
    
    /*
     * Show a few more Fibonacci numbers using the memoized version
     */
    printf("\nFirst 20 Fibonacci numbers (using memoized version):\n");
    reset_memo_cache();
    for (int i = 0; i < 20; i++) {
        printf("  fib(%2d) = %lld\n", i, (long long)fib_memo(i));
    }
    
    return EXIT_SUCCESS;
}

extern "C" __global__ void program_entry(int *status, int n, unsigned int clock_khz) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *status = program_main(n, clock_khz);
}
