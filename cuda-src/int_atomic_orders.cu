/*
 * int_atomic_orders.c
 *
 * Atomic operations with varied memory orders and sizes.
 * Exercises atomic builtins across multiple widths and fences.
 *
 * Features exercised:
 *   - __atomic builtins with different sizes
 *   - Acquire/release/seq_cst ordering
 *   - Compare-exchange (strong/weak)
 *   - Thread fences
 */

#include <stdint.h>
#include <cuda/atomic>

__device__ volatile uint64_t sink;

__device__ static cuda::atomic<uint8_t, cuda::thread_scope_device> atomic_u8;
__device__ static cuda::atomic<uint16_t, cuda::thread_scope_device> atomic_u16;
__device__ static cuda::atomic<uint32_t, cuda::thread_scope_device> atomic_u32;
__device__ static cuda::atomic<uint64_t, cuda::thread_scope_device> atomic_u64;

__device__ __attribute__((noinline))
uint64_t atomic_mix(uint32_t seed) {
    uint64_t sum = 0;

    atomic_u32.store(seed, cuda::memory_order_relaxed);
    sum += atomic_u32.load(cuda::memory_order_acquire);

    uint32_t expected = seed;
    atomic_u32.compare_exchange_strong(expected, seed ^ 0xa5a5a5a5u,
                                cuda::memory_order_acq_rel, cuda::memory_order_acquire);
    sum += expected;

    sum += atomic_u32.fetch_add(7u, cuda::memory_order_seq_cst);

    uint8_t old8 = atomic_u8.fetch_xor((uint8_t)seed, cuda::memory_order_release);
    uint16_t old16 = atomic_u16.fetch_or((uint16_t)(seed >> 1),
                                       cuda::memory_order_acq_rel);

    uint64_t add64 = ((uint64_t)seed << 32) | (uint64_t)seed;
    uint64_t old64 = atomic_u64.fetch_add(add64, cuda::memory_order_acq_rel);
    sum ^= old64;

    uint64_t expected64 = old64 + add64;
    atomic_u64.compare_exchange_weak(expected64, expected64 ^ 0x5a5a5a5a5a5a5a5aull,
                                cuda::memory_order_release, cuda::memory_order_relaxed);
    sum += expected64;

    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
    sum += atomic_u64.load(cuda::memory_order_acquire);

    sum ^= atomic_u64.exchange((uint64_t)seed * 0x9e3779b97f4a7c15ull,
                               cuda::memory_order_seq_cst);

    return sum + old8 + old16;
}

__device__ int program_main(void) {
    uint64_t result = 0;

    for (uint32_t i = 0; i < 10000; i++) {
        result += atomic_mix(i * 3u + 1u);
    }

    sink = result;
    return 0;
}

extern "C" __global__ void program_entry(int *status) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *status = program_main();
}
