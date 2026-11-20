// Vulkan SC port of programs/02_fibonacci.c. Explicit stack replaces device recursion.
static ulong fib_naive(uint index) {
    uint stack[31];
    uint count = 1;
    ulong sum = 0;
    stack[0] = index;
    while (count) {
        uint n = stack[--count];
        if (n < 2) {
            sum += n;
        } else {
            stack[count++] = n - 2;
            stack[count++] = n - 1;
        }
    }
    return sum;
}

static ulong fib_memo(uint index) {
    ulong cache[93] = {0, 1};
    for (uint n = 2; n <= index; ++n)
        cache[n] = cache[n - 1] + cache[n - 2];
    return cache[index];
}

static void output_word(__global uint *results, uint index, ulong value) {
    for (uint byte = 0; byte < 8; ++byte)
        results[index + byte] = (uint)((value >> (byte * 8)) & 255UL);
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    uint index = results[63];
    if (index > 92) {
        results[0] = 1;
        return;
    }
    ulong memo = fib_memo(index);
    output_word(results, 1, memo);
    results[33] = index <= 30;
    if (index <= 30) {
        ulong naive = fib_naive(index);
        output_word(results, 17, naive);
        results[0] = memo != naive;
    } else {
        results[0] = 0;
    }
}
