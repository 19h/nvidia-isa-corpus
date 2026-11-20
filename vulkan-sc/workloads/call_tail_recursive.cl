// Arithmetic and state transitions from programs/call_tail_recursive.c.
// Bounded loops replace recursive calls in the shader.
static long factorial(int n) {
    long acc = 1;
    while (n > 1) acc *= n--;
    return acc;
}

static long fibonacci(int n) {
    long a = 0, b = 1;
    for (int i = 0; i < n; ++i) {
        long sum = a + b;
        a = b;
        b = sum;
    }
    return a;
}

static long sum_tail(int n, long acc) {
    while (n > 0) acc += n--;
    return acc;
}

static long gcd(long a, long b) {
    while (b) {
        long remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

static long iterate_tail(const int *arr, int len, long acc) {
    for (int i = 0; i < len; ++i)
        acc += arr[i];
    return acc;
}

static long power_tail(long base, int exponent, long acc) {
    while (exponent) {
        if (exponent & 1) {
            acc *= base;
            --exponent;
        } else {
            base *= base;
            exponent /= 2;
        }
    }
    return acc;
}

static int is_even_tail(int n) { return (n & 1) == 0; }
static int is_odd_tail(int n) { return (n & 1) != 0; }

typedef enum { STATE_A, STATE_B, STATE_C, STATE_DONE } State;

static long state_machine(State state, int counter, long acc) {
    while (state != STATE_DONE) {
        switch (state) {
            case STATE_A:
                if (counter <= 0) {
                    state = STATE_DONE;
                } else {
                    state = STATE_B;
                    acc += 1;
                }
                break;
            case STATE_B: state = STATE_C; acc *= 2; break;
            case STATE_C: state = STATE_A; --counter; acc += 3; break;
            default: return acc;
        }
    }
    return acc;
}

static long workload(void) {
    long result = 0;
    int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    for (int i = 0; i < 10000; i++) {
        result += factorial(12);

        result += fibonacci(30);

        result += sum_tail(100, 0);

        result += gcd(i + 1000, (i % 100) + 1);

        result += iterate_tail(arr, 10, 0);

        result += power_tail(2, i % 20, 1);

        result += is_even_tail(i % 100);
        result += is_odd_tail(i % 100);

        if (i % 100 == 0) {
            result += state_machine(STATE_A, 10, 0);
        }
    }

    return result;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ulong checksum = (ulong)workload();
    results[0] = 0;
    for (uint i = 0; i < 8; ++i)
        results[1 + i] = (uint)((checksum >> (8u * i)) & 255UL);
}
