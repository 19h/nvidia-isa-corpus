// Instantiate the original templates and emit their compile-time results.
#include <algorithm>

#define main corpus_main
#include "../../programs/10_template_madness.cpp"
#undef main

template<typename Range>
void emit(const char *name, const Range& values) {
    std::cout << "const uint " << name << "[] = uint[](";
    bool first = true;
    for (auto value : values) {
        if (!first)
            std::cout << ", ";
        first = false;
        std::cout << value << 'u';
    }
    std::cout << ");\n";
}

int main() {
    using namespace compile_time;
    using namespace compile_time_sort;
    emit("numeric_constants", std::array<std::size_t, 5>{
        Fibonacci_v<20>, Fibonacci_v<30>, Factorial_v<10>, GCD_v<48, 18>, LCM_v<21, 6>
    });
    emit("primes", primes_to_50);
    emit("unsorted_keys", IntListToArray<UnsortedList>::value);
    emit("selection_sorted", IntListToArray<SortedBySelection>::value);
    emit("bubble_sorted", IntListToArray<SortedByBubble>::value);
    emit("factorial_sequence", fold_expressions::factorials_0_to_9);
    emit("fibonacci_sequence", fold_expressions::fibs_0_to_19);
}
