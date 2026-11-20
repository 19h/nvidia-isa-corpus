// Generate the shader's constants using the original compile-time C++ tables.
#define main corpus_main
#include "../programs/cpp_consteval_tuple_automaton.cpp"
#undef main

template <typename T, std::size_t N>
static void emit(const char *type, const char *name, const std::array<T, N> &entries)
{
    std::printf("__constant %s %s[%zu] = {\n", type, name, N);
    for (std::size_t i = 0; i < N; ++i) {
        if (i % 8 == 0)
            std::printf("    ");
        std::printf("%" PRIu64 "UL,", static_cast<std::uint64_t>(entries[i]));
        std::printf(i % 8 == 7 || i == N - 1 ? "\n" : " ");
    }
    std::puts("};");
}

int main()
{
    std::printf("#define StateCount %zu\n#define Alphabet %zu\n#define InputCount %zu\n",
                StateCount, Alphabet, InputCount);
    std::printf("#define Fingerprint %" PRIu64 "UL\n", Fingerprint);
    emit("uchar", "TableNext", Tables.next);
    emit("uint", "TableEmit", Tables.emit);
    emit("uchar", "TableAccepting", Tables.accepting);
    emit("ulong", "TableRowSalt", Tables.row_salt);
    emit("uchar", "Input", Input);
}
