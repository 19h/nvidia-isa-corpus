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
 * Computed Goto (Labels as Values)
 *
 * GCC/Clang extension for direct threaded code.
 * Common in interpreter loops and dispatch tables.
 *
 * Features exercised:
 *   - Label addresses
 *   - Indirect goto
 *   - Threaded code dispatch
 */





/* Simple bytecode interpreter using computed goto */
typedef enum {
    OP_NOP = 0,
    OP_PUSH,
    OP_POP,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_DUP,
    OP_SWAP,
    OP_LOAD,
    OP_STORE,
    OP_JMP,
    OP_JZ,
    OP_JNZ,
    OP_HALT,
    OP_COUNT
} Opcode;

#define STACK_SIZE 64
#define MEM_SIZE 16

typedef struct {
    volatile int64_t sink;
} ProgramState;


int64_t interpret(__private ProgramState *program_state, const uint8_t *code, int code_len) {
    int64_t stack[STACK_SIZE];
    int sp = 0;
    int64_t mem[MEM_SIZE] = {0};
    int pc = 0;

    for (;;) {
        int op = pc < code_len ? code[pc] : OP_HALT;
        switch (op) {
            case OP_NOP: {
            { ++pc; continue; }
            }
            case OP_PUSH: {
            pc++;
            if (pc < code_len && sp < STACK_SIZE) {
                stack[sp++] = code[pc];
            }
            { ++pc; continue; }
            }
            case OP_POP: {
            if (sp > 0) sp--;
            { ++pc; continue; }
            }
            case OP_ADD: {
            if (sp >= 2) {
                int64_t b = stack[--sp];
                int64_t a = stack[--sp];
                stack[sp++] = a + b;
            }
            { ++pc; continue; }
            }
            case OP_SUB: {
            if (sp >= 2) {
                int64_t b = stack[--sp];
                int64_t a = stack[--sp];
                stack[sp++] = a - b;
            }
            { ++pc; continue; }
            }
            case OP_MUL: {
            if (sp >= 2) {
                int64_t b = stack[--sp];
                int64_t a = stack[--sp];
                stack[sp++] = a * b;
            }
            { ++pc; continue; }
            }
            case OP_DIV: {
            if (sp >= 2) {
                int64_t b = stack[--sp];
                int64_t a = stack[--sp];
                stack[sp++] = b != 0 ? a / b : 0;
            }
            { ++pc; continue; }
            }
            case OP_DUP: {
            if (sp > 0 && sp < STACK_SIZE) {
                stack[sp] = stack[sp - 1];
                sp++;
            }
            { ++pc; continue; }
            }
            case OP_SWAP: {
            if (sp >= 2) {
                int64_t tmp = stack[sp - 1];
                stack[sp - 1] = stack[sp - 2];
                stack[sp - 2] = tmp;
            }
            { ++pc; continue; }
            }
            case OP_LOAD: {
            pc++;
            if (pc < code_len && sp < STACK_SIZE) {
                int addr = code[pc] % MEM_SIZE;
                stack[sp++] = mem[addr];
            }
            { ++pc; continue; }
            }
            case OP_STORE: {
            pc++;
            if (pc < code_len && sp > 0) {
                int addr = code[pc] % MEM_SIZE;
                mem[addr] = stack[--sp];
            }
            { ++pc; continue; }
            }
            case OP_JMP: {
            pc++;
            if (pc < code_len) {
                pc = code[pc] % code_len;
                continue;
            }
            return sp > 0 ? stack[sp - 1] : 0;
            }
            case OP_JZ: {
            pc++;
            if (pc < code_len && sp > 0) {
                if (stack[sp - 1] == 0) {
                    sp--;
                    pc = code[pc] % code_len;
                    continue;
                }
            }
            { ++pc; continue; }
            }
            case OP_JNZ: {
            pc++;
            if (pc < code_len && sp > 0) {
                if (stack[sp - 1] != 0) {
                    sp--;
                    pc = code[pc] % code_len;
                    continue;
                }
            }
            { ++pc; continue; }
            }
            case OP_HALT:
            default: return sp > 0 ? stack[sp - 1] : 0;
        }
    }
}

/* State machine using computed goto */

int64_t state_machine_cgoto(__private ProgramState *program_state, int iterations) {
    int64_t acc = 0;
    int state = 0;
    int counter = 0;
    for (;;) {
        switch (state) {
            case 0: acc = 1; counter = 0; state = 1; break;
            case 1:
                acc += counter++;
                state = counter >= iterations ? 3 : counter % 10 == 0 ? 2 : 1;
                break;
            case 2: acc *= 2; state = 1; break;
            default: return acc;
        }
    }
}

int match_pattern(__private ProgramState *program_state, const char *text, int len) {
    int count[4] = {0};
    for (int i = 0; i < len; ++i) {
        char c = text[i];
        int type = ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) ? 0 :
                   (c >= '0' && c <= '9') ? 1 :
                   (c == ' ' || c == '\t' || c == '\n') ? 2 : 3;
        ++count[type];
    }
    return (count[0] << 24) | (count[1] << 16) | (count[2] << 8) | count[3];
}

int64_t threaded_exec(__private ProgramState *program_state) {
    enum { LIT, ADD, DUP, MUL, EXIT };
    int code[] = {LIT, 10, LIT, 20, ADD, DUP, MUL, EXIT};
    int64_t stack[32];
    int sp = 0;
    int pc = 0;
    for (;;) {
        switch (code[pc++]) {
            case LIT: stack[sp++] = code[pc++]; break;
            case ADD:
                if (sp >= 2) { int64_t b = stack[--sp]; stack[sp-1] += b; }
                break;
            case DUP:
                if (sp > 0 && sp < 32) { stack[sp] = stack[sp-1]; ++sp; }
                break;
            case MUL:
                if (sp >= 2) { int64_t b = stack[--sp]; stack[sp-1] *= b; }
                break;
            default: return sp > 0 ? stack[sp-1] : 0;
        }
    }
}

int program_main(__private ProgramState *program_state) {
    int64_t result = 0;

    /* Test bytecode interpreter */
    uint8_t programs[][20] = {
        /* Program 1: push 10, push 20, add -> 30 */
        {OP_PUSH, 10, OP_PUSH, 20, OP_ADD, OP_HALT},
        /* Program 2: push 5, dup, mul -> 25 */
        {OP_PUSH, 5, OP_DUP, OP_MUL, OP_HALT},
        /* Program 3: push 100, push 7, div -> 14 */
        {OP_PUSH, 100, OP_PUSH, 7, OP_DIV, OP_HALT},
        /* Program 4: push 3, push 4, mul, push 5, add -> 17 */
        {OP_PUSH, 3, OP_PUSH, 4, OP_MUL, OP_PUSH, 5, OP_ADD, OP_HALT},
        /* Program 5: push 10, store 0, push 5, load 0, add -> 15 */
        {OP_PUSH, 10, OP_STORE, 0, OP_PUSH, 5, OP_LOAD, 0, OP_ADD, OP_HALT}
    };
    int prog_lens[] = {6, 5, 6, 9, 10};

    const char test_strings[][16] = {
        "Hello World 123",
        "abc123 def456",
        "    ",
        "12345",
        "!@#$%"
    };

    for (int i = 0; i < 100000; i++) {
        /* Run bytecode interpreter */
        int prog_idx = i % 5;
        result += interpret(program_state, programs[prog_idx], prog_lens[prog_idx]);

        /* Run state machine */
        if (i % 100 == 0) {
            result += state_machine_cgoto(program_state, 50);
        }

        /* Run pattern matcher */
        const char *str = test_strings[i % 5];
        int len = 0;
        while (str[len]) len++;
        result += match_pattern(program_state, str, len);

        /* Run threaded code */
        result += threaded_exec(program_state);
    }

    program_state->sink = result;
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)(((ulong)storage.sink >> (8u * i)) & 255u);
}
