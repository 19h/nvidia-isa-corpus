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
 * Indirect Function Calls
 *
 * Tests function pointer dispatch patterns.
 * Exercises indirect call/branch instructions.
 *
 * Features exercised:
 *   - Function pointer calls
 *   - Virtual dispatch simulation
 *   - Callback patterns
 *   - Jump tables via function pointers
 */






/* Basic function pointer type */
typedef int BinaryOp;
typedef int UnaryOp;

/* Simple operations */
typedef struct {
    volatile int64_t sink;
} ProgramState;

 int op_add(__private ProgramState *program_state, int a, int b) { return a + b; }
 int op_sub(__private ProgramState *program_state, int a, int b) { return a - b; }
 int op_mul(__private ProgramState *program_state, int a, int b) { return a * b; }
 int op_div(__private ProgramState *program_state, int a, int b) { return b ? a / b : 0; }
 int op_mod(__private ProgramState *program_state, int a, int b) { return b ? a % b : 0; }
 int op_and(__private ProgramState *program_state, int a, int b) { return a & b; }
 int op_or(__private ProgramState *program_state, int a, int b) { return a | b; }
 int op_xor(__private ProgramState *program_state, int a, int b) { return a ^ b; }
 int op_min(__private ProgramState *program_state, int a, int b) { return a < b ? a : b; }
 int op_max(__private ProgramState *program_state, int a, int b) { return a > b ? a : b; }

/* Unary operations */
 int unary_neg(__private ProgramState *program_state, int a) { return -a; }
 int unary_not(__private ProgramState *program_state, int a) { return ~a; }
 int unary_abs(__private ProgramState *program_state, int a) { return a < 0 ? -a : a; }
 int unary_square(__private ProgramState *program_state, int a) { return a * a; }
 int unary_double(__private ProgramState *program_state, int a) { return a * 2; }
 int unary_half(__private ProgramState *program_state, int a) { return a / 2; }

/* Function pointer table - dispatch by index */
enum { SELECT_OP_ADD, SELECT_OP_SUB, SELECT_OP_MUL, SELECT_OP_DIV, SELECT_OP_MOD, SELECT_OP_AND, SELECT_OP_OR, SELECT_OP_XOR, SELECT_OP_MIN, SELECT_OP_MAX };
#define NUM_BINARY_OPS 10

enum { SELECT_UNARY_NEG, SELECT_UNARY_NOT, SELECT_UNARY_ABS, SELECT_UNARY_SQUARE, SELECT_UNARY_DOUBLE, SELECT_UNARY_HALF };
#define NUM_UNARY_OPS 6


int dispatch_binary(__private ProgramState *program_state, int op, int a, int b) {
    if (op >= 0 && (size_t)op < NUM_BINARY_OPS) {
        switch (op) {
            case SELECT_OP_ADD: return op_add(program_state, a, b);
            case SELECT_OP_SUB: return op_sub(program_state, a, b);
            case SELECT_OP_MUL: return op_mul(program_state, a, b);
            case SELECT_OP_DIV: return op_div(program_state, a, b);
            case SELECT_OP_MOD: return op_mod(program_state, a, b);
            case SELECT_OP_AND: return op_and(program_state, a, b);
            case SELECT_OP_OR: return op_or(program_state, a, b);
            case SELECT_OP_XOR: return op_xor(program_state, a, b);
            case SELECT_OP_MIN: return op_min(program_state, a, b);
            case SELECT_OP_MAX: return op_max(program_state, a, b);
        }
    }
    return 0;
}


int dispatch_unary(__private ProgramState *program_state, int op, int a) {
    if (op >= 0 && (size_t)op < NUM_UNARY_OPS) {
        switch (op) {
            case SELECT_UNARY_NEG: return unary_neg(program_state, a);
            case SELECT_UNARY_NOT: return unary_not(program_state, a);
            case SELECT_UNARY_ABS: return unary_abs(program_state, a);
            case SELECT_UNARY_SQUARE: return unary_square(program_state, a);
            case SELECT_UNARY_DOUBLE: return unary_double(program_state, a);
            case SELECT_UNARY_HALF: return unary_half(program_state, a);
        }
    }
    return a;
}

/* Simulated virtual dispatch - struct with function pointers */

typedef struct {
    int variant;
    int data;
} Object;

/* Implementation A */

int impl_a_get(__private ProgramState *program_state, Object *self) {
    Object *obj = self;
    return obj->data;
}


void impl_a_set(__private ProgramState *program_state, Object *self, int value) {
    Object *obj = self;
    obj->data = value;
}


int impl_a_process(__private ProgramState *program_state, Object *self, int input) {
    Object *obj = self;
    return obj->data + input;
}



/* Implementation B */

int impl_b_get(__private ProgramState *program_state, Object *self) {
    Object *obj = self;
    return obj->data * 2;
}


void impl_b_set(__private ProgramState *program_state, Object *self, int value) {
    Object *obj = self;
    obj->data = value / 2;
}


int impl_b_process(__private ProgramState *program_state, Object *self, int input) {
    Object *obj = self;
    return obj->data * input;
}



/* Callback pattern */
typedef int Callback;


void callback_accumulate(__private ProgramState *program_state, int value, int64_t *context) {
    int64_t *acc = context;
    *acc += value;
}


void callback_multiply(__private ProgramState *program_state, int value, int64_t *context) {
    int64_t *acc = context;
    *acc *= value;
}


void iterate_with_callback(__private ProgramState *program_state, const int *arr, int len, Callback cb, int64_t *context) {
    for (int i = 0; i < len; i++) {
        if (cb) callback_multiply(program_state, arr[i], context); else callback_accumulate(program_state, arr[i], context);
    }
}

/* Higher-order function simulation */

int apply_twice(__private ProgramState *program_state, UnaryOp f, int x) {
    return dispatch_unary(program_state, f, dispatch_unary(program_state, f, x));
}


int compose(__private ProgramState *program_state, UnaryOp f, UnaryOp g, int x) {
    return dispatch_unary(program_state, f, dispatch_unary(program_state, g, x));
}


int fold_left(__private ProgramState *program_state, BinaryOp f, const int *arr, int len, int init) {
    int acc = init;
    for (int i = 0; i < len; i++) {
        acc = dispatch_binary(program_state, f, acc, arr[i]);
    }
    return acc;
}

/* State machine with function pointer transitions */
typedef struct StateMachine StateMachine;
typedef enum { IDLE, RUNNING, PAUSED } StateHandler;

struct StateMachine {
    StateHandler current_state;
    int64_t accumulator;
    int counter;
};

 void state_idle(__private ProgramState *program_state, StateMachine *sm, int input);
 void state_running(__private ProgramState *program_state, StateMachine *sm, int input);
 void state_paused(__private ProgramState *program_state, StateMachine *sm, int input);

void state_idle(__private ProgramState *program_state, StateMachine *sm, int input) {
    if (input > 0) {
        sm->accumulator = input;
        sm->counter = 0;
        sm->current_state = RUNNING;
    }
}

void state_running(__private ProgramState *program_state, StateMachine *sm, int input) {
    sm->accumulator += input;
    sm->counter++;
    if (sm->counter >= 10) {
        sm->current_state = PAUSED;
    } else if (input < 0) {
        sm->current_state = IDLE;
    }
}

void state_paused(__private ProgramState *program_state, StateMachine *sm, int input) {
    if (input == 0) {
        sm->current_state = IDLE;
    } else if (input > 0) {
        sm->counter = 0;
        sm->current_state = RUNNING;
    }
}


void step_machine(__private ProgramState *program_state, StateMachine *sm, int input) {
    switch (sm->current_state) {
        case IDLE: state_idle(program_state, sm, input); break;
        case RUNNING: state_running(program_state, sm, input); break;
        case PAUSED: state_paused(program_state, sm, input); break;
    }
}

int program_main(__private ProgramState *program_state) {
    int64_t result = 0;
    int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    /* Virtual dispatch simulation */
    Object obj_a = {0, 100};
    Object obj_b = {1, 100};

    /* State machine */
    StateMachine sm = {IDLE, 0, 0};

    for (int i = 0; i < 100000; i++) {
        /* Table dispatch */
        result += dispatch_binary(program_state, i % NUM_BINARY_OPS, i, i % 10 + 1);
        result += dispatch_unary(program_state, i % NUM_UNARY_OPS, i);

        /* Virtual dispatch */
        Object *obj = (i % 2 == 0 ? &obj_a : &obj_b);
        result += (obj->variant ? impl_b_get(program_state, obj) : impl_a_get(program_state, obj));
        if (obj->variant) impl_b_set(program_state, obj, i); else impl_a_set(program_state, obj, i);
        result += (obj->variant ? impl_b_process(program_state, obj, i % 100) : impl_a_process(program_state, obj, i % 100));

        /* Callback iteration */
        int64_t acc = 0;
        iterate_with_callback(program_state, arr, 10,
            i % 2 == 0 ? 0 : 1,
            &acc);
        if (i % 2 == 1) {
            acc = 1; /* Reset for multiply */
            iterate_with_callback(program_state, arr, 5, 1, &acc);
        }
        result += acc;

        /* Higher-order functions */
        result += apply_twice(program_state, SELECT_UNARY_DOUBLE, i % 100);
        result += compose(program_state, SELECT_UNARY_SQUARE, SELECT_UNARY_ABS, -(i % 50));
        result += fold_left(program_state, SELECT_OP_ADD, arr, 10, 0);
        result += fold_left(program_state, i % 2 == 0 ? SELECT_OP_ADD : SELECT_OP_MUL, arr, 5, i % 2 == 0 ? 0 : 1);

        /* State machine */
        step_machine(program_state, &sm, (i % 20) - 5);
        result += sm.accumulator;
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
