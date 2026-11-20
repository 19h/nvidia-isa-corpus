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
 * cf_irreducible_state_maze.c
 *
 * Goto-heavy state maze with multi-entry loops and opaque-ish predicates.
 * Stresses PHIs, switches, pointer aliasing, and hard-to-structure control flow.
 */








typedef struct {
    uint32_t pc;
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint16_t stack[8];
    uint8_t tape[64];
} Maze;

typedef struct {
    volatile uint32_t cf_irreducible_fence;
    volatile uint64_t cf_irreducible_sink;
} ProgramState;

static uint32_t rotl32(__private ProgramState *program_state, uint32_t value, unsigned amount) {
    amount &= 31u;
    return amount == 0u ? value : (value << amount) | (value >> (32u - amount));
}

static uint32_t mix32(__private ProgramState *program_state, uint32_t value) {
    value ^= value >> 16u;
    value *= UINT32_C(0x7feb352d);
    value ^= value >> 15u;
    value *= UINT32_C(0x846ca68b);
    value ^= value >> 16u;
    return value;
}


static uint32_t scramble_slot(__private ProgramState *program_state, Maze *maze, uint32_t index, uint32_t salt) {
    uint8_t *bytes = maze->tape;
    uint16_t *slot = &maze->stack[index & 7u];
    uint8_t *nearby = &bytes[(index * 9u + salt) & 63u];

    uint32_t before = (uint32_t)*slot;
    uint32_t folded = mix32(program_state, before ^ salt ^ bytes[(index + 17u) & 63u]);
    *slot = (uint16_t)(before + (folded & UINT32_C(0xffff)));
    *nearby = (uint8_t)((uint32_t)*nearby ^ (folded >> ((index & 3u) * 8u)));

    return folded ^ (uint32_t)*slot ^ (uint32_t)*nearby;
}


static uint32_t run_maze(__private ProgramState *program_state, Maze *maze, uint32_t seed) {
    uint32_t acc = seed ^ UINT32_C(0x9e3779b9);
    uint32_t tag = (seed >> 3u) & 15u;
    uint32_t latch = mix32(program_state, seed + UINT32_C(0x632be59b));
    uint32_t steps = 0;
    const uint32_t max_steps = 96u + (seed & 31u);

    enum { NODE_Hub, NODE_North, NODE_East, NODE_South, NODE_West, NODE_Cross, NODE_Fold, NODE_Spill, NODE_Trap } node;
    node = ((mix32(program_state, seed) ^ seed ^ program_state->cf_irreducible_fence) & 3u) == 0u ? NODE_Cross : NODE_North;
    while (++steps < max_steps) {
        switch (node) {
            case NODE_Hub: {
                maze->pc = (maze->pc + tag + (acc & 7u)) & 31u;
                acc ^= scramble_slot(program_state, maze, maze->pc + steps, latch);
                program_state->cf_irreducible_fence = acc ^ maze->pc;

                switch ((acc ^ latch ^ maze->pc) & 7u) {
                    case 0u: { node = NODE_North; continue; }
                    case 1u: { node = NODE_East; continue; }
                    case 2u: { node = NODE_South; continue; }
                    case 3u: { node = NODE_West; continue; }
                    case 4u: { node = NODE_Cross; continue; }
                    case 5u: { node = NODE_Fold; continue; }
                    case 6u: { node = NODE_Spill; continue; }
                    default: { node = NODE_Trap; continue; }
                }
            }
            case NODE_North: {
                maze->a += rotl32(program_state, acc + maze->pc, (tag & 7u) + 1u);
                latch ^= maze->a + UINT32_C(0x45d9f3b);
                tag = (tag + ((maze->a >> 27u) | 1u)) & 15u;
                if (((maze->a ^ maze->c ^ program_state->cf_irreducible_fence) & 5u) == 0u) {
                    { node = NODE_Cross; continue; }
                }
                if ((tag & 1u) != 0u) {
                    { node = NODE_East; continue; }
                }
                { node = NODE_Hub; continue; }
            }
            case NODE_East: {
                maze->b ^= rotl32(program_state, latch + maze->stack[tag & 7u], (maze->a >> 29u) + 3u);
                acc += maze->b ^ maze->tape[(maze->b + tag) & 63u];
                if (((acc + maze->d) & 7u) == 2u) {
                    { node = NODE_South; continue; }
                }
                if ((maze->b & UINT32_C(0x1040)) == UINT32_C(0x1000)) {
                    { node = NODE_Fold; continue; }
                }
                { node = NODE_Cross; continue; }
            }
            case NODE_South: {
                maze->c += mix32(program_state, acc ^ maze->b ^ steps);
                maze->tape[(maze->c >> 3u) & 63u] =
                    (uint8_t)(maze->tape[(maze->c >> 11u) & 63u] + tag + steps);
                latch += maze->c | UINT32_C(1);
                if (((maze->c ^ latch) & 3u) == 1u) {
                    { node = NODE_West; continue; }
                }
                if ((steps & 5u) == 5u) {
                    { node = NODE_Cross; continue; }
                }
                { node = NODE_Hub; continue; }
            }
            case NODE_West: {
                maze->d = rotl32(program_state, maze->d ^ acc ^ maze->a, (steps & 15u) + 1u);
                acc ^= maze->d + maze->stack[(steps + tag) & 7u];
                if (((maze->d + program_state->cf_irreducible_fence) & 11u) == 8u) {
                    { node = NODE_North; continue; }
                }
                if ((acc & 9u) == 9u) {
                    { node = NODE_Spill; continue; }
                }
                { node = NODE_Hub; continue; }
            }
            case NODE_Cross: {
                acc = mix32(program_state, acc + maze->a + (maze->b ^ maze->c));
                maze->stack[(acc >> 5u) & 7u] ^=
                    (uint16_t)(acc + maze->d + (uint32_t)maze->tape[acc & 63u]);
                if (((acc ^ tag) & 13u) == 4u) {
                    { node = NODE_South; continue; }
                }
                if (((acc >> 8u) & 3u) == 0u) {
                    { node = NODE_Fold; continue; }
                }
                { node = NODE_West; continue; }
            }
            case NODE_Fold: {
                {
                    uint32_t left = maze->stack[(tag + 1u) & 7u];
                    uint32_t right = maze->tape[(acc + steps) & 63u];
                    acc += mix32(program_state, (left << 16u) | right);
                    maze->a ^= acc + left;
                    maze->c ^= rotl32(program_state, right + latch, (tag & 7u) + 1u);
                }
                if ((maze->a & maze->c & 1u) != 0u) {
                    { node = NODE_East; continue; }
                }
                { node = NODE_Hub; continue; }
            }
            case NODE_Spill: {
                {
                    uint32_t first = (steps + maze->pc) & 63u;
                    uint32_t second = (first + 23u + tag) & 63u;
                    uint8_t tmp = maze->tape[first];
                    maze->tape[first] = (uint8_t)(maze->tape[second] + acc);
                    maze->tape[second] = (uint8_t)(tmp ^ (latch >> 24u));
                    acc ^= ((uint32_t)maze->tape[first] << 8u) | maze->tape[second];
                }
                if (((acc ^ steps) & 6u) == 6u) {
                    { node = NODE_Trap; continue; }
                }
                { node = NODE_Cross; continue; }
            }
            case NODE_Trap: {
                latch = mix32(program_state, latch ^ acc ^ UINT32_C(0xa5a5a5a5));
                maze->b += latch | 1u;
                tag = (tag + 7u + (latch & 3u)) & 15u;
                if ((latch & 0x80u) != 0u) {
                    { node = NODE_North; continue; }
                }
                { node = NODE_Hub; continue; }
            }
        }
    }

    return mix32(program_state, acc ^ latch ^ maze->a ^ rotl32(program_state, maze->b, 7u) ^
                 rotl32(program_state, maze->c, 13u) ^ rotl32(program_state, maze->d, 19u) ^ steps);
}

int program_main(__private ProgramState *program_state) {
    Maze maze;
    uint64_t checksum = UINT64_C(0x6a09e667f3bcc909);

    maze.pc = 0;
    maze.a = UINT32_C(0x243f6a88);
    maze.b = UINT32_C(0x85a308d3);
    maze.c = UINT32_C(0x13198a2e);
    maze.d = UINT32_C(0x03707344);

    for (uint32_t i = 0; i < 8u; ++i) {
        maze.stack[i] = (uint16_t)(UINT32_C(0x1357) + i * UINT32_C(0x2221));
    }
    for (uint32_t i = 0; i < 64u; ++i) {
        maze.tape[i] = (uint8_t)(mix32(program_state, i * UINT32_C(37) + UINT32_C(11)) >> 24u);
    }

    for (uint32_t round = 0; round < 4096u; ++round) {
        uint32_t seed = mix32(program_state, round ^ (uint32_t)checksum ^ (uint32_t)(checksum >> 32u));
        uint32_t out = run_maze(program_state, &maze, seed);
        checksum ^= (uint64_t)out + UINT64_C(0x9e3779b97f4a7c15) +
                    (checksum << 6u) + (checksum >> 2u);
        checksum = (checksum << 17u) | (checksum >> 47u);
    }

    program_state->cf_irreducible_sink = checksum ^ maze.a ^ ((uint64_t)maze.d << 32u);
    printf("cf_irreducible_state_maze %016" PRIx64 "\n", program_state->cf_irreducible_sink);
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.cf_irreducible_sink); ++i)
        results[1 + i] = (uint)(((ulong)storage.cf_irreducible_sink >> (8u * i)) & 255u);
}
