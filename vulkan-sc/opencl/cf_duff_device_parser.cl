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
// SPDX-License-Identifier: MIT
/*
 * Duff's-device parser/state machine.
 *
 * The parser consumes deterministic in-memory command streams through an
 * unrolled Duff loop. Each byte also drives nested switches, parser state,
 * indirect parser actions, pointer movement, and rolling integer state.
 */






typedef enum {
    ST_SEEK = 0,
    ST_KEY,
    ST_VALUE,
    ST_ARRAY,
    ST_QUOTE,
    ST_SKIP
} ParseState;

typedef enum {
    CC_ALPHA = 0,
    CC_DIGIT,
    CC_SIGN,
    CC_EQ,
    CC_SEMI,
    CC_COMMA,
    CC_OPEN,
    CC_CLOSE,
    CC_QUOTE,
    CC_OTHER
} CharClass;

typedef struct Parser Parser;

struct Parser {
    size_t cursor;
    size_t length;
    uint32_t hash;
    uint32_t key;
    uint32_t value;
    uint32_t items;
    uint32_t depth;
    uint32_t state;
    uint32_t sign;
    uint32_t lanes[8];
    uint8_t window[64];
};




typedef struct {
    volatile uint32_t cf_duff_device_parser_fence;
    volatile uint64_t cf_duff_device_parser_sink;
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

static uint32_t lower_alpha(__private ProgramState *program_state, uint32_t ch) {
    if (ch >= 'A' && ch <= 'Z') {
        return ch + ('a' - 'A');
    }
    return ch;
}


static uint32_t classify(__private ProgramState *program_state, uint32_t ch) {
    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_') {
        return CC_ALPHA;
    }
    if (ch >= '0' && ch <= '9') {
        return CC_DIGIT;
    }
    switch (ch) {
        case '+':
        case '-':
            return CC_SIGN;
        case '=':
        case ':':
            return CC_EQ;
        case ';':
        case '\n':
            return CC_SEMI;
        case ',':
            return CC_COMMA;
        case '[':
        case '{':
        case '(':
            return CC_OPEN;
        case ']':
        case '}':
        case ')':
            return CC_CLOSE;
        case '"':
        case '\'':
            return CC_QUOTE;
        default:
            return CC_OTHER;
    }
}


static uint32_t action_lace(__private ProgramState *program_state, Parser *parser, uint32_t ch, uint32_t cls) {
    uint32_t lane = (parser->items + cls + parser->state) & 7u;
    parser->lanes[lane] ^= mix32(program_state, ch + parser->hash + parser->key);
    return rotl32(program_state, parser->lanes[lane], lane + 1u);
}


static uint32_t action_stride(__private ProgramState *program_state, Parser *parser, uint32_t ch, uint32_t cls) {
    uint32_t pos = (uint32_t)parser->cursor;
    uint32_t slot = (pos * 5u + cls + parser->depth) & 63u;
    parser->window[slot] = (uint8_t)(parser->window[(slot + 17u) & 63u] ^ ch ^ parser->state);
    return mix32(program_state, (uint32_t)parser->window[slot] + pos + (parser->depth << 8u));
}


static uint32_t action_gate(__private ProgramState *program_state, Parser *parser, uint32_t ch, uint32_t cls) {
    uint32_t lane = (ch + parser->state + parser->depth) & 7u;
    uint32_t mask = 0u - ((cls ^ parser->items) & 1u);
    parser->lanes[lane] = (parser->lanes[lane] & mask) |
                          (rotl32(program_state, parser->hash, lane + 9u) & ~mask);
    return parser->lanes[lane] ^ mask;
}


static uint32_t action_echo(__private ProgramState *program_state, Parser *parser, uint32_t ch, uint32_t cls) {
    uint32_t lane = (parser->key + parser->value + cls) & 7u;
    parser->lanes[lane] += mix32(program_state, (ch << 16u) ^ parser->hash ^ parser->items);
    return parser->lanes[lane] + rotl32(program_state, ch + cls, (lane & 7u) + 1u);
}



static void emit_value(__private ProgramState *program_state, Parser *parser, uint32_t tag) {
    uint32_t encoded = parser->sign != 0u ? (~parser->value + 1u) : parser->value;
    uint32_t lane = (parser->items + parser->depth + tag) & 7u;

    parser->lanes[lane] ^= mix32(program_state, parser->key + rotl32(program_state, encoded, lane + 3u) + tag);
    parser->hash += mix32(program_state, parser->lanes[lane] ^ parser->items ^ parser->state);
    parser->items++;
    parser->value = 0u;
    parser->key = mix32(program_state, parser->key ^ tag) & UINT32_C(0x00ffffff);
    parser->sign = 0u;
}


static void parse_one(__private ProgramState *program_state, Parser *parser, const unsigned char *text) {
    uint32_t pos = (uint32_t)parser->cursor;
    uint32_t ch = text[parser->cursor++];
    uint32_t cls = classify(program_state, ch);
    uint32_t old_state = parser->state;
    uint32_t action = (cls ^ old_state ^ (ch >> 4u)) & 3u;

    parser->window[(pos + old_state * 11u) & 63u] =
        (uint8_t)(ch ^ parser->depth ^ parser->items);
    switch (action) {
        case 0: parser->hash ^= action_lace(program_state, parser, ch, cls); break;
        case 1: parser->hash ^= action_stride(program_state, parser, ch, cls); break;
        case 2: parser->hash ^= action_gate(program_state, parser, ch, cls); break;
        case 3: parser->hash ^= action_echo(program_state, parser, ch, cls); break;
    }
    program_state->cf_duff_device_parser_fence = parser->hash ^ pos;

    switch (parser->state) {
        case ST_SEEK:
            switch (cls) {
                case CC_ALPHA:
                    parser->key = lower_alpha(program_state, ch) ^ UINT32_C(0x9e37);
                    parser->state = ST_KEY;
                    break;
                case CC_OPEN:
                    parser->depth++;
                    parser->hash += mix32(program_state, ch + parser->depth);
                    break;
                case CC_CLOSE:
                    if (parser->depth != 0u) {
                        parser->depth--;
                    }
                    parser->hash ^= mix32(program_state, ch + parser->depth);
                    break;
                default:
                    parser->hash += cls + ch;
                    break;
            }
            break;

        case ST_KEY:
            switch (cls) {
                case CC_ALPHA:
                case CC_DIGIT:
                    parser->key = parser->key * 33u + lower_alpha(program_state, ch) + (parser->depth << 3u);
                    break;
                case CC_EQ:
                    parser->state = ST_VALUE;
                    parser->value = 0u;
                    parser->sign = 0u;
                    break;
                case CC_OPEN:
                    parser->depth++;
                    parser->state = ST_ARRAY;
                    break;
                case CC_SEMI:
                case CC_COMMA:
                    emit_value(program_state, parser, cls);
                    parser->state = ST_SEEK;
                    break;
                default:
                    parser->state = ST_SKIP;
                    parser->hash ^= mix32(program_state, parser->key + ch);
                    break;
            }
            break;

        case ST_VALUE:
            switch (cls) {
                case CC_DIGIT:
                    parser->value = parser->value * 10u + (ch - '0');
                    break;
                case CC_SIGN:
                    parser->sign ^= (ch == '-') ? 1u : 0u;
                    parser->hash += ch;
                    break;
                case CC_ALPHA:
                    parser->value = rotl32(program_state, parser->value ^ lower_alpha(program_state, ch), 5u);
                    break;
                case CC_QUOTE:
                    parser->state = ST_QUOTE;
                    break;
                case CC_OPEN:
                    parser->depth++;
                    parser->state = ST_ARRAY;
                    break;
                case CC_COMMA:
                    emit_value(program_state, parser, cls);
                    parser->state = parser->depth == 0u ? ST_SEEK : ST_ARRAY;
                    break;
                case CC_SEMI:
                case CC_CLOSE:
                    emit_value(program_state, parser, cls);
                    if (cls == CC_CLOSE && parser->depth != 0u) {
                        parser->depth--;
                    }
                    parser->state = ST_SEEK;
                    break;
                default:
                    parser->hash ^= mix32(program_state, ch + parser->value);
                    break;
            }
            break;

        case ST_ARRAY:
            switch (cls) {
                case CC_DIGIT:
                    parser->value = parser->value * 10u + (ch - '0') + parser->depth;
                    parser->state = ST_VALUE;
                    break;
                case CC_SIGN:
                    parser->sign ^= (ch == '-') ? 1u : 0u;
                    parser->state = ST_VALUE;
                    break;
                case CC_ALPHA:
                    parser->key = parser->key * 17u + lower_alpha(program_state, ch) + parser->depth;
                    break;
                case CC_EQ:
                    parser->state = ST_VALUE;
                    break;
                case CC_OPEN:
                    parser->depth++;
                    parser->hash += mix32(program_state, ch ^ parser->depth);
                    break;
                case CC_CLOSE:
                    emit_value(program_state, parser, cls);
                    if (parser->depth != 0u) {
                        parser->depth--;
                    }
                    parser->state = parser->depth == 0u ? ST_SEEK : ST_ARRAY;
                    break;
                case CC_COMMA:
                case CC_SEMI:
                    emit_value(program_state, parser, cls);
                    parser->state = cls == CC_SEMI ? ST_SEEK : ST_ARRAY;
                    break;
                case CC_QUOTE:
                    parser->state = ST_QUOTE;
                    break;
                default:
                    parser->hash += cls + parser->depth;
                    break;
            }
            break;

        case ST_QUOTE:
            switch (cls) {
                case CC_QUOTE:
                    emit_value(program_state, parser, cls);
                    parser->state = parser->depth == 0u ? ST_SEEK : ST_ARRAY;
                    break;
                default:
                    parser->value = rotl32(program_state, parser->value + ch + parser->key, 3u);
                    parser->hash ^= mix32(program_state, ch + parser->value + parser->depth);
                    break;
            }
            break;

        default:
            switch (cls) {
                case CC_SEMI:
                case CC_COMMA:
                    parser->state = ST_SEEK;
                    parser->value = 0u;
                    parser->sign = 0u;
                    break;
                case CC_ALPHA:
                    parser->key = lower_alpha(program_state, ch);
                    parser->state = ST_KEY;
                    break;
                default:
                    parser->hash += mix32(program_state, ch ^ cls);
                    break;
            }
            break;
    }

    if (((parser->hash ^ parser->lanes[(pos + cls) & 7u]) & 15u) == 7u) {
        parser->state = (parser->state + ST_VALUE + cls) % (ST_SKIP + 1u);
    }
}


static uint64_t parse_duff(__private ProgramState *program_state, const unsigned char *text, size_t len, uint32_t seed) {
    Parser parser;
    uint64_t result;

    parser.cursor = 0;
    parser.length = len;
    parser.hash = mix32(program_state, seed ^ (uint32_t)len);
    parser.key = 0u;
    parser.value = 0u;
    parser.items = 0u;
    parser.depth = 0u;
    parser.state = ST_SEEK;
    parser.sign = 0u;
    for (uint32_t i = 0u; i < 8u; ++i) {
        parser.lanes[i] = mix32(program_state, seed + i * UINT32_C(0x01010101));
    }
    for (uint32_t i = 0u; i < 64u; ++i) {
        parser.window[i] = (uint8_t)(mix32(program_state, seed ^ i) >> 24u);
    }

    if (len != 0u) {
        // Execute the initial partial group, then the full groups. Sharing
        // the step body keeps the structured SPIR-V small on older compilers.
        size_t groups = (len + 7u) / 8u;
        size_t steps = len & 7u;
        if (steps == 0u)
            steps = 8u;
        do {
#pragma clang loop unroll(disable)
            for (size_t step = 0; step < steps; ++step)
                parse_one(program_state, &parser, text);
            steps = 8u;
        } while (--groups != 0u);
    }

    if (parser.value != 0u || parser.key != 0u || parser.state == ST_VALUE) {
        emit_value(program_state, &parser, UINT32_C(0xff));
    }

    result = (uint64_t)mix32(program_state, parser.hash ^ parser.items ^ parser.depth);
    for (uint32_t i = 0u; i < 8u; ++i) {
        result ^= (uint64_t)mix32(program_state, parser.lanes[i] + i) << ((i & 1u) * 32u);
        result = (result << 7u) | (result >> 57u);
    }
    for (uint32_t i = 0u; i < 64u; i += 9u) {
        result += (uint64_t)parser.window[i] * UINT64_C(0x100000001b3);
    }

    return result ^ (uint64_t)(parser.cursor == parser.length ? UINT32_C(0xa5a55a5a) : 0u);
}

int program_main(__private ProgramState *program_state) {
    const unsigned char script0[] =
        "alpha=19;beta=[3,5,8,13];gamma=-21;zeta={q:7,r:11};omega=\"m0r0k\";";
    const unsigned char script1[] =
        "root{left=144,right=[1,-2,3],note='fold'};tail=377;mask=(a:5,b:9);";
    const unsigned char script2[] =
        "mix=42;maze=[north=6,east=10,south=-4,west=2];quote=\"duff/device\";end=1;";
    uint64_t checksum = UINT64_C(0x6a09e667f3bcc909);

    checksum ^= parse_duff(program_state, script0, sizeof(script0) - 1u, UINT32_C(0x10203040));
    checksum = (checksum << 11u) | (checksum >> 53u);
    checksum ^= parse_duff(program_state, script1, sizeof(script1) - 1u, (uint32_t)checksum);
    checksum = (checksum << 17u) | (checksum >> 47u);
    checksum ^= parse_duff(program_state, script2, sizeof(script2) - 1u, (uint32_t)(checksum >> 32u));

    program_state->cf_duff_device_parser_sink = checksum;
    printf("cf_duff_device_parser %016" PRIx64 "\n", program_state->cf_duff_device_parser_sink);
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.cf_duff_device_parser_sink); ++i)
        results[1 + i] = (uint)(((ulong)storage.cf_duff_device_parser_sink >> (8u * i)) & 255u);
}
