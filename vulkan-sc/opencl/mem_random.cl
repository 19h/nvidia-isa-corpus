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
 * Random/Indirect Memory Access Patterns
 *
 * Tests pointer chasing and irregular access patterns.
 * Exercises memory latency and TLB behavior.
 *
 * Features exercised:
 *   - Pointer chasing (linked structures)
 *   - Index-based indirection
 *   - Hash table access patterns
 *   - Tree traversal patterns
 */




#define ARRAY_SIZE 4096
#define HASH_SIZE 1024



/* Linked list node */
typedef struct Node {
    int32_t value;
    struct Node *next;
} Node;

/* Tree node */
typedef struct TreeNode {
    int32_t value;
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;

/* Arrays for indirect access */




/* Linked list storage */



/* Tree storage */



/* Simple hash table */
typedef struct {
    int32_t key;
    int32_t value;
    int32_t next_idx; /* -1 if none */
} HashEntry;




/* Simple PRNG for reproducible "random" access */


typedef struct {
    volatile int64_t sink;
    int32_t data_array[ARRAY_SIZE];
    size_t index_array[ARRAY_SIZE];
    size_t permutation[ARRAY_SIZE];
    Node nodes[ARRAY_SIZE];
    Node *list_head;
    TreeNode tree_nodes[ARRAY_SIZE];
    TreeNode *tree_root;
    HashEntry hash_table[HASH_SIZE];
    int32_t hash_buckets[HASH_SIZE];
    uint32_t prng_state;
} ProgramState;


uint32_t prng_next(__private ProgramState *program_state) {
    program_state->prng_state = program_state->prng_state * 1103515245 + 12345;
    return program_state->prng_state;
}

/* Pointer chasing through linked list */

int64_t pointer_chase_list(__private ProgramState *program_state, Node *head, int steps) {
    int64_t sum = 0;
    Node *current = head;
    for (int i = 0; i < steps && current != NULL; i++) {
        sum += current->value;
        current = current->next;
    }
    return sum;
}

/* Index-based indirect access */

int64_t indirect_access(__private ProgramState *program_state, const int32_t *data, const size_t *indices, size_t count) {
    int64_t sum = 0;
    for (size_t i = 0; i < count; i++) {
        sum += data[indices[i]];
    }
    return sum;
}

/* Double indirection */

int64_t double_indirect(__private ProgramState *program_state, const int32_t *data, const size_t *idx1, const size_t *idx2, size_t count) {
    int64_t sum = 0;
    for (size_t i = 0; i < count; i++) {
        sum += data[idx1[idx2[i]]];
    }
    return sum;
}

/* Chained index chasing (like pointer chasing but with indices) */

int64_t index_chase(__private ProgramState *program_state, const int32_t *data, const size_t *next, size_t start, int steps) {
    int64_t sum = 0;
    size_t idx = start;
    for (int i = 0; i < steps; i++) {
        sum += data[idx];
        idx = next[idx];
        if (idx >= ARRAY_SIZE) break;
    }
    return sum;
}

/* Permutation access */

int64_t permuted_access(__private ProgramState *program_state, const int32_t *data, const size_t *perm, size_t len) {
    int64_t sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum += data[perm[i]];
    }
    return sum;
}

/* Permutation write (scatter) */

void permuted_write(__private ProgramState *program_state, int32_t *data, const size_t *perm, const int32_t *values, size_t len) {
    for (size_t i = 0; i < len; i++) {
        data[perm[i]] = values[i];
    }
}

/* Hash table lookup */

uint32_t hash_function(__private ProgramState *program_state, int32_t key) {
    uint32_t h = (uint32_t)key;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h % HASH_SIZE;
}


int32_t hash_lookup(__private ProgramState *program_state, int32_t key) {
    uint32_t bucket = hash_function(program_state, key);
    int32_t idx = program_state->hash_buckets[bucket];

    while (idx >= 0) {
        if (program_state->hash_table[idx].key == key) {
            return program_state->hash_table[idx].value;
        }
        idx = program_state->hash_table[idx].next_idx;
    }

    return -1; /* Not found */
}

/* Tree traversal - preorder */

int64_t tree_preorder(__private ProgramState *program_state, TreeNode *node) {
    if (node == NULL) return 0;
    return node->value + tree_preorder(program_state, node->left) + tree_preorder(program_state, node->right);
}

/* Tree traversal - iterative with stack simulation */

int64_t tree_iterative(__private ProgramState *program_state, TreeNode *root) {
    if (root == NULL) return 0;

    TreeNode *stack[64];
    int sp = 0;
    int64_t sum = 0;

    stack[sp++] = root;
    while (sp > 0) {
        TreeNode *node = stack[--sp];
        sum += node->value;
        if (node->right && sp < 63) stack[sp++] = node->right;
        if (node->left && sp < 63) stack[sp++] = node->left;
    }
    return sum;
}

/* Binary search tree lookup */

TreeNode *bst_lookup(__private ProgramState *program_state, TreeNode *root, int32_t key) {
    while (root != NULL) {
        if (key < root->value) {
            root = root->left;
        } else if (key > root->value) {
            root = root->right;
        } else {
            return root;
        }
    }
    return NULL;
}

/* Random access pattern */

int64_t random_access(__private ProgramState *program_state, const int32_t *data, size_t len, int accesses) {
    int64_t sum = 0;
    for (int i = 0; i < accesses; i++) {
        size_t idx = prng_next(program_state) % len;
        sum += data[idx];
    }
    return sum;
}

/* Dependent random access */

int64_t dependent_random(__private ProgramState *program_state, const int32_t *data, size_t len, int accesses) {
    int64_t sum = 0;
    size_t idx = 0;
    for (int i = 0; i < accesses; i++) {
        sum += data[idx];
        idx = ((size_t)data[idx] * 7 + 13) % len;
    }
    return sum;
}

/* Strided random access (random within blocks) */

int64_t strided_random(__private ProgramState *program_state, const int32_t *data, size_t len, size_t block_size, int accesses) {
    int64_t sum = 0;
    for (int i = 0; i < accesses; i++) {
        size_t block = (prng_next(program_state) % (len / block_size)) * block_size;
        size_t offset = prng_next(program_state) % block_size;
        sum += data[block + offset];
    }
    return sum;
}

/* Initialize data structures */
void init_structures(__private ProgramState *program_state) {
    /* Initialize data and indices */
    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        program_state->data_array[i] = (int32_t)i;
        program_state->index_array[i] = (i * 7 + 13) % ARRAY_SIZE;
        program_state->permutation[i] = i;
    }

    /* Fisher-Yates shuffle for permutation */
    program_state->prng_state = 42;
    for (size_t i = ARRAY_SIZE - 1; i > 0; i--) {
        size_t j = prng_next(program_state) % (i + 1);
        size_t tmp = program_state->permutation[i];
        program_state->permutation[i] = program_state->permutation[j];
        program_state->permutation[j] = tmp;
    }

    /* Build linked list (randomized order) */
    program_state->list_head = &program_state->nodes[program_state->permutation[0]];
    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        program_state->nodes[program_state->permutation[i]].value = (int32_t)program_state->permutation[i];
        if (i + 1 < ARRAY_SIZE) {
            program_state->nodes[program_state->permutation[i]].next = &program_state->nodes[program_state->permutation[i + 1]];
        } else {
            program_state->nodes[program_state->permutation[i]].next = NULL;
        }
    }

    /* Build simple binary tree */
    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        program_state->tree_nodes[i].value = (int32_t)i;
        program_state->tree_nodes[i].left = NULL;
        program_state->tree_nodes[i].right = NULL;
    }

    /* Connect as balanced-ish tree */
    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        size_t left = 2 * i + 1;
        size_t right = 2 * i + 2;
        if (left < ARRAY_SIZE) program_state->tree_nodes[i].left = &program_state->tree_nodes[left];
        if (right < ARRAY_SIZE) program_state->tree_nodes[i].right = &program_state->tree_nodes[right];
    }
    program_state->tree_root = &program_state->tree_nodes[0];

    /* Initialize hash table */
    for (size_t i = 0; i < HASH_SIZE; i++) {
        program_state->hash_buckets[i] = -1;
    }

    for (size_t i = 0; i < HASH_SIZE; i++) {
        int32_t key = (int32_t)(i * 17);
        uint32_t bucket = hash_function(program_state, key);

        program_state->hash_table[i].key = key;
        program_state->hash_table[i].value = (int32_t)(i * 100);
        program_state->hash_table[i].next_idx = program_state->hash_buckets[bucket];
        program_state->hash_buckets[bucket] = (int32_t)i;
    }
}

int program_main(__private ProgramState *program_state) {
    int64_t result = 0;

    init_structures(program_state);

    int32_t values[ARRAY_SIZE];
    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        values[i] = (int32_t)(i * 3);
    }

    for (int iter = 0; iter < 2000; iter++) {
        /* Pointer chasing */
        result += pointer_chase_list(program_state, program_state->list_head, 1000);

        /* Indirect access */
        result += indirect_access(program_state, program_state->data_array, program_state->index_array, ARRAY_SIZE);

        /* Double indirection */
        result += double_indirect(program_state, program_state->data_array, program_state->index_array, program_state->permutation, 1024);

        /* Index chasing */
        result += index_chase(program_state, program_state->data_array, program_state->index_array, iter % ARRAY_SIZE, 500);

        /* Permuted access */
        result += permuted_access(program_state, program_state->data_array, program_state->permutation, ARRAY_SIZE);

        /* Permuted write */
        if (iter % 100 == 0) {
            permuted_write(program_state, program_state->data_array, program_state->permutation, values, ARRAY_SIZE);
        }

        /* Hash table lookups */
        for (int i = 0; i < 100; i++) {
            int32_t key = (int32_t)((iter + i) * 17) % (HASH_SIZE * 17);
            result += hash_lookup(program_state, key);
        }

        /* Tree traversal */
        if (iter % 50 == 0) {
            result += tree_preorder(program_state, program_state->tree_root);
        }
        result += tree_iterative(program_state, program_state->tree_root);

        /* BST lookups */
        for (int i = 0; i < 20; i++) {
            TreeNode *found = bst_lookup(program_state, program_state->tree_root, (iter + i) % ARRAY_SIZE);
            if (found) result += found->value;
        }

        /* Random access */
        program_state->prng_state = (uint32_t)iter;
        result += random_access(program_state, program_state->data_array, ARRAY_SIZE, 500);

        /* Dependent random */
        result += dependent_random(program_state, program_state->data_array, ARRAY_SIZE, 200);

        /* Strided random */
        result += strided_random(program_state, program_state->data_array, ARRAY_SIZE, 64, 300);
    }

    program_state->sink = result;
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {.prng_state = 12345};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)(((ulong)storage.sink >> (8u * i)) & 255u);
}
