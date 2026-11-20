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
 * Struct Return Values
 *
 * Functions returning structs of various sizes.
 * Small structs may be returned in registers, large ones via hidden pointer.
 *
 * Features exercised:
 *   - Register return (small structs)
 *   - Hidden pointer return (large structs)
 *   - Struct packing and alignment
 */





/* Small struct - may fit in registers */
typedef struct {
    int x, y;
} Point2D;

typedef struct {
    int x, y, z;
} Point3D;

typedef struct {
    int x, y, z, w;
} Point4D;

/* Larger structs - require hidden pointer */
typedef struct {
    int data[8];
} Array8;

typedef struct {
    int data[16];
} Array16;

/* Mixed types */
typedef struct {
    int i;
    double d;
} IntDouble;

typedef struct {
    float f;
    int i;
    double d;
} FloatIntDouble;

/* Nested */
typedef struct {
    Point2D p1;
    Point2D p2;
} TwoPoints;

typedef struct {
    volatile int64_t sink;
} ProgramState;


Point2D make_point2d(__private ProgramState *program_state, int x, int y) {
    Point2D p = {x, y};
    return p;
}


Point3D make_point3d(__private ProgramState *program_state, int x, int y, int z) {
    Point3D p = {x, y, z};
    return p;
}


Point4D make_point4d(__private ProgramState *program_state, int x, int y, int z, int w) {
    Point4D p = {x, y, z, w};
    return p;
}


Array8 make_array8(__private ProgramState *program_state, int base) {
    Array8 a;
    for (int i = 0; i < 8; i++) {
        a.data[i] = base + i;
    }
    return a;
}


Array16 make_array16(__private ProgramState *program_state, int base) {
    Array16 a;
    for (int i = 0; i < 16; i++) {
        a.data[i] = base + i;
    }
    return a;
}


IntDouble make_int_double(__private ProgramState *program_state, int i, double d) {
    IntDouble s = {i, d};
    return s;
}


FloatIntDouble make_fid(__private ProgramState *program_state, float f, int i, double d) {
    FloatIntDouble s = {f, i, d};
    return s;
}


TwoPoints make_two_points(__private ProgramState *program_state, int x1, int y1, int x2, int y2) {
    TwoPoints tp = {{x1, y1}, {x2, y2}};
    return tp;
}


Point2D add_points(__private ProgramState *program_state, Point2D a, Point2D b) {
    Point2D r = {a.x + b.x, a.y + b.y};
    return r;
}

int program_main(__private ProgramState *program_state) {
    int64_t result = 0;

    for (int i = 0; i < 100000; i++) {
        Point2D p2 = make_point2d(program_state, i, i + 1);
        result += p2.x + p2.y;

        Point3D p3 = make_point3d(program_state, i, i + 1, i + 2);
        result += p3.x + p3.y + p3.z;

        Point4D p4 = make_point4d(program_state, i, i + 1, i + 2, i + 3);
        result += p4.x + p4.y + p4.z + p4.w;

        Array8 a8 = make_array8(program_state, i);
        for (int j = 0; j < 8; j++) result += a8.data[j];

        Array16 a16 = make_array16(program_state, i);
        for (int j = 0; j < 16; j++) result += a16.data[j];

        IntDouble id = make_int_double(program_state, i, (double)i * 0.5);
        result += id.i + (int64_t)id.d;

        FloatIntDouble fid = make_fid(program_state, (float)i, i, (double)i);
        result += (int64_t)fid.f + fid.i + (int64_t)fid.d;

        TwoPoints tp = make_two_points(program_state, i, i+1, i+2, i+3);
        result += tp.p1.x + tp.p1.y + tp.p2.x + tp.p2.y;

        Point2D sum = add_points(program_state, p2, make_point2d(program_state, 1, 1));
        result += sum.x + sum.y;
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
