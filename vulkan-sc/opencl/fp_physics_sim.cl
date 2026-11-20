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
 * N-Body Gravitational Simulation
 *
 * Simulates gravitational interactions between particles.
 * Heavy use of sqrt, division, multiply-accumulate.
 *
 * Features exercised:
 *   - FP sqrt (or approximation)
 *   - FP division
 *   - FP multiply-accumulate
 *   - Vectorizable force calculation
 */

#define NUM_BODIES 128
#define TIME_STEPS 100
#define DT 0.01



typedef struct {
    double x, y, z;       /* Position */
    double vx, vy, vz;    /* Velocity */
    double ax, ay, az;    /* Acceleration */
    double mass;
} Body;



/* Approximate sqrt using Newton-Raphson */
typedef struct {
    volatile double sink;
    Body bodies[NUM_BODIES];
} ProgramState;

static double approx_sqrt(__private ProgramState *program_state, double x) {
    if (x <= 0.0) return 0.0;

    /* Initial guess using bit manipulation */
    double guess = x * 0.5;

    /* Newton-Raphson iterations */
    for (int i = 0; i < 10; i++) {
        guess = 0.5 * (guess + x / guess);
    }

    return guess;
}

/* Approximate 1/sqrt(x) - important for gravitational force */
static double approx_rsqrt(__private ProgramState *program_state, double x) {
    if (x <= 0.0) return 0.0;

    double half_count = 0.5 * x;
    double guess = 1.0 / approx_sqrt(program_state, x);

    /* One Newton-Raphson refinement for rsqrt */
    guess = guess * (1.5 - half_count * guess * guess);
    guess = guess * (1.5 - half_count * guess * guess);

    return guess;
}

/* Initialize bodies in a pseudo-random configuration */
static void init_bodies(__private ProgramState *program_state) {
    unsigned int seed = 42;

    for (int i = 0; i < NUM_BODIES; i++) {
        /* Position in unit cube */
        seed = seed * 1103515245 + 12345;
        program_state->bodies[i].x = (double)((seed >> 16) & 0xFFFF) / 65536.0 - 0.5;
        seed = seed * 1103515245 + 12345;
        program_state->bodies[i].y = (double)((seed >> 16) & 0xFFFF) / 65536.0 - 0.5;
        seed = seed * 1103515245 + 12345;
        program_state->bodies[i].z = (double)((seed >> 16) & 0xFFFF) / 65536.0 - 0.5;

        /* Small random velocity */
        seed = seed * 1103515245 + 12345;
        program_state->bodies[i].vx = 0.01 * ((double)((seed >> 16) & 0xFFFF) / 65536.0 - 0.5);
        seed = seed * 1103515245 + 12345;
        program_state->bodies[i].vy = 0.01 * ((double)((seed >> 16) & 0xFFFF) / 65536.0 - 0.5);
        seed = seed * 1103515245 + 12345;
        program_state->bodies[i].vz = 0.01 * ((double)((seed >> 16) & 0xFFFF) / 65536.0 - 0.5);

        program_state->bodies[i].ax = 0.0;
        program_state->bodies[i].ay = 0.0;
        program_state->bodies[i].az = 0.0;

        program_state->bodies[i].mass = 1.0 + 0.5 * ((double)((seed >> 16) & 0xFFFF) / 65536.0);
    }
}

/* Compute gravitational forces between all pairs */
static void compute_forces(__private ProgramState *program_state) {
    /* Reset accelerations */
    for (int i = 0; i < NUM_BODIES; i++) {
        program_state->bodies[i].ax = 0.0;
        program_state->bodies[i].ay = 0.0;
        program_state->bodies[i].az = 0.0;
    }

    /* Compute pairwise forces */
    const double G = 0.001;  /* Gravitational constant */
    const double softening = 0.01;  /* Avoid singularity */

    for (int i = 0; i < NUM_BODIES; i++) {
        for (int j = i + 1; j < NUM_BODIES; j++) {
            /* Distance vector */
            double dx = program_state->bodies[j].x - program_state->bodies[i].x;
            double dy = program_state->bodies[j].y - program_state->bodies[i].y;
            double dz = program_state->bodies[j].z - program_state->bodies[i].z;

            /* Distance squared */
            double dist_sq = dx * dx + dy * dy + dz * dz + softening;

            /* Force magnitude: G * m1 * m2 / r^2 */
            /* For acceleration: F/m = G * m_other / r^2 */
            /* We need 1/r^3 for the vector form */
            double inv_dist = approx_rsqrt(program_state, dist_sq);
            double inv_dist3 = inv_dist * inv_dist * inv_dist;

            double force_factor = G * inv_dist3;

            /* Force on i from j */
            double fi = force_factor * program_state->bodies[j].mass;
            program_state->bodies[i].ax += fi * dx;
            program_state->bodies[i].ay += fi * dy;
            program_state->bodies[i].az += fi * dz;

            /* Force on j from i (Newton's 3rd law) */
            double fj = force_factor * program_state->bodies[i].mass;
            program_state->bodies[j].ax -= fj * dx;
            program_state->bodies[j].ay -= fj * dy;
            program_state->bodies[j].az -= fj * dz;
        }
    }
}

/* Update positions and velocities (leapfrog integration) */
static void integrate(__private ProgramState *program_state, double dt) {
    for (int i = 0; i < NUM_BODIES; i++) {
        /* Update velocity */
        program_state->bodies[i].vx += program_state->bodies[i].ax * dt;
        program_state->bodies[i].vy += program_state->bodies[i].ay * dt;
        program_state->bodies[i].vz += program_state->bodies[i].az * dt;

        /* Update position */
        program_state->bodies[i].x += program_state->bodies[i].vx * dt;
        program_state->bodies[i].y += program_state->bodies[i].vy * dt;
        program_state->bodies[i].z += program_state->bodies[i].vz * dt;
    }
}

/* Compute total kinetic energy */
static double kinetic_energy(__private ProgramState *program_state) {
    double ke = 0.0;
    for (int i = 0; i < NUM_BODIES; i++) {
        double v_sq = program_state->bodies[i].vx * program_state->bodies[i].vx
                    + program_state->bodies[i].vy * program_state->bodies[i].vy
                    + program_state->bodies[i].vz * program_state->bodies[i].vz;
        ke += 0.5 * program_state->bodies[i].mass * v_sq;
    }
    return ke;
}

int program_main(__private ProgramState *program_state) {
    double total_energy = 0.0;

    for (int run = 0; run < 10; run++) {
        init_bodies(program_state);

        for (int step = 0; step < TIME_STEPS; step++) {
            compute_forces(program_state);
            integrate(program_state, DT);

            /* Sample energy periodically */
            if (step % 10 == 0) {
                total_energy += kinetic_energy(program_state);
            }
        }
    }

    program_state->sink = total_energy;
    return 0;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ProgramState storage = {0};
    results[0] = (uint)program_main(&storage);
    for (uint i = 0; i < sizeof(storage.sink); ++i)
        results[1 + i] = (uint)((as_ulong(storage.sink) >> (8u * i)) & 255u);
}
