// Portable bitfield layout of programs/int_bitfields.c using explicit masks.
static long signed_field(uint field, uint width) {
    uint sign_bit = 1u << (width - 1);
    return (int)((field ^ sign_bit) - sign_bit);
}

static ulong pack_fields(uint seed) {
    // A zero-width field aligns d to the next 32-bit word.
    uint first = seed & 511u;
    uint second = ((seed * 3u) & 0x1ffffu) | (((seed & 0x7ffu) - 512u) & 0x7ffu) << 17;
    ulong f = (((ulong)seed << 13) ^ 0x1ffffffffUL) & 0x1ffffffffUL;
    ulong g = (f >> 1) & 0x7fffffffUL;
    // f starts at bit 28 in the second allocation unit and g follows f.
    volatile ulong units[3] = {first, second | (f << 28) | (g << 61), g >> 3};
    return ((units[0] & 511UL) | ((units[1] & 0x1ffffUL) << 9)) ^
           ((units[1] >> 28) & 0x1ffffffffUL);
}

static long sign_fields(int seed) {
    uint useed = (uint)seed;
    uint s1 = ((useed & 7u) - 4u) & 15u;
    uint s2 = (((useed >> 3) & 63u) - 32u) & 127u;
    uint u1 = (useed >> 1) & 31u;
    uint u2 = (useed * 5u) & 511u;
    uint s3 = (uint)(((long)seed * 17L & 0xfffffL) - 0x80000L) & 0xfffffu;
    uint u3 = (useed >> 2) & 4095u;
    volatile uint packed[2] = {s1 | (s2 << 4) | (u1 << 11) | (u2 << 16), s3 | (u3 << 20)};
    return signed_field(packed[0] & 15u, 4) + signed_field((packed[0] >> 4) & 127u, 7)
         + ((packed[0] >> 11) & 31u) + ((packed[0] >> 16) & 511u)
         + signed_field(packed[1] & 0xfffffu, 20) - (packed[1] >> 20);
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void program_entry(__global uint *results) {
    ulong sink = 0;
    for (uint i = 0; i < 50000; ++i) {
        sink += pack_fields(i * 11u + 7u);
        sink += (ulong)sign_fields((int)i - 25000);
    }
    results[0] = 0;
    for (uint byte = 0; byte < 8; ++byte)
        results[1 + byte] = (uint)((sink >> (byte * 8)) & 255UL);
}
