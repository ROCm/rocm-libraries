// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// K1: implicit-GEMM convolution, forward and backward data, channels-last (NDHWC x, K T R S C/g w,
// NDHWK y).
// HIPRTC-compatible: no standard headers.
//
// Per group g the GEMM is y[m][n] = sum_k A[m][k] * B[n][k] with
//   m = (image, output position), n = output channel within the group,
//   k = (tap, input channel within the group), the tap outer, so one K-step of BK channels
//   stays within one tap and every row's tap offset is one add.
// A is never formed: each block loads its A tile from x, bounds-checking each row's tap.
//
// Atoms: one 16x16 matrix-core step per arch family (MFMA wave64, WMMA gfx11 / gfx12 wave32),
// or FMA micro-tiles for archs or dtypes without one. The order of k within an atom is free
// so long as A and B agree, so every atom reads a contiguous run of k from LDS.
#pragma once

#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__)
#define K1_HAS_MFMA 1
#else
#define K1_HAS_MFMA 0
#endif
#if defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__)
#define K1_HAS_MFMA_BF16_1K 1
#else
#define K1_HAS_MFMA_BF16_1K 0
#endif
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) \
    || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
#define K1_HAS_WMMA11 1
#else
#define K1_HAS_WMMA11 0
#endif
#if defined(__gfx1200__) || defined(__gfx1201__)
#define K1_HAS_WMMA12 1
#else
#define K1_HAS_WMMA12 0
#endif

typedef _Float16 k1_half;
typedef unsigned short k1_bf16; // bf16 bits

#include "ConvIgemmArgs.h"

// ---------------------------------------------------------------------------------------------
// Element conversion

__device__ inline float k1BfToFloat(unsigned short b)
{
    return __builtin_bit_cast(float, static_cast<unsigned>(b) << 16);
}
__device__ inline unsigned short k1FloatToBf(float v)
{
    unsigned u = __builtin_bit_cast(unsigned, v);
    if((u & 0x7f800000u) == 0x7f800000u)
    {
        return static_cast<unsigned short>((u >> 16) | ((u & 0xffffu) != 0 ? 0x40u : 0u));
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<unsigned short>(u >> 16);
}

template <class T>
struct K1Elem;
template <>
struct K1Elem<float>
{
    __device__ static float load(float v)
    {
        return v;
    }
    __device__ static float store(float v)
    {
        return v;
    }
};
template <>
struct K1Elem<k1_half>
{
    __device__ static float load(k1_half v)
    {
        return static_cast<float>(v);
    }
    __device__ static k1_half store(float v)
    {
        return static_cast<k1_half>(v);
    }
};
template <>
struct K1Elem<k1_bf16>
{
    __device__ static float load(k1_bf16 v)
    {
        return k1BfToFloat(v);
    }
    __device__ static k1_bf16 store(float v)
    {
        return k1FloatToBf(v);
    }
};

template <bool C, class A, class B>
struct K1If
{
    typedef A type;
};
template <class A, class B>
struct K1If<false, A, B>
{
    typedef B type;
};

template <int BYTES>
struct K1Raw;
template <>
struct K1Raw<16>
{
    typedef unsigned type __attribute__((ext_vector_type(4)));
};
template <>
struct K1Raw<8>
{
    typedef unsigned type __attribute__((ext_vector_type(2)));
};
template <>
struct K1Raw<4>
{
    typedef unsigned type;
};
template <>
struct K1Raw<2>
{
    typedef unsigned short type;
};

// Buffer loads: an offset past the buffer's end returns zeros, so padding and tails need no
// branch, and offsets are 32-bit from a per-block base. Off (K1_BUFFER 0) when one image's
// rows could span 2^31 bytes; the global loads then take 64-bit addresses.
#ifndef K1_BUFFER
#define K1_BUFFER 1
#endif
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) \
    || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)                      \
    || defined(__gfx1200__) || defined(__gfx1201__)
#define K1_RSRC_FLAGS 0x31004000
#elif defined(__gfx1010__) || defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__)
#define K1_RSRC_FLAGS 0x31014000
#else
#define K1_RSRC_FLAGS 0x00020000
#endif
#define K1_OUT_OF_RANGE 0x80000000u

__device__ inline __amdgpu_buffer_rsrc_t k1Rsrc(const void* base, long long bytes)
{
    const int records = bytes < 0x7fffffffLL ? static_cast<int>(bytes) : 0x7fffffff;
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<void*>(base), static_cast<short>(0), records, K1_RSRC_FLAGS);
}

template <class Raw>
__device__ inline Raw k1BufferLoad(__amdgpu_buffer_rsrc_t r, unsigned offset)
{
    if constexpr(sizeof(Raw) == 16)
        return __builtin_amdgcn_raw_buffer_load_b128(r, offset, 0, 0);
    else if constexpr(sizeof(Raw) == 8)
        return __builtin_amdgcn_raw_buffer_load_b64(r, offset, 0, 0);
    else if constexpr(sizeof(Raw) == 4)
        return __builtin_amdgcn_raw_buffer_load_b32(r, offset, 0, 0);
    else
        return __builtin_amdgcn_raw_buffer_load_b16(r, offset, 0, 0);
}

// ---------------------------------------------------------------------------------------------
// Atoms. One step multiplies a 16 x KA slice of A by a KA x 16 slice of B. A lane holds KPL
// consecutive k of row (or column) lane % 16, starting at kBase(lane); its accumulator element i
// is D[dRow(lane, i)][lane % 16].

template <int N>
using k1f = float __attribute__((ext_vector_type(N)));
template <int N>
using k1h = _Float16 __attribute__((ext_vector_type(N)));
template <int N>
using k1s = short __attribute__((ext_vector_type(N)));

// MFMA 16x16 (wave64): the four 16-lane groups split k and hold row groups of 4 of D.
struct K1Mfma
{
    static constexpr int WAVE = 64;
    static constexpr int DIM = 16;
    static constexpr int ACC = 4;
    typedef k1f<4> acc;
    __device__ static int dRow(int lane, int i)
    {
        return (lane >> 4) * 4 + i;
    }
    template <int KPL>
    __device__ static int kBase(int lane)
    {
        return (lane >> 4) * KPL;
    }
};

struct K1AtomMfmaF16 : K1Mfma // v_mfma_f32_16x16x16_f16: gfx908+
{
    static constexpr int KA = 16, KPL = 4;
    typedef _Float16 elem;
    typedef k1h<4> frag;
    __device__ static acc mma(frag a, frag b, acc c)
    {
#if K1_HAS_MFMA
        return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
#else
        __builtin_trap();
        return c;
#endif
    }
};

struct K1AtomMfmaBf16 : K1Mfma // v_mfma_f32_16x16x16_bf16 (1k): gfx90a+
{
    static constexpr int KA = 16, KPL = 4;
    typedef short elem;
    typedef k1s<4> frag;
    __device__ static acc mma(frag a, frag b, acc c)
    {
#if K1_HAS_MFMA_BF16_1K
        return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
#else
        __builtin_trap();
        return c;
#endif
    }
};

struct K1AtomMfmaBf16Gfx908 : K1Mfma // v_mfma_f32_16x16x8_bf16: gfx908 (half rate)
{
    static constexpr int KA = 8, KPL = 2;
    typedef short elem;
    typedef k1s<2> frag;
    __device__ static acc mma(frag a, frag b, acc c)
    {
#if defined(__gfx908__) || defined(__gfx90a__)
        return __builtin_amdgcn_mfma_f32_16x16x8bf16(a, b, c, 0, 0, 0);
#else
        __builtin_trap();
        return c;
#endif
    }
};

struct K1AtomMfmaF32 : K1Mfma // v_mfma_f32_16x16x4_f32: gfx908+
{
    static constexpr int KA = 4, KPL = 1;
    typedef float elem;
    typedef float frag;
    __device__ static acc mma(frag a, frag b, acc c)
    {
#if K1_HAS_MFMA
        return __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0);
#else
        __builtin_trap();
        return c;
#endif
    }
};

struct K1AtomMfmaXf32 : K1Mfma // v_mfma_f32_16x16x8_xf32: gfx942 only
{
    static constexpr int KA = 8, KPL = 2;
    typedef float elem;
    typedef k1f<2> frag;
    __device__ static acc mma(frag a, frag b, acc c)
    {
#if defined(__gfx942__)
        return __builtin_amdgcn_mfma_f32_16x16x8_xf32(a, b, c, 0, 0, 0);
#else
        __builtin_trap();
        return c;
#endif
    }
};

struct K1AtomMfmaF16X32 : K1Mfma // v_mfma_f32_16x16x32_f16: gfx950
{
    static constexpr int KA = 32, KPL = 8;
    typedef _Float16 elem;
    typedef k1h<8> frag;
    __device__ static acc mma(frag a, frag b, acc c)
    {
#if defined(__gfx950__)
        return __builtin_amdgcn_mfma_f32_16x16x32_f16(a, b, c, 0, 0, 0);
#else
        __builtin_trap();
        return c;
#endif
    }
};

struct K1AtomMfmaBf16X32 : K1Mfma // v_mfma_f32_16x16x32_bf16: gfx950
{
    static constexpr int KA = 32, KPL = 8;
    typedef short elem;
    typedef k1s<8> frag;
    __device__ static acc mma(frag a, frag b, acc c)
    {
#if defined(__gfx950__)
        typedef __bf16 bf8 __attribute__((ext_vector_type(8)));
        return __builtin_amdgcn_mfma_f32_16x16x32_bf16(
            __builtin_bit_cast(bf8, a), __builtin_bit_cast(bf8, b), c, 0, 0, 0);
#else
        __builtin_trap();
        return c;
#endif
    }
};

// MFMA 32x32 (wave64): the two 32-lane halves split k; a lane holds 16 rows of one column of D,
// in groups of 4 rows 8 apart. Half the LDS reads per flop of the 16x16 atoms.
struct K1Mfma32
{
    static constexpr int WAVE = 64;
    static constexpr int DIM = 32;
    static constexpr int ACC = 16;
    typedef k1f<16> acc;
    __device__ static int dRow(int lane, int i)
    {
        return (i >> 2) * 8 + (lane >> 5) * 4 + (i & 3);
    }
    template <int KPL>
    __device__ static int kBase(int lane)
    {
        return (lane >> 5) * KPL;
    }
};

#define K1_MFMA32_ATOM(NAME, KA_, KPL_, ELEM, FRAG, GUARD, CALL) \
    struct NAME : K1Mfma32                                       \
    {                                                            \
        static constexpr int KA = KA_, KPL = KPL_;               \
        typedef ELEM elem;                                       \
        typedef FRAG frag;                                       \
        __device__ static acc mma(frag a, frag b, acc c)         \
        {                                                        \
            if constexpr(GUARD)                                  \
            {                                                    \
                return CALL;                                     \
            }                                                    \
            else                                                 \
            {                                                    \
                __builtin_trap();                                \
                return c;                                        \
            }                                                    \
        }                                                        \
    };

#if K1_HAS_MFMA
#define K1_CALL_32F16 __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, 0)
#define K1_CALL_32F32 __builtin_amdgcn_mfma_f32_32x32x2f32(a, b, c, 0, 0, 0)
#else
#define K1_CALL_32F16 c
#define K1_CALL_32F32 c
#endif
#if K1_HAS_MFMA_BF16_1K
#define K1_CALL_32BF16 __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a, b, c, 0, 0, 0)
#else
#define K1_CALL_32BF16 c
#endif
#if defined(__gfx908__) || defined(__gfx90a__)
#define K1_CALL_32BF16_908 __builtin_amdgcn_mfma_f32_32x32x4bf16(a, b, c, 0, 0, 0)
#define K1_HAS_MFMA_BF16_908 1
#else
#define K1_CALL_32BF16_908 c
#define K1_HAS_MFMA_BF16_908 0
#endif
#if defined(__gfx942__)
#define K1_CALL_32XF32 __builtin_amdgcn_mfma_f32_32x32x4_xf32(a, b, c, 0, 0, 0)
#define K1_HAS_XF32 1
#else
#define K1_CALL_32XF32 c
#define K1_HAS_XF32 0
#endif
#if defined(__gfx950__)
#define K1_HAS_MFMA950 1
#define K1_CALL_32F16X16 __builtin_amdgcn_mfma_f32_32x32x16_f16(a, b, c, 0, 0, 0)
#define K1_CALL_32BF16X16                    \
    __builtin_amdgcn_mfma_f32_32x32x16_bf16( \
        __builtin_bit_cast(k1bf8, a), __builtin_bit_cast(k1bf8, b), c, 0, 0, 0)
typedef __bf16 k1bf8 __attribute__((ext_vector_type(8)));
#else
#define K1_HAS_MFMA950 0
#define K1_CALL_32F16X16 c
#define K1_CALL_32BF16X16 c
#endif

K1_MFMA32_ATOM(K1AtomMfma32F16, 8, 4, _Float16, k1h<4>, K1_HAS_MFMA, K1_CALL_32F16)
K1_MFMA32_ATOM(K1AtomMfma32Bf16, 8, 4, short, k1s<4>, K1_HAS_MFMA_BF16_1K, K1_CALL_32BF16)
K1_MFMA32_ATOM(
    K1AtomMfma32Bf16Gfx908, 4, 2, short, k1s<2>, K1_HAS_MFMA_BF16_908, K1_CALL_32BF16_908)
K1_MFMA32_ATOM(K1AtomMfma32F32, 2, 1, float, float, K1_HAS_MFMA, K1_CALL_32F32)
K1_MFMA32_ATOM(K1AtomMfma32Xf32, 4, 2, float, k1f<2>, K1_HAS_XF32, K1_CALL_32XF32)
K1_MFMA32_ATOM(K1AtomMfma32F16X16, 16, 8, _Float16, k1h<8>, K1_HAS_MFMA950, K1_CALL_32F16X16)
K1_MFMA32_ATOM(K1AtomMfma32Bf16X16, 16, 8, short, k1s<8>, K1_HAS_MFMA950, K1_CALL_32BF16X16)

// gfx11 WMMA (wave32): both half-waves hold all 16 k of the same rows; D rows interleave.
struct K1Wmma11
{
    static constexpr int WAVE = 32;
    static constexpr int DIM = 16;
    static constexpr int ACC = 8;
    static constexpr int KA = 16, KPL = 16;
    typedef k1f<8> acc;
    __device__ static int dRow(int lane, int i)
    {
        return 2 * i + (lane >> 4);
    }
    template <int>
    __device__ static int kBase(int)
    {
        return 0;
    }
};

struct K1AtomWmma11F16 : K1Wmma11
{
    typedef _Float16 elem;
    typedef k1h<16> frag;
    __device__ static acc mma(frag a, frag b, acc c)
    {
#if K1_HAS_WMMA11
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
        __builtin_trap();
        return c;
#endif
    }
};

struct K1AtomWmma11Bf16 : K1Wmma11
{
    typedef short elem;
    typedef k1s<16> frag;
    __device__ static acc mma(frag a, frag b, acc c)
    {
#if K1_HAS_WMMA11
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#else
        __builtin_trap();
        return c;
#endif
    }
};

// gfx12 WMMA (wave32): the half-waves split k; D rows in blocks of 8.
struct K1Wmma12
{
    static constexpr int WAVE = 32;
    static constexpr int DIM = 16;
    static constexpr int ACC = 8;
    static constexpr int KA = 16, KPL = 8;
    typedef k1f<8> acc;
    __device__ static int dRow(int lane, int i)
    {
        return (lane >> 4) * 8 + i;
    }
    template <int P>
    __device__ static int kBase(int lane)
    {
        return (lane >> 4) * P;
    }
};

struct K1AtomWmma12F16 : K1Wmma12
{
    typedef _Float16 elem;
    typedef k1h<8> frag;
    __device__ static acc mma(frag a, frag b, acc c)
    {
#if K1_HAS_WMMA12
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
        __builtin_trap();
        return c;
#endif
    }
};

struct K1AtomWmma12Bf16 : K1Wmma12
{
    typedef short elem;
    typedef k1s<8> frag;
    __device__ static acc mma(frag a, frag b, acc c)
    {
#if K1_HAS_WMMA12
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
#else
        __builtin_trap();
        return c;
#endif
    }
};

// MATH_NATIVE: the tile type is the atom's own. MATH_BF16X3: fp32 tiles, each operand split into
// bf16 hi + lo, three products (lo*hi, hi*lo, hi*hi) accumulated in fp32.
enum K1Math
{
    K1_MATH_NATIVE = 0,
    K1_MATH_BF16X3 = 1
};

template <class Atom, class T>
__device__ inline typename Atom::frag k1Frag(const T* p)
{
    typename Atom::frag f;
    if constexpr(Atom::KPL == 1)
    {
        f = __builtin_bit_cast(typename Atom::elem, p[0]);
    }
    else
    {
#pragma unroll
        for(int i = 0; i < Atom::KPL; ++i)
        {
            f[i] = __builtin_bit_cast(typename Atom::elem, p[i]);
        }
    }
    return f;
}

// ---------------------------------------------------------------------------------------------
// Problem geometry per block.
//
// Forward, per group: rows m = (image, output position), columns n = output channels, k = (tap,
// input channel). Backward data, per group and stride phase: rows m = (image, input position in
// the phase), columns n = input channels, k = (tap valid in the phase, output channel). Within a
// phase the input positions step by the stride and their output positions by one, so the dy
// gather is the forward gather with stride 1, an offset and a negative tap step: one gather
// serves both.

struct K1Gather
{
    int chans; // channels per pixel of the gathered tensor
    long long chanOffset; // the group's first channel
    int cgroup; // reduction channels (per group)
    int in[3]; // gathered tensor's spatial dims
    int out[3]; // row positions per image
    int stride[3], pad[3], step[3], taps[3];
    long long rows; // images * positions
};

// A filter tap of the gather, as the filter element it reads: element = first + q * step.
struct K1TapMap
{
    int first[3], step[3];
    int fr, fs; // filter extents (for the linear tap index)
};

enum K1Direction
{
    K1_FORWARD = 0,
    K1_BACKWARD_DATA = 1
};

__device__ inline int k1Gcd(int a, int b)
{
    while(b != 0)
    {
        const int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

// Phase phi of one axis: input positions i with (i + pad) % stride == phi, the filter taps that
// reach them, and the gather parameters over dy. False when the phase has no positions.
__device__ inline void k1PhaseAxis(int phi,
                                   int size,
                                   int outSize,
                                   int filter,
                                   int stride,
                                   int pad,
                                   int dil,
                                   int& first, // first input position
                                   int& count, // positions
                                   int& tapFirst,
                                   int& tapStep,
                                   int& taps,
                                   int& gatherPad,
                                   int& gatherStep)
{
    first = ((phi - pad) % stride + stride) % stride;
    count = first < size ? (size - 1 - first) / stride + 1 : 0;
    tapStep = stride / k1Gcd(stride, dil);
    tapFirst = -1;
    for(int r = 0; r < filter && r < tapStep; ++r)
    {
        if(((r * dil - phi) % stride + stride) % stride == 0)
        {
            tapFirst = r;
            break;
        }
    }
    taps = tapFirst < 0 ? 0 : (filter - 1 - tapFirst) / tapStep + 1;
    if(tapFirst < 0)
    {
        tapFirst = 0;
    }
    // Output position of input first + stride * j at tap q: (first + pad - (tapFirst + q *
    // tapStep) * dil) / stride = j + o0 - q * (tapStep * dil / stride).
    const int o0 = (first + pad - tapFirst * dil) / stride;
    gatherPad = -o0;
    gatherStep = -(tapStep * dil / stride);
    (void)outSize;
}

template <int DIR>
__device__ inline void
    k1Setup(const K1Args& a, int z, K1Gather& gather, K1TapMap& map, int& reduceGroup, int& phase)
{
    const int cg = a.c / a.groups, kg = a.k / a.groups;
    map.fr = a.fr;
    map.fs = a.fs;
    if constexpr(DIR == K1_FORWARD)
    {
        const int g = z;
        gather.chans = a.c;
        gather.chanOffset = static_cast<long long>(g) * cg;
        gather.cgroup = cg;
        gather.in[0] = a.di, gather.in[1] = a.hi, gather.in[2] = a.wi;
        gather.out[0] = a.dout, gather.out[1] = a.ho, gather.out[2] = a.wo;
        gather.stride[0] = a.sd, gather.stride[1] = a.sh, gather.stride[2] = a.sw;
        gather.pad[0] = a.pd, gather.pad[1] = a.ph, gather.pad[2] = a.pw;
        gather.step[0] = a.dd, gather.step[1] = a.dh, gather.step[2] = a.dw;
        gather.taps[0] = a.ft, gather.taps[1] = a.fr, gather.taps[2] = a.fs;
        gather.rows = a.m;
        for(int i = 0; i < 3; ++i)
        {
            map.first[i] = 0;
            map.step[i] = 1;
        }
        reduceGroup = g;
        phase = 0;
    }
    else
    {
        const int phases = a.sd * a.sh * a.sw;
        const int g = z / phases;
        phase = z - g * phases;
        const int phiW = phase % a.sw;
        const int phiH = (phase / a.sw) % a.sh;
        const int phiD = phase / (a.sw * a.sh);
        int first[3], count[3];
        k1PhaseAxis(phiD,
                    a.di,
                    a.dout,
                    a.ft,
                    a.sd,
                    a.pd,
                    a.dd,
                    first[0],
                    count[0],
                    map.first[0],
                    map.step[0],
                    gather.taps[0],
                    gather.pad[0],
                    gather.step[0]);
        k1PhaseAxis(phiH,
                    a.hi,
                    a.ho,
                    a.fr,
                    a.sh,
                    a.ph,
                    a.dh,
                    first[1],
                    count[1],
                    map.first[1],
                    map.step[1],
                    gather.taps[1],
                    gather.pad[1],
                    gather.step[1]);
        k1PhaseAxis(phiW,
                    a.wi,
                    a.wo,
                    a.fs,
                    a.sw,
                    a.pw,
                    a.dw,
                    first[2],
                    count[2],
                    map.first[2],
                    map.step[2],
                    gather.taps[2],
                    gather.pad[2],
                    gather.step[2]);
        gather.chans = a.k;
        gather.chanOffset = static_cast<long long>(g) * kg;
        gather.cgroup = kg;
        gather.in[0] = a.dout, gather.in[1] = a.ho, gather.in[2] = a.wo;
        for(int i = 0; i < 3; ++i)
        {
            gather.out[i] = count[i];
            gather.stride[i] = 1;
        }
        gather.rows = static_cast<long long>(a.n) * count[0] * count[1] * count[2];
        reduceGroup = g;
    }
}

// Where row m, column n of the GEMM lands: forward y[m][g kg + n]; backward data dx at the
// phase's input position.
template <int DIR>
__device__ inline long long
    k1OutIndex(const K1Args& a, const K1Gather& gather, int g, int phase, long long m, int n)
{
    if constexpr(DIR == K1_FORWARD)
    {
        return m * a.k + static_cast<long long>(g) * (a.k / a.groups) + n;
    }
    else
    {
        const int pw = phase % a.sw;
        const int ph = (phase / a.sw) % a.sh;
        const int pd = phase / (a.sw * a.sh);
        const int spatial = gather.out[0] * gather.out[1] * gather.out[2];
        const long long img = m / spatial;
        int rest = static_cast<int>(m - img * spatial);
        const int jw = rest % gather.out[2];
        rest /= gather.out[2];
        const int jh = rest % gather.out[1];
        const int jd = rest / gather.out[1];
        const int iw = ((pw - a.pw) % a.sw + a.sw) % a.sw + jw * a.sw;
        const int ih = ((ph - a.ph) % a.sh + a.sh) % a.sh + jh * a.sh;
        const int id = ((pd - a.pd) % a.sd + a.sd) % a.sd + jd * a.sd;
        return (((img * a.di + id) * a.hi + ih) * a.wi + iw) * a.c
               + static_cast<long long>(g) * (a.c / a.groups) + n;
    }
}

// ---------------------------------------------------------------------------------------------
// Global tile loads. A: VEC reduction channels of PASSES gathered rows, each row's origin decoded
// once. B: VEC reduction channels of PASSES filter rows, contiguous along the reduction:
//   forward w[g kg + n][tap][c] (K T R S C/g, as given);
//   backward data wT[g][tap][c = n][k] (from k1TransposeWeights).
// Out-of-range rows, taps and channels load zeros.

template <class T, int BM, int BN, int BK, int VEC, int THREADS, int DIR>
struct K1Tiles
{
    typedef typename K1Raw<VEC * sizeof(T)>::type Raw;
    static constexpr int ROW_THREADS = BK / VEC;
    static constexpr int ROWS_PER_PASS = THREADS / ROW_THREADS;
    // Rows over the passes; a tile shorter than one pass leaves threads idle.
    static constexpr int A_PASSES = (BM + ROWS_PER_PASS - 1) / ROWS_PER_PASS;
    static constexpr int B_PASSES = (BN + ROWS_PER_PASS - 1) / ROWS_PER_PASS;
    static_assert(THREADS % ROW_THREADS == 0 && (BM % ROWS_PER_PASS == 0 || BM < ROWS_PER_PASS)
                      && (BN % ROWS_PER_PASS == 0 || BN < ROWS_PER_PASS),
                  "tile loads");
    __device__ bool aRow(int p) const
    {
        return row + p * ROWS_PER_PASS < BM;
    }
    __device__ bool bRow(int p) const
    {
        return row + p * ROWS_PER_PASS < BN;
    }

    const T* src;
    const T* w;
    K1Gather gather;
    K1TapMap map;
    int cChunks, kv, row, nTaps;
    long long tapStride;
    long long aBase[A_PASSES];
    int aD[A_PASSES], aH[A_PASSES], aW[A_PASSES];
    long long bBase[B_PASSES];
    bool bOk[B_PASSES];
    Raw ra[A_PASSES], rb[B_PASSES];
#if K1_BUFFER
    __amdgpu_buffer_rsrc_t aRsrc, bRsrc;
    int aOff[A_PASSES], bOff[B_PASSES]; // bytes from the buffers' bases
#endif

    __device__ void init(const K1Args& a,
                         const K1Gather& gth,
                         const K1TapMap& mp,
                         int g,
                         long long m0,
                         int n0,
                         int tid)
    {
        gather = gth;
        map = mp;
        src = static_cast<const T*>(DIR == K1_FORWARD ? a.x : a.y);
        w = static_cast<const T*>(a.w);
        const int cg = a.c / a.groups, kg = a.k / a.groups;
        cChunks = (gather.cgroup + BK - 1) / BK;
        nTaps = gather.taps[0] * gather.taps[1] * gather.taps[2];
        const long long filterTaps = static_cast<long long>(a.ft) * a.fr * a.fs;
        kv = (tid % ROW_THREADS) * VEC;
        row = tid / ROW_THREADS;
        const int spatial = gather.out[0] * gather.out[1] * gather.out[2];
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const long long m = m0 + row + p * ROWS_PER_PASS;
            if(m < gather.rows && aRow(p))
            {
                const int img = static_cast<int>(m / spatial);
                int rest = static_cast<int>(m - static_cast<long long>(img) * spatial);
                const int ow = rest % gather.out[2];
                rest /= gather.out[2];
                const int oh = rest % gather.out[1];
                const int od = rest / gather.out[1];
                aD[p] = od * gather.stride[0] - gather.pad[0];
                aH[p] = oh * gather.stride[1] - gather.pad[1];
                aW[p] = ow * gather.stride[2] - gather.pad[2];
                aBase[p]
                    = (((static_cast<long long>(img) * gather.in[0] + aD[p]) * gather.in[1] + aH[p])
                           * gather.in[2]
                       + aW[p])
                          * gather.chans
                      + gather.chanOffset;
            }
            else
            {
                aD[p] = -0x40000000;
                aH[p] = aW[p] = 0;
                aBase[p] = 0;
            }
        }
#if K1_BUFFER
        // A buffer from the image of the block's first row: the rows span at most a few images.
        const long long imageElems
            = static_cast<long long>(gather.in[0]) * gather.in[1] * gather.in[2] * gather.chans;
        const long long firstImage = m0 / spatial;
        const long long blockBase = firstImage * imageElems;
        aRsrc = k1Rsrc(src + blockBase,
                       (a.n - firstImage) * imageElems * static_cast<long long>(sizeof(T)));
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
            aOff[p] = static_cast<int>((aBase[p] - blockBase) * static_cast<long long>(sizeof(T)));
#endif
        // Filter rows: the GEMM's columns (forward output channels, backward data input channels).
        const int ncols = DIR == K1_FORWARD ? kg : cg;
        tapStride = DIR == K1_FORWARD ? cg : static_cast<long long>(cg) * kg;
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            const int nn = n0 + row + p * ROWS_PER_PASS;
            bOk[p] = nn < ncols && bRow(p);
            const int n = bOk[p] ? nn : 0;
            bBase[p] = DIR == K1_FORWARD ? (static_cast<long long>(g) * kg + n) * filterTaps * cg
                                         : static_cast<long long>(g) * filterTaps * cg * kg
                                               + static_cast<long long>(n) * kg;
        }
#if K1_BUFFER
        bRsrc = k1Rsrc(
            w, static_cast<long long>(a.k) * filterTaps * cg * static_cast<long long>(sizeof(T)));
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
            bOff[p] = static_cast<int>(bBase[p] * static_cast<long long>(sizeof(T)));
#endif
    }

    __device__ int iterations() const
    {
        return nTaps * cChunks;
    }

    __device__ void load(const K1Args& a, int it)
    {
        (void)a;
        const int q = it / cChunks;
        const int c0 = (it - q * cChunks) * BK;
        const int qs = q % gather.taps[2];
        const int qr = (q / gather.taps[2]) % gather.taps[1];
        const int qt = q / (gather.taps[2] * gather.taps[1]);
        const int td = qt * gather.step[0], th = qr * gather.step[1], tw = qs * gather.step[2];
        const long long delta
            = ((static_cast<long long>(td) * gather.in[1] + th) * gather.in[2] + tw) * gather.chans;
        const int c = c0 + kv;
        const bool cOk = c < gather.cgroup;
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const int id = aD[p] + td, ih = aH[p] + th, iw = aW[p] + tw;
            const bool ok = cOk && static_cast<unsigned>(id) < static_cast<unsigned>(gather.in[0])
                            && static_cast<unsigned>(ih) < static_cast<unsigned>(gather.in[1])
                            && static_cast<unsigned>(iw) < static_cast<unsigned>(gather.in[2]);
#if K1_BUFFER
            const unsigned offset
                = ok ? static_cast<unsigned>(
                           aOff[p]
                           + static_cast<int>((delta + c) * static_cast<long long>(sizeof(T))))
                     : K1_OUT_OF_RANGE;
            ra[p] = k1BufferLoad<Raw>(aRsrc, offset);
#else
            ra[p] = Raw{};
            if(ok)
            {
                ra[p] = *reinterpret_cast<const Raw*>(src + aBase[p] + delta + c);
            }
#endif
        }
        // The filter element this tap reads.
        const long long tap = (static_cast<long long>(map.first[0] + qt * map.step[0]) * map.fr
                               + map.first[1] + qr * map.step[1])
                                  * map.fs
                              + map.first[2] + qs * map.step[2];
        const long long off = tap * tapStride + c;
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
#if K1_BUFFER
            const unsigned offset
                = cOk && bOk[p]
                      ? static_cast<unsigned>(
                            bOff[p] + static_cast<int>(off * static_cast<long long>(sizeof(T))))
                      : K1_OUT_OF_RANGE;
            rb[p] = k1BufferLoad<Raw>(bRsrc, offset);
#else
            rb[p] = Raw{};
            if(cOk && bOk[p])
            {
                rb[p] = *reinterpret_cast<const Raw*>(w + bBase[p] + off);
            }
#endif
        }
    }

    // Row-major LDS tiles: As[m][k], Bs[n][k], ld elements per row.
    __device__ void storeRows(T* As, T* Bs, int ld) const
    {
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
            if(aRow(p))
                *reinterpret_cast<Raw*>(As + (row + p * ROWS_PER_PASS) * ld + kv) = ra[p];
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
            if(bRow(p))
                *reinterpret_cast<Raw*>(Bs + (row + p * ROWS_PER_PASS) * ld + kv) = rb[p];
    }

    // Three-bf16: fp32 tiles stored row-major as bf16 hi planes (As, Bs) and lo planes (As +
    // aPlane, Bs + bPlane), split once here rather than per fragment read.
    __device__ static void storeSplit(const Raw& r, unsigned short* d, int plane)
    {
        typedef unsigned short Half __attribute__((ext_vector_type(VEC)));
        const float* v = reinterpret_cast<const float*>(&r);
        Half h, l;
#pragma unroll
        for(int e = 0; e < VEC; ++e)
        {
            const unsigned short he = k1FloatToBf(v[e]);
            h[e] = he;
            l[e] = k1FloatToBf(v[e] - k1BfToFloat(he));
        }
        *reinterpret_cast<Half*>(d) = h;
        *reinterpret_cast<Half*>(d + plane) = l;
    }
    __device__ void
        storeRowsSplit(unsigned short* As, unsigned short* Bs, int ld, int aPlane, int bPlane) const
    {
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
            if(aRow(p))
                storeSplit(ra[p], As + (row + p * ROWS_PER_PASS) * ld + kv, aPlane);
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
            if(bRow(p))
                storeSplit(rb[p], Bs + (row + p * ROWS_PER_PASS) * ld + kv, bPlane);
    }

    // k-major LDS tiles: As[k][m] (ldA per k), Bs[k][n] (ldB per k).
    __device__ void storeColumns(T* As, T* Bs, int ldA, int ldB) const
    {
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            if(!aRow(p))
                continue;
            const T* v = reinterpret_cast<const T*>(&ra[p]);
#pragma unroll
            for(int e = 0; e < VEC; ++e)
                As[(kv + e) * ldA + row + p * ROWS_PER_PASS] = v[e];
        }
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            if(!bRow(p))
                continue;
            const T* v = reinterpret_cast<const T*>(&rb[p]);
#pragma unroll
            for(int e = 0; e < VEC; ++e)
                Bs[(kv + e) * ldB + row + p * ROWS_PER_PASS] = v[e];
        }
    }
};

// Backward data reads the filter as wT[g][tap][c][k]: per group and tap, w's [kg][cg] slice
// transposed through 32 x 32 LDS tiles. Grid (ceil(cg / 32), ceil(kg / 32), groups * taps),
// 256 threads.
template <class T>
__device__ void k1TransposeWeights(const K1Args& a, T* wT)
{
    __shared__ T tile[32][33];
    const int cg = a.c / a.groups, kg = a.k / a.groups;
    const long long taps = static_cast<long long>(a.ft) * a.fr * a.fs;
    const int g = static_cast<int>(blockIdx.z / taps);
    const long long tap = blockIdx.z - static_cast<long long>(g) * taps;
    const T* w = static_cast<const T*>(a.w);
    const int c0 = blockIdx.x * 32, k0 = blockIdx.y * 32;
    const int tx = threadIdx.x % 32, ty = threadIdx.x / 32;
    for(int r = ty; r < 32; r += 8)
    {
        const int k = k0 + r, c = c0 + tx;
        if(k < kg && c < cg)
            tile[r][tx] = w[((static_cast<long long>(g) * kg + k) * taps + tap) * cg + c];
    }
    __syncthreads();
    for(int r = ty; r < 32; r += 8)
    {
        const int c = c0 + r, k = k0 + tx;
        if(k < kg && c < cg)
            wT[((static_cast<long long>(g) * taps + tap) * cg + c) * kg + k] = tile[tx][r];
    }
}

// ---------------------------------------------------------------------------------------------
// Matrix-core kernel. Block: WM x WN waves, tile BM x BN, k-step BK, STAGES LDS buffers (2: the
// next tile is stored while the current one is multiplied, one barrier per k-step). Grid: (m
// tiles, n tiles, groups [x phases for backward data]).

template <class T,
          class Atom,
          int MATH,
          int BM,
          int BN,
          int BK,
          int WM,
          int WN,
          int VEC,
          int DIR,
          int STAGES = 1>
__device__ void k1ConvMma(const K1Args& a)
{
    constexpr int WAVE = Atom::WAVE;
    constexpr int THREADS = WM * WN * WAVE;
    constexpr bool SPLIT = MATH == K1_MATH_BF16X3;
    // Tiles of the multiply's element type: T, or bf16 planes (hi, lo) for three-bf16.
    typedef typename K1If<SPLIT, unsigned short, T>::type S;
    constexpr int LD = BK + 16 / static_cast<int>(sizeof(S));
    constexpr int WTM = BM / WM, WTN = BN / WN;
    constexpr int DIM = Atom::DIM;
    constexpr int FM = WTM / DIM, FN = WTN / DIM;
    constexpr int PLANES = SPLIT ? 2 : 1;
    constexpr int STAGE_S = PLANES * (BM + BN) * LD; // in S
    constexpr int STAGE = (STAGE_S * static_cast<int>(sizeof(S)) + static_cast<int>(sizeof(T)) - 1)
                          / static_cast<int>(sizeof(T)); // in T
    // Epilogue: each wave stages 16 rows of its tile at a time.
    constexpr int OLD = WTN + 16 / static_cast<int>(sizeof(T));
    constexpr int OUT = WM * WN * DIM * OLD;
    constexpr int LDS = STAGES * STAGE > OUT ? STAGES * STAGE : OUT;
    static_assert(WTM % DIM == 0 && WTN % DIM == 0 && BK % Atom::KA == 0, "tile");
    // STAGES 2 raced on gfx11 while backward data stored its B tile with transposed 2-byte LDS
    // writes (a barrier hid it, a fence did not); with vector row stores only it passes.
    static_assert(STAGES == 1 || STAGES == 2, "stages");

    __shared__ __attribute__((aligned(16))) T lds[LDS];

    K1Gather gather;
    K1TapMap map;
    int g, phase;
    k1Setup<DIR>(a, blockIdx.z, gather, map, g, phase);
    const long long m0 = static_cast<long long>(blockIdx.x) * BM;
    if(m0 >= gather.rows)
    {
        return;
    }
    const int ncols = DIR == K1_FORWARD ? a.k / a.groups : a.c / a.groups;
    const int n0 = blockIdx.y * BN;
    const int tid = threadIdx.x;
    const int lane = tid % WAVE;
    const int wave = tid / WAVE;
    const int wm = wave % WM, wn = wave / WM;

    K1Tiles<T, BM, BN, BK, VEC, THREADS, DIR> tiles;
    tiles.init(a, gather, map, g, m0, n0, tid);

    typename Atom::acc acc[FM][FN];
#pragma unroll
    for(int i = 0; i < FM; ++i)
#pragma unroll
        for(int j = 0; j < FN; ++j)
            acc[i][j] = typename Atom::acc{};

    const int fragRow = lane % DIM;
    const int kb = Atom::template kBase<Atom::KPL>(lane);
    const int iters = tiles.iterations();

    // A planes at As (hi) and As + BM * LD (lo); B planes likewise with BN.
    auto multiply = [&](const S* As, const S* Bs) {
#pragma unroll
        for(int kk = 0; kk < BK; kk += Atom::KA)
        {
            if constexpr(SPLIT)
            {
                typename Atom::frag ah[FM], al[FM], bh[FN], bl[FN];
#pragma unroll
                for(int i = 0; i < FM; ++i)
                {
                    const S* p = As + (wm * WTM + i * DIM + fragRow) * LD + kk + kb;
                    ah[i] = k1Frag<Atom>(p);
                    al[i] = k1Frag<Atom>(p + BM * LD);
                }
#pragma unroll
                for(int j = 0; j < FN; ++j)
                {
                    const S* p = Bs + (wn * WTN + j * DIM + fragRow) * LD + kk + kb;
                    bh[j] = k1Frag<Atom>(p);
                    bl[j] = k1Frag<Atom>(p + BN * LD);
                }
#pragma unroll
                for(int i = 0; i < FM; ++i)
#pragma unroll
                    for(int j = 0; j < FN; ++j)
                    {
                        acc[i][j] = Atom::mma(al[i], bh[j], acc[i][j]);
                        acc[i][j] = Atom::mma(ah[i], bl[j], acc[i][j]);
                        acc[i][j] = Atom::mma(ah[i], bh[j], acc[i][j]);
                    }
            }
            else
            {
                typename Atom::frag af[FM], bf[FN];
#pragma unroll
                for(int i = 0; i < FM; ++i)
                    af[i] = k1Frag<Atom>(As + (wm * WTM + i * DIM + fragRow) * LD + kk + kb);
#pragma unroll
                for(int j = 0; j < FN; ++j)
                    bf[j] = k1Frag<Atom>(Bs + (wn * WTN + j * DIM + fragRow) * LD + kk + kb);
#pragma unroll
                for(int i = 0; i < FM; ++i)
#pragma unroll
                    for(int j = 0; j < FN; ++j)
                        acc[i][j] = Atom::mma(af[i], bf[j], acc[i][j]);
            }
        }
    };

    if(iters > 0)
    {
        tiles.load(a, 0);
    }
    S* tile = reinterpret_cast<S*>(lds);
    // A then B; each PLANES planes.
    auto store = [&](S* base) {
        if constexpr(SPLIT)
        {
            tiles.storeRowsSplit(base, base + 2 * BM * LD, LD, BM * LD, BN * LD);
        }
        else
        {
            tiles.storeRows(base, base + BM * LD, LD);
        }
    };
    if constexpr(STAGES == 2)
    {
        if(iters > 0)
        {
            store(tile);
        }
        for(int it = 0; it < iters; ++it)
        {
            __syncthreads();
            if(it + 1 < iters)
            {
                tiles.load(a, it + 1);
            }
            const S* cur = tile + (it & 1) * STAGE_S;
            multiply(cur, cur + PLANES * BM * LD);
            if(it + 1 < iters)
            {
                store(tile + ((it + 1) & 1) * STAGE_S);
            }
        }
    }
    else
    {
        for(int it = 0; it < iters; ++it)
        {
            __syncthreads();
            store(tile);
            __syncthreads();
            if(it + 1 < iters)
            {
                tiles.load(a, it + 1);
            }
            multiply(tile, tile + PLANES * BM * LD);
        }
    }

    // Epilogue: per DIM-row slab, accumulators to the wave's LDS strip, then rows of VEC columns
    // to global (single columns past the last full vector).
    __syncthreads();
    T* strip = lds + wave * DIM * OLD;
    T* out = static_cast<T*>(DIR == K1_FORWARD ? a.y : a.x);
    typedef typename K1Raw<VEC * sizeof(T)>::type Raw;
    constexpr int ROW_VECS = WTN / VEC;
#pragma unroll
    for(int i = 0; i < FM; ++i)
    {
#pragma unroll
        for(int j = 0; j < FN; ++j)
#pragma unroll
            for(int e = 0; e < Atom::ACC; ++e)
                strip[Atom::dRow(lane, e) * OLD + j * DIM + lane % DIM]
                    = K1Elem<T>::store(acc[i][j][e]);
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        for(int v = lane; v < DIM * ROW_VECS; v += WAVE)
        {
            const int r = v / ROW_VECS;
            const int col = (v - r * ROW_VECS) * VEC;
            const long long m = m0 + wm * WTM + i * DIM + r;
            const int nn = n0 + wn * WTN + col;
            if(m >= gather.rows || nn >= ncols)
            {
                continue;
            }
            const long long o = k1OutIndex<DIR>(a, gather, g, phase, m, nn);
            if(nn + VEC <= ncols && o % VEC == 0)
            {
                *reinterpret_cast<Raw*>(out + o)
                    = *reinterpret_cast<const Raw*>(strip + r * OLD + col);
            }
            else
            {
                for(int e = 0; e < VEC && nn + e < ncols; ++e)
                    out[o + e] = strip[r * OLD + col + e];
            }
        }
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }
}

// ---------------------------------------------------------------------------------------------
// FMA kernel: every arch and dtype (fp32 on RDNA, everything on gfx906). k-major LDS tiles; each
// thread computes TM x TN outputs in runs of 4 rows and 4 columns.

template <class T, int BM, int BN, int BK, int THREADS, int VEC, int DIR>
__device__ void k1ConvFma(const K1Args& a)
{
    constexpr int TX = 16, TY = THREADS / TX; // threads along n, along m
    constexpr int TM = BM / TY, TN = BN / TX;
    static_assert(TM % 4 == 0 && TN % 4 == 0, "4-runs");
    constexpr int LDA = BM + 16 / static_cast<int>(sizeof(T)),
                  LDB = BN + 16 / static_cast<int>(sizeof(T));

    __shared__ __attribute__((aligned(16))) T lds[BK * (LDA + LDB)];
    T* As = lds;
    T* Bs = lds + BK * LDA;

    K1Gather gather;
    K1TapMap map;
    int g, phase;
    k1Setup<DIR>(a, blockIdx.z, gather, map, g, phase);
    const long long m0 = static_cast<long long>(blockIdx.x) * BM;
    if(m0 >= gather.rows)
    {
        return;
    }
    const int ncols = DIR == K1_FORWARD ? a.k / a.groups : a.c / a.groups;
    const int n0 = blockIdx.y * BN;
    const int tid = threadIdx.x;
    const int tx = tid % TX, ty = tid / TX;

    K1Tiles<T, BM, BN, BK, VEC, THREADS, DIR> tiles;
    tiles.init(a, gather, map, g, m0, n0, tid);

    float acc[TM][TN];
#pragma unroll
    for(int i = 0; i < TM; ++i)
#pragma unroll
        for(int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    const int iters = tiles.iterations();
    if(iters > 0)
    {
        tiles.load(a, 0);
    }
    for(int it = 0; it < iters; ++it)
    {
        __syncthreads();
        tiles.storeColumns(As, Bs, LDA, LDB);
        __syncthreads();
        if(it + 1 < iters)
        {
            tiles.load(a, it + 1);
        }
#pragma unroll 2
        for(int kk = 0; kk < BK; ++kk)
        {
            float av[TM], bv[TN];
#pragma unroll
            for(int q = 0; q < TM / 4; ++q)
#pragma unroll
                for(int e = 0; e < 4; ++e)
                    av[q * 4 + e] = K1Elem<T>::load(As[kk * LDA + (q * TY + ty) * 4 + e]);
#pragma unroll
            for(int q = 0; q < TN / 4; ++q)
#pragma unroll
                for(int e = 0; e < 4; ++e)
                    bv[q * 4 + e] = K1Elem<T>::load(Bs[kk * LDB + (q * TX + tx) * 4 + e]);
#pragma unroll
            for(int i = 0; i < TM; ++i)
#pragma unroll
                for(int j = 0; j < TN; ++j)
                    acc[i][j] = __builtin_fmaf(av[i], bv[j], acc[i][j]);
        }
    }

    T* out = static_cast<T*>(DIR == K1_FORWARD ? a.y : a.x);
#pragma unroll
    for(int i = 0; i < TM; ++i)
    {
        const long long m = m0 + ((i / 4) * TY + ty) * 4 + i % 4;
#pragma unroll
        for(int j = 0; j < TN; ++j)
        {
            const int nn = n0 + ((j / 4) * TX + tx) * 4 + j % 4;
            if(m < gather.rows && nn < ncols)
            {
                out[k1OutIndex<DIR>(a, gather, g, phase, m, nn)] = K1Elem<T>::store(acc[i][j]);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Strided copy: stages an operand to or from the layout the convolution reads. One element per
// thread, grid-strided; 64-bit indices.
__device__ inline void k1StridedCopy(const ConvIgemmCopyArgs& a)
{
    const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
    for(long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < a.total;
        i += step)
    {
        long long rest = i, src = 0, dst = 0;
        for(int d = a.rank - 1; d >= 0; --d)
        {
            const long long coord = rest % a.dims[d];
            rest /= a.dims[d];
            src += coord * a.srcStrides[d];
            dst += coord * a.dstStrides[d];
        }
        if(a.elementBytes == 4)
            static_cast<unsigned*>(a.dst)[dst] = static_cast<const unsigned*>(a.src)[src];
        else
            static_cast<unsigned short*>(a.dst)[dst]
                = static_cast<const unsigned short*>(a.src)[src];
    }
}

// Batched 2-D transpose through a 32 x 33 LDS tile: coalesced reads and writes. A 1-D grid of
// (column tile, row tile, batch), column tiles fastest; 256 threads.
template <class E>
__device__ inline void k1TransposeTile(const ConvIgemmTransposeArgs& a,
                                       long long tileCol,
                                       long long tileRow,
                                       long long b)
{
    __shared__ E tile[32][33];
    const E* src = static_cast<const E*>(a.src) + b * a.rows * a.cols;
    E* dst = static_cast<E*>(a.dst) + b * a.rows * a.cols;
    const int tx = threadIdx.x % 32, ty = threadIdx.x / 32;
    const long long r0 = tileRow * 32, c0 = tileCol * 32;
    for(int r = ty; r < 32; r += 8)
    {
        if(r0 + r < a.rows && c0 + tx < a.cols)
            tile[r][tx] = src[(r0 + r) * a.cols + c0 + tx];
    }
    __syncthreads();
    for(int c = ty; c < 32; c += 8)
    {
        if(c0 + c < a.cols && r0 + tx < a.rows)
            dst[(c0 + c) * a.rows + r0 + tx] = tile[tx][c];
    }
}

__device__ inline void k1Transpose(const ConvIgemmTransposeArgs& a)
{
    const long long colTiles = (a.cols + 31) / 32;
    const long long rowTiles = (a.rows + 31) / 32;
    const long long block = blockIdx.x;
    const long long tileCol = block % colTiles;
    const long long tileRow = (block / colTiles) % rowTiles;
    const long long b = block / (colTiles * rowTiles);
    if(a.elementBytes == 4)
        k1TransposeTile<unsigned>(a, tileCol, tileRow, b);
    else
        k1TransposeTile<unsigned short>(a, tileCol, tileRow, b);
}
