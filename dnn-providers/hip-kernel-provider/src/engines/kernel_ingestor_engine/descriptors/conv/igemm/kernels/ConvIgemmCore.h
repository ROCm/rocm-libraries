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
// RDNA families, not chip lists, so a gfx11-generic / gfx12-generic build keeps its WMMA atoms.
#include "ConvArchFamily.h"
#include "ConvPackedMath.h"
#define K1_HAS_WMMA11 HKP_ARCH_GFX11
#define K1_HAS_WMMA12 HKP_ARCH_GFX12
#define K1_HAS_WMMA125 HKP_ARCH_GFX1250

typedef _Float16 k1_half;
typedef unsigned short k1_bf16; // bf16 bits

#include "ConvEpilogue.h"
#include "ConvIgemmArgs.h"

// ---------------------------------------------------------------------------------------------
// Element conversion

__device__ inline float k1BfToFloat(unsigned short b)
{
    return __builtin_bit_cast(float, static_cast<unsigned>(b) << 16);
}
__device__ inline unsigned short k1FloatToBf(float v)
{
    // Nearest-even; a NaN stays a (quiet) NaN, an infinity infinite. Both results and a select,
    // not a branch: per-lane values, and a branch costs an exec-masked block per element.
    const unsigned u = __builtin_bit_cast(unsigned, v);
    const unsigned special = (u >> 16) | ((u & 0xffffu) != 0 ? 0x40u : 0u);
    const unsigned rounded = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
    return static_cast<unsigned short>((u & 0x7f800000u) == 0x7f800000u ? special : rounded);
}
/// Three-bf16's low part of v, given its high part h = k1FloatToBf(v): zero where v - h is not a
/// number (v infinite), so an infinite operand stays infinite rather than turning into NaN.
__device__ inline unsigned short k1BfLo(float v, unsigned short h)
{
    const float r = v - k1BfToFloat(h);
    return k1FloatToBf(r == r ? r : 0.0f);
}
/// Three-bf16's high part for the cross terms (hi x the other operand's lo): zero for an infinity.
/// The hi x hi term alone carries an infinity, so inf x b is inf (NaN only where b is 0), as in
/// fp32; a cross term would add inf x lo, NaN when lo is 0 or of the other sign.
__device__ inline unsigned short k1BfFinite(unsigned short h)
{
    return (h & 0x7fffu) == 0x7f80u ? static_cast<unsigned short>(0) : h;
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
#if defined(__gfx950__)
        // The hardware conversion (v_cvt_pk_bf16_f32: nearest-even; a NaN stays a NaN, its
        // payload aside), what k1FloatToBf computes in software on the other targets.
        return __builtin_bit_cast(k1_bf16, static_cast<__bf16>(v));
#else
        return k1FloatToBf(v);
#endif
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
// Positions within one image: 32-bit, or 64-bit in the large-image configs (K1_LARGE 1, admitted
// only for images past 2^31 - 1 pixels; the generic tier's 64-bit variant).
#ifndef K1_LARGE
#define K1_LARGE 0
#endif
// Patch backward data (K1_PATCH 1): the output index of a patch convolution run as one GEMM
// (K1Args::patchT, patchR); forward patch configs run the kernel unchanged on a reshaped problem.
#ifndef K1_PATCH
#define K1_PATCH 0
#endif
#if K1_LARGE
typedef long long K1Pos;
#else
typedef int K1Pos;
#endif
#if HKP_ARCH_GFX1250
#define K1_RSRC_FLAGS 0
#elif HKP_ARCH_GFX11 || HKP_ARCH_GFX12
#define K1_RSRC_FLAGS 0x31004000
#elif defined(__gfx1010__) || defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__)
#define K1_RSRC_FLAGS 0x31014000
#else
#define K1_RSRC_FLAGS 0x00020000
#endif
#define K1_OUT_OF_RANGE 0x80000000u
// Rows of shift range the halo A tile is sized for (the launcher checks the problem fits).
#ifndef K1_HALO_CAP
#define K1_HALO_CAP 64
#endif

// Every caller passes wave-uniform values; readfirstlane says so. A base derived through a
// 64-bit division otherwise stays in VGPRs and the compiler wraps every buffer access in a
// waterfall loop (TheRock 10.3 HIPRTC; K1, K7 and stem-bwd). Already-scalar values fold away.
__device__ inline __amdgpu_buffer_rsrc_t k1Rsrc(const void* base, long long bytes)
{
    const int records = bytes < 0x7fffffffLL ? static_cast<int>(bytes) : 0x7fffffff;
    const unsigned long long p = reinterpret_cast<unsigned long long>(base);
    const unsigned lo = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(p));
    const unsigned hi = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(p >> 32));
    return __builtin_amdgcn_make_buffer_rsrc(
        reinterpret_cast<void*>((static_cast<unsigned long long>(hi) << 32) | lo),
        static_cast<short>(0),
        __builtin_amdgcn_readfirstlane(records),
        K1_RSRC_FLAGS);
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

/// xf32 operands rounded to nearest-even TF32 (10 mantissa bits) before the MFMA, which itself
/// would truncate (a bias toward zero); NaN stays NaN (quiet), inf stays inf. As hipBLASLt (#9176)
/// and NVIDIA's cvt.rna.tf32.f32 do. The xf32 atoms take operands already rounded: the multiply
/// rounds each fragment once where it loads it (K1Xf32), not at every MFMA that reuses it.
__device__ inline float k1Tf32(float v)
{
    unsigned u = __builtin_bit_cast(unsigned, v);
    if((u & 0x7f800000u) == 0x7f800000u)
    {
        u |= (u & 0x7fffffu) != 0 ? 0x400000u : 0u;
    }
    else
    {
        u += 0xfffu + ((u >> 13) & 1u);
    }
    return __builtin_bit_cast(float, u & 0xffffe000u);
}
// Element by element into a new vector: a loop writing a vector's elements miscompiles
// (every element takes element 0's result).
__device__ inline k1f<2> k1Tf32(k1f<2> v)
{
    return k1f<2>{k1Tf32(v[0]), k1Tf32(v[1])};
}

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

/// Whether @p Atom is an xf32 atom (its fragments rounded to TF32 once loaded).
template <class Atom>
struct K1Xf32
{
    static constexpr bool value = false;
};
template <>
struct K1Xf32<K1AtomMfmaXf32>
{
    static constexpr bool value = true;
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
template <>
struct K1Xf32<K1AtomMfma32Xf32>
{
    static constexpr bool value = true;
};

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

// gfx1250's wave32 instructions keep the accumulator mapping of gfx1200/1201, but
// each half-wave supplies 16 reduction elements for f16/bf16 and two for f32.
struct K1Wmma125 : K1Wmma12
{
    static constexpr int KA = 32, KPL = 16;
};

struct K1AtomWmma125F16 : K1Wmma125
{
    typedef _Float16 elem;
    typedef k1h<16> frag;
    __device__ static acc mma(frag a, frag b, acc c)
    {
#if K1_HAS_WMMA125
        return __builtin_amdgcn_wmma_f32_16x16x32_f16(false, a, false, b, 0, c, false, false);
#else
        __builtin_trap();
        return c;
#endif
    }
};

struct K1AtomWmma125Bf16 : K1Wmma125
{
    typedef short elem;
    typedef k1s<16> frag;
    __device__ static acc mma(frag a, frag b, acc c)
    {
#if K1_HAS_WMMA125
        typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
        return __builtin_amdgcn_wmma_f32_16x16x32_bf16(false,
                                                       __builtin_bit_cast(bf16x16, a),
                                                       false,
                                                       __builtin_bit_cast(bf16x16, b),
                                                       0,
                                                       c,
                                                       false,
                                                       false);
#else
        __builtin_trap();
        return c;
#endif
    }
};

struct K1AtomWmma125F32 : K1Wmma125
{
    static constexpr int KA = 4, KPL = 2;
    typedef float elem;
    typedef k1f<2> frag;
    __device__ static acc mma(frag a, frag b, acc c)
    {
#if K1_HAS_WMMA125
        return __builtin_amdgcn_wmma_f32_16x16x4_f32(false, a, false, b, 0, c, false, false);
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

/// @p f with each bf16 infinity zeroed (k1BfFinite): three-bf16's cross-term high parts.
template <class Atom>
__device__ inline typename Atom::frag k1FiniteFrag(typename Atom::frag f)
{
    if constexpr(Atom::KPL == 1)
    {
        return f;
    }
    else
    {
        // On whole dwords (two bf16 each): a per-element 16-bit select miscompiles on gfx12
        // (fragments rebuilt from the wrong halves).
        constexpr int N = Atom::KPL / 2;
        typedef unsigned W __attribute__((ext_vector_type(N)));
        W w = __builtin_bit_cast(W, f);
#pragma unroll
        for(int i = 0; i < N; ++i)
        {
            const unsigned v = w[i];
            const unsigned low = (v & 0x7fffu) == 0x7f80u ? 0x0000ffffu : 0u;
            const unsigned high = (v & 0x7fff0000u) == 0x7f800000u ? 0xffff0000u : 0u;
            w[i] = v & ~(low | high);
        }
        return __builtin_bit_cast(typename Atom::frag, w);
    }
}

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
#if K1_PATCH == 3
        // Col2im: the GEMM over dy's positions, its T R S C columns stored row by row.
        (void)gather;
        (void)g;
        (void)phase;
        return m * a.c + n;
#elif K1_PATCH
        // Patch convolution as one GEMM (one group, unit stride over dy's positions): column n
        // is the filter element ((t R + r) S + s) C + c, and dx is (D T) x (H R) x (W S) x C, so
        // a row of S C columns is one contiguous run of dx.
        (void)g;
        (void)phase;
        const int run = a.c / (a.patchT * a.patchR); // S C
        const int tr = n / run;
        const int j = n - tr * run;
        const int t = tr / a.patchR;
        const int r = tr - t * a.patchR;
        const K1Pos spatial = static_cast<K1Pos>(gather.out[0]) * gather.out[1] * gather.out[2];
        const long long img = m / spatial;
        K1Pos rest = static_cast<K1Pos>(m - img * spatial);
        const int jw = static_cast<int>(rest % gather.out[2]);
        rest /= gather.out[2];
        const int jh = static_cast<int>(rest % gather.out[1]);
        const int jd = static_cast<int>(rest / gather.out[1]);
        const long long id = static_cast<long long>(jd) * a.patchT + t;
        const long long ih = static_cast<long long>(jh) * a.patchR + r;
        return (((img * a.di * a.patchT + id) * a.hi * a.patchR + ih) * a.wi + jw) * run + j;
#else
        const int pw = phase % a.sw;
        const int ph = (phase / a.sw) % a.sh;
        const int pd = phase / (a.sw * a.sh);
        const K1Pos spatial = static_cast<K1Pos>(gather.out[0]) * gather.out[1] * gather.out[2];
        const long long img = m / spatial;
        K1Pos rest = static_cast<K1Pos>(m - img * spatial);
        const int jw = static_cast<int>(rest % gather.out[2]);
        rest /= gather.out[2];
        const int jh = static_cast<int>(rest % gather.out[1]);
        const int jd = static_cast<int>(rest / gather.out[1]);
        const int iw = ((pw - a.pw) % a.sw + a.sw) % a.sw + jw * a.sw;
        const int ih = ((ph - a.ph) % a.sh + a.sh) % a.sh + jh * a.sh;
        const int id = ((pd - a.pd) % a.sd + a.sd) % a.sd + jd * a.sd;
        return (((img * a.di + id) * a.hi + ih) * a.wi + iw) * a.c
               + static_cast<long long>(g) * (a.c / a.groups) + n;
#endif
    }
}

// Walks the channels-first output positions of consecutive GEMM rows (one column): the row's
// position along each axis (count[] per image), the output step per row along it, and the
// channel planes between images. Starts by division once, then steps by additions.
struct K1CfWalk
{
    int count[3]; // positions per image along d, h, w
    long long step[3]; // output elements per position along d, h, w
    long long image; // output elements from a position to the same one in the next image
    int j[3];
    long long o;

    template <int DIR>
    __device__ void init(const K1Args& a, const K1Gather& gather)
    {
        for(int i = 0; i < 3; ++i)
            count[i] = gather.out[i];
        if constexpr(DIR == K1_FORWARD)
        {
            step[2] = 1;
            step[1] = a.wo;
            step[0] = static_cast<long long>(a.ho) * a.wo;
            image = static_cast<long long>(a.k) * a.dout * step[0];
        }
        else
        {
            step[2] = a.sw;
            step[1] = static_cast<long long>(a.sh) * a.wi;
            step[0] = static_cast<long long>(a.sd) * a.hi * a.wi;
            image = static_cast<long long>(a.c) * a.di * a.hi * a.wi;
        }
    }

    // Row m's element (column index @p o0 from k1OutIndexCf) and its axis positions.
    __device__ void start(long long m, long long o0)
    {
        const K1Pos spatial = static_cast<K1Pos>(count[0]) * count[1] * count[2];
        K1Pos rest = static_cast<K1Pos>(m % spatial);
        j[2] = static_cast<int>(rest % count[2]);
        rest /= count[2];
        j[1] = static_cast<int>(rest % count[1]);
        j[0] = static_cast<int>(rest / count[1]);
        o = o0;
    }

    // The next row's element.
    __device__ void next()
    {
        o += step[2];
        if(++j[2] < count[2])
            return;
        j[2] = 0;
        o += step[1] - count[2] * step[2];
        if(++j[1] < count[1])
            return;
        j[1] = 0;
        o += step[0] - count[1] * step[1];
        if(++j[0] < count[0])
            return;
        j[0] = 0;
        // Back to the image's first position, then on to the next image's plane.
        o += image - count[0] * step[0];
    }
};

// The same element in a packed channels-first output (CF 1): forward y[img][g kg + n][pos],
// backward data dx[img][g cg + n][the phase's input position].
template <int DIR>
__device__ inline long long
    k1OutIndexCf(const K1Args& a, const K1Gather& gather, int g, int phase, long long m, int n)
{
    if constexpr(DIR == K1_FORWARD)
    {
        const long long spatial = static_cast<long long>(a.dout) * a.ho * a.wo;
        const long long img = m / spatial;
        return (img * a.k + static_cast<long long>(g) * (a.k / a.groups) + n) * spatial
               + (m - img * spatial);
    }
    else
    {
        const int pw = phase % a.sw;
        const int ph = (phase / a.sw) % a.sh;
        const int pd = phase / (a.sw * a.sh);
        const K1Pos spatial = static_cast<K1Pos>(gather.out[0]) * gather.out[1] * gather.out[2];
        const long long img = m / spatial;
        K1Pos rest = static_cast<K1Pos>(m - img * spatial);
        const int jw = static_cast<int>(rest % gather.out[2]);
        rest /= gather.out[2];
        const int jh = static_cast<int>(rest % gather.out[1]);
        const int jd = static_cast<int>(rest / gather.out[1]);
        const int iw = ((pw - a.pw) % a.sw + a.sw) % a.sw + jw * a.sw;
        const int ih = ((ph - a.ph) % a.sh + a.sh) % a.sh + jh * a.sh;
        const int id = ((pd - a.pd) % a.sd + a.sd) % a.sd + jd * a.sd;
        return ((img * a.c + static_cast<long long>(g) * (a.c / a.groups) + n) * a.di + id) * a.hi
                   * a.wi
               + static_cast<long long>(ih) * a.wi + iw;
    }
}

// ---------------------------------------------------------------------------------------------
// XOR-swizzled row-major LDS tiles (SWZ 1): rows of BK 16-bit elements, no padding, each row's
// 16-byte chunks permuted by an XOR key that changes every 128 bytes of rows, so the eight lanes
// a 16-byte LDS access serves at once (eight consecutive rows, one chunk) hit 32 distinct banks.
// Element offset of (row, k): the chunk k / 8 moves to chunk (k / 8) ^ key(row).
template <int BK>
struct K1Swz
{
    static constexpr int CPR = BK / 8; // 16-byte chunks per row
    static constexpr int RPL = CPR >= 8 ? 1 : 8 / CPR; // rows per 128 bytes
    static_assert(CPR >= 2 && (CPR & (CPR - 1)) == 0 && CPR <= 8, "swizzled rows");
    __device__ static int key(int row)
    {
        return (row / RPL) & (CPR - 1);
    }
    // Offset of the chunk holding k (a multiple of 8) of a row with key @p key.
    __device__ static int at(int row, int key, int k)
    {
        return row * BK + (((k >> 3) ^ key) << 3);
    }
};

// ---------------------------------------------------------------------------------------------
// Global tile loads. A: VEC reduction channels of PASSES gathered rows, each row's origin decoded
// once. B: VEC reduction channels of PASSES filter rows, contiguous along the reduction:
//   forward w[g kg + n][tap][c] (K T R S C/g, as given);
//   backward data wT[g][tap][c = n][k] (from k1TransposeWeights).
// Out-of-range rows, taps and channels load zeros.
//
// WD 1 (backward data only): B straight from the graph's K T R S C/g filter, no transposed copy.
// Each thread loads KV consecutive output channels (the reduction) of one VEC run of input
// channels (the GEMM's columns) and stores them transposed into the row-major [n][k] tile: one
// 4-byte LDS store per column for 16-bit types (the KV = 2 pair packed), one element for fp32.
// The output-channel pairs vary fastest across lanes, so the stores hit consecutive banks.
//
// PACK 1: a k-step spans BK / cgroup taps when the reduction channels per tap (cgroup) divide BK,
// instead of padding each tap to BK. A thread's VEC channels sit in sub-tap kv / cgroup at
// channel kv % cgroup, so its tap cursor is its own.
// PACK 2 (flat): the reduction is the taps' channels end to end (taps x cgroup, cgroup a multiple
// of VEC), cut into k-steps of BK whatever cgroup is, so no tap pads to a whole k-step; each
// thread's cursor is its own (tap, channel), stepped BK channels on per k-step.

template <class T, int BM, int BN, int BK, int VEC, int THREADS, int DIR, int WD = 0, int PACK = 0>
struct K1Tiles
{
    typedef typename K1Raw<VEC * sizeof(T)>::type Raw;
    static constexpr int ROW_THREADS = BK / VEC;
    static constexpr int ROWS_PER_PASS = THREADS / ROW_THREADS;
    // Rows over the passes; a tile shorter than one pass leaves threads idle.
    static constexpr int A_PASSES = (BM + ROWS_PER_PASS - 1) / ROWS_PER_PASS;
    static constexpr bool DIRECT = DIR == K1_BACKWARD_DATA && WD != 0;
    // Direct filter: KV output channels by VEC input channels per unit, KPAIRS units per column run.
    static constexpr int KV = sizeof(T) == 2 ? 2 : 1;
    static constexpr int KPAIRS = BK / KV;
    static constexpr int DIRECT_UNITS = KPAIRS * (BN / VEC);
    static constexpr int B_PASSES = DIRECT ? (DIRECT_UNITS + THREADS - 1) / THREADS
                                           : (BN + ROWS_PER_PASS - 1) / ROWS_PER_PASS;
    static constexpr int B_LOADS = DIRECT ? B_PASSES * KV : B_PASSES;
    static_assert(THREADS % ROW_THREADS == 0 && (BM % ROWS_PER_PASS == 0 || BM < ROWS_PER_PASS)
                      && (DIRECT || BN % ROWS_PER_PASS == 0 || BN < ROWS_PER_PASS),
                  "tile loads");
    static_assert(!DIRECT || (BK % KV == 0 && BN % VEC == 0), "direct filter tile");
    static_assert(PACK == 0 || !DIRECT, "tap packing reads the transposed filter");
    // Every thread's row stays inside the tile when the passes divide it (row <
    // ROWS_PER_PASS), which the compiler cannot see from the thread index: say so, so no
    // branch guards a load or store.
    __device__ bool aRow(int p) const
    {
        if constexpr(BM % ROWS_PER_PASS == 0)
        {
            return true;
        }
        return row + p * ROWS_PER_PASS < BM;
    }
    __device__ bool bRow(int p) const
    {
        if constexpr(DIRECT)
        {
            if constexpr(DIRECT_UNITS % THREADS == 0)
            {
                return true;
            }
            return dUnit[p] < DIRECT_UNITS;
        }
        else
        {
            if constexpr(BN % ROWS_PER_PASS == 0)
            {
                return true;
            }
            return row + p * ROWS_PER_PASS < BN;
        }
    }

    const T* src;
    const T* w;
    K1Gather gather;
    K1TapMap map;
    int cChunks, kv, row, nTaps;
    // The next load's iteration and its decode (tap qt qr qs, channel chunk), advanced by
    // compare-and-wrap rather than four runtime divisions per load. Iterations at or past endIt
    // load zeros, so the next tile's prefetch needs no branch. PACK: the decode is this thread's
    // sub-tap's, packTaps taps on per iteration; packChan is its channel within the tap.
    int curIt, endIt, curC, curQs, curQr, curQt;
    int packTaps, packSub, packChan;
    // PACK 2: unsigned magic division by cgroup, taps[2] and taps[1] (uniform, set in init), so a
    // k-step's carry through the taps is a few multiplies, not a per-thread subtract loop.
    unsigned cgMul, cgSh, sMul, sSh, rMul, rSh;
    __device__ static void magicOf(int d, unsigned& mul, unsigned& sh)
    {
        unsigned s = 0;
        while((1u << s) < static_cast<unsigned>(d))
            ++s;
        sh = s;
        mul = static_cast<unsigned>(((static_cast<unsigned long long>((1ull << s) - d)) << 32)
                                        / static_cast<unsigned long long>(d)
                                    + 1);
    }
    /// n / d for 0 <= n < 2^31 (mul, sh from magicOf(d)).
    __device__ static int magicDiv(int n, unsigned mul, unsigned sh)
    {
        return static_cast<int>((__umulhi(static_cast<unsigned>(n), mul) + static_cast<unsigned>(n))
                                >> sh);
    }
    long long tapStride;
    long long aBase[A_PASSES];
    int aD[A_PASSES], aH[A_PASSES], aW[A_PASSES];
    long long bBase[B_PASSES];
    bool bOk[B_PASSES];
    // Direct filter: each pass's unit, its column offset and its output-channel offset in the tile.
    int dUnit[B_PASSES], dCol[B_PASSES], dK[B_PASSES];
    int kgroup; // output channels per group (direct filter bound)
    long long wRow; // filter elements per output channel (direct filter)
    Raw ra[A_PASSES], rb[B_LOADS];
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
        const K1Pos spatial = static_cast<K1Pos>(gather.out[0]) * gather.out[1] * gather.out[2];
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const long long m = m0 + row + p * ROWS_PER_PASS;
            if(m < gather.rows && aRow(p))
            {
                const int img = static_cast<int>(m / spatial);
                K1Pos rest = static_cast<K1Pos>(m - static_cast<long long>(img) * spatial);
                const int ow = static_cast<int>(rest % gather.out[2]);
                rest /= gather.out[2];
                const int oh = static_cast<int>(rest % gather.out[1]);
                const int od = static_cast<int>(rest / gather.out[1]);
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
        tapStride = DIR == K1_FORWARD || DIRECT ? cg
                    : a.wTapStride > 0          ? a.wTapStride
                                                : static_cast<long long>(cg) * kg;
        const long long rowStride = a.wRowStride > 0 ? a.wRowStride : kg;
        kgroup = kg;
        wRow = filterTaps * cg;
        if constexpr(DIRECT)
        {
            // Unit u: output channels (u % KPAIRS) * KV.., input channels (u / KPAIRS) * VEC..
#pragma unroll
            for(int p = 0; p < B_PASSES; ++p)
            {
                dUnit[p] = tid + p * THREADS;
                dK[p] = (dUnit[p] % KPAIRS) * KV;
                dCol[p] = (dUnit[p] / KPAIRS) * VEC;
                bOk[p] = n0 + dCol[p] < ncols && bRow(p);
                bBase[p]
                    = bOk[p] ? (static_cast<long long>(g) * kg + dK[p]) * wRow + n0 + dCol[p] : 0;
            }
        }
        else
        {
#pragma unroll
            for(int p = 0; p < B_PASSES; ++p)
            {
                const int nn = n0 + row + p * ROWS_PER_PASS;
                bOk[p] = nn < ncols && bRow(p);
                const int n = bOk[p] ? nn : 0;
                bBase[p] = DIR == K1_FORWARD
                               ? (static_cast<long long>(g) * kg + n) * filterTaps * cg
                               : static_cast<long long>(g) * filterTaps * cg * kg
                                     + static_cast<long long>(n) * rowStride;
            }
        }
        if constexpr(PACK == 1)
        {
            // Whole taps per k-step (the launcher checks cgroup divides BK).
            packTaps = BK / gather.cgroup;
            packSub = kv / gather.cgroup;
            packChan = kv - packSub * gather.cgroup;
            cChunks = 1;
        }
        else if constexpr(PACK == 2)
        {
            cChunks = 1;
            magicOf(gather.cgroup, cgMul, cgSh);
            magicOf(gather.taps[2], sMul, sSh);
            magicOf(gather.taps[1], rMul, rSh);
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
        if constexpr(PACK == 1)
        {
            return (nTaps + packTaps - 1) / packTaps;
        }
        else if constexpr(PACK == 2)
        {
            return static_cast<int>((static_cast<long long>(nTaps) * gather.cgroup + BK - 1) / BK);
        }
        return nTaps * cChunks;
    }

    /// Loads start at iteration @p first; those at or past @p end read zeros.
    __device__ void cursorInit(int first, int end)
    {
        if constexpr(PACK == 2)
        {
            // This thread's flat reduction index, as (tap, channel).
            const long long f = static_cast<long long>(first) * BK + kv;
            const int q = static_cast<int>(f / gather.cgroup);
            packChan = static_cast<int>(f - static_cast<long long>(q) * gather.cgroup);
            curIt = first;
            endIt = end;
            curC = 0;
            curQs = q % gather.taps[2];
            curQr = (q / gather.taps[2]) % gather.taps[1];
            curQt = q / (gather.taps[2] * gather.taps[1]);
            return;
        }
        const int q = PACK != 0 ? first * packTaps + packSub : first / cChunks;
        curIt = first;
        endIt = end;
        curC = PACK != 0 ? 0 : first - q * cChunks;
        curQs = q % gather.taps[2];
        curQr = (q / gather.taps[2]) % gather.taps[1];
        curQt = q / (gather.taps[2] * gather.taps[1]);
    }

    /// The cursor's iteration into the registers, and the cursor one iteration on.
    __device__ void load()
    {
        const int it = curIt;
        const int c0 = curC * BK;
        const int qs = curQs, qr = curQr, qt = curQt;
        const int chan = packChan;
        ++curIt;
        if constexpr(PACK == 2)
        {
            // BK channels on, carried through the taps (several when cgroup < BK).
            const int t = packChan + BK;
            const int adv = magicDiv(t, cgMul, cgSh);
            packChan = t - adv * gather.cgroup;
            const int s = curQs + adv;
            const int sCarry = magicDiv(s, sMul, sSh);
            curQs = s - sCarry * gather.taps[2];
            const int r = curQr + sCarry;
            const int rCarry = magicDiv(r, rMul, rSh);
            curQr = r - rCarry * gather.taps[1];
            curQt += rCarry;
        }
        else if constexpr(PACK != 0)
        {
            // packTaps taps on, carried through the tap extents.
            curQs += packTaps;
            while(curQs >= gather.taps[2])
            {
                curQs -= gather.taps[2];
                if(++curQr == gather.taps[1])
                {
                    curQr = 0;
                    ++curQt;
                }
            }
        }
        else if(++curC == cChunks)
        {
            curC = 0;
            if(++curQs == gather.taps[2])
            {
                curQs = 0;
                if(++curQr == gather.taps[1])
                {
                    curQr = 0;
                    ++curQt;
                }
            }
        }
        const int td = qt * gather.step[0], th = qr * gather.step[1], tw = qs * gather.step[2];
        const long long delta
            = ((static_cast<long long>(td) * gather.in[1] + th) * gather.in[2] + tw) * gather.chans;
        // PACK: this thread's channel in its own tap, valid while the tap is.
        const int c = PACK != 0 ? chan : c0 + kv;
        // Bitwise (not short-circuit) predicates: on per-lane values && lets the compiler turn
        // the offset select into exec-masked branches around the loads; & keeps a v_cndmask.
        const bool cOk = (PACK != 0 ? qt < gather.taps[0] : c < gather.cgroup) & (it < endIt);
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const int id = aD[p] + td, ih = aH[p] + th, iw = aW[p] + tw;
            const bool ok = cOk & (static_cast<unsigned>(id) < static_cast<unsigned>(gather.in[0]))
                            & (static_cast<unsigned>(ih) < static_cast<unsigned>(gather.in[1]))
                            & (static_cast<unsigned>(iw) < static_cast<unsigned>(gather.in[2]));
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
        loadBTap(qt, qr, qs, c0, c, cOk, it < endIt);
    }

#if K1_BUFFER && defined(__gfx950__)
    /// DMA (gfx950): the cursor's iteration straight into the swizzled LDS tiles (K1Swz) by
    /// buffer_load ... lds, 16 bytes a lane. A wave's 64 lanes fill 1 KB of consecutive rows, so
    /// the lane in slot kv / VEC of its row loads the global chunk slot ^ key(row) instead of the
    /// LDS address moving. Out-of-range rows, taps and channels land as zeros.
    __device__ void loadLds(T* As, T* Bs, int wave)
    {
        typedef K1Swz<BK> Swz;
        static_assert(VEC * sizeof(T) == 16 && !DIRECT && PACK != 1, "DMA tiles");
        static_assert(BM % ROWS_PER_PASS == 0 && BN % ROWS_PER_PASS == 0, "DMA: whole passes");
        constexpr int RPW = 64 / ROW_THREADS; // rows a wave fills per pass
        if constexpr(PACK == 2)
        {
            loadLdsFlat(As, Bs, wave);
            return;
        }
        const int it = curIt;
        const int c0 = curC * BK;
        const int qs = curQs, qr = curQr, qt = curQt;
        ++curIt;
        if(++curC == cChunks)
        {
            curC = 0;
            if(++curQs == gather.taps[2])
            {
                curQs = 0;
                if(++curQr == gather.taps[1])
                {
                    curQr = 0;
                    ++curQt;
                }
            }
        }
        const int td = qt * gather.step[0], th = qr * gather.step[1], tw = qs * gather.step[2];
        const long long delta
            = ((static_cast<long long>(td) * gather.in[1] + th) * gather.in[2] + tw) * gather.chans;
        const int slot = kv / VEC;
        const bool itOk = it < endIt;
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const int r = row + p * ROWS_PER_PASS;
            const int c = c0 + ((slot ^ Swz::key(r)) * VEC);
            const int id = aD[p] + td, ih = aH[p] + th, iw = aW[p] + tw;
            const bool ok = itOk & (c < gather.cgroup)
                            & (static_cast<unsigned>(id) < static_cast<unsigned>(gather.in[0]))
                            & (static_cast<unsigned>(ih) < static_cast<unsigned>(gather.in[1]))
                            & (static_cast<unsigned>(iw) < static_cast<unsigned>(gather.in[2]));
            const unsigned offset
                = ok ? static_cast<unsigned>(
                           aOff[p]
                           + static_cast<int>((delta + c) * static_cast<long long>(sizeof(T))))
                     : K1_OUT_OF_RANGE;
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                aRsrc,
                (__attribute__((address_space(3))) void*)(As
                                                          + (p * ROWS_PER_PASS + wave * RPW) * BK),
                16,
                offset,
                0,
                0,
                0);
        }
        const long long tap = (static_cast<long long>(map.first[0] + qt * map.step[0]) * map.fr
                               + map.first[1] + qr * map.step[1])
                                  * map.fs
                              + map.first[2] + qs * map.step[2];
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            const int r = row + p * ROWS_PER_PASS;
            const int c = c0 + ((slot ^ Swz::key(r)) * VEC);
            const bool ok = itOk & bOk[p] & (c < gather.cgroup);
            const unsigned offset
                = ok ? static_cast<unsigned>(
                           bOff[p]
                           + static_cast<int>((tap * tapStride + c)
                                              * static_cast<long long>(sizeof(T))))
                     : K1_OUT_OF_RANGE;
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                bRsrc,
                (__attribute__((address_space(3))) void*)(Bs
                                                          + (p * ROWS_PER_PASS + wave * RPW) * BK),
                16,
                offset,
                0,
                0,
                0);
        }
    }

    /// DMA with the flat reduction (PACK 2): a thread's 16-byte LDS slot holds reduction chunk
    /// kv / VEC of its rows, which the caller set to the swizzled chunk of its row (the swizzle key
    /// is the same for all of a thread's rows, ROWS_PER_PASS being a whole number of key periods),
    /// so one flat cursor serves every pass, as in load().
    __device__ void loadLdsFlat(T* As, T* Bs, int wave)
    {
        constexpr int RPW = 64 / ROW_THREADS;
        const int it = curIt;
        const int qs = curQs, qr = curQr, qt = curQt;
        const int chan = packChan;
        ++curIt;
        const int t = packChan + BK;
        const int adv = magicDiv(t, cgMul, cgSh);
        packChan = t - adv * gather.cgroup;
        const int sNext = curQs + adv;
        const int sCarry = magicDiv(sNext, sMul, sSh);
        curQs = sNext - sCarry * gather.taps[2];
        const int rNext = curQr + sCarry;
        const int rCarry = magicDiv(rNext, rMul, rSh);
        curQr = rNext - rCarry * gather.taps[1];
        curQt += rCarry;
        const int td = qt * gather.step[0], th = qr * gather.step[1], tw = qs * gather.step[2];
        const long long delta
            = ((static_cast<long long>(td) * gather.in[1] + th) * gather.in[2] + tw) * gather.chans;
        const bool cOk = (qt < gather.taps[0]) & (it < endIt);
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
        {
            const int id = aD[p] + td, ih = aH[p] + th, iw = aW[p] + tw;
            const bool ok = cOk & (static_cast<unsigned>(id) < static_cast<unsigned>(gather.in[0]))
                            & (static_cast<unsigned>(ih) < static_cast<unsigned>(gather.in[1]))
                            & (static_cast<unsigned>(iw) < static_cast<unsigned>(gather.in[2]));
            const unsigned offset
                = ok ? static_cast<unsigned>(
                           aOff[p]
                           + static_cast<int>((delta + chan) * static_cast<long long>(sizeof(T))))
                     : K1_OUT_OF_RANGE;
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                aRsrc,
                (__attribute__((address_space(3))) void*)(As
                                                          + (p * ROWS_PER_PASS + wave * RPW) * BK),
                16,
                offset,
                0,
                0,
                0);
        }
        const long long tap = (static_cast<long long>(map.first[0] + qt * map.step[0]) * map.fr
                               + map.first[1] + qr * map.step[1])
                                  * map.fs
                              + map.first[2] + qs * map.step[2];
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            const bool ok = cOk & bOk[p];
            const unsigned offset
                = ok ? static_cast<unsigned>(
                           bOff[p]
                           + static_cast<int>((tap * tapStride + chan)
                                              * static_cast<long long>(sizeof(T))))
                     : K1_OUT_OF_RANGE;
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                bRsrc,
                (__attribute__((address_space(3))) void*)(Bs
                                                          + (p * ROWS_PER_PASS + wave * RPW) * BK),
                16,
                offset,
                0,
                0,
                0);
        }
    }
#endif

    /// B for tap (qt, qr, qs): the filter rows at reduction channel @p c (this thread's, valid when
    /// @p cOk), or with the direct filter KV output channels of the chunk from @p c0 (while
    /// @p itOk).
    __device__ void loadBTap(int qt, int qr, int qs, int c0, int c, bool cOk, bool itOk)
    {
        const long long tap = (static_cast<long long>(map.first[0] + qt * map.step[0]) * map.fr
                               + map.first[1] + qr * map.step[1])
                                  * map.fs
                              + map.first[2] + qs * map.step[2];
        if constexpr(DIRECT)
        {
#pragma unroll
            for(int p = 0; p < B_PASSES; ++p)
            {
#pragma unroll
                for(int j = 0; j < KV; ++j)
                {
                    const bool ok = itOk & bOk[p] & (c0 + dK[p] + j < kgroup);
                    const long long off = (c0 + j) * wRow + tap * tapStride;
#if K1_BUFFER
                    const unsigned offset
                        = ok ? static_cast<unsigned>(
                                   bOff[p]
                                   + static_cast<int>(off * static_cast<long long>(sizeof(T))))
                             : K1_OUT_OF_RANGE;
                    rb[p * KV + j] = k1BufferLoad<Raw>(bRsrc, offset);
#else
                    rb[p * KV + j] = Raw{};
                    if(ok)
                    {
                        rb[p * KV + j] = *reinterpret_cast<const Raw*>(w + bBase[p] + off);
                    }
#endif
                }
            }
        }
        else
        {
            const long long off = tap * tapStride + c;
#pragma unroll
            for(int p = 0; p < B_PASSES; ++p)
            {
#if K1_BUFFER
                const unsigned offset
                    = (cOk & bOk[p])
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
    }

    /// B only, for tap (qt, qr, qs) and channel chunk start c0 (the halo loop; buffer loads).
    __device__ void loadB(int qt, int qr, int qs, int c0)
    {
        const int c = c0 + kv;
        loadBTap(qt, qr, qs, c0, c, c < gather.cgroup, true);
    }

    /// The B registers into Bs (row-major, ld per row); S unsigned short: three-bf16 planes,
    /// the lo plane @p plane elements on.
    template <class S>
    __device__ void storeB(S* Bs, int ld, int plane) const
    {
        if constexpr(DIRECT)
        {
            storeDirect(Bs, ld, plane);
            return;
        }
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            if(!bRow(p))
                continue;
            if constexpr(sizeof(S) != sizeof(T))
                storeSplit(rb[p], Bs + (row + p * ROWS_PER_PASS) * ld + kv, plane);
            else
                *reinterpret_cast<Raw*>(Bs + (row + p * ROWS_PER_PASS) * ld + kv) = rb[p];
        }
    }

    /// Direct filter: the KV x VEC unit transposed into the row-major Bs[n][k]. 16-bit: each
    /// column's output-channel pair as one 4-byte store; fp32 (S float) one element per column;
    /// three-bf16 (S unsigned short from fp32) the hi and lo planes.
    template <class S>
    __device__ void storeDirect(S* Bs, int ld, int plane) const
    {
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
        {
            if(!bRow(p))
                continue;
            S* d = Bs + dCol[p] * ld + dK[p];
            if constexpr(sizeof(S) != sizeof(T))
            {
                const float* v = reinterpret_cast<const float*>(&rb[p]);
#pragma unroll
                for(int e = 0; e < VEC; ++e)
                {
                    const unsigned short h = k1FloatToBf(v[e]);
                    d[e * ld] = h;
                    d[e * ld + plane] = k1BfLo(v[e], h);
                }
            }
            else if constexpr(KV == 2)
            {
                const unsigned short* lo = reinterpret_cast<const unsigned short*>(&rb[p * 2]);
                const unsigned short* hi = reinterpret_cast<const unsigned short*>(&rb[p * 2 + 1]);
#pragma unroll
                for(int e = 0; e < VEC; ++e)
                    *reinterpret_cast<unsigned*>(d + e * ld)
                        = static_cast<unsigned>(lo[e]) | (static_cast<unsigned>(hi[e]) << 16);
            }
            else
            {
                const T* v = reinterpret_cast<const T*>(&rb[p]);
#pragma unroll
                for(int e = 0; e < VEC; ++e)
                    d[e * ld] = v[e];
            }
        }
    }

    // Swizzled row-major LDS tiles (K1Swz): As[m][k], Bs[n][k], 16-bit, VEC 8.
    __device__ void storeRowsSwz(T* As, T* Bs) const
    {
        typedef K1Swz<BK> Swz;
        static_assert(VEC * sizeof(T) == 16 && !DIRECT && PACK != 1, "swizzled tiles");
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
            if(aRow(p))
            {
                const int r = row + p * ROWS_PER_PASS;
                *reinterpret_cast<Raw*>(As + Swz::at(r, Swz::key(r), kv)) = ra[p];
            }
#pragma unroll
        for(int p = 0; p < B_PASSES; ++p)
            if(bRow(p))
            {
                const int r = row + p * ROWS_PER_PASS;
                *reinterpret_cast<Raw*>(Bs + Swz::at(r, Swz::key(r), kv)) = rb[p];
            }
    }

    // Row-major LDS tiles: As[m][k], Bs[n][k], ld elements per row.
    __device__ void storeRows(T* As, T* Bs, int ld) const
    {
#pragma unroll
        for(int p = 0; p < A_PASSES; ++p)
            if(aRow(p))
                *reinterpret_cast<Raw*>(As + (row + p * ROWS_PER_PASS) * ld + kv) = ra[p];
        storeB(Bs, ld, 0);
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
            l[e] = k1BfLo(v[e], he);
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
        storeB(Bs, ld, bPlane);
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
            if constexpr(DIRECT)
            {
                // The filter rows are already k-major: one vector per output channel.
#pragma unroll
                for(int j = 0; j < KV; ++j)
                    *reinterpret_cast<Raw*>(Bs + (dK[p] + j) * ldB + dCol[p]) = rb[p * KV + j];
            }
            else
            {
                const T* v = reinterpret_cast<const T*>(&rb[p]);
#pragma unroll
                for(int e = 0; e < VEC; ++e)
                    Bs[(kv + e) * ldB + row + p * ROWS_PER_PASS] = v[e];
            }
        }
    }
};

// ---------------------------------------------------------------------------------------------
// Register transpose of the channels-first loaders: KV vectors of consecutive positions (one per
// channel) to the KV channels of position e, as one vector (16-bit: channel pairs packed into
// dwords by byte permutes, the low halves for an even position, the high halves for an odd one).
template <class R>
__device__ inline unsigned k1Word(const R& r, int i)
{
    if constexpr(sizeof(R) == 4)
        return r;
    else
        return r[i];
}
template <class R>
__device__ inline void k1SetWord(R& r, int i, unsigned v)
{
    if constexpr(sizeof(R) == 4)
        r = v;
    else
        r[i] = v;
}
template <class T, int KV, class RawK, class RawM>
__device__ inline RawK k1TransposeRow(const RawM (&v)[KV], int e)
{
    RawK r;
    if constexpr(sizeof(T) == 2)
    {
        const unsigned sel = (e & 1) != 0 ? 0x07060302u : 0x05040100u;
#pragma unroll
        for(int j = 0; j < KV; j += 2)
            k1SetWord(
                r, j / 2, __builtin_amdgcn_perm(k1Word(v[j + 1], e / 2), k1Word(v[j], e / 2), sel));
    }
    else
    {
#pragma unroll
        for(int j = 0; j < KV; ++j)
            k1SetWord(r, j, k1Word(v[j], e));
    }
    return r;
}

// ---------------------------------------------------------------------------------------------
// Channels-first A tile (CFX 1): the gathered tensor (forward x, backward data dy) is read in
// place from packed N C (D H) W, where consecutive GEMM rows are consecutive output positions
// along the width. The launcher decides which problems it serves (the gathered tensor packed
// channels-first, the output written by the CF epilogue, the block's images addressed by 32-bit
// buffer offsets); B (the filter) is loaded exactly as in K1Tiles, the tap-outer reduction
// unchanged.
//
// A unit is KV consecutive reduction channels of MV consecutive rows. A thread loads each of its
// channels' MV positions as one vector (width stride 1, all MV rows in one output row and their
// input positions inside the input row) or element by element (a row edge, padding, an image
// boundary, a width stride above 1), and stores the unit transposed: MV runs of KV channels, one
// LDS store per row. Units take channels fastest, so a wave's LDS stores spread over the rows'
// banks while each channel's loads stay runs of consecutive positions.
template <class T, int BM, int BN, int BK, int VEC, int THREADS, int DIR>
struct K1TilesCf : K1Tiles<T, BM, BN, BK, VEC, THREADS, DIR, 0, 0>
{
    typedef K1Tiles<T, BM, BN, BK, VEC, THREADS, DIR, 0, 0> Base;
    static constexpr int ELEMS = BM * BK / THREADS; // A elements a thread loads per k-step
    static constexpr int VMAX = 16 / static_cast<int>(sizeof(T));
    static constexpr int MV = ELEMS / 2 < VMAX ? ELEMS / 2 : VMAX;
    static constexpr int KV0 = ELEMS / MV;
    static constexpr int KV = KV0 < VMAX ? KV0 : VMAX;
    static constexpr int RG = BM / MV; // row groups
    static constexpr int CPASSES = ELEMS / (MV * KV);
    static_assert(K1_BUFFER != 0 || sizeof(T) == 0,
                  "channels-first tiles address the block's images by buffer offsets");
    static_assert(ELEMS >= 2 && BM * BK == ELEMS * THREADS && BM % MV == 0 && BK % KV == 0
                      && ELEMS % (MV * KV) == 0,
                  "channels-first tile units");
    typedef typename K1Raw<MV * sizeof(T)>::type RawM; // MV positions of one channel
    typedef typename K1Raw<KV * sizeof(T)>::type RawK; // KV channels of one row
    typedef typename K1Raw<sizeof(T)>::type One;
    // Element views of the vectors, by value (bit casts keep them in registers).
    struct ElemsM
    {
        One v[MV];
    };

    int plane; // gathered tensor's pixels per channel (D H W)
    // Aligned form (16-bit, MV 8, width stride 1 or 2, input and output widths multiples of 8):
    // every unit's rows are 8 consecutive output columns of one row, so its inputs span whole
    // aligned 16-byte vectors of one input row, loaded aligned and shifted in registers by the
    // tap's offset within a vector (uniform over the block).
    bool aligned;
    int cImg[CPASSES]; // image of the unit's first row, from the block's first image
    int cD[CPASSES], cH[CPASSES], cW[CPASSES]; // first row's output position
    int cRows[CPASSES]; // the unit's rows that exist (MV when all of them do)
    int cRg[CPASSES], cKg[CPASSES];
    RawM ca[CPASSES][KV];

    __device__ void init(const K1Args& a,
                         const K1Gather& gth,
                         const K1TapMap& mp,
                         int g,
                         long long m0,
                         int n0,
                         int tid)
    {
        Base::init(a, gth, mp, g, m0, n0, tid);
        const K1Gather& gt = this->gather;
        plane = gt.in[0] * gt.in[1] * gt.in[2];
        aligned = sizeof(T) == 2 && MV == 8 && (gt.stride[2] == 1 || gt.stride[2] == 2)
                  && gt.in[2] % 8 == 0 && gt.out[2] % 8 == 0;
        const long long spatial = static_cast<long long>(gt.out[0]) * gt.out[1] * gt.out[2];
        const long long firstImage = m0 / spatial;
#pragma unroll
        for(int p = 0; p < CPASSES; ++p)
        {
            const int u = tid + p * THREADS;
            cKg[p] = u % (BK / KV);
            cRg[p] = u / (BK / KV);
            const long long m = m0 + cRg[p] * MV;
            const long long img = m / spatial;
            int rest = static_cast<int>(m - img * spatial);
            cImg[p] = static_cast<int>(img - firstImage);
            cW[p] = rest % gt.out[2];
            rest /= gt.out[2];
            cH[p] = rest % gt.out[1];
            cD[p] = rest / gt.out[1];
            const long long left = gt.rows - m;
            cRows[p] = left >= MV ? MV : (left > 0 ? static_cast<int>(left) : 0);
        }
    }

    /// One channel's MV rows of unit @p p at tap offsets (td, th, tw) into ca[p][j] (zeros when
    /// !cOk).
    __device__ void loadChannel(int p, int j, int c, int td, int th, int tw, bool cOk)
    {
        const K1Gather& gt = this->gather;
        const int sw = gt.stride[2];
        const int chans = gt.chans;
        const int chan0 = static_cast<int>(gt.chanOffset);
        const int id = cD[p] * gt.stride[0] - gt.pad[0] + td;
        const int ih = cH[p] * gt.stride[1] - gt.pad[1] + th;
        const int iw = cW[p] * sw - gt.pad[2] + tw;
        const bool rowIn = (static_cast<unsigned>(id) < static_cast<unsigned>(gt.in[0]))
                           & (static_cast<unsigned>(ih) < static_cast<unsigned>(gt.in[1]));
        if constexpr(sizeof(T) == 2 && MV == 8)
        {
            if(aligned & (cRows[p] == MV))
            {
                // a0: the aligned vector at or left of iw; dl: iw's element within it (uniform).
                const int dl = iw & 7;
                const int a0 = iw - dl;
                const int rowOff
                    = (cImg[p] * chans + chan0 + c) * plane + (id * gt.in[1] + ih) * gt.in[2];
                const bool live = cOk & rowIn;
                auto vec = [&](int a) {
                    const bool ok
                        = live & (static_cast<unsigned>(a) < static_cast<unsigned>(gt.in[2]));
                    return k1BufferLoad<RawM>(
                        this->aRsrc,
                        ok ? static_cast<unsigned>((rowOff + a) * static_cast<int>(sizeof(T)))
                           : K1_OUT_OF_RANGE);
                };
                unsigned w[12];
                const RawM v0 = vec(a0), v1 = vec(a0 + 8);
#pragma unroll
                for(int q = 0; q < 4; ++q)
                {
                    w[q] = v0[q];
                    w[4 + q] = v1[q];
                    w[8 + q] = 0;
                }
                RawM r;
                if(sw == 1)
                {
                    // Elements dl .. dl + 7 of the 16.
                    const int q0 = dl >> 1;
#pragma unroll
                    for(int k = 0; k < 4; ++k)
                    {
                        unsigned lo = w[k], hi = w[k + 1];
#pragma unroll
                        for(int q = 1; q < 4; ++q)
                            if(q == q0)
                            {
                                lo = w[k + q];
                                hi = w[k + q + 1];
                            }
                        r[k] = (dl & 1) != 0 ? __builtin_amdgcn_alignbyte(hi, lo, 2) : lo;
                    }
                }
                else
                {
                    // Elements dl, dl + 2, .., dl + 14 of the 24: one half of each dword.
                    const RawM v2 = vec(a0 + 16);
#pragma unroll
                    for(int q = 0; q < 4; ++q)
                        w[8 + q] = v2[q];
                    const int q0 = dl >> 1;
                    const unsigned sel = (dl & 1) != 0 ? 0x07060302u : 0x05040100u;
#pragma unroll
                    for(int k = 0; k < 4; ++k)
                    {
                        unsigned x0 = w[2 * k], x1 = w[2 * k + 1];
#pragma unroll
                        for(int q = 1; q < 4; ++q)
                            if(q == q0)
                            {
                                x0 = w[2 * k + q];
                                x1 = w[2 * k + q + 1];
                            }
                        r[k] = __builtin_amdgcn_perm(x1, x0, sel);
                    }
                }
                ca[p][j] = r;
                return;
            }
        }
        // One vector when the MV rows are consecutive positions of one input row, all inside.
        if((sw == 1) & (cRows[p] == MV) & (cW[p] + MV <= gt.out[2]) & rowIn & (iw >= 0)
           & (iw + MV <= gt.in[2]))
        {
            const int off
                = ((cImg[p] * chans + chan0 + c) * plane + (id * gt.in[1] + ih) * gt.in[2] + iw)
                  * static_cast<int>(sizeof(T));
            ca[p][j] = k1BufferLoad<RawM>(this->aRsrc,
                                          cOk ? static_cast<unsigned>(off) : K1_OUT_OF_RANGE);
            return;
        }
        // Element by element, walking the rows' output positions.
        ElemsM e;
        int img = cImg[p], od = cD[p], oh = cH[p], ow = cW[p];
#pragma unroll
        for(int i = 0; i < MV; ++i)
        {
            const int xd = od * gt.stride[0] - gt.pad[0] + td;
            const int xh = oh * gt.stride[1] - gt.pad[1] + th;
            const int xw = ow * sw - gt.pad[2] + tw;
            const bool ok = cOk & (i < cRows[p])
                            & (static_cast<unsigned>(xd) < static_cast<unsigned>(gt.in[0]))
                            & (static_cast<unsigned>(xh) < static_cast<unsigned>(gt.in[1]))
                            & (static_cast<unsigned>(xw) < static_cast<unsigned>(gt.in[2]));
            const int off
                = ((img * chans + chan0 + c) * plane + (xd * gt.in[1] + xh) * gt.in[2] + xw)
                  * static_cast<int>(sizeof(T));
            e.v[i]
                = k1BufferLoad<One>(this->aRsrc, ok ? static_cast<unsigned>(off) : K1_OUT_OF_RANGE);
            if(++ow == gt.out[2])
            {
                ow = 0;
                if(++oh == gt.out[1])
                {
                    oh = 0;
                    if(++od == gt.out[0])
                    {
                        od = 0;
                        ++img;
                    }
                }
            }
        }
        ca[p][j] = __builtin_bit_cast(RawM, e);
    }

    /// The cursor's iteration into the registers (A in place, B as K1Tiles), the cursor one on.
    __device__ void load()
    {
        const int it = this->curIt;
        const int c0 = this->curC * BK;
        const int qs = this->curQs, qr = this->curQr, qt = this->curQt;
        ++this->curIt;
        if(++this->curC == this->cChunks)
        {
            this->curC = 0;
            if(++this->curQs == this->gather.taps[2])
            {
                this->curQs = 0;
                if(++this->curQr == this->gather.taps[1])
                {
                    this->curQr = 0;
                    ++this->curQt;
                }
            }
        }
        const int td = qt * this->gather.step[0], th = qr * this->gather.step[1],
                  tw = qs * this->gather.step[2];
        const bool itOk = it < this->endIt;
#pragma unroll
        for(int p = 0; p < CPASSES; ++p)
#pragma unroll
            for(int j = 0; j < KV; ++j)
            {
                const int c = c0 + cKg[p] * KV + j;
                loadChannel(p, j, c, td, th, tw, itOk & (c < this->gather.cgroup));
            }
        const int c = c0 + this->kv;
        this->loadBTap(qt, qr, qs, c0, c, (c < this->gather.cgroup) & itOk, itOk);
    }

    /// Row @p e of unit @p p: its KV channels as one vector.
    __device__ RawK rowOf(int p, int e) const
    {
        return k1TransposeRow<T, KV, RawK>(ca[p], e);
    }

    // Row-major LDS tiles: As[m][k], Bs[n][k], ld elements per row.
    __device__ void storeRows(T* As, T* Bs, int ld) const
    {
#pragma unroll
        for(int p = 0; p < CPASSES; ++p)
#pragma unroll
            for(int e = 0; e < MV; ++e)
                *reinterpret_cast<RawK*>(As + (cRg[p] * MV + e) * ld + cKg[p] * KV) = rowOf(p, e);
        this->storeB(Bs, ld, 0);
    }

    // Swizzled row-major LDS tiles (K1Swz), 16-bit, VEC 8: a unit's KV channels stay within one
    // 16-byte chunk.
    __device__ void storeRowsSwz(T* As, T* Bs) const
    {
        typedef K1Swz<BK> Swz;
        static_assert(VEC * sizeof(T) == 16 && 8 % KV == 0, "swizzled tiles");
#pragma unroll
        for(int p = 0; p < CPASSES; ++p)
#pragma unroll
            for(int e = 0; e < MV; ++e)
            {
                const int r = cRg[p] * MV + e;
                const int k = cKg[p] * KV;
                *reinterpret_cast<RawK*>(As + Swz::at(r, Swz::key(r), k & ~7) + (k & 7))
                    = rowOf(p, e);
            }
#pragma unroll
        for(int p = 0; p < Base::B_PASSES; ++p)
            if(this->bRow(p))
            {
                const int r = this->row + p * Base::ROWS_PER_PASS;
                *reinterpret_cast<typename Base::Raw*>(Bs + Swz::at(r, Swz::key(r), this->kv))
                    = this->rb[p];
            }
    }

    // k-major LDS tiles (the FMA kernel): As[k][m] takes each channel's MV positions as they
    // were loaded, Bs[k][n] as in K1Tiles.
    __device__ void storeColumns(T* As, T* Bs, int ldA, int ldB) const
    {
#pragma unroll
        for(int p = 0; p < CPASSES; ++p)
#pragma unroll
            for(int j = 0; j < KV; ++j)
                *reinterpret_cast<RawM*>(As + (cKg[p] * KV + j) * ldA + cRg[p] * MV) = ca[p][j];
#pragma unroll
        for(int p = 0; p < Base::B_PASSES; ++p)
        {
            if(!this->bRow(p))
                continue;
            const T* v = reinterpret_cast<const T*>(&this->rb[p]);
#pragma unroll
            for(int e = 0; e < VEC; ++e)
                Bs[(this->kv + e) * ldB + this->row + p * Base::ROWS_PER_PASS] = v[e];
        }
    }
};

// Pointwise channels-first input: coalesced spatial vectors, transposed inside the A tile.
// The matcher requires whole spatial vectors, one group, unit stride and no padding.
template <class T, int BM, int BN, int BK, int VEC, int THREADS, int DIR, int WD, int PACK>
struct K1CFPointTiles : K1Tiles<T, BM, BN, BK, VEC, THREADS, DIR, WD, PACK>
{
    using Base = K1Tiles<T, BM, BN, BK, VEC, THREADS, DIR, WD, PACK>;
    using Raw = typename Base::Raw;
    static constexpr int PASSES = BM * BK / (VEC * THREADS);
    static_assert(DIR == K1_FORWARD && WD == 0 && PACK == 0 && BM % VEC == 0);
    static_assert(BM * BK % (VEC * THREADS) == 0);
    long long offsets[PASSES];
    bool live[PASSES];
    int channels[PASSES], rows[PASSES], spatial;
    Raw values[PASSES];
    __device__ void init(const K1Args& a,
                         const K1Gather& gth,
                         const K1TapMap& mp,
                         int g,
                         long long m0,
                         int n0,
                         int tid)
    {
        Base::init(a, gth, mp, g, m0, n0, tid);
        spatial = a.di * a.hi * a.wi;
#pragma unroll
        for(int p = 0; p < PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            channels[p] = u / (BM / VEC);
            rows[p] = u % (BM / VEC) * VEC;
            const long long m = m0 + rows[p];
            const long long img = m / spatial, pos = m % spatial;
            live[p] = m < gth.rows;
            offsets[p] = (img * a.c + gth.chanOffset + channels[p]) * spatial + pos;
        }
    }
    __device__ void load()
    {
        const int c0 = this->curIt * BK;
        const bool valid = this->curIt++ < this->endIt;
#pragma unroll
        for(int p = 0; p < PASSES; ++p)
        {
            values[p] = Raw{};
            if(valid && live[p] && c0 + channels[p] < this->gather.cgroup)
                values[p] = *reinterpret_cast<const Raw*>(this->src + offsets[p]
                                                          + static_cast<long long>(c0) * spatial);
        }
        const int c = c0 + this->kv;
        this->loadBTap(0, 0, 0, c0, c, c < this->gather.cgroup && valid, valid);
    }
    __device__ void storeRows(T* as, T* bs, int ld) const
    {
        typedef T Elements __attribute__((ext_vector_type(VEC)));
#pragma unroll
        for(int p = 0; p < PASSES; ++p)
        {
            const Elements v = __builtin_bit_cast(Elements, values[p]);
#pragma unroll
            for(int e = 0; e < VEC; ++e)
                as[(rows[p] + e) * ld + channels[p]] = v[e];
        }
        this->storeB(bs, ld, 0);
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
// Halo A tile. At unit gather stride, with the gathered tensor's spatial dims equal to the rows'
// (forward stride 1 with "same" padding, backward data at stride 1), GEMM row m reads pixel
// m + shift(tap) of the gathered tensor in its linear (image, d, h, w) order. So all taps of a
// block's BM rows read one run of BM + (largest shift - smallest shift) pixels: it is staged once
// per channel chunk, every tap reads it at a row offset, and a row whose tap leaves the image
// reads a zero LDS row instead.

/// The gather's smallest pixel shift over its taps in @p minShift; false unless the gather has
/// unit stride, equal dims and a shift range within @p cap rows.
__device__ inline bool k1HaloFits(const K1Gather& g, int cap, long long& minShift)
{
    long long lo = 0, hi = 0;
    const long long S[3] = {static_cast<long long>(g.in[1]) * g.in[2], g.in[2], 1};
    for(int i = 0; i < 3; ++i)
    {
        if(g.stride[i] != 1 || g.out[i] != g.in[i])
            return false;
        const long long e = static_cast<long long>(g.taps[i] - 1) * g.step[i];
        lo += ((e < 0 ? e : 0) - g.pad[i]) * S[i];
        hi += ((e > 0 ? e : 0) - g.pad[i]) * S[i];
    }
    minShift = lo;
    return hi - lo <= cap;
}

/// HR consecutive pixels of the gathered tensor from pixel pix0 (possibly before the first),
/// BK channels each, through registers into LDS rows. The buffer starts at the first real pixel
/// the block reads, so offsets stay small whatever the tensor's size.
template <class T, int HR, int BK, int VEC, int THREADS>
struct K1Halo
{
    typedef typename K1Raw<VEC * sizeof(T)>::type Raw;
    static constexpr int ROW_THREADS = BK / VEC;
    static constexpr int RPP = THREADS / ROW_THREADS;
    static constexpr int PASSES = HR / RPP;
    static_assert(HR % RPP == 0, "halo rows");
    /// Rows from the block's first staged pixel to LDS row 0 (K1HaloCf aligns its base).
    static constexpr int delta = 0;
    __amdgpu_buffer_rsrc_t rsrc;
    long long pix0, base, pixels;
    int chans, cgroup, kv, row, chanOffset;
    Raw r[PASSES];

    __device__ void init(const void* src, const K1Gather& g, long long firstPixel, int tid)
    {
        chans = g.chans;
        cgroup = g.cgroup;
        chanOffset = static_cast<int>(g.chanOffset);
        pixels = g.rows;
        pix0 = firstPixel;
        base = firstPixel < 0 ? 0 : firstPixel;
        kv = (tid % ROW_THREADS) * VEC;
        row = tid / ROW_THREADS;
        rsrc = k1Rsrc(static_cast<const T*>(src) + base * chans,
                      (pixels - base) * chans * static_cast<long long>(sizeof(T)));
    }
    __device__ void load(int c0)
    {
        const int c = c0 + kv;
#pragma unroll
        for(int p = 0; p < PASSES; ++p)
        {
            const long long pix = pix0 + row + p * RPP;
            const bool ok = (c < cgroup) & (pix >= base) & (pix < pixels);
            const unsigned offset
                = ok ? static_cast<unsigned>(((pix - base) * chans + chanOffset + c)
                                             * static_cast<long long>(sizeof(T)))
                     : K1_OUT_OF_RANGE;
            r[p] = k1BufferLoad<Raw>(rsrc, offset);
        }
    }
    /// S unsigned short: three-bf16 planes through Split (K1Tiles::storeSplit).
    template <class S, class Split>
    __device__ void store(S* As, int ld, int plane) const
    {
#pragma unroll
        for(int p = 0; p < PASSES; ++p)
        {
            if constexpr(sizeof(S) != sizeof(T))
                Split::storeSplit(r[p], As + (row + p * RPP) * ld + kv, plane);
            else
                *reinterpret_cast<Raw*>(As + (row + p * RPP) * ld + kv) = r[p];
        }
    }
};

/// The halo A tile of a problem whose gathered tensor is packed channels-first (CFX 1), read in
/// place. The caller gives the linear (image, d, h, w) pixels its rows' taps reach, first to last
/// (any stride, padding or output extent); a channel's pixels of one image are contiguous, so
/// the run is loaded channel by channel. LDS row 0 is pixel base, the first pixel rounded down to
/// a whole vector of MV pixels, so a channel's MV pixels are one aligned vector load wherever the
/// image's pixel count is a multiple of MV. A unit is KV channels of MV pixels, stored transposed
/// (one LDS row per pixel, as K1TilesCf). The launcher checks the run fits HR rows.
template <class T, int HR, int BK, int THREADS>
struct K1HaloCf
{
    static constexpr int MV = 16 / static_cast<int>(sizeof(T));
#ifndef K1_CF_HKV
#define K1_CF_HKV 4
#endif
    static constexpr int KV = sizeof(T) == 2 ? K1_CF_HKV : K1_CF_HKV / 2;
    static constexpr int KG = BK / KV;
    static constexpr int UNITS = HR / MV * KG;
    static constexpr int PASSES = (UNITS + THREADS - 1) / THREADS;
    static_assert(HR % MV == 0 && BK % KV == 0, "channels-first halo units");
    typedef typename K1Raw<MV * sizeof(T)>::type RawM;
    typedef typename K1Raw<KV * sizeof(T)>::type RawK;
    typedef typename K1Raw<sizeof(T)>::type One;
    struct ElemsM
    {
        One v[MV];
    };
    __amdgpu_buffer_rsrc_t rsrc;
    int delta;
    int chans, cgroup, chanOffset, lin, need;
    long long base; // linear pixel of LDS row 0
    int uOff[PASSES], uPos[PASSES], uMask[PASSES], uRg[PASSES], uKg[PASSES];
    bool uVec[PASSES], uLive[PASSES];
    long long pixels;
    RawM r[PASSES][KV];

    __device__ void
        init(const void* src, const K1Gather& g, long long firstPixel, long long lastPixel, int tid)
    {
        chans = g.chans;
        cgroup = g.cgroup;
        chanOffset = static_cast<int>(g.chanOffset);
        lin = g.in[0] * g.in[1] * g.in[2]; // pixels per channel
        const long long images = g.rows / (static_cast<long long>(g.out[0]) * g.out[1] * g.out[2]);
        pixels = images * lin;
        base = firstPixel >= 0 ? firstPixel / MV * MV : -((-firstPixel + MV - 1) / MV) * MV;
        delta = static_cast<int>(firstPixel - base);
        need = static_cast<int>(lastPixel - base + 1);
        // The buffer starts at the image of the first real pixel.
        const long long img0 = base > 0 ? base / lin : 0;
        const long long imageElems = static_cast<long long>(chans) * lin;
        rsrc = k1Rsrc(static_cast<const T*>(src) + img0 * imageElems,
                      (pixels / lin - img0) * imageElems * static_cast<long long>(sizeof(T)));
#pragma unroll
        for(int p = 0; p < PASSES; ++p)
        {
            const int u = tid + p * THREADS;
            uKg[p] = u % KG;
            uRg[p] = u / KG;
            uLive[p] = u < UNITS && uRg[p] * MV < need;
            const long long p0 = base + uRg[p] * MV;
            // Floor division: pixels before the first have image -1.
            const long long img = p0 >= 0 ? p0 / lin : -((-p0 + lin - 1) / lin);
            uPos[p] = static_cast<int>(p0 - img * lin);
            uOff[p] = static_cast<int>(((img - img0) * chans + chanOffset) * lin + uPos[p]);
            uMask[p] = 0;
#pragma unroll
            for(int i = 0; i < MV; ++i)
                uMask[p] |= (p0 + i >= 0 && p0 + i < pixels ? 1 : 0) << i;
            uVec[p] = p0 >= 0 && uPos[p] + MV <= lin && p0 + MV <= pixels;
        }
    }

    __device__ void load(int c0)
    {
        // Element offset of channel c's unit pixels: uOff + c lin, plus (chans - 1) lin at each
        // image boundary the unit crosses (the next image's same channel).
        const int jump = (chans - 1) * lin;
#pragma unroll
        for(int p = 0; p < PASSES; ++p)
        {
            const int cb = c0 + uKg[p] * KV;
            if(uVec[p])
            {
#pragma unroll
                for(int j = 0; j < KV; ++j)
                {
                    const bool cOk = uLive[p] & (cb + j < cgroup);
                    const int off = (uOff[p] + (cb + j) * lin) * static_cast<int>(sizeof(T));
                    r[p][j] = k1BufferLoad<RawM>(
                        rsrc, cOk ? static_cast<unsigned>(off) : K1_OUT_OF_RANGE);
                }
                continue;
            }
#pragma unroll
            for(int j = 0; j < KV; ++j)
            {
                const bool cOk = uLive[p] & (cb + j < cgroup);
                ElemsM e;
                int o = uOff[p] + (cb + j) * lin, pos = uPos[p];
#pragma unroll
                for(int i = 0; i < MV; ++i)
                {
                    const bool ok = cOk & (((uMask[p] >> i) & 1) != 0);
                    e.v[i] = k1BufferLoad<One>(
                        rsrc,
                        ok ? static_cast<unsigned>(o * static_cast<int>(sizeof(T)))
                           : K1_OUT_OF_RANGE);
                    ++o;
                    if(++pos == lin)
                    {
                        pos = 0;
                        o += jump;
                    }
                }
                r[p][j] = __builtin_bit_cast(RawM, e);
            }
        }
    }

    /// Row-major As[row][k] (ld per row); S is T (no three-bf16 form).
    template <class S, class Split>
    __device__ void store(S* As, int ld, int /*plane*/) const
    {
        static_assert(sizeof(S) == sizeof(T), "channels-first halo: native math only");
#pragma unroll
        for(int p = 0; p < PASSES; ++p)
        {
            if(!uLive[p])
                continue;
#pragma unroll
            for(int e = 0; e < MV; ++e)
                *reinterpret_cast<RawK*>(As + (uRg[p] * MV + e) * ld + uKg[p] * KV)
                    = k1TransposeRow<T, KV, RawK>(r[p], e);
        }
    }
};

// ---------------------------------------------------------------------------------------------
// XCD-contiguous block order. gfx942 and gfx950 deal workgroups to their XCDs (6 on MI300A, 8 on
// MI300X / MI325X / MI350X / MI355X) round-robin in launch order, and each XCD has its own L2: the
// blocks sharing an A tile (the n tiles of one m tile) and its halo rows (neighbouring m tiles)
// land on different L2s. The remap gives each of the `xcds` XCDs one contiguous range of logical
// blocks, n tile fastest. A bijection of the grid; a no-op for one XCD and on other parts.

#if defined(__gfx942__) || defined(__gfx950__)
#define K1_XCDS 8
#else
#define K1_XCDS 1
#endif

__device__ inline void k1XcdOrder(unsigned& bx, unsigned& by, unsigned& bz, int xcds)
{
#if defined(__gfx950__)
    // Every gfx950 part has 8.
    (void)xcds;
    constexpr unsigned x = K1_XCDS;
#else
    if(xcds <= 1)
    {
        return;
    }
    const unsigned x = static_cast<unsigned>(xcds);
#endif
    const unsigned gx = gridDim.x, gy = gridDim.y;
    const unsigned total = gx * gy * gridDim.z;
    const unsigned id = blockIdx.x + gx * (blockIdx.y + gy * blockIdx.z);
    const unsigned per = total / x, rem = total % x;
    const unsigned xcd = id % x, local = id / x;
    const unsigned logical
        = xcd < rem ? xcd * (per + 1) + local : rem * (per + 1) + (xcd - rem) * per + local;
    by = logical % gy;
    bx = (logical / gy) % gx;
    bz = logical / (gy * gx);
}

// ---------------------------------------------------------------------------------------------
// DMA k-step barrier: this wave's loads older than the last N have landed in LDS and its LDS
// reads are done, then the workgroup barrier (no full vmcnt drain, which __syncthreads forces).
template <int N>
__device__ inline void k1DmaBarrier()
{
    asm volatile("s_waitcnt vmcnt(%0) lgkmcnt(0)\n\ts_barrier" ::"n"(N) : "memory");
}

// LDS hand-off barrier of the matrix-core kernel: its waves exchange data through LDS only. On
// gfx12, __syncthreads() fences every address space at workgroup scope and so adds a global_inv
// (L0 invalidate) after each barrier, two per k-step; fences on the local address space suffice.
// Other families keep __syncthreads() (its code there is what this would emit, or close to it).
__device__ inline void k1LdsBarrier()
{
#if HKP_ARCH_GFX12
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
    __syncthreads();
#endif
}

// Matrix-core kernel. Block: WM x WN waves, tile BM x BN, k-step BK, STAGES LDS buffers (2: the
// next tile is stored while the current one is multiplied, one barrier per k-step). Grid: (m
// tiles, n tiles, groups [x phases for backward data] [x splits]). XCD 1: XCD-contiguous block
// order. HALO 1: the halo A loop (the launcher checks k1HaloFits with K1_HALO_CAP rows; STAGES 1).
// WD 1: backward data reads the graph's K T R S C/g filter directly (K1Tiles). PACK 1: k-steps of
// whole taps when the reduction channels per tap divide BK (K1Tiles; not with the halo loop).
// CF 1: the output (forward y, backward data dx) is written packed channels-first straight from
// the epilogue (and the split-K partials indexed alike), so it needs no staging transpose.
// SWZ 1: XOR-swizzled, unpadded LDS tiles (K1Swz) read in whole 16-byte chunks, 8-byte fragments
// two k-steps at a time: no LDS bank conflicts on the fragment reads (16-bit, VEC 8, no halo,
// direct filter or tap packing; the flat reduction, PACK 2, swizzles at the LDS store, so it
// takes no DMA).
// CFX 1: 1-D problems read the gathered tensor in place from packed channels-first activations
// (K1TilesCf), with the CF output: no activation is staged.

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
          int STAGES = 1,
          int XCD = 0,
          int HALO = 0,
          int WD = 0,
          int PACK = 0,
          int CF = 0,
          int SWZ = 0,
          int DMA = 0,
          int CFX = 0>
__device__ void k1ConvMma(const K1Args& a CONV_EPI_FN_PARAM)
{
    constexpr int WAVE = Atom::WAVE;
    constexpr int THREADS = WM * WN * WAVE;
    constexpr bool SPLIT = MATH == K1_MATH_BF16X3;
    // Tiles of the multiply's element type: T, or bf16 planes (hi, lo) for three-bf16.
    typedef typename K1If<SPLIT, unsigned short, T>::type S;
    constexpr int LD = SWZ ? BK : BK + 16 / static_cast<int>(sizeof(S));
    constexpr int WTM = BM / WM, WTN = BN / WN;
    constexpr int DIM = Atom::DIM;
    constexpr int FM = WTM / DIM, FN = WTN / DIM;
    constexpr int PLANES = SPLIT ? 2 : 1;
    constexpr int STAGE_S = PLANES * (BM + BN) * LD; // in S
    constexpr int STAGE = (STAGE_S * static_cast<int>(sizeof(S)) + static_cast<int>(sizeof(T)) - 1)
                          / static_cast<int>(sizeof(T)); // in T
    // Epilogue: each wave stages 16 rows of its tile at a time.
    constexpr int OLD = WTN + 16 / static_cast<int>(sizeof(T));
    // Channels-first epilogue (CF 1): each wave stages DIM columns x CFR rows, column-major.
    constexpr int CFR = WTM < 64 ? WTM : 64;
    constexpr int OLDC = CFR + 16 / static_cast<int>(sizeof(T));
    constexpr int OUT = CF ? WM * WN * DIM * OLDC : WM * WN * DIM * OLD;
    // The epilogues store 16-byte output vectors wherever a strip row (a run of CFR rows
    // channels-first) holds whole ones, whatever the reduction's channel vector VEC: narrow
    // reductions (VEC 1) write wide outputs as vectors, not one element and one decoded output
    // position at a time. Patch backward data keeps VEC: its columns are contiguous in dx only
    // within runs of S C, which VEC divides.
    constexpr int OV = 16 / static_cast<int>(sizeof(T));
    constexpr int OVEC = WTN % OV == 0 && !(K1_PATCH && DIR == K1_BACKWARD_DATA) ? OV : VEC;
    constexpr int OVECC = CFR % OV == 0 ? OV : VEC;
    // Halo: A is [PLANES][AROWS][LD], the staged rows and one zero row, then one B tile.
    constexpr int HRPP = THREADS / (BK / VEC);
    // Channels-first halo rows: up to a vector of pixels more, for its aligned base.
    constexpr int CFMV = 16 / static_cast<int>(sizeof(T));
    constexpr int HROWS = CFX != 0 ? (HALO * BM + K1_HALO_CAP + 2 * CFMV - 1) / CFMV * CFMV
                                   : (BM + K1_HALO_CAP + HRPP - 1) / HRPP * HRPP;
    constexpr int AROWS = HALO ? HROWS + 1 : BM;
    constexpr int APL = AROWS * LD; // A plane stride, in S
    constexpr int HSTAGE_S = PLANES * (AROWS + BN) * LD;
    constexpr int HSTAGE
        = (HSTAGE_S * static_cast<int>(sizeof(S)) + static_cast<int>(sizeof(T)) - 1)
          / static_cast<int>(sizeof(T));
    constexpr int MAIN = HALO ? HSTAGE : (DMA != 0 ? DMA : STAGES) * STAGE;
    constexpr int LDS = MAIN > OUT ? MAIN : OUT;
    static_assert(WTM % DIM == 0 && WTN % DIM == 0 && BK % Atom::KA == 0, "tile");
    static_assert(HALO == 0 || STAGES == 1, "halo is single-stage");
    static_assert(HALO == 0 || PACK == 0, "the halo loop runs one tap per k-step");
    // STAGES 2 raced on gfx11 while backward data stored its B tile with transposed 2-byte LDS
    // writes (a barrier hid it, a fence did not); with vector row stores only it passes.
    static_assert(STAGES == 1 || STAGES == 2, "stages");
    static_assert(
        SWZ == 0 || (!SPLIT && HALO == 0 && WD == 0 && PACK != 1 && sizeof(T) == 2 && VEC == 8),
        "swizzled tiles: 16-bit native, no halo, direct filter or tap packing");
    static_assert(DMA == 0 || (SWZ != 0 && STAGES == 1 && DMA >= 2 && DMA <= 4 && PACK != 1),
                  "DMA: swizzled tiles, 2 to 4 LDS stages, no tap packing");
    static_assert(K1_PATCH == 0 || DIR == K1_FORWARD || CF == 0,
                  "patch backward data writes channels-last dx");
    static_assert(CFX == 0
                      || (CF == 1 && !SPLIT && WD == 0 && PACK == 0 && DMA == 0 && K1_PATCH == 0),
                  "channels-first input: CF 1 output, native math, no direct filter, tap packing, "
                  "DMA or patch view");

    __shared__ __attribute__((aligned(16))) T lds[LDS];

    K1Gather gather;
    K1TapMap map;
    int g, phase;
    unsigned bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    if constexpr(XCD != 0 && K1_XCDS > 1)
    {
        k1XcdOrder(bx, by, bz, a.xcds);
    }
    const int splits = a.splits > 1 ? a.splits : 1;
    const int split = static_cast<int>(bz) % splits;
    k1Setup<DIR>(a, static_cast<int>(bz) / splits, gather, map, g, phase);
    const long long m0 = (static_cast<long long>(bx) + a.tileBase) * BM;
    if(m0 >= gather.rows)
    {
        return;
    }
    const int ncols = DIR == K1_FORWARD ? a.k / a.groups : a.c / a.groups;
    const int n0 = by * BN;
    const int tid = threadIdx.x;
    const int lane = tid % WAVE;
    const int wave = tid / WAVE;
    const int wm = wave % WM, wn = wave / WM;
#if defined(CONV_EPI) && CONV_EPI
    // The fused epilogue: each output column's per-channel operands, loaded before the
    // reduction so their latency hides behind it (a short reduction otherwise waits on them at
    // the store).
    // Patch backward data's columns are filter elements, not channels: it takes no epilogue
    // (the graph parse declines fused backward data).
    static_assert(K1_PATCH == 0 || DIR == K1_FORWARD, "patch backward data fuses no epilogue");
    ConvEpiGen::Chan chans[FN];
#pragma unroll
    for(int j = 0; j < FN; ++j)
    {
        const int en = n0 + wn * WTN + j * DIM + lane % DIM;
        chans[j] = epi.chan(g * ncols + en, en < ncols);
    }
#endif

    typedef typename K1If<
        CFX != 0,
        K1TilesCf<T, BM, BN, BK, VEC, THREADS, DIR>,
        typename K1If<CF == 2,
                      K1CFPointTiles<T, BM, BN, BK, VEC, THREADS, DIR, WD, PACK>,
                      K1Tiles<T, BM, BN, BK, VEC, THREADS, DIR, WD, PACK>>::type>::type InputTiles;
    InputTiles tiles;
    tiles.init(a, gather, map, g, m0, n0, tid);
    if constexpr(DMA != 0 && PACK == 2)
    {
        // DMA with the flat reduction: the thread's LDS slot takes its row's swizzled chunk
        // (loadLdsFlat), the same for each of its rows.
        typedef K1Swz<BK> Swz;
        static_assert(decltype(tiles)::ROWS_PER_PASS % (Swz::RPL * Swz::CPR) == 0,
                      "a thread's rows share one swizzle key");
        tiles.kv = ((tiles.kv / VEC) ^ Swz::key(tiles.row)) * VEC;
    }

    typename Atom::acc acc[FM][FN];
#pragma unroll
    for(int i = 0; i < FM; ++i)
#pragma unroll
        for(int j = 0; j < FN; ++j)
            acc[i][j] = typename Atom::acc{};

    const int fragRow = lane % DIM;
    const int kb = Atom::template kBase<Atom::KPL>(lane);
    // This split's k-iterations: a contiguous range of the reduction.
    const int total = tiles.iterations();
    const int first = static_cast<int>(static_cast<long long>(total) * split / splits);
    const int iters
        = static_cast<int>(static_cast<long long>(total) * (split + 1) / splits) - first;
    // The LDS row of each A fragment (the halo loop moves them per tap).
    int arow[FM];
#pragma unroll
    for(int i = 0; i < FM; ++i)
        arow[i] = wm * WTM + i * DIM + fragRow;

    // Swizzled tiles: each fragment read is whole 16-byte chunks. 8-byte fragments are read in
    // pairs (two k-steps per chunk; A and B permute k alike, so the products are unchanged).
    typedef K1Swz<BK> Swz;
    constexpr int FB = Atom::KPL * static_cast<int>(sizeof(T)); // fragment bytes
    constexpr int PAIR = SWZ && FB == 8 ? 2 : 1;
    int aKey[FM], bKey[FN];
    if constexpr(SWZ != 0)
    {
#pragma unroll
        for(int i = 0; i < FM; ++i)
            aKey[i] = Swz::key(arow[i]);
#pragma unroll
        for(int j = 0; j < FN; ++j)
            bKey[j] = Swz::key(wn * WTN + j * DIM + fragRow);
    }
    const int kbp = Atom::template kBase<Atom::KPL * PAIR>(lane);
    auto multiplySwz = [&](const S* As, const S* Bs) {
        if constexpr(SWZ != 0)
        {
            typedef typename K1Raw<16>::type Chunk;
            static_assert(FB == 8 || FB == 16 || FB == 32, "fragment size");
            static_assert(BK % (Atom::KA * PAIR) == 0, "chunked reads span whole k-steps");
#pragma unroll
            for(int kk = 0; kk < BK; kk += Atom::KA * PAIR)
            {
                typename Atom::frag af[FM][PAIR], bf[FN][PAIR];
                auto read = [&](const S* base, int r, int key, typename Atom::frag* f) {
                    if constexpr(FB == 32)
                    {
                        typedef unsigned Two __attribute__((ext_vector_type(8)));
                        const Chunk c0
                            = *reinterpret_cast<const Chunk*>(base + Swz::at(r, key, kk + kbp));
                        const Chunk c1
                            = *reinterpret_cast<const Chunk*>(base + Swz::at(r, key, kk + kbp + 8));
                        Two t;
#pragma unroll
                        for(int e = 0; e < 4; ++e)
                        {
                            t[e] = c0[e];
                            t[e + 4] = c1[e];
                        }
                        f[0] = __builtin_bit_cast(typename Atom::frag, t);
                    }
                    else if constexpr(FB == 16)
                    {
                        const Chunk c
                            = *reinterpret_cast<const Chunk*>(base + Swz::at(r, key, kk + kbp));
                        f[0] = __builtin_bit_cast(typename Atom::frag, c);
                    }
                    else
                    {
                        typedef typename K1Raw<8>::type Half;
                        const Chunk c
                            = *reinterpret_cast<const Chunk*>(base + Swz::at(r, key, kk + kbp));
                        Half lo, hi;
                        lo[0] = c[0];
                        lo[1] = c[1];
                        hi[0] = c[2];
                        hi[1] = c[3];
                        f[0] = __builtin_bit_cast(typename Atom::frag, lo);
                        f[1] = __builtin_bit_cast(typename Atom::frag, hi);
                    }
                };
#pragma unroll
                for(int i = 0; i < FM; ++i)
                    read(As, arow[i], aKey[i], af[i]);
#pragma unroll
                for(int j = 0; j < FN; ++j)
                    read(Bs, wn * WTN + j * DIM + fragRow, bKey[j], bf[j]);
#pragma unroll
                for(int q = 0; q < PAIR; ++q)
#pragma unroll
                    for(int i = 0; i < FM; ++i)
#pragma unroll
                        for(int j = 0; j < FN; ++j)
                            acc[i][j] = Atom::mma(af[i][q], bf[j][q], acc[i][j]);
            }
        }
    };

    // A planes at As (hi) and As + APL (lo); B planes at Bs and Bs + BN * LD.
    auto multiply = [&](const S* As, const S* Bs) {
        if constexpr(SWZ != 0)
        {
            multiplySwz(As, Bs);
            return;
        }
#pragma unroll
        for(int kk = 0; kk < BK; kk += Atom::KA)
        {
            if constexpr(SPLIT)
            {
                typename Atom::frag ah[FM], al[FM], bh[FN], bl[FN];
#pragma unroll
                for(int i = 0; i < FM; ++i)
                {
                    const S* p = As + arow[i] * LD + kk + kb;
                    ah[i] = k1Frag<Atom>(p);
                    al[i] = k1Frag<Atom>(p + APL);
                }
#pragma unroll
                for(int j = 0; j < FN; ++j)
                {
                    const S* p = Bs + (wn * WTN + j * DIM + fragRow) * LD + kk + kb;
                    bh[j] = k1Frag<Atom>(p);
                    bl[j] = k1Frag<Atom>(p + BN * LD);
                }
                // Cross terms take the high parts with infinities zeroed (k1BfFinite).
                typename Atom::frag af[FM], bf[FN];
#pragma unroll
                for(int i = 0; i < FM; ++i)
                    af[i] = k1FiniteFrag<Atom>(ah[i]);
#pragma unroll
                for(int j = 0; j < FN; ++j)
                    bf[j] = k1FiniteFrag<Atom>(bh[j]);
#pragma unroll
                for(int i = 0; i < FM; ++i)
#pragma unroll
                    for(int j = 0; j < FN; ++j)
                    {
                        acc[i][j] = Atom::mma(al[i], bf[j], acc[i][j]);
                        acc[i][j] = Atom::mma(af[i], bl[j], acc[i][j]);
                        acc[i][j] = Atom::mma(ah[i], bh[j], acc[i][j]);
                    }
            }
            else
            {
                typename Atom::frag af[FM], bf[FN];
#pragma unroll
                for(int i = 0; i < FM; ++i)
                    af[i] = k1Frag<Atom>(As + arow[i] * LD + kk + kb);
#pragma unroll
                for(int j = 0; j < FN; ++j)
                    bf[j] = k1Frag<Atom>(Bs + (wn * WTN + j * DIM + fragRow) * LD + kk + kb);
                if constexpr(K1Xf32<Atom>::value)
                {
                    // xf32: each fragment rounded once, for every MFMA that reuses it.
#pragma unroll
                    for(int i = 0; i < FM; ++i)
                        af[i] = k1Tf32(af[i]);
#pragma unroll
                    for(int j = 0; j < FN; ++j)
                        bf[j] = k1Tf32(bf[j]);
                }
#pragma unroll
                for(int i = 0; i < FM; ++i)
#pragma unroll
                    for(int j = 0; j < FN; ++j)
                        acc[i][j] = Atom::mma(af[i], bf[j], acc[i][j]);
            }
        }
    };

    S* tile = reinterpret_cast<S*>(lds);
    if constexpr(HALO != 0)
    {
        // Reduction order: channel chunk outer, tap inner (split ranges follow this order). Per
        // chunk the halo rows are stored once, the next chunk's are loaded into registers, and
        // each tap's B tile is loaded into registers before the previous tap's MFMAs and stored
        // after them (one B buffer: two barriers per tap).
        long long minShift = 0;
        k1HaloFits(gather, K1_HALO_CAP, minShift);
        typename K1If<CFX != 0,
                      K1HaloCf<T, HROWS, BK, THREADS>,
                      K1Halo<T, HROWS, BK, VEC, THREADS>>::type halo;
        // Channels-first: the linear input pixel of row m's tap 0 (pads aside), and the tap
        // offsets' range, give the run of pixels the block reads.
        const K1Pos cfSpatial = static_cast<K1Pos>(gather.out[0]) * gather.out[1] * gather.out[2];
        const int cfPlane = gather.in[0] * gather.in[1] * gather.in[2];
        auto cfPixel = [&](long long m) {
            const long long img = m / cfSpatial;
            K1Pos rest = static_cast<K1Pos>(m - img * cfSpatial);
            const int w = static_cast<int>(rest % gather.out[2]);
            rest /= gather.out[2];
            const int h = static_cast<int>(rest % gather.out[1]);
            const int d = static_cast<int>(rest / gather.out[1]);
            return img * cfPlane
                   + (static_cast<long long>(d) * gather.stride[0] * gather.in[1]
                      + static_cast<long long>(h) * gather.stride[1])
                         * gather.in[2]
                   + static_cast<long long>(w) * gather.stride[2];
        };
        if constexpr(CFX != 0)
        {
            const int axis[3] = {gather.in[1] * gather.in[2], gather.in[2], 1};
            long long lo = 0, hi = 0;
#pragma unroll
            for(int i = 0; i < 3; ++i)
            {
                const int e = (gather.taps[i] - 1) * gather.step[i];
                lo += static_cast<long long>((e < 0 ? e : 0) - gather.pad[i]) * axis[i];
                hi += static_cast<long long>((e > 0 ? e : 0) - gather.pad[i]) * axis[i];
            }
            // A row reads only pixels of its own image (the rest are padding), so the run is
            // clamped to the first row's image from below and the last row's from above; within
            // an image a later row's pixels never start before an earlier row's.
            const long long mLast = (m0 + BM < gather.rows ? m0 + BM : gather.rows) - 1;
            const long long img0 = m0 / cfSpatial, imgL = mLast / cfSpatial;
            long long first = cfPixel(m0) + lo, last = cfPixel(mLast) + hi;
            first = first < img0 * cfPlane ? img0 * cfPlane : first;
            first = first > (img0 + 1) * cfPlane ? (img0 + 1) * cfPlane : first;
            last = last < imgL * cfPlane - 1 ? imgL * cfPlane - 1 : last;
            last = last > (imgL + 1) * cfPlane - 1 ? (imgL + 1) * cfPlane - 1 : last;
            halo.init(DIR == K1_FORWARD ? a.x : a.y, gather, first, last, tid);
        }
        else
        {
            halo.init(DIR == K1_FORWARD ? a.x : a.y, gather, m0 + minShift, tid);
        }
        typedef K1Tiles<T, BM, BN, BK, VEC, THREADS, DIR, WD, PACK> Tiles;
        S* As = tile;
        S* Bs = tile + PLANES * APL;
        for(int e = tid; e < PLANES * BK; e += THREADS)
        {
            As[(e / BK) * APL + HROWS * LD + e % BK] = S{};
        }
        int od[FM], oh[FM], ow[FM];
        const K1Pos spatial = static_cast<K1Pos>(gather.out[0]) * gather.out[1] * gather.out[2];
#pragma unroll
        for(int i = 0; i < FM; ++i)
        {
            const long long m = m0 + arow[i];
            const long long img = m / spatial;
            K1Pos rest = static_cast<K1Pos>(m - img * spatial);
            ow[i] = static_cast<int>(rest % gather.out[2]);
            rest /= gather.out[2];
            oh[i] = static_cast<int>(rest % gather.out[1]);
            od[i] = m < gather.rows ? static_cast<int>(rest / gather.out[1]) : -0x40000000;
        }
        // Channels-first: each row's LDS row at tap 0 (pads aside); a tap adds its linear offset.
        int cfRow[FM];
        if constexpr(CFX != 0)
        {
#pragma unroll
            for(int i = 0; i < FM; ++i)
            {
                // Input coordinates of tap 0 (pads aside); rows past the last stay out of range.
                cfRow[i] = static_cast<int>(cfPixel(m0 + arow[i]) - halo.base);
                od[i] = m0 + arow[i] < gather.rows ? od[i] * gather.stride[0] : -0x40000000;
                oh[i] *= gather.stride[1];
                ow[i] *= gather.stride[2];
            }
        }
        const int nT = tiles.nTaps;
        const int t1 = gather.taps[1], t2 = gather.taps[2];
        const int end = first + iters;
        int ch = iters > 0 ? first / nT : 0;
        int q = first - ch * nT;
        // The current tap's (qt, qr, qs).
        int qt = q / (t2 * t1), qr = (q / t2) % t1, qs = q % t2;
        if(iters > 0)
        {
            halo.load(ch * BK);
            tiles.loadB(qt, qr, qs, ch * BK);
        }
        for(int it = first; it < end;)
        {
            const int qEnd = (end - ch * nT) < nT ? end - ch * nT : nT;
            const bool nextChunk = (ch + 1) * nT < end;
            k1LdsBarrier();
            halo.template store<S, Tiles>(As, LD, APL);
            tiles.storeB(Bs, LD, BN * LD);
            k1LdsBarrier();
            if(nextChunk)
            {
                halo.load((ch + 1) * BK);
            }
            for(; q < qEnd; ++q)
            {
                int nt = qt, nr = qr, ns = qs + 1;
                if(ns == t2)
                {
                    ns = 0;
                    if(++nr == t1)
                    {
                        nr = 0;
                        ++nt;
                    }
                }
                if(q + 1 < qEnd)
                {
                    tiles.loadB(nt, nr, ns, ch * BK);
                }
                else if(nextChunk)
                {
                    tiles.loadB(0, 0, 0, (ch + 1) * BK);
                }
                const int d0 = qt * gather.step[0] - gather.pad[0];
                const int d1 = qr * gather.step[1] - gather.pad[1];
                const int d2 = qs * gather.step[2] - gather.pad[2];
                const int shift = static_cast<int>((static_cast<long long>(d0) * gather.in[1] + d1)
                                                       * gather.in[2]
                                                   + d2 - minShift)
                                  + halo.delta;
#pragma unroll
                for(int i = 0; i < FM; ++i)
                {
                    const bool ok
                        = static_cast<unsigned>(od[i] + d0) < static_cast<unsigned>(gather.in[0])
                          && static_cast<unsigned>(oh[i] + d1) < static_cast<unsigned>(gather.in[1])
                          && static_cast<unsigned>(ow[i] + d2)
                                 < static_cast<unsigned>(gather.in[2]);
                    if constexpr(CFX != 0)
                        arow[i] = ok ? cfRow[i]
                                           + static_cast<int>(
                                               (static_cast<long long>(d0) * gather.in[1] + d1)
                                                   * gather.in[2]
                                               + d2)
                                     : HROWS;
                    else
                        arow[i] = ok ? wm * WTM + i * DIM + fragRow + shift : HROWS;
                }
                multiply(As, Bs);
                if(q + 1 < qEnd)
                {
                    k1LdsBarrier();
                    tiles.storeB(Bs, LD, BN * LD);
                    k1LdsBarrier();
                }
                qt = nt;
                qr = nr;
                qs = ns;
            }
            it = (ch + 1) * nT;
            ++ch;
            q = 0;
            qt = qr = qs = 0;
        }
    }
    else
    {
        // A then B; each PLANES planes.
        auto store = [&](S* base) {
            if constexpr(SPLIT)
            {
                tiles.storeRowsSplit(base, base + 2 * BM * LD, LD, BM * LD, BN * LD);
            }
            else if constexpr(SWZ != 0)
            {
                tiles.storeRowsSwz(base, base + BM * LD);
            }
            else
            {
                tiles.storeRows(base, base + BM * LD, LD);
            }
        };
        // Iterations past the split's end load zeros, so the prefetch needs no branch.
        tiles.cursorInit(first, first + iters);
        if(DMA == 0 && iters > 0)
        {
            tiles.load();
        }
        if constexpr(DMA != 0)
        {
#if K1_BUFFER && defined(__gfx950__)
            // DMA stages in a ring: tile it + DMA - 1 loads while tile it multiplies; one barrier
            // per k-step. Loads past the split's end write zeros to a free stage.
            typedef K1Tiles<T, BM, BN, BK, VEC, THREADS, DIR, WD, PACK> Tiles;
            constexpr int LOADS = Tiles::A_PASSES + Tiles::B_PASSES;
#pragma unroll
            for(int st = 0; st < DMA - 1; ++st)
            {
                tiles.loadLds(tile + st * STAGE_S, tile + st * STAGE_S + BM * LD, wave);
            }
            for(int it = 0; it < iters; ++it)
            {
                k1DmaBarrier<LOADS*(DMA - 2)>();
                const int next = (it + DMA - 1) % DMA;
                tiles.loadLds(tile + next * STAGE_S, tile + next * STAGE_S + BM * LD, wave);
                const S* cur = tile + (it % DMA) * STAGE_S;
                multiply(cur, cur + BM * LD);
            }
#else
            __builtin_trap();
#endif
        }
        else if constexpr(STAGES == 2)
        {
            if(iters > 0)
            {
                store(tile);
            }
            for(int it = 0; it < iters; ++it)
            {
                k1LdsBarrier();
                tiles.load();
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
                k1LdsBarrier();
                store(tile);
                k1LdsBarrier();
                tiles.load();
                multiply(tile, tile + PLANES * BM * LD);
            }
        }
    }

    if(splits > 1 || K1_PATCH == 3)
    {
        // Split-K: this split's fp32 partial, indexed like the output; the reduce adds them.
        // Col2im (K1_PATCH 3) always writes its columns there, for ConvIgemmCol2im.
        float* partial = a.partial + static_cast<long long>(split) * a.outElems;
#pragma unroll
        for(int i = 0; i < FM; ++i)
#pragma unroll
            for(int j = 0; j < FN; ++j)
            {
                const int nn = n0 + wn * WTN + j * DIM + lane % DIM;
#pragma unroll
                for(int e = 0; e < Atom::ACC; ++e)
                {
                    const long long m = m0 + wm * WTM + i * DIM + Atom::dRow(lane, e);
                    if(m < gather.rows && nn < ncols)
                    {
                        partial[CF ? k1OutIndexCf<DIR>(a, gather, g, phase, m, nn)
                                   : k1OutIndex<DIR>(a, gather, g, phase, m, nn)]
                            = acc[i][j][e];
                    }
                }
            }
        return;
    }

    T* out = static_cast<T*>(DIR == K1_FORWARD ? a.y : a.x);
    if constexpr(CF != 0)
    {
        // Channels-first epilogue: per DIM-column slab and CFR-row chunk, accumulators to the
        // wave's column-major LDS strip, then runs of OVECC rows (consecutive positions of one
        // channel) to global; single elements where a run is not contiguous in the output (an
        // image boundary, a stride phase) or past the last row.
        typedef typename K1Raw<OVECC * sizeof(T)>::type Raw;
        k1LdsBarrier();
        T* strip = lds + wave * DIM * OLDC;
        constexpr int RUNS = CFR / OVECC;
        K1CfWalk walk;
        walk.template init<DIR>(a, gather);
#pragma unroll
        for(int j = 0; j < FN; ++j)
#if defined(CONV_EPI) && CONV_EPI
        {
            const ConvEpiGen::Chan& chan = chans[j];
#endif
#pragma unroll
            for(int rc = 0; rc < WTM / CFR; ++rc)
            {
#pragma unroll
                for(int i = 0; i < CFR / DIM; ++i)
#pragma unroll
                    for(int e = 0; e < Atom::ACC; ++e)
                        strip[(lane % DIM) * OLDC + i * DIM + Atom::dRow(lane, e)]
#if defined(CONV_EPI) && CONV_EPI
                            = K1Elem<T>::store(epi(acc[rc * (CFR / DIM) + i][j][e], chan, 0, true));
#else
                            = K1Elem<T>::store(acc[rc * (CFR / DIM) + i][j][e]);
#endif
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
                for(int v = lane; v < DIM * RUNS; v += WAVE)
                {
                    const int col = v / RUNS;
                    const int run = v - col * RUNS;
                    const long long m = m0 + wm * WTM + rc * CFR + run * OVECC;
                    const int nn = n0 + wn * WTN + j * DIM + col;
                    if(m >= gather.rows || nn >= ncols)
                    {
                        continue;
                    }
                    const T* src = strip + col * OLDC + run * OVECC;
                    const long long o = k1OutIndexCf<DIR>(a, gather, g, phase, m, nn);
                    walk.start(m, o);
                    long long at[OVECC];
#pragma unroll
                    for(int e = 0; e < OVECC; ++e)
                    {
                        at[e] = walk.o;
                        walk.next();
                    }
                    // Positions only increase with m, so a span of OVECC - 1 means contiguous.
                    if(m + OVECC <= gather.rows && o % OVECC == 0 && at[OVECC - 1] == o + OVECC - 1)
                    {
                        *reinterpret_cast<Raw*>(out + o) = *reinterpret_cast<const Raw*>(src);
                    }
                    else
                    {
#pragma unroll
                        for(int e = 0; e < OVECC; ++e)
                            if(m + e < gather.rows)
                                out[at[e]] = src[e];
                    }
                }
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            }
#if defined(CONV_EPI) && CONV_EPI
        }
#endif
        return;
    }

    // Epilogue: per DIM-row slab, accumulators to the wave's LDS strip, then rows of OVEC columns
    // to global (single columns past the last full vector).
    typedef typename K1Raw<OVEC * sizeof(T)>::type Raw;
    k1LdsBarrier();
    T* strip = lds + wave * DIM * OLD;
    constexpr int ROW_VECS = WTN / OVEC;
#pragma unroll
    for(int i = 0; i < FM; ++i)
    {
#pragma unroll
        for(int j = 0; j < FN; ++j)
#pragma unroll
            for(int e = 0; e < Atom::ACC; ++e)
                strip[Atom::dRow(lane, e) * OLD + j * DIM + lane % DIM]
#if defined(CONV_EPI) && CONV_EPI
                    = K1Elem<T>::store(epi(acc[i][j][e], chans[j], 0, true));
#else
                    = K1Elem<T>::store(acc[i][j][e]);
#endif
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        for(int v = lane; v < DIM * ROW_VECS; v += WAVE)
        {
            const int r = v / ROW_VECS;
            const int col = (v - r * ROW_VECS) * OVEC;
            const long long m = m0 + wm * WTM + i * DIM + r;
            const int nn = n0 + wn * WTN + col;
            if(m >= gather.rows || nn >= ncols)
            {
                continue;
            }
            const long long o = k1OutIndex<DIR>(a, gather, g, phase, m, nn);
            if(nn + OVEC <= ncols && o % OVEC == 0)
            {
                *reinterpret_cast<Raw*>(out + o)
                    = *reinterpret_cast<const Raw*>(strip + r * OLD + col);
            }
            else
            {
                for(int e = 0; e < OVEC && nn + e < ncols; ++e)
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
// thread computes TM x TN outputs in runs of 4 rows and 4 columns. fp16 multiplies pairs of k on
// v_dot2_f32_f16 (CONV_IGEMM_FMA_DOT2, on by default; 0 restores one fma a product).
#ifndef CONV_IGEMM_FMA_DOT2
#define CONV_IGEMM_FMA_DOT2 1
#endif

template <class T,
          int BM,
          int BN,
          int BK,
          int THREADS,
          int VEC,
          int DIR,
          int WD = 0,
          int PACK = 0,
          int CFX = 0>
__device__ void k1ConvFma(const K1Args& a CONV_EPI_FN_PARAM)
{
    static_assert(CFX == 0 || (WD == 0 && PACK == 0 && K1_PATCH == 0),
                  "channels-first input: no direct filter, tap packing or patch view");
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
    const int splits = a.splits > 1 ? a.splits : 1;
    const int split = static_cast<int>(blockIdx.z) % splits;
    k1Setup<DIR>(a, static_cast<int>(blockIdx.z) / splits, gather, map, g, phase);
    const long long m0 = (static_cast<long long>(blockIdx.x) + a.tileBase) * BM;
    if(m0 >= gather.rows)
    {
        return;
    }
    const int ncols = DIR == K1_FORWARD ? a.k / a.groups : a.c / a.groups;
    const int n0 = blockIdx.y * BN;
    const int tid = threadIdx.x;
    const int tx = tid % TX, ty = tid / TX;
#if defined(CONV_EPI) && CONV_EPI
    // The fused epilogue: each output column's per-channel operands, loaded before the
    // reduction so their latency hides behind it.
    static_assert(K1_PATCH == 0 || DIR == K1_FORWARD, "patch backward data fuses no epilogue");
    ConvEpiGen::Chan chans[TN];
#pragma unroll
    for(int j = 0; j < TN; ++j)
    {
        const int nn = n0 + ((j / 4) * TX + tx) * 4 + j % 4;
        chans[j] = epi.chan(g * ncols + nn, nn < ncols);
    }
#endif

    typename K1If<CFX != 0,
                  K1TilesCf<T, BM, BN, BK, VEC, THREADS, DIR>,
                  K1Tiles<T, BM, BN, BK, VEC, THREADS, DIR, WD, PACK>>::type tiles;
    tiles.init(a, gather, map, g, m0, n0, tid);

    float acc[TM][TN];
#pragma unroll
    for(int i = 0; i < TM; ++i)
#pragma unroll
        for(int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    // This split's k-iterations: a contiguous range of the reduction.
    const int total = tiles.iterations();
    const int first = static_cast<int>(static_cast<long long>(total) * split / splits);
    const int iters
        = static_cast<int>(static_cast<long long>(total) * (split + 1) / splits) - first;
    tiles.cursorInit(first, first + iters);
    if(iters > 0)
    {
        tiles.load();
    }
    for(int it = 0; it < iters; ++it)
    {
        __syncthreads();
        tiles.storeColumns(As, Bs, LDA, LDB);
        __syncthreads();
        tiles.load();
#if CONV_IGEMM_FMA_DOT2
        if constexpr(__is_same(T, k1_half) && BK % 2 == 0)
        {
            // fp16: pairs of k on v_dot2_f32_f16 (two exact fp16 products added to the fp32
            // sum), half the VALU work of one fma_mix a product. A pair is the 16-bit elements
            // of rows kk and kk + 1 of the k-major tile, packed into one dword.
            typedef _Float16 K1H2 __attribute__((ext_vector_type(2)));
#pragma unroll 2
            for(int kk = 0; kk < BK; kk += 2)
            {
                unsigned ap[TM], bp[TN];
                // Runs of 4 elements are 8-byte aligned (the tiles' leading dimensions are
                // multiples of 8 elements): one 8-byte read a row, the pairs built by a byte
                // permute (row kk's element in the low half, row kk + 1's in the high).
                typedef unsigned K1U2 __attribute__((ext_vector_type(2)));
#pragma unroll
                for(int q = 0; q < TM / 4; ++q)
                {
                    const int m = (q * TY + ty) * 4;
                    const K1U2 lo = *reinterpret_cast<const K1U2*>(As + kk * LDA + m);
                    const K1U2 hi = *reinterpret_cast<const K1U2*>(As + (kk + 1) * LDA + m);
                    ap[q * 4 + 0] = __builtin_amdgcn_perm(hi[0], lo[0], 0x05040100u);
                    ap[q * 4 + 1] = __builtin_amdgcn_perm(hi[0], lo[0], 0x07060302u);
                    ap[q * 4 + 2] = __builtin_amdgcn_perm(hi[1], lo[1], 0x05040100u);
                    ap[q * 4 + 3] = __builtin_amdgcn_perm(hi[1], lo[1], 0x07060302u);
                }
#pragma unroll
                for(int q = 0; q < TN / 4; ++q)
                {
                    const int nn = (q * TX + tx) * 4;
                    const K1U2 lo = *reinterpret_cast<const K1U2*>(Bs + kk * LDB + nn);
                    const K1U2 hi = *reinterpret_cast<const K1U2*>(Bs + (kk + 1) * LDB + nn);
                    bp[q * 4 + 0] = __builtin_amdgcn_perm(hi[0], lo[0], 0x05040100u);
                    bp[q * 4 + 1] = __builtin_amdgcn_perm(hi[0], lo[0], 0x07060302u);
                    bp[q * 4 + 2] = __builtin_amdgcn_perm(hi[1], lo[1], 0x05040100u);
                    bp[q * 4 + 3] = __builtin_amdgcn_perm(hi[1], lo[1], 0x07060302u);
                }
#pragma unroll
                for(int i = 0; i < TM; ++i)
#pragma unroll
                    for(int j = 0; j < TN; ++j)
                        acc[i][j] = k1Dot2(__builtin_bit_cast(K1H2, ap[i]),
                                           __builtin_bit_cast(K1H2, bp[j]),
                                           acc[i][j]);
            }
            continue;
        }
#endif
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
    float* partial = a.partial + static_cast<long long>(split) * a.outElems;
    if constexpr(CFX != 0)
    {
        // Channels-first output: a thread's run of 4 rows is 4 consecutive positions of each of
        // its channels, one vector store where they are contiguous and aligned in the output,
        // else element by element (an image boundary, a stride phase, past the last row).
        typedef typename K1Raw<4 * sizeof(T)>::type Raw4;
        K1CfWalk walk;
        walk.template init<DIR>(a, gather);
#pragma unroll
        for(int q = 0; q < TM / 4; ++q)
        {
            const long long m = m0 + (q * TY + ty) * 4;
#pragma unroll
            for(int j = 0; j < TN; ++j)
            {
                const int nn = n0 + ((j / 4) * TX + tx) * 4 + j % 4;
                if(m >= gather.rows || nn >= ncols)
                    continue;
                const long long o = k1OutIndexCf<DIR>(a, gather, g, phase, m, nn);
                walk.start(m, o);
                long long at[4];
                T v[4];
#pragma unroll
                for(int e = 0; e < 4; ++e)
                {
                    at[e] = walk.o;
                    walk.next();
                    if(splits > 1)
                        continue;
#if defined(CONV_EPI) && CONV_EPI
                    v[e] = K1Elem<T>::store(epi(acc[q * 4 + e][j], chans[j], at[e], true));
#else
                    v[e] = K1Elem<T>::store(acc[q * 4 + e][j]);
#endif
                }
                const bool whole = m + 4 <= gather.rows && at[3] == o + 3;
                if(splits > 1)
                {
#pragma unroll
                    for(int e = 0; e < 4; ++e)
                        if(m + e < gather.rows)
                            partial[at[e]] = acc[q * 4 + e][j];
                }
                else if(whole && o % 4 == 0)
                {
                    Raw4 r;
                    __builtin_memcpy(&r, v, sizeof(r));
                    *reinterpret_cast<Raw4*>(out + o) = r;
                }
                else
                {
#pragma unroll
                    for(int e = 0; e < 4; ++e)
                        if(m + e < gather.rows)
                            out[at[e]] = v[e];
                }
            }
        }
        return;
    }
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
                const long long o = k1OutIndex<DIR>(a, gather, g, phase, m, nn);
                if(splits > 1 || K1_PATCH == 3)
                    partial[o] = acc[i][j];
                else
#if defined(CONV_EPI) && CONV_EPI
                    out[o] = K1Elem<T>::store(epi(acc[i][j], chans[j], o, true));
#else
                    out[o] = K1Elem<T>::store(acc[i][j]);
#endif
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

// Group fusion's filter expansion (ConvIgemmExpandArgs): every element of the block-diagonal
// filter, one per thread, grid-strided.
__device__ inline void k1ExpandGroups(const ConvIgemmExpandArgs& a)
{
    const long long cf = a.f * a.cg;
    const long long total = a.k * a.taps * cf;
    const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
    for(long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
        i += step)
    {
        const long long c = i % cf;
        const long long t = i / cf;
        const long long tap = t % a.taps;
        const long long kk = t / a.taps;
        const bool own = c / a.cg == (kk / a.kg) % a.f;
        const long long src = kk * a.srcK + (c % a.cg) * a.srcC + tap * a.srcTap;
        if(a.elementBytes == 4)
            static_cast<unsigned*>(a.dst)[i] = own ? static_cast<const unsigned*>(a.src)[src] : 0u;
        else
            static_cast<unsigned short*>(a.dst)[i]
                = own ? static_cast<const unsigned short*>(a.src)[src]
                      : static_cast<unsigned short>(0);
    }
}

// Batched 2-D transpose through a TT x TT LDS tile (CONV_IGEMM_TRANSPOSE_TILE): a 1-D grid of
// (column tile, row tile, batch), column tiles fastest; 256 threads. Reads are 16-byte vectors
// along the source rows when the columns are whole vectors and the source 16-byte aligned, and
// writes 16-byte vectors along the destination rows (gathered from LDS) when the rows are whole
// vectors and the destination aligned; otherwise one element per access. A thread issues all its
// loads before storing any (clamped in-range addresses, no per-load branch).
template <class E>
__device__ inline void k1TransposeTile(const ConvIgemmTransposeArgs& a,
                                       long long tileCol,
                                       long long tileRow,
                                       long long b)
{
    constexpr int TT = CONV_IGEMM_TRANSPOSE_TILE;
    constexpr int V = 16 / static_cast<int>(sizeof(E));
    constexpr int RV = TT / V; // vectors per tile row
    constexpr int VPT = TT * RV / 256; // vectors per thread
    constexpr int EPT = TT * TT / 256; // elements per thread
    constexpr int P = TT + (sizeof(E) == 2 ? 2 : 1); // LDS pitch: rows on different banks
    static_assert(VPT >= 1 && EPT >= 1, "tile");
    typedef typename K1Raw<16>::type Raw;
    __shared__ E tile[TT * P];
    const E* src = static_cast<const E*>(a.src) + b * a.rows * a.cols;
    E* dst = static_cast<E*>(a.dst) + b * a.rows * a.cols;
    const long long r0 = tileRow * TT, c0 = tileCol * TT;
    const int tid = threadIdx.x;
    const bool vecIn = a.cols % V == 0 && (reinterpret_cast<unsigned long long>(a.src) & 15) == 0;
    const bool vecOut = a.rows % V == 0 && (reinterpret_cast<unsigned long long>(a.dst) & 15) == 0;
    if(vecIn)
    {
        Raw v[VPT];
#pragma unroll
        for(int k = 0; k < VPT; ++k)
        {
            const int i = tid + k * 256, r = i / RV, c = (i - r * RV) * V;
            const long long rr = r0 + r < a.rows ? r0 + r : a.rows - 1;
            const long long cc = c0 + c < a.cols ? c0 + c : a.cols - V;
            v[k] = *reinterpret_cast<const Raw*>(src + rr * a.cols + cc);
        }
#pragma unroll
        for(int k = 0; k < VPT; ++k)
        {
            const int i = tid + k * 256, r = i / RV, c = (i - r * RV) * V;
            const E* e = reinterpret_cast<const E*>(&v[k]);
#pragma unroll
            for(int j = 0; j < V; ++j)
                tile[r * P + c + j] = e[j];
        }
    }
    else
    {
        E v[EPT];
#pragma unroll
        for(int k = 0; k < EPT; ++k)
        {
            const int i = tid + k * 256, r = i / TT, c = i - r * TT;
            const long long rr = r0 + r < a.rows ? r0 + r : a.rows - 1;
            const long long cc = c0 + c < a.cols ? c0 + c : a.cols - 1;
            v[k] = src[rr * a.cols + cc];
        }
#pragma unroll
        for(int k = 0; k < EPT; ++k)
        {
            const int i = tid + k * 256, r = i / TT, c = i - r * TT;
            tile[r * P + c] = v[k];
        }
    }
    __syncthreads();
    if(vecOut)
    {
#pragma unroll
        for(int k = 0; k < VPT; ++k)
        {
            const int i = tid + k * 256, c = i / RV, r = (i - c * RV) * V;
            if(c0 + c < a.cols && r0 + r < a.rows)
            {
                Raw v;
                E* e = reinterpret_cast<E*>(&v);
#pragma unroll
                for(int j = 0; j < V; ++j)
                    e[j] = tile[(r + j) * P + c];
                *reinterpret_cast<Raw*>(dst + (c0 + c) * a.rows + r0 + r) = v;
            }
        }
    }
    else
    {
#pragma unroll
        for(int k = 0; k < EPT; ++k)
        {
            const int i = tid + k * 256, c = i / TT, r = i - c * TT;
            if(c0 + c < a.cols && r0 + r < a.rows)
                dst[(c0 + c) * a.rows + r0 + r] = tile[r * P + c];
        }
    }
}

__device__ inline void k1Transpose(const ConvIgemmTransposeArgs& a)
{
    constexpr int TT = CONV_IGEMM_TRANSPOSE_TILE;
    const long long colTiles = (a.cols + TT - 1) / TT;
    const long long rowTiles = (a.rows + TT - 1) / TT;
    const long long block = blockIdx.x;
    const long long tileCol = block % colTiles;
    const long long tileRow = (block / colTiles) % rowTiles;
    const long long b = block / (colTiles * rowTiles);
    if(b >= a.batches)
    {
        return;
    }
    if(a.elementBytes == 4)
        k1TransposeTile<unsigned>(a, tileCol, tileRow, b);
    else
        k1TransposeTile<unsigned short>(a, tileCol, tileRow, b);
}

// Split-K reduce: each output element is the sum of the splits' fp32 partials, in split order.
template <class T>
__device__ inline void k1ReduceSplits(const K1ReduceArgs& a)
{
    T* out = static_cast<T*>(a.out);
    const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
    for(long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < a.elems;
        i += step)
    {
        float sum = 0.0f;
        for(int sp = 0; sp < a.splits; ++sp)
            sum += a.partial[static_cast<long long>(sp) * a.elems + i];
        out[i] = K1Elem<T>::store(sum);
    }
}

// Col2im backward data: each dx element (packed N D H W C) sums, tap by tap in filter order and
// split by split, the GEMM columns of the dy positions it reaches. A thread takes four channels of
// one position (whole vectors where the channels are a multiple of four).
template <class T>
__device__ inline void k1Col2im(const K1Col2imArgs& a)
{
    T* dx = static_cast<T*>(a.dx);
    const int quads = (a.c + 3) / 4;
    const long long cols = static_cast<long long>(a.c) * a.ft * a.fr * a.fs;
    const long long elems = static_cast<long long>(a.n) * a.di * a.hi * a.wi * quads;
    const bool whole = a.c % 4 == 0;
    const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
    for(long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < elems;
        i += step)
    {
        long long r = i / quads;
        const int c0 = static_cast<int>(i - r * quads) * 4;
        const int iw = static_cast<int>(r % a.wi);
        r /= a.wi;
        const int ih = static_cast<int>(r % a.hi);
        r /= a.hi;
        const int id = static_cast<int>(r % a.di);
        const long long n = r / a.di;
        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for(int t = 0; t < a.ft; ++t)
        {
            const int zd = id + a.pd - t * a.dd;
            const int od = zd / a.sd;
            if(zd < 0 || od * a.sd != zd || od >= a.dout)
                continue;
            for(int q = 0; q < a.fr; ++q)
            {
                const int zh = ih + a.ph - q * a.dh;
                const int oh = zh / a.sh;
                if(zh < 0 || oh * a.sh != zh || oh >= a.ho)
                    continue;
                for(int s = 0; s < a.fs; ++s)
                {
                    const int zw = iw + a.pw - s * a.dw;
                    const int ow = zw / a.sw;
                    if(zw < 0 || ow * a.sw != zw || ow >= a.wo)
                        continue;
                    const long long row
                        = ((n * a.dout + od) * a.ho + oh) * static_cast<long long>(a.wo) + ow;
                    const long long col
                        = ((static_cast<long long>(t) * a.fr + q) * a.fs + s) * a.c + c0;
                    for(int sp = 0; sp < a.splits; ++sp)
                    {
                        const float* p = a.partial + sp * a.splitElems + row * cols + col;
                        if(whole)
                        {
                            typedef float F4 __attribute__((ext_vector_type(4)));
                            const F4 v = *reinterpret_cast<const F4*>(p);
#pragma unroll
                            for(int e = 0; e < 4; ++e)
                                sum[e] += v[e];
                        }
                        else
                        {
                            for(int e = 0; e < 4 && c0 + e < a.c; ++e)
                                sum[e] += p[e];
                        }
                    }
                }
            }
        }
        T* o = dx + (((n * a.di + id) * a.hi + ih) * static_cast<long long>(a.wi) + iw) * a.c + c0;
        if(whole)
        {
            T v[4];
            for(int e = 0; e < 4; ++e)
                v[e] = K1Elem<T>::store(sum[e]);
            typedef typename K1Raw<4 * sizeof(T)>::type Raw4;
            Raw4 raw;
            __builtin_memcpy(&raw, v, sizeof(raw));
            *reinterpret_cast<Raw4*>(o) = raw;
        }
        else
        {
            for(int e = 0; e < 4 && c0 + e < a.c; ++e)
                o[e] = K1Elem<T>::store(sum[e]);
        }
    }
}

#if defined(CONV_EPI) && CONV_EPI
// The fused split-K reduce: the epilogue applied once to each summed element. The channel comes
// from the flat walk (ConvEpiFlatChannel), with no division per element.
template <class T>
__device__ inline void k1ReduceSplitsEpi(const K1ReduceArgs& a, const ConvEpiGen& epi)
{
    T* out = static_cast<T*>(a.out);
    const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
    const long long first = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    ConvEpiFlatChannel flat(epi.e, first, step);
    for(long long i = first; i < a.elems; i += step)
    {
        float sum = 0.0f;
        for(int sp = 0; sp < a.splits; ++sp)
            sum += a.partial[static_cast<long long>(sp) * a.elems + i];
        out[i] = K1Elem<T>::store(epi(sum, epi.chan(flat.ch, true), i, true));
        flat.advance();
    }
}
#endif

// ---------------------------------------------------------------------------------------------
// Three-bf16 channel split (K1Split3Args): the pre-pass of the engines' TF32 forms. A config
// compiled with CONV_TF32_SPLIT3 gains the entry point ConvTf32Split; no other config's code
// changes.

__device__ inline void k1Split3(const K1Split3Tensor& t)
{
    if(t.src == nullptr)
        return;
    const long long total = t.outer * t.groups * t.cg * t.inner;
    const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
    for(long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
        i += step)
    {
        const long long r = i % t.inner;
        long long rest = i / t.inner;
        const long long c = rest % t.cg;
        rest /= t.cg;
        const long long g = rest % t.groups;
        const long long o = rest / t.groups;
        const float v = t.src[i];
        const unsigned short h = k1FloatToBf(v);
        const unsigned short l = k1BfLo(v, h);
        const unsigned short f = k1BfFinite(h);
        unsigned short* d = t.dst + ((o * t.groups + g) * 3 * t.cg + c) * t.inner + r;
#pragma unroll
        for(int part = 0; part < 3; ++part)
        {
            const int kind = (t.pattern >> (2 * part)) & 3;
            d[part * t.cg * t.inner] = kind == 1 ? l : kind == 2 ? f : h;
        }
    }
}

#if defined(CONV_TF32_SPLIT3)
extern "C" __global__ void __launch_bounds__(256) ConvTf32Split(K1Split3Args a)
{
    k1Split3(a.t[0]);
    k1Split3(a.t[1]);
}
#endif
