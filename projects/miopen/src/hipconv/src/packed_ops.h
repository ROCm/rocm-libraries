#pragma once

#include "types.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

// Packed vector type conversion via function template full specialization.
// Only explicitly specialized type/width combinations are supported;
// unsupported combos produce a linker error.
//
// N=2 specializations are the fundamental building blocks: they map directly
// to the 2-element packed HIP intrinsics (__float22bfloat162_rn etc).
// Wider widths (N=4, N=8) compose multiple N=2 calls.
template <typename DstT, typename SrcT, int N>
__device__ __forceinline__ auto packed_convert(__attribute__((ext_vector_type(N))) SrcT a) ->
    __attribute__((ext_vector_type(N))) DstT;

// ---------------------------------------------------------------------------
// N=2 base specializations (one packed intrinsic per call)
// ---------------------------------------------------------------------------

// fp32x2 -> fp16x2
template <>
__device__ __forceinline__ auto packed_convert<fp16_t, fp32_t, 2>(fp32x2_t a) -> fp16x2_t
{
    __half2 h = __float22half2_rn({a[0], a[1]});
    fp16x2_t r;
    __builtin_memcpy(&r, &h, sizeof(r));
    return r;
}

// fp32x2 -> bf16x2
template <>
__device__ __forceinline__ auto packed_convert<bf16_t, fp32_t, 2>(fp32x2_t a) -> bf16x2_t
{
    // __hip_bfloat162 is not trivially copyable, so use __builtin_memcpy
    // with void* casts to silence -Wnontrivial-memcall.
    __hip_bfloat162 h = __float22bfloat162_rn({a[0], a[1]});
    bf16x2_t r;
    __builtin_memcpy(static_cast<void*>(&r), static_cast<void*>(&h), sizeof(r));
    return r;
}

// bf16x2 -> fp32x2 (used by TF32 residual computation)
template <>
__device__ __forceinline__ auto packed_convert<fp32_t, bf16_t, 2>(bf16x2_t a) -> fp32x2_t
{
    __hip_bfloat162 h;
    __builtin_memcpy(static_cast<void*>(&h), static_cast<void*>(&a), sizeof(h));
    float2 f = __bfloat1622float2(h);
    return fp32x2_t{f.x, f.y};
}

// ---------------------------------------------------------------------------
// N=4 specializations: compose two N=2 calls
// ---------------------------------------------------------------------------

// fp32x4 -> fp16x4
template <>
__device__ __forceinline__ auto packed_convert<fp16_t, fp32_t, 4>(fp32x4_t a) -> fp16x4_t
{
    fp16x2_t halves[2];
    halves[0] = packed_convert<fp16_t, fp32_t, 2>(fp32x2_t{a[0], a[1]});
    halves[1] = packed_convert<fp16_t, fp32_t, 2>(fp32x2_t{a[2], a[3]});
    fp16x4_t r;
    __builtin_memcpy(&r, &halves, sizeof(r));
    return r;
}

// fp32x4 -> bf16x4
template <>
__device__ __forceinline__ auto packed_convert<bf16_t, fp32_t, 4>(fp32x4_t a) -> bf16x4_t
{
    bf16x2_t halves[2];
    halves[0] = packed_convert<bf16_t, fp32_t, 2>(fp32x2_t{a[0], a[1]});
    halves[1] = packed_convert<bf16_t, fp32_t, 2>(fp32x2_t{a[2], a[3]});
    bf16x4_t r;
    __builtin_memcpy(&r, &halves, sizeof(r));
    return r;
}

// ---------------------------------------------------------------------------
// TF32 helper: split fp32x4 into (bf16-big, bf16-small) pair.
//
// Implements the canonical TF32-via-BF16 decomposition:
//   big   = round_to_bf16(a)
//   small = round_to_bf16(a - float(big))
// such that  a ≈ float(big) + float(small)  with small being the residual
// captured at bf16 precision.
//
// Emits 2 packed __float22bfloat162_rn for big, 2 packed __bfloat1622float2
// for the round-trip, and 2 more packed __float22bfloat162_rn for small.
// All arithmetic stays at the 2-element packed granularity, matching the
// natural width of the underlying HIP intrinsics.
// ---------------------------------------------------------------------------

struct bf16_pair_x4
{
    bf16x4_t big;
    bf16x4_t small;
};

__device__ __forceinline__ bf16_pair_x4 fp32x4_to_bf16_pair(fp32x4_t a)
{
    fp32x2_t a01{a[0], a[1]};
    fp32x2_t a23{a[2], a[3]};

    bf16x2_t big_halves[2];
    big_halves[0] = packed_convert<bf16_t, fp32_t, 2>(a01);
    big_halves[1] = packed_convert<bf16_t, fp32_t, 2>(a23);

    fp32x2_t big01_f = packed_convert<fp32_t, bf16_t, 2>(big_halves[0]);
    fp32x2_t big23_f = packed_convert<fp32_t, bf16_t, 2>(big_halves[1]);

    bf16x2_t small_halves[2];
    small_halves[0] =
        packed_convert<bf16_t, fp32_t, 2>(fp32x2_t{a[0] - big01_f[0], a[1] - big01_f[1]});
    small_halves[1] =
        packed_convert<bf16_t, fp32_t, 2>(fp32x2_t{a[2] - big23_f[0], a[3] - big23_f[1]});

    bf16_pair_x4 r;
    __builtin_memcpy(&r.big, &big_halves, sizeof(r.big));
    __builtin_memcpy(&r.small, &small_halves, sizeof(r.small));
    return r;
}

// ---------------------------------------------------------------------------
// TF32 helper (8-wide): split fp32x8 into (bf16-big, bf16-small) pair.
//
// 8-wide twin of fp32x4_to_bf16_pair, used by MFMA wrappers whose operands
// are bf16x8 (mfma_16x16x32). Internally just runs the 4-wide split twice
// and memcpys the resulting bf16x4 halves into a contiguous bf16x8. The
// emitted instruction sequence is identical to two back-to-back invocations
// of the 4-wide helper (4 packed __float22bfloat162_rn for big, 4 packed
// __bfloat1622float2 for the round-trip, 4 packed __float22bfloat162_rn for
// small).
// ---------------------------------------------------------------------------

struct bf16_pair_x8
{
    bf16x8_t big;
    bf16x8_t small;
};

__device__ __forceinline__ bf16_pair_x8 fp32x8_to_bf16_pair(fp32x8_t a)
{
    bf16_pair_x4 lo = fp32x4_to_bf16_pair(fp32x4_t{a[0], a[1], a[2], a[3]});
    bf16_pair_x4 hi = fp32x4_to_bf16_pair(fp32x4_t{a[4], a[5], a[6], a[7]});

    bf16x4_t big_halves[2]   = {lo.big, hi.big};
    bf16x4_t small_halves[2] = {lo.small, hi.small};

    bf16_pair_x8 r;
    __builtin_memcpy(&r.big, &big_halves, sizeof(r.big));
    __builtin_memcpy(&r.small, &small_halves, sizeof(r.small));
    return r;
}
