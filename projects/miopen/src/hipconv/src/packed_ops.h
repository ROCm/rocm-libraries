#pragma once

#include "types.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// N-element register vector, the shape every packed op here works on.
template <typename T, int N>
using packed_vec_t = __attribute__((ext_vector_type(N))) T;

template <int N>
using bf16x = __attribute__((ext_vector_type(N))) __bf16;

// A tf32 value, carried as the (big, small) bf16 decomposition the MMAs consume.
struct bf16_pair
{
    __bf16 big;
    __bf16 small;

    __host__ __device__ bf16_pair() : big(0), small(0) {}
    __host__ __device__ bf16_pair(__bf16 big, __bf16 small) : big(big), small(small) {}

    __host__ __device__ bf16_pair& operator=(bf16_pair other)
    {
        big   = other.big;
        small = other.small;
        return *this;
    }
};

template <int N>
struct bf16_pair_x
{
    bf16x<N> big;
    bf16x<N> small;

    __host__ __device__ bf16_pair_x() : big(0), small(0) {}
    __host__ __device__ bf16_pair_x(bf16x<N> big, bf16x<N> small) : big(big), small(small) {}

    __host__ __device__ bf16_pair_x& operator=(bf16_pair_x other)
    {
        big   = other.big;
        small = other.small;
        return *this;
    }
};

using bf16_pair_x2  = bf16_pair_x<2>;
using bf16_pair_x4  = bf16_pair_x<4>;
using bf16_pair_x8  = bf16_pair_x<8>;
using bf16_pair_x16 = bf16_pair_x<16>;

// Packed vector conversion, N in {2, 4, 8}.
//
// Only the 2-element case is real work: it maps to a single packed HIP
// intrinsic (__float22bfloat162_rn etc), and every (Dst, Src) combination that
// has such an intrinsic gets a specialization below. Wider widths go through
// the primary template, which is N/2 of those calls. A combination without a
// 2-element specialization is declared but never defined, so it fails to link
// instead of silently falling back to element-at-a-time conversion.
template <typename DstT, typename SrcT, int N>
__device__ __forceinline__ auto packed_convert(packed_vec_t<SrcT, N> a) -> packed_vec_t<DstT, N>
{
    return __builtin_convertvector(a, packed_vec_t<DstT, N>);
}


// ---------------------------------------------------------------------------
// TF32 helper: split an fp32 vector into its (bf16-big, bf16-small) pair.
//
// Implements the canonical TF32-via-BF16 decomposition:
//   big   = round_to_bf16(a)
//   small = round_to_bf16(a - float(big))
// such that  a ≈ float(big) + float(small)  with small being the residual
// captured at bf16 precision.
// ---------------------------------------------------------------------------

template <int N>
__device__ __forceinline__ auto fp32xN_to_bf16_pair(packed_vec_t<fp32_t, N> a) -> bf16_pair_x<N>
{
    bf16_pair_x<N> r;
    r.big   = packed_convert<bf16_t>(a);
    r.small = packed_convert<bf16_t>(a - packed_convert<fp32_t>(r.big));
    return r;
}
