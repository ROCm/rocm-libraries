// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>

#include "ck_tile/core.hpp"

namespace ck_tile {

/// @file
/// Shared pieces of the target model's fused epilogues (row-wise elementwise math around the
/// GEMMs at decode M): model constants, bf16 rounding, lane-group and workgroup reductions,
/// 16-byte row-chunk loads and stores, and the fp8 / MXFP8 / MXFP4 encoders.
///
/// Numerics: math is fp32 and every value is rounded to bf16 where the HF reference
/// (the model's modeling file) produces a bf16 tensor, so a CPU emulation of the reference matches
/// up to the last-ulp differences of rsqrt / exp / sin.

struct MkConstants
{
    static constexpr index_t kHidden    = 8192; // H
    static constexpr index_t kMoeHidden = 4096; // P, pooler output
    static constexpr index_t kHeadDim   = 64;

    static constexpr float kRmsEps         = 1e-5f; // qk_norm, o_norm, Centered norms
    static constexpr float kPostNormEps    = 1e-8f; // post_attention / post_feedforward norms
    static constexpr float kQkScale        = 5.4730064863875f;
    static constexpr float kRopeTheta      = 5e5f;
    static constexpr float kGateTemp       = 0.3f;
    static constexpr float kGateAlphaFloor = 1e-3f;
};

/// fp32 -> bf16 (round to nearest even) -> fp32.
CK_TILE_HOST_DEVICE float mk_round_bf16(float x)
{
    return static_cast<float>(float_to_bf16<bf16_rounding_mode::standard>(x));
}

/// Device: v_exp + v_rcp (~1 ulp), the result is rounded to bf16 by every caller. Host: exact.
CK_TILE_HOST_DEVICE float mk_sigmoid(float x)
{
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_rcpf(1.f + __expf(-x));
#else
    return 1.f / (1.f + expf(-x));
#endif
}

/// bf16(1 + w): HF computes (1.0 + weight) in the weight's dtype (bf16) before .float().
/// @p w is a bf16 weight widened to fp32.
CK_TILE_HOST_DEVICE float mk_one_plus_w(float w) { return mk_round_bf16(1.f + w); }

/// ResidualGate coefficients, out = alpha * residual + beta * h, from the fp32 gate vector.
/// They depend on the weights only: compute them once at weight-load time.
CK_TILE_HOST void mk_residual_gate_coeffs(const float* gate, index_t n, float* alpha, float* beta)
{
    for(index_t i = 0; i < n; ++i)
    {
        const float w = gate[i] / MkConstants::kGateTemp;
        beta[i]       = 1.f / (1.f + std::exp(-w));
        alpha[i]      = std::sqrt(
            std::max(1.f / (1.f + std::exp(w)) * (1.f + beta[i]), MkConstants::kGateAlphaFloor));
    }
}

// ---------------------------------------------------------------------------------------------
// Reductions. Lanes of a group are consecutive and the group size is a power of two; every lane
// of the group must be active.

template <index_t kLanes, typename F>
CK_TILE_DEVICE float mk_group_reduce(float v, F op)
{
    static_assert(kLanes >= 1 && kLanes <= 64 && (kLanes & (kLanes - 1)) == 0);
    const index_t lane = get_lane_id();
    static_for<0, integer_log2_floor(kLanes), 1>{}([&](auto i) {
        constexpr index_t mask = 1 << i.value;
        v                      = op(v, warp_shuffle(v, static_cast<uint32_t>(lane ^ mask)));
    });
    return v;
}

template <index_t kLanes>
CK_TILE_DEVICE float mk_group_sum(float v)
{
    return mk_group_reduce<kLanes>(v, [](float a, float b) { return a + b; });
}

template <index_t kLanes>
CK_TILE_DEVICE float mk_group_max(float v)
{
    return mk_group_reduce<kLanes>(v, [](float a, float b) { return fmaxf(a, b); });
}

/// Workgroup reduction through @p lds (kWarps floats). Collective; the result is in every
/// thread. Two calls must use different @p lds or be separated by a barrier.
template <index_t kWarps, typename F>
CK_TILE_DEVICE float mk_block_reduce(float v, float* lds, F op)
{
    v = mk_group_reduce<get_warp_size()>(v, op);
    if constexpr(kWarps > 1)
    {
        if(get_lane_id() == 0)
            lds[get_thread_id() / get_warp_size()] = v;
        block_sync_lds();
        v = lds[0];
        static_for<1, kWarps, 1>{}([&](auto i) { v = op(v, lds[i.value]); });
    }
    return v;
}

// ---------------------------------------------------------------------------------------------
// Row chunks: 8 consecutive elements (16 bytes of bf16) per lane.

CK_TILE_DEVICE void mk_load_bf16x8(const bf16_t* p, float (&x)[8])
{
    const bf16x8_t v = *c_style_pointer_cast<const bf16x8_t*>(p);
    static_for<0, 8, 1>{}([&](auto i) { x[i.value] = static_cast<float>(v[i.value]); });
}

CK_TILE_DEVICE void mk_load_f32x8(const float* p, float (&x)[8])
{
    const fp32x4_t a = *c_style_pointer_cast<const fp32x4_t*>(p);
    const fp32x4_t b = *c_style_pointer_cast<const fp32x4_t*>(p + 4);
    static_for<0, 4, 1>{}([&](auto i) {
        x[i.value]     = a[i.value];
        x[i.value + 4] = b[i.value];
    });
}

template <typename T>
CK_TILE_DEVICE void mk_load_x8(const T* p, float (&x)[8])
{
    if constexpr(std::is_same_v<T, float>)
        mk_load_f32x8(p, x);
    else
    {
        static_assert(std::is_same_v<T, bf16_t>, "row inputs are bf16 or fp32");
        mk_load_bf16x8(p, x);
    }
}

CK_TILE_DEVICE bf16x8_t mk_to_bf16x8(const float (&x)[8])
{
    bf16x8_t v;
    static_for<0, 8, 1>{}(
        [&](auto i) { v[i.value] = float_to_bf16<bf16_rounding_mode::standard>(x[i.value]); });
    return v;
}

CK_TILE_DEVICE void mk_store_bf16x8(bf16_t* p, const float (&x)[8])
{
    *c_style_pointer_cast<bf16x8_t*>(p) = mk_to_bf16x8(x);
}

// ---------------------------------------------------------------------------------------------
// Encoders. fp8 is OCP e4m3fn (gfx1250's native fp8), saturating to +-448, round to nearest even.

/// Four fp32 -> four e4m3fn bytes, little endian (element 0 in bits [7:0]).
CK_TILE_HOST_DEVICE uint32_t mk_pack_fp8x4(float a, float b, float c, float d)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
    auto sat   = [](float x) { return __builtin_amdgcn_fmed3f(x, 448.f, -448.f); };
    uint32_t r = __builtin_amdgcn_cvt_pk_fp8_f32(sat(a), sat(b), 0, false);
    r          = __builtin_amdgcn_cvt_pk_fp8_f32(sat(c), sat(d), r, true);
    return r;
#else
    auto enc = [](float x) { return static_cast<uint32_t>(bit_cast<uint8_t>(float_to_fp8(x))); };
    return enc(a) | (enc(b) << 8) | (enc(c) << 16) | (enc(d) << 24);
#endif
}

/// fp32 -> e2m1 nibble (sign in bit 3), saturating to +-6, round to nearest even.
/// Grid 0, 0.5, 1, 1.5, 2, 3, 4, 6; ties go to the even code.
CK_TILE_HOST_DEVICE uint32_t mk_e2m1_encode(float x)
{
    const float a = fabsf(x);
    // Each threshold passed adds one code; <= at a tie keeps the even code below.
    const uint32_t code = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) +
                          (a >= 3.5f) + (a > 5.f);
    return code | (bit_cast<uint32_t>(x) >> 31 << 3);
}

/// Shared exponent of an MX block (e8m0, biased by 127) for elements whose largest power of two
/// is 2^kEmaxElem (e4m3: 8, e2m1: 2). amax is first rounded at mantissa 1.75 (bits + 0x200000,
/// as aiter's dynamic_mxfp4_quant), so the scaled amax stays below the element max: e4m3 never
/// saturates, e2m1 saturates only in [6, 7).
template <index_t kEmaxElem>
CK_TILE_HOST_DEVICE uint32_t mk_mx_scale_e8m0(float amax)
{
    const uint32_t bits = (bit_cast<uint32_t>(amax) + 0x200000u) & 0x7F800000u;
    const int32_t e     = static_cast<int32_t>(bits >> 23) - kEmaxElem; // biased
    return static_cast<uint32_t>(e < 0 ? 0 : (e > 254 ? 254 : e));
}

/// 1 / 2^(e8m0 - 127), the multiplier that maps a block onto the element grid.
CK_TILE_HOST_DEVICE float mk_mx_inv_scale(uint32_t e8m0)
{
    return ldexpf(1.f, 127 - static_cast<int32_t>(e8m0));
}

} // namespace ck_tile
