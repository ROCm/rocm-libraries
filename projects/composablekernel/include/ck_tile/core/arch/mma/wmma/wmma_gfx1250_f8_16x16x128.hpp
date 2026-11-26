// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core/arch/mma/mma_data_format.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/ignore.hpp"

namespace ck_tile::core::arch::mma {

// FP8/BF8 16x16x128 dense WMMA on gfx125x.
//
// gfx1250-strict has no V_WMMA_F{16,32}_16X16X128_{FP8,BF8}_{FP8,BF8}. LLVM gates their builtins
// on the wmma-16x16x128fp8-insts feature, which only gfx1250 has (llvm/llvm-project#228564).
// On gfx1250-strict the same product goes through V_WMMA_F32_16X16X128_F8F6F4, which reads A and
// B in the same register layout; LLVM lowers it to the scaled form with a scale of 1.0 there.
// The f16-accumulator forms have no F8F6F4 equivalent: they accumulate in f32 and round once.
// gfx1250 and gfx1251 keep the native instructions.

template <typename AType, typename BType, bool ReuseA = false, bool ReuseB = false>
CK_TILE_DEVICE fp32x8_t wmma_f32_16x16x128_f8(const int32x16_t& a,
                                              const int32x16_t& b,
                                              const fp32x8_t& c)
{
    static_assert((std::is_same_v<AType, fp8_t> || std::is_same_v<AType, bf8_t>) &&
                      (std::is_same_v<BType, fp8_t> || std::is_same_v<BType, bf8_t>),
                  "A and B must be fp8_t or bf8_t");
#if defined(__gfx1250_strict__)
    return __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(
        PackedDataTypeToFlag_v<AType>, a, PackedDataTypeToFlag_v<BType>, b, 0, c);
#elif defined(__gfx125__)
    if constexpr(std::is_same_v<AType, fp8_t> && std::is_same_v<BType, fp8_t>)
        return __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(a, b, 0, c, ReuseA, ReuseB);
    else if constexpr(std::is_same_v<AType, fp8_t> && std::is_same_v<BType, bf8_t>)
        return __builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8(a, b, 0, c, ReuseA, ReuseB);
    else if constexpr(std::is_same_v<AType, bf8_t> && std::is_same_v<BType, fp8_t>)
        return __builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8(a, b, 0, c, ReuseA, ReuseB);
    else
        return __builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8(a, b, 0, c, ReuseA, ReuseB);
#else
    ignore(a, b);
    return c;
#endif
}

template <typename AType, typename BType, bool ReuseA = false, bool ReuseB = false>
CK_TILE_DEVICE fp16x8_t wmma_f16_16x16x128_f8(const int32x16_t& a,
                                              const int32x16_t& b,
                                              const fp16x8_t& c)
{
    static_assert((std::is_same_v<AType, fp8_t> || std::is_same_v<AType, bf8_t>) &&
                      (std::is_same_v<BType, fp8_t> || std::is_same_v<BType, bf8_t>),
                  "A and B must be fp8_t or bf8_t");
#if defined(__gfx1250_strict__)
    return __builtin_convertvector(
        wmma_f32_16x16x128_f8<AType, BType>(a, b, __builtin_convertvector(c, fp32x8_t)), fp16x8_t);
#elif defined(__gfx125__)
    if constexpr(std::is_same_v<AType, fp8_t> && std::is_same_v<BType, fp8_t>)
        return __builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8(a, b, 0, c, ReuseA, ReuseB);
    else if constexpr(std::is_same_v<AType, fp8_t> && std::is_same_v<BType, bf8_t>)
        return __builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8(a, b, 0, c, ReuseA, ReuseB);
    else if constexpr(std::is_same_v<AType, bf8_t> && std::is_same_v<BType, fp8_t>)
        return __builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8(a, b, 0, c, ReuseA, ReuseB);
    else
        return __builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8(a, b, 0, c, ReuseA, ReuseB);
#else
    ignore(a, b);
    return c;
#endif
}

} // namespace ck_tile::core::arch::mma
