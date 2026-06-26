// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mfma/mfma_traits.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/scale/scale_traits.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_params.hpp"

namespace ck_tile::core::arch::mma {

// TODO: c++20 template <amdgcn_target CompilerTarget>
// TODO: c++20 requires
// clang-format off
#define MMA_SCALE_ARG_F8(vec) bit_cast<int32x8_t>(vec)
#define MMA_SCALE_ARG_F6(vec) int32x8_t{vec.data[0], vec.data[1], vec.data[2], vec.data[3], vec.data[4], vec.data[5], 0, 0}
#define MMA_SCALE_ARG_F4(vec) int32x8_t{bit_cast<int32x4_t>(vec)[0], bit_cast<int32x4_t>(vec)[1], bit_cast<int32x4_t>(vec)[2], bit_cast<int32x4_t>(vec)[3], 0, 0, 0, 0}

#define DEFINE_MMA_SCALE_GFX950_16(AType, BType, EXPAND_A, EXPAND_B)            \
template <typename CompilerTarget>                                              \
struct amdgcn_mma<AType, BType, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>> \
: amdgcn_mma_base<AType, BType, fp32_t, 16u, 16u, 128u, 64u, 32, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SCALE> \
{                                                                               \
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4"; \
    template <typename... Params>                                               \
    CK_TILE_DEVICE static CVecType                                              \
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B) \
    {                                                                           \
        using P = WarpGemmParamsParser<Params...>;                              \
        return {__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(               \
            EXPAND_A(aVec),                                                     \
            EXPAND_B(bVec),                                                     \
            cVec,                                                               \
            scale::detail::ScaleDataTypeToFlag_v<AType>,                        \
            scale::detail::ScaleDataTypeToFlag_v<BType>,                        \
            P::op_sel_a, scale_A,                                               \
            P::op_sel_b, scale_B)};                                             \
    }                                                                           \
};

#define DEFINE_MMA_SCALE_GFX950_32(AType, BType, EXPAND_A, EXPAND_B)            \
template <typename CompilerTarget>                                              \
struct amdgcn_mma<AType, BType, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>> \
: amdgcn_mma_base<AType, BType, fp32_t, 32u, 32u, 64u, 64u, 32, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SCALE> \
{                                                                               \
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4"; \
    template <typename... Params>                                               \
    CK_TILE_DEVICE static CVecType                                              \
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B) \
    {                                                                           \
        using P = WarpGemmParamsParser<Params...>;                              \
        return {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(                \
            EXPAND_A(aVec),                                                     \
            EXPAND_B(bVec),                                                     \
            cVec,                                                               \
            scale::detail::ScaleDataTypeToFlag_v<AType>,                        \
            scale::detail::ScaleDataTypeToFlag_v<BType>,                        \
            P::op_sel_a, scale_A,                                               \
            P::op_sel_b, scale_B)};                                             \
    }                                                                           \
};

// 25 intrinsics for __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4
DEFINE_MMA_SCALE_GFX950_16(fp8_t,       fp8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_16(fp8_t,       bf8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_16(bf8_t,       fp8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_16(bf8_t,       bf8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_16(fp8_t,       pk_fp6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_16(fp8_t,       pk_bf6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_16(fp8_t,       pk_fp4_t,    MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F4)
DEFINE_MMA_SCALE_GFX950_16(bf8_t,       pk_fp6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_16(bf8_t,       pk_bf6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_16(bf8_t,       pk_fp4_t,    MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F4)
DEFINE_MMA_SCALE_GFX950_16(pk_fp6x16_t, fp8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_16(pk_fp6x16_t, bf8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_16(pk_fp6x16_t, pk_fp6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_16(pk_fp6x16_t, pk_bf6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_16(pk_fp6x16_t, pk_fp4_t,    MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F4)
DEFINE_MMA_SCALE_GFX950_16(pk_bf6x16_t, fp8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_16(pk_bf6x16_t, bf8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_16(pk_bf6x16_t, pk_fp6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_16(pk_bf6x16_t, pk_bf6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_16(pk_bf6x16_t, pk_fp4_t,    MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F4)
DEFINE_MMA_SCALE_GFX950_16(pk_fp4_t,    fp8_t,       MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_16(pk_fp4_t,    bf8_t,       MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_16(pk_fp4_t,    pk_fp6x16_t, MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_16(pk_fp4_t,    pk_bf6x16_t, MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_16(pk_fp4_t,    pk_fp4_t,    MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F4)

// 25 intrinsics for __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4
DEFINE_MMA_SCALE_GFX950_32(fp8_t,       fp8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_32(fp8_t,       bf8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_32(bf8_t,       fp8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_32(bf8_t,       bf8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_32(fp8_t,       pk_fp6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_32(fp8_t,       pk_bf6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_32(fp8_t,       pk_fp4_t,    MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F4)
DEFINE_MMA_SCALE_GFX950_32(bf8_t,       pk_fp6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_32(bf8_t,       pk_bf6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_32(bf8_t,       pk_fp4_t,    MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F4)
DEFINE_MMA_SCALE_GFX950_32(pk_fp6x16_t, fp8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_32(pk_fp6x16_t, bf8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_32(pk_fp6x16_t, pk_fp6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_32(pk_fp6x16_t, pk_bf6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_32(pk_fp6x16_t, pk_fp4_t,    MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F4)
DEFINE_MMA_SCALE_GFX950_32(pk_bf6x16_t, fp8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_32(pk_bf6x16_t, bf8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_32(pk_bf6x16_t, pk_fp6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_32(pk_bf6x16_t, pk_bf6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_32(pk_bf6x16_t, pk_fp4_t,    MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F4)
DEFINE_MMA_SCALE_GFX950_32(pk_fp4_t,    fp8_t,       MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_32(pk_fp4_t,    bf8_t,       MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F8)
DEFINE_MMA_SCALE_GFX950_32(pk_fp4_t,    pk_fp6x16_t, MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_32(pk_fp4_t,    pk_bf6x16_t, MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F6)
DEFINE_MMA_SCALE_GFX950_32(pk_fp4_t,    pk_fp4_t,    MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F4)

// clang-format on

} // namespace ck_tile::core::arch::mma
