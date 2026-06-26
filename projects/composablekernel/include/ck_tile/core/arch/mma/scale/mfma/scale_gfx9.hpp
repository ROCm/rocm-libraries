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

#define DEFINE_MMA_SCALE_GFX950(ATYPE, BTYPE, EXPAND_A, EXPAND_B, SSMN, SSK, LAY1, LAY2)            \
template <typename CompilerTarget>                                              \
struct amdgcn_mma<ATYPE, BTYPE, fp32_t, SSMN, SSMN, SSK, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>> \
: amdgcn_mma_base<ATYPE, BTYPE, fp32_t, SSMN, SSMN, SSK, 64u, 32, 1, 1, 1, 1, LAY1, LAY2, MfmaOp, MmaOpFamily::SCALE> \
{                                                                               \
    static constexpr const char* instruction_name = "__builtin_amdgcn_mfma_scale_f32_SSMNxSSMNxSSK_f8f6f4"; \
    template <typename... Params>                                               \
    CK_TILE_DEVICE static CVecType                                              \
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B) \
    {                                                                           \
        using P = WarpGemmParamsParser<Params...>;                              \
        return {__builtin_amdgcn_mfma_scale_f32_SSMNxSSMNxSSK_f8f6f4(           \
            EXPAND_A(aVec),                                                     \
            EXPAND_B(bVec),                                                     \
            cVec,                                                               \
            scale::detail::ScaleDataTypeToFlag_v<ATYPE>,                        \
            scale::detail::ScaleDataTypeToFlag_v<BTYPE>,                        \
            P::op_sel_a, scale_A,                                               \
            P::op_sel_b, scale_B)};                                             \
    }                                                                           \
};

// 25 intrinsics for __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4
DEFINE_MMA_SCALE_GFX950(fp8_t,       fp8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(fp8_t,       bf8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(bf8_t,       fp8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(bf8_t,       bf8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(fp8_t,       pk_fp6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(fp8_t,       pk_bf6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(fp8_t,       pk_fp4_t,    MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F4, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(bf8_t,       pk_fp6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(bf8_t,       pk_bf6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(bf8_t,       pk_fp4_t,    MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F4, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(pk_fp6x16_t, fp8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(pk_fp6x16_t, bf8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(pk_fp6x16_t, pk_fp6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(pk_fp6x16_t, pk_bf6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(pk_fp6x16_t, pk_fp4_t,    MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F4, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(pk_bf6x16_t, fp8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(pk_bf6x16_t, bf8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(pk_bf6x16_t, pk_fp6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(pk_bf6x16_t, pk_bf6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(pk_bf6x16_t, pk_fp4_t,    MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F4, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(pk_fp4_t,    fp8_t,       MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F8, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(pk_fp4_t,    bf8_t,       MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F8, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(pk_fp4_t,    pk_fp6x16_t, MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F6, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(pk_fp4_t,    pk_bf6x16_t, MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F6, 16, 128, 4, 1)
DEFINE_MMA_SCALE_GFX950(pk_fp4_t,    pk_fp4_t,    MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F4, 16, 128, 4, 1)

// 25 intrinsics for __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4
DEFINE_MMA_SCALE_GFX950(fp8_t,       fp8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(fp8_t,       bf8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(bf8_t,       fp8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(bf8_t,       bf8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(fp8_t,       pk_fp6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(fp8_t,       pk_bf6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(fp8_t,       pk_fp4_t,    MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F4, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(bf8_t,       pk_fp6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(bf8_t,       pk_bf6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(bf8_t,       pk_fp4_t,    MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F4, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(pk_fp6x16_t, fp8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(pk_fp6x16_t, bf8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(pk_fp6x16_t, pk_fp6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(pk_fp6x16_t, pk_bf6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(pk_fp6x16_t, pk_fp4_t,    MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F4, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(pk_bf6x16_t, fp8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(pk_bf6x16_t, bf8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(pk_bf6x16_t, pk_fp6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(pk_bf6x16_t, pk_bf6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(pk_bf6x16_t, pk_fp4_t,    MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F4, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(pk_fp4_t,    fp8_t,       MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F8, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(pk_fp4_t,    bf8_t,       MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F8, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(pk_fp4_t,    pk_fp6x16_t, MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F6, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(pk_fp4_t,    pk_bf6x16_t, MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F6, 32, 64, 16, 4)
DEFINE_MMA_SCALE_GFX950(pk_fp4_t,    pk_fp4_t,    MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F4, 32, 64, 16, 4)
// clang-format on
} // namespace ck_tile::core::arch::mma
