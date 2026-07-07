// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/wmma/wmma_traits.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/int8.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_int4.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_params.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @defgroup sparse_wmma_gfx12 Sparse WMMA for GFX12
 * @brief Sparse specializations of @ref amdgcn_mma for GFX12 family.
 *
 * Template parameters A/B/C denote input/output types,
 * M/N/K are the fragment (MmaTile) sizes,
 * and `enable_if_target_*` restricts the specialization to specific GPU targets.
 *
 * @tparam CompilerTarget Current compiler target.
 *
 * @sa amdgcn_mma_base for base template parameter documentation.
 * @{
 */

// TODO: c++20 template <amdgcn_target CompilerTarget>
// TODO: c++20 requires

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
//                                                     |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(
            bit_cast<llvm_fp16x8_t>(aVec), bit_cast<llvm_fp16x16_t>(bVec), cVec, idx)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
//                                                     |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32(
            bit_cast<int16x8_t>(aVec), bit_cast<int16x16_t>(bVec), cVec, idx)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<fp16_t, fp16_t, fp16_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
//                                                     |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp16_t, fp16_t, fp16_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f16_16x16x32_f16_w32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return bit_cast<CVecType>(
            __builtin_amdgcn_swmmac_f16_16x16x32_f16_w32(bit_cast<llvm_fp16x8_t>(aVec),
                                                         bit_cast<llvm_fp16x16_t>(bVec),
                                                         bit_cast<llvm_fp16x8_t>(cVec),
                                                         idx));
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<bf16_t, bf16_t, bf16_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
//                                                     |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf16_t, bf16_t, bf16_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return bit_cast<CVecType>(__builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w32(
            bit_cast<int16x8_t>(aVec), bit_cast<int16x16_t>(bVec), bit_cast<int16x8_t>(cVec), idx));
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK           |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
//                                                      |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(true, // A signedness
                                                             aVec,
                                                             true, // B signedness
                                                             bVec,
                                                             cVec,
                                                             idx,
                                                             P::clamp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(
            bit_cast<int32x2_t>(aVec), bit_cast<int32x4_t>(bVec), cVec, idx)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32(
            bit_cast<int32x2_t>(aVec), bit_cast<int32x4_t>(bVec), cVec, idx)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32(
            bit_cast<int32x2_t>(aVec), bit_cast<int32x4_t>(bVec), cVec, idx)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32(
            bit_cast<int32x2_t>(aVec), bit_cast<int32x4_t>(bVec), cVec, idx)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes              |MNK           |
struct amdgcn_mma<pk_int4_t, pk_int4_t, int32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
//                                                            |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<pk_int4_t, pk_int4_t, int32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32(true, // A signedness
                                                             bit_cast<int32_t>(aVec),
                                                             true, // B signedness
                                                             bit_cast<int32x2_t>(bVec),
                                                             cVec,
                                                             idx,
                                                             P::clamp)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes              |MNK           |
struct amdgcn_mma<pk_int4_t, pk_int4_t, int32_t, 16u, 16u, 64u, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
//                                                            |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<pk_int4_t, pk_int4_t, int32_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(true, // A signedness
                                                             bit_cast<int32x2_t>(aVec),
                                                             true, // B signedness
                                                             bit_cast<int32x4_t>(bVec),
                                                             cVec,
                                                             idx,
                                                             P::clamp)};
    }
};

/** @} */ // sparse_wmma_gfx12

} // namespace ck_tile::core::arch::mma
