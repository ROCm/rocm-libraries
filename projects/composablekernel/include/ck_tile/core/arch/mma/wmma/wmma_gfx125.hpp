// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_data_format.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/wmma/wmma_traits.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/int8.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_params.hpp"

namespace ck_tile::core::arch::mma {

// NOTE: At this point forward, we are specializing amdgcn_mma for each target id as needed.
// This is because some built-ins are only available on certain target ids.
// We can also do things, such add some padding specializations for when we need to use
// smaller values of K that aren't directly supported by the built-ins.
// For flexibility, it is recommended that for each backend wrapper it supports at least
// one packed register for each input to be able to process smaller K values by padding.

/**
 * @defgroup dense_wmma_gfx125 Dense WMMA for GFX125
 * @brief Dense specializations of @ref amdgcn_mma for GFX125 family.
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
//               |A B C DataTypes       |MNK          |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 16u, 16u, 4u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                    |WS  |AParams |BPar |CPar |
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 16u, 16u, 4u, 32u, 2, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x4_f32";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f32_16x16x4_f32(false, // A_neg
                                                      aVec,
                                                      false, // B_neg
                                                      bVec,
                                                      0, // C_mod
                                                      cVec,
                                                      P::reuse_a,   // matrix_a_reuse
                                                      P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                     |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x32_bf16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f32_16x16x32_bf16(false, // A_neg
                                                        aVec,
                                                        false, // B_neg
                                                        bVec,
                                                        0, // C_mod
                                                        cVec,
                                                        P::reuse_a,   // matrix_a_reuse
                                                        P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<bf16_t, bf16_t, bf16_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                     |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf16_t, bf16_t, bf16_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_bf16_16x16x32_bf16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_bf16_16x16x32_bf16(false, // A_neg
                                                         aVec,
                                                         false, // B_neg
                                                         bVec,
                                                         0, // C_mod
                                                         cVec,
                                                         P::reuse_a,   // matrix_a_reuse
                                                         P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 64u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(bit_cast<int32x8_t>(aVec),
                                                           bit_cast<int32x8_t>(bVec),
                                                           0, // C_mod
                                                           cVec,
                                                           P::reuse_a,   // matrix_a_reuse
                                                           P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 64u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8(bit_cast<int32x8_t>(aVec),
                                                           bit_cast<int32x8_t>(bVec),
                                                           0, // C_mod
                                                           cVec,
                                                           P::reuse_a,   // matrix_a_reuse
                                                           P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 64u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8(bit_cast<int32x8_t>(aVec),
                                                           bit_cast<int32x8_t>(bVec),
                                                           0, // C_mod
                                                           cVec,
                                                           P::reuse_a,   // matrix_a_reuse
                                                           P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 64u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8(bit_cast<int32x8_t>(aVec),
                                                           bit_cast<int32x8_t>(bVec),
                                                           0, // C_mod
                                                           cVec,
                                                           P::reuse_a,   // matrix_a_reuse
                                                           P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, fp8_t, fp16_t, 16u, 16u, 64u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp8_t, fp8_t, fp16_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8(bit_cast<int32x8_t>(aVec),
                                                           bit_cast<int32x8_t>(bVec),
                                                           0, // C_mod
                                                           cVec,
                                                           P::reuse_a,   // matrix_a_reuse
                                                           P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<fp8_t, bf8_t, fp16_t, 16u, 16u, 64u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp8_t, bf8_t, fp16_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8(bit_cast<int32x8_t>(aVec),
                                                           bit_cast<int32x8_t>(bVec),
                                                           0, // C_mod
                                                           cVec,
                                                           P::reuse_a,   // matrix_a_reuse
                                                           P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, fp8_t, fp16_t, 16u, 16u, 64u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf8_t, fp8_t, fp16_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8(bit_cast<int32x8_t>(aVec),
                                                           bit_cast<int32x8_t>(bVec),
                                                           0, // C_mod
                                                           cVec,
                                                           P::reuse_a,   // matrix_a_reuse
                                                           P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK           |
struct amdgcn_mma<bf8_t, bf8_t, fp16_t, 16u, 16u, 64u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                   |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf8_t, bf8_t, fp16_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8(bit_cast<int32x8_t>(aVec),
                                                           bit_cast<int32x8_t>(bVec),
                                                           0, // C_mod
                                                           cVec,
                                                           P::reuse_a,   // matrix_a_reuse
                                                           P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes        |MNK           |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 64u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                      |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_i32_16x16x64_iu8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_i32_16x16x64_iu8(true, // A signedness
                                                       bit_cast<int32x8_t>(aVec),
                                                       true, // B signedness
                                                       bit_cast<int32x8_t>(bVec),
                                                       cVec,
                                                       P::reuse_a,   // matrix_a_reuse
                                                       P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK            |
struct amdgcn_mma<fp8_t, fp8_t, fp16_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                    |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp8_t, fp8_t, fp16_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8(bit_cast<int32x16_t>(aVec),
                                                            bit_cast<int32x16_t>(bVec),
                                                            0, // C_mod
                                                            cVec,
                                                            P::reuse_a,   // matrix_a_reuse
                                                            P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK            |
struct amdgcn_mma<fp8_t, bf8_t, fp16_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                    |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp8_t, bf8_t, fp16_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8(bit_cast<int32x16_t>(aVec),
                                                            bit_cast<int32x16_t>(bVec),
                                                            0, // C_mod
                                                            cVec,
                                                            P::reuse_a,   // matrix_a_reuse
                                                            P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK            |
struct amdgcn_mma<bf8_t, fp8_t, fp16_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                    |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf8_t, fp8_t, fp16_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8(bit_cast<int32x16_t>(aVec),
                                                            bit_cast<int32x16_t>(bVec),
                                                            0, // C_mod
                                                            cVec,
                                                            P::reuse_a,   // matrix_a_reuse
                                                            P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK            |
struct amdgcn_mma<bf8_t, bf8_t, fp16_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                    |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf8_t, bf8_t, fp16_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8(bit_cast<int32x16_t>(aVec),
                                                            bit_cast<int32x16_t>(bVec),
                                                            0, // C_mod
                                                            cVec,
                                                            P::reuse_a,   // matrix_a_reuse
                                                            P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes                 |MNK            |
struct amdgcn_mma<pk_fp6x16_t, pk_fp6x16_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                                |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<pk_fp6x16_t, pk_fp6x16_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(PackedDataTypeToFlag_v<pk_fp6x16_t>,
                                                           to_type<int32x16_t>(aVec),
                                                           PackedDataTypeToFlag_v<pk_fp6x16_t>,
                                                           to_type<int32x16_t>(bVec),
                                                           0, // C_mod
                                                           cVec)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes           |MNK            |
struct amdgcn_mma<pk_fp4_t, pk_fp4_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                          |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<pk_fp4_t, pk_fp4_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(PackedDataTypeToFlag_v<pk_fp4_t>,
                                                           to_type<int32x16_t>(aVec),
                                                           PackedDataTypeToFlag_v<pk_fp4_t>,
                                                           to_type<int32x16_t>(bVec),
                                                           0, // C_mod
                                                           cVec)};
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK            |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                    |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(bit_cast<int32x16_t>(aVec),
                                                            bit_cast<int32x16_t>(bVec),
                                                            0, // C_mod
                                                            cVec,
                                                            P::reuse_a,   // matrix_a_reuse
                                                            P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK            |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                    |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8(bit_cast<int32x16_t>(aVec),
                                                            bit_cast<int32x16_t>(bVec),
                                                            0, // C_mod
                                                            cVec,
                                                            P::reuse_a,   // matrix_a_reuse
                                                            P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK            |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                    |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8(bit_cast<int32x16_t>(aVec),
                                                            bit_cast<int32x16_t>(bVec),
                                                            0, // C_mod
                                                            cVec,
                                                            P::reuse_a,   // matrix_a_reuse
                                                            P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes     |MNK            |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                    |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8(bit_cast<int32x16_t>(aVec),
                                                            bit_cast<int32x16_t>(bVec),
                                                            0, // C_mod
                                                            cVec,
                                                            P::reuse_a,   // matrix_a_reuse
                                                            P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                     |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x32_f16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f32_16x16x32_f16(false, // A_neg
                                                       aVec,
                                                       false, // B_neg
                                                       bVec,
                                                       0, // C_mod
                                                       cVec,
                                                       P::reuse_a,   // matrix_a_reuse
                                                       P::reuse_b)}; // matrix_b_reuse
    }
};

template <typename CompilerTarget>
// clang-format off
//               |A B C DataTypes       |MNK           |
struct amdgcn_mma<fp16_t, fp16_t, fp16_t, 16u, 16u, 32u, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_gfx1250_t<CompilerTarget>>
//                                                     |WS  |AParams  |BPar |CPar |
: amdgcn_mma_base<fp16_t, fp16_t, fp16_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f16_16x16x32_f16";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_wmma_f16_16x16x32_f16(false, // A_neg
                                                       aVec,
                                                       false, // B_neg
                                                       bVec,
                                                       0, // C_mod
                                                       cVec,
                                                       P::reuse_a,   // matrix_a_reuse
                                                       P::reuse_b)}; // matrix_b_reuse
    }
};

/** @} */ // dense_wmma_gfx125

} // namespace ck_tile::core::arch::mma
