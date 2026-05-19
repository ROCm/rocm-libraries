// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/wmma/wmma_traits.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

namespace ck_tile::core::arch::mma {

// /**
//  * @struct amdgcn_mma
//  * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp32_t MMA operation on GFX12
//  * architecture.
//  * @tparam CtrlFlags Control flags for the WMMA operation
//  * @tparam CompilerTarget Current compiler target
//  */
// // TODO: c++20 template <CtrlFlagsSparseWmmaI CtrlFlags, amdgcn_target CompilerTarget>
// // TODO: c++20 requires
// template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes      | MNK + WaveSize    |AParams  |BPar |CPar |
// struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
//     {
//         return {__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(aVec, bVec, cVec, idx)};
//     }
// };

// /**
//  * @struct amdgcn_mma
//  * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX12
//  * architecture.
//  * @tparam CtrlFlags Control flags for the WMMA operation
//  * @tparam CompilerTarget Current compiler target
//  */
// // TODO: c++20 template <CtrlFlagsGfx12I CtrlFlags, amdgcn_target CompilerTarget>
// // TODO: c++20 requires
// template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes      | MNK + WaveSize    |AParams  |BPar |CPar |
// struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
//     {
//         return {__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32(aVec, bVec, cVec, idx)};
//     }
// };

// /**
//  * @struct amdgcn_mma
//  * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp16_t MMA operation on GFX12
//  * architecture.
//  * @tparam CtrlFlags Control flags for the WMMA operation
//  * @tparam CompilerTarget Current compiler target
//  */
// // TODO: c++20 template <CtrlFlagsGfx12I CtrlFlags, amdgcn_target CompilerTarget>
// // TODO: c++20 requires
// template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes      | MNK + WaveSize    |AParams  |BPar |CPar |
// struct amdgcn_mma<fp16_t, fp16_t, fp16_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<fp16_t, fp16_t, fp16_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f16_16x16x32_f16_w32";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
//     {
//         return {__builtin_amdgcn_swmmac_f16_16x16x32_f16_w32(aVec, bVec, cVec, idx)};
//     }
// };

// /**
//  * @struct amdgcn_mma
//  * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, bf16_t MMA operation on GFX12
//  * architecture.
//  * @tparam CtrlFlags Control flags for the WMMA operation
//  * @tparam CompilerTarget Current compiler target
//  */
// // TODO: c++20 template <CtrlFlagsGfx12I CtrlFlags, amdgcn_target CompilerTarget>
// // TODO: c++20 requires
// template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes      | MNK + WaveSize    |AParams  |BPar |CPar |
// struct amdgcn_mma<bf16_t, bf16_t, bf16_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<bf16_t, bf16_t, bf16_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w32";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
//     {
//         return {__builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w32(aVec, bVec, cVec, idx)};
//     }
// };

// /**
//  * @struct amdgcn_mma
//  * @brief Specialization of amdgcn_mma for int8_t, int8_t, int32_t MMA operation on GFX12
//  * architecture.
//  * @tparam CtrlFlags Control flags for the WMMA operation
//  * @tparam CompilerTarget Current compiler target
//  */
// // TODO: c++20 template <CtrlFlagsGfx12I CtrlFlags, amdgcn_target CompilerTarget>
// // TODO: c++20 requires
// template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes       | MNK + WaveSize    |AParams  |BPar |CPar |
// struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
//     {
//         return {__builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(true, // A signedness
//                                                              aVec,
//                                                              true, // B signedness
//                                                              bVec,
//                                                              cVec,
//                                                              idx,
//                                                              CtrlFlags::Clamp)};
//     }
// };

// /**
//  * @struct amdgcn_mma
//  * @brief Specialization of amdgcn_mma for fp8_t, fp8_t, fp32_t MMA operation on GFX12
//  * architecture.
//  * @tparam CtrlFlags Control flags for the WMMA operation
//  * @tparam CompilerTarget Current compiler target
//  */
// // TODO: c++20 template <CtrlFlagsGfx12I CtrlFlags, amdgcn_target CompilerTarget>
// // TODO: c++20 requires
// template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes    | MNK + WaveSize    |AParams  |BPar |CPar |
// struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
//     {
//         return {__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(aVec, bVec, cVec, idx)};
//     }
// };

// /**
//  * @struct amdgcn_mma
//  * @brief Specialization of amdgcn_mma for fp8_t, bf8_t, fp32_t MMA operation on GFX12
//  * architecture.
//  * @tparam CtrlFlags Control flags for the WMMA operation
//  * @tparam CompilerTarget Current compiler target
//  */
// // TODO: c++20 template <CtrlFlagsGfx12I CtrlFlags, amdgcn_target CompilerTarget>
// // TODO: c++20 requires
// template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes    | MNK + WaveSize    |AParams  |BPar |CPar |
// struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
//     {
//         return {__builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32(aVec, bVec, cVec, idx)};
//     }
// };

// /**
//  * @struct amdgcn_mma
//  * @brief Specialization of amdgcn_mma for bf8_t, fp8_t, fp32_t MMA operation on GFX12
//  * architecture.
//  * @tparam CtrlFlags Control flags for the WMMA operation
//  * @tparam CompilerTarget Current compiler target
//  */
// // TODO: c++20 template <CtrlFlagsGfx12I CtrlFlags, amdgcn_target CompilerTarget>
// // TODO: c++20 requires
// template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes    | MNK + WaveSize    |AParams  |BPar |CPar |
// struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
//     {
//         return {__builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32(aVec, bVec, cVec, idx)};
//     }
// };

// /**
//  * @struct amdgcn_mma
//  * @brief Specialization of amdgcn_mma for bf8_t, bf8_t, fp32_t MMA operation on GFX12
//  * architecture.
//  * @tparam CtrlFlags Control flags for the WMMA operation
//  * @tparam CompilerTarget Current compiler target
//  */
// // TODO: c++20 template <CtrlFlagsGfx12I CtrlFlags, amdgcn_target CompilerTarget>
// // TODO: c++20 requires
// template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes    | MNK + WaveSize    |AParams  |BPar |CPar |
// struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
//     {
//         return {__builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32(aVec, bVec, cVec, idx)};
//     }
// };

// // TODO: c++20 template <CtrlFlagsSparseWmmaI CtrlFlags, amdgcn_target CompilerTarget>
// // TODO: c++20 requires
// template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes             | MNK + WaveSize    |AParams  |BPar |CPar |
// struct amdgcn_mma<pk_int4_t, pk_int4_t, int32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<pk_int4_t, pk_int4_t, int32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32";

//     CK_TILE_DEVICE static auto
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx) -> CVecType
//     {
//         return {__builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32(true, // A signedness
//                                                              bit_cast<int32_t>(aVec),
//                                                              true, // B signedness
//                                                              bit_cast<int32x2_t>(bVec),
//                                                              cVec,
//                                                              idx,
//                                                              CtrlFlags::Clamp)};
//     }
// };

// // TODO: c++20 template <CtrlFlagsSparseWmmaI CtrlFlags, amdgcn_target CompilerTarget>
// // TODO: c++20 requires
// template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes             | MNK + WaveSize    |AParams  |BPar |CPar |
// struct amdgcn_mma<pk_int4_t, pk_int4_t, int32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<pk_int4_t, pk_int4_t, int32_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32";

//     CK_TILE_DEVICE static auto
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx) -> CVecType
//     {
//         return {__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(true, // A signedness
//                                                              bit_cast<int32x2_t>(aVec),
//                                                              true, // B signedness
//                                                              bit_cast<int32x4_t>(bVec),
//                                                              cVec,
//                                                              idx,
//                                                              CtrlFlags::Clamp)};
//     }
// };

// gfx1250

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, fp32_t MMA operation on GFX1250
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx1250I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f32_16x16x64_bf16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_f32_16x16x64_bf16(0, aVec, 0, bVec, cVec, idx, 0, 0)};
    }
};


/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf16_t, bf16_t, bf16_t MMA operation on GFX1250
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx1250I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, bf16_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, bf16_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_bf16_16x16x64_bf16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_bf16_16x16x64_bf16(0, aVec, 0, bVec, cVec, idx, 0, 0)};
    }
};


/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp8_t, fp8_t, fp32_t MMA operation on GFX1250
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx1250I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f32_16x16x128_fp8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_f32_16x16x128_fp8_fp8(bit_cast<int32x8_t>(aVec), bit_cast<int32x16_t>(bVec), cVec, idx, 0, 0)};
    }
};


/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp8_t, bf8_t, fp32_t MMA operation on GFX1250
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx1250I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f32_16x16x128_fp8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_f32_16x16x128_fp8_bf8(bit_cast<int32x8_t>(aVec), bit_cast<int32x16_t>(bVec), cVec, idx, 0, 0)};
    }
};


/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf8_t, fp8_t, fp32_t MMA operation on GFX1250
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx1250I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f32_16x16x128_bf8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_f32_16x16x128_bf8_fp8(bit_cast<int32x8_t>(aVec), bit_cast<int32x16_t>(bVec), cVec, idx, 0, 0)};
    }
};


/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf8_t, bf8_t, fp32_t MMA operation on GFX1250
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx1250I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f32_16x16x128_bf8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_f32_16x16x128_bf8_bf8(bit_cast<int32x8_t>(aVec), bit_cast<int32x16_t>(bVec), cVec, idx, 0, 0)};
    }
};


/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp8_t, fp8_t, fp16_t MMA operation on GFX1250
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx1250I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, fp8_t, fp16_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, fp8_t, fp16_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f16_16x16x128_fp8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_f16_16x16x128_fp8_fp8(bit_cast<int32x8_t>(aVec), bit_cast<int32x16_t>(bVec), cVec, idx, 0, 0)};
    }
};


/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp8_t, bf8_t, fp16_t MMA operation on GFX1250
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx1250I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, bf8_t, fp16_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, bf8_t, fp16_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f16_16x16x128_fp8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_f16_16x16x128_fp8_bf8(bit_cast<int32x8_t>(aVec), bit_cast<int32x16_t>(bVec), cVec, idx, 0, 0)};
    }
};


/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf8_t, fp8_t, fp16_t MMA operation on GFX1250
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx1250I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, fp8_t, fp16_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, fp8_t, fp16_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f16_16x16x128_bf8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_f16_16x16x128_bf8_fp8(bit_cast<int32x8_t>(aVec), bit_cast<int32x16_t>(bVec), cVec, idx, 0, 0)};
    }
};


/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for bf8_t, bf8_t, fp16_t MMA operation on GFX1250
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx1250I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, bf8_t, fp16_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, bf8_t, fp16_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f16_16x16x128_bf8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_f16_16x16x128_bf8_bf8(bit_cast<int32x8_t>(aVec), bit_cast<int32x16_t>(bVec), cVec, idx, 0, 0)};
    }
};


/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for int8_t, int8_t, int32_t MMA operation on GFX1250
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx1250I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_i32_16x16x128_iu8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_i32_16x16x128_iu8(true, // A signedness
                                                          aVec,
                                                          true, // B signedness
                                                          bVec,
                                                          cVec,
                                                          idx,
                                                          CtrlFlags::Clamp,
                                                          0)};
    }
};


/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp32_t MMA operation on GFX1250
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx1250I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f32_16x16x64_f16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_f32_16x16x64_f16(0, aVec, 0, bVec, cVec, idx, 0, 0)};
    }
};


/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp16_t MMA operation on GFX1250
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx1250I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp16_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::SPARSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp16_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::SPARSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_swmmac_f16_16x16x64_f16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int32_t idx)
    {
        return {__builtin_amdgcn_swmmac_f16_16x16x64_f16(0, aVec, 0, bVec, cVec, idx, 0, 0)};
    }
};

} // namespace ck_tile::core::arch::mma
