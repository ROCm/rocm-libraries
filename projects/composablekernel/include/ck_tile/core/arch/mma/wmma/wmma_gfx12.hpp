// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "wmma_traits.hpp"

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/int8.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

namespace ck_tile::core::arch::mma {

// NOTE: At this point forward, we are specializing amdgcn_mma for each target id as needed.
// This is because some built-ins are only available on certain target ids.
// We can also do things, such add some padding specializations for when we need to use
// smaller values of K that aren't directly supported by the built-ins.
// For flexibility, it is recommended that for each backend wrapper it supports at least
// one packed register for each input to be able to process smaller K values by padding.

// /**
//  * @struct amdgcn_mma
//  * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp32_t MMA operation on GFX12
//  * architecture.
//  * @tparam CtrlFlags Control flags for the WMMA operation
//  * @tparam CompilerTarget Current compiler target
//  */
// // TODO: c++20 template <CtrlFlagsGfx12I CtrlFlags, amdgcn_target CompilerTarget>
// // TODO: c++20 requires
// template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
// struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name =
//         "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
//     {
//         return {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(aVec, bVec, cVec)};
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
// //               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
// struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name =
//         "__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
//     {
//         return {__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(aVec, bVec, cVec)};
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
// //               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
// struct amdgcn_mma<fp16_t, fp16_t, fp16_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<fp16_t, fp16_t, fp16_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name =
//         "__builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
//     {
//         return {__builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(aVec, bVec, cVec)};
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
// //               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
// struct amdgcn_mma<bf16_t, bf16_t, bf16_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<bf16_t, bf16_t, bf16_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name =
//         "__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
//     {
//         return {__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12(aVec, bVec, cVec)};
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
// //               | A B C DataTypes       | MNK + WaveSize    |AParams |BPar |CPar |
// struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name =
//         "__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
//     {
//         return {__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, // A signedness
//                                                                  bit_cast<int32x2_t>(aVec),
//                                                                  true, // B signedness
//                                                                  bit_cast<int32x2_t>(bVec),
//                                                                  cVec,
//                                                                  CtrlFlags::Clamp)};
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
// //               | A B C DataTypes    | MNK + WaveSize    |AParams |BPar |CPar |
// struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name =
//         "__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
//     {
//         return {__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
//             bit_cast<int32x2_t>(aVec), bit_cast<int32x2_t>(bVec), cVec)};
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
// //               | A B C DataTypes    | MNK + WaveSize    |AParams |BPar |CPar |
// struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name =
//         "__builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
//     {
//         return {__builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(
//             bit_cast<int32x2_t>(aVec), bit_cast<int32x2_t>(bVec), cVec)};
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
// //               | A B C DataTypes    | MNK + WaveSize    |AParams |BPar |CPar |
// struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name =
//         "__builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
//     {
//         return {__builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(
//             bit_cast<int32x2_t>(aVec), bit_cast<int32x2_t>(bVec), cVec)};
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
// //               | A B C DataTypes    | MNK + WaveSize    |AParams |BPar |CPar |
// struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name =
//         "__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
//     {
//         return {__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(
//             bit_cast<int32x2_t>(aVec), bit_cast<int32x2_t>(bVec), cVec)};
//     }
// };

// /**
//  * @struct amdgcn_mma
//  * @brief Specialization of amdgcn_mma for pk_int4_t, pk_int4_t, int32_t MMA operation on GFX12
//  * architecture.
//  * @tparam CtrlFlags Control flags for the WMMA operation
//  * @tparam CompilerTarget Current compiler target
//  */
// // TODO: c++20 template <CtrlFlagsGfx12I CtrlFlags, amdgcn_target CompilerTarget>
// // TODO: c++20 requires
// template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes             | MNK + WaveSize    |AParams |BPar |CPar |
// struct amdgcn_mma<pk_int4_t, pk_int4_t, int32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<pk_int4_t, pk_int4_t, int32_t, 16u, 16u, 16u, 32u, 8, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name =
//         "__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12";

//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
//     {
//         return {__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, // A signedness
//                                                                  bit_cast<int32_t>(aVec),
//                                                                  true, // B signedness
//                                                                  bit_cast<int32_t>(bVec),
//                                                                  cVec,
//                                                                  CtrlFlags::Clamp)};
//     }
// };

// /**
//  * @struct amdgcn_mma
//  * @brief Specialization of amdgcn_wmma for pk_int4_t, pk_int4_t, int32_t MMA operation on GFX12
//  * architecture.
//  * @tparam CtrlFlags Control flags for the WMMA operation
//  * @tparam CompilerTarget Current compiler target
//  */
// // TODO: c++20 template <CtrlFlagsGfx12I CtrlFlags, amdgcn_target CompilerTarget>
// // TODO: c++20 requires
// template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes             | MNK + WaveSize    |AParams  |BPar |CPar |
// struct amdgcn_mma<pk_int4_t, pk_int4_t, int32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx12_t<CompilerTarget>>
// : amdgcn_mma_base<pk_int4_t, pk_int4_t, int32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// // clang-format on
// {
//     static constexpr const char* instruction_name =
//         "__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12";

//     CK_TILE_DEVICE static auto
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
//     {
//         return {__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, // A signedness
//                                                                  bit_cast<int32x2_t>(aVec),
//                                                                  true, // B signedness
//                                                                  bit_cast<int32x2_t>(bVec),
//                                                                  cVec,
//                                                                  CtrlFlags::Clamp)};
//     }
// };

// gfx1250

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp32_t, fp32_t, fp32_t MMA operation on GFX1250
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx1250I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 16u, 16u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 16u, 16u, 4u, 32u, 2, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x4_f32";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f32_16x16x4_f32(0, aVec, 0, bVec, 0, cVec, 0, 0)};
    }
};


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
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x32_bf16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f32_16x16x32_bf16(0, aVec, 0, bVec, 0, cVec, 0, 0)};
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
struct amdgcn_mma<bf16_t, bf16_t, bf16_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, bf16_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_bf16_16x16x32_bf16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_bf16_16x16x32_bf16(0, aVec, 0, bVec, 0, cVec, 0, 0)};
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
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(
            bit_cast<int32x8_t>(aVec), bit_cast<int32x8_t>(bVec), 0, cVec, 0, 0)};
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
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8(
            bit_cast<int32x8_t>(aVec), bit_cast<int32x8_t>(bVec), 0, cVec, 0, 0)};
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
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8(
            bit_cast<int32x8_t>(aVec), bit_cast<int32x8_t>(bVec), 0, cVec, 0, 0)};
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
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8(
            bit_cast<int32x8_t>(aVec), bit_cast<int32x8_t>(bVec), 0, cVec, 0, 0)};
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
struct amdgcn_mma<fp8_t, fp8_t, fp16_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, fp8_t, fp16_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8(
            bit_cast<int32x8_t>(aVec), bit_cast<int32x8_t>(bVec), 0, cVec, 0, 0)};
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
struct amdgcn_mma<fp8_t, bf8_t, fp16_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, bf8_t, fp16_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8(
            bit_cast<int32x8_t>(aVec), bit_cast<int32x8_t>(bVec), 0, cVec, 0, 0)};
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
struct amdgcn_mma<bf8_t, fp8_t, fp16_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, fp8_t, fp16_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8(
            bit_cast<int32x8_t>(aVec), bit_cast<int32x8_t>(bVec), 0, cVec, 0, 0)};
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
struct amdgcn_mma<bf8_t, bf8_t, fp16_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, bf8_t, fp16_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8(
            bit_cast<int32x8_t>(aVec), bit_cast<int32x8_t>(bVec), 0, cVec, 0, 0)};
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
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 64u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 64u, 32u, 32, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_i32_16x16x64_iu8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_i32_16x16x64_iu8(true, // A signedness
                                                       bit_cast<int32x8_t>(aVec),
                                                       true, // B signedness
                                                       bit_cast<int32x8_t>(bVec),
                                                       cVec,
                                                       CtrlFlags::Clamp,
                                                       0)};
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
struct amdgcn_mma<fp8_t, fp8_t, fp16_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, fp8_t, fp16_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8(
            bit_cast<int32x16_t>(aVec), bit_cast<int32x16_t>(bVec), 0, cVec, 0, 0)};
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
struct amdgcn_mma<fp8_t, bf8_t, fp16_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, bf8_t, fp16_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8(
            bit_cast<int32x16_t>(aVec), bit_cast<int32x16_t>(bVec), 0, cVec, 0, 0)};
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
struct amdgcn_mma<bf8_t, fp8_t, fp16_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, fp8_t, fp16_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8(
            bit_cast<int32x16_t>(aVec), bit_cast<int32x16_t>(bVec), 0, cVec, 0, 0)};
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
struct amdgcn_mma<bf8_t, bf8_t, fp16_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, bf8_t, fp16_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8(
            bit_cast<int32x16_t>(aVec), bit_cast<int32x16_t>(bVec), 0, cVec, 0, 0)};
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
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(
            0, bit_cast<int32x16_t>(aVec), 0, bit_cast<int32x16_t>(bVec), 0, cVec)};
    }
};


// /**
//  * @struct amdgcn_mma
//  * @brief Specialization of amdgcn_mma for fp6_t, fp6_t, fp32_t MMA operation on GFX1250
//  * architecture.
//  * @tparam CtrlFlags Control flags for the WMMA operation
//  * @tparam CompilerTarget Current compiler target
//  */
// // TODO: c++20 template <CtrlFlagsGfx1250I CtrlFlags, amdgcn_target CompilerTarget>
// // TODO: c++20 requires
// template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
// struct amdgcn_mma<fp6_t, fp6_t, fp32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
// : amdgcn_mma_base<fp6_t, fp6_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// // clang-format on
// {
//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
//     {
//         return {__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(
//             0, bit_cast<int32x16_t>(aVec), 0, bit_cast<int32x16_t>(bVec), 0, cVec)};
//     }
// };


// /**
//  * @struct amdgcn_mma
//  * @brief Specialization of amdgcn_mma for fp4_t, fp4_t, fp32_t MMA operation on GFX1250
//  * architecture.
//  * @tparam CtrlFlags Control flags for the WMMA operation
//  * @tparam CompilerTarget Current compiler target
//  */
// // TODO: c++20 template <CtrlFlagsGfx1250I CtrlFlags, amdgcn_target CompilerTarget>
// // TODO: c++20 requires
// template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
// struct amdgcn_mma<fp4_t, fp4_t, fp32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
// : amdgcn_mma_base<fp4_t, fp4_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// // clang-format on
// {
//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
//     {
//         return {__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(
//             0, bit_cast<int32x16_t>(aVec), 0, bit_cast<int32x16_t>(bVec), 0, cVec)};
//     }
// };


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
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8(
            bit_cast<int32x16_t>(aVec), bit_cast<int32x16_t>(bVec), 0, cVec, 0, 0)};
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
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8(
            bit_cast<int32x16_t>(aVec), bit_cast<int32x16_t>(bVec), 0, cVec, 0, 0)};
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
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8(
            bit_cast<int32x16_t>(aVec), bit_cast<int32x16_t>(bVec), 0, cVec, 0, 0)};
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
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f32_16x16x32_f16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f32_16x16x32_f16(0, aVec, 0, bVec, 0, cVec, 0, 0)};
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
struct amdgcn_mma<fp16_t, fp16_t, fp16_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp16_t, 16u, 16u, 32u, 32u, 16, 1, 1, 1, 1, 8, 1, WmmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    static constexpr const char* instruction_name = "__builtin_amdgcn_wmma_f16_16x16x32_f16";

    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
    {
        return {__builtin_amdgcn_wmma_f16_16x16x32_f16(0, aVec, 0, bVec, 0, cVec, 0, 0)};
    }
};


// /**
//  * @struct amdgcn_mma
//  * @brief Specialization of amdgcn_mma for fp4_t, fp4_t, fp32_t MMA operation on GFX1250
//  * architecture.
//  * @tparam CtrlFlags Control flags for the WMMA operation
//  * @tparam CompilerTarget Current compiler target
//  */
// // TODO: c++20 template <CtrlFlagsGfx1250I CtrlFlags, amdgcn_target CompilerTarget>
// // TODO: c++20 requires
// template <typename CtrlFlags, typename CompilerTarget>
// // clang-format off
// //               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
// struct amdgcn_mma<fp4_t, fp4_t, fp32_t, 32u, 16u, 128u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx1250_t<CompilerTarget>>
// : amdgcn_mma_base<fp4_t, fp4_t, fp32_t, 32u, 16u, 128u, 32u, 64, 1, 1, 1, 1, 16, 2, WmmaOp, MmaOpFamily::DENSE>
// // clang-format on
// {
//     CK_TILE_DEVICE static CVecType
//     exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec)
//     {
//         return {__builtin_amdgcn_wmma_f32_32x16x128_f4(bit_cast<int32x16_t>(aVec), bit_cast<int32x8_t>(bVec), 0, cVec)};
//     }
// };

} // namespace ck_tile::core::arch::mma
