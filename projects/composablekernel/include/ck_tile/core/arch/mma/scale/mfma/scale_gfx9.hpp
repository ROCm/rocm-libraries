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

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for Scale MFMA on GFX950 targets
 *
 * This specialization implements the Scale MFMA instruction for fp8_t A and B
 * matrices with fp32_t accumulator, with 16x16x128 block sizes.
 *
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <amdgcn_target CompilerTarget>
// TODO: c++20 requires

// clang-format off
#define MMA_SCALE_ARG_F8(vec) bit_cast<int32x8_t>(vec)
#define MMA_SCALE_ARG_F6(vec) int32x8_t{vec.data[0], vec.data[1], vec.data[2], vec.data[3], vec.data[4], vec.data[5], 0, 0}
#define MMA_SCALE_ARG_F4(vec) int32x8_t{bit_cast<int32x4_t>(vec)[0], bit_cast<int32x4_t>(vec)[1], bit_cast<int32x4_t>(vec)[2], bit_cast<int32x4_t>(vec)[3], 0, 0, 0, 0}

#define DEFINE_MMA_SCALE_GFX950_16(AType, BType, EXPAND_A, EXPAND_B, NUM_ACC_A, NUM_ACC_B)            \
template <typename CompilerTarget>                                              \
struct amdgcn_mma<AType, BType, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>> \
: amdgcn_mma_base<AType, BType, fp32_t, 16u, 16u, 128u, 64u, 32, NUM_ACC_A, 1, NUM_ACC_B, 1, 4, 1, MfmaOp, MmaOpFamily::SCALE> \
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

#define DEFINE_MMA_SCALE_GFX950_32(AType, BType, EXPAND_A, EXPAND_B, NUM_ACC_A, NUM_ACC_B)            \
template <typename CompilerTarget>                                              \
struct amdgcn_mma<AType, BType, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>> \
: amdgcn_mma_base<AType, BType, fp32_t, 32u, 32u, 64u, 64u, 32, NUM_ACC_A, 1, NUM_ACC_B, 1, 16, 4, MfmaOp, MmaOpFamily::SCALE> \
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

// Note on the intrinsic NumAccess values we use here: In principle the "canonical" NumAccess values
// for A and B for gfx950 scale intrinsic is determined by the A and B datatypes. 8-bit datatypes
// require a NumAccess of 2, and 4 and 6-bit types a NumAccess of 1. We follow this *BUT* we do
// allow (1,1) for the cases where A and B are both 8 bit. In these cases, NumAccess (1,1) could
// still be valid when not using scale values.

// 25 intrinsics for __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4
DEFINE_MMA_SCALE_GFX950_16(fp8_t,       fp8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(fp8_t,       bf8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(bf8_t,       fp8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(bf8_t,       bf8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(fp8_t,       pk_fp6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 2, 1)
DEFINE_MMA_SCALE_GFX950_16(fp8_t,       pk_bf6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 2, 1)
DEFINE_MMA_SCALE_GFX950_16(fp8_t,       pk_fp4_t,    MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F4, 2, 1)
DEFINE_MMA_SCALE_GFX950_16(bf8_t,       pk_fp6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 2, 1)
DEFINE_MMA_SCALE_GFX950_16(bf8_t,       pk_bf6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 2, 1)
DEFINE_MMA_SCALE_GFX950_16(bf8_t,       pk_fp4_t,    MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F4, 2, 1)
DEFINE_MMA_SCALE_GFX950_16(pk_fp6x16_t, fp8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_16(pk_fp6x16_t, bf8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_16(pk_fp6x16_t, pk_fp6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(pk_fp6x16_t, pk_bf6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(pk_fp6x16_t, pk_fp4_t,    MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F4, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(pk_bf6x16_t, fp8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_16(pk_bf6x16_t, bf8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_16(pk_bf6x16_t, pk_fp6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(pk_bf6x16_t, pk_bf6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(pk_bf6x16_t, pk_fp4_t,    MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F4, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(pk_fp4_t,    fp8_t,       MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_16(pk_fp4_t,    bf8_t,       MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_16(pk_fp4_t,    pk_fp6x16_t, MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(pk_fp4_t,    pk_bf6x16_t, MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_16(pk_fp4_t,    pk_fp4_t,    MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F4, 1, 1)

// 25 intrinsics for __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4
DEFINE_MMA_SCALE_GFX950_32(fp8_t,       fp8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(fp8_t,       bf8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(bf8_t,       fp8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(bf8_t,       bf8_t,       MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F8, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(fp8_t,       pk_fp6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 2, 1)
DEFINE_MMA_SCALE_GFX950_32(fp8_t,       pk_bf6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 2, 1)
DEFINE_MMA_SCALE_GFX950_32(fp8_t,       pk_fp4_t,    MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F4, 2, 1)
DEFINE_MMA_SCALE_GFX950_32(bf8_t,       pk_fp6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 2, 1)
DEFINE_MMA_SCALE_GFX950_32(bf8_t,       pk_bf6x16_t, MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F6, 2, 1)
DEFINE_MMA_SCALE_GFX950_32(bf8_t,       pk_fp4_t,    MMA_SCALE_ARG_F8, MMA_SCALE_ARG_F4, 2, 1)
DEFINE_MMA_SCALE_GFX950_32(pk_fp6x16_t, fp8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_32(pk_fp6x16_t, bf8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_32(pk_fp6x16_t, pk_fp6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(pk_fp6x16_t, pk_bf6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(pk_fp6x16_t, pk_fp4_t,    MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F4, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(pk_bf6x16_t, fp8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_32(pk_bf6x16_t, bf8_t,       MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_32(pk_bf6x16_t, pk_fp6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(pk_bf6x16_t, pk_bf6x16_t, MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(pk_bf6x16_t, pk_fp4_t,    MMA_SCALE_ARG_F6, MMA_SCALE_ARG_F4, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(pk_fp4_t,    fp8_t,       MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_32(pk_fp4_t,    bf8_t,       MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F8, 1, 2)
DEFINE_MMA_SCALE_GFX950_32(pk_fp4_t,    pk_fp6x16_t, MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(pk_fp4_t,    pk_bf6x16_t, MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F6, 1, 1)
DEFINE_MMA_SCALE_GFX950_32(pk_fp4_t,    pk_fp4_t,    MMA_SCALE_ARG_F4, MMA_SCALE_ARG_F4, 1, 1)

#undef MMA_SCALE_ARG_F8
#undef MMA_SCALE_ARG_F6
#undef MMA_SCALE_ARG_F4
#undef DEFINE_MMA_SCALE_GFX950_16
#undef DEFINE_MMA_SCALE_GFX950_32
// clang-format on

=======
template <typename CompilerTarget>
// clang-format off
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u, 64u, 32, 2, 1, 2, 1, 4, 1, MfmaOp, MmaOpFamily::SCALE>
//               | A B C DataTypes    | MNK + WaveSize     |AParams  |BPar |CPar |
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            bit_cast<int32x8_t>(aVec),
            bit_cast<int32x8_t>(bVec),
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<fp8_t>,
            scale::detail::ScaleDataTypeToFlag_v<fp8_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};

    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            int32x8_t{aVec.data[0], aVec.data[1], aVec.data[2], aVec.data[3], aVec.data[4], aVec.data[5], 0, 0},
            int32x8_t{bVec.data[0], bVec.data[1], bVec.data[2], bVec.data[3], bVec.data[4], bVec.data[5], 0, 0},
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<pk_fp6x16_t>,
            scale::detail::ScaleDataTypeToFlag_v<pk_fp6x16_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};
// clang-format on

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for Scale MFMA on GFX950 targets
 *
 * This specialization implements the Scale MFMA instruction for pk_bf6x16_t A and B
 * matrices with fp32_t accumulator, with 16x16x128 block sizes.
 *
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CompilerTarget>
// clang-format off
struct amdgcn_mma<pk_bf6x16_t, pk_bf6x16_t, fp32_t, 16u, 16u, 128u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<pk_bf6x16_t, pk_bf6x16_t, fp32_t, 16u, 16u, 128u, 64u, 32, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::SCALE>
//               | A B C DataTypes                | MNK + WaveSize     |AParams  |BPar |CPar |
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4";
    
    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            int32x8_t{aVec.data[0], aVec.data[1], aVec.data[2], aVec.data[3], aVec.data[4], aVec.data[5], 0, 0},
            int32x8_t{bVec.data[0], bVec.data[1], bVec.data[2], bVec.data[3], bVec.data[4], bVec.data[5], 0, 0},
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<pk_bf6x16_t>,
            scale::detail::ScaleDataTypeToFlag_v<pk_bf6x16_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};
// clang-format on

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for Scale MFMA on GFX950 targets
 *
 * This specialization implements the Scale MFMA instruction for fp8_t A and B
 * matrices with fp32_t accumulator, with 32x32x64 block sizes.
 *
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CompilerTarget>
// clang-format off
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 32u, 32u, 64u, 64u, 32, 2, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SCALE>
//               | A B C DataTypes    | MNK + WaveSize    |AParams  |BPar |CPar  |
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            bit_cast<int32x8_t>(aVec),
            bit_cast<int32x8_t>(bVec),
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<fp8_t>,
            scale::detail::ScaleDataTypeToFlag_v<fp8_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for Scale MFMA on GFX950 targets
 *
 * This specialization implements the Scale MFMA instruction for bf8_t A and B
 * matrices with fp32_t accumulator, with 32x32x64 block sizes.
 *
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CompilerTarget>
// clang-format off
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 32u, 32u, 64u, 64u, 32, 2, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SCALE>
//               | A B C DataTypes    | MNK + WaveSize    |AParams  |BPar |CPar  |
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            bit_cast<int32x8_t>(aVec),
            bit_cast<int32x8_t>(bVec),
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<bf8_t>,
            scale::detail::ScaleDataTypeToFlag_v<bf8_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};

template <typename CompilerTarget>
// clang-format off
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 32u, 32u, 64u, 64u, 32, 2, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SCALE>
//               | A B C DataTypes    | MNK + WaveSize    |AParams  |BPar |CPar  |
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            bit_cast<int32x8_t>(aVec),
            bit_cast<int32x8_t>(bVec),
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<fp8_t>,
            scale::detail::ScaleDataTypeToFlag_v<bf8_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};

template <typename CompilerTarget>
// clang-format off
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 32u, 32u, 64u, 64u, 32, 2, 1, 2, 1, 16, 4, MfmaOp, MmaOpFamily::SCALE>
//               | A B C DataTypes    | MNK + WaveSize    |AParams  |BPar |CPar  |
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            bit_cast<int32x8_t>(aVec),
            bit_cast<int32x8_t>(bVec),
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<bf8_t>,
            scale::detail::ScaleDataTypeToFlag_v<fp8_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for Scale MFMA on GFX950 targets
 *
 * This specialization implements the Scale MFMA instruction for pk_fp4_t A and B
 * matrices with fp32_t accumulator, with 32x32x64 block sizes.
 *
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CompilerTarget>
// clang-format off
struct amdgcn_mma<pk_fp4_t, pk_fp4_t, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>> 
: amdgcn_mma_base<pk_fp4_t, pk_fp4_t, fp32_t, 32u, 32u, 64u, 64u, 32, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SCALE>
//               | A B C DataTypes          | MNK + WaveSize    |AParams  |BPar |CPar  |
// clang-format on
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B)
    {
        using P         = WarpGemmParamsParser<Params...>;
        int32x4_t arg_a = bit_cast<int32x4_t>(aVec);
        int32x4_t arg_b = bit_cast<int32x4_t>(bVec);

        return {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            int32x8_t{arg_a[0], arg_a[1], arg_a[2], arg_a[3], 0, 0, 0, 0},
            int32x8_t{arg_b[0], arg_b[1], arg_b[2], arg_b[3], 0, 0, 0, 0},
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<pk_fp4_t>,
            scale::detail::ScaleDataTypeToFlag_v<pk_fp4_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for Scale MFMA on GFX950 targets
 *
 * This specialization implements the Scale MFMA instruction for pk_fp6x16_t A and B
 * matrices with fp32_t accumulator, with 32x32x64 block sizes.
 *
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CompilerTarget>
// clang-format off
struct amdgcn_mma<pk_fp6x16_t, pk_fp6x16_t, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<pk_fp6x16_t, pk_fp6x16_t, fp32_t, 32u, 32u, 64u, 64u, 32, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SCALE>
//               | A B C DataTypes                | MNK + WaveSize    |AParams  |BPar |CPar  |
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            int32x8_t{aVec.data[0], aVec.data[1], aVec.data[2], aVec.data[3], aVec.data[4], aVec.data[5], 0, 0},
            int32x8_t{bVec.data[0], bVec.data[1], bVec.data[2], bVec.data[3], bVec.data[4], bVec.data[5], 0, 0},
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<pk_fp6x16_t>,
            scale::detail::ScaleDataTypeToFlag_v<pk_fp6x16_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};
// clang-format on

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for Scale MFMA on GFX950 targets
 *
 * This specialization implements the Scale MFMA instruction for pk_bf6x16_t A and B
 * matrices with fp32_t accumulator, with 32x32x64 block sizes.
 *
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CompilerTarget>
// clang-format off
struct amdgcn_mma<pk_bf6x16_t, pk_bf6x16_t, fp32_t, 32u, 32u, 64u, CompilerTarget, MmaOpFamily::SCALE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<pk_bf6x16_t, pk_bf6x16_t, fp32_t, 32u, 32u, 64u, 64u, 32, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::SCALE>
//               | A B C DataTypes                | MNK + WaveSize    |AParams  |BPar |CPar  |
{
    static constexpr const char* instruction_name =
        "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4";

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec, int scale_A, int scale_B)
    {
        using P = WarpGemmParamsParser<Params...>;
        return {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            int32x8_t{aVec.data[0], aVec.data[1], aVec.data[2], aVec.data[3], aVec.data[4], aVec.data[5], 0, 0},
            int32x8_t{bVec.data[0], bVec.data[1], bVec.data[2], bVec.data[3], bVec.data[4], bVec.data[5], 0, 0},
            cVec,
            scale::detail::ScaleDataTypeToFlag_v<pk_bf6x16_t>,
            scale::detail::ScaleDataTypeToFlag_v<pk_bf6x16_t>,
            P::op_sel_a,
            scale_A,
            P::op_sel_b,
            scale_B)};
    }
};
// clang-format on

>>>>>>> 25b558dce0f (Add some mixed precision gfx9 scale intrinsics (f8bf8 bf8f8))
} // namespace ck_tile::core::arch::mma
