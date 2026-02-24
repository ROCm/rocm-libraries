// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_smfmac_impl.hpp"

namespace ck_tile::core::arch::mma {

struct DefaultSparseMfmaCtrlFlags
{
    static constexpr SparseCompressionIndex CompressionIndex = SparseCompressionIndex::LE16;
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for Sparse MFMA (SMFMA) on GFX942, GFX950 targets
 *
 * This specialization implements the SMFMA instruction for fp16_t A and B
 * matrices with structured sparsity, fp32_t accumulator, with 16x16x32 block sizes.
 *
 * @tparam CtrlFlags Control flags for the Sparse MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsSparseMfmaI CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
struct amdgcn_mma<
    fp16_t,
    fp16_t,
    fp32_t,
    16u,
    16u,
    32u,
    CtrlFlags,
    CompilerTarget,
    MmaOpFamily::SPARSE,
    std::enable_if_t<is_any_value_of(
        CompilerTarget::TARGET_ID, amdgcn_target_id::GFX942, amdgcn_target_id::GFX950)>>
{
    using OpType                          = MfmaOp;
    static constexpr MmaOpFamily OpFamily = MmaOpFamily::SPARSE;

    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 8>;
    using CVecType = ext_vector_t<fp32_t, 4>;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 8;

    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    static constexpr index_t kCompressionRatio = 2;

    CK_TILE_DEVICE static auto
    exec(AVecType& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        // TODO: Compressing A on-the-fly should be OK for now, but  we need to validate
        // and evaluate changing this to a transform at a higher level.
        const int32_t idx = ck_tile::compress_a_impl<fp16_t>(aVec);

        using namespace sparse::detail;
        static constexpr BuiltinParams PARAMS =
            get_builtin_params<CtrlFlags::CompressionIndex>::value;
        return {__builtin_amdgcn_smfmac_f32_16x16x32_f16(
            aVec, bVec, cVec, idx, PARAMS.Override16BitDefaultMask, PARAMS.ByteIndexToOverride)};
    }
};

} // namespace ck_tile::core::arch::mma
