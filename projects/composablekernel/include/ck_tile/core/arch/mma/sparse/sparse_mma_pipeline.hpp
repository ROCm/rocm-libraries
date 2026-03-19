// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core/arch/mma/mma_pipeline.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_transforms.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include <cstdint>
#include <type_traits>

namespace ck_tile::core::arch::mma {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t FragM,
          uint32_t FragN,
          uint32_t FragK,
          typename CompilerTarget =
              decltype(get_compiler_target()), // TODO: c++20 amdgcn_target_arch_id GfxTargetId =
                                               // get_compiler_target(),
          typename MmaOp =
              typename MmaDefaultSelector<ADataType, // TODO: c++20 MmaOpI MmaOp = typename
                                                     // MmaDefaultSelector<ADataType,
                                          BDataType,
                                          CDataType,
                                          FragM,
                                          FragN,
                                          FragK,
                                          CompilerTarget,
                                          MmaOpFamily::SPARSE>::SelectedOp,
          typename MmaTransforms = // TODO: c++20 MmaTransformsI MmaTransforms =
          typename MmaTransformsDefaultSelector<MmaOp, CompilerTarget>::SelectedTransforms>
// clang-format off
struct SparseMma : public MmaPipelineBase<static_cast<int>(MmaPipelineOptionFlag::COMPRESS_A), // TODO: c++20: use MmaPipelineOptionFlags directly
                                          SparseMma<ADataType, BDataType, CDataType, FragM, FragN, FragK, CompilerTarget, MmaOp, MmaTransforms>>
{
    static_assert(MmaOpTraits<MmaOp>::IsSupported && MmaOpTraits<MmaOp>::IsSparse);
    using Base = MmaPipelineBase<static_cast<int>(MmaPipelineOptionFlag::COMPRESS_A), // TODO: c++20: use MmaPipelineOptionFlags directly
                                 SparseMma<ADataType, BDataType, CDataType, FragM, FragN, FragK, CompilerTarget, MmaOp, MmaTransforms>>;
    // clang-format on

    // Calculate the uncompressed A vector type
    struct InternalAVecCalculator
    {
        using AVecTraits               = vector_traits<typename MmaOp::AVecType>;
        static constexpr index_t ASize = AVecTraits::vector_size * MmaOp::kCompressionRatio;
        using AVecType                 = ext_vector_t<typename AVecTraits::scalar_type, ASize>;
    };

    // Expose caller-side vector types
    using AVecType = typename InternalAVecCalculator::AVecType;
    using BVecType = typename MmaOp::BVecType;
    using CVecType = typename MmaOp::CVecType;

    // Transforms
    using ATransform = typename MmaTransforms::ATransform;
    using BTransform = typename MmaTransforms::BTransform;
    using CTransform = typename MmaTransforms::CTransform;
    using DTransform = typename MmaTransforms::DTransform;

    template <MmaPipelineOptionFlags::Type Flags, typename VecTA, typename VecTB, typename VecTC>
    CK_TILE_DEVICE static decltype(auto) preApply(VecTA&& a, VecTB&& b, VecTC&& accum)
    {
        static_assert(Flags == MmaPipelineOptionFlags(MmaPipelineOptionFlag::COMPRESS_A));
        static_assert(
            std::is_same_v<ATransform, SparseCompressTransform<MmaOp::kCompressionRatio>>);

        using InternalAVecT = typename MmaOp::AVecType;
        using InternalBVecT = typename MmaOp::BVecType;
        using InternalCVecT = typename MmaOp::CVecType;

        int32_t idx{};
        auto a_frag = Base::template preApplyTransform<InternalAVecT, ATransform>(
            std::forward<VecTA>(a), idx);
        auto b_frag =
            Base::template preApplyTransform<InternalBVecT, BTransform>(std::forward<VecTB>(b));
        auto c_frag =
            Base::template preApplyTransform<InternalCVecT, CTransform>(std::forward<VecTC>(accum));

        return std::make_tuple(
            std::move(a_frag), std::move(b_frag), std::move(c_frag), std::move(idx));
    }

    template <MmaPipelineOptionFlags::Type Flags, typename VecTA, typename VecTB, typename VecTC>
    CK_TILE_DEVICE static decltype(auto) postApply(std::tuple<VecTA, VecTB, VecTC, int32_t>&& vecs)
    {
        static_assert(Flags == MmaPipelineOptionFlags(MmaPipelineOptionFlag::COMPRESS_A));

        auto& [a_frag, b_frag, c_frag, idx] = vecs;
        // Convert native vector results back to the output fragment format
        // and then return after we apply the final output transform.
        return Base::template postApplyTransform<std::decay_t<VecTC>, DTransform>(c_frag);
    }

    template <typename VecTA, typename VecTB, typename VecTC>
    CK_TILE_DEVICE static void execImpl(std::tuple<VecTA, VecTB, VecTC, int32_t>& vecs)
    {
        auto& [a_frag, b_frag, c_frag, idx] = vecs;
        c_frag                              = MmaOp::exec(a_frag, b_frag, c_frag, idx);
    }
};

} // namespace ck_tile::core::arch::mma
