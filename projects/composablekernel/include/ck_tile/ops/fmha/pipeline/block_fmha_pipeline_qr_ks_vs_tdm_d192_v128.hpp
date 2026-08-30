// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_output.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_schedule_executor.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_softmax.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_ = BlockFmhaPipelineQRKSVSTdmD192V128Policy>
struct BlockFmhaPipelineQRKSVSTdmD192V128 : BlockFmhaPipelineQRKSVSTdm<Problem_, Policy_>
{
    using Base             = BlockFmhaPipelineQRKSVSTdm<Problem_, Policy_>;
    using Problem          = remove_cvref_t<Problem_>;
    using Policy           = remove_cvref_t<Policy_>;
    using ScheduleExecutor = BlockFmhaPipelineQRKSVSTdmD192V128ScheduleExecutor;
    using SplitSoftmax     = FmhaD192SplitSoftmax;

    static_assert(Policy::template IsSupportedProblem<Problem>(),
                  "qr_tdm_d192_v128 requires the exact BF16 D192/V128 geometry");

    static constexpr const char* name = "qr_tdm_d192_v128";

    static constexpr bool kUsesUntransposedVKernelPath = true;
    static constexpr bool kUsesTdmAffineDramPath       = true;
    static constexpr bool kUsesFixedSegmentedLdsArena  = true;
    static constexpr index_t kBlockPerCu               = 1;

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
    using QKBlockGemm = remove_cvref_t<decltype(Policy::template GetQKBlockGemm<Problem>())>;
    using ScoreTile   = decltype(QKBlockGemm::MakeCBlockTile());
    static_assert(ScoreTile::get_thread_buffer_size() == SplitSoftmax::Mapping::kThreadBufferSize);
#endif

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        static_assert(Policy::template GetSmemSize<Problem>() == Policy::GetLdsArenaSize());
        return Policy::GetLdsArenaSize();
    }

    using Base::operator();
    using Base::run;
};

} // namespace ck_tile
