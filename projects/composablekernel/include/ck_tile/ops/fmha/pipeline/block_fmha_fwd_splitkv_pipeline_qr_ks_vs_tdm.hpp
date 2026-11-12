// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_pingpong.hpp"

namespace ck_tile {

// qr_tdm and qr_tdm_pingpong run one split of FmhaFwdSplitKVKernel unchanged: the
// SplitKVAttentionMask of BlockFmhaFwdSplitKVTdmPipelineProblem limits the K tile range to the
// split.
template <typename Problem_, typename Policy_ = BlockFmhaPipelineQRKSVSTdmDefaultPolicy>
struct BlockFmhaFwdSplitKVPipelineQRKSVSTdm : BlockFmhaPipelineQRKSVSTdm<Problem_, Policy_>
{
    static constexpr const char* name = "qr_tdm_splitkv";

    // the kernel paths qr_tdm takes by name
    static constexpr bool kUsesUntransposedVKernelPath = true;
    static constexpr bool kUsesTdmAffineDramPath       = true;
};

template <typename Problem_, typename Policy_ = BlockFmhaPipelineQRKSVSTdmPingPongPolicy>
struct BlockFmhaFwdSplitKVPipelineQRKSVSTdmPingPong
    : BlockFmhaPipelineQRKSVSTdmPingPong<Problem_, Policy_>
{
    static constexpr const char* name = "qr_tdm_pingpong_splitkv";
};

} // namespace ck_tile
