// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_v128.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_ = BlockFmhaPipelineQRKSVSTdmD192V128Policy>
struct BlockFmhaPipelineQRKSVSTdmD192V128 : BlockFmhaPipelineQRKSVSTdmV128<Problem_, Policy_>
{
    using ScheduleExecutor = BlockFmhaPipelineQRKSVSTdmD192V128ScheduleExecutor;

    static constexpr const char* name = "qr_tdm_d192_v128";
    // The legacy occupancy override does not impose a new member on custom policies.
    static constexpr index_t kBlockPerCu = LegacyD192Tuning::kBlockPerCu;
};

} // namespace ck_tile
