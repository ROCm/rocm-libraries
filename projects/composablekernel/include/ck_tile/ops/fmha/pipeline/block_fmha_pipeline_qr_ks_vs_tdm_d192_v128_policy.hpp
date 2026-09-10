// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_v128_policy.hpp"

namespace ck_tile {

struct BlockFmhaPipelineQRKSVSTdmD192V128Policy
    : BlockFmhaPipelineQRKSVSTdmV128Policy<LegacyD192Geometry,
                                           LegacyD192Tuning,
                                           FmhaTdmV128LegacySchedule<LegacyD192Geometry>>
{
};

} // namespace ck_tile
