// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_default_policy.hpp"

namespace ck_tile {

/// Policy of GemmPipelineAgBgCrCompTDMV3. Derived from, never edits, the shared TDM default
/// policy (wave-specialized form, as used by comp_tdm_v2), so that v3-only knobs do not change
/// any other pipeline.
struct GemmPipelineAgBgCrCompTDMV3Policy : public GemmPipelineAgBgCrCompTDMDefaultPolicy<true>
{
};

} // namespace ck_tile
