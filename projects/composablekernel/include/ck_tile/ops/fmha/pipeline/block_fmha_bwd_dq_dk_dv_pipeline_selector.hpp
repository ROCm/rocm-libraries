// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr_iglp.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr_iglp_dkdv_opt.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_only_qmajor.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_kr_ktr_vr.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_qr_qtr_dor.hpp"

namespace ck_tile {

template <typename Problem, typename Policy>
class BlockFmhaBwdDQDKDVPipelineSelector
{
    static constexpr bool has_dpad1 =
        Problem::Traits::kPadHeadDimQ == 1 || Problem::Traits::kPadHeadDimV == 1;
    static constexpr bool is_decode = Problem::BlockFmhaShape::kMaxSeqLenQ > 0;
    // D128 generic fallback uses KRKTRVR. Product-dual selects DKDVOpt explicitly in codegen.

    public:
    template <typename... TS>
    using type_ = std::conditional_t<
        Problem::kUseTrLoad,
        std::conditional_t<is_decode,
                           BlockFmhaBwdDQDKDVPipelineTrLoadQRQTRDOR<TS...>,
                           BlockFmhaBwdDQDKDVPipelineTrLoadKRKTRVR<TS...>>,
        std::conditional_t<has_dpad1 || (Problem::BlockFmhaShape::kQKHeaddim == 128 &&
                                         Problem::BlockFmhaShape::kVHeaddim == 128),
                           BlockFmhaBwdDQDKDVPipelineKRKTRVR<TS...>,
                           BlockFmhaBwdDQDKDVPipelineKRKTRVRIGLP<TS...>>>;
    using type = std::conditional_t<std::is_same_v<Policy, void>, //
                                    type_<Problem>,
                                    type_<Problem, Policy>>;
};

template <typename Problem, typename Policy = void>
class BlockFmhaBwdDQDKDVPipeline : public BlockFmhaBwdDQDKDVPipelineSelector<Problem, Policy>::type
{
    public:
    static constexpr const char* name = "auto";
};

} // namespace ck_tile
