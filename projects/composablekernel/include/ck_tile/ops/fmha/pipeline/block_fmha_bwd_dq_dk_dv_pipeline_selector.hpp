// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr_iglp.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_ldsacc_kr_ktr_vr.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_kr_ktr_vr.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_qr_qtr_dor.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_qr_qtr_dor_tdm.hpp"

#ifndef CK_TILE_FMHA_BWD_TRLOAD_TDM
#define CK_TILE_FMHA_BWD_TRLOAD_TDM 0
#endif

namespace ck_tile {

template <typename Problem, typename Policy>
class BlockFmhaBwdDQDKDVPipelineSelector
{
    static constexpr bool has_dpad1 =
        Problem::Traits::kPadHeadDimQ == 1 || Problem::Traits::kPadHeadDimV == 1;
    static constexpr bool is_decode = Problem::BlockFmhaShape::kMaxSeqLenQ > 0;

    public:
    template <typename... TS>
    using type_ =
        std::conditional_t<Problem::kUseTrLoad,
                           std::conditional_t<
                               is_decode,
#if CK_TILE_FMHA_BWD_TRLOAD_TDM
                               BlockFmhaBwdDQDKDVPipelineTrLoadQRQTRDORTDM<TS...>,
#else
                               BlockFmhaBwdDQDKDVPipelineTrLoadQRQTRDOR<TS...>,
#endif
                               BlockFmhaBwdDQDKDVPipelineTrLoadKRKTRVR<TS...>>,
                           // LdsAcc is the IGLP pipeline with the dK/dV
                           // accumulators moved to LDS, so it slots in wherever
                           // IGLP would have been chosen. It outranks has_dpad1:
                           // that gate predates this pipeline and all three treat
                           // kPadHeadDim identically (it only lowers the load
                           // alignment), so sending dpad=1 to KRKTRVR cost 3.5-9x
                           // for no reason.
                           std::conditional_t<
                               Problem::kUseLdsAcc,
                               BlockFmhaBwdDQDKDVPipelineLdsAccKRKTRVR<TS...>,
                               std::conditional_t<
                                   has_dpad1,
                                   BlockFmhaBwdDQDKDVPipelineKRKTRVR<TS...>,
                                   BlockFmhaBwdDQDKDVPipelineKRKTRVRIGLP<TS...>>>>;
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
