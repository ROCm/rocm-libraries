// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "SdpaModuleCache.hpp"
#include "SdpaPlanUtils.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"

#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include <cstdint>
#include <string>

namespace asm_sdpa_engine
{

class SdpaBwdPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder<Handle, Settings, Context>
{
public:
    static SdpaModuleCache& moduleCache()
    {
        static SdpaModuleCache s_cache;
        return s_cache;
    }

    bool isApplicable(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getMaxWorkspaceSize(const Handle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const Settings& executionSettings) const override;

    void initializeExecutionSettings(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        Settings& executionSettings) const override;

    void buildPlan(const Handle& handle,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                   Context& executionContext) const override;

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>
        // NOLINTNEXTLINE(portability-template-virtual-member-function)
        getCustomKnobs(
            const Handle& handle,
            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
};

namespace bwd_dispatch
{

// Pipeline-stage tag used by the registry-lookup helper to select which
// CSV-derived registry to walk.
enum class PipelineStage
{
    ODO,
    DQDKDV,
    DQ_CONVERT
};

// Value stored in the bf16_cvt CSV column for rows that serve every rounding
// mode: the FP16 rows (where rounding mode is not applicable) and the bf16 rows
// that ship one kernel for all modes. FP16 lookups request it. Shared between
// the builder and its unit tests.
inline constexpr int BF16_CVT_FP16_SENTINEL = 3;

// True for the device architectures the backward engine dispatches on.
bool isSupportedArch(const std::string& archId);

// False for batch-mode head dims that AITER v0.1.29 sends to CK instead of the
// ASM backward on archId although the kernel registry has rows for them
// (aiter/ops/mha.py _flash_attn_backward): on gfx950 everything except
// 64 < D <= 128 with D_qk == D_v, or D_qk = 192 with D_v = 128. Of those the
// engine serves D_qk == D_v == 128. gfx942 head dims are left to the registry.
bool isBatchShapeSupported(const std::string& archId, int64_t headDimQk, int64_t headDimV);

// The dqdkdv CSV `mask` ordinal for a classified mask on archId. Mirrors AITER
// mha_bwd.cu asm_mask_type(): on gfx950 a sliding window with bottom-right
// alignment selects the BWD_MASK_BOTTOM_RIGHT_SLIDING_WINDOW kernels; every
// other mask keeps its plan_utils::MaskType ordinal.
int dqdkdvMaskOrdinal(const std::string& archId,
                      plan_utils::MaskType maskType,
                      hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment alignment);

// Resolves a registry row matching the supplied dispatch tuple. Returns the
// composite key (arch + knl_name) on a hit, or an empty string when no row
// in the chosen registry matches. The bf16Cvt argument is the rounding mode the
// CSV column encodes (0/1/2 for the bf16 rounding modes; 3 for fp16). A row
// matches when its bf16_cvt equals bf16Cvt or is BF16_CVT_FP16_SENTINEL.
// Exposed for unit testing of the dispatch logic.
std::string lookupKernelNameKey(PipelineStage stage,
                                const std::string& archId,
                                const std::string& dataType,
                                int hdimQ,
                                int hdimV,
                                int mask,
                                int atomic32,
                                int pssk,
                                int pddv,
                                int mode,
                                int bf16Cvt);

} // namespace bwd_dispatch

} // namespace asm_sdpa_engine
