// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "SdpaModuleCache.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"

#include <cstdint>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

namespace asm_sdpa_engine
{

// True when the byte extent of one K head and of one V head stays below 2^32.
// The extent is S_kv rows of the tensor's sequence stride at `inputBytes` per
// element, with S_kv read from K (dims[2]) for both, since that is the length
// the kernel walks for each. The hd192x128 forward kernels address a head's
// rows through a 32-bit buffer offset, and AITER declines them when either
// extent reaches 2^32 bytes (`_fmha_kv_byte_extent_ge_u32` in aiter/ops/mha.py).
// AITER computes `max_seqlen_k * stride(-3) * element_size`: dim -3 is the token
// axis of its dense [B, S, H, D] and packed [total, H, D] layouts, which is
// strides[2] of hipDNN's logical [B, H, S, D] in batch and group mode alike, and
// in group mode hipDNN's S_kv is the maximum per-batch length AITER passes as
// max_seqlen_k. A negative length or stride, or a non-positive element size,
// does not fit.
bool fwdKvRowExtentsFitU32(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& k,
                           const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& v,
                           int64_t inputBytes);

class SdpaFwdPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder<Handle, Settings, Context>
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

} // namespace asm_sdpa_engine
