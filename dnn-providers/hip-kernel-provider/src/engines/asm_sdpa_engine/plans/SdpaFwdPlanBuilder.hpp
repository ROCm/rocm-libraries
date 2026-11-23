// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "SdpaModuleCache.hpp"
#include "SdpaPlanUtils.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"

#include <cstdint>

#include <hip_kernel_provider_common/SdpaConfigEnumerations.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include <optional>
#include <string>

// Generated forward kernel table row (asm_fmha_v3_fwd_configs.hpp).
struct fmha_v3_fwdConfig; // NOLINT(readability-identifier-naming)

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

// Threads per block computeFwdLaunchParams launches a table row's kernel with:
// fwdBlockDimX for the row's arch, dtype, head dims and mode. A graph resolves
// only to a row whose fields equal its own, so this is the block its plan uses.
unsigned int fwdConfigBlockDimX(const fmha_v3_fwdConfig& config);

// Key of the forward kernel table row (cfg_fmha_fwd) a graph with these fields runs on,
// or an empty string when the table has none. gfx942 also matches bf16Cvt; gfx950
// ignores it. maskType is the registry ordinal, after fwdKernelMaskType's mapping.
std::string fwdKernelNameKey(const std::string& archId,
                             const std::string& dataType,
                             int hdimQ,
                             int hdimV,
                             plan_utils::MaskType maskType,
                             hip_kernel_provider_common::RoundingMode bf16Cvt,
                             hip_kernel_provider_common::BatchMode mode);

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

protected:
    // Largest block, in threads, the resolved kernel can be launched with. The
    // generated table records it from each code object, so no kernel is loaded
    // to answer it. Virtual so tests can drive the launch-limit decline: every
    // vendored kernel accepts the block it is launched with.
    virtual unsigned int maxThreadsPerBlock(const fmha_v3_fwdConfig& config) const;

private:
    // Describes why the resolved kernel cannot be launched with the block the
    // plan uses, or std::nullopt when it can.
    std::optional<std::string> launchLimitViolation(const fmha_v3_fwdConfig& config) const;

    // launchLimitViolation as an isApplicable verdict: logs the reason and
    // returns false when the kernel cannot be launched with the plan's block.
    bool acceptsLaunchBlock(const fmha_v3_fwdConfig& config) const;
};

} // namespace asm_sdpa_engine
