// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>

#include <cmath>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <stdexcept>

#include "HipFlash2FwdPlan.hpp"
#include "HipFlash2FwdPlanBuilder_v2.hpp"
#include "HipFlash2KernelUtils.hpp"

namespace hip_flash2_engine
{

using namespace hip_kernel_provider_common;
using namespace hipdnn_flatbuffers_sdk;

// ---------------------------------------------------------------------------
// isApplicable
// ---------------------------------------------------------------------------
bool HipFlash2FwdPlanBuilder::isApplicable(const Handle& handle,
                                           const flatbuffer_utilities::IGraph& opGraph) const
{
    // NOLINTNEXTLINE(readability-identifier-naming)
    static const char* LOG_PREFIX = "[HipFlash2FwdPlanBuilder::isApplicable] ";

    // ── rocWMMA availability (I3) ─────────────────────────────────────────────
    // rocWMMA MFMA is baked into the precompiled .co at offline build time
    // (HipFlash2FwdPlan.hip is compiled with -I/opt/rocm/include which provides
    // rocwmma/rocwmma.hpp). The engine ships the compiled binary; no runtime
    // check is needed because gfx942 and gfx950 always support MFMA. The arch
    // guard below is the effective rocWMMA gate: the compiled .co is only valid
    // for those two arches and hipModuleLoad will fail on any other device.
    // If the .co was ever built without rocWMMA (which cannot happen from the
    // in-tree offline build), hipModuleGetFunction would not find the kernel
    // and buildPlan() would throw, surfacing the failure visibly.

    // ── Device check ─────────────────────────────────────────────────────────
    std::string archId;
    try
    {
        archId = getDeviceString(handle.getStream());
        HIP_KERNEL_RETURN_FALSE_IF(archId != "gfx942" && archId != "gfx950",
                                   "Device not gfx942/gfx950 (actual: " + archId + ")");
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_ERROR(LOG_PREFIX << "getDeviceString failed: " << e.what());
        return false;
    }

    // ── Single SDPA node ─────────────────────────────────────────────────────
    auto& nodeWrappers = opGraph.nodeWrappers();
    HIP_KERNEL_RETURN_FALSE_IF(nodeWrappers.size() != 1, "Graph must have exactly one node");
    HIP_KERNEL_RETURN_FALSE_IF(nodeWrappers.front()->attributesType()
                                   != data_objects::NodeAttributes::SdpaAttributes,
                               "Node must be SdpaAttributes");

    const auto& attrs = nodeWrappers.front()->attributesAs<data_objects::SdpaAttributes>();

    // ── Unsupported optional features ────────────────────────────────────────
    HIP_KERNEL_RETURN_FALSE_IF(attrs.dropout_probability().has_value()
                                   && attrs.dropout_probability().value() != 0.f,
                               "dropout not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.alibi_mask(), "alibi_mask not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.padding_mask(), "padding_mask not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.attn_mask_tensor_uid(), "attn_mask tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.page_table_k_tensor_uid(), "page_table_k not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.page_table_v_tensor_uid(), "page_table_v not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.generate_stats(), "LSE stats output not supported");
    // Variable-length (grouped) batches not yet supported
    HIP_KERNEL_RETURN_FALSE_IF(attrs.seq_len_q_tensor_uid().has_value()
                                   || attrs.seq_len_kv_tensor_uid().has_value(),
                               "variable-length (group) batch mode not supported");

    // ── Tensor shapes ─────────────────────────────────────────────────────────
    const auto& tensorMap = opGraph.getTensorMap();
    auto* qTensor = tensorMap.at(attrs.q_tensor_uid());
    auto* kTensor = tensorMap.at(attrs.k_tensor_uid());
    auto* vTensor = tensorMap.at(attrs.v_tensor_uid());
    auto* oTensor = tensorMap.at(attrs.o_tensor_uid());

    HIP_KERNEL_RETURN_FALSE_IF(qTensor->dims()->size() != 4, "Q must be rank-4");
    HIP_KERNEL_RETURN_FALSE_IF(kTensor->dims()->size() != 4, "K must be rank-4");
    HIP_KERNEL_RETURN_FALSE_IF(vTensor->dims()->size() != 4, "V must be rank-4");
    HIP_KERNEL_RETURN_FALSE_IF(oTensor->dims()->size() != 4, "O must be rank-4");

    // -- Tensor layout: innermost dim must be contiguous (I2) -----------------
    // The kernel d-loop assumes stride-1 on dim 3 (head_dim). Reject non-contiguous layouts.
    HIP_KERNEL_RETURN_FALSE_IF(qTensor->strides()->Get(3) != 1,
                               "Q head_dim stride must be 1 (contiguous)");
    HIP_KERNEL_RETURN_FALSE_IF(kTensor->strides()->Get(3) != 1,
                               "K head_dim stride must be 1 (contiguous)");
    HIP_KERNEL_RETURN_FALSE_IF(vTensor->strides()->Get(3) != 1,
                               "V head_dim stride must be 1 (contiguous)");
    HIP_KERNEL_RETURN_FALSE_IF(oTensor->strides()->Get(3) != 1,
                               "O head_dim stride must be 1 (contiguous)");

    // ── Data type: FP16 only (BF16 not yet supported by V7 kernel) ────────────
    const auto qType = qTensor->data_type();
    const auto kType = kTensor->data_type();
    const auto vType = vTensor->data_type();
    const auto oType = oTensor->data_type();
    const bool fp16
        = (qType == data_objects::DataType::HALF) && (kType == data_objects::DataType::HALF)
          && (vType == data_objects::DataType::HALF) && (oType == data_objects::DataType::HALF);
    HIP_KERNEL_RETURN_FALSE_IF(!fp16, "only FP16 Q/K/V/O is supported");

    // ── head_dim: {64, 128} ───────────────────────────────────────────────────
    // Q layout: [B, H_q, S_q, D_qk]
    const int headDim = static_cast<int>(qTensor->dims()->Get(3));
    HIP_KERNEL_RETURN_FALSE_IF(headDim != 64 && headDim != 128,
                               "head_dim must be 64 or 128 (actual: " + std::to_string(headDim)
                                   + ")");

    // head_dim_v must equal head_dim_qk (V7 kernel assumes D_v == D_qk)
    const int headDimV = static_cast<int>(vTensor->dims()->Get(3));
    HIP_KERNEL_RETURN_FALSE_IF(headDimV != headDim, "head_dim_v must equal head_dim_qk");

    // ── GQA divisibility (I1) ─────────────────────────────────────────────────
    // Kernel computes hk = hq / (Hq / Hk); integer division is only correct when
    // Hq is exactly divisible by Hk.  Reject non-divisible ratios rather than
    // produce silently wrong output.
    const int numHeadsQ = static_cast<int>(qTensor->dims()->Get(1));
    const int numHeadsKv = static_cast<int>(kTensor->dims()->Get(1));
    HIP_KERNEL_RETURN_FALSE_IF(numHeadsKv <= 0 || numHeadsQ % numHeadsKv != 0,
                               "num_heads_q must be divisible by num_heads_kv for GQA (q="
                                   + std::to_string(numHeadsQ) + " kv=" + std::to_string(numHeadsKv)
                                   + ")");

    // ── Flash2 crossover heuristic ────────────────────────────────────────────
    const int seqLenQ = static_cast<int>(qTensor->dims()->Get(2));
    const int seqLenKv = static_cast<int>(kTensor->dims()->Get(2));
    HIP_KERNEL_RETURN_FALSE_IF(!useFlash2ForShape(seqLenQ, seqLenKv),
                               "shape below Flash2 crossover threshold (seq_q="
                                   + std::to_string(seqLenQ) + " seq_kv=" + std::to_string(seqLenKv)
                                   + ")");

    return true;
}

// ---------------------------------------------------------------------------
// getMaxWorkspaceSize
// ---------------------------------------------------------------------------
size_t HipFlash2FwdPlanBuilder::getMaxWorkspaceSize(const Handle& /*handle*/,
                                                    const flatbuffer_utilities::IGraph& /*opGraph*/,
                                                    const Settings& /*executionSettings*/) const
{
    // Flash-Attention 2 V7 uses only registers and LDS — no external workspace.
    return 0;
}

// ---------------------------------------------------------------------------
// initializeExecutionSettings
// ---------------------------------------------------------------------------
void HipFlash2FwdPlanBuilder::initializeExecutionSettings(
    const Handle& /*handle*/,
    const flatbuffer_utilities::IGraph& /*opGraph*/,
    const flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
    Settings& /*executionSettings*/) const
{
    // No per-execution settings needed for Flash2 V7 (all state captured at
    // buildPlan)
    HIPDNN_PLUGIN_LOG_INFO("HipFlash2FwdPlanBuilder::initializeExecutionSettings — no-op");
}

// ---------------------------------------------------------------------------
// buildPlan
// ---------------------------------------------------------------------------
void HipFlash2FwdPlanBuilder::buildPlan(const Handle& handle,
                                        const flatbuffer_utilities::IGraph& opGraph,
                                        const flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
                                        Context& executionContext) const
{
    // ── 1. Device string ─────────────────────────────────────────────────────
    // I6: fail-closed — throw on any error so the framework sees a hard failure
    // rather than leaving the execution context without a plan.
    std::string archId;
    try
    {
        archId = getDeviceString(handle.getStream());
    }
    catch(const std::exception& e)
    {
        const std::string msg
            = std::string("HipFlash2FwdPlanBuilder::buildPlan — getDeviceString: ") + e.what();
        HIPDNN_PLUGIN_LOG_ERROR(msg);
        throw std::runtime_error(msg);
    }

    // ── 2. Extract params from graph ─────────────────────────────────────────
    Flash2FwdParams params = extractParams(handle, opGraph);
    params.archString = archId;

    // ── 3. Load .co and get kernel function ──────────────────────────────────
    const std::string coPath = flash2CoPath(archId);
    const char* funcName = flash2KernelName(params.head_dim);
    if(funcName == nullptr)
    {
        const std::string msg = "HipFlash2FwdPlanBuilder::buildPlan — unsupported head_dim="
                                + std::to_string(params.head_dim);
        HIPDNN_PLUGIN_LOG_ERROR(msg);
        throw std::runtime_error(msg);
    }

    HIPDNN_PLUGIN_LOG_INFO("HipFlash2FwdPlanBuilder::buildPlan — loading " << coPath
                                                                           << " fn=" << funcName);

    auto kernelOpt = loadKernelModule(coPath, funcName);
    if(!kernelOpt)
    {
        const std::string msg
            = "HipFlash2FwdPlanBuilder::buildPlan — failed to load kernel from: " + coPath;
        HIPDNN_PLUGIN_LOG_ERROR(msg);
        throw std::runtime_error(msg);
    }

    // ── 4. Store plan in execution context ───────────────────────────────────
    executionContext.setPlan(
        std::make_unique<HipFlash2FwdPlan>(std::move(*kernelOpt), std::move(params)));
}

// ---------------------------------------------------------------------------
// getCustomKnobs
// ---------------------------------------------------------------------------
std::vector<data_objects::KnobT>
    HipFlash2FwdPlanBuilder::getCustomKnobs(const Handle& /*handle*/,
                                            const flatbuffer_utilities::IGraph& /*opGraph*/) const
{
    // V7 kernel has no tunable knobs exposed to the hipDNN planner
    return {};
}

// ---------------------------------------------------------------------------
// extractParams (private helper)
// ---------------------------------------------------------------------------
Flash2FwdParams
    HipFlash2FwdPlanBuilder::extractParams(const Handle& /*handle*/,
                                           const flatbuffer_utilities::IGraph& opGraph) const
{
