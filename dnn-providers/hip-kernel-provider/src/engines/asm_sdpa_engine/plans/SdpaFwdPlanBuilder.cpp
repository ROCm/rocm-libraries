// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaFwdPlanBuilder.hpp"
#include "asm/AsmKernelPath.hpp"
#include "plans/SdpaFwdPlan.hpp"
#include <cmath>
#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace asm_sdpa_engine
{

bool SdpaFwdPlanBuilder::isApplicable(
    const HipKernelHandle& /*handle*/,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    auto& nodeWrappers = opGraph.nodeWrappers();

    if(nodeWrappers.size() != 1
       || nodeWrappers.front()->attributesType()
              != hipdnn_data_sdk::data_objects::NodeAttributes::SdpaAttributes)
    {
        return false;
    }

    // TODO: Add more expansive checks
    HIPDNN_PLUGIN_LOG_WARN("SdpaFwdPlanBuilder::isApplicable not fully implemented");

    return true;
}

size_t SdpaFwdPlanBuilder::getMaxWorkspaceSize(
    const HipKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const HipKernelSettings& /* executionSettings */) const
{
    // Forward-only kernel uses 64KB LDS internally, no external workspace needed
    // LSE (when present) is an optional output tensor, not workspace
    return 0;
}

void SdpaFwdPlanBuilder::initializeExecutionSettings(
    const HipKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    HipKernelSettings& /* executionSettings */) const
{
    HIPDNN_PLUGIN_LOG_ERROR("SdpaFwdPlanBuilder::initializeExecutionContext not implemented");
}

void SdpaFwdPlanBuilder::buildPlan(
    const HipKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    HipKernelContext& executionContext) const
{
    // Load kernel module
    std::string coPath
        = asm_kernels::getAsmKernelPath("gfx942/fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co");

    hipModule_t module;
    hipError_t err = hipModuleLoad(&module, coPath.c_str());
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR(
            "Failed to load kernel module: " << coPath << " error: " << hipGetErrorString(err));
        return;
    }

    hipFunction_t function;
    err = hipModuleGetFunction(&function, module, "_ZN5aiter24fmha_fwd_hd128_bf16_rtneE");
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Failed to get kernel function, error: " << hipGetErrorString(err));
        err = hipModuleUnload(module);
        if(err != hipSuccess)
        {
            HIPDNN_PLUGIN_LOG_ERROR(
                "Failed to unload kernel module on error, error: " << hipGetErrorString(err));
        }
        return;
    }

    // Extract SDPA attributes and tensor metadata
    auto& sdpaNode = opGraph.getNodeWrapper(0);
    auto& sdpaAttrs = sdpaNode.attributesAs<hipdnn_data_sdk::data_objects::SdpaAttributes>();
    auto& tensorMap = opGraph.getTensorMap();

    // Get tensor UIDs
    int64_t qUid = sdpaAttrs.q_tensor_uid();
    int64_t kUid = sdpaAttrs.k_tensor_uid();
    int64_t vUid = sdpaAttrs.v_tensor_uid();
    int64_t oUid = sdpaAttrs.o_tensor_uid();

    // Get tensor attributes
    auto* qTensor = tensorMap.at(qUid);
    auto* kTensor = tensorMap.at(kUid);
    auto* vTensor = tensorMap.at(vUid);
    auto* oTensor = tensorMap.at(oUid);

    // Extract dimensions from Q tensor: [B, H_q, S_q, D_qk]
    auto* qDims = qTensor->dims();
    auto batchSize = static_cast<unsigned int>(qDims->Get(0));
    auto numHeadsQ = static_cast<unsigned int>(qDims->Get(1));
    auto seqLenQ = static_cast<unsigned int>(qDims->Get(2));
    auto headDimQk = static_cast<unsigned int>(qDims->Get(3));

    // Extract dimensions from K tensor: [B, H_kv, S_kv, D_qk]
    auto* kDims = kTensor->dims();
    auto numHeadsKv = static_cast<unsigned int>(kDims->Get(1));
    auto seqLenKv = static_cast<unsigned int>(kDims->Get(2));

    // Extract dimensions from V tensor: [B, H_kv, S_kv, D_v]
    auto* vDims = vTensor->dims();
    auto headDimV = static_cast<unsigned int>(vDims->Get(3));

    // Extract strides (in elements) - Q: [B, H_q, S_q, D_qk]
    auto* qStrides = qTensor->strides();
    auto qStrideBatch = static_cast<unsigned int>(qStrides->Get(0));
    auto qStrideHead = static_cast<unsigned int>(qStrides->Get(1));
    auto qStrideSeq = static_cast<unsigned int>(qStrides->Get(2));
    auto qStrideRow = qStrideSeq; // Same as sequence stride

    // Extract strides - K: [B, H_kv, S_kv, D_qk]
    auto* kStrides = kTensor->strides();
    auto kStrideBatch = static_cast<unsigned int>(kStrides->Get(0));
    auto kStrideHead = static_cast<unsigned int>(kStrides->Get(1));
    auto kStrideSeq = static_cast<unsigned int>(kStrides->Get(2));

    // Extract strides - V: [B, H_kv, S_kv, D_v]
    auto* vStrides = vTensor->strides();
    auto vStrideBatch = static_cast<unsigned int>(vStrides->Get(0));
    auto vStrideHead = static_cast<unsigned int>(vStrides->Get(1));
    auto vStrideSeq = static_cast<unsigned int>(vStrides->Get(2));

    // Extract strides - O: [B, H_q, S_q, D_v]
    auto* oStrides = oTensor->strides();
    auto oStrideBatch = static_cast<unsigned int>(oStrides->Get(0));
    auto oStrideHead = static_cast<unsigned int>(oStrides->Get(1));
    auto oStrideSeq = static_cast<unsigned int>(oStrides->Get(2));

    // Get attention scale (default: 1/sqrt(D_qk) if not provided)
    float attnScale = 1.0f / std::sqrt(static_cast<float>(headDimQk));
    auto scaleValue = sdpaAttrs.attn_scale_value();
    if(scaleValue.has_value())
    {
        attnScale = scaleValue.value();
    }

    // Create params struct with all metadata
    SdpaFwdParams params{};
    params.qUid = qUid;
    params.kUid = kUid;
    params.vUid = vUid;
    params.oUid = oUid;
    params.batchSize = batchSize;
    params.numHeadsQ = numHeadsQ;
    params.numHeadsKv = numHeadsKv;
    params.seqLenQ = seqLenQ;
    params.seqLenKv = seqLenKv;
    params.headDimQk = headDimQk;
    params.headDimV = headDimV;
    params.qStrideSeq = qStrideSeq;
    params.qStrideRow = qStrideRow;
    params.qStrideHead = qStrideHead;
    params.qStrideBatch = qStrideBatch;
    params.kStrideSeq = kStrideSeq;
    params.kStrideHead = kStrideHead;
    params.kStrideBatch = kStrideBatch;
    params.vStrideSeq = vStrideSeq;
    params.vStrideHead = vStrideHead;
    params.vStrideBatch = vStrideBatch;
    params.oStrideSeq = oStrideSeq;
    params.oStrideHead = oStrideHead;
    params.oStrideBatch = oStrideBatch;
    params.attnScale = attnScale;

    executionContext.setPlan(std::make_unique<SdpaFwdPlan>(module, function, params));
}

std::vector<hipdnn_data_sdk::data_objects::KnobT> SdpaFwdPlanBuilder::getCustomKnobs(
    const HipKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */) const
{
    return {};
}

} // namespace asm_sdpa_engine
