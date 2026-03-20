// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaKernelPlanBuilder.hpp"
#include "SdpaKernelPlan.hpp"
#include "asm/AsmKernelPath.hpp"
#include <cmath>
#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace sdpa_kernel_provider
{

bool SdpaKernelPlanBuilder::isApplicable(
    const SdpaKernelHandle& /*handle*/,
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
    HIPDNN_PLUGIN_LOG_WARN("SdpaKernelPlanBuilder::isApplicable not fully implemented");

    return true;
}

size_t SdpaKernelPlanBuilder::getMaxWorkspaceSize(
    const SdpaKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const SdpaKernelSettings& /* executionSettings */) const
{
    // Forward-only kernel uses 64KB LDS internally, no external workspace needed
    // LSE (when present) is an optional output tensor, not workspace
    return 0;
}

void SdpaKernelPlanBuilder::initializeExecutionSettings(
    const SdpaKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    SdpaKernelSettings& /* executionSettings */) const
{
    HIPDNN_PLUGIN_LOG_ERROR("SdpaKernelPlanBuilder::initializeExecutionContext not implemented");
}

void SdpaKernelPlanBuilder::buildPlan(
    const SdpaKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    SdpaKernelContext& executionContext) const
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
    size_t batchSize = static_cast<size_t>(qDims->Get(0));
    size_t numHeadsQ = static_cast<size_t>(qDims->Get(1));
    size_t seqLenQ = static_cast<size_t>(qDims->Get(2));
    size_t headDimQk = static_cast<size_t>(qDims->Get(3));

    // Extract dimensions from K tensor: [B, H_kv, S_kv, D_qk]
    auto* kDims = kTensor->dims();
    size_t numHeadsKv = static_cast<size_t>(kDims->Get(1));
    size_t seqLenKv = static_cast<size_t>(kDims->Get(2));

    // Extract dimensions from V tensor: [B, H_kv, S_kv, D_v]
    auto* vDims = vTensor->dims();
    size_t headDimV = static_cast<size_t>(vDims->Get(3));

    // Extract strides (in elements) - Q: [B, H_q, S_q, D_qk]
    auto* qStrides = qTensor->strides();
    size_t qStrideBatch = static_cast<size_t>(qStrides->Get(0));
    size_t qStrideHead = static_cast<size_t>(qStrides->Get(1));
    size_t qStrideSeq = static_cast<size_t>(qStrides->Get(2));
    size_t qStrideRow = qStrideSeq; // Same as sequence stride

    // Extract strides - K: [B, H_kv, S_kv, D_qk]
    auto* kStrides = kTensor->strides();
    size_t kStrideBatch = static_cast<size_t>(kStrides->Get(0));
    size_t kStrideHead = static_cast<size_t>(kStrides->Get(1));
    size_t kStrideSeq = static_cast<size_t>(kStrides->Get(2));

    // Extract strides - V: [B, H_kv, S_kv, D_v]
    auto* vStrides = vTensor->strides();
    size_t vStrideBatch = static_cast<size_t>(vStrides->Get(0));
    size_t vStrideHead = static_cast<size_t>(vStrides->Get(1));
    size_t vStrideSeq = static_cast<size_t>(vStrides->Get(2));

    // Extract strides - O: [B, H_q, S_q, D_v]
    auto* oStrides = oTensor->strides();
    size_t oStrideBatch = static_cast<size_t>(oStrides->Get(0));
    size_t oStrideHead = static_cast<size_t>(oStrides->Get(1));
    size_t oStrideSeq = static_cast<size_t>(oStrides->Get(2));

    // Get attention scale (default: 1/sqrt(D_qk) if not provided)
    float attnScale = 1.0f / std::sqrt(static_cast<float>(headDimQk));
    auto scaleValue = sdpaAttrs.attn_scale_value();
    if(scaleValue.has_value())
    {
        attnScale = scaleValue.value();
    }

    // Create plan with all metadata
    executionContext.setPlan(std::make_unique<SdpaKernelPlan>(module,
                                                              function,
                                                              qUid,
                                                              kUid,
                                                              vUid,
                                                              oUid,
                                                              batchSize,
                                                              numHeadsQ,
                                                              numHeadsKv,
                                                              seqLenQ,
                                                              seqLenKv,
                                                              headDimQk,
                                                              headDimV,
                                                              qStrideSeq,
                                                              qStrideRow,
                                                              qStrideHead,
                                                              qStrideBatch,
                                                              kStrideSeq,
                                                              kStrideHead,
                                                              kStrideBatch,
                                                              vStrideSeq,
                                                              vStrideHead,
                                                              vStrideBatch,
                                                              oStrideSeq,
                                                              oStrideHead,
                                                              oStrideBatch,
                                                              attnScale));
}

std::vector<hipdnn_data_sdk::data_objects::KnobT> SdpaKernelPlanBuilder::getCustomKnobs(
    const SdpaKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */) const
{
    return {};
}

} // namespace sdpa_kernel_provider
