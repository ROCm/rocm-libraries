// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaBwdPlanBuilder.hpp"
#include "HipKernelUtils.hpp"
#include "asm/AsmKernelPath.hpp"
#include "plans/SdpaBwdPlan.hpp"

#include <cmath>
#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <utility>

namespace asm_sdpa_engine
{

bool SdpaBwdPlanBuilder::isApplicable(
    const HipKernelHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    // NOLINTNEXTLINE(readability-identifier-naming)
    static const char* HIP_KERNEL_LOG_PREFIX = "[SdpaBwdPlanBuilder::isApplicable] ";

    auto& nodeWrappers = opGraph.nodeWrappers();

    try
    {
        auto deviceString = hip_kernel_provider_common::getDeviceString(handle.getStream());
        HIP_KERNEL_RETURN_FALSE_IF(
            deviceString != "gfx942",
            "Device string does not match gfx942 (Actual value: " + deviceString + ")");
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Could not query device string: " << e.what());
        return false;
    }

    HIP_KERNEL_RETURN_FALSE_IF(nodeWrappers.size() != 1, "Graph has more than one node");
    HIP_KERNEL_RETURN_FALSE_IF(nodeWrappers.front()->attributesType()
                                   != NodeAttributes::SdpaBackwardAttributes,
                               "Node attribute type is not SdpaBackwardAttributes");

    const auto& attrs = nodeWrappers.front()->attributesAs<SdpaBackwardAttributes>();

    // --- POC restrictions: no masking, no dropout, no variable-length sequences ---
    HIP_KERNEL_RETURN_FALSE_IF(attrs.causal_mask(), "causal_mask must be false");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.causal_mask_bottom_right(),
                               "causal_mask_bottom_right must be false");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.left_bound().has_value(), "left_bound must be unset");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.right_bound().has_value(), "right_bound must be unset");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.dropout_probability().has_value()
                                   && attrs.dropout_probability().value() != 0.f,
                               "dropout_probability must be unset or zero (Actual value: "
                                   + std::to_string(attrs.dropout_probability().value()) + ")");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.alibi_mask(), "alibi_mask must be false");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.padding_mask(), "padding_mask must be false");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.seq_len_q_tensor_uid(), "seq_len_q tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.attn_mask_tensor_uid(), "attn_mask tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.seed_tensor_uid(), "seed tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.offset_tensor_uid(), "offset tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.dropout_mask_tensor_uid(),
                               "dropout_mask tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.dbias_tensor_uid(), "dbias tensor not supported");

    // --- Validate required tensors ---
    const auto& tensorMap = opGraph.getTensorMap();

    // Required input tensor UIDs
    int64_t qUid = attrs.q_tensor_uid();
    int64_t kUid = attrs.k_tensor_uid();
    int64_t vUid = attrs.v_tensor_uid();
    int64_t oUid = attrs.o_tensor_uid();
    int64_t doUid = attrs.do_tensor_uid();
    int64_t statsUid = attrs.stats_tensor_uid();

    // Required output tensor UIDs
    int64_t dqUid = attrs.dq_tensor_uid();
    int64_t dkUid = attrs.dk_tensor_uid();
    int64_t dvUid = attrs.dv_tensor_uid();

    // Q tensor: BF16, rank-4, head dim 128
    auto* qTensor = tensorMap.at(qUid);
    HIP_KERNEL_RETURN_FALSE_IF(qTensor->data_type() != DataType::BFLOAT16,
                               "q tensor datatype must be BF16 (Actual type: "
                                   + EnumNameDataType(qTensor->data_type()) + ")");
    HIP_KERNEL_RETURN_FALSE_IF(
        qTensor->dims()->size() != 4,
        "q tensor must be rank 4 (Actual rank: " + std::to_string(qTensor->dims()->size()) + ")");
    HIP_KERNEL_RETURN_FALSE_IF(qTensor->dims()->Get(3) != 128,
                               "q tensor head dimension must be 128 (Actual value: "
                                   + std::to_string(qTensor->dims()->Get(3)) + ")");

    // K tensor: BF16
    auto* kTensor = tensorMap.at(kUid);
    HIP_KERNEL_RETURN_FALSE_IF(kTensor->data_type() != DataType::BFLOAT16,
                               "k tensor datatype must be BF16 (Actual type: "
                                   + EnumNameDataType(kTensor->data_type()) + ")");

    // V tensor: BF16
    auto* vTensor = tensorMap.at(vUid);
    HIP_KERNEL_RETURN_FALSE_IF(vTensor->data_type() != DataType::BFLOAT16,
                               "v tensor datatype must be BF16 (Actual type: "
                                   + EnumNameDataType(vTensor->data_type()) + ")");

    // O tensor: BF16
    auto* oTensor = tensorMap.at(oUid);
    HIP_KERNEL_RETURN_FALSE_IF(oTensor->data_type() != DataType::BFLOAT16,
                               "o tensor datatype must be BF16 (Actual type: "
                                   + EnumNameDataType(oTensor->data_type()) + ")");

    // dO tensor: BF16
    auto* doTensor = tensorMap.at(doUid);
    HIP_KERNEL_RETURN_FALSE_IF(doTensor->data_type() != DataType::BFLOAT16,
                               "do tensor datatype must be BF16 (Actual type: "
                                   + EnumNameDataType(doTensor->data_type()) + ")");

    // STATS tensor: FP32
    auto* statsTensor = tensorMap.at(statsUid);
    HIP_KERNEL_RETURN_FALSE_IF(statsTensor->data_type() != DataType::FLOAT,
                               "stats tensor datatype must be FP32 (Actual type: "
                                   + EnumNameDataType(statsTensor->data_type()) + ")");

    // dQ tensor: BF16
    auto* dqTensor = tensorMap.at(dqUid);
    HIP_KERNEL_RETURN_FALSE_IF(dqTensor->data_type() != DataType::BFLOAT16,
                               "dq tensor datatype must be BF16 (Actual type: "
                                   + EnumNameDataType(dqTensor->data_type()) + ")");

    // dK tensor: BF16
    auto* dkTensor = tensorMap.at(dkUid);
    HIP_KERNEL_RETURN_FALSE_IF(dkTensor->data_type() != DataType::BFLOAT16,
                               "dk tensor datatype must be BF16 (Actual type: "
                                   + EnumNameDataType(dkTensor->data_type()) + ")");

    // dV tensor: BF16
    auto* dvTensor = tensorMap.at(dvUid);
    HIP_KERNEL_RETURN_FALSE_IF(dvTensor->data_type() != DataType::BFLOAT16,
                               "dv tensor datatype must be BF16 (Actual type: "
                                   + EnumNameDataType(dvTensor->data_type()) + ")");

    return true;
}

size_t SdpaBwdPlanBuilder::getMaxWorkspaceSize(
    const HipKernelHandle& /* handle */,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const HipKernelSettings& /* executionSettings */) const
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    const auto& attrs = opGraph.nodeWrappers().front()->attributesAs<SdpaBackwardAttributes>();
    const auto& tensorMap = opGraph.getTensorMap();
    const auto* qTensor = tensorMap.at(attrs.q_tensor_uid());

    // Q tensor layout is [B, H_q, S_q, D_qk]
    auto batch = static_cast<size_t>(qTensor->dims()->Get(0));
    auto headsQ = static_cast<size_t>(qTensor->dims()->Get(1));
    auto seqLenQ = static_cast<size_t>(qTensor->dims()->Get(2));
    auto headDim = static_cast<size_t>(qTensor->dims()->Get(3));

    return sdpaBwdDBufferSize(batch, headsQ, seqLenQ)
           + sdpaBwdDqAccBufferSize(batch, headsQ, seqLenQ, headDim);
}

void SdpaBwdPlanBuilder::initializeExecutionSettings(
    const HipKernelHandle& /* handle */,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    HipKernelSettings& /* executionSettings */) const
{
    HIPDNN_PLUGIN_LOG_ERROR("SdpaBwdPlanBuilder::initializeExecutionSettings not implemented");
}

void SdpaBwdPlanBuilder::buildPlan(
    const HipKernelHandle& /* handle */,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    HipKernelContext& executionContext) const
{
    // -------------------------------------------------------------------------
    // 1. Load 3 kernel modules and get kernel functions from modules
    // -------------------------------------------------------------------------
    std::string odoCoPath
        = asm_kernels::getAsmKernelPath("gfx942/fmha_v3_bwd/MI300/bwd_hd128_odo_bf16.co");
    std::string dqdkdvCoPath = asm_kernels::getAsmKernelPath(
        "gfx942/fmha_v3_bwd/MI300/bwd_hd128_bf16_a32_rtne_psskddv.co");
    std::string postCoPath = asm_kernels::getAsmKernelPath(
        "gfx942/fmha_v3_bwd/MI300/bwd_hd128_dq_convert_bf16_rtne.co");

    auto odoKernel = loadKernelModule(odoCoPath, "_ZN5aiter23fmha_bwd_hd128_odo_bf16E");
    if(!odoKernel)
    {
        return;
    }

    auto dqdkdvKernel
        = loadKernelModule(dqdkdvCoPath, "_ZN5aiter36fmha_bwd_hd128_bf16_a32_rtne_psskddvE");
    if(!dqdkdvKernel)
    {
        return;
    }

    auto postKernel
        = loadKernelModule(postCoPath, "_ZN5aiter35fmha_bwd_hd128_dq_convert_bf16_rtneE");
    if(!postKernel)
    {
        return;
    }

    // -------------------------------------------------------------------------
    // 2. Extract SDPA backward attributes and tensor metadata from graph
    // -------------------------------------------------------------------------
    auto& sdpaNode = opGraph.getNodeWrapper(0);
    auto& sdpaAttrs
        = sdpaNode.attributesAs<hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes>();
    auto& tensorMap = opGraph.getTensorMap();

    // Tensor UIDs
    int64_t qUid = sdpaAttrs.q_tensor_uid();
    int64_t kUid = sdpaAttrs.k_tensor_uid();
    int64_t vUid = sdpaAttrs.v_tensor_uid();
    int64_t oUid = sdpaAttrs.o_tensor_uid();
    int64_t doUid = sdpaAttrs.do_tensor_uid();
    int64_t statsUid = sdpaAttrs.stats_tensor_uid();
    int64_t dqUid = sdpaAttrs.dq_tensor_uid();
    int64_t dkUid = sdpaAttrs.dk_tensor_uid();
    int64_t dvUid = sdpaAttrs.dv_tensor_uid();

    // Tensor objects
    auto* qTensor = tensorMap.at(qUid);
    auto* kTensor = tensorMap.at(kUid);
    auto* vTensor = tensorMap.at(vUid);
    auto* oTensor = tensorMap.at(oUid);
    auto* doTensor = tensorMap.at(doUid);
    auto* statsTensor = tensorMap.at(statsUid);
    auto* dqTensor = tensorMap.at(dqUid);
    auto* dkTensor = tensorMap.at(dkUid);
    auto* dvTensor = tensorMap.at(dvUid);

    // Dimensions from Q: [B, H_q, S_q, D_qk]
    auto* qDims = qTensor->dims();
    auto batchSize = static_cast<unsigned int>(qDims->Get(0));
    auto numHeadsQ = static_cast<unsigned int>(qDims->Get(1));
    auto seqLenQ = static_cast<unsigned int>(qDims->Get(2));
    auto headDimQk = static_cast<unsigned int>(qDims->Get(3));

    // Dimensions from K: [B, H_kv, S_kv, D_qk]
    auto numHeadsKv = static_cast<unsigned int>(kTensor->dims()->Get(1));
    auto seqLenKv = static_cast<unsigned int>(kTensor->dims()->Get(2));

    // Dimensions from V: [B, H_kv, S_kv, D_v]
    auto headDimV = static_cast<unsigned int>(vTensor->dims()->Get(3));

    // -------------------------------------------------------------------------
    // 3. Extract strides (in elements) from tensor metadata
    // -------------------------------------------------------------------------
    // Q: [B, H_q, S_q, D_qk]
    auto* qStrides = qTensor->strides();
    auto qStrideBatch = static_cast<unsigned int>(qStrides->Get(0));
    auto qStrideHead = static_cast<unsigned int>(qStrides->Get(1));
    auto qStrideSeq = static_cast<unsigned int>(qStrides->Get(2));

    // K: [B, H_kv, S_kv, D_qk]
    auto* kStrides = kTensor->strides();
    auto kStrideBatch = static_cast<unsigned int>(kStrides->Get(0));
    auto kStrideHead = static_cast<unsigned int>(kStrides->Get(1));
    auto kStrideSeq = static_cast<unsigned int>(kStrides->Get(2));

    // V: [B, H_kv, S_kv, D_v]
    auto* vStrides = vTensor->strides();
    auto vStrideBatch = static_cast<unsigned int>(vStrides->Get(0));
    auto vStrideHead = static_cast<unsigned int>(vStrides->Get(1));
    auto vStrideSeq = static_cast<unsigned int>(vStrides->Get(2));

    // O: [B, H_q, S_q, D_v]
    auto* oStrides = oTensor->strides();
    auto oStrideBatch = static_cast<unsigned int>(oStrides->Get(0));
    auto oStrideHead = static_cast<unsigned int>(oStrides->Get(1));
    auto oStrideSeq = static_cast<unsigned int>(oStrides->Get(2));

    // dO: [B, H_q, S_q, D_v]
    auto* doStrides = doTensor->strides();
    auto doStrideBatch = static_cast<unsigned int>(doStrides->Get(0));
    auto doStrideHead = static_cast<unsigned int>(doStrides->Get(1));
    auto doStrideSeq = static_cast<unsigned int>(doStrides->Get(2));

    // dQ: [B, H_q, S_q, D_qk]
    auto* dqStrides = dqTensor->strides();
    auto dqStrideBatch = static_cast<unsigned int>(dqStrides->Get(0));
    auto dqStrideHead = static_cast<unsigned int>(dqStrides->Get(1));
    auto dqStrideSeq = static_cast<unsigned int>(dqStrides->Get(2));

    // dK: [B, H_kv, S_kv, D_qk]
    auto* dkStrides = dkTensor->strides();
    auto dkStrideBatch = static_cast<unsigned int>(dkStrides->Get(0));
    auto dkStrideHead = static_cast<unsigned int>(dkStrides->Get(1));
    auto dkStrideSeq = static_cast<unsigned int>(dkStrides->Get(2));

    // dV: [B, H_kv, S_kv, D_v]
    auto* dvStrides = dvTensor->strides();
    auto dvStrideBatch = static_cast<unsigned int>(dvStrides->Get(0));
    auto dvStrideHead = static_cast<unsigned int>(dvStrides->Get(1));
    auto dvStrideSeq = static_cast<unsigned int>(dvStrides->Get(2));

    // Stats (LSE): [B, H_q, S_q] — rank 3, FP32
    auto* statsStrides = statsTensor->strides();
    auto statsStrideBatch = static_cast<unsigned int>(statsStrides->Get(0));
    auto statsStrideHead = static_cast<unsigned int>(statsStrides->Get(1));

    // -------------------------------------------------------------------------
    // 4. Attention scale
    // -------------------------------------------------------------------------
    float attnScale = 1.0f / std::sqrt(static_cast<float>(headDimQk));
    auto scaleValue = sdpaAttrs.attn_scale_value();
    if(scaleValue.has_value())
    {
        attnScale = scaleValue.value();
    }

    // -------------------------------------------------------------------------
    // 5. Build params and create plan
    // -------------------------------------------------------------------------
    SdpaBwdParams params{};
    params.qUid = qUid;
    params.kUid = kUid;
    params.vUid = vUid;
    params.oUid = oUid;
    params.doUid = doUid;
    params.statsUid = statsUid;
    params.dqUid = dqUid;
    params.dkUid = dkUid;
    params.dvUid = dvUid;

    params.batchSize = batchSize;
    params.numHeadsQ = numHeadsQ;
    params.numHeadsKv = numHeadsKv;
    params.seqLenQ = seqLenQ;
    params.seqLenKv = seqLenKv;
    params.headDimQk = headDimQk;
    params.headDimV = headDimV;

    params.qStrideSeq = qStrideSeq;
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
    params.doStrideSeq = doStrideSeq;
    params.doStrideHead = doStrideHead;
    params.doStrideBatch = doStrideBatch;
    params.dqStrideSeq = dqStrideSeq;
    params.dqStrideHead = dqStrideHead;
    params.dqStrideBatch = dqStrideBatch;
    params.dkStrideSeq = dkStrideSeq;
    params.dkStrideHead = dkStrideHead;
    params.dkStrideBatch = dkStrideBatch;
    params.dvStrideSeq = dvStrideSeq;
    params.dvStrideHead = dvStrideHead;
    params.dvStrideBatch = dvStrideBatch;
    params.statsStrideHead = statsStrideHead;
    params.statsStrideBatch = statsStrideBatch;
    params.attnScale = attnScale;

    executionContext.setPlan(std::make_unique<SdpaBwdPlan>(
        std::move(*odoKernel), std::move(*dqdkdvKernel), std::move(*postKernel), params));
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> SdpaBwdPlanBuilder::getCustomKnobs(
    const HipKernelHandle& /* handle */,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /* opGraph */) const
{
    return {};
}

} // namespace asm_sdpa_engine
