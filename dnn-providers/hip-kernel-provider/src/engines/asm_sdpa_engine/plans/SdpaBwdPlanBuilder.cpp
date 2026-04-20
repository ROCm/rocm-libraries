// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaBwdPlanBuilder.hpp"
#include "HipKernelUtils.hpp"

#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace asm_sdpa_engine
{

bool SdpaBwdPlanBuilder::isApplicable(
    const HipKernelHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    using namespace hipdnn_data_sdk::data_objects;
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
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const HipKernelSettings& /* executionSettings */) const
{
    // TODO(Task I4): Compute actual workspace size for backward 3-kernel pipeline
    // Backward requires workspace for D buffer and dq_acc accumulator
    return 0;
}

void SdpaBwdPlanBuilder::initializeExecutionSettings(
    const HipKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    HipKernelSettings& /* executionSettings */) const
{
    HIPDNN_PLUGIN_LOG_ERROR("SdpaBwdPlanBuilder::initializeExecutionSettings not implemented");
}

void SdpaBwdPlanBuilder::buildPlan(
    const HipKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    HipKernelContext& /* executionContext */) const
{
    // TODO(Task I5): Implement backward 3-kernel plan (odo -> dqdkdv -> dq_convert)
    HIPDNN_PLUGIN_LOG_ERROR("SdpaBwdPlanBuilder::buildPlan not implemented");
}

std::vector<hipdnn_data_sdk::data_objects::KnobT> SdpaBwdPlanBuilder::getCustomKnobs(
    const HipKernelHandle& /* handle */,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /* opGraph */) const
{
    return {};
}

} // namespace asm_sdpa_engine
