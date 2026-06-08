// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaFwdPlanBuilder.hpp"
#include "asm/AsmKernelPath.hpp"
#include "asm_fmha_v3_fwd_configs.hpp"
#include "core/Utils.hpp"
#include "plans/SdpaFwdPlan.hpp"
#include "plans/SdpaPlanUtils.hpp"

#include <cmath>
#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hip_kernel_provider_common/SdpaConfigEnumerations.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace asm_sdpa_engine
{

using namespace hip_kernel_provider_common;

static RoundingMode
    getRoundingMode(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& /*attrs*/)
{
    // TODO Cannot be specified in the graph, this will require specialized handling
    return RoundingMode::RTNE;
}

static BatchMode getBatchMode(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attrs)
{
    return (attrs.seq_len_q_tensor_uid().has_value() || attrs.seq_len_kv_tensor_uid().has_value())
               ? BatchMode::GROUP
               : BatchMode::BATCH;
}

static std::string getKernelNameKey(const std::string& archId,
                                    const std::string& dataType,
                                    int hdim_q, // NOLINT(readability-identifier-naming)
                                    int hdim_v, // NOLINT(readability-identifier-naming)
                                    plan_utils::MaskType maskType,
                                    RoundingMode bf16_cvt, // NOLINT(readability-identifier-naming)
                                    BatchMode mode,
                                    const CFG* cfgs)
{
    std::string kernelNameKey{};
    for(const auto& el : *cfgs)
    {
        const auto& cfg = el.second;
        if(cfg.arch != archId)
        {
            continue;
        }

        if(cfg.dtype == dataType && cfg.hdim_q == hdim_q && cfg.hdim_v == hdim_v
           && cfg.mask == static_cast<int>(maskType) && static_cast<int>(cfg.mode) == mode)
        {
            if(archId == "gfx950")
            {
                kernelNameKey = el.first;
                break;
            }
            if(archId == "gfx942" && cfg.bf16_cvt == bf16_cvt)
            {
                kernelNameKey = el.first;
                break;
            }
        }
    }

    return kernelNameKey;
}

static std::string getDataTypeIdentifier(hipdnn_flatbuffers_sdk::data_objects::DataType qType,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType kType,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType vType,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType oType)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    if(plan_utils::allDataTypesEqual(DataType::BFLOAT16, {qType, kType, vType, oType}))
    {
        return "bf16";
    }
    if(plan_utils::allDataTypesEqual(DataType::FP8_E4M3, {qType, kType, vType})
       && oType == DataType::BFLOAT16)
    {
        return "fp8bf16";
    }

    return "";
}

static bool isMi308Device(hipStream_t stream)
{
    int deviceId;
    auto status = hipStreamGetDevice(stream, &deviceId);
    if(status != hipSuccess)
    {
        throw std::runtime_error("hipStreamGetDevice failed with error code: "
                                 + std::to_string(status));
    }
    int chipId;
    status = hipDeviceGetAttribute(&chipId, hipDeviceAttributePciChipId, deviceId);
    if(status != hipSuccess)
    {
        throw std::runtime_error("hipDeviceGetAttribute failed with error code: "
                                 + std::to_string(status));
    }

    HIPDNN_PLUGIN_LOG_INFO("pciDeviceID  = " << std::hex << std::to_string(chipId));
    return chipId == 0x74a2 || chipId == 0x74a8 || chipId == 0x74b6 || chipId == 0x74bc;
}

static std::string getKernelCoPath(std::string coName, const std::string& archId, bool isMi308)
{
    if(archId == "gfx942")
    {
        auto pos = coName.rfind('/');
        if(isMi308)
        {
            coName = coName.substr(0, pos + 1) + "MI308/" + coName.substr(pos + 1);
        }
        else
        {
            coName = coName.substr(0, pos + 1) + "MI300/" + coName.substr(pos + 1);
        }
    }
    return asm_kernels::getAsmKernelPath(coName);
}

bool SdpaFwdPlanBuilder::isApplicable(
    const Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    // NOLINTNEXTLINE(readability-identifier-naming)
    static const char* HIP_KERNEL_LOG_PREFIX = "[SdpaFwdPlanBuilder::isApplicable] ";

    auto& nodeWrappers = opGraph.nodeWrappers();

    std::string deviceString;

    try
    {
        deviceString = hip_kernel_provider_common::getDeviceString(handle.getStream());
        HIP_KERNEL_RETURN_FALSE_IF(
            deviceString != "gfx942" && deviceString != "gfx950",
            "Device string does not match gfx942 or gfx950 (Actual value: " + deviceString + ")");
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Could not query device string: " << e.what());
        return false;
    }

    HIP_KERNEL_RETURN_FALSE_IF(nodeWrappers.size() != 1, "Graph has more than one node");
    HIP_KERNEL_RETURN_FALSE_IF(nodeWrappers.front()->attributesType()
                                   != NodeAttributes::SdpaAttributes,
                               "Node attribute type is not SdpaAttributes");

    const auto& attrs = nodeWrappers.front()->attributesAs<SdpaAttributes>();
    HIP_KERNEL_RETURN_FALSE_IF(attrs.dropout_probability().has_value()
                                   && attrs.dropout_probability().value() != 0.f,
                               "dropout_probability must be unset or zero (Actual value: "
                                   + std::to_string(attrs.dropout_probability().value()) + ")");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.alibi_mask(), "alibi_mask must be false");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.padding_mask(), "padding_mask must be false");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.attn_mask_tensor_uid(), // Change to bias
                               "attn_mask tensor not supported");

    HIP_KERNEL_RETURN_FALSE_IF(attrs.page_table_k_tensor_uid(),
                               "page_table_k tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.page_table_v_tensor_uid(),
                               "page_table_v tensor not supported");

    HIP_KERNEL_RETURN_FALSE_IF(attrs.generate_stats(), "Stats output not supported");

    const auto& tensorMap = opGraph.getTensorMap();

    const int64_t qUid = attrs.q_tensor_uid();
    const int64_t kUid = attrs.k_tensor_uid();
    const int64_t vUid = attrs.v_tensor_uid();
    const int64_t oUid = attrs.o_tensor_uid();

    auto* qTensor = tensorMap.at(qUid);
    auto* kTensor = tensorMap.at(kUid);
    auto* vTensor = tensorMap.at(vUid);
    auto* oTensor = tensorMap.at(oUid);

    HIP_KERNEL_RETURN_FALSE_IF(
        qTensor->dims()->size() != 4,
        "q tensor must be rank 4 (Actual rank: " + std::to_string(qTensor->dims()->size()) + ")");
    HIP_KERNEL_RETURN_FALSE_IF(
        vTensor->dims()->size() != 4,
        "v tensor must be rank 4 (Actual rank: " + std::to_string(vTensor->dims()->size()) + ")");
    HIP_KERNEL_RETURN_FALSE_IF(
        kTensor->dims()->size() != 4,
        "k tensor must be rank 4 (Actual rank: " + std::to_string(kTensor->dims()->size()) + ")");
    HIP_KERNEL_RETURN_FALSE_IF(
        oTensor->dims()->size() != 4,
        "o tensor must be rank 4 (Actual rank: " + std::to_string(oTensor->dims()->size()) + ")");

    HIP_KERNEL_RETURN_FALSE_IF(qTensor->data_type() != kTensor->data_type()
                                   || qTensor->data_type() != vTensor->data_type(),
                               "Input tensors must all share a type (q tensor: "
                                   + EnumNameDataType(qTensor->data_type())
                                   + ", k tensor: " + EnumNameDataType(kTensor->data_type())
                                   + ", v tensor: " + EnumNameDataType(vTensor->data_type()) + ")");

    HIP_KERNEL_RETURN_FALSE_IF(
        kTensor->dims()->Get(1) != vTensor->dims()->Get(1),
        "k tensor and v tensor must shared the same head count (Actual value: k = "
            + std::to_string(kTensor->dims()->Get(1))
            + " v = " + std::to_string(vTensor->dims()->Get(1)) + ")");

    auto dataTypeId = getDataTypeIdentifier(
        qTensor->data_type(), kTensor->data_type(), vTensor->data_type(), oTensor->data_type());

    HIP_KERNEL_RETURN_FALSE_IF(
        dataTypeId.empty(),
        "output tensor must have datatype BFLOAT16 (Actual type: "
            + EnumNameDataType(oTensor->data_type())
            + ") and input tensors must have datatype BFLOAT16 or FP8_E4M3 (Actual type: "
            + EnumNameDataType(qTensor->data_type()) + ")");

    // Classify the mask; contradictory mask attributes are an invalid-input
    // condition the engine declines rather than dispatches.
    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    try
    {
        maskType = plan_utils::getMaskType(attrs);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(std::string{HIP_KERNEL_LOG_PREFIX} + e.what());
        return false;
    }

    auto key = getKernelNameKey(deviceString,
                                dataTypeId,
                                static_cast<int>(qTensor->dims()->Get(3)),
                                static_cast<int>(vTensor->dims()->Get(3)),
                                maskType,
                                getRoundingMode(attrs),
                                getBatchMode(attrs),
                                &cfg_fmha_fwd);

    HIP_KERNEL_RETURN_FALSE_IF(key.empty(),
                               "Could not find matching kernel for parameter combination");

    return true;
}

size_t SdpaFwdPlanBuilder::getMaxWorkspaceSize(
    const Handle& /* handle */,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const Settings& /* executionSettings */) const
{
    // Forward-only kernel uses 64KB LDS internally, no external workspace needed
    // LSE (when present) is an optional output tensor, not workspace
    return 0;
}

void SdpaFwdPlanBuilder::initializeExecutionSettings(
    const Handle& /* handle */,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    Settings& /* executionSettings */) const
{
    HIPDNN_PLUGIN_LOG_ERROR("SdpaFwdPlanBuilder::initializeExecutionContext not implemented");
}

void SdpaFwdPlanBuilder::buildPlan(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    Context& executionContext) const
{

    // Get device properties
    std::string deviceString;
    bool isMi308;
    try
    {
        deviceString = hip_kernel_provider_common::getDeviceString(handle.getStream());
        isMi308 = isMi308Device(handle.getStream());
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Failed to query device properties with error: " << e.what());
        return;
    }

    // Extract SDPA attributes and tensor metadata
    auto& sdpaNode = opGraph.getNodeWrapper(0);
    auto& sdpaAttrs = sdpaNode.attributesAs<hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes>();
    auto& tensorMap = opGraph.getTensorMap();

    // Get tensor UIDs
    const int64_t qUid = sdpaAttrs.q_tensor_uid();
    const int64_t kUid = sdpaAttrs.k_tensor_uid();
    const int64_t vUid = sdpaAttrs.v_tensor_uid();
    const int64_t oUid = sdpaAttrs.o_tensor_uid();

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
    params.archString = deviceString;
    const plan_utils::MaskType maskType = plan_utils::getMaskType(sdpaAttrs);
    params.noMask = maskType == plan_utils::MaskType::NO_MASK;

    // Find matching kernel to graph
    fmha_v3_fwdConfig config;
    auto kernelKey = getKernelNameKey(
        deviceString,
        getDataTypeIdentifier(
            qTensor->data_type(), kTensor->data_type(), vTensor->data_type(), oTensor->data_type()),
        static_cast<int>(headDimQk),
        static_cast<int>(headDimV),
        maskType,
        getRoundingMode(sdpaAttrs),
        getBatchMode(sdpaAttrs),
        &cfg_fmha_fwd);

    if(kernelKey.empty())
    {
        HIPDNN_PLUGIN_LOG_ERROR("Failed to find matching kernel with error");
    }
    config = cfg_fmha_fwd.at(kernelKey);

    params.tileSizeQo = static_cast<unsigned int>(config.ts_qo);

    // Load kernel module
    auto coPath = getKernelCoPath(config.co_name, deviceString, isMi308);

    HIPDNN_PLUGIN_LOG_INFO("Using kernel with path: " << coPath);

    auto kernel = loadKernelModule(coPath, config.knl_name.c_str());
    if(!kernel)
    {
        return;
    }

    executionContext.setPlan(std::make_unique<SdpaFwdPlan>(std::move(*kernel), params));
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> SdpaFwdPlanBuilder::getCustomKnobs(
    const Handle& /* handle */,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /* opGraph */) const
{
    return {};
}

} // namespace asm_sdpa_engine
