// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaFwdPlanBuilder.hpp"
#include "asm/AsmKernelPath.hpp"
#include "asm_fmha_v3_fwd_configs.hpp"
#include "core/Utils.hpp"
#include "plans/SdpaFwdPlan.hpp"
#include "plans/SdpaPlanUtils.hpp"

#include <cmath>
#include <string>
#include <utility>

#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/SdpaConfigEnumerations.hpp>
#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/ApplicabilityUtils.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/DeviceQuery.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace asm_sdpa_engine
{

using namespace hip_kernel_provider_common;

constexpr auto SDPA_SEQ_AXIS
    = static_cast<flatbuffers::uoffset_t>(hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS);

static RoundingMode
    getRoundingMode(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& /*attrs*/)
{
    // TODO Cannot be specified in the graph, this will require specialized handling
    return RoundingMode::RTNE;
}

// Selects the bf16-output conversion mode used for kernel lookup. FP8 forward
// kernels are only shipped with the RTNA conversion (bf16_cvt=1 in the kernel
// CSV), so fp8 must request RTNA; bf16 uses the default rounding.
static RoundingMode
    getBf16ConvertMode(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attrs,
                       const std::string& dataTypeId)
{
    if(dataTypeId == "fp8bf16")
    {
        return RoundingMode::RTNA;
    }
    return getRoundingMode(attrs);
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
    // gfx942/MI300 fp8 is the FNUZ variant (exp bias 8, NaN=0x80, no negative zero),
    // which is the exact encoding the vendored AITER gfx942 fp8 forward kernels
    // dequantize. OCP FP8_E4M3 / E4M3FN (bias 7, gfx950/MI350) is a different,
    // non-bit-compatible encoding: the same byte decodes to a different value, so
    // feeding OCP bytes to these kernels would silently produce wrong results. Only
    // FP8_E4M3_FNUZ is accepted here; OCP fp8 is diagnosed and declined in
    // isApplicable until the gfx950 OCP fp8 forward kernels land.
    if(plan_utils::allDataTypesEqual(DataType::FP8_E4M3_FNUZ, {qType, kType, vType})
       && oType == DataType::BFLOAT16)
    {
        return "fp8bf16";
    }

    return "";
}

// A per-tensor (scalar) descale has a single element — dims collapse to [1,...,1].
// The forward kernel-arg builder only wires scalar descales (all s_descale_* strides
// are zero), so isApplicable must reject any non-scalar descale rather than silently
// reading descale[0] for every batch/head.
static bool isScalarTensor(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    if(dims == nullptr)
    {
        return false;
    }
    int64_t elementCount = 1;
    for(flatbuffers::uoffset_t i = 0; i < dims->size(); ++i)
    {
        elementCount *= dims->Get(i);
    }
    return elementCount == 1;
}

// AITER group mode reads INT32 [B+1] token seqstarts.
static bool isTokenSeqstartTensor(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid,
    int64_t batch)
{
    const auto it = tensorMap.find(uid);
    if(it == tensorMap.end())
    {
        return false;
    }
    const auto* tensor = it->second;
    if(tensor->virtual_() || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(tensor)
       || tensor->data_type() != hipdnn_flatbuffers_sdk::data_objects::DataType::INT32)
    {
        return false;
    }
    const auto* dims = tensor->dims();
    const auto* strides = tensor->strides();
    return dims != nullptr && strides != nullptr && dims->size() == 4 && strides->size() == 4
           && dims->Get(0) == batch + 1 && dims->Get(1) == 1 && dims->Get(2) == 1
           && dims->Get(3) == 1 && strides->Get(0) == 1;
}

// hipDNN addresses a ragged element at offset[b] * multiplier + strides . (h, s, d), while
// AITER uses seqstart[b] * seq_stride + strides . (h, s, d) with one seqstart for Q/O and one
// for K/V. The two agree for any stride permutation under the conditions below.
static bool hasAiterCompatibleRaggedAddressing(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& qTensor,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& kTensor,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& vTensor,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& oTensor)
{
    // NOLINTNEXTLINE(readability-identifier-naming)
    static const char* HIP_KERNEL_LOG_PREFIX = "[SdpaFwdPlanBuilder::isApplicable] ";

    const int64_t qoOffsetUid = qTensor.ragged_offset_tensor_uid().value();
    const int64_t kvOffsetUid = kTensor.ragged_offset_tensor_uid().value();
    HIP_KERNEL_RETURN_FALSE_IF(oTensor.ragged_offset_tensor_uid().value() != qoOffsetUid,
                               "Q and O must share a ragged offset tensor");
    HIP_KERNEL_RETURN_FALSE_IF(vTensor.ragged_offset_tensor_uid().value() != kvOffsetUid,
                               "K and V must share a ragged offset tensor");

    for(const auto& [label, tensor] : {std::pair{"q", &qTensor},
                                       std::pair{"k", &kTensor},
                                       std::pair{"v", &vTensor},
                                       std::pair{"o", &oTensor}})
    {
        HIP_KERNEL_RETURN_FALSE_IF(tensor->strides() == nullptr || tensor->strides()->size() != 4,
                                   std::string(label) + " tensor must have rank 4 strides");
        HIP_KERNEL_RETURN_FALSE_IF(
            tensor->ragged_offset_multiplier() != tensor->strides()->Get(SDPA_SEQ_AXIS),
            std::string(label) + " tensor ragged_offset_multiplier must equal its sequence "
                + "stride (Actual multiplier: " + std::to_string(tensor->ragged_offset_multiplier())
                + ", sequence stride: " + std::to_string(tensor->strides()->Get(SDPA_SEQ_AXIS))
                + ")");
    }

    const int64_t batch = qTensor.dims()->Get(0);
    HIP_KERNEL_RETURN_FALSE_IF(
        !isTokenSeqstartTensor(tensorMap, qoOffsetUid, batch)
            || !isTokenSeqstartTensor(tensorMap, kvOffsetUid, batch),
        "Ragged offset tensors must be non-virtual INT32 [B+1, 1, 1, 1] device tensors");
    return true;
}

static bool isMi308Device(hipStream_t stream)
{
    // Seeded, not left indeterminate: a concrete stream's hipStreamGetDevice query can
    // report hipSuccess without writing the out-parameter. Default tokens query the
    // current device; either path must produce an ordinal.
    int deviceId = -1;
    auto status = hipdnn_plugin_sdk::getDeviceFromStream(stream, &deviceId);
    if(status != hipSuccess || deviceId < 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "Stream device query produced no device ordinal, error code: " + std::to_string(status)
                + ", ordinal: " + std::to_string(deviceId));
    }
    int chipId;
    status = hipDeviceGetAttribute(&chipId, hipDeviceAttributePciChipId, deviceId);
    if(status != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "hipDeviceGetAttribute failed with error code: " + std::to_string(status));
    }

    HIPDNN_PLUGIN_LOG_INFO("pciDeviceID  = " << std::hex << std::to_string(chipId));
    return chipId == 0x74a2 || chipId == 0x74a8 || chipId == 0x74b6 || chipId == 0x74bc;
}

// True when the SDPA attributes request stats (LSE) output.
// Shared by isApplicable and buildPlan so the predicate never drifts.
static bool hasStatsOutput(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attrs)
{
    return attrs.generate_stats().value_or(false);
}

// Byte sizes the forward kernargs scale element strides by. The output stays
// BF16 and the stats/LSE tensor is FP32. Q/K/V take the size of the resolved
// input type, which is one byte for fp8 and two for bf16.
constexpr int64_t K_BF16_BYTES = 2;
constexpr int64_t K_FP32_BYTES = 4;
constexpr int64_t K_FP8_BYTES = 1;

// Bytes per Q/K/V element for a resolved forward data-type id. One definition
// so the stride bounds isApplicable applies and the inBytesPerElement buildPlan
// hands the kernarg builder cannot drift apart.
static int64_t fwdInputBytesPerElement(const std::string& dataTypeId)
{
    return (dataTypeId == "fp8bf16") ? K_FP8_BYTES : K_BF16_BYTES;
}

// Validate that every forward dimension fits the 32-bit field that carries it.
// Batch and the head counts reach the kernel through a uint32 kernarg field or
// an unsigned grid dimension. The head dims are narrowed to int for the kernel
// registry lookup, and the sequence lengths drive `(seqLenQ + tileSizeQo - 1) /
// tileSizeQo` grid math evaluated in unsigned int, so both of those take the
// signed limit. Checked before the first narrowing cast so the engine declines
// instead of dispatching a kernel on a truncated shape.
static bool
    wouldFwdDimsFitKernargFields(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& q,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& k,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& v)
{
    bool ok = true;
    ok &= plan_utils::dimFitsU32("batch", q.dims()->Get(0));
    ok &= plan_utils::dimFitsU32("nhead_q", q.dims()->Get(1));
    ok &= plan_utils::dimFitsU32("nhead_kv", k.dims()->Get(1));
    ok &= plan_utils::dimFitsI32("seqlen_q", q.dims()->Get(2));
    ok &= plan_utils::dimFitsI32("seqlen_kv", k.dims()->Get(2));
    ok &= plan_utils::dimFitsI32("hdim_qk", q.dims()->Get(3));
    ok &= plan_utils::dimFitsI32("hdim_v", v.dims()->Get(3));
    return ok;
}

// Validate that every forward-pass byte stride fits in uint32_t.  The ASM
// kernarg struct stores strides as uint32 so values that overflow silently
// truncate, producing wrong results.  Checked early in isApplicable so the
// engine declines rather than dispatching with bad strides. `inputBytes` is
// the resolved Q/K/V element size, so an fp8 descriptor is bounded at the one
// byte the kernarg builder actually scales by; O stays BF16.
static bool wouldFwdByteStridesFitUint32(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& q,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& k,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& v,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& o,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* stats,
    int64_t inputBytes)
{
    auto checkTensor = [](const char* prefix,
                          const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& t,
                          int64_t elementBytes) {
        const auto* s = t.strides();
        bool ok = true;
        ok &= plan_utils::byteStrideFitsU32(
            (std::string("batch_stride_") + prefix).c_str(), s->Get(0), elementBytes);
        ok &= plan_utils::byteStrideFitsU32(
            (std::string("nhead_stride_") + prefix).c_str(), s->Get(1), elementBytes);
        ok &= plan_utils::byteStrideFitsU32(
            (std::string("stride_") + prefix).c_str(), s->Get(2), elementBytes);
        return ok;
    };

    bool ok = true;
    ok &= checkTensor("q", q, inputBytes);
    ok &= checkTensor("k", k, inputBytes);
    ok &= checkTensor("v", v, inputBytes);
    ok &= checkTensor("o", o, K_BF16_BYTES);

    if(stats != nullptr)
    {
        // LSE/stats: rank 3 [B, H_q, S_q] or rank 4 [B, H_q, S_q, 1] in FP32;
        // only batch and head strides reach the kernel.
        const auto* statsStrides = stats->strides();
        ok &= plan_utils::byteStrideFitsU32("batch_stride_lse", statsStrides->Get(0), K_FP32_BYTES);
        ok &= plan_utils::byteStrideFitsU32("nhead_stride_lse", statsStrides->Get(1), K_FP32_BYTES);
    }

    return ok;
}

static std::string getKernelTocKey(std::string coName, const std::string& archId, bool isMi308)
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
    return asm_kernels::getAsmKernelTocKey(coName);
}

bool SdpaFwdPlanBuilder::isApplicable(
    const Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    using namespace hipdnn_flatbuffers_sdk::utilities;
    // NOLINTNEXTLINE(readability-identifier-naming)
    static const char* HIP_KERNEL_LOG_PREFIX = "[SdpaFwdPlanBuilder::isApplicable] ";

    // Execute-time override shapes can diverge from the compile-time dims this
    // builder matched exactly; the family serves fixed prebuilt shapes, so decline
    // rather than risk a mismatch (RFC 0008 §4.6).
    HIP_KERNEL_RETURN_FALSE_IF(opGraph.getGraph().is_override_shape_enabled(),
                               "Graph has override shapes enabled");
    HIP_KERNEL_RETURN_FALSE_IF(
        !hip_kernel_provider::core::utils::hasOnly16ByteAlignedTensors(opGraph.getTensorMap()),
        "All tensors must be aligned to a multiple of 16 bytes");

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

    HIP_KERNEL_RETURN_FALSE_IF(nodeWrappers.front()->computeDataType() != DataType::FLOAT,
                               "Compute data type must be FLOAT");

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
    // Attention sinks join the softmax denominator; the vendored kernels have no sink input.
    HIP_KERNEL_RETURN_FALSE_IF(attrs.sink_token_tensor_uid(), "sink_token tensor not supported");

    // Accept scale_tensor_uid only when it is a runtime pass-by-value scalar
    // (RFC 0016).  Non-pass-by-value scale tensors are not supported.
    if(attrs.scale_tensor_uid().has_value())
    {
        const auto& tensorMap = opGraph.getTensorMap();
        const auto scaleIt = tensorMap.find(attrs.scale_tensor_uid().value());
        HIP_KERNEL_RETURN_FALSE_IF(scaleIt == tensorMap.end(),
                                   "scale_tensor_uid not found in tensor map");
        HIP_KERNEL_RETURN_FALSE_IF(
            !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(scaleIt->second),
            "scale tensor must be pass-by-value (compile-time constant or runtime)");
    }

    const auto& tensorMap = opGraph.getTensorMap();

    const int64_t qUid = attrs.q_tensor_uid();
    const int64_t kUid = attrs.k_tensor_uid();
    const int64_t vUid = attrs.v_tensor_uid();
    const int64_t oUid = attrs.o_tensor_uid();

    auto* qTensor = tensorMap.at(qUid);
    auto* kTensor = tensorMap.at(kUid);
    auto* vTensor = tensorMap.at(vUid);
    auto* oTensor = tensorMap.at(oUid);

    BatchMode batchMode;
    {
        const auto unsupportedRaggedTensorIds
            = listUnsupportedRaggedTensorIds(tensorMap, {qUid, kUid, vUid, oUid});
        HIP_KERNEL_RETURN_FALSE_IF(
            !unsupportedRaggedTensorIds.empty(),
            "Only the Q, K, V and O tensors can be ragged. The following tensor uids are ragged "
            "and not supported: "
                + hipdnn_data_sdk::utilities::vecToString(unsupportedRaggedTensorIds));

        const int numberRagged
            = static_cast<int>(qTensor->ragged_offset_tensor_uid().has_value())
              + static_cast<int>(kTensor->ragged_offset_tensor_uid().has_value())
              + static_cast<int>(vTensor->ragged_offset_tensor_uid().has_value())
              + static_cast<int>(oTensor->ragged_offset_tensor_uid().has_value());

        HIP_KERNEL_RETURN_FALSE_IF(attrs.seq_len_kv_tensor_uid().has_value()
                                       || attrs.seq_len_q_tensor_uid().has_value(),
                                   "SEQ_LENS_Q and SEQ_LENS_KV not supported");

        HIP_KERNEL_RETURN_FALSE_IF(numberRagged != 0 && numberRagged != 4,
                                   "Either all or none of Q, K, V or O must be ragged");
        batchMode = (numberRagged == 4) ? BatchMode::GROUP : BatchMode::BATCH;
    }

    // Validate Q/K/V/O ranks before accessing their dims by index.
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

    // Every dimension below is narrowed to 32 bits before it reaches a kernel,
    // starting with the mask classification and the registry lookup further
    // down, so the range gate comes first.
    HIP_KERNEL_RETURN_FALSE_IF(
        !wouldFwdDimsFitKernargFields(*qTensor, *kTensor, *vTensor),
        "Tensor dimensions overflow the 32-bit kernarg fields that carry them");

    if(batchMode == BatchMode::GROUP
       && !hasAiterCompatibleRaggedAddressing(tensorMap, *qTensor, *kTensor, *vTensor, *oTensor))
    {
        return false;
    }

    // Validate optional stats (LSE) output tensor.
    // Gate purely on generate_stats; the UID is required when the flag is set.
    const bool hasStats = hasStatsOutput(attrs);
    // Varlen LSE is [H, total_q] indexed by seqstart, not the dense [B, H, Sq] stats layout.
    HIP_KERNEL_RETURN_FALSE_IF(hasStats && batchMode == BatchMode::GROUP,
                               "stats (LSE) output is not supported for ragged tensors");
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* statsTensor = nullptr;
    if(hasStats)
    {
        HIP_KERNEL_RETURN_FALSE_IF(!attrs.stats_tensor_uid().has_value(),
                                   "generate_stats is set but stats_tensor_uid is missing");

        const auto statsIt = tensorMap.find(attrs.stats_tensor_uid().value());
        HIP_KERNEL_RETURN_FALSE_IF(statsIt == tensorMap.end(),
                                   "stats_tensor_uid not found in tensor map");

        statsTensor = statsIt->second;

        HIP_KERNEL_RETURN_FALSE_IF(statsTensor->virtual_(), "stats tensor must not be virtual");
        HIP_KERNEL_RETURN_FALSE_IF(
            hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(statsTensor),
            "stats tensor must not be pass-by-value");

        HIP_KERNEL_RETURN_FALSE_IF(statsTensor->data_type() != DataType::FLOAT,
                                   "stats tensor datatype must be FP32 (Actual type: "
                                       + EnumNameDataType(statsTensor->data_type()) + ")");

        const auto statsRank = statsTensor->dims()->size();
        HIP_KERNEL_RETURN_FALSE_IF(
            statsRank != 3 && statsRank != 4,
            "stats tensor must be rank 3 [B,H,Sq] or rank 4 [B,H,Sq,1] (Actual rank: "
                + std::to_string(statsRank) + ")");

        if(statsRank == 4)
        {
            HIP_KERNEL_RETURN_FALSE_IF(statsTensor->dims()->Get(3) != 1,
                                       "stats tensor rank-4 last dim must be 1 (Actual: "
                                           + std::to_string(statsTensor->dims()->Get(3)) + ")");
        }

        // Shape comparison is safe now — Q rank-4 is validated above.
        HIP_KERNEL_RETURN_FALSE_IF(statsTensor->dims()->Get(0) != qTensor->dims()->Get(0)
                                       || statsTensor->dims()->Get(1) != qTensor->dims()->Get(1)
                                       || statsTensor->dims()->Get(2) != qTensor->dims()->Get(2),
                                   "stats tensor shape [B,H,Sq] must match Q tensor");
    }

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

    // Targeted diagnostic for the FNUZ/OCP fp8 gotcha: OCP FP8_E4M3 (bias 7, the
    // gfx950/MI350 encoding) is not bit-compatible with the FP8_E4M3_FNUZ (bias 8)
    // the gfx942 fp8 forward kernels consume, so it is declined here with a specific
    // message rather than the generic "unsupported datatype" one below. Inputs are
    // proven same-type above, so checking q is sufficient. Remove once gfx950 OCP
    // fp8 forward kernels are wired in.
    HIP_KERNEL_RETURN_FALSE_IF(
        qTensor->data_type() == DataType::FP8_E4M3,
        "gfx942 fp8 SDPA requires FP8_E4M3_FNUZ inputs; OCP FP8_E4M3 (gfx950/MI350) is a "
        "non-bit-compatible encoding and is not yet supported by these kernels");

    HIP_KERNEL_RETURN_FALSE_IF(
        dataTypeId.empty(),
        "output tensor must have datatype BFLOAT16 (Actual type: "
            + EnumNameDataType(oTensor->data_type())
            + ") and input tensors must have datatype BFLOAT16 or FP8_E4M3_FNUZ (Actual type: "
            + EnumNameDataType(qTensor->data_type()) + ")");

    // mma_core_mode is the MMA operand precision. The bf16 kernels run 16-bit MFMA on
    // the graph's own bf16 inputs, so UNSET (the provider's choice), HALF and BFLOAT16
    // all describe them. HALF is accepted on bf16 graphs because the cuDNN frontend,
    // and hipDNN's cuDNN-compatibility shim, write HALF into Graph::sdpa() whenever
    // the caller leaves the field unset. The fp8 kernels run fp8 MFMA, which no 16-bit
    // mode describes, so they still require UNSET. An allow-list, so an enum value
    // added later declines until judged.
    const auto mmaCoreMode = attrs.mma_core_mode();
    const bool isSixteenBitMmaMode
        = mmaCoreMode == DataType::HALF || mmaCoreMode == DataType::BFLOAT16;
    const bool mmaCoreModeSupported
        = mmaCoreMode == DataType::UNSET || (dataTypeId == "bf16" && isSixteenBitMmaMode);
    HIP_KERNEL_RETURN_FALSE_IF(
        !mmaCoreModeSupported,
        "mma_core_mode must be UNSET, or HALF/BFLOAT16 for bf16 inputs (Actual value: "
            + EnumNameDataType(mmaCoreMode) + ", inputs: " + EnumNameDataType(qTensor->data_type())
            + ")");

    // FP8 inputs require q/k/v descales (mirrors AITER's TORCH_CHECK). Without them
    // the kernel would read garbage dequantization scales.
    if(dataTypeId == "fp8bf16")
    {
        HIP_KERNEL_RETURN_FALSE_IF(!attrs.descale_q_tensor_uid().has_value()
                                       || !attrs.descale_k_tensor_uid().has_value()
                                       || !attrs.descale_v_tensor_uid().has_value(),
                                   "fp8 inputs require q, k, and v descale tensors");

        // Only per-tensor (scalar) descales are supported: the kernel-arg builder
        // wires all s_descale_*_Bs/Hs strides to zero, so a non-scalar descale (e.g.
        // per-[B, H_kv]) would be silently mis-read as descale[0] for every batch and
        // head. Decline it here rather than miscompute. Per-[B, H_kv] descales are a
        // future extension (tracked with the gfx950 fp8 path).
        const auto* qDescale = tensorMap.at(attrs.descale_q_tensor_uid().value());
        const auto* kDescale = tensorMap.at(attrs.descale_k_tensor_uid().value());
        const auto* vDescale = tensorMap.at(attrs.descale_v_tensor_uid().value());
        HIP_KERNEL_RETURN_FALSE_IF(
            !isScalarTensor(*qDescale) || !isScalarTensor(*kDescale) || !isScalarTensor(*vDescale),
            "fp8 q/k/v descales must be per-tensor scalars; per-[B, H_kv] descales are not yet "
            "supported");
    }

    // Softmax/output (de)quantization is not implemented by the v3 forward kernel
    // (its argument struct exposes only q/k/v descales).
    HIP_KERNEL_RETURN_FALSE_IF(attrs.descale_s_tensor_uid().has_value(),
                               "descale_s tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.scale_s_tensor_uid().has_value(),
                               "scale_s tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.scale_o_tensor_uid().has_value(),
                               "scale_o tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.amax_s_tensor_uid().has_value(),
                               "amax_s tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.amax_o_tensor_uid().has_value(),
                               "amax_o tensor not supported");

    // Classify the mask; contradictory mask attributes are an invalid-input
    // condition the engine declines rather than dispatches.
    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    try
    {
        const auto band
            = plan_utils::makeCoveringBoundsUnbound(plan_utils::resolveDiagonalBand(attrs),
                                                    qTensor->dims()->Get(2),
                                                    kTensor->dims()->Get(2));
        maskType = plan_utils::getMaskType(band.leftBound, band.rightBound, band.alignment);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(std::string{HIP_KERNEL_LOG_PREFIX} + e.what());
        return false;
    }

    // The current causal kernel binaries do not compute LSE stats correctly
    // when s_lse=1 is set at runtime. Decline causal+stats until the upstream
    // ASM kernels support this combination.
    HIP_KERNEL_RETURN_FALSE_IF(
        hasStats && maskType != plan_utils::MaskType::NO_MASK,
        "stats (LSE) output is not supported with causal masks on the current ASM kernels");

    auto key = getKernelNameKey(deviceString,
                                dataTypeId,
                                static_cast<int>(qTensor->dims()->Get(3)),
                                static_cast<int>(vTensor->dims()->Get(3)),
                                maskType,
                                getBf16ConvertMode(attrs, dataTypeId),
                                batchMode,
                                &cfg_fmha_fwd);

    HIP_KERNEL_RETURN_FALSE_IF(
        key.empty(),
        "Could not find matching kernel for parameter combination. "
        " dataType = "
            + dataTypeId + " qDim = " + std::to_string(qTensor->dims()->Get(3))
            + " vDim  = " + std::to_string(vTensor->dims()->Get(3))
            + " mask = " + std::to_string(static_cast<int>(maskType)) + " rounding = "
            + std::to_string(static_cast<int>(getBf16ConvertMode(attrs, dataTypeId)))
            + " batch = " + std::to_string(static_cast<int>(batchMode)));

    // Q/K/V byte strides and the Ts product are scaled by the resolved input
    // size, which the datatype checks above have narrowed to bf16 or fp8bf16.
    const int64_t inputBytes = fwdInputBytesPerElement(dataTypeId);

    HIP_KERNEL_RETURN_FALSE_IF(!wouldFwdByteStridesFitUint32(
                                   *qTensor, *kTensor, *vTensor, *oTensor, statsTensor, inputBytes),
                               "Forward byte strides overflow uint32_t kernarg fields");

    // s_Ts = tileSizeQo * qStrideRow * inBytesPerElement (SdpaFwdArgsBuilder) is
    // the one forward kernarg that is a product rather than a single stride, and
    // it is formed in unsigned int, so it wraps for Q sequence strides the
    // per-field check above accepts. The resolved row's ts_qo folds into the
    // element size so the product is never formed outside the predicate.
    // Mirrors the backward Ts check.
    HIP_KERNEL_RETURN_FALSE_IF(
        !plan_utils::byteStrideFitsU32("Ts (ts_qo * stride_q)",
                                       qTensor->strides()->Get(2),
                                       inputBytes * cfg_fmha_fwd.at(key).ts_qo),
        "Forward Ts product overflows the uint32_t s_Ts kernarg field");

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
    // Forward exposes no knobs — nothing to parse.
}

void SdpaFwdPlanBuilder::buildPlan(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    Context& executionContext) const
{

    // Get device properties
    auto deviceStringOpt = plan_utils::tryGetDeviceString(
        handle.getStream(), "SdpaFwdPlanBuilder::buildPlan: failed to query device properties: ");
    if(!deviceStringOpt)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaFwdPlanBuilder::buildPlan: failed to query device string");
    }
    const std::string& deviceString = *deviceStringOpt;
    const bool isMi308 = isMi308Device(handle.getStream());

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

    const unsigned int headAxis = 1U;
    const unsigned int seqAxis = SDPA_SEQ_AXIS;

    // Extract dimensions from Q tensor
    auto* qDims = qTensor->dims();
    auto batchSize = static_cast<unsigned int>(qDims->Get(0));
    auto numHeadsQ = static_cast<unsigned int>(qDims->Get(headAxis));
    auto seqLenQ = static_cast<unsigned int>(qDims->Get(seqAxis));
    auto headDimQk = static_cast<unsigned int>(qDims->Get(3));

    // Extract dimensions from K tensor
    auto* kDims = kTensor->dims();
    auto numHeadsKv = static_cast<unsigned int>(kDims->Get(headAxis));
    auto seqLenKv = static_cast<unsigned int>(kDims->Get(seqAxis));

    // Extract dimensions from V tensor: [B, H_kv, S_kv, D_v]
    auto* vDims = vTensor->dims();
    auto headDimV = static_cast<unsigned int>(vDims->Get(3));

    // Extract strides (in elements) - Q
    auto* qStrides = qTensor->strides();
    auto qStrideBatch = static_cast<unsigned int>(qStrides->Get(0));
    auto qStrideHead = static_cast<unsigned int>(qStrides->Get(headAxis));
    auto qStrideSeq = static_cast<unsigned int>(qStrides->Get(seqAxis));
    auto qStrideRow = qStrideSeq; // Same as sequence stride

    // Extract strides - K
    auto* kStrides = kTensor->strides();
    auto kStrideBatch = static_cast<unsigned int>(kStrides->Get(0));
    auto kStrideHead = static_cast<unsigned int>(kStrides->Get(headAxis));
    auto kStrideSeq = static_cast<unsigned int>(kStrides->Get(seqAxis));

    // Extract strides - V
    auto* vStrides = vTensor->strides();
    auto vStrideBatch = static_cast<unsigned int>(vStrides->Get(0));
    auto vStrideHead = static_cast<unsigned int>(vStrides->Get(headAxis));
    auto vStrideSeq = static_cast<unsigned int>(vStrides->Get(seqAxis));

    // Extract strides - O
    auto* oStrides = oTensor->strides();
    auto oStrideBatch = static_cast<unsigned int>(oStrides->Get(0));
    auto oStrideHead = static_cast<unsigned int>(oStrides->Get(headAxis));
    auto oStrideSeq = static_cast<unsigned int>(oStrides->Get(seqAxis));

    hipdnn_plugin_sdk::ScalarOperand attnScale{};
    if(sdpaAttrs.scale_tensor_uid().has_value())
    {
        attnScale = hipdnn_plugin_sdk::makeScalarOperand(
            tensorMap, sdpaAttrs.scale_tensor_uid().value(), "attn_scale");
    }
    else
    {
        const float scaleVal = plan_utils::attnScaleOrDefault(sdpaAttrs);
        attnScale = hipdnn_plugin_sdk::ScalarOperand{
            0,
            hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
            false,
            hipdnn_plugin_sdk::ScalarValue{scaleVal}};
    }

    // Extract optional LSE output metadata
    std::optional<int64_t> lseUid;
    unsigned int lseStrideHead = 0;
    const bool hasStats = hasStatsOutput(sdpaAttrs);
    if(hasStats)
    {
        if(!sdpaAttrs.stats_tensor_uid().has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "SdpaFwdPlanBuilder::buildPlan: generate_stats is set but stats_tensor_uid "
                "is missing");
        }
        lseUid = sdpaAttrs.stats_tensor_uid();
        const auto lseIt = tensorMap.find(lseUid.value());
        if(lseIt == tensorMap.end())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "SdpaFwdPlanBuilder::buildPlan: stats_tensor_uid not found in tensor map");
        }
        lseStrideHead = static_cast<unsigned int>(lseIt->second->strides()->Get(1));
    }

    // Create params struct with all metadata
    SdpaFwdParams params{};
    params.qUid = qUid;
    params.kUid = kUid;
    params.vUid = vUid;
    params.oUid = oUid;
    params.lseUid = lseUid;
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
    params.lseStrideHead = lseStrideHead;
    params.attnScale = attnScale;
    params.archString = deviceString;
    // Classify on the graph dims, exactly as isApplicable does. Passing the
    // narrowed unsigned locals instead would let the two sites resolve
    // different mask types for the same graph.
    const auto band = plan_utils::makeCoveringBoundsUnbound(
        plan_utils::resolveDiagonalBand(sdpaAttrs), qDims->Get(2), kDims->Get(2));
    params.maskType = plan_utils::getMaskType(band.leftBound, band.rightBound, band.alignment);

    const auto dataTypeId = getDataTypeIdentifier(
        qTensor->data_type(), kTensor->data_type(), vTensor->data_type(), oTensor->data_type());

    // FP8 inputs are 1 byte; bf16 inputs are 2 bytes. Output stays 2-byte BF16.
    params.inBytesPerElement = static_cast<unsigned int>(fwdInputBytesPerElement(dataTypeId));

    // FP8 requires q/k/v descales (guaranteed present by isApplicable). They are set as
    // a unit — the SdpaFwdParams::DescaleUids optional encodes the all-or-none invariant.
    if(dataTypeId == "fp8bf16")
    {
        params.descaleUids = SdpaFwdParams::DescaleUids{sdpaAttrs.descale_q_tensor_uid().value(),
                                                        sdpaAttrs.descale_k_tensor_uid().value(),
                                                        sdpaAttrs.descale_v_tensor_uid().value()};
    }

    // isApplicable ensures all of q, k, v and o tensors are ragged if any are, and that O and V
    // share the Q and K offset tensors respectively
    const auto batchMode
        = qTensor->ragged_offset_tensor_uid().has_value() ? BatchMode::GROUP : BatchMode::BATCH;

    if(batchMode == BatchMode::GROUP)
    {
        params.group = SdpaFwdGroupModeParams{qTensor->ragged_offset_tensor_uid().value(),
                                              kTensor->ragged_offset_tensor_uid().value()};
    }

    // Find matching kernel to graph
    fmha_v3_fwdConfig config;
    auto kernelKey = getKernelNameKey(deviceString,
                                      dataTypeId,
                                      static_cast<int>(headDimQk),
                                      static_cast<int>(headDimV),
                                      params.maskType,
                                      getBf16ConvertMode(sdpaAttrs, dataTypeId),
                                      batchMode,
                                      &cfg_fmha_fwd);

    if(kernelKey.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaFwdPlanBuilder::buildPlan: failed to find matching kernel for arch=" + deviceString
                + " dtype="
                + getDataTypeIdentifier(qTensor->data_type(),
                                        kTensor->data_type(),
                                        vTensor->data_type(),
                                        oTensor->data_type())
                + " hdim_q=" + std::to_string(headDimQk) + " hdim_v=" + std::to_string(headDimV));
    }
    config = cfg_fmha_fwd.at(kernelKey);

    params.tileSizeQo = static_cast<unsigned int>(config.ts_qo);

    // Load kernel module from kpack archive
    auto tocKey = getKernelTocKey(config.co_name, deviceString, isMi308);

    HIPDNN_PLUGIN_LOG_INFO("Using kernel with tocKey: " << tocKey << " arch: " << deviceString);

    auto kernel = moduleCache().getOrLoad(tocKey, deviceString, config.knl_name.c_str());
    if(!kernel)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaFwdPlanBuilder::buildPlan: failed to load kernel tocKey='" + tocKey + "' arch='"
                + deviceString + "'");
    }

    executionContext.setPlan(std::make_unique<SdpaFwdPlan>(std::move(kernel), params));
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> SdpaFwdPlanBuilder::getCustomKnobs(
    const Handle& /* handle */,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /* opGraph */) const
{
    return {};
}

} // namespace asm_sdpa_engine
