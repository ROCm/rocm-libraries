// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaFwdPlanBuilder.hpp"
#include "asm/AsmKernelPath.hpp"
#include "asm_fmha_v3_fwd_configs.hpp"
#include "core/Utils.hpp"
#include "plans/SdpaFwdPlan.hpp"
#include "plans/SdpaPlanUtils.hpp"

#include <cmath>
#include <exception>
#include <optional>
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
using hip_kernel_provider::core::utils::Rejection;

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
    // Only the batch-axis stride is checked: dims 1-3 are size 1, so their index is always 0
    // and their strides never contribute to an element address. stride[0] == 1 makes the
    // B+1 INT32 entries contiguous, which is how AITER reads seqstart[b].
    return dims != nullptr && strides != nullptr && dims->size() == 4 && strides->size() == 4
           && dims->Get(0) == batch + 1 && dims->Get(1) == 1 && dims->Get(2) == 1
           && dims->Get(3) == 1 && strides->Get(0) == 1;
}

// hipDNN addresses a ragged element at offset[b] * multiplier + strides . (h, s, d), while
// AITER uses seqstart[b] * seq_stride + strides . (h, s, d) with one seqstart for Q/O and one
// for K/V. The two agree for any stride permutation under the conditions below.
static Rejection checkAiterRaggedAddressing(
    const plan_utils::TensorMap& tensorMap,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& qTensor,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& kTensor,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& vTensor,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& oTensor)
{
    const int64_t qoOffsetUid = qTensor.ragged_offset_tensor_uid().value();
    const int64_t kvOffsetUid = kTensor.ragged_offset_tensor_uid().value();
    HIP_KERNEL_REJECT_IF(oTensor.ragged_offset_tensor_uid().value() != qoOffsetUid,
                         "Q and O must share a ragged offset tensor");
    HIP_KERNEL_REJECT_IF(vTensor.ragged_offset_tensor_uid().value() != kvOffsetUid,
                         "K and V must share a ragged offset tensor");

    for(const auto& [label, tensor] : {std::pair{"q", &qTensor},
                                       std::pair{"k", &kTensor},
                                       std::pair{"v", &vTensor},
                                       std::pair{"o", &oTensor}})
    {
        HIP_KERNEL_REJECT_IF(tensor->strides() == nullptr || tensor->strides()->size() != 4,
                             std::string(label) + " tensor must have rank 4 strides");
        HIP_KERNEL_REJECT_IF(
            tensor->ragged_offset_multiplier() != tensor->strides()->Get(SDPA_SEQ_AXIS),
            std::string(label) + " tensor ragged_offset_multiplier must equal its sequence "
                + "stride (Actual multiplier: " + std::to_string(tensor->ragged_offset_multiplier())
                + ", sequence stride: " + std::to_string(tensor->strides()->Get(SDPA_SEQ_AXIS))
                + ")");
    }

    const int64_t batch = qTensor.dims()->Get(0);
    HIP_KERNEL_REJECT_IF(
        !isTokenSeqstartTensor(tensorMap, qoOffsetUid, batch)
            || !isTokenSeqstartTensor(tensorMap, kvOffsetUid, batch),
        "Ragged offset tensors must be non-virtual INT32 [B+1, 1, 1, 1] device tensors");
    return std::nullopt;
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

// Validate that every forward-pass Q/K/V/O byte stride fits in uint32_t. The
// ASM kernarg struct stores strides as uint32 so values that overflow silently
// truncate, producing wrong results. Checked in isApplicable so the engine
// declines rather than dispatching with bad strides. `inputBytes` is the
// resolved Q/K/V element size, so an fp8 descriptor is bounded at the one byte
// the kernarg builder actually scales by; O stays BF16. The stats/LSE strides
// are checked with the rest of the stats tensor (checkStatsTensor).
static bool
    wouldFwdByteStridesFitUint32(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& q,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& k,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& v,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& o,
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

// Group-mode (ragged) layout: only Q/K/V/O may be ragged, all of them or none, and a
// ragged layout must match AITER's seqstart addressing.
static Rejection
    checkRaggedLayout(const plan_utils::TensorMap& tensorMap,
                      const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attrs,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& qTensor,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& kTensor,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& vTensor,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& oTensor)
{
    const auto unsupportedRaggedTensorIds
        = hipdnn_flatbuffers_sdk::utilities::listUnsupportedRaggedTensorIds(tensorMap,
                                                                            {attrs.q_tensor_uid(),
                                                                             attrs.k_tensor_uid(),
                                                                             attrs.v_tensor_uid(),
                                                                             attrs.o_tensor_uid()});
    HIP_KERNEL_REJECT_IF(
        !unsupportedRaggedTensorIds.empty(),
        "Only the Q, K, V and O tensors can be ragged. The following tensor uids are ragged "
        "and not supported: "
            + hipdnn_data_sdk::utilities::vecToString(unsupportedRaggedTensorIds));

    const int numberRagged = static_cast<int>(qTensor.ragged_offset_tensor_uid().has_value())
                             + static_cast<int>(kTensor.ragged_offset_tensor_uid().has_value())
                             + static_cast<int>(vTensor.ragged_offset_tensor_uid().has_value())
                             + static_cast<int>(oTensor.ragged_offset_tensor_uid().has_value());
    HIP_KERNEL_REJECT_IF(numberRagged != 0 && numberRagged != 4,
                         "Either all or none of Q, K, V or O must be ragged");
    if(numberRagged == 0)
    {
        return std::nullopt;
    }
    return checkAiterRaggedAddressing(tensorMap, qTensor, kTensor, vTensor, oTensor);
}

// Graph features only the forward attributes carry that the v3 forward kernel does not dispatch.
static Rejection
    checkFwdUnsupportedFeatures(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attrs)
{
    HIP_KERNEL_REJECT_IF(attrs.page_table_k_tensor_uid(), "page_table_k tensor not supported");
    HIP_KERNEL_REJECT_IF(attrs.page_table_v_tensor_uid(), "page_table_v tensor not supported");
    // Attention sinks join the softmax denominator; the vendored kernels have no sink input.
    HIP_KERNEL_REJECT_IF(attrs.sink_token_tensor_uid(), "sink_token tensor not supported");
    HIP_KERNEL_REJECT_IF(attrs.seq_len_kv_tensor_uid().has_value()
                             || attrs.seq_len_q_tensor_uid().has_value(),
                         "SEQ_LENS_Q and SEQ_LENS_KV not supported");

    // Softmax/output (de)quantization is not implemented by the v3 forward kernel
    // (its argument struct exposes only q/k/v descales).
    HIP_KERNEL_REJECT_IF(attrs.descale_s_tensor_uid().has_value(),
                         "descale_s tensor not supported");
    HIP_KERNEL_REJECT_IF(attrs.scale_s_tensor_uid().has_value(), "scale_s tensor not supported");
    HIP_KERNEL_REJECT_IF(attrs.scale_o_tensor_uid().has_value(), "scale_o tensor not supported");
    HIP_KERNEL_REJECT_IF(attrs.amax_s_tensor_uid().has_value(), "amax_s tensor not supported");
    HIP_KERNEL_REJECT_IF(attrs.amax_o_tensor_uid().has_value(), "amax_o tensor not supported");
    return std::nullopt;
}

// The kernel reads `tensor` through a device pointer as `expectedType`. A virtual
// tensor has no buffer, a pass-by-value one is delivered as a host pointer (or not
// at all), and any other dtype would have its bytes reinterpreted.
static Rejection
    checkDeviceTensor(const char* label,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensor,
                      hipdnn_flatbuffers_sdk::data_objects::DataType expectedType)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    HIP_KERNEL_REJECT_IF(
        tensor.virtual_() || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor),
        std::string(label) + " tensor must be a non-virtual, non-pass-by-value device tensor");
    HIP_KERNEL_REJECT_IF(tensor.data_type() != expectedType,
                         std::string(label) + " tensor must be " + EnumNameDataType(expectedType)
                             + " (Actual value: " + EnumNameDataType(tensor.data_type()) + ")");
    return std::nullopt;
}

// Validates the stats (LSE) output tensor generate_stats requests. Q must already be
// validated rank 4.
static Rejection checkStatsTensor(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attrs,
                                  const plan_utils::TensorMap& tensorMap,
                                  BatchMode batchMode,
                                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& q)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // Varlen LSE is [H, total_q] indexed by seqstart, not the dense [B, H, Sq] stats layout.
    HIP_KERNEL_REJECT_IF(batchMode == BatchMode::GROUP,
                         "stats (LSE) output is not supported for ragged tensors");
    // Gate purely on generate_stats; the UID is required when the flag is set.
    HIP_KERNEL_REJECT_IF(!attrs.stats_tensor_uid().has_value(),
                         "generate_stats is set but stats_tensor_uid is missing");

    const auto statsIt = tensorMap.find(attrs.stats_tensor_uid().value());
    HIP_KERNEL_REJECT_IF(statsIt == tensorMap.end(), "stats_tensor_uid not found in tensor map");
    const auto& stats = *statsIt->second;

    HIP_KERNEL_RETURN_IF_REJECTED(checkDeviceTensor("stats", stats, DataType::FLOAT));

    const auto statsRank = stats.dims()->size();
    HIP_KERNEL_REJECT_IF(statsRank != 3 && statsRank != 4,
                         "stats tensor must be rank 3 [B,H,Sq] or rank 4 [B,H,Sq,1] (Actual rank: "
                             + std::to_string(statsRank) + ")");
    HIP_KERNEL_REJECT_IF(statsRank == 4 && stats.dims()->Get(3) != 1,
                         "stats tensor rank-4 last dim must be 1 (Actual: "
                             + std::to_string(stats.dims()->Get(3)) + ")");

    HIP_KERNEL_REJECT_IF(stats.dims()->Get(0) != q.dims()->Get(0)
                             || stats.dims()->Get(1) != q.dims()->Get(1)
                             || stats.dims()->Get(2) != q.dims()->Get(2),
                         "stats tensor shape [B,H,Sq] must match Q tensor");

    // Only the FP32 batch and head strides reach the kernel.
    const auto* statsStrides = stats.strides();
    bool ok = plan_utils::byteStrideFitsU32("batch_stride_lse", statsStrides->Get(0), K_FP32_BYTES);
    ok &= plan_utils::byteStrideFitsU32("nhead_stride_lse", statsStrides->Get(1), K_FP32_BYTES);
    HIP_KERNEL_REJECT_IF(!ok, "stats byte strides overflow uint32_t kernarg fields");
    return std::nullopt;
}

// Input/output dtype requirements; `dataTypeId` is getDataTypeIdentifier()'s result.
static Rejection
    checkFwdDataTypes(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& qTensor,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& kTensor,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& vTensor,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& oTensor,
                      const std::string& dataTypeId)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    HIP_KERNEL_REJECT_IF(
        qTensor.data_type() != kTensor.data_type() || qTensor.data_type() != vTensor.data_type(),
        "Input tensors must all share a type (q tensor: " + EnumNameDataType(qTensor.data_type())
            + ", k tensor: " + EnumNameDataType(kTensor.data_type())
            + ", v tensor: " + EnumNameDataType(vTensor.data_type()) + ")");

    // Targeted diagnostic for the FNUZ/OCP fp8 gotcha: OCP FP8_E4M3 (bias 7, the
    // gfx950/MI350 encoding) is not bit-compatible with the FP8_E4M3_FNUZ (bias 8)
    // the gfx942 fp8 forward kernels consume, so it is declined here with a specific
    // message rather than the generic "unsupported datatype" one below. Inputs are
    // proven same-type above, so checking q is sufficient. Remove once gfx950 OCP
    // fp8 forward kernels are wired in.
    HIP_KERNEL_REJECT_IF(
        qTensor.data_type() == DataType::FP8_E4M3,
        "gfx942 fp8 SDPA requires FP8_E4M3_FNUZ inputs; OCP FP8_E4M3 (gfx950/MI350) is a "
        "non-bit-compatible encoding and is not yet supported by these kernels");

    HIP_KERNEL_REJECT_IF(
        dataTypeId.empty(),
        "output tensor must have datatype BFLOAT16 (Actual type: "
            + EnumNameDataType(oTensor.data_type())
            + ") and input tensors must have datatype BFLOAT16 or FP8_E4M3_FNUZ (Actual type: "
            + EnumNameDataType(qTensor.data_type()) + ")");
    return std::nullopt;
}

// mma_core_mode is the MMA operand precision. The bf16 kernels run 16-bit MFMA on
// the graph's own bf16 inputs, so UNSET (the provider's choice), HALF and BFLOAT16
// all describe them. HALF is accepted on bf16 graphs because the cuDNN frontend,
// and hipDNN's cuDNN-compatibility shim, write HALF into Graph::sdpa() whenever
// the caller leaves the field unset. The fp8 kernels run fp8 MFMA, which no 16-bit
// mode describes, so they still require UNSET. An allow-list, so an enum value
// added later declines until judged.
static Rejection checkMmaCoreMode(hipdnn_flatbuffers_sdk::data_objects::DataType mmaCoreMode,
                                  const std::string& dataTypeId,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType inputType)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    const bool isSixteenBitMmaMode
        = mmaCoreMode == DataType::HALF || mmaCoreMode == DataType::BFLOAT16;
    const bool mmaCoreModeSupported
        = mmaCoreMode == DataType::UNSET || (dataTypeId == "bf16" && isSixteenBitMmaMode);
    HIP_KERNEL_REJECT_IF(
        !mmaCoreModeSupported,
        "mma_core_mode must be UNSET, or HALF/BFLOAT16 for bf16 inputs (Actual value: "
            + EnumNameDataType(mmaCoreMode) + ", inputs: " + EnumNameDataType(inputType) + ")");
    return std::nullopt;
}

// FP8 inputs require q/k/v descales (mirrors AITER's TORCH_CHECK). Without them
// the kernel would read garbage dequantization scales.
static Rejection checkFp8Descales(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attrs,
                                  const plan_utils::TensorMap& tensorMap)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    HIP_KERNEL_REJECT_IF(!attrs.descale_q_tensor_uid().has_value()
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
    HIP_KERNEL_REJECT_IF(
        !isScalarTensor(*qDescale) || !isScalarTensor(*kDescale) || !isScalarTensor(*vDescale),
        "fp8 q/k/v descales must be per-tensor scalars; per-[B, H_kv] descales are not yet "
        "supported");

    // The kernel reads each descale as an FP32 value through a device pointer.
    for(const auto& [label, tensor] : {std::pair{"descale_q", qDescale},
                                       std::pair{"descale_k", kDescale},
                                       std::pair{"descale_v", vDescale}})
    {
        HIP_KERNEL_RETURN_IF_REJECTED(checkDeviceTensor(label, *tensor, DataType::FLOAT));
    }
    return std::nullopt;
}

static Rejection
    checkFwdApplicability(const Handle& handle,
                          const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    HIP_KERNEL_RETURN_IF_REJECTED(
        plan_utils::checkSdpaGraphBasics(opGraph, NodeAttributes::SdpaAttributes));

    const auto deviceStringOpt = plan_utils::tryGetDeviceString(
        handle.getStream(), "[SdpaFwdPlanBuilder::isApplicable] Could not query device string: ");
    HIP_KERNEL_REJECT_IF(!deviceStringOpt, "Could not query device string");
    const std::string& deviceString = *deviceStringOpt;
    HIP_KERNEL_REJECT_IF(
        deviceString != "gfx942" && deviceString != "gfx950",
        "Device string does not match gfx942 or gfx950 (Actual value: " + deviceString + ")");

    const auto& attrs = opGraph.nodeWrappers().front()->attributesAs<SdpaAttributes>();
    const auto& tensorMap = opGraph.getTensorMap();
    HIP_KERNEL_RETURN_IF_REJECTED(plan_utils::checkCommonUnsupportedFeatures(attrs, tensorMap));
    HIP_KERNEL_RETURN_IF_REJECTED(checkFwdUnsupportedFeatures(attrs));
    HIP_KERNEL_RETURN_IF_REJECTED(plan_utils::checkAttnScale(attrs, tensorMap));

    const auto* qTensor = tensorMap.at(attrs.q_tensor_uid());
    const auto* kTensor = tensorMap.at(attrs.k_tensor_uid());
    const auto* vTensor = tensorMap.at(attrs.v_tensor_uid());
    const auto* oTensor = tensorMap.at(attrs.o_tensor_uid());

    // Validate Q/K/V/O before accessing their dims by index.
    for(const auto& [label, tensor] : {std::pair{"q", qTensor},
                                       std::pair{"k", kTensor},
                                       std::pair{"v", vTensor},
                                       std::pair{"o", oTensor}})
    {
        HIP_KERNEL_RETURN_IF_REJECTED(plan_utils::checkRank4DeviceTensor(label, *tensor));
    }

    HIP_KERNEL_RETURN_IF_REJECTED(
        checkRaggedLayout(tensorMap, attrs, *qTensor, *kTensor, *vTensor, *oTensor));
    // All-or-none raggedness was enforced above; the same derivation buildPlan uses.
    const auto batchMode
        = qTensor->ragged_offset_tensor_uid().has_value() ? BatchMode::GROUP : BatchMode::BATCH;

    // Every dimension below is narrowed to 32 bits before it reaches a kernel,
    // starting with the mask classification and the registry lookup further
    // down, so the range gate comes first.
    HIP_KERNEL_REJECT_IF(!wouldFwdDimsFitKernargFields(*qTensor, *kTensor, *vTensor),
                         "Tensor dimensions overflow the 32-bit kernarg fields that carry them");

    HIP_KERNEL_REJECT_IF(kTensor->dims()->Get(1) != vTensor->dims()->Get(1),
                         "k tensor and v tensor must shared the same head count (Actual value: k = "
                             + std::to_string(kTensor->dims()->Get(1))
                             + " v = " + std::to_string(vTensor->dims()->Get(1)) + ")");

    const bool hasStats = hasStatsOutput(attrs);
    if(hasStats)
    {
        HIP_KERNEL_RETURN_IF_REJECTED(checkStatsTensor(attrs, tensorMap, batchMode, *qTensor));
    }

    const auto dataTypeId = getDataTypeIdentifier(
        qTensor->data_type(), kTensor->data_type(), vTensor->data_type(), oTensor->data_type());
    HIP_KERNEL_RETURN_IF_REJECTED(
        checkFwdDataTypes(*qTensor, *kTensor, *vTensor, *oTensor, dataTypeId));
    HIP_KERNEL_RETURN_IF_REJECTED(
        checkMmaCoreMode(attrs.mma_core_mode(), dataTypeId, qTensor->data_type()));
    if(dataTypeId == "fp8bf16")
    {
        HIP_KERNEL_RETURN_IF_REJECTED(checkFp8Descales(attrs, tensorMap));
    }

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    HIP_KERNEL_RETURN_IF_REJECTED(plan_utils::tryClassifyMask(
        attrs, qTensor->dims()->Get(2), kTensor->dims()->Get(2), maskType));

    // The current causal kernel binaries do not compute LSE stats correctly
    // when s_lse=1 is set at runtime. Decline causal+stats until the upstream
    // ASM kernels support this combination.
    HIP_KERNEL_REJECT_IF(
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

    HIP_KERNEL_REJECT_IF(
        key.empty(),
        "Could not find matching kernel for parameter combination. dataType = " + dataTypeId
            + " qDim = " + std::to_string(qTensor->dims()->Get(3))
            + " vDim = " + std::to_string(vTensor->dims()->Get(3))
            + " mask = " + std::to_string(static_cast<int>(maskType)) + " rounding = "
            + std::to_string(static_cast<int>(getBf16ConvertMode(attrs, dataTypeId)))
            + " batch = " + std::to_string(static_cast<int>(batchMode)));

    // Q/K/V byte strides and the Ts product are scaled by the resolved input
    // size, which the datatype checks above have narrowed to bf16 or fp8bf16.
    const int64_t inputBytes = fwdInputBytesPerElement(dataTypeId);

    HIP_KERNEL_REJECT_IF(
        !wouldFwdByteStridesFitUint32(*qTensor, *kTensor, *vTensor, *oTensor, inputBytes),
        "Forward byte strides overflow uint32_t kernarg fields");

    // s_Ts = tileSizeQo * qStrideRow * inBytesPerElement (SdpaFwdArgsBuilder) is
    // the one forward kernarg that is a product rather than a single stride, and
    // it is formed in unsigned int, so it wraps for Q sequence strides the
    // per-field check above accepts. The resolved row's ts_qo folds into the
    // element size so the product is never formed outside the predicate.
    // Mirrors the backward Ts check.
    HIP_KERNEL_REJECT_IF(!plan_utils::byteStrideFitsU32("Ts (ts_qo * stride_q)",
                                                        qTensor->strides()->Get(2),
                                                        inputBytes * cfg_fmha_fwd.at(key).ts_qo),
                         "Forward Ts product overflows the uint32_t s_Ts kernarg field");

    return std::nullopt;
}

bool SdpaFwdPlanBuilder::isApplicable(
    const Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    const auto rejection = checkFwdApplicability(handle, opGraph);
    if(rejection.has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO("[SdpaFwdPlanBuilder::isApplicable] " << *rejection);
        return false;
    }
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
