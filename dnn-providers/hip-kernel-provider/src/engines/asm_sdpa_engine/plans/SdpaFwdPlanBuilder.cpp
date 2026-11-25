// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaFwdPlanBuilder.hpp"
#include "asm/AsmKernelPath.hpp"
#include "asm_fmha_v3_fwd_configs.hpp"
#include "core/Utils.hpp"
#include "plans/SdpaFwdLaunchParams.hpp"
#include "plans/SdpaFwdPlan.hpp"
#include "plans/SdpaPlanUtils.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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
using plan_utils::Rejection;

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

std::string fwdKernelNameKey(const std::string& archId,
                             const std::string& dataType,
                             int hdimQ,
                             int hdimV,
                             plan_utils::MaskType maskType,
                             RoundingMode bf16Cvt,
                             BatchMode mode)
{
    std::string kernelNameKey{};
    for(const auto& el : cfg_fmha_fwd)
    {
        const auto& cfg = el.second;
        if(cfg.arch != archId)
        {
            continue;
        }

        if(cfg.dtype == dataType && cfg.hdim_q == hdimQ && cfg.hdim_v == hdimV
           && cfg.mask == static_cast<int>(maskType) && static_cast<int>(cfg.mode) == mode)
        {
            if(archId == "gfx950")
            {
                kernelNameKey = el.first;
                break;
            }
            if(archId == "gfx942" && cfg.bf16_cvt == bf16Cvt)
            {
                kernelNameKey = el.first;
                break;
            }
        }
    }

    return kernelNameKey;
}

// The registry mask ordinal of the forward kernel for a graph whose band classifies as
// maskType. AITER's get_cfg_mask_type() (mha_fwd.cu) runs a top-left causal mask on the
// bottom-right causal (mask 2) kernels when the Q and K/V lengths are equal, where the two
// masks agree; the registry has no top-left causal rows. See hasEqualPerBatchLengths.
static plan_utils::MaskType fwdKernelMaskType(plan_utils::MaskType maskType,
                                              bool perBatchLengthsEqual)
{
    if(maskType == plan_utils::MaskType::TOP_LEFT_CAUSAL && perBatchLengthsEqual)
    {
        return plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL;
    }
    return maskType;
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

// AITER group mode reads INT32 [B+1] token tables: the seqstarts (physical starts) and the
// cumulative logical lengths.
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
// for K/V. The two agree for any stride permutation under the conditions below. The strides
// must already be known to be rank 4.
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

// The sequence-length inputs the kernels can take. AITER's group mode reads each batch's
// logical length from a cumulative table (ptr_qseq/ptr_kseq) and its physical start from the
// ragged offset table (ptr_*seq_padding), so a ragged graph may carry cu_seq_len_q/kv, which
// the frontend pairs with padding_mask. Per-batch seq_len_q/kv would need a device prefix sum
// into that cumulative form, and batch mode has no length input ("batch mode does not
// support padded", AITER init_fmha_fwd_v3_args). Q must already be known to be rank 4.
static Rejection
    checkSequenceLengths(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attrs,
                         const plan_utils::TensorMap& tensorMap,
                         const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& qTensor,
                         BatchMode batchMode)
{
    HIP_KERNEL_REJECT_IF(attrs.seq_len_q_tensor_uid().has_value()
                             || attrs.seq_len_kv_tensor_uid().has_value(),
                         "seq_len_q/kv not supported; ragged graphs can give the lengths "
                         "as cu_seq_len_q/kv");

    const auto cuSeqLenQUid = attrs.cu_seq_len_q_tensor_uid();
    const auto cuSeqLenKvUid = attrs.cu_seq_len_kv_tensor_uid();
    if(!cuSeqLenQUid.has_value() && !cuSeqLenKvUid.has_value())
    {
        HIP_KERNEL_REJECT_IF(
            attrs.padding_mask(),
            "padding_mask requires cu_seq_len_q/kv and ragged Q, K, V and O tensors");
        return std::nullopt;
    }

    HIP_KERNEL_REJECT_IF(batchMode != BatchMode::GROUP,
                         "cu_seq_len_q/kv require ragged Q, K, V and O tensors");
    HIP_KERNEL_REJECT_IF(!cuSeqLenQUid.has_value() || !cuSeqLenKvUid.has_value(),
                         "cu_seq_len_q and cu_seq_len_kv must be set together");
    HIP_KERNEL_REJECT_IF(!attrs.padding_mask(), "cu_seq_len_q/kv require padding_mask");

    const int64_t batch = qTensor.dims()->Get(0);
    HIP_KERNEL_REJECT_IF(!isTokenSeqstartTensor(tensorMap, cuSeqLenQUid.value(), batch)
                             || !isTokenSeqstartTensor(tensorMap, cuSeqLenKvUid.value(), batch),
                         "cu_seq_len_q/kv must be non-virtual INT32 [B+1, 1, 1, 1] device tensors");
    return std::nullopt;
}

// The group-mode token tables of a graph isApplicable accepted in group mode.
static SdpaFwdGroupModeParams
    makeGroupModeParams(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attrs,
                        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& qTensor,
                        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& kTensor)
{
    SdpaFwdGroupModeParams group{qTensor.ragged_offset_tensor_uid().value(),
                                 kTensor.ragged_offset_tensor_uid().value()};
    if(attrs.cu_seq_len_q_tensor_uid().has_value())
    {
        group.cuSeqLenUids = SdpaFwdGroupModeParams::CumulativeLengthUids{
            attrs.cu_seq_len_q_tensor_uid().value(), attrs.cu_seq_len_kv_tensor_uid().value()};
    }
    return group;
}

// True when every batch's logical Q length equals its K/V length, where the top-left and
// bottom-right causal masks agree. Batch mode compares S_q with S_kv. Group mode reads each
// side's lengths from one table, the cumulative lengths when the graph has them and the
// ragged offsets otherwise, so the lengths agree per batch when both sides name the same
// table. AITER compares seqlen_q with seqlen_k in group mode too, but its varlen path sets
// those to the packed totals, and equal totals do not make the per-batch lengths equal.
static bool
    hasEqualPerBatchLengths(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& qTensor,
                            const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& kTensor,
                            const std::optional<SdpaFwdGroupModeParams>& group)
{
    if(!group.has_value())
    {
        return qTensor.dims()->Get(SDPA_SEQ_AXIS) == kTensor.dims()->Get(SDPA_SEQ_AXIS);
    }
    const int64_t qLengthTable
        = group->cuSeqLenUids ? group->cuSeqLenUids->q : group->qoRaggedOffsetUid;
    const int64_t kvLengthTable
        = group->cuSeqLenUids ? group->cuSeqLenUids->kv : group->kvRaggedOffsetUid;
    return qLengthTable == kvLengthTable;
}

// The gfx950 bf16 head-dim-256 kernels index K/V by the query head, so they compute only
// equal Q and KV head counts. AITER sends GQA/MQA for them to CK.
//
// Known limitation, not declined: these kernels round Q * scale * log2(e) to bf16, so
// their error grows with the attention scale. AITER launches them at any scale too.
// See "Forward precision" in asm/asm_kernels/README.md.
static bool kernelServesHeadCounts(const std::string& archId,
                                   const std::string& dataTypeId,
                                   const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& q,
                                   const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& k)
{
    return archId != "gfx950" || dataTypeId != "bf16" || q.dims()->Get(3) != 256
           || q.dims()->Get(1) == k.dims()->Get(1);
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

bool fwdKvRowExtentsFitU32(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& k,
                           const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& v,
                           int64_t inputBytes)
{
    constexpr auto K_U32_MAX_AS_I64 = static_cast<int64_t>(UINT32_MAX);
    const int64_t seqLenKv = k.dims()->Get(SDPA_SEQ_AXIS);

    auto extentFits
        = [&](const char* name, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& t) {
              const int64_t seqStride = t.strides()->Get(SDPA_SEQ_AXIS);
              // seqLenKv * seqStride * inputBytes <= UINT32_MAX, evaluated by division so
              // no product is formed.
              const bool fits = seqLenKv >= 0 && seqStride >= 0 && inputBytes > 0
                                && (seqLenKv == 0 || seqStride == 0
                                    || seqStride <= K_U32_MAX_AS_I64 / inputBytes / seqLenKv);
              if(!fits)
              {
                  HIPDNN_PLUGIN_LOG_INFO("SDPA: per-head row byte extent reaches 2^32 (tensor="
                                         << name << ", seqlen_kv=" << seqLenKv << ", seq_stride="
                                         << seqStride << ", elementBytes=" << inputBytes << ")");
              }
              return fits;
          };

    bool ok = true;
    ok &= extentFits("k", k);
    ok &= extentFits("v", v);
    return ok;
}

// Batch-mode kernels form each LSE row's byte offset (b * H + h) * s_lse_Hs in 32-bit
// arithmetic. The check bounds the byte offset of the last element,
// 4 * ((B * H - 1) * headStride + Sq - 1), to 32 bits, which covers the last row's base with
// Sq - 1 elements to spare. Evaluated by division so no product of graph values can overflow.
static bool batchLseOffsetsFitU32(int64_t batch, int64_t heads, int64_t headStride, int64_t seqLenQ)
{
    constexpr int64_t K_MAX_LSE_ELEMENTS = static_cast<int64_t>(UINT32_MAX) / K_FP32_BYTES;
    if(batch < 1 || heads < 1 || seqLenQ < 1 || seqLenQ > K_MAX_LSE_ELEMENTS
       || heads > K_MAX_LSE_ELEMENTS / batch)
    {
        return false;
    }
    const int64_t headSteps = (batch * heads) - 1;
    return headSteps == 0
           || (headStride >= 0 && headStride <= (K_MAX_LSE_ELEMENTS - (seqLenQ - 1)) / headSteps);
}

// Checks every byte offset the kernel forms in 32 bits: the per-field strides,
// the Ts product and, for hd192x128, the K/V per-head row extent. The stats (LSE)
// strides and extent are bounded in checkStatsTensor. `inputBytes` is the resolved
// Q/K/V element size and `tsQo` the matched row's Q-tile size.
static Rejection checkFwdByteRanges(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& q,
                                    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& k,
                                    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& v,
                                    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& o,
                                    int64_t inputBytes,
                                    int64_t tsQo)
{
    HIP_KERNEL_REJECT_IF(!wouldFwdByteStridesFitUint32(q, k, v, o, inputBytes),
                         "Forward byte strides overflow uint32_t kernarg fields");

    // s_Ts = tileSizeQo * qStrideRow * inBytesPerElement (SdpaFwdArgsBuilder) is
    // the one forward kernarg that is a product rather than a single stride, and
    // it is formed in unsigned int, so it wraps for Q sequence strides the
    // per-field check above accepts. The resolved row's ts_qo folds into the
    // element size so the product is never formed outside the predicate.
    // Mirrors the backward Ts check.
    HIP_KERNEL_REJECT_IF(!plan_utils::byteStrideFitsU32(
                             "Ts (ts_qo * stride_q)", q.strides()->Get(2), inputBytes * tsQo),
                         "Forward Ts product overflows the uint32_t s_Ts kernarg field");

    // The hd192x128 kernels on both architectures address each K and V head
    // through a 32-bit buffer offset, so a head whose rows span 2^32 bytes or
    // more is out of their reach even when every per-field stride fits. Ports
    // AITER's can_impl_fmha_v3_fwd rule (aiter/ops/mha.py, batch and varlen).
    const bool isHd192x128 = q.dims()->Get(3) == 192 && v.dims()->Get(3) == 128;
    HIP_KERNEL_REJECT_IF(isHd192x128 && !fwdKvRowExtentsFitU32(k, v, inputBytes),
                         "hd192x128 K or V per-head row extent reaches 2^32 bytes");
    return std::nullopt;
}

// True when no two elements of the view share an address: the axes longer than one,
// sorted by stride, each have a stride at least the span of the axes below them. A zero
// stride on such an axis overlaps. A span past int64 does not count as addressable.
static bool isNonOverlappingView(std::vector<std::pair<int64_t, int64_t>> strideExtents)
{
    std::sort(strideExtents.begin(), strideExtents.end());
    int64_t span = 1;
    for(const auto& [stride, extent] : strideExtents)
    {
        int64_t step = 0;
        if(stride < span || __builtin_mul_overflow(extent - 1, stride, &step)
           || __builtin_add_overflow(span, step, &span))
        {
            return false;
        }
    }
    return true;
}

// Q, K, V and O layouts the kernels cannot address. The forward argument block carries only
// batch, head and sequence strides, and each row is read or written as head-dim contiguous
// elements, so:
// - a head-dim stride other than 1 is taken as 1: rows read the gaps between elements, and a
//   strided O is left partly unwritten;
// - a sequence stride below the tensor's own head dim ends each (batch, head) slice early:
//   the tail of the last row reads as zero and writes there are dropped. A zero sequence
//   stride is declined with it: on Q and V it gives wrong results, and the kernels bound each
//   buffer at S * sequence stride. A zero K sequence stride (one key row broadcast over the
//   sequence) computed correctly where measured, so for K the decline is conservative.
// Q, K and V may broadcast a batch or head with a zero stride; the kernels only re-read a
// slice. O must not overlap itself, because each (batch, head) slice is written by its own
// workgroups and aliased slices overwrite each other. Group mode locates each batch through
// the Q/O offset table and ignores O's batch stride, so only O's head, sequence and head-dim
// axes are checked there; whether two batches overlap depends on the offset values, which
// only the device holds. Q, K, V and O dims and strides must already be known to be rank 4.
static Rejection checkFwdIoLayouts(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& q,
                                   const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& k,
                                   const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& v,
                                   const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& o,
                                   BatchMode batchMode)
{
    for(const auto& [label, tensor] :
        {std::pair{"q", &q}, std::pair{"k", &k}, std::pair{"v", &v}, std::pair{"o", &o}})
    {
        const auto* dims = tensor->dims();
        const auto* strides = tensor->strides();
        HIP_KERNEL_REJECT_IF(strides->Get(3) != 1,
                             std::string(label) + " tensor head-dim stride must be 1 (Actual: "
                                 + std::to_string(strides->Get(3)) + ")");
        HIP_KERNEL_REJECT_IF(strides->Get(SDPA_SEQ_AXIS) < dims->Get(3),
                             std::string(label)
                                 + " tensor sequence stride must be at least its head dim (Actual "
                                 + "sequence stride: " + std::to_string(strides->Get(SDPA_SEQ_AXIS))
                                 + ", head dim: " + std::to_string(dims->Get(3)) + ")");
    }

    std::vector<std::pair<int64_t, int64_t>> oStrideExtents;
    const flatbuffers::uoffset_t firstAxis = (batchMode == BatchMode::GROUP) ? 1 : 0;
    for(flatbuffers::uoffset_t axis = firstAxis; axis < 4; ++axis)
    {
        if(o.dims()->Get(axis) > 1)
        {
            oStrideExtents.emplace_back(o.strides()->Get(axis), o.dims()->Get(axis));
        }
    }
    HIP_KERNEL_REJECT_IF(!isNonOverlappingView(std::move(oStrideExtents)),
                         "o tensor strides make output elements share an address");
    return std::nullopt;
}

unsigned int SdpaFwdPlanBuilder::maxThreadsPerBlock(const fmha_v3_fwdConfig& config) const
{
    return static_cast<unsigned int>(config.max_flat_workgroup_size);
}

std::optional<std::string> SdpaFwdPlanBuilder::deviceArch(const Handle& handle) const
{
    return plan_utils::tryGetDeviceString(
        handle.getStream(), "[SdpaFwdPlanBuilder::isApplicable] Could not query device string: ");
}

unsigned int fwdConfigBlockDimX(const fmha_v3_fwdConfig& config)
{
    return fwdBlockDimX(config.arch,
                        static_cast<unsigned int>(config.hdim_q),
                        static_cast<unsigned int>(config.hdim_v),
                        config.dtype == "bf16",
                        static_cast<BatchMode>(config.mode) == BatchMode::GROUP);
}

// A kernel launched with more threads per block than its code object's
// max_flat_workgroup_size fails at hipModuleLaunchKernel.
std::optional<std::string>
    SdpaFwdPlanBuilder::launchLimitViolation(const fmha_v3_fwdConfig& config) const
{
    const unsigned int blockDimX = fwdConfigBlockDimX(config);
    const unsigned int maxThreads = maxThreadsPerBlock(config);
    if(blockDimX <= maxThreads)
    {
        return std::nullopt;
    }
    return "Kernel " + config.knl_name + " accepts at most " + std::to_string(maxThreads)
           + " threads per block but is launched with " + std::to_string(blockDimX);
}

// Group-mode (ragged) layout: only Q/K/V/O and the stats tensor may be ragged, Q/K/V/O all
// of them or none, and a ragged layout must match AITER's seqstart addressing.
// checkStatsTensor decides whether a ragged stats layout is one the kernel writes.
static Rejection
    checkRaggedLayout(const plan_utils::TensorMap& tensorMap,
                      const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attrs,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& qTensor,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& kTensor,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& vTensor,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& oTensor)
{
    std::unordered_set<int64_t> raggedCapableUids{
        attrs.q_tensor_uid(), attrs.k_tensor_uid(), attrs.v_tensor_uid(), attrs.o_tensor_uid()};
    if(hasStatsOutput(attrs) && attrs.stats_tensor_uid().has_value())
    {
        raggedCapableUids.insert(attrs.stats_tensor_uid().value());
    }
    const auto unsupportedRaggedTensorIds
        = hipdnn_flatbuffers_sdk::utilities::listUnsupportedRaggedTensorIds(tensorMap,
                                                                            raggedCapableUids);
    HIP_KERNEL_REJECT_IF(
        !unsupportedRaggedTensorIds.empty(),
        "Only the Q, K, V, O and stats tensors can be ragged. The following tensor uids are "
        "ragged and not supported: "
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
    // Dropout operands the kernels never read: a graph that carries one asks for dropout
    // or its debug output, which these kernels do not produce. seed and offset are inert
    // at dropout_probability 0, and a non-zero probability is declined with the common
    // features.
    HIP_KERNEL_REJECT_IF(attrs.dropout_mask_tensor_uid(), "dropout_mask tensor not supported");
    HIP_KERNEL_REJECT_IF(attrs.dropout_scale_tensor_uid(), "dropout_scale tensor not supported");
    HIP_KERNEL_REJECT_IF(attrs.rng_dump_tensor_uid(), "rng_dump tensor not supported");
    HIP_KERNEL_REJECT_IF(attrs.block_mask_tensor_uid(), "block_mask tensor not supported");
    // The plan writes only the output and the optional stats tensor, never max or sum_exp.
    HIP_KERNEL_REJECT_IF(attrs.max_tensor_uid() || attrs.sum_exp_tensor_uid(),
                         "max and sum_exp output tensors not supported");
    // max_seq_len_kv sizes a paged K/V cache, which is not honored.
    HIP_KERNEL_REJECT_IF(attrs.max_seq_len_kv().has_value(), "max_seq_len_kv not supported");
    // The plan writes a stats tensor only when generate_stats asks for one.
    HIP_KERNEL_REJECT_IF(attrs.stats_tensor_uid().has_value() && !hasStatsOutput(attrs),
                         "stats_tensor_uid is set but generate_stats is not");
    HIP_KERNEL_REJECT_IF(attrs.page_table_k_tensor_uid(), "page_table_k tensor not supported");
    HIP_KERNEL_REJECT_IF(attrs.page_table_v_tensor_uid(), "page_table_v tensor not supported");
    // Attention sinks join the softmax denominator; the vendored kernels have no sink input.
    HIP_KERNEL_REJECT_IF(attrs.sink_token_tensor_uid(), "sink_token tensor not supported");

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
    HIP_KERNEL_RETURN_IF_REJECTED(plan_utils::checkDeviceBufferTensor(label, tensor));
    HIP_KERNEL_REJECT_IF(tensor.data_type() != expectedType,
                         std::string(label) + " tensor must be " + EnumNameDataType(expectedType)
                             + " (Actual value: " + EnumNameDataType(tensor.data_type()) + ")");
    return std::nullopt;
}

// Batch mode writes FP32 [B, H, Sq] element (b, h, s) at (b * H + h) * s_lse_Hs + s: the
// kernel takes only a head stride (s_lse_Hs), with no batch or sequence stride, and steps
// batches by H * s_lse_Hs. The strides must therefore be [H * s_h, s_h, 1] with s_h >= Sq so
// rows of different heads do not overlap, and the last element's byte offset must fit in 32
// bits; any other layout is declined. stats must already be known to be [B, H, Sq] or
// [B, H, Sq, 1].
static Rejection
    checkBatchStatsLayout(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& stats)
{
    // The kernel steps batches by H * head stride even when H == 1, so the head stride is
    // checked whenever B > 1 or H > 1. The batch and sequence strides are unused, and not
    // checked, when their axis has size 1.
    const int64_t batch = stats.dims()->Get(0);
    const int64_t heads = stats.dims()->Get(1);
    const int64_t seqLenQ = stats.dims()->Get(2);
    const auto* strides = stats.strides();
    HIP_KERNEL_REJECT_IF(strides == nullptr || strides->size() != stats.dims()->size(),
                         "batch-mode stats tensor must have one stride per dim");
    // Checked first: it bounds the head stride, so the product below cannot overflow.
    HIP_KERNEL_REJECT_IF(!batchLseOffsetsFitU32(batch, heads, strides->Get(1), seqLenQ),
                         "batch-mode stats tensor LSE byte offset 4 * ((B * H - 1) * "
                         "head stride + Sq - 1) must fit in 32 bits");
    HIP_KERNEL_REJECT_IF(
        (batch != 1 && strides->Get(0) != heads * strides->Get(1))
            || ((batch != 1 || heads != 1) && strides->Get(1) < seqLenQ)
            || (seqLenQ != 1 && strides->Get(SDPA_SEQ_AXIS) != 1),
        "batch-mode stats tensor strides must be [H * s_h, s_h, 1] with head stride s_h >= Sq "
        "(the kernel derives the batch offset from the head stride)");
    return std::nullopt;
}

// Validates the stats (LSE) output tensor generate_stats requests. Batch mode's layout is
// checked by checkBatchStatsLayout. Group mode writes AITER's varlen layout [H, total_q]: one
// FP32 per Q token at h * s_lse_Hs + seqstart_q[b] + s, with no per-token stride, so the stats
// tensor must be ragged on the Q/O offset table in token units with a sequence stride of 1 and,
// with more than one head, a head stride of at least B * Sq. Q must already be validated rank 4.
static Rejection checkStatsTensor(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attrs,
                                  const plan_utils::TensorMap& tensorMap,
                                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& q,
                                  const std::optional<SdpaFwdGroupModeParams>& group)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
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

    const auto statsOffsetUid = stats.ragged_offset_tensor_uid();
    HIP_KERNEL_REJECT_IF(!group.has_value() && statsOffsetUid.has_value(),
                         "stats tensor must not be ragged when Q, K, V and O are dense");
    HIP_KERNEL_REJECT_IF(
        group.has_value()
            && (!statsOffsetUid.has_value() || statsOffsetUid.value() != group->qoRaggedOffsetUid
                || stats.strides() == nullptr || stats.strides()->size() != statsRank
                || stats.strides()->Get(SDPA_SEQ_AXIS) != 1
                || stats.ragged_offset_multiplier() != 1),
        "group-mode stats tensor must be ragged on the Q/O offset tensor with a sequence "
        "stride and ragged_offset_multiplier of 1 (AITER's [H, total_q] LSE layout)");

    HIP_KERNEL_REJECT_IF(stats.dims()->Get(0) != q.dims()->Get(0)
                             || stats.dims()->Get(1) != q.dims()->Get(1)
                             || stats.dims()->Get(2) != q.dims()->Get(2),
                         "stats tensor shape [B,H,Sq] must match Q tensor");
    if(!group.has_value())
    {
        HIP_KERNEL_RETURN_IF_REJECTED(checkBatchStatsLayout(stats));
    }
    else
    {
        // Heads write disjoint slots only when the head stride covers every physical Q token
        // row: seqstart_q[b] + len_b <= head stride for every batch. That row count lives in the
        // device-resident offset table, so the check takes B * Sq, the Q token capacity of a
        // packed table. Sq is only a necessary lower bound: the frontend's default stats strides
        // [H * Sq, Sq, 1, 1] meet it, yet with B > 1 a batch at offset Sq or later writes the
        // next head's slots. B * Sq declines AITER's packed [H, total_q] layout whenever
        // total_q < B * Sq. Such a graph is not guaranteed a fallback: on gfx942 and gfx950 no
        // other engine may take it. A table whose rows reach past B * Sq stays the caller's
        // precondition (SdpaFwdArgsBuilder.hpp). The check also declines a zero (broadcast)
        // head stride, which the layout rule exempts but which makes every head write the
        // same slots. A single head is never stepped.
        const int64_t batch = stats.dims()->Get(0);
        const int64_t heads = stats.dims()->Get(1);
        const int64_t seqLenQ = stats.dims()->Get(2);
        const int64_t headStride = stats.strides()->Get(1);
        // headStride / batch < Sq is headStride < B * Sq without forming the product.
        HIP_KERNEL_REJECT_IF(heads != 1 && (batch < 1 || headStride / batch < seqLenQ),
                             "group-mode stats head stride must be at least B * Sq when "
                             "H > 1 (head stride: "
                                 + std::to_string(headStride) + ", B: " + std::to_string(batch)
                                 + ", Sq: " + std::to_string(seqLenQ) + ")");
    }

    // Only the FP32 head stride reaches the kernel, which steps batches by H * head stride.
    // The batch step is checked only once the head stride is known to fit.
    const auto* statsStrides = stats.strides();
    HIP_KERNEL_REJECT_IF(
        !plan_utils::byteStrideFitsU32("nhead_stride_lse", statsStrides->Get(1), K_FP32_BYTES)
            || !plan_utils::byteStrideFitsU32(
                "batch_stride_lse", stats.dims()->Get(1) * statsStrides->Get(1), K_FP32_BYTES),
        "stats byte strides overflow uint32_t kernarg fields");
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
    // fp8 forward kernels are wired in, and port with them AITER's power-of-two GQA
    // ratio requirement for fp8 (can_impl_fmha_v3_fwd in aiter/ops/mha.py), which the
    // gfx942 fp8 objects do not need because they divide by the ratio in full.
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

// mma_core_mode selects the MMA path of the attention matmuls (cuDNN semantics); it is
// not the element or the accumulator type. The bf16 kernels run the 16-bit path, which
// cuDNN's Graph::sdpa() and hipDNN's cuDNN shim write as HALF for fp16 and bf16 alike;
// BFLOAT16 is accepted as its equivalent. The fp8 kernels run the FP8 path, which cuDNN
// spells FP8_E4M3 (the sdpa_fp8() default) or FP8_E5M2 and treats as one path. The mode
// does not name the input encoding or the precision: that comes from the tensor types,
// checked above (FP8_E4M3_FNUZ only). UNSET leaves the choice to the engine. An
// allow-list, so an enum value added later declines until judged.
static Rejection checkMmaCoreMode(hipdnn_flatbuffers_sdk::data_objects::DataType mmaCoreMode,
                                  const std::string& dataTypeId,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType inputType)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    const bool isSixteenBitMmaMode
        = mmaCoreMode == DataType::HALF || mmaCoreMode == DataType::BFLOAT16;
    const bool mmaCoreModeSupported
        = mmaCoreMode == DataType::UNSET || (dataTypeId == "bf16" && isSixteenBitMmaMode)
          || (dataTypeId == "fp8bf16"
              && (mmaCoreMode == DataType::FP8_E4M3 || mmaCoreMode == DataType::FP8_E5M2));
    HIP_KERNEL_REJECT_IF(
        !mmaCoreModeSupported,
        "mma_core_mode must be UNSET, or HALF or BFLOAT16 for bf16 inputs, or FP8_E4M3 or "
        "FP8_E5M2 for FP8_E4M3_FNUZ inputs (Actual value: "
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
    const TensorAttributes* qDescale = nullptr;
    const TensorAttributes* kDescale = nullptr;
    const TensorAttributes* vDescale = nullptr;
    HIP_KERNEL_RETURN_IF_REJECTED(plan_utils::findRequiredTensor(
        tensorMap, "descale_q", attrs.descale_q_tensor_uid().value(), qDescale));
    HIP_KERNEL_RETURN_IF_REJECTED(plan_utils::findRequiredTensor(
        tensorMap, "descale_k", attrs.descale_k_tensor_uid().value(), kDescale));
    HIP_KERNEL_RETURN_IF_REJECTED(plan_utils::findRequiredTensor(
        tensorMap, "descale_v", attrs.descale_v_tensor_uid().value(), vDescale));
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

Rejection SdpaFwdPlanBuilder::checkFwdApplicability(
    const Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    HIP_KERNEL_RETURN_IF_REJECTED(
        plan_utils::checkSdpaGraphBasics(opGraph, NodeAttributes::SdpaAttributes));

    const auto deviceStringOpt = deviceArch(handle);
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

    const TensorAttributes* qTensor = nullptr;
    const TensorAttributes* kTensor = nullptr;
    const TensorAttributes* vTensor = nullptr;
    const TensorAttributes* oTensor = nullptr;
    HIP_KERNEL_RETURN_IF_REJECTED(
        plan_utils::findRequiredTensor(tensorMap, "q", attrs.q_tensor_uid(), qTensor));
    HIP_KERNEL_RETURN_IF_REJECTED(
        plan_utils::findRequiredTensor(tensorMap, "k", attrs.k_tensor_uid(), kTensor));
    HIP_KERNEL_RETURN_IF_REJECTED(
        plan_utils::findRequiredTensor(tensorMap, "v", attrs.v_tensor_uid(), vTensor));
    HIP_KERNEL_RETURN_IF_REJECTED(
        plan_utils::findRequiredTensor(tensorMap, "o", attrs.o_tensor_uid(), oTensor));

    // Validate Q/K/V/O before accessing their dims and strides by index.
    for(const auto& [label, tensor] : {std::pair{"q", qTensor},
                                       std::pair{"k", kTensor},
                                       std::pair{"v", vTensor},
                                       std::pair{"o", oTensor}})
    {
        HIP_KERNEL_RETURN_IF_REJECTED(plan_utils::checkRank4DeviceTensor(label, *tensor));
        HIP_KERNEL_REJECT_IF(tensor->strides() == nullptr || tensor->strides()->size() != 4,
                             std::string(label) + " tensor must have rank 4 strides");
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

    // The kernel derives the GQA ratio as Hq / Hkv and indexes K, V and O with Q's batch,
    // so the four tensors must agree on every extent they share.
    HIP_KERNEL_REJECT_IF(
        !plan_utils::hasConsistentFwdShapes(*qTensor, *kTensor, *vTensor, *oTensor),
        "Q, K, V and O dimensions are inconsistent: every dimension must be positive, "
        "K and V must share batch, head count and sequence length, the batch must match Q, "
        "K's head size must match Q's, Hq must be a multiple of Hkv, and O must be "
        "[B, Hq, Sq, Dv]");

    HIP_KERNEL_RETURN_IF_REJECTED(
        checkFwdIoLayouts(*qTensor, *kTensor, *vTensor, *oTensor, batchMode));

    HIP_KERNEL_RETURN_IF_REJECTED(checkSequenceLengths(attrs, tensorMap, *qTensor, batchMode));
    const auto group = (batchMode == BatchMode::GROUP)
                           ? std::optional{makeGroupModeParams(attrs, *qTensor, *kTensor)}
                           : std::nullopt;
    if(hasStatsOutput(attrs))
    {
        HIP_KERNEL_RETURN_IF_REJECTED(checkStatsTensor(attrs, tensorMap, *qTensor, group));
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
    maskType = fwdKernelMaskType(maskType, hasEqualPerBatchLengths(*qTensor, *kTensor, group));

    auto key = fwdKernelNameKey(deviceString,
                                dataTypeId,
                                static_cast<int>(qTensor->dims()->Get(3)),
                                static_cast<int>(vTensor->dims()->Get(3)),
                                maskType,
                                getBf16ConvertMode(attrs, dataTypeId),
                                batchMode);

    HIP_KERNEL_REJECT_IF(
        key.empty(),
        "Could not find matching kernel for parameter combination. dataType = " + dataTypeId
            + " qDim = " + std::to_string(qTensor->dims()->Get(3))
            + " vDim = " + std::to_string(vTensor->dims()->Get(3))
            + " mask = " + std::to_string(static_cast<int>(maskType)) + " rounding = "
            + std::to_string(static_cast<int>(getBf16ConvertMode(attrs, dataTypeId)))
            + " batch = " + std::to_string(static_cast<int>(batchMode)));

    HIP_KERNEL_REJECT_IF(!kernelServesHeadCounts(deviceString, dataTypeId, *qTensor, *kTensor),
                         "gfx950 bf16 head dim 256 kernels do not support GQA/MQA (q heads: "
                             + std::to_string(qTensor->dims()->Get(1))
                             + ", kv heads: " + std::to_string(kTensor->dims()->Get(1)) + ")");

    // Q/K/V byte strides and the Ts product are scaled by the resolved input
    // size, which the datatype checks above have narrowed to bf16 or fp8bf16.
    const auto& config = cfg_fmha_fwd.at(key);
    HIP_KERNEL_RETURN_IF_REJECTED(checkFwdByteRanges(
        *qTensor, *kTensor, *vTensor, *oTensor, fwdInputBytesPerElement(dataTypeId), config.ts_qo));

    // Decline a kernel that cannot take the block it would be launched with, so
    // the graph falls through to another engine instead of failing at launch.
    HIP_KERNEL_RETURN_IF_REJECTED(launchLimitViolation(config));
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

    // isApplicable ensures all of q, k, v and o tensors are ragged if any are, that O and V
    // share the Q and K offset tensors respectively, and that cu_seq_len_q/kv come together
    // and only in group mode.
    const auto batchMode
        = qTensor->ragged_offset_tensor_uid().has_value() ? BatchMode::GROUP : BatchMode::BATCH;

    if(batchMode == BatchMode::GROUP)
    {
        params.group = makeGroupModeParams(sdpaAttrs, *qTensor, *kTensor);
    }

    // Classify on the graph dims, exactly as isApplicable does. Passing the
    // narrowed unsigned locals instead would let the two sites resolve
    // different mask types for the same graph.
    const auto band = plan_utils::makeCoveringBoundsUnbound(
        plan_utils::resolveDiagonalBand(sdpaAttrs), qDims->Get(2), kDims->Get(2));
    params.maskType = fwdKernelMaskType(
        plan_utils::getMaskType(band.leftBound, band.rightBound, band.alignment),
        hasEqualPerBatchLengths(*qTensor, *kTensor, params.group));

    // Find matching kernel to graph
    fmha_v3_fwdConfig config;
    auto kernelKey = fwdKernelNameKey(deviceString,
                                      dataTypeId,
                                      static_cast<int>(headDimQk),
                                      static_cast<int>(headDimV),
                                      params.maskType,
                                      getBf16ConvertMode(sdpaAttrs, dataTypeId),
                                      batchMode);

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

    if(const auto launchViolation = launchLimitViolation(config))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "SdpaFwdPlanBuilder::buildPlan: "
                                                           + *launchViolation);
    }

    params.tileSizeQo = static_cast<unsigned int>(config.ts_qo);

    // Load kernel module from kpack archive
    auto tocKey = asm_kernels::getFwdKernelTocKey(config.co_name, deviceString, isMi308);

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
