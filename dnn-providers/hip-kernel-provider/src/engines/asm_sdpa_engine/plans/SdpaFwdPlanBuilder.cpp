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
#include <hip_kernel_provider_common/SdpaConfigEnumerations.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/DeviceQuery.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
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

// Validate that every forward-pass byte stride fits in uint32_t.  The ASM
// kernarg struct stores strides as uint32 so values that overflow silently
// truncate, producing wrong results.  Checked early in isApplicable so the
// engine declines rather than dispatching with bad strides.
static bool wouldFwdByteStridesFitUint32(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& q,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& k,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& v,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& o,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* stats)
{
    constexpr int64_t K_BF16_BYTES = 2;
    constexpr int64_t K_FP32_BYTES = 4;

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
    ok &= checkTensor("q", q, K_BF16_BYTES);
    ok &= checkTensor("k", k, K_BF16_BYTES);
    ok &= checkTensor("v", v, K_BF16_BYTES);
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

// Exclusive bounds on the sequence byte stride of Q, K, V and O. A kernel that
// forms a tensor's row offsets with v_mul_i32_i24 addresses its rows correctly
// only while the stride stays below 2^23 bytes; the fp8 kernels multiply
// 16 * stride for Q and K, which lowers their bound to 2^19 bytes. Which tensors
// go through that multiply differs by kernel (from the disassembly of the
// vendored forward .co files; MI300 and MI308 builds and the causal variants
// agree):
// - gfx942 bf16 hd128: O.
// - gfx942 bf16 hd192x128: Q, K and O.
// - gfx942 fp8 hd128: Q and K (16 * stride), V and O.
// - gfx950: Q, K, V and O.
// The other strides keep the 2^32 bound of their uint32 kernarg field. The
// gfx942 bf16 bounds were checked on a GPU; the fp8 and gfx950 ones were not.
struct FwdSeqStrideBounds
{
    uint64_t q;
    uint64_t k;
    uint64_t v;
    uint64_t o;
};

static FwdSeqStrideBounds getFwdSeqStrideBounds(const std::string& archId,
                                                const std::string& dataTypeId,
                                                int64_t headDimQk)
{
    constexpr uint64_t K_I24 = plan_utils::K_I24_STRIDE_BOUND;
    constexpr uint64_t K_FP8_QK = K_I24 / 16;
    constexpr uint64_t K_FIELD = plan_utils::K_U32_ADDRESS_MAX + 1;
    if(archId == "gfx950")
    {
        return {K_I24, K_I24, K_I24, K_I24};
    }
    if(dataTypeId == "fp8bf16")
    {
        return {K_FP8_QK, K_FP8_QK, K_I24, K_I24};
    }
    if(headDimQk == 192)
    {
        return {K_I24, K_I24, K_FIELD, K_I24};
    }
    return {K_FIELD, K_FIELD, K_FIELD, K_I24};
}

// Largest power of two that divides `bytes`, capped at 2^32 (2^32 for 0). Every
// lo32(b * bytes) is a multiple of it.
static uint64_t lowestSetBitUpTo2Pow32(uint64_t bytes)
{
    constexpr uint64_t K_2_POW_32 = uint64_t{1} << 32;
    const uint64_t lowestSetBit = bytes & (~bytes + 1);
    return (lowestSetBit == 0 || lowestSetBit > K_2_POW_32) ? K_2_POW_32 : lowestSetBit;
}

// Validate that the selected forward kernel forms every Q/K/V/O/LSE offset
// correctly. wouldFwdByteStridesFitUint32 bounds each stride field on its own;
// every forward kernel also combines the strides as follows (from the .co
// disassembly), and an offset that overflows gives a wrong O or LSE with no
// error. Each overflow was confirmed on gfx942 unless marked otherwise:
// - Slice base. The kernel adds b * Bs to the base pointer as a 64-bit value,
//   but forms lo32(b * Bs) + h * Hs in 32 bits and drops its carry (h is the
//   KV head for K and V). With Bs = 2^32 - 2^15 and Hs = 2^16 bytes, slice
//   (1, 1) of Q, K or O was read or written 4 GiB low. Every lo32(b * Bs) is a
//   multiple of the largest power of two dividing Bs, so nothing carries when
//   (H - 1) * Hs is below it, which keeps packed power-of-two shapes over
//   4 GiB. Otherwise (B - 1) * Bs + (H - 1) * Hs must fit 32 bits.
// - Buffer size. Each (batch, head) buffer ends S * Seqs bytes after its base,
//   a 32-bit product, and reads past the end return 0: K with Seqs = 2^24 bytes
//   at S_kv = 256, and an SBHD decode with S_kv = 2^20 and 4 GiB K and V, gave
//   O = 0. The kernel also stores the O rows of the last Q tile past S_q, up to
//   the next multiple of ts_qo, at r * Seqs in 32 bits, and a store that wraps
//   lands back inside the buffer, so O is bounded at that row count. With
//   S_q = 1025 and an O stride of 4186128 bytes, the padded rows of the last
//   tile overwrote rows of the first.
// - Row offset. v_mul_i32_i24 reproduces a sequence byte stride only below
//   2^23 (getFwdSeqStrideBounds lists the tensors each kernel passes through
//   it). On hd128 bf16 an O stride of 2^23 bytes permuted the O rows, while Q,
//   K and V strides of 2^23 bytes were correct. On hd192x128 a Q or K stride
//   of 2^23 bytes gave a wrong O, and 2^23 - 128 bytes did not.
// - LSE. The base of slice (b, h), (b * H_q + h) * lse_Hs * 4, is formed in 32
//   bits and added to the 64-bit LSE pointer; the rows are buffer offsets from
//   that base, so only the base must fit. With H_q = 3 and lse_Hs = 2^30 - 1
//   elements, the LSE of head 2 was written 4 GiB low and its own rows were
//   left unwritten, while rows of head 1 more than 2^32 bytes from the LSE
//   pointer were correct.
// Logs the first overflow found and returns false; returns true when every
// offset fits.
static bool wouldFwdOffsetsFitKernelArithmetic(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& q,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& k,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& v,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& o,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* stats,
    const std::string& archId,
    const std::string& dataTypeId,
    const fmha_v3_fwdConfig& config)
{
    using hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
    using plan_utils::lastIndex;
    using plan_utils::saturatingProduct;
    using plan_utils::saturatingSum;
    // NOLINTNEXTLINE(readability-identifier-naming)
    static const char* HIP_KERNEL_LOG_PREFIX = "[SdpaFwdPlanBuilder::isApplicable] ";
    constexpr uint64_t K_U32_MAX = plan_utils::K_U32_ADDRESS_MAX;

    // Q/K/V elements are 1 byte for fp8 and 2 for bf16; O is BF16, LSE is FP32.
    const int64_t inputBytes = (dataTypeId == "fp8bf16") ? 1 : 2;
    constexpr int64_t K_OUTPUT_BYTES = 2;
    constexpr int64_t K_LSE_BYTES = 4;

    const int64_t seqLenQ = q.dims()->Get(2);
    const int64_t paddedSeqLenQ = plan_utils::roundUpToTile(seqLenQ, config.ts_qo);
    const auto seqStrideBounds = getFwdSeqStrideBounds(archId, dataTypeId, q.dims()->Get(3));

    auto toString = [](const flatbuffers::Vector<int64_t>* values) {
        return hipdnn_data_sdk::utilities::vecToString(
            hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(values));
    };

    struct AddressedTensor
    {
        const char* name;
        const TensorAttributes* tensor;
        int64_t elementBytes;
        // Rows the kernel addresses from each (batch, head) base.
        int64_t rows;
        // Exclusive bound on the sequence byte stride.
        uint64_t seqStrideBound;
    };
    for(const AddressedTensor& io :
        {AddressedTensor{"q", &q, inputBytes, seqLenQ, seqStrideBounds.q},
         AddressedTensor{"k", &k, inputBytes, k.dims()->Get(2), seqStrideBounds.k},
         AddressedTensor{"v", &v, inputBytes, v.dims()->Get(2), seqStrideBounds.v},
         AddressedTensor{"o", &o, K_OUTPUT_BYTES, paddedSeqLenQ, seqStrideBounds.o}})
    {
        const auto* dims = io.tensor->dims();
        const auto* strides = io.tensor->strides();
        const auto describe = [&]() {
            return std::string(io.name) + " tensor (Actual dims: " + toString(dims)
                   + ", strides: " + toString(strides) + ")";
        };

        const uint64_t batchSpan
            = saturatingProduct({lastIndex(dims->Get(0)), strides->Get(0), io.elementBytes});
        const uint64_t headSpan
            = saturatingProduct({lastIndex(dims->Get(1)), strides->Get(1), io.elementBytes});
        HIP_KERNEL_RETURN_FALSE_IF(
            saturatingSum({batchSpan, headSpan}) > K_U32_MAX
                && headSpan >= lowestSetBitUpTo2Pow32(
                       saturatingProduct({strides->Get(0), io.elementBytes})),
            "batch and head byte offsets carry out of the kernel's 32-bit slice offset for "
                + describe());
        HIP_KERNEL_RETURN_FALSE_IF(
            saturatingProduct({io.rows, strides->Get(2), io.elementBytes}) > K_U32_MAX,
            std::to_string(io.rows)
                + " rows of the sequence stride overflow the kernel's 32-bit buffer size for "
                + describe());
        HIP_KERNEL_RETURN_FALSE_IF(
            saturatingProduct({strides->Get(2), io.elementBytes}) >= io.seqStrideBound,
            "sequence byte stride must be below " + std::to_string(io.seqStrideBound)
                + " for the kernel's 24-bit row-offset multiply for " + describe());
    }

    if(stats != nullptr)
    {
        // The kernel takes only the stats head stride (lse_Hs); the batch stride
        // it uses is H_q * lse_Hs. The last slice base is (B * H_q - 1) * lse_Hs * 4.
        const int64_t batchSize = q.dims()->Get(0);
        const int64_t numHeadsQ = q.dims()->Get(1);
        const int64_t lseHeadStride = stats->strides()->Get(1);
        const uint64_t lastSliceBase = saturatingSum(
            {saturatingProduct({lastIndex(batchSize), numHeadsQ, lseHeadStride, K_LSE_BYTES}),
             saturatingProduct({lastIndex(numHeadsQ), lseHeadStride, K_LSE_BYTES})});
        const auto describeStats = [&]() {
            return "stats tensor (Actual dims: " + toString(stats->dims()) + ", strides: "
                   + toString(stats->strides()) + ") for q tensor dims " + toString(q.dims());
        };
        HIP_KERNEL_RETURN_FALSE_IF(
            lastSliceBase > K_U32_MAX,
            "LSE batch and head byte offsets overflow the kernel's 32-bit slice offset for "
                + describeStats());
    }

    return true;
}

bool SdpaFwdPlanBuilder::isApplicable(
    const Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    // NOLINTNEXTLINE(readability-identifier-naming)
    static const char* HIP_KERNEL_LOG_PREFIX = "[SdpaFwdPlanBuilder::isApplicable] ";

    // Execute-time override shapes can diverge from the compile-time dims this
    // builder matched exactly; the family serves fixed prebuilt shapes, so decline
    // rather than risk a mismatch (RFC 0008 §4.6).
    HIP_KERNEL_RETURN_FALSE_IF(opGraph.getGraph().is_override_shape_enabled(),
                               "Graph has override shapes enabled");

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

    HIP_KERNEL_RETURN_FALSE_IF(attrs.mma_core_mode() != DataType::UNSET,
                               "mma_core_mode must be unset");

    const auto& tensorMap = opGraph.getTensorMap();

    const int64_t qUid = attrs.q_tensor_uid();
    const int64_t kUid = attrs.k_tensor_uid();
    const int64_t vUid = attrs.v_tensor_uid();
    const int64_t oUid = attrs.o_tensor_uid();

    auto* qTensor = tensorMap.at(qUid);
    auto* kTensor = tensorMap.at(kUid);
    auto* vTensor = tensorMap.at(vUid);
    auto* oTensor = tensorMap.at(oUid);

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

    // Validate optional stats (LSE) output tensor.
    // Gate purely on generate_stats; the UID is required when the flag is set.
    const bool hasStats = hasStatsOutput(attrs);
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
                                getBatchMode(attrs),
                                &cfg_fmha_fwd);

    HIP_KERNEL_RETURN_FALSE_IF(key.empty(),
                               "Could not find matching kernel for parameter combination");

    HIP_KERNEL_RETURN_FALSE_IF(
        !wouldFwdByteStridesFitUint32(*qTensor, *kTensor, *vTensor, *oTensor, statsTensor),
        "Forward byte strides overflow uint32_t kernarg fields");

    HIP_KERNEL_RETURN_FALSE_IF(!wouldFwdOffsetsFitKernelArithmetic(*qTensor,
                                                                   *kTensor,
                                                                   *vTensor,
                                                                   *oTensor,
                                                                   statsTensor,
                                                                   deviceString,
                                                                   dataTypeId,
                                                                   cfg_fmha_fwd.at(key)),
                               "Forward tensor offsets overflow the kernel's 32-bit or 24-bit "
                               "address arithmetic");

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
    int64_t lseUid = -1;
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
        lseUid = sdpaAttrs.stats_tensor_uid().value();
        const auto lseIt = tensorMap.find(lseUid);
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
    const auto band = plan_utils::makeCoveringBoundsUnbound(
        plan_utils::resolveDiagonalBand(sdpaAttrs), seqLenQ, seqLenKv);
    params.maskType = plan_utils::getMaskType(band.leftBound, band.rightBound, band.alignment);

    const auto dataTypeId = getDataTypeIdentifier(
        qTensor->data_type(), kTensor->data_type(), vTensor->data_type(), oTensor->data_type());

    // FP8 inputs are 1 byte; bf16 inputs are 2 bytes. Output stays 2-byte BF16.
    params.inBytesPerElement = (dataTypeId == "fp8bf16") ? 1U : 2U;

    // FP8 requires q/k/v descales (guaranteed present by isApplicable). They are set as
    // a unit — the SdpaFwdParams::DescaleUids optional encodes the all-or-none invariant.
    if(dataTypeId == "fp8bf16")
    {
        params.descaleUids = SdpaFwdParams::DescaleUids{sdpaAttrs.descale_q_tensor_uid().value(),
                                                        sdpaAttrs.descale_k_tensor_uid().value(),
                                                        sdpaAttrs.descale_v_tensor_uid().value()};
    }

    // Find matching kernel to graph
    fmha_v3_fwdConfig config;
    auto kernelKey = getKernelNameKey(deviceString,
                                      dataTypeId,
                                      static_cast<int>(headDimQk),
                                      static_cast<int>(headDimV),
                                      params.maskType,
                                      getBf16ConvertMode(sdpaAttrs, dataTypeId),
                                      getBatchMode(sdpaAttrs),
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
