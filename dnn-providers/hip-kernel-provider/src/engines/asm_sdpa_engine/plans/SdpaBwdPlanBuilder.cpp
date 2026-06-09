// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaBwdPlanBuilder.hpp"
#include "asm/AsmKernelPath.hpp"
#include "asm_fmha_v3_bwd_configs.hpp"
#include "core/Utils.hpp"
#include "plans/SdpaBwdPlan.hpp"
#include "plans/SdpaPlanUtils.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hip_kernel_provider_common/SdpaConfigEnumerations.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

// Backward CSV columns consumed by this builder:
//   dtype, hdim_q, hdim_v, mask, atomic32, pssk, pddv, mode, bf16_cvt,
//   ts_qo, ts, knl_name, co_name, arch

namespace asm_sdpa_engine
{
namespace
{

// Dispatch enums shared with the forward path: MaskType via plan_utils;
// RoundingMode and BatchMode via hip_kernel_provider_common (the same header the
// forward builder consumes). AccumulatorMode stays backward-local because the
// forward dispatch has no 16/32-bit accumulator axis.
using hip_kernel_provider_common::BatchMode;
using hip_kernel_provider_common::RoundingMode;
using plan_utils::MaskType;

using bwd_dispatch::BF16_CVT_FP16_SENTINEL;

enum class AccumulatorMode : int
{
    A16 = 0, // 16-bit accumulator (atomic32 = 0)
    A32 = 1 // 32-bit accumulator (atomic32 = 1)
};

// Per-stage CSV-row selector. Computed once from the graph and consumed by
// both isApplicable and buildPlan, so the two sites cannot drift.
struct BwdDispatchTuple
{
    int mask;
    int atomic32;
    int pssk;
    int pddv;
    int bf16Cvt;
    // See "Verified Kernel Matrix" in asm/asm_kernels/README.md; true only for
    // calibrated (dtype, hdim).
    bool verified;
};

// Output of resolveStage: the .co file path, kernel symbol name, and tile
// sizes for the resolved registry row.
struct ResolvedKernel
{
    std::string coPath;
    std::string knlName;
    SdpaBwdParams::KernelTiles tiles;
};

// Per-pipeline-stage dispatch tuples for the three backward kernels.
struct BwdDispatchTuples
{
    BwdDispatchTuple odo;
    BwdDispatchTuple dqdkdv;
    BwdDispatchTuple dqConvert;
};

RoundingMode
    getRoundingMode(const hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes& /*attrs*/)
{
    // TODO(ALMIOPEN-1824): plumb rounding mode from graph; for now always RTNE.
    return RoundingMode::RTNE;
}

BatchMode getBatchMode(const hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes& attrs)
{
    return (attrs.seq_len_q_tensor_uid().has_value() || attrs.seq_len_kv_tensor_uid().has_value())
               ? BatchMode::GROUP
               : BatchMode::BATCH;
}

// Map the seven backward tensor dtypes to a CSV dtype identifier. The backward
// graph carries Q/K/V/dO inputs and dQ/dK/dV gradient outputs that all share
// a single floating-point type per the CSV schema; FP32 stats are validated
// separately. Returns std::nullopt when the seven tensors do not share a
// supported dtype (BF16 or FP16); FP8 falls into this bucket because the
// backward CSV does not define FP8 rows.
std::optional<std::string>
    tryGetDataTypeIdentifier(hipdnn_flatbuffers_sdk::data_objects::DataType qType,
                             hipdnn_flatbuffers_sdk::data_objects::DataType kType,
                             hipdnn_flatbuffers_sdk::data_objects::DataType vType,
                             hipdnn_flatbuffers_sdk::data_objects::DataType doType,
                             hipdnn_flatbuffers_sdk::data_objects::DataType dqType,
                             hipdnn_flatbuffers_sdk::data_objects::DataType dkType,
                             hipdnn_flatbuffers_sdk::data_objects::DataType dvType)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    const std::initializer_list<DataType> tensorTypes
        = {qType, kType, vType, doType, dqType, dkType, dvType};

    if(plan_utils::allDataTypesEqual(DataType::BFLOAT16, tensorTypes))
    {
        return std::string("bf16");
    }
    if(plan_utils::allDataTypesEqual(DataType::HALF, tensorTypes))
    {
        return std::string("fp16");
    }
    return std::nullopt;
}

// Walk the chosen registry and return a copy of the row that matches the
// requested tuple, or std::nullopt when no row matches.  The registry type
// (CFG) is the std::unordered_map alias emitted by codegen.py; returning a
// copy decouples the caller from the registry's storage so the result is
// stable regardless of any subsequent mutation to the underlying map.
std::optional<fmha_v3_bwdConfig> findConfig(const CFG& registry,
                                            const std::string& archId,
                                            const std::string& dataType,
                                            int hdimQ,
                                            int hdimV,
                                            int mask,
                                            int atomic32,
                                            int pssk,
                                            int pddv,
                                            int mode,
                                            int bf16Cvt)
{
    for(const auto& [unusedKey, cfg] : registry)
    {
        if(cfg.arch != archId)
        {
            continue;
        }
        if(cfg.dtype != dataType)
        {
            continue;
        }
        if(cfg.hdim_q != hdimQ || cfg.hdim_v != hdimV)
        {
            continue;
        }
        if(cfg.mask != mask)
        {
            continue;
        }
        if(cfg.atomic32 != atomic32)
        {
            continue;
        }
        if(cfg.pssk != pssk || cfg.pddv != pddv)
        {
            continue;
        }
        if(cfg.mode != mode)
        {
            continue;
        }
        // gfx950 BF16/FP16 rows are emitted with `bf16_cvt = 3` (the FP16
        // sentinel) regardless of the BF16 rounding mode the caller asked
        // for, because gfx950 ships only one kernel per (dtype, hdim, mask,
        // atomic, pssk, pddv, mode) tuple — there are no per-rounding-mode
        // variants to disambiguate.  Mirrors the equivalent special case
        // in SdpaFwdPlanBuilder.cpp::getKernelNameKey for gfx950.
        if(archId != "gfx950" && cfg.bf16_cvt != bf16Cvt)
        {
            continue;
        }
        return cfg;
    }
    return std::nullopt;
}

// Query the HIP device string for the stream, logging `logPrefix` on failure.
// Returns std::nullopt when the HIP runtime throws.
std::optional<std::string> tryGetDeviceString(hipStream_t stream, const char* logPrefix)
{
    try
    {
        return hip_kernel_provider_common::getDeviceString(stream);
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_ERROR(logPrefix << e.what());
        return std::nullopt;
    }
}

// Backward kernels live in a flat layout under
//   asm_kernels/<arch>/fmha_v3_bwd/<co_name>
// The codegen-emitted co_name already includes the "<arch>/fmha_v3_bwd/"
// prefix, so this helper simply resolves to the absolute install path.
// (Forward splits gfx942 into MI300/MI308 sub-folders and threads the arch
// through; backward does not because AITER ships a single backward set.)
std::string getKernelCoPath(const std::string& coName)
{
    return asm_kernels::getAsmKernelPath(coName);
}

constexpr int64_t K_BF16_BYTES = 2;
constexpr int64_t K_FP32_BYTES = 4;

// Validates that every byte stride consumed by the backward kernels fits in
// the uint32_t fields of the kernarg structs.  Catches overflow at
// applicability time so the engine can be excluded from heuristics for graphs
// it cannot represent, instead of failing at execute time.  Returns false (and
// logs the offending field) on any overflow; returns true when every stride is
// safe.
bool wouldBwdByteStridesFitUint32(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& q,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& k,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& v,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& o,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dO,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dq,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dk,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dv,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& stats,
    int64_t headsQ,
    int64_t seqLenQ,
    int64_t headDimQk,
    int64_t tsKv,
    bool useA32)
{
    auto check = [](const char* name, int64_t elements, int64_t elementBytes) {
        return plan_utils::byteStrideFitsU32(name, elements, elementBytes);
    };

    // [B, H, S, D] tensors: Get(0)=batch, Get(1)=head, Get(2)=seq.
    auto checkBwdTensor
        = [&](const char* prefix, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& t) {
              const auto* s = t.strides();
              bool ok = true;
              ok &= check((std::string("batch_stride_") + prefix).c_str(), s->Get(0), K_BF16_BYTES);
              ok &= check((std::string("nhead_stride_") + prefix).c_str(), s->Get(1), K_BF16_BYTES);
              ok &= check((std::string("stride_") + prefix).c_str(), s->Get(2), K_BF16_BYTES);
              return ok;
          };

    bool ok = true;
    ok &= checkBwdTensor("q", q);
    ok &= checkBwdTensor("k", k);
    ok &= checkBwdTensor("v", v);
    ok &= checkBwdTensor("o", o);
    ok &= checkBwdTensor("do", dO);
    ok &= checkBwdTensor("dq", dq);
    ok &= checkBwdTensor("dk", dk);
    ok &= checkBwdTensor("dv", dv);

    // LSE/D buffer: rank 3 [B, H_q, S_q] in FP32; only batch+head strides used.
    const auto* statsStrides = stats.strides();
    ok &= check("batch_stride_lsed", statsStrides->Get(0), K_FP32_BYTES);
    ok &= check("nhead_stride_lsed", statsStrides->Get(1), K_FP32_BYTES);

    // dq_acc: contiguous [B, H_q, S_q, D_qk] FP32, derived from dims (A32 only).
    if(useA32)
    {
        const int64_t strideDqAcc = headDimQk;
        const int64_t nheadStrideDqAcc = seqLenQ * headDimQk;
        const int64_t batchStrideDqAcc = headsQ * nheadStrideDqAcc;
        ok &= check("stride_dq_acc", strideDqAcc, K_FP32_BYTES);
        ok &= check("nhead_stride_dq_acc", nheadStrideDqAcc, K_FP32_BYTES);
        ok &= check("batch_stride_dq_acc", batchStrideDqAcc, K_FP32_BYTES);
    }

    // DQDKDV's `Ts` kernarg is the 3-way product (tsKv * stride_k * BF16).
    ok &= check(
        "Ts (tsKv * stride_k)", tsKv * static_cast<int64_t>(k.strides()->Get(2)), K_BF16_BYTES);

    return ok;
}

// Per-stage dispatch tuples differ. The odo (D-reduction) and dq_convert
// (FP32 -> output dtype cast) kernels are not parameterised by mask/
// accumulator/padding — every row in those CSVs has
//   mask=0, atomic32=0, pssk=0, pddv=0
// and odo additionally always uses the bf16_cvt=3 sentinel. The dqdkdv
// (main backward) kernel carries the full dispatch axes; its atomic32 column
// follows the resolved accumulator type (A32 → 3-kernel FP32-accumulator path,
// A16 → 2-kernel BF16 path), while pssk=1, pddv=1 are pinned because the
// registry rows for pssk=1, pddv=0 do not exist in batch mode and the unpadded
// (pssk=0, pddv=0) row uses a different kernarg layout than the engine builds
// today. TODO: lift the padding pin once the engine emits the unpadded kernarg
// layout for shapes where seqLenKv % tsKv == 0.
//
// The `verified` flag records whether the resolved (dtype, hdim) kernels have a
// CPU backward reference that has been calibrated against the in-tree kernels.
// It is keyed on (dtype, hdim) — not on pipeline stage — so all three stages of
// a dispatch share the same value. Today (bf16, hd128) and (fp16, hd128) are
// calibrated.
BwdDispatchTuples computeDispatchTuples(const std::string& dataType,
                                        int hdimQ,
                                        MaskType maskType,
                                        int bf16CvtValue,
                                        AccumulatorMode accMode)
{
    const bool verified = (dataType == "bf16" || dataType == "fp16") && hdimQ == 128;

    BwdDispatchTuples tuples{};
    tuples.odo = {0, 0, 0, 0, BF16_CVT_FP16_SENTINEL, verified};
    tuples.dqdkdv
        = {static_cast<int>(maskType), static_cast<int>(accMode), 1, 1, bf16CvtValue, verified};
    tuples.dqConvert = {0, 0, 0, 0, bf16CvtValue, verified};
    return tuples;
}

// Bridge the public AccumulatorType knob enum to the CSV-local AccumulatorMode
// used for the `atomic32` dispatch column.
AccumulatorMode toAccumulatorMode(AccumulatorType accType)
{
    return (accType == AccumulatorType::A32) ? AccumulatorMode::A32 : AccumulatorMode::A16;
}

// One-time dispatch logging. Each (dtype, hdim, mask) combination logs exactly
// once over the program lifetime: an INFO line for a CPU-reference-verified
// kernel, or a WARN line for an unverified one. The flag array is sized to the
// full enum cardinality (dtype x hdim x mask) so adding mask support later does
// not require resizing. Static storage duration; lifetime is the program
// duration.
constexpr size_t K_NUM_DTYPES = 2; // bf16, fp16
constexpr size_t K_NUM_HDIMS = 3; // hd64, hd128, hd192
constexpr size_t K_NUM_MASKS = 4; // plan_utils::MaskType cardinality
constexpr size_t K_LOG_FLAGS = K_NUM_DTYPES * K_NUM_HDIMS * K_NUM_MASKS;
std::array<std::once_flag, K_LOG_FLAGS> dispatchLogFlags;

// Map a (dtype, hdim, mask) triple to a flat index into dispatchLogFlags.
// isApplicable filters to the supported axes before buildPlan reaches this, so
// the asserts document the contract and the clamps are a defensive guard
// against an out-of-bounds index should an unexpected combination slip through.
size_t dispatchLogIndex(const std::string& dtype, int hdim, MaskType mask)
{
    const size_t dtypeIdx = (dtype == "fp16") ? 1U : 0U; // bf16 -> 0, fp16 -> 1

    size_t hdimIdx = 0U; // hd64 -> 0, hd128 -> 1, hd192 -> 2
    switch(hdim)
    {
    case 64:
        hdimIdx = 0U;
        break;
    case 128:
        hdimIdx = 1U;
        break;
    case 192:
        hdimIdx = 2U;
        break;
    default:
        assert(false && "dispatchLogIndex: unexpected head dimension");
        hdimIdx = 0U;
        break;
    }

    auto maskIdx = static_cast<size_t>(mask);
    assert(maskIdx < K_NUM_MASKS && "dispatchLogIndex: mask ordinal out of range");
    if(maskIdx >= K_NUM_MASKS)
    {
        maskIdx = 0U;
    }

    return (dtypeIdx * K_NUM_HDIMS + hdimIdx) * K_NUM_MASKS + maskIdx;
}

} // namespace

namespace bwd_dispatch
{

std::string lookupKernelNameKey(PipelineStage stage,
                                const std::string& archId,
                                const std::string& dataType,
                                int hdimQ,
                                int hdimV,
                                int mask,
                                int atomic32,
                                int pssk,
                                int pddv,
                                int mode,
                                int bf16Cvt)
{
    std::optional<fmha_v3_bwdConfig> cfg;
    switch(stage)
    {
    case PipelineStage::ODO:
        cfg = findConfig(cfg_fmha_bwd_odo,
                         archId,
                         dataType,
                         hdimQ,
                         hdimV,
                         mask,
                         atomic32,
                         pssk,
                         pddv,
                         mode,
                         bf16Cvt);
        break;
    case PipelineStage::DQDKDV:
        cfg = findConfig(cfg_fmha_bwd_dqdkdv,
                         archId,
                         dataType,
                         hdimQ,
                         hdimV,
                         mask,
                         atomic32,
                         pssk,
                         pddv,
                         mode,
                         bf16Cvt);
        break;
    case PipelineStage::DQ_CONVERT:
        cfg = findConfig(cfg_fmha_bwd_dq_convert,
                         archId,
                         dataType,
                         hdimQ,
                         hdimV,
                         mask,
                         atomic32,
                         pssk,
                         pddv,
                         mode,
                         bf16Cvt);
        break;
    default:
        break;
    }
    return cfg.has_value() ? cfg->arch + cfg->knl_name : std::string{};
}

} // namespace bwd_dispatch

// Engine knob identifier for accumulator precision (a32 vs a16)
static constexpr const char* K_ACC_TYPE_KNOB_NAME = "sdpa.bwd.accumulator_type";

bool SdpaBwdPlanBuilder::isApplicable(
    const Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    // NOLINTNEXTLINE(readability-identifier-naming)
    static const char* HIP_KERNEL_LOG_PREFIX = "[SdpaBwdPlanBuilder::isApplicable] ";

    auto& nodeWrappers = opGraph.nodeWrappers();

    auto deviceStringOpt
        = tryGetDeviceString(handle.getStream(), "Could not query device string: ");
    if(!deviceStringOpt)
    {
        return false;
    }
    const std::string& deviceString = *deviceStringOpt;

    // The codegen-generated registry contains both gfx942 and gfx950 rows;
    // only gfx942 is dispatched here.
    HIP_KERNEL_RETURN_FALSE_IF(deviceString != "gfx942",
                               "Device string does not match gfx942 (Actual value: " + deviceString
                                   + ")");

    HIP_KERNEL_RETURN_FALSE_IF(nodeWrappers.size() != 1, "Graph has more than one node");
    HIP_KERNEL_RETURN_FALSE_IF(nodeWrappers.front()->attributesType()
                                   != NodeAttributes::SdpaBackwardAttributes,
                               "Node attribute type is not SdpaBackwardAttributes");

    const auto& attrs = nodeWrappers.front()->attributesAs<SdpaBackwardAttributes>();

    // Graph features the engine does not currently dispatch.
    HIP_KERNEL_RETURN_FALSE_IF(attrs.dropout_probability().has_value()
                                   && attrs.dropout_probability().value() != 0.f,
                               "dropout_probability must be unset or zero (Actual value: "
                                   + std::to_string(attrs.dropout_probability().value()) + ")");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.alibi_mask(), "alibi_mask must be false");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.padding_mask(), "padding_mask must be false");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.attn_mask_tensor_uid(), "attn_mask tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.seed_tensor_uid(), "seed tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.offset_tensor_uid(), "offset tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.dropout_mask_tensor_uid(),
                               "dropout_mask tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.dbias_tensor_uid(), "dbias tensor not supported");

    // --- Validate required tensors ---

    // Group mode (variable sequence lengths) requires a different kernarg
    // layout than the POC; deferred to the kernarg-layout abstraction.
    HIP_KERNEL_RETURN_FALSE_IF(
        getBatchMode(attrs) != BatchMode::BATCH,
        "group mode (seq_len_q_tensor_uid or seq_len_kv_tensor_uid set) is not supported");

    const auto& tensorMap = opGraph.getTensorMap();

    // Required input tensor UIDs
    const int64_t qUid = attrs.q_tensor_uid();
    const int64_t kUid = attrs.k_tensor_uid();
    const int64_t vUid = attrs.v_tensor_uid();
    const int64_t oUid = attrs.o_tensor_uid();
    const int64_t doUid = attrs.do_tensor_uid();
    const int64_t statsUid = attrs.stats_tensor_uid();

    // Required output tensor UIDs
    const int64_t dqUid = attrs.dq_tensor_uid();
    const int64_t dkUid = attrs.dk_tensor_uid();
    const int64_t dvUid = attrs.dv_tensor_uid();

    auto findTensor = [&](const char* name, int64_t uid) -> const TensorAttributes* {
        auto it = tensorMap.find(uid);
        if(it == tensorMap.end())
        {
            HIPDNN_PLUGIN_LOG_INFO(std::string{HIP_KERNEL_LOG_PREFIX} + name + " tensor UID "
                                   + std::to_string(uid) + " not present in graph");
            return nullptr;
        }
        return it->second;
    };

    const auto* qTensor = findTensor("q", qUid);
    const auto* kTensor = findTensor("k", kUid);
    const auto* vTensor = findTensor("v", vUid);
    const auto* oTensor = findTensor("o", oUid);
    const auto* doTensor = findTensor("do", doUid);
    const auto* statsTensor = findTensor("stats", statsUid);
    const auto* dqTensor = findTensor("dq", dqUid);
    const auto* dkTensor = findTensor("dk", dkUid);
    const auto* dvTensor = findTensor("dv", dvUid);
    if(qTensor == nullptr || kTensor == nullptr || vTensor == nullptr || oTensor == nullptr
       || doTensor == nullptr || statsTensor == nullptr || dqTensor == nullptr
       || dkTensor == nullptr || dvTensor == nullptr)
    {
        return false;
    }

    HIP_KERNEL_RETURN_FALSE_IF(
        qTensor->dims()->size() != 4,
        "q tensor must be rank 4 (Actual rank: " + std::to_string(qTensor->dims()->size()) + ")");
    HIP_KERNEL_RETURN_FALSE_IF(
        kTensor->dims()->size() != 4,
        "k tensor must be rank 4 (Actual rank: " + std::to_string(kTensor->dims()->size()) + ")");
    HIP_KERNEL_RETURN_FALSE_IF(
        vTensor->dims()->size() != 4,
        "v tensor must be rank 4 (Actual rank: " + std::to_string(vTensor->dims()->size()) + ")");

    // GQA: SdpaBwdPlan packs ratio = nhead_q / nhead_k (integer division) into
    // the dqdkdv kernarg.  A fractional ratio is a kernel-correctness violation
    // (silent truncation), not a "no row matches" registry miss, so reject it
    // here rather than letting buildPlan succeed and execute corrupt dQ/dK/dV.
    auto numHeadsQ = qTensor->dims()->Get(1);
    auto numHeadsKv = kTensor->dims()->Get(1);
    HIP_KERNEL_RETURN_FALSE_IF(numHeadsKv == 0 || numHeadsQ % numHeadsKv != 0,
                               "GQA requires nhead_q % nhead_k == 0 (Actual: nhead_q="
                                   + std::to_string(numHeadsQ)
                                   + ", nhead_k=" + std::to_string(numHeadsKv) + ")");

    // Stats is FP32 (LSE from forward pass)
    HIP_KERNEL_RETURN_FALSE_IF(statsTensor->data_type() != DataType::FLOAT,
                               "stats tensor datatype must be FP32 (Actual type: "
                                   + EnumNameDataType(statsTensor->data_type()) + ")");

    auto dataTypeIdOpt = tryGetDataTypeIdentifier(qTensor->data_type(),
                                                  kTensor->data_type(),
                                                  vTensor->data_type(),
                                                  doTensor->data_type(),
                                                  dqTensor->data_type(),
                                                  dkTensor->data_type(),
                                                  dvTensor->data_type());

    HIP_KERNEL_RETURN_FALSE_IF(
        !dataTypeIdOpt,
        "All Q/K/V/dO/dQ/dK/dV tensors must share a supported dtype (BF16 or FP16). "
        "Actual: q="
            + std::string(EnumNameDataType(qTensor->data_type()))
            + ", k=" + EnumNameDataType(kTensor->data_type())
            + ", v=" + EnumNameDataType(vTensor->data_type())
            + ", do=" + EnumNameDataType(doTensor->data_type())
            + ", dq=" + EnumNameDataType(dqTensor->data_type())
            + ", dk=" + EnumNameDataType(dkTensor->data_type())
            + ", dv=" + EnumNameDataType(dvTensor->data_type()));
    const auto& dataTypeId = *dataTypeIdOpt;

    auto headDimQk = static_cast<int>(qTensor->dims()->Get(3));
    auto headDimV = static_cast<int>(vTensor->dims()->Get(3));

    HIP_KERNEL_RETURN_FALSE_IF(headDimQk != headDimV,
                               "Asymmetric head dimensions not supported (D_qk = "
                                   + std::to_string(headDimQk)
                                   + ", D_v = " + std::to_string(headDimV) + ")");

    // Registry-supported (dtype, hdim) combinations are dispatched; the one-time
    // log in buildPlan warns when the kernel is not yet CPU-reference validated.
    // Head dims outside the registry's {64, 128, 192} set
    // have no row at all and are rejected here so unsupported geometries still
    // fail fast; the per-stage checkRegistry lookups below then reject the
    // (dtype, hdim) combinations that have no matching CSV row (e.g. hd64, whose
    // registry carries only pddv=0 rows that the day-one dispatch tuple does not
    // request).
    HIP_KERNEL_RETURN_FALSE_IF(headDimQk != 64 && headDimQk != 128 && headDimQk != 192,
                               "Head dimension must be one of {64, 128, 192} (Actual value: "
                                   + std::to_string(headDimQk) + ")");

    // Classify the mask; contradictory mask attributes are an invalid-input
    // condition the engine declines rather than dispatches.
    MaskType maskType = MaskType::NO_MASK;
    try
    {
        maskType = plan_utils::getMaskType(attrs);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(std::string{HIP_KERNEL_LOG_PREFIX} + e.what());
        return false;
    }
    HIP_KERNEL_RETURN_FALSE_IF(maskType != MaskType::NO_MASK,
                               "Masked attention not currently dispatched (Mask type ordinal: "
                                   + std::to_string(static_cast<int>(maskType)) + ")");

    const int bf16CvtValue = (dataTypeId == "fp16") ? BF16_CVT_FP16_SENTINEL
                                                    : static_cast<int>(getRoundingMode(attrs));
    // Applicability validates the default (A32) accumulator path: it is the
    // default knob value and the most demanding (dq_acc + DQ_CONVERT). The user
    // may still select A16 via the knob at buildPlan time.
    auto dispatchTuples = computeDispatchTuples(
        dataTypeId, headDimQk, maskType, bf16CvtValue, AccumulatorMode::A32);

    auto checkRegistry = [&](const char* registryName,
                             bwd_dispatch::PipelineStage stage,
                             const BwdDispatchTuple& tuple) {
        auto key = bwd_dispatch::lookupKernelNameKey(stage,
                                                     deviceString,
                                                     dataTypeId,
                                                     headDimQk,
                                                     headDimV,
                                                     tuple.mask,
                                                     tuple.atomic32,
                                                     tuple.pssk,
                                                     tuple.pddv,
                                                     static_cast<int>(BatchMode::BATCH),
                                                     tuple.bf16Cvt);
        if(key.empty())
        {
            HIPDNN_PLUGIN_LOG_INFO(
                std::string{HIP_KERNEL_LOG_PREFIX} + "No matching " + registryName
                + " kernel for arch=" + deviceString + " dtype=" + dataTypeId
                + " hdim=" + std::to_string(headDimQk) + " mask=" + std::to_string(tuple.mask)
                + " atomic32=" + std::to_string(tuple.atomic32)
                + " pssk=" + std::to_string(tuple.pssk) + " pddv=" + std::to_string(tuple.pddv)
                + " mode=batch bf16_cvt=" + std::to_string(tuple.bf16Cvt));
            return false;
        }
        return true;
    };

    HIP_KERNEL_RETURN_FALSE_IF(
        !checkRegistry("odo", bwd_dispatch::PipelineStage::ODO, dispatchTuples.odo),
        "Failed odo registry lookup");
    HIP_KERNEL_RETURN_FALSE_IF(
        !checkRegistry("dqdkdv", bwd_dispatch::PipelineStage::DQDKDV, dispatchTuples.dqdkdv),
        "Failed dqdkdv registry lookup");
    HIP_KERNEL_RETURN_FALSE_IF(!checkRegistry("dq_convert",
                                              bwd_dispatch::PipelineStage::DQ_CONVERT,
                                              dispatchTuples.dqConvert),
                               "Failed dq_convert registry lookup");

    // Reject oversized graphs whose byte strides would silently truncate when
    // packed into the kernarg uint32_t fields. Caught here rather than at
    // execute time so the engine can be excluded from heuristics for graphs it
    // cannot represent. Uses the resolved DQDKDV row's tile size for the Ts
    // product (tsKv * stride_k * BF16).
    const auto& dqdkdvTuple = dispatchTuples.dqdkdv;
    auto dqdkdvCfgOpt = findConfig(cfg_fmha_bwd_dqdkdv,
                                   deviceString,
                                   dataTypeId,
                                   headDimQk,
                                   headDimV,
                                   dqdkdvTuple.mask,
                                   dqdkdvTuple.atomic32,
                                   dqdkdvTuple.pssk,
                                   dqdkdvTuple.pddv,
                                   static_cast<int>(BatchMode::BATCH),
                                   dqdkdvTuple.bf16Cvt);
    HIP_KERNEL_RETURN_FALSE_IF(!dqdkdvCfgOpt,
                               "Failed to resolve dqdkdv config for byte-stride validation");
    const bool useA32 = (dqdkdvTuple.atomic32 == static_cast<int>(AccumulatorMode::A32));
    const int64_t seqLenQ = qTensor->dims()->Get(2);
    HIP_KERNEL_RETURN_FALSE_IF(!wouldBwdByteStridesFitUint32(*qTensor,
                                                             *kTensor,
                                                             *vTensor,
                                                             *oTensor,
                                                             *doTensor,
                                                             *dqTensor,
                                                             *dkTensor,
                                                             *dvTensor,
                                                             *statsTensor,
                                                             numHeadsQ,
                                                             seqLenQ,
                                                             headDimQk,
                                                             dqdkdvCfgOpt->ts,
                                                             useA32),
                               "Backward byte strides overflow uint32_t kernarg fields");

    return true;
}

size_t SdpaBwdPlanBuilder::getMaxWorkspaceSize(
    const Handle& /* handle */,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const Settings& executionSettings) const
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

    const AccumulatorType accType
        = executionSettings.accumulatorType.value_or(AccumulatorType::A32);

    return sdpaBwdWorkspaceSize(batch, headsQ, seqLenQ, headDim, accType);
}

void SdpaBwdPlanBuilder::initializeExecutionSettings(
    const Handle& /* handle */,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /* opGraph */,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    Settings& executionSettings) const
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    if(!engineConfig.isValid() || !engineConfig.hasKnobSetting(K_ACC_TYPE_KNOB_NAME))
    {
        return; // No user preference — default (nullopt → A32) applies
    }

    const auto& knobSetting = engineConfig.getKnobSettingByName(K_ACC_TYPE_KNOB_NAME);

    if(knobSetting.valueType() != KnobValue::StringValue)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "accumulator_type knob value must be a string");
    }

    const auto& value = knobSetting.valueAs<StringValue>().value()->str();

    if(value == "a32")
    {
        executionSettings.accumulatorType = AccumulatorType::A32;
    }
    else if(value == "a16")
    {
        executionSettings.accumulatorType = AccumulatorType::A16;
    }
    else
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                       "Invalid accumulator_type: '" + value
                                                           + "'. Must be 'a32' or 'a16'");
    }
}

void SdpaBwdPlanBuilder::buildPlan(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /* engineConfig */,
    Context& executionContext) const
{
    // -------------------------------------------------------------------------
    // 0. Resolve accumulator type from the user knob / execution settings.
    // This is the single source of truth for a32 (3-kernel) vs a16 (2-kernel)
    // selection; it drives CSV row selection, DQ_CONVERT loading, the dq_acc
    // workspace, and params below. Defaults to A32 when the knob is unset.
    // -------------------------------------------------------------------------
    const AccumulatorType accType
        = executionContext.executionSettings().accumulatorType.value_or(AccumulatorType::A32);

    auto deviceStringOpt
        = tryGetDeviceString(handle.getStream(), "Failed to query device properties with error: ");
    if(!deviceStringOpt)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaBwdPlanBuilder::buildPlan: failed to query device string");
    }
    const std::string& deviceString = *deviceStringOpt;

    // -------------------------------------------------------------------------
    // 1. Extract SDPA backward attributes and tensor metadata from graph
    // -------------------------------------------------------------------------
    auto& sdpaNode = opGraph.getNodeWrapper(0);
    auto& sdpaAttrs
        = sdpaNode.attributesAs<hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes>();
    auto& tensorMap = opGraph.getTensorMap();

    // Tensor UIDs
    const int64_t qUid = sdpaAttrs.q_tensor_uid();
    const int64_t kUid = sdpaAttrs.k_tensor_uid();
    const int64_t vUid = sdpaAttrs.v_tensor_uid();
    const int64_t oUid = sdpaAttrs.o_tensor_uid();
    const int64_t doUid = sdpaAttrs.do_tensor_uid();
    const int64_t statsUid = sdpaAttrs.stats_tensor_uid();
    const int64_t dqUid = sdpaAttrs.dq_tensor_uid();
    const int64_t dkUid = sdpaAttrs.dk_tensor_uid();
    const int64_t dvUid = sdpaAttrs.dv_tensor_uid();

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
    // 2. Extract strides (in elements) from tensor metadata
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
    // 3. Attention scale
    // -------------------------------------------------------------------------
    // Default to 1/sqrt(D_qk) if not provided
    float attnScale = 1.0f / std::sqrt(static_cast<float>(headDimQk));
    auto scaleValue = sdpaAttrs.attn_scale_value();
    if(scaleValue.has_value())
    {
        attnScale = scaleValue.value();
    }

    // -------------------------------------------------------------------------
    // 4. Resolve dispatch parameters and select kernels per stage
    // -------------------------------------------------------------------------
    // Resolve dispatch parameters from the graph. isApplicable already verified
    // the dtype is supported; the empty check here is a defensive guard that
    // mirrors the resolveStage pattern below.
    auto dataTypeIdOpt = tryGetDataTypeIdentifier(qTensor->data_type(),
                                                  kTensor->data_type(),
                                                  vTensor->data_type(),
                                                  doTensor->data_type(),
                                                  dqTensor->data_type(),
                                                  dkTensor->data_type(),
                                                  dvTensor->data_type());
    if(!dataTypeIdOpt)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaBwdPlanBuilder::buildPlan: unsupported tensor dtype combination "
            "(isApplicable should have rejected)");
    }
    const auto& dataTypeId = *dataTypeIdOpt;
    auto maskType = plan_utils::getMaskType(sdpaAttrs);
    auto batchMode = getBatchMode(sdpaAttrs);
    const int bf16CvtValue = (dataTypeId == "fp16") ? BF16_CVT_FP16_SENTINEL
                                                    : static_cast<int>(getRoundingMode(sdpaAttrs));
    // The resolved accumulator type (from the user knob, defaulting to A32) is
    // authoritative: it selects the a32 vs a16 dqdkdv CSV row and gates whether
    // the DQ_CONVERT stage is resolved and loaded below.
    auto dispatchTuples = computeDispatchTuples(dataTypeId,
                                                static_cast<int>(headDimQk),
                                                maskType,
                                                bf16CvtValue,
                                                toAccumulatorMode(accType));

    // Surface, exactly once per (dtype, hdim, mask) over the program lifetime,
    // whether the kernel about to be dispatched has a calibrated CPU reference.
    // The verified flag is keyed on (dtype, hdim) so any stage is representative;
    // the dqdkdv stage is used here. The lambda is noexcept because it only logs
    // (a throwing call_once initializer would let the exception escape, which
    // bugprone-exception-escape forbids).
    std::call_once(
        dispatchLogFlags[dispatchLogIndex(dataTypeId, static_cast<int>(headDimQk), maskType)],
        [&]() noexcept {
            const int maskOrdinal = static_cast<int>(maskType);
            if(dispatchTuples.dqdkdv.verified)
            {
                HIPDNN_PLUGIN_LOG_INFO("SDPA bwd dispatch: verified kernel dtype="
                                       << dataTypeId << " hd=" << headDimQk
                                       << " mask=" << maskOrdinal);
            }
            else
            {
                HIPDNN_PLUGIN_LOG_WARN("SDPA bwd dispatch: UNVERIFIED kernel dtype="
                                       << dataTypeId << " hd=" << headDimQk
                                       << " mask=" << maskOrdinal
                                       << " - results not validated against CPU reference");
            }
        });

    auto resolveStage
        = [&](const char* stageName, std::optional<fmha_v3_bwdConfig> cfgOpt) -> ResolvedKernel {
        if(!cfgOpt)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                std::string("SdpaBwdPlanBuilder::buildPlan: failed to resolve ") + stageName
                    + " kernel for arch=" + deviceString + " dtype=" + dataTypeId + " hdim="
                    + std::to_string(headDimQk) + " (isApplicable should have rejected)");
        }
        return ResolvedKernel{getKernelCoPath(cfgOpt->co_name),
                              cfgOpt->knl_name,
                              SdpaBwdParams::KernelTiles{static_cast<unsigned int>(cfgOpt->ts)}};
    };

    const auto& odtuple = dispatchTuples.odo;
    const auto& dqdtuple = dispatchTuples.dqdkdv;
    const auto& dqctuple = dispatchTuples.dqConvert;

    auto odoResolved = resolveStage("odo",
                                    findConfig(cfg_fmha_bwd_odo,
                                               deviceString,
                                               dataTypeId,
                                               static_cast<int>(headDimQk),
                                               static_cast<int>(headDimV),
                                               odtuple.mask,
                                               odtuple.atomic32,
                                               odtuple.pssk,
                                               odtuple.pddv,
                                               static_cast<int>(batchMode),
                                               odtuple.bf16Cvt));
    auto dqdkdvResolved = resolveStage("dqdkdv",
                                       findConfig(cfg_fmha_bwd_dqdkdv,
                                                  deviceString,
                                                  dataTypeId,
                                                  static_cast<int>(headDimQk),
                                                  static_cast<int>(headDimV),
                                                  dqdtuple.mask,
                                                  dqdtuple.atomic32,
                                                  dqdtuple.pssk,
                                                  dqdtuple.pddv,
                                                  static_cast<int>(batchMode),
                                                  dqdtuple.bf16Cvt));

    // The DQ_CONVERT stage (FP32 dq_acc → BF16 cast) is only part of the A32
    // 3-kernel path. For A16 the DQDKDV kernel writes dQ directly in BF16, so
    // dq_convert is not resolved, the dq_acc workspace is skipped, and the plan
    // is built with a nullopt post-kernel. The resolved tuple's atomic32 column
    // already mirrors accType (see computeDispatchTuples above).
    const bool useA32 = (accType == AccumulatorType::A32);

    std::optional<ResolvedKernel> dqConvertResolved;
    if(useA32)
    {
        dqConvertResolved = resolveStage("dq_convert",
                                         findConfig(cfg_fmha_bwd_dq_convert,
                                                    deviceString,
                                                    dataTypeId,
                                                    static_cast<int>(headDimQk),
                                                    static_cast<int>(headDimV),
                                                    dqctuple.mask,
                                                    dqctuple.atomic32,
                                                    dqctuple.pssk,
                                                    dqctuple.pddv,
                                                    static_cast<int>(batchMode),
                                                    dqctuple.bf16Cvt));
    }

    HIPDNN_PLUGIN_LOG_INFO("Using bwd odo kernel: " << odoResolved.coPath
                                                    << " :: " << odoResolved.knlName);
    HIPDNN_PLUGIN_LOG_INFO("Using bwd dqdkdv kernel: " << dqdkdvResolved.coPath
                                                       << " :: " << dqdkdvResolved.knlName);
    if(dqConvertResolved)
    {
        HIPDNN_PLUGIN_LOG_INFO("Using bwd dq_convert kernel: " << dqConvertResolved->coPath
                                                               << " :: "
                                                               << dqConvertResolved->knlName);
    }

    // -------------------------------------------------------------------------
    // 5. Load kernel modules for resolved stages
    // -------------------------------------------------------------------------
    auto odoKernel = loadKernelModule(odoResolved.coPath, odoResolved.knlName.c_str());
    if(!odoKernel)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaBwdPlanBuilder::buildPlan: failed to load odo kernel module from "
                + odoResolved.coPath);
    }

    auto dqdkdvKernel = loadKernelModule(dqdkdvResolved.coPath, dqdkdvResolved.knlName.c_str());
    if(!dqdkdvKernel)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaBwdPlanBuilder::buildPlan: failed to load dqdkdv kernel module from "
                + dqdkdvResolved.coPath);
    }

    std::optional<HipModuleGuard> postKernel;
    if(dqConvertResolved)
    {
        postKernel
            = loadKernelModule(dqConvertResolved->coPath, dqConvertResolved->knlName.c_str());
        if(!postKernel)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "SdpaBwdPlanBuilder::buildPlan: failed to load dq_convert kernel module from "
                    + dqConvertResolved->coPath);
        }
    }

    // -------------------------------------------------------------------------
    // 6. Build params and create plan
    // -------------------------------------------------------------------------
    SdpaBwdParams params{};
    params.odoTiles = odoResolved.tiles;
    params.dqdkdvTiles = dqdkdvResolved.tiles;
    if(dqConvertResolved)
    {
        params.dqConvertTiles = dqConvertResolved->tiles;
    }

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
    params.accumulatorType = accType;

    // postKernel is nullopt for the A16 path; the optional-taking ctor handles
    // both paths uniformly.
    executionContext.setPlan(std::make_unique<SdpaBwdPlan>(
        std::move(*odoKernel), std::move(*dqdkdvKernel), std::move(postKernel), params));
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> SdpaBwdPlanBuilder::getCustomKnobs(
    const Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    if(!isApplicable(handle, opGraph))
    {
        return {};
    }

    std::vector<KnobT> knobs;

    KnobT accKnob;
    accKnob.knob_id = K_ACC_TYPE_KNOB_NAME;
    accKnob.description
        = "Accumulator precision for backward dQ gradient: a32 (FP32, 3-kernel) or a16 (BF16, "
          "2-kernel)";

    StringValueT defaultValue;
    defaultValue.value = "a32";
    accKnob.default_value.Set(defaultValue);

    StringConstraintT constraint;
    constraint.valid_values.emplace_back("a32");
    constraint.valid_values.emplace_back("a16");
    accKnob.constraint.Set(constraint);

    knobs.push_back(std::move(accKnob));
    return knobs;
}

} // namespace asm_sdpa_engine
