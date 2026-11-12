// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaBwdPlanBuilder.hpp"
#include "asm/AsmKernelPath.hpp"
#include "asm_fmha_v3_bwd_configs.hpp"
#include "core/Utils.hpp"
#include "plans/SdpaBwdPlan.hpp"
#include "plans/SdpaPlanUtils.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hip_kernel_provider_common/SdpaConfigEnumerations.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/ApplicabilityUtils.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
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
using hip_kernel_provider::core::utils::Rejection;
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

// Output of resolveStage: the TOC key (for kpack archive lookup), kernel
// symbol name, and tile sizes for the resolved registry row.
struct ResolvedKernel
{
    std::string tocKey;
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

// Backward kernels live in a flat layout under
//   asm_kernels/<arch>/fmha_v3_bwd/<co_name>
// The codegen-emitted co_name already includes the "<arch>/fmha_v3_bwd/"
// prefix, so this helper strips the arch prefix to produce the TOC key
// matching the .kpack archive index.
// (Forward splits gfx942 into MI300/MI308 sub-folders and threads the arch
// through; backward does not because AITER ships a single backward set.)
std::string getKernelTocKey(const std::string& coName)
{
    return asm_kernels::getAsmKernelTocKey(coName);
}

constexpr int64_t K_BF16_BYTES = 2;
constexpr int64_t K_FP32_BYTES = 4;

// Validate that every backward dimension fits the 32-bit field that carries it.
// Batch reaches the kernels only as gridDimZ and the head counts land in uint32
// kernarg fields. The head dims are narrowed to int for the registry lookup,
// and the sequence lengths become int32 mask coordinates on the sliding-window
// path (computeMaskCoordinates) besides driving `(extent + ts - 1) / ts` grid
// math evaluated in unsigned int, so both of those take the signed limit.
// Checked before the first narrowing cast so the engine declines instead of
// dispatching a kernel on a truncated shape.
bool wouldBwdDimsFitKernargFields(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& q,
                                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& k,
                                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& v)
{
    bool ok = true;
    ok &= plan_utils::dimFitsU32("batch", q.dims()->Get(0));
    ok &= plan_utils::dimFitsU32("nhead_q", q.dims()->Get(1));
    ok &= plan_utils::dimFitsU32("nhead_k", k.dims()->Get(1));
    ok &= plan_utils::dimFitsI32("seqlen_q", q.dims()->Get(2));
    ok &= plan_utils::dimFitsI32("seqlen_k", k.dims()->Get(2));
    ok &= plan_utils::dimFitsI32("head_dim_q", q.dims()->Get(3));
    ok &= plan_utils::dimFitsI32("head_dim_v", v.dims()->Get(3));
    return ok;
}

// Validates that every byte stride consumed by the backward kernels fits in
// the uint32_t fields of the kernarg structs.  Catches overflow at
// applicability time so the engine can be excluded from heuristics for graphs
// it cannot represent, instead of failing at execute time.  Returns false (and
// logs the offending field) on any overflow; returns true when every stride is
// safe.
bool wouldBwdByteStridesFitUint32(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& q,
                                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& k,
                                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& v,
                                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& o,
                                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dO,
                                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dq,
                                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dk,
                                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dv,
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

    // LSE/D buffer: packed [B, H_q, S_q] in FP32 (isApplicable declines any other
    // stats layout); only the batch and head strides are kernel arguments.
    ok &= check("batch_stride_lsed", headsQ * seqLenQ, K_FP32_BYTES);
    ok &= check("nhead_stride_lsed", seqLenQ, K_FP32_BYTES);

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

    // DQDKDV's `Ts` kernarg is the 3-way product (tsKv * stride_k * BF16). The
    // tile size folds into the element size so the product is never formed
    // outside the predicate.
    ok &= check("Ts (tsKv * stride_k)", k.strides()->Get(2), K_BF16_BYTES * tsKv);

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

// Looks up a required backward operand; a missing UID declines the graph.
Rejection findBwdTensor(const plan_utils::TensorMap& tensorMap,
                        const char* name,
                        int64_t uid,
                        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*& tensor)
{
    const auto it = tensorMap.find(uid);
    HIP_KERNEL_REJECT_IF(it == tensorMap.end(),
                         std::string(name) + " tensor UID " + std::to_string(uid)
                             + " not present in graph");
    tensor = it->second;
    return std::nullopt;
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

static Rejection
    checkBwdApplicability(const Handle& handle,
                          const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    HIP_KERNEL_RETURN_IF_REJECTED(
        plan_utils::checkSdpaGraphBasics(opGraph, NodeAttributes::SdpaBackwardAttributes));

    const auto deviceStringOpt = plan_utils::tryGetDeviceString(
        handle.getStream(), "[SdpaBwdPlanBuilder::isApplicable] Could not query device string: ");
    HIP_KERNEL_REJECT_IF(!deviceStringOpt, "Could not query device string");
    const std::string& deviceString = *deviceStringOpt;

    // The codegen-generated registry contains both gfx942 and gfx950 rows;
    // only gfx942 is dispatched here.
    HIP_KERNEL_REJECT_IF(deviceString != "gfx942",
                         "Device string does not match gfx942 (Actual value: " + deviceString
                             + ")");

    const auto& attrs = opGraph.nodeWrappers().front()->attributesAs<SdpaBackwardAttributes>();
    const auto& tensorMap = opGraph.getTensorMap();

    // Graph features the engine does not currently dispatch.
    HIP_KERNEL_RETURN_IF_REJECTED(plan_utils::checkCommonUnsupportedFeatures(attrs, tensorMap));
    HIP_KERNEL_REJECT_IF(attrs.seed_tensor_uid(), "seed tensor not supported");
    HIP_KERNEL_REJECT_IF(attrs.offset_tensor_uid(), "offset tensor not supported");
    HIP_KERNEL_REJECT_IF(attrs.dropout_mask_tensor_uid(), "dropout_mask tensor not supported");
    HIP_KERNEL_REJECT_IF(attrs.dbias_tensor_uid(), "dbias tensor not supported");

    HIP_KERNEL_RETURN_IF_REJECTED(plan_utils::checkAttnScale(attrs, tensorMap));

    // --- Validate required tensors ---

    // Group mode (variable sequence lengths) requires a different kernarg
    // layout than the POC; deferred to the kernarg-layout abstraction.
    HIP_KERNEL_REJECT_IF(
        getBatchMode(attrs) != BatchMode::BATCH,
        "group mode (seq_len_q_tensor_uid or seq_len_kv_tensor_uid set) is not supported");

    HIP_KERNEL_REJECT_IF(!hipdnn_flatbuffers_sdk::utilities::hasNoRaggedTensorIds(tensorMap),
                         "Ragged tensors are not supported");

    // Required inputs (q, k, v, o, do, stats) and outputs (dq, dk, dv).
    const TensorAttributes* qTensor = nullptr;
    const TensorAttributes* kTensor = nullptr;
    const TensorAttributes* vTensor = nullptr;
    const TensorAttributes* oTensor = nullptr;
    const TensorAttributes* doTensor = nullptr;
    const TensorAttributes* statsTensor = nullptr;
    const TensorAttributes* dqTensor = nullptr;
    const TensorAttributes* dkTensor = nullptr;
    const TensorAttributes* dvTensor = nullptr;
    HIP_KERNEL_RETURN_IF_REJECTED(findBwdTensor(tensorMap, "q", attrs.q_tensor_uid(), qTensor));
    HIP_KERNEL_RETURN_IF_REJECTED(findBwdTensor(tensorMap, "k", attrs.k_tensor_uid(), kTensor));
    HIP_KERNEL_RETURN_IF_REJECTED(findBwdTensor(tensorMap, "v", attrs.v_tensor_uid(), vTensor));
    HIP_KERNEL_RETURN_IF_REJECTED(findBwdTensor(tensorMap, "o", attrs.o_tensor_uid(), oTensor));
    HIP_KERNEL_RETURN_IF_REJECTED(findBwdTensor(tensorMap, "do", attrs.do_tensor_uid(), doTensor));
    HIP_KERNEL_RETURN_IF_REJECTED(
        findBwdTensor(tensorMap, "stats", attrs.stats_tensor_uid(), statsTensor));
    HIP_KERNEL_RETURN_IF_REJECTED(findBwdTensor(tensorMap, "dq", attrs.dq_tensor_uid(), dqTensor));
    HIP_KERNEL_RETURN_IF_REJECTED(findBwdTensor(tensorMap, "dk", attrs.dk_tensor_uid(), dkTensor));
    HIP_KERNEL_RETURN_IF_REJECTED(findBwdTensor(tensorMap, "dv", attrs.dv_tensor_uid(), dvTensor));

    // Virtual tensors have no device buffer, and execute needs a device pointer for every
    // operand. Q/K/V get the same check from checkRank4DeviceTensor below.
    for(const auto& [label, tensor] : {std::pair{"o", oTensor},
                                       std::pair{"do", doTensor},
                                       std::pair{"stats", statsTensor},
                                       std::pair{"dq", dqTensor},
                                       std::pair{"dk", dkTensor},
                                       std::pair{"dv", dvTensor}})
    {
        HIP_KERNEL_REJECT_IF(tensor->virtual_(),
                             std::string(label) + " tensor must not be virtual");
    }
    for(const auto& [label, tensor] :
        {std::pair{"q", qTensor}, std::pair{"k", kTensor}, std::pair{"v", vTensor}})
    {
        HIP_KERNEL_RETURN_IF_REJECTED(plan_utils::checkRank4DeviceTensor(label, *tensor));
    }

    // Every dimension below is narrowed to 32 bits before it reaches a kernel,
    // starting with the head-dim cast and the mask classification further down,
    // so the range gate comes first.
    HIP_KERNEL_REJECT_IF(!wouldBwdDimsFitKernargFields(*qTensor, *kTensor, *vTensor),
                         "Tensor dimensions overflow the 32-bit kernarg fields that carry them");

    auto toString = [](const flatbuffers::Vector<int64_t>* values) {
        return hipdnn_data_sdk::utilities::vecToString(
            hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(values));
    };

    // Strides the kernels cannot address. The ODO, DQDKDV and DQ_CONVERT argument
    // blocks carry only batch, head and sequence byte strides, and each row is
    // read or written as D contiguous elements. On gfx942 every layout declined
    // below gave silent wrong gradients:
    // - Head-dim (innermost) stride other than 1: it is treated as 1 (a stride-2
    //   Q gave Inf dK/dV; a stride-2 dQ left dQ elements unwritten).
    // - Sequence stride below D, even on a length-1 sequence axis: each buffer
    //   ends S * seq stride elements (S * Seqs bytes) after the start of its
    //   (batch, head) slice, so the end of the last row reads as 0 and writes
    //   there are dropped. At D=192, DQDKDV reaches bytes 128-383 of a Q, K or V
    //   row through the instruction offset, which the record-index bound does not
    //   check, so 64 elements would be enough there; the rule stays at D for all
    //   tensors.
    // - Odd stride on an axis whose offset DQDKDV puts in its 4-byte record index
    //   (byte offset >> 2): the sequence axis of Q, K, V, dO, dK and dV, and the
    //   head axis of dK and dV (no-mask and sliding-window kernels; the causal
    //   kernels add the dK/dV head offset to the buffer base, but this check does
    //   not depend on the mask). A row that starts 2 bytes into a dword is read or
    //   written one element early. DQDKDV adds the other offsets to the buffer
    //   base, and ODO and DQ_CONVERT (A32 path) address O and dQ by byte offset,
    //   so odd strides there are addressed correctly.
    struct IoTensor
    {
        const char* name;
        const TensorAttributes* tensor;
        // Outermost axis whose offset is in the record index (3: none).
        flatbuffers::uoffset_t firstIndexedAxis;
        // Written by the kernels (dQ, dK, dV).
        bool isOutput;
    };
    const std::array<IoTensor, 8> ioTensors{{{"q", qTensor, 2, false},
                                             {"k", kTensor, 2, false},
                                             {"v", vTensor, 2, false},
                                             {"o", oTensor, 3, false},
                                             {"do", doTensor, 2, false},
                                             {"dq", dqTensor, 3, true},
                                             {"dk", dkTensor, 1, true},
                                             {"dv", dvTensor, 1, true}}};
    for(const auto& io : ioTensors)
    {
        const auto* dims = io.tensor->dims();
        const auto* strides = io.tensor->strides();
        const auto describe = [&]() {
            return std::string(io.name) + " tensor (Actual dims: " + toString(dims)
                   + ", strides: " + toString(strides) + ")";
        };
        HIP_KERNEL_REJECT_IF(dims->size() != 4 || strides->size() != 4,
                             "rank-4 dims and strides required for " + describe());
        HIP_KERNEL_REJECT_IF(strides->Get(3) != 1,
                             "head-dim (innermost) stride must be 1 for " + describe());
        HIP_KERNEL_REJECT_IF(strides->Get(2) < dims->Get(3),
                             "sequence stride must be at least the head dim for " + describe());
        bool oddIndexedStride = false;
        for(auto axis = io.firstIndexedAxis; axis < 3; ++axis)
        {
            if(dims->Get(axis) > 1 && strides->Get(axis) % 2 != 0)
            {
                oddIndexedStride = true;
            }
        }
        HIP_KERNEL_REJECT_IF(oddIndexedStride,
                             "4-byte-indexed stride must be even for " + describe());

        // A zero batch or head stride on an output axis longer than 1 makes the
        // (batch, head) slices of dQ, dK or dV share memory. Each slice is written
        // by its own workgroups (grid y and z are head and batch), so the slices
        // overwrite each other: on gfx942 such dQ, dK and dV were wrong and changed
        // between runs with identical inputs. A zero stride on an input (a
        // broadcast) just re-reads one slice and stays accepted.
        // Only this overlap is checked, deliberately: a zero stride is how a
        // broadcast is written, so an output gets one when it is given the strides
        // of a broadcast input. Nonzero strides that make an output overlap itself,
        // such as equal batch and head strides, have to be built by hand; they are
        // not checked here, although on gfx942 they also gave wrong dK and dV.
        const bool zeroOutputStride = io.isOutput
                                      && ((dims->Get(0) > 1 && strides->Get(0) == 0)
                                          || (dims->Get(1) > 1 && strides->Get(1) == 0));
        HIP_KERNEL_REJECT_IF(
            zeroOutputStride,
            "batch and head strides must be nonzero on axes longer than 1 for output "
                + describe());
    }

    // The AITER dqdkdv kernel indexes dK/dV by q-head: workgroup.y spans nhead_q and
    // the kernel divides by ratio only for the K/V *reads*, never for the dK/dV
    // *writes*.  Supporting GQA therefore requires caller-side head expansion -
    // allocate dK/dV with nhead_q head slots, launch, then sum each ratio-group into
    // the user's nhead_k tensors - which is what AITER does in asm_mha_bwd.cu via
    // dk_expanded/dv_expanded plus at::sum_out().
    //
    // SdpaBwdPlan does not implement that protocol; it passes the user's nhead_k
    // buffers straight through.  The kernel then writes
    // (nhead_q - nhead_k) * head_dim * sizeof(bf16) bytes past the end of dK/dV.
    // The buffer-descriptor bounds check cannot catch this: num_records is relative
    // to the already-advanced base pointer, so the overrun faults (observed as
    // HSA_STATUS_ERROR_MEMORY_FAULT on gfx942) instead of clamping.
    //
    // Reject GQA/MQA here rather than letting buildPlan succeed and execute a kernel
    // that corrupts memory.  Restore the weaker "nhead_q % nhead_k == 0" check once
    // the expansion and reduction are implemented.
    auto numHeadsQ = qTensor->dims()->Get(1);
    auto numHeadsKv = kTensor->dims()->Get(1);
    HIP_KERNEL_REJECT_IF(numHeadsKv == 0 || numHeadsQ != numHeadsKv,
                         "GQA/MQA backward requires dK/dV head expansion, which is not "
                         "implemented (Actual: nhead_q="
                             + std::to_string(numHeadsQ) + ", nhead_k=" + std::to_string(numHeadsKv)
                             + ")");

    // Stats is FP32 (LSE from forward pass)
    HIP_KERNEL_REJECT_IF(statsTensor->data_type() != DataType::FLOAT,
                         "stats tensor datatype must be FP32 (Actual type: "
                             + EnumNameDataType(statsTensor->data_type()) + ")");

    // LSE and the D workspace must be packed [B, H_q, S_q] FP32. Every gfx942
    // batch-mode DQDKDV kernel computes the offset of both as
    // ((b * H_q + h) * S_q + s) * 4 from nhead_q and seqlen_q, and never loads its
    // Hs_lsed argument (from the .co disassembly). ODO writes D into a
    // B * H_q * S_q workspace with the same packed strides (buildPlan). On gfx942
    // a stats head stride padded to S_q + 64 gave wrong, run-to-run varying
    // gradients, so decline any other layout. No kernel reads the stats strides,
    // so the stride of a size-1 axis is not checked.
    const int64_t batchSize = qTensor->dims()->Get(0);
    const int64_t seqLenQ = qTensor->dims()->Get(2);
    const auto* statsDims = statsTensor->dims();
    const auto* statsStrides = statsTensor->strides();
    HIP_KERNEL_REJECT_IF(
        (statsDims->size() != 3 && (statsDims->size() != 4 || statsDims->Get(3) != 1))
            || statsStrides->size() != statsDims->size() || statsDims->Get(0) != batchSize
            || statsDims->Get(1) != numHeadsQ || statsDims->Get(2) != seqLenQ,
        "stats tensor must be [B, H_q, S_q] or [B, H_q, S_q, 1] matching Q (Actual dims: "
            + toString(statsDims) + ", strides: " + toString(statsStrides) + ")");
    HIP_KERNEL_REJECT_IF((batchSize != 1 && statsStrides->Get(0) != numHeadsQ * seqLenQ)
                             || (numHeadsQ != 1 && statsStrides->Get(1) != seqLenQ)
                             || (seqLenQ != 1 && statsStrides->Get(2) != 1),
                         "stats tensor strides must be packed [H_q * S_q, S_q, 1] (Actual strides: "
                             + toString(statsStrides) + ")");

    auto dataTypeIdOpt = tryGetDataTypeIdentifier(qTensor->data_type(),
                                                  kTensor->data_type(),
                                                  vTensor->data_type(),
                                                  doTensor->data_type(),
                                                  dqTensor->data_type(),
                                                  dkTensor->data_type(),
                                                  dvTensor->data_type());

    HIP_KERNEL_REJECT_IF(
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

    HIP_KERNEL_REJECT_IF(headDimQk != headDimV,
                         "Asymmetric head dimensions not supported (D_qk = "
                             + std::to_string(headDimQk) + ", D_v = " + std::to_string(headDimV)
                             + ")");

    // Registry-supported (dtype, hdim) combinations are dispatched; the one-time
    // log in buildPlan warns when the kernel is not yet CPU-reference validated.
    // Head dims outside the registry's {64, 128, 192} set
    // have no row at all and are rejected here so unsupported geometries still
    // fail fast; the per-stage checkRegistry lookups below then reject the
    // (dtype, hdim) combinations that have no matching CSV row (e.g. hd64, whose
    // registry carries only pddv=0 rows that the day-one dispatch tuple does not
    // request).
    HIP_KERNEL_REJECT_IF(headDimQk != 64 && headDimQk != 128 && headDimQk != 192,
                         "Head dimension must be one of {64, 128, 192} (Actual value: "
                             + std::to_string(headDimQk) + ")");

    // Each kernel takes B, H, S_q, S_kv and D once, from Q, K and V, and addresses
    // every tensor with them, so O, dO and dQ must have Q's dims and V, dK and dV
    // must have K's. The frontend lets V have fewer heads than K (it checks only
    // that both divide H_q); DQDKDV then reads V and writes dV past their last
    // head. On gfx942, H_q = H_k = 4 with H_v = 2 or 1 gave wrong dQ, dK and dV.
    HIP_KERNEL_REJECT_IF(kTensor->dims()->Get(0) != batchSize,
                         "k tensor batch must equal q batch (Actual dims: k "
                             + toString(kTensor->dims()) + ", q " + toString(qTensor->dims())
                             + ")");
    struct SameDims
    {
        const char* name;
        const TensorAttributes* tensor;
        const char* refName;
        const TensorAttributes* ref;
    };
    const std::array<SameDims, 6> sameDims{{{"o", oTensor, "q", qTensor},
                                            {"do", doTensor, "q", qTensor},
                                            {"dq", dqTensor, "q", qTensor},
                                            {"v", vTensor, "k", kTensor},
                                            {"dk", dkTensor, "k", kTensor},
                                            {"dv", dvTensor, "k", kTensor}}};
    for(const auto& match : sameDims)
    {
        const auto* dims = match.tensor->dims();
        const auto* refDims = match.ref->dims();
        HIP_KERNEL_REJECT_IF(
            !std::equal(dims->begin(), dims->end(), refDims->begin(), refDims->end()),
            std::string(match.name) + " tensor dims must equal " + match.refName
                + " dims (Actual: " + match.name + " " + toString(dims) + ", " + match.refName + " "
                + toString(refDims) + ")");
    }

    // Classify the mask; contradictory mask attributes are an invalid-input
    // condition the engine declines rather than dispatches.
    const int64_t seqLenKv = kTensor->dims()->Get(2);
    MaskType maskType = MaskType::NO_MASK;
    HIP_KERNEL_RETURN_IF_REJECTED(plan_utils::tryClassifyMask(attrs, seqLenQ, seqLenKv, maskType));
    // Mask types NO_MASK (0), TOP_LEFT_CAUSAL (1), BOTTOM_RIGHT_CAUSAL (2),
    // and SLIDING_WINDOW (3) are all dispatched. The DQDKDV CSV registry
    // carries rows for each mask ordinal; ODO and DQ_CONVERT are mask-agnostic
    // (always mask=0). Applicability is gated by the registry lookup below.

    const int bf16CvtValue = (dataTypeId == "fp16") ? BF16_CVT_FP16_SENTINEL
                                                    : static_cast<int>(getRoundingMode(attrs));
    // Applicability validates the default (A32) accumulator path: it is the
    // default knob value and the most demanding (dq_acc + DQ_CONVERT). The user
    // may still select A16 via the knob at buildPlan time.
    auto dispatchTuples = computeDispatchTuples(
        dataTypeId, headDimQk, maskType, bf16CvtValue, AccumulatorMode::A32);

    auto checkRegistry = [&](const char* registryName,
                             bwd_dispatch::PipelineStage stage,
                             const BwdDispatchTuple& tuple) -> Rejection {
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
        HIP_KERNEL_REJECT_IF(
            key.empty(),
            std::string("No matching ") + registryName + " kernel for arch=" + deviceString
                + " dtype=" + dataTypeId + " hdim=" + std::to_string(headDimQk) + " mask="
                + std::to_string(tuple.mask) + " atomic32=" + std::to_string(tuple.atomic32)
                + " pssk=" + std::to_string(tuple.pssk) + " pddv=" + std::to_string(tuple.pddv)
                + " mode=batch bf16_cvt=" + std::to_string(tuple.bf16Cvt));
        return std::nullopt;
    };

    HIP_KERNEL_RETURN_IF_REJECTED(
        checkRegistry("odo", bwd_dispatch::PipelineStage::ODO, dispatchTuples.odo));
    HIP_KERNEL_RETURN_IF_REJECTED(
        checkRegistry("dqdkdv", bwd_dispatch::PipelineStage::DQDKDV, dispatchTuples.dqdkdv));
    HIP_KERNEL_RETURN_IF_REJECTED(checkRegistry(
        "dq_convert", bwd_dispatch::PipelineStage::DQ_CONVERT, dispatchTuples.dqConvert));

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
    HIP_KERNEL_REJECT_IF(!dqdkdvCfgOpt,
                         "Failed to resolve dqdkdv config for byte-stride validation");
    const bool useA32 = (dqdkdvTuple.atomic32 == static_cast<int>(AccumulatorMode::A32));
    HIP_KERNEL_REJECT_IF(!wouldBwdByteStridesFitUint32(*qTensor,
                                                       *kTensor,
                                                       *vTensor,
                                                       *oTensor,
                                                       *doTensor,
                                                       *dqTensor,
                                                       *dkTensor,
                                                       *dvTensor,
                                                       numHeadsQ,
                                                       seqLenQ,
                                                       headDimQk,
                                                       dqdkdvCfgOpt->ts,
                                                       useA32),
                         "Backward byte strides overflow uint32_t kernarg fields");

    return std::nullopt;
}

bool SdpaBwdPlanBuilder::isApplicable(
    const Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    const auto rejection = checkBwdApplicability(handle, opGraph);
    if(rejection.has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO("[SdpaBwdPlanBuilder::isApplicable] " << *rejection);
        return false;
    }
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
    auto headDimQk = static_cast<size_t>(qTensor->dims()->Get(3));

    const AccumulatorType accType
        = executionSettings.accumulatorType.value_or(AccumulatorType::A32);

    return sdpaBwdWorkspaceSize(batch, headsQ, seqLenQ, headDimQk, accType);
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

    auto deviceStringOpt = plan_utils::tryGetDeviceString(
        handle.getStream(), "Failed to query device properties with error: ");
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

    // Stats (LSE) and the D workspace: packed [B, H_q, S_q] FP32. isApplicable
    // declines any stats tensor that is not packed on every axis of size > 1, so
    // the strides follow from Q's dims. (The user's stride on a size-1 axis is
    // arbitrary and must not reach the kernels.)
    const unsigned int statsStrideBatch = numHeadsQ * seqLenQ;
    const unsigned int statsStrideHead = seqLenQ;

    // -------------------------------------------------------------------------
    // 3. Attention scale — resolve as ScalarOperand (RFC 0016)
    // -------------------------------------------------------------------------
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
    // Classify on the graph dims, exactly as isApplicable does. Passing the
    // narrowed unsigned locals instead would let the two sites resolve
    // different mask types for the same graph.
    const auto maskBand = plan_utils::makeCoveringBoundsUnbound(
        plan_utils::resolveDiagonalBand(sdpaAttrs), qDims->Get(2), kTensor->dims()->Get(2));
    const auto maskType
        = plan_utils::getMaskType(maskBand.leftBound, maskBand.rightBound, maskBand.alignment);
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
        return ResolvedKernel{getKernelTocKey(cfgOpt->co_name),
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

    HIPDNN_PLUGIN_LOG_INFO("Using bwd odo kernel: " << odoResolved.tocKey
                                                    << " :: " << odoResolved.knlName);
    HIPDNN_PLUGIN_LOG_INFO("Using bwd dqdkdv kernel: " << dqdkdvResolved.tocKey
                                                       << " :: " << dqdkdvResolved.knlName);
    if(dqConvertResolved)
    {
        HIPDNN_PLUGIN_LOG_INFO("Using bwd dq_convert kernel: " << dqConvertResolved->tocKey
                                                               << " :: "
                                                               << dqConvertResolved->knlName);
    }

    // -------------------------------------------------------------------------
    // 5. Load kernel modules for resolved stages (from kpack archive)
    // -------------------------------------------------------------------------
    auto odoKernel
        = moduleCache().getOrLoad(odoResolved.tocKey, deviceString, odoResolved.knlName.c_str());
    if(!odoKernel)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaBwdPlanBuilder::buildPlan: failed to load odo kernel tocKey='" + odoResolved.tocKey
                + "' arch='" + deviceString + "'");
    }

    auto dqdkdvKernel = moduleCache().getOrLoad(
        dqdkdvResolved.tocKey, deviceString, dqdkdvResolved.knlName.c_str());
    if(!dqdkdvKernel)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaBwdPlanBuilder::buildPlan: failed to load dqdkdv kernel tocKey='"
                + dqdkdvResolved.tocKey + "' arch='" + deviceString + "'");
    }

    std::optional<CachedModule> postKernel;
    if(dqConvertResolved)
    {
        auto loaded = moduleCache().getOrLoad(
            dqConvertResolved->tocKey, deviceString, dqConvertResolved->knlName.c_str());
        if(!loaded)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "SdpaBwdPlanBuilder::buildPlan: failed to load dq_convert kernel tocKey='"
                    + dqConvertResolved->tocKey + "' arch='" + deviceString + "'");
        }
        postKernel = std::move(loaded);
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
    params.maskOrdinal = static_cast<int32_t>(maskType);
    // A surviving bound is strictly below its sequence span (the covering ones
    // became -1), and isApplicable capped both sequence lengths at INT32_MAX, so
    // the narrowing below and the mask coordinates computed from it in
    // SdpaBwdPlan are exact.
    if(maskType == MaskType::SLIDING_WINDOW)
    {
        params.windowLeft = static_cast<int32_t>(maskBand.leftBound);
        params.windowRight = static_cast<int32_t>(maskBand.rightBound);
        params.topLeftAlignment
            = maskBand.alignment
              != hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    }

    if(postKernel)
    {
        executionContext.setPlan(std::make_unique<SdpaBwdPlan>(
            std::move(odoKernel), std::move(dqdkdvKernel), std::move(*postKernel), params));
    }
    else
    {
        executionContext.setPlan(
            std::make_unique<SdpaBwdPlan>(std::move(odoKernel), std::move(dqdkdvKernel), params));
    }
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
