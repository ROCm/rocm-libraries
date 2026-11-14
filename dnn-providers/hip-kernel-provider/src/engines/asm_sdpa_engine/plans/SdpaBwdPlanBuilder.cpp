// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaBwdPlanBuilder.hpp"
#include "asm/AsmKernelPath.hpp"
#include "asm_fmha_v3_bwd_configs.hpp"
#include "core/Utils.hpp"
#include "plans/SdpaBwdPlan.hpp"
#include "plans/SdpaKernelUtils.hpp"
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
#include <tuple>
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
// forward builder consumes). AccumulatorMode is bwd_dispatch's because the
// forward dispatch has no 16/32-bit accumulator axis.
using bwd_dispatch::AccumulatorMode;
using hip_kernel_provider_common::BatchMode;
using hip_kernel_provider_common::RoundingMode;
using plan_utils::MaskType;

using bwd_dispatch::BF16_CVT_FP16_SENTINEL;

// Output of resolveStage: the TOC key (for kpack archive lookup), kernel
// symbol name, and tile sizes for the resolved registry row.
struct ResolvedKernel
{
    std::string tocKey;
    std::string knlName;
    SdpaBwdParams::KernelTiles tiles;
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

// A row stored with bf16_cvt = 3 serves every rounding mode (the fp16 rows, and
// the bf16 rows that ship one kernel for all modes); any other row serves only
// the mode it names. AITER v0.1.29 mha_bwd.cu get_heuristic_kernel() (:81,
// :121): `(cfg.bf16_cvt == 3) || (cfg.bf16_cvt == bf16_cvt)`.
bool servesBf16Cvt(const fmha_v3_bwdConfig& cfg, int bf16Cvt)
{
    return cfg.bf16_cvt == BF16_CVT_FP16_SENTINEL || cfg.bf16_cvt == bf16Cvt;
}

// AITER v0.1.29 mha_bwd.cu get_heuristic_kernel() odo loop (:58-68): the odo
// row is keyed on dtype, padded hdim_v and mode only.
std::optional<fmha_v3_bwdConfig> findOdoConfig(const std::string& archId,
                                               const bwd_dispatch::KernelRequest& request,
                                               const bwd_dispatch::HeadDims& padded)
{
    for(const auto& [unusedKey, cfg] : cfg_fmha_bwd_odo)
    {
        if(cfg.arch == archId && cfg.dtype == request.dataType && cfg.hdim_v == padded.v
           && cfg.mode == request.mode)
        {
            return cfg;
        }
    }
    return std::nullopt;
}

// AITER v0.1.29 mha_bwd.cu get_heuristic_kernel() post loop (:113-133), for
// both post registries (dq_convert and dq_shuffle): the post row is keyed on
// padded hdim_q, mode and bf16_cvt, and on dtype only for the A32 accumulator
// (`(cfg.dtype == data_type) || (atomic32 == 0)`, :123).
std::optional<fmha_v3_bwdConfig> findPostConfig(const CFG& registry,
                                                const std::string& archId,
                                                const bwd_dispatch::KernelRequest& request,
                                                const bwd_dispatch::HeadDims& padded)
{
    for(const auto& [unusedKey, cfg] : registry)
    {
        if(cfg.arch == archId && cfg.hdim_q == padded.q && cfg.mode == request.mode
           && servesBf16Cvt(cfg, request.bf16Cvt)
           && (cfg.dtype == request.dataType || request.atomic32 == 0))
        {
            return cfg;
        }
    }
    return std::nullopt;
}

// The pssk value AITER v0.1.29 mha_bwd.cu get_heuristic_kernel() requests
// (:84-96), from the first dqdkdv row that matches the key: the sequence needs
// no padding only when seqlen_q == seqlen_k and seqlen_k is a multiple of the
// row's K/V tile. A16 rows on gfx942 and *recompile* A16 rows count in 64-row
// tiles whatever their ts. Every row of one match class shares ts and atomic32.
int requestedPssk(const std::string& archId,
                  const fmha_v3_bwdConfig& cfg,
                  int64_t seqLenQ,
                  int64_t seqLenKv)
{
    const bool tiles64
        = cfg.atomic32 == 0
          && (archId == "gfx942" || cfg.knl_name.find("recompile") != std::string::npos);
    const int64_t tsKv = tiles64 ? 64 : cfg.ts;
    return (tsKv <= 0 || seqLenQ != seqLenKv || seqLenKv % tsKv != 0) ? 1 : 0;
}

// AITER v0.1.29 mha_bwd.cu get_heuristic_kernel() dqdkdv loop (:70-112). The
// key is dtype, padded hdim_q and hdim_v, mask, atomic32, bf16_cvt and mode;
// pddv is 1 when padding changed a head dim (:50) and pssk comes from
// requestedPssk(). AITER takes the exact (pssk, pddv) row, or else the last
// row its unordered_map yields with pssk and pddv both at least the requested
// values. Here the smallest qualifying (pssk, pddv) wins, which is the exact
// row when there is one, so the choice does not depend on map order.
//
// Every row shares one kernarg layout (fmha_bwd_dqdkdv_args). What the
// pssk = 0 kernels need beyond the sequence condition is the stride layout
// that AITER's Python gate checks before it sends a shape to them
// (psskOffStridesMatch), and (batch, head) offsets that fit the 32 bits they
// form them in (psskOffOffsetsFitU32), which AITER does not check; without
// both a pssk = 0 row is skipped and the lookup falls through to a pssk = 1
// row.
std::optional<fmha_v3_bwdConfig> findDqdkdvConfig(const std::string& archId,
                                                  const bwd_dispatch::KernelRequest& request,
                                                  const bwd_dispatch::HeadDims& padded)
{
    const int pddv = (padded.q != request.hdimQ || padded.v != request.hdimV) ? 1 : 0;
    const bool psskOffServes
        = request.psskOffStridesOk && bwd_dispatch::psskOffOffsetsFitU32(request);
    std::optional<int> pssk;
    const fmha_v3_bwdConfig* best = nullptr;
    for(const auto& [unusedKey, cfg] : cfg_fmha_bwd_dqdkdv)
    {
        if(cfg.arch != archId || cfg.dtype != request.dataType || cfg.hdim_q != padded.q
           || cfg.hdim_v != padded.v || cfg.mask != request.mask || cfg.atomic32 != request.atomic32
           || !servesBf16Cvt(cfg, request.bf16Cvt) || cfg.mode != request.mode)
        {
            continue;
        }
        if(!pssk)
        {
            pssk = requestedPssk(archId, cfg, request.seqLenQ, request.seqLenKv);
        }
        if(cfg.pssk < *pssk || cfg.pddv < pddv || (cfg.pssk == 0 && !psskOffServes))
        {
            continue;
        }
        if(best == nullptr
           || std::tie(cfg.pssk, cfg.pddv, cfg.knl_name)
                  < std::tie(best->pssk, best->pddv, best->knl_name))
        {
            best = &cfg;
        }
    }
    if(best == nullptr)
    {
        return std::nullopt;
    }
    return *best;
}

// The registry row for `stage` (AITER v0.1.29 get_heuristic_kernel(), plus the
// pssk = 0 guards and the deterministic (pssk, pddv) choice of
// findDqdkdvConfig), or std::nullopt when no row matches. Returns a copy so
// callers do not hold references into the registry.
std::optional<fmha_v3_bwdConfig> resolveStageConfig(bwd_dispatch::PipelineStage stage,
                                                    const std::string& archId,
                                                    const bwd_dispatch::KernelRequest& request)
{
    const auto padded = bwd_dispatch::paddedHeadDims(archId, request.hdimQ, request.hdimV);
    switch(stage)
    {
    case bwd_dispatch::PipelineStage::ODO:
        return findOdoConfig(archId, request, padded);
    case bwd_dispatch::PipelineStage::DQDKDV:
        return findDqdkdvConfig(archId, request, padded);
    case bwd_dispatch::PipelineStage::DQ_CONVERT:
        return findPostConfig(cfg_fmha_bwd_dq_convert, archId, request, padded);
    case bwd_dispatch::PipelineStage::DQ_SHUFFLE:
        return findPostConfig(cfg_fmha_bwd_dq_shuffle, archId, request, padded);
    default:
        return std::nullopt;
    }
}

const char* stageName(bwd_dispatch::PipelineStage stage)
{
    switch(stage)
    {
    case bwd_dispatch::PipelineStage::ODO:
        return "odo";
    case bwd_dispatch::PipelineStage::DQDKDV:
        return "dqdkdv";
    case bwd_dispatch::PipelineStage::DQ_CONVERT:
        return "dq_convert";
    case bwd_dispatch::PipelineStage::DQ_SHUFFLE:
        return "dq_shuffle";
    default:
        return "unknown";
    }
}

// Element sizes of the 16-bit tensors (bf16 and fp16) and of the FP32 ones.
constexpr int64_t K_BF16_BYTES = 2;
constexpr int64_t K_FP32_BYTES = 4;

bwd_dispatch::BhsStrides
    bhsStrides(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensor)
{
    const auto* strides = tensor.strides();
    return {strides->Get(0), strides->Get(1), strides->Get(2)};
}

// bwd_dispatch::bhsdSpanBytes() of a 16-bit [B, H, S, D] tensor.
int64_t spanBytes(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    return bwd_dispatch::bhsdSpanBytes(
        bhsStrides(tensor), dims->Get(0), dims->Get(1), dims->Get(2), dims->Get(3), K_BF16_BYTES);
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
                                  int64_t batch,
                                  int64_t headsQ,
                                  int64_t seqLenQ,
                                  int64_t headDimQk,
                                  int64_t tsKv)
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

    // dq_acc: the A32 layout, FP32 [B, H_q, S_q, D_qk]; buildPlan checks the A16
    // one. The post kernel takes its head and batch strides as 64-bit, but the
    // DQDKDV kernels address dq_acc within a batch with 32-bit byte offsets.
    ok &= sdpaBwdDqAccFitsU32(sdpaBwdDqAccLayout(static_cast<size_t>(batch),
                                                 static_cast<size_t>(headsQ),
                                                 static_cast<size_t>(seqLenQ),
                                                 static_cast<size_t>(headDimQk),
                                                 AccumulatorType::A32),
                              static_cast<size_t>(headsQ));

    // DQDKDV's `Ts` kernarg is the 3-way product (tsKv * stride_k * BF16). The
    // tile size folds into the element size so the product is never formed
    // outside the predicate.
    ok &= check("Ts (tsKv * stride_k)", k.strides()->Get(2), K_BF16_BYTES * tsKv);

    return ok;
}

// Registry rows whose dqdkdv kernel has been calibrated against the CPU
// backward reference, by (arch, dtype, row hdim_q, row hdim_v); see "Verified
// Kernel Matrix" in asm/asm_kernels/README.md. Calibration ran the psskddv
// rows (pssk = 1, pddv = 1) at the row's own head dims, so isVerifiedDispatch()
// accepts only that case.
struct VerifiedKernel
{
    const char* arch;
    const char* dtype;
    int hdimQ;
    int hdimV;
};
constexpr std::array<VerifiedKernel, 2> K_VERIFIED_KERNELS{{
    {"gfx942", "bf16", 128, 128},
    {"gfx942", "fp16", 128, 128},
}};

bool isVerifiedDispatch(const std::string& archId,
                        const bwd_dispatch::KernelRequest& request,
                        const fmha_v3_bwdConfig& dqdkdv)
{
    if(dqdkdv.pssk != 1 || dqdkdv.pddv != 1 || request.hdimQ != dqdkdv.hdim_q
       || request.hdimV != dqdkdv.hdim_v)
    {
        return false;
    }
    return std::any_of(
        K_VERIFIED_KERNELS.begin(), K_VERIFIED_KERNELS.end(), [&](const VerifiedKernel& kernel) {
            return archId == kernel.arch && dqdkdv.dtype == kernel.dtype
                   && dqdkdv.hdim_q == kernel.hdimQ && dqdkdv.hdim_v == kernel.hdimV;
        });
}

// Bridge the public AccumulatorType knob enum to the CSV-local AccumulatorMode
// used for the `atomic32` dispatch column.
AccumulatorMode toAccumulatorMode(AccumulatorType accType)
{
    return (accType == AccumulatorType::A32) ? AccumulatorMode::A32 : AccumulatorMode::A16;
}

// One-time dispatch logging. Each (arch, dtype, row head dims, mask,
// accumulator, verified) combination logs exactly once over the program
// lifetime: an INFO line for a CPU-reference-verified kernel, or a WARN line
// for an unverified one. Static storage duration; lifetime is the program
// duration.
constexpr size_t K_NUM_ARCHS = 2; // gfx942, gfx950
constexpr size_t K_NUM_DTYPES = 2; // bf16, fp16
constexpr size_t K_NUM_HDIMS = 4; // 64x64, 128x128, 192x192, 192x128
constexpr size_t K_NUM_MASKS = 5; // dqdkdv CSV mask ordinals 0-4
constexpr size_t K_NUM_ACCUMULATORS = 2; // atomic32 0 (A16), 1 (A32)
constexpr size_t K_NUM_VERIFIED = 2; // unverified, verified
constexpr size_t K_LOG_FLAGS
    = K_NUM_ARCHS * K_NUM_DTYPES * K_NUM_HDIMS * K_NUM_MASKS * K_NUM_ACCUMULATORS * K_NUM_VERIFIED;
std::array<std::once_flag, K_LOG_FLAGS> dispatchLogFlags;

// Map a dispatch to a flat index into dispatchLogFlags. `padded` are the head
// dims of the resolved rows (bwd_dispatch::paddedHeadDims). isApplicable
// filters to the supported axes before buildPlan reaches this, so the asserts
// document the contract and the clamps are a defensive guard against an
// out-of-bounds index should an unexpected combination slip through.
size_t dispatchLogIndex(const std::string& archId,
                        const std::string& dtype,
                        bwd_dispatch::HeadDims padded,
                        int mask,
                        int atomic32,
                        bool verified)
{
    const size_t archIdx = (archId == "gfx950") ? 1U : 0U; // gfx942 -> 0, gfx950 -> 1
    const size_t dtypeIdx = (dtype == "fp16") ? 1U : 0U; // bf16 -> 0, fp16 -> 1

    size_t hdimIdx = 0U;
    if(padded.q == 64 && padded.v == 64)
    {
        hdimIdx = 0U;
    }
    else if(padded.q == 128 && padded.v == 128)
    {
        hdimIdx = 1U;
    }
    else if(padded.q == 192 && padded.v == 192)
    {
        hdimIdx = 2U;
    }
    else if(padded.q == 192 && padded.v == 128)
    {
        hdimIdx = 3U;
    }
    else
    {
        assert(false && "dispatchLogIndex: unexpected head dimensions");
    }

    auto maskIdx = static_cast<size_t>(mask);
    assert(maskIdx < K_NUM_MASKS && "dispatchLogIndex: mask ordinal out of range");
    if(maskIdx >= K_NUM_MASKS)
    {
        maskIdx = 0U;
    }

    const size_t accIdx = atomic32 == 0 ? 0U : 1U;
    const size_t verifiedIdx = verified ? 1U : 0U;
    return ((((archIdx * K_NUM_DTYPES + dtypeIdx) * K_NUM_HDIMS + hdimIdx) * K_NUM_MASKS + maskIdx)
                * K_NUM_ACCUMULATORS
            + accIdx)
               * K_NUM_VERIFIED
           + verifiedIdx;
}

} // namespace

namespace bwd_dispatch
{

bool isSupportedArch(const std::string& archId)
{
    return archId == "gfx942" || archId == "gfx950";
}

std::optional<PipelineStage> postStage(const std::string& archId, int atomic32, int hdimQ)
{
    // AITER v0.1.29 mha_bwd.cu fmha_v3_bwd(): `need_post_processing =
    // ((arch_id == "gfx950") && (a.hdim_q != 64)) || (a.v3_atomic_fp32 == 1)`
    // (:354-355), with post_cfgs dq_convert for A32 and, on gfx950, dq_shuffle
    // for A16 (:327-352).
    if(atomic32 == 1)
    {
        return PipelineStage::DQ_CONVERT;
    }
    if(archId == "gfx950" && hdimQ != 64)
    {
        return PipelineStage::DQ_SHUFFLE;
    }
    return std::nullopt;
}

int requestedMaskOrdinal(
    const std::string& archId, int mask, int atomic32, int64_t seqLenQ, int64_t seqLenKv)
{
    // AITER v0.1.29 mha_bwd.cu fmha_v3_bwd() (:374-377): `if(arch_id ==
    // "gfx942" && !a.v3_atomic_fp32 && mt == 2 && a.seqlen_q == a.seqlen_k)
    // mt = 1;`
    if(archId == "gfx942" && atomic32 == 0
       && mask == static_cast<int>(MaskType::BOTTOM_RIGHT_CAUSAL) && seqLenQ == seqLenKv)
    {
        return static_cast<int>(MaskType::TOP_LEFT_CAUSAL);
    }
    return mask;
}

bool dqSharesQStrides(
    const BhsStrides& q, const BhsStrides& dq, int64_t batch, int64_t heads, int64_t seqLen)
{
    return (batch <= 1 || q.batch == dq.batch) && (heads <= 1 || q.head == dq.head)
           && (seqLen <= 1 || q.seq == dq.seq);
}

HeadDims paddedHeadDims(const std::string& archId, int hdimQ, int hdimV)
{
    // AITER v0.1.29 mha_bwd.cu get_padded_hdim() (:10-32).
    if(hdimQ == 192 && hdimV == 128 && archId == "gfx950")
    {
        return {hdimQ, hdimV};
    }
    if(hdimQ == hdimV)
    {
        for(const int padded : {64, 128, 192})
        {
            if(hdimQ <= padded)
            {
                return {padded, padded};
            }
        }
    }
    return {hdimQ, hdimV};
}

bool isBatchShapeSupported(const std::string& archId, int64_t headDimQk, int64_t headDimV)
{
    // AITER v0.1.29 mha_bwd.cu fmha_v3_bwd() returns -1 for `hdim_q % 8` or
    // `hdim_v % 8` (:276-281).
    if(headDimQk % 8 != 0 || headDimV % 8 != 0)
    {
        return false;
    }
    if(archId == "gfx942")
    {
        // aiter/ops/mha.py can_impl_fmha_v3_bwd() (:2729-2731): `hdim_q ==
        // hdim_v` and `hdim_q >= 64 and hdim_q <= 192`.
        return headDimQk == headDimV && headDimQk >= 64 && headDimQk <= 192;
    }
    if(archId == "gfx950")
    {
        // aiter/ops/mha.py can_impl_fmha_v3_bwd_gfx950() (:2851-2854):
        // `(hdim_q > 64 and hdim_q <= 128 and hdim_q == hdim_v) or (hdim_q == 192
        // and hdim_v == 128)`. gfx950 hd64 and hd192 x hd192 rows ship but go to CK.
        return (headDimQk > 64 && headDimQk <= 128 && headDimQk == headDimV)
               || (headDimQk == 192 && headDimV == 128);
    }
    return false;
}

bool isBatchWindowSupported(const std::string& archId, int64_t headDimQk)
{
    if(archId == "gfx950")
    {
        // aiter/ops/mha.py can_impl_fmha_v3_bwd_gfx950() (:2855-2856): "sliding
        // window kernels are hd128 only", `not swa or hdim_q <= 128`, where any
        // window counts (:2829-2830, "a left size of 0 is a window too").
        return headDimQk <= 128;
    }
    // aiter/ops/mha.py can_impl_fmha_v3_bwd() serves windows through psskddv()
    // for 64 < D <= 128 only (:2718). Its window test (:2604, `swa =
    // (window_size_left > 0) or (window_size_right > 0)`) does not count a
    // window with left bound 0 and right bound 0 or -1, which then falls out of
    // every family. That is a classification artifact, not a kernel limit: the
    // engine classifies the band itself, and the gfx942 window kernel serves it
    // like any other window.
    return headDimQk > 64 && headDimQk <= 128;
}

bool psskOffStridesMatch(const BhsStrides& q,
                         const BhsStrides& k,
                         const BhsStrides& v,
                         const BhsStrides& dO,
                         const BhsStrides& dk,
                         const BhsStrides& dv,
                         int64_t nheadQ,
                         int64_t nheadK)
{
    // aiter/ops/mha.py can_impl_fmha_v3_bwd() np() (:2633-2642) and pddv()
    // (:2684-2693), on element strides. The batch-stride ratio test
    // `batch_stride_dk / batch_stride_k == nhead_q / nhead_k` is true division
    // in Python, cross-multiplied here; a zero K or V batch stride, or a product
    // that overflows, fails it.
    const auto batchRatioMatches = [&](int64_t gradBatch, int64_t inputBatch) {
        int64_t lhs = 0;
        int64_t rhs = 0;
        return inputBatch != 0 && nheadK != 0 && !__builtin_mul_overflow(gradBatch, nheadK, &lhs)
               && !__builtin_mul_overflow(inputBatch, nheadQ, &rhs) && lhs == rhs;
    };
    return q.seq == dO.seq && q.head == dO.head && q.batch == dO.batch && k.seq == v.seq
           && k.head == v.head && k.batch == v.batch && k.head == dk.head && v.head == dv.head
           && batchRatioMatches(dk.batch, k.batch) && batchRatioMatches(dv.batch, v.batch);
}

int64_t bhsdSpanBytes(const BhsStrides& strides,
                      int64_t batch,
                      int64_t heads,
                      int64_t seqLen,
                      int64_t headDim,
                      int64_t elementBytes)
{
    if(batch <= 0 || heads <= 0 || seqLen <= 0 || headDim <= 0)
    {
        return 0;
    }
    // ((B - 1) * batch + (H - 1) * head + (S - 1) * seq + D) * elementBytes
    int64_t elements = headDim;
    for(const auto& [extent, stride] : {std::pair{batch, strides.batch},
                                        std::pair{heads, strides.head},
                                        std::pair{seqLen, strides.seq}})
    {
        int64_t offset = 0;
        if(__builtin_mul_overflow(extent - 1, stride, &offset)
           || __builtin_add_overflow(elements, offset, &elements))
        {
            return INT64_MAX;
        }
    }
    int64_t bytes = 0;
    return __builtin_mul_overflow(elements, elementBytes, &bytes) ? INT64_MAX : bytes;
}

bool qBatchStrideMatchesDqAcc(
    const BhsStrides& q, int64_t batch, int64_t heads, int64_t seqLen, int64_t headDim)
{
    // bwd_hd128_bf16_a32_rtne.co prologue, on byte strides: t = b *
    // batch_stride_q, then t >>= 6 when batch_stride_q >= stride_q and
    // t = (t >> 8) * S_q * 4 otherwise, and the dq_acc byte offset is
    // (4 * S_q * h + t) * 128. That is the packed dq_acc offset
    // (b * H_q + h) * S_q * D_qk * 4 at D_qk = 128 only for the two batch
    // strides below.
    if(batch <= 1)
    {
        return true;
    }
    int64_t headElems = 0;
    int64_t batchElems = 0;
    if(__builtin_mul_overflow(heads, headDim, &headElems)
       || __builtin_mul_overflow(headElems, seqLen, &batchElems))
    {
        return false;
    }
    return q.batch >= q.seq ? q.batch == batchElems : q.batch == headElems;
}

bool psskOffOffsetsFitU32(const KernelRequest& request)
{
    constexpr auto K_U32_MAX = static_cast<int64_t>(UINT32_MAX);
    if(request.gradKvSpanBytes > K_U32_MAX)
    {
        return false;
    }
    if(request.atomic32 == 0)
    {
        return request.inputSpanBytes <= K_U32_MAX;
    }
    // dq_acc, whose batch offset the kernels derive from Q's batch stride.
    const auto dqAcc = sdpaBwdDqAccLayout(static_cast<size_t>(request.batch),
                                          static_cast<size_t>(request.nheadQ),
                                          static_cast<size_t>(request.seqLenQ),
                                          static_cast<size_t>(request.hdimQ),
                                          AccumulatorType::A32);
    return request.qBatchStrideMatchesDqAcc
           && sdpaBwdDqAccBytesFitU32(
               dqAcc, static_cast<size_t>(request.batch), static_cast<size_t>(request.nheadQ));
}

ClassifiedRequest
    makeKernelRequest(const std::string& archId,
                      const std::string& dataTypeId,
                      const hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes& attrs,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& q,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& k,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& v,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dO,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dk,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dv,
                      AccumulatorMode accMode)
{
    ClassifiedRequest result{};
    auto& request = result.request;
    const auto* qDims = q.dims();
    request.dataType = dataTypeId;
    request.hdimQ = static_cast<int>(qDims->Get(3));
    request.hdimV = static_cast<int>(v.dims()->Get(3));
    request.seqLenQ = qDims->Get(2);
    request.seqLenKv = k.dims()->Get(2);

    result.band = plan_utils::makeCoveringBoundsUnbound(
        plan_utils::resolveDiagonalBand(attrs), request.seqLenQ, request.seqLenKv);
    result.maskType = plan_utils::getMaskType(
        result.band.leftBound, result.band.rightBound, result.band.alignment);
    request.mask
        = requestedMaskOrdinal(archId,
                               dqdkdvMaskOrdinal(archId, result.maskType, result.band.alignment),
                               static_cast<int>(accMode),
                               request.seqLenQ,
                               request.seqLenKv);

    request.atomic32 = static_cast<int>(accMode);
    request.mode = static_cast<int>(getBatchMode(attrs));
    request.bf16Cvt = (dataTypeId == "fp16") ? BF16_CVT_FP16_SENTINEL
                                             : static_cast<int>(getRoundingMode(attrs));
    request.psskOffStridesOk = psskOffStridesMatch(bhsStrides(q),
                                                   bhsStrides(k),
                                                   bhsStrides(v),
                                                   bhsStrides(dO),
                                                   bhsStrides(dk),
                                                   bhsStrides(dv),
                                                   qDims->Get(1),
                                                   k.dims()->Get(1));
    request.batch = qDims->Get(0);
    request.nheadQ = qDims->Get(1);
    request.gradKvSpanBytes = std::max(spanBytes(dk), spanBytes(dv));
    request.inputSpanBytes = std::max({spanBytes(q), spanBytes(k), spanBytes(v), spanBytes(dO)});
    request.qBatchStrideMatchesDqAcc = qBatchStrideMatchesDqAcc(
        bhsStrides(q), request.batch, request.nheadQ, request.seqLenQ, qDims->Get(3));
    return result;
}

int dqdkdvMaskOrdinal(const std::string& archId,
                      MaskType maskType,
                      hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment alignment)
{
    // AITER v0.1.29 mha_bwd.cu asm_mask_type(): `else if(arch_id == "gfx950" &&
    // a.mask_type == mask_bottom_right) return 4;` for every mask that is
    // neither causal nor unmasked.
    if(archId == "gfx950" && maskType == MaskType::SLIDING_WINDOW
       && alignment == hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::BOTTOM_RIGHT)
    {
        return BWD_MASK_BOTTOM_RIGHT_SLIDING_WINDOW;
    }
    return static_cast<int>(maskType);
}

std::string lookupKernelNameKey(PipelineStage stage,
                                const std::string& archId,
                                const KernelRequest& request)
{
    const auto cfg = resolveStageConfig(stage, archId, request);
    return cfg.has_value() ? cfg->arch + cfg->knl_name : std::string{};
}

} // namespace bwd_dispatch

// Engine knob identifier for accumulator precision (a32 vs a16)
static constexpr const char* K_ACC_TYPE_KNOB_NAME = "hipkernel.sdpa.bwd.accumulator_type";

bool SdpaBwdPlanBuilder::isApplicable(
    const Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    // NOLINTNEXTLINE(readability-identifier-naming)
    static const char* HIP_KERNEL_LOG_PREFIX = "[SdpaBwdPlanBuilder::isApplicable] ";

    // Execute-time override shapes can diverge from the compile-time dims this
    // builder matched exactly; the family serves fixed prebuilt shapes, so decline
    // rather than risk a mismatch (RFC 0008 §4.6).
    HIP_KERNEL_RETURN_FALSE_IF(opGraph.getGraph().is_override_shape_enabled(),
                               "Graph has override shapes enabled");

    auto& nodeWrappers = opGraph.nodeWrappers();

    auto deviceStringOpt
        = plan_utils::tryGetDeviceString(handle.getStream(), "Could not query device string: ");
    if(!deviceStringOpt)
    {
        return false;
    }
    const std::string& deviceString = *deviceStringOpt;

    HIP_KERNEL_RETURN_FALSE_IF(
        !bwd_dispatch::isSupportedArch(deviceString),
        "Device string does not match gfx942 or gfx950 (Actual value: " + deviceString + ")");

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

    // Accept scale_tensor_uid only when it is a runtime pass-by-value scalar
    // (RFC 0016).  Non-pass-by-value scale tensors are not supported.
    if(attrs.scale_tensor_uid().has_value())
    {
        const auto& scaleTensorMap = opGraph.getTensorMap();
        const auto scaleIt = scaleTensorMap.find(attrs.scale_tensor_uid().value());
        HIP_KERNEL_RETURN_FALSE_IF(scaleIt == scaleTensorMap.end(),
                                   "scale_tensor_uid not found in tensor map");
        HIP_KERNEL_RETURN_FALSE_IF(
            !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(scaleIt->second),
            "scale tensor must be pass-by-value (compile-time constant or runtime)");
    }

    if(!plan_utils::isAttnScaleApplicable(attrs, opGraph.getTensorMap(), HIP_KERNEL_LOG_PREFIX))
    {
        return false;
    }

    // --- Validate required tensors ---

    // Group mode (variable sequence lengths) requires a different kernarg
    // layout than the POC; deferred to the kernarg-layout abstraction.
    HIP_KERNEL_RETURN_FALSE_IF(
        getBatchMode(attrs) != BatchMode::BATCH,
        "group mode (seq_len_q_tensor_uid or seq_len_kv_tensor_uid set) is not supported");

    const auto& tensorMap = opGraph.getTensorMap();
    HIP_KERNEL_RETURN_FALSE_IF(!hipdnn_flatbuffers_sdk::utilities::hasNoRaggedTensorIds(tensorMap),
                               "Ragged tensors are not supported");
    HIP_KERNEL_RETURN_FALSE_IF(
        !hip_kernel_provider::core::utils::hasOnly16ByteAlignedTensors(tensorMap),
        "All tensors must be aligned to a multiple of 16 bytes");

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

    // Every dimension below is narrowed to 32 bits before it reaches a kernel,
    // starting with the head-dim cast and the mask classification further down,
    // so the range gate comes first.
    HIP_KERNEL_RETURN_FALSE_IF(
        !wouldBwdDimsFitKernargFields(*qTensor, *kTensor, *vTensor),
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
        HIP_KERNEL_RETURN_FALSE_IF(dims->size() != 4 || strides->size() != 4,
                                   "rank-4 dims and strides required for " + describe());
        HIP_KERNEL_RETURN_FALSE_IF(strides->Get(3) != 1,
                                   "head-dim (innermost) stride must be 1 for " + describe());
        HIP_KERNEL_RETURN_FALSE_IF(strides->Get(2) < dims->Get(3),
                                   "sequence stride must be at least the head dim for "
                                       + describe());
        bool oddIndexedStride = false;
        for(auto axis = io.firstIndexedAxis; axis < 3; ++axis)
        {
            if(dims->Get(axis) > 1 && strides->Get(axis) % 2 != 0)
            {
                oddIndexedStride = true;
            }
        }
        HIP_KERNEL_RETURN_FALSE_IF(oddIndexedStride,
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
        HIP_KERNEL_RETURN_FALSE_IF(
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
    HIP_KERNEL_RETURN_FALSE_IF(numHeadsKv == 0 || numHeadsQ != numHeadsKv,
                               "GQA/MQA backward requires dK/dV head expansion, which is not "
                               "implemented (Actual: nhead_q="
                                   + std::to_string(numHeadsQ)
                                   + ", nhead_k=" + std::to_string(numHeadsKv) + ")");

    // Stats is FP32 (LSE from forward pass)
    HIP_KERNEL_RETURN_FALSE_IF(statsTensor->data_type() != DataType::FLOAT,
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
    HIP_KERNEL_RETURN_FALSE_IF(
        (statsDims->size() != 3 && (statsDims->size() != 4 || statsDims->Get(3) != 1))
            || statsStrides->size() != statsDims->size() || statsDims->Get(0) != batchSize
            || statsDims->Get(1) != numHeadsQ || statsDims->Get(2) != seqLenQ,
        "stats tensor must be [B, H_q, S_q] or [B, H_q, S_q, 1] matching Q (Actual dims: "
            + toString(statsDims) + ", strides: " + toString(statsStrides) + ")");
    HIP_KERNEL_RETURN_FALSE_IF(
        (batchSize != 1 && statsStrides->Get(0) != numHeadsQ * seqLenQ)
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

    // Head dims AITER does not send to the ASM kernels on this arch, including
    // the ones the registry has rows for (see bwd_dispatch::isBatchShapeSupported).
    HIP_KERNEL_RETURN_FALSE_IF(
        !bwd_dispatch::isBatchShapeSupported(deviceString, headDimQk, headDimV),
        deviceString + " backward does not serve D_qk = " + std::to_string(headDimQk)
            + ", D_v = " + std::to_string(headDimV));

    // Each kernel takes B, H, S_q, S_kv, D_qk and D_v once, from Q, K and V, and
    // addresses every tensor with them, so O, dO and dQ must have Q's batch,
    // heads and sequence, and V, dK and dV must have K's; dQ and dK carry D_qk,
    // O, dO and dV carry D_v. The frontend lets V have fewer heads than K (it
    // checks only that both divide H_q); DQDKDV then reads V and writes dV past
    // their last head. On gfx942, H_q = H_k = 4 with H_v = 2 or 1 gave wrong dQ,
    // dK and dV.
    HIP_KERNEL_RETURN_FALSE_IF(
        kTensor->dims()->Get(0) != batchSize || kTensor->dims()->Get(3) != headDimQk,
        "k tensor batch and head dim must equal q's (Actual dims: k " + toString(kTensor->dims())
            + ", q " + toString(qTensor->dims()) + ")");
    struct SameDims
    {
        const char* name;
        const TensorAttributes* tensor;
        const char* refName;
        // Batch, heads and sequence come from ref.
        const TensorAttributes* ref;
        int64_t headDim;
    };
    const std::array<SameDims, 6> sameDims{{{"o", oTensor, "q", qTensor, headDimV},
                                            {"do", doTensor, "q", qTensor, headDimV},
                                            {"dq", dqTensor, "q", qTensor, headDimQk},
                                            {"v", vTensor, "k", kTensor, headDimV},
                                            {"dk", dkTensor, "k", kTensor, headDimQk},
                                            {"dv", dvTensor, "k", kTensor, headDimV}}};
    for(const auto& match : sameDims)
    {
        const auto* dims = match.tensor->dims();
        const auto* refDims = match.ref->dims();
        HIP_KERNEL_RETURN_FALSE_IF(!std::equal(dims->begin(), dims->begin() + 3, refDims->begin())
                                       || dims->Get(3) != match.headDim,
                                   std::string(match.name) + " tensor dims must be " + match.refName
                                       + " dims with head dim " + std::to_string(match.headDim)
                                       + " (Actual: " + match.name + " " + toString(dims) + ", "
                                       + match.refName + " " + toString(refDims) + ")");
    }

    // The odo, dqdkdv and dq_convert stages issue wide row loads/stores on every graph tensor,
    // and the backend rejects any pointer that violates the declared alignment, so decline when
    // a tensor declares less than the kernel needs. See SdpaPlanUtils.hpp for the per-tensor
    // needs; the dq_acc and D buffers come from the plan workspace and are not graph tensors.
    const auto underAligned = plan_utils::findUnderAlignedTensor({
        {"q", qTensor, plan_utils::K_ROW_ALIGNMENT_BYTES},
        {"k", kTensor, plan_utils::K_ROW_ALIGNMENT_BYTES},
        {"v", vTensor, plan_utils::K_ROW_ALIGNMENT_BYTES},
        {"o", oTensor, plan_utils::K_ROW_ALIGNMENT_BYTES},
        {"do", doTensor, plan_utils::K_ROW_ALIGNMENT_BYTES},
        {"dq", dqTensor, plan_utils::K_ROW_ALIGNMENT_BYTES},
        {"dk", dkTensor, plan_utils::K_ROW_ALIGNMENT_BYTES},
        {"dv", dvTensor, plan_utils::K_ROW_ALIGNMENT_BYTES},
        {"stats", statsTensor, plan_utils::K_STATS_ALIGNMENT_BYTES},
    });
    HIP_KERNEL_RETURN_FALSE_IF(underAligned.has_value(), *underAligned);

    // Classify the mask and build the lookup request. Contradictory mask
    // attributes are an invalid-input condition the engine declines rather than
    // dispatches. Mask types NO_MASK (0), TOP_LEFT_CAUSAL (1),
    // BOTTOM_RIGHT_CAUSAL (2), and SLIDING_WINDOW (3, or 4 for a bottom-right
    // window on gfx950) are all dispatched. The DQDKDV CSV registry carries rows
    // for each mask ordinal; ODO and DQ_CONVERT are mask-agnostic. Applicability
    // is gated by the window rules and the registry lookup below.
    // Applicability does not depend on the execution settings, so it validates
    // the default A32 accumulator. buildPlan declines an A16 request the A16
    // kernels cannot serve.
    const int64_t seqLenKv = kTensor->dims()->Get(2);
    bwd_dispatch::ClassifiedRequest classified{};
    try
    {
        classified = bwd_dispatch::makeKernelRequest(deviceString,
                                                     dataTypeId,
                                                     attrs,
                                                     *qTensor,
                                                     *kTensor,
                                                     *vTensor,
                                                     *doTensor,
                                                     *dkTensor,
                                                     *dvTensor,
                                                     AccumulatorMode::A32);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(std::string{HIP_KERNEL_LOG_PREFIX} + e.what());
        return false;
    }
    const auto& request = classified.request;

    // Windows AITER sends to CK (see bwd_dispatch::isBatchWindowSupported).
    HIP_KERNEL_RETURN_FALSE_IF(
        classified.maskType == MaskType::SLIDING_WINDOW
            && !bwd_dispatch::isBatchWindowSupported(deviceString, headDimQk),
        deviceString
            + " backward does not serve a sliding window at D_qk = " + std::to_string(headDimQk));

    std::optional<fmha_v3_bwdConfig> dqdkdvCfg;
    for(const auto stage : {bwd_dispatch::PipelineStage::ODO,
                            bwd_dispatch::PipelineStage::DQDKDV,
                            bwd_dispatch::PipelineStage::DQ_CONVERT})
    {
        auto cfg = resolveStageConfig(stage, deviceString, request);
        if(!cfg)
        {
            HIPDNN_PLUGIN_LOG_INFO(
                HIP_KERNEL_LOG_PREFIX
                << "No matching " << stageName(stage) << " kernel for arch=" << deviceString
                << " dtype=" << dataTypeId << " D_qk=" << headDimQk << " D_v=" << headDimV
                << " mask=" << request.mask << " atomic32=" << request.atomic32
                << " S_q=" << seqLenQ << " S_kv=" << seqLenKv << " pssk0_strides="
                << static_cast<int>(request.psskOffStridesOk) << " pssk0_offsets="
                << static_cast<int>(bwd_dispatch::psskOffOffsetsFitU32(request))
                << " mode=batch bf16_cvt=" << request.bf16Cvt);
            return false;
        }
        if(stage == bwd_dispatch::PipelineStage::DQDKDV)
        {
            dqdkdvCfg = std::move(cfg);
        }
    }

    // Reject oversized graphs whose byte strides would silently truncate when
    // packed into the kernarg uint32_t fields. Caught here rather than at
    // execute time so the engine can be excluded from heuristics for graphs it
    // cannot represent. Uses the resolved DQDKDV row's tile size for the Ts
    // product (tsKv * stride_k * BF16).
    HIP_KERNEL_RETURN_FALSE_IF(!wouldBwdByteStridesFitUint32(*qTensor,
                                                             *kTensor,
                                                             *vTensor,
                                                             *oTensor,
                                                             *doTensor,
                                                             *dqTensor,
                                                             *dkTensor,
                                                             *dvTensor,
                                                             batchSize,
                                                             numHeadsQ,
                                                             seqLenQ,
                                                             headDimQk,
                                                             dqdkdvCfg->ts),
                               "Backward byte strides overflow uint32_t kernarg fields");

    return true;
}

size_t SdpaBwdPlanBuilder::getMaxWorkspaceSize(
    const Handle& handle,
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

    // Whether a post stage reads a dq_acc depends on the arch for A16 only.
    // When the device cannot be queried, size for the dq_acc so the bound
    // holds on every arch.
    bool hasDqAcc = true;
    if(accType == AccumulatorType::A16)
    {
        const auto deviceString
            = plan_utils::tryGetDeviceString(handle.getStream(), "Could not query device string: ");
        hasDqAcc = !deviceString
                   || bwd_dispatch::postStage(*deviceString,
                                              static_cast<int>(AccumulatorMode::A16),
                                              static_cast<int>(headDimQk))
                          .has_value();
    }

    return sdpaBwdWorkspaceSize(batch, headsQ, seqLenQ, headDimQk, accType, hasDqAcc);
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
    // Defaults to A32 when the knob is unset.
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
    // The accumulator selects the atomic32 dqdkdv rows, the mask the lookup
    // asks for (requestedMaskOrdinal) and the post stage (postStage).
    const bool useA32 = (accType == AccumulatorType::A32);
    const auto classified = bwd_dispatch::makeKernelRequest(deviceString,
                                                            dataTypeId,
                                                            sdpaAttrs,
                                                            *qTensor,
                                                            *kTensor,
                                                            *vTensor,
                                                            *doTensor,
                                                            *dkTensor,
                                                            *dvTensor,
                                                            toAccumulatorMode(accType));
    const auto& request = classified.request;
    const auto& maskBand = classified.band;

    // isApplicable resolved every A32 stage, so an A32 miss here is an internal
    // error. isApplicable does not see the accumulator knob, and the A16
    // kernels serve fewer graphs (every gfx942 A16 row is pssk = 0, which needs
    // S_q == S_kv, S_kv % 64 == 0, the pssk = 0 stride layout and tensors that
    // span at most 4 GiB; neither arch ships A16 rows for every mask and head
    // dim), so an A16 request the A16 kernels cannot serve is declined as a bad
    // knob value. AITER's Python layer sends these graphs to CK (aiter/ops/mha.py
    // can_impl_fmha_v3_bwd()).
    const auto declineA16 = [&](const std::string& reason) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "SdpaBwdPlanBuilder::buildPlan: the a16 accumulator does not serve this graph on "
                + deviceString + " (" + reason + " dtype=" + dataTypeId
                + " D_qk=" + std::to_string(headDimQk) + " D_v=" + std::to_string(headDimV)
                + " mask=" + std::to_string(request.mask) + " S_q=" + std::to_string(seqLenQ)
                + " S_kv=" + std::to_string(seqLenKv) + " pssk0_strides="
                + std::to_string(static_cast<int>(request.psskOffStridesOk)) + " pssk0_offsets="
                + std::to_string(static_cast<int>(bwd_dispatch::psskOffOffsetsFitU32(request)))
                + "); use the a32 accumulator");
    };
    auto resolveStage = [&](bwd_dispatch::PipelineStage stage) {
        auto cfgOpt = resolveStageConfig(stage, deviceString, request);
        if(!cfgOpt)
        {
            if(!useA32)
            {
                declineA16(std::string("no ") + stageName(stage) + " kernel:");
            }
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                std::string("SdpaBwdPlanBuilder::buildPlan: failed to resolve ") + stageName(stage)
                    + " kernel for arch=" + deviceString + " dtype=" + dataTypeId
                    + " D_qk=" + std::to_string(headDimQk) + " D_v=" + std::to_string(headDimV)
                    + " (isApplicable should have rejected)");
        }
        return *cfgOpt;
    };
    auto toResolvedKernel = [](const fmha_v3_bwdConfig& cfg) {
        return ResolvedKernel{getKernelTocKey(cfg.co_name),
                              cfg.knl_name,
                              SdpaBwdParams::KernelTiles{static_cast<unsigned int>(cfg.ts)}};
    };

    const auto dqdkdvCfg = resolveStage(bwd_dispatch::PipelineStage::DQDKDV);
    const auto odoResolved = toResolvedKernel(resolveStage(bwd_dispatch::PipelineStage::ODO));
    const auto dqdkdvResolved = toResolvedKernel(dqdkdvCfg);

    // The post stage that writes dQ from dq_acc, if any (AITER mha_bwd.cu
    // :327-355). For the batch-mode graphs the engine serves, every post
    // lookup finds a row, so AITER's fallback to no post stage on a miss
    // (:404-412) never applies.
    const auto postStage = bwd_dispatch::postStage(deviceString, request.atomic32, request.hdimQ);
    std::optional<ResolvedKernel> postResolved;
    if(postStage)
    {
        postResolved = toResolvedKernel(resolveStage(*postStage));
    }

    if(!useA32)
    {
        // isApplicable checked the A32 rows' tile and dq_acc against the
        // 32-bit kernarg fields; check the A16 ones.
        const auto* kStridesAll = kTensor->strides();
        if(!plan_utils::byteStrideFitsU32(
               "Ts (tsKv * stride_k)", kStridesAll->Get(2), K_BF16_BYTES * dqdkdvCfg.ts))
        {
            declineA16("Ts overflows its 32-bit kernarg field:");
        }
        if(postStage)
        {
            // The DQDKDV kernels address dq_acc within a batch in 32 bits.
            if(!sdpaBwdDqAccFitsU32(
                   sdpaBwdDqAccLayout(
                       batchSize, numHeadsQ, seqLenQ, headDimQk, AccumulatorType::A16),
                   numHeadsQ))
            {
                declineA16("the dq_acc offsets overflow 32 bits:");
            }
        }
        else if(!bwd_dispatch::dqSharesQStrides(
                    bhsStrides(*qTensor), bhsStrides(*dqTensor), batchSize, numHeadsQ, seqLenQ))
        {
            // DQDKDV writes dQ itself and has no dQ strides.
            declineA16("dQ strides differ from Q strides and the kernel writes dQ directly:");
        }
    }

    // Surface, exactly once per (arch, dtype, row head dims, mask, accumulator,
    // verified) over the program lifetime, whether the dqdkdv kernel about to
    // be dispatched has a calibrated CPU reference. The lambda is noexcept
    // because it only logs (a throwing call_once initializer would let the
    // exception escape, which bugprone-exception-escape forbids).
    const bool verified = isVerifiedDispatch(deviceString, request, dqdkdvCfg);
    std::call_once(dispatchLogFlags[dispatchLogIndex(
                       deviceString,
                       dataTypeId,
                       bwd_dispatch::HeadDims{dqdkdvCfg.hdim_q, dqdkdvCfg.hdim_v},
                       request.mask,
                       request.atomic32,
                       verified)],
                   [&]() noexcept {
                       if(verified)
                       {
                           HIPDNN_PLUGIN_LOG_INFO("SDPA bwd dispatch: verified kernel arch="
                                                  << deviceString << " dtype=" << dataTypeId
                                                  << " D_qk=" << headDimQk << " D_v=" << headDimV
                                                  << " mask=" << request.mask
                                                  << " atomic32=" << request.atomic32);
                       }
                       else
                       {
                           HIPDNN_PLUGIN_LOG_WARN(
                               "SDPA bwd dispatch: UNVERIFIED kernel arch="
                               << deviceString << " dtype=" << dataTypeId << " D_qk=" << headDimQk
                               << " D_v=" << headDimV << " mask=" << request.mask << " atomic32="
                               << request.atomic32 << " kernel=" << dqdkdvCfg.knl_name
                               << " - results not validated against CPU reference");
                       }
                   });

    HIPDNN_PLUGIN_LOG_INFO("Using bwd odo kernel: " << odoResolved.tocKey
                                                    << " :: " << odoResolved.knlName);
    HIPDNN_PLUGIN_LOG_INFO("Using bwd dqdkdv kernel: " << dqdkdvResolved.tocKey
                                                       << " :: " << dqdkdvResolved.knlName);
    if(postResolved)
    {
        HIPDNN_PLUGIN_LOG_INFO("Using bwd " << stageName(*postStage)
                                            << " kernel: " << postResolved->tocKey
                                            << " :: " << postResolved->knlName);
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

    // One line per built plan: the main kernel (dq/dk/dv), not the helper stages. The
    // integration harness reads it (the format is fixed: "kernel: <engine> <kernel>").
    HIPDNN_PLUGIN_LOG_INFO("kernel: " << hipdnn_data_sdk::utilities::ASM_SDPA_ENGINE_NAME << ' '
                                      << dqdkdvResolved.knlName);

    std::optional<CachedModule> postKernel;
    if(postResolved)
    {
        auto loaded = moduleCache().getOrLoad(
            postResolved->tocKey, deviceString, postResolved->knlName.c_str());
        if(!loaded)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                std::string("SdpaBwdPlanBuilder::buildPlan: failed to load ")
                    + stageName(*postStage) + " kernel tocKey='" + postResolved->tocKey + "' arch='"
                    + deviceString + "'");
        }
        postKernel = std::move(loaded);
    }

    // -------------------------------------------------------------------------
    // 6. Build params and create plan
    // -------------------------------------------------------------------------
    SdpaBwdParams params{};
    params.odoTiles = odoResolved.tiles;
    params.dqdkdvTiles = dqdkdvResolved.tiles;
    if(postResolved)
    {
        params.postTiles = postResolved->tiles;
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
    params.archString = deviceString;
    params.maskOrdinal = request.mask;
    // A surviving bound is strictly below its sequence span (the covering ones
    // became -1), and isApplicable capped both sequence lengths at INT32_MAX, so
    // the narrowing below and the mask coordinates SdpaBwdPlan passes from it
    // are exact.
    if(classified.maskType == MaskType::SLIDING_WINDOW)
    {
        params.windowLeft = static_cast<int32_t>(maskBand.leftBound);
        params.windowRight = static_cast<int32_t>(maskBand.rightBound);
        params.topLeftAlignment
            = maskBand.alignment
              != hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    }

    executionContext.setPlan(std::make_unique<SdpaBwdPlan>(
        std::move(odoKernel), std::move(dqdkdvKernel), std::move(postKernel), params));
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
        = "Accumulator precision for the backward dQ gradient. a32: dQ accumulates in FP32 and a "
          "dq_convert kernel writes it. a16: dQ accumulates in 16 bits; gfx942 writes dQ "
          "directly, gfx950 accumulates into a 16-bit buffer that a dq_shuffle kernel writes "
          "to dQ. a16 serves fewer graphs than a32 and fails plan building for the others";

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
