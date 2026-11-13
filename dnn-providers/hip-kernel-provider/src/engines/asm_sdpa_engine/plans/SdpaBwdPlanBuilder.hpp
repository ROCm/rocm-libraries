// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "SdpaModuleCache.hpp"
#include "SdpaPlanUtils.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include <cstdint>
#include <optional>
#include <string>

namespace asm_sdpa_engine
{

class SdpaBwdPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder<Handle, Settings, Context>
{
public:
    static SdpaModuleCache& moduleCache()
    {
        static SdpaModuleCache s_cache;
        return s_cache;
    }

    bool isApplicable(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getMaxWorkspaceSize(const Handle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const Settings& executionSettings) const override;

    void initializeExecutionSettings(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        Settings& executionSettings) const override;

    void buildPlan(const Handle& handle,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                   Context& executionContext) const override;

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>
        // NOLINTNEXTLINE(portability-template-virtual-member-function)
        getCustomKnobs(
            const Handle& handle,
            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
};

namespace bwd_dispatch
{

// Pipeline-stage tag used by the registry-lookup helper to select which
// CSV-derived registry to walk.
enum class PipelineStage
{
    ODO,
    DQDKDV,
    DQ_CONVERT,
    DQ_SHUFFLE
};

// Value stored in the bf16_cvt CSV column for rows that serve every rounding
// mode: the FP16 rows (where rounding mode is not applicable) and the bf16 rows
// that ship one kernel for all modes. FP16 lookups request it. Shared between
// the builder and its unit tests.
inline constexpr int BF16_CVT_FP16_SENTINEL = 3;

// True for the device architectures the backward engine dispatches on.
bool isSupportedArch(const std::string& archId);

// The post stage that writes dQ from dq_acc for the accumulator `atomic32`
// (1 = A32, 0 = A16), or std::nullopt when DQDKDV writes dQ itself. Mirrors
// AITER v0.1.29 mha_bwd.cu fmha_v3_bwd() (:327-355) on gfx942 and gfx950:
// A32 runs DQ_CONVERT; A16 runs DQ_SHUFFLE on gfx950 for hdim_q != 64 and has
// no post stage otherwise.
std::optional<PipelineStage> postStage(const std::string& archId, int atomic32, int hdimQ);

// The dqdkdv CSV mask ordinal the lookup requests. Mirrors AITER v0.1.29
// mha_bwd.cu fmha_v3_bwd() (:370-377): gfx942 ships no A16 bottom-right causal
// kernels, so with seqlen_q == seqlen_k, where the two causal masks agree, an
// A16 bottom-right causal mask (2) runs the top-left one (1).
int requestedMaskOrdinal(
    const std::string& archId, int mask, int atomic32, int64_t seqLenQ, int64_t seqLenKv);

// Head dims of a registry row.
struct HeadDims
{
    int q;
    int v;
};

// The head dims of the registry rows that serve (hdimQ, hdimV) on archId.
// Mirrors AITER v0.1.29 mha_bwd.cu get_padded_hdim(): D_qk == D_v rounds up
// to 64, 128 or 192; gfx950 (192, 128) and every other pair stay as they are.
HeadDims paddedHeadDims(const std::string& archId, int hdimQ, int hdimV);

// False for batch-mode head dims that AITER v0.1.29 does not send to the ASM
// backward on archId (aiter/ops/mha.py _flash_attn_backward and
// mha_bwd.cu fmha_v3_bwd()): D % 8 != 0 on either tensor; on gfx942 anything
// but 64 <= D <= 192 with D_qk == D_v; on gfx950 anything but
// 64 < D <= 128 with D_qk == D_v, or D_qk = 192 with D_v = 128.
bool isBatchShapeSupported(const std::string& archId, int64_t headDimQk, int64_t headDimV);

// False for a batch-mode sliding window (MaskType::SLIDING_WINDOW) that AITER
// v0.1.29 sends to CK on archId: gfx950 serves windows for D_qk <= 128 and
// gfx942 for 64 < D_qk <= 128 (aiter/ops/mha.py can_impl_fmha_v3_bwd_gfx950()
// and can_impl_fmha_v3_bwd() psskddv()). Any band the engine classifies as a
// window counts, including one that keeps only the diagonal.
bool isBatchWindowSupported(const std::string& archId, int64_t headDimQk);

// Element strides of one [B, H, S, D] tensor (the head-dim stride is 1).
struct BhsStrides
{
    int64_t batch;
    int64_t head;
    int64_t seq;
};

// True when dQ can be written with Q's strides: they agree on every axis
// longer than 1. The dqdkdv kernels that write dQ directly (no post stage)
// take no dQ strides (AITER v0.1.29 mha_bwd.h fmha_bwd_dqdkdv_args); AITER
// allocates dQ like Q (aiter/ops/mha.py:3031, torch.empty_like(q)).
bool dqSharesQStrides(
    const BhsStrides& q, const BhsStrides& dq, int64_t batch, int64_t heads, int64_t seqLen);

// True when the strides satisfy what the dqdkdv kernels without sequence
// padding (pssk = 0) assume. Mirrors the stride equalities of AITER v0.1.29
// aiter/ops/mha.py can_impl_fmha_v3_bwd() np() and pddv(): Q and dO share
// strides, K and V share strides, dK and dV share the K and V head strides,
// and the dK and dV batch strides are the K and V batch strides scaled by
// nhead_q / nhead_k.
bool psskOffStridesMatch(const BhsStrides& q,
                         const BhsStrides& k,
                         const BhsStrides& v,
                         const BhsStrides& dO,
                         const BhsStrides& dk,
                         const BhsStrides& dv,
                         int64_t nheadQ,
                         int64_t nheadK);

// Bytes from the first element of a [B, H, S, D] tensor with unit head-dim
// stride and non-negative strides to one past its last, saturating at
// INT64_MAX instead of overflowing. 0 for a tensor with no elements.
int64_t bhsdSpanBytes(const BhsStrides& strides,
                      int64_t batch,
                      int64_t heads,
                      int64_t seqLen,
                      int64_t headDim,
                      int64_t elementBytes);

// True when Q's batch stride is one the A32 dqdkdv kernels without sequence
// padding (pssk = 0) derive the dq_acc batch offset from correctly: B == 1, or
// H_q * S_q * D_qk elements when it is at least Q's sequence stride, or
// H_q * D_qk elements when it is below it (from the .co disassembly; dq_acc is
// FP32 [B, H_q, S_q, D_qk]).
bool qBatchStrideMatchesDqAcc(
    const BhsStrides& q, int64_t batch, int64_t heads, int64_t seqLen, int64_t headDim);

// The dqdkdv CSV `mask` ordinal for a classified mask on archId. Mirrors AITER
// mha_bwd.cu asm_mask_type(): on gfx950 a sliding window with bottom-right
// alignment selects the BWD_MASK_BOTTOM_RIGHT_SLIDING_WINDOW kernels; every
// other mask keeps its plan_utils::MaskType ordinal.
int dqdkdvMaskOrdinal(const std::string& archId,
                      plan_utils::MaskType maskType,
                      hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment alignment);

// The graph-side inputs of AITER v0.1.29 mha_bwd.cu get_heuristic_kernel().
// Head dims and sequence lengths are the graph's; the lookups pad the head
// dims themselves (paddedHeadDims).
struct KernelRequest
{
    std::string dataType; // "bf16" or "fp16"
    int hdimQ;
    int hdimV;
    int64_t seqLenQ;
    int64_t seqLenKv;
    int mask; // dqdkdv CSV mask ordinal (dqdkdvMaskOrdinal)
    int atomic32; // requested accumulator: 1 = A32, 0 = A16
    int mode; // 0 = batch, 1 = group
    // Rounding mode the CSV column encodes: 0/1/2 for the bf16 modes, 3 for fp16.
    // A row matches when its bf16_cvt equals this or is BF16_CVT_FP16_SENTINEL.
    int bf16Cvt;
    // psskOffStridesMatch() for the graph. False excludes every pssk = 0 row.
    bool psskOffStridesOk;
    // Graph facts the pssk = 0 offset guard (psskOffOffsetsFitU32) reads.
    // Zero spans fit.
    int64_t batch; // B
    int64_t nheadQ; // H_q
    // Largest bhsdSpanBytes() of dK and dV, and of Q, K, V and dO (dQ shares
    // Q's strides on every row without a post stage).
    int64_t gradKvSpanBytes;
    int64_t inputSpanBytes;
    // qBatchStrideMatchesDqAcc() for the graph.
    bool qBatchStrideMatchesDqAcc;
};

// True when every (batch, head) offset the dqdkdv kernels without sequence
// padding (pssk = 0) form in 32 bits fits request's graph. Those kernels add
// the offset to the tensor base in 32 bits, where the pssk = 1 kernels add it
// in 64 bits (from the .co disassembly): A32 rows for dq_acc (FP32
// [B, H_q, S_q, D_qk], whose batch offset they derive from Q's batch stride),
// dK and dV; A16 rows for Q, K, V, dO, dQ, dK and dV as well. Each such tensor
// must span at most UINT32_MAX bytes, and on A32 rows Q's batch stride must be
// one the kernels read correctly (request.qBatchStrideMatchesDqAcc). False
// excludes every pssk = 0 row; the rule is applied to every pssk = 0 row,
// including the few whose disassembly shows 64-bit bases.
bool psskOffOffsetsFitU32(const KernelRequest& request);

// The dqdkdv CSV `atomic32` value of each accumulator.
enum class AccumulatorMode : int
{
    A16 = 0, // 16-bit accumulator (atomic32 = 0)
    A32 = 1 // 32-bit accumulator (atomic32 = 1)
};

// The KernelRequest for a backward graph, with the mask it classified.
struct ClassifiedRequest
{
    KernelRequest request;
    plan_utils::DiagonalBand band;
    plan_utils::MaskType maskType;
};

// The KernelRequest for a backward graph on archId with accumulator accMode.
// isApplicable and buildPlan both build it here, so the two sites resolve the
// same rows. The mask is classified on the graph dims (dqdkdvMaskOrdinal), and
// the request carries the ordinal the lookup asks for (requestedMaskOrdinal).
// Expects rank-4 [B, H, S, D] tensors. Throws HipdnnPluginException for
// contradictory mask attributes.
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
                      AccumulatorMode accMode);

// Resolves the registry row AITER v0.1.29 get_heuristic_kernel() picks for
// `stage`, and returns its composite key (arch + knl_name), or an empty string
// when no row matches. Each stage matches on AITER's key for it:
// - ODO: dtype, padded hdim_v and mode.
// - DQDKDV: dtype, padded hdim_q and hdim_v, mask, atomic32, bf16_cvt and mode,
//   then (pssk, pddv). pddv is 1 when padding changed a head dim; pssk is 1
//   unless seqlen_q == seqlen_k and seqlen_k is a multiple of the row's tile
//   (64 for A16 rows on gfx942 and for *recompile* A16 rows). The exact
//   (pssk, pddv) row wins; otherwise the smallest (pssk, pddv) above the
//   requested one in both fields, in (pssk, pddv) order (AITER takes whichever
//   such row its unordered_map yields last). pssk = 0 rows also need
//   request.psskOffStridesOk and psskOffOffsetsFitU32(request).
// - DQ_CONVERT and DQ_SHUFFLE: padded hdim_q, mode and bf16_cvt, plus dtype
//   when atomic32 = 1 (an A16 fp16 request takes the bf16 dq_shuffle rows).
// Exposed for unit testing of the dispatch logic.
std::string lookupKernelNameKey(PipelineStage stage,
                                const std::string& archId,
                                const KernelRequest& request);

} // namespace bwd_dispatch

} // namespace asm_sdpa_engine
