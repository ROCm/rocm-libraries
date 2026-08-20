// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "asm/AsmKernelPath.hpp"
#include "core/Handle.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaBwdParams.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaBwdPlan.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaModuleCache.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

/**
 * @file AsmSdpaBackwardNative.cpp
 * @brief The ASM SDPA backward engine's native half, against descriptors carrying what
 *        used to be `fmha_v3_bwd/fmha_bwd_dqdkdv.csv`.
 *
 * Three kernels cooperate to run one backward pass -- odo (D reduction), dqdkdv (the
 * gradient kernel), and, on the a32 accumulator path only, dq_convert (FP32 -> BF16
 * cast) -- but only dqdkdv is a real choice: the CSV keys it on dtype, head dim, mask
 * and accumulator type, while odo and dq_convert vary only by (dtype, head dim), a
 * strict subset dqdkdv already carries. Modeling all three as one catalog would need a
 * join the ingestor has no vocabulary for and would let a graph mismatch its own D
 * buffer to a different (dtype, hdim) odo kernel, which is not a real configuration
 * AITER ships. So this pack's `kernels` array holds only the 32 dqdkdv variants the
 * kernel matcher below prunes against, and the dispatch handler looks the odo and
 * dq_convert companions up from the winning dqdkdv kernel's own dtype/hdim metadata,
 * against a small literal table generated from the same CSVs (see ODO_TABLE_GFX942 and
 * DQ_CONVERT_TABLE_GFX942 below) -- the same relationship SdpaBwdPlanBuilder::buildPlan()
 * expressed as three CFG lookups keyed off one resolved dtype/hdim.
 *
 * The accumulator axis (CSV column `atomic32`) is `acc_type`, a *string* KMD field
 * over {"a32", "a16"} rather than an int: it is also this engine's user-facing knob,
 * and its two values pick qualitatively different pipelines (3 kernels vs 2), not
 * points on a numeric range GenericPlanBuilder's IntConstraint would suggest they are.
 *
 * What no descriptor field can carry is the same three things the forward pack found:
 * applicability (the mask/dropout/dtype/hdim/GQA/stride gate, byte-stride overflow
 * check, and group-mode exclusion, all restated from SdpaBwdPlanBuilder::isApplicable),
 * which code object (the CSV path plus which arch table the running device selects),
 * and launch geometry plus the kernarg structs -- all delegated to the existing
 * SdpaBwdParams/SdpaBwdPlan pair rather than restated here.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace plan_utils = asm_sdpa_engine::plan_utils;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.asm_sdpa_bwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.asm_sdpa_bwd.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.asm_sdpa_bwd.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.asm_sdpa_bwd.dispatch";

// KMD fields, restating the dqdkdv CSV's dispatch tuple (see
// AsmSdpaBackwardVariantFields.kmd.json for why hdim_q/hdim_v collapse to one field
// and pssk/pddv/mode are carried despite every authored row pinning them).
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HDIM_FIELD = "hdim";
constexpr std::string_view MASK_FIELD = "mask";
constexpr std::string_view ACC_TYPE_FIELD = "sdpa.bwd.accumulator_type";
constexpr std::string_view TS_FIELD = "ts";

constexpr std::string_view ACC_TYPE_A32 = "a32";

// Tokens the graph matcher binds for dispatch, so prepare() re-reads a decision rather
// than re-deriving it (RFC 0017 §8.1).
constexpr std::string_view Q_TOKEN = "asm_sdpa_bwd.q.uid";
constexpr std::string_view K_TOKEN = "asm_sdpa_bwd.k.uid";
constexpr std::string_view V_TOKEN = "asm_sdpa_bwd.v.uid";
constexpr std::string_view O_TOKEN = "asm_sdpa_bwd.o.uid";
constexpr std::string_view DO_TOKEN = "asm_sdpa_bwd.do.uid";
constexpr std::string_view STATS_TOKEN = "asm_sdpa_bwd.stats.uid";
constexpr std::string_view DQ_TOKEN = "asm_sdpa_bwd.dq.uid";
constexpr std::string_view DK_TOKEN = "asm_sdpa_bwd.dk.uid";
constexpr std::string_view DV_TOKEN = "asm_sdpa_bwd.dv.uid";

/// Rank every operand of a servable graph has.
constexpr uint32_t SDPA_RANK = 4;
/// Every shipped kernel is uniform bf16 or fp16, 2 bytes either way.
constexpr int64_t K_ELEM_BYTES = 2;
constexpr int64_t K_FP32_BYTES = 4;

// ---------------------------------------------------------------------------
// odo / dq_convert companion tables
// ---------------------------------------------------------------------------

/// One row of the odo or dq_convert CSV, keyed by (dtype, hdim) -- the only axes
/// either kernel varies along in batch mode (see the file comment above).
struct CompanionRow
{
    std::string_view dtype;
    int hdim;
    std::string_view coName;
    std::string_view knlName;
    unsigned int ts;
};

// clang-format off
constexpr std::array<CompanionRow, 6> ODO_TABLE_GFX942 = {{
    {"bf16", 64, "gfx942/fmha_v3_bwd/bwd_hd64_odo_bf16.co", "_ZN5aiter22fmha_bwd_hd64_odo_bf16E", 128},
    {"bf16", 128, "gfx942/fmha_v3_bwd/bwd_hd128_odo_bf16.co", "_ZN5aiter23fmha_bwd_hd128_odo_bf16E", 128},
    {"bf16", 192, "gfx942/fmha_v3_bwd/bwd_hd192_odo_bf16.co", "_ZN5aiter23fmha_bwd_hd192_odo_bf16E", 128},
    {"fp16", 64, "gfx942/fmha_v3_bwd/bwd_hd64_odo_fp16.co", "_ZN5aiter22fmha_bwd_hd64_odo_fp16E", 128},
    {"fp16", 128, "gfx942/fmha_v3_bwd/bwd_hd128_odo_fp16.co", "_ZN5aiter23fmha_bwd_hd128_odo_fp16E", 128},
    {"fp16", 192, "gfx942/fmha_v3_bwd/bwd_hd192_odo_fp16.co", "_ZN5aiter23fmha_bwd_hd192_odo_fp16E", 128},
}};

constexpr std::array<CompanionRow, 6> DQ_CONVERT_TABLE_GFX942 = {{
    {"bf16", 64, "gfx942/fmha_v3_bwd/bwd_hd64_dq_convert_bf16_rtne.co", "_ZN5aiter34fmha_bwd_hd64_dq_convert_bf16_rtneE", 64},
    {"bf16", 128, "gfx942/fmha_v3_bwd/bwd_hd128_dq_convert_bf16_rtne.co", "_ZN5aiter35fmha_bwd_hd128_dq_convert_bf16_rtneE", 64},
    {"bf16", 192, "gfx942/fmha_v3_bwd/bwd_hd192_dq_convert_bf16_rtne.co", "_ZN5aiter35fmha_bwd_hd192_dq_convert_bf16_rtneE", 64},
    {"fp16", 64, "gfx942/fmha_v3_bwd/bwd_hd64_dq_convert_fp16.co", "_ZN5aiter29fmha_bwd_hd64_dq_convert_fp16E", 64},
    {"fp16", 128, "gfx942/fmha_v3_bwd/bwd_hd128_dq_convert_fp16.co", "_ZN5aiter30fmha_bwd_hd128_dq_convert_fp16E", 64},
    {"fp16", 192, "gfx942/fmha_v3_bwd/bwd_hd192_dq_convert_fp16.co", "_ZN5aiter30fmha_bwd_hd192_dq_convert_fp16E", 64},
}};

constexpr std::array<CompanionRow, 6> ODO_TABLE_GFX950 = {{
    {"bf16", 64, "gfx950/fmha_v3_bwd/bwd_hd64_odo_bf16.co", "_ZN5aiter22fmha_bwd_hd64_odo_bf16E", 128},
    {"bf16", 128, "gfx950/fmha_v3_bwd/bwd_hd128_odo_bf16.co", "_ZN5aiter23fmha_bwd_hd128_odo_bf16E", 128},
    {"bf16", 192, "gfx950/fmha_v3_bwd/bwd_hd192_odo_bf16.co", "_ZN5aiter23fmha_bwd_hd192_odo_bf16E", 128},
    {"fp16", 64, "gfx950/fmha_v3_bwd/bwd_hd64_odo_fp16.co", "_ZN5aiter22fmha_bwd_hd64_odo_fp16E", 128},
    {"fp16", 128, "gfx950/fmha_v3_bwd/bwd_hd128_odo_fp16.co", "_ZN5aiter23fmha_bwd_hd128_odo_fp16E", 128},
    {"fp16", 192, "gfx950/fmha_v3_bwd/bwd_hd192_odo_fp16.co", "_ZN5aiter23fmha_bwd_hd192_odo_fp16E", 128},
}};

constexpr std::array<CompanionRow, 6> DQ_CONVERT_TABLE_GFX950 = {{
    {"bf16", 64, "gfx950/fmha_v3_bwd/bwd_hd64_dq_convert_bf16.co", "_ZN5aiter29fmha_bwd_hd64_dq_convert_bf16E", 64},
    {"bf16", 128, "gfx950/fmha_v3_bwd/bwd_hd128_dq_convert_bf16.co", "_ZN5aiter30fmha_bwd_hd128_dq_convert_bf16E", 64},
    {"bf16", 192, "gfx950/fmha_v3_bwd/bwd_hd192_dq_convert_bf16.co", "_ZN5aiter30fmha_bwd_hd192_dq_convert_bf16E", 64},
    {"fp16", 64, "gfx950/fmha_v3_bwd/bwd_hd64_dq_convert_fp16.co", "_ZN5aiter29fmha_bwd_hd64_dq_convert_fp16E", 64},
    {"fp16", 128, "gfx950/fmha_v3_bwd/bwd_hd128_dq_convert_fp16.co", "_ZN5aiter30fmha_bwd_hd128_dq_convert_fp16E", 64},
    {"fp16", 192, "gfx950/fmha_v3_bwd/bwd_hd192_dq_convert_fp16.co", "_ZN5aiter30fmha_bwd_hd192_dq_convert_fp16E", 64},
}};
// clang-format on

/// True when @p deviceArch (feature-suffix intact) is a gfx950 device. Only two arch
/// families ship backward kernels, so the alternative is gfx942; a third would need
/// its own table pair here, same as it would need its own KDP file.
bool isGfx950(std::string_view deviceArch)
{
    return deviceArch.rfind("gfx950", 0) == 0;
}

const CompanionRow*
    findCompanion(const std::array<CompanionRow, 6>& table, std::string_view dtype, int hdim)
{
    for(const auto& row : table)
    {
        if(row.dtype == dtype && row.hdim == hdim)
        {
            return &row;
        }
    }
    return nullptr;
}

/// The odo kernel for (dtype, hdim) on the resolved device arch, or nullptr. Every
/// (dtype, hdim) the graph matcher admits has one on both arches, so nullptr here is a
/// table/matcher drift, not an expected decline.
const CompanionRow* odoCompanion(std::string_view deviceArch, std::string_view dtype, int hdim)
{
    return findCompanion(isGfx950(deviceArch) ? ODO_TABLE_GFX950 : ODO_TABLE_GFX942, dtype, hdim);
}

/// The dq_convert kernel for (dtype, hdim), or nullptr. Only resolved for the a32 path.
const CompanionRow*
    dqConvertCompanion(std::string_view deviceArch, std::string_view dtype, int hdim)
{
    return findCompanion(
        isGfx950(deviceArch) ? DQ_CONVERT_TABLE_GFX950 : DQ_CONVERT_TABLE_GFX942, dtype, hdim);
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The tensor uids a matched SDPA backward graph binds.
struct SdpaBwdBinding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t doGrad = 0;
    int64_t stats = 0;
    int64_t dq = 0;
    int64_t dk = 0;
    int64_t dv = 0;
};

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    const auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// True when the tensor is a rank-4 operand whose byte strides fit the kernarg's
/// uint32 fields. A value that does not fit truncates silently and returns wrong
/// numbers, so this is a refusal rather than a runtime check.
bool isServableOperand(const data_objects::TensorAttributes* tensor, int64_t elemBytes)
{
    if(tensor == nullptr)
    {
        return false;
    }

    // A matcher runs on a graph nothing has validated: dims()/strides() are as
    // optional as each other even on a graph the frontend would otherwise reject.
    const auto* dims = tensor->dims();
    const auto* strides = tensor->strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK)
    {
        return false;
    }

    for(uint32_t axis = 0; axis < 3U; ++axis)
    {
        if(!plan_utils::byteStrideFitsU32("stride", strides->Get(axis), elemBytes))
        {
            return false;
        }
    }
    return true;
}

/// The single SDPA backward node this engine serves, or nullptr when the graph is not
/// one.
const data_objects::SdpaBackwardAttributes* sdpaBackwardNode(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return nullptr;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::SdpaBackwardAttributes
       || node.computeDataType() != data_objects::DataType::FLOAT)
    {
        return nullptr;
    }
    return &node.attributesAs<data_objects::SdpaBackwardAttributes>();
}

/// The mask class the graph asks for, matching the CSV's `mask` column, or nullopt
/// when the attributes contradict each other.
std::optional<int64_t> graphMask(const data_objects::SdpaBackwardAttributes& attrs)
{
    try
    {
        return static_cast<int64_t>(plan_utils::getMaskType(attrs));
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException&)
    {
        // Contradictory mask attributes are an invalid-input condition this engine
        // declines rather than dispatches.
        return std::nullopt;
    }
}

/// True when the graph asks for group (variable-length) mode, which needs a different
/// kernarg layout than this engine builds (see SdpaBwdPlanBuilder.cpp's equivalent
/// gate). Every catalog kernel is a batch-mode row, so a group-mode graph must be
/// declined at the graph gate rather than left to fail the kernel matcher.
bool isGroupMode(const data_objects::SdpaBackwardAttributes& attrs)
{
    return attrs.seq_len_q_tensor_uid().has_value() || attrs.seq_len_kv_tensor_uid().has_value();
}

/// The CSV dtype identifier the seven backward tensors share, or nullopt when they do
/// not all agree on one of the two supported dtypes.
std::optional<std::string_view> graphDataTypeId(const data_objects::TensorAttributes& q,
                                                const data_objects::TensorAttributes& k,
                                                const data_objects::TensorAttributes& v,
                                                const data_objects::TensorAttributes& doGrad,
                                                const data_objects::TensorAttributes& dq,
                                                const data_objects::TensorAttributes& dk,
                                                const data_objects::TensorAttributes& dv)
{
    const std::initializer_list<data_objects::DataType> types = {q.data_type(),
                                                                 k.data_type(),
                                                                 v.data_type(),
                                                                 doGrad.data_type(),
                                                                 dq.data_type(),
                                                                 dk.data_type(),
                                                                 dv.data_type()};
    if(plan_utils::allDataTypesEqual(data_objects::DataType::BFLOAT16, types))
    {
        return std::string_view("bf16");
    }
    if(plan_utils::allDataTypesEqual(data_objects::DataType::HALF, types))
    {
        return std::string_view("fp16");
    }
    return std::nullopt;
}

/**
 * @brief Graph-scoped applicability: is this a single SDPA backward this engine's
 *        prebuilt kernels can serve at all?
 *
 * Restates SdpaBwdPlanBuilder::isApplicable's graph-level gate: everything a kernel's
 * own metadata cannot answer. The shape question -- which of the catalog's dqdkdv rows
 * fits -- is the kernel-scoped matcher below.
 */
std::optional<BoundTokens> asmSdpaBackwardGraphMatches(const MatchContext& context)
{
    const auto* attrs = sdpaBackwardNode(context);
    if(attrs == nullptr)
    {
        return std::nullopt;
    }

    // Execute-time override shapes can diverge from the dims matched here, and this
    // engine serves fixed prebuilt shapes (RFC 0008 §4.6).
    if(context.graph.getGraph().is_override_shape_enabled())
    {
        return std::nullopt;
    }

    // Features with no prebuilt kernel. Each is a decline, not an error.
    if((attrs->dropout_probability().has_value() && attrs->dropout_probability().value() != 0.F)
       || attrs->alibi_mask() || attrs->padding_mask() || attrs->attn_mask_tensor_uid()
       || attrs->seed_tensor_uid() || attrs->offset_tensor_uid() || attrs->dropout_mask_tensor_uid()
       || attrs->dbias_tensor_uid())
    {
        return std::nullopt;
    }

    // Group mode (variable sequence lengths) needs a kernarg layout this engine does
    // not build; every catalog kernel is a batch-mode row.
    if(isGroupMode(*attrs))
    {
        return std::nullopt;
    }

    // A scale tensor is servable only pass-by-value; the kernarg carries a scalar, not
    // a pointer to read at launch (RFC 0016).
    if(attrs->scale_tensor_uid().has_value())
    {
        const auto* scale = findTensor(context, attrs->scale_tensor_uid().value());
        if(scale == nullptr || !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(scale))
        {
            return std::nullopt;
        }
    }

    const SdpaBwdBinding binding{attrs->q_tensor_uid(),
                                 attrs->k_tensor_uid(),
                                 attrs->v_tensor_uid(),
                                 attrs->o_tensor_uid(),
                                 attrs->do_tensor_uid(),
                                 attrs->stats_tensor_uid(),
                                 attrs->dq_tensor_uid(),
                                 attrs->dk_tensor_uid(),
                                 attrs->dv_tensor_uid()};

    const auto* q = findTensor(context, binding.q);
    const auto* k = findTensor(context, binding.k);
    const auto* v = findTensor(context, binding.v);
    const auto* o = findTensor(context, binding.o);
    const auto* doGrad = findTensor(context, binding.doGrad);
    const auto* stats = findTensor(context, binding.stats);
    const auto* dq = findTensor(context, binding.dq);
    const auto* dk = findTensor(context, binding.dk);
    const auto* dv = findTensor(context, binding.dv);
    if(q == nullptr || k == nullptr || v == nullptr || o == nullptr || doGrad == nullptr
       || stats == nullptr || dq == nullptr || dk == nullptr || dv == nullptr)
    {
        return std::nullopt;
    }

    if(q->dims() == nullptr || q->dims()->size() != SDPA_RANK || k->dims() == nullptr
       || k->dims()->size() != SDPA_RANK || v->dims() == nullptr || v->dims()->size() != SDPA_RANK)
    {
        return std::nullopt;
    }

    // GQA: the dqdkdv kernarg packs ratio = nhead_q / nhead_k (integer division). A
    // fractional ratio is a kernel-correctness violation (silent truncation), not a
    // "no row matches" catalog miss.
    const auto numHeadsQ = q->dims()->Get(1);
    const auto numHeadsKv = k->dims()->Get(1);
    if(numHeadsKv == 0 || numHeadsQ % numHeadsKv != 0)
    {
        return std::nullopt;
    }

    // Stats is FP32 (LSE from the forward pass).
    if(stats->data_type() != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }

    const auto dataTypeId = graphDataTypeId(*q, *k, *v, *doGrad, *dq, *dk, *dv);
    if(!dataTypeId.has_value())
    {
        return std::nullopt;
    }
    // Both supported dtypes (bf16, fp16) are 2 bytes; kept as a named constant rather
    // than inlined so a future third dtype's element size has an obvious edit point.
    const int64_t elemBytes = K_ELEM_BYTES;

    // Inputs only. o, do, dq, dk and dv are outputs the frontend infers, and their dims
    // and strides are not necessarily populated when matching runs, so requiring a rank-4
    // shape of them declines every graph that has not been through shape inference. The
    // builder this replaced rank-checked q, k and v alone for the same reason.
    if(!isServableOperand(q, elemBytes) || !isServableOperand(k, elemBytes)
       || !isServableOperand(v, elemBytes))
    {
        return std::nullopt;
    }
    // Stats/D-buffer strides are FP32, checked separately (only batch/head axes used,
    // matching wouldBwdByteStridesFitUint32's checkBwdTensor's partial check for it).
    const auto* statsStrides = stats->strides();
    if(statsStrides == nullptr || statsStrides->size() < 2
       || !plan_utils::byteStrideFitsU32("batch_stride_lsed", statsStrides->Get(0), K_FP32_BYTES)
       || !plan_utils::byteStrideFitsU32("nhead_stride_lsed", statsStrides->Get(1), K_FP32_BYTES))
    {
        return std::nullopt;
    }

    const auto headDimQk = q->dims()->Get(3);
    const auto headDimV = v->dims()->Get(3);
    if(headDimQk != headDimV)
    {
        return std::nullopt;
    }
    if(headDimQk != 64 && headDimQk != 128 && headDimQk != 192)
    {
        return std::nullopt;
    }

    if(!graphMask(*attrs).has_value())
    {
        return std::nullopt;
    }

    // dq_acc is FP32, contiguous [B, H_q, S_q, D_qk] -- the a32 path's byte-stride
    // guard from wouldBwdByteStridesFitUint32. Checked unconditionally: a graph this
    // engine accepts must be servable by its default (a32) knob value, and the knob
    // filter -- not this matcher -- is what a caller uses to ask for a16 instead.
    const int64_t seqLenQ = q->dims()->Get(2);
    const int64_t nheadStrideDqAcc = seqLenQ * headDimQk;
    const int64_t batchStrideDqAcc = numHeadsQ * nheadStrideDqAcc;
    if(!plan_utils::byteStrideFitsU32("stride_dq_acc", headDimQk, K_FP32_BYTES)
       || !plan_utils::byteStrideFitsU32("nhead_stride_dq_acc", nheadStrideDqAcc, K_FP32_BYTES)
       || !plan_utils::byteStrideFitsU32("batch_stride_dq_acc", batchStrideDqAcc, K_FP32_BYTES))
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound.emplace(std::string(Q_TOKEN), binding.q);
    bound.emplace(std::string(K_TOKEN), binding.k);
    bound.emplace(std::string(V_TOKEN), binding.v);
    bound.emplace(std::string(O_TOKEN), binding.o);
    bound.emplace(std::string(DO_TOKEN), binding.doGrad);
    bound.emplace(std::string(STATS_TOKEN), binding.stats);
    bound.emplace(std::string(DQ_TOKEN), binding.dq);
    bound.emplace(std::string(DK_TOKEN), binding.dk);
    bound.emplace(std::string(DV_TOKEN), binding.dv);
    return bound;
}

/**
 * @brief Kernel-scoped applicability: does this dqdkdv kernel's row fit the graph?
 *
 * The accumulator axis is deliberately absent here: both a32 and a16 rows for a shape
 * survive to the catalog, and the acc_type knob (not this matcher) decides between
 * them, the same way GenericPlanBuilder's knob filter narrows any other field.
 */
bool asmSdpaBackwardKernelMatches(const MatchContext& context,
                                  const BoundTokens& /*bound*/,
                                  const KernelDefinition& kernel)
{
    const auto* attrs = sdpaBackwardNode(context);
    if(attrs == nullptr)
    {
        return false;
    }

    const auto& tensorMap = context.graph.getTensorMap();
    const auto qIt = tensorMap.find(attrs->q_tensor_uid());
    if(qIt == tensorMap.end() || qIt->second == nullptr || qIt->second->dims() == nullptr
       || qIt->second->dims()->size() != SDPA_RANK)
    {
        return false;
    }

    const auto kIt = tensorMap.find(attrs->k_tensor_uid());
    const auto vIt = tensorMap.find(attrs->v_tensor_uid());
    const auto doIt = tensorMap.find(attrs->do_tensor_uid());
    const auto dqIt = tensorMap.find(attrs->dq_tensor_uid());
    const auto dkIt = tensorMap.find(attrs->dk_tensor_uid());
    const auto dvIt = tensorMap.find(attrs->dv_tensor_uid());
    if(kIt == tensorMap.end() || vIt == tensorMap.end() || doIt == tensorMap.end()
       || dqIt == tensorMap.end() || dkIt == tensorMap.end() || dvIt == tensorMap.end())
    {
        return false;
    }

    const auto dataTypeId = graphDataTypeId(*qIt->second,
                                            *kIt->second,
                                            *vIt->second,
                                            *doIt->second,
                                            *dqIt->second,
                                            *dkIt->second,
                                            *dvIt->second);
    const auto mask = graphMask(*attrs);
    if(!dataTypeId.has_value() || !mask.has_value())
    {
        return false;
    }

    return kernel.getStringMetadata(std::string(DTYPE_FIELD)) == *dataTypeId
           && kernel.getIntMetadata(std::string(HDIM_FIELD)) == qIt->second->dims()->Get(3)
           && kernel.getIntMetadata(std::string(MASK_FIELD)) == *mask;
}

/// Every surviving kernel is an exact shape match for its accumulator type, so ranking
/// only has to be total. Prefers a32 as the more numerically conservative default when
/// a caller sets no knob.
double asmSdpaBackwardScore(const MatchContext& /*context*/,
                            const BoundTokens& /*bound*/,
                            const KernelDefinition& kernel)
{
    return kernel.getStringMetadata(std::string(ACC_TYPE_FIELD)) == ACC_TYPE_A32 ? 1.0 : 0.0;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The absolute path of a backward code object on @p deviceId. Unlike the forward
/// pack, backward ships one code object set per arch with no MI300/MI308 split
/// (AITER's backward kernels are not die-specific), so this is a plain concatenation.
std::string codeObjectPath(std::string_view relativeFile)
{
    return asm_sdpa_engine::asm_kernels::getAsmKernelPath(std::string(relativeFile));
}

SdpaBwdBinding readBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "asm sdpa backward dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };
    return {read(Q_TOKEN),
            read(K_TOKEN),
            read(V_TOKEN),
            read(O_TOKEN),
            read(DO_TOKEN),
            read(STATS_TOKEN),
            read(DQ_TOKEN),
            read(DK_TOKEN),
            read(DV_TOKEN)};
}

/// Loads one module by (path, symbol), throwing with @p stageName in the message on
/// failure so a load failure names which of the three kernels is missing.
asm_sdpa_engine::CachedModule loadStage(asm_sdpa_engine::SdpaModuleCache& cache,
                                        const char* stageName,
                                        const std::string& coPath,
                                        const std::string& symbol)
{
    auto module = cache.getOrLoad(coPath, symbol.c_str());
    if(!module)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "asm sdpa backward dispatch failed to load "
                                                           + std::string(stageName) + " kernel '"
                                                           + symbol + "' from " + coPath);
    }
    return module;
}

/// Owns the plan the existing engine would have built: three loaded modules (or two,
/// for a16) plus the launch parameters SdpaBwdPlan::execute() derives grid/kernarg
/// state from.
class PreparedAsmSdpaBackward : public PreparedDispatch
{
public:
    explicit PreparedAsmSdpaBackward(std::unique_ptr<asm_sdpa_engine::SdpaBwdPlan> plan)
        : _plan(std::move(plan))
    {
    }

    const asm_sdpa_engine::SdpaBwdPlan& plan() const
    {
        return *_plan;
    }

private:
    std::unique_ptr<asm_sdpa_engine::SdpaBwdPlan> _plan;
};

/**
 * @brief The native dispatch behind this engine's UDD.
 *
 * Resolves the odo and (a32 only) dq_convert companions from the selected dqdkdv
 * kernel's own metadata, loads all three (or two) modules, derives launch parameters,
 * and wraps them in an SdpaBwdPlan -- reusing that class's execute() rather than
 * restating its three-kernel sequencing and kernarg packing here.
 */
class AsmSdpaBackwardDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    /// These kernels are prebuilt code objects; nothing here compiles source.
    bool supportsSourceKind(KernelSourceKind kind) const override
    {
        return kind == KernelSourceKind::HSACO_FILE;
    }

    /// Non-zero: a32 needs the D buffer plus an FP32 dq_acc accumulator; a16 needs
    /// only the D buffer. Delegates to the same sizing helper SdpaBwdPlanBuilder and
    /// SdpaBwdPlan both use, so the three never drift.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel) const override
    {
        const auto binding = readBinding(bound);
        const auto& tensors = context.graph.getTensorMap();
        const auto* q = tensors.at(binding.q);
        const auto accType = kernel.getStringMetadata(std::string(ACC_TYPE_FIELD)) == ACC_TYPE_A32
                                 ? asm_sdpa_engine::AccumulatorType::A32
                                 : asm_sdpa_engine::AccumulatorType::A16;
        return asm_sdpa_engine::sdpaBwdWorkspaceSize(static_cast<size_t>(q->dims()->Get(0)),
                                                     static_cast<size_t>(q->dims()->Get(1)),
                                                     static_cast<size_t>(q->dims()->Get(2)),
                                                     static_cast<size_t>(q->dims()->Get(3)),
                                                     accType);
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = readBinding(bound);
        const auto dtype = kernel.getStringMetadata(std::string(DTYPE_FIELD));
        const auto hdim = static_cast<int>(kernel.getIntMetadata(std::string(HDIM_FIELD)));
        const auto isA32 = kernel.getStringMetadata(std::string(ACC_TYPE_FIELD)) == ACC_TYPE_A32;
        const std::string_view deviceArch = context.deviceProperties.gcnArchName;

        const auto* odoRow = odoCompanion(deviceArch, dtype, hdim);
        if(odoRow == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "asm sdpa backward dispatch found no odo companion for dtype=" + dtype
                    + " hdim=" + std::to_string(hdim));
        }

        auto odoModule = loadStage(
            _moduleCache, "odo", codeObjectPath(odoRow->coName), std::string(odoRow->knlName));
        auto dqdkdvModule = loadStage(_moduleCache,
                                      "dqdkdv",
                                      codeObjectPath(kernel.source.codeObjectFile),
                                      kernel.source.codeObjectSymbol);

        auto params = buildParams(context, binding, kernel, isA32);

        if(isA32)
        {
            const auto* dqConvertRow = dqConvertCompanion(deviceArch, dtype, hdim);
            if(dqConvertRow == nullptr)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "asm sdpa backward dispatch found no dq_convert companion for dtype=" + dtype
                        + " hdim=" + std::to_string(hdim));
            }
            params.dqConvertTiles = asm_sdpa_engine::SdpaBwdParams::KernelTiles{dqConvertRow->ts};
            auto dqConvertModule = loadStage(_moduleCache,
                                             "dq_convert",
                                             codeObjectPath(dqConvertRow->coName),
                                             std::string(dqConvertRow->knlName));
            auto plan = std::make_unique<asm_sdpa_engine::SdpaBwdPlan>(
                std::move(odoModule), std::move(dqdkdvModule), std::move(dqConvertModule), params);
            return std::make_unique<PreparedAsmSdpaBackward>(std::move(plan));
        }

        auto plan = std::make_unique<asm_sdpa_engine::SdpaBwdPlan>(
            std::move(odoModule), std::move(dqdkdvModule), params);
        return std::make_unique<PreparedAsmSdpaBackward>(std::move(plan));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        dynamic_cast<const PreparedAsmSdpaBackward&>(prepared).plan().execute(
            handle, deviceBuffers, numDeviceBuffers, workspace);
    }

private:
    /**
     * @brief Everything SdpaBwdParams needs, read off the graph and the winning
     *        kernel's metadata once at plan build.
     *
     * Mirrors SdpaBwdPlanBuilder::buildPlan()'s dimension/stride/scale extraction; the
     * per-stage tile sizes come from the resolved kernel and its companions rather
     * than a second CSV lookup, and `odoTiles` is fixed (every odo row shares one tile
     * size, see ODO_TABLE_*), matching SdpaBwdPlanBuilder's own resolveStage.
     */
    static asm_sdpa_engine::SdpaBwdParams buildParams(const MatchContext& context,
                                                      const SdpaBwdBinding& binding,
                                                      const KernelDefinition& kernel,
                                                      bool isA32)
    {
        const auto& tensors = context.graph.getTensorMap();
        const auto tensor = [&tensors](int64_t uid) -> const data_objects::TensorAttributes& {
            const auto it = tensors.find(uid);
            if(it == tensors.end() || it->second == nullptr)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "matched sdpa backward graph has no tensor for uid " + std::to_string(uid));
            }
            return *it->second;
        };

        const auto& q = tensor(binding.q);
        const auto& k = tensor(binding.k);
        const auto& v = tensor(binding.v);
        const auto& o = tensor(binding.o);
        const auto& doGrad = tensor(binding.doGrad);
        const auto& stats = tensor(binding.stats);
        const auto& dq = tensor(binding.dq);
        const auto& dk = tensor(binding.dk);
        const auto& dv = tensor(binding.dv);

        // The matcher admitted this graph, so every operand is rank 4 (rank 3 for
        // stats) with strides that fit; an out-of-range read here is not reachable
        // from a matched graph.
        const auto dim = [](const data_objects::TensorAttributes& t, uint32_t axis) {
            return static_cast<unsigned int>(t.dims()->Get(axis));
        };
        const auto stride = [](const data_objects::TensorAttributes& t, uint32_t axis) {
            return static_cast<unsigned int>(t.strides()->Get(axis));
        };

        const auto& attrs
            = context.graph.getNodeWrapper(0).attributesAs<data_objects::SdpaBackwardAttributes>();

        asm_sdpa_engine::SdpaBwdParams params{};
        params.qUid = binding.q;
        params.kUid = binding.k;
        params.vUid = binding.v;
        params.oUid = binding.o;
        params.doUid = binding.doGrad;
        params.statsUid = binding.stats;
        params.dqUid = binding.dq;
        params.dkUid = binding.dk;
        params.dvUid = binding.dv;

        params.batchSize = dim(q, 0);
        params.numHeadsQ = dim(q, 1);
        params.numHeadsKv = dim(k, 1);
        params.seqLenQ = dim(q, 2);
        params.seqLenKv = dim(k, 2);
        params.headDimQk = dim(q, 3);
        params.headDimV = dim(v, 3);

        params.qStrideSeq = stride(q, 2);
        params.qStrideHead = stride(q, 1);
        params.qStrideBatch = stride(q, 0);
        params.kStrideSeq = stride(k, 2);
        params.kStrideHead = stride(k, 1);
        params.kStrideBatch = stride(k, 0);
        params.vStrideSeq = stride(v, 2);
        params.vStrideHead = stride(v, 1);
        params.vStrideBatch = stride(v, 0);
        params.oStrideSeq = stride(o, 2);
        params.oStrideHead = stride(o, 1);
        params.oStrideBatch = stride(o, 0);
        params.doStrideSeq = stride(doGrad, 2);
        params.doStrideHead = stride(doGrad, 1);
        params.doStrideBatch = stride(doGrad, 0);
        params.dqStrideSeq = stride(dq, 2);
        params.dqStrideHead = stride(dq, 1);
        params.dqStrideBatch = stride(dq, 0);
        params.dkStrideSeq = stride(dk, 2);
        params.dkStrideHead = stride(dk, 1);
        params.dkStrideBatch = stride(dk, 0);
        params.dvStrideSeq = stride(dv, 2);
        params.dvStrideHead = stride(dv, 1);
        params.dvStrideBatch = stride(dv, 0);
        params.statsStrideHead = stride(stats, 1);
        params.statsStrideBatch = stride(stats, 0);

        params.odoTiles = asm_sdpa_engine::SdpaBwdParams::KernelTiles{
            static_cast<unsigned int>(128)}; // every odo row shares tile size 128
        params.dqdkdvTiles = asm_sdpa_engine::SdpaBwdParams::KernelTiles{
            static_cast<unsigned int>(kernel.getIntMetadata(std::string(TS_FIELD)))};

        params.accumulatorType
            = isA32 ? asm_sdpa_engine::AccumulatorType::A32 : asm_sdpa_engine::AccumulatorType::A16;

        const auto mask = plan_utils::getMaskType(attrs);
        params.maskOrdinal = static_cast<int32_t>(mask);
        if(mask == plan_utils::MaskType::SLIDING_WINDOW)
        {
            params.windowLeft = attrs.left_bound().has_value()
                                    ? static_cast<int32_t>(attrs.left_bound().value())
                                    : -1;
            params.windowRight = attrs.right_bound().has_value()
                                     ? static_cast<int32_t>(attrs.right_bound().value())
                                     : -1;
            params.topLeftAlignment
                = attrs.diagonal_alignment() != data_objects::DiagonalAlignment::BOTTOM_RIGHT;
        }

        if(attrs.scale_tensor_uid().has_value())
        {
            params.attnScale = hipdnn_plugin_sdk::makeScalarOperand(
                tensors, attrs.scale_tensor_uid().value(), "attn_scale");
        }
        else
        {
            const float scale = attrs.attn_scale_value().value_or(
                1.0F / std::sqrt(static_cast<float>(params.headDimQk)));
            params.attnScale = hipdnn_plugin_sdk::ScalarOperand{
                0, data_objects::DataType::FLOAT, false, hipdnn_plugin_sdk::ScalarValue{scale}};
        }
        return params;
    }

    // hipModuleLoad dominates cost and the set of distinct code objects is bounded by
    // the descriptor count, so modules are loaded once per process, same as forward.
    mutable asm_sdpa_engine::SdpaModuleCache _moduleCache;
};

/// This engine's dispatch handler. Process-lifetime: the registry holds a non-owning
/// pointer while a Container is created and destroyed per handle.
const AsmSdpaBackwardDispatchHandler& asmSdpaBackwardDispatchHandler()
{
    static const AsmSdpaBackwardDispatchHandler s_dispatchHandler;
    return s_dispatchHandler;
}

} // namespace

void registerAsmSdpaBackwardSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &asmSdpaBackwardGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &asmSdpaBackwardKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &asmSdpaBackwardScore);
    scope.add(std::string(DISPATCH_SYMBOL), &asmSdpaBackwardDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
