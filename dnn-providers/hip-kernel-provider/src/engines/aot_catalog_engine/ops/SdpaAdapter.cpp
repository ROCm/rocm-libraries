// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ops/SdpaAdapter.hpp"

#include <cmath>
#include <optional>
#include <string>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/TensorAttributesWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "launch/PluginError.hpp"

namespace aot_catalog_engine::ops
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper;

namespace
{

// log2(e): a base-2 (exp2) softmax kernel takes scale_log2 = attn_scale * log2(e),
// not the raw scale. The adapter emits BOTH scale_log2 and scale_raw so a family's
// args_signature can name whichever its kernel expects (see the header).
constexpr float K_LOG2E = 1.4426950408889634F;

// hipDNN element dtype -> the provider dtype token kernel authors use in
// family.json `dtype` constraints. Distinct tokens per fp8 encoding because the
// OCP (E4M3/E5M2) and FNUZ variants are NOT interchangeable across archs -- a
// gfx942/gfx950 kernel constrains dtype to its exact encoding. Unsupported
// dtypes yield nullopt (decode declines -> another engine serves the graph).
std::optional<std::string> providerDtype(data_objects::DataType dtype)
{
    switch(dtype)
    {
    case data_objects::DataType::HALF:
        return std::string("f16");
    case data_objects::DataType::BFLOAT16:
        return std::string("bf16");
    case data_objects::DataType::FP8_E4M3:
        return std::string("f8");
    case data_objects::DataType::FP8_E5M2:
        return std::string("bf8");
    case data_objects::DataType::FP8_E4M3_FNUZ:
        return std::string("f8fnuz");
    case data_objects::DataType::FP8_E5M2_FNUZ:
        return std::string("bf8fnuz");
    default:
        return std::nullopt;
    }
}

// Byte width of one element, used to publish *_bytes stride variants for ABIs
// (e.g. hand-written ASM) that take byte strides instead of element strides.
int64_t elementBytes(data_objects::DataType dtype)
{
    switch(dtype)
    {
    case data_objects::DataType::HALF:
    case data_objects::DataType::BFLOAT16:
        return 2;
    case data_objects::DataType::FP8_E4M3:
    case data_objects::DataType::FP8_E5M2:
    case data_objects::DataType::FP8_E4M3_FNUZ:
    case data_objects::DataType::FP8_E5M2_FNUZ:
        return 1;
    default:
        return 0;
    }
}

bool isFp8Token(const std::string& token)
{
    return token == "f8" || token == "bf8" || token == "f8fnuz" || token == "bf8fnuz";
}

int64_t problemInt(const catalog::ProblemShape& problem, const std::string& key)
{
    auto it = problem.find(key);
    if(it == problem.end() || !std::holds_alternative<int64_t>(it->second))
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         "aot-catalog(sdpa): problem missing integer key '" + key + "'");
    }
    return std::get<int64_t>(it->second);
}

// An ABI with only a (token, head) stride pair per tensor -- no batch-stride arg --
// folds batch into the grid's z axis assuming batch_stride == seqlen * stride_token.
// That holds trivially for B==1; for B>1 it holds only when the tensor is packed on
// the batch axis. `batch_foldable` is published as a fact so a kernel that DOES take
// a batch-stride arg is free to accept the non-packed case, while the gfx1151 kernel
// (no batch-stride arg) constrains batch_foldable==true and declines otherwise.
bool batchFoldable(const TensorAttributesWrapper& t, int64_t batch, int64_t seqLen)
{
    if(batch == 1)
    {
        return true;
    }
    const auto strides = t.strides();
    return strides.size() == 4 && strides[0] == seqLen * strides[2];
}

// A kernel with NO stride arguments at all cannot merely require a contiguous D
// axis -- it dictates the whole layout. rocKE's gfx950 dense attention bakes
// packed BSHD (physical [B, S, H, D]): stride over D is 1, over heads is D, over
// tokens is H*D, over batch is S*H*D. `d_contiguous` + `batch_foldable` do NOT
// imply that: canonical contiguous BHSD also satisfies both at batch == 1, and
// selecting a BSHD kernel for a BHSD graph reads the wrong elements silently.
// Published as a fact so such a kernel can constrain it, exactly as
// `batch_foldable` exists for kernels with no batch-stride argument.
bool bshdPacked(const TensorAttributesWrapper& t, int64_t heads, int64_t seqLen, int64_t headDim)
{
    const auto strides = t.strides();
    if(strides.size() != 4)
    {
        return false;
    }
    return strides[3] == 1 && strides[1] == headDim && strides[2] == heads * headDim
           && strides[0] == seqLen * heads * headDim;
}

// Mask classification, resolved from the full attribute set.
//
// This intentionally DUPLICATES asm_sdpa_engine::plan_utils::getMaskType rather
// than including it. The AOT catalog engine is a throwaway POC and must stay
// fully decoupled from the ASM SDPA engine so it can be removed cleanly later; a
// cross-engine include would leave the ASM engine's header on the AOT engine's
// dependency graph. The precedence rules below must be kept in sync with that
// source of truth by hand.
//
// Two sources can describe the mask: the modern left_bound / right_bound /
// diagonal_alignment trio, and the deprecated causal_mask /
// causal_mask_bottom_right booleans. When a deprecated boolean is set it wins and
// the trio is ignored; otherwise the trio is authoritative. The deprecated
// booleans are mutually exclusive. left_bound / right_bound are Optional; an
// unset bound is treated as unbounded (-1), so a partially specified trio (e.g.
// only right_bound = 0) still derives a mask rather than silently reading as
// NO_MASK. The deprecated booleans default to false with no has_*() accessor, so
// "explicitly false" and "unset" are indistinguishable and both mean "not
// requested".
enum class AotMaskType
{
    NO_MASK,
    TOP_LEFT_CAUSAL,
    BOTTOM_RIGHT_CAUSAL,
    SLIDING_WINDOW
};

AotMaskType resolveMaskType(const data_objects::SdpaAttributes& attrs)
{
    const bool causalDeprecated = attrs.causal_mask();
    const bool bottomRightDeprecated = attrs.causal_mask_bottom_right();

    // The two deprecated booleans are mutually exclusive; a graph setting both is
    // malformed. Throwing here is caught by decode()'s std::exception handler,
    // which declines the graph (another engine / native serves it).
    if(causalDeprecated && bottomRightDeprecated)
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                         "aot-catalog(sdpa): causal_mask and causal_mask_bottom_right are "
                         "mutually exclusive but both are set");
    }

    // Deprecated booleans take precedence: when either is set, defer to it and
    // ignore the modern bounds trio.
    if(causalDeprecated)
    {
        return AotMaskType::TOP_LEFT_CAUSAL;
    }
    if(bottomRightDeprecated)
    {
        return AotMaskType::BOTTOM_RIGHT_CAUSAL;
    }

    // No deprecated boolean set: the modern bounds trio is authoritative. An
    // unset bound means unbounded, represented here as -1.
    const int64_t left = attrs.left_bound().has_value() ? attrs.left_bound().value() : -1;
    const int64_t right = attrs.right_bound().has_value() ? attrs.right_bound().value() : -1;
    if(left == -1 && right == -1) // both unbounded
    {
        return AotMaskType::NO_MASK;
    }
    if(left == -1 && right == 0) // causal: attend up to the diagonal
    {
        return attrs.diagonal_alignment() == data_objects::DiagonalAlignment::BOTTOM_RIGHT
                   ? AotMaskType::BOTTOM_RIGHT_CAUSAL
                   : AotMaskType::TOP_LEFT_CAUSAL;
    }
    return AotMaskType::SLIDING_WINDOW; // anything else is a sliding window
}

} // namespace

std::optional<catalog::ProblemShape> SdpaAdapter::decode(const IGraph& graph) const
{
    // Single SdpaAttributes node only (the op adapter matches one node).
    if(!graph.isValid() || graph.nodeCount() != 1)
    {
        return std::nullopt;
    }
    const auto& node = graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::SdpaAttributes)
    {
        return std::nullopt;
    }
    const auto& attrs = node.attributesAs<data_objects::SdpaAttributes>();

    const auto& tensorMap = graph.getTensorMap();
    auto findTensor = [&](int64_t uid) -> const data_objects::TensorAttributes* {
        auto it = tensorMap.find(uid);
        return it == tensorMap.end() ? nullptr : it->second;
    };

    const auto* qTensor = findTensor(attrs.q_tensor_uid());
    const auto* kTensor = findTensor(attrs.k_tensor_uid());
    const auto* vTensor = findTensor(attrs.v_tensor_uid());
    const auto* oTensor = findTensor(attrs.o_tensor_uid());
    if(qTensor == nullptr || kTensor == nullptr || vTensor == nullptr || oTensor == nullptr)
    {
        return std::nullopt;
    }

    // TensorAttributesWrapper::dims()/strides()/dataType() throw on null
    // dims/strides; a malformed graph should make the engine decline, not throw
    // out of isApplicable, so treat any decode failure as "not applicable".
    try
    {
        const TensorAttributesWrapper q(qTensor);
        const TensorAttributesWrapper k(kTensor);
        const TensorAttributesWrapper v(vTensor);
        const TensorAttributesWrapper o(oTensor);

        // ---- Universal safety gates ----------------------------------------
        // These are correctness invariants no forward SDPA kernel can waive:
        // violating one would make the strides()[1..3]/dims() accesses below (and
        // in buildBindings) unsafe or the decoded shape meaningless. They stay in
        // C++; every *feature* decision, by contrast, is published as a fact and
        // decided by per-kernel family.json constraints.

        // All four operands are [B, H, S, D] (BHSD), rank 4.
        const auto qDims = q.dims();
        const auto kDims = k.dims();
        const auto vDims = v.dims();
        const auto oDims = o.dims();
        if(qDims.size() != 4 || kDims.size() != 4 || vDims.size() != 4 || oDims.size() != 4)
        {
            return std::nullopt;
        }

        const int64_t batch = qDims[0];
        const int64_t numHeads = qDims[1];
        const int64_t seqLenQ = qDims[2];
        const int64_t headDim = qDims[3];

        const int64_t numHeadsKv = kDims[1];
        const int64_t seqLenKv = kDims[2];

        // K/V must agree on head count and KV sequence length, and every head dim
        // (Q, K, V, O) must match the single D. O must mirror Q's [B,H,Sq,D].
        if(kDims[3] != headDim || vDims[3] != headDim)
        {
            return std::nullopt;
        }
        if(vDims[1] != numHeadsKv || vDims[2] != seqLenKv)
        {
            return std::nullopt;
        }
        if(oDims[0] != batch || oDims[1] != numHeads || oDims[2] != seqLenQ || oDims[3] != headDim)
        {
            return std::nullopt;
        }

        // A KV head count that does not evenly divide the Q head count is a
        // malformed GQA grouping (no integer group ratio) -- decline. A valid
        // ratio (incl. 1 == MHA, numHeads == MQA) is published as gqa_ratio and
        // left to per-kernel constraints.
        if(numHeadsKv <= 0 || numHeads % numHeadsKv != 0)
        {
            return std::nullopt;
        }
        const int64_t gqaRatio = numHeads / numHeadsKv;

        // A single supported element dtype across Q/K/V/O; the shape carries one
        // dtype token and a mixed-I/O-dtype kernel is a documented future gate.
        const auto dtype = providerDtype(q.dataType());
        if(!dtype.has_value() || k.dataType() != q.dataType() || v.dataType() != q.dataType()
           || o.dataType() != q.dataType())
        {
            return std::nullopt;
        }

        // Rank-4 strides on every operand -- makes the strides()[0..3] accesses in
        // buildBindings/batchFoldable and the d_contiguous fact below safe.
        const auto qStrides = q.strides();
        const auto kStrides = k.strides();
        const auto vStrides = v.strides();
        const auto oStrides = o.strides();
        if(qStrides.size() != 4 || kStrides.size() != 4 || vStrides.size() != 4
           || oStrides.size() != 4)
        {
            return std::nullopt;
        }

        // ---- Published facts (the capability vocabulary) -------------------
        // Structural facts a kernel opts into via constraints rather than a
        // hard-coded decline. `d_contiguous`: innermost (D) axis is unit-stride on
        // all operands -- a kernel with no D-stride arg constrains this true.
        const bool dContiguous
            = qStrides[3] == 1 && kStrides[3] == 1 && vStrides[3] == 1 && oStrides[3] == 1;
        const bool batchFold = batchFoldable(q, batch, seqLenQ) && batchFoldable(o, batch, seqLenQ)
                               && batchFoldable(k, batch, seqLenKv)
                               && batchFoldable(v, batch, seqLenKv);
        const bool bshdPack = bshdPacked(q, numHeads, seqLenQ, headDim)
                              && bshdPacked(o, numHeads, seqLenQ, headDim)
                              && bshdPacked(k, numHeadsKv, seqLenKv, headDim)
                              && bshdPacked(v, numHeadsKv, seqLenKv, headDim);

        // Masking / bias features (each formerly a hard decline). The mask facts
        // are derived from the resolved mask type so a graph expressing a causal
        // mask via the modern left_bound/right_bound/diagonal_alignment trio is
        // classified identically to one using the deprecated booleans -- reading
        // the deprecated booleans alone would misclassify the trio form as
        // NO_MASK and hand a masked problem to an unmasked kernel.
        const AotMaskType maskType = resolveMaskType(attrs);
        const bool causal = maskType == AotMaskType::TOP_LEFT_CAUSAL;
        const bool causalBottomRight = maskType == AotMaskType::BOTTOM_RIGHT_CAUSAL;
        const bool hasDiagonalBand = maskType == AotMaskType::SLIDING_WINDOW;
        const bool hasAlibi = attrs.alibi_mask();
        const bool hasPaddingMask = attrs.padding_mask();
        const bool hasAttnMask = attrs.attn_mask_tensor_uid().has_value();
        const bool hasBlockMask = attrs.block_mask_tensor_uid().has_value();
        const bool hasSink = attrs.sink_token_tensor_uid().has_value();

        // A non-default mma_core_mode pins the softmax/matmul compute dtype; a
        // kernel baked for the default (UNSET) mode cannot honor it, so publish it
        // as a fact those kernels constrain to false.
        const bool hasMmaCoreMode = attrs.mma_core_mode() != data_objects::DataType::UNSET;

        // Dropout: a nonzero probability or any dropout-plumbing tensor. The
        // dropout_scale tensor is part of that plumbing -- omitting it here let a
        // graph carrying only dropout_scale read as dropout-free.
        const bool hasDropout = (attrs.dropout_probability().has_value()
                                 && attrs.dropout_probability().value() != 0.0F)
                                || attrs.dropout_mask_tensor_uid().has_value()
                                || attrs.dropout_scale_tensor_uid().has_value()
                                || attrs.seed_tensor_uid().has_value()
                                || attrs.offset_tensor_uid().has_value();

        const bool paged = attrs.page_table_k_tensor_uid().has_value()
                           || attrs.page_table_v_tensor_uid().has_value();
        const bool varlen
            = attrs.seq_len_q_tensor_uid().has_value() || attrs.seq_len_kv_tensor_uid().has_value();
        const bool genStats
            = attrs.generate_stats().value_or(false) || attrs.stats_tensor_uid().has_value();

        // FP8 (de)scaling machinery, or an fp8 element dtype.
        const bool fp8 = isFp8Token(*dtype) || attrs.descale_q_tensor_uid().has_value()
                         || attrs.descale_k_tensor_uid().has_value()
                         || attrs.descale_v_tensor_uid().has_value()
                         || attrs.descale_s_tensor_uid().has_value()
                         || attrs.scale_s_tensor_uid().has_value()
                         || attrs.scale_o_tensor_uid().has_value();

        // A runtime scale tensor (vs. the plan-time-baked attn_scale_value).
        const bool runtimeScale = attrs.scale_tensor_uid().has_value();

        catalog::ProblemShape shape;
        shape.emplace("dtype", catalog::ShapeValue{*dtype});
        shape.emplace("B", catalog::ShapeValue{batch});
        shape.emplace("H", catalog::ShapeValue{numHeads});
        shape.emplace("H_kv", catalog::ShapeValue{numHeadsKv});
        shape.emplace("S_q", catalog::ShapeValue{seqLenQ});
        shape.emplace("S_kv", catalog::ShapeValue{seqLenKv});
        shape.emplace("D", catalog::ShapeValue{headDim});
        shape.emplace("gqa_ratio", catalog::ShapeValue{gqaRatio});
        shape.emplace("d_contiguous", catalog::ShapeValue{dContiguous});
        shape.emplace("batch_foldable", catalog::ShapeValue{batchFold});
        shape.emplace("bshd_packed", catalog::ShapeValue{bshdPack});
        shape.emplace("causal", catalog::ShapeValue{causal});
        shape.emplace("causal_bottom_right", catalog::ShapeValue{causalBottomRight});
        shape.emplace("has_diagonal_band", catalog::ShapeValue{hasDiagonalBand});
        shape.emplace("has_mma_core_mode", catalog::ShapeValue{hasMmaCoreMode});
        shape.emplace("has_alibi", catalog::ShapeValue{hasAlibi});
        shape.emplace("has_padding_mask", catalog::ShapeValue{hasPaddingMask});
        shape.emplace("has_attn_mask", catalog::ShapeValue{hasAttnMask});
        shape.emplace("has_block_mask", catalog::ShapeValue{hasBlockMask});
        shape.emplace("has_sink", catalog::ShapeValue{hasSink});
        shape.emplace("has_dropout", catalog::ShapeValue{hasDropout});
        shape.emplace("paged", catalog::ShapeValue{paged});
        shape.emplace("varlen", catalog::ShapeValue{varlen});
        shape.emplace("gen_stats", catalog::ShapeValue{genStats});
        shape.emplace("fp8", catalog::ShapeValue{fp8});
        shape.emplace("runtime_scale", catalog::ShapeValue{runtimeScale});
        return shape;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO("aot-catalog(sdpa): declining graph, decode failed: " << e.what());
        return std::nullopt;
    }
}

catalog::LaunchBindings SdpaAdapter::buildBindings(const IGraph& graph,
                                                   const catalog::ProblemShape& problem,
                                                   const catalog::KernelEntry& kernel) const
{
    // The adapter emits a SUPERSET of named quantities; each family's
    // args_signature selects and orders the subset its kernel takes (launch::
    // bindArgs resolves by name and fails closed on an unemitted name). So a new
    // arch is served as data as long as its ABI is drawn from this vocabulary;
    // one added emission here (reviewed) is the single extension point.
    (void)kernel;

    const auto& node = graph.getNodeWrapper(0);
    const auto& attrs = node.attributesAs<data_objects::SdpaAttributes>();

    const auto& tensorMap = graph.getTensorMap();
    auto tensorFor = [&](int64_t uid) -> TensorAttributesWrapper {
        auto it = tensorMap.find(uid);
        if(it == tensorMap.end())
        {
            throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                             "aot-catalog(sdpa): tensor uid missing at bind time");
        }
        return TensorAttributesWrapper(it->second);
    };

    // BHSD layout: within-batch token stride is the S-axis stride (index 2), head
    // stride is the H-axis stride (index 1), batch stride is the B-axis (index 0).
    const TensorAttributesWrapper q = tensorFor(attrs.q_tensor_uid());
    const TensorAttributesWrapper k = tensorFor(attrs.k_tensor_uid());
    const TensorAttributesWrapper v = tensorFor(attrs.v_tensor_uid());
    const TensorAttributesWrapper o = tensorFor(attrs.o_tensor_uid());

    const int64_t headDim = problemInt(problem, "D");
    const float scale
        = attrs.attn_scale_value().value_or(1.0F / std::sqrt(static_cast<float>(headDim)));
    const float scaleLog2 = scale * K_LOG2E;
    const int64_t elemBytes = elementBytes(q.dataType());

    catalog::LaunchBindings bindings;
    bindings.pointerUids.emplace("Q", attrs.q_tensor_uid());
    bindings.pointerUids.emplace("K", attrs.k_tensor_uid());
    bindings.pointerUids.emplace("V", attrs.v_tensor_uid());
    bindings.pointerUids.emplace("O", attrs.o_tensor_uid());

    // Optional tensor operands: bound by name only when the graph carries them, so
    // a kernel that names one in its args_signature gets it and one that doesn't is
    // unaffected. This covers the full forward feature surface decode() publishes as
    // facts (masks, paged KV, varlen, dropout, fp8 (de)scaling, extra stats/LSE
    // outputs) -- a family constrains itself to a feature via the fact, then names
    // the corresponding pointer(s) here. A missing name still fails closed in
    // launch::bindArgs, so omitting one is safe; adding a new optional operand is
    // one bindOptionalPtr() line (the single, reviewed extension point).
    auto bindOptionalPtr = [&](const char* name, ::flatbuffers::Optional<int64_t> uid) {
        if(uid.has_value())
        {
            bindings.pointerUids.emplace(name, uid.value());
        }
    };

    bindOptionalPtr("attn_mask", attrs.attn_mask_tensor_uid());
    bindOptionalPtr("block_mask", attrs.block_mask_tensor_uid());
    bindOptionalPtr("sink", attrs.sink_token_tensor_uid());

    // Runtime scale operand (vs. the plan-time-baked attn_scale_value / scale_log2).
    bindOptionalPtr("scale_tensor", attrs.scale_tensor_uid());

    // Variable-length sequence tables (pointers), distinct from the seqlen_q/k
    // scalar values emitted below for the fixed-length case.
    bindOptionalPtr("seqlen_q_ptr", attrs.seq_len_q_tensor_uid());
    bindOptionalPtr("seqlen_kv_ptr", attrs.seq_len_kv_tensor_uid());

    // Paged-KV block tables.
    bindOptionalPtr("page_table_k", attrs.page_table_k_tensor_uid());
    bindOptionalPtr("page_table_v", attrs.page_table_v_tensor_uid());

    // Dropout plumbing.
    bindOptionalPtr("dropout_mask", attrs.dropout_mask_tensor_uid());
    bindOptionalPtr("dropout_scale", attrs.dropout_scale_tensor_uid());
    bindOptionalPtr("dropout_seed", attrs.seed_tensor_uid());
    bindOptionalPtr("dropout_offset", attrs.offset_tensor_uid());
    bindOptionalPtr("rng_dump", attrs.rng_dump_tensor_uid());

    // FP8 (de)scaling factors: descale inputs/intermediates, scale outputs, and the
    // output amax accumulators a quantizing kernel writes back.
    bindOptionalPtr("descale_q", attrs.descale_q_tensor_uid());
    bindOptionalPtr("descale_k", attrs.descale_k_tensor_uid());
    bindOptionalPtr("descale_v", attrs.descale_v_tensor_uid());
    bindOptionalPtr("descale_s", attrs.descale_s_tensor_uid());
    bindOptionalPtr("scale_s", attrs.scale_s_tensor_uid());
    bindOptionalPtr("scale_o", attrs.scale_o_tensor_uid());
    bindOptionalPtr("amax_s", attrs.amax_s_tensor_uid());
    bindOptionalPtr("amax_o", attrs.amax_o_tensor_uid());

    // Log-sum-exp / softmax statistics outputs. "stats" and "lse" alias the same
    // combined LSE tensor; max + sum_exp are the split form some kernels emit.
    if(attrs.stats_tensor_uid().has_value())
    {
        bindings.pointerUids.emplace("stats", attrs.stats_tensor_uid().value());
        bindings.pointerUids.emplace("lse", attrs.stats_tensor_uid().value());
    }
    bindOptionalPtr("max", attrs.max_tensor_uid());
    bindOptionalPtr("sum_exp", attrs.sum_exp_tensor_uid());

    // Scale, both forms: base-2 (scale_log2) and raw (scale_raw).
    bindings.scalars.emplace("scale_log2", catalog::ScalarValue{scaleLog2});
    bindings.scalars.emplace("scale_raw", catalog::ScalarValue{scale});

    // Sequence lengths.
    bindings.scalars.emplace("seqlen_q", catalog::ScalarValue{problemInt(problem, "S_q")});
    bindings.scalars.emplace("seqlen_k", catalog::ScalarValue{problemInt(problem, "S_kv")});

    // Per-tensor strides in element units and byte units (token / head / batch).
    auto emitStrides = [&](const std::string& prefix, const TensorAttributesWrapper& t) {
        const int64_t token = t.strides()[2];
        const int64_t head = t.strides()[1];
        const int64_t batch = t.strides()[0];
        bindings.scalars.emplace("stride_" + prefix + "_token", catalog::ScalarValue{token});
        bindings.scalars.emplace("stride_" + prefix + "_head", catalog::ScalarValue{head});
        bindings.scalars.emplace("stride_" + prefix + "_batch", catalog::ScalarValue{batch});
        bindings.scalars.emplace("stride_" + prefix + "_token_bytes",
                                 catalog::ScalarValue{token * elemBytes});
        bindings.scalars.emplace("stride_" + prefix + "_head_bytes",
                                 catalog::ScalarValue{head * elemBytes});
        bindings.scalars.emplace("stride_" + prefix + "_batch_bytes",
                                 catalog::ScalarValue{batch * elemBytes});
    };
    emitStrides("q", q);
    emitStrides("k", k);
    emitStrides("v", v);
    emitStrides("o", o);

    // Dimensions / derived counts as scalar args, for ABIs that take them.
    bindings.scalars.emplace("H", catalog::ScalarValue{problemInt(problem, "H")});
    bindings.scalars.emplace("H_kv", catalog::ScalarValue{problemInt(problem, "H_kv")});
    bindings.scalars.emplace("D", catalog::ScalarValue{headDim});
    bindings.scalars.emplace("B", catalog::ScalarValue{problemInt(problem, "B")});
    bindings.scalars.emplace("gqa_ratio", catalog::ScalarValue{problemInt(problem, "gqa_ratio")});
    return bindings;
}

launch::SymbolTable SdpaAdapter::gridSymbols(const catalog::ProblemShape& problem,
                                             const catalog::KernelEntry& kernel) const
{
    // Superset of grid symbols; a family's grid formula references whichever it
    // needs and extra symbols are harmless. Heuristic grid transforms (axis-swap,
    // mask-halving) are a deferred launch-layer extension point (see the README).
    (void)kernel;

    launch::SymbolTable symbols;
    symbols.emplace("S_q", problemInt(problem, "S_q"));
    symbols.emplace("S_kv", problemInt(problem, "S_kv"));
    symbols.emplace("H", problemInt(problem, "H"));
    symbols.emplace("H_kv", problemInt(problem, "H_kv"));
    symbols.emplace("B", problemInt(problem, "B"));
    symbols.emplace("D", problemInt(problem, "D"));
    symbols.emplace("gqa_ratio", problemInt(problem, "gqa_ratio"));
    return symbols;
}

} // namespace aot_catalog_engine::ops
