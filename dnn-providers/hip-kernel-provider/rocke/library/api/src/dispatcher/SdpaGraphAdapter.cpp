// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "dispatcher/SdpaGraphAdapter.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/NodeWrapper.hpp>

namespace rocke_client::dispatcher
{

namespace
{

namespace fb = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
using hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes;
using hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;

constexpr int RANK = 4; // [B, H, S, D]

const TensorAttributes*
    findTensor(const std::unordered_map<std::int64_t, const TensorAttributes*>& tensorMap,
               std::int64_t uid)
{
    const auto it = tensorMap.find(uid);
    return it == tensorMap.end() ? nullptr : it->second;
}

bool isRank4(const TensorAttributes* tensor)
{
    return tensor != nullptr && tensor->dims() != nullptr && tensor->dims()->size() == RANK
           && tensor->strides() != nullptr && tensor->strides()->size() == RANK;
}

// Provider-facing dtype spelling for the rocKE FMHA-fwd-MFMA family (matches
// PR #8866 compile_spec.dtype). Only the family's supported float types map; any
// other type returns "" so the caller declines the graph.
const char* providerDtype(DataType type)
{
    switch(type)
    {
    case DataType::HALF:
        return "fp16";
    case DataType::BFLOAT16:
        return "bf16";
    default:
        return "";
    }
}

// Classify the physical layout of a rank-4 [B, H, S, D] tensor from its strides.
// Only the two contiguous packings are recognized; padded/other layouts are
// TensorLayout::OTHER (and therefore decline against a BSHD instance).
//
// TODO(kpack): confirm whether real hipDNN SDPA graphs arrive as BSHD or BHSD
// physical and whether the rocKE FMHA MFMA family requires a transpose; the
// #8866 canonical_layout "BSHD" convention must be reconciled with the graph
// contract before this drives production accept/reject.
TensorLayout inferLayout(const flatbuffers::Vector<std::int64_t>& dims,
                         const flatbuffers::Vector<std::int64_t>& strides)
{
    const std::int64_t h = dims.Get(1);
    const std::int64_t s = dims.Get(2);
    const std::int64_t d = dims.Get(3);

    const std::int64_t sb = strides.Get(0);
    const std::int64_t sh = strides.Get(1);
    const std::int64_t ss = strides.Get(2);
    const std::int64_t sd = strides.Get(3);

    // BHSD contiguous (row-major over [B, H, S, D]).
    if(sd == 1 && ss == d && sh == s * d && sb == h * s * d)
    {
        return TensorLayout::BHSD;
    }
    // BSHD contiguous (physical order B, S, H, D).
    if(sd == 1 && sh == d && ss == h * d && sb == s * h * d)
    {
        return TensorLayout::BSHD;
    }
    return TensorLayout::OTHER;
}

// True when the graph requests any SDPA feature the FMHA-fwd-MFMA family cannot
// serve today AND that has no representation in the selection contract (no
// compile_spec/attribute_constraint field). Such a graph must be declined at
// translate time: capturing it would normalize to a plain-SDPA problem and let
// selection wrongly match a kernel that silently ignores the extra tensors.
bool usesUnsupportedTensor(const SdpaAttributes& a)
{
    return a.attn_mask_tensor_uid().has_value() || a.seq_len_q_tensor_uid().has_value()
           || a.seq_len_kv_tensor_uid().has_value() || a.seed_tensor_uid().has_value()
           || a.offset_tensor_uid().has_value() || a.dropout_mask_tensor_uid().has_value()
           || a.dropout_scale_tensor_uid().has_value() || a.page_table_k_tensor_uid().has_value()
           || a.page_table_v_tensor_uid().has_value() || a.block_mask_tensor_uid().has_value()
           || a.sink_token_tensor_uid().has_value() || a.descale_q_tensor_uid().has_value()
           || a.descale_k_tensor_uid().has_value() || a.descale_v_tensor_uid().has_value()
           || a.descale_s_tensor_uid().has_value() || a.scale_s_tensor_uid().has_value()
           || a.scale_o_tensor_uid().has_value() || a.stats_tensor_uid().has_value()
           || a.max_tensor_uid().has_value() || a.sum_exp_tensor_uid().has_value()
           || a.rng_dump_tensor_uid().has_value() || a.amax_s_tensor_uid().has_value()
           || a.amax_o_tensor_uid().has_value();
}

// Classify the SDPA mask into the provider mask_mode spelling. Faithful local
// copy of asm_sdpa_engine::plan_utils::getMaskType (deprecated causal booleans
// take precedence over the left/right-bound trio; an unset bound is unbounded
// = -1). Kept local rather than a cross-layer include so the isolated rocke
// library keeps no dependency on the asm engine; TestSdpaGraphAdapter pins the
// equivalence. Returns std::nullopt for a contradictory configuration (both
// deprecated causal booleans set) that cannot form a valid problem.
std::optional<std::string> classifyMaskMode(const SdpaAttributes& attrs)
{
    const bool causal = attrs.causal_mask();
    const bool causalBottomRight = attrs.causal_mask_bottom_right();
    if(causal && causalBottomRight)
    {
        return std::nullopt;
    }
    if(causal)
    {
        return "causal_top_left";
    }
    if(causalBottomRight)
    {
        return "causal_bottom_right";
    }
    const auto lb = attrs.left_bound();
    const auto rb = attrs.right_bound();
    const std::int64_t left = lb ? *lb : std::int64_t{-1};
    const std::int64_t right = rb ? *rb : std::int64_t{-1};
    if(left == -1 && right == -1)
    {
        return "none";
    }
    if(left == -1 && right == 0)
    {
        return attrs.diagonal_alignment() == DiagonalAlignment::BOTTOM_RIGHT ? "causal_bottom_right"
                                                                             : "causal_top_left";
    }
    // NOTE: window magnitudes (left/right bound values) are intentionally NOT
    // captured here; only the mode string is. Safe while no sliding_window instance
    // exists (selection declines them). A sliding_window launch path must carry the
    // bound values as launch params before relying on this classification.
    return "sliding_window";
}

} // namespace

std::optional<SdpaProblem> translate(const fb::IGraph& graph)
{
    // Graph-level: execute-time override shapes can diverge from the compile-time
    // dims we match exactly, so an exact-shape instance could be dispatched against
    // different runtime shapes. The family serves fixed prebuilt shapes; decline.
    if(graph.getGraph().is_override_shape_enabled())
    {
        return std::nullopt;
    }

    if(graph.nodeCount() != 1)
    {
        return std::nullopt;
    }

    const fb::INodeWrapper& node = graph.getNodeWrapper(0);
    if(node.attributesType() != NodeAttributes::SdpaAttributes)
    {
        return std::nullopt;
    }
    const auto& attrs = node.attributesAs<SdpaAttributes>();

    // Allowlist gate 1: unsupported feature tensors / outputs with no selection
    // representation (paged KV, additive mask, varlen, dropout machinery, FP8,
    // stats). Declined here so selection can never wrongly match them.
    if(usesUnsupportedTensor(attrs))
    {
        return std::nullopt;
    }
    if(attrs.generate_stats().has_value() && attrs.generate_stats().value())
    {
        return std::nullopt;
    }
    if(attrs.max_seq_len_kv().has_value())
    {
        return std::nullopt;
    }

    const auto& tensorMap = graph.getTensorMap();
    const TensorAttributes* q = findTensor(tensorMap, attrs.q_tensor_uid());
    const TensorAttributes* k = findTensor(tensorMap, attrs.k_tensor_uid());
    const TensorAttributes* v = findTensor(tensorMap, attrs.v_tensor_uid());
    const TensorAttributes* o = findTensor(tensorMap, attrs.o_tensor_uid());

    if(!isRank4(q) || !isRank4(k) || !isRank4(v) || !isRank4(o))
    {
        return std::nullopt;
    }

    // Q/K/V/O must share one supported element type.
    const DataType dtype = q->data_type();
    if(k->data_type() != dtype || v->data_type() != dtype || o->data_type() != dtype)
    {
        return std::nullopt;
    }
    const char* providerType = providerDtype(dtype);
    if(providerType[0] == '\0')
    {
        return std::nullopt; // outside the fp16/bf16 family
    }

    // Accumulation/compute precision. The fmha_fwd_mfma family accumulates in
    // fp32, so the I/O dtype above is the only variable type key. compute_data_type
    // (the node's logical accumulation dtype) must be fp32, declared explicitly;
    // UNSET (unspecified) is not accepted.
    if(node.computeDataType() != DataType::FLOAT)
    {
        return std::nullopt;
    }
    // mma_core_mode is an optional SDPA matrix-core operand-precision override. It
    // has no axis in the AOT selection contract and no consumer in the fmha_fwd_mfma
    // family today, so decline any explicit value; only the unspecified default is
    // served.
    if(attrs.mma_core_mode() != DataType::UNSET)
    {
        return std::nullopt;
    }
    // implementation selects the execution strategy. The family is a fused kernel,
    // so an explicit COMPOSITE (decomposed) request cannot be honored; AUTO and
    // UNIFIED permit fusion.
    if(attrs.implementation() == AttentionImplementation::COMPOSITE)
    {
        return std::nullopt;
    }

    // Head sizes: Q and K share the QK^T contraction dim; the family serves a
    // single head_size, so K's, V's and O's head dims must all equal Q's.
    const std::int64_t headSizeQK = q->dims()->Get(3);
    if(k->dims()->Get(3) != headSizeQK || v->dims()->Get(3) != headSizeQK
       || o->dims()->Get(3) != headSizeQK)
    {
        return std::nullopt;
    }

    // Cross-tensor shape consistency. The captured selection keys come from Q
    // (batch, query heads, seqlen_q) and K (kv heads, seqlen_k); verify V/O/K agree
    // on the dims those keys stand for, so the keys describe the whole problem and
    // not just Q/K. head_size (dim 3) is already equal (checked above).
    const auto* qDims = q->dims();
    const auto* kDims = k->dims();
    const auto* vDims = v->dims();
    const auto* oDims = o->dims();
    if(kDims->Get(0) != qDims->Get(0) || vDims->Get(0) != qDims->Get(0)
       || oDims->Get(0) != qDims->Get(0))
    {
        return std::nullopt; // batch must match across Q/K/V/O
    }
    if(oDims->Get(1) != qDims->Get(1) || oDims->Get(2) != qDims->Get(2))
    {
        return std::nullopt; // O must mirror Q's heads and seqlen_q
    }
    if(vDims->Get(1) != kDims->Get(1) || vDims->Get(2) != kDims->Get(2))
    {
        return std::nullopt; // V must mirror K's kv-heads and seqlen_k
    }

    // Physical layout must be one recognized contiguous packing shared by all four
    // tensors (the family's canonical_layout is a single value). Inferring from Q
    // alone would let a graph with a differently packed K/V/O match by mistake.
    const TensorLayout layout = inferLayout(*q->dims(), *q->strides());
    if(layout == TensorLayout::OTHER || inferLayout(*k->dims(), *k->strides()) != layout
       || inferLayout(*v->dims(), *v->strides()) != layout
       || inferLayout(*o->dims(), *o->strides()) != layout)
    {
        return std::nullopt;
    }

    // Capability gates: attributes the family cannot serve today. (They are also
    // #8866 selection keys, so an accepted problem carries the supported values
    // below and still matches a real catalog instance.)
    if(attrs.alibi_mask() || attrs.padding_mask())
    {
        return std::nullopt;
    }
    if(attrs.dropout_probability().has_value() && attrs.dropout_probability().value() > 0.0F)
    {
        return std::nullopt;
    }
    // An explicit scale (value or tensor) is not the family's baked 1/sqrt(d).
    if(attrs.attn_scale_value().has_value() || attrs.scale_tensor_uid().has_value())
    {
        return std::nullopt;
    }

    std::optional<std::string> maskMode = classifyMaskMode(attrs);
    if(!maskMode.has_value())
    {
        return std::nullopt; // contradictory mask flags
    }

    SdpaProblem problem;
    problem.dtype = providerType;
    problem.layout = layout;
    // dims are [B, H, S, D].
    problem.batch = q->dims()->Get(0);
    problem.numQueryHeads = q->dims()->Get(1);
    problem.seqlenQ = q->dims()->Get(2);
    problem.headSize = headSizeQK;
    problem.numKvHeads = k->dims()->Get(1);
    problem.seqlenK = k->dims()->Get(2);
    problem.maskMode = std::move(*maskMode);
    // Supported values guaranteed by the gates above; kept explicit so
    // SdpaProblem::attributes() still satisfies the #8866 attribute_constraints.
    problem.dropoutProbability = 0.0;
    problem.paddingMask = false;
    problem.alibiMask = false;
    problem.scalePolicy = "default_1_over_sqrt_d";

    // arch is filled by the dispatcher (needs the HIP stream).
    return problem;
}

} // namespace rocke_client::dispatcher
