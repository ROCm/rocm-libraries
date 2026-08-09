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

// log2(e): the kernel computes softmax base-2 (exp2), so it takes
// scale_log2 = attn_scale * log2(e), not the raw scale. See the header.
constexpr float K_LOG2E = 1.4426950408889634F;

// The S_q/S_kv tiling granularity the kernel's unmasked global loads require
// (must match the family.json multiple_of predicate and the grid ceil_div).
constexpr int64_t K_TILE_M = 16;

// hipDNN element dtype -> the provider dtype token kernel authors use in
// family.json constraints. Unsupported dtypes yield nullopt (decode declines).
std::optional<std::string> providerDtype(data_objects::DataType dtype)
{
    switch(dtype)
    {
    case data_objects::DataType::HALF: return std::string("f16");
    case data_objects::DataType::BFLOAT16: return std::string("bf16");
    default: return std::nullopt;
    }
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

// The 15-arg ABI carries only a (token, head) stride pair per tensor -- there is
// no batch-stride argument. The kernel therefore folds batch into the grid's z
// axis assuming batch_stride == seqlen * stride_token. That holds trivially for
// B==1 (z==0, no batch offset); for B>1 we require the packed relation to hold on
// every tensor, else decline (a wrong batch offset would silently corrupt).
bool batchFoldable(const TensorAttributesWrapper& t, int64_t batch, int64_t seqLen)
{
    if(batch == 1)
    {
        return true;
    }
    const auto strides = t.strides();
    return strides.size() == 4 && strides[0] == seqLen * strides[2];
}

} // namespace

std::optional<catalog::ProblemShape> SdpaAdapter::decode(const IGraph& graph) const
{
    // Single SdpaAttributes node only (Tier-D allowlist for the POC).
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

    // Fail closed on every feature the mask=none forward kernel does not handle,
    // mirroring the ASM SdpaFwdPlanBuilder::isApplicable. Any of these means a
    // different kernel, so decline and let another engine serve the graph.
    if(attrs.causal_mask() || attrs.causal_mask_bottom_right() || attrs.alibi_mask()
       || attrs.padding_mask())
    {
        return std::nullopt; // masked attention -- our kernel is mask_mode="none"
    }
    if(attrs.attn_mask_tensor_uid().has_value() || attrs.block_mask_tensor_uid().has_value()
       || attrs.sink_token_tensor_uid().has_value())
    {
        return std::nullopt; // additive/block/sink masking -- unsupported
    }
    if(attrs.dropout_probability().has_value() && attrs.dropout_probability().value() != 0.0F)
    {
        return std::nullopt; // dropout -- unsupported
    }
    if(attrs.dropout_mask_tensor_uid().has_value() || attrs.seed_tensor_uid().has_value()
       || attrs.offset_tensor_uid().has_value())
    {
        return std::nullopt; // dropout plumbing -- unsupported
    }
    if(attrs.page_table_k_tensor_uid().has_value()
       || attrs.page_table_v_tensor_uid().has_value())
    {
        return std::nullopt; // paged-KV -- unsupported
    }
    if(attrs.seq_len_q_tensor_uid().has_value() || attrs.seq_len_kv_tensor_uid().has_value())
    {
        return std::nullopt; // varlen / group batch mode -- unsupported
    }
    if(attrs.generate_stats().value_or(false) || attrs.stats_tensor_uid().has_value())
    {
        return std::nullopt; // LSE / stats output -- unsupported
    }
    if(attrs.descale_q_tensor_uid().has_value() || attrs.descale_k_tensor_uid().has_value()
       || attrs.descale_v_tensor_uid().has_value() || attrs.descale_s_tensor_uid().has_value()
       || attrs.scale_s_tensor_uid().has_value() || attrs.scale_o_tensor_uid().has_value())
    {
        return std::nullopt; // FP8 (de)scaling -- unsupported
    }
    // The scale is baked at plan-build from attn_scale_value (or the 1/sqrt(D)
    // default). A runtime scale tensor would need LaunchAbi support we do not
    // have, so decline rather than launch with a wrong scale.
    if(attrs.scale_tensor_uid().has_value())
    {
        return std::nullopt;
    }

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
        // (Q, K, V, O) must match the single baked D. O must mirror Q's [B,H,Sq,D].
        if(kDims[3] != headDim || vDims[3] != headDim)
        {
            return std::nullopt;
        }
        if(vDims[1] != numHeadsKv || vDims[2] != seqLenKv)
        {
            return std::nullopt;
        }
        if(oDims[0] != batch || oDims[1] != numHeads || oDims[2] != seqLenQ
           || oDims[3] != headDim)
        {
            return std::nullopt;
        }

        // Multi-head attention only: the kernel indexes K/V by the same head as Q
        // and does not broadcast KV heads, so it cannot serve GQA/MQA. Decline in
        // code when the KV head count differs (the family.json H_kv constraint is a
        // data backstop; the header promises this gate, so enforce it here too).
        if(numHeadsKv != numHeads)
        {
            return std::nullopt;
        }

        // A single supported element dtype (f16 or bf16) across Q/K/V/O; the
        // kernel reads and writes one dtype.
        const auto dtype = providerDtype(q.dataType());
        if(!dtype.has_value() || k.dataType() != q.dataType() || v.dataType() != q.dataType()
           || o.dataType() != q.dataType())
        {
            return std::nullopt;
        }

        // Sequence lengths must be tile multiples: the kernel's global loads are
        // unmasked along the S axes. (D is pinned to 64 by the family constraint,
        // which is a multiple of 16, so no separate D check is needed here.)
        if(seqLenQ % K_TILE_M != 0 || seqLenKv % K_TILE_M != 0)
        {
            return std::nullopt;
        }

        // The kernel has no D-stride arg: it assumes each operand's innermost (D)
        // axis is contiguous and addresses within a row by element index. Decline a
        // non-unit last-dim stride (a transposed/strided view) rather than silently
        // mis-address. Also validates rank-4 strides, making the strides()[1..3]
        // accesses in buildBindings/batchFoldable safe once decode accepts.
        const auto qStrides = q.strides();
        const auto kStrides = k.strides();
        const auto vStrides = v.strides();
        const auto oStrides = o.strides();
        if(qStrides.size() != 4 || kStrides.size() != 4 || vStrides.size() != 4
           || oStrides.size() != 4)
        {
            return std::nullopt;
        }
        if(qStrides[3] != 1 || kStrides[3] != 1 || vStrides[3] != 1 || oStrides[3] != 1)
        {
            return std::nullopt;
        }

        // Batch must be foldable into the grid z axis with only a token stride
        // (no batch-stride ABI arg) -- see batchFoldable().
        if(!batchFoldable(q, batch, seqLenQ) || !batchFoldable(o, batch, seqLenQ)
           || !batchFoldable(k, batch, seqLenKv) || !batchFoldable(v, batch, seqLenKv))
        {
            return std::nullopt;
        }

        catalog::ProblemShape shape;
        shape.emplace("dtype", catalog::ShapeValue{*dtype});
        shape.emplace("B", catalog::ShapeValue{batch});
        shape.emplace("H", catalog::ShapeValue{numHeads});
        shape.emplace("H_kv", catalog::ShapeValue{numHeadsKv});
        shape.emplace("S_q", catalog::ShapeValue{seqLenQ});
        shape.emplace("S_kv", catalog::ShapeValue{seqLenKv});
        shape.emplace("D", catalog::ShapeValue{headDim});
        shape.emplace("causal", catalog::ShapeValue{false});
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
    (void)kernel; // sdpa binds a fixed 15-arg ABI regardless of the chosen kernel

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
    // stride is the H-axis stride (index 1). The kernel needs exactly this pair.
    const TensorAttributesWrapper q = tensorFor(attrs.q_tensor_uid());
    const TensorAttributesWrapper k = tensorFor(attrs.k_tensor_uid());
    const TensorAttributesWrapper v = tensorFor(attrs.v_tensor_uid());
    const TensorAttributesWrapper o = tensorFor(attrs.o_tensor_uid());

    const int64_t headDim = problemInt(problem, "D");
    const float scale = attrs.attn_scale_value().value_or(
        1.0F / std::sqrt(static_cast<float>(headDim)));
    const float scaleLog2 = scale * K_LOG2E;

    catalog::LaunchBindings bindings;
    bindings.pointerUids.emplace("Q", attrs.q_tensor_uid());
    bindings.pointerUids.emplace("K", attrs.k_tensor_uid());
    bindings.pointerUids.emplace("V", attrs.v_tensor_uid());
    bindings.pointerUids.emplace("O", attrs.o_tensor_uid());

    bindings.scalars.emplace("scale_log2", catalog::ScalarValue{scaleLog2});
    bindings.scalars.emplace("seqlen_q", catalog::ScalarValue{problemInt(problem, "S_q")});
    bindings.scalars.emplace("seqlen_k", catalog::ScalarValue{problemInt(problem, "S_kv")});

    bindings.scalars.emplace("stride_q_token", catalog::ScalarValue{q.strides()[2]});
    bindings.scalars.emplace("stride_q_head", catalog::ScalarValue{q.strides()[1]});
    bindings.scalars.emplace("stride_k_token", catalog::ScalarValue{k.strides()[2]});
    bindings.scalars.emplace("stride_k_head", catalog::ScalarValue{k.strides()[1]});
    bindings.scalars.emplace("stride_v_token", catalog::ScalarValue{v.strides()[2]});
    bindings.scalars.emplace("stride_v_head", catalog::ScalarValue{v.strides()[1]});
    bindings.scalars.emplace("stride_o_token", catalog::ScalarValue{o.strides()[2]});
    bindings.scalars.emplace("stride_o_head", catalog::ScalarValue{o.strides()[1]});
    return bindings;
}

launch::SymbolTable SdpaAdapter::gridSymbols(const catalog::ProblemShape& problem,
                                             const catalog::KernelEntry& kernel) const
{
    (void)kernel;

    launch::SymbolTable symbols;
    symbols.emplace("S_q", problemInt(problem, "S_q"));
    symbols.emplace("H", problemInt(problem, "H"));
    symbols.emplace("B", problemInt(problem, "B"));
    return symbols;
}

} // namespace aot_catalog_engine::ops
