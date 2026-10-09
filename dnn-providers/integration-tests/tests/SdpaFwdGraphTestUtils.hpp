// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/RaggedSdpaTestUtils.hpp>

namespace hipdnn_integration_tests::test_utils
{

// Use the canonical implementation from data_sdk.
using hipdnn_data_sdk::utilities::generateStrides;

// Creates a minimal single-node SDPA-forward flatbuffer graph with packed Q/K/V/O tensors.
// The q/k/v/o tensor uids are written into `attrs`, so callers only need to set behavior
// fields (attn_scale_value, bounds, causal flags, attn_mask_tensor_uid, or any unsupported-mode
// uid/flag) on `attrs` to exercise the signature-key, applicability, and validation paths.
//
// When `statsUid` is provided, a FLOAT log-sum-exp (LSE) output tensor is added with dims
// [B, H, Sq, 1] (derived from `qDims`) and its uid is written into attrs.stats_tensor_uid,
// exercising the LSE/stats output path. Defaults to no stats output for existing callers.
inline flatbuffers::FlatBufferBuilder
    createSdpaFwdGraph(int64_t qUid,
                       int64_t kUid,
                       int64_t vUid,
                       int64_t oUid,
                       const std::vector<int64_t>& qDims,
                       const std::vector<int64_t>& kDims,
                       const std::vector<int64_t>& vDims,
                       const std::vector<int64_t>& oDims,
                       hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                       hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT attrs = {},
                       std::optional<int64_t> statsUid = std::nullopt)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    attrs.q_tensor_uid = qUid;
    attrs.k_tensor_uid = kUid;
    attrs.v_tensor_uid = vUid;
    attrs.o_tensor_uid = oUid;
    if(statsUid.has_value())
    {
        attrs.stats_tensor_uid = statsUid;
    }

    flatbuffers::FlatBufferBuilder builder;

    const auto qStrides = generateStrides(qDims);
    const auto kStrides = generateStrides(kDims);
    const auto vStrides = generateStrides(vDims);
    const auto oStrides = generateStrides(oDims);

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, qUid, "Q", dataType, &qStrides, &qDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, kUid, "K", dataType, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, vUid, "V", dataType, &vStrides, &vDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, oUid, "O", dataType, &oStrides, &oDims));

    // Stats/LSE output tensor: rank-4 [B, H, Sq, 1] derived from Q dims, FLOAT typed.
    std::vector<int64_t> statsDims;
    std::vector<int64_t> statsStrides;
    if(statsUid.has_value())
    {
        statsDims = {qDims[0], qDims[1], qDims[2], 1};
        statsStrides = generateStrides(statsDims);
        tensors.push_back(CreateTensorAttributesDirect(
            builder, statsUid.value(), "Stats", DataType::FLOAT, &statsStrides, &statsDims));
    }

    auto sdpaAttrs = CreateSdpaAttributes(builder, &attrs);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_fwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttrs.Union()));

    auto graph = CreateGraphDirect(
        builder, "SdpaFwdTestGraph", dataType, dataType, DataType::FLOAT, &tensors, &nodes);

    builder.Finish(graph);
    return builder;
}

// Layout of the optional stats (LSE) output of a ragged SDPA graph.
enum class RaggedStatsLayout
{
    // [B, Sq_max, H, 1], contiguous, no ragged_offset. The frontend produces this when stats
    // strides are unset.
    DENSE,
    // [B, Sq_max, H, 1], packed by token (seq stride = H) with its own ragged_offset.
    PACKED,
};

// How a FLOAT operand (attention scale, fp8 descale) is stored.
enum class OperandStorage
{
    DEVICE, // device pointer in the variant pack
    BAKED, // value lives in the graph; the variant pack may omit it
    RUNTIME_PASS_BY_VALUE, // host pointer in the variant pack
};

// BAKED and RUNTIME_PASS_BY_VALUE operands must be scalars ([1]). DEVICE operands may be larger,
// e.g. per-KV-head descales [B, H_kv, 1, 1].
struct FloatOperandSpec
{
    int64_t uid = 0;
    OperandStorage storage = OperandStorage::DEVICE;
    float bakedValue = 0.0f; // BAKED only
    std::vector<int64_t> dims = {1};
};

// Optional parts of a ragged SDPA graph. Defaults leave each one out.
struct RaggedSdpaFwdGraphOptions
{
    // Node attributes such as bounds and scale. Tensor uids are filled in by the builder.
    hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT attrs{};
    // Output dtype. UNSET means the input dtype. fp8 graphs use bf16 here.
    hipdnn_flatbuffers_sdk::data_objects::DataType oDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    // FLOAT LSE output [B, Sq, H, 1].
    std::optional<int64_t> statsUid;
    RaggedStatsLayout statsLayout = RaggedStatsLayout::DENSE;
    // INT32 ragged_offset uid for the stats tensor. Required for PACKED stats.
    std::optional<int64_t> raggedOffsetStatsUid;
    // fp8 Q/K/V descales.
    std::optional<FloatOperandSpec> descaleQ;
    std::optional<FloatOperandSpec> descaleK;
    std::optional<FloatOperandSpec> descaleV;
    // Attention scale tensor.
    std::optional<FloatOperandSpec> scale;
    // Separate ragged_offset uids for V and O. If absent, V reuses K's and O reuses Q's, which is
    // only valid when per-token widths match (Hv*Dv == Hk*D, H*Dv == H*D).
    std::optional<int64_t> raggedOffsetVUid;
    std::optional<int64_t> raggedOffsetOUid;
    // Token-unit offsets, AITER's form: each ragged tensor's ragged_offset_multiplier is its seq
    // stride (strides[1] = H*D, or H for a packed LSE), so the Q table also serves O and the K
    // table also serves V even when their widths differ. Default: element offsets (multiplier 1).
    bool tokenOffsets = false;
};

// Builds a one-node ragged SDPA forward graph (RFC-0014: packed [B,S,H,D] plus ragged_offset).
// Q/K/V/O are packed by token and each carries a ragged_offset uid, which routes the node to the
// ragged reference. Each ragged_offset aux is INT32 [batch+1,1,1,1].
inline flatbuffers::FlatBufferBuilder
    createRaggedSdpaFwdGraph(int64_t qUid,
                             int64_t kUid,
                             int64_t vUid,
                             int64_t oUid,
                             int64_t raggedOffsetQUid,
                             int64_t raggedOffsetKvUid,
                             int64_t batch,
                             const std::vector<int64_t>& qDims,
                             const std::vector<int64_t>& kDims,
                             const std::vector<int64_t>& vDims,
                             const std::vector<int64_t>& oDims,
                             hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                             const RaggedSdpaFwdGraphOptions& options = {})
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    using hipdnn_test_sdk::utilities::raggedDims;
    using hipdnn_test_sdk::utilities::raggedHeads;
    using hipdnn_test_sdk::utilities::raggedSeqExtent;
    using hipdnn_test_sdk::utilities::raggedStrides;

    const DataType outputDataType
        = (options.oDataType == DataType::UNSET) ? dataType : options.oDataType;

    auto attrs = options.attrs;
    attrs.q_tensor_uid = qUid;
    attrs.k_tensor_uid = kUid;
    attrs.v_tensor_uid = vUid;
    attrs.o_tensor_uid = oUid;
    if(options.statsUid.has_value())
    {
        attrs.stats_tensor_uid = options.statsUid;
    }
    if(options.descaleQ.has_value())
    {
        attrs.descale_q_tensor_uid = options.descaleQ->uid;
    }
    if(options.descaleK.has_value())
    {
        attrs.descale_k_tensor_uid = options.descaleK->uid;
    }
    if(options.descaleV.has_value())
    {
        attrs.descale_v_tensor_uid = options.descaleV->uid;
    }
    if(options.scale.has_value())
    {
        attrs.scale_tensor_uid = options.scale->uid;
    }

    flatbuffers::FlatBufferBuilder builder;

    const auto qStrides = raggedStrides(qDims);
    const auto kStrides = raggedStrides(kDims);
    const auto vStrides = raggedStrides(vDims);
    const auto oStrides = raggedStrides(oDims);
    const auto multiplier = [&](const std::vector<int64_t>& strides) {
        return options.tokenOffsets ? strides[1] : 1;
    };

    const auto raggedOffsetVUid = options.raggedOffsetVUid.value_or(raggedOffsetKvUid);
    const auto raggedOffsetOUid = options.raggedOffsetOUid.value_or(raggedOffsetQUid);

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   qUid,
                                                   "Q",
                                                   dataType,
                                                   &qStrides,
                                                   &qDims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   /*is_runtime_pass_by_value=*/false,
                                                   raggedOffsetQUid,
                                                   /*alignment=*/16,
                                                   multiplier(qStrides)));
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   kUid,
                                                   "K",
                                                   dataType,
                                                   &kStrides,
                                                   &kDims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   /*is_runtime_pass_by_value=*/false,
                                                   raggedOffsetKvUid,
                                                   /*alignment=*/16,
                                                   multiplier(kStrides)));
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   vUid,
                                                   "V",
                                                   dataType,
                                                   &vStrides,
                                                   &vDims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   /*is_runtime_pass_by_value=*/false,
                                                   raggedOffsetVUid,
                                                   /*alignment=*/16,
                                                   multiplier(vStrides)));
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   oUid,
                                                   "O",
                                                   outputDataType,
                                                   &oStrides,
                                                   &oDims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   /*is_runtime_pass_by_value=*/false,
                                                   raggedOffsetOUid,
                                                   /*alignment=*/16,
                                                   multiplier(oStrides)));

    const std::vector<int64_t> offsetDims = {batch + 1, 1, 1, 1};
    const auto offsetStrides = generateStrides(offsetDims);
    tensors.push_back(CreateTensorAttributesDirect(
        builder, raggedOffsetQUid, "RaggedOffsetQ", DataType::INT32, &offsetStrides, &offsetDims));
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   raggedOffsetKvUid,
                                                   "RaggedOffsetKv",
                                                   DataType::INT32,
                                                   &offsetStrides,
                                                   &offsetDims));
    if(options.raggedOffsetVUid.has_value())
    {
        tensors.push_back(CreateTensorAttributesDirect(builder,
                                                       raggedOffsetVUid,
                                                       "RaggedOffsetV",
                                                       DataType::INT32,
                                                       &offsetStrides,
                                                       &offsetDims));
    }
    if(options.raggedOffsetOUid.has_value())
    {
        tensors.push_back(CreateTensorAttributesDirect(builder,
                                                       raggedOffsetOUid,
                                                       "RaggedOffsetO",
                                                       DataType::INT32,
                                                       &offsetStrides,
                                                       &offsetDims));
    }

    if(options.statsUid.has_value())
    {
        const auto statsDims = raggedDims(qDims[0], raggedSeqExtent(qDims), raggedHeads(qDims), 1);
        if(options.statsLayout == RaggedStatsLayout::DENSE)
        {
            const auto statsStrides = generateStrides(statsDims);
            tensors.push_back(CreateTensorAttributesDirect(builder,
                                                           options.statsUid.value(),
                                                           "Stats",
                                                           DataType::FLOAT,
                                                           &statsStrides,
                                                           &statsDims));
        }
        else
        {
            const auto statsStrides = raggedStrides(statsDims);
            tensors.push_back(CreateTensorAttributesDirect(builder,
                                                           options.statsUid.value(),
                                                           "Stats",
                                                           DataType::FLOAT,
                                                           &statsStrides,
                                                           &statsDims,
                                                           /*virtual_=*/false,
                                                           TensorValue::NONE,
                                                           /*value=*/0,
                                                           /*is_runtime_pass_by_value=*/false,
                                                           options.raggedOffsetStatsUid,
                                                           /*alignment=*/16,
                                                           multiplier(statsStrides)));
            tensors.push_back(CreateTensorAttributesDirect(builder,
                                                           options.raggedOffsetStatsUid.value(),
                                                           "RaggedOffsetStats",
                                                           DataType::INT32,
                                                           &offsetStrides,
                                                           &offsetDims));
        }
    }

    // Adds a FLOAT operand in the storage mode its spec asks for.
    const auto addFloatOperand = [&](const FloatOperandSpec& spec, const char* name) {
        const auto strides = generateStrides(spec.dims);
        const bool baked = spec.storage == OperandStorage::BAKED;
        const Float32Value bakedValue(spec.bakedValue);
        const flatbuffers::Offset<void> value
            = baked ? builder.CreateStruct(bakedValue).Union() : flatbuffers::Offset<void>();
        tensors.push_back(CreateTensorAttributesDirect(
            builder,
            spec.uid,
            name,
            DataType::FLOAT,
            &strides,
            &spec.dims,
            /*virtual_=*/false,
            baked ? TensorValue::Float32Value : TensorValue::NONE,
            value,
            /*is_runtime_pass_by_value=*/spec.storage == OperandStorage::RUNTIME_PASS_BY_VALUE));
    };

    for(const auto& [spec, name] : {std::pair{&options.scale, "Scale"},
                                    std::pair{&options.descaleQ, "DescaleQ"},
                                    std::pair{&options.descaleK, "DescaleK"},
                                    std::pair{&options.descaleV, "DescaleV"}})
    {
        if(spec->has_value())
        {
            addFloatOperand(spec->value(), name);
        }
    }

    auto sdpaAttrs = CreateSdpaAttributes(builder, &attrs);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_ragged_fwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttrs.Union()));

    auto graph = CreateGraphDirect(
        builder, "SdpaRaggedFwdTestGraph", dataType, dataType, DataType::FLOAT, &tensors, &nodes);

    builder.Finish(graph);
    return builder;
}

} // namespace hipdnn_integration_tests::test_utils
