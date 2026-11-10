// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/FloatOperandTestUtils.hpp>

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
//
// When `scale` is provided, a FLOAT scale operand tensor is added in the storage mode the spec
// asks for and its uid is written into attrs.scale_tensor_uid. Defaults to no scale tensor.
//
// When `maskUid` is provided it is written into attrs.attn_mask_tensor_uid. When `maskDataType`
// is also provided, a mask tensor with that dtype (including DataType::UNSET) and `maskDims` is
// added under `maskUid`. A `maskUid` (or an attrs.attn_mask_tensor_uid) without a `maskDataType`
// leaves the uid dangling, with no tensor behind it. Defaults to no mask.
inline flatbuffers::FlatBufferBuilder createSdpaFwdGraph(
    int64_t qUid,
    int64_t kUid,
    int64_t vUid,
    int64_t oUid,
    const std::vector<int64_t>& qDims,
    const std::vector<int64_t>& kDims,
    const std::vector<int64_t>& vDims,
    const std::vector<int64_t>& oDims,
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
    hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT attrs = {},
    std::optional<int64_t> statsUid = std::nullopt,
    std::optional<hipdnn_test_sdk::utilities::FloatOperandSpec> scale = std::nullopt,
    std::optional<int64_t> maskUid = std::nullopt,
    const std::vector<int64_t>& maskDims = {},
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> maskDataType = std::nullopt)
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
    if(scale.has_value())
    {
        attrs.scale_tensor_uid = scale->uid;
    }
    if(maskUid.has_value())
    {
        attrs.attn_mask_tensor_uid = maskUid;
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

    if(scale.has_value())
    {
        hipdnn_test_sdk::utilities::addFloatOperandTensor(builder, tensors, scale.value(), "Scale");
    }

    std::vector<int64_t> maskStrides;
    if(maskUid.has_value() && maskDataType.has_value())
    {
        maskStrides = generateStrides(maskDims);
        tensors.push_back(CreateTensorAttributesDirect(
            builder, maskUid.value(), "Mask", maskDataType.value(), &maskStrides, &maskDims));
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

} // namespace hipdnn_integration_tests::test_utils
