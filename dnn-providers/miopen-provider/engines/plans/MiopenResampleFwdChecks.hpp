// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "MiopenApi.hpp"

namespace miopen_plugin::resample_fwd_applicability
{

// Everything MiopenResampleFwdPlan needs, resolved from a graph the checks accepted.
//
// The X and Y strides are the canonical packed strides of the one layout both tensors share,
// not the graph's own strides. They address the same elements (only the strides of length-1
// axes can differ), but MIOpen infers a tensor's layout from its strides and picks its native or
// transposing pooling path per tensor, while it sizes the transpose workspace from Y alone.
// Canonical strides make MIOpen classify X and Y identically.
struct ResampleFwdConfig
{
    miopenPoolingMode_t mode{miopenPoolingMax};
    std::vector<int> window;
    std::vector<int> padding; // pre_padding == post_padding, MIOpen takes one symmetric pad
    std::vector<int> stride;

    hipdnn_flatbuffers_sdk::data_objects::DataType dataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};

    int64_t xUid{0};
    std::vector<int64_t> xDims;
    std::vector<int64_t> xStrides;

    int64_t yUid{0};
    std::vector<int64_t> yDims;
    std::vector<int64_t> yStrides;

    // Set when the node has an index tensor. MIOpen writes the indices, packed in NC(D)HW order,
    // to the front of its workspace; the plan copies them into this tensor.
    std::optional<int64_t> indexUid;
    miopenIndexType_t indexType{miopenIndexUint8};
};

// Resolves a single-node ResampleFwd graph to a MIOpen pooling forward configuration. Throws
// HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM) naming the reason when the graph is not
// supported. Does not look at is_override_shape_enabled; the plan builder declines that first.
ResampleFwdConfig resolve(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph);

// Returns whether resolve() accepts the graph, logging the reason at INFO with prefix
// "Resample forward plan builder: " if it does not.
bool isSupported(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph);

} // namespace miopen_plugin::resample_fwd_applicability
