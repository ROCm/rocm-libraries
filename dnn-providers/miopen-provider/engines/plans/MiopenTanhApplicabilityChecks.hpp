// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

namespace miopen_plugin
{

namespace tanh_applicability
{

bool isTanhSupported(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph);

void checkTanhModeSupported(const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attrs);

void checkTanhTensorsSupported(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attrs,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

} // namespace tanh_applicability

} // namespace miopen_plugin
