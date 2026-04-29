// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <unordered_map>

#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/IGraph.hpp>

namespace miopen_plugin
{

namespace pointwise_applicability
{

bool isSupported(const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph);

void checkPointwiseModeSupported(
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& attrs);

void checkPointwiseTensorsSupported(
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& attrs,
    const std::unordered_map<int64_t,
                             const hipdnn_data_sdk::data_objects::TensorAttributes*>& tensorMap);

} // namespace pointwise_applicability

} // namespace miopen_plugin