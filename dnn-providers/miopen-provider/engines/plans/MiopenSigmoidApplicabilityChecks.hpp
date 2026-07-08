// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

namespace miopen_plugin::sigmoid_applicability
{

bool isSigmoidSupported(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph);

void checkSigmoidModeSupported(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attrs);

} // namespace miopen_plugin::sigmoid_applicability
