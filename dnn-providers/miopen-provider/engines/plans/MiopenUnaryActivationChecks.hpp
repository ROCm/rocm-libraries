// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <functional>
#include <string>
#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

namespace miopen_plugin::unary_activation_applicability
{

using ModeCheckFn
    = std::function<void(const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes&)>;

// Generic IO-tensor validation shared by every unary pointwise activation:
// non-virtual, FLOAT/HALF only, matching dtypes, rank in [1, 4], matching element counts.
// `opName` is used only to make exception/log messages activation-specific
// (e.g. "Relu plan builder: ...").
void checkTensorsSupported(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attrs,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    const std::string& opName);

// Generic applicability skeleton shared by every unary pointwise activation:
// single-node graph, only PointwiseAttributes, fp32 compute type, then the
// op-specific mode check followed by the shared tensor check.
bool isSupported(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                 const std::string& opName,
                 const ModeCheckFn& checkModeSupported);

} // namespace miopen_plugin::unary_activation_applicability
