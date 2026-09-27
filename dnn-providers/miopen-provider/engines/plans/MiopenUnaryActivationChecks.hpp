// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

namespace miopen_plugin::unary_activation_applicability
{

// Applicability for every unary pointwise activation this provider supports:
// single-node graph, only PointwiseAttributes, fp32 compute type, a supported
// pointwise mode with valid parameters, then the shared IO-tensor check.
bool isSupported(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph);

} // namespace miopen_plugin::unary_activation_applicability
