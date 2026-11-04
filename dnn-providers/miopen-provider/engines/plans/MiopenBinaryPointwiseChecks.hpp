// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

namespace miopen_plugin::binary_pointwise_applicability
{

// Returns whether a single-node binary pointwise graph (ADD, SUB, MUL, MAX_OP, MIN_OP) is
// supported by this provider, logging at INFO with prefix "Binary pointwise plan builder: " if
// it is not.
bool isSupported(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph);

} // namespace miopen_plugin::binary_pointwise_applicability
