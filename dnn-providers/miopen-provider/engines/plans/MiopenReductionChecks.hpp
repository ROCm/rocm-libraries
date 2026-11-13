// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>

#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "MiopenApi.hpp"

namespace miopen_plugin::reduction_applicability
{

// Maps a hipDNN reduction mode onto MIOpen's legacy reduce op, or returns nullopt when legacy
// has no equivalent (NOT_SET, MUL_NO_ZEROS). Shared by the gate and the plan so the two cannot
// disagree on which modes are served.
std::optional<miopenReduceTensorOp_t>
    toMiopenReduceTensorOp(hipdnn_flatbuffers_sdk::data_objects::ReductionMode mode);

// Bytes from a FLOAT or HALF tensor's first element to one past its last. Each axis term is
// capped at INT32_MAX, so an out-of-range tensor yields a large value instead of wrapping.
int64_t tensorSpanBytes(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensor);

// Throws HipdnnPluginException(BAD_PARAM) naming the first rule a single-node reduction graph
// breaks, if MIOpen's legacy miopenReduceTensor cannot serve it correctly. Needs no MIOpen
// handle.
void checkSupported(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph);

// checkSupported() as a predicate, logging the reason at INFO with prefix
// "Reduction plan builder: " when the graph is declined.
bool isSupported(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph);

} // namespace miopen_plugin::reduction_applicability
