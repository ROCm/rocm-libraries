// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipdnnStatus.h"
#include <cstddef>
#include <cstdint>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/serialized_graph_and_plan_generated.h>
#include <memory>

namespace hipdnn_backend::flatbuffer_utilities
{
void convertSerializedGraphToGraph(
    const uint8_t* buffer,
    size_t size,
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::GraphT>& graphOut);

// Returns true when @p blob is large enough to hold a container header and
// carries the "HDGP" file identifier. Does not run the full verifier; pair
// with verifyAndGetGraphAndPlanContainer() before reading fields.
bool isGraphAndPlanContainer(const uint8_t* blob, size_t size);

// Runs the FlatBuffers verifier over @p blob and returns the typed root. The
// returned pointer aliases into @p blob and is only valid for the lifetime of
// that buffer; no inner bytes are copied. Throws a HipdnnException on
// verification failure.
const hipdnn_flatbuffers_sdk::data_objects::SerializedGraphAndPlan*
    verifyAndGetGraphAndPlanContainer(const uint8_t* blob, size_t size);
} // namespace hipdnn_backend::flatbuffer_utilities
