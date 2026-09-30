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
// Unpacks @p buffer into a GraphT. Accepts either a bare serialized Graph or an
// "HDGP" SerializedGraphAndPlan container, peeling the embedded graph blob out
// of the container before unpacking. Throws a HipdnnException with
// HIPDNN_STATUS_BAD_PARAM on malformed input.
void convertSerializedGraphToGraph(
    const uint8_t* buffer,
    size_t size,
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::GraphT>& graphOut);

// Returns true when @p blob is large enough to hold a container header and
// carries the "HDGP" file identifier. Does not run the full verifier; pair
// with verifyAndGetGraphAndPlanContainer() before reading fields. Delegates to
// the flatbuffers SDK container utilities.
bool isGraphAndPlanContainer(const uint8_t* blob, size_t size);

// Runs the FlatBuffers verifier over @p blob and returns the typed root. The
// returned pointer aliases into @p blob and is only valid for the lifetime of
// that buffer; no inner bytes are copied. Delegates to the flatbuffers SDK
// container utilities, translating any verification failure into a
// HipdnnException with HIPDNN_STATUS_BAD_PARAM.
const hipdnn_flatbuffers_sdk::data_objects::SerializedGraphAndPlan*
    verifyAndGetGraphAndPlanContainer(const uint8_t* blob, size_t size);
} // namespace hipdnn_backend::flatbuffer_utilities
