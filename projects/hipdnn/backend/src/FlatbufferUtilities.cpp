// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "FlatbufferUtilities.hpp"
#include "HipdnnException.hpp"
#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>

namespace hipdnn_backend
{
namespace flatbuffer_utilities
{

void convertSerializedGraphToGraph(
    const uint8_t* buffer,
    size_t size,
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::GraphT>& graphOut)
{
    flatbuffers::Verifier verifier(buffer, size);
    if(!verifier.VerifyBuffer<hipdnn_flatbuffers_sdk::data_objects::Graph>())
    {
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              "Invalid buffer: unable to verify the flatbuffer schema.");
    }

    auto graph = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(buffer);
    if(graph == nullptr)
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "Invalid buffer: unable to unpack the flatbuffer schema.");
    }

    graphOut = std::move(graph);
}

bool isGraphAndPlanContainer(const uint8_t* blob, size_t size)
{
    // SerializedGraphAndPlanBufferHasIdentifier() reads the 4-byte file
    // identifier at bytes [4, 8) without bounds-checking, so the buffer must
    // be at least 8 bytes (sizeof(flatbuffers::uoffset_t) + the 4-byte
    // identifier) before we may inspect it.
    if(blob == nullptr || size < 8)
    {
        return false;
    }

    return hipdnn_flatbuffers_sdk::data_objects::SerializedGraphAndPlanBufferHasIdentifier(blob);
}

const hipdnn_flatbuffers_sdk::data_objects::SerializedGraphAndPlan*
    verifyAndGetGraphAndPlanContainer(const uint8_t* blob, size_t size)
{
    // Defense-in-depth: verifies independently of any prior
    // isGraphAndPlanContainer() check, so this helper is safe to call alone.
    flatbuffers::Verifier verifier(blob, size);
    if(!hipdnn_flatbuffers_sdk::data_objects::VerifySerializedGraphAndPlanBuffer(verifier))
    {
        throw HipdnnException(
            HIPDNN_STATUS_BAD_PARAM,
            "Invalid buffer: unable to verify the serialized graph-and-plan container.");
    }

    const auto* container = hipdnn_flatbuffers_sdk::data_objects::GetSerializedGraphAndPlan(blob);
    if(container == nullptr)
    {
        throw HipdnnException(
            HIPDNN_STATUS_INTERNAL_ERROR,
            "Invalid buffer: unable to read the serialized graph-and-plan container root.");
    }

    return container;
}

} // namespace flatbuffer_utilities
} // namespace hipdnn_backend
