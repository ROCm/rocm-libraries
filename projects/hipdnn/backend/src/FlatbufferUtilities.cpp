// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "FlatbufferUtilities.hpp"
#include "HipdnnException.hpp"
#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/SerializedGraphContainer.hpp>

namespace hipdnn_backend
{
namespace flatbuffer_utilities
{

void convertSerializedGraphToGraph(
    const uint8_t* buffer,
    size_t size,
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::GraphT>& graphOut)
{
    // Peel the graph blob out of the buffer first: this accepts either a bare
    // serialized Graph (passthrough) or an "HDGP" container that embeds the
    // graph. The SDK throws std exceptions on malformed input; translate to the
    // backend status code callers expect.
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::SerializedBlobView view;
    try
    {
        view = hipdnn_flatbuffers_sdk::flatbuffer_utilities::extractGraphBlob(buffer, size);
    }
    catch(const std::exception&)
    {
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              "Invalid buffer: unable to extract graph from serialized data.");
    }

    flatbuffers::Verifier verifier(view.data, view.size);
    if(!verifier.VerifyBuffer<hipdnn_flatbuffers_sdk::data_objects::Graph>())
    {
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              "Invalid buffer: unable to verify the flatbuffer schema.");
    }

    auto graph = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(view.data);
    if(graph == nullptr)
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "Invalid buffer: unable to unpack the flatbuffer schema.");
    }

    graphOut = std::move(graph);
}

bool isGraphAndPlanContainer(const uint8_t* blob, size_t size)
{
    return hipdnn_flatbuffers_sdk::flatbuffer_utilities::isGraphAndPlanContainer(blob, size);
}

const hipdnn_flatbuffers_sdk::data_objects::SerializedGraphAndPlan*
    verifyAndGetGraphAndPlanContainer(const uint8_t* blob, size_t size)
{
    try
    {
        return hipdnn_flatbuffers_sdk::flatbuffer_utilities::verifyGraphAndPlanContainer(blob,
                                                                                         size);
    }
    catch(const std::exception&)
    {
        throw HipdnnException(
            HIPDNN_STATUS_BAD_PARAM,
            "Invalid buffer: unable to verify the serialized graph-and-plan container.");
    }
}

} // namespace flatbuffer_utilities
} // namespace hipdnn_backend
