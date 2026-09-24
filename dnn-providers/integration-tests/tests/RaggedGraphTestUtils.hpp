// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

namespace hipdnn_integration_tests::test_utils
{

// Re-serializes a graph with its first non-virtual tensor marked ragged, so any
// graph builder in these tests can produce a ragged variant of itself without
// growing a parameter. The GPU-side twin of makeGraphWithRaggedTensor() in
// TestCpuReferenceRaggedRejection.cpp.
//
// Both the applicability check and the registration gate key only on
// ragged_offset_tensor_uid being *set*, so no real offset tensor is needed; the uid
// points at another existing tensor to keep the value plausible.
inline flatbuffers::DetachedBuffer markFirstTensorRagged(const void* graphBuffer)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    const Graph* graph = GetGraph(graphBuffer);
    auto graphT = std::unique_ptr<GraphT>(graph->UnPack());

    for(auto& tensor : graphT->tensors)
    {
        if(!tensor->virtual_)
        {
            tensor->ragged_offset_tensor_uid = graphT->tensors.back()->uid;
            break;
        }
    }

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, graphT.get()));
    return builder.Release();
}

} // namespace hipdnn_integration_tests::test_utils
