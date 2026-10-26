// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

namespace hipdnn_integration_tests::test_utils
{

// Re-serializes a graph with its first non-virtual tensor marked ragged; the offset uid
// points at an arbitrary tensor since the gates only check that it is set.
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
