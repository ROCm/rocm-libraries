// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

namespace hip_kernel_provider::test_utils
{

/// Uids of the graph's non-virtual tensors, in graph order.
inline std::vector<int64_t> nonVirtualTensorUids(const void* graphBuffer)
{
    std::vector<int64_t> uids;
    const auto* graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuffer);
    for(const auto* tensor : *graph->tensors())
    {
        if(!tensor->virtual_())
        {
            uids.push_back(tensor->uid());
        }
    }
    return uids;
}

/// Re-serializes a graph with the tensor @p uid marked ragged. The offset uid names the
/// graph's last tensor: the engines under test only look at whether the field is set.
inline flatbuffers::DetachedBuffer markTensorRagged(const void* graphBuffer, int64_t uid)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    const auto graphT = std::unique_ptr<GraphT>(GetGraph(graphBuffer)->UnPack());
    const int64_t offsetUid = graphT->tensors.back()->uid;
    for(auto& tensor : graphT->tensors)
    {
        if(tensor->uid == uid)
        {
            tensor->ragged_offset_tensor_uid = offsetUid;
        }
    }

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, graphT.get()));
    return builder.Release();
}

/// Asserts that @p isApplicable accepts @p valid, and declines the same graph once any one
/// of its non-virtual tensors is marked ragged. The accepted baseline is what makes the
/// decline mean something: a check that refuses every graph cannot pass.
///
/// @param isApplicable Callable taking `const IGraph&` and returning bool.
template <typename IsApplicableFn>
void expectRaggedGraphsDeclined(const flatbuffers::FlatBufferBuilder& valid,
                                IsApplicableFn&& isApplicable)
{
    {
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper baseline(
            valid.GetBufferPointer(), valid.GetSize());
        EXPECT_TRUE(isApplicable(baseline)) << "the graph without a ragged tensor must be accepted";
    }

    const auto uids = nonVirtualTensorUids(valid.GetBufferPointer());
    ASSERT_FALSE(uids.empty());
    for(const int64_t uid : uids)
    {
        const auto ragged = markTensorRagged(valid.GetBufferPointer(), uid);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(ragged.data(),
                                                                               ragged.size());
        EXPECT_FALSE(isApplicable(graph)) << "ragged tensor uid " << uid << " was accepted";
    }
}

} // namespace hip_kernel_provider::test_utils
