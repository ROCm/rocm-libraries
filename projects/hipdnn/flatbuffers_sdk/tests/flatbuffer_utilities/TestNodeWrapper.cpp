// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <set>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/NodeWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;

TEST(TestNodeWrapper, NullBufferIsInvalid)
{
    EXPECT_THROW(const NodeWrapper wrapper(nullptr), std::invalid_argument);
}

TEST(TestNodeWrapper, EnsureTheNodeIsWrappedCorrectly)
{
    flatbuffers::FlatBufferBuilder builder
        = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto serializedGraph = builder.Release();
    auto shallowGraph
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Graph>(serializedGraph.data());

    const NodeWrapper wrapper(shallowGraph->nodes()->Get(0));

    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.attributesType(),
              hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes);
    EXPECT_NO_THROW(
        wrapper.attributesAs<hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes>());
    EXPECT_THROW(
        wrapper.attributesAs<hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes>(),
        std::invalid_argument);
}
