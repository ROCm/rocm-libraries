// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <set>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/NodeWrapper.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>

using namespace hipdnn_plugin;
using namespace hipdnn_sdk::data_objects;

TEST(TestNodeWrapper, NullBufferIsInvalid)
{
    EXPECT_THROW(NodeWrapper wrapper(nullptr), HipdnnPluginException);
}

TEST(TestNodeWrapper, EnsureTheNodeIsWrappedCorrectly)
{
    flatbuffers::FlatBufferBuilder builder
        = hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph();
    auto serializedGraph = builder.Release();
    auto shallowGraph
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Graph>(serializedGraph.data());

    NodeWrapper wrapper(shallowGraph->nodes()->Get(0));

    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.attributesType(),
              hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes);
    EXPECT_EQ(wrapper.attributesClassType(),
              typeid(hipdnn_sdk::data_objects::BatchnormInferenceAttributes));
    EXPECT_NO_THROW(wrapper.attributesAs<hipdnn_sdk::data_objects::BatchnormInferenceAttributes>());
    EXPECT_THROW(wrapper.attributesAs<hipdnn_sdk::data_objects::BatchnormBackwardAttributes>(),
                 HipdnnPluginException);
}
