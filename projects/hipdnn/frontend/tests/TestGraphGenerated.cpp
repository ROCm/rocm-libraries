// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>

TEST(TestGraphGenerated, SerializeAndDeserialize)
{
    flatbuffers::FlatBufferBuilder builder;

    auto graph = hipdnn_data_sdk::data_objects::CreateGraphDirect(builder, "Graph");
    builder.Finish(graph);

    auto deserializedGraph
        = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Graph>(builder.GetBufferPointer());
    EXPECT_EQ(deserializedGraph->name()->str(), "Graph");
}
