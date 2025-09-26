// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <string>
#include <vector>

namespace hipdnn_sdk::test_utilities
{

inline flatbuffers::FlatBufferBuilder createValidGraph()
{
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    flatbuffers::FlatBufferBuilder builder;
    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::HALF,
                                                      hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder createValidEngineDetails(int64_t engineId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto engineDetailsOffset = hipdnn_sdk::data_objects::CreateEngineDetails(builder, engineId);
    builder.Finish(engineDetailsOffset);
    return builder;
}

}
