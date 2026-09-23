// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

namespace hipdnn_test_sdk::utilities
{

class MockGraph : public hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph
{
public:
    MockGraph()
    {
        // Default: a graph with override shapes disabled, so isApplicable()
        // guards checking is_override_shape_enabled() don't require every
        // existing test to separately stub getGraph(). Tests that specifically
        // exercise override-shape behavior override this via ON_CALL/EXPECT_CALL.
        _defaultGraphBuilder = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
            _defaultGraphBuffer, "mock_graph");
        _defaultGraphBuffer.Finish(_defaultGraphBuilder);
        ON_CALL(*this, getGraph())
            .WillByDefault(::testing::ReturnRef(
                *flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Graph>(
                    _defaultGraphBuffer.GetBufferPointer())));
        // Bytes must come from the same buffer getGraph() reads, or a content key built
        // from this mock would not match the graph it reports.
        ON_CALL(*this, bytes())
            .WillByDefault(
                ::testing::Return(hipdnn_flatbuffers_sdk::flatbuffer_utilities::SerializedBlobView{
                    _defaultGraphBuffer.GetBufferPointer(), _defaultGraphBuffer.GetSize()}));
    }

    MOCK_METHOD(const hipdnn_flatbuffers_sdk::data_objects::Graph&,
                getGraph,
                (),
                (const, override));
    MOCK_METHOD(bool, isValid, (), (const, override));
    MOCK_METHOD(uint32_t, nodeCount, (), (const, override));
    MOCK_METHOD(
        bool,
        hasOnlySupportedAttributes,
        (std::set<hipdnn_flatbuffers_sdk::data_objects::NodeAttributes> supportedAttributes),
        (const, override));
    MOCK_METHOD(const hipdnn_flatbuffers_sdk::data_objects::Node&,
                getNode,
                (uint32_t index),
                (const, override));
    MOCK_METHOD(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper&,
                getNodeWrapper,
                (uint32_t index),
                (const, override));
    MOCK_METHOD(
        (const std::unordered_map<int64_t,
                                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&),
        getTensorMap,
        (),
        (const, override));
    MOCK_METHOD(const std::vector<
                    std::unique_ptr<hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper>>&,
                nodeWrappers,
                (),
                (const, override));
    MOCK_METHOD(hipdnn_flatbuffers_sdk::flatbuffer_utilities::SerializedBlobView,
                bytes,
                (),
                (const, override));

    ~MockGraph() override = default;

private:
    flatbuffers::FlatBufferBuilder _defaultGraphBuffer;
    flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Graph> _defaultGraphBuilder;
};

}
