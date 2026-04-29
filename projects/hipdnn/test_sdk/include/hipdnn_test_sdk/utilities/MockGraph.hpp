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

    ~MockGraph() override = default;
};

}
