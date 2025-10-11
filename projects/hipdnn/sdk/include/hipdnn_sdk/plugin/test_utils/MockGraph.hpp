// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>

namespace hipdnn_plugin
{

class MockGraph : public IGraph
{
public:
    MOCK_METHOD(const hipdnn_sdk::data_objects::Graph&, getGraph, (), (const, override));
    MOCK_METHOD(bool, isValid, (), (const, override));
    MOCK_METHOD(uint, nodeCount, (), (const, override));
    MOCK_METHOD(bool,
                hasOnlySupportedAttributes,
                (std::set<hipdnn_sdk::data_objects::NodeAttributes> supportedAttributes),
                (const, override));
    MOCK_METHOD(const hipdnn_sdk::data_objects::Node&, getNode, (uint index), (const, override));
    MOCK_METHOD(const INodeWrapper&, getNodeWrapper, (uint index), (const, override));
    MOCK_METHOD(
        (const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&),
        getTensorMap,
        (),
        (const, override));
};

}
