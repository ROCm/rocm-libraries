// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/NodeWrapper.hpp>

namespace hipdnn_test_sdk::utilities
{

class MockNode : public hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper
{
public:
    MOCK_METHOD(bool, isValid, (), (const, override));
    MOCK_METHOD(const hipdnn_flatbuffers_sdk::data_objects::Node&, node, (), (const, override));

    MOCK_METHOD(const void*, attributes, (), (const, override));
    MOCK_METHOD(hipdnn_flatbuffers_sdk::data_objects::NodeAttributes,
                attributesType,
                (),
                (const, override));
    MOCK_METHOD(std::string, name, (), (const, override));
    MOCK_METHOD(hipdnn_flatbuffers_sdk::data_objects::DataType,
                computeDataType,
                (),
                (const, override));
};

}
