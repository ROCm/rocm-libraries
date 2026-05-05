// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_flatbuffers_sdk::utilities::extractValueFromTensorValue;

namespace
{

TensorAttributesT makeBoolValueAttr(bool value)
{
    TensorAttributesT attr;
    attr.uid = 1;
    attr.name = "boolean_value";
    attr.data_type = DataType::BOOLEAN;
    attr.dims = {1};
    attr.strides = {1};
    attr.value.Set(BoolValue(value));
    return attr;
}

} // namespace

TEST(TestFlatbufferUtils, ExtractBoolValueAsBoolTrue)
{
    auto attr = makeBoolValueAttr(true);
    EXPECT_TRUE(extractValueFromTensorValue<bool>(attr, "p"));
}

TEST(TestFlatbufferUtils, ExtractBoolValueAsBoolFalse)
{
    auto attr = makeBoolValueAttr(false);
    EXPECT_FALSE(extractValueFromTensorValue<bool>(attr, "p"));
}
