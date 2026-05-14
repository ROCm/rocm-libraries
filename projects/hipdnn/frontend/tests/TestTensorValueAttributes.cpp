// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <limits>
#include <vector>

// using namespace hipdnn_frontend::graph;
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;
using namespace hipdnn_data_sdk::types;

constexpr float PI_FLOAT = 3.14159265358979323846f;

TEST(TestTensorValueAttributes, SetGetClearFloat)
{
    hipdnn_frontend::graph::TensorAttributes tensor;
    EXPECT_FALSE(tensor.get_pass_by_value());

    constexpr float TEST_VALUE = PI_FLOAT;
    tensor.set_value(TEST_VALUE);
    EXPECT_TRUE(tensor.get_pass_by_value());

    auto opt = tensor.get_pass_by_value<float>();
    ASSERT_TRUE(opt.has_value());
    EXPECT_FLOAT_EQ(opt.value(), TEST_VALUE);

    tensor.clear_value();
    EXPECT_FALSE(tensor.get_pass_by_value());
    EXPECT_FALSE(tensor.get_pass_by_value<float>().has_value());
}

TEST(TestTensorValueAttributes, ConstructorValues)
{
    constexpr float TEST_VALUE = 42.0f;
    const hipdnn_frontend::graph::TensorAttributes tensor(TEST_VALUE);

    auto opt = tensor.get_pass_by_value<float>();
    ASSERT_TRUE(opt.has_value());
    EXPECT_EQ(opt.value(), TEST_VALUE);
}

TEST(TestTensorValueAttributes, TypeSafety)
{
    hipdnn_frontend::graph::TensorAttributes tensor;
    tensor.set_value(42.0f);

    auto floatOpt = tensor.get_pass_by_value<float>();
    ASSERT_TRUE(floatOpt.has_value());
    EXPECT_FLOAT_EQ(floatOpt.value(), 42.0f);

    EXPECT_FALSE(tensor.get_pass_by_value<half>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<bfloat16>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<uint8_t>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<int32_t>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<int64_t>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<double>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<bool>().has_value());

    tensor.set_value(int32_t{123});

    EXPECT_FALSE(tensor.get_pass_by_value<float>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<int64_t>().has_value());

    auto intOpt = tensor.get_pass_by_value<int32_t>();
    ASSERT_TRUE(intOpt.has_value());
    EXPECT_EQ(intOpt.value(), 123);

    tensor.set_value(int64_t{456789012345LL});

    EXPECT_FALSE(tensor.get_pass_by_value<float>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<int32_t>().has_value());

    auto int64Opt = tensor.get_pass_by_value<int64_t>();
    ASSERT_TRUE(int64Opt.has_value());
    EXPECT_EQ(int64Opt.value(), 456789012345LL);

    tensor.set_value(true);

    EXPECT_FALSE(tensor.get_pass_by_value<float>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<int32_t>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<int64_t>().has_value());

    auto boolOpt = tensor.get_pass_by_value<bool>();
    ASSERT_TRUE(boolOpt.has_value());
    EXPECT_EQ(boolOpt.value(), true);
}
