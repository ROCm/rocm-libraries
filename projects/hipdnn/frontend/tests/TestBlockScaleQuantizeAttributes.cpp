// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/BlockScaleQuantizeAttributes.hpp>

TEST(TestBlockScaleQuantizeAttributes, CreateBlockScaleQuantizeAttributes)
{
    hipdnn_frontend::graph::BlockScaleQuantizeAttributes attrs;

    attrs.set_x(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    attrs.set_y(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    attrs.set_scale(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    attrs.set_block_size(32);
    attrs.set_axis(1);

    auto inputTensor = attrs.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = attrs.get_y();
    outputTensor->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto scaleTensor = attrs.get_scale();
    scaleTensor->set_uid(3)
        .set_name("ScaleTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});

    EXPECT_EQ(inputTensor->get_uid(), 1);
    EXPECT_EQ(inputTensor->get_name(), "InputTensor");
    EXPECT_EQ(inputTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(inputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(inputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(outputTensor->get_uid(), 2);
    EXPECT_EQ(outputTensor->get_name(), "OutputTensor");

    EXPECT_EQ(scaleTensor->get_uid(), 3);
    EXPECT_EQ(scaleTensor->get_name(), "ScaleTensor");

    EXPECT_EQ(attrs.get_block_size(), std::optional<int32_t>(32));
    EXPECT_EQ(attrs.get_axis(), std::optional<int64_t>(1));
    EXPECT_EQ(attrs.get_transpose(), false);
}

TEST(TestBlockScaleQuantizeAttributes, SetXWithMove)
{
    hipdnn_frontend::graph::BlockScaleQuantizeAttributes attrs;

    auto xTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    xTensor->set_uid(1).set_name("XTensor");

    auto rawPtr = xTensor.get();

    attrs.set_x(std::move(xTensor));

    auto retrieved = attrs.get_x();
    EXPECT_EQ(retrieved->get_uid(), 1);
    EXPECT_EQ(retrieved->get_name(), "XTensor");

    EXPECT_EQ(xTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestBlockScaleQuantizeAttributes, SetYWithMove)
{
    hipdnn_frontend::graph::BlockScaleQuantizeAttributes attrs;

    auto yTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    yTensor->set_uid(2).set_name("YTensor");

    auto rawPtr = yTensor.get();

    attrs.set_y(std::move(yTensor));

    auto retrieved = attrs.get_y();
    EXPECT_EQ(retrieved->get_uid(), 2);
    EXPECT_EQ(retrieved->get_name(), "YTensor");

    EXPECT_EQ(yTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestBlockScaleQuantizeAttributes, SetScaleWithMove)
{
    hipdnn_frontend::graph::BlockScaleQuantizeAttributes attrs;

    auto scaleTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    scaleTensor->set_uid(3).set_name("ScaleTensor");

    auto rawPtr = scaleTensor.get();

    attrs.set_scale(std::move(scaleTensor));

    auto retrieved = attrs.get_scale();
    EXPECT_EQ(retrieved->get_uid(), 3);
    EXPECT_EQ(retrieved->get_name(), "ScaleTensor");

    EXPECT_EQ(scaleTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestBlockScaleQuantizeAttributes, BlockSizeOptional)
{
    hipdnn_frontend::graph::BlockScaleQuantizeAttributes attrs;

    EXPECT_EQ(attrs.get_block_size(), std::nullopt);

    attrs.set_block_size(32);
    EXPECT_EQ(attrs.get_block_size(), std::optional<int32_t>(32));

    attrs.set_block_size(64);
    EXPECT_EQ(attrs.get_block_size(), std::optional<int32_t>(64));
}

TEST(TestBlockScaleQuantizeAttributes, AxisOptional)
{
    hipdnn_frontend::graph::BlockScaleQuantizeAttributes attrs;

    EXPECT_EQ(attrs.get_axis(), std::nullopt);

    attrs.set_axis(1);
    EXPECT_EQ(attrs.get_axis(), std::optional<int64_t>(1));

    attrs.set_axis(0);
    EXPECT_EQ(attrs.get_axis(), std::optional<int64_t>(0));
}

TEST(TestBlockScaleQuantizeAttributes, Transpose)
{
    hipdnn_frontend::graph::BlockScaleQuantizeAttributes attrs;

    EXPECT_EQ(attrs.get_transpose(), false);

    attrs.set_transpose(true);
    EXPECT_EQ(attrs.get_transpose(), true);

    attrs.set_transpose(false);
    EXPECT_EQ(attrs.get_transpose(), false);
}

TEST(TestBlockScaleQuantizeAttributes, TypeAliasWorks)
{
    // Verify the compatibility alias compiles and works
    hipdnn_frontend::graph::Block_scale_quantize_attributes attrs;

    attrs.set_x(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    EXPECT_NE(attrs.get_x(), nullptr);
}

TEST(TestBlockScaleQuantizeAttributes, SimplifiedSetXWithMove)
{
    hipdnn_frontend::graph::BlockScaleQuantizeAttributes attrs;

    auto xTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    attrs.set_x(std::move(xTensor));

    EXPECT_NE(attrs.get_x(), nullptr);
}

TEST(TestBlockScaleQuantizeAttributes, SimplifiedSetYWithMove)
{
    hipdnn_frontend::graph::BlockScaleQuantizeAttributes attrs;

    auto yTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    attrs.set_y(std::move(yTensor));

    EXPECT_NE(attrs.get_y(), nullptr);
}

TEST(TestBlockScaleQuantizeAttributes, SimplifiedSetScaleWithMove)
{
    hipdnn_frontend::graph::BlockScaleQuantizeAttributes attrs;

    auto scaleTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    attrs.set_scale(std::move(scaleTensor));

    EXPECT_NE(attrs.get_scale(), nullptr);
}
