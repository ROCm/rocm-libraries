// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/ResampleFwdAttributes.hpp>

TEST(TestResampleFwdAttributes, CreateResample)
{
    hipdnn_frontend::graph::ResampleFwdAttributes resampleFwdAttributes;

    resampleFwdAttributes.set_x(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    resampleFwdAttributes.set_y(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    resampleFwdAttributes.set_index(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());

    auto xTensor = resampleFwdAttributes.get_x();
    xTensor->set_uid(1)
        .set_name("xTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto yTensor = resampleFwdAttributes.get_y();
    yTensor->set_uid(2)
        .set_name("yTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto indexTensor = resampleFwdAttributes.get_index();
    indexTensor->set_uid(3)
        .set_name("indexTensor")
        .set_data_type(hipdnn_frontend::DataType::INT32)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    EXPECT_EQ(xTensor->get_uid(), 1);
    EXPECT_EQ(xTensor->get_name(), "xTensor");
    EXPECT_EQ(xTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(xTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(xTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(yTensor->get_uid(), 2);
    EXPECT_EQ(yTensor->get_name(), "yTensor");
    EXPECT_EQ(yTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(yTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(yTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(indexTensor->get_uid(), 3);
    EXPECT_EQ(indexTensor->get_name(), "indexTensor");
    EXPECT_EQ(indexTensor->get_data_type(), hipdnn_frontend::DataType::INT32);
    EXPECT_EQ(indexTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(indexTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(TestResampleFwdAttributes, SetXWithMove)
{
    hipdnn_frontend::graph::ResampleFwdAttributes resampleFwdAttributes;

    auto xTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    xTensor->set_uid(1).set_name("xTensor");

    auto rawPtr = xTensor.get();

    resampleFwdAttributes.set_x(std::move(xTensor));

    auto retrieved = resampleFwdAttributes.get_x();
    EXPECT_EQ(retrieved->get_uid(), 1);
    EXPECT_EQ(retrieved->get_name(), "xTensor");

    EXPECT_EQ(xTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestResampleFwdAttributes, SetYWithMove)
{
    hipdnn_frontend::graph::ResampleFwdAttributes resampleFwdAttributes;

    auto yTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    yTensor->set_uid(1).set_name("yTensor");

    auto rawPtr = yTensor.get();

    resampleFwdAttributes.set_y(std::move(yTensor));

    auto retrieved = resampleFwdAttributes.get_y();
    EXPECT_EQ(retrieved->get_uid(), 1);
    EXPECT_EQ(retrieved->get_name(), "yTensor");

    EXPECT_EQ(yTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestResampleFwdAttributes, SetIndexWithMove)
{
    hipdnn_frontend::graph::ResampleFwdAttributes resampleFwdAttributes;

    auto indexTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    indexTensor->set_uid(1).set_name("indexTensor");

    auto rawPtr = indexTensor.get();

    resampleFwdAttributes.set_index(std::move(indexTensor));

    auto retrieved = resampleFwdAttributes.get_index();
    EXPECT_EQ(retrieved->get_uid(), 1);
    EXPECT_EQ(retrieved->get_name(), "indexTensor");

    EXPECT_EQ(indexTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestResampleFwdAttributes, GenerateIndex)
{
    hipdnn_frontend::graph::ResampleFwdAttributes resampleFwdAttributes;

    EXPECT_FALSE(resampleFwdAttributes.get_generate_index().has_value());

    resampleFwdAttributes.set_generate_index(true);
    ASSERT_TRUE(resampleFwdAttributes.get_generate_index().has_value());

    resampleFwdAttributes.set_generate_index(true);
    EXPECT_TRUE(resampleFwdAttributes.get_generate_index().value());

    resampleFwdAttributes.set_generate_index(false);
    EXPECT_FALSE(resampleFwdAttributes.get_generate_index().value());
}

TEST(TestResampleFwdAttributes, Window)
{
    hipdnn_frontend::graph::ResampleFwdAttributes resampleFwdAttributes;
    EXPECT_EQ(resampleFwdAttributes.get_window().size(), 0);

    resampleFwdAttributes.set_window({1, 2, 3, 4});
    EXPECT_EQ(resampleFwdAttributes.get_window(), std::vector<int64_t>({1, 2, 3, 4}));
}

TEST(TestResampleFwdAttributes, Stride)
{
    hipdnn_frontend::graph::ResampleFwdAttributes resampleFwdAttributes;
    EXPECT_EQ(resampleFwdAttributes.get_stride().size(), 0);

    resampleFwdAttributes.set_stride({1, 2, 3, 4});
    EXPECT_EQ(resampleFwdAttributes.get_stride(), std::vector<int64_t>({1, 2, 3, 4}));
}

TEST(TestResampleFwdAttributes, PostPadding)
{
    hipdnn_frontend::graph::ResampleFwdAttributes resampleFwdAttributes;
    EXPECT_EQ(resampleFwdAttributes.get_post_padding().size(), 0);

    resampleFwdAttributes.set_post_padding({1, 2, 3, 4});
    EXPECT_EQ(resampleFwdAttributes.get_post_padding(), std::vector<int64_t>({1, 2, 3, 4}));
}

TEST(TestResampleFwdAttributes, PrePadding)
{
    hipdnn_frontend::graph::ResampleFwdAttributes resampleFwdAttributes;
    EXPECT_EQ(resampleFwdAttributes.get_pre_padding().size(), 0);

    resampleFwdAttributes.set_pre_padding({1, 2, 3, 4});
    EXPECT_EQ(resampleFwdAttributes.get_pre_padding(), std::vector<int64_t>({1, 2, 3, 4}));
}

TEST(TestResampleFwdAttributes, ResampleMode)
{
    hipdnn_frontend::graph::ResampleFwdAttributes resampleFwdAttributes;
    EXPECT_EQ(resampleFwdAttributes.get_resample_mode(), hipdnn_frontend::ResampleMode::NOT_SET);

    resampleFwdAttributes.set_resample_mode(hipdnn_frontend::ResampleMode::MAXPOOL);
    EXPECT_EQ(resampleFwdAttributes.get_resample_mode(), hipdnn_frontend::ResampleMode::MAXPOOL);
}

TEST(TestResampleFwdAttributes, PaddingMode)
{
    hipdnn_frontend::graph::ResampleFwdAttributes resampleFwdAttributes;
    EXPECT_EQ(resampleFwdAttributes.get_padding_mode(), hipdnn_frontend::PaddingMode::NOT_SET);

    resampleFwdAttributes.set_padding_mode(hipdnn_frontend::PaddingMode::ZERO_PAD);
    EXPECT_EQ(resampleFwdAttributes.get_padding_mode(), hipdnn_frontend::PaddingMode::ZERO_PAD);
}
