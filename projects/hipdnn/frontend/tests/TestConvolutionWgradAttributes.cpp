// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>

TEST(TestConvolutionWgradAttributes, CreateConvolutionWgradAttributes)
{
    hipdnn_frontend::graph::ConvWgradAttributes convAttributes;

    // Set tensors
    convAttributes.set_x(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    convAttributes.set_dy(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    convAttributes.set_dw(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());

    // Set convolution parameters
    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});
    convAttributes.set_convolution_mode(hipdnn_frontend::ConvolutionMode::CROSS_CORRELATION);

    // Configure x tensor (input)
    auto xTensor = convAttributes.get_x();
    xTensor->set_uid(1)
        .set_name("XTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 3, 32, 32}) // NCHW format
        .set_stride({3072, 1024, 32, 1});

    // Configure dy tensor (gradient of output)
    auto dyTensor = convAttributes.get_dy();
    dyTensor->set_uid(2)
        .set_name("DyTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 64, 32, 32}) // NCHW format
        .set_stride({65536, 1024, 32, 1});

    // Configure dw tensor (gradient of weights)
    auto dwTensor = convAttributes.get_dw();
    dwTensor->set_uid(3)
        .set_name("DwTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({64, 3, 3, 3}) // KCHW format
        .set_stride({27, 9, 3, 1});

    // Verify tensor attributes
    EXPECT_EQ(xTensor->get_uid(), 1);
    EXPECT_EQ(xTensor->get_name(), "XTensor");
    EXPECT_EQ(xTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(xTensor->get_dim(), (std::vector<int64_t>{1, 3, 32, 32}));
    EXPECT_EQ(xTensor->get_stride(), (std::vector<int64_t>{3072, 1024, 32, 1}));

    EXPECT_EQ(dyTensor->get_uid(), 2);
    EXPECT_EQ(dyTensor->get_name(), "DyTensor");
    EXPECT_EQ(dyTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(dyTensor->get_dim(), (std::vector<int64_t>{1, 64, 32, 32}));
    EXPECT_EQ(dyTensor->get_stride(), (std::vector<int64_t>{65536, 1024, 32, 1}));

    EXPECT_EQ(dwTensor->get_uid(), 3);
    EXPECT_EQ(dwTensor->get_name(), "DwTensor");
    EXPECT_EQ(dwTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(dwTensor->get_dim(), (std::vector<int64_t>{64, 3, 3, 3}));
    EXPECT_EQ(dwTensor->get_stride(), (std::vector<int64_t>{27, 9, 3, 1}));

    // Verify convolution parameters
    EXPECT_EQ(convAttributes.get_pre_padding(), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(convAttributes.get_post_padding(), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(convAttributes.get_stride(), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(convAttributes.get_dilation(), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(convAttributes.get_convolution_mode(),
              hipdnn_frontend::ConvolutionMode::CROSS_CORRELATION);
}

TEST(TestConvolutionWgradAttributes, PackAttributes)
{
    hipdnn_frontend::graph::ConvWgradAttributes convAttributes;

    // Set tensors
    auto xTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    xTensor->set_uid(1);
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dyTensor->set_uid(2);
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dwTensor->set_uid(3);
    convAttributes.set_dw(dwTensor);

    // Set convolution parameters
    convAttributes.set_pre_padding({2, 2});
    convAttributes.set_post_padding({2, 2});
    convAttributes.set_stride({2, 2});
    convAttributes.set_dilation({1, 1});
    convAttributes.set_convolution_mode(hipdnn_frontend::ConvolutionMode::CROSS_CORRELATION);

    // Pack attributes
    flatbuffers::FlatBufferBuilder builder;
    auto packedAttributes = convAttributes.pack_attributes(builder);
    builder.Finish(packedAttributes);

    auto buffer = builder.GetBufferPointer();
    auto convAttributesFb
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::ConvolutionWrwAttributes>(buffer);

    // Verify packed tensor UIDs
    EXPECT_EQ(convAttributesFb->x_tensor_uid(), 1);
    EXPECT_EQ(convAttributesFb->dy_tensor_uid(), 2);
    EXPECT_EQ(convAttributesFb->dw_tensor_uid(), 3);

    // Verify packed convolution parameters
    ASSERT_EQ(convAttributesFb->pre_padding()->size(), 2);
    EXPECT_EQ(convAttributesFb->pre_padding()->Get(0), 2);
    EXPECT_EQ(convAttributesFb->pre_padding()->Get(1), 2);

    ASSERT_EQ(convAttributesFb->post_padding()->size(), 2);
    EXPECT_EQ(convAttributesFb->post_padding()->Get(0), 2);
    EXPECT_EQ(convAttributesFb->post_padding()->Get(1), 2);

    ASSERT_EQ(convAttributesFb->stride()->size(), 2);
    EXPECT_EQ(convAttributesFb->stride()->Get(0), 2);
    EXPECT_EQ(convAttributesFb->stride()->Get(1), 2);

    ASSERT_EQ(convAttributesFb->dilation()->size(), 2);
    EXPECT_EQ(convAttributesFb->dilation()->Get(0), 1);
    EXPECT_EQ(convAttributesFb->dilation()->Get(1), 1);

    EXPECT_EQ(convAttributesFb->conv_mode(), hipdnn_sdk::data_objects::ConvMode::CROSS_CORRELATION);
}

TEST(TestConvolutionWgradAttributes, DefaultValues)
{
    hipdnn_frontend::graph::ConvWgradAttributes convAttributes;

    // Check default convolution mode
    EXPECT_EQ(convAttributes.get_convolution_mode(),
              hipdnn_frontend::ConvolutionMode::CROSS_CORRELATION);

    // Check that parameters are empty by default
    EXPECT_TRUE(convAttributes.get_pre_padding().empty());
    EXPECT_TRUE(convAttributes.get_post_padding().empty());
    EXPECT_TRUE(convAttributes.get_stride().empty());
    EXPECT_TRUE(convAttributes.get_dilation().empty());

    // Check that tensors are null by default
    EXPECT_EQ(convAttributes.get_x(), nullptr);
    EXPECT_EQ(convAttributes.get_dy(), nullptr);
    EXPECT_EQ(convAttributes.get_dw(), nullptr);
}

TEST(TestConvolutionWgradAttributes, SetPrePaddingWithMove)
{
    hipdnn_frontend::graph::ConvWgradAttributes convAttributes;

    std::vector<int64_t> prePaddingMove = {3, 3};
    convAttributes.set_pre_padding(std::move(prePaddingMove));

    EXPECT_EQ(convAttributes.get_pre_padding(), (std::vector<int64_t>{3, 3}));
}

TEST(TestConvolutionWgradAttributes, SetPostPaddingWithMove)
{
    hipdnn_frontend::graph::ConvWgradAttributes convAttributes;

    std::vector<int64_t> postPaddingMove = {4, 4};
    convAttributes.set_post_padding(std::move(postPaddingMove));

    EXPECT_EQ(convAttributes.get_post_padding(), (std::vector<int64_t>{4, 4}));
}

TEST(TestConvolutionWgradAttributes, SetStrideWithMove)
{
    hipdnn_frontend::graph::ConvWgradAttributes convAttributes;

    std::vector<int64_t> strideMove = {2, 2};
    convAttributes.set_stride(std::move(strideMove));

    EXPECT_EQ(convAttributes.get_stride(), (std::vector<int64_t>{2, 2}));
}

TEST(TestConvolutionWgradAttributes, SetDilationWithMove)
{
    hipdnn_frontend::graph::ConvWgradAttributes convAttributes;

    std::vector<int64_t> dilationMove = {2, 2};
    convAttributes.set_dilation(std::move(dilationMove));

    EXPECT_EQ(convAttributes.get_dilation(), (std::vector<int64_t>{2, 2}));
}

TEST(TestConvolutionWgradAttributes, SetPaddingBothPreAndPost)
{
    hipdnn_frontend::graph::ConvWgradAttributes convAttributes;

    std::vector<int64_t> padding = {5, 5};
    convAttributes.set_padding(padding);

    // set_padding should set both pre and post padding to the same value
    EXPECT_EQ(convAttributes.get_pre_padding(), (std::vector<int64_t>{5, 5}));
    EXPECT_EQ(convAttributes.get_post_padding(), (std::vector<int64_t>{5, 5}));
}

TEST(TestConvolutionWgradAttributes, SetXMove)
{
    hipdnn_frontend::graph::ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    xTensor->set_uid(10)
        .set_name("MovedXTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 3, 224, 224})
        .set_stride({150528, 50176, 224, 1});

    // Store the raw pointer before moving
    auto rawPtr = xTensor.get();

    convAttributes.set_x(std::move(xTensor));

    // After move, xTensor should be nullptr
    EXPECT_EQ(xTensor, nullptr);

    // The moved tensor should be accessible through get_x()
    auto retrievedTensor = convAttributes.get_x();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestConvolutionWgradAttributes, SetDyMove)
{
    hipdnn_frontend::graph::ConvWgradAttributes convAttributes;

    auto dyTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dyTensor->set_uid(20)
        .set_name("MovedDyTensor")
        .set_data_type(hipdnn_frontend::DataType::HALF)
        .set_dim({1, 64, 224, 224})
        .set_stride({3211264, 50176, 224, 1});

    // Store the raw pointer before moving
    auto rawPtr = dyTensor.get();

    convAttributes.set_dy(std::move(dyTensor));

    // After move, dyTensor should be nullptr
    EXPECT_EQ(dyTensor, nullptr);

    // The moved tensor should be accessible through get_dy()
    auto retrievedTensor = convAttributes.get_dy();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestConvolutionWgradAttributes, SetDwWithMove)
{
    hipdnn_frontend::graph::ConvWgradAttributes convAttributes;

    auto dwTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dwTensor->set_uid(30)
        .set_name("MovedDwTensor")
        .set_data_type(hipdnn_frontend::DataType::BFLOAT16)
        .set_dim({128, 64, 5, 5})
        .set_stride({1600, 25, 5, 1});

    // Store the raw pointer before moving
    auto rawPtr = dwTensor.get();

    convAttributes.set_dw(std::move(dwTensor));

    // After move, dwTensor should be nullptr
    EXPECT_EQ(dwTensor, nullptr);

    // The moved tensor should be accessible through get_dw()
    auto retrievedTensor = convAttributes.get_dw();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestConvolutionWgradAttributes, SetTensorsConstRef)
{
    hipdnn_frontend::graph::ConvWgradAttributes convAttributes;

    // Create tensors
    auto xTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    xTensor->set_uid(40).set_name("XConstRef");

    auto dyTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dyTensor->set_uid(50).set_name("DyConstRef");

    auto dwTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dwTensor->set_uid(60).set_name("DwConstRef");

    // Set using const reference (copy)
    convAttributes.set_x(xTensor);
    convAttributes.set_dy(dyTensor);
    convAttributes.set_dw(dwTensor);

    // Original tensors should still be valid
    EXPECT_NE(xTensor, nullptr);
    EXPECT_NE(dyTensor, nullptr);
    EXPECT_NE(dwTensor, nullptr);
}
