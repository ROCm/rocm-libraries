// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>

TEST(TestConvolutionDgradAttributes, CreateConvolutionDgradAttributes)
{
    hipdnn_frontend::graph::ConvDgradAttributes convAttributes;

    // Set tensors
    convAttributes.set_dy(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    convAttributes.set_w(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    convAttributes.set_dx(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());

    // Set convolution parameters
    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});
    convAttributes.set_convolution_mode(hipdnn_frontend::ConvolutionMode::CROSS_CORRELATION);

    // Configure dy tensor (gradient of output)
    auto dyTensor = convAttributes.get_dy();
    dyTensor->set_uid(1)
        .set_name("DyTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 64, 32, 32}) // NCHW format
        .set_stride({65536, 1024, 32, 1});

    // Configure weights tensor
    auto wTensor = convAttributes.get_w();
    wTensor->set_uid(2)
        .set_name("WeightsTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({64, 3, 3, 3}) // KCHW format
        .set_stride({27, 9, 3, 1});

    // Configure dx tensor (gradient of input)
    auto dxTensor = convAttributes.get_dx();
    dxTensor->set_uid(3)
        .set_name("DxTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 3, 32, 32}) // NCHW format
        .set_stride({3072, 1024, 32, 1});

    // Verify tensor attributes
    EXPECT_EQ(dyTensor->get_uid(), 1);
    EXPECT_EQ(dyTensor->get_name(), "DyTensor");
    EXPECT_EQ(dyTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(dyTensor->get_dim(), (std::vector<int64_t>{1, 64, 32, 32}));
    EXPECT_EQ(dyTensor->get_stride(), (std::vector<int64_t>{65536, 1024, 32, 1}));

    EXPECT_EQ(wTensor->get_uid(), 2);
    EXPECT_EQ(wTensor->get_name(), "WeightsTensor");
    EXPECT_EQ(wTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(wTensor->get_dim(), (std::vector<int64_t>{64, 3, 3, 3}));
    EXPECT_EQ(wTensor->get_stride(), (std::vector<int64_t>{27, 9, 3, 1}));

    EXPECT_EQ(dxTensor->get_uid(), 3);
    EXPECT_EQ(dxTensor->get_name(), "DxTensor");
    EXPECT_EQ(dxTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(dxTensor->get_dim(), (std::vector<int64_t>{1, 3, 32, 32}));
    EXPECT_EQ(dxTensor->get_stride(), (std::vector<int64_t>{3072, 1024, 32, 1}));

    // Verify convolution parameters
    EXPECT_EQ(convAttributes.get_pre_padding(), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(convAttributes.get_post_padding(), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(convAttributes.get_stride(), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(convAttributes.get_dilation(), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(convAttributes.get_convolution_mode(),
              hipdnn_frontend::ConvolutionMode::CROSS_CORRELATION);
}

TEST(TestConvolutionDgradAttributes, PackAttributes)
{
    hipdnn_frontend::graph::ConvDgradAttributes convAttributes;

    // Set tensors
    auto dyTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dyTensor->set_uid(1);
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    wTensor->set_uid(2);
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dxTensor->set_uid(3);
    convAttributes.set_dx(dxTensor);

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
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::ConvolutionBwdAttributes>(buffer);

    // Verify packed tensor UIDs
    EXPECT_EQ(convAttributesFb->dy_tensor_uid(), 1);
    EXPECT_EQ(convAttributesFb->w_tensor_uid(), 2);
    EXPECT_EQ(convAttributesFb->dx_tensor_uid(), 3);

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

TEST(TestConvolutionDgradAttributes, DefaultValues)
{
    hipdnn_frontend::graph::ConvDgradAttributes convAttributes;

    // Check default convolution mode
    EXPECT_EQ(convAttributes.get_convolution_mode(),
              hipdnn_frontend::ConvolutionMode::CROSS_CORRELATION);

    // Check that parameters are empty by default
    EXPECT_TRUE(convAttributes.get_pre_padding().empty());
    EXPECT_TRUE(convAttributes.get_post_padding().empty());
    EXPECT_TRUE(convAttributes.get_stride().empty());
    EXPECT_TRUE(convAttributes.get_dilation().empty());

    // Check that tensors are null by default
    EXPECT_EQ(convAttributes.get_dy(), nullptr);
    EXPECT_EQ(convAttributes.get_w(), nullptr);
    EXPECT_EQ(convAttributes.get_dx(), nullptr);
}

TEST(TestConvolutionDgradAttributes, SetPrePaddingWithMove)
{
    hipdnn_frontend::graph::ConvDgradAttributes convAttributes;

    std::vector<int64_t> prePaddingMove = {3, 3};
    convAttributes.set_pre_padding(std::move(prePaddingMove));

    EXPECT_EQ(convAttributes.get_pre_padding(), (std::vector<int64_t>{3, 3}));
}

TEST(TestConvolutionDgradAttributes, SetPostPaddingWithMove)
{
    hipdnn_frontend::graph::ConvDgradAttributes convAttributes;

    std::vector<int64_t> postPaddingMove = {4, 4};
    convAttributes.set_post_padding(std::move(postPaddingMove));

    EXPECT_EQ(convAttributes.get_post_padding(), (std::vector<int64_t>{4, 4}));
}

TEST(TestConvolutionDgradAttributes, SetStrideWithMove)
{
    hipdnn_frontend::graph::ConvDgradAttributes convAttributes;

    std::vector<int64_t> strideMove = {2, 2};
    convAttributes.set_stride(std::move(strideMove));

    EXPECT_EQ(convAttributes.get_stride(), (std::vector<int64_t>{2, 2}));
}

TEST(TestConvolutionDgradAttributes, SetDilationWithMove)
{
    hipdnn_frontend::graph::ConvDgradAttributes convAttributes;

    std::vector<int64_t> dilationMove = {2, 2};
    convAttributes.set_dilation(std::move(dilationMove));

    EXPECT_EQ(convAttributes.get_dilation(), (std::vector<int64_t>{2, 2}));
}

TEST(TestConvolutionDgradAttributes, SetPaddingBothPreAndPost)
{
    hipdnn_frontend::graph::ConvDgradAttributes convAttributes;

    std::vector<int64_t> padding = {5, 5};
    convAttributes.set_padding(padding);

    // set_padding should set both pre and post padding to the same value
    EXPECT_EQ(convAttributes.get_pre_padding(), (std::vector<int64_t>{5, 5}));
    EXPECT_EQ(convAttributes.get_post_padding(), (std::vector<int64_t>{5, 5}));
}

TEST(TestConvolutionDgradAttributes, SetDyMove)
{
    hipdnn_frontend::graph::ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dyTensor->set_uid(10)
        .set_name("MovedDyTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
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

TEST(TestConvolutionDgradAttributes, SetWMove)
{
    hipdnn_frontend::graph::ConvDgradAttributes convAttributes;

    auto wTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    wTensor->set_uid(20)
        .set_name("MovedWeightTensor")
        .set_data_type(hipdnn_frontend::DataType::HALF)
        .set_dim({128, 64, 5, 5})
        .set_stride({1600, 25, 5, 1});

    // Store the raw pointer before moving
    auto rawPtr = wTensor.get();

    convAttributes.set_w(std::move(wTensor));

    // After move, wTensor should be nullptr
    EXPECT_EQ(wTensor, nullptr);

    // The moved tensor should be accessible through get_w()
    auto retrievedTensor = convAttributes.get_w();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestConvolutionDgradAttributes, SetDxWithMove)
{
    hipdnn_frontend::graph::ConvDgradAttributes convAttributes;

    auto dxTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dxTensor->set_uid(30)
        .set_name("MovedDxTensor")
        .set_data_type(hipdnn_frontend::DataType::BFLOAT16)
        .set_dim({1, 128, 112, 112})
        .set_stride({1605632, 12544, 112, 1});

    // Store the raw pointer before moving
    auto rawPtr = dxTensor.get();

    convAttributes.set_dx(std::move(dxTensor));

    // After move, dxTensor should be nullptr
    EXPECT_EQ(dxTensor, nullptr);

    // The moved tensor should be accessible through get_dx()
    auto retrievedTensor = convAttributes.get_dx();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestConvolutionDgradAttributes, SetTensorsConstRef)
{
    hipdnn_frontend::graph::ConvDgradAttributes convAttributes;

    // Create tensors
    auto dyTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dyTensor->set_uid(40).set_name("DyConstRef");

    auto wTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    wTensor->set_uid(50).set_name("WeightConstRef");

    auto dxTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dxTensor->set_uid(60).set_name("DxConstRef");

    // Set using const reference (copy)
    convAttributes.set_dy(dyTensor);
    convAttributes.set_w(wTensor);
    convAttributes.set_dx(dxTensor);

    // Original tensors should still be valid
    EXPECT_NE(dyTensor, nullptr);
    EXPECT_NE(wTensor, nullptr);
    EXPECT_NE(dxTensor, nullptr);
}
