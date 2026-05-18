// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/ResampleBwdAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend::graph;
using hipdnn_frontend::PaddingMode;
using hipdnn_frontend::ResampleMode;

// --- Test suite: TestResampleBwdAttributes ---

TEST(TestResampleBwdAttributes, CreateResampleBwdAttributes)
{
    ResampleBwdAttributes attrs;

    // Set all tensors
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(50);
    attrs.set_dy(dyTensor);
    auto indexTensor = std::make_shared<TensorAttributes>();
    indexTensor->set_uid(52);
    attrs.set_index(indexTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(51);
    attrs.set_dx(dxTensor);

    // Set data fields
    attrs.set_pre_padding({1, 1});
    attrs.set_post_padding({1, 1});
    attrs.set_stride({2, 2});
    attrs.set_window({3, 3});
    attrs.set_resample_mode(ResampleMode::MAXPOOL);
    attrs.set_padding_mode(PaddingMode::ZERO_PAD);

    // Verify tensor getters
    EXPECT_NE(attrs.get_dy(), nullptr);
    EXPECT_EQ(attrs.get_dy()->get_uid(), 50);
    EXPECT_NE(attrs.get_index(), nullptr);
    EXPECT_EQ(attrs.get_index()->get_uid(), 52);
    EXPECT_NE(attrs.get_dx(), nullptr);
    EXPECT_EQ(attrs.get_dx()->get_uid(), 51);

    // Verify data field getters
    EXPECT_EQ(attrs.get_pre_padding(), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(attrs.get_post_padding(), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(attrs.get_stride(), (std::vector<int64_t>{2, 2}));
    EXPECT_EQ(attrs.get_window(), (std::vector<int64_t>{3, 3}));
}

TEST(TestResampleBwdAttributes, DefaultValues)
{
    const ResampleBwdAttributes attrs;

    // Tensors should be null by default
    EXPECT_EQ(attrs.get_dy(), nullptr);
    EXPECT_EQ(attrs.get_index(), nullptr);
    EXPECT_EQ(attrs.get_dx(), nullptr);

    // Vector fields should be empty by default
    EXPECT_TRUE(attrs.get_pre_padding().empty());
    EXPECT_TRUE(attrs.get_post_padding().empty());
    EXPECT_TRUE(attrs.get_stride().empty());
    EXPECT_TRUE(attrs.get_window().empty());
}

TEST(TestResampleBwdAttributes, SetDyMove)
{
    ResampleBwdAttributes attrs;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(50).set_name("MovedDyTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = dyTensor.get();

    attrs.set_dy(std::move(dyTensor));

    // After move, original should be nullptr
    EXPECT_EQ(dyTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_dy();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestResampleBwdAttributes, SetIndexMove)
{
    ResampleBwdAttributes attrs;

    auto indexTensor = std::make_shared<TensorAttributes>();
    indexTensor->set_uid(52)
        .set_name("MovedIndexTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = indexTensor.get();

    attrs.set_index(std::move(indexTensor));

    // After move, original should be nullptr
    EXPECT_EQ(indexTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_index();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestResampleBwdAttributes, SetDxMove)
{
    ResampleBwdAttributes attrs;

    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(51).set_name("MovedDxTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = dxTensor.get();

    attrs.set_dx(std::move(dxTensor));

    // After move, original should be nullptr
    EXPECT_EQ(dxTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_dx();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestResampleBwdAttributes, SetPrePaddingWithMove)
{
    ResampleBwdAttributes attrs;

    std::vector<int64_t> prePaddingMove = {1, 1};
    attrs.set_pre_padding(std::move(prePaddingMove));

    EXPECT_EQ(attrs.get_pre_padding(), (std::vector<int64_t>{1, 1}));
}

TEST(TestResampleBwdAttributes, SetPostPaddingWithMove)
{
    ResampleBwdAttributes attrs;

    std::vector<int64_t> postPaddingMove = {1, 1};
    attrs.set_post_padding(std::move(postPaddingMove));

    EXPECT_EQ(attrs.get_post_padding(), (std::vector<int64_t>{1, 1}));
}

TEST(TestResampleBwdAttributes, SetStrideWithMove)
{
    ResampleBwdAttributes attrs;

    std::vector<int64_t> strideMove = {2, 2};
    attrs.set_stride(std::move(strideMove));

    EXPECT_EQ(attrs.get_stride(), (std::vector<int64_t>{2, 2}));
}

TEST(TestResampleBwdAttributes, SetWindowWithMove)
{
    ResampleBwdAttributes attrs;

    std::vector<int64_t> windowMove = {3, 3};
    attrs.set_window(std::move(windowMove));

    EXPECT_EQ(attrs.get_window(), (std::vector<int64_t>{3, 3}));
}

TEST(TestResampleBwdAttributes, SetTensorsConstRef)
{
    ResampleBwdAttributes attrs;

    // Create tensors
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(50).set_name("DyConstRef");
    auto indexTensor = std::make_shared<TensorAttributes>();
    indexTensor->set_uid(52).set_name("IndexConstRef");
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(51).set_name("DxConstRef");

    // Set using const reference (copy)
    attrs.set_dy(dyTensor);
    attrs.set_index(indexTensor);
    attrs.set_dx(dxTensor);

    // Original tensors should still be valid
    EXPECT_NE(dyTensor, nullptr);
    EXPECT_NE(indexTensor, nullptr);
    EXPECT_NE(dxTensor, nullptr);
}
