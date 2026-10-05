// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/ResampleFwdAttributes.hpp>

#include <cstdint>
#include <memory>
#include <vector>

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

// cuDNN's spelling must reach the same field, and must chain like every other
// setter on this class.
TEST(TestResampleFwdAttributes, ResamplingModeIsTheCudnnSpellingOfResampleMode)
{
    hipdnn_frontend::graph::ResampleFwdAttributes resampleFwdAttributes;

    resampleFwdAttributes
        .set_resampling_mode(hipdnn_frontend::ResampleMode::AVGPOOL_INCLUDE_PADDING)
        .set_window({2, 3});

    EXPECT_EQ(resampleFwdAttributes.get_resample_mode(),
              hipdnn_frontend::ResampleMode::AVGPOOL_INCLUDE_PADDING);
    EXPECT_EQ(resampleFwdAttributes.get_window(), (std::vector<int64_t>{2, 3}));
}

TEST(TestResampleFwdAttributes, PaddingMode)
{
    hipdnn_frontend::graph::ResampleFwdAttributes resampleFwdAttributes;
    EXPECT_EQ(resampleFwdAttributes.get_padding_mode(), hipdnn_frontend::PaddingMode::NOT_SET);

    resampleFwdAttributes.set_padding_mode(hipdnn_frontend::PaddingMode::ZERO_PAD);
    EXPECT_EQ(resampleFwdAttributes.get_padding_mode(), hipdnn_frontend::PaddingMode::ZERO_PAD);
}

TEST(TestResampleFwdAttributes, LogicalAndStrictEquality)
{
    hipdnn_frontend::graph::ResampleFwdAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_pre_padding({1, 1});
    attr1.set_post_padding({1, 1});
    attr1.set_stride({2, 2});
    attr1.set_window({3, 3});
    attr1.set_resample_mode(hipdnn_frontend::ResampleMode::MAXPOOL);
    attr1.set_padding_mode(hipdnn_frontend::PaddingMode::ZERO_PAD);
    attr1.set_generate_index(true);

    auto x1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    x1->set_uid(1).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_x(x1);

    auto y1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    y1->set_uid(2).set_name("Y").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_y(y1);

    auto index1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    index1->set_uid(3).set_name("Index").set_data_type(hipdnn_frontend::DataType::INT32);
    attr1.set_index(index1);

    hipdnn_frontend::graph::ResampleFwdAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_pre_padding({1, 1});
    attr2.set_post_padding({1, 1});
    attr2.set_stride({2, 2});
    attr2.set_window({3, 3});
    attr2.set_resample_mode(hipdnn_frontend::ResampleMode::MAXPOOL);
    attr2.set_padding_mode(hipdnn_frontend::PaddingMode::ZERO_PAD);
    attr2.set_generate_index(true);

    auto x2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    x2->set_uid(1).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_x(x2);

    auto y2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    y2->set_uid(2).set_name("Y").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_y(y2);

    auto index2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    index2->set_uid(3).set_name("Index").set_data_type(hipdnn_frontend::DataType::INT32);
    attr2.set_index(index2);

    // Initial check: everything matches exactly
    EXPECT_TRUE(attr1 == attr2);
    EXPECT_FALSE(attr1 != attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));

    // Structural tensor mismatch: different UID/name/type entirely
    auto structuralMismatchX = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    structuralMismatchX->set_uid(99).set_name("MismatchedX");
    attr2.set_x(structuralMismatchX);

    EXPECT_TRUE(attr1 != attr2);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2)); // Structural/type gap implies logical inequality
    attr2.set_x(x2); // Revert

    // pre_padding mismatch: semantic, must fail both checks
    attr2.set_pre_padding({0, 0});
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_pre_padding({1, 1}); // Revert

    // post_padding mismatch: semantic, must fail both checks
    attr2.set_post_padding({0, 0});
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_post_padding({1, 1}); // Revert

    // stride mismatch: semantic, must fail both checks
    attr2.set_stride({1, 1});
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_stride({2, 2}); // Revert

    // window mismatch: semantic, must fail both checks
    attr2.set_window({2, 2});
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_window({3, 3}); // Revert

    // resample_mode mismatch: semantic, must fail both checks
    attr2.set_resample_mode(hipdnn_frontend::ResampleMode::AVGPOOL_EXCLUDE_PADDING);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_resample_mode(hipdnn_frontend::ResampleMode::MAXPOOL); // Revert

    // padding_mode mismatch: semantic, must fail both checks
    attr2.set_padding_mode(hipdnn_frontend::PaddingMode::EDGE_VAL_PAD);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_padding_mode(hipdnn_frontend::PaddingMode::ZERO_PAD); // Revert

    // generate_index mismatch: semantic, must fail both checks
    attr2.set_generate_index(false);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_generate_index(true); // Revert

    // Unset-vs-unset generate_index: two attrs that never called
    // set_generate_index should still compare equal
    hipdnn_frontend::graph::ResampleFwdAttributes sparse1;
    sparse1.set_resample_mode(hipdnn_frontend::ResampleMode::MAXPOOL);
    hipdnn_frontend::graph::ResampleFwdAttributes sparse2;
    sparse2.set_resample_mode(hipdnn_frontend::ResampleMode::MAXPOOL);
    EXPECT_TRUE(sparse1 == sparse2);
    EXPECT_TRUE(sparse1.logicallyEquals(sparse2));

    // Set-vs-unset generate_index should differ
    sparse2.set_generate_index(true);
    EXPECT_FALSE(sparse1 == sparse2);
    EXPECT_FALSE(sparse1.logicallyEquals(sparse2));

    // Change metadata (UID/Name) on a tensor while keeping mathematical layout intact
    auto logicalMatchX = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    logicalMatchX
        ->set_uid(555) // Diverges from attr1's x1 (uid: 1)
        .set_name("DIVERGENT_NAME") // Diverges from attr1's x1 ("X")
        .set_data_type(hipdnn_frontend::DataType::FLOAT); // Layout matches
    attr2.set_x(logicalMatchX);

    // Expecting: strict evaluation fails, but functional logical comparison passes
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));
}
