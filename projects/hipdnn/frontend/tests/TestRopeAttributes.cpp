// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/RopeAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// --- Test suite: TestRopeAttributes ---

TEST(TestRopeAttributes, CreateRopeAttributes)
{
    RopeAttributes attrs;

    // Set all tensors
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9100);
    attrs.set_x(xTensor);
    auto freqsTensor = std::make_shared<TensorAttributes>();
    freqsTensor->set_uid(9101);
    attrs.set_freqs(freqsTensor);
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(9102);
    attrs.set_y(yTensor);

    // Set data fields
    attrs.set_output_scale(0.5);
    attrs.set_rope_dim(4);
    attrs.set_rope_offset(0);

    // Verify tensor getters
    EXPECT_NE(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_x()->get_uid(), 9100);
    EXPECT_NE(attrs.get_freqs(), nullptr);
    EXPECT_EQ(attrs.get_freqs()->get_uid(), 9101);
    EXPECT_NE(attrs.get_y(), nullptr);
    EXPECT_EQ(attrs.get_y()->get_uid(), 9102);

    // Verify data field getters
    EXPECT_EQ(attrs.get_output_scale(), 0.5);
    EXPECT_EQ(attrs.get_rope_dim(), 4);
    EXPECT_EQ(attrs.get_rope_offset(), 0);
}

TEST(TestRopeAttributes, DefaultValues)
{
    const RopeAttributes attrs;

    // Tensors should be null by default
    EXPECT_EQ(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_freqs(), nullptr);
    EXPECT_EQ(attrs.get_y(), nullptr);

    // Vector fields should be empty by default
    EXPECT_EQ(attrs.get_output_scale(), 1.0f);
    EXPECT_EQ(attrs.get_rope_dim(), 0);
    EXPECT_FALSE(attrs.get_rope_offset().has_value());
}

TEST(TestRopeAttributes, SetXMove)
{
    RopeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9100).set_name("MovedXTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = xTensor.get();

    attrs.set_x(std::move(xTensor));

    // After move, original should be nullptr
    EXPECT_EQ(xTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_x();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestRopeAttributes, SetFreqsMove)
{
    RopeAttributes attrs;

    auto freqsTensor = std::make_shared<TensorAttributes>();
    freqsTensor->set_uid(9101)
        .set_name("MovedFreqsTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = freqsTensor.get();

    attrs.set_freqs(std::move(freqsTensor));

    // After move, original should be nullptr
    EXPECT_EQ(freqsTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_freqs();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestRopeAttributes, SetYMove)
{
    RopeAttributes attrs;

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(9102).set_name("MovedYTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = yTensor.get();

    attrs.set_y(std::move(yTensor));

    // After move, original should be nullptr
    EXPECT_EQ(yTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_y();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestRopeAttributes, SetTensorsConstRef)
{
    RopeAttributes attrs;

    // Create tensors
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9100).set_name("XConstRef");
    auto freqsTensor = std::make_shared<TensorAttributes>();
    freqsTensor->set_uid(9101).set_name("FreqsConstRef");
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(9102).set_name("YConstRef");

    // Set using const reference (copy)
    attrs.set_x(xTensor);
    attrs.set_freqs(freqsTensor);
    attrs.set_y(yTensor);

    // Original tensors should still be valid
    EXPECT_NE(xTensor, nullptr);
    EXPECT_NE(freqsTensor, nullptr);
    EXPECT_NE(yTensor, nullptr);
}

TEST(TestRopeAttributes, LogicalAndStrictEquality)
{
    RopeAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_output_scale(0.5);
    attr1.set_rope_dim(4);
    attr1.set_rope_offset(0);
    auto x1 = std::make_shared<TensorAttributes>();
    x1->set_uid(1).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_x(x1);
    auto freqs1 = std::make_shared<TensorAttributes>();
    freqs1->set_uid(2).set_name("Freqs").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_freqs(freqs1);
    auto y1 = std::make_shared<TensorAttributes>();
    y1->set_uid(3).set_name("Y").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_y(y1);

    RopeAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_output_scale(0.5);
    attr2.set_rope_dim(4);
    attr2.set_rope_offset(0);
    auto x2 = std::make_shared<TensorAttributes>();
    x2->set_uid(1).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_x(x2);
    auto freqs2 = std::make_shared<TensorAttributes>();
    freqs2->set_uid(2).set_name("Freqs").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_freqs(freqs2);
    auto y2 = std::make_shared<TensorAttributes>();
    y2->set_uid(3).set_name("Y").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_y(y2);

    // Initial check: everything matches exactly
    EXPECT_TRUE(attr1 == attr2);
    EXPECT_FALSE(attr1 != attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));

    // Structural tensor mismatch: different UID/name/type entirely
    auto structuralMismatchX = std::make_shared<TensorAttributes>();
    structuralMismatchX->set_uid(9999).set_name("MismatchedX");
    attr2.set_x(structuralMismatchX);

    EXPECT_TRUE(attr1 != attr2);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2)); // Structural/type gap implies logical inequality
    attr2.set_x(x2); // Revert

    // output_scale mismatch: semantic, must fail both checks
    attr2.set_output_scale(1.5);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_output_scale(0.5); // Revert

    // rope_dim mismatch: semantic, must fail both checks
    attr2.set_rope_dim(5);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_rope_dim(4); // Revert

    // rope_offset mismatch: semantic, must fail both checks
    attr2.set_rope_offset(1);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_rope_offset(0); // Revert

    // Change metadata (UID/Name) on a tensor while keeping mathematical layout intact
    auto logicalMatchX = std::make_shared<TensorAttributes>();
    logicalMatchX
        ->set_uid(555) // Diverges from attr1's x1 (uid: 1)
        .set_name("DIVERGENT_NAME") // Diverges from attr1's x1 ("X")
        .set_data_type(hipdnn_frontend::DataType::FLOAT); // Layout matches
    attr2.set_x(logicalMatchX);

    // Expecting: strict evaluation fails, but functional logical comparison passes
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));
}
