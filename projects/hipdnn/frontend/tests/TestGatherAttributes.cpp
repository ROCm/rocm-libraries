// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/GatherAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// --- Test suite: TestGatherAttributes ---

TEST(TestGatherAttributes, CreateGatherAttributes)
{
    GatherAttributes attrs;

    // Set all tensors
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9400);
    attrs.set_x(xTensor);
    auto indicesTensor = std::make_shared<TensorAttributes>();
    indicesTensor->set_uid(9401);
    attrs.set_indices(indicesTensor);
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(9402);
    attrs.set_y(yTensor);

    // Set data fields
    attrs.set_axis(0);

    // Verify tensor getters
    EXPECT_NE(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_x()->get_uid(), 9400);
    EXPECT_NE(attrs.get_indices(), nullptr);
    EXPECT_EQ(attrs.get_indices()->get_uid(), 9401);
    EXPECT_NE(attrs.get_y(), nullptr);
    EXPECT_EQ(attrs.get_y()->get_uid(), 9402);

    // Verify data field getters
    EXPECT_EQ(attrs.get_axis(), 0);
}

TEST(TestGatherAttributes, DefaultValues)
{
    const GatherAttributes attrs;

    // Tensors should be null by default
    EXPECT_EQ(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_indices(), nullptr);
    EXPECT_EQ(attrs.get_y(), nullptr);

    // Vector fields should be empty by default
    EXPECT_EQ(attrs.get_axis(), 0);
}

TEST(TestGatherAttributes, SetXMove)
{
    GatherAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9400).set_name("MovedXTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = xTensor.get();

    attrs.set_x(std::move(xTensor));

    // After move, original should be nullptr
    EXPECT_EQ(xTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_x();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestGatherAttributes, SetIndicesMove)
{
    GatherAttributes attrs;

    auto indicesTensor = std::make_shared<TensorAttributes>();
    indicesTensor->set_uid(9401)
        .set_name("MovedIndicesTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = indicesTensor.get();

    attrs.set_indices(std::move(indicesTensor));

    // After move, original should be nullptr
    EXPECT_EQ(indicesTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_indices();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestGatherAttributes, SetYMove)
{
    GatherAttributes attrs;

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(9402).set_name("MovedYTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = yTensor.get();

    attrs.set_y(std::move(yTensor));

    // After move, original should be nullptr
    EXPECT_EQ(yTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_y();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestGatherAttributes, SetTensorsConstRef)
{
    GatherAttributes attrs;

    // Create tensors
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9400).set_name("XConstRef");
    auto indicesTensor = std::make_shared<TensorAttributes>();
    indicesTensor->set_uid(9401).set_name("IndicesConstRef");
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(9402).set_name("YConstRef");

    // Set using const reference (copy)
    attrs.set_x(xTensor);
    attrs.set_indices(indicesTensor);
    attrs.set_y(yTensor);

    // Original tensors should still be valid
    EXPECT_NE(xTensor, nullptr);
    EXPECT_NE(indicesTensor, nullptr);
    EXPECT_NE(yTensor, nullptr);
}

TEST(TestGatherAttributes, LogicalAndStrictEquality)
{
    GatherAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_axis(0);
    auto x1 = std::make_shared<TensorAttributes>();
    x1->set_uid(1).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_x(x1);
    auto indices1 = std::make_shared<TensorAttributes>();
    indices1->set_uid(2).set_name("Indices").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_indices(indices1);
    auto y1 = std::make_shared<TensorAttributes>();
    y1->set_uid(3).set_name("Y").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_y(y1);

    GatherAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_axis(0);
    auto x2 = std::make_shared<TensorAttributes>();
    x2->set_uid(1).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_x(x2);
    auto indices2 = std::make_shared<TensorAttributes>();
    indices2->set_uid(2).set_name("Indices").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_indices(indices2);
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

    // axis mismatch: semantic, must fail both checks
    attr2.set_axis(1);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_axis(0); // Revert

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
