// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/attributes/TopkAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// --- Test suite: TestTopkAttributes ---

TEST(TestTopkAttributes, CreateTopkAttributes)
{
    TopkAttributes attrs;

    // Set all tensors
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9300);
    attrs.set_x(xTensor);
    auto valuesTensor = std::make_shared<TensorAttributes>();
    valuesTensor->set_uid(9302);
    attrs.set_values(valuesTensor);
    auto indicesTensor = std::make_shared<TensorAttributes>();
    indicesTensor->set_uid(9301);
    attrs.set_indices(indicesTensor);

    // Set data fields
    attrs.set_k(4);
    attrs.set_largest(false);

    // Verify tensor getters
    EXPECT_NE(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_x()->get_uid(), 9300);
    EXPECT_NE(attrs.get_values(), nullptr);
    EXPECT_EQ(attrs.get_values()->get_uid(), 9302);
    EXPECT_NE(attrs.get_indices(), nullptr);
    EXPECT_EQ(attrs.get_indices()->get_uid(), 9301);

    // Verify data field getters
    EXPECT_EQ(attrs.get_k(), 4);
    EXPECT_EQ(attrs.get_largest(), false);
}

TEST(TestTopkAttributes, DefaultValues)
{
    const TopkAttributes attrs;

    // Tensors should be null by default
    EXPECT_EQ(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_values(), nullptr);
    EXPECT_EQ(attrs.get_indices(), nullptr);

    // Vector fields should be empty by default
    EXPECT_EQ(attrs.get_k(), 0);
    EXPECT_EQ(attrs.get_largest(), true);
}

TEST(TestTopkAttributes, SetXMove)
{
    TopkAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9300).set_name("MovedXTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = xTensor.get();

    attrs.set_x(std::move(xTensor));

    // After move, original should be nullptr
    EXPECT_EQ(xTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_x();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestTopkAttributes, SetValuesMove)
{
    TopkAttributes attrs;

    auto valuesTensor = std::make_shared<TensorAttributes>();
    valuesTensor->set_uid(9302)
        .set_name("MovedValuesTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = valuesTensor.get();

    attrs.set_values(std::move(valuesTensor));

    // After move, original should be nullptr
    EXPECT_EQ(valuesTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_values();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestTopkAttributes, SetIndicesMove)
{
    TopkAttributes attrs;

    auto indicesTensor = std::make_shared<TensorAttributes>();
    indicesTensor->set_uid(9301)
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

TEST(TestTopkAttributes, SetTensorsConstRef)
{
    TopkAttributes attrs;

    // Create tensors
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9300).set_name("XConstRef");
    auto valuesTensor = std::make_shared<TensorAttributes>();
    valuesTensor->set_uid(9302).set_name("ValuesConstRef");
    auto indicesTensor = std::make_shared<TensorAttributes>();
    indicesTensor->set_uid(9301).set_name("IndicesConstRef");

    // Set using const reference (copy)
    attrs.set_x(xTensor);
    attrs.set_values(valuesTensor);
    attrs.set_indices(indicesTensor);

    // Original tensors should still be valid
    EXPECT_NE(xTensor, nullptr);
    EXPECT_NE(valuesTensor, nullptr);
    EXPECT_NE(indicesTensor, nullptr);
}

TEST(TestTopkAttributes, LogicalAndStrictEquality)
{
    TopkAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_k(4);
    attr1.set_largest(false);
    auto x1 = std::make_shared<TensorAttributes>();
    x1->set_uid(1).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_x(x1);
    auto values1 = std::make_shared<TensorAttributes>();
    values1->set_uid(2).set_name("Values").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_values(values1);
    auto indices1 = std::make_shared<TensorAttributes>();
    indices1->set_uid(3).set_name("Indices").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_indices(indices1);

    TopkAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_k(4);
    attr2.set_largest(false);
    auto x2 = std::make_shared<TensorAttributes>();
    x2->set_uid(1).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_x(x2);
    auto values2 = std::make_shared<TensorAttributes>();
    values2->set_uid(2).set_name("Values").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_values(values2);
    auto indices2 = std::make_shared<TensorAttributes>();
    indices2->set_uid(3).set_name("Indices").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_indices(indices2);

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

    // k mismatch: semantic, must fail both checks
    attr2.set_k(5);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_k(4); // Revert

    // largest mismatch: semantic, must fail both checks
    attr2.set_largest(true);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_largest(false); // Revert

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
