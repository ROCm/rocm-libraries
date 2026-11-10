// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/PagedCacheStoreAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// --- Test suite: TestPagedCacheStoreAttributes ---

TEST(TestPagedCacheStoreAttributes, CreatePagedCacheStoreAttributes)
{
    PagedCacheStoreAttributes attrs;

    // Set all tensors
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9500);
    attrs.set_x(xTensor);
    auto slotMappingTensor = std::make_shared<TensorAttributes>();
    slotMappingTensor->set_uid(9501);
    attrs.set_slot_mapping(slotMappingTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(9503);
    attrs.set_scale(scaleTensor);
    auto containerTensor = std::make_shared<TensorAttributes>();
    containerTensor->set_uid(9502);
    attrs.set_container(containerTensor);

    // Set data fields

    // Verify tensor getters
    EXPECT_NE(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_x()->get_uid(), 9500);
    EXPECT_NE(attrs.get_slot_mapping(), nullptr);
    EXPECT_EQ(attrs.get_slot_mapping()->get_uid(), 9501);
    EXPECT_NE(attrs.get_scale(), nullptr);
    EXPECT_EQ(attrs.get_scale()->get_uid(), 9503);
    EXPECT_NE(attrs.get_container(), nullptr);
    EXPECT_EQ(attrs.get_container()->get_uid(), 9502);

    // Verify data field getters
}

TEST(TestPagedCacheStoreAttributes, DefaultValues)
{
    const PagedCacheStoreAttributes attrs;

    // Tensors should be null by default
    EXPECT_EQ(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_slot_mapping(), nullptr);
    EXPECT_EQ(attrs.get_scale(), nullptr);
    EXPECT_EQ(attrs.get_container(), nullptr);

    // Vector fields should be empty by default
}

TEST(TestPagedCacheStoreAttributes, SetXMove)
{
    PagedCacheStoreAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9500).set_name("MovedXTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = xTensor.get();

    attrs.set_x(std::move(xTensor));

    // After move, original should be nullptr
    EXPECT_EQ(xTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_x();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestPagedCacheStoreAttributes, SetSlotMappingMove)
{
    PagedCacheStoreAttributes attrs;

    auto slotMappingTensor = std::make_shared<TensorAttributes>();
    slotMappingTensor->set_uid(9501)
        .set_name("MovedSlotMappingTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = slotMappingTensor.get();

    attrs.set_slot_mapping(std::move(slotMappingTensor));

    // After move, original should be nullptr
    EXPECT_EQ(slotMappingTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_slot_mapping();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestPagedCacheStoreAttributes, SetScaleMove)
{
    PagedCacheStoreAttributes attrs;

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(9503)
        .set_name("MovedScaleTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = scaleTensor.get();

    attrs.set_scale(std::move(scaleTensor));

    // After move, original should be nullptr
    EXPECT_EQ(scaleTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_scale();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestPagedCacheStoreAttributes, SetContainerMove)
{
    PagedCacheStoreAttributes attrs;

    auto containerTensor = std::make_shared<TensorAttributes>();
    containerTensor->set_uid(9502)
        .set_name("MovedContainerTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = containerTensor.get();

    attrs.set_container(std::move(containerTensor));

    // After move, original should be nullptr
    EXPECT_EQ(containerTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_container();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestPagedCacheStoreAttributes, SetTensorsConstRef)
{
    PagedCacheStoreAttributes attrs;

    // Create tensors
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9500).set_name("XConstRef");
    auto slotMappingTensor = std::make_shared<TensorAttributes>();
    slotMappingTensor->set_uid(9501).set_name("SlotMappingConstRef");
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(9503).set_name("ScaleConstRef");
    auto containerTensor = std::make_shared<TensorAttributes>();
    containerTensor->set_uid(9502).set_name("ContainerConstRef");

    // Set using const reference (copy)
    attrs.set_x(xTensor);
    attrs.set_slot_mapping(slotMappingTensor);
    attrs.set_scale(scaleTensor);
    attrs.set_container(containerTensor);

    // Original tensors should still be valid
    EXPECT_NE(xTensor, nullptr);
    EXPECT_NE(slotMappingTensor, nullptr);
    EXPECT_NE(scaleTensor, nullptr);
    EXPECT_NE(containerTensor, nullptr);
}

TEST(TestPagedCacheStoreAttributes, LogicalAndStrictEquality)
{
    PagedCacheStoreAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    auto x1 = std::make_shared<TensorAttributes>();
    x1->set_uid(1).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_x(x1);
    auto slotMapping1 = std::make_shared<TensorAttributes>();
    slotMapping1->set_uid(2)
        .set_name("SlotMapping")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_slot_mapping(slotMapping1);
    auto scale1 = std::make_shared<TensorAttributes>();
    scale1->set_uid(3).set_name("Scale").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_scale(scale1);
    auto container1 = std::make_shared<TensorAttributes>();
    container1->set_uid(4).set_name("Container").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_container(container1);

    PagedCacheStoreAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    auto x2 = std::make_shared<TensorAttributes>();
    x2->set_uid(1).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_x(x2);
    auto slotMapping2 = std::make_shared<TensorAttributes>();
    slotMapping2->set_uid(2)
        .set_name("SlotMapping")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_slot_mapping(slotMapping2);
    auto scale2 = std::make_shared<TensorAttributes>();
    scale2->set_uid(3).set_name("Scale").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_scale(scale2);
    auto container2 = std::make_shared<TensorAttributes>();
    container2->set_uid(4).set_name("Container").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_container(container2);

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
