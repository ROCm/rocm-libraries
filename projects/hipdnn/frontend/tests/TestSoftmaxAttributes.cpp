// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/SoftmaxAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// --- Test suite: TestSoftmaxAttributes ---

TEST(TestSoftmaxAttributes, CreateSoftmaxAttributes)
{
    SoftmaxAttributes attrs;

    // Set all tensors
    auto pTensor = std::make_shared<TensorAttributes>();
    pTensor->set_uid(9200);
    attrs.set_p(pTensor);
    auto sTensor = std::make_shared<TensorAttributes>();
    sTensor->set_uid(9201);
    attrs.set_s(sTensor);
    auto statsTensor = std::make_shared<TensorAttributes>();
    statsTensor->set_uid(9202);
    attrs.set_stats(statsTensor);

    // Set data fields
    attrs.set_stats_use_log2(true);

    // Verify tensor getters
    EXPECT_NE(attrs.get_p(), nullptr);
    EXPECT_EQ(attrs.get_p()->get_uid(), 9200);
    EXPECT_NE(attrs.get_s(), nullptr);
    EXPECT_EQ(attrs.get_s()->get_uid(), 9201);
    EXPECT_NE(attrs.get_stats(), nullptr);
    EXPECT_EQ(attrs.get_stats()->get_uid(), 9202);

    // Verify data field getters
    EXPECT_EQ(attrs.get_stats_use_log2(), true);
}

TEST(TestSoftmaxAttributes, DefaultValues)
{
    const SoftmaxAttributes attrs;

    // Tensors should be null by default
    EXPECT_EQ(attrs.get_p(), nullptr);
    EXPECT_EQ(attrs.get_s(), nullptr);
    EXPECT_EQ(attrs.get_stats(), nullptr);

    // Vector fields should be empty by default
    EXPECT_EQ(attrs.get_stats_use_log2(), 0);
}

TEST(TestSoftmaxAttributes, SetPMove)
{
    SoftmaxAttributes attrs;

    auto pTensor = std::make_shared<TensorAttributes>();
    pTensor->set_uid(9200).set_name("MovedPTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = pTensor.get();

    attrs.set_p(std::move(pTensor));

    // After move, original should be nullptr
    EXPECT_EQ(pTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_p();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestSoftmaxAttributes, SetSMove)
{
    SoftmaxAttributes attrs;

    auto sTensor = std::make_shared<TensorAttributes>();
    sTensor->set_uid(9201).set_name("MovedSTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = sTensor.get();

    attrs.set_s(std::move(sTensor));

    // After move, original should be nullptr
    EXPECT_EQ(sTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_s();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestSoftmaxAttributes, SetStatsMove)
{
    SoftmaxAttributes attrs;

    auto statsTensor = std::make_shared<TensorAttributes>();
    statsTensor->set_uid(9202)
        .set_name("MovedStatsTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = statsTensor.get();

    attrs.set_stats(std::move(statsTensor));

    // After move, original should be nullptr
    EXPECT_EQ(statsTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_stats();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestSoftmaxAttributes, SetTensorsConstRef)
{
    SoftmaxAttributes attrs;

    // Create tensors
    auto pTensor = std::make_shared<TensorAttributes>();
    pTensor->set_uid(9200).set_name("PConstRef");
    auto sTensor = std::make_shared<TensorAttributes>();
    sTensor->set_uid(9201).set_name("SConstRef");
    auto statsTensor = std::make_shared<TensorAttributes>();
    statsTensor->set_uid(9202).set_name("StatsConstRef");

    // Set using const reference (copy)
    attrs.set_p(pTensor);
    attrs.set_s(sTensor);
    attrs.set_stats(statsTensor);

    // Original tensors should still be valid
    EXPECT_NE(pTensor, nullptr);
    EXPECT_NE(sTensor, nullptr);
    EXPECT_NE(statsTensor, nullptr);
}

TEST(TestSoftmaxAttributes, LogicalAndStrictEquality)
{
    SoftmaxAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_stats_use_log2(true);
    auto p1 = std::make_shared<TensorAttributes>();
    p1->set_uid(1).set_name("P").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_p(p1);
    auto s1 = std::make_shared<TensorAttributes>();
    s1->set_uid(2).set_name("S").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_s(s1);
    auto stats1 = std::make_shared<TensorAttributes>();
    stats1->set_uid(3).set_name("Stats").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_stats(stats1);

    SoftmaxAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_stats_use_log2(true);
    auto p2 = std::make_shared<TensorAttributes>();
    p2->set_uid(1).set_name("P").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_p(p2);
    auto s2 = std::make_shared<TensorAttributes>();
    s2->set_uid(2).set_name("S").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_s(s2);
    auto stats2 = std::make_shared<TensorAttributes>();
    stats2->set_uid(3).set_name("Stats").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_stats(stats2);

    // Initial check: everything matches exactly
    EXPECT_TRUE(attr1 == attr2);
    EXPECT_FALSE(attr1 != attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));

    // Structural tensor mismatch: different UID/name/type entirely
    auto structuralMismatchP = std::make_shared<TensorAttributes>();
    structuralMismatchP->set_uid(9999).set_name("MismatchedP");
    attr2.set_p(structuralMismatchP);

    EXPECT_TRUE(attr1 != attr2);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2)); // Structural/type gap implies logical inequality
    attr2.set_p(p2); // Revert

    // stats_use_log2 mismatch: semantic, must fail both checks
    attr2.set_stats_use_log2(false);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_stats_use_log2(true); // Revert

    // Change metadata (UID/Name) on a tensor while keeping mathematical layout intact
    auto logicalMatchP = std::make_shared<TensorAttributes>();
    logicalMatchP
        ->set_uid(555) // Diverges from attr1's p1 (uid: 1)
        .set_name("DIVERGENT_NAME") // Diverges from attr1's p1 ("P")
        .set_data_type(hipdnn_frontend::DataType::FLOAT); // Layout matches
    attr2.set_p(logicalMatchP);

    // Expecting: strict evaluation fails, but functional logical comparison passes
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));
}

TEST(TestSoftmaxAttributes, GenerateStatsDefaultsOffAndIsSettable)
{
    SoftmaxAttributes attrs;
    EXPECT_FALSE(attrs.get_generate_stats());
    EXPECT_FALSE(attrs.get_stats_use_log2());
    EXPECT_EQ(attrs.get_stats(), nullptr);

    attrs.set_generate_stats(true);
    EXPECT_TRUE(attrs.get_generate_stats());
}
