// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/MoeRoutingAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// --- Test suite: TestMoeRoutingAttributes ---

TEST(TestMoeRoutingAttributes, CreateMoeRoutingAttributes)
{
    MoeRoutingAttributes attrs;

    // Set all tensors
    auto topkIdsTensor = std::make_shared<TensorAttributes>();
    topkIdsTensor->set_uid(9700);
    attrs.set_topk_ids(topkIdsTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_uid(9701);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto tokenIndexTensor = std::make_shared<TensorAttributes>();
    tokenIndexTensor->set_uid(9702);
    attrs.set_token_index(tokenIndexTensor);
    auto tokenKsTensor = std::make_shared<TensorAttributes>();
    tokenKsTensor->set_uid(9703);
    attrs.set_token_ks(tokenKsTensor);

    // Set data fields
    attrs.set_num_experts(4);

    // Verify tensor getters
    EXPECT_NE(attrs.get_topk_ids(), nullptr);
    EXPECT_EQ(attrs.get_topk_ids()->get_uid(), 9700);
    EXPECT_NE(attrs.get_first_token_offset(), nullptr);
    EXPECT_EQ(attrs.get_first_token_offset()->get_uid(), 9701);
    EXPECT_NE(attrs.get_token_index(), nullptr);
    EXPECT_EQ(attrs.get_token_index()->get_uid(), 9702);
    EXPECT_NE(attrs.get_token_ks(), nullptr);
    EXPECT_EQ(attrs.get_token_ks()->get_uid(), 9703);

    // Verify data field getters
    EXPECT_EQ(attrs.get_num_experts(), 4);
}

TEST(TestMoeRoutingAttributes, DefaultValues)
{
    const MoeRoutingAttributes attrs;

    // Tensors should be null by default
    EXPECT_EQ(attrs.get_topk_ids(), nullptr);
    EXPECT_EQ(attrs.get_first_token_offset(), nullptr);
    EXPECT_EQ(attrs.get_token_index(), nullptr);
    EXPECT_EQ(attrs.get_token_ks(), nullptr);

    // Vector fields should be empty by default
    EXPECT_EQ(attrs.get_num_experts(), 0);
}

TEST(TestMoeRoutingAttributes, SetTopkIdsMove)
{
    MoeRoutingAttributes attrs;

    auto topkIdsTensor = std::make_shared<TensorAttributes>();
    topkIdsTensor->set_uid(9700)
        .set_name("MovedTopkIdsTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = topkIdsTensor.get();

    attrs.set_topk_ids(std::move(topkIdsTensor));

    // After move, original should be nullptr
    EXPECT_EQ(topkIdsTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_topk_ids();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestMoeRoutingAttributes, SetFirstTokenOffsetMove)
{
    MoeRoutingAttributes attrs;

    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_uid(9701)
        .set_name("MovedFirstTokenOffsetTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = firstTokenOffsetTensor.get();

    attrs.set_first_token_offset(std::move(firstTokenOffsetTensor));

    // After move, original should be nullptr
    EXPECT_EQ(firstTokenOffsetTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_first_token_offset();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestMoeRoutingAttributes, SetTokenIndexMove)
{
    MoeRoutingAttributes attrs;

    auto tokenIndexTensor = std::make_shared<TensorAttributes>();
    tokenIndexTensor->set_uid(9702)
        .set_name("MovedTokenIndexTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = tokenIndexTensor.get();

    attrs.set_token_index(std::move(tokenIndexTensor));

    // After move, original should be nullptr
    EXPECT_EQ(tokenIndexTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_token_index();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestMoeRoutingAttributes, SetTokenKsMove)
{
    MoeRoutingAttributes attrs;

    auto tokenKsTensor = std::make_shared<TensorAttributes>();
    tokenKsTensor->set_uid(9703)
        .set_name("MovedTokenKsTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = tokenKsTensor.get();

    attrs.set_token_ks(std::move(tokenKsTensor));

    // After move, original should be nullptr
    EXPECT_EQ(tokenKsTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_token_ks();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestMoeRoutingAttributes, SetTensorsConstRef)
{
    MoeRoutingAttributes attrs;

    // Create tensors
    auto topkIdsTensor = std::make_shared<TensorAttributes>();
    topkIdsTensor->set_uid(9700).set_name("TopkIdsConstRef");
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_uid(9701).set_name("FirstTokenOffsetConstRef");
    auto tokenIndexTensor = std::make_shared<TensorAttributes>();
    tokenIndexTensor->set_uid(9702).set_name("TokenIndexConstRef");
    auto tokenKsTensor = std::make_shared<TensorAttributes>();
    tokenKsTensor->set_uid(9703).set_name("TokenKsConstRef");

    // Set using const reference (copy)
    attrs.set_topk_ids(topkIdsTensor);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    attrs.set_token_index(tokenIndexTensor);
    attrs.set_token_ks(tokenKsTensor);

    // Original tensors should still be valid
    EXPECT_NE(topkIdsTensor, nullptr);
    EXPECT_NE(firstTokenOffsetTensor, nullptr);
    EXPECT_NE(tokenIndexTensor, nullptr);
    EXPECT_NE(tokenKsTensor, nullptr);
}

TEST(TestMoeRoutingAttributes, LogicalAndStrictEquality)
{
    MoeRoutingAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_num_experts(4);
    auto topkIds1 = std::make_shared<TensorAttributes>();
    topkIds1->set_uid(1).set_name("TopkIds").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_topk_ids(topkIds1);
    auto firstTokenOffset1 = std::make_shared<TensorAttributes>();
    firstTokenOffset1->set_uid(2)
        .set_name("FirstTokenOffset")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_first_token_offset(firstTokenOffset1);
    auto tokenIndex1 = std::make_shared<TensorAttributes>();
    tokenIndex1->set_uid(3).set_name("TokenIndex").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_token_index(tokenIndex1);
    auto tokenKs1 = std::make_shared<TensorAttributes>();
    tokenKs1->set_uid(4).set_name("TokenKs").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_token_ks(tokenKs1);

    MoeRoutingAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_num_experts(4);
    auto topkIds2 = std::make_shared<TensorAttributes>();
    topkIds2->set_uid(1).set_name("TopkIds").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_topk_ids(topkIds2);
    auto firstTokenOffset2 = std::make_shared<TensorAttributes>();
    firstTokenOffset2->set_uid(2)
        .set_name("FirstTokenOffset")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_first_token_offset(firstTokenOffset2);
    auto tokenIndex2 = std::make_shared<TensorAttributes>();
    tokenIndex2->set_uid(3).set_name("TokenIndex").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_token_index(tokenIndex2);
    auto tokenKs2 = std::make_shared<TensorAttributes>();
    tokenKs2->set_uid(4).set_name("TokenKs").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_token_ks(tokenKs2);

    // Initial check: everything matches exactly
    EXPECT_TRUE(attr1 == attr2);
    EXPECT_FALSE(attr1 != attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));

    // Structural tensor mismatch: different UID/name/type entirely
    auto structuralMismatchTopkIds = std::make_shared<TensorAttributes>();
    structuralMismatchTopkIds->set_uid(9999).set_name("MismatchedTopkIds");
    attr2.set_topk_ids(structuralMismatchTopkIds);

    EXPECT_TRUE(attr1 != attr2);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2)); // Structural/type gap implies logical inequality
    attr2.set_topk_ids(topkIds2); // Revert

    // num_experts mismatch: semantic, must fail both checks
    attr2.set_num_experts(5);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_num_experts(4); // Revert

    // Change metadata (UID/Name) on a tensor while keeping mathematical layout intact
    auto logicalMatchTopkIds = std::make_shared<TensorAttributes>();
    logicalMatchTopkIds
        ->set_uid(555) // Diverges from attr1's topkIds1 (uid: 1)
        .set_name("DIVERGENT_NAME") // Diverges from attr1's topkIds1 ("TopkIds")
        .set_data_type(hipdnn_frontend::DataType::FLOAT); // Layout matches
    attr2.set_topk_ids(logicalMatchTopkIds);

    // Expecting: strict evaluation fails, but functional logical comparison passes
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));
}
