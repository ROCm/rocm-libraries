// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// --- Test suite: TestMoeGroupedMatmulAttributes ---

TEST(TestMoeGroupedMatmulAttributes, CreateMoeGroupedMatmulAttributes)
{
    MoeGroupedMatmulAttributes attrs;

    // Set all tensors
    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_uid(1900);
    attrs.set_token(tokenTensor);
    auto weightTensor = std::make_shared<TensorAttributes>();
    weightTensor->set_uid(1901);
    attrs.set_weight(weightTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_uid(1902);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto tokenIndexTensor = std::make_shared<TensorAttributes>();
    tokenIndexTensor->set_uid(1903);
    attrs.set_token_index(tokenIndexTensor);
    auto tokenKsTensor = std::make_shared<TensorAttributes>();
    tokenKsTensor->set_uid(1904);
    attrs.set_token_ks(tokenKsTensor);
    auto outputTensor = std::make_shared<TensorAttributes>();
    outputTensor->set_uid(1905);
    attrs.set_output(outputTensor);

    // Set data fields
    attrs.set_mode(MoeGroupedMatmulMode::SCATTER);
    attrs.set_top_k(2);

    // Verify tensor getters
    EXPECT_NE(attrs.get_token(), nullptr);
    EXPECT_EQ(attrs.get_token()->get_uid(), 1900);
    EXPECT_NE(attrs.get_weight(), nullptr);
    EXPECT_EQ(attrs.get_weight()->get_uid(), 1901);
    EXPECT_NE(attrs.get_first_token_offset(), nullptr);
    EXPECT_EQ(attrs.get_first_token_offset()->get_uid(), 1902);
    EXPECT_NE(attrs.get_token_index(), nullptr);
    EXPECT_EQ(attrs.get_token_index()->get_uid(), 1903);
    EXPECT_NE(attrs.get_token_ks(), nullptr);
    EXPECT_EQ(attrs.get_token_ks()->get_uid(), 1904);
    EXPECT_NE(attrs.get_output(), nullptr);
    EXPECT_EQ(attrs.get_output()->get_uid(), 1905);

    // Verify data field getters
    EXPECT_EQ(attrs.get_top_k(), 2);
}

TEST(TestMoeGroupedMatmulAttributes, DefaultValues)
{
    const MoeGroupedMatmulAttributes attrs;

    // Tensors should be null by default
    EXPECT_EQ(attrs.get_token(), nullptr);
    EXPECT_EQ(attrs.get_weight(), nullptr);
    EXPECT_EQ(attrs.get_first_token_offset(), nullptr);
    EXPECT_EQ(attrs.get_token_index(), nullptr);
    EXPECT_EQ(attrs.get_token_ks(), nullptr);
    EXPECT_EQ(attrs.get_output(), nullptr);

    // Vector fields should be empty by default
    EXPECT_EQ(attrs.get_mode(), MoeGroupedMatmulMode::NONE);
    EXPECT_EQ(attrs.get_top_k(), 0);
}

TEST(TestMoeGroupedMatmulAttributes, SetTokenMove)
{
    MoeGroupedMatmulAttributes attrs;

    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_uid(1900)
        .set_name("MovedTokenTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = tokenTensor.get();

    attrs.set_token(std::move(tokenTensor));

    // After move, original should be nullptr
    EXPECT_EQ(tokenTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_token();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestMoeGroupedMatmulAttributes, SetWeightMove)
{
    MoeGroupedMatmulAttributes attrs;

    auto weightTensor = std::make_shared<TensorAttributes>();
    weightTensor->set_uid(1901)
        .set_name("MovedWeightTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = weightTensor.get();

    attrs.set_weight(std::move(weightTensor));

    // After move, original should be nullptr
    EXPECT_EQ(weightTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_weight();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestMoeGroupedMatmulAttributes, SetFirstTokenOffsetMove)
{
    MoeGroupedMatmulAttributes attrs;

    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_uid(1902)
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

TEST(TestMoeGroupedMatmulAttributes, SetTokenIndexMove)
{
    MoeGroupedMatmulAttributes attrs;

    auto tokenIndexTensor = std::make_shared<TensorAttributes>();
    tokenIndexTensor->set_uid(1903)
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

TEST(TestMoeGroupedMatmulAttributes, SetTokenKsMove)
{
    MoeGroupedMatmulAttributes attrs;

    auto tokenKsTensor = std::make_shared<TensorAttributes>();
    tokenKsTensor->set_uid(1904)
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

TEST(TestMoeGroupedMatmulAttributes, SetOutputMove)
{
    MoeGroupedMatmulAttributes attrs;

    auto outputTensor = std::make_shared<TensorAttributes>();
    outputTensor->set_uid(1905)
        .set_name("MovedOutputTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = outputTensor.get();

    attrs.set_output(std::move(outputTensor));

    // After move, original should be nullptr
    EXPECT_EQ(outputTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_output();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestMoeGroupedMatmulAttributes, SetTensorsConstRef)
{
    MoeGroupedMatmulAttributes attrs;

    // Create tensors
    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_uid(1900).set_name("TokenConstRef");
    auto weightTensor = std::make_shared<TensorAttributes>();
    weightTensor->set_uid(1901).set_name("WeightConstRef");
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_uid(1902).set_name("FirstTokenOffsetConstRef");
    auto tokenIndexTensor = std::make_shared<TensorAttributes>();
    tokenIndexTensor->set_uid(1903).set_name("TokenIndexConstRef");
    auto tokenKsTensor = std::make_shared<TensorAttributes>();
    tokenKsTensor->set_uid(1904).set_name("TokenKsConstRef");
    auto outputTensor = std::make_shared<TensorAttributes>();
    outputTensor->set_uid(1905).set_name("OutputConstRef");

    // Set using const reference (copy)
    attrs.set_token(tokenTensor);
    attrs.set_weight(weightTensor);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    attrs.set_token_index(tokenIndexTensor);
    attrs.set_token_ks(tokenKsTensor);
    attrs.set_output(outputTensor);

    // Original tensors should still be valid
    EXPECT_NE(tokenTensor, nullptr);
    EXPECT_NE(weightTensor, nullptr);
    EXPECT_NE(firstTokenOffsetTensor, nullptr);
    EXPECT_NE(tokenIndexTensor, nullptr);
    EXPECT_NE(tokenKsTensor, nullptr);
    EXPECT_NE(outputTensor, nullptr);
}

TEST(TestMoeGroupedMatmulAttributes, LogicalAndStrictEquality)
{
    MoeGroupedMatmulAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_mode(MoeGroupedMatmulMode::SCATTER);
    attr1.set_top_k(2);

    auto token1 = std::make_shared<TensorAttributes>();
    token1->set_uid(1).set_name("Token").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_token(token1);

    auto weight1 = std::make_shared<TensorAttributes>();
    weight1->set_uid(2).set_name("Weight").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_weight(weight1);

    MoeGroupedMatmulAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_mode(MoeGroupedMatmulMode::SCATTER);
    attr2.set_top_k(2);

    auto token2 = std::make_shared<TensorAttributes>();
    token2->set_uid(1).set_name("Token").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_token(token2);

    auto weight2 = std::make_shared<TensorAttributes>();
    weight2->set_uid(2).set_name("Weight").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_weight(weight2);

    // Initial check: everything matches exactly
    EXPECT_TRUE(attr1 == attr2);
    EXPECT_FALSE(attr1 != attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));

    // Structural tensor mismatch: different UID/name/type entirely
    auto structuralMismatchToken = std::make_shared<TensorAttributes>();
    structuralMismatchToken->set_uid(99).set_name("MismatchedToken");
    attr2.set_token(structuralMismatchToken);

    EXPECT_TRUE(attr1 != attr2);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2)); // Structural/type gap implies logical inequality
    attr2.set_token(token2); // Revert

    // mode mismatch: semantic, must fail both checks
    attr2.set_mode(MoeGroupedMatmulMode::GATHER);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_mode(MoeGroupedMatmulMode::SCATTER); // Revert

    // top_k mismatch: semantic, must fail both checks
    attr2.set_top_k(4);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_top_k(2); // Revert

    // Unset-vs-unset optional fields: two default-constructed attrs with
    // matching tensors should still compare equal
    MoeGroupedMatmulAttributes sparse1;
    MoeGroupedMatmulAttributes sparse2;
    EXPECT_TRUE(sparse1 == sparse2);
    EXPECT_TRUE(sparse1.logicallyEquals(sparse2));

    // Set-vs-unset top_k should differ
    sparse2.set_top_k(1);
    EXPECT_FALSE(sparse1 == sparse2);
    EXPECT_FALSE(sparse1.logicallyEquals(sparse2));

    // Change metadata (UID/Name) on a tensor while keeping mathematical layout intact
    auto logicalMatchToken = std::make_shared<TensorAttributes>();
    logicalMatchToken
        ->set_uid(555) // Diverges from attr1's token1 (uid: 1)
        .set_name("DIVERGENT_NAME") // Diverges from attr1's token1 ("Token")
        .set_data_type(hipdnn_frontend::DataType::FLOAT); // Layout matches
    attr2.set_token(logicalMatchToken);

    // Expecting: strict evaluation fails, but functional logical comparison passes
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));
}
