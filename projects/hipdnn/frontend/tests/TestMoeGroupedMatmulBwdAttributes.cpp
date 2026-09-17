// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulBwdAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// --- Test suite: TestMoeGroupedMatmulBwdAttributes ---

TEST(TestMoeGroupedMatmulBwdAttributes, CreateMoeGroupedMatmulBwdAttributes)
{
    MoeGroupedMatmulBwdAttributes attrs;

    // Set all tensors
    auto doutputTensor = std::make_shared<TensorAttributes>();
    doutputTensor->set_uid(1910);
    attrs.set_doutput(doutputTensor);
    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_uid(1911);
    attrs.set_token(tokenTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_uid(1912);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto dweightTensor = std::make_shared<TensorAttributes>();
    dweightTensor->set_uid(1913);
    attrs.set_dweight(dweightTensor);

    // Set data fields

    // Verify tensor getters
    EXPECT_NE(attrs.get_doutput(), nullptr);
    EXPECT_EQ(attrs.get_doutput()->get_uid(), 1910);
    EXPECT_NE(attrs.get_token(), nullptr);
    EXPECT_EQ(attrs.get_token()->get_uid(), 1911);
    EXPECT_NE(attrs.get_first_token_offset(), nullptr);
    EXPECT_EQ(attrs.get_first_token_offset()->get_uid(), 1912);
    EXPECT_NE(attrs.get_dweight(), nullptr);
    EXPECT_EQ(attrs.get_dweight()->get_uid(), 1913);

    // Verify data field getters
}

TEST(TestMoeGroupedMatmulBwdAttributes, DefaultValues)
{
    const MoeGroupedMatmulBwdAttributes attrs;

    // Tensors should be null by default
    EXPECT_EQ(attrs.get_doutput(), nullptr);
    EXPECT_EQ(attrs.get_token(), nullptr);
    EXPECT_EQ(attrs.get_first_token_offset(), nullptr);
    EXPECT_EQ(attrs.get_dweight(), nullptr);

    // Vector fields should be empty by default
}

TEST(TestMoeGroupedMatmulBwdAttributes, SetDoutputMove)
{
    MoeGroupedMatmulBwdAttributes attrs;

    auto doutputTensor = std::make_shared<TensorAttributes>();
    doutputTensor->set_uid(1910)
        .set_name("MovedDoutputTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = doutputTensor.get();

    attrs.set_doutput(std::move(doutputTensor));

    // After move, original should be nullptr
    EXPECT_EQ(doutputTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_doutput();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestMoeGroupedMatmulBwdAttributes, SetTokenMove)
{
    MoeGroupedMatmulBwdAttributes attrs;

    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_uid(1911)
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

TEST(TestMoeGroupedMatmulBwdAttributes, SetFirstTokenOffsetMove)
{
    MoeGroupedMatmulBwdAttributes attrs;

    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_uid(1912)
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

TEST(TestMoeGroupedMatmulBwdAttributes, SetDweightMove)
{
    MoeGroupedMatmulBwdAttributes attrs;

    auto dweightTensor = std::make_shared<TensorAttributes>();
    dweightTensor->set_uid(1913)
        .set_name("MovedDweightTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = dweightTensor.get();

    attrs.set_dweight(std::move(dweightTensor));

    // After move, original should be nullptr
    EXPECT_EQ(dweightTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_dweight();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestMoeGroupedMatmulBwdAttributes, SetTensorsConstRef)
{
    MoeGroupedMatmulBwdAttributes attrs;

    // Create tensors
    auto doutputTensor = std::make_shared<TensorAttributes>();
    doutputTensor->set_uid(1910).set_name("DoutputConstRef");
    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_uid(1911).set_name("TokenConstRef");
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_uid(1912).set_name("FirstTokenOffsetConstRef");
    auto dweightTensor = std::make_shared<TensorAttributes>();
    dweightTensor->set_uid(1913).set_name("DweightConstRef");

    // Set using const reference (copy)
    attrs.set_doutput(doutputTensor);
    attrs.set_token(tokenTensor);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    attrs.set_dweight(dweightTensor);

    // Original tensors should still be valid
    EXPECT_NE(doutputTensor, nullptr);
    EXPECT_NE(tokenTensor, nullptr);
    EXPECT_NE(firstTokenOffsetTensor, nullptr);
    EXPECT_NE(dweightTensor, nullptr);
}

TEST(TestMoeGroupedMatmulBwdAttributes, LogicalAndStrictEquality)
{
    MoeGroupedMatmulBwdAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    auto doutput1 = std::make_shared<TensorAttributes>();
    doutput1->set_uid(1).set_name("Doutput").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_doutput(doutput1);

    auto token1 = std::make_shared<TensorAttributes>();
    token1->set_uid(2).set_name("Token").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_token(token1);

    MoeGroupedMatmulBwdAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    auto doutput2 = std::make_shared<TensorAttributes>();
    doutput2->set_uid(1).set_name("Doutput").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_doutput(doutput2);

    auto token2 = std::make_shared<TensorAttributes>();
    token2->set_uid(2).set_name("Token").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_token(token2);

    // Initial check: everything matches exactly
    EXPECT_TRUE(attr1 == attr2);
    EXPECT_FALSE(attr1 != attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));

    // Structural tensor mismatch: different UID/name/type entirely
    auto structuralMismatchDoutput = std::make_shared<TensorAttributes>();
    structuralMismatchDoutput->set_uid(99).set_name("MismatchedDoutput");
    attr2.set_doutput(structuralMismatchDoutput);

    EXPECT_TRUE(attr1 != attr2);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2)); // Structural/type gap implies logical inequality
    attr2.set_doutput(doutput2); // Revert

    // No extra scalar/enum fields on this class: logical/strict equality is
    // driven entirely by the base class's tensor/metadata comparison.

    // Change metadata (UID/Name) on a tensor while keeping mathematical layout intact
    auto logicalMatchDoutput = std::make_shared<TensorAttributes>();
    logicalMatchDoutput
        ->set_uid(555) // Diverges from attr1's doutput1 (uid: 1)
        .set_name("DIVERGENT_NAME") // Diverges from attr1's doutput1 ("Doutput")
        .set_data_type(hipdnn_frontend::DataType::FLOAT); // Layout matches
    attr2.set_doutput(logicalMatchDoutput);

    // Expecting: strict evaluation fails, but functional logical comparison passes
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));
}
