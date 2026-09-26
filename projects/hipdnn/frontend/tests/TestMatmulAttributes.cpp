// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include "hipdnn_frontend/attributes/MatmulAttributes.hpp"
#include <gtest/gtest.h>

TEST(TestMatmulAttributes, CreateMatmulAttributes)
{
    hipdnn_frontend::graph::MatmulAttributes matmulAttributes;

    // Set tensors
    matmulAttributes.set_a(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    matmulAttributes.set_b(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    matmulAttributes.set_c(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());

    // Configure A tensor
    auto aTensor = matmulAttributes.get_a();
    aTensor->set_uid(1)
        .set_name("ATensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({4, 8}) // rows, cols
        .set_stride({8, 1});

    // Configure B tensor
    auto bTensor = matmulAttributes.get_b();
    bTensor->set_uid(2)
        .set_name("BTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({8, 16})
        .set_stride({16, 1});

    // Configure C tensor (output)
    auto cTensor = matmulAttributes.get_c();
    cTensor->set_uid(3)
        .set_name("CTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({4, 16})
        .set_stride({16, 1});

    // Verify A tensor attributes
    EXPECT_EQ(aTensor->get_uid(), 1);
    EXPECT_EQ(aTensor->get_name(), "ATensor");
    EXPECT_EQ(aTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(aTensor->get_dim(), (std::vector<int64_t>{4, 8}));
    EXPECT_EQ(aTensor->get_stride(), (std::vector<int64_t>{8, 1}));

    // Verify B tensor attributes
    EXPECT_EQ(bTensor->get_uid(), 2);
    EXPECT_EQ(bTensor->get_name(), "BTensor");
    EXPECT_EQ(bTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(bTensor->get_dim(), (std::vector<int64_t>{8, 16}));
    EXPECT_EQ(bTensor->get_stride(), (std::vector<int64_t>{16, 1}));

    // Verify C tensor attributes
    EXPECT_EQ(cTensor->get_uid(), 3);
    EXPECT_EQ(cTensor->get_name(), "CTensor");
    EXPECT_EQ(cTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(cTensor->get_dim(), (std::vector<int64_t>{4, 16}));
    EXPECT_EQ(cTensor->get_stride(), (std::vector<int64_t>{16, 1}));
}

TEST(TestMatmulAttributes, DefaultValues)
{
    const hipdnn_frontend::graph::MatmulAttributes matmulAttributes;

    // Check that tensors are null by default
    EXPECT_EQ(matmulAttributes.get_a(), nullptr);
    EXPECT_EQ(matmulAttributes.get_b(), nullptr);
    EXPECT_EQ(matmulAttributes.get_c(), nullptr);
}

TEST(TestMatmulAttributes, SetAMove)
{
    hipdnn_frontend::graph::MatmulAttributes matmulAttributes;

    auto aTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    aTensor->set_uid(10)
        .set_name("MovedATensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({2, 3})
        .set_stride({3, 1});

    // Store the raw pointer before moving
    auto rawPtr = aTensor.get();

    matmulAttributes.set_a(std::move(aTensor));

    // After move, dwTensor should be nullptr
    EXPECT_EQ(aTensor, nullptr);

    // The moved tensor should be accessible through get_a()
    auto retrievedTensor = matmulAttributes.get_a();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestMatmulAttributes, SetBMove)
{
    hipdnn_frontend::graph::MatmulAttributes matmulAttributes;

    auto bTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    bTensor->set_uid(20)
        .set_name("MovedBTensor")
        .set_data_type(hipdnn_frontend::DataType::HALF)
        .set_dim({3, 4})
        .set_stride({4, 1});

    // Store the raw pointer before moving
    auto rawPtr = bTensor.get();

    matmulAttributes.set_b(std::move(bTensor));

    // After move, dwTensor should be nullptr
    EXPECT_EQ(bTensor, nullptr);

    // The moved tensor should be accessible through get_b()
    auto retrievedTensor = matmulAttributes.get_b();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestMatmulAttributes, SetCMove)
{
    hipdnn_frontend::graph::MatmulAttributes matmulAttributes;

    auto cTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    cTensor->set_uid(30)
        .set_name("MovedCTensor")
        .set_data_type(hipdnn_frontend::DataType::BFLOAT16)
        .set_dim({2, 4})
        .set_stride({4, 1});

    // Store the raw pointer before moving
    auto rawPtr = cTensor.get();

    matmulAttributes.set_c(std::move(cTensor));

    // After move, dwTensor should be nullptr
    EXPECT_EQ(cTensor, nullptr);

    // The moved tensor should be accessible through get_c()
    auto retrievedTensor = matmulAttributes.get_c();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestMatmulAttributes, SetTensorsConstRef)
{
    hipdnn_frontend::graph::MatmulAttributes matmulAttributes;

    // Create tensors
    auto aTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    aTensor->set_uid(40).set_name("AConstRef");

    auto bTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    bTensor->set_uid(50).set_name("BConstRef");

    auto cTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    cTensor->set_uid(60).set_name("CConstRef");

    // Set using const reference (copy)
    matmulAttributes.set_a(aTensor);
    matmulAttributes.set_b(bTensor);
    matmulAttributes.set_c(cTensor);

    // Original tensors should still be valid
    EXPECT_NE(aTensor, nullptr);
    EXPECT_NE(bTensor, nullptr);
    EXPECT_NE(cTensor, nullptr);

    // Getters should return shared_ptrs that point to the same underlying object
    EXPECT_EQ(matmulAttributes.get_a(), aTensor);
    EXPECT_EQ(matmulAttributes.get_b(), bTensor);
    EXPECT_EQ(matmulAttributes.get_c(), cTensor);
}

TEST(TestMatmulAttributes, LogicalAndStrictEquality)
{
    hipdnn_frontend::graph::MatmulAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    auto a1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    a1->set_uid(1).set_name("A").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_a(a1);

    auto b1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    b1->set_uid(2).set_name("B").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_b(b1);

    auto c1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    c1->set_uid(3).set_name("C").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_c(c1);

    hipdnn_frontend::graph::MatmulAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    auto a2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    a2->set_uid(1).set_name("A").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_a(a2);

    auto b2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    b2->set_uid(2).set_name("B").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_b(b2);

    auto c2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    c2->set_uid(3).set_name("C").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_c(c2);

    // Initial check: everything matches exactly
    EXPECT_TRUE(attr1 == attr2);
    EXPECT_FALSE(attr1 != attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));

    // Structural tensor mismatch: different UID/name/type entirely
    auto structuralMismatchA = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    structuralMismatchA->set_uid(99).set_name("MismatchedA");
    attr2.set_a(structuralMismatchA);

    EXPECT_TRUE(attr1 != attr2);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2)); // Structural/type gap implies logical inequality
    attr2.set_a(a2); // Revert

    // Change metadata (UID/Name) on a tensor while keeping mathematical layout intact
    auto logicalMatchA = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    logicalMatchA
        ->set_uid(555) // Diverges from attr1's a1 (uid: 1)
        .set_name("DIVERGENT_NAME") // Diverges from attr1's a1 ("A")
        .set_data_type(hipdnn_frontend::DataType::FLOAT); // Layout matches
    attr2.set_a(logicalMatchA);

    // Expecting: strict evaluation fails, but functional logical comparison passes
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));

    // Sanity check on the trivial hooks themselves: since MatmulAttributes has
    // no extra fields, two default-constructed instances (no tensors, no name,
    // NOT_SET compute type) should still be logically and strictly equal.
    const hipdnn_frontend::graph::MatmulAttributes emptyAttr1;
    const hipdnn_frontend::graph::MatmulAttributes emptyAttr2;
    EXPECT_TRUE(emptyAttr1 == emptyAttr2);
    EXPECT_TRUE(emptyAttr1.logicallyEquals(emptyAttr2));
}
