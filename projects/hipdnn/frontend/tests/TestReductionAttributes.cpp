// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/ReductionAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestReductionAttributes, SetGetMode)
{
    ReductionAttributes attrs;
    EXPECT_FALSE(attrs.get_mode().has_value());

    attrs.set_mode(ReductionMode::ADD);
    EXPECT_TRUE(attrs.get_mode().has_value());
    EXPECT_EQ(attrs.get_mode().value(), ReductionMode::ADD);
}

TEST(TestReductionAttributes, SetGetAllModes)
{
    const std::vector<ReductionMode> modes = {ReductionMode::ADD,
                                              ReductionMode::MUL,
                                              ReductionMode::MIN,
                                              ReductionMode::MAX,
                                              ReductionMode::AMAX,
                                              ReductionMode::AVG,
                                              ReductionMode::NORM1,
                                              ReductionMode::NORM2,
                                              ReductionMode::MUL_NO_ZEROS};

    for(auto mode : modes)
    {
        ReductionAttributes attrs;
        attrs.set_mode(mode);
        EXPECT_EQ(attrs.get_mode().value(), mode);
    }
}

TEST(TestReductionAttributes, SetGetXTensor)
{
    ReductionAttributes attrs;
    EXPECT_EQ(attrs.get_x(), nullptr);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 4})
        .set_stride({4, 1});

    attrs.set_x(x);

    auto retrieved = attrs.get_x();
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->get_uid(), 1);
    EXPECT_EQ(retrieved->get_name(), "InputTensor");
}

TEST(TestReductionAttributes, SetGetYTensor)
{
    ReductionAttributes attrs;
    EXPECT_EQ(attrs.get_y(), nullptr);

    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 4})
        .set_stride({4, 1});

    attrs.set_y(y);

    auto retrieved = attrs.get_y();
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->get_uid(), 2);
    EXPECT_EQ(retrieved->get_name(), "OutputTensor");
}

TEST(TestReductionAttributes, SetXWithMove)
{
    ReductionAttributes attrs;
    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_name("Input");
    auto rawPtr = x.get();

    attrs.set_x(std::move(x));

    EXPECT_EQ(x, nullptr);
    EXPECT_EQ(attrs.get_x().get(), rawPtr);
}

TEST(TestReductionAttributes, SetYWithMove)
{
    ReductionAttributes attrs;
    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(2).set_name("Output");
    auto rawPtr = y.get();

    attrs.set_y(std::move(y));

    EXPECT_EQ(y, nullptr);
    EXPECT_EQ(attrs.get_y().get(), rawPtr);
}

TEST(TestReductionAttributes, IsDeterministicDefaultsFalse)
{
    const ReductionAttributes attrs;
    EXPECT_FALSE(attrs.get_is_deterministic());
}

TEST(TestReductionAttributes, SetGetIsDeterministic)
{
    ReductionAttributes attrs;
    attrs.set_is_deterministic(true);
    EXPECT_TRUE(attrs.get_is_deterministic());

    attrs.set_is_deterministic(false);
    EXPECT_FALSE(attrs.get_is_deterministic());
}

TEST(TestReductionAttributes, ReductionAttributesTypedefExists)
{
    // Verify typedef alias exists and works
    Reduction_attributes attrs;
    attrs.set_mode(ReductionMode::ADD);
    EXPECT_EQ(attrs.get_mode().value(), ReductionMode::ADD);
}

TEST(TestReductionAttributes, LogicalAndStrictEquality)
{
    hipdnn_frontend::graph::ReductionAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_mode(hipdnn_frontend::ReductionMode::ADD);
    attr1.set_is_deterministic(true);

    auto x1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    x1->set_uid(1).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_x(x1);

    auto y1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    y1->set_uid(2).set_name("Y").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_y(y1);

    hipdnn_frontend::graph::ReductionAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_mode(hipdnn_frontend::ReductionMode::ADD);
    attr2.set_is_deterministic(true);

    auto x2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    x2->set_uid(1).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_x(x2);

    auto y2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    y2->set_uid(2).set_name("Y").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_y(y2);

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

    // Mode mismatch: semantic, must fail both checks
    attr2.set_mode(hipdnn_frontend::ReductionMode::AMAX);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_mode(hipdnn_frontend::ReductionMode::ADD); // Revert

    // is_deterministic mismatch: semantic, must fail both checks
    attr2.set_is_deterministic(false);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_is_deterministic(true); // Revert

    // Unset-vs-unset mode: two attrs that never called set_mode should
    // still compare equal
    hipdnn_frontend::graph::ReductionAttributes sparse1;
    sparse1.set_is_deterministic(false);
    hipdnn_frontend::graph::ReductionAttributes sparse2;
    sparse2.set_is_deterministic(false);
    EXPECT_TRUE(sparse1 == sparse2);
    EXPECT_TRUE(sparse1.logicallyEquals(sparse2));

    // Set-vs-unset mode should differ
    sparse2.set_mode(hipdnn_frontend::ReductionMode::MUL);
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
