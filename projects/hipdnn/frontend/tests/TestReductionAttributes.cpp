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
