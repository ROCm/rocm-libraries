// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/ReductionAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/node/ReductionNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{
ReductionAttributes makeValidAttrs()
{
    ReductionAttributes attrs;
    attrs.set_mode(ReductionMode::ADD);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({4, 8})
        .set_stride({8, 1});

    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 8})
        .set_stride({8, 1});

    attrs.set_x(x);
    attrs.set_y(y);
    return attrs;
}
} // namespace

TEST(TestReductionNode, ValidNodePreValidates)
{
    const GraphAttributes graphAttributes;
    const ReductionNode node(makeValidAttrs(), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestReductionNode, MissingXError)
{
    ReductionAttributes attrs;
    attrs.set_mode(ReductionMode::ADD);
    attrs.set_y(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    const ReductionNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestReductionNode, MissingYError)
{
    ReductionAttributes attrs;
    attrs.set_mode(ReductionMode::ADD);
    attrs.set_x(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    const ReductionNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestReductionNode, MissingModeError)
{
    ReductionAttributes attrs;
    attrs.set_x(std::make_shared<TensorAttributes>());
    attrs.set_y(std::make_shared<TensorAttributes>());
    // mode intentionally not set

    const GraphAttributes graphAttributes;
    const ReductionNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestReductionNode, NotSetModeError)
{
    ReductionAttributes attrs;
    attrs.set_mode(ReductionMode::NOT_SET);
    attrs.set_x(std::make_shared<TensorAttributes>());
    attrs.set_y(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    const ReductionNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestReductionNode, InferPropertiesNode)
{
    const GraphAttributes graphAttributes;
    auto attrs = makeValidAttrs();
    ReductionNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

class TestReductionNodeAllModes : public ::testing::TestWithParam<ReductionMode>
{
};

TEST_P(TestReductionNodeAllModes, ValidNodePreValidates)
{
    auto mode = GetParam();

    ReductionAttributes attrs;
    attrs.set_mode(mode);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({4, 8})
        .set_stride({8, 1});

    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 8})
        .set_stride({8, 1});

    attrs.set_x(x);
    attrs.set_y(y);

    const GraphAttributes graphAttributes;
    const ReductionNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK) << "Mode " << static_cast<int>(mode) << " should be valid";
}

INSTANTIATE_TEST_SUITE_P(,
                         TestReductionNodeAllModes,
                         ::testing::Values(ReductionMode::ADD,
                                           ReductionMode::MUL,
                                           ReductionMode::MIN,
                                           ReductionMode::MAX,
                                           ReductionMode::AMAX,
                                           ReductionMode::AVG,
                                           ReductionMode::NORM1,
                                           ReductionMode::NORM2,
                                           ReductionMode::MUL_NO_ZEROS));

TEST(TestReductionNode, RankMismatchError)
{
    ReductionAttributes attrs;
    attrs.set_mode(ReductionMode::ADD);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_data_type(DataType::FLOAT).set_dim({4, 8, 16}).set_stride({128, 16, 1});

    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(2).set_data_type(DataType::FLOAT).set_dim({1, 8}).set_stride({8, 1});

    attrs.set_x(x);
    attrs.set_y(y);

    const GraphAttributes graphAttributes;
    const ReductionNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestReductionNode, OutputDimExceedsInputError)
{
    ReductionAttributes attrs;
    attrs.set_mode(ReductionMode::ADD);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_data_type(DataType::FLOAT).set_dim({4, 8}).set_stride({8, 1});

    // Y dim[1] = 16 > X dim[1] = 8
    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(2).set_data_type(DataType::FLOAT).set_dim({1, 16}).set_stride({16, 1});

    attrs.set_x(x);
    attrs.set_y(y);

    const GraphAttributes graphAttributes;
    const ReductionNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestReductionNode, NoReductionDimError)
{
    ReductionAttributes attrs;
    attrs.set_mode(ReductionMode::ADD);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_data_type(DataType::FLOAT).set_dim({4, 8}).set_stride({8, 1});

    // Y dims identical to X — not a reduction
    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(2).set_data_type(DataType::FLOAT).set_dim({4, 8}).set_stride({8, 1});

    attrs.set_x(x);
    attrs.set_y(y);

    const GraphAttributes graphAttributes;
    const ReductionNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestReductionNode, PartialReductionValid)
{
    ReductionAttributes attrs;
    attrs.set_mode(ReductionMode::ADD);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 8, 16, 64})
        .set_stride({8192, 1024, 64, 1});

    // Reduce last two dims to 1
    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(2).set_data_type(DataType::FLOAT).set_dim({2, 8, 1, 1}).set_stride({8, 1, 1, 1});

    attrs.set_x(x);
    attrs.set_y(y);

    const GraphAttributes graphAttributes;
    const ReductionNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestReductionNode, ReducedDimNotOneError)
{
    ReductionAttributes attrs;
    attrs.set_mode(ReductionMode::ADD);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_data_type(DataType::FLOAT).set_dim({4, 8}).set_stride({8, 1});

    // Y dim[0] = 2 < X dim[0] = 4, but not 1
    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(2).set_data_type(DataType::FLOAT).set_dim({2, 8}).set_stride({8, 1});

    attrs.set_x(x);
    attrs.set_y(y);

    const GraphAttributes graphAttributes;
    const ReductionNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestReductionNode, NoDimsSkipsDimValidation)
{
    // When dims are not set on tensors, dim validation should be skipped
    ReductionAttributes attrs;
    attrs.set_mode(ReductionMode::ADD);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_data_type(DataType::FLOAT);

    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(2).set_data_type(DataType::FLOAT);

    attrs.set_x(x);
    attrs.set_y(y);

    const GraphAttributes graphAttributes;
    const ReductionNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}
