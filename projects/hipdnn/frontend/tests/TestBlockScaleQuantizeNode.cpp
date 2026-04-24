// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/BlockScaleQuantizeAttributes.hpp>
#include <hipdnn_frontend/node/BlockScaleQuantizeNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestBlockScaleQuantizeNode, PreValidateNode)
{
    BlockScaleQuantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);

    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(32);

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBlockScaleQuantizeNode, PreValidateNodeMissingX)
{
    BlockScaleQuantizeAttributes attrs;

    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(32);

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestBlockScaleQuantizeNode, PreValidateNodeMissingY)
{
    BlockScaleQuantizeAttributes attrs;

    attrs.set_x(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(32);

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestBlockScaleQuantizeNode, PreValidateNodeMissingScale)
{
    BlockScaleQuantizeAttributes attrs;

    attrs.set_x(std::make_shared<TensorAttributes>());
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_block_size(32);

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestBlockScaleQuantizeNode, PreValidateNodeBlockSizeNotSet)
{
    // block_size is required for quantize
    BlockScaleQuantizeAttributes attrs;

    attrs.set_x(std::make_shared<TensorAttributes>());
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestBlockScaleQuantizeNode, PreValidateNodeBlockSizeZero)
{
    BlockScaleQuantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32});
    attrs.set_x(xTensor);

    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(0);

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestBlockScaleQuantizeNode, PreValidateNodeBlockSizeNegative)
{
    BlockScaleQuantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32});
    attrs.set_x(xTensor);

    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(-1);

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestBlockScaleQuantizeNode, PreValidateNodeYShapeMismatch)
{
    BlockScaleQuantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32});
    attrs.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 64, 16, 16}); // Mismatched dims
    attrs.set_y(yTensor);

    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(32);

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestBlockScaleQuantizeNode, PreValidateNodeYRankMismatch)
{
    BlockScaleQuantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32});
    attrs.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 64, 32}); // Different rank
    attrs.set_y(yTensor);

    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(32);

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestBlockScaleQuantizeNode, PreValidateNodeYDimsNotSetPassesValidation)
{
    // When Y dims are not set, shape match is skipped (will be inferred later)
    BlockScaleQuantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32});
    attrs.set_x(xTensor);

    attrs.set_y(std::make_shared<TensorAttributes>()); // No dims set
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(32);

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBlockScaleQuantizeNode, PreValidateNodeXDimsNotSetSkipsDimChecks)
{
    // When X dims are not set, dimension-dependent checks are skipped
    BlockScaleQuantizeAttributes attrs;

    attrs.set_x(std::make_shared<TensorAttributes>()); // No dims
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(32);

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBlockScaleQuantizeNode, PreValidateNodeAxisNegative)
{
    BlockScaleQuantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);

    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(32);
    attrs.set_axis(-1);

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestBlockScaleQuantizeNode, PreValidateNodeAxisExceedsRank)
{
    BlockScaleQuantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);

    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(32);
    attrs.set_axis(4); // rank is 4, so axis=4 is out of bounds

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestBlockScaleQuantizeNode, PreValidateNodeAxisValid)
{
    BlockScaleQuantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);

    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(32);
    attrs.set_axis(1);

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBlockScaleQuantizeNode, PreValidateNodeAxisWithoutXDimsSkipsCheck)
{
    // When X dims are not set, axis validation is deferred
    BlockScaleQuantizeAttributes attrs;

    attrs.set_x(std::make_shared<TensorAttributes>()); // No dims
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(32);
    attrs.set_axis(100); // Would be invalid if dims were set

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBlockScaleQuantizeNode, PreValidateNodeDimNotDivisibleByBlockSize)
{
    // axis dim must be divisible by block_size
    BlockScaleQuantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);

    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(30); // 64 is not divisible by 30
    attrs.set_axis(1);

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestBlockScaleQuantizeNode, PreValidateNodeDefaultAxisDivisibility)
{
    // When axis is not set, default is last dim. Last dim 32 is divisible by 8.
    BlockScaleQuantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);

    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(8); // 32 % 8 == 0

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBlockScaleQuantizeNode, PreValidateNodeDefaultAxisNotDivisible)
{
    // When axis is not set, default is last dim. Last dim 32 is not divisible by 7.
    BlockScaleQuantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);

    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(7); // 32 % 7 != 0

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestBlockScaleQuantizeNode, InferPropertiesNode)
{
    BlockScaleQuantizeAttributes attrs;
    attrs.set_x(std::make_shared<TensorAttributes>());
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(32);

    auto inputTensor = attrs.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = attrs.get_y();
    outputTensor->set_uid(2).set_name("OutputTensor");

    const GraphAttributes graphAttributes;
    BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(TestBlockScaleQuantizeNode, InferPropertiesNodeMissingX)
{
    BlockScaleQuantizeAttributes attrs;
    attrs.set_y(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestBlockScaleQuantizeNode, InferPropertiesNodeMissingY)
{
    BlockScaleQuantizeAttributes attrs;
    attrs.set_x(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestBlockScaleQuantizeNode, InferPropertiesNodeMissingScale)
{
    BlockScaleQuantizeAttributes attrs;
    attrs.set_x(std::make_shared<TensorAttributes>());
    attrs.set_y(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestBlockScaleQuantizeNode, InferPropertiesTransposedYStridesWithAxis)
{
    BlockScaleQuantizeAttributes attrs;
    attrs.set_block_size(32);
    attrs.set_axis(1);
    attrs.set_transpose(true);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1)
        .set_name("X")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 64, 32, 32})
        .set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(2).set_name("Y");
    attrs.set_y(yTensor);

    attrs.set_scale(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(yTensor->get_dim(), (std::vector<int64_t>{2, 64, 32, 32}));
    // Derivation: sort indices by X strides ascending → [3,2,1,0], rotate axis=1 to front
    // → [1,0,3,2], inverse permutation gives strideOrder=[1,0,3,2], so dim 1 gets stride 1.
    EXPECT_EQ(yTensor->get_stride(), (std::vector<int64_t>{64, 1, 4096, 128}));
}

TEST(TestBlockScaleQuantizeNode, InferPropertiesTransposedScaleStridesWithAxis)
{
    BlockScaleQuantizeAttributes attrs;
    attrs.set_block_size(32);
    attrs.set_axis(1);
    attrs.set_transpose(true);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1)
        .set_name("X")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 64, 32, 32})
        .set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(2).set_name("Y");
    attrs.set_y(yTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(3).set_name("Scale");
    attrs.set_scale(scaleTensor);

    const GraphAttributes graphAttributes;
    BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(scaleTensor->get_dim(), (std::vector<int64_t>{2, 2, 32, 32}));
    EXPECT_EQ(scaleTensor->get_stride(), (std::vector<int64_t>{2, 1, 128, 4}));
}

TEST(TestBlockScaleQuantizeNode, InferPropertiesTransposedDefaultAxis)
{
    BlockScaleQuantizeAttributes attrs;
    attrs.set_block_size(32);
    // No axis set
    attrs.set_transpose(true);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1)
        .set_name("X")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 64, 32, 32})
        .set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(2).set_name("Y");
    attrs.set_y(yTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(3).set_name("Scale");
    attrs.set_scale(scaleTensor);

    const GraphAttributes graphAttributes;
    BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    // Without axis, transpose produces same stride ordering as X
    EXPECT_EQ(yTensor->get_stride(), (std::vector<int64_t>{65536, 1024, 32, 1}));
    // Scale dims: last dim divided by block_size
    EXPECT_EQ(scaleTensor->get_dim(), (std::vector<int64_t>{2, 64, 32, 1}));
    EXPECT_EQ(scaleTensor->get_stride(), (std::vector<int64_t>{2048, 32, 1, 1}));
}

TEST(TestBlockScaleQuantizeNode, InferPropertiesTransposedNhwcInput)
{
    BlockScaleQuantizeAttributes attrs;
    attrs.set_block_size(32);
    attrs.set_axis(3);
    attrs.set_transpose(true);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1)
        .set_name("X")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 64, 32, 32})
        .set_stride({65536, 1, 2048, 64}); // NHWC layout
    attrs.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(2).set_name("Y");
    attrs.set_y(yTensor);

    attrs.set_scale(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(yTensor->get_dim(), (std::vector<int64_t>{2, 64, 32, 32}));
    EXPECT_EQ(yTensor->get_stride(), (std::vector<int64_t>{1024, 2048, 32, 1}));
}

TEST(TestBlockScaleQuantizeNode, InferPropertiesTransposePreservesExistingStrides)
{
    BlockScaleQuantizeAttributes attrs;
    attrs.set_block_size(32);
    attrs.set_axis(1);
    attrs.set_transpose(true);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1)
        .set_name("X")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 64, 32, 32})
        .set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(2)
        .set_name("Y")
        .set_dim({2, 64, 32, 32})
        .set_stride({65536, 1024, 32, 1}); // User-set strides
    attrs.set_y(yTensor);

    attrs.set_scale(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    // Existing strides should not be overwritten
    EXPECT_EQ(yTensor->get_stride(), (std::vector<int64_t>{65536, 1024, 32, 1}));
}

TEST(TestBlockScaleQuantizeNode, InferPropertiesNonTransposeScaleStrides)
{
    BlockScaleQuantizeAttributes attrs;
    attrs.set_block_size(32);
    attrs.set_axis(1);
    // transpose is false by default

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1)
        .set_name("X")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 64, 32, 32})
        .set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(2).set_name("Y");
    attrs.set_y(yTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(3).set_name("Scale");
    attrs.set_scale(scaleTensor);

    const GraphAttributes graphAttributes;
    BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    // Non-transpose: Y gets X strides, scale gets stride order from extractStrideOrder
    EXPECT_EQ(yTensor->get_stride(), (std::vector<int64_t>{65536, 1024, 32, 1}));
    EXPECT_EQ(scaleTensor->get_dim(), (std::vector<int64_t>{2, 2, 32, 32}));
    EXPECT_EQ(scaleTensor->get_stride(), (std::vector<int64_t>{2048, 1024, 32, 1}));
}

TEST(TestBlockScaleQuantizeNode, InferPropertiesYStridesFallbackNoXStrides)
{
    // When X has dims but no strides, Y strides fall back to generateStrides(y->get_dim())
    BlockScaleQuantizeAttributes attrs;
    attrs.set_block_size(32);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1)
        .set_name("X")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 64, 32, 32}); // No strides set
    attrs.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(2).set_name("Y");
    attrs.set_y(yTensor);

    attrs.set_scale(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(yTensor->get_dim(), (std::vector<int64_t>{2, 64, 32, 32}));
    // Default generateStrides produces row-major strides
    EXPECT_EQ(yTensor->get_stride(), (std::vector<int64_t>{65536, 1024, 32, 1}));
}

TEST(TestBlockScaleQuantizeNode, InferPropertiesScaleStridesFallbackNoXStrides)
{
    // When X has dims but no strides, scale strides fall back to generateStrides(scale->get_dim())
    BlockScaleQuantizeAttributes attrs;
    attrs.set_block_size(32);
    attrs.set_axis(1);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1)
        .set_name("X")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 64, 32, 32}); // No strides set
    attrs.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(2).set_name("Y");
    attrs.set_y(yTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(3).set_name("Scale");
    attrs.set_scale(scaleTensor);

    const GraphAttributes graphAttributes;
    BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    // Scale dims inferred: axis=1 → dim[1]/32 = 64/32 = 2
    EXPECT_EQ(scaleTensor->get_dim(), (std::vector<int64_t>{2, 2, 32, 32}));
    // Default generateStrides produces row-major strides
    EXPECT_EQ(scaleTensor->get_stride(), (std::vector<int64_t>{2048, 1024, 32, 1}));
}

TEST(TestBlockScaleQuantizeNode, GatherHipdnnTensors)
{
    BlockScaleQuantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1).set_name("X");
    attrs.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(2).set_name("Y");
    attrs.set_y(yTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(3).set_name("Scale");
    attrs.set_scale(scaleTensor);

    const GraphAttributes graphAttributes;
    const BlockScaleQuantizeNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(yTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(scaleTensor) != allTensors.end());

    EXPECT_EQ(allTensors.size(), 3);
}

TEST(TestBlockScaleQuantizeNode, GetNodeTypeReturnsBlockScaleQuantize)
{
    const GraphAttributes graphAttrs;
    const BlockScaleQuantizeNode node(BlockScaleQuantizeAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::BLOCK_SCALE_QUANTIZE);
}
