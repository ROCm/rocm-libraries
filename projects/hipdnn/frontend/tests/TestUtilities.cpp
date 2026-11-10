// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_frontend::detail;
using namespace hipdnn_test_sdk::utilities;

TEST(TestUtilities, FindCommonShapeValid)
{
    const std::vector<std::vector<int64_t>> inputShapes = {{1, 2, 3}, {1, 2, 1}, {1, 1, 3}};
    std::vector<int64_t> commonShape;

    const auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(commonShape, (std::vector<int64_t>{1, 2, 3}));
}

TEST(TestUtilities, FindCommonShapeEmptyInput)
{
    const std::vector<std::vector<int64_t>> inputShapes = {};
    std::vector<int64_t> commonShape;

    const auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestUtilities, FindCommonShapeIncompatibleShapes)
{
    const std::vector<std::vector<int64_t>> inputShapes = {{1, 2, 3}, {1, 2, 4}, {1, 2}};
    std::vector<int64_t> commonShape;

    const auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestUtilities, FindCommonShapeSingleInput)
{
    const std::vector<std::vector<int64_t>> inputShapes = {{1, 2, 3}};
    std::vector<int64_t> commonShape;

    const auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(commonShape, (std::vector<int64_t>{1, 2, 3}));
}

// ============================================================================
// Batch Normalization Validation Tests
// ============================================================================

// Tests for isBatchNormSpatialMode()
TEST(TestUtilities, IsBatchNormSpatialModeNullTensor)
{
    const std::shared_ptr<TensorAttributes> nullScale = nullptr;
    EXPECT_TRUE(isBatchNormSpatialMode(nullScale));
}

TEST(TestUtilities, IsBatchNormSpatialModeUninitializedDims)
{
    const auto scale = std::make_shared<TensorAttributes>();
    // Empty dimensions - should default to spatial mode
    EXPECT_TRUE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModeDimsSizeLessThanTwo)
{
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({32}); // Only 1 dimension
    EXPECT_TRUE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModeSpatialMode2D)
{
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 1, 1});
    EXPECT_TRUE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModeSpatialMode2DLargeChannels)
{
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 256, 1, 1});
    EXPECT_TRUE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModeSpatialMode3D)
{
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 1, 1, 1});
    EXPECT_TRUE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModeSpatialMode3DLargeChannels)
{
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 256, 1, 1, 1});
    EXPECT_TRUE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModePerActivation2DHGreaterThanOne)
{
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 14, 14}); // H and W > 1
    EXPECT_FALSE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModePerActivation2DWGreaterThanOne)
{
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 1, 14}); // Only W > 1
    EXPECT_FALSE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModePerActivation2DHOnlyGreaterThanOne)
{
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 14, 1}); // Only H > 1
    EXPECT_FALSE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModePerActivation3DDGreaterThanOne)
{
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 2, 1, 1}); // D > 1
    EXPECT_FALSE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModePerActivation3DMultipleSpatialDims)
{
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 1, 14, 14}); // H and W > 1
    EXPECT_FALSE(isBatchNormSpatialMode(scale));
}

// Tests for validateBatchNormTrainingSpatialDimensions()
TEST(TestUtilities, ValidateBNTrainingSpatialDimsNullXTensor)
{
    const std::shared_ptr<TensorAttributes> nullX = nullptr;
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 3, 1, 1});

    const auto error = validateBatchNormTrainingSpatialDimensions(nullX, scale);
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("Input tensor is not set") != std::string::npos);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsNullScaleTensor)
{
    const auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 3, 14, 14});
    const std::shared_ptr<TensorAttributes> nullScale = nullptr;

    const auto error = validateBatchNormTrainingSpatialDimensions(x, nullScale);
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("Scale tensor is not set") != std::string::npos);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsXDimsLessThanTwo)
{
    const auto x = std::make_shared<TensorAttributes>();
    x->set_dim({32}); // Only 1 dimension
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 1, 1});

    const auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Input tensor must have at least 2 dimensions")
                != std::string::npos);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsValid2DLargeNHW)
{
    const auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 3, 2, 2}); // N*H*W = 2*2*2 = 8
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 3, 1, 1});

    const auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsValid2DEdgeCase)
{
    const auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 3, 1, 2}); // N*H*W = 1*1*2 = 2 (just above threshold)
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 3, 1, 1});

    const auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsValid2DTypicalCase)
{
    const auto x = std::make_shared<TensorAttributes>();
    x->set_dim({8, 32, 7, 7}); // N*H*W = 8*7*7 = 392
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 1, 1});

    const auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsValid3D)
{
    const auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 3, 2, 2, 2}); // N*D*H*W = 2*2*2*2 = 16
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 3, 1, 1, 1});

    const auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsInvalidSingleElement)
{
    const auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 256, 1, 1}); // N*H*W = 1*1*1 = 1 (invalid!)
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 256, 1, 1});

    const auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("N * spatial_dimensions must be > 1")
                != std::string::npos);
    EXPECT_TRUE(error.get_message().find("N=1") != std::string::npos);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsInvalidSingleElement3D)
{
    const auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 64, 1, 1, 1}); // N*D*H*W = 1*1*1*1 = 1 (invalid!)
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 64, 1, 1, 1});

    const auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("N * spatial_dimensions must be > 1")
                != std::string::npos);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsPerActivationNotSupported)
{
    const auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 3, 14, 14});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 3, 14, 14}); // Per-activation mode (H, W match input)

    const auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("per-activation mode is not currently supported")
                != std::string::npos);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsCustomOperationName)
{
    const auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 32, 1, 1}); // Invalid: N*H*W = 1
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 1, 1});

    const auto error
        = validateBatchNormTrainingSpatialDimensions(x, scale, "Batch normalization backward");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Batch normalization backward") != std::string::npos);
}

// Layout-specific tests to ensure validation is layout-agnostic
TEST(TestUtilities, ValidateBNTrainingSpatialDimsNhwc)
{
    const auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 3, 14, 14}); // NCHW: N*H*W = 2*14*14 = 392
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 3, 1, 1}); // Always channel-first

    const auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsNhwcInvalid)
{
    const auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 256, 1, 1}); // NCHW: N*H*W = 1*1*1 = 1 (invalid!)
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 256, 1, 1});

    const auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("N * spatial_dimensions must be > 1")
                != std::string::npos);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsNDhwc)
{
    const auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 3, 2, 2, 2}); // NCDHW: N*D*H*W = 2*2*2*2 = 16
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 3, 1, 1, 1});

    const auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsNDhwcInvalid)
{
    const auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 256, 1, 1, 1}); // NCDHW: N*D*H*W = 1*1*1*1 = 1 (invalid!)
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 256, 1, 1, 1});

    const auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("N * spatial_dimensions must be > 1")
                != std::string::npos);
}

// ============================================================================
// validateMinimumTensorDimensions Tests
// ============================================================================

TEST(TestUtilities, ValidateMinimumTensorDimensionsNullTensor)
{
    const std::shared_ptr<TensorAttributes> nullTensor = nullptr;
    const auto error = validateMinimumTensorDimensions(nullTensor, 2, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("TestTensor is not set") != std::string::npos);
}

TEST(TestUtilities, ValidateMinimumTensorDimensionsEmptyDims)
{
    const auto tensor = std::make_shared<TensorAttributes>();
    // Empty dimensions
    const auto error = validateMinimumTensorDimensions(tensor, 2, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must have at least 2 dimensions") != std::string::npos);
    EXPECT_TRUE(error.get_message().find("but has 0") != std::string::npos);
}

TEST(TestUtilities, ValidateMinimumTensorDimensionsValid2DMeetsRequirement)
{
    const auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({2, 64});
    const auto error = validateMinimumTensorDimensions(tensor, 2, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateMinimumTensorDimensionsValid4DExceedsRequirement)
{
    const auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({2, 64, 32, 32});
    const auto error = validateMinimumTensorDimensions(tensor, 2, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateMinimumTensorDimensionsInvalid1DRequires2D)
{
    const auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({64});
    const auto error = validateMinimumTensorDimensions(tensor, 2, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must have at least 2 dimensions") != std::string::npos);
    EXPECT_TRUE(error.get_message().find("but has 1") != std::string::npos);
}

TEST(TestUtilities, ValidateMinimumTensorDimensionsInvalid2DRequires4D)
{
    const auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({2, 64});
    const auto error = validateMinimumTensorDimensions(tensor, 4, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must have at least 4 dimensions") != std::string::npos);
    EXPECT_TRUE(error.get_message().find("but has 2") != std::string::npos);
}

// ============================================================================
// validateTensorShapesMatch Tests
// ============================================================================

TEST(TestUtilities, ValidateTensorShapesMatchNullTensor1)
{
    const std::shared_ptr<TensorAttributes> nullTensor = nullptr;
    const auto tensor2 = std::make_shared<TensorAttributes>();
    tensor2->set_dim({2, 64});

    const auto error = validateTensorShapesMatch(nullTensor, tensor2, "Tensor1", "Tensor2");
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("Tensor1 is not set") != std::string::npos);
}

TEST(TestUtilities, ValidateTensorShapesMatchNullTensor2)
{
    const auto tensor1 = std::make_shared<TensorAttributes>();
    tensor1->set_dim({2, 64});
    const std::shared_ptr<TensorAttributes> nullTensor = nullptr;

    const auto error = validateTensorShapesMatch(tensor1, nullTensor, "Tensor1", "Tensor2");
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("Tensor2 is not set") != std::string::npos);
}

TEST(TestUtilities, ValidateTensorShapesMatchEmptyDims1)
{
    const auto tensor1 = std::make_shared<TensorAttributes>();
    const auto tensor2 = std::make_shared<TensorAttributes>();
    tensor2->set_dim({2, 64});

    const auto error = validateTensorShapesMatch(tensor1, tensor2, "Tensor1", "Tensor2");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must have the same number of dimensions")
                != std::string::npos);
    EXPECT_TRUE(error.get_message().find("0 vs 2") != std::string::npos);
}

TEST(TestUtilities, ValidateTensorShapesMatchEmptyDims2)
{
    const auto tensor1 = std::make_shared<TensorAttributes>();
    tensor1->set_dim({2, 64});
    const auto tensor2 = std::make_shared<TensorAttributes>();

    const auto error = validateTensorShapesMatch(tensor1, tensor2, "Tensor1", "Tensor2");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must have the same number of dimensions")
                != std::string::npos);
    EXPECT_TRUE(error.get_message().find("2 vs 0") != std::string::npos);
}

TEST(TestUtilities, ValidateTensorShapesMatchValid)
{
    const auto tensor1 = std::make_shared<TensorAttributes>();
    tensor1->set_dim({2, 64, 32, 32});
    const auto tensor2 = std::make_shared<TensorAttributes>();
    tensor2->set_dim({2, 64, 32, 32});

    const auto error = validateTensorShapesMatch(tensor1, tensor2, "Tensor1", "Tensor2");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateTensorShapesMatchDifferentDimCount)
{
    const auto tensor1 = std::make_shared<TensorAttributes>();
    tensor1->set_dim({2, 64, 32, 32});
    const auto tensor2 = std::make_shared<TensorAttributes>();
    tensor2->set_dim({2, 64});

    const auto error = validateTensorShapesMatch(tensor1, tensor2, "Tensor1", "Tensor2");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must have the same number of dimensions")
                != std::string::npos);
    EXPECT_TRUE(error.get_message().find("4 vs 2") != std::string::npos);
}

TEST(TestUtilities, ValidateTensorShapesMatchDifferentDimValues)
{
    const auto tensor1 = std::make_shared<TensorAttributes>();
    tensor1->set_dim({2, 64, 32, 32});
    const auto tensor2 = std::make_shared<TensorAttributes>();
    tensor2->set_dim({2, 64, 16, 16});

    const auto error = validateTensorShapesMatch(tensor1, tensor2, "Tensor1", "Tensor2");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("dimension mismatch") != std::string::npos);
}

// ============================================================================
// validateChannelOnlyTensorShape Tests
// ============================================================================

TEST(TestUtilities, ValidateChannelOnlyTensorShapeNullTensor)
{
    const std::shared_ptr<TensorAttributes> nullTensor = nullptr;
    const auto error = validateChannelOnlyTensorShape(nullTensor, 64, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("TestTensor is not set") != std::string::npos);
}

TEST(TestUtilities, ValidateChannelOnlyTensorShapeEmptyDims)
{
    const auto tensor = std::make_shared<TensorAttributes>();
    const auto error = validateChannelOnlyTensorShape(tensor, 64, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must have at least 2 dimensions") != std::string::npos);
}

TEST(TestUtilities, ValidateChannelOnlyTensorShapeValid4D)
{
    const auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({1, 64, 1, 1});
    const auto error = validateChannelOnlyTensorShape(tensor, 64, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateChannelOnlyTensorShapeValid5D)
{
    const auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({1, 256, 1, 1, 1});
    const auto error = validateChannelOnlyTensorShape(tensor, 256, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateChannelOnlyTensorShapeInvalidBatchDim)
{
    const auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({2, 64, 1, 1}); // Batch should be 1
    const auto error = validateChannelOnlyTensorShape(tensor, 64, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("batch dimension") != std::string::npos);
}

TEST(TestUtilities, ValidateChannelOnlyTensorShapeInvalidChannelDim)
{
    const auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({1, 128, 1, 1}); // Channel should be 64
    const auto error = validateChannelOnlyTensorShape(tensor, 64, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("channel dimension") != std::string::npos);
}

TEST(TestUtilities, ValidateChannelOnlyTensorShapeInvalidSpatialDim)
{
    const auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({1, 64, 32, 32}); // Spatial should be 1
    const auto error = validateChannelOnlyTensorShape(tensor, 64, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("spatial dimension") != std::string::npos);
}

TEST(TestUtilities, ValidateChannelOnlyTensorShapeInvalidLessThan2D)
{
    const auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({64}); // Only 1D
    const auto error = validateChannelOnlyTensorShape(tensor, 64, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must have at least 2 dimensions") != std::string::npos);
}

// ============================================================================
// validateScaleNormalizedShape Tests
// ============================================================================

TEST(TestUtilities, ValidateScaleNormalizedShapeFullRank4)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({1, 2, 3, 3});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 2, 3, 3});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeTrailingHW)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({1, 2, 3, 3});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 1, 3, 3});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeTrailingW)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({1, 2, 3, 3});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 1, 1, 3});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeRank3Full)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({1, 4, 8});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 4, 8});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeRank3TrailingW)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({1, 4, 8});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 1, 8});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeRank5TrailingHW)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({1, 2, 2, 3, 3});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 1, 1, 3, 3});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeRejectsPerChannelCanonical)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({1, 2, 3, 3});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 2, 1, 1});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("no trailing dims matching input") != std::string::npos);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeRejectsTrailingOneAfterNonOne)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({1, 2, 3, 3});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 2, 3, 1});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("no trailing dims matching input") != std::string::npos);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeRejectsOneGapInMiddle)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({1, 2, 3, 3});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 2, 1, 3});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("leading region before normalized shape")
                != std::string::npos);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeRejectsNonTrailingMiddleDim)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({1, 2, 3, 3});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 1, 3, 1});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("no trailing dims matching input") != std::string::npos);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeRejectsAllOnes)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({1, 2, 3, 3});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 1, 1, 1});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("no trailing dims matching input") != std::string::npos);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeAcceptsAllOnesRank4)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({1, 1, 1, 1});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 1, 1, 1});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeAcceptsAllOnesInputRank4WithBatch)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({2, 1, 1, 1});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 1, 1, 1});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeAcceptsAllOnesInputRank3)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({3, 1, 1});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 1, 1});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeRejectsAllOnesScaleWithTrailingNonOneInput)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({2, 1, 1, 4});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 1, 1, 1});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("no trailing dims matching input") != std::string::npos);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeRejectsScaleRankGreaterThanInput)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({1, 2, 3, 3});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 1, 2, 3, 3});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must match input rank") != std::string::npos);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeRejectsRank3Rows)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({1, 6, 3});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 6, 1});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("no trailing dims matching input") != std::string::npos);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeRejectsRank5Gap)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({1, 2, 2, 3, 3});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 2, 1, 3, 3});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("leading region before normalized shape")
                != std::string::npos);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeRejectsBatchNotOne)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({2, 2, 3, 3});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({2, 2, 3, 3});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("batch dimension (index 0) must be 1")
                != std::string::npos);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeRejectsMismatchedNonOneDim)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({1, 2, 3, 3});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 5, 3, 3});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("leading region before normalized shape")
                != std::string::npos);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeRejectsRankMismatch)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({1, 2, 3, 3});
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 2, 3});
    const auto error = validateScaleNormalizedShape(scale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must match input rank") != std::string::npos);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeRejectsNullScale)
{
    const auto input = std::make_shared<TensorAttributes>();
    input->set_dim({1, 2, 3, 3});
    const std::shared_ptr<TensorAttributes> nullScale = nullptr;
    const auto error = validateScaleNormalizedShape(nullScale, input, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("is not set") != std::string::npos);
}

TEST(TestUtilities, ValidateScaleNormalizedShapeRejectsNullInput)
{
    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 2, 3, 3});
    const std::shared_ptr<TensorAttributes> nullInput = nullptr;
    const auto error = validateScaleNormalizedShape(scale, nullInput, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("is not set") != std::string::npos);
}

// ============================================================================
// validateScalarParameter Tests
// ============================================================================

TEST(TestUtilities, ValidateScalarParameterNullTensor)
{
    const std::shared_ptr<TensorAttributes> nullParam = nullptr;
    const auto error = validateScalarParameter(nullParam, "TestParameter");
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("TestParameter parameter is not set")
                != std::string::npos);
}

TEST(TestUtilities, ValidateScalarParameterEmptyDims)
{
    const auto param = std::make_shared<TensorAttributes>();
    const auto error = validateScalarParameter(param, "TestParameter");
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("dimensions are not set") != std::string::npos);
}

TEST(TestUtilities, ValidateScalarParameterValidSingleElement1D)
{
    const auto param = std::make_shared<TensorAttributes>();
    param->set_dim({1});
    param->set_value(1.0f);
    const auto error = validateScalarParameter(param, "TestParameter");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateScalarParameterValidSingleElement4D)
{
    const auto param = std::make_shared<TensorAttributes>();
    param->set_dim({1, 1, 1, 1});
    param->set_value(1.0f);
    const auto error = validateScalarParameter(param, "TestParameter");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateScalarParameterInvalidNotPassByValue)
{
    const auto param = std::make_shared<TensorAttributes>();
    param->set_dim({1});
    // No value set
    const auto error = validateScalarParameter(param, "TestParameter");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must be a pass-by-value tensor") != std::string::npos);
}

TEST(TestUtilities, ValidateScalarParameterInvalidNonScalar)
{
    const auto param = std::make_shared<TensorAttributes>();
    param->set_dim({2, 3});
    const auto error = validateScalarParameter(param, "TestParameter");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must be a scalar") != std::string::npos);
    EXPECT_TRUE(error.get_message().find("but has 6 elements") != std::string::npos);
}

TEST(TestUtilities, ValidateScalarParameterInvalidMultipleElements)
{
    const auto param = std::make_shared<TensorAttributes>();
    param->set_dim({1, 64, 1, 1});
    const auto error = validateScalarParameter(param, "TestParameter");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must be a scalar") != std::string::npos);
    EXPECT_TRUE(error.get_message().find("but has 64 elements") != std::string::npos);
}
