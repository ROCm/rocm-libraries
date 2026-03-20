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

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(commonShape, (std::vector<int64_t>{1, 2, 3}));
}

TEST(TestUtilities, FindCommonShapeEmptyInput)
{
    const std::vector<std::vector<int64_t>> inputShapes = {};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestUtilities, FindCommonShapeIncompatibleShapes)
{
    const std::vector<std::vector<int64_t>> inputShapes = {{1, 2, 3}, {1, 2, 4}, {1, 2}};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestUtilities, FindCommonShapeSingleInput)
{
    const std::vector<std::vector<int64_t>> inputShapes = {{1, 2, 3}};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
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
    auto scale = std::make_shared<TensorAttributes>();
    // Empty dimensions - should default to spatial mode
    EXPECT_TRUE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModeDimsSizeLessThanTwo)
{
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({32}); // Only 1 dimension
    EXPECT_TRUE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModeSpatialMode2D)
{
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 1, 1});
    EXPECT_TRUE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModeSpatialMode2DLargeChannels)
{
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 256, 1, 1});
    EXPECT_TRUE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModeSpatialMode3D)
{
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 1, 1, 1});
    EXPECT_TRUE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModeSpatialMode3DLargeChannels)
{
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 256, 1, 1, 1});
    EXPECT_TRUE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModePerActivation2DHGreaterThanOne)
{
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 14, 14}); // H and W > 1
    EXPECT_FALSE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModePerActivation2DWGreaterThanOne)
{
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 1, 14}); // Only W > 1
    EXPECT_FALSE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModePerActivation2DHOnlyGreaterThanOne)
{
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 14, 1}); // Only H > 1
    EXPECT_FALSE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModePerActivation3DDGreaterThanOne)
{
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 2, 1, 1}); // D > 1
    EXPECT_FALSE(isBatchNormSpatialMode(scale));
}

TEST(TestUtilities, IsBatchNormSpatialModePerActivation3DMultipleSpatialDims)
{
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 1, 14, 14}); // H and W > 1
    EXPECT_FALSE(isBatchNormSpatialMode(scale));
}

// Tests for validateBatchNormTrainingSpatialDimensions()
TEST(TestUtilities, ValidateBNTrainingSpatialDimsNullXTensor)
{
    const std::shared_ptr<TensorAttributes> nullX = nullptr;
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 3, 1, 1});

    auto error = validateBatchNormTrainingSpatialDimensions(nullX, scale);
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("Input tensor is not set") != std::string::npos);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsNullScaleTensor)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 3, 14, 14});
    const std::shared_ptr<TensorAttributes> nullScale = nullptr;

    auto error = validateBatchNormTrainingSpatialDimensions(x, nullScale);
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("Scale tensor is not set") != std::string::npos);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsXDimsLessThanTwo)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({32}); // Only 1 dimension
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 1, 1});

    auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Input tensor must have at least 2 dimensions")
                != std::string::npos);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsValid2DLargeNHW)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 3, 2, 2}); // N*H*W = 2*2*2 = 8
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 3, 1, 1});

    auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsValid2DEdgeCase)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 3, 1, 2}); // N*H*W = 1*1*2 = 2 (just above threshold)
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 3, 1, 1});

    auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsValid2DTypicalCase)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({8, 32, 7, 7}); // N*H*W = 8*7*7 = 392
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 1, 1});

    auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsValid3D)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 3, 2, 2, 2}); // N*D*H*W = 2*2*2*2 = 16
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 3, 1, 1, 1});

    auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsInvalidSingleElement)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 256, 1, 1}); // N*H*W = 1*1*1 = 1 (invalid!)
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 256, 1, 1});

    auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("N * spatial_dimensions must be > 1")
                != std::string::npos);
    EXPECT_TRUE(error.get_message().find("N=1") != std::string::npos);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsInvalidSingleElement3D)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 64, 1, 1, 1}); // N*D*H*W = 1*1*1*1 = 1 (invalid!)
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 64, 1, 1, 1});

    auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("N * spatial_dimensions must be > 1")
                != std::string::npos);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsPerActivationNotSupported)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 3, 14, 14});
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 3, 14, 14}); // Per-activation mode (H, W match input)

    auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("per-activation mode is not currently supported")
                != std::string::npos);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsCustomOperationName)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 32, 1, 1}); // Invalid: N*H*W = 1
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 1, 1});

    auto error
        = validateBatchNormTrainingSpatialDimensions(x, scale, "Batch normalization backward");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Batch normalization backward") != std::string::npos);
}

// Layout-specific tests to ensure validation is layout-agnostic
TEST(TestUtilities, ValidateBNTrainingSpatialDimsNhwc)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 3, 14, 14}); // NCHW: N*H*W = 2*14*14 = 392
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 3, 1, 1}); // Always channel-first

    auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsNhwcInvalid)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 256, 1, 1}); // NCHW: N*H*W = 1*1*1 = 1 (invalid!)
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 256, 1, 1});

    auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("N * spatial_dimensions must be > 1")
                != std::string::npos);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsNDhwc)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 3, 2, 2, 2}); // NCDHW: N*D*H*W = 2*2*2*2 = 16
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 3, 1, 1, 1});

    auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsNDhwcInvalid)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 256, 1, 1, 1}); // NCDHW: N*D*H*W = 1*1*1*1 = 1 (invalid!)
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 256, 1, 1, 1});

    auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
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
    auto error = validateMinimumTensorDimensions(nullTensor, 2, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("TestTensor is not set") != std::string::npos);
}

TEST(TestUtilities, ValidateMinimumTensorDimensionsEmptyDims)
{
    auto tensor = std::make_shared<TensorAttributes>();
    // Empty dimensions
    auto error = validateMinimumTensorDimensions(tensor, 2, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must have at least 2 dimensions") != std::string::npos);
    EXPECT_TRUE(error.get_message().find("but has 0") != std::string::npos);
}

TEST(TestUtilities, ValidateMinimumTensorDimensionsValid2DMeetsRequirement)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({2, 64});
    auto error = validateMinimumTensorDimensions(tensor, 2, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateMinimumTensorDimensionsValid4DExceedsRequirement)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({2, 64, 32, 32});
    auto error = validateMinimumTensorDimensions(tensor, 2, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateMinimumTensorDimensionsInvalid1DRequires2D)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({64});
    auto error = validateMinimumTensorDimensions(tensor, 2, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must have at least 2 dimensions") != std::string::npos);
    EXPECT_TRUE(error.get_message().find("but has 1") != std::string::npos);
}

TEST(TestUtilities, ValidateMinimumTensorDimensionsInvalid2DRequires4D)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({2, 64});
    auto error = validateMinimumTensorDimensions(tensor, 4, "TestTensor");
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
    auto tensor2 = std::make_shared<TensorAttributes>();
    tensor2->set_dim({2, 64});

    auto error = validateTensorShapesMatch(nullTensor, tensor2, "Tensor1", "Tensor2");
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("Tensor1 is not set") != std::string::npos);
}

TEST(TestUtilities, ValidateTensorShapesMatchNullTensor2)
{
    auto tensor1 = std::make_shared<TensorAttributes>();
    tensor1->set_dim({2, 64});
    const std::shared_ptr<TensorAttributes> nullTensor = nullptr;

    auto error = validateTensorShapesMatch(tensor1, nullTensor, "Tensor1", "Tensor2");
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("Tensor2 is not set") != std::string::npos);
}

TEST(TestUtilities, ValidateTensorShapesMatchEmptyDims1)
{
    auto tensor1 = std::make_shared<TensorAttributes>();
    auto tensor2 = std::make_shared<TensorAttributes>();
    tensor2->set_dim({2, 64});

    auto error = validateTensorShapesMatch(tensor1, tensor2, "Tensor1", "Tensor2");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must have the same number of dimensions")
                != std::string::npos);
    EXPECT_TRUE(error.get_message().find("0 vs 2") != std::string::npos);
}

TEST(TestUtilities, ValidateTensorShapesMatchEmptyDims2)
{
    auto tensor1 = std::make_shared<TensorAttributes>();
    tensor1->set_dim({2, 64});
    auto tensor2 = std::make_shared<TensorAttributes>();

    auto error = validateTensorShapesMatch(tensor1, tensor2, "Tensor1", "Tensor2");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must have the same number of dimensions")
                != std::string::npos);
    EXPECT_TRUE(error.get_message().find("2 vs 0") != std::string::npos);
}

TEST(TestUtilities, ValidateTensorShapesMatchValid)
{
    auto tensor1 = std::make_shared<TensorAttributes>();
    tensor1->set_dim({2, 64, 32, 32});
    auto tensor2 = std::make_shared<TensorAttributes>();
    tensor2->set_dim({2, 64, 32, 32});

    auto error = validateTensorShapesMatch(tensor1, tensor2, "Tensor1", "Tensor2");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateTensorShapesMatchDifferentDimCount)
{
    auto tensor1 = std::make_shared<TensorAttributes>();
    tensor1->set_dim({2, 64, 32, 32});
    auto tensor2 = std::make_shared<TensorAttributes>();
    tensor2->set_dim({2, 64});

    auto error = validateTensorShapesMatch(tensor1, tensor2, "Tensor1", "Tensor2");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must have the same number of dimensions")
                != std::string::npos);
    EXPECT_TRUE(error.get_message().find("4 vs 2") != std::string::npos);
}

TEST(TestUtilities, ValidateTensorShapesMatchDifferentDimValues)
{
    auto tensor1 = std::make_shared<TensorAttributes>();
    tensor1->set_dim({2, 64, 32, 32});
    auto tensor2 = std::make_shared<TensorAttributes>();
    tensor2->set_dim({2, 64, 16, 16});

    auto error = validateTensorShapesMatch(tensor1, tensor2, "Tensor1", "Tensor2");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("dimension mismatch") != std::string::npos);
}

// ============================================================================
// validateChannelOnlyTensorShape Tests
// ============================================================================

TEST(TestUtilities, ValidateChannelOnlyTensorShapeNullTensor)
{
    const std::shared_ptr<TensorAttributes> nullTensor = nullptr;
    auto error = validateChannelOnlyTensorShape(nullTensor, 64, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("TestTensor is not set") != std::string::npos);
}

TEST(TestUtilities, ValidateChannelOnlyTensorShapeEmptyDims)
{
    auto tensor = std::make_shared<TensorAttributes>();
    auto error = validateChannelOnlyTensorShape(tensor, 64, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must have at least 2 dimensions") != std::string::npos);
}

TEST(TestUtilities, ValidateChannelOnlyTensorShapeValid4D)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({1, 64, 1, 1});
    auto error = validateChannelOnlyTensorShape(tensor, 64, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateChannelOnlyTensorShapeValid5D)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({1, 256, 1, 1, 1});
    auto error = validateChannelOnlyTensorShape(tensor, 256, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateChannelOnlyTensorShapeInvalidBatchDim)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({2, 64, 1, 1}); // Batch should be 1
    auto error = validateChannelOnlyTensorShape(tensor, 64, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("batch dimension") != std::string::npos);
}

TEST(TestUtilities, ValidateChannelOnlyTensorShapeInvalidChannelDim)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({1, 128, 1, 1}); // Channel should be 64
    auto error = validateChannelOnlyTensorShape(tensor, 64, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("channel dimension") != std::string::npos);
}

TEST(TestUtilities, ValidateChannelOnlyTensorShapeInvalidSpatialDim)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({1, 64, 32, 32}); // Spatial should be 1
    auto error = validateChannelOnlyTensorShape(tensor, 64, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("spatial dimension") != std::string::npos);
}

TEST(TestUtilities, ValidateChannelOnlyTensorShapeInvalidLessThan2D)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({64}); // Only 1D
    auto error = validateChannelOnlyTensorShape(tensor, 64, "TestTensor");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must have at least 2 dimensions") != std::string::npos);
}

// ============================================================================
// validateScalarParameter Tests
// ============================================================================

TEST(TestUtilities, ValidateScalarParameterNullTensor)
{
    const std::shared_ptr<TensorAttributes> nullParam = nullptr;
    auto error = validateScalarParameter(nullParam, "TestParameter");
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("TestParameter parameter is not set")
                != std::string::npos);
}

TEST(TestUtilities, ValidateScalarParameterEmptyDims)
{
    auto param = std::make_shared<TensorAttributes>();
    auto error = validateScalarParameter(param, "TestParameter");
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("dimensions are not set") != std::string::npos);
}

TEST(TestUtilities, ValidateScalarParameterValidSingleElement1D)
{
    auto param = std::make_shared<TensorAttributes>();
    param->set_dim({1});
    param->set_value(1.0f);
    auto error = validateScalarParameter(param, "TestParameter");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateScalarParameterValidSingleElement4D)
{
    auto param = std::make_shared<TensorAttributes>();
    param->set_dim({1, 1, 1, 1});
    param->set_value(1.0f);
    auto error = validateScalarParameter(param, "TestParameter");
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateScalarParameterInvalidNotPassByValue)
{
    auto param = std::make_shared<TensorAttributes>();
    param->set_dim({1});
    // No value set
    auto error = validateScalarParameter(param, "TestParameter");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must be a pass-by-value tensor") != std::string::npos);
}

TEST(TestUtilities, ValidateScalarParameterInvalidNonScalar)
{
    auto param = std::make_shared<TensorAttributes>();
    param->set_dim({2, 3});
    auto error = validateScalarParameter(param, "TestParameter");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must be a scalar") != std::string::npos);
    EXPECT_TRUE(error.get_message().find("but has 6 elements") != std::string::npos);
}

TEST(TestUtilities, ValidateScalarParameterInvalidMultipleElements)
{
    auto param = std::make_shared<TensorAttributes>();
    param->set_dim({1, 64, 1, 1});
    auto error = validateScalarParameter(param, "TestParameter");
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("must be a scalar") != std::string::npos);
    EXPECT_TRUE(error.get_message().find("but has 64 elements") != std::string::npos);
}
