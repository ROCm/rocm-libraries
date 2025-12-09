// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_sdk/test_utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_sdk::test_utilities;

TEST(TestUtilities, FindCommonShapeValid)
{
    std::vector<std::vector<int64_t>> inputShapes = {{1, 2, 3}, {1, 2, 1}, {1, 1, 3}};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(commonShape, (std::vector<int64_t>{1, 2, 3}));
}

TEST(TestUtilities, FindCommonShapeEmptyInput)
{
    std::vector<std::vector<int64_t>> inputShapes = {};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestUtilities, FindCommonShapeIncompatibleShapes)
{
    std::vector<std::vector<int64_t>> inputShapes = {{1, 2, 3}, {1, 2, 4}, {1, 2}};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestUtilities, FindCommonShapeSingleInput)
{
    std::vector<std::vector<int64_t>> inputShapes = {{1, 2, 3}};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(commonShape, (std::vector<int64_t>{1, 2, 3}));
}

TEST(TestUtilities, InitializeFrontendLoggingReturnsCorrectly)
{
    ScopedEnvironmentVariableSetter guard("HIPDNN_LOG_LEVEL", "info");

    EXPECT_EQ(hipdnn_frontend::initializeFrontendLogging(nullptr), -1);
    EXPECT_EQ(hipdnn_frontend::initializeFrontendLogging(), 0);
}

// ============================================================================
// Batch Normalization Validation Tests
// ============================================================================

// Tests for isBatchNormSpatialMode()
TEST(TestUtilities, IsBatchNormSpatialModeNullTensor)
{
    std::shared_ptr<TensorAttributes> nullScale = nullptr;
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
    std::shared_ptr<TensorAttributes> nullX = nullptr;
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 3, 1, 1});

    auto error = validateBatchNormTrainingSpatialDimensions(nullX, scale);
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsNullScaleTensor)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 3, 14, 14});
    std::shared_ptr<TensorAttributes> nullScale = nullptr;

    auto error = validateBatchNormTrainingSpatialDimensions(x, nullScale);
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestUtilities, ValidateBNTrainingSpatialDimsXDimsLessThanTwo)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({32}); // Only 1 dimension
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 32, 1, 1});

    auto error = validateBatchNormTrainingSpatialDimensions(x, scale);
    EXPECT_EQ(error.code, ErrorCode::OK);
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
