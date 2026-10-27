// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GpuResampleFwdRefTestFixture.hpp"

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::resample;
using namespace hipdnn_gpu_ref;
using namespace gpu_resample_ref_test;
using namespace gpu_resample_fwd_ref_test;

using HalfType = hipdnn_data_sdk::types::half;
using BFloat16Type = hipdnn_data_sdk::types::bfloat16;

// --- Valid configurations ---

TEST(TestGpuResampleFwdRefValidation, AcceptsValidParamsTensorDims)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x4D({2, 4, 8, 8});
    Tensor<float> y4D({2, 4, 4, 4});

    EXPECT_NO_THROW(GpuFpReferenceResample::forward<float>(
        x4D, y4D, {0, 0}, {0, 0}, {2, 2}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD));

    Tensor<float> x5D({2, 4, 8, 8, 8});
    Tensor<float> y5D({2, 4, 4, 4, 4});

    EXPECT_NO_THROW(GpuFpReferenceResample::forward<float>(x5D,
                                                           y5D,
                                                           {0, 0, 0},
                                                           {0, 0, 0},
                                                           {2, 2, 2},
                                                           {2, 2, 2},
                                                           ResampleMode::MAXPOOL,
                                                           PaddingMode::ZERO_PAD));
}

TEST(TestGpuResampleFwdRefValidation, AcceptsValidParamsChannelLastLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x4D({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> y4D({2, 4, 4, 4}, TensorLayout::NHWC);

    EXPECT_NO_THROW(GpuFpReferenceResample::forward<float>(
        x4D, y4D, {0, 0}, {0, 0}, {2, 2}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::NEG_INF_PAD));

    Tensor<float> x5D({2, 4, 8, 8, 8}, TensorLayout::NDHWC);
    Tensor<float> y5D({2, 4, 4, 4, 4}, TensorLayout::NDHWC);

    EXPECT_NO_THROW(GpuFpReferenceResample::forward<float>(x5D,
                                                           y5D,
                                                           {0, 0, 0},
                                                           {0, 0, 0},
                                                           {2, 2, 2},
                                                           {2, 2, 2},
                                                           ResampleMode::MAXPOOL,
                                                           PaddingMode::NEG_INF_PAD));
}

TEST(TestGpuResampleFwdRefValidation, AcceptsValidParamsWithIndex)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x4D({2, 4, 8, 8});
    Tensor<float> y4D({2, 4, 4, 4});
    Tensor<int32_t> index4D({2, 4, 4, 4});

    EXPECT_NO_THROW(GpuFpReferenceResample::forward<float>(x4D,
                                                           y4D,
                                                           {0, 0},
                                                           {0, 0},
                                                           {2, 2},
                                                           {2, 2},
                                                           ResampleMode::MAXPOOL,
                                                           PaddingMode::ZERO_PAD,
                                                           &index4D));

    Tensor<float> x5D({2, 4, 8, 8, 8});
    Tensor<float> y5D({2, 4, 4, 4, 4});
    Tensor<int32_t> index5D({2, 4, 4, 4, 4});

    EXPECT_NO_THROW(GpuFpReferenceResample::forward<float>(x5D,
                                                           y5D,
                                                           {0, 0, 0},
                                                           {0, 0, 0},
                                                           {2, 2, 2},
                                                           {2, 2, 2},
                                                           ResampleMode::MAXPOOL,
                                                           PaddingMode::ZERO_PAD,
                                                           &index5D));
}

TEST(TestGpuResampleFwdRefValidation, AcceptsValidParamsWithPadding)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x4D({2, 4, 8, 8});
    Tensor<float> y4D({2, 4, 4, 4});

    EXPECT_NO_THROW(GpuFpReferenceResample::forward<float>(x4D,
                                                           y4D,
                                                           {1, 1},
                                                           {1, 1},
                                                           {2, 2},
                                                           {3, 3},
                                                           ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                           PaddingMode::NEG_INF_PAD));

    Tensor<float> x5D({2, 4, 8, 8, 8});
    Tensor<float> y5D({2, 4, 4, 4, 4});

    EXPECT_NO_THROW(GpuFpReferenceResample::forward<float>(x5D,
                                                           y5D,
                                                           {1, 1, 1},
                                                           {1, 1, 1},
                                                           {2, 2, 2},
                                                           {3, 3, 3},
                                                           ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                           PaddingMode::NEG_INF_PAD));
}

TEST(TestGpuResampleFwdRefValidation, AcceptsValidParamsAsymmetricPadding)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x4D({2, 4, 8, 6});
    Tensor<float> y4D({2, 4, 4, 6});

    EXPECT_NO_THROW(GpuFpReferenceResample::forward<float>(x4D,
                                                           y4D,
                                                           {1, 0},
                                                           {0, 1},
                                                           {2, 1},
                                                           {3, 2},
                                                           ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                           PaddingMode::ZERO_PAD));

    Tensor<float> x5D({2, 3, 6, 8, 10});
    Tensor<float> y5D({2, 3, 6, 4, 3});

    EXPECT_NO_THROW(GpuFpReferenceResample::forward<float>(x5D,
                                                           y5D,
                                                           {0, 1, 1},
                                                           {1, 1, 0},
                                                           {1, 2, 3},
                                                           {2, 3, 4},
                                                           ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                           PaddingMode::ZERO_PAD));
}

TEST(TestGpuResampleFwdRefValidation, AcceptsValidParamsGlobalWindow)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x4D({2, 4, 8, 8});
    Tensor<float> y4D({2, 4, 1, 1});

    EXPECT_NO_THROW(GpuFpReferenceResample::forward<float>(x4D,
                                                           y4D,
                                                           {0, 0},
                                                           {0, 0},
                                                           {1, 1},
                                                           {8, 8},
                                                           ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                           PaddingMode::NEG_INF_PAD));

    Tensor<float> x5D({2, 4, 8, 8, 8});
    Tensor<float> y5D({2, 4, 1, 1, 1});

    EXPECT_NO_THROW(GpuFpReferenceResample::forward<float>(x5D,
                                                           y5D,
                                                           {0, 0, 0},
                                                           {0, 0, 0},
                                                           {1, 1, 1},
                                                           {8, 8, 8},
                                                           ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                           PaddingMode::NEG_INF_PAD));
}

TEST(TestGpuResampleFwdRefValidation, AcceptsValidParamsUnitWindowUnitStride)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x4D({2, 4, 8, 8});
    Tensor<float> y4D({2, 4, 8, 8});

    EXPECT_NO_THROW(GpuFpReferenceResample::forward<float>(x4D,
                                                           y4D,
                                                           {0, 0},
                                                           {0, 0},
                                                           {1, 1},
                                                           {1, 1},
                                                           ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                           PaddingMode::ZERO_PAD));

    Tensor<float> x5D({2, 4, 8, 8, 8});
    Tensor<float> y5D({2, 4, 8, 8, 8});

    EXPECT_NO_THROW(GpuFpReferenceResample::forward<float>(x5D,
                                                           y5D,
                                                           {0, 0, 0},
                                                           {0, 0, 0},
                                                           {1, 1, 1},
                                                           {1, 1, 1},
                                                           ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                           PaddingMode::ZERO_PAD));
}

// --- validateInput() throw paths ---

TEST(TestGpuResampleFwdRefValidation, ThrowsOnInputRankTooSmall)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 8});
    Tensor<float> y({4, 4, 4});

    EXPECT_THROW(
        GpuFpReferenceResample::forward<float>(
            x, y, {0}, {0}, {2}, {2}, ResampleMode::AVGPOOL_INCLUDE_PADDING, PaddingMode::ZERO_PAD),
        std::runtime_error);
}

TEST(TestGpuResampleFwdRefValidation, ThrowsOnInputRankTooLarge)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8, 8, 8});
    Tensor<float> y({2, 4, 4, 4, 4, 4});

    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0, 0, 0},
                                                        {0, 0, 0, 0},
                                                        {2, 2, 2, 2},
                                                        {2, 2, 2, 2},
                                                        ResampleMode::MAXPOOL,
                                                        PaddingMode::ZERO_PAD),
                 std::runtime_error);
}

TEST(TestGpuResampleFwdRefValidation, ThrowsOnOutputRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({2, 4, 4, 4, 4});

    EXPECT_THROW(
        GpuFpReferenceResample::forward<float>(
            x, y, {0, 0}, {0, 0}, {2, 2}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD),
        std::runtime_error);
}

TEST(TestGpuResampleFwdRefValidation, ThrowsOnBatchDimMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({3, 4, 4, 4});

    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0},
                                                        {0, 0},
                                                        {2, 2},
                                                        {2, 2},
                                                        ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                        PaddingMode::NEG_INF_PAD),
                 std::runtime_error);
}

TEST(TestGpuResampleFwdRefValidation, ThrowsOnChannelDimMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({2, 5, 4, 4});

    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0},
                                                        {0, 0},
                                                        {2, 2},
                                                        {2, 2},
                                                        ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                        PaddingMode::NEG_INF_PAD),
                 std::runtime_error);
}

TEST(TestGpuResampleFwdRefValidation, ThrowsOnIOTensorNeitherChannelFirstNorChannelLast)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x4D({2, 4, 8, 8}, std::vector<int64_t>{1, 2, 3, 4});
    Tensor<float> y4D({2, 4, 4, 4});

    EXPECT_THROW(
        GpuFpReferenceResample::forward<float>(
            x4D, y4D, {0, 0}, {0, 0}, {2, 2}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD),
        std::invalid_argument);

    Tensor<float> x({2, 4, 8, 8, 8});
    Tensor<float> y({2, 4, 4, 4, 4}, std::vector<int64_t>{1, 2, 3, 4, 5});

    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0, 0},
                                                        {0, 0, 0},
                                                        {2, 2, 2},
                                                        {2, 2, 2},
                                                        ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                        PaddingMode::NEG_INF_PAD),
                 std::invalid_argument);
}

TEST(TestGpuResampleFwdRefValidation, ThrowsOnIOTensorLayoutsInconsistent)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x4D({2, 4, 8, 8}, TensorLayout::NCHW);
    Tensor<float> y4D({2, 4, 4, 4}, TensorLayout::NHWC);

    EXPECT_THROW(
        GpuFpReferenceResample::forward<float>(
            x4D, y4D, {0, 0}, {0, 0}, {2, 2}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD),
        std::invalid_argument);

    Tensor<float> x5D({2, 4, 8, 8, 8}, TensorLayout::NDHWC);
    Tensor<float> y5D({2, 4, 4, 4, 4}, TensorLayout::NCDHW);

    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x5D,
                                                        y5D,
                                                        {0, 0, 0},
                                                        {0, 0, 0},
                                                        {2, 2, 2},
                                                        {2, 2, 2},
                                                        ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                        PaddingMode::NEG_INF_PAD),
                 std::invalid_argument);
}

TEST(TestGpuResampleFwdRefValidation, ThrowsOnSpatialParamRankMismatch4D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({2, 4, 4, 4});

    EXPECT_THROW(
        GpuFpReferenceResample::forward<float>(
            x, y, {0}, {0, 0}, {2, 2}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD),
        std::runtime_error);
    EXPECT_THROW(
        GpuFpReferenceResample::forward<float>(
            x, y, {0, 0}, {0, 0}, {2}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD),
        std::runtime_error);
    EXPECT_THROW(
        GpuFpReferenceResample::forward<float>(
            x, y, {0, 0}, {0, 0}, {2, 2}, {2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD),
        std::runtime_error);
}

TEST(TestGpuResampleFwdRefValidation, ThrowsOnSpatialParamRankMismatch5D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8, 8});
    Tensor<float> y({2, 4, 4, 4, 4});

    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0},
                                                        {0, 0, 0},
                                                        {2, 2, 2},
                                                        {2, 2, 2},
                                                        ResampleMode::MAXPOOL,
                                                        PaddingMode::ZERO_PAD),
                 std::runtime_error);
    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0, 0},
                                                        {0, 0, 0},
                                                        {2, 2},
                                                        {2, 2, 2},
                                                        ResampleMode::MAXPOOL,
                                                        PaddingMode::ZERO_PAD),
                 std::runtime_error);
    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0, 0},
                                                        {0, 0, 0},
                                                        {2, 2, 2},
                                                        {2, 2},
                                                        ResampleMode::MAXPOOL,
                                                        PaddingMode::ZERO_PAD),
                 std::runtime_error);
}

TEST(TestGpuResampleFwdRefValidation, ThrowsOnEmptySpatialParams)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({2, 4, 4, 4});

    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {},
                                                        {0, 0},
                                                        {2, 2},
                                                        {2, 2},
                                                        ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                        PaddingMode::NEG_INF_PAD),
                 std::runtime_error);
    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0},
                                                        {0, 0},
                                                        {},
                                                        {2, 2},
                                                        ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                        PaddingMode::NEG_INF_PAD),
                 std::runtime_error);
    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0},
                                                        {0, 0},
                                                        {2, 2},
                                                        {},
                                                        ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                        PaddingMode::NEG_INF_PAD),
                 std::runtime_error);
}

TEST(TestGpuResampleFwdRefValidation, ThrowsOnInvalidSpatialParams)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({2, 4, 4, 4});

    EXPECT_THROW(
        GpuFpReferenceResample::forward<float>(
            x, y, {0, -1}, {0, 0}, {2, 2}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD),
        std::runtime_error);
    EXPECT_THROW(
        GpuFpReferenceResample::forward<float>(
            x, y, {0, 0}, {0, 0}, {-2, 2}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD),
        std::runtime_error);
    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0},
                                                        {0, 0},
                                                        {0, 2},
                                                        {2, 2},
                                                        ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                        PaddingMode::NEG_INF_PAD),
                 std::runtime_error);
    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0},
                                                        {0, 0},
                                                        {2, 2},
                                                        {2, 0},
                                                        ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                        PaddingMode::ZERO_PAD),
                 std::runtime_error);
    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0},
                                                        {0, 0},
                                                        {2, 2},
                                                        {2, -2},
                                                        ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                        PaddingMode::ZERO_PAD),
                 std::runtime_error);
}

TEST(TestGpuResampleFwdRefValidation, ThrowsOnUnsupportedResampleMode)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({2, 4, 4, 4});

    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0},
                                                        {0, 0},
                                                        {2, 2},
                                                        {2, 2},
                                                        static_cast<ResampleMode>(100),
                                                        PaddingMode::ZERO_PAD),
                 std::invalid_argument);
}

TEST(TestGpuResampleFwdRefValidation, ThrowsOnUnsupportedPaddingMode)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({2, 4, 4, 4});

    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0},
                                                        {0, 0},
                                                        {2, 2},
                                                        {2, 2},
                                                        ResampleMode::MAXPOOL,
                                                        static_cast<PaddingMode>(100)),
                 std::invalid_argument);
}

TEST(TestGpuResampleFwdRefValidation, ThrowsOnIndexTensorRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({2, 4, 4, 4});
    Tensor<int32_t> index({2, 4, 4});

    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0},
                                                        {0, 0},
                                                        {2, 2},
                                                        {2, 2},
                                                        ResampleMode::MAXPOOL,
                                                        PaddingMode::ZERO_PAD,
                                                        &index),
                 std::invalid_argument);
}

TEST(TestGpuResampleFwdRefValidation, ThrowsOnIndexTensorShapeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({2, 4, 4, 4});
    Tensor<int32_t> index({2, 4, 5, 4});

    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0},
                                                        {0, 0},
                                                        {2, 2},
                                                        {2, 2},
                                                        ResampleMode::MAXPOOL,
                                                        PaddingMode::ZERO_PAD,
                                                        &index),
                 std::invalid_argument);
}

TEST(TestGpuResampleFwdRefValidation, ThrowsOnIndexTensorLayoutMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({2, 4, 4, 4});
    Tensor<int32_t> index({2, 4, 4, 4}, TensorLayout::NHWC);

    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0},
                                                        {0, 0},
                                                        {2, 2},
                                                        {2, 2},
                                                        ResampleMode::MAXPOOL,
                                                        PaddingMode::ZERO_PAD,
                                                        &index),
                 std::invalid_argument);
}

TEST(TestGpuResampleFwdRefValidation, ThrowsOnIndexTensorInputForAvgPool)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({2, 4, 4, 4});
    Tensor<int32_t> index({2, 4, 4, 4});

    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0},
                                                        {0, 0},
                                                        {2, 2},
                                                        {2, 2},
                                                        ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                        PaddingMode::ZERO_PAD,
                                                        &index),
                 std::invalid_argument);
}

TEST(TestGpuResampleFwdRefValidation, ThrowsOnWindowSizeExceedsPaddedInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({2, 4, 1, 1});

    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0},
                                                        {0, 0},
                                                        {2, 2},
                                                        {9, 9},
                                                        ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                        PaddingMode::ZERO_PAD),
                 std::runtime_error);
}

TEST(TestGpuResampleFwdRefValidation, ThrowsOnPaddingSizeExceedsWindowSize)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({2, 4, 4, 4});
    Tensor<int32_t> index({2, 4, 4, 4});

    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {3, 1},
                                                        {0, 0},
                                                        {2, 2},
                                                        {2, 2},
                                                        ResampleMode::MAXPOOL,
                                                        PaddingMode::NEG_INF_PAD,
                                                        &index),
                 std::runtime_error);

    EXPECT_THROW(GpuFpReferenceResample::forward<float>(x,
                                                        y,
                                                        {0, 0},
                                                        {1, 3},
                                                        {2, 2},
                                                        {2, 2},
                                                        ResampleMode::MAXPOOL,
                                                        PaddingMode::NEG_INF_PAD,
                                                        &index),
                 std::runtime_error);
}

// --- Mixed type tests ---

TEST(TestGpuResampleFwdRefMixedType, FloatInputHalfOutput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({2, 3, 8, 8});
    Tensor<HalfType> yCpu({2, 3, 4, 4});
    Tensor<HalfType> yGpu({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(xTensor, -1.0f, 1.0f, seed);
    xTensor.memory().hostData();

    CpuFpReferenceResampleFwd::forward<float, HalfType, float, int32_t>(
        xTensor,
        yCpu,
        {0, 0},
        {2, 2},
        {2, 2},
        ResampleMode::AVGPOOL_EXCLUDE_PADDING,
        PaddingMode::ZERO_PAD);

    GpuFpReferenceResample::forward<float, HalfType, float, int32_t>(
        xTensor,
        yGpu,
        {0, 0},
        {0, 0},
        {2, 2},
        {2, 2},
        ResampleMode::AVGPOOL_EXCLUDE_PADDING,
        PaddingMode::ZERO_PAD);

    assertAllClose(yCpu, yGpu, getTolerance<HalfType>());
}

TEST(TestGpuResampleFwdRefMixedType, HalfInputFloatOutput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<HalfType> xTensor({2, 3, 8, 8});
    Tensor<float> yCpu({2, 3, 4, 4});
    Tensor<float> yGpu({2, 3, 4, 4});
    Tensor<int32_t> indexCpu({2, 3, 4, 4});
    Tensor<int32_t> indexGpu({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(xTensor, static_cast<HalfType>(-1.0f), static_cast<HalfType>(1.0f), seed);
    xTensor.memory().hostData();

    CpuFpReferenceResampleFwd::forward<HalfType, float, float, int32_t>(xTensor,
                                                                        yCpu,
                                                                        {0, 0},
                                                                        {2, 2},
                                                                        {2, 2},
                                                                        ResampleMode::MAXPOOL,
                                                                        PaddingMode::ZERO_PAD,
                                                                        &indexCpu);

    GpuFpReferenceResample::forward<HalfType, float, float, int32_t>(xTensor,
                                                                     yGpu,
                                                                     {0, 0},
                                                                     {0, 0},
                                                                     {2, 2},
                                                                     {2, 2},
                                                                     ResampleMode::MAXPOOL,
                                                                     PaddingMode::ZERO_PAD,
                                                                     &indexGpu);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
    assertAllExact(indexCpu, indexGpu);
}

TEST(TestGpuResampleFwdRefMixedType, HalfInputHalfOutput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<HalfType> xTensor({2, 3, 8, 8});
    Tensor<HalfType> yCpu({2, 3, 4, 4});
    Tensor<HalfType> yGpu({2, 3, 4, 4});
    Tensor<int32_t> indexCpu({2, 3, 4, 4});
    Tensor<int32_t> indexGpu({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(xTensor, static_cast<HalfType>(-1.0f), static_cast<HalfType>(1.0f), seed);
    xTensor.memory().hostData();

    CpuFpReferenceResampleFwd::forward<HalfType, HalfType, float, int32_t>(xTensor,
                                                                           yCpu,
                                                                           {0, 0},
                                                                           {2, 2},
                                                                           {2, 2},
                                                                           ResampleMode::MAXPOOL,
                                                                           PaddingMode::NEG_INF_PAD,
                                                                           &indexCpu);

    GpuFpReferenceResample::forward<HalfType, HalfType, float, int32_t>(xTensor,
                                                                        yGpu,
                                                                        {0, 0},
                                                                        {0, 0},
                                                                        {2, 2},
                                                                        {2, 2},
                                                                        ResampleMode::MAXPOOL,
                                                                        PaddingMode::NEG_INF_PAD,
                                                                        &indexGpu);

    assertAllClose(yCpu, yGpu, getTolerance<HalfType>());
    assertAllExact(indexCpu, indexGpu);
}

TEST(TestGpuResampleFwdRefMixedType, BfloatInputFloatOutput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<BFloat16Type> xTensor({2, 3, 8, 8});
    Tensor<float> yCpu({2, 3, 4, 4});
    Tensor<float> yGpu({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(
        xTensor, static_cast<BFloat16Type>(-1.0f), static_cast<BFloat16Type>(1.0f), seed);
    xTensor.memory().hostData();

    CpuFpReferenceResampleFwd::forward<BFloat16Type, float, float, int32_t>(
        xTensor,
        yCpu,
        {0, 0},
        {2, 2},
        {2, 2},
        ResampleMode::AVGPOOL_INCLUDE_PADDING,
        PaddingMode::ZERO_PAD);

    GpuFpReferenceResample::forward<BFloat16Type, float, float, int32_t>(
        xTensor,
        yGpu,
        {0, 0},
        {0, 0},
        {2, 2},
        {2, 2},
        ResampleMode::AVGPOOL_INCLUDE_PADDING,
        PaddingMode::ZERO_PAD);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

TEST(TestGpuResampleFwdRefMixedType, BfloatInputHalfOutput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<BFloat16Type> xTensor({2, 3, 8, 8});
    Tensor<HalfType> yCpu({2, 3, 4, 4});
    Tensor<HalfType> yGpu({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(
        xTensor, static_cast<BFloat16Type>(-1.0f), static_cast<BFloat16Type>(1.0f), seed);
    xTensor.memory().hostData();

    CpuFpReferenceResampleFwd::forward<BFloat16Type, HalfType, float, int32_t>(
        xTensor,
        yCpu,
        {0, 0},
        {2, 2},
        {2, 2},
        ResampleMode::AVGPOOL_EXCLUDE_PADDING,
        PaddingMode::NEG_INF_PAD);

    GpuFpReferenceResample::forward<BFloat16Type, HalfType, float, int32_t>(
        xTensor,
        yGpu,
        {0, 0},
        {0, 0},
        {2, 2},
        {2, 2},
        ResampleMode::AVGPOOL_EXCLUDE_PADDING,
        PaddingMode::NEG_INF_PAD);

    assertAllClose(yCpu, yGpu, getTolerance<HalfType>());
}

TEST(TestGpuResampleFwdRefMixedType, DoubleInputDoubleOutput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<double> xTensor({2, 3, 8, 8});
    Tensor<double> yCpu({2, 3, 4, 4});
    Tensor<double> yGpu({2, 3, 4, 4});
    Tensor<int32_t> indexCpu({2, 3, 4, 4});
    Tensor<int32_t> indexGpu({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(xTensor, -1.0, 1.0, seed);
    xTensor.memory().hostData();

    CpuFpReferenceResampleFwd::forward<double, double, double, int32_t>(xTensor,
                                                                        yCpu,
                                                                        {0, 0},
                                                                        {2, 2},
                                                                        {2, 2},
                                                                        ResampleMode::MAXPOOL,
                                                                        PaddingMode::NEG_INF_PAD,
                                                                        &indexCpu);

    GpuFpReferenceResample::forward<double, double, double, int32_t>(xTensor,
                                                                     yGpu,
                                                                     {0, 0},
                                                                     {0, 0},
                                                                     {2, 2},
                                                                     {2, 2},
                                                                     ResampleMode::MAXPOOL,
                                                                     PaddingMode::NEG_INF_PAD,
                                                                     &indexGpu);

    assertAllClose(yCpu, yGpu, getTolerance<double>());
    assertAllExact(indexCpu, indexGpu);
}

// --- Optional arguments tests ---

TEST(TestGpuResampleFwdRefOptionalArgs, MaxPoolWithIndex)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({2, 3, 4, 5, 6});
    Tensor<float> yCpu({2, 3, 3, 3, 3});
    Tensor<float> yGpu({2, 3, 3, 3, 3});
    Tensor<int32_t> indexCpu({2, 3, 3, 3, 3});
    Tensor<int32_t> indexGpu({2, 3, 3, 3, 3});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(xTensor, -1.0f, 1.0f, seed);
    xTensor.memory().hostData();

    CpuFpReferenceResampleFwd::forward<float, float, float, int32_t>(xTensor,
                                                                     yCpu,
                                                                     {1, 2, 1},
                                                                     {1, 2, 2},
                                                                     {3, 4, 3},
                                                                     ResampleMode::MAXPOOL,
                                                                     PaddingMode::ZERO_PAD,
                                                                     &indexCpu);
    GpuFpReferenceResample::forward<float, float, float, int32_t>(xTensor,
                                                                  yGpu,
                                                                  {1, 2, 1},
                                                                  {0, 1, 0},
                                                                  {1, 2, 2},
                                                                  {3, 4, 3},
                                                                  ResampleMode::MAXPOOL,
                                                                  PaddingMode::ZERO_PAD,
                                                                  &indexGpu);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
    assertAllExact(indexCpu, indexGpu);
}

TEST(TestGpuResampleFwdRefOptionalArgs, AverageExcludePadding)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({2, 3, 6, 6});
    Tensor<float> yCpu({2, 3, 3, 3});
    Tensor<float> yGpu({2, 3, 3, 3});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(xTensor, -1.0f, 1.0f, seed);
    xTensor.memory().hostData();

    CpuFpReferenceResampleFwd::forward<float, float, float>(xTensor,
                                                            yCpu,
                                                            {2, 2},
                                                            {2, 2},
                                                            {4, 4},
                                                            ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                            PaddingMode::NEG_INF_PAD);
    GpuFpReferenceResample::forward<float, float, float>(xTensor,
                                                         yGpu,
                                                         {2, 2},
                                                         {0, 0},
                                                         {2, 2},
                                                         {4, 4},
                                                         ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                         PaddingMode::NEG_INF_PAD);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

TEST(TestGpuResampleFwdRefOptionalArgs, AvgPoolIncludePadding)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({2, 3, 6, 6});
    Tensor<float> yCpu({2, 3, 4, 3});
    Tensor<float> yGpu({2, 3, 4, 3});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(xTensor, -1.0f, 1.0f, seed);
    xTensor.memory().hostData();

    CpuFpReferenceResampleFwd::forward<float, float, float>(xTensor,
                                                            yCpu,
                                                            {2, 2},
                                                            {2, 2},
                                                            {4, 4},
                                                            ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                            PaddingMode::ZERO_PAD);
    GpuFpReferenceResample::forward<float, float, float>(xTensor,
                                                         yGpu,
                                                         {2, 2},
                                                         {2, 0},
                                                         {2, 2},
                                                         {4, 4},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

// --- Channel-last layout tests ---

TEST(TestGpuResampleFwdRefChannelLast, AvgPoolMatchesCpuRef4D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> xTensor({2, 4, 9, 7}, TensorLayout::NHWC);
    Tensor<float> yCpu({2, 4, 5, 7}, TensorLayout::NHWC);
    Tensor<float> yGpu({2, 4, 5, 7}, TensorLayout::NHWC);

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(xTensor, -1.0f, 1.0f, seed);
    xTensor.memory().hostData();

    CpuFpReferenceResampleFwd::forward<float, float, float>(xTensor,
                                                            yCpu,
                                                            {1, 1},
                                                            {2, 1},
                                                            {3, 3},
                                                            ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                            PaddingMode::ZERO_PAD);
    GpuFpReferenceResample::forward<float, float, float>(xTensor,
                                                         yGpu,
                                                         {1, 1},
                                                         {1, 1},
                                                         {2, 1},
                                                         {3, 3},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

TEST(TestGpuResampleFwdRefChannelLast, MaxPoolMatchesCpuRefWithIndex4D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> xTensor({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> yCpu({2, 4, 4, 4}, TensorLayout::NHWC);
    Tensor<float> yGpu({2, 4, 4, 4}, TensorLayout::NHWC);
    Tensor<int32_t> indexCpu({2, 4, 4, 4}, TensorLayout::NHWC);
    Tensor<int32_t> indexGpu({2, 4, 4, 4}, TensorLayout::NHWC);

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(xTensor, -1.0f, 1.0f, seed);
    xTensor.memory().hostData();

    CpuFpReferenceResampleFwd::forward<float, float, float>(xTensor,
                                                            yCpu,
                                                            {0, 0},
                                                            {2, 2},
                                                            {2, 2},
                                                            ResampleMode::MAXPOOL,
                                                            PaddingMode::ZERO_PAD,
                                                            &indexCpu);
    GpuFpReferenceResample::forward<float, float, float>(xTensor,
                                                         yGpu,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 2},
                                                         ResampleMode::MAXPOOL,
                                                         PaddingMode::ZERO_PAD,
                                                         &indexGpu);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
    assertAllExact(indexCpu, indexGpu);
}

TEST(TestGpuResampleFwdRefChannelLast, AvgPoolMatchesCpuRef5D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> xTensor({2, 4, 8, 8, 8}, TensorLayout::NDHWC);
    Tensor<float> yCpu({2, 4, 4, 4, 4}, TensorLayout::NDHWC);
    Tensor<float> yGpu({2, 4, 4, 4, 4}, TensorLayout::NDHWC);

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(xTensor, -1.0f, 1.0f, seed);
    xTensor.memory().hostData();

    CpuFpReferenceResampleFwd::forward<float, float, float>(xTensor,
                                                            yCpu,
                                                            {1, 1, 1},
                                                            {2, 2, 2},
                                                            {3, 3, 3},
                                                            ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                            PaddingMode::NEG_INF_PAD);
    GpuFpReferenceResample::forward<float, float, float>(xTensor,
                                                         yGpu,
                                                         {1, 1, 1},
                                                         {1, 1, 1},
                                                         {2, 2, 2},
                                                         {3, 3, 3},
                                                         ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                         PaddingMode::NEG_INF_PAD);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

TEST(TestGpuResampleFwdRefChannelLast, MaxPoolMatchesCpuRefWithIndex5D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> xTensor({2, 4, 8, 8, 8}, TensorLayout::NDHWC);
    Tensor<float> yCpu({2, 4, 4, 4, 4}, TensorLayout::NDHWC);
    Tensor<float> yGpu({2, 4, 4, 4, 4}, TensorLayout::NDHWC);
    Tensor<int32_t> indexCpu({2, 4, 4, 4, 4}, TensorLayout::NDHWC);
    Tensor<int32_t> indexGpu({2, 4, 4, 4, 4}, TensorLayout::NDHWC);

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(xTensor, -1.0f, 1.0f, seed);
    xTensor.memory().hostData();

    CpuFpReferenceResampleFwd::forward<float, float, float>(xTensor,
                                                            yCpu,
                                                            {0, 0, 0},
                                                            {2, 2, 2},
                                                            {2, 2, 2},
                                                            ResampleMode::MAXPOOL,
                                                            PaddingMode::ZERO_PAD,
                                                            &indexCpu);
    GpuFpReferenceResample::forward<float, float, float>(xTensor,
                                                         yGpu,
                                                         {0, 0, 0},
                                                         {0, 0, 0},
                                                         {2, 2, 2},
                                                         {2, 2, 2},
                                                         ResampleMode::MAXPOOL,
                                                         PaddingMode::ZERO_PAD,
                                                         &indexGpu);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
    assertAllExact(indexCpu, indexGpu);
}

// --- Test suite instantiations ---

using TestGpuResampleFwdRef4DFp32 = ResampleFwdTestSuite<float>;
using TestGpuResampleFwdRef4DFp16 = ResampleFwdTestSuite<HalfType>;
using TestGpuResampleFwdRef4DBfp16 = ResampleFwdTestSuite<BFloat16Type>;
using TestGpuResampleFwdRef5DFp32 = ResampleFwdTestSuite<float>;
using TestGpuResampleFwdRef5DFp16 = ResampleFwdTestSuite<HalfType>;
using TestGpuResampleFwdRef5DBfp16 = ResampleFwdTestSuite<BFloat16Type>;

TEST_P(TestGpuResampleFwdRef4DFp32, MatchesCpuRef)
{
    this->runResampleFwdTest();
}
TEST_P(TestGpuResampleFwdRef4DFp16, MatchesCpuRef)
{
    this->runResampleFwdTest();
}
TEST_P(TestGpuResampleFwdRef4DBfp16, MatchesCpuRef)
{
    this->runResampleFwdTest();
}
TEST_P(TestGpuResampleFwdRef5DFp32, MatchesCpuRef)
{
    this->runResampleFwdTest();
}
TEST_P(TestGpuResampleFwdRef5DFp16, MatchesCpuRef)
{
    this->runResampleFwdTest();
}
TEST_P(TestGpuResampleFwdRef5DBfp16, MatchesCpuRef)
{
    this->runResampleFwdTest();
}

// ============================================================================
// 4D tests
// ============================================================================

// --- Quick tests ---

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuResampleFwdRef4DFp32,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleSmall4DShapes(),
                                                                   getResample4DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::FORWARD)));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuResampleFwdRef4DFp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleSmall4DShapes(),
                                                                   getResample4DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::FORWARD)));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuResampleFwdRef4DBfp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleSmall4DShapes(),
                                                                   getResample4DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::FORWARD)));

// --- Standard tests ---

INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuResampleFwdRef4DFp32,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleMedium4DShapes(),
                                                                   getResample4DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::FORWARD)));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuResampleFwdRef4DFp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleMedium4DShapes(),
                                                                   getResample4DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::FORWARD)));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuResampleFwdRef4DBfp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleMedium4DShapes(),
                                                                   getResample4DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::FORWARD)));

// --- Comprehensive tests ---

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuResampleFwdRef4DFp32,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleLarge4DShapes(),
                                                                   getResample4DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::FORWARD)));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuResampleFwdRef4DFp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleLarge4DShapes(),
                                                                   getResample4DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::FORWARD)));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuResampleFwdRef4DBfp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleLarge4DShapes(),
                                                                   getResample4DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::FORWARD)));

// ============================================================================
// 5D tests
// ============================================================================

// --- Quick tests ---

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuResampleFwdRef5DFp32,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleSmall5DShapes(),
                                                                   getResample5DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::FORWARD)));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuResampleFwdRef5DFp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleSmall5DShapes(),
                                                                   getResample5DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::FORWARD)));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuResampleFwdRef5DBfp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleSmall5DShapes(),
                                                                   getResample5DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::FORWARD)));

// --- Standard tests ---

INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuResampleFwdRef5DFp32,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleMedium5DShapes(),
                                                                   getResample5DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::FORWARD)));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuResampleFwdRef5DFp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleMedium5DShapes(),
                                                                   getResample5DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::FORWARD)));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuResampleFwdRef5DBfp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleMedium5DShapes(),
                                                                   getResample5DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::FORWARD)));

// --- Comprehensive tests ---

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuResampleFwdRef5DFp32,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleLarge5DShapes(),
                                                                   getResample5DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::FORWARD)));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuResampleFwdRef5DFp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleLarge5DShapes(),
                                                                   getResample5DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::FORWARD)));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuResampleFwdRef5DBfp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleLarge5DShapes(),
                                                                   getResample5DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::FORWARD)));
