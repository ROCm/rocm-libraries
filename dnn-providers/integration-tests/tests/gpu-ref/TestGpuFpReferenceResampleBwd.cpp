// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GpuResampleBwdRefTestFixture.hpp"

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::resample;
using namespace hipdnn_gpu_ref;
using namespace gpu_resample_ref_test;
using namespace gpu_resample_bwd_ref_test;

using HalfType = hipdnn_data_sdk::types::half;
using BFloat16Type = hipdnn_data_sdk::types::bfloat16;

// --- Valid configurations ---

TEST(TestGpuResampleBwdRefValidation, AcceptsValidParamsTensorDims)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy4D({2, 4, 4, 4});
    Tensor<float> dx4D({2, 4, 8, 8});

    EXPECT_NO_THROW(GpuFpReferenceResample::backward<float>(dy4D,
                                                            dx4D,
                                                            {0, 0},
                                                            {0, 0},
                                                            {2, 2},
                                                            {2, 2},
                                                            ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                            PaddingMode::ZERO_PAD));

    Tensor<float> dy5D({2, 4, 4, 4, 4});
    Tensor<float> dx5D({2, 4, 8, 8, 8});

    EXPECT_NO_THROW(GpuFpReferenceResample::backward<float>(dy5D,
                                                            dx5D,
                                                            {0, 0, 0},
                                                            {0, 0, 0},
                                                            {2, 2, 2},
                                                            {2, 2, 2},
                                                            ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                            PaddingMode::ZERO_PAD));
}

TEST(TestGpuResampleBwdRefValidation, AcceptsValidParamsChannelLastLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy4D({2, 4, 4, 4}, TensorLayout::NHWC);
    Tensor<float> dx4D({2, 4, 8, 8}, TensorLayout::NHWC);

    EXPECT_NO_THROW(GpuFpReferenceResample::backward<float>(dy4D,
                                                            dx4D,
                                                            {0, 0},
                                                            {0, 0},
                                                            {2, 2},
                                                            {2, 2},
                                                            ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                            PaddingMode::NEG_INF_PAD));

    Tensor<float> dy5D({2, 4, 4, 4, 4}, TensorLayout::NDHWC);
    Tensor<float> dx5D({2, 4, 8, 8, 8}, TensorLayout::NDHWC);

    EXPECT_NO_THROW(GpuFpReferenceResample::backward<float>(dy5D,
                                                            dx5D,
                                                            {0, 0, 0},
                                                            {0, 0, 0},
                                                            {2, 2, 2},
                                                            {2, 2, 2},
                                                            ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                            PaddingMode::NEG_INF_PAD));
}

TEST(TestGpuResampleBwdRefValidation, AcceptsValidParamsWithIndex)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy4D({2, 4, 4, 4});
    Tensor<float> dx4D({2, 4, 8, 8});
    Tensor<int32_t> index4D({2, 4, 4, 4});

    EXPECT_NO_THROW(GpuFpReferenceResample::backward<float>(dy4D,
                                                            dx4D,
                                                            {0, 0},
                                                            {0, 0},
                                                            {2, 2},
                                                            {2, 2},
                                                            ResampleMode::MAXPOOL,
                                                            PaddingMode::ZERO_PAD,
                                                            &index4D));

    Tensor<float> dy5D({2, 4, 4, 4, 4});
    Tensor<float> dx5D({2, 4, 8, 8, 8});
    Tensor<int32_t> index5D({2, 4, 4, 4, 4});

    EXPECT_NO_THROW(GpuFpReferenceResample::backward<float>(dy5D,
                                                            dx5D,
                                                            {0, 0, 0},
                                                            {0, 0, 0},
                                                            {2, 2, 2},
                                                            {2, 2, 2},
                                                            ResampleMode::MAXPOOL,
                                                            PaddingMode::ZERO_PAD,
                                                            &index5D));
}

TEST(TestGpuResampleBwdRefValidation, AcceptsValidParamsWithIndexChannelLast)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy4D({2, 4, 4, 4}, TensorLayout::NHWC);
    Tensor<float> dx4D({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<int32_t> index4D({2, 4, 4, 4}, TensorLayout::NHWC);

    EXPECT_NO_THROW(GpuFpReferenceResample::backward<float>(dy4D,
                                                            dx4D,
                                                            {0, 0},
                                                            {0, 0},
                                                            {2, 2},
                                                            {2, 2},
                                                            ResampleMode::MAXPOOL,
                                                            PaddingMode::NEG_INF_PAD,
                                                            &index4D));

    Tensor<float> dy5D({2, 4, 4, 4, 4}, TensorLayout::NDHWC);
    Tensor<float> dx5D({2, 4, 8, 8, 8}, TensorLayout::NDHWC);
    Tensor<int32_t> index5D({2, 4, 4, 4, 4}, TensorLayout::NDHWC);

    EXPECT_NO_THROW(GpuFpReferenceResample::backward<float>(dy5D,
                                                            dx5D,
                                                            {0, 0, 0},
                                                            {0, 0, 0},
                                                            {2, 2, 2},
                                                            {2, 2, 2},
                                                            ResampleMode::MAXPOOL,
                                                            PaddingMode::NEG_INF_PAD,
                                                            &index5D));
}

TEST(TestGpuResampleBwdRefValidation, AcceptsValidParamsWithPadding)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy4D({2, 4, 4, 4});
    Tensor<float> dx4D({2, 4, 8, 8});

    EXPECT_NO_THROW(GpuFpReferenceResample::backward<float>(dy4D,
                                                            dx4D,
                                                            {1, 1},
                                                            {1, 1},
                                                            {2, 2},
                                                            {3, 3},
                                                            ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                            PaddingMode::NEG_INF_PAD));

    Tensor<float> dy5D({2, 4, 4, 4, 4});
    Tensor<float> dx5D({2, 4, 8, 8, 8});

    EXPECT_NO_THROW(GpuFpReferenceResample::backward<float>(dy5D,
                                                            dx5D,
                                                            {1, 1, 1},
                                                            {1, 1, 1},
                                                            {2, 2, 2},
                                                            {3, 3, 3},
                                                            ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                            PaddingMode::NEG_INF_PAD));
}

TEST(TestGpuResampleBwdRefValidation, AcceptsValidParamsAsymmetricPadding)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy4D({2, 4, 4, 6});
    Tensor<float> dx4D({2, 4, 8, 6});

    EXPECT_NO_THROW(GpuFpReferenceResample::backward<float>(dy4D,
                                                            dx4D,
                                                            {1, 0},
                                                            {0, 1},
                                                            {2, 1},
                                                            {3, 2},
                                                            ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                            PaddingMode::ZERO_PAD));

    Tensor<float> dy5D({2, 3, 6, 4, 3});
    Tensor<float> dx5D({2, 3, 6, 8, 10});

    EXPECT_NO_THROW(GpuFpReferenceResample::backward<float>(dy5D,
                                                            dx5D,
                                                            {0, 1, 1},
                                                            {1, 1, 0},
                                                            {1, 2, 3},
                                                            {2, 3, 4},
                                                            ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                            PaddingMode::ZERO_PAD));
}

TEST(TestGpuResampleBwdRefValidation, AcceptsValidParamsGlobalWindow)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy4D({2, 4, 1, 1});
    Tensor<float> dx4D({2, 4, 8, 8});

    EXPECT_NO_THROW(GpuFpReferenceResample::backward<float>(dy4D,
                                                            dx4D,
                                                            {0, 0},
                                                            {0, 0},
                                                            {1, 1},
                                                            {8, 8},
                                                            ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                            PaddingMode::NEG_INF_PAD));

    Tensor<float> dy5D({2, 4, 1, 1, 1});
    Tensor<float> dx5D({2, 4, 8, 8, 8});

    EXPECT_NO_THROW(GpuFpReferenceResample::backward<float>(dy5D,
                                                            dx5D,
                                                            {0, 0, 0},
                                                            {0, 0, 0},
                                                            {1, 1, 1},
                                                            {8, 8, 8},
                                                            ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                            PaddingMode::NEG_INF_PAD));
}

TEST(TestGpuResampleBwdRefValidation, AcceptsValidParamsUnitWindowUnitStride)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy4D({2, 4, 8, 8});
    Tensor<float> dx4D({2, 4, 8, 8});

    EXPECT_NO_THROW(GpuFpReferenceResample::backward<float>(dy4D,
                                                            dx4D,
                                                            {0, 0},
                                                            {0, 0},
                                                            {1, 1},
                                                            {1, 1},
                                                            ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                            PaddingMode::ZERO_PAD));

    Tensor<float> dy5D({2, 4, 8, 8, 8});
    Tensor<float> dx5D({2, 4, 8, 8, 8});

    EXPECT_NO_THROW(GpuFpReferenceResample::backward<float>(dy5D,
                                                            dx5D,
                                                            {0, 0, 0},
                                                            {0, 0, 0},
                                                            {1, 1, 1},
                                                            {1, 1, 1},
                                                            ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                            PaddingMode::ZERO_PAD));
}

// --- validateInput() throw paths ---

TEST(TestGpuResampleBwdRefValidation, ThrowsOnInputRankTooSmall)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({4, 4, 4});
    Tensor<float> dx({4, 8, 8});

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0},
                                                         {0},
                                                         {2},
                                                         {2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::runtime_error);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnInputRankTooLarge)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 4, 4, 4, 4});
    Tensor<float> dx({2, 4, 8, 8, 8, 8});

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0, 0, 0},
                                                         {0, 0, 0, 0},
                                                         {2, 2, 2, 2},
                                                         {2, 2, 2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::runtime_error);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnOutputRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 4, 4, 4});
    Tensor<float> dx({2, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::runtime_error);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnBatchDimMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({3, 4, 4, 4});
    Tensor<float> dx({2, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::NEG_INF_PAD),
                 std::runtime_error);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnChannelDimMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 5, 4, 4});
    Tensor<float> dx({2, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::NEG_INF_PAD),
                 std::runtime_error);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnIOTensorNeitherChannelFirstNorChannelLast)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy4D({2, 4, 4, 4});
    Tensor<float> dx4D({2, 4, 8, 8}, std::vector<int64_t>{1, 2, 3, 4});

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy4D,
                                                         dx4D,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::invalid_argument);

    Tensor<float> dy({2, 4, 4, 4, 4}, std::vector<int64_t>{1, 2, 3, 4, 5});
    Tensor<float> dx({2, 4, 8, 8, 8});

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0, 0},
                                                         {0, 0, 0},
                                                         {2, 2, 2},
                                                         {2, 2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::NEG_INF_PAD),
                 std::invalid_argument);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnIOTensorLayoutsInconsistent)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy4D({2, 4, 4, 4}, TensorLayout::NCHW);
    Tensor<float> dx4D({2, 4, 8, 8}, TensorLayout::NHWC);

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy4D,
                                                         dx4D,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::invalid_argument);

    Tensor<float> dy5D({2, 4, 4, 4, 4}, TensorLayout::NDHWC);
    Tensor<float> dx5D({2, 4, 8, 8, 8}, TensorLayout::NCDHW);

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy5D,
                                                         dx5D,
                                                         {0, 0, 0},
                                                         {0, 0, 0},
                                                         {2, 2, 2},
                                                         {2, 2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::NEG_INF_PAD),
                 std::invalid_argument);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnSpatialParamRankMismatch4D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 4, 4});
    Tensor<float> dx({2, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::runtime_error);
    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2},
                                                         {2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::runtime_error);
    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::runtime_error);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnSpatialParamRankMismatch5D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 4, 4, 4});
    Tensor<float> dx({2, 4, 8, 8, 8});

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0, 0},
                                                         {2, 2, 2},
                                                         {2, 2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::runtime_error);
    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0, 0},
                                                         {0, 0, 0},
                                                         {2, 2},
                                                         {2, 2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::runtime_error);
    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0, 0},
                                                         {0, 0, 0},
                                                         {2, 2, 2},
                                                         {2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::runtime_error);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnEmptySpatialParams)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 4, 4});
    Tensor<float> dx({2, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 2},
                                                         ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                         PaddingMode::NEG_INF_PAD),
                 std::runtime_error);
    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {},
                                                         {2, 2},
                                                         ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                         PaddingMode::NEG_INF_PAD),
                 std::runtime_error);
    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {},
                                                         ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                         PaddingMode::NEG_INF_PAD),
                 std::runtime_error);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnInvalidSpatialParams)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 4, 4});
    Tensor<float> dx({2, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, -1},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::runtime_error);
    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {-2, 2},
                                                         {2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::runtime_error);
    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {0, 2},
                                                         {2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::NEG_INF_PAD),
                 std::runtime_error);
    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 0},
                                                         ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::runtime_error);
    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, -2},
                                                         ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::runtime_error);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnUnsupportedResampleMode)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 4, 4});
    Tensor<float> dx({2, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 2},
                                                         static_cast<ResampleMode>(100),
                                                         PaddingMode::ZERO_PAD),
                 std::invalid_argument);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnUnsupportedPaddingMode)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 4, 4});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<int32_t> index({2, 4, 4, 4});

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 2},
                                                         ResampleMode::MAXPOOL,
                                                         static_cast<PaddingMode>(100),
                                                         &index),
                 std::invalid_argument);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnMissingIndexTensorForMaxPool)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 4, 4});
    Tensor<float> dx({2, 4, 8, 8});

    EXPECT_THROW(
        GpuFpReferenceResample::backward<float>(
            dy, dx, {0, 0}, {0, 0}, {2, 2}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD),
        std::invalid_argument);

    Tensor<float> dy5D({2, 4, 4, 4, 4});
    Tensor<float> dx5D({2, 4, 8, 8, 8});

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy5D,
                                                         dx5D,
                                                         {0, 0, 0},
                                                         {0, 0, 0},
                                                         {2, 2, 2},
                                                         {2, 2, 2},
                                                         ResampleMode::MAXPOOL,
                                                         PaddingMode::NEG_INF_PAD),
                 std::invalid_argument);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnIndexTensorRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 4, 4});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<int32_t> index({2, 4, 4});

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 2},
                                                         ResampleMode::MAXPOOL,
                                                         PaddingMode::ZERO_PAD,
                                                         &index),
                 std::invalid_argument);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnIndexTensorShapeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 4, 4});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<int32_t> index({2, 4, 5, 4});

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 2},
                                                         ResampleMode::MAXPOOL,
                                                         PaddingMode::ZERO_PAD,
                                                         &index),
                 std::invalid_argument);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnIndexTensorLayoutMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 4, 4});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<int32_t> index({2, 4, 4, 4}, TensorLayout::NHWC);

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 2},
                                                         ResampleMode::MAXPOOL,
                                                         PaddingMode::ZERO_PAD,
                                                         &index),
                 std::invalid_argument);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnIndexTensorInputForAvgPool)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 4, 4});
    Tensor<float> dx({2, 4, 8, 8});
    Tensor<int32_t> index({2, 4, 4, 4});

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD,
                                                         &index),
                 std::invalid_argument);
    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 2},
                                                         ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD,
                                                         &index),
                 std::invalid_argument);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnWindowSizeExceedsPaddedDx)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 4, 1, 1});
    Tensor<float> dx({2, 4, 8, 8});

    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {9, 9},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::runtime_error);
}

TEST(TestGpuResampleBwdRefValidation, ThrowsOnDySpatialShapeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dx({2, 4, 8, 8});

    Tensor<float> dyTooLarge({2, 4, 5, 4});
    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dyTooLarge,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::runtime_error);

    Tensor<float> dyTooSmall({2, 4, 4, 3});
    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dyTooSmall,
                                                         dx,
                                                         {0, 0},
                                                         {0, 0},
                                                         {2, 2},
                                                         {2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::runtime_error);

    Tensor<float> dy({2, 4, 4, 4});
    EXPECT_THROW(GpuFpReferenceResample::backward<float>(dy,
                                                         dx,
                                                         {1, 1},
                                                         {1, 1},
                                                         {2, 2},
                                                         {2, 2},
                                                         ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                         PaddingMode::ZERO_PAD),
                 std::runtime_error);
}

// --- Mixed type tests ---

namespace
{

void generateMaxPoolIndex(Tensor<int32_t>& index,
                          const std::vector<int64_t>& dxDims,
                          const std::vector<int64_t>& dyDims,
                          const TensorLayout& layout,
                          const std::vector<int64_t>& prePadding,
                          const std::vector<int64_t>& postPadding,
                          const std::vector<int64_t>& stride,
                          const std::vector<int64_t>& window,
                          PaddingMode paddingMode,
                          unsigned int seed)
{
    Tensor<float> x(dxDims, layout);
    Tensor<float> y(dyDims, layout);
    fillWithRandomValues(x, -1.0f, 1.0f, seed);
    x.memory().hostData();

    GpuFpReferenceResample::forward<float, float, float, int32_t>(
        x, y, prePadding, postPadding, stride, window, ResampleMode::MAXPOOL, paddingMode, &index);
    index.memory().hostData();
}

} // namespace

TEST(TestGpuResampleBwdRefMixedType, FloatDyHalfDx)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> dyTensor({2, 3, 4, 4});
    Tensor<HalfType> dxCpu({2, 3, 8, 8});
    Tensor<HalfType> dxGpu({2, 3, 8, 8});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(dyTensor, -1.0f, 1.0f, seed);
    dyTensor.memory().hostData();

    CpuFpReferenceResampleBwd::backward<float, HalfType, float, int32_t>(
        dyTensor,
        dxCpu,
        {0, 0},
        {2, 2},
        {2, 2},
        ResampleMode::AVGPOOL_EXCLUDE_PADDING,
        PaddingMode::ZERO_PAD);

    GpuFpReferenceResample::backward<float, HalfType, float, int32_t>(
        dyTensor,
        dxGpu,
        {0, 0},
        {0, 0},
        {2, 2},
        {2, 2},
        ResampleMode::AVGPOOL_EXCLUDE_PADDING,
        PaddingMode::ZERO_PAD);

    assertAllClose(dxCpu, dxGpu, getTolerance<HalfType>());
}

TEST(TestGpuResampleBwdRefMixedType, HalfDyFloatDx)
{
    SKIP_IF_NO_DEVICES();

    Tensor<HalfType> dyTensor({2, 3, 4, 4});
    Tensor<float> dxCpu({2, 3, 8, 8});
    Tensor<float> dxGpu({2, 3, 8, 8});
    Tensor<int32_t> index({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(dyTensor, static_cast<HalfType>(-1.0f), static_cast<HalfType>(1.0f), seed);
    dyTensor.memory().hostData();
    generateMaxPoolIndex(index,
                         {2, 3, 8, 8},
                         {2, 3, 4, 4},
                         TensorLayout::NCHW,
                         {0, 0},
                         {0, 0},
                         {2, 2},
                         {2, 2},
                         PaddingMode::ZERO_PAD,
                         seed + 1);

    CpuFpReferenceResampleBwd::backward<HalfType, float, float, int32_t>(dyTensor,
                                                                         dxCpu,
                                                                         {0, 0},
                                                                         {2, 2},
                                                                         {2, 2},
                                                                         ResampleMode::MAXPOOL,
                                                                         PaddingMode::ZERO_PAD,
                                                                         &index);

    GpuFpReferenceResample::backward<HalfType, float, float, int32_t>(dyTensor,
                                                                      dxGpu,
                                                                      {0, 0},
                                                                      {0, 0},
                                                                      {2, 2},
                                                                      {2, 2},
                                                                      ResampleMode::MAXPOOL,
                                                                      PaddingMode::ZERO_PAD,
                                                                      &index);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
}

TEST(TestGpuResampleBwdRefMixedType, HalfDyHalfDx)
{
    SKIP_IF_NO_DEVICES();

    Tensor<HalfType> dyTensor({2, 3, 4, 4});
    Tensor<HalfType> dxCpu({2, 3, 8, 8});
    Tensor<HalfType> dxGpu({2, 3, 8, 8});
    Tensor<int32_t> index({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(dyTensor, static_cast<HalfType>(-1.0f), static_cast<HalfType>(1.0f), seed);
    dyTensor.memory().hostData();
    generateMaxPoolIndex(index,
                         {2, 3, 8, 8},
                         {2, 3, 4, 4},
                         TensorLayout::NCHW,
                         {0, 0},
                         {0, 0},
                         {2, 2},
                         {2, 2},
                         PaddingMode::NEG_INF_PAD,
                         seed + 1);

    CpuFpReferenceResampleBwd::backward<HalfType, HalfType, float, int32_t>(
        dyTensor,
        dxCpu,
        {0, 0},
        {2, 2},
        {2, 2},
        ResampleMode::MAXPOOL,
        PaddingMode::NEG_INF_PAD,
        &index);

    GpuFpReferenceResample::backward<HalfType, HalfType, float, int32_t>(dyTensor,
                                                                         dxGpu,
                                                                         {0, 0},
                                                                         {0, 0},
                                                                         {2, 2},
                                                                         {2, 2},
                                                                         ResampleMode::MAXPOOL,
                                                                         PaddingMode::NEG_INF_PAD,
                                                                         &index);

    assertAllClose(dxCpu, dxGpu, getTolerance<HalfType>());
}

TEST(TestGpuResampleBwdRefMixedType, BfloatDyFloatDx)
{
    SKIP_IF_NO_DEVICES();

    Tensor<BFloat16Type> dyTensor({2, 3, 4, 4});
    Tensor<float> dxCpu({2, 3, 8, 8});
    Tensor<float> dxGpu({2, 3, 8, 8});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(
        dyTensor, static_cast<BFloat16Type>(-1.0f), static_cast<BFloat16Type>(1.0f), seed);
    dyTensor.memory().hostData();

    CpuFpReferenceResampleBwd::backward<BFloat16Type, float, float, int32_t>(
        dyTensor,
        dxCpu,
        {0, 0},
        {2, 2},
        {2, 2},
        ResampleMode::AVGPOOL_INCLUDE_PADDING,
        PaddingMode::ZERO_PAD);

    GpuFpReferenceResample::backward<BFloat16Type, float, float, int32_t>(
        dyTensor,
        dxGpu,
        {0, 0},
        {0, 0},
        {2, 2},
        {2, 2},
        ResampleMode::AVGPOOL_INCLUDE_PADDING,
        PaddingMode::ZERO_PAD);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
}

TEST(TestGpuResampleBwdRefMixedType, BfloatDyHalfDx)
{
    SKIP_IF_NO_DEVICES();

    Tensor<BFloat16Type> dyTensor({2, 3, 4, 4});
    Tensor<HalfType> dxCpu({2, 3, 8, 8});
    Tensor<HalfType> dxGpu({2, 3, 8, 8});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(
        dyTensor, static_cast<BFloat16Type>(-1.0f), static_cast<BFloat16Type>(1.0f), seed);
    dyTensor.memory().hostData();

    CpuFpReferenceResampleBwd::backward<BFloat16Type, HalfType, float, int32_t>(
        dyTensor,
        dxCpu,
        {0, 0},
        {2, 2},
        {2, 2},
        ResampleMode::AVGPOOL_EXCLUDE_PADDING,
        PaddingMode::NEG_INF_PAD);

    GpuFpReferenceResample::backward<BFloat16Type, HalfType, float, int32_t>(
        dyTensor,
        dxGpu,
        {0, 0},
        {0, 0},
        {2, 2},
        {2, 2},
        ResampleMode::AVGPOOL_EXCLUDE_PADDING,
        PaddingMode::NEG_INF_PAD);

    assertAllClose(dxCpu, dxGpu, getTolerance<HalfType>());
}

TEST(TestGpuResampleBwdRefMixedType, DoubleDyDoubleDx)
{
    SKIP_IF_NO_DEVICES();

    Tensor<double> dyTensor({2, 3, 4, 4});
    Tensor<double> dxCpu({2, 3, 8, 8});
    Tensor<double> dxGpu({2, 3, 8, 8});
    Tensor<int32_t> index({2, 3, 4, 4});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(dyTensor, -1.0, 1.0, seed);
    dyTensor.memory().hostData();
    generateMaxPoolIndex(index,
                         {2, 3, 8, 8},
                         {2, 3, 4, 4},
                         TensorLayout::NCHW,
                         {0, 0},
                         {0, 0},
                         {2, 2},
                         {2, 2},
                         PaddingMode::NEG_INF_PAD,
                         seed + 1);

    CpuFpReferenceResampleBwd::backward<double, double, double, int32_t>(dyTensor,
                                                                         dxCpu,
                                                                         {0, 0},
                                                                         {2, 2},
                                                                         {2, 2},
                                                                         ResampleMode::MAXPOOL,
                                                                         PaddingMode::NEG_INF_PAD,
                                                                         &index);

    GpuFpReferenceResample::backward<double, double, double, int32_t>(dyTensor,
                                                                      dxGpu,
                                                                      {0, 0},
                                                                      {0, 0},
                                                                      {2, 2},
                                                                      {2, 2},
                                                                      ResampleMode::MAXPOOL,
                                                                      PaddingMode::NEG_INF_PAD,
                                                                      &index);

    assertAllClose(dxCpu, dxGpu, getTolerance<double>());
}

// --- Optional arguments tests ---

TEST(TestGpuResampleBwdRefOptionalArgs, MaxPoolWithIndex)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> dyTensor({2, 3, 3, 3, 3});
    Tensor<float> dxCpu({2, 3, 4, 5, 6});
    Tensor<float> dxGpu({2, 3, 4, 5, 6});
    Tensor<int32_t> index({2, 3, 3, 3, 3});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(dyTensor, -1.0f, 1.0f, seed);
    dyTensor.memory().hostData();
    generateMaxPoolIndex(index,
                         {2, 3, 4, 5, 6},
                         {2, 3, 3, 3, 3},
                         TensorLayout::NCDHW,
                         {1, 2, 1},
                         {0, 1, 0},
                         {1, 2, 2},
                         {3, 4, 3},
                         PaddingMode::ZERO_PAD,
                         seed + 1);

    CpuFpReferenceResampleBwd::backward<float, float, float, int32_t>(dyTensor,
                                                                      dxCpu,
                                                                      {1, 2, 1},
                                                                      {1, 2, 2},
                                                                      {3, 4, 3},
                                                                      ResampleMode::MAXPOOL,
                                                                      PaddingMode::ZERO_PAD,
                                                                      &index);
    GpuFpReferenceResample::backward<float, float, float, int32_t>(dyTensor,
                                                                   dxGpu,
                                                                   {1, 2, 1},
                                                                   {0, 1, 0},
                                                                   {1, 2, 2},
                                                                   {3, 4, 3},
                                                                   ResampleMode::MAXPOOL,
                                                                   PaddingMode::ZERO_PAD,
                                                                   &index);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
}

TEST(TestGpuResampleBwdRefOptionalArgs, AverageExcludePadding)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> dyTensor({2, 3, 3, 3});
    Tensor<float> dxCpu({2, 3, 6, 6});
    Tensor<float> dxGpu({2, 3, 6, 6});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(dyTensor, -1.0f, 1.0f, seed);
    dyTensor.memory().hostData();

    CpuFpReferenceResampleBwd::backward<float, float, float>(dyTensor,
                                                             dxCpu,
                                                             {2, 2},
                                                             {2, 2},
                                                             {4, 4},
                                                             ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                             PaddingMode::NEG_INF_PAD);
    GpuFpReferenceResample::backward<float, float, float>(dyTensor,
                                                          dxGpu,
                                                          {2, 2},
                                                          {0, 0},
                                                          {2, 2},
                                                          {4, 4},
                                                          ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                          PaddingMode::NEG_INF_PAD);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
}

TEST(TestGpuResampleBwdRefOptionalArgs, AvgPoolIncludePadding)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> dyTensor({2, 3, 4, 3});
    Tensor<float> dxCpu({2, 3, 6, 6});
    Tensor<float> dxGpu({2, 3, 6, 6});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(dyTensor, -1.0f, 1.0f, seed);
    dyTensor.memory().hostData();

    CpuFpReferenceResampleBwd::backward<float, float, float>(dyTensor,
                                                             dxCpu,
                                                             {2, 2},
                                                             {2, 2},
                                                             {4, 4},
                                                             ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD);
    GpuFpReferenceResample::backward<float, float, float>(dyTensor,
                                                          dxGpu,
                                                          {2, 2},
                                                          {2, 0},
                                                          {2, 2},
                                                          {4, 4},
                                                          ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                          PaddingMode::ZERO_PAD);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
}

// --- Channel-last layout tests ---

TEST(TestGpuResampleBwdRefChannelLast, AvgPoolMatchesCpuRef4D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dyTensor({2, 4, 5, 7}, TensorLayout::NHWC);
    Tensor<float> dxCpu({2, 4, 9, 7}, TensorLayout::NHWC);
    Tensor<float> dxGpu({2, 4, 9, 7}, TensorLayout::NHWC);

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(dyTensor, -1.0f, 1.0f, seed);
    dyTensor.memory().hostData();

    CpuFpReferenceResampleBwd::backward<float, float, float>(dyTensor,
                                                             dxCpu,
                                                             {1, 1},
                                                             {2, 1},
                                                             {3, 3},
                                                             ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD);
    GpuFpReferenceResample::backward<float, float, float>(dyTensor,
                                                          dxGpu,
                                                          {1, 1},
                                                          {1, 1},
                                                          {2, 1},
                                                          {3, 3},
                                                          ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                          PaddingMode::ZERO_PAD);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
}

TEST(TestGpuResampleBwdRefChannelLast, MaxPoolMatchesCpuRefWithIndex4D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dyTensor({2, 4, 4, 4}, TensorLayout::NHWC);
    Tensor<float> dxCpu({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> dxGpu({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<int32_t> index({2, 4, 4, 4}, TensorLayout::NHWC);

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(dyTensor, -1.0f, 1.0f, seed);
    dyTensor.memory().hostData();
    generateMaxPoolIndex(index,
                         {2, 4, 8, 8},
                         {2, 4, 4, 4},
                         TensorLayout::NHWC,
                         {0, 0},
                         {0, 0},
                         {2, 2},
                         {2, 2},
                         PaddingMode::ZERO_PAD,
                         seed + 1);

    CpuFpReferenceResampleBwd::backward<float, float, float>(dyTensor,
                                                             dxCpu,
                                                             {0, 0},
                                                             {2, 2},
                                                             {2, 2},
                                                             ResampleMode::MAXPOOL,
                                                             PaddingMode::ZERO_PAD,
                                                             &index);
    GpuFpReferenceResample::backward<float, float, float>(dyTensor,
                                                          dxGpu,
                                                          {0, 0},
                                                          {0, 0},
                                                          {2, 2},
                                                          {2, 2},
                                                          ResampleMode::MAXPOOL,
                                                          PaddingMode::ZERO_PAD,
                                                          &index);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
}

TEST(TestGpuResampleBwdRefChannelLast, AvgPoolMatchesCpuRef5D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dyTensor({2, 4, 4, 4, 4}, TensorLayout::NDHWC);
    Tensor<float> dxCpu({2, 4, 8, 8, 8}, TensorLayout::NDHWC);
    Tensor<float> dxGpu({2, 4, 8, 8, 8}, TensorLayout::NDHWC);

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(dyTensor, -1.0f, 1.0f, seed);
    dyTensor.memory().hostData();

    CpuFpReferenceResampleBwd::backward<float, float, float>(dyTensor,
                                                             dxCpu,
                                                             {1, 1, 1},
                                                             {2, 2, 2},
                                                             {3, 3, 3},
                                                             ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                             PaddingMode::NEG_INF_PAD);
    GpuFpReferenceResample::backward<float, float, float>(dyTensor,
                                                          dxGpu,
                                                          {1, 1, 1},
                                                          {1, 1, 1},
                                                          {2, 2, 2},
                                                          {3, 3, 3},
                                                          ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                          PaddingMode::NEG_INF_PAD);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
}

TEST(TestGpuResampleBwdRefChannelLast, MaxPoolMatchesCpuRefWithIndex5D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dyTensor({2, 4, 4, 4, 4}, TensorLayout::NDHWC);
    Tensor<float> dxCpu({2, 4, 8, 8, 8}, TensorLayout::NDHWC);
    Tensor<float> dxGpu({2, 4, 8, 8, 8}, TensorLayout::NDHWC);
    Tensor<int32_t> index({2, 4, 4, 4, 4}, TensorLayout::NDHWC);

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(dyTensor, -1.0f, 1.0f, seed);
    dyTensor.memory().hostData();
    generateMaxPoolIndex(index,
                         {2, 4, 8, 8, 8},
                         {2, 4, 4, 4, 4},
                         TensorLayout::NDHWC,
                         {0, 0, 0},
                         {0, 0, 0},
                         {2, 2, 2},
                         {2, 2, 2},
                         PaddingMode::ZERO_PAD,
                         seed + 1);

    CpuFpReferenceResampleBwd::backward<float, float, float>(dyTensor,
                                                             dxCpu,
                                                             {0, 0, 0},
                                                             {2, 2, 2},
                                                             {2, 2, 2},
                                                             ResampleMode::MAXPOOL,
                                                             PaddingMode::ZERO_PAD,
                                                             &index);
    GpuFpReferenceResample::backward<float, float, float>(dyTensor,
                                                          dxGpu,
                                                          {0, 0, 0},
                                                          {0, 0, 0},
                                                          {2, 2, 2},
                                                          {2, 2, 2},
                                                          ResampleMode::MAXPOOL,
                                                          PaddingMode::ZERO_PAD,
                                                          &index);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
}

TEST(TestGpuResampleBwdRefOverlap, MaxPoolOverlappingWindowsMatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> dyTensor({2, 3, 7, 7});
    Tensor<float> dxCpu({2, 3, 9, 9});
    Tensor<float> dxGpu({2, 3, 9, 9});
    Tensor<int32_t> index({2, 3, 7, 7});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(dyTensor, -1.0f, 1.0f, seed);
    dyTensor.memory().hostData();
    generateMaxPoolIndex(index,
                         {2, 3, 9, 9},
                         {2, 3, 7, 7},
                         TensorLayout::NCHW,
                         {0, 0},
                         {0, 0},
                         {1, 1},
                         {3, 3},
                         PaddingMode::NEG_INF_PAD,
                         seed + 1);

    CpuFpReferenceResampleBwd::backward<float, float, float>(dyTensor,
                                                             dxCpu,
                                                             {0, 0},
                                                             {1, 1},
                                                             {3, 3},
                                                             ResampleMode::MAXPOOL,
                                                             PaddingMode::NEG_INF_PAD,
                                                             &index);
    GpuFpReferenceResample::backward<float, float, float>(dyTensor,
                                                          dxGpu,
                                                          {0, 0},
                                                          {0, 0},
                                                          {1, 1},
                                                          {3, 3},
                                                          ResampleMode::MAXPOOL,
                                                          PaddingMode::NEG_INF_PAD,
                                                          &index);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
}

TEST(TestGpuResampleBwdRefOverlap, AvgPoolOverlappingWindowsMatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> dyTensor({2, 3, 9, 9});
    Tensor<float> dxCpu({2, 3, 9, 9});
    Tensor<float> dxGpu({2, 3, 9, 9});

    const unsigned int seed = getGlobalTestSeed();
    fillWithRandomValues(dyTensor, -1.0f, 1.0f, seed);
    dyTensor.memory().hostData();

    CpuFpReferenceResampleBwd::backward<float, float, float>(dyTensor,
                                                             dxCpu,
                                                             {1, 1},
                                                             {1, 1},
                                                             {3, 3},
                                                             ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                             PaddingMode::NEG_INF_PAD);
    GpuFpReferenceResample::backward<float, float, float>(dyTensor,
                                                          dxGpu,
                                                          {1, 1},
                                                          {1, 1},
                                                          {1, 1},
                                                          {3, 3},
                                                          ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                          PaddingMode::NEG_INF_PAD);

    assertAllClose(dxCpu, dxGpu, getTolerance<float>());
}

// --- Test suite instantiations ---

using TestGpuResampleBwdRef4DFp32 = ResampleBwdTestSuite<float>;
using TestGpuResampleBwdRef4DFp16 = ResampleBwdTestSuite<HalfType>;
using TestGpuResampleBwdRef4DBfp16 = ResampleBwdTestSuite<BFloat16Type>;
using TestGpuResampleBwdRef5DFp32 = ResampleBwdTestSuite<float>;
using TestGpuResampleBwdRef5DFp16 = ResampleBwdTestSuite<HalfType>;
using TestGpuResampleBwdRef5DBfp16 = ResampleBwdTestSuite<BFloat16Type>;

TEST_P(TestGpuResampleBwdRef4DFp32, MatchesCpuRef)
{
    this->runResampleBwdTest();
}
TEST_P(TestGpuResampleBwdRef4DFp16, MatchesCpuRef)
{
    this->runResampleBwdTest();
}
TEST_P(TestGpuResampleBwdRef4DBfp16, MatchesCpuRef)
{
    this->runResampleBwdTest();
}
TEST_P(TestGpuResampleBwdRef5DFp32, MatchesCpuRef)
{
    this->runResampleBwdTest();
}
TEST_P(TestGpuResampleBwdRef5DFp16, MatchesCpuRef)
{
    this->runResampleBwdTest();
}
TEST_P(TestGpuResampleBwdRef5DBfp16, MatchesCpuRef)
{
    this->runResampleBwdTest();
}

// ============================================================================
// 4D tests
// ============================================================================

// --- Quick tests ---

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuResampleBwdRef4DFp32,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleSmall4DShapes(),
                                                                   getResample4DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::BACKWARD)));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuResampleBwdRef4DFp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleSmall4DShapes(),
                                                                   getResample4DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::BACKWARD)));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuResampleBwdRef4DBfp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleSmall4DShapes(),
                                                                   getResample4DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::BACKWARD)));

// --- Standard tests ---

INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuResampleBwdRef4DFp32,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleMedium4DShapes(),
                                                                   getResample4DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::BACKWARD)));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuResampleBwdRef4DFp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleMedium4DShapes(),
                                                                   getResample4DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::BACKWARD)));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuResampleBwdRef4DBfp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleMedium4DShapes(),
                                                                   getResample4DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::BACKWARD)));

// --- Comprehensive tests ---

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuResampleBwdRef4DFp32,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleLarge4DShapes(),
                                                                   getResample4DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::BACKWARD)));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuResampleBwdRef4DFp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleLarge4DShapes(),
                                                                   getResample4DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::BACKWARD)));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuResampleBwdRef4DBfp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleLarge4DShapes(),
                                                                   getResample4DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::BACKWARD)));

// ============================================================================
// 5D tests
// ============================================================================

// --- Quick tests ---

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuResampleBwdRef5DFp32,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleSmall5DShapes(),
                                                                   getResample5DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::BACKWARD)));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuResampleBwdRef5DFp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleSmall5DShapes(),
                                                                   getResample5DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::BACKWARD)));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuResampleBwdRef5DBfp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleSmall5DShapes(),
                                                                   getResample5DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::BACKWARD)));

// --- Standard tests ---

INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuResampleBwdRef5DFp32,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleMedium5DShapes(),
                                                                   getResample5DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::BACKWARD)));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuResampleBwdRef5DFp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleMedium5DShapes(),
                                                                   getResample5DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::BACKWARD)));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuResampleBwdRef5DBfp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleMedium5DShapes(),
                                                                   getResample5DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::BACKWARD)));

// --- Comprehensive tests ---

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuResampleBwdRef5DFp32,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleLarge5DShapes(),
                                                                   getResample5DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::BACKWARD)));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuResampleBwdRef5DFp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleLarge5DShapes(),
                                                                   getResample5DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::BACKWARD)));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuResampleBwdRef5DBfp16,
                         ::testing::ValuesIn(makeResampleTestCases(getResampleLarge5DShapes(),
                                                                   getResample5DLayouts(),
                                                                   getResampleModes(),
                                                                   getPaddingModes(),
                                                                   ResampleDirection::BACKWARD)));
