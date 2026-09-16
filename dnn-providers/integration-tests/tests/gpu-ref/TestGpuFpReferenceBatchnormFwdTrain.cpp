// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GpuBatchnormFwdTrainRefTestFixture.hpp"

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::batchnorm;
using namespace hipdnn_gpu_ref;
using namespace gpu_batchnorm_ref_test;
using namespace gpu_batchnorm_fwd_train_ref_test;

// --- Valid configurations ---

TEST(TestGpuBatchnormFwdTrainRefValidation, AcceptsValidParamsAllDims)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x3D({2, 4, 8});
    Tensor<float> scale3D({1, 4, 1});
    Tensor<float> bias3D({1, 4, 1});
    Tensor<float> y3D({2, 4, 8});
    EXPECT_NO_THROW(GpuFpReferenceBatchnorm::fwdTraining(x3D, scale3D, bias3D, y3D, 1.0e-5, 0.1));

    Tensor<float> x4D({2, 4, 8, 8});
    Tensor<float> scale4D({1, 4, 1, 1});
    Tensor<float> bias4D({1, 4, 1, 1});
    Tensor<float> y4D({2, 4, 8, 8});
    EXPECT_NO_THROW(GpuFpReferenceBatchnorm::fwdTraining(x4D, scale4D, bias4D, y4D, 1.0e-5, 0.1));

    Tensor<float> x5D({2, 4, 8, 8, 8});
    Tensor<float> scale5D({1, 4, 1, 1, 1});
    Tensor<float> bias5D({1, 4, 1, 1, 1});
    Tensor<float> y5D({2, 4, 8, 8, 8});
    EXPECT_NO_THROW(GpuFpReferenceBatchnorm::fwdTraining(x5D, scale5D, bias5D, y5D, 1.0e-5, 0.1));
}

TEST(TestGpuBatchnormFwdTrainRefValidation, AcceptsValidParamsChannelLastLayout)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x3D({2, 4, 8}, TensorLayout::NLC);
    Tensor<float> scale3D({1, 4, 1}, TensorLayout::NLC);
    Tensor<float> bias3D({1, 4, 1}, TensorLayout::NLC);
    Tensor<float> y3D({2, 4, 8}, TensorLayout::NLC);
    EXPECT_NO_THROW(GpuFpReferenceBatchnorm::fwdTraining(x3D, scale3D, bias3D, y3D, 1.0e-5, 0.1));

    Tensor<float> x4D({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> scale4D({1, 4, 1, 1}, TensorLayout::NHWC);
    Tensor<float> bias4D({1, 4, 1, 1}, TensorLayout::NHWC);
    Tensor<float> y4D({2, 4, 8, 8}, TensorLayout::NHWC);
    EXPECT_NO_THROW(GpuFpReferenceBatchnorm::fwdTraining(x4D, scale4D, bias4D, y4D, 1.0e-5, 0.1));

    Tensor<float> x5D({2, 4, 8, 8, 8}, TensorLayout::NDHWC);
    Tensor<float> scale5D({1, 4, 1, 1, 1}, TensorLayout::NDHWC);
    Tensor<float> bias5D({1, 4, 1, 1, 1}, TensorLayout::NDHWC);
    Tensor<float> y5D({2, 4, 8, 8, 8}, TensorLayout::NDHWC);
    EXPECT_NO_THROW(GpuFpReferenceBatchnorm::fwdTraining(x5D, scale5D, bias5D, y5D, 1.0e-5, 0.1));
}

TEST(TestGpuBatchnormFwdTrainRefValidation, AcceptsValidParamsWithSaveAndRunningStats)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> scale({1, 4, 1, 1});
    Tensor<float> bias({1, 4, 1, 1});
    Tensor<float> y({2, 4, 8, 8});
    Tensor<float> mean({1, 4, 1, 1});
    Tensor<float> invVar({1, 4, 1, 1});
    Tensor<float> prevRunningMean({1, 4, 1, 1});
    Tensor<float> prevRunningVar({1, 4, 1, 1});
    Tensor<float> nextRunningMean({1, 4, 1, 1});
    Tensor<float> nextRunningVar({1, 4, 1, 1});

    // With save stats
    EXPECT_NO_THROW(
        GpuFpReferenceBatchnorm::fwdTraining(x, scale, bias, y, 1.0e-5, 0.1, &mean, &invVar));

    // With running stats
    EXPECT_NO_THROW(
        (GpuFpReferenceBatchnorm::fwdTraining<float, float, float, float, float>(x,
                                                                                 scale,
                                                                                 bias,
                                                                                 y,
                                                                                 1.0e-5,
                                                                                 0.1,
                                                                                 nullptr,
                                                                                 nullptr,
                                                                                 &prevRunningMean,
                                                                                 &prevRunningVar,
                                                                                 &nextRunningMean,
                                                                                 &nextRunningVar)));

    // With save and running stats
    EXPECT_NO_THROW((GpuFpReferenceBatchnorm::fwdTraining(x,
                                                          scale,
                                                          bias,
                                                          y,
                                                          1.0e-5,
                                                          0.1,
                                                          &mean,
                                                          &invVar,
                                                          &prevRunningMean,
                                                          &prevRunningVar,
                                                          &nextRunningMean,
                                                          &nextRunningVar)));
}

TEST(TestGpuBatchnormFwdTrainRefValidation, AcceptsAffineBroadcast)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({4, 8, 2, 2});
    Tensor<float> y({4, 8, 2, 2});
    Tensor<float> scale({1, 8, 1, 1, 1});
    Tensor<float> bias({1, 8, 1, 1});
    Tensor<float> mean({1, 8, 1});
    Tensor<float> invVar({1, 8});

    EXPECT_NO_THROW(
        GpuFpReferenceBatchnorm::fwdTraining(x, scale, bias, y, 1.0e-5, 0.1, &mean, &invVar));
}

// --- Invalid configurations ---

TEST(TestGpuBatchnormFwdTrainRefValidation, ThrowsOnInputRankTooSmall)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({4, 8});
    Tensor<float> y({4, 8});
    Tensor<float> scale({1, 8});
    Tensor<float> bias({1, 8});

    EXPECT_THROW(GpuFpReferenceBatchnorm::fwdTraining(x, scale, bias, y, 1.0e-5, 0.1),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormFwdTrainRefValidation, ThrowsOnInputRankTooLarge)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({4, 8, 2, 2, 2, 2});
    Tensor<float> y({4, 8, 2, 2, 2, 2});
    Tensor<float> scale({1, 8, 1, 1, 1, 1});
    Tensor<float> bias({1, 8, 1, 1, 1, 1});

    EXPECT_THROW(GpuFpReferenceBatchnorm::fwdTraining(x, scale, bias, y, 1.0e-5, 0.1),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormFwdTrainRefValidation, ThrowsOnOutputRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2});
    Tensor<float> y({4, 8, 2});
    Tensor<float> scale({1, 8, 1, 1});
    Tensor<float> bias({1, 8, 1, 1});

    EXPECT_THROW(GpuFpReferenceBatchnorm::fwdTraining(x, scale, bias, y, 1.0e-5, 0.1),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormFwdTrainRefValidation, ThrowsOnAffineRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2});
    Tensor<float> y({4, 8, 2, 2});
    Tensor<float> scale({1, 8, 2});
    Tensor<float> bias({1, 8, 1, 1});

    EXPECT_THROW(GpuFpReferenceBatchnorm::fwdTraining(x, scale, bias, y, 1.0e-5, 0.1),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormFwdTrainRefValidation, ThrowsOnAffineNotChannelOnly)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2});
    Tensor<float> y({4, 8, 2, 2});
    Tensor<float> scale({1, 8, 2, 1});
    Tensor<float> bias({1, 8, 1, 1});

    EXPECT_THROW(GpuFpReferenceBatchnorm::fwdTraining(x, scale, bias, y, 1.0e-5, 0.1),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormFwdTrainRefValidation, ThrowsOnAffineWrongChannel)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2});
    Tensor<float> y({4, 8, 2, 2});
    Tensor<float> scale({1, 8, 1, 1});
    Tensor<float> bias({1, 8, 1, 1});
    Tensor<float> mean({1, 4, 1, 1});
    Tensor<float> invVar({1, 8, 1, 1});

    EXPECT_THROW(
        GpuFpReferenceBatchnorm::fwdTraining(x, scale, bias, y, 1.0e-5, 0.1, &mean, &invVar),
        std::invalid_argument);
}

TEST(TestGpuBatchnormFwdTrainRefValidation, ThrowsOnInconsistentLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2}, TensorLayout::NHWC);
    Tensor<float> y({4, 8, 2, 2}, TensorLayout::NCHW);
    Tensor<float> scale({1, 8}, TensorLayout::NHWC);
    Tensor<float> bias({1, 8}, TensorLayout::NHWC);
    Tensor<float> prevRunningMean({1, 8}, TensorLayout::NHWC);
    Tensor<float> prevRunningVar({1, 8}, TensorLayout::NHWC);
    Tensor<float> nextRunningMean({1, 8}, TensorLayout::NHWC);
    Tensor<float> nextRunningVar({1, 8}, TensorLayout::NHWC);

    EXPECT_THROW(
        (GpuFpReferenceBatchnorm::fwdTraining<float, float, float, float, float>(x,
                                                                                 scale,
                                                                                 bias,
                                                                                 y,
                                                                                 1.0e-5,
                                                                                 0.1,
                                                                                 nullptr,
                                                                                 nullptr,
                                                                                 &prevRunningMean,
                                                                                 &prevRunningVar,
                                                                                 &nextRunningMean,
                                                                                 &nextRunningVar)),
        std::invalid_argument);
}

TEST(TestGpuBatchnormFwdTrainRefValidation, ThrowsOnInvalidLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2}, TensorLayout::BSHD);
    Tensor<float> y({4, 8, 2, 2}, TensorLayout::BSHD);
    Tensor<float> scale({1, 8}, TensorLayout::BSHD);
    Tensor<float> bias({1, 8}, TensorLayout::BSHD);

    EXPECT_THROW(GpuFpReferenceBatchnorm::fwdTraining(x, scale, bias, y, 1.0e-5, 0.1),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormFwdTrainRefValidation, ThrowsOnNonPackedIOLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 2, 1, 1}, {16, 4, 1, 1});
    Tensor<float> y({4, 2, 1, 1}, {16, 4, 1, 1});
    Tensor<float> scale({1, 2}, {2, 1});
    Tensor<float> bias({1, 2}, {2, 1});

    EXPECT_THROW(GpuFpReferenceBatchnorm::fwdTraining(x, scale, bias, y, 1.0e-5, 0.1),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormFwdTrainRefValidation, ThrowsOnNonPackedAffineLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 2, 1, 1});
    Tensor<float> y({4, 2, 1, 1});
    Tensor<float> scale({1, 2}, {4, 2});
    Tensor<float> bias({1, 2}, {4, 2});

    EXPECT_THROW(GpuFpReferenceBatchnorm::fwdTraining(x, scale, bias, y, 1.0e-5, 0.1),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormFwdTrainRefValidation, ThrowsIfAnySaveStatsTensorIsMissing)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2});
    Tensor<float> y({4, 8, 2, 2});
    Tensor<float> scale({1, 8, 1, 1});
    Tensor<float> bias({1, 8, 1, 1});
    Tensor<float> mean({1, 8, 1, 1});

    EXPECT_THROW((GpuFpReferenceBatchnorm::fwdTraining<float, float, float, float, float>(
                     x, scale, bias, y, 1.0e-5, 0.1, &mean, nullptr)),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormFwdTrainRefValidation, ThrowsIfAnyRunningStatsTensorIsMissing)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({4, 8, 2, 2});
    Tensor<float> y({4, 8, 2, 2});
    Tensor<float> scale({1, 8, 1, 1});
    Tensor<float> bias({1, 8, 1, 1});
    Tensor<float> prevRunningVar({1, 8, 1, 1});
    Tensor<float> nextRunningMean({1, 8, 1, 1});

    EXPECT_THROW(
        (GpuFpReferenceBatchnorm::fwdTraining<float, float, float, float, float>(x,
                                                                                 scale,
                                                                                 bias,
                                                                                 y,
                                                                                 1.0e-5,
                                                                                 0.1,
                                                                                 nullptr,
                                                                                 nullptr,
                                                                                 nullptr,
                                                                                 &prevRunningVar,
                                                                                 &nextRunningMean,
                                                                                 nullptr)),
        std::invalid_argument);
}

// --- Test 3D/4D/5D shapes ---

TEST(TestGpuBatchnormFwdTrainRef3DShapes, Broadcast2D)
{
    SKIP_IF_NO_DEVICES();

    runGpuVsCpuBatchnormFwdTrain<float, float, float, float, float>(
        {3, 2, 4}, {1, 2}, TensorLayout::NCL, 1.0f, true, true);
}

TEST(TestGpuBatchnormFwdTrainRef4DShapes, Broadcast2D)
{
    SKIP_IF_NO_DEVICES();

    runGpuVsCpuBatchnormFwdTrain<float, float, float, float, float>(
        {3, 2, 4, 4}, {1, 2}, TensorLayout::NCHW, 1.0f, true, true);
}

TEST(TestGpuBatchnormFwdTrainRef4DShapes, Broadcast3D)
{
    SKIP_IF_NO_DEVICES();

    runGpuVsCpuBatchnormFwdTrain<float, float, float, float, float>(
        {3, 2, 4, 4}, {1, 2, 1}, TensorLayout::NCHW, 1.0f, true, true);
}

TEST(TestGpuBatchnormFwdTrainRef5DShapes, Broadcast3D)
{
    SKIP_IF_NO_DEVICES();

    runGpuVsCpuBatchnormFwdTrain<float, float, float, float, float>(
        {3, 2, 4, 4, 4}, {1, 2, 1}, TensorLayout::NCDHW, 1.0f, true, true);
}

TEST(TestGpuBatchnormFwdTrainRef5DShapes, Broadcast4D)
{
    SKIP_IF_NO_DEVICES();

    runGpuVsCpuBatchnormFwdTrain<float, float, float, float, float>(
        {3, 2, 4, 4, 4}, {1, 2, 1, 1}, TensorLayout::NCDHW, 1.0f, true, true);
}

// --- Test mixed precision ---

TEST(TestGpuBatchnormFwdTrainRefMixedPrecision, UpcastX)
{
    SKIP_IF_NO_DEVICES();

    using XDataType = bfloat16;
    using ScaleBiasType = float;
    using MeanVarType = float;
    using YDataType = float;
    using ComputeDataType = float;

    runGpuVsCpuBatchnormFwdTrain<XDataType, YDataType, ScaleBiasType, MeanVarType, ComputeDataType>(
        {3, 2, 4, 4}, {1, 2, 1, 1}, TensorLayout::NCHW, 1.0f, true, true);
}

TEST(TestGpuBatchnormFwdTrainRefMixedPrecision, DowncastX)
{
    SKIP_IF_NO_DEVICES();

    using XDataType = float;
    using ScaleBiasType = half;
    using MeanVarType = half;
    using YDataType = half;
    using ComputeDataType = float;

    runGpuVsCpuBatchnormFwdTrain<XDataType, YDataType, ScaleBiasType, MeanVarType, ComputeDataType>(
        {3, 2, 4, 4}, {1, 2, 1, 1}, TensorLayout::NCHW, 1.0f, true, true);
}

TEST(TestGpuBatchnormFwdTrainRefMixedPrecision, UpcastY)
{
    SKIP_IF_NO_DEVICES();

    using XDataType = half;
    using ScaleBiasType = half;
    using MeanVarType = half;
    using YDataType = float;
    using ComputeDataType = float;

    runGpuVsCpuBatchnormFwdTrain<XDataType, YDataType, ScaleBiasType, MeanVarType, ComputeDataType>(
        {3, 2, 4, 4}, {1, 2, 1, 1}, TensorLayout::NCHW, 1.0f, true, true);
}

TEST(TestGpuBatchnormFwdTrainRefMixedPrecision, DowncastY)
{
    SKIP_IF_NO_DEVICES();

    using XDataType = float;
    using ScaleBiasType = float;
    using MeanVarType = float;
    using YDataType = bfloat16;
    using ComputeDataType = float;

    runGpuVsCpuBatchnormFwdTrain<XDataType, YDataType, ScaleBiasType, MeanVarType, ComputeDataType>(
        {3, 2, 4, 4}, {1, 2, 1, 1}, TensorLayout::NCHW, 1.0f, true, true);
}

TEST(TestGpuBatchnormFwdTrainRefMixedPrecision, UpcastAffine)
{
    SKIP_IF_NO_DEVICES();

    using XDataType = bfloat16;
    using ScaleBiasType = float;
    using MeanVarType = float;
    using YDataType = half;
    using ComputeDataType = float;

    runGpuVsCpuBatchnormFwdTrain<XDataType, YDataType, ScaleBiasType, MeanVarType, ComputeDataType>(
        {3, 2, 4, 4}, {1, 2, 1, 1}, TensorLayout::NCHW, 1.0f, true, true);
}

TEST(TestGpuBatchnormFwdTrainRefMixedPrecision, DowncastAffine)
{
    SKIP_IF_NO_DEVICES();

    using XDataType = float;
    using ScaleBiasType = half;
    using MeanVarType = bfloat16;
    using YDataType = float;
    using ComputeDataType = float;

    runGpuVsCpuBatchnormFwdTrain<XDataType, YDataType, ScaleBiasType, MeanVarType, ComputeDataType>(
        {3, 2, 4, 4}, {1, 2, 1, 1}, TensorLayout::NCHW, 1.0f, true, true);
}

// --- Test suite instantiations ---

using TestGpuBatchnormFwdTrainRef3DFp32 = BatchnormFwdTrainTestSuite<float>;
using TestGpuBatchnormFwdTrainRef3DFp16 = BatchnormFwdTrainTestSuite<half>;
using TestGpuBatchnormFwdTrainRef3DBfp16 = BatchnormFwdTrainTestSuite<bfloat16>;
using TestGpuBatchnormFwdTrainRef4DFp32 = BatchnormFwdTrainTestSuite<float>;
using TestGpuBatchnormFwdTrainRef4DFp16 = BatchnormFwdTrainTestSuite<half>;
using TestGpuBatchnormFwdTrainRef4DBfp16 = BatchnormFwdTrainTestSuite<bfloat16>;
using TestGpuBatchnormFwdTrainRef5DFp32 = BatchnormFwdTrainTestSuite<float>;
using TestGpuBatchnormFwdTrainRef5DFp16 = BatchnormFwdTrainTestSuite<half>;
using TestGpuBatchnormFwdTrainRef5DBfp16 = BatchnormFwdTrainTestSuite<bfloat16>;

TEST_P(TestGpuBatchnormFwdTrainRef3DFp32, MatchesCpuRef)
{
    this->runBatchnormFwdTrainTest();
}
TEST_P(TestGpuBatchnormFwdTrainRef3DFp16, MatchesCpuRef)
{
    this->runBatchnormFwdTrainTest();
}
TEST_P(TestGpuBatchnormFwdTrainRef3DBfp16, MatchesCpuRef)
{
    this->runBatchnormFwdTrainTest();
}
TEST_P(TestGpuBatchnormFwdTrainRef4DFp32, MatchesCpuRef)
{
    this->runBatchnormFwdTrainTest();
}
TEST_P(TestGpuBatchnormFwdTrainRef4DFp16, MatchesCpuRef)
{
    this->runBatchnormFwdTrainTest();
}
TEST_P(TestGpuBatchnormFwdTrainRef4DBfp16, MatchesCpuRef)
{
    this->runBatchnormFwdTrainTest();
}
TEST_P(TestGpuBatchnormFwdTrainRef5DFp32, MatchesCpuRef)
{
    this->runBatchnormFwdTrainTest();
}
TEST_P(TestGpuBatchnormFwdTrainRef5DFp16, MatchesCpuRef)
{
    this->runBatchnormFwdTrainTest();
}
TEST_P(TestGpuBatchnormFwdTrainRef5DBfp16, MatchesCpuRef)
{
    this->runBatchnormFwdTrainTest();
}

// ============================================================================
// 3D (NCL/NLC) tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdTrainRef3DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormSmall3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdTrainRef3DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormSmall3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdTrainRef3DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormSmall3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdTrainRef3DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormMedium3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdTrainRef3DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormMedium3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdTrainRef3DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormMedium3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdTrainRef3DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdTrainRef3DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge3DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdTrainRef3DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge3DTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdTrainRef3DFp32,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     ::testing::ValuesIn(getBatchnormLargeStress3DTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdTrainRef3DFp16,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     ::testing::ValuesIn(getBatchnormLargeStress3DTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdTrainRef3DBfp16,
    testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                     ::testing::ValuesIn(getBatchnormLargeStress3DTestCases())));

// ============================================================================
// 4D (NCHW/NHWC) tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdTrainRef4DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormSmall4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdTrainRef4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormSmall4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdTrainRef4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormSmall4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdTrainRef4DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormMedium4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdTrainRef4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormMedium4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdTrainRef4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormMedium4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdTrainRef4DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdTrainRef4DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge4DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdTrainRef4DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge4DTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdTrainRef4DFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     ::testing::ValuesIn(getBatchnormLargeStress4DTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdTrainRef4DFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     ::testing::ValuesIn(getBatchnormLargeStress4DTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdTrainRef4DBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     ::testing::ValuesIn(getBatchnormLargeStress4DTestCases())));

// ============================================================================
// 5D (NCDHW/NDHWC) shape tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdTrainRef5DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormSmall5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdTrainRef5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormSmall5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormFwdTrainRef5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormSmall5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdTrainRef5DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormMedium5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdTrainRef5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormMedium5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuBatchnormFwdTrainRef5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormMedium5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdTrainRef5DFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdTrainRef5DFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge5DTestCases())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuBatchnormFwdTrainRef5DBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          ::testing::ValuesIn(getBatchnormLargeEdge5DTestCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdTrainRef5DFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     ::testing::ValuesIn(getBatchnormLargeStress5DTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdTrainRef5DFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     ::testing::ValuesIn(getBatchnormLargeStress5DTestCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    TestGpuBatchnormFwdTrainRef5DBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     ::testing::ValuesIn(getBatchnormLargeStress5DTestCases())));
