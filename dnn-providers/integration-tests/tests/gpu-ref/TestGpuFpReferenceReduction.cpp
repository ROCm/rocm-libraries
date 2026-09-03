// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GpuReductionRefTestFixture.hpp"

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::reduction;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_gpu_ref;
using namespace gpu_reduction_ref_test;

// --- Valid configurations ---

TEST(TestGpuReductionRefValidation, AcceptsValidParams)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x1D({8});
    Tensor<float> y1D({1});
    EXPECT_NO_THROW(GpuFpReferenceReduction::reduce<float>(x1D, y1D, ReductionMode::ADD));

    Tensor<float> x2D({2, 4});
    Tensor<float> y2D({2, 1});
    EXPECT_NO_THROW(GpuFpReferenceReduction::reduce<float>(x2D, y2D, ReductionMode::ADD));

    Tensor<float> x3D({2, 4, 8});
    Tensor<float> y3D({2, 1, 8});
    EXPECT_NO_THROW(GpuFpReferenceReduction::reduce<float>(x3D, y3D, ReductionMode::ADD));

    Tensor<float> x4D({2, 4, 8, 8});
    Tensor<float> y4D({2, 4, 1, 1});
    EXPECT_NO_THROW(GpuFpReferenceReduction::reduce<float>(x4D, y4D, ReductionMode::ADD));

    Tensor<float> x5D({2, 4, 8, 8, 8});
    Tensor<float> y5D({2, 1, 8, 1, 8});
    EXPECT_NO_THROW(GpuFpReferenceReduction::reduce<float>(x5D, y5D, ReductionMode::ADD));
}

TEST(TestGpuReductionRefValidation, AcceptsValidParamsChannelLastLayout)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> y({2, 4, 8, 1}, TensorLayout::NHWC);
    EXPECT_NO_THROW(GpuFpReferenceReduction::reduce<float>(x, y, ReductionMode::ADD));
}

TEST(TestGpuReductionRefValidation, AcceptsValidParamsReduceSingleAxis)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({2, 4, 1, 8});
    EXPECT_NO_THROW(GpuFpReferenceReduction::reduce<float>(x, y, ReductionMode::ADD));
}

TEST(TestGpuReductionRefValidation, AcceptsValidParamsReduceMultipleAxes)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({1, 1, 8, 8});
    EXPECT_NO_THROW(GpuFpReferenceReduction::reduce<float>(x, y, ReductionMode::ADD));
}

TEST(TestGpuReductionRefValidation, AcceptsValidParamsReduceAllAxes)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({1, 1, 1, 1});
    EXPECT_NO_THROW(GpuFpReferenceReduction::reduce<float>(x, y, ReductionMode::ADD));
}

TEST(TestGpuReductionRefValidation, AcceptsValidParamsPreExistingSizeOneAxis)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({2, 4, 1, 8});
    Tensor<float> y({1, 4, 1, 8});
    EXPECT_NO_THROW(GpuFpReferenceReduction::reduce<float>(x, y, ReductionMode::ADD));
}

TEST(TestGpuReductionRefValidation, AcceptsDifferentIOLayouts)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({2, 4, 8, 8}, TensorLayout::NCHW);
    Tensor<float> y({2, 4, 8, 1}, TensorLayout::NHWC);
    EXPECT_NO_THROW(GpuFpReferenceReduction::reduce<float>(x, y, ReductionMode::ADD));
}

// --- Validation throw paths ---

TEST(TestGpuReductionRefValidation, ThrowsOnOutputRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({2, 4, 8});

    EXPECT_THROW(GpuFpReferenceReduction::reduce<float>(x, y, ReductionMode::ADD),
                 std::invalid_argument);
}

TEST(TestGpuReductionRefValidation, ThrowsOnOutputDimNeitherOneNorMatchingInput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({2, 4, 1, 4});
    EXPECT_THROW(GpuFpReferenceReduction::reduce<float>(x, y, ReductionMode::ADD),
                 std::invalid_argument);
}

TEST(TestGpuReductionRefValidation, ThrowsOnOutputDimLargerThanInput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({2, 8, 8, 1});
    EXPECT_THROW(GpuFpReferenceReduction::reduce<float>(x, y, ReductionMode::ADD),
                 std::invalid_argument);
}

TEST(TestGpuReductionRefValidation, ThrowsOnUnsupportedMode)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({2, 4, 8, 8});
    Tensor<float> y({2, 4, 8, 1});

    EXPECT_THROW(GpuFpReferenceReduction::reduce<float>(x, y, static_cast<ReductionMode>(99)),
                 std::invalid_argument);
}

// -- Data type tests ---

TEST(TestGpuReductionRefMixedType, FloatInputFloatOutput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({2, 3, 4, 4});
    Tensor<float> yCpu({2, 3, 4, 1});
    Tensor<float> yGpu({2, 3, 4, 1});

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);

    CpuFpReferenceReduction::reduce<float, float, double>(xTensor, yCpu, ReductionMode::ADD);
    GpuFpReferenceReduction::reduce<float, float, double>(xTensor, yGpu, ReductionMode::ADD);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

TEST(TestGpuReductionRefMixedType, HalfInputFloatOutput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> xTensor({2, 3, 4, 4});
    Tensor<float> yCpu({2, 3, 4, 1});
    Tensor<float> yGpu({2, 3, 4, 1});

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed);

    CpuFpReferenceReduction::reduce<half, float, double>(xTensor, yCpu, ReductionMode::ADD);
    GpuFpReferenceReduction::reduce<half, float, double>(xTensor, yGpu, ReductionMode::ADD);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

TEST(TestGpuReductionRefMixedType, HalfInputHalfOutput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> xTensor({2, 3, 4, 4});
    Tensor<half> yCpu({2, 3, 4, 1});
    Tensor<half> yGpu({2, 3, 4, 1});

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed);

    CpuFpReferenceReduction::reduce<half, half, double>(xTensor, yCpu, ReductionMode::ADD);
    GpuFpReferenceReduction::reduce<half, half, double>(xTensor, yGpu, ReductionMode::ADD);

    assertAllClose(yCpu, yGpu, getTolerance<half>());
}

TEST(TestGpuReductionRefMixedType, BfloatInputFloatOutput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<bfloat16> xTensor({2, 3, 4, 4});
    Tensor<float> yCpu({2, 3, 4, 1});
    Tensor<float> yGpu({2, 3, 4, 1});

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed);

    CpuFpReferenceReduction::reduce<bfloat16, float, double>(xTensor, yCpu, ReductionMode::ADD);
    GpuFpReferenceReduction::reduce<bfloat16, float, double>(xTensor, yGpu, ReductionMode::ADD);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

TEST(TestGpuReductionRefMixedType, BfloatInputBfloatOutput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<bfloat16> xTensor({2, 3, 4, 4});
    Tensor<bfloat16> yCpu({2, 3, 4, 1});
    Tensor<bfloat16> yGpu({2, 3, 4, 1});

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed);

    CpuFpReferenceReduction::reduce<bfloat16, bfloat16, double>(xTensor, yCpu, ReductionMode::ADD);
    GpuFpReferenceReduction::reduce<bfloat16, bfloat16, double>(xTensor, yGpu, ReductionMode::ADD);

    assertAllClose(yCpu, yGpu, getTolerance<bfloat16>());
}

// --- Channel last layout tests ---

TEST(TestGpuReductionRefChannelLast, MatchesCpuRefReducingSpatialAxes)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> yCpu({2, 4, 1, 1}, TensorLayout::NHWC);
    Tensor<float> yGpu({2, 4, 1, 1}, TensorLayout::NHWC);

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);

    CpuFpReferenceReduction::reduce<float, float, double>(xTensor, yCpu, ReductionMode::ADD);
    GpuFpReferenceReduction::reduce<float, float, double>(xTensor, yGpu, ReductionMode::ADD);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

TEST(TestGpuReductionRefChannelLast, MatchesCpuRefReducingChannelAxis)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({2, 4, 8, 8}, TensorLayout::NHWC);
    Tensor<float> yCpu({2, 1, 8, 8}, TensorLayout::NHWC);
    Tensor<float> yGpu({2, 1, 8, 8}, TensorLayout::NHWC);

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);

    CpuFpReferenceReduction::reduce<float, float, double>(xTensor, yCpu, ReductionMode::ADD);
    GpuFpReferenceReduction::reduce<float, float, double>(xTensor, yGpu, ReductionMode::ADD);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

TEST(TestGpuReductionRefChannelLast, MatchesCpuRefDifferentIOLayouts)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({2, 4, 8, 8}, TensorLayout::NCHW);
    Tensor<float> yCpu({2, 4, 1, 1}, TensorLayout::NHWC);
    Tensor<float> yGpu({2, 4, 1, 1}, TensorLayout::NHWC);

    const unsigned int seed = getGlobalTestSeed();
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);

    CpuFpReferenceReduction::reduce<float, float, double>(xTensor, yCpu, ReductionMode::ADD);
    GpuFpReferenceReduction::reduce<float, float, double>(xTensor, yGpu, ReductionMode::ADD);

    assertAllClose(yCpu, yGpu, getTolerance<float>());
}

// --- Test suite instantiations ---

using TestGpuReductionRef4DFp32 = ReductionTestSuite<float>;
using TestGpuReductionRef4DFp16 = ReductionTestSuite<half>;
using TestGpuReductionRef4DBfp16 = ReductionTestSuite<bfloat16>;
using TestGpuReductionRef5DFp32 = ReductionTestSuite<float>;
using TestGpuReductionRef5DFp16 = ReductionTestSuite<half>;
using TestGpuReductionRef5DBfp16 = ReductionTestSuite<bfloat16>;

TEST_P(TestGpuReductionRef4DFp32, MatchesCpuRef)
{
    this->runReductionTest();
}
TEST_P(TestGpuReductionRef4DFp16, MatchesCpuRef)
{
    this->runReductionTest();
}
TEST_P(TestGpuReductionRef4DBfp16, MatchesCpuRef)
{
    this->runReductionTest();
}
TEST_P(TestGpuReductionRef5DFp32, MatchesCpuRef)
{
    this->runReductionTest();
}
TEST_P(TestGpuReductionRef5DFp16, MatchesCpuRef)
{
    this->runReductionTest();
}
TEST_P(TestGpuReductionRef5DBfp16, MatchesCpuRef)
{
    this->runReductionTest();
}

// ============================================================================
// 4D tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuReductionRef4DFp32,
                         ::testing::ValuesIn(makeReductionTestCases(getReductionSmall4DShapes(),
                                                                    getReduction4DLayouts(),
                                                                    getReductionModes())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuReductionRef4DFp16,
                         ::testing::ValuesIn(makeReductionTestCases(getReductionSmall4DShapes(),
                                                                    getReduction4DLayouts(),
                                                                    getReductionModes())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuReductionRef4DBfp16,
                         ::testing::ValuesIn(makeReductionTestCases(getReductionSmall4DShapes(),
                                                                    getReduction4DLayouts(),
                                                                    getReductionModes())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuReductionRef4DFp32,
                         ::testing::ValuesIn(makeReductionTestCases(getReductionMedium4DShapes(),
                                                                    getReduction4DLayouts(),
                                                                    getReductionModes())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuReductionRef4DFp16,
                         ::testing::ValuesIn(makeReductionTestCases(getReductionMedium4DShapes(),
                                                                    getReduction4DLayouts(),
                                                                    getReductionModes())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuReductionRef4DBfp16,
                         ::testing::ValuesIn(makeReductionTestCases(getReductionMedium4DShapes(),
                                                                    getReduction4DLayouts(),
                                                                    getReductionModes())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuReductionRef4DFp32,
                         ::testing::ValuesIn(makeReductionTestCases(getReductionLarge4DShapes(),
                                                                    getReduction4DLayouts(),
                                                                    getReductionModes())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuReductionRef4DFp16,
                         ::testing::ValuesIn(makeReductionTestCases(getReductionLarge4DShapes(),
                                                                    getReduction4DLayouts(),
                                                                    getReductionModes())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuReductionRef4DBfp16,
                         ::testing::ValuesIn(makeReductionTestCases(getReductionLarge4DShapes(),
                                                                    getReduction4DLayouts(),
                                                                    getReductionModes())));

// ============================================================================
// 5D tests
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuReductionRef5DFp32,
                         ::testing::ValuesIn(makeReductionTestCases(getReductionSmall5DShapes(),
                                                                    getReduction5DLayouts(),
                                                                    getReductionModes())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuReductionRef5DFp16,
                         ::testing::ValuesIn(makeReductionTestCases(getReductionSmall5DShapes(),
                                                                    getReduction5DLayouts(),
                                                                    getReductionModes())));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuReductionRef5DBfp16,
                         ::testing::ValuesIn(makeReductionTestCases(getReductionSmall5DShapes(),
                                                                    getReduction5DLayouts(),
                                                                    getReductionModes())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuReductionRef5DFp32,
                         ::testing::ValuesIn(makeReductionTestCases(getReductionMedium5DShapes(),
                                                                    getReduction5DLayouts(),
                                                                    getReductionModes())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuReductionRef5DFp16,
                         ::testing::ValuesIn(makeReductionTestCases(getReductionMedium5DShapes(),
                                                                    getReduction5DLayouts(),
                                                                    getReductionModes())));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuReductionRef5DBfp16,
                         ::testing::ValuesIn(makeReductionTestCases(getReductionMedium5DShapes(),
                                                                    getReduction5DLayouts(),
                                                                    getReductionModes())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuReductionRef5DFp32,
                         ::testing::ValuesIn(makeReductionTestCases(getReductionLarge5DShapes(),
                                                                    getReduction5DLayouts(),
                                                                    getReductionModes())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuReductionRef5DFp16,
                         ::testing::ValuesIn(makeReductionTestCases(getReductionLarge5DShapes(),
                                                                    getReduction5DLayouts(),
                                                                    getReductionModes())));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuReductionRef5DBfp16,
                         ::testing::ValuesIn(makeReductionTestCases(getReductionLarge5DShapes(),
                                                                    getReduction5DLayouts(),
                                                                    getReductionModes())));
