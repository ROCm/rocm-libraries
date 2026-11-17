// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GpuBatchnormBwdRefTestFixture.hpp"

using namespace hipdnn_data_sdk::types;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_gpu_ref;
using namespace hipdnn_test_sdk::utilities;
using namespace gpu_batchnorm_bwd_ref_test;
using namespace gpu_batchnorm_ref_test;

namespace
{

template <typename DataType>
using BatchnormBwdPureTestSuite
    = BatchnormBwdTestSuite<DataType, DataType, DataType, DataType, DataType, float>;

using TestGpuBatchnormBwdRefFp32 = BatchnormBwdPureTestSuite<float>;
using TestGpuBatchnormBwdRefFp16 = BatchnormBwdPureTestSuite<half>;
using TestGpuBatchnormBwdRefBfp16 = BatchnormBwdPureTestSuite<bfloat16>;
using TestGpuBatchnormBwdRefUpcast
    = BatchnormBwdTestSuite<half, bfloat16, float, float, float, float>;
using TestGpuBatchnormBwdRefDowncast
    = BatchnormBwdTestSuite<float, float, half, half, bfloat16, float>;
using TestGpuBatchnormBwdRefDouble
    = BatchnormBwdTestSuite<double, double, double, double, double, double>;

TEST_P(TestGpuBatchnormBwdRefFp32, MatchesCpuRef)
{
    this->runBatchnormBwdTest();
}

TEST_P(TestGpuBatchnormBwdRefFp16, MatchesCpuRef)
{
    this->runBatchnormBwdTest();
}

TEST_P(TestGpuBatchnormBwdRefBfp16, MatchesCpuRef)
{
    this->runBatchnormBwdTest();
}

TEST_P(TestGpuBatchnormBwdRefUpcast, MatchesCpuRef)
{
    this->runBatchnormBwdTest();
}

TEST_P(TestGpuBatchnormBwdRefDowncast, MatchesCpuRef)
{
    this->runBatchnormBwdTest();
}

TEST_P(TestGpuBatchnormBwdRefDouble, MatchesCpuRef)
{
    this->runBatchnormBwdTest();
}

const std::vector<BatchnormTestCase> QUICK_4D_CASES = {{{2, 3, 4, 5}}};
const std::vector<BatchnormTestCase> STANDARD_4D_CASES = {{{4, 3, 17, 19}}};
const std::vector<BatchnormTestCase> STANDARD_5D_CASES = {{{3, 2, 5, 7, 11}}};

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormBwdRefFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(QUICK_4D_CASES),
                                          testing::Bool()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormBwdRefFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(QUICK_4D_CASES),
                                          testing::Bool()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuBatchnormBwdRefBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(QUICK_4D_CASES),
                                          testing::Bool()));

INSTANTIATE_TEST_SUITE_P(Standard4DGridStride,
                         TestGpuBatchnormBwdRefFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(STANDARD_4D_CASES),
                                          testing::Bool()));
INSTANTIATE_TEST_SUITE_P(Standard5DGridStride,
                         TestGpuBatchnormBwdRefFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(STANDARD_5D_CASES),
                                          testing::Bool()));

INSTANTIATE_TEST_SUITE_P(MixedPrecisionUpcast,
                         TestGpuBatchnormBwdRefUpcast,
                         testing::Combine(testing::Values(TensorLayout::NHWC),
                                          testing::ValuesIn(QUICK_4D_CASES),
                                          testing::Bool()));
INSTANTIATE_TEST_SUITE_P(MixedPrecisionDowncast,
                         TestGpuBatchnormBwdRefDowncast,
                         testing::Combine(testing::Values(TensorLayout::NCHW),
                                          testing::ValuesIn(QUICK_4D_CASES),
                                          testing::Bool()));
INSTANTIATE_TEST_SUITE_P(ComprehensiveDouble,
                         TestGpuBatchnormBwdRefDouble,
                         testing::Combine(testing::Values(TensorLayout::NCHW),
                                          testing::ValuesIn(QUICK_4D_CASES),
                                          testing::Bool()));

TEST(TestGpuBatchnormBwdRefValidation, AcceptsRanksThreeThroughFiveAndAffineBroadcast)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> dy3D({2, 4, 8});
    Tensor<float> x3D({2, 4, 8});
    Tensor<float> scale3D({1, 4});
    Tensor<float> dx3D({2, 4, 8});
    Tensor<float> dscale3D({1, 4, 1});
    Tensor<float> dbias3D({1, 4, 1, 1});
    EXPECT_NO_THROW(GpuFpReferenceBatchnorm::backward(dy3D, x3D, scale3D, dx3D, dscale3D, dbias3D));

    runGpuVsCpuBatchnormBackward<float, float, float, float, float, float>(
        {2, 4, 3, 5}, TensorLayout::NHWC, true);
    runGpuVsCpuBatchnormBackward<float, float, float, float, float, float>(
        {2, 4, 2, 3, 5}, TensorLayout::NCDHW, false);
}

TEST(TestGpuBatchnormBwdRefValidation, RejectsIncompleteSavedStats)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 3, 4});
    Tensor<float> x({2, 3, 4});
    Tensor<float> scale({1, 3, 1});
    Tensor<float> dx({2, 3, 4});
    Tensor<float> dscale({1, 3, 1});
    Tensor<float> dbias({1, 3, 1});
    Tensor<float> mean({1, 3, 1});

    EXPECT_THROW((GpuFpReferenceBatchnorm::backward<float, float, float, float, float, float>(
                     dy, x, scale, dx, dscale, dbias, &mean, nullptr)),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormBwdRefValidation, RejectsInputRankOutsideThreeThroughFive)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 3});
    Tensor<float> x({2, 3});
    Tensor<float> scale({1, 3});
    Tensor<float> dx({2, 3});
    Tensor<float> dscale({1, 3});
    Tensor<float> dbias({1, 3});

    EXPECT_THROW(GpuFpReferenceBatchnorm::backward(dy, x, scale, dx, dscale, dbias),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormBwdRefValidation, RejectsMismatchedIoShapes)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 3, 4, 4});
    Tensor<float> x({2, 3, 4, 4});
    Tensor<float> scale({1, 3, 1, 1});
    Tensor<float> dx({2, 3, 4, 3});
    Tensor<float> dscale({1, 3, 1, 1});
    Tensor<float> dbias({1, 3, 1, 1});

    EXPECT_THROW(GpuFpReferenceBatchnorm::backward(dy, x, scale, dx, dscale, dbias),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormBwdRefValidation, RejectsMismatchedDyShape)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 3, 4, 3});
    Tensor<float> x({2, 3, 4, 4});
    Tensor<float> scale({1, 3, 1, 1});
    Tensor<float> dx({2, 3, 4, 4});
    Tensor<float> dscale({1, 3, 1, 1});
    Tensor<float> dbias({1, 3, 1, 1});

    EXPECT_THROW(GpuFpReferenceBatchnorm::backward(dy, x, scale, dx, dscale, dbias),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormBwdRefValidation, RejectsNonChannelOnlyAffineShape)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 3, 4, 4});
    Tensor<float> x({2, 3, 4, 4});
    Tensor<float> scale({1, 3, 2, 1});
    Tensor<float> dx({2, 3, 4, 4});
    Tensor<float> dscale({1, 3, 1, 1});
    Tensor<float> dbias({1, 3, 1, 1});

    EXPECT_THROW(GpuFpReferenceBatchnorm::backward(dy, x, scale, dx, dscale, dbias),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormBwdRefValidation, RejectsInconsistentLayout)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 3, 4, 4}, TensorLayout::NCHW);
    Tensor<float> x({2, 3, 4, 4}, TensorLayout::NHWC);
    Tensor<float> scale({1, 3, 1, 1}, TensorLayout::NHWC);
    Tensor<float> dx({2, 3, 4, 4}, TensorLayout::NHWC);
    Tensor<float> dscale({1, 3, 1, 1}, TensorLayout::NHWC);
    Tensor<float> dbias({1, 3, 1, 1}, TensorLayout::NHWC);

    EXPECT_THROW(GpuFpReferenceBatchnorm::backward(dy, x, scale, dx, dscale, dbias),
                 std::invalid_argument);
}

TEST(TestGpuBatchnormBwdRefValidation, RejectsNonPackedInput)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dy({2, 3, 4}, {12, 4, 1});
    Tensor<float> x({2, 3, 4}, {24, 8, 1});
    Tensor<float> scale({1, 3}, {3, 1});
    Tensor<float> dx({2, 3, 4}, {12, 4, 1});
    Tensor<float> dscale({1, 3}, {3, 1});
    Tensor<float> dbias({1, 3}, {3, 1});

    EXPECT_THROW(GpuFpReferenceBatchnorm::backward(dy, x, scale, dx, dscale, dbias),
                 std::invalid_argument);
}

} // namespace
