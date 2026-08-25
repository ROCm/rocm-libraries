// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GpuMatmulRefTestFixture.hpp"
#include "MatmulShapeCatalog.hpp"
#include "hipdnn-gpu-ref/GpuFpReferenceMatmul.hpp"
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMatmul.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <stdexcept>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::matmul;
using namespace hipdnn_gpu_ref;
using namespace gpu_matmul_ref_test;

// --- Valid configurations ---

TEST(TestGpuMatmulRefValidation, AcceptsValidParams2D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> a({3, 2});
    Tensor<float> b({2, 3});
    Tensor<float> c({3, 3});

    EXPECT_NO_THROW(GpuFpReferenceMatmul::matmul<float>(a, b, c));
}

TEST(TestGpuMatmulRefValidation, AcceptsValidParams3D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> a({5, 3, 2});
    Tensor<float> b({5, 2, 3});
    Tensor<float> c({5, 3, 3});

    EXPECT_NO_THROW(GpuFpReferenceMatmul::matmul<float>(a, b, c));
}

TEST(TestGpuMatmulRefValidation, AcceptsValidParams4D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> a({3, 2, 3, 2});
    Tensor<float> b({3, 2, 2, 3});
    Tensor<float> c({3, 2, 3, 3});

    EXPECT_NO_THROW(GpuFpReferenceMatmul::matmul<float>(a, b, c));
}

TEST(TestGpuMatmulRefValidation, AcceptsValidParams5D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> a({5, 3, 2, 3, 2});
    Tensor<float> b({5, 3, 2, 2, 3});
    Tensor<float> c({5, 3, 2, 3, 3});

    EXPECT_NO_THROW(GpuFpReferenceMatmul::matmul<float>(a, b, c));
}

TEST(TestGpuMatmulRefValidation, AcceptsValidBroadcast5D)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> a({1, 2, 3, 3, 2});
    Tensor<float> b({7, 6, 6, 2, 3});
    Tensor<float> c({7, 6, 6, 3, 3});

    EXPECT_NO_THROW(GpuFpReferenceMatmul::matmul<float>(a, b, c));
}

// --- validateConsistentDimensions() throw paths ---

TEST(TestGpuMatmulRefValidation, ThrowsOnInputRankTooSmall)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> a({2});
    Tensor<float> b({2});
    Tensor<float> c({2});

    EXPECT_THROW(GpuFpReferenceMatmul::matmul<float>(a, b, c), std::invalid_argument);
}

TEST(TestGpuMatmulRefValidation, ThrowsOnInputRankTooLarge)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> a({2, 2, 2, 2, 2, 2});
    Tensor<float> b({2, 2, 2, 2, 2, 2});
    Tensor<float> c({2, 2, 2, 2, 2, 2});

    EXPECT_THROW(GpuFpReferenceMatmul::matmul<float>(a, b, c), std::invalid_argument);
}

TEST(TestGpuMatmulRefValidation, ThrowsOnInputRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> a({3, 2});
    Tensor<float> b({2, 2, 3});
    Tensor<float> c({2, 3, 3});

    EXPECT_THROW(GpuFpReferenceMatmul::matmul<float>(a, b, c), std::invalid_argument);
}

TEST(TestGpuMatmulRefValidation, ThrowsOnOutputRankMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> a({2, 3, 2});
    Tensor<float> b({2, 2, 3});
    Tensor<float> c({3, 3});

    EXPECT_THROW(GpuFpReferenceMatmul::matmul<float>(a, b, c), std::invalid_argument);
}

TEST(TestGpuMatmulRefValidation, ThrowsOnBatchMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> a({2, 3, 2});
    Tensor<float> b({3, 2, 3});
    Tensor<float> c({2, 3, 3});

    EXPECT_THROW(GpuFpReferenceMatmul::matmul<float>(a, b, c), std::invalid_argument);
}

TEST(TestGpuMatmulRefValidation, ThrowsInvalidBroadcast)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> a({2, 3, 5, 3, 2});
    Tensor<float> b({3, 5, 7, 2, 3});
    Tensor<float> c({3, 5, 7, 3, 3});

    EXPECT_THROW(GpuFpReferenceMatmul::matmul<float>(a, b, c), std::invalid_argument);
}

TEST(TestGpuMatmulRefValidation, ThrowsKMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> a({2, 3, 5});
    Tensor<float> b({2, 2, 3});
    Tensor<float> c({2, 3, 3});

    EXPECT_THROW(GpuFpReferenceMatmul::matmul<float>(a, b, c), std::invalid_argument);
}

TEST(TestGpuMatmulRefValidation, ThrowsOnCBatchDimsMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> a({2, 3, 2});
    Tensor<float> b({2, 2, 3});
    Tensor<float> c({4, 3, 3});

    EXPECT_THROW(GpuFpReferenceMatmul::matmul<float>(a, b, c), std::invalid_argument);
}

TEST(TestGpuMatmulRefValidation, ThrowsOnCMMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> a({2, 3, 2});
    Tensor<float> b({2, 2, 3});
    Tensor<float> c({2, 5, 3});

    EXPECT_THROW(GpuFpReferenceMatmul::matmul<float>(a, b, c), std::invalid_argument);
}

TEST(TestGpuMatmulRefValidation, ThrowsOnCNMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> a({2, 3, 2});
    Tensor<float> b({2, 2, 3});
    Tensor<float> c({2, 3, 5});

    EXPECT_THROW(GpuFpReferenceMatmul::matmul<float>(a, b, c), std::invalid_argument);
}

// --- validateConsistentLayouts() throw paths ---

TEST(TestGpuMatmulRefValidation, ThrowsOnInputStrideOrderMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> a({2, 3, 2}, generateStrides({2, 3, 2}, {2, 0, 1}));
    Tensor<float> b({2, 2, 3}, generateStrides({2, 2, 3}, {2, 1, 0}));
    Tensor<float> c({2, 3, 3}, generateStrides({2, 3, 3}, {2, 1, 0}));

    EXPECT_THROW(GpuFpReferenceMatmul::matmul<float>(a, b, c), std::invalid_argument);
}

TEST(TestGpuMatmulRefValidation, ThrowsOnOutputStrideOrderMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> a({2, 3, 2}, generateStrides({2, 3, 2}, {2, 1, 0}));
    Tensor<float> b({2, 2, 3}, generateStrides({2, 2, 3}, {2, 1, 0}));
    Tensor<float> c({2, 3, 3}, generateStrides({2, 3, 3}, {2, 0, 1}));

    EXPECT_THROW(GpuFpReferenceMatmul::matmul<float>(a, b, c), std::invalid_argument);
}

TEST(TestGpuMatmulRefValidation, ThrowsOnNonContiguous)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> a({2, 3, 2}, generateStrides({2, 3, 2}, {2, 0, 1}));
    Tensor<float> b({2, 2, 3}, generateStrides({2, 2, 3}, {2, 0, 1}));
    Tensor<float> c({2, 3, 3}, generateStrides({2, 3, 3}, {2, 0, 1}));

    EXPECT_THROW(GpuFpReferenceMatmul::matmul<float>(a, b, c), std::invalid_argument);
}

// --- Mixed type tests ---

TEST(TestGpuMatmulRefValidation, HalfAFloatBFloatC)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> aTensor({2, 3, 2});
    Tensor<float> bTensor({2, 2, 3});
    Tensor<float> cCpu({2, 3, 3});
    Tensor<float> cGpu({2, 3, 3});

    const unsigned int seed = getGlobalTestSeed();
    aTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed);
    bTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    GpuFpReferenceMatmul::matmul<half, float, float>(aTensor, bTensor, cGpu);

    CpuFpReferenceMatmul::matmul<half, float, float>(aTensor, bTensor, cCpu);

    assertAllClose(cCpu, cGpu, getTolerance<float>(), "C");
}

TEST(TestGpuMatmulRefValidation, HalfAFloatBHalfC)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> aTensor({2, 3, 2});
    Tensor<float> bTensor({2, 2, 3});
    Tensor<half> cCpu({2, 3, 3});
    Tensor<half> cGpu({2, 3, 3});

    const unsigned int seed = getGlobalTestSeed();
    aTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed);
    bTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    GpuFpReferenceMatmul::matmul<half, float, half>(aTensor, bTensor, cGpu);

    CpuFpReferenceMatmul::matmul<half, float, half>(aTensor, bTensor, cCpu);

    assertAllClose(cCpu, cGpu, getTolerance<half>(), "C");
}

TEST(TestGpuMatmulRefValidation, HalfAHalfBFloatC)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> aTensor({2, 3, 2});
    Tensor<half> bTensor({2, 2, 3});
    Tensor<float> cCpu({2, 3, 3});
    Tensor<float> cGpu({2, 3, 3});

    const unsigned int seed = getGlobalTestSeed();
    aTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed);
    bTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 1);

    GpuFpReferenceMatmul::matmul<half, half, float>(aTensor, bTensor, cGpu);

    CpuFpReferenceMatmul::matmul<half, half, float>(aTensor, bTensor, cCpu);

    assertAllClose(cCpu, cGpu, getTolerance<float>(), "C");
}

TEST(TestGpuMatmulRefValidation, FloatAFloatBHalfC)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> aTensor({2, 3, 2});
    Tensor<float> bTensor({2, 2, 3});
    Tensor<half> cCpu({2, 3, 3});
    Tensor<half> cGpu({2, 3, 3});

    const unsigned int seed = getGlobalTestSeed();
    aTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    bTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    GpuFpReferenceMatmul::matmul<float, float, half>(aTensor, bTensor, cGpu);

    CpuFpReferenceMatmul::matmul<float, float, half>(aTensor, bTensor, cCpu);

    assertAllClose(cCpu, cGpu, getTolerance<half>(), "C");
}

TEST(TestGpuMatmulRefValidation, HalfAHalfBHalfC)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> aTensor({2, 3, 2});
    Tensor<half> bTensor({2, 2, 3});
    Tensor<half> cCpu({2, 3, 3});
    Tensor<half> cGpu({2, 3, 3});

    const unsigned int seed = getGlobalTestSeed();
    aTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed);
    bTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 1);

    GpuFpReferenceMatmul::matmul<half, half, half>(aTensor, bTensor, cGpu);

    CpuFpReferenceMatmul::matmul<half, half, half>(aTensor, bTensor, cCpu);

    assertAllClose(cCpu, cGpu, getTolerance<half>(), "C");
}

TEST(TestGpuMatmulRefValidation, BfloatABfloatBFloatC)
{
    SKIP_IF_NO_DEVICES();

    Tensor<bfloat16> aTensor({2, 3, 2});
    Tensor<bfloat16> bTensor({2, 2, 3});
    Tensor<float> cCpu({2, 3, 3});
    Tensor<float> cGpu({2, 3, 3});

    const unsigned int seed = getGlobalTestSeed();
    aTensor.fillWithRandomValues(static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed);
    bTensor.fillWithRandomValues(
        static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed + 1);

    GpuFpReferenceMatmul::matmul<bfloat16, bfloat16, float>(aTensor, bTensor, cGpu);

    CpuFpReferenceMatmul::matmul<bfloat16, bfloat16, float>(aTensor, bTensor, cCpu);

    assertAllClose(cCpu, cGpu, getTolerance<float>(), "C");
}

TEST(TestGpuMatmulRefValidation, BFloat16AHalfBBfloatC)
{
    SKIP_IF_NO_DEVICES();

    Tensor<bfloat16> aTensor({2, 3, 2});
    Tensor<half> bTensor({2, 2, 3});
    Tensor<bfloat16> cCpu({2, 3, 3});
    Tensor<bfloat16> cGpu({2, 3, 3});

    const unsigned int seed = getGlobalTestSeed();
    aTensor.fillWithRandomValues(static_cast<bfloat16>(-1.0f), static_cast<bfloat16>(1.0f), seed);
    bTensor.fillWithRandomValues(static_cast<half>(-1.0f), static_cast<half>(1.0f), seed + 1);

    GpuFpReferenceMatmul::matmul<bfloat16, half, bfloat16>(aTensor, bTensor, cGpu);

    CpuFpReferenceMatmul::matmul<bfloat16, half, bfloat16>(aTensor, bTensor, cCpu);

    assertAllClose(cCpu, cGpu, getTolerance<bfloat16>(), "C");
}

// --- Test suite instantiations ---

using TestGpuMatmulRef2DFp32 = MatmulShapeSuite<float, float, float>;
using TestGpuMatmulRef2DFp16 = MatmulShapeSuite<half, half, half>;
using TestGpuMatmulRef2DBfp16 = MatmulShapeSuite<bfloat16, bfloat16, bfloat16>;
using TestGpuMatmulRef3DFp32 = MatmulShapeSuite<float, float, float>;
using TestGpuMatmulRef3DFp16 = MatmulShapeSuite<half, half, half>;
using TestGpuMatmulRef3DBfp16 = MatmulShapeSuite<bfloat16, bfloat16, bfloat16>;
using TestGpuMatmulRef4DFp32 = MatmulShapeSuite<float, float, float>;
using TestGpuMatmulRef4DFp16 = MatmulShapeSuite<half, half, half>;
using TestGpuMatmulRef4DBfp16 = MatmulShapeSuite<bfloat16, bfloat16, bfloat16>;
using TestGpuMatmulRef5DFp32 = MatmulShapeSuite<float, float, float>;
using TestGpuMatmulRef5DFp16 = MatmulShapeSuite<half, half, half>;
using TestGpuMatmulRef5DBfp16 = MatmulShapeSuite<bfloat16, bfloat16, bfloat16>;

TEST_P(TestGpuMatmulRef2DFp32, MatchesCpuRef)
{
    this->runMatmulTest();
}
TEST_P(TestGpuMatmulRef2DFp16, MatchesCpuRef)
{
    this->runMatmulTest();
}
TEST_P(TestGpuMatmulRef2DBfp16, MatchesCpuRef)
{
    this->runMatmulTest();
}
TEST_P(TestGpuMatmulRef3DFp32, MatchesCpuRef)
{
    this->runMatmulTest();
}
TEST_P(TestGpuMatmulRef3DFp16, MatchesCpuRef)
{
    this->runMatmulTest();
}
TEST_P(TestGpuMatmulRef3DBfp16, MatchesCpuRef)
{
    this->runMatmulTest();
}
TEST_P(TestGpuMatmulRef4DFp32, MatchesCpuRef)
{
    this->runMatmulTest();
}
TEST_P(TestGpuMatmulRef4DFp16, MatchesCpuRef)
{
    this->runMatmulTest();
}
TEST_P(TestGpuMatmulRef4DBfp16, MatchesCpuRef)
{
    this->runMatmulTest();
}
TEST_P(TestGpuMatmulRef5DFp32, MatchesCpuRef)
{
    this->runMatmulTest();
}
TEST_P(TestGpuMatmulRef5DFp16, MatchesCpuRef)
{
    this->runMatmulTest();
}
TEST_P(TestGpuMatmulRef5DBfp16, MatchesCpuRef)
{
    this->runMatmulTest();
}

// ========
// 2D tests
// ========

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuMatmulRef2DFp32,
                         ::testing::ValuesIn(getMatmulSmall2DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuMatmulRef2DFp16,
                         ::testing::ValuesIn(getMatmulSmall2DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuMatmulRef2DBfp16,
                         ::testing::ValuesIn(getMatmulSmall2DTestCases()));

INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuMatmulRef2DFp32,
                         ::testing::ValuesIn(getMatmulMedium2DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuMatmulRef2DFp16,
                         ::testing::ValuesIn(getMatmulMedium2DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuMatmulRef2DBfp16,
                         ::testing::ValuesIn(getMatmulMedium2DTestCases()));

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuMatmulRef2DFp32,
                         ::testing::ValuesIn(getMatmulLarge2DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuMatmulRef2DFp16,
                         ::testing::ValuesIn(getMatmulLarge2DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuMatmulRef2DBfp16,
                         ::testing::ValuesIn(getMatmulLarge2DTestCases()));

INSTANTIATE_TEST_SUITE_P(Full, TestGpuMatmulRef2DFp32, ::testing::ValuesIn([]() {
                             auto v = getMatmulSmall2DTestCases();
                             auto m = getMatmulMedium2DTestCases();
                             auto l = getMatmulLarge2DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuMatmulRef2DFp16, ::testing::ValuesIn([]() {
                             auto v = getMatmulSmall2DTestCases();
                             auto m = getMatmulMedium2DTestCases();
                             auto l = getMatmulLarge2DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuMatmulRef2DBfp16, ::testing::ValuesIn([]() {
                             auto v = getMatmulSmall2DTestCases();
                             auto m = getMatmulMedium2DTestCases();
                             auto l = getMatmulLarge2DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));

// ========
// 3D tests
// ========

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuMatmulRef3DFp32,
                         ::testing::ValuesIn(getMatmulSmall3DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuMatmulRef3DFp16,
                         ::testing::ValuesIn(getMatmulSmall3DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuMatmulRef3DBfp16,
                         ::testing::ValuesIn(getMatmulSmall3DTestCases()));

INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuMatmulRef3DFp32,
                         ::testing::ValuesIn(getMatmulMedium3DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuMatmulRef3DFp16,
                         ::testing::ValuesIn(getMatmulMedium3DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuMatmulRef3DBfp16,
                         ::testing::ValuesIn(getMatmulMedium3DTestCases()));

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuMatmulRef3DFp32,
                         ::testing::ValuesIn(getMatmulLarge3DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuMatmulRef3DFp16,
                         ::testing::ValuesIn(getMatmulLarge3DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuMatmulRef3DBfp16,
                         ::testing::ValuesIn(getMatmulLarge3DTestCases()));

INSTANTIATE_TEST_SUITE_P(Full, TestGpuMatmulRef3DFp32, ::testing::ValuesIn([]() {
                             auto v = getMatmulSmall3DTestCases();
                             auto m = getMatmulMedium3DTestCases();
                             auto l = getMatmulLarge3DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuMatmulRef3DFp16, ::testing::ValuesIn([]() {
                             auto v = getMatmulSmall3DTestCases();
                             auto m = getMatmulMedium3DTestCases();
                             auto l = getMatmulLarge3DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuMatmulRef3DBfp16, ::testing::ValuesIn([]() {
                             auto v = getMatmulSmall3DTestCases();
                             auto m = getMatmulMedium3DTestCases();
                             auto l = getMatmulLarge3DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));

// ========
// 4D tests
// ========

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuMatmulRef4DFp32,
                         ::testing::ValuesIn(getMatmulSmall4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuMatmulRef4DFp16,
                         ::testing::ValuesIn(getMatmulSmall4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuMatmulRef4DBfp16,
                         ::testing::ValuesIn(getMatmulSmall4DTestCases()));

INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuMatmulRef4DFp32,
                         ::testing::ValuesIn(getMatmulMedium4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuMatmulRef4DFp16,
                         ::testing::ValuesIn(getMatmulMedium4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuMatmulRef4DBfp16,
                         ::testing::ValuesIn(getMatmulMedium4DTestCases()));

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuMatmulRef4DFp32,
                         ::testing::ValuesIn(getMatmulLarge4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuMatmulRef4DFp16,
                         ::testing::ValuesIn(getMatmulLarge4DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuMatmulRef4DBfp16,
                         ::testing::ValuesIn(getMatmulLarge4DTestCases()));

INSTANTIATE_TEST_SUITE_P(Full, TestGpuMatmulRef4DFp32, ::testing::ValuesIn([]() {
                             auto v = getMatmulSmall4DTestCases();
                             auto m = getMatmulMedium4DTestCases();
                             auto l = getMatmulLarge4DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuMatmulRef4DFp16, ::testing::ValuesIn([]() {
                             auto v = getMatmulSmall4DTestCases();
                             auto m = getMatmulMedium4DTestCases();
                             auto l = getMatmulLarge4DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuMatmulRef4DBfp16, ::testing::ValuesIn([]() {
                             auto v = getMatmulSmall4DTestCases();
                             auto m = getMatmulMedium4DTestCases();
                             auto l = getMatmulLarge4DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));

// ========
// 5D tests
// ========

INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuMatmulRef5DFp32,
                         ::testing::ValuesIn(getMatmulSmall5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuMatmulRef5DFp16,
                         ::testing::ValuesIn(getMatmulSmall5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Quick,
                         TestGpuMatmulRef5DBfp16,
                         ::testing::ValuesIn(getMatmulSmall5DTestCases()));

INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuMatmulRef5DFp32,
                         ::testing::ValuesIn(getMatmulMedium5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuMatmulRef5DFp16,
                         ::testing::ValuesIn(getMatmulMedium5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuMatmulRef5DBfp16,
                         ::testing::ValuesIn(getMatmulMedium5DTestCases()));

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuMatmulRef5DFp32,
                         ::testing::ValuesIn(getMatmulLarge5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuMatmulRef5DFp16,
                         ::testing::ValuesIn(getMatmulLarge5DTestCases()));
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuMatmulRef5DBfp16,
                         ::testing::ValuesIn(getMatmulLarge5DTestCases()));

INSTANTIATE_TEST_SUITE_P(Full, TestGpuMatmulRef5DFp32, ::testing::ValuesIn([]() {
                             auto v = getMatmulSmall5DTestCases();
                             auto m = getMatmulMedium5DTestCases();
                             auto l = getMatmulLarge5DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuMatmulRef5DFp16, ::testing::ValuesIn([]() {
                             auto v = getMatmulSmall5DTestCases();
                             auto m = getMatmulMedium5DTestCases();
                             auto l = getMatmulLarge5DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));
INSTANTIATE_TEST_SUITE_P(Full, TestGpuMatmulRef5DBfp16, ::testing::ValuesIn([]() {
                             auto v = getMatmulSmall5DTestCases();
                             auto m = getMatmulMedium5DTestCases();
                             auto l = getMatmulLarge5DTestCases();
                             v.insert(v.end(), m.begin(), m.end());
                             v.insert(v.end(), l.begin(), l.end());
                             return v;
                         }()));

// ============================================================================
// Edge case tests with DISABLED_ prefix to avoid running in CI.
// Run the tests manually with --gtest_also_run_disabled_tests
// --gtest_filter=*TestGpuMatmulRefEdgeCaseValidation* flags.
// ============================================================================

namespace
{

int64_t getMaxMatrixMForCurrentDevice()
{
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
    {
        // No devices available, return a default value to skip the tests.
        return 1;
    }

    int deviceId = 0;
    EXPECT_EQ(hipGetDevice(&deviceId), hipSuccess);

    hipDeviceProp_t props{};
    EXPECT_EQ(hipGetDeviceProperties(&props, deviceId), hipSuccess);

    return static_cast<int64_t>(GpuFpReferenceMatmul::TILE_SIZE)
           * static_cast<int64_t>(props.maxGridSize[0]);
}

int64_t getMaxMatrixNForCurrentDevice()
{
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
    {
        // No devices available, return a default value to skip the tests.
        return 1;
    }

    int deviceId = 0;
    EXPECT_EQ(hipGetDevice(&deviceId), hipSuccess);

    hipDeviceProp_t props{};
    EXPECT_EQ(hipGetDeviceProperties(&props, deviceId), hipSuccess);

    return static_cast<int64_t>(GpuFpReferenceMatmul::TILE_SIZE)
           * static_cast<int64_t>(props.maxGridSize[1]);
}

} // namespace

TEST(TestGpuMatmulRefEdgeCaseValidation, DISABLED_MAtMaxMMinusOneSucceeds)
{
    SKIP_IF_NO_DEVICES();

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    ASSERT_EQ(hipMemGetInfo(&freeBytes, &totalBytes), hipSuccess);

    const int64_t maxM = getMaxMatrixMForCurrentDevice();

    // Calculate the required memory for a, b and c tensors
    const size_t requiredBytes
        = (static_cast<size_t>(maxM) - 1 + 1 + static_cast<size_t>(maxM) - 1) // a + b + c
          * sizeof(float);
    if(requiredBytes > freeBytes)
    {
        GTEST_SKIP() << "Insufficient GPU memory for the test. Required: " << requiredBytes
                     << " bytes, Free: " << freeBytes << " bytes.";
    }

    Tensor<float> a({maxM - 1, 1});
    Tensor<float> b({1, 1});
    Tensor<float> c({maxM - 1, 1});

    EXPECT_NO_THROW(GpuFpReferenceMatmul::matmul(a, b, c));
}

TEST(TestGpuMatmulRefEdgeCaseValidation, DISABLED_MAtMaxMSucceeds)
{
    SKIP_IF_NO_DEVICES();

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    ASSERT_EQ(hipMemGetInfo(&freeBytes, &totalBytes), hipSuccess);

    const int64_t maxM = getMaxMatrixMForCurrentDevice();

    // Calculate the required memory for a, b and c tensors
    const size_t requiredBytes
        = (static_cast<size_t>(maxM) + 1 + static_cast<size_t>(maxM)) // a + b + c
          * sizeof(float);
    if(requiredBytes > freeBytes)
    {
        GTEST_SKIP() << "Insufficient GPU memory for the test. Required: " << requiredBytes
                     << " bytes, Free: " << freeBytes << " bytes.";
    }

    Tensor<float> a({maxM, 1});
    Tensor<float> b({1, 1});
    Tensor<float> c({maxM, 1});

    EXPECT_NO_THROW(GpuFpReferenceMatmul::matmul(a, b, c));
}

TEST(TestGpuMatmulRefEdgeCaseValidation, DISABLED_MAboveMaxMThrows)
{
    SKIP_IF_NO_DEVICES();

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    ASSERT_EQ(hipMemGetInfo(&freeBytes, &totalBytes), hipSuccess);

    const int64_t maxM = getMaxMatrixMForCurrentDevice();

    // Calculate the required memory for a, b and c tensors
    const size_t requiredBytes
        = (static_cast<size_t>(maxM) + 1 + 1 + static_cast<size_t>(maxM) + 1) // a + b + c
          * sizeof(float);
    if(requiredBytes > freeBytes)
    {
        GTEST_SKIP() << "Insufficient GPU memory for the test. Required: " << requiredBytes
                     << " bytes, Free: " << freeBytes << " bytes.";
    }

    Tensor<float> a({maxM + 1, 1});
    Tensor<float> b({1, 1});
    Tensor<float> c({maxM + 1, 1});

    EXPECT_THROW(GpuFpReferenceMatmul::matmul(a, b, c), std::runtime_error);
}

TEST(TestGpuMatmulRefEdgeCaseValidation, DISABLED_NAtMaxNMinusOneSucceeds)
{
    SKIP_IF_NO_DEVICES();

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    ASSERT_EQ(hipMemGetInfo(&freeBytes, &totalBytes), hipSuccess);

    const int64_t maxN = getMaxMatrixNForCurrentDevice();

    // Calculate the required memory for a, b and c tensors
    const size_t requiredBytes
        = (static_cast<size_t>(maxN) - 1 + 1 + static_cast<size_t>(maxN) - 1) // a + b + c
          * sizeof(float);
    if(requiredBytes > freeBytes)
    {
        GTEST_SKIP() << "Insufficient GPU memory for the test. Required: " << requiredBytes
                     << " bytes, Free: " << freeBytes << " bytes.";
    }

    Tensor<float> a({1, 1});
    Tensor<float> b({1, maxN - 1});
    Tensor<float> c({1, maxN - 1});

    EXPECT_NO_THROW(GpuFpReferenceMatmul::matmul(a, b, c));
}

TEST(TestGpuMatmulRefEdgeCaseValidation, DISABLED_NAtMaxNSucceeds)
{
    SKIP_IF_NO_DEVICES();

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    ASSERT_EQ(hipMemGetInfo(&freeBytes, &totalBytes), hipSuccess);

    const int64_t maxN = getMaxMatrixNForCurrentDevice();

    // Calculate the required memory for a, b and c tensors
    const size_t requiredBytes
        = (static_cast<size_t>(maxN) + 1 + static_cast<size_t>(maxN)) // a + b + c
          * sizeof(float);
    if(requiredBytes > freeBytes)
    {
        GTEST_SKIP() << "Insufficient GPU memory for the test. Required: " << requiredBytes
                     << " bytes, Free: " << freeBytes << " bytes.";
    }

    Tensor<float> a({1, 1});
    Tensor<float> b({1, maxN});
    Tensor<float> c({1, maxN});

    EXPECT_NO_THROW(GpuFpReferenceMatmul::matmul(a, b, c));
}

TEST(TestGpuMatmulRefEdgeCaseValidation, DISABLED_NAboveMaxNThrows)
{
    SKIP_IF_NO_DEVICES();

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    ASSERT_EQ(hipMemGetInfo(&freeBytes, &totalBytes), hipSuccess);

    const int64_t maxN = getMaxMatrixNForCurrentDevice();

    // Calculate the required memory for a, b and c tensors
    const size_t requiredBytes
        = (static_cast<size_t>(maxN) + 1 + 1 + static_cast<size_t>(maxN) + 1) // a + b + c
          * sizeof(float);
    if(requiredBytes > freeBytes)
    {
        GTEST_SKIP() << "Insufficient GPU memory for the test. Required: " << requiredBytes
                     << " bytes, Free: " << freeBytes << " bytes.";
    }

    Tensor<float> a({1, 1});
    Tensor<float> b({1, maxN + 1});
    Tensor<float> c({1, maxN + 1});

    EXPECT_THROW(GpuFpReferenceMatmul::matmul(a, b, c), std::runtime_error);
}

TEST(TestGpuMatmulRefEdgeCaseValidation, DISABLED_BeyondInt32MatrixIfMemoryAllows)
{
    SKIP_IF_NO_DEVICES();

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    ASSERT_EQ(hipMemGetInfo(&freeBytes, &totalBytes), hipSuccess);

    constexpr int64_t M = 128;
    constexpr int64_t N = 128;
    // NOTE: K in this test should be 2^25+1, but is reduced here due to
    // slow CPU fill/reference functions. Revisit once rocRAND-based GPU fill and
    // golden references for large tensors are available.
    constexpr int64_t K = 1000000; // 128 million elements for matrix A and B

    // Calculate the required memory for a, b and c tensors
    const size_t requiredBytes = (M * K + K * N + 2 * M * N) // a + b + 2 * c (CPU + GPU)
                                 * sizeof(float);
    if(requiredBytes > freeBytes)
    {
        GTEST_SKIP() << "Insufficient GPU memory for the test. Required: " << requiredBytes
                     << " bytes, Free: " << freeBytes << " bytes.";
    }

    Tensor<float> a({M, K});
    Tensor<float> b({K, N});
    Tensor<float> cCpu({M, N});
    Tensor<float> cGpu({M, N});

    const unsigned int seed = getGlobalTestSeed();
    a.fillWithRandomValues(-1.0f, 1.0f, seed);
    b.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceMatmul::matmul(a, b, cCpu);
    GpuFpReferenceMatmul::matmul(a, b, cGpu);

    // Substantially raised tolerance due to the many elements that need to be multiplied and added
    assertAllClose(cCpu, cGpu, 250 * getTolerance<float>(), "C");
}
