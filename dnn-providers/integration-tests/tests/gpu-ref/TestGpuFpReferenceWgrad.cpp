// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GpuConvWgradRefTestFixture.hpp"

#include "ConvShapeCatalog.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gpu_conv_wgrad_ref_test;
using namespace gpu_conv_ref_test;

// One-liner subclasses — each creates a distinct GTest-visible type so that
// INSTANTIATE_TEST_SUITE_P can use clean tier-only prefixes (Smoke, Standard, Comprehensive, Full)
// while the suite name itself carries dimensionality and layout information.

// Default layout (NCL / NCHW / NCDHW)
class TestGpuConvWrwRef1dFp32 : public ConvWgradShapeSuite<float>
{
};
class TestGpuConvWrwRef2dFp32 : public ConvWgradShapeSuite<float>
{
};
class TestGpuConvWrwRef3dFp32 : public ConvWgradShapeSuite<float>
{
};
class TestGpuConvWrwRef1dFp16 : public ConvWgradShapeSuite<half>
{
};
class TestGpuConvWrwRef2dFp16 : public ConvWgradShapeSuite<half>
{
};
class TestGpuConvWrwRef3dFp16 : public ConvWgradShapeSuite<half>
{
};
class TestGpuConvWrwRef1dBfp16 : public ConvWgradShapeSuite<bfloat16>
{
};
class TestGpuConvWrwRef2dBfp16 : public ConvWgradShapeSuite<bfloat16>
{
};
class TestGpuConvWrwRef3dBfp16 : public ConvWgradShapeSuite<bfloat16>
{
};

// Channel-last layout (NLC / NHWC / NDHWC)
class TestGpuConvWrwRefNlc1dFp32 : public ConvWgradShapeSuite<float>
{
};
class TestGpuConvWrwRefNhwc2dFp32 : public ConvWgradShapeSuite<float>
{
};
class TestGpuConvWrwRefNdhwc3dFp32 : public ConvWgradShapeSuite<float>
{
};
class TestGpuConvWrwRefNlc1dFp16 : public ConvWgradShapeSuite<half>
{
};
class TestGpuConvWrwRefNhwc2dFp16 : public ConvWgradShapeSuite<half>
{
};
class TestGpuConvWrwRefNdhwc3dFp16 : public ConvWgradShapeSuite<half>
{
};
class TestGpuConvWrwRefNlc1dBfp16 : public ConvWgradShapeSuite<bfloat16>
{
};
class TestGpuConvWrwRefNhwc2dBfp16 : public ConvWgradShapeSuite<bfloat16>
{
};
class TestGpuConvWrwRefNdhwc3dBfp16 : public ConvWgradShapeSuite<bfloat16>
{
};

// Alias to avoid verbose braced-init-list issues inside EXPECT_THROW macros
using Vec = std::vector<int64_t>;

// ============================================================================
// TestGpuConvWrwRefValidation — validateInput throw paths (via wgrad)
// ============================================================================

TEST(TestGpuConvWrwRefValidation, ThrowsOnInvalidDimCount)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({8, 8});
    Tensor<float> dw({8, 8});
    Tensor<float> dy({8, 8});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(x, dw, dy, Vec{1}, Vec{1}, Vec{0}, Vec{0}),
                 std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnWeightDimMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(
                     x, dw, dy, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnOutputDimMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(
                     x, dw, dy, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnStridesSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(
        GpuFpReferenceConvolution::wgrad<float>(x, dw, dy, Vec{1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
        std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnDilationsSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(
                     x, dw, dy, Vec{1, 1}, Vec{1, 1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnPrePaddingSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(
        GpuFpReferenceConvolution::wgrad<float>(x, dw, dy, Vec{1, 1}, Vec{1, 1}, Vec{0}, Vec{0, 0}),
        std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnPostPaddingSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(
                     x, dw, dy, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnZeroStride)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(
                     x, dw, dy, Vec{0, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnNegativeDilation)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(
                     x, dw, dy, Vec{1, 1}, Vec{1, -1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnNegativePrePadding)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(
                     x, dw, dy, Vec{1, 1}, Vec{1, 1}, Vec{-1, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnNegativePostPadding)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(
                     x, dw, dy, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, -1}),
                 std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnOutputDimValueMismatch)
{
    SKIP_IF_NO_DEVICES();
    // Input [1,1,4,4], kernel [1,1,3,3], no padding, stride 1 -> expected dy [1,1,2,2]
    // Provide wrong dy dims [1,1,3,3]
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 3, 3});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(
                     x, dw, dy, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

// ============================================================================
// TestGpuConvWrwRefAsymPad — asymmetric (pre != post) padding tests
// ============================================================================

TEST(TestGpuConvWrwRefAsymPadFp32, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvWrw<float>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 1e-5f);
}

TEST(TestGpuConvWrwRefAsymPadFp16, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvWrw<half>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 5e-2f);
}

TEST(TestGpuConvWrwRefAsymPadBfp16, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvWrw<bfloat16>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 0.1f);
}

// ============================================================================
// TestGpuConvWrwRefAlphaBeta — alpha/beta scaling tests
// ============================================================================

TEST(TestGpuConvWrwRefAlphaBeta, AlphaOnly)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> dyTensor({1, 1, 2, 2});
    Tensor<float> dwRef({1, 1, 3, 3});
    Tensor<float> dwScaled({1, 1, 3, 3});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Compute with alpha=1.0
    GpuFpReferenceConvolution::wgrad<float>(xTensor, dwRef, dyTensor, {1, 1}, {1, 1}, {0, 0});

    // Compute with alpha=2.0
    GpuFpReferenceConvolution::wgrad<float>(
        xTensor, dwScaled, dyTensor, {1, 1}, {1, 1}, {0, 0}, 2.0);

    const auto* refData = dwRef.memory().hostData();
    const auto* scaledData = dwScaled.memory().hostData();

    for(size_t i = 0; i < 9; ++i)
    {
        ASSERT_NEAR(scaledData[i], 2.0f * refData[i], 1e-5f) << "Alpha scaling failed at " << i;
    }
}

TEST(TestGpuConvWrwRefAlphaBeta, BetaAccumulate)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> dyTensor({1, 1, 2, 2});
    Tensor<float> dwTensor({1, 1, 3, 3});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    dwTensor.fillWithValue(1.0f);

    // Pre-fill dw with 1.0, then compute with alpha=1.0, beta=1.0
    // Result should be wgrad(x, dy) + 1.0
    Tensor<float> dwNoAccum({1, 1, 3, 3});
    GpuFpReferenceConvolution::wgrad<float>(xTensor, dwNoAccum, dyTensor, {1, 1}, {1, 1}, {0, 0});
    GpuFpReferenceConvolution::wgrad<float>(
        xTensor, dwTensor, dyTensor, {1, 1}, {1, 1}, {0, 0}, 1.0, 1.0);

    const auto* noAccumData = dwNoAccum.memory().hostData();
    const auto* accumData = dwTensor.memory().hostData();

    for(size_t i = 0; i < 9; ++i)
    {
        ASSERT_NEAR(accumData[i], noAccumData[i] + 1.0f, 1e-5f)
            << "Beta accumulation failed at " << i;
    }
}

TEST(TestGpuConvWrwRefAlphaBeta, BetaZeroSkipsRead)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> dyTensor({1, 1, 2, 2});
    Tensor<float> dwBetaZero({1, 1, 3, 3});
    Tensor<float> dwDefault({1, 1, 3, 3});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Pre-fill with garbage — should be ignored when beta=0
    dwBetaZero.fillWithValue(999.0f);

    GpuFpReferenceConvolution::wgrad<float>(xTensor, dwDefault, dyTensor, {1, 1}, {1, 1}, {0, 0});
    GpuFpReferenceConvolution::wgrad<float>(
        xTensor, dwBetaZero, dyTensor, {1, 1}, {1, 1}, {0, 0}, 1.0, 0.0);

    const auto* defaultData = dwDefault.memory().hostData();
    const auto* betaZeroData = dwBetaZero.memory().hostData();

    for(size_t i = 0; i < 9; ++i)
    {
        ASSERT_NEAR(betaZeroData[i], defaultData[i], 1e-5f)
            << "Beta=0 should ignore pre-filled data at " << i;
    }
}

// ============================================================================
// TestGpuConvWrwRefStridedFp32 — non-packed (strided) tensor tests
// ============================================================================

TEST(TestGpuConvWrwRefStridedFp32, NonPackedInput)
{
    SKIP_IF_NO_DEVICES();

    // x: [1, 2, 4, 4] with inter-channel gap (stride[1]=32 vs packed 16)
    const std::vector<int64_t> xDims = {1, 2, 4, 4};
    const std::vector<int64_t> xStrides = {64, 32, 4, 1}; // packed would be {32, 16, 4, 1}

    Tensor<float> xTensor(xDims, xStrides);
    Tensor<float> dyTensor({1, 1, 2, 2});
    Tensor<float> dwCpu({1, 2, 3, 3});
    Tensor<float> dwGpu({1, 2, 3, 3});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::wgrad<float, float, float, double>(
        xTensor, dwCpu, dyTensor, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::wgrad<float, float, float, double>(
        xTensor, dwGpu, dyTensor, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(dwCpu, dwGpu, 1e-5f);
}

TEST(TestGpuConvWrwRefStridedFp32, NonPackedDy)
{
    SKIP_IF_NO_DEVICES();

    // dy: [1, 1, 4, 4] with inter-row gap (stride[2]=8 vs packed 4)
    const std::vector<int64_t> dyDims = {1, 1, 4, 4};
    const std::vector<int64_t> dyStrides = {32, 32, 8, 1}; // packed would be {16, 16, 4, 1}

    Tensor<float> xTensor({1, 1, 6, 6});
    Tensor<float> dyTensor(dyDims, dyStrides);
    Tensor<float> dwCpu({1, 1, 3, 3});
    Tensor<float> dwGpu({1, 1, 3, 3});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::wgrad<float, float, float, double>(
        xTensor, dwCpu, dyTensor, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::wgrad<float, float, float, double>(
        xTensor, dwGpu, dyTensor, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(dwCpu, dwGpu, 1e-5f);
}

TEST(TestGpuConvWrwRefStridedFp32, NonPackedInputAndDy)
{
    SKIP_IF_NO_DEVICES();

    // Both x and dy have non-packed strides with inter-row gaps
    const std::vector<int64_t> xDims = {1, 2, 4, 4};
    const std::vector<int64_t> xStrides = {64, 32, 6, 1}; // packed would be {32, 16, 4, 1}

    const std::vector<int64_t> dyDims = {1, 1, 2, 2};
    const std::vector<int64_t> dyStrides = {8, 8, 4, 1}; // packed would be {4, 4, 2, 1}

    Tensor<float> xTensor(xDims, xStrides);
    Tensor<float> dyTensor(dyDims, dyStrides);
    Tensor<float> dwCpu({1, 2, 3, 3});
    Tensor<float> dwGpu({1, 2, 3, 3});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::wgrad<float, float, float, double>(
        xTensor, dwCpu, dyTensor, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::wgrad<float, float, float, double>(
        xTensor, dwGpu, dyTensor, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(dwCpu, dwGpu, 1e-5f);
}

TEST(TestGpuConvWrwRefStridedFp32, NonPackedWithPadding)
{
    SKIP_IF_NO_DEVICES();

    // Non-packed input with padding to exercise both features together
    const std::vector<int64_t> xDims = {1, 2, 3, 3};
    const std::vector<int64_t> xStrides = {36, 18, 3, 1}; // packed would be {18, 9, 3, 1}

    Tensor<float> xTensor(xDims, xStrides);
    Tensor<float> dyTensor({1, 1, 3, 3});
    Tensor<float> dwCpu({1, 2, 3, 3});
    Tensor<float> dwGpu({1, 2, 3, 3});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::wgrad<float, float, float, double>(
        xTensor, dwCpu, dyTensor, {1, 1}, {1, 1}, {1, 1});

    GpuFpReferenceConvolution::wgrad<float, float, float, double>(
        xTensor, dwGpu, dyTensor, {1, 1}, {1, 1}, {1, 1});

    assertAllClose(dwCpu, dwGpu, 1e-5f);
}

// ============================================================================
// TestGpuConvWrwRefShapes — parameterized shape coverage across types
// TEST_P definitions for the fixture classes from the shared header.
// ============================================================================

// Default layout (NCL / NCHW / NCDHW)
TEST_P(TestGpuConvWrwRef1dFp32, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRef2dFp32, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRef3dFp32, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRef1dFp16, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRef2dFp16, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRef3dFp16, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRef1dBfp16, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRef2dBfp16, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRef3dBfp16, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}

// Channel-last layout (NLC / NHWC / NDHWC)
TEST_P(TestGpuConvWrwRefNlc1dFp32, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRefNhwc2dFp32, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRefNdhwc3dFp32, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRefNlc1dFp16, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRefNhwc2dFp16, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRefNdhwc3dFp16, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRefNlc1dBfp16, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRefNhwc2dBfp16, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRefNdhwc3dBfp16, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}

// ============================================================================
// Default layout (NCHW/NCDHW/NCL) instantiations.
// Smoke shapes run on every commit; Standard/Full shapes filtered via --gtest_filter.
// ============================================================================

// fp32 NCHW/NCDHW
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvWrwRef2dFp32,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvWrwRef3dFp32,
                         ::testing::ValuesIn(getSmall3dConvCases()),
                         byTag());

// fp32 NCL: 1D shapes
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvWrwRef1dFp32,
                         ::testing::ValuesIn(getSmall1dConvCases()),
                         byTag());

// fp16 NCHW/NCDHW/NCL
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvWrwRef2dFp16,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvWrwRef1dFp16,
                         ::testing::ValuesIn(getSmall1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvWrwRef3dFp16,
                         ::testing::ValuesIn(getSmall3dConvCases()),
                         byTag());

// bfp16 NCHW/NCDHW/NCL
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvWrwRef2dBfp16,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvWrwRef1dBfp16,
                         ::testing::ValuesIn(getSmall1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvWrwRef3dBfp16,
                         ::testing::ValuesIn(getSmall3dConvCases()),
                         byTag());

// ============================================================================
// Channel-last (NHWC/NDHWC/NLC) instantiations — small shapes.
// ============================================================================

// fp32 NHWC/NDHWC/NLC
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvWrwRefNhwc2dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvWrwRefNdhwc3dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvWrwRefNlc1dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall1dConvCases())),
                         byTag());

// fp16 NHWC/NDHWC/NLC
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvWrwRefNhwc2dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvWrwRefNdhwc3dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvWrwRefNlc1dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall1dConvCases())),
                         byTag());

// bfp16 NHWC/NDHWC/NLC
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvWrwRefNhwc2dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvWrwRefNdhwc3dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvWrwRefNlc1dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall1dConvCases())),
                         byTag());

// ============================================================================
// Default layout (NCL/NCHW/NCDHW) — standard/comprehensive/full shapes.
// Filter with --gtest_filter="-Standard*:Comprehensive*:Full*" for quick runs.
// ============================================================================

// fp32
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvWrwRef1dFp32,
                         ::testing::ValuesIn(getMedium1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvWrwRef2dFp32,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvWrwRef3dFp32,
                         ::testing::ValuesIn(getMedium3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvWrwRef1dFp32,
                         ::testing::ValuesIn(getLargeEdge1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvWrwRef2dFp32,
                         ::testing::ValuesIn(getLargeEdge2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvWrwRef3dFp32,
                         ::testing::ValuesIn(getLargeEdge3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvWrwRef1dFp32,
                         ::testing::ValuesIn(getLargeStress1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvWrwRef2dFp32,
                         ::testing::ValuesIn(getLargeStress2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvWrwRef3dFp32,
                         ::testing::ValuesIn(getLargeStress3dConvCases()),
                         byTag());

// fp16
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvWrwRef1dFp16,
                         ::testing::ValuesIn(getMedium1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvWrwRef2dFp16,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvWrwRef3dFp16,
                         ::testing::ValuesIn(getMedium3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvWrwRef1dFp16,
                         ::testing::ValuesIn(getLargeEdge1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvWrwRef2dFp16,
                         ::testing::ValuesIn(getLargeEdge2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvWrwRef3dFp16,
                         ::testing::ValuesIn(getLargeEdge3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvWrwRef1dFp16,
                         ::testing::ValuesIn(getLargeStress1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvWrwRef2dFp16,
                         ::testing::ValuesIn(getLargeStress2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvWrwRef3dFp16,
                         ::testing::ValuesIn(getLargeStress3dConvCases()),
                         byTag());

// bfp16
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvWrwRef1dBfp16,
                         ::testing::ValuesIn(getMedium1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvWrwRef2dBfp16,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvWrwRef3dBfp16,
                         ::testing::ValuesIn(getMedium3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvWrwRef1dBfp16,
                         ::testing::ValuesIn(getLargeEdge1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvWrwRef2dBfp16,
                         ::testing::ValuesIn(getLargeEdge2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvWrwRef3dBfp16,
                         ::testing::ValuesIn(getLargeEdge3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvWrwRef1dBfp16,
                         ::testing::ValuesIn(getLargeStress1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvWrwRef2dBfp16,
                         ::testing::ValuesIn(getLargeStress2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvWrwRef3dBfp16,
                         ::testing::ValuesIn(getLargeStress3dConvCases()),
                         byTag());

// ============================================================================
// Channel-last (NLC/NHWC/NDHWC) — standard/comprehensive/full shapes.
// ============================================================================

// fp32
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvWrwRefNlc1dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvWrwRefNhwc2dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvWrwRefNdhwc3dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvWrwRefNlc1dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvWrwRefNhwc2dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvWrwRefNdhwc3dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvWrwRefNlc1dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvWrwRefNhwc2dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvWrwRefNdhwc3dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress3dConvCases())),
                         byTag());

// fp16
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvWrwRefNlc1dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvWrwRefNhwc2dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvWrwRefNdhwc3dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvWrwRefNlc1dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvWrwRefNhwc2dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvWrwRefNdhwc3dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvWrwRefNlc1dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvWrwRefNhwc2dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvWrwRefNdhwc3dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress3dConvCases())),
                         byTag());

// bfp16
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvWrwRefNlc1dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvWrwRefNhwc2dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvWrwRefNdhwc3dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvWrwRefNlc1dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvWrwRefNhwc2dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvWrwRefNdhwc3dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvWrwRefNlc1dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvWrwRefNhwc2dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvWrwRefNdhwc3dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress3dConvCases())),
                         byTag());
