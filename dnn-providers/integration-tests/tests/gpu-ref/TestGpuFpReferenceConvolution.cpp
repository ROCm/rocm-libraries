// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GpuConvFwdRefTestFixture.hpp"

#include "ConvShapeCatalog.hpp"

#include <hipdnn_test_sdk/utilities/ConvolutionValidation.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gpu_conv_fwd_ref_test;
using namespace gpu_conv_ref_test;

// One-liner subclasses — each creates a distinct GTest-visible type so that
// INSTANTIATE_TEST_SUITE_P can use clean tier-only prefixes (Smoke, Standard, Comprehensive, Full)
// while the suite name itself carries dimensionality and layout information.

// Default layout (NCL / NCHW / NCDHW)
class TestGpuConvFwdRef1dFp32 : public ConvFwdShapeSuite<float>
{
};
class TestGpuConvFwdRef2dFp32 : public ConvFwdShapeSuite<float>
{
};
class TestGpuConvFwdRef3dFp32 : public ConvFwdShapeSuite<float>
{
};
class TestGpuConvFwdRef1dFp16 : public ConvFwdShapeSuite<half>
{
};
class TestGpuConvFwdRef2dFp16 : public ConvFwdShapeSuite<half>
{
};
class TestGpuConvFwdRef3dFp16 : public ConvFwdShapeSuite<half>
{
};
class TestGpuConvFwdRef1dBfp16 : public ConvFwdShapeSuite<bfloat16>
{
};
class TestGpuConvFwdRef2dBfp16 : public ConvFwdShapeSuite<bfloat16>
{
};
class TestGpuConvFwdRef3dBfp16 : public ConvFwdShapeSuite<bfloat16>
{
};

// Channel-last layout (NLC / NHWC / NDHWC)
class TestGpuConvFwdRefNlc1dFp32 : public ConvFwdShapeSuite<float>
{
};
class TestGpuConvFwdRefNhwc2dFp32 : public ConvFwdShapeSuite<float>
{
};
class TestGpuConvFwdRefNdhwc3dFp32 : public ConvFwdShapeSuite<float>
{
};
class TestGpuConvFwdRefNlc1dFp16 : public ConvFwdShapeSuite<half>
{
};
class TestGpuConvFwdRefNhwc2dFp16 : public ConvFwdShapeSuite<half>
{
};
class TestGpuConvFwdRefNdhwc3dFp16 : public ConvFwdShapeSuite<half>
{
};
class TestGpuConvFwdRefNlc1dBfp16 : public ConvFwdShapeSuite<bfloat16>
{
};
class TestGpuConvFwdRefNhwc2dBfp16 : public ConvFwdShapeSuite<bfloat16>
{
};
class TestGpuConvFwdRefNdhwc3dBfp16 : public ConvFwdShapeSuite<bfloat16>
{
};

// Alias to avoid verbose braced-init-list issues inside EXPECT_THROW macros
using Vec = std::vector<int64_t>;

// ============================================================================
// TestConvolutionValidation — input validation error paths (CPU-only).
// The parameterized shape tests (INSTANTIATE_TEST_SUITE_P) only run valid
// configurations. These verify that invalid inputs (mismatched dims, zero
// stride, negative dilation/padding) are properly rejected with exceptions.
// ============================================================================

TEST(TestConvolutionValidation, AcceptsValidParams)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_NO_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
        x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}));
}

TEST(TestConvolutionValidation, ThrowsOnWeightDimMismatch)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnOutputDimMismatch)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnStridesSizeMismatch)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnDilationsSizeMismatch)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnPrePaddingSizeMismatch)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnPostPaddingSizeMismatch)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnZeroStride)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{0, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnNegativeDilation)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, -1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnNegativePrePadding)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{-1, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnNegativePostPadding)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, -1}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnOutputDimValueMismatch)
{
    // Input [1,1,4,4], kernel [1,1,3,3], no padding, stride 1 -> expected output [1,1,2,2]
    // Provide wrong output dims [1,1,3,3]
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 3, 3});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

// ============================================================================
// TestGpuConvFwdRefValidation — same validation error paths as above, but
// exercised through the GPU code path (GpuFpReferenceConvolution::fprop).
// Verifies that the GPU entry point rejects invalid inputs before launching
// any kernels.
// ============================================================================

TEST(TestGpuConvFwdRefValidation, ThrowsOnInvalidDimCount)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({8, 8});
    Tensor<float> w({8, 8});
    Tensor<float> y({8, 8});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(x, w, y, Vec{1}, Vec{1}, Vec{0}, Vec{0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnWeightDimMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnOutputDimMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnStridesSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(
        GpuFpReferenceConvolution::fprop<float>(x, w, y, Vec{1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
        std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnDilationsSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnPrePaddingSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(
        GpuFpReferenceConvolution::fprop<float>(x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0}, Vec{0, 0}),
        std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnPostPaddingSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnZeroStride)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{0, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnNegativeDilation)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, -1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnNegativePrePadding)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{-1, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnNegativePostPadding)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, -1}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnOutputDimValueMismatch)
{
    SKIP_IF_NO_DEVICES();
    // Input [1,1,4,4], kernel [1,1,3,3], no padding, stride 1 -> expected output [1,1,2,2]
    // Provide wrong output dims [1,1,3,3]
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 3, 3});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

// ============================================================================
// Standalone tests (TEST, not TEST_P).
// These cover execution features that the parameterized shape tests do not
// vary: asymmetric padding, alpha/beta scaling, non-packed strides, int8
// types, TF32 mode, and mixed input/weight precision.
// ============================================================================

// ============================================================================
// TestGpuConvFwdRefAsymPad — asymmetric padding (pre_pad != post_pad).
// Shape tests use symmetric padding only; these verify the kernel handles
// different pad_before vs pad_after values correctly.
// ============================================================================

TEST(TestGpuConvFwdRefAsymPadFp32, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 1e-5f);
}

TEST(TestGpuConvFwdRefAsymPadFp16, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<half>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 5e-2f);
}

TEST(TestGpuConvFwdRefAsymPadBfp16, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<bfloat16>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 0.1f);
}

// ============================================================================
// TestGpuConvFwdRefAlphaBeta — alpha/beta scaling.
// Shape tests always use alpha=1, beta=0. These verify that non-default
// scaling (alpha!=1, beta!=0 accumulation) works correctly.
// ============================================================================

TEST(TestGpuConvFwdRefAlphaBeta, AlphaOnly)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yRef({1, 1, 2, 2});
    Tensor<float> yScaled({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Compute with alpha=1.0
    GpuFpReferenceConvolution::fprop<float>(xTensor, wTensor, yRef, {1, 1}, {1, 1}, {0, 0});

    // Compute with alpha=2.0
    GpuFpReferenceConvolution::fprop<float>(xTensor, wTensor, yScaled, {1, 1}, {1, 1}, {0, 0}, 2.0);

    const auto* refData = yRef.memory().hostData();
    const auto* scaledData = yScaled.memory().hostData();

    for(size_t i = 0; i < 4; ++i)
    {
        ASSERT_NEAR(scaledData[i], 2.0f * refData[i], 1e-5f) << "Alpha scaling failed at " << i;
    }
}

TEST(TestGpuConvFwdRefAlphaBeta, BetaAccumulate)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yTensor({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    yTensor.fillWithValue(1.0f);

    // Pre-fill y with 1.0, then compute with alpha=1.0, beta=1.0
    // Result should be conv(x,w) + 1.0
    Tensor<float> yNoAccum({1, 1, 2, 2});
    GpuFpReferenceConvolution::fprop<float>(xTensor, wTensor, yNoAccum, {1, 1}, {1, 1}, {0, 0});
    GpuFpReferenceConvolution::fprop<float>(
        xTensor, wTensor, yTensor, {1, 1}, {1, 1}, {0, 0}, 1.0, 1.0);

    const auto* noAccumData = yNoAccum.memory().hostData();
    const auto* accumData = yTensor.memory().hostData();

    for(size_t i = 0; i < 4; ++i)
    {
        ASSERT_NEAR(accumData[i], noAccumData[i] + 1.0f, 1e-5f)
            << "Beta accumulation failed at " << i;
    }
}

TEST(TestGpuConvFwdRefAlphaBeta, BetaZeroSkipsRead)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yBetaZero({1, 1, 2, 2});
    Tensor<float> yDefault({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Pre-fill with garbage — should be ignored when beta=0
    yBetaZero.fillWithValue(999.0f);

    GpuFpReferenceConvolution::fprop<float>(xTensor, wTensor, yDefault, {1, 1}, {1, 1}, {0, 0});
    GpuFpReferenceConvolution::fprop<float>(
        xTensor, wTensor, yBetaZero, {1, 1}, {1, 1}, {0, 0}, 1.0, 0.0);

    const auto* defaultData = yDefault.memory().hostData();
    const auto* betaZeroData = yBetaZero.memory().hostData();

    for(size_t i = 0; i < 4; ++i)
    {
        ASSERT_NEAR(betaZeroData[i], defaultData[i], 1e-5f)
            << "Beta=0 should ignore pre-filled data at " << i;
    }
}

// ============================================================================
// TestGpuConvFwdRefStridedFp32 — non-packed (strided) tensor memory layout.
// Shape tests use packed (contiguous) tensors. These verify that the kernel
// correctly handles memory gaps between elements (e.g., inter-channel or
// inter-row stride larger than the packed stride).
// ============================================================================

TEST(TestGpuConvFwdRefStridedFp32, NonPackedInput)
{
    SKIP_IF_NO_DEVICES();

    // x: [1, 2, 4, 4] with inter-channel gap (stride[1]=32 vs packed 16)
    const std::vector<int64_t> xDims = {1, 2, 4, 4};
    const std::vector<int64_t> xStrides = {64, 32, 4, 1}; // packed would be {32, 16, 4, 1}

    Tensor<float> xTensor(xDims, xStrides);
    Tensor<float> wTensor({1, 2, 3, 3});
    Tensor<float> yCpu({1, 1, 2, 2});
    Tensor<float> yGpu({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yCpu, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(yCpu, yGpu, 1e-5f);
}

TEST(TestGpuConvFwdRefStridedFp32, NonPackedOutput)
{
    SKIP_IF_NO_DEVICES();

    // y: [1, 1, 4, 4] with inter-row gap (stride[2]=8 vs packed 4)
    const std::vector<int64_t> yDims = {1, 1, 4, 4};
    const std::vector<int64_t> yStrides = {32, 32, 8, 1}; // packed would be {16, 16, 4, 1}

    Tensor<float> xTensor({1, 1, 6, 6});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yCpu(yDims, yStrides);
    Tensor<float> yGpu(yDims, yStrides);

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yCpu, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(yCpu, yGpu, 1e-5f);
}

TEST(TestGpuConvFwdRefStridedFp32, NonPackedInputAndOutput)
{
    SKIP_IF_NO_DEVICES();

    // Both x and y have non-packed strides with inter-row gaps
    const std::vector<int64_t> xDims = {1, 2, 4, 4};
    const std::vector<int64_t> xStrides = {64, 32, 6, 1}; // packed would be {32, 16, 4, 1}

    const std::vector<int64_t> yDims = {1, 1, 2, 2};
    const std::vector<int64_t> yStrides = {8, 8, 4, 1}; // packed would be {4, 4, 2, 1}

    Tensor<float> xTensor(xDims, xStrides);
    Tensor<float> wTensor({1, 2, 3, 3});
    Tensor<float> yCpu(yDims, yStrides);
    Tensor<float> yGpu(yDims, yStrides);

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yCpu, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(yCpu, yGpu, 1e-5f);
}

TEST(TestGpuConvFwdRefStridedFp32, NonPackedWithPadding)
{
    SKIP_IF_NO_DEVICES();

    // Non-packed input with padding to exercise both features together
    const std::vector<int64_t> xDims = {1, 2, 3, 3};
    const std::vector<int64_t> xStrides = {36, 18, 3, 1}; // packed would be {18, 9, 3, 1}

    Tensor<float> xTensor(xDims, xStrides);
    Tensor<float> wTensor({1, 2, 3, 3});
    Tensor<float> yCpu({1, 1, 3, 3});
    Tensor<float> yGpu({1, 1, 3, 3});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yCpu, {1, 1}, {1, 1}, {1, 1});

    GpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {1, 1});

    assertAllClose(yCpu, yGpu, 1e-5f);
}

// ============================================================================
// TestGpuConvFwdRefInt8 — int8 input with int32 or float output.
// Shape tests only cover fp32/fp16/bfp16. These verify integer quantized
// convolution paths (int8→int32 and int8→float).
// ============================================================================

TEST(TestGpuConvFwdRefInt8, Int8ToInt32)
{
    SKIP_IF_NO_DEVICES();

    Tensor<int8_t> xTensor({1, 1, 4, 4});
    Tensor<int8_t> wTensor({1, 1, 3, 3});
    Tensor<int32_t> yGpu({1, 1, 2, 2});

    // Fill with small values that won't overflow
    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(static_cast<int8_t>(-3), static_cast<int8_t>(3), seed);
    wTensor.fillWithRandomValues(static_cast<int8_t>(-2), static_cast<int8_t>(2), seed + 1);

    GpuFpReferenceConvolution::fprop<int8_t, int8_t, int32_t, int32_t>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {0, 0});

    // Verify manually: compute expected output
    const auto* x = xTensor.memory().hostData();
    const auto* w = wTensor.memory().hostData();
    const auto* y = yGpu.memory().hostData();

    // Each output element should be the sum of element-wise products of a 3x3 patch
    for(int ho = 0; ho < 2; ++ho)
    {
        for(int wo = 0; wo < 2; ++wo)
        {
            int32_t expected = 0;
            for(int kh = 0; kh < 3; ++kh)
            {
                for(int kw = 0; kw < 3; ++kw)
                {
                    expected += static_cast<int32_t>(x[(ho + kh) * 4 + (wo + kw)])
                                * static_cast<int32_t>(w[kh * 3 + kw]);
                }
            }
            ASSERT_EQ(y[ho * 2 + wo], expected)
                << "Int8->Int32 mismatch at (" << ho << "," << wo << ")";
        }
    }
}

TEST(TestGpuConvFwdRefInt8, Int8ToFloat)
{
    SKIP_IF_NO_DEVICES();

    Tensor<int8_t> xTensor({1, 1, 4, 4});
    Tensor<int8_t> wTensor({1, 1, 3, 3});
    Tensor<float> yGpu({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(static_cast<int8_t>(-3), static_cast<int8_t>(3), seed);
    wTensor.fillWithRandomValues(static_cast<int8_t>(-2), static_cast<int8_t>(2), seed + 1);

    GpuFpReferenceConvolution::fprop<int8_t, int8_t, float, float>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {0, 0});

    const auto* x = xTensor.memory().hostData();
    const auto* w = wTensor.memory().hostData();
    const auto* y = yGpu.memory().hostData();

    for(int ho = 0; ho < 2; ++ho)
    {
        for(int wo = 0; wo < 2; ++wo)
        {
            float expected = 0.0f;
            for(int kh = 0; kh < 3; ++kh)
            {
                for(int kw = 0; kw < 3; ++kw)
                {
                    expected += static_cast<float>(x[(ho + kh) * 4 + (wo + kw)])
                                * static_cast<float>(w[kh * 3 + kw]);
                }
            }
            ASSERT_NEAR(y[ho * 2 + wo], expected, 1e-5f)
                << "Int8->Float mismatch at (" << ho << "," << wo << ")";
        }
    }
}

// ============================================================================
// TestGpuConvFwdRefTf32 — TF32 truncation mode.
// Shape tests use full-precision accumulation. This verifies that TF32
// (10-bit mantissa truncation) produces results that differ from full
// precision but remain within acceptable bounds.
// ============================================================================

TEST(TestGpuConvFwdRefTf32, DiffersFromNonTf32)
{
    SKIP_IF_NO_DEVICES();

    // Use values with enough mantissa bits to show TF32 truncation
    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yNoTf32({1, 1, 2, 2});
    Tensor<float> yTf32({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Regular computation with float accumulation
    GpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yNoTf32, {1, 1}, {1, 1}, {0, 0}, 1.0, 0.0, false);

    // TF32 computation
    GpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yTf32, {1, 1}, {1, 1}, {0, 0}, 1.0, 0.0, true);

    const auto* noTf32Data = yNoTf32.memory().hostData();
    const auto* tf32Data = yTf32.memory().hostData();

    // TF32 results should be close but not identical to full-precision
    bool hasDifference = false;
    for(size_t i = 0; i < 4; ++i)
    {
        if(std::abs(noTf32Data[i] - tf32Data[i]) > 1e-10f)
        {
            hasDifference = true;
        }
        // But should still be close
        ASSERT_NEAR(noTf32Data[i], tf32Data[i], 0.1f)
            << "TF32 too far from full precision at " << i;
    }
    ASSERT_TRUE(hasDifference) << "TF32 should produce different results from full precision";
}

// ============================================================================
// TestGpuConvFwdRefMixedType — mixed input/weight precision.
// Shape tests use the same type for input and weight tensors. These verify
// cross-type combinations (e.g., float input × half weight).
// ============================================================================

TEST(TestGpuConvFwdRefMixedType, FloatInputHalfWeight)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<half> wTensor({1, 1, 3, 3});
    Tensor<float> yCpu({1, 1, 2, 2});
    Tensor<float> yGpu({1, 1, 2, 2});

    compareGpuVsCpuConvFwd<float, half, float, double>(
        xTensor, wTensor, yCpu, yGpu, {1, 1}, {1, 1}, {0, 0}, {0, 0}, 5e-2f, 1.0f);
}

TEST(TestGpuConvFwdRefMixedType, HalfInputFloatWeight)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<half> yCpu({1, 1, 2, 2});
    Tensor<half> yGpu({1, 1, 2, 2});

    compareGpuVsCpuConvFwd<half, float, half, double>(
        xTensor, wTensor, yCpu, yGpu, {1, 1}, {1, 1}, {0, 0}, {0, 0}, 5e-2f, 1.0f);
}

// ============================================================================
// TestGpuConvFwdRefShapes — parameterized shape coverage across types
// TEST_P definitions for the fixture classes from the shared header.
// ============================================================================

// Default layout (NCL / NCHW / NCDHW)
TEST_P(TestGpuConvFwdRef1dFp32, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRef2dFp32, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRef3dFp32, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRef1dFp16, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRef2dFp16, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRef3dFp16, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRef1dBfp16, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRef2dBfp16, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRef3dBfp16, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}

// Channel-last layout (NLC / NHWC / NDHWC)
TEST_P(TestGpuConvFwdRefNlc1dFp32, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRefNhwc2dFp32, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRefNdhwc3dFp32, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRefNlc1dFp16, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRefNhwc2dFp16, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRefNdhwc3dFp16, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRefNlc1dBfp16, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRefNhwc2dBfp16, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRefNdhwc3dBfp16, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}

// ============================================================================
// Default layout (NCHW/NCDHW/NCL) instantiations — packed strides, no layout set.
// Smoke shapes run on every commit; Standard/Full shapes filtered via --gtest_filter.
// ============================================================================

// fp32
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRef2dFp32,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRef3dFp32,
                         ::testing::ValuesIn(getSmall3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRef1dFp32,
                         ::testing::ValuesIn(getSmall1dConvCases()),
                         byTag());

// fp16
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRef2dFp16,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRef1dFp16,
                         ::testing::ValuesIn(getSmall1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRef3dFp16,
                         ::testing::ValuesIn(getSmall3dConvCases()),
                         byTag());

// bfp16
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRef2dBfp16,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRef1dBfp16,
                         ::testing::ValuesIn(getSmall1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRef3dBfp16,
                         ::testing::ValuesIn(getSmall3dConvCases()),
                         byTag());

// ============================================================================
// Channel-last (NLC/NHWC/NDHWC) instantiations — small shapes.
// ============================================================================

// fp32 NLC/NHWC/NDHWC
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRefNlc1dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRefNhwc2dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRefNdhwc3dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dConvCases())),
                         byTag());

// fp16 NLC/NHWC/NDHWC
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRefNlc1dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRefNhwc2dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRefNdhwc3dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dConvCases())),
                         byTag());

// bfp16 NLC/NHWC/NDHWC
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRefNlc1dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRefNhwc2dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRefNdhwc3dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dConvCases())),
                         byTag());

// ============================================================================
// Default layout (NCL/NCHW/NCDHW) — standard/comprehensive/full shapes.
// Filter with --gtest_filter="-Standard*:Comprehensive*:Full*" for quick runs.
// ============================================================================

// fp32
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvFwdRef1dFp32,
                         ::testing::ValuesIn(getMedium1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvFwdRef2dFp32,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvFwdRef3dFp32,
                         ::testing::ValuesIn(getMedium3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvFwdRef1dFp32,
                         ::testing::ValuesIn(getLargeEdge1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvFwdRef2dFp32,
                         ::testing::ValuesIn(getLargeEdge2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvFwdRef3dFp32,
                         ::testing::ValuesIn(getLargeEdge3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRef1dFp32,
                         ::testing::ValuesIn(getLargeStress1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRef2dFp32,
                         ::testing::ValuesIn(getLargeStress2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRef3dFp32,
                         ::testing::ValuesIn(getLargeStress3dConvCases()),
                         byTag());

// fp16
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvFwdRef1dFp16,
                         ::testing::ValuesIn(getMedium1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvFwdRef2dFp16,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvFwdRef3dFp16,
                         ::testing::ValuesIn(getMedium3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvFwdRef1dFp16,
                         ::testing::ValuesIn(getLargeEdge1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvFwdRef2dFp16,
                         ::testing::ValuesIn(getLargeEdge2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvFwdRef3dFp16,
                         ::testing::ValuesIn(getLargeEdge3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRef1dFp16,
                         ::testing::ValuesIn(getLargeStress1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRef2dFp16,
                         ::testing::ValuesIn(getLargeStress2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRef3dFp16,
                         ::testing::ValuesIn(getLargeStress3dConvCases()),
                         byTag());

// bfp16
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvFwdRef1dBfp16,
                         ::testing::ValuesIn(getMedium1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvFwdRef2dBfp16,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvFwdRef3dBfp16,
                         ::testing::ValuesIn(getMedium3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvFwdRef1dBfp16,
                         ::testing::ValuesIn(getLargeEdge1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvFwdRef2dBfp16,
                         ::testing::ValuesIn(getLargeEdge2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvFwdRef3dBfp16,
                         ::testing::ValuesIn(getLargeEdge3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRef1dBfp16,
                         ::testing::ValuesIn(getLargeStress1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRef2dBfp16,
                         ::testing::ValuesIn(getLargeStress2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRef3dBfp16,
                         ::testing::ValuesIn(getLargeStress3dConvCases()),
                         byTag());

// ============================================================================
// Channel-last (NLC/NHWC/NDHWC) — standard/comprehensive/full shapes.
// ============================================================================

// fp32
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvFwdRefNlc1dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvFwdRefNhwc2dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvFwdRefNdhwc3dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvFwdRefNlc1dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvFwdRefNhwc2dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvFwdRefNdhwc3dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRefNlc1dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRefNhwc2dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRefNdhwc3dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress3dConvCases())),
                         byTag());

// fp16
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvFwdRefNlc1dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvFwdRefNhwc2dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvFwdRefNdhwc3dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvFwdRefNlc1dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvFwdRefNhwc2dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvFwdRefNdhwc3dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRefNlc1dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRefNhwc2dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRefNdhwc3dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress3dConvCases())),
                         byTag());

// bfp16
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvFwdRefNlc1dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvFwdRefNhwc2dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvFwdRefNdhwc3dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvFwdRefNlc1dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvFwdRefNhwc2dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvFwdRefNdhwc3dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRefNlc1dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRefNhwc2dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRefNdhwc3dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress3dConvCases())),
                         byTag());
