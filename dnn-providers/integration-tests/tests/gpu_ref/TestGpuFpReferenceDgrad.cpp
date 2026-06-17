// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GpuConvBwdRefTestFixture.hpp"

#include "ConvShapeCatalog.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gpu_conv_bwd_ref_test;
using namespace gpu_conv_ref_test;

// One-liner subclasses — each creates a distinct GTest-visible type so that
// INSTANTIATE_TEST_SUITE_P can use clean tier-only prefixes (Smoke, Standard, Comprehensive, Full)
// while the suite name itself carries dimensionality and layout information.

// Default layout (NCL / NCHW / NCDHW)
class TestGpuConvBwdRef1dFp32 : public ConvBwdShapeSuite<float>
{
};
class TestGpuConvBwdRef2dFp32 : public ConvBwdShapeSuite<float>
{
};
class TestGpuConvBwdRef3dFp32 : public ConvBwdShapeSuite<float>
{
};
class TestGpuConvBwdRef1dFp16 : public ConvBwdShapeSuite<half>
{
};
class TestGpuConvBwdRef2dFp16 : public ConvBwdShapeSuite<half>
{
};
class TestGpuConvBwdRef3dFp16 : public ConvBwdShapeSuite<half>
{
};
class TestGpuConvBwdRef1dBfp16 : public ConvBwdShapeSuite<bfloat16>
{
};
class TestGpuConvBwdRef2dBfp16 : public ConvBwdShapeSuite<bfloat16>
{
};
class TestGpuConvBwdRef3dBfp16 : public ConvBwdShapeSuite<bfloat16>
{
};

// Channel-last layout (NLC / NHWC / NDHWC)
class TestGpuConvBwdRefNlc1dFp32 : public ConvBwdShapeSuite<float>
{
};
class TestGpuConvBwdRefNhwc2dFp32 : public ConvBwdShapeSuite<float>
{
};
class TestGpuConvBwdRefNdhwc3dFp32 : public ConvBwdShapeSuite<float>
{
};
class TestGpuConvBwdRefNlc1dFp16 : public ConvBwdShapeSuite<half>
{
};
class TestGpuConvBwdRefNhwc2dFp16 : public ConvBwdShapeSuite<half>
{
};
class TestGpuConvBwdRefNdhwc3dFp16 : public ConvBwdShapeSuite<half>
{
};
class TestGpuConvBwdRefNlc1dBfp16 : public ConvBwdShapeSuite<bfloat16>
{
};
class TestGpuConvBwdRefNhwc2dBfp16 : public ConvBwdShapeSuite<bfloat16>
{
};
class TestGpuConvBwdRefNdhwc3dBfp16 : public ConvBwdShapeSuite<bfloat16>
{
};

// Alias to avoid verbose braced-init-list issues inside EXPECT_THROW macros
using Vec = std::vector<int64_t>;

// ============================================================================
// TestGpuConvBwdRefValidation — validateInput throw paths (via GpuFpReferenceConvolution::dgrad)
// ============================================================================

TEST(TestGpuConvBwdRefValidation, ThrowsOnInvalidDimCount)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dx({8, 8});
    Tensor<float> w({8, 8});
    Tensor<float> dy({8, 8});

    EXPECT_THROW(GpuFpReferenceConvolution::dgrad<float>(dx, w, dy, Vec{1}, Vec{1}, Vec{0}, Vec{0}),
                 std::invalid_argument);
}

TEST(TestGpuConvBwdRefValidation, ThrowsOnWeightDimMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dx({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::dgrad<float>(
                     dx, w, dy, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvBwdRefValidation, ThrowsOnOutputDimMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> dx({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::dgrad<float>(
                     dx, w, dy, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

// ============================================================================
// Standalone tests (TEST, not TEST_P)
// ============================================================================

// ============================================================================
// TestGpuConvBwdRefAsymPad — asymmetric (pre != post) padding tests
// ============================================================================

TEST(TestGpuConvBwdRefAsymPadFp32, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    // x=[1,1,3,3], w=[1,1,3,3], padding=(1,0)/(0,1) -> y=[1,1,2,2]
    runGpuVsCpuConvBwd<float>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 1e-5f);
}

TEST(TestGpuConvBwdRefAsymPadFp16, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvBwd<half>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 5e-2f);
}

TEST(TestGpuConvBwdRefAsymPadBfp16, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvBwd<bfloat16>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 0.1f);
}

// ============================================================================
// TestGpuConvBwdRefAlphaBeta — alpha/beta scaling tests
// ============================================================================

TEST(TestGpuConvBwdRefAlphaBeta, AlphaOnly)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> dyTensor({1, 1, 2, 2});
    Tensor<float> dxRef({1, 1, 4, 4});
    Tensor<float> dxScaled({1, 1, 4, 4});

    const unsigned int seed = 42;
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Compute with alpha=1.0
    GpuFpReferenceConvolution::dgrad<float>(dxRef, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0});

    // Compute with alpha=2.0
    GpuFpReferenceConvolution::dgrad<float>(
        dxScaled, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0}, 2.0);

    const auto* refData = dxRef.memory().hostData();
    const auto* scaledData = dxScaled.memory().hostData();
    auto numElements = dxRef.elementCount();

    for(size_t i = 0; i < numElements; ++i)
    {
        ASSERT_NEAR(scaledData[i], 2.0f * refData[i], 1e-5f) << "Alpha scaling failed at " << i;
    }
}

TEST(TestGpuConvBwdRefAlphaBeta, BetaAccumulate)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> dyTensor({1, 1, 2, 2});
    Tensor<float> dxTensor({1, 1, 4, 4});

    const unsigned int seed = 42;
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    dxTensor.fillWithValue(1.0f);

    // Pre-fill dx with 1.0, then compute with alpha=1.0, beta=1.0
    // Result should be dgrad(dy,w) + 1.0
    Tensor<float> dxNoAccum({1, 1, 4, 4});
    GpuFpReferenceConvolution::dgrad<float>(dxNoAccum, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0});
    GpuFpReferenceConvolution::dgrad<float>(
        dxTensor, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0}, 1.0, 1.0);

    const auto* noAccumData = dxNoAccum.memory().hostData();
    const auto* accumData = dxTensor.memory().hostData();
    auto numElements = dxTensor.elementCount();

    for(size_t i = 0; i < numElements; ++i)
    {
        ASSERT_NEAR(accumData[i], noAccumData[i] + 1.0f, 1e-5f)
            << "Beta accumulation failed at " << i;
    }
}

TEST(TestGpuConvBwdRefAlphaBeta, BetaZeroSkipsRead)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> dyTensor({1, 1, 2, 2});
    Tensor<float> dxBetaZero({1, 1, 4, 4});
    Tensor<float> dxDefault({1, 1, 4, 4});

    const unsigned int seed = 42;
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Pre-fill with garbage — should be ignored when beta=0
    dxBetaZero.fillWithValue(999.0f);

    GpuFpReferenceConvolution::dgrad<float>(dxDefault, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0});
    GpuFpReferenceConvolution::dgrad<float>(
        dxBetaZero, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0}, 1.0, 0.0);

    const auto* defaultData = dxDefault.memory().hostData();
    const auto* betaZeroData = dxBetaZero.memory().hostData();
    auto numElements = dxDefault.elementCount();

    for(size_t i = 0; i < numElements; ++i)
    {
        ASSERT_NEAR(betaZeroData[i], defaultData[i], 1e-5f)
            << "Beta=0 should ignore pre-filled data at " << i;
    }
}

// ============================================================================
// TestGpuConvBwdRefStridedFp32 — non-packed (strided) tensor tests
// ============================================================================

TEST(TestGpuConvBwdRefStridedFp32, NonPackedOutput)
{
    SKIP_IF_NO_DEVICES();

    // dx: [1, 2, 4, 4] with inter-channel gap (stride[1]=32 vs packed 16)
    const std::vector<int64_t> dxDims = {1, 2, 4, 4};
    const std::vector<int64_t> dxStrides = {64, 32, 4, 1};

    Tensor<float> dxCpu(dxDims, dxStrides);
    Tensor<float> dxGpu(dxDims, dxStrides);
    Tensor<float> wTensor({1, 2, 3, 3});
    Tensor<float> dyTensor({1, 1, 2, 2});

    const unsigned int seed = 42;
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::dgrad<float, float, float, double>(
        dxCpu, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::dgrad<float, float, float, double>(
        dxGpu, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(dxCpu, dxGpu, 1e-5f);
}

TEST(TestGpuConvBwdRefStridedFp32, NonPackedInput)
{
    SKIP_IF_NO_DEVICES();

    // dy: [1, 1, 4, 4] with inter-row gap (stride[2]=8 vs packed 4)
    const std::vector<int64_t> dyDims = {1, 1, 4, 4};
    const std::vector<int64_t> dyStrides = {32, 32, 8, 1};

    Tensor<float> dxCpu({1, 1, 6, 6});
    Tensor<float> dxGpu({1, 1, 6, 6});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> dyTensor(dyDims, dyStrides);

    const unsigned int seed = 42;
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::dgrad<float, float, float, double>(
        dxCpu, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::dgrad<float, float, float, double>(
        dxGpu, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(dxCpu, dxGpu, 1e-5f);
}

// ============================================================================
// TestGpuConvBwdRefMixedType — separate DX/DY type tests
// ============================================================================

TEST(TestGpuConvBwdRefMixedType, FloatDxHalfWeight)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> dxCpu({1, 1, 4, 4});
    Tensor<float> dxGpu({1, 1, 4, 4});
    Tensor<half> wTensor({1, 1, 3, 3});
    Tensor<float> dyTensor({1, 1, 2, 2});

    compareGpuVsCpuConvBwd<float, half, float, double>(
        dxCpu, dxGpu, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0}, {0, 0}, 5e-2f, 1.0f);
}

TEST(TestGpuConvBwdRefMixedType, HalfDxFloatWeight)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> dxCpu({1, 1, 4, 4});
    Tensor<half> dxGpu({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<half> dyTensor({1, 1, 2, 2});

    compareGpuVsCpuConvBwd<half, float, half, double>(
        dxCpu, dxGpu, wTensor, dyTensor, {1, 1}, {1, 1}, {0, 0}, {0, 0}, 5e-2f, 1.0f);
}

// ============================================================================
// TestGpuConvBwdRefShapes — parameterized shape coverage across types
// TEST_P definitions for the fixture classes from the shared header.
// ============================================================================

// Default layout (NCL / NCHW / NCDHW)
TEST_P(TestGpuConvBwdRef1dFp32, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}
TEST_P(TestGpuConvBwdRef2dFp32, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}
TEST_P(TestGpuConvBwdRef3dFp32, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}
TEST_P(TestGpuConvBwdRef1dFp16, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}
TEST_P(TestGpuConvBwdRef2dFp16, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}
TEST_P(TestGpuConvBwdRef3dFp16, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}
TEST_P(TestGpuConvBwdRef1dBfp16, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}
TEST_P(TestGpuConvBwdRef2dBfp16, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}
TEST_P(TestGpuConvBwdRef3dBfp16, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}

// Channel-last layout (NLC / NHWC / NDHWC)
TEST_P(TestGpuConvBwdRefNlc1dFp32, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}
TEST_P(TestGpuConvBwdRefNhwc2dFp32, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}
TEST_P(TestGpuConvBwdRefNdhwc3dFp32, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}
TEST_P(TestGpuConvBwdRefNlc1dFp16, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}
TEST_P(TestGpuConvBwdRefNhwc2dFp16, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}
TEST_P(TestGpuConvBwdRefNdhwc3dFp16, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}
TEST_P(TestGpuConvBwdRefNlc1dBfp16, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}
TEST_P(TestGpuConvBwdRefNhwc2dBfp16, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}
TEST_P(TestGpuConvBwdRefNdhwc3dBfp16, MatchesCpuRef)
{
    this->runConvBwdShapeTest();
}

// ============================================================================
// Default layout (NCHW/NCDHW/NCL) instantiations.
// Smoke shapes run on every commit; Standard/Full shapes filtered via --gtest_filter.
// ============================================================================

// fp32
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvBwdRef2dFp32,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvBwdRef3dFp32,
                         ::testing::ValuesIn(getSmall3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvBwdRef1dFp32,
                         ::testing::ValuesIn(getSmall1dConvCases()),
                         byTag());

// fp16
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvBwdRef2dFp16,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvBwdRef1dFp16,
                         ::testing::ValuesIn(getSmall1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvBwdRef3dFp16,
                         ::testing::ValuesIn(getSmall3dConvCases()),
                         byTag());

// bfp16
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvBwdRef2dBfp16,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvBwdRef1dBfp16,
                         ::testing::ValuesIn(getSmall1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvBwdRef3dBfp16,
                         ::testing::ValuesIn(getSmall3dConvCases()),
                         byTag());

// ============================================================================
// Channel-last (NLC/NHWC/NDHWC) instantiations — small shapes.
// ============================================================================

// fp32 NLC/NHWC/NDHWC
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvBwdRefNlc1dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvBwdRefNhwc2dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvBwdRefNdhwc3dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dConvCases())),
                         byTag());

// fp16 NLC/NHWC/NDHWC
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvBwdRefNlc1dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvBwdRefNhwc2dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvBwdRefNdhwc3dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dConvCases())),
                         byTag());

// bfp16 NLC/NHWC/NDHWC
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvBwdRefNlc1dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvBwdRefNhwc2dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvBwdRefNdhwc3dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dConvCases())),
                         byTag());

// ============================================================================
// Default layout (NCL/NCHW/NCDHW) — standard/comprehensive/full shapes.
// Filter with --gtest_filter="-Standard*:Comprehensive*:Full*" for quick runs.
// ============================================================================

// fp32
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvBwdRef1dFp32,
                         ::testing::ValuesIn(getMedium1dDgradCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvBwdRef2dFp32,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvBwdRef3dFp32,
                         ::testing::ValuesIn(getMedium3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvBwdRef1dFp32,
                         ::testing::ValuesIn(getLargeEdge1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvBwdRef2dFp32,
                         ::testing::ValuesIn(getLargeEdge2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvBwdRef3dFp32,
                         ::testing::ValuesIn(getLargeEdge3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvBwdRef1dFp32,
                         ::testing::ValuesIn(getLargeStress1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvBwdRef2dFp32,
                         ::testing::ValuesIn(getLargeStress2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvBwdRef3dFp32,
                         ::testing::ValuesIn(getLargeStress3dConvCases()),
                         byTag());

// fp16
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvBwdRef1dFp16,
                         ::testing::ValuesIn(getMedium1dDgradCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvBwdRef2dFp16,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvBwdRef3dFp16,
                         ::testing::ValuesIn(getMedium3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvBwdRef1dFp16,
                         ::testing::ValuesIn(getLargeEdge1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvBwdRef2dFp16,
                         ::testing::ValuesIn(getLargeEdge2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvBwdRef3dFp16,
                         ::testing::ValuesIn(getLargeEdge3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvBwdRef1dFp16,
                         ::testing::ValuesIn(getLargeStress1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvBwdRef2dFp16,
                         ::testing::ValuesIn(getLargeStress2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvBwdRef3dFp16,
                         ::testing::ValuesIn(getLargeStress3dConvCases()),
                         byTag());

// bfp16
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvBwdRef1dBfp16,
                         ::testing::ValuesIn(getMedium1dDgradCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvBwdRef2dBfp16,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvBwdRef3dBfp16,
                         ::testing::ValuesIn(getMedium3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvBwdRef1dBfp16,
                         ::testing::ValuesIn(getLargeEdge1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvBwdRef2dBfp16,
                         ::testing::ValuesIn(getLargeEdge2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvBwdRef3dBfp16,
                         ::testing::ValuesIn(getLargeEdge3dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvBwdRef1dBfp16,
                         ::testing::ValuesIn(getLargeStress1dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvBwdRef2dBfp16,
                         ::testing::ValuesIn(getLargeStress2dConvCases()),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvBwdRef3dBfp16,
                         ::testing::ValuesIn(getLargeStress3dConvCases()),
                         byTag());

// ============================================================================
// Channel-last (NLC/NHWC/NDHWC) — standard/comprehensive/full shapes.
// ============================================================================

// fp32
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvBwdRefNlc1dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dDgradCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvBwdRefNhwc2dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvBwdRefNdhwc3dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvBwdRefNlc1dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvBwdRefNhwc2dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvBwdRefNdhwc3dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvBwdRefNlc1dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvBwdRefNhwc2dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvBwdRefNdhwc3dFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress3dConvCases())),
                         byTag());

// fp16
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvBwdRefNlc1dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dDgradCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvBwdRefNhwc2dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvBwdRefNdhwc3dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvBwdRefNlc1dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvBwdRefNhwc2dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvBwdRefNdhwc3dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvBwdRefNlc1dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvBwdRefNhwc2dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvBwdRefNdhwc3dFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress3dConvCases())),
                         byTag());

// bfp16
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvBwdRefNlc1dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dDgradCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvBwdRefNhwc2dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuConvBwdRefNdhwc3dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvBwdRefNlc1dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvBwdRefNhwc2dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         TestGpuConvBwdRefNdhwc3dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeEdge3dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvBwdRefNlc1dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress1dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvBwdRefNhwc2dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress2dConvCases())),
                         byTag());
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvBwdRefNdhwc3dBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getLargeStress3dConvCases())),
                         byTag());
