// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// ============================================================================
// GPU-vs-CPU reference correctness gate for the SDPA forward GPU reference.
//
// For every dtype, the SAME randomized Q/K/V (and mask, where used) are fed to
// both CpuFpReferenceSdpa::forward and GpuFpReferenceSdpa::fprop, and the
// outputs are compared with assertAllClose. assertAllClose
// (CpuFpReferenceValidation::allClose) treats NaN/Inf as a hard failure, so a
// fully-masked row that produces NaN instead of 0 will fail the comparison.
// The fully-masked-row case additionally asserts the output is exactly 0 and
// finite, to document and lock that behavior independently of the oracle.
// ============================================================================

#include "ConvShapeCase.hpp" // for gpu_conv_ref_test::assertAllClose

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>

#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn-gpu-ref/GpuFpReferenceCommon.hpp>
#include <hipdnn-gpu-ref/GpuFpReferenceSdpa.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_gpu_ref;
using namespace hipdnn_gpu_ref::common::gpu_fp_reference_tensor;

using gpu_conv_ref_test::assertAllClose;

using HalfType = hipdnn_data_sdk::types::half;
using BFloat16Type = hipdnn_data_sdk::types::bfloat16;

namespace
{

// Deterministic per-tensor seeds so every run uses identical inputs.
constexpr unsigned int SEED_Q = 42;
constexpr unsigned int SEED_K = 43;
constexpr unsigned int SEED_V = 44;
constexpr unsigned int SEED_MASK = 45;

// Core helper: seed Q/K/V (and optional mask), run CPU and GPU SDPA forward on
// the SAME inputs, and assert the outputs match within tolerance.
//
// Template params allow distinct Q/K/V/O element types; ComputeDataType is the
// float compute type shared by both reference implementations. MaskT is the dtype of the
// additive mask, independent of ComputeDataType; the mask parameter is non-deduced, so a
// non-float mask must name MaskT explicitly.
template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename ODataType,
          typename ComputeDataType = float,
          typename MaskT = float>
void compareGpuVsCpuSdpaFwd(Tensor<QDataType>& q,
                            Tensor<KDataType>& k,
                            Tensor<VDataType>& v,
                            Tensor<ODataType>& oCpu,
                            Tensor<ODataType>& oGpu,
                            float tolerance,
                            std::optional<float> attnScaleValue = std::nullopt,
                            Tensor<typename NonDeduced<MaskT>::type>* attnMask = nullptr,
                            int64_t leftBound = -1,
                            int64_t rightBound = -1,
                            bool topLeftAlignment = true)
{
    fillWithRandomValues(q, static_cast<QDataType>(-1.0f), static_cast<QDataType>(1.0f), SEED_Q);
    fillWithRandomValues(k, static_cast<KDataType>(-1.0f), static_cast<KDataType>(1.0f), SEED_K);
    fillWithRandomValues(v, static_cast<VDataType>(-1.0f), static_cast<VDataType>(1.0f), SEED_V);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    q.memory().hostData();
    k.memory().hostData();
    v.memory().hostData();

    CpuFpReferenceSdpa::forward<QDataType, KDataType, VDataType, ODataType, ComputeDataType, MaskT>(
        q,
        k,
        v,
        oCpu,
        SdpaFwdOptions<MaskT>{}
            .setScale(attnScaleValue)
            .setMask(static_cast<const TensorBase<MaskT>*>(attnMask))
            .setBand(leftBound, rightBound, topLeftAlignment));

    GpuFpReferenceSdpa::fprop<QDataType, KDataType, VDataType, ODataType, ComputeDataType, MaskT>(
        q,
        k,
        v,
        oGpu,
        GpuSdpaFwdOptions<MaskT>{}
            .setScale(attnScaleValue)
            .setMask(static_cast<TensorBase<MaskT>*>(attnMask))
            .setBand(leftBound, rightBound, topLeftAlignment));

    assertAllClose(oCpu, oGpu, tolerance);
}

// dtype -> GPU-reference forward tolerance. Slightly looser than the shared CPU bound
// (sdpa::getToleranceFwd<T>) for float: the GPU reference enables FMA contraction so its
// matmuls round like the provider asm kernel, adding FMA-vs-separate-multiply/add noise
// on top of host-libm-vs-device-math differences. Measured worst-case GPU-vs-CPU
// divergence with FMA enabled is ~5e-7 (a few ULP) across this suite; the 2e-5 float
// bound keeps ~40x margin for cross-architecture device-math variation. bf16/half carry
// ample slack at 1e-2 and are unchanged from the shared bound.
template <typename T>
float gpuRefFwdTolerance()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return 2e-5f;
    }
    else if constexpr(std::is_same_v<T, HalfType> || std::is_same_v<T, BFloat16Type>)
    {
        return 1e-2f;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

float storeProbabilityForProvider(float probability, SdpaSoftmaxProbabilityMode mode)
{
    switch(mode)
    {
    case SdpaSoftmaxProbabilityMode::FLOAT:
        return probability;
    case SdpaSoftmaxProbabilityMode::BFLOAT16_RTNE:
        return static_cast<float>(BFloat16Type(probability));
    case SdpaSoftmaxProbabilityMode::BFLOAT16_RTZ:
    {
        // Non-negative probability: clearing the low 16 mantissa bits is round-toward-zero.
        const uint32_t bits = __builtin_bit_cast(uint32_t, probability) & 0xFFFF0000U;
        return __builtin_bit_cast(float, bits);
    }
    default:
        throw std::invalid_argument("Unsupported SDPA probability mode");
    }
}

float expectedSingleQueryProviderOutput(const Tensor<BFloat16Type>& q,
                                        const Tensor<BFloat16Type>& k,
                                        const Tensor<BFloat16Type>& v,
                                        SdpaSoftmaxProbabilityMode probabilityMode)
{
    const auto* qData = q.memory().hostData();
    const auto* kData = k.memory().hostData();
    const auto* vData = v.memory().hostData();
    const int64_t seqKv = k.dims()[2];

    std::vector<float> scores(static_cast<size_t>(seqKv));
    float maxVal = -std::numeric_limits<float>::infinity();
    for(int64_t skv = 0; skv < seqKv; ++skv)
    {
        scores[static_cast<size_t>(skv)]
            = static_cast<float>(qData[0]) * static_cast<float>(kData[skv]);
        maxVal = std::max(maxVal, scores[static_cast<size_t>(skv)]);
    }

    float sumExp = 0.0f;
    for(const float score : scores)
    {
        sumExp += std::exp(score - maxVal);
    }

    float weighted = 0.0f;
    for(int64_t skv = 0; skv < seqKv; ++skv)
    {
        const float probability = std::exp(scores[static_cast<size_t>(skv)] - maxVal) / sumExp;
        weighted += storeProbabilityForProvider(probability, probabilityMode)
                    * static_cast<float>(vData[skv]);
    }

    return weighted;
}

// Same contract as compareGpuVsCpuSdpaFwd, but also requests the optional
// log-sum-exp output from both references. The CPU oracle exposes graph-shaped
// rank-4 stats [B, H, Sq, 1]; the GPU reference kernel exposes squeezed rank-3
// LSE [B, H, Sq]. Use ONLY for configs with no fully-masked rows: a fully-masked
// row yields LSE = -inf, which assertAllClose treats as a hard failure (see
// LseFullyMaskedRowIsNegInf for that case, which checks -inf explicitly).
template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename ODataType,
          typename ComputeDataType = float>
void compareGpuVsCpuSdpaFwdWithLse(Tensor<QDataType>& q,
                                   Tensor<KDataType>& k,
                                   Tensor<VDataType>& v,
                                   Tensor<ODataType>& oCpu,
                                   Tensor<ODataType>& oGpu,
                                   Tensor<float>& lseCpu,
                                   Tensor<float>& lseGpu,
                                   float tolerance,
                                   std::optional<float> attnScaleValue = std::nullopt,
                                   int64_t leftBound = -1,
                                   int64_t rightBound = -1,
                                   bool topLeftAlignment = true)
{
    fillWithRandomValues(q, static_cast<QDataType>(-1.0f), static_cast<QDataType>(1.0f), SEED_Q);
    fillWithRandomValues(k, static_cast<KDataType>(-1.0f), static_cast<KDataType>(1.0f), SEED_K);
    fillWithRandomValues(v, static_cast<VDataType>(-1.0f), static_cast<VDataType>(1.0f), SEED_V);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    q.memory().hostData();
    k.memory().hostData();
    v.memory().hostData();

    CpuFpReferenceSdpa::forward<QDataType, KDataType, VDataType, ODataType, ComputeDataType>(
        q,
        k,
        v,
        oCpu,
        SdpaFwdOptions<ComputeDataType>{}
            .setScale(attnScaleValue)
            .setBand(leftBound, rightBound, topLeftAlignment)
            .setLse(&lseCpu));

    GpuFpReferenceSdpa::fprop<QDataType, KDataType, VDataType, ODataType, ComputeDataType>(
        q,
        k,
        v,
        oGpu,
        GpuSdpaFwdOptions<ComputeDataType>{}
            .setScale(attnScaleValue)
            .setBand(leftBound, rightBound, topLeftAlignment)
            .setLse(&lseGpu));

    assertAllClose(oCpu, oGpu, tolerance);
    // LSE is always FP32 regardless of the output dtype, so it uses the float
    // tolerance rather than the (possibly bf16/half) output tolerance. Compare
    // elementwise because CPU stats are rank-4 and GPU LSE is rank-3.
    TensorView<float> lseCpuView(lseCpu);
    TensorView<float> lseGpuView(lseGpu);
    for(int64_t b = 0; b < q.dims()[0]; ++b)
    {
        for(int64_t h = 0; h < q.dims()[1]; ++h)
        {
            for(int64_t sq = 0; sq < q.dims()[2]; ++sq)
            {
                EXPECT_NEAR(lseGpuView.getHostValue(std::vector<int64_t>{b, h, sq}),
                            lseCpuView.getHostValue(std::vector<int64_t>{b, h, sq, 0}),
                            gpuRefFwdTolerance<float>())
                    << "GPU/CPU LSE mismatch at b=" << b << ", h=" << h << ", sq=" << sq;
            }
        }
    }
}

} // namespace

// ============================================================================
// Plain attention (no mask, no window) — MHA across all dtypes.
// ============================================================================

template <typename T>
class TestGpuSdpaFwdPlain : public ::testing::Test
{
};

using SdpaFwdTypes = ::testing::Types<float, HalfType, BFloat16Type>;
TYPED_TEST_SUITE(TestGpuSdpaFwdPlain, SdpaFwdTypes, );

TYPED_TEST(TestGpuSdpaFwdPlain, BasicMha)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    // [B=2, H=4, Sq=8, Skv=8, D=16, Dv=16]
    Tensor<T> q({2, 4, 8, 16});
    Tensor<T> k({2, 4, 8, 16});
    Tensor<T> v({2, 4, 8, 16});
    Tensor<T> oCpu({2, 4, 8, 16});
    Tensor<T> oGpu({2, 4, 8, 16});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q, k, v, oCpu, oGpu, gpuRefFwdTolerance<T>());
}

TEST(TestGpuSdpaFwdProviderContract, Bfloat16RtneRoundsProbabilityBeforePv)
{
    SKIP_IF_NO_DEVICES();

    Tensor<BFloat16Type> q({1, 1, 1, 1});
    Tensor<BFloat16Type> k({1, 1, 4, 1});
    Tensor<BFloat16Type> v({1, 1, 4, 1});
    Tensor<float> oGpu({1, 1, 1, 1});

    q.memory().hostData()[0] = BFloat16Type(1.0f);
    k.memory().hostData()[0] = BFloat16Type(-3.0f);
    k.memory().hostData()[1] = BFloat16Type(-1.0f);
    k.memory().hostData()[2] = BFloat16Type(0.0f);
    k.memory().hostData()[3] = BFloat16Type(2.0f);
    v.memory().hostData()[0] = BFloat16Type(1000.0f);
    v.memory().hostData()[1] = BFloat16Type(-1000.0f);
    v.memory().hostData()[2] = BFloat16Type(500.0f);
    v.memory().hostData()[3] = BFloat16Type(-500.0f);

    const float fp32ProbabilityExpected
        = expectedSingleQueryProviderOutput(q, k, v, SdpaSoftmaxProbabilityMode::FLOAT);
    const float providerExpected
        = expectedSingleQueryProviderOutput(q, k, v, SdpaSoftmaxProbabilityMode::BFLOAT16_RTNE);
    ASSERT_GT(std::abs(providerExpected - fp32ProbabilityExpected), 0.1f);

    GpuFpReferenceSdpa::fprop<BFloat16Type, BFloat16Type, BFloat16Type, float, float>(
        q,
        k,
        v,
        oGpu,
        GpuSdpaFwdOptions<float>{}.setScale(1.0f).setProbabilityMode(
            SdpaSoftmaxProbabilityMode::BFLOAT16_RTNE));

    const float actual = oGpu.memory().hostData()[0];
    EXPECT_NEAR(actual, providerExpected, 1e-2f);
    EXPECT_GT(std::abs(actual - fp32ProbabilityExpected), 0.1f);
}

TEST(TestGpuSdpaFwdProviderContract, Bfloat16RtzModeTruncatesProbabilityBeforePv)
{
    SKIP_IF_NO_DEVICES();

    Tensor<BFloat16Type> q({1, 1, 1, 1});
    Tensor<BFloat16Type> k({1, 1, 4, 1});
    Tensor<BFloat16Type> v({1, 1, 4, 1});
    Tensor<float> oGpu({1, 1, 1, 1});

    q.memory().hostData()[0] = BFloat16Type(1.0f);
    k.memory().hostData()[0] = BFloat16Type(-3.0f);
    k.memory().hostData()[1] = BFloat16Type(-1.0f);
    k.memory().hostData()[2] = BFloat16Type(0.0f);
    k.memory().hostData()[3] = BFloat16Type(2.0f);
    v.memory().hostData()[0] = BFloat16Type(1000.0f);
    v.memory().hostData()[1] = BFloat16Type(-1000.0f);
    v.memory().hostData()[2] = BFloat16Type(500.0f);
    v.memory().hostData()[3] = BFloat16Type(-500.0f);

    const float rtneExpected
        = expectedSingleQueryProviderOutput(q, k, v, SdpaSoftmaxProbabilityMode::BFLOAT16_RTNE);
    const float rtzExpected
        = expectedSingleQueryProviderOutput(q, k, v, SdpaSoftmaxProbabilityMode::BFLOAT16_RTZ);
    ASSERT_GT(std::abs(rtneExpected - rtzExpected), 1.0f);

    GpuFpReferenceSdpa::fprop<BFloat16Type, BFloat16Type, BFloat16Type, float, float>(
        q,
        k,
        v,
        oGpu,
        GpuSdpaFwdOptions<float>{}.setScale(1.0f).setProbabilityMode(
            SdpaSoftmaxProbabilityMode::BFLOAT16_RTZ));

    const float actual = oGpu.memory().hostData()[0];
    EXPECT_NEAR(actual, rtzExpected, 1e-2f);
    EXPECT_GT(std::abs(actual - rtneExpected), 1.0f);
}

// ============================================================================
// Causal top-left (leftBound=-1, rightBound=0, topLeftAlignment=true).
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, CausalTopLeft)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    // Square Sq=Skv exercises the standard lower-triangular causal mask.
    Tensor<T> q({1, 2, 8, 16});
    Tensor<T> k({1, 2, 8, 16});
    Tensor<T> v({1, 2, 8, 16});
    Tensor<T> oCpu({1, 2, 8, 16});
    Tensor<T> oGpu({1, 2, 8, 16});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q,
                                       k,
                                       v,
                                       oCpu,
                                       oGpu,
                                       gpuRefFwdTolerance<T>(),
                                       std::nullopt,
                                       /*attnMask=*/nullptr,
                                       /*leftBound=*/-1,
                                       /*rightBound=*/0,
                                       /*topLeftAlignment=*/true);
}

// ============================================================================
// Skv != Sq mask coverage — the cartesian product requested in review:
//   topLeftAlignment  x  (Skv smaller / larger than Sq)  x  (causal / sliding).
// Causal is leftBound=-1, rightBound=0; the sliding window uses leftBound=2,
// rightBound=1. All eight cells use Skv != Sq (Sq=8 with Skv=4 or Skv=16).
//
// Uses the output-only comparator: several cells (bottom-right + smaller Skv,
// tight sliding windows) intentionally fully mask some query rows, which both
// references collapse to a zero output row. The LSE comparator would reject
// those rows (LSE = -inf), so it is deliberately not used here.
// ============================================================================

namespace
{

struct SdpaMaskParams
{
    bool topLeftAlignment;
    bool skvSmaller; // Skv < Sq when true, Skv > Sq when false.
    int64_t leftBound;
    int64_t rightBound;
    const char* name;
};

template <typename T>
void runMaskCartesianCase(const SdpaMaskParams& params)
{
    const int64_t batch = 1;
    const int64_t numHeads = 2;
    const int64_t headDim = 16;
    const int64_t seqQ = 8;
    const int64_t seqKv = params.skvSmaller ? 4 : 16;

    Tensor<T> q({batch, numHeads, seqQ, headDim});
    Tensor<T> k({batch, numHeads, seqKv, headDim});
    Tensor<T> v({batch, numHeads, seqKv, headDim});
    Tensor<T> oCpu({batch, numHeads, seqQ, headDim});
    Tensor<T> oGpu({batch, numHeads, seqQ, headDim});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q,
                                       k,
                                       v,
                                       oCpu,
                                       oGpu,
                                       gpuRefFwdTolerance<T>(),
                                       std::nullopt,
                                       /*attnMask=*/nullptr,
                                       params.leftBound,
                                       params.rightBound,
                                       params.topLeftAlignment);
}

const std::vector<SdpaMaskParams>& maskCartesianCases()
{
    static const std::vector<SdpaMaskParams> s_cases = {
        {/*topLeft=*/true, /*skvSmaller=*/true, -1, 0, "CausalTopLeftSmallerSkv"},
        {/*topLeft=*/true, /*skvSmaller=*/false, -1, 0, "CausalTopLeftLargerSkv"},
        {/*topLeft=*/false, /*skvSmaller=*/true, -1, 0, "CausalBottomRightSmallerSkv"},
        {/*topLeft=*/false, /*skvSmaller=*/false, -1, 0, "CausalBottomRightLargerSkv"},
        {/*topLeft=*/true, /*skvSmaller=*/true, 2, 1, "SlidingTopLeftSmallerSkv"},
        {/*topLeft=*/true, /*skvSmaller=*/false, 2, 1, "SlidingTopLeftLargerSkv"},
        {/*topLeft=*/false, /*skvSmaller=*/true, 2, 1, "SlidingBottomRightSmallerSkv"},
        {/*topLeft=*/false, /*skvSmaller=*/false, 2, 1, "SlidingBottomRightLargerSkv"},
    };
    return s_cases;
}

std::string maskCartesianName(const ::testing::TestParamInfo<SdpaMaskParams>& info)
{
    return info.param.name;
}

} // namespace

template <typename T>
class TestGpuSdpaFwdMaskCartesianBase : public ::testing::TestWithParam<SdpaMaskParams>
{
};

// Explicit per-dtype aliases with their own TEST_P/INSTANTIATE (project style:
// prefer aliases over TYPED_TEST_P macros for readable gtest names + failures).
using TestGpuSdpaFwdMaskCartesianFp32 = TestGpuSdpaFwdMaskCartesianBase<float>;
using TestGpuSdpaFwdMaskCartesianFp16 = TestGpuSdpaFwdMaskCartesianBase<HalfType>;
using TestGpuSdpaFwdMaskCartesianBfp16 = TestGpuSdpaFwdMaskCartesianBase<BFloat16Type>;

TEST_P(TestGpuSdpaFwdMaskCartesianFp32, SkvNotEqualSq)
{
    SKIP_IF_NO_DEVICES();
    runMaskCartesianCase<float>(GetParam());
}

TEST_P(TestGpuSdpaFwdMaskCartesianFp16, SkvNotEqualSq)
{
    SKIP_IF_NO_DEVICES();
    runMaskCartesianCase<HalfType>(GetParam());
}

TEST_P(TestGpuSdpaFwdMaskCartesianBfp16, SkvNotEqualSq)
{
    SKIP_IF_NO_DEVICES();
    runMaskCartesianCase<BFloat16Type>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(MaskCombos,
                         TestGpuSdpaFwdMaskCartesianFp32,
                         ::testing::ValuesIn(maskCartesianCases()),
                         maskCartesianName);
INSTANTIATE_TEST_SUITE_P(MaskCombos,
                         TestGpuSdpaFwdMaskCartesianFp16,
                         ::testing::ValuesIn(maskCartesianCases()),
                         maskCartesianName);
INSTANTIATE_TEST_SUITE_P(MaskCombos,
                         TestGpuSdpaFwdMaskCartesianBfp16,
                         ::testing::ValuesIn(maskCartesianCases()),
                         maskCartesianName);

// ============================================================================
// Generic sliding window with BOTH leftBound>=0 and rightBound>=0.
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, SlidingWindowBothBounds)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({1, 2, 8, 16});
    Tensor<T> k({1, 2, 8, 16});
    Tensor<T> v({1, 2, 8, 16});
    Tensor<T> oCpu({1, 2, 8, 16});
    Tensor<T> oGpu({1, 2, 8, 16});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q,
                                       k,
                                       v,
                                       oCpu,
                                       oGpu,
                                       gpuRefFwdTolerance<T>(),
                                       std::nullopt,
                                       /*attnMask=*/nullptr,
                                       /*leftBound=*/2,
                                       /*rightBound=*/1,
                                       /*topLeftAlignment=*/true);
}

// ============================================================================
// Additive mask — full rank-4 [B, H, Sq, Skv].
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, AdditiveMaskRank4)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({1, 2, 8, 16});
    Tensor<T> k({1, 2, 8, 16});
    Tensor<T> v({1, 2, 8, 16});
    Tensor<T> oCpu({1, 2, 8, 16});
    Tensor<T> oGpu({1, 2, 8, 16});
    Tensor<float> mask({1, 2, 8, 8});

    fillWithRandomValues(mask, -2.0f, 2.0f, SEED_MASK);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    mask.memory().hostData();

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q,
                                       k,
                                       v,
                                       oCpu,
                                       oGpu,
                                       gpuRefFwdTolerance<T>(),
                                       std::nullopt,
                                       /*attnMask=*/&mask);
}

// ============================================================================
// Additive mask — BROADCAST rank-2 [Sq, Skv]. Right-aligned to [B, H, Sq, Skv],
// broadcasting over batch and head.
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, AdditiveMaskBroadcastRank2)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({2, 4, 8, 16});
    Tensor<T> k({2, 4, 8, 16});
    Tensor<T> v({2, 4, 8, 16});
    Tensor<T> oCpu({2, 4, 8, 16});
    Tensor<T> oGpu({2, 4, 8, 16});
    Tensor<float> mask({8, 8}); // [Sq, Skv]

    fillWithRandomValues(mask, -2.0f, 2.0f, SEED_MASK);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    mask.memory().hostData();

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q,
                                       k,
                                       v,
                                       oCpu,
                                       oGpu,
                                       gpuRefFwdTolerance<T>(),
                                       std::nullopt,
                                       /*attnMask=*/&mask);
}

// ============================================================================
// Additive mask — BROADCAST rank-3 [H, Sq, Skv]. Broadcasts over batch only.
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, AdditiveMaskBroadcastRank3)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({2, 2, 8, 16});
    Tensor<T> k({2, 2, 8, 16});
    Tensor<T> v({2, 2, 8, 16});
    Tensor<T> oCpu({2, 2, 8, 16});
    Tensor<T> oGpu({2, 2, 8, 16});
    Tensor<float> mask({2, 8, 8}); // [H, Sq, Skv]

    fillWithRandomValues(mask, -2.0f, 2.0f, SEED_MASK);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    mask.memory().hostData();

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q,
                                       k,
                                       v,
                                       oCpu,
                                       oGpu,
                                       gpuRefFwdTolerance<T>(),
                                       std::nullopt,
                                       /*attnMask=*/&mask);
}

// ============================================================================
// Mask dtype sweep: the additive mask is read with its own dtype (float, half or
// bfloat16), independent of Q/K/V/O. The mask is non-uniform along Skv with values
// exactly representable in all three dtypes; a uniform mask would pass by coincidence
// because softmax is shift-invariant.
// ============================================================================

namespace
{

template <typename MaskT>
void fillExactlyRepresentableMask(Tensor<MaskT>& mask)
{
    constexpr std::array<float, 8> VALUES = {-2.0f, -1.0f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 1.0f};
    auto* data = mask.memory().hostData();
    for(size_t i = 0; i < mask.elementCount(); ++i)
    {
        data[i] = static_cast<MaskT>(VALUES[(i * 3 + i / 8) % VALUES.size()]);
    }
}

} // namespace

template <typename T>
class TestGpuSdpaFwdMaskDtype : public ::testing::Test
{
};

TYPED_TEST_SUITE(TestGpuSdpaFwdMaskDtype, SdpaFwdTypes, );

TYPED_TEST(TestGpuSdpaFwdMaskDtype, AdditiveMaskRank4)
{
    SKIP_IF_NO_DEVICES();
    using MaskT = TypeParam;

    Tensor<float> q({1, 2, 8, 16});
    Tensor<float> k({1, 2, 8, 16});
    Tensor<float> v({1, 2, 8, 16});
    Tensor<float> oCpu({1, 2, 8, 16});
    Tensor<float> oGpu({1, 2, 8, 16});
    Tensor<MaskT> mask({1, 2, 8, 8});
    fillExactlyRepresentableMask(mask);

    compareGpuVsCpuSdpaFwd<float, float, float, float, float, MaskT>(
        q, k, v, oCpu, oGpu, gpuRefFwdTolerance<float>(), std::nullopt, /*attnMask=*/&mask);
}

TYPED_TEST(TestGpuSdpaFwdMaskDtype, AdditiveMaskBroadcastRank2)
{
    SKIP_IF_NO_DEVICES();
    using MaskT = TypeParam;

    Tensor<float> q({2, 4, 8, 16});
    Tensor<float> k({2, 4, 8, 16});
    Tensor<float> v({2, 4, 8, 16});
    Tensor<float> oCpu({2, 4, 8, 16});
    Tensor<float> oGpu({2, 4, 8, 16});
    Tensor<MaskT> mask({8, 8}); // [Sq, Skv]
    fillExactlyRepresentableMask(mask);

    compareGpuVsCpuSdpaFwd<float, float, float, float, float, MaskT>(
        q, k, v, oCpu, oGpu, gpuRefFwdTolerance<float>(), std::nullopt, /*attnMask=*/&mask);
}

// The mask dtype does not follow the Q/K/V/O dtype.
TYPED_TEST(TestGpuSdpaFwdMaskDtype, Bfloat16InputsAdditiveMask)
{
    SKIP_IF_NO_DEVICES();
    using MaskT = TypeParam;

    Tensor<BFloat16Type> q({1, 2, 8, 16});
    Tensor<BFloat16Type> k({1, 2, 8, 16});
    Tensor<BFloat16Type> v({1, 2, 8, 16});
    Tensor<BFloat16Type> oCpu({1, 2, 8, 16});
    Tensor<BFloat16Type> oGpu({1, 2, 8, 16});
    Tensor<MaskT> mask({1, 2, 8, 8});
    fillExactlyRepresentableMask(mask);

    compareGpuVsCpuSdpaFwd<BFloat16Type, BFloat16Type, BFloat16Type, BFloat16Type, float, MaskT>(
        q,
        k,
        v,
        oCpu,
        oGpu,
        gpuRefFwdTolerance<BFloat16Type>(),
        std::nullopt,
        /*attnMask=*/&mask);
}

// Same kernel, same inputs, mask values identical after conversion: the output must not
// depend on the mask storage dtype at all.
TYPED_TEST(TestGpuSdpaFwdMaskDtype, MatchesFloatMaskBitwise)
{
    SKIP_IF_NO_DEVICES();
    using MaskT = TypeParam;

    Tensor<float> q({1, 2, 8, 16});
    Tensor<float> k({1, 2, 8, 16});
    Tensor<float> v({1, 2, 8, 16});
    fillWithRandomValues(q, -1.0f, 1.0f, SEED_Q);
    fillWithRandomValues(k, -1.0f, 1.0f, SEED_K);
    fillWithRandomValues(v, -1.0f, 1.0f, SEED_V);

    Tensor<MaskT> mask({1, 2, 8, 8});
    fillExactlyRepresentableMask(mask);
    Tensor<float> floatMask({1, 2, 8, 8});
    for(size_t i = 0; i < mask.elementCount(); ++i)
    {
        floatMask.memory().hostData()[i] = static_cast<float>(mask.memory().hostData()[i]);
    }

    Tensor<float> oMask({1, 2, 8, 16});
    Tensor<float> oFloatMask({1, 2, 8, 16});
    GpuFpReferenceSdpa::fprop<float, float, float, float, float, MaskT>(
        q, k, v, oMask, GpuSdpaFwdOptions<MaskT>{}.setMask(&mask));
    GpuFpReferenceSdpa::fprop<float, float, float, float, float, float>(
        q, k, v, oFloatMask, GpuSdpaFwdOptions<float>{}.setMask(&floatMask));

    const auto* actual = oMask.memory().hostData();
    const auto* expected = oFloatMask.memory().hostData();
    for(size_t i = 0; i < oMask.elementCount(); ++i)
    {
        ASSERT_EQ(actual[i], expected[i]) << "at flat index " << i;
    }
}

// ============================================================================
// GQA with numHeadsK != numHeadsV (both set and differing).
// H=8, Hk=4, Hv=2.
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, GqaDifferentKvHeads)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({1, 8, 8, 16});
    Tensor<T> k({1, 4, 8, 16});
    Tensor<T> v({1, 2, 8, 16});
    Tensor<T> oCpu({1, 8, 8, 16});
    Tensor<T> oGpu({1, 8, 8, 16});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q, k, v, oCpu, oGpu, gpuRefFwdTolerance<T>());
}

// ============================================================================
// MQA — single KV head shared by all Q heads (Hk = Hv = 1).
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, Mqa)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({1, 8, 8, 16});
    Tensor<T> k({1, 1, 8, 16});
    Tensor<T> v({1, 1, 8, 16});
    Tensor<T> oCpu({1, 8, 8, 16});
    Tensor<T> oGpu({1, 8, 8, 16});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q, k, v, oCpu, oGpu, gpuRefFwdTolerance<T>());
}

// ============================================================================
// Cross-attention — headDimV != headDim (D=16, Dv=32).
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, CrossAttentionHeadDimVDiffers)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    // Q/K share head_dim=16; V (and O) use head_dim_v=32.
    Tensor<T> q({1, 2, 8, 16});
    Tensor<T> k({1, 2, 6, 16});
    Tensor<T> v({1, 2, 6, 32});
    Tensor<T> oCpu({1, 2, 8, 32});
    Tensor<T> oGpu({1, 2, 8, 32});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q, k, v, oCpu, oGpu, gpuRefFwdTolerance<T>());
}

// ============================================================================
// Fully-masked row — a sliding-window/causal config that masks an ENTIRE query
// row. The masked row must produce 0 (not NaN). assertAllClose already fails on
// NaN; here we additionally assert the GPU output for the masked row is exactly
// 0 and finite, locking the behavior independently of the CPU oracle.
//
// Bottom-right causal (rightBound=0) with Sq>Skv: offset = Skv - Sq < 0 means
// early query rows see no kv positions and are fully masked.
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, FullyMaskedRowYieldsZeroNotNan)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    // Sq=4, Skv=2 -> offset = -2. Rows sq=0,1 are fully masked (see CPU oracle
    // BottomRightCausalMaskLargerSq).
    const int64_t batch = 1;
    const int64_t numHeads = 2;
    const int64_t seqQ = 4;
    const int64_t seqKv = 2;
    const int64_t headDim = 8;
    const int64_t headDimV = 8;

    Tensor<T> q({batch, numHeads, seqQ, headDim});
    Tensor<T> k({batch, numHeads, seqKv, headDim});
    Tensor<T> v({batch, numHeads, seqKv, headDimV});
    Tensor<T> oCpu({batch, numHeads, seqQ, headDimV});
    Tensor<T> oGpu({batch, numHeads, seqQ, headDimV});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q,
                                       k,
                                       v,
                                       oCpu,
                                       oGpu,
                                       gpuRefFwdTolerance<T>(),
                                       std::nullopt,
                                       /*attnMask=*/nullptr,
                                       /*leftBound=*/-1,
                                       /*rightBound=*/0,
                                       /*topLeftAlignment=*/false);

    // Rows sq=0 and sq=1 are fully masked: output must be exactly 0 and finite.
    for(int64_t h = 0; h < numHeads; ++h)
    {
        for(int64_t sq = 0; sq < 2; ++sq)
        {
            for(int64_t dv = 0; dv < headDimV; ++dv)
            {
                const auto gpuVal
                    = static_cast<float>(oGpu.getHostValue(std::vector<int64_t>{0, h, sq, dv}));
                EXPECT_TRUE(std::isfinite(gpuVal))
                    << "Fully-masked row produced non-finite output at h=" << h << ", sq=" << sq
                    << ", dv=" << dv << " (got " << gpuVal << ")";
                EXPECT_EQ(gpuVal, 0.0f) << "Fully-masked row must produce 0 at h=" << h
                                        << ", sq=" << sq << ", dv=" << dv;
            }
        }
    }
}

// ============================================================================
// Realistic head dim (D=Dv=128) with a moderate sequence length. The small
// cases above (D=16, Sk<=8) never exercise long float accumulations: the QK^T
// dot reduces over D and the softmax sum over Sk, so realistic lengths are what
// actually stress (a) the GPU/CPU float-accumulation match behind the fp32
// tolerance (2e-5, see gpuRefFwdTolerance) and (b) the subtract-max softmax stability (larger D ->
// larger pre-softmax magnitudes). One HipRTC compile per dtype is reused across
// all shapes, so this only costs kernel runtime, not a recompile.
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, RealisticHeadDim)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    // [B=1, H=4, Sq=128, Skv=128, D=128, Dv=128]
    Tensor<T> q({1, 4, 128, 128});
    Tensor<T> k({1, 4, 128, 128});
    Tensor<T> v({1, 4, 128, 128});
    Tensor<T> oCpu({1, 4, 128, 128});
    Tensor<T> oGpu({1, 4, 128, 128});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q, k, v, oCpu, oGpu, gpuRefFwdTolerance<T>());
}

// ============================================================================
// Long sequence (Sq=Skv=256). Probes the softmax reduction length directly: the
// running sum over 256 keys is where a GPU-vs-CPU accumulation mismatch would
// show up at the fp32 tolerance. The 2e-5 bound (see gpuRefFwdTolerance) already
// absorbs FMA contraction and host-vs-device libm noise (~5e-7 measured); a
// divergence approaching 2e-5 here is a real signal (accumulation not matched /
// tolerance must scale), not a flaky test to loosen blindly.
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, LongSequence)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    // [B=1, H=2, Sq=256, Skv=256, D=64, Dv=64]
    Tensor<T> q({1, 2, 256, 64});
    Tensor<T> k({1, 2, 256, 64});
    Tensor<T> v({1, 2, 256, 64});
    Tensor<T> oCpu({1, 2, 256, 64});
    Tensor<T> oGpu({1, 2, 256, 64});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q, k, v, oCpu, oGpu, gpuRefFwdTolerance<T>());
}

// ============================================================================
// Non-packed (strided) layout. Every case above uses packed/contiguous tensors,
// so the kernel's per-tensor stride indexing is otherwise untested. BSHD is the
// realistic attention case: logical dims stay [B, H, Sq, D] but the physical
// layout stores sequence-major (strideOrder {3,1,2,0}), so the head/seq strides
// are non-trivial. Both references are stride-aware (the kernel reads via the
// SdpaStrides POD; the CPU oracle via getHostValue), so a stride-math bug would
// surface as a mismatch here.
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, NonPackedBshdLayout)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    // Logical [B=2, H=4, Sq=8, Skv=8, D=16, Dv=16] with BSHD physical layout.
    Tensor<T> q({2, 4, 8, 16}, TensorLayout::BSHD);
    Tensor<T> k({2, 4, 8, 16}, TensorLayout::BSHD);
    Tensor<T> v({2, 4, 8, 16}, TensorLayout::BSHD);
    Tensor<T> oCpu({2, 4, 8, 16}, TensorLayout::BSHD);
    Tensor<T> oGpu({2, 4, 8, 16}, TensorLayout::BSHD);

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q, k, v, oCpu, oGpu, gpuRefFwdTolerance<T>());
}

// ============================================================================
// Larger, production-representative single-attention-layer shape: batch x heads
// = 16, a 512-token context, and a 128 head dim. Pushes the reduction lengths
// (Skv=512, D=128) and grid size further than RealisticHeadDim/LongSequence to
// confirm the fp32 tolerance and softmax stability still hold at scale and the
// launch/bounds handling is correct for a large output grid. (The reference
// kernel recomputes QK^T per output element, so this is intentionally the
// largest case kept in the fast suite.)
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, LargerShape)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    // [B=2, H=8, Sq=512, Skv=512, D=128, Dv=128]
    Tensor<T> q({2, 8, 512, 128});
    Tensor<T> k({2, 8, 512, 128});
    Tensor<T> v({2, 8, 512, 128});
    Tensor<T> oCpu({2, 8, 512, 128});
    Tensor<T> oGpu({2, 8, 512, 128});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q, k, v, oCpu, oGpu, gpuRefFwdTolerance<T>());
}

// ============================================================================
// Explicit (non-default) attention scale. Every case above leaves attnScaleValue
// unset, so both references apply no scaling; this passes an explicit value to
// exercise the provided-scale path on both sides.
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, ExplicitAttnScale)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({2, 4, 8, 16});
    Tensor<T> k({2, 4, 8, 16});
    Tensor<T> v({2, 4, 8, 16});
    Tensor<T> oCpu({2, 4, 8, 16});
    Tensor<T> oGpu({2, 4, 8, 16});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(
        q, k, v, oCpu, oGpu, gpuRefFwdTolerance<T>(), /*attnScaleValue=*/0.25f);
}

// An absent scale is 1.0 (no scaling), as in cuDNN. One query against two keys,
// D = 4: the logits are q.k = 4 and 0, so the output is 1 / (e^4 + 1). With the
// 1/sqrt(D) scale it would be 1 / (e^2 + 1), about 0.119.
TEST(TestGpuSdpaFwdDefaultScale, AbsentScaleIsOne)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> q({1, 1, 1, 4});
    Tensor<float> k({1, 1, 2, 4});
    Tensor<float> v({1, 1, 2, 1});
    Tensor<float> o({1, 1, 1, 1});
    for(int i = 0; i < 4; ++i)
    {
        q.memory().hostData()[i] = 1.0f;
        k.memory().hostData()[i] = 1.0f;
        k.memory().hostData()[4 + i] = 0.0f;
    }
    v.memory().hostData()[0] = 0.0f;
    v.memory().hostData()[1] = 1.0f;

    GpuFpReferenceSdpa::fprop<float>(q, k, v, o);

    EXPECT_NEAR(o.memory().hostData()[0], 1.0f / (std::exp(4.0f) + 1.0f), 1e-6f);
}

// ============================================================================
// Additive mask combined with a sliding window. Exercises the add-then-overwrite
// ordering: the additive mask is applied first, then window positions are forced
// to -inf. No prior case combines the two.
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, AdditiveMaskWithSlidingWindow)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({1, 2, 8, 16});
    Tensor<T> k({1, 2, 8, 16});
    Tensor<T> v({1, 2, 8, 16});
    Tensor<T> oCpu({1, 2, 8, 16});
    Tensor<T> oGpu({1, 2, 8, 16});
    Tensor<float> mask({1, 2, 8, 8});

    fillWithRandomValues(mask, -2.0f, 2.0f, SEED_MASK);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    mask.memory().hostData();

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q,
                                       k,
                                       v,
                                       oCpu,
                                       oGpu,
                                       gpuRefFwdTolerance<T>(),
                                       std::nullopt,
                                       /*attnMask=*/&mask,
                                       /*leftBound=*/2,
                                       /*rightBound=*/1,
                                       /*topLeftAlignment=*/true);
}

// ============================================================================
// Additive mask with per-dim size-1 broadcast at the LEADING positions
// (mask [1, 1, Sq, Skv] against [B, H, Sq, Skv]). The rank-N broadcast cases
// above never set an interior/leading dim to 1, so the maskDims[i]==1 -> index 0
// branch is otherwise unexercised where it actually broadcasts.
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, AdditiveMaskBroadcastBatchHead)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({2, 4, 8, 16});
    Tensor<T> k({2, 4, 8, 16});
    Tensor<T> v({2, 4, 8, 16});
    Tensor<T> oCpu({2, 4, 8, 16});
    Tensor<T> oGpu({2, 4, 8, 16});
    Tensor<float> mask({1, 1, 8, 8}); // broadcast over batch and head

    fillWithRandomValues(mask, -2.0f, 2.0f, SEED_MASK);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    mask.memory().hostData();

    compareGpuVsCpuSdpaFwd<T, T, T, T>(
        q, k, v, oCpu, oGpu, gpuRefFwdTolerance<T>(), std::nullopt, /*attnMask=*/&mask);
}

// ============================================================================
// Additive mask with a size-1 broadcast at an INTERIOR position
// (mask [B, 1, Sq, Skv] against [B, H, Sq, Skv]): broadcasts over head only,
// with batch and the sequence dims fully indexed.
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, AdditiveMaskBroadcastHeadOnly)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({2, 4, 8, 16});
    Tensor<T> k({2, 4, 8, 16});
    Tensor<T> v({2, 4, 8, 16});
    Tensor<T> oCpu({2, 4, 8, 16});
    Tensor<T> oGpu({2, 4, 8, 16});
    Tensor<float> mask({2, 1, 8, 8}); // broadcast over head only

    fillWithRandomValues(mask, -2.0f, 2.0f, SEED_MASK);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    mask.memory().hostData();

    compareGpuVsCpuSdpaFwd<T, T, T, T>(
        q, k, v, oCpu, oGpu, gpuRefFwdTolerance<T>(), std::nullopt, /*attnMask=*/&mask);
}

// ============================================================================
// Mixed precision: 16-bit Q/K/V inputs with a float (fp32) output tensor. The
// FLOAT-output plan builders are registered (HALF/BFLOAT16 inputs -> FLOAT out)
// but the typed suite above only uses uniform input/output dtypes. The Q/K/V
// inputs are bit-identical across both backends, so the GPU-vs-CPU difference is
// the float compute path only; the float tolerance applies.
// ============================================================================

TEST(TestGpuSdpaFwdMixedPrecision, HalfInputsFloatOutput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<HalfType> q({2, 4, 8, 16});
    Tensor<HalfType> k({2, 4, 8, 16});
    Tensor<HalfType> v({2, 4, 8, 16});
    Tensor<float> oCpu({2, 4, 8, 16});
    Tensor<float> oGpu({2, 4, 8, 16});

    compareGpuVsCpuSdpaFwd<HalfType, HalfType, HalfType, float>(
        q, k, v, oCpu, oGpu, gpuRefFwdTolerance<float>());
}

TEST(TestGpuSdpaFwdMixedPrecision, Bfloat16InputsFloatOutput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<BFloat16Type> q({2, 4, 8, 16});
    Tensor<BFloat16Type> k({2, 4, 8, 16});
    Tensor<BFloat16Type> v({2, 4, 8, 16});
    Tensor<float> oCpu({2, 4, 8, 16});
    Tensor<float> oGpu({2, 4, 8, 16});

    compareGpuVsCpuSdpaFwd<BFloat16Type, BFloat16Type, BFloat16Type, float>(
        q, k, v, oCpu, oGpu, gpuRefFwdTolerance<float>());
}

// ============================================================================
// LSE (log-sum-exp) output. The GPU reference exposes optional squeezed [B, H, Sq]
// LSE while CpuFpReferenceSdpa produces graph-shaped [B, H, Sq, 1] stats; these
// compare the values against the CPU oracle. Configs are chosen so no query row is
// fully masked (LSE there is -inf, covered separately below).
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, LseBasicMha)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({2, 4, 8, 16});
    Tensor<T> k({2, 4, 8, 16});
    Tensor<T> v({2, 4, 8, 16});
    Tensor<T> oCpu({2, 4, 8, 16});
    Tensor<T> oGpu({2, 4, 8, 16});
    Tensor<float> lseCpu({2, 4, 8, 1}); // CPU graph-shaped stats [B, H, Sq, 1]
    Tensor<float> lseGpu({2, 4, 8});

    compareGpuVsCpuSdpaFwdWithLse<T, T, T, T>(
        q, k, v, oCpu, oGpu, lseCpu, lseGpu, gpuRefFwdTolerance<T>());
}

// Top-left causal on a square shape: every query row sees at least its diagonal
// key, so no row is fully masked and every LSE entry is finite.
TYPED_TEST(TestGpuSdpaFwdPlain, LseCausalTopLeft)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({1, 2, 8, 16});
    Tensor<T> k({1, 2, 8, 16});
    Tensor<T> v({1, 2, 8, 16});
    Tensor<T> oCpu({1, 2, 8, 16});
    Tensor<T> oGpu({1, 2, 8, 16});
    Tensor<float> lseCpu({1, 2, 8, 1});
    Tensor<float> lseGpu({1, 2, 8});

    compareGpuVsCpuSdpaFwdWithLse<T, T, T, T>(q,
                                              k,
                                              v,
                                              oCpu,
                                              oGpu,
                                              lseCpu,
                                              lseGpu,
                                              gpuRefFwdTolerance<T>(),
                                              std::nullopt,
                                              /*leftBound=*/-1,
                                              /*rightBound=*/0,
                                              /*topLeftAlignment=*/true);
}

// Fully-masked rows: the CPU oracle writes LSE = -inf (maxVal=-inf, sumExp=0).
// assertAllClose rejects Inf, so this checks the masked rows are -inf on BOTH
// references and the surviving rows match within tolerance.
TYPED_TEST(TestGpuSdpaFwdPlain, LseFullyMaskedRowIsNegInf)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    // Sq=4, Skv=2, bottom-right causal -> offset=-2: rows sq=0,1 are fully masked.
    const int64_t batch = 1;
    const int64_t numHeads = 2;
    const int64_t seqQ = 4;
    const int64_t seqKv = 2;
    const int64_t headDim = 8;
    const int64_t headDimV = 8;

    Tensor<T> q({batch, numHeads, seqQ, headDim});
    Tensor<T> k({batch, numHeads, seqKv, headDim});
    Tensor<T> v({batch, numHeads, seqKv, headDimV});
    Tensor<T> oCpu({batch, numHeads, seqQ, headDimV});
    Tensor<T> oGpu({batch, numHeads, seqQ, headDimV});
    Tensor<float> lseCpu({batch, numHeads, seqQ, 1});
    Tensor<float> lseGpu({batch, numHeads, seqQ});

    fillWithRandomValues(q, static_cast<T>(-1.0f), static_cast<T>(1.0f), SEED_Q);
    fillWithRandomValues(k, static_cast<T>(-1.0f), static_cast<T>(1.0f), SEED_K);
    fillWithRandomValues(v, static_cast<T>(-1.0f), static_cast<T>(1.0f), SEED_V);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    q.memory().hostData();
    k.memory().hostData();
    v.memory().hostData();

    CpuFpReferenceSdpa::forward<T, T, T, T, float>(
        q, k, v, oCpu, SdpaFwdOptions<float>{}.setBand(-1, 0, false).setLse(&lseCpu));
    GpuFpReferenceSdpa::fprop<T, T, T, T, float>(
        q, k, v, oGpu, GpuSdpaFwdOptions<float>{}.setBand(-1, 0, false).setLse(&lseGpu));

    // Non-const TensorView triggers the device->host sync that getHostValue
    // (const) cannot; the GPU wrote lse on-device and only marked it modified.
    TensorView<float> lseCpuView(lseCpu);
    TensorView<float> lseGpuView(lseGpu);

    for(int64_t h = 0; h < numHeads; ++h)
    {
        for(int64_t sq = 0; sq < seqQ; ++sq)
        {
            const float cpuVal = lseCpuView.getHostValue(std::vector<int64_t>{0, h, sq, 0});
            const float gpuVal = lseGpuView.getHostValue(std::vector<int64_t>{0, h, sq});
            if(sq < 2)
            {
                // Fully-masked row: -inf on both references.
                EXPECT_TRUE(std::isinf(cpuVal) && cpuVal < 0.0f)
                    << "CPU LSE should be -inf for masked row h=" << h << ", sq=" << sq;
                EXPECT_TRUE(std::isinf(gpuVal) && gpuVal < 0.0f)
                    << "GPU LSE should be -inf for masked row h=" << h << ", sq=" << sq;
            }
            else
            {
                EXPECT_TRUE(std::isfinite(gpuVal))
                    << "GPU LSE should be finite for visible row h=" << h << ", sq=" << sq;
                // LSE is always FP32, so it uses the float tolerance regardless of T.
                EXPECT_NEAR(gpuVal, cpuVal, gpuRefFwdTolerance<float>())
                    << "GPU/CPU LSE mismatch at h=" << h << ", sq=" << sq;
            }
        }
    }
}

// ============================================================================
// Combinatorial coverage. The cases above exercise each feature in isolation;
// these cross features (GQA/MQA, strided layout, explicit scale, window edges,
// cross-attention, mixed precision) against the CPU oracle to catch interactions
// a one-feature-at-a-time suite would miss.
// ============================================================================

// --- GQA/MQA combined with masking ---

TYPED_TEST(TestGpuSdpaFwdPlain, GqaCausalTopLeft)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({1, 8, 8, 16});
    Tensor<T> k({1, 2, 8, 16});
    Tensor<T> v({1, 2, 8, 16});
    Tensor<T> oCpu({1, 8, 8, 16});
    Tensor<T> oGpu({1, 8, 8, 16});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q,
                                       k,
                                       v,
                                       oCpu,
                                       oGpu,
                                       gpuRefFwdTolerance<T>(),
                                       std::nullopt,
                                       /*attnMask=*/nullptr,
                                       /*leftBound=*/-1,
                                       /*rightBound=*/0,
                                       /*topLeftAlignment=*/true);
}

TYPED_TEST(TestGpuSdpaFwdPlain, GqaSlidingWindowBothBounds)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({1, 8, 8, 16});
    Tensor<T> k({1, 2, 8, 16});
    Tensor<T> v({1, 2, 8, 16});
    Tensor<T> oCpu({1, 8, 8, 16});
    Tensor<T> oGpu({1, 8, 8, 16});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q,
                                       k,
                                       v,
                                       oCpu,
                                       oGpu,
                                       gpuRefFwdTolerance<T>(),
                                       std::nullopt,
                                       /*attnMask=*/nullptr,
                                       /*leftBound=*/2,
                                       /*rightBound=*/1,
                                       /*topLeftAlignment=*/true);
}

TYPED_TEST(TestGpuSdpaFwdPlain, GqaAdditiveMaskRank4)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({1, 8, 8, 16});
    Tensor<T> k({1, 2, 8, 16});
    Tensor<T> v({1, 2, 8, 16});
    Tensor<T> oCpu({1, 8, 8, 16});
    Tensor<T> oGpu({1, 8, 8, 16});
    Tensor<float> mask({1, 8, 8, 8}); // mask is per Q head, not KV head

    fillWithRandomValues(mask, -2.0f, 2.0f, SEED_MASK);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    mask.memory().hostData();

    compareGpuVsCpuSdpaFwd<T, T, T, T>(
        q, k, v, oCpu, oGpu, gpuRefFwdTolerance<T>(), std::nullopt, /*attnMask=*/&mask);
}

TYPED_TEST(TestGpuSdpaFwdPlain, MqaCausalTopLeft)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({1, 8, 8, 16});
    Tensor<T> k({1, 1, 8, 16});
    Tensor<T> v({1, 1, 8, 16});
    Tensor<T> oCpu({1, 8, 8, 16});
    Tensor<T> oGpu({1, 8, 8, 16});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q,
                                       k,
                                       v,
                                       oCpu,
                                       oGpu,
                                       gpuRefFwdTolerance<T>(),
                                       std::nullopt,
                                       /*attnMask=*/nullptr,
                                       /*leftBound=*/-1,
                                       /*rightBound=*/0,
                                       /*topLeftAlignment=*/true);
}

// --- Strided (BSHD) layout combined with other modes ---

TYPED_TEST(TestGpuSdpaFwdPlain, NonPackedBshdCausal)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({1, 2, 8, 16}, TensorLayout::BSHD);
    Tensor<T> k({1, 2, 8, 16}, TensorLayout::BSHD);
    Tensor<T> v({1, 2, 8, 16}, TensorLayout::BSHD);
    Tensor<T> oCpu({1, 2, 8, 16}, TensorLayout::BSHD);
    Tensor<T> oGpu({1, 2, 8, 16}, TensorLayout::BSHD);

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q,
                                       k,
                                       v,
                                       oCpu,
                                       oGpu,
                                       gpuRefFwdTolerance<T>(),
                                       std::nullopt,
                                       /*attnMask=*/nullptr,
                                       /*leftBound=*/-1,
                                       /*rightBound=*/0,
                                       /*topLeftAlignment=*/true);
}

TYPED_TEST(TestGpuSdpaFwdPlain, NonPackedBshdAdditiveMask)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({2, 4, 8, 16}, TensorLayout::BSHD);
    Tensor<T> k({2, 4, 8, 16}, TensorLayout::BSHD);
    Tensor<T> v({2, 4, 8, 16}, TensorLayout::BSHD);
    Tensor<T> oCpu({2, 4, 8, 16}, TensorLayout::BSHD);
    Tensor<T> oGpu({2, 4, 8, 16}, TensorLayout::BSHD);
    Tensor<float> mask({2, 4, 8, 8}); // packed mask against strided q/k/v

    fillWithRandomValues(mask, -2.0f, 2.0f, SEED_MASK);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    mask.memory().hostData();

    compareGpuVsCpuSdpaFwd<T, T, T, T>(
        q, k, v, oCpu, oGpu, gpuRefFwdTolerance<T>(), std::nullopt, /*attnMask=*/&mask);
}

TYPED_TEST(TestGpuSdpaFwdPlain, NonPackedBshdGqa)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({1, 8, 8, 16}, TensorLayout::BSHD);
    Tensor<T> k({1, 2, 8, 16}, TensorLayout::BSHD);
    Tensor<T> v({1, 2, 8, 16}, TensorLayout::BSHD);
    Tensor<T> oCpu({1, 8, 8, 16}, TensorLayout::BSHD);
    Tensor<T> oGpu({1, 8, 8, 16}, TensorLayout::BSHD);

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q, k, v, oCpu, oGpu, gpuRefFwdTolerance<T>());
}

// --- Explicit (non-default) scale combined with other modes ---

TYPED_TEST(TestGpuSdpaFwdPlain, ExplicitScaleCausal)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({1, 2, 8, 16});
    Tensor<T> k({1, 2, 8, 16});
    Tensor<T> v({1, 2, 8, 16});
    Tensor<T> oCpu({1, 2, 8, 16});
    Tensor<T> oGpu({1, 2, 8, 16});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q,
                                       k,
                                       v,
                                       oCpu,
                                       oGpu,
                                       gpuRefFwdTolerance<T>(),
                                       /*attnScaleValue=*/0.25f,
                                       /*attnMask=*/nullptr,
                                       /*leftBound=*/-1,
                                       /*rightBound=*/0,
                                       /*topLeftAlignment=*/true);
}

TYPED_TEST(TestGpuSdpaFwdPlain, ExplicitScaleAdditiveMask)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({1, 2, 8, 16});
    Tensor<T> k({1, 2, 8, 16});
    Tensor<T> v({1, 2, 8, 16});
    Tensor<T> oCpu({1, 2, 8, 16});
    Tensor<T> oGpu({1, 2, 8, 16});
    Tensor<float> mask({1, 2, 8, 8});

    fillWithRandomValues(mask, -2.0f, 2.0f, SEED_MASK);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    mask.memory().hostData();

    compareGpuVsCpuSdpaFwd<T, T, T, T>(
        q, k, v, oCpu, oGpu, gpuRefFwdTolerance<T>(), /*attnScaleValue=*/0.25f, /*attnMask=*/&mask);
}

// --- Sliding-window edge cases ---

// Left bound only (rightBound = -1): exercises the leftBound>=0 branch in
// isolation; every other window case pairs it with a right bound.
TYPED_TEST(TestGpuSdpaFwdPlain, LeftBoundOnly)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({1, 2, 8, 16});
    Tensor<T> k({1, 2, 8, 16});
    Tensor<T> v({1, 2, 8, 16});
    Tensor<T> oCpu({1, 2, 8, 16});
    Tensor<T> oGpu({1, 2, 8, 16});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q,
                                       k,
                                       v,
                                       oCpu,
                                       oGpu,
                                       gpuRefFwdTolerance<T>(),
                                       std::nullopt,
                                       /*attnMask=*/nullptr,
                                       /*leftBound=*/3,
                                       /*rightBound=*/-1,
                                       /*topLeftAlignment=*/true);
}

// Bottom-right sliding window with seqKv != seqQ, so the diagonal offset is
// non-zero (offset = Skv - Sq = -2) and differs from the top-left case.
TYPED_TEST(TestGpuSdpaFwdPlain, SlidingWindowBottomRight)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({1, 2, 8, 16});
    Tensor<T> k({1, 2, 6, 16});
    Tensor<T> v({1, 2, 6, 16});
    Tensor<T> oCpu({1, 2, 8, 16});
    Tensor<T> oGpu({1, 2, 8, 16});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q,
                                       k,
                                       v,
                                       oCpu,
                                       oGpu,
                                       gpuRefFwdTolerance<T>(),
                                       std::nullopt,
                                       /*attnMask=*/nullptr,
                                       /*leftBound=*/2,
                                       /*rightBound=*/1,
                                       /*topLeftAlignment=*/false);
}

// --- Cross-attention (headDimV != headDim, seqKv != seqQ) combined with modes ---

TYPED_TEST(TestGpuSdpaFwdPlain, CrossAttentionCausal)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({1, 2, 8, 16});
    Tensor<T> k({1, 2, 6, 16});
    Tensor<T> v({1, 2, 6, 32});
    Tensor<T> oCpu({1, 2, 8, 32});
    Tensor<T> oGpu({1, 2, 8, 32});

    compareGpuVsCpuSdpaFwd<T, T, T, T>(q,
                                       k,
                                       v,
                                       oCpu,
                                       oGpu,
                                       gpuRefFwdTolerance<T>(),
                                       std::nullopt,
                                       /*attnMask=*/nullptr,
                                       /*leftBound=*/-1,
                                       /*rightBound=*/0,
                                       /*topLeftAlignment=*/false);
}

TYPED_TEST(TestGpuSdpaFwdPlain, CrossAttentionAdditiveMask)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    Tensor<T> q({1, 2, 8, 16});
    Tensor<T> k({1, 2, 6, 16});
    Tensor<T> v({1, 2, 6, 32});
    Tensor<T> oCpu({1, 2, 8, 32});
    Tensor<T> oGpu({1, 2, 8, 32});
    Tensor<float> mask({1, 2, 8, 6}); // [B, H, Sq, Skv]

    fillWithRandomValues(mask, -2.0f, 2.0f, SEED_MASK);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    mask.memory().hostData();

    compareGpuVsCpuSdpaFwd<T, T, T, T>(
        q, k, v, oCpu, oGpu, gpuRefFwdTolerance<T>(), std::nullopt, /*attnMask=*/&mask);
}

// --- Mixed precision (16-bit in, float out) combined with modes ---

TEST(TestGpuSdpaFwdMixedPrecision, HalfInputsFloatOutputCausal)
{
    SKIP_IF_NO_DEVICES();

    Tensor<HalfType> q({1, 2, 8, 16});
    Tensor<HalfType> k({1, 2, 8, 16});
    Tensor<HalfType> v({1, 2, 8, 16});
    Tensor<float> oCpu({1, 2, 8, 16});
    Tensor<float> oGpu({1, 2, 8, 16});

    compareGpuVsCpuSdpaFwd<HalfType, HalfType, HalfType, float>(q,
                                                                k,
                                                                v,
                                                                oCpu,
                                                                oGpu,
                                                                gpuRefFwdTolerance<float>(),
                                                                std::nullopt,
                                                                /*attnMask=*/nullptr,
                                                                /*leftBound=*/-1,
                                                                /*rightBound=*/0,
                                                                /*topLeftAlignment=*/true);
}

TEST(TestGpuSdpaFwdMixedPrecision, Bfloat16InputsFloatOutputAdditiveMask)
{
    SKIP_IF_NO_DEVICES();

    Tensor<BFloat16Type> q({1, 2, 8, 16});
    Tensor<BFloat16Type> k({1, 2, 8, 16});
    Tensor<BFloat16Type> v({1, 2, 8, 16});
    Tensor<float> oCpu({1, 2, 8, 16});
    Tensor<float> oGpu({1, 2, 8, 16});
    Tensor<float> mask({1, 2, 8, 8});

    fillWithRandomValues(mask, -2.0f, 2.0f, SEED_MASK);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    mask.memory().hostData();

    compareGpuVsCpuSdpaFwd<BFloat16Type, BFloat16Type, BFloat16Type, float>(
        q, k, v, oCpu, oGpu, gpuRefFwdTolerance<float>(), std::nullopt, /*attnMask=*/&mask);
}

TEST(TestGpuSdpaFwdMixedPrecision, HalfInputsFloatOutputGqa)
{
    SKIP_IF_NO_DEVICES();

    Tensor<HalfType> q({1, 8, 8, 16});
    Tensor<HalfType> k({1, 2, 8, 16});
    Tensor<HalfType> v({1, 2, 8, 16});
    Tensor<float> oCpu({1, 8, 8, 16});
    Tensor<float> oGpu({1, 8, 8, 16});

    compareGpuVsCpuSdpaFwd<HalfType, HalfType, HalfType, float>(
        q, k, v, oCpu, oGpu, gpuRefFwdTolerance<float>());
}

// ============================================================================
// Padding (seq_len_q / seq_len_kv): GPU reference vs CPU reference.
// ============================================================================

// Per-batch lengths with Sq != Skv in every batch, so a wrong bottom-right offset or a read of
// padding shows up. Every padding slot of Q/K/V holds NaN. Padded query rows must give O = 0
// and LSE = -inf on both references; live rows must agree within the GPU tolerance.
struct GpuPaddingCase
{
    const char* name;
    int64_t rightBound; // -1 = no causal, 0 = causal
    bool topLeft;
    bool withBias;
};

class TestGpuSdpaFwdPaddingFp32 : public ::testing::TestWithParam<GpuPaddingCase>
{
};

TEST_P(TestGpuSdpaFwdPaddingFp32, MatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();
    const auto p = GetParam();
    constexpr int64_t BATCH = 3;
    constexpr int64_t H = 2;
    constexpr int64_t SQ = 9;
    constexpr int64_t SKV = 12;
    constexpr int64_t D = 16;
    const std::vector<int32_t> lensQ = {4, 9, 2};
    const std::vector<int32_t> lensKv = {12, 5, 7};
    const float nan = std::numeric_limits<float>::quiet_NaN();

    Tensor<float> q({BATCH, H, SQ, D});
    Tensor<float> k({BATCH, H, SKV, D});
    Tensor<float> v({BATCH, H, SKV, D});
    Tensor<float> oCpu({BATCH, H, SQ, D});
    Tensor<float> oGpu({BATCH, H, SQ, D});
    Tensor<float> lseCpu({BATCH, H, SQ, 1});
    Tensor<float> lseGpu({BATCH, H, SQ});
    Tensor<float> bias({SQ, SKV});
    Tensor<int32_t> seqLenQ({BATCH, 1, 1, 1});
    Tensor<int32_t> seqLenKv({BATCH, 1, 1, 1});
    q.fillWithRandomValues(-1.0f, 1.0f, SEED_Q);
    k.fillWithRandomValues(-1.0f, 1.0f, SEED_K);
    v.fillWithRandomValues(-1.0f, 1.0f, SEED_K + 1);
    bias.fillWithRandomValues(-1.0f, 1.0f, SEED_K + 2);
    for(int64_t b = 0; b < BATCH; ++b)
    {
        seqLenQ.setHostValue(lensQ[static_cast<size_t>(b)], b, 0, 0, 0);
        seqLenKv.setHostValue(lensKv[static_cast<size_t>(b)], b, 0, 0, 0);
        for(int64_t h = 0; h < H; ++h)
        {
            for(int64_t d = 0; d < D; ++d)
            {
                for(int64_t s = lensQ[static_cast<size_t>(b)]; s < SQ; ++s)
                {
                    q.setHostValue(nan, b, h, s, d);
                }
                for(int64_t s = lensKv[static_cast<size_t>(b)]; s < SKV; ++s)
                {
                    k.setHostValue(nan, b, h, s, d);
                    v.setHostValue(nan, b, h, s, d);
                }
            }
        }
    }
    oGpu.fillWithValue(nan);
    lseGpu.fillWithValue(nan);
    // Non-const accesses so the host edits migrate to the device (see compareGpuVsCpuSdpaFwd).
    q.memory().hostData();
    k.memory().hostData();
    v.memory().hostData();
    bias.memory().hostData();
    seqLenQ.memory().hostData();
    seqLenKv.memory().hostData();

    Tensor<float>* biasPtr = p.withBias ? &bias : nullptr;
    CpuFpReferenceSdpa::forward(q,
                                k,
                                v,
                                oCpu,
                                SdpaFwdOptions<float>{}
                                    .setScale(0.25f)
                                    .setMask(static_cast<const TensorBase<float>*>(biasPtr))
                                    .setBand(-1, p.rightBound, p.topLeft)
                                    .setSeqLens(&seqLenQ, &seqLenKv)
                                    .setLse(&lseCpu));
    GpuFpReferenceSdpa::fprop(q,
                              k,
                              v,
                              oGpu,
                              GpuSdpaFwdOptions<float>{}
                                  .setScale(0.25f)
                                  .setMask(static_cast<TensorBase<float>*>(biasPtr))
                                  .setBand(-1, p.rightBound, p.topLeft)
                                  .setSeqLens(&seqLenQ, &seqLenKv)
                                  .setLse(&lseGpu));

    assertAllClose(oCpu, oGpu, gpuRefFwdTolerance<float>());
    TensorView<float> lseCpuView(lseCpu);
    TensorView<float> lseGpuView(lseGpu);
    for(int64_t b = 0; b < BATCH; ++b)
    {
        for(int64_t h = 0; h < H; ++h)
        {
            for(int64_t s = 0; s < SQ; ++s)
            {
                const float cpu = lseCpuView.getHostValue(std::vector<int64_t>{b, h, s, 0});
                const float gpu = lseGpuView.getHostValue(std::vector<int64_t>{b, h, s});
                if(std::isinf(cpu))
                {
                    EXPECT_EQ(gpu, cpu) << "b=" << b << " h=" << h << " s=" << s;
                }
                else
                {
                    EXPECT_NEAR(gpu, cpu, gpuRefFwdTolerance<float>())
                        << "b=" << b << " h=" << h << " s=" << s;
                }
                if(s >= lensQ[static_cast<size_t>(b)])
                {
                    EXPECT_TRUE(std::isinf(cpu) && cpu < 0) << "padded row LSE b=" << b;
                    EXPECT_EQ(oGpu.getHostValue(b, h, s, 0), 0.0f) << "padded row O b=" << b;
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuSdpaFwdPaddingFp32,
                         ::testing::Values(GpuPaddingCase{"NoMask", -1, true, false},
                                           GpuPaddingCase{"CausalTopLeft", 0, true, false},
                                           GpuPaddingCase{"CausalBottomRight", 0, false, false},
                                           GpuPaddingCase{"CausalBottomRightBias", 0, false, true}),
                         [](const ::testing::TestParamInfo<GpuPaddingCase>& info) {
                             return std::string(info.param.name);
                         });

// ============================================================================
// Paged KV cache: GPU reference (direct page-table addressing) vs CPU reference
// (gather then dense). NHD cache strides, shuffled blocks, and NaN in the null
// block 0 and in every unused slot, which neither reference may read.
// ============================================================================

class TestGpuSdpaFwdPagedFp32 : public ::testing::TestWithParam<int64_t>
{
};

TEST_P(TestGpuSdpaFwdPagedFp32, MatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();
    const int64_t blockSize = GetParam();
    constexpr int64_t BATCH = 3;
    constexpr int64_t H = 2;
    constexpr int64_t SQ = 5;
    constexpr int64_t MAX_KV = 11;
    constexpr int64_t D = 16;
    const std::vector<int32_t> lensQ = {5, 2, 4};
    const std::vector<int32_t> lensKv = {11, 4, 7};
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const int64_t tableSize = (MAX_KV + blockSize - 1) / blockSize;
    const int64_t numBlocks = 1 + BATCH * tableSize;

    Tensor<float> q({BATCH, H, SQ, D});
    const std::vector<int64_t> cDims = {numBlocks, H, blockSize, D};
    const std::vector<int64_t> nhdStrides = {blockSize * H * D, D, H * D, 1};
    Tensor<float> kCache(cDims, nhdStrides);
    Tensor<float> vCache(cDims, nhdStrides);
    Tensor<float> bias({SQ, MAX_KV});
    Tensor<int32_t> seqLenQ({BATCH, 1, 1, 1});
    Tensor<int32_t> seqLenKv({BATCH, 1, 1, 1});
    Tensor<int32_t> pageTable({BATCH, 1, tableSize, 1});
    q.fillWithRandomValues(-1.0f, 1.0f, SEED_Q);
    bias.fillWithRandomValues(-1.0f, 1.0f, SEED_K + 3);
    kCache.fillWithValue(nan);
    vCache.fillWithValue(nan);
    Tensor<float> kSrc({1, 1, 1, D});
    int32_t nextBlock = 0;
    uint32_t seed = SEED_K;
    for(int64_t b = 0; b < BATCH; ++b)
    {
        seqLenQ.setHostValue(lensQ[static_cast<size_t>(b)], b, 0, 0, 0);
        seqLenKv.setHostValue(lensKv[static_cast<size_t>(b)], b, 0, 0, 0);
        const int64_t used = (lensKv[static_cast<size_t>(b)] + blockSize - 1) / blockSize;
        for(int64_t t = 0; t < tableSize; ++t)
        {
            // Shuffled block ids from 1; unused entries point at the null block 0.
            const auto id = static_cast<int32_t>(
                t < used ? ((static_cast<int64_t>(nextBlock++) * 7) % (numBlocks - 1)) + 1 : 0);
            pageTable.setHostValue(id, b, 0, t, 0);
        }
        for(int64_t j = 0; j < lensKv[static_cast<size_t>(b)]; ++j)
        {
            const int64_t block = pageTable.getHostValue(b, 0, j / blockSize, 0);
            for(int64_t h = 0; h < H; ++h)
            {
                kSrc.fillWithRandomValues(-1.0f, 1.0f, seed++);
                for(int64_t d = 0; d < D; ++d)
                {
                    kCache.setHostValue(kSrc.getHostValue(0, 0, 0, d), block, h, j % blockSize, d);
                    vCache.setHostValue(
                        -kSrc.getHostValue(0, 0, 0, (d + 1) % D), block, h, j % blockSize, d);
                }
            }
        }
    }
    q.memory().hostData();
    kCache.memory().hostData();
    vCache.memory().hostData();
    bias.memory().hostData();
    seqLenQ.memory().hostData();
    seqLenKv.memory().hostData();
    pageTable.memory().hostData();

    auto kDense = CpuFpReferenceSdpa::gatherPagedKv(kCache, pageTable, seqLenKv, MAX_KV);
    auto vDense = CpuFpReferenceSdpa::gatherPagedKv(vCache, pageTable, seqLenKv, MAX_KV);
    Tensor<float> oCpu({BATCH, H, SQ, D});
    Tensor<float> oGpu({BATCH, H, SQ, D});
    Tensor<float> lseCpu({BATCH, H, SQ, 1});
    Tensor<float> lseGpu({BATCH, H, SQ});
    oGpu.fillWithValue(nan);
    lseGpu.fillWithValue(nan);
    CpuFpReferenceSdpa::forward(q,
                                kDense,
                                vDense,
                                oCpu,
                                SdpaFwdOptions<float>{}
                                    .setScale(0.3f)
                                    .setMask(static_cast<const TensorBase<float>*>(&bias))
                                    .setBand(-1, 0, false)
                                    .setSeqLens(&seqLenQ, &seqLenKv)
                                    .setLse(&lseCpu));
    GpuFpReferenceSdpa::fprop(q,
                              kCache,
                              vCache,
                              oGpu,
                              GpuSdpaFwdOptions<float>{}
                                  .setScale(0.3f)
                                  .setMask(static_cast<TensorBase<float>*>(&bias))
                                  .setBand(-1, 0, false)
                                  .setSeqLens(&seqLenQ, &seqLenKv)
                                  .setPaged(&pageTable, &pageTable, MAX_KV)
                                  .setLse(&lseGpu));

    assertAllClose(oCpu, oGpu, gpuRefFwdTolerance<float>());
    TensorView<float> lseCpuView(lseCpu);
    TensorView<float> lseGpuView(lseGpu);
    for(int64_t b = 0; b < BATCH; ++b)
    {
        for(int64_t h = 0; h < H; ++h)
        {
            for(int64_t s = 0; s < SQ; ++s)
            {
                const float cpu = lseCpuView.getHostValue(std::vector<int64_t>{b, h, s, 0});
                const float gpu = lseGpuView.getHostValue(std::vector<int64_t>{b, h, s});
                if(std::isinf(cpu))
                {
                    EXPECT_EQ(gpu, cpu) << "b=" << b << " h=" << h << " s=" << s;
                }
                else
                {
                    EXPECT_NEAR(gpu, cpu, gpuRefFwdTolerance<float>())
                        << "b=" << b << " h=" << h << " s=" << s;
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuSdpaFwdPagedFp32,
                         ::testing::Values(1, 3, 4),
                         [](const ::testing::TestParamInfo<int64_t>& info) {
                             return "BlockSize" + std::to_string(info.param);
                         });

// ============================================================================
// Attention sink: GPU reference vs CPU reference, with GQA, padding, and a row
// whose keys are all masked by the bias (O = 0, LSE = sink).
// ============================================================================

TEST(TestGpuSdpaFwdSinkFp32, MatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();
    constexpr int64_t BATCH = 2;
    constexpr int64_t H = 4;
    constexpr int64_t HKV = 2;
    constexpr int64_t SQ = 6;
    constexpr int64_t SKV = 9;
    constexpr int64_t D = 16;
    const float negInf = -std::numeric_limits<float>::infinity();

    Tensor<float> q({BATCH, H, SQ, D});
    Tensor<float> k({BATCH, HKV, SKV, D});
    Tensor<float> v({BATCH, HKV, SKV, D});
    Tensor<float> bias({SQ, SKV});
    Tensor<float> sink({1, H, 1, 1});
    Tensor<int32_t> seqLenQ({BATCH, 1, 1, 1});
    Tensor<int32_t> seqLenKv({BATCH, 1, 1, 1});
    q.fillWithRandomValues(-1.0f, 1.0f, SEED_Q);
    k.fillWithRandomValues(-1.0f, 1.0f, SEED_K);
    v.fillWithRandomValues(-1.0f, 1.0f, SEED_K + 1);
    bias.fillWithRandomValues(-1.0f, 1.0f, SEED_K + 2);
    sink.fillWithRandomValues(-2.0f, 2.0f, SEED_K + 3);
    for(int64_t j = 0; j < SKV; ++j)
    {
        bias.setHostValue(negInf, 1, j); // query row 1 has no live key
    }
    seqLenQ.setHostValue(6, 0, 0, 0, 0);
    seqLenQ.setHostValue(4, 1, 0, 0, 0);
    seqLenKv.setHostValue(9, 0, 0, 0, 0);
    seqLenKv.setHostValue(5, 1, 0, 0, 0);
    for(auto* t : {&q, &k, &v, &bias, &sink})
    {
        t->memory().hostData();
    }
    seqLenQ.memory().hostData();
    seqLenKv.memory().hostData();

    Tensor<float> oCpu({BATCH, H, SQ, D});
    Tensor<float> oGpu({BATCH, H, SQ, D});
    Tensor<float> lseCpu({BATCH, H, SQ, 1});
    Tensor<float> lseGpu({BATCH, H, SQ});
    // cuDNN Max / Sum_exp: GPU rank-3 like the LSE.
    Tensor<float> maxCpu({BATCH, H, SQ, 1});
    Tensor<float> sumExpCpu({BATCH, H, SQ, 1});
    Tensor<float> maxGpu({BATCH, H, SQ});
    Tensor<float> sumExpGpu({BATCH, H, SQ});
    oGpu.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    lseGpu.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    maxGpu.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    sumExpGpu.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    CpuFpReferenceSdpa::forward(q,
                                k,
                                v,
                                oCpu,
                                SdpaFwdOptions<float>{}
                                    .setScale(0.25f)
                                    .setMask(static_cast<const TensorBase<float>*>(&bias))
                                    .setSeqLens(&seqLenQ, &seqLenKv)
                                    .setSink(&sink)
                                    .setLse(&lseCpu)
                                    .setRowStats(&maxCpu, &sumExpCpu));
    GpuFpReferenceSdpa::fprop(q,
                              k,
                              v,
                              oGpu,
                              GpuSdpaFwdOptions<float>{}
                                  .setScale(0.25f)
                                  .setMask(static_cast<TensorBase<float>*>(&bias))
                                  .setSeqLens(&seqLenQ, &seqLenKv)
                                  .setSink(&sink)
                                  .setLse(&lseGpu)
                                  .setRowStats(&maxGpu, &sumExpGpu));

    // Max / Sum_exp: equal where infinite (dead rows: -inf / 0), close elsewhere, and for a live
    // row with no live key the sink alone (Max = sink, Sum_exp = 1).
    maxGpu.memory().hostData(); // sync the device results to the host before const reads
    sumExpGpu.memory().hostData();
    for(int64_t b = 0; b < BATCH; ++b)
    {
        for(int64_t h = 0; h < H; ++h)
        {
            for(int64_t s = 0; s < SQ; ++s)
            {
                const float mCpu = maxCpu.getHostValue(b, h, s, 0);
                const float mGpu = maxGpu.getHostValue(b, h, s);
                const float eCpu = sumExpCpu.getHostValue(b, h, s, 0);
                const float eGpu = sumExpGpu.getHostValue(b, h, s);
                if(std::isinf(mCpu))
                {
                    EXPECT_EQ(mGpu, mCpu) << "b=" << b << " h=" << h << " s=" << s;
                }
                else
                {
                    EXPECT_NEAR(mGpu, mCpu, 1e-5F) << "b=" << b << " h=" << h << " s=" << s;
                }
                EXPECT_NEAR(eGpu, eCpu, 1e-4F * std::max(1.0F, eCpu))
                    << "b=" << b << " h=" << h << " s=" << s;
                if(s == 1 && s < (b == 0 ? 6 : 4))
                {
                    EXPECT_EQ(mCpu, sink.getHostValue(0, h, 0, 0));
                    EXPECT_EQ(eCpu, 1.0F);
                }
            }
        }
    }

    assertAllClose(oCpu, oGpu, gpuRefFwdTolerance<float>());
    TensorView<float> lseCpuView(lseCpu);
    TensorView<float> lseGpuView(lseGpu);
    for(int64_t b = 0; b < BATCH; ++b)
    {
        for(int64_t h = 0; h < H; ++h)
        {
            for(int64_t s = 0; s < SQ; ++s)
            {
                const float cpu = lseCpuView.getHostValue(std::vector<int64_t>{b, h, s, 0});
                const float gpu = lseGpuView.getHostValue(std::vector<int64_t>{b, h, s});
                if(std::isinf(cpu))
                {
                    EXPECT_EQ(gpu, cpu) << "b=" << b << " h=" << h << " s=" << s;
                }
                else
                {
                    EXPECT_NEAR(gpu, cpu, gpuRefFwdTolerance<float>())
                        << "b=" << b << " h=" << h << " s=" << s;
                }
            }
            // The fully masked row keeps LSE = sink on both references.
            EXPECT_NEAR(lseGpuView.getHostValue(std::vector<int64_t>{b, h, 1}),
                        sink.getHostValue(0, h, 0, 0),
                        gpuRefFwdTolerance<float>());
        }
    }
}

// ============================================================================
// Explicit-mask dropout (cuDNN custom dropout): GPU reference vs CPU reference,
// mask broadcast over batch, with GQA, bias and bottom-right causal.
// ============================================================================

TEST(TestGpuSdpaFwdDropoutFp32, MatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();
    constexpr int64_t BATCH = 2;
    constexpr int64_t H = 4;
    constexpr int64_t HKV = 2;
    constexpr int64_t SQ = 5;
    constexpr int64_t SKV = 8;
    constexpr int64_t D = 16;
    Tensor<float> q({BATCH, H, SQ, D});
    Tensor<float> k({BATCH, HKV, SKV, D});
    Tensor<float> v({BATCH, HKV, SKV, D});
    Tensor<float> bias({SQ, SKV});
    Tensor<float> keepMask({1, H, SQ, SKV});
    q.fillWithRandomValues(-1.0f, 1.0f, SEED_Q);
    k.fillWithRandomValues(-1.0f, 1.0f, SEED_K);
    v.fillWithRandomValues(-1.0f, 1.0f, SEED_K + 1);
    bias.fillWithRandomValues(-1.0f, 1.0f, SEED_K + 2);
    for(size_t i = 0; i < keepMask.elementCount(); ++i)
    {
        keepMask.memory().hostData()[i] = (i * 7 % 5 < 3) ? 1.0f : 0.0f;
    }
    for(auto* t : {&q, &k, &v, &bias, &keepMask})
    {
        t->memory().hostData();
    }
    Tensor<float> oCpu({BATCH, H, SQ, D});
    Tensor<float> oGpu({BATCH, H, SQ, D});
    oGpu.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    CpuFpReferenceSdpa::forward(q,
                                k,
                                v,
                                oCpu,
                                SdpaFwdOptions<float>{}
                                    .setScale(0.25f)
                                    .setMask(static_cast<const TensorBase<float>*>(&bias))
                                    .setBand(-1, 0, false)
                                    .setDropout(&keepMask, 1.0f / 0.6f));
    GpuFpReferenceSdpa::fprop(q,
                              k,
                              v,
                              oGpu,
                              GpuSdpaFwdOptions<float>{}
                                  .setScale(0.25f)
                                  .setMask(static_cast<TensorBase<float>*>(&bias))
                                  .setBand(-1, 0, false)
                                  .setDropout(&keepMask, 1.0f / 0.6f));
    assertAllClose(oCpu, oGpu, gpuRefFwdTolerance<float>());
}

// ============================================================================
// fp8 Q/K/V with descales: GPU reference (device dequantize, then the float pipeline) vs the CPU
// reference, for OCP E4M3 and the gfx942 FNUZ encoding.
// ============================================================================

namespace
{

// fp8 bit patterns that decode to NaN: OCP E4M3 has 0x7F / 0xFF, OCP E5M2 every exponent-31
// code with a nonzero mantissa, the FNUZ formats the single 0x80.
template <typename Fp8>
bool isFp8NanBits(unsigned bits)
{
    if constexpr(std::is_same_v<Fp8, fp8_e4m3_fnuz> || std::is_same_v<Fp8, fp8_e5m2_fnuz>)
    {
        return bits == 0x80u;
    }
    else if constexpr(std::is_same_v<Fp8, fp8_e5m2>)
    {
        return (bits & 0x7Cu) == 0x7Cu && (bits & 0x3u) != 0;
    }
    else
    {
        return (bits & 0x7Fu) == 0x7Fu;
    }
}

template <typename Fp8>
void fillFp8(Tensor<Fp8>& t, unsigned int seed)
{
    Tensor<float> values(t.dims());
    values.fillWithRandomValues(-2.0f, 2.0f, seed);
    const auto& d = t.dims();
    for(int64_t b = 0; b < d[0]; ++b)
    {
        for(int64_t h = 0; h < d[1]; ++h)
        {
            for(int64_t s = 0; s < d[2]; ++s)
            {
                for(int64_t e = 0; e < d[3]; ++e)
                {
                    t.setHostValue(Fp8(values.getHostValue(b, h, s, e)), b, h, s, e);
                }
            }
        }
    }
}

// GQA, causal, per-(batch, KV head) Q/K descales and a scalar V descale.
template <typename Fp8>
void expectFp8ForwardMatchesCpu()
{
    constexpr int64_t BATCH = 2;
    constexpr int64_t H = 4;
    constexpr int64_t HKV = 2;
    constexpr int64_t SQ = 5;
    constexpr int64_t SKV = 7;
    constexpr int64_t D = 16;
    Tensor<Fp8> q({BATCH, H, SQ, D});
    Tensor<Fp8> k({BATCH, HKV, SKV, D});
    Tensor<Fp8> v({BATCH, HKV, SKV, D});
    fillFp8(q, SEED_Q);
    fillFp8(k, SEED_K);
    fillFp8(v, SEED_V);
    Tensor<float> descaleQ({BATCH, HKV, 1, 1});
    Tensor<float> descaleK({BATCH, HKV, 1, 1});
    Tensor<float> descaleV({1});
    descaleQ.fillWithRandomValues(0.25f, 2.0f, SEED_MASK);
    descaleK.fillWithRandomValues(0.25f, 2.0f, SEED_MASK + 1);
    descaleV.fillWithValue(0.75f);

    Tensor<float> oCpu({BATCH, H, SQ, D});
    Tensor<float> oGpu({BATCH, H, SQ, D});
    Tensor<float> lseCpu({BATCH, H, SQ, 1}); // the GPU reference takes the LSE as rank 3
    Tensor<float> lseGpu({BATCH, H, SQ});
    oGpu.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    lseGpu.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    CpuFpReferenceSdpa::forward<Fp8, Fp8, Fp8, float>(
        q,
        k,
        v,
        oCpu,
        SdpaFwdOptions<float>{}
            .setScale(0.25f)
            .setBand(-1, 0, true)
            .setLse(&lseCpu)
            .setDescales(&descaleQ, &descaleK, &descaleV));
    GpuFpReferenceSdpa::fprop<Fp8, Fp8, Fp8, float>(
        q,
        k,
        v,
        oGpu,
        GpuSdpaFwdOptions<float>{}
            .setScale(0.25f)
            .setBand(-1, 0, true)
            .setLse(&lseGpu)
            .setDescales(&descaleQ, &descaleK, &descaleV));
    assertAllClose(oCpu, oGpu, gpuRefFwdTolerance<float>());
    lseGpu.memory().hostData();
    for(int64_t b = 0; b < BATCH; ++b)
    {
        for(int64_t h = 0; h < H; ++h)
        {
            for(int64_t s = 0; s < SQ; ++s)
            {
                EXPECT_NEAR(lseGpu.getHostValue(b, h, s), lseCpu.getHostValue(b, h, s, 0), 1e-5F)
                    << "b=" << b << " h=" << h << " s=" << s;
            }
        }
    }
}

// Every non-NaN bit pattern decodes on the device exactly as on the host: with one key the
// softmax weight is exactly 1, so O = V * descaleV elementwise.
template <typename Fp8>
void expectFp8DeviceDecodeMatchesHost()
{
    constexpr int64_t D = 256;
    Tensor<Fp8> q({1, 1, 1, D});
    Tensor<Fp8> k({1, 1, 1, D});
    Tensor<Fp8> v({1, 1, 1, D});
    for(int64_t e = 0; e < D; ++e)
    {
        Fp8 bits{};
        bits.data = static_cast<uint8_t>(isFp8NanBits<Fp8>(static_cast<unsigned>(e)) ? 0 : e);
        q.setHostValue(Fp8(0.0f), 0, 0, 0, e);
        k.setHostValue(Fp8(0.0f), 0, 0, 0, e);
        v.setHostValue(bits, 0, 0, 0, e);
    }
    Tensor<float> one({1});
    one.fillWithValue(1.0f);
    Tensor<float> o({1, 1, 1, D});
    o.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    GpuFpReferenceSdpa::fprop<Fp8, Fp8, Fp8, float>(
        q, k, v, o, GpuSdpaFwdOptions<float>{}.setDescales(&one, &one, &one));
    o.memory().hostData();
    for(int64_t e = 0; e < D; ++e)
    {
        EXPECT_EQ(o.getHostValue(0, 0, 0, e), static_cast<float>(v.getHostValue(0, 0, 0, e)))
            << "bits 0x" << std::hex << e;
    }
}

} // namespace

TEST(TestGpuSdpaFwdFp8, OcpE4m3MatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();
    expectFp8ForwardMatchesCpu<fp8_e4m3>();
}

TEST(TestGpuSdpaFwdFp8, FnuzE4m3MatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();
    expectFp8ForwardMatchesCpu<fp8_e4m3_fnuz>();
}

TEST(TestGpuSdpaFwdFp8, OcpE4m3DeviceDecodeMatchesHost)
{
    SKIP_IF_NO_DEVICES();
    expectFp8DeviceDecodeMatchesHost<fp8_e4m3>();
}

TEST(TestGpuSdpaFwdFp8, FnuzE4m3DeviceDecodeMatchesHost)
{
    SKIP_IF_NO_DEVICES();
    expectFp8DeviceDecodeMatchesHost<fp8_e4m3_fnuz>();
}

TEST(TestGpuSdpaFwdFp8, OcpE5m2DeviceDecodeMatchesHost)
{
    SKIP_IF_NO_DEVICES();
    expectFp8DeviceDecodeMatchesHost<fp8_e5m2>();
}

TEST(TestGpuSdpaFwdFp8, FnuzE5m2DeviceDecodeMatchesHost)
{
    SKIP_IF_NO_DEVICES();
    expectFp8DeviceDecodeMatchesHost<fp8_e5m2_fnuz>();
}

TEST(TestGpuSdpaFwdFp8, OcpE5m2MatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();
    expectFp8ForwardMatchesCpu<fp8_e5m2>();
}

TEST(TestGpuSdpaFwdFp8, FnuzE5m2MatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();
    expectFp8ForwardMatchesCpu<fp8_e5m2_fnuz>();
}

// A per-head descale on a paged K or V is ill-defined (container blocks are not batches).
TEST(TestGpuSdpaFwdFp8, PerHeadDescaleOnPagedKvThrows)
{
    SKIP_IF_NO_DEVICES();
    Tensor<fp8_e4m3> q({1, 2, 2, 8});
    Tensor<fp8_e4m3> kCache({2, 2, 4, 8});
    Tensor<fp8_e4m3> vCache({2, 2, 4, 8});
    Tensor<int32_t> table({1, 1, 2, 1});
    Tensor<int32_t> lenQ({1, 1, 1, 1});
    Tensor<int32_t> lenKv({1, 1, 1, 1});
    table.setHostValue(1, 0, 0, 0, 0);
    table.setHostValue(0, 0, 0, 1, 0);
    lenQ.setHostValue(2, 0, 0, 0, 0);
    lenKv.setHostValue(6, 0, 0, 0, 0);
    Tensor<float> scalar({1});
    scalar.fillWithValue(1.0f);
    Tensor<float> perHead({1, 2, 1, 1});
    perHead.fillWithValue(1.0f);
    Tensor<float> o({1, 2, 2, 8});
    EXPECT_THROW((GpuFpReferenceSdpa::fprop<fp8_e4m3, fp8_e4m3, fp8_e4m3, float>(
                     q,
                     kCache,
                     vCache,
                     o,
                     GpuSdpaFwdOptions<float>{}
                         .setSeqLens(&lenQ, &lenKv)
                         .setPaged(&table, &table, 6)
                         .setDescales(&scalar, &perHead, &scalar))),
                 std::invalid_argument);
}

// ============================================================================
// cuDNN sdpa_fp8 softmax and output scaling (SEM-024) on the GPU.
// ============================================================================

namespace
{

// Floats that probe an fp8 encoder: every code's value, the midpoints between neighbours (ties)
// and the floats either side of them, out-of-range and tiny values, and random values.
template <typename Fp8>
std::vector<float> fp8EncodeProbes()
{
    std::vector<float> xs;
    for(unsigned c = 0; c < 256; ++c)
    {
        if(isFp8NanBits<Fp8>(c))
        {
            continue;
        }
        Fp8 a{};
        a.data = static_cast<uint8_t>(c);
        const auto va = static_cast<float>(a);
        xs.push_back(va);
        if(c + 1 < 256 && !isFp8NanBits<Fp8>(c + 1) && (c & 0x7Fu) != 0x7Fu)
        {
            Fp8 b{};
            b.data = static_cast<uint8_t>(c + 1);
            const float mid = 0.5f * (va + static_cast<float>(b));
            xs.push_back(mid);
            xs.push_back(std::nextafter(mid, 1e9f));
            xs.push_back(std::nextafter(mid, -1e9f));
        }
    }
    for(const float x : {1000.0f, -1000.0f, 1e-30f, -1e-30f, 0.0f})
    {
        xs.push_back(x);
    }
    Tensor<float> random({512});
    random.fillWithRandomValues(-300.0f, 300.0f, SEED_Q);
    for(int64_t i = 0; i < 512; ++i)
    {
        xs.push_back(random.getHostValue(i));
    }
    // No -0.0 (OCP code 0x80 decodes to it): P @ V accumulates from +0, so it reaches the encoder
    // as +0. -1e-30 still probes the negative-zero encoding.
    xs.erase(
        std::remove_if(xs.begin(), xs.end(), [](float x) { return x == 0.0f && std::signbit(x); }),
        xs.end());
    return xs;
}

// With one key the softmax weight is exactly 1, so O_f = V and O = Fp8(V * 1): the device encoder
// (GpuRefTypes.h fp8E4m3FromFloat) must give the host encoder's bits for every probe.
template <typename Fp8>
void expectFp8DeviceEncodeMatchesHost()
{
    const auto xs = fp8EncodeProbes<Fp8>();
    const auto n = static_cast<int64_t>(xs.size());
    Tensor<float> q({1, 1, 1, 1});
    Tensor<float> k({1, 1, 1, 1});
    Tensor<float> v({1, 1, 1, n});
    q.fillWithValue(0.0f);
    k.fillWithValue(0.0f);
    for(int64_t e = 0; e < n; ++e)
    {
        v.setHostValue(xs[static_cast<size_t>(e)], 0, 0, 0, e);
    }
    Tensor<Fp8> o({1, 1, 1, n});
    GpuFpReferenceSdpa::fprop<float, float, float, Fp8>(
        q, k, v, o, GpuSdpaFwdOptions<float>{}.setOutputScale(1.0f));
    o.memory().hostData();
    int mismatches = 0;
    for(int64_t e = 0; e < n; ++e)
    {
        const float x = xs[static_cast<size_t>(e)];
        const auto want = Fp8(x).data;
        const auto got = o.getHostValue(0, 0, 0, e).data;
        if(want != got && ++mismatches <= 8)
        {
            ADD_FAILURE() << std::hexfloat << "x=" << x << std::hex << " host=0x"
                          << static_cast<int>(want) << " device=0x" << static_cast<int>(got);
        }
    }
    EXPECT_EQ(mismatches, 0) << "of " << n << " probes";
}

// GQA, bottom-right causal, every cuDNN sdpa_fp8 scale and both amax outputs, fp8 O. O_f differs
// between devices by summation order, so an fp8 code may flip where O_f * scaleO sits on a
// rounding boundary: decoded values must agree to one fp8 step and almost every code exactly.
template <typename Fp8>
void expectFp8ScaledForwardMatchesCpu()
{
    constexpr int64_t BATCH = 2;
    constexpr int64_t H = 4;
    constexpr int64_t HKV = 2;
    constexpr int64_t SQ = 6;
    constexpr int64_t SKV = 9;
    constexpr int64_t D = 16;
    Tensor<Fp8> q({BATCH, H, SQ, D});
    Tensor<Fp8> k({BATCH, HKV, SKV, D});
    Tensor<Fp8> v({BATCH, HKV, SKV, D});
    fillFp8(q, SEED_Q + 10);
    fillFp8(k, SEED_K + 10);
    fillFp8(v, SEED_V + 10);
    Tensor<float> descaleQ({1});
    Tensor<float> descaleK({1});
    Tensor<float> descaleV({1});
    descaleQ.fillWithValue(0.5f);
    descaleK.fillWithValue(0.75f);
    descaleV.fillWithValue(1.25f);

    Tensor<Fp8> oCpu({BATCH, H, SQ, D});
    Tensor<Fp8> oGpu({BATCH, H, SQ, D});
    Tensor<float> amaxSCpu({1});
    Tensor<float> amaxOCpu({1});
    Tensor<float> amaxSGpu({1});
    Tensor<float> amaxOGpu({1});
    amaxSGpu.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    amaxOGpu.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    const auto scaleS = 32.0f;
    const auto scaleO = 8.0f;
    CpuFpReferenceSdpa::forward<Fp8, Fp8, Fp8, Fp8>(
        q,
        k,
        v,
        oCpu,
        SdpaFwdOptions<float>{}
            .setScale(0.25f)
            .setBand(-1, 0, false)
            .setDescales(&descaleQ, &descaleK, &descaleV)
            .setSoftmaxScales(scaleS, 1.0f / scaleS)
            .setOutputScale(scaleO)
            .setAmax(&amaxSCpu, &amaxOCpu));
    GpuFpReferenceSdpa::fprop<Fp8, Fp8, Fp8, Fp8>(q,
                                                  k,
                                                  v,
                                                  oGpu,
                                                  GpuSdpaFwdOptions<float>{}
                                                      .setScale(0.25f)
                                                      .setBand(-1, 0, false)
                                                      .setDescales(&descaleQ, &descaleK, &descaleV)
                                                      .setSoftmaxScales(scaleS, 1.0f / scaleS)
                                                      .setOutputScale(scaleO)
                                                      .setAmax(&amaxSGpu, &amaxOGpu));
    oGpu.memory().hostData();
    amaxSGpu.memory().hostData();
    amaxOGpu.memory().hostData();
    EXPECT_EQ(amaxSGpu.getHostValue(0), amaxSCpu.getHostValue(0));
    EXPECT_NEAR(
        amaxOGpu.getHostValue(0), amaxOCpu.getHostValue(0), 1e-5f * amaxOCpu.getHostValue(0));

    constexpr bool E5M2 = std::is_same_v<Fp8, fp8_e5m2> || std::is_same_v<Fp8, fp8_e5m2_fnuz>;
    constexpr float STEP = E5M2 ? 0.25f : 0.125f;
    int64_t exact = 0;
    int64_t total = 0;
    for(int64_t b = 0; b < BATCH; ++b)
    {
        for(int64_t h = 0; h < H; ++h)
        {
            for(int64_t s = 0; s < SQ; ++s)
            {
                for(int64_t d = 0; d < D; ++d)
                {
                    const auto c = static_cast<float>(oCpu.getHostValue(b, h, s, d));
                    const auto g = static_cast<float>(oGpu.getHostValue(b, h, s, d));
                    // One fp8 step is at most 1/8 (E4M3) or 1/4 (E5M2) of the value, plus the
                    // subnormal spacing near zero.
                    EXPECT_NEAR(g, c, std::fabs(c) * STEP + 0.002f)
                        << "b=" << b << " h=" << h << " s=" << s << " d=" << d;
                    exact
                        += oCpu.getHostValue(b, h, s, d).data == oGpu.getHostValue(b, h, s, d).data
                               ? 1
                               : 0;
                    ++total;
                }
            }
        }
    }
    EXPECT_GE(exact * 100, total * 99) << exact << " of " << total << " codes equal";
}

} // namespace

TEST(TestGpuSdpaFwdFp8, OcpE4m3DeviceEncodeMatchesHost)
{
    SKIP_IF_NO_DEVICES();
    expectFp8DeviceEncodeMatchesHost<fp8_e4m3>();
}

TEST(TestGpuSdpaFwdFp8, FnuzE4m3DeviceEncodeMatchesHost)
{
    SKIP_IF_NO_DEVICES();
    expectFp8DeviceEncodeMatchesHost<fp8_e4m3_fnuz>();
}

TEST(TestGpuSdpaFwdFp8, OcpE4m3ScaledForwardMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    expectFp8ScaledForwardMatchesCpu<fp8_e4m3>();
}

TEST(TestGpuSdpaFwdFp8, FnuzE4m3ScaledForwardMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    expectFp8ScaledForwardMatchesCpu<fp8_e4m3_fnuz>();
}

TEST(TestGpuSdpaFwdFp8, OcpE5m2DeviceEncodeMatchesHost)
{
    SKIP_IF_NO_DEVICES();
    expectFp8DeviceEncodeMatchesHost<fp8_e5m2>();
}

TEST(TestGpuSdpaFwdFp8, FnuzE5m2DeviceEncodeMatchesHost)
{
    SKIP_IF_NO_DEVICES();
    expectFp8DeviceEncodeMatchesHost<fp8_e5m2_fnuz>();
}

TEST(TestGpuSdpaFwdFp8, OcpE5m2ScaledForwardMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    expectFp8ScaledForwardMatchesCpu<fp8_e5m2>();
}

TEST(TestGpuSdpaFwdFp8, FnuzE5m2ScaledForwardMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    expectFp8ScaledForwardMatchesCpu<fp8_e5m2_fnuz>();
}

// P is stored in Q's fp8 format, so softmax scaling needs fp8 Q, as on the CPU.
TEST(TestGpuSdpaFwdFp8, SoftmaxScalingNeedsFp8Q)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> q({1, 1, 1, 2});
    Tensor<float> o({1, 1, 1, 2});
    q.fillWithValue(1.0f);
    EXPECT_THROW(GpuFpReferenceSdpa::fprop(
                     q, q, q, o, GpuSdpaFwdOptions<float>{}.setSoftmaxScales(2.0f, 0.5f)),
                 std::invalid_argument);
}

// ============================================================================
// cuDNN block mask (SEM-025): GPU reference vs CPU reference, with GQA, padding and bottom-right
// causal, on shapes that cross tile and mask-byte boundaries.
// ============================================================================

TEST(TestGpuSdpaFwdBlockMaskFp32, MatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();
    constexpr int64_t BATCH = 2;
    constexpr int64_t H = 4;
    constexpr int64_t HKV = 2;
    constexpr int64_t SQ = 260;
    constexpr int64_t SKV = 1100;
    constexpr int64_t D = 8;
    Tensor<float> q({BATCH, H, SQ, D});
    Tensor<float> k({BATCH, HKV, SKV, D});
    Tensor<float> v({BATCH, HKV, SKV, D});
    q.fillWithRandomValues(-1.0f, 1.0f, SEED_Q);
    k.fillWithRandomValues(-1.0f, 1.0f, SEED_K);
    v.fillWithRandomValues(-1.0f, 1.0f, SEED_V);
    Tensor<uint8_t> blockMask({BATCH, H, 3, 2});
    for(int64_t i = 0; i < static_cast<int64_t>(blockMask.elementCount()); ++i)
    {
        blockMask.memory().hostData()[i] = static_cast<uint8_t>((0x5A + 29 * i) & 0xFF);
    }
    Tensor<int32_t> seqLenQ({BATCH, 1, 1, 1});
    Tensor<int32_t> seqLenKv({BATCH, 1, 1, 1});
    seqLenQ.setHostValue(260, 0, 0, 0, 0);
    seqLenQ.setHostValue(200, 1, 0, 0, 0);
    seqLenKv.setHostValue(1100, 0, 0, 0, 0);
    seqLenKv.setHostValue(700, 1, 0, 0, 0);

    Tensor<float> oCpu({BATCH, H, SQ, D});
    Tensor<float> oGpu({BATCH, H, SQ, D});
    oGpu.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    CpuFpReferenceSdpa::forward(q,
                                k,
                                v,
                                oCpu,
                                SdpaFwdOptions<float>{}
                                    .setScale(0.3f)
                                    .setBand(-1, 0, false)
                                    .setSeqLens(&seqLenQ, &seqLenKv)
                                    .setBlockMask(&blockMask));
    GpuFpReferenceSdpa::fprop(q,
                              k,
                              v,
                              oGpu,
                              GpuSdpaFwdOptions<float>{}
                                  .setScale(0.3f)
                                  .setBand(-1, 0, false)
                                  .setSeqLens(&seqLenQ, &seqLenKv)
                                  .setBlockMask(&blockMask));
    assertAllClose(oCpu, oGpu, gpuRefFwdTolerance<float>());
}

// ============================================================================
// Standard tier: CPU vs GPU forward at medium shapes. Rows of hundreds of keys exercise the
// accumulation order and the multi-tile / multi-byte block mask; each case combines several
// features. Quick runs the small per-feature cases above on every change.
// ============================================================================

namespace
{

struct FwdSweepCase
{
    const char* name;
    int64_t batch, heads, headsKv, seqQ, seqKv, headDim, headDimV;
    int64_t leftBound, rightBound;
    bool topLeft;
    bool bias, padding, sink, blockMask, alibi, softcap;
};

std::string fwdSweepName(const ::testing::TestParamInfo<FwdSweepCase>& info)
{
    return info.param.name;
}

} // namespace

class TestGpuSdpaFwdSweepFp32 : public ::testing::TestWithParam<FwdSweepCase>
{
};

TEST_P(TestGpuSdpaFwdSweepFp32, MatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();
    const auto& c = GetParam();
    const std::vector<int64_t> oDims = {c.batch, c.heads, c.seqQ, c.headDimV};
    Tensor<float> q({c.batch, c.heads, c.seqQ, c.headDim});
    Tensor<float> k({c.batch, c.headsKv, c.seqKv, c.headDim});
    Tensor<float> v({c.batch, c.headsKv, c.seqKv, c.headDimV});
    q.fillWithRandomValues(-1.0f, 1.0f, 301);
    k.fillWithRandomValues(-1.0f, 1.0f, 302);
    v.fillWithRandomValues(-1.0f, 1.0f, 303);
    Tensor<float> bias({1, c.heads, c.seqQ, c.seqKv});
    bias.fillWithRandomValues(-2.0f, 2.0f, 304);
    Tensor<float> sink({1, c.heads, 1, 1});
    sink.fillWithRandomValues(-1.0f, 3.0f, 305);
    Tensor<int32_t> seqLenQ({c.batch, 1, 1, 1});
    Tensor<int32_t> seqLenKv({c.batch, 1, 1, 1});
    for(int64_t b = 0; b < c.batch; ++b)
    {
        seqLenQ.setHostValue(static_cast<int32_t>(c.seqQ - 37 * b), b, 0, 0, 0);
        seqLenKv.setHostValue(static_cast<int32_t>(c.seqKv - 53 * b), b, 0, 0, 0);
    }
    const auto tilesKv = (c.seqKv + 127) / 128;
    Tensor<uint8_t> blockMask({c.batch, c.heads, (c.seqQ + 127) / 128, (tilesKv + 7) / 8});
    for(int64_t i = 0; i < static_cast<int64_t>(blockMask.elementCount()); ++i)
    {
        blockMask.memory().hostData()[i] = static_cast<uint8_t>((0x6B + 41 * i) & 0xFF);
    }

    Tensor<float> oCpu(oDims);
    Tensor<float> oGpu(oDims);
    Tensor<float> lseCpu({c.batch, c.heads, c.seqQ, 1});
    Tensor<float> lseGpu({c.batch, c.heads, c.seqQ});
    oGpu.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    lseGpu.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    const std::optional<float> softcap = c.softcap ? std::optional<float>(5.0f) : std::nullopt;
    CpuFpReferenceSdpa::forward(
        q,
        k,
        v,
        oCpu,
        SdpaFwdOptions<float>{}
            .setScale(0.125f)
            .setMask(c.bias ? static_cast<const TensorBase<float>*>(&bias) : nullptr)
            .setBand(c.leftBound, c.rightBound, c.topLeft)
            .setSeqLens(c.padding ? &seqLenQ : nullptr, c.padding ? &seqLenKv : nullptr)
            .setSink(c.sink ? &sink : nullptr)
            .setBlockMask(c.blockMask ? &blockMask : nullptr)
            .setAlibi(c.alibi)
            .setSoftcap(softcap)
            .setLse(&lseCpu));
    GpuFpReferenceSdpa::fprop(
        q,
        k,
        v,
        oGpu,
        GpuSdpaFwdOptions<float>{}
            .setScale(0.125f)
            .setMask(c.bias ? static_cast<TensorBase<float>*>(&bias) : nullptr)
            .setBand(c.leftBound, c.rightBound, c.topLeft)
            .setSeqLens(c.padding ? &seqLenQ : nullptr, c.padding ? &seqLenKv : nullptr)
            .setSink(c.sink ? &sink : nullptr)
            .setBlockMask(c.blockMask ? &blockMask : nullptr)
            .setAlibi(c.alibi)
            .setSoftcap(softcap)
            .setLse(&lseGpu));
    assertAllClose(oCpu, oGpu, 1e-4f);
    lseGpu.memory().hostData();
    for(int64_t b = 0; b < c.batch; ++b)
    {
        for(int64_t h = 0; h < c.heads; ++h)
        {
            for(int64_t s = 0; s < c.seqQ; ++s)
            {
                const float want = lseCpu.getHostValue(b, h, s, 0);
                const float got = lseGpu.getHostValue(b, h, s);
                if(std::isinf(want))
                {
                    ASSERT_EQ(got, want) << "b=" << b << " h=" << h << " s=" << s;
                }
                else
                {
                    ASSERT_NEAR(got, want, 1e-4f * std::max(1.0f, std::fabs(want)))
                        << "b=" << b << " h=" << h << " s=" << s;
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuSdpaFwdSweepFp32,
                         ::testing::Values(FwdSweepCase{"GqaCausalBrPaddingSink",
                                                        2,
                                                        8,
                                                        2,
                                                        300,
                                                        700,
                                                        64,
                                                        64,
                                                        -1,
                                                        0,
                                                        false,
                                                        false,
                                                        true,
                                                        true,
                                                        false,
                                                        false,
                                                        false},
                                           FwdSweepCase{"BiasWindowBlockMask",
                                                        1,
                                                        4,
                                                        4,
                                                        260,
                                                        1100,
                                                        64,
                                                        32,
                                                        200,
                                                        0,
                                                        false,
                                                        true,
                                                        false,
                                                        false,
                                                        true,
                                                        false,
                                                        false},
                                           FwdSweepCase{"AlibiCausalTlSoftcap",
                                                        1,
                                                        6,
                                                        3,
                                                        384,
                                                        512,
                                                        128,
                                                        128,
                                                        -1,
                                                        0,
                                                        true,
                                                        false,
                                                        false,
                                                        false,
                                                        false,
                                                        true,
                                                        true},
                                           FwdSweepCase{"EverythingButAlibi",
                                                        2,
                                                        4,
                                                        1,
                                                        270,
                                                        1040,
                                                        32,
                                                        48,
                                                        300,
                                                        0,
                                                        false,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        false,
                                                        true}),
                         fwdSweepName);
