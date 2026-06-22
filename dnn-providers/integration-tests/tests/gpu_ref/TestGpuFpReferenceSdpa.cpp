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

#include <hipdnn_gpu_ref/GpuFpReferenceSdpa.hpp>

#include <cmath>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_gpu_ref;

using gpu_conv_ref_test::assertAllClose;

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
// float compute/mask type shared by both reference implementations.
template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename ODataType,
          typename ComputeDataType = float>
void compareGpuVsCpuSdpaFwd(Tensor<QDataType>& q,
                            Tensor<KDataType>& k,
                            Tensor<VDataType>& v,
                            Tensor<ODataType>& oCpu,
                            Tensor<ODataType>& oGpu,
                            float tolerance,
                            std::optional<float> attnScaleValue = std::nullopt,
                            Tensor<float>* attnMask = nullptr,
                            int64_t leftBound = -1,
                            int64_t rightBound = -1,
                            bool topLeftAlignment = true)
{
    q.fillWithRandomValues(static_cast<QDataType>(-1.0f), static_cast<QDataType>(1.0f), SEED_Q);
    k.fillWithRandomValues(static_cast<KDataType>(-1.0f), static_cast<KDataType>(1.0f), SEED_K);
    v.fillWithRandomValues(static_cast<VDataType>(-1.0f), static_cast<VDataType>(1.0f), SEED_V);

    CpuFpReferenceSdpa::forward<QDataType, KDataType, VDataType, ODataType, ComputeDataType>(
        q,
        k,
        v,
        oCpu,
        attnScaleValue,
        static_cast<const TensorBase<ComputeDataType>*>(attnMask),
        leftBound,
        rightBound,
        topLeftAlignment);

    GpuFpReferenceSdpa::fprop<QDataType, KDataType, VDataType, ODataType, ComputeDataType>(
        q,
        k,
        v,
        oGpu,
        attnScaleValue,
        static_cast<TensorBase<ComputeDataType>*>(attnMask),
        leftBound,
        rightBound,
        topLeftAlignment);

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
    else if constexpr(std::is_same_v<T, half> || std::is_same_v<T, bfloat16>)
    {
        return 1e-2f;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

// Same contract as compareGpuVsCpuSdpaFwd, but also requests the optional
// log-sum-exp [B, H, Sq] output from both references and compares it. LSE is
// always float. Use ONLY for configs with no fully-masked rows: a fully-masked
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
    q.fillWithRandomValues(static_cast<QDataType>(-1.0f), static_cast<QDataType>(1.0f), SEED_Q);
    k.fillWithRandomValues(static_cast<KDataType>(-1.0f), static_cast<KDataType>(1.0f), SEED_K);
    v.fillWithRandomValues(static_cast<VDataType>(-1.0f), static_cast<VDataType>(1.0f), SEED_V);

    CpuFpReferenceSdpa::forward<QDataType, KDataType, VDataType, ODataType, ComputeDataType>(
        q,
        k,
        v,
        oCpu,
        attnScaleValue,
        /*attnMask=*/nullptr,
        leftBound,
        rightBound,
        topLeftAlignment,
        &lseCpu);

    GpuFpReferenceSdpa::fprop<QDataType, KDataType, VDataType, ODataType, ComputeDataType>(
        q,
        k,
        v,
        oGpu,
        attnScaleValue,
        /*attnMask=*/nullptr,
        leftBound,
        rightBound,
        topLeftAlignment,
        &lseGpu);

    assertAllClose(oCpu, oGpu, tolerance);
    // LSE is always FP32 regardless of the output dtype, so it uses the float
    // tolerance rather than the (possibly bf16/half) output tolerance.
    assertAllClose(lseCpu, lseGpu, gpuRefFwdTolerance<float>());
}

} // namespace

// ============================================================================
// Plain attention (no mask, no window) — MHA across all dtypes.
// ============================================================================

template <typename T>
class TestGpuSdpaFwdPlain : public ::testing::Test
{
};

using SdpaFwdTypes = ::testing::Types<float, half, bfloat16>;
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
// Causal bottom-right with seqKv < seqQ.
// diagonal offset = seqKv - seqQ < 0, exercising the max(..., 0) clamp on the
// start of the masked region for early query rows.
// ============================================================================

TYPED_TEST(TestGpuSdpaFwdPlain, CausalBottomRightSmallerSkv)
{
    SKIP_IF_NO_DEVICES();
    using T = TypeParam;

    // Sq=8, Skv=4 -> offset = -4 -> early query rows are fully masked.
    Tensor<T> q({1, 2, 8, 16});
    Tensor<T> k({1, 2, 4, 16});
    Tensor<T> v({1, 2, 4, 16});
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
                                       /*topLeftAlignment=*/false);
}

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

    mask.fillWithRandomValues(-2.0f, 2.0f, SEED_MASK);

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

    mask.fillWithRandomValues(-2.0f, 2.0f, SEED_MASK);

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

    mask.fillWithRandomValues(-2.0f, 2.0f, SEED_MASK);

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
// unset, so both references fall back to 1/sqrt(D); this passes an explicit value
// to exercise the provided-scale path on both sides.
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

    mask.fillWithRandomValues(-2.0f, 2.0f, SEED_MASK);

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

    mask.fillWithRandomValues(-2.0f, 2.0f, SEED_MASK);

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

    mask.fillWithRandomValues(-2.0f, 2.0f, SEED_MASK);

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

    Tensor<half> q({2, 4, 8, 16});
    Tensor<half> k({2, 4, 8, 16});
    Tensor<half> v({2, 4, 8, 16});
    Tensor<float> oCpu({2, 4, 8, 16});
    Tensor<float> oGpu({2, 4, 8, 16});

    compareGpuVsCpuSdpaFwd<half, half, half, float>(
        q, k, v, oCpu, oGpu, gpuRefFwdTolerance<float>());
}

TEST(TestGpuSdpaFwdMixedPrecision, Bfloat16InputsFloatOutput)
{
    SKIP_IF_NO_DEVICES();

    Tensor<bfloat16> q({2, 4, 8, 16});
    Tensor<bfloat16> k({2, 4, 8, 16});
    Tensor<bfloat16> v({2, 4, 8, 16});
    Tensor<float> oCpu({2, 4, 8, 16});
    Tensor<float> oGpu({2, 4, 8, 16});

    compareGpuVsCpuSdpaFwd<bfloat16, bfloat16, bfloat16, float>(
        q, k, v, oCpu, oGpu, gpuRefFwdTolerance<float>());
}

// ============================================================================
// LSE (log-sum-exp) output. The GPU reference exposes the optional [B, H, Sq]
// LSE that CpuFpReferenceSdpa produces (LSE = maxVal + log(sumExp)); these
// compare it against the CPU oracle. Configs are chosen so no query row is fully
// masked (LSE there is -inf, covered separately below).
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
    Tensor<float> lseCpu({2, 4, 8}); // [B, H, Sq]
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
    Tensor<float> lseCpu({1, 2, 8});
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
    Tensor<float> lseCpu({batch, numHeads, seqQ});
    Tensor<float> lseGpu({batch, numHeads, seqQ});

    q.fillWithRandomValues(static_cast<T>(-1.0f), static_cast<T>(1.0f), SEED_Q);
    k.fillWithRandomValues(static_cast<T>(-1.0f), static_cast<T>(1.0f), SEED_K);
    v.fillWithRandomValues(static_cast<T>(-1.0f), static_cast<T>(1.0f), SEED_V);

    CpuFpReferenceSdpa::forward<T, T, T, T, float>(q,
                                                   k,
                                                   v,
                                                   oCpu,
                                                   std::nullopt,
                                                   /*attnMask=*/nullptr,
                                                   /*leftBound=*/-1,
                                                   /*rightBound=*/0,
                                                   /*topLeftAlignment=*/false,
                                                   &lseCpu);
    GpuFpReferenceSdpa::fprop<T, T, T, T, float>(q,
                                                 k,
                                                 v,
                                                 oGpu,
                                                 std::nullopt,
                                                 /*attnMask=*/nullptr,
                                                 /*leftBound=*/-1,
                                                 /*rightBound=*/0,
                                                 /*topLeftAlignment=*/false,
                                                 &lseGpu);

    // Non-const TensorView triggers the device->host sync that getHostValue
    // (const) cannot; the GPU wrote lse on-device and only marked it modified.
    TensorView<float> lseCpuView(lseCpu);
    TensorView<float> lseGpuView(lseGpu);

    for(int64_t h = 0; h < numHeads; ++h)
    {
        for(int64_t sq = 0; sq < seqQ; ++sq)
        {
            const float cpuVal = lseCpuView.getHostValue(std::vector<int64_t>{0, h, sq});
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

    mask.fillWithRandomValues(-2.0f, 2.0f, SEED_MASK);

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

    mask.fillWithRandomValues(-2.0f, 2.0f, SEED_MASK);

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

    mask.fillWithRandomValues(-2.0f, 2.0f, SEED_MASK);

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

    mask.fillWithRandomValues(-2.0f, 2.0f, SEED_MASK);

    compareGpuVsCpuSdpaFwd<T, T, T, T>(
        q, k, v, oCpu, oGpu, gpuRefFwdTolerance<T>(), std::nullopt, /*attnMask=*/&mask);
}

// --- Mixed precision (16-bit in, float out) combined with modes ---

TEST(TestGpuSdpaFwdMixedPrecision, HalfInputsFloatOutputCausal)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> q({1, 2, 8, 16});
    Tensor<half> k({1, 2, 8, 16});
    Tensor<half> v({1, 2, 8, 16});
    Tensor<float> oCpu({1, 2, 8, 16});
    Tensor<float> oGpu({1, 2, 8, 16});

    compareGpuVsCpuSdpaFwd<half, half, half, float>(q,
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

    Tensor<bfloat16> q({1, 2, 8, 16});
    Tensor<bfloat16> k({1, 2, 8, 16});
    Tensor<bfloat16> v({1, 2, 8, 16});
    Tensor<float> oCpu({1, 2, 8, 16});
    Tensor<float> oGpu({1, 2, 8, 16});
    Tensor<float> mask({1, 2, 8, 8});

    mask.fillWithRandomValues(-2.0f, 2.0f, SEED_MASK);

    compareGpuVsCpuSdpaFwd<bfloat16, bfloat16, bfloat16, float>(
        q, k, v, oCpu, oGpu, gpuRefFwdTolerance<float>(), std::nullopt, /*attnMask=*/&mask);
}

TEST(TestGpuSdpaFwdMixedPrecision, HalfInputsFloatOutputGqa)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> q({1, 8, 8, 16});
    Tensor<half> k({1, 2, 8, 16});
    Tensor<half> v({1, 2, 8, 16});
    Tensor<float> oCpu({1, 8, 8, 16});
    Tensor<float> oGpu({1, 8, 8, 16});

    compareGpuVsCpuSdpaFwd<half, half, half, float>(
        q, k, v, oCpu, oGpu, gpuRefFwdTolerance<float>());
}
