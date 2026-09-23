// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// ============================================================================
// GPU-vs-CPU reference correctness gate for the RAGGED (RFC-0014: packed
// [B,H,S,D] + ragged_offset) SDPA forward GPU reference.
//
// The GPU reference runs on plain device tensors + explicit ragged_offset aux
// (GpuFpReferenceSdpaRagged::fpropRagged). The CPU mirror consumes the same host
// data through RFC-0014 ragged tensors (ShallowRaggedTensor over the identical
// packed buffers + per-primary ragged_offset aux), then the packed outputs are
// compared element-for-element. The GPU reference runs in the default FLOAT
// probability mode so it matches the fp32 CPU oracle (the bf16 P-storage mode is
// a provider-divergence concern tested elsewhere). The CPU mirror itself is
// validated against the dense CpuFpReferenceSdpa in the test_sdk suite
// (TestCpuFpReferenceSdpaRagged).
// ============================================================================

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpaRagged.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn-gpu-ref/GpuFpReferenceSdpaRagged.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_gpu_ref;

namespace
{

// Deterministic per-tensor seeds so every run uses identical inputs.
constexpr unsigned int SEED_Q = 42;
constexpr unsigned int SEED_K = 43;
constexpr unsigned int SEED_V = 44;

// dtype -> GPU-reference forward tolerance (mirrors the dense SDPA suite): the GPU
// reference enables FMA contraction, so float carries ~40x margin at 2e-5 and
// bf16/half get ample slack at 1e-2.
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

int64_t sum(const std::vector<int64_t>& v)
{
    return std::accumulate(v.begin(), v.end(), int64_t{0});
}

int64_t maxOf(const std::vector<int64_t>& v)
{
    return *std::max_element(v.begin(), v.end());
}

// ragged_offset device aux [B+1,1,1,1] INT32 = cumTokens * seqStride (elements) for the GPU path.
Tensor<int32_t> makeRaggedOffset(const std::vector<int64_t>& cum, int64_t seqStride)
{
    Tensor<int32_t> off({static_cast<int64_t>(cum.size()), 1, 1, 1});
    auto* p = off.memory().hostData();
    for(size_t i = 0; i < cum.size(); ++i)
    {
        p[i] = static_cast<int32_t>(cum[i] * seqStride);
    }
    off.memory().markHostModified();
    return off;
}

// Wrap a borrowed packed host buffer as an RFC-0014 ragged tensor ([B,H,S,D], seqAxis=2, BSHD).
template <typename T>
ShallowRaggedTensor<T> wrapRagged(T* buf,
                                  const std::vector<int64_t>& dims,
                                  int64_t seqStride,
                                  const std::vector<int64_t>& cum)
{
    return ShallowRaggedTensor<T>(
        buf, dims, bshd(dims), SEQ_AXIS, makeRaggedOffsetAux(cum, seqStride));
}

// Fill the front (packed) `count` elements of a padded tensor's buffer with random values.
template <typename T>
void fillPackedRandom(Tensor<T>& t, int64_t count, float lo, float hi, unsigned int seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    auto* p = t.memory().hostData();
    for(int64_t i = 0; i < count; ++i)
    {
        p[i] = static_cast<T>(dist(gen));
    }
    t.memory().markHostModified();
}

// Compare the packed output region: the GPU output's packed front (device->host sync) against the
// CPU mirror's packed backing. Both use identical global-token BSHD packing, so index i aligns.
template <typename T>
void compareRaggedPacked(Tensor<T>& oGpu, const std::vector<T>& oCpuBack, float tolerance)
{
    const auto* g = oGpu.memory().hostData();
    for(size_t i = 0; i < oCpuBack.size(); ++i)
    {
        EXPECT_NEAR(static_cast<float>(g[i]), static_cast<float>(oCpuBack[i]), tolerance)
            << "packed output mismatch at element " << i;
    }
}

// Core check: build packed rank-4 Q/K/V, run the ragged GPU reference on device tensors and the
// ragged CPU mirror on ShallowRaggedTensors over the same host buffers, and compare the packed
// output.
template <typename T, typename ComputeType = float>
void checkRagged(const std::vector<int64_t>& seqQ,
                 const std::vector<int64_t>& seqKv,
                 int64_t numHeads,
                 int64_t numHeadsK,
                 int64_t numHeadsV,
                 int64_t headDim,
                 int64_t headDimV,
                 int64_t leftBound = -1,
                 int64_t rightBound = -1,
                 bool topLeftAlignment = true,
                 std::optional<float> scale = std::nullopt)
{
    ASSERT_EQ(seqQ.size(), seqKv.size());
    const auto batch = static_cast<int64_t>(seqQ.size());
    const auto sMaxQ = maxOf(seqQ);
    const auto sMaxKv = maxOf(seqKv);
    const auto totalQ = sum(seqQ);
    const auto totalKv = sum(seqKv);
    const auto cumQ = cumTokens(seqQ);
    const auto cumKv = cumTokens(seqKv);

    const std::vector<int64_t> qDims = {batch, numHeads, sMaxQ, headDim};
    const std::vector<int64_t> kDims = {batch, numHeadsK, sMaxKv, headDim};
    const std::vector<int64_t> vDims = {batch, numHeadsV, sMaxKv, headDimV};
    const std::vector<int64_t> oDims = {batch, numHeads, sMaxQ, headDimV};

    Tensor<T> q(qDims, bshd(qDims));
    Tensor<T> k(kDims, bshd(kDims));
    Tensor<T> v(vDims, bshd(vDims));
    Tensor<T> oGpu(oDims, bshd(oDims));
    fillPackedRandom(q, totalQ * numHeads * headDim, -1.0f, 1.0f, SEED_Q);
    fillPackedRandom(k, totalKv * numHeadsK * headDim, -1.0f, 1.0f, SEED_K);
    fillPackedRandom(v, totalKv * numHeadsV * headDimV, -1.0f, 1.0f, SEED_V);

    // CPU mirror first (reads pristine host buffers), into a packed backing.
    std::vector<T> oCpuBack(static_cast<size_t>(totalQ * numHeads * headDimV),
                            static_cast<T>(0.0f));
    {
        auto qR = wrapRagged(q.memory().hostData(), qDims, numHeads * headDim, cumQ);
        auto kR = wrapRagged(k.memory().hostData(), kDims, numHeadsK * headDim, cumKv);
        auto vR = wrapRagged(v.memory().hostData(), vDims, numHeadsV * headDimV, cumKv);
        auto oR = wrapRagged(oCpuBack.data(), oDims, numHeads * headDimV, cumQ);
        CpuFpReferenceSdpaRagged::forward<T, T, T, T, ComputeType>(
            qR, kR, vR, oR, scale, leftBound, rightBound, topLeftAlignment);
    }

    // GPU reference on device tensors.
    auto offQ = makeRaggedOffset(cumQ, numHeads * headDim);
    auto offKv = makeRaggedOffset(cumKv, numHeadsK * headDim);
    GpuFpReferenceSdpaRagged::fpropRagged<T, T, T, T, ComputeType>(
        q, k, v, oGpu, offQ, offKv, scale, leftBound, rightBound, topLeftAlignment);

    compareRaggedPacked(oGpu, oCpuBack, gpuRefFwdTolerance<T>());
}

} // namespace

// --- Plain ragged MHA (differing per-batch lengths), self-attention ---

TEST(TestGpuSdpaRaggedFwdFp32, RaggedBasicMha)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<float>({3, 5, 1}, {3, 5, 1}, 4, 4, 4, 16, 16);
}

TEST(TestGpuSdpaRaggedFwdBfp16, RaggedBasicMha)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<bfloat16>({3, 5, 1}, {3, 5, 1}, 4, 4, 4, 16, 16);
}

// --- Cross-attention: per-batch Q and KV lengths differ ---

TEST(TestGpuSdpaRaggedFwdFp32, RaggedCrossAttention)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<float>({2, 4, 3}, {5, 1, 6}, 2, 2, 2, 16, 16);
}

// --- Per-batch causal (top-left) and bottom-right, ragged lengths ---

TEST(TestGpuSdpaRaggedFwdFp32, RaggedCausalTopLeft)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<float>({4, 7}, {4, 7}, 2, 2, 2, 16, 16, -1, 0, true);
}

TEST(TestGpuSdpaRaggedFwdBfp16, RaggedCausalBottomRight)
{
    SKIP_IF_NO_DEVICES();
    // Cross-attention causal with bottom-right alignment exercises the per-batch windowOffset.
    checkRagged<bfloat16>({3, 5}, {6, 8}, 2, 2, 2, 16, 16, -1, 0, false);
}

// --- Per-batch sliding window (both bounds) ---

TEST(TestGpuSdpaRaggedFwdFp32, RaggedSlidingWindow)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<float>({8, 6}, {8, 6}, 2, 2, 2, 16, 16, 2, 2, true);
}

// --- GQA / MQA over ragged batches ---

TEST(TestGpuSdpaRaggedFwdFp32, RaggedGqa)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<float>({5, 3}, {5, 3}, 8, 2, 2, 16, 16);
}

TEST(TestGpuSdpaRaggedFwdBfp16, RaggedMqa)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<bfloat16>({4, 6}, {4, 6}, 8, 1, 1, 16, 16);
}

// --- Edge case: a zero-length-KV batch produces exactly-zero output (both refs agree) ---

TEST(TestGpuSdpaRaggedFwdFp32, ZeroLengthKvBatch)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<float>({3, 2, 4}, {3, 0, 4}, 2, 2, 2, 16, 16);
}

// --- Head-dim coverage at the ticket's shapes (hdim_q in {128,192}, hdim_v=128) ---

TEST(TestGpuSdpaRaggedFwdBfp16, RaggedHeadDim128)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<bfloat16>({4, 6}, {4, 6}, 2, 2, 2, 128, 128);
}

TEST(TestGpuSdpaRaggedFwdFp32, RaggedHeadDim192xV128)
{
    SKIP_IF_NO_DEVICES();
    // hdim_q = 192, hdim_v = 128 (asymmetric head dims, as on the ASM v3 path).
    checkRagged<float>({3, 5}, {3, 5}, 2, 2, 2, 192, 128);
}

// --- Explicit LSE output: GPU vs CPU ragged mirror, compared over the packed region ---

TEST(TestGpuSdpaRaggedFwdFp32, RaggedLseOutput)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> seqQ = {3, 5};
    const std::vector<int64_t> seqKv = {3, 5};
    const int64_t numHeads = 2;
    const int64_t headDim = 16;
    const auto batch = static_cast<int64_t>(seqQ.size());
    const auto sMax = maxOf(seqQ);
    const auto totalQ = sum(seqQ);
    const auto cumQ = cumTokens(seqQ);
    const auto cumKv = cumTokens(seqKv);

    const std::vector<int64_t> dims = {batch, numHeads, sMax, headDim};
    const std::vector<int64_t> lseDims = {batch, numHeads, sMax, 1};

    Tensor<float> q(dims, bshd(dims));
    Tensor<float> k(dims, bshd(dims));
    Tensor<float> v(dims, bshd(dims));
    Tensor<float> oGpu(dims, bshd(dims));
    Tensor<float> lseGpu(lseDims, bshd(lseDims));
    fillPackedRandom(q, totalQ * numHeads * headDim, -1.0f, 1.0f, SEED_Q);
    fillPackedRandom(k, totalQ * numHeads * headDim, -1.0f, 1.0f, SEED_K);
    fillPackedRandom(v, totalQ * numHeads * headDim, -1.0f, 1.0f, SEED_V);

    std::vector<float> oCpuBack(static_cast<size_t>(totalQ * numHeads * headDim), 0.0f);
    std::vector<float> lseCpuBack(static_cast<size_t>(totalQ * numHeads), 0.0f);
    {
        auto qR = wrapRagged(q.memory().hostData(), dims, numHeads * headDim, cumQ);
        auto kR = wrapRagged(k.memory().hostData(), dims, numHeads * headDim, cumQ);
        auto vR = wrapRagged(v.memory().hostData(), dims, numHeads * headDim, cumQ);
        auto oR = wrapRagged(oCpuBack.data(), dims, numHeads * headDim, cumQ);
        auto lseR = wrapRagged(lseCpuBack.data(), lseDims, numHeads, cumQ);
        CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
            qR, kR, vR, oR, std::nullopt, -1, -1, true, &lseR);
    }

    auto offQ = makeRaggedOffset(cumQ, numHeads * headDim);
    auto offKv = makeRaggedOffset(cumKv, numHeads * headDim);
    GpuFpReferenceSdpaRagged::fpropRagged<float, float, float, float, float>(
        q, k, v, oGpu, offQ, offKv, std::nullopt, -1, -1, true, &lseGpu);

    const float tolerance = gpuRefFwdTolerance<float>();
    compareRaggedPacked(oGpu, oCpuBack, tolerance);

    const auto* lg = lseGpu.memory().hostData();
    for(size_t i = 0; i < lseCpuBack.size(); ++i)
    {
        EXPECT_NEAR(lg[i], lseCpuBack[i], tolerance) << "LSE mismatch at element " << i;
    }
}

namespace
{

Tensor<float> makeScalarDescale(float value)
{
    Tensor<float> d({1});
    d.memory().hostData()[0] = value;
    d.memory().markHostModified();
    return d;
}

// Per-KV-head descale [B, heads, 1, 1] with a distinct value per (b, head).
Tensor<float> makePerHeadDescale(int64_t batch, int64_t heads, float base)
{
    Tensor<float> d({batch, heads, 1, 1});
    auto* p = d.memory().hostData();
    for(int64_t i = 0; i < batch * heads; ++i)
    {
        p[i] = base + 0.1f * static_cast<float>(i);
    }
    d.memory().markHostModified();
    return d;
}

// fp8 core: GPU vs CPU ragged mirror on identical packed fp8 inputs, bf16 output, with descale.
void checkRaggedFp8(const std::vector<int64_t>& seqQ,
                    const std::vector<int64_t>& seqKv,
                    int64_t numHeads,
                    int64_t numHeadsKv,
                    int64_t headDim,
                    Tensor<float>& descaleQ,
                    Tensor<float>& descaleK,
                    Tensor<float>& descaleV,
                    int64_t leftBound,
                    int64_t rightBound,
                    bool topLeftAlignment)
{
    const auto batch = static_cast<int64_t>(seqQ.size());
    const auto sMaxQ = maxOf(seqQ);
    const auto sMaxKv = maxOf(seqKv);
    const auto totalQ = sum(seqQ);
    const auto totalKv = sum(seqKv);
    const auto cumQ = cumTokens(seqQ);
    const auto cumKv = cumTokens(seqKv);

    const std::vector<int64_t> qDims = {batch, numHeads, sMaxQ, headDim};
    const std::vector<int64_t> kvDims = {batch, numHeadsKv, sMaxKv, headDim};
    const std::vector<int64_t> oDims = {batch, numHeads, sMaxQ, headDim};

    Tensor<fp8_e4m3> q(qDims, bshd(qDims));
    Tensor<fp8_e4m3> k(kvDims, bshd(kvDims));
    Tensor<fp8_e4m3> v(kvDims, bshd(kvDims));
    Tensor<bfloat16> oGpu(oDims, bshd(oDims));
    fillPackedRandom(q, totalQ * numHeads * headDim, -1.0f, 1.0f, SEED_Q);
    fillPackedRandom(k, totalKv * numHeadsKv * headDim, -1.0f, 1.0f, SEED_K);
    fillPackedRandom(v, totalKv * numHeadsKv * headDim, -1.0f, 1.0f, SEED_V);

    std::vector<bfloat16> oCpuBack(static_cast<size_t>(totalQ * numHeads * headDim),
                                   bfloat16(0.0f));
    {
        auto qR = wrapRagged(q.memory().hostData(), qDims, numHeads * headDim, cumQ);
        auto kR = wrapRagged(k.memory().hostData(), kvDims, numHeadsKv * headDim, cumKv);
        auto vR = wrapRagged(v.memory().hostData(), kvDims, numHeadsKv * headDim, cumKv);
        auto oR = wrapRagged(oCpuBack.data(), oDims, numHeads * headDim, cumQ);
        CpuFpReferenceSdpaRagged::forward<fp8_e4m3, fp8_e4m3, fp8_e4m3, bfloat16, float>(
            qR,
            kR,
            vR,
            oR,
            std::nullopt,
            leftBound,
            rightBound,
            topLeftAlignment,
            nullptr,
            &descaleQ,
            &descaleK,
            &descaleV);
    }

    auto offQ = makeRaggedOffset(cumQ, numHeads * headDim);
    auto offKv = makeRaggedOffset(cumKv, numHeadsKv * headDim);
    GpuFpReferenceSdpaRagged::fpropRagged<fp8_e4m3, fp8_e4m3, fp8_e4m3, bfloat16, float>(
        q,
        k,
        v,
        oGpu,
        offQ,
        offKv,
        std::nullopt,
        leftBound,
        rightBound,
        topLeftAlignment,
        nullptr,
        SdpaSoftmaxProbabilityMode::FLOAT,
        &descaleQ,
        &descaleK,
        &descaleV);

    // bf16 output rounding + fp8 inputs -> looser tolerance than the float path.
    compareRaggedPacked(oGpu, oCpuBack, 2e-2f);
}

} // namespace

// --- fp8 (E4M3) ragged: GPU vs CPU ragged mirror. fp8 Q/K/V decode identically on host (data_sdk
// fp8_e4m3) and device (GpuRefFp8E4M3); the only divergence is the bf16 output + device-vs-host
// math, covered by the 2e-2 tolerance. ---

TEST(TestGpuSdpaRaggedFwdFp8, RaggedPerTensorDescale)
{
    SKIP_IF_NO_DEVICES();
    auto descaleQ = makeScalarDescale(0.5f);
    auto descaleK = makeScalarDescale(0.25f);
    auto descaleV = makeScalarDescale(2.0f);
    checkRaggedFp8({3, 5}, {3, 5}, 2, 2, 128, descaleQ, descaleK, descaleV, -1, -1, true);
}

TEST(TestGpuSdpaRaggedFwdFp8, RaggedCausalGqaPerKvHeadDescale)
{
    SKIP_IF_NO_DEVICES();
    const int64_t batch = 2;
    const int64_t numHeadsKv = 2; // GQA (numHeads = 4)
    auto descaleQ = makeScalarDescale(0.5f);
    auto descaleK = makePerHeadDescale(batch, numHeadsKv, 0.2f);
    auto descaleV = makePerHeadDescale(batch, numHeadsKv, 0.3f);
    checkRaggedFp8({4, 6}, {4, 6}, 4, numHeadsKv, 128, descaleQ, descaleK, descaleV, -1, 0, true);
}
