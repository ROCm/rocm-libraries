// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU-vs-CPU tests for the ragged SDPA forward GPU reference (RFC 0014: tensors addressed through
// ragged_offset aux, in any layout RFC 0014 §4.5.1 accepts; most cases here are packed BSHD, the
// layout-row cases at the end use the table's other layouts).
//
// The GPU reference takes device tensors and explicit ragged_offset aux. The CPU mirror reads the
// same host buffers as ShallowRaggedTensors, and the outputs are compared per live element.
// The GPU side uses the default FLOAT probability mode to match the fp32 CPU oracle. The CPU
// mirror is checked against the dense CpuFpReferenceSdpa in TestCpuFpReferenceSdpaRagged.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpaRagged.hpp>
#include <hipdnn_test_sdk/utilities/RaggedLayoutTestSupport.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn-gpu-ref/GpuFpReferenceSdpaRagged.hpp>
#include <hipdnn-gpu-ref/ShallowGpuTensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_gpu_ref;
using hipdnn_test_sdk::utilities::ragged_layout_test::makeInt32Aux;

namespace
{

// Fixed seeds so every run sees the same inputs.
constexpr unsigned int SEED_Q = 42;
constexpr unsigned int SEED_K = 43;
constexpr unsigned int SEED_V = 44;

// Same bounds as the dense SDPA GPU suite. Float keeps about 40x margin over FMA rounding.
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

// GPU ragged_offset aux: [B+1,1,1,1] int32 element offsets, cum[i] * seqStride.
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

// View a borrowed packed host buffer as an RFC-0014 ragged tensor ([B, H, S, D], SDPA_SEQ_AXIS).
template <typename T>
ShallowRaggedTensor<T> wrapRagged(T* buf,
                                  const std::vector<int64_t>& dims,
                                  int64_t seqStride,
                                  const std::vector<int64_t>& cum)
{
    return ShallowRaggedTensor<T>(
        buf, dims, raggedStrides(dims), SDPA_SEQ_AXIS, makeRaggedOffsetAux(cum, seqStride));
}

// Randomize only the packed prefix (first `count` elements) of a padded buffer.
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

// Compare the packed prefix of the GPU output (hostData() syncs it from device) with the CPU
// mirror. Both use the same BSHD token packing, so index i lines up.
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

// Run the GPU reference and the CPU mirror on the same packed Q/K/V and compare the outputs.
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

    const std::vector<int64_t> qDims = raggedDims(batch, sMaxQ, numHeads, headDim);
    const std::vector<int64_t> kDims = raggedDims(batch, sMaxKv, numHeadsK, headDim);
    const std::vector<int64_t> vDims = raggedDims(batch, sMaxKv, numHeadsV, headDimV);
    const std::vector<int64_t> oDims = raggedDims(batch, sMaxQ, numHeads, headDimV);

    Tensor<T> q(qDims, raggedStrides(qDims));
    Tensor<T> k(kDims, raggedStrides(kDims));
    Tensor<T> v(vDims, raggedStrides(vDims));
    Tensor<T> oGpu(oDims, raggedStrides(oDims));
    fillPackedRandom(q, totalQ * numHeads * headDim, -1.0f, 1.0f, SEED_Q);
    fillPackedRandom(k, totalKv * numHeadsK * headDim, -1.0f, 1.0f, SEED_K);
    fillPackedRandom(v, totalKv * numHeadsV * headDimV, -1.0f, 1.0f, SEED_V);

    // Run the CPU mirror first, while the host buffers are untouched.
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

    // Each primary needs its own offsets. When Hv*Dv != Hk*D or Dv != D, the V and O offsets
    // differ from K and Q for the same per-batch token counts.
    auto offQ = makeRaggedOffset(cumQ, numHeads * headDim);
    auto offK = makeRaggedOffset(cumKv, numHeadsK * headDim);
    auto offV = makeRaggedOffset(cumKv, numHeadsV * headDimV);
    auto offO = makeRaggedOffset(cumQ, numHeads * headDimV);
    GpuFpReferenceSdpaRagged::fpropRagged<T, T, T, T, ComputeType>(
        q, k, v, oGpu, offQ, offK, offV, offO, scale, leftBound, rightBound, topLeftAlignment);

    compareRaggedPacked(oGpu, oCpuBack, gpuRefFwdTolerance<T>());
}

} // namespace

// --- Ragged MHA self-attention ---

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

// --- Cross-attention with different Q and KV lengths ---

TEST(TestGpuSdpaRaggedFwdFp32, RaggedCrossAttention)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<float>({2, 4, 3}, {5, 1, 6}, 2, 2, 2, 16, 16);
}

// --- Causal masks, top-left and bottom-right ---

TEST(TestGpuSdpaRaggedFwdFp32, RaggedCausalTopLeft)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<float>({4, 7}, {4, 7}, 2, 2, 2, 16, 16, -1, 0, true);
}

TEST(TestGpuSdpaRaggedFwdBfp16, RaggedCausalBottomRight)
{
    SKIP_IF_NO_DEVICES();
    // Bottom-right causal with Sq != Skv tests the per-batch windowOffset.
    checkRagged<bfloat16>({3, 5}, {6, 8}, 2, 2, 2, 16, 16, -1, 0, false);
}

// --- Sliding window (both bounds) ---

TEST(TestGpuSdpaRaggedFwdFp32, RaggedSlidingWindow)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<float>({8, 6}, {8, 6}, 2, 2, 2, 16, 16, 2, 2, true);
}

// --- Explicit attention scale ---

// 0.125 instead of the default 1.0, checked against the CPU mirror.
TEST(TestGpuSdpaRaggedFwdFp32, RaggedExplicitAttnScale)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<float>({4, 6}, {3, 6}, 2, 2, 2, 16, 16, -1, -1, true, /*scale=*/0.125f);
}

// An absent scale means 1.0, as in cuDNN and GpuFpReferenceSdpa: the output must equal an
// explicit 1.0 bit for bit, and differ from the old 1/sqrt(D) default.
TEST(TestGpuSdpaRaggedFwdFp32, AbsentAttnScaleIsOne)
{
    SKIP_IF_NO_DEVICES();
    const auto dims = raggedDims(2, 5, 2, 16);
    const int64_t tokenWidth = int64_t{2} * 16; // H * D
    const auto cum = cumTokens({3, 5});
    Tensor<float> q(dims, raggedStrides(dims));
    Tensor<float> k(dims, raggedStrides(dims));
    Tensor<float> v(dims, raggedStrides(dims));
    fillPackedRandom(q, cum.back() * tokenWidth, -1.0f, 1.0f, SEED_Q);
    fillPackedRandom(k, cum.back() * tokenWidth, -1.0f, 1.0f, SEED_K);
    fillPackedRandom(v, cum.back() * tokenWidth, -1.0f, 1.0f, SEED_V);
    auto off = makeRaggedOffset(cum, tokenWidth);

    const auto run = [&](std::optional<float> scale) {
        Tensor<float> o(dims, raggedStrides(dims));
        o.fillWithValue(0.0f);
        GpuFpReferenceSdpaRagged::fpropRagged<float>(q, k, v, o, off, off, off, off, scale);
        const auto* p = o.memory().hostData();
        return std::vector<float>(p, p + cum.back() * tokenWidth);
    };
    const auto absent = run(std::nullopt);
    EXPECT_EQ(absent, run(1.0f));
    EXPECT_NE(absent, run(0.25f));
}

// --- GQA / MQA ---

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

// --- Zero-length batches. An empty KV batch gives zero output in both refs ---

TEST(TestGpuSdpaRaggedFwdFp32, ZeroLengthKvBatch)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<float>({3, 2, 4}, {3, 0, 4}, 2, 2, 2, 16, 16);
}

TEST(TestGpuSdpaRaggedFwdFp32, ZeroLengthQBatch)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<float>({3, 0, 4}, {3, 2, 4}, 2, 2, 2, 16, 16);
}

// --- Head dims from the ticket: hdim_q 128 or 192, hdim_v 128 ---

TEST(TestGpuSdpaRaggedFwdBfp16, RaggedHeadDim128)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<bfloat16>({4, 6}, {4, 6}, 2, 2, 2, 128, 128);
}

TEST(TestGpuSdpaRaggedFwdFp32, RaggedHeadDim192xV128)
{
    SKIP_IF_NO_DEVICES();
    // Asymmetric head dims, as on the ASM v3 path.
    checkRagged<float>({3, 5}, {3, 5}, 2, 2, 2, 192, 128);
}

// The ASM kernel's bf16 192/128 case, with GQA and bottom-right causal.
TEST(TestGpuSdpaRaggedFwdBfp16, RaggedHeadDim192xV128GqaCausal)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<bfloat16>({3, 5}, {6, 8}, 4, 2, 2, 192, 128, -1, 0, /*topLeftAlignment=*/false);
}

// The layout of the hd192 ragged golden bundles (quick/SdpaFwd/bshd/bf16/hd192_*_ragged): literal
// [B, H, S, D] dims with BSHD strides, D = 192 / Dv = 128, bottom-right causal. S_max is
// 64 rather than 256 to keep the CPU mirror fast, and the lengths differ per batch.
TEST(TestGpuSdpaRaggedFwdBfp16, RfcLayoutHd192BundleShape)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> seqLens = {64, 40, 17};
    const std::vector<int64_t> qkDims = {3, 2, 64, 192};
    const std::vector<int64_t> qkStrides = {24576, 192, 384, 1};
    const std::vector<int64_t> voDims = {3, 2, 64, 128};
    const std::vector<int64_t> voStrides = {16384, 128, 256, 1};
    const auto cum = cumTokens(seqLens);
    const auto total = cum.back();

    Tensor<bfloat16> q(qkDims, qkStrides);
    Tensor<bfloat16> k(qkDims, qkStrides);
    Tensor<bfloat16> v(voDims, voStrides);
    Tensor<bfloat16> oGpu(voDims, voStrides);
    fillPackedRandom(q, total * 384, -1.0f, 1.0f, SEED_Q);
    fillPackedRandom(k, total * 384, -1.0f, 1.0f, SEED_K);
    fillPackedRandom(v, total * 256, -1.0f, 1.0f, SEED_V);

    std::vector<bfloat16> oCpuBack(static_cast<size_t>(total * 256), bfloat16(0.0f));
    {
        const auto wrap = [&](bfloat16* buf,
                              const std::vector<int64_t>& dims,
                              const std::vector<int64_t>& strides,
                              int64_t tokenWidth) {
            return ShallowRaggedTensor<bfloat16>(
                buf, dims, strides, SDPA_SEQ_AXIS, makeRaggedOffsetAux(cum, tokenWidth));
        };
        auto qR = wrap(q.memory().hostData(), qkDims, qkStrides, 384);
        auto kR = wrap(k.memory().hostData(), qkDims, qkStrides, 384);
        auto vR = wrap(v.memory().hostData(), voDims, voStrides, 256);
        auto oR = wrap(oCpuBack.data(), voDims, voStrides, 256);
        CpuFpReferenceSdpaRagged::forward<bfloat16, bfloat16, bfloat16, bfloat16, float>(
            qR, kR, vR, oR, std::nullopt, -1, 0, /*topLeftAlignment=*/false);
    }

    auto offQk = makeRaggedOffset(cum, 384);
    auto offVo = makeRaggedOffset(cum, 256);
    GpuFpReferenceSdpaRagged::fpropRagged<bfloat16, bfloat16, bfloat16, bfloat16, float>(
        q, k, v, oGpu, offQk, offQk, offVo, offVo, std::nullopt, -1, 0, false);

    compareRaggedPacked(oGpu, oCpuBack, gpuRefFwdTolerance<bfloat16>());
}

// --- Ragged LSE output ---

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

    const std::vector<int64_t> dims = raggedDims(batch, sMax, numHeads, headDim);
    const std::vector<int64_t> lseDims = raggedDims(batch, sMax, numHeads, 1);

    Tensor<float> q(dims, raggedStrides(dims));
    Tensor<float> k(dims, raggedStrides(dims));
    Tensor<float> v(dims, raggedStrides(dims));
    Tensor<float> oGpu(dims, raggedStrides(dims));
    Tensor<float> lseGpu(lseDims, raggedStrides(lseDims));
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
    // Ragged LSE is [B,H,S,1] packed by token with seq stride H, so its offsets are cum * H.
    auto offLse = makeRaggedOffset(cumQ, numHeads);
    GpuFpReferenceSdpaRagged::fpropRagged<float, float, float, float, float>(
        q, k, v, oGpu, offQ, offKv, offKv, offQ, std::nullopt, -1, -1, true, &lseGpu, &offLse);

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

constexpr float LSE_SENTINEL = -99.0f;

// Dense LSE, the frontend's default stats layout: contiguous [B, H, Sq_max, 1], so batch b
// starts at b * Sq_max * H whatever the Q packing. Both LSE buffers start at a sentinel, so
// padding rows must stay untouched and a misaddressed write shows up as a mismatch.
// With zeroQk every score is 0, so a valid row's LSE is log(seqKv[b]).
void checkRaggedDenseLse(const std::vector<int64_t>& seqQ,
                         const std::vector<int64_t>& seqKv,
                         int64_t numHeads,
                         int64_t headDim,
                         bool zeroQk)
{
    const auto batch = static_cast<int64_t>(seqQ.size());
    const auto sMaxQ = maxOf(seqQ);
    const auto sMaxKv = maxOf(seqKv);
    const auto totalQ = sum(seqQ);
    const auto totalKv = sum(seqKv);
    const auto cumQ = cumTokens(seqQ);
    const auto cumKv = cumTokens(seqKv);

    const std::vector<int64_t> qDims = raggedDims(batch, sMaxQ, numHeads, headDim);
    const std::vector<int64_t> kvDims = raggedDims(batch, sMaxKv, numHeads, headDim);
    const std::vector<int64_t> lseDims = raggedDims(batch, sMaxQ, numHeads, 1);

    Tensor<float> q(qDims, raggedStrides(qDims));
    Tensor<float> k(kvDims, raggedStrides(kvDims));
    Tensor<float> v(kvDims, raggedStrides(kvDims));
    Tensor<float> oGpu(qDims, raggedStrides(qDims));
    if(zeroQk)
    {
        q.fillWithValue(0.0f);
        k.fillWithValue(0.0f);
    }
    else
    {
        fillPackedRandom(q, totalQ * numHeads * headDim, -1.0f, 1.0f, SEED_Q);
        fillPackedRandom(k, totalKv * numHeads * headDim, -1.0f, 1.0f, SEED_K);
    }
    fillPackedRandom(v, totalKv * numHeads * headDim, -1.0f, 1.0f, SEED_V);

    Tensor<float> lseCpu(lseDims);
    Tensor<float> lseGpu(lseDims);
    lseCpu.fillWithValue(LSE_SENTINEL);
    lseGpu.fillWithValue(LSE_SENTINEL);

    std::vector<float> oCpuBack(static_cast<size_t>(totalQ * numHeads * headDim), 0.0f);
    {
        auto qR = wrapRagged(q.memory().hostData(), qDims, numHeads * headDim, cumQ);
        auto kR = wrapRagged(k.memory().hostData(), kvDims, numHeads * headDim, cumKv);
        auto vR = wrapRagged(v.memory().hostData(), kvDims, numHeads * headDim, cumKv);
        auto oR = wrapRagged(oCpuBack.data(), qDims, numHeads * headDim, cumQ);
        CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
            qR, kR, vR, oR, std::nullopt, -1, -1, true, &lseCpu);
    }

    auto offQ = makeRaggedOffset(cumQ, numHeads * headDim);
    auto offKv = makeRaggedOffset(cumKv, numHeads * headDim);
    GpuFpReferenceSdpaRagged::fpropRagged<float, float, float, float, float>(
        q,
        k,
        v,
        oGpu,
        offQ,
        offKv,
        offKv,
        offQ,
        std::nullopt,
        -1,
        -1,
        true,
        &lseGpu,
        /*raggedOffsetLse=*/nullptr);

    const float tolerance = gpuRefFwdTolerance<float>();
    compareRaggedPacked(oGpu, oCpuBack, tolerance);

    for(int64_t b = 0; b < batch; ++b)
    {
        for(int64_t h = 0; h < numHeads; ++h)
        {
            for(int64_t s = 0; s < sMaxQ; ++s)
            {
                const float gpu = lseGpu(raggedIndex(b, s, h, 0));
                EXPECT_NEAR(gpu, lseCpu(raggedIndex(b, s, h, 0)), tolerance)
                    << "dense LSE mismatch at [" << b << ", " << s << ", " << h << ", 0]";
                if(s >= seqQ[static_cast<size_t>(b)])
                {
                    EXPECT_EQ(gpu, LSE_SENTINEL)
                        << "padding row written at [" << b << ", " << s << ", " << h << ", 0]";
                }
                else if(zeroQk)
                {
                    EXPECT_NEAR(
                        gpu, std::log(static_cast<float>(seqKv[static_cast<size_t>(b)])), tolerance)
                        << "zero-score LSE must be log(seqKv) at [" << b << ", " << s << ", " << h
                        << ", 0]";
                }
            }
        }
    }
}

} // namespace

// Reviewer repro: B=2, H=2, Sq=2, Q/K lengths {1, 2}, zero Q/K. stats[1,1,1,0] must be log(2).
TEST(TestGpuSdpaRaggedFwdFp32, RaggedDenseLseUnequalLengths)
{
    SKIP_IF_NO_DEVICES();
    checkRaggedDenseLse({1, 2}, {1, 2}, 2, 16, /*zeroQk=*/true);
}

TEST(TestGpuSdpaRaggedFwdFp32, RaggedDenseLseCrossAttention)
{
    SKIP_IF_NO_DEVICES();
    checkRaggedDenseLse({3, 5, 1}, {4, 2, 6}, 2, 16, /*zeroQk=*/false);
}

// An empty query batch has no LSE rows to write; its whole Sq_max block keeps the sentinel.
TEST(TestGpuSdpaRaggedFwdFp32, RaggedDenseLseZeroLengthQ)
{
    SKIP_IF_NO_DEVICES();
    checkRaggedDenseLse({2, 0, 3}, {3, 2, 1}, 2, 16, /*zeroQk=*/false);
}

// No query tokens at all: nothing runs and nothing is written.
TEST(TestGpuSdpaRaggedFwdFp32, AllQueriesEmpty)
{
    SKIP_IF_NO_DEVICES();
    const int64_t headDim = 16;
    const std::vector<int64_t> dims = raggedDims(2, 2, 1, headDim);
    const std::vector<int64_t> lseDims = raggedDims(2, 2, 1, 1);
    Tensor<float> q(dims, raggedStrides(dims));
    Tensor<float> k(dims, raggedStrides(dims));
    Tensor<float> v(dims, raggedStrides(dims));
    Tensor<float> o(dims, raggedStrides(dims));
    Tensor<float> lse(lseDims);
    q.fillWithValue(1.0f);
    k.fillWithValue(1.0f);
    v.fillWithValue(1.0f);
    o.fillWithValue(LSE_SENTINEL);
    lse.fillWithValue(LSE_SENTINEL);
    auto offQ = makeRaggedOffset({0, 0, 0}, headDim);
    auto offKv = makeRaggedOffset({0, 2, 3}, headDim);

    EXPECT_NO_THROW((GpuFpReferenceSdpaRagged::fpropRagged<float>(
        q, k, v, o, offQ, offKv, offKv, offQ, std::nullopt, -1, -1, true, &lse)));

    const auto* op = o.memory().hostData();
    for(size_t i = 0; i < o.elementCount(); ++i)
    {
        EXPECT_EQ(op[i], LSE_SENTINEL) << "output written at element " << i;
    }
    const auto* lp = lse.memory().hostData();
    for(size_t i = 0; i < lse.elementCount(); ++i)
    {
        EXPECT_EQ(lp[i], LSE_SENTINEL) << "LSE written at element " << i;
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

// fp8 Q/K/V with descales and bf16 output, GPU vs CPU mirror.
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

    const std::vector<int64_t> qDims = raggedDims(batch, sMaxQ, numHeads, headDim);
    const std::vector<int64_t> kvDims = raggedDims(batch, sMaxKv, numHeadsKv, headDim);
    const std::vector<int64_t> oDims = raggedDims(batch, sMaxQ, numHeads, headDim);

    Tensor<fp8_e4m3> q(qDims, raggedStrides(qDims));
    Tensor<fp8_e4m3> k(kvDims, raggedStrides(kvDims));
    Tensor<fp8_e4m3> v(kvDims, raggedStrides(kvDims));
    Tensor<bfloat16> oGpu(oDims, raggedStrides(oDims));
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
        offKv,
        offQ,
        std::nullopt,
        leftBound,
        rightBound,
        topLeftAlignment,
        nullptr,
        nullptr,
        SdpaSoftmaxProbabilityMode::FLOAT,
        &descaleQ,
        &descaleK,
        &descaleV);

    // fp8 inputs and bf16 output need a looser bound than fp32.
    compareRaggedPacked(oGpu, oCpuBack, 2e-2f);
}

} // namespace

// --- fp8 (E4M3). Host and device decode fp8 the same way, so the 2e-2 bound only covers bf16
// output rounding and device-vs-host math ---

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

// AITER's [B, H_kv] descale on Q, K and V under GQA. Distinct Q values per KV head catch a Q
// descale indexed by query head.
TEST(TestGpuSdpaRaggedFwdFp8, RaggedGqaPerKvHeadDescaleQkv)
{
    SKIP_IF_NO_DEVICES();
    const int64_t batch = 2;
    const int64_t numHeadsKv = 2; // GQA (numHeads = 4)
    auto descaleQ = makePerHeadDescale(batch, numHeadsKv, 0.4f);
    auto descaleK = makePerHeadDescale(batch, numHeadsKv, 0.2f);
    auto descaleV = makePerHeadDescale(batch, numHeadsKv, 0.3f);
    checkRaggedFp8({4, 6}, {5, 3}, 4, numHeadsKv, 128, descaleQ, descaleK, descaleV, -1, -1, true);
}

TEST(TestGpuSdpaRaggedFwdFp8, ThrowsOnPerQueryHeadQDescaleUnderGqa)
{
    SKIP_IF_NO_DEVICES();
    // Q descale is per KV head, so under GQA a [B, H_q, 1, 1] Q descale must be rejected.
    const std::vector<int64_t> qDims = raggedDims(1, 4, 4, 128);
    const std::vector<int64_t> kvDims = raggedDims(1, 4, 2, 128);
    Tensor<fp8_e4m3> q(qDims, raggedStrides(qDims));
    Tensor<fp8_e4m3> k(kvDims, raggedStrides(kvDims));
    Tensor<fp8_e4m3> v(kvDims, raggedStrides(kvDims));
    Tensor<bfloat16> o(qDims, raggedStrides(qDims));
    const auto cum = cumTokens({4});
    auto offQ = makeRaggedOffset(cum, int64_t{4} * 128);
    auto offKv = makeRaggedOffset(cum, int64_t{2} * 128);

    auto perQueryHead = makePerHeadDescale(1, 4, 0.5f);
    auto descaleK = makeScalarDescale(1.0f);
    auto descaleV = makeScalarDescale(1.0f);
    EXPECT_THROW((GpuFpReferenceSdpaRagged::fpropRagged<fp8_e4m3, fp8_e4m3, fp8_e4m3, bfloat16>(
                     q,
                     k,
                     v,
                     o,
                     offQ,
                     offKv,
                     offKv,
                     offQ,
                     std::nullopt,
                     -1,
                     -1,
                     true,
                     nullptr,
                     nullptr,
                     SdpaSoftmaxProbabilityMode::FLOAT,
                     &perQueryHead,
                     &descaleK,
                     &descaleV)),
                 std::invalid_argument);
}

// Checks the device fp8 decode table against the host fp8_e4m3 for all 256 byte values. With one
// key, the softmax weight is exactly 1, so the fp32 output is the decoded V. Only -0 can't be told
// apart, because the P@V sum starts at +0.
TEST(TestGpuSdpaRaggedFwdFp8, DecodesEveryFp8ValueExactly)
{
    SKIP_IF_NO_DEVICES();
    const int64_t headDim = 16;
    const int64_t headDimV = 256;
    const std::vector<int64_t> qkDims = raggedDims(1, 1, 1, headDim);
    const std::vector<int64_t> voDims = raggedDims(1, 1, 1, headDimV);
    Tensor<fp8_e4m3> q(qkDims, raggedStrides(qkDims));
    Tensor<fp8_e4m3> k(qkDims, raggedStrides(qkDims));
    Tensor<fp8_e4m3> v(voDims, raggedStrides(voDims));
    Tensor<float> o(voDims, raggedStrides(voDims));
    q.fillWithValue(fp8_e4m3::from_bits(0));
    k.fillWithValue(fp8_e4m3::from_bits(0));
    auto* vp = v.memory().hostData();
    for(int64_t i = 0; i < headDimV; ++i)
    {
        vp[i] = fp8_e4m3::from_bits(static_cast<uint8_t>(i));
    }
    v.memory().markHostModified();
    const auto cum = cumTokens({1});
    auto offQk = makeRaggedOffset(cum, headDim);
    auto offVo = makeRaggedOffset(cum, headDimV);

    GpuFpReferenceSdpaRagged::fpropRagged<fp8_e4m3, fp8_e4m3, fp8_e4m3, float>(
        q, k, v, o, offQk, offQk, offVo, offVo);

    const auto* op = o.memory().hostData();
    for(int64_t i = 0; i < headDimV; ++i)
    {
        const float expected = static_cast<float>(fp8_e4m3::from_bits(static_cast<uint8_t>(i)));
        const float got = op[i];
        if(std::isnan(expected))
        {
            EXPECT_TRUE(std::isnan(got)) << "byte " << i;
        }
        else
        {
            EXPECT_EQ(got, expected) << "byte " << i;
        }
    }
}

// --- Tensors sharing a packing must describe the same per-batch sequence lengths ---

namespace
{

// fp32 fpropRagged with B=2, H=1, D=16, S_max=2 and per-tensor lengths. A non-empty lseLens adds
// a ragged LSE. Returns true only if it threw std::invalid_argument.
bool throwsOnLengths(const std::vector<int64_t>& qLens,
                     const std::vector<int64_t>& kLens,
                     const std::vector<int64_t>& vLens,
                     const std::vector<int64_t>& oLens,
                     const std::vector<int64_t>& lseLens = {})
{
    const int64_t headDim = 16;
    const std::vector<int64_t> dims = raggedDims(2, 2, 1, headDim);
    const std::vector<int64_t> lseDims = raggedDims(2, 2, 1, 1);
    Tensor<float> q(dims, raggedStrides(dims));
    Tensor<float> k(dims, raggedStrides(dims));
    Tensor<float> v(dims, raggedStrides(dims));
    Tensor<float> o(dims, raggedStrides(dims));
    Tensor<float> lse(lseDims, raggedStrides(lseDims));
    q.fillWithValue(0.0f);
    k.fillWithValue(0.0f);
    v.fillWithValue(1.0f);
    auto offQ = makeRaggedOffset(cumTokens(qLens), headDim);
    auto offK = makeRaggedOffset(cumTokens(kLens), headDim);
    auto offV = makeRaggedOffset(cumTokens(vLens), headDim);
    auto offO = makeRaggedOffset(cumTokens(oLens), headDim);
    auto offLse = makeRaggedOffset(cumTokens(lseLens.empty() ? qLens : lseLens), 1);
    const bool withLse = !lseLens.empty();
    try
    {
        GpuFpReferenceSdpaRagged::fpropRagged<float>(q,
                                                     k,
                                                     v,
                                                     o,
                                                     offQ,
                                                     offK,
                                                     offV,
                                                     offO,
                                                     std::nullopt,
                                                     -1,
                                                     -1,
                                                     true,
                                                     withLse ? &lse : nullptr,
                                                     withLse ? &offLse : nullptr);
    }
    catch(const std::invalid_argument&)
    {
        return true;
    }
    return false;
}

} // namespace

// Reviewer repro (K lengths {2, 1}, V lengths {1, 2}, same S_max) plus the Q/O and Q/LSE pairs.
TEST(TestGpuSdpaRaggedFwdFp32, ThrowsOnSequenceLengthMismatch)
{
    SKIP_IF_NO_DEVICES();
    EXPECT_TRUE(throwsOnLengths({1, 1}, {2, 1}, {1, 2}, {1, 1})) << "K/V mismatch accepted";
    EXPECT_TRUE(throwsOnLengths({2, 1}, {2, 2}, {2, 2}, {1, 2})) << "Q/O mismatch accepted";
    EXPECT_TRUE(throwsOnLengths({2, 1}, {2, 2}, {2, 2}, {2, 1}, {1, 2}))
        << "Q/LSE mismatch accepted";
    // Consistent lengths with a ragged LSE run normally.
    EXPECT_FALSE(throwsOnLengths({2, 1}, {1, 2}, {1, 2}, {2, 1}, {2, 1}));
}

namespace
{

// fp32 fpropRagged with B=2, H=1, D=16, S_max=2 and K/V lengths {2, 2}. Q and O use the offsets
// qTokens * offsetUnit (offsetUnit 1 writes raw element offsets). lseSq > 0 adds a dense LSE
// [2, lseSq, 1, 1]. Returns true only if it threw std::invalid_argument.
bool throwsOnQTokens(const std::vector<int64_t>& qTokens,
                     int64_t lseSq = 0,
                     int64_t offsetUnit = 16)
{
    const int64_t headDim = 16;
    const std::vector<int64_t> dims = raggedDims(2, 2, 1, headDim);
    Tensor<float> q(dims, raggedStrides(dims));
    Tensor<float> k(dims, raggedStrides(dims));
    Tensor<float> v(dims, raggedStrides(dims));
    Tensor<float> o(dims, raggedStrides(dims));
    Tensor<float> lse(raggedDims(2, std::max<int64_t>(lseSq, 1), 1, 1));
    q.fillWithValue(0.0f);
    k.fillWithValue(0.0f);
    v.fillWithValue(1.0f);
    auto offQ = makeRaggedOffset(qTokens, offsetUnit);
    auto offKv = makeRaggedOffset({0, 2, 4}, headDim);
    try
    {
        GpuFpReferenceSdpaRagged::fpropRagged<float>(q,
                                                     k,
                                                     v,
                                                     o,
                                                     offQ,
                                                     offKv,
                                                     offKv,
                                                     offQ,
                                                     std::nullopt,
                                                     -1,
                                                     -1,
                                                     true,
                                                     lseSq > 0 ? &lse : nullptr);
    }
    catch(const std::invalid_argument&)
    {
        return true;
    }
    return false;
}

} // namespace

// Offset tables the CPU side rejects must be rejected here too.
TEST(TestGpuSdpaRaggedFwdFp32, ThrowsOnBadOffsetTable)
{
    SKIP_IF_NO_DEVICES();
    EXPECT_TRUE(throwsOnQTokens({0, 2, 1})) << "decreasing offsets accepted";
    EXPECT_TRUE(throwsOnQTokens({0, 17, 32}, /*lseSq=*/0, /*offsetUnit=*/1))
        << "offset that is not a whole token accepted";
    EXPECT_FALSE(throwsOnQTokens({0, 2, 3}));
}

// A dense LSE with a smaller Sq than Q would take rows from the next batch.
TEST(TestGpuSdpaRaggedFwdFp32, ThrowsOnLseShorterThanQ)
{
    SKIP_IF_NO_DEVICES();
    EXPECT_TRUE(throwsOnQTokens({0, 2, 3}, /*lseSq=*/1));
    EXPECT_FALSE(throwsOnQTokens({0, 2, 3}, /*lseSq=*/2));
}

// A contiguous [B, H, S, D] tensor (strides [256, 64, 16, 1]) is head-major with head stride
// S * D: a valid layout.
TEST(TestGpuSdpaRaggedFwdFp32, AcceptsHeadMajorLayout)
{
    SKIP_IF_NO_DEVICES();
    const std::vector<int64_t> dims = {1, 4, 4, 16}; // [B, H, S, D]
    const std::vector<int64_t> strides = {256, 64, 16, 1};
    Tensor<float> q(dims, strides);
    Tensor<float> k(dims, strides);
    Tensor<float> v(dims, strides);
    Tensor<float> o(dims, strides);
    q.fillWithValue(0.5f);
    k.fillWithValue(0.5f);
    v.fillWithValue(1.0f);
    auto off = makeRaggedOffset(cumTokens({4}), 16);
    EXPECT_NO_THROW((GpuFpReferenceSdpaRagged::fpropRagged<float, float, float, float, float>(
        q, k, v, o, off, off, off, off)));
}

// BHSD strides whose head stride (8) is smaller than the dense view's sequence span (rows 0-7
// at stride 2) alias rows of different heads; the container rule rejects them (RFC 0014 §4.5.1, L11).
TEST(TestGpuSdpaRaggedFwdFp32, ThrowsOnAliasingBhsdLayout)
{
    SKIP_IF_NO_DEVICES();
    const std::vector<int64_t> dims = {2, 2, 4, 2}; // [B, H, S, D]
    const std::vector<int64_t> strides = {16, 8, 2, 1};
    Tensor<float> q(dims, strides);
    Tensor<float> k(dims, strides);
    Tensor<float> v(dims, strides);
    Tensor<float> o(dims, strides);
    q.fillWithValue(0.5f);
    k.fillWithValue(0.5f);
    v.fillWithValue(1.0f);
    auto off = makeRaggedOffset(cumTokens({4, 4}), 2); // batch 1 starts at row 4
    EXPECT_THROW((GpuFpReferenceSdpaRagged::fpropRagged<float, float, float, float, float>(
                     q, k, v, o, off, off, off, off)),
                 std::invalid_argument);
}

namespace
{

// Runs fp32 fpropRagged on views with the given dims (token-major strides) and all-zero offset
// tables. Every batch is empty, so no tensor element is read and valid shapes launch nothing.
// The views are ShallowGpuTensors, as on the plan path: unlike Tensor they accept a zero dim,
// so the reference's own checks are what reject it. offsetRows overrides the tables' B + 1.
// Returns the std::invalid_argument message, or "" if nothing was thrown.
std::string shapeError(const std::vector<int64_t>& qDims,
                       const std::vector<int64_t>& kDims,
                       const std::vector<int64_t>& vDims,
                       const std::vector<int64_t>& oDims,
                       int64_t offsetRows = -1,
                       bool lseOffsetWithoutLse = false)
{
    const auto stridesOf = [](const std::vector<int64_t>& dims) {
        return dims.size() == 4 ? raggedStrides(dims) : generateStrides(dims);
    };
    Tensor<float> backing({1});
    void* mem = backing.memory().deviceData();
    hipdnn_gpu_ref::ShallowGpuTensor<float> q(mem, qDims, stridesOf(qDims));
    hipdnn_gpu_ref::ShallowGpuTensor<float> k(mem, kDims, stridesOf(kDims));
    hipdnn_gpu_ref::ShallowGpuTensor<float> v(mem, vDims, stridesOf(vDims));
    hipdnn_gpu_ref::ShallowGpuTensor<float> o(mem, oDims, stridesOf(oDims));
    Tensor<int32_t> off({offsetRows >= 0 ? offsetRows : qDims[0] + 1, 1, 1, 1});
    off.fillWithValue(0);
    try
    {
        GpuFpReferenceSdpaRagged::fpropRagged<float>(q,
                                                     k,
                                                     v,
                                                     o,
                                                     off,
                                                     off,
                                                     off,
                                                     off,
                                                     std::nullopt,
                                                     -1,
                                                     -1,
                                                     true,
                                                     nullptr,
                                                     lseOffsetWithoutLse ? &off : nullptr);
    }
    catch(const std::invalid_argument& e)
    {
        return e.what();
    }
    return "";
}

} // namespace

// Each shape the GPU reference must reject, one field wrong at a time, with the check that has to
// catch it. The all-valid shape runs without error.
TEST(TestGpuSdpaRaggedFwdFp32, ThrowsOnBadShapes)
{
    SKIP_IF_NO_DEVICES();
    const auto d = raggedDims(2, 4, 4, 16);
    const auto expectError = [](const std::string& error, const char* expected) {
        EXPECT_NE(error.find(expected), std::string::npos)
            << "expected \"" << expected << "\", got \"" << error << "\"";
    };

    EXPECT_EQ(shapeError(d, d, d, d), "");
    expectError(shapeError({2, 4, 64}, d, d, d), "rank-4 [B, H, S, D]");
    expectError(shapeError(d, d, d, d, /*offsetRows=*/2), "must have shape [B + 1, 1, 1, 1]");
    expectError(shapeError(raggedDims(2, 4, 4, 0), raggedDims(2, 4, 4, 0), d, d),
                "all dimensions must be positive");
    expectError(shapeError(d, raggedDims(1, 4, 4, 16), d, d), "batch dimension mismatch");
    expectError(shapeError(d, d, raggedDims(2, 3, 4, 16), d), "S_max");
    expectError(shapeError(d, raggedDims(2, 4, 4, 8), d, d), "Q head_dim != K head_dim");
    expectError(shapeError(d, raggedDims(2, 4, 3, 16), d, d), "must be divisible");
    expectError(shapeError(d, d, d, raggedDims(2, 4, 2, 16)), "output shape");
    expectError(shapeError(d, d, d, d, -1, /*lseOffsetWithoutLse=*/true),
                "raggedOffsetLse given without an lse tensor");
}

// --- Token offsets with ragged_offset_multiplier (AITER's cu_seqlens form) ---

namespace
{

// AITER binds cu_seqlens_q to Q and O and cu_seqlens_k to K and V: one token table per pair,
// each tensor scaling it by its own seq stride (H*D). GQA and D != Dv make every width
// different, and the ragged LSE scales the same Q table by H. The CPU mirror reads the same
// token tables through ShallowRaggedTensor's multiplier.
void checkTokenOffsets(const std::vector<int64_t>& seqQ,
                       const std::vector<int64_t>& seqKv,
                       int64_t numHeads,
                       int64_t numHeadsKv,
                       int64_t headDim,
                       int64_t headDimV)
{
    const auto batch = static_cast<int64_t>(seqQ.size());
    const auto cumQ = cumTokens(seqQ);
    const auto cumKv = cumTokens(seqKv);
    const auto totalQ = cumQ.back();
    const auto totalKv = cumKv.back();

    const auto qDims = raggedDims(batch, maxOf(seqQ), numHeads, headDim);
    const auto kDims = raggedDims(batch, maxOf(seqKv), numHeadsKv, headDim);
    const auto vDims = raggedDims(batch, maxOf(seqKv), numHeadsKv, headDimV);
    const auto oDims = raggedDims(batch, maxOf(seqQ), numHeads, headDimV);
    const auto lseDims = raggedDims(batch, maxOf(seqQ), numHeads, 1);

    Tensor<float> q(qDims, raggedStrides(qDims));
    Tensor<float> k(kDims, raggedStrides(kDims));
    Tensor<float> v(vDims, raggedStrides(vDims));
    Tensor<float> oGpu(oDims, raggedStrides(oDims));
    Tensor<float> lseGpu(lseDims, raggedStrides(lseDims));
    fillPackedRandom(q, totalQ * numHeads * headDim, -1.0f, 1.0f, SEED_Q);
    fillPackedRandom(k, totalKv * numHeadsKv * headDim, -1.0f, 1.0f, SEED_K);
    fillPackedRandom(v, totalKv * numHeadsKv * headDimV, -1.0f, 1.0f, SEED_V);

    RaggedOffsetMultipliers mult;
    mult.q = numHeads * headDim;
    mult.k = numHeadsKv * headDim;
    mult.v = numHeadsKv * headDimV;
    mult.o = numHeads * headDimV;
    mult.lse = numHeads;

    std::vector<float> oCpuBack(static_cast<size_t>(totalQ * mult.o), 0.0f);
    std::vector<float> lseCpuBack(static_cast<size_t>(totalQ * mult.lse), 0.0f);
    {
        auto qoTable = makeRaggedOffsetAux(cumQ, 1);
        auto kvTable = makeRaggedOffsetAux(cumKv, 1);
        const auto wrap = [](float* buf,
                             const std::vector<int64_t>& dims,
                             const std::shared_ptr<ITensor>& table,
                             int64_t multiplier) {
            return ShallowRaggedTensor<float>(
                buf, dims, raggedStrides(dims), SDPA_SEQ_AXIS, table, std::nullopt, multiplier);
        };
        auto qR = wrap(q.memory().hostData(), qDims, qoTable, mult.q);
        auto kR = wrap(k.memory().hostData(), kDims, kvTable, mult.k);
        auto vR = wrap(v.memory().hostData(), vDims, kvTable, mult.v);
        auto oR = wrap(oCpuBack.data(), oDims, qoTable, mult.o);
        auto lseR = wrap(lseCpuBack.data(), lseDims, qoTable, mult.lse);
        CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
            qR, kR, vR, oR, std::nullopt, -1, 0, /*topLeftAlignment=*/false, &lseR);
    }

    auto qoTokens = makeRaggedOffset(cumQ, 1);
    auto kvTokens = makeRaggedOffset(cumKv, 1);
    GpuFpReferenceSdpaRagged::fpropRagged<float, float, float, float, float>(
        q,
        k,
        v,
        oGpu,
        qoTokens,
        kvTokens,
        kvTokens,
        qoTokens,
        std::nullopt,
        -1,
        0,
        /*topLeftAlignment=*/false,
        &lseGpu,
        &qoTokens,
        SdpaSoftmaxProbabilityMode::FLOAT,
        nullptr,
        nullptr,
        nullptr,
        mult);

    const float tolerance = gpuRefFwdTolerance<float>();
    compareRaggedPacked(oGpu, oCpuBack, tolerance);
    const auto* lg = lseGpu.memory().hostData();
    for(size_t i = 0; i < lseCpuBack.size(); ++i)
    {
        EXPECT_NEAR(lg[i], lseCpuBack[i], tolerance) << "LSE mismatch at element " << i;
    }
}

} // namespace

TEST(TestGpuSdpaRaggedFwdFp32, TokenOffsetsHd192Gqa)
{
    SKIP_IF_NO_DEVICES();
    checkTokenOffsets({5, 0, 9}, {7, 3, 6}, 4, 2, 192, 128);
}

// H*D = 16 and every batch 16 tokens long: read as element offsets, the token table {0, 16, 32}
// is one token per batch and still passes every offset check, so ignoring the multiplier would
// compute on the wrong rows without an error.
TEST(TestGpuSdpaRaggedFwdFp32, TokenOffsetsTokenWidth16)
{
    SKIP_IF_NO_DEVICES();
    checkTokenOffsets({16, 16}, {16, 16}, 1, 1, 16, 16);
}

TEST(TestGpuSdpaRaggedFwdFp32, ThrowsOnZeroOffsetMultiplier)
{
    SKIP_IF_NO_DEVICES();
    const auto dims = raggedDims(1, 4, 2, 16);
    Tensor<float> q(dims, raggedStrides(dims));
    Tensor<float> k(dims, raggedStrides(dims));
    Tensor<float> v(dims, raggedStrides(dims));
    Tensor<float> o(dims, raggedStrides(dims));
    auto tokens = makeRaggedOffset(cumTokens({4}), 1);
    RaggedOffsetMultipliers mult;
    mult.q = mult.o = 32;
    mult.k = 0;
    mult.v = 32;
    EXPECT_THROW((GpuFpReferenceSdpaRagged::fpropRagged<float, float, float, float, float>(
                     q,
                     k,
                     v,
                     o,
                     tokens,
                     tokens,
                     tokens,
                     tokens,
                     std::nullopt,
                     -1,
                     -1,
                     true,
                     nullptr,
                     nullptr,
                     SdpaSoftmaxProbabilityMode::FLOAT,
                     nullptr,
                     nullptr,
                     nullptr,
                     mult)),
                 std::invalid_argument);
}

// A seq_len whose values lie on the second axis ([1, B, 1, 1], packed) is rejected before any
// device read; the same values on [B, 1, 1, 1] are accepted.
TEST(TestGpuSdpaRaggedFwdFp32, ThrowsOnSeqLenOffTheBatchAxis)
{
    SKIP_IF_NO_DEVICES();
    const auto dims = raggedDims(2, 4, 2, 16);
    Tensor<float> q(dims, raggedStrides(dims));
    Tensor<float> k(dims, raggedStrides(dims));
    Tensor<float> v(dims, raggedStrides(dims));
    Tensor<float> o(dims, raggedStrides(dims));
    q.fillWithValue(0.5f);
    k.fillWithValue(0.5f);
    v.fillWithValue(1.0f);
    auto tokens = makeRaggedOffset(cumTokens({4, 4}), 1);
    const auto run = [&](const std::shared_ptr<Tensor<int32_t>>& seqLen) {
        GpuFpReferenceSdpaRagged::fpropRagged<float, float, float, float, float>(
            q,
            k,
            v,
            o,
            tokens,
            tokens,
            tokens,
            tokens,
            std::nullopt,
            -1,
            -1,
            true,
            nullptr,
            nullptr,
            SdpaSoftmaxProbabilityMode::FLOAT,
            nullptr,
            nullptr,
            nullptr,
            RaggedOffsetMultipliers{32, 32, 32, 32},
            seqLen.get(),
            seqLen.get());
    };
    auto offAxis = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{1, 2, 1, 1},
                                                     std::vector<int64_t>{2, 1, 1, 1});
    offAxis->memory().hostData()[0] = 4;
    offAxis->memory().hostData()[1] = 4;
    offAxis->memory().markHostModified();
    EXPECT_THROW(run(offAxis), std::invalid_argument);
    EXPECT_NO_THROW(run(makeInt32Aux({4, 4})));
}

// --- Edge cases the batch lookup, per-batch mask alignment and strides must handle ---

// The kernel finds each token's batch with a linear scan over the Q boundaries. Empty batches
// at the front, at the back and back to back each share a boundary with a neighbour.
TEST(TestGpuSdpaRaggedFwdFp32, LeadingEmptyQBatch)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<float>({0, 3, 2}, {2, 3, 1}, 2, 2, 2, 16, 16);
}

TEST(TestGpuSdpaRaggedFwdFp32, TrailingAndConsecutiveEmptyQBatches)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<float>({2, 0, 0, 3, 0}, {1, 2, 0, 3, 2}, 2, 2, 2, 16, 16);
}

// Bottom-right causal with more queries than keys in some batches: the first Sq - Skv rows of
// those batches are fully masked, in the middle of the packed Q buffer.
TEST(TestGpuSdpaRaggedFwdFp32, CausalBottomRightMoreQueriesThanKeys)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<float>({5, 2, 4}, {2, 5, 1}, 2, 2, 2, 16, 16, -1, 0, /*topLeftAlignment=*/false);
}

// A left-only window with bottom-right alignment uses the per-batch windowOffset in the left
// bound, which no other case reaches.
TEST(TestGpuSdpaRaggedFwdFp32, SlidingWindowLeftOnlyBottomRight)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<float>({6, 3, 4}, {4, 7, 4}, 2, 2, 2, 16, 16, 1, -1, /*topLeftAlignment=*/false);
}

// K and V head counts are independent (H = 8, Hk = 2, Hv = 4), with Dv != D.
TEST(TestGpuSdpaRaggedFwdFp32, DistinctKAndVHeadCounts)
{
    SKIP_IF_NO_DEVICES();
    checkRagged<float>({3, 5}, {4, 2}, 8, 2, 4, 16, 32);
}

namespace
{

// Token-major layouts other than BSHD-strided [B, H, S, D]. Both pass the RFC-0014 layout check.
enum class TokenLayout
{
    PADDED_TOKEN_STRIDE, // strides[2] = H * D + 8: gaps between tokens
    DIM_MAJOR_TOKEN, // inside a token, D is the outer axis: strides {.., 1, H * D, H}
};

std::vector<int64_t> tokenLayoutStrides(const std::vector<int64_t>& dims, TokenLayout layout)
{
    const auto heads = dims[1];
    const auto seq = dims[2];
    const auto dim = dims[3];
    if(layout == TokenLayout::PADDED_TOKEN_STRIDE)
    {
        const auto tokenStride = heads * dim + 8;
        return {seq * tokenStride, dim, tokenStride, 1};
    }
    return {seq * heads * dim, 1, heads * dim, heads};
}

// fp32 self-shaped attention (Hk = Hv = H, Dv = D) on non-contiguous token-major tensors.
// Offsets are cum * strides[2]. Only valid tokens are compared, through the strides.
void checkRaggedTokenLayout(const std::vector<int64_t>& seqQ,
                            const std::vector<int64_t>& seqKv,
                            int64_t numHeads,
                            int64_t headDim,
                            TokenLayout layout)
{
    const auto batch = static_cast<int64_t>(seqQ.size());
    const auto cumQ = cumTokens(seqQ);
    const auto cumKv = cumTokens(seqKv);
    const auto qDims = raggedDims(batch, maxOf(seqQ), numHeads, headDim);
    const auto kvDims = raggedDims(batch, maxOf(seqKv), numHeads, headDim);
    const auto qStrides = tokenLayoutStrides(qDims, layout);
    const auto kvStrides = tokenLayoutStrides(kvDims, layout);

    Tensor<float> q(qDims, qStrides);
    Tensor<float> k(kvDims, kvStrides);
    Tensor<float> v(kvDims, kvStrides);
    Tensor<float> oGpu(qDims, qStrides);
    q.fillWithRandomValues(-1.0f, 1.0f, SEED_Q);
    k.fillWithRandomValues(-1.0f, 1.0f, SEED_K);
    v.fillWithRandomValues(-1.0f, 1.0f, SEED_V);

    std::vector<float> oCpuBack(static_cast<size_t>(cumQ.back() * raggedSeqStride(qStrides)), 0.0f);
    {
        const auto wrap = [](float* buf,
                             const std::vector<int64_t>& dims,
                             const std::vector<int64_t>& strides,
                             const std::vector<int64_t>& cum) {
            return ShallowRaggedTensor<float>(buf,
                                              dims,
                                              strides,
                                              SDPA_SEQ_AXIS,
                                              makeRaggedOffsetAux(cum, raggedSeqStride(strides)));
        };
        auto qR = wrap(q.memory().hostData(), qDims, qStrides, cumQ);
        auto kR = wrap(k.memory().hostData(), kvDims, kvStrides, cumKv);
        auto vR = wrap(v.memory().hostData(), kvDims, kvStrides, cumKv);
        auto oR = wrap(oCpuBack.data(), qDims, qStrides, cumQ);
        CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(qR, kR, vR, oR);
    }

    auto offQ = makeRaggedOffset(cumQ, raggedSeqStride(qStrides));
    auto offKv = makeRaggedOffset(cumKv, raggedSeqStride(kvStrides));
    GpuFpReferenceSdpaRagged::fpropRagged<float, float, float, float, float>(
        q, k, v, oGpu, offQ, offKv, offKv, offQ);

    const auto* g = oGpu.memory().hostData();
    const float tolerance = gpuRefFwdTolerance<float>();
    for(int64_t token = 0; token < cumQ.back(); ++token)
    {
        for(int64_t h = 0; h < numHeads; ++h)
        {
            for(int64_t d = 0; d < headDim; ++d)
            {
                const auto i = static_cast<size_t>(token * raggedSeqStride(qStrides)
                                                   + h * qStrides[1] + d * qStrides[3]);
                EXPECT_NEAR(g[i], oCpuBack[i], tolerance)
                    << "output mismatch at token " << token << " head " << h << " dim " << d;
            }
        }
    }
}

} // namespace

TEST(TestGpuSdpaRaggedFwdFp32, PaddedTokenStride)
{
    SKIP_IF_NO_DEVICES();
    checkRaggedTokenLayout({3, 0, 5}, {4, 2, 3}, 2, 16, TokenLayout::PADDED_TOKEN_STRIDE);
}

TEST(TestGpuSdpaRaggedFwdFp32, DimMajorTokenLayout)
{
    SKIP_IF_NO_DEVICES();
    checkRaggedTokenLayout({3, 0, 5}, {4, 2, 3}, 2, 16, TokenLayout::DIM_MAJOR_TOKEN);
}

// Fully masked rows (Sq > Skv under bottom-right causal, and an empty KV batch) must write a zero
// output and LSE = -inf at their own packed row, and the rows next to them must stay correct.
// EXPECT_NEAR cannot compare -inf, so -inf rows are compared exactly.
TEST(TestGpuSdpaRaggedFwdFp32, PackedLseOfFullyMaskedRows)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> seqQ = {4, 3, 2};
    const std::vector<int64_t> seqKv = {2, 0, 3};
    const int64_t numHeads = 2;
    const int64_t headDim = 16;
    const auto batch = static_cast<int64_t>(seqQ.size());
    const auto totalQ = sum(seqQ);
    const auto totalKv = sum(seqKv);
    const auto cumQ = cumTokens(seqQ);
    const auto cumKv = cumTokens(seqKv);
    const auto qDims = raggedDims(batch, maxOf(seqQ), numHeads, headDim);
    const auto kvDims = raggedDims(batch, maxOf(seqKv), numHeads, headDim);
    const auto lseDims = raggedDims(batch, maxOf(seqQ), numHeads, 1);
    const int64_t seqStride = numHeads * headDim;

    Tensor<float> q(qDims, raggedStrides(qDims));
    Tensor<float> k(kvDims, raggedStrides(kvDims));
    Tensor<float> v(kvDims, raggedStrides(kvDims));
    Tensor<float> oGpu(qDims, raggedStrides(qDims));
    Tensor<float> lseGpu(lseDims, raggedStrides(lseDims));
    fillPackedRandom(q, totalQ * seqStride, -1.0f, 1.0f, SEED_Q);
    fillPackedRandom(k, totalKv * seqStride, -1.0f, 1.0f, SEED_K);
    fillPackedRandom(v, totalKv * seqStride, -1.0f, 1.0f, SEED_V);
    lseGpu.fillWithValue(LSE_SENTINEL);

    std::vector<float> oCpuBack(static_cast<size_t>(totalQ * seqStride), 0.0f);
    std::vector<float> lseCpuBack(static_cast<size_t>(totalQ * numHeads), LSE_SENTINEL);
    {
        auto qR = wrapRagged(q.memory().hostData(), qDims, seqStride, cumQ);
        auto kR = wrapRagged(k.memory().hostData(), kvDims, seqStride, cumKv);
        auto vR = wrapRagged(v.memory().hostData(), kvDims, seqStride, cumKv);
        auto oR = wrapRagged(oCpuBack.data(), qDims, seqStride, cumQ);
        auto lseR = wrapRagged(lseCpuBack.data(), lseDims, numHeads, cumQ);
        CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
            qR, kR, vR, oR, std::nullopt, -1, 0, /*topLeftAlignment=*/false, &lseR);
    }

    auto offQ = makeRaggedOffset(cumQ, seqStride);
    auto offKv = makeRaggedOffset(cumKv, seqStride);
    auto offLse = makeRaggedOffset(cumQ, numHeads);
    GpuFpReferenceSdpaRagged::fpropRagged<float, float, float, float, float>(
        q, k, v, oGpu, offQ, offKv, offKv, offQ, std::nullopt, -1, 0, false, &lseGpu, &offLse);

    const float tolerance = gpuRefFwdTolerance<float>();
    compareRaggedPacked(oGpu, oCpuBack, tolerance);

    // Batch 0 rows 0-1 (Sq - Skv = 2) and all of batch 1 (Skv = 0) are fully masked.
    int64_t maskedRows = 0;
    const auto* lg = lseGpu.memory().hostData();
    for(size_t i = 0; i < lseCpuBack.size(); ++i)
    {
        if(std::isinf(lseCpuBack[i]))
        {
            ++maskedRows;
            EXPECT_EQ(lg[i], lseCpuBack[i]) << "masked-row LSE mismatch at element " << i;
        }
        else
        {
            EXPECT_NEAR(lg[i], lseCpuBack[i], tolerance) << "LSE mismatch at element " << i;
        }
    }
    EXPECT_EQ(maskedRows, (2 + 3) * numHeads);
}

namespace
{

// --- Layout rows of RFC 0014 §4.5.1 (L2, L5s, L6, L7a, L7c, L8, L9) ---
//
// Each case builds owning RaggedTensors with the row's strides and origins. Inputs hold a
// deterministic non-constant value in every live element and a large poison in every gap, so a
// kernel that reads outside the live rows fails. Outputs start at a sentinel in the whole
// storage: live elements are compared with the CPU ragged reference at their logical
// coordinates, and every gap element must keep the sentinel. Storage is addressed by the
// explicit formula origin[b] + h * strides[1] + s * strides[2] + d * strides[3], independently of
// the Data SDK's own addressing.

constexpr float POISON_INPUT = 1000.0f;
constexpr int64_t SALT_Q = 1;
constexpr int64_t SALT_K = 2;
constexpr int64_t SALT_V = 3;

// Layout of one ragged tensor: dims [B, H, S_max, D], strides (the batch stride is unused) and
// B + 1 element-unit offsets.
struct OperandSpec
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    std::vector<int64_t> offsets;
};

struct RaggedFwdCase
{
    OperandSpec q;
    OperandSpec k;
    OperandSpec v;
    OperandSpec o;
    std::optional<OperandSpec> lse;
    std::vector<int64_t> lensQ;
    std::vector<int64_t> lensKv;
    // Bind seq_len_q / seq_len_kv to the lengths, else the offset blocks define them.
    bool bindLengths = false;
    // Batch-axis stride of the bound seq_len tensors; the slots between their values hold
    // STRIDED_AUX_POISON.
    int64_t seqLenStride = 1;
    // Batch-axis stride of every offset table; the slots between entries hold STRIDED_AUX_POISON.
    int64_t offsetStride = 1;
};

struct RaggedOperand
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    std::vector<int64_t> origins;
    std::vector<int64_t> lens;
    std::shared_ptr<Tensor<int32_t>> offsets;
    std::shared_ptr<Tensor<int32_t>> lengths; // null unless bound
    std::unique_ptr<RaggedTensor<float>> tensor;
};

struct LiveElement
{
    int64_t index;
    int64_t b;
    int64_t h;
    int64_t s;
    int64_t d;
};

// rowOffsets are in token rows (B + 1 entries); the offset table holds rowOffsets * seqStride.
OperandSpec makeSpec(const std::vector<int64_t>& lens,
                     int64_t heads,
                     int64_t dim,
                     int64_t headStride,
                     int64_t seqStride,
                     const std::vector<int64_t>& rowOffsets)
{
    OperandSpec spec;
    const auto sMax = maxOf(lens);
    spec.dims = raggedDims(static_cast<int64_t>(lens.size()), sMax, heads, dim);
    spec.strides = {sMax * seqStride, headStride, seqStride, 1};
    spec.offsets.reserve(rowOffsets.size());
    for(const auto row : rowOffsets)
    {
        spec.offsets.push_back(row * seqStride);
    }
    return spec;
}

RaggedOperand makeOperand(const OperandSpec& spec,
                          const std::vector<int64_t>& lens,
                          bool bindLengths,
                          int64_t seqLenStride,
                          int64_t offsetStride)
{
    RaggedOperand op;
    op.dims = spec.dims;
    op.strides = spec.strides;
    op.lens = lens;
    op.origins.assign(spec.offsets.begin(), spec.offsets.end() - 1);
    const auto toInt32 = [](const std::vector<int64_t>& values) {
        return std::vector<int32_t>(values.begin(), values.end());
    };
    op.offsets = makeInt32Aux(toInt32(spec.offsets), offsetStride);
    if(bindLengths)
    {
        op.lengths = makeInt32Aux(toInt32(lens), seqLenStride);
    }
    op.tensor = std::make_unique<RaggedTensor<float>>(
        spec.dims, spec.strides, SDPA_SEQ_AXIS, op.offsets, std::nullopt, 1, op.lengths);
    // The container must derive the lengths the case intends (block clamped to S_max, or seq_len).
    const auto info = op.tensor->raggedIterationInfo();
    EXPECT_TRUE(info.has_value());
    if(info.has_value())
    {
        EXPECT_EQ(info->lengths, lens);
    }
    return op;
}

int64_t storageIndex(const RaggedOperand& op, int64_t b, int64_t h, int64_t s, int64_t d)
{
    return op.origins[static_cast<size_t>(b)] + h * op.strides[1] + s * op.strides[2]
           + d * op.strides[3];
}

template <typename Fn>
void forEachLive(const RaggedOperand& op, Fn&& fn)
{
    for(int64_t b = 0; b < op.dims[0]; ++b)
    {
        for(int64_t h = 0; h < op.dims[1]; ++h)
        {
            for(int64_t s = 0; s < op.lens[static_cast<size_t>(b)]; ++s)
            {
                for(int64_t d = 0; d < op.dims[3]; ++d)
                {
                    fn(LiveElement{storageIndex(op, b, h, s, d), b, h, s, d});
                }
            }
        }
    }
}

// Distinct per batch, head, token, dim and tensor, and never constant.
float inputValue(int64_t salt, const LiveElement& e)
{
    const int64_t key = (e.b * 7 + e.h * 5 + e.s * 3 + e.d * 11 + salt * 13) % 17;
    return 0.1f * static_cast<float>(key) - 0.8f;
}

void fillWholeStorage(RaggedOperand& op, float value)
{
    auto& memory = op.tensor->memory();
    std::fill_n(memory.hostData(), memory.count(), value);
    memory.markHostModified();
}

void fillInput(RaggedOperand& op, int64_t salt)
{
    auto& memory = op.tensor->memory();
    float* p = memory.hostData();
    std::fill_n(p, memory.count(), POISON_INPUT);
    forEachLive(op, [&](const LiveElement& e) { p[e.index] = inputValue(salt, e); });
    memory.markHostModified();
}

void expectLiveClose(const RaggedOperand& gpu,
                     const RaggedOperand& cpu,
                     float tolerance,
                     const char* name)
{
    const float* g = gpu.tensor->memory().hostData();
    const float* c = cpu.tensor->memory().hostData();
    forEachLive(gpu, [&](const LiveElement& e) {
        EXPECT_NEAR(g[e.index], c[storageIndex(cpu, e.b, e.h, e.s, e.d)], tolerance)
            << name << " mismatch at [b=" << e.b << ", h=" << e.h << ", s=" << e.s << ", d=" << e.d
            << "]";
    });
}

void expectGapsUntouched(const RaggedOperand& op, const char* name)
{
    const float* g = op.tensor->memory().hostData();
    const auto count = op.tensor->memory().count();
    std::vector<uint8_t> live(count, 0);
    forEachLive(op, [&](const LiveElement& e) { live[static_cast<size_t>(e.index)] = 1; });
    for(size_t i = 0; i < count; ++i)
    {
        if(live[i] == 0)
        {
            EXPECT_EQ(g[i], LSE_SENTINEL) << name << " gap element " << i << " was written";
        }
    }
}

// A batch mix-up would make batches 0 and 1 agree. Both batches must have a live first row.
void expectBatchesDiffer(const RaggedOperand& op, const char* name)
{
    const float* p = op.tensor->memory().hostData();
    bool differs = false;
    for(int64_t h = 0; h < op.dims[1]; ++h)
    {
        for(int64_t d = 0; d < op.dims[3]; ++d)
        {
            const bool valueDiffers
                = p[storageIndex(op, 0, h, 0, d)] != p[storageIndex(op, 1, h, 0, d)];
            differs = differs || valueDiffers;
        }
    }
    EXPECT_TRUE(differs) << name << " is identical in batches 0 and 1";
}

// Runs the CPU ragged reference and the GPU reference on the same tensors and compares the
// live elements of O (and the LSE), then checks that the GPU left every gap element alone.
void runRaggedFwdCase(const RaggedFwdCase& c)
{
    auto q = makeOperand(c.q, c.lensQ, c.bindLengths, c.seqLenStride, c.offsetStride);
    auto k = makeOperand(c.k, c.lensKv, c.bindLengths, c.seqLenStride, c.offsetStride);
    auto v = makeOperand(c.v, c.lensKv, c.bindLengths, c.seqLenStride, c.offsetStride);
    fillInput(q, SALT_Q);
    fillInput(k, SALT_K);
    fillInput(v, SALT_V);

    auto oCpu = makeOperand(c.o, c.lensQ, c.bindLengths, c.seqLenStride, c.offsetStride);
    auto oGpu = makeOperand(c.o, c.lensQ, c.bindLengths, c.seqLenStride, c.offsetStride);
    fillWholeStorage(oCpu, LSE_SENTINEL);
    fillWholeStorage(oGpu, LSE_SENTINEL);

    std::optional<RaggedOperand> lseCpu;
    std::optional<RaggedOperand> lseGpu;
    if(c.lse.has_value())
    {
        lseCpu = makeOperand(*c.lse, c.lensQ, c.bindLengths, c.seqLenStride, c.offsetStride);
        lseGpu = makeOperand(*c.lse, c.lensQ, c.bindLengths, c.seqLenStride, c.offsetStride);
        fillWholeStorage(*lseCpu, LSE_SENTINEL);
        fillWholeStorage(*lseGpu, LSE_SENTINEL);
    }

    CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
        *q.tensor,
        *k.tensor,
        *v.tensor,
        *oCpu.tensor,
        std::nullopt,
        -1,
        -1,
        true,
        lseCpu.has_value() ? lseCpu->tensor.get() : nullptr);

    GpuFpReferenceSdpaRagged::fpropRagged<float, float, float, float, float>(
        *q.tensor,
        *k.tensor,
        *v.tensor,
        *oGpu.tensor,
        *q.offsets,
        *k.offsets,
        *v.offsets,
        *oGpu.offsets,
        std::nullopt,
        -1,
        -1,
        true,
        lseGpu.has_value() ? lseGpu->tensor.get() : nullptr,
        lseGpu.has_value() ? lseGpu->offsets.get() : nullptr,
        SdpaSoftmaxProbabilityMode::FLOAT,
        nullptr,
        nullptr,
        nullptr,
        RaggedOffsetMultipliers{},
        q.lengths.get(),
        k.lengths.get());

    const float tolerance = gpuRefFwdTolerance<float>();
    expectLiveClose(oGpu, oCpu, tolerance, "O");
    expectGapsUntouched(oGpu, "O");
    expectBatchesDiffer(oGpu, "O");
    if(lseGpu.has_value())
    {
        expectLiveClose(*lseGpu, *lseCpu, tolerance, "LSE");
        expectGapsUntouched(*lseGpu, "LSE");
        expectBatchesDiffer(*lseGpu, "LSE");
    }
}

} // namespace

// Row L2: V and O sit in a fused-QKV style buffer whose token stride (3 * H * D = 96) is three
// times the token width, so each of their tokens is followed by 64 gap elements. Q and K are
// packed. PaddedTokenStride covers the same property with a +8 token gap but checks only live
// elements, with random (not poisoned) gaps.
TEST(TestGpuSdpaRaggedFwdFp32, FusedQkvTokenStrideLeavesGapsUntouched)
{
    SKIP_IF_NO_DEVICES();
    const std::vector<int64_t> lensQ = {2, 3, 4};
    const std::vector<int64_t> lensKv = {3, 2, 4};
    const int64_t heads = 4;
    const int64_t dim = 8;
    const int64_t tokenWidth = heads * dim;
    const int64_t fusedStride = 3 * tokenWidth;

    RaggedFwdCase c;
    c.lensQ = lensQ;
    c.lensKv = lensKv;
    c.q = makeSpec(lensQ, heads, dim, dim, tokenWidth, cumTokens(lensQ));
    c.k = makeSpec(lensKv, heads, dim, dim, tokenWidth, cumTokens(lensKv));
    c.v = makeSpec(lensKv, heads, dim, dim, fusedStride, cumTokens(lensKv));
    c.o = makeSpec(lensQ, heads, dim, dim, fusedStride, cumTokens(lensQ));
    runRaggedFwdCase(c);
}

// Row L5s: every batch block is 8 rows, longer than S_max (5 for Q, 5 for K/V). seq_len_q and
// seq_len_kv bind lengths of 3 and 5 (Q) and 5 and 2 (KV), shorter than the blocks, so the rows
// between a batch's length and the next origin are gaps. Without the seq_len binding a length
// would clamp to S_max = 5 and the results would differ. The ragged LSE is [T, H] with the same
// block origins.
TEST(TestGpuSdpaRaggedFwdFp32, SeqLenShorterThanBlockLongerThanSmax)
{
    SKIP_IF_NO_DEVICES();
    const std::vector<int64_t> lensQ = {3, 5, 2};
    const std::vector<int64_t> lensKv = {5, 2, 4};
    const std::vector<int64_t> blockRows = {0, 8, 16, 24}; // last entry unused with seq_len
    const int64_t heads = 2;
    const int64_t dim = 8;
    const int64_t tokenWidth = heads * dim;

    RaggedFwdCase c;
    c.lensQ = lensQ;
    c.lensKv = lensKv;
    c.bindLengths = true;
    c.q = makeSpec(lensQ, heads, dim, dim, tokenWidth, blockRows);
    c.k = makeSpec(lensKv, heads, dim, dim, tokenWidth, blockRows);
    c.v = makeSpec(lensKv, heads, dim, dim, tokenWidth, blockRows);
    c.o = makeSpec(lensQ, heads, dim, dim, tokenWidth, blockRows);
    c.lse = makeSpec(lensQ, heads, 1, 1, heads, blockRows);
    runRaggedFwdCase(c);
}

// seq_len_q / seq_len_kv with batch stride 2 (odd slots poisoned out of range): the GPU reference
// must read element b at b * 2, as the CPU reference does. Lengths are shorter than the offset
// blocks, so a packed read would also pick the wrong lengths.
TEST(TestGpuSdpaRaggedFwdFp32, StridedSeqLenMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    const std::vector<int64_t> lensQ = {3, 5, 2};
    const std::vector<int64_t> lensKv = {5, 2, 4};
    const std::vector<int64_t> blockRows = {0, 8, 16, 24}; // last entry unused with seq_len
    const int64_t heads = 2;
    const int64_t dim = 8;
    const int64_t tokenWidth = heads * dim;

    RaggedFwdCase c;
    c.lensQ = lensQ;
    c.lensKv = lensKv;
    c.bindLengths = true;
    c.seqLenStride = 2;
    c.q = makeSpec(lensQ, heads, dim, dim, tokenWidth, blockRows);
    c.k = makeSpec(lensKv, heads, dim, dim, tokenWidth, blockRows);
    c.v = makeSpec(lensKv, heads, dim, dim, tokenWidth, blockRows);
    c.o = makeSpec(lensQ, heads, dim, dim, tokenWidth, blockRows);
    c.lse = makeSpec(lensQ, heads, 1, 1, heads, blockRows);
    runRaggedFwdCase(c);
}

// Offset tables with batch stride 3 and seq_len with stride 2, odd slots poisoned: the GPU
// reference must read entry i of each at i * stride, as the CPU reference does.
TEST(TestGpuSdpaRaggedFwdFp32, StridedOffsetsAndSeqLenMatchCpu)
{
    SKIP_IF_NO_DEVICES();
    const std::vector<int64_t> lensQ = {3, 5, 2};
    const std::vector<int64_t> lensKv = {5, 2, 4};
    const std::vector<int64_t> blockRows = {0, 8, 16, 24};
    const int64_t heads = 2;
    const int64_t dim = 8;
    const int64_t tokenWidth = heads * dim;

    RaggedFwdCase c;
    c.lensQ = lensQ;
    c.lensKv = lensKv;
    c.bindLengths = true;
    c.seqLenStride = 2;
    c.offsetStride = 3;
    c.q = makeSpec(lensQ, heads, dim, dim, tokenWidth, blockRows);
    c.k = makeSpec(lensKv, heads, dim, dim, tokenWidth, blockRows);
    c.v = makeSpec(lensKv, heads, dim, dim, tokenWidth, blockRows);
    c.o = makeSpec(lensQ, heads, dim, dim, tokenWidth, blockRows);
    c.lse = makeSpec(lensQ, heads, 1, 1, heads, blockRows);
    runRaggedFwdCase(c);
}

// Lengths come from the offset blocks (no seq_len), so both offsets b and b + 1 are read strided.
TEST(TestGpuSdpaRaggedFwdFp32, StridedOffsetsMatchCpu)
{
    SKIP_IF_NO_DEVICES();
    const std::vector<int64_t> lensQ = {2, 3, 4};
    const std::vector<int64_t> lensKv = {3, 2, 4};
    const int64_t heads = 2;
    const int64_t dim = 8;
    const int64_t tokenWidth = heads * dim;

    RaggedFwdCase c;
    c.lensQ = lensQ;
    c.lensKv = lensKv;
    c.offsetStride = 2;
    c.q = makeSpec(lensQ, heads, dim, dim, tokenWidth, cumTokens(lensQ));
    c.k = makeSpec(lensKv, heads, dim, dim, tokenWidth, cumTokens(lensKv));
    c.v = makeSpec(lensKv, heads, dim, dim, tokenWidth, cumTokens(lensKv));
    c.o = makeSpec(lensQ, heads, dim, dim, tokenWidth, cumTokens(lensQ));
    c.lse = makeSpec(lensQ, heads, 1, 1, heads, cumTokens(lensQ));
    runRaggedFwdCase(c);
}

// Row L6: head-major. Each head owns a contiguous plane (head stride = total rows * D, token
// stride D). Q and O are head-major, V is head-major with its own row count, K stays token-major,
// so each tensor's strides must be honoured independently. AcceptsHeadMajorLayout only checks
// that such a layout is accepted.
TEST(TestGpuSdpaRaggedFwdFp32, HeadMajorLayoutMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    const std::vector<int64_t> lensQ = {2, 3, 4};
    const std::vector<int64_t> lensKv = {3, 1, 4};
    const int64_t heads = 2;
    const int64_t dim = 8;
    const int64_t rowsQ = sum(lensQ);
    const int64_t rowsKv = sum(lensKv);

    RaggedFwdCase c;
    c.lensQ = lensQ;
    c.lensKv = lensKv;
    c.q = makeSpec(lensQ, heads, dim, rowsQ * dim, dim, cumTokens(lensQ));
    c.k = makeSpec(lensKv, heads, dim, dim, heads * dim, cumTokens(lensKv));
    c.v = makeSpec(lensKv, heads, dim, rowsKv * dim, dim, cumTokens(lensKv));
    c.o = makeSpec(lensQ, heads, dim, rowsQ * dim, dim, cumTokens(lensQ));
    runRaggedFwdCase(c);
}

namespace
{

// Packed token-major Q/K/V/O with a ragged [H, T] LSE: element (b, h, s) lives at
// cum[b] + s + h * lseHeadPitch, and the LSE offsets are token offsets (seq stride 1).
void checkHeadTokenLse(int64_t lseHeadPitch)
{
    const std::vector<int64_t> lensQ = {2, 3, 3};
    const std::vector<int64_t> lensKv = {3, 2, 4};
    const int64_t heads = 4;
    const int64_t dim = 8;
    const int64_t tokenWidth = heads * dim;
    const auto cumQ = cumTokens(lensQ);

    RaggedFwdCase c;
    c.lensQ = lensQ;
    c.lensKv = lensKv;
    c.q = makeSpec(lensQ, heads, dim, dim, tokenWidth, cumQ);
    c.k = makeSpec(lensKv, heads, dim, dim, tokenWidth, cumTokens(lensKv));
    c.v = makeSpec(lensKv, heads, dim, dim, tokenWidth, cumTokens(lensKv));
    c.o = makeSpec(lensQ, heads, dim, dim, tokenWidth, cumQ);
    c.lse = makeSpec(lensQ, heads, 1, lseHeadPitch, 1, cumQ);
    runRaggedFwdCase(c);
}

} // namespace

// Row L7a: [H, T] ragged LSE with H = 4: head pitch equals the total token count (8).
TEST(TestGpuSdpaRaggedFwdFp32, RaggedLseHeadTokenLayout)
{
    SKIP_IF_NO_DEVICES();
    checkHeadTokenLse(/*lseHeadPitch=*/8);
}

// Row L7c: [H, T] ragged LSE whose head pitch (16) exceeds the token count (8), leaving a gap
// after each head's tokens that the GPU must not write.
TEST(TestGpuSdpaRaggedFwdFp32, RaggedLseHeadPitchBeyondTokenCount)
{
    SKIP_IF_NO_DEVICES();
    checkHeadTokenLse(/*lseHeadPitch=*/16);
}

// Row L8: shifted origins, offset[0] != 0. Each tensor's first batch starts after leading gap
// rows (3 for Q, 1 for K and V, 2 for O); lengths come from the offset blocks.
TEST(TestGpuSdpaRaggedFwdFp32, ShiftedOriginOffsets)
{
    SKIP_IF_NO_DEVICES();
    const std::vector<int64_t> lensQ = {3, 2, 4};
    const std::vector<int64_t> lensKv = {2, 4, 3};
    const int64_t heads = 2;
    const int64_t dim = 8;
    const int64_t tokenWidth = heads * dim;

    RaggedFwdCase c;
    c.lensQ = lensQ;
    c.lensKv = lensKv;
    c.q = makeSpec(lensQ, heads, dim, dim, tokenWidth, {3, 6, 8, 12});
    c.k = makeSpec(lensKv, heads, dim, dim, tokenWidth, {1, 3, 7, 10});
    c.v = makeSpec(lensKv, heads, dim, dim, tokenWidth, {1, 3, 7, 10});
    c.o = makeSpec(lensQ, heads, dim, dim, tokenWidth, {2, 5, 7, 11});
    runRaggedFwdCase(c);
}

// Row L9: non-monotonic origins with seq_len. Batch 0 sits after batches 1 and 2 in Q (origin rows
// 10, 0, 5) and K/V (6, 0, 2), and after batch 2 only in O (5, 10, 0); the lengths come from
// seq_len_q / seq_len_kv and the last offset entry is unused.
TEST(TestGpuSdpaRaggedFwdFp32, NonMonotonicOriginsWithSeqLen)
{
    SKIP_IF_NO_DEVICES();
    const std::vector<int64_t> lensQ = {2, 3, 4};
    const std::vector<int64_t> lensKv = {3, 2, 4};
    const int64_t heads = 2;
    const int64_t dim = 8;
    const int64_t tokenWidth = heads * dim;

    RaggedFwdCase c;
    c.lensQ = lensQ;
    c.lensKv = lensKv;
    c.bindLengths = true;
    c.q = makeSpec(lensQ, heads, dim, dim, tokenWidth, {10, 0, 5, 0});
    c.k = makeSpec(lensKv, heads, dim, dim, tokenWidth, {6, 0, 2, 0});
    c.v = makeSpec(lensKv, heads, dim, dim, tokenWidth, {6, 0, 2, 0});
    c.o = makeSpec(lensQ, heads, dim, dim, tokenWidth, {5, 10, 0, 0});
    runRaggedFwdCase(c);
}
