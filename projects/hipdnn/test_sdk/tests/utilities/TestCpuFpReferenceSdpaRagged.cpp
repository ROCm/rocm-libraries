// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ============================================================================
// Validates the ragged CPU reference (CpuFpReferenceSdpaRagged) against the
// trusted dense CpuFpReferenceSdpa. Inputs are RFC-0014 ragged tensors
// (ShallowRaggedTensor over packed buffers + per-primary ragged_offset aux);
// each batch is extracted into a dense [1,H,seqlen_b,D] tensor, the dense
// reference is run on it, and its output (and LSE) is compared with the ragged
// reference. This is the middle link of the validation chain: dense CPU
// (trusted) -> CPU ragged (here) -> GPU ragged (TestGpuFpReferenceSdpaRagged).
//
// This suite deliberately covers the fp8/descale, LSE, and fully-masked branches
// on a CPU-only path (the GPU-vs-CPU suite that also exercises them is
// SKIP_IF_NO_DEVICES and does not run in the coverage lane), and gives fp8 +
// descale an independent dense oracle via dequantize-to-float. CPU-only.
// ============================================================================

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpaRagged.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaTestUtils.hpp>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;

namespace
{

int64_t sum(const std::vector<int64_t>& v)
{
    return std::accumulate(v.begin(), v.end(), int64_t{0});
}

int64_t maxOf(const std::vector<int64_t>& v)
{
    return *std::max_element(v.begin(), v.end());
}

// Deterministic fill of a packed backing buffer in [-1, 1); avoids <random> cross-TU drift.
template <typename T>
void fillPacked(std::vector<T>& buf, unsigned int seed)
{
    uint32_t state = seed;
    for(auto& x : buf)
    {
        state = state * 1664525U + 1013904223U;
        const float u = static_cast<float>(state >> 8) / static_cast<float>(1U << 24); // [0,1)
        x = static_cast<T>(2.0f * u - 1.0f);
    }
}

// Wrap a borrowed packed backing buffer as an RFC-0014 ragged tensor ([B,H,S,D], seqAxis=2, BSHD).
template <typename T>
ShallowRaggedTensor<T> wrapRagged(T* buf,
                                  const std::vector<int64_t>& dims,
                                  int64_t seqStride,
                                  const std::vector<int64_t>& cum)
{
    return ShallowRaggedTensor<T>(
        buf, dims, bshd(dims), SEQ_AXIS, makeRaggedOffsetAux(cum, seqStride));
}

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

// Descale value for (batch, head): scalar [1] or per-head [B, heads, 1, 1].
float descaleValue(TensorBase<float>& descale, int64_t b, int64_t head)
{
    if(descale.elementCount() == 1)
    {
        return descale.getHostValue(std::vector<int64_t>{0});
    }
    return descale.getHostValue(std::vector<int64_t>{b, head, 0, 0});
}

// Extract a dense [1, heads, seqLen, dim] slice for batch b from a ragged tensor (batch-relative
// seq index; ragged addressing handles the packing).
template <typename T>
Tensor<float> extractDenseSlice(TensorBase<T>& ragged, int64_t b, int64_t seqLen)
{
    const auto heads = ragged.dims()[1];
    const auto dim = ragged.dims()[3];
    Tensor<float> dense({1, heads, seqLen, dim});
    for(int64_t h = 0; h < heads; ++h)
    {
        for(int64_t s = 0; s < seqLen; ++s)
        {
            for(int64_t d = 0; d < dim; ++d)
            {
                dense(0, h, s, d)
                    = static_cast<float>(ragged.getHostValue(std::vector<int64_t>{b, h, s, d}));
            }
        }
    }
    dense.memory().markHostModified();
    return dense;
}

// Like extractDenseSlice but dequantizes fp8 -> float and folds in the per-(batch,head) descale.
// Folding descale into the inputs is algebraically the reference's score*=dQ*dK; out*=dV.
template <typename FP8>
Tensor<float> dequantDenseSlice(TensorBase<FP8>& ragged,
                                int64_t b,
                                int64_t seqLen,
                                TensorBase<float>& descale)
{
    const auto heads = ragged.dims()[1];
    const auto dim = ragged.dims()[3];
    Tensor<float> dense({1, heads, seqLen, dim});
    for(int64_t h = 0; h < heads; ++h)
    {
        const float dsc = descaleValue(descale, b, h);
        for(int64_t s = 0; s < seqLen; ++s)
        {
            for(int64_t d = 0; d < dim; ++d)
            {
                dense(0, h, s, d)
                    = static_cast<float>(ragged.getHostValue(std::vector<int64_t>{b, h, s, d}))
                      * dsc;
            }
        }
    }
    dense.memory().markHostModified();
    return dense;
}

// Build packed ragged float inputs (ShallowRaggedTensor), run the ragged CPU reference (with LSE),
// then validate each batch's output and LSE against the dense CPU reference on [1,H,seqlen_b,D]
// slices.
void checkRaggedVsDense(const std::vector<int64_t>& seqQ,
                        const std::vector<int64_t>& seqKv,
                        int64_t numHeads,
                        int64_t numHeadsKv,
                        int64_t headDim,
                        int64_t headDimV,
                        int64_t leftBound = -1,
                        int64_t rightBound = -1,
                        bool topLeftAlignment = true)
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
    const std::vector<int64_t> kDims = {batch, numHeadsKv, sMaxKv, headDim};
    const std::vector<int64_t> vDims = {batch, numHeadsKv, sMaxKv, headDimV};
    const std::vector<int64_t> oDims = {batch, numHeads, sMaxQ, headDimV};
    const std::vector<int64_t> lseDims = {batch, numHeads, sMaxQ, 1};

    std::vector<float> qBack(static_cast<size_t>(totalQ * numHeads * headDim));
    std::vector<float> kBack(static_cast<size_t>(totalKv * numHeadsKv * headDim));
    std::vector<float> vBack(static_cast<size_t>(totalKv * numHeadsKv * headDimV));
    std::vector<float> oBack(static_cast<size_t>(totalQ * numHeads * headDimV), 0.0f);
    std::vector<float> lseBack(static_cast<size_t>(totalQ * numHeads), 0.0f);
    fillPacked(qBack, 11);
    fillPacked(kBack, 22);
    fillPacked(vBack, 33);

    auto q = wrapRagged(qBack.data(), qDims, numHeads * headDim, cumQ);
    auto k = wrapRagged(kBack.data(), kDims, numHeadsKv * headDim, cumKv);
    auto v = wrapRagged(vBack.data(), vDims, numHeadsKv * headDimV, cumKv);
    auto o = wrapRagged(oBack.data(), oDims, numHeads * headDimV, cumQ);
    auto lse = wrapRagged(lseBack.data(), lseDims, numHeads, cumQ);

    CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
        q, k, v, o, std::nullopt, leftBound, rightBound, topLeftAlignment, &lse);

    for(int64_t b = 0; b < batch; ++b)
    {
        const auto sQ = seqQ[static_cast<size_t>(b)];
        const auto sKv = seqKv[static_cast<size_t>(b)];
        auto qd = extractDenseSlice(q, b, sQ);
        auto kd = extractDenseSlice(k, b, sKv);
        auto vd = extractDenseSlice(v, b, sKv);
        Tensor<float> oDense({1, numHeads, sQ, headDimV});
        Tensor<float> lseDense({1, numHeads, sQ, 1});

        CpuFpReferenceSdpa::forward<float, float, float, float, float>(qd,
                                                                       kd,
                                                                       vd,
                                                                       oDense,
                                                                       std::nullopt,
                                                                       /*attnMask=*/nullptr,
                                                                       leftBound,
                                                                       rightBound,
                                                                       topLeftAlignment,
                                                                       &lseDense);

        for(int64_t s = 0; s < sQ; ++s)
        {
            for(int64_t h = 0; h < numHeads; ++h)
            {
                EXPECT_NEAR(
                    lse.getHostValue(std::vector<int64_t>{b, h, s, 0}), lseDense(0, h, s, 0), 1e-4f)
                    << "LSE mismatch batch " << b << " token " << s << " head " << h;
                for(int64_t dv = 0; dv < headDimV; ++dv)
                {
                    EXPECT_NEAR(o.getHostValue(std::vector<int64_t>{b, h, s, dv}),
                                oDense(0, h, s, dv),
                                1e-4f)
                        << "output mismatch batch " << b << " token " << s << " head " << h
                        << " dv " << dv;
                }
            }
        }
    }
}

// fp8 (E4M3) + descale vs an independent dense oracle (dequantize-to-float). bf16 output.
void checkRaggedFp8VsDense(const std::vector<int64_t>& seqQ,
                           const std::vector<int64_t>& seqKv,
                           int64_t numHeads,
                           int64_t numHeadsKv,
                           int64_t headDim,
                           TensorBase<float>& descaleQ,
                           TensorBase<float>& descaleK,
                           TensorBase<float>& descaleV,
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

    std::vector<fp8_e4m3> qBack(static_cast<size_t>(totalQ * numHeads * headDim));
    std::vector<fp8_e4m3> kBack(static_cast<size_t>(totalKv * numHeadsKv * headDim));
    std::vector<fp8_e4m3> vBack(static_cast<size_t>(totalKv * numHeadsKv * headDim));
    std::vector<bfloat16> oBack(static_cast<size_t>(totalQ * numHeads * headDim), bfloat16(0.0f));
    fillPacked(qBack, 11);
    fillPacked(kBack, 22);
    fillPacked(vBack, 33);

    auto q = wrapRagged(qBack.data(), qDims, numHeads * headDim, cumQ);
    auto k = wrapRagged(kBack.data(), kvDims, numHeadsKv * headDim, cumKv);
    auto v = wrapRagged(vBack.data(), kvDims, numHeadsKv * headDim, cumKv);
    auto o = wrapRagged(oBack.data(), oDims, numHeads * headDim, cumQ);

    CpuFpReferenceSdpaRagged::forward<fp8_e4m3, fp8_e4m3, fp8_e4m3, bfloat16, float>(
        q,
        k,
        v,
        o,
        std::nullopt,
        leftBound,
        rightBound,
        topLeftAlignment,
        nullptr,
        &descaleQ,
        &descaleK,
        &descaleV);

    for(int64_t b = 0; b < batch; ++b)
    {
        const auto sQ = seqQ[static_cast<size_t>(b)];
        const auto sKv = seqKv[static_cast<size_t>(b)];
        auto qd = dequantDenseSlice(q, b, sQ, descaleQ);
        auto kd = dequantDenseSlice(k, b, sKv, descaleK);
        auto vd = dequantDenseSlice(v, b, sKv, descaleV);
        Tensor<bfloat16> oDense({1, numHeads, sQ, headDim});
        CpuFpReferenceSdpa::forward<float, float, float, bfloat16, float>(qd,
                                                                          kd,
                                                                          vd,
                                                                          oDense,
                                                                          std::nullopt,
                                                                          /*attnMask=*/nullptr,
                                                                          leftBound,
                                                                          rightBound,
                                                                          topLeftAlignment);

        for(int64_t s = 0; s < sQ; ++s)
        {
            for(int64_t h = 0; h < numHeads; ++h)
            {
                for(int64_t dv = 0; dv < headDim; ++dv)
                {
                    EXPECT_NEAR(
                        static_cast<float>(o.getHostValue(std::vector<int64_t>{b, h, s, dv})),
                        static_cast<float>(oDense(0, h, s, dv)),
                        2e-2f)
                        << "fp8 output mismatch batch " << b << " token " << s << " head " << h
                        << " dv " << dv;
                }
            }
        }
    }
}

} // namespace

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedBasicMha)
{
    checkRaggedVsDense({3, 5, 1}, {3, 5, 1}, 4, 4, 16, 16);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedCrossAttention)
{
    checkRaggedVsDense({2, 4, 3}, {5, 1, 6}, 2, 2, 16, 16);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedCausalTopLeft)
{
    checkRaggedVsDense({4, 7}, {4, 7}, 2, 2, 16, 16, -1, 0, true);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedCausalBottomRight)
{
    checkRaggedVsDense({3, 5}, {6, 8}, 2, 2, 16, 16, -1, 0, false);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedSlidingWindow)
{
    checkRaggedVsDense({8, 6}, {8, 6}, 2, 2, 16, 16, 2, 2, true);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedGqa)
{
    checkRaggedVsDense({5, 3}, {5, 3}, 8, 2, 16, 16);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedAsymmetricHeadDim)
{
    // hdim_q = 192, hdim_v = 128 (asymmetric head dims, as on the ASM v3 path).
    checkRaggedVsDense({3, 5}, {3, 5}, 2, 2, 192, 128);
}

// --- fp8 (E4M3) + descale vs the dense reference (dequantize-to-float oracle) ---

TEST(TestCpuFpReferenceSdpaRaggedFp8, RaggedPerTensorDescale)
{
    auto descaleQ = makeScalarDescale(0.5f);
    auto descaleK = makeScalarDescale(0.25f);
    auto descaleV = makeScalarDescale(2.0f);
    checkRaggedFp8VsDense({3, 5}, {3, 5}, 2, 2, 128, descaleQ, descaleK, descaleV, -1, -1, true);
}

TEST(TestCpuFpReferenceSdpaRaggedFp8, RaggedCausalGqaPerKvHeadDescale)
{
    const int64_t batch = 2;
    const int64_t numHeadsKv = 2; // GQA (numHeads = 4)
    auto descaleQ = makeScalarDescale(0.5f);
    auto descaleK = makePerHeadDescale(batch, numHeadsKv, 0.2f);
    auto descaleV = makePerHeadDescale(batch, numHeadsKv, 0.3f);
    checkRaggedFp8VsDense(
        {4, 6}, {4, 6}, 4, numHeadsKv, 128, descaleQ, descaleK, descaleV, -1, 0, true);
}

// --- Fully-masked branch: a zero-length-KV batch yields zero output and LSE = -inf ---

TEST(TestCpuFpReferenceSdpaRaggedFp32, ZeroLengthKvFullyMasked)
{
    const std::vector<int64_t> seqQ = {3, 2};
    const std::vector<int64_t> seqKv = {3, 0}; // batch 1: queries but no keys -> fully masked
    const int64_t numHeads = 2;
    const int64_t headDim = 16;
    const auto batch = static_cast<int64_t>(seqQ.size());
    const auto sMaxQ = maxOf(seqQ);
    const auto sMaxKv = maxOf(seqKv);
    const auto totalQ = sum(seqQ);
    const auto totalKv = sum(seqKv);
    const auto cumQ = cumTokens(seqQ);
    const auto cumKv = cumTokens(seqKv);

    const std::vector<int64_t> qDims = {batch, numHeads, sMaxQ, headDim};
    const std::vector<int64_t> kvDims = {batch, numHeads, sMaxKv, headDim};
    const std::vector<int64_t> lseDims = {batch, numHeads, sMaxQ, 1};

    std::vector<float> qBack(static_cast<size_t>(totalQ * numHeads * headDim));
    std::vector<float> kBack(static_cast<size_t>(totalKv * numHeads * headDim));
    std::vector<float> vBack(static_cast<size_t>(totalKv * numHeads * headDim));
    std::vector<float> oBack(static_cast<size_t>(totalQ * numHeads * headDim), -1.0f); // sentinel
    std::vector<float> lseBack(static_cast<size_t>(totalQ * numHeads), 123.0f); // sentinel
    fillPacked(qBack, 11);
    fillPacked(kBack, 22);
    fillPacked(vBack, 33);

    auto q = wrapRagged(qBack.data(), qDims, numHeads * headDim, cumQ);
    auto k = wrapRagged(kBack.data(), kvDims, numHeads * headDim, cumKv);
    auto v = wrapRagged(vBack.data(), kvDims, numHeads * headDim, cumKv);
    auto o = wrapRagged(oBack.data(), qDims, numHeads * headDim, cumQ);
    auto lse = wrapRagged(lseBack.data(), lseDims, numHeads, cumQ);

    CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
        q, k, v, o, std::nullopt, -1, -1, true, &lse);

    // Batch 1 has seqKv == 0: every query is fully masked -> output 0, LSE -inf.
    const int64_t b = 1;
    for(int64_t s = 0; s < seqQ[static_cast<size_t>(b)]; ++s)
    {
        for(int64_t h = 0; h < numHeads; ++h)
        {
            const float lseVal = lse.getHostValue(std::vector<int64_t>{b, h, s, 0});
            EXPECT_TRUE(std::isinf(lseVal) && lseVal < 0.0f)
                << "expected -inf LSE at fully-masked batch " << b << " token " << s << " head "
                << h;
            for(int64_t dv = 0; dv < headDim; ++dv)
            {
                EXPECT_EQ(o.getHostValue(std::vector<int64_t>{b, h, s, dv}), 0.0f)
                    << "expected zero output at fully-masked batch " << b << " token " << s;
            }
        }
    }
}

// --- Validation (negative) cases ---

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnNonRaggedInput)
{
    // Plain (non-ragged) tensors: raggedIterationInfo() is nullopt -> reference rejects them.
    Tensor<float> q({1, 2, 4, 16});
    Tensor<float> k({1, 2, 4, 16});
    Tensor<float> v({1, 2, 4, 16});
    Tensor<float> o({1, 2, 4, 16});
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnBadLseShape)
{
    const std::vector<int64_t> dims = {1, 2, 4, 16};
    const auto cum = cumTokens({4});
    std::vector<float> qB(static_cast<size_t>(2 * 4 * 16));
    std::vector<float> kB(qB.size());
    std::vector<float> vB(qB.size());
    std::vector<float> oB(qB.size());
    auto q = wrapRagged(qB.data(), dims, 2 * 16, cum);
    auto k = wrapRagged(kB.data(), dims, 2 * 16, cum);
    auto v = wrapRagged(vB.data(), dims, 2 * 16, cum);
    auto o = wrapRagged(oB.data(), dims, 2 * 16, cum);

    Tensor<float> badLse({1, 2, 4, 2}); // last dim must be 1
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
                     q, k, v, o, std::nullopt, -1, -1, true, &badLse)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnBadDescaleShape)
{
    const std::vector<int64_t> dims = {1, 2, 4, 16};
    const auto cum = cumTokens({4});
    std::vector<float> qB(static_cast<size_t>(2 * 4 * 16));
    std::vector<float> kB(qB.size());
    std::vector<float> vB(qB.size());
    std::vector<float> oB(qB.size());
    auto q = wrapRagged(qB.data(), dims, 2 * 16, cum);
    auto k = wrapRagged(kB.data(), dims, 2 * 16, cum);
    auto v = wrapRagged(vB.data(), dims, 2 * 16, cum);
    auto o = wrapRagged(oB.data(), dims, 2 * 16, cum);

    Tensor<float> badDescale({1, 3, 1, 1}); // heads (3) != numHeads (2)
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
                     q, k, v, o, std::nullopt, -1, -1, true, nullptr, &badDescale)),
                 std::invalid_argument);
}
