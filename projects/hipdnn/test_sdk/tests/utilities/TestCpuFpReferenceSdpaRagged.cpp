// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Checks CpuFpReferenceSdpaRagged against the dense CpuFpReferenceSdpa. Each batch of the ragged
// inputs is copied to a dense [1, H, seqlen_b, D] tensor and run through the dense reference.
// Chain: dense CPU (trusted) -> ragged CPU (here) -> ragged GPU (TestGpuFpReferenceSdpaRagged).
//
// The fp8/descale, LSE and fully-masked paths are covered here too, because the GPU suite is
// skipped without a device (as in the coverage lane). fp8 is checked against a dequantized
// dense run.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
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

// Deterministic fill in [-1, 1). A local LCG avoids <random> drift between builds.
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

// Wraps a borrowed packed buffer as a ragged tensor ([B, S, H, D], BSHD_SEQ_AXIS).
template <typename T>
ShallowRaggedTensor<T> wrapRagged(T* buf,
                                  const std::vector<int64_t>& dims,
                                  int64_t seqStride,
                                  const std::vector<int64_t>& cum)
{
    return ShallowRaggedTensor<T>(
        buf, dims, raggedStrides(dims), BSHD_SEQ_AXIS, makeRaggedOffsetAux(cum, seqStride));
}

// Valid ragged tensor over a caller-owned buffer sized for the packed tokens. The negative tests
// use it so that only the one thing under test is wrong.
ShallowRaggedTensor<float> makeValidRagged(std::vector<float>& backing,
                                           const std::vector<int64_t>& dims,
                                           const std::vector<int64_t>& seqLens)
{
    const int64_t seqStride = raggedHeads(dims) * dims[3]; // H * D
    backing.assign(static_cast<size_t>(sum(seqLens) * seqStride), 0.0f);
    return wrapRagged(backing.data(), dims, seqStride, cumTokens(seqLens));
}

Tensor<float> makeScalarDescale(float value)
{
    Tensor<float> d({1});
    d.memory().hostData()[0] = value;
    d.memory().markHostModified();
    return d;
}

// Per-KV-head descale [B, heads, 1, 1], distinct per (b, head).
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

// Descale for (batch, head), from a scalar [1] or a [B, heads, 1, 1] tensor.
float descaleValue(TensorBase<float>& descale, int64_t b, int64_t head)
{
    if(descale.elementCount() == 1)
    {
        return descale.getHostValue(std::vector<int64_t>{0});
    }
    return descale.getHostValue(std::vector<int64_t>{b, head, 0, 0});
}

// Copies batch b of a ragged tensor into a dense [1, heads, seqLen, dim] tensor.
template <typename T>
Tensor<float> extractDenseSlice(TensorBase<T>& ragged, int64_t b, int64_t seqLen)
{
    const auto heads = raggedHeads(ragged.dims());
    const auto dim = ragged.dims()[3];
    Tensor<float> dense({1, heads, seqLen, dim});
    for(int64_t h = 0; h < heads; ++h)
    {
        for(int64_t s = 0; s < seqLen; ++s)
        {
            for(int64_t d = 0; d < dim; ++d)
            {
                dense(0, h, s, d)
                    = static_cast<float>(ragged.getHostValue(raggedIndex(b, s, h, d)));
            }
        }
    }
    dense.memory().markHostModified();
    return dense;
}

// Like extractDenseSlice, but dequantizes fp8 and multiplies in the descale. Scaling the inputs is
// equivalent to the reference's score *= dQ * dK and out *= dV. Head h reads descale head
// h / headsPerDescaleHead (H_q / H_kv for Q, 1 for K/V).
template <typename FP8>
Tensor<float> dequantDenseSlice(TensorBase<FP8>& ragged,
                                int64_t b,
                                int64_t seqLen,
                                TensorBase<float>& descale,
                                int64_t headsPerDescaleHead)
{
    const auto heads = raggedHeads(ragged.dims());
    const auto dim = ragged.dims()[3];
    Tensor<float> dense({1, heads, seqLen, dim});
    for(int64_t h = 0; h < heads; ++h)
    {
        const float dsc = descaleValue(descale, b, h / headsPerDescaleHead);
        for(int64_t s = 0; s < seqLen; ++s)
        {
            for(int64_t d = 0; d < dim; ++d)
            {
                dense(0, h, s, d)
                    = static_cast<float>(ragged.getHostValue(raggedIndex(b, s, h, d))) * dsc;
            }
        }
    }
    dense.memory().markHostModified();
    return dense;
}

// Runs the ragged reference (with a ragged LSE) and checks each batch's output and LSE against
// the dense reference.
void checkRaggedVsDense(const std::vector<int64_t>& seqQ,
                        const std::vector<int64_t>& seqKv,
                        int64_t numHeads,
                        int64_t numHeadsKv,
                        int64_t headDim,
                        int64_t headDimV,
                        int64_t leftBound = -1,
                        int64_t rightBound = -1,
                        bool topLeftAlignment = true,
                        std::optional<float> attnScale = std::nullopt)
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
    const std::vector<int64_t> kDims = raggedDims(batch, sMaxKv, numHeadsKv, headDim);
    const std::vector<int64_t> vDims = raggedDims(batch, sMaxKv, numHeadsKv, headDimV);
    const std::vector<int64_t> oDims = raggedDims(batch, sMaxQ, numHeads, headDimV);
    const std::vector<int64_t> lseDims = raggedDims(batch, sMaxQ, numHeads, 1);

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
        q, k, v, o, attnScale, leftBound, rightBound, topLeftAlignment, &lse);

    for(int64_t b = 0; b < batch; ++b)
    {
        const auto sQ = seqQ[static_cast<size_t>(b)];
        const auto sKv = seqKv[static_cast<size_t>(b)];
        if(sQ == 0)
        {
            continue; // nothing to check, and the dense reference rejects empty dims
        }
        auto qd = extractDenseSlice(q, b, sQ);
        auto kd = extractDenseSlice(k, b, sKv);
        auto vd = extractDenseSlice(v, b, sKv);
        Tensor<float> oDense({1, numHeads, sQ, headDimV});
        Tensor<float> lseDense({1, numHeads, sQ, 1});

        CpuFpReferenceSdpa::forward<float, float, float, float, float>(qd,
                                                                       kd,
                                                                       vd,
                                                                       oDense,
                                                                       attnScale,
                                                                       /*attnMask=*/nullptr,
                                                                       leftBound,
                                                                       rightBound,
                                                                       topLeftAlignment,
                                                                       &lseDense);

        for(int64_t s = 0; s < sQ; ++s)
        {
            for(int64_t h = 0; h < numHeads; ++h)
            {
                // A fully masked row has LSE = -inf, which EXPECT_NEAR cannot compare.
                const float lseRagged = lse.getHostValue(raggedIndex(b, s, h, 0));
                const float lseExpected = lseDense(0, h, s, 0);
                if(std::isinf(lseExpected))
                {
                    EXPECT_EQ(lseRagged, lseExpected)
                        << "LSE mismatch batch " << b << " token " << s << " head " << h;
                }
                else
                {
                    EXPECT_NEAR(lseRagged, lseExpected, 1e-4f)
                        << "LSE mismatch batch " << b << " token " << s << " head " << h;
                }
                for(int64_t dv = 0; dv < headDimV; ++dv)
                {
                    EXPECT_NEAR(
                        o.getHostValue(raggedIndex(b, s, h, dv)), oDense(0, h, s, dv), 1e-4f)
                        << "output mismatch batch " << b << " token " << s << " head " << h
                        << " dv " << dv;
                }
            }
        }
    }
}

// fp8 E4M3 inputs with descales and bf16 output, checked against a dequantized dense run.
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

    const std::vector<int64_t> qDims = raggedDims(batch, sMaxQ, numHeads, headDim);
    const std::vector<int64_t> kvDims = raggedDims(batch, sMaxKv, numHeadsKv, headDim);
    const std::vector<int64_t> oDims = raggedDims(batch, sMaxQ, numHeads, headDim);

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
        auto qd = dequantDenseSlice(q, b, sQ, descaleQ, numHeads / numHeadsKv);
        auto kd = dequantDenseSlice(k, b, sKv, descaleK, 1);
        auto vd = dequantDenseSlice(v, b, sKv, descaleV, 1);
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
                    EXPECT_NEAR(static_cast<float>(o.getHostValue(raggedIndex(b, s, h, dv))),
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
    // Asymmetric head dims (192 / 128), as on the ASM v3 path.
    checkRaggedVsDense({3, 5}, {3, 5}, 2, 2, 192, 128);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedExplicitAttnScale)
{
    // Explicit attnScale instead of the default 1.0. Both references get the same value.
    checkRaggedVsDense({4, 6}, {4, 6}, 2, 2, 16, 16, -1, -1, true, /*attnScale=*/0.125f);
}

// An absent scale means 1.0, as in cuDNN and CpuFpReferenceSdpa: the output must equal an
// explicit 1.0 bit for bit, and differ from the old 1/sqrt(D) default.
TEST(TestCpuFpReferenceSdpaRaggedFp32, AbsentAttnScaleIsOne)
{
    const std::vector<int64_t> seqLens = {3, 5};
    const auto dims = raggedDims(2, 5, 2, 16);
    const int64_t tokenWidth = int64_t{2} * 16; // H * D
    const auto cum = cumTokens(seqLens);
    const auto count = static_cast<size_t>(cum.back() * tokenWidth);
    std::vector<float> qB(count);
    std::vector<float> kB(count);
    std::vector<float> vB(count);
    fillPacked(qB, 11);
    fillPacked(kB, 22);
    fillPacked(vB, 33);
    auto q = wrapRagged(qB.data(), dims, tokenWidth, cum);
    auto k = wrapRagged(kB.data(), dims, tokenWidth, cum);
    auto v = wrapRagged(vB.data(), dims, tokenWidth, cum);

    const auto run = [&](std::optional<float> scale) {
        std::vector<float> oB(count, 0.0f);
        auto o = wrapRagged(oB.data(), dims, tokenWidth, cum);
        CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o, scale);
        return oB;
    };
    const auto absent = run(std::nullopt);
    EXPECT_EQ(absent, run(1.0f));
    EXPECT_NE(absent, run(0.25f));
}

// Pins the RFC-0014 layout with literals rather than the raggedDims helpers: dims [B, S, H, D],
// sequence at axis 1, contiguous strides, and element (token t, head h, dim d) stored at
// t * H * D + h * D + d. Each batch is copied straight from the packed buffers into the dense
// reference's [1, H, S, D] layout.
TEST(TestCpuFpReferenceSdpaRaggedFp32, RfcLayoutLiteralShape)
{
    const std::vector<int64_t> seqLens = {3, 5};
    const int64_t heads = 2;
    const int64_t dim = 4;
    const std::vector<int64_t> dims = {2, 5, 2, 4};
    const std::vector<int64_t> strides = {40, 8, 4, 1};
    const int64_t tokenWidth = heads * dim;
    const auto cum = cumTokens(seqLens);
    const auto packedCount = static_cast<size_t>(cum.back() * tokenWidth);

    std::vector<float> qB(packedCount);
    std::vector<float> kB(packedCount);
    std::vector<float> vB(packedCount);
    std::vector<float> oB(packedCount, 0.0f);
    fillPacked(qB, 11);
    fillPacked(kB, 22);
    fillPacked(vB, 33);
    const auto wrap = [&](std::vector<float>& buf) {
        return ShallowRaggedTensor<float>(
            buf.data(), dims, strides, /*seqAxis=*/1, makeRaggedOffsetAux(cum, tokenWidth));
    };
    auto q = wrap(qB);
    auto k = wrap(kB);
    auto v = wrap(vB);
    auto o = wrap(oB);

    CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o);

    const auto packedAt = [&](int64_t token, int64_t h, int64_t d) {
        return static_cast<size_t>(token * tokenWidth + h * dim + d);
    };
    for(int64_t b = 0; b < 2; ++b)
    {
        const auto len = seqLens[static_cast<size_t>(b)];
        const auto base = cum[static_cast<size_t>(b)];
        Tensor<float> qd({1, heads, len, dim});
        Tensor<float> kd({1, heads, len, dim});
        Tensor<float> vd({1, heads, len, dim});
        for(int64_t h = 0; h < heads; ++h)
        {
            for(int64_t s = 0; s < len; ++s)
            {
                for(int64_t d = 0; d < dim; ++d)
                {
                    qd(0, h, s, d) = qB[packedAt(base + s, h, d)];
                    kd(0, h, s, d) = kB[packedAt(base + s, h, d)];
                    vd(0, h, s, d) = vB[packedAt(base + s, h, d)];
                }
            }
        }
        qd.memory().markHostModified();
        kd.memory().markHostModified();
        vd.memory().markHostModified();
        Tensor<float> oDense({1, heads, len, dim});
        CpuFpReferenceSdpa::forward<float, float, float, float, float>(qd, kd, vd, oDense);

        for(int64_t h = 0; h < heads; ++h)
        {
            for(int64_t s = 0; s < len; ++s)
            {
                for(int64_t d = 0; d < dim; ++d)
                {
                    EXPECT_NEAR(oB[packedAt(base + s, h, d)], oDense(0, h, s, d), 1e-4f)
                        << "batch " << b << " token " << s << " head " << h << " dim " << d;
                }
            }
        }
    }
}

// --- fp8 (E4M3) + descale vs a dequantized dense reference ---

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

// Per-KV-head descales on Q, K and V under GQA (AITER's [B, H_kv] shape). Distinct Q values catch
// a Q descale indexed by query head instead of KV head.
TEST(TestCpuFpReferenceSdpaRaggedFp8, RaggedGqaPerKvHeadDescaleQkv)
{
    const int64_t batch = 2;
    const int64_t numHeadsKv = 2; // GQA (numHeads = 4)
    auto descaleQ = makePerHeadDescale(batch, numHeadsKv, 0.4f);
    auto descaleK = makePerHeadDescale(batch, numHeadsKv, 0.2f);
    auto descaleV = makePerHeadDescale(batch, numHeadsKv, 0.3f);
    checkRaggedFp8VsDense(
        {4, 6}, {5, 3}, 4, numHeadsKv, 128, descaleQ, descaleK, descaleV, -1, -1, true);
}

// --- Dense LSE ([B, Sq_max, H, 1], the frontend's default stats layout) ---
// A dense LSE must match the ragged LSE on valid rows and leave padding rows untouched. The GPU
// reference follows the same contract.
TEST(TestCpuFpReferenceSdpaRaggedFp32, DenseLseMatchesRaggedLse)
{
    const std::vector<int64_t> seqQ = {3, 5, 1};
    const std::vector<int64_t> seqKv = {4, 2, 6};
    const int64_t numHeads = 2;
    const int64_t headDim = 16;
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

    std::vector<float> qBack(static_cast<size_t>(totalQ * numHeads * headDim));
    std::vector<float> kBack(static_cast<size_t>(totalKv * numHeads * headDim));
    std::vector<float> vBack(static_cast<size_t>(totalKv * numHeads * headDim));
    std::vector<float> oBack(qBack.size(), 0.0f);
    std::vector<float> lseRaggedBack(static_cast<size_t>(totalQ * numHeads), 0.0f);
    fillPacked(qBack, 11);
    fillPacked(kBack, 22);
    fillPacked(vBack, 33);

    auto q = wrapRagged(qBack.data(), qDims, numHeads * headDim, cumQ);
    auto k = wrapRagged(kBack.data(), kvDims, numHeads * headDim, cumKv);
    auto v = wrapRagged(vBack.data(), kvDims, numHeads * headDim, cumKv);
    auto o = wrapRagged(oBack.data(), qDims, numHeads * headDim, cumQ);
    auto lseRagged = wrapRagged(lseRaggedBack.data(), lseDims, numHeads, cumQ);

    constexpr float SENTINEL = -99.0f;
    Tensor<float> lseDense(lseDims);
    lseDense.fillWithValue(SENTINEL);

    CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
        q, k, v, o, std::nullopt, -1, -1, true, &lseRagged);
    CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
        q, k, v, o, std::nullopt, -1, -1, true, &lseDense);

    for(int64_t b = 0; b < batch; ++b)
    {
        for(int64_t h = 0; h < numHeads; ++h)
        {
            for(int64_t s = 0; s < sMaxQ; ++s)
            {
                const float dense = lseDense.getHostValue(raggedIndex(b, s, h, 0));
                if(s < seqQ[static_cast<size_t>(b)])
                {
                    EXPECT_EQ(dense, lseRagged.getHostValue(raggedIndex(b, s, h, 0)))
                        << "dense LSE mismatch batch " << b << " head " << h << " token " << s;
                }
                else
                {
                    EXPECT_EQ(dense, SENTINEL)
                        << "padding row written batch " << b << " head " << h << " token " << s;
                }
            }
        }
    }
}

// --- Fully-masked rows: a batch with no keys gives zero output and LSE = -inf ---

TEST(TestCpuFpReferenceSdpaRaggedFp32, ZeroLengthKvFullyMasked)
{
    const std::vector<int64_t> seqQ = {3, 2};
    const std::vector<int64_t> seqKv = {3, 0}; // batch 1 has queries but no keys
    const int64_t numHeads = 2;
    const int64_t headDim = 16;
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

    const int64_t b = 1;
    for(int64_t s = 0; s < seqQ[static_cast<size_t>(b)]; ++s)
    {
        for(int64_t h = 0; h < numHeads; ++h)
        {
            const float lseVal = lse.getHostValue(raggedIndex(b, s, h, 0));
            EXPECT_TRUE(std::isinf(lseVal) && lseVal < 0.0f)
                << "expected -inf LSE at fully-masked batch " << b << " token " << s << " head "
                << h;
            for(int64_t dv = 0; dv < headDim; ++dv)
            {
                EXPECT_EQ(o.getHostValue(raggedIndex(b, s, h, dv)), 0.0f)
                    << "expected zero output at fully-masked batch " << b << " token " << s;
            }
        }
    }
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ZeroLengthQBatch)
{
    checkRaggedVsDense({3, 0, 4}, {3, 2, 4}, 2, 2, 16, 16);
}

// No query tokens at all: nothing is written.
TEST(TestCpuFpReferenceSdpaRaggedFp32, AllQueriesEmpty)
{
    constexpr float SENTINEL = -99.0f;
    const std::vector<int64_t> dims = raggedDims(2, 2, 1, 16);
    std::vector<float> qB(32, 1.0f);
    std::vector<float> kB(48, 1.0f);
    std::vector<float> vB(48, 1.0f);
    std::vector<float> oB(32, SENTINEL);
    auto q = wrapRagged(qB.data(), dims, 16, {0, 0, 0});
    auto k = wrapRagged(kB.data(), dims, 16, {0, 2, 3});
    auto v = wrapRagged(vB.data(), dims, 16, {0, 2, 3});
    auto o = wrapRagged(oB.data(), dims, 16, {0, 0, 0});
    Tensor<float> lse(raggedDims(2, 2, 1, 1));
    lse.fillWithValue(SENTINEL);

    EXPECT_NO_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
        q, k, v, o, std::nullopt, -1, -1, true, &lse)));

    for(size_t i = 0; i < oB.size(); ++i)
    {
        EXPECT_EQ(oB[i], SENTINEL) << "output written at element " << i;
    }
    const auto* lp = lse.memory().hostData();
    for(size_t i = 0; i < lse.elementCount(); ++i)
    {
        EXPECT_EQ(lp[i], SENTINEL) << "LSE written at element " << i;
    }
}

// --- Validation (negative) cases ---

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnNonRaggedInput)
{
    // Plain tensors have no raggedIterationInfo(), so they are rejected.
    Tensor<float> q(raggedDims(1, 4, 2, 16));
    Tensor<float> k(raggedDims(1, 4, 2, 16));
    Tensor<float> v(raggedDims(1, 4, 2, 16));
    Tensor<float> o(raggedDims(1, 4, 2, 16));
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnBadLseShape)
{
    const std::vector<int64_t> dims = raggedDims(1, 4, 2, 16);
    const auto cum = cumTokens({4});
    std::vector<float> qB(static_cast<size_t>(2 * 4 * 16));
    std::vector<float> kB(qB.size());
    std::vector<float> vB(qB.size());
    std::vector<float> oB(qB.size());
    auto q = wrapRagged(qB.data(), dims, int64_t{2} * 16, cum);
    auto k = wrapRagged(kB.data(), dims, int64_t{2} * 16, cum);
    auto v = wrapRagged(vB.data(), dims, int64_t{2} * 16, cum);
    auto o = wrapRagged(oB.data(), dims, int64_t{2} * 16, cum);

    Tensor<float> badLse(raggedDims(1, 4, 2, 2)); // last dim must be 1
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
                     q, k, v, o, std::nullopt, -1, -1, true, &badLse)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnBadDescaleShape)
{
    const std::vector<int64_t> dims = raggedDims(1, 4, 2, 16);
    const auto cum = cumTokens({4});
    std::vector<float> qB(static_cast<size_t>(2 * 4 * 16));
    std::vector<float> kB(qB.size());
    std::vector<float> vB(qB.size());
    std::vector<float> oB(qB.size());
    auto q = wrapRagged(qB.data(), dims, int64_t{2} * 16, cum);
    auto k = wrapRagged(kB.data(), dims, int64_t{2} * 16, cum);
    auto v = wrapRagged(vB.data(), dims, int64_t{2} * 16, cum);
    auto o = wrapRagged(oB.data(), dims, int64_t{2} * 16, cum);

    Tensor<float> badDescale({1, 3, 1, 1}); // heads (3) != H_kv (2)
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
                     q, k, v, o, std::nullopt, -1, -1, true, nullptr, &badDescale)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnPerQueryHeadQDescaleUnderGqa)
{
    // Q descale is per KV head, so a [B, H_q, 1, 1] Q descale is rejected under GQA.
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, raggedDims(1, 4, 4, 16), {4});
    auto k = makeValidRagged(kB, raggedDims(1, 4, 2, 16), {4});
    auto v = makeValidRagged(vB, raggedDims(1, 4, 2, 16), {4});
    auto o = makeValidRagged(oB, raggedDims(1, 4, 4, 16), {4});

    auto perQueryHead = makePerHeadDescale(1, 4, 0.5f);
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
                     q, k, v, o, std::nullopt, -1, -1, true, nullptr, &perQueryHead)),
                 std::invalid_argument);
}

// --- validateInput() cases: one dimension wrong, everything else valid ---

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnBatchMismatch)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, raggedDims(1, 4, 2, 16), {4});
    auto k = makeValidRagged(kB, raggedDims(2, 4, 2, 16), {4, 4}); // batch 2 != q batch 1
    auto v = makeValidRagged(vB, raggedDims(1, 4, 2, 16), {4});
    auto o = makeValidRagged(oB, raggedDims(1, 4, 2, 16), {4});
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnQkHeadDimMismatch)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, raggedDims(1, 4, 2, 16), {4});
    auto k = makeValidRagged(kB, raggedDims(1, 4, 2, 32), {4}); // K head_dim 32 != Q head_dim 16
    auto v = makeValidRagged(vB, raggedDims(1, 4, 2, 16), {4});
    auto o = makeValidRagged(oB, raggedDims(1, 4, 2, 16), {4});
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnKvSeqExtentMismatch)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, raggedDims(1, 4, 2, 16), {4});
    auto k = makeValidRagged(kB, raggedDims(1, 4, 2, 16), {4});
    auto v = makeValidRagged(vB, raggedDims(1, 6, 2, 16), {6}); // V S_max 6 != K S_max 4
    auto o = makeValidRagged(oB, raggedDims(1, 4, 2, 16), {4});
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

// --- Tensors sharing a packing must describe the same per-batch sequence lengths ---

// Reviewer repro: K lengths {2, 1} and V lengths {1, 2} with the same S_max. Without the check,
// batch 0 averaged in batch 1's V row and returned {15, 20}.
TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnKvSequenceLengthMismatch)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, raggedDims(2, 1, 1, 1), {1, 1});
    auto k = makeValidRagged(kB, raggedDims(2, 2, 1, 1), {2, 1});
    auto v = makeValidRagged(vB, raggedDims(2, 2, 1, 1), {1, 2});
    auto o = makeValidRagged(oB, raggedDims(2, 1, 1, 1), {1, 1});
    vB[0] = 10.0f;
    vB[1] = 20.0f;
    vB[2] = 30.0f;
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnQoSequenceLengthMismatch)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, raggedDims(2, 2, 2, 16), {2, 1});
    auto k = makeValidRagged(kB, raggedDims(2, 2, 2, 16), {2, 2});
    auto v = makeValidRagged(vB, raggedDims(2, 2, 2, 16), {2, 2});
    auto o = makeValidRagged(oB, raggedDims(2, 2, 2, 16), {1, 2}); // O lengths != Q lengths
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnRaggedLseSequenceLengthMismatch)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    std::vector<float> lseB;
    auto q = makeValidRagged(qB, raggedDims(2, 2, 1, 16), {2, 1});
    auto k = makeValidRagged(kB, raggedDims(2, 2, 1, 16), {2, 2});
    auto v = makeValidRagged(vB, raggedDims(2, 2, 1, 16), {2, 2});
    auto o = makeValidRagged(oB, raggedDims(2, 2, 1, 16), {2, 1});
    auto lse = makeValidRagged(lseB, raggedDims(2, 2, 1, 1), {1, 2}); // LSE lengths != Q lengths
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
                     q, k, v, o, std::nullopt, -1, -1, true, &lse)),
                 std::invalid_argument);
}

// A dense (non-ragged) output cannot follow the packed Q layout.
TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnNonRaggedOutput)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    auto q = makeValidRagged(qB, raggedDims(1, 4, 2, 16), {4});
    auto k = makeValidRagged(kB, raggedDims(1, 4, 2, 16), {4});
    auto v = makeValidRagged(vB, raggedDims(1, 4, 2, 16), {4});
    Tensor<float> o(raggedDims(1, 4, 2, 16));
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

namespace
{

// Rewrites a ragged tensor's offset table after construction. The tensor only checked the
// original table, so the reference must catch a bad one itself.
void setTokenOffsets(ITensor& aux, const std::vector<int64_t>& tokens, int64_t seqStride)
{
    auto& offsets = static_cast<Tensor<int32_t>&>(aux);
    for(size_t i = 0; i < tokens.size(); ++i)
    {
        offsets.setHostValue(
            static_cast<int32_t>(tokens[i] * seqStride), static_cast<int64_t>(i), 0, 0, 0);
    }
}

// Builds valid B = 2, S_max = 2, H * D = 16 ragged inputs, then rewrites Q's and O's offsets to
// qTokens * offsetUnit (offsetUnit 1 writes raw element offsets). Returns whether forward() threw
// std::invalid_argument.
bool throwsOnEditedQTokens(const std::vector<int64_t>& qTokens, int64_t offsetUnit = 16)
{
    const std::vector<int64_t> dims = raggedDims(2, 2, 1, 16);
    const std::vector<int64_t> valid = {0, 2, 4};
    std::vector<float> qB(64, 0.0f);
    std::vector<float> kB(64, 0.0f);
    std::vector<float> vB(64, 1.0f);
    std::vector<float> oB(64, 0.0f);
    auto qAux = makeRaggedOffsetAux(valid, 16);
    auto oAux = makeRaggedOffsetAux(valid, 16);
    ShallowRaggedTensor<float> q(qB.data(), dims, raggedStrides(dims), BSHD_SEQ_AXIS, qAux);
    auto k = wrapRagged(kB.data(), dims, 16, valid);
    auto v = wrapRagged(vB.data(), dims, 16, valid);
    ShallowRaggedTensor<float> o(oB.data(), dims, raggedStrides(dims), BSHD_SEQ_AXIS, oAux);
    setTokenOffsets(*qAux, qTokens, offsetUnit);
    setTokenOffsets(*oAux, qTokens, offsetUnit);
    try
    {
        CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o);
    }
    catch(const std::invalid_argument&)
    {
        return true;
    }
    return false;
}

} // namespace

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnBadOffsetTable)
{
    EXPECT_TRUE(throwsOnEditedQTokens({1, 2, 4})) << "ragged_offset[0] != 0 accepted";
    EXPECT_TRUE(throwsOnEditedQTokens({0, 3, 4})) << "batch longer than S_max accepted";
    EXPECT_TRUE(throwsOnEditedQTokens({0, 2, 1})) << "decreasing offsets accepted";
    EXPECT_TRUE(throwsOnEditedQTokens({0, 17, 32}, /*offsetUnit=*/1))
        << "offset that is not a whole token accepted";
    EXPECT_FALSE(throwsOnEditedQTokens({0, 2, 3}));
}

// A dense LSE with a smaller Sq than Q would take rows from the next batch.
TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnLseShorterThanQ)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, raggedDims(2, 2, 1, 16), {2, 1});
    auto k = makeValidRagged(kB, raggedDims(2, 2, 1, 16), {2, 2});
    auto v = makeValidRagged(vB, raggedDims(2, 2, 1, 16), {2, 2});
    auto o = makeValidRagged(oB, raggedDims(2, 2, 1, 16), {2, 1});
    Tensor<float> lse(raggedDims(2, 1, 1, 1));
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
                     q, k, v, o, std::nullopt, -1, -1, true, &lse)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnNonDivisibleHeads)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, raggedDims(1, 4, 4, 16), {4});
    auto k = makeValidRagged(kB, raggedDims(1, 4, 3, 16), {4}); // 4 % 3 != 0
    auto v = makeValidRagged(vB, raggedDims(1, 4, 3, 16), {4});
    auto o = makeValidRagged(oB, raggedDims(1, 4, 4, 16), {4});
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnBadOutputShape)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, raggedDims(1, 4, 2, 16), {4});
    auto k = makeValidRagged(kB, raggedDims(1, 4, 2, 16), {4});
    auto v = makeValidRagged(vB, raggedDims(1, 4, 2, 16), {4});
    auto o = makeValidRagged(oB, raggedDims(1, 4, 2, 32), {4}); // O head_dim 32 != V head_dim 16
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

// A pre-RFC [B, H, S, D] tensor (BSHD strides, ragged along axis 2) is rejected. With H == S and
// a full batch the offsets and S_max checks pass, so only the layout check can catch it.
TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnHeadsBeforeSequenceLayout)
{
    const std::vector<int64_t> dims = {1, 4, 4, 16}; // [B, H, S, D]
    const std::vector<int64_t> strides = {256, 16, 64, 1};
    const auto cum = cumTokens({4});
    std::vector<float> qB(256, 0.5f);
    std::vector<float> kB(256, 0.5f);
    std::vector<float> vB(256, 1.0f);
    std::vector<float> oB(256, 0.0f);
    const auto wrap = [&](std::vector<float>& buf) {
        return ShallowRaggedTensor<float>(
            buf.data(), dims, strides, /*seqAxis=*/2, makeRaggedOffsetAux(cum, 64));
    };
    auto q = wrap(qB);
    auto k = wrap(kB);
    auto v = wrap(vB);
    auto o = wrap(oB);
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

// The layout check itself: the RFC-0014 order passes; heads before sequence, or a seq stride too
// small for one token, fail.
TEST(TestCpuFpReferenceSdpaRaggedFp32, TokenMajorLayoutCheck)
{
    using hipdnn_test_sdk::detail::isTokenMajorRaggedLayout;
    EXPECT_TRUE(isTokenMajorRaggedLayout({2, 5, 3, 8}, {120, 24, 8, 1}));
    EXPECT_TRUE(isTokenMajorRaggedLayout({2, 5, 3, 8}, {160, 32, 8, 1})); // padded token
    EXPECT_FALSE(isTokenMajorRaggedLayout({2, 3, 5, 8}, {120, 8, 24, 1})); // [B, H, S, D]
    EXPECT_FALSE(isTokenMajorRaggedLayout({2, 5, 3, 8}, {120, 16, 8, 1})); // tokens overlap
    EXPECT_FALSE(isTokenMajorRaggedLayout({2, 5, 3}, {15, 3, 1})); // not rank 4
}

// AITER's cu_seqlens form: one token table shared by Q/O and one by K/V, each tensor scaling it
// by its own seq stride through ragged_offset_multiplier. GQA and D != Dv make every width
// different. The result must equal the element-offset run bit for bit.
TEST(TestCpuFpReferenceSdpaRaggedFp32, TokenOffsetsMatchElementOffsets)
{
    const std::vector<int64_t> seqQ = {3, 0, 4};
    const std::vector<int64_t> seqKv = {2, 5, 1};
    const int64_t heads = 4;
    const int64_t headsKv = 2;
    const int64_t dim = 8;
    const int64_t dimV = 4;
    const auto batch = static_cast<int64_t>(seqQ.size());
    const auto cumQ = cumTokens(seqQ);
    const auto cumKv = cumTokens(seqKv);
    const auto qDims = raggedDims(batch, maxOf(seqQ), heads, dim);
    const auto kDims = raggedDims(batch, maxOf(seqKv), headsKv, dim);
    const auto vDims = raggedDims(batch, maxOf(seqKv), headsKv, dimV);
    const auto oDims = raggedDims(batch, maxOf(seqQ), heads, dimV);
    const auto lseDims = raggedDims(batch, maxOf(seqQ), heads, 1);

    std::vector<float> qB(static_cast<size_t>(cumQ.back() * heads * dim));
    std::vector<float> kB(static_cast<size_t>(cumKv.back() * headsKv * dim));
    std::vector<float> vB(static_cast<size_t>(cumKv.back() * headsKv * dimV));
    fillPacked(qB, 11);
    fillPacked(kB, 22);
    fillPacked(vB, 33);

    std::vector<float> oElem(static_cast<size_t>(cumQ.back() * heads * dimV), 0.0f);
    std::vector<float> lseElem(static_cast<size_t>(cumQ.back() * heads), 0.0f);
    {
        auto q = wrapRagged(qB.data(), qDims, heads * dim, cumQ);
        auto k = wrapRagged(kB.data(), kDims, headsKv * dim, cumKv);
        auto v = wrapRagged(vB.data(), vDims, headsKv * dimV, cumKv);
        auto o = wrapRagged(oElem.data(), oDims, heads * dimV, cumQ);
        auto lse = wrapRagged(lseElem.data(), lseDims, heads, cumQ);
        CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
            q, k, v, o, std::nullopt, -1, 0, false, &lse);
    }

    std::vector<float> oTok(oElem.size(), 0.0f);
    std::vector<float> lseTok(lseElem.size(), 0.0f);
    {
        auto qoTable = makeRaggedOffsetAux(cumQ, 1);
        auto kvTable = makeRaggedOffsetAux(cumKv, 1);
        const auto wrap = [](float* buf,
                             const std::vector<int64_t>& dims,
                             const std::shared_ptr<ITensor>& table,
                             int64_t multiplier) {
            return ShallowRaggedTensor<float>(
                buf, dims, raggedStrides(dims), BSHD_SEQ_AXIS, table, std::nullopt, multiplier);
        };
        auto q = wrap(qB.data(), qDims, qoTable, heads * dim);
        auto k = wrap(kB.data(), kDims, kvTable, headsKv * dim);
        auto v = wrap(vB.data(), vDims, kvTable, headsKv * dimV);
        auto o = wrap(oTok.data(), oDims, qoTable, heads * dimV);
        auto lse = wrap(lseTok.data(), lseDims, qoTable, heads);
        CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
            q, k, v, o, std::nullopt, -1, 0, false, &lse);
    }

    EXPECT_EQ(oTok, oElem);
    EXPECT_EQ(lseTok, lseElem);
}

// --- Edge cases checked against the dense reference ---

// Empty Q batches at the front, at the back and back to back.
TEST(TestCpuFpReferenceSdpaRaggedFp32, LeadingTrailingAndConsecutiveEmptyQBatches)
{
    checkRaggedVsDense({0, 3, 0, 0, 2, 0}, {2, 3, 1, 4, 2, 3}, 2, 2, 16, 16);
}

// Bottom-right causal with Sq > Skv: the first Sq - Skv rows of a batch are fully masked
// (zero output, LSE = -inf).
TEST(TestCpuFpReferenceSdpaRaggedFp32, CausalBottomRightMoreQueriesThanKeys)
{
    checkRaggedVsDense({5, 2, 4}, {2, 5, 1}, 2, 2, 16, 16, -1, 0, /*topLeftAlignment=*/false);
}

// A left-only window with bottom-right alignment uses windowOffset in the left bound.
TEST(TestCpuFpReferenceSdpaRaggedFp32, SlidingWindowLeftOnlyBottomRight)
{
    checkRaggedVsDense({6, 3, 4}, {4, 7, 4}, 2, 2, 16, 16, 1, -1, /*topLeftAlignment=*/false);
}

// GpuFpReferenceSdpaRagged rejects a zero head_dim ("all dimensions must be positive"). The CPU
// mirror must too, instead of running attention over empty Q and K rows.
TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnZeroHeadDim)
{
    // [1, 4, 2, 0] has a zero H x D block, so give it a 2-element token stride to be constructible.
    const std::vector<int64_t> qkDims = {1, 4, 2, 0};
    const std::vector<int64_t> qkStrides = {8, 2, 1, 1};
    const auto cum = cumTokens({4});
    std::vector<float> qB(8, 1.0f);
    std::vector<float> kB(8, 1.0f);
    ShallowRaggedTensor<float> q(
        qB.data(), qkDims, qkStrides, BSHD_SEQ_AXIS, makeRaggedOffsetAux(cum, qkStrides[1]));
    ShallowRaggedTensor<float> k(
        kB.data(), qkDims, qkStrides, BSHD_SEQ_AXIS, makeRaggedOffsetAux(cum, qkStrides[1]));
    std::vector<float> vB;
    std::vector<float> oB;
    auto v = makeValidRagged(vB, raggedDims(1, 4, 2, 16), {4});
    auto o = makeValidRagged(oB, raggedDims(1, 4, 2, 16), {4});

    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

// A zero K head count must be rejected before numHeads % numHeadsK divides by it. [1, 4, 0, 16]
// has an empty H x D block, so it gets an explicit 16-element token stride to be constructible.
TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnZeroKvHeads)
{
    const std::vector<int64_t> kDims = {1, 4, 0, 16};
    const std::vector<int64_t> kStrides = {64, 16, 16, 1};
    std::vector<float> kB(64, 1.0f);
    ShallowRaggedTensor<float> k(kB.data(),
                                 kDims,
                                 kStrides,
                                 BSHD_SEQ_AXIS,
                                 makeRaggedOffsetAux(cumTokens({4}), kStrides[1]));
    std::vector<float> qB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, raggedDims(1, 4, 2, 16), {4});
    auto v = makeValidRagged(vB, raggedDims(1, 4, 2, 16), {4});
    auto o = makeValidRagged(oB, raggedDims(1, 4, 2, 16), {4});

    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}
