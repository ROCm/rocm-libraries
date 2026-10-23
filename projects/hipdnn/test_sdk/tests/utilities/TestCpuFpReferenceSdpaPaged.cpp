// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Checks CpuFpReferenceSdpaRagged::forwardPaged against the dense CpuFpReferenceSdpa. For each
// batch, the live Q rows and the K/V rows reached through the page tables are gathered into dense
// [1, H, seqlen_b, D] tensors and run through the dense reference.
// Chain: dense CPU (trusted) -> paged CPU (here) -> paged GPU (TestGpuFpReferenceSdpaRagged).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpaRagged.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesSdpa.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaTestUtils.hpp>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;

namespace
{

// Written to O and LSE before a run, so rows the reference must not touch can be checked. Exactly
// representable in float, half and bfloat16.
constexpr float SENTINEL = 7.0f;

// Deterministic fill in [-1, 1). A local LCG avoids <random> distribution drift between builds.
template <typename T>
void fillUniform(std::vector<T>& buf, uint32_t seed)
{
    uint32_t state = seed;
    for(auto& x : buf)
    {
        state = state * 1664525U + 1013904223U;
        const float u = static_cast<float>(state >> 8) / static_cast<float>(1U << 24); // [0,1)
        x = static_cast<T>(2.0f * u - 1.0f);
    }
}

// One paged problem. K/V are block pools read through INT32 [B, 1, M, 1] page tables with the
// vLLM block_table strides [M, M, 1, 1]. Block ids are a random permutation of the pool, so
// logically adjacent blocks are scattered, and unused table entries are -1.
struct PagedCase
{
    std::vector<int64_t> seqLenKv;
    // Live Q rows per batch. Empty means no seq_len_q tensor (dense Q only): every batch has sq.
    std::vector<int64_t> seqLenQ;
    // Ragged Q only: rows stored after the live rows of each batch, which must stay untouched.
    std::vector<int64_t> qGap;
    int64_t sq = 1; // dense Q rows per batch
    int64_t numHeads = 4;
    int64_t numHeadsKv = 4;
    int64_t headDim = 64;
    int64_t blockSize = 16;
    int64_t extraBlocks = 3; // pool blocks no page-table entry references
    int64_t leftBound = -1;
    int64_t rightBound = -1;
    bool topLeftAlignment = true;
    bool raggedQ = false;
    // K and V are views into one buffer: per (token, head) a K row then a V row, so the token
    // stride is H_kv * 2D and V starts at +D.
    bool interleavedKv = false;
    // One page-table tensor serves K and V (cuDNN allows the same uid for both).
    bool sharedPageTable = false;
    // Attention sink per query head; empty means no sink.
    std::vector<float> sinks;
    uint32_t seed = 1234;
};

int64_t ceilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

std::unique_ptr<Tensor<int32_t>> makeLengthTensor(const std::vector<int64_t>& lengths)
{
    auto t = std::make_unique<Tensor<int32_t>>(
        std::vector<int64_t>{static_cast<int64_t>(lengths.size()), 1, 1, 1});
    for(size_t b = 0; b < lengths.size(); ++b)
    {
        t->setHostValue(static_cast<int32_t>(lengths[b]),
                        std::vector<int64_t>{static_cast<int64_t>(b), 0, 0, 0});
    }
    return t;
}

template <typename T>
struct PagedProblem
{
    explicit PagedProblem(const PagedCase& pagedCase)
        : c(pagedCase)
        , batch(static_cast<int64_t>(pagedCase.seqLenKv.size()))
    {
        const auto batchCount = static_cast<size_t>(batch);
        const int64_t numHeads = c.numHeads;
        const int64_t numHeadsKv = c.numHeadsKv;
        const int64_t headDim = c.headDim;
        const int64_t blockSize = c.blockSize;

        maxSeqLenKv = std::max<int64_t>(*std::max_element(c.seqLenKv.begin(), c.seqLenKv.end()), 1);
        maxBlocks = ceilDiv(maxSeqLenKv, blockSize);
        int64_t usedTotal = 0;
        for(const auto len : c.seqLenKv)
        {
            usedTotal += ceilDiv(len, blockSize);
        }
        numBlocks = usedTotal + c.extraBlocks;

        // --- Q / O / LSE ---
        for(size_t b = 0; b < batchCount; ++b)
        {
            const int64_t live = c.seqLenQ.empty() ? c.sq : c.seqLenQ[b];
            seqQ.push_back(live);
            if(c.raggedQ)
            {
                extentQ.push_back(live + (c.qGap.empty() ? 0 : c.qGap[b]));
            }
            else
            {
                extentQ.push_back(c.sq);
            }
        }

        if(c.raggedQ)
        {
            const int64_t sMax
                = std::max<int64_t>(*std::max_element(extentQ.begin(), extentQ.end()), 1);
            const auto cum = cumTokens(extentQ);
            const int64_t tokens = cum.back();
            qBuf.resize(static_cast<size_t>(tokens * numHeads * headDim));
            oBuf.assign(static_cast<size_t>(tokens * numHeads * headDim), static_cast<T>(SENTINEL));
            lseBuf.assign(static_cast<size_t>(tokens * numHeads), SENTINEL);
            fillUniform(qBuf, c.seed + 1);

            const auto qDims = raggedDims(batch, sMax, numHeads, headDim);
            const auto lseDims = raggedDims(batch, sMax, numHeads, 1);
            q = std::make_unique<ShallowRaggedTensor<T>>(
                qBuf.data(),
                qDims,
                raggedStrides(qDims),
                SDPA_SEQ_AXIS,
                makeRaggedOffsetAux(cum, numHeads * headDim));
            o = std::make_unique<ShallowRaggedTensor<T>>(
                oBuf.data(),
                qDims,
                raggedStrides(qDims),
                SDPA_SEQ_AXIS,
                makeRaggedOffsetAux(cum, numHeads * headDim));
            lse = std::make_unique<ShallowRaggedTensor<float>>(lseBuf.data(),
                                                               lseDims,
                                                               raggedStrides(lseDims),
                                                               SDPA_SEQ_AXIS,
                                                               makeRaggedOffsetAux(cum, numHeads));
        }
        else
        {
            const std::vector<int64_t> qDims = {batch, numHeads, c.sq, headDim};
            const std::vector<int64_t> lseDims = {batch, numHeads, c.sq, 1};
            qBuf.resize(static_cast<size_t>(batch * numHeads * c.sq * headDim));
            oBuf.assign(qBuf.size(), static_cast<T>(SENTINEL));
            lseBuf.assign(static_cast<size_t>(batch * numHeads * c.sq), SENTINEL);
            fillUniform(qBuf, c.seed + 1);
            q = std::make_unique<ShallowTensor<T>>(qBuf.data(), qDims, generateStrides(qDims));
            o = std::make_unique<ShallowTensor<T>>(oBuf.data(), qDims, generateStrides(qDims));
            lse = std::make_unique<ShallowTensor<float>>(
                lseBuf.data(), lseDims, generateStrides(lseDims));
        }

        // --- K / V block pools [num_blocks, H_kv, block_size, D] ---
        const std::vector<int64_t> kvDims = {numBlocks, numHeadsKv, blockSize, headDim};
        if(c.interleavedKv)
        {
            const int64_t rowPitch = 2 * headDim;
            kBuf.resize(static_cast<size_t>(numBlocks * blockSize * numHeadsKv * rowPitch));
            fillUniform(kBuf, c.seed + 2);
            const std::vector<int64_t> kvStrides
                = {blockSize * numHeadsKv * rowPitch, rowPitch, numHeadsKv * rowPitch, 1};
            k = std::make_unique<ShallowTensor<T>>(kBuf.data(), kvDims, kvStrides);
            v = std::make_unique<ShallowTensor<T>>(kBuf.data() + headDim, kvDims, kvStrides);
        }
        else
        {
            // vLLM's NHD block layout: [num_blocks, block_size, H_kv, D] viewed as
            // [num_blocks, H_kv, block_size, D].
            const std::vector<int64_t> kvStrides
                = {blockSize * numHeadsKv * headDim, headDim, numHeadsKv * headDim, 1};
            kBuf.resize(static_cast<size_t>(numBlocks * blockSize * numHeadsKv * headDim));
            vBuf.resize(kBuf.size());
            fillUniform(kBuf, c.seed + 2);
            fillUniform(vBuf, c.seed + 3);
            k = std::make_unique<ShallowTensor<T>>(kBuf.data(), kvDims, kvStrides);
            v = std::make_unique<ShallowTensor<T>>(vBuf.data(), kvDims, kvStrides);
        }

        // --- Page tables and lengths ---
        pageTableK = makePageTable(c.seed + 4);
        pageTableV = makePageTable(c.seed + 5);
        seqLenKv = makeLengthTensor(c.seqLenKv);
        if(!c.seqLenQ.empty())
        {
            seqLenQ = makeLengthTensor(c.seqLenQ);
        }
    }

    // Shuffled block ids for the used entries, -1 for the rest.
    std::unique_ptr<Tensor<int32_t>> makePageTable(uint32_t seed) const
    {
        auto table
            = std::make_unique<Tensor<int32_t>>(std::vector<int64_t>{batch, 1, maxBlocks, 1},
                                                std::vector<int64_t>{maxBlocks, maxBlocks, 1, 1});
        std::vector<int32_t> ids(static_cast<size_t>(numBlocks));
        std::iota(ids.begin(), ids.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(ids.begin(), ids.end(), rng);

        size_t next = 0;
        for(int64_t b = 0; b < batch; ++b)
        {
            const int64_t used = ceilDiv(c.seqLenKv[static_cast<size_t>(b)], c.blockSize);
            for(int64_t j = 0; j < maxBlocks; ++j)
            {
                const int32_t id = j < used ? ids[next++] : -1;
                table->setHostValue(id, std::vector<int64_t>{b, 0, j, 0});
            }
        }
        return table;
    }

    Tensor<int32_t>& tableV()
    {
        return c.sharedPageTable ? *pageTableK : *pageTableV;
    }

    float scale() const
    {
        return static_cast<float>(1.0 / std::sqrt(static_cast<double>(c.headDim)));
    }

    const std::vector<float>* sinkPerHead() const
    {
        return c.sinks.empty() ? nullptr : &c.sinks;
    }

    void run(bool passSeqLenQ = true)
    {
        CpuFpReferenceSdpaRagged::forwardPaged<T, T, T, T, float>(*q,
                                                                  *k,
                                                                  *v,
                                                                  *o,
                                                                  *pageTableK,
                                                                  tableV(),
                                                                  *seqLenKv,
                                                                  passSeqLenQ ? seqLenQ.get()
                                                                              : nullptr,
                                                                  maxSeqLenKv,
                                                                  scale(),
                                                                  c.leftBound,
                                                                  c.rightBound,
                                                                  c.topLeftAlignment,
                                                                  lse.get(),
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  sinkPerHead());
    }

    // Gathers batch b's K or V rows through a page table into a dense [1, H_kv, seqKv, D] tensor.
    Tensor<float> gatherDense(const TensorBase<T>& pool, Tensor<int32_t>& table, int64_t b) const
    {
        const int64_t seqKv = c.seqLenKv[static_cast<size_t>(b)];
        Tensor<float> dense({1, c.numHeadsKv, seqKv, c.headDim});
        for(int64_t skv = 0; skv < seqKv; ++skv)
        {
            const int64_t block
                = table.getHostValue(std::vector<int64_t>{b, 0, skv / c.blockSize, 0});
            for(int64_t hk = 0; hk < c.numHeadsKv; ++hk)
            {
                for(int64_t d = 0; d < c.headDim; ++d)
                {
                    dense(0, hk, skv, d) = static_cast<float>(
                        pool.getHostValue(std::vector<int64_t>{block, hk, skv % c.blockSize, d}));
                }
            }
        }
        dense.memory().markHostModified();
        return dense;
    }

    Tensor<float> gatherQ(int64_t b) const
    {
        const int64_t rows = seqQ[static_cast<size_t>(b)];
        Tensor<float> dense({1, c.numHeads, rows, c.headDim});
        for(int64_t h = 0; h < c.numHeads; ++h)
        {
            for(int64_t s = 0; s < rows; ++s)
            {
                for(int64_t d = 0; d < c.headDim; ++d)
                {
                    dense(0, h, s, d)
                        = static_cast<float>(q->getHostValue(std::vector<int64_t>{b, h, s, d}));
                }
            }
        }
        dense.memory().markHostModified();
        return dense;
    }

    // Live rows match the gathered dense reference; rows past seqQ[b] keep the sentinel.
    void expectMatchesGatheredDense()
    {
        for(int64_t b = 0; b < batch; ++b)
        {
            const auto bIdx = static_cast<size_t>(b);
            const int64_t rows = seqQ[bIdx];
            const int64_t seqKv = c.seqLenKv[bIdx];

            for(int64_t s = rows; s < extentQ[bIdx]; ++s)
            {
                for(int64_t h = 0; h < c.numHeads; ++h)
                {
                    EXPECT_EQ(lse->getHostValue(std::vector<int64_t>{b, h, s, 0}), SENTINEL)
                        << "LSE row past seq_len_q written, batch " << b << " row " << s;
                    for(int64_t d = 0; d < c.headDim; ++d)
                    {
                        EXPECT_EQ(
                            static_cast<float>(o->getHostValue(std::vector<int64_t>{b, h, s, d})),
                            SENTINEL)
                            << "O row past seq_len_q written, batch " << b << " row " << s;
                    }
                }
            }
            if(rows == 0)
            {
                continue;
            }

            if(seqKv == 0)
            {
                // No keys: every live row is fully masked. O = 0 and LSE = sink (-inf without).
                for(int64_t s = 0; s < rows; ++s)
                {
                    for(int64_t h = 0; h < c.numHeads; ++h)
                    {
                        const float expectedLse = c.sinks.empty()
                                                      ? -std::numeric_limits<float>::infinity()
                                                      : c.sinks[static_cast<size_t>(h)];
                        EXPECT_EQ(lse->getHostValue(std::vector<int64_t>{b, h, s, 0}), expectedLse);
                        for(int64_t d = 0; d < c.headDim; ++d)
                        {
                            EXPECT_EQ(static_cast<float>(
                                          o->getHostValue(std::vector<int64_t>{b, h, s, d})),
                                      0.0f);
                        }
                    }
                }
                continue;
            }

            const auto qd = gatherQ(b);
            const auto kd = gatherDense(*k, *pageTableK, b);
            const auto vd = gatherDense(*v, tableV(), b);
            Tensor<T> oDense({1, c.numHeads, rows, c.headDim});
            Tensor<float> lseDense({1, c.numHeads, rows, 1});
            CpuFpReferenceSdpa::forward<float, float, float, T, float>(qd,
                                                                       kd,
                                                                       vd,
                                                                       oDense,
                                                                       scale(),
                                                                       /*attnMask=*/nullptr,
                                                                       c.leftBound,
                                                                       c.rightBound,
                                                                       c.topLeftAlignment,
                                                                       &lseDense,
                                                                       nullptr,
                                                                       nullptr,
                                                                       nullptr,
                                                                       sinkPerHead());

            const float tolerance = sdpa::calculateSdpaFwdTolerance<T, T>(
                -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, c.headDim, seqKv, scale());
            for(int64_t s = 0; s < rows; ++s)
            {
                for(int64_t h = 0; h < c.numHeads; ++h)
                {
                    const float lsePaged = lse->getHostValue(std::vector<int64_t>{b, h, s, 0});
                    const float lseExpected = lseDense(0, h, s, 0);
                    if(std::isinf(lseExpected))
                    {
                        EXPECT_EQ(lsePaged, lseExpected)
                            << "LSE mismatch batch " << b << " row " << s << " head " << h;
                    }
                    else
                    {
                        EXPECT_NEAR(lsePaged, lseExpected, 1e-4f)
                            << "LSE mismatch batch " << b << " row " << s << " head " << h;
                    }
                    for(int64_t d = 0; d < c.headDim; ++d)
                    {
                        EXPECT_NEAR(
                            static_cast<float>(o->getHostValue(std::vector<int64_t>{b, h, s, d})),
                            static_cast<float>(oDense(0, h, s, d)),
                            tolerance)
                            << "output mismatch batch " << b << " row " << s << " head " << h
                            << " d " << d;
                    }
                }
            }
        }
    }

    // True if no O element still differs from the sentinel, i.e. nothing was written.
    bool outputUntouched() const
    {
        return std::all_of(
            oBuf.begin(), oBuf.end(), [](T x) { return static_cast<float>(x) == SENTINEL; });
    }

    PagedCase c;
    int64_t batch;
    int64_t maxSeqLenKv = 0;
    int64_t maxBlocks = 0;
    int64_t numBlocks = 0;
    std::vector<int64_t> seqQ;
    std::vector<int64_t> extentQ;
    std::vector<T> qBuf;
    std::vector<T> kBuf;
    std::vector<T> vBuf;
    std::vector<T> oBuf;
    std::vector<float> lseBuf;
    std::unique_ptr<TensorBase<T>> q;
    std::unique_ptr<TensorBase<T>> k;
    std::unique_ptr<TensorBase<T>> v;
    std::unique_ptr<TensorBase<T>> o;
    std::unique_ptr<TensorBase<float>> lse;
    std::unique_ptr<Tensor<int32_t>> pageTableK;
    std::unique_ptr<Tensor<int32_t>> pageTableV;
    std::unique_ptr<Tensor<int32_t>> seqLenKv;
    std::unique_ptr<Tensor<int32_t>> seqLenQ;
};

template <typename T>
void checkPagedVsGatheredDense(const PagedCase& pagedCase)
{
    PagedProblem<T> problem(pagedCase);
    problem.run();
    problem.expectMatchesGatheredDense();
}

// Decode-shaped case: one query row per sequence, sequence lengths spanning partial and full
// blocks.
PagedCase decodeCase(int64_t headDim)
{
    PagedCase c;
    c.seqLenKv = {37, 16, 70};
    c.headDim = headDim;
    return c;
}

} // namespace

TEST(TestCpuFpReferenceSdpaPaged, ShuffledPageTableBf16HeadDim64)
{
    checkPagedVsGatheredDense<bfloat16>(decodeCase(64));
}

TEST(TestCpuFpReferenceSdpaPaged, ShuffledPageTableBf16HeadDim128)
{
    checkPagedVsGatheredDense<bfloat16>(decodeCase(128));
}

TEST(TestCpuFpReferenceSdpaPaged, ShuffledPageTableHalfHeadDim64)
{
    checkPagedVsGatheredDense<half>(decodeCase(64));
}

TEST(TestCpuFpReferenceSdpaPaged, ShuffledPageTableHalfHeadDim128)
{
    checkPagedVsGatheredDense<half>(decodeCase(128));
}

TEST(TestCpuFpReferenceSdpaPaged, ShuffledPageTableFloat)
{
    checkPagedVsGatheredDense<float>(decodeCase(64));
}

// seq_len_kv 0 gives zero output and LSE = -inf; the other batches are unaffected.
TEST(TestCpuFpReferenceSdpaPaged, MixedSeqLenKvIncludingZero)
{
    PagedCase c;
    c.seqLenKv = {0, 5, 33, 0, 16};
    c.sq = 2;
    checkPagedVsGatheredDense<half>(c);
    checkPagedVsGatheredDense<bfloat16>(c);
}

// max_seq_len_kv far above every length: most table entries are -1 and never read.
TEST(TestCpuFpReferenceSdpaPaged, UnusedPageTableEntriesNeverRead)
{
    PagedCase c;
    c.seqLenKv = {3, 9, 1};
    c.blockSize = 4;
    PagedProblem<bfloat16> problem(c);
    // Widen the bound and the tables without adding used entries.
    problem.maxSeqLenKv = 64;
    problem.maxBlocks = ceilDiv(problem.maxSeqLenKv, c.blockSize);
    problem.pageTableK = problem.makePageTable(c.seed + 4);
    problem.pageTableV = problem.makePageTable(c.seed + 5);
    problem.run();
    problem.expectMatchesGatheredDense();
}

TEST(TestCpuFpReferenceSdpaPaged, Gqa)
{
    PagedCase c = decodeCase(64);
    c.numHeads = 8;
    c.numHeadsKv = 2;
    checkPagedVsGatheredDense<bfloat16>(c);
    checkPagedVsGatheredDense<half>(c);
}

TEST(TestCpuFpReferenceSdpaPaged, Mqa)
{
    PagedCase c = decodeCase(128);
    c.numHeads = 4;
    c.numHeadsKv = 1;
    checkPagedVsGatheredDense<bfloat16>(c);
}

// Speculative-decode shape: 4 query rows per sequence, bottom-right causal. The batch with fewer
// keys than queries has fully masked leading rows.
TEST(TestCpuFpReferenceSdpaPaged, BottomRightCausalDecodeSq4)
{
    PagedCase c;
    c.seqLenKv = {4, 19, 40, 2};
    c.sq = 4;
    c.numHeads = 8;
    c.numHeadsKv = 2;
    c.rightBound = 0;
    c.topLeftAlignment = false;
    checkPagedVsGatheredDense<bfloat16>(c);
    checkPagedVsGatheredDense<half>(c);
}

TEST(TestCpuFpReferenceSdpaPaged, SlidingWindowBottomRight)
{
    PagedCase c;
    c.seqLenKv = {30, 9, 50};
    c.sq = 4;
    c.leftBound = 7;
    c.rightBound = 0;
    c.topLeftAlignment = false;
    checkPagedVsGatheredDense<bfloat16>(c);
    checkPagedVsGatheredDense<half>(c);
}

// Dense Q with seq_len_q: rows past seq_len_q[b] (padding) are not written.
TEST(TestCpuFpReferenceSdpaPaged, DenseQSeqLenQLeavesPaddingRows)
{
    PagedCase c;
    c.seqLenKv = {20, 7, 33};
    c.seqLenQ = {4, 2, 0};
    c.sq = 4;
    c.rightBound = 0;
    c.topLeftAlignment = false;
    checkPagedVsGatheredDense<bfloat16>(c);
}

// Ragged (packed) Q/O/LSE with seq_len_q, including an empty batch and gap rows after the live
// rows of a batch.
TEST(TestCpuFpReferenceSdpaPaged, RaggedQWithSeqLenQ)
{
    PagedCase c;
    c.raggedQ = true;
    c.seqLenKv = {10, 7, 20, 0};
    c.seqLenQ = {3, 0, 5, 2};
    c.qGap = {1, 2, 0, 0};
    c.numHeads = 8;
    c.numHeadsKv = 2;
    c.rightBound = 0;
    c.topLeftAlignment = false;
    checkPagedVsGatheredDense<bfloat16>(c);
    checkPagedVsGatheredDense<half>(c);
}

TEST(TestCpuFpReferenceSdpaPaged, RaggedQSlidingWindowHeadDim128)
{
    PagedCase c;
    c.raggedQ = true;
    c.seqLenKv = {40, 18};
    c.seqLenQ = {6, 1};
    c.headDim = 128;
    c.leftBound = 5;
    c.rightBound = 0;
    c.topLeftAlignment = false;
    checkPagedVsGatheredDense<bfloat16>(c);
}

TEST(TestCpuFpReferenceSdpaPaged, InterleavedKvStrides)
{
    PagedCase c = decodeCase(64);
    c.interleavedKv = true;
    c.numHeadsKv = 1;
    checkPagedVsGatheredDense<bfloat16>(c);

    c.numHeadsKv = 2;
    c.numHeads = 8;
    c.sq = 4;
    c.rightBound = 0;
    c.topLeftAlignment = false;
    checkPagedVsGatheredDense<half>(c);
}

TEST(TestCpuFpReferenceSdpaPaged, SharedPageTable)
{
    PagedCase c = decodeCase(64);
    c.sharedPageTable = true;
    checkPagedVsGatheredDense<bfloat16>(c);

    c.interleavedKv = true;
    checkPagedVsGatheredDense<half>(c);
}

// Distinct page tables put a logical row's K and V in different blocks.
TEST(TestCpuFpReferenceSdpaPaged, SeparatePageTablesDiffer)
{
    const PagedProblem<float> problem(decodeCase(64));
    bool anyDiffer = false;
    for(int64_t j = 0; j < problem.maxBlocks; ++j)
    {
        const std::vector<int64_t> index = {0, 0, j, 0};
        anyDiffer
            = anyDiffer
              || problem.pageTableK->getHostValue(index) != problem.pageTableV->getHostValue(index);
    }
    EXPECT_TRUE(anyDiffer);
}

// --- Validation (negative) cases. Every input is checked before any output is written. ---

TEST(TestCpuFpReferenceSdpaPaged, ThrowsOnOutOfRangeUsedBlockId)
{
    PagedProblem<float> problem(decodeCase(64));
    problem.pageTableK->setHostValue(static_cast<int32_t>(problem.numBlocks),
                                     std::vector<int64_t>{2, 0, 1, 0});
    EXPECT_THROW(problem.run(), std::invalid_argument);
    EXPECT_TRUE(problem.outputUntouched());
}

TEST(TestCpuFpReferenceSdpaPaged, ThrowsOnNegativeUsedBlockId)
{
    PagedProblem<float> problem(decodeCase(64));
    problem.pageTableV->setHostValue(-1, std::vector<int64_t>{0, 0, 2, 0});
    EXPECT_THROW(problem.run(), std::invalid_argument);
    EXPECT_TRUE(problem.outputUntouched());
}

TEST(TestCpuFpReferenceSdpaPaged, ThrowsOnSeqLenKvAboveMax)
{
    PagedProblem<float> problem(decodeCase(64));
    problem.maxSeqLenKv = 69; // batch 2 has 70 keys
    EXPECT_THROW(problem.run(), std::invalid_argument);
    EXPECT_TRUE(problem.outputUntouched());
}

TEST(TestCpuFpReferenceSdpaPaged, ThrowsWhenPageTablesTooNarrow)
{
    PagedProblem<float> problem(decodeCase(64));
    // Tables must be ceil(max_seq_len_kv / block_size) wide, so a max_seq_len_kv that lets a
    // sequence need more blocks than the tables hold is rejected.
    problem.maxSeqLenKv = 1000;
    problem.seqLenKv->setHostValue(
        static_cast<int32_t>(problem.maxBlocks * problem.c.blockSize + 1),
        std::vector<int64_t>{1, 0, 0, 0});
    EXPECT_THROW(problem.run(), std::invalid_argument);
    EXPECT_TRUE(problem.outputUntouched());
}

TEST(TestCpuFpReferenceSdpaPaged, ThrowsOnNegativeSeqLenKv)
{
    PagedProblem<float> problem(decodeCase(64));
    problem.seqLenKv->setHostValue(-1, std::vector<int64_t>{0, 0, 0, 0});
    EXPECT_THROW(problem.run(), std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaPaged, ThrowsOnRaggedQWithoutSeqLenQ)
{
    PagedCase c;
    c.raggedQ = true;
    c.seqLenKv = {10, 7};
    c.seqLenQ = {3, 2};
    PagedProblem<float> problem(c);
    EXPECT_THROW(problem.run(/*passSeqLenQ=*/false), std::invalid_argument);
    EXPECT_TRUE(problem.outputUntouched());
}

TEST(TestCpuFpReferenceSdpaPaged, ThrowsOnSeqLenQBeyondDenseExtent)
{
    PagedCase c;
    c.seqLenKv = {10, 7};
    c.seqLenQ = {2, 2};
    c.sq = 2;
    PagedProblem<float> problem(c);
    problem.seqLenQ->setHostValue(3, std::vector<int64_t>{1, 0, 0, 0});
    EXPECT_THROW(problem.run(), std::invalid_argument);
    EXPECT_TRUE(problem.outputUntouched());
}

TEST(TestCpuFpReferenceSdpaPaged, ThrowsOnSeqLenQBeyondRaggedExtent)
{
    PagedCase c;
    c.raggedQ = true;
    c.seqLenKv = {10, 7};
    c.seqLenQ = {3, 2};
    c.qGap = {1, 0};
    PagedProblem<float> problem(c);
    problem.seqLenQ->setHostValue(5, std::vector<int64_t>{0, 0, 0, 0}); // batch 0 holds 4 rows
    EXPECT_THROW(problem.run(), std::invalid_argument);
    EXPECT_TRUE(problem.outputUntouched());
}

TEST(TestCpuFpReferenceSdpaPaged, ThrowsOnBadPageTableShape)
{
    PagedProblem<float> problem(decodeCase(64));
    problem.pageTableV = std::make_unique<Tensor<int32_t>>(
        std::vector<int64_t>{problem.batch, 1, problem.maxBlocks + 1, 1});
    EXPECT_THROW(problem.run(), std::invalid_argument);
}

// The page tables must be exactly ceil(max_seq_len_kv / block_size) wide, as in the frontend and
// the GPU reference, even when every used entry would fit.
TEST(TestCpuFpReferenceSdpaPaged, ThrowsOnPageTablesWiderThanMaxSeqLenKv)
{
    PagedProblem<float> problem(decodeCase(64));
    ++problem.maxBlocks;
    problem.pageTableK = problem.makePageTable(problem.c.seed + 4);
    problem.pageTableV = problem.makePageTable(problem.c.seed + 5);
    EXPECT_THROW(problem.run(), std::invalid_argument);
    EXPECT_TRUE(problem.outputUntouched());
}

// max_seq_len_kv must be positive, even when every sequence is empty.
TEST(TestCpuFpReferenceSdpaPaged, ThrowsOnZeroMaxSeqLenKv)
{
    PagedCase c;
    c.seqLenKv = {0, 0};
    PagedProblem<float> problem(c);
    problem.maxSeqLenKv = 0; // the tables stay one block wide
    EXPECT_THROW(problem.run(), std::invalid_argument);
    EXPECT_TRUE(problem.outputUntouched());
}

// Every dim of q is positive, as the frontend requires of every tensor: an empty sequence is
// seq_len_q[b] == 0, not Sq == 0.
TEST(TestCpuFpReferenceSdpaPaged, ThrowsOnZeroSq)
{
    PagedProblem<float> problem(decodeCase(64));
    const int64_t tokenStride = problem.c.numHeads * problem.c.headDim;
    const std::vector<int64_t> emptyDims
        = {problem.batch, problem.c.numHeads, 0, problem.c.headDim};
    const std::vector<int64_t> emptyStrides = {tokenStride, problem.c.headDim, tokenStride, 1};
    std::vector<float> placeholder(1);
    problem.q = std::make_unique<ShallowTensor<float>>(placeholder.data(), emptyDims, emptyStrides);
    problem.o = std::make_unique<ShallowTensor<float>>(placeholder.data(), emptyDims, emptyStrides);
    EXPECT_THROW(problem.run(), std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaPaged, ThrowsOnBadSeqLenShape)
{
    PagedProblem<float> problem(decodeCase(64));
    problem.seqLenKv = std::make_unique<Tensor<int32_t>>(std::vector<int64_t>{problem.batch});
    EXPECT_THROW(problem.run(), std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaPaged, ThrowsOnBlockSizeMismatch)
{
    PagedProblem<float> problem(decodeCase(64));
    const auto& kDims = problem.k->dims();
    std::vector<float> vBuf(static_cast<size_t>(kDims[0] * kDims[1] * 2 * kDims[2] * kDims[3]));
    problem.v = std::make_unique<ShallowTensor<float>>(
        vBuf.data(),
        std::vector<int64_t>{kDims[0], kDims[1], 2 * kDims[2], kDims[3]},
        generateStrides(std::vector<int64_t>{kDims[0], kDims[1], 2 * kDims[2], kDims[3]}));
    EXPECT_THROW(problem.run(), std::invalid_argument);
}

// A dense output cannot follow a packed Q.
TEST(TestCpuFpReferenceSdpaPaged, ThrowsOnDenseOutputWithRaggedQ)
{
    PagedCase c;
    c.raggedQ = true;
    c.seqLenKv = {10, 7};
    c.seqLenQ = {3, 2};
    PagedProblem<float> problem(c);
    const std::vector<int64_t> oDims = problem.o->dims();
    std::vector<float> denseOut(static_cast<size_t>(oDims[0] * oDims[1] * oDims[2] * oDims[3]));
    problem.o
        = std::make_unique<ShallowTensor<float>>(denseOut.data(), oDims, generateStrides(oDims));
    EXPECT_THROW(problem.run(), std::invalid_argument);
}

// Attention sinks: one logit per query head, compared with the dense reference given the same
// sinks. A sequence with seq_len_kv == 0 must give O = 0 and LSE = sink.
TEST(TestCpuFpReferenceSdpaPaged, SinksGqaDecodeWithEmptySequence)
{
    PagedCase c;
    c.seqLenKv = {37, 0, 16, 70};
    c.numHeads = 8;
    c.numHeadsKv = 2;
    c.sinks = {0.5f, -1.25f, 2.0f, -0.75f, 0.0f, 1.5f, -2.0f, 0.25f};
    checkPagedVsGatheredDense<bfloat16>(c);
    checkPagedVsGatheredDense<half>(c);
    checkPagedVsGatheredDense<float>(c);
}

TEST(TestCpuFpReferenceSdpaPaged, SinksRaggedQSlidingWindow)
{
    PagedCase c;
    c.raggedQ = true;
    c.seqLenKv = {10, 7, 40, 0};
    c.seqLenQ = {3, 0, 6, 2};
    c.qGap = {1, 2, 0, 0};
    c.numHeads = 4;
    c.numHeadsKv = 2;
    c.leftBound = 5;
    c.rightBound = 0;
    c.topLeftAlignment = false;
    c.sinks = {1.0f, -0.5f, 0.75f, -3.0f};
    checkPagedVsGatheredDense<bfloat16>(c);
}

// A sink of -inf is exactly no sink.
TEST(TestCpuFpReferenceSdpaPaged, NegativeInfinitySinkEqualsNoSink)
{
    PagedCase c = decodeCase(64);
    c.seqLenKv = {37, 0, 16};
    PagedProblem<float> plain(c);
    plain.run();
    c.sinks.assign(static_cast<size_t>(c.numHeads), -std::numeric_limits<float>::infinity());
    PagedProblem<float> withSink(c);
    withSink.run();
    EXPECT_EQ(plain.oBuf, withSink.oBuf);
    EXPECT_EQ(plain.lseBuf, withSink.lseBuf);
}

TEST(TestCpuFpReferenceSdpaPaged, ThrowsOnSinkCountMismatch)
{
    PagedCase c = decodeCase(64);
    c.sinks = {0.0f, 0.0f};
    PagedProblem<float> problem(c);
    EXPECT_THROW(problem.run(), std::invalid_argument);
}
