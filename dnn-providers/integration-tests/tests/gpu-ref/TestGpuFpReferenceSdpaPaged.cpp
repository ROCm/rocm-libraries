// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU tests for the paged-KV SDPA forward reference (GpuFpReferenceSdpaRagged::fpropPaged).
//
// Each case runs fpropPaged and checks it two ways:
//   1. against CpuFpReferenceSdpaRagged::forwardPaged on the same host buffers (whole O/LSE
//      element space, so untouched padding rows must match too), and
//   2. against a gather-then-dense oracle: each batch's pages are gathered into dense K/V and run
//      through the dense CpuFpReferenceSdpa with that batch's own Sq/Skv.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpaRagged.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn-gpu-ref/GpuFpReferenceSdpaRagged.hpp>
#include <hipdnn-gpu-ref/ShallowGpuTensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_gpu_ref;

namespace
{

template <typename T>
struct TypeTag
{
    using type = T;
};

constexpr unsigned int SEED_Q = 42;
constexpr unsigned int SEED_K = 43;
constexpr unsigned int SEED_PAGES_K = 45;
constexpr unsigned int SEED_PAGES_V = 46;

// Rows the reference must not write keep these values.
constexpr float O_SENTINEL = 7.0f;
constexpr float LSE_SENTINEL = -99.0f;

template <typename T>
float tolerance()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return 2e-5f;
    }
    else if constexpr(std::is_same_v<T, half> || std::is_same_v<T, bfloat16>)
    {
        return 1e-2f;
    }
    else if constexpr(std::is_same_v<T, fp8_e4m3>)
    {
        // fp8 inputs with a bf16 output. Host and device decode fp8 the same way, so this only
        // covers bf16 output rounding and device-vs-host math.
        return 2e-2f;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

int64_t maxOf(const std::vector<int64_t>& v)
{
    return *std::max_element(v.begin(), v.end());
}

int64_t ceilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

// -inf only matches -inf; everything else is compared within tolerance.
void expectClose(float actual, float expected, float tol, const std::string& what)
{
    if(std::isinf(expected) || std::isinf(actual))
    {
        EXPECT_EQ(actual, expected) << what;
    }
    else
    {
        EXPECT_NEAR(actual, expected, tol) << what;
    }
}

// Optional fp8 descales, each scalar [1] or per KV head [B, H_kv, 1, 1]. Both references get the
// same tensors; the dense oracle applies them to the inputs it gathers.
struct PagedDescales
{
    Tensor<float>* q = nullptr;
    Tensor<float>* k = nullptr;
    Tensor<float>* v = nullptr;
};

// A descale as the references read it: 1 when absent, else the scalar or the (b, kvHead) entry.
float descaleAt(const Tensor<float>* descale, int64_t b, int64_t kvHead)
{
    if(descale == nullptr)
    {
        return 1.0f;
    }
    if(descale->elementCount() == 1)
    {
        return descale->getHostValue(std::vector<int64_t>(descale->dims().size(), 0));
    }
    return descale->getHostValue(b, kvHead, int64_t{0}, int64_t{0});
}

Tensor<float> makeScalarDescale(float value)
{
    Tensor<float> t({1});
    t.memory().hostData()[0] = value;
    t.memory().markHostModified();
    return t;
}

// Per-KV-head descale [B, heads, 1, 1] with a distinct value per (b, head).
Tensor<float> makePerHeadDescale(int64_t batch, int64_t heads, float base)
{
    Tensor<float> t({batch, heads, 1, 1});
    auto* p = t.memory().hostData();
    for(int64_t i = 0; i < batch * heads; ++i)
    {
        p[i] = base + 0.1f * static_cast<float>(i);
    }
    t.memory().markHostModified();
    return t;
}

// INT32 [B, 1, 1, 1] lengths.
Tensor<int32_t> makeSeqLens(const std::vector<int64_t>& lens)
{
    Tensor<int32_t> t({static_cast<int64_t>(lens.size()), 1, 1, 1});
    auto* p = t.memory().hostData();
    for(size_t i = 0; i < lens.size(); ++i)
    {
        p[i] = static_cast<int32_t>(lens[i]);
    }
    t.memory().markHostModified();
    return t;
}

// INT32 [B+1, 1, 1, 1] element offsets, cum[i] * seqStride.
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

// vLLM block_table [B, M] viewed zero-copy as [B, 1, M, 1] (strides [M, M, 1, 1]). Each batch's
// used entries take the next ids of a shuffled pool; unused entries hold -1.
Tensor<int32_t> makePageTable(const std::vector<int64_t>& seqLenKv,
                              int64_t blockSize,
                              int64_t maxBlocks,
                              int64_t numBlocks,
                              unsigned int seed)
{
    const auto batch = static_cast<int64_t>(seqLenKv.size());
    Tensor<int32_t> table({batch, 1, maxBlocks, 1}, {maxBlocks, maxBlocks, 1, 1});
    std::vector<int32_t> pool(static_cast<size_t>(numBlocks));
    std::iota(pool.begin(), pool.end(), 0);
    std::mt19937 gen(seed);
    std::shuffle(pool.begin(), pool.end(), gen);

    auto* p = table.memory().hostData();
    size_t next = 0;
    for(int64_t b = 0; b < batch; ++b)
    {
        const int64_t used = ceilDiv(seqLenKv[static_cast<size_t>(b)], blockSize);
        for(int64_t j = 0; j < maxBlocks; ++j)
        {
            p[b * maxBlocks + j] = j < used ? pool[next++] : -1;
        }
    }
    table.memory().markHostModified();
    return table;
}

struct PagedCase
{
    std::vector<int64_t> seqLenKv;
    // Dense: Sq. Ragged: S_max of the Q/O tensors.
    int64_t seqQMax = 1;
    std::optional<std::vector<int64_t>> seqLenQ;
    // Ragged Q token counts per batch. Unset means dense Q.
    std::optional<std::vector<int64_t>> raggedQTokens;
    int64_t numHeads = 4;
    int64_t numHeadsK = 4;
    int64_t numHeadsV = 4;
    int64_t headDim = 64;
    int64_t blockSize = 16;
    int64_t leftBound = -1;
    int64_t rightBound = -1;
    bool topLeftAlignment = true;
    // K and V share one buffer: per token and head, D K values then D V values.
    bool interleavedKv = false;
    // One page table tensor serves K and V.
    bool sharedPageTable = false;
    bool withLse = false;
    // Attention sink per query head; empty means none. sinkInIoType binds it as T (rounded),
    // otherwise as float.
    std::vector<float> sinks;
    bool sinkInIoType = false;
};

// T is the Q/K/V type and OutT the O type (fp8 Q/K/V write bf16).
template <typename T, typename OutT = T>
void checkPaged(const PagedCase& c, const PagedDescales& descales = {})
{
    const auto batch = static_cast<int64_t>(c.seqLenKv.size());
    const bool raggedQ = c.raggedQTokens.has_value();
    const int64_t h = c.numHeads;
    const int64_t d = c.headDim;
    const int64_t bs = c.blockSize;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    // Pool: every used block plus two spares, shuffled.
    int64_t usedBlocks = 0;
    for(const auto len : c.seqLenKv)
    {
        usedBlocks += ceilDiv(len, bs);
    }
    const int64_t numBlocks = usedBlocks + 2;
    // One spare block per batch so every table has unused (-1) entries.
    const int64_t maxSeqLenKv = maxOf(c.seqLenKv) + bs;
    const int64_t maxBlocks = ceilDiv(maxSeqLenKv, bs);
    auto pageTableK = makePageTable(c.seqLenKv, bs, maxBlocks, numBlocks, SEED_PAGES_K);
    auto pageTableV = makePageTable(c.seqLenKv, bs, maxBlocks, numBlocks, SEED_PAGES_V);
    auto& pageTableVUsed = c.sharedPageTable ? pageTableK : pageTableV;

    // K/V containers [num_blocks, H, block_size, D], token-major inside each block.
    const std::vector<int64_t> kDims{numBlocks, c.numHeadsK, bs, d};
    const std::vector<int64_t> vDims{numBlocks, c.numHeadsV, bs, d};
    std::vector<int64_t> kStrides = raggedStrides(kDims);
    std::vector<int64_t> vStrides = raggedStrides(vDims);
    int64_t vOffset = 0;
    const int64_t kvElements = numBlocks * bs * (c.numHeadsK + c.numHeadsV) * d;
    if(c.interleavedKv)
    {
        ASSERT_EQ(c.numHeadsK, c.numHeadsV);
        const int64_t hk = c.numHeadsK;
        kStrides = {bs * hk * 2 * d, 2 * d, hk * 2 * d, 1};
        vStrides = kStrides;
        vOffset = d;
    }
    else
    {
        vOffset = numBlocks * bs * c.numHeadsK * d;
    }
    Tensor<T> kvBuffer({kvElements});
    kvBuffer.fillTensorWithRandomValues(-1.0f, 1.0f, SEED_K);
    T* kvHost = kvBuffer.memory().hostData();
    ShallowTensor<T> kHost(kvHost, kDims, kStrides);
    ShallowTensor<T> vHost(kvHost + vOffset, vDims, vStrides);

    // Q/O [B, H, S, D], BSHD. Ragged Q packs the token counts of raggedQTokens.
    const std::vector<int64_t> qDims = raggedDims(batch, c.seqQMax, h, d);
    const auto qStrides = raggedStrides(qDims);
    std::vector<int64_t> qExtent(static_cast<size_t>(batch), c.seqQMax);
    if(raggedQ)
    {
        qExtent = *c.raggedQTokens;
    }
    const auto cumQ = cumTokens(qExtent);
    Tensor<T> q(qDims, qStrides);
    q.fillTensorWithRandomValues(-1.0f, 1.0f, SEED_Q);
    Tensor<OutT> oGpu(qDims, qStrides);
    Tensor<OutT> oCpu(qDims, qStrides);
    oGpu.fillWithValue(static_cast<OutT>(O_SENTINEL));
    oCpu.fillWithValue(static_cast<OutT>(O_SENTINEL));
    const std::vector<int64_t> lseDims{batch, h, c.seqQMax, 1};
    Tensor<float> lseGpu(lseDims);
    Tensor<float> lseCpu(lseDims);
    lseGpu.fillWithValue(LSE_SENTINEL);
    lseCpu.fillWithValue(LSE_SENTINEL);

    auto seqLenKv = makeSeqLens(c.seqLenKv);
    std::optional<Tensor<int32_t>> seqLenQ;
    if(c.seqLenQ.has_value())
    {
        seqLenQ.emplace(makeSeqLens(*c.seqLenQ));
    }

    // The sink as the GPU reads it, and the same values (rounded to T for an I/O-type sink) as
    // the CPU references take them.
    Tensor<float> sinkFloat({1, h, 1, 1});
    Tensor<T> sinkIo({1, h, 1, 1});
    std::vector<float> sinkValues = c.sinks;
    for(int64_t hh = 0; hh < static_cast<int64_t>(c.sinks.size()); ++hh)
    {
        auto& value = sinkValues[static_cast<size_t>(hh)];
        if(c.sinkInIoType)
        {
            value = static_cast<float>(static_cast<T>(value));
        }
        sinkFloat.setHostValue(value, 0, hh, 0, 0);
        sinkIo.setHostValue(static_cast<T>(value), 0, hh, 0, 0);
    }
    sinkFloat.memory().markHostModified();
    sinkIo.memory().markHostModified();
    const std::vector<float>* sinkPerHead = c.sinks.empty() ? nullptr : &sinkValues;

    // 1. CPU paged reference, first, while the host buffers are untouched.
    {
        auto* seqLenQPtr = seqLenQ.has_value() ? &seqLenQ.value() : nullptr;
        auto* lsePtr = c.withLse ? &lseCpu : nullptr;
        if(raggedQ)
        {
            ShallowRaggedTensor<T> qR(q.memory().hostData(),
                                      qDims,
                                      qStrides,
                                      SDPA_SEQ_AXIS,
                                      makeRaggedOffsetAux(cumQ, h * d));
            ShallowRaggedTensor<OutT> oR(oCpu.memory().hostData(),
                                         qDims,
                                         qStrides,
                                         SDPA_SEQ_AXIS,
                                         makeRaggedOffsetAux(cumQ, h * d));
            CpuFpReferenceSdpaRagged::forwardPaged<T, T, T, OutT, float>(qR,
                                                                         kHost,
                                                                         vHost,
                                                                         oR,
                                                                         pageTableK,
                                                                         pageTableVUsed,
                                                                         seqLenKv,
                                                                         seqLenQPtr,
                                                                         maxSeqLenKv,
                                                                         scale,
                                                                         c.leftBound,
                                                                         c.rightBound,
                                                                         c.topLeftAlignment,
                                                                         lsePtr,
                                                                         descales.q,
                                                                         descales.k,
                                                                         descales.v,
                                                                         sinkPerHead);
        }
        else
        {
            CpuFpReferenceSdpaRagged::forwardPaged<T, T, T, OutT, float>(q,
                                                                         kHost,
                                                                         vHost,
                                                                         oCpu,
                                                                         pageTableK,
                                                                         pageTableVUsed,
                                                                         seqLenKv,
                                                                         seqLenQPtr,
                                                                         maxSeqLenKv,
                                                                         scale,
                                                                         c.leftBound,
                                                                         c.rightBound,
                                                                         c.topLeftAlignment,
                                                                         lsePtr,
                                                                         descales.q,
                                                                         descales.k,
                                                                         descales.v,
                                                                         sinkPerHead);
        }
    }

    // GPU paged reference.
    {
        T* kvDevice = static_cast<T*>(kvBuffer.memory().deviceData());
        ShallowGpuTensor<T> kDev(kvDevice, kDims, kStrides);
        ShallowGpuTensor<T> vDev(kvDevice + vOffset, vDims, vStrides);
        auto offQ = makeRaggedOffset(cumQ, h * d);
        auto offO = makeRaggedOffset(cumQ, h * d);
        const auto runGpu = [&](auto* sink, auto sinkTag) {
            using SinkT = typename decltype(sinkTag)::type;
            GpuFpReferenceSdpaRagged::fpropPaged<T, T, T, OutT, float, SinkT>(
                q,
                kDev,
                vDev,
                oGpu,
                raggedQ ? &offQ : nullptr,
                raggedQ ? &offO : nullptr,
                pageTableK,
                pageTableVUsed,
                seqLenKv,
                seqLenQ.has_value() ? &seqLenQ.value() : nullptr,
                maxSeqLenKv,
                scale,
                c.leftBound,
                c.rightBound,
                c.topLeftAlignment,
                c.withLse ? &lseGpu : nullptr,
                nullptr,
                SdpaSoftmaxProbabilityMode::FLOAT,
                descales.q,
                descales.k,
                descales.v,
                {},
                sink);
        };
        if(c.sinks.empty())
        {
            runGpu(static_cast<Tensor<float>*>(nullptr), TypeTag<float>{});
        }
        else if(c.sinkInIoType)
        {
            runGpu(&sinkIo, TypeTag<T>{});
        }
        else
        {
            runGpu(&sinkFloat, TypeTag<float>{});
        }
    }

    const float tol = tolerance<T>();
    const OutT* og = oGpu.memory().hostData();
    const float* lg = lseGpu.memory().hostData();

    // 1. GPU vs CPU over the whole element space.
    {
        const OutT* oc = oCpu.memory().hostData();
        for(size_t i = 0; i < oGpu.elementSpace(); ++i)
        {
            expectClose(static_cast<float>(og[i]),
                        static_cast<float>(oc[i]),
                        tol,
                        "O vs CPU paged at element " + std::to_string(i));
        }
        const float* lc = lseCpu.memory().hostData();
        for(size_t i = 0; i < lseGpu.elementSpace(); ++i)
        {
            expectClose(lg[i], lc[i], tol, "LSE vs CPU paged at element " + std::to_string(i));
        }
    }

    // 2. Gather-then-dense oracle per batch, plus untouched padding rows.
    const auto* pk = pageTableK.memory().hostData();
    const auto* pv = pageTableVUsed.memory().hostData();
    const auto& lseStr = lseGpu.strides();
    const auto qRow = [&](int64_t b, int64_t s) {
        return raggedQ ? (cumQ[static_cast<size_t>(b)] + s) * qStrides[2]
                       : b * qStrides[0] + s * qStrides[2];
    };
    for(int64_t b = 0; b < batch; ++b)
    {
        const auto bIdx = static_cast<size_t>(b);
        const int64_t extent = qExtent[bIdx];
        const int64_t sQ = c.seqLenQ.has_value() ? (*c.seqLenQ)[bIdx] : extent;
        const int64_t sKv = c.seqLenKv[bIdx];

        for(int64_t s = sQ; s < extent; ++s)
        {
            for(int64_t hh = 0; hh < h; ++hh)
            {
                for(int64_t dd = 0; dd < d; ++dd)
                {
                    EXPECT_EQ(static_cast<float>(og[qRow(b, s) + hh * qStrides[1] + dd]),
                              O_SENTINEL)
                        << "padding row written: batch " << b << " row " << s;
                }
                if(c.withLse)
                {
                    EXPECT_EQ(lg[b * lseStr[0] + hh * lseStr[1] + s * lseStr[2]], LSE_SENTINEL)
                        << "padding LSE written: batch " << b << " row " << s;
                }
            }
        }
        if(sQ == 0)
        {
            continue;
        }
        if(sKv == 0)
        {
            for(int64_t s = 0; s < sQ; ++s)
            {
                for(int64_t hh = 0; hh < h; ++hh)
                {
                    for(int64_t dd = 0; dd < d; ++dd)
                    {
                        EXPECT_EQ(static_cast<float>(og[qRow(b, s) + hh * qStrides[1] + dd]), 0.0f)
                            << "empty-KV batch " << b << " row " << s;
                    }
                    if(c.withLse)
                    {
                        EXPECT_EQ(lg[b * lseStr[0] + hh * lseStr[1] + s * lseStr[2]],
                                  c.sinks.empty() ? -std::numeric_limits<float>::infinity()
                                                  : sinkValues[static_cast<size_t>(hh)]);
                    }
                }
            }
            continue;
        }

        // Dense float oracle on dequantized inputs: Q head hh reads its KV head's Q and K
        // descales, and V rows take the V head's descale.
        Tensor<float> qb({1, h, sQ, d});
        Tensor<float> kb({1, c.numHeadsK, sKv, d});
        Tensor<float> vb({1, c.numHeadsV, sKv, d});
        Tensor<float> ob({1, h, sQ, d});
        Tensor<float> lseb({1, h, sQ, 1});
        const T* qh = q.memory().hostData();
        for(int64_t s = 0; s < sQ; ++s)
        {
            for(int64_t hh = 0; hh < h; ++hh)
            {
                const float dq = descaleAt(descales.q, b, hh / (h / c.numHeadsK));
                for(int64_t dd = 0; dd < d; ++dd)
                {
                    qb.setHostValue(static_cast<float>(qh[qRow(b, s) + hh * qStrides[1] + dd]) * dq,
                                    0,
                                    hh,
                                    s,
                                    dd);
                }
            }
        }
        for(int64_t skv = 0; skv < sKv; ++skv)
        {
            const int64_t blockK = pk[b * maxBlocks + skv / bs];
            const int64_t blockV = pv[b * maxBlocks + skv / bs];
            for(int64_t hh = 0; hh < c.numHeadsK; ++hh)
            {
                const float dk = descaleAt(descales.k, b, hh);
                for(int64_t dd = 0; dd < d; ++dd)
                {
                    kb.setHostValue(static_cast<float>(kHost.getHostValue(blockK, hh, skv % bs, dd))
                                        * dk,
                                    0,
                                    hh,
                                    skv,
                                    dd);
                }
            }
            for(int64_t hh = 0; hh < c.numHeadsV; ++hh)
            {
                const float dv = descaleAt(descales.v, b, hh);
                for(int64_t dd = 0; dd < d; ++dd)
                {
                    vb.setHostValue(static_cast<float>(vHost.getHostValue(blockV, hh, skv % bs, dd))
                                        * dv,
                                    0,
                                    hh,
                                    skv,
                                    dd);
                }
            }
        }
        CpuFpReferenceSdpa::forward<float, float, float, float>(qb,
                                                                kb,
                                                                vb,
                                                                ob,
                                                                scale,
                                                                nullptr,
                                                                c.leftBound,
                                                                c.rightBound,
                                                                c.topLeftAlignment,
                                                                &lseb,
                                                                nullptr,
                                                                nullptr,
                                                                nullptr,
                                                                sinkPerHead);
        for(int64_t s = 0; s < sQ; ++s)
        {
            for(int64_t hh = 0; hh < h; ++hh)
            {
                for(int64_t dd = 0; dd < d; ++dd)
                {
                    expectClose(static_cast<float>(og[qRow(b, s) + hh * qStrides[1] + dd]),
                                ob.getHostValue(0, hh, s, dd),
                                tol,
                                "O vs gathered dense: batch " + std::to_string(b) + " row "
                                    + std::to_string(s));
                }
                if(c.withLse)
                {
                    expectClose(lg[b * lseStr[0] + hh * lseStr[1] + s * lseStr[2]],
                                lseb.getHostValue(0, hh, s, 0),
                                tol,
                                "LSE vs gathered dense: batch " + std::to_string(b) + " row "
                                    + std::to_string(s));
                }
            }
        }
    }
}

} // namespace

// Bottom-right causal decode (Sq = 1) over shuffled pages, with an empty batch (zeros, -inf).
TEST(TestGpuSdpaPagedFwdBf16, DecodeShuffledPagesMixedLengths)
{
    SKIP_IF_NO_DEVICES();
    PagedCase c;
    c.seqLenKv = {37, 0, 16, 5};
    c.rightBound = 0;
    c.topLeftAlignment = false;
    c.withLse = true;
    checkPaged<bfloat16>(c);
}

TEST(TestGpuSdpaPagedFwdFp16, DecodeShuffledPagesMixedLengthsD128)
{
    SKIP_IF_NO_DEVICES();
    PagedCase c;
    c.seqLenKv = {37, 0, 16, 5};
    c.headDim = 128;
    c.rightBound = 0;
    c.topLeftAlignment = false;
    c.withLse = true;
    checkPaged<half>(c);
}

// Sq = 4 bottom-right causal. seq_len_kv 3 < Sq leaves row 0 fully masked.
TEST(TestGpuSdpaPagedFwdBf16, CausalSq4)
{
    SKIP_IF_NO_DEVICES();
    PagedCase c;
    c.seqLenKv = {20, 3, 48};
    c.seqQMax = 4;
    c.rightBound = 0;
    c.topLeftAlignment = false;
    c.withLse = true;
    checkPaged<bfloat16>(c);
}

TEST(TestGpuSdpaPagedFwdFp16, GqaD128)
{
    SKIP_IF_NO_DEVICES();
    PagedCase c;
    c.seqLenKv = {33, 17, 64};
    c.seqQMax = 2;
    c.numHeads = 8;
    c.numHeadsK = 2;
    c.numHeadsV = 2;
    c.headDim = 128;
    checkPaged<half>(c);
}

TEST(TestGpuSdpaPagedFwdBf16, GqaD64)
{
    SKIP_IF_NO_DEVICES();
    PagedCase c;
    c.seqLenKv = {33, 1, 64};
    c.numHeads = 8;
    c.numHeadsK = 2;
    c.numHeadsV = 2;
    c.rightBound = 0;
    c.topLeftAlignment = false;
    checkPaged<bfloat16>(c);
}

TEST(TestGpuSdpaPagedFwdBf16, SlidingWindow)
{
    SKIP_IF_NO_DEVICES();
    PagedCase c;
    c.seqLenKv = {40, 9};
    c.seqQMax = 4;
    c.leftBound = 7;
    c.rightBound = 0;
    c.topLeftAlignment = false;
    c.withLse = true;
    checkPaged<bfloat16>(c);
}

// Ragged Q (vLLM varlen) with seq_len_q, a 0-length batch and a gap row in batch 0.
TEST(TestGpuSdpaPagedFwdBf16, RaggedQWithSeqLenQ)
{
    SKIP_IF_NO_DEVICES();
    PagedCase c;
    c.seqLenKv = {10, 7, 33};
    c.raggedQTokens = std::vector<int64_t>{3, 0, 5};
    c.seqLenQ = std::vector<int64_t>{2, 0, 5};
    c.seqQMax = 5;
    c.rightBound = 0;
    c.topLeftAlignment = false;
    c.withLse = true;
    checkPaged<bfloat16>(c);
}

TEST(TestGpuSdpaPagedFwdFp16, RaggedQWithSeqLenQD128)
{
    SKIP_IF_NO_DEVICES();
    PagedCase c;
    c.seqLenKv = {10, 7, 33};
    c.raggedQTokens = std::vector<int64_t>{3, 0, 5};
    c.seqLenQ = std::vector<int64_t>{3, 0, 4};
    c.seqQMax = 5;
    c.headDim = 128;
    c.rightBound = 0;
    c.topLeftAlignment = false;
    checkPaged<half>(c);
}

// Dense Q with seq_len_q: padding rows stay untouched, and an empty batch writes nothing.
TEST(TestGpuSdpaPagedFwdBf16, DenseQWithSeqLenQ)
{
    SKIP_IF_NO_DEVICES();
    PagedCase c;
    c.seqLenKv = {12, 5, 0};
    c.seqQMax = 4;
    c.seqLenQ = std::vector<int64_t>{4, 1, 0};
    c.rightBound = 0;
    c.topLeftAlignment = false;
    c.withLse = true;
    checkPaged<bfloat16>(c);
}

// One K/V buffer: per token and head, D K values then D V values (V at +D).
TEST(TestGpuSdpaPagedFwdBf16, InterleavedKvStrides)
{
    SKIP_IF_NO_DEVICES();
    PagedCase c;
    c.seqLenKv = {21, 16, 2};
    c.numHeadsK = 1;
    c.numHeadsV = 1;
    c.rightBound = 0;
    c.topLeftAlignment = false;
    c.interleavedKv = true;
    checkPaged<bfloat16>(c);
}

TEST(TestGpuSdpaPagedFwdFp16, SharedPageTable)
{
    SKIP_IF_NO_DEVICES();
    PagedCase c;
    c.seqLenKv = {50, 8};
    c.seqQMax = 3;
    c.blockSize = 32;
    c.sharedPageTable = true;
    c.withLse = true;
    checkPaged<half>(c);
}

// A used page-table entry outside [0, num_blocks) is rejected before launch.
TEST(TestGpuSdpaPagedFwdBf16, OutOfRangeUsedBlockThrows)
{
    SKIP_IF_NO_DEVICES();
    const int64_t numBlocks = 4;
    const int64_t bs = 16;
    const std::vector<int64_t> lens = {20, 5};
    const int64_t maxSeqLenKv = 32;
    Tensor<bfloat16> q(raggedDims(2, 1, 2, 64));
    Tensor<bfloat16> o(raggedDims(2, 1, 2, 64));
    Tensor<bfloat16> k({numBlocks, 2, bs, 64});
    Tensor<bfloat16> v({numBlocks, 2, bs, 64});
    auto pageTable = makePageTable(lens, bs, 2, numBlocks, SEED_PAGES_K);
    auto seqLenKv = makeSeqLens(lens);

    // Batch 1 entry 0 is used (seq_len_kv 5).
    pageTable.memory().hostData()[2] = static_cast<int32_t>(numBlocks);
    pageTable.memory().markHostModified();
    EXPECT_THROW(
        (GpuFpReferenceSdpaRagged::fpropPaged<bfloat16>(
            q, k, v, o, nullptr, nullptr, pageTable, pageTable, seqLenKv, nullptr, maxSeqLenKv)),
        std::invalid_argument);

    // An unused entry may hold any value: batch 1 entry 1 is past ceil(5 / 16).
    pageTable.memory().hostData()[2] = 0;
    pageTable.memory().hostData()[3] = 1234;
    pageTable.memory().markHostModified();
    EXPECT_NO_THROW((GpuFpReferenceSdpaRagged::fpropPaged<bfloat16>(
        q, k, v, o, nullptr, nullptr, pageTable, pageTable, seqLenKv, nullptr, maxSeqLenKv)));
}

TEST(TestGpuSdpaPagedFwdBf16, SeqLenKvAboveMaxThrows)
{
    SKIP_IF_NO_DEVICES();
    const int64_t bs = 16;
    Tensor<bfloat16> q(raggedDims(1, 1, 2, 64));
    Tensor<bfloat16> o(raggedDims(1, 1, 2, 64));
    Tensor<bfloat16> k({4, 2, bs, 64});
    Tensor<bfloat16> v({4, 2, bs, 64});
    auto pageTable = makePageTable({16}, bs, 1, 4, SEED_PAGES_K);
    auto seqLenKv = makeSeqLens({17});
    EXPECT_THROW((GpuFpReferenceSdpaRagged::fpropPaged<bfloat16>(
                     q, k, v, o, nullptr, nullptr, pageTable, pageTable, seqLenKv, nullptr, 16)),
                 std::invalid_argument);
}

// Attention sinks, GPU vs CPU and vs the gathered dense oracle with the same sinks. The empty
// batch must give O = 0 and LSE = sink.
TEST(TestGpuSdpaPagedFwdBf16, SinksGqaDecodeInIoType)
{
    SKIP_IF_NO_DEVICES();
    PagedCase c;
    c.seqLenKv = {37, 0, 16, 70};
    c.numHeads = 8;
    c.numHeadsK = 2;
    c.numHeadsV = 2;
    c.rightBound = 0;
    c.topLeftAlignment = false;
    c.withLse = true;
    c.sinks = {0.5f, -1.25f, 2.0f, -0.75f, 0.0f, 1.5f, -2.0f, 0.25f};
    c.sinkInIoType = true;
    checkPaged<bfloat16>(c);
}

TEST(TestGpuSdpaPagedFwdFp16, SinksFloatCausalSq4)
{
    SKIP_IF_NO_DEVICES();
    PagedCase c;
    c.seqLenKv = {20, 3, 48};
    c.seqQMax = 4;
    c.rightBound = 0;
    c.topLeftAlignment = false;
    c.withLse = true;
    c.sinks = {1.0f, -0.5f, 3.0f, -4.0f};
    checkPaged<half>(c);
}

TEST(TestGpuSdpaPagedFwdBf16, SinksRaggedQSlidingWindow)
{
    SKIP_IF_NO_DEVICES();
    PagedCase c;
    c.seqLenKv = {10, 0, 40};
    c.raggedQTokens = std::vector<int64_t>{3, 2, 6};
    c.seqLenQ = std::vector<int64_t>{2, 2, 6};
    c.seqQMax = 6;
    c.leftBound = 5;
    c.rightBound = 0;
    c.topLeftAlignment = false;
    c.withLse = true;
    c.sinks = {0.25f, -1.0f, 1.75f, 0.5f};
    c.sinkInIoType = true;
    checkPaged<bfloat16>(c);
}

TEST(TestGpuSdpaPagedFwdBf16, SinkShapeMismatchThrows)
{
    SKIP_IF_NO_DEVICES();
    Tensor<bfloat16> q({1, 2, 1, 16});
    Tensor<bfloat16> k({1, 2, 16, 16});
    Tensor<bfloat16> v({1, 2, 16, 16});
    Tensor<bfloat16> o({1, 2, 1, 16});
    auto pageTable = makeSeqLens({0});
    auto seqLenKv = makeSeqLens({1});
    Tensor<float> sink({1, 1, 1, 1}); // one sink for two query heads
    EXPECT_THROW((GpuFpReferenceSdpaRagged::fpropPaged<bfloat16>(q,
                                                                 k,
                                                                 v,
                                                                 o,
                                                                 nullptr,
                                                                 nullptr,
                                                                 pageTable,
                                                                 pageTable,
                                                                 seqLenKv,
                                                                 nullptr,
                                                                 16,
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
                                                                 {},
                                                                 &sink)),
                 std::invalid_argument);
}

// --- fp8 E4M3 Q/K/V with a bf16 output, the fp8 tuple the GPU paged plan serves. Both
// references and the dense oracle get the same descales; Q, K and V descales differ, so a
// descale applied to the wrong operand changes the result. ---

TEST(TestGpuSdpaPagedFwdFp8, ScalarDescalesDecode)
{
    SKIP_IF_NO_DEVICES();
    PagedCase c;
    c.seqLenKv = {37, 0, 16, 5};
    c.rightBound = 0;
    c.topLeftAlignment = false;
    c.withLse = true;
    auto descaleQ = makeScalarDescale(0.5f);
    auto descaleK = makeScalarDescale(0.25f);
    auto descaleV = makeScalarDescale(2.0f);
    checkPaged<fp8_e4m3, bfloat16>(c, {&descaleQ, &descaleK, &descaleV});
}

// AITER's [B, H_kv] descales under GQA, with ragged Q. Distinct Q values per KV head catch a Q
// descale indexed by query head.
TEST(TestGpuSdpaPagedFwdFp8, PerKvHeadDescalesGqaRaggedQ)
{
    SKIP_IF_NO_DEVICES();
    PagedCase c;
    c.seqLenKv = {10, 7, 33};
    c.raggedQTokens = std::vector<int64_t>{3, 0, 5};
    c.seqLenQ = std::vector<int64_t>{2, 0, 5};
    c.seqQMax = 5;
    c.numHeads = 8;
    c.numHeadsK = 2;
    c.numHeadsV = 2;
    c.headDim = 128;
    c.rightBound = 0;
    c.topLeftAlignment = false;
    c.withLse = true;
    const auto batch = static_cast<int64_t>(c.seqLenKv.size());
    auto descaleQ = makePerHeadDescale(batch, c.numHeadsK, 0.4f);
    auto descaleK = makePerHeadDescale(batch, c.numHeadsK, 0.2f);
    auto descaleV = makePerHeadDescale(batch, c.numHeadsV, 0.3f);
    checkPaged<fp8_e4m3, bfloat16>(c, {&descaleQ, &descaleK, &descaleV});
}

namespace
{

constexpr int64_t CHECK_HEADS = 2;
constexpr int64_t CHECK_HEAD_DIM = 64;
constexpr int64_t CHECK_BLOCK_SIZE = 16;
constexpr int64_t CHECK_NUM_BLOCKS = 4;
constexpr int64_t CHECK_MAX_SEQ_LEN_KV = 32; // page tables two blocks wide
constexpr int64_t CHECK_MAX_BLOCKS = 2;

// A valid problem for the argument checks: B = 2, seq_len_kv {20, 5} on a pool of four blocks.
// Dense Q holds 2 rows per batch; ragged Q packs {3, 1} tokens with seq_len_q {3, 1}.
struct PagedArgs
{
    explicit PagedArgs(bool ragged)
        : raggedQ(ragged)
        , qDims(raggedDims(2, ragged ? 3 : 2, CHECK_HEADS, CHECK_HEAD_DIM))
        , q(qDims, raggedStrides(qDims))
        , o(qDims, raggedStrides(qDims))
        , k({CHECK_NUM_BLOCKS, CHECK_HEADS, CHECK_BLOCK_SIZE, CHECK_HEAD_DIM})
        , v({CHECK_NUM_BLOCKS, CHECK_HEADS, CHECK_BLOCK_SIZE, CHECK_HEAD_DIM})
        , pageTableK(makePageTable(
              {20, 5}, CHECK_BLOCK_SIZE, CHECK_MAX_BLOCKS, CHECK_NUM_BLOCKS, SEED_PAGES_K))
        , pageTableV(makePageTable(
              {20, 5}, CHECK_BLOCK_SIZE, CHECK_MAX_BLOCKS, CHECK_NUM_BLOCKS, SEED_PAGES_V))
        , seqLenKv(makeSeqLens({20, 5}))
        , offQ(makeRaggedOffset({0, 3, 4}, CHECK_HEADS * CHECK_HEAD_DIM))
        , offO(makeRaggedOffset({0, 3, 4}, CHECK_HEADS * CHECK_HEAD_DIM))
    {
        if(ragged)
        {
            seqLenQ.emplace(makeSeqLens({3, 1}));
        }
    }

    void run()
    {
        GpuFpReferenceSdpaRagged::fpropPaged<bfloat16>(q,
                                                       k,
                                                       v,
                                                       o,
                                                       raggedQ ? &offQ : nullptr,
                                                       raggedQ ? &offO : nullptr,
                                                       pageTableK,
                                                       pageTableV,
                                                       seqLenKv,
                                                       seqLenQ.has_value() ? &seqLenQ.value()
                                                                           : nullptr,
                                                       CHECK_MAX_SEQ_LEN_KV);
    }

    bool raggedQ;
    std::vector<int64_t> qDims;
    Tensor<bfloat16> q;
    Tensor<bfloat16> o;
    Tensor<bfloat16> k;
    Tensor<bfloat16> v;
    Tensor<int32_t> pageTableK;
    Tensor<int32_t> pageTableV;
    Tensor<int32_t> seqLenKv;
    Tensor<int32_t> offQ;
    Tensor<int32_t> offO;
    std::optional<Tensor<int32_t>> seqLenQ;
};

} // namespace

// The unmodified argument-check problems run, so each rejection below comes from its one change.
TEST(TestGpuSdpaPagedFwdBf16, ArgumentCheckBaselineRuns)
{
    SKIP_IF_NO_DEVICES();
    for(const bool ragged : {false, true})
    {
        PagedArgs args(ragged);
        EXPECT_NO_THROW(args.run()) << "raggedQ=" << ragged;
    }
}

TEST(TestGpuSdpaPagedFwdBf16, NegativeUsedBlockIdInVTableOnlyThrows)
{
    SKIP_IF_NO_DEVICES();
    PagedArgs args(/*ragged=*/false);
    // Batch 0 entry 1 is used: 20 keys need two blocks of 16.
    args.pageTableV.memory().hostData()[1] = -1;
    args.pageTableV.memory().markHostModified();
    EXPECT_THROW(args.run(), std::invalid_argument);
}

TEST(TestGpuSdpaPagedFwdBf16, SeqLenQAboveDenseExtentThrows)
{
    SKIP_IF_NO_DEVICES();
    PagedArgs args(/*ragged=*/false);
    args.seqLenQ.emplace(makeSeqLens({2, 3})); // dense Q holds 2 rows per batch
    EXPECT_THROW(args.run(), std::invalid_argument);
}

TEST(TestGpuSdpaPagedFwdBf16, SeqLenQAboveRaggedExtentThrows)
{
    SKIP_IF_NO_DEVICES();
    PagedArgs args(/*ragged=*/true);
    args.seqLenQ.emplace(makeSeqLens({3, 2})); // batch 1 holds 1 token
    EXPECT_THROW(args.run(), std::invalid_argument);
}

TEST(TestGpuSdpaPagedFwdBf16, RaggedQWithoutSeqLenQThrows)
{
    SKIP_IF_NO_DEVICES();
    PagedArgs args(/*ragged=*/true);
    args.seqLenQ.reset();
    EXPECT_THROW(args.run(), std::invalid_argument);
}

// max_seq_len_kv 32 with blocks of 16 needs page tables exactly two blocks wide.
TEST(TestGpuSdpaPagedFwdBf16, PageTableOneBlockTooWideThrows)
{
    SKIP_IF_NO_DEVICES();
    PagedArgs args(/*ragged=*/false);
    args.pageTableK = makePageTable(
        {20, 5}, CHECK_BLOCK_SIZE, CHECK_MAX_BLOCKS + 1, CHECK_NUM_BLOCKS, SEED_PAGES_K);
    args.pageTableV = makePageTable(
        {20, 5}, CHECK_BLOCK_SIZE, CHECK_MAX_BLOCKS + 1, CHECK_NUM_BLOCKS, SEED_PAGES_V);
    EXPECT_THROW(args.run(), std::invalid_argument);
}
