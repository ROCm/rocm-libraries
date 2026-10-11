// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Host-only behaviour tests for test_sdpa_common::fillForwardOutputsForBackward,
// the helper the shared SDPA backward integration test uses to give its graph an
// O and an LSE that belong to Q, K and V.
//
// The invariant under test is the one the backward pass depends on: the LSE the
// helper writes is the normalizer of the softmax that the backward recomputes, so
// sum_j exp(s_ij - lse_i) over the unmasked band is 1, and O is that softmax
// applied to V. Random O and LSE values satisfy neither.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>

#include "common/SdpaCommon.hpp"

namespace
{

using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::utilities::generateStrides;
using hipdnn_data_sdk::utilities::ShallowTensor;
using test_sdpa_common::SdpaBwdTestCase;

// Written into O and the stats before the helper runs. Any element that still
// holds it afterwards was never produced by the forward.
constexpr float K_SENTINEL = 1.0e9F;

int64_t elementCount(const std::vector<int64_t>& dims)
{
    int64_t count = 1;
    for(const auto dim : dims)
    {
        count *= dim;
    }
    return count;
}

template <class DataType>
void fillDeterministic(std::vector<DataType>& buffer, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for(auto& value : buffer)
    {
        value = static_cast<DataType>(dist(rng));
    }
}

// Independent restatement of the band CpuFpReferenceSdpa keeps: the diagonal sits
// at skv == sq for top-left alignment and at skv == sq + (Skv - Sq) for
// bottom-right, and a bound of -1 leaves that side unbounded.
bool isOutsideBand(int64_t sq, int64_t skv, int64_t seqQ, int64_t seqKv, const SdpaBwdTestCase& tc)
{
    const int64_t offset = tc.topLeftAlignment ? 0 : seqKv - seqQ;
    if(tc.rightBound >= 0 && skv > sq + offset + tc.rightBound)
    {
        return true;
    }
    return tc.leftBound >= 0 && skv < sq + offset - tc.leftBound;
}

// Q, K, V and the helper's O/stats for one test case. Q, K and V depend only on
// the seed, so two cases built with the same seed share their inputs exactly.
template <class DataType>
struct ForwardOutputs
{
    explicit ForwardOutputs(SdpaBwdTestCase testCase, uint32_t seed = 0)
        : tc(std::move(testCase))
        , oDims{tc.qDims[0], tc.qDims[1], tc.qDims[2], tc.vDims[3]}
        , statsDims{tc.qDims[0], tc.qDims[1], tc.qDims[2], 1}
        , q(static_cast<size_t>(elementCount(tc.qDims)))
        , k(static_cast<size_t>(elementCount(tc.kDims)))
        , v(static_cast<size_t>(elementCount(tc.vDims)))
        , o(static_cast<size_t>(elementCount(oDims)), static_cast<DataType>(K_SENTINEL))
        , stats(static_cast<size_t>(elementCount(statsDims)), K_SENTINEL)
    {
        fillDeterministic(q, seed + 1U);
        fillDeterministic(k, seed + 2U);
        fillDeterministic(v, seed + 3U);

        const ShallowTensor<DataType> qT(q.data(), tc.qDims, tc.qStrides);
        const ShallowTensor<DataType> kT(k.data(), tc.kDims, tc.kStrides);
        const ShallowTensor<DataType> vT(v.data(), tc.vDims, tc.vStrides);
        ShallowTensor<DataType> oT(o.data(), oDims, generateStrides(oDims));
        ShallowTensor<float> statsT(stats.data(), statsDims, generateStrides(statsDims));

        test_sdpa_common::fillForwardOutputsForBackward<DataType>(qT, kT, vT, oT, statsT, tc);
    }

    // Scores of row (b, h, sq) recomputed in double from the same Q and K, at
    // `scale` instead of the case's own scale.
    std::vector<double> scoresAt(int64_t b, int64_t h, int64_t sq, double scale) const
    {
        const int64_t headsPerKvHead = tc.qDims[1] / tc.kDims[1];
        const int64_t kvHead = h / headsPerKvHead;
        const int64_t headDim = tc.qDims[3];
        const int64_t seqKv = tc.kDims[2];

        std::vector<double> scores(static_cast<size_t>(seqKv), 0.0);
        for(int64_t skv = 0; skv < seqKv; ++skv)
        {
            double dot = 0.0;
            for(int64_t d = 0; d < headDim; ++d)
            {
                dot += static_cast<double>(at(q, tc.qStrides, {b, h, sq, d}))
                       * static_cast<double>(at(k, tc.kStrides, {b, kvHead, skv, d}));
            }
            scores[static_cast<size_t>(skv)] = dot * scale;
        }
        return scores;
    }

    double lseAt(int64_t b, int64_t h, int64_t sq) const
    {
        return static_cast<double>(at(stats, generateStrides(statsDims), {b, h, sq, 0}));
    }

    double outputAt(int64_t b, int64_t h, int64_t sq, int64_t dv) const
    {
        return static_cast<double>(at(o, generateStrides(oDims), {b, h, sq, dv}));
    }

    double valueAt(int64_t b, int64_t kvHead, int64_t skv, int64_t dv) const
    {
        return static_cast<double>(at(v, tc.vStrides, {b, kvHead, skv, dv}));
    }

    SdpaBwdTestCase tc;
    std::vector<int64_t> oDims;
    std::vector<int64_t> statsDims;
    std::vector<DataType> q;
    std::vector<DataType> k;
    std::vector<DataType> v;
    std::vector<DataType> o;
    std::vector<float> stats;

private:
    template <class T>
    static T at(const std::vector<T>& buffer,
                const std::vector<int64_t>& strides,
                const std::vector<int64_t>& indices)
    {
        int64_t offset = 0;
        for(size_t i = 0; i < indices.size(); ++i)
        {
            offset += indices[i] * strides[i];
        }
        return buffer[static_cast<size_t>(offset)];
    }
};

// Every mask regime the shipped backward cases use, at dims small enough for an
// independent double-precision recompute, plus an explicit attention scale (the
// shipped cases all leave it unset, which the reference reads as 1.0).
std::vector<std::pair<const char*, SdpaBwdTestCase>> maskRegimes()
{
    return {
        {"nomask", SdpaBwdTestCase({1, 2, 16, 8}, {1, 2, 16, 8})},
        {"scaled_nomask", SdpaBwdTestCase({1, 2, 16, 8}, {1, 2, 16, 8}, 0.3535F)},
        {"causal_top_left",
         SdpaBwdTestCase({1, 2, 16, 8}, {1, 2, 16, 8}, std::nullopt, -1, 0, true)},
        {"causal_bottom_right",
         SdpaBwdTestCase({1, 2, 16, 8}, {1, 2, 16, 8}, std::nullopt, -1, 0, false)},
        {"window_top_left",
         SdpaBwdTestCase({1, 2, 16, 8}, {1, 2, 16, 8}, std::nullopt, 4, 4, true)},
        {"window_bottom_right",
         SdpaBwdTestCase({1, 2, 16, 8}, {1, 2, 16, 8}, std::nullopt, 4, 4, false)},
        {"asymmetric_nomask", SdpaBwdTestCase({1, 2, 16, 8}, {1, 2, 24, 8})},
        {"asymmetric_causal_bottom_right",
         SdpaBwdTestCase({1, 2, 16, 8}, {1, 2, 24, 8}, std::nullopt, -1, 0, false)},
    };
}

} // namespace

// The LSE normalizes the band's softmax and O is that softmax applied to V. The
// masked positions contribute nothing, so a row that drops half its band still
// sums to 1 rather than to the fraction it kept.
TEST(TestSdpaBackwardInputs, ForwardOutputsNormalizeTheRecomputedSoftmax)
{
    for(const auto& [name, tc] : maskRegimes())
    {
        SCOPED_TRACE(name);
        const ForwardOutputs<float> outputs(tc);

        const int64_t batch = tc.qDims[0];
        const int64_t numHeads = tc.qDims[1];
        const int64_t seqQ = tc.qDims[2];
        const int64_t seqKv = tc.kDims[2];
        const int64_t headDimV = tc.vDims[3];
        const int64_t kvHeads = tc.kDims[1];
        const double scale = static_cast<double>(tc.attnScaleValue.value_or(1.0F));

        for(int64_t b = 0; b < batch; ++b)
        {
            for(int64_t h = 0; h < numHeads; ++h)
            {
                for(int64_t sq = 0; sq < seqQ; ++sq)
                {
                    SCOPED_TRACE("b=" + std::to_string(b) + " h=" + std::to_string(h)
                                 + " sq=" + std::to_string(sq));
                    const auto scores = outputs.scoresAt(b, h, sq, scale);
                    const double lse = outputs.lseAt(b, h, sq);
                    ASSERT_NE(lse, static_cast<double>(K_SENTINEL)) << "stats left unwritten";
                    ASSERT_TRUE(std::isfinite(lse)) << "every row here keeps at least one position";

                    std::vector<double> probs(static_cast<size_t>(seqKv), 0.0);
                    double sum = 0.0;
                    for(int64_t skv = 0; skv < seqKv; ++skv)
                    {
                        if(isOutsideBand(sq, skv, seqQ, seqKv, tc))
                        {
                            continue;
                        }
                        probs[static_cast<size_t>(skv)]
                            = std::exp(scores[static_cast<size_t>(skv)] - lse);
                        sum += probs[static_cast<size_t>(skv)];
                    }
                    EXPECT_NEAR(sum, 1.0, 1e-5) << "lse is not the band's log-sum-exp";

                    const int64_t kvHead = h / (numHeads / kvHeads);
                    for(int64_t dv = 0; dv < headDimV; ++dv)
                    {
                        double expected = 0.0;
                        for(int64_t skv = 0; skv < seqKv; ++skv)
                        {
                            expected += probs[static_cast<size_t>(skv)]
                                        * outputs.valueAt(b, kvHead, skv, dv);
                        }
                        const double actual = outputs.outputAt(b, h, sq, dv);
                        ASSERT_NE(actual, static_cast<double>(K_SENTINEL)) << "O left unwritten";
                        EXPECT_NEAR(actual, expected, 1e-5) << "dv=" << dv;
                    }
                }
            }
        }
    }
}

// The same invariant through the bf16 instantiation the shared fixture uses for
// IntegrationGpuSdpaBwdBfp16. Q and K are bf16, so the recomputed scores carry
// their rounding; the LSE stays float and still normalizes them.
TEST(TestSdpaBackwardInputs, Bfloat16InputsStillNormalize)
{
    const SdpaBwdTestCase tc({1, 2, 16, 8}, {1, 2, 16, 8}, std::nullopt, -1, 0, true);
    const ForwardOutputs<bfloat16> outputs(tc);

    for(int64_t h = 0; h < tc.qDims[1]; ++h)
    {
        for(int64_t sq = 0; sq < tc.qDims[2]; ++sq)
        {
            SCOPED_TRACE("h=" + std::to_string(h) + " sq=" + std::to_string(sq));
            const auto scores = outputs.scoresAt(0, h, sq, 1.0);
            const double lse = outputs.lseAt(0, h, sq);
            double sum = 0.0;
            for(int64_t skv = 0; skv < tc.kDims[2]; ++skv)
            {
                if(!isOutsideBand(sq, skv, tc.qDims[2], tc.kDims[2], tc))
                {
                    sum += std::exp(scores[static_cast<size_t>(skv)] - lse);
                }
            }
            EXPECT_NEAR(sum, 1.0, 1e-4);
        }
    }
}

// Liveness control for the test above. The LSE belongs to one scale; evaluating
// the same rows at a different scale must leave the sum far from 1, otherwise
// ForwardOutputsNormalizeTheRecomputedSoftmax would pass on any LSE at all.
TEST(TestSdpaBackwardInputs, AMismatchedScaleLeavesTheSumFarFromOne)
{
    const SdpaBwdTestCase tc({1, 2, 16, 8}, {1, 2, 16, 8});
    const ForwardOutputs<float> outputs(tc);

    double worstDeviation = 0.0;
    for(int64_t h = 0; h < tc.qDims[1]; ++h)
    {
        for(int64_t sq = 0; sq < tc.qDims[2]; ++sq)
        {
            const auto scores = outputs.scoresAt(0, h, sq, 4.0);
            const double lse = outputs.lseAt(0, h, sq);
            double sum = 0.0;
            for(int64_t skv = 0; skv < tc.kDims[2]; ++skv)
            {
                sum += std::exp(scores[static_cast<size_t>(skv)] - lse);
            }
            worstDeviation = std::max(worstDeviation, std::abs(sum - 1.0));
        }
    }
    EXPECT_GT(worstDeviation, 0.1) << "the normalizer check is insensitive to the scores";
}

// The helper forwards the test case's scale and band bounds to the reference
// rather than defaulting them, so changing either changes the LSE it writes.
TEST(TestSdpaBackwardInputs, ScaleAndBandBoundsReachTheReference)
{
    const SdpaBwdTestCase nomask({1, 2, 16, 8}, {1, 2, 16, 8});
    const SdpaBwdTestCase scaled({1, 2, 16, 8}, {1, 2, 16, 8}, 0.3535F);
    const SdpaBwdTestCase causal({1, 2, 16, 8}, {1, 2, 16, 8}, std::nullopt, -1, 0, true);

    const ForwardOutputs<float> nomaskOutputs(nomask);
    const ForwardOutputs<float> scaledOutputs(scaled);
    const ForwardOutputs<float> causalOutputs(causal);

    ASSERT_EQ(nomaskOutputs.q, scaledOutputs.q) << "the three cases must share their inputs";
    ASSERT_EQ(nomaskOutputs.q, causalOutputs.q);

    EXPECT_NE(nomaskOutputs.stats, scaledOutputs.stats) << "attn_scale was ignored";
    EXPECT_NE(nomaskOutputs.stats, causalOutputs.stats) << "the band bounds were ignored";

    // Row 0 of a top-left causal mask keeps a single position, so its LSE is that
    // position's score and its output is that position's V row.
    const auto scores = causalOutputs.scoresAt(0, 0, 0, 1.0);
    EXPECT_NEAR(causalOutputs.lseAt(0, 0, 0), scores[0], 1e-5);
    for(int64_t dv = 0; dv < causal.vDims[3]; ++dv)
    {
        EXPECT_NEAR(causalOutputs.outputAt(0, 0, 0, dv), causalOutputs.valueAt(0, 0, 0, dv), 1e-5)
            << "dv=" << dv;
    }
}
