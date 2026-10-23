// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Validation of CpuFpReferenceSdpa that does not depend on another implementation of SDPA:
//
//  - Closed-form oracles: inputs whose exact output is known analytically.
//  - Properties: two graphs that must produce the same result because one feature is defined in
//    terms of another (GQA as repeated KV heads, a band mask as a -inf bias, and so on).
//
// The semantics pinned here are the cuDNN contract recorded in
// projects/hipdnn/docs/SdpaSemanticsRegister.md. Sliding-window left bounds are deliberately not
// covered while SEM-001 is open.

#include <cmath>
#include <cstdint>
#include <functional>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;

namespace
{

constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
constexpr float TOL = 1e-5F;

void forEachIndex(const std::vector<int64_t>& dims,
                  const std::function<void(const std::vector<int64_t>&)>& fn)
{
    for(int64_t a = 0; a < dims[0]; ++a)
    {
        for(int64_t b = 0; b < dims[1]; ++b)
        {
            for(int64_t c = 0; c < dims[2]; ++c)
            {
                for(int64_t d = 0; d < dims[3]; ++d)
                {
                    fn({a, b, c, d});
                }
            }
        }
    }
}

void expectAllNear(const TensorBase<float>& actual,
                   const TensorBase<float>& expected,
                   float tol,
                   const char* what)
{
    ASSERT_EQ(actual.dims(), expected.dims()) << what;
    forEachIndex(actual.dims(), [&](const std::vector<int64_t>& i) {
        EXPECT_NEAR(actual.getHostValue(i), expected.getHostValue(i), tol)
            << what << " at [" << i[0] << "," << i[1] << "," << i[2] << "," << i[3] << "]";
    });
}

void expectAllFinite(const TensorBase<float>& t, const char* what)
{
    forEachIndex(t.dims(), [&](const std::vector<int64_t>& i) {
        EXPECT_TRUE(std::isfinite(t.getHostValue(i)))
            << what << " at [" << i[0] << "," << i[1] << "," << i[2] << "," << i[3] << "]";
    });
}

// Forward + backward of one SDPA problem, all tensors float, LSE always produced.
struct SdpaRun
{
    SdpaRun(const std::vector<int64_t>& qDims,
            const std::vector<int64_t>& kDims,
            const std::vector<int64_t>& vDims)
        : q(qDims)
        , k(kDims)
        , v(vDims)
        , o({qDims[0], qDims[1], qDims[2], vDims[3]})
        , lse({qDims[0], qDims[1], qDims[2], 1})
        , dO({qDims[0], qDims[1], qDims[2], vDims[3]})
        , dQ(qDims)
        , dK(kDims)
        , dV(vDims)
    {
    }

    void run(std::optional<float> scale = std::nullopt,
             const TensorBase<float>* mask = nullptr,
             int64_t leftBound = -1,
             int64_t rightBound = -1,
             bool topLeft = true,
             bool backwardUsesLse = true)
    {
        CpuFpReferenceSdpa::forward(q, k, v, o, scale, mask, leftBound, rightBound, topLeft, &lse);
        CpuFpReferenceSdpa::backward(q,
                                     k,
                                     v,
                                     o,
                                     dO,
                                     dQ,
                                     dK,
                                     dV,
                                     scale,
                                     backwardUsesLse ? &lse : nullptr,
                                     mask,
                                     leftBound,
                                     rightBound,
                                     topLeft);
    }

    Tensor<float> q;
    Tensor<float> k;
    Tensor<float> v;
    Tensor<float> o;
    Tensor<float> lse;
    Tensor<float> dO;
    Tensor<float> dQ;
    Tensor<float> dK;
    Tensor<float> dV;
};

} // namespace

// ---------------------------------------------------------------------------
// Closed-form oracles
// ---------------------------------------------------------------------------

// Q = 0 makes every score 0, so softmax is uniform: O[i] = mean_j V[j] and LSE = ln(Skv).
// The value gradient is then dV[j] = (1/Skv) * sum_i dO[i].
TEST(TestCpuFpReferenceSdpaClosedForm, UniformScores)
{
    constexpr int64_t SQ = 3;
    constexpr int64_t SKV = 5;
    SdpaRun r({1, 1, SQ, 4}, {1, 1, SKV, 4}, {1, 1, SKV, 3});
    r.q.fillWithValue(0.0F);
    r.k.fillWithRandomValues(-1.0F, 1.0F, 11);
    r.v.fillWithRandomValues(-1.0F, 1.0F, 12);
    r.dO.fillWithRandomValues(-1.0F, 1.0F, 13);
    r.run();

    for(int64_t sq = 0; sq < SQ; ++sq)
    {
        EXPECT_NEAR(r.lse.getHostValue(0, 0, sq, 0), std::log(static_cast<float>(SKV)), TOL);
        for(int64_t dv = 0; dv < 3; ++dv)
        {
            float mean = 0.0F;
            for(int64_t skv = 0; skv < SKV; ++skv)
            {
                mean += r.v.getHostValue(0, 0, skv, dv);
            }
            mean /= static_cast<float>(SKV);
            EXPECT_NEAR(r.o.getHostValue(0, 0, sq, dv), mean, TOL);
        }
    }

    for(int64_t skv = 0; skv < SKV; ++skv)
    {
        for(int64_t dv = 0; dv < 3; ++dv)
        {
            float sumDo = 0.0F;
            for(int64_t sq = 0; sq < SQ; ++sq)
            {
                sumDo += r.dO.getHostValue(0, 0, sq, dv);
            }
            EXPECT_NEAR(r.dV.getHostValue(0, 0, skv, dv), sumDo / static_cast<float>(SKV), TOL);
        }
    }
}

// Sixteen equal scores with one raised by 0.25: LSE = ln(15 + e^0.25). A natural-log/log2 mix-up
// or a dropped max term shows up here.
TEST(TestCpuFpReferenceSdpaClosedForm, OneRaisedScoreLse)
{
    constexpr int64_t SKV = 16;
    constexpr int64_t RAISED = 5;
    SdpaRun r({1, 1, 1, 4}, {1, 1, SKV, 4}, {1, 1, SKV, 2});
    r.q.fillWithValue(0.0F);
    r.k.fillWithRandomValues(-1.0F, 1.0F, 21);
    r.v.fillWithRandomValues(-1.0F, 1.0F, 22);
    r.dO.fillWithRandomValues(-1.0F, 1.0F, 23);

    Tensor<float> bias({1, SKV});
    bias.fillWithValue(0.0F);
    bias.setHostValue(0.25F, 0, RAISED);
    r.run(std::nullopt, &bias);

    const float e = std::exp(0.25F);
    EXPECT_NEAR(r.lse.getHostValue(0, 0, 0, 0), std::log(15.0F + e), TOL);
    for(int64_t dv = 0; dv < 2; ++dv)
    {
        float expected = 0.0F;
        for(int64_t skv = 0; skv < SKV; ++skv)
        {
            const float w = (skv == RAISED) ? e : 1.0F;
            expected += w * r.v.getHostValue(0, 0, skv, dv);
        }
        expected /= (15.0F + e);
        EXPECT_NEAR(r.o.getHostValue(0, 0, 0, dv), expected, TOL);
    }
}

// With every key but one masked by a -inf bias, O copies that key's value row and LSE is its
// (scaled) score.
TEST(TestCpuFpReferenceSdpaClosedForm, SingleLiveKeyCopiesValue)
{
    constexpr int64_t SKV = 6;
    constexpr int64_t LIVE = 2;
    constexpr float SCALE = 0.5F;
    SdpaRun r({1, 1, 1, 4}, {1, 1, SKV, 4}, {1, 1, SKV, 3});
    r.q.fillWithRandomValues(-1.0F, 1.0F, 31);
    r.k.fillWithRandomValues(-1.0F, 1.0F, 32);
    r.v.fillWithRandomValues(-1.0F, 1.0F, 33);
    r.dO.fillWithRandomValues(-1.0F, 1.0F, 34);

    Tensor<float> bias({1, SKV});
    bias.fillWithValue(NEG_INF);
    bias.setHostValue(0.0F, 0, LIVE);
    r.run(SCALE, &bias);

    float score = 0.0F;
    for(int64_t d = 0; d < 4; ++d)
    {
        score += r.q.getHostValue(0, 0, 0, d) * r.k.getHostValue(0, 0, LIVE, d);
    }
    EXPECT_NEAR(r.lse.getHostValue(0, 0, 0, 0), score * SCALE, TOL);
    for(int64_t dv = 0; dv < 3; ++dv)
    {
        EXPECT_NEAR(r.o.getHostValue(0, 0, 0, dv), r.v.getHostValue(0, 0, LIVE, dv), TOL);
    }
    // A one-hot softmax has zero Jacobian, so no gradient reaches Q or K.
    Tensor<float> zerosQ(r.dQ.dims());
    zerosQ.fillWithValue(0.0F);
    Tensor<float> zerosK(r.dK.dims());
    zerosK.fillWithValue(0.0F);
    expectAllNear(r.dQ, zerosQ, TOL, "dQ");
    expectAllNear(r.dK, zerosK, TOL, "dK");
}

// A row with no live key: O = 0, LSE = -inf, and the row contributes nothing to any gradient
// (register SEM-010). The expected gradients come from the same problem with that row removed.
class TestCpuFpReferenceSdpaClosedFormDeadRow : public ::testing::TestWithParam<bool>
{
};

TEST_P(TestCpuFpReferenceSdpaClosedFormDeadRow, ContributesNothing)
{
    const bool backwardUsesLse = GetParam();
    constexpr int64_t SQ = 3;
    constexpr int64_t SKV = 4;
    SdpaRun full({1, 1, SQ, 4}, {1, 1, SKV, 4}, {1, 1, SKV, 4});
    full.q.fillWithRandomValues(-1.0F, 1.0F, 41);
    full.k.fillWithRandomValues(-1.0F, 1.0F, 42);
    full.v.fillWithRandomValues(-1.0F, 1.0F, 43);
    full.dO.fillWithRandomValues(-1.0F, 1.0F, 44);

    // Row 0 is fully masked; rows 1 and 2 see every key.
    Tensor<float> bias({SQ, SKV});
    bias.fillWithValue(0.0F);
    for(int64_t skv = 0; skv < SKV; ++skv)
    {
        bias.setHostValue(NEG_INF, 0, skv);
    }
    full.run(std::nullopt, &bias, -1, -1, true, backwardUsesLse);

    EXPECT_TRUE(std::isinf(full.lse.getHostValue(0, 0, 0, 0))
                && full.lse.getHostValue(0, 0, 0, 0) < 0);
    for(int64_t d = 0; d < 4; ++d)
    {
        EXPECT_EQ(full.o.getHostValue(0, 0, 0, d), 0.0F);
        EXPECT_EQ(full.dQ.getHostValue(0, 0, 0, d), 0.0F);
    }
    expectAllFinite(full.dQ, "dQ");
    expectAllFinite(full.dK, "dK");
    expectAllFinite(full.dV, "dV");

    // Same problem without row 0.
    SdpaRun live({1, 1, SQ - 1, 4}, {1, 1, SKV, 4}, {1, 1, SKV, 4});
    forEachIndex(live.q.dims(), [&](const std::vector<int64_t>& i) {
        live.q.setHostValue(full.q.getHostValue(i[0], i[1], i[2] + 1, i[3]), i);
        live.dO.setHostValue(full.dO.getHostValue(i[0], i[1], i[2] + 1, i[3]), i);
    });
    forEachIndex(live.k.dims(), [&](const std::vector<int64_t>& i) {
        live.k.setHostValue(full.k.getHostValue(i), i);
        live.v.setHostValue(full.v.getHostValue(i), i);
    });
    live.run(std::nullopt, nullptr, -1, -1, true, backwardUsesLse);

    expectAllNear(full.dK, live.dK, TOL, "dK");
    expectAllNear(full.dV, live.dV, TOL, "dV");
    forEachIndex(live.dQ.dims(), [&](const std::vector<int64_t>& i) {
        EXPECT_NEAR(full.dQ.getHostValue(i[0], i[1], i[2] + 1, i[3]), live.dQ.getHostValue(i), TOL);
    });
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuFpReferenceSdpaClosedFormDeadRow,
                         ::testing::Values(true, false),
                         [](const ::testing::TestParamInfo<bool>& info) {
                             return info.param ? "WithLse" : "RecomputedSoftmax";
                         });

// ---------------------------------------------------------------------------
// Properties: features defined in terms of other features
// ---------------------------------------------------------------------------

// GQA is MHA with each KV head repeated for its query-head group; dK/dV of a shared head are the
// sum over the group.
TEST(TestCpuFpReferenceSdpaProperty, GqaEqualsMhaWithRepeatedKv)
{
    constexpr int64_t H = 4;
    constexpr int64_t HKV = 2;
    constexpr int64_t GROUP = H / HKV;
    SdpaRun gqa({2, H, 3, 4}, {2, HKV, 5, 4}, {2, HKV, 5, 3});
    gqa.q.fillWithRandomValues(-1.0F, 1.0F, 51);
    gqa.k.fillWithRandomValues(-1.0F, 1.0F, 52);
    gqa.v.fillWithRandomValues(-1.0F, 1.0F, 53);
    gqa.dO.fillWithRandomValues(-1.0F, 1.0F, 54);
    gqa.run(0.7F);

    SdpaRun mha({2, H, 3, 4}, {2, H, 5, 4}, {2, H, 5, 3});
    forEachIndex(mha.q.dims(), [&](const std::vector<int64_t>& i) {
        mha.q.setHostValue(gqa.q.getHostValue(i), i);
    });
    forEachIndex(mha.dO.dims(), [&](const std::vector<int64_t>& i) {
        mha.dO.setHostValue(gqa.dO.getHostValue(i), i);
    });
    forEachIndex(mha.k.dims(), [&](const std::vector<int64_t>& i) {
        mha.k.setHostValue(gqa.k.getHostValue(i[0], i[1] / GROUP, i[2], i[3]), i);
    });
    forEachIndex(mha.v.dims(), [&](const std::vector<int64_t>& i) {
        mha.v.setHostValue(gqa.v.getHostValue(i[0], i[1] / GROUP, i[2], i[3]), i);
    });
    mha.run(0.7F);

    expectAllNear(gqa.o, mha.o, TOL, "O");
    expectAllNear(gqa.lse, mha.lse, TOL, "LSE");
    expectAllNear(gqa.dQ, mha.dQ, TOL, "dQ");
    forEachIndex(gqa.dK.dims(), [&](const std::vector<int64_t>& i) {
        float sum = 0.0F;
        for(int64_t g = 0; g < GROUP; ++g)
        {
            sum += mha.dK.getHostValue(i[0], i[1] * GROUP + g, i[2], i[3]);
        }
        EXPECT_NEAR(gqa.dK.getHostValue(i), sum, TOL);
    });
    forEachIndex(gqa.dV.dims(), [&](const std::vector<int64_t>& i) {
        float sum = 0.0F;
        for(int64_t g = 0; g < GROUP; ++g)
        {
            sum += mha.dV.getHostValue(i[0], i[1] * GROUP + g, i[2], i[3]);
        }
        EXPECT_NEAR(gqa.dV.getHostValue(i), sum, TOL);
    });
}

// A causal band equals an additive bias of -inf above the diagonal. Covers both alignments with
// Sq < Skv (decode / chunked prefill) and Sq > Skv.
struct CausalShape
{
    int64_t sq;
    int64_t skv;
    bool topLeft;
};

class TestCpuFpReferenceSdpaPropertyCausal : public ::testing::TestWithParam<CausalShape>
{
};

TEST_P(TestCpuFpReferenceSdpaPropertyCausal, BandEqualsNegInfBias)
{
    const auto p = GetParam();
    SdpaRun band({1, 2, p.sq, 4}, {1, 2, p.skv, 4}, {1, 2, p.skv, 4});
    band.q.fillWithRandomValues(-1.0F, 1.0F, 61);
    band.k.fillWithRandomValues(-1.0F, 1.0F, 62);
    band.v.fillWithRandomValues(-1.0F, 1.0F, 63);
    band.dO.fillWithRandomValues(-1.0F, 1.0F, 64);
    band.run(std::nullopt, nullptr, /*leftBound=*/-1, /*rightBound=*/0, p.topLeft);

    // Keep j <= i + off, off = 0 (TOP_LEFT) or Skv - Sq (BOTTOM_RIGHT).
    const int64_t off = p.topLeft ? 0 : p.skv - p.sq;
    Tensor<float> bias({p.sq, p.skv});
    for(int64_t i = 0; i < p.sq; ++i)
    {
        for(int64_t j = 0; j < p.skv; ++j)
        {
            bias.setHostValue(j <= i + off ? 0.0F : NEG_INF, i, j);
        }
    }
    SdpaRun biased({1, 2, p.sq, 4}, {1, 2, p.skv, 4}, {1, 2, p.skv, 4});
    forEachIndex(band.q.dims(), [&](const std::vector<int64_t>& i) {
        biased.q.setHostValue(band.q.getHostValue(i), i);
        biased.dO.setHostValue(band.dO.getHostValue(i), i);
    });
    forEachIndex(band.k.dims(), [&](const std::vector<int64_t>& i) {
        biased.k.setHostValue(band.k.getHostValue(i), i);
        biased.v.setHostValue(band.v.getHostValue(i), i);
    });
    biased.run(std::nullopt, &bias);

    expectAllNear(band.o, biased.o, TOL, "O");
    expectAllNear(band.dQ, biased.dQ, TOL, "dQ");
    expectAllNear(band.dK, biased.dK, TOL, "dK");
    expectAllNear(band.dV, biased.dV, TOL, "dV");
    // LSE agrees on live rows; dead rows are -inf in both.
    forEachIndex(band.lse.dims(), [&](const std::vector<int64_t>& i) {
        const float a = band.lse.getHostValue(i);
        const float b = biased.lse.getHostValue(i);
        if(std::isinf(a) || std::isinf(b))
        {
            EXPECT_EQ(a, b);
        }
        else
        {
            EXPECT_NEAR(a, b, TOL);
        }
    });
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuFpReferenceSdpaPropertyCausal,
                         ::testing::Values(CausalShape{4, 4, true},
                                           CausalShape{4, 4, false},
                                           CausalShape{3, 7, true},
                                           CausalShape{3, 7, false},
                                           CausalShape{1, 6, false},
                                           CausalShape{6, 3, true},
                                           CausalShape{6, 3, false}),
                         [](const ::testing::TestParamInfo<CausalShape>& info) {
                             return "Sq" + std::to_string(info.param.sq) + "Skv"
                                    + std::to_string(info.param.skv)
                                    + (info.param.topLeft ? "TopLeft" : "BottomRight");
                         });

// With Sq == Skv the two diagonal alignments are the same mask.
TEST(TestCpuFpReferenceSdpaProperty, AlignmentsAgreeWhenSqEqualsSkv)
{
    SdpaRun tl({1, 2, 5, 4}, {1, 2, 5, 4}, {1, 2, 5, 4});
    tl.q.fillWithRandomValues(-1.0F, 1.0F, 71);
    tl.k.fillWithRandomValues(-1.0F, 1.0F, 72);
    tl.v.fillWithRandomValues(-1.0F, 1.0F, 73);
    tl.dO.fillWithRandomValues(-1.0F, 1.0F, 74);
    tl.run(std::nullopt, nullptr, -1, 0, /*topLeft=*/true);

    SdpaRun br({1, 2, 5, 4}, {1, 2, 5, 4}, {1, 2, 5, 4});
    forEachIndex(tl.q.dims(), [&](const std::vector<int64_t>& i) {
        br.q.setHostValue(tl.q.getHostValue(i), i);
        br.k.setHostValue(tl.k.getHostValue(i), i);
        br.v.setHostValue(tl.v.getHostValue(i), i);
        br.dO.setHostValue(tl.dO.getHostValue(i), i);
    });
    br.run(std::nullopt, nullptr, -1, 0, /*topLeft=*/false);

    expectAllNear(tl.o, br.o, 0.0F, "O");
    expectAllNear(tl.lse, br.lse, 0.0F, "LSE");
    expectAllNear(tl.dQ, br.dQ, 0.0F, "dQ");
    expectAllNear(tl.dK, br.dK, 0.0F, "dK");
    expectAllNear(tl.dV, br.dV, 0.0F, "dV");
}

// The attention scale is a factor on Q K^T, so scale s on Q equals scale 1 on s*Q. The chain rule
// gives dQ = s * dQ', and dK, dV are unchanged.
TEST(TestCpuFpReferenceSdpaProperty, ScaleEqualsPrescaledQuery)
{
    constexpr float SCALE = 0.35F;
    SdpaRun scaled({1, 2, 3, 4}, {1, 2, 5, 4}, {1, 2, 5, 3});
    scaled.q.fillWithRandomValues(-1.0F, 1.0F, 81);
    scaled.k.fillWithRandomValues(-1.0F, 1.0F, 82);
    scaled.v.fillWithRandomValues(-1.0F, 1.0F, 83);
    scaled.dO.fillWithRandomValues(-1.0F, 1.0F, 84);
    scaled.run(SCALE);

    SdpaRun pre({1, 2, 3, 4}, {1, 2, 5, 4}, {1, 2, 5, 3});
    forEachIndex(pre.q.dims(), [&](const std::vector<int64_t>& i) {
        pre.q.setHostValue(SCALE * scaled.q.getHostValue(i), i);
    });
    forEachIndex(pre.dO.dims(), [&](const std::vector<int64_t>& i) {
        pre.dO.setHostValue(scaled.dO.getHostValue(i), i);
    });
    forEachIndex(pre.k.dims(), [&](const std::vector<int64_t>& i) {
        pre.k.setHostValue(scaled.k.getHostValue(i), i);
    });
    forEachIndex(pre.v.dims(), [&](const std::vector<int64_t>& i) {
        pre.v.setHostValue(scaled.v.getHostValue(i), i);
    });
    pre.run();

    expectAllNear(scaled.o, pre.o, TOL, "O");
    expectAllNear(scaled.lse, pre.lse, TOL, "LSE");
    forEachIndex(scaled.dQ.dims(), [&](const std::vector<int64_t>& i) {
        EXPECT_NEAR(scaled.dQ.getHostValue(i), SCALE * pre.dQ.getHostValue(i), TOL);
    });
    expectAllNear(scaled.dK, pre.dK, TOL, "dK");
    expectAllNear(scaled.dV, pre.dV, TOL, "dV");
}

// ---------------------------------------------------------------------------
// Padding (seq_len_q / seq_len_kv)
// ---------------------------------------------------------------------------

// A padded batch equals each batch run on its own truncated tensors: O, LSE, dQ, dK and dV agree
// on live positions; padded query rows give O = 0, LSE = -inf, dQ = 0; padded keys give dK = dV = 0.
// Every padding slot of Q, K, V and dO holds NaN, so any read of padding fails the test.
// Bottom-right alignment uses each batch's own lengths (cuDNN).
struct PaddingCase
{
    const char* name;
    int64_t rightBound; // -1 = no causal, 0 = causal
    bool topLeft;
};

class TestCpuFpReferenceSdpaPropertyPadding : public ::testing::TestWithParam<PaddingCase>
{
};

TEST_P(TestCpuFpReferenceSdpaPropertyPadding, EqualsTruncatedBatches)
{
    const auto p = GetParam();
    constexpr int64_t BATCH = 3;
    constexpr int64_t H = 2;
    constexpr int64_t SQ = 5;
    constexpr int64_t SKV = 6;
    constexpr int64_t D = 4;
    constexpr float SCALE = 0.8F;
    const std::vector<int32_t> lensQ = {3, 5, 2};
    const std::vector<int32_t> lensKv = {6, 2, 4};
    const float nan = std::numeric_limits<float>::quiet_NaN();

    SdpaRun padded({BATCH, H, SQ, D}, {BATCH, H, SKV, D}, {BATCH, H, SKV, D});
    padded.q.fillWithRandomValues(-1.0F, 1.0F, 101);
    padded.k.fillWithRandomValues(-1.0F, 1.0F, 102);
    padded.v.fillWithRandomValues(-1.0F, 1.0F, 103);
    padded.dO.fillWithRandomValues(-1.0F, 1.0F, 104);
    forEachIndex(padded.q.dims(), [&](const std::vector<int64_t>& i) {
        if(i[2] >= lensQ[static_cast<size_t>(i[0])])
        {
            padded.q.setHostValue(nan, i);
            padded.dO.setHostValue(nan, i);
        }
    });
    forEachIndex(padded.k.dims(), [&](const std::vector<int64_t>& i) {
        if(i[2] >= lensKv[static_cast<size_t>(i[0])])
        {
            padded.k.setHostValue(nan, i);
            padded.v.setHostValue(nan, i);
        }
    });
    Tensor<int32_t> seqLenQ({BATCH, 1, 1, 1});
    Tensor<int32_t> seqLenKv({BATCH, 1, 1, 1});
    for(int64_t b = 0; b < BATCH; ++b)
    {
        seqLenQ.setHostValue(lensQ[static_cast<size_t>(b)], b, 0, 0, 0);
        seqLenKv.setHostValue(lensKv[static_cast<size_t>(b)], b, 0, 0, 0);
    }

    CpuFpReferenceSdpa::forward(padded.q,
                                padded.k,
                                padded.v,
                                padded.o,
                                SCALE,
                                nullptr,
                                -1,
                                p.rightBound,
                                p.topLeft,
                                &padded.lse,
                                nullptr,
                                nullptr,
                                nullptr,
                                &seqLenQ,
                                &seqLenKv);
    CpuFpReferenceSdpa::backward(padded.q,
                                 padded.k,
                                 padded.v,
                                 padded.o,
                                 padded.dO,
                                 padded.dQ,
                                 padded.dK,
                                 padded.dV,
                                 SCALE,
                                 &padded.lse,
                                 nullptr,
                                 -1,
                                 p.rightBound,
                                 p.topLeft,
                                 nullptr,
                                 &seqLenQ,
                                 &seqLenKv);

    for(int64_t b = 0; b < BATCH; ++b)
    {
        const int64_t lq = lensQ[static_cast<size_t>(b)];
        const int64_t lkv = lensKv[static_cast<size_t>(b)];
        SdpaRun live({1, H, lq, D}, {1, H, lkv, D}, {1, H, lkv, D});
        forEachIndex(live.q.dims(), [&](const std::vector<int64_t>& i) {
            live.q.setHostValue(padded.q.getHostValue(b, i[1], i[2], i[3]), i);
            live.dO.setHostValue(padded.dO.getHostValue(b, i[1], i[2], i[3]), i);
        });
        forEachIndex(live.k.dims(), [&](const std::vector<int64_t>& i) {
            live.k.setHostValue(padded.k.getHostValue(b, i[1], i[2], i[3]), i);
            live.v.setHostValue(padded.v.getHostValue(b, i[1], i[2], i[3]), i);
        });
        live.run(SCALE, nullptr, -1, p.rightBound, p.topLeft);

        for(int64_t h = 0; h < H; ++h)
        {
            for(int64_t sq = 0; sq < SQ; ++sq)
            {
                const float lse = padded.lse.getHostValue(b, h, sq, 0);
                const float lseLive = sq < lq ? live.lse.getHostValue(0, h, sq, 0)
                                              : -std::numeric_limits<float>::infinity();
                if(std::isinf(lseLive))
                {
                    EXPECT_EQ(lse, lseLive) << "b=" << b << " h=" << h << " sq=" << sq;
                }
                else
                {
                    EXPECT_NEAR(lse, lseLive, TOL) << "b=" << b << " h=" << h << " sq=" << sq;
                }
                for(int64_t d = 0; d < D; ++d)
                {
                    const float oLive = sq < lq ? live.o.getHostValue(0, h, sq, d) : 0.0F;
                    const float dqLive = sq < lq ? live.dQ.getHostValue(0, h, sq, d) : 0.0F;
                    EXPECT_NEAR(padded.o.getHostValue(b, h, sq, d), oLive, TOL)
                        << "O b=" << b << " h=" << h << " sq=" << sq;
                    EXPECT_NEAR(padded.dQ.getHostValue(b, h, sq, d), dqLive, TOL)
                        << "dQ b=" << b << " h=" << h << " sq=" << sq;
                }
            }
            for(int64_t skv = 0; skv < SKV; ++skv)
            {
                for(int64_t d = 0; d < D; ++d)
                {
                    const float dkLive = skv < lkv ? live.dK.getHostValue(0, h, skv, d) : 0.0F;
                    const float dvLive = skv < lkv ? live.dV.getHostValue(0, h, skv, d) : 0.0F;
                    EXPECT_NEAR(padded.dK.getHostValue(b, h, skv, d), dkLive, TOL)
                        << "dK b=" << b << " h=" << h << " skv=" << skv;
                    EXPECT_NEAR(padded.dV.getHostValue(b, h, skv, d), dvLive, TOL)
                        << "dV b=" << b << " h=" << h << " skv=" << skv;
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuFpReferenceSdpaPropertyPadding,
                         ::testing::Values(PaddingCase{"NoMask", -1, true},
                                           PaddingCase{"CausalTopLeft", 0, true},
                                           PaddingCase{"CausalBottomRight", 0, false}),
                         [](const ::testing::TestParamInfo<PaddingCase>& info) {
                             return std::string(info.param.name);
                         });

// A batch with no keys (seq_len_kv = 0) has no live row: O = 0, LSE = -inf, all gradients 0.
TEST(TestCpuFpReferenceSdpaClosedForm, EmptyKeySequence)
{
    SdpaRun r({1, 1, 2, 4}, {1, 1, 3, 4}, {1, 1, 3, 4});
    r.q.fillWithRandomValues(-1.0F, 1.0F, 111);
    r.k.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    r.v.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    r.dO.fillWithRandomValues(-1.0F, 1.0F, 112);
    Tensor<int32_t> seqLenQ({1, 1, 1, 1});
    Tensor<int32_t> seqLenKv({1, 1, 1, 1});
    seqLenQ.fillWithValue(2);
    seqLenKv.fillWithValue(0);

    CpuFpReferenceSdpa::forward(r.q,
                                r.k,
                                r.v,
                                r.o,
                                std::nullopt,
                                nullptr,
                                -1,
                                -1,
                                true,
                                &r.lse,
                                nullptr,
                                nullptr,
                                nullptr,
                                &seqLenQ,
                                &seqLenKv);
    CpuFpReferenceSdpa::backward(r.q,
                                 r.k,
                                 r.v,
                                 r.o,
                                 r.dO,
                                 r.dQ,
                                 r.dK,
                                 r.dV,
                                 std::nullopt,
                                 &r.lse,
                                 nullptr,
                                 -1,
                                 -1,
                                 true,
                                 nullptr,
                                 &seqLenQ,
                                 &seqLenKv);

    for(int64_t sq = 0; sq < 2; ++sq)
    {
        EXPECT_TRUE(std::isinf(r.lse.getHostValue(0, 0, sq, 0))
                    && r.lse.getHostValue(0, 0, sq, 0) < 0);
        for(int64_t d = 0; d < 4; ++d)
        {
            EXPECT_EQ(r.o.getHostValue(0, 0, sq, d), 0.0F);
            EXPECT_EQ(r.dQ.getHostValue(0, 0, sq, d), 0.0F);
        }
    }
    expectAllFinite(r.dK, "dK");
    expectAllFinite(r.dV, "dV");
}

// The two seq_len tensors go together, and each value must fit the padded length.
TEST(TestCpuFpReferenceSdpaClosedForm, SeqLenValidation)
{
    SdpaRun r({1, 1, 2, 4}, {1, 1, 3, 4}, {1, 1, 3, 4});
    r.q.fillWithValue(0.0F);
    r.k.fillWithValue(0.0F);
    r.v.fillWithValue(0.0F);
    Tensor<int32_t> lenOk({1, 1, 1, 1});
    lenOk.fillWithValue(1);
    Tensor<int32_t> lenTooLong({1, 1, 1, 1});
    lenTooLong.fillWithValue(4);
    const TensorBase<int32_t>* none = nullptr;
    const auto fwd = [&](const TensorBase<int32_t>* lq, const TensorBase<int32_t>* lkv) {
        CpuFpReferenceSdpa::forward(r.q,
                                    r.k,
                                    r.v,
                                    r.o,
                                    std::nullopt,
                                    nullptr,
                                    -1,
                                    -1,
                                    true,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    lq,
                                    lkv);
    };
    EXPECT_THROW(fwd(&lenOk, none), std::invalid_argument);
    EXPECT_THROW(fwd(none, &lenOk), std::invalid_argument);
    EXPECT_THROW(fwd(&lenOk, &lenTooLong), std::invalid_argument);
    EXPECT_NO_THROW(fwd(&lenOk, &lenOk));
}

// ---------------------------------------------------------------------------
// Paged KV cache
// ---------------------------------------------------------------------------

// A paged cache holding the same keys as a dense padded K/V gives the same output. The cache
// uses vLLM's NHD layout ([blocks, blockSize, H, D] in memory, so container strides
// {bs*H*D, D, H*D, 1}), the blocks are shuffled, unused page-table entries point at block 0 (the
// "null block"), and block 0 plus every unused slot holds NaN: none of it may be read.
class TestCpuFpReferenceSdpaPropertyPaged : public ::testing::TestWithParam<int64_t>
{
};

TEST_P(TestCpuFpReferenceSdpaPropertyPaged, EqualsDensePadded)
{
    const int64_t blockSize = GetParam();
    constexpr int64_t BATCH = 3;
    constexpr int64_t H = 2;
    constexpr int64_t SQ = 4;
    constexpr int64_t MAX_KV = 10;
    constexpr int64_t D = 8;
    const std::vector<int32_t> lensQ = {4, 1, 4};
    const std::vector<int32_t> lensKv = {10, 3, 7};
    const float nan = std::numeric_limits<float>::quiet_NaN();

    SdpaRun dense({BATCH, H, SQ, D}, {BATCH, H, MAX_KV, D}, {BATCH, H, MAX_KV, D});
    dense.q.fillWithRandomValues(-1.0F, 1.0F, 121);
    dense.k.fillWithRandomValues(-1.0F, 1.0F, 122);
    dense.v.fillWithRandomValues(-1.0F, 1.0F, 123);
    Tensor<int32_t> seqLenQ({BATCH, 1, 1, 1});
    Tensor<int32_t> seqLenKv({BATCH, 1, 1, 1});
    for(int64_t b = 0; b < BATCH; ++b)
    {
        seqLenQ.setHostValue(lensQ[static_cast<size_t>(b)], b, 0, 0, 0);
        seqLenKv.setHostValue(lensKv[static_cast<size_t>(b)], b, 0, 0, 0);
    }

    // Page tables: block ids handed out in a shuffled order starting at 1 (0 is the null block).
    const int64_t tableSize = (MAX_KV + blockSize - 1) / blockSize;
    const int64_t numBlocks = 1 + BATCH * tableSize;
    std::vector<int32_t> ids;
    for(int64_t i = 1; i < numBlocks; ++i)
    {
        ids.push_back(static_cast<int32_t>((i * 7) % (numBlocks - 1) + 1));
    }
    Tensor<int32_t> pageTable({BATCH, 1, tableSize, 1});
    const std::vector<int64_t> cDims = {numBlocks, H, blockSize, D};
    const std::vector<int64_t> nhdStrides = {blockSize * H * D, D, H * D, 1};
    Tensor<float> kCache(cDims, nhdStrides);
    Tensor<float> vCache(cDims, nhdStrides);
    kCache.fillWithValue(nan);
    vCache.fillWithValue(nan);
    size_t next = 0;
    for(int64_t b = 0; b < BATCH; ++b)
    {
        const int64_t usedBlocks = (lensKv[static_cast<size_t>(b)] + blockSize - 1) / blockSize;
        for(int64_t t = 0; t < tableSize; ++t)
        {
            pageTable.setHostValue(t < usedBlocks ? ids[next++] : 0, b, 0, t, 0);
        }
        for(int64_t j = 0; j < lensKv[static_cast<size_t>(b)]; ++j)
        {
            const int64_t block = pageTable.getHostValue(b, 0, j / blockSize, 0);
            for(int64_t h = 0; h < H; ++h)
            {
                for(int64_t d = 0; d < D; ++d)
                {
                    kCache.setHostValue(
                        dense.k.getHostValue(b, h, j, d), block, h, j % blockSize, d);
                    vCache.setHostValue(
                        dense.v.getHostValue(b, h, j, d), block, h, j % blockSize, d);
                }
            }
        }
    }

    auto kPaged = CpuFpReferenceSdpa::gatherPagedKv(kCache, pageTable, seqLenKv, MAX_KV);
    auto vPaged = CpuFpReferenceSdpa::gatherPagedKv(vCache, pageTable, seqLenKv, MAX_KV);
    Tensor<float> oPaged({BATCH, H, SQ, D});
    Tensor<float> lsePaged({BATCH, H, SQ, 1});
    for(const bool paged : {false, true})
    {
        CpuFpReferenceSdpa::forward(dense.q,
                                    paged ? static_cast<const TensorBase<float>&>(kPaged) : dense.k,
                                    paged ? static_cast<const TensorBase<float>&>(vPaged) : dense.v,
                                    paged ? oPaged : dense.o,
                                    0.5F,
                                    nullptr,
                                    -1,
                                    0,
                                    false,
                                    paged ? &lsePaged : &dense.lse,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    &seqLenQ,
                                    &seqLenKv);
    }
    expectAllNear(oPaged, dense.o, 0.0F, "O");
    forEachIndex(lsePaged.dims(), [&](const std::vector<int64_t>& i) {
        EXPECT_EQ(lsePaged.getHostValue(i), dense.lse.getHostValue(i));
    });
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuFpReferenceSdpaPropertyPaged,
                         ::testing::Values(1, 3, 4),
                         [](const ::testing::TestParamInfo<int64_t>& info) {
                             return "BlockSize" + std::to_string(info.param);
                         });

// Page-table entries outside the container, and tables of the wrong shape, are rejected.
TEST(TestCpuFpReferenceSdpaClosedForm, PagedGatherValidation)
{
    Tensor<float> cache({2, 1, 4, 2});
    cache.fillWithValue(1.0F);
    Tensor<int32_t> seqLenKv({1, 1, 1, 1});
    seqLenKv.fillWithValue(5);
    Tensor<int32_t> table({1, 1, 2, 1});
    table.setHostValue(1, 0, 0, 0, 0);
    table.setHostValue(2, 0, 0, 1, 0); // block 2 does not exist
    EXPECT_THROW(CpuFpReferenceSdpa::gatherPagedKv(cache, table, seqLenKv, 8),
                 std::invalid_argument);
    table.setHostValue(0, 0, 0, 1, 0);
    EXPECT_NO_THROW(CpuFpReferenceSdpa::gatherPagedKv(cache, table, seqLenKv, 8));
    // maxSeqKv beyond tableSize * blockSize.
    EXPECT_THROW(CpuFpReferenceSdpa::gatherPagedKv(cache, table, seqLenKv, 9),
                 std::invalid_argument);
    const Tensor<int32_t> flatTable({1, 2});
    EXPECT_THROW(CpuFpReferenceSdpa::gatherPagedKv(cache, flatTable, seqLenKv, 8),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Attention sink (cuDNN sink_token)
// ---------------------------------------------------------------------------

namespace
{
void runWithSink(SdpaRun& r, const Tensor<float>& sink, const TensorBase<float>* bias = nullptr)
{
    CpuFpReferenceSdpa::forward(r.q,
                                r.k,
                                r.v,
                                r.o,
                                0.5F,
                                bias,
                                -1,
                                -1,
                                true,
                                &r.lse,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                &sink);
}
} // namespace

// Q = 0 gives n equal zero scores; with sink s the row's LSE is ln(n + e^s) and O is the mean of V
// scaled by n / (n + e^s).
TEST(TestCpuFpReferenceSdpaClosedForm, SinkWithUniformScores)
{
    constexpr int64_t SKV = 6;
    SdpaRun r({1, 2, 2, 4}, {1, 2, SKV, 4}, {1, 2, SKV, 3});
    r.q.fillWithValue(0.0F);
    r.k.fillWithRandomValues(-1.0F, 1.0F, 131);
    r.v.fillWithRandomValues(-1.0F, 1.0F, 132);
    Tensor<float> sink({1, 2, 1, 1});
    sink.setHostValue(0.7F, 0, 0, 0, 0);
    sink.setHostValue(-1.3F, 0, 1, 0, 0);
    runWithSink(r, sink);

    for(int64_t h = 0; h < 2; ++h)
    {
        const float s = sink.getHostValue(0, h, 0, 0);
        const float denom = static_cast<float>(SKV) + std::exp(s);
        for(int64_t sq = 0; sq < 2; ++sq)
        {
            EXPECT_NEAR(r.lse.getHostValue(0, h, sq, 0), std::log(denom), TOL);
            for(int64_t dv = 0; dv < 3; ++dv)
            {
                float sum = 0.0F;
                for(int64_t j = 0; j < SKV; ++j)
                {
                    sum += r.v.getHostValue(0, h, j, dv);
                }
                EXPECT_NEAR(r.o.getHostValue(0, h, sq, dv), sum / denom, TOL);
            }
        }
    }
}

// A row with no live key but a finite sink: O = 0 and LSE = the sink logit (cuDNN), not -inf.
TEST(TestCpuFpReferenceSdpaClosedForm, SinkOnFullyMaskedRow)
{
    SdpaRun r({1, 1, 2, 4}, {1, 1, 3, 4}, {1, 1, 3, 4});
    r.q.fillWithRandomValues(-1.0F, 1.0F, 141);
    r.k.fillWithRandomValues(-1.0F, 1.0F, 142);
    r.v.fillWithRandomValues(-1.0F, 1.0F, 143);
    Tensor<float> bias({2, 3});
    bias.fillWithValue(0.0F);
    for(int64_t j = 0; j < 3; ++j)
    {
        bias.setHostValue(NEG_INF, 0, j);
    }
    Tensor<float> sink({1, 1, 1, 1});
    sink.fillWithValue(0.4F);
    runWithSink(r, sink, &bias);

    EXPECT_FLOAT_EQ(r.lse.getHostValue(0, 0, 0, 0), 0.4F);
    for(int64_t d = 0; d < 4; ++d)
    {
        EXPECT_EQ(r.o.getHostValue(0, 0, 0, d), 0.0F);
    }
    EXPECT_TRUE(std::isfinite(r.lse.getHostValue(0, 0, 1, 0)));
}

// A sink is exactly one more key whose score is the sink logit and whose value row is zero:
// append a zero key (score 0) with bias = sink and a zero V row.
TEST(TestCpuFpReferenceSdpaProperty, SinkEqualsExtraZeroValueKey)
{
    constexpr int64_t H = 2;
    constexpr int64_t SQ = 3;
    constexpr int64_t SKV = 5;
    constexpr int64_t D = 4;
    SdpaRun withSink({2, H, SQ, D}, {2, H, SKV, D}, {2, H, SKV, D});
    withSink.q.fillWithRandomValues(-1.0F, 1.0F, 151);
    withSink.k.fillWithRandomValues(-1.0F, 1.0F, 152);
    withSink.v.fillWithRandomValues(-1.0F, 1.0F, 153);
    Tensor<float> sink({1, H, 1, 1});
    sink.setHostValue(0.9F, 0, 0, 0, 0);
    sink.setHostValue(-0.2F, 0, 1, 0, 0);
    runWithSink(withSink, sink);

    SdpaRun extra({2, H, SQ, D}, {2, H, SKV + 1, D}, {2, H, SKV + 1, D});
    Tensor<float> bias({1, H, SQ, SKV + 1});
    bias.fillWithValue(0.0F);
    forEachIndex(extra.q.dims(), [&](const std::vector<int64_t>& i) {
        extra.q.setHostValue(withSink.q.getHostValue(i), i);
    });
    forEachIndex(extra.k.dims(), [&](const std::vector<int64_t>& i) {
        const bool sinkKey = i[2] == SKV;
        extra.k.setHostValue(sinkKey ? 0.0F : withSink.k.getHostValue(i), i);
        extra.v.setHostValue(sinkKey ? 0.0F : withSink.v.getHostValue(i), i);
    });
    for(int64_t h = 0; h < H; ++h)
    {
        for(int64_t sq = 0; sq < SQ; ++sq)
        {
            bias.setHostValue(sink.getHostValue(0, h, 0, 0), 0, h, sq, SKV);
        }
    }
    extra.run(0.5F, &bias);

    expectAllNear(withSink.o, extra.o, TOL, "O");
    expectAllNear(withSink.lse, extra.lse, TOL, "LSE");
}
