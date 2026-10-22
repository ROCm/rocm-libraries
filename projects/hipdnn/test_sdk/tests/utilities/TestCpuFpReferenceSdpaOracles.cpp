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
