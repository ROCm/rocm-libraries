// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;

// ---------------------------------------------------------------------------
// TypePair helper for TYPED_TEST
// ---------------------------------------------------------------------------

template <typename T1, typename T2>
struct TypePair
{
    using First = T1;
    using Second = T2;
};

// ---------------------------------------------------------------------------
// Fp64 precision unit tests
// ---------------------------------------------------------------------------

TEST(TestCpuFpReferenceSdpaFp64, SanityCheck)
{
    // [B=1, H=1, Sq=2, Skv=2, D=2, Dv=2]
    // Q[0,0,0,:] = [1,0], Q[0,0,1,:] = [0,1]
    // K[0,0,0,:] = [1,0], K[0,0,1,:] = [0,1]
    // V[0,0,0,:] = [1,2], V[0,0,1,:] = [3,4]
    // Default scale = 1/sqrt(2)

    Tensor<double> q({1, 1, 2, 2});
    Tensor<double> k({1, 1, 2, 2});
    Tensor<double> v({1, 1, 2, 2});
    Tensor<double> o({1, 1, 2, 2});

    q.fillWithValue(0.0);
    k.fillWithValue(0.0);

    q.setHostValue(1.0, 0, 0, 0, 0);
    q.setHostValue(1.0, 0, 0, 1, 1);

    k.setHostValue(1.0, 0, 0, 0, 0);
    k.setHostValue(1.0, 0, 0, 1, 1);

    v.setHostValue(1.0, 0, 0, 0, 0);
    v.setHostValue(2.0, 0, 0, 0, 1);
    v.setHostValue(3.0, 0, 0, 1, 0);
    v.setHostValue(4.0, 0, 0, 1, 1);

    CpuFpReferenceSdpa::forward(q, k, v, o);

    // Expected using default scale = 1/sqrt(2):
    // sq=0: S[0]=scale, S[1]=0 → P[0]=1/(1+exp(-scale)), P[1]=exp(-scale)/(1+exp(-scale))
    // sq=1: S[0]=0, S[1]=scale → P[0]=exp(-scale)/(1+exp(-scale)), P[1]=1/(1+exp(-scale))
    const float scale = 1.0f / std::sqrt(2.0f);
    const float eLow = std::exp(-scale);
    const float sumExp = 1.0f + eLow;
    const float pHigh = 1.0f / sumExp; // weight for the matching kv token
    const float pLow = eLow / sumExp; // weight for the non-matching kv token

    const double tol = 1e-5;

    // sq=0 biased toward kv=0 (V=[1,2])
    EXPECT_NEAR(o.getHostValue(0, 0, 0, 0), static_cast<double>(pHigh * 1.0f + pLow * 3.0f), tol);
    EXPECT_NEAR(o.getHostValue(0, 0, 0, 1), static_cast<double>(pHigh * 2.0f + pLow * 4.0f), tol);

    // sq=1 biased toward kv=1 (V=[3,4])
    EXPECT_NEAR(o.getHostValue(0, 0, 1, 0), static_cast<double>(pLow * 1.0f + pHigh * 3.0f), tol);
    EXPECT_NEAR(o.getHostValue(0, 0, 1, 1), static_cast<double>(pLow * 2.0f + pHigh * 4.0f), tol);
}

TEST(TestCpuFpReferenceSdpaFp64, DefaultScaleIs1OverSqrtD)
{
    // Verify that the default attention scale equals 1/sqrt(headDim).
    // Q[0,0,0,0]=1, rest zero; K[0,0,0,0]=1 (dot=1), K[0,0,1,:]=0 (dot=0).
    const int64_t headDim = 4;

    Tensor<double> q({1, 1, 1, headDim});
    Tensor<double> k({1, 1, 2, headDim});
    Tensor<double> v({1, 1, 2, 1});
    Tensor<double> o({1, 1, 1, 1});

    q.fillWithValue(0.0);
    k.fillWithValue(0.0);
    v.fillWithValue(0.0);

    q.setHostValue(1.0, 0, 0, 0, 0);
    k.setHostValue(1.0, 0, 0, 0, 0);

    v.setHostValue(1.0, 0, 0, 0, 0);
    v.setHostValue(2.0, 0, 0, 1, 0);

    CpuFpReferenceSdpa::forward(q, k, v, o);

    // Default scale = 1/sqrt(headDim)
    // S[0] = 1 * scale, S[1] = 0
    // P[0] = 1/(1+exp(-scale)), P[1] = exp(-scale)/(1+exp(-scale))
    const float defaultScale = 1.0f / std::sqrt(static_cast<float>(headDim));
    const float e1 = std::exp(-defaultScale);
    const float sumE = 1.0f + e1;
    const float p0 = 1.0f / sumE;
    const float p1 = e1 / sumE;
    const auto expected = static_cast<double>(p0 * 1.0f + p1 * 2.0f);

    EXPECT_NEAR(o.getHostValue(0, 0, 0, 0), expected, 1e-5);
}

TEST(TestCpuFpReferenceSdpaFp64, CustomScale)
{
    // Verify that an explicit attnScaleValue overrides the default 1/sqrt(D).
    const int64_t headDim = 4;

    Tensor<double> q({1, 1, 1, headDim});
    Tensor<double> k({1, 1, 2, headDim});
    Tensor<double> v({1, 1, 2, 1});
    Tensor<double> oDefault({1, 1, 1, 1});
    Tensor<double> oCustom({1, 1, 1, 1});

    q.fillWithValue(0.0);
    k.fillWithValue(0.0);
    v.fillWithValue(0.0);

    q.setHostValue(1.0, 0, 0, 0, 0);
    k.setHostValue(1.0, 0, 0, 0, 0);
    v.setHostValue(1.0, 0, 0, 0, 0);
    v.setHostValue(2.0, 0, 0, 1, 0);

    CpuFpReferenceSdpa::forward(q, k, v, oDefault); // default scale = 0.5
    CpuFpReferenceSdpa::forward(q, k, v, oCustom, std::optional<float>{2.0f}); // custom scale = 2.0

    // Expected with custom scale = 2.0: S[0]=2, S[1]=0
    const float customScale = 2.0f;
    const float e1Custom = std::exp(-customScale);
    const float sumCustom = 1.0f + e1Custom;
    const float p0Custom = 1.0f / sumCustom;
    const float p1Custom = e1Custom / sumCustom;
    const auto expectedCustom = static_cast<double>(p0Custom * 1.0f + p1Custom * 2.0f);

    EXPECT_NEAR(oCustom.getHostValue(0, 0, 0, 0), expectedCustom, 1e-5);

    // Custom scale result must differ from default scale result
    EXPECT_NE(oDefault.getHostValue(0, 0, 0, 0), oCustom.getHostValue(0, 0, 0, 0));
}

TEST(TestCpuFpReferenceSdpaFp64, WithAttnMask)
{
    // Same Q/K/V as SanityCheck, but with an additive mask that suppresses kv=1 for sq=0.
    Tensor<double> q({1, 1, 2, 2});
    Tensor<double> k({1, 1, 2, 2});
    Tensor<double> v({1, 1, 2, 2});
    Tensor<double> o({1, 1, 2, 2});
    Tensor<float> mask({2, 2}); // [Sq, Skv]

    q.fillWithValue(0.0);
    k.fillWithValue(0.0);

    q.setHostValue(1.0, 0, 0, 0, 0);
    q.setHostValue(1.0, 0, 0, 1, 1);

    k.setHostValue(1.0, 0, 0, 0, 0);
    k.setHostValue(1.0, 0, 0, 1, 1);

    v.setHostValue(1.0, 0, 0, 0, 0);
    v.setHostValue(2.0, 0, 0, 0, 1);
    v.setHostValue(3.0, 0, 0, 1, 0);
    v.setHostValue(4.0, 0, 0, 1, 1);

    // For sq=0: block kv=1 with a large negative value
    mask.setHostValue(0.0f, 0, 0);
    mask.setHostValue(-1e4f, 0, 1);
    // For sq=1: no masking
    mask.setHostValue(0.0f, 1, 0);
    mask.setHostValue(0.0f, 1, 1);

    CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, &mask);

    // sq=0: kv=1 is masked → O ≈ V[0,0,0,:] = [1, 2]
    EXPECT_NEAR(o.getHostValue(0, 0, 0, 0), 1.0, 1e-3);
    EXPECT_NEAR(o.getHostValue(0, 0, 0, 1), 2.0, 1e-3);

    // sq=1: unmasked, biased toward kv=1 (higher score) → O closer to V[0,0,1,:]=[3,4]
    EXPECT_GT(o.getHostValue(0, 0, 1, 0), 2.0);
    EXPECT_GT(o.getHostValue(0, 0, 1, 1), 3.0);
}

TEST(TestCpuFpReferenceSdpaFp64, GqaSupport)
{
    // H=4, Hkv=2: each KV head serves 2 Q heads.
    // Skv=1: softmax is trivially 1.0, so O = V[b, kvHead, 0, :].
    Tensor<double> q({1, 4, 1, 2});
    Tensor<double> k({1, 2, 1, 2});
    Tensor<double> v({1, 2, 1, 2});
    Tensor<double> o({1, 4, 1, 2});

    q.fillWithValue(0.0);
    k.fillWithValue(0.0);

    v.setHostValue(1.0, 0, 0, 0, 0);
    v.setHostValue(2.0, 0, 0, 0, 1); // V[kvHead=0] = [1, 2]
    v.setHostValue(3.0, 0, 1, 0, 0);
    v.setHostValue(4.0, 0, 1, 0, 1); // V[kvHead=1] = [3, 4]

    CpuFpReferenceSdpa::forward(q, k, v, o);

    const double tol = 1e-5;

    // h=0,1: kvHead = h/2 = 0 → O = V[0,0,0,:] = [1, 2]
    for(int64_t h = 0; h < 2; ++h)
    {
        EXPECT_NEAR(o.getHostValue(0, h, 0, 0), 1.0, tol);
        EXPECT_NEAR(o.getHostValue(0, h, 0, 1), 2.0, tol);
    }

    // h=2,3: kvHead = h/2 = 1 → O = V[0,1,0,:] = [3, 4]
    for(int64_t h = 2; h < 4; ++h)
    {
        EXPECT_NEAR(o.getHostValue(0, h, 0, 0), 3.0, tol);
        EXPECT_NEAR(o.getHostValue(0, h, 0, 1), 4.0, tol);
    }
}

TEST(TestCpuFpReferenceSdpaFp64, MqaSupport)
{
    // H=4, Hkv=1: single KV head for all Q heads.
    // Skv=1: O = V[0, 0, 0, :] for all Q heads.
    Tensor<double> q({1, 4, 1, 2});
    Tensor<double> k({1, 1, 1, 2});
    Tensor<double> v({1, 1, 1, 2});
    Tensor<double> o({1, 4, 1, 2});

    q.fillWithValue(0.0);
    k.fillWithValue(0.0);

    v.setHostValue(5.0, 0, 0, 0, 0);
    v.setHostValue(6.0, 0, 0, 0, 1); // V[kvHead=0] = [5, 6]

    CpuFpReferenceSdpa::forward(q, k, v, o);

    const double tol = 1e-5;

    for(int64_t h = 0; h < 4; ++h)
    {
        EXPECT_NEAR(o.getHostValue(0, h, 0, 0), 5.0, tol);
        EXPECT_NEAR(o.getHostValue(0, h, 0, 1), 6.0, tol);
    }
}

TEST(TestCpuFpReferenceSdpaFp64, AttnMaskBroadcastRank2)
{
    // Rank-2 mask [1, Skv]: dim[0]=1 broadcasts over all sq (and also over batch and head
    // which are absent from the rank-2 mask).
    Tensor<double> q({1, 1, 2, 2});
    Tensor<double> k({1, 1, 2, 2});
    Tensor<double> v({1, 1, 2, 2});
    Tensor<double> o({1, 1, 2, 2});
    Tensor<float> mask({1, 2}); // [1, Skv]

    q.fillWithValue(0.0);
    k.fillWithValue(0.0);

    q.setHostValue(1.0, 0, 0, 0, 0);
    q.setHostValue(1.0, 0, 0, 1, 1);

    k.setHostValue(1.0, 0, 0, 0, 0);
    k.setHostValue(1.0, 0, 0, 1, 1);

    v.setHostValue(1.0, 0, 0, 0, 0);
    v.setHostValue(2.0, 0, 0, 0, 1);
    v.setHostValue(3.0, 0, 0, 1, 0);
    v.setHostValue(4.0, 0, 0, 1, 1);

    // Block kv=1 for ALL sq positions (mask broadcasts over sq via dim[0]=1)
    mask.setHostValue(0.0f, 0, 0);
    mask.setHostValue(-1e4f, 0, 1);

    CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, &mask);

    // Both sq=0 and sq=1 should have O ≈ V[0,0,0,:] = [1, 2]
    const double tol = 1e-3;
    EXPECT_NEAR(o.getHostValue(0, 0, 0, 0), 1.0, tol);
    EXPECT_NEAR(o.getHostValue(0, 0, 0, 1), 2.0, tol);
    EXPECT_NEAR(o.getHostValue(0, 0, 1, 0), 1.0, tol);
    EXPECT_NEAR(o.getHostValue(0, 0, 1, 1), 2.0, tol);
}

TEST(TestCpuFpReferenceSdpaFp64, CausalMask)
{
    // causalMask=true: sq can attend to kv <= sq only.
    // [B=1, H=1, Sq=3, Skv=3, D=3, Dv=1]
    // Q[sq,:] = one-hot(sq), K[skv,:] = one-hot(skv) → dot(Q[sq], K[skv]) = δ(sq,skv)
    // V[skv,0] = skv+1: [1, 2, 3]
    // scale = 1/sqrt(3), eNeg = exp(-scale)
    //
    // sq=0: only kv=0 unmasked, S[0]=scale → P[0]≈1       → O≈1
    // sq=1: kv=0,1 unmasked, S[0]=0, S[1]=scale           → O = eNeg/(1+eNeg)*1 + 1/(1+eNeg)*2
    // sq=2: all unmasked, S[0]=S[1]=0, S[2]=scale         → O = (eNeg+2*eNeg+3)/(2*eNeg+1)

    Tensor<double> q({1, 1, 3, 3});
    Tensor<double> k({1, 1, 3, 3});
    Tensor<double> v({1, 1, 3, 1});
    Tensor<double> o({1, 1, 3, 1});

    q.fillWithValue(0.0);
    k.fillWithValue(0.0);

    // Q[sq,:] = one-hot(sq)
    q.setHostValue(1.0, 0, 0, 0, 0);
    q.setHostValue(1.0, 0, 0, 1, 1);
    q.setHostValue(1.0, 0, 0, 2, 2);

    // K[skv,:] = one-hot(skv)
    k.setHostValue(1.0, 0, 0, 0, 0);
    k.setHostValue(1.0, 0, 0, 1, 1);
    k.setHostValue(1.0, 0, 0, 2, 2);

    // V[skv,0] = skv+1
    v.setHostValue(1.0, 0, 0, 0, 0);
    v.setHostValue(2.0, 0, 0, 1, 0);
    v.setHostValue(3.0, 0, 0, 2, 0);

    const TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, noMask, /*causalMask=*/true);

    const float scale = 1.0f / std::sqrt(3.0f);
    const float eNeg = std::exp(-scale);

    // sq=0: only kv=0 unmasked → P[0]≈1 → O≈V[0,0]=1
    EXPECT_NEAR(o.getHostValue(0, 0, 0, 0), 1.0, 1e-3);

    // sq=1: kv=0,1 unmasked
    //   S[0]=0, S[1]=scale → maxVal=scale
    //   P[0]=eNeg/(1+eNeg), P[1]=1/(1+eNeg)
    const float p0sq1 = eNeg / (1.0f + eNeg);
    const float p1sq1 = 1.0f / (1.0f + eNeg);
    const auto expSq1 = static_cast<double>(p0sq1 * 1.0f + p1sq1 * 2.0f);
    EXPECT_NEAR(o.getHostValue(0, 0, 1, 0), expSq1, 1e-5);

    // sq=2: all kv unmasked
    //   S[0]=S[1]=0, S[2]=scale → maxVal=scale
    //   P[0]=P[1]=eNeg/(2*eNeg+1), P[2]=1/(2*eNeg+1)
    const float denom = 2.0f * eNeg + 1.0f;
    const auto expSq2 = static_cast<double>((eNeg * 1.0f + eNeg * 2.0f + 1.0f * 3.0f) / denom);
    EXPECT_NEAR(o.getHostValue(0, 0, 2, 0), expSq2, 1e-5);
}

TEST(TestCpuFpReferenceSdpaFp64, CausalMaskFutureTokensHaveNoEffect)
{
    // Verify the causal property: changing V values at masked (future) kv positions
    // produces identical output, since those positions contribute zero probability.
    // [B=1, H=1, Sq=2, Skv=3, D=2, Dv=2]
    // causalMask=true: sq=0 sees kv=0 only; sq=1 sees kv=0,1; kv=2 is always masked.

    Tensor<double> q({1, 1, 2, 2});
    Tensor<double> k({1, 1, 3, 2});
    Tensor<double> vBase({1, 1, 3, 2});
    Tensor<double> vAlt({1, 1, 3, 2}); // identical except kv=2 is wildly different
    Tensor<double> oBase({1, 1, 2, 2});
    Tensor<double> oAlt({1, 1, 2, 2});

    // Uniform Q and K so all unmasked scores are equal (softmax is uniform)
    q.fillWithValue(1.0);
    k.fillWithValue(1.0);
    vBase.fillWithValue(1.0);
    vAlt.fillWithValue(1.0);

    // Make the always-masked kv=2 position very different in vAlt
    vAlt.setHostValue(999.0, 0, 0, 2, 0);
    vAlt.setHostValue(999.0, 0, 0, 2, 1);

    const TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward(q, k, vBase, oBase, std::nullopt, noMask, /*causalMask=*/true);
    CpuFpReferenceSdpa::forward(q, k, vAlt, oAlt, std::nullopt, noMask, /*causalMask=*/true);

    // Outputs must be identical since kv=2 is always masked out
    const double tol = 1e-6;
    for(int64_t sq = 0; sq < 2; ++sq)
    {
        for(int64_t dv = 0; dv < 2; ++dv)
        {
            EXPECT_NEAR(oBase.getHostValue(std::vector<int64_t>{0, 0, sq, dv}),
                        oAlt.getHostValue(std::vector<int64_t>{0, 0, sq, dv}),
                        tol);
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-type smoke test
// ---------------------------------------------------------------------------

using TypesSdpaFwd
    = ::testing::Types<TypePair<float, float>, TypePair<half, float>, TypePair<bfloat16, float>>;

template <class T>
class CpuFpReferenceSdpaFwd : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceSdpaFwd, TypesSdpaFwd, );

TYPED_TEST(CpuFpReferenceSdpaFwd, BasicFwd)
{
    using InT = typename TypeParam::First;

    // Uniform Q/K/V: all dot products are equal → uniform softmax → O = V_val
    Tensor<InT> q({1, 2, 4, 8});
    Tensor<InT> k({1, 2, 4, 8});
    Tensor<InT> v({1, 2, 4, 8});
    Tensor<InT> o({1, 2, 4, 8});

    q.fillWithValue(static_cast<InT>(0.1f));
    k.fillWithValue(static_cast<InT>(0.1f));

    const float vVal = 0.5f;
    v.fillWithValue(static_cast<InT>(vVal));

    CpuFpReferenceSdpa::forward(q, k, v, o);

    // With uniform scores, softmax gives equal weights 1/Skv for each kv token.
    // With uniform V = vVal, O = sum_skv(1/Skv * vVal) = vVal.
    // Use a generous tolerance to accommodate half/bfloat16 precision limits.
    const float tol = 0.05f;
    EXPECT_NEAR(static_cast<float>(o.getHostValue(0, 0, 0, 0)), vVal, tol);
    EXPECT_NEAR(static_cast<float>(o.getHostValue(0, 1, 3, 7)), vVal, tol);
}
