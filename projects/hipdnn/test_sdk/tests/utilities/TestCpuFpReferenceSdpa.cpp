// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>
#include <hipdnn_test_sdk/utilities/detail/GradientCheckUtilities.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using hipdnn_test_sdk::detail::safeTestTypeCast;

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

TEST(TestCpuFpReferenceSdpaFp64, GqaDifferentKVHeads)
{
    // H=4, Hk=2, Hv=1, Skv=2: K has 2 heads, V has 1 head.
    // With Skv=2, distinct K-head values produce different attention patterns,
    // verifying both K-head and V-head mappings.
    // K head 0: [1,0] at skv=0, [0,0] at skv=1 → Q·K biased toward skv=0
    // K head 1: [0,0] at skv=0, [1,0] at skv=1 → Q·K biased toward skv=1
    // V head 0: [1,2] at skv=0, [3,4] at skv=1
    // Q heads 0,1 → K head 0 (biased toward V[skv=0]=[1,2])
    // Q heads 2,3 → K head 1 (biased toward V[skv=1]=[3,4])
    Tensor<double> q({1, 4, 1, 2});
    Tensor<double> k({1, 2, 2, 2});
    Tensor<double> v({1, 1, 2, 2});
    Tensor<double> o({1, 4, 1, 2});

    q.fillWithValue(0.0);
    k.fillWithValue(0.0);

    // Q: all heads have query [1, 0]
    for(int64_t h = 0; h < 4; ++h)
    {
        q.setHostValue(1.0, 0, h, 0, 0);
    }

    // K head 0: [1,0] at skv=0, [0,0] at skv=1
    k.setHostValue(1.0, 0, 0, 0, 0);
    // K head 1: [0,0] at skv=0, [1,0] at skv=1
    k.setHostValue(1.0, 0, 1, 1, 0);

    // V head 0: [1,2] at skv=0, [3,4] at skv=1
    v.setHostValue(1.0, 0, 0, 0, 0);
    v.setHostValue(2.0, 0, 0, 0, 1);
    v.setHostValue(3.0, 0, 0, 1, 0);
    v.setHostValue(4.0, 0, 0, 1, 1);

    CpuFpReferenceSdpa::forward(q, k, v, o);

    const float scale = 1.0f / std::sqrt(2.0f);
    const float eLow = std::exp(-scale);
    const float sumExp = 1.0f + eLow;
    const float pHigh = 1.0f / sumExp;
    const float pLow = eLow / sumExp;
    const double tol = 1e-5;

    // Q heads 0,1 → K head 0 → biased toward skv=0: O ≈ pHigh*V[0] + pLow*V[1]
    const auto expectedK0D0 = static_cast<double>(pHigh * 1.0f + pLow * 3.0f);
    const auto expectedK0D1 = static_cast<double>(pHigh * 2.0f + pLow * 4.0f);
    for(int64_t h = 0; h < 2; ++h)
    {
        EXPECT_NEAR(o.getHostValue(0, h, 0, 0), expectedK0D0, tol);
        EXPECT_NEAR(o.getHostValue(0, h, 0, 1), expectedK0D1, tol);
    }

    // Q heads 2,3 → K head 1 → biased toward skv=1: O ≈ pLow*V[0] + pHigh*V[1]
    const auto expectedK1D0 = static_cast<double>(pLow * 1.0f + pHigh * 3.0f);
    const auto expectedK1D1 = static_cast<double>(pLow * 2.0f + pHigh * 4.0f);
    for(int64_t h = 2; h < 4; ++h)
    {
        EXPECT_NEAR(o.getHostValue(0, h, 0, 0), expectedK1D0, tol);
        EXPECT_NEAR(o.getHostValue(0, h, 0, 1), expectedK1D1, tol);
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

    q.fillWithValue(safeTestTypeCast<InT>(0.1f));
    k.fillWithValue(safeTestTypeCast<InT>(0.1f));

    const float vVal = 0.5f;
    v.fillWithValue(safeTestTypeCast<InT>(vVal));

    CpuFpReferenceSdpa::forward(q, k, v, o);

    // With uniform scores, softmax gives equal weights 1/Skv for each kv token.
    // With uniform V = vVal, O = sum_skv(1/Skv * vVal) = vVal.
    // Use a generous tolerance to accommodate half/bfloat16 precision limits.
    const float tol = 0.05f;
    EXPECT_NEAR(static_cast<float>(o.getHostValue(0, 0, 0, 0)), vVal, tol);
    EXPECT_NEAR(static_cast<float>(o.getHostValue(0, 1, 3, 7)), vVal, tol);
}

// ---------------------------------------------------------------------------
// LSE (Log-Sum-Exp) Output Tests
// ---------------------------------------------------------------------------

TEST(TestCpuFpReferenceSdpaFp64, LseOutputMatchesFormula)
{
    // Small controlled input: [B=1, H=1, Sq=2, Skv=2, D=2, Dv=2]
    // Use one-hot Q/K for known scores, then verify LSE = maxVal + log(sumExp)

    Tensor<double> q({1, 1, 2, 2});
    Tensor<double> k({1, 1, 2, 2});
    Tensor<double> v({1, 1, 2, 2});
    Tensor<double> o({1, 1, 2, 2});
    Tensor<float> lse({1, 1, 2});

    q.fillWithValue(0.0);
    k.fillWithValue(0.0);

    // One-hot pattern: Q[0,0,0,:] = [1,0], Q[0,0,1,:] = [0,1]
    q.setHostValue(1.0, 0, 0, 0, 0);
    q.setHostValue(1.0, 0, 0, 1, 1);

    // K[0,0,0,:] = [1,0], K[0,0,1,:] = [0,1]
    k.setHostValue(1.0, 0, 0, 0, 0);
    k.setHostValue(1.0, 0, 0, 1, 1);

    v.setHostValue(1.0, 0, 0, 0, 0);
    v.setHostValue(2.0, 0, 0, 0, 1);
    v.setHostValue(3.0, 0, 0, 1, 0);
    v.setHostValue(4.0, 0, 0, 1, 1);

    const hipdnn_data_sdk::utilities::TensorBase<double>* noMask = nullptr;
    CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, noMask, false, &lse);

    // Manually compute expected LSE
    const float scale = 1.0f / std::sqrt(2.0f);

    // For sq=0: dot products are [1,0] → scores = [scale, 0]
    const float maxVal0 = scale;
    const float sumExp0 = std::exp(scale - maxVal0) + std::exp(0.0f - maxVal0);
    const float expectedLse0 = maxVal0 + std::log(sumExp0);

    EXPECT_NEAR(lse.getHostValue(0, 0, 0), expectedLse0, 1e-5f);

    // For sq=1: dot products are [0,1] → scores = [0, scale]
    const float maxVal1 = scale;
    const float sumExp1 = std::exp(0.0f - maxVal1) + std::exp(scale - maxVal1);
    const float expectedLse1 = maxVal1 + std::log(sumExp1);

    EXPECT_NEAR(lse.getHostValue(0, 0, 1), expectedLse1, 1e-5f);
}

TYPED_TEST(CpuFpReferenceSdpaFwd, LseAlwaysFloatType)
{
    // Verify: LSE is always float, even when inputs are half/bfloat16
    using InT = typename TypeParam::First;

    Tensor<InT> q({1, 2, 4, 8});
    Tensor<InT> k({1, 2, 4, 8});
    Tensor<InT> v({1, 2, 4, 8});
    Tensor<InT> o({1, 2, 4, 8});
    Tensor<float> lse({1, 2, 4});

    q.fillWithRandomValues(safeTestTypeCast<InT>(-1.0f), safeTestTypeCast<InT>(1.0f), 100);
    k.fillWithRandomValues(safeTestTypeCast<InT>(-1.0f), safeTestTypeCast<InT>(1.0f), 101);
    v.fillWithRandomValues(safeTestTypeCast<InT>(-1.0f), safeTestTypeCast<InT>(1.0f), 102);

    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, noMask, false, &lse);

    // LSE values should be reasonable (not NaN, not inf for non-masked rows)
    for(int b = 0; b < 1; ++b)
    {
        for(int h = 0; h < 2; ++h)
        {
            for(int sq = 0; sq < 4; ++sq)
            {
                const float lseVal = lse.getHostValue(b, h, sq);
                EXPECT_FALSE(std::isnan(lseVal)) << "NaN at [" << b << "," << h << "," << sq << "]";
                // LSE typically in range [-10, 10] for random inputs with default scale
                EXPECT_GT(lseVal, -20.0f)
                    << "Unexpectedly small LSE at [" << b << "," << h << "," << sq << "]";
                EXPECT_LT(lseVal, 20.0f)
                    << "Unexpectedly large LSE at [" << b << "," << h << "," << sq << "]";
            }
        }
    }
}

TEST(TestCpuFpReferenceSdpaFp64, LseWithMultipleBatchHeads)
{
    // Verify: LSE computed correctly for B > 1, H > 1

    Tensor<double> q({2, 4, 16, 32});
    Tensor<double> k({2, 4, 16, 32});
    Tensor<double> v({2, 4, 16, 32});
    Tensor<double> o({2, 4, 16, 32});
    Tensor<float> lse({2, 4, 16});

    q.fillWithRandomValues(-1.0, 1.0, 200);
    k.fillWithRandomValues(-1.0, 1.0, 201);
    v.fillWithRandomValues(-1.0, 1.0, 202);

    const hipdnn_data_sdk::utilities::TensorBase<double>* noMask = nullptr;
    CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, noMask, false, &lse);

    // Verify all LSE values are reasonable (no NaN, finite)
    for(int b = 0; b < 2; ++b)
    {
        for(int h = 0; h < 4; ++h)
        {
            for(int sq = 0; sq < 16; ++sq)
            {
                const float lseVal = lse.getHostValue(b, h, sq);
                EXPECT_TRUE(std::isfinite(lseVal))
                    << "LSE not finite at [" << b << "," << h << "," << sq << "]";
            }
        }
    }
}

TEST(TestCpuFpReferenceSdpaFp64, LseWithCausalMask)
{
    // [B=1, H=1, Sq=4, Skv=4]
    // Causal mask: sq=0 sees 1 position, sq=1 sees 2, sq=2 sees 3, sq=3 sees 4

    Tensor<double> q({1, 1, 4, 8});
    Tensor<double> k({1, 1, 4, 8});
    Tensor<double> v({1, 1, 4, 8});
    Tensor<double> o({1, 1, 4, 8});
    Tensor<float> lse({1, 1, 4});

    q.fillWithValue(1.0);
    k.fillWithValue(1.0);
    v.fillWithValue(1.0);

    const hipdnn_data_sdk::utilities::TensorBase<double>* noMask = nullptr;
    CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, noMask, true, &lse);

    // With uniform Q/K and causal mask:
    // sq=0: 1 valid position → smaller sumExp → smaller LSE
    // sq=3: 4 valid positions → larger sumExp → larger LSE
    // LSE should increase monotonically as sq increases

    const float lse0 = lse.getHostValue(0, 0, 0);
    const float lse1 = lse.getHostValue(0, 0, 1);
    const float lse2 = lse.getHostValue(0, 0, 2);
    const float lse3 = lse.getHostValue(0, 0, 3);

    EXPECT_LT(lse0, lse1) << "LSE should increase with more unmasked positions";
    EXPECT_LT(lse1, lse2);
    EXPECT_LT(lse2, lse3);
}

TEST(TestCpuFpReferenceSdpaFp64, LseWithFullyMaskedRow)
{
    // Edge case: All scores masked to -inf
    // LSE should be -inf (mathematically correct: log(0) = -inf)

    Tensor<double> q({1, 1, 2, 2});
    Tensor<double> k({1, 1, 2, 2});
    Tensor<double> v({1, 1, 2, 2});
    Tensor<double> o({1, 1, 2, 2});
    Tensor<float> lse({1, 1, 2});

    q.fillWithValue(1.0);
    k.fillWithValue(1.0);
    v.fillWithValue(1.0);

    // Create additive mask that blocks ALL positions for sq=0
    Tensor<float> mask({2, 2});
    mask.fillWithValue(-std::numeric_limits<float>::infinity());

    // But allow sq=1 to see all positions
    mask.setHostValue(0.0f, 1, 0);
    mask.setHostValue(0.0f, 1, 1);

    const hipdnn_data_sdk::utilities::TensorBase<float>* maskPtr = &mask;
    CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, maskPtr, false, &lse);

    // sq=0: all masked → LSE = -inf (or NaN due to -inf - (-inf) in exp)
    const float lse0 = lse.getHostValue(0, 0, 0);
    // When all scores are -inf: maxVal = -inf, exp(-inf - (-inf)) = exp(NaN) = NaN
    // So LSE may be NaN rather than -inf. Both indicate "no valid attention weights"
    EXPECT_TRUE((std::isinf(lse0) && lse0 < 0) || std::isnan(lse0))
        << "LSE should be -inf or NaN for fully masked row, got: " << lse0;

    // sq=1: normal → LSE finite
    const float lse1 = lse.getHostValue(0, 0, 1);
    EXPECT_TRUE(std::isfinite(lse1)) << "LSE should be finite for normal row";
}

TEST(TestCpuFpReferenceSdpaFp64, LseWithAdditiveMask)
{
    // Verify: LSE accounts for additive mask contributions

    Tensor<double> q({1, 1, 2, 2});
    Tensor<double> k({1, 1, 2, 2});
    Tensor<double> v({1, 1, 2, 2});
    Tensor<double> o1({1, 1, 2, 2});
    Tensor<double> o2({1, 1, 2, 2});
    Tensor<float> lse1({1, 1, 2});
    Tensor<float> lse2({1, 1, 2});

    // One-hot Q/K for predictable scores
    q.fillWithValue(0.0);
    k.fillWithValue(0.0);
    q.setHostValue(1.0, 0, 0, 0, 0);
    q.setHostValue(1.0, 0, 0, 1, 1);
    k.setHostValue(1.0, 0, 0, 0, 0);
    k.setHostValue(1.0, 0, 0, 1, 1);
    v.fillWithValue(1.0);

    // Forward without mask
    const hipdnn_data_sdk::utilities::TensorBase<double>* noMask = nullptr;
    CpuFpReferenceSdpa::forward(q, k, v, o1, std::nullopt, noMask, false, &lse1);

    // Forward with mask that suppresses one position
    Tensor<float> mask({2, 2});
    mask.fillWithValue(0.0f);
    mask.setHostValue(-1e4f, 0, 1); // Block skv=1 for sq=0

    CpuFpReferenceSdpa::forward(q, k, v, o2, std::nullopt, &mask, false, &lse2);

    // LSE should differ between masked and unmasked cases
    const float lseNomask = lse1.getHostValue(0, 0, 0);
    const float lseWithmask = lse2.getHostValue(0, 0, 0);

    EXPECT_NE(lseNomask, lseWithmask) << "LSE should change with additive mask";

    // With mask blocking skv=1, LSE should be lower (fewer effective positions)
    EXPECT_LT(lseWithmask, lseNomask) << "LSE should decrease when positions are masked";
}

TEST(TestCpuFpReferenceSdpaFp64, LseWrongRank)
{
    Tensor<double> q({1, 1, 2, 2});
    Tensor<double> k({1, 1, 2, 2});
    Tensor<double> v({1, 1, 2, 2});
    Tensor<double> o({1, 1, 2, 2});

    // Wrong rank: rank-4 instead of rank-3
    Tensor<float> lseWrong({1, 1, 2, 1});

    EXPECT_THROW(
        {
            q.fillWithValue(1.0);
            k.fillWithValue(1.0);
            v.fillWithValue(1.0);
            const hipdnn_data_sdk::utilities::TensorBase<double>* noMask = nullptr;
            CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, noMask, false, &lseWrong);
        },
        std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaFp64, LseWrongShape)
{
    Tensor<double> q({2, 4, 16, 32});
    Tensor<double> k({2, 4, 16, 32});
    Tensor<double> v({2, 4, 16, 32});
    Tensor<double> o({2, 4, 16, 32});

    // Wrong shape: [2, 4, 8] instead of [2, 4, 16]
    Tensor<float> lseWrong({2, 4, 8});

    EXPECT_THROW(
        {
            q.fillWithRandomValues(-1.0, 1.0, 42);
            k.fillWithRandomValues(-1.0, 1.0, 43);
            v.fillWithRandomValues(-1.0, 1.0, 44);
            const hipdnn_data_sdk::utilities::TensorBase<double>* noMask = nullptr;
            CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, noMask, false, &lseWrong);
        },
        std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaFp64, LseMismatchedBatch)
{
    Tensor<double> q({2, 1, 4, 8});
    Tensor<double> k({2, 1, 4, 8});
    Tensor<double> v({2, 1, 4, 8});
    Tensor<double> o({2, 1, 4, 8});

    // Wrong batch: [1, 1, 4] instead of [2, 1, 4]
    Tensor<float> lseWrong({1, 1, 4});

    EXPECT_THROW(
        {
            q.fillWithRandomValues(-1.0, 1.0, 50);
            k.fillWithRandomValues(-1.0, 1.0, 51);
            v.fillWithRandomValues(-1.0, 1.0, 52);
            const hipdnn_data_sdk::utilities::TensorBase<double>* noMask = nullptr;
            CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, noMask, false, &lseWrong);
        },
        std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaFp64, LseWithGqa)
{
    // GQA: H=8, HkV=2 (headsPerKvHead = 4)
    // Verify: LSE computed correctly despite K/V head sharing

    Tensor<double> q({1, 8, 16, 32});
    Tensor<double> k({1, 2, 16, 32});
    Tensor<double> v({1, 2, 16, 32});
    Tensor<double> o({1, 8, 16, 32});
    Tensor<float> lse({1, 8, 16});

    q.fillWithRandomValues(-1.0, 1.0, 300);
    k.fillWithRandomValues(-1.0, 1.0, 301);
    v.fillWithRandomValues(-1.0, 1.0, 302);

    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, noMask, false, &lse);

    // All LSE values should be reasonable
    for(int h = 0; h < 8; ++h)
    {
        for(int sq = 0; sq < 16; ++sq)
        {
            const float lseVal = lse.getHostValue(0, h, sq);
            EXPECT_TRUE(std::isfinite(lseVal)) << "LSE not finite at h=" << h << ", sq=" << sq;
            EXPECT_GT(lseVal, -20.0f);
            EXPECT_LT(lseVal, 20.0f);
        }
    }

    // Different Q heads using same KV head should have different LSE
    // (because Q differs, even though K/V are shared)
    const float lseH0 = lse.getHostValue(0, 0, 0);
    const float lseH1 = lse.getHostValue(0, 1, 0);

    // With random Q, LSE should almost certainly differ
    EXPECT_NE(lseH0, lseH1) << "LSE should differ for different Q heads even with shared KV";
}

// ---------------------------------------------------------------------------
// Backward pass tests
//
// These tests use Fp32 (unlike forward tests which use Fp64) because the GPU
// backward kernels (AITER/Flash Attention) only support fp16/bf16 input/output
// with fp32 internal compute precision. The CPU reference backward tests use
// Fp32 to match the actual kernel compute precision.
// ---------------------------------------------------------------------------

TEST(TestCpuFpReferenceSdpaBwdFp32, BackwardSanity)
{
    // Hand-verified backward pass with small deterministic inputs.
    // [B=1, H=1, Sq=2, Skv=2, D=2, Dv=2]
    //
    // Q = [[1,0],[0,1]]  K = [[1,0],[0,1]]  V = [[1,2],[3,4]]  dO = [[1,1],[1,1]]
    // scale = 1/sqrt(2)
    //
    // By symmetry of Q and K (identity matrices), the softmax probabilities
    // for sq=0 are [pH, pL] and for sq=1 are [pL, pH], where:
    //   pH = exp(scale) / (exp(scale) + 1)   (probability of the matching key)
    //   pL = 1 / (exp(scale) + 1)            (probability of the non-matching key)
    //
    // With dO = ones, dP[sq,skv] = sum(V[skv,:]) = [3, 7].
    // dS[sq,skv] = P[sq,skv] * (dP[sq,skv] - D[sq]) simplifies to:
    //   dS[*,matching] = -4*pH*pL,  dS[*,non-matching] = +4*pH*pL
    //
    // Expected gradients (c = 4*pH*pL*scale):
    //   dQ = [[-c, c], [-c, c]]
    //   dK = [[-c, -c], [c, c]]
    //   dV = [[1, 1], [1, 1]]
    Tensor<float> q({1, 1, 2, 2});
    Tensor<float> k({1, 1, 2, 2});
    Tensor<float> v({1, 1, 2, 2});
    Tensor<float> o({1, 1, 2, 2});
    Tensor<float> dO({1, 1, 2, 2});
    Tensor<float> dQ({1, 1, 2, 2});
    Tensor<float> dK({1, 1, 2, 2});
    Tensor<float> dV({1, 1, 2, 2});

    q.fillWithValue(0.0f);
    k.fillWithValue(0.0f);
    v.fillWithValue(0.0f);

    q.setHostValue(1.0f, 0, 0, 0, 0);
    q.setHostValue(1.0f, 0, 0, 1, 1);
    k.setHostValue(1.0f, 0, 0, 0, 0);
    k.setHostValue(1.0f, 0, 0, 1, 1);
    v.setHostValue(1.0f, 0, 0, 0, 0);
    v.setHostValue(2.0f, 0, 0, 0, 1);
    v.setHostValue(3.0f, 0, 0, 1, 0);
    v.setHostValue(4.0f, 0, 0, 1, 1);

    dO.fillWithValue(1.0f);

    // Forward pass
    CpuFpReferenceSdpa::forward(q, k, v, o);

    // Backward pass
    CpuFpReferenceSdpa::backward(q, k, v, o, dO, dQ, dK, dV);

    // Compute expected values
    const float scale = 1.0f / std::sqrt(2.0f);
    const float eS = std::exp(scale);
    const float pH = eS / (eS + 1.0f);
    const float pL = 1.0f / (eS + 1.0f);
    const float c = 4.0f * pH * pL * scale;
    const float tol = 1e-5f;

    // dQ = [[-c, c], [-c, c]]
    EXPECT_NEAR(dQ.getHostValue(0, 0, 0, 0), -c, tol);
    EXPECT_NEAR(dQ.getHostValue(0, 0, 0, 1), c, tol);
    EXPECT_NEAR(dQ.getHostValue(0, 0, 1, 0), -c, tol);
    EXPECT_NEAR(dQ.getHostValue(0, 0, 1, 1), c, tol);

    // dK = [[-c, -c], [c, c]]
    EXPECT_NEAR(dK.getHostValue(0, 0, 0, 0), -c, tol);
    EXPECT_NEAR(dK.getHostValue(0, 0, 0, 1), -c, tol);
    EXPECT_NEAR(dK.getHostValue(0, 0, 1, 0), c, tol);
    EXPECT_NEAR(dK.getHostValue(0, 0, 1, 1), c, tol);

    // dV = [[1, 1], [1, 1]]  (P columns sum to 1 by softmax symmetry)
    EXPECT_NEAR(dV.getHostValue(0, 0, 0, 0), 1.0f, tol);
    EXPECT_NEAR(dV.getHostValue(0, 0, 0, 1), 1.0f, tol);
    EXPECT_NEAR(dV.getHostValue(0, 0, 1, 0), 1.0f, tol);
    EXPECT_NEAR(dV.getHostValue(0, 0, 1, 1), 1.0f, tol);
}

TEST(TestCpuFpReferenceSdpaBwdFp32, BackwardMHA)
{
    // Test Multi-Head Attention (H_q == H_kv)
    // [B=1, H=2, Sq=4, Skv=4, D=8, Dv=8]
    Tensor<float> q({1, 2, 4, 8});
    Tensor<float> k({1, 2, 4, 8});
    Tensor<float> v({1, 2, 4, 8});
    Tensor<float> o({1, 2, 4, 8});
    Tensor<float> dO({1, 2, 4, 8});
    Tensor<float> dQ({1, 2, 4, 8});
    Tensor<float> dK({1, 2, 4, 8});
    Tensor<float> dV({1, 2, 4, 8});

    q.fillWithRandomValues(-1.0f, 1.0f, 100);
    k.fillWithRandomValues(-1.0f, 1.0f, 101);
    v.fillWithRandomValues(-1.0f, 1.0f, 102);
    dO.fillWithRandomValues(-1.0f, 1.0f, 103);

    // Forward pass
    CpuFpReferenceSdpa::forward(q, k, v, o);

    // Backward pass
    CpuFpReferenceSdpa::backward(q, k, v, o, dO, dQ, dK, dV);

    // Check gradients are finite
    for(int h = 0; h < 2; ++h)
    {
        for(int sq = 0; sq < 4; ++sq)
        {
            for(int d = 0; d < 8; ++d)
            {
                EXPECT_TRUE(std::isfinite(dQ.getHostValue(0, h, sq, d)))
                    << "dQ not finite at h=" << h << ", sq=" << sq << ", d=" << d;
                EXPECT_TRUE(std::isfinite(dK.getHostValue(0, h, sq, d)))
                    << "dK not finite at h=" << h << ", sq=" << sq << ", d=" << d;
                EXPECT_TRUE(std::isfinite(dV.getHostValue(0, h, sq, d)))
                    << "dV not finite at h=" << h << ", sq=" << sq << ", d=" << d;
            }
        }
    }
}

TEST(TestCpuFpReferenceSdpaBwdFp32, BackwardGQA)
{
    // Test Grouped Query Attention (H_q = 4, H_kv = 1)
    // [B=1, H_q=4, H_kv=1, Sq=4, Skv=4, D=8, Dv=8]
    Tensor<float> q({1, 4, 4, 8});
    Tensor<float> k({1, 1, 4, 8});
    Tensor<float> v({1, 1, 4, 8});
    Tensor<float> o({1, 4, 4, 8});
    Tensor<float> dO({1, 4, 4, 8});
    Tensor<float> dQ({1, 4, 4, 8});
    Tensor<float> dK({1, 1, 4, 8});
    Tensor<float> dV({1, 1, 4, 8});

    q.fillWithRandomValues(-1.0f, 1.0f, 200);
    k.fillWithRandomValues(-1.0f, 1.0f, 201);
    v.fillWithRandomValues(-1.0f, 1.0f, 202);
    dO.fillWithRandomValues(-1.0f, 1.0f, 203);

    // Forward pass
    CpuFpReferenceSdpa::forward(q, k, v, o);

    // Backward pass (should handle GQA accumulation correctly)
    CpuFpReferenceSdpa::backward(q, k, v, o, dO, dQ, dK, dV);

    // Check gradients are finite
    for(int h = 0; h < 4; ++h)
    {
        for(int sq = 0; sq < 4; ++sq)
        {
            for(int d = 0; d < 8; ++d)
            {
                EXPECT_TRUE(std::isfinite(dQ.getHostValue(0, h, sq, d)))
                    << "dQ not finite at h=" << h << ", sq=" << sq << ", d=" << d;
            }
        }
    }
    for(int sq = 0; sq < 4; ++sq)
    {
        for(int d = 0; d < 8; ++d)
        {
            EXPECT_TRUE(std::isfinite(dK.getHostValue(0, 0, sq, d)))
                << "dK not finite at sq=" << sq << ", d=" << d;
            EXPECT_TRUE(std::isfinite(dV.getHostValue(0, 0, sq, d)))
                << "dV not finite at sq=" << sq << ", d=" << d;
        }
    }

    // For GQA, dK and dV should accumulate contributions from multiple Q heads
    // Verify that gradients are non-zero (as they should accumulate from 4 Q heads)
    float dkSum = 0.0f;
    for(int sq = 0; sq < 4; ++sq)
    {
        for(int d = 0; d < 8; ++d)
        {
            dkSum += std::abs(dK.getHostValue(0, 0, sq, d));
        }
    }
    EXPECT_GT(dkSum, 0.0f) << "dK should have non-zero gradients from GQA accumulation";
}

TEST(TestCpuFpReferenceSdpaBwdFp32, BackwardCausalMask)
{
    // Test backward pass with causal masking
    // [B=1, H=1, Sq=4, Skv=4, D=4, Dv=4]
    Tensor<float> q({1, 1, 4, 4});
    Tensor<float> k({1, 1, 4, 4});
    Tensor<float> v({1, 1, 4, 4});
    Tensor<float> o({1, 1, 4, 4});
    Tensor<float> dO({1, 1, 4, 4});
    Tensor<float> dQ({1, 1, 4, 4});
    Tensor<float> dK({1, 1, 4, 4});
    Tensor<float> dV({1, 1, 4, 4});

    q.fillWithRandomValues(-1.0f, 1.0f, 300);
    k.fillWithRandomValues(-1.0f, 1.0f, 301);
    v.fillWithRandomValues(-1.0f, 1.0f, 302);
    dO.fillWithRandomValues(-1.0f, 1.0f, 303);

    const bool causalMask = true;
    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;

    // Forward pass with causal mask
    CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, noMask, causalMask);

    // Backward pass with causal mask
    CpuFpReferenceSdpa::backward(
        q, k, v, o, dO, dQ, dK, dV, std::nullopt, nullptr, noMask, causalMask);

    // Check gradients are finite
    for(int sq = 0; sq < 4; ++sq)
    {
        for(int d = 0; d < 4; ++d)
        {
            EXPECT_TRUE(std::isfinite(dQ.getHostValue(0, 0, sq, d)))
                << "dQ not finite at sq=" << sq << ", d=" << d;
            EXPECT_TRUE(std::isfinite(dK.getHostValue(0, 0, sq, d)))
                << "dK not finite at sq=" << sq << ", d=" << d;
            EXPECT_TRUE(std::isfinite(dV.getHostValue(0, 0, sq, d)))
                << "dV not finite at sq=" << sq << ", d=" << d;
        }
    }
}

// Compares LSE-based backward path against full softmax recomputation path.
// Both paths must use float tensors (matching ComputeDataType = float default)
// so that forward and backward compute at the same precision.
TEST(TestCpuFpReferenceSdpaBwd, BackwardWithLSE)
{
    // [B=1, H=2, Sq=4, Skv=4, D=8, Dv=8]
    Tensor<float> q({1, 2, 4, 8});
    Tensor<float> k({1, 2, 4, 8});
    Tensor<float> v({1, 2, 4, 8});
    Tensor<float> o({1, 2, 4, 8});
    Tensor<float> lse({1, 2, 4});
    Tensor<float> dO({1, 2, 4, 8});
    Tensor<float> dQWithLse({1, 2, 4, 8});
    Tensor<float> dKWithLse({1, 2, 4, 8});
    Tensor<float> dVWithLse({1, 2, 4, 8});
    Tensor<float> dQNoLse({1, 2, 4, 8});
    Tensor<float> dKNoLse({1, 2, 4, 8});
    Tensor<float> dVNoLse({1, 2, 4, 8});

    q.fillWithRandomValues(-1.0f, 1.0f, 400);
    k.fillWithRandomValues(-1.0f, 1.0f, 401);
    v.fillWithRandomValues(-1.0f, 1.0f, 402);
    dO.fillWithRandomValues(-1.0f, 1.0f, 403);

    const TensorBase<float>* noMask = nullptr;

    // Forward pass with LSE
    CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, noMask, false, &lse);

    // Backward pass WITH LSE (efficient)
    CpuFpReferenceSdpa::backward(
        q, k, v, o, dO, dQWithLse, dKWithLse, dVWithLse, std::nullopt, &lse);

    // Backward pass WITHOUT LSE (recompute from scratch)
    CpuFpReferenceSdpa::backward(q, k, v, o, dO, dQNoLse, dKNoLse, dVNoLse);

    // Both paths compute in float, so results should be very close.
    // LSE path uses exp(score - lse) while no-LSE path uses exp(score - max) / sumExp.
    // These differ by FP operation ordering. If this tolerance causes flaky failures,
    // it can be relaxed to 1e-4f.
    const float tol = 1e-5f;
    for(int h = 0; h < 2; ++h)
    {
        for(int sq = 0; sq < 4; ++sq)
        {
            for(int d = 0; d < 8; ++d)
            {
                EXPECT_NEAR(
                    dQWithLse.getHostValue(0, h, sq, d), dQNoLse.getHostValue(0, h, sq, d), tol)
                    << "dQ mismatch at h=" << h << ", sq=" << sq << ", d=" << d;
                EXPECT_NEAR(
                    dKWithLse.getHostValue(0, h, sq, d), dKNoLse.getHostValue(0, h, sq, d), tol)
                    << "dK mismatch at h=" << h << ", sq=" << sq << ", d=" << d;
                EXPECT_NEAR(
                    dVWithLse.getHostValue(0, h, sq, d), dVNoLse.getHostValue(0, h, sq, d), tol)
                    << "dV mismatch at h=" << h << ", sq=" << sq << ", d=" << d;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Finite-difference gradient checking
// ---------------------------------------------------------------------------
//
// Validates backward() by comparing its analytical gradients against numerical
// gradients computed via central finite differences on the forward pass:
//
//   dL/dX[i] ≈ (L(X + ε·e_i) − L(X − ε·e_i)) / (2ε)
//
// where L = Σ(dO ⊙ O) is a scalar loss and O = forward(Q, K, V).
// This is the gold-standard test for gradient correctness.

using hipdnn_test_sdk::detail::compareGradients;
using hipdnn_test_sdk::detail::computeDotProductLoss;
using hipdnn_test_sdk::detail::numericalGradient;

TEST(TestCpuFpReferenceSdpaGradCheck, MHA)
{
    // MHA: H_q == H_kv, no masks
    // [B=1, H=2, Sq=3, Skv=3, D=4, Dv=4]
    constexpr int64_t BATCH = 1;
    constexpr int64_t HEADS = 2;
    constexpr int64_t SEQ_Q = 3;
    constexpr int64_t SEQ_KV = 3;
    constexpr int64_t HEAD_DIM = 4;
    constexpr int64_t HEAD_DIM_V = 4;

    Tensor<float> q({BATCH, HEADS, SEQ_Q, HEAD_DIM});
    Tensor<float> k({BATCH, HEADS, SEQ_KV, HEAD_DIM});
    Tensor<float> v({BATCH, HEADS, SEQ_KV, HEAD_DIM_V});
    Tensor<float> o({BATCH, HEADS, SEQ_Q, HEAD_DIM_V});
    Tensor<float> dO({BATCH, HEADS, SEQ_Q, HEAD_DIM_V});

    q.fillWithRandomValues(-0.5f, 0.5f, 1000);
    k.fillWithRandomValues(-0.5f, 0.5f, 1001);
    v.fillWithRandomValues(-0.5f, 0.5f, 1002);
    dO.fillWithRandomValues(-0.5f, 0.5f, 1003);

    // Analytical backward
    CpuFpReferenceSdpa::forward(q, k, v, o);

    Tensor<float> dQ({BATCH, HEADS, SEQ_Q, HEAD_DIM});
    Tensor<float> dK({BATCH, HEADS, SEQ_KV, HEAD_DIM});
    Tensor<float> dV({BATCH, HEADS, SEQ_KV, HEAD_DIM_V});
    CpuFpReferenceSdpa::backward(q, k, v, o, dO, dQ, dK, dV);

    // Numerical gradient check (float precision)
    constexpr double EPS = 1e-3;
    constexpr double REL_TOL = 1e-2;
    constexpr double ABS_TOL = 1e-4;

    auto fwdLoss = [&]() -> double {
        CpuFpReferenceSdpa::forward(q, k, v, o);
        return computeDotProductLoss(dO, o);
    };

    // dQ check
    {
        Tensor<float> dQNum({BATCH, HEADS, SEQ_Q, HEAD_DIM});
        numericalGradient(q, dQNum, EPS, fwdLoss);
        auto result = compareGradients(dQ, dQNum, REL_TOL, ABS_TOL);
        EXPECT_EQ(result.failCount, 0)
            << "dQ grad check failed: maxRelErr=" << result.maxRelErr
            << ", maxAbsErr=" << result.maxAbsErr << ", failures=" << result.failCount;
    }

    // dK check
    {
        Tensor<float> dKNum({BATCH, HEADS, SEQ_KV, HEAD_DIM});
        numericalGradient(k, dKNum, EPS, fwdLoss);
        auto result = compareGradients(dK, dKNum, REL_TOL, ABS_TOL);
        EXPECT_EQ(result.failCount, 0)
            << "dK grad check failed: maxRelErr=" << result.maxRelErr
            << ", maxAbsErr=" << result.maxAbsErr << ", failures=" << result.failCount;
    }

    // dV check
    {
        Tensor<float> dVNum({BATCH, HEADS, SEQ_KV, HEAD_DIM_V});
        numericalGradient(v, dVNum, EPS, fwdLoss);
        auto result = compareGradients(dV, dVNum, REL_TOL, ABS_TOL);
        EXPECT_EQ(result.failCount, 0)
            << "dV grad check failed: maxRelErr=" << result.maxRelErr
            << ", maxAbsErr=" << result.maxAbsErr << ", failures=" << result.failCount;
    }
}

TEST(TestCpuFpReferenceSdpaGradCheck, GQA)
{
    // GQA: H_q=4, H_kv=1 — tests KV gradient accumulation across Q heads
    // [B=1, H_q=4, H_kv=1, Sq=3, Skv=3, D=4, Dv=4]
    constexpr int64_t BATCH = 1;
    constexpr int64_t HEADS_Q = 4;
    constexpr int64_t HEADS_KV = 1;
    constexpr int64_t SEQ_Q = 3;
    constexpr int64_t SEQ_KV = 3;
    constexpr int64_t HEAD_DIM = 4;
    constexpr int64_t HEAD_DIM_V = 4;

    Tensor<float> q({BATCH, HEADS_Q, SEQ_Q, HEAD_DIM});
    Tensor<float> k({BATCH, HEADS_KV, SEQ_KV, HEAD_DIM});
    Tensor<float> v({BATCH, HEADS_KV, SEQ_KV, HEAD_DIM_V});
    Tensor<float> o({BATCH, HEADS_Q, SEQ_Q, HEAD_DIM_V});
    Tensor<float> dO({BATCH, HEADS_Q, SEQ_Q, HEAD_DIM_V});

    q.fillWithRandomValues(-0.5f, 0.5f, 2000);
    k.fillWithRandomValues(-0.5f, 0.5f, 2001);
    v.fillWithRandomValues(-0.5f, 0.5f, 2002);
    dO.fillWithRandomValues(-0.5f, 0.5f, 2003);

    CpuFpReferenceSdpa::forward(q, k, v, o);

    Tensor<float> dQ({BATCH, HEADS_Q, SEQ_Q, HEAD_DIM});
    Tensor<float> dK({BATCH, HEADS_KV, SEQ_KV, HEAD_DIM});
    Tensor<float> dV({BATCH, HEADS_KV, SEQ_KV, HEAD_DIM_V});
    CpuFpReferenceSdpa::backward(q, k, v, o, dO, dQ, dK, dV);

    constexpr double EPS = 1e-3;
    constexpr double REL_TOL = 1e-2;
    constexpr double ABS_TOL = 1e-4;

    auto fwdLoss = [&]() -> double {
        CpuFpReferenceSdpa::forward(q, k, v, o);
        return computeDotProductLoss(dO, o);
    };

    // dQ check
    {
        Tensor<float> dQNum({BATCH, HEADS_Q, SEQ_Q, HEAD_DIM});
        numericalGradient(q, dQNum, EPS, fwdLoss);
        auto result = compareGradients(dQ, dQNum, REL_TOL, ABS_TOL);
        EXPECT_EQ(result.failCount, 0)
            << "dQ grad check failed: maxRelErr=" << result.maxRelErr
            << ", maxAbsErr=" << result.maxAbsErr << ", failures=" << result.failCount;
    }

    // dK check — accumulates from 4 Q heads
    {
        Tensor<float> dKNum({BATCH, HEADS_KV, SEQ_KV, HEAD_DIM});
        numericalGradient(k, dKNum, EPS, fwdLoss);
        auto result = compareGradients(dK, dKNum, REL_TOL, ABS_TOL);
        EXPECT_EQ(result.failCount, 0)
            << "dK grad check failed: maxRelErr=" << result.maxRelErr
            << ", maxAbsErr=" << result.maxAbsErr << ", failures=" << result.failCount;
    }

    // dV check — accumulates from 4 Q heads
    {
        Tensor<float> dVNum({BATCH, HEADS_KV, SEQ_KV, HEAD_DIM_V});
        numericalGradient(v, dVNum, EPS, fwdLoss);
        auto result = compareGradients(dV, dVNum, REL_TOL, ABS_TOL);
        EXPECT_EQ(result.failCount, 0)
            << "dV grad check failed: maxRelErr=" << result.maxRelErr
            << ", maxAbsErr=" << result.maxAbsErr << ", failures=" << result.failCount;
    }
}

TEST(TestCpuFpReferenceSdpaGradCheck, CausalMask)
{
    // MHA with causal masking
    // [B=1, H=1, Sq=4, Skv=4, D=4, Dv=4]
    constexpr int64_t BATCH = 1;
    constexpr int64_t HEADS = 1;
    constexpr int64_t SEQ_Q = 4;
    constexpr int64_t SEQ_KV = 4;
    constexpr int64_t HEAD_DIM = 4;
    constexpr int64_t HEAD_DIM_V = 4;

    Tensor<float> q({BATCH, HEADS, SEQ_Q, HEAD_DIM});
    Tensor<float> k({BATCH, HEADS, SEQ_KV, HEAD_DIM});
    Tensor<float> v({BATCH, HEADS, SEQ_KV, HEAD_DIM_V});
    Tensor<float> o({BATCH, HEADS, SEQ_Q, HEAD_DIM_V});
    Tensor<float> dO({BATCH, HEADS, SEQ_Q, HEAD_DIM_V});

    q.fillWithRandomValues(-0.5f, 0.5f, 3000);
    k.fillWithRandomValues(-0.5f, 0.5f, 3001);
    v.fillWithRandomValues(-0.5f, 0.5f, 3002);
    dO.fillWithRandomValues(-0.5f, 0.5f, 3003);

    const TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, noMask, /*causalMask=*/true);

    Tensor<float> dQ({BATCH, HEADS, SEQ_Q, HEAD_DIM});
    Tensor<float> dK({BATCH, HEADS, SEQ_KV, HEAD_DIM});
    Tensor<float> dV({BATCH, HEADS, SEQ_KV, HEAD_DIM_V});
    CpuFpReferenceSdpa::backward(
        q, k, v, o, dO, dQ, dK, dV, std::nullopt, nullptr, noMask, /*causalMask=*/true);

    constexpr double EPS = 1e-3;
    constexpr double REL_TOL = 1e-2;
    constexpr double ABS_TOL = 1e-4;

    auto fwdLoss = [&]() -> double {
        CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, noMask, /*causalMask=*/true);
        return computeDotProductLoss(dO, o);
    };

    {
        Tensor<float> dQNum({BATCH, HEADS, SEQ_Q, HEAD_DIM});
        numericalGradient(q, dQNum, EPS, fwdLoss);
        auto result = compareGradients(dQ, dQNum, REL_TOL, ABS_TOL);
        EXPECT_EQ(result.failCount, 0)
            << "dQ grad check (causal) failed: maxRelErr=" << result.maxRelErr
            << ", maxAbsErr=" << result.maxAbsErr;
    }

    {
        Tensor<float> dKNum({BATCH, HEADS, SEQ_KV, HEAD_DIM});
        numericalGradient(k, dKNum, EPS, fwdLoss);
        auto result = compareGradients(dK, dKNum, REL_TOL, ABS_TOL);
        EXPECT_EQ(result.failCount, 0)
            << "dK grad check (causal) failed: maxRelErr=" << result.maxRelErr
            << ", maxAbsErr=" << result.maxAbsErr;
    }

    {
        Tensor<float> dVNum({BATCH, HEADS, SEQ_KV, HEAD_DIM_V});
        numericalGradient(v, dVNum, EPS, fwdLoss);
        auto result = compareGradients(dV, dVNum, REL_TOL, ABS_TOL);
        EXPECT_EQ(result.failCount, 0)
            << "dV grad check (causal) failed: maxRelErr=" << result.maxRelErr
            << ", maxAbsErr=" << result.maxAbsErr;
    }
}

TEST(TestCpuFpReferenceSdpaGradCheck, CustomScale)
{
    // MHA with a non-default attention scale
    // [B=1, H=1, Sq=3, Skv=3, D=4, Dv=4]
    constexpr int64_t BATCH = 1;
    constexpr int64_t HEADS = 1;
    constexpr int64_t SEQ_Q = 3;
    constexpr int64_t SEQ_KV = 3;
    constexpr int64_t HEAD_DIM = 4;
    constexpr int64_t HEAD_DIM_V = 4;

    Tensor<float> q({BATCH, HEADS, SEQ_Q, HEAD_DIM});
    Tensor<float> k({BATCH, HEADS, SEQ_KV, HEAD_DIM});
    Tensor<float> v({BATCH, HEADS, SEQ_KV, HEAD_DIM_V});
    Tensor<float> o({BATCH, HEADS, SEQ_Q, HEAD_DIM_V});
    Tensor<float> dO({BATCH, HEADS, SEQ_Q, HEAD_DIM_V});

    q.fillWithRandomValues(-0.5f, 0.5f, 4000);
    k.fillWithRandomValues(-0.5f, 0.5f, 4001);
    v.fillWithRandomValues(-0.5f, 0.5f, 4002);
    dO.fillWithRandomValues(-0.5f, 0.5f, 4003);

    const auto customScale = std::optional<float>{0.3f};
    CpuFpReferenceSdpa::forward(q, k, v, o, customScale);

    Tensor<float> dQ({BATCH, HEADS, SEQ_Q, HEAD_DIM});
    Tensor<float> dK({BATCH, HEADS, SEQ_KV, HEAD_DIM});
    Tensor<float> dV({BATCH, HEADS, SEQ_KV, HEAD_DIM_V});
    CpuFpReferenceSdpa::backward(q, k, v, o, dO, dQ, dK, dV, customScale);

    constexpr double EPS = 1e-3;
    constexpr double REL_TOL = 1e-2;
    constexpr double ABS_TOL = 1e-4;

    auto fwdLoss = [&]() -> double {
        CpuFpReferenceSdpa::forward(q, k, v, o, customScale);
        return computeDotProductLoss(dO, o);
    };

    {
        Tensor<float> dQNum({BATCH, HEADS, SEQ_Q, HEAD_DIM});
        numericalGradient(q, dQNum, EPS, fwdLoss);
        auto result = compareGradients(dQ, dQNum, REL_TOL, ABS_TOL);
        EXPECT_EQ(result.failCount, 0)
            << "dQ grad check (scale=0.3) failed: maxRelErr=" << result.maxRelErr
            << ", maxAbsErr=" << result.maxAbsErr;
    }

    {
        Tensor<float> dKNum({BATCH, HEADS, SEQ_KV, HEAD_DIM});
        numericalGradient(k, dKNum, EPS, fwdLoss);
        auto result = compareGradients(dK, dKNum, REL_TOL, ABS_TOL);
        EXPECT_EQ(result.failCount, 0)
            << "dK grad check (scale=0.3) failed: maxRelErr=" << result.maxRelErr
            << ", maxAbsErr=" << result.maxAbsErr;
    }

    {
        Tensor<float> dVNum({BATCH, HEADS, SEQ_KV, HEAD_DIM_V});
        numericalGradient(v, dVNum, EPS, fwdLoss);
        auto result = compareGradients(dV, dVNum, REL_TOL, ABS_TOL);
        EXPECT_EQ(result.failCount, 0)
            << "dV grad check (scale=0.3) failed: maxRelErr=" << result.maxRelErr
            << ", maxAbsErr=" << result.maxAbsErr;
    }
}

TEST(TestCpuFpReferenceSdpaGradCheck, AdditiveMask)
{
    // MHA with an additive attention mask
    // [B=1, H=1, Sq=3, Skv=3, D=4, Dv=4], mask [Sq, Skv]
    constexpr int64_t BATCH = 1;
    constexpr int64_t HEADS = 1;
    constexpr int64_t SEQ_Q = 3;
    constexpr int64_t SEQ_KV = 3;
    constexpr int64_t HEAD_DIM = 4;
    constexpr int64_t HEAD_DIM_V = 4;

    Tensor<float> q({BATCH, HEADS, SEQ_Q, HEAD_DIM});
    Tensor<float> k({BATCH, HEADS, SEQ_KV, HEAD_DIM});
    Tensor<float> v({BATCH, HEADS, SEQ_KV, HEAD_DIM_V});
    Tensor<float> o({BATCH, HEADS, SEQ_Q, HEAD_DIM_V});
    Tensor<float> dO({BATCH, HEADS, SEQ_Q, HEAD_DIM_V});
    Tensor<float> mask({SEQ_Q, SEQ_KV});

    q.fillWithRandomValues(-0.5f, 0.5f, 5000);
    k.fillWithRandomValues(-0.5f, 0.5f, 5001);
    v.fillWithRandomValues(-0.5f, 0.5f, 5002);
    dO.fillWithRandomValues(-0.5f, 0.5f, 5003);

    // Partially suppress some positions
    mask.fillWithValue(0.0f);
    mask.setHostValue(-1e4f, 0, 2); // Block sq=0 attending to skv=2
    mask.setHostValue(-1e4f, 1, 0); // Block sq=1 attending to skv=0

    CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, &mask);

    Tensor<float> dQ({BATCH, HEADS, SEQ_Q, HEAD_DIM});
    Tensor<float> dK({BATCH, HEADS, SEQ_KV, HEAD_DIM});
    Tensor<float> dV({BATCH, HEADS, SEQ_KV, HEAD_DIM_V});
    CpuFpReferenceSdpa::backward(q, k, v, o, dO, dQ, dK, dV, std::nullopt, nullptr, &mask, false);

    constexpr double EPS = 1e-3;
    constexpr double REL_TOL = 1e-2;
    constexpr double ABS_TOL = 1e-4;

    auto fwdLoss = [&]() -> double {
        CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, &mask);
        return computeDotProductLoss(dO, o);
    };

    {
        Tensor<float> dQNum({BATCH, HEADS, SEQ_Q, HEAD_DIM});
        numericalGradient(q, dQNum, EPS, fwdLoss);
        auto result = compareGradients(dQ, dQNum, REL_TOL, ABS_TOL);
        EXPECT_EQ(result.failCount, 0)
            << "dQ grad check (additive mask) failed: maxRelErr=" << result.maxRelErr
            << ", maxAbsErr=" << result.maxAbsErr;
    }

    {
        Tensor<float> dKNum({BATCH, HEADS, SEQ_KV, HEAD_DIM});
        numericalGradient(k, dKNum, EPS, fwdLoss);
        auto result = compareGradients(dK, dKNum, REL_TOL, ABS_TOL);
        EXPECT_EQ(result.failCount, 0)
            << "dK grad check (additive mask) failed: maxRelErr=" << result.maxRelErr
            << ", maxAbsErr=" << result.maxAbsErr;
    }

    {
        Tensor<float> dVNum({BATCH, HEADS, SEQ_KV, HEAD_DIM_V});
        numericalGradient(v, dVNum, EPS, fwdLoss);
        auto result = compareGradients(dV, dVNum, REL_TOL, ABS_TOL);
        EXPECT_EQ(result.failCount, 0)
            << "dV grad check (additive mask) failed: maxRelErr=" << result.maxRelErr
            << ", maxAbsErr=" << result.maxAbsErr;
    }
}

TEST(TestCpuFpReferenceSdpaGradCheck, GQACausalMask)
{
    // Combined: GQA + causal mask
    // [B=1, H_q=4, H_kv=2, Sq=4, Skv=4, D=4, Dv=4]
    constexpr int64_t BATCH = 1;
    constexpr int64_t HEADS_Q = 4;
    constexpr int64_t HEADS_KV = 2;
    constexpr int64_t SEQ_Q = 4;
    constexpr int64_t SEQ_KV = 4;
    constexpr int64_t HEAD_DIM = 4;
    constexpr int64_t HEAD_DIM_V = 4;

    Tensor<float> q({BATCH, HEADS_Q, SEQ_Q, HEAD_DIM});
    Tensor<float> k({BATCH, HEADS_KV, SEQ_KV, HEAD_DIM});
    Tensor<float> v({BATCH, HEADS_KV, SEQ_KV, HEAD_DIM_V});
    Tensor<float> o({BATCH, HEADS_Q, SEQ_Q, HEAD_DIM_V});
    Tensor<float> dO({BATCH, HEADS_Q, SEQ_Q, HEAD_DIM_V});

    q.fillWithRandomValues(-0.5f, 0.5f, 6000);
    k.fillWithRandomValues(-0.5f, 0.5f, 6001);
    v.fillWithRandomValues(-0.5f, 0.5f, 6002);
    dO.fillWithRandomValues(-0.5f, 0.5f, 6003);

    const TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, noMask, /*causalMask=*/true);

    Tensor<float> dQ({BATCH, HEADS_Q, SEQ_Q, HEAD_DIM});
    Tensor<float> dK({BATCH, HEADS_KV, SEQ_KV, HEAD_DIM});
    Tensor<float> dV({BATCH, HEADS_KV, SEQ_KV, HEAD_DIM_V});
    CpuFpReferenceSdpa::backward(
        q, k, v, o, dO, dQ, dK, dV, std::nullopt, nullptr, noMask, /*causalMask=*/true);

    constexpr double EPS = 1e-3;
    constexpr double REL_TOL = 1e-2;
    constexpr double ABS_TOL = 1e-4;

    auto fwdLoss = [&]() -> double {
        CpuFpReferenceSdpa::forward(q, k, v, o, std::nullopt, noMask, /*causalMask=*/true);
        return computeDotProductLoss(dO, o);
    };

    {
        Tensor<float> dQNum({BATCH, HEADS_Q, SEQ_Q, HEAD_DIM});
        numericalGradient(q, dQNum, EPS, fwdLoss);
        auto result = compareGradients(dQ, dQNum, REL_TOL, ABS_TOL);
        EXPECT_EQ(result.failCount, 0)
            << "dQ grad check (GQA+causal) failed: maxRelErr=" << result.maxRelErr
            << ", maxAbsErr=" << result.maxAbsErr;
    }

    {
        Tensor<float> dKNum({BATCH, HEADS_KV, SEQ_KV, HEAD_DIM});
        numericalGradient(k, dKNum, EPS, fwdLoss);
        auto result = compareGradients(dK, dKNum, REL_TOL, ABS_TOL);
        EXPECT_EQ(result.failCount, 0)
            << "dK grad check (GQA+causal) failed: maxRelErr=" << result.maxRelErr
            << ", maxAbsErr=" << result.maxAbsErr;
    }

    {
        Tensor<float> dVNum({BATCH, HEADS_KV, SEQ_KV, HEAD_DIM_V});
        numericalGradient(v, dVNum, EPS, fwdLoss);
        auto result = compareGradients(dV, dVNum, REL_TOL, ABS_TOL);
        EXPECT_EQ(result.failCount, 0)
            << "dV grad check (GQA+causal) failed: maxRelErr=" << result.maxRelErr
            << ", maxAbsErr=" << result.maxAbsErr;
    }
}

TEST(TestCpuFpReferenceSdpaBwdBf16, BackwardBasic)
{
    // Test backward pass with BFloat16 data type
    // [B=1, H=1, Sq=4, Skv=4, D=8, Dv=8]
    Tensor<bfloat16> q({1, 1, 4, 8});
    Tensor<bfloat16> k({1, 1, 4, 8});
    Tensor<bfloat16> v({1, 1, 4, 8});
    Tensor<bfloat16> o({1, 1, 4, 8});
    Tensor<bfloat16> dO({1, 1, 4, 8});
    Tensor<bfloat16> dQ({1, 1, 4, 8});
    Tensor<bfloat16> dK({1, 1, 4, 8});
    Tensor<bfloat16> dV({1, 1, 4, 8});

    q.fillWithRandomValues(bfloat16(-1.0f), bfloat16(1.0f), 500);
    k.fillWithRandomValues(bfloat16(-1.0f), bfloat16(1.0f), 501);
    v.fillWithRandomValues(bfloat16(-1.0f), bfloat16(1.0f), 502);
    dO.fillWithRandomValues(bfloat16(-1.0f), bfloat16(1.0f), 503);

    // Forward pass
    CpuFpReferenceSdpa::forward(q, k, v, o);

    // Backward pass
    CpuFpReferenceSdpa::backward(q, k, v, o, dO, dQ, dK, dV);

    // Check gradients are finite (BFloat16 has limited precision)
    for(int sq = 0; sq < 4; ++sq)
    {
        for(int d = 0; d < 8; ++d)
        {
            const float dqVal = static_cast<float>(dQ.getHostValue(0, 0, sq, d));
            EXPECT_TRUE(std::isfinite(dqVal)) << "dQ not finite at sq=" << sq << ", d=" << d;
            const float dkVal = static_cast<float>(dK.getHostValue(0, 0, sq, d));
            EXPECT_TRUE(std::isfinite(dkVal)) << "dK not finite at sq=" << sq << ", d=" << d;
            const float dvVal = static_cast<float>(dV.getHostValue(0, 0, sq, d));
            EXPECT_TRUE(std::isfinite(dvVal)) << "dV not finite at sq=" << sq << ", d=" << d;
        }
    }
}

TEST(TestCpuFpReferenceSdpaBwdFp32, BackwardGqaDifferentKVHeads)
{
    // Test backward with independent K and V head counts: H_q=4, H_k=2, H_v=1
    // Mirrors the forward GqaDifferentKVHeads test.
    // [B=1, H_q=4, H_k=2, H_v=1, Sq=3, Skv=3, D=4, Dv=4]
    Tensor<float> q({1, 4, 3, 4});
    Tensor<float> k({1, 2, 3, 4});
    Tensor<float> v({1, 1, 3, 4});
    Tensor<float> o({1, 4, 3, 4});
    Tensor<float> dO({1, 4, 3, 4});
    Tensor<float> dQ({1, 4, 3, 4});
    Tensor<float> dK({1, 2, 3, 4});
    Tensor<float> dV({1, 1, 3, 4});

    q.fillWithRandomValues(-1.0f, 1.0f, 600);
    k.fillWithRandomValues(-1.0f, 1.0f, 601);
    v.fillWithRandomValues(-1.0f, 1.0f, 602);
    dO.fillWithRandomValues(-1.0f, 1.0f, 603);

    // Forward pass
    CpuFpReferenceSdpa::forward(q, k, v, o);

    // Backward pass with different K and V head counts
    CpuFpReferenceSdpa::backward(q, k, v, o, dO, dQ, dK, dV);

    // Check dQ gradients are finite (4 Q heads)
    for(int h = 0; h < 4; ++h)
    {
        for(int sq = 0; sq < 3; ++sq)
        {
            for(int d = 0; d < 4; ++d)
            {
                EXPECT_TRUE(std::isfinite(dQ.getHostValue(0, h, sq, d)))
                    << "dQ not finite at h=" << h << ", sq=" << sq << ", d=" << d;
            }
        }
    }

    // Check dK gradients are finite (2 K heads)
    for(int h = 0; h < 2; ++h)
    {
        for(int sq = 0; sq < 3; ++sq)
        {
            for(int d = 0; d < 4; ++d)
            {
                EXPECT_TRUE(std::isfinite(dK.getHostValue(0, h, sq, d)))
                    << "dK not finite at h=" << h << ", sq=" << sq << ", d=" << d;
            }
        }
    }

    // Check dV gradients are finite (1 V head)
    for(int sq = 0; sq < 3; ++sq)
    {
        for(int d = 0; d < 4; ++d)
        {
            EXPECT_TRUE(std::isfinite(dV.getHostValue(0, 0, sq, d)))
                << "dV not finite at sq=" << sq << ", d=" << d;
        }
    }

    // dK should accumulate from 2 Q heads per K head
    float dkSum = 0.0f;
    for(int h = 0; h < 2; ++h)
    {
        for(int sq = 0; sq < 3; ++sq)
        {
            for(int d = 0; d < 4; ++d)
            {
                dkSum += std::abs(dK.getHostValue(0, h, sq, d));
            }
        }
    }
    EXPECT_GT(dkSum, 0.0f) << "dK should have non-zero gradients from GQA accumulation";

    // dV should accumulate from all 4 Q heads into the single V head
    float dvSum = 0.0f;
    for(int sq = 0; sq < 3; ++sq)
    {
        for(int d = 0; d < 4; ++d)
        {
            dvSum += std::abs(dV.getHostValue(0, 0, sq, d));
        }
    }
    EXPECT_GT(dvSum, 0.0f) << "dV should have non-zero gradients from GQA accumulation";
}

// ---------------------------------------------------------------------------
// Backward validation tests (error cases)
// ---------------------------------------------------------------------------

TEST(TestCpuFpReferenceSdpaBwd, BackwardWrongRankQ)
{
    const Tensor<float> q({1, 2, 4}); // rank-3 instead of rank-4
    const Tensor<float> k({1, 1, 4, 8});
    const Tensor<float> v({1, 1, 4, 8});
    const Tensor<float> o({1, 1, 4, 8});
    const Tensor<float> dO({1, 1, 4, 8});
    Tensor<float> dQ({1, 2, 4});
    Tensor<float> dK({1, 1, 4, 8});
    Tensor<float> dV({1, 1, 4, 8});

    EXPECT_THROW(CpuFpReferenceSdpa::backward(q, k, v, o, dO, dQ, dK, dV), std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaBwd, BackwardWrongRankDK)
{
    const Tensor<float> q({1, 1, 4, 8});
    const Tensor<float> k({1, 1, 4, 8});
    const Tensor<float> v({1, 1, 4, 8});
    const Tensor<float> o({1, 1, 4, 8});
    const Tensor<float> dO({1, 1, 4, 8});
    Tensor<float> dQ({1, 1, 4, 8});
    Tensor<float> dK({1, 4, 8}); // rank-3 instead of rank-4
    Tensor<float> dV({1, 1, 4, 8});

    EXPECT_THROW(CpuFpReferenceSdpa::backward(q, k, v, o, dO, dQ, dK, dV), std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaBwd, BackwardBatchMismatch)
{
    const Tensor<float> q({2, 1, 4, 8});
    const Tensor<float> k({2, 1, 4, 8});
    const Tensor<float> v({2, 1, 4, 8});
    const Tensor<float> o({2, 1, 4, 8});
    const Tensor<float> dO({2, 1, 4, 8});
    Tensor<float> dQ({2, 1, 4, 8});
    Tensor<float> dK({1, 1, 4, 8}); // batch=1 instead of batch=2
    Tensor<float> dV({2, 1, 4, 8});

    EXPECT_THROW(CpuFpReferenceSdpa::backward(q, k, v, o, dO, dQ, dK, dV), std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaBwd, BackwardHeadDimMismatch)
{
    const Tensor<float> q({1, 1, 4, 8});
    const Tensor<float> k({1, 1, 4, 16}); // D=16 instead of D=8
    const Tensor<float> v({1, 1, 4, 8});
    const Tensor<float> o({1, 1, 4, 8});
    const Tensor<float> dO({1, 1, 4, 8});
    Tensor<float> dQ({1, 1, 4, 8});
    Tensor<float> dK({1, 1, 4, 16});
    Tensor<float> dV({1, 1, 4, 8});

    EXPECT_THROW(CpuFpReferenceSdpa::backward(q, k, v, o, dO, dQ, dK, dV), std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaBwd, BackwardGqaNotDivisibleK)
{
    // H_q=3 is not divisible by H_k=2
    const Tensor<float> q({1, 3, 4, 8});
    const Tensor<float> k({1, 2, 4, 8});
    const Tensor<float> v({1, 1, 4, 8});
    const Tensor<float> o({1, 3, 4, 8});
    const Tensor<float> dO({1, 3, 4, 8});
    Tensor<float> dQ({1, 3, 4, 8});
    Tensor<float> dK({1, 2, 4, 8});
    Tensor<float> dV({1, 1, 4, 8});

    EXPECT_THROW(CpuFpReferenceSdpa::backward(q, k, v, o, dO, dQ, dK, dV), std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaBwd, BackwardGqaNotDivisibleV)
{
    // H_q=4 is not divisible by H_v=3
    const Tensor<float> q({1, 4, 4, 8});
    const Tensor<float> k({1, 2, 4, 8});
    const Tensor<float> v({1, 3, 4, 8});
    const Tensor<float> o({1, 4, 4, 8});
    const Tensor<float> dO({1, 4, 4, 8});
    Tensor<float> dQ({1, 4, 4, 8});
    Tensor<float> dK({1, 2, 4, 8});
    Tensor<float> dV({1, 3, 4, 8});

    EXPECT_THROW(CpuFpReferenceSdpa::backward(q, k, v, o, dO, dQ, dK, dV), std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaBwd, BackwardLseWrongRank)
{
    Tensor<float> q({1, 1, 4, 8});
    Tensor<float> k({1, 1, 4, 8});
    Tensor<float> v({1, 1, 4, 8});
    Tensor<float> o({1, 1, 4, 8});
    Tensor<float> dO({1, 1, 4, 8});
    Tensor<float> dQ({1, 1, 4, 8});
    Tensor<float> dK({1, 1, 4, 8});
    Tensor<float> dV({1, 1, 4, 8});
    Tensor<float> lseWrong({1, 1, 4, 1}); // rank-4 instead of rank-3

    q.fillWithRandomValues(-1.0f, 1.0f, 700);
    k.fillWithRandomValues(-1.0f, 1.0f, 701);
    v.fillWithRandomValues(-1.0f, 1.0f, 702);
    dO.fillWithValue(1.0f);

    CpuFpReferenceSdpa::forward(q, k, v, o);

    EXPECT_THROW(CpuFpReferenceSdpa::backward(q, k, v, o, dO, dQ, dK, dV, std::nullopt, &lseWrong),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaBwd, BackwardLseWrongShape)
{
    Tensor<float> q({2, 4, 16, 8});
    Tensor<float> k({2, 4, 16, 8});
    Tensor<float> v({2, 4, 16, 8});
    Tensor<float> o({2, 4, 16, 8});
    Tensor<float> dO({2, 4, 16, 8});
    Tensor<float> dQ({2, 4, 16, 8});
    Tensor<float> dK({2, 4, 16, 8});
    Tensor<float> dV({2, 4, 16, 8});
    Tensor<float> lseWrong({2, 4, 8}); // Sq=8 instead of Sq=16

    q.fillWithRandomValues(-1.0f, 1.0f, 710);
    k.fillWithRandomValues(-1.0f, 1.0f, 711);
    v.fillWithRandomValues(-1.0f, 1.0f, 712);
    dO.fillWithValue(1.0f);

    CpuFpReferenceSdpa::forward(q, k, v, o);

    EXPECT_THROW(CpuFpReferenceSdpa::backward(q, k, v, o, dO, dQ, dK, dV, std::nullopt, &lseWrong),
                 std::invalid_argument);
}
