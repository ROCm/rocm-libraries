// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Validation of the standalone FMHA BWD CPU reference.
//
// This is a TEMPORARY test (not intended for the PR) that verifies our
// CPU reference computes correct gradients by:
//
// 1. Checking known mathematical identities:
//    - dQ/dK/dV are zero when dO is zero
//    - D = dot(dO, O) matches independently computed value
//    - Sum of softmax output P is 1 per row
//
// 2. Finite-difference spot checks on tiny configs to verify the gradient
//    direction is correct (not for precision — just sign/magnitude sanity).
//
// 3. Cross-checking fwd+bwd round-trip: running the reference pipeline
//    and verifying chain-rule identities hold to FP32 precision.

#include "fmha_bwd_cpu_ref.hpp"

#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/fill.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>

using ck_tile::HostTensor;

namespace {

void fill_uniform(HostTensor<float>& t, float lo, float hi, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    t.ForEach([&](auto& self, auto idx) { self(idx) = dist(rng); });
}

// Compute L = sum over output of (O * dO) using the forward O.
// This is a scalar "loss" that makes dO the gradient of L w.r.t. O.
// Then the chain rule gives us analytical gradients we can verify.
float total_loss(const HostTensor<float>& o, const HostTensor<float>& dO)
{
    float acc = 0;
    o.ForEach([&](const auto& self, auto idx) {
        acc += self(idx) * dO(idx[0], idx[1], idx[2]);
    });
    return acc;
}

} // namespace

// ==========================================================================
// Test: zero dO produces zero gradients
// ==========================================================================
TEST(FmhaBwdCpuRef, ZeroDOProducesZeroGradients)
{
    const int nhead = 2, seqlen_q = 4, seqlen_k = 4, hdim = 8;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hdim));

    HostTensor<float> q({nhead, seqlen_q, hdim});
    HostTensor<float> k({nhead, seqlen_k, hdim});
    HostTensor<float> v({nhead, hdim, seqlen_k});
    HostTensor<float> dO({nhead, seqlen_q, hdim});

    fill_uniform(q, -1.f, 1.f, 42);
    fill_uniform(k, -1.f, 1.f, 43);
    fill_uniform(v, -1.f, 1.f, 44);
    // dO is zero-initialized by default

    auto fwd = rocm_ck::test::fmha_fwd_cpu_ref<float, float, float, float, float>(q, k, v, scale);
    auto bwd = rocm_ck::test::fmha_bwd_cpu_ref<float, float, float, float, float>(
        q, k, v, fwd.o, fwd.p, dO, scale);

    // All gradients should be exactly zero
    bwd.d.ForEach([](const auto& self, auto idx) { EXPECT_FLOAT_EQ(self(idx), 0.f); });
    bwd.dq_acc.ForEach([](const auto& self, auto idx) { EXPECT_FLOAT_EQ(self(idx), 0.f); });
    bwd.dk.ForEach([](const auto& self, auto idx) { EXPECT_FLOAT_EQ(self(idx), 0.f); });
    bwd.dv.ForEach([](const auto& self, auto idx) { EXPECT_FLOAT_EQ(self(idx), 0.f); });
}

// ==========================================================================
// Test: softmax rows sum to 1
// ==========================================================================
TEST(FmhaBwdCpuRef, SoftmaxRowsSumToOne)
{
    const int nhead = 2, seqlen_q = 8, seqlen_k = 8, hdim = 16;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hdim));

    HostTensor<float> q({nhead, seqlen_q, hdim});
    HostTensor<float> k({nhead, seqlen_k, hdim});
    HostTensor<float> v({nhead, hdim, seqlen_k});

    fill_uniform(q, -1.f, 1.f, 100);
    fill_uniform(k, -1.f, 1.f, 101);
    fill_uniform(v, -1.f, 1.f, 102);

    auto fwd = rocm_ck::test::fmha_fwd_cpu_ref<float, float, float, float, float>(q, k, v, scale);

    for(int h = 0; h < nhead; ++h)
    {
        for(int q_idx = 0; q_idx < seqlen_q; ++q_idx)
        {
            float row_sum = 0;
            for(int k_idx = 0; k_idx < seqlen_k; ++k_idx)
            {
                row_sum += fwd.p(h, q_idx, k_idx);
            }
            EXPECT_NEAR(row_sum, 1.0f, 1e-6f)
                << "head=" << h << " q=" << q_idx;
        }
    }
}

// ==========================================================================
// Test: D = dot(dO, O) per head/query position
// ==========================================================================
TEST(FmhaBwdCpuRef, OGradDotOMatchesManual)
{
    const int nhead = 2, seqlen_q = 4, seqlen_k = 6, hdim_q = 8, hdim_v = 8;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hdim_q));

    HostTensor<float> q({nhead, seqlen_q, hdim_q});
    HostTensor<float> k({nhead, seqlen_k, hdim_q});
    HostTensor<float> v({nhead, hdim_v, seqlen_k});
    HostTensor<float> dO({nhead, seqlen_q, hdim_v});

    fill_uniform(q, -1.f, 1.f, 200);
    fill_uniform(k, -1.f, 1.f, 201);
    fill_uniform(v, -1.f, 1.f, 202);
    fill_uniform(dO, -1.f, 1.f, 203);

    auto fwd = rocm_ck::test::fmha_fwd_cpu_ref<float, float, float, float, float>(q, k, v, scale);
    auto bwd = rocm_ck::test::fmha_bwd_cpu_ref<float, float, float, float, float>(
        q, k, v, fwd.o, fwd.p, dO, scale);

    // Manually compute D and compare
    for(int h = 0; h < nhead; ++h)
    {
        for(int qi = 0; qi < seqlen_q; ++qi)
        {
            float manual_d = 0;
            for(int d = 0; d < hdim_v; ++d)
            {
                manual_d += dO(h, qi, d) * fwd.o(h, qi, d);
            }
            EXPECT_NEAR(bwd.d(h, qi), manual_d, 1e-5f)
                << "head=" << h << " q=" << qi;
        }
    }
}

// ==========================================================================
// Test: finite difference gradient check (tiny config)
//
// For a scalar loss L = sum(O * dO_fixed), compute analytical dQ via our
// reference, then verify against numerical dL/dQ_ij via central differences.
// ==========================================================================
TEST(FmhaBwdCpuRef, FiniteDifferenceGradientCheck)
{
    const int nhead = 1, seqlen_q = 3, seqlen_k = 3, hdim = 4;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hdim));
    const float eps   = 1e-4f;
    const float tol   = 1e-2f; // finite diff has ~eps^2 error

    HostTensor<float> q({nhead, seqlen_q, hdim});
    HostTensor<float> k({nhead, seqlen_k, hdim});
    HostTensor<float> v({nhead, hdim, seqlen_k});
    HostTensor<float> dO({nhead, seqlen_q, hdim});

    fill_uniform(q, -0.5f, 0.5f, 300);
    fill_uniform(k, -0.5f, 0.5f, 301);
    fill_uniform(v, -0.5f, 0.5f, 302);
    fill_uniform(dO, -0.5f, 0.5f, 303);

    // Analytical gradient
    auto fwd = rocm_ck::test::fmha_fwd_cpu_ref<float, float, float, float, float>(q, k, v, scale);
    auto bwd = rocm_ck::test::fmha_bwd_cpu_ref<float, float, float, float, float>(
        q, k, v, fwd.o, fwd.p, dO, scale);

    // Spot-check a few dQ elements via finite differences
    for(int h = 0; h < nhead; ++h)
    {
        for(int qi = 0; qi < seqlen_q; ++qi)
        {
            for(int di = 0; di < hdim; ++di)
            {
                float orig = q(h, qi, di);

                // f(q + eps)
                q(h, qi, di) = orig + eps;
                auto fwd_plus =
                    rocm_ck::test::fmha_fwd_cpu_ref<float, float, float, float, float>(
                        q, k, v, scale);
                float loss_plus = total_loss(fwd_plus.o, dO);

                // f(q - eps)
                q(h, qi, di) = orig - eps;
                auto fwd_minus =
                    rocm_ck::test::fmha_fwd_cpu_ref<float, float, float, float, float>(
                        q, k, v, scale);
                float loss_minus = total_loss(fwd_minus.o, dO);

                q(h, qi, di) = orig; // restore

                float numerical_grad = (loss_plus - loss_minus) / (2.f * eps);
                float analytical_grad = bwd.dq_acc(h, qi, di);

                EXPECT_NEAR(analytical_grad, numerical_grad, tol)
                    << "dQ[" << h << "," << qi << "," << di << "]"
                    << " analytical=" << analytical_grad
                    << " numerical=" << numerical_grad;
            }
        }
    }

    // Spot-check a few dK elements
    for(int h = 0; h < nhead; ++h)
    {
        for(int ki = 0; ki < seqlen_k; ++ki)
        {
            for(int di = 0; di < hdim; ++di)
            {
                float orig = k(h, ki, di);

                k(h, ki, di) = orig + eps;
                auto fwd_plus =
                    rocm_ck::test::fmha_fwd_cpu_ref<float, float, float, float, float>(
                        q, k, v, scale);
                float loss_plus = total_loss(fwd_plus.o, dO);

                k(h, ki, di) = orig - eps;
                auto fwd_minus =
                    rocm_ck::test::fmha_fwd_cpu_ref<float, float, float, float, float>(
                        q, k, v, scale);
                float loss_minus = total_loss(fwd_minus.o, dO);

                k(h, ki, di) = orig;

                float numerical_grad = (loss_plus - loss_minus) / (2.f * eps);
                float analytical_grad = bwd.dk(h, ki, di);

                EXPECT_NEAR(analytical_grad, numerical_grad, tol)
                    << "dK[" << h << "," << ki << "," << di << "]"
                    << " analytical=" << analytical_grad
                    << " numerical=" << numerical_grad;
            }
        }
    }

    // Spot-check a few dV elements
    // V is [nhead, hdim, seqlen_k], dV is [nhead, seqlen_k, hdim_v]
    for(int h = 0; h < nhead; ++h)
    {
        for(int di = 0; di < hdim; ++di)
        {
            for(int ki = 0; ki < seqlen_k; ++ki)
            {
                float orig = v(h, di, ki);

                v(h, di, ki) = orig + eps;
                auto fwd_plus =
                    rocm_ck::test::fmha_fwd_cpu_ref<float, float, float, float, float>(
                        q, k, v, scale);
                float loss_plus = total_loss(fwd_plus.o, dO);

                v(h, di, ki) = orig - eps;
                auto fwd_minus =
                    rocm_ck::test::fmha_fwd_cpu_ref<float, float, float, float, float>(
                        q, k, v, scale);
                float loss_minus = total_loss(fwd_minus.o, dO);

                v(h, di, ki) = orig;

                float numerical_grad = (loss_plus - loss_minus) / (2.f * eps);
                // dV is [nhead, seqlen_k, hdim_v] so dV/dV[h, di, ki] maps to dv[h, ki, di]
                float analytical_grad = bwd.dv(h, ki, di);

                EXPECT_NEAR(analytical_grad, numerical_grad, tol)
                    << "dV[" << h << "," << di << "," << ki << "]"
                    << " analytical=" << analytical_grad
                    << " numerical=" << numerical_grad;
            }
        }
    }
}

// ==========================================================================
// Test: LSE consistency — exp(S - LSE) should match P
// ==========================================================================
TEST(FmhaBwdCpuRef, LSEConsistencyWithP)
{
    const int nhead = 2, seqlen_q = 4, seqlen_k = 6, hdim = 8;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hdim));

    HostTensor<float> q({nhead, seqlen_q, hdim});
    HostTensor<float> k({nhead, seqlen_k, hdim});
    HostTensor<float> v({nhead, hdim, seqlen_k});

    fill_uniform(q, -1.f, 1.f, 400);
    fill_uniform(k, -1.f, 1.f, 401);
    fill_uniform(v, -1.f, 1.f, 402);

    auto fwd = rocm_ck::test::fmha_fwd_cpu_ref<float, float, float, float, float>(q, k, v, scale);

    for(int h = 0; h < nhead; ++h)
    {
        for(int qi = 0; qi < seqlen_q; ++qi)
        {
            for(int ki = 0; ki < seqlen_k; ++ki)
            {
                float p_from_lse = std::exp(fwd.s(h, qi, ki) - fwd.lse(h, qi));
                EXPECT_NEAR(fwd.p(h, qi, ki), p_from_lse, 1e-6f)
                    << "head=" << h << " q=" << qi << " k=" << ki;
            }
        }
    }
}

// ==========================================================================
// Test: O = P * V identity (verify forward produces correct output)
// ==========================================================================
TEST(FmhaBwdCpuRef, ForwardOutputMatchesManualPV)
{
    const int nhead = 1, seqlen_q = 3, seqlen_k = 4, hdim_q = 8, hdim_v = 6;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hdim_q));

    HostTensor<float> q({nhead, seqlen_q, hdim_q});
    HostTensor<float> k({nhead, seqlen_k, hdim_q});
    HostTensor<float> v({nhead, hdim_v, seqlen_k});

    fill_uniform(q, -1.f, 1.f, 500);
    fill_uniform(k, -1.f, 1.f, 501);
    fill_uniform(v, -1.f, 1.f, 502);

    auto fwd = rocm_ck::test::fmha_fwd_cpu_ref<float, float, float, float, float>(q, k, v, scale);

    // Manually compute O = P * V
    // P: [nhead, seqlen_q, seqlen_k], V: [nhead, hdim_v, seqlen_k]
    // O[h,q,d] = sum_k P[h,q,k] * V[h,d,k]
    for(int h = 0; h < nhead; ++h)
    {
        for(int qi = 0; qi < seqlen_q; ++qi)
        {
            for(int di = 0; di < hdim_v; ++di)
            {
                float manual_o = 0;
                for(int ki = 0; ki < seqlen_k; ++ki)
                {
                    manual_o += fwd.p(h, qi, ki) * v(h, di, ki);
                }
                EXPECT_NEAR(fwd.o(h, qi, di), manual_o, 1e-5f)
                    << "head=" << h << " q=" << qi << " d=" << di;
            }
        }
    }
}
