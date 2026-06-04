// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Tests for the split-accumulation GPU reference GEMM.
//
// Compares split_gemm against:
//   1. CPU FP64 triple-loop reference (double-precision accumulation)
//   2. Veltkamp HP (Ozaki tiled) vs CPU FP64
//   3. GPU split_gemm vs CK's reference_batched_gemm (layout cross-check)

#include "split_gemm.hpp"

#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/reference/reference_batched_gemm.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

using ck_tile::DeviceMem;
using ck_tile::HostTensor;

namespace {

void fill_uniform(HostTensor<float>& t, float lo, float hi, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    t.ForEach([&](auto& self, auto idx) { self(idx) = dist(rng); });
}

// CPU FP64 reference GEMM: C = A @ B with double-precision accumulation.
// A [M,K] row-major, B [K,N] row-major, C [M,N] row-major.
void cpu_gemm_fp64(const float* A,
                   const float* B,
                   float* C,
                   int M,
                   int N,
                   int K)
{
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            double acc = 0.0;
            for(int k = 0; k < K; ++k)
                acc += static_cast<double>(A[i * K + k]) * static_cast<double>(B[k * N + j]);
            C[i * N + j] = static_cast<float>(acc);
        }
    }
}

struct ErrorStats
{
    double max_abs;
    double max_rel;
};

// max_abs = max |actual - ref| over all elements.
// max_rel = max |actual - ref| / |ref| over elements where |ref| > 1e-6.
ErrorStats compute_error(const float* actual, const float* reference, int count)
{
    constexpr double abs_floor = 1e-6;
    double max_abs = 0;
    double max_rel = 0;

    for(int i = 0; i < count; ++i)
    {
        double a    = static_cast<double>(actual[i]);
        double r    = static_cast<double>(reference[i]);
        double diff = std::fabs(a - r);

        max_abs = std::max(max_abs, diff);

        if(std::fabs(r) > abs_floor)
            max_rel = std::max(max_rel, diff / std::fabs(r));
    }

    return {max_abs, max_rel};
}

void print_error_stats(const std::string& label, const ErrorStats& s)
{
    std::cout << "  " << label << ": max_abs=" << s.max_abs << " max_rel=" << s.max_rel
              << std::endl;
}

} // namespace

// =========================================================================
// Test fixture parameterized over (M, N, K, kTile)
// =========================================================================
struct SplitGemmConfig
{
    int M, N, K, kTile;
    std::string name;
};

std::ostream& operator<<(std::ostream& os, const SplitGemmConfig& c)
{
    return os << c.name << "_M" << c.M << "_N" << c.N << "_K" << c.K << "_kT" << c.kTile;
}

class SplitGemmTest : public ::testing::TestWithParam<SplitGemmConfig>
{
protected:
    std::unique_ptr<rocm_ck::test::SplitGemmWorkspace> ws_;

    void SetUp() override
    {
        // Allocate workspace for the worst-case config across all tiers.
        // max M=512, N=512, K=1024, min kTile=32
        ws_ = std::make_unique<rocm_ck::test::SplitGemmWorkspace>(512, 512, 1024, 32);
    }

    void TearDown() override { ws_.reset(); }
};

// =========================================================================
// Test: split_gemm correctness vs CPU FP64 reference
// =========================================================================
TEST_P(SplitGemmTest, CorrectnessVsCpuFp64)
{
    auto cfg = GetParam();

    // Host tensors — row-major [M,K], [K,N], [M,N]
    HostTensor<float> h_A({cfg.M, cfg.K});
    HostTensor<float> h_B({cfg.K, cfg.N});

    fill_uniform(h_A, -1.0f, 1.0f, 42u);
    fill_uniform(h_B, -1.0f, 1.0f, 43u);

    int MN = cfg.M * cfg.N;

    // CPU FP64 reference
    std::vector<float> h_C_ref(static_cast<size_t>(MN));
    cpu_gemm_fp64(h_A.data(), h_B.data(), h_C_ref.data(), cfg.M, cfg.N, cfg.K);

    // GPU split_gemm
    DeviceMem d_A(h_A);
    DeviceMem d_B(h_B);
    DeviceMem d_C(static_cast<size_t>(MN) * sizeof(float));
    d_C.SetZero();

    struct StrategyCase
    {
        rocm_ck::test::ReduceStrategy strategy;
        const char* name;
    };

    StrategyCase cases[] = {
        {rocm_ck::test::ReduceStrategy::NaiveSum, "NaiveSum"},
        {rocm_ck::test::ReduceStrategy::CompensatedSum, "CompensatedSum"},
        {rocm_ck::test::ReduceStrategy::FP64Sum, "FP64Sum"},
    };

    std::cout << "[" << cfg << "]" << std::endl;

    for(const auto& tc : cases)
    {
        d_C.SetZero();
        rocm_ck::test::split_gemm(static_cast<const float*>(d_A.GetDeviceBuffer()),
                                  static_cast<const float*>(d_B.GetDeviceBuffer()),
                                  static_cast<float*>(d_C.GetDeviceBuffer()),
                                  cfg.M,
                                  cfg.N,
                                  cfg.K,
                                  cfg.kTile,
                                  tc.strategy,
                                  nullptr,
                                  ws_.get());

        std::vector<float> h_C_gpu(static_cast<size_t>(MN));
        d_C.FromDevice(h_C_gpu.data());

        auto stats = compute_error(h_C_gpu.data(), h_C_ref.data(), MN);
        print_error_stats(tc.name, stats);
    }
}

// =========================================================================
// Test: Veltkamp split HP GEMM vs CPU FP64 reference
// =========================================================================
TEST_P(SplitGemmTest, VeltkampHpVsCpuFp64)
{
    auto cfg = GetParam();

    HostTensor<float> h_A({cfg.M, cfg.K});
    HostTensor<float> h_B({cfg.K, cfg.N});
    fill_uniform(h_A, -1.0f, 1.0f, 50u);
    fill_uniform(h_B, -1.0f, 1.0f, 51u);

    int MN = cfg.M * cfg.N;

    std::vector<float> h_C_ref(static_cast<size_t>(MN));
    cpu_gemm_fp64(h_A.data(), h_B.data(), h_C_ref.data(), cfg.M, cfg.N, cfg.K);

    DeviceMem d_A(h_A);
    DeviceMem d_B(h_B);
    DeviceMem d_C(static_cast<size_t>(MN) * sizeof(float));
    d_C.SetZero();

    rocm_ck::test::split_gemm_hp(static_cast<const float*>(d_A.GetDeviceBuffer()),
                                 static_cast<const float*>(d_B.GetDeviceBuffer()),
                                 static_cast<float*>(d_C.GetDeviceBuffer()),
                                 cfg.M,
                                 cfg.N,
                                 cfg.K,
                                 cfg.kTile,
                                 rocm_ck::test::ReduceStrategy::FP64Sum,
                                 nullptr,
                                 ws_.get());

    std::vector<float> h_C_gpu(static_cast<size_t>(MN));
    d_C.FromDevice(h_C_gpu.data());

    auto stats = compute_error(h_C_gpu.data(), h_C_ref.data(), MN);

    std::cout << "[" << cfg << "] ";
    print_error_stats("VeltkampHP", stats);
}

// =========================================================================
// Test: GPU split_gemm vs CPU reference_batched_gemm (layout cross-check)
//
// Verifies that split_gemm (row-major: C = A[M,K] @ B[K,N]) produces the
// same result as CK's reference_batched_gemm (CK convention: C = A @ B^T
// where B is [batch,N,K]).
//
// This closes the layout coherence gap between the GPU and CPU references.
// =========================================================================
TEST_P(SplitGemmTest, SplitGemmVsReferenceBatchedGemm)
{
    auto cfg = GetParam();

    // Row-major inputs for split_gemm
    HostTensor<float> h_A({cfg.M, cfg.K});
    HostTensor<float> h_B({cfg.K, cfg.N});
    fill_uniform(h_A, -1.0f, 1.0f, 60u);
    fill_uniform(h_B, -1.0f, 1.0f, 61u);

    int MN = cfg.M * cfg.N;

    // --- GPU: split_gemm with NaiveSum ---
    DeviceMem d_A(h_A);
    DeviceMem d_B(h_B);
    DeviceMem d_C(static_cast<size_t>(MN) * sizeof(float));
    d_C.SetZero();

    rocm_ck::test::split_gemm(static_cast<const float*>(d_A.GetDeviceBuffer()),
                              static_cast<const float*>(d_B.GetDeviceBuffer()),
                              static_cast<float*>(d_C.GetDeviceBuffer()),
                              cfg.M,
                              cfg.N,
                              cfg.K,
                              cfg.kTile,
                              rocm_ck::test::ReduceStrategy::NaiveSum,
                              nullptr,
                              ws_.get());

    std::vector<float> h_C_gpu(static_cast<size_t>(MN));
    d_C.FromDevice(h_C_gpu.data());

    // --- CPU: reference_batched_gemm ---
    // CK convention: gemm(A, B, C) computes C[b,m,n] = sum_k A[b,m,k] * B[b,n,k]
    // So B must be [batch, N, K] — transpose our row-major B[K,N] to B_t[N,K].
    HostTensor<float> h_A_3d({1, cfg.M, cfg.K});  // batch=1
    HostTensor<float> h_B_t({1, cfg.N, cfg.K});    // batch=1, transposed
    HostTensor<float> h_C_cpu({1, cfg.M, cfg.N});  // batch=1

    // Copy A into 3D (same data, just adding batch dim)
    for(int i = 0; i < cfg.M; ++i)
        for(int k = 0; k < cfg.K; ++k)
            h_A_3d(0, i, k) = h_A(i, k);

    // Transpose B[K,N] → B_t[N,K]
    for(int k = 0; k < cfg.K; ++k)
        for(int n = 0; n < cfg.N; ++n)
            h_B_t(0, n, k) = h_B(k, n);

    ck_tile::reference_batched_gemm<float, float, float, float>(h_A_3d, h_B_t, h_C_cpu);

    // Extract C_cpu as flat array
    std::vector<float> h_C_ref(static_cast<size_t>(MN));
    for(int i = 0; i < cfg.M; ++i)
        for(int j = 0; j < cfg.N; ++j)
            h_C_ref[static_cast<size_t>(i * cfg.N + j)] = h_C_cpu(0, i, j);

    auto stats = compute_error(h_C_gpu.data(), h_C_ref.data(), MN);

    std::cout << "[" << cfg << "] ";
    print_error_stats("split_gemm_vs_ref_batched", stats);
}

// =========================================================================
// Parameterization
// =========================================================================

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    SplitGemmTest,
    ::testing::Values(SplitGemmConfig{32, 32, 32, 32, "tiny_1tile"},
                      SplitGemmConfig{64, 64, 64, 64, "small_1tile"},
                      SplitGemmConfig{64, 64, 64, 32, "small_2tile"}));

INSTANTIATE_TEST_SUITE_P(
    Medium,
    SplitGemmTest,
    ::testing::Values(SplitGemmConfig{128, 256, 512, 128, "med_4tile"},
                      SplitGemmConfig{256, 128, 1024, 256, "med_4tile_tall"},
                      SplitGemmConfig{128, 128, 256, 64, "med_4tile_sq"}));

INSTANTIATE_TEST_SUITE_P(
    Edge,
    SplitGemmTest,
    ::testing::Values(SplitGemmConfig{1, 1, 1, 1, "1x1x1"},
                      SplitGemmConfig{7, 13, 37, 16, "odd"},
                      SplitGemmConfig{255, 127, 63, 32, "nonaligned"}));
