// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Tests for the split-accumulation GPU reference GEMM.
//
// Compares split_gemm against:
//   1. CPU reference (reference_batched_gemm with double-precision accumulation)
//   2. Different reduction strategies against each other
//
// Reports per-config error statistics (max, mean, p99 relative error)
// to feed the FMHA error model.

#include "split_gemm.hpp"

#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/hip_check_error.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
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
    double mean_rel;
    double p99_rel;
    int num_elements;
};

// Compute error stats using the mixed tolerance formula:
//   error = |actual - ref| / (atol + rtol * |ref|)
// This avoids inflated relative error on near-zero values.
// The "max_rel" field reports the max of |actual - ref| / |ref| but only
// for elements where |ref| > abs_floor (to skip near-zero denominators).
ErrorStats compute_error(const float* actual, const float* reference, int count)
{
    constexpr double abs_floor = 1e-6;

    std::vector<double> rel_errors;
    rel_errors.reserve(count);

    double max_abs = 0;
    double max_rel = 0;

    for(int i = 0; i < count; ++i)
    {
        double a    = static_cast<double>(actual[i]);
        double r    = static_cast<double>(reference[i]);
        double diff = std::fabs(a - r);

        max_abs = std::max(max_abs, diff);

        if(std::fabs(r) > abs_floor)
        {
            double rel = diff / std::fabs(r);
            max_rel    = std::max(max_rel, rel);
            rel_errors.push_back(rel);
        }
    }

    if(rel_errors.empty())
        return {max_abs, 0, 0, 0, count};

    std::sort(rel_errors.begin(), rel_errors.end());
    double mean_rel = 0;
    for(auto e : rel_errors)
        mean_rel += e;
    mean_rel /= static_cast<double>(rel_errors.size());

    auto p99_idx = std::min(static_cast<size_t>(rel_errors.size() * 0.99), rel_errors.size() - 1);
    double p99   = rel_errors[p99_idx];

    return {max_abs, max_rel, mean_rel, p99, count};
}

void print_error_stats(const std::string& label, const ErrorStats& s)
{
    std::cout << "  " << label << ": max_abs=" << s.max_abs << " max_rel=" << s.max_rel
              << " mean_rel=" << s.mean_rel << " p99_rel=" << s.p99_rel << " (n=" << s.num_elements
              << ")" << std::endl;
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
// Test: single-tile case should match naive (no reduce)
// =========================================================================
TEST_P(SplitGemmTest, SingleTileMatchesFullK)
{
    auto cfg = GetParam();

    // Use kTile >= K so there's only one tile
    int big_kTile = cfg.K;

    HostTensor<float> h_A({cfg.M, cfg.K});
    HostTensor<float> h_B({cfg.K, cfg.N});
    fill_uniform(h_A, -1.0f, 1.0f, 44u);
    fill_uniform(h_B, -1.0f, 1.0f, 45u);

    int MN = cfg.M * cfg.N;

    DeviceMem d_A(h_A);
    DeviceMem d_B(h_B);
    DeviceMem d_C1(static_cast<size_t>(MN) * sizeof(float));
    DeviceMem d_C2(static_cast<size_t>(MN) * sizeof(float));

    // Run with small kTile (multi-tile + reduce)
    rocm_ck::test::split_gemm(static_cast<const float*>(d_A.GetDeviceBuffer()),
                              static_cast<const float*>(d_B.GetDeviceBuffer()),
                              static_cast<float*>(d_C1.GetDeviceBuffer()),
                              cfg.M,
                              cfg.N,
                              cfg.K,
                              cfg.kTile,
                              rocm_ck::test::ReduceStrategy::FP64Sum,
                              nullptr,
                              ws_.get());

    // Run with kTile = K (single tile, no reduce)
    rocm_ck::test::split_gemm(static_cast<const float*>(d_A.GetDeviceBuffer()),
                              static_cast<const float*>(d_B.GetDeviceBuffer()),
                              static_cast<float*>(d_C2.GetDeviceBuffer()),
                              cfg.M,
                              cfg.N,
                              cfg.K,
                              big_kTile,
                              rocm_ck::test::ReduceStrategy::NaiveSum,
                              nullptr,
                              ws_.get());

    std::vector<float> h1(static_cast<size_t>(MN)), h2(static_cast<size_t>(MN));
    d_C1.FromDevice(h1.data());
    d_C2.FromDevice(h2.data());

    auto stats = compute_error(h1.data(), h2.data(), MN);

    std::cout << "[" << cfg << "] SingleTile vs Split: ";
    print_error_stats("FP64Sum_split_vs_NaiveSum_full", stats);
}

// =========================================================================
// Test: FP64Sum should be tighter than NaiveSum
// =========================================================================
TEST_P(SplitGemmTest, FP64SumTighterThanNaive)
{
    auto cfg = GetParam();
    if(cfg.K <= cfg.kTile)
        GTEST_SKIP() << "Single tile — no reduction difference";

    HostTensor<float> h_A({cfg.M, cfg.K});
    HostTensor<float> h_B({cfg.K, cfg.N});
    fill_uniform(h_A, -1.0f, 1.0f, 46u);
    fill_uniform(h_B, -1.0f, 1.0f, 47u);

    int MN = cfg.M * cfg.N;

    std::vector<float> h_ref(static_cast<size_t>(MN));
    cpu_gemm_fp64(h_A.data(), h_B.data(), h_ref.data(), cfg.M, cfg.N, cfg.K);

    DeviceMem d_A(h_A);
    DeviceMem d_B(h_B);
    DeviceMem d_C(static_cast<size_t>(MN) * sizeof(float));

    // NaiveSum
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
    std::vector<float> h_naive(static_cast<size_t>(MN));
    d_C.FromDevice(h_naive.data());

    // FP64Sum
    d_C.SetZero();
    rocm_ck::test::split_gemm(static_cast<const float*>(d_A.GetDeviceBuffer()),
                              static_cast<const float*>(d_B.GetDeviceBuffer()),
                              static_cast<float*>(d_C.GetDeviceBuffer()),
                              cfg.M,
                              cfg.N,
                              cfg.K,
                              cfg.kTile,
                              rocm_ck::test::ReduceStrategy::FP64Sum,
                              nullptr,
                              ws_.get());
    std::vector<float> h_fp64(static_cast<size_t>(MN));
    d_C.FromDevice(h_fp64.data());

    auto err_naive = compute_error(h_naive.data(), h_ref.data(), MN);
    auto err_fp64  = compute_error(h_fp64.data(), h_ref.data(), MN);

    if(err_fp64.max_rel > err_naive.max_rel)
        std::cout << "  NOTE: FP64Sum (" << err_fp64.max_rel
                  << ") is worse than NaiveSum (" << err_naive.max_rel << ")" << std::endl;
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

    // CPU FP64 reference
    std::vector<float> h_C_ref(static_cast<size_t>(MN));
    cpu_gemm_fp64(h_A.data(), h_B.data(), h_C_ref.data(), cfg.M, cfg.N, cfg.K);

    // GPU Veltkamp HP
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

    std::cout << "[" << cfg << "] VeltkampHP: ";
    print_error_stats("VeltkampHP+FP64Sum", stats);

    // Report only — you set the thresholds after reviewing the data
    std::cout << "  (no threshold asserted — review error stats above)" << std::endl;
}

// =========================================================================
// Test: Veltkamp HP should be tighter than plain FP64Sum
// =========================================================================
TEST_P(SplitGemmTest, VeltkampTighterThanFP64Sum)
{
    auto cfg = GetParam();
    if(cfg.K <= cfg.kTile)
        GTEST_SKIP() << "Single tile — minimal accumulation error to improve on";

    HostTensor<float> h_A({cfg.M, cfg.K});
    HostTensor<float> h_B({cfg.K, cfg.N});
    fill_uniform(h_A, -1.0f, 1.0f, 52u);
    fill_uniform(h_B, -1.0f, 1.0f, 53u);

    int MN = cfg.M * cfg.N;

    std::vector<float> h_ref(static_cast<size_t>(MN));
    cpu_gemm_fp64(h_A.data(), h_B.data(), h_ref.data(), cfg.M, cfg.N, cfg.K);

    DeviceMem d_A(h_A);
    DeviceMem d_B(h_B);
    DeviceMem d_C(static_cast<size_t>(MN) * sizeof(float));

    // Plain FP64Sum (no Veltkamp)
    d_C.SetZero();
    rocm_ck::test::split_gemm(static_cast<const float*>(d_A.GetDeviceBuffer()),
                              static_cast<const float*>(d_B.GetDeviceBuffer()),
                              static_cast<float*>(d_C.GetDeviceBuffer()),
                              cfg.M,
                              cfg.N,
                              cfg.K,
                              cfg.kTile,
                              rocm_ck::test::ReduceStrategy::FP64Sum,
                              nullptr,
                              ws_.get());
    std::vector<float> h_fp64(static_cast<size_t>(MN));
    d_C.FromDevice(h_fp64.data());

    // Veltkamp HP + FP64Sum
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
    std::vector<float> h_veltkamp(static_cast<size_t>(MN));
    d_C.FromDevice(h_veltkamp.data());

    auto err_fp64     = compute_error(h_fp64.data(), h_ref.data(), MN);
    auto err_veltkamp = compute_error(h_veltkamp.data(), h_ref.data(), MN);

    std::cout << "[" << cfg << "] FP64Sum max_rel=" << err_fp64.max_rel
              << " VeltkampHP max_rel=" << err_veltkamp.max_rel << std::endl;

    // Report comparison — you decide if the ordering holds as expected
    if(err_veltkamp.max_rel > err_fp64.max_rel)
        std::cout << "  NOTE: VeltkampHP (" << err_veltkamp.max_rel
                  << ") is worse than FP64Sum (" << err_fp64.max_rel << ")" << std::endl;
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
    FAScale,
    SplitGemmTest,
    ::testing::Values(SplitGemmConfig{512, 512, 512, 256, "fa_sq"},
                      SplitGemmConfig{512, 128, 256, 128, "fa_qk"},
                      SplitGemmConfig{512, 256, 512, 256, "fa_pv"}));

INSTANTIATE_TEST_SUITE_P(
    Edge,
    SplitGemmTest,
    ::testing::Values(SplitGemmConfig{1, 1, 1, 1, "1x1x1"},
                      SplitGemmConfig{7, 13, 37, 16, "odd"},
                      SplitGemmConfig{255, 127, 63, 32, "nonaligned"}));
