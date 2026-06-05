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
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/reference/reference_batched_gemm.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

using ck_tile::HostTensor;

namespace {

bool env_enabled(const char* var)
{
    const char* v = std::getenv(var);
    return v && v[0] == '1';
}

void fill_uniform(HostTensor<float>& t, float lo, float hi, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    t.ForEach([&](auto& self, auto idx) { self(idx) = dist(rng); });
}

// CPU FP64 reference GEMM with CK convention: C[i,j] = sum_k A[i,k] * B[j,k]
//   A is [M, K] row-major (stride = K)
//   B is [N, K] row-major (stride = K)  ← CK convention
//   C is [M, N] row-major (stride = N)
// Parallelized over M rows.
void cpu_gemm_fp64(const float* A,
                   const float* B,
                   float* C,
                   int M,
                   int N,
                   int K)
{
    unsigned num_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> threads(num_threads);

    for(unsigned t = 0; t < num_threads; ++t)
    {
        int i_begin = static_cast<int>(static_cast<long>(t) * M / num_threads);
        int i_end   = static_cast<int>(static_cast<long>(t + 1) * M / num_threads);

        threads[t] = std::thread([=]() {
            for(int i = i_begin; i < i_end; ++i)
            {
                for(int j = 0; j < N; ++j)
                {
                    double acc = 0.0;
                    for(int k = 0; k < K; ++k)
                        acc += static_cast<double>(A[i * K + k]) *
                               static_cast<double>(B[j * K + k]);
                    C[i * N + j] = static_cast<float>(acc);
                }
            }
        });
    }

    for(auto& th : threads)
        th.join();
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
    static std::unique_ptr<rocm_ck::test::SplitGemmWorkspace> ws_;

    static void SetUpTestSuite()
    {
        ws_ = std::make_unique<rocm_ck::test::SplitGemmWorkspace>(512, 512, 1024, 32);
    }

    static void TearDownTestSuite() { ws_.reset(); }
};

std::unique_ptr<rocm_ck::test::SplitGemmWorkspace> SplitGemmTest::ws_;

// =========================================================================
// Test: split_gemm correctness vs CPU FP64 reference
// =========================================================================
TEST_P(SplitGemmTest, CorrectnessVsCpuFp64)
{
    if(!env_enabled("CK_TEST_CPU_FP64"))
        GTEST_SKIP() << "set CK_TEST_CPU_FP64=1 to enable";

    auto cfg = GetParam();

    HostTensor<float> h_A({cfg.M, cfg.K});
    HostTensor<float> h_B({cfg.N, cfg.K});
    fill_uniform(h_A, -1.0f, 1.0f, 42u);
    fill_uniform(h_B, -1.0f, 1.0f, 43u);

    int MN = cfg.M * cfg.N;

    std::vector<float> h_C_ref(static_cast<size_t>(MN));
    cpu_gemm_fp64(h_A.data(), h_B.data(), h_C_ref.data(), cfg.M, cfg.N, cfg.K);

    HIP_CHECK_ERROR(hipMemcpy(ws_->d_A, h_A.data(), h_A.get_element_space_size_in_bytes(), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(ws_->d_B, h_B.data(), h_B.get_element_space_size_in_bytes(), hipMemcpyHostToDevice));

    struct ReduceCase
    {
        rocm_ck::test::ReduceStrategy strategy;
        const char* name;
    };

    struct AccumCase
    {
        rocm_ck::test::AccumKLoopPolicy policy;
        const char* name;
    };

    ReduceCase reduce_cases[] = {
        {rocm_ck::test::ReduceStrategy::NaiveSum, "NaiveSum"},
        {rocm_ck::test::ReduceStrategy::CompensatedSum, "Kahan"},
        {rocm_ck::test::ReduceStrategy::FP64Sum, "FP64Sum"},
    };

    AccumCase accum_cases[] = {
        {rocm_ck::test::AccumKLoopPolicy::Naive, "Naive"},
        {rocm_ck::test::AccumKLoopPolicy::TwoSum, "TwoSum"},
        {rocm_ck::test::AccumKLoopPolicy::Kahan, "Kahan"},
    };

    std::cout << "[" << cfg << "]" << std::endl;

    for(const auto& rc : reduce_cases)
    {
        for(const auto& ac : accum_cases)
        {
            HIP_CHECK_ERROR(hipMemset(ws_->d_C, 0, static_cast<size_t>(MN) * sizeof(float)));
            rocm_ck::test::split_gemm(ws_->d_A, ws_->d_B, ws_->d_C,
                                      cfg.M, cfg.N, cfg.K, cfg.kTile,
                                      rc.strategy,
                                      nullptr, ws_.get(), ac.policy);

            std::vector<float> h_C_gpu(static_cast<size_t>(MN));
            HIP_CHECK_ERROR(hipMemcpy(h_C_gpu.data(), ws_->d_C, static_cast<size_t>(MN) * sizeof(float), hipMemcpyDeviceToHost));

            auto stats = compute_error(h_C_gpu.data(), h_C_ref.data(), MN);

            std::string label = std::string("inter=") + rc.name + " intra=" + ac.name;
            print_error_stats(label, stats);
        }
    }
}

// =========================================================================
// Test: Veltkamp split HP GEMM vs CPU FP64 reference
// =========================================================================
TEST_P(SplitGemmTest, VeltkampHpVsCpuFp64)
{
    if(!env_enabled("CK_TEST_VELTKAMP"))
        GTEST_SKIP() << "set CK_TEST_VELTKAMP=1 to enable";

    auto cfg = GetParam();

    HostTensor<float> h_A({cfg.M, cfg.K});
    HostTensor<float> h_B({cfg.N, cfg.K});
    fill_uniform(h_A, -1.0f, 1.0f, 50u);
    fill_uniform(h_B, -1.0f, 1.0f, 51u);

    int MN = cfg.M * cfg.N;

    std::vector<float> h_C_ref(static_cast<size_t>(MN));
    cpu_gemm_fp64(h_A.data(), h_B.data(), h_C_ref.data(), cfg.M, cfg.N, cfg.K);

    HIP_CHECK_ERROR(hipMemcpy(ws_->d_A, h_A.data(), h_A.get_element_space_size_in_bytes(), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(ws_->d_B, h_B.data(), h_B.get_element_space_size_in_bytes(), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemset(ws_->d_C, 0, static_cast<size_t>(MN) * sizeof(float)));

    rocm_ck::test::split_gemm_hp(ws_->d_A, ws_->d_B, ws_->d_C,
                                 cfg.M, cfg.N, cfg.K, cfg.kTile,
                                 rocm_ck::test::ReduceStrategy::FP64Sum,
                                 nullptr, ws_.get());

    std::vector<float> h_C_gpu(static_cast<size_t>(MN));
    HIP_CHECK_ERROR(hipMemcpy(h_C_gpu.data(), ws_->d_C, static_cast<size_t>(MN) * sizeof(float), hipMemcpyDeviceToHost));

    auto stats = compute_error(h_C_gpu.data(), h_C_ref.data(), MN);

    std::cout << "[" << cfg << "] ";
    print_error_stats("VeltkampHP", stats);
}

// =========================================================================
// Test: GPU split_gemm vs CPU reference_batched_gemm
//
// Both use CK convention: C[m,n] = sum_k A[m,k] * B[n,k].
// Same layout, same math — only difference is accumulation order
// (GPU tiled vs CPU sequential).
// =========================================================================
TEST_P(SplitGemmTest, SplitGemmVsReferenceBatchedGemm)
{
    auto cfg = GetParam();

    HostTensor<float> h_A({cfg.M, cfg.K});
    HostTensor<float> h_B({cfg.N, cfg.K});
    fill_uniform(h_A, -1.0f, 1.0f, 60u);
    fill_uniform(h_B, -1.0f, 1.0f, 61u);

    int MN = cfg.M * cfg.N;

    // --- CPU: reference_batched_gemm (batch=1 wrapper) ---
    HostTensor<float> h_A_3d({1, cfg.M, cfg.K});
    HostTensor<float> h_B_3d({1, cfg.N, cfg.K});
    HostTensor<float> h_C_cpu({1, cfg.M, cfg.N});

    for(int i = 0; i < cfg.M; ++i)
        for(int k = 0; k < cfg.K; ++k)
            h_A_3d(0, i, k) = h_A(i, k);

    for(int n = 0; n < cfg.N; ++n)
        for(int k = 0; k < cfg.K; ++k)
            h_B_3d(0, n, k) = h_B(n, k);

    ck_tile::reference_batched_gemm<float, float, float, float>(h_A_3d, h_B_3d, h_C_cpu);

    std::vector<float> h_C_ref(static_cast<size_t>(MN));
    for(int i = 0; i < cfg.M; ++i)
        for(int j = 0; j < cfg.N; ++j)
            h_C_ref[static_cast<size_t>(i * cfg.N + j)] = h_C_cpu(0, i, j);

    // --- GPU: split_gemm with each AccumKLoopPolicy ---
    HIP_CHECK_ERROR(hipMemcpy(ws_->d_A, h_A.data(), h_A.get_element_space_size_in_bytes(), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(ws_->d_B, h_B.data(), h_B.get_element_space_size_in_bytes(), hipMemcpyHostToDevice));

    struct AccumCase
    {
        rocm_ck::test::AccumKLoopPolicy policy;
        const char* name;
    };

    AccumCase accum_cases[] = {
        {rocm_ck::test::AccumKLoopPolicy::Naive, "Naive"},
        {rocm_ck::test::AccumKLoopPolicy::TwoSum, "TwoSum"},
        {rocm_ck::test::AccumKLoopPolicy::Kahan, "Kahan"},
    };

    std::cout << "[" << cfg << "]" << std::endl;

    for(const auto& ac : accum_cases)
    {
        HIP_CHECK_ERROR(hipMemset(ws_->d_C, 0, static_cast<size_t>(MN) * sizeof(float)));

        rocm_ck::test::split_gemm(ws_->d_A, ws_->d_B, ws_->d_C,
                                  cfg.M, cfg.N, cfg.K, cfg.kTile,
                                  rocm_ck::test::ReduceStrategy::NaiveSum,
                                  nullptr, ws_.get(), ac.policy);

        std::vector<float> h_C_gpu(static_cast<size_t>(MN));
        HIP_CHECK_ERROR(hipMemcpy(h_C_gpu.data(), ws_->d_C, static_cast<size_t>(MN) * sizeof(float), hipMemcpyDeviceToHost));

        auto stats = compute_error(h_C_gpu.data(), h_C_ref.data(), MN);
        print_error_stats(ac.name, stats);
    }
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
