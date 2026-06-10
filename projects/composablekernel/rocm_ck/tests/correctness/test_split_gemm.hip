// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// GPU reference GEMM accuracy test.
//
// Single test: all Product × IntraTile × MergePolicy × InterTile combos
// vs CPU FP64 triple-loop reference. Pure FP32 pipeline (no FP64 inter-tile).
// See Notes/Hierarchy_Gemm.md for the decomposition.

#include "split_gemm.hpp"

#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/hip_check_error.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

using ck_tile::HostTensor;

namespace {

// ---- Constants ----

constexpr float kInputLo  = -1.0f;
constexpr float kInputHi  = 1.0f;
constexpr unsigned kSeedA = 42u;
constexpr unsigned kSeedB = 43u;

// ---- Case tables ----

struct AccumCase
{
    rocm_ck::test::AccumKLoopPolicy policy;
    const char* label;
};

struct ReduceCase
{
    rocm_ck::test::ReduceStrategy strategy;
    const char* label;
};

// Product × IntraTile × MergePolicy (see Hierarchy_Gemm.md)
const AccumCase kAccumCases[] = {
    {rocm_ck::test::AccumKLoopPolicy::Naive,           "Standard/HWChain/Naive"},
    {rocm_ck::test::AccumKLoopPolicy::TwoSum,           "Standard/HWChain/TwoSum"},
    {rocm_ck::test::AccumKLoopPolicy::Kahan,            "Standard/HWChain/Kahan"},
    {rocm_ck::test::AccumKLoopPolicy::NaiveK4,          "Standard/Decomposed/Naive"},
    {rocm_ck::test::AccumKLoopPolicy::TwoSumK4,         "Standard/Decomposed/TwoSum"},
    {rocm_ck::test::AccumKLoopPolicy::KahanK4,          "Standard/Decomposed/Kahan"},
    {rocm_ck::test::AccumKLoopPolicy::VeltkampK4,       "Veltkamp/Decomposed/TwoSum"},
    {rocm_ck::test::AccumKLoopPolicy::VeltkampNaiveK4,  "Veltkamp/Decomposed/Naive"},
};

// Inter-tile reduce — pure FP32 only (no FP64Sum)
const ReduceCase kReduceCases[] = {
    {rocm_ck::test::ReduceStrategy::NaiveSum,       "NaiveSum"},
    {rocm_ck::test::ReduceStrategy::CompensatedSum, "Kahan"},
    {rocm_ck::test::ReduceStrategy::TwoSumReduce,   "TwoSum"},
    {rocm_ck::test::ReduceStrategy::PairwiseSum,    "PairwiseSum"},
};

// ---- Helpers ----

void fill_uniform(HostTensor<float>& t, float lo, float hi, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    t.ForEach([&](auto& self, auto idx) { self(idx) = dist(rng); });
}

// CPU FP64 reference: C[i,j] = sum_k A[i,k] * B[j,k]  (CK convention)
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

double frobenius_norm(const float* data, int count)
{
    double sum_sq = 0.0;
    for(int i = 0; i < count; ++i)
    {
        double v = static_cast<double>(data[i]);
        sum_sq += v * v;
    }
    return std::sqrt(sum_sq);
}

struct ErrorStats
{
    double max_abs;
    double frob_err;
    double frob_rel;
};

// frob_rel = ||actual - ref||_F / (||A||_F * ||B||_F)
ErrorStats compute_error(const float* actual, const float* reference,
                         int count, double norm_AB)
{
    double max_abs = 0;
    double sum_sq  = 0;

    for(int i = 0; i < count; ++i)
    {
        double a    = static_cast<double>(actual[i]);
        double r    = static_cast<double>(reference[i]);
        double diff = a - r;

        max_abs = std::max(max_abs, std::fabs(diff));
        sum_sq += diff * diff;
    }

    double frob_err = std::sqrt(sum_sq);
    return {max_abs, frob_err, frob_err / norm_AB};
}

ErrorStats compute_error_compensated(const float* c_buf,
                                     const float* err_buf,
                                     const float* reference,
                                     int count, double norm_AB)
{
    double max_abs = 0;
    double sum_sq  = 0;

    for(int i = 0; i < count; ++i)
    {
        double a    = static_cast<double>(c_buf[i]) + static_cast<double>(err_buf[i]);
        double r    = static_cast<double>(reference[i]);
        double diff = a - r;

        max_abs = std::max(max_abs, std::fabs(diff));
        sum_sq += diff * diff;
    }

    double frob_err = std::sqrt(sum_sq);
    return {max_abs, frob_err, frob_err / norm_AB};
}

void print_error(const std::string& label, const ErrorStats& s)
{
    std::printf("  %-45s  max_abs=%.6e  frob_rel=%.6e\n",
                label.c_str(), s.max_abs, s.frob_rel);
}

// Prepare inputs, compute CPU FP64 reference, upload A and B to device.
struct TestInputs
{
    HostTensor<float> h_A;
    HostTensor<float> h_B;
    std::vector<float> h_C_ref;
    int MN;
    double norm_AB;

    TestInputs(int M, int N, int K)
        : h_A({M, K}), h_B({N, K}), h_C_ref(static_cast<size_t>(M * N)),
          MN(M * N), norm_AB(0.0)
    {
    }
};

TestInputs prepare_inputs(int M, int N, int K,
                          rocm_ck::test::SplitGemmWorkspace* ws)
{
    TestInputs inp(M, N, K);

    fill_uniform(inp.h_A, kInputLo, kInputHi, kSeedA);
    fill_uniform(inp.h_B, kInputLo, kInputHi, kSeedB);

    inp.norm_AB = frobenius_norm(inp.h_A.data(), M * K) *
                  frobenius_norm(inp.h_B.data(), N * K);

    cpu_gemm_fp64(inp.h_A.data(), inp.h_B.data(),
                  inp.h_C_ref.data(), M, N, K);

    HIP_CHECK_ERROR(hipMemcpy(ws->d_A, inp.h_A.data(),
                              inp.h_A.get_element_space_size_in_bytes(),
                              hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(ws->d_B, inp.h_B.data(),
                              inp.h_B.get_element_space_size_in_bytes(),
                              hipMemcpyHostToDevice));

    return inp;
}

// Run split_gemm on device, copy result back.
std::vector<float> run_and_readback(rocm_ck::test::SplitGemmWorkspace* ws,
                                    int M, int N, int K, int kTile,
                                    rocm_ck::test::ReduceStrategy reduce,
                                    rocm_ck::test::AccumKLoopPolicy accum)
{
    int MN = M * N;
    HIP_CHECK_ERROR(hipMemset(ws->d_C, 0,
                              static_cast<size_t>(MN) * sizeof(float)));

    rocm_ck::test::split_gemm(ws->d_A, ws->d_B, ws->d_C,
                              M, N, K, kTile, reduce,
                              nullptr, ws, accum);

    std::vector<float> h_C(static_cast<size_t>(MN));
    HIP_CHECK_ERROR(hipMemcpy(h_C.data(), ws->d_C,
                              static_cast<size_t>(MN) * sizeof(float),
                              hipMemcpyDeviceToHost));
    return h_C;
}

} // namespace

// =========================================================================
// Test fixture
// =========================================================================

struct SplitGemmConfig
{
    int M, N, K, kTile;
    std::string name;
};

std::ostream& operator<<(std::ostream& os, const SplitGemmConfig& c)
{
    return os << c.name << "_M" << c.M << "_N" << c.N
              << "_K" << c.K << "_kT" << c.kTile;
}

class SplitGemmTest : public ::testing::TestWithParam<SplitGemmConfig>
{
protected:
    static std::unique_ptr<rocm_ck::test::SplitGemmWorkspace> ws_;

    static void SetUpTestSuite()
    {
        ws_ = std::make_unique<rocm_ck::test::SplitGemmWorkspace>(
            4096, 4096, 4096, 256);
    }

    static void TearDownTestSuite() { ws_.reset(); }
};

std::unique_ptr<rocm_ck::test::SplitGemmWorkspace> SplitGemmTest::ws_;

// =========================================================================
// Single test: all accum × reduce combos vs CPU FP64
// + VeltkampK4 split output sub-section
// =========================================================================
TEST_P(SplitGemmTest, CorrectnessVsCpuFp64)
{
    auto cfg = GetParam();
    auto inp = prepare_inputs(cfg.M, cfg.N, cfg.K, ws_.get());

    std::cout << "\n[" << cfg << "]\n" << std::endl;

    // --- Main grid: 7 accum × 2 inter-tile reduce ---

    for(const auto& rc : kReduceCases)
    {
        std::cout << "  -- inter-tile: " << rc.label << " --" << std::endl;

        for(const auto& ac : kAccumCases)
        {
            auto h_C = run_and_readback(ws_.get(),
                                        cfg.M, cfg.N, cfg.K, cfg.kTile,
                                        rc.strategy, ac.policy);

            auto stats = compute_error(h_C.data(), inp.h_C_ref.data(), inp.MN, inp.norm_AB);
            print_error(ac.label, stats);
        }

        std::cout << std::endl;
    }

    // --- Split output: all compensated variants (single tile, kTile=K) ---
    // Writes c and err separately, reconstructs (double)c + (double)err on CPU.
    // Bypasses FP32 finalize floor. Naive variants have err=0, so split = finalized.

    if(rocm_ck::test::can_use_mfma(cfg.M, cfg.N, cfg.K))
    {
        std::cout << "  -- split output (single tile, kTile=K) --" << std::endl;

        for(const auto& ac : kAccumCases)
        {
            HIP_CHECK_ERROR(hipMemset(ws_->d_C, 0,
                                      static_cast<size_t>(inp.MN) * sizeof(float)));
            HIP_CHECK_ERROR(hipMemset(ws_->err_buf, 0,
                                      static_cast<size_t>(inp.MN) * sizeof(float)));

            if(ac.policy == rocm_ck::test::AccumKLoopPolicy::VeltkampK4)
            {
                rocm_ck::test::launch_mfma_tile_gemm_veltkamp_k4(
                    ws_->d_A, ws_->d_B, ws_->d_C,
                    cfg.M, cfg.N, cfg.K, 0, cfg.K,
                    nullptr, ws_->err_buf);
            }
            else if(ac.policy == rocm_ck::test::AccumKLoopPolicy::VeltkampNaiveK4)
            {
                rocm_ck::test::launch_mfma_tile_gemm_veltkamp_naive_k4(
                    ws_->d_A, ws_->d_B, ws_->d_C,
                    cfg.M, cfg.N, cfg.K, 0, cfg.K,
                    nullptr, ws_->err_buf);
            }
            else if(ac.policy == rocm_ck::test::AccumKLoopPolicy::NaiveK4 ||
                    ac.policy == rocm_ck::test::AccumKLoopPolicy::TwoSumK4 ||
                    ac.policy == rocm_ck::test::AccumKLoopPolicy::KahanK4)
            {
                rocm_ck::test::launch_mfma_tile_gemm_k4(
                    ws_->d_A, ws_->d_B, ws_->d_C,
                    cfg.M, cfg.N, cfg.K, 0, cfg.K,
                    nullptr, ac.policy, ws_->err_buf);
            }
            else
            {
                rocm_ck::test::launch_mfma_tile_gemm(
                    ws_->d_A, ws_->d_B, ws_->d_C,
                    cfg.M, cfg.N, cfg.K, 0, cfg.K,
                    nullptr, ac.policy, ws_->err_buf);
            }
            HIP_CHECK_ERROR(hipDeviceSynchronize());

            std::vector<float> h_C(static_cast<size_t>(inp.MN));
            std::vector<float> h_err(static_cast<size_t>(inp.MN));
            HIP_CHECK_ERROR(hipMemcpy(h_C.data(), ws_->d_C,
                                      static_cast<size_t>(inp.MN) * sizeof(float),
                                      hipMemcpyDeviceToHost));
            HIP_CHECK_ERROR(hipMemcpy(h_err.data(), ws_->err_buf,
                                      static_cast<size_t>(inp.MN) * sizeof(float),
                                      hipMemcpyDeviceToHost));

            auto stats_fin = compute_error(
                h_C.data(), inp.h_C_ref.data(), inp.MN, inp.norm_AB);
            auto stats_split = compute_error_compensated(
                h_C.data(), h_err.data(), inp.h_C_ref.data(), inp.MN, inp.norm_AB);

            std::string lbl(ac.label);
            print_error(lbl + " finalized", stats_fin);
            print_error(lbl + " split (c+err@FP64)", stats_split);
        }

        std::cout << std::endl;
    }
}

// =========================================================================
// Parameterization
// =========================================================================

INSTANTIATE_TEST_SUITE_P(
    Phase2,
    SplitGemmTest,
    ::testing::Values(
        SplitGemmConfig{256, 128, 1024, 256, "k1024_4tile"},
        SplitGemmConfig{2048, 2048, 2048, 256, "k2048_8tile"},
        SplitGemmConfig{4096, 4096, 4096, 256, "k4096_16tile"}));
