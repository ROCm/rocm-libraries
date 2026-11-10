// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// miopenConvolutionForwardGetWorkSpaceSize must report the GemmFwdRest workspace for dilated
// convolutions, up to the 288 MiB cap in GetWorkSpaceSizeGEMM (src/convolution.cpp).
//
// Applications such as PyTorch size the Find workspace from that query. When it under-reports,
// Find drops GemmFwdRest for lack of workspace, and if no other solver applies it fails with
// "No suitable algorithm was found" (rocm-libraries#14965). A larger dilated Im2Col is split along
// the output so its workspace fits the cap; one the split cannot fit is not reported, so callers
// do not allocate it (#6974).
//
// MIOPEN_DEBUG_FIND_ONLY_SOLVER=GemmFwdRest makes GemmFwdRest the only candidate on every arch,
// and ScopedFindDb keeps find-db records from short-circuiting either the query or Find.

#include <gtest/gtest.h>

#include <miopen/config.h>
#include <miopen/convolution.hpp>
#include <miopen/find_db.hpp>
#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>

#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "../random.hpp"
#include "../tensor_holder.hpp"
#include "../verify.hpp"
#include "../workspace.hpp"

namespace {

constexpr std::size_t DilatedGemmWorkspaceCap = 301989888; // 288 MiB

struct ScopedFindDb
{
    ScopedFindDb() : prev_(miopen::debug::testing_find_db_enabled)
    {
        miopen::debug::testing_find_db_enabled = false;
    }
    ~ScopedFindDb() { miopen::debug::testing_find_db_enabled = prev_; }

    bool prev_;
};

struct DilatedConv
{
    tensor<float> x;
    tensor<float> w;
    miopen::ConvolutionDescriptor conv;
    tensor<float> y;

    DilatedConv(std::vector<std::size_t> x_lens,
                std::vector<std::size_t> w_lens,
                std::vector<int> pads,
                std::vector<int> dilations)
        : x{x_lens},
          w{w_lens},
          conv{2, miopenConvolution, miopenPaddingDefault, pads, {1, 1}, dilations},
          y{conv.GetForwardOutputTensor(x.desc, w.desc)}
    {
    }

    std::size_t QueriedWorkspace(miopenHandle_t handle)
    {
        std::size_t size = 0;
        EXPECT_EQ(miopenConvolutionForwardGetWorkSpaceSize(
                      handle, &w.desc, &x.desc, &conv, &y.desc, &size),
                  miopenStatusSuccess);
        return size;
    }

    std::size_t GemmFwdRestWorkspace(miopenHandle_t handle)
    {
        std::size_t size = 0;
        EXPECT_EQ(miopenConvolutionForwardGetSolutionWorkspaceSize(
                      handle,
                      &w.desc,
                      &x.desc,
                      &conv,
                      &y.desc,
                      miopen::solver::Id{"GemmFwdRest"}.Value(),
                      &size),
                  miopenStatusSuccess);
        return size;
    }
};

} // namespace

// rocm-libraries#14965: dilated fp32 Conv1d. Find given the queried workspace must select
// GemmFwdRest instead of failing.
TEST(GPU_ConvDilatedGemmWorkspace_FP32, FindWithQueriedWorkspaceRunsGemmFwdRest)
{
#if !MIOPEN_USE_GEMM
    GTEST_SKIP() << "Built without GEMM support";
#else
    auto& handle_ref      = get_handle();
    miopenHandle_t handle = &handle_ref;

    ScopedFindDb no_find_db;
    ScopedEnvironment<std::string> only_gemm(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                             std::string{"GemmFwdRest"});

    DilatedConv problem{{1, 32, 1, 1024}, {32, 32, 1, 3}, {0, 3}, {1, 3}};

    const auto required = problem.GemmFwdRestWorkspace(handle);
    ASSERT_GT(required, 0) << "GemmFwdRest needs Im2Col workspace for this problem";
    ASSERT_LE(required, DilatedGemmWorkspaceCap);

    const auto queried = problem.QueriedWorkspace(handle);
    EXPECT_GE(queried, required);

    auto x_dev = handle_ref.Write(problem.x.data);
    auto w_dev = handle_ref.Write(problem.w.data);
    auto y_dev = handle_ref.Write(problem.y.data);
    Workspace workspace{queried};

    int returned_algo_count = 0;
    miopenConvAlgoPerf_t perf{};
    ASSERT_EQ(miopenFindConvolutionForwardAlgorithm(handle,
                                                    &problem.x.desc,
                                                    x_dev.get(),
                                                    &problem.w.desc,
                                                    w_dev.get(),
                                                    &problem.conv,
                                                    &problem.y.desc,
                                                    y_dev.get(),
                                                    1,
                                                    &returned_algo_count,
                                                    &perf,
                                                    workspace.ptr(),
                                                    workspace.size(),
                                                    false),
              miopenStatusSuccess);
    ASSERT_EQ(returned_algo_count, 1);
    EXPECT_EQ(perf.fwd_algo, miopenConvolutionFwdAlgoGEMM);
#endif
}

namespace {

// Runs the problem through Find + miopenConvolutionForward with only `solver` enabled.
std::vector<float> RunOnly(DilatedConv& problem, const std::string& solver)
{
    auto& handle_ref      = get_handle();
    miopenHandle_t handle = &handle_ref;
    ScopedEnvironment<std::string> only(MIOPEN_DEBUG_FIND_ONLY_SOLVER, solver);

    auto x_dev = handle_ref.Write(problem.x.data);
    auto w_dev = handle_ref.Write(problem.w.data);
    auto y_dev = handle_ref.Write(problem.y.data);
    Workspace workspace{problem.QueriedWorkspace(handle)};

    int returned_algo_count = 0;
    miopenConvAlgoPerf_t perf{};
    EXPECT_EQ(miopenFindConvolutionForwardAlgorithm(handle,
                                                    &problem.x.desc,
                                                    x_dev.get(),
                                                    &problem.w.desc,
                                                    w_dev.get(),
                                                    &problem.conv,
                                                    &problem.y.desc,
                                                    y_dev.get(),
                                                    1,
                                                    &returned_algo_count,
                                                    &perf,
                                                    workspace.ptr(),
                                                    workspace.size(),
                                                    false),
              miopenStatusSuccess);
    EXPECT_EQ(returned_algo_count, 1);

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    EXPECT_EQ(miopenConvolutionForward(handle,
                                       &alpha,
                                       &problem.x.desc,
                                       x_dev.get(),
                                       &problem.w.desc,
                                       w_dev.get(),
                                       &problem.conv,
                                       perf.fwd_algo,
                                       &beta,
                                       &problem.y.desc,
                                       y_dev.get(),
                                       workspace.ptr(),
                                       workspace.size()),
              miopenStatusSuccess);
    return handle_ref.Read<float>(y_dev, problem.y.data.size());
}

void CheckSplitMatchesNaive(DilatedConv problem, std::size_t full_im2col)
{
    auto& handle_ref      = get_handle();
    miopenHandle_t handle = &handle_ref;

    ScopedFindDb no_find_db;
    prng::reset_seed();
    problem.x.generate([](auto...) { return prng::gen_A_to_B(-1.0f, 1.0f); });
    problem.w.generate([](auto...) { return prng::gen_A_to_B(-1.0f, 1.0f); });

    std::size_t gemm_ws = 0;
    {
        ScopedEnvironment<std::string> only_gemm(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                                 std::string{"GemmFwdRest"});
        ASSERT_GT(full_im2col, DilatedGemmWorkspaceCap);
        gemm_ws = problem.GemmFwdRestWorkspace(handle);
        ASSERT_GT(gemm_ws, 0);
        ASSERT_LE(gemm_ws, DilatedGemmWorkspaceCap);
        EXPECT_GE(problem.QueriedWorkspace(handle), gemm_ws);
    }

    const auto ref = RunOnly(problem, "ConvDirectNaiveConvFwd");
    const auto obs = RunOnly(problem, "GemmFwdRest");
    EXPECT_LT(miopen::rms_range(ref, obs), 1e-5);
}

} // namespace

// Kokoro-82M vocoder Conv1d (k11, d5): 727 MiB of Im2Col, split into 3 pieces along the width.
TEST(GPU_ConvDilatedGemmWorkspace_FP32, SplitAlongWidthMatchesNaive)
{
#if !MIOPEN_USE_GEMM
    GTEST_SKIP() << "Built without GEMM support";
#else
    CheckSplitMatchesNaive({{1, 128, 1, 135361}, {128, 128, 1, 11}, {0, 25}, {1, 5}},
                           std::size_t{128} * 11 * 135361 * sizeof(float));
#endif
}

// 2d, 255 output rows: 2 pieces of 128 and 127 rows.
TEST(GPU_ConvDilatedGemmWorkspace_FP32, SplitAlongHeightMatchesNaive)
{
#if !MIOPEN_USE_GEMM
    GTEST_SKIP() << "Built without GEMM support";
#else
    CheckSplitMatchesNaive({{1, 256, 255, 256}, {16, 256, 3, 3}, {2, 2}, {2, 2}},
                           std::size_t{256} * 3 * 3 * 255 * 256 * sizeof(float));
#endif
}

// When a single output row of Im2Col exceeds the cap the split cannot fit it, so it is not
// reported.
TEST(GPU_ConvDilatedGemmWorkspace_FP32, WorkspaceAboveCapIsNotReported)
{
#if !MIOPEN_USE_GEMM
    GTEST_SKIP() << "Built without GEMM support";
#else
    auto& handle_ref      = get_handle();
    miopenHandle_t handle = &handle_ref;

    ScopedFindDb no_find_db;
    ScopedEnvironment<std::string> only_gemm(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                             std::string{"GemmFwdRest"});

    // One row of Im2Col: 1024 * 7 * 7 * 2048 * sizeof(float) = 392 MiB.
    DilatedConv problem{{1, 1024, 4, 2048}, {1, 1024, 7, 7}, {3, 9}, {1, 3}};

    ASSERT_GT(problem.GemmFwdRestWorkspace(handle), DilatedGemmWorkspaceCap);
    EXPECT_EQ(problem.QueriedWorkspace(handle), 0);
#endif
}
