// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// miopenConvolutionForwardGetWorkSpaceSize must report the GemmFwdRest workspace for dilated
// convolutions, up to the 288 MiB cap in GetWorkSpaceSizeGEMM (src/convolution.cpp).
//
// Applications such as PyTorch size the Find workspace from that query. When it under-reports,
// Find drops GemmFwdRest for lack of workspace, and if no other solver applies it fails with
// "No suitable algorithm was found" (rocm-libraries#14965). Above the cap the size is not
// reported, so callers do not allocate it (#6974).
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
#include "../tensor_holder.hpp"
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

// Above the cap, the dilated GemmFwdRest workspace must not be reported.
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

    // Im2Col: 256 * 3 * 3 * 256 * 256 * sizeof(float) = 576 MiB.
    DilatedConv problem{{1, 256, 256, 256}, {16, 256, 3, 3}, {2, 2}, {2, 2}};

    ASSERT_GT(problem.GemmFwdRestWorkspace(handle), DilatedGemmWorkspaceCap);
    EXPECT_EQ(problem.QueriedWorkspace(handle), 0);
#endif
}
