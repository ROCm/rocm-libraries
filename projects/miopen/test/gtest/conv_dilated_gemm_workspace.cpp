// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// miopenConvolutionForwardGetWorkSpaceSize must report the GemmFwdRest workspace for dilated
// convolutions when it fits in the device memory free at query time (GetWorkSpaceSizeGEMM in
// src/convolution.cpp).
//
// Applications such as PyTorch size the Find workspace from that query. When it under-reports,
// Find drops GemmFwdRest for lack of workspace, and if no other solver applies it fails with
// "No suitable algorithm was found" (rocm-libraries#14965). A size that does not fit in free
// memory is not reported, so callers do not try to allocate it (#6974).
//
// MIOPEN_DEBUG_FIND_ONLY_SOLVER=GemmFwdRest makes GemmFwdRest the only candidate on every arch.
// ScopedFindDb keeps find-db records from short-circuiting either the query or Find, except in the
// test of the find-db path, which points the find-db at a temporary file.

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
#include "find_db.hpp"

namespace {

// A real device handle that reports a fixed amount of free memory.
class AvailableMemoryHandle final : public miopen::Handle
{
public:
    explicit AvailableMemoryHandle(std::size_t available) : available_(available) {}
    std::size_t GetAvailableMemory() const override { return available_; }

private:
    std::size_t available_;
};

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

    // Find with the given workspace; returns the chosen algorithm.
    miopenConvFwdAlgorithm_t FindForward(miopen::Handle& handle, std::size_t workspace_size)
    {
        auto x_dev = handle.Write(x.data);
        auto w_dev = handle.Write(w.data);
        auto y_dev = handle.Write(y.data);
        Workspace workspace{workspace_size};

        int returned_algo_count = 0;
        miopenConvAlgoPerf_t perf{};
        EXPECT_EQ(miopenFindConvolutionForwardAlgorithm(&handle,
                                                        &x.desc,
                                                        x_dev.get(),
                                                        &w.desc,
                                                        w_dev.get(),
                                                        &conv,
                                                        &y.desc,
                                                        y_dev.get(),
                                                        1,
                                                        &returned_algo_count,
                                                        &perf,
                                                        workspace.ptr(),
                                                        workspace.size(),
                                                        false),
                  miopenStatusSuccess);
        EXPECT_EQ(returned_algo_count, 1);
        return perf.fwd_algo;
    }
};

// rocm-libraries#14965 in miniature: dilated fp32 Conv1d.
DilatedConv ReproDilatedConv() { return {{1, 32, 1, 1024}, {32, 32, 1, 3}, {0, 3}, {1, 3}}; }

// Im2Col: 256 * 3 * 3 * 256 * 256 * sizeof(float) = 576 MiB.
DilatedConv LargeDilatedConv() { return {{1, 256, 256, 256}, {16, 256, 3, 3}, {2, 2}, {2, 2}}; }

} // namespace

// rocm-libraries#14965: dilated fp32 Conv1d. Find given the queried workspace must select
// GemmFwdRest instead of failing.
TEST(GPU_ConvDilatedGemmWorkspace_FP32, FindWithQueriedWorkspaceRunsGemmFwdRest)
{
#if !MIOPEN_USE_GEMM
    GTEST_SKIP() << "Built without GEMM support";
#else
    auto& handle = get_handle();

    ScopedFindDb no_find_db;
    ScopedEnvironment<std::string> only_gemm(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                             std::string{"GemmFwdRest"});

    auto problem = ReproDilatedConv();

    const auto required = problem.GemmFwdRestWorkspace(&handle);
    ASSERT_GT(required, 0) << "GemmFwdRest needs Im2Col workspace for this problem";

    const auto queried = problem.QueriedWorkspace(&handle);
    EXPECT_GE(queried, required);
    EXPECT_EQ(problem.FindForward(handle, queried), miopenConvolutionFwdAlgoGEMM);
#endif
}

// With twice the requirement free, the dilated GemmFwdRest workspace is reported.
TEST(GPU_ConvDilatedGemmWorkspace_FP32, WorkspaceFittingFreeMemoryIsReported)
{
#if !MIOPEN_USE_GEMM
    GTEST_SKIP() << "Built without GEMM support";
#else
    ScopedFindDb no_find_db;
    ScopedEnvironment<std::string> only_gemm(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                             std::string{"GemmFwdRest"});

    auto problem        = LargeDilatedConv();
    const auto required = problem.GemmFwdRestWorkspace(&get_handle());
    ASSERT_GT(required, 0) << "GemmFwdRest needs Im2Col workspace for this problem";

    AvailableMemoryHandle handle{2 * required};
    EXPECT_EQ(problem.QueriedWorkspace(&handle), required);
#endif
}

// With only half the requirement free, the dilated GemmFwdRest workspace is not reported.
TEST(GPU_ConvDilatedGemmWorkspace_FP32, WorkspaceExceedingFreeMemoryIsNotReported)
{
#if !MIOPEN_USE_GEMM
    GTEST_SKIP() << "Built without GEMM support";
#else
    ScopedFindDb no_find_db;
    ScopedEnvironment<std::string> only_gemm(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                             std::string{"GemmFwdRest"});

    auto problem        = LargeDilatedConv();
    const auto required = problem.GemmFwdRestWorkspace(&get_handle());
    ASSERT_GT(required, 0) << "GemmFwdRest needs Im2Col workspace for this problem";

    AvailableMemoryHandle handle{required / 2};
    EXPECT_EQ(problem.QueriedWorkspace(&handle), 0);
#endif
}

// A find-db record of GemmFwdRest must not make the query report a workspace that does not fit in
// free memory: the record is honoured while it fits and skipped once it does not.
TEST(GPU_ConvDilatedGemmWorkspace_FP32, FindDbRecordDoesNotBypassFreeMemoryBudget)
{
#if !MIOPEN_USE_GEMM
    GTEST_SKIP() << "Built without GEMM support";
#else
    auto& handle = get_handle();

    const miopen::TempFile find_db{"miopen.test.conv_dilated_gemm_workspace"};
    const miopen::FindDbPathOverride find_db_override{find_db.Path()};
    ScopedEnvironment<std::string> only_gemm(MIOPEN_DEBUG_FIND_ONLY_SOLVER,
                                             std::string{"GemmFwdRest"});

    auto problem        = ReproDilatedConv();
    const auto required = problem.GemmFwdRestWorkspace(&handle);
    ASSERT_GT(required, 0) << "GemmFwdRest needs Im2Col workspace for this problem";
    ASSERT_EQ(problem.FindForward(handle, required), miopenConvolutionFwdAlgoGEMM)
        << "Find did not record GemmFwdRest";

    AvailableMemoryHandle roomy{2 * required};
    EXPECT_EQ(problem.QueriedWorkspace(&roomy), required);

    AvailableMemoryHandle tight{required / 2};
    EXPECT_EQ(problem.QueriedWorkspace(&tight), 0);
#endif
}
