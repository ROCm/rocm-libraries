// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Immediate-mode GetSolution used to copy workspace_size straight from the find-db record.
// Shipped find-db records can hold a smaller value than the solver needs (ROCM-30131:
// gfx120140 stores 0 for GemmFwd1x1_0_2 at 1024-14-14-1x1-2048-7-7-128 stride 2 FP16), so
// frameworks that allocate exactly the reported size got "Not enough workspace for GEMM" from
// Immediate. These tests use temporary find-db records to check that GetSolution reports enough
// workspace for the configuration Immediate will run, retains a larger stored value, and executes
// correctly with the size it reports.

#include <miopen/any_solver.hpp>
#include <miopen/conv/db_getter.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/miopen.h>
#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/db_record.hpp>
#include <miopen/find_db.hpp>
#include <miopen/readonlyramdb.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/temp_file.hpp>
#include <miopen/tensor.hpp>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr const char* solver_name           = "GemmFwd1x1_0_2";
constexpr const char* group_fwd_solver_name = "ConvHipImplicitGemmGroupFwdXdlops";
constexpr const char* implicit_gemm_algo    = "miopenConvolutionFwdAlgoImplicitGEMM";

struct FindDbOverride
{
    explicit FindDbOverride(const miopen::fs::path& path)
        : cached_path(miopen::debug::testing_find_db_path_override()),
          cached_embed(miopen::debug::rordb_embed_fs_override())
    {
        miopen::debug::testing_find_db_path_override() = path;
        miopen::debug::rordb_embed_fs_override()       = true;
    }
    ~FindDbOverride()
    {
        miopen::debug::testing_find_db_path_override() = cached_path;
        miopen::debug::rordb_embed_fs_override()       = cached_embed;
    }
    FindDbOverride(const FindDbOverride&)            = delete;
    FindDbOverride& operator=(const FindDbOverride&) = delete;

private:
    std::optional<miopen::fs::path> cached_path;
    bool cached_embed;
};

struct DeviceBuffer
{
    void* ptr = nullptr;
    explicit DeviceBuffer(std::size_t bytes)
    {
        if(bytes != 0 && hipMalloc(&ptr, bytes) != hipSuccess)
            ptr = nullptr;
    }
    ~DeviceBuffer()
    {
        if(ptr != nullptr)
            (void)hipFree(ptr);
    }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

class GPU_ConvApiGetSolutionWorkspace_FP16 : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(miopenCreateWithStream(&handle, nullptr), miopenStatusSuccess);
        ASSERT_EQ(miopenCreateTensorDescriptor(&xDesc), miopenStatusSuccess);
        ASSERT_EQ(miopenCreateTensorDescriptor(&wDesc), miopenStatusSuccess);
        ASSERT_EQ(miopenCreateTensorDescriptor(&yDesc), miopenStatusSuccess);
        ASSERT_EQ(miopenCreateConvolutionDescriptor(&convDesc), miopenStatusSuccess);

        // 1x1, stride 2, pad 0: the GemmFwd1x1_0_2 path, which needs workspace for the
        // transposed x and y copies.
        ASSERT_EQ(miopenSet4dTensorDescriptor(xDesc, miopenHalf, 2, 16, 8, 8), miopenStatusSuccess);
        ASSERT_EQ(miopenSet4dTensorDescriptor(wDesc, miopenHalf, 32, 16, 1, 1),
                  miopenStatusSuccess);
        ASSERT_EQ(miopenInitConvolutionDescriptor(convDesc, miopenConvolution, 0, 0, 2, 2, 1, 1),
                  miopenStatusSuccess);
        int n = 0, c = 0, h = 0, w = 0;
        ASSERT_EQ(miopenGetConvolutionForwardOutputDim(convDesc, xDesc, wDesc, &n, &c, &h, &w),
                  miopenStatusSuccess);
        ASSERT_EQ(miopenSet4dTensorDescriptor(yDesc, miopenHalf, n, c, h, w), miopenStatusSuccess);
        x_bytes = 2 * 16 * 8 * 8 * sizeof(uint16_t);
        w_bytes = 32 * 16 * sizeof(uint16_t);
        y_bytes = static_cast<std::size_t>(n) * c * h * w * sizeof(uint16_t);
    }

    void TearDown() override
    {
        miopenDestroyConvolutionDescriptor(convDesc);
        miopenDestroyTensorDescriptor(yDesc);
        miopenDestroyTensorDescriptor(wDesc);
        miopenDestroyTensorDescriptor(xDesc);
        miopenDestroy(handle);
    }

    // Writes a one-record find-db in the shipped text format.
    void WriteFindDb(const miopen::fs::path& path,
                     const char* selected_solver,
                     const char* algorithm,
                     std::size_t db_workspace) const
    {
        const auto problem = miopen::conv::ProblemDescription{miopen::deref(xDesc),
                                                              miopen::deref(wDesc),
                                                              miopen::deref(yDesc),
                                                              miopen::deref(convDesc),
                                                              miopen::conv::Direction::Forward};
        const auto record  = miopen::DbRecord{miopen::DbKinds::FindDb, problem};
        std::ofstream file(path);
        file << record.GetKey() << '=' << selected_solver << ":0.5," << db_workspace << ','
             << algorithm << '\n';
    }

    std::vector<miopenConvSolution_t> GetSolutions() const
    {
        std::size_t count = 0;
        EXPECT_EQ(
            miopenConvolutionForwardGetSolutionCount(handle, wDesc, xDesc, convDesc, yDesc, &count),
            miopenStatusSuccess);
        std::vector<miopenConvSolution_t> solutions(count);
        std::size_t returned = 0;
        EXPECT_EQ(miopenConvolutionForwardGetSolution(
                      handle, wDesc, xDesc, convDesc, yDesc, count, &returned, solutions.data()),
                  miopenStatusSuccess);
        solutions.resize(returned);
        return solutions;
    }

    void Check(const char* selected_solver,
               const char* algorithm,
               std::size_t db_workspace,
               std::optional<std::size_t> expected_reported,
               bool verify_grouped_forward = false)
    {
        const miopen::TempFile find_db{"miopen.test.get_solution_workspace"};
        WriteFindDb(find_db, selected_solver, algorithm, db_workspace);
        const FindDbOverride override_guard{find_db};

        const auto solver_id = miopen::solver::Id{selected_solver}.Value();
        const auto solutions = GetSolutions();
        ASSERT_EQ(solutions.size(), 1) << "expected only the record from the temporary find-db";
        ASSERT_EQ(solutions[0].solution_id, solver_id);
        if(expected_reported)
            ASSERT_EQ(solutions[0].workspace_size, *expected_reported);
        else
            ASSERT_GT(solutions[0].workspace_size, db_workspace);

        // What a framework does next: allocate exactly the reported size and run Immediate.
        DeviceBuffer x{x_bytes}, w{w_bytes}, y{y_bytes}, ws{solutions[0].workspace_size};
        ASSERT_NE(x.ptr, nullptr);
        ASSERT_NE(w.ptr, nullptr);
        ASSERT_NE(y.ptr, nullptr);
        if(verify_grouped_forward)
        {
            const std::vector<uint16_t> input(x_bytes / sizeof(uint16_t), 0x3c00);
            const std::vector<uint16_t> weights(w_bytes / sizeof(uint16_t), 0x3c00);
            ASSERT_EQ(hipMemcpy(x.ptr, input.data(), x_bytes, hipMemcpyHostToDevice), hipSuccess);
            ASSERT_EQ(hipMemcpy(w.ptr, weights.data(), w_bytes, hipMemcpyHostToDevice), hipSuccess);
        }
        else
        {
            ASSERT_EQ(hipMemset(x.ptr, 0, x_bytes), hipSuccess);
            ASSERT_EQ(hipMemset(w.ptr, 0, w_bytes), hipSuccess);
        }
        ASSERT_EQ(miopenConvolutionForwardCompileSolution(
                      handle, wDesc, xDesc, convDesc, yDesc, solver_id),
                  miopenStatusSuccess);
        ASSERT_EQ(miopenConvolutionForwardImmediate(handle,
                                                    wDesc,
                                                    w.ptr,
                                                    xDesc,
                                                    x.ptr,
                                                    convDesc,
                                                    yDesc,
                                                    y.ptr,
                                                    ws.ptr,
                                                    solutions[0].workspace_size,
                                                    solver_id),
                  miopenStatusSuccess);
        if(verify_grouped_forward)
        {
            std::vector<uint16_t> output(y_bytes / sizeof(uint16_t));
            ASSERT_EQ(hipMemcpy(output.data(), y.ptr, y_bytes, hipMemcpyDeviceToHost), hipSuccess);
            for(const auto value : output)
                EXPECT_EQ(value, 0x4c00) << "each output should sum 16 half-precision ones";
        }
    }

    // Live requirement, or 0 with a skip reason when the solver is not available here.
    bool LiveWorkspace(std::size_t& live, const char* selected_solver = solver_name)
    {
        const auto solver_id = miopen::solver::Id{selected_solver}.Value();
        return miopenConvolutionForwardGetSolutionWorkspaceSize(
                   handle, wDesc, xDesc, convDesc, yDesc, solver_id, &live) == miopenStatusSuccess;
    }
    std::optional<std::size_t> SelectedWorkspace(const char* selected_solver) const
    {
        auto& handle_ref = miopen::deref(handle);
        auto ctx         = miopen::ExecutionContext{&handle_ref};
        auto problem     = miopen::conv::ProblemDescription{miopen::deref(xDesc),
                                                        miopen::deref(wDesc),
                                                        miopen::deref(yDesc),
                                                        miopen::deref(convDesc),
                                                        miopen::conv::Direction::Forward};
        problem.SetupFloats(ctx);
        problem.SetupComputeType(ctx);
        ctx.do_search              = false;
        ctx.disable_search_enforce = true;

        const auto solver   = miopen::solver::Id{selected_solver}.GetSolver();
        auto db             = miopen::MakeConvDbGetter(ctx);
        const auto solution = solver.FindSolution(ctx, problem, db, {});
        if(!solution.Succeeded())
            return std::nullopt;
        return solution.workspace_sz;
    }

    void SetGroupFwdShape()
    {
        ASSERT_EQ(miopenSet4dTensorDescriptor(xDesc, miopenHalf, 2, 32, 8, 8), miopenStatusSuccess);
        ASSERT_EQ(miopenSet4dTensorDescriptor(wDesc, miopenHalf, 64, 16, 1, 1),
                  miopenStatusSuccess);
        ASSERT_EQ(miopenInitConvolutionDescriptor(convDesc, miopenConvolution, 0, 0, 2, 2, 1, 1),
                  miopenStatusSuccess);
        ASSERT_EQ(miopenSetConvolutionGroupCount(convDesc, 2), miopenStatusSuccess);
        int n = 0, c = 0, h = 0, w = 0;
        ASSERT_EQ(miopenGetConvolutionForwardOutputDim(convDesc, xDesc, wDesc, &n, &c, &h, &w),
                  miopenStatusSuccess);
        ASSERT_EQ(miopenSet4dTensorDescriptor(yDesc, miopenHalf, n, c, h, w), miopenStatusSuccess);
        x_bytes = 2 * 32 * 8 * 8 * sizeof(uint16_t);
        w_bytes = 64 * 16 * sizeof(uint16_t);
        y_bytes = static_cast<std::size_t>(n) * c * h * w * sizeof(uint16_t);
    }

    miopenHandle_t handle                  = nullptr;
    miopenTensorDescriptor_t xDesc         = nullptr;
    miopenTensorDescriptor_t wDesc         = nullptr;
    miopenTensorDescriptor_t yDesc         = nullptr;
    miopenConvolutionDescriptor_t convDesc = nullptr;
    std::size_t x_bytes = 0, w_bytes = 0, y_bytes = 0;
};

} // namespace

TEST_F(GPU_ConvApiGetSolutionWorkspace_FP16, ZeroInFindDbReportsLiveRequirement)
{
    std::size_t live = 0;
    if(!LiveWorkspace(live))
        GTEST_SKIP() << solver_name << " is not applicable on this device or build";
    // x transposed to C x N x 4 x 4 plus y in CNHW: (2*16*16 + 2*32*16) halfs.
    ASSERT_EQ(live, (2 * 16 * 16 + 2 * 32 * 16) * sizeof(uint16_t));
    Check(solver_name, "miopenConvolutionFwdAlgoGEMM", 0, live);
}

TEST_F(GPU_ConvApiGetSolutionWorkspace_FP16, LargerFindDbValueIsKept)
{
    std::size_t live = 0;
    if(!LiveWorkspace(live))
        GTEST_SKIP() << solver_name << " is not applicable on this device or build";
    Check(solver_name, "miopenConvolutionFwdAlgoGEMM", live + 4096, live + 4096);
}

TEST_F(GPU_ConvApiGetSolutionWorkspace_FP16, GroupFwdNonzeroUndersizedWorkspaceExecutesCorrectly)
{
    SetGroupFwdShape();
    std::size_t live = 0;
    if(!LiveWorkspace(live, group_fwd_solver_name))
        GTEST_SKIP() << group_fwd_solver_name << " is not applicable on this device or build";
    ASSERT_GT(live, 1);
    const auto selected_workspace = SelectedWorkspace(group_fwd_solver_name);
    ASSERT_TRUE(selected_workspace);
    ASSERT_GT(*selected_workspace, 1);
    Check(group_fwd_solver_name, implicit_gemm_algo, 1, selected_workspace, true);
}
