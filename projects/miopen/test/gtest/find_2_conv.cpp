// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "../lib_env_var.hpp"
#include "../tensor_holder.hpp"
#include "../workspace.hpp"

#include <miopen/miopen.h>
#include <miopen/convolution.hpp>
#include <miopen/solution.hpp>
#include <miopen/solver_id.hpp>
#include <nlohmann/json.hpp>

#include <vector>
#include <limits>

namespace {

struct Find2ConvTestCase
{
    miopenProblemDirection_t direction;
    int tune;
    bool preallocate;
    std::size_t workspace_limit;
    bool attach_binaries;

    friend std::ostream& operator<<(std::ostream& os, const Find2ConvTestCase& tc)
    {
        return os << "direction:" << tc.direction << " tune:" << tc.tune
                  << " preallocate:" << tc.preallocate << " ws_limit:" << tc.workspace_limit
                  << " attach_binaries:" << tc.attach_binaries;
    }
};

std::vector<Find2ConvTestCase> GetFind2ConvTestCases()
{
    return {
        // direction, tune, preallocate, workspace_limit, attach_binaries
        {miopenProblemDirectionForward, 0, false, std::numeric_limits<std::size_t>::max(), false},
        {miopenProblemDirectionForward, 1, false, 0, true},
        {miopenProblemDirectionBackward, 0, true, std::numeric_limits<std::size_t>::max(), false},
        {miopenProblemDirectionBackwardWeights, 1, false, 0, false},
    };
}

void RunFind2ConvTest(const Find2ConvTestCase& test_case)
{
    auto& handle_deref = get_handle();

    // Generate tensors
    tensor<float> x{16, 192, 28, 28};
    x.generate(tensor_elem_gen_integer{17});
    tensor<float> w{32, 192, 5, 5};
    w.generate(tensor_elem_gen_integer{17});

    miopen::ConvolutionDescriptor filter = {
        2, miopenConvolution, miopenPaddingDefault, {1, 1}, {1, 1}, {1, 1}};
    tensor<float> y{filter.GetForwardOutputTensor(x.desc, w.desc)};

    auto x_dev = handle_deref.Write(x.data);
    auto w_dev = handle_deref.Write(w.data);
    auto y_dev = handle_deref.Write(y.data);

    // Test convolution
    miopenHandle_t handle = &handle_deref;
    miopenProblem_t problem;

    EXPECT_EQ(miopenCreateConvProblem(&problem, &filter, test_case.direction), miopenStatusSuccess);

    // Add tensor descriptors
    EXPECT_EQ(miopenSetProblemTensorDescriptor(problem, miopenTensorConvolutionX, &x.desc),
              miopenStatusSuccess);
    EXPECT_EQ(miopenSetProblemTensorDescriptor(problem, miopenTensorConvolutionW, &w.desc),
              miopenStatusSuccess);
    EXPECT_EQ(miopenSetProblemTensorDescriptor(problem, miopenTensorConvolutionY, &y.desc),
              miopenStatusSuccess);

    // Adding x descriptor again to validate that error is produced
    EXPECT_EQ(miopenSetProblemTensorDescriptor(problem, miopenTensorConvolutionX, &x.desc),
              miopenStatusBadParm);

    // Test FindSolutions
    auto solutions    = std::vector<miopenSolution_t>{};
    std::size_t found = 0;
    solutions.resize(100);

    EXPECT_EQ(
        miopenFindSolutions(handle, problem, nullptr, solutions.data(), &found, solutions.size()),
        miopenStatusSuccess);
    EXPECT_GE(found, 0);
    solutions.resize(found);

    // Test FindSolutions with options
    solutions.resize(100);
    found = 0;

    {
        miopenFindOptions_t options;
        EXPECT_EQ(miopenCreateFindOptions(&options), miopenStatusSuccess);

        EXPECT_EQ(miopenSetFindOptionTuning(options, test_case.tune), miopenStatusSuccess);
        EXPECT_EQ(miopenSetFindOptionResultsOrder(options, miopenFindResultsOrderByTime),
                  miopenStatusSuccess);
        EXPECT_EQ(miopenSetFindOptionWorkspaceLimit(options, test_case.workspace_limit),
                  miopenStatusSuccess);
        EXPECT_EQ(miopenSetFindOptionAttachBinaries(options, test_case.attach_binaries),
                  miopenStatusSuccess);

        if(test_case.preallocate)
        {
            std::size_t workspace_max = 0;
            switch(test_case.direction)
            {
            case miopenProblemDirectionForward:
                EXPECT_EQ(miopenConvolutionForwardGetWorkSpaceSize(
                              handle, &x.desc, &w.desc, &filter, &y.desc, &workspace_max),
                          miopenStatusSuccess);
                break;
            case miopenProblemDirectionBackward:
                EXPECT_EQ(miopenConvolutionBackwardDataGetWorkSpaceSize(
                              handle, &y.desc, &w.desc, &filter, &x.desc, &workspace_max),
                          miopenStatusSuccess);
                break;
            case miopenProblemDirectionBackwardWeights:
                EXPECT_EQ(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
                              handle, &y.desc, &x.desc, &filter, &w.desc, &workspace_max),
                          miopenStatusSuccess);
                break;
            default: MIOPEN_THROW(miopenStatusNotImplemented);
            }

            const auto workspace_size = std::min(test_case.workspace_limit, workspace_max);
            Workspace wspace{workspace_size};

            EXPECT_EQ(
                miopenSetFindOptionPreallocatedWorkspace(options, wspace.ptr(), wspace.size()),
                miopenStatusSuccess);

            EXPECT_EQ(miopenSetFindOptionPreallocatedTensor(
                          options, miopenTensorConvolutionX, x_dev.get()),
                      miopenStatusSuccess);

            EXPECT_EQ(miopenSetFindOptionPreallocatedTensor(
                          options, miopenTensorConvolutionW, w_dev.get()),
                      miopenStatusSuccess);

            EXPECT_EQ(miopenSetFindOptionPreallocatedTensor(
                          options, miopenTensorConvolutionY, y_dev.get()),
                      miopenStatusSuccess);
        }

        EXPECT_EQ(miopenFindSolutions(
                      handle, problem, options, solutions.data(), &found, solutions.size()),
                  miopenStatusSuccess);

        EXPECT_EQ(miopenDestroyFindOptions(options), miopenStatusSuccess);
    }

    EXPECT_GE(found, 0);
    solutions.resize(found);

    // Test solution attributes
    for(const auto& solution : solutions)
    {
        float time;
        std::size_t workspace_size;
        uint64_t solver_id;
        miopenConvAlgorithm_t algo;

        EXPECT_EQ(miopenGetSolutionTime(solution, &time), miopenStatusSuccess);
        EXPECT_EQ(miopenGetSolutionWorkspaceSize(solution, &workspace_size), miopenStatusSuccess);
        EXPECT_EQ(miopenGetSolutionSolverId(solution, &solver_id), miopenStatusSuccess);
        EXPECT_EQ(miopenGetSolverIdConvAlgorithm(solver_id, &algo), miopenStatusSuccess);
    }

    // Test running solutions
    miopenTensorDescriptor_t x_desc = &x.desc, w_desc = &w.desc, y_desc = &y.desc;

    for(const auto& solution : solutions)
    {
        uint64_t solver_id;
        EXPECT_EQ(miopenGetSolutionSolverId(solution, &solver_id), miopenStatusSuccess);

        miopenTensorArgumentId_t names[3] = {
            miopenTensorConvolutionX, miopenTensorConvolutionW, miopenTensorConvolutionY};
        void* buffers[3]                        = {x_dev.get(), w_dev.get(), y_dev.get()};
        miopenTensorDescriptor_t descriptors[3] = {x_desc, w_desc, y_desc};

        // Run solution
        std::size_t workspace_size;
        EXPECT_EQ(miopenGetSolutionWorkspaceSize(solution, &workspace_size), miopenStatusSuccess);

        Workspace wspace{workspace_size};

        auto arguments = std::make_unique<miopenTensorArgument_t[]>(3);
        for(auto i = 0; i < 3; ++i)
        {
            arguments[i].id         = names[i];
            arguments[i].descriptor = &descriptors[i];
            arguments[i].buffer     = buffers[i];
        }

        EXPECT_EQ(
            miopenRunSolution(handle, solution, 3, arguments.get(), wspace.ptr(), wspace.size()),
            miopenStatusSuccess);

        // Save-load cycle
        std::size_t solution_size;
        EXPECT_EQ(miopenGetSolutionSize(solution, &solution_size), miopenStatusSuccess);
        EXPECT_GT(solution_size, 0);

        auto solution_binary = std::vector<char>{};
        solution_binary.resize(solution_size);

        EXPECT_EQ(miopenSaveSolution(solution, solution_binary.data()), miopenStatusSuccess);
        EXPECT_EQ(miopenDestroySolution(solution), miopenStatusSuccess);

        miopenSolution_t read_solution;
        EXPECT_EQ(
            miopenLoadSolution(&read_solution, solution_binary.data(), solution_binary.size()),
            miopenStatusSuccess);

        // Run loaded solution
        auto read_arguments = std::make_unique<miopenTensorArgument_t[]>(3);
        for(auto i = 0; i < 3; ++i)
        {
            read_arguments[i].id         = names[i];
            read_arguments[i].descriptor = &descriptors[i];
            read_arguments[i].buffer     = buffers[i];
        }

        EXPECT_EQ(miopenRunSolution(
                      handle, read_solution, 3, read_arguments.get(), wspace.ptr(), wspace.size()),
                  miopenStatusSuccess);
        EXPECT_EQ(miopenDestroySolution(read_solution), miopenStatusSuccess);
    }

    EXPECT_EQ(miopenDestroyProblem(problem), miopenStatusSuccess);
}

} // namespace

class GPU_Find2Conv_FP32 : public testing::TestWithParam<Find2ConvTestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        if(!IsTestSupportedByDevice(Gpu::All))
        {
            GTEST_SKIP();
        }
        // Set up environment variables
        lib_env::update(MIOPEN_LOG_LEVEL, 2);
    }
};

TEST_P(GPU_Find2Conv_FP32, FloatTest_find_2_conv) { RunFind2ConvTest(GetParam()); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Find2Conv_FP32, testing::ValuesIn(GetFind2ConvTestCases()));
