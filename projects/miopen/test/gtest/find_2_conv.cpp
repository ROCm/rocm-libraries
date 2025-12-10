// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "../driver.hpp"
#include "../lib_env_var.hpp"
#include "../workspace.hpp"

#include <miopen/miopen.h>
#include <miopen/convolution.hpp>
#include <miopen/solution.hpp>
#include <miopen/solver_id.hpp>
#include <nlohmann/json.hpp>

#include <vector>

namespace {

struct Find2ConvTest : test_driver
{
    tensor<float> x;
    tensor<float> w;
    tensor<float> y;
    miopen::Allocator::ManageDataPtr x_dev;
    miopen::Allocator::ManageDataPtr w_dev;
    miopen::Allocator::ManageDataPtr y_dev;

    miopenProblemDirection_t direction = miopenProblemDirectionForward;
    // --input 16,192,28,28 --weights 32,192,5,5 --filter 2,2,1,1,1,1,
    miopen::ConvolutionDescriptor filter = {
        2, miopenConvolution, miopenPaddingDefault, {1, 1}, {1, 1}, {1, 1}};
    int tune;
    bool preallocate;
    std::size_t workspace_limit;
    bool attach_binaries;

    Find2ConvTest()
    {
        add(attach_binaries, "attach_binaries", generate_data({0, 1}));

        add(direction,
            "direction",
            generate_data({
                miopenProblemDirectionForward,
                miopenProblemDirectionBackward,
                miopenProblemDirectionBackwardWeights,
            }));

        add(workspace_limit,
            "workspace_limit",
            generate_data({
                std::numeric_limits<std::size_t>::max(),
                static_cast<size_t>(0),
            }));

        add(tune, "tune", generate_data({0, 1}));
        add(preallocate, "preallocate", generate_data({0, 1}));
    }

    void run()
    {
        ReleaseMemory();
        GenerateTensors();
        TestConv();
    }

private:
    void ReleaseMemory()
    {
        x_dev = nullptr;
        w_dev = nullptr;
        y_dev = nullptr;

        x = {};
        w = {};
        y = {};
    }

    void GenerateTensors()
    {
        auto& handle_deref = get_handle();

        x = tensor<float>{16, 192, 28, 28}.generate(tensor_elem_gen_integer{17});
        w = tensor<float>{32, 192, 5, 5}.generate(tensor_elem_gen_integer{17});
        y = tensor<float>{filter.GetForwardOutputTensor(x.desc, w.desc)};

        x_dev = handle_deref.Write(x.data);
        w_dev = handle_deref.Write(w.data);
        y_dev = handle_deref.Write(y.data);
    }

    void TestConv()
    {
        miopenHandle_t handle = &get_handle();
        miopenProblem_t problem;

        EXPECT_EQ(miopenCreateConvProblem(&problem, &filter, direction), miopenStatusSuccess);

        AddConvTensorDescriptors(problem);

        std::ignore          = TestFindSolutions(handle, problem);
        const auto solutions = TestFindSolutionsWithOptions(handle, problem);

        TestSolutionAttributes(solutions);
        TestRunSolutions(handle, solutions);

        EXPECT_EQ(miopenDestroyProblem(problem), miopenStatusSuccess);
    }

    void AddConvTensorDescriptors(miopenProblem_t problem)
    {
        std::cerr << "Creating conv tensor descriptos..." << std::endl;

        auto test_set_tensor_descriptor = [problem](miopenTensorArgumentId_t name,
                                                    miopen::TensorDescriptor& desc) {
            EXPECT_EQ(miopenSetProblemTensorDescriptor(problem, name, &desc), miopenStatusSuccess);
        };

        test_set_tensor_descriptor(miopenTensorConvolutionX, x.desc);
        test_set_tensor_descriptor(miopenTensorConvolutionW, w.desc);
        test_set_tensor_descriptor(miopenTensorConvolutionY, y.desc);

        // adding x descriptor again to validate that error is produced
        EXPECT_EQ(miopenSetProblemTensorDescriptor(problem, miopenTensorConvolutionX, &x.desc),
                  miopenStatusBadParm);

        std::cerr << "Created conv tensor descriptos." << std::endl;
    }

    std::vector<miopenSolution_t> TestFindSolutions(miopenHandle_t handle, miopenProblem_t problem)
    {
        std::cerr << "Testing miopenFindSolutions..." << std::endl;

        auto solutions = std::vector<miopenSolution_t>{};
        std::size_t found;

        solutions.resize(100);

        EXPECT_EQ(miopenFindSolutions(
                      handle, problem, nullptr, solutions.data(), &found, solutions.size()),
                  miopenStatusSuccess);
        EXPECT_GE(found, 0);

        solutions.resize(found);

        std::cerr << "Finished testing miopenFindSolutions." << std::endl;
        return solutions;
    }

    std::vector<miopenSolution_t> TestFindSolutionsWithOptions(miopenHandle_t handle,
                                                               miopenProblem_t problem)
    {
        std::cerr << "Testing miopenFindSolutions with options..." << std::endl;

        auto solutions    = std::vector<miopenSolution_t>{};
        std::size_t found = 0;

        solutions.resize(100);

        {
            miopenFindOptions_t options;

            EXPECT_EQ(miopenCreateFindOptions(&options), miopenStatusSuccess);

            EXPECT_EQ(miopenSetFindOptionTuning(options, tune), miopenStatusSuccess);
            EXPECT_EQ(miopenSetFindOptionResultsOrder(options, miopenFindResultsOrderByTime),
                      miopenStatusSuccess);
            EXPECT_EQ(miopenSetFindOptionWorkspaceLimit(options, workspace_limit),
                      miopenStatusSuccess);
            EXPECT_EQ(miopenSetFindOptionAttachBinaries(options, attach_binaries),
                      miopenStatusSuccess);

            miopen::Allocator::ManageDataPtr workspace_dev;

            if(preallocate)
            {
                std::size_t workspace_max = 0;
                switch(direction)
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

                const auto workspace_size = std::min(workspace_limit, workspace_max);
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

            std::cerr << "Testing with: ";
            std::cerr << (tune != 0 ? "tuning" : "no tuning") << ", ";
            std::cerr << (attach_binaries ? "attached binaries" : "no binaries") << ", ";
            std::cerr << workspace_limit << " ws limit";

            EXPECT_EQ(miopenFindSolutions(
                          handle, problem, options, solutions.data(), &found, solutions.size()),
                      miopenStatusSuccess);

            EXPECT_EQ(miopenDestroyFindOptions(options), miopenStatusSuccess);
        }

        EXPECT_GE(found, 0);
        solutions.resize(found);

        std::cerr << "Finished testing miopenFindSolutions with options." << std::endl;
        return solutions;
    }

    void TestSolutionAttributes(const std::vector<miopenSolution_t>& solutions)
    {
        std::cerr << "Testing miopenGetSolution<Attribute>..." << std::endl;

        for(const auto& solution : solutions)
        {
            float time;
            std::size_t workspace_size;
            uint64_t solver_id;
            miopenConvAlgorithm_t algo;

            EXPECT_EQ(miopenGetSolutionTime(solution, &time), miopenStatusSuccess);
            EXPECT_EQ(miopenGetSolutionWorkspaceSize(solution, &workspace_size),
                      miopenStatusSuccess);
            EXPECT_EQ(miopenGetSolutionSolverId(solution, &solver_id), miopenStatusSuccess);
            EXPECT_EQ(miopenGetSolverIdConvAlgorithm(solver_id, &algo), miopenStatusSuccess);
        }

        std::cerr << "Finished testing miopenGetSolution<Attribute>." << std::endl;
    }

    void TestRunSolutions(miopenHandle_t handle, const std::vector<miopenSolution_t>& solutions)
    {
        std::cerr << "Testing solution functions..." << std::endl;

        miopenTensorDescriptor_t x_desc = &x.desc, w_desc = &w.desc, y_desc = &y.desc;

        for(const auto& solution : solutions)
        {
            uint64_t solver_id;
            EXPECT_EQ(miopenGetSolutionSolverId(solution, &solver_id), miopenStatusSuccess);
            std::cerr << "Testing solver " << solver_id << " ("
                      << miopen::solver::Id(solver_id).ToString() << ")" << std::endl;

            miopenTensorArgumentId_t names[3] = {
                miopenTensorConvolutionX, miopenTensorConvolutionW, miopenTensorConvolutionY};
            void* buffers[3]                        = {x_dev.get(), w_dev.get(), y_dev.get()};
            miopenTensorDescriptor_t descriptors[3] = {x_desc, w_desc, y_desc};

            TestRunSolution(handle, solution, 3, names, descriptors, buffers);

            // Save-load cycle
            std::cerr << "Testing miopenGetSolutionSize..." << std::endl;
            std::size_t solution_size;
            EXPECT_EQ(miopenGetSolutionSize(solution, &solution_size), miopenStatusSuccess);
            EXPECT_GT(solution_size, 0);

            auto solution_binary = std::vector<char>{};
            solution_binary.resize(solution_size);

            std::cerr << "Testing miopenSaveSolution..." << std::endl;
            EXPECT_EQ(miopenSaveSolution(solution, solution_binary.data()), miopenStatusSuccess);
            std::cerr << "Destroying original solution..." << std::endl;
            EXPECT_EQ(miopenDestroySolution(solution), miopenStatusSuccess);

            std::cerr << "Testing miopenLoadSolution..." << std::endl;
            miopenSolution_t read_solution;
            EXPECT_EQ(
                miopenLoadSolution(&read_solution, solution_binary.data(), solution_binary.size()),
                miopenStatusSuccess);

            TestRunSolution(handle, read_solution, 3, names, descriptors, buffers);
            EXPECT_EQ(miopenDestroySolution(read_solution), miopenStatusSuccess);
        }

        std::cerr << "Finished testing solution functions." << std::endl;
    }

    void TestRunSolution(miopenHandle_t handle,
                         miopenSolution_t solution,
                         std::size_t num_arguments,
                         const miopenTensorArgumentId_t* names,
                         miopenTensorDescriptor_t* descriptors,
                         void** buffers)
    {
        std::cerr << "Running a solution..." << std::endl;

        std::size_t workspace_size;
        EXPECT_EQ(miopenGetSolutionWorkspaceSize(solution, &workspace_size), miopenStatusSuccess);

        Workspace wspace{workspace_size};

        const auto checked_run_solution = [&](miopenTensorDescriptor_t* descriptors_) {
            auto arguments = std::make_unique<miopenTensorArgument_t[]>(num_arguments);

            for(auto i = 0; i < num_arguments; ++i)
            {
                arguments[i].id         = names[i];
                arguments[i].descriptor = descriptors_ != nullptr ? &descriptors_[i] : nullptr;
                arguments[i].buffer     = buffers[i];
            }

            EXPECT_EQ(miopenRunSolution(
                          handle, solution, 3, arguments.get(), wspace.ptr(), wspace.size()),
                      miopenStatusSuccess);
        };

        // Without descriptors
        checked_run_solution(nullptr);
        // With descriptors
        checked_run_solution(descriptors);

        std::cerr << "Ran a solution." << std::endl;
    }
};

void RunFind2ConvTests()
{
    Find2ConvTest test;
    test.full_set          = false;
    test.dataset_id        = 0;
    test.config_iter_start = 0;

    std::vector<typename Find2ConvTest::argument*> data_args;
    for(auto&& arg : test.arguments)
    {
        data_args.push_back(&arg);
    }

    test.iteration = 0;
    try
    {
        run_data(data_args.begin(), data_args.end(), [&] { test.run(); });
    }
    catch(const std::exception& e)
    {
        FAIL() << "Exception in find_2_conv test: " << e.what();
    }
    catch(...)
    {
        FAIL() << "Unknown exception in find_2_conv test";
    }
}

bool IsTestSupportedForDevice(const miopen::Handle& handle) { return true; }

} // namespace

class GPU_Find2Conv_FP32 : public testing::TestWithParam<int>
{
    void SetUp() override
    {
        prng::reset_seed();
        const auto& handle = get_handle();
        if(!IsTestSupportedForDevice(handle))
        {
            GTEST_SKIP();
        }
        // Set up environment variables
        lib_env::update(MIOPEN_LOG_LEVEL, 2);
    }
};

TEST_P(GPU_Find2Conv_FP32, FloatTest_find_2_conv) { RunFind2ConvTests(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_Find2Conv_FP32, testing::ValuesIn({0}));
