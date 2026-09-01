// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Seed test for hipDNN-forwarding backend-swap coverage. Setup and execution go entirely
// through public miopen.h entry points so that swapping the implementation behind them is
// the only thing the test can observe, and the result is checked against an independent CPU
// reference rather than a second MIOpen run. The "Forwarding" token in the suite name is
// what selects it into the parity surface; see README.md. Built only under
// MIOPEN_ENABLE_HIPDNN_WRAPPER, which keeps ctest -N identical to the flag-off baseline.

#ifdef MIOPEN_ENABLE_HIPDNN_WRAPPER

#include <gtest/gtest.h>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "../cpu_conv.hpp"
#include "../verify.hpp"
#include "../workspace.hpp"

#include <miopen/miopen.h>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

TEST(GPU_ForwardingConvFwdApi_FP32, RunSolutionMatchesCpuReference)
{
    auto& handle_deref = get_handle();

    const std::vector<int> pads{1, 1};
    const std::vector<int> strides{1, 1};
    const std::vector<int> dilations{1, 1};
    const std::size_t group_count = 1;

    tensor<float> x{2, 4, 8, 8};
    x.generate(tensor_elem_gen_integer{17});
    tensor<float> w{4, 4, 3, 3};
    w.generate(tensor_elem_gen_integer{17});

    miopenConvolutionDescriptor_t conv_desc;
    ASSERT_EQ(miopenCreateConvolutionDescriptor(&conv_desc), miopenStatusSuccess);
    ASSERT_EQ(miopenInitConvolutionNdDescriptor(conv_desc,
                                                static_cast<int>(pads.size()),
                                                pads.data(),
                                                strides.data(),
                                                dilations.data(),
                                                miopenConvolution),
              miopenStatusSuccess);

    // Ask the library for the output shape rather than recomputing it here, so the shape is
    // part of what the two implementations must agree on.
    int out_dim_count = 0;
    std::vector<int> out_dims(4);
    ASSERT_EQ(miopenGetConvolutionNdForwardOutputDim(
                  conv_desc, &x.desc, &w.desc, &out_dim_count, out_dims.data()),
              miopenStatusSuccess);
    ASSERT_EQ(out_dim_count, 4);
    const std::vector<std::size_t> out_lengths(out_dims.begin(), out_dims.end());

    tensor<float> y{out_lengths};

    auto x_dev = handle_deref.Write(x.data);
    auto w_dev = handle_deref.Write(w.data);
    auto y_dev = handle_deref.Write(y.data);

    miopenHandle_t handle = &handle_deref;
    miopenProblem_t problem;
    ASSERT_EQ(miopenCreateConvProblem(&problem, conv_desc, miopenProblemDirectionForward),
              miopenStatusSuccess);
    ASSERT_EQ(miopenSetProblemTensorDescriptor(problem, miopenTensorConvolutionX, &x.desc),
              miopenStatusSuccess);
    ASSERT_EQ(miopenSetProblemTensorDescriptor(problem, miopenTensorConvolutionW, &w.desc),
              miopenStatusSuccess);
    ASSERT_EQ(miopenSetProblemTensorDescriptor(problem, miopenTensorConvolutionY, &y.desc),
              miopenStatusSuccess);

    std::vector<miopenSolution_t> solutions(1);
    std::size_t found = 0;
    ASSERT_EQ(
        miopenFindSolutions(handle, problem, nullptr, solutions.data(), &found, solutions.size()),
        miopenStatusSuccess);
    ASSERT_GT(found, 0);
    solutions.resize(found);

    std::size_t workspace_size;
    ASSERT_EQ(miopenGetSolutionWorkspaceSize(solutions[0], &workspace_size), miopenStatusSuccess);
    Workspace wspace{workspace_size};

    miopenTensorArgumentId_t names[3] = {
        miopenTensorConvolutionX, miopenTensorConvolutionW, miopenTensorConvolutionY};
    void* buffers[3]                        = {x_dev.get(), w_dev.get(), y_dev.get()};
    miopenTensorDescriptor_t descriptors[3] = {&x.desc, &w.desc, &y.desc};

    auto arguments = std::make_unique<miopenTensorArgument_t[]>(3);
    for(auto i = 0; i < 3; ++i)
    {
        arguments[i].id         = names[i];
        arguments[i].descriptor = &descriptors[i];
        arguments[i].buffer     = buffers[i];
    }

    ASSERT_EQ(
        miopenRunSolution(handle, solutions[0], 3, arguments.get(), wspace.ptr(), wspace.size()),
        miopenStatusSuccess);
    ASSERT_EQ(miopenDestroyProblem(problem), miopenStatusSuccess);

    y.data = handle_deref.Read<float>(y_dev, y.data.size());

    // Scaffolding, not code under test, so internal helpers are fine here; what matters is
    // that the reference is not another MIOpen solver.
    tensor<float> ref_y{out_lengths};
    cpu_convolution_forward(pads.size(), x, w, ref_y, pads, strides, dilations, group_count);

    ASSERT_EQ(miopenDestroyConvolutionDescriptor(conv_desc), miopenStatusSuccess);

    // Cross-implementation comparison, not bit-reproducibility: same tolerance used by
    // ConvFwdSolverTestBase::ThresholdChecks() for FP32.
    const double tolerance = std::numeric_limits<float>::epsilon() * 80;
    const double error     = miopen::rms_range(ref_y, y);
    EXPECT_TRUE(std::isfinite(error));
    EXPECT_LT(error, tolerance) << "Forwarding seed test error beyond tolerance";
}

#endif // MIOPEN_ENABLE_HIPDNN_WRAPPER
