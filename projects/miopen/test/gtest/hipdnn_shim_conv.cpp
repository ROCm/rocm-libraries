// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Convolution coverage for the hipDNN shim surface. Every call that selects or performs
// compute goes through a public miopen.h entry point, so that swapping the implementation
// behind them is the only thing these tests can observe; the handle and the host/device
// buffer staging around them come from the shared test infrastructure, which the wrapper
// does not sit in front of. Results are checked against an independent CPU reference rather
// than a second MIOpen run. The "HipdnnShim" token in each suite name is what selects them
// into the parity surface; see README.md. Built only under MIOPEN_ENABLE_HIPDNN_WRAPPER,
// which keeps ctest -N identical to the flag-off baseline.
//
// Convolution is covered through both public entry points into it, because they are separate
// code paths that will be swapped over to hipDNN independently.

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

namespace {

const std::vector<int> pads{1, 1};
const std::vector<int> strides{1, 1};
const std::vector<int> dilations{1, 1};
constexpr std::size_t group_count = 1;

// Small enough to stay cheap in a doubled replay, large enough that a wrong kernel cannot
// coincidentally match the reference.
tensor<float> MakeInput()
{
    tensor<float> x{2, 4, 8, 8};
    x.generate(tensor_elem_gen_integer{17});
    return x;
}

tensor<float> MakeWeights()
{
    tensor<float> w{4, 4, 3, 3};
    w.generate(tensor_elem_gen_integer{17});
    return w;
}

// Releases the handle on scope exit, so an ASSERT_* that stops a test early does not leak
// it into an ASAN lane.
template <class T, miopenStatus_t (*Destroy)(T)>
struct Owned
{
    Owned()                        = default;
    Owned(const Owned&)            = delete;
    Owned& operator=(const Owned&) = delete;
    ~Owned()
    {
        if(handle != nullptr)
            EXPECT_EQ(Destroy(handle), miopenStatusSuccess);
    }

    T handle = nullptr;
};

using OwnedConvDescriptor =
    Owned<miopenConvolutionDescriptor_t, miopenDestroyConvolutionDescriptor>;
using OwnedProblem = Owned<miopenProblem_t, miopenDestroyProblem>;

// Every solution the find call handed back, released together on scope exit. Declared after
// the vector it refers to so it runs before the vector goes away.
struct OwnedSolutions
{
    explicit OwnedSolutions(const std::vector<miopenSolution_t>& s) : solutions(s) {}
    OwnedSolutions(const OwnedSolutions&)            = delete;
    OwnedSolutions& operator=(const OwnedSolutions&) = delete;
    ~OwnedSolutions()
    {
        for(auto* solution : solutions)
        {
            if(solution != nullptr)
                EXPECT_EQ(miopenDestroySolution(solution), miopenStatusSuccess);
        }
    }

    const std::vector<miopenSolution_t>& solutions;
};

void InitConvDescriptor(OwnedConvDescriptor& conv)
{
    EXPECT_EQ(miopenCreateConvolutionDescriptor(&conv.handle), miopenStatusSuccess);
    EXPECT_EQ(miopenInitConvolutionNdDescriptor(conv.handle,
                                                static_cast<int>(pads.size()),
                                                pads.data(),
                                                strides.data(),
                                                dilations.data(),
                                                miopenConvolution),
              miopenStatusSuccess);
}

// Ask the library for the output shape rather than recomputing it here, so the shape is part
// of what the two implementations must agree on.
std::vector<std::size_t> OutputLengths(miopenConvolutionDescriptor_t conv_desc,
                                       const tensor<float>& x,
                                       const tensor<float>& w)
{
    int out_dim_count = 0;
    std::vector<int> out_dims(4);
    EXPECT_EQ(miopenGetConvolutionNdForwardOutputDim(
                  conv_desc, &x.desc, &w.desc, &out_dim_count, out_dims.data()),
              miopenStatusSuccess);
    EXPECT_EQ(out_dim_count, 4);
    return std::vector<std::size_t>(out_dims.begin(), out_dims.end());
}

// Scaffolding, not code under test, so internal helpers are fine here; what matters is that
// the reference is not another MIOpen solver.
void ExpectMatchesCpuReference(const tensor<float>& x, const tensor<float>& w, tensor<float>& y)
{
    tensor<float> ref_y{y.desc.GetLengths()};
    cpu_convolution_forward(pads.size(), x, w, ref_y, pads, strides, dilations, group_count);

    // Cross-implementation comparison, not bit-reproducibility: same tolerance used by
    // ConvFwdSolverTestBase::ThresholdChecks() for FP32.
    const double tolerance = std::numeric_limits<float>::epsilon() * 80;
    const double error     = miopen::rms_range(ref_y, y);
    EXPECT_TRUE(std::isfinite(error));
    EXPECT_LT(error, tolerance) << "convolution result beyond cross-implementation tolerance";
}

} // namespace

// The Find/Run pair: the older of the two public convolution paths, and the one most callers
// still use.
TEST(GPU_HipdnnShimConvFwdApi_FP32, FindAndForwardMatchCpuReference)
{
    auto& handle_deref    = get_handle();
    miopenHandle_t handle = &handle_deref;

    auto x = MakeInput();
    auto w = MakeWeights();
    OwnedConvDescriptor conv;
    InitConvDescriptor(conv);
    const auto out_lengths = OutputLengths(conv.handle, x, w);
    tensor<float> y{out_lengths};

    auto x_dev = handle_deref.Write(x.data);
    auto w_dev = handle_deref.Write(w.data);
    auto y_dev = handle_deref.Write(y.data);

    std::size_t workspace_size = 0;
    ASSERT_EQ(miopenConvolutionForwardGetWorkSpaceSize(
                  handle, &w.desc, &x.desc, conv.handle, &y.desc, &workspace_size),
              miopenStatusSuccess);
    Workspace wspace{workspace_size};

    int returned_algo_count = 0;
    miopenConvAlgoPerf_t perf{};
    ASSERT_EQ(miopenFindConvolutionForwardAlgorithm(handle,
                                                    &x.desc,
                                                    x_dev.get(),
                                                    &w.desc,
                                                    w_dev.get(),
                                                    conv.handle,
                                                    &y.desc,
                                                    y_dev.get(),
                                                    1,
                                                    &returned_algo_count,
                                                    &perf,
                                                    wspace.ptr(),
                                                    wspace.size(),
                                                    false),
              miopenStatusSuccess);
    ASSERT_GT(returned_algo_count, 0);

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    ASSERT_EQ(miopenConvolutionForward(handle,
                                       &alpha,
                                       &x.desc,
                                       x_dev.get(),
                                       &w.desc,
                                       w_dev.get(),
                                       conv.handle,
                                       perf.fwd_algo,
                                       &beta,
                                       &y.desc,
                                       y_dev.get(),
                                       wspace.ptr(),
                                       wspace.size()),
              miopenStatusSuccess);

    y.data = handle_deref.Read<float>(y_dev, y.data.size());

    ExpectMatchesCpuReference(x, w, y);
}

// The Problem/Solution path reaches the same convolution through different public entry
// points, so it has to be swapped over separately and is covered separately.
TEST(GPU_HipdnnShimConvSolutionApi_FP32, RunSolutionMatchesCpuReference)
{
    auto& handle_deref    = get_handle();
    miopenHandle_t handle = &handle_deref;

    auto x = MakeInput();
    auto w = MakeWeights();
    OwnedConvDescriptor conv;
    InitConvDescriptor(conv);
    const auto out_lengths = OutputLengths(conv.handle, x, w);
    tensor<float> y{out_lengths};

    auto x_dev = handle_deref.Write(x.data);
    auto w_dev = handle_deref.Write(w.data);
    auto y_dev = handle_deref.Write(y.data);

    OwnedProblem problem;
    ASSERT_EQ(miopenCreateConvProblem(&problem.handle, conv.handle, miopenProblemDirectionForward),
              miopenStatusSuccess);
    ASSERT_EQ(miopenSetProblemTensorDescriptor(problem.handle, miopenTensorConvolutionX, &x.desc),
              miopenStatusSuccess);
    ASSERT_EQ(miopenSetProblemTensorDescriptor(problem.handle, miopenTensorConvolutionW, &w.desc),
              miopenStatusSuccess);
    ASSERT_EQ(miopenSetProblemTensorDescriptor(problem.handle, miopenTensorConvolutionY, &y.desc),
              miopenStatusSuccess);

    std::vector<miopenSolution_t> solutions(1);
    OwnedSolutions owned_solutions{solutions};
    std::size_t found = 0;
    ASSERT_EQ(miopenFindSolutions(
                  handle, problem.handle, nullptr, solutions.data(), &found, solutions.size()),
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

    y.data = handle_deref.Read<float>(y_dev, y.data.size());

    ExpectMatchesCpuReference(x, w, y);
}

#endif // MIOPEN_ENABLE_HIPDNN_WRAPPER
