// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Convolution through each public entry point, since each is forwarded to hipDNN separately.
// "HipdnnShim" in a suite name is what puts the suite in the parity run; see README.md.

#include "hipdnn_shim_helpers.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

using namespace hipdnn_shim_test;

namespace {

// Half precision in a channels-last layout, because the only MIOpen solver that matches the
// four-operation plan this entry point always builds requires both. Whole numbers in [-4, 4]
// keep every product and partial sum on a value half precision holds exactly, so a
// disagreement with the reference is a real one and not accumulated rounding; the negatives
// are what make the ReLU do something.
using FusedType = half_float::half;

auto FusedElementGenerator()
{
    return [](auto...) { return static_cast<FusedType>(prng::gen_A_to_B(-4, 5)); };
}

// The shapes the fused path is checked on. Kept as a list because which shapes an
// implementation actually has a kernel for varies by device, and adding one is a line here.
// Sizes are small so the doubled replay stays cheap.
//
// The second shape has more output than input channels, so a bias add indexed by the wrong
// channel cannot pass both.
//
// Only three-dimensional shapes are listed. The two-dimensional NHWC equivalent is served by
// MIOpen on gfx90a but disagrees with the CPU reference by far more than half-precision
// rounding, while the reference is the same code that agrees on the three-dimensional shapes.
// That is a convolution bug to raise on its own, not something this test should carry.
struct FusedCase
{
    const char* name;
    miopenTensorLayout_t layout;
    std::vector<std::size_t> x_lengths;
    std::vector<std::size_t> w_lengths;
    ConvGeometry geometry;
};

std::vector<FusedCase> FusedCases()
{
    return {{"3d-ndhwc",
             miopenTensorNDHWC,
             {1, 4, 14, 11, 1},
             {4, 4, 3, 3, 3},
             {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}},
            {"3d-ndhwc-8-out",
             miopenTensorNDHWC,
             {1, 4, 14, 11, 1},
             {8, 4, 3, 3, 3},
             {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}}};
}

// One value per output channel. Unit strides throughout: the bias is one value per channel,
// and a layout-derived stride would otherwise describe a stride over dimensions of length one.
tensor<FusedType> MakeFusedBias(const FusedCase& config)
{
    std::vector<std::size_t> lengths(config.x_lengths.size(), 1);
    lengths[1] = config.w_lengths[0];
    const std::vector<std::size_t> unit_strides(lengths.size(), 1);
    tensor<FusedType> b{config.layout, lengths, unit_strides};
    b.generate(FusedElementGenerator());
    return b;
}

struct FusedScales
{
    float alpha1 = 1.0f;
    float alpha2 = 0.0f;
};

// The reference is built in the same layout as y so the two can be compared element by
// element; the CPU convolution indexes logically, so it is layout-agnostic already.
void CheckMatchesCpuBiasActivation(const FusedCase& config,
                                   const FusedScales& scales,
                                   const tensor<FusedType>& x,
                                   const tensor<FusedType>& w,
                                   const tensor<FusedType>& z,
                                   const tensor<FusedType>& bias,
                                   const tensor<FusedType>& y)
{
    const ConvGeometry& geometry = config.geometry;
    tensor<FusedType> ref_y{y.desc};
    cpu_convolution_forward(geometry.pads.size(),
                            x,
                            w,
                            ref_y,
                            geometry.pads,
                            geometry.strides,
                            geometry.dilations,
                            geometry.groups);

    ref_y.par_for_each([&](auto n, auto k, auto... spatial) {
        auto& value        = ref_y(n, k, spatial...);
        const float scaled = scales.alpha1 * static_cast<float>(value) +
                             scales.alpha2 * static_cast<float>(z(n, k, spatial...));
        const auto biased = static_cast<FusedType>(scaled + static_cast<float>(bias.data[k]));
        value             = std::max(FusedType(0), biased);
    });

    CheckWithinTolerance(ref_y, y, "fused bias+activation result");
}

// The setup the forward tests share: the same input, weights and convolution, the output shape
// the library reports for them, and all three tensors on the device. A fatal failure in SetUp()
// skips the test body, and the members still clean up.
class HipdnnShimConvFwd : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv));
        std::vector<std::size_t> out_lengths;
        ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));
        y = tensor<float>{out_lengths};

        x_dev = handle_deref.Write(x.data);
        w_dev = handle_deref.Write(w.data);
        y_dev = handle_deref.Write(y.data);
    }

    void ReadBackAndCheck()
    {
        y.data = handle_deref.Read<float>(y_dev, y.data.size());
        CheckMatchesCpuReference(x, w, y);
    }

    miopen::Handle& handle_deref = get_handle();
    miopenHandle_t handle        = &handle_deref;
    tensor<float> x              = MakeInput();
    tensor<float> w              = MakeWeights();
    tensor<float> y;
    OwnedConvDescriptor conv;
    miopen::Allocator::ManageDataPtr x_dev;
    miopen::Allocator::ManageDataPtr w_dev;
    miopen::Allocator::ManageDataPtr y_dev;
};

// One subclass per path, so each keeps its own suite name: the parity filter and the GPU
// exclusion patterns select by it.
class GPU_HipdnnShimConvFwdApi_FP32 : public HipdnnShimConvFwd
{
};

class GPU_HipdnnShimConvSolutionApi_FP32 : public HipdnnShimConvFwd
{
};

class GPU_HipdnnShimConvDeclined_FP32 : public HipdnnShimConvFwd
{
protected:
    // Runs the fused entry point on the fixture's problem. z is always passed because the
    // entry point needs a valid descriptor, even when alpha2 = 0 drops it.
    void RunFused(const float* alpha2, miopenActivationMode_t mode, CallResult& result)
    {
        tensor<float> bias{1, 4, 1, 1};
        tensor<float> z{y.desc.GetLengths()};
        auto bias_dev = handle_deref.Write(bias.data);
        auto z_dev    = handle_deref.Write(z.data);

        OwnedActivationDescriptor activation;
        ASSERT_EQ(miopenCreateActivationDescriptor(&activation.handle), miopenStatusSuccess);
        ASSERT_EQ(miopenSetActivationDescriptor(activation.handle, mode, 0.0, 0.0, 0.0),
                  miopenStatusSuccess);

        result = CaptureCall([&] {
            return miopenConvolutionBiasActivationForward(handle,
                                                          &kOne,
                                                          &x.desc,
                                                          x_dev.get(),
                                                          &w.desc,
                                                          w_dev.get(),
                                                          conv.handle,
                                                          miopenConvolutionFwdAlgoImplicitGEMM,
                                                          nullptr,
                                                          0ull,
                                                          alpha2,
                                                          &z.desc,
                                                          z_dev.get(),
                                                          &bias.desc,
                                                          bias_dev.get(),
                                                          activation.handle,
                                                          &y.desc,
                                                          y_dev.get());
        });
    }
};

using OwnedStream = Owned<hipStream_t, hipStreamDestroy>;

} // namespace

// The Find/Run pair: the older of the two public convolution paths, and the one most callers
// still use.
TEST_F(GPU_HipdnnShimConvFwdApi_FP32, FindAndForwardMatchCpuReference)
{
    const auto result = FindAndRun(Direction::Forward, conv.handle, x, w, y);
    ASSERT_NO_FATAL_FAILURE(AssertServed(result));
    CheckMatchesCpuReference(x, w, y);
}

// The one convolution attribute the forwarded path does not decline: its non-default value only
// allows TF32. The inputs are small whole numbers, which TF32 holds exactly, so the result must
// match the reference whichever precision ran.
TEST_F(GPU_HipdnnShimConvFwdApi_FP32, MathTypeDefaultMatchesCpuReference)
{
    ASSERT_EQ(miopenSetConvolutionAttribute(
                  conv.handle, MIOPEN_CONVOLUTION_ATTRIB_MATH_TYPE, miopenMathDefault),
              miopenStatusSuccess);
    const auto result = FindAndRun(Direction::Forward, conv.handle, x, w, y);
    ASSERT_NO_FATAL_FAILURE(AssertServed(result));
    CheckMatchesCpuReference(x, w, y);
}

// A stride above INT_MAX, which the int tensor getter cannot report, is read at full width and
// served. Only a dim of length 1 has the large stride, so the buffers stay small. Forwarding
// only, as in the declined-problem test: MIOpen's own int-indexed paths are not expected to take
// this layout.
TEST_F(GPU_HipdnnShimConvFwdApi_FP32, StrideAboveIntMaxMatchesCpuReference)
{
    if(!ForwardingEnabled())
        return;
    const std::size_t above_int_max = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
    tensor<float> wide_x{std::vector<std::size_t>{1, 4, 8, 8},
                         std::vector<std::size_t>{above_int_max, 64, 8, 1}};
    wide_x.generate(tensor_elem_gen_integer{17});
    tensor<float> narrow_w{w.desc.GetLengths()};
    narrow_w.generate(tensor_elem_gen_integer{17});
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, wide_x, narrow_w, out_lengths));
    tensor<float> wide_y{out_lengths};
    const auto result = FindAndRun(Direction::Forward, conv.handle, wide_x, narrow_w, wide_y);
    ASSERT_NO_FATAL_FAILURE(AssertServed(result));
    CheckMatchesCpuReference(wide_x, narrow_w, wide_y);
}

// The Problem/Solution path reaches the same convolution through different public entry
// points, so it has to be swapped over separately and is covered separately.
TEST_F(GPU_HipdnnShimConvSolutionApi_FP32, RunSolutionMatchesCpuReference)
{
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

    std::size_t workspace_size = 0;
    ASSERT_EQ(miopenGetSolutionWorkspaceSize(solutions[0], &workspace_size), miopenStatusSuccess);
    Workspace wspace{workspace_size};

    miopenTensorDescriptor_t descriptors[3] = {&x.desc, &w.desc, &y.desc};
    const std::array<miopenTensorArgument_t, 3> arguments{{
        {miopenTensorConvolutionX, &descriptors[0], x_dev.get()},
        {miopenTensorConvolutionW, &descriptors[1], w_dev.get()},
        {miopenTensorConvolutionY, &descriptors[2], y_dev.get()},
    }};

    ASSERT_EQ(
        miopenRunSolution(
            handle, solutions[0], arguments.size(), arguments.data(), wspace.ptr(), wspace.size()),
        miopenStatusSuccess);

    ReadBackAndCheck();
}

TEST(GPU_HipdnnShimConvBwdDataApi_FP32, BackwardDataMatchesCpuReference)
{
    auto x = MakeInput();
    auto w = MakeWeights();
    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv));
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));

    tensor<float> dy{out_lengths};
    dy.generate(tensor_elem_gen_integer{17});
    tensor<float> dx{x.desc.GetLengths()};

    const auto result = FindAndRun(Direction::BackwardData, conv.handle, dx, w, dy);
    ASSERT_NO_FATAL_FAILURE(AssertServed(result));
    CheckMatchesCpuBackwardData(dx, w, dy);
}

TEST(GPU_HipdnnShimConvBwdWeightsApi_FP32, BackwardWeightsMatchesCpuReference)
{
    auto x = MakeInput();
    auto w = MakeWeights();
    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv));
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));

    tensor<float> dy{out_lengths};
    dy.generate(tensor_elem_gen_integer{17});
    tensor<float> dw{w.desc.GetLengths()};

    const auto result = FindAndRun(Direction::BackwardWeights, conv.handle, x, dw, dy);
    ASSERT_NO_FATAL_FAILURE(AssertServed(result));
    CheckMatchesCpuBackwardWeights(x, dw, dy);
}

// The caller may destroy the old stream after a stream switch, so the next forwarded call must
// not touch it.
TEST(GPU_HipdnnShimConvStreamSwitch_FP32, CallAfterOldStreamIsDestroyed)
{
    OwnedStream stream_a;
    OwnedStream stream_b;
    ASSERT_EQ(hipStreamCreate(&stream_a.handle), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream_b.handle), hipSuccess);

    Owned<miopenHandle_t, miopenDestroy> owned_handle;
    ASSERT_EQ(miopenCreateWithStream(&owned_handle.handle, stream_a.handle), miopenStatusSuccess);
    miopenHandle_t handle = owned_handle.handle;

    const ConvGeometry geometry{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    tensor<float> x{1, 4, 6, 8, 8};
    x.generate(tensor_elem_gen_integer{17});
    tensor<float> w{4, 4, 3, 3, 3};
    w.generate(tensor_elem_gen_integer{17});
    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv, miopenConvolution, geometry));
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));
    tensor<float> y{out_lengths};

    auto& staging = get_handle();
    auto x_dev    = staging.Write(x.data);
    auto w_dev    = staging.Write(w.data);
    auto y_dev    = staging.Write(y.data);

    std::size_t workspace_size = 0;
    ASSERT_EQ(miopenConvolutionForwardGetWorkSpaceSize(
                  handle, &w.desc, &x.desc, conv.handle, &y.desc, &workspace_size),
              miopenStatusSuccess);
    Workspace workspace{workspace_size};

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
                                                    workspace.ptr(),
                                                    workspace.size(),
                                                    false),
              miopenStatusSuccess);
    ASSERT_GT(returned_algo_count, 0);

    auto forward = [&] {
        return miopenConvolutionForward(handle,
                                        &kOne,
                                        &x.desc,
                                        x_dev.get(),
                                        &w.desc,
                                        w_dev.get(),
                                        conv.handle,
                                        perf.fwd_algo,
                                        &kZero,
                                        &y.desc,
                                        y_dev.get(),
                                        workspace.ptr(),
                                        workspace.size());
    };

    ASSERT_EQ(forward(), miopenStatusSuccess);
    ASSERT_EQ(miopenSetStream(handle, stream_b.handle), miopenStatusSuccess);
    ASSERT_EQ(hipStreamDestroy(stream_a.handle), hipSuccess);
    stream_a.handle = nullptr;
    ResetAfterFind(y_dev, y);

    EXPECT_EQ(forward(), miopenStatusSuccess) << "the first call after the switch";
    EXPECT_EQ(forward(), miopenStatusSuccess) << "a later call";
    ASSERT_EQ(hipStreamSynchronize(stream_b.handle), hipSuccess);

    y.data = staging.Read<float>(y_dev, y.data.size());
    CheckMatchesCpuReference(x, w, y, geometry);
}

// A caller may give less workspace than the size query reports (PyTorch does when it cannot
// allocate the full size). The call must then run the best choice that fits, not fail: here no
// workspace at all, which a direct solver needs.
TEST(GPU_HipdnnShimConvLessWorkspace_FP32, FindAndRunWithoutWorkspaceMatchCpuReference)
{
    auto& handle_deref    = get_handle();
    miopenHandle_t handle = &handle_deref;

    const ConvGeometry geometry{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    tensor<float> x{1, 4, 6, 8, 8};
    x.generate(tensor_elem_gen_integer{17});
    tensor<float> w{4, 4, 3, 3, 3};
    w.generate(tensor_elem_gen_integer{17});
    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv, miopenConvolution, geometry));
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));
    tensor<float> y{out_lengths};

    auto x_dev = handle_deref.Write(x.data);
    auto w_dev = handle_deref.Write(w.data);
    auto y_dev = handle_deref.Write(y.data);

    std::size_t workspace_size = 0;
    ASSERT_EQ(miopenConvolutionForwardGetWorkSpaceSize(
                  handle, &w.desc, &x.desc, conv.handle, &y.desc, &workspace_size),
              miopenStatusSuccess);
    if(workspace_size == 0)
        GTEST_SKIP() << "the size query reports no workspace, so giving less cannot be tested";

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
                                                    nullptr,
                                                    0,
                                                    false),
              miopenStatusSuccess);
    ASSERT_GT(returned_algo_count, 0);
    EXPECT_EQ(perf.memory, std::size_t{0})
        << "Find reported a solution that needs workspace it was not given";
    ResetAfterFind(y_dev, y);

    ASSERT_EQ(miopenConvolutionForward(handle,
                                       &kOne,
                                       &x.desc,
                                       x_dev.get(),
                                       &w.desc,
                                       w_dev.get(),
                                       conv.handle,
                                       perf.fwd_algo,
                                       &kZero,
                                       &y.desc,
                                       y_dev.get(),
                                       nullptr,
                                       0),
              miopenStatusSuccess);

    y.data = handle_deref.Read<float>(y_dev, y.data.size());
    CheckMatchesCpuReference(x, w, y, geometry);
}

void RunFusedCase(const FusedCase& config, const FusedScales& scales = {})
{
    SCOPED_TRACE(config.name);

    auto& handle_deref    = get_handle();
    miopenHandle_t handle = &handle_deref;

    tensor<FusedType> x{config.layout, config.x_lengths};
    x.generate(FusedElementGenerator());
    tensor<FusedType> w{config.layout, config.w_lengths};
    w.generate(FusedElementGenerator());
    auto bias = MakeFusedBias(config);

    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv, miopenConvolution, config.geometry));
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));
    tensor<FusedType> y{config.layout, out_lengths};
    tensor<FusedType> z{config.layout, out_lengths};
    if(scales.alpha2 != 0.0f)
        z.generate(FusedElementGenerator());

    OwnedActivationDescriptor activation;
    ASSERT_EQ(miopenCreateActivationDescriptor(&activation.handle), miopenStatusSuccess);
    ASSERT_EQ(miopenSetActivationDescriptor(activation.handle, miopenActivationRELU, 0.0, 0.0, 0.0),
              miopenStatusSuccess);

    auto x_dev    = handle_deref.Write(x.data);
    auto w_dev    = handle_deref.Write(w.data);
    auto bias_dev = handle_deref.Write(bias.data);
    auto y_dev    = handle_deref.Write(y.data);
    auto z_dev    = handle_deref.Write(z.data);

    // The entry point needs a valid z even when alpha2 is zero.
    const auto result = CaptureCall([&] {
        return miopenConvolutionBiasActivationForward(handle,
                                                      &scales.alpha1,
                                                      &x.desc,
                                                      x_dev.get(),
                                                      &w.desc,
                                                      w_dev.get(),
                                                      conv.handle,
                                                      miopenConvolutionFwdAlgoImplicitGEMM,
                                                      nullptr,
                                                      0ull,
                                                      &scales.alpha2,
                                                      &z.desc,
                                                      z_dev.get(),
                                                      &bias.desc,
                                                      bias_dev.get(),
                                                      activation.handle,
                                                      &y.desc,
                                                      y_dev.get());
    });

    // Fused conv+bias+activation is unimplemented on some devices -- the only MIOpen solver that
    // matches the plan this entry point builds needs a whitelisted device, and hipDNN has its own
    // set of cases it will not express -- so a decline is a legitimate answer rather than a
    // failure. It ends the test as a pass, not a skip: a skip in one parity replay against a pass
    // in the other reads as a divergence that is not one. Where the decline came from is still
    // checkable, and with forwarding on it must carry the forwarded-error prefix, which rules out
    // a silent fall back to MIOpen. The recorded property is what keeps the decline visible to
    // the parity comparison, which a pass on its own would not be.
    if(result.status == miopenStatusUnsupportedOp)
    {
        RecordServed(config.name, false);
        ExpectForwardedDecline(result);
        GTEST_LOG_(INFO) << "fused conv+bias+activation was declined here, so the result was not "
                            "checked against the CPU reference: "
                         << result.message;
        return;
    }
    ASSERT_EQ(result.status, miopenStatusSuccess) << result.message << '\n' << result.logged;
    RecordServed(config.name, true);

    y.data = handle_deref.Read<FusedType>(y_dev, y.data.size());

    CheckMatchesCpuBiasActivation(config, scales, x, w, z, bias, y);
}

TEST(GPU_HipdnnShimConvBiasActivApi_FP16, FusedForwardMatchesCpuReference)
{
    for(const auto& config : FusedCases())
        ASSERT_NO_FATAL_FAILURE(RunFusedCase(config));
}

// Forwarding declines these: its graph has no z and no scale on the convolution. Kept out of
// FusedCases so these gaps get their own parity entry.
TEST(GPU_HipdnnShimConvFusedAlpha_FP16, FusedForwardMatchesCpuReference)
{
    const auto ndhwc = [](const char* name) {
        return FusedCase{name,
                         miopenTensorNDHWC,
                         {1, 4, 14, 11, 1},
                         {4, 4, 3, 3, 3},
                         {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}};
    };
    ASSERT_NO_FATAL_FAILURE(RunFusedCase(ndhwc("3d-ndhwc-plus-z"), {1.0f, 1.0f}));
    ASSERT_NO_FATAL_FAILURE(RunFusedCase(ndhwc("3d-ndhwc-alpha1-2"), {2.0f, 0.0f}));
}

// fp16, so the forwarded graph is served only if its bias add computes in the bias type (in
// fp32 that is also the compute type). NCHW because MIOpen declines it before launching a
// kernel, so it cannot fault the test process. Both modes decline it on gfx90a, where the
// MIOpen provider turns fusion off.
TEST(GPU_HipdnnShimConvBiasActiv2d_FP16, FusedForwardMatchesCpuReference)
{
    ASSERT_NO_FATAL_FAILURE(RunFusedCase(FusedCase{
        "2d-nchw", miopenTensorNCHW, {1, 16, 8, 8}, {16, 16, 3, 3}, {{1, 1}, {1, 1}, {1, 1}}}));
}

// MIOpen accepts int8 input and weights with an int32 or float output, so the forwarded path must
// keep each tensor's type instead of declining the mix. No hipDNN provider has an int8 convolution
// engine yet, so with forwarding on it is declined, a known parity divergence.
template <class Out>
void RunInt8Case(const std::string& case_name)
{
    SCOPED_TRACE(case_name);

    auto& handle_deref    = get_handle();
    miopenHandle_t handle = &handle_deref;

    tensor<int8_t> x{2, 4, 8, 8};
    x.generate(tensor_elem_gen_integer{17});
    tensor<int8_t> w{4, 4, 3, 3};
    w.generate(tensor_elem_gen_integer{17});
    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv));
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));
    tensor<Out> y{out_lengths};

    auto x_dev = handle_deref.Write(x.data);
    auto w_dev = handle_deref.Write(w.data);
    auto y_dev = handle_deref.Write(y.data);

    // GetWorkSpaceSize, Find and the run are all forwarded and each builds the graph, so with
    // no engine for these types the decline comes from the first of them. It must come from
    // hipDNN's engine search: a decline from the up-front checks means the mixed types were
    // never translated. A decline passes rather than skips, as in the fused test.
    const auto declined_by_engine_search = [&](const CallResult& call) {
        if(call.status != miopenStatusUnsupportedOp || !ForwardingEnabled())
            return false;
        RecordServed(case_name, false);
        ExpectForwardedDecline(call);
        for(const char* check_reason : {"has no hipDNN convolution", "not forwarded to hipDNN"})
        {
            EXPECT_EQ(call.logged.find(check_reason), std::string::npos)
                << "the mixed types were refused instead of translated: " << call.logged;
        }
        return true;
    };

    std::size_t workspace_size = 0;
    const auto sized           = CaptureCall([&] {
        return miopenConvolutionForwardGetWorkSpaceSize(
            handle, &w.desc, &x.desc, conv.handle, &y.desc, &workspace_size);
    });
    if(declined_by_engine_search(sized))
        return;
    ASSERT_EQ(sized.status, miopenStatusSuccess) << sized.message << '\n' << sized.logged;
    Workspace wspace{workspace_size};

    int returned_algo_count = 0;
    miopenConvAlgoPerf_t perf{};
    const auto found = CaptureCall([&] {
        return miopenFindConvolutionForwardAlgorithm(handle,
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
                                                     false);
    });
    if(declined_by_engine_search(found))
        return;
    ASSERT_EQ(found.status, miopenStatusSuccess) << found.message << '\n' << found.logged;
    ASSERT_GT(returned_algo_count, 0);
    ResetAfterFind(y_dev, y);

    const auto result = CaptureCall([&] {
        return miopenConvolutionForward(handle,
                                        &kOne,
                                        &x.desc,
                                        x_dev.get(),
                                        &w.desc,
                                        w_dev.get(),
                                        conv.handle,
                                        perf.fwd_algo,
                                        &kZero,
                                        &y.desc,
                                        y_dev.get(),
                                        wspace.ptr(),
                                        wspace.size());
    });

    if(declined_by_engine_search(result))
        return;
    ASSERT_EQ(result.status, miopenStatusSuccess) << result.message << '\n' << result.logged;
    RecordServed(case_name, true);

    y.data = handle_deref.Read<Out>(y_dev, y.data.size());
    tensor<Out> ref_y{out_lengths};
    const ConvGeometry geometry;
    cpu_convolution_forward(geometry.pads.size(),
                            x,
                            w,
                            ref_y,
                            geometry.pads,
                            geometry.strides,
                            geometry.dilations,
                            geometry.groups);
    ASSERT_FALSE(miopen::range_zero(ref_y)) << "CPU reference is all zeros";
    // Exact, because every sum is a whole number small enough for a float to hold exactly.
    EXPECT_EQ(ref_y.data, y.data);
}

TEST(GPU_HipdnnShimConvMixedTypeApi_I8, Int8InputMatchesCpuReference)
{
    ASSERT_NO_FATAL_FAILURE(RunInt8Case<int>("int8-in-int32-out"));
    ASSERT_NO_FATAL_FAILURE(RunInt8Case<float>("int8-in-float-out"));
}

// One call per check the forwarded path makes before building a graph. Forwarding only, since
// natively several of these problems are malformed or crash. The reason is checked, not just
// the status, because a hipDNN decline for lack of an engine would hide a missing check. The
// data-type check has no case: no type the public API can describe reaches it.
// hipdnn_shim_conv_gaps.cpp runs the alpha/beta, grouped, transposed and rank problems in both
// modes; StrideAboveIntMaxMatchesCpuReference serves a stride above INT_MAX.
TEST_F(GPU_HipdnnShimConvDeclined_FP32, UnsupportedProblemsAreDeclinedWithReason)
{
    if(!ForwardingEnabled())
        return;

    auto expect_declined_because = [](const CallResult& result,
                                      const char* reason,
                                      miopenStatus_t expected_status = miopenStatusUnsupportedOp) {
        EXPECT_EQ(result.status, expected_status) << reason;
        ExpectForwardedDecline(result, reason);
    };

    // The declines happen before any hipDNN object is built, so no workspace is needed to
    // reach them.
    auto forward = [&](const float* alpha,
                       const float* beta,
                       tensor<float>& in,
                       tensor<float>& weights,
                       miopenConvolutionDescriptor_t c,
                       tensor<float>& out) {
        auto in_dev       = handle_deref.Write(in.data);
        auto weights_dev  = handle_deref.Write(weights.data);
        auto out_dev      = handle_deref.Write(out.data);
        const auto result = CaptureCall([&] {
            return miopenConvolutionForward(handle,
                                            alpha,
                                            &in.desc,
                                            in_dev.get(),
                                            &weights.desc,
                                            weights_dev.get(),
                                            c,
                                            miopenConvolutionFwdAlgoGEMM,
                                            beta,
                                            &out.desc,
                                            out_dev.get(),
                                            nullptr,
                                            0);
        });
        // With a check missing the call is served, so its work must finish before the buffers
        // are freed.
        handle_deref.Finish();
        return result;
    };

    const float two = 2.0f;
    expect_declined_because(forward(&two, &kZero, x, w, conv.handle, y),
                            "supports only alpha=1, beta=0",
                            miopenStatusNotImplemented);
    expect_declined_because(forward(&kOne, &kOne, x, w, conv.handle, y),
                            "supports only alpha=1, beta=0",
                            miopenStatusNotImplemented);
    expect_declined_because(forward(nullptr, &kZero, x, w, conv.handle, y),
                            "supports only alpha=1, beta=0",
                            miopenStatusNotImplemented);

    {
        tensor<float> vectorized_x{miopenTensorNCHWc4, x.desc.GetLengths()};
        tensor<float> vectorized_w{miopenTensorNCHWc4, w.desc.GetLengths()};
        expect_declined_because(forward(&kOne, &kZero, vectorized_x, vectorized_w, conv.handle, y),
                                "vectorized tensor layouts");
    }

    {
        const ConvGeometry geometry{{1}, {1}, {1}};
        OwnedConvDescriptor one_d;
        ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(one_d, miopenConvolution, geometry));
        // Written out because MIOpen's output-size query fails for 1-D. A pad of 1 around a
        // 3-wide filter keeps the length.
        tensor<float> one_d_x{std::vector<std::size_t>{2, 4, 8}};
        tensor<float> one_d_w{std::vector<std::size_t>{4, 4, 3}};
        tensor<float> one_d_y{std::vector<std::size_t>{2, 4, 8}};
        expect_declined_because(forward(&kOne, &kZero, one_d_x, one_d_w, one_d.handle, one_d_y),
                                "only 2-D and 3-D convolutions");
    }

    {
        const ConvGeometry geometry{{0, 0, 0, 0}, {1, 1, 1, 1}, {1, 1, 1, 1}};
        OwnedConvDescriptor four_d;
        ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(four_d, miopenConvolution, geometry));
        // A 1x1x1x1 filter with no padding keeps the input's shape. Left zeroed because the
        // element generator does not take six dimensions.
        const std::vector<std::size_t> lengths{1, 1, 2, 2, 2, 2};
        tensor<float> four_d_x{lengths};
        tensor<float> four_d_w{std::vector<std::size_t>(6, 1)};
        tensor<float> four_d_y{lengths};
        expect_declined_because(forward(&kOne, &kZero, four_d_x, four_d_w, four_d.handle, four_d_y),
                                "only 2-D and 3-D convolutions");
    }

    struct AttributeCase
    {
        miopenConvolutionAttrib_t attribute;
        int value;
        const char* reason;
    };
    for(const AttributeCase& attribute_case :
        {AttributeCase{MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL,
                       1,
                       "non-default MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL"},
         AttributeCase{MIOPEN_CONVOLUTION_ATTRIB_FP8_ROUNDING_MODE,
                       miopenF8RoundingModeStandard,
                       "non-default MIOPEN_CONVOLUTION_ATTRIB_FP8_ROUNDING_MODE"}})
    {
        OwnedConvDescriptor with_attribute;
        ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(with_attribute));
        ASSERT_EQ(miopenSetConvolutionAttribute(
                      with_attribute.handle, attribute_case.attribute, attribute_case.value),
                  miopenStatusSuccess)
            << attribute_case.reason;
        expect_declined_because(forward(&kOne, &kZero, x, w, with_attribute.handle, y),
                                attribute_case.reason);
    }

    CallResult fused;
    ASSERT_NO_FATAL_FAILURE(RunFused(nullptr, miopenActivationRELU, fused));
    expect_declined_because(fused, "alpha2 is null");
}

// Runs in both modes: for an activation other than ReLU, the forwarded path must return the
// same status as MIOpen's own fused path.
TEST_F(GPU_HipdnnShimConvDeclined_FP32, NonReluActivationIsNotImplemented)
{
    CallResult result;
    ASSERT_NO_FATAL_FAILURE(RunFused(&kZero, miopenActivationPASTHRU, result));
    EXPECT_EQ(result.status, miopenStatusNotImplemented);
    ExpectForwardedDecline(result, "miopenActivationRELU is supported");
}
