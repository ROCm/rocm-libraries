// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Problems that test how the forwarded path translates shapes into a hipDNN graph: data types,
// layouts, 3-D, per-axis geometry and strided tensors. Each must be served in both modes.

#include "hipdnn_shim_helpers.hpp"

#include <algorithm>
#include <array>
#include <cstdint>

using namespace hipdnn_shim_test;

namespace {

struct TranslationCase
{
    const char* name;
    // Ignored for x when x_strides is set.
    miopenTensorLayout_t layout;
    std::vector<std::size_t> x_lengths;
    // Empty for packed.
    std::vector<std::size_t> x_strides;
    std::vector<std::size_t> w_lengths;
    ConvGeometry geometry;
    bool forward;
    bool backward_data;
    bool backward_weights;
};

// Keeps ctest names stable. Without it gtest prints the struct's bytes, which include pointers.
void PrintTo(const TranslationCase& c, std::ostream* os) { *os << c.name; }

std::string CaseName(const ::testing::TestParamInfo<TranslationCase>& info)
{
    return info.param.name;
}

// With the fp32 values, an fp16 backward-weights sum over 512 products could pass fp16's
// maximum of 65504.
template <class T>
std::uint64_t FillMax(std::uint64_t fp32_max)
{
    return std::is_same_v<T, float> ? fp32_max : 5;
}

template <class T>
void CheckForward(miopenTensorLayout_t layout,
                  tensor<T>& x,
                  tensor<T>& w,
                  const ConvGeometry& geometry)
{
    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv, miopenConvolution, geometry));
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));
    tensor<T> y{layout, out_lengths};

    const auto result = FindAndRun(Direction::Forward, conv.handle, x, w, y);
    ASSERT_NO_FATAL_FAILURE(AssertServed(result));
    CheckMatchesCpuReference(x, w, y, geometry);
}

// dx takes x's descriptor, so a strided x also tests a strided dx.
template <class T>
void CheckBackwardData(miopenTensorLayout_t layout,
                       tensor<T>& x,
                       tensor<T>& w,
                       const ConvGeometry& geometry)
{
    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv, miopenConvolution, geometry));
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));
    tensor<T> dy = MakeFilled<T>(layout, out_lengths, {}, FillMax<T>(13));
    tensor<T> dx{x.desc};

    const auto result = FindAndRun(Direction::BackwardData, conv.handle, dx, w, dy);
    ASSERT_NO_FATAL_FAILURE(AssertServed(result));
    CheckMatchesCpuBackwardData(dx, w, dy, geometry);
}

template <class T>
void CheckBackwardWeights(miopenTensorLayout_t layout,
                          tensor<T>& x,
                          tensor<T>& w,
                          const ConvGeometry& geometry)
{
    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv, miopenConvolution, geometry));
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));
    tensor<T> dy = MakeFilled<T>(layout, out_lengths, {}, FillMax<T>(13));
    tensor<T> dw{w.desc};

    const auto result = FindAndRun(Direction::BackwardWeights, conv.handle, x, dw, dy);
    ASSERT_NO_FATAL_FAILURE(AssertServed(result));
    CheckMatchesCpuBackwardWeights(x, dw, dy, geometry);
}

template <class T>
void RunTranslationCase(const TranslationCase& config)
{
    tensor<T> x = MakeFilled<T>(config.layout, config.x_lengths, config.x_strides, FillMax<T>(17));
    tensor<T> w = MakeFilled<T>(config.layout, config.w_lengths, {}, FillMax<T>(17));

    if(config.forward)
    {
        SCOPED_TRACE("forward");
        ASSERT_NO_FATAL_FAILURE(CheckForward(config.layout, x, w, config.geometry));
    }
    if(config.backward_data)
    {
        SCOPED_TRACE("backward data");
        ASSERT_NO_FATAL_FAILURE(CheckBackwardData(config.layout, x, w, config.geometry));
    }
    if(config.backward_weights)
    {
        SCOPED_TRACE("backward weights");
        ASSERT_NO_FATAL_FAILURE(CheckBackwardWeights(config.layout, x, w, config.geometry));
    }
}

std::vector<TranslationCase> Fp32Cases()
{
    return {
        {"Ncdhw",
         miopenTensorNCDHW,
         {1, 4, 6, 8, 8},
         {},
         {4, 4, 3, 3, 3},
         {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
         true,
         true,
         true},
        // The problem from conv_api_strided_tensors.cpp, which checks only the status.
        {"NcdhwStrided",
         miopenTensorNCDHW,
         {4, 4, 16, 9, 16},
         {10240, 2560, 160, 16, 1},
         {8, 4, 3, 3, 3},
         {{1, 0, 1}, {2, 2, 2}, {1, 1, 1}},
         true,
         false,
         false},
        // Each axis differs, so mixing up stride and dilation or the axis order changes the
        // output.
        {"PerAxisGeometry",
         miopenTensorNCHW,
         {2, 4, 9, 13},
         {},
         {4, 4, 3, 5},
         {{1, 2}, {2, 1}, {1, 2}},
         true,
         true,
         true},
        // No stride is 1, yet the layout is plain NCHW, not vectorized.
        {"InnermostStride2",
         miopenTensorNCHW,
         {2, 4, 8, 8},
         {512, 128, 16, 2},
         {4, 4, 3, 3},
         {},
         true,
         true,
         true},
        {"OneByOneStride2",
         miopenTensorNCHW,
         {1, 16, 8, 8},
         {},
         {16, 16, 1, 1},
         {{0, 0}, {2, 2}, {1, 1}},
         true,
         true,
         false},
        {"WeightsFilter5x5",
         miopenTensorNCHW,
         {2, 4, 12, 12},
         {},
         {4, 4, 5, 5},
         {{2, 2}, {1, 1}, {1, 1}},
         false,
         false,
         true},
        {"WeightsFilter7x1",
         miopenTensorNCHW,
         {2, 4, 12, 12},
         {},
         {4, 4, 7, 1},
         {{3, 0}, {1, 1}, {1, 1}},
         false,
         false,
         true},
        // Rows padded to 16, so strides recomputed from the lengths would be wrong.
        {"PaddedRows",
         miopenTensorNCHW,
         {2, 4, 8, 8},
         {512, 128, 16, 1},
         {4, 4, 3, 3},
         {},
         true,
         false,
         false},
        // With one channel, NHWC has the same strides as NCHW, so the layout cannot be told
        // from the strides alone.
        {"NhwcOneChannel", miopenTensorNHWC, {2, 1, 8, 8}, {}, {4, 1, 3, 3}, {}, true, true, true},
    };
}

std::vector<TranslationCase> Fp16Cases()
{
    return {
        {"Nchw", miopenTensorNCHW, {2, 16, 16, 16}, {}, {16, 16, 3, 3}, {}, true, true, true},
    };
}

std::vector<TranslationCase> Bf16Cases()
{
    return {
        {"Nhwc", miopenTensorNHWC, {2, 16, 16, 16}, {}, {16, 16, 3, 3}, {}, true, true, true},
    };
}

} // namespace

class GPU_HipdnnShimConvTranslation_FP32 : public ::testing::TestWithParam<TranslationCase>
{
};

TEST_P(GPU_HipdnnShimConvTranslation_FP32, MatchesCpuReference)
{
    RunTranslationCase<float>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_HipdnnShimConvTranslation_FP32,
                         ::testing::ValuesIn(Fp32Cases()),
                         CaseName);

class GPU_HipdnnShimConvTranslation_FP16 : public ::testing::TestWithParam<TranslationCase>
{
};

TEST_P(GPU_HipdnnShimConvTranslation_FP16, MatchesCpuReference)
{
    RunTranslationCase<half_float::half>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_HipdnnShimConvTranslation_FP16,
                         ::testing::ValuesIn(Fp16Cases()),
                         CaseName);

class GPU_HipdnnShimConvTranslation_BFP16 : public ::testing::TestWithParam<TranslationCase>
{
};

TEST_P(GPU_HipdnnShimConvTranslation_BFP16, MatchesCpuReference)
{
    RunTranslationCase<bfloat16>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_HipdnnShimConvTranslation_BFP16,
                         ::testing::ValuesIn(Bf16Cases()),
                         CaseName);

// Same tensor shapes, different padding and dilation. A plan cache keyed without those would
// reuse the first plan for the second problem.
TEST(GPU_HipdnnShimConvPlanKey_FP32, SameShapeDifferentPadsAndDilations)
{
    tensor<float> x = MakeFilled<float>(miopenTensorNCHW, {2, 4, 8, 8}, {}, 17);
    tensor<float> w = MakeFilled<float>(miopenTensorNCHW, {4, 4, 3, 3}, {}, 17);

    for(const ConvGeometry& geometry :
        {ConvGeometry{{1, 1}, {1, 1}, {1, 1}}, ConvGeometry{{2, 2}, {1, 1}, {2, 2}}})
    {
        SCOPED_TRACE("pad and dilation " + std::to_string(geometry.pads[0]));
        ASSERT_NO_FATAL_FAILURE(CheckForward(miopenTensorNCHW, x, w, geometry));
        ASSERT_NO_FATAL_FAILURE(CheckBackwardData(miopenTensorNCHW, x, w, geometry));
        ASSERT_NO_FATAL_FAILURE(CheckBackwardWeights(miopenTensorNCHW, x, w, geometry));
    }
}

namespace {

// Normal mode runs the full search, so what Find returns does not depend on a find database.
void InitNormalFindConvDescriptor(OwnedConvDescriptor& conv)
{
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv));
    ASSERT_EQ(miopenSetConvolutionFindMode(conv.handle, miopenConvolutionFindModeNormal),
              miopenStatusSuccess);
}

bool WithinTolerance(const tensor<float>& reference, const tensor<float>& got)
{
    return !miopen::range_zero(got) &&
           miopen::range_distance(reference) == miopen::range_distance(got) &&
           miopen::rms_range(reference, got) < Tolerance<float>();
}

} // namespace

// MIOpen's kernel lookup key leaves out strides, so a non-packed run with no Find of its own
// reuses the packed problem's kernel and returns wrong values with a success status. MIOpen
// caches kernels per handle, so a separate handle keeps earlier tests from changing the result.
// Sharing the staging handle's stream keeps the copies ordered with the runs.
TEST(GPU_HipdnnShimConvNonPackedAfterPacked_FP32, MatchesCpuReference)
{
    auto& handle_deref = get_handle();
    Owned<miopenHandle_t, miopenDestroy> owned_handle;
    ASSERT_EQ(miopenCreateWithStream(&owned_handle.handle, handle_deref.GetStream()),
              miopenStatusSuccess);
    miopenHandle_t handle = owned_handle.handle;

    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitNormalFindConvDescriptor(conv));

    tensor<float> x = MakeFilled<float>(miopenTensorNCHW, {1, 8, 8, 8}, {}, 17);
    tensor<float> x_strided =
        MakeFilled<float>(miopenTensorNCHW, {1, 8, 8, 8}, {1024, 128, 16, 1}, 17);
    tensor<float> w = MakeFilled<float>(miopenTensorNCHW, {8, 8, 3, 3}, {}, 17);
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));
    tensor<float> y{out_lengths};

    auto x_dev         = handle_deref.Write(x.data);
    auto x_strided_dev = handle_deref.Write(x_strided.data);
    auto w_dev         = handle_deref.Write(w.data);
    auto y_dev         = handle_deref.Write(y.data);

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
    ResetAfterFind(y_dev, y);
    wspace.resize(std::max(workspace_size, perf.memory));

    {
        SCOPED_TRACE("packed");
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
                                           wspace.ptr(),
                                           wspace.size()),
                  miopenStatusSuccess);
        y.data = handle_deref.Read<float>(y_dev, y.data.size());
        ASSERT_NO_FATAL_FAILURE(CheckMatchesCpuReference(x, w, y));
    }

    tensor<float> y_strided{out_lengths};
    auto y_strided_dev = handle_deref.Write(y_strided.data);
    ASSERT_EQ(miopenConvolutionForward(handle,
                                       &kOne,
                                       &x_strided.desc,
                                       x_strided_dev.get(),
                                       &w.desc,
                                       w_dev.get(),
                                       conv.handle,
                                       perf.fwd_algo,
                                       &kZero,
                                       &y_strided.desc,
                                       y_strided_dev.get(),
                                       wspace.ptr(),
                                       wspace.size()),
              miopenStatusSuccess);
    y_strided.data = handle_deref.Read<float>(y_strided_dev, y_strided.data.size());

    const ConvGeometry geometry;
    tensor<float> reference{y_strided.desc};
    cpu_convolution_forward(geometry.pads.size(),
                            x_strided,
                            w,
                            reference,
                            geometry.pads,
                            geometry.strides,
                            geometry.dilations,
                            geometry.groups);
    RecordCorrect("non-packed-after-packed", WithinTolerance(reference, y_strided));
    // A known MIOpen bug makes the native result wrong, so it is only recorded.
    if(ForwardingEnabled())
        CheckWithinTolerance(reference, y_strided, "non-packed run");
}

// Find is never forwarded, so both modes offer the same algorithms. Forwarded, the provider
// runs its own choice, whichever one the caller passes.
TEST(GPU_HipdnnShimConvAlgoChoice_FP32, EveryFoundAlgorithmMatchesCpuReference)
{
    auto& handle_deref    = get_handle();
    miopenHandle_t handle = &handle_deref;

    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitNormalFindConvDescriptor(conv));

    tensor<float> x = MakeFilled<float>(miopenTensorNCHW, {2, 8, 10, 10}, {}, 17);
    tensor<float> w = MakeFilled<float>(miopenTensorNCHW, {8, 8, 3, 3}, {}, 17);
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
    Workspace wspace{workspace_size};

    constexpr int kRequested = 4;
    int returned_algo_count  = 0;
    std::array<miopenConvAlgoPerf_t, kRequested> perfs{};
    ASSERT_EQ(miopenFindConvolutionForwardAlgorithm(handle,
                                                    &x.desc,
                                                    x_dev.get(),
                                                    &w.desc,
                                                    w_dev.get(),
                                                    conv.handle,
                                                    &y.desc,
                                                    y_dev.get(),
                                                    kRequested,
                                                    &returned_algo_count,
                                                    perfs.data(),
                                                    wspace.ptr(),
                                                    wspace.size(),
                                                    false),
              miopenStatusSuccess);
    ASSERT_GT(returned_algo_count, 0);

    for(int i = 0; i < returned_algo_count; ++i)
        workspace_size = std::max(workspace_size, perfs[i].memory);
    wspace.resize(workspace_size);

    for(int i = 0; i < returned_algo_count; ++i)
    {
        SCOPED_TRACE("algorithm " + std::to_string(perfs[i].fwd_algo));
        // Otherwise an algorithm that writes nothing would pass on what Find or the last one wrote.
        std::fill(y.data.begin(), y.data.end(), 0.0f);
        y_dev = handle_deref.Write(y.data);
        ASSERT_EQ(miopenConvolutionForward(handle,
                                           &kOne,
                                           &x.desc,
                                           x_dev.get(),
                                           &w.desc,
                                           w_dev.get(),
                                           conv.handle,
                                           perfs[i].fwd_algo,
                                           &kZero,
                                           &y.desc,
                                           y_dev.get(),
                                           wspace.ptr(),
                                           wspace.size()),
                  miopenStatusSuccess);
        y.data = handle_deref.Read<float>(y_dev, y.data.size());
        ASSERT_NO_FATAL_FAILURE(CheckMatchesCpuReference(x, w, y));
    }
}
