// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_depthwise_row_strip_bf16.hpp"
#include "ck/utility/type_convert.hpp"

namespace {

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using BaseOp =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeight<2,
                                                             ck::tensor_layout::convolution::NHWGC,
                                                             ck::tensor_layout::convolution::GKYXC,
                                                             ck::tensor_layout::convolution::NHWGK,
                                                             ck::bhalf_t,
                                                             ck::bhalf_t,
                                                             ck::bhalf_t,
                                                             PassThrough,
                                                             PassThrough,
                                                             PassThrough>;
template <ck::index_t FilterSize = 11>
using RowStripOp =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeightDepthwiseRowStripBf16<FilterSize>;

struct Shape
{
    int n;
    int g;
    int h;
    int w;
    int filter = 11;
    int stride = 1;
    int dilation = 1;
    int pad = -1;
    int right_pad = -1;
    int stride_w = 0;

    int Pad() const { return pad < 0 ? (filter - 1) * dilation / 2 : pad; }
    int RightPad() const { return right_pad < 0 ? Pad() : right_pad; }
    int StrideW() const { return stride_w == 0 ? stride : stride_w; }
    int OutH() const { return (h + Pad() + RightPad() - (filter - 1) * dilation - 1) / stride + 1; }
    int OutW() const { return (w + Pad() + RightPad() - (filter - 1) * dilation - 1) / StrideW() + 1; }
    int FilterElements() const { return filter * filter; }
};

template <typename Index>
struct Problem
{
    std::array<Index, 5> in_lengths, in_strides, wei_lengths, wei_strides, out_lengths, out_strides;
    std::array<Index, 2> filter_strides, filter_dilations, left_pads, right_pads;

    explicit Problem(Shape shape)
        : in_lengths{shape.g, shape.n, 1, shape.h, shape.w},
          in_strides{1, static_cast<Index>(shape.h) * shape.w * shape.g, 1,
                     static_cast<Index>(shape.w) * shape.g, shape.g},
          wei_lengths{shape.g, 1, 1, shape.filter, shape.filter},
          wei_strides{shape.FilterElements(), shape.FilterElements(), 1, shape.filter, 1},
          out_lengths{shape.g, shape.n, 1, shape.OutH(), shape.OutW()},
          out_strides{1, static_cast<Index>(shape.OutH()) * shape.OutW() * shape.g, 1,
                      static_cast<Index>(shape.OutW()) * shape.g, shape.g},
          filter_strides{shape.stride, shape.StrideW()},
          filter_dilations{shape.dilation, shape.dilation},
          left_pads{shape.Pad(), shape.Pad()},
          right_pads{shape.RightPad(), shape.RightPad()}
    {
    }

    std::unique_ptr<ck::tensor_operation::device::BaseArgument>
    MakeArgument(BaseOp& op, const void* in, void* wei, const void* out, ck::index_t split) const
    {
        return op.MakeArgumentPointer(in,
                                      wei,
                                      out,
                                      in_lengths,
                                      in_strides,
                                      wei_lengths,
                                      wei_strides,
                                      out_lengths,
                                      out_strides,
                                      filter_strides,
                                      filter_dilations,
                                      left_pads,
                                      right_pads,
                                      PassThrough{},
                                      PassThrough{},
                                      PassThrough{},
                                      split);
    }
};

std::size_t NchwOffset(int n, int g, int h, int w, int groups, int height, int width)
{
    return ((static_cast<std::size_t>(n) * groups + g) * height + h) * width + w;
}

std::size_t NhwgcOffset(int n, int g, int h, int w, int groups, int height, int width)
{
    return ((static_cast<std::size_t>(n) * height + h) * width + w) * groups + g;
}

struct Reference
{
    std::vector<double> weights;
    std::vector<double> partials;
    std::vector<double> partial_magnitudes;
    std::vector<double> weight_magnitudes;
};

// Independent NCHW serial convolution: filter-major, then n/ho/wo. The
// per-strip totals are recorded only to check every private workspace owner.
Reference ComputeReference(Shape shape,
                           const std::vector<ck::bhalf_t>& x_nchw,
                           const std::vector<ck::bhalf_t>& dy_nchw)
{
    const int strips = (shape.OutH() + 11) / 12;
    const std::size_t filter_count = static_cast<std::size_t>(shape.g) * shape.FilterElements();
    Reference ref{std::vector<double>(filter_count),
                  std::vector<double>(static_cast<std::size_t>(shape.n) * strips * filter_count),
                  std::vector<double>(static_cast<std::size_t>(shape.n) * strips * filter_count),
                  std::vector<double>(filter_count)};
    for(int g = 0; g < shape.g; ++g)
    {
        for(int fy = 0; fy < shape.filter; ++fy)
        {
            for(int fx = 0; fx < shape.filter; ++fx)
            {
                const std::size_t f =
                    static_cast<std::size_t>(g) * shape.FilterElements() + fy * shape.filter + fx;
                for(int n = 0; n < shape.n; ++n)
                {
                    for(int ho = 0; ho < shape.OutH(); ++ho)
                    {
                        const int hi = ho * shape.stride + fy * shape.dilation - shape.Pad();
                        if(hi < 0 || hi >= shape.h)
                            continue;
                        for(int wo = 0; wo < shape.OutW(); ++wo)
                        {
                            const int wi = wo * shape.StrideW() + fx * shape.dilation - shape.Pad();
                            if(wi < 0 || wi >= shape.w)
                                continue;
                            const double x = ck::type_convert<float>(
                                x_nchw[NchwOffset(n, g, hi, wi, shape.g, shape.h, shape.w)]);
                            const double dy = ck::type_convert<float>(
                                dy_nchw[NchwOffset(n, g, ho, wo, shape.g, shape.OutH(), shape.OutW())]);
                            const double product = x * dy;
                            const std::size_t p =
                                (static_cast<std::size_t>(n) * strips + ho / 12) * filter_count + f;
                            ref.weights[f] += product;
                            ref.weight_magnitudes[f] += std::abs(product);
                            ref.partials[p] += product;
                            ref.partial_magnitudes[p] += std::abs(product);
                        }
                    }
                }
            }
        }
    }
    return ref;
}

void FillInputs(Shape shape,
                int repetition,
                std::vector<ck::bhalf_t>& x_nchw,
                std::vector<ck::bhalf_t>& dy_nchw,
                std::vector<ck::bhalf_t>& x_packed,
                std::vector<ck::bhalf_t>& dy_packed)
{
    constexpr std::array<float, 5> large_x{64.f, 0.25f, 4.f, 0.5f, 16.f};
    constexpr std::array<float, 6> large_dy{16.f, 0.0625f, 1.f, 0.25f, 4.f, 0.5f};
    for(int n = 0; n < shape.n; ++n)
    {
        for(int g = 0; g < shape.g; ++g)
        {
            for(int h = 0; h < shape.h; ++h)
            {
                for(int w = 0; w < shape.w; ++w)
                {
                    const float x_sign = (n * 7 + g * 3 + h * 5 + w) % 11 < 5 ? -1.f : 1.f;
                    const float x_value = repetition == 0
                                              ? 0.5f * (1 + (n + 2 * g + h + w) % 5)
                                              : large_x[(n + 2 * g + 3 * h + w) % large_x.size()];
                    const auto nchw = NchwOffset(n, g, h, w, shape.g, shape.h, shape.w);
                    const auto packed = NhwgcOffset(n, g, h, w, shape.g, shape.h, shape.w);
                    x_nchw[nchw] = x_packed[packed] =
                        ck::type_convert<ck::bhalf_t>(x_sign * x_value);
                }
            }
            for(int h = 0; h < shape.OutH(); ++h)
            {
                for(int w = 0; w < shape.OutW(); ++w)
                {
                    const float dy_sign = (n * 3 + g * 5 + h + w * 7) % 13 < 6 ? -1.f : 1.f;
                    const float dy_value =
                        repetition == 0 ? 0.25f * (1 + (3 * n + g + 2 * h + w) % 7)
                                        : large_dy[(2 * n + g + h + 3 * w) % large_dy.size()];
                    const auto nchw = NchwOffset(n, g, h, w, shape.g, shape.OutH(), shape.OutW());
                    const auto packed = NhwgcOffset(n, g, h, w, shape.g, shape.OutH(), shape.OutW());
                    dy_nchw[nchw] = dy_packed[packed] =
                        ck::type_convert<ck::bhalf_t>(dy_sign * dy_value);
                }
            }
        }
    }
}

std::size_t WorkspaceBytes(Shape shape)
{
    const std::size_t logical =
        static_cast<std::size_t>(shape.n) * ((shape.OutH() + 11) / 12) * shape.g *
        shape.FilterElements() * sizeof(float);
    return (logical + 255) / 256 * 256;
}

template <ck::index_t FilterSize = 11>
void CheckShape(Shape shape)
{
    RowStripOp<FilterSize> concrete;
    BaseOp& op = concrete; // Exercise the same virtual interface used by the factory.
    const Problem<ck::index_t> problem(shape);

    const std::size_t input_count = static_cast<std::size_t>(shape.n) * shape.g * shape.h * shape.w;
    const std::size_t output_count =
        static_cast<std::size_t>(shape.n) * shape.g * shape.OutH() * shape.OutW();
    const std::size_t filter_count = static_cast<std::size_t>(shape.g) * shape.FilterElements();
    std::vector<ck::bhalf_t> x_nchw(input_count), dy_nchw(output_count), x_packed(input_count),
        dy_packed(output_count), actual(filter_count);
    ck::DeviceMem x_device(input_count * sizeof(ck::bhalf_t));
    ck::DeviceMem dy_device(output_count * sizeof(ck::bhalf_t));
    ck::DeviceMem dw_device(filter_count * sizeof(ck::bhalf_t));
    auto arg = problem.MakeArgument(op,
                                    x_device.GetDeviceBuffer(),
                                    dw_device.GetDeviceBuffer(),
                                    dy_device.GetDeviceBuffer(),
                                    1);
    ASSERT_TRUE(op.IsSupportedArgument(arg.get()));
    const std::size_t workspace_bytes = op.GetWorkSpaceSize(arg.get());
    ASSERT_EQ(workspace_bytes, WorkspaceBytes(shape));
    ck::DeviceMem workspace(workspace_bytes);
    auto invoker = op.MakeInvokerPointer();
    EXPECT_THROW(invoker->Run(arg.get(), StreamConfig{nullptr, false}), std::runtime_error);
    op.SetWorkSpacePointer(arg.get(),
                           static_cast<std::uint8_t*>(workspace.GetDeviceBuffer()) + 1);
    EXPECT_THROW(invoker->Run(arg.get(), StreamConfig{nullptr, false}), std::runtime_error);
    op.SetWorkSpacePointer(arg.get(), workspace.GetDeviceBuffer());
    std::vector<std::uint8_t> poison_workspace(workspace_bytes, 0xff);
    std::vector<std::uint8_t> poison_dw(filter_count * sizeof(ck::bhalf_t), 0xff);
    std::vector<float> observed_partials(workspace_bytes / sizeof(float));

    for(int repetition = 0; repetition < 2; ++repetition)
    {
        FillInputs(shape, repetition, x_nchw, dy_nchw, x_packed, dy_packed);
        const auto reference = ComputeReference(shape, x_nchw, dy_nchw);
        x_device.ToDevice(x_packed.data());
        dy_device.ToDevice(dy_packed.data());
        dw_device.ToDevice(poison_dw.data());
        workspace.ToDevice(poison_workspace.data());
        invoker->Run(arg.get(), StreamConfig{nullptr, false});
        dw_device.FromDevice(actual.data());
        workspace.FromDevice(observed_partials.data());

        for(std::size_t i = 0; i < filter_count; ++i)
        {
            const double got  = ck::type_convert<float>(actual[i]);
            const double want = reference.weights[i];
            const double tolerance =
                std::max(0.03125, 0.008 * std::abs(want) + 0.0001 * reference.weight_magnitudes[i]);
            EXPECT_TRUE(std::isfinite(got)) << "weight=" << i << " repetition=" << repetition;
            EXPECT_NEAR(got, want, tolerance)
                << "group=" << i / shape.FilterElements()
                << " fy=" << (i % shape.FilterElements()) / shape.filter
                << " fx=" << i % shape.filter
                << " repetition=" << repetition;
            if(reference.weight_magnitudes[i] <= 0)
                EXPECT_EQ(got, 0) << "untouched filter weight=" << i;
        }
        for(std::size_t i = 0; i < reference.partials.size(); ++i)
        {
            const double got       = observed_partials[i];
            const double want      = reference.partials[i];
            const double tolerance = 0.001 + 0.0001 * reference.partial_magnitudes[i];
            EXPECT_TRUE(std::isfinite(got)) << "partial=" << i << " repetition=" << repetition;
            EXPECT_NEAR(got, want, tolerance)
                << "strip=" << i / filter_count
                << " group=" << (i % filter_count) / shape.FilterElements()
                << " filter=" << i % shape.FilterElements() << " repetition=" << repetition;
            if(reference.partial_magnitudes[i] <= 0)
                EXPECT_EQ(got, 0) << "empty partial=" << i;
        }
        const auto first_weights  = actual;
        const auto first_partials = observed_partials;
        dw_device.ToDevice(poison_dw.data());
        workspace.ToDevice(poison_workspace.data());
        invoker->Run(arg.get(), StreamConfig{nullptr, false});
        dw_device.FromDevice(actual.data());
        workspace.FromDevice(observed_partials.data());
        for(std::size_t i = 0; i < filter_count; ++i)
            EXPECT_EQ(ck::type_convert<float>(actual[i]), ck::type_convert<float>(first_weights[i]))
                << "weight=" << i;
        for(std::size_t i = 0; i < reference.partials.size(); ++i)
            EXPECT_EQ(observed_partials[i], first_partials[i]) << "partial=" << i;
        if(repetition == 0)
        {
            auto* typed_arg = dynamic_cast<typename RowStripOp<FilterSize>::Argument*>(arg.get());
            ASSERT_NE(typed_arg, nullptr);
            const bool original_geometry =
                shape.filter == 11 && shape.stride == 1 && shape.StrideW() == 1 &&
                shape.dilation == 1 && shape.Pad() == 5 && shape.RightPad() == 5;
            EXPECT_EQ(typed_arg->narrow_device_indices, original_geometry);
            if(!typed_arg->narrow_device_indices)
                continue;
            typed_arg->narrow_device_indices = false;
            dw_device.ToDevice(poison_dw.data());
            workspace.ToDevice(poison_workspace.data());
            invoker->Run(arg.get(), StreamConfig{nullptr, false});
            dw_device.FromDevice(actual.data());
            workspace.FromDevice(observed_partials.data());
            for(std::size_t i = 0; i < filter_count; ++i)
                EXPECT_EQ(ck::type_convert<float>(actual[i]),
                          ck::type_convert<float>(first_weights[i]))
                    << "wide weight=" << i;
            for(std::size_t i = 0; i < reference.partials.size(); ++i)
                EXPECT_EQ(observed_partials[i], first_partials[i]) << "wide partial=" << i;
            typed_arg->narrow_device_indices = true;
        }
    }
}

TEST(TestGroupedConvndBwdWeightRowStripBf16, DryQueriesAndAdmission)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "gfx1250-only candidate";

    RowStripOp<> concrete;
    BaseOp& op = concrete;
    const Shape shape{2, 3, 13, 131};
    const Problem<ck::index_t> problem(shape);
    const Problem<ck::long_index_t> long_problem(shape);
    alignas(256) std::array<std::uint8_t, 256> dry_workspace{};
    for(const ck::index_t split : {-1, 0, 1})
    {
        auto dry = problem.MakeArgument(op, nullptr, nullptr, nullptr, split);
        ASSERT_TRUE(op.IsSupportedArgument(dry.get())) << "split=" << split;
        EXPECT_EQ(op.GetWorkSpaceSize(dry.get()), WorkspaceBytes(shape));
        op.SetWorkSpacePointer(dry.get(), dry_workspace.data());
        EXPECT_THROW(op.MakeInvokerPointer()->Run(dry.get(), StreamConfig{nullptr, false}),
                     std::runtime_error);
        auto long_dry = long_problem.MakeArgument(op, nullptr, nullptr, nullptr, split);
        ASSERT_TRUE(op.IsSupportedArgument(long_dry.get())) << "long split=" << split;
        EXPECT_EQ(op.GetWorkSpaceSize(long_dry.get()), WorkspaceBytes(shape));
        op.SetWorkSpacePointer(long_dry.get(), dry_workspace.data());
    }
    auto small_dry            = long_problem.MakeArgument(op, nullptr, nullptr, nullptr, 1);
    const auto* small_indices = dynamic_cast<const RowStripOp<>::Argument*>(small_dry.get());
    ASSERT_NE(small_indices, nullptr);
    EXPECT_TRUE(small_indices->narrow_device_indices);
    // Query near INT_MAX without allocating a huge input; wide indexing remains legal.
    const Problem<ck::long_index_t> wide_problem(Shape{1, 3, 1, 715827881});
    auto wide_dry = wide_problem.MakeArgument(op, nullptr, nullptr, nullptr, 1);
    ASSERT_TRUE(op.IsSupportedArgument(wide_dry.get()));
    const auto* wide_indices = dynamic_cast<const RowStripOp<>::Argument*>(wide_dry.get());
    ASSERT_NE(wide_indices, nullptr);
    EXPECT_FALSE(wide_indices->narrow_device_indices);
    EXPECT_EQ(op.GetWorkSpaceSize(wide_dry.get()), WorkspaceBytes({1, 3, 1, 715827881}));
    // Cover small positive groups independently of batch, spatial extent and split.
    for(const int groups : {1, 2, 3, 4, 5, 6, 7})
    {
        for(const Shape covered :
            {Shape{2, groups, 13, 131}, Shape{1, groups, 25, 17}, Shape{1, groups, 1, 1}})
        {
            const Problem<ck::index_t> covered_problem(covered);
            const Problem<ck::long_index_t> covered_long_problem(covered);
            for(const ck::index_t split : {-1, 0, 1})
            {
                auto covered_dry =
                    covered_problem.MakeArgument(op, nullptr, nullptr, nullptr, split);
                ASSERT_TRUE(op.IsSupportedArgument(covered_dry.get()))
                    << "G=" << groups << " split=" << split;
                EXPECT_EQ(op.GetWorkSpaceSize(covered_dry.get()), WorkspaceBytes(covered));
                auto covered_long_dry =
                    covered_long_problem.MakeArgument(op, nullptr, nullptr, nullptr, split);
                ASSERT_TRUE(op.IsSupportedArgument(covered_long_dry.get()))
                    << "long G=" << groups << " split=" << split;
                EXPECT_EQ(op.GetWorkSpaceSize(covered_long_dry.get()), WorkspaceBytes(covered));
            }
        }
    }
    for(const ck::index_t split : {-2, 2, 3})
    {
        auto dry = problem.MakeArgument(op, nullptr, nullptr, nullptr, split);
        EXPECT_FALSE(op.IsSupportedArgument(dry.get())) << "split=" << split;
    }

    auto wrong_input_stride = problem;
    ++wrong_input_stride.in_strides[4];
    EXPECT_FALSE(op.IsSupportedArgument(
        wrong_input_stride.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_output_stride = problem;
    ++wrong_output_stride.out_strides[0];
    EXPECT_FALSE(op.IsSupportedArgument(
        wrong_output_stride.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_weight_stride = problem;
    ++wrong_weight_stride.wei_strides[0];
    EXPECT_FALSE(op.IsSupportedArgument(
        wrong_weight_stride.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_filter           = problem;
    wrong_filter.wei_lengths[3] = 3;
    EXPECT_FALSE(
        op.IsSupportedArgument(wrong_filter.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_filter_width           = problem;
    wrong_filter_width.wei_lengths[4] = 3;
    EXPECT_FALSE(op.IsSupportedArgument(
        wrong_filter_width.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_pad         = problem;
    wrong_pad.left_pads[0] = 4;
    EXPECT_FALSE(
        op.IsSupportedArgument(wrong_pad.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_right_pad          = problem;
    wrong_right_pad.right_pads[1] = 4;
    EXPECT_FALSE(op.IsSupportedArgument(
        wrong_right_pad.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_dilation                = problem;
    wrong_dilation.filter_dilations[1] = 2;
    EXPECT_FALSE(op.IsSupportedArgument(
        wrong_dilation.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_filter_stride              = problem;
    wrong_filter_stride.filter_strides[0] = 2;
    EXPECT_FALSE(op.IsSupportedArgument(
        wrong_filter_stride.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    for(const int groups : {8, 64, 4096})
    {
        const Problem<ck::index_t> larger_groups(Shape{2, groups, 13, 131});
        EXPECT_TRUE(op.IsSupportedArgument(
            larger_groups.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()))
            << "G=" << groups;
    }
    auto huge           = long_problem;
    huge.out_lengths[3] = std::numeric_limits<ck::long_index_t>::max();
    EXPECT_FALSE(op.IsSupportedArgument(huge.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    // Genuine per-dimension launch and aligned scratch bounds, without allocation.
    const Problem<ck::long_index_t> too_many_splits(Shape{16777216, 1, 1, 1});
    EXPECT_FALSE(op.IsSupportedArgument(
        too_many_splits.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    const Problem<ck::long_index_t> too_many_weights(Shape{1, 2218475, 1, 1});
    EXPECT_FALSE(op.IsSupportedArgument(
        too_many_weights.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    const Problem<ck::long_index_t> scratch_overflow(
        Shape{std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), 1, 1});
    EXPECT_FALSE(op.IsSupportedArgument(
        scratch_overflow.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto negative_pad = long_problem;
    negative_pad.left_pads[0] = -1;
    EXPECT_FALSE(op.IsSupportedArgument(
        negative_pad.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto oversized_stride = long_problem;
    oversized_stride.filter_strides[0] =
        static_cast<ck::long_index_t>(std::numeric_limits<ck::index_t>::max()) + 1;
    EXPECT_FALSE(op.IsSupportedArgument(
        oversized_stride.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
}

TEST(TestGroupedConvndBwdWeightRowStripBf16, NchwSerialReferenceAndWorkspaceOwnership)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "gfx1250-only candidate";

    CheckShape({2, 1, 13, 131}); // batch, multiple strips, 128-lane column tail
    CheckShape({1, 2, 25, 17});  // three strips, short columns
    CheckShape({2, 3, 13, 131}); // original G3 row, column, batch and filter tails
    CheckShape({1, 3, 25, 17});  // original G3 three-strip workload
    CheckShape({1, 3, 1, 1});    // original G3 empty filter-row/column partials
    CheckShape({2, 4, 13, 131});
    CheckShape({1, 5, 1, 1});
    CheckShape({2, 6, 25, 17});
    CheckShape({2, 7, 13, 131});
    CheckShape({2, 8, 13, 131});
    CheckShape({2, 64, 13, 131});
    CheckShape({2, 4096, 13, 131});
    CheckShape<5>({2, 17, 25, 131, 5, 2});
    CheckShape<7>({2, 31, 13, 133, 7, 2});
    CheckShape<3>({2, 15, 17, 130, 3, 3, 2, 1, 3, 2});
    CheckShape({2, 9, 25, 133, 11, 2});
}

} // namespace
