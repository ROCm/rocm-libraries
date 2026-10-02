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
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_depthwise_grouped_row_strip_bf16.hpp"
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
template <ck::index_t FilterSize = 3>
using RowStripOp =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeightDepthwiseGroupedRowStripBf16<FilterSize>;

struct Shape
{
    int n;
    int g;
    int h;
    int w;
    int stride = 2;
    int filter = 3;
    int dilation = 1;
    int pad = -1;
    int right_pad = -1;
    int stride_w = 0;

    int Pad() const { return pad < 0 ? (filter - 1) * dilation / 2 : pad; }
    int RightPad() const { return right_pad < 0 ? Pad() : right_pad; }
    int StrideW() const { return stride_w == 0 ? stride : stride_w; }
    int FilterElements() const { return filter * filter; }

    int OutH() const { return (h + Pad() + RightPad() - (filter - 1) * dilation - 1) / stride + 1; }
    int OutW() const { return (w + Pad() + RightPad() - (filter - 1) * dilation - 1) / StrideW() + 1; }
    int Strips() const { return (OutH() + 7) / 8 * ((OutW() + 79) / 80); }
};

template <typename Index>
struct Problem
{
    std::array<Index, 5> in_lengths, in_strides, wei_lengths, wei_strides, out_lengths, out_strides;
    std::array<Index, 2> filter_strides;
    std::array<Index, 2> filter_dilations, left_pads, right_pads;

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

std::size_t PackedOffset(int n, int g, int h, int w, int groups, int height, int width)
{
    return ((static_cast<std::size_t>(n) * height + h) * width + w) * groups + g;
}

struct Reference
{
    std::vector<double> weights;
    std::vector<double> partials;
    std::vector<double> weight_magnitudes;
    std::vector<double> partial_magnitudes;
};

// Intentionally independent of the GPU's group/lane mapping and NHWGC addressing.
// NCHW serial convolution accumulates each filter element in double precision.
Reference ComputeReference(Shape shape,
                           const std::vector<ck::bhalf_t>& x_nchw,
                           const std::vector<ck::bhalf_t>& dy_nchw)
{
    const std::size_t weight_count = static_cast<std::size_t>(shape.g) * shape.FilterElements();
    const std::size_t partial_count =
        static_cast<std::size_t>(shape.n) * shape.Strips() * weight_count;
    Reference ref{std::vector<double>(weight_count),
                  std::vector<double>(partial_count),
                  std::vector<double>(weight_count),
                  std::vector<double>(partial_count)};
    for(int g = 0; g < shape.g; ++g)
    {
        for(int fy = 0; fy < shape.filter; ++fy)
        {
            for(int fx = 0; fx < shape.filter; ++fx)
            {
                const auto f =
                    static_cast<std::size_t>(g) * shape.FilterElements() + fy * shape.filter + fx;
                for(int n = 0; n < shape.n; ++n)
                {
                    for(int ho = 0; ho < shape.OutH(); ++ho)
                    {
                        const int hi = shape.stride * ho + fy * shape.dilation - shape.Pad();
                        if(hi < 0 || hi >= shape.h)
                            continue;
                        for(int wo = 0; wo < shape.OutW(); ++wo)
                        {
                            const int wi = shape.StrideW() * wo + fx * shape.dilation - shape.Pad();
                            if(wi < 0 || wi >= shape.w)
                                continue;
                            const double x = ck::type_convert<float>(
                                x_nchw[NchwOffset(n, g, hi, wi, shape.g, shape.h, shape.w)]);
                            const double dy        = ck::type_convert<float>(dy_nchw[NchwOffset(
                                n, g, ho, wo, shape.g, shape.OutH(), shape.OutW())]);
                            const double product   = x * dy;
                            const int width_strips = (shape.OutW() + 79) / 80;
                            const auto p           = (static_cast<std::size_t>(n) * shape.Strips() +
                                            (ho / 8) * width_strips + wo / 80) *
                                               weight_count +
                                           f;
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
    constexpr std::array<float, 6> x_levels{0.25f, 4.f, 0.5f, 16.f, 1.f, 64.f};
    constexpr std::array<float, 6> dy_levels{8.f, 0.125f, 2.f, 0.5f, 16.f, 0.25f};
    for(int n = 0; n < shape.n; ++n)
    {
        for(int g = 0; g < shape.g; ++g)
        {
            for(int h = 0; h < shape.h; ++h)
            {
                for(int w = 0; w < shape.w; ++w)
                {
                    const int x_key    = n * 17 + g * 13 + h * 7 + w * 3 + repetition * 11;
                    const float x_sign = x_key % 11 < 5 ? -1.f : 1.f;
                    x_nchw[NchwOffset(n, g, h, w, shape.g, shape.h, shape.w)] =
                        ck::type_convert<ck::bhalf_t>(x_sign * x_levels[x_key % x_levels.size()]);
                }
            }
            for(int h = 0; h < shape.OutH(); ++h)
            {
                for(int w = 0; w < shape.OutW(); ++w)
                {
                    const int dy_key    = n * 5 + g * 19 + h * 11 + w * 7 + repetition * 23;
                    const float dy_sign = dy_key % 13 < 6 ? -1.f : 1.f;
                    dy_nchw[NchwOffset(n, g, h, w, shape.g, shape.OutH(), shape.OutW())] =
                        ck::type_convert<ck::bhalf_t>(dy_sign *
                                                      dy_levels[dy_key % dy_levels.size()]);
                }
            }
        }
    }
    for(int n = 0; n < shape.n; ++n)
    {
        for(int h = 0; h < shape.h; ++h)
        {
            for(int w = 0; w < shape.w; ++w)
            {
                for(int g = 0; g < shape.g; ++g)
                {
                    x_packed[PackedOffset(n, g, h, w, shape.g, shape.h, shape.w)] =
                        x_nchw[NchwOffset(n, g, h, w, shape.g, shape.h, shape.w)];
                }
            }
        }
        for(int h = 0; h < shape.OutH(); ++h)
        {
            for(int w = 0; w < shape.OutW(); ++w)
            {
                for(int g = 0; g < shape.g; ++g)
                {
                    dy_packed[PackedOffset(n, g, h, w, shape.g, shape.OutH(), shape.OutW())] =
                        dy_nchw[NchwOffset(n, g, h, w, shape.g, shape.OutH(), shape.OutW())];
                }
            }
        }
    }
}

template <ck::index_t FilterSize = 3>
void CheckShape(Shape shape, bool canonical_right_pad = false)
{
    RowStripOp<FilterSize> concrete;
    BaseOp& op = concrete;
    Problem<ck::index_t> problem(shape);
    if(canonical_right_pad)
        problem.right_pads = {shape.Pad() - 1, shape.Pad() - 1};

    const auto x_count  = static_cast<std::size_t>(shape.n) * shape.g * shape.h * shape.w;
    const auto dy_count = static_cast<std::size_t>(shape.n) * shape.g * shape.OutH() * shape.OutW();
    const auto weight_count = static_cast<std::size_t>(shape.g) * shape.FilterElements();
    std::vector<ck::bhalf_t> x_nchw(x_count), dy_nchw(dy_count), x_packed(x_count),
        dy_packed(dy_count), actual(weight_count);
    ck::DeviceMem x_device(x_count * sizeof(ck::bhalf_t));
    ck::DeviceMem dy_device(dy_count * sizeof(ck::bhalf_t));
    ck::DeviceMem dw_device(weight_count * sizeof(ck::bhalf_t));
    auto workspace_arg = problem.MakeArgument(op,
                                              x_device.GetDeviceBuffer(),
                                              dw_device.GetDeviceBuffer(),
                                              dy_device.GetDeviceBuffer(),
                                              1);
    ASSERT_TRUE(op.IsSupportedArgument(workspace_arg.get()));
    const auto workspace_bytes = op.GetWorkSpaceSize(workspace_arg.get());
    const auto logical_bytes =
        static_cast<std::size_t>(shape.n) * shape.Strips() * weight_count * sizeof(float);
    ASSERT_EQ(workspace_bytes, (logical_bytes + 255) / 256 * 256);
    ck::DeviceMem workspace(workspace_bytes);
    std::vector<std::uint8_t> poison_workspace(workspace_bytes, 0xff);
    std::vector<std::uint8_t> poison_dw(weight_count * sizeof(ck::bhalf_t), 0xff);
    std::vector<float> observed_partials(workspace_bytes / sizeof(float));
    auto invoker = op.MakeInvokerPointer();
    std::vector<double> old_weights;

    for(int repetition = 0; repetition < 2; ++repetition)
    {
        FillInputs(shape, repetition, x_nchw, dy_nchw, x_packed, dy_packed);
        const auto reference  = ComputeReference(shape, x_nchw, dy_nchw);
        const auto tail_group = shape.g - 1;
        const auto tail_center =
            (static_cast<std::size_t>(shape.n) * shape.Strips() - 1) * weight_count +
            tail_group * shape.FilterElements() + shape.FilterElements() / 2;
        ASSERT_GT(reference.partial_magnitudes[tail_center], 0);
        EXPECT_GT(reference.weight_magnitudes[
                      tail_group * shape.FilterElements() + shape.FilterElements() / 2], 0);
        if(repetition == 1)
            EXPECT_NE(reference.weights, old_weights) << "the second call must change input";
        old_weights = reference.weights;
        x_device.ToDevice(x_packed.data());
        dy_device.ToDevice(dy_packed.data());

        std::vector<float> first_weights;
        std::vector<float> first_partials;
        for(const ck::index_t split : {-1, 0, 1})
        {
            auto arg = problem.MakeArgument(op,
                                            x_device.GetDeviceBuffer(),
                                            dw_device.GetDeviceBuffer(),
                                            dy_device.GetDeviceBuffer(),
                                            split);
            ASSERT_TRUE(op.IsSupportedArgument(arg.get())) << "split=" << split;
            EXPECT_THROW(invoker->Run(arg.get(), StreamConfig{nullptr, false}), std::runtime_error);
            op.SetWorkSpacePointer(arg.get(),
                                   static_cast<std::uint8_t*>(workspace.GetDeviceBuffer()) + 1);
            EXPECT_THROW(invoker->Run(arg.get(), StreamConfig{nullptr, false}), std::runtime_error);
            op.SetWorkSpacePointer(arg.get(), workspace.GetDeviceBuffer());
            dw_device.ToDevice(poison_dw.data());
            workspace.ToDevice(poison_workspace.data());
            invoker->Run(arg.get(), StreamConfig{nullptr, false});
            dw_device.FromDevice(actual.data());
            workspace.FromDevice(observed_partials.data());
            if(split == -1)
            {
                first_weights.reserve(weight_count);
                for(const auto value : actual)
                    first_weights.push_back(ck::type_convert<float>(value));
                first_partials.assign(observed_partials.begin(),
                                      observed_partials.begin() + reference.partials.size());
            }
            else
            {
                for(std::size_t i = 0; i < weight_count; ++i)
                    EXPECT_EQ(ck::type_convert<float>(actual[i]), first_weights[i])
                        << "weight=" << i;
                for(std::size_t i = 0; i < reference.partials.size(); ++i)
                    EXPECT_EQ(observed_partials[i], first_partials[i]) << "partial=" << i;
            }

            for(std::size_t i = 0; i < weight_count; ++i)
            {
                const double got       = ck::type_convert<float>(actual[i]);
                const double want      = reference.weights[i];
                const double tolerance = std::max(
                    0.03125, 0.008 * std::abs(want) + 0.0002 * reference.weight_magnitudes[i]);
                EXPECT_TRUE(std::isfinite(got)) << "weight=" << i << " split=" << split;
                EXPECT_NEAR(got, want, tolerance)
                    << "g=" << i / shape.FilterElements()
                    << " fy=" << (i % shape.FilterElements()) / shape.filter
                    << " fx=" << i % shape.filter
                    << " split=" << split << " repetition=" << repetition;
                if(reference.weight_magnitudes[i] <= 0)
                    EXPECT_EQ(got, 0) << "empty weight=" << i;
            }
            for(std::size_t i = 0; i < reference.partials.size(); ++i)
            {
                const double got       = observed_partials[i];
                const double want      = reference.partials[i];
                const double tolerance = 0.001 + 0.0001 * reference.partial_magnitudes[i];
                EXPECT_TRUE(std::isfinite(got)) << "partial=" << i << " split=" << split;
                EXPECT_NEAR(got, want, tolerance)
                    << "s=" << i / weight_count
                    << " g=" << (i % weight_count) / shape.FilterElements()
                    << " f=" << i % shape.FilterElements()
                    << " split=" << split << " repetition=" << repetition;
                if(reference.partial_magnitudes[i] <= 0)
                    EXPECT_EQ(got, 0) << "empty partial=" << i;
            }
        }
    }
}

TEST(TestGroupedConvndBwdWeightGroupedRowStripBf16, DryQueriesAndAdmission)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "gfx1250-only candidate";

    RowStripOp<> concrete;
    BaseOp& op = concrete;
    const Shape shape{2, 193, 13, 131};
    const Problem<ck::index_t> problem(shape);
    const Problem<ck::long_index_t> long_problem(shape);
    alignas(256) std::array<std::uint8_t, 256> dry_workspace{};
    for(const ck::index_t split : {-1, 0, 1})
    {
        auto dry = problem.MakeArgument(op, nullptr, nullptr, nullptr, split);
        ASSERT_TRUE(op.IsSupportedArgument(dry.get())) << "split=" << split;
        op.SetWorkSpacePointer(dry.get(), dry_workspace.data());
        EXPECT_THROW(op.MakeInvokerPointer()->Run(dry.get(), StreamConfig{nullptr, false}),
                     std::runtime_error);
        auto long_dry = long_problem.MakeArgument(op, nullptr, nullptr, nullptr, split);
        ASSERT_TRUE(op.IsSupportedArgument(long_dry.get())) << "long split=" << split;
        op.SetWorkSpacePointer(long_dry.get(), dry_workspace.data());
    }
    for(const ck::index_t split : {-2, 2, 3})
        EXPECT_FALSE(op.IsSupportedArgument(
            problem.MakeArgument(op, nullptr, nullptr, nullptr, split).get()))
            << "split=" << split;

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
    auto wrong_channels          = problem;
    wrong_channels.in_lengths[2] = 2;
    EXPECT_FALSE(op.IsSupportedArgument(
        wrong_channels.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_filter           = problem;
    wrong_filter.wei_lengths[3] = 5;
    EXPECT_FALSE(
        op.IsSupportedArgument(wrong_filter.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_output = problem;
    ++wrong_output.out_lengths[3];
    EXPECT_FALSE(
        op.IsSupportedArgument(wrong_output.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_stride              = problem;
    wrong_stride.filter_strides[1] = 1;
    EXPECT_FALSE(
        op.IsSupportedArgument(wrong_stride.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto unsupported_stride           = problem;
    unsupported_stride.filter_strides = {3, 3};
    EXPECT_FALSE(op.IsSupportedArgument(
        unsupported_stride.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_pad          = problem;
    wrong_pad.right_pads[0] = 0;
    EXPECT_FALSE(
        op.IsSupportedArgument(wrong_pad.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_dilation                = problem;
    wrong_dilation.filter_dilations[1] = 2;
    EXPECT_FALSE(op.IsSupportedArgument(
        wrong_dilation.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto huge           = long_problem;
    huge.out_lengths[3] = std::numeric_limits<ck::long_index_t>::max();
    EXPECT_FALSE(op.IsSupportedArgument(huge.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));

    const Problem<ck::index_t> below_tile(Shape{2, 15, 13, 131});
    EXPECT_TRUE(
        op.IsSupportedArgument(below_tile.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    CheckShape({2, 15, 13, 131});
    for(const Shape low_ctas : {Shape{336, 16, 5, 3},
                                Shape{255, 17, 5, 3},
                                Shape{255, 24, 5, 3},
                                Shape{255, 32, 5, 3},
                                Shape{127, 58, 5, 3},
                                Shape{85, 88, 7, 3},
                                Shape{63, 122, 9, 3},
                                Shape{46, 176, 13, 3},
                                Shape{2, 513, 13, 131},
                                Shape{11, 513, 14, 14},
                                Shape{2, 576, 18, 162},
                                Shape{11, 576, 14, 14}})
    {
        const Problem<ck::index_t> narrow(low_ctas);
        const Problem<ck::long_index_t> wide(low_ctas);
        EXPECT_TRUE(
            op.IsSupportedArgument(narrow.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()))
            << "groups=" << low_ctas.g << " CTAs=" << low_ctas.Strips() * ((low_ctas.g + 15) / 16);
        EXPECT_TRUE(
            op.IsSupportedArgument(wide.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
        CheckShape(low_ctas);
    }
    for(const Shape boundary : {Shape{256, 17, 5, 3},
                                Shape{256, 24, 5, 3},
                                Shape{256, 32, 5, 3},
                                Shape{128, 58, 5, 3},
                                Shape{256, 88, 5, 3},
                                Shape{64, 122, 9, 3},
                                Shape{47, 176, 13, 3},
                                Shape{2, 192, 13, 131},
                                Shape{2, 256, 13, 131},
                                Shape{2, 257, 13, 131},
                                Shape{2, 512, 13, 131},
                                Shape{16, 513, 14, 14},  // S=16, 528 CTAs
                                Shape{16, 576, 14, 14}}) // S=16, 576 CTAs
    {
        const Problem<ck::index_t> narrow(boundary);
        const Problem<ck::long_index_t> wide(boundary);
        for(const ck::index_t split : {-1, 0, 1})
        {
            auto narrow_arg = narrow.MakeArgument(op, nullptr, nullptr, nullptr, split);
            auto wide_arg   = wide.MakeArgument(op, nullptr, nullptr, nullptr, split);
            ASSERT_TRUE(op.IsSupportedArgument(narrow_arg.get()))
                << "groups=" << boundary.g << " split=" << split;
            ASSERT_TRUE(op.IsSupportedArgument(wide_arg.get()))
                << "long groups=" << boundary.g << " split=" << split;
        }
    }
    // Large resource-eligible candidate retains wide indexing and bounded P.
    const Problem<ck::long_index_t> wide_problem(Shape{2, 512, 630, 640, 2});
    auto wide_dry = wide_problem.MakeArgument(op, nullptr, nullptr, nullptr, 1);
    ASSERT_TRUE(op.IsSupportedArgument(wide_dry.get()));
    for(const Shape admitted : {Shape{48, 192, 56, 64, 1},
                                Shape{21, 192, 120, 80, 1},
                                Shape{8, 192, 120, 127, 1},
                                Shape{48, 512, 56, 64, 1},
                                Shape{325, 513, 1, 3}, // S=325, 10725 CTAs
                                Shape{298, 576, 1, 3}, // S=298, 10728 CTAs
                                Shape{2, 193, 9, 160, 1}})
    {
        const Problem<ck::index_t> narrow_descriptor(admitted);
        const Problem<ck::long_index_t> wide_descriptor(admitted);
        for(const ck::index_t split : {-1, 0, 1})
        {
            auto narrow = narrow_descriptor.MakeArgument(op, nullptr, nullptr, nullptr, split);
            auto wide   = wide_descriptor.MakeArgument(op, nullptr, nullptr, nullptr, split);
            ASSERT_TRUE(op.IsSupportedArgument(narrow.get()))
                << "n=" << admitted.n << " split=" << split;
            ASSERT_TRUE(op.IsSupportedArgument(wide.get()))
                << "wide n=" << admitted.n << " split=" << split;
        }
        for(const ck::index_t split : {2, 3})
        {
            EXPECT_FALSE(op.IsSupportedArgument(
                narrow_descriptor.MakeArgument(op, nullptr, nullptr, nullptr, split).get()));
            EXPECT_FALSE(op.IsSupportedArgument(
                wide_descriptor.MakeArgument(op, nullptr, nullptr, nullptr, split).get()));
        }
    }
    const Shape just_over_r{1, 192, 8, 25201, 1};    // R=201608; S=316 fits
    const Shape just_over_s{337, 192, 1, 2, 1};      // R=674; S=337
    const Shape just_over_ctas{326, 513, 1, 3};      // S=326, CTAs=10758
    for(const Shape excluded : {just_over_r, just_over_s, just_over_ctas})
    {
        const Problem<ck::index_t> narrow_descriptor(excluded);
        const Problem<ck::long_index_t> wide_descriptor(excluded);
        EXPECT_TRUE(op.IsSupportedArgument(
            narrow_descriptor.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
        EXPECT_TRUE(op.IsSupportedArgument(
            wide_descriptor.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
        CheckShape(excluded);
    }
    const Problem<ck::index_t> too_short(Shape{1, 193, 1, 159});
    EXPECT_TRUE(
        op.IsSupportedArgument(too_short.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    CheckShape({1, 193, 1, 159});
    const Shape target{42, 192, 120, 160};
    const Problem<ck::index_t> target_problem(target);
    auto target_arg = target_problem.MakeArgument(op, nullptr, nullptr, nullptr, 1);
    ASSERT_TRUE(op.IsSupportedArgument(target_arg.get()));
    auto adjusted_target       = target_problem;
    adjusted_target.right_pads = {0, 0};
    auto adjusted_arg          = adjusted_target.MakeArgument(op, nullptr, nullptr, nullptr, 1);
    ASSERT_TRUE(op.IsSupportedArgument(adjusted_arg.get()));
    for(const Shape stride_one_shape : {Shape{42, 256, 60, 80, 1}, Shape{42, 512, 30, 40, 1}})
    {
        const Problem<ck::index_t> stride_one(stride_one_shape);
        const Problem<ck::long_index_t> long_stride_one(stride_one_shape);
        for(const ck::index_t split : {-1, 0, 1})
        {
            auto dry = stride_one.MakeArgument(op, nullptr, nullptr, nullptr, split);
            ASSERT_TRUE(op.IsSupportedArgument(dry.get()))
                << "groups=" << stride_one_shape.g << " split=" << split;
            op.SetWorkSpacePointer(dry.get(), dry_workspace.data());
            EXPECT_THROW(op.MakeInvokerPointer()->Run(dry.get(), StreamConfig{nullptr, false}),
                         std::runtime_error);

            auto long_dry = long_stride_one.MakeArgument(op, nullptr, nullptr, nullptr, split);
            ASSERT_TRUE(op.IsSupportedArgument(long_dry.get()))
                << "long groups=" << stride_one_shape.g << " split=" << split;
            op.SetWorkSpacePointer(long_dry.get(), dry_workspace.data());
        }
        auto missing_height_pad          = stride_one;
        missing_height_pad.right_pads[0] = 0;
        EXPECT_FALSE(op.IsSupportedArgument(
            missing_height_pad.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
        auto missing_width_pad          = stride_one;
        missing_width_pad.right_pads[1] = 0;
        EXPECT_FALSE(op.IsSupportedArgument(
            missing_width_pad.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    }
    const Problem<ck::long_index_t> too_many_splits(Shape{16777216, 1, 1, 1, 1});
    EXPECT_FALSE(op.IsSupportedArgument(
        too_many_splits.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    const Problem<ck::long_index_t> too_many_weights(Shape{1, 29826161, 1, 1, 1});
    EXPECT_FALSE(op.IsSupportedArgument(
        too_many_weights.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    const Problem<ck::long_index_t> scratch_overflow(
        Shape{std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), 1, 1, 1});
    EXPECT_FALSE(op.IsSupportedArgument(
        scratch_overflow.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    // The producer never stores logical R in int32; its individual grids still fit.
    const Problem<ck::long_index_t> large_reduction(Shape{2, 1, 8, 200000000, 1});
    EXPECT_TRUE(op.IsSupportedArgument(
        large_reduction.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
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

TEST(TestGroupedConvndBwdWeightGroupedRowStripBf16, NchwSerialReferenceAndWorkspaceOwnership)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "gfx1250-only candidate";

    CheckShape({256, 17, 5, 3});            // 512 CTAs, first group-tail tile
    CheckShape({256, 24, 5, 3});            // 512 CTAs, half-filled last tile
    CheckShape({256, 32, 5, 3});            // 512 CTAs, two full tiles
    CheckShape({128, 58, 5, 3});            // 512 CTAs, partial fourth tile
    CheckShape({256, 88, 5, 3});            // 1536 CTAs, partial sixth tile
    CheckShape({64, 122, 9, 3});            // 512 CTAs, partial eighth tile
    CheckShape({47, 176, 13, 3});           // 517 CTAs, eleven full tiles
    CheckShape({16, 513, 14, 14});          // 528 CTAs, odd group tail
    CheckShape({16, 576, 14, 14}, true);    // 576 CTAs, adjusted right padding
    CheckShape({2, 193, 13, 131});          // one strip, odd width and the ninth group-lane tail
    CheckShape({5, 193, 25, 17});           // two strips per image, short width and row tails
    CheckShape({7, 193, 1, 159});           // both outer filter rows are empty for every split
    CheckShape({3, 257, 25, 17, 1});        // group tail, four strips with one row in the last
    CheckShape({2, 512, 13, 39, 1});        // full group tiles and a five-row strip tail
    CheckShape({7, 193, 1, 79, 1});         // empty outer filter rows at stride one
    CheckShape({7, 193, 1, 81, 1});         // new width strip, empty outer filter rows
    CheckShape({2, 193, 7, 127, 1});        // row seven and a short second width strip
    CheckShape({2, 257, 8, 159, 1});        // eight rows, group tail, second width strip
    CheckShape({2, 193, 9, 160, 1});        // ninth row and exact two-strip width
    CheckShape({2, 193, 17, 161, 2});       // stride two, width 81
    CheckShape({2, 193, 18, 162, 2}, true); // even extents, canonical right pad zero
    CheckShape({2, 1, 1, 1, 1});
    CheckShape<5>({2, 17, 19, 163, 2, 5});
    CheckShape<7>({2, 31, 17, 81, 2, 7});
    CheckShape<7>({2, 9, 9, 83, 1, 7});
    CheckShape({2, 15, 17, 130, 3, 3, 2, 1, 3, 2});
    CheckShape<11>({1, 17, 9, 17, 2, 11});
}

} // namespace
