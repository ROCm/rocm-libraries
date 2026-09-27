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
using RowStripOp =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeightDepthwiseGroupedRowStripBf16;

struct Shape
{
    int n;
    int g;
    int h;
    int w;

    int OutH() const { return (h + 1) / 2; }
    int OutW() const { return (w + 1) / 2; }
    int Strips() const { return (OutH() + 7) / 8; }
};

template <typename Index>
struct Problem
{
    std::array<Index, 5> in_lengths, in_strides, wei_lengths, wei_strides, out_lengths, out_strides;
    std::array<Index, 2> filter_strides{2, 2}, filter_dilations{1, 1}, left_pads{1, 1},
        right_pads{1, 1};

    explicit Problem(Shape shape)
        : in_lengths{shape.g, shape.n, 1, shape.h, shape.w},
          in_strides{1, shape.h * shape.w * shape.g, 1, shape.w * shape.g, shape.g},
          wei_lengths{shape.g, 1, 1, 3, 3},
          wei_strides{9, 9, 1, 3, 1},
          out_lengths{shape.g, shape.n, 1, shape.OutH(), shape.OutW()},
          out_strides{1, shape.OutH() * shape.OutW() * shape.g, 1, shape.OutW() * shape.g, shape.g}
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
    const std::size_t weight_count = static_cast<std::size_t>(shape.g) * 9;
    const std::size_t partial_count =
        static_cast<std::size_t>(shape.n) * shape.Strips() * weight_count;
    Reference ref{std::vector<double>(weight_count),
                  std::vector<double>(partial_count),
                  std::vector<double>(weight_count),
                  std::vector<double>(partial_count)};
    for(int g = 0; g < shape.g; ++g)
    {
        for(int fy = 0; fy < 3; ++fy)
        {
            for(int fx = 0; fx < 3; ++fx)
            {
                const auto f = static_cast<std::size_t>(g) * 9 + fy * 3 + fx;
                for(int n = 0; n < shape.n; ++n)
                {
                    for(int ho = 0; ho < shape.OutH(); ++ho)
                    {
                        const int hi = 2 * ho + fy - 1;
                        if(hi < 0 || hi >= shape.h)
                            continue;
                        for(int wo = 0; wo < shape.OutW(); ++wo)
                        {
                            const int wi = 2 * wo + fx - 1;
                            if(wi < 0 || wi >= shape.w)
                                continue;
                            const double x = ck::type_convert<float>(
                                x_nchw[NchwOffset(n, g, hi, wi, shape.g, shape.h, shape.w)]);
                            const double dy      = ck::type_convert<float>(dy_nchw[NchwOffset(
                                n, g, ho, wo, shape.g, shape.OutH(), shape.OutW())]);
                            const double product = x * dy;
                            const auto p = (static_cast<std::size_t>(n) * shape.Strips() + ho / 8) *
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

std::size_t WorkspaceBytes(Shape shape)
{
    const auto logical =
        static_cast<std::size_t>(shape.n) * shape.Strips() * shape.g * 9 * sizeof(float);
    return (logical + 255) / 256 * 256;
}

void CheckShape(Shape shape)
{
    RowStripOp concrete;
    BaseOp& op = concrete;
    const Problem<ck::index_t> problem(shape);
    EXPECT_EQ(op.GetTypeString().find("DeviceGroupedConvBwdWeightDepthwiseGroupedRowStripBf16<"),
              0u);

    const auto x_count  = static_cast<std::size_t>(shape.n) * shape.g * shape.h * shape.w;
    const auto dy_count = static_cast<std::size_t>(shape.n) * shape.g * shape.OutH() * shape.OutW();
    const auto weight_count = static_cast<std::size_t>(shape.g) * 9;
    std::vector<ck::bhalf_t> x_nchw(x_count), dy_nchw(dy_count), x_packed(x_count),
        dy_packed(dy_count), actual(weight_count);
    ck::DeviceMem x_device(x_count * sizeof(ck::bhalf_t));
    ck::DeviceMem dy_device(dy_count * sizeof(ck::bhalf_t));
    ck::DeviceMem dw_device(weight_count * sizeof(ck::bhalf_t));
    const auto workspace_bytes = WorkspaceBytes(shape);
    ck::DeviceMem workspace(workspace_bytes);
    std::vector<std::uint8_t> poison_workspace(workspace_bytes, 0xff);
    std::vector<std::uint8_t> poison_dw(weight_count * sizeof(ck::bhalf_t), 0xff);
    std::vector<float> observed_partials(workspace_bytes / sizeof(float));
    auto invoker = op.MakeInvokerPointer();
    std::vector<double> old_weights;

    for(int repetition = 0; repetition < 2; ++repetition)
    {
        FillInputs(shape, repetition, x_nchw, dy_nchw, x_packed, dy_packed);
        const auto reference = ComputeReference(shape, x_nchw, dy_nchw);
        ASSERT_EQ(reference.partials.size(),
                  static_cast<std::size_t>(shape.n) * shape.Strips() * weight_count);
        const auto tail_center =
            (static_cast<std::size_t>(shape.n) * shape.Strips() - 1) * weight_count + 192 * 9 + 4;
        ASSERT_GT(reference.partial_magnitudes[tail_center], 0);
        EXPECT_GT(reference.weight_magnitudes[192 * 9 + 4], 0);
        if(repetition == 1)
            EXPECT_NE(reference.weights, old_weights) << "the second call must change input";
        old_weights = reference.weights;
        x_device.ToDevice(x_packed.data());
        dy_device.ToDevice(dy_packed.data());

        for(const ck::index_t split : {-1, 0, 1})
        {
            auto arg = problem.MakeArgument(op,
                                            x_device.GetDeviceBuffer(),
                                            dw_device.GetDeviceBuffer(),
                                            dy_device.GetDeviceBuffer(),
                                            split);
            ASSERT_TRUE(op.IsSupportedArgument(arg.get())) << "split=" << split;
            ASSERT_EQ(op.GetWorkSpaceSize(arg.get()), workspace_bytes);
            EXPECT_THROW(invoker->Run(arg.get(), StreamConfig{nullptr, false}), std::runtime_error);
            op.SetWorkSpacePointer(arg.get(), workspace.GetDeviceBuffer());
            dw_device.ToDevice(poison_dw.data());
            workspace.ToDevice(poison_workspace.data());
            invoker->Run(arg.get(), StreamConfig{nullptr, false});
            dw_device.FromDevice(actual.data());
            workspace.FromDevice(observed_partials.data());

            for(std::size_t i = 0; i < weight_count; ++i)
            {
                const double got       = ck::type_convert<float>(actual[i]);
                const double want      = reference.weights[i];
                const double tolerance = std::max(
                    0.03125, 0.008 * std::abs(want) + 0.0002 * reference.weight_magnitudes[i]);
                EXPECT_TRUE(std::isfinite(got)) << "weight=" << i << " split=" << split;
                EXPECT_NEAR(got, want, tolerance)
                    << "g=" << i / 9 << " fy=" << (i % 9) / 3 << " fx=" << i % 3
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
                    << "s=" << i / weight_count << " g=" << (i % weight_count) / 9 << " f=" << i % 9
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

    RowStripOp concrete;
    BaseOp& op = concrete;
    const Shape shape{2, 193, 13, 131};
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

    for(int groups : {191, 257})
    {
        const Problem<ck::index_t> out_of_range(Shape{2, groups, 13, 131});
        EXPECT_FALSE(op.IsSupportedArgument(
            out_of_range.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    }
    for(int groups : {192, 256})
    {
        const Shape boundary{2, groups, 13, 131};
        const Problem<ck::index_t> supported(boundary);
        auto arg = supported.MakeArgument(op, nullptr, nullptr, nullptr, 1);
        ASSERT_TRUE(op.IsSupportedArgument(arg.get())) << "groups=" << groups;
        EXPECT_EQ(op.GetWorkSpaceSize(arg.get()), WorkspaceBytes(boundary));
    }
    const Problem<ck::index_t> too_short(Shape{1, 193, 1, 159});
    EXPECT_FALSE(
        op.IsSupportedArgument(too_short.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    const Shape target{42, 192, 120, 160};
    const Problem<ck::index_t> target_problem(target);
    auto target_arg = target_problem.MakeArgument(op, nullptr, nullptr, nullptr, 1);
    ASSERT_TRUE(op.IsSupportedArgument(target_arg.get()));
    EXPECT_EQ(op.GetWorkSpaceSize(target_arg.get()), 2322432u);
    auto adjusted_target       = target_problem;
    adjusted_target.right_pads = {0, 0};
    auto adjusted_arg          = adjusted_target.MakeArgument(op, nullptr, nullptr, nullptr, 1);
    ASSERT_TRUE(op.IsSupportedArgument(adjusted_arg.get()));
    EXPECT_EQ(op.GetWorkSpaceSize(adjusted_arg.get()), 2322432u);
}

TEST(TestGroupedConvndBwdWeightGroupedRowStripBf16, NchwSerialReferenceAndWorkspaceOwnership)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "gfx1250-only candidate";

    CheckShape({2, 193, 13, 131}); // one strip, odd width and the ninth group-lane tail
    CheckShape({5, 193, 25, 17});  // two strips per image, short width and row tails
    CheckShape({7, 193, 1, 159});  // both outer filter rows are empty for every split
}

} // namespace
