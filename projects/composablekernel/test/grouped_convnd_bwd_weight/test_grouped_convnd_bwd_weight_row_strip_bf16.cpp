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
using RowStripOp = ck::tensor_operation::device::DeviceGroupedConvBwdWeightDepthwiseRowStripBf16;

struct Shape
{
    int n;
    int g;
    int h;
    int w;
};

template <typename Index>
struct Problem
{
    std::array<Index, 5> in_lengths, in_strides, wei_lengths, wei_strides, out_lengths, out_strides;
    std::array<Index, 2> filter_strides{1, 1}, filter_dilations{1, 1}, left_pads{5, 5},
        right_pads{5, 5};

    explicit Problem(Shape shape)
        : in_lengths{shape.g, shape.n, 1, shape.h, shape.w},
          in_strides{1, shape.h * shape.w * shape.g, 1, shape.w * shape.g, shape.g},
          wei_lengths{shape.g, 1, 1, 11, 11},
          wei_strides{121, 121, 1, 11, 1},
          out_lengths{shape.g, shape.n, 1, shape.h, shape.w},
          out_strides(in_strides)
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

std::size_t NchwOffset(Shape shape, int n, int g, int h, int w)
{
    return ((static_cast<std::size_t>(n) * shape.g + g) * shape.h + h) * shape.w + w;
}

std::size_t NhwgcOffset(Shape shape, int n, int g, int h, int w)
{
    return ((static_cast<std::size_t>(n) * shape.h + h) * shape.w + w) * shape.g + g;
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
    const int strips               = (shape.h + 11) / 12;
    const std::size_t filter_count = static_cast<std::size_t>(shape.g) * 121;
    Reference ref{std::vector<double>(filter_count),
                  std::vector<double>(static_cast<std::size_t>(shape.n) * strips * filter_count),
                  std::vector<double>(static_cast<std::size_t>(shape.n) * strips * filter_count),
                  std::vector<double>(filter_count)};
    for(int g = 0; g < shape.g; ++g)
    {
        for(int fy = 0; fy < 11; ++fy)
        {
            for(int fx = 0; fx < 11; ++fx)
            {
                const std::size_t f = static_cast<std::size_t>(g) * 121 + fy * 11 + fx;
                for(int n = 0; n < shape.n; ++n)
                {
                    for(int ho = 0; ho < shape.h; ++ho)
                    {
                        const int hi = ho + fy - 5;
                        if(hi < 0 || hi >= shape.h)
                            continue;
                        for(int wo = 0; wo < shape.w; ++wo)
                        {
                            const int wi = wo + fx - 5;
                            if(wi < 0 || wi >= shape.w)
                                continue;
                            const double x =
                                ck::type_convert<float>(x_nchw[NchwOffset(shape, n, g, hi, wi)]);
                            const double dy =
                                ck::type_convert<float>(dy_nchw[NchwOffset(shape, n, g, ho, wo)]);
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
                    const float x_sign  = (n * 7 + g * 3 + h * 5 + w) % 11 < 5 ? -1.f : 1.f;
                    const float dy_sign = (n * 3 + g * 5 + h + w * 7) % 13 < 6 ? -1.f : 1.f;
                    const float x_value = repetition == 0
                                              ? 0.5f * (1 + (n + 2 * g + h + w) % 5)
                                              : large_x[(n + 2 * g + 3 * h + w) % large_x.size()];
                    const float dy_value =
                        repetition == 0 ? 0.25f * (1 + (3 * n + g + 2 * h + w) % 7)
                                        : large_dy[(2 * n + g + h + 3 * w) % large_dy.size()];
                    const auto nchw   = NchwOffset(shape, n, g, h, w);
                    const auto packed = NhwgcOffset(shape, n, g, h, w);
                    x_nchw[nchw]      = x_packed[packed] =
                        ck::type_convert<ck::bhalf_t>(x_sign * x_value);
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
        static_cast<std::size_t>(shape.n) * ((shape.h + 11) / 12) * shape.g * 121 * sizeof(float);
    return (logical + 255) / 256 * 256;
}

void CheckShape(Shape shape)
{
    RowStripOp concrete;
    BaseOp& op = concrete; // Exercise the same virtual interface used by the factory.
    const Problem<ck::index_t> problem(shape);
    EXPECT_EQ(op.GetTypeString().find("DeviceGroupedConvBwdWeightDepthwiseRowStripBf16<"), 0u);

    const std::size_t input_count = static_cast<std::size_t>(shape.n) * shape.g * shape.h * shape.w;
    const std::size_t filter_count = static_cast<std::size_t>(shape.g) * 121;
    std::vector<ck::bhalf_t> x_nchw(input_count), dy_nchw(input_count), x_packed(input_count),
        dy_packed(input_count), actual(filter_count);
    ck::DeviceMem x_device(input_count * sizeof(ck::bhalf_t));
    ck::DeviceMem dy_device(input_count * sizeof(ck::bhalf_t));
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
                << "group=" << i / 121 << " fy=" << (i % 121) / 11 << " fx=" << (i % 121) % 11
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
                << "strip=" << i / filter_count << " group=" << (i % filter_count) / 121
                << " filter=" << i % 121 << " repetition=" << repetition;
            if(reference.partial_magnitudes[i] <= 0)
                EXPECT_EQ(got, 0) << "empty partial=" << i;
        }
    }
}

TEST(TestGroupedConvndBwdWeightRowStripBf16, DryQueriesAndAdmission)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "gfx1250-only candidate";

    RowStripOp concrete;
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
    auto wrong_pad         = problem;
    wrong_pad.left_pads[0] = 4;
    EXPECT_FALSE(
        op.IsSupportedArgument(wrong_pad.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_dilation                = problem;
    wrong_dilation.filter_dilations[1] = 2;
    EXPECT_FALSE(op.IsSupportedArgument(
        wrong_dilation.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    const Problem<ck::index_t> non_target_groups(Shape{2, 2, 13, 131});
    EXPECT_FALSE(op.IsSupportedArgument(
        non_target_groups.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto huge           = long_problem;
    huge.out_lengths[3] = std::numeric_limits<ck::long_index_t>::max();
    EXPECT_FALSE(op.IsSupportedArgument(huge.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
}

TEST(TestGroupedConvndBwdWeightRowStripBf16, NchwSerialReferenceAndWorkspaceOwnership)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "gfx1250-only candidate";

    CheckShape({2, 3, 13, 131}); // row, column, batch and filter tails
    CheckShape({1, 3, 25, 17});  // three strips; two are not a hardcoded partition count
    CheckShape({1, 3, 1, 1});    // padded taps and entire empty filter-row/column partials
}

} // namespace
