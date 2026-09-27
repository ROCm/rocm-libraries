// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_nchw_pointwise_bf16_wmma.hpp"
#include "ck/utility/type_convert.hpp"

namespace {

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using NativeOp    = ck::tensor_operation::device::DeviceGroupedConvBwdWeightNchwPointwiseBf16Wmma;
using BaseOp =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeight<2,
                                                             ck::tensor_layout::convolution::NGCHW,
                                                             ck::tensor_layout::convolution::GKCYX,
                                                             ck::tensor_layout::convolution::NGKHW,
                                                             ck::bhalf_t,
                                                             ck::bhalf_t,
                                                             ck::bhalf_t,
                                                             PassThrough,
                                                             PassThrough,
                                                             PassThrough>;

constexpr int Splits                = 8;
constexpr std::size_t GuardBytes    = 256;
constexpr std::uint8_t GuardPattern = 0xa5;

struct Shape
{
    int n;
    int c;
    int k;
    int h;
    int w;

    int Spatial() const { return h * w; }
    std::size_t Weights() const { return static_cast<std::size_t>(k) * c; }
    std::size_t Partials() const { return static_cast<std::size_t>(Splits) * n * Weights(); }
    std::size_t WorkspaceBytes() const
    {
        const auto bytes = Partials() * sizeof(float);
        return (bytes + 255) / 256 * 256;
    }
};

template <typename Index>
struct Problem
{
    std::array<Index, 5> in_lengths, in_strides, wei_lengths, wei_strides, out_lengths, out_strides;
    std::array<Index, 2> filter_strides{1, 1}, filter_dilations{1, 1}, left_pads{0, 0},
        right_pads{0, 0};

    explicit Problem(Shape shape)
        : in_lengths{1, shape.n, shape.c, shape.h, shape.w},
          in_strides{static_cast<Index>(shape.n) * shape.c * shape.Spatial(),
                     static_cast<Index>(shape.c) * shape.Spatial(),
                     static_cast<Index>(shape.Spatial()),
                     static_cast<Index>(shape.w),
                     1},
          wei_lengths{1, shape.k, shape.c, 1, 1},
          wei_strides{static_cast<Index>(shape.Weights()), shape.c, 1, 1, 1},
          out_lengths{1, shape.n, shape.k, shape.h, shape.w},
          out_strides{static_cast<Index>(shape.n) * shape.k * shape.Spatial(),
                      static_cast<Index>(shape.k) * shape.Spatial(),
                      static_cast<Index>(shape.Spatial()),
                      static_cast<Index>(shape.w),
                      1}
    {
    }

    std::unique_ptr<ck::tensor_operation::device::BaseArgument>
    MakeArgument(BaseOp& op, const void* x, void* dw, const void* dy, ck::index_t split) const
    {
        return op.MakeArgumentPointer(x,
                                      dw,
                                      dy,
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

std::size_t XOffset(Shape shape, int n, int c, int spatial)
{
    return (static_cast<std::size_t>(n) * shape.c + c) * shape.Spatial() + spatial;
}

std::size_t DyOffset(Shape shape, int n, int k, int spatial)
{
    return (static_cast<std::size_t>(n) * shape.k + k) * shape.Spatial() + spatial;
}

std::size_t POffset(Shape shape, int split, int n, int k, int c)
{
    return ((static_cast<std::size_t>(split) * shape.n + n) * shape.k + k) * shape.c + c;
}

void FillInputs(Shape shape,
                int repetition,
                std::vector<ck::bhalf_t>& x,
                std::vector<ck::bhalf_t>& dy)
{
    for(int n = 0; n < shape.n; ++n)
    {
        for(int c = 0; c < shape.c; ++c)
        {
            for(int h = 0; h < shape.h; ++h)
            {
                for(int w = 0; w < shape.w; ++w)
                {
                    const unsigned s = static_cast<unsigned>(h * shape.w + w);
                    const unsigned key =
                        s * 17u + (s / 127u) * 3u + c * 13u + n * 7u + 5u + repetition * 9u;
                    x[XOffset(shape, n, c, static_cast<int>(s))] =
                        ck::type_convert<ck::bhalf_t>((static_cast<int>(key % 31u) - 15) / 32.f);
                }
            }
        }
        for(int k = 0; k < shape.k; ++k)
        {
            for(int h = 0; h < shape.h; ++h)
            {
                for(int w = 0; w < shape.w; ++w)
                {
                    const unsigned s = static_cast<unsigned>(h * shape.w + w);
                    const unsigned key =
                        s * 11u + (s / 79u) * 17u + k * 7u + n * 13u + 3u + repetition * 5u;
                    dy[DyOffset(shape, n, k, static_cast<int>(s))] =
                        ck::type_convert<ck::bhalf_t>((static_cast<int>(key % 29u) - 14) / 32.f);
                }
            }
        }
    }
}

struct Reference
{
    std::vector<double> partials;
    std::vector<double> weights;
};

// Serial double-precision NCHW pointwise convolution, independent of the GPU's
// row/column tiles and split/image workspace addressing.
Reference
ComputeReference(Shape shape, const std::vector<ck::bhalf_t>& x, const std::vector<ck::bhalf_t>& dy)
{
    Reference reference{std::vector<double>(shape.Partials()),
                        std::vector<double>(shape.Weights())};
    for(int split = 0; split < Splits; ++split)
    {
        for(int n = 0; n < shape.n; ++n)
        {
            for(int k = 0; k < shape.k; ++k)
            {
                for(int c = 0; c < shape.c; ++c)
                {
                    double partial = 0;
                    for(int s = split * shape.Spatial() / Splits;
                        s < (split + 1) * shape.Spatial() / Splits;
                        ++s)
                    {
                        partial += static_cast<double>(
                                       ck::type_convert<float>(dy[DyOffset(shape, n, k, s)])) *
                                   static_cast<double>(
                                       ck::type_convert<float>(x[XOffset(shape, n, c, s)]));
                    }
                    reference.partials[POffset(shape, split, n, k, c)] = partial;
                    reference.weights[static_cast<std::size_t>(k) * shape.c + c] += partial;
                }
            }
        }
    }
    return reference;
}

void CheckGuards(const std::vector<std::uint8_t>& bytes, std::size_t payload, int repetition)
{
    ASSERT_EQ(bytes.size(), GuardBytes + payload + GuardBytes);
    EXPECT_TRUE(std::all_of(bytes.begin(),
                            bytes.begin() + GuardBytes,
                            [](std::uint8_t v) { return v == GuardPattern; }))
        << "leading guard, repetition=" << repetition;
    EXPECT_TRUE(std::all_of(bytes.begin() + GuardBytes + payload,
                            bytes.end(),
                            [](std::uint8_t v) { return v == GuardPattern; }))
        << "trailing guard, repetition=" << repetition;
}

void CheckShape(Shape shape)
{
    NativeOp concrete;
    BaseOp& op = concrete;
    const Problem<ck::index_t> problem(shape);
    const Problem<ck::long_index_t> long_problem(shape);
    ASSERT_EQ(
        op.GetTypeString().find("DeviceGroupedConvBwdWeightNchwPointwiseBf16Wmma<8,32,64,64>"), 0u);

    std::vector<ck::bhalf_t> x(static_cast<std::size_t>(shape.n) * shape.c * shape.Spatial());
    std::vector<ck::bhalf_t> dy(static_cast<std::size_t>(shape.n) * shape.k * shape.Spatial());
    ck::DeviceMem x_device(x.size() * sizeof(ck::bhalf_t));
    ck::DeviceMem dy_device(dy.size() * sizeof(ck::bhalf_t));
    const auto p_bytes  = shape.WorkspaceBytes();
    const auto dw_bytes = shape.Weights() * sizeof(ck::bhalf_t);
    ck::DeviceMem workspace(GuardBytes + p_bytes + GuardBytes);
    ck::DeviceMem dw_device(GuardBytes + dw_bytes + GuardBytes);
    auto* p_pointer  = static_cast<std::uint8_t*>(workspace.GetDeviceBuffer()) + GuardBytes;
    auto* dw_pointer = static_cast<std::uint8_t*>(dw_device.GetDeviceBuffer()) + GuardBytes;
    std::vector<std::uint8_t> observed_p(GuardBytes + p_bytes + GuardBytes);
    std::vector<std::uint8_t> observed_dw(GuardBytes + dw_bytes + GuardBytes);
    std::vector<std::uint8_t> alternative_p(observed_p.size());
    std::vector<std::uint8_t> alternative_dw(observed_dw.size());
    std::vector<std::uint8_t> poison_p(observed_p.size(), GuardPattern);
    std::vector<std::uint8_t> poison_dw(observed_dw.size(), GuardPattern);
    std::fill(poison_p.begin() + GuardBytes, poison_p.begin() + GuardBytes + p_bytes, 0xff);
    std::fill(poison_dw.begin() + GuardBytes, poison_dw.begin() + GuardBytes + dw_bytes, 0xff);
    auto invoker = op.MakeInvokerPointer();
    std::vector<double> previous_weights;

    auto aliased_inputs = problem.MakeArgument(
        op, x_device.GetDeviceBuffer(), dw_pointer, x_device.GetDeviceBuffer(), 1);
    EXPECT_FALSE(op.IsSupportedArgument(aliased_inputs.get()));
    auto aliased_workspace = problem.MakeArgument(
        op, x_device.GetDeviceBuffer(), dw_pointer, dy_device.GetDeviceBuffer(), 1);
    op.SetWorkSpacePointer(aliased_workspace.get(), dw_pointer);
    EXPECT_FALSE(op.IsSupportedArgument(aliased_workspace.get()));

    for(int repetition = 0; repetition < 2; ++repetition)
    {
        FillInputs(shape, repetition, x, dy);
        const auto reference = ComputeReference(shape, x, dy);
        ASSERT_NE(
            reference.partials[POffset(shape, Splits - 1, shape.n - 1, shape.k - 1, shape.c - 1)],
            0.0)
            << "the last split/image/row/channel must exercise a nonzero partial";
        if(repetition != 0)
            ASSERT_NE(reference.weights, previous_weights)
                << "the second invocation must change dW";
        previous_weights = reference.weights;
        x_device.ToDevice(x.data());
        dy_device.ToDevice(dy.data());

        auto arg = problem.MakeArgument(
            op, x_device.GetDeviceBuffer(), dw_pointer, dy_device.GetDeviceBuffer(), 1);
        ASSERT_TRUE(op.IsSupportedArgument(arg.get()));
        ASSERT_EQ(op.GetWorkSpaceSize(arg.get()), p_bytes);
        if(repetition == 0)
            EXPECT_THROW(invoker->Run(arg.get(), StreamConfig{nullptr, false}), std::runtime_error);
        op.SetWorkSpacePointer(arg.get(), p_pointer);
        workspace.ToDevice(poison_p.data());
        dw_device.ToDevice(poison_dw.data());
        invoker->Run(arg.get(), StreamConfig{nullptr, false});
        workspace.FromDevice(observed_p.data());
        dw_device.FromDevice(observed_dw.data());
        CheckGuards(observed_p, p_bytes, repetition);
        CheckGuards(observed_dw, dw_bytes, repetition);

        ASSERT_EQ(reference.partials.size(), shape.Partials());
        for(int split = 0; split < Splits; ++split)
        {
            for(int n = 0; n < shape.n; ++n)
            {
                for(int k = 0; k < shape.k; ++k)
                {
                    for(int c = 0; c < shape.c; ++c)
                    {
                        const auto offset = POffset(shape, split, n, k, c);
                        float got;
                        std::memcpy(&got,
                                    observed_p.data() + GuardBytes + offset * sizeof(float),
                                    sizeof(got));
                        // Both operands are multiples of 1/32: 128 products per
                        // split and every FP32 partial are exactly representable.
                        ASSERT_TRUE(std::isfinite(got))
                            << "split=" << split << " n=" << n << " k=" << k << " c=" << c
                            << " repetition=" << repetition;
                        ASSERT_EQ(static_cast<double>(got), reference.partials[offset])
                            << "split=" << split << " n=" << n << " k=" << k << " c=" << c
                            << " repetition=" << repetition;
                    }
                }
            }
        }
        for(std::size_t weight = 0; weight < shape.Weights(); ++weight)
        {
            ck::bhalf_t got;
            std::memcpy(&got, observed_dw.data() + GuardBytes + weight * sizeof(got), sizeof(got));
            const auto wanted =
                ck::type_convert<ck::bhalf_t>(static_cast<float>(reference.weights[weight]));
            ASSERT_EQ(std::memcmp(&got, &wanted, sizeof(got)), 0)
                << "k=" << weight / shape.c << " c=" << weight % shape.c
                << " repetition=" << repetition << " expected=" << ck::type_convert<float>(wanted)
                << " actual=" << ck::type_convert<float>(got);
        }
        // Public fixed-1 and automatic modes must reproduce the same eight
        // independent partials and BF16 reduction, not just report support.
        for(ck::index_t split : {-1, 0})
        {
            auto alternative = problem.MakeArgument(
                op, x_device.GetDeviceBuffer(), dw_pointer, dy_device.GetDeviceBuffer(), split);
            ASSERT_TRUE(op.IsSupportedArgument(alternative.get()));
            ASSERT_EQ(op.GetWorkSpaceSize(alternative.get()), p_bytes);
            op.SetWorkSpacePointer(alternative.get(), p_pointer);
            workspace.ToDevice(poison_p.data());
            dw_device.ToDevice(poison_dw.data());
            invoker->Run(alternative.get(), StreamConfig{nullptr, false});
            workspace.FromDevice(alternative_p.data());
            dw_device.FromDevice(alternative_dw.data());
            ASSERT_TRUE(alternative_p == observed_p)
                << "P split=" << split << " repetition=" << repetition;
            ASSERT_TRUE(alternative_dw == observed_dw)
                << "dW split=" << split << " repetition=" << repetition;
        }
        auto long_arg = long_problem.MakeArgument(
            op, x_device.GetDeviceBuffer(), dw_pointer, dy_device.GetDeviceBuffer(), 1);
        ASSERT_TRUE(op.IsSupportedArgument(long_arg.get()));
        ASSERT_EQ(op.GetWorkSpaceSize(long_arg.get()), p_bytes);
        op.SetWorkSpacePointer(long_arg.get(), p_pointer);
        workspace.ToDevice(poison_p.data());
        dw_device.ToDevice(poison_dw.data());
        invoker->Run(long_arg.get(), StreamConfig{nullptr, false});
        workspace.FromDevice(alternative_p.data());
        dw_device.FromDevice(alternative_dw.data());
        ASSERT_TRUE(alternative_p == observed_p) << "long-index P repetition=" << repetition;
        ASSERT_TRUE(alternative_dw == observed_dw) << "long-index dW repetition=" << repetition;
    }
}

TEST(TestGroupedConvndBwdWeightNchwPointwiseBf16, DryQueriesAndAdmission)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "gfx1250-only candidate";

    NativeOp concrete;
    BaseOp& op = concrete;
    const Shape shape{2, 24, 112, 32, 32};
    const Problem<ck::index_t> problem(shape);
    const Problem<ck::long_index_t> long_problem(shape);
    for(ck::index_t split : {-1, 0, 1})
    {
        auto dry = problem.MakeArgument(op, nullptr, nullptr, nullptr, split);
        ASSERT_TRUE(op.IsSupportedArgument(dry.get())) << "split=" << split;
        EXPECT_EQ(op.GetWorkSpaceSize(dry.get()), shape.WorkspaceBytes());
        auto long_dry = long_problem.MakeArgument(op, nullptr, nullptr, nullptr, split);
        ASSERT_TRUE(op.IsSupportedArgument(long_dry.get())) << "long split=" << split;
        EXPECT_EQ(op.GetWorkSpaceSize(long_dry.get()), shape.WorkspaceBytes());
    }
    for(ck::index_t split : {-2, 2, 8})
        EXPECT_FALSE(op.IsSupportedArgument(
            problem.MakeArgument(op, nullptr, nullptr, nullptr, split).get()))
            << "split=" << split;

    auto wrong_x_stride = problem;
    ++wrong_x_stride.in_strides[4];
    EXPECT_FALSE(op.IsSupportedArgument(
        wrong_x_stride.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_dy_stride = problem;
    ++wrong_dy_stride.out_strides[2];
    EXPECT_FALSE(op.IsSupportedArgument(
        wrong_dy_stride.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_dw_stride = problem;
    ++wrong_dw_stride.wei_strides[1];
    EXPECT_FALSE(op.IsSupportedArgument(
        wrong_dw_stride.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto channel_last       = problem;
    channel_last.in_strides = {static_cast<ck::index_t>(shape.n * shape.c * shape.Spatial()),
                               static_cast<ck::index_t>(shape.c * shape.Spatial()),
                               1,
                               static_cast<ck::index_t>(shape.w * shape.c),
                               shape.c};
    EXPECT_FALSE(
        op.IsSupportedArgument(channel_last.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_filter           = problem;
    wrong_filter.wei_lengths[3] = 3;
    EXPECT_FALSE(
        op.IsSupportedArgument(wrong_filter.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_filter_width           = problem;
    wrong_filter_width.wei_lengths[4] = 2;
    EXPECT_FALSE(op.IsSupportedArgument(
        wrong_filter_width.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_stride              = problem;
    wrong_stride.filter_strides[1] = 2;
    EXPECT_FALSE(
        op.IsSupportedArgument(wrong_stride.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_dilation                = problem;
    wrong_dilation.filter_dilations[0] = 2;
    EXPECT_FALSE(op.IsSupportedArgument(
        wrong_dilation.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_pad          = problem;
    wrong_pad.right_pads[1] = 1;
    EXPECT_FALSE(
        op.IsSupportedArgument(wrong_pad.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_left_pad         = problem;
    wrong_left_pad.left_pads[0] = 1;
    EXPECT_FALSE(op.IsSupportedArgument(
        wrong_left_pad.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_output = problem;
    ++wrong_output.out_lengths[4];
    EXPECT_FALSE(
        op.IsSupportedArgument(wrong_output.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto wrong_group          = problem;
    wrong_group.in_lengths[0] = wrong_group.wei_lengths[0] = wrong_group.out_lengths[0] = 2;
    EXPECT_FALSE(
        op.IsSupportedArgument(wrong_group.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));
    auto huge           = long_problem;
    huge.out_lengths[3] = std::numeric_limits<ck::long_index_t>::max();
    EXPECT_FALSE(op.IsSupportedArgument(huge.MakeArgument(op, nullptr, nullptr, nullptr, 1).get()));

    // Query only: the full target must not trigger an expensive CPU/GPU reference in this test.
    const Shape target{42, 24, 128, 240, 320};
    auto target_dry = Problem<ck::index_t>(target).MakeArgument(op, nullptr, nullptr, nullptr, 1);
    ASSERT_TRUE(op.IsSupportedArgument(target_dry.get()));
    EXPECT_EQ(op.GetWorkSpaceSize(target_dry.get()), 4128768u);
}

TEST(TestGroupedConvndBwdWeightNchwPointwiseBf16, SerialNchwPartialsAndBf16Reduction)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "gfx1250-only candidate";

    CheckShape({2, 24, 128, 32, 32}); // exact row tile, channel tile tail
    CheckShape({2, 24, 112, 16, 64}); // partial M tile, channel tail, different row stride
    CheckShape({2, 8, 16, 32, 32});   // minimum packed C and K at padded WMMA tile edges
}

} // namespace
