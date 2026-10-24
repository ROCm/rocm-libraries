// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_nchw_pointwise_instance.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"

namespace {

using Op          = ck::tensor_operation::device::DeviceGroupedConvBwdDataNCHWPointwise<>;
using Lengths     = Op::Lengths;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

void CheckSignedDirtyOutput(
    ck::index_t n, ck::index_t c, ck::index_t k, ck::index_t h, ck::index_t w, ck::index_t padding)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "Native NCHW pointwise backward data is gfx1250-specific";
    const auto spatial        = h * w;
    const auto channel_stride = spatial + padding;
    const Lengths a{1, n, k, h, w}, as{1, k * channel_stride + padding, channel_stride, w, 1};
    const Lengths b{1, k, c, 1, 1}, bs{1, c + padding, 1, 1, 1};
    const Lengths e{1, n, c, h, w}, es{1, c * channel_stride + padding, channel_stride, w, 1};
    auto descriptor = [](const Lengths& lengths, const Lengths& strides, auto layout) {
        return ck::HostTensorDescriptor(std::vector<std::size_t>(lengths.begin(), lengths.end()),
                                        std::vector<std::size_t>(strides.begin(), strides.end()),
                                        layout);
    };
    ck::Tensor<float> dy(descriptor(a, as, ck::tensor_layout::convolution::NGKHW{}));
    ck::Tensor<float> weights(descriptor(b, bs, ck::tensor_layout::convolution::GKCYX{}));
    const auto dx_descriptor = descriptor(e, es, ck::tensor_layout::convolution::NGCHW{});
    ck::Tensor<float> expected(dx_descriptor), actual(dx_descriptor);
    dy.ForEach([&](auto& tensor, const auto& index) {
        const auto value = (index[1] * 17 + index[2] * 3 + index[3] * 5 + index[4] * 7) % 97;
        tensor(index)    = static_cast<float>(static_cast<int>(value) - 48) / 31.0f;
    });
    weights.ForEach([&](auto& tensor, const auto& index) {
        const auto value = (index[1] * 13 + index[2] * 11) % 71;
        tensor(index)    = static_cast<float>(static_cast<int>(value) - 35) / 63.0f;
    });
    using Reference = ck::tensor_operation::host::
        ReferenceConvBwdData<2, float, float, float, PassThrough, PassThrough, PassThrough>;
    auto ref_arg = Reference::MakeArgument(expected,
                                           weights,
                                           dy,
                                           {1, 1},
                                           {1, 1},
                                           {0, 0},
                                           {0, 0},
                                           PassThrough{},
                                           PassThrough{},
                                           PassThrough{});
    Reference::MakeInvoker().Run(ref_arg);
    ck::DeviceMem dy_device(dy.mData.size() * sizeof(float));
    ck::DeviceMem weights_device(weights.mData.size() * sizeof(float));
    ck::DeviceMem dx_device(actual.mData.size() * sizeof(float));
    dy_device.ToDevice(dy.mData.data());
    weights_device.ToDevice(weights.mData.data());
    std::vector<std::unique_ptr<ck::tensor_operation::device::instance::DeviceNCHWPointwiseBwdData>>
        instances;
    ck::tensor_operation::device::instance::
        add_device_grouped_conv2d_bwd_data_nchw_pointwise_f32_instances(instances);
    for(const auto& op : instances)
    {
        auto arg = op->MakeArgumentPointer(dy_device.GetDeviceBuffer(),
                                           weights_device.GetDeviceBuffer(),
                                           {},
                                           dx_device.GetDeviceBuffer(),
                                           a,
                                           as,
                                           b,
                                           bs,
                                           {},
                                           {},
                                           e,
                                           es,
                                           {1, 1},
                                           {1, 1},
                                           {0, 0},
                                           {0, 0},
                                           {},
                                           {},
                                           {},
                                           1);
        ASSERT_TRUE(op->IsSupportedArgument(arg.get())) << op->GetTypeString();
        ASSERT_EQ(op->GetWorkSpaceSize(arg.get()), 0u);
        for(const auto poison : {113.0f, std::numeric_limits<float>::quiet_NaN()})
        {
            std::fill(actual.mData.begin(), actual.mData.end(), poison);
            dx_device.ToDevice(actual.mData.data());
            op->MakeInvokerPointer()->Run(arg.get(), {});
            dx_device.FromDevice(actual.mData.data());
            for(ck::index_t ni = 0; ni < n; ++ni)
                for(ck::index_t ci = 0; ci < c; ++ci)
                {
                    for(ck::index_t hi = 0; hi < h; ++hi)
                        for(ck::index_t wi = 0; wi < w; ++wi)
                        {
                            const float want = expected(0, ni, ci, hi, wi);
                            const float got  = actual(0, ni, ci, hi, wi);
                            ASSERT_TRUE(std::isfinite(got));
                            // Bound the difference between two FP32 reductions by
                            // 2*gamma_k*sum(abs(products)); this also covers separate
                            // reference multiplies versus device FMA for k<=16.
                            double sum_abs = 0;
                            for(ck::index_t ki = 0; ki < k; ++ki)
                                sum_abs += std::abs(static_cast<double>(dy(0, ni, ki, hi, wi)) *
                                                    weights(0, ki, ci, 0, 0));
                            const double ku = k * std::numeric_limits<float>::epsilon();
                            ASSERT_NEAR(got, want, 2 * ku / (1 - ku) * sum_abs)
                                << op->GetTypeString() << " n=" << ni << " c=" << ci << " h=" << hi
                                << " w=" << wi;
                        }
                    // Inter-channel padding must not be overwritten by a masked vector store.
                    for(ck::index_t si = spatial; si < channel_stride; ++si)
                    {
                        const auto offset = ni * es[1] + ci * es[2] + si;
                        if(offset < static_cast<ck::long_index_t>(actual.mData.size()))
                        {
                            const float got = actual.mData[offset];
                            if(std::isnan(poison))
                                ASSERT_TRUE(std::isnan(got));
                            else
                                ASSERT_EQ(got, poison);
                        }
                    }
                }
        }
    }
}

TEST(NCHWPointwiseBackwardData, SignedOddChannelsSpatialAndReduction)
{
    CheckSignedDirtyOutput(2, 65, 7, 33, 35, 0);
}

TEST(NCHWPointwiseBackwardData, PaddedChannelsAndBatch)
{
    CheckSignedDirtyOutput(2, 65, 7, 33, 35, 1);
}

TEST(NCHWPointwiseBackwardData, VectorizedAndReductionBoundary)
{
    CheckSignedDirtyOutput(2, 128, 16, 32, 36, 0);
}

TEST(NCHWPointwiseBackwardData, RejectsUnsupportedConvolutionAndUnsafeDescriptors)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "Native NCHW pointwise backward data is gfx1250-specific";
    Op op;
    const Lengths a{1, 2, 3, 32, 36}, as{1, 3456, 1152, 36, 1};
    const Lengths b{1, 3, 65, 1, 1}, bs{1, 65, 1, 1, 1};
    const Lengths e{1, 2, 65, 32, 36}, es{1, 74880, 1152, 36, 1};
    auto make = [&](Lengths al,
                    Lengths ast,
                    Lengths bl,
                    Lengths el,
                    Lengths est,
                    Op::Filter strides,
                    Op::Filter dilation,
                    Op::Filter left,
                    int split) {
        return op.MakeArgumentPointer(nullptr,
                                      nullptr,
                                      {},
                                      nullptr,
                                      al,
                                      ast,
                                      bl,
                                      bs,
                                      {},
                                      {},
                                      el,
                                      est,
                                      strides,
                                      dilation,
                                      left,
                                      {0, 0},
                                      {},
                                      {},
                                      {},
                                      split);
    };
    auto supported = make(a, as, b, e, es, {1, 1}, {1, 1}, {0, 0}, 1);
    ASSERT_TRUE(op.IsSupportedArgument(supported.get()));
    auto split = make(a, as, b, e, es, {1, 1}, {1, 1}, {0, 0}, 2);
    EXPECT_FALSE(op.IsSupportedArgument(split.get()));
    EXPECT_THROW(op.MakeInvokerPointer()->Run(split.get(), {}), std::runtime_error);
    EXPECT_FALSE(op.IsSupportedArgument(make(a, as, b, e, es, {2, 1}, {1, 1}, {0, 0}, 1).get()));
    EXPECT_FALSE(op.IsSupportedArgument(make(a, as, b, e, es, {1, 1}, {2, 1}, {0, 0}, 1).get()));
    EXPECT_FALSE(op.IsSupportedArgument(make(a, as, b, e, es, {1, 1}, {1, 1}, {1, 0}, 1).get()));
    auto overlapping = es;
    overlapping[2]   = 1;
    EXPECT_FALSE(
        op.IsSupportedArgument(make(a, as, b, e, overlapping, {1, 1}, {1, 1}, {0, 0}, 1).get()));
    auto byte_overflow = es;
    byte_overflow[1]   = ck::long_index_t{1} << 29;
    EXPECT_FALSE(
        op.IsSupportedArgument(make(a, as, b, e, byte_overflow, {1, 1}, {1, 1}, {0, 0}, 1).get()));
    auto excessive_reduction_a = a;
    auto excessive_reduction_b = b;
    excessive_reduction_a[2] = excessive_reduction_b[1] = 17;
    EXPECT_FALSE(op.IsSupportedArgument(
        make(excessive_reduction_a, as, excessive_reduction_b, e, es, {1, 1}, {1, 1}, {0, 0}, 1)
            .get()));
}

TEST(NCHWPointwiseBackwardData, RejectsAliasedOutputBeforeLaunch)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "Native NCHW pointwise backward data is gfx1250-specific";
    Op op;
    const Lengths a{1, 2, 3, 32, 36}, as{1, 3456, 1152, 36, 1};
    const Lengths b{1, 3, 65, 1, 1}, bs{1, 65, 1, 1, 1};
    const Lengths e{1, 2, 65, 32, 36}, es{1, 74880, 1152, 36, 1};
    const auto output_bytes = 2 * es[1] * sizeof(float);
    const auto input_bytes  = 2 * as[1] * sizeof(float);
    ck::DeviceMem buffer(output_bytes + input_bytes);
    ck::DeviceMem weights(3 * 65 * sizeof(float));
    auto* output = static_cast<char*>(buffer.GetDeviceBuffer());
    auto make    = [&](const void* dy, const void* w) {
        return op.MakeArgumentPointer(dy,
                                      w,
                                         {},
                                      output,
                                      a,
                                      as,
                                      b,
                                      bs,
                                         {},
                                         {},
                                      e,
                                      es,
                                         {1, 1},
                                         {1, 1},
                                         {0, 0},
                                         {0, 0},
                                         {},
                                         {},
                                         {},
                                      1);
    };
    auto adjacent = make(output + output_bytes, weights.GetDeviceBuffer());
    ASSERT_TRUE(op.IsSupportedArgument(adjacent.get()));
    for(const auto offset : {0, 128})
    {
        auto aliased = make(output + offset, weights.GetDeviceBuffer());
        EXPECT_FALSE(op.IsSupportedArgument(aliased.get()));
        EXPECT_THROW(op.MakeInvokerPointer()->Run(aliased.get(), {}), std::runtime_error);
    }
    auto aliased_weight = make(output + output_bytes, output + 64);
    EXPECT_FALSE(op.IsSupportedArgument(aliased_weight.get()));
    EXPECT_THROW(op.MakeInvokerPointer()->Run(aliased_weight.get(), {}), std::runtime_error);
}

TEST(NCHWPointwiseBackwardData, RejectsMisalignedScalarPointers)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "Native NCHW pointwise backward data is gfx1250-specific";
    Op op;
    const Lengths a{1, 2, 3, 32, 36}, as{1, 3456, 1152, 36, 1};
    const Lengths b{1, 3, 65, 1, 1}, bs{1, 65, 1, 1, 1};
    const Lengths e{1, 2, 65, 32, 36}, es{1, 74880, 1152, 36, 1};
    ck::DeviceMem dy(2 * as[1] * sizeof(float) + 16);
    ck::DeviceMem weights(3 * 65 * sizeof(float) + 16);
    ck::DeviceMem dx(2 * es[1] * sizeof(float) + 16);
    const auto check = [&](const void* dy_ptr, const void* w_ptr, void* dx_ptr) {
        auto arg = op.MakeArgumentPointer(dy_ptr,
                                          w_ptr,
                                          {},
                                          dx_ptr,
                                          a,
                                          as,
                                          b,
                                          bs,
                                          {},
                                          {},
                                          e,
                                          es,
                                          {1, 1},
                                          {1, 1},
                                          {0, 0},
                                          {0, 0},
                                          {},
                                          {},
                                          {},
                                          1);
        ASSERT_FALSE(op.IsSupportedArgument(arg.get()));
        EXPECT_THROW(op.MakeInvokerPointer()->Run(arg.get(), {}), std::runtime_error);
    };
    check(static_cast<char*>(dy.GetDeviceBuffer()) + 1,
          weights.GetDeviceBuffer(),
          dx.GetDeviceBuffer());
    check(dy.GetDeviceBuffer(),
          static_cast<char*>(weights.GetDeviceBuffer()) + 2,
          dx.GetDeviceBuffer());
    check(dy.GetDeviceBuffer(),
          weights.GetDeviceBuffer(),
          static_cast<char*>(dx.GetDeviceBuffer()) + 1);
}

} // namespace
