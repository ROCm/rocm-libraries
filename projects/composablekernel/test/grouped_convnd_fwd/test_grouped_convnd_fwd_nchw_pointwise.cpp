// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_nchw_pointwise.hpp"

namespace {

using Op      = ck::tensor_operation::device::DeviceGroupedConvFwdNCHWPointwise;
using Lengths = Op::Lengths;
using Half    = ck::half_t;

void CheckSignedForward(int n, int c, int k, int h, int w)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "Native NCHW pointwise forward is gfx1250-specific";
    const auto spatial   = h * w;
    const auto in_count  = static_cast<std::size_t>(n) * c * spatial;
    const auto out_count = static_cast<std::size_t>(n) * k * spatial;
    std::vector<Half> input(in_count), weights(c * k), actual(out_count);
    for(std::size_t i = 0; i < input.size(); ++i)
        input[i] = ck::type_convert<Half>((static_cast<int>((i * 17 + 3) % 97) - 48) / 32.0f);
    for(std::size_t i = 0; i < weights.size(); ++i)
        weights[i] = ck::type_convert<Half>((static_cast<int>((i * 13 + 5) % 71) - 35) / 64.0f);
    ck::DeviceMem in_device(in_count * sizeof(Half));
    ck::DeviceMem weight_device(weights.size() * sizeof(Half));
    ck::DeviceMem out_device(out_count * sizeof(Half));
    in_device.ToDevice(input.data());
    weight_device.ToDevice(weights.data());
    Op op;
    const Lengths a{1, n, c, h, w}, as{1, c * spatial, spatial, w, 1};
    const Lengths b{1, k, c, 1, 1}, bs{1, c, 1, 1, 1};
    const Lengths e{1, n, k, h, w}, es{1, k * spatial, spatial, w, 1};
    auto arg = op.MakeArgumentPointer(in_device.GetDeviceBuffer(),
                                      weight_device.GetDeviceBuffer(),
                                      {},
                                      out_device.GetDeviceBuffer(),
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
                                      {});
    ASSERT_TRUE(op.IsSupportedArgument(arg.get()));
    ASSERT_EQ(op.GetWorkSpaceSize(arg.get()), 0u);
    for(const auto poison : {113.0f, std::numeric_limits<float>::quiet_NaN()})
    {
        std::fill(actual.begin(), actual.end(), ck::type_convert<Half>(poison));
        out_device.ToDevice(actual.data());
        op.MakeInvokerPointer()->Run(arg.get(), {});
        out_device.FromDevice(actual.data());
        for(int ni = 0; ni < n; ++ni)
            for(int ki = 0; ki < k; ++ki)
                for(int s = 0; s < spatial; ++s)
                {
                    double expected = 0;
                    for(int ci = 0; ci < c; ++ci)
                        expected +=
                            static_cast<double>(ck::type_convert<float>(
                                input[(static_cast<std::size_t>(ni) * c + ci) * spatial + s])) *
                            ck::type_convert<float>(weights[ki * c + ci]);
                    // Binary fractions make all products and their <=64-term sum exactly
                    // representable in FP32. Compare the final FP16 rounding exactly.
                    const auto rounded = ck::type_convert<Half>(static_cast<float>(expected));
                    const float got    = ck::type_convert<float>(
                        actual[(static_cast<std::size_t>(ni) * k + ki) * spatial + s]);
                    ASSERT_TRUE(std::isfinite(got));
                    ASSERT_EQ(got, ck::type_convert<float>(rounded))
                        << "n=" << ni << " k=" << ki << " spatial=" << s;
                }
    }
}

TEST(NCHWPointwiseForward, SignedBroadcastWeightsAndDirtyOutputs)
{
    CheckSignedForward(2, 32, 128, 64, 64);
}

TEST(NCHWPointwiseForward, OddHeightSpatialBlockTailAndOutputChannelTail)
{
    CheckSignedForward(2, 64, 225, 65, 72);
}

TEST(NCHWPointwiseForward, PartialReductionAndOddWidth)
{
    // C=40 pads the 32-term GEMM reduction, independently of spatial/output tails.
    CheckSignedForward(2, 40, 129, 64, 65);
}

TEST(NCHWPointwiseForward, RejectsUnsafeLayoutsAndAliasesBeforeLaunch)
{
    if(!ck::is_gfx125_supported())
        GTEST_SKIP() << "Native NCHW pointwise forward is gfx1250-specific";
    Op op;
    const Lengths a{1, 2, 32, 64, 64}, as{1, 131072, 4096, 64, 1};
    const Lengths b{1, 128, 32, 1, 1}, bs{1, 32, 1, 1, 1};
    const Lengths e{1, 2, 128, 64, 64}, es{1, 524288, 4096, 64, 1};
    auto make = [&](const void* input,
                    const void* weights,
                    void* output,
                    Lengths al,
                    Lengths ast,
                    Lengths el,
                    Lengths est,
                    Op::Filter strides,
                    Op::Filter left) {
        return op.MakeArgumentPointer(input,
                                      weights,
                                      {},
                                      output,
                                      al,
                                      ast,
                                      b,
                                      bs,
                                      {},
                                      {},
                                      el,
                                      est,
                                      strides,
                                      {1, 1},
                                      left,
                                      {0, 0},
                                      {},
                                      {},
                                      {});
    };
    auto dry = make(nullptr, nullptr, nullptr, a, as, e, es, {1, 1}, {0, 0});
    ASSERT_TRUE(op.IsSupportedArgument(dry.get()));
    EXPECT_FALSE(op.IsSupportedArgument(
        make(nullptr, nullptr, nullptr, a, as, e, es, {2, 1}, {0, 0}).get()));
    EXPECT_FALSE(op.IsSupportedArgument(
        make(nullptr, nullptr, nullptr, a, as, e, es, {1, 1}, {1, 0}).get()));
    auto overlapping = es;
    overlapping[2]   = 1;
    EXPECT_FALSE(op.IsSupportedArgument(
        make(nullptr, nullptr, nullptr, a, as, e, overlapping, {1, 1}, {0, 0}).get()));
    auto too_large = es;
    too_large[1]   = ck::long_index_t{1} << 30;
    EXPECT_FALSE(op.IsSupportedArgument(
        make(nullptr, nullptr, nullptr, a, as, e, too_large, {1, 1}, {0, 0}).get()));
    const auto output_bytes = 2 * es[1] * sizeof(Half);
    ck::DeviceMem buffers(output_bytes + 2 * as[1] * sizeof(Half));
    ck::DeviceMem weights(128 * 32 * sizeof(Half));
    auto* out = static_cast<char*>(buffers.GetDeviceBuffer());
    for(const auto offset : {0, 128})
    {
        auto alias =
            make(out + offset, weights.GetDeviceBuffer(), out, a, as, e, es, {1, 1}, {0, 0});
        EXPECT_FALSE(op.IsSupportedArgument(alias.get()));
        EXPECT_THROW(op.MakeInvokerPointer()->Run(alias.get(), {}), std::runtime_error);
    }
    auto weight_alias = make(out + output_bytes, out, out, a, as, e, es, {1, 1}, {0, 0});
    EXPECT_FALSE(op.IsSupportedArgument(weight_alias.get()));
    auto misaligned =
        make(out + output_bytes, weights.GetDeviceBuffer(), out + 2, a, as, e, es, {1, 1}, {0, 0});
    EXPECT_FALSE(op.IsSupportedArgument(misaligned.get()));
}

} // namespace
