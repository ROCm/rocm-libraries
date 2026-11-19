// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)
#include "Fp64WrwReference.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

namespace
{
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using hip_kernel_provider::conv_validation::Fp64WrwReference;
using hip_kernel_provider::conv_validation::ReferenceGeometry;
using hipdnn_data_sdk::utilities::Tensor;

void normalized(const std::vector<int64_t>& values, int64_t* target, int64_t fill)
{
    std::fill(target, target + 5, fill);
    target[0] = values[0];
    target[1] = values[1];
    for(size_t i = 2; i < values.size(); ++i)
    {
        target[i + 5 - values.size()] = values[i];
    }
}
ReferenceGeometry geometry(const ConvGraphSpec& spec, bool flip)
{
    ReferenceGeometry a{};
    std::array<int64_t, 5> xd{};
    std::array<int64_t, 5> yd{};
    std::array<int64_t, 5> wd{};
    normalized(spec.x.dims, xd.data(), 1);
    normalized(spec.y.dims, yd.data(), 1);
    normalized(spec.w.dims, wd.data(), 1);
    normalized(spec.x.strides, a.xs, 0);
    normalized(spec.y.strides, a.ys, 0);
    a.n = xd[0];
    a.c = xd[1];
    a.k = yd[1];
    a.groups = a.c / wd[1];
    a.weights = 1;
    for(const int64_t dim : wd)
    {
        a.weights *= dim;
    }
    for(size_t i = 0; i < 3; ++i)
    {
        a.input[i] = xd[i + 2];
        a.output[i] = yd[i + 2];
        a.filter[i] = wd[i + 2];
        a.stride[i] = a.dilation[i] = 1;
    }
    for(size_t i = 0; i < spec.stride.size(); ++i)
    {
        const size_t axis = i + 3 - spec.stride.size();
        a.stride[axis] = spec.stride[i];
        a.dilation[axis] = spec.dilation[i];
        a.padding[axis] = spec.prePadding[i];
    }
    a.flip = flip ? 1 : 0;
    return a;
}
TEST(TestGpuWrwReference, GroupedStridedBordersAndBothModesMatchIndependentCpuDouble)
{
    SKIP_IF_NO_DEVICES();
    const Fp64WrwReference oracle;
    for(const int rank : {1, 2, 3})
    {
        for(const bool last : {false, true})
        {
            for(const bool flip : {false, true})
            {
                SCOPED_TRACE(::testing::Message()
                             << "rank=" << rank << " last=" << last << " flip=" << flip);
                ConvShape shape;
                shape.n = 2;
                shape.c = 6;
                shape.k = 8;
                shape.groups = 2;
                shape.channelsLast = last;
                if(rank == 1)
                {
                    shape.input = {13};
                    shape.filter = {5};
                    shape.stride = {2};
                    shape.dilation = {2};
                    shape.prePadding = {3};
                    shape.postPadding = {3};
                }
                else if(rank == 2)
                {
                    shape.input = {7, 9};
                    shape.filter = {3, 3};
                    shape.stride = {2, 1};
                    shape.dilation = {1, 2};
                    shape.prePadding = {1, 2};
                    shape.postPadding = {2, 0};
                }
                else
                {
                    shape.input = {5, 7, 9};
                    shape.filter = {3, 3, 3};
                    shape.stride = {1, 2, 1};
                    shape.dilation = {1, 1, 2};
                    shape.prePadding = {1, 1, 2};
                    shape.postPadding = {0, 2, 0};
                }
                const auto spec = specFor(shape, ConvDirection::BACKWARD_WEIGHTS);
                Tensor<float> x(spec.x.dims, spec.x.strides);
                Tensor<float> dy(spec.y.dims, spec.y.strides);
                Tensor<double> expected(spec.w.dims, spec.w.strides);
                x.fillTensorWithRandomValues(-0.25f, 0.25f, 8821);
                dy.fillTensorWithRandomValues(-0.25f, 0.25f, 8822);
                hipdnn_test_sdk::utilities::CpuFpReferenceConvolution::
                    wgrad<float, double, float, double>(x,
                                                        expected,
                                                        dy,
                                                        spec.stride,
                                                        spec.dilation,
                                                        spec.prePadding,
                                                        spec.postPadding);
                const auto actual = oracle.execute(static_cast<const float*>(x.rawDeviceData()),
                                                   static_cast<const float*>(dy.rawDeviceData()),
                                                   geometry(spec, flip),
                                                   nullptr);
                const auto* cpu = static_cast<const double*>(expected.rawHostData());
                for(size_t index = 0; index < actual.size(); ++index)
                {
                    size_t rest = index;
                    size_t offset = 0;
                    for(size_t axis = spec.w.dims.size(); axis-- > 0;)
                    {
                        size_t coordinate = rest % static_cast<size_t>(spec.w.dims[axis]);
                        rest /= static_cast<size_t>(spec.w.dims[axis]);
                        if(flip && axis >= 2)
                        {
                            coordinate = static_cast<size_t>(spec.w.dims[axis] - 1) - coordinate;
                        }
                        offset += coordinate * static_cast<size_t>(spec.w.strides[axis]);
                    }
                    EXPECT_NEAR(actual[index], cpu[offset], 1e-11 * (1.0 + std::abs(cpu[offset])))
                        << "weight=" << index;
                }
            }
        }
    }
}
TEST(TestGpuWrwReference, LongCancellationRetainsSmallTerms)
{
    SKIP_IF_NO_DEVICES();
    constexpr int64_t LENGTH = 131072;
    Tensor<float> x({1, 1, LENGTH});
    Tensor<float> dy({1, 1, LENGTH});
    auto* xp = static_cast<float*>(x.rawHostData());
    auto* yp = static_cast<float*>(dy.rawHostData());
    constexpr std::array<float, 4> PATTERN = {16777216.0f, 1.0f, -16777216.0f, 1.0f};
    for(int64_t i = 0; i < LENGTH; ++i)
    {
        xp[i] = PATTERN[static_cast<size_t>(i % 4)];
        yp[i] = 1.0f;
    }
    x.markHostModified();
    dy.markHostModified();
    ReferenceGeometry a{};
    a.n = a.c = a.k = a.groups = a.weights = 1;
    a.input[0] = a.input[1] = a.output[0] = a.output[1] = 1;
    a.input[2] = a.output[2] = LENGTH;
    for(int j = 0; j < 3; ++j)
    {
        a.filter[j] = a.stride[j] = a.dilation[j] = 1;
    }
    a.xs[0] = a.xs[1] = a.ys[0] = a.ys[1] = LENGTH;
    a.xs[4] = a.ys[4] = 1;
    const Fp64WrwReference oracle;
    const auto result = oracle.execute(static_cast<const float*>(x.rawDeviceData()),
                                       static_cast<const float*>(dy.rawDeviceData()),
                                       a,
                                       nullptr);
    EXPECT_DOUBLE_EQ(result.at(0), 65536.0);
}
}
#endif
