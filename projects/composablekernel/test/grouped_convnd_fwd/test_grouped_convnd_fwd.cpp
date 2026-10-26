// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <gtest/gtest.h>

#include "ck/utility/common_header.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "profiler/profile_grouped_conv_fwd_impl.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace {
template <ck::index_t D, typename T>
void CheckQualifiedForwardDirtyOutput()
{
    using Pass = ck::tensor_operation::element_wise::PassThrough;
    using namespace ck::tensor_layout::convolution;
    using ALayout = std::conditional_t<D == 2, NHWGC, NDHWGC>;
    using BLayout = std::conditional_t<D == 2, GKYXC, GKZYXC>;
    using ELayout = std::conditional_t<D == 2, NHWGK, NDHWGK>;
    using Op = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
        D, ALayout, BLayout, ck::Tuple<>, ELayout, T, T, ck::Tuple<>, T, Pass, Pass, Pass>;
    constexpr ck::index_t G = 3, N = 2, C = 16, K = 16;
    std::array<ck::index_t, D + 3> a{}, as{}, b{}, bs{}, e{}, es{};
    std::array<ck::index_t, D> stride{}, dilation{}, pads{};
    a[0] = b[0] = e[0] = G;
    a[1] = e[1] = N;
    b[1] = e[2] = K;
    a[2] = b[2] = C;
    ck::index_t spatial = 1;
    for(ck::index_t d = D; d > 0; --d)
    {
        a[d + 2] = e[d + 2] = d == D ? 5 : 3;
        b[d + 2] = 1;
        as[d + 2] = spatial * G * C;
        es[d + 2] = spatial * G * K;
        bs[d + 2] = C;
        spatial *= a[d + 2];
        stride[d - 1] = dilation[d - 1] = 1;
    }
    as[0] = C;
    as[1] = spatial * G * C;
    as[2] = bs[2] = es[2] = 1;
    bs[0] = K * C;
    bs[1] = C;
    es[0] = K;
    es[1] = spatial * G * K;
    std::vector<T> input(N * spatial * G * C), weight(G * K * C);
    std::vector<T> actual(N * spatial * G * K);
    for(std::size_t i = 0; i < input.size(); ++i)
        input[i] = ck::type_convert<T>((static_cast<int>(i % 7) - 3) * 0.25f);
    for(std::size_t i = 0; i < weight.size(); ++i)
        weight[i] = ck::type_convert<T>((static_cast<int>(i % 5) - 2) * 0.5f);
    ck::DeviceMem input_device(input.size() * sizeof(T));
    ck::DeviceMem weight_device(weight.size() * sizeof(T));
    ck::DeviceMem output_device(actual.size() * sizeof(T));
    input_device.ToDevice(input.data());
    weight_device.ToDevice(weight.data());
    const auto instances =
        ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<Op>::GetInstances();
    bool executed = false;
    for(const auto& op : instances)
    {
        auto arg = op->MakeArgumentPointer(input_device.GetDeviceBuffer(),
                                          weight_device.GetDeviceBuffer(), {},
                                          output_device.GetDeviceBuffer(), a, as, b, bs, {}, {},
                                          e, es, stride, dilation, pads, pads, Pass{}, Pass{}, Pass{});
        if(!ck::tensor_operation::device::IsOutputFullyDefined(*op, arg.get()))
            continue;
        ASSERT_TRUE(op->IsSupportedArgument(arg.get()));
        ck::DeviceMem workspace(op->GetWorkSpaceSize(arg.get()));
        op->SetWorkSpacePointer(arg.get(), workspace.GetDeviceBuffer());
        for(float poison : {113.f, std::numeric_limits<float>::quiet_NaN()})
        {
            std::fill(actual.begin(), actual.end(), ck::type_convert<T>(poison));
            output_device.ToDevice(actual.data());
            op->MakeInvokerPointer()->Run(arg.get(), StreamConfig{nullptr, false});
            output_device.FromDevice(actual.data());
            // Independent 1x1 convolution reference, checking every group, batch,
            // channel and spatial cell, including the non-tile-aligned spatial tail.
            for(ck::index_t n = 0; n < N; ++n)
                for(ck::index_t s = 0; s < spatial; ++s)
                    for(ck::index_t g = 0; g < G; ++g)
                        for(ck::index_t k = 0; k < K; ++k)
                        {
                            float want = 0;
                            for(ck::index_t c = 0; c < C; ++c)
                                want += ck::type_convert<float>(input[((n * spatial + s) * G + g) * C + c]) *
                                        ck::type_convert<float>(weight[(g * K + k) * C + c]);
                            const float got = ck::type_convert<float>(
                                actual[((n * spatial + s) * G + g) * K + k]);
                            EXPECT_TRUE(std::isfinite(got)) << op->GetTypeString();
                            EXPECT_EQ(got, want) << op->GetTypeString() << " group=" << g;
                        }
        }
        executed = true;
    }
    EXPECT_TRUE(executed) << "No qualified forward invocation executed";
}

TEST(TestGroupedConvndFwdOutputContract, PackedDirtyOutput)
{
#ifdef CK_ENABLE_FP16
    CheckQualifiedForwardDirtyOutput<2, ck::half_t>();
    CheckQualifiedForwardDirtyOutput<3, ck::half_t>();
#else
    GTEST_SKIP() << "FP16 instances disabled";
#endif
}
} // namespace


static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;
using I8                          = int8_t;
using F8                          = ck::f8_t;
using BF8                         = ck::bf8_t;
using F16                         = ck::half_t;
using BF16                        = ck::bhalf_t;
using F32                         = float;

template <typename Tuple>
class TestGroupedConvndFwd : public ::testing::Test
{
    protected:
    using InDataType   = std::tuple_element_t<0, Tuple>;
    using WeiDataType  = std::tuple_element_t<1, Tuple>;
    using OutDataType  = std::tuple_element_t<2, Tuple>;
    using AComputeType = std::tuple_element_t<3, Tuple>;
    using BComputeType = std::tuple_element_t<4, Tuple>;
    using InLayout     = std::tuple_element_t<5, Tuple>;
    using WeiLayout    = std::tuple_element_t<6, Tuple>;
    using OutLayout    = std::tuple_element_t<7, Tuple>;
    using IndexType    = ck::index_t;

    std::vector<ck::utils::conv::ConvParam> conv_params;
#if defined(CK_TEST_DISABLE_GPU_VALIDATION)
    static constexpr int verify_ = 1; // CPU reference
#else
    static constexpr int verify_ = 2; // GPU reference
#endif
    // On gfx1250, the naive GPU reference kernel can hang. Always use CPU reference.
    int get_verify() const { return (ck::is_gfx125_supported() && verify_ == 2) ? 1 : verify_; }
    template <ck::index_t NDimSpatial>
    void Run()
    {
        EXPECT_FALSE(conv_params.empty());
        bool pass = true;
        for(size_t i = 0; i < conv_params.size(); i++)
        {
            if((param_mask & (1 << i)) == 0)
            {
                continue;
            }
            // FP8 workaround for CDNA1/2 is currently broken, do not test.
            if(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a")
            {
                if(std::is_same<InDataType, F8>::value || std::is_same<InDataType, BF8>::value)
                {
                    printf("Skipping FP8 / BF8 tests on CDNA1/2.\n");
                    continue;
                }
            }
            auto& param = conv_params[i];
            pass        = pass && ck::profiler::profile_grouped_conv_fwd_impl<NDimSpatial,
                                                                              InLayout,
                                                                              WeiLayout,
                                                                              OutLayout,
                                                                              InDataType,
                                                                              WeiDataType,
                                                                              OutDataType,
                                                                              AComputeType,
                                                                              BComputeType,
                                                                              IndexType>(
                               get_verify(), // do_verification
                               1,            // init_method: integer value
                               false,        // do_log
                               false,        // time_kernel
                               param,
                               ck::tensor_operation::element_wise::PassThrough{},
                               instance_index);
        }
        EXPECT_TRUE(pass);
    }
};

using namespace ck::tensor_layout::convolution;

using KernelTypes1d = ::testing::Types<std::tuple<F32, F32, F32, F32, F32, GNWC, GKXC, GNWK>,
                                       std::tuple<F16, F16, F16, F16, F16, GNWC, GKXC, GNWK>,
                                       std::tuple<BF16, BF16, BF16, BF16, BF16, GNWC, GKXC, GNWK>,
                                       std::tuple<I8, I8, I8, I8, I8, GNWC, GKXC, GNWK>>;

using KernelTypes2d =
    ::testing::Types<std::tuple<F32, F32, F32, F32, F32, GNHWC, GKYXC, GNHWK>,
                     std::tuple<F16, F16, F16, F16, F16, GNHWC, GKYXC, GNHWK>,
                     std::tuple<BF16, BF16, BF16, BF16, BF16, GNHWC, GKYXC, GNHWK>,
                     std::tuple<F32, F32, F32, F32, F32, NHWGC, GKYXC, NHWGK>,
                     std::tuple<F16, F16, F16, F16, F16, NHWGC, GKYXC, NHWGK>,
                     std::tuple<BF16, BF16, BF16, BF16, BF16, NHWGC, GKYXC, NHWGK>,
                     std::tuple<I8, I8, I8, I8, I8, NHWGC, GKYXC, NHWGK>,
                     std::tuple<F32, F32, F32, F32, F32, NGCHW, GKYXC, NGKHW>,
                     std::tuple<F16, F16, F16, F16, F16, NGCHW, GKYXC, NGKHW>,
                     std::tuple<BF16, BF16, BF16, BF16, BF16, NGCHW, GKYXC, NGKHW>,
                     std::tuple<I8, I8, I8, I8, I8, NGCHW, GKYXC, NGKHW>,
                     std::tuple<F32, F32, F32, F32, F32, NGCHW, GKCYX, NGKHW>,
                     std::tuple<F16, F16, F16, F16, F16, NGCHW, GKCYX, NGKHW>,
                     std::tuple<BF16, BF16, BF16, BF16, BF16, NGCHW, GKCYX, NGKHW>>;

using KernelTypes3d =
    ::testing::Types<std::tuple<F32, F32, F32, F32, F32, GNDHWC, GKZYXC, GNDHWK>,
                     std::tuple<F16, F16, F16, F16, F16, GNDHWC, GKZYXC, GNDHWK>,
                     std::tuple<BF16, BF16, BF16, BF16, BF16, GNDHWC, GKZYXC, GNDHWK>,
                     std::tuple<I8, I8, I8, I8, I8, GNDHWC, GKZYXC, GNDHWK>,
                     std::tuple<I8, I8, I8, I8, I8, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<F8, F8, F8, F8, F8, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<BF8, BF8, F8, BF8, BF8, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<F8, BF8, F8, F8, BF8, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<BF8, F8, F8, BF8, F8, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<F32, F32, F32, F32, F32, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<F16, F16, F16, F16, F16, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<BF16, BF16, BF16, BF16, BF16, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<F32, F32, F32, F32, F32, NGCDHW, GKCZYX, NGKDHW>,
                     std::tuple<F16, F16, F16, F16, F16, NGCDHW, GKCZYX, NGKDHW>,
                     std::tuple<BF16, BF16, BF16, BF16, BF16, NGCDHW, GKCZYX, NGKDHW>>;

template <typename Tuple>
class TestGroupedConvndFwd1d : public TestGroupedConvndFwd<Tuple>
{
};

template <typename Tuple>
class TestGroupedConvndFwd2d : public TestGroupedConvndFwd<Tuple>
{
};

template <typename Tuple>
class TestGroupedConvndFwd3d : public TestGroupedConvndFwd<Tuple>
{
};

TYPED_TEST_SUITE(TestGroupedConvndFwd1d, KernelTypes1d);
TYPED_TEST_SUITE(TestGroupedConvndFwd2d, KernelTypes2d);
TYPED_TEST_SUITE(TestGroupedConvndFwd3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndFwd1d, Test1D)
{
    this->conv_params.clear();
    this->conv_params.push_back({1, 2, 32, 128, 256, {1}, {14}, {2}, {1}, {0}, {0}});
    this->conv_params.push_back({1, 2, 32, 128, 256, {3}, {28}, {1}, {1}, {1}, {1}});
    this->conv_params.push_back({1, 2, 32, 128, 256, {1}, {3}, {1}, {1}, {0}, {0}});
    this->conv_params.push_back({1, 1, 1, 1, 32, {3}, {32}, {1}, {1}, {1}, {1}});
    this->conv_params.push_back({1, 1, 1, 64, 3, {3}, {32}, {1}, {1}, {1}, {1}});
    this->conv_params.push_back({1, 96, 1, 1, 1, {3}, {512}, {1}, {1}, {1}, {1}});
    this->template Run<1>();
}

TYPED_TEST(TestGroupedConvndFwd2d, Test2D)
{
    this->conv_params.clear();
    this->conv_params.push_back(
        {2, 3, 5, 96, 200, {1, 1}, {73, 128}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 1, 1, 32, 32, {1, 1}, {128, 128}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 1, 1, 32, 32, {2, 2}, {128, 128}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 1, 1, 32, 32, {3, 3}, {128, 128}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 1, 1, 32, 32, {5, 5}, {128, 128}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 1, 1, 32, 32, {9, 9}, {128, 128}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});

    this->conv_params.push_back(
        {2, 2, 32, 128, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 2, 32, 128, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});

    this->conv_params.push_back(
        {2, 2, 32, 128, 256, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back({2, 1, 1, 1, 32, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back({2, 1, 1, 64, 3, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back({2, 1, 1, 1, 1, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});

    this->conv_params.push_back(
        {2, 96, 1, 1, 1, {1, 1}, {120, 160}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 96, 1, 1, 1, {3, 3}, {120, 160}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->template Run<2>();
}

TYPED_TEST(TestGroupedConvndFwd3d, Test3D)
{
    this->conv_params.clear();

    this->conv_params.push_back(
        {3, 3, 5, 96, 200, {1, 1, 1}, {37, 37, 16}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {1, 1, 1}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {2, 2, 2}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {5, 5, 5}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {9, 9, 9}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});

    this->conv_params.push_back(
        {3, 2, 32, 128, 256, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 2, 32, 128, 256, {3, 3, 3}, {14, 14, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});

    this->conv_params.push_back(
        {3, 2, 32, 128, 256, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {1, 1, 1}, {16, 16, 16}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});

    this->conv_params.push_back(
        {3, 1, 1, 1, 32, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 1, 1, 64, 3, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 1, 1, 1, 1, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});

    this->conv_params.push_back(
        {3, 96, 1, 1, 1, {1, 1, 1}, {120, 40, 20}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 96, 1, 1, 1, {3, 3, 3}, {120, 40, 20}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->template Run<3>();
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    if(argc == 1) {}
    else if(argc == 3)
    {
        param_mask     = strtol(argv[1], nullptr, 0);
        instance_index = atoi(argv[2]);
    }
    else
    {
        std::cout << "Usage of " << argv[0] << std::endl;
        std::cout << "Arg1,2: param_mask instance_index(-1 means all)" << std::endl;
    }
    return RUN_ALL_TESTS();
}
