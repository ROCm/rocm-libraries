// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "ck/utility/common_header.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "profiler/profile_grouped_conv_fwd_impl.hpp"

static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;
using F32                         = float;
using TF32                        = ck::tf32_t;

// fp32 tensors with tf32 compute: native xf32 MFMA on gfx942, bf16x3 emulation on gfx950,
// gfx11 and gfx120. Integer init keeps every value exactly representable in bf16, so the
// result must match the fp32 reference.
template <typename Tuple>
class TestGroupedConvndFwdTf32 : public ::testing::Test
{
    protected:
    using InLayout  = std::tuple_element_t<0, Tuple>;
    using WeiLayout = std::tuple_element_t<1, Tuple>;
    using OutLayout = std::tuple_element_t<2, Tuple>;

    std::vector<ck::utils::conv::ConvParam> conv_params;
#if defined(CK_TEST_DISABLE_GPU_VALIDATION)
    static constexpr int verify_ = 1; // CPU reference
#else
    static constexpr int verify_ = 2; // GPU reference
#endif

    template <ck::index_t NDimSpatial>
    void Run()
    {
        if(!ck::is_tf32_conv_fwd_supported())
        {
            GTEST_SKIP() << "tf32 is not supported on " << ck::get_device_name();
        }
        EXPECT_FALSE(conv_params.empty());
        bool pass = true;
        for(size_t i = 0; i < conv_params.size(); i++)
        {
            if((param_mask & (1 << i)) == 0)
            {
                continue;
            }
            pass = pass && ck::profiler::profile_grouped_conv_fwd_impl<NDimSpatial,
                                                                       InLayout,
                                                                       WeiLayout,
                                                                       OutLayout,
                                                                       F32,
                                                                       F32,
                                                                       F32,
                                                                       TF32,
                                                                       TF32,
                                                                       ck::index_t>(
                               verify_,
                               1,     // init_method: integer value
                               false, // do_log
                               false, // time_kernel
                               conv_params[i],
                               ck::tensor_operation::element_wise::PassThrough{},
                               instance_index);
        }
        EXPECT_TRUE(pass);
    }
};

using namespace ck::tensor_layout::convolution;

using KernelTypes2d = ::testing::Types<std::tuple<NHWGC, GKYXC, NHWGK>>;
using KernelTypes3d = ::testing::Types<std::tuple<NDHWGC, GKZYXC, NDHWGK>>;

template <typename Tuple>
class TestGroupedConvndFwdTf32_2d : public TestGroupedConvndFwdTf32<Tuple>
{
};

template <typename Tuple>
class TestGroupedConvndFwdTf32_3d : public TestGroupedConvndFwdTf32<Tuple>
{
};

TYPED_TEST_SUITE(TestGroupedConvndFwdTf32_2d, KernelTypes2d);
TYPED_TEST_SUITE(TestGroupedConvndFwdTf32_3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndFwdTf32_2d, Test2D)
{
    this->conv_params.clear();
    this->conv_params.push_back(
        {2, 1, 1, 32, 32, {3, 3}, {128, 128}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 2, 32, 128, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 2, 32, 128, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 1, 16, 64, 64, {3, 3}, {56, 56}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back({2, 1, 1, 64, 3, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->template Run<2>();
}

TYPED_TEST(TestGroupedConvndFwdTf32_3d, Test3D)
{
    this->conv_params.clear();
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 2, 32, 128, 256, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 2, 32, 128, 256, {3, 3, 3}, {14, 14, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
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
