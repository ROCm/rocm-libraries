// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp"
#include "ck_tile/ops/grouped_convolution/pipeline/wcnn_pipeline_problem.hpp"
#include "ck_tile/ops/grouped_convolution/pipeline/wcnn_pipeline_ag_bg_cr_comp_default.hpp"
#include "ck_tile/ops/grouped_convolution/pipeline/wcnn_forward_ag_bg_cr_default_policy.hpp"

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    return ck_tile::make_tuple(rtol, atol);
}

template <typename Tuple>
class TestCkTileWcnnForward : public ::testing::Test
{
    protected:
    // Data types
    using InDataType  = std::tuple_element_t<0, Tuple>;
    using WeiDataType = std::tuple_element_t<1, Tuple>;
    using AccDataType = std::tuple_element_t<2, Tuple>;
    using OutDataType = std::tuple_element_t<3, Tuple>;

    // Block shape parameters
    static constexpr ck_tile::index_t HPerBlock = std::tuple_element_t<4, Tuple>::value;
    static constexpr ck_tile::index_t WPerBlock = std::tuple_element_t<5, Tuple>::value;
    static constexpr ck_tile::index_t CPerBlock = std::tuple_element_t<6, Tuple>::value;
    static constexpr ck_tile::index_t KPerBlock = std::tuple_element_t<7, Tuple>::value;
    static constexpr ck_tile::index_t HPerWcnn  = std::tuple_element_t<8, Tuple>::value;
    static constexpr ck_tile::index_t WPerWcnn  = std::tuple_element_t<9, Tuple>::value;
    static constexpr ck_tile::index_t WarpsInH  = std::tuple_element_t<10, Tuple>::value;
    static constexpr ck_tile::index_t WarpsInW  = std::tuple_element_t<11, Tuple>::value;
    static constexpr ck_tile::index_t WarpsInK  = std::tuple_element_t<12, Tuple>::value;
    static constexpr ck_tile::index_t BlockSize = WarpsInH * WarpsInW * WarpsInK * 32;

    // Layouts
    using InLayout  = ck_tile::tensor_layout::convolution::NHWGC;
    using WeiLayout = ck_tile::tensor_layout::convolution::GKYXC;
    using OutLayout = ck_tile::tensor_layout::convolution::NHWGK;

    using Traits =
        ck_tile::GroupedConvTraits<2,
                                   ck_tile::ConvolutionSpecialization::Filter1x1Stride1Pad0,
                                   InLayout,
                                   WeiLayout,
                                   ck_tile::tuple<>,
                                   OutLayout,
                                   8,
                                   8,
                                   8,
                                   1,
                                   false,
                                   false,
                                   true>;

    using TestBlockShape = ck_tile::BlockWcnnFwdShape<
        ck_tile::WcnnBlockTile<HPerBlock, WPerBlock, CPerBlock, KPerBlock>,
        ck_tile::WcnnWarpTile<HPerWcnn, WPerWcnn>,
        ck_tile::WcnnWarpCount<WarpsInH, WarpsInW, WarpsInK>>;

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<TestBlockShape, 8, 4>;

    using PipelineProblem = ck_tile::WcnnFwdPipelineProblem<Traits,
                                                            InDataType,
                                                            WeiDataType,
                                                            AccDataType,
                                                            OutDataType,
                                                            TestBlockShape,
                                                            1, // FilterY
                                                            1, // FilterX
                                                            1, // DilationY
                                                            1  // DilationX
                                                            >;

    using Pipeline = ck_tile::WcnnFwdPipeline<PipelineProblem>;

    using EpilogueProblem =
        ck_tile::Default2DEpilogueProblem<AccDataType, OutDataType, false, false>;

    using Epilogue = ck_tile::Default2DEpilogue<EpilogueProblem>;

    using Kernel =
        ck_tile::GroupedConvolutionForwardKernel<Traits, TilePartitioner, Pipeline, Epilogue>;

    public:
    void Run(const ck_tile::index_t G,
             const ck_tile::index_t N,
             const ck_tile::index_t K,
             const ck_tile::index_t C,
             const ck_tile::index_t Hi,
             const ck_tile::index_t Wi)
    {
        using namespace ck_tile;

        const index_t Y = 1, X = 1;
        conv::ConvParam conv_param{2, G, N, K, C, {Y, X}, {Hi, Wi}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};

        // Create host tensors
        const auto in_desc =
            conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
        const auto wei_desc =
            conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
        const auto out_desc =
            conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

        HostTensor<InDataType> input(in_desc);
        HostTensor<WeiDataType> weight(wei_desc);
        HostTensor<OutDataType> output(out_desc);

        FillUniformDistributionIntegerValue<InDataType>{-5, 5, 11939}(input);
        FillUniformDistributionIntegerValue<WeiDataType>{-5, 5, 11940}(weight);

        // Device memory
        DeviceMem input_dev(input.get_element_space_size_in_bytes());
        DeviceMem weight_dev(weight.get_element_space_size_in_bytes());
        DeviceMem output_dev(output.get_element_space_size_in_bytes());

        input_dev.ToDevice(input.data());
        weight_dev.ToDevice(weight.data());
        output_dev.SetZero();

        // Build and launch kernel
        GroupedConvFwdHostArgs<> host_args(conv_param,
                                           input_dev.GetDeviceBuffer(),
                                           weight_dev.GetDeviceBuffer(),
                                           {},
                                           output_dev.GetDeviceBuffer(),
                                           1);

        auto kargs = Kernel::MakeKernelArgs(host_args);
        ASSERT_TRUE(Kernel::IsSupportedArgument(kargs));

        const dim3 grids  = Kernel::GridSize(kargs);
        const dim3 blocks = Kernel::BlockSize();

        const auto kernel_func =
            make_kernel<1>(Kernel{}, grids, blocks, Kernel::GetSmemSize(), kargs);
        launch_kernel(stream_config{nullptr, false}, kernel_func);

        // Copy result back
        output_dev.FromDevice(output.data());

        // CPU reference
        HostTensor<OutDataType> output_ref(out_desc);
        output_ref.SetZero();

        reference_grouped_conv_fwd<2, InDataType, WeiDataType, OutDataType>(
            input,
            weight,
            output_ref,
            conv_param.conv_filter_strides_,
            conv_param.conv_filter_dilations_,
            conv_param.input_left_pads_,
            conv_param.input_right_pads_);

        // Compare
        const float max_accumulated_value =
            *std::max_element(output_ref.mData.begin(), output_ref.mData.end());
        const auto rtol_atol =
            calculate_rtol_atol<InDataType, WeiDataType, AccDataType, OutDataType>(
                C, 1, max_accumulated_value);
        bool pass = check_err(output,
                              output_ref,
                              "WCNN Fwd: GPU vs CPU mismatch!",
                              rtol_atol.at(ck_tile::number<0>{}),
                              rtol_atol.at(ck_tile::number<1>{}));
        EXPECT_TRUE(pass);
    }
};
