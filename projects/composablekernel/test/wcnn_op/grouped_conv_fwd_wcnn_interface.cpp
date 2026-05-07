// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wcnn_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/host_utility/device_prop.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

#include <gtest/gtest.h>

#define DEFAULT_H_PERWAVE 8
#define DEFAULT_W_PERWAVE 8
#define DEFAULT_H_PERBLOCK 16
#define DEFAULT_W_PERBLOCK 16
#define DEFAULT_C_PERBLOCK 16
#define DEFAULT_K_PERBLOCK 16
#define DEFAULT_BLOCKSIZE 128
using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::UnaryConvert;
using ConvolutionForwardSpecialization =
    ck::tensor_operation::device::ConvolutionForwardSpecialization;

template <typename InputLay, typename WeightLay, typename OutputLay>
struct CommonLayoutSetting
{
    using InputLayout  = InputLay;
    using WeightLayout = WeightLay;
    using OutputLayout = OutputLay;
};

template <ck::index_t NDimSpatial>
struct PackedLayout;
template <ck::index_t NDimSpatial>
struct StridedLayout;
template <ck::index_t NDimSpatial>
struct GCPackedLayout;

using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

namespace ctl = ck::tensor_layout::convolution;

// Packed layout
template <>
struct PackedLayout<1> final : CommonLayoutSetting<ctl::GNWC, ctl::GKXC, ctl::GNWK>
{
};

template <>
struct PackedLayout<2> final : CommonLayoutSetting<ctl::GNHWC, ctl::GKYXC, ctl::GNHWK>
{
};

template <>
struct PackedLayout<3> final : CommonLayoutSetting<ctl::GNDHWC, ctl::GKZYXC, ctl::GNDHWK>
{
};

// Strided layout
template <>
struct StridedLayout<1> final : CommonLayoutSetting<ctl::G_NW_C, ctl::G_K_X_C, ctl::G_NW_K>
{
};

template <>
struct StridedLayout<2> final : CommonLayoutSetting<ctl::G_NHW_C, ctl::G_K_YX_C, ctl::G_NHW_K>
{
};

template <>
struct StridedLayout<3> final : CommonLayoutSetting<ctl::G_NDHW_C, ctl::G_K_ZYX_C, ctl::G_NDHW_K>
{
};

// GC Packed layout
template <>
struct GCPackedLayout<1> final : CommonLayoutSetting<ctl::NWGC, ctl::KXGC, ctl::NWGK>
{
};

template <>
struct GCPackedLayout<2> final : CommonLayoutSetting<ctl::NHWGC, ctl::KYXGC, ctl::NHWGK>
{
};

template <>
struct GCPackedLayout<3> final : CommonLayoutSetting<ctl::NDHWGC, ctl::KZYXGC, ctl::NDHWGK>
{
};

template <ck::index_t NDimSpatial, ck::index_t FilterSize, ck::index_t DilationSize>
class TestGroupedConvFwdWcnnInterface : public ::testing::Test
{
    protected:
    ck::utils::conv::ConvParam conv_param;
    HostTensorDescriptor in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<
            typename PackedLayout<NDimSpatial>::InputLayout>(conv_param);
    HostTensorDescriptor wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<
            typename PackedLayout<NDimSpatial>::WeightLayout>(conv_param);
    HostTensorDescriptor out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<
            typename PackedLayout<NDimSpatial>::OutputLayout>(conv_param);
    static constexpr ck::index_t HPerWcnn  = 4;
    static constexpr ck::index_t WPerWcnn  = 2;
    static constexpr ck::index_t BlockSize = DEFAULT_BLOCKSIZE;
    static constexpr ck::index_t HPerWave  = DEFAULT_H_PERWAVE;
    static constexpr ck::index_t WPerWave  = DEFAULT_W_PERWAVE;
    static constexpr ck::index_t HPerBlock = DEFAULT_H_PERBLOCK;
    static constexpr ck::index_t WPerBlock = DEFAULT_W_PERBLOCK;
    static constexpr ck::index_t CPerBlock = DEFAULT_C_PERBLOCK;
    static constexpr ck::index_t KPerBlock = DEFAULT_K_PERBLOCK;
    static constexpr ck::index_t HRepeat   = HPerWave / HPerWcnn;
    static constexpr ck::index_t WRepeat   = WPerWave / WPerWcnn;

    using EmptyTuple = ck::Tuple<>;

    void Init(const ck::utils::conv::ConvParam& param, int desc_mode = 0)
    {
        conv_param = param;
        if(desc_mode == 0 || desc_mode == 1)
        {
            in_g_n_c_wis_desc = ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<
                typename PackedLayout<NDimSpatial>::InputLayout>(conv_param);
            wei_g_k_c_xs_desc = ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<
                typename PackedLayout<NDimSpatial>::WeightLayout>(conv_param);
            out_g_n_k_wos_desc =
                ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<
                    typename PackedLayout<NDimSpatial>::OutputLayout>(conv_param);
            if(desc_mode == 1)
            {
                auto in_strides = in_g_n_c_wis_desc.GetStrides();
                in_strides[0] *= 4;
                in_strides[1] *= 2;
                for(ck::index_t i = 0; i < NDimSpatial; i++)
                {
                    in_strides[i + 3] *= 2;
                }

                if constexpr(NDimSpatial == 1)
                {
                    in_g_n_c_wis_desc =
                        HostTensorDescriptor(in_g_n_c_wis_desc.GetLengths(),
                                             in_strides,
                                             ck::tensor_layout::convolution::GNCW{});
                }
                else if constexpr(NDimSpatial == 2)
                {
                    in_g_n_c_wis_desc =
                        HostTensorDescriptor(in_g_n_c_wis_desc.GetLengths(),
                                             in_strides,
                                             ck::tensor_layout::convolution::GNCHW{});
                }
                else if constexpr(NDimSpatial == 3)
                {
                    in_g_n_c_wis_desc =
                        HostTensorDescriptor(in_g_n_c_wis_desc.GetLengths(),
                                             in_strides,
                                             ck::tensor_layout::convolution::GNCDHW{});
                }

                auto wei_strides = wei_g_k_c_xs_desc.GetStrides();
                wei_strides[0] *= 8; // G
                wei_strides[1] *= 4; // K
                for(ck::index_t i = 0; i < NDimSpatial; i++)
                {
                    wei_strides[i + 3] *= 2;
                }

                if constexpr(NDimSpatial == 1)
                {
                    wei_g_k_c_xs_desc =
                        HostTensorDescriptor(wei_g_k_c_xs_desc.GetLengths(),
                                             wei_strides,
                                             ck::tensor_layout::convolution::GKCX{});
                }
                else if constexpr(NDimSpatial == 2)
                {
                    wei_g_k_c_xs_desc =
                        HostTensorDescriptor(wei_g_k_c_xs_desc.GetLengths(),
                                             wei_strides,
                                             ck::tensor_layout::convolution::GKCYX{});
                }
                else if constexpr(NDimSpatial == 3)
                {
                    wei_g_k_c_xs_desc =
                        HostTensorDescriptor(wei_g_k_c_xs_desc.GetLengths(),
                                             wei_strides,
                                             ck::tensor_layout::convolution::GKCZYX{});
                }

                auto out_strides = out_g_n_k_wos_desc.GetStrides();
                out_strides[0] *= 4;
                out_strides[1] *= 2;
                for(ck::index_t i = 0; i < NDimSpatial; i++)
                {
                    out_strides[i + 3] *= 2;
                }

                if constexpr(NDimSpatial == 1)
                {
                    out_g_n_k_wos_desc =
                        HostTensorDescriptor(out_g_n_k_wos_desc.GetLengths(),
                                             out_strides,
                                             ck::tensor_layout::convolution::GNKW{});
                }
                else if constexpr(NDimSpatial == 2)
                {
                    out_g_n_k_wos_desc =
                        HostTensorDescriptor(out_g_n_k_wos_desc.GetLengths(),
                                             out_strides,
                                             ck::tensor_layout::convolution::GNKHW{});
                }
                else if constexpr(NDimSpatial == 3)
                {
                    out_g_n_k_wos_desc =
                        HostTensorDescriptor(out_g_n_k_wos_desc.GetLengths(),
                                             out_strides,
                                             ck::tensor_layout::convolution::GNKDHW{});
                }
            }
        }
        else if(desc_mode == 2)
        {
            in_g_n_c_wis_desc = ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<
                typename GCPackedLayout<NDimSpatial>::InputLayout>(conv_param);
            wei_g_k_c_xs_desc = ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<
                typename GCPackedLayout<NDimSpatial>::WeightLayout>(conv_param);
            out_g_n_k_wos_desc =
                ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<
                    typename GCPackedLayout<NDimSpatial>::OutputLayout>(conv_param);
        }
    }
    template <ConvolutionForwardSpecialization ConvSpec,
              typename InDataType,
              typename WeiDataType,
              typename AccDataType,
              typename DataLayout>
    bool Run()
    {
        using InputLayout  = DataLayout::InputLayout;
        using WeightLayout = DataLayout::WeightLayout;
        using OutputLayout = DataLayout::OutputLayout;

        constexpr ck::index_t InBlockTransferScalarPerVector =
            sizeof(uint32_t) / sizeof(InDataType);
        constexpr ck::index_t Cluster_In_C = CPerBlock / InBlockTransferScalarPerVector;
        constexpr ck::index_t Cluster_In_W = 4;

        constexpr ck::index_t ActiveBlockSize = BlockSize;
        constexpr ck::index_t Cluster_In_H    = ActiveBlockSize / Cluster_In_C / Cluster_In_W;
        using InBlockTransferThreadClusterLengths =
            ck::Sequence<Cluster_In_H, Cluster_In_W, Cluster_In_C>;

        constexpr ck::index_t WeiBlockTransferScalarPerVector =
            sizeof(uint32_t) / sizeof(WeiDataType);
        constexpr ck::index_t Cluster_Wei_C        = CPerBlock / WeiBlockTransferScalarPerVector;
        constexpr ck::index_t Cluster_Wei_K        = (ActiveBlockSize / Cluster_Wei_C) > KPerBlock
                                                         ? KPerBlock
                                                         : (ActiveBlockSize / Cluster_Wei_C);
        using WeiBlockTransferThreadClusterLengths = ck::Sequence<Cluster_Wei_K, 1, Cluster_Wei_C>;

        constexpr ck::index_t AccBlockTransferScalarPerVector = 2;
        constexpr ck::index_t Cluster_Acc_K = KPerBlock / AccBlockTransferScalarPerVector;
        constexpr ck::index_t Cluster_Acc_W = 4;
        constexpr ck::index_t Cluster_Acc_H = ActiveBlockSize / Cluster_Acc_K / Cluster_Acc_W;
        using AccBlockTransferClusterLengths =
            ck::Sequence<Cluster_Acc_H, Cluster_Acc_W, Cluster_Acc_K>;

        using DeviceConvFwdInstance =
            ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Wcnn_CShuffle<
                NDimSpatial,
                InputLayout,
                WeightLayout,
                EmptyTuple,
                OutputLayout,
                InDataType,
                WeiDataType,
                EmptyTuple,
                AccDataType,
                AccDataType,
                InElementOp,
                WeiElementOp,
                OutElementOp,
                ConvSpec,
                1,
                BlockSize,
                HPerBlock,
                WPerBlock,
                CPerBlock,
                KPerBlock,
                HRepeat,
                WRepeat,
                HPerWcnn,
                WPerWcnn,
                FilterSize,
                DilationSize,
                DilationSize,
                InBlockTransferThreadClusterLengths,
                InBlockTransferScalarPerVector,
                InBlockTransferScalarPerVector,
                false,
                true,
                false, // InTileLoad
                WeiBlockTransferThreadClusterLengths,
                WeiBlockTransferScalarPerVector,
                WeiBlockTransferScalarPerVector,
                false,
                true,
                false, // WeiTileLoad
                EmptyTuple,
                ck::Sequence<>,
                ck::Sequence<>,
                false,
                true,
                AccBlockTransferClusterLengths,
                AccBlockTransferScalarPerVector,
                false,
                false,
                false>;

        auto conv    = DeviceConvFwdInstance{};
        auto invoker = conv.MakeInvoker();

        std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
        std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
        std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
        std::array<ck::index_t, NDimSpatial> input_left_pads{};
        std::array<ck::index_t, NDimSpatial> input_right_pads{};

        auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

        copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
        copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
        copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
        copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
        copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
        copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
        copy(conv_param.conv_filter_strides_, conv_filter_strides);
        copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
        copy(conv_param.input_left_pads_, input_left_pads);
        copy(conv_param.input_right_pads_, input_right_pads);

        std::array<const void*, 0> ds{};

        auto argument = conv.MakeArgument(nullptr,
                                          nullptr,
                                          ds,
                                          nullptr,
                                          a_g_n_c_wis_lengths,
                                          a_g_n_c_wis_strides,
                                          b_g_k_c_xs_lengths,
                                          b_g_k_c_xs_strides,
                                          std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                          std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                          e_g_n_k_wos_lengths,
                                          e_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          InElementOp{},
                                          WeiElementOp{},
                                          OutElementOp{});

        return conv.IsSupportedArgument(argument);
    }
};

class TestGroupedConv2DFwdWcnnFilter1 : public TestGroupedConvFwdWcnnInterface<2, 1, 1>
{
};

class TestGroupedConv2DFwdWcnnFilter2 : public TestGroupedConvFwdWcnnInterface<2, 2, 1>
{
};

class TestGroupedConv2DFwdWcnnFilter3 : public TestGroupedConvFwdWcnnInterface<2, 3, 1>
{
};

class TestGroupedConv2DFwdWcnnFilter3Dilation2 : public TestGroupedConvFwdWcnnInterface<2, 3, 2>
{
};

class TestGroupedConv1DFwdWcnnFilter1 : public TestGroupedConvFwdWcnnInterface<1, 1, 1>
{
};

class TestGroupedConv3DFwdWcnnFilter1 : public TestGroupedConvFwdWcnnInterface<3, 1, 1>
{
};

class TestGroupedConv3DFwdWcnnFilter3 : public TestGroupedConvFwdWcnnInterface<3, 3, 1>
{
};

class TestGroupedConv3DFwdWcnnFilter3Dilation2 : public TestGroupedConvFwdWcnnInterface<3, 3, 2>
{
};

TEST_F(TestGroupedConv2DFwdWcnnFilter1, 2DFilter1)
{
    // Initialize parameter with packed layout.
    const ck::utils::conv::ConvParam param = {
        2, 1, 1, 64, 64, {1, 1}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    this->Init(param);

    // Packed layout
    bool packed_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<2>>();
    EXPECT_TRUE(packed_supported);

    // Strided layout compatible with packed layout
    bool strided_compatible =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           StridedLayout<2>>();
    EXPECT_TRUE(strided_compatible);

    // GCPacked layout incompatible with packed layout
    bool gc_packed_incompatible =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           GCPackedLayout<2>>();
    EXPECT_FALSE(gc_packed_incompatible);

    // Check stride != 1
    ck::utils::conv::ConvParam param_strid2 = {
        2, 1, 1, 64, 64, {1, 1}, {64, 64}, {2, 2}, {1, 1}, {0, 0}, {0, 0}};
    this->Init(param_strid2);
    bool is_supported_strid2 = this->template Run<ConvolutionForwardSpecialization::Filter1x1Pad0,
                                                  ck::half_t,
                                                  ck::half_t,
                                                  float,
                                                  PackedLayout<2>>();
    EXPECT_TRUE(is_supported_strid2);

    bool strid2_incompatible =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<2>>();
    EXPECT_FALSE(strid2_incompatible);

    // Check pad
    ck::utils::conv::ConvParam param_invalid_pad_left = {
        2, 1, 1, 64, 64, {1, 1}, {64, 64}, {2, 2}, {1, 1}, {0, 1}, {0, 0}};
    this->Init(param_invalid_pad_left);
    bool invalid_pad_left = this->template Run<ConvolutionForwardSpecialization::Filter1x1Pad0,
                                               ck::half_t,
                                               ck::half_t,
                                               float,
                                               PackedLayout<2>>();
    EXPECT_FALSE(invalid_pad_left);

    ck::utils::conv::ConvParam param_invalid_pad_right = {
        2, 1, 1, 64, 64, {1, 1}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, {0, 1}};
    this->Init(param_invalid_pad_right);
    bool invalid_pad_right =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<2>>();
    EXPECT_FALSE(invalid_pad_right);

    // Check C, K alignment
    ck::utils::conv::ConvParam param_invalid_c = {
        2, 1, 1, 64, 65, {1, 1}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    this->Init(param_invalid_c);
    bool invalid_c = this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                                        ck::half_t,
                                        ck::half_t,
                                        float,
                                        PackedLayout<2>>();
    EXPECT_FALSE(invalid_c);

    ck::utils::conv::ConvParam param_invalid_k = {
        2, 1, 1, 65, 64, {1, 1}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    this->Init(param_invalid_k);
    bool invalid_k = this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                                        ck::half_t,
                                        ck::half_t,
                                        float,
                                        PackedLayout<2>>();
    EXPECT_FALSE(invalid_k);

    // check Strided layout
    this->Init(param, 1);
    bool strided_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           StridedLayout<2>>();
    EXPECT_TRUE(strided_supported);

    bool packed_incompatible =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<2>>();
    EXPECT_FALSE(packed_incompatible);

    bool gc_packed_incompatible2 =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           GCPackedLayout<2>>();
    EXPECT_FALSE(gc_packed_incompatible2);

    // check GC Packed layout
    this->Init(param, 2);
    bool gc_packed_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           GCPackedLayout<2>>();
    EXPECT_TRUE(gc_packed_supported);

    bool strided_compatible2 =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           StridedLayout<2>>();
    EXPECT_TRUE(strided_compatible2);

    bool packed_incompatible2 =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<2>>();
    EXPECT_FALSE(packed_incompatible2);

    // check c mismatch
    const ck::utils::conv::ConvParam param_c_mismatch = {
        2, 1, 1, 64, 128, {1, 1}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    this->Init(param_c_mismatch);
    auto in_c_mismatch = this->in_g_n_c_wis_desc;
    this->Init(param);
    this->in_g_n_c_wis_desc = in_c_mismatch;
    bool c_mismatch = this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                                         ck::half_t,
                                         ck::half_t,
                                         float,
                                         PackedLayout<2>>();
    EXPECT_FALSE(c_mismatch);

    // check k mismatch
    const ck::utils::conv::ConvParam param_k_mismatch = {
        2, 1, 1, 128, 64, {1, 1}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    this->Init(param_k_mismatch);
    auto out_k_mismatch = this->out_g_n_k_wos_desc;
    this->Init(param);
    this->out_g_n_k_wos_desc = out_k_mismatch;
    bool k_mismatch = this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                                         ck::half_t,
                                         ck::half_t,
                                         float,
                                         PackedLayout<2>>();
    EXPECT_FALSE(k_mismatch);

    // check filter size mismatch
    const ck::utils::conv::ConvParam param_filter_mismatch = {
        2, 1, 1, 64, 64, {3, 3}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    this->Init(param_filter_mismatch);

    bool filter_mismatch =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<2>>();
    EXPECT_FALSE(filter_mismatch);

    // check size mismatch
    const ck::utils::conv::ConvParam param_size_mismatch = {
        2, 1, 1, 64, 64, {1, 1}, {128, 128}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    this->Init(param_size_mismatch);
    auto in_size_mismatch = this->in_g_n_c_wis_desc;
    this->Init(param);
    this->in_g_n_c_wis_desc = in_size_mismatch;
    bool size_mismatch = this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                                            ck::half_t,
                                            ck::half_t,
                                            float,
                                            PackedLayout<2>>();
    EXPECT_FALSE(size_mismatch);

    // check g mismatch
    const ck::utils::conv::ConvParam param_g_mismatch = {
        2, 2, 1, 64, 64, {1, 1}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    this->Init(param_g_mismatch);
    auto in_g_mismatch = this->in_g_n_c_wis_desc;
    this->Init(param);
    this->in_g_n_c_wis_desc = in_g_mismatch;
    bool g_mismatch = this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                                         ck::half_t,
                                         ck::half_t,
                                         float,
                                         PackedLayout<2>>();
    EXPECT_FALSE(g_mismatch);
}

TEST_F(TestGroupedConv2DFwdWcnnFilter3, 2DFilter3)
{
    // Initialize parameter with packed layout.
    const ck::utils::conv::ConvParam param = {
        2, 3, 1, 64, 64, {3, 3}, {64, 64}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    this->Init(param);

    // Packed layout
    bool packed_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<2>>();
    EXPECT_TRUE(packed_supported);

    // Strided layout compatible with packed layout
    bool strided_compatible =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           StridedLayout<2>>();
    EXPECT_TRUE(strided_compatible);

    bool multi_n_compatible =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<2>>();
    EXPECT_TRUE(multi_n_compatible);

    // GCPacked layout incompatible with packed layout
    bool gc_packed_incompatible =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           GCPackedLayout<2>>();
    EXPECT_FALSE(gc_packed_incompatible);

    // multi-n
    const ck::utils::conv::ConvParam param_multi_n = {
        2, 3, 4, 64, 64, {3, 3}, {64, 64}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    this->Init(param_multi_n);

    bool multi_n_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<2>>();
    EXPECT_TRUE(multi_n_supported);

    bool multi_n_incompatible =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<2>>();
    EXPECT_FALSE(multi_n_incompatible);

    // Check stride != 1
    ck::utils::conv::ConvParam param_strid2 = {
        2, 3, 1, 64, 64, {3, 3}, {64, 64}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};
    this->Init(param_strid2);

    bool strid2_incompatible =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<2>>();
    EXPECT_FALSE(strid2_incompatible);

    // Check pad != 1
    ck::utils::conv::ConvParam param_invalid_pad = {
        2, 3, 1, 64, 64, {3, 3}, {64, 64}, {1, 1}, {1, 1}, {2, 2}, {2, 2}};
    this->Init(param_invalid_pad);
    bool invalid_pad = this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1Pad0,
                                          ck::half_t,
                                          ck::half_t,
                                          float,
                                          PackedLayout<2>>();
    EXPECT_FALSE(invalid_pad);

    // Check dilation != 1
    ck::utils::conv::ConvParam param_invalid_dilation = {
        2, 3, 1, 64, 64, {3, 3}, {64, 64}, {1, 1}, {2, 2}, {1, 1}, {1, 1}};
    this->Init(param_invalid_dilation);
    bool invalid_dilation =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<2>>();
    EXPECT_FALSE(invalid_dilation);

    // check Strided layout
    this->Init(param_multi_n, 1);
    bool strided_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           StridedLayout<2>>();
    EXPECT_TRUE(strided_supported);

    // check GC Packed layout
    this->Init(param_multi_n, 2);
    bool gc_packed_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           GCPackedLayout<2>>();
    EXPECT_TRUE(gc_packed_supported);

    // check filter size mismatch
    const ck::utils::conv::ConvParam param_filter_mismatch = {
        2, 3, 1, 64, 64, {1, 1}, {64, 64}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    this->Init(param_filter_mismatch);

    bool filter_mismatch =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<2>>();
    EXPECT_FALSE(filter_mismatch);

    // check size mismatch
    const ck::utils::conv::ConvParam param_size_mismatch = {
        2, 3, 1, 64, 64, {3, 3}, {128, 128}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    this->Init(param_size_mismatch);
    auto in_size_mismatch = this->in_g_n_c_wis_desc;
    this->Init(param);
    this->in_g_n_c_wis_desc = in_size_mismatch;
    bool size_mismatch = this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1Pad0,
                                            ck::half_t,
                                            ck::half_t,
                                            float,
                                            PackedLayout<2>>();
    EXPECT_FALSE(size_mismatch);
}

TEST_F(TestGroupedConv2DFwdWcnnFilter3Dilation2, 2DFilter3Dilation2)
{
    // Initialize parameter with packed layout.
    const ck::utils::conv::ConvParam param = {
        2, 3, 1, 64, 64, {3, 3}, {64, 64}, {1, 1}, {2, 2}, {2, 2}, {2, 2}};
    this->Init(param);

    // Packed layout
    bool packed_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<2>>();
    EXPECT_TRUE(packed_supported);

    // Check pad != 2
    ck::utils::conv::ConvParam param_invalid_pad = {
        2, 3, 1, 64, 64, {3, 3}, {64, 64}, {1, 1}, {2, 2}, {1, 1}, {2, 2}};
    this->Init(param_invalid_pad);
    bool invalid_pad = this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1Pad0,
                                          ck::half_t,
                                          ck::half_t,
                                          float,
                                          PackedLayout<2>>();
    EXPECT_FALSE(invalid_pad);
}

TEST_F(TestGroupedConv2DFwdWcnnFilter2, 2DFilter2Stride2)
{
    // Initialize parameter with packed layout.
    const ck::utils::conv::ConvParam param = {
        2, 3, 1, 64, 64, {2, 2}, {64, 64}, {2, 2}, {1, 1}, {0, 0}, {0, 0}};
    this->Init(param);

    // Packed aligned layout
    bool packed_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter2x2Stride2Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<2>>();
    EXPECT_TRUE(packed_supported);

    // odd HW layout compatible with aligned layout
    bool odd_hw_compatible =
        this->template Run<ConvolutionForwardSpecialization::Filter2x2Stride2OddHWPad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           StridedLayout<2>>();
    EXPECT_TRUE(odd_hw_compatible);

    // odd hw with right pad
    const ck::utils::conv::ConvParam param_odd_hw = {
        2, 3, 4, 64, 64, {2, 2}, {65, 65}, {2, 2}, {1, 1}, {0, 0}, {1, 1}};
    this->Init(param_odd_hw);
    bool odd_hw_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter2x2Stride2OddHWPad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           StridedLayout<2>>();
    EXPECT_TRUE(odd_hw_supported);

    // hw incompatible with conv spec
    const ck::utils::conv::ConvParam param_odd_hw_incompatible = {
        2, 3, 1, 64, 64, {2, 2}, {67, 67}, {2, 2}, {1, 1}, {0, 0}, {0, 0}};
    this->Init(param_odd_hw_incompatible);
    bool odd_hw_incompatible =
        this->template Run<ConvolutionForwardSpecialization::Filter2x2Stride2Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<2>>();
    EXPECT_FALSE(odd_hw_incompatible);

    // Check stride != 2
    const ck::utils::conv::ConvParam param_invalid_stride = {
        2, 3, 1, 64, 64, {2, 2}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    this->Init(param_invalid_stride);

    bool invalid_stride = this->template Run<ConvolutionForwardSpecialization::Filter2x2Stride2Pad0,
                                             ck::half_t,
                                             ck::half_t,
                                             float,
                                             PackedLayout<2>>();
    EXPECT_FALSE(invalid_stride);

    // Check pad != 0
    ck::utils::conv::ConvParam param_invalid_pad = {
        2, 3, 1, 64, 64, {2, 2}, {64, 64}, {2, 2}, {1, 1}, {0, 0}, {1, 1}};
    this->Init(param_invalid_pad);
    bool invalid_pad = this->template Run<ConvolutionForwardSpecialization::Filter2x2Stride2Pad0,
                                          ck::half_t,
                                          ck::half_t,
                                          float,
                                          PackedLayout<2>>();
    EXPECT_FALSE(invalid_pad);

    // check size mismatch
    const ck::utils::conv::ConvParam param_size_mismatch = {
        2, 3, 1, 64, 64, {2, 2}, {128, 128}, {2, 2}, {1, 1}, {0, 0}, {0, 0}};
    this->Init(param_size_mismatch);
    auto in_size_mismatch = this->in_g_n_c_wis_desc;
    this->Init(param);
    this->in_g_n_c_wis_desc = in_size_mismatch;
    bool size_mismatch = this->template Run<ConvolutionForwardSpecialization::Filter2x2Stride2Pad0,
                                            ck::half_t,
                                            ck::half_t,
                                            float,
                                            PackedLayout<2>>();
    EXPECT_FALSE(size_mismatch);
}

TEST_F(TestGroupedConv1DFwdWcnnFilter1, 1DFilter1)
{
    // Initialize parameter with packed layout.
    const ck::utils::conv::ConvParam param = {1, 1, 1, 64, 64, {1}, {64}, {1}, {1}, {0}, {0}};
    this->Init(param);

    // Packed layout
    bool packed_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<1>>();
    EXPECT_TRUE(packed_supported);

    // Strided layout compatible with packed layout
    bool strided_compatible =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           StridedLayout<1>>();
    EXPECT_TRUE(strided_compatible);

    // GCPacked layout incompatible with packed layout
    bool gc_packed_incompatible =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           GCPackedLayout<1>>();
    EXPECT_FALSE(gc_packed_incompatible);

    // Check stride != 1
    ck::utils::conv::ConvParam param_strid2 = {1, 1, 1, 64, 64, {1}, {64}, {2}, {1}, {0}, {0}};
    this->Init(param_strid2);
    bool is_supported_strid2 = this->template Run<ConvolutionForwardSpecialization::Filter1x1Pad0,
                                                  ck::half_t,
                                                  ck::half_t,
                                                  float,
                                                  PackedLayout<1>>();
    EXPECT_TRUE(is_supported_strid2);

    bool strid2_incompatible =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<1>>();
    EXPECT_FALSE(strid2_incompatible);

    // Check pad
    ck::utils::conv::ConvParam param_invalid_pad_left = {
        1, 1, 1, 64, 64, {1}, {64}, {2}, {1}, {1}, {0}};
    this->Init(param_invalid_pad_left);
    bool invalid_pad_left =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<1>>();
    EXPECT_FALSE(invalid_pad_left);

    ck::utils::conv::ConvParam param_invalid_pad_right = {
        1, 1, 1, 64, 64, {1}, {64}, {2}, {1}, {0}, {1}};
    this->Init(param_invalid_pad_right);
    bool invalid_pad_right =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<1>>();
    EXPECT_FALSE(invalid_pad_right);

    // check Strided layout
    this->Init(param, 1);
    bool strided_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           StridedLayout<1>>();
    EXPECT_TRUE(strided_supported);

    bool packed_incompatible =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<1>>();
    EXPECT_FALSE(packed_incompatible);

    bool gc_packed_incompatible2 =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           GCPackedLayout<1>>();
    EXPECT_FALSE(gc_packed_incompatible2);

    // check GC Packed layout
    this->Init(param, 2);
    bool gc_packed_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           GCPackedLayout<1>>();
    EXPECT_TRUE(gc_packed_supported);

    bool strided_compatible2 =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           StridedLayout<1>>();
    EXPECT_TRUE(strided_compatible2);

    bool packed_incompatible2 =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<1>>();
    EXPECT_FALSE(packed_incompatible2);

    // check filter size mismatch
    const ck::utils::conv::ConvParam param_filter_mismatch = {
        1, 1, 1, 64, 64, {3}, {64}, {1}, {1}, {0}, {0}};
    this->Init(param_filter_mismatch);

    bool filter_mismatch =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<1>>();
    EXPECT_FALSE(filter_mismatch);

    // check size mismatch
    const ck::utils::conv::ConvParam param_size_mismatch = {
        1, 1, 1, 64, 64, {1}, {128}, {1}, {1}, {0}, {0}};
    this->Init(param_size_mismatch);
    auto in_size_mismatch = this->in_g_n_c_wis_desc;
    this->Init(param);
    this->in_g_n_c_wis_desc = in_size_mismatch;
    bool size_mismatch = this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                                            ck::half_t,
                                            ck::half_t,
                                            float,
                                            PackedLayout<1>>();
    EXPECT_FALSE(size_mismatch);
}

TEST_F(TestGroupedConv1DFwdWcnnFilter1, 1DFilterN)
{
    // Initialize parameter with packed layout.
    const ck::utils::conv::ConvParam param = {1, 1, 1, 64, 64, {4}, {64}, {1}, {1}, {0}, {0}};
    this->Init(param);

    // Packed layout
    bool packed_supported = this->template Run<ConvolutionForwardSpecialization::Default,
                                               ck::half_t,
                                               ck::half_t,
                                               float,
                                               PackedLayout<1>>();
    EXPECT_TRUE(packed_supported);

    // Check stride != 1
    ck::utils::conv::ConvParam param_strid2 = {1, 1, 1, 64, 64, {4}, {64}, {2}, {1}, {0}, {0}};
    this->Init(param_strid2);
    bool is_supported_strid2 = this->template Run<ConvolutionForwardSpecialization::Default,
                                                  ck::half_t,
                                                  ck::half_t,
                                                  float,
                                                  PackedLayout<1>>();
    EXPECT_TRUE(is_supported_strid2);

    // Check dilation != 1
    ck::utils::conv::ConvParam param_dilation2 = {1, 1, 1, 64, 64, {4}, {64}, {1}, {2}, {0}, {0}};
    this->Init(param_dilation2);
    bool is_supported_dilation2 = this->template Run<ConvolutionForwardSpecialization::Default,
                                                     ck::half_t,
                                                     ck::half_t,
                                                     float,
                                                     PackedLayout<1>>();
    EXPECT_TRUE(is_supported_dilation2);

    // Check pad != 0
    ck::utils::conv::ConvParam param_pad = {1, 1, 1, 64, 64, {5}, {64}, {1}, {1}, {2}, {2}};
    this->Init(param_pad);
    bool is_supported_pad = this->template Run<ConvolutionForwardSpecialization::Default,
                                               ck::half_t,
                                               ck::half_t,
                                               float,
                                               PackedLayout<1>>();
    EXPECT_TRUE(is_supported_pad);

    // Check stride != 1, dilation != 1, pad != 0
    ck::utils::conv::ConvParam param_any = {1, 1, 1, 64, 64, {5}, {64}, {2}, {2}, {2}, {2}};
    this->Init(param_any);
    bool is_supported_any = this->template Run<ConvolutionForwardSpecialization::Default,
                                               ck::half_t,
                                               ck::half_t,
                                               float,
                                               PackedLayout<1>>();
    EXPECT_TRUE(is_supported_any);
}

TEST_F(TestGroupedConv3DFwdWcnnFilter1, 3DFilter1)
{
    // Initialize parameter with packed layout.
    const ck::utils::conv::ConvParam param = {
        3, 1, 1, 64, 64, {1, 1, 1}, {64, 64, 64}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    this->Init(param);

    // Packed layout
    bool packed_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<3>>();
    EXPECT_TRUE(packed_supported);

    // Strided layout compatible with packed layout
    bool strided_compatible =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           StridedLayout<3>>();
    EXPECT_TRUE(strided_compatible);

    // GCPacked layout incompatible with packed layout
    bool gc_packed_incompatible =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           GCPackedLayout<3>>();
    EXPECT_FALSE(gc_packed_incompatible);

    // Check stride != 1
    ck::utils::conv::ConvParam param_strid2 = {
        3, 1, 1, 64, 64, {1, 1, 1}, {64, 64, 64}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    this->Init(param_strid2);
    bool is_supported_strid2 = this->template Run<ConvolutionForwardSpecialization::Filter1x1Pad0,
                                                  ck::half_t,
                                                  ck::half_t,
                                                  float,
                                                  PackedLayout<3>>();
    EXPECT_TRUE(is_supported_strid2);

    bool strid2_incompatible =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<3>>();
    EXPECT_FALSE(strid2_incompatible);

    // Check pad
    ck::utils::conv::ConvParam param_invalid_pad_left = {
        3, 1, 1, 64, 64, {1, 1, 1}, {64, 64, 64}, {1, 1, 1}, {1, 1, 1}, {0, 0, 1}, {0, 0, 0}};
    this->Init(param_invalid_pad_left);
    bool invalid_pad_left =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<3>>();
    EXPECT_FALSE(invalid_pad_left);

    ck::utils::conv::ConvParam param_invalid_pad_right = {
        3, 1, 1, 64, 64, {1, 1, 1}, {64, 64, 64}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 1}};
    this->Init(param_invalid_pad_right);
    bool invalid_pad_right =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<3>>();
    EXPECT_FALSE(invalid_pad_right);

    // Check default 1x1 layout
    ck::utils::conv::ConvParam param_z2 = {
        3, 1, 1, 64, 64, {3, 1, 1}, {64, 64, 64}, {2, 1, 1}, {2, 1, 1}, {1, 0, 0}, {1, 0, 0}};
    this->Init(param_z2);
    bool is_supported_z2 = this->template Run<ConvolutionForwardSpecialization::Filter1x1Pad0,
                                              ck::half_t,
                                              ck::half_t,
                                              float,
                                              PackedLayout<3>>();
    EXPECT_TRUE(is_supported_z2);

    // Check Strided layout
    this->Init(param, 1);
    bool strided_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           StridedLayout<3>>();
    EXPECT_TRUE(strided_supported);

    bool packed_incompatible =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<3>>();
    EXPECT_FALSE(packed_incompatible);

    bool gc_packed_incompatible2 =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           GCPackedLayout<3>>();
    EXPECT_FALSE(gc_packed_incompatible2);

    // Check GC Packed layout
    this->Init(param, 2);
    bool gc_packed_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           GCPackedLayout<3>>();
    EXPECT_TRUE(gc_packed_supported);

    bool strided_compatible2 =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           StridedLayout<3>>();
    EXPECT_TRUE(strided_compatible2);

    bool packed_incompatible2 =
        this->template Run<ConvolutionForwardSpecialization::Filter1x1Stride1Pad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<3>>();
    EXPECT_FALSE(packed_incompatible2);

    // Check filter size mismatch
    const ck::utils::conv::ConvParam param_filter_mismatch = {
        3, 1, 1, 64, 64, {1, 3, 3}, {64, 64, 64}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    this->Init(param_filter_mismatch);

    bool filter_mismatch = this->template Run<ConvolutionForwardSpecialization::Filter1x1Pad0,
                                              ck::half_t,
                                              ck::half_t,
                                              float,
                                              PackedLayout<3>>();
    EXPECT_FALSE(filter_mismatch);

    // Check size mismatch
    const ck::utils::conv::ConvParam param_size_mismatch = {
        3, 1, 1, 64, 64, {1, 1, 1}, {128, 64, 64}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    this->Init(param_size_mismatch);
    auto in_size_mismatch = this->in_g_n_c_wis_desc;
    this->Init(param);
    this->in_g_n_c_wis_desc = in_size_mismatch;
    bool size_mismatch      = this->template Run<ConvolutionForwardSpecialization::Filter1x1Pad0,
                                                 ck::half_t,
                                                 ck::half_t,
                                                 float,
                                                 PackedLayout<3>>();
    EXPECT_FALSE(size_mismatch);
}

TEST_F(TestGroupedConv3DFwdWcnnFilter3, 3DFilter3)
{
    // Initialize parameter with packed layout.
    const ck::utils::conv::ConvParam param = {
        3, 3, 1, 64, 64, {3, 3, 3}, {64, 64, 64}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    this->Init(param);

    // Packed layout
    bool packed_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<3>>();
    EXPECT_TRUE(packed_supported);

    // Strided layout compatible with packed layout
    bool strided_compatible =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           StridedLayout<3>>();
    EXPECT_TRUE(strided_compatible);

    // GCPacked layout incompatible with packed layout
    bool gc_packed_incompatible =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           GCPackedLayout<3>>();
    EXPECT_FALSE(gc_packed_incompatible);

    // multi-n, filter_z != 3, stride_z != 1, dilation_z != 1, pad != 1
    const ck::utils::conv::ConvParam param_multi_n = {
        3, 3, 4, 64, 64, {5, 3, 3}, {64, 64, 64}, {2, 1, 1}, {2, 1, 1}, {2, 1, 1}, {2, 1, 1}};
    this->Init(param_multi_n);

    bool multi_n_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<3>>();
    EXPECT_TRUE(multi_n_supported);

    // Check dilation_xy != 1
    ck::utils::conv::ConvParam param_invalid_dilation = {
        3, 3, 1, 64, 64, {3, 3, 3}, {64, 64, 64}, {1, 1, 1}, {1, 2, 2}, {1, 1, 1}, {1, 1, 1}};
    this->Init(param_invalid_dilation);
    bool invalid_dilation =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<3>>();
    EXPECT_FALSE(invalid_dilation);

    // check Strided layout
    this->Init(param_multi_n, 1);
    bool strided_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           StridedLayout<3>>();
    EXPECT_TRUE(strided_supported);

    // check GC Packed layout
    this->Init(param_multi_n, 2);
    bool gc_packed_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           GCPackedLayout<3>>();
    EXPECT_TRUE(gc_packed_supported);

    // check filter size mismatch
    const ck::utils::conv::ConvParam param_filter_mismatch = {
        3, 3, 1, 64, 64, {1, 1, 1}, {64, 64, 64}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    this->Init(param_filter_mismatch);

    bool filter_mismatch =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<3>>();
    EXPECT_FALSE(filter_mismatch);

    // check size mismatch
    const ck::utils::conv::ConvParam param_size_mismatch = {
        3, 3, 1, 64, 64, {3, 3, 3}, {128, 64, 64}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    this->Init(param_size_mismatch);
    auto in_size_mismatch = this->in_g_n_c_wis_desc;
    this->Init(param);
    this->in_g_n_c_wis_desc = in_size_mismatch;
    bool size_mismatch =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<3>>();
    EXPECT_FALSE(size_mismatch);
}

TEST_F(TestGroupedConv3DFwdWcnnFilter3Dilation2, 3DFilter3Dilation2)
{
    // Initialize parameter with packed layout.
    const ck::utils::conv::ConvParam param = {
        3, 3, 4, 64, 64, {5, 3, 3}, {64, 64, 64}, {2, 1, 1}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}};
    this->Init(param);

    // Packed layout
    bool packed_supported =
        this->template Run<ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0,
                           ck::half_t,
                           ck::half_t,
                           float,
                           PackedLayout<3>>();
    EXPECT_TRUE(packed_supported);
}
