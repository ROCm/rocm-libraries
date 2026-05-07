// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "common_wcnn.hpp"

#define DEFAULT_H_PERWAVE 8
#define DEFAULT_W_PERWAVE 8
#define DEFAULT_H_PERBLOCK 16
#define DEFAULT_W_PERBLOCK 16
#define DEFAULT_C_PERBLOCK 16
#define DEFAULT_K_PERBLOCK 16
#define DEFAULT_BLOCKSIZE 128
#define DEFAULT_WAVEGROUP_BLOCKSIZE 512

template <typename ResidualLay, typename ScaleLay>
struct FmaLayoutSetting
{
    using ResidualLayout = ResidualLay;
    using ScaleLayout    = ScaleLay;
};

template <ck::index_t NDimSpatial>
struct FmaLayoutSettingSelector;

namespace ctl = ck::tensor_layout::convolution;

template <>
struct FmaLayoutSettingSelector<1> final : FmaLayoutSetting<ctl::G_NW_K, ctl::G_NW_K>
{
};

template <>
struct FmaLayoutSettingSelector<2> final : FmaLayoutSetting<ctl::G_NHW_K, ctl::G_NHW_K>
{
};

template <>
struct FmaLayoutSettingSelector<3> final : FmaLayoutSetting<ctl::G_NDHW_K, ctl::G_NDHW_K>
{
};

template <ck::index_t NDimSpatial>
using ResidualLayout = typename FmaLayoutSettingSelector<NDimSpatial>::ResidualLayout;

template <ck::index_t NDimSpatial>
using ScaleLayout = typename FmaLayoutSettingSelector<NDimSpatial>::ScaleLayout;

template <typename InDataType,
          typename WeiDataType,
          typename ResidualDataType,
          typename GPUAccType,
          typename EDataType,
          ShapeType Shape,
          int LdsMode,
          bool EnableWaveGroup,
          int FmaMode,
          ck::index_t ActiveFun,
          bool CvtToTensor>
bool run_test()
{
    constexpr FilterType Filter = Filter_1X1;
    constexpr bool Dilation     = false;
    constexpr ck::index_t FilterSize =
        (Filter == Filter_1X1) ? 1 : ((Filter == Filter_2X2) ? 2 : 3);
    constexpr ck::index_t HPerWcnn = (Shape == Shape_8X4) ? 8 : 4;
    constexpr ck::index_t WPerWcnn = (Shape == Shape_4X2) ? 2 : 4;

    const ck::index_t Width          = config.w;
    const ck::index_t Height         = config.h;
    const ck::index_t InputChannels  = config.c;
    const ck::index_t OutputChannels = config.k;
    constexpr ck::index_t BlockSize =
        EnableWaveGroup ? DEFAULT_WAVEGROUP_BLOCKSIZE : DEFAULT_BLOCKSIZE;

    using GPUOutType = typename std::conditional<CvtToTensor, EDataType, GPUAccType>::type;
    using CPUOutType = typename std::conditional<CvtToTensor, EDataType, GPUAccType>::type;

    // on gfx13, the wavegroup count is fixed to 4. so, HPerWave/WPerWave is fixed too.
    constexpr ck::index_t HPerWave = EnableWaveGroup ? DEFAULT_H_PERBLOCK / 2 : DEFAULT_H_PERWAVE;
    constexpr ck::index_t WPerWave = EnableWaveGroup ? DEFAULT_W_PERBLOCK / 2 : DEFAULT_W_PERWAVE;

    constexpr ck::index_t DefaultHScale =
        (Filter == Filter_2X2 && Shape == Shape_8X4 && HPerWave < 16) ? 2 : 1;

    constexpr ck::index_t HPerBlock = DEFAULT_H_PERBLOCK * DefaultHScale;
    constexpr ck::index_t WPerBlock = DEFAULT_W_PERBLOCK;
    constexpr ck::index_t CPerBlock = DEFAULT_C_PERBLOCK;
    constexpr ck::index_t KPerBlock = DEFAULT_K_PERBLOCK;
    constexpr ck::index_t HRepeat   = HPerWave * DefaultHScale / HPerWcnn;
    constexpr ck::index_t WRepeat   = WPerWave / WPerWcnn;

    constexpr ck::index_t n_dim       = 2;
    constexpr ck::index_t group_count = 1;
    constexpr ck::index_t n_batch     = 1;
    const ck::index_t n_out_channels  = OutputChannels;
    const ck::index_t n_in_channels   = InputChannels;

    constexpr ck::index_t DilationSize = Dilation ? 2 : 1;

    const std::vector<ck::index_t> filters_1x1{1, 1};
    const std::vector<ck::index_t> filters_2x2{2, 2};
    const std::vector<ck::index_t> filters_3x3{3, 3};
    const std::vector<ck::index_t> dilations_1{1, 1};
    const std::vector<ck::index_t> dilations_2{2, 2};
    const std::vector<ck::index_t> pads_0{0, 0};
    const std::vector<ck::index_t> pads_1{1, 1};
    const std::vector<ck::index_t> pads_2{2, 2};

    const std::vector<ck::index_t>& filters_len = (Filter == Filter_1X1)   ? filters_1x1
                                                  : (Filter == Filter_2X2) ? filters_2x2
                                                                           : filters_3x3;
    const std::vector<ck::index_t> input_len    = {Height, Width};
    const std::vector<ck::index_t> strides_1{1, 1};
    const std::vector<ck::index_t> strides_2{2, 2};
    const std::vector<ck::index_t>& strides   = (Filter == Filter_2X2) ? strides_2 : strides_1;
    const std::vector<ck::index_t>& dilations = Dilation ? dilations_2 : dilations_1;
    const std::vector<ck::index_t>& left_pads =
        (Filter == Filter_3X3) ? (Dilation ? pads_2 : pads_1) : pads_0;
    const std::vector<ck::index_t>& right_pads =
        (Filter == Filter_3X3) ? (Dilation ? pads_2 : pads_1) : pads_0;

    constexpr bool InEnableLds  = LdsMode & 1 ? true : false;
    constexpr bool WeiEnableLds = LdsMode & 2 ? true : false;
    constexpr bool AccEnableLds = LdsMode & 4 ? true : false;
    constexpr bool EnableAsync  = LdsMode & 8 ? true : false;
    constexpr bool DsEnableLds  = LdsMode & 16 ? true : false;

    ck::utils::conv::ConvParam conv_param{n_dim,
                                          group_count,
                                          n_batch,
                                          n_out_channels,
                                          n_in_channels,
                                          filters_len,
                                          input_len,
                                          strides,
                                          dilations,
                                          left_pads,
                                          right_pads};

    constexpr auto NDimSpatial = ck::Number<n_dim>{};
    const auto in_element_op   = InElementOp{};
    const auto wei_element_op  = WeiElementOp{};
    const auto pass_through_op = PassThrough{};
    // const ck::index_t cvtTensorScale = 1; // Scale=1 to update

    auto in_layout  = ctl::GNHWC{};
    auto wei_layout = ctl::GKYXC{};
    auto out_layout = ctl::GNHWK{};
    using InLayout  = decltype(in_layout);
    using WeiLayout = decltype(wei_layout);
    using OutLayout = decltype(out_layout);

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    const auto scale_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    const auto residual_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<ResidualDataType> residual(residual_g_n_k_wos_desc);
    Tensor<GPUAccType> scale(scale_g_n_k_wos_desc);
    Tensor<CPUOutType> out_host(out_g_n_k_wos_desc);
    Tensor<GPUOutType> out_device(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "residual: " << residual.mDesc << std::endl;
    std::cout << "scale: " << scale.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;
    float scale_value = 2.0f;
    switch(config.init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        residual.GenerateTensorValue(GeneratorTensor_2<ResidualDataType>{-10, 10});
        if constexpr(FmaMode == 1)
        {
            scale.GenerateTensorValue(GeneratorTensor_2<GPUAccType>{-3, 3});
        }
        else
        {
            scale_value = 2.0f;
        }
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        residual.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        if constexpr(FmaMode == 1)
        {
            scale.GenerateTensorValue(GeneratorTensor_2<GPUAccType>{-5, 5});
        }
        else
        {
            scale_value = 3.0f;
        }
        break;
    }

    constexpr bool Clamp =
        ck::is_same_v<GPUOutType, int8_t> || ck::is_same_v<GPUOutType, ck::f8_ocp_t>;

    const auto out_element_op = [&]() {
        if constexpr(FmaMode == 0)
        {
            if constexpr(ActiveFun == 0)
            {
                if constexpr(Clamp)
                {
                    return ScaleAddClampRev{scale_value};
                }
                else
                {
                    return ScaleAddRev{scale_value};
                }
            }
            else if constexpr(ActiveFun == 1)
            {
                if constexpr(Clamp)
                {
                    return ScaleAddClampRevRelu{scale_value};
                }
                else
                {
                    return ScaleAddRevRelu{scale_value};
                }
            }
        }
        else if constexpr(FmaMode == 1)
        {
            if constexpr(ActiveFun == 0)
            {
                if constexpr(Clamp)
                {
                    return MultiplyAddClampRev{};
                }
                else
                {
                    return MultiplyAddRev{};
                }
            }
            else if constexpr(ActiveFun == 1)
            {
                if constexpr(Clamp)
                {
                    return MultiplyAddClampRevRelu{};
                }
                else
                {
                    return MultiplyAddRevRelu{};
                }
            }
        }
        else
        {
            static_assert(0);
        }
    }();

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> residual_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> residual_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial + 3> scale_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> scale_g_n_k_wos_strides{};
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
    copy(residual_g_n_k_wos_desc.GetLengths(), residual_g_n_k_wos_lengths);
    copy(residual_g_n_k_wos_desc.GetStrides(), residual_g_n_k_wos_strides);
    copy(scale_g_n_k_wos_desc.GetLengths(), scale_g_n_k_wos_lengths);
    copy(scale_g_n_k_wos_desc.GetStrides(), scale_g_n_k_wos_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    Tensor<GPUAccType> c_host(out_g_n_k_wos_desc);
    constexpr ck::index_t CPerWcnn_Bhalf = (Shape == Shape_4X2) ? 8 : 4;

    constexpr ck::long_index_t Acc_Convert_Interval =
        std::is_same<GPUAccType, ck::bhalf_t>::value ? CPerWcnn_Bhalf : CPerBlock;

    using RefConvType =
        std::conditional_t<(std::is_same<GPUAccType, ck::bhalf_t>::value ||
                            std::is_same<GPUAccType, ck::half_t>::value),
                           ck::tensor_operation::host::ReferenceConvFwd_GFX13<NDimSpatial,
                                                                              InDataType,
                                                                              WeiDataType,
                                                                              GPUAccType,
                                                                              InElementOp,
                                                                              WeiElementOp,
                                                                              PassThrough>,
                           ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                        InDataType,
                                                                        WeiDataType,
                                                                        GPUAccType,
                                                                        InElementOp,
                                                                        WeiElementOp,
                                                                        PassThrough>>;
    RefConvType ref_conv{};

    auto ref_invoker  = ref_conv.MakeInvoker();
    auto ref_argument = ref_conv.MakeArgument(in,
                                              wei,
                                              c_host,
                                              conv_param.conv_filter_strides_,
                                              conv_param.conv_filter_dilations_,
                                              conv_param.input_left_pads_,
                                              conv_param.input_right_pads_,
                                              in_element_op,
                                              wei_element_op,
                                              pass_through_op,
                                              {},
                                              {},
                                              {},
                                              Acc_Convert_Interval);

    if(config.do_verification)
    {
        ref_invoker.Run(ref_argument);
        out_host.ForEach([&](auto&, auto idx) {
            if constexpr(FmaMode == 0)
            {
                out_element_op(out_host(idx), c_host(idx), residual(idx));
            }
            else if constexpr(FmaMode == 1)
            {
                out_element_op(out_host(idx), c_host(idx), scale(idx), residual(idx));
            }
        });
    }

    dump_tensor(in, "Input");
    dump_tensor(wei, "Weight");
    dump_tensor(scale, "Scale");
    dump_tensor(residual, "Residual");
    dump_tensor(out_host, "Output");

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem scale_device_buf(sizeof(GPUAccType) * scale.mDesc.GetElementSpaceSize());
    DeviceMem residual_device_buf(sizeof(ResidualDataType) * residual.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(GPUOutType) * out_device.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());
    scale_device_buf.ToDevice(scale.mData.data());
    residual_device_buf.ToDevice(residual.mData.data());

    // do Conv
    static constexpr auto ConvSpec =
        FilterSize == 1
            ? ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0
        : FilterSize == 2
            ? ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter2x2Stride2Pad0
            : ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter3x3Stride1Pad0;

    constexpr ck::index_t InBlockTransferScalarPerVector = sizeof(uint32_t) / sizeof(InDataType);
    constexpr ck::index_t Cluster_In_C = CPerBlock / InBlockTransferScalarPerVector;
    constexpr ck::index_t Cluster_In_W = 4;

    constexpr ck::index_t ActiveBlockSize = EnableWaveGroup ? 128 : BlockSize;
    constexpr ck::index_t Cluster_In_H    = ActiveBlockSize / Cluster_In_C / Cluster_In_W;
    using InBlockTransferThreadClusterLengths =
        ck::Sequence<Cluster_In_H, Cluster_In_W, Cluster_In_C>;
    constexpr bool InBlockLdsAddExtraM = true;

    constexpr ck::index_t WeiBlockTransferScalarPerVector = sizeof(uint32_t) / sizeof(WeiDataType);
    constexpr ck::index_t Cluster_Wei_C        = CPerBlock / WeiBlockTransferScalarPerVector;
    constexpr ck::index_t Cluster_Wei_K        = (ActiveBlockSize / Cluster_Wei_C) > KPerBlock
                                                     ? KPerBlock
                                                     : (ActiveBlockSize / Cluster_Wei_C);
    using WeiBlockTransferThreadClusterLengths = ck::Sequence<Cluster_Wei_K, 1, Cluster_Wei_C>;
    constexpr ck::index_t WeiBlockLdsAddExtraM = true;

    constexpr ck::index_t AccBlockTransferScalarPerVector = sizeof(uint32_t) / sizeof(GPUOutType);
    constexpr ck::index_t Cluster_Acc_K = KPerBlock / AccBlockTransferScalarPerVector;
    constexpr ck::index_t Cluster_Acc_W = 4;
    constexpr ck::index_t Cluster_Acc_H = ActiveBlockSize / Cluster_Acc_K / Cluster_Acc_W;
    using AccBlockTransferClusterLengths =
        ck::Sequence<Cluster_Acc_H, Cluster_Acc_W, Cluster_Acc_K>;

    constexpr ck::index_t ScaleBlockTransferScalarPerVector = sizeof(uint32_t) / sizeof(GPUAccType);
    constexpr ck::index_t Cluster_Scale_K = KPerBlock / ScaleBlockTransferScalarPerVector;
    constexpr ck::index_t Cluster_Scale_W = 4;
    constexpr ck::index_t Cluster_Scale_H = ActiveBlockSize / Cluster_Scale_K / Cluster_Scale_W;

    constexpr ck::index_t ResidualBlockTransferScalarPerVector =
        sizeof(uint32_t) / sizeof(ResidualDataType);
    constexpr ck::index_t Cluster_Residual_K = KPerBlock / ResidualBlockTransferScalarPerVector;
    constexpr ck::index_t Cluster_Residual_W = 4;
    constexpr ck::index_t Cluster_Residual_H =
        ActiveBlockSize / Cluster_Residual_K / Cluster_Residual_W;

    constexpr ck::index_t DsBlockLdsAddExtraM = true;

    float avg_time    = 0;
    using AccDataType = GPUAccType;

    constexpr auto DsDataTypeInfo = [&]() {
        if constexpr(FmaMode == 0)
        {
            using DsDataType   = ck::Tuple<ResidualDataType>;
            using DsDataLayout = ck::Tuple<ResidualLayout<NDimSpatial>>;
            using DsBlockTransferThreadClusterLengths =
                ck::Tuple<ck::Sequence<Cluster_Residual_H, Cluster_Residual_W, Cluster_Residual_K>>;
            using DsBlockTransferScalarPerVector =
                ck::Sequence<ResidualBlockTransferScalarPerVector>;
            return make_tuple(DsDataType{},
                              DsDataLayout{},
                              DsBlockTransferThreadClusterLengths{},
                              DsBlockTransferScalarPerVector{});
        }
        else if constexpr(FmaMode == 1)
        {
            using DsDataType   = ck::Tuple<GPUAccType, ResidualDataType>;
            using DsDataLayout = ck::Tuple<ScaleLayout<NDimSpatial>, ResidualLayout<NDimSpatial>>;
            using DsBlockTransferThreadClusterLengths =
                ck::Tuple<ck::Sequence<Cluster_Scale_H, Cluster_Scale_W, Cluster_Scale_K>,
                          ck::Sequence<Cluster_Residual_H, Cluster_Residual_W, Cluster_Residual_K>>;
            using DsBlockTransferScalarPerVector =
                ck::Sequence<ScaleBlockTransferScalarPerVector,
                             ResidualBlockTransferScalarPerVector>;
            return make_tuple(DsDataType{},
                              DsDataLayout{},
                              DsBlockTransferThreadClusterLengths{},
                              DsBlockTransferScalarPerVector{});
        }
    }();

    using DsDataType   = ck::remove_cvref_t<ck::tuple_element_t<0, decltype(DsDataTypeInfo)>>;
    using DsDataLayout = ck::remove_cvref_t<ck::tuple_element_t<1, decltype(DsDataTypeInfo)>>;
    using DsBlockTransferThreadClusterLengths =
        ck::remove_cvref_t<ck::tuple_element_t<2, decltype(DsDataTypeInfo)>>;
    using DsBlockTransferScalarPerVector =
        ck::remove_cvref_t<ck::tuple_element_t<3, decltype(DsDataTypeInfo)>>;

    using DeviceConvFwdFmaInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Wcnn_CShuffle<
            NDimSpatial,
            InputLayout<NDimSpatial>,
            WeightLayout<NDimSpatial>,
            DsDataLayout,
            OutputLayout<NDimSpatial>,
            InDataType,
            WeiDataType,
            DsDataType,
            AccDataType,
            GPUOutType,
            InElementOp,
            WeiElementOp,
            decltype(out_element_op),
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
            InEnableLds,
            InBlockLdsAddExtraM,
            false, // InTileLoad
            WeiBlockTransferThreadClusterLengths,
            WeiBlockTransferScalarPerVector,
            WeiBlockTransferScalarPerVector,
            WeiEnableLds,
            WeiBlockLdsAddExtraM,
            false, // WeiTileLoad
            DsBlockTransferThreadClusterLengths,
            DsBlockTransferScalarPerVector,
            DsBlockTransferScalarPerVector,
            DsEnableLds,
            DsBlockLdsAddExtraM,
            AccBlockTransferClusterLengths,
            AccBlockTransferScalarPerVector,
            AccEnableLds,
            EnableAsync,
            EnableWaveGroup>; // wavegrouop

    std::array<const void*, FmaMode + 1> ds_bufs;
    std::array<std::array<ck::index_t, NDimSpatial + 3>, FmaMode + 1> ds_lengths;
    std::array<std::array<ck::index_t, NDimSpatial + 3>, FmaMode + 1> ds_strides;
    if constexpr(FmaMode == 0)
    {
        ds_bufs[0]    = residual_device_buf.GetDeviceBuffer();
        ds_lengths[0] = residual_g_n_k_wos_lengths;
        ds_strides[0] = residual_g_n_k_wos_strides;
    }
    else if constexpr(FmaMode == 1)
    {
        ds_bufs[0]    = scale_device_buf.GetDeviceBuffer();
        ds_bufs[1]    = residual_device_buf.GetDeviceBuffer();
        ds_lengths[0] = scale_g_n_k_wos_lengths;
        ds_lengths[1] = residual_g_n_k_wos_lengths;
        ds_strides[0] = scale_g_n_k_wos_strides;
        ds_strides[1] = residual_g_n_k_wos_strides;
    }
    auto conv     = DeviceConvFwdFmaInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(in_device_buf.GetDeviceBuffer(),
                                      wei_device_buf.GetDeviceBuffer(),
                                      ds_bufs,
                                      out_device_buf.GetDeviceBuffer(),
                                      a_g_n_c_wis_lengths,
                                      a_g_n_c_wis_strides,
                                      b_g_k_c_xs_lengths,
                                      b_g_k_c_xs_strides,
                                      ds_lengths,
                                      ds_strides,
                                      e_g_n_k_wos_lengths,
                                      e_g_n_k_wos_strides,
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads,
                                      InElementOp{},
                                      WeiElementOp{},
                                      out_element_op);

    if(!conv.IsSupportedArgument(argument))
    {
        std::cout << "wrong! device_conv with the specified compilation parameters does "
                     "not support this Conv problem"
                  << std::endl;
        return false;
    }
    avg_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

    out_device_buf.FromDevice(out_device.mData.data());
    dump_tensor(out_device, "Out_Device");

    std::cout << (EnableWaveGroup ? "grouped_conv_fwd_wcnn_fma_wavegroup<In/Wei:"
                                  : "grouped_conv_fwd_wcnn_fma<In/Wei:")
              << get_string<InDataType>() << ", Out:" << get_string<GPUAccType>() << ", "
              << get_string(Shape) << ", " << get_string(Filter) << ", Dilation:" << DilationSize
              << ", LdsMod:" << LdsMode << ", FmaMode:" << FmaMode << ", ActiveFun:" << ActiveFun
              << ", CvtToTensor:" << CvtToTensor << ", WaveGroup:" << EnableWaveGroup << " Size: { "
              << config.h << "x" << config.w << "x" << config.c << "x" << config.k
              << " }>: Status: ";

    if(config.time_kernel)
    {
        std::cout << "Execute Time: " << avg_time << " ";
    }

    if(config.do_verification)
    {
        bool ret = false;
        ret      = ck::utils::check_err(out_device,
                                   out_host,
                                   "Error: incorrect results!",
                                   get_rtol<GPUOutType>(),
                                   get_atol<GPUOutType>());

        if(ret)
        {
            std::cout << "Passed\n";
        }
        else
        {
            std::cout << "Failed\n";
        }

        return ret;
    }
    else
    {
        return true;
    }
}
