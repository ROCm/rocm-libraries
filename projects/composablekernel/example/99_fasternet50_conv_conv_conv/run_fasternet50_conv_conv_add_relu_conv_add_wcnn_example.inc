// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

template <typename BiasLay, typename ResidualLay>
struct LayoutSetting
{
    using BiasLayout     = BiasLay;
    using ResidualLayout = ResidualLay;
};

template <ck::index_t NDimSpatial>
struct LayoutSettingSelector;

template <>
struct LayoutSettingSelector<1> final : LayoutSetting<ctl::G_K, ctl::G_NW_K>
{
};

template <>
struct LayoutSettingSelector<2> final : LayoutSetting<ctl::G_K, ctl::G_NHW_K>
{
};

template <>
struct LayoutSettingSelector<3> final : LayoutSetting<ctl::G_K, ctl::G_NDHW_K>
{
};

template <ck::index_t NDimSpatial>
using BiasLayout = typename LayoutSettingSelector<NDimSpatial>::BiasLayout;

template <ck::index_t NDimSpatial>
using ScaleLayout = typename LayoutSettingSelector<NDimSpatial>::BiasLayout;

using ConvolutionForwardSpecialization =
    typename ck::tensor_operation::device::ConvolutionForwardSpecialization;

static constexpr ck::index_t HPerWcnn      = 4;
static constexpr ck::index_t WPerWcnn      = 4;
static constexpr ck::index_t HPerBlock     = 96;
static constexpr ck::index_t WPerBlock     = 16;
static constexpr bool EnableAsync          = false;
static constexpr bool EnableWaveGroup      = false;
static constexpr bool EnableSpatialCluster = false;

template <ck::index_t NDimSpatial>
using HostConvFwdInstance = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                         InUserDataType,
                                                                         WeiUserDataType,
                                                                         CShuffleDataType,
                                                                         InElementOp,
                                                                         WeiElementOp,
                                                                         PassThrough>;

bool run_fasternet50_conv_conv_add_relu_conv_add_wcnn_example(int argc, char* argv[])
{
    constexpr ck::index_t NDimSpatial                                 = 2;
    static const std::array<ck::utils::conv::ConvParam, 3> conv_param = {
        FasterNet50_DefaultConvParam_3x3,
        FasterNet50_DefaultConvParam_1x1_ReluAdd,
        FasterNet50_DefaultConvParam_1x1};

    bool do_verification = true;
    int init_method      = 1;

    const auto in_g_n_c_wis_desc       = make_input_descriptor(conv_param[0]);
    const auto wei_g_k_c_xs_3x3_desc   = make_weight_descriptor(conv_param[0]);
    const auto wei_g_k_c_xs_1x1_0_desc = make_weight_descriptor(conv_param[1]);
    const auto wei_g_k_c_xs_1x1_1_desc = make_weight_descriptor(conv_param[2]);

    const auto out_g_n_k_wos_0_desc = make_output_descriptor(conv_param[0]);
    const auto out_g_n_k_wos_1_desc = make_output_descriptor(conv_param[1]);
    const auto out_g_n_k_wos_2_desc = make_output_descriptor(conv_param[2]);

    const auto scalebias_g_n_k_wos_0_desc = make_bias_descriptor(conv_param[0]);
    const auto scalebias_g_n_k_wos_1_desc = make_bias_descriptor(conv_param[1]);
    const auto scalebias_g_n_k_wos_2_desc = make_bias_descriptor(conv_param[2]);

    Tensor<InUserDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiUserDataType> wei_0(wei_g_k_c_xs_3x3_desc);
    Tensor<WeiUserDataType> wei_1(wei_g_k_c_xs_1x1_0_desc);
    Tensor<WeiUserDataType> wei_2(wei_g_k_c_xs_1x1_1_desc);

    Tensor<BiasKernelDataType> bias_0(scalebias_g_n_k_wos_0_desc);
    Tensor<BiasKernelDataType> bias_1(scalebias_g_n_k_wos_1_desc);
    Tensor<BiasKernelDataType> bias_2(scalebias_g_n_k_wos_2_desc);

    Tensor<ScaleKernelDataType> scale_0(scalebias_g_n_k_wos_0_desc);
    Tensor<ScaleKernelDataType> scale_1(scalebias_g_n_k_wos_1_desc);
    Tensor<ScaleKernelDataType> scale_2(scalebias_g_n_k_wos_2_desc);

    Tensor<OutUserDataType> out_host_0(out_g_n_k_wos_0_desc);
    Tensor<OutUserDataType> out_host_1(out_g_n_k_wos_1_desc);
    Tensor<OutUserDataType> out_host_2(out_g_n_k_wos_2_desc);

#if defined(CASCADE_1X_OUT)
    Tensor<OutKernelDataType> out_device(out_g_n_k_wos_0_desc);
#elif defined(CASCADE_2X_OUT)
    Tensor<OutKernelDataType> out_device(out_g_n_k_wos_1_desc);
#else
    Tensor<OutKernelDataType> out_device(out_g_n_k_wos_2_desc);
#endif

    std::cout << "in: " << in.mDesc << std::endl;

    std::cout << "wei_0: " << wei_0.mDesc << std::endl;
    std::cout << "wei_1: " << wei_1.mDesc << std::endl;
    std::cout << "wei_2: " << wei_2.mDesc << std::endl;

    std::cout << "bias_0: " << bias_0.mDesc << std::endl;
    std::cout << "bias_1: " << bias_1.mDesc << std::endl;
    std::cout << "bias_2: " << bias_2.mDesc << std::endl;

    std::cout << "scale_0: " << scale_0.mDesc << std::endl;
    std::cout << "scale_1: " << scale_1.mDesc << std::endl;
    std::cout << "scale_2: " << scale_2.mDesc << std::endl;

    std::cout << "out: " << out_host_0.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InUserDataType>{-5, 5});
        wei_0.GenerateTensorValue(GeneratorTensor_2<WeiUserDataType>{-5, 5});
        wei_1.GenerateTensorValue(GeneratorTensor_2<WeiUserDataType>{-1, 1});
        wei_2.GenerateTensorValue(GeneratorTensor_3<WeiUserDataType>{-0.5, 0.5});
        bias_0.GenerateTensorValue(GeneratorTensor_2<BiasKernelDataType>{-2, 2});
        bias_1.GenerateTensorValue(GeneratorTensor_2<BiasKernelDataType>{-2, 2});
        bias_2.GenerateTensorValue(GeneratorTensor_2<BiasKernelDataType>{-2, 2});
        scale_0.GenerateTensorValue(GeneratorTensor_2<ScaleKernelDataType>{-1, 1});
        scale_1.GenerateTensorValue(GeneratorTensor_2<ScaleKernelDataType>{-1, 1});
        scale_2.GenerateTensorValue(GeneratorTensor_2<ScaleKernelDataType>{-1, 1});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InUserDataType>{0.0, 1.0});
        wei_0.GenerateTensorValue(GeneratorTensor_3<WeiUserDataType>{-0.5, 0.5});
        wei_1.GenerateTensorValue(GeneratorTensor_3<WeiUserDataType>{-0.5, 0.5});
        wei_2.GenerateTensorValue(GeneratorTensor_3<WeiUserDataType>{-0.5, 0.5});
        bias_0.GenerateTensorValue(GeneratorTensor_3<BiasKernelDataType>{-0.5, 0.5});
        bias_1.GenerateTensorValue(GeneratorTensor_3<BiasKernelDataType>{-0.5, 0.5});
        bias_2.GenerateTensorValue(GeneratorTensor_3<BiasKernelDataType>{-0.5, 0.5});
        scale_0.GenerateTensorValue(GeneratorTensor_3<ScaleKernelDataType>{-0.5, 0.5});
        scale_1.GenerateTensorValue(GeneratorTensor_3<ScaleKernelDataType>{-0.5, 0.5});
        scale_2.GenerateTensorValue(GeneratorTensor_3<ScaleKernelDataType>{-0.5, 0.5});
        break;
    }

    DeviceMem in_device_buf(sizeof(InKernelDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_0_device_buf(sizeof(WeiKernelDataType) * wei_0.mDesc.GetElementSpaceSize());
    DeviceMem wei_1_device_buf(sizeof(WeiKernelDataType) * wei_1.mDesc.GetElementSpaceSize());
    DeviceMem wei_2_device_buf(sizeof(WeiKernelDataType) * wei_2.mDesc.GetElementSpaceSize());
    DeviceMem bias_0_device_buf(sizeof(BiasKernelDataType) * bias_0.mDesc.GetElementSpaceSize());
    DeviceMem bias_1_device_buf(sizeof(BiasKernelDataType) * bias_1.mDesc.GetElementSpaceSize());
    DeviceMem bias_2_device_buf(sizeof(BiasKernelDataType) * bias_2.mDesc.GetElementSpaceSize());
    DeviceMem scale_0_device_buf(sizeof(ScaleKernelDataType) * scale_0.mDesc.GetElementSpaceSize());
    DeviceMem scale_1_device_buf(sizeof(ScaleKernelDataType) * scale_1.mDesc.GetElementSpaceSize());
    DeviceMem scale_2_device_buf(sizeof(ScaleKernelDataType) * scale_2.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutKernelDataType) * out_device.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in.mData.data());
    wei_0_device_buf.ToDevice(wei_0.mData.data());
    wei_1_device_buf.ToDevice(wei_1.mData.data());
    wei_2_device_buf.ToDevice(wei_2.mData.data());
    bias_0_device_buf.ToDevice(bias_0.mData.data());
    bias_1_device_buf.ToDevice(bias_1.mData.data());
    bias_2_device_buf.ToDevice(bias_2.mData.data());
    scale_0_device_buf.ToDevice(scale_0.mData.data());
    scale_1_device_buf.ToDevice(scale_1.mData.data());
    scale_2_device_buf.ToDevice(scale_2.mData.data());

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<std::array<ck::index_t, NDimSpatial + 3>, 3> b_g_k_c_xs_lengths{};
    std::array<std::array<ck::index_t, NDimSpatial + 3>, 3> b_g_k_c_xs_strides{};

    std::array<std::array<ck::index_t, NDimSpatial + 3>, 2> ds_lengths_0{};
    std::array<std::array<ck::index_t, NDimSpatial + 3>, 2> ds_lengths_1{};
    std::array<std::array<ck::index_t, NDimSpatial + 3>, 2> ds_lengths_2{};
    std::array<std::array<ck::index_t, NDimSpatial + 3>, 2> ds_strides_0{};
    std::array<std::array<ck::index_t, NDimSpatial + 3>, 2> ds_strides_1{};
    std::array<std::array<ck::index_t, NDimSpatial + 3>, 2> ds_strides_2{};

    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{};

    std::array<std::array<ck::index_t, NDimSpatial>, 3> conv_filter_strides{};
    std::array<std::array<ck::index_t, NDimSpatial>, 3> conv_filter_dilations{};
    std::array<std::array<ck::index_t, NDimSpatial>, 3> input_left_pads{};
    std::array<std::array<ck::index_t, NDimSpatial>, 3> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_3x3_desc.GetLengths(), b_g_k_c_xs_lengths[0]);
    copy(wei_g_k_c_xs_3x3_desc.GetStrides(), b_g_k_c_xs_strides[0]);
    copy(wei_g_k_c_xs_1x1_0_desc.GetLengths(), b_g_k_c_xs_lengths[1]);
    copy(wei_g_k_c_xs_1x1_0_desc.GetStrides(), b_g_k_c_xs_strides[1]);
    copy(wei_g_k_c_xs_1x1_1_desc.GetLengths(), b_g_k_c_xs_lengths[2]);
    copy(wei_g_k_c_xs_1x1_1_desc.GetStrides(), b_g_k_c_xs_strides[2]);

    copy(scalebias_g_n_k_wos_0_desc.GetLengths(), ds_lengths_0[0]);
    copy(scalebias_g_n_k_wos_0_desc.GetStrides(), ds_strides_0[0]);
    copy(scalebias_g_n_k_wos_1_desc.GetLengths(), ds_lengths_1[0]);
    copy(scalebias_g_n_k_wos_1_desc.GetStrides(), ds_strides_1[0]);
    copy(scalebias_g_n_k_wos_2_desc.GetLengths(), ds_lengths_2[0]);
    copy(scalebias_g_n_k_wos_2_desc.GetStrides(), ds_strides_2[0]);

    copy(scalebias_g_n_k_wos_0_desc.GetLengths(), ds_lengths_0[1]);
    copy(scalebias_g_n_k_wos_0_desc.GetStrides(), ds_strides_0[1]);
    copy(scalebias_g_n_k_wos_1_desc.GetLengths(), ds_lengths_1[1]);
    copy(scalebias_g_n_k_wos_1_desc.GetStrides(), ds_strides_1[1]);
    copy(scalebias_g_n_k_wos_2_desc.GetLengths(), ds_lengths_2[1]);
    copy(scalebias_g_n_k_wos_2_desc.GetStrides(), ds_strides_2[1]);

#if defined(CASCADE_1X_OUT)
    copy(out_g_n_k_wos_0_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_0_desc.GetStrides(), e_g_n_k_wos_strides);
#elif defined(CASCADE_2X_OUT)
    copy(out_g_n_k_wos_1_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_1_desc.GetStrides(), e_g_n_k_wos_strides);
#else
    copy(out_g_n_k_wos_2_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_2_desc.GetStrides(), e_g_n_k_wos_strides);
#endif

    copy(conv_param[0].conv_filter_strides_, conv_filter_strides[0]);
    copy(conv_param[1].conv_filter_strides_, conv_filter_strides[1]);
    copy(conv_param[2].conv_filter_strides_, conv_filter_strides[2]);

    copy(conv_param[0].conv_filter_dilations_, conv_filter_dilations[0]);
    copy(conv_param[1].conv_filter_dilations_, conv_filter_dilations[1]);
    copy(conv_param[2].conv_filter_dilations_, conv_filter_dilations[2]);

    copy(conv_param[0].input_left_pads_, input_left_pads[0]);
    copy(conv_param[0].input_right_pads_, input_right_pads[0]);
    copy(conv_param[1].input_left_pads_, input_left_pads[1]);
    copy(conv_param[1].input_right_pads_, input_right_pads[1]);
    copy(conv_param[2].input_left_pads_, input_left_pads[2]);
    copy(conv_param[2].input_right_pads_, input_right_pads[2]);

    using CPerBlocks = ck::Sequence<32, 32, 64>;
    using KPerBlocks = ck::Sequence<16, 64, 32>;

    constexpr bool InEnableLds         = true;
    constexpr bool WeiEnableLds        = true;
    constexpr bool DsEnableLds         = true;
    constexpr bool DsBlockLdsAddExtraM = true;
    constexpr ck::index_t WeiBlockTransferScalarPerVector =
        sizeof(uint32_t) / sizeof(WeiKernelDataType);

    constexpr ck::index_t DBlockTransferScalarPerVector =
        sizeof(uint32_t) / sizeof(ScaleKernelDataType);
    constexpr ck::index_t Cluster_Ds_K_0 = KPerBlocks{}.At(0) / DBlockTransferScalarPerVector;
    constexpr ck::index_t Cluster_Ds_K_1 = KPerBlocks{}.At(1) / DBlockTransferScalarPerVector;
    constexpr ck::index_t Cluster_Ds_K_2 = KPerBlocks{}.At(2) / DBlockTransferScalarPerVector;

    using DsBlockTransferScalarPerVector_0 =
        ck::Sequence<DBlockTransferScalarPerVector, DBlockTransferScalarPerVector>;
    using DsBlockTransferScalarPerVector_1 =
        ck::Sequence<DBlockTransferScalarPerVector, DBlockTransferScalarPerVector>;
    using DsBlockTransferScalarPerVector_2 =
        ck::Sequence<DBlockTransferScalarPerVector, DBlockTransferScalarPerVector>;

    using DsDataType =
        ck::Tuple<ck::Tuple<>, ck::Tuple<ScaleKernelDataType, BiasKernelDataType>, ck::Tuple<>>;
    using DsDataLayout =
        ck::Tuple<ck::Tuple<>, ck::Tuple<ScaleLayout<2>, BiasLayout<2>>, ck::Tuple<>>;
    using DsBlockTransferThreadClusterLengths = ck::Tuple<ck::Sequence<Cluster_Ds_K_0>,
                                                          ck::Sequence<Cluster_Ds_K_1>,
                                                          ck::Sequence<Cluster_Ds_K_2>>;
    using DsBlockTransferScalarPerVector      = ck::Tuple<DsBlockTransferScalarPerVector_0,
                                                     DsBlockTransferScalarPerVector_1,
                                                     DsBlockTransferScalarPerVector_2>;

    using DeviceConvFwdInstance = ck::tensor_operation::device::DeviceFasternet50_Wcnn_CShuffle<
        NDimSpatial,
        InputLayout<NDimSpatial>,
        WeightLayout<NDimSpatial>,
        DsDataLayout,
        OutputLayout<NDimSpatial>,
        InKernelDataType,
        WeiKernelDataType,
        DsDataType,
        AccDataType,
        OutKernelDataType,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        1,
        128,
        HPerBlock,
        WPerBlock,
        CPerBlocks,
        KPerBlocks,
        HPerBlock / 2 / HPerWcnn,
        WPerBlock / 2 / WPerWcnn,
        HPerWcnn,
        WPerWcnn,
        ck::Sequence<3, 1, 1>,
        ck::Sequence<1, 1, 1>,
        ck::Sequence<1, 1, 1>,
        S<8, 4, 4>,
        4,
        4,
        InEnableLds,
        false,
        false,
        S<16, 1, 8>,
        WeiBlockTransferScalarPerVector,
        WeiBlockTransferScalarPerVector,
        WeiEnableLds,
        false,
        false,
        DsBlockTransferThreadClusterLengths,
        DsBlockTransferScalarPerVector,
        DsBlockTransferScalarPerVector,
        DsEnableLds,
        DsBlockLdsAddExtraM,
        S<1, 4, 4>,
        4,
        true,
        EnableAsync,
        EnableWaveGroup,
        EnableSpatialCluster,
        0,
        false,
        false,
        false>;

    std::array<const void*, 3> p_wei_buf = {wei_0_device_buf.GetDeviceBuffer(),
                                            wei_1_device_buf.GetDeviceBuffer(),
                                            wei_2_device_buf.GetDeviceBuffer()};

    auto conv    = DeviceConvFwdInstance{};
    auto invoker = conv.MakeInvoker();
    auto argument =
        conv.MakeArgument(in_device_buf.GetDeviceBuffer(),
                          p_wei_buf,
                          {}, // std::array<const void*, 2>{scale_0_device_buf.GetDeviceBuffer(),
                              // bias_0_device_buf.GetDeviceBuffer()},
                          std::array<const void*, 2>{scale_1_device_buf.GetDeviceBuffer(),
                                                     bias_1_device_buf.GetDeviceBuffer()},
                          {}, // std::array<const void*, 2>{scale_2_device_buf.GetDeviceBuffer(),
                              // bias_2_device_buf.GetDeviceBuffer()},
                          out_device_buf.GetDeviceBuffer(),
                          a_g_n_c_wis_lengths,
                          a_g_n_c_wis_strides,
                          b_g_k_c_xs_lengths,
                          b_g_k_c_xs_strides,
                          {}, // ds_lengths_0,
                          ds_lengths_1,
                          {}, // ds_lengths_2,
                          {}, // ds_strides_0,
                          ds_strides_1,
                          {}, // ds_strides_2,
                          e_g_n_k_wos_lengths,
                          e_g_n_k_wos_strides,
                          conv_filter_strides,
                          conv_filter_dilations,
                          input_left_pads,
                          input_right_pads,
                          InElementOp{},
                          WeiElementOp{},
                          OutElementOp{});

    if(!conv.IsSupportedArgument(argument))
    {
        std::cout << "wrong! device_conv with the specified compilation parameters does "
                     "not support this Conv problem"
                  << std::endl;
        return false;
    }

    invoker.Run(argument, StreamConfig{nullptr, false});

    out_device_buf.FromDevice(out_device.mData.data());

    std::cout <<
#ifdef ENABLE_WAVEGROUP
        "grouped_conv_fwd_wcnn_wavegroup<In/Wei:"
#else
        "grouped_conv_fwd_wcnn<In/Wei:"
#endif
              << get_string<InUserDataType>() << ", Out:" << get_string<AccDataType>() << ", "
              << "WaveGroup:" << EnableWaveGroup << " }>: Status: ";

    if(do_verification)
    {
        Tensor<CShuffleDataType> c0_host(out_g_n_k_wos_0_desc);
        Tensor<CShuffleDataType> c1_host(out_g_n_k_wos_1_desc);
        Tensor<CShuffleDataType> c2_host(out_g_n_k_wos_2_desc);

        const auto in_split_g_n_c_wis_desc =
            make_input_descriptor(FasterNet50_DefaultConvParam_3x3_split);
        const auto wei_split_g_n_c_wis_desc =
            make_weight_descriptor(FasterNet50_DefaultConvParam_3x3_split);
        Tensor<InUserDataType> in_split_tensor(in_split_g_n_c_wis_desc);
        Tensor<WeiUserDataType> wei_split(wei_split_g_n_c_wis_desc);
        partition_tensor(in, in_split_tensor);
        partition_tensor(wei_0, wei_split);

        auto ref_conv_0          = HostConvFwdInstance<NDimSpatial>{};
        auto ref0_invoker        = ref_conv_0.MakeInvoker();
        auto ref_conv_0_argument = ref_conv_0.MakeArgument(in_split_tensor,
                                                           wei_split,
                                                           c0_host,
                                                           conv_param[0].conv_filter_strides_,
                                                           conv_param[0].conv_filter_dilations_,
                                                           conv_param[0].input_left_pads_,
                                                           conv_param[0].input_right_pads_,
                                                           InElementOp{},
                                                           WeiElementOp{},
                                                           PassThrough{});

        ref0_invoker.Run(ref_conv_0_argument);
        dump_tensor(in, "in_0");
        dump_tensor(wei_0, "wei_0");
        dump_tensor(c0_host, "c0_host");
        using OutElementOp_0 = ck::tuple_element_t<0, OutElementOp>;
        out_host_0.ForEach(
            [&](auto&, auto idx) { OutElementOp_0{}(out_host_0(idx), c0_host(idx)); });

        const auto in_merge_g_n_c_wis_desc = make_input_descriptor(conv_param[1]);
        Tensor<InUserDataType> in_1(in_merge_g_n_c_wis_desc);
        merge_tensor(out_host_0, in, in_1);

        auto ref_conv_1          = HostConvFwdInstance<NDimSpatial>{};
        auto ref1_invoker        = ref_conv_1.MakeInvoker();
        auto ref_conv_1_argument = ref_conv_1.MakeArgument(in_1,
                                                           wei_1,
                                                           c1_host,
                                                           conv_param[1].conv_filter_strides_,
                                                           conv_param[1].conv_filter_dilations_,
                                                           conv_param[1].input_left_pads_,
                                                           conv_param[1].input_right_pads_,
                                                           InElementOp{},
                                                           WeiElementOp{},
                                                           PassThrough{});

        ref1_invoker.Run(ref_conv_1_argument);
        dump_tensor(out_host_0, "in_out_1:");
        dump_tensor(wei_1, "wei_1");
        dump_tensor(scale_1, "scale_1");
        dump_tensor(bias_1, "bias_1");
        dump_tensor(c1_host, "c1_host");

        using OutElementOp_1 = ck::tuple_element_t<1, OutElementOp>;
        out_host_1.ForEach([&](auto&, auto idx) {
            OutElementOp_1{}(out_host_1(idx), c1_host(idx), scale_1(idx), bias_1(idx));
        });

        auto ref_conv_2          = HostConvFwdInstance<NDimSpatial>{};
        auto ref2_invoker        = ref_conv_2.MakeInvoker();
        auto ref_conv_2_argument = ref_conv_2.MakeArgument(out_host_1,
                                                           wei_2,
                                                           c2_host,
                                                           conv_param[2].conv_filter_strides_,
                                                           conv_param[2].conv_filter_dilations_,
                                                           conv_param[2].input_left_pads_,
                                                           conv_param[2].input_right_pads_,
                                                           InElementOp{},
                                                           WeiElementOp{},
                                                           PassThrough{});

        ref2_invoker.Run(ref_conv_2_argument);
        dump_tensor(out_host_1, "in_out_1:");
        dump_tensor(wei_2, "wei_2");
        dump_tensor(c2_host, "c1_host");

        using OutElementOp_2 = ck::tuple_element_t<2, OutElementOp>;
        out_host_2.ForEach(
            [&](auto&, auto idx) { OutElementOp_2{}(out_host_2(idx), c2_host(idx)); });

        out_device_buf.FromDevice(out_device.mData.data());
        dump_tensor(out_host_2, "out_host_2");
        dump_tensor(out_device, "out_device");

#if defined(CASCADE_1X_OUT)
        bool ret =
            ck::utils::check_err(out_device, out_host_0, "Error: incorrect results!", 1e-3f, 1e-3f);
#elif defined(CASCADE_2X_OUT)
        bool ret =
            ck::utils::check_err(out_device, out_host_1, "Error: incorrect results!", 1e-3f, 1e-3f);
#else
        bool ret =
            ck::utils::check_err(out_device, out_host_2, "Error: incorrect results!", 1e-3f, 1e-3f);
#endif

        if(ret)
        {
            std::cout << "Passed\n";
        }
        else
        {
            std::cout << "Failed\n";
        }
    }

    (void)argc;
    (void)argv;
    return true;
}
