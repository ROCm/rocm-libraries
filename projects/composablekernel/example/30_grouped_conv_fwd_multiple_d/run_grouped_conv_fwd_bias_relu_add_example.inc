// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

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
using ResidualLayout = typename LayoutSettingSelector<NDimSpatial>::ResidualLayout;

// instance for double rate mfma on gfx950 (vs gfx942)
template <ck::index_t NDimSpatial>
using DeviceConvFwdInstance2 =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
        NDimSpatial,
        InputLayout<NDimSpatial>,
        WeightLayout<NDimSpatial>,
        ck::Tuple<BiasLayout<NDimSpatial>, ResidualLayout<NDimSpatial>>,
        OutputLayout<NDimSpatial>,
        InKernelDataType,
        WeiKernelDataType,
        AccDataType,
        CShuffleDataType,
        ck::Tuple<BiasKernelDataType, ResidualKernelDataType>,
        OutKernelDataType,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        ConvSpec,    // ConvForwardSpecialization
        GemmSpec,    // GemmSpecialization
        1,           //
        256,         // BlockSize
        128,         // MPerBlock
        64,          // NPerBlock
        64,          // KPerBlock
        16,          // AK1
        16,          // BK1
        32,          // MPerXdl
        32,          // NPerXdl
        2,           // MXdlPerWave
        1,           // NXdlPerWave
        S<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // ABlockTransferSrcAccessOrder
        2,           // ABlockTransferSrcVectorDim
        4,           // ABlockTransferSrcScalarPerVector
        4,           // ABlockTransferDstScalarPerVector_AK1
        1,           // ABlockLdsExtraM
        S<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // BBlockTransferSrcAccessOrder
        2,           // BBlockTransferSrcVectorDim
        4,           // BBlockTransferSrcScalarPerVector
        4,           // BBlockTransferDstScalarPerVector_BK1
        1,           // BBlockLdsExtraN
        1,
        1,
        S<1, 16, 1, 16>,
        4>;
// instance for gfx942-
template <ck::index_t NDimSpatial>
using DeviceConvFwdInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
        NDimSpatial,
        InputLayout<NDimSpatial>,
        WeightLayout<NDimSpatial>,
        ck::Tuple<BiasLayout<NDimSpatial>, ResidualLayout<NDimSpatial>>,
        OutputLayout<NDimSpatial>,
        InKernelDataType,
        WeiKernelDataType,
        AccDataType,
        CShuffleDataType,
        ck::Tuple<BiasKernelDataType, ResidualKernelDataType>,
        OutKernelDataType,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        ConvSpec,    // ConvForwardSpecialization
        GemmSpec,    // GemmSpecialization
        1,           //
        256,         // BlockSize
        128,         // MPerBlock
        128,         // NPerBlock
        32,          // KPerBlock
        4,           // AK1
        4,           // BK1
        32,          // MPerXdl
        32,          // NPerXdl
        2,           // MXdlPerWave
        2,           // NXdlPerWave
        S<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // ABlockTransferSrcAccessOrder
        2,           // ABlockTransferSrcVectorDim
        4,           // ABlockTransferSrcScalarPerVector
        4,           // ABlockTransferDstScalarPerVector_AK1
        1,           // ABlockLdsExtraM
        S<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // BBlockTransferSrcAccessOrder
        2,           // BBlockTransferSrcVectorDim
        4,           // BBlockTransferSrcScalarPerVector
        4,           // BBlockTransferDstScalarPerVector_BK1
        1,           // BBlockLdsExtraN
        1,
        1,
        S<1, 16, 1, 16>,
        4>;

template <ck::index_t NDimSpatial>
using HostConvFwdInstance = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                         InUserDataType,
                                                                         WeiUserDataType,
                                                                         CShuffleDataType,
                                                                         InElementOp,
                                                                         WeiElementOp,
                                                                         PassThrough>;

template <ck::index_t NDimSpatial>
bool run_grouped_conv_fwd_bias_relu_add(const ExecutionConfig& config,
                                        const ck::utils::conv::ConvParam& conv_param)
{
    static_assert(1 <= NDimSpatial && NDimSpatial <= 3, "Unsupported NDimSpatial");

    const auto in_g_n_c_wis_desc   = make_input_descriptor(conv_param);
    const auto wei_g_k_c_xs_desc   = make_weight_descriptor(conv_param);
    const auto bias_g_n_k_wos_desc = make_bias_descriptor(conv_param);
    const auto out_g_n_k_wos_desc  = make_output_descriptor(conv_param);

    Tensor<InUserDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiUserDataType> wei(wei_g_k_c_xs_desc);
    Tensor<OutUserDataType> bias(bias_g_n_k_wos_desc);
    Tensor<OutUserDataType> residual(bias_g_n_k_wos_desc);
    Tensor<OutUserDataType> out_host(out_g_n_k_wos_desc);
    Tensor<OutKernelDataType> out_device(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "bias: " << bias.mDesc << std::endl;
    std::cout << "residual: " << residual.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InUserDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiUserDataType>{-5, 5});
        bias.GenerateTensorValue(GeneratorTensor_2<OutUserDataType>{-5, 5});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InUserDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiUserDataType>{-0.5, 0.5});
        bias.GenerateTensorValue(GeneratorTensor_3<OutUserDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(sizeof(InKernelDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiKernelDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem bias_device_buf(sizeof(OutKernelDataType) * bias.mDesc.GetElementSpaceSize());
    DeviceMem residual_device_buf(sizeof(OutKernelDataType) * residual.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutKernelDataType) * out_device.mDesc.GetElementSpaceSize());

#ifdef BUILD_INT4_EXAMPLE
    const Tensor<InKernelDataType> in_converted(in);
    const Tensor<WeiKernelDataType> wei_converted(wei);
    const Tensor<OutKernelDataType> bias_converted(bias);
    const Tensor<OutKernelDataType> residual_converted(residual);

    in_device_buf.ToDevice(in_converted.mData.data());
    wei_device_buf.ToDevice(wei_converted.mData.data());
    bias_device_buf.ToDevice(bias_converted.mData.data());
    residual_device_buf.ToDevice(residual_converted.mData.data());
#else
    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());
    bias_device_buf.ToDevice(bias.mData.data());
    residual_device_buf.ToDevice(residual.mData.data());
#endif

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> d0_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> d0_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial + 3> d1_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> d1_g_n_k_wos_strides{};
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
    copy(bias_g_n_k_wos_desc.GetLengths(), d0_g_n_k_wos_lengths);
    copy(bias_g_n_k_wos_desc.GetStrides(), d0_g_n_k_wos_strides);
    copy(bias_g_n_k_wos_desc.GetLengths(), d1_g_n_k_wos_lengths);
    copy(bias_g_n_k_wos_desc.GetStrides(), d1_g_n_k_wos_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    // do Conv
    using BaseGroupedConvFwdMultipleABD =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
            NDimSpatial,
            InputLayout<NDimSpatial>,
            WeightLayout<NDimSpatial>,
            ck::Tuple<BiasLayout<NDimSpatial>, ResidualLayout<NDimSpatial>>,
            OutputLayout<NDimSpatial>,
            InKernelDataType,
            WeiKernelDataType,
            ck::Tuple<BiasKernelDataType, ResidualKernelDataType>,
            OutKernelDataType,
            InElementOp,
            WeiElementOp,
            OutElementOp,
            InKernelDataType,  // AComputeDataType
            InKernelDataType>; // BComputeDataType

    static_assert(
        std::is_base_of<BaseGroupedConvFwdMultipleABD, DeviceConvFwdInstance<NDimSpatial>>::value &&
        std::is_base_of<BaseGroupedConvFwdMultipleABD, DeviceConvFwdInstance2<NDimSpatial>>::value);

    auto conv  = DeviceConvFwdInstance<NDimSpatial>{};  // instance for gfx942-
    auto conv2 = DeviceConvFwdInstance2<NDimSpatial>{}; // instance for double rate mfma instruction
                                                        // on gfx950
    BaseGroupedConvFwdMultipleABD* op_ptr =
        (ck::get_device_name() == "gfx950") ? static_cast<BaseGroupedConvFwdMultipleABD*>(&conv2)
                                            : static_cast<BaseGroupedConvFwdMultipleABD*>(&conv);
    auto invoker_ptr  = op_ptr->MakeInvokerPointer();
    auto argument_ptr = op_ptr->MakeArgumentPointer(
        in_device_buf.GetDeviceBuffer(),
        wei_device_buf.GetDeviceBuffer(),
        std::array<const void*, 2>{bias_device_buf.GetDeviceBuffer(),
                                   residual_device_buf.GetDeviceBuffer()},
        out_device_buf.GetDeviceBuffer(),
        a_g_n_c_wis_lengths,
        a_g_n_c_wis_strides,
        b_g_k_c_xs_lengths,
        b_g_k_c_xs_strides,
        std::array<std::array<ck::index_t, NDimSpatial + 3>, 2>{
            {d0_g_n_k_wos_lengths, d1_g_n_k_wos_lengths}},
        std::array<std::array<ck::index_t, NDimSpatial + 3>, 2>{
            {d0_g_n_k_wos_strides, d1_g_n_k_wos_strides}},
        e_g_n_k_wos_lengths,
        e_g_n_k_wos_strides,
        conv_filter_strides,
        conv_filter_dilations,
        input_left_pads,
        input_right_pads,
        InElementOp{},
        WeiElementOp{},
        OutElementOp{});

    if(!op_ptr->IsSupportedArgument(argument_ptr.get()))
    {
        throw std::runtime_error(
            "wrong! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    float avg_time =
        invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, config.time_kernel});

    std::size_t flop      = conv_param.GetFlops();
    std::size_t num_btype = conv_param.GetByte<InUserDataType, WeiUserDataType, OutUserDataType>();

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_btype / 1.E6 / avg_time;
    std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << op_ptr->GetTypeString() << std::endl;

    if(config.do_verification)
    {
        Tensor<CShuffleDataType> c_host(out_g_n_k_wos_desc);

        auto ref_conv     = HostConvFwdInstance<NDimSpatial>{};
        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(in,
                                                  wei,
                                                  c_host,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  InElementOp{},
                                                  WeiElementOp{},
                                                  PassThrough{});

        ref_invoker.Run(ref_argument);

        // TODO: implement elementwise operation for host
        out_host.ForEach([&](auto&, auto idx) {
            OutElementOp{}(out_host(idx), c_host(idx), bias(idx), residual(idx));
        });

        out_device_buf.FromDevice(out_device.mData.data());

#ifdef BUILD_INT4_EXAMPLE
        const Tensor<OutUserDataType> out_device_converted(out_device);

        return ck::utils::check_err(
            out_device_converted, out_host, "Error: incorrect results!", 1e-5f, 1e-4f);
#else
        return ck::utils::check_err(
            out_device, out_host, "Error: incorrect results!", 1e-5f, 1e-4f);
#endif
    }

    return true;
}

bool run_grouped_conv_fwd_bias_relu_add_example(int argc, char* argv[])
{
    ExecutionConfig config;
    ck::utils::conv::ConvParam conv_param = DefaultConvParam;

    if(!parse_cmd_args(argc, argv, config, conv_param))
    {
        return false;
    }

    switch(conv_param.num_dim_spatial_)
    {
    case 1: return run_grouped_conv_fwd_bias_relu_add<1>(config, conv_param);
    case 2: return run_grouped_conv_fwd_bias_relu_add<2>(config, conv_param);
    case 3: return run_grouped_conv_fwd_bias_relu_add<3>(config, conv_param);
    }

    return false;
}
