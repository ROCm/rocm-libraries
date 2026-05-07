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

template <ck::index_t NDimSpatial>
using ResidualLayout = typename LayoutSettingSelector<NDimSpatial>::ResidualLayout;

template <ck::index_t NDimSpatial>
using ResidualScaleLayout = typename LayoutSettingSelector<NDimSpatial>::ResidualLayout;

using ConvolutionForwardSpecialization =
    typename ck::tensor_operation::device::ConvolutionForwardSpecialization;

static constexpr ck::index_t HPerWcnn      = 4;
static constexpr ck::index_t WPerWcnn      = 2;
static constexpr ck::index_t HPerBlock     = 16;
static constexpr ck::index_t WPerBlock     = 16;
static constexpr bool EnableAsync          = false;
static constexpr bool EnableWaveGroup      = false;
static constexpr bool EnableSpatialCluster = false;

template <ck::index_t NDimSpatial,
          ck::index_t FilterSize,
          ck::index_t DilationSize,
          ConvolutionForwardSpecialization ConvSpec>
using DeviceConvFwdInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Wcnn_CShuffle<
        NDimSpatial,
        InputLayout<NDimSpatial>,
        WeightLayout<NDimSpatial>,
        ck::Tuple<BiasLayout<NDimSpatial>, ResidualLayout<NDimSpatial>>,
        OutputLayout<NDimSpatial>,
        InKernelDataType,
        WeiKernelDataType,
        ck::Tuple<BiasKernelDataType, ResidualKernelDataType>,
        AccDataType,
        OutKernelDataType,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        ConvSpec,                     // ConvForwardSpecialization
        1,                            // NumPrefetch
        128,                          // BlockSize
        HPerBlock,                    // HPerBlock
        WPerBlock,                    // WPerBlock
        16,                           // CPerBlock
        16,                           // KPerBlock
        HPerBlock / 2 / HPerWcnn,     // HRepeat
        WPerBlock / 2 / WPerWcnn,     // WRepeat
        HPerWcnn,                     // HPerWcnn
        WPerWcnn,                     // WPerWcnn
        FilterSize,                   // FilterSize
        DilationSize,                 // DilationX
        DilationSize,                 // DilationY
        S<8, 4, 4>,                   // InBlockTransferThreadClusterLengths (H W C)
        4,                            // InBlockTransferSrcScalarPerVector
        4,                            // InBlockTransferDstScalarPerVector
        true,                         // InEnableLds
        false,                        // InBlockLdsAddExtraM
        false,                        // InTileLoad
        S<16, 1, 8>,                  // WeiBlockTransferThreadClusterLengths
        2,                            // WeiBlockTransferSrcScalarPerVector
        2,                            // WeiBlockTransferDstScalarPerVector
        true,                         // WeiEnableLds
        false,                        // WeiBlockLdsAddExtraM
        false,                        // WeiTileLoad
        ck::Tuple<S<16>, S<8, 4, 4>>, // DsBlockTransferThreadClusterLengths (K, HWK), not used when
                                      // lds disabled
        S<1, 4>,    // DsBlockTransferSrcScalarPerVector, not used when lds disabled
        S<1, 4>,    // DsBlockTransferDstScalarPerVector, not used when lds disabled
        false,      // DsEnableLds
        false,      // DsBlockLdsAddExtraM
        S<1, 4, 4>, // AccBlockTransferClusterLengths
        4,          // AccBlockTransferScalarPerVector
        true,       // AccEnableLds
        EnableAsync,
        EnableWaveGroup,
        false, // EnableSpatialCluster
        0,
        false,  // ShuffleOnLoad
        false>; // Transposed

using DeviceWcnnConv1DFactory = std::tuple<
    DeviceConvFwdInstance<1, 1, 1, ConvolutionForwardSpecialization::Filter1x1Stride1Pad0>,
    DeviceConvFwdInstance<1, 1, 1, ConvolutionForwardSpecialization::Filter1x1Pad0>,
    DeviceConvFwdInstance<1, 1, 1, ConvolutionForwardSpecialization::Default>>;

using DeviceWcnnConv2DFactory = std::tuple<
    DeviceConvFwdInstance<2, 1, 1, ConvolutionForwardSpecialization::Filter1x1Stride1Pad0>,
    DeviceConvFwdInstance<2, 1, 1, ConvolutionForwardSpecialization::Filter1x1Pad0>,
    DeviceConvFwdInstance<2, 3, 1, ConvolutionForwardSpecialization::Filter3x3Stride1Pad0>,
    DeviceConvFwdInstance<2,
                          3,
                          1,
                          ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0>,
    DeviceConvFwdInstance<2, 3, 2, ConvolutionForwardSpecialization::Filter3x3Stride1Pad0>,
    DeviceConvFwdInstance<2,
                          3,
                          2,
                          ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0>,
    DeviceConvFwdInstance<2, 2, 1, ConvolutionForwardSpecialization::Filter2x2Stride2Pad0>,
    DeviceConvFwdInstance<2, 2, 1, ConvolutionForwardSpecialization::Filter2x2Stride2OddHWPad0>>;

using DeviceWcnnConv3DFactory = std::tuple<
    DeviceConvFwdInstance<3, 1, 1, ConvolutionForwardSpecialization::Filter1x1Stride1Pad0>,
    DeviceConvFwdInstance<3, 1, 1, ConvolutionForwardSpecialization::Filter1x1Pad0>,
    DeviceConvFwdInstance<3,
                          3,
                          1,
                          ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0>,
    DeviceConvFwdInstance<3,
                          3,
                          2,
                          ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0>>;

using DeviceWcnnConvFactory =
    std::tuple<DeviceWcnnConv1DFactory, DeviceWcnnConv2DFactory, DeviceWcnnConv3DFactory>;

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

    const auto in_g_n_c_wis_desc       = make_input_descriptor(conv_param);
    const auto wei_g_k_c_xs_desc       = make_weight_descriptor(conv_param);
    const auto bias_g_n_k_wos_desc     = make_bias_descriptor(conv_param);
    const auto residual_g_n_k_wos_desc = make_output_descriptor(conv_param);
    const auto out_g_n_k_wos_desc      = make_output_descriptor(conv_param);

    Tensor<InUserDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiUserDataType> wei(wei_g_k_c_xs_desc);
    Tensor<BiasKernelDataType> bias(bias_g_n_k_wos_desc);
    Tensor<ResidualKernelDataType> residual(residual_g_n_k_wos_desc);
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
        bias.GenerateTensorValue(GeneratorTensor_2<BiasKernelDataType>{-5, 5});
        residual.GenerateTensorValue(GeneratorTensor_2<ResidualKernelDataType>{-10, 10});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InUserDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiUserDataType>{-0.5, 0.5});
        bias.GenerateTensorValue(GeneratorTensor_3<BiasKernelDataType>{-0.5, 0.5});
        residual.GenerateTensorValue(GeneratorTensor_3<ResidualKernelDataType>{-5, 5});
        break;
    }

    DeviceMem in_device_buf(sizeof(InKernelDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiKernelDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem bias_device_buf(sizeof(BiasKernelDataType) * bias.mDesc.GetElementSpaceSize());
    DeviceMem residual_device_buf(sizeof(ResidualKernelDataType) *
                                  residual.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutKernelDataType) * out_device.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());
    bias_device_buf.ToDevice(bias.mData.data());
    residual_device_buf.ToDevice(residual.mData.data());

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
    copy(residual_g_n_k_wos_desc.GetLengths(), d1_g_n_k_wos_lengths);
    copy(residual_g_n_k_wos_desc.GetStrides(), d1_g_n_k_wos_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    bool found_kernel = false;

    auto conv_factory = std::get<NDimSpatial - 1>(DeviceWcnnConvFactory{});

    ck::static_for<0, std::tuple_size_v<decltype(conv_factory)>, 1>{}([&](auto i) -> void {
        const auto device_conv_instance = std::get<i>(conv_factory);
        using DeviceConvInstance        = ck::remove_cvref_t<decltype(device_conv_instance)>;
        // do Conv
        auto conv    = DeviceConvInstance{};
        auto invoker = conv.MakeInvoker();
        auto argument =
            conv.MakeArgument(in_device_buf.GetDeviceBuffer(),
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

        // run kernel with first find kernel
        if(conv.IsSupportedArgument(argument) && (found_kernel == false))
        {
            found_kernel = true;

            float avg_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

            std::size_t flop = conv_param.GetFlops();
            std::size_t num_btype =
                conv_param.GetByte<InUserDataType, WeiUserDataType, OutUserDataType>();

            float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
            float gb_per_sec = num_btype / 1.E6 / avg_time;
            std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                      << " GB/s, " << conv.GetTypeString() << std::endl;
        }
    });

    if(found_kernel == false)
    {
        throw std::runtime_error(
            "wrong! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

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
        // dump_tensor(in, "in");
        // dump_tensor(wei, "wei");
        // dump_tensor(c_host, "c_host");
        out_host.ForEach([&](auto&, auto idx) {
            OutElementOp{}(out_host(idx), c_host(idx), bias(idx), residual(idx));
        });
        // dump_tensor(out_host, "out_host");
        out_device_buf.FromDevice(out_device.mData.data());
        // dump_tensor(out_device, "out_device");
        return ck::utils::check_err(
            out_device, out_host, "Error: incorrect results!", 1e-5f, 1e-4f);
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
    // TODO: remove unsupported kernel instance for dimension 1 & 3.
    // case 1: return run_grouped_conv_fwd_bias_relu_add<1>(config, conv_param);
    case 2:
        return run_grouped_conv_fwd_bias_relu_add<2>(config, conv_param);
        // case 3: return run_grouped_conv_fwd_bias_relu_add<3>(config, conv_param);
    }

    return false;
}
