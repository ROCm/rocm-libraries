// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "common_wcnn.hpp"

// #define ENABLE_WAVEGROUP 1
// #define ENABLE_CONST_LAYOUT 1

#define DEFAULT_H_PERWAVE 8
#define DEFAULT_W_PERWAVE 8
#define DEFAULT_H_PERBLOCK 16
#define DEFAULT_W_PERBLOCK 16
#define DEFAULT_C_PERBLOCK 16
#define DEFAULT_K_PERBLOCK 16
#define DEFAULT_BLOCKSIZE 128
#define DEFAULT_WAVEGROUP_BLOCKSIZE 256

template <typename InDataType,
          typename WeiDataType,
          typename GPUAccType,
          ShapeType Shape,
          FilterType Filter,
          bool Dilation,
          int LdsMode,
          bool EnableWaveGroup,
          int32_t TestMask>
bool run_test()
{
    if((config.test_mask & 0xFFFF0000 & TestMask) == 0)
    {
        return true;
    }
    constexpr ck::index_t FilterSize =
        (Filter == Filter_1X1) ? 1 : ((Filter == Filter_2X2) ? 2 : 3);
    constexpr ck::index_t HPerWcnn = (Shape == Shape_8X4) ? 8 : 4;
    constexpr ck::index_t WPerWcnn = (Shape == Shape_4X2) ? 2 : 4;

    const ck::index_t Width          = config.w;
    const ck::index_t InputChannels  = config.c;
    const ck::index_t OutputChannels = config.k;
    constexpr ck::index_t BlockSize =
        EnableWaveGroup ? DEFAULT_WAVEGROUP_BLOCKSIZE : DEFAULT_BLOCKSIZE;

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

    constexpr ck::index_t n_dim       = 1;
    constexpr ck::index_t group_count = 1;
    constexpr ck::index_t n_batch     = 1;
    const ck::index_t n_out_channels  = OutputChannels;
    const ck::index_t n_in_channels   = InputChannels;

    constexpr ck::index_t DilationSize = Dilation ? 2 : 1;

    const std::vector<ck::index_t> filters_1x1{1};
    const std::vector<ck::index_t> dilations_1{1};
    const std::vector<ck::index_t> dilations_2{2};
    const std::vector<ck::index_t> pads_0{0};

    const std::vector<ck::index_t>& filters_len = filters_1x1;
    const std::vector<ck::index_t> input_len    = {Width};
    const std::vector<ck::index_t> strides_1{1};
    const std::vector<ck::index_t>& strides    = strides_1;
    const std::vector<ck::index_t>& dilations  = Dilation ? dilations_2 : dilations_1;
    const std::vector<ck::index_t>& left_pads  = pads_0;
    const std::vector<ck::index_t>& right_pads = pads_0;

    constexpr bool InEnableLds  = LdsMode & 1 ? true : false;
    constexpr bool WeiEnableLds = LdsMode & 2 ? true : false;
    constexpr bool AccEnableLds = LdsMode & 4 ? true : false;
    constexpr bool EnableAsync  = LdsMode & 8 ? true : false;

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

    namespace ctc   = ck::tensor_layout::convolution;
    auto in_layout  = ctc::GNWC{};
    auto wei_layout = ctc::GKXC{};
    auto out_layout = ctc::GNWK{};
    using InLayout  = decltype(in_layout);
    using WeiLayout = decltype(wei_layout);
    using OutLayout = decltype(out_layout);

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<GPUAccType> out_host(out_g_n_k_wos_desc);
    Tensor<GPUAccType> out_device(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        break;
    }

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

    constexpr ck::index_t CPerWcnn_Bhalf = (Shape == Shape_4X2) ? 8 : 4;

    constexpr ck::long_index_t Acc_Convert_Interval =
        std::is_same<GPUAccType, ck::bhalf_t>::value ? CPerWcnn_Bhalf : CPerBlock;

    using RefConvType = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                     InDataType,
                                                                     WeiDataType,
                                                                     GPUAccType,
                                                                     InElementOp,
                                                                     WeiElementOp,
                                                                     PassThrough>;
    RefConvType ref_conv{};

    auto ref_invoker  = ref_conv.MakeInvoker();
    auto ref_argument = ref_conv.MakeArgument(in,
                                              wei,
                                              out_host,
                                              conv_param.conv_filter_strides_,
                                              conv_param.conv_filter_dilations_,
                                              conv_param.input_left_pads_,
                                              conv_param.input_right_pads_,
                                              in_element_op,
                                              wei_element_op,
                                              PassThrough{},
                                              {},
                                              {},
                                              {},
                                              Acc_Convert_Interval);

    if(config.do_verification)
    {
        ref_invoker.Run(ref_argument);
    }

    dump_tensor(in, "Input");
    dump_tensor(wei, "Weight");
    dump_tensor(out_host, "Accum");

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(GPUAccType) * out_device.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());

    // do Conv
    static constexpr auto ConvSpec =
        (FilterSize == 1)
            ? ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0
            : ((FilterSize == 2) ? ck::tensor_operation::device::ConvolutionForwardSpecialization::
                                       Filter2x2Stride2Pad0
                                 : ck::tensor_operation::device::ConvolutionForwardSpecialization::
                                       Filter3x3Stride1Pad0);
    //? ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter2x2Stride2OddHWPad0
    //: ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0;

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

    constexpr ck::index_t AccBlockTransferScalarPerVector = sizeof(uint32_t) / sizeof(GPUAccType);
    constexpr ck::index_t Cluster_Acc_K = KPerBlock / AccBlockTransferScalarPerVector;
    constexpr ck::index_t Cluster_Acc_W = 4;
    constexpr ck::index_t Cluster_Acc_H = ActiveBlockSize / Cluster_Acc_K / Cluster_Acc_W;
    using AccBlockTransferClusterLengths =
        ck::Sequence<Cluster_Acc_H, Cluster_Acc_W, Cluster_Acc_K>;
    using EmptyTuple = ck::Tuple<>;
    float avg_time   = 0;
    using DeviceConvFwdInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Wcnn_CShuffle<
            NDimSpatial,
#ifdef ENABLE_CONST_LAYOUT
            ConstInputLayout<FilterSize>,
            ConstWeightLayout<FilterSize>,
            EmptyTuple,
            ConstOutputLayout<FilterSize>,
#else
            InputLayout<NDimSpatial>,
            WeightLayout<NDimSpatial>,
            EmptyTuple,
            OutputLayout<NDimSpatial>,
#endif
            InDataType,
            WeiDataType,
            EmptyTuple,
            GPUAccType,
            GPUAccType,
            InElementOp,
            WeiElementOp,
            PassThrough,
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
            EmptyTuple,
            ck::Sequence<>,
            ck::Sequence<>,
            false,
            true,
            AccBlockTransferClusterLengths,
            AccBlockTransferScalarPerVector,
            AccEnableLds,
            EnableAsync,
            EnableWaveGroup>;

    auto conv     = DeviceConvFwdInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(in_device_buf.GetDeviceBuffer(),
                                      wei_device_buf.GetDeviceBuffer(),
                                      std::array<const void*, 0>{},
                                      out_device_buf.GetDeviceBuffer(),
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
                                      PassThrough{});

    if(!conv.IsSupportedArgument(argument))
    {
        std::cout << "wrong! device_conv with the specified compilation parameters does "
                     "not support this Conv problem"
                  << std::endl;
        return false;
    }
    avg_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});
    out_device_buf.FromDevice(out_device.mData.data());

    dump_tensor(out_device, "Accum_Device");

    std::cout <<
#ifdef ENABLE_WAVEGROUP
        "grouped_conv_fwd_wcnn_wavegroup_1d<In/Wei:"
#else
        "grouped_conv_fwd_wcnn_1d<In/Wei:"
#endif
              << get_string<InDataType>() << ", Out:" << get_string<GPUAccType>() << ", "
              << get_string(Shape) << ", " << get_string(Filter) << ", Dilation:" << DilationSize
              << ", LdsMod:" << LdsMode << ", WaveGroup:" << EnableWaveGroup << ", Id : 0x"
              << std::hex << TestMask << " Size: { " << config.h << "x" << config.w << "x"
              << config.c << "x" << config.k << " }>: Status: ";

    if(config.time_kernel)
    {
        std::cout << "Execute Time: " << avg_time << " ";
    }

    if(config.do_verification)
    {
        bool ret = ck::utils::check_err(out_device,
                                        out_host,
                                        "Error: incorrect results!",
                                        get_rtol<GPUAccType>(),
                                        get_atol<GPUAccType>());
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

template <typename SrcType, typename GPUAccType, int LdsMode, int32_t TestMask>
bool run_test_fmt()
{
    if((config.test_mask & TestMask) == 0)
    {
        return true;
    }
    bool pass = true;

#ifdef ENABLE_WAVEGROUP
    constexpr bool WaveGroup = true;
#else
    constexpr bool WaveGroup = false;
#endif
    // clang-format off
    {
        //                                                           |ShapeType  |FilterType |Dilation |Lds |WaveGroup |TestMask
        if constexpr (std::is_same<GPUAccType, float>::value || std::is_same<GPUAccType, int32_t>::value)
        {
            pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, TestMask | 0x10000>();
            pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, TestMask | 0x10000>();
        }
        else
        {
            pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, TestMask | 0x10000>();
            // TODO: fix it.
            pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_1X1, false, 0,       WaveGroup, TestMask | 0x20000>();
            pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_1X1, false, 0,       WaveGroup, TestMask | 0x40000>();

            pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, TestMask | 0x2000000>();
            pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_1X1, false, LdsMode, WaveGroup, TestMask | 0x4000000>();
            pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_1X1, false, LdsMode, WaveGroup, TestMask | 0x8000000>();
        }

    }
    // clang-format on
    return pass;
}

int main(int argc, char* argv[])
{
    bool pass = true;

    if(parse_cmd_args(argc, argv, config) == false)
    {
        return -1;
    }

    // clang-format off
    //                |SrcType |GPUAccType | LdsMode |TestMask
    pass &= run_test_fmt<half_t,  float,   0x7, 0x1  >();
    pass &= run_test_fmt<bhalf_t, float,   0xf, 0x2  >();
    pass &= run_test_fmt<f8_t,    float,   0x5, 0x4  >();
    pass &= run_test_fmt<bf8_t,   float,   0x6, 0x8  >();
    pass &= run_test_fmt<int8_t,  float,   0xd, 0x10 >();
    pass &= run_test_fmt<int8_t,  int32_t, 0xe, 0x20 >();
    pass &= run_test_fmt<half_t,  half_t,  0xf, 0x40 >();
    pass &= run_test_fmt<bhalf_t, bhalf_t, 0x7, 0x80 >();
    pass &= run_test_fmt<f8_t,    half_t,  0x3, 0x100>();
    pass &= run_test_fmt<bf8_t,   half_t,  0xb, 0x200>();
    pass &= run_test_fmt<int8_t,  half_t,  0x9, 0x400>();
    // clang-format on

    std::cout << "grouped_conv_fwd_wcnn_1d: ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
