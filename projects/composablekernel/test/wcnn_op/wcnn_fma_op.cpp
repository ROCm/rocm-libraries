// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
#include "common_wcnn.hpp"

#define DEFAULT_K 32
#define DEFAULT_C 32
#define DEFAULT_W 8
#define DEFAULT_H 16

namespace ck {

namespace conv_op_util {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
template <typename InDataType,
          typename AccDataType,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t Width,
          index_t Height,
          index_t UnpackedInputChannels,
          index_t UnpackedOutputChannels>
__global__ void __launch_bounds__(64, 1) conv_fma(const InDataType* in_, AccDataType* c_)
{
    constexpr ck::WcnnFmaFromTensor<InDataType, AccDataType, HPerWcnn, WPerWcnn> wcnn_fma;
    constexpr index_t Aco    = wcnn_fma.GetAccumChannelOrder();
    constexpr auto wcnn_conv = ck::
        WcnnConv<InDataType, InDataType, AccDataType, HPerWcnn, WPerWcnn, 1, 1, 1, 1, false, Aco>();

    auto in = reinterpret_cast<const typename decltype(wcnn_conv)::KernelInDataType*>(in_);

    static_assert(Width % WPerWcnn == 0, "");
    static_assert(Height % HPerWcnn == 0, "");
    static_assert(UnpackedInputChannels % wcnn_conv.GetUnpackedNumInputChannels() == 0, "");
    static_assert(UnpackedOutputChannels % wcnn_conv.GetUnpackedNumOutputChannels() == 0, "");
    static_assert(UnpackedInputChannels == UnpackedOutputChannels, "");

    constexpr index_t DataTileHeight = 4;
    constexpr index_t InputChannels  = UnpackedInputChannels * wcnn_conv.GetNumInputChannels() /
                                      wcnn_conv.GetUnpackedNumInputChannels();
    constexpr index_t OutputChannels = UnpackedOutputChannels * wcnn_conv.GetNumOutputChannels() /
                                       wcnn_conv.GetUnpackedNumOutputChannels();

    constexpr index_t HRepeat = Height / HPerWcnn;
    constexpr index_t WRepeat = Width / WPerWcnn;
    constexpr index_t CRepeat = UnpackedInputChannels / wcnn_conv.GetUnpackedNumInputChannels();
    constexpr index_t KRepeat = UnpackedOutputChannels / wcnn_conv.GetUnpackedNumOutputChannels();
    constexpr index_t AccVectorCount = 1;

    using InDataVec     = typename decltype(wcnn_conv)::InDataVec::type;
    using InDataTileVec = typename decltype(wcnn_conv)::InDataTileVec::type;
    using AccDataVec    = typename decltype(wcnn_conv)::AccDataVec;
    using AccVec        = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                                    AccDataType,
                                                    AccVectorCount,
                                                    wcnn_conv.GetNumAccumComponents(),
                                                    true>;
    const int lIdx      = threadIdx.x;

    // Data layout: HWC, unit: InDataType
    constexpr index_t data_H_stride = Width * InputChannels;
    constexpr index_t data_W_stride = InputChannels;
    constexpr index_t data_C_stride = wcnn_conv.GetNumInputChannels();

    // Acc layout: HWK, unit: AccDataType
    constexpr index_t acc_H_stride = Width * OutputChannels;
    constexpr index_t acc_W_stride = OutputChannels;
    constexpr index_t acc_K_stride = wcnn_conv.GetNumOutputChannels();

    // Load inData
    const index_t compIdx =
        lIdx * wcnn_conv.GetNumDataComponents() / wcnn_conv.GetNumImageSubTilesInVertical();

    const index_t accCompIdx =
        lIdx * wcnn_conv.GetNumAccumComponents() / wcnn_conv.GetNumSubTilesPerImageTile();

    auto load_in_data = [&](index_t h, index_t w, index_t c) {
        const index_t subC = compIdx % wcnn_conv.GetNumInputChannels();
        const index_t subW = (compIdx / wcnn_conv.GetNumInputChannels()) % WPerWcnn;
        const index_t subH = (compIdx / wcnn_conv.GetNumInputChannels()) / WPerWcnn;

        const index_t offset = (h * HPerWcnn + subH) * data_H_stride +
                               (w * WPerWcnn + subW) * data_W_stride + (c * data_C_stride + subC);
        if constexpr(wcnn_conv.GetNumImageSubTilesInVertical() > 1)
        {
            // shape 8x4
            typename decltype(wcnn_conv)::InDataVec inDataVec;

            static_for<0, wcnn_conv.GetNumImageSubTilesInVertical(), 1>{}([&](auto tileId) {
                inDataVec.template AsType<InDataTileVec>()(Number<tileId>{}) =
                    *reinterpret_cast<const InDataTileVec*>(
                        in + offset + tileId * DataTileHeight * data_H_stride);
            });
            return inDataVec.template AsType<InDataVec>()(Number<0>{});
        }
        else
        {
            return *reinterpret_cast<const InDataVec*>(in + offset);
        }
    };

    auto store_acc_data = [&](index_t h, index_t w, index_t k, AccDataVec& c_vec) {
        const index_t subW = (accCompIdx / wcnn_conv.GetNumOutputChannels()) % WPerWcnn;
        const index_t subH = (accCompIdx / wcnn_conv.GetNumOutputChannels()) / WPerWcnn;

        static_assert(wcnn_conv.GetNumAccumComponents() >= 4);
        if constexpr(Aco)
        {
            // Aco = 1, do swizzle after 2 channels.
            using AccSwizzleVec              = typename vector_type<AccDataType, 2>::type;
            constexpr index_t NumLanePerPair = (WPerWcnn == 2) && (HPerWcnn == 4) ? 4 : 2;

            const index_t subK = accCompIdx % wcnn_conv.GetNumOutputChannels() /
                                 (wcnn_conv.GetNumAccumComponents() * NumLanePerPair) *
                                 (wcnn_conv.GetNumAccumComponents() * NumLanePerPair);
            const index_t offset = (h * HPerWcnn + subH) * acc_H_stride +
                                   (w * WPerWcnn + subW) * acc_W_stride + (k * acc_K_stride + subK);
            constexpr index_t secOffset =
                NumLanePerPair * 2; // 8 channel for 4x2 and 4 channel for others.
            const index_t swizzleOffset = (lIdx & (NumLanePerPair - 1)) * 2;
            *reinterpret_cast<AccSwizzleVec*>(c_ + offset + swizzleOffset) =
                c_vec.template AsType<AccSwizzleVec>()(Number<0>{});
            *reinterpret_cast<AccSwizzleVec*>(c_ + offset + swizzleOffset + secOffset) =
                c_vec.template AsType<AccSwizzleVec>()(Number<1>{});
            if constexpr(wcnn_conv.GetNumAccumComponents() == 8)
            {
                constexpr index_t tileOffset =
                    wcnn_conv.GetNumSubTilesPerImageTile() == 2 // 8x4
                        ? HPerWcnn / wcnn_conv.GetNumSubTilesPerImageTile() * acc_H_stride
                        : 8;

                *reinterpret_cast<AccSwizzleVec*>(c_ + offset + swizzleOffset + tileOffset) =
                    c_vec.template AsType<AccSwizzleVec>()(Number<2>{});
                *reinterpret_cast<AccSwizzleVec*>(c_ + offset + swizzleOffset + secOffset +
                                                  tileOffset) =
                    c_vec.template AsType<AccSwizzleVec>()(Number<3>{});
            }
        }
        else
        {
            if constexpr(wcnn_conv.GetNumAccumComponents() == 4)
            {
                const index_t subK   = accCompIdx % wcnn_conv.GetNumOutputChannels();
                const index_t offset = (h * HPerWcnn + subH) * acc_H_stride +
                                       (w * WPerWcnn + subW) * acc_W_stride +
                                       (k * acc_K_stride + subK);
                *reinterpret_cast<typename AccDataVec::type*>(c_ + offset) =
                    c_vec.template AsType<typename AccDataVec::type>()(Number<0>{});
            }
            else
            {
                static_assert(wcnn_conv.GetNumAccumComponents() == 8, "unexpected value");
                // Aco = 0, do swizzle after 4 channels.
                using AccSwizzleVec = typename vector_type<AccDataType, 4>::type;
                const index_t subK  = accCompIdx % wcnn_conv.GetNumOutputChannels() /
                                     (wcnn_conv.GetNumAccumComponents() * 2) *
                                     (wcnn_conv.GetNumAccumComponents() * 2);
                const index_t offset = (h * HPerWcnn + subH) * acc_H_stride +
                                       (w * WPerWcnn + subW) * acc_W_stride +
                                       (k * acc_K_stride + subK);
                index_t secOffset = 8;
                if constexpr(wcnn_conv.GetNumSubTilesPerImageTile() > 1)
                {
                    secOffset = HPerWcnn / wcnn_conv.GetNumSubTilesPerImageTile() * acc_H_stride;
                }
                const index_t swizzleOffset = (lIdx & 1) * 4;
                *reinterpret_cast<AccSwizzleVec*>(c_ + offset + swizzleOffset) =
                    c_vec.template AsType<AccSwizzleVec>()(Number<0>{});
                *reinterpret_cast<AccSwizzleVec*>(c_ + offset + swizzleOffset + secOffset) =
                    c_vec.template AsType<AccSwizzleVec>()(Number<1>{});
            }
        }
    };

    __syncthreads();
    constexpr index_t numAccum = wcnn_fma.GetNumAccum();
    static_assert(KRepeat % numAccum == 0);
    // Do conv fma
    static_for<0, HRepeat, 1>{}([&](auto h) {
        static_for<0, WRepeat, 1>{}([&](auto w) {
            static_for<0, KRepeat, numAccum>{}([&](auto k) {
                AccVec c_thread_buf_            = {};
                auto& c_vec                     = c_thread_buf_.GetVectorTypeReference(Number<0>{});
                typename AccDataVec::type accum = {};
                typename AccDataVec::type scale = {};
                static_for<0, wcnn_conv.GetNumAccumComponents(), 1>{}(
                    [&](auto s) { scale[s.value] = type_convert<AccDataType>(1.0f); });

                InDataVec in_tile_data1 = {};
                InDataVec in_tile_data2 = {};
                if constexpr(wcnn_fma.GetNumResidual() == 4)
                {
                    InDataVec in_tile_data3 = {};
                    InDataVec in_tile_data4 = {};
                    static_assert(CRepeat == 4 * KRepeat);
                    constexpr index_t c = k * 4;
                    in_tile_data1       = load_in_data(h, w, c);
                    in_tile_data2       = load_in_data(h, w, c + 1);
                    in_tile_data3       = load_in_data(h, w, c + 2);
                    in_tile_data4       = load_in_data(h, w, c + 3);
                    wcnn_fma.fma_instr.Run(accum,
                                           in_tile_data1,
                                           in_tile_data2,
                                           in_tile_data3,
                                           in_tile_data4,
                                           scale,
                                           c_vec);
                }
                else if constexpr(wcnn_fma.GetNumResidual() == 2)
                {
                    static_assert(CRepeat == 2 * KRepeat);
                    constexpr index_t c = k * 2;
                    in_tile_data1       = load_in_data(h, w, c);
                    in_tile_data2       = load_in_data(h, w, c + 1);
                    wcnn_fma.fma_instr.Run(accum, in_tile_data1, in_tile_data2, scale, c_vec);
                }
                else if constexpr(wcnn_fma.GetNumResidual() == 1)
                {
                    constexpr index_t c = k / numAccum;
                    in_tile_data1       = load_in_data(h, w, c);
                    wcnn_fma.fma_instr.Run(accum, in_tile_data1, scale, c_vec);
                }

                store_acc_data(h, w, k, c_vec);

                if constexpr(numAccum == 2)
                {
                    constexpr index_t ChanOff = (HPerWcnn == 8) && (WPerWcnn == 4) ? 8 : 16;

                    constexpr ck::
                        WcnnFmaFromTensor<InDataType, AccDataType, HPerWcnn, WPerWcnn, ChanOff>
                            wcnn_fma2;
                    wcnn_fma2.fma_instr.Run(accum, in_tile_data1, scale, c_vec);
                    store_acc_data(h, w, k + 1, c_vec);
                }
            });
        });
    });
}
#pragma clang diagnostic pop
}; // namespace conv_op_util
}; // namespace ck

template <typename InDataType, typename GPUAccType, ShapeType Shape, int32_t TestMask>
bool run_test()
{
    if((config.test_mask & 0xFFFF0000 & TestMask) == 0)
    {
        return true;
    }
    constexpr ck::index_t HPerWcnn = (Shape == Shape_8X4) ? 8 : 4;
    constexpr ck::index_t WPerWcnn = (Shape == Shape_4X2) ? 2 : 4;

    constexpr ck::index_t Width          = DEFAULT_W;
    constexpr ck::index_t Height         = DEFAULT_H;
    constexpr ck::index_t InputChannels  = DEFAULT_C;
    constexpr ck::index_t OutputChannels = DEFAULT_K;

    constexpr ck::index_t n_dim          = 2;
    constexpr ck::index_t group_count    = 1;
    constexpr ck::index_t n_batch        = 1;
    constexpr ck::index_t n_out_channels = OutputChannels;
    constexpr ck::index_t n_in_channels  = InputChannels;

    const std::vector<ck::index_t> filters_1x1{1, 1};
    const std::vector<ck::index_t> dilations_1{1, 1};
    const std::vector<ck::index_t> pads_0{0, 0};

    const std::vector<ck::index_t> input_len = {Height, Width};
    const std::vector<ck::index_t> strides{1, 1};

    ck::utils::conv::ConvParam conv_param{n_dim,
                                          group_count,
                                          n_batch,
                                          n_out_channels,
                                          n_in_channels,
                                          filters_1x1,
                                          input_len,
                                          strides,
                                          dilations_1,
                                          pads_0,
                                          pads_0};

    constexpr auto NDimSpatial = ck::Number<n_dim>{};

    namespace ctc   = ck::tensor_layout::convolution;
    auto in_layout  = ctc::GNHWC{};
    auto out_layout = ctc::GNHWK{};
    using InLayout  = decltype(in_layout);
    using OutLayout = decltype(out_layout);

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<GPUAccType> out_host(out_g_n_k_wos_desc);
    Tensor<GPUAccType> out_device(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

    switch(config.init_method)
    {
    case 0: break;
    case 1: in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}); break;
    default: in.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0}); break;
    }

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    dump_tensor(in, "Input");

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(GPUAccType) * out_device.mDesc.GetElementSpaceSize());

    if constexpr(std::is_same<ck::int4_t, InDataType>::value)
    {
        std::vector<uint8_t> in_packed;
        in_packed.resize(in.mData.size());
        for(size_t i = 0; i < in.mData.size(); i += 2)
        {
            uint8_t val0 = (in.mData[i] & 0xf);
            uint8_t val1 = (in.mData[i + 1] & 0xf);

            in_packed[i / 2] = val0 | (val1 << 4);
        }
        in_device_buf.ToDevice(in_packed.data());
    }
    else
    {
        in_device_buf.ToDevice(in.mData.data());
    }

    auto conv_kernel = ck::conv_op_util::conv_fma<InDataType,
                                                  GPUAccType,
                                                  HPerWcnn,
                                                  WPerWcnn,
                                                  Width,
                                                  Height,
                                                  InputChannels,
                                                  OutputChannels>;
    conv_kernel<<<1, 32>>>(static_cast<const InDataType*>(in_device_buf.GetDeviceBuffer()),
                           static_cast<GPUAccType*>(out_device_buf.GetDeviceBuffer()));

    out_device_buf.FromDevice(out_device.mData.data());

    dump_tensor(out_device, "Fma_Device");
    std::cout << "wcnn_fma_op<In/Wei:" << get_string<InDataType>()
              << ", Out:" << get_string<GPUAccType>() << ", " << get_string(Shape) << ", Id:0x"
              << std::hex << TestMask << ">: Status: ";

    if(config.do_verification)
    {
        auto in2 = in.template CopyAsType<GPUAccType>();
        bool ret = ck::utils::check_err(out_device,
                                        in2,
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

template <typename SrcType, typename GPUAccType, int32_t TestMask>
bool run_test_fmt()
{
    if((config.test_mask & TestMask) == 0)
    {
        return true;
    }
    bool pass = true;
    // clang-format off
    //                                                        |ShapeType |TestMask
    if constexpr(std::is_same<GPUAccType, float>::value || std::is_same<GPUAccType, int32_t>::value)
    {
        pass &= run_test<SrcType, GPUAccType, Shape_4X2, TestMask | 0x10000  >();
    }
    else
    {
        pass &= run_test<SrcType, GPUAccType, Shape_4X2, TestMask | 0x80000  >();
        pass &= run_test<SrcType, GPUAccType, Shape_4X4, TestMask | 0x100000 >();
        pass &= run_test<SrcType, GPUAccType, Shape_8X4, TestMask | 0x200000 >();
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
    //                  |SrcType     |GPUAccType
    pass &= run_test_fmt<ck::half_t,  float,       0x1   >();
    pass &= run_test_fmt<ck::bhalf_t, float,       0x2   >();
    pass &= run_test_fmt<ck::f8_t,    float,       0x4   >();
    pass &= run_test_fmt<ck::bf8_t,   float,       0x8   >();
    pass &= run_test_fmt<int8_t,      float,       0x10  >();

    pass &= run_test_fmt<ck::half_t,  ck::half_t,  0x40  >();
    pass &= run_test_fmt<ck::bhalf_t, ck::bhalf_t, 0x80  >();
    pass &= run_test_fmt<ck::f8_t,    ck::half_t,  0x100 >();
    pass &= run_test_fmt<ck::bf8_t,   ck::half_t,  0x200 >();
    pass &= run_test_fmt<int8_t,      ck::half_t,  0x400 >();

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    pass &= run_test_fmt<ck::int4_t,  float,       0x800 >();
    pass &= run_test_fmt<ck::int4_t,  ck::bhalf_t, 0x1000>();
    pass &= run_test_fmt<ck::int4_t,  ck::half_t,  0x2000>();
#endif
    // clang-format on
    std::cout << "wcnn_fma_op: ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
