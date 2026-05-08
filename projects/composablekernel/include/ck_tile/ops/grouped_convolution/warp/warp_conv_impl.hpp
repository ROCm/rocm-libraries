
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/grouped_convolution/warp/warp_conv_intrinsic.hpp"

namespace ck_tile {

// Trait for AccWarpDstrEncoding, specialized by AcoFlag and tile size
template <bool AcoFlag, index_t HPerWcnn, index_t WPerWcnn, index_t OutputChannelNum>
struct WcnnAccWarpDstrEncodingTrait;

// 8x4 tile
template <bool AcoFlag, index_t OutputChannelNum>
struct WcnnAccWarpDstrEncodingTrait<AcoFlag, 8, 4, OutputChannelNum>
{
    static_assert(OutputChannelNum == 8, "OutputChannelNum should be 8 for 8x4 tile");
    using type =
        std::conditional_t<AcoFlag,
                           tile_distribution_encoding<sequence<>,
                                                      tuple<sequence<2, 16>, sequence<2, 2, 2>>,
                                                      tuple<sequence<1, 2>>,
                                                      tuple<sequence<1, 1>>,
                                                      sequence<1, 2, 2>,
                                                      sequence<0, 0, 2>>,
                           tile_distribution_encoding<sequence<>,
                                                      tuple<sequence<2, 16>, sequence<2, 4>>,
                                                      tuple<sequence<1, 2>>,
                                                      tuple<sequence<1, 0>>,
                                                      sequence<1, 2>,
                                                      sequence<0, 1>>>;
};

// 4x4 tile
template <bool AcoFlag, index_t OutputChannelNum>
struct WcnnAccWarpDstrEncodingTrait<AcoFlag, 4, 4, OutputChannelNum>
{
    static_assert(OutputChannelNum == 16, "OutputChannelNum should be 16 for 4x4 tile");
    using type =
        std::conditional_t<AcoFlag,
                           tile_distribution_encoding<sequence<>,
                                                      tuple<sequence<16>, sequence<2, 2, 2, 2>>,
                                                      tuple<sequence<1, 2>>,
                                                      tuple<sequence<0, 2>>,
                                                      sequence<2, 2, 2>,
                                                      sequence<0, 1, 3>>,
                           tile_distribution_encoding<sequence<>,
                                                      tuple<sequence<16>, sequence<2, 2, 4>>,
                                                      tuple<sequence<1, 2>>,
                                                      tuple<sequence<0, 1>>,
                                                      sequence<2, 2>,
                                                      sequence<0, 2>>>;
};

// 4x2 tile
template <bool AcoFlag, index_t OutputChannelNum>
struct WcnnAccWarpDstrEncodingTrait<AcoFlag, 4, 2, OutputChannelNum>
{
    static_assert(OutputChannelNum == 16, "OutputChannelNum should be 16 for 4x2 tile");
    using type =
        std::conditional_t<AcoFlag,
                           tile_distribution_encoding<sequence<>,
                                                      tuple<sequence<8>, sequence<2, 4, 2>>,
                                                      tuple<sequence<1, 2>>,
                                                      tuple<sequence<0, 1>>,
                                                      sequence<2, 2>,
                                                      sequence<0, 2>>,
                           tile_distribution_encoding<sequence<>,
                                                      tuple<sequence<8>, sequence<4, 4>>,
                                                      tuple<sequence<1, 2>>,
                                                      tuple<sequence<0, 0>>,
                                                      sequence<2>,
                                                      sequence<1>>>;
};

// Common Img/Acc distribution encodings shared across warp conv implementations
template <typename ImgDataType,
          typename InAccDataType,
          typename OutAccDataType,
          bool AcoFlag,
          index_t HPerWcnn_,
          index_t WPerWcnn_,
          index_t NumIter_>
struct WcnnConvImgAccBase
{
    static constexpr index_t HPerWcnn = HPerWcnn_;
    static constexpr index_t WPerWcnn = WPerWcnn_;
    static constexpr index_t NumIter  = NumIter_;

    static constexpr index_t ImgChannelsPerVgpr =
        sizeof(int32_t) / sizeof(ImgDataType) * numeric_traits<ImgDataType>::PackedSize;

    static constexpr index_t OutputChannelNum = (HPerWcnn == 8 && WPerWcnn == 4) ? 8 : 16;

    static constexpr index_t InputChannelNum =
        ((HPerWcnn == 4 && WPerWcnn == 2) ? 4 : 2) * ImgChannelsPerVgpr;

    using ImgWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<HPerWcnn / 4, 4, WPerWcnn>,
              sequence<NumIter, InputChannelNum / ImgChannelsPerVgpr, ImgChannelsPerVgpr>>,
        tuple<sequence<1, 1, 2>>,
        tuple<sequence<1, 2, 1>>,
        sequence<2, 1, 2>,
        sequence<0, 0, 2>>;

    // based on ACO setting and tile size
    using AccWarpDstrEncoding =
        typename WcnnAccWarpDstrEncodingTrait<AcoFlag, HPerWcnn, WPerWcnn, OutputChannelNum>::type;

    using ImgDataVecType =
        ext_vector_t<ext_vector_t<ImgDataType, ImgChannelsPerVgpr * HPerWcnn / 4>, NumIter>;

    // 32 is warp_num
    static constexpr index_t ElementsPerAcc = HPerWcnn * WPerWcnn * OutputChannelNum / 32;

    using InAccDataVecType  = ext_vector_t<InAccDataType, ElementsPerAcc>;
    using OutAccDataVecType = ext_vector_t<OutAccDataType, ElementsPerAcc>;
};

template <typename ImgDataType,
          typename WeiDataType,
          typename InAccDataType,
          typename OutAccDataType,
          bool AcoFlag,
          index_t HPerWcnn_,
          index_t WPerWcnn_,
          index_t NumIter_ = 1>
struct Wcnn1x1ConvImpl : WcnnConvImgAccBase<ImgDataType,
                                            InAccDataType,
                                            OutAccDataType,
                                            AcoFlag,
                                            HPerWcnn_,
                                            WPerWcnn_,
                                            NumIter_>
{
    using Base = WcnnConvImgAccBase<ImgDataType,
                                    InAccDataType,
                                    OutAccDataType,
                                    AcoFlag,
                                    HPerWcnn_,
                                    WPerWcnn_,
                                    NumIter_>;

    using Base::HPerWcnn;
    using Base::ImgChannelsPerVgpr;
    using Base::InputChannelNum;
    using Base::NumIter;
    using Base::OutputChannelNum;
    using Base::WPerWcnn;

    static constexpr index_t FilterSizeY = 1;
    static constexpr index_t FilterSizeX = 1;
    static constexpr index_t DilationY   = 1;
    static constexpr index_t DilationX   = 1;
    static constexpr index_t NumIter     = NumIter_;

    static constexpr index_t WeiChannelsPerVgpr =
        sizeof(int32_t) / sizeof(WeiDataType) * numeric_traits<WeiDataType>::PackedSize;

    // most of time WeiKernelNum = OutputChannelNum, but for 8x4 tile uses 16 WeiKernelNum and will
    // use StartLane16 to select which 8 kernels to use
    static constexpr index_t WeiKernelNum = 16;

    static constexpr index_t WeiChannelPackNum = InputChannelNum / 2 / WeiChannelsPerVgpr;

    using WeiWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<WeiKernelNum>, sequence<NumIter, WeiChannelPackNum, 2, WeiChannelsPerVgpr>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<0, 2>>,
        sequence<2, 2, 2>,
        sequence<0, 1, 3>>;

    using WeiDataVecType =
        ext_vector_t<WeiDataType, WeiChannelsPerVgpr * WeiChannelPackNum * NumIter>;

    using typename Base::ImgDataVecType;
    using typename Base::InAccDataVecType;
    using typename Base::OutAccDataVecType;

    using Intrinsic = WarpConvIntrinsic<ImgDataType,
                                        InAccDataType,
                                        OutAccDataType,
                                        AcoFlag,
                                        HPerWcnn,
                                        WPerWcnn,
                                        FilterSizeY,
                                        FilterSizeX,
                                        DilationY,
                                        DilationX,
                                        NumIter>;

    template <typename... ImgVecs>
    CK_TILE_DEVICE OutAccDataVecType operator()(InAccDataVecType acc,
                                                WeiDataVecType wei,
                                                ImgVecs... imgs) const
    {
        return Intrinsic::call(acc, wei, imgs...);
    }
};

}; // namespace ck_tile
