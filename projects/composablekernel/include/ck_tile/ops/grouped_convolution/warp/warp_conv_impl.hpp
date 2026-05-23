
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
    using type = std::conditional_t<
        AcoFlag,
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<2, 4>, sequence<4>, sequence<2, 2, 2>>,
                                   tuple<sequence<1, 2, 3>>,
                                   tuple<sequence<1, 0, 1>>,
                                   sequence<1, 3, 3>,
                                   sequence<0, 0, 2>>,
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<2, 4>, sequence<4>, sequence<2, 4>>,
                                   tuple<sequence<1, 2, 3>>,
                                   tuple<sequence<1, 0, 0>>,
                                   sequence<1, 3>,
                                   sequence<0, 1>>>;
};

// 4x4 tile
template <bool AcoFlag, index_t OutputChannelNum>
struct WcnnAccWarpDstrEncodingTrait<AcoFlag, 4, 4, OutputChannelNum>
{
    static_assert(OutputChannelNum == 16, "OutputChannelNum should be 16 for 4x4 tile");
    using type = std::conditional_t<
        AcoFlag,
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<4>, sequence<4>, sequence<2, 2, 2, 2>>,
                                   tuple<sequence<1, 2, 3>>,
                                   tuple<sequence<0, 0, 2>>,
                                   sequence<3, 3, 3>,
                                   sequence<0, 1, 3>>,
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<4>, sequence<4>, sequence<2, 2, 4>>,
                                   tuple<sequence<1, 2, 3>>,
                                   tuple<sequence<0, 0, 1>>,
                                   sequence<3, 3>,
                                   sequence<0, 2>>>;
};

// 4x2 tile
template <bool AcoFlag, index_t OutputChannelNum>
struct WcnnAccWarpDstrEncodingTrait<AcoFlag, 4, 2, OutputChannelNum>
{
    static_assert(OutputChannelNum == 16, "OutputChannelNum should be 16 for 4x2 tile");
    using type = std::conditional_t<
        AcoFlag,
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<4>, sequence<2>, sequence<2, 4, 2>>,
                                   tuple<sequence<1, 2, 3>>,
                                   tuple<sequence<0, 0, 1>>,
                                   sequence<3, 3>,
                                   sequence<0, 2>>,
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<4>, sequence<2>, sequence<4, 4>>,
                                   tuple<sequence<1, 2, 3>>,
                                   tuple<sequence<0, 0, 0>>,
                                   sequence<3>,
                                   sequence<1>>>;
};

// Common Img/Acc distribution encodings shared across warp conv implementations
template <typename ADataType,
          typename AccDataType,
          bool AcoFlag,
          index_t HPerWcnn_,
          index_t WPerWcnn_,
          index_t NumIter_>
struct WcnnConvImgAccBase
{
    static constexpr index_t HPerWcnn = HPerWcnn_;
    static constexpr index_t WPerWcnn = WPerWcnn_;
    static constexpr index_t NumIter  = NumIter_;

    static constexpr index_t AChannelsPerVgpr =
        sizeof(int32_t) / sizeof(ADataType) * numeric_traits<ADataType>::PackedSize;

    static constexpr index_t OutputChannelNum = (HPerWcnn == 8 && WPerWcnn == 4) ? 8 : 16;

    static constexpr index_t InputChannelNum =
        ((HPerWcnn == 4 && WPerWcnn == 2) ? 4 : 2) * AChannelsPerVgpr;

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<HPerWcnn / 4, 4>,
              sequence<WPerWcnn>,
              sequence<NumIter, InputChannelNum / AChannelsPerVgpr, AChannelsPerVgpr>>,
        tuple<sequence<1, 2, 3>>,
        tuple<sequence<1, 0, 1>>,
        sequence<3, 1, 3>,
        sequence<0, 0, 2>>;

    // based on ACO setting and tile size
    using AccWarpDstrEncoding =
        typename WcnnAccWarpDstrEncodingTrait<AcoFlag, HPerWcnn, WPerWcnn, OutputChannelNum>::type;

    using ADataVecType = ext_vector_t<ADataType, AChannelsPerVgpr * HPerWcnn / 4 * NumIter>;

    // 32 is warp_num
    static constexpr index_t ElementsPerAcc = HPerWcnn * WPerWcnn * OutputChannelNum / 32;

    using AccDataVecType = ext_vector_t<AccDataType, ElementsPerAcc>;
};

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          bool AcoFlag,
          index_t HPerWcnn_,
          index_t WPerWcnn_,
          index_t NumIter_ = 1>
struct Wcnn1x1ConvImpl
    : WcnnConvImgAccBase<ADataType, AccDataType, AcoFlag, HPerWcnn_, WPerWcnn_, NumIter_>
{
    using Base =
        WcnnConvImgAccBase<ADataType, AccDataType, AcoFlag, HPerWcnn_, WPerWcnn_, NumIter_>;

    using Base::AChannelsPerVgpr;
    using Base::HPerWcnn;
    using Base::InputChannelNum;
    using Base::NumIter;
    using Base::OutputChannelNum;
    using Base::WPerWcnn;

    // for 8x4 tile; when NumIter = 1, will load NumIter = 2 worth of data
    // when NumIter = 2, will load NumIter = 2 worth of data
    // when NumIter = 3, will load NumIter = 4 worth of data
    // when NumIter = 4, will load NumIter = 4 worth of data
    static constexpr index_t CNumIter_ =
        (HPerWcnn == 8 && WPerWcnn == 4) ? (NumIter_ + (NumIter_ & 1)) : NumIter_;

    static constexpr index_t CPackedNum = (HPerWcnn == 8 && WPerWcnn == 4) ? 2 : 1;

    static constexpr index_t CPerWcnn = Base::InputChannelNum * NumIter;
    static constexpr index_t KPerWcnn = Base::OutputChannelNum;

    using typename Base::AccWarpDstrEncoding;
    using typename Base::AWarpDstrEncoding;

    static constexpr index_t FilterSizeY = 1;
    static constexpr index_t FilterSizeX = 1;
    static constexpr index_t DilationY   = 1;
    static constexpr index_t DilationX   = 1;

    static constexpr index_t BChannelsPerVgpr =
        sizeof(int32_t) / sizeof(BDataType) * numeric_traits<BDataType>::PackedSize;

    static constexpr index_t BKernelNum = (HPerWcnn == 8 && WPerWcnn == 4) ? 8 : 16;

    static constexpr index_t BChannelPackNum = InputChannelNum / 2 / BChannelsPerVgpr;

    using BWarpDstrEncoding = std::conditional_t<
        (HPerWcnn == 8 && WPerWcnn == 4),
        tile_distribution_encoding<
            sequence<>,
            tuple<sequence<BKernelNum>,
                  sequence<1>,
                  sequence<CNumIter_ / 2, 2, BChannelPackNum, 2, BChannelsPerVgpr>>,
            tuple<sequence<3, 1, 3>>,
            tuple<sequence<1, 0, 3>>,
            sequence<3, 3, 3>,
            sequence<0, 2, 4>>,
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<BKernelNum>,
                                         sequence<1>, // because 1x1 filter
                                         sequence<NumIter, BChannelPackNum, 2, BChannelsPerVgpr>>,
                                   tuple<sequence<1, 3>>,
                                   tuple<sequence<0, 2>>,
                                   sequence<3, 3, 3>,
                                   sequence<0, 1, 3>>>;

    using AWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(AWarpDstrEncoding{}))>;
    using BWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(BWarpDstrEncoding{}))>;
    using AccWarpDstr =
        remove_cvref_t<decltype(make_static_tile_distribution(AccWarpDstrEncoding{}))>;

    using AWarpTensor   = static_distributed_tensor<ADataType, AWarpDstr>;
    using BWarpTensor   = static_distributed_tensor<BDataType, BWarpDstr>;
    using AccWarpTensor = static_distributed_tensor<AccDataType, AccWarpDstr>;

    using BDataVecType = ext_vector_t<BDataType, BChannelsPerVgpr * BChannelPackNum * NumIter>;

    using typename Base::AccDataVecType;
    using typename Base::ADataVecType;

    using Intrinsic = WarpConvIntrinsic<ADataType,
                                        AccDataType,
                                        AcoFlag,
                                        HPerWcnn,
                                        WPerWcnn,
                                        FilterSizeY,
                                        FilterSizeX,
                                        DilationY,
                                        DilationX,
                                        NumIter>;

    template <bool HighLane = false, typename AccTensor, typename BTensor, typename ATensor>
    CK_TILE_DEVICE void operator()(AccTensor& acc, const BTensor& b, const ATensor& a) const
    {
        invoke_call<HighLane>(acc, b, a, make_index_sequence<NumIter>{});
    }

    private:
    template <bool HighLane, typename AccTensor, typename BTensor, typename ATensor, index_t... Is>
    CK_TILE_DEVICE void
    invoke_call(AccTensor& acc, const BTensor& b, const ATensor& a, sequence<Is...>) const
    {
        using BVec        = ext_vector_t<BDataType, BTensor::get_thread_buffer_size()>;
        using AccVec      = ext_vector_t<AccDataType, AccTensor::get_thread_buffer_size()>;
        using PerIterAVec = ext_vector_t<ADataType, ATensor::get_thread_buffer_size() / NumIter>;

        constexpr auto I0 = number<0>{};

        const auto b_vec = b.get_thread_buffer().template get_as<BVec>(I0);
        auto acc_vec     = acc.get_thread_buffer().template get_as<AccVec>(I0);

        acc_vec = Intrinsic::template call<HighLane>(
            acc_vec, b_vec, a.get_thread_buffer().template get_as<PerIterAVec>(number<Is>{})...);

        acc.get_thread_buffer().template set_as<AccVec>(I0, acc_vec);
    }
};

} // namespace ck_tile
