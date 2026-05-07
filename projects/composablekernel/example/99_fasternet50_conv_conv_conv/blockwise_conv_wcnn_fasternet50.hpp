// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/warp/wcnn_conv.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_conv_wcnn.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_conv_elementwise_op_wcnn.hpp"

namespace ck {
template <typename ThisThreadBlock,
          typename WeiDataType,
          typename InDataType,
          typename DsDataType,
          typename AccDataType,
          typename EDataType,
          typename AccBlockwiseOperation,
          typename AccBlockwiseNextOperation,
          typename WeiDataBlockDesc,
          typename InDataBlockDesc,
          typename InDataBorderDesc,
          typename DsBlockDesc,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t CPerBlock,
          index_t KPerBlock,
          index_t HRepeat,
          index_t WRepeat,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          bool WeiDataEnableLds     = false,
          bool WeiDataTileLoad      = false,
          bool InDataEnableLds      = false,
          bool InDataTileLoad       = false,
          bool DsEnableLds          = false,
          bool Transposed           = false,
          bool TileStore            = false,
          bool EnableSpatialCluster = false>
/* Option: Read from LDS, big buffer hold all threads required data
 * Source
 * Weight: NumYX x KPerBlock x  CPerBlock (YXKC)
 * InData: HPerBlock x WPerBlock x CPerBlock (HWC)
 * Destination
 * Accum
 * thread level: HRepeat x WRepeat xKRepeat x AccVgprs
 * block  level: HWave x WWave x KWave x HRepeat x WRepeat x KRepeat x AccVgprs
 *
 * Option: Read from VGPR, small buffer hold each thread own required data (Skip LDS)
 * Source:
 * Weight(if skip LDS): KRepeat x CRepeat x WeightVgprs
 * InData(if skip LDS): WRepeat x CRepeat x HRepeat x InDataVgprs
 * Destination
 * Accum
 * block level: HWave x WWave x KWave x HRepeat x WRepeat x KRepeat x AccVgprs
 * NOTE: Assume stride is 2 if FilterSize = 2
 */
struct BlockwiseFasternet50Wcnn
{
    static constexpr auto I0            = Number<0>{};
    static constexpr auto I1            = Number<1>{};
    static constexpr auto I2            = Number<2>{};
    static constexpr auto I3            = Number<3>{};
    static constexpr auto I4            = Number<4>{};
    static constexpr auto I5            = Number<5>{};
    static constexpr auto I6            = Number<6>{};
    static constexpr index_t NumDTensor = DsBlockDesc::Size();

    // Hardcode of WaveSize, since GFX13 conv only support wave32 mode
    static constexpr index_t WaveSize = 32;

    static constexpr auto NumOfThread      = ThisThreadBlock::GetNumOfThread();
    static constexpr bool EnableWaveGroup  = ThisThreadBlock::InWaveGroup();
    static constexpr bool EnableWaveGroup4 = ThisThreadBlock::GetNumWavePerGroup() == 4;

    static constexpr index_t WaveFilterSize = (FilterSize == 2) ? 1 : FilterSize;
    static constexpr bool Aco               = [] { return sizeof(InDataType) > 1; }();

    // Wave properties
    static constexpr index_t Iters  = GetFilterIters<WeiDataType,
                                                     InDataType,
                                                     AccDataType,
                                                     CPerBlock,
                                                     HPerWcnn,
                                                     WPerWcnn,
                                                     FilterSize,
                                                     Transposed>();
    static constexpr auto wcnn_conv = WcnnConv<WeiDataType,
                                               InDataType,
                                               AccDataType,
                                               HPerWcnn,
                                               WPerWcnn,
                                               WaveFilterSize,
                                               DilationX,
                                               DilationY,
                                               Iters,
                                               ThisThreadBlock::InWaveGroup(),
                                               Aco,
                                               false>{};

    static constexpr index_t CPerWcnn = wcnn_conv.GetNumInputChannels();
    static constexpr index_t KPerWcnn = wcnn_conv.GetNumOutputChannels();

    // Output Size Per Wave: (HRepeat * HPerWcnn) * (WRepeat * WPerWcnn) * (KRepeat * KPerWcnn)
    static constexpr index_t HWaves = HPerBlock / (HRepeat * HPerWcnn);
    static constexpr index_t WWaves = WPerBlock / (WRepeat * WPerWcnn);
    static constexpr index_t KWaves = NumOfThread / WaveSize / HWaves / WWaves;

    static constexpr index_t KRepeat = KPerBlock / KWaves / KPerWcnn;
    static constexpr index_t CRepeat = CPerBlock / CPerWcnn;

    static constexpr index_t DataTileHeight = 4;
    static constexpr index_t H_Pad          = (FilterSize == 3) ? DataTileHeight : 0;
    static constexpr index_t W_Pad = ((FilterSize == 3) && !EnableSpatialCluster) ? WPerWcnn : 0;
    static constexpr index_t HPerBlockIn = HPerBlock + H_Pad * 2;
    static constexpr index_t WPerBlockIn = WPerBlock + W_Pad * 2;

    static constexpr index_t NumSubTilePerImage      = wcnn_conv.GetNumSubTilesPerImageTile();
    static constexpr index_t NumSubTilesPerWeightTap = wcnn_conv.GetNumSubTilesPerWeightTap();
    static constexpr index_t NumWeightCompPerTile    = wcnn_conv.GetNumWeightCompPerTile();
    static constexpr index_t NumWeightTap            = wcnn_conv.GetNumWeightTap();
    static constexpr index_t HPerWave                = HRepeat * HPerWcnn;
    static constexpr index_t WPerWave                = WRepeat * WPerWcnn;
    static constexpr index_t CPerWave                = CPerBlock;
    static constexpr index_t KPerWave                = KPerBlock;
    static constexpr index_t HPerWaveIn              = HPerWave + H_Pad * 2;
    static constexpr index_t WPerWaveIn              = WPerWave + W_Pad * 2;
    static constexpr index_t OutputScale2            = (FilterSize == 2) ? (Transposed ? 4 : 1) : 2;
    static constexpr index_t HRepeatOut              = HRepeat * OutputScale2 / 2;
    static constexpr index_t WRepeatOut              = WRepeat * OutputScale2 / 2;
    static constexpr index_t HPerBlockOut            = HPerBlock * OutputScale2 / 2;
    static constexpr index_t WPerBlockOut            = WPerBlock * OutputScale2 / 2;
    static constexpr index_t HPerWaveOut             = HPerWave * OutputScale2 / 2;
    static constexpr index_t WPerWaveOut             = WPerWave * OutputScale2 / 2;

    // Accum descriptor info
    static constexpr auto NumAccComp        = wcnn_conv.GetNumAccumComponents();
    static constexpr auto NumAccCompPerTile = NumAccComp / NumSubTilePerImage;
    static constexpr auto NumAccSwizzleComp = Aco ? 2 : 4;
    static constexpr auto NumAccCompSubTile =
        NumAccCompPerTile > NumAccSwizzleComp ? NumAccCompPerTile / NumAccSwizzleComp : 1;
    static constexpr auto NumDataCompPerTile = wcnn_conv.GetNumDataCompPerTile();

    template <bool HasMainLoop>
    __host__ __device__ constexpr auto MakeAccumThreadBuffer(AccDataType* data)
    {
        if constexpr(EnableWaveGroup4)
        {
            constexpr index_t AccRingVectorCount =
                LaneSharedMemTrait::acc_ring_block_space_aligned /
                wcnn_conv.GetNumAccumComponents() / sizeof(AccDataType);
            constexpr auto VectorCount = HasMainLoop ? math::max(HRepeatOut * WRepeatOut * KRepeat,
                                                                 LaneSharedMemTrait::acc_ring_size)
                                                     : AccRingVectorCount;

            return StaticBufferTupleOfVector<
                AddressSpaceEnum::Vgpr,
                AccDataType,
                VectorCount,
                wcnn_conv.GetNumAccumComponents(),
                true,
                StaticallyIndexedArray_v3<
                    vector_type<AccDataType, wcnn_conv.GetNumAccumComponents()>,
                    VectorCount>>(data);
        }
        else
        {
            return StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                             AccDataType,
                                             HRepeatOut * WRepeatOut * KRepeat,
                                             wcnn_conv.GetNumAccumComponents(),
                                             true>{};
        }
    }

    using KernelEDataType = decltype(wcnn_conv.template GetKernelDataType<EDataType>());
    __host__ __device__ constexpr auto MakeOutThreadBuffer(KernelEDataType* data)
    {
        if constexpr(EnableWaveGroup4)
        {
            return StaticBufferTupleOfVector<
                AddressSpaceEnum::Vgpr,
                KernelEDataType,
                HRepeatOut * WRepeatOut * KRepeat,
                wcnn_conv.GetNumAccumComponents(),
                true,
                StaticallyIndexedArray_v3<
                    vector_type<KernelEDataType, wcnn_conv.GetNumAccumComponents()>,
                    HRepeatOut * WRepeatOut * KRepeat>>(data);
        }
        else
        {
            return StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                             KernelEDataType,
                                             HRepeatOut * WRepeatOut * KRepeat,
                                             wcnn_conv.GetNumAccumComponents(),
                                             true>{};
        }
    }
    __device__ static auto GetWaveIdx()
    {
        return GetWcnnWaveIdx<ThisThreadBlock,
                              HPerBlock,
                              WPerBlock,
                              HRepeat,
                              WRepeat,
                              HPerWcnn,
                              WPerWcnn>();
    }

    // Input data descriptor help functions

    // Describe how the InData in global memory is viewed in block level.
    template <typename InGridDesc>
    __host__ __device__ static constexpr auto
    MakeInGridBlockDescriptor(const InGridDesc& in_grid_pad_desc)
    {
        if constexpr(InDataEnableLds)
        {
            // H x W x C
            return in_grid_pad_desc;
        }
        else
        {
            // H x W x C -> W0 x C0 x H0 x H1 x H2 x W1 x C1
            const auto H = in_grid_pad_desc.GetLength(I0);
            const auto W = in_grid_pad_desc.GetLength(I1);
            const auto C = in_grid_pad_desc.GetLength(I2);

            return transform_tensor_descriptor(
                in_grid_pad_desc,
                make_tuple(
                    make_unmerge_transform(make_tuple(H / HPerWcnn,
                                                      Number<NumSubTilePerImage>{},
                                                      Number<HPerWcnn / NumSubTilePerImage>{})),
                    make_unmerge_transform(make_tuple(W / WPerWcnn, Number<WPerWcnn>{})),
                    make_unmerge_transform(make_tuple(C / CPerWcnn, Number<CPerWcnn>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<2, 3, 4>{}, Sequence<0, 5>{}, Sequence<1, 6>{}));
        }
    }

    // Describe how data read from (LDS/VGPR) buffer, used by Block level classes.
    template <typename InBlockDesc_>
    __host__ __device__ static constexpr auto MakeInWaveDescriptor(const InBlockDesc_&)
    {
        constexpr auto in_wave_desc = [&]() {
            if constexpr(InDataEnableLds)
            {
                // H x W x C -> W0 x C0 x H0 x H1 x H2 x W1 x C1
                return transform_tensor_descriptor(
                    InBlockDesc_{},
                    make_tuple(
                        make_unmerge_transform(make_tuple(Number<HPerBlockIn / HPerWcnn>{},
                                                          Number<NumSubTilePerImage>{},
                                                          Number<HPerWcnn / NumSubTilePerImage>{})),
                        make_unmerge_transform(
                            make_tuple(Number<WPerBlockIn / WPerWcnn>{}, Number<WPerWcnn>{})),
                        make_unmerge_transform(
                            make_tuple(Number<CPerBlock / CPerWcnn>{}, Number<CPerWcnn>{}))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<2, 3, 4>{}, Sequence<0, 5>{}, Sequence<1, 6>{}));
            }
            else
            {
                return InBlockDesc_{};
            }
        }();

        return in_wave_desc;
    }

    __host__ __device__ static constexpr auto MakeInBlockSliceCopyStep()
    {
        constexpr auto in_block_copy_step = [&]() {
            if constexpr(InDataEnableLds)
            {
                return Sequence<I0, I0, Number<CPerBlock>{}>{};
            }
            else
            {
                return Sequence<I0, Number<CPerBlock / CPerWcnn>{}, I0, I0, I0, I0, I0>{};
            }
        }();

        return in_block_copy_step;
    }

    __host__ __device__ static constexpr auto GetInWaveDescLength()
    {
        return Sequence<WPerWaveIn / WPerWcnn,
                        CPerWave / CPerWcnn,
                        HPerWaveIn / HPerWcnn,
                        NumSubTilePerImage,
                        1,
                        1,
                        NumDataCompPerTile>{};
    }

    __host__ __device__ static constexpr auto GetInBorderWaveDescLength()
    {
        return Sequence<1,
                        CPerWave / CPerWcnn,
                        HPerWaveIn / HPerWcnn,
                        NumSubTilePerImage,
                        1,
                        1,
                        NumDataCompPerTile>{};
    }

    __device__ __host__ static auto CalculateInDataThreadOriginDataIndex()
    {
#ifdef __HIP_DEVICE_COMPILE__
        const auto wave_idx = GetWaveIdx();
        const auto waveId_h = wave_idx[I0];
        const auto waveId_w = wave_idx[I1];

        const auto wcnn_in_data_idx =
            wcnn_conv.template CalculateInDataThreadOriginDataIndex<InDataTileLoad>();
        // W0 x C0 x H0 x H1 x H2 x W1 x C1
        return make_tuple(waveId_w * WRepeat,
                          0,
                          waveId_h * HRepeat,
                          0,
                          wcnn_in_data_idx[I0],
                          wcnn_in_data_idx[I1],
                          wcnn_in_data_idx[I2]);
#else
        return make_tuple(0, 0, 0, 0, 0, 0, 0);
#endif
    }

    // Wei data descriptor help functions

    // Describe how the WeiData in global memory is viewed in block level.
    template <typename WeiGridDesc>
    __host__ __device__ static constexpr auto
    MakeWeiGridBlockDescriptor(const WeiGridDesc& wei_grid_desc)
    {
        if constexpr(WeiDataEnableLds)
        {
            // K x YX x C
            return wei_grid_desc;
        }
        else
        {
            // K x YX x C -> K0 x C0 x YX x K1 x C1 x C2
            const auto K = wei_grid_desc.GetLength(I0);
            const auto C = wei_grid_desc.GetLength(I2);
            return transform_tensor_descriptor(
                wei_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(K / KPerWcnn, Number<KPerWcnn>{})),
                           make_pass_through_transform(Number<FilterSize * FilterSize>{}),
                           make_unmerge_transform(
                               make_tuple(C / CPerWcnn,
                                          Number<NumSubTilesPerWeightTap>{},
                                          Number<CPerWcnn / NumSubTilesPerWeightTap>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0, 3>{}, Sequence<2>{}, Sequence<1, 4, 5>{}));
        }
    }

    // Describe how wei data read from (LDS/VGPR) buffer, used by Block level classes.
    template <typename WeiBlockDesc_>
    __host__ __device__ static constexpr auto MakeWeiWaveDescriptor(const WeiBlockDesc_&)
    {
        constexpr auto wei_wave_desc = [&]() {
            if constexpr(WeiDataEnableLds)
            {
                // K x YX x C -> K0 x C0 x YX x K1 x C1 x C2
                return transform_tensor_descriptor(
                    WeiBlockDesc_{},
                    make_tuple(make_unmerge_transform(
                                   make_tuple(Number<KPerBlock / KPerWcnn>{}, Number<KPerWcnn>{})),
                               make_pass_through_transform(Number<FilterSize * FilterSize>{}),
                               make_unmerge_transform(
                                   make_tuple(Number<CPerBlock / CPerWcnn>{},
                                              Number<NumSubTilesPerWeightTap>{},
                                              Number<CPerWcnn / NumSubTilesPerWeightTap>{}))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0, 3>{}, Sequence<2>{}, Sequence<1, 4, 5>{}));
            }
            else
            {
                return WeiBlockDesc_{};
            }
        }();

        return wei_wave_desc;
    }

    __host__ __device__ static constexpr auto MakeWeiBlockSliceCopyStep()
    {
        constexpr auto wei_block_copy_step = [&]() {
            if constexpr(WeiDataEnableLds)
            {
                return Sequence<I0, I0, Number<CPerBlock>{}>{};
            }
            else
            {
                return Sequence<I0, Number<CPerBlock / CPerWcnn>{}, I0, I0, I0, I0>{};
            }
        }();

        return wei_block_copy_step;
    }

    __host__ __device__ static constexpr auto GetWeiWaveDescLength()
    {
        constexpr index_t NumTapPerCopy = (FilterSize == 3) ? 1 : FilterSize * FilterSize;
        return Sequence<KPerWave / KPerWcnn,
                        CPerWave / CPerWcnn,
                        NumTapPerCopy,
                        1,
                        NumSubTilesPerWeightTap,
                        NumWeightCompPerTile>{};
    }

    __device__ __host__ static auto CalculateWeiDataThreadOriginDataIndex()
    {
#ifdef __HIP_DEVICE_COMPILE__
        const auto wave_idx = GetWaveIdx();
        const auto waveId_k = wave_idx[I2];
        const auto wcnn_wei_idx =
            wcnn_conv.template CalculateWeiDataThreadOriginDataIndex<InDataTileLoad>();

        if constexpr(FilterSize == 2)
        {
            return make_tuple(waveId_k * KRepeat,
                              0,
                              wcnn_wei_idx[I0],
                              wcnn_wei_idx[I1],
                              wcnn_wei_idx[I2],
                              wcnn_wei_idx[I3]);
        }
        else if constexpr(Iters > 1)
        {
            return make_tuple(waveId_k * KRepeat,
                              wcnn_wei_idx[I0],
                              0,
                              wcnn_wei_idx[I1],
                              wcnn_wei_idx[I2],
                              wcnn_wei_idx[I3]);
        }
        else
        {
            return make_tuple(waveId_k * KRepeat,
                              0,
                              wcnn_wei_idx[I0],
                              wcnn_wei_idx[I1],
                              wcnn_wei_idx[I2],
                              wcnn_wei_idx[I3]);
        }
#else
        return make_tuple(0, 0, 0, 0, 0, 0);
#endif
    }
    // Acc Data descriptor help functions

    // Describe how to read data from accum_thread_buf_ (VGPR)
    __host__ __device__ static constexpr auto GetAccThreadDescriptor()
    {
        static_assert(NumAccComp == 4 || NumAccComp == 8, "");
        static_assert(NumAccCompPerTile % NumAccCompSubTile == 0, "");

        if constexpr(TileStore)
        {
            // HRepeat x WRepeat x KRepeat x H1 x H2 x W1 x K1
            return make_naive_tensor_descriptor_packed(make_tuple(Number<HRepeatOut>{},
                                                                  Number<WRepeatOut>{},
                                                                  Number<KRepeat>{},
                                                                  I1,
                                                                  I1,
                                                                  I1,
                                                                  Number<NumAccComp>{}));
        }
        else
        {
            // HRepeat x WRepeat x KRepeat x H1 x H2 x W1 x K1 x K2
            return make_naive_tensor_descriptor_packed(
                make_tuple(Number<HRepeatOut>{},
                           Number<WRepeatOut>{},
                           Number<KRepeat>{},
                           Number<NumSubTilePerImage>{},
                           I1,
                           I1,
                           Number<NumAccCompSubTile>{},
                           Number<NumAccCompPerTile / NumAccCompSubTile>{}));
        }
    }

    __host__ __device__ static constexpr auto GetAccThreadDescLength()
    {
        if constexpr(TileStore)
        {
            return Sequence<HRepeatOut, WRepeatOut, KRepeat, I1, I1, I1, NumAccComp>{};
        }
        else
        {
            return Sequence<HRepeatOut,
                            WRepeatOut,
                            KRepeat,
                            NumSubTilePerImage,
                            I1,
                            I1,
                            NumAccCompSubTile,
                            NumAccCompPerTile / NumAccCompSubTile>{};
        }
    }

    // Describe how data store to LDS buffer
    __host__ __device__ static constexpr auto GetAccBlockDescriptor()
    {
        constexpr auto acc_block_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<HPerBlockOut>{}, Number<WPerBlockOut>{}, Number<KPerBlock>{}));
        return acc_block_desc;
    }

    // Describe how to store accum data to LDS or Global memory.
    template <typename AccBlockDesc_>
    __host__ __device__ static constexpr auto GetAccWaveDescriptor(const AccBlockDesc_& accDesc)
    {
        if constexpr(TileStore)
        {
            // H x W x K -> H0 x W0 x K0 x H1 x H2 x W1 x K1
            return transform_tensor_descriptor(
                accDesc,
                make_tuple(
                    make_unmerge_transform(make_tuple(Number<HPerBlockOut / HPerWcnn>{},
                                                      Number<NumSubTilePerImage>{},
                                                      Number<HPerWcnn / NumSubTilePerImage>{})),
                    make_unmerge_transform(
                        make_tuple(Number<WPerBlockOut / WPerWcnn>{}, Number<WPerWcnn>{})),
                    make_unmerge_transform(
                        make_tuple(Number<KPerBlock / KPerWcnn>{}, Number<KPerWcnn>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0, 3, 4>{}, Sequence<1, 5>{}, Sequence<2, 6>{}));
        }
        else
        {
            // H x W x K -> H0 x W0 x K0 x H1 x H2 x W1 x K1 x K2
            return transform_tensor_descriptor(
                accDesc,
                make_tuple(
                    make_unmerge_transform(make_tuple(Number<HPerBlockOut / HPerWcnn>{},
                                                      Number<NumSubTilePerImage>{},
                                                      Number<HPerWcnn / NumSubTilePerImage>{})),
                    make_unmerge_transform(
                        make_tuple(Number<WPerBlockOut / WPerWcnn>{}, Number<WPerWcnn>{})),
                    make_unmerge_transform(make_tuple(Number<KPerBlock / KPerWcnn>{},
                                                      Number<NumAccCompSubTile>{},
                                                      Number<KPerWcnn / NumAccCompSubTile>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0, 3, 4>{}, Sequence<1, 5>{}, Sequence<2, 6, 7>{}));
        }
    }

    __device__ __host__ static auto CalculateAccThreadOriginDataIndex()
    {
        if constexpr(TileStore)
        {
#ifdef __HIP_DEVICE_COMPILE__
            const auto wave_idx = GetWaveIdx();
            const auto wcnn_in_data_idx =
                wcnn_conv.template CalculateAccThreadOriginDataIndex<true>();
            const auto waveId_h = wave_idx[I0];
            const auto waveId_w = wave_idx[I1];
            const auto waveId_k = wave_idx[I2];
            return make_tuple(waveId_h * HRepeatOut,
                              waveId_w * WRepeatOut,
                              waveId_k * KRepeat,
                              wcnn_in_data_idx[I0],
                              wcnn_in_data_idx[I1],
                              wcnn_in_data_idx[I2],
                              wcnn_in_data_idx[I3]);
#else
            return make_tuple(0, 0, 0, 0, 0, 0, 0);
#endif
        }
        else
        {
#ifdef __HIP_DEVICE_COMPILE__
            const auto wave_idx = GetWaveIdx();
            const auto wcnn_in_data_idx =
                wcnn_conv.template CalculateAccThreadOriginDataIndex<false>();
            const auto waveId_h = wave_idx[I0];
            const auto waveId_w = wave_idx[I1];
            const auto waveId_k = wave_idx[I2];
            return make_tuple(waveId_h * HRepeatOut,
                              waveId_w * WRepeatOut,
                              waveId_k * KRepeat,
                              wcnn_in_data_idx[I0],
                              wcnn_in_data_idx[I1],
                              wcnn_in_data_idx[I2],
                              wcnn_in_data_idx[I3],
                              wcnn_in_data_idx[I4]);
#else
            return make_tuple(0, 0, 0, 0, 0, 0, 0, 0);
#endif
        }
    }

    using TupleWeiData    = decltype(CalculateWeiDataThreadOriginDataIndex());
    using TupleInData     = decltype(CalculateInDataThreadOriginDataIndex());
    using InDataWaveDesc  = decltype(MakeInWaveDescriptor(InDataBlockDesc{}));
    using WeiDataWaveDesc = decltype(MakeWeiWaveDescriptor(WeiDataBlockDesc{}));

    template <bool HasMainLoop>
    static constexpr index_t GetLaneSharedMemCount()
    {
        return HasMainLoop ? 2 : 1;
    }

    __host__ __device__
    BlockwiseFasternet50Wcnn(const AccBlockwiseOperation& acc_blockwise_op,
                             const AccBlockwiseNextOperation& acc_blockwise_next_op,
                             TupleWeiData weight_origin = CalculateWeiDataThreadOriginDataIndex(),
                             TupleInData indata_origin  = CalculateInDataThreadOriginDataIndex())
        : weight_thread_copy_(weight_origin),
          indata_thread_copy_(indata_origin),
          element_op_(acc_blockwise_op)
    {
        (void)acc_blockwise_next_op;
        static_assert(WeiDataBlockDesc::IsKnownAtCompileTime() &&
                          InDataBlockDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");
        static_assert(WeiDataWaveDesc::IsKnownAtCompileTime() &&
                          InDataWaveDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ThisThreadBlock::GetNumOfThread() == HWaves * WWaves * KWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize\n");

        static_assert(HPerBlock % (HPerWcnn * HRepeat) == 0, "wrong!");

        static_assert(WPerBlock % (WPerWcnn * WRepeat) == 0, "wrong!");

        static_assert(KPerBlock % (KPerWcnn * KRepeat) == 0, "wrong!");

        static_assert(!(Transposed && (FilterSize != 2)),
                      "Only support strided conv2x2 transpose conv");
        static_assert(
            !((Transposed == false) && (FilterSize == 2) &&
              ((HRepeat % 2 != 0) || ((EnableSpatialCluster == false) && (WRepeat % 2 != 0)))),
            "Repeat must be even for strided conv 2x2 conv");
        static_assert(HPerBlock % (HRepeat * HPerWcnn) == 0, "");
        static_assert(WPerBlock % (WRepeat * WPerWcnn) == 0, "");
        static_assert(KWaves > 0, "");
        static_assert(NumOfThread == (WaveSize * HWaves * WWaves * KWaves), "");
        static_assert(KRepeat >= 1, "");
        static_assert(CRepeat >= 1, "");
        static_assert(HPerWaveIn % HPerWcnn == 0, "");
        static_assert(WPerWaveIn % WPerWcnn == 0, "");
    }

    __host__ __device__ static constexpr auto GetWeightRemapTable()
    {
        return wcnn_conv.GetWeightRemapTable();
    }

    __host__ __device__ static constexpr auto GetWeightSecondTapMapTable()
    {
        return wcnn_conv.GetWeightSecondTapMapTable();
    }
    // Describe how data allocated in thread copy src buffer
    static constexpr WeiDataWaveDesc weight_wave_desc_;
    static constexpr InDataWaveDesc indata_wave_desc_;
    static constexpr InDataBorderDesc indata_border_wave_desc_;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
    template <typename VectorType, typename StaticBuffer>
    static constexpr void CopyVector(VectorType& x, const StaticBuffer& y)
    {
        using ScalarType = typename StaticBuffer::type;
        if constexpr(sizeof(StaticBuffer) == sizeof(VectorType))
        {
            x = bit_cast<VectorType>(y);
        }
        else
        {
            static_for<0, y.Size(), 1>{}(
                [&](auto i) { x.template AsType<ScalarType>()(i) = y[i]; });
        }
    }

    template <typename VectorType,
              typename StaticBuffer,
              typename NumDstOffset,
              typename NumSrcOffset,
              typename NumCount>
    static constexpr void
    CopySubVector(VectorType& x, const StaticBuffer& y, NumDstOffset, NumSrcOffset, NumCount)
    {
        using ScalarType = typename StaticBuffer::type;
#if 1
        static_assert(sizeof(ScalarType) * NumCount{} % sizeof(uint32_t) == 0);
        static_assert(sizeof(ScalarType) * NumDstOffset{} % sizeof(uint32_t) == 0);
        static_assert(sizeof(ScalarType) * NumSrcOffset{} % sizeof(uint32_t) == 0);
        constexpr index_t dwCount     = sizeof(ScalarType) * NumCount{} / sizeof(uint32_t);
        constexpr index_t dwSrcOffset = sizeof(ScalarType) * NumSrcOffset{} / sizeof(uint32_t);
        constexpr index_t dwDstOffset = sizeof(ScalarType) * NumDstOffset{} / sizeof(uint32_t);
        uint32_t* pDst                = reinterpret_cast<uint32_t*>(&x);
        const uint32_t* pSrc          = reinterpret_cast<const uint32_t*>(&y[I0]);
        static_for<0, dwCount, 1>{}([&](auto i) { pDst[i + dwDstOffset] = pSrc[i + dwSrcOffset]; });
#else
        // LLVM crash
        static_for<0, NumCount{}, 1>{}([&](auto i) {
            x.template AsType<ScalarType>()(i + NumDstOffset{}) = y[i + NumSrcOffset{}];
        });
#endif
    }

    template <index_t conv_phase,
              typename WeightBlockBuffer,
              typename InDataBlockBuffer,
              typename InDataThreadBuffer,
              typename ExchangeDataBlockBuffer,
              typename DsBlockBuffer,
              typename AccumThreadBuffer,
              typename OutTensorThreadBuffer,
              typename WaveGroupSemaphores,
              typename HasMainLoop,
              typename IsLast>
    __device__ void Run(const WeightBlockBuffer& weight_block_buf,
                        const InDataBlockBuffer& indata_from_block_buf,   // channel: 16~31
                        const InDataThreadBuffer& indata_from_thread_buf, // channel: 0~15
                        const ExchangeDataBlockBuffer& predata_block_buf,
                        const ExchangeDataBlockBuffer& nextdata_block_buf,
                        const DsBlockBuffer& ds_block_buf,
                        AccumThreadBuffer& accum_thread_buf,
                        OutTensorThreadBuffer& out_thread_buf,
                        WaveGroupSemaphores& semaAccums,
                        HasMainLoop,
                        IsLast) const
    {
        (void)predata_block_buf;
        (void)nextdata_block_buf;
        (void)semaAccums;
        auto weight_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, WeiDataType>(
            weight_thread_desc_.GetElementSpaceSize());

        using WeiDataVec    = typename decltype(wcnn_conv)::WeiDataVec;
        using WeiDataTapVec = typename decltype(wcnn_conv)::WeiDataTapVec;
        using InDataVec     = typename decltype(wcnn_conv)::InDataVec;
        using AccDataVec    = typename decltype(wcnn_conv)::AccDataVec;

        const typename InDataVec::type* indata_thread_vec_ptr[4];
        static_assert(Iters <= 4 && FilterSize < 4, "");
        constexpr index_t CStep = Iters;

        auto load_in_data_in_phase = [&](auto h0, auto w0, auto c0, auto& tmp_buf) {
            if constexpr(InDataEnableLds)
            {
                // Load input tensor data
                if constexpr(conv_phase == 0)
                {
                    //  read tensor
                    static_assert(FilterSize == 3, "Fasternet50 phase1 filter should be 3x3!");
                    static_for<0, FilterSize, 1>{}([&](auto array_idx) {
                        auto indata_thread_buf =
                            make_static_buffer_v3<AddressSpaceEnum::Vgpr, InDataType>(
                                indata_thread_desc_.GetElementSpaceSize(),
                                &tmp_buf[array_idx].template AsType<InDataType>()(I0));
                        indata_thread_copy_.Run(indata_wave_desc_,
                                                make_tuple(w0 + array_idx, c0, h0, I0, I0, I0, I0),
                                                indata_from_block_buf,
                                                indata_thread_desc_,
                                                make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                                indata_thread_buf);
                        indata_thread_vec_ptr[array_idx] =
                            &tmp_buf[array_idx].template AsType<typename InDataVec::type>()[I0];
                    });
                }
                else if constexpr(conv_phase == 1)
                {
                    static_assert(FilterSize == 1, "Fasternet50 phase1 filter should be 1x1!");
                    static_for<0, Iters, 1>{}([&](auto iter_idx) {
                        auto indata_thread_buf =
                            make_static_buffer_v3<AddressSpaceEnum::Vgpr, InDataType>(
                                indata_thread_desc_.GetElementSpaceSize(),
                                &tmp_buf[iter_idx].template AsType<InDataType>()(I0));
                        indata_thread_copy_.Run(indata_wave_desc_,
                                                make_tuple(w0, c0 + iter_idx, h0, I0, I0, I0, I0),
                                                indata_from_block_buf,
                                                indata_thread_desc_,
                                                make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                                indata_thread_buf);
                        indata_thread_vec_ptr[iter_idx] =
                            &tmp_buf[iter_idx].template AsType<typename InDataVec::type>()[I0];
                    });
                }
                else
                {
                    static_assert(0, "Unreachable here!");
                }
            }
            else
            {
                static_assert(0, "Not support this case yet!");
            }
        };

        auto load_weight_data = [&](auto k0, auto c0, WeiDataVec& tmp_buf) {
            const typename WeiDataVec::type* weight_thread_vec_ptr;
            if constexpr(WeiDataEnableLds)
            {
                if constexpr(FilterSize == 1)
                {
                    constexpr auto TapPerIter = wcnn_conv.GetNumWeightTapPerWave();
                    static_for<0, Iters, TapPerIter>{}([&](auto i) {
                        weight_thread_copy_.Run(weight_wave_desc_,
                                                make_tuple(k0, Number<c0 + i>{}, I0, I0, I0, I0),
                                                weight_block_buf,
                                                weight_thread_desc_,
                                                make_tuple(I0, I0, I0, I0, I0, I0),
                                                weight_thread_buf);
                        constexpr auto CompCount  = sizeof(WeiDataTapVec) / sizeof(WeiDataType);
                        constexpr auto CompOffset = i * CompCount / TapPerIter;
                        CopySubVector(tmp_buf,
                                      weight_thread_buf,
                                      Number<CompOffset>{},
                                      I0,
                                      Number<CompCount>{});
                    });
                    weight_thread_vec_ptr =
                        &tmp_buf.template AsType<typename WeiDataVec::type>()[I0];
                }
                else if constexpr(FilterSize == 3)
                {
                    static_assert(Iters == 1, "");
                    constexpr index_t WeightTapPerWave = wcnn_conv.GetNumWeightTapPerWave();
                    static_for<0, wcnn_conv.GetNumWeightTap(), 1>{}([&](auto tape_idx) {
                        weight_thread_copy_.Run(
                            weight_wave_desc_,
                            make_tuple(k0, c0, Number<WeightTapPerWave * tape_idx>{}, I0, I0, I0),
                            weight_block_buf,
                            weight_thread_desc_,
                            make_tuple(I0, I0, I0, I0, I0, I0),
                            weight_thread_buf);

                        constexpr auto CompCount  = sizeof(WeiDataTapVec) / sizeof(WeiDataType);
                        constexpr auto CompOffset = tape_idx * CompCount;
                        CopySubVector(tmp_buf,
                                      weight_thread_buf,
                                      Number<CompOffset>{},
                                      I0,
                                      Number<CompCount>{});
                    });
                    weight_thread_vec_ptr =
                        &tmp_buf.template AsType<typename WeiDataVec::type>()[I0];
                }
            }
            else
            {
                constexpr index_t wei_offset =
                    weight_wave_desc_.CalculateOffset(make_tuple(k0, c0, I0, I0, I0, I0));
                if constexpr((Iters == 4) && (wcnn_conv.GetNumWeightTapPerWave() == 2))
                {
                    constexpr auto CompCount = sizeof(WeiDataTapVec) / sizeof(WeiDataType);
                    constexpr index_t wei_offset2 =
                        weight_wave_desc_.CalculateOffset(make_tuple(k0, c0 + I2, I0, I0, I0, I0));
                    CopySubVector(
                        tmp_buf, weight_block_buf, I0, Number<wei_offset>{}, Number<CompCount>{});
                    CopySubVector(tmp_buf,
                                  weight_block_buf,
                                  Number<CompCount>{},
                                  Number<wei_offset2>{},
                                  Number<CompCount>{});
                    weight_thread_vec_ptr =
                        &tmp_buf.template AsType<typename WeiDataVec::type>()[I0];
                }
                else
                {
                    weight_thread_vec_ptr = reinterpret_cast<const typename WeiDataVec::type*>(
                        &weight_block_buf[Number<wei_offset>{}]);
                }
            }
            return weight_thread_vec_ptr;
        };

        constexpr auto get_accum_offset = [&](auto h0, auto w0, auto k0) {
            if constexpr(HasMainLoop{} || (EnableWaveGroup4 == false))
            {
                return accum_thread_desc_.CalculateOffset(make_tuple(h0, w0, k0, I0));
            }
            else
            {
                constexpr index_t Idx = (h0 * WRepeat + w0) & 1 ? 1 : 0;
                return Idx * NumAccComp;
            }
        };

        constexpr auto get_in_1_fromthread_offset = [&](auto h0, auto w0) {
            if constexpr(HasMainLoop{} || (EnableWaveGroup4 == false))
            {
                return in_1_from_thread_desc_.CalculateOffset(make_tuple(h0, w0, I0));
            }
            else
            {
                static_assert(0, "Not support this paramter yet!");
            }
        };

        constexpr auto get_in_2_fromthread_offset = [&](auto h0, auto w0, auto c0) {
            if constexpr(HasMainLoop{} || (EnableWaveGroup4 == false))
            {
                return in_2_from_thread_desc_.CalculateOffset(make_tuple(h0, w0, c0, I0));
            }
            else
            {
                static_assert(0, "Not support this paramter yet!");
            }
        };

        if constexpr(EnableWaveGroup4)
        {
            static_assert(0, "Not support this mode yet!");
        }
        else
        {
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                constexpr index_t CRepeatInPhase = (conv_phase == 0) ? (CRepeat / 2) : CRepeat;
                WeiDataVec weight_thread_vec[CRepeatInPhase / CStep];
                const typename WeiDataVec::type* weight_thread_vec_ptr[CRepeatInPhase / CStep];
                static_for<0, CRepeatInPhase, CStep>{}([&](auto c0) {
                    // Load weights
                    weight_thread_vec_ptr[c0 / CStep] =
                        load_weight_data(k0, c0, weight_thread_vec[c0 / CStep]);
                });

                static_assert(HRepeat == 12, "TileH should be 96");
                static_assert(WRepeat == 2, "TileW should be 16");

                static_for<0, HRepeat, 1>{}([&](auto h0) {
                    static_for<0, WRepeat, 1>{}([&](auto w0) {
                        constexpr index_t accum_offset =
                            get_accum_offset(Number<h0>{}, Number<w0>{}, Number<k0>{});
                        InDataVec indata_thread_vec[4];
                        AccDataVec acc_vec = {};
                        if constexpr(HasMainLoop{})
                        {
                            acc_vec =
                                accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{});
                        }

                        if constexpr(conv_phase == 0)
                        {
                            static_assert(CRepeat == 8, "BlockChannels for input should be 16.");
                            static_assert(KRepeat == 1, "BlockChannels for output should be 16.");
                            // Only 16 channels for convolution
                            static_for<0, CRepeat / 2, CStep>{}([&](auto c0) {
                                load_in_data_in_phase(h0, w0, c0, indata_thread_vec);

                                wcnn_conv.conv_instr.Run(*weight_thread_vec_ptr[c0 / CStep],
                                                         indata_thread_vec_ptr,
                                                         acc_vec,
                                                         I0);
                            });
                        }
                        else if constexpr(conv_phase == 1)
                        {
                            static_assert(CStep == 4,
                                          "BlockChannels for input should be 32, step is 4");
                            static_assert(CRepeat == 8,
                                          "BlockChannels for input should be 32, repeat is 8");
                            static_assert(KRepeat == 4, "BlockChannels for output should be 64.");

                            constexpr index_t in_fromthread_offset =
                                get_in_1_fromthread_offset(Number<h0>{}, Number<w0>{});
                            auto indata_thread_vec_temp =
                                indata_from_thread_buf.GetVectorTypeReference(
                                    Number<in_fromthread_offset>{});

                            static_for<0, Iters, 1>{}([&](auto iter_idx) {
                                indata_thread_vec_ptr[iter_idx] =
                                    &indata_thread_vec_temp.template AsType<
                                        typename InDataVec::type>()[Number<iter_idx>{}];
                            });

                            wcnn_conv.conv_instr.Run(
                                *weight_thread_vec_ptr[0], indata_thread_vec_ptr, acc_vec, I0);

                            load_in_data_in_phase(h0, w0, Number<4>{}, indata_thread_vec);
                            wcnn_conv.conv_instr.Run(
                                *weight_thread_vec_ptr[1], indata_thread_vec_ptr, acc_vec, I0);
                        }
                        else if constexpr(conv_phase == 2)
                        {
                            static_assert(CStep == 4,
                                          "BlockChannels for input should be 64, step is 4");
                            static_assert(CRepeat == 16,
                                          "BlockChannels for input should be 64, repeat is 16");
                            static_assert(KRepeat == 2,
                                          "BlockChannels for output should be 32, repeart is 2");

                            static_for<0, CRepeat, CStep>{}([&](auto c0) {
                                constexpr index_t in_fromthread_offset = get_in_2_fromthread_offset(
                                    Number<h0>{}, Number<w0>{}, Number<c0 / CStep>{});
                                auto indata_thread_vec_temp =
                                    indata_from_thread_buf.GetVectorTypeReference(
                                        Number<in_fromthread_offset>{});
                                static_for<0, CStep, 1>{}([&](auto iter_idx) {
                                    indata_thread_vec_ptr[iter_idx] =
                                        &indata_thread_vec_temp.template AsType<
                                            typename InDataVec::type>()[Number<iter_idx>{}];
                                });

                                wcnn_conv.conv_instr.Run(*weight_thread_vec_ptr[c0 / CStep],
                                                         indata_thread_vec_ptr,
                                                         acc_vec,
                                                         I0);
                            });
                        }

                        if constexpr(IsLast{})
                        {
                            element_op_.Run(ds_block_buf, acc_vec, h0, w0, k0);
                            element_next_op_.Run(acc_vec, out_thread_buf, h0, w0, k0);
                        }
                        else
                        {
                            accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{}) =
                                acc_vec;
                        }
                    });
                });
            });
        }
    };

    template <typename WeightBlockBuffer,
              typename InDataBlockBuffer,
              typename DsBlockBuffer,
              typename AccumThreadBuffer,
              typename OutTensorThreadBuffer,
              typename WaveGroupSemaphores,
              typename HasMainLoop,
              typename IsLast>
    __device__ void RunEmulateConv2(const WeightBlockBuffer& weight_block_buf,
                                    const InDataBlockBuffer& indata_block_buf,
                                    const DsBlockBuffer& ds_block_buf,
                                    AccumThreadBuffer& accum_thread_buf,
                                    OutTensorThreadBuffer& out_thread_buf,
                                    WaveGroupSemaphores& semaAccums,
                                    HasMainLoop,
                                    IsLast) const
    {
        auto weight_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, WeiDataType>(
            weight_thread_desc_.GetElementSpaceSize());
        using WeiDataVec    = typename decltype(wcnn_conv)::WeiDataVec;
        using WeiDataTapVec = typename decltype(wcnn_conv)::WeiDataTapVec;
        using InDataVec     = typename decltype(wcnn_conv)::InDataVec;
        using AccDataVec    = typename decltype(wcnn_conv)::AccDataVec;

        const typename InDataVec::type* indata_thread_vec_ptr[4];
        static_assert(Iters <= 4 && FilterSize == 2, "");

        static constexpr auto NumYX          = FilterSize * FilterSize;
        static constexpr index_t x_offset[4] = {0, 1, 0, 1};
        static constexpr index_t y_offset[4] = {0, 0, 1, 1};
        static constexpr index_t CStep       = Transposed == false ? 1 : Iters;
        static constexpr index_t HStep       = Transposed == false ? 2 : 1;
        static constexpr index_t WStep       = Transposed == false ? 2 : 1;

        auto load_weight_data = [&](auto k0, auto c0, auto& tmp_buf, auto& weight_ptr) {
            constexpr auto TapPerIter = wcnn_conv.GetNumWeightTapPerWave();
            if constexpr(WeiDataEnableLds)
            {
                static_for<0, NumYX, TapPerIter>{}([&](auto i) {
                    weight_thread_copy_.Run(weight_wave_desc_,
                                            make_tuple(k0, c0, i, I0, I0, I0),
                                            weight_block_buf,
                                            weight_thread_desc_,
                                            make_tuple(I0, I0, I0, I0, I0, I0),
                                            weight_thread_buf);
                    if constexpr(Transposed)
                    {
                        CopyVector(tmp_buf[i / TapPerIter], weight_thread_buf);
                        weight_ptr[i / TapPerIter] =
                            &tmp_buf[i / TapPerIter]
                                 .template AsType<typename WeiDataVec::type>()[I0];
                    }
                    else
                    {
                        static_assert(sizeof(WeiDataTapVec) % sizeof(WeiDataType) == 0, "");
                        constexpr auto CompCount  = sizeof(WeiDataTapVec) / sizeof(WeiDataType);
                        constexpr auto CompOffset = i * CompCount / TapPerIter;
                        CopySubVector(tmp_buf[0],
                                      weight_thread_buf,
                                      Number<CompOffset>{},
                                      I0,
                                      Number<CompCount>{});
                        weight_ptr[0] =
                            &tmp_buf[0].template AsType<typename WeiDataVec::type>()[I0];
                    }
                });
            }
            else
            {
                if constexpr(Transposed)
                {
                    static_for<0, NumYX, TapPerIter>{}([&](auto i) {
                        constexpr index_t wei_offset =
                            weight_wave_desc_.CalculateOffset(make_tuple(k0, c0, i, I0, I0, I0));
                        weight_ptr[i / TapPerIter] =
                            reinterpret_cast<const typename WeiDataVec::type*>(
                                &weight_block_buf[Number<wei_offset>{}]);
                    });
                }
                else
                {
                    constexpr index_t wei_offset =
                        weight_wave_desc_.CalculateOffset(make_tuple(k0, c0, I0, I0, I0, I0));
                    if constexpr(wcnn_conv.GetNumWeightTapPerWave() == 2)
                    {
                        constexpr index_t wei_offset2 =
                            weight_wave_desc_.CalculateOffset(make_tuple(k0, c0, I2, I0, I0, I0));
                        static_assert(sizeof(WeiDataTapVec) % sizeof(WeiDataType) == 0, "");
                        constexpr auto CompCount = sizeof(WeiDataTapVec) / sizeof(WeiDataType);
                        CopySubVector(tmp_buf[0],
                                      weight_block_buf,
                                      I0,
                                      Number<wei_offset>{},
                                      Number<CompCount>{});
                        CopySubVector(tmp_buf[0],
                                      weight_block_buf,
                                      Number<CompCount>{},
                                      Number<wei_offset2>{},
                                      Number<CompCount>{});
                        weight_ptr[0] =
                            &tmp_buf[0].template AsType<typename WeiDataVec::type>()[I0];
                    }
                    else
                    {
                        weight_ptr[0] = reinterpret_cast<const typename WeiDataVec::type*>(
                            &weight_block_buf[Number<wei_offset>{}]);
                    }
                }
            }
        };

        auto load_in_data = [&](auto h0, auto w0, auto c0, auto& tmp_buf) {
            if constexpr(InDataEnableLds)
            {
                // Load input tensor data
                if constexpr(Transposed == false)
                {
                    static_assert(Iters == NumYX, "");
                    InDataVec indata_thread_vec_tmp[4];
                    static_for<0, NumYX, 1>{}([&](auto i) {
                        auto indata_thread_buf =
                            make_static_buffer_v3<AddressSpaceEnum::Vgpr, InDataType>(
                                indata_thread_desc_.GetElementSpaceSize(),
                                &indata_thread_vec_tmp[i].template AsType<InDataType>()(I0));
                        indata_thread_copy_.Run(indata_wave_desc_,
                                                make_tuple(Number<w0 + x_offset[i]>{},
                                                           c0,
                                                           Number<h0 + y_offset[i]>{},
                                                           I0,
                                                           I0,
                                                           I0,
                                                           I0),
                                                indata_block_buf,
                                                indata_thread_desc_,
                                                make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                                indata_thread_buf);
                        indata_thread_vec_ptr[i] =
                            &tmp_buf[i].template AsType<typename InDataVec::type>()[I0];
                    });
                    wcnn_conv.UnshuffleStridedConv2Data(indata_thread_vec_tmp, tmp_buf);
                }
                else
                {
                    auto indata_thread_buf =
                        make_static_buffer_v3<AddressSpaceEnum::Vgpr, InDataType>(
                            indata_thread_desc_.GetElementSpaceSize(),
                            &tmp_buf[I0].template AsType<InDataType>()(I0));
                    indata_thread_copy_.Run(indata_wave_desc_,
                                            make_tuple(w0, c0, h0, I0, I0, I0, I0),
                                            indata_block_buf,
                                            indata_thread_desc_,
                                            make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                            indata_thread_buf);
                    indata_thread_vec_ptr[I0] =
                        &tmp_buf[I0].template AsType<typename InDataVec::type>()[I0];
                }
            }
            else
            {
                // Load input tensor data
                if constexpr(Transposed == false)
                {
                    InDataVec indata_thread_vec_tmp[4];
                    constexpr auto CompCount = sizeof(InDataVec) / sizeof(InDataType);
                    static_for<0, NumYX, 1>{}([&](auto i) {
                        constexpr index_t indata_offset = indata_wave_desc_.CalculateOffset(
                            make_tuple(w0 + x_offset[i], c0, h0 + y_offset[i], I0, I0, I0, I0));
                        CopySubVector(indata_thread_vec_tmp[i],
                                      indata_block_buf,
                                      I0,
                                      Number<indata_offset>{},
                                      Number<CompCount>{});
                        indata_thread_vec_ptr[i] =
                            &tmp_buf[i].template AsType<typename InDataVec::type>()[I0];
                    });
                    wcnn_conv.UnshuffleStridedConv2Data(indata_thread_vec_tmp, tmp_buf);
                }
                else
                {
                    constexpr index_t indata_offset =
                        indata_wave_desc_.CalculateOffset(make_tuple(w0, c0, h0, I0, I0, I0, I0));
                    indata_thread_vec_ptr[I0] = reinterpret_cast<const typename InDataVec::type*>(
                        &indata_block_buf[Number<indata_offset>{}]);
                }
            }
        };

        constexpr auto get_accum_offset_transposed = [&](auto h0, auto w0, auto k0, auto i) {
            if constexpr(HasMainLoop{} || (EnableWaveGroup4 == false))
            {
                return accum_thread_desc_.CalculateOffset(make_tuple(
                    Number<h0 * 2 + y_offset[i]>{}, Number<w0 * 2 + x_offset[i]>{}, k0, I0));
            }
            else
            {
                constexpr index_t WStride = 1;
                constexpr index_t HStride = WRepeat / WStep;
                constexpr index_t Idx = (h0 / HStep * HStride + w0 / WStep * WStride) & 1 ? 1 : 0;
                return (NumYX * Idx + i) * NumAccComp;
            }
        };

        constexpr auto get_accum_offset = [&](auto h0, auto w0, auto k0) {
            if constexpr(HasMainLoop{} || (EnableWaveGroup4 == false))
            {
                return accum_thread_desc_.CalculateOffset(
                    make_tuple(Number<h0 / HStep>{}, Number<w0 / WStep>{}, k0, I0));
            }
            else
            {
                constexpr index_t WStride = 1;
                constexpr index_t HStride = WRepeat / WStep;
                constexpr index_t Idx = (h0 / HStep * HStride + w0 / WStep * WStride) & 1 ? 1 : 0;
                return Idx * NumAccComp;
            }
        };

        if constexpr(EnableWaveGroup4)
        {
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                const typename WeiDataVec::type* weight_thread_vec_ptr[CRepeat / CStep][4];

                // Load weights
                WeiDataVec weight_thread_vec[CRepeat / CStep][4];
                auto semaAccumReady = semaAccums[I0];
                auto semaAccumFree  = semaAccums[I1];
                if(ThisThreadBlock::GetWaveIdInWaveGroup() == WaveIdRun)
                {
                    static_for<0, CRepeat, CStep>{}([&](auto c0) {
                        load_weight_data(k0,
                                         c0,
                                         weight_thread_vec[c0 / CStep],
                                         weight_thread_vec_ptr[c0 / CStep]);
                    });

                    InDataVec indata_thread_vec[4];
                    static_for<0, HRepeat, HStep>{}([&](auto h0) {
                        static_for<0, WRepeat, WStep>{}([&](auto w0) {
                            if constexpr(Transposed == false)
                            {
                                constexpr index_t accum_offset =
                                    get_accum_offset(Number<h0>{}, Number<w0>{}, Number<k0>{});
                                AccDataVec acc_vec = {};
                                if constexpr(HasMainLoop{})
                                {
                                    acc_vec = accum_thread_buf.GetVectorTypeReference(
                                        Number<accum_offset>{});
                                }

                                static_for<0, CRepeat, CStep>{}([&](auto c0) {
                                    load_in_data(h0, w0, c0, indata_thread_vec);
                                    wcnn_conv.conv_instr.Run(*weight_thread_vec_ptr[c0 / CStep][0],
                                                             indata_thread_vec_ptr,
                                                             acc_vec,
                                                             I0);
                                });

                                if constexpr(IsLast{})
                                {
                                    semaAccumFree->template wait<0>();
                                    accum_thread_buf.GetVectorTypeReference(
                                        Number<accum_offset>{}) = acc_vec;
                                    semaAccumReady->template signal<0>();
                                }
                                else
                                {
                                    accum_thread_buf.GetVectorTypeReference(
                                        Number<accum_offset>{}) = acc_vec;
                                }
                            }
                            else
                            {
                                AccDataVec acc_vec[NumYX] = {};
                                if constexpr(HasMainLoop{})
                                {
                                    static_for<0, NumYX, 1>{}([&](auto i) {
                                        constexpr index_t accum_offset =
                                            get_accum_offset_transposed(Number<h0>{},
                                                                        Number<w0>{},
                                                                        Number<k0>{},
                                                                        Number<i>{});
                                        acc_vec[i] = accum_thread_buf.GetVectorTypeReference(
                                            Number<accum_offset>{});
                                    });
                                }

                                static_for<0, CRepeat, CStep>{}([&](auto c0) {
                                    load_in_data(h0, w0, c0, indata_thread_vec);
                                    static_for<0, NumYX, 1>{}([&](auto i) {
                                        constexpr index_t tapIdx =
                                            i / wcnn_conv.GetNumWeightTapPerWave();
                                        constexpr bool isHigh =
                                            i % wcnn_conv.GetNumWeightTapPerWave();
                                        constexpr auto Mod = []() {
                                            if constexpr(isHigh)
                                                return I1;
                                            else
                                                return I0;
                                        }();

                                        wcnn_conv.conv_instr.Run(
                                            *weight_thread_vec_ptr[c0 / CStep][tapIdx],
                                            indata_thread_vec_ptr,
                                            acc_vec[i],
                                            Mod);
                                    });
                                });

                                if constexpr(IsLast{})
                                {
                                    semaAccumFree->template wait<0>();
                                    static_for<0, NumYX, 1>{}([&](auto i) {
                                        constexpr index_t accum_offset =
                                            get_accum_offset_transposed(Number<h0>{},
                                                                        Number<w0>{},
                                                                        Number<k0>{},
                                                                        Number<i>{});
                                        accum_thread_buf.GetVectorTypeReference(
                                            Number<accum_offset>{}) = acc_vec[i];
                                    });
                                    semaAccumReady->template signal<0>();
                                }
                                else
                                {
                                    static_for<0, NumYX, 1>{}([&](auto i) {
                                        constexpr index_t accum_offset =
                                            get_accum_offset_transposed(Number<h0>{},
                                                                        Number<w0>{},
                                                                        Number<k0>{},
                                                                        Number<i>{});
                                        accum_thread_buf.GetVectorTypeReference(
                                            Number<accum_offset>{}) = acc_vec[i];
                                    });
                                }
                            }
                        });
                    });
                }
                else if(IsLast{} && ThisThreadBlock::GetWaveIdInWaveGroup() == WaveIdPostRun)
                {
                    static_for<0, HRepeat, HStep>{}([&](auto h0) {
                        static_for<0, WRepeat, WStep>{}([&](auto w0) {
                            if constexpr(Transposed == false)
                            {
                                constexpr index_t accum_offset =
                                    get_accum_offset(Number<h0>{}, Number<w0>{}, Number<k0>{});

                                semaAccumReady->template wait<0>();
                                AccDataVec acc_vec =
                                    accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{});
                                element_op_.Run(ds_block_buf,
                                                acc_vec,
                                                Number<h0 / HStep>{},
                                                Number<w0 / WStep>{},
                                                k0);

                                element_next_op_.Run(acc_vec,
                                                     out_thread_buf,
                                                     Number<h0 / HStep>{},
                                                     Number<w0 / WStep>{},
                                                     k0);

                                semaAccumFree->template signal<0>();
                            }
                            else
                            {
                                semaAccumReady->template wait<0>();
                                AccDataVec acc_vec[NumYX] = {};
                                typename AccDataVec::type* accdata_thread_vec_ptr[NumYX];
                                static_for<0, NumYX, 1>{}([&](auto i) {
                                    constexpr index_t accum_offset = get_accum_offset_transposed(
                                        Number<h0>{}, Number<w0>{}, Number<k0>{}, Number<i>{});
                                    acc_vec[i] = accum_thread_buf.GetVectorTypeReference(
                                        Number<accum_offset>{});
                                    accdata_thread_vec_ptr[i] =
                                        &acc_vec[i].template AsType<typename AccDataVec::type>()(
                                            Number<0>{});
                                });
                                wcnn_conv.ShuffleConv2TransposedData(accdata_thread_vec_ptr);
                                static_for<0, NumYX, 1>{}([&](auto i) {
                                    element_op_.Run(ds_block_buf,
                                                    acc_vec[i],
                                                    Number<h0 * 2 + y_offset[i]>{},
                                                    Number<w0 * 2 + x_offset[i]>{},
                                                    k0);
                                    element_next_op_.Run(acc_vec[i],
                                                         out_thread_buf,
                                                         Number<h0 * 2 + y_offset[i]>{},
                                                         Number<w0 * 2 + x_offset[i]>{},
                                                         k0);
                                });
                                semaAccumFree->template signal<0>();
                            }
                        });
                    });
                }
            });
        }
        else
        {
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                const typename WeiDataVec::type* weight_thread_vec_ptr[CRepeat / CStep][4];

                // Load weights
                WeiDataVec weight_thread_vec[CRepeat / CStep][4];
                static_for<0, CRepeat, CStep>{}([&](auto c0) {
                    load_weight_data(
                        k0, c0, weight_thread_vec[c0 / CStep], weight_thread_vec_ptr[c0 / CStep]);
                });

                InDataVec indata_thread_vec[4];
                static_for<0, HRepeat, HStep>{}([&](auto h0) {
                    static_for<0, WRepeat, WStep>{}([&](auto w0) {
                        if constexpr(Transposed == false)
                        {
                            constexpr index_t accum_offset =
                                get_accum_offset(Number<h0>{}, Number<w0>{}, Number<k0>{});
                            AccDataVec acc_vec = {};
                            if constexpr(HasMainLoop{})
                            {
                                acc_vec =
                                    accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{});
                            }
                            static_for<0, CRepeat, CStep>{}([&](auto c0) {
                                load_in_data(h0, w0, c0, indata_thread_vec);
                                wcnn_conv.conv_instr.Run(*weight_thread_vec_ptr[c0 / CStep][0],
                                                         indata_thread_vec_ptr,
                                                         acc_vec,
                                                         I0);
                            });
                            if constexpr(IsLast{})
                            {
                                element_op_.Run(ds_block_buf,
                                                acc_vec,
                                                Number<h0 / HStep>{},
                                                Number<w0 / WStep>{},
                                                k0);
                                element_next_op_.Run(acc_vec,
                                                     out_thread_buf,
                                                     Number<h0 / HStep>{},
                                                     Number<w0 / WStep>{},
                                                     k0);
                            }
                            else
                            {
                                accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{}) =
                                    acc_vec;
                            }
                        }
                        else
                        {
                            AccDataVec acc_vec[NumYX] = {};
                            typename AccDataVec::type* accdata_thread_vec_ptr[NumYX];
                            if constexpr(HasMainLoop{})
                            {
                                static_for<0, NumYX, 1>{}([&](auto i) {
                                    constexpr index_t accum_offset = get_accum_offset_transposed(
                                        Number<h0>{}, Number<w0>{}, Number<k0>{}, Number<i>{});
                                    acc_vec[i] = accum_thread_buf.GetVectorTypeReference(
                                        Number<accum_offset>{});
                                });
                            }
                            static_for<0, CRepeat, CStep>{}([&](auto c0) {
                                load_in_data(h0, w0, c0, indata_thread_vec);
                                static_for<0, NumYX, 1>{}([&](auto i) {
                                    constexpr index_t tapIdx =
                                        i / wcnn_conv.GetNumWeightTapPerWave();
                                    constexpr bool isHigh = i % wcnn_conv.GetNumWeightTapPerWave();
                                    constexpr auto Mod    = []() {
                                        if constexpr(isHigh)
                                            return I1;
                                        else
                                            return I0;
                                    }();
                                    accdata_thread_vec_ptr[i] =
                                        &acc_vec[i].template AsType<typename AccDataVec::type>()(
                                            I0);

                                    wcnn_conv.conv_instr.Run(
                                        *weight_thread_vec_ptr[c0 / CStep][tapIdx],
                                        indata_thread_vec_ptr,
                                        acc_vec[i],
                                        Mod);
                                });
                            });

                            if constexpr(IsLast{})
                            {
                                wcnn_conv.ShuffleConv2TransposedData(accdata_thread_vec_ptr);
                                static_for<0, NumYX, 1>{}([&](auto i) {
                                    element_op_.Run(ds_block_buf,
                                                    acc_vec[i],
                                                    Number<h0 * 2 + y_offset[i]>{},
                                                    Number<w0 * 2 + x_offset[i]>{},
                                                    k0);
                                    element_next_op_.Run(acc_vec[i],
                                                         out_thread_buf,
                                                         Number<h0 * 2 + y_offset[i]>{},
                                                         Number<w0 * 2 + x_offset[i]>{},
                                                         k0);
                                });
                            }
                            else
                            {
                                static_for<0, NumYX, 1>{}([&](auto i) {
                                    constexpr index_t accum_offset = get_accum_offset_transposed(
                                        Number<h0>{}, Number<w0>{}, Number<k0>{}, Number<i>{});
                                    accum_thread_buf.GetVectorTypeReference(
                                        Number<accum_offset>{}) = acc_vec[i];
                                });
                            }
                        }
                    });
                });
            });
        }
    }

#pragma clang diagnostic pop

    template <typename InDataGridDesc,
              typename InDataBlockBuffer,
              typename ExchangeDataBlockBuffer,
              typename WaveGroupSemaphores,
              typename LaneShareData>
    __device__ void static exchange_neighbor_data(const InDataGridDesc& in_grid_desc,
                                                  InDataBlockBuffer& indata_block_buf,
                                                  ExchangeDataBlockBuffer& prev_block_buf,
                                                  ExchangeDataBlockBuffer& next_block_buf,
                                                  LaneShareData& dataFromPrev,
                                                  WaveGroupSemaphores& semFromPrev,
                                                  LaneShareData& dataFromNext,
                                                  WaveGroupSemaphores& semFromNext)
    {
#if defined(__gfx13__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
        auto* pSemPrev = (__attribute__((address_space(3))) WavegroupSemaphore<0>*)(&semFromPrev);
        auto* pSemNext = (__attribute__((address_space(3))) WavegroupSemaphore<0>*)(&semFromNext);
#pragma clang diagnostic pop
        constexpr index_t HRepeatIn = HPerWaveIn / HPerWcnn;
        // H x W x C -> W0 x C0 x H0 x H1 x H2 x W1 x C1
        const auto W0                  = in_grid_desc.GetLength(I0);
        const auto W1                  = in_grid_desc.GetLength(I5);
        const auto Gridsize_W          = math::integer_divide_ceil(W0 * W1, WPerBlock);
        constexpr index_t numWaveGroup = ThisThreadBlock::GetNumWaveGroups();
        constexpr index_t hSubTile     = NumSubTilePerImage;

        uint32_t blockId     = blockIdx.x;
        uint32_t waveGroupId = __builtin_amdgcn_wavegroup_id();

        bool isChainStartVal = __builtin_amdgcn_spatial_cluster_is_chain_start();
        bool isChainStart    = __builtin_amdgcn_readfirstlane(isChainStartVal);
        bool isChainEndVal   = __builtin_amdgcn_spatial_cluster_is_chain_end();
        bool isChainEnd      = __builtin_amdgcn_readfirstlane(isChainEndVal);

        static_for<0, CRepeat, 1>{}([&](auto c0) {
            static_for<0, HRepeatIn, 1>{}([&](auto h0) {
                static_for<0, hSubTile, 1>{}([&](auto h1) {
                    constexpr index_t offset =
                        indata_border_wave_desc_.CalculateOffset(make_tuple(c0, h0, h1));
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
                    auto* pPrev =
                        (__attribute__((address_space(10))) int*)(&dataFromPrev[Number<offset>{}]);
                    auto* pNext =
                        (__attribute__((address_space(10))) int*)(&dataFromNext[Number<offset>{}]);
#pragma clang diagnostic pop

                    // Head of Neighbor Chain or Left_W_border in the Grid
                    if(!isChainStart &&
                       ((waveGroupId + blockId * numWaveGroup) % (Gridsize_W * numWaveGroup) != 0))
                    {
                        constexpr index_t w0_indata_offset = indata_wave_desc_.CalculateOffset(
                            make_tuple(I0, c0, h0, h1, I0, I0, I0));
                        const int32_t* pExchange_head_data = reinterpret_cast<const int32_t*>(
                            &indata_block_buf[Number<w0_indata_offset>{}]);
                        __builtin_amdgcn_spatial_cluster_send_prev(
                            *pExchange_head_data, pNext, nullptr, pPrev, nullptr, 0);
                    }

                    // Tail of Neighbor Chain or Right_W_border in the Grid
                    if(!isChainEnd &&
                       ((waveGroupId + blockId * numWaveGroup + 1) % (Gridsize_W * numWaveGroup) !=
                        0))
                    {
                        constexpr index_t w3_indata_offset = indata_wave_desc_.CalculateOffset(
                            make_tuple(Number<WRepeat - 1>{}, c0, h0, h1, I0, I0, I0));
                        const int32_t* pExchange_tail_data = reinterpret_cast<const int32_t*>(
                            &indata_block_buf[Number<w3_indata_offset>{}]);
                        __builtin_amdgcn_spatial_cluster_send_next(
                            *pExchange_tail_data, pPrev, nullptr, pNext, nullptr, 0);
                    }
                });
            });
        });

        __builtin_amdgcn_spatial_cluster_signal_prev(pSemNext, pSemPrev);
        __builtin_amdgcn_spatial_cluster_signal_next(pSemPrev, pSemNext);
        __builtin_amdgcn_s_sema_wait(&semFromPrev);
        __builtin_amdgcn_s_sema_wait(&semFromNext);

        static_for<0, CRepeat, 1>{}([&](auto c0) {
            static_for<0, HRepeatIn, 1>{}([&](auto h0) {
                static_for<0, hSubTile, 1>{}([&](auto h1) {
                    constexpr index_t offset =
                        indata_border_wave_desc_.CalculateOffset(make_tuple(c0, h0, h1));
                    prev_block_buf(Number<offset>{}) = dataFromPrev[Number<offset>{}];
                    next_block_buf(Number<offset>{}) = dataFromNext[Number<offset>{}];
                });
            });
        });
#else
        ignore = in_grid_desc;
        ignore = indata_block_buf;
        ignore = prev_block_buf;
        ignore = next_block_buf;
        ignore = dataFromPrev;
        ignore = semFromPrev;
        ignore = dataFromNext;
        ignore = semFromNext;
#endif
    };

    protected:
    // Thread descriptor for weight and input data
    static constexpr auto weight_thread_desc_ = make_naive_tensor_descriptor_packed(make_tuple(
        I1, I1, I1, I1, Number<NumSubTilesPerWeightTap>{}, Number<NumWeightCompPerTile>{}));

    static constexpr auto NumDataSubTiles = wcnn_conv.GetNumSubTilesPerImageTile();
    static constexpr auto NumDataTiles    = wcnn_conv.GetNumImageTilesInVertical();
    static constexpr auto indata_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                       I1,
                                                       Number<NumDataTiles>{},
                                                       Number<NumDataSubTiles>{},
                                                       I1,
                                                       I1,
                                                       Number<NumDataCompPerTile>{}));

    // C[H, W, K, NumAccComp]
    static constexpr auto accum_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<HRepeatOut>{},
                   Number<WRepeatOut>{},
                   Number<KRepeat>{},
                   Number<wcnn_conv.GetNumAccumComponents()>{}));

    static constexpr auto out_tensor_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<HRepeatOut>{},
                   Number<WRepeatOut>{},
                   Number<KRepeat>{},
                   Number<wcnn_conv.GetNumOutTensorComponents()>{}));

    // 3X3 for phase_1
    static constexpr auto in_1_from_thread_desc_ = make_naive_tensor_descriptor_packed(make_tuple(
        Number<HRepeatOut>{}, Number<WRepeatOut>{}, Number<wcnn_conv.GetNumAccumComponents()>{}));

    // 1X1 for phase_2: REPEAT:4 x ACCSIZE:8
    static constexpr auto in_2_from_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<HRepeatOut>{},
                   Number<WRepeatOut>{},
                   Number<4>{},
                   Number<wcnn_conv.GetNumAccumComponents()>{}));

    // Initialize thread copy classes
    using WeiDataThreadLdsCopy = ThreadwiseTensorSliceTransfer_v4<
        WeiDataType,
        WeiDataType,
        decltype(weight_wave_desc_),
        decltype(weight_thread_desc_),
        Sequence<1, 1, 1, 1, NumSubTilesPerWeightTap, NumWeightCompPerTile>,
        Sequence<0, 1, 2, 3, 4, 5>,
        5,
        NumWeightCompPerTile,
        NumWeightCompPerTile>;

    WeiDataThreadLdsCopy weight_thread_copy_;

    using InDataThreadLdsCopy = ThreadwiseTensorSliceTransfer_v4<
        InDataType,
        InDataType,
        decltype(indata_wave_desc_),
        decltype(indata_thread_desc_),
        Sequence<1, 1, NumDataTiles, NumDataSubTiles, 1, 1, NumDataCompPerTile>,
        Sequence<0, 1, 2, 3, 4, 5, 6>,
        6,
        NumDataCompPerTile,
        NumDataCompPerTile>;
    InDataThreadLdsCopy indata_thread_copy_;

    template <bool IsSuba, bool IsFma>
    struct BlockwiseElementSelect
    {
        using type = BlockwiseElementPassThroughWcnn;
    };
    template <>
    struct BlockwiseElementSelect<true, false>
    {
        using type = BlockwiseElementSbaWcnn<ThisThreadBlock,
                                             AccDataType,
                                             DsBlockDesc, // {Scale, Bias}
                                             HPerBlock,
                                             WPerBlock,
                                             KPerBlock,
                                             HPerWcnn,
                                             WPerWcnn,
                                             KPerWcnn,
                                             HRepeat,
                                             WRepeat,
                                             KRepeat,
                                             DsEnableLds,
                                             Aco,
                                             AccBlockwiseOperation>;
    };

    template <>
    struct BlockwiseElementSelect<false, true>
    {
        using type = BlockwiseElementFmaWcnn<ThisThreadBlock,
                                             DsDataType,
                                             AccDataType,
                                             DsBlockDesc, // {Scale, Residual}
                                             HPerBlock,
                                             WPerBlock,
                                             KPerBlock,
                                             HPerWcnn,
                                             WPerWcnn,
                                             HRepeat,
                                             WRepeat,
                                             KRepeat,
                                             DsEnableLds,
                                             Aco,
                                             AccBlockwiseOperation>;
    };

    template <>
    struct BlockwiseElementSelect<true, true>
    {
        using type = BlockwiseElementSbaFmaWcnn<ThisThreadBlock,
                                                DsDataType,
                                                AccDataType,
                                                DsBlockDesc,
                                                HPerBlock,
                                                WPerBlock,
                                                KPerBlock,
                                                HPerWcnn,
                                                WPerWcnn,
                                                KPerWcnn,
                                                HRepeat,
                                                WRepeat,
                                                KRepeat,
                                                DsEnableLds,
                                                Aco,
                                                AccBlockwiseOperation>;
    };

    using BlockwiseElementOpType =
        typename BlockwiseElementSelect<AccBlockwiseOperation::IsSuba,
                                        AccBlockwiseOperation::IsFma>::type;
    BlockwiseElementOpType element_op_;

    template <bool shouldCvtToTensor>
    struct BlockwiseNextElementSelect
    {
        using type = BlockwiseElementOutPassThroughWcnn<decltype(accum_thread_desc_)>;
    };
    template <>
    struct BlockwiseNextElementSelect<true>
    {
        using type = BlockwiseElementCvtTensorWcnn<ThisThreadBlock,
                                                   EDataType,
                                                   AccDataType,
                                                   decltype(out_tensor_thread_desc_),
                                                   decltype(accum_thread_desc_),
                                                   HPerBlock,
                                                   WPerBlock,
                                                   KPerBlock,
                                                   HPerWcnn,
                                                   WPerWcnn,
                                                   KPerWcnn,
                                                   CPerWcnn,
                                                   HRepeat,
                                                   WRepeat,
                                                   KRepeat,
                                                   Aco,
                                                   AccBlockwiseNextOperation>;
    };
    using BlockwiseElementNextOpType =
        typename BlockwiseNextElementSelect<AccBlockwiseNextOperation::cvt_to_tensor>::type;

    BlockwiseElementNextOpType element_next_op_;

    public:
    template <typename DsGridDesc>
    __host__ __device__ static constexpr auto
    MakeDsGridBlockDescriptor(const DsGridDesc& ds_grid_desc)
    {
        return BlockwiseElementOpType::MakeDsGridBlockDescriptor(ds_grid_desc);
    }

    template <index_t conv_phase, typename DsGridDesc>
    __host__ __device__ static constexpr auto
    MakeSingleDsGridBlockDescriptor(const DsGridDesc& ds_grid_desc)
    {
        return BlockwiseElementOpType::template MakeSingleDsGridBlockDescriptor<conv_phase>(
            ds_grid_desc);
    }

    __host__ __device__ static constexpr auto CalculateThreadOriginDataIndex()
    {
        return BlockwiseElementOpType::CalculateThreadOriginDataIndex();
    }

    __host__ __device__ static constexpr auto GetDsWaveDescLength()
    {
        return BlockwiseElementOpType::GetDsWaveDescLength();
    }

    __host__ __device__ static constexpr auto GetDsWaveDesc()
    {
        return BlockwiseElementOpType::ds_wave_desc_;
    }

    struct SharedMemTrait
    {
        // LDS allocation for A and B: be careful of alignment
        static constexpr index_t max_lds_align = 8;

        static constexpr index_t in_block_space_size_aligned =
            InDataEnableLds ? math::integer_least_multiple(InDataBlockDesc{}.GetElementSpaceSize(),
                                                           max_lds_align)
                            : 0;
        static constexpr index_t wei_block_space_size_aligned =
            WeiDataEnableLds ? math::integer_least_multiple(
                                   WeiDataBlockDesc{}.GetElementSpaceSize(), max_lds_align)
                             : 0;

        static constexpr auto ds_block_space_size_aligned =
            BlockwiseElementOpType::SharedMemTrait::ds_block_space_size_aligned;

        static constexpr index_t in_block_space_offset  = 0;
        static constexpr index_t wei_block_space_offset = in_block_space_size_aligned;

        // LDS allocation for Ds in LDS
        static constexpr index_t ds_base_offset =
            (wei_block_space_offset + wei_block_space_size_aligned) * sizeof(WeiDataType) /
            sizeof(AccDataType);

        static constexpr auto ds_block_space_offset = make_tuple(
            BlockwiseElementOpType::SharedMemTrait::ds_block_space_offset[I0] + ds_base_offset,
            BlockwiseElementOpType::SharedMemTrait::ds_block_space_offset[I1] + ds_base_offset);

        // LDS allocation for C shuffle in LDS
        static constexpr index_t acc_block_space_size =
            GetAccBlockDescriptor().GetElementSpaceSize();

        static constexpr index_t acc_block_space_offset = 0;

        static constexpr index_t lds_size =
            math::max(acc_block_space_size * sizeof(AccDataType),
                      in_block_space_size_aligned * sizeof(InDataType) +
                          wei_block_space_size_aligned * sizeof(WeiDataType) +
                          BlockwiseElementOpType::SharedMemTrait::lds_size);

        static constexpr index_t out_tensor_block_space_size =
            GetAccBlockDescriptor().GetElementSpaceSize();
        static constexpr index_t out_tensor_block_space_offset = 0;
    };

    struct LaneSharedMemTrait
    {
        static constexpr index_t max_lane_shared_align = 4;

        static constexpr index_t in_block_space_size_aligned =
            EnableWaveGroup && (InDataEnableLds == false)
                ? math::integer_least_multiple(InDataWaveDesc{}.GetElementSpaceSize() *
                                                   sizeof(InDataType),
                                               max_lane_shared_align)
                : 0;
        static constexpr index_t wei_block_space_size_aligned =
            EnableWaveGroup && (WeiDataEnableLds == false)
                ? math::integer_least_multiple(WeiDataWaveDesc{}.GetElementSpaceSize() *
                                                   sizeof(WeiDataType),
                                               max_lane_shared_align)
                : 0;
        static constexpr index_t ds_block_space_size_aligned =
            BlockwiseElementOpType::LaneSharedMemTrait::lane_shared_size;

        static constexpr auto ds_block_space_offset =
            BlockwiseElementOpType::LaneSharedMemTrait::ds_block_space_offset;

        static constexpr index_t out_block_space_aligned = math::integer_least_multiple(
            HRepeatOut * WRepeatOut * KRepeat * wcnn_conv.GetNumAccumComponents() *
                wcnn_conv.template SizeOfBits<EDataType>() / 8,
            max_lane_shared_align);

        static constexpr index_t acc_ring_size                = 2;
        static constexpr index_t acc_ring_block_space_aligned = math::integer_least_multiple(
            (FilterSize == 2 && Transposed)
                ? acc_ring_size * 4 * wcnn_conv.GetNumAccumComponents() * sizeof(AccDataType)
                : acc_ring_size * wcnn_conv.GetNumAccumComponents() * sizeof(AccDataType),
            max_lane_shared_align);

        static constexpr index_t acc_block_space_aligned = math::integer_least_multiple(
            HRepeatOut * WRepeatOut * KRepeat * wcnn_conv.GetNumAccumComponents() *
                sizeof(AccDataType),
            max_lane_shared_align);
    };
};
} // namespace ck
