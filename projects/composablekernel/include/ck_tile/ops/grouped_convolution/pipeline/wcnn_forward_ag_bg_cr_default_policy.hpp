// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/grouped_convolution/warp/warp_conv_dispatcher.hpp"
#include "ck_tile/ops/grouped_convolution/block/block_wcnn_asmem_bsmem_creg.hpp"

namespace ck_tile {
struct WcnnForwardDefaultPolicy
{
    template <typename Problem,
              typename DataType,
              index_t MNPerBlock,
              index_t XPerTile,
              bool IsWave32Host>
    CK_TILE_HOST_DEVICE static constexpr auto GetGlobalVectorLoadSize()
    {
        constexpr index_t BlockSize = IsWave32Host ? Problem::BlockSize / 2 : Problem::BlockSize;
        constexpr index_t elements_per_thread = MNPerBlock * XPerTile / BlockSize;
        constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<DataType>>::PackedSize;

        // Assume DataType is even!
        if constexpr(XPerTile % (PackedSize * 16 / sizeof(DataType)) == 0 &&
                     elements_per_thread % (PackedSize * 16 / sizeof(DataType)) == 0)
        {
            return (PackedSize * 16 / sizeof(DataType));
        }
        else if constexpr(XPerTile % (PackedSize * 8 / sizeof(DataType)) == 0 &&
                          elements_per_thread % (PackedSize * 8 / sizeof(DataType)) == 0)
        {
            return (PackedSize * 8 / sizeof(DataType));
        }
        else if constexpr(sizeof(DataType) >= PackedSize * 4 &&
                          XPerTile % (PackedSize * 4 / sizeof(DataType)) == 0 &&
                          elements_per_thread % (PackedSize * 4 / sizeof(DataType)) == 0)
        {
            return (PackedSize * 4 / sizeof(DataType));
        }
        else if constexpr(sizeof(DataType) >= PackedSize * 2 &&
                          XPerTile % (PackedSize * 2 / sizeof(DataType)) == 0 &&
                          elements_per_thread % (PackedSize * 2 / sizeof(DataType)) == 0)
        {
            return (PackedSize * 2 / sizeof(DataType));
        }
        else
        {
            return PackedSize;
        }
    }

    template <typename Problem, bool IsWave32Host = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeA()
    {
        using ADataType = typename Problem::ADataType;
        return GetGlobalVectorLoadSize<Problem,
                                       ADataType,
                                       Problem::BlockWcnnShape::HPerBlock *
                                           Problem::BlockWcnnShape::WPerBlock,
                                       Problem::BlockWcnnShape::CPerBlock,
                                       IsWave32Host>();
    }

    template <typename Problem, bool IsWave32Host = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeB()
    {
        using BDataType             = typename Problem::BDataType;
        constexpr index_t KPerBlock = Problem::BlockWcnnShape::KPerBlock;
        constexpr index_t CPerBlock = Problem::BlockWcnnShape::CPerBlock;
        constexpr index_t FilterY   = Problem::FilterY;
        constexpr index_t FilterX   = Problem::FilterX;
        return GetGlobalVectorLoadSize<Problem,
                                       BDataType,
                                       KPerBlock * FilterY * FilterX,
                                       CPerBlock,
                                       IsWave32Host>();
    }

    template <typename Problem, bool IsWave32Host = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeC()
    {
        using OutDataType           = typename Problem::OutDataType;
        constexpr index_t KPerBlock = Problem::BlockWcnnShape::KPerBlock;
        constexpr index_t HPerBlock = Problem::BlockWcnnShape::HPerBlock;
        constexpr index_t WPerBlock = Problem::BlockWcnnShape::WPerBlock;
        return GetGlobalVectorLoadSize<Problem,
                                       OutDataType,
                                       HPerBlock * WPerBlock,
                                       KPerBlock,
                                       IsWave32Host>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        // TODO: check lds bank conflict
        constexpr index_t HPerBlock    = Problem::BlockWcnnShape::HPerBlock;
        constexpr index_t WPerBlock    = Problem::BlockWcnnShape::WPerBlock;
        constexpr index_t CPerBlock    = Problem::BlockWcnnShape::CPerBlock;
        constexpr index_t VecLoadSizeA = GetVectorSizeA<Problem>();
        return make_naive_tensor_descriptor(
            make_tuple(number<HPerBlock * WPerBlock>{}, number<CPerBlock>{}),
            make_tuple(number<CPerBlock>{}, number<1>{}),
            number<VecLoadSizeA>{},
            number<1>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        // TODO: check lds bank conflict
        constexpr index_t KPerBlock    = Problem::BlockWcnnShape::KPerBlock;
        constexpr index_t CPerBlock    = Problem::BlockWcnnShape::CPerBlock;
        constexpr index_t FilterY      = Problem::FilterY;
        constexpr index_t FilterX      = Problem::FilterX;
        constexpr index_t VecLoadSizeB = GetVectorSizeB<Problem>();
        return make_naive_tensor_descriptor(
            make_tuple(number<KPerBlock * FilterY * FilterX>{}, number<CPerBlock>{}),
            make_tuple(number<CPerBlock>{}, number<1>{}),
            number<VecLoadSizeB>{},
            number<1>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        constexpr index_t HPerBlock = Problem::BlockWcnnShape::HPerBlock;
        constexpr index_t WPerBlock = Problem::BlockWcnnShape::WPerBlock;
        constexpr index_t CPerBlock = Problem::BlockWcnnShape::CPerBlock;
        constexpr index_t BlockSize = Problem::BlockSize;

        constexpr index_t VecLoadSize =
            Problem::FixedVectorSize ? Problem::VectorSizeA : GetVectorSizeA<Problem>();

        using TileEncodingPattern =
            tile_distribution_encoding_pattern_2d<BlockSize,
                                                  HPerBlock * WPerBlock,
                                                  CPerBlock,
                                                  VecLoadSize,
                                                  tile_distribution_pattern::warp_raked>;

        return TileEncodingPattern::make_2d_static_tile_distribution();
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        constexpr index_t KPerBlock = Problem::BlockWcnnShape::KPerBlock;
        constexpr index_t CPerBlock = Problem::BlockWcnnShape::CPerBlock;
        constexpr index_t BlockSize = Problem::BlockSize;

        constexpr index_t FilterY = Problem::FilterY;
        constexpr index_t FilterX = Problem::FilterX;

        constexpr index_t VecLoadSize =
            Problem::FixedVectorSize ? Problem::VectorSizeB : GetVectorSizeB<Problem>();

        using TileEncodingPattern =
            tile_distribution_encoding_pattern_2d<BlockSize,
                                                  KPerBlock * FilterY * FilterX,
                                                  CPerBlock,
                                                  VecLoadSize,
                                                  tile_distribution_pattern::warp_raked>;

        return TileEncodingPattern::make_2d_static_tile_distribution();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeA()
    {
        using ADataType                  = typename Problem::ADataType;
        constexpr index_t PackedSize     = numeric_traits<ADataType>::PackedSize;
        constexpr auto in_lds_block_desc = MakeALdsBlockDescriptor<Problem>();
        constexpr index_t smem_size_a    = integer_least_multiple(
            in_lds_block_desc.get_element_space_size() * sizeof(ADataType) / PackedSize, 16);
        return smem_size_a;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemSizeB()
    {
        using BDataType                 = typename Problem::BDataType;
        constexpr index_t PackedSize    = numeric_traits<BDataType>::PackedSize;
        constexpr auto b_lds_block_desc = MakeBLdsBlockDescriptor<Problem>();
        constexpr index_t smem_size_b   = integer_least_multiple(
            b_lds_block_desc.get_element_space_size() * sizeof(BDataType) / PackedSize, 16);
        return smem_size_b;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemSize()
    {
        return GetSmemSizeA<Problem>() + GetSmemSizeB<Problem>();
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetBlockWcnn()
    {
        using ADataType             = typename Problem::ADataType;
        using BDataType             = typename Problem::BDataType;
        using AccDataType           = typename Problem::AccDataType;
        using OutDataType           = typename Problem::OutDataType;
        constexpr index_t HPerWcnn  = Problem::BlockWcnnShape::HPerWcnn;
        constexpr index_t WPerWcnn  = Problem::BlockWcnnShape::WPerWcnn;
        constexpr index_t CPerBlock = Problem::BlockWcnnShape::CPerBlock;
        constexpr index_t FilterY   = Problem::FilterY;
        constexpr index_t FilterX   = Problem::FilterX;
        constexpr index_t DilationY = Problem::DilationY;
        constexpr index_t DilationX = Problem::DilationX;

        constexpr auto IterNum = []() constexpr {
            if constexpr(FilterX == 1 && FilterY == 1)
            {
                constexpr index_t CPerWcnn =
                    WarpConvDispatcher<ADataType,
                                       BDataType,
                                       AccDataType,
                                       (sizeof(OutDataType) > 1), // AcoFlag
                                       HPerWcnn,
                                       WPerWcnn,
                                       FilterY,
                                       FilterX,
                                       DilationY,
                                       DilationX>::CPerWcnn;
                static_assert(CPerBlock % CPerWcnn == 0, "CPerBlock must be divisible by CPerWcnn");
                constexpr index_t IterLoop = CPerBlock / CPerWcnn;
                if constexpr(IterLoop % 4 == 0)
                {
                    return 4;
                }
                else if constexpr(IterLoop % 2 == 0)
                {
                    return 2;
                }
                else
                {
                    return 1;
                }
            }
            else // TODO need to change if want to support 3x3 convolution
            {
                return 1;
            }
        }();

        using WarpWcnn = WarpConvDispatcher<ADataType,
                                            BDataType,
                                            AccDataType,
                                            (sizeof(OutDataType) > 1), // AcoFlag
                                            HPerWcnn,
                                            WPerWcnn,
                                            FilterY,
                                            FilterX,
                                            DilationY,
                                            DilationX,
                                            IterNum>;
        return BlockWcnnASmemBSmemCReg<Problem, WarpWcnn>{};
    }
};
} // namespace ck_tile
