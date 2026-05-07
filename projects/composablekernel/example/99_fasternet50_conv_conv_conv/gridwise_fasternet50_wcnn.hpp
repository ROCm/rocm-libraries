// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "gridwise_fasternet50_wcnn_pipeline.hpp"
#include "blockwise_conv_wcnn_fasternet50.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_async.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseOp,
          typename InDataType,
          typename WeiDataType,
          typename WeiPointer,
          typename DsPointer,
          typename AccDataType,
          typename EDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename AccElementwiseOperation,
          typename InGridDesc,
          typename WeiGridDesc,
          typename DsGridDesc,
          typename EGridDesc,
          typename Block2CTileMap,
          typename ComputePtrOffsetOfBatch0,
          typename ComputePtrOffsetOfBatch1,
          typename ComputePtrOffsetOfBatch2,
          bool HasMainBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_fasternet50_wcnn(const InDataType* __restrict__ p_in_grid,
                            WeiPointer p_wei_grid,
                            DsPointer p_ds_grid,
                            EDataType* __restrict__ p_e_grid,
                            const InElementwiseOperation in_element_op,
                            const WeiElementwiseOperation wei_element_op,
                            const AccElementwiseOperation acc_element_op,
                            const index_t batch_count,
                            const InGridDesc in_grid_desc,
                            const WeiGridDesc wei_grid_desc,
                            const DsGridDesc ds_grid_desc,
                            const EGridDesc e_grid_desc,
                            const Block2CTileMap block_2_ctile_map,
                            const ComputePtrOffsetOfBatch0 compute_ptr_offset_of_batch_0,
                            const ComputePtrOffsetOfBatch1 compute_ptr_offset_of_batch_1,
                            const ComputePtrOffsetOfBatch2 compute_ptr_offset_of_batch_2)
{
#if defined(__gfx13__)
    // offset base pointer for each work-group
    static constexpr index_t NumDTensor_0 = tuple_element_t<0, DsGridDesc>::Size();
    static constexpr index_t NumDTensor_1 = tuple_element_t<1, DsGridDesc>::Size();
    static constexpr index_t NumDTensor_2 = tuple_element_t<2, DsGridDesc>::Size();

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);

    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);
    const auto& as_group_offset        = compute_ptr_offset_of_batch_0.GetAPtrOffset(g_idx);
    const auto& bs_0_group_offset      = compute_ptr_offset_of_batch_0.GetBPtrOffset(g_idx);
    const auto& bs_1_group_offset      = compute_ptr_offset_of_batch_1.GetBPtrOffset(g_idx);
    const auto& bs_2_group_offset      = compute_ptr_offset_of_batch_2.GetBPtrOffset(g_idx);
    const long_index_t in_batch_offset = amd_wave_read_first_lane(as_group_offset);

#if defined(CASCADE_1X_OUT)
    const long_index_t e_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch_0.GetEPtrOffset(g_idx));
#elif defined(CASCADE_2X_OUT)
    const long_index_t e_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch_1.GetEPtrOffset(g_idx));
#else
    const long_index_t e_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch_2.GetEPtrOffset(g_idx));
#endif

    const auto ds_batch_offset_0 = compute_ptr_offset_of_batch_0.GetDsPtrOffset(g_idx);
    const auto ds_batch_offset_1 = compute_ptr_offset_of_batch_1.GetDsPtrOffset(g_idx);
    const auto ds_batch_offset_2 = compute_ptr_offset_of_batch_2.GetDsPtrOffset(g_idx);

    const index_t conv_0_in_lds_size =
        GridwiseOp::BlockwiseConv_0::SharedMemTrait::in_block_space_size_aligned *
        sizeof(InDataType);
    const index_t conv_0_weight_lds_size =
        GridwiseOp::BlockwiseConv_0::SharedMemTrait::wei_block_space_size_aligned *
        sizeof(WeiDataType);
    const index_t conv_1_weight_lds_size =
        GridwiseOp::BlockwiseConv_1::SharedMemTrait::wei_block_space_size_aligned *
        sizeof(WeiDataType);
    const index_t conv_2_weight_lds_size =
        GridwiseOp::BlockwiseConv_2::SharedMemTrait::wei_block_space_size_aligned *
        sizeof(WeiDataType);
    const index_t conv_0_acc_lds_size =
        GridwiseOp::BlockwiseConv_0::SharedMemTrait::acc_block_space_size * sizeof(AccDataType);
    const index_t conv_1_acc_lds_size =
        GridwiseOp::BlockwiseConv_1::SharedMemTrait::acc_block_space_size * sizeof(AccDataType);
    const index_t conv_2_acc_lds_size =
        GridwiseOp::BlockwiseConv_2::SharedMemTrait::acc_block_space_size * sizeof(AccDataType);

    const index_t conv_0_lds_size =
        math::max(conv_0_acc_lds_size, conv_0_in_lds_size + conv_0_weight_lds_size);
    const index_t conv_1_lds_size =
        math::max(conv_1_acc_lds_size, conv_0_in_lds_size + conv_1_weight_lds_size);
    const index_t conv_2_lds_size = math::max(conv_2_acc_lds_size, conv_2_weight_lds_size);

    static constexpr index_t lds_size =
        math::max(conv_0_lds_size, conv_1_lds_size, conv_2_lds_size);

    __shared__ char p_shared[lds_size];

    WeiPointer p_wei_grid_grp;
    p_wei_grid_grp(I0) = p_wei_grid[I0] + amd_wave_read_first_lane(bs_0_group_offset);
    p_wei_grid_grp(I1) = p_wei_grid[I1] + amd_wave_read_first_lane(bs_1_group_offset);
    p_wei_grid_grp(I2) = p_wei_grid[I2] + amd_wave_read_first_lane(bs_2_group_offset);

    DsPointer p_ds_grid_grp;

    static_for<0, NumDTensor_0, 1>{}(
        [&](auto i) { p_ds_grid_grp(I0)(i) = p_ds_grid[I0][i] + ds_batch_offset_0[i]; });
    static_for<0, NumDTensor_1, 1>{}(
        [&](auto i) { p_ds_grid_grp(I1)(i) = p_ds_grid[I1][i] + ds_batch_offset_1[i]; });
    static_for<0, NumDTensor_2, 1>{}(
        [&](auto i) { p_ds_grid_grp(I2)(i) = p_ds_grid[I2][i] + ds_batch_offset_2[i]; });

    GridwiseOp::template Run<HasMainBlockLoop>(p_in_grid + in_batch_offset,
                                               p_wei_grid_grp,
                                               p_ds_grid_grp,
                                               p_e_grid + e_batch_offset,
                                               p_shared,
                                               nullptr,
                                               in_grid_desc,
                                               wei_grid_desc,
                                               ds_grid_desc,
                                               e_grid_desc,
                                               in_element_op,
                                               wei_element_op,
                                               acc_element_op,
                                               block_2_ctile_map);
#else
    ignore = p_in_grid;
    ignore = p_wei_grid;
    ignore = p_ds_grid;
    ignore = p_e_grid;
    ignore = in_element_op;
    ignore = wei_element_op;
    ignore = acc_element_op;
    ignore = batch_count;
    ignore = in_grid_desc;
    ignore = wei_grid_desc;
    ignore = ds_grid_desc;
    ignore = e_grid_desc;
    ignore = compute_ptr_offset_of_batch_0;
    ignore = compute_ptr_offset_of_batch_1;
    ignore = compute_ptr_offset_of_batch_2;
    ignore = block_2_ctile_map;
#endif
}

template <index_t BlockSize,
          typename InDataType,
          typename WeiDataType,
          typename DsDataType,
          typename AccDataType,
          typename EDataType,
          typename InGridDesc,
          typename WeiGridDesc,
          typename DsGridDesc,
          typename EGridDesc,
          typename DsLayout,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename AccElementwiseOperation,
          index_t HPerBlock,
          index_t WPerBlock,
          typename CPerBlock,
          typename KPerBlock,
          index_t HRepeat,
          index_t WRepeat,
          index_t HPerWcnn,
          index_t WPerWcnn,
          typename FilterSize,
          typename DilationX,
          typename DilationY,
          typename InBlockTransferThreadClusterLengths,
          index_t InBlockTransferSrcScalarPerVector,
          index_t InBlockTransferDstScalarPerVector,
          bool InEnableLds,
          bool InBlockLdsAddExtraM,
          bool InTileLoad,
          typename WeiBlockTransferThreadClusterLengths,
          index_t WeiBlockTransferSrcScalarPerVector,
          index_t WeiBlockTransferDstScalarPerVector,
          bool WeiEnableLds,
          bool WeiBlockLdsAddExtraM,
          bool WeiTileLoad,
          typename DsBlockTransferThreadClusterLengths,
          typename DsBlockTransferSrcScalarPerVector,
          typename DsBlockTransferDstScalarPerVector,
          bool DsEnableLds,
          bool DsBlockLdsAddExtraM,
          typename AccBlockTransferClusterLengths,
          index_t AccBlockTransferScalarPerVector,
          bool AccEnableLds,
          bool EnableAsync,
          index_t NumConvCPrefetchStage,
          bool EnableWaveGroup,
          bool EnableSpatialCluster,
          index_t ClusterDimSize,
          bool Transposed,
          bool TileStore>
struct GridwiseFasternet50_Wcnn_CShuffle
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    static constexpr index_t WaveSize      = 32;
    static constexpr index_t NumWaveGroup  = EnableWaveGroup ? 4 : 0;
    static constexpr bool EnableWaveGroup4 = EnableWaveGroup && (BlockSize == 512);

    using DsDataType_0 = remove_cvref_t<tuple_element_t<0, DsDataType>>;
    using DsDataType_1 = remove_cvref_t<tuple_element_t<1, DsDataType>>;
    using DsDataType_2 = remove_cvref_t<tuple_element_t<2, DsDataType>>;

    using DsLayout_0 = remove_cvref_t<tuple_element_t<0, DsLayout>>;
    using DsLayout_1 = remove_cvref_t<tuple_element_t<1, DsLayout>>;
    using DsLayout_2 = remove_cvref_t<tuple_element_t<2, DsLayout>>;

    static_assert((EnableWaveGroup == false) || (BlockSize % (WaveSize * 4) == 0), "");
    static_assert((EnableWaveGroup == true) || (EnableSpatialCluster == false), "");

    using ThisThreadBlockGrid =
        typename std::conditional<EnableWaveGroup,
                                  ThisThreadBlockWaveGroup<BlockSize, WaveSize, NumWaveGroup>,
                                  ThisThreadBlock<BlockSize>>::type;

    static constexpr std::array<index_t, 3> YX = {FilterSize::At(0) * FilterSize::At(0),
                                                  FilterSize::At(1) * FilterSize::At(1),
                                                  FilterSize::At(2) * FilterSize::At(2)};
    static constexpr index_t NumDTensor_0      = DsDataType_0::Size();
    static constexpr index_t NumDTensor_1      = DsDataType_1::Size();
    static constexpr index_t NumDTensor_2      = DsDataType_2::Size();

    static constexpr auto wcnn_conv_3x3 = WcnnConv<WeiDataType,
                                                   InDataType,
                                                   AccDataType,
                                                   HPerWcnn,
                                                   WPerWcnn,
                                                   3,
                                                   DilationX::At(0),
                                                   DilationY::At(0),
                                                   1,
                                                   EnableWaveGroup,
                                                   false,
                                                   false>{};

    static constexpr auto wcnn_conv_1x1 = WcnnConv<WeiDataType,
                                                   InDataType,
                                                   AccDataType,
                                                   HPerWcnn,
                                                   WPerWcnn,
                                                   1,
                                                   DilationX::At(1),
                                                   DilationY::At(1),
                                                   1,
                                                   EnableWaveGroup,
                                                   false,
                                                   false>{};

    using GridwiseFasternet50Pipe =
        remove_cvref_t<decltype(GridwiseFasternet50Pipeline_Selector<NumConvCPrefetchStage,
                                                                     InEnableLds,
                                                                     WeiEnableLds,
                                                                     DsEnableLds,
                                                                     EnableAsync,
                                                                     EnableWaveGroup,
                                                                     EnableSpatialCluster>())>;

    static constexpr index_t CPerWcnn         = wcnn_conv_3x3.GetNumInputChannels();
    static constexpr index_t KPerWcnn         = wcnn_conv_3x3.GetNumOutputChannels();
    static constexpr index_t NumWeightTap_3x3 = wcnn_conv_3x3.GetNumWeightTap();
    static constexpr index_t NumWeightTap_1x1 = wcnn_conv_1x1.GetNumWeightTap();
    static constexpr index_t NumSubTilesPerWeightTap_3x3 =
        wcnn_conv_3x3.GetNumSubTilesPerWeightTap();
    static constexpr index_t NumSubTilesPerWeightTap_1x1 =
        wcnn_conv_1x1.GetNumSubTilesPerWeightTap();
    static constexpr index_t NumWeightCompPerTile_3x3 = wcnn_conv_3x3.GetNumWeightCompPerTile();
    static constexpr index_t NumWeightCompPerTile_1x1 = wcnn_conv_1x1.GetNumWeightCompPerTile();
    static constexpr index_t NumSubTilePerImage       = wcnn_conv_3x3.GetNumSubTilesPerImageTile();
    static constexpr std::array<index_t, 3> NumDataCompPerTile{
        wcnn_conv_3x3.GetNumDataCompPerTile(),
        wcnn_conv_1x1.GetNumDataCompPerTile(),
        wcnn_conv_1x1.GetNumDataCompPerTile()};

    static constexpr index_t DataTileHeight = 4;

    static constexpr std::array<index_t, 3> H_Pad{DataTileHeight, 0, 0};
    static constexpr std::array<index_t, 3> W_Pad{WPerWcnn, 0, 0};

    static constexpr std::array<index_t, 3> HPerBlockIn{
        HPerBlock + DataTileHeight * 2, HPerBlock, HPerBlock};
    static constexpr std::array<index_t, 3> WPerBlockIn{
        WPerBlock + WPerWcnn * 2, WPerBlock, WPerBlock};

    static constexpr index_t HPerWave = HRepeat * HPerWcnn;
    static constexpr index_t WPerWave = WRepeat * WPerWcnn;
    static constexpr std::array<index_t, 3> CPerWave{
        CPerBlock::At(0), CPerBlock::At(1), CPerBlock::At(2)};
    static constexpr std::array<index_t, 3> KPerWave{
        KPerBlock::At(0), KPerBlock::At(1), KPerBlock::At(2)};
    static constexpr std::array<index_t, 3> HPerWaveIn{
        HPerWave + DataTileHeight * 2, HPerWave, HPerWave};
    static constexpr std::array<index_t, 3> WPerWaveIn{WPerWave + WPerWcnn * 2, WPerWave, WPerWave};

    template <typename DLayout>
    static constexpr bool IsGNHWKLayout()
    {
        return is_same_v<DLayout, tensor_layout::convolution::G_NW_K> ||
               is_same_v<DLayout, tensor_layout::convolution::G_NHW_K> ||
               is_same_v<DLayout, tensor_layout::convolution::G_NDHW_K> ||
               is_same_v<DLayout, tensor_layout::convolution::GNWK> ||
               is_same_v<DLayout, tensor_layout::convolution::GNHWK> ||
               is_same_v<DLayout, tensor_layout::convolution::GNDHWK> ||
               is_same_v<DLayout, tensor_layout::convolution::NWGK> ||
               is_same_v<DLayout, tensor_layout::convolution::NHWGK> ||
               is_same_v<DLayout, tensor_layout::convolution::NDHWGK>;
    }

    template <index_t relu = 0, index_t clamp = 0>
    static auto constexpr AccNextOp()
    {
        if constexpr(is_same_v<AccDataType, EDataType>)
        {
            return BlockwiseElementPassThrough{};
        }
        else
        {
            return BlockwiseElementCvtTensor<true, relu, 0, clamp>{};
        }
    }

    template <index_t conv_phase, typename SingleAccElementwiseOperation>
    static auto __device__ __host__
    MakeAccBlockwiseOp(const SingleAccElementwiseOperation& acc_element_op)
    {
        using AccElementwiseOperation_   = remove_cvref_t<SingleAccElementwiseOperation>;
        auto constexpr AccReluInternalOp = []() {
            if constexpr(is_same_v<AccDataType, EDataType>)
            {
                return tensor_operation::element_wise::Relu{};
            }
            else
            {
                return tensor_operation::element_wise::PassThrough{};
            }
        };

        constexpr index_t NumSingleDTensor = (conv_phase == 0)   ? NumDTensor_0
                                             : (conv_phase == 1) ? NumDTensor_1
                                                                 : NumDTensor_2;
        if constexpr(NumSingleDTensor == 0)
        {
            if constexpr(is_same_v<AccElementwiseOperation_,
                                   tensor_operation::element_wise::PassThrough> ||
                         is_same_v<AccElementwiseOperation_,
                                   tensor_operation::element_wise::UnaryConvert>)
            {

                return make_tuple(BlockwiseElementPassThrough{},
                                  AccNextOp<>(),
                                  tensor_operation::element_wise::PassThrough{});
            }
            else if constexpr(is_same_v<AccElementwiseOperation_,
                                        ck::tensor_operation::element_wise::Scale>)
            {
                return make_tuple(BlockwiseElementSba<0, true, false>{acc_element_op.scale_},
                                  AccNextOp<>(),
                                  tensor_operation::element_wise::PassThrough{});
            }
            else if constexpr(is_same_v<AccElementwiseOperation_,
                                        ck::tensor_operation::element_wise::ScaleClamp>)
            {
                return make_tuple(BlockwiseElementSba<0, true, false>{acc_element_op.scale_},
                                  AccNextOp<false, true>(),
                                  tensor_operation::element_wise::PassThrough{});
            }
            else if constexpr(is_same_v<AccElementwiseOperation_,
                                        ck::tensor_operation::element_wise::ScaleRelu<>>)
            {
                return make_tuple(BlockwiseElementSba<1, true, false>{acc_element_op.scale_},
                                  AccNextOp<>(),
                                  tensor_operation::element_wise::PassThrough{});
            }
            else if constexpr(is_same_v<AccElementwiseOperation_,
                                        ck::tensor_operation::element_wise::ScaleRelu<true>>)
            {
                return make_tuple(BlockwiseElementSba<1, true, false>{acc_element_op.scale_},
                                  AccNextOp<true, true>(),
                                  tensor_operation::element_wise::PassThrough{});
            }
            else if constexpr(is_same_v<AccElementwiseOperation_,
                                        ck::tensor_operation::element_wise::ScaleHardTanh>)
            {
                return make_tuple(BlockwiseElementSba<2, true, false>{acc_element_op.scale_},
                                  AccNextOp<>(),
                                  tensor_operation::element_wise::PassThrough{});
            }
            else
            {
                return make_tuple(
                    BlockwiseElementPassThrough{}, BlockwiseElementPassThrough{}, acc_element_op);
            }
        }
        else if constexpr(NumSingleDTensor == 2)
        {
            using SingleConvDLayout = remove_cvref_t<tuple_element_t<conv_phase, DsLayout>>;
            using D0Layout          = remove_cvref_t<tuple_element_t<0, SingleConvDLayout>>;
            using D1Layout          = remove_cvref_t<tuple_element_t<1, SingleConvDLayout>>;
            if constexpr(is_same_v<D0Layout, tensor_layout::convolution::G_K> &&
                         is_same_v<D1Layout, tensor_layout::convolution::G_K>)
            {
                if constexpr(is_same_v<AccElementwiseOperation_,
                                       ck::tensor_operation::element_wise::MultiplyAdd>)
                {
                    return make_tuple(BlockwiseElementSba<0, false, true>{},
                                      AccNextOp(),
                                      tensor_operation::element_wise::PassThrough{});
                }
                else if constexpr(is_same_v<AccElementwiseOperation_,
                                            ck::tensor_operation::element_wise::MultiplyAddRelu<>>)
                {
                    return make_tuple(BlockwiseElementSba<1, false, true>{},
                                      AccNextOp<true, false>(),
                                      tensor_operation::element_wise::PassThrough{});
                }
                else if constexpr(is_same_v<AccElementwiseOperation_,
                                            ck::tensor_operation::element_wise::MultiplyAddClamp>)
                {
                    return make_tuple(BlockwiseElementSba<0, false, true>{},
                                      AccNextOp<false, true>,
                                      tensor_operation::element_wise::PassThrough{});
                }
                else if constexpr(is_same_v<
                                      AccElementwiseOperation_,
                                      ck::tensor_operation::element_wise::MultiplyAddRelu<true>>)
                {
                    return make_tuple(BlockwiseElementSba<1, false, true>{},
                                      AccNextOp<true, true>(),
                                      tensor_operation::element_wise::PassThrough{});
                }
                else if constexpr(is_same_v<
                                      AccElementwiseOperation_,
                                      ck::tensor_operation::element_wise::MultiplyAddHardTanh>)
                {
                    return make_tuple(BlockwiseElementSba<2, false, true>{},
                                      AccNextOp<>(),
                                      tensor_operation::element_wise::PassThrough{});
                }
                else
                {
                    return make_tuple(BlockwiseElementPassThrough{},
                                      BlockwiseElementPassThrough{},
                                      acc_element_op);
                }
            }
            else if constexpr(IsGNHWKLayout<D0Layout>() && IsGNHWKLayout<D1Layout>())
            {
                if constexpr(is_same_v<AccElementwiseOperation_,
                                       ck::tensor_operation::element_wise::MultiplyAddRev<>>)
                {
                    return make_tuple(BlockwiseElementFma<false>{},
                                      AccNextOp<>(),
                                      tensor_operation::element_wise::PassThrough{});
                }
                else if constexpr(is_same_v<
                                      AccElementwiseOperation_,
                                      ck::tensor_operation::element_wise::MultiplyAddRevRelu<>>)
                {
                    return make_tuple(BlockwiseElementFma<false>{},
                                      AccNextOp<true, false>(),
                                      AccReluInternalOp());
                }
                else if constexpr(is_same_v<
                                      AccElementwiseOperation_,
                                      ck::tensor_operation::element_wise::MultiplyAddRev<true>>)
                {
                    return make_tuple(BlockwiseElementFma<false>{},
                                      AccNextOp<false, true>(),
                                      tensor_operation::element_wise::PassThrough{});
                }
                else if constexpr(is_same_v<
                                      AccElementwiseOperation_,
                                      ck::tensor_operation::element_wise::MultiplyAddRevRelu<true>>)
                {
                    return make_tuple(
                        BlockwiseElementFma<false>{}, AccNextOp<true, true>(), AccReluInternalOp());
                }
                else
                {
                    return make_tuple(BlockwiseElementPassThrough{},
                                      BlockwiseElementPassThrough{},
                                      acc_element_op);
                }
            }
            else if constexpr(is_same_v<D0Layout, tensor_layout::convolution::G_K> &&
                              IsGNHWKLayout<D1Layout>())
            {
                if constexpr(is_same_v<AccElementwiseOperation_,
                                       ck::tensor_operation::element_wise::AddReluAdd>)
                {
                    return make_tuple(BlockwiseElementSbaFma<1, false, false, true>{},
                                      AccNextOp<>(),
                                      tensor_operation::element_wise::PassThrough{});
                }
                else
                {
                    return make_tuple(BlockwiseElementPassThrough{},
                                      BlockwiseElementPassThrough{},
                                      acc_element_op);
                }
            }
            else
            {
                return make_tuple(
                    BlockwiseElementPassThrough{}, BlockwiseElementPassThrough{}, acc_element_op);
            }
        }
        else if constexpr(NumSingleDTensor == 1)
        {
            using SingleConvDLayout = remove_cvref_t<tuple_element_t<conv_phase, DsLayout>>;
            using D0Layout          = remove_cvref_t<tuple_element_t<0, SingleConvDLayout>>;
            if constexpr(is_same_v<D0Layout, tensor_layout::convolution::G_K>)
            {
                if constexpr(is_same_v<AccElementwiseOperation_,
                                       ck::tensor_operation::element_wise::ScaleAdd>)
                {
                    return make_tuple(BlockwiseElementSba<0, false, false>{acc_element_op.scale_},
                                      AccNextOp<>(),
                                      tensor_operation::element_wise::PassThrough{});
                }
                else if constexpr(is_same_v<AccElementwiseOperation_,
                                            ck::tensor_operation::element_wise::ScaleAddRelu<>>)
                {
                    return make_tuple(BlockwiseElementSba<1, false, false>{acc_element_op.scale_},
                                      AccNextOp<true, false>(),
                                      tensor_operation::element_wise::PassThrough{});
                }
                else if constexpr(is_same_v<AccElementwiseOperation_,
                                            ck::tensor_operation::element_wise::ScaleAddClamp>)
                {
                    return make_tuple(BlockwiseElementSba<0, false, false>{acc_element_op.scale_},
                                      AccNextOp<false, true>(),
                                      tensor_operation::element_wise::PassThrough{});
                }
                else if constexpr(is_same_v<AccElementwiseOperation_,
                                            ck::tensor_operation::element_wise::ScaleAddRelu<true>>)
                {
                    return make_tuple(BlockwiseElementSba<1, false, false>{acc_element_op.scale_},
                                      AccNextOp<true, true>(),
                                      tensor_operation::element_wise::PassThrough{});
                }
                else if constexpr(is_same_v<AccElementwiseOperation_,
                                            ck::tensor_operation::element_wise::ScaleAddHardTanh>)
                {
                    return make_tuple(BlockwiseElementSba<2, false, false>{acc_element_op.scale_},
                                      AccNextOp<>(),
                                      tensor_operation::element_wise::PassThrough{});
                }
                else
                {
                    return make_tuple(BlockwiseElementPassThrough{},
                                      BlockwiseElementPassThrough{},
                                      acc_element_op);
                }
            }
            else if constexpr(IsGNHWKLayout<D0Layout>())
            {
                if constexpr(is_same_v<AccElementwiseOperation_,
                                       ck::tensor_operation::element_wise::ScaleAddRev<>>)
                {
                    return make_tuple(BlockwiseElementFma<true>{acc_element_op.scale_},
                                      AccNextOp<>(),
                                      tensor_operation::element_wise::PassThrough{});
                }
                else if constexpr(is_same_v<AccElementwiseOperation_,
                                            ck::tensor_operation::element_wise::ScaleAddRevRelu<>>)
                {
                    return make_tuple(BlockwiseElementFma<true>{acc_element_op.scale_},
                                      AccNextOp<true, false>(),
                                      AccReluInternalOp());
                }
                else if constexpr(is_same_v<AccElementwiseOperation_,
                                            ck::tensor_operation::element_wise::ScaleAddRev<true>>)
                {
                    return make_tuple(BlockwiseElementFma<true>{acc_element_op.scale_},
                                      AccNextOp<false, true>(),
                                      tensor_operation::element_wise::PassThrough{});
                }
                else if constexpr(is_same_v<
                                      AccElementwiseOperation_,
                                      ck::tensor_operation::element_wise::ScaleAddRevRelu<true>>)
                {
                    return make_tuple(BlockwiseElementFma<true>{acc_element_op.scale_},
                                      AccNextOp<true, true>(),
                                      AccReluInternalOp());
                }
                else
                {
                    return make_tuple(BlockwiseElementPassThrough{},
                                      BlockwiseElementPassThrough{},
                                      acc_element_op);
                }
            }
            else
            {
                return make_tuple(
                    BlockwiseElementPassThrough{}, BlockwiseElementPassThrough{}, acc_element_op);
            }
        }
        else
        {
            return make_tuple(
                BlockwiseElementPassThrough{}, BlockwiseElementPassThrough{}, acc_element_op);
        }
    }
    using AccBlockwiseOp_0        = decltype(MakeAccBlockwiseOp<0>(AccElementwiseOperation{}[I0]));
    using AccBlockwiseOp_1        = decltype(MakeAccBlockwiseOp<1>(AccElementwiseOperation{}[I1]));
    using AccBlockwiseOp_2        = decltype(MakeAccBlockwiseOp<2>(AccElementwiseOperation{}[I2]));
    using AccBlockwiseOperation_0 = remove_cvref_t<tuple_element_t<0, AccBlockwiseOp_0>>;
    using AccBlockwiseOperation_1 = remove_cvref_t<tuple_element_t<0, AccBlockwiseOp_1>>;
    using AccBlockwiseOperation_2 = remove_cvref_t<tuple_element_t<0, AccBlockwiseOp_2>>;
    using AccBlockwiseNextOperation_0 = remove_cvref_t<tuple_element_t<1, AccBlockwiseOp_0>>;
    using AccBlockwiseNextOperation_1 = remove_cvref_t<tuple_element_t<1, AccBlockwiseOp_1>>;
    using AccBlockwiseNextOperation_2 = remove_cvref_t<tuple_element_t<1, AccBlockwiseOp_2>>;

    template <index_t conv_phase>
    __host__ __device__ static constexpr bool IsPassthroughBlockwiseOp()
    {
        if constexpr(conv_phase == 0)
        {
            return AccBlockwiseOperation_0::IsFma == false &&
                   AccBlockwiseOperation_0::IsSuba == false;
        }
        else if constexpr(conv_phase == 1)
        {
            return AccBlockwiseOperation_1::IsFma == false &&
                   AccBlockwiseOperation_1::IsSuba == false;
        }
        else if constexpr(conv_phase == 2)
        {
            return AccBlockwiseOperation_2::IsFma == false &&
                   AccBlockwiseOperation_2::IsSuba == false;
        }
        else
        {
            static_assert(0, "unreachable branch!");
        }
    }

    static constexpr bool AccEnableLdsInternal = AccEnableLds;
    // Describe the layout of InData in block level (LDS or VGPR)

    template <index_t conv_phase>
    __host__ __device__ static constexpr auto MakeInBlockDescriptor()
    {
        if constexpr(InEnableLds)
        {
            // H x W x C Per Block
            return make_naive_tensor_descriptor_packed(make_tuple(Number<HPerBlockIn[conv_phase]>{},
                                                                  Number<WPerBlockIn[conv_phase]>{},
                                                                  Number<CPerBlock::At(0)>{}));
        }
        else
        {
            // W0 x C0 x H0 x H1 x H2 x W1 x C1
            return make_naive_tensor_descriptor_packed(
                make_tuple(Number<WPerWaveIn[conv_phase] / WPerWcnn>{},
                           Number<CPerWave[conv_phase] / CPerWcnn>{},
                           Number<HPerWaveIn[conv_phase] / HPerWcnn>{},
                           Number<NumSubTilePerImage>{},
                           I1,
                           I1,
                           Number<NumDataCompPerTile[conv_phase]>{}));
        }
    }

    template <bool isClusterBorder, index_t conv_phase>
    __host__ __device__ static constexpr auto MakeInClusterBorderBlockDescriptor()
    {
        if constexpr(isClusterBorder)
        {
            // W0 x C0 x H0 x H1 x H2 x W1 x C1
            return make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<CPerWave[conv_phase] / CPerWcnn>{},
                           Number<HPerWaveIn[conv_phase] / HPerWcnn>{},
                           Number<NumSubTilePerImage>{},
                           I1,
                           I1,
                           Number<NumDataCompPerTile[conv_phase]>{}));
        }
        else
        {
            return make_naive_tensor_descriptor_packed(
                make_tuple(Number<CPerWave[conv_phase] / CPerWcnn>{},
                           Number<HPerWaveIn[conv_phase] / HPerWcnn>{},
                           Number<NumSubTilePerImage>{}));
        }
    }

    // Describe the layout of WeiData in block level (LDS or VGPR)
    template <index_t index>
    __host__ __device__ static constexpr auto MakeSingleWeiBlockDescriptor()
    {
        if constexpr(WeiEnableLds)
        {
            // K x YX x C per block
            return make_naive_tensor_descriptor_packed(
                make_tuple(Number<KPerBlock::At(index)>{},
                           Number<FilterSize::At(index) * FilterSize::At(index)>{},
                           Number<CPerBlock::At(index)>{}));
        }
        else
        {
            // K0 x C0 x YX x K1 x C1 x C2
            constexpr index_t weightTap =
                (index == 0) ? NumSubTilesPerWeightTap_3x3 : NumSubTilesPerWeightTap_1x1;
            constexpr index_t weightPerTile =
                (index == 0) ? NumWeightCompPerTile_3x3 : NumWeightCompPerTile_1x1;
            constexpr index_t NumXY =
                (index == 0) ? NumWeightTap_3x3 : FilterSize::At(index) * FilterSize::At(index);
            return make_naive_tensor_descriptor_packed(
                make_tuple(Number<KPerWave[index] / KPerWcnn>{},
                           Number<CPerWave[index] / CPerWcnn>{},
                           Number<NumXY>{},
                           I1,
                           Number<weightTap>{},
                           Number<weightPerTile>{}));
        }
    }

    template <typename DLayout, index_t conv_phase>
    __host__ __device__ static constexpr auto MakeSingleDBlockDescriptor()
    {
        // TODO: need a rule for per-wave layout when lds isn't enabled
        if constexpr(std::is_same_v<DLayout, tensor_layout::convolution::G_K>)
        {
            return make_naive_tensor_descriptor_packed(
                make_tuple(Number<KPerBlock::At(conv_phase)>{}));
        }
        else
        {
            // H x W x K Per Block
            return make_naive_tensor_descriptor_packed(make_tuple(
                Number<HPerBlock>{}, Number<WPerBlock>{}, Number<KPerBlock::At(conv_phase)>{}));
        }
    }

    // Ds desc for source in blockwise copy
    template <index_t conv_phase>
    __host__ __device__ static constexpr auto MakeDsBlockDescriptor()
    {
        if constexpr(IsPassthroughBlockwiseOp<conv_phase>())
        {
            return Tuple<>{};
        }
        else
        {
            constexpr index_t NumSingleDTensor = (conv_phase == 0)   ? NumDTensor_0
                                                 : (conv_phase == 1) ? NumDTensor_1
                                                                     : NumDTensor_2;
            using SingleConvDsLayout = remove_cvref_t<tuple_element_t<conv_phase, DsLayout>>;
            return generate_tuple(
                [&](auto i) {
                    using DLayout = remove_cvref_t<tuple_element_t<i.value, SingleConvDsLayout>>;
                    return MakeSingleDBlockDescriptor<DLayout, conv_phase>();
                },
                Number<NumSingleDTensor>{});
        }
    }

    static constexpr auto ds_0_block_desc = MakeDsBlockDescriptor<0>();
    static constexpr auto ds_1_block_desc = MakeDsBlockDescriptor<1>();
    static constexpr auto ds_2_block_desc = MakeDsBlockDescriptor<2>();

    using BlockOut_0_DataType = typename std::
        conditional<AccBlockwiseNextOperation_0::cvt_to_tensor, EDataType, AccDataType>::type;
    using BlockOut_1_DataType = typename std::
        conditional<AccBlockwiseNextOperation_1::cvt_to_tensor, EDataType, AccDataType>::type;
    using BlockOut_2_DataType = typename std::
        conditional<AccBlockwiseNextOperation_2::cvt_to_tensor, EDataType, AccDataType>::type;

    using BlockwiseConv_0 =
        BlockwiseFasternet50Wcnn<ThisThreadBlockGrid,
                                 WeiDataType,
                                 InDataType,
                                 DsDataType_0,
                                 AccDataType,
                                 BlockOut_0_DataType,
                                 AccBlockwiseOperation_0,
                                 AccBlockwiseNextOperation_0,
                                 decltype(MakeSingleWeiBlockDescriptor<0>()),
                                 decltype(MakeInBlockDescriptor<0>()),
                                 decltype(MakeInClusterBorderBlockDescriptor<false, 0>()),
                                 decltype(MakeDsBlockDescriptor<0>()),
                                 HPerBlock,
                                 WPerBlock,
                                 CPerBlock::At(0),
                                 KPerBlock::At(0),
                                 HRepeat,
                                 WRepeat,
                                 HPerWcnn,
                                 WPerWcnn,
                                 FilterSize::At(0),
                                 DilationX::At(0),
                                 DilationY::At(0),
                                 WeiEnableLds,
                                 WeiTileLoad,
                                 InEnableLds,
                                 InTileLoad,
                                 DsEnableLds,
                                 Transposed,
                                 TileStore,
                                 EnableSpatialCluster>;

    using BlockwiseConv_1 =
        BlockwiseFasternet50Wcnn<ThisThreadBlockGrid,
                                 WeiDataType,
                                 InDataType,
                                 DsDataType_1,
                                 AccDataType,
                                 BlockOut_1_DataType,
                                 AccBlockwiseOperation_1,
                                 AccBlockwiseNextOperation_1,
                                 decltype(MakeSingleWeiBlockDescriptor<1>()),
                                 decltype(MakeInBlockDescriptor<1>()),
                                 decltype(MakeInClusterBorderBlockDescriptor<false, 1>()),
                                 decltype(MakeDsBlockDescriptor<1>()),
                                 HPerBlock,
                                 WPerBlock,
                                 CPerBlock::At(1),
                                 KPerBlock::At(1),
                                 HRepeat,
                                 WRepeat,
                                 HPerWcnn,
                                 WPerWcnn,
                                 FilterSize::At(1),
                                 DilationX::At(1),
                                 DilationY::At(1),
                                 WeiEnableLds,
                                 WeiTileLoad,
                                 InEnableLds,
                                 InTileLoad,
                                 DsEnableLds,
                                 Transposed,
                                 TileStore,
                                 EnableSpatialCluster>;

    using BlockwiseConv_2 =
        BlockwiseFasternet50Wcnn<ThisThreadBlockGrid,
                                 WeiDataType,
                                 InDataType,
                                 DsDataType_2,
                                 AccDataType,
                                 BlockOut_2_DataType,
                                 AccBlockwiseOperation_2,
                                 AccBlockwiseNextOperation_2,
                                 decltype(MakeSingleWeiBlockDescriptor<2>()),
                                 decltype(MakeInBlockDescriptor<2>()),
                                 decltype(MakeInClusterBorderBlockDescriptor<false, 2>()),
                                 decltype(MakeDsBlockDescriptor<2>()),
                                 HPerBlock,
                                 WPerBlock,
                                 CPerBlock::At(2),
                                 KPerBlock::At(2),
                                 HRepeat,
                                 WRepeat,
                                 HPerWcnn,
                                 WPerWcnn,
                                 FilterSize::At(2),
                                 DilationX::At(2),
                                 DilationY::At(2),
                                 WeiEnableLds,
                                 WeiTileLoad,
                                 InEnableLds,
                                 InTileLoad,
                                 DsEnableLds,
                                 Transposed,
                                 TileStore,
                                 EnableSpatialCluster>;

    // Pad input and weight data grid description according to Filter size
    __host__ __device__ static constexpr auto
    MakeInGridPadDescriptor(const InGridDesc& in_grid_desc)
    {
        return generate_tuple(
            [&](auto i) {
                if constexpr(H_Pad[i] > 0 || W_Pad[i] > 0)
                {
                    const auto Hi = in_grid_desc[Number<i>{}].GetLength(I0);
                    const auto Wi = in_grid_desc[Number<i>{}].GetLength(I1);
                    const auto Ci = in_grid_desc[Number<i>{}].GetLength(I2);

                    return transform_tensor_descriptor(
                        in_grid_desc[Number<i>{}],
                        make_tuple(make_pad_transform(Hi, H_Pad[i], H_Pad[i]),
                                   make_pad_transform(Wi, W_Pad[i], W_Pad[i]),
                                   make_pass_through_transform(Ci)),
                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
                }
                else
                {
                    return in_grid_desc[Number<i>{}];
                }
            },
            Number<3>{});
    }

    static constexpr auto MakeWeiGridPointer()
    {
        return generate_tuple(
            [&](auto) {
                using WeiType = remove_cvref_t<WeiDataType>;

                return static_cast<const WeiType*>(nullptr);
            },
            Number<3>{});
    }

    using WeiGridPointer = decltype(MakeWeiGridPointer());

    // ck::Tuple<const D0DataType*, const D1DataType*, ...>
    // ck::Tuple<const D0DataType*, const D1DataType*, ...>
    static constexpr auto MakeDsGridPointer()
    {
        const auto ds_grid_0 = generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType_0>>;

                return static_cast<const DDataType*>(nullptr);
            },
            Number<NumDTensor_0>{});

        const auto ds_grid_1 = generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType_1>>;

                return static_cast<const DDataType*>(nullptr);
            },
            Number<NumDTensor_1>{});

        const auto ds_grid_2 = generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType_2>>;

                return static_cast<const DDataType*>(nullptr);
            },
            Number<NumDTensor_2>{});
        return make_tuple(ds_grid_0, ds_grid_1, ds_grid_2);
    }

    using DsGridPointer    = remove_cvref_t<decltype(MakeDsGridPointer())>;
    using BlockOutDataType = typename std::
        conditional<AccBlockwiseNextOperation_2::cvt_to_tensor, EDataType, AccDataType>::type;

    template <index_t conv_phase,
              typename OutTensorThreadBuffer,
              typename NamedBarrier,
              typename BlockWiseConv,
              typename AccElementwiseOperationInternal>
    __host__ __device__ static void
    StoreOutTensorData(const DsGridDesc& ds_grid_desc,
                       DsGridPointer p_ds_grid,
                       const EGridDesc& e_grid_desc,
                       EDataType* __restrict__ p_e_grid,
                       OutTensorThreadBuffer& out_thread_buf,
                       BlockWiseConv& blockwise_conv,
                       const AccElementwiseOperationInternal& out_element_op,
                       void* __restrict__ p_shared,
                       NamedBarrier& barrier_output,
                       index_t h_block_data_idx_on_grid,
                       index_t w_block_data_idx_on_grid,
                       index_t k_block_data_idx_on_grid)
    {
        auto out_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_e_grid, e_grid_desc.GetElementSpaceSize());
        // C mapping in single thread.
        constexpr auto out_tensor_thread_desc   = BlockWiseConv::GetAccThreadDescriptor();
        constexpr auto out_tensor_thread_length = BlockWiseConv::GetAccThreadDescLength();
        constexpr bool ForceAlignToUint32 =
            EnableWaveGroup4 && (sizeof(BlockOutDataType) < sizeof(uint32_t));
        static constexpr auto EGlobalMemoryDataOperation = InMemoryDataOperationEnum::Set;

        // calculate origin of thread output tensor on global memory
        // blockwise conv out tensor starting index
        const auto out_thread_mtx_on_block = blockwise_conv.CalculateAccThreadOriginDataIndex();

        if constexpr(AccEnableLdsInternal == false)
        {
            const auto out_tensor_grid_wave_desc = blockwise_conv.GetAccWaveDescriptor(e_grid_desc);

            constexpr index_t threadsPerTensorTile = (WPerWcnn == 4) ? 2 : 4;
            constexpr index_t vgprPerTensorTile    = (HPerWcnn == 8) ? 2 : 1;
            // Threadwise copy C from VGPR to global memory
            if constexpr(TileStore)
            {
                auto out_tensor_thread_copy_vgpr_to_global =
                    ThreadwiseTensorSliceTransfer_v1r3<BlockOutDataType,
                                                       EDataType,
                                                       decltype(out_tensor_thread_desc),
                                                       decltype(out_tensor_grid_wave_desc),
                                                       AccElementwiseOperationInternal,
                                                       decltype(out_tensor_thread_length),
                                                       Sequence<0, 1, 2, 3, 4, 5, 6>,
                                                       6,
                                                       out_tensor_thread_length[I6], // vector write
                                                                                     // pixel
                                                       EGlobalMemoryDataOperation,
                                                       1,
                                                       true,
                                                       ForceAlignToUint32,
                                                       true, // TileStore,
                                                       threadsPerTensorTile,
                                                       vgprPerTensorTile>{
                        out_tensor_grid_wave_desc,
                        out_thread_mtx_on_block +
                            make_multi_index(h_block_data_idx_on_grid / HPerWcnn,
                                             w_block_data_idx_on_grid / WPerWcnn,
                                             k_block_data_idx_on_grid / KPerWcnn,
                                             I0,
                                             I0,
                                             I0,
                                             I0),
                        out_element_op};

                // each thread write its data from VGPR to global
                out_tensor_thread_copy_vgpr_to_global.Run(out_tensor_thread_desc,
                                                          make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                                          out_thread_buf,
                                                          out_tensor_grid_wave_desc,
                                                          out_grid_buf);
            }
            else
            {
                auto out_tensor_thread_copy_vgpr_to_global =
                    ThreadwiseTensorSliceTransfer_v1r3<BlockOutDataType,
                                                       EDataType,
                                                       decltype(out_tensor_thread_desc),
                                                       decltype(out_tensor_grid_wave_desc),
                                                       AccElementwiseOperationInternal,
                                                       decltype(out_tensor_thread_length),
                                                       Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                       7,
                                                       out_tensor_thread_length[I7], // vector write
                                                                                     // pixel
                                                       EGlobalMemoryDataOperation,
                                                       1,
                                                       true,
                                                       ForceAlignToUint32>{
                        out_tensor_grid_wave_desc,
                        out_thread_mtx_on_block +
                            make_multi_index(h_block_data_idx_on_grid / HPerWcnn,
                                             w_block_data_idx_on_grid / WPerWcnn,
                                             k_block_data_idx_on_grid / KPerWcnn,
                                             I0,
                                             I0,
                                             I0,
                                             I0,
                                             I0),
                        out_element_op};

                // each thread write its data from VGPR to global
                out_tensor_thread_copy_vgpr_to_global.Run(
                    out_tensor_thread_desc,
                    make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                    out_thread_buf,
                    out_tensor_grid_wave_desc,
                    out_grid_buf);
            }
        }
        else
        {
            // C mapping in single block
            // LDS descriptor, shuffle and write out in HRepeat x WRepeat x KRepeat times
            constexpr auto out_tensor_block_desc = BlockWiseConv::GetAccBlockDescriptor();
            constexpr auto out_tensor_block_wave_desc =
                BlockWiseConv::GetAccWaveDescriptor(out_tensor_block_desc);

            auto out_tensor_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<BlockOutDataType*>(p_shared) +
                    BlockWiseConv::SharedMemTrait::out_tensor_block_space_offset,
                BlockWiseConv::SharedMemTrait::out_tensor_block_space_size);

            // Threadwise copy C from VGPR to LDS
            auto out_tensor_thread_copy_vgpr_to_lds = ThreadwiseTensorSliceTransfer_v1r3<
                BlockOutDataType,
                BlockOutDataType,
                decltype(out_tensor_thread_desc),
                decltype(out_tensor_block_wave_desc),
                ck::tensor_operation::element_wise::PassThrough,
                decltype(out_tensor_thread_length),
                Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                7,
                out_tensor_thread_length[I7], // vector write pixel
                InMemoryDataOperationEnum::Set,
                1,
                true,
                ForceAlignToUint32>{out_tensor_block_wave_desc,
                                    out_thread_mtx_on_block,
                                    ck::tensor_operation::element_wise::PassThrough{}};

            constexpr index_t NumSingleDTensor = (conv_phase == 0)   ? NumDTensor_0
                                                 : (conv_phase == 1) ? NumDTensor_1
                                                                     : NumDTensor_2;
            constexpr index_t NumOutDTensor =
                IsPassthroughBlockwiseOp<conv_phase>() ? NumSingleDTensor : 0;

            // each block copy its data from LDS to global
            if constexpr(NumOutDTensor == 0)
            {
                // blockwise copy C from LDS to global
                auto out_tensor_block_copy_lds_to_global =
                    ThreadGroupTensorSliceTransfer_v6r1<ThisThreadBlockGrid,
                                                        AccElementwiseOperationInternal,
                                                        EGlobalMemoryDataOperation,
                                                        Sequence<BlockWiseConv::HPerBlockOut,
                                                                 BlockWiseConv::WPerBlockOut,
                                                                 KPerBlock::At(conv_phase)>,
                                                        AccBlockTransferClusterLengths,
                                                        Sequence<0, 1, 2>,
                                                        BlockOutDataType,
                                                        EDataType,
                                                        decltype(out_tensor_block_desc),
                                                        decltype(e_grid_desc),
                                                        Sequence<0, 1, 2>,
                                                        2,
                                                        AccBlockTransferScalarPerVector,
                                                        true,
                                                        false>{
                        out_tensor_block_desc,
                        make_multi_index(0, 0, 0),
                        e_grid_desc,
                        make_multi_index(h_block_data_idx_on_grid,
                                         w_block_data_idx_on_grid,
                                         k_block_data_idx_on_grid),
                        out_element_op};

                // make sure it's safe to write to LDS
                if constexpr(EnableWaveGroup == false)
                {
                    block_sync_lds();
                }
                else
                {
                    barrier_output.signal();
                    barrier_output.wait();
                }

                // each thread write its data from VGPR to LDS
                out_tensor_thread_copy_vgpr_to_lds.Run(out_tensor_thread_desc,
                                                       make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                                       out_thread_buf,
                                                       out_tensor_block_wave_desc,
                                                       out_tensor_block_buf);

                // make sure it's safe to read from LDS
                if constexpr(EnableWaveGroup == false)
                {
                    block_sync_lds();
                }
                else
                {
                    barrier_output.template sync_lds<false>();
                }

                out_tensor_block_copy_lds_to_global.Run(
                    out_tensor_block_desc, out_tensor_block_buf, e_grid_desc, out_grid_buf);
            }
            else
            {
                const auto ds_grid_buf = generate_tuple(
                    [&](auto i) {
                        return make_dynamic_buffer<AddressSpaceEnum::Global>(
                            p_ds_grid[Number<conv_phase>{}][i],
                            ds_grid_desc[Number<conv_phase>{}][i].GetElementSpaceSize());
                    },
                    Number<NumSingleDTensor>{});

                // tuple of reference to C/Ds tensor descriptors
                const auto c_ds_desc_refs = concat_tuple_of_reference(
                    tie(out_tensor_block_desc),
                    generate_tie([&](auto i) -> const auto& // return type should be reference
                                 { return ds_grid_desc[Number<conv_phase>{}][i]; },
                                 Number<NumSingleDTensor>{}));

                // tuple of reference to C/Ds tensor buffers
                const auto c_ds_buf_refs = concat_tuple_of_reference(
                    tie(out_tensor_block_buf),
                    generate_tie([&](auto i) -> const auto& // return type should be reference
                                 { return ds_grid_buf[Number<conv_phase>{}][i]; },
                                 Number<NumSingleDTensor>{}));

                // tuple of starting index of C/Ds blockwise copy
                const auto idx_c_ds_block_begin =
                    container_concat(make_tuple(make_multi_index(0, 0, 0)),
                                     generate_tuple(
                                         [&](auto) {
                                             return make_multi_index(h_block_data_idx_on_grid,
                                                                     w_block_data_idx_on_grid,
                                                                     k_block_data_idx_on_grid);
                                         },
                                         Number<NumSingleDTensor>{}));

                using DsSingleConvDataType = std::conditional_t<
                    conv_phase == 0,
                    DsDataType_0,
                    std::conditional_t<conv_phase == 1, DsDataType_1, DsDataType_2>>;
                auto out_tensor_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v7<
                    ThisThreadBlockGrid, // ThreadGroup
                    decltype(container_concat(make_tuple(BlockOutDataType{}),
                                              DsSingleConvDataType{})), // SrcDatas
                    Tuple<EDataType>,                                   // DstDatas
                    decltype(c_ds_desc_refs),                           // SrcDescs
                    decltype(tie(e_grid_desc)),                         // DstDescs
                    AccElementwiseOperationInternal,                    // ElementwiseOperation,
                    Sequence<static_cast<index_t>(EGlobalMemoryDataOperation)>, // DstInMemOp,
                    Sequence<BlockWiseConv::HPerBlockOut,
                             BlockWiseConv::WPerBlockOut,
                             KPerBlock::At(conv_phase)>, // SliceLengths
                    AccBlockTransferClusterLengths,      // ThreadClusterLengths
                    Sequence<0, 1, 2>,                   // ThreadClusterArrangeOrder
                    Sequence<0, 1, 2>,                   // DimAccessOrder
                    2,                                   // VectorDim
                    AccBlockTransferScalarPerVector,     // ScalarPerVector
                    sequence_merge_t<Sequence<true>,
                                     uniform_sequence_gen_t<
                                         NumSingleDTensor,
                                         false>>, // bool ThreadTransferSrcResetCoordinateAfterRun,
                    Sequence<false>>              // bool ThreadTransferDstResetCoordinateAfterRun>
                    {c_ds_desc_refs,
                     idx_c_ds_block_begin,
                     tie(e_grid_desc),
                     make_tuple(make_multi_index(h_block_data_idx_on_grid,
                                                 w_block_data_idx_on_grid,
                                                 k_block_data_idx_on_grid)),
                     out_element_op};

                // make sure it's safe to write to LDS
                if constexpr(EnableWaveGroup == false)
                {
                    block_sync_lds();
                }
                else
                {
                    barrier_output.signal();
                    barrier_output.wait();
                }

                // each thread write its data from VGPR to LDS
                out_tensor_thread_copy_vgpr_to_lds.Run(out_tensor_thread_desc,
                                                       make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                                       out_thread_buf,
                                                       out_tensor_block_wave_desc,
                                                       out_tensor_block_buf);

                // make sure it's safe to read from LDS
                if constexpr(EnableWaveGroup == false)
                {
                    block_sync_lds();
                }
                else
                {
                    barrier_output.template sync_lds<false>();
                }

                out_tensor_block_copy_lds_to_global.Run(
                    c_ds_desc_refs, c_ds_buf_refs, tie(e_grid_desc), tie(out_grid_buf));
            }
        }
    }

    template <index_t conv_phase,
              typename DLayout,
              typename DDataType,
              typename DGridBlockDesc,
              typename DBlockDesc,
              typename DThreadCoord,
              typename DWaveDescLength>
    __host__ __device__ static constexpr auto
    MakeSignleDThreadwiseTransfer(const DLayout&,
                                  const DDataType&,
                                  const DGridBlockDesc& ds_grid_block_desc,
                                  const DBlockDesc&,
                                  const DThreadCoord& thread_origin_coord,
                                  const DWaveDescLength&,
                                  index_t h_block_data_idx_on_grid,
                                  index_t w_block_data_idx_on_grid,
                                  index_t k_block_data_idx_on_grid)
    {
        auto wave_idx = GetWcnnWaveIdx<ThisThreadBlockGrid,
                                       HPerBlock,
                                       WPerBlock,
                                       HRepeat,
                                       WRepeat,
                                       HPerWcnn,
                                       WPerWcnn>();

        constexpr auto CoordDim = DThreadCoord::Size();

        static_assert(BlockwiseConv_0::HPerWaveOut == BlockwiseConv_2::HPerWaveOut,
                      "H should be same between input and output!");
        static_assert(BlockwiseConv_0::HPerWaveOut == BlockwiseConv_1::HPerWaveOut,
                      "H should be same between input and output!");
        static_assert(BlockwiseConv_0::WPerWaveOut == BlockwiseConv_2::WPerWaveOut,
                      "W should be same between input and output!");
        static_assert(BlockwiseConv_0::WPerWaveOut == BlockwiseConv_1::WPerWaveOut,
                      "W should be same between input and output!");

        auto h0 =
            (h_block_data_idx_on_grid + wave_idx[I0] * BlockwiseConv_0::HPerWaveOut) / HPerWcnn;
        auto w0 =
            (w_block_data_idx_on_grid + wave_idx[I1] * BlockwiseConv_0::WPerWaveOut) / WPerWcnn;
        auto k0 = (k_block_data_idx_on_grid + wave_idx[I2] * KPerWave[conv_phase]) / KPerWcnn;
        // residual use input tensor layout
        auto c0 = (k_block_data_idx_on_grid + wave_idx[I2] * KPerWave[conv_phase]) / CPerWcnn;

        if constexpr(std::is_same_v<DLayout, tensor_layout::convolution::G_K>)
        {
            constexpr auto NumComp = DWaveDescLength{}[I2];
            constexpr bool ForceAlignToUint32 =
                EnableWaveGroup && (sizeof(DDataType) * NumComp < sizeof(uint32_t));
            return ThreadwiseTensorSliceTransfer_v2<DDataType,
                                                    DDataType,
                                                    DGridBlockDesc,
                                                    DBlockDesc,
                                                    DWaveDescLength,
                                                    Sequence<0, 1, 2>,
                                                    2,
                                                    NumComp,
                                                    1,
                                                    false,
                                                    false,
                                                    false,
                                                    ForceAlignToUint32>(
                ds_grid_block_desc,
                make_multi_index(k0,
                                 thread_origin_coord[Number<CoordDim - 2>{}],
                                 thread_origin_coord[Number<CoordDim - 1>{}]));
        }
        else
        {
            // Thread-wise copy
            if constexpr(CoordDim == 7)
            {
                // W0 x C0 x H0 x H1 x H2 x W1 x C1
                constexpr auto NumComp = DWaveDescLength{}[I6];
                static_assert(NumComp == NumDataCompPerTile[conv_phase]);
                return ThreadwiseTensorSliceTransfer_v2<DDataType,
                                                        DDataType,
                                                        DGridBlockDesc,
                                                        DBlockDesc,
                                                        DWaveDescLength,
                                                        Sequence<0, 1, 2, 3, 4, 5, 6>,
                                                        6,
                                                        NumComp,
                                                        1,
                                                        false>(
                    ds_grid_block_desc,
                    make_multi_index(w0,
                                     c0,
                                     h0,
                                     0,
                                     thread_origin_coord[Number<CoordDim - 3>{}],
                                     thread_origin_coord[Number<CoordDim - 2>{}],
                                     thread_origin_coord[Number<CoordDim - 1>{}]));
            }
            else
            {
                // H0 x W0 x K0 x H1 x H2 x W1 x K1 x K2
                static_assert(CoordDim == 8);
                constexpr auto NumComp = DWaveDescLength{}[I7];
                return ThreadwiseTensorSliceTransfer_v2<DDataType,
                                                        DDataType,
                                                        DGridBlockDesc,
                                                        DBlockDesc,
                                                        DWaveDescLength,
                                                        Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                        7,
                                                        NumComp,
                                                        1,
                                                        false>(
                    ds_grid_block_desc,
                    make_multi_index(h0,
                                     w0,
                                     k0,
                                     thread_origin_coord[Number<CoordDim - 5>{}],
                                     thread_origin_coord[Number<CoordDim - 4>{}],
                                     thread_origin_coord[Number<CoordDim - 3>{}],
                                     thread_origin_coord[Number<CoordDim - 2>{}],
                                     thread_origin_coord[Number<CoordDim - 1>{}]));
            }
        }
    }

    template <index_t conv_phase, typename DIndex, typename DGridBlockDesc, typename DBlockDesc>
    __host__ __device__ static constexpr auto
    MakeSignleDThreadgroupTransfer(const DIndex&,
                                   const DGridBlockDesc& d_grid_block_desc,
                                   const DBlockDesc&,
                                   index_t h_block_data_idx_on_grid,
                                   index_t w_block_data_idx_on_grid,
                                   index_t k_block_data_idx_on_grid)
    {
        using DSingleConvLayout = remove_cvref_t<tuple_element_t<conv_phase, DsLayout>>;
        using DLayout         = remove_cvref_t<tuple_element_t<DIndex{}.value, DSingleConvLayout>>;
        using DSingleConvType = remove_cvref_t<tuple_element_t<conv_phase, DsDataType>>;
        using DDataType       = remove_cvref_t<tuple_element_t<DIndex{}.value, DSingleConvType>>;
        using DBlockTransferThreadClusterLengths =
            remove_cvref_t<tuple_element_t<conv_phase, DsBlockTransferThreadClusterLengths>>;
        constexpr auto DBlockTransferDstScalarPerVector = remove_cvref_t<
            tuple_element_t<conv_phase, DsBlockTransferDstScalarPerVector>>{}[DIndex{}];
        constexpr auto DBlockTransferSrcScalarPerVector = remove_cvref_t<
            tuple_element_t<conv_phase, DsBlockTransferSrcScalarPerVector>>{}[DIndex{}];

        if constexpr(std::is_same_v<DLayout, tensor_layout::convolution::G_K>)
        {
            using DBlockTransferThreadClusterArrangeOrder = Sequence<0>;
            using DBlockTransferAccessOrder               = Sequence<0>;

            constexpr index_t DBlockTransferVectorDim = 0;

            if constexpr(EnableAsync)
            {
                return ThreadGroupTensorSliceTransferAsync<ThisThreadBlockGrid,
                                                           Sequence<KPerBlock::At(conv_phase)>,
                                                           DBlockTransferThreadClusterLengths,
                                                           DBlockTransferThreadClusterArrangeOrder,
                                                           DDataType,
                                                           DDataType,
                                                           DGridBlockDesc,
                                                           DBlockDesc,
                                                           DBlockTransferVectorDim,
                                                           DBlockTransferVectorDim,
                                                           DBlockTransferDstScalarPerVector,
                                                           false,
                                                           true>(
                    d_grid_block_desc,
                    make_multi_index(k_block_data_idx_on_grid),
                    DBlockDesc{},
                    make_multi_index(0));
            }
            else
            {
                return ThreadGroupTensorSliceTransfer_v4r1<
                    ThisThreadBlockGrid,
                    ck::tensor_operation::element_wise::PassThrough,
                    ck::tensor_operation::element_wise::PassThrough,
                    InMemoryDataOperationEnum::Set,
                    Sequence<KPerBlock::At(conv_phase)>,
                    DBlockTransferThreadClusterLengths,
                    DBlockTransferThreadClusterArrangeOrder,
                    DDataType,
                    DDataType,
                    DGridBlockDesc,
                    DBlockDesc,
                    DBlockTransferAccessOrder,
                    DBlockTransferAccessOrder,
                    DBlockTransferVectorDim,
                    DBlockTransferVectorDim,
                    DBlockTransferSrcScalarPerVector,
                    DBlockTransferDstScalarPerVector,
                    1,
                    1,
                    false,
                    true,
                    NumConvCPrefetchStage>(d_grid_block_desc,
                                           make_multi_index(k_block_data_idx_on_grid),
                                           ck::tensor_operation::element_wise::PassThrough{},
                                           DBlockDesc{},
                                           make_multi_index(0),
                                           ck::tensor_operation::element_wise::PassThrough{});
            }
        }
        else
        {
            using DBlockTransferThreadClusterArrangeOrder = Sequence<0, 1, 2>;
            using DBlockTransferAccessOrder               = Sequence<0, 1, 2>;
            constexpr index_t DBlockTransferVectorDim     = 2;
            static_assert(DBlockDesc{}.GetLength(I0) == HPerBlock);
            static_assert(DBlockDesc{}.GetLength(I1) == WPerBlock);
            static_assert(DBlockDesc{}.GetLength(I2) == KPerBlock::At(conv_phase));
            if constexpr(EnableAsync)
            {
                return ThreadGroupTensorSliceTransferAsync<
                    ThisThreadBlockGrid,
                    Sequence<HPerBlock, WPerBlock, KPerBlock::At(conv_phase)>,
                    DBlockTransferThreadClusterLengths,
                    DBlockTransferThreadClusterArrangeOrder,
                    DDataType,
                    DDataType,
                    DGridBlockDesc,
                    DBlockDesc,
                    DBlockTransferVectorDim,
                    DBlockTransferVectorDim,
                    DBlockTransferDstScalarPerVector,
                    false,
                    true>(d_grid_block_desc,
                          make_multi_index(h_block_data_idx_on_grid,
                                           w_block_data_idx_on_grid,
                                           k_block_data_idx_on_grid),
                          DBlockDesc{},
                          make_multi_index(0, 0, 0));
            }
            else
            {
                return ThreadGroupTensorSliceTransfer_v4r1<
                    ThisThreadBlockGrid,
                    ck::tensor_operation::element_wise::PassThrough,
                    ck::tensor_operation::element_wise::PassThrough,
                    InMemoryDataOperationEnum::Set,
                    Sequence<HPerBlock, WPerBlock, KPerBlock::At(conv_phase)>,
                    DBlockTransferThreadClusterLengths,
                    DBlockTransferThreadClusterArrangeOrder,
                    DDataType,
                    DDataType,
                    DGridBlockDesc,
                    DBlockDesc,
                    DBlockTransferAccessOrder,
                    DBlockTransferAccessOrder,
                    DBlockTransferVectorDim,
                    DBlockTransferVectorDim,
                    DBlockTransferSrcScalarPerVector,
                    DBlockTransferDstScalarPerVector,
                    1,
                    1,
                    false,
                    true,
                    NumConvCPrefetchStage>(d_grid_block_desc,
                                           make_multi_index(h_block_data_idx_on_grid,
                                                            w_block_data_idx_on_grid,
                                                            k_block_data_idx_on_grid),
                                           ck::tensor_operation::element_wise::PassThrough{},
                                           DBlockDesc{},
                                           make_multi_index(0, 0, 0),
                                           ck::tensor_operation::element_wise::PassThrough{});
            }
        }
    }

    template <typename Block2CTileMap>
    __host__ __device__ static constexpr bool CheckValidity(const InGridDesc& in_grid_desc,
                                                            const WeiGridDesc& wei_grid_desc,
                                                            const EGridDesc& e_grid_desc,
                                                            const Block2CTileMap& block_2_ctile_map)
    {
        static_assert(HPerBlock % (HRepeat * HPerWcnn) == 0, "");
        static_assert(WPerBlock % (WRepeat * WPerWcnn) == 0, "");

        const auto GetIn0Problemsize = [&]() {
            return make_tuple(in_grid_desc[I0].GetLength(I0),
                              in_grid_desc[I0].GetLength(I1),
                              in_grid_desc[I0].GetLength(I2));
        };

        const auto GetWei0Problemsize = [&]() {
            return make_tuple(wei_grid_desc[I0].GetLength(I0),
                              wei_grid_desc[I0].GetLength(I1),
                              wei_grid_desc[I0].GetLength(I2));
        };

        const auto GetWei1Problemsize = [&]() {
            return make_tuple(wei_grid_desc[I1].GetLength(I0),
                              wei_grid_desc[I1].GetLength(I1),
                              wei_grid_desc[I1].GetLength(I2));
        };

        const auto GetWei2Problemsize = [&]() {
            return make_tuple(wei_grid_desc[I2].GetLength(I0),
                              wei_grid_desc[I2].GetLength(I1),
                              wei_grid_desc[I2].GetLength(I2));
        };

        const auto H      = GetIn0Problemsize()[I0];
        const auto W      = GetIn0Problemsize()[I1];
        const auto C_0    = 32;
        const auto C_1    = 32;
        const auto C_2    = 64;
        const auto K_0    = GetWei0Problemsize()[I0];
        const auto K_1    = GetWei1Problemsize()[I0];
        const auto K_2    = GetWei2Problemsize()[I0];
        const auto Wei0_C = GetWei0Problemsize()[I2];
        const auto Wei1_C = GetWei1Problemsize()[I2];
        const auto Wei2_C = GetWei2Problemsize()[I2];

#if defined(CASCADE_1X_OUT)
        if(!(W == e_grid_desc.GetLength(I1) && K_0 == e_grid_desc.GetLength(I2)) ||
#elif defined(CASCADE_2X_OUT)
        if(!(W == e_grid_desc.GetLength(I1) && K_1 == e_grid_desc.GetLength(I2)) ||
#else
        if(!(W == e_grid_desc.GetLength(I1) && K_2 == e_grid_desc.GetLength(I2)) ||
#endif
           !(C_0 == Wei0_C && C_1 == Wei1_C && C_2 == Wei2_C))
        {
            printf("Tensor: HW = %d x %d , Filter: K0:%d K1:%d K2:%d C0:%d C1:%d C2:%d, wei0_C:%d "
                   "wei1_C:%d wei2_C:%d, Out: HWK = %d "
                   "x %d x %d\n",
                   H,
                   W,
                   K_0,
                   K_1,
                   K_2,
                   C_0,
                   C_1,
                   C_2,
                   Wei0_C,
                   Wei1_C,
                   Wei2_C,
                   e_grid_desc.GetLength(I0),
                   e_grid_desc.GetLength(I1),
                   e_grid_desc.GetLength(I2));
            printf("GridwiseOp err: ProblemSize check\n");
            return false;
        }

        if(!(H % HPerBlock == 0 && W % WPerBlock == 0 && K_0 % KPerBlock::At(0) == 0 &&
             K_1 % KPerBlock::At(1) == 0 && K_2 % KPerBlock::At(2) == 0 &&
             C_0 % CPerBlock::At(0) == 0 && C_1 % CPerBlock::At(1) == 0 &&
             C_2 % CPerBlock::At(2) == 0))
        {
            printf("GridwiseOp err: ProblemSize division\n");
            return false;
        }

        // check gridwise conv pipeline
        const auto num_c_0_loop = C_0 / CPerBlock::At(0);
        const auto num_c_1_loop = C_1 / CPerBlock::At(1);
        const auto num_c_2_loop = C_2 / CPerBlock::At(2);

        if(!GridwiseFasternet50Pipe::IsSupported(num_c_0_loop))
        {
            printf("GridwiseOp err: Pipeline not support this c_0_loop\n");
            return false;
        }

        if(!GridwiseFasternet50Pipe::IsSupported(num_c_1_loop))
        {
            printf("GridwiseOp err: Pipeline not support this c_1_loop\n");
            return false;
        }

        if(!GridwiseFasternet50Pipe::IsSupported(num_c_2_loop))
        {
            printf("GridwiseOp err: Pipeline not support this c_2_loop\n");
            return false;
        }

        if(!block_2_ctile_map.CheckValidity(e_grid_desc))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        constexpr long_index_t TwoGB = (long_index_t{1} << 31);

        if(!(in_grid_desc[I0].GetElementSpaceSize() * sizeof(InDataType) <= TwoGB &&
             wei_grid_desc[I0].GetElementSpaceSize() * sizeof(WeiDataType) <= TwoGB &&
             wei_grid_desc[I1].GetElementSpaceSize() * sizeof(WeiDataType) <= TwoGB &&
             wei_grid_desc[I2].GetElementSpaceSize() * sizeof(WeiDataType) <= TwoGB &&
             e_grid_desc.GetElementSpaceSize() * sizeof(InDataType) <= TwoGB))
        {
            return false;
        }

        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainBlockLoop(index_t C)
    {
        const index_t num_loop = C / CPerBlock::At(0);

        return GridwiseFasternet50Pipe::CalculateHasMainLoop(num_loop);
    }

    static constexpr auto in_0_block_desc  = MakeInBlockDescriptor<0>();
    static constexpr auto in_1_block_desc  = MakeInBlockDescriptor<1>();
    static constexpr auto wei_0_block_desc = MakeSingleWeiBlockDescriptor<0>();
    static constexpr auto wei_1_block_desc = MakeSingleWeiBlockDescriptor<1>();
    static constexpr auto wei_2_block_desc = MakeSingleWeiBlockDescriptor<2>();
    static constexpr auto in_cluster_border_block_desc =
        MakeInClusterBorderBlockDescriptor<true, 0>();
    static constexpr auto in_wavegroup_border_block_desc =
        MakeInClusterBorderBlockDescriptor<false, 0>();
    static constexpr index_t max_lane_shared_align = 4;

    static constexpr index_t in_border_block_space_size_aligned =
        EnableWaveGroup
            ? math::integer_least_multiple(in_wavegroup_border_block_desc.GetElementSpaceSize() *
                                               sizeof(int32_t),
                                           max_lane_shared_align)
            : 0;
    static constexpr index_t in_cluster_block_space_size_aligned =
        EnableWaveGroup
            ? math::integer_least_multiple(in_cluster_border_block_desc.GetElementSpaceSize() *
                                               sizeof(InDataType),
                                           max_lane_shared_align)
            : 0;

    template <bool HasMainBlockLoop>
    struct LaneSharedMemTrait
    {
        static constexpr index_t mem_count_0 =
            BlockwiseConv_0::template GetLaneSharedMemCount<HasMainBlockLoop>();
        static constexpr index_t mem_count_1 =
            BlockwiseConv_1::template GetLaneSharedMemCount<HasMainBlockLoop>();
        static constexpr index_t mem_count_2 =
            BlockwiseConv_1::template GetLaneSharedMemCount<HasMainBlockLoop>();
        static constexpr index_t in_block_space_offset = 0;

        static constexpr index_t pre_in_block_space_offset =
            EnableSpatialCluster
                ? (in_block_space_offset +
                   math::max(
                       BlockwiseConv_0::LaneSharedMemTrait::in_block_space_size_aligned *
                           mem_count_0,
                       BlockwiseConv_1::LaneSharedMemTrait::out_block_space_aligned* mem_count_1))
                : in_block_space_offset;

        static constexpr index_t pre_cluster_in_block_space_offset = pre_in_block_space_offset;

        static constexpr index_t pre_offset = math::max(
            pre_in_block_space_offset + in_border_block_space_size_aligned * mem_count_0,
            pre_cluster_in_block_space_offset + in_cluster_block_space_size_aligned * mem_count_0);

        static constexpr index_t next_in_block_space_offset =
            EnableSpatialCluster ? pre_offset : in_block_space_offset;

        static constexpr index_t next_cluster_in_block_space_offset = next_in_block_space_offset;

        static constexpr index_t next_offset = math::max(
            next_in_block_space_offset + in_border_block_space_size_aligned * mem_count_0,
            next_cluster_in_block_space_offset + in_cluster_block_space_size_aligned * mem_count_0);

        static constexpr index_t wei_0_block_space_offset =
            EnableSpatialCluster
                ? next_offset
                : (in_block_space_offset +
                   math::max(
                       BlockwiseConv_0::LaneSharedMemTrait::in_block_space_size_aligned *
                           mem_count_0,
                       BlockwiseConv_1::LaneSharedMemTrait::out_block_space_aligned* mem_count_1));

        static constexpr index_t wei_1_block_space_offset = wei_0_block_space_offset;

        static constexpr index_t wei_2_block_space_offset = wei_0_block_space_offset;

        static constexpr index_t ds_0_base_offset =
            wei_0_block_space_offset +
            BlockwiseConv_0::LaneSharedMemTrait::wei_block_space_size_aligned * mem_count_0;

        static constexpr index_t ds_1_base_offset =
            wei_1_block_space_offset +
            BlockwiseConv_1::LaneSharedMemTrait::wei_block_space_size_aligned * mem_count_1;

        static constexpr index_t ds_2_base_offset =
            wei_2_block_space_offset +
            BlockwiseConv_2::LaneSharedMemTrait::wei_block_space_size_aligned * mem_count_2;

        static constexpr auto ds_0_block_space_offset = generate_tuple(
            [](auto i) {
                return BlockwiseConv_0::LaneSharedMemTrait::ds_block_space_offset[i] +
                       ds_0_base_offset;
            },
            Number<NumDTensor_0>{});
        static constexpr auto ds_1_block_space_offset = generate_tuple(
            [](auto i) {
                return BlockwiseConv_1::LaneSharedMemTrait::ds_block_space_offset[i] +
                       ds_1_base_offset;
            },
            Number<NumDTensor_1>{});
        static constexpr auto ds_2_block_space_offset = generate_tuple(
            [](auto i) {
                return BlockwiseConv_2::LaneSharedMemTrait::ds_block_space_offset[i] +
                       ds_2_base_offset;
            },
            Number<NumDTensor_2>{});

        static constexpr auto acc_0_block_space_offset =
            ds_0_base_offset + BlockwiseConv_0::LaneSharedMemTrait::ds_block_space_size_aligned;
        static constexpr auto acc_1_block_space_offset =
            ds_1_base_offset + BlockwiseConv_1::LaneSharedMemTrait::ds_block_space_size_aligned;
        static constexpr auto acc_2_block_space_offset =
            ds_2_base_offset + BlockwiseConv_2::LaneSharedMemTrait::ds_block_space_size_aligned;

        static constexpr auto acc_0_block_space_size_aligned =
            HasMainBlockLoop ? BlockwiseConv_0::LaneSharedMemTrait::acc_block_space_aligned
                             : BlockwiseConv_0::LaneSharedMemTrait::acc_ring_block_space_aligned;
        static constexpr auto acc_1_block_space_size_aligned =
            HasMainBlockLoop ? BlockwiseConv_1::LaneSharedMemTrait::acc_block_space_aligned
                             : BlockwiseConv_1::LaneSharedMemTrait::acc_ring_block_space_aligned;
        static constexpr auto acc_2_block_space_size_aligned =
            HasMainBlockLoop ? BlockwiseConv_2::LaneSharedMemTrait::acc_block_space_aligned
                             : BlockwiseConv_2::LaneSharedMemTrait::acc_ring_block_space_aligned;

        static constexpr auto out_0_block_space_offset =
            acc_0_block_space_offset + acc_0_block_space_size_aligned;
        static constexpr auto out_1_block_space_offset =
            acc_1_block_space_offset + acc_1_block_space_size_aligned;
        static constexpr auto out_2_block_space_offset =
            acc_2_block_space_offset + acc_2_block_space_size_aligned;

        static constexpr auto lane_shared_size =
            EnableWaveGroup4
                ? math::max(out_0_block_space_offset +
                                BlockwiseConv_0::LaneSharedMemTrait::out_block_space_aligned,
                            out_1_block_space_offset +
                                BlockwiseConv_1::LaneSharedMemTrait::out_block_space_aligned,
                            out_2_block_space_offset +
                                BlockwiseConv_2::LaneSharedMemTrait::out_block_space_aligned)
                : math::max(
                      acc_0_block_space_offset, acc_1_block_space_offset, acc_2_block_space_offset);
    };

    template <bool HasMainLoop>
    static constexpr auto GetLaneSharedMemTrait()
    {
        return LaneSharedMemTrait<HasMainLoop>{};
    }
    // Return block_id to Acc tensor tile idx (k0, h0, w0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2CTileMap(const EGridDesc& c_grid_desc_h_w_k, index_t M01, index_t /* N01 */)
    {
        static_assert(BlockwiseConv_0::HPerWaveOut == BlockwiseConv_2::HPerWaveOut,
                      "H should be same between input and output for Block2CTileMap!");
        static_assert(BlockwiseConv_0::HPerWaveOut == BlockwiseConv_1::HPerWaveOut,
                      "H should be same between input and output for Block2CTileMap!");
        static_assert(BlockwiseConv_0::WPerWaveOut == BlockwiseConv_2::WPerWaveOut,
                      "W should be same between input and output for Block2CTileMap!");
        static_assert(BlockwiseConv_0::WPerWaveOut == BlockwiseConv_1::WPerWaveOut,
                      "W should be same between input and output for Block2CTileMap!");

#if defined(CASCADE_1X_OUT)
        constexpr index_t K_OUT_PERBLOCK = KPerBlock::At(0);
#elif defined(CASCADE_2X_OUT)
        constexpr index_t K_OUT_PERBLOCK = KPerBlock::At(1);
#else
        constexpr index_t K_OUT_PERBLOCK = KPerBlock::At(2);
#endif
        return BlockToCTileMap_KSplit_M00_N0_M01Adapt<BlockwiseConv_1::HPerBlockOut,
                                                      BlockwiseConv_1::WPerBlockOut,
                                                      EGridDesc>(
            c_grid_desc_h_w_k, M01, c_grid_desc_h_w_k.GetLength(I2) / K_OUT_PERBLOCK);
    }

    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(EGridDesc{}, 1, 1))>;

    template <bool HasMainBlockLoop, typename Block2CTileMap = DefaultBlock2CTileMap>
    __device__ static void Run(const InDataType* __restrict__ p_in_grid,
                               WeiGridPointer p_wei_grid,
                               DsGridPointer p_ds_grid,
                               EDataType* __restrict__ p_e_grid,
                               void* __restrict__ p_shared,
                               void* __restrict__ p_lane_shared,
                               const InGridDesc& in_grid_desc,
                               const WeiGridDesc& wei_grid_desc,
                               const DsGridDesc& ds_grid_desc,
                               const EGridDesc& e_grid_desc,
                               const InElementwiseOperation& in_element_op,
                               const WeiElementwiseOperation& wei_element_op,
                               const AccElementwiseOperation& acc_element_op,
                               const Block2CTileMap& block_2_ctile_map)
    {
#if defined(__gfx13__)
        /*******************************************************************************/
        // Memory buffer zone.
        const auto in_0_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_grid, in_grid_desc[I0].GetElementSpaceSize());
        const auto in_1_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_grid, in_grid_desc[I1].GetElementSpaceSize());
        const auto in_2_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_grid, in_grid_desc[I2].GetElementSpaceSize());
        const auto wei_0_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_wei_grid[I0], wei_grid_desc[I0].GetElementSpaceSize());
        const auto wei_1_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_wei_grid[I1], wei_grid_desc[I1].GetElementSpaceSize());
        const auto wei_2_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_wei_grid[I2], wei_grid_desc[I2].GetElementSpaceSize());
        const auto ds_0_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_ds_grid[I0][i], ds_grid_desc[I0][i].GetElementSpaceSize());
            },
            Number<NumDTensor_0>{});
        const auto ds_1_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_ds_grid[I1][i], ds_grid_desc[I1][i].GetElementSpaceSize());
            },
            Number<NumDTensor_1>{});
        const auto ds_2_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_ds_grid[I2][i], ds_grid_desc[I2][i].GetElementSpaceSize());
            },
            Number<NumDTensor_2>{});
        const auto in_grid_pad_desc = MakeInGridPadDescriptor(in_grid_desc);
        const auto in_0_grid_block_desc =
            BlockwiseConv_0::MakeInGridBlockDescriptor(in_grid_pad_desc[I0]);
        const auto in_1_grid_block_desc =
            BlockwiseConv_1::MakeInGridBlockDescriptor(in_grid_pad_desc[I1]);
        const auto wei_0_grid_block_desc =
            BlockwiseConv_0::MakeWeiGridBlockDescriptor(wei_grid_desc[I0]);
        const auto wei_1_grid_block_desc =
            BlockwiseConv_1::MakeWeiGridBlockDescriptor(wei_grid_desc[I1]);
        const auto wei_2_grid_block_desc =
            BlockwiseConv_2::MakeWeiGridBlockDescriptor(wei_grid_desc[I2]);

        const auto ds_0_grid_block_desc =
            BlockwiseConv_0::template MakeSingleDsGridBlockDescriptor<0>(ds_grid_desc);
        const auto ds_1_grid_block_desc =
            BlockwiseConv_1::template MakeSingleDsGridBlockDescriptor<1>(ds_grid_desc);
        const auto ds_2_grid_block_desc =
            BlockwiseConv_2::template MakeSingleDsGridBlockDescriptor<2>(ds_grid_desc);

        /*******************************************************************************/
        // BlockIdx.x -> [BlockId.k, BlockId.h, BlockId.w]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));
        if(!block_2_ctile_map.ValidCTileIndex(block_work_idx,
                                              make_tuple(e_grid_desc.GetLength(I0),
                                                         e_grid_desc.GetLength(I1),
                                                         e_grid_desc.GetLength(I2))))
        {
            printf("The block_2_ctile is invalid!");
            return;
        }

        // Store BlockId into SGPR
        const index_t k0_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * KPerBlock::At(0));
        const index_t k1_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * KPerBlock::At(1));
        const index_t k2_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * KPerBlock::At(2));
        const index_t w_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I2] * WPerBlock);
        const index_t h_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * HPerBlock);

        const index_t w_out_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I2] * BlockwiseConv_1::WPerBlockOut);
        const index_t h_out_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * BlockwiseConv_1::HPerBlockOut);

        /*******************************************************************************/
        // BlockLevel, Tensor and filter ThreadMapping in WCNN Source buffer, As Destinaion of
        // BlockWise_Copy
        auto wave_idx = GetWcnnWaveIdx<ThisThreadBlockGrid,
                                       HPerBlock,
                                       WPerBlock,
                                       HRepeat,
                                       WRepeat,
                                       HPerWcnn,
                                       WPerWcnn>();

        constexpr auto laneSharedMemTrait = GetLaneSharedMemTrait<HasMainBlockLoop>();

        auto in_0_block_trait = [&]() {
            // input data blockwise copy
            if constexpr(InEnableLds)
            {
                auto in_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<InDataType*>(p_shared),
                    BlockwiseConv_0::SharedMemTrait::in_block_space_size_aligned);
                using InBlockTransferThreadClusterArrangeOrder = Sequence<0, 1, 2>;
                using InBlockTransferAccessOrder               = Sequence<0, 1, 2>;
                constexpr index_t InBlockTransferVectorDim     = 2;

                if constexpr(EnableAsync)
                {
                    auto in_blockwise_copy = ThreadGroupTensorSliceTransferAsync<
                        ThisThreadBlockGrid,
                        Sequence<HPerBlockIn[0], WPerBlockIn[0], CPerBlock::At(0)>,
                        InBlockTransferThreadClusterLengths,
                        InBlockTransferThreadClusterArrangeOrder,
                        InDataType,
                        InDataType,
                        decltype(in_0_grid_block_desc),
                        decltype(in_0_block_desc),
                        InBlockTransferVectorDim,
                        InBlockTransferVectorDim,
                        InBlockTransferDstScalarPerVector,
                        false,
                        true>(
                        in_0_grid_block_desc,
                        make_multi_index(h_block_data_idx_on_grid, w_block_data_idx_on_grid, 0),
                        in_0_block_desc,
                        make_multi_index(0, 0, 0));
                    return make_tuple(in_block_buf, in_blockwise_copy);
                }
                else
                {
                    auto in_blockwise_copy = ThreadGroupTensorSliceTransfer_v4r1<
                        ThisThreadBlockGrid,
                        InElementwiseOperation,
                        ck::tensor_operation::element_wise::PassThrough,
                        InMemoryDataOperationEnum::Set,
                        Sequence<HPerBlockIn[0], WPerBlockIn[0], CPerBlock::At(0)>,
                        InBlockTransferThreadClusterLengths,
                        InBlockTransferThreadClusterArrangeOrder,
                        InDataType,
                        InDataType,
                        decltype(in_0_grid_block_desc),
                        decltype(in_0_block_desc),
                        InBlockTransferAccessOrder,
                        InBlockTransferAccessOrder,
                        InBlockTransferVectorDim,
                        InBlockTransferVectorDim,
                        InBlockTransferSrcScalarPerVector,
                        InBlockTransferDstScalarPerVector,
                        1,
                        1,
                        false,
                        true,
                        NumConvCPrefetchStage>(
                        in_0_grid_block_desc,
                        make_multi_index(h_block_data_idx_on_grid, w_block_data_idx_on_grid, 0),
                        in_element_op,
                        in_0_block_desc,
                        make_multi_index(0, 0, 0),
                        ck::tensor_operation::element_wise::PassThrough{});
                    return make_tuple(in_block_buf, in_blockwise_copy);
                }
            }
            else
            {
                auto indata_slice_origin_idx =
                    wcnn_conv_3x3.template CalculateInDataThreadOriginDataIndex<InTileLoad>();
                auto h0 = (h_block_data_idx_on_grid + wave_idx[I0] * HPerWave) / HPerWcnn;
                auto w0 = (w_block_data_idx_on_grid + wave_idx[I1] * WPerWave) / WPerWcnn;
                constexpr index_t threadsPerTensorTile = (WPerWcnn == 4) ? 2 : 4;
                constexpr index_t vgprPerTensorTile    = (HPerWcnn == 8) ? 2 : 1;
                // Limitation: NumDim of Src and Dst descriptor should be identical
                auto in_blockwise_copy = ThreadwiseTensorSliceTransfer_v2<
                    InDataType,
                    InDataType,
                    decltype(in_0_grid_block_desc),
                    decltype(in_0_block_desc),
                    decltype(BlockwiseConv_0::GetInWaveDescLength()),
                    Sequence<0, 1, 2, 3, 4, 5, 6>,
                    6,
                    NumDataCompPerTile[0],
                    1,
                    false,
                    false,
                    false,
                    false,
                    InTileLoad,
                    threadsPerTensorTile,
                    vgprPerTensorTile>(in_0_grid_block_desc,
                                       make_multi_index(w0,
                                                        0,
                                                        h0,
                                                        0,
                                                        indata_slice_origin_idx[I0],
                                                        indata_slice_origin_idx[I1],
                                                        indata_slice_origin_idx[I2]));

                if constexpr(EnableWaveGroup)
                {
                    static_assert(laneSharedMemTrait.in_block_space_offset == 0, "");
                    return make_tuple(
                        make_static_buffer_v4<AddressSpaceEnum::Vgpr, InDataType>(
                            in_0_block_desc.GetElementSpaceSize(),
                            static_cast<InDataType*>(p_lane_shared) +
                                laneSharedMemTrait.in_block_space_offset / sizeof(InDataType)),
                        in_blockwise_copy);
                }
                else
                {
                    return make_tuple(make_static_buffer_v2<AddressSpaceEnum::Vgpr, InDataType>(
                                          in_0_block_desc.GetElementSpaceSize()),
                                      in_blockwise_copy);
                }
            }
        };

        auto in_1_block_trait = [&]() {
            // input data blockwise copy
            if constexpr(InEnableLds)
            {
                auto in_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<InDataType*>(p_shared),
                    BlockwiseConv_1::SharedMemTrait::in_block_space_size_aligned);
                using InBlockTransferThreadClusterArrangeOrder = Sequence<0, 1, 2>;
                using InBlockTransferAccessOrder               = Sequence<0, 1, 2>;
                constexpr index_t InBlockTransferVectorDim     = 2;

                if constexpr(EnableAsync)
                {
                    auto in_blockwise_copy = ThreadGroupTensorSliceTransferAsync<
                        ThisThreadBlockGrid,
                        Sequence<HPerBlockIn[1], WPerBlockIn[1], CPerBlock::At(0)>,
                        InBlockTransferThreadClusterLengths,
                        InBlockTransferThreadClusterArrangeOrder,
                        InDataType,
                        InDataType,
                        decltype(in_1_grid_block_desc),
                        decltype(in_1_block_desc),
                        InBlockTransferVectorDim,
                        InBlockTransferVectorDim,
                        InBlockTransferDstScalarPerVector,
                        false,
                        true>(
                        in_1_grid_block_desc,
                        make_multi_index(h_block_data_idx_on_grid, w_block_data_idx_on_grid, 0),
                        in_1_block_desc,
                        make_multi_index(0, 0, 0));
                    return make_tuple(in_block_buf, in_blockwise_copy);
                }
                else
                {
                    auto in_blockwise_copy = ThreadGroupTensorSliceTransfer_v4r1<
                        ThisThreadBlockGrid,
                        InElementwiseOperation,
                        ck::tensor_operation::element_wise::PassThrough,
                        InMemoryDataOperationEnum::Set,
                        Sequence<HPerBlockIn[1], WPerBlockIn[1], CPerBlock::At(0)>,
                        InBlockTransferThreadClusterLengths,
                        InBlockTransferThreadClusterArrangeOrder,
                        InDataType,
                        InDataType,
                        decltype(in_1_grid_block_desc),
                        decltype(in_1_block_desc),
                        InBlockTransferAccessOrder,
                        InBlockTransferAccessOrder,
                        InBlockTransferVectorDim,
                        InBlockTransferVectorDim,
                        InBlockTransferSrcScalarPerVector,
                        InBlockTransferDstScalarPerVector,
                        1,
                        1,
                        false,
                        true,
                        NumConvCPrefetchStage>(
                        in_1_grid_block_desc,
                        make_multi_index(h_block_data_idx_on_grid, w_block_data_idx_on_grid, 0),
                        in_element_op,
                        in_1_block_desc,
                        make_multi_index(0, 0, 0),
                        ck::tensor_operation::element_wise::PassThrough{});
                    return make_tuple(in_block_buf, in_blockwise_copy);
                }
            }
            else
            {
                auto indata_slice_origin_idx =
                    wcnn_conv_1x1.template CalculateInDataThreadOriginDataIndex<InTileLoad>();
                auto h0 = (h_block_data_idx_on_grid + wave_idx[I0] * HPerWave) / HPerWcnn;
                auto w0 = (w_block_data_idx_on_grid + wave_idx[I1] * WPerWave) / WPerWcnn;
                constexpr index_t threadsPerTensorTile = (WPerWcnn == 4) ? 2 : 4;
                constexpr index_t vgprPerTensorTile    = (HPerWcnn == 8) ? 2 : 1;
                // Limitation: NumDim of Src and Dst descriptor should be identical
                auto in_blockwise_copy = ThreadwiseTensorSliceTransfer_v2<
                    InDataType,
                    InDataType,
                    decltype(in_1_grid_block_desc),
                    decltype(in_1_block_desc),
                    decltype(BlockwiseConv_1::GetInWaveDescLength()),
                    Sequence<0, 1, 2, 3, 4, 5, 6>,
                    6,
                    NumDataCompPerTile[1],
                    1,
                    false,
                    false,
                    false,
                    false,
                    InTileLoad,
                    threadsPerTensorTile,
                    vgprPerTensorTile>(in_1_grid_block_desc,
                                       make_multi_index(w0,
                                                        0,
                                                        h0,
                                                        0,
                                                        indata_slice_origin_idx[I0],
                                                        indata_slice_origin_idx[I1],
                                                        indata_slice_origin_idx[I2]));

                if constexpr(EnableWaveGroup)
                {
                    static_assert(laneSharedMemTrait.in_block_space_offset == 0, "");
                    return make_tuple(
                        make_static_buffer_v4<AddressSpaceEnum::Vgpr, InDataType>(
                            in_1_block_desc.GetElementSpaceSize(),
                            static_cast<InDataType*>(p_lane_shared) +
                                laneSharedMemTrait.in_block_space_offset / sizeof(InDataType)),
                        in_blockwise_copy);
                }
                else
                {
                    return make_tuple(make_static_buffer_v2<AddressSpaceEnum::Vgpr, InDataType>(
                                          in_1_block_desc.GetElementSpaceSize()),
                                      in_blockwise_copy);
                }
            }
        };

        auto pre_block_buf = make_static_buffer_v4<AddressSpaceEnum::Vgpr, int32_t>(
            in_wavegroup_border_block_desc.GetElementSpaceSize(),
            static_cast<int32_t*>(p_lane_shared) +
                laneSharedMemTrait.pre_in_block_space_offset / sizeof(int32_t));

        auto next_block_buf = make_static_buffer_v4<AddressSpaceEnum::Vgpr, int32_t>(
            in_wavegroup_border_block_desc.GetElementSpaceSize(),
            static_cast<int32_t*>(p_lane_shared) +
                laneSharedMemTrait.next_in_block_space_offset / sizeof(int32_t));

        auto pre_cluster_buf = make_static_buffer_v4<AddressSpaceEnum::Vgpr, InDataType>(
            in_cluster_border_block_desc.GetElementSpaceSize(),
            static_cast<InDataType*>(p_lane_shared) +
                laneSharedMemTrait.pre_cluster_in_block_space_offset / sizeof(InDataType));

        auto next_cluster_buf = make_static_buffer_v4<AddressSpaceEnum::Vgpr, InDataType>(
            in_cluster_border_block_desc.GetElementSpaceSize(),
            static_cast<InDataType*>(p_lane_shared) +
                laneSharedMemTrait.next_cluster_in_block_space_offset / sizeof(InDataType));

        auto pre_border_block_trait = [&]() {
            // input data blockwise copy
            if constexpr(InEnableLds)
            {
                // Not support in LDS
                auto in_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<InDataType*>(p_shared),
                    BlockwiseConv_0::SharedMemTrait::in_block_space_size_aligned);
                using InBlockTransferThreadClusterArrangeOrder = Sequence<0, 1, 2>;
                using InBlockTransferAccessOrder               = Sequence<0, 1, 2>;
                constexpr index_t InBlockTransferVectorDim     = 2;

                if constexpr(EnableAsync)
                {
                    auto in_blockwise_copy = ThreadGroupTensorSliceTransferAsync<
                        ThisThreadBlockGrid,
                        Sequence<HPerBlockIn[0], WPerBlockIn[0], CPerBlock::At(0)>,
                        InBlockTransferThreadClusterLengths,
                        InBlockTransferThreadClusterArrangeOrder,
                        InDataType,
                        InDataType,
                        decltype(in_0_grid_block_desc),
                        decltype(in_0_block_desc),
                        InBlockTransferVectorDim,
                        InBlockTransferVectorDim,
                        InBlockTransferDstScalarPerVector,
                        false,
                        true>(
                        in_0_grid_block_desc,
                        make_multi_index(h_block_data_idx_on_grid, w_block_data_idx_on_grid, 0),
                        in_0_block_desc,
                        make_multi_index(0, 0, 0));
                    return make_tuple(in_block_buf, in_blockwise_copy);
                }
                else
                {
                    auto in_blockwise_copy = ThreadGroupTensorSliceTransfer_v4r1<
                        ThisThreadBlockGrid,
                        InElementwiseOperation,
                        ck::tensor_operation::element_wise::PassThrough,
                        InMemoryDataOperationEnum::Set,
                        Sequence<HPerBlockIn[0], WPerBlockIn[0], CPerBlock::At(0)>,
                        InBlockTransferThreadClusterLengths,
                        InBlockTransferThreadClusterArrangeOrder,
                        InDataType,
                        InDataType,
                        decltype(in_0_grid_block_desc),
                        decltype(in_0_block_desc),
                        InBlockTransferAccessOrder,
                        InBlockTransferAccessOrder,
                        InBlockTransferVectorDim,
                        InBlockTransferVectorDim,
                        InBlockTransferSrcScalarPerVector,
                        InBlockTransferDstScalarPerVector,
                        1,
                        1,
                        false,
                        true,
                        NumConvCPrefetchStage>(
                        in_0_grid_block_desc,
                        make_multi_index(h_block_data_idx_on_grid, w_block_data_idx_on_grid, 0),
                        in_element_op,
                        in_0_block_desc,
                        make_multi_index(0, 0, 0),
                        ck::tensor_operation::element_wise::PassThrough{});
                    return make_tuple(in_block_buf, in_blockwise_copy);
                }
            }
            else
            {
                auto indata_slice_origin_idx =
                    wcnn_conv_3x3.template CalculateInDataThreadOriginDataIndex<InTileLoad>();
                auto h0 = (h_block_data_idx_on_grid + wave_idx[I0] * HPerWave) / HPerWcnn;
                auto w0 = (w_block_data_idx_on_grid + wave_idx[I1] * WPerWave) / WPerWcnn;
                constexpr index_t threadsPerTensorTile = (WPerWcnn == 4) ? 2 : 4;
                constexpr index_t vgprPerTensorTile    = (HPerWcnn == 8) ? 2 : 1;

                // Limitation: NumDim of Src and Dst descriptor should be identical
                auto pre_blockwise_copy = ThreadwiseTensorSliceTransfer_v2<
                    InDataType,
                    InDataType,
                    decltype(in_0_grid_block_desc),
                    decltype(in_cluster_border_block_desc),
                    decltype(BlockwiseConv_0::GetInBorderWaveDescLength()),
                    Sequence<0, 1, 2, 3, 4, 5, 6>,
                    6,
                    NumDataCompPerTile[0],
                    1,
                    false,
                    false,
                    false,
                    false,
                    InTileLoad,
                    threadsPerTensorTile,
                    vgprPerTensorTile>(in_0_grid_block_desc,
                                       make_multi_index(w0 - 1,
                                                        0,
                                                        h0,
                                                        0,
                                                        indata_slice_origin_idx[I0],
                                                        indata_slice_origin_idx[I1],
                                                        indata_slice_origin_idx[I2]));

                return make_tuple(pre_cluster_buf, pre_blockwise_copy);
            }
        };

        auto next_border_block_trait = [&]() {
            // input data blockwise copy
            if constexpr(InEnableLds)
            {
                auto in_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<InDataType*>(p_shared),
                    BlockwiseConv_0::SharedMemTrait::in_block_space_size_aligned);
                using InBlockTransferThreadClusterArrangeOrder = Sequence<0, 1, 2>;
                using InBlockTransferAccessOrder               = Sequence<0, 1, 2>;
                constexpr index_t InBlockTransferVectorDim     = 2;

                if constexpr(EnableAsync)
                {
                    auto in_blockwise_copy = ThreadGroupTensorSliceTransferAsync<
                        ThisThreadBlockGrid,
                        Sequence<HPerBlockIn[0], WPerBlockIn[0], CPerBlock::At(0)>,
                        InBlockTransferThreadClusterLengths,
                        InBlockTransferThreadClusterArrangeOrder,
                        InDataType,
                        InDataType,
                        decltype(in_0_grid_block_desc),
                        decltype(in_0_block_desc),
                        InBlockTransferVectorDim,
                        InBlockTransferVectorDim,
                        InBlockTransferDstScalarPerVector,
                        false,
                        true>(
                        in_0_grid_block_desc,
                        make_multi_index(h_block_data_idx_on_grid, w_block_data_idx_on_grid, 0),
                        in_0_block_desc,
                        make_multi_index(0, 0, 0));
                    return make_tuple(in_block_buf, in_blockwise_copy);
                }
                else
                {
                    auto in_blockwise_copy = ThreadGroupTensorSliceTransfer_v4r1<
                        ThisThreadBlockGrid,
                        InElementwiseOperation,
                        ck::tensor_operation::element_wise::PassThrough,
                        InMemoryDataOperationEnum::Set,
                        Sequence<HPerBlockIn[0], WPerBlockIn[0], CPerBlock::At(0)>,
                        InBlockTransferThreadClusterLengths,
                        InBlockTransferThreadClusterArrangeOrder,
                        InDataType,
                        InDataType,
                        decltype(in_0_grid_block_desc),
                        decltype(in_0_block_desc),
                        InBlockTransferAccessOrder,
                        InBlockTransferAccessOrder,
                        InBlockTransferVectorDim,
                        InBlockTransferVectorDim,
                        InBlockTransferSrcScalarPerVector,
                        InBlockTransferDstScalarPerVector,
                        1,
                        1,
                        false,
                        true,
                        NumConvCPrefetchStage>(
                        in_0_grid_block_desc,
                        make_multi_index(h_block_data_idx_on_grid, w_block_data_idx_on_grid, 0),
                        in_element_op,
                        in_0_block_desc,
                        make_multi_index(0, 0, 0),
                        ck::tensor_operation::element_wise::PassThrough{});
                    return make_tuple(in_block_buf, in_blockwise_copy);
                }
            }
            else
            {
                auto indata_slice_origin_idx =
                    wcnn_conv_3x3.template CalculateInDataThreadOriginDataIndex<InTileLoad>();
                auto h0 = (h_block_data_idx_on_grid + wave_idx[I0] * HPerWave) / HPerWcnn;
                auto w0 = (w_block_data_idx_on_grid + (wave_idx[I1] + 1) * WPerWave) / WPerWcnn;
                constexpr index_t threadsPerTensorTile = (WPerWcnn == 4) ? 2 : 4;
                constexpr index_t vgprPerTensorTile    = (HPerWcnn == 8) ? 2 : 1;

                auto next_blockwise_copy = ThreadwiseTensorSliceTransfer_v2<
                    InDataType,
                    InDataType,
                    decltype(in_0_grid_block_desc),
                    decltype(in_cluster_border_block_desc),
                    decltype(BlockwiseConv_0::GetInBorderWaveDescLength()),
                    Sequence<0, 1, 2, 3, 4, 5, 6>,
                    6,
                    NumDataCompPerTile[0],
                    1,
                    false,
                    false,
                    false,
                    false,
                    InTileLoad,
                    threadsPerTensorTile,
                    vgprPerTensorTile>(in_0_grid_block_desc,
                                       make_multi_index(w0,
                                                        0,
                                                        h0,
                                                        0,
                                                        indata_slice_origin_idx[I0],
                                                        indata_slice_origin_idx[I1],
                                                        indata_slice_origin_idx[I2]));

                return make_tuple(next_cluster_buf, next_blockwise_copy);
            }
        };

        auto wei_0_block_trait = [&]() {
            if constexpr(WeiEnableLds)
            {
                constexpr index_t NumTapPerCopy = 1;
                constexpr auto NumWeiCopy       = 9;

                using WeiBlockTransferThreadClusterArrangeOrder = Sequence<0, 1, 2>;
                using WeiBlockTransferAccessOrder               = Sequence<0, 1, 2>;
                constexpr index_t WeiBlockTransferVectorDim     = 2;

                auto wei_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<WeiDataType*>(p_shared) +
                        BlockwiseConv_0::SharedMemTrait::in_block_space_size_aligned,
                    BlockwiseConv_0::SharedMemTrait::wei_block_space_size_aligned);

                if constexpr(EnableAsync)
                {
                    using WeiThreadGroupTensorSliceTransfer = ThreadGroupTensorSliceTransferAsync<
                        ThisThreadBlockGrid,
                        Sequence<KPerBlock::At(0), NumTapPerCopy, CPerBlock::At(0)>,
                        WeiBlockTransferThreadClusterLengths,
                        WeiBlockTransferThreadClusterArrangeOrder,
                        WeiDataType,
                        WeiDataType,
                        decltype(wei_grid_desc[I0]),
                        decltype(wei_0_block_desc),
                        WeiBlockTransferVectorDim,
                        WeiBlockTransferVectorDim,
                        WeiBlockTransferSrcScalarPerVector,
                        false,
                        true>;

                    auto wei_blockwise_copy = generate_tuple(
                        [&](auto I) {
                            return WeiThreadGroupTensorSliceTransfer(
                                wei_grid_desc[I0],
                                make_multi_index(k0_block_data_idx_on_grid, I, 0),
                                wei_0_block_desc,
                                make_multi_index(0, wcnn_conv_3x3.GetWeight3RemapTable()[I], 0));
                        },
                        Number<NumWeiCopy>{});

                    return make_tuple(wei_block_buf, wei_blockwise_copy);
                }
                else
                {
                    using WeiThreadGroupTensorSliceTransfer = ThreadGroupTensorSliceTransfer_v4r1<
                        ThisThreadBlockGrid,
                        WeiElementwiseOperation,
                        ck::tensor_operation::element_wise::PassThrough,
                        InMemoryDataOperationEnum::Set,
                        Sequence<KPerBlock::At(0), NumTapPerCopy, CPerBlock::At(0)>,
                        WeiBlockTransferThreadClusterLengths,
                        WeiBlockTransferThreadClusterArrangeOrder,
                        WeiDataType,
                        WeiDataType,
                        decltype(wei_0_grid_block_desc),
                        decltype(wei_0_block_desc),
                        WeiBlockTransferAccessOrder,
                        WeiBlockTransferAccessOrder,
                        WeiBlockTransferVectorDim,
                        WeiBlockTransferVectorDim,
                        WeiBlockTransferSrcScalarPerVector,
                        WeiBlockTransferDstScalarPerVector,
                        1,
                        1,
                        false,
                        true,
                        NumConvCPrefetchStage>;

                    auto wei_blockwise_copy = generate_tuple(
                        [&](auto I) {
                            return WeiThreadGroupTensorSliceTransfer(
                                wei_grid_desc[I0],
                                make_multi_index(k0_block_data_idx_on_grid, I, 0),
                                wei_element_op,
                                wei_0_block_desc,
                                make_multi_index(0, wcnn_conv_3x3.GetWeight3RemapTable()[I], 0),
                                ck::tensor_operation::element_wise::PassThrough{});
                        },
                        Number<NumWeiCopy>{});
                    return make_tuple(wei_block_buf, wei_blockwise_copy);
                }
            }
            else
            {
                auto wei_slice_origin_idx =
                    wcnn_conv_3x3.template CalculateWeiDataThreadOriginDataIndex<WeiTileLoad>();
                auto k0 = (k0_block_data_idx_on_grid + wave_idx[I2] * KPerWave[0]) / KPerWcnn;

                constexpr index_t threadsPerSubWeiTile = 2;
                constexpr index_t vgprPerSubWeiTile    = 1;
                // Limitation: NumDim of Src and Dst descriptor should be identical
                using WeiThreadGroupTensorSliceTransfer = ThreadwiseTensorSliceTransfer_v2<
                    WeiDataType,
                    WeiDataType,
                    decltype(wei_0_grid_block_desc),
                    decltype(wei_0_block_desc),
                    decltype(BlockwiseConv_0::GetWeiWaveDescLength()),
                    Sequence<0, 1, 2, 3, 4, 5>,
                    5,
                    NumWeightCompPerTile_3x3,
                    1,
                    false,
                    false,
                    false,
                    false,
                    WeiTileLoad,
                    threadsPerSubWeiTile,
                    vgprPerSubWeiTile>;

                auto wei_blockwise_copy = generate_tuple(
                    [&](auto I) {
                        return WeiThreadGroupTensorSliceTransfer(
                            wei_0_grid_block_desc,
                            make_multi_index(k0,
                                             0,
                                             I * wcnn_conv_3x3.GetNumWeightTapPerWave() +
                                                 wei_slice_origin_idx[I0] *
                                                     wcnn_conv_3x3.GetWeightSecondTapMapTable()[I],
                                             wei_slice_origin_idx[I1],
                                             wei_slice_origin_idx[I2],
                                             wei_slice_origin_idx[I3]));
                    },
                    Number<NumWeightTap_3x3>{});

                if constexpr(EnableWaveGroup)
                {
                    return make_tuple(
                        make_static_buffer_v4<AddressSpaceEnum::Vgpr, WeiDataType>(
                            wei_0_block_desc.GetElementSpaceSize(),
                            static_cast<WeiDataType*>(p_lane_shared) +
                                laneSharedMemTrait.wei_block_space_offset / sizeof(WeiDataType)),
                        wei_blockwise_copy);
                }
                else
                {
                    return make_tuple(make_static_buffer_v2<AddressSpaceEnum::Vgpr, WeiDataType>(
                                          wei_0_block_desc.GetElementSpaceSize()),
                                      wei_blockwise_copy);
                }
            }
        };

        auto wei_1_block_trait = [&]() {
            if constexpr(WeiEnableLds)
            {
                constexpr index_t NumTapPerCopy = 1;
                constexpr auto NumWeiCopy       = 1;

                using WeiBlockTransferThreadClusterArrangeOrder = Sequence<0, 1, 2>;
                using WeiBlockTransferAccessOrder               = Sequence<0, 1, 2>;
                constexpr index_t WeiBlockTransferVectorDim     = 2;

                auto wei_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<WeiDataType*>(p_shared) +
                        BlockwiseConv_0::SharedMemTrait::
                            in_block_space_size_aligned, // Should reuse the 16~31 channel for 1st
                                                         // input
                    BlockwiseConv_1::SharedMemTrait::wei_block_space_size_aligned);

                if constexpr(EnableAsync)
                {
                    using WeiThreadGroupTensorSliceTransfer = ThreadGroupTensorSliceTransferAsync<
                        ThisThreadBlockGrid,
                        Sequence<KPerBlock::At(1), NumTapPerCopy, CPerBlock::At(1)>,
                        WeiBlockTransferThreadClusterLengths,
                        WeiBlockTransferThreadClusterArrangeOrder,
                        WeiDataType,
                        WeiDataType,
                        decltype(wei_grid_desc[I1]),
                        decltype(wei_1_block_desc),
                        WeiBlockTransferVectorDim,
                        WeiBlockTransferVectorDim,
                        WeiBlockTransferSrcScalarPerVector,
                        false,
                        true>;

                    auto wei_blockwise_copy = generate_tuple(
                        [&](auto I) {
                            return WeiThreadGroupTensorSliceTransfer(
                                wei_grid_desc[I1],
                                make_multi_index(k1_block_data_idx_on_grid, I, 0),
                                wei_1_block_desc,
                                make_multi_index(0, 0, 0));
                        },
                        Number<NumWeiCopy>{});

                    return make_tuple(wei_block_buf, wei_blockwise_copy);
                }
                else
                {
                    using WeiThreadGroupTensorSliceTransfer = ThreadGroupTensorSliceTransfer_v4r1<
                        ThisThreadBlockGrid,
                        WeiElementwiseOperation,
                        ck::tensor_operation::element_wise::PassThrough,
                        InMemoryDataOperationEnum::Set,
                        Sequence<KPerBlock::At(1), NumTapPerCopy, CPerBlock::At(1)>,
                        WeiBlockTransferThreadClusterLengths,
                        WeiBlockTransferThreadClusterArrangeOrder,
                        WeiDataType,
                        WeiDataType,
                        decltype(wei_1_grid_block_desc),
                        decltype(wei_1_block_desc),
                        WeiBlockTransferAccessOrder,
                        WeiBlockTransferAccessOrder,
                        WeiBlockTransferVectorDim,
                        WeiBlockTransferVectorDim,
                        WeiBlockTransferSrcScalarPerVector,
                        WeiBlockTransferDstScalarPerVector,
                        1,
                        1,
                        false,
                        true,
                        NumConvCPrefetchStage>;

                    auto wei_blockwise_copy = generate_tuple(
                        [&](auto I) {
                            return WeiThreadGroupTensorSliceTransfer(
                                wei_grid_desc[I1],
                                make_multi_index(k1_block_data_idx_on_grid, I, 0),
                                wei_element_op,
                                wei_1_block_desc,
                                make_multi_index(0, 0, 0),
                                ck::tensor_operation::element_wise::PassThrough{});
                        },
                        Number<NumWeiCopy>{});
                    return make_tuple(wei_block_buf, wei_blockwise_copy);
                }
            }
            else
            {
                auto wei_slice_origin_idx =
                    wcnn_conv_1x1.template CalculateWeiDataThreadOriginDataIndex<WeiTileLoad>();
                auto k0 = (k1_block_data_idx_on_grid + wave_idx[I2] * KPerWave) / KPerWcnn;

                constexpr index_t threadsPerSubWeiTile = 2;
                constexpr index_t vgprPerSubWeiTile    = 1;
                // Limitation: NumDim of Src and Dst descriptor should be identical
                using WeiThreadGroupTensorSliceTransfer = ThreadwiseTensorSliceTransfer_v2<
                    WeiDataType,
                    WeiDataType,
                    decltype(wei_1_grid_block_desc),
                    decltype(wei_1_block_desc),
                    decltype(BlockwiseConv_1::GetWeiWaveDescLength()),
                    Sequence<0, 1, 2, 3, 4, 5>,
                    5,
                    NumWeightCompPerTile_1x1,
                    1,
                    false,
                    false,
                    false,
                    false,
                    WeiTileLoad,
                    threadsPerSubWeiTile,
                    vgprPerSubWeiTile>;

                auto wei_blockwise_copy = generate_tuple(
                    [&](auto I) {
                        return WeiThreadGroupTensorSliceTransfer(
                            wei_1_grid_block_desc,
                            make_multi_index(k0,
                                             0,
                                             I * wcnn_conv_1x1.GetNumWeightTapPerWave() +
                                                 wei_slice_origin_idx[I0] *
                                                     wcnn_conv_1x1.GetWeightSecondTapMapTable()[I],
                                             wei_slice_origin_idx[I1],
                                             wei_slice_origin_idx[I2],
                                             wei_slice_origin_idx[I3]));
                    },
                    Number<NumWeightTap_1x1>{});

                if constexpr(EnableWaveGroup)
                {
                    return make_tuple(
                        make_static_buffer_v4<AddressSpaceEnum::Vgpr, WeiDataType>(
                            wei_1_block_desc.GetElementSpaceSize(),
                            static_cast<WeiDataType*>(p_lane_shared) +
                                laneSharedMemTrait.wei_block_space_offset / sizeof(WeiDataType)),
                        wei_blockwise_copy);
                }
                else
                {
                    return make_tuple(make_static_buffer_v2<AddressSpaceEnum::Vgpr, WeiDataType>(
                                          wei_1_block_desc.GetElementSpaceSize()),
                                      wei_blockwise_copy);
                }
            }
        };

        auto wei_2_block_trait = [&]() {
            if constexpr(WeiEnableLds)
            {
                constexpr index_t NumTapPerCopy = 1;
                constexpr auto NumWeiCopy       = 1;

                using WeiBlockTransferThreadClusterArrangeOrder = Sequence<0, 1, 2>;
                using WeiBlockTransferAccessOrder               = Sequence<0, 1, 2>;
                constexpr index_t WeiBlockTransferVectorDim     = 2;

                auto wei_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<WeiDataType*>(p_shared), // Be 0 as no input tensor in phase 2
                    BlockwiseConv_2::SharedMemTrait::wei_block_space_size_aligned);

                if constexpr(EnableAsync)
                {
                    using WeiThreadGroupTensorSliceTransfer = ThreadGroupTensorSliceTransferAsync<
                        ThisThreadBlockGrid,
                        Sequence<KPerBlock::At(2), NumTapPerCopy, CPerBlock::At(2)>,
                        WeiBlockTransferThreadClusterLengths,
                        WeiBlockTransferThreadClusterArrangeOrder,
                        WeiDataType,
                        WeiDataType,
                        decltype(wei_grid_desc[I2]),
                        decltype(wei_2_block_desc),
                        WeiBlockTransferVectorDim,
                        WeiBlockTransferVectorDim,
                        WeiBlockTransferSrcScalarPerVector,
                        false,
                        true>;

                    auto wei_blockwise_copy = generate_tuple(
                        [&](auto I) {
                            return WeiThreadGroupTensorSliceTransfer(
                                wei_grid_desc[I2],
                                make_multi_index(k2_block_data_idx_on_grid, I, 0),
                                wei_2_block_desc,
                                make_multi_index(0, 0, 0));
                        },
                        Number<NumWeiCopy>{});

                    return make_tuple(wei_block_buf, wei_blockwise_copy);
                }
                else
                {
                    using WeiThreadGroupTensorSliceTransfer = ThreadGroupTensorSliceTransfer_v4r1<
                        ThisThreadBlockGrid,
                        WeiElementwiseOperation,
                        ck::tensor_operation::element_wise::PassThrough,
                        InMemoryDataOperationEnum::Set,
                        Sequence<KPerBlock::At(2), NumTapPerCopy, CPerBlock::At(2)>,
                        WeiBlockTransferThreadClusterLengths,
                        WeiBlockTransferThreadClusterArrangeOrder,
                        WeiDataType,
                        WeiDataType,
                        decltype(wei_2_grid_block_desc),
                        decltype(wei_2_block_desc),
                        WeiBlockTransferAccessOrder,
                        WeiBlockTransferAccessOrder,
                        WeiBlockTransferVectorDim,
                        WeiBlockTransferVectorDim,
                        WeiBlockTransferSrcScalarPerVector,
                        WeiBlockTransferDstScalarPerVector,
                        1,
                        1,
                        false,
                        true,
                        NumConvCPrefetchStage>;

                    auto wei_blockwise_copy = generate_tuple(
                        [&](auto I) {
                            return WeiThreadGroupTensorSliceTransfer(
                                wei_grid_desc[I2],
                                make_multi_index(k2_block_data_idx_on_grid, I, 0),
                                wei_element_op,
                                wei_2_block_desc,
                                make_multi_index(0, 0, 0),
                                ck::tensor_operation::element_wise::PassThrough{});
                        },
                        Number<NumWeiCopy>{});
                    return make_tuple(wei_block_buf, wei_blockwise_copy);
                }
            }
            else
            {
                auto wei_slice_origin_idx =
                    wcnn_conv_1x1.template CalculateWeiDataThreadOriginDataIndex<WeiTileLoad>();
                auto k0 = (k2_block_data_idx_on_grid + wave_idx[I2] * KPerWave[2]) / KPerWcnn;

                constexpr index_t threadsPerSubWeiTile = 2;
                constexpr index_t vgprPerSubWeiTile    = 1;
                // Limitation: NumDim of Src and Dst descriptor should be identical
                using WeiThreadGroupTensorSliceTransfer = ThreadwiseTensorSliceTransfer_v2<
                    WeiDataType,
                    WeiDataType,
                    decltype(wei_2_grid_block_desc),
                    decltype(wei_2_block_desc),
                    decltype(BlockwiseConv_2::GetWeiWaveDescLength()),
                    Sequence<0, 1, 2, 3, 4, 5>,
                    5,
                    NumWeightCompPerTile_1x1,
                    1,
                    false,
                    false,
                    false,
                    false,
                    WeiTileLoad,
                    threadsPerSubWeiTile,
                    vgprPerSubWeiTile>;

                auto wei_blockwise_copy = generate_tuple(
                    [&](auto I) {
                        return WeiThreadGroupTensorSliceTransfer(
                            wei_2_grid_block_desc,
                            make_multi_index(k0,
                                             0,
                                             I * wcnn_conv_1x1.GetNumWeightTapPerWave() +
                                                 wei_slice_origin_idx[I0] *
                                                     wcnn_conv_1x1.GetWeightSecondTapMapTable()[I],
                                             wei_slice_origin_idx[I1],
                                             wei_slice_origin_idx[I2],
                                             wei_slice_origin_idx[I3]));
                    },
                    Number<NumWeightTap_1x1>{});

                if constexpr(EnableWaveGroup)
                {
                    return make_tuple(
                        make_static_buffer_v4<AddressSpaceEnum::Vgpr, WeiDataType>(
                            wei_2_block_desc.GetElementSpaceSize(),
                            static_cast<WeiDataType*>(p_lane_shared) +
                                laneSharedMemTrait.wei_block_space_offset / sizeof(WeiDataType)),
                        wei_blockwise_copy);
                }
                else
                {
                    return make_tuple(make_static_buffer_v2<AddressSpaceEnum::Vgpr, WeiDataType>(
                                          wei_2_block_desc.GetElementSpaceSize()),
                                      wei_blockwise_copy);
                }
            }
        };

        auto ds_0_block_trait = [&]() {
            if constexpr(IsPassthroughBlockwiseOp<0>())
            {
                return make_tuple(Tuple<>{}, Tuple<>{}, Tuple<>{});
            }
            else if constexpr(DsEnableLds)
            {
                auto ds_array_block_buf = generate_tuple(
                    [&](auto i) {
                        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType_0>>;
                        static_assert(sizeof(AccDataType) % sizeof(DDataType) == 0);
                        constexpr auto DDataSizeScale = sizeof(AccDataType) / sizeof(DDataType);
                        return make_dynamic_buffer<AddressSpaceEnum::Lds>(
                            static_cast<DDataType*>(p_shared) +
                                BlockwiseConv_0::SharedMemTrait::ds_block_space_offset[i] *
                                    DDataSizeScale,
                            BlockwiseConv_0::SharedMemTrait::ds_block_space_size_aligned[i] *
                                DDataSizeScale);
                    },
                    Number<NumDTensor_0>{});

                auto ds_blockwise_copy = generate_tuple(
                    [&](auto i) {
                        return MakeSignleDThreadgroupTransfer<0>(i,
                                                                 ds_0_grid_block_desc[Number<i>{}],
                                                                 ds_0_block_desc[Number<i>{}],
                                                                 h_out_block_data_idx_on_grid,
                                                                 w_out_block_data_idx_on_grid,
                                                                 k0_block_data_idx_on_grid);
                    },
                    Number<NumDTensor_0>{});
                return make_tuple(ds_array_block_buf, ds_blockwise_copy, ds_0_block_desc);
            }
            else
            {
                auto thread_origin_data_idx = BlockwiseConv_0::CalculateThreadOriginDataIndex();
                auto ds_wave_desc_length    = BlockwiseConv_0::GetDsWaveDescLength();
                auto ds_wave_desc           = BlockwiseConv_0::GetDsWaveDesc();
                auto ds_array_block_buf     = generate_tuple(
                    [&](auto i) {
                        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType_1>>;
                        if constexpr(EnableWaveGroup)
                        {
                            return make_static_buffer_v5<
                                    AddressSpaceEnum::Vgpr,
                                    DDataType,
                                    laneSharedMemTrait.ds_0_block_space_offset[i] / sizeof(DDataType)>(
                                ds_wave_desc[Number<i>{}].GetElementSpaceSize(),
                                static_cast<DDataType*>(p_lane_shared));
                        }
                        else
                        {
                            return make_static_buffer_v2<AddressSpaceEnum::Vgpr, DDataType>(
                                ds_wave_desc[Number<i>{}].GetElementSpaceSize());
                        }
                    },
                    Number<NumDTensor_0>{});

                // Limitation: NumDim of Src and Dst descriptor should be identical
                auto ds_blockwise_copy = generate_tuple(
                    [&](auto i) {
                        using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout_0>>;
                        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType_0>>;
                        return MakeSignleDThreadwiseTransfer<0>(DLayout{},
                                                                DDataType{},
                                                                ds_0_grid_block_desc[Number<i>{}],
                                                                ds_wave_desc[Number<i>{}],
                                                                thread_origin_data_idx[Number<i>{}],
                                                                ds_wave_desc_length[Number<i>{}],
                                                                h_out_block_data_idx_on_grid,
                                                                w_out_block_data_idx_on_grid,
                                                                k0_block_data_idx_on_grid);
                    },
                    Number<NumDTensor_0>{});
                return make_tuple(ds_array_block_buf, ds_blockwise_copy, ds_wave_desc);
            }
        };

        auto ds_1_block_trait = [&]() {
            if constexpr(IsPassthroughBlockwiseOp<1>())
            {
                return make_tuple(Tuple<>{}, Tuple<>{}, Tuple<>{});
            }
            else if constexpr(DsEnableLds)
            {
                auto ds_array_block_buf = generate_tuple(
                    [&](auto i) {
                        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType_1>>;
                        static_assert(sizeof(AccDataType) % sizeof(DDataType) == 0);
                        constexpr auto DDataSizeScale = sizeof(AccDataType) / sizeof(DDataType);
                        return make_dynamic_buffer<AddressSpaceEnum::Lds>(
                            static_cast<DDataType*>(p_shared) +
                                BlockwiseConv_1::SharedMemTrait::ds_block_space_offset[i] *
                                    DDataSizeScale,
                            BlockwiseConv_1::SharedMemTrait::ds_block_space_size_aligned[i] *
                                DDataSizeScale);
                    },
                    Number<NumDTensor_1>{});

                auto ds_blockwise_copy = generate_tuple(
                    [&](auto i) {
                        return MakeSignleDThreadgroupTransfer<1>(i,
                                                                 ds_1_grid_block_desc[Number<i>{}],
                                                                 ds_1_block_desc[Number<i>{}],
                                                                 h_out_block_data_idx_on_grid,
                                                                 w_out_block_data_idx_on_grid,
                                                                 k1_block_data_idx_on_grid);
                    },
                    Number<NumDTensor_1>{});
                return make_tuple(ds_array_block_buf, ds_blockwise_copy, ds_1_block_desc);
            }
            else
            {
                auto thread_origin_data_idx = BlockwiseConv_1::CalculateThreadOriginDataIndex();
                auto ds_wave_desc_length    = BlockwiseConv_1::GetDsWaveDescLength();
                auto ds_wave_desc           = BlockwiseConv_1::GetDsWaveDesc();
                auto ds_array_block_buf     = generate_tuple(
                    [&](auto i) {
                        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType_1>>;
                        if constexpr(EnableWaveGroup)
                        {
                            return make_static_buffer_v5<
                                    AddressSpaceEnum::Vgpr,
                                    DDataType,
                                    laneSharedMemTrait.ds_1_block_space_offset[i] / sizeof(DDataType)>(
                                ds_wave_desc[Number<i>{}].GetElementSpaceSize(),
                                static_cast<DDataType*>(p_lane_shared));
                        }
                        else
                        {
                            return make_static_buffer_v2<AddressSpaceEnum::Vgpr, DDataType>(
                                ds_wave_desc[Number<i>{}].GetElementSpaceSize());
                        }
                    },
                    Number<NumDTensor_1>{});

                // Limitation: NumDim of Src and Dst descriptor should be identical
                auto ds_blockwise_copy = generate_tuple(
                    [&](auto i) {
                        using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout_1>>;
                        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType_1>>;
                        return MakeSignleDThreadwiseTransfer<1>(DLayout{},
                                                                DDataType{},
                                                                ds_1_grid_block_desc[Number<i>{}],
                                                                ds_wave_desc[Number<i>{}],
                                                                thread_origin_data_idx[Number<i>{}],
                                                                ds_wave_desc_length[Number<i>{}],
                                                                h_out_block_data_idx_on_grid,
                                                                w_out_block_data_idx_on_grid,
                                                                k1_block_data_idx_on_grid);
                    },
                    Number<NumDTensor_1>{});
                return make_tuple(ds_array_block_buf, ds_blockwise_copy, ds_wave_desc);
            }
        };

        auto ds_2_block_trait = [&]() {
            if constexpr(IsPassthroughBlockwiseOp<2>())
            {
                return make_tuple(Tuple<>{}, Tuple<>{}, Tuple<>{});
            }
            else if constexpr(DsEnableLds)
            {
                auto ds_array_block_buf = generate_tuple(
                    [&](auto i) {
                        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType_2>>;
                        static_assert(sizeof(AccDataType) % sizeof(DDataType) == 0);
                        constexpr auto DDataSizeScale = sizeof(AccDataType) / sizeof(DDataType);
                        return make_dynamic_buffer<AddressSpaceEnum::Lds>(
                            static_cast<DDataType*>(p_shared) +
                                BlockwiseConv_2::SharedMemTrait::ds_block_space_offset[i] *
                                    DDataSizeScale,
                            BlockwiseConv_2::SharedMemTrait::ds_block_space_size_aligned[i] *
                                DDataSizeScale);
                    },
                    Number<NumDTensor_2>{});

                auto ds_blockwise_copy = generate_tuple(
                    [&](auto i) {
                        return MakeSignleDThreadgroupTransfer<2>(i,
                                                                 ds_2_grid_block_desc[Number<i>{}],
                                                                 ds_2_block_desc[Number<i>{}],
                                                                 h_out_block_data_idx_on_grid,
                                                                 w_out_block_data_idx_on_grid,
                                                                 k2_block_data_idx_on_grid);
                    },
                    Number<NumDTensor_2>{});
                return make_tuple(ds_array_block_buf, ds_blockwise_copy, ds_2_block_desc);
            }
            else
            {
                auto thread_origin_data_idx = BlockwiseConv_2::CalculateThreadOriginDataIndex();
                auto ds_wave_desc_length    = BlockwiseConv_2::GetDsWaveDescLength();
                auto ds_wave_desc           = BlockwiseConv_2::GetDsWaveDesc();
                auto ds_array_block_buf     = generate_tuple(
                    [&](auto i) {
                        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType_2>>;
                        if constexpr(EnableWaveGroup)
                        {
                            return make_static_buffer_v5<
                                    AddressSpaceEnum::Vgpr,
                                    DDataType,
                                    laneSharedMemTrait.ds_2_block_space_offset[i] / sizeof(DDataType)>(
                                ds_wave_desc[Number<i>{}].GetElementSpaceSize(),
                                static_cast<DDataType*>(p_lane_shared));
                        }
                        else
                        {
                            return make_static_buffer_v2<AddressSpaceEnum::Vgpr, DDataType>(
                                ds_wave_desc[Number<i>{}].GetElementSpaceSize());
                        }
                    },
                    Number<NumDTensor_2>{});

                // Limitation: NumDim of Src and Dst descriptor should be identical
                auto ds_blockwise_copy = generate_tuple(
                    [&](auto i) {
                        using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout_2>>;
                        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType_2>>;
                        return MakeSignleDThreadwiseTransfer<2>(DLayout{},
                                                                DDataType{},
                                                                ds_2_grid_block_desc[Number<i>{}],
                                                                ds_wave_desc[Number<i>{}],
                                                                thread_origin_data_idx[Number<i>{}],
                                                                ds_wave_desc_length[Number<i>{}],
                                                                h_out_block_data_idx_on_grid,
                                                                w_out_block_data_idx_on_grid,
                                                                k2_block_data_idx_on_grid);
                    },
                    Number<NumDTensor_2>{});
                return make_tuple(ds_array_block_buf, ds_blockwise_copy, ds_wave_desc);
            }
        };

        /*******************************************************************************/
        // CONV
        auto acc_0_blockwise_op = MakeAccBlockwiseOp<0>(acc_element_op[I0]);
        auto acc_1_blockwise_op = MakeAccBlockwiseOp<1>(acc_element_op[I1]);
        auto acc_2_blockwise_op = MakeAccBlockwiseOp<2>(acc_element_op[I2]);
        BlockwiseConv_0 blockwise_conv_0(acc_0_blockwise_op[I0], acc_0_blockwise_op[I1]);
        BlockwiseConv_1 blockwise_conv_1(acc_1_blockwise_op[I0], acc_1_blockwise_op[I1]);
        BlockwiseConv_2 blockwise_conv_2(acc_2_blockwise_op[I0], acc_2_blockwise_op[I1]);

        // Prepare Register for Accum
        auto pOutData_0 = static_cast<BlockOutDataType*>(p_lane_shared) +
                          laneSharedMemTrait.out_0_block_space_offset / sizeof(BlockOutDataType);
        auto pOutData_1 = static_cast<BlockOutDataType*>(p_lane_shared) +
                          laneSharedMemTrait.out_1_block_space_offset / sizeof(BlockOutDataType);
        auto pOutData_2 = static_cast<BlockOutDataType*>(p_lane_shared) +
                          laneSharedMemTrait.out_2_block_space_offset / sizeof(BlockOutDataType);
        auto pAccData_0 = static_cast<AccDataType*>(p_lane_shared) +
                          laneSharedMemTrait.acc_0_block_space_offset / sizeof(AccDataType);
        auto pAccData_1 = static_cast<AccDataType*>(p_lane_shared) +
                          laneSharedMemTrait.acc_1_block_space_offset / sizeof(AccDataType);
        auto pAccData_2 = static_cast<AccDataType*>(p_lane_shared) +
                          laneSharedMemTrait.acc_2_block_space_offset / sizeof(AccDataType);

        auto acc_0_thread_buf =
            blockwise_conv_0.template MakeAccumThreadBuffer<HasMainBlockLoop>(pAccData_0);
        auto acc_1_thread_buf =
            blockwise_conv_1.template MakeAccumThreadBuffer<HasMainBlockLoop>(pAccData_1);
        auto acc_2_thread_buf =
            blockwise_conv_2.template MakeAccumThreadBuffer<HasMainBlockLoop>(pAccData_2);

        auto out_0_thread_buf = blockwise_conv_0.MakeOutThreadBuffer(pOutData_0);
        auto out_1_thread_buf = blockwise_conv_1.MakeOutThreadBuffer(pOutData_1);
        auto out_2_thread_buf = blockwise_conv_2.MakeOutThreadBuffer(pOutData_2);
        /*******************************************************************************/
        // Shift Per CPerBlock
        constexpr auto in_0_block_slice_copy_step  = BlockwiseConv_0::MakeInBlockSliceCopyStep();
        constexpr auto in_1_block_slice_copy_step  = BlockwiseConv_1::MakeInBlockSliceCopyStep();
        constexpr auto wei_0_block_slice_copy_step = BlockwiseConv_0::MakeWeiBlockSliceCopyStep();
        constexpr auto wei_1_block_slice_copy_step = BlockwiseConv_1::MakeWeiBlockSliceCopyStep();
        constexpr auto wei_2_block_slice_copy_step = BlockwiseConv_2::MakeWeiBlockSliceCopyStep();

        // Gridwise conv pipeline
        const auto C_0                = in_grid_desc[I0].GetLength(I2);
        const index_t C0BlockMainLoop = __builtin_amdgcn_readfirstlane(C_0 / CPerBlock::At(0));
        const auto C_1                = in_grid_desc[I1].GetLength(I2);
        const index_t C1BlockMainLoop = __builtin_amdgcn_readfirstlane(C_1 / CPerBlock::At(1));
        const auto C_2                = in_grid_desc[I2].GetLength(I2);
        const index_t C2BlockMainLoop = __builtin_amdgcn_readfirstlane(C_2 / CPerBlock::At(2));

        auto in_0_block_buf      = in_0_block_trait()[I0];
        auto in_0_blockwise_copy = in_0_block_trait()[I1];

        auto in_1_block_buf      = in_1_block_trait()[I0];
        auto in_1_blockwise_copy = in_1_block_trait()[I1];

        auto wei_0_block_buf      = wei_0_block_trait()[I0];
        auto wei_0_blockwise_copy = wei_0_block_trait()[I1];

        auto wei_1_block_buf      = wei_1_block_trait()[I0];
        auto wei_1_blockwise_copy = wei_1_block_trait()[I1];

        auto wei_2_block_buf      = wei_2_block_trait()[I0];
        auto wei_2_blockwise_copy = wei_2_block_trait()[I1];

        auto ds_0_block_buf       = ds_0_block_trait()[I0];
        auto ds_0_blockwise_copy  = ds_0_block_trait()[I1];
        auto ds_0_copy_block_desc = ds_0_block_trait()[I2];

        auto ds_1_block_buf       = ds_1_block_trait()[I0];
        auto ds_1_blockwise_copy  = ds_1_block_trait()[I1];
        auto ds_1_copy_block_desc = ds_1_block_trait()[I2];

        auto ds_2_block_buf       = ds_2_block_trait()[I0];
        auto ds_2_blockwise_copy  = ds_2_block_trait()[I1];
        auto ds_2_copy_block_desc = ds_2_block_trait()[I2];

        auto pre_blockwise_copy  = pre_border_block_trait()[I1];
        auto next_blockwise_copy = next_border_block_trait()[I1];

        __shared__ WavegroupSemaphore<WaveIdOutput> sema_output;

        if constexpr(EnableWaveGroup4)
        {
            sema_output.init();
        }
#ifdef CK_USE_AMD_NAMED_BARRIER_ASM
        NamedBarrier<2, 4> barrier_output;
#else
        __shared__ NamedBarrier<4> barrier_output;
#endif
        constexpr auto OutputWaveId = EnableWaveGroup4 ? WaveIdOutput : WaveIdRun;
        if constexpr(EnableWaveGroup && AccEnableLdsInternal)
        {
            if(get_wave_id_in_wavegroup() == OutputWaveId)
            {
                barrier_output.init();
                barrier_output.join();
            }
        }

        const std::array<index_t, 3> CBlockMainLoop{
            C0BlockMainLoop, C1BlockMainLoop, C2BlockMainLoop};
        auto ins_grid_block_desc = ck::make_tuple(in_0_grid_block_desc, in_1_grid_block_desc);
        auto ins_block_desc      = ck::make_tuple(in_0_block_desc, in_1_block_desc);
        auto ins_block_buf       = ck::make_tuple(in_0_block_buf, in_1_block_buf);
        auto ins_blockwise_copy  = ck::make_tuple(in_0_blockwise_copy, in_1_blockwise_copy);
        auto ins_block_slice_copy_step =
            ck::make_tuple(in_0_block_slice_copy_step, in_1_block_slice_copy_step);
        auto ins_grid_buf = ck::make_tuple(in_0_grid_buf, in_1_grid_buf, in_2_grid_buf);
        auto weis_grid_block_desc =
            ck::make_tuple(wei_0_grid_block_desc, wei_1_grid_block_desc, wei_2_grid_block_desc);
        auto weis_block_desc = ck::make_tuple(wei_0_block_desc, wei_1_block_desc, wei_2_block_desc);
        auto weis_blockwise_copy =
            ck::make_tuple(wei_0_blockwise_copy, wei_1_blockwise_copy, wei_2_blockwise_copy);
        auto weis_grid_buf  = ck::make_tuple(wei_0_grid_buf, wei_1_grid_buf, wei_2_grid_buf);
        auto weis_block_buf = ck::make_tuple(wei_0_block_buf, wei_1_block_buf, wei_2_block_buf);
        auto weis_block_slice_copy_step = ck::make_tuple(
            wei_0_block_slice_copy_step, wei_1_block_slice_copy_step, wei_2_block_slice_copy_step);
        auto dss_grid_buf = ck::make_tuple(ds_0_grid_buf, ds_1_grid_buf, ds_2_grid_buf);
        auto dss_grid_block_desc =
            ck::make_tuple(ds_0_grid_block_desc, ds_1_grid_block_desc, ds_2_grid_block_desc);
        auto dss_copy_block_desc =
            ck::make_tuple(ds_0_copy_block_desc, ds_1_copy_block_desc, ds_2_copy_block_desc);
        auto dss_blockwise_copy =
            ck::make_tuple(ds_0_blockwise_copy, ds_1_blockwise_copy, ds_2_blockwise_copy);
        auto dss_block_buf = ck::make_tuple(ds_0_block_buf, ds_1_block_buf, ds_2_block_buf);
        auto blockwise_conv_x3 =
            ck::make_tuple(blockwise_conv_0, blockwise_conv_1, blockwise_conv_2);

        if constexpr(EnableWaveGroup)
        {
            GridwiseFasternet50Pipe::template Run<HasMainBlockLoop>(ins_grid_block_desc,
                                                                    ins_block_desc,
                                                                    in_cluster_border_block_desc,
                                                                    ins_blockwise_copy,
                                                                    ins_grid_buf,
                                                                    ins_block_buf,
                                                                    pre_block_buf,
                                                                    pre_cluster_buf,
                                                                    pre_blockwise_copy,
                                                                    next_block_buf,
                                                                    next_cluster_buf,
                                                                    next_blockwise_copy,
                                                                    ins_block_slice_copy_step,
                                                                    weis_grid_block_desc,
                                                                    weis_block_desc,
                                                                    weis_blockwise_copy,
                                                                    weis_grid_buf,
                                                                    weis_block_buf,
                                                                    weis_block_slice_copy_step,
                                                                    dss_grid_block_desc,
                                                                    dss_copy_block_desc,
                                                                    dss_blockwise_copy,
                                                                    dss_grid_buf,
                                                                    dss_block_buf,
                                                                    blockwise_conv_x3,
                                                                    acc_0_thread_buf,
                                                                    acc_1_thread_buf,
                                                                    acc_2_thread_buf,
                                                                    out_0_thread_buf,
                                                                    out_1_thread_buf,
                                                                    out_2_thread_buf,
                                                                    CBlockMainLoop);
        }
        else
        {
            GridwiseFasternet50Pipe::template Run<HasMainBlockLoop>(ins_grid_block_desc,
                                                                    ins_block_desc,
                                                                    ins_blockwise_copy,
                                                                    ins_grid_buf,
                                                                    ins_block_buf,
                                                                    ins_block_slice_copy_step,
                                                                    weis_grid_block_desc,
                                                                    weis_block_desc,
                                                                    weis_blockwise_copy,
                                                                    weis_grid_buf,
                                                                    weis_block_buf,
                                                                    weis_block_slice_copy_step,
                                                                    dss_grid_block_desc,
                                                                    dss_copy_block_desc,
                                                                    dss_blockwise_copy,
                                                                    dss_grid_buf,
                                                                    dss_block_buf,
                                                                    blockwise_conv_x3,
                                                                    acc_0_thread_buf,
                                                                    acc_1_thread_buf,
                                                                    acc_2_thread_buf,
                                                                    out_0_thread_buf,
                                                                    out_1_thread_buf,
                                                                    out_2_thread_buf,
                                                                    CBlockMainLoop);
        }

        // sync post-run wave and output wave
        if constexpr(EnableWaveGroup4)
        {
            if(get_wave_id_in_wavegroup() == WaveIdPostRun)
            {
                sema_output.template signal<0>();
            }
            if(get_wave_id_in_wavegroup() == WaveIdOutput)
            {
                sema_output.template wait<0>();
            }
        }

        /*******************************************************************************/
        // Store accum buffer
        if((EnableWaveGroup == false) || (get_wave_id_in_wavegroup() == OutputWaveId))
        {
#if defined(CASCADE_1X_OUT)
            StoreOutTensorData<0>(ds_grid_desc,
                                  p_ds_grid,
                                  e_grid_desc,
                                  p_e_grid,
                                  out_0_thread_buf,
                                  blockwise_conv_0,
                                  acc_0_blockwise_op[I2],
                                  p_shared,
                                  barrier_output,
                                  h_out_block_data_idx_on_grid,
                                  w_out_block_data_idx_on_grid,
                                  k0_block_data_idx_on_grid);
#elif defined(CASCADE_2X_OUT)
            StoreOutTensorData<1>(ds_grid_desc,
                                  p_ds_grid,
                                  e_grid_desc,
                                  p_e_grid,
                                  out_1_thread_buf,
                                  blockwise_conv_1,
                                  acc_1_blockwise_op[I2],
                                  p_shared,
                                  barrier_output,
                                  h_out_block_data_idx_on_grid,
                                  w_out_block_data_idx_on_grid,
                                  k1_block_data_idx_on_grid);
#else
            // Store the result: AccElementOp(None/fma/sba/uba) + NextElementOp(cvt_tensor)
            StoreOutTensorData<2>(ds_grid_desc,
                                  p_ds_grid,
                                  e_grid_desc,
                                  p_e_grid,
                                  out_2_thread_buf,
                                  blockwise_conv_2,
                                  acc_2_blockwise_op[I2],
                                  p_shared,
                                  barrier_output,
                                  h_out_block_data_idx_on_grid,
                                  w_out_block_data_idx_on_grid,
                                  k2_block_data_idx_on_grid);
#endif
        }
#else
        ignore = p_in_grid;
        ignore = p_wei_grid;
        ignore = p_ds_grid;
        ignore = p_e_grid;
        ignore = p_shared;
        ignore = p_lane_shared;
        ignore = in_element_op;
        ignore = wei_element_op;
        ignore = acc_element_op;
        ignore = in_grid_desc;
        ignore = wei_grid_desc;
        ignore = ds_grid_desc;
        ignore = e_grid_desc;
        ignore = block_2_ctile_map;
#endif
    }
};

} // namespace ck
