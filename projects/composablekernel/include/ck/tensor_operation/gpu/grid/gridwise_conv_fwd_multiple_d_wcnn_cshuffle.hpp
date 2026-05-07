// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_conv_wcnn_pipeline.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_conv_wcnn.hpp"
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
          typename ComputePtrOffsetOfBatch,
          bool HasMainBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_grouped_conv_fwd_wcnn(const InDataType* __restrict__ p_in_grid,
                                 const WeiDataType* __restrict__ p_wei_grid,
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
                                 const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch)
{
#if defined(__gfx13__)
    // offset base pointer for each work-group
    static constexpr index_t NumDTensor = DsGridDesc::Size();
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);

    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);
    const long_index_t in_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
    const long_index_t wei_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
    const long_index_t e_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx));
    const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);

    __shared__ char p_shared[GridwiseOp::BlockwiseConv::SharedMemTrait::lds_size];

    DsPointer p_ds_grid_grp;
    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = p_ds_grid[i] + ds_batch_offset[i]; });

    bool canUseInTiledLoad = [&]() {
        if constexpr(GridwiseOp::hasInTileLoad)
        {
            constexpr auto I0 = Number<0>{};
            constexpr auto I1 = Number<1>{};
            constexpr auto I2 = Number<2>{};

            const auto block_work_idx =
                block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));
            const index_t h_block_data_idx_on_grid =
                __builtin_amdgcn_readfirstlane(block_work_idx[I1] * GridwiseOp::TileHPerBlock);
            const index_t w_block_data_idx_on_grid =
                __builtin_amdgcn_readfirstlane(block_work_idx[I2] * GridwiseOp::TileWPerBlock);
            const index_t h_block_end = in_grid_desc.GetLength(I0);
            const index_t w_block_end = in_grid_desc.GetLength(I1);

            if(h_block_data_idx_on_grid == 0 || w_block_data_idx_on_grid == 0 ||
               h_block_data_idx_on_grid >= (h_block_end - GridwiseOp::TileHPerBlock) ||
               w_block_data_idx_on_grid >= (w_block_end - GridwiseOp::TileWPerBlock))
            {
                return false;
            }
        }
        return true;
    }();

    bool canUseWeiTiledLoad = [&]() {
        if constexpr(GridwiseOp::hasWeiTileLoad)
        {
            constexpr auto I0 = Number<0>{};
            const auto block_work_idx =
                block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));
            const index_t k_block_data_idx_on_grid = __builtin_amdgcn_readfirstlane(
                (block_work_idx[I0] + 1) * GridwiseOp::TileKPerBlock - 1);
            // const index_t k_block_data_idx_on_grid = wei_grid_desc.GetLength(I0);

            auto src_slice_origin_idx = make_multi_index(k_block_data_idx_on_grid, I0, 0);
            auto src_coord_           = make_tensor_coordinate(wei_grid_desc, src_slice_origin_idx);

            const bool is_src_valid = coordinate_has_valid_offset_assuming_visible_index_is_valid(
                wei_grid_desc, src_coord_);

            if(!is_src_valid)
            {
                return false;
            }
        }
        return true;
    }();

    if(canUseInTiledLoad && canUseWeiTiledLoad)
    {
        GridwiseOp::template Run<HasMainBlockLoop, 0x3 /* canUseInTileLoad + canUseWeiTileLoad */>(
            p_in_grid + in_batch_offset,
            p_wei_grid + wei_batch_offset,
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
    }
    else if(canUseInTiledLoad)
    {
        GridwiseOp::template Run<HasMainBlockLoop, 0x1 /* canUseInTileLoad */>(
            p_in_grid + in_batch_offset,
            p_wei_grid + wei_batch_offset,
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
    }
    else if(canUseWeiTiledLoad)
    {
        GridwiseOp::template Run<HasMainBlockLoop, 0x2 /* canUseWeiTileLoad */>(
            p_in_grid + in_batch_offset,
            p_wei_grid + wei_batch_offset,
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
    }
    else
    {
        GridwiseOp::template Run<HasMainBlockLoop>(p_in_grid + in_batch_offset,
                                                   p_wei_grid + wei_batch_offset,
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
    }
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
    ignore = compute_ptr_offset_of_batch;
    ignore = block_2_ctile_map;
#endif
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wzero-length-array"
template <typename GridwiseOp,
          typename InDataType,
          typename WeiDataType,
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
          typename ComputePtrOffsetOfBatch,
          bool HasMainBlockLoop,
          index_t ClusterSize>
#if defined(__gfx13__)
__attribute__((amdgpu_spatial_cluster_kernel))
__attribute__((cluster_dims(ClusterSize))) __exp_amd_wavegroup_kernel(4, 32, 256, 1, 1)
#endif
    __global__ void kernel_grouped_conv_fwd_wcnn_wavegroup256_spatial_cluster(
        const InDataType* __restrict__ p_in_grid,
        const WeiDataType* __restrict__ p_wei_grid,
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
        const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch)
{
#if defined(__gfx13__)
    // offset base pointer for each work-group
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t in_batch_offset = amd_wave_read_first_lane(
        static_cast<int64_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const long_index_t wei_batch_offset = amd_wave_read_first_lane(
        static_cast<int64_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const long_index_t e_batch_offset = amd_wave_read_first_lane(
        static_cast<int64_t>(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx)));

    const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);
    DsPointer p_ds_grid_grp;
    static constexpr index_t NumDTensor = DsGridDesc::Size();
    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = p_ds_grid[i] + ds_batch_offset[i]; });

    uint32_t waveGroupIdInCluster = __builtin_amdgcn_wavegroup_id_in_cluster();
    if(waveGroupIdInCluster == 0)
    {
        __builtin_amdgcn_spatial_cluster_set_chain_start(true);
    }
    else if(waveGroupIdInCluster == (ClusterSize * 4 - 1)) // cluster_size * wavegroupIncluster - 1
    {
        __builtin_amdgcn_spatial_cluster_set_chain_end(true);
    }

    constexpr auto laneSharedMemTrait =
        GridwiseOp::template GetLaneSharedMemTrait<HasMainBlockLoop>();

    static constexpr index_t lds_size =
        math::max(GridwiseOp::BlockwiseConv::SharedMemTrait::lds_size, 4);
    static constexpr index_t lane_shared_size = math::max(laneSharedMemTrait.lane_shared_size, 4);
    __shared__ char p_shared[lds_size];
    static __exp_amd_laneshared__ char p_lane_shared[lane_shared_size];

    static_assert(laneSharedMemTrait.lane_shared_size <= 512 * 4, "");

    GridwiseOp::template Run<HasMainBlockLoop>(p_in_grid + in_batch_offset,
                                               p_wei_grid + wei_batch_offset,
                                               p_ds_grid_grp,
                                               p_e_grid + e_batch_offset,
                                               p_shared,
                                               p_lane_shared,
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
    ignore = compute_ptr_offset_of_batch;
    ignore = block_2_ctile_map;
#endif
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wzero-length-array"
template <typename GridwiseOp,
          typename InDataType,
          typename WeiDataType,
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
          typename ComputePtrOffsetOfBatch,
          bool HasMainBlockLoop,
          index_t ClusterSize>
#if defined(__gfx13__)
__attribute__((amdgpu_spatial_cluster_kernel))
__attribute__((cluster_dims(ClusterSize))) __exp_amd_wavegroup_kernel(4, 32, 512, 1, 1)
#endif
    __global__ void kernel_grouped_conv_fwd_wcnn_wavegroup512_spatial_cluster(
        const InDataType* __restrict__ p_in_grid,
        const WeiDataType* __restrict__ p_wei_grid,
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
        const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch)
{
#if defined(__gfx13__)
    // offset base pointer for each work-group
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t in_batch_offset = amd_wave_read_first_lane(
        static_cast<int64_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const long_index_t wei_batch_offset = amd_wave_read_first_lane(
        static_cast<int64_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const long_index_t e_batch_offset = amd_wave_read_first_lane(
        static_cast<int64_t>(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx)));

    const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);
    DsPointer p_ds_grid_grp;
    static constexpr index_t NumDTensor = DsGridDesc::Size();
    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = p_ds_grid[i] + ds_batch_offset[i]; });

    uint32_t waveGroupIdInCluster = __builtin_amdgcn_wavegroup_id_in_cluster();
    if(waveGroupIdInCluster == 0)
    {
        __builtin_amdgcn_spatial_cluster_set_chain_start(true);
    }
    else if(waveGroupIdInCluster == (ClusterSize * 4 - 1))
    {
        __builtin_amdgcn_spatial_cluster_set_chain_end(true);
    }

    constexpr auto laneSharedMemTrait =
        GridwiseOp::template GetLaneSharedMemTrait<HasMainBlockLoop>();

    static constexpr index_t lds_size =
        math::max(GridwiseOp::BlockwiseConv::SharedMemTrait::lds_size, 4);
    static constexpr index_t lane_shared_size = math::max(laneSharedMemTrait.lane_shared_size, 4);
    __shared__ char p_shared[lds_size];
    static __exp_amd_laneshared__ char p_lane_shared[lane_shared_size];

    static_assert(laneSharedMemTrait.lane_shared_size <= 512 * 4, "");

    GridwiseOp::template Run<HasMainBlockLoop>(p_in_grid + in_batch_offset,
                                               p_wei_grid + wei_batch_offset,
                                               p_ds_grid_grp,
                                               p_e_grid + e_batch_offset,
                                               p_shared,
                                               p_lane_shared,
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
    ignore = compute_ptr_offset_of_batch;
    ignore = block_2_ctile_map;
#endif
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wzero-length-array"
template <typename GridwiseOp,
          typename InDataType,
          typename WeiDataType,
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
          typename ComputePtrOffsetOfBatch,
          bool HasMainBlockLoop>
#if defined(__gfx13__)
__exp_amd_wavegroup_kernel(4, 32, 256, 1, 1)
#endif
    __global__ void kernel_grouped_conv_fwd_wcnn_wavegroup256(
        const InDataType* __restrict__ p_in_grid,
        const WeiDataType* __restrict__ p_wei_grid,
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
        const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch)
{
#if defined(__gfx13__)
    // offset base pointer for each work-group
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t in_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
    const long_index_t wei_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
    const long_index_t e_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx));

    const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);
    DsPointer p_ds_grid_grp;
    static constexpr index_t NumDTensor = DsGridDesc::Size();
    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = p_ds_grid[i] + ds_batch_offset[i]; });

    constexpr auto laneSharedMemTrait =
        GridwiseOp::template GetLaneSharedMemTrait<HasMainBlockLoop>();

    static constexpr index_t lds_size =
        math::max(GridwiseOp::BlockwiseConv::SharedMemTrait::lds_size, 4);
    static constexpr index_t lane_shared_size = math::max(laneSharedMemTrait.lane_shared_size, 4);
    __shared__ char p_shared[lds_size];
    static __exp_amd_laneshared__ char p_lane_shared[lane_shared_size];

    static_assert(laneSharedMemTrait.lane_shared_size <= 512 * 4, "");

    GridwiseOp::template Run<HasMainBlockLoop>(p_in_grid + in_batch_offset,
                                               p_wei_grid + wei_batch_offset,
                                               p_ds_grid_grp,
                                               p_e_grid + e_batch_offset,
                                               p_shared,
                                               p_lane_shared,
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
    ignore = compute_ptr_offset_of_batch;
    ignore = block_2_ctile_map;
#endif
}

template <typename GridwiseOp,
          typename InDataType,
          typename WeiDataType,
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
          typename ComputePtrOffsetOfBatch,
          bool HasMainBlockLoop>
#if defined(__gfx13__)
__exp_amd_wavegroup_kernel(4, 32, 512, 1, 1)
#endif
    __global__ void kernel_grouped_conv_fwd_wcnn_wavegroup512(
        const InDataType* __restrict__ p_in_grid,
        const WeiDataType* __restrict__ p_wei_grid,
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
        const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch)
{
#if defined(__gfx13__)
    // offset base pointer for each work-group
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t in_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
    const long_index_t wei_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
    const long_index_t e_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx));

    const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);
    DsPointer p_ds_grid_grp;
    static constexpr index_t NumDTensor = DsGridDesc::Size();
    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = p_ds_grid[i] + ds_batch_offset[i]; });

    constexpr auto laneSharedMemTrait =
        GridwiseOp::template GetLaneSharedMemTrait<HasMainBlockLoop>();

    __shared__ char p_shared[GridwiseOp::BlockwiseConv::SharedMemTrait::lds_size + 4];
    static __exp_amd_laneshared__ char p_lane_shared[laneSharedMemTrait.lane_shared_size];

    static_assert(laneSharedMemTrait.lane_shared_size <= 512 * 4, "");

    GridwiseOp::template Run<HasMainBlockLoop>(p_in_grid + in_batch_offset,
                                               p_wei_grid + wei_batch_offset,
                                               p_ds_grid_grp,
                                               p_e_grid + e_batch_offset,
                                               p_shared,
                                               p_lane_shared,
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
    ignore = compute_ptr_offset_of_batch;
    ignore = block_2_ctile_map;
#endif
}
#pragma clang diagnostic pop

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
          index_t CPerBlock,
          index_t KPerBlock,
          index_t HRepeat,
          index_t WRepeat,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
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
struct GridwiseConvMultipleD_Wcnn_CShuffle
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
    static constexpr bool hasInTileLoad    = InTileLoad;
    static constexpr bool hasWeiTileLoad   = WeiTileLoad;
    static constexpr index_t NumDTensor    = DsDataType::Size();

    static_assert((EnableWaveGroup == false) || (BlockSize % (WaveSize * 4) == 0), "");
    static_assert((EnableWaveGroup == true) || (EnableSpatialCluster == false), "");

    using ThisThreadBlockGrid =
        typename std::conditional<EnableWaveGroup,
                                  ThisThreadBlockWaveGroup<BlockSize, WaveSize, NumWaveGroup>,
                                  ThisThreadBlock<BlockSize>>::type;

    static constexpr index_t YX = FilterSize * FilterSize;
    using NumberYX              = Number<FilterSize * FilterSize>;

    static constexpr index_t WaveFilterSize = (FilterSize == 2) ? 1 : FilterSize;
    static constexpr auto wcnn_conv         = WcnnConv<WeiDataType,
                                                       InDataType,
                                                       AccDataType,
                                                       HPerWcnn,
                                                       WPerWcnn,
                                                       WaveFilterSize,
                                                       DilationX,
                                                       DilationY,
                                                       1,
                                                       EnableWaveGroup,
                                                       false,
                                                       false>{};

    using GridwiseConvPipe =
        remove_cvref_t<decltype(GridwiseConvPipeline_Selector<NumConvCPrefetchStage,
                                                              InEnableLds,
                                                              WeiEnableLds,
                                                              DsEnableLds,
                                                              EnableAsync,
                                                              EnableWaveGroup,
                                                              EnableSpatialCluster>())>;

    static constexpr index_t CPerWcnn                = wcnn_conv.GetNumInputChannels();
    static constexpr index_t KPerWcnn                = wcnn_conv.GetNumOutputChannels();
    static constexpr index_t NumWeightTap            = wcnn_conv.GetNumWeightTap();
    static constexpr index_t NumSubTilesPerWeightTap = wcnn_conv.GetNumSubTilesPerWeightTap();
    static constexpr index_t NumWeightCompPerTile    = wcnn_conv.GetNumWeightCompPerTile();
    static constexpr index_t NumSubTilePerImage      = wcnn_conv.GetNumSubTilesPerImageTile();
    static constexpr index_t NumDataCompPerTile      = wcnn_conv.GetNumDataCompPerTile();

    static constexpr index_t DataTileHeight = 4;
    static constexpr index_t H_Pad          = (FilterSize == 3) ? DataTileHeight : 0;
    static constexpr index_t W_Pad = ((FilterSize == 3) && !EnableSpatialCluster) ? WPerWcnn : 0;
    static constexpr index_t TileWPerBlock = WPerBlock;
    static constexpr index_t TileHPerBlock = HPerBlock;
    static constexpr index_t TileKPerBlock = KPerBlock;
    static constexpr index_t HPerBlockIn   = HPerBlock + H_Pad * 2;
    static constexpr index_t WPerBlockIn   = WPerBlock + W_Pad * 2;
    static constexpr index_t HPerWave      = HRepeat * HPerWcnn;
    static constexpr index_t WPerWave      = WRepeat * WPerWcnn;
    static constexpr index_t CPerWave      = CPerBlock;
    static constexpr index_t KPerWave      = KPerBlock;
    static constexpr index_t HPerWaveIn    = HPerWave + H_Pad * 2;
    static constexpr index_t WPerWaveIn    = WPerWave + W_Pad * 2;

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
    static auto __device__ __host__
    MakeAccBlockwiseOp(const AccElementwiseOperation& acc_element_op)
    {
        using AccElementwiseOperation_   = remove_cvref_t<AccElementwiseOperation>;
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

        if constexpr(NumDTensor == 0)
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
        else if constexpr(NumDTensor == 2)
        {
            using D0Layout = remove_cvref_t<tuple_element_t<0, DsLayout>>;
            using D1Layout = remove_cvref_t<tuple_element_t<1, DsLayout>>;
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
        else if constexpr(NumDTensor == 1)
        {
            using D0Layout = remove_cvref_t<tuple_element_t<0, DsLayout>>;
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
    using AccBlockwiseOp                  = decltype(MakeAccBlockwiseOp(AccElementwiseOperation{}));
    using AccElementwiseOperationInternal = remove_cvref_t<tuple_element_t<2, AccBlockwiseOp>>;
    using AccBlockwiseOperation           = remove_cvref_t<tuple_element_t<0, AccBlockwiseOp>>;
    using AccBlockwiseNextOperation       = remove_cvref_t<tuple_element_t<1, AccBlockwiseOp>>;

    __host__ __device__ static constexpr bool IsPassthroughBlockwiseOp()
    {
        return AccBlockwiseOperation::IsFma == false && AccBlockwiseOperation::IsSuba == false;
    }
    static constexpr bool SupportDirectOutput =
        (IsPassthroughBlockwiseOp() == false) || (NumDTensor == 0);
    static constexpr bool AccEnableLdsInternal = AccEnableLds || (SupportDirectOutput == false);
    // Describe the layout of InData in block level (LDS or VGPR)
    __host__ __device__ static constexpr auto MakeInBlockDescriptor()
    {
        if constexpr(InEnableLds)
        {
            // H x W x C Per Block
            return make_naive_tensor_descriptor_packed(
                make_tuple(Number<HPerBlockIn>{}, Number<WPerBlockIn>{}, Number<CPerBlock>{}));
        }
        else
        {
            // W0 x C0 x H0 x H1 x H2 x W1 x C1
            return make_naive_tensor_descriptor_packed(make_tuple(Number<WPerWaveIn / WPerWcnn>{},
                                                                  Number<CPerWave / CPerWcnn>{},
                                                                  Number<HPerWaveIn / HPerWcnn>{},
                                                                  Number<NumSubTilePerImage>{},
                                                                  I1,
                                                                  I1,
                                                                  Number<NumDataCompPerTile>{}));
        }
    }

    template <bool isClusterBorder>
    __host__ __device__ static constexpr auto MakeInClusterBorderBlockDescriptor()
    {
        if constexpr(isClusterBorder)
        {
            // W0 x C0 x H0 x H1 x H2 x W1 x C1
            return make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                                  Number<CPerWave / CPerWcnn>{},
                                                                  Number<HPerWaveIn / HPerWcnn>{},
                                                                  Number<NumSubTilePerImage>{},
                                                                  I1,
                                                                  I1,
                                                                  Number<NumDataCompPerTile>{}));
        }
        else
        {
            return make_naive_tensor_descriptor_packed(make_tuple(Number<CPerWave / CPerWcnn>{},
                                                                  Number<HPerWaveIn / HPerWcnn>{},
                                                                  Number<NumSubTilePerImage>{}));
        }
    }

    // Describe the layout of WeiData in block level (LDS or VGPR)
    __host__ __device__ static constexpr auto MakeWeiBlockDescriptor()
    {
        if constexpr(WeiEnableLds)
        {
            // K x YX x C per block
            return make_naive_tensor_descriptor_packed(make_tuple(
                Number<KPerBlock>{}, Number<FilterSize * FilterSize>{}, Number<CPerBlock>{}));
        }
        else
        {
            // K0 x C0 x YX x K1 x C1 x C2
            constexpr index_t NumXY = (FilterSize == 3) ? NumWeightTap : FilterSize * FilterSize;
            return make_naive_tensor_descriptor_packed(make_tuple(Number<KPerWave / KPerWcnn>{},
                                                                  Number<CPerWave / CPerWcnn>{},
                                                                  Number<NumXY>{},
                                                                  I1,
                                                                  Number<NumSubTilesPerWeightTap>{},
                                                                  Number<NumWeightCompPerTile>{}));
        }
    }

    template <typename DLayout>
    __host__ __device__ static constexpr auto MakeSingleDBlockDescriptor()
    {
        // TODO: need a rule for per-wave layout when lds isn't enabled
        if constexpr(std::is_same_v<DLayout, tensor_layout::convolution::G_K>)
        {
            return make_naive_tensor_descriptor_packed(make_tuple(Number<KPerBlock>{}));
        }
        else
        {
            // H x W x K Per Block
            return make_naive_tensor_descriptor_packed(
                make_tuple(Number<HPerBlock>{}, Number<WPerBlock>{}, Number<KPerBlock>{}));
        }
    }

    // Ds desc for source in blockwise copy
    __host__ __device__ static constexpr auto MakeDsBlockDescriptor()
    {
        if constexpr(IsPassthroughBlockwiseOp())
        {
            return Tuple<>{};
        }
        else
        {
            return generate_tuple(
                [&](auto i) {
                    using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                    return MakeSingleDBlockDescriptor<DLayout>();
                },
                Number<NumDTensor>{});
        }
    }

    static constexpr auto ds_block_desc = MakeDsBlockDescriptor();

    using BlockOutDataType = typename std::
        conditional<AccBlockwiseNextOperation::cvt_to_tensor, EDataType, AccDataType>::type;

    using BlockwiseConv = BlockwiseConvWcnn<ThisThreadBlockGrid,
                                            WeiDataType,
                                            InDataType,
                                            DsDataType,
                                            AccDataType,
                                            BlockOutDataType,
                                            AccBlockwiseOperation,
                                            AccBlockwiseNextOperation,
                                            decltype(MakeWeiBlockDescriptor()),
                                            decltype(MakeInBlockDescriptor()),
                                            decltype(MakeInClusterBorderBlockDescriptor<false>()),
                                            decltype(MakeDsBlockDescriptor()),
                                            HPerBlock,
                                            WPerBlock,
                                            CPerBlock,
                                            KPerBlock,
                                            HRepeat,
                                            WRepeat,
                                            HPerWcnn,
                                            WPerWcnn,
                                            FilterSize,
                                            DilationX,
                                            DilationY,
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
        if constexpr(H_Pad > 0 || W_Pad > 0)
        {
            const auto Hi = in_grid_desc.GetLength(I0);
            const auto Wi = in_grid_desc.GetLength(I1);
            const auto Ci = in_grid_desc.GetLength(I2);

            return transform_tensor_descriptor(
                in_grid_desc,
                make_tuple(make_pad_transform(Hi, H_Pad, H_Pad),
                           make_pad_transform(Wi, W_Pad, W_Pad),
                           make_pass_through_transform(Ci)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
        }
        else
        {
            return in_grid_desc;
        }
    }
    // ck::Tuple<const D0DataType*, const D1DataType*, ...>
    static constexpr auto MakeDsGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                return static_cast<const DDataType*>(nullptr);
            },
            Number<NumDTensor>{});
    }

    using DsGridPointer = decltype(MakeDsGridPointer());

    template <typename OutTensorThreadBuffer, typename NamedBarrier, typename BlockWiseConv>
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
                    BlockwiseConv::SharedMemTrait::out_tensor_block_space_offset,
                BlockwiseConv::SharedMemTrait::out_tensor_block_space_size);

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

            constexpr index_t NumOutDTensor = IsPassthroughBlockwiseOp() ? NumDTensor : 0;

            // each block copy its data from LDS to global
            if constexpr(NumOutDTensor == 0)
            {
                // blockwise copy C from LDS to global
                auto out_tensor_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
                    ThisThreadBlockGrid,
                    AccElementwiseOperationInternal,
                    EGlobalMemoryDataOperation,
                    Sequence<BlockwiseConv::HPerBlockOut, BlockwiseConv::WPerBlockOut, KPerBlock>,
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
                    false>{out_tensor_block_desc,
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
                            p_ds_grid[i], ds_grid_desc[i].GetElementSpaceSize());
                    },
                    Number<NumDTensor>{});

                // tuple of reference to C/Ds tensor descriptors
                const auto c_ds_desc_refs = concat_tuple_of_reference(
                    tie(out_tensor_block_desc),
                    generate_tie([&](auto i) -> const auto& // return type should be reference
                                 { return ds_grid_desc[i]; },
                                 Number<NumDTensor>{}));

                // tuple of reference to C/Ds tensor buffers
                const auto c_ds_buf_refs = concat_tuple_of_reference(
                    tie(out_tensor_block_buf),
                    generate_tie([&](auto i) -> const auto& // return type should be reference
                                 { return ds_grid_buf[i]; },
                                 Number<NumDTensor>{}));

                // tuple of starting index of C/Ds blockwise copy
                const auto idx_c_ds_block_begin =
                    container_concat(make_tuple(make_multi_index(0, 0, 0)),
                                     generate_tuple(
                                         [&](auto) {
                                             return make_multi_index(h_block_data_idx_on_grid,
                                                                     w_block_data_idx_on_grid,
                                                                     k_block_data_idx_on_grid);
                                         },
                                         Number<NumDTensor>{}));

                auto out_tensor_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v7<
                    ThisThreadBlockGrid, // ThreadGroup
                    decltype(container_concat(make_tuple(BlockOutDataType{}),
                                              DsDataType{})), // SrcDatas
                    Tuple<EDataType>,                         // DstDatas
                    decltype(c_ds_desc_refs),                 // SrcDescs
                    decltype(tie(e_grid_desc)),               // DstDescs
                    AccElementwiseOperationInternal,          // ElementwiseOperation,
                    Sequence<static_cast<index_t>(EGlobalMemoryDataOperation)>, // DstInMemOp,
                    Sequence<BlockwiseConv::HPerBlockOut,
                             BlockwiseConv::WPerBlockOut,
                             KPerBlock>,             // SliceLengths
                    AccBlockTransferClusterLengths,  // ThreadClusterLengths
                    Sequence<0, 1, 2>,               // ThreadClusterArrangeOrder
                    Sequence<0, 1, 2>,               // DimAccessOrder
                    2,                               // VectorDim
                    AccBlockTransferScalarPerVector, // ScalarPerVector
                    sequence_merge_t<Sequence<true>,
                                     uniform_sequence_gen_t<
                                         NumDTensor,
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

    template <typename DLayout,
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

        auto h0 = (h_block_data_idx_on_grid + wave_idx[I0] * BlockwiseConv::HPerWaveOut) / HPerWcnn;
        auto w0 = (w_block_data_idx_on_grid + wave_idx[I1] * BlockwiseConv::WPerWaveOut) / WPerWcnn;
        auto k0 = (k_block_data_idx_on_grid + wave_idx[I2] * KPerWave) / KPerWcnn;
        // residual use input tensor layout
        auto c0 = (k_block_data_idx_on_grid + wave_idx[I2] * KPerWave) / CPerWcnn;

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
                static_assert(NumComp == NumDataCompPerTile);
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

    template <typename DIndex, typename DGridBlockDesc, typename DBlockDesc>
    __host__ __device__ static constexpr auto
    MakeSignleDThreadgroupTransfer(const DIndex&,
                                   const DGridBlockDesc& d_grid_block_desc,
                                   const DBlockDesc&,
                                   index_t h_block_data_idx_on_grid,
                                   index_t w_block_data_idx_on_grid,
                                   index_t k_block_data_idx_on_grid)
    {
        using DLayout   = remove_cvref_t<tuple_element_t<DIndex{}.value, DsLayout>>;
        using DDataType = remove_cvref_t<tuple_element_t<DIndex{}.value, DsDataType>>;
        using DBlockTransferThreadClusterLengths =
            tuple_element_t<DIndex{}.value, DsBlockTransferThreadClusterLengths>;
        constexpr auto DBlockTransferDstScalarPerVector =
            DsBlockTransferDstScalarPerVector{}[DIndex{}];
        constexpr auto DBlockTransferSrcScalarPerVector =
            DsBlockTransferSrcScalarPerVector{}[DIndex{}];

        if constexpr(std::is_same_v<DLayout, tensor_layout::convolution::G_K>)
        {
            using DBlockTransferThreadClusterArrangeOrder = Sequence<0>;
            using DBlockTransferAccessOrder               = Sequence<0>;

            constexpr index_t DBlockTransferVectorDim = 0;
            static_assert(DBlockDesc{}.GetLength(I0) == KPerBlock);
            if constexpr(EnableAsync)
            {
                return ThreadGroupTensorSliceTransferAsync<ThisThreadBlockGrid,
                                                           Sequence<KPerBlock>,
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
                    Sequence<KPerBlock>,
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
            static_assert(DBlockDesc{}.GetLength(I2) == KPerBlock);
            if constexpr(EnableAsync)
            {
                return ThreadGroupTensorSliceTransferAsync<
                    ThisThreadBlockGrid,
                    Sequence<HPerBlock, WPerBlock, KPerBlock>,
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
                    Sequence<HPerBlock, WPerBlock, KPerBlock>,
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
                                                            const DsGridDesc& ds_grid_desc,
                                                            const EGridDesc& e_grid_desc,
                                                            const Block2CTileMap& block_2_ctile_map)
    {
        static_assert(HPerBlock % (HRepeat * HPerWcnn) == 0, "");
        static_assert(WPerBlock % (WRepeat * WPerWcnn) == 0, "");

        const auto GetInProblemsize = [&]() {
            return make_tuple(
                in_grid_desc.GetLength(I0), in_grid_desc.GetLength(I1), in_grid_desc.GetLength(I2));
        };

        const auto GetWeiProblemsize = [&]() {
            return make_tuple(wei_grid_desc.GetLength(I0),
                              wei_grid_desc.GetLength(I1),
                              wei_grid_desc.GetLength(I2));
        };

        const auto GetOutSize = [&](auto i) {
            if constexpr(FilterSize == 2)
            {
                return Transposed ? i * 2 : i / 2;
            }
            else
            {
                return i;
            }
        };

        const auto Ho = GetOutSize(GetInProblemsize()[I0]);
        const auto Wo = GetOutSize(GetInProblemsize()[I1]);
        const auto H  = GetInProblemsize()[I0];
        const auto W  = GetInProblemsize()[I1];
        const auto C  = GetInProblemsize()[I2];
        const auto K  = GetWeiProblemsize()[I0];

        bool valid = true;
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            valid = valid && (Ho == ds_grid_desc[i].GetLength(I0));
            valid = valid && (Wo == ds_grid_desc[i].GetLength(I1));
            valid = valid && (K == ds_grid_desc[i].GetLength(I2));
        });
        if(!valid)
        {
            printf("GridwiseOp: D descriptor dimension check failure\n");
            return false;
        }

        if(!(Wo == e_grid_desc.GetLength(I1) && K == e_grid_desc.GetLength(I2)) ||
           !(C == GetWeiProblemsize()[I2]))
        {
            printf("Tensor: HWC = %d x %d x %d, Filter: KXYC = %d x {%d, %d} x %d, Out: HWK = %d "
                   "x %d x %d\n",
                   H,
                   W,
                   C,
                   K,
                   FilterSize,
                   FilterSize,
                   GetWeiProblemsize()[I2],
                   e_grid_desc.GetLength(I0),
                   e_grid_desc.GetLength(I1),
                   e_grid_desc.GetLength(I2));
            printf("GridwiseOp err: ProblemSize check\n");
            return false;
        }

        if(!(H % HPerBlock == 0 && W % WPerBlock == 0 && K % KPerBlock == 0 && C % CPerBlock == 0))
        {
            printf("GridwiseOp err: ProblemSize division\n");
            return false;
        }

        // check gridwise conv pipeline
        const auto num_c_loop = C / CPerBlock;

        if(!GridwiseConvPipe::IsSupported(num_c_loop))
        {
            printf("GridwiseOp err: Pipeline not support this c_loop\n");
            return false;
        }

        if(!block_2_ctile_map.CheckValidity(e_grid_desc))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        constexpr long_index_t TwoGB = (long_index_t{1} << 31);

        if(!(in_grid_desc.GetElementSpaceSize() * sizeof(InDataType) <= TwoGB &&
             wei_grid_desc.GetElementSpaceSize() * sizeof(WeiDataType) <= TwoGB &&
             e_grid_desc.GetElementSpaceSize() * sizeof(InDataType) <= TwoGB))
        {
            return false;
        }

        if(EnableSpatialCluster && (FilterSize == 2))
        {
            return false;
        }

#if 0
        // Tensor & transform debugging code
        BlockwiseConv blockwise_conv = {};
        auto out_thread_buf = blockwise_conv.MakeOutThreadBuffer(nullptr);
        const AccElementwiseOperationInternal acc_element_op;
        StoreOutTensorData(acc_grid_desc,
            nullptr,
            out_thread_buf,
            blockwise_conv,
            acc_element_op,
            nullptr,
            0,
            0,
            0);
#endif

        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainBlockLoop(index_t C)
    {
        const index_t num_loop = C / CPerBlock;

        return GridwiseConvPipe::CalculateHasMainLoop(num_loop);
    }

    static constexpr auto in_block_desc                = MakeInBlockDescriptor();
    static constexpr auto wei_block_desc               = MakeWeiBlockDescriptor();
    static constexpr auto in_cluster_border_block_desc = MakeInClusterBorderBlockDescriptor<true>();
    static constexpr auto in_wavegroup_border_block_desc =
        MakeInClusterBorderBlockDescriptor<false>();
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
        static constexpr index_t mem_count =
            BlockwiseConv::template GetLaneSharedMemCount<HasMainBlockLoop>();
        static constexpr index_t in_block_space_offset = 0;

        static constexpr index_t pre_in_block_space_offset =
            EnableSpatialCluster
                ? (in_block_space_offset +
                   BlockwiseConv::LaneSharedMemTrait::in_block_space_size_aligned * mem_count)
                : in_block_space_offset;

        static constexpr index_t pre_cluster_in_block_space_offset = pre_in_block_space_offset;

        static constexpr index_t pre_offset = math::max(
            pre_in_block_space_offset + in_border_block_space_size_aligned * mem_count,
            pre_cluster_in_block_space_offset + in_cluster_block_space_size_aligned * mem_count);

        static constexpr index_t next_in_block_space_offset =
            EnableSpatialCluster ? pre_offset : in_block_space_offset;

        static constexpr index_t next_cluster_in_block_space_offset = next_in_block_space_offset;

        static constexpr index_t next_offset = math::max(
            next_in_block_space_offset + in_border_block_space_size_aligned * mem_count,
            next_cluster_in_block_space_offset + in_cluster_block_space_size_aligned * mem_count);

        static constexpr index_t wei_block_space_offset =
            EnableSpatialCluster
                ? next_offset
                : (in_block_space_offset +
                   BlockwiseConv::LaneSharedMemTrait::in_block_space_size_aligned * mem_count);

        static constexpr index_t ds_base_offset =
            wei_block_space_offset +
            BlockwiseConv::LaneSharedMemTrait::wei_block_space_size_aligned * mem_count;
        static constexpr auto ds_block_space_offset = generate_tuple(
            [](auto i) {
                return BlockwiseConv::LaneSharedMemTrait::ds_block_space_offset[i] + ds_base_offset;
            },
            Number<NumDTensor>{});

        static constexpr auto acc_block_space_offset =
            ds_base_offset + BlockwiseConv::LaneSharedMemTrait::ds_block_space_size_aligned;

        static constexpr auto acc_block_space_size_aligned =
            HasMainBlockLoop ? BlockwiseConv::LaneSharedMemTrait::acc_block_space_aligned
                             : BlockwiseConv::LaneSharedMemTrait::acc_ring_block_space_aligned;
        static constexpr auto out_block_space_offset =
            acc_block_space_offset + acc_block_space_size_aligned;

        static constexpr auto lane_shared_size =
            EnableWaveGroup4 ? out_block_space_offset +
                                   BlockwiseConv::LaneSharedMemTrait::out_block_space_aligned
                             : acc_block_space_offset;
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
        return BlockToCTileMap_KSplit_M00_N0_M01Adapt<BlockwiseConv::HPerBlockOut,
                                                      BlockwiseConv::WPerBlockOut,
                                                      EGridDesc>(
            c_grid_desc_h_w_k, M01, c_grid_desc_h_w_k.GetLength(I2) / KPerBlock);
    }

    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(EGridDesc{}, 1, 1))>;

    template <bool HasMainBlockLoop,
              index_t CanUseTiledLoad = 0,
              typename Block2CTileMap = DefaultBlock2CTileMap>
    __device__ static void Run(const InDataType* __restrict__ p_in_grid,
                               const WeiDataType* __restrict__ p_wei_grid,
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
        const auto in_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_grid, in_grid_desc.GetElementSpaceSize());
        const auto wei_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_wei_grid, wei_grid_desc.GetElementSpaceSize());
        const auto ds_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_ds_grid[i], ds_grid_desc[i].GetElementSpaceSize());
            },
            Number<NumDTensor>{});
        const auto in_grid_block_desc =
            BlockwiseConv::MakeInGridBlockDescriptor(MakeInGridPadDescriptor(in_grid_desc));
        const auto wei_grid_block_desc = BlockwiseConv::MakeWeiGridBlockDescriptor(wei_grid_desc);

        const auto ds_grid_block_desc = BlockwiseConv::MakeDsGridBlockDescriptor(ds_grid_desc);

        constexpr bool UseInTileLoad = InTileLoad && ((CanUseTiledLoad & 0x1) == 0x1);

        /*******************************************************************************/
        // BlockIdx.x -> [BlockId.k, BlockId.h, BlockId.w]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));
        if(!block_2_ctile_map.ValidCTileIndex(block_work_idx,
                                              make_tuple(e_grid_desc.GetLength(I0),
                                                         e_grid_desc.GetLength(I1),
                                                         e_grid_desc.GetLength(I2))))
        {
            return;
        }

        // Store BlockId into SGPR
        const index_t k_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * KPerBlock);
        const index_t w_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I2] * WPerBlock);
        const index_t h_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * HPerBlock);

        const index_t w_out_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I2] * BlockwiseConv::WPerBlockOut);
        const index_t h_out_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * BlockwiseConv::HPerBlockOut);

        /*******************************************************************************/
        // BlockLevel, Tensor and filter ThreadMapping in WCNN Source buffer, As Destinaion of
        // BlockWise_Copy
        const auto C = in_grid_desc.GetLength(I2);

        auto wave_idx = GetWcnnWaveIdx<ThisThreadBlockGrid,
                                       HPerBlock,
                                       WPerBlock,
                                       HRepeat,
                                       WRepeat,
                                       HPerWcnn,
                                       WPerWcnn>();

        constexpr auto laneSharedMemTrait = GetLaneSharedMemTrait<HasMainBlockLoop>();

        auto in_block_trait = [&]() {
            // input data blockwise copy
            if constexpr(InEnableLds)
            {
                auto in_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<InDataType*>(p_shared),
                    BlockwiseConv::SharedMemTrait::in_block_space_size_aligned);
                using InBlockTransferThreadClusterArrangeOrder = Sequence<0, 1, 2>;
                using InBlockTransferAccessOrder               = Sequence<0, 1, 2>;
                constexpr index_t InBlockTransferVectorDim     = 2;

                if constexpr(EnableAsync)
                {
                    auto in_blockwise_copy = ThreadGroupTensorSliceTransferAsync<
                        ThisThreadBlockGrid,
                        Sequence<HPerBlockIn, WPerBlockIn, CPerBlock>,
                        InBlockTransferThreadClusterLengths,
                        InBlockTransferThreadClusterArrangeOrder,
                        InDataType,
                        InDataType,
                        decltype(in_grid_block_desc),
                        decltype(in_block_desc),
                        InBlockTransferVectorDim,
                        InBlockTransferVectorDim,
                        InBlockTransferDstScalarPerVector,
                        false,
                        true>(
                        in_grid_block_desc,
                        make_multi_index(h_block_data_idx_on_grid, w_block_data_idx_on_grid, 0),
                        in_block_desc,
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
                        Sequence<HPerBlockIn, WPerBlockIn, CPerBlock>,
                        InBlockTransferThreadClusterLengths,
                        InBlockTransferThreadClusterArrangeOrder,
                        InDataType,
                        InDataType,
                        decltype(in_grid_block_desc),
                        decltype(in_block_desc),
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
                        in_grid_block_desc,
                        make_multi_index(h_block_data_idx_on_grid, w_block_data_idx_on_grid, 0),
                        in_element_op,
                        in_block_desc,
                        make_multi_index(0, 0, 0),
                        ck::tensor_operation::element_wise::PassThrough{});
                    return make_tuple(in_block_buf, in_blockwise_copy);
                }
            }
            else
            {
                auto indata_slice_origin_idx =
                    wcnn_conv.template CalculateInDataThreadOriginDataIndex<UseInTileLoad>();
                auto h0 = (h_block_data_idx_on_grid + wave_idx[I0] * HPerWave) / HPerWcnn;
                auto w0 = (w_block_data_idx_on_grid + wave_idx[I1] * WPerWave) / WPerWcnn;
                constexpr index_t threadsPerTensorTile = (WPerWcnn == 4) ? 2 : 4;
                constexpr index_t vgprPerTensorTile    = (HPerWcnn == 8) ? 2 : 1;
                // Limitation: NumDim of Src and Dst descriptor should be identical
                auto in_blockwise_copy =
                    ThreadwiseTensorSliceTransfer_v2<InDataType,
                                                     InDataType,
                                                     decltype(in_grid_block_desc),
                                                     decltype(in_block_desc),
                                                     decltype(BlockwiseConv::GetInWaveDescLength()),
                                                     Sequence<0, 1, 2, 3, 4, 5, 6>,
                                                     6,
                                                     NumDataCompPerTile,
                                                     1,
                                                     false,
                                                     false,
                                                     false,
                                                     false,
                                                     UseInTileLoad,
                                                     threadsPerTensorTile,
                                                     vgprPerTensorTile>(
                        in_grid_block_desc,
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
                            in_block_desc.GetElementSpaceSize(),
                            static_cast<InDataType*>(p_lane_shared) +
                                laneSharedMemTrait.in_block_space_offset / sizeof(InDataType)),
                        in_blockwise_copy);
                }
                else
                {
                    return make_tuple(make_static_buffer_v2<AddressSpaceEnum::Vgpr, InDataType>(
                                          in_block_desc.GetElementSpaceSize()),
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
                    BlockwiseConv::SharedMemTrait::in_block_space_size_aligned);
                using InBlockTransferThreadClusterArrangeOrder = Sequence<0, 1, 2>;
                using InBlockTransferAccessOrder               = Sequence<0, 1, 2>;
                constexpr index_t InBlockTransferVectorDim     = 2;

                if constexpr(EnableAsync)
                {
                    auto in_blockwise_copy = ThreadGroupTensorSliceTransferAsync<
                        ThisThreadBlockGrid,
                        Sequence<HPerBlockIn, WPerBlockIn, CPerBlock>,
                        InBlockTransferThreadClusterLengths,
                        InBlockTransferThreadClusterArrangeOrder,
                        InDataType,
                        InDataType,
                        decltype(in_grid_block_desc),
                        decltype(in_block_desc),
                        InBlockTransferVectorDim,
                        InBlockTransferVectorDim,
                        InBlockTransferDstScalarPerVector,
                        false,
                        true>(
                        in_grid_block_desc,
                        make_multi_index(h_block_data_idx_on_grid, w_block_data_idx_on_grid, 0),
                        in_block_desc,
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
                        Sequence<HPerBlockIn, WPerBlockIn, CPerBlock>,
                        InBlockTransferThreadClusterLengths,
                        InBlockTransferThreadClusterArrangeOrder,
                        InDataType,
                        InDataType,
                        decltype(in_grid_block_desc),
                        decltype(in_block_desc),
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
                        in_grid_block_desc,
                        make_multi_index(h_block_data_idx_on_grid, w_block_data_idx_on_grid, 0),
                        in_element_op,
                        in_block_desc,
                        make_multi_index(0, 0, 0),
                        ck::tensor_operation::element_wise::PassThrough{});
                    return make_tuple(in_block_buf, in_blockwise_copy);
                }
            }
            else
            {
                auto indata_slice_origin_idx =
                    wcnn_conv.template CalculateInDataThreadOriginDataIndex<UseInTileLoad>();
                auto h0 = (h_block_data_idx_on_grid + wave_idx[I0] * HPerWave) / HPerWcnn;
                auto w0 = (w_block_data_idx_on_grid + wave_idx[I1] * WPerWave) / WPerWcnn;
                constexpr index_t threadsPerTensorTile = (WPerWcnn == 4) ? 2 : 4;
                constexpr index_t vgprPerTensorTile    = (HPerWcnn == 8) ? 2 : 1;

                // Limitation: NumDim of Src and Dst descriptor should be identical
                auto pre_blockwise_copy = ThreadwiseTensorSliceTransfer_v2<
                    InDataType,
                    InDataType,
                    decltype(in_grid_block_desc),
                    decltype(in_cluster_border_block_desc),
                    decltype(BlockwiseConv::GetInBorderWaveDescLength()),
                    Sequence<0, 1, 2, 3, 4, 5, 6>,
                    6,
                    NumDataCompPerTile,
                    1,
                    false,
                    false,
                    false,
                    false,
                    UseInTileLoad,
                    threadsPerTensorTile,
                    vgprPerTensorTile>(in_grid_block_desc,
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
                    BlockwiseConv::SharedMemTrait::in_block_space_size_aligned);
                using InBlockTransferThreadClusterArrangeOrder = Sequence<0, 1, 2>;
                using InBlockTransferAccessOrder               = Sequence<0, 1, 2>;
                constexpr index_t InBlockTransferVectorDim     = 2;

                if constexpr(EnableAsync)
                {
                    auto in_blockwise_copy = ThreadGroupTensorSliceTransferAsync<
                        ThisThreadBlockGrid,
                        Sequence<HPerBlockIn, WPerBlockIn, CPerBlock>,
                        InBlockTransferThreadClusterLengths,
                        InBlockTransferThreadClusterArrangeOrder,
                        InDataType,
                        InDataType,
                        decltype(in_grid_block_desc),
                        decltype(in_block_desc),
                        InBlockTransferVectorDim,
                        InBlockTransferVectorDim,
                        InBlockTransferDstScalarPerVector,
                        false,
                        true>(
                        in_grid_block_desc,
                        make_multi_index(h_block_data_idx_on_grid, w_block_data_idx_on_grid, 0),
                        in_block_desc,
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
                        Sequence<HPerBlockIn, WPerBlockIn, CPerBlock>,
                        InBlockTransferThreadClusterLengths,
                        InBlockTransferThreadClusterArrangeOrder,
                        InDataType,
                        InDataType,
                        decltype(in_grid_block_desc),
                        decltype(in_block_desc),
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
                        in_grid_block_desc,
                        make_multi_index(h_block_data_idx_on_grid, w_block_data_idx_on_grid, 0),
                        in_element_op,
                        in_block_desc,
                        make_multi_index(0, 0, 0),
                        ck::tensor_operation::element_wise::PassThrough{});
                    return make_tuple(in_block_buf, in_blockwise_copy);
                }
            }
            else
            {
                auto indata_slice_origin_idx =
                    wcnn_conv.template CalculateInDataThreadOriginDataIndex<UseInTileLoad>();
                auto h0 = (h_block_data_idx_on_grid + wave_idx[I0] * HPerWave) / HPerWcnn;
                auto w0 = (w_block_data_idx_on_grid + (wave_idx[I1] + 1) * WPerWave) / WPerWcnn;
                constexpr index_t threadsPerTensorTile = (WPerWcnn == 4) ? 2 : 4;
                constexpr index_t vgprPerTensorTile    = (HPerWcnn == 8) ? 2 : 1;

                auto next_blockwise_copy = ThreadwiseTensorSliceTransfer_v2<
                    InDataType,
                    InDataType,
                    decltype(in_grid_block_desc),
                    decltype(in_cluster_border_block_desc),
                    decltype(BlockwiseConv::GetInBorderWaveDescLength()),
                    Sequence<0, 1, 2, 3, 4, 5, 6>,
                    6,
                    NumDataCompPerTile,
                    1,
                    false,
                    false,
                    false,
                    false,
                    UseInTileLoad,
                    threadsPerTensorTile,
                    vgprPerTensorTile>(in_grid_block_desc,
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

        auto wei_block_trait = [&]() {
            if constexpr(WeiEnableLds)
            {
                auto wei_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<WeiDataType*>(p_shared) +
                        BlockwiseConv::SharedMemTrait::wei_block_space_offset,
                    BlockwiseConv::SharedMemTrait::wei_block_space_size_aligned);

                using WeiBlockTransferThreadClusterArrangeOrder = Sequence<0, 1, 2>;
                using WeiBlockTransferAccessOrder               = Sequence<0, 1, 2>;
                constexpr index_t WeiBlockTransferVectorDim     = 2;
                constexpr index_t NumTapPerCopy                 = (FilterSize == 3) ? 1 : YX;
                constexpr auto NumWeiCopy                       = (FilterSize == 3) ? YX : 1;
                if constexpr(EnableAsync)
                {
                    using WeiThreadGroupTensorSliceTransfer = ThreadGroupTensorSliceTransferAsync<
                        ThisThreadBlockGrid,
                        Sequence<KPerBlock, NumTapPerCopy, CPerBlock>,
                        WeiBlockTransferThreadClusterLengths,
                        WeiBlockTransferThreadClusterArrangeOrder,
                        WeiDataType,
                        WeiDataType,
                        decltype(wei_grid_desc),
                        decltype(wei_block_desc),
                        WeiBlockTransferVectorDim,
                        WeiBlockTransferVectorDim,
                        WeiBlockTransferSrcScalarPerVector,
                        false,
                        true>;

                    auto wei_blockwise_copy = generate_tuple(
                        [&](auto I) {
                            return WeiThreadGroupTensorSliceTransfer(
                                wei_grid_desc,
                                make_multi_index(k_block_data_idx_on_grid, I, 0),
                                wei_block_desc,
                                (FilterSize == 3)
                                    ? make_multi_index(0, wcnn_conv.GetWeight3RemapTable()[I], 0)
                                    : make_multi_index(0, 0, 0));
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
                        Sequence<KPerBlock, NumTapPerCopy, CPerBlock>,
                        WeiBlockTransferThreadClusterLengths,
                        WeiBlockTransferThreadClusterArrangeOrder,
                        WeiDataType,
                        WeiDataType,
                        decltype(wei_grid_block_desc),
                        decltype(wei_block_desc),
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
                                wei_grid_desc,
                                make_multi_index(k_block_data_idx_on_grid, I, 0),
                                wei_element_op,
                                wei_block_desc,
                                (FilterSize == 3)
                                    ? make_multi_index(0, wcnn_conv.GetWeight3RemapTable()[I], 0)
                                    : make_multi_index(0, 0, 0),
                                ck::tensor_operation::element_wise::PassThrough{});
                        },
                        Number<NumWeiCopy>{});
                    return make_tuple(wei_block_buf, wei_blockwise_copy);
                }
            }
            else
            {
                constexpr bool UseWeiTileLoad = WeiTileLoad && ((CanUseTiledLoad & 0x2) == 0x2);
                auto wei_slice_origin_idx =
                    wcnn_conv.template CalculateWeiDataThreadOriginDataIndex<UseWeiTileLoad>();
                auto k0 = (k_block_data_idx_on_grid + wave_idx[I2] * KPerWave) / KPerWcnn;

                constexpr index_t threadsPerSubWeiTile = 2;
                constexpr index_t vgprPerSubWeiTile    = 1;
                // Limitation: NumDim of Src and Dst descriptor should be identical
                using WeiThreadGroupTensorSliceTransfer = ThreadwiseTensorSliceTransfer_v2<
                    WeiDataType,
                    WeiDataType,
                    decltype(wei_grid_block_desc),
                    decltype(wei_block_desc),
                    decltype(BlockwiseConv::GetWeiWaveDescLength()),
                    Sequence<0, 1, 2, 3, 4, 5>,
                    5,
                    NumWeightCompPerTile,
                    1,
                    false,
                    false,
                    false,
                    false,
                    UseWeiTileLoad,
                    threadsPerSubWeiTile,
                    vgprPerSubWeiTile>;

                auto wei_blockwise_copy = generate_tuple(
                    [&](auto I) {
                        if constexpr(FilterSize != 3)
                        {
                            static_assert(NumWeightTap == 1, "");
                            if(FilterSize == 2)
                            {
                                // Use tap index in dim YX
                                return WeiThreadGroupTensorSliceTransfer(
                                    wei_grid_block_desc,
                                    make_multi_index(k0,
                                                     0,
                                                     wei_slice_origin_idx[I0],
                                                     wei_slice_origin_idx[I1],
                                                     wei_slice_origin_idx[I2],
                                                     wei_slice_origin_idx[I3]));
                            }
                            else
                            {
                                // Use tap index in dim C0
                                return WeiThreadGroupTensorSliceTransfer(
                                    wei_grid_block_desc,
                                    make_multi_index(k0,
                                                     wei_slice_origin_idx[I0],
                                                     0,
                                                     wei_slice_origin_idx[I1],
                                                     wei_slice_origin_idx[I2],
                                                     wei_slice_origin_idx[I3]));
                            }
                        }
                        else
                        {
                            return WeiThreadGroupTensorSliceTransfer(
                                wei_grid_block_desc,
                                make_multi_index(k0,
                                                 0,
                                                 I * wcnn_conv.GetNumWeightTapPerWave() +
                                                     wei_slice_origin_idx[I0] *
                                                         wcnn_conv.GetWeightSecondTapMapTable()[I],
                                                 wei_slice_origin_idx[I1],
                                                 wei_slice_origin_idx[I2],
                                                 wei_slice_origin_idx[I3]));
                        }
                    },
                    Number<NumWeightTap>{});

                if constexpr(EnableWaveGroup)
                {
                    return make_tuple(
                        make_static_buffer_v4<AddressSpaceEnum::Vgpr, WeiDataType>(
                            wei_block_desc.GetElementSpaceSize(),
                            static_cast<WeiDataType*>(p_lane_shared) +
                                laneSharedMemTrait.wei_block_space_offset / sizeof(WeiDataType)),
                        wei_blockwise_copy);
                }
                else
                {
                    return make_tuple(make_static_buffer_v2<AddressSpaceEnum::Vgpr, WeiDataType>(
                                          wei_block_desc.GetElementSpaceSize()),
                                      wei_blockwise_copy);
                }
            }
        };

        auto ds_block_trait = [&]() {
            if constexpr(IsPassthroughBlockwiseOp())
            {
                return make_tuple(Tuple<>{}, Tuple<>{}, Tuple<>{});
            }
            else if constexpr(DsEnableLds)
            {
                auto ds_array_block_buf = generate_tuple(
                    [&](auto i) {
                        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                        static_assert(sizeof(AccDataType) % sizeof(DDataType) == 0);
                        constexpr auto DDataSizeScale = sizeof(AccDataType) / sizeof(DDataType);
                        return make_dynamic_buffer<AddressSpaceEnum::Lds>(
                            static_cast<DDataType*>(p_shared) +
                                BlockwiseConv::SharedMemTrait::ds_block_space_offset[i] *
                                    DDataSizeScale,
                            BlockwiseConv::SharedMemTrait::ds_block_space_size_aligned[i] *
                                DDataSizeScale);
                    },
                    Number<NumDTensor>{});

                auto ds_blockwise_copy = generate_tuple(
                    [&](auto i) {
                        return MakeSignleDThreadgroupTransfer(i,
                                                              ds_grid_block_desc[Number<i>{}],
                                                              ds_block_desc[Number<i>{}],
                                                              h_out_block_data_idx_on_grid,
                                                              w_out_block_data_idx_on_grid,
                                                              k_block_data_idx_on_grid);
                    },
                    Number<NumDTensor>{});
                return make_tuple(ds_array_block_buf, ds_blockwise_copy, ds_block_desc);
            }
            else
            {
                auto thread_origin_data_idx = BlockwiseConv::CalculateThreadOriginDataIndex();
                auto ds_wave_desc_length    = BlockwiseConv::GetDsWaveDescLength();
                auto ds_wave_desc           = BlockwiseConv::GetDsWaveDesc();
                auto ds_array_block_buf     = generate_tuple(
                    [&](auto i) {
                        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                        if constexpr(EnableWaveGroup)
                        {
                            return make_static_buffer_v5<
                                    AddressSpaceEnum::Vgpr,
                                    DDataType,
                                    laneSharedMemTrait.ds_block_space_offset[i] / sizeof(DDataType)>(
                                ds_wave_desc[Number<i>{}].GetElementSpaceSize(),
                                static_cast<DDataType*>(p_lane_shared));
                        }
                        else
                        {
                            return make_static_buffer_v2<AddressSpaceEnum::Vgpr, DDataType>(
                                ds_wave_desc[Number<i>{}].GetElementSpaceSize());
                        }
                    },
                    Number<NumDTensor>{});

                // Limitation: NumDim of Src and Dst descriptor should be identical
                auto ds_blockwise_copy = generate_tuple(
                    [&](auto i) {
                        using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                        return MakeSignleDThreadwiseTransfer(DLayout{},
                                                             DDataType{},
                                                             ds_grid_block_desc[Number<i>{}],
                                                             ds_wave_desc[Number<i>{}],
                                                             thread_origin_data_idx[Number<i>{}],
                                                             ds_wave_desc_length[Number<i>{}],
                                                             h_out_block_data_idx_on_grid,
                                                             w_out_block_data_idx_on_grid,
                                                             k_block_data_idx_on_grid);
                    },
                    Number<NumDTensor>{});
                return make_tuple(ds_array_block_buf, ds_blockwise_copy, ds_wave_desc);
            }
        };

        /*******************************************************************************/
        // CONV
        auto acc_blockwise_op = MakeAccBlockwiseOp(acc_element_op);
        BlockwiseConv blockwise_conv(acc_blockwise_op[I0], acc_blockwise_op[I1]);

        // Prepare Register for Accum
        auto pOutData = static_cast<BlockOutDataType*>(p_lane_shared) +
                        laneSharedMemTrait.out_block_space_offset / sizeof(BlockOutDataType);
        auto pAccData = static_cast<AccDataType*>(p_lane_shared) +
                        laneSharedMemTrait.acc_block_space_offset / sizeof(AccDataType);
        auto acc_thread_buf =
            blockwise_conv.template MakeAccumThreadBuffer<HasMainBlockLoop>(pAccData);
        auto out_thread_buf = blockwise_conv.MakeOutThreadBuffer(pOutData);

        /*******************************************************************************/
        // Shift Per CPerBlock
        constexpr auto in_block_slice_copy_step  = BlockwiseConv::MakeInBlockSliceCopyStep();
        constexpr auto wei_block_slice_copy_step = BlockwiseConv::MakeWeiBlockSliceCopyStep();

        // Gridwise conv pipeline
        const index_t CBlockMainLoop = __builtin_amdgcn_readfirstlane(C / CPerBlock);

        auto in_block_buf      = in_block_trait()[I0];
        auto in_blockwise_copy = in_block_trait()[I1];

        auto wei_block_buf      = wei_block_trait()[I0];
        auto wei_blockwise_copy = wei_block_trait()[I1];

        auto ds_block_buf       = ds_block_trait()[I0];
        auto ds_blockwise_copy  = ds_block_trait()[I1];
        auto ds_copy_block_desc = ds_block_trait()[I2];

        auto pre_blockwise_copy  = pre_border_block_trait()[I1];
        auto next_blockwise_copy = next_border_block_trait()[I1];

#ifdef CK_USE_AMD_SEMAPHORE_ASM
        WavegroupSemaphore<WaveIdOutput, 1> sema_output;
#else
        __shared__ WavegroupSemaphore<WaveIdOutput> sema_output;
#endif
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

        if constexpr(EnableWaveGroup)
        {
            GridwiseConvPipe::template Run<HasMainBlockLoop>(in_grid_block_desc,
                                                             in_block_desc,
                                                             in_cluster_border_block_desc,
                                                             in_blockwise_copy,
                                                             in_grid_buf,
                                                             in_block_buf,
                                                             pre_block_buf,
                                                             pre_cluster_buf,
                                                             pre_blockwise_copy,
                                                             next_block_buf,
                                                             next_cluster_buf,
                                                             next_blockwise_copy,
                                                             in_block_slice_copy_step,
                                                             wei_grid_block_desc,
                                                             wei_block_desc,
                                                             wei_blockwise_copy,
                                                             wei_grid_buf,
                                                             wei_block_buf,
                                                             wei_block_slice_copy_step,
                                                             ds_grid_block_desc,
                                                             ds_copy_block_desc,
                                                             ds_blockwise_copy,
                                                             ds_grid_buf,
                                                             ds_block_buf,
                                                             blockwise_conv,
                                                             acc_thread_buf,
                                                             out_thread_buf,
                                                             CBlockMainLoop);
        }
        else
        {
            GridwiseConvPipe::template Run<HasMainBlockLoop>(in_grid_block_desc,
                                                             in_block_desc,
                                                             in_blockwise_copy,
                                                             in_grid_buf,
                                                             in_block_buf,
                                                             in_block_slice_copy_step,
                                                             wei_grid_block_desc,
                                                             wei_block_desc,
                                                             wei_blockwise_copy,
                                                             wei_grid_buf,
                                                             wei_block_buf,
                                                             wei_block_slice_copy_step,
                                                             ds_grid_block_desc,
                                                             ds_copy_block_desc,
                                                             ds_blockwise_copy,
                                                             ds_grid_buf,
                                                             ds_block_buf,
                                                             blockwise_conv,
                                                             acc_thread_buf,
                                                             out_thread_buf,
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
            // Store the result: AccElementOp(None/fma/sba/uba) + NextElementOp(cvt_tensor)
            StoreOutTensorData(ds_grid_desc,
                               p_ds_grid,
                               e_grid_desc,
                               p_e_grid,
                               out_thread_buf,
                               blockwise_conv,
                               acc_blockwise_op[I2],
                               p_shared,
                               barrier_output,
                               h_out_block_data_idx_on_grid,
                               w_out_block_data_idx_on_grid,
                               k_block_data_idx_on_grid);
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
