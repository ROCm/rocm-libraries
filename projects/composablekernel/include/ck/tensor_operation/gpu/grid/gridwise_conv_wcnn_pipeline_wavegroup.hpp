// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/utility/amd_semaphore.hpp"
#include "ck/utility/amd_named_barrier.hpp"

#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"

namespace ck {
// Gridwise convolution pipeline with wavegroup enabled
// wave 0: load data
// wave 1: do convolution
// NOTE: Prefetch doesn't supported when LDS is enabled.
template <index_t NumPrefetch,
          bool InDataEnableLds,
          bool WeiDataEnableLds,
          bool DsDataEnableLds,
          bool EnableAsync,
          bool EnableSpatialCluster>
struct GridwiseConvPipeline_v2
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename InDataGridDesc,
              typename InDataBlockDesc,
              typename InClusterBorderDesc,
              typename InDataBlockTransfer,
              typename InDataGridBuffer,
              typename InDataBlockBuffer,
              typename ExchangeDataBlockBuffer,
              typename ClusterBorderDataBlockBuffer,
              typename PreNextDataBlockTransfer,
              typename InDataBlockTransferStep,
              typename WeiDataGridDesc,
              typename WeiDataBlockDesc,
              typename WeiDataBlockTransfer,
              typename WeiDataGridBuffer,
              typename WeiDataBlockBuffer,
              typename WeiDataBlockTransferStep,
              typename DsDataGridDesc,
              typename DsDataBlockDesc,
              typename DsDataBlockTransfer,
              typename DsDataGridBuffer,
              typename DsDataBlockBuffer,
              typename BlockwiseConv,
              typename AccumThreadBuffer,
              typename OutThreadBuffer>
    __device__ static void Run(const InDataGridDesc& in_grid_desc,
                               const InDataBlockDesc& in_block_desc,
                               const InClusterBorderDesc& in_cluster_border_desc,
                               InDataBlockTransfer& in_blockwise_copy,
                               const InDataGridBuffer& in_grid_buf,
                               InDataBlockBuffer& in_block_buf,
                               ExchangeDataBlockBuffer& prev_block_buf,
                               ClusterBorderDataBlockBuffer& pre_cluster_buf,
                               PreNextDataBlockTransfer& pre_blockwise_copy,
                               ExchangeDataBlockBuffer& next_block_buf,
                               ClusterBorderDataBlockBuffer& next_cluster_buf,
                               PreNextDataBlockTransfer& next_blockwise_copy,
                               const InDataBlockTransferStep&,
                               const WeiDataGridDesc& wei_grid_desc,
                               const WeiDataBlockDesc& wei_block_desc,
                               WeiDataBlockTransfer& wei_blockwise_copy,
                               const WeiDataGridBuffer& wei_grid_buf,
                               WeiDataBlockBuffer& wei_block_buf,
                               const WeiDataBlockTransferStep&,
                               const DsDataGridDesc& ds_grid_desc,
                               const DsDataBlockDesc& ds_block_desc,
                               DsDataBlockTransfer& ds_blockwise_copy,
                               const DsDataGridBuffer& ds_grid_buf,
                               DsDataBlockBuffer& ds_block_buf,
                               const BlockwiseConv& blockwise_conv,
                               AccumThreadBuffer& accum_thread_buf,
                               OutThreadBuffer& out_thread_buf,
                               index_t num_loop)
    {
#if defined(__gfx13__)
        constexpr auto wei_block_copy_step = to_multi_index(WeiDataBlockTransferStep{});
        constexpr auto in_block_copy_step  = to_multi_index(InDataBlockTransferStep{});

        // sync between data load wave (0) and conv wave (1)
#ifdef CK_USE_AMD_SEMAPHORE_ASM
        WavegroupSemaphore<WaveIdRun, 1> semaDataReady;
        WavegroupSemaphore<WaveIdRun, 2> semaLdsReady;
        WavegroupSemaphore<WaveIdLoad, 1> semaDataLdsFree;
        WavegroupSemaphore<WaveIdRun, 3> semaAccumFree;
        WavegroupSemaphore<WaveIdPostRun, 1> semaAccumReady;
        WavegroupSemaphore<WaveIdLoad, 2> semFromNext;
        WavegroupSemaphore<WaveIdLoad, 3> semFromPrev;
#else
        __shared__ WavegroupSemaphore<WaveIdRun> semaDataReady;
        __shared__ WavegroupSemaphore<WaveIdRun> semaLdsReady;
        __shared__ WavegroupSemaphore<WaveIdLoad> semaDataLdsFree;
        __shared__ WavegroupSemaphore<WaveIdRun> semaAccumFree;
        __shared__ WavegroupSemaphore<WaveIdPostRun> semaAccumReady;
        __shared__ WavegroupSemaphore<WaveIdLoad> semFromNext;
        __shared__ WavegroupSemaphore<WaveIdLoad> semFromPrev;
#endif
        // sync for all wave with id = 0 in a group
#ifdef CK_USE_AMD_NAMED_BARRIER_ASM
        NamedBarrier<1, 4> barrierLds;
#else
        __shared__ NamedBarrier<4> barrierLds;
#endif

        static
            __attribute__((exp_amd_laneshared)) int dataFromPrev[ExchangeDataBlockBuffer::Size()];
        static
            __attribute__((exp_amd_laneshared)) int dataFromNext[ExchangeDataBlockBuffer::Size()];

        constexpr index_t NumTap           = WeiDataBlockTransfer::Size();
        constexpr auto in_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
        constexpr auto wei_remap_table     = BlockwiseConv::GetWeightRemapTable();

        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0])>>;

        if(get_wave_id_in_wavegroup() == WaveIdRun)
        {
            // Initialize C
            if constexpr(HasMainLoop)
            {
                accum_thread_buf.Clear();
            }
        }

        if(get_wave_id_in_wavegroup() == WaveIdLoad)
        {
            barrierLds.init();
            barrierLds.join();
        }

        semaDataReady.init();
        semaLdsReady.init();
        semaDataLdsFree.init(0, 1, true);
        semaAccumReady.init();
        semaAccumFree.init(0, 1, true);
        semFromNext.init();
        semFromPrev.init();

        auto semaAccums = make_tuple(&semaAccumReady, &semaAccumFree);

        // wait semaphore, named-barrier init
        __syncthreads();

        bool isChainStartVal = __builtin_amdgcn_spatial_cluster_is_chain_start();
        bool isChainStart    = __builtin_amdgcn_readfirstlane(isChainStartVal);
        bool isChainEndVal   = __builtin_amdgcn_spatial_cluster_is_chain_end();
        bool isChainEnd      = __builtin_amdgcn_readfirstlane(isChainEndVal);

        // pre-fetch data
        if(get_wave_id_in_wavegroup() == WaveIdLoad)
        {
            if constexpr(WeiDataEnableLds == false)
            {
                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .Run(wei_grid_desc,
                             wei_grid_buf,
                             wei_block_desc,
                             make_tuple(I0, I0, wei_remap_table[tapIdx], I0, I0, I0),
                             wei_block_buf);
                });
                if constexpr(HasMainLoop)
                {
                    static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                            .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                    });
                    wei_block_buf.SwitchBuffer();
                }
            }

            if constexpr(InDataEnableLds == false)
            {
                in_blockwise_copy.Run(
                    in_grid_desc, in_grid_buf, in_block_desc, in_block_origin_idx, in_block_buf);

                if constexpr(EnableSpatialCluster)
                {
                    blockwise_conv.exchange_neighbor_data(in_grid_desc,
                                                          in_block_buf,
                                                          prev_block_buf,
                                                          next_block_buf,
                                                          dataFromPrev,
                                                          semFromPrev,
                                                          dataFromNext,
                                                          semFromNext);
                    if(isChainStart)
                    {
                        pre_blockwise_copy.Run(in_grid_desc,
                                               in_grid_buf,
                                               in_cluster_border_desc,
                                               in_block_origin_idx,
                                               pre_cluster_buf);
                    }
                    if(isChainEnd)
                    {
                        next_blockwise_copy.Run(in_grid_desc,
                                                in_grid_buf,
                                                in_cluster_border_desc,
                                                in_block_origin_idx,
                                                next_cluster_buf);
                    }
                }

                if constexpr(HasMainLoop)
                {
                    in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);
                    in_block_buf.SwitchBuffer();
                    if constexpr(EnableSpatialCluster)
                    {
                        prev_block_buf.SwitchBuffer();
                        next_block_buf.SwitchBuffer();
                        pre_cluster_buf.SwitchBuffer();
                        next_cluster_buf.SwitchBuffer();
                    }
                }
            }

            if constexpr(DsDataEnableLds == false)
            {
                constexpr index_t NumDs = DsDataBlockTransfer::Size();
                static_for<0, NumDs, 1>{}([&](auto i) {
                    using DDataBlockTransfer =
                        std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
                    using DBlockDesc        = remove_cvref_t<decltype(ds_block_desc[i])>;
                    auto d_block_origin_idx = generate_tuple(
                        [&](auto) { return I0; }, Number<DBlockDesc::GetNumOfDimension()>{});
                    const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                        .Run(ds_grid_desc[Number<i>{}],
                             ds_grid_buf[Number<i>{}],
                             ds_block_desc[Number<i>{}],
                             d_block_origin_idx,
                             ds_block_buf(i));
                });
            }
            semaDataReady.template signal<SemaphoreAddressSpaceGlobal>();
        }

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                if(get_wave_id_in_wavegroup() == WaveIdLoad)
                {
                    semaDataLdsFree.template wait<0>(); // sync within wavegroup
                    barrierLds.signal();                // sync in workgroup
                    barrierLds.wait();
                    if constexpr(WeiDataEnableLds)
                    {
                        if constexpr(EnableAsync)
                        {
                            static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                                const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                                    .Run(
                                        wei_grid_desc, wei_grid_buf, wei_block_desc, wei_block_buf);
                            });
                        }
                        else
                        {
                            static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                                const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                                    .RunRead(wei_grid_desc, wei_grid_buf);
                            });
                            static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                                const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                                    .RunWrite(wei_block_desc, wei_block_buf);
                            });
                        }
                    }
                    else
                    {
                        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                                .Run(wei_grid_desc,
                                     wei_grid_buf,
                                     wei_block_desc,
                                     make_tuple(I0, I0, wei_remap_table[tapIdx], I0, I0, I0),
                                     wei_block_buf);
                        });
                        wei_block_buf.SwitchBuffer();
                    }

                    if constexpr(InDataEnableLds)
                    {
                        if constexpr(EnableAsync)
                        {
                            in_blockwise_copy.Run(
                                in_grid_desc, in_grid_buf, in_block_desc, in_block_buf);
                        }
                        else
                        {
                            in_blockwise_copy.RunRead(in_grid_desc, in_grid_buf);
                            in_blockwise_copy.RunWrite(in_block_desc, in_block_buf);
                        }
                    }
                    else
                    {
                        in_blockwise_copy.Run(in_grid_desc,
                                              in_grid_buf,
                                              in_block_desc,
                                              in_block_origin_idx,
                                              in_block_buf);

                        if constexpr(EnableSpatialCluster)
                        {
                            blockwise_conv.exchange_neighbor_data(in_grid_desc,
                                                                  in_block_buf,
                                                                  prev_block_buf,
                                                                  next_block_buf,
                                                                  dataFromPrev,
                                                                  semFromPrev,
                                                                  dataFromNext,
                                                                  semFromNext);
                            if(isChainStart)
                            {
                                pre_blockwise_copy.Run(in_grid_desc,
                                                       in_grid_buf,
                                                       in_cluster_border_desc,
                                                       in_block_origin_idx,
                                                       pre_cluster_buf);
                            }
                            if(isChainEnd)
                            {
                                next_blockwise_copy.Run(in_grid_desc,
                                                        in_grid_buf,
                                                        in_cluster_border_desc,
                                                        in_block_origin_idx,
                                                        next_cluster_buf);
                            }
                            prev_block_buf.SwitchBuffer();
                            next_block_buf.SwitchBuffer();
                            pre_cluster_buf.SwitchBuffer();
                            next_cluster_buf.SwitchBuffer();
                        }

                        in_block_buf.SwitchBuffer();
                    }

                    static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                            .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                    });
                    in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

                    barrierLds.template sync_lds<EnableAsync>();

                    semaLdsReady.template signal<0>();
                    semaDataReady.template signal<SemaphoreAddressSpaceGlobal>();
                }

                if(get_wave_id_in_wavegroup() == WaveIdRun)
                {
                    semaLdsReady.template wait<0>();
                    semaDataReady.template wait<0>();
                }
                if(get_wave_id_in_wavegroup() == WaveIdRun ||
                   get_wave_id_in_wavegroup() == WaveIdPostRun)
                {
                    ExchangeDataBlockBuffer preBuf =
                        EnableSpatialCluster
                            ? (isChainStart ? bit_cast<ExchangeDataBlockBuffer>(pre_cluster_buf)
                                            : prev_block_buf)
                            : nullptr;
                    ExchangeDataBlockBuffer nextBuf =
                        EnableSpatialCluster
                            ? (isChainEnd ? bit_cast<ExchangeDataBlockBuffer>(next_cluster_buf)
                                          : next_block_buf)
                            : nullptr;

                    blockwise_conv.Run(wei_block_buf,
                                       in_block_buf,
                                       preBuf,
                                       nextBuf,
                                       ds_block_buf,
                                       accum_thread_buf,
                                       out_thread_buf,
                                       semaAccums,
                                       Number<HasMainLoop>{},
                                       Number<false>{});
                }
                if(get_wave_id_in_wavegroup() == WaveIdRun)
                {
                    if constexpr(InDataEnableLds == false)
                    {
                        in_block_buf.SwitchBuffer();
                        if constexpr(EnableSpatialCluster)
                        {
                            prev_block_buf.SwitchBuffer();
                            next_block_buf.SwitchBuffer();
                            pre_cluster_buf.SwitchBuffer();
                            next_cluster_buf.SwitchBuffer();
                        }
                    }
                    if constexpr(WeiDataEnableLds == false)
                    {
                        wei_block_buf.SwitchBuffer();
                    }

                    semaDataLdsFree.template signal<0>();
                }

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            if(get_wave_id_in_wavegroup() == WaveIdLoad)
            {
                semaDataLdsFree.template wait<0>();
                if constexpr(WeiDataEnableLds)
                {
                    if constexpr(EnableAsync)
                    {
                        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                                .Run(wei_grid_desc, wei_grid_buf, wei_block_desc, wei_block_buf);
                        });
                    }
                    else
                    {
                        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                                .RunRead(wei_grid_desc, wei_grid_buf);
                        });
                        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                                .RunWrite(wei_block_desc, wei_block_buf);
                        });
                    }
                }

                if constexpr(InDataEnableLds)
                {
                    if constexpr(EnableAsync)
                    {
                        in_blockwise_copy.Run(
                            in_grid_desc, in_grid_buf, in_block_desc, in_block_buf);
                    }
                    else
                    {
                        in_blockwise_copy.RunRead(in_grid_desc, in_grid_buf);
                        in_blockwise_copy.RunWrite(in_block_desc, in_block_buf);
                    }
                }
                if constexpr(DsDataEnableLds)
                {
                    constexpr index_t NumDs = DsDataBlockTransfer::Size();
                    if constexpr(EnableAsync)
                    {
                        static_for<0, NumDs, 1>{}([&](auto i) {
                            using DDataBlockTransfer =
                                std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
                            const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                                .Run(ds_grid_desc[Number<i>{}],
                                     ds_grid_buf[Number<i>{}],
                                     ds_block_desc[Number<i>{}],
                                     ds_block_buf(i));
                        });
                    }
                    else
                    {
                        static_for<0, NumDs, 1>{}([&](auto i) {
                            using DDataBlockTransfer =
                                std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
                            const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                                .RunRead(ds_grid_desc[Number<i>{}], ds_grid_buf[Number<i>{}]);
                        });
                        static_for<0, NumDs, 1>{}([&](auto i) {
                            using DDataBlockTransfer =
                                std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
                            const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                                .RunWrite(ds_block_desc[Number<i>{}], ds_block_buf(i));
                        });
                    }
                }

                barrierLds.template sync_lds<EnableAsync>();
                semaLdsReady.template signal<0>();
            }

            if(get_wave_id_in_wavegroup() == WaveIdRun)
            {
                semaLdsReady.template wait<0>();
                semaDataReady.template wait<0>();
            }
            if(get_wave_id_in_wavegroup() == WaveIdRun ||
               get_wave_id_in_wavegroup() == WaveIdPostRun)
            {
                ExchangeDataBlockBuffer preBuf =
                    EnableSpatialCluster
                        ? (isChainStart ? bit_cast<ExchangeDataBlockBuffer>(pre_cluster_buf)
                                        : prev_block_buf)
                        : nullptr;
                ExchangeDataBlockBuffer nextBuf =
                    EnableSpatialCluster
                        ? (isChainEnd ? bit_cast<ExchangeDataBlockBuffer>(next_cluster_buf)
                                      : next_block_buf)
                        : nullptr;

                blockwise_conv.Run(wei_block_buf,
                                   in_block_buf,
                                   preBuf,
                                   nextBuf,
                                   ds_block_buf,
                                   accum_thread_buf,
                                   out_thread_buf,
                                   semaAccums,
                                   Number<HasMainLoop>{},
                                   Number<true>{});
            }
        }
#else
        ignore = in_grid_desc;
        ignore = in_block_desc;
        ignore = in_cluster_border_desc;
        ignore = in_blockwise_copy;
        ignore = in_grid_buf;
        ignore = in_block_buf;
        ignore = prev_block_buf;
        ignore = pre_cluster_buf;
        ignore = pre_blockwise_copy;
        ignore = next_block_buf;
        ignore = next_cluster_buf;
        ignore = next_blockwise_copy;
        ignore = wei_grid_desc;
        ignore = wei_block_desc;
        ignore = wei_blockwise_copy;
        ignore = wei_grid_buf;
        ignore = wei_block_buf;
        ignore = ds_grid_desc;
        ignore = ds_block_desc;
        ignore = ds_blockwise_copy;
        ignore = ds_grid_buf;
        ignore = ds_block_buf;
        ignore = blockwise_conv;
        ignore = accum_thread_buf;
        ignore = out_thread_buf;
        ignore = num_loop;
#endif
    }
};

template <bool EnableAsync, bool EnableSpatialCluster>
struct GridwiseConvPipeline_v2<1, false, false, false, EnableAsync, EnableSpatialCluster>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename InDataGridDesc,
              typename InDataBlockDesc,
              typename InClusterBorderDesc,
              typename InDataBlockTransfer,
              typename InDataGridBuffer,
              typename InDataBlockBuffer,
              typename ExchangeDataBlockBuffer,
              typename ClusterBorderDataBlockBuffer,
              typename PreNextDataBlockTransfer,
              typename InDataBlockTransferStep,
              typename WeiDataGridDesc,
              typename WeiDataBlockDesc,
              typename WeiDataBlockTransfer,
              typename WeiDataGridBuffer,
              typename WeiDataBlockBuffer,
              typename WeiDataBlockTransferStep,
              typename DsDataGridDesc,
              typename DsDataBlockDesc,
              typename DsDataBlockTransfer,
              typename DsDataGridBuffer,
              typename DsDataBlockBuffer,
              typename BlockwiseConv,
              typename AccumThreadBuffer,
              typename OutThreadBuffer>
    __device__ static void Run(const InDataGridDesc& in_grid_desc,
                               const InDataBlockDesc& in_block_desc,
                               const InClusterBorderDesc& in_cluster_border_desc,
                               InDataBlockTransfer& in_blockwise_copy,
                               const InDataGridBuffer& in_grid_buf,
                               InDataBlockBuffer& in_block_buf,
                               ExchangeDataBlockBuffer& prev_block_buf,
                               ClusterBorderDataBlockBuffer& pre_cluster_buf,
                               PreNextDataBlockTransfer& pre_blockwise_copy,
                               ExchangeDataBlockBuffer& next_block_buf,
                               ClusterBorderDataBlockBuffer& next_cluster_buf,
                               PreNextDataBlockTransfer& next_blockwise_copy,
                               const InDataBlockTransferStep&,
                               const WeiDataGridDesc& wei_grid_desc,
                               const WeiDataBlockDesc& wei_block_desc,
                               WeiDataBlockTransfer& wei_blockwise_copy,
                               const WeiDataGridBuffer& wei_grid_buf,
                               WeiDataBlockBuffer& wei_block_buf,
                               const WeiDataBlockTransferStep&,
                               const DsDataGridDesc& ds_grid_desc,
                               const DsDataBlockDesc& ds_block_desc,
                               DsDataBlockTransfer& ds_blockwise_copy,
                               const DsDataGridBuffer& ds_grid_buf,
                               DsDataBlockBuffer& ds_block_buf,
                               const BlockwiseConv& blockwise_conv,
                               AccumThreadBuffer& accum_thread_buf,
                               OutThreadBuffer& out_thread_buf,
                               index_t num_loop)
    {
#if defined(__gfx13__)
        constexpr auto wei_block_copy_step = to_multi_index(WeiDataBlockTransferStep{});
        constexpr auto in_block_copy_step  = to_multi_index(InDataBlockTransferStep{});

        // sync between data load wave (0) and conv wave (1)
#ifdef CK_USE_AMD_SEMAPHORE_ASM
        WavegroupSemaphore<WaveIdRun, 1> semaDataReady;
        WavegroupSemaphore<WaveIdLoad, 1> semaDataFree;
        WavegroupSemaphore<WaveIdRun, 2> semaAccumFree;
        WavegroupSemaphore<WaveIdPostRun, 1> semaAccumReady;
        WavegroupSemaphore<WaveIdLoad, 2> semFromNext;
        WavegroupSemaphore<WaveIdLoad, 3> semFromPrev;
#else
        __shared__ WavegroupSemaphore<WaveIdRun> semaDataReady;
        __shared__ WavegroupSemaphore<WaveIdLoad> semaDataFree;
        __shared__ WavegroupSemaphore<WaveIdRun> semaAccumFree;
        __shared__ WavegroupSemaphore<WaveIdPostRun> semaAccumReady;
        __shared__ WavegroupSemaphore<WaveIdLoad> semFromNext;
        __shared__ WavegroupSemaphore<WaveIdLoad> semFromPrev;
#endif

        static
            __attribute__((exp_amd_laneshared)) int dataFromPrev[ExchangeDataBlockBuffer::Size()];
        static
            __attribute__((exp_amd_laneshared)) int dataFromNext[ExchangeDataBlockBuffer::Size()];

        constexpr index_t NumTap           = WeiDataBlockTransfer::Size();
        constexpr auto in_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
        constexpr auto wei_remap_table     = BlockwiseConv::GetWeightRemapTable();

        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0])>>;

        if(get_wave_id_in_wavegroup() == WaveIdRun)
        {
            // Initialize C
            if constexpr(HasMainLoop)
            {
                accum_thread_buf.Clear();
            }
        }

        semaDataReady.init();
        semaDataFree.init(0, 1, true);
        semaAccumReady.init();
        semaAccumFree.init(0, 1, true);
        semFromNext.init();
        semFromPrev.init();

        auto semaAccums = make_tuple(&semaAccumReady, &semaAccumFree);
        // wait semaphore init
        __syncthreads();

        bool isChainStartVal = __builtin_amdgcn_spatial_cluster_is_chain_start();
        bool isChainStart    = __builtin_amdgcn_readfirstlane(isChainStartVal);
        bool isChainEndVal   = __builtin_amdgcn_spatial_cluster_is_chain_end();
        bool isChainEnd      = __builtin_amdgcn_readfirstlane(isChainEndVal);

        // pre-fetch data
        if(get_wave_id_in_wavegroup() == WaveIdLoad)
        {
            static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                    .Run(wei_grid_desc,
                         wei_grid_buf,
                         wei_block_desc,
                         make_tuple(I0, I0, wei_remap_table[tapIdx], I0, I0, I0),
                         wei_block_buf);
            });

            in_blockwise_copy.Run(
                in_grid_desc, in_grid_buf, in_block_desc, in_block_origin_idx, in_block_buf);

            if constexpr(EnableSpatialCluster)
            {
                blockwise_conv.exchange_neighbor_data(in_grid_desc,
                                                      in_block_buf,
                                                      prev_block_buf,
                                                      next_block_buf,
                                                      dataFromPrev,
                                                      semFromPrev,
                                                      dataFromNext,
                                                      semFromNext);
                if(isChainStart)
                {
                    pre_blockwise_copy.Run(in_grid_desc,
                                           in_grid_buf,
                                           in_cluster_border_desc,
                                           in_block_origin_idx,
                                           pre_cluster_buf);
                }
                if(isChainEnd)
                {
                    next_blockwise_copy.Run(in_grid_desc,
                                            in_grid_buf,
                                            in_cluster_border_desc,
                                            in_block_origin_idx,
                                            next_cluster_buf);
                }
            }

            constexpr index_t NumDs = DsDataBlockTransfer::Size();
            static_for<0, NumDs, 1>{}([&](auto i) {
                using DDataBlockTransfer =
                    std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
                using DBlockDesc        = remove_cvref_t<decltype(ds_block_desc[i])>;
                auto d_block_origin_idx = generate_tuple([&](auto) { return I0; },
                                                         Number<DBlockDesc::GetNumOfDimension()>{});
                const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                    .Run(ds_grid_desc[Number<i>{}],
                         ds_grid_buf[Number<i>{}],
                         ds_block_desc[Number<i>{}],
                         d_block_origin_idx,
                         ds_block_buf(i));
            });

            if constexpr(HasMainLoop)
            {
                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                });
                in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);
                in_block_buf.SwitchBuffer();
                wei_block_buf.SwitchBuffer();
                if constexpr(EnableSpatialCluster)
                {
                    prev_block_buf.SwitchBuffer();
                    next_block_buf.SwitchBuffer();
                    pre_cluster_buf.SwitchBuffer();
                    next_cluster_buf.SwitchBuffer();
                }
            }

            semaDataReady.template signal<SemaphoreAddressSpaceGlobal>();
        }

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                if(get_wave_id_in_wavegroup() == WaveIdLoad)
                {
                    semaDataFree.template wait<0>();
                    static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                            .Run(wei_grid_desc,
                                 wei_grid_buf,
                                 wei_block_desc,
                                 make_tuple(I0, I0, wei_remap_table[tapIdx], I0, I0, I0),
                                 wei_block_buf);
                    });
                    wei_block_buf.SwitchBuffer();

                    in_blockwise_copy.Run(in_grid_desc,
                                          in_grid_buf,
                                          in_block_desc,
                                          in_block_origin_idx,
                                          in_block_buf);

                    if constexpr(EnableSpatialCluster)
                    {
                        blockwise_conv.exchange_neighbor_data(in_grid_desc,
                                                              in_block_buf,
                                                              prev_block_buf,
                                                              next_block_buf,
                                                              dataFromPrev,
                                                              semFromPrev,
                                                              dataFromNext,
                                                              semFromNext);
                        if(isChainStart)
                        {
                            pre_blockwise_copy.Run(in_grid_desc,
                                                   in_grid_buf,
                                                   in_cluster_border_desc,
                                                   in_block_origin_idx,
                                                   pre_cluster_buf);
                            pre_cluster_buf.SwitchBuffer();
                        }
                        if(isChainEnd)
                        {
                            next_blockwise_copy.Run(in_grid_desc,
                                                    in_grid_buf,
                                                    in_cluster_border_desc,
                                                    in_block_origin_idx,
                                                    next_cluster_buf);
                            next_cluster_buf.SwitchBuffer();
                        }
                        prev_block_buf.SwitchBuffer();
                        next_block_buf.SwitchBuffer();
                    }

                    in_block_buf.SwitchBuffer();

                    static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                            .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                    });
                    in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

                    semaDataReady.template signal<SemaphoreAddressSpaceGlobal>();
                }

                if(get_wave_id_in_wavegroup() == WaveIdRun)
                {
                    semaDataReady.template wait<0>();
                }
                if(get_wave_id_in_wavegroup() == WaveIdRun ||
                   get_wave_id_in_wavegroup() == WaveIdPostRun)
                {
                    ExchangeDataBlockBuffer preBuf =
                        EnableSpatialCluster
                            ? (isChainStart ? bit_cast<ExchangeDataBlockBuffer>(pre_cluster_buf)
                                            : prev_block_buf)
                            : nullptr;
                    ExchangeDataBlockBuffer nextBuf =
                        EnableSpatialCluster
                            ? (isChainEnd ? bit_cast<ExchangeDataBlockBuffer>(next_cluster_buf)
                                          : next_block_buf)
                            : nullptr;

                    blockwise_conv.Run(wei_block_buf,
                                       in_block_buf,
                                       preBuf,
                                       nextBuf,
                                       ds_block_buf,
                                       accum_thread_buf,
                                       out_thread_buf,
                                       semaAccums,
                                       Number<HasMainLoop>{},
                                       Number<false>{});
                }
                if(get_wave_id_in_wavegroup() == WaveIdRun)
                {
                    in_block_buf.SwitchBuffer();
                    wei_block_buf.SwitchBuffer();
                    if constexpr(EnableSpatialCluster)
                    {
                        prev_block_buf.SwitchBuffer();
                        next_block_buf.SwitchBuffer();
                        pre_cluster_buf.SwitchBuffer();
                        next_cluster_buf.SwitchBuffer();
                    }

                    semaDataFree.template signal<0>();
                }

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            if(get_wave_id_in_wavegroup() == WaveIdRun)
            {
                semaDataReady.template wait<0>();
            }
            if(get_wave_id_in_wavegroup() == WaveIdRun ||
               get_wave_id_in_wavegroup() == WaveIdPostRun)
            {
                ExchangeDataBlockBuffer preBuf =
                    EnableSpatialCluster
                        ? (isChainStart ? bit_cast<ExchangeDataBlockBuffer>(pre_cluster_buf)
                                        : prev_block_buf)
                        : nullptr;
                ExchangeDataBlockBuffer nextBuf =
                    EnableSpatialCluster
                        ? (isChainEnd ? bit_cast<ExchangeDataBlockBuffer>(next_cluster_buf)
                                      : next_block_buf)
                        : nullptr;

                blockwise_conv.Run(wei_block_buf,
                                   in_block_buf,
                                   preBuf,
                                   nextBuf,
                                   ds_block_buf,
                                   accum_thread_buf,
                                   out_thread_buf,
                                   semaAccums,
                                   Number<HasMainLoop>{},
                                   Number<true>{});
            }
        }
#else
        ignore = in_grid_desc;
        ignore = in_block_desc;
        ignore = in_cluster_border_desc;
        ignore = in_blockwise_copy;
        ignore = in_grid_buf;
        ignore = in_block_buf;
        ignore = prev_block_buf;
        ignore = pre_cluster_buf;
        ignore = pre_blockwise_copy;
        ignore = next_block_buf;
        ignore = next_cluster_buf;
        ignore = next_blockwise_copy;
        ignore = wei_grid_desc;
        ignore = wei_block_desc;
        ignore = wei_blockwise_copy;
        ignore = wei_grid_buf;
        ignore = wei_block_buf;
        ignore = ds_grid_desc;
        ignore = ds_block_desc;
        ignore = ds_blockwise_copy;
        ignore = ds_grid_buf;
        ignore = ds_block_buf;
        ignore = blockwise_conv;
        ignore = accum_thread_buf;
        ignore = out_thread_buf;
        ignore = num_loop;
#endif
    }
};

template <bool EnableAsync, bool EnableSpatialCluster>
struct GridwiseConvPipeline_v2<1, true, true, true, EnableAsync, EnableSpatialCluster>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename InDataGridDesc,
              typename InDataBlockDesc,
              typename InClusterBorderDesc,
              typename InDataBlockTransfer,
              typename InDataGridBuffer,
              typename InDataBlockBuffer,
              typename ExchangeDataBlockBuffer,
              typename ClusterBorderDataBlockBuffer,
              typename PreNextDataBlockTransfer,
              typename InDataBlockTransferStep,
              typename WeiDataGridDesc,
              typename WeiDataBlockDesc,
              typename WeiDataBlockTransfer,
              typename WeiDataGridBuffer,
              typename WeiDataBlockBuffer,
              typename WeiDataBlockTransferStep,
              typename DsDataGridDesc,
              typename DsDataBlockDesc,
              typename DsDataBlockTransfer,
              typename DsDataGridBuffer,
              typename DsDataBlockBuffer,
              typename BlockwiseConv,
              typename AccumThreadBuffer,
              typename OutThreadBuffer>
    __device__ static void Run(const InDataGridDesc& in_grid_desc,
                               const InDataBlockDesc& in_block_desc,
                               const InClusterBorderDesc& in_cluster_border_desc,
                               InDataBlockTransfer& in_blockwise_copy,
                               const InDataGridBuffer& in_grid_buf,
                               InDataBlockBuffer& in_block_buf,
                               ExchangeDataBlockBuffer& prev_block_buf,
                               ClusterBorderDataBlockBuffer& pre_cluster_buf,
                               PreNextDataBlockTransfer& pre_blockwise_copy,
                               ExchangeDataBlockBuffer& next_block_buf,
                               ClusterBorderDataBlockBuffer& next_cluster_buf,
                               PreNextDataBlockTransfer& next_blockwise_copy,
                               const InDataBlockTransferStep&,
                               const WeiDataGridDesc& wei_grid_desc,
                               const WeiDataBlockDesc& wei_block_desc,
                               WeiDataBlockTransfer& wei_blockwise_copy,
                               const WeiDataGridBuffer& wei_grid_buf,
                               WeiDataBlockBuffer& wei_block_buf,
                               const WeiDataBlockTransferStep&,
                               const DsDataGridDesc& ds_grid_desc,
                               const DsDataBlockDesc& ds_block_desc,
                               DsDataBlockTransfer& ds_blockwise_copy,
                               const DsDataGridBuffer& ds_grid_buf,
                               DsDataBlockBuffer& ds_block_buf,
                               const BlockwiseConv& blockwise_conv,
                               AccumThreadBuffer& accum_thread_buf,
                               OutThreadBuffer& out_thread_buf,
                               index_t num_loop)
    {
#if defined(__gfx13__)
        constexpr auto wei_block_copy_step = to_multi_index(WeiDataBlockTransferStep{});
        constexpr auto in_block_copy_step  = to_multi_index(InDataBlockTransferStep{});

        ignore = in_cluster_border_desc;
        ignore = prev_block_buf;
        ignore = pre_cluster_buf;
        ignore = pre_blockwise_copy;
        ignore = next_block_buf;
        ignore = next_cluster_buf;
        ignore = next_blockwise_copy;
        // sync between data load wave (0) and conv wave (1)
#ifdef CK_USE_AMD_SEMAPHORE_ASM
        WavegroupSemaphore<WaveIdRun, 1> semaLdsReady;
        WavegroupSemaphore<WaveIdLoad, 1> semaLdsFree;
        WavegroupSemaphore<WaveIdRun, 2> semaAccumFree;
        WavegroupSemaphore<WaveIdPostRun, 1> semaAccumReady;
#else
        __shared__ WavegroupSemaphore<WaveIdRun> semaLdsReady;
        __shared__ WavegroupSemaphore<WaveIdLoad> semaLdsFree;
        __shared__ WavegroupSemaphore<WaveIdRun> semaAccumFree;
        __shared__ WavegroupSemaphore<WaveIdPostRun> semaAccumReady;
#endif
#ifdef CK_USE_AMD_NAMED_BARRIER_ASM
        NamedBarrier<1, 4> barrierLds;
#else
        __shared__ NamedBarrier<4> barrierLds;
#endif
        constexpr index_t NumTap = WeiDataBlockTransfer::Size();

        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0])>>;

        if(get_wave_id_in_wavegroup() == WaveIdRun)
        {
            // Initialize C
            if constexpr(HasMainLoop)
            {
                accum_thread_buf.Clear();
            }
        }

        if(get_wave_id_in_wavegroup() == WaveIdLoad)
        {
            barrierLds.init();
            barrierLds.join();
        }

        semaLdsReady.init();
        semaLdsFree.init(0, 1, true);
        semaAccumReady.init();
        semaAccumFree.init(0, 1, true);

        auto semaAccums = make_tuple(&semaAccumReady, &semaAccumFree);
        // wait semaphore init
        __syncthreads();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                if(get_wave_id_in_wavegroup() == WaveIdLoad)
                {
                    semaLdsFree.template wait<0>();
                    barrierLds.signal();
                    barrierLds.wait();
                    if constexpr(EnableAsync)
                    {
                        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                                .Run(wei_grid_desc, wei_grid_buf, wei_block_desc, wei_block_buf);
                        });
                    }
                    else
                    {
                        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                                .RunRead(wei_grid_desc, wei_grid_buf);
                        });
                        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                                .RunWrite(wei_block_desc, wei_block_buf);
                        });
                    }

                    if constexpr(EnableAsync)
                    {
                        in_blockwise_copy.Run(
                            in_grid_desc, in_grid_buf, in_block_desc, in_block_buf);
                    }
                    else
                    {
                        in_blockwise_copy.RunRead(in_grid_desc, in_grid_buf);
                        in_blockwise_copy.RunWrite(in_block_desc, in_block_buf);
                    }

                    static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                            .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                    });
                    in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);
                    barrierLds.template sync_lds<EnableAsync>();

                    semaLdsReady.template signal<0>();
                }

                if(get_wave_id_in_wavegroup() == WaveIdRun)
                {
                    semaLdsReady.template wait<0>();
                }
                if(get_wave_id_in_wavegroup() == WaveIdRun ||
                   get_wave_id_in_wavegroup() == WaveIdPostRun)
                {
                    blockwise_conv.Run(wei_block_buf,
                                       in_block_buf,
                                       nullptr,
                                       nullptr,
                                       ds_block_buf,
                                       accum_thread_buf,
                                       out_thread_buf,
                                       semaAccums,
                                       Number<HasMainLoop>{},
                                       Number<false>{});
                }
                if(get_wave_id_in_wavegroup() == WaveIdRun)
                {
                    semaLdsFree.template signal<0>();
                }
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            if(get_wave_id_in_wavegroup() == WaveIdLoad)
            {
                semaLdsFree.template wait<0>();
                barrierLds.signal();
                barrierLds.wait();
                if constexpr(EnableAsync)
                {
                    static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                            .Run(wei_grid_desc, wei_grid_buf, wei_block_desc, wei_block_buf);
                    });
                }
                else
                {
                    static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                            .RunRead(wei_grid_desc, wei_grid_buf);
                    });
                    static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                            .RunWrite(wei_block_desc, wei_block_buf);
                    });
                }

                if constexpr(EnableAsync)
                {
                    in_blockwise_copy.Run(in_grid_desc, in_grid_buf, in_block_desc, in_block_buf);
                }
                else
                {
                    in_blockwise_copy.RunRead(in_grid_desc, in_grid_buf);
                    in_blockwise_copy.RunWrite(in_block_desc, in_block_buf);
                }

                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                });
                in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

                constexpr index_t NumDs = DsDataBlockTransfer::Size();
                if constexpr(EnableAsync)
                {
                    static_for<0, NumDs, 1>{}([&](auto i) {
                        using DDataBlockTransfer =
                            std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
                        const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                            .Run(ds_grid_desc[Number<i>{}],
                                 ds_grid_buf[Number<i>{}],
                                 ds_block_desc[Number<i>{}],
                                 ds_block_buf(i));
                    });
                }
                else
                {
                    static_for<0, NumDs, 1>{}([&](auto i) {
                        using DDataBlockTransfer =
                            std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
                        const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                            .RunRead(ds_grid_desc[Number<i>{}], ds_grid_buf[Number<i>{}]);
                    });
                    static_for<0, NumDs, 1>{}([&](auto i) {
                        using DDataBlockTransfer =
                            std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
                        const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                            .RunWrite(ds_block_desc[Number<i>{}], ds_block_buf(i));
                    });
                }

                barrierLds.template sync_lds<EnableAsync>();

                semaLdsReady.template signal<0>();
            }

            if(get_wave_id_in_wavegroup() == WaveIdRun)
            {
                semaLdsReady.template wait<0>();
            }
            if(get_wave_id_in_wavegroup() == WaveIdRun ||
               get_wave_id_in_wavegroup() == WaveIdPostRun)
            {
                blockwise_conv.Run(wei_block_buf,
                                   in_block_buf,
                                   nullptr,
                                   nullptr,
                                   ds_block_buf,
                                   accum_thread_buf,
                                   out_thread_buf,
                                   semaAccums,
                                   Number<HasMainLoop>{},
                                   Number<true>{});
            }
        }
#else
        ignore = in_grid_desc;
        ignore = in_block_desc;
        ignore = in_cluster_border_desc;
        ignore = in_blockwise_copy;
        ignore = in_grid_buf;
        ignore = in_block_buf;
        ignore = prev_block_buf;
        ignore = pre_cluster_buf;
        ignore = pre_blockwise_copy;
        ignore = next_block_buf;
        ignore = next_cluster_buf;
        ignore = next_blockwise_copy;
        ignore = wei_grid_desc;
        ignore = wei_block_desc;
        ignore = wei_blockwise_copy;
        ignore = wei_grid_buf;
        ignore = wei_block_buf;
        ignore = ds_grid_desc;
        ignore = ds_block_desc;
        ignore = ds_blockwise_copy;
        ignore = ds_grid_buf;
        ignore = ds_block_buf;
        ignore = blockwise_conv;
        ignore = accum_thread_buf;
        ignore = out_thread_buf;
        ignore = num_loop;
#endif
    }
};

} // namespace ck
