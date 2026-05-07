// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_conv_wcnn_pipeline_wavegroup.hpp"

namespace ck {

template <index_t NumPrefetch,
          bool InDataEnableLds,
          bool WeiDataEnableLds,
          bool DsDataEnableLds,
          bool EnableAsync>
struct GridwiseConvPipeline_v1
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
              typename InDataBlockTransfer,
              typename InDataGridBuffer,
              typename InDataBlockBuffer,
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
                               InDataBlockTransfer& in_blockwise_copy,
                               const InDataGridBuffer& in_grid_buf,
                               InDataBlockBuffer& in_block_buf,
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
        constexpr auto wei_block_copy_step = to_multi_index(WeiDataBlockTransferStep{});
        constexpr auto in_block_copy_step  = to_multi_index(InDataBlockTransferStep{});
        constexpr index_t NumTap           = WeiDataBlockTransfer::Size();
        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0])>>;
        using EmptySemas = Tuple<>;
        EmptySemas emptySemas;
        constexpr index_t NumDs = DsDataBlockTransfer::Size();

        // Initialize C
        if constexpr(HasMainLoop)
        {
            accum_thread_buf.Clear();
        }

        auto in_block_buf_switch  = in_block_buf;
        auto wei_block_buf_switch = wei_block_buf;

        // preload ds data into LDS
        if constexpr(DsDataEnableLds && EnableAsync == false)
        {
            static_for<0, NumDs, 1>{}([&](auto i) {
                using DDataBlockTransfer =
                    std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
                const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                    .RunRead(ds_grid_desc[i], ds_grid_buf[i]);
            });
        }
        else if constexpr(DsDataEnableLds && EnableAsync)
        {
            static_for<0, NumDs, 1>{}([&](auto i) {
                using DDataBlockTransfer =
                    std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
                const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                    .Run(ds_grid_desc[i], ds_grid_buf[i], ds_block_desc[i], ds_block_buf(i));
            });
        }
        else
        {
            static_for<0, NumDs, 1>{}([&](auto i) {
                using DDataBlockTransfer =
                    std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
                using DBlockDesc        = remove_cvref_t<decltype(ds_block_desc[i])>;
                auto d_block_origin_idx = generate_tuple([&](auto) { return I0; },
                                                         Number<DBlockDesc::GetNumOfDimension()>{});
                const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                    .Run(ds_grid_desc[i],
                         ds_grid_buf[i],
                         ds_block_desc[i],
                         d_block_origin_idx,
                         ds_block_buf(i));
            });
        }

        if constexpr(WeiDataEnableLds && EnableAsync == false)
        {
            static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                    .RunRead(wei_grid_desc, wei_grid_buf);
            });
        }
        else if constexpr(WeiDataEnableLds && EnableAsync)
        {
            static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                    .Run(wei_grid_desc, wei_grid_buf, wei_block_desc, wei_block_buf);
            });
        }
        else
        {
            constexpr auto wei_remap_table = BlockwiseConv::GetWeightRemapTable();
            // preload data into LDS
            static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                    .Run(wei_grid_desc,
                         wei_grid_buf,
                         wei_block_desc,
                         make_tuple(I0, I0, wei_remap_table[tapIdx], I0, I0, I0),
                         wei_block_buf);
            });
        }

        if constexpr(InDataEnableLds && EnableAsync == false)
        {
            in_blockwise_copy.RunRead(in_grid_desc, in_grid_buf);
        }
        else if constexpr(InDataEnableLds && EnableAsync)
        {
            in_blockwise_copy.Run(in_grid_desc, in_grid_buf, in_block_desc, in_block_buf);
        }
        else
        {
            constexpr auto in_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
            in_blockwise_copy.Run(
                in_grid_desc, in_grid_buf, in_block_desc, in_block_origin_idx, in_block_buf);
        }

        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
        });
        in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);
        // TODO: move slice windows for DS

        if constexpr(WeiDataEnableLds && EnableAsync == false)
        {
            static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                    .RunWrite(wei_block_desc, wei_block_buf);
            });
        }
        if constexpr(InDataEnableLds && EnableAsync == false)
        {
            in_blockwise_copy.RunWrite(in_block_desc, in_block_buf);
        }
        if constexpr(DsDataEnableLds && EnableAsync == false)
        {
            static_for<0, NumDs, 1>{}([&](auto i) {
                using DDataBlockTransfer =
                    std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
                const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                    .RunWrite(ds_block_desc[i], ds_block_buf(i));
            });
        }

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                if constexpr(WeiDataEnableLds && EnableAsync == false)
                {
                    static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                            .RunRead(wei_grid_desc, wei_grid_buf);
                    });
                }
                else if constexpr(WeiDataEnableLds && EnableAsync)
                {
                    // TODO: support double blocks
                }
                else
                {
                    constexpr auto wei_remap_table = BlockwiseConv::GetWeightRemapTable();
                    static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                            .Run(wei_grid_desc,
                                 wei_grid_buf,
                                 wei_block_desc,
                                 make_tuple(I0, I0, wei_remap_table[tapIdx], I0, I0, I0),
                                 wei_block_buf_switch);
                    });
                }

                if constexpr(InDataEnableLds && EnableAsync == false)
                {
                    in_blockwise_copy.RunRead(in_grid_desc, in_grid_buf);
                }
                else if constexpr(InDataEnableLds && EnableAsync)
                {
                    // TODO: support double blocks
                }
                else
                {
                    constexpr auto in_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
                    in_blockwise_copy.Run(in_grid_desc,
                                          in_grid_buf,
                                          in_block_desc,
                                          in_block_origin_idx,
                                          in_block_buf_switch);
                }
                if constexpr(EnableAsync)
                {
                    block_sync_lds_async_load();
                }
                else
                {
                    block_sync_lds();
                }
                blockwise_conv.Run(wei_block_buf,
                                   in_block_buf,
                                   nullptr,
                                   nullptr,
                                   ds_block_buf,
                                   accum_thread_buf,
                                   out_thread_buf,
                                   emptySemas,
                                   Number<HasMainLoop>{},
                                   Number<false>{});

                block_sync_lds();

                if constexpr(WeiDataEnableLds && EnableAsync == false)
                {
                    static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                            .RunWrite(wei_block_desc, wei_block_buf);
                    });
                }
                else if constexpr(WeiDataEnableLds && EnableAsync)
                {
                    static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                        const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                            .Run(wei_grid_desc, wei_grid_buf, wei_block_desc, wei_block_buf);
                    });
                }
                else
                {
                    wei_block_buf = wei_block_buf_switch;
                }

                if constexpr(InDataEnableLds && EnableAsync == false)
                {
                    in_blockwise_copy.RunWrite(in_block_desc, in_block_buf);
                }
                else if constexpr(InDataEnableLds && EnableAsync)
                {
                    in_blockwise_copy.Run(in_grid_desc, in_grid_buf, in_block_desc, in_block_buf);
                }
                else
                {
                    in_block_buf = in_block_buf_switch;
                }

                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                });

                in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            if constexpr(EnableAsync)
            {
                block_sync_lds_async_load();
            }
            else
            {
                block_sync_lds();
            }

            blockwise_conv.Run(wei_block_buf,
                               in_block_buf,
                               nullptr,
                               nullptr,
                               ds_block_buf,
                               accum_thread_buf,
                               out_thread_buf,
                               emptySemas,
                               Number<HasMainLoop>{},
                               Number<true>{});
            block_sync_lds();
        }
    }
};

// 1-stage prefetch
template <>
struct GridwiseConvPipeline_v1<1, true, true, true, false>
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
              typename InDataBlockTransfer,
              typename InDataGridBuffer,
              typename InDataBlockBuffer,
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
                               InDataBlockTransfer& in_blockwise_copy,
                               const InDataGridBuffer& in_grid_buf,
                               InDataBlockBuffer& in_block_buf,
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
        constexpr auto wei_block_copy_step = to_multi_index(WeiDataBlockTransferStep{});
        constexpr auto in_block_copy_step  = to_multi_index(InDataBlockTransferStep{});
        constexpr index_t NumTap           = WeiDataBlockTransfer::Size();
        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0])>>;
        using EmptySemas = Tuple<>;
        EmptySemas emptySemas;
        constexpr index_t NumDs = DsDataBlockTransfer::Size();

        // preload ds data into LDS
        static_for<0, NumDs, 1>{}([&](auto i) {
            using DDataBlockTransfer =
                std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
            const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                .RunRead(ds_grid_desc[i], ds_grid_buf[i]);
        });

        // preload data into LDS
        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                .RunRead(wei_grid_desc, wei_grid_buf);
        });

        in_blockwise_copy.RunRead(in_grid_desc, in_grid_buf);

        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
        });

        in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

        // Initialize C
        if constexpr(HasMainLoop)
        {
            accum_thread_buf.Clear();
        }

        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                .RunWrite(wei_block_desc, wei_block_buf);
        });

        in_blockwise_copy.RunWrite(in_block_desc, in_block_buf);

        static_for<0, NumDs, 1>{}([&](auto i) {
            using DDataBlockTransfer =
                std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
            const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                .RunWrite(ds_block_desc[i], ds_block_buf(i));
        });

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .RunRead(wei_grid_desc, wei_grid_buf);
                });

                block_sync_lds();

                in_blockwise_copy.RunRead(in_grid_desc, in_grid_buf);

                blockwise_conv.Run(wei_block_buf,
                                   in_block_buf,
                                   nullptr,
                                   nullptr,
                                   ds_block_buf,
                                   accum_thread_buf,
                                   out_thread_buf,
                                   emptySemas,
                                   Number<HasMainLoop>{},
                                   Number<false>{});

                block_sync_lds();

                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                });

                in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .RunWrite(wei_block_desc, wei_block_buf);
                });

                in_blockwise_copy.RunWrite(in_block_desc, in_block_buf);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_conv.Run(wei_block_buf,
                               in_block_buf,
                               nullptr,
                               nullptr,
                               ds_block_buf,
                               accum_thread_buf,
                               out_thread_buf,
                               emptySemas,
                               Number<HasMainLoop>{},
                               Number<true>{});
        }
    }
};

template <>
struct GridwiseConvPipeline_v1<1, false, true, false, false>
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
              typename InDataBlockTransfer,
              typename InDataGridBuffer,
              typename InDataBlockBuffer,
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
                               InDataBlockTransfer& in_blockwise_copy,
                               const InDataGridBuffer& in_grid_buf,
                               InDataBlockBuffer& in_block_buf,
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
        constexpr auto wei_block_copy_step = to_multi_index(WeiDataBlockTransferStep{});
        constexpr auto in_block_copy_step  = to_multi_index(InDataBlockTransferStep{});
        constexpr index_t NumTap           = WeiDataBlockTransfer::Size();
        constexpr auto in_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
        auto in_block_buf_switch           = in_block_buf;

        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0])>>;
        using EmptySemas = Tuple<>;
        EmptySemas emptySemas;

        // preload data into LDS
        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                .RunRead(wei_grid_desc, wei_grid_buf);
        });

        constexpr index_t NumDs = DsDataBlockTransfer::Size();

        static_for<0, NumDs, 1>{}([&](auto i) {
            using DDataBlockTransfer =
                std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
            using DBlockDesc = remove_cvref_t<decltype(ds_block_desc[i])>;
            auto d_block_origin_idx =
                generate_tuple([&](auto) { return I0; }, Number<DBlockDesc::GetNumOfDimension()>{});
            const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                .Run(ds_grid_desc[i],
                     ds_grid_buf[i],
                     ds_block_desc[i],
                     d_block_origin_idx,
                     ds_block_buf(i));
        });

        in_blockwise_copy.Run(
            in_grid_desc, in_grid_buf, in_block_desc, in_block_origin_idx, in_block_buf);

        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
        });

        in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

        // Initialize C
        if constexpr(HasMainLoop)
        {
            accum_thread_buf.Clear();
        }

        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                .RunWrite(wei_block_desc, wei_block_buf);
        });

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                in_blockwise_copy.Run(in_grid_desc,
                                      in_grid_buf,
                                      in_block_desc,
                                      in_block_origin_idx,
                                      in_block_buf_switch);

                block_sync_lds();

                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .RunRead(wei_grid_desc, wei_grid_buf);
                });

                blockwise_conv.Run(wei_block_buf,
                                   in_block_buf,
                                   nullptr,
                                   nullptr,
                                   ds_block_buf,
                                   accum_thread_buf,
                                   out_thread_buf,
                                   emptySemas,
                                   Number<HasMainLoop>{},
                                   Number<false>{});

                block_sync_lds();

                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                });

                in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .RunWrite(wei_block_desc, wei_block_buf);
                });

                in_block_buf = in_block_buf_switch;
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_conv.Run(wei_block_buf,
                               in_block_buf,
                               nullptr,
                               nullptr,
                               ds_block_buf,
                               accum_thread_buf,
                               out_thread_buf,
                               emptySemas,
                               Number<HasMainLoop>{},
                               Number<true>{});
        }
    }
};

template <bool EnableAsync>
struct GridwiseConvPipeline_v1<1, false, false, false, EnableAsync>
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
              typename InDataBlockTransfer,
              typename InDataGridBuffer,
              typename InDataBlockBuffer,
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
                               InDataBlockTransfer& in_blockwise_copy,
                               const InDataGridBuffer& in_grid_buf,
                               InDataBlockBuffer& in_block_buf,
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
        constexpr auto wei_block_copy_step = to_multi_index(WeiDataBlockTransferStep{});
        constexpr auto in_block_copy_step  = to_multi_index(InDataBlockTransferStep{});
        constexpr index_t NumTap           = WeiDataBlockTransfer::Size();
        constexpr auto in_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
        auto in_block_buf_switch           = in_block_buf;
        auto wei_block_buf_switch          = wei_block_buf;
        constexpr auto wei_remap_table     = BlockwiseConv::GetWeightRemapTable();

        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0])>>;
        using EmptySemas = Tuple<>;
        EmptySemas emptySemas;

        // preload data into LDS
        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                .Run(wei_grid_desc,
                     wei_grid_buf,
                     wei_block_desc,
                     make_tuple(I0, I0, wei_remap_table[tapIdx], I0, I0, I0),
                     wei_block_buf);
        });

        constexpr index_t NumDs = DsDataBlockTransfer::Size();

        static_for<0, NumDs, 1>{}([&](auto i) {
            using DDataBlockTransfer =
                std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
            using DBlockDesc = remove_cvref_t<decltype(ds_block_desc[i])>;
            auto d_block_origin_idx =
                generate_tuple([&](auto) { return I0; }, Number<DBlockDesc::GetNumOfDimension()>{});
            const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                .Run(ds_grid_desc[i],
                     ds_grid_buf[i],
                     ds_block_desc[i],
                     d_block_origin_idx,
                     ds_block_buf(i));
        });

        in_blockwise_copy.Run(
            in_grid_desc, in_grid_buf, in_block_desc, in_block_origin_idx, in_block_buf);

        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
        });

        in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

        // Initialize C
        if constexpr(HasMainLoop)
        {
            accum_thread_buf.Clear();
        }

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .Run(wei_grid_desc,
                             wei_grid_buf,
                             wei_block_desc,
                             make_tuple(I0, I0, wei_remap_table[tapIdx], I0, I0, I0),
                             wei_block_buf_switch);
                });
                in_blockwise_copy.Run(in_grid_desc,
                                      in_grid_buf,
                                      in_block_desc,
                                      in_block_origin_idx,
                                      in_block_buf_switch);

                blockwise_conv.Run(wei_block_buf,
                                   in_block_buf,
                                   nullptr,
                                   nullptr,
                                   ds_block_buf,
                                   accum_thread_buf,
                                   out_thread_buf,
                                   emptySemas,
                                   Number<HasMainLoop>{},
                                   Number<false>{});

                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                });

                in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

                in_block_buf  = in_block_buf_switch;
                wei_block_buf = wei_block_buf_switch;
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            blockwise_conv.Run(wei_block_buf,
                               in_block_buf,
                               nullptr,
                               nullptr,
                               ds_block_buf,
                               accum_thread_buf,
                               out_thread_buf,
                               emptySemas,
                               Number<HasMainLoop>{},
                               Number<true>{});
        }
    }
};

template <>
struct GridwiseConvPipeline_v1<1, true, false, true, false>
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
              typename InDataBlockTransfer,
              typename InDataGridBuffer,
              typename InDataBlockBuffer,
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
                               InDataBlockTransfer& in_blockwise_copy,
                               const InDataGridBuffer& in_grid_buf,
                               InDataBlockBuffer& in_block_buf,
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
        constexpr auto wei_block_copy_step = to_multi_index(WeiDataBlockTransferStep{});
        constexpr auto in_block_copy_step  = to_multi_index(InDataBlockTransferStep{});

        constexpr index_t NumTap       = WeiDataBlockTransfer::Size();
        constexpr auto wei_remap_table = BlockwiseConv::GetWeightRemapTable();
        auto wei_block_buf_switch      = wei_block_buf;

        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0])>>;
        using EmptySemas = Tuple<>;
        EmptySemas emptySemas;
        // preload data into LDS
        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                .Run(wei_grid_desc,
                     wei_grid_buf,
                     wei_block_desc,
                     make_tuple(I0, I0, wei_remap_table[tapIdx], I0, I0, I0),
                     wei_block_buf);
        });

        constexpr index_t NumDs = DsDataBlockTransfer::Size();
        static_for<0, NumDs, 1>{}([&](auto i) {
            using DDataBlockTransfer =
                std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
            const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                .RunRead(ds_grid_desc[i], ds_grid_buf[i]);
        });

        in_blockwise_copy.RunRead(in_grid_desc, in_grid_buf);

        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
        });

        in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

        // Initialize C
        if constexpr(HasMainLoop)
        {
            accum_thread_buf.Clear();
        }

        in_blockwise_copy.RunWrite(in_block_desc, in_block_buf);

        static_for<0, NumDs, 1>{}([&](auto i) {
            using DDataBlockTransfer =
                std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
            const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                .RunWrite(ds_block_desc[i], ds_block_buf(i));
        });

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .Run(wei_grid_desc,
                             wei_grid_buf,
                             wei_block_desc,
                             make_tuple(I0, I0, wei_remap_table[tapIdx], I0, I0, I0),
                             wei_block_buf_switch);
                });

                block_sync_lds();

                in_blockwise_copy.RunRead(in_grid_desc, in_grid_buf);

                blockwise_conv.Run(wei_block_buf,
                                   in_block_buf,
                                   nullptr,
                                   nullptr,
                                   ds_block_buf,
                                   accum_thread_buf,
                                   out_thread_buf,
                                   emptySemas,
                                   Number<HasMainLoop>{},
                                   Number<false>{});

                block_sync_lds();

                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                });

                in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

                in_blockwise_copy.RunWrite(in_block_desc, in_block_buf);

                wei_block_buf = wei_block_buf_switch;
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_conv.Run(wei_block_buf,
                               in_block_buf,
                               nullptr,
                               nullptr,
                               ds_block_buf,
                               accum_thread_buf,
                               out_thread_buf,
                               emptySemas,
                               Number<HasMainLoop>{},
                               Number<true>{});
        }
    }
};

// Async load
// TODO: support prefetch
template <>
struct GridwiseConvPipeline_v1<1, true, true, true, true>
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
              typename InDataBlockTransfer,
              typename InDataGridBuffer,
              typename InDataBlockBuffer,
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
                               InDataBlockTransfer& in_blockwise_copy,
                               const InDataGridBuffer& in_grid_buf,
                               InDataBlockBuffer& in_block_buf,
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
        constexpr auto wei_block_copy_step = to_multi_index(WeiDataBlockTransferStep{});
        constexpr auto in_block_copy_step  = to_multi_index(InDataBlockTransferStep{});
        constexpr index_t NumTap           = WeiDataBlockTransfer::Size();
        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0])>>;
        using EmptySemas = Tuple<>;
        EmptySemas emptySemas;
        // Initialize C
        if constexpr(HasMainLoop)
        {
            accum_thread_buf.Clear();
        }

        // preload data into LDS
        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                .Run(wei_grid_desc, wei_grid_buf, wei_block_desc, wei_block_buf);
        });

        in_blockwise_copy.Run(in_grid_desc, in_grid_buf, in_block_desc, in_block_buf);

        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
        });
        in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

        constexpr index_t NumDs = DsDataBlockTransfer::Size();

        static_for<0, NumDs, 1>{}([&](auto i) {
            using DDataBlockTransfer =
                std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
            const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                .Run(ds_grid_desc[i], ds_grid_buf[i], ds_block_desc[i], ds_block_buf(i));
        });

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                // do convolution
                block_sync_lds_async_load();

                blockwise_conv.Run(wei_block_buf,
                                   in_block_buf,
                                   nullptr,
                                   nullptr,
                                   ds_block_buf,
                                   accum_thread_buf,
                                   out_thread_buf,
                                   emptySemas,
                                   Number<HasMainLoop>{},
                                   Number<false>{});

                block_sync_lds();

                // copy data
                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .Run(wei_grid_desc, wei_grid_buf, wei_block_desc, wei_block_buf);
                });

                in_blockwise_copy.Run(in_grid_desc, in_grid_buf, in_block_desc, in_block_buf);

                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                });

                in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds_async_load();

            blockwise_conv.Run(wei_block_buf,
                               in_block_buf,
                               nullptr,
                               nullptr,
                               ds_block_buf,
                               accum_thread_buf,
                               out_thread_buf,
                               emptySemas,
                               Number<HasMainLoop>{},
                               Number<true>{});
        }
    }
};

template <>
struct GridwiseConvPipeline_v1<1, false, true, false, true>
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
              typename InDataBlockTransfer,
              typename InDataGridBuffer,
              typename InDataBlockBuffer,
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
                               InDataBlockTransfer& in_blockwise_copy,
                               const InDataGridBuffer& in_grid_buf,
                               InDataBlockBuffer& in_block_buf,
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
        constexpr auto wei_block_copy_step = to_multi_index(WeiDataBlockTransferStep{});
        constexpr auto in_block_copy_step  = to_multi_index(InDataBlockTransferStep{});
        constexpr index_t NumTap           = WeiDataBlockTransfer::Size();
        constexpr auto in_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
        auto in_block_buf_switch           = in_block_buf;

        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0])>>;
        using EmptySemas = Tuple<>;
        EmptySemas emptySemas;

        // Initialize C
        if constexpr(HasMainLoop)
        {
            accum_thread_buf.Clear();
        }

        // Load data
        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                .Run(wei_grid_desc, wei_grid_buf, wei_block_desc, wei_block_buf);
        });

        in_blockwise_copy.Run(
            in_grid_desc, in_grid_buf, in_block_desc, in_block_origin_idx, in_block_buf);

        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
        });
        in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

        constexpr index_t NumDs = DsDataBlockTransfer::Size();

        static_for<0, NumDs, 1>{}([&](auto i) {
            using DDataBlockTransfer =
                std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
            using DBlockDesc = remove_cvref_t<decltype(ds_block_desc[i])>;
            auto d_block_origin_idx =
                generate_tuple([&](auto) { return I0; }, Number<DBlockDesc::GetNumOfDimension()>{});
            const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                .Run(ds_grid_desc[i],
                     ds_grid_buf[i],
                     ds_block_desc[i],
                     d_block_origin_idx,
                     ds_block_buf(i));
        });

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                block_sync_lds_async_load();

                in_blockwise_copy.Run(in_grid_desc,
                                      in_grid_buf,
                                      in_block_desc,
                                      in_block_origin_idx,
                                      in_block_buf_switch);
                in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

                blockwise_conv.Run(wei_block_buf,
                                   in_block_buf,
                                   nullptr,
                                   nullptr,
                                   ds_block_buf,
                                   accum_thread_buf,
                                   out_thread_buf,
                                   emptySemas,
                                   Number<HasMainLoop>{},
                                   Number<false>{});

                block_sync_lds();

                in_block_buf = in_block_buf_switch;

                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .Run(wei_grid_desc, wei_grid_buf, wei_block_desc, wei_block_buf);
                });

                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                });

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds_async_load();

            blockwise_conv.Run(wei_block_buf,
                               in_block_buf,
                               nullptr,
                               nullptr,
                               ds_block_buf,
                               accum_thread_buf,
                               out_thread_buf,
                               emptySemas,
                               Number<HasMainLoop>{},
                               Number<true>{});
        }
    }
};

template <>
struct GridwiseConvPipeline_v1<1, true, false, true, true>
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
              typename InDataBlockTransfer,
              typename InDataGridBuffer,
              typename InDataBlockBuffer,
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
                               InDataBlockTransfer& in_blockwise_copy,
                               const InDataGridBuffer& in_grid_buf,
                               InDataBlockBuffer& in_block_buf,
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
        constexpr auto wei_block_copy_step = to_multi_index(WeiDataBlockTransferStep{});
        constexpr auto in_block_copy_step  = to_multi_index(InDataBlockTransferStep{});
        constexpr index_t NumTap           = WeiDataBlockTransfer::Size();
        constexpr auto wei_remap_table     = BlockwiseConv::GetWeightRemapTable();
        auto wei_block_buf_switch          = wei_block_buf;

        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0])>>;
        using EmptySemas = Tuple<>;
        EmptySemas emptySemas;
        // Initialize C
        if constexpr(HasMainLoop)
        {
            accum_thread_buf.Clear();
        }

        // Load data
        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                .Run(wei_grid_desc,
                     wei_grid_buf,
                     wei_block_desc,
                     make_tuple(I0, I0, wei_remap_table[tapIdx], I0, I0, I0),
                     wei_block_buf);
        });

        in_blockwise_copy.Run(in_grid_desc, in_grid_buf, in_block_desc, in_block_buf);

        static_for<0, NumTap, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
        });
        in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

        constexpr index_t NumDs = DsDataBlockTransfer::Size();

        static_for<0, NumDs, 1>{}([&](auto i) {
            using DDataBlockTransfer =
                std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[i])>>;
            const_cast<DDataBlockTransfer&>(ds_blockwise_copy[i])
                .Run(ds_grid_desc[i], ds_grid_buf[i], ds_block_desc[i], ds_block_buf(i));
        });

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                block_sync_lds_async_load();

                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .Run(wei_grid_desc,
                             wei_grid_buf,
                             wei_block_desc,
                             make_tuple(I0, I0, wei_remap_table[tapIdx], I0, I0, I0),
                             wei_block_buf_switch);
                });
                static_for<0, NumTap, 1>{}([&](auto tapIdx) {
                    const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[tapIdx])
                        .MoveSrcSliceWindow(wei_grid_desc, wei_block_copy_step);
                });

                blockwise_conv.Run(wei_block_buf,
                                   in_block_buf,
                                   nullptr,
                                   nullptr,
                                   ds_block_buf,
                                   accum_thread_buf,
                                   out_thread_buf,
                                   emptySemas,
                                   Number<HasMainLoop>{},
                                   Number<false>{});

                wei_block_buf = wei_block_buf_switch;

                block_sync_lds();

                in_blockwise_copy.Run(in_grid_desc, in_grid_buf, in_block_desc, in_block_buf);

                in_blockwise_copy.MoveSrcSliceWindow(in_grid_desc, in_block_copy_step);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_conv.Run(wei_block_buf,
                               in_block_buf,
                               nullptr,
                               nullptr,
                               ds_block_buf,
                               accum_thread_buf,
                               out_thread_buf,
                               emptySemas,
                               Number<HasMainLoop>{},
                               Number<true>{});
        }
    }
};

template <index_t NumPrefetch,
          bool InDataEnableLds,
          bool WeiDataEnableLds,
          bool DsDataEnableLds,
          bool AsyncLoad,
          bool WaveGroup,
          bool SpatialCluster>
constexpr auto GridwiseConvPipeline_Selector()
{
    if constexpr(WaveGroup)
    {
        return GridwiseConvPipeline_v2<NumPrefetch,
                                       InDataEnableLds,
                                       WeiDataEnableLds,
                                       DsDataEnableLds,
                                       AsyncLoad,
                                       SpatialCluster>{};
    }
    else
    {
        return GridwiseConvPipeline_v1<NumPrefetch,
                                       InDataEnableLds,
                                       WeiDataEnableLds,
                                       DsDataEnableLds,
                                       AsyncLoad>{};
    }
}

} // namespace ck
