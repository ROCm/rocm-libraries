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
struct GridwiseFasternet50Pipeline_v1
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

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
              typename AccumThreadBuffer0,
              typename AccumThreadBuffer1,
              typename AccumThreadBuffer2,
              typename OutThreadBuffer0,
              typename OutThreadBuffer1,
              typename OutThreadBuffer2>
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
                               AccumThreadBuffer0& accum0_thread_buf,
                               AccumThreadBuffer1& accum1_thread_buf,
                               AccumThreadBuffer2& accum2_thread_buf,
                               OutThreadBuffer0& out0_thread_buf,
                               OutThreadBuffer1& out1_thread_buf,
                               OutThreadBuffer2& out2_thread_buf,
                               const std::array<index_t, 3>& num_loop)
    {
        constexpr auto wei_0_block_copy_step = to_multi_index(WeiDataBlockTransferStep{}[I0]);
        constexpr auto wei_1_block_copy_step = to_multi_index(WeiDataBlockTransferStep{}[I1]);
        constexpr auto wei_2_block_copy_step = to_multi_index(WeiDataBlockTransferStep{}[I2]);

        (void)num_loop;

        constexpr index_t NumTap_0 =
            remove_cvref_t<tuple_element_t<0, WeiDataBlockTransfer>>::Size();
        constexpr index_t NumTap_1 =
            remove_cvref_t<tuple_element_t<1, WeiDataBlockTransfer>>::Size();
        constexpr index_t NumTap_2 =
            remove_cvref_t<tuple_element_t<2, WeiDataBlockTransfer>>::Size();
        using WeiDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I0][I0])>>;
        using WeiDataBlockTransfer1 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I1][I0])>>;
        using WeiDataBlockTransfer2 =
            std::remove_const_t<remove_cvref_t<decltype(wei_blockwise_copy[I2][I0])>>;
        using WeiBlockBuf0 = std::remove_const_t<remove_cvref_t<decltype(wei_block_buf[I0])>>;
        using WeiBlockBuf1 = std::remove_const_t<remove_cvref_t<decltype(wei_block_buf[I1])>>;
        using WeiBlockBuf2 = std::remove_const_t<remove_cvref_t<decltype(wei_block_buf[I2])>>;

        using InDataBlockTransfer0 =
            std::remove_const_t<remove_cvref_t<decltype(in_blockwise_copy[I0])>>;
        using InDataBlockTransfer1 =
            std::remove_const_t<remove_cvref_t<decltype(in_blockwise_copy[I1])>>;
        using InBlockBuf0 = std::remove_const_t<remove_cvref_t<decltype(in_block_buf[I0])>>;
        using InBlockBuf1 = std::remove_const_t<remove_cvref_t<decltype(in_block_buf[I1])>>;

        using DsBlockBuf0 = std::remove_const_t<remove_cvref_t<decltype(ds_block_buf[I0])>>;
        using DsBlockBuf1 = std::remove_const_t<remove_cvref_t<decltype(ds_block_buf[I1])>>;
        using DsBlockBuf2 = std::remove_const_t<remove_cvref_t<decltype(ds_block_buf[I2])>>;

        auto wei_0_block_buf = const_cast<WeiBlockBuf0&>(wei_block_buf[I0]);
        auto wei_1_block_buf = const_cast<WeiBlockBuf1&>(wei_block_buf[I1]);
        auto wei_2_block_buf = const_cast<WeiBlockBuf2&>(wei_block_buf[I2]);

        auto in_0_block_buf = const_cast<InBlockBuf0&>(in_block_buf[I0]);
        auto in_1_block_buf = const_cast<InBlockBuf1&>(in_block_buf[I1]);

        auto ds_0_block_buf = const_cast<DsBlockBuf0&>(ds_block_buf[I0]);
        auto ds_1_block_buf = const_cast<DsBlockBuf1&>(ds_block_buf[I1]);
        auto ds_2_block_buf = const_cast<DsBlockBuf2&>(ds_block_buf[I2]);

        using EmptySemas = Tuple<>;
        EmptySemas emptySemas;

        constexpr index_t NumDs_0 = remove_cvref_t<tuple_element_t<0, DsDataBlockTransfer>>::Size();
        constexpr index_t NumDs_1 = remove_cvref_t<tuple_element_t<1, DsDataBlockTransfer>>::Size();
        constexpr index_t NumDs_2 = remove_cvref_t<tuple_element_t<2, DsDataBlockTransfer>>::Size();

        static_assert(NumDs_0 == 0, "The 1st phase should have no D tensor");
        static_assert(NumDs_2 == 0, "The 3rd phase should have no D tensor");

        // Phase 0 for 1x1 conv:
        if constexpr(WeiDataEnableLds && EnableAsync == false)
        {
            static_for<0, NumTap_0, 1>{}([&](auto tapIdx) {
                const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[I0][tapIdx])
                    .RunRead(wei_grid_desc[I0], wei_grid_buf[I0]);
            });
        }
        else if constexpr(WeiDataEnableLds && EnableAsync)
        {
            static_for<0, NumTap_0, 1>{}([&](auto tapIdx) {
                const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[I0][tapIdx])
                    .Run(wei_grid_desc[I0], wei_grid_buf[I0], wei_block_desc[I0], wei_0_block_buf);
            });
        }
        else
        {
            constexpr auto wei_remap_table =
                remove_cvref_t<tuple_element_t<0, BlockwiseConv>>::GetWeightRemapTable();
            // preload data into LDS
            static_for<0, NumTap_0, 1>{}([&](auto tapIdx) {
                const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[I0][tapIdx])
                    .Run(wei_grid_desc[I0],
                         wei_grid_buf[I0],
                         wei_block_desc[I0],
                         make_tuple(I0, I0, wei_remap_table[tapIdx], I0, I0, I0),
                         wei_0_block_buf);
            });
        }

        if constexpr(InDataEnableLds && EnableAsync == false)
        {
            const_cast<InDataBlockTransfer0&>(in_blockwise_copy[I0])
                .RunRead(in_grid_desc[I0], in_grid_buf[I0]);
        }
        else if constexpr(InDataEnableLds && EnableAsync)
        {
            const_cast<InDataBlockTransfer0&>(in_blockwise_copy[I0])
                .Run(in_grid_desc[I0], in_grid_buf[I0], in_block_desc[I0], in_0_block_buf);
        }
        else
        {
            constexpr auto in_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
            const_cast<InDataBlockTransfer0&>(in_blockwise_copy[I0])
                .Run(in_grid_desc[I0],
                     in_grid_buf[I0],
                     in_block_desc[I0],
                     in_block_origin_idx,
                     in_0_block_buf);
        }

        static_for<0, NumTap_0, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[I0][tapIdx])
                .MoveSrcSliceWindow(wei_grid_desc[I0], wei_0_block_copy_step);
        });

        if constexpr(WeiDataEnableLds && EnableAsync == false)
        {
            static_for<0, NumTap_0, 1>{}([&](auto tapIdx) {
                const_cast<WeiDataBlockTransfer0&>(wei_blockwise_copy[I0][tapIdx])
                    .RunWrite(wei_block_desc[I0], wei_0_block_buf);
            });
        }
        if constexpr(InDataEnableLds && EnableAsync == false)
        {
            const_cast<InDataBlockTransfer0&>(in_blockwise_copy[I0])
                .RunWrite(in_block_desc[I0], in_0_block_buf);
        }

        if constexpr(NumDs_0 > 0)
        {
            if constexpr(DsDataEnableLds && EnableAsync == false)
            {
                static_for<0, NumDs_0, 1>{}([&](auto i) {
                    using DsDataBlockTransfer0 =
                        std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[I0][I0])>>;
                    const_cast<DsDataBlockTransfer0&>(ds_blockwise_copy[I0][i])
                        .RunRead(ds_grid_desc[I0][i], ds_grid_buf[I0][i]);
                });
            }
            else if constexpr(DsDataEnableLds && EnableAsync)
            {
                static_for<0, NumDs_0, 1>{}([&](auto i) {
                    using DsDataBlockTransfer0 =
                        std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[I0][I0])>>;
                    const_cast<DsDataBlockTransfer0&>(ds_blockwise_copy[I0][i])
                        .Run(ds_grid_desc[I0][i],
                             ds_grid_buf[I0][i],
                             ds_block_desc[I0][i],
                             ds_0_block_buf(i));
                });
            }
            else
            {
                static_for<0, NumDs_0, 1>{}([&](auto i) {
                    using DDataBlockTransfer =
                        std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[I0][i])>>;
                    using DBlockDesc        = remove_cvref_t<decltype(ds_block_desc[I0][i])>;
                    auto d_block_origin_idx = generate_tuple(
                        [&](auto) { return I0; }, Number<DBlockDesc::GetNumOfDimension()>{});
                    const_cast<DDataBlockTransfer&>(ds_blockwise_copy[I0][i])
                        .Run(ds_grid_desc[I0][i],
                             ds_grid_buf[I0][i],
                             ds_block_desc[I0][i],
                             d_block_origin_idx,
                             ds_0_block_buf(i));
                });
            }

            if constexpr(DsDataEnableLds && EnableAsync == false)
            {
                static_for<0, NumDs_0, 1>{}([&](auto i) {
                    using DsDataBlockTransfer0 =
                        std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[I0][i])>>;
                    const_cast<DsDataBlockTransfer0&>(ds_blockwise_copy[I0][i])
                        .RunWrite(ds_block_desc[I0][i], ds_0_block_buf(i));
                });
            }
        }

        if constexpr(EnableAsync)
        {
            block_sync_lds_async_load();
        }
        else
        {
            block_sync_lds();
        }

        blockwise_conv[I0].template Run<0>(wei_0_block_buf,
                                           in_0_block_buf,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           ds_0_block_buf,
                                           accum0_thread_buf,
                                           out0_thread_buf,
                                           emptySemas,
                                           Number<HasMainLoop>{},
                                           Number<true>{});
        block_sync_lds();

        // Phase 1 for 1x1 conv:
        // preload ds data into LDS
        if constexpr(NumDs_1 > 0)
        {
            if constexpr(DsDataEnableLds && EnableAsync == false)
            {
                static_for<0, NumDs_1, 1>{}([&](auto i) {
                    using DsDataBlockTransfer1 =
                        std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[I1][I0])>>;
                    const_cast<DsDataBlockTransfer1&>(ds_blockwise_copy[I1][i])
                        .RunRead(ds_grid_desc[I1][i], ds_grid_buf[I1][i]);
                });
            }
            else if constexpr(DsDataEnableLds && EnableAsync)
            {
                static_for<0, NumDs_1, 1>{}([&](auto i) {
                    using DsDataBlockTransfer1 =
                        std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[I1][I0])>>;
                    const_cast<DsDataBlockTransfer1&>(ds_blockwise_copy[I1][i])
                        .Run(ds_grid_desc[I1][i],
                             ds_grid_buf[I1][i],
                             ds_block_desc[I1][i],
                             ds_1_block_buf(i));
                });
            }
            else
            {
                static_for<0, NumDs_1, 1>{}([&](auto i) {
                    using DDataBlockTransfer =
                        std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[I1][i])>>;
                    using DBlockDesc        = remove_cvref_t<decltype(ds_block_desc[I1][i])>;
                    auto d_block_origin_idx = generate_tuple(
                        [&](auto) { return I0; }, Number<DBlockDesc::GetNumOfDimension()>{});
                    const_cast<DDataBlockTransfer&>(ds_blockwise_copy[I1][i])
                        .Run(ds_grid_desc[I1][i],
                             ds_grid_buf[I1][i],
                             ds_block_desc[I1][i],
                             d_block_origin_idx,
                             ds_1_block_buf(i));
                });
            }
        }

        if constexpr(WeiDataEnableLds && EnableAsync == false)
        {
            static_for<0, NumTap_1, 1>{}([&](auto tapIdx) {
                const_cast<WeiDataBlockTransfer1&>(wei_blockwise_copy[I1][tapIdx])
                    .RunRead(wei_grid_desc[I1], wei_grid_buf[I1]);
            });
        }
        else if constexpr(WeiDataEnableLds && EnableAsync)
        {
            static_for<0, NumTap_1, 1>{}([&](auto tapIdx) {
                const_cast<WeiDataBlockTransfer1&>(wei_blockwise_copy[I1][tapIdx])
                    .Run(wei_grid_desc[I1], wei_grid_buf[I1], wei_block_desc[I1], wei_1_block_buf);
            });
        }
        else
        {
            constexpr auto wei_remap_table =
                remove_cvref_t<tuple_element_t<1, BlockwiseConv>>::GetWeightRemapTable();
            // preload data into LDS
            static_for<0, NumTap_1, 1>{}([&](auto tapIdx) {
                const_cast<WeiDataBlockTransfer1&>(wei_blockwise_copy[I1][tapIdx])
                    .Run(wei_grid_desc[I1],
                         wei_grid_buf[I1],
                         wei_block_desc[I1],
                         make_tuple(I0, I0, wei_remap_table[tapIdx], I0, I0, I0),
                         wei_1_block_buf);
            });
        }

        static_for<0, NumTap_1, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer1&>(wei_blockwise_copy[I1][tapIdx])
                .MoveSrcSliceWindow(wei_grid_desc[I1], wei_1_block_copy_step);
        });

        if constexpr(InDataEnableLds && EnableAsync == false)
        {
            const_cast<InDataBlockTransfer1&>(in_blockwise_copy[I1])
                .RunRead(in_grid_desc[I1], in_grid_buf[I1]);
        }
        else if constexpr(InDataEnableLds && EnableAsync)
        {
            const_cast<InDataBlockTransfer1&>(in_blockwise_copy[I1])
                .Run(in_grid_desc[I1], in_grid_buf[I1], in_block_desc[I1], in_1_block_buf);
        }
        else
        {
            constexpr auto in_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
            const_cast<InDataBlockTransfer1&>(in_blockwise_copy[I1])
                .Run(in_grid_desc[I1],
                     in_grid_buf[I1],
                     in_block_desc[I1],
                     in_block_origin_idx,
                     in_1_block_buf);
        }

        if constexpr(WeiDataEnableLds && EnableAsync == false)
        {
            static_for<0, NumTap_1, 1>{}([&](auto tapIdx) {
                const_cast<WeiDataBlockTransfer1&>(wei_blockwise_copy[I1][tapIdx])
                    .RunWrite(wei_block_desc[I1], wei_1_block_buf);
            });
        }

        if constexpr(InDataEnableLds && EnableAsync == false)
        {
            const_cast<InDataBlockTransfer1&>(in_blockwise_copy[I1])
                .RunWrite(in_block_desc[I1], in_1_block_buf);
        }

        if constexpr(NumDs_1 > 0)
        {
            if constexpr(DsDataEnableLds && EnableAsync == false)
            {
                static_for<0, NumDs_1, 1>{}([&](auto i) {
                    using DsDataBlockTransfer1 =
                        std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[I1][I0])>>;
                    const_cast<DsDataBlockTransfer1&>(ds_blockwise_copy[I1][i])
                        .RunWrite(ds_block_desc[I1][i], ds_1_block_buf(i));
                });
            }
        }

        // main body
        if constexpr(EnableAsync)
        {
            block_sync_lds_async_load();
        }
        else
        {
            block_sync_lds();
        }

        blockwise_conv[I1].template Run<1>(wei_1_block_buf,
                                           in_1_block_buf,
                                           out0_thread_buf,
                                           nullptr,
                                           nullptr,
                                           ds_1_block_buf,
                                           accum1_thread_buf,
                                           out1_thread_buf,
                                           emptySemas,
                                           Number<HasMainLoop>{},
                                           Number<true>{});
        block_sync_lds();

        // Phase 2 for 1x1 conv:
        // preload ds data into LDS
        if constexpr(NumDs_2 > 0)
        {
            if constexpr(DsDataEnableLds && EnableAsync == false)
            {
                static_for<0, NumDs_2, 1>{}([&](auto i) {
                    using DsDataBlockTransfer2 =
                        std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[I2][I0])>>;
                    const_cast<DsDataBlockTransfer2&>(ds_blockwise_copy[I2][i])
                        .RunRead(ds_grid_desc[I2][i], ds_grid_buf[I2][i]);
                });
            }
            else if constexpr(DsDataEnableLds && EnableAsync)
            {
                static_for<0, NumDs_2, 1>{}([&](auto i) {
                    using DsDataBlockTransfer2 =
                        std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[I2][I0])>>;
                    const_cast<DsDataBlockTransfer2&>(ds_blockwise_copy[I2][i])
                        .Run(ds_grid_desc[I2][i],
                             ds_grid_buf[I2][i],
                             ds_block_desc[I2][i],
                             ds_2_block_buf(i));
                });
            }
            else
            {
                static_for<0, NumDs_2, 1>{}([&](auto i) {
                    using DDataBlockTransfer =
                        std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[I2][i])>>;
                    using DBlockDesc        = remove_cvref_t<decltype(ds_block_desc[I2][i])>;
                    auto d_block_origin_idx = generate_tuple(
                        [&](auto) { return I0; }, Number<DBlockDesc::GetNumOfDimension()>{});
                    const_cast<DDataBlockTransfer&>(ds_blockwise_copy[I2][i])
                        .Run(ds_grid_desc[I2][i],
                             ds_grid_buf[I2][i],
                             ds_block_desc[I2][i],
                             d_block_origin_idx,
                             ds_2_block_buf(i));
                });
            }
        }

        if constexpr(WeiDataEnableLds && EnableAsync == false)
        {
            static_for<0, NumTap_2, 1>{}([&](auto tapIdx) {
                const_cast<WeiDataBlockTransfer2&>(wei_blockwise_copy[I2][tapIdx])
                    .RunRead(wei_grid_desc[I2], wei_grid_buf[I2]);
            });
        }
        else if constexpr(WeiDataEnableLds && EnableAsync)
        {
            static_for<0, NumTap_2, 1>{}([&](auto tapIdx) {
                const_cast<WeiDataBlockTransfer2&>(wei_blockwise_copy[I2][tapIdx])
                    .Run(wei_grid_desc[I2], wei_grid_buf[I2], wei_block_desc[I2], wei_2_block_buf);
            });
        }
        else
        {
            constexpr auto wei_remap_table =
                remove_cvref_t<tuple_element_t<2, BlockwiseConv>>::GetWeightRemapTable();
            // preload data into LDS
            static_for<0, NumTap_2, 1>{}([&](auto tapIdx) {
                const_cast<WeiDataBlockTransfer2&>(wei_blockwise_copy[I2][tapIdx])
                    .Run(wei_grid_desc[I2],
                         wei_grid_buf[I2],
                         wei_block_desc[I2],
                         make_tuple(I0, I0, wei_remap_table[tapIdx], I0, I0, I0),
                         wei_2_block_buf);
            });
        }

        static_for<0, NumTap_2, 1>{}([&](auto tapIdx) {
            const_cast<WeiDataBlockTransfer2&>(wei_blockwise_copy[I2][tapIdx])
                .MoveSrcSliceWindow(wei_grid_desc[I2], wei_2_block_copy_step);
        });

        if constexpr(WeiDataEnableLds && EnableAsync == false)
        {
            static_for<0, NumTap_2, 1>{}([&](auto tapIdx) {
                const_cast<WeiDataBlockTransfer2&>(wei_blockwise_copy[I2][tapIdx])
                    .RunWrite(wei_block_desc[I2], wei_2_block_buf);
            });
        }

        if constexpr(NumDs_2 > 0)
        {
            if constexpr(DsDataEnableLds && EnableAsync == false)
            {
                static_for<0, NumDs_2, 1>{}([&](auto i) {
                    using DsDataBlockTransfer2 =
                        std::remove_const_t<remove_cvref_t<decltype(ds_blockwise_copy[I2][I0])>>;
                    const_cast<DsDataBlockTransfer2&>(ds_blockwise_copy[I2][i])
                        .RunWrite(ds_block_desc[I2][i], ds_2_block_buf(i));
                });
            }
        }

        // main body
        if constexpr(EnableAsync)
        {
            block_sync_lds_async_load();
        }
        else
        {
            block_sync_lds();
        }

        blockwise_conv[I2].template Run<2>(wei_2_block_buf,
                                           nullptr,
                                           out1_thread_buf,
                                           nullptr,
                                           nullptr,
                                           ds_2_block_buf,
                                           accum2_thread_buf,
                                           out2_thread_buf,
                                           emptySemas,
                                           Number<HasMainLoop>{},
                                           Number<true>{});
        block_sync_lds();
    }
};

template <index_t NumPrefetch,
          bool InDataEnableLds,
          bool WeiDataEnableLds,
          bool DsDataEnableLds,
          bool AsyncLoad,
          bool WaveGroup,
          bool SpatialCluster>
constexpr auto GridwiseFasternet50Pipeline_Selector()
{
    if constexpr(WaveGroup)
    {
        /* Todo: not support yet
        return GridwiseFasternet50Pipeline_v2<NumPrefetch,
                                       InDataEnableLds,
                                       WeiDataEnableLds,
                                       DsDataEnableLds,
                                       AsyncLoad,
                                       SpatialCluster>{};
                                       */
    }
    else
    {
        return GridwiseFasternet50Pipeline_v1<NumPrefetch,
                                              InDataEnableLds,
                                              WeiDataEnableLds,
                                              DsDataEnableLds,
                                              AsyncLoad>{};
    }
}

} // namespace ck
