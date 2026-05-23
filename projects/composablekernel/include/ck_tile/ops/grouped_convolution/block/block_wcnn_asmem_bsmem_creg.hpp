// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// A is block window on shared memory
// B is block window on shared memory
// C is block distributed tensor
template <typename Problem_, typename WarpWcnn_>
struct BlockWcnnASmemBSmemCReg
{
    using Problem        = remove_cvref_t<Problem_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using AccDataType    = remove_cvref_t<typename Problem::AccDataType>;
    using BlockWcnnShape = remove_cvref_t<typename Problem::BlockWcnnShape>;

    using WarpWcnn = remove_cvref_t<WarpWcnn_>;

    using AWarpDstr   = typename WarpWcnn::AWarpDstr;
    using BWarpDstr   = typename WarpWcnn::BWarpDstr;
    using AccWarpDstr = typename WarpWcnn::AccWarpDstr;

    using AWarpTensor   = typename WarpWcnn::AWarpTensor;
    using BWarpTensor   = typename WarpWcnn::BWarpTensor;
    using AccWarpTensor = typename WarpWcnn::AccWarpTensor;

    static constexpr index_t kBlockSize = BlockWcnnShape::BlockSize;

    static constexpr index_t HPerBlock = BlockWcnnShape::HPerBlock;
    static constexpr index_t WPerBlock = BlockWcnnShape::WPerBlock;
    static constexpr index_t CPerBlock = BlockWcnnShape::CPerBlock;
    static constexpr index_t KPerBlock = BlockWcnnShape::KPerBlock;
    static constexpr index_t FilterY   = Problem::FilterY;
    static constexpr index_t FilterX   = Problem::FilterX;

    static constexpr index_t HPerWcnn = BlockWcnnShape::HPerWcnn;
    static constexpr index_t WPerWcnn = BlockWcnnShape::WPerWcnn;

    static constexpr index_t HWarp = BlockWcnnShape::WarpsInH;
    static constexpr index_t WWarp = BlockWcnnShape::WarpsInW;
    static constexpr index_t KWarp = BlockWcnnShape::WarpsInK;

    static constexpr index_t HIterPerWarp = HPerBlock / (HPerWcnn * HWarp);
    static constexpr index_t WIterPerWarp = WPerBlock / (WPerWcnn * WWarp);
    static constexpr index_t CIterPerWarp = CPerBlock / WarpWcnn::CPerWcnn;
    static constexpr index_t KIterPerWarp = KPerBlock / WarpWcnn::KPerWcnn;

    static constexpr index_t CPackedNum = min(CIterPerWarp, WarpWcnn::CPackedNum);

    static constexpr index_t BKIterPerWarp = KPerBlock / WarpWcnn::BKernelNum;

    CK_TILE_DEVICE static constexpr auto MakeABlockDistributionEncode()
    {
        constexpr auto a_block_out_desc_encoding =
            tile_distribution_encoding<sequence<KWarp>,
                                       tuple<sequence<HIterPerWarp, HWarp>,
                                             sequence<WIterPerWarp, WWarp>,
                                             sequence<CIterPerWarp>>,
                                       tuple<sequence<0, 1, 2>>,
                                       tuple<sequence<0, 1, 1>>,
                                       sequence<1, 2, 3>,
                                       sequence<0, 0, 0>>{};

        constexpr auto a_block_desc_encode = detail::make_embed_tile_distribution_encoding(
            a_block_out_desc_encoding, typename WarpWcnn::AWarpDstrEncoding{});

        return a_block_desc_encode;
    }

    CK_TILE_DEVICE static constexpr auto MakeBBlockDistributionEncode()
    {
        constexpr auto b_block_out_desc_encoding = tile_distribution_encoding<
            sequence<HWarp, WWarp>,
            tuple<sequence<BKIterPerWarp, KWarp>, sequence<>, sequence<CIterPerWarp / CPackedNum>>,
            tuple<sequence<1, 0, 0>>,
            tuple<sequence<1, 0, 1>>,
            sequence<1, 3>,
            sequence<0, 0>>{};

        constexpr auto b_block_desc_encode = detail::make_embed_tile_distribution_encoding(
            b_block_out_desc_encoding, typename WarpWcnn::BWarpDstrEncoding{});

        return b_block_desc_encode;
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        constexpr auto acc_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<HIterPerWarp, HWarp>,
                                             sequence<WIterPerWarp, WWarp>,
                                             sequence<KIterPerWarp, KWarp>>,
                                       tuple<sequence<3, 1, 2>>,
                                       tuple<sequence<1, 1, 1>>,
                                       sequence<1, 2, 3>,
                                       sequence<0, 0, 0>>{};

        constexpr auto acc_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            acc_block_outer_dstr_encoding, typename WarpWcnn::AccWarpDstrEncoding{});
        constexpr auto acc_block_dstr = make_static_tile_distribution(acc_block_dstr_encode);
        auto acc_block_tensor         = make_static_distributed_tensor<AccDataType>(acc_block_dstr);
        return acc_block_tensor;
    }

    static constexpr auto ALdsTileDistr =
        decltype(make_static_tile_distribution(MakeABlockDistributionEncode())){};
    static constexpr auto BLdsTileDistr =
        decltype(make_static_tile_distribution(MakeBBlockDistributionEncode())){};

    using ALdsTile = decltype(make_static_distributed_tensor<ADataType>(ALdsTileDistr));
    using BLdsTile = decltype(make_static_distributed_tensor<BDataType>(BLdsTileDistr));

    ALdsTile a_warp_tile_;
    BLdsTile b_warp_tile_;

    template <typename ASmemBlockWindow, typename BSmemBlockWindow>
    CK_TILE_DEVICE void LocalPrefetch(ASmemBlockWindow& a_block_window,
                                      BSmemBlockWindow& b_block_window)
    {
        load_tile(a_warp_tile_, a_block_window);
        load_tile(b_warp_tile_, b_block_window);
    }

    // C += A * B
    template <typename CBlockTensor, typename ASmemBlockWindow, typename BSmemBlockWindow>
    CK_TILE_DEVICE void
    operator()(CBlockTensor& c_block_tensor, const ASmemBlockWindow&, const BSmemBlockWindow&) const
    {
        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_lengths =
            to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(AccWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<AccWarpDstr::NDimY, 0>{};

        // Phase 1: Pre-extract all weight warp tensors from b_warp_tile_
        // b_warp_tensors[bKIter][cIter]
        statically_indexed_array<statically_indexed_array<BWarpTensor, CIterPerWarp / CPackedNum>,
                                 BKIterPerWarp>
            b_warp_tensors;

        static_for<0, BKIterPerWarp, 1>{}([&](auto bKIter) {
            static_for<0, CIterPerWarp / CPackedNum, 1>{}([&](auto cIter) {
                b_warp_tensors(bKIter)(cIter).get_thread_buffer() =
                    b_warp_tile_.get_y_sliced_thread_data(
                        merge_sequences(sequence<bKIter, cIter>{}, b_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));
            });
        });

        // Phase 2: Iterate over (H, W, C), load input, compute against all K
        // C loop is split: outer loop changes weight (cOuterIter), inner loop
        // within CPackedNum reuses same weight but toggles HighLane.
        static_for<0, HIterPerWarp, 1>{}([&](auto hIter) {
            static_for<0, WIterPerWarp, 1>{}([&](auto wIter) {
                static_for<0, CIterPerWarp / CPackedNum, 1>{}([&](auto cOuterIter) {
                    static_for<0, CPackedNum, 1>{}([&](auto cPackIter) {
                        constexpr index_t cIter =
                            decltype(cOuterIter)::value * CPackedNum + decltype(cPackIter)::value;
                        constexpr bool highlane = decltype(cPackIter)::value > 0;

                        // Extract input warp tensor
                        AWarpTensor a_warp_tensor;
                        a_warp_tensor.get_thread_buffer() = a_warp_tile_.get_y_sliced_thread_data(
                            merge_sequences(sequence<hIter, wIter, cIter>{}, a_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1, 1>{}, a_warp_y_lengths));

                        static_for<0, BKIterPerWarp, 1>{}([&](auto bKIter) {
                            constexpr index_t k_idx = decltype(bKIter)::value;

                            // Read C warp tensor from C block tensor
                            AccWarpTensor c_warp_tensor;
                            c_warp_tensor.get_thread_buffer() =
                                c_block_tensor.get_y_sliced_thread_data(
                                    merge_sequences(sequence<hIter, wIter, k_idx>{},
                                                    c_warp_y_index_zeros),
                                    merge_sequences(sequence<1, 1, 1>{}, c_warp_y_lengths));

                            // Warp convolution: c += wcnn(a, b)
                            // Same weight (cOuterIter) within CPackedNum, only HighLane changes
                            WarpWcnn{}.template operator()<highlane>(
                                c_warp_tensor, b_warp_tensors(bKIter)(cOuterIter), a_warp_tensor);

                            // Write C warp tensor back to C block tensor
                            c_block_tensor.set_y_sliced_thread_data(
                                merge_sequences(sequence<hIter, wIter, k_idx>{},
                                                c_warp_y_index_zeros),
                                merge_sequences(sequence<1, 1, 1>{}, c_warp_y_lengths),
                                c_warp_tensor.get_thread_buffer());
                        });
                    });
                });
            });
        });
    }
};

} // namespace ck_tile
