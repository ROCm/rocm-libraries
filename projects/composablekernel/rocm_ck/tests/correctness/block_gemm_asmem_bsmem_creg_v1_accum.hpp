// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Block GEMM with pluggable warp-level accumulation policy.
//
// Adapted from CK's BlockGemmASmemBSmemCRegV1. The hot loop uses
// the 2-arg WarpGemm (returns fresh delta) instead of the 3-arg
// form (c += a*b), then merges via AccumPolicy at warp granularity.
// This lets us control accumulation error at the finest software
// level (per-MFMA-instruction) with minimal register overhead
// (+4 floats/thread for the error term).

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_default_policy.hpp"

namespace rocm_ck::test {

// =========================================================================
// Accumulation policies — operate on warp tensors (4 floats/thread)
// =========================================================================

struct AccumNaive
{
    template <typename T>
    CK_TILE_DEVICE static void merge(T& c, T& /*err*/, T& delta)
    {
        auto& c_buf       = c.get_thread_buffer();
        auto& d_buf = delta.get_thread_buffer();
        ck_tile::static_for<0, T::get_thread_buffer_size(), 1>{}(
            [&](auto i) { c_buf(i) += d_buf(i); });
    }

    template <typename T>
    CK_TILE_DEVICE static void finalize(T& /*c*/, T& /*err*/)
    {
    }
};

struct AccumTwoSum
{
    template <typename T>
    CK_TILE_DEVICE static void merge(T& c, T& err, T& delta)
    {
        auto& c_buf       = c.get_thread_buffer();
        auto& e_buf       = err.get_thread_buffer();
        auto& d_buf = delta.get_thread_buffer();
        ck_tile::static_for<0, T::get_thread_buffer_size(), 1>{}([&](auto i) {
            float s = c_buf(i) + d_buf(i);
            float v = s - c_buf(i);
            e_buf(i) += (c_buf(i) - (s - v)) + (d_buf(i) - v);
            c_buf(i) = s;
        });
    }

    template <typename T>
    CK_TILE_DEVICE static void finalize(T& c, T& err)
    {
        auto& c_buf = c.get_thread_buffer();
        auto& e_buf = err.get_thread_buffer();
        ck_tile::static_for<0, T::get_thread_buffer_size(), 1>{}(
            [&](auto i) { c_buf(i) += e_buf(i); });
    }
};

struct AccumKahan
{
    template <typename T>
    CK_TILE_DEVICE static void merge(T& c, T& comp, T& delta)
    {
        auto& c_buf       = c.get_thread_buffer();
        auto& comp_buf    = comp.get_thread_buffer();
        auto& d_buf = delta.get_thread_buffer();
        ck_tile::static_for<0, T::get_thread_buffer_size(), 1>{}([&](auto i) {
            float y     = d_buf(i) - comp_buf(i);
            float t     = c_buf(i) + y;
            comp_buf(i) = (t - c_buf(i)) - y;
            c_buf(i)    = t;
        });
    }

    template <typename T>
    CK_TILE_DEVICE static void finalize(T& /*c*/, T& /*comp*/)
    {
    }
};

// =========================================================================
// BlockGemmASmemBSmemCRegV1Accum
// =========================================================================

template <typename Problem_,
          typename Policy_       = ck_tile::BlockGemmASmemBSmemCRegV1DefaultPolicy,
          typename AccumPolicy_  = AccumNaive>
struct BlockGemmASmemBSmemCRegV1Accum
{
    using Problem        = ck_tile::remove_cvref_t<Problem_>;
    using Policy         = ck_tile::remove_cvref_t<Policy_>;
    using AccumPolicy    = ck_tile::remove_cvref_t<AccumPolicy_>;
    using ADataType      = ck_tile::remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = ck_tile::remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = ck_tile::remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = ck_tile::remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;

    // C += A * B  with compensated accumulation via AccumPolicy
    template <typename CBlockTensor, typename ABlockWindow, typename BBlockWindow>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   CBlockTensor& err_block_tensor,
                                   const ABlockWindow& a_block_window,
                                   const BBlockWindow& b_block_window) const
    {
        using namespace ck_tile;

        static_assert(std::is_same_v<ADataType, typename ABlockWindow::DataType> &&
                          std::is_same_v<BDataType, typename BBlockWindow::DataType> &&
                          std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockWindow{}.get_window_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockWindow{}.get_window_lengths()[number<0>{}];
        constexpr index_t KPerBlock = ABlockWindow{}.get_window_lengths()[number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr index_t MPerBlockPerIter = MPerBlock / MIterPerWarp;
        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        const index_t iMWarp = get_warp_id() / NWarp;
        const index_t iNWarp = get_warp_id() % NWarp;

        // A warp windows
        auto a_warp_window_tmp = make_tile_window(
            a_block_window.get_bottom_tensor_view(),
            make_tuple(number<WG::kM>{}, number<WG::kK>{}),
            a_block_window.get_window_origin() + multi_index<2>{iMWarp * WG::kM, 0},
            make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(a_warp_window_tmp), KIterPerWarp>,
            MIterPerWarp>
            a_warp_windows;

        static_ford<sequence<MIterPerWarp, KIterPerWarp>>{}([&](auto mk) {
            constexpr auto mIter         = number<mk[number<0>{}]>{};
            constexpr auto kIter         = number<mk[number<1>{}]>{};
            a_warp_windows(mIter)(kIter) = a_warp_window_tmp;

            move_tile_window(a_warp_windows(mIter)(kIter),
                             {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
        });

        // B warp windows
        auto b_warp_window_tmp = make_tile_window(
            b_block_window.get_bottom_tensor_view(),
            make_tuple(number<WG::kN>{}, number<WG::kK>{}),
            b_block_window.get_window_origin() + multi_index<2>{iNWarp * WG::kN, 0},
            make_static_tile_distribution(typename WG::BWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(b_warp_window_tmp), KIterPerWarp>,
            NIterPerWarp>
            b_warp_windows;

        static_ford<sequence<NIterPerWarp, KIterPerWarp>>{}([&](auto nk) {
            constexpr auto nIter         = number<nk[number<0>{}]>{};
            constexpr auto kIter         = number<nk[number<1>{}]>{};
            b_warp_windows(nIter)(kIter) = b_warp_window_tmp;

            move_tile_window(b_warp_windows(nIter)(kIter),
                             {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
        });

        using CWarpDstr   = typename WG::CWarpDstr;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop — same iteration order as CK, but with policy-controlled accumulation
        static_ford<sequence<KIterPerWarp, MIterPerWarp>>{}([&](auto km) {
            constexpr auto kIter = number<km[number<0>{}]>{};
            constexpr auto mIter = number<km[number<1>{}]>{};

            const auto a_warp_tensor = load_tile(a_warp_windows(mIter)(kIter));

            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                const auto b_warp_tensor = load_tile(b_warp_windows(nIter)(kIter));

                // 2-arg WarpGemm: fresh delta (4 floats/thread)
                auto delta_warp = WG{}(a_warp_tensor, b_warp_tensor);

                // Load c_warp and err_warp from block tensors
                CWarpTensor c_warp_tensor;
                c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                CWarpTensor err_warp_tensor;
                err_warp_tensor.get_thread_buffer() = err_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                // Policy-controlled merge
                AccumPolicy::merge(c_warp_tensor, err_warp_tensor, delta_warp);

                // Write back
                c_block_tensor.set_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                    c_warp_tensor.get_thread_buffer());

                err_block_tensor.set_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                    err_warp_tensor.get_thread_buffer());
            });
        });
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        using namespace ck_tile;

        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);

        auto c_block_tensor = make_static_distributed_tensor<CDataType>(c_block_dstr);
        return c_block_tensor;
    }
};

} // namespace rocm_ck::test
