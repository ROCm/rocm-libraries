// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/grouped_convolution/pipeline/wcnn_forward_ag_bg_cr_default_policy.hpp"
#include "ck_tile/ops/grouped_convolution/pipeline/wcnn_forward_pipeline_ag_bg_cr_base.hpp"

namespace ck_tile {

/// @brief Base class for WCNN forward pipeline — provides hotloop/tail dispatching logic.
///
/// Mirrors BaseGemmPipelineAgBgCrCompV3 but simplified for the WCNN case:
///   - PrefetchStages = 2 (two global loads ahead of compute)
///   - No 8-warp special case (WCNN uses fixed wavegroup configuration)
///   - TailHandler converts runtime (has_hot_loop, tail_number) into compile-time template args
///
/// @tparam Problem  WcnnFwdPipelineProblem type.
template <typename Problem>
struct BaseWcnnFwdPipeline
{
    static constexpr index_t PrefetchStages  = 2;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 1;

    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        if(BlockHasHotloop(num_loop))
            return TailNumber::Odd;
        else if(num_loop == 2)
            return TailNumber::Even;
        else
            return TailNumber::Odd;
    }

    template <size_t I = 0, typename RunFunction>
    CK_TILE_HOST_DEVICE static auto
    TailHandler(const RunFunction& run_func, bool has_hot_loop, TailNumber tail_number)
    {
        const bool has_hot_loop_first_lane      = amd_wave_read_first_lane(has_hot_loop);
        const TailNumber tail_number_first_lane = amd_wave_read_first_lane(tail_number);

        constexpr auto scenarios = std::array<std::pair<bool, ck_tile::TailNumber>, 3>{
            std::make_pair(true, TailNumber::Odd),   // hot loop + odd tail (common case)
            std::make_pair(false, TailNumber::Odd),  // no hot loop, 1 iteration
            std::make_pair(false, TailNumber::Even), // no hot loop, 2 iterations
        };

        if(has_hot_loop_first_lane == scenarios[I].first &&
           tail_number_first_lane == scenarios[I].second)
            return run_func(bool_constant<scenarios[I].first>{}, constant<scenarios[I].second>{});
        else if constexpr(I + 1 < scenarios.size())
            return TailHandler<I + 1>(run_func, has_hot_loop, tail_number);

#if defined(__HIP_DEVICE_COMPILE__)
        __builtin_unreachable();
#else
        throw std::logic_error("Invalid TailNumber value for WCNN pipeline");
#endif
    }
};

/// @brief WCNN forward pipeline — iterates over the C dimension, loading input/weight tiles
///        and accumulating via warp convolution intrinsics.
///
/// @par Overview
///      This pipeline follows the same pattern as the old gridwise WCNN:
///      1. Prefetch first C-tile of input and weight from global → LDS/VGPR
///      2. Main C-loop: compute warp conv on current tile while loading next tile
///      3. After loop: compute final tile
///
///      Unlike the implicit-GEMM pipeline (which uses warp GEMM on M×K × N×K tiles),
///      this pipeline operates on H×W×C input tiles and K×YXC weight tiles, and uses
///      `__builtin_amdgcn_convolve_*` intrinsics for the actual computation.
///
/// @tparam Problem  WcnnFwdPipelineProblem type containing data types and tile sizes.
/// @tparam Policy   Policy class providing LDS descriptors and tile distributions.
template <typename Problem, typename Policy = WcnnForwardDefaultPolicy>
struct WcnnFwdPipeline : public BaseWcnnFwdPipeline<Problem>
{
    using Base             = BaseWcnnFwdPipeline<Problem>;
    using PipelineImplBase = WcnnPipelineImplBase<Problem, Policy>;

    using ADataType   = typename Problem::ADataType;
    using BDataType   = typename Problem::BDataType;
    using AccDataType = typename Problem::AccDataType;

    using BlockWcnnShape = typename Problem::BlockWcnnShape;

    static constexpr index_t FilterY = Problem::FilterY;
    static constexpr index_t FilterX = Problem::FilterX;

    static constexpr index_t HPerBlock = BlockWcnnShape::HPerBlock;
    static constexpr index_t WPerBlock = BlockWcnnShape::WPerBlock;
    static constexpr index_t CPerBlock = BlockWcnnShape::CPerBlock;
    static constexpr index_t KPerBlock = BlockWcnnShape::KPerBlock;
    static constexpr index_t HPerWcnn  = BlockWcnnShape::HPerWcnn;
    static constexpr index_t WPerWcnn  = BlockWcnnShape::WPerWcnn;

    static constexpr index_t BlockSize = Problem::BlockSize;

    static constexpr bool kPadH = false;
    static constexpr bool kPadW = false;
    static constexpr bool kPadK = false;

    using BlockWcnn = remove_cvref_t<decltype(Policy::template GetBlockWcnn<Problem>())>;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return Problem::FixedVectorSize ? Problem::VectorSizeA
                                        : Policy::template GetVectorSizeA<Problem, IsWave32Host>();
    }

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return Problem::FixedVectorSize ? Problem::VectorSizeB
                                        : Policy::template GetVectorSizeB<Problem, IsWave32Host>();
    }

    struct PipelineImpl : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename ABlockWindowType,
                  typename BBlockWindowType>
        CK_TILE_DEVICE auto operator()(const ABlockWindowType& a_dram_block_window_tmp,
                                       const BBlockWindowType& b_dram_block_window_tmp,
                                       index_t num_loop,
                                       void* smem_ptr) const
        {
            auto&& [a_lds_block, b_lds_block] = Base::GetABLdsTensorView(smem_ptr);

            // tile distribution for load from lds
            constexpr auto a_lds_load_tile_distr =
                make_static_tile_distribution(BlockWcnn::MakeABlockDistributionEncode());
            constexpr auto b_lds_load_tile_distr =
                make_static_tile_distribution(BlockWcnn::MakeBBlockDistributionEncode());

            // get dram/lds tile window and lds tile for convolution

            auto&& [a_copy_dram_window, a_copy_lds_window, a_lds_wcnn_window] =
                Base::GetAWindows(a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);

            auto&& [b_copy_dram_window, b_copy_lds_window, b_lds_wcnn_window] =
                Base::GetBWindows(b_dram_block_window_tmp, b_lds_block, b_lds_load_tile_distr);

            // c_block_tile initialization

            auto block_wcnn   = BlockWcnn();
            auto c_block_tile = block_wcnn.MakeCBlockTile();

            // tile slice step

            // DRAM windows are 2D: A=[HW, C], B=[KYX, C]
            // Step along C dimension only
            using ADramTileWindowStep =
                typename remove_cvref_t<decltype(a_copy_dram_window)>::BottomTensorIndex;
            using BDramTileWindowStep =
                typename remove_cvref_t<decltype(b_copy_dram_window)>::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step = make_array(0, CPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step = make_array(0, CPerBlock);

            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // load tile from global to register
            auto a_block_tile = load_tile(a_copy_dram_window);
            auto b_block_tile = load_tile(b_copy_dram_window);

            move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
            move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

            // store to lds
            Base::LocalPrefill(a_copy_lds_window, a_block_tile);
            Base::LocalPrefill(b_copy_lds_window, b_block_tile);

            // load next tile from global to register
            a_block_tile = load_tile(a_copy_dram_window);
            b_block_tile = load_tile(b_copy_dram_window);

            move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
            move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

            block_sync_lds();
            block_wcnn.LocalPrefetch(a_lds_wcnn_window, b_lds_wcnn_window);

            if constexpr(HasHotLoop)
            {
                index_t iloop = 0;
                do
                {
                    block_sync_lds();

                    Base::LocalPrefill(a_copy_lds_window, a_block_tile);
                    Base::LocalPrefill(b_copy_lds_window, b_block_tile);

                    a_block_tile = load_tile(a_copy_dram_window);
                    move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
                    b_block_tile = load_tile(b_copy_dram_window);
                    move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

                    // block level convolution for i - 2
                    block_wcnn(c_block_tile, a_lds_wcnn_window, b_lds_wcnn_window);

                    block_sync_lds();

                    block_wcnn.LocalPrefetch(a_lds_wcnn_window, b_lds_wcnn_window);

                    ++iloop;
                } while(iloop < (num_loop - 1));
            }

            if constexpr(TailNum == TailNumber::Odd)
            {
                block_wcnn(c_block_tile, a_lds_wcnn_window, b_lds_wcnn_window);
            }
            else
            {
                block_wcnn(c_block_tile, a_lds_wcnn_window, b_lds_wcnn_window);
                block_sync_lds();

                Base::LocalPrefill(a_copy_lds_window, a_block_tile);
                Base::LocalPrefill(b_copy_lds_window, b_block_tile);
                block_sync_lds();

                block_wcnn.LocalPrefetch(a_lds_wcnn_window, b_lds_wcnn_window);
                block_wcnn(c_block_tile, a_lds_wcnn_window, b_lds_wcnn_window);
            }
            return c_block_tile;
        }
    };

    /// @brief Public entry point — dispatches to PipelineImpl via TailHandler.
    ///
    /// @param a_dram_block_window_tmp  Tile window into the input tensor (HW × C)
    /// @param b_dram_block_window_tmp  Tile window into the weight tensor (KYX × C)
    /// @param num_loop        Number of C-loop iterations (= C / CPerBlock)
    /// @param smem_ptr        Shared memory pointer for LDS staging
    /// @return                Accumulated output tile (H × W × K) in registers
    template <typename ABlockWindowType, typename BBlockWindowType>
    CK_TILE_DEVICE auto operator()(const ABlockWindowType& a_dram_block_window_tmp,
                                   const BBlockWindowType& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* smem_ptr) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);

        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp, b_dram_block_window_tmp, num_loop, smem_ptr);
        };

        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }
};

} // namespace ck_tile
