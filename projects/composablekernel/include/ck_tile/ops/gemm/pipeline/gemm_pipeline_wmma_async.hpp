// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_wmma_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_wmma_base.hpp"

namespace ck_tile {

template <typename Problem>
struct BaseGemmPipelineAgBgCrWmmaAsync
{
    static constexpr index_t PrefetchStages  = 2;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 1;

    CK_TILE_HOST static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }
    // this 2 is used for double lds buffer
    CK_TILE_HOST static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        if(num_loop % 2)
        {
            return TailNumber::Odd;
        }
        else
        {
            return TailNumber::Even;
        }
    }
};
template <typename Problem, typename Policy = GemmPipelineWmmaTrLoadAsyncCopyPolicy>
struct GemmWmmaPipelineAsync
{
    using PipelineImplBase = GemmPipelineAgBgCrWmmaImplBase<Problem, Policy>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    using I0        = number<0>;
    using I1        = number<1>;
    using I2        = number<2>;

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t GetVectorSizeA() { return Policy::template GetVectorSizeA<Problem>(); }
    static constexpr index_t GetVectorSizeB() { return Policy::template GetVectorSizeB<Problem>(); }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;
    static constexpr auto Scheduler  = Problem::Scheduler;

    static constexpr bool kTransLoadEn = Policy::TransLoadEn;

    static constexpr bool kAsyncCopy  = Policy::AsyncCopy;
    static constexpr bool kAsyncStore = Policy::AsyncStore;

    // don't use this variable, just for compatibility
    static constexpr bool DoubleSmemBuffer = false;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };
    template <>
    struct PipelineImpl<GemmPipelineScheduler::Default> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename ADramBlockWindowTmp,
                  typename BDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction>
        CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       index_t num_loop,
                                       void* p_smem) const
        {
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "A/B Dram block window should have the same data type as appropriate "
                "([A|B]DataType) defined in Problem definition!");

            // TODO: for async load, only support no data change in a_element_func and
            // b_element_func; need to add static check
            ignore = a_element_func;
            ignore = b_element_func;

            constexpr bool is_a_col_major =
                std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

            static_assert(is_a_col_major
                              ? (KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "A block window has incorrect lengths for defined ALayout!");
            static_assert(is_b_row_major
                              ? (KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "B block window has incorrect lengths for defined BLayout!");

            // ------------------------------------------------------------------------------------
            // Definitions of all needed tiles

            // A/B tiles in LDS
            auto&& [a_lds_block, b_lds_block] = Base::GetABLdsTensorViews(p_smem);

            // Tile distribution for load from lds
            constexpr auto a_lds_load_tile_distr = decltype(make_static_tile_distribution(
                BlockGemm::MakeABlockDistributionEncode())){};
            constexpr auto b_lds_load_tile_distr = decltype(make_static_tile_distribution(
                BlockGemm::MakeBBlockDistributionEncode())){};

            // A DRAM tile window for load
            // A LDS tile window for store
            // A LDS tile for block GEMM
            auto&& [a_copy_dram_window, a_copy_lds_window, a_lds_gemm_window] =
                Base::GetAMultiLdsWindows(
                    a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);

            // B DRAM tile window for load
            // B LDS tile window for store
            // B LDS tile for block GEMM
            auto&& [b_copy_dram_window, b_copy_lds_window, b_lds_gemm_window] =
                Base::GetBMultiLdsWindows(
                    b_dram_block_window_tmp, b_lds_block, b_lds_load_tile_distr);

            // Block GEMM
            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            // the below is used for async cnt calculation
            using ACopyDramWindow = remove_cvref_t<decltype(a_copy_dram_window)>;
            using BCopyDramWindow = remove_cvref_t<decltype(b_copy_dram_window)>;

            constexpr auto a_number_of_access = ACopyDramWindow{}.get_num_of_access();
            constexpr auto b_number_of_access = BCopyDramWindow{}.get_num_of_access();

            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

            // -----------------------------------------------------------------------------------------
            // Gemm pipeline start

            // prefetch
            // global read 0
            Base::GlobalPrefetchAsync(
                a_copy_dram_window, a_dram_tile_window_step, a_copy_lds_window(I0{}));
            Base::GlobalPrefetchAsync(
                b_copy_dram_window, b_dram_tile_window_step, b_copy_lds_window(I0{}));

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            Base::GlobalPrefetchAsync(
                a_copy_dram_window, a_dram_tile_window_step, a_copy_lds_window(I1{}));
            Base::GlobalPrefetchAsync(
                b_copy_dram_window, b_dram_tile_window_step, b_copy_lds_window(I1{}));

            if constexpr(HasHotLoop)
            {
                index_t iCounter = __builtin_amdgcn_readfirstlane(num_loop - 1);
                do
                {

                    block_async_lds(a_number_of_access + b_number_of_access);
                    block_gemm(c_block_tile, a_lds_gemm_window(I0{}), b_lds_gemm_window(I0{}));
                    block_sync_lds();

                    Base::GlobalPrefetchAsync(
                        a_copy_dram_window, a_dram_tile_window_step, a_copy_lds_window(I0{}));
                    Base::GlobalPrefetchAsync(
                        b_copy_dram_window, b_dram_tile_window_step, b_copy_lds_window(I0{}));

                    block_async_lds(a_number_of_access + b_number_of_access);
                    block_gemm(c_block_tile, a_lds_gemm_window(I1{}), b_lds_gemm_window(I1{}));
                    block_sync_lds();

                    Base::GlobalPrefetchAsync(
                        a_copy_dram_window, a_dram_tile_window_step, a_copy_lds_window(I1{}));
                    Base::GlobalPrefetchAsync(
                        b_copy_dram_window, b_dram_tile_window_step, b_copy_lds_window(I1{}));
                    iCounter -= 2;
                } while(iCounter > 1);
            }

            // tail
            if constexpr(TailNum == TailNumber::Odd)
            {
                block_async_lds(0);
                block_gemm(c_block_tile, a_lds_gemm_window(I0{}), b_lds_gemm_window(I0{}));
            }
            else
            {
                block_async_lds(a_number_of_access + b_number_of_access);
                block_gemm(c_block_tile, a_lds_gemm_window(I0{}), b_lds_gemm_window(I0{}));

                block_async_lds(0);
                block_gemm(c_block_tile, a_lds_gemm_window(I1{}), b_lds_gemm_window(I1{}));
            }

            return c_block_tile;
        }
    };

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            a_element_func,
            b_dram_block_window_tmp,
            b_element_func,
            num_loop,
            p_smem);
    }

    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; },
            b_dram_block_window_tmp,
            [](const BDataType& b) { return b; },
            num_loop,
            p_smem);
    }
};

} // namespace ck_tile
