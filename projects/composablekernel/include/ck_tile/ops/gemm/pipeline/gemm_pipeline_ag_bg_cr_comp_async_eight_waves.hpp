// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_eight_waves_policy.hpp"

namespace ck_tile {

/**
 * @brief Compute optimized pipeline version async; which is based on V4.
 *
 * This pipeline introduces asynchronous load from global memory to LDS,
 * skipping the intermediate loading into pipeline registers.
 */
template <typename Problem, typename Policy = GemmPipelineAgBgCrCompAsyncEightWavesPolicy>
struct GemmPipelineAgBgCrCompAsyncEightWaves : public BaseGemmPipelineAgBgCrCompV3<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrCompV3<Problem>;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using AsDataType     = remove_cvref_t<typename Problem::AsDataTypeTuple>;
    using BsDataType     = remove_cvref_t<typename Problem::BsDataTypeTuple>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using AsLayout = remove_cvref_t<typename Problem::AsLayoutTuple>;
    using BsLayout = remove_cvref_t<typename Problem::BsLayoutTuple>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    using AElementWise = remove_cvref_t<typename Problem::AElementWise>;
    using BElementWise = remove_cvref_t<typename Problem::BElementWise>;

    using ALayout = remove_cvref_t<std::tuple_element_t<0, AsLayout>>;
    using BLayout = remove_cvref_t<std::tuple_element_t<0, BsLayout>>;

    using ADataType = remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<0, BsDataType>>;

    static_assert(!std::is_same_v<BDataType, pk_int4_t>, "Not implemented");

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    using WarpGemm  = typename BlockGemm::WarpGemm;

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t MWarps      = BlockGemmShape::BlockWarps::at(I0);
    static constexpr index_t NWarps      = BlockGemmShape::BlockWarps::at(I1);
    static constexpr index_t KWarps      = BlockGemmShape::BlockWarps::at(I2);
    static constexpr index_t warp_groups = 2; // ping-pong

    static constexpr index_t kflatKPerBlock = BlockGemmShape::flatKPerBlock;
    static constexpr index_t flatKPerWarp   = BlockGemmShape::flatKPerWarp;
    static constexpr index_t flatNPerWarp   = BlockGemmShape::flatNPerWarp;
    static constexpr index_t WarpTileN      = BlockGemmShape::WarpTile::at(I1);

    static constexpr index_t MIterPerWarp = MPerBlock / (MWarps * WarpGemm::kM);
    static constexpr index_t NIterPerWarp = NPerBlock / (NWarps * WarpGemm::kN);
    static constexpr index_t KIterPerWarp = KPerBlock / (KWarps * WarpGemm::kK);

    static constexpr bool Async = true;

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return Policy::template GetVectorSizeA<Problem>();
    }
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return Policy::template GetVectorSizeB<Problem>();
    }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
    static constexpr index_t Preshuffle    = Problem::Preshuffle;

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;

    static constexpr auto Scheduler = Problem::Scheduler;

    static constexpr auto is_a_load_tr_v = bool_constant<PipelineImplBase::is_a_load_tr>{};
    static constexpr auto is_b_load_tr_v = bool_constant<PipelineImplBase::is_b_load_tr>{};

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName()
    {
        // clang-format off
        return "COMPUTE_ASYNC";
        // clang-format on
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0);
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1);
        return concat('_', "pipeline_AgBgCrCompAsyncEightWaves", 
                      concat('x', MPerBlock, NPerBlock, KPerBlock),  BlockSize,
                      concat('x', GetVectorSizeA(), GetVectorSizeB()),
                      concat('x', WaveNumM, WaveNumN),
                      concat('x', kPadM, kPadN, kPadK),
                      Problem::GetName());
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    static constexpr index_t A_LOAD_INST = MPerBlock * KPerBlock / BlockSize / GetVectorSizeA();
    static constexpr index_t B_LOAD_INST = NPerBlock * KPerBlock / BlockSize / GetVectorSizeB();
    static constexpr index_t MFMA_INST   = MIterPerWarp * NIterPerWarp * KIterPerWarp;

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        CK_TILE_DEVICE static constexpr auto HotLoopScheduler()
        {
            // TODO
        }

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename AsDramBlockWindowTmp,
                  typename BsDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction,
                  typename std::enable_if_t<!is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                                !is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                            bool>* = nullptr>
        CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       index_t num_loop,
                                       void* __restrict__ p_smem) const
        {
            //  ping-pong swap for lds access
            const index_t warp_group_id = get_warp_id() / (MWarps * NWarps * KWarps / warp_groups);
            const bool is_ping          = warp_group_id == 0;
            const bool is_pong          = warp_group_id != 0;
            const auto smem             = reinterpret_cast<uint8_t*>(p_smem);
            constexpr index_t lds_0_offset = 0;
            constexpr index_t lds_1_offset = lds_0_offset +
                                             Policy::template GetSmemSizeA<Problem>() +
                                             Policy::template GetSmemSizeB<Problem>();

            return operator()<HasHotLoop, TailNum>(a_dram_block_window_tmp,
                                                   a_element_func,
                                                   b_dram_block_window_tmp,
                                                   b_element_func,
                                                   num_loop,
                                                   smem + (is_ping ? lds_0_offset : lds_1_offset),
                                                   smem + (is_pong ? lds_0_offset : lds_1_offset));
        }

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename AsDramBlockWindowTmp,
                  typename BsDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction,
                  typename std::enable_if_t<!is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                                !is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                            bool>* = nullptr>
        CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                       [[maybe_unused]] const AElementFunction& a_element_func,
                                       const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                       [[maybe_unused]] const BElementFunction& b_element_func,
                                       index_t num_loop,
                                       void* __restrict__ p_smem0,
                                       void* __restrict__ p_smem1) const
        {
            // TODO: A/B element func are currently not used

            // ------
            // Checks
            // ------
            static_assert(!is_detected<is_tuple, AsDramBlockWindowTmp>::value);
            static_assert(!is_detected<is_tuple, BsDramBlockWindowTmp>::value);
            static_assert(
                std::is_same_v<ADataType,
                               remove_cvref_t<typename AsDramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BsDramBlockWindowTmp::DataType>>,
                "A/B Dram block window should have the same data type as appropriate "
                "([A|B]DataType) defined in Problem definition!");

            static_assert(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>, "Wrong!");
            static_assert(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>, "Wrong!");

            static_assert((MPerBlock == AsDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                           KPerBlock == AsDramBlockWindowTmp{}.get_window_lengths()[I1]),
                          "A block window has incorrect lengths for defined ALayout!");
            static_assert(Preshuffle //
                              ? (NWarps == BsDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                                 kflatKPerBlock == BsDramBlockWindowTmp{}.get_window_lengths()[I1])
                              : (NPerBlock == BsDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                                 KPerBlock == BsDramBlockWindowTmp{}.get_window_lengths()[I1]),
                          "B block window has incorrect lengths for defined BLayout!");

            constexpr index_t N_LOOP = HasHotLoop                    ? 4
                                       : TailNum == TailNumber::One  ? 1
                                       : TailNum == TailNumber::Even ? 2
                                       : TailNum == TailNumber::Odd  ? 3
                                                                     : 0;
            static_assert(N_LOOP >= 1, "wrong!");

            // -----
            // Setup
            // -----
            const index_t warp_group_id = get_warp_id() / (MWarps * NWarps * KWarps / warp_groups);
            const bool is_ping          = warp_group_id == 0;
            const bool is_pong          = warp_group_id != 0;

            const auto smem01 = make_array(reinterpret_cast<uint8_t*>(p_smem0),
                                           reinterpret_cast<uint8_t*>(p_smem1));

            constexpr auto LDS = address_space_enum::lds;
            auto lds_a         = make_tensor_view<LDS>(static_cast<ADataType*>(nullptr),
                                               Policy::template MakeALdsBlockDescriptor<Problem>());
            auto lds_b         = make_tensor_view<LDS>(static_cast<BDataType*>(nullptr),
                                               Policy::template MakeBLdsBlockDescriptor<Problem>());
            auto lds_b_read =
                make_tensor_view<LDS>(static_cast<BDataType*>(nullptr),
                                      Policy::template MakeBLdsReadBlockDescriptor<Problem>());

            constexpr auto lds_offset_a = 0;
            constexpr auto lds_offset_b = lds_offset_a + Policy::template GetSmemSizeA<Problem>();

            constexpr auto a_load_distr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto b_load_distr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());
            constexpr auto a_copy_distr = Policy::template MakeADramTileDistribution<Problem>();
            constexpr auto b_copy_distr = Policy::template MakeBDramTileDistribution<Problem>();
            constexpr auto a_lds_size   = number_tuple<MPerBlock, KPerBlock>{};
            constexpr auto b_lds_size =
                number_tuple<(Preshuffle ? NPerBlock / WarpTileN : NPerBlock),
                             (Preshuffle ? KPerBlock * WarpTileN : KPerBlock)>{};
            constexpr auto b_lds_read_size = number_tuple<NPerBlock, KPerBlock>{};

            auto a_copy_dram_window = make_tile_window(
                Policy::template MakeAsyncLoadADramWindow<Problem>(a_dram_block_window_tmp),
                a_copy_distr);
            auto b_copy_dram_window = make_tile_window(
                Policy::template MakeAsyncLoadBDramWindow<Problem>(b_dram_block_window_tmp),
                b_copy_distr);
            auto a_copy_lds_window = make_tile_window(lds_a, a_lds_size, {0, 0}, a_copy_distr);
            auto b_copy_lds_window = make_tile_window(lds_b, b_lds_size, {0, 0}, b_copy_distr);
            auto a_lds_gemm_window = make_tile_window(lds_a, a_lds_size, {0, 0}, a_load_distr);
            auto b_lds_gemm_window =
                make_tile_window(lds_b_read, b_lds_read_size, {0, 0}, b_load_distr);

            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            typename BlockGemm::ALdsTile a_block_tile;
            typename BlockGemm::BLdsTile b_block_tile;

            // Lambdas
            auto load_global = [&](index_t i) {
                constexpr auto NEG1 = number<-1>{};
                a_copy_lds_window.set_bottom_tensor_view_data_ptr(
                    reinterpret_cast<ADataType*>(smem01[i] + lds_offset_a));
                async_load_tile(
                    a_copy_lds_window, a_copy_dram_window, NEG1, false_type{}, true_type{});

                b_copy_lds_window.set_bottom_tensor_view_data_ptr(
                    reinterpret_cast<BDataType*>(smem01[i] + lds_offset_b));
                async_load_tile(
                    b_copy_lds_window, b_copy_dram_window, NEG1, false_type{}, true_type{});
            };
            constexpr typename decltype(a_copy_dram_window)::BottomTensorIndex a_move_step = //
                {0, KPerBlock};
            constexpr typename decltype(b_copy_dram_window)::BottomTensorIndex b_move_step = //
                {0, Preshuffle ? kflatKPerBlock : KPerBlock};
            auto move_global = [&]() {
                move_tile_window(a_copy_dram_window, a_move_step);
                move_tile_window(b_copy_dram_window, b_move_step);
            };
            auto load_local = [&](index_t i) {
                a_lds_gemm_window.set_bottom_tensor_view_data_ptr(
                    reinterpret_cast<ADataType*>(smem01[i] + lds_offset_a));
                a_lds_gemm_window.load(a_block_tile, number<-1>{}, true_type{}, true_type{});

                b_lds_gemm_window.set_bottom_tensor_view_data_ptr(
                    reinterpret_cast<BDataType*>(smem01[i] + lds_offset_b));
                static_for_product<number<NIterPerWarp>, number<KIterPerWarp>>{}(
                    [&](auto nIter, auto kIter) {
                        b_lds_gemm_window.load_with_offset(
                            number_tuple<WarpGemm::kN * nIter, WarpGemm::kK * kIter>{},
                            b_block_tile[nIter][kIter],
                            number<-1>{},
                            true_type{},
                            true_type{});
                    });
            };

            auto calc_gemm = [&]() {
                block_gemm(c_block_tile, a_block_tile, b_block_tile);

                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                __builtin_amdgcn_sched_group_barrier(0x002, MIterPerWarp, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                s_waitcnt_lgkm<4>();
                __builtin_amdgcn_sched_group_barrier(0x004, 1, 0); // lgkmcnt
                static_for<0, MFMA_INST - 3, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(0x002, 4, 0);
                });
                __builtin_amdgcn_sched_group_barrier(0x002, 12, 0);

                __builtin_amdgcn_sched_barrier(0);
            };

            auto main_body = [&](auto tic, auto toc) {
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_setprio(1);

                s_nop();
                calc_gemm(); // tic

                s_waitcnt</*vmcnt*/ 0>();
                move_tile_window(a_copy_dram_window, a_move_step);
                __builtin_amdgcn_s_barrier();

                __builtin_amdgcn_sched_barrier(0);

                constexpr auto NEG1 = number<-1>{};
                a_copy_lds_window.set_bottom_tensor_view_data_ptr(
                    reinterpret_cast<ADataType*>(smem01[tic] + lds_offset_a));
                async_load_tile(
                    a_copy_lds_window, a_copy_dram_window, NEG1, false_type{}, true_type{});

                __builtin_amdgcn_s_setprio(0);
                move_tile_window(b_copy_dram_window, b_move_step);

                a_lds_gemm_window.set_bottom_tensor_view_data_ptr(
                    reinterpret_cast<ADataType*>(smem01[toc] + lds_offset_a));
                a_lds_gemm_window.load(a_block_tile, number<-1>{}, true_type{}, true_type{});

                b_copy_lds_window.set_bottom_tensor_view_data_ptr(
                    reinterpret_cast<BDataType*>(smem01[tic] + lds_offset_b));
                async_load_tile(
                    b_copy_lds_window, b_copy_dram_window, NEG1, false_type{}, true_type{});

                b_lds_gemm_window.set_bottom_tensor_view_data_ptr(
                    reinterpret_cast<BDataType*>(smem01[toc] + lds_offset_b));
                static_for_product<number<NIterPerWarp>, number<KIterPerWarp>>{}(
                    [&](auto nIter, auto kIter) {
                        b_lds_gemm_window.load_with_offset(
                            number_tuple<WarpGemm::kN * nIter, WarpGemm::kK * kIter>{},
                            b_block_tile[nIter][kIter],
                            number<-1>{},
                            true_type{},
                            true_type{});
                    });
                __builtin_amdgcn_sched_barrier(0);
                s_waitcnt</*vmcnt*/ B_LOAD_INST>();
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
            };

            // -------
            // Compute
            // -------
            __builtin_amdgcn_sched_barrier(0);
            if(is_pong)
            {
                load_global(1);
                s_waitcnt</*vmcnt*/ B_LOAD_INST>();
                __builtin_amdgcn_s_barrier();
                move_global();
            }
            __builtin_amdgcn_sched_barrier(0);

            clear_tile(c_block_tile);
            s_waitcnt</*vmcnt*/ 0>();
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            if constexpr(N_LOOP >= 2)
            {
                load_global(0);
            }
            else if(is_ping)
            {
                load_global(0);
            }
            if(is_pong)
                load_local(1);
            s_waitcnt</*vmcnt*/ B_LOAD_INST>();
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            if(is_pong)
                calc_gemm(); // 1
            if constexpr(N_LOOP >= 2)
                move_global();
            s_waitcnt</*vmcnt*/ 0>();
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            if constexpr(N_LOOP >= 3)
            {
                load_global(1);
                load_local(0);
                s_waitcnt</*vmcnt*/ B_LOAD_INST>();
                __builtin_amdgcn_s_barrier();
            }

            if constexpr(HasHotLoop)
            {
                index_t loop_count = num_loop - 3 - 1;
                while(0 < loop_count)
                {
                    main_body(I0, I1);
                    --loop_count;

                    main_body(I1, I0);
                    --loop_count;
                };
            }
            // tail
            if constexpr(HasHotLoop && TailNum == TailNumber::Even)
            {
                asm volatile(";; Even Tail Start ;;");
                __builtin_amdgcn_s_barrier();
                main_body(I0, I1);
                __builtin_amdgcn_s_barrier();
                asm volatile(";; Even Tail End ;;");
                __builtin_amdgcn_s_barrier();
            }

            constexpr int tic = HasHotLoop ? (TailNum == TailNumber::Odd ? 0 : 1) : 1 - N_LOOP % 2;
            constexpr int toc = 1 - tic;
            if constexpr(N_LOOP >= 3)
            {
                calc_gemm(); // tic
                move_global();
                s_waitcnt</*vmcnt*/ 0>();
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
            }

            if constexpr(N_LOOP >= 2)
            {
                // if(is_ping) // extra pong load to avoid reg spill
                load_global(tic);

                __builtin_amdgcn_sched_barrier(0);
                load_local(toc);
                s_waitcnt</*vmcnt*/ B_LOAD_INST>();

                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                calc_gemm(); // toc
                s_waitcnt</*vmcnt*/ 0>();
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
            }

            if(is_ping)
            {
                load_local(toc ^ 1);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                calc_gemm(); // toc ^ 1
            }

            return c_block_tile;
        }
    };

    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction,
              typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        // TODO: A/B windows are tuple of windows, but the implementation doesn't take that into
        // account yet and just the first element is passed
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);
        const auto RunPipeline  = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp[I0],
                a_element_func,
                b_dram_block_window_tmp[I0],
                b_element_func,
                num_loop,
                p_smem);
        };

        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }
};
} // namespace ck_tile
