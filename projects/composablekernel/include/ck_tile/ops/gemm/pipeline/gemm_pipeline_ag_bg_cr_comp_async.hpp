// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_default_policy.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem>
struct BaseGemmPipelineAgBgCrCompAsync
{
    static constexpr index_t PrefetchStages   = 2;
    static constexpr index_t PrefillStages    = 1;
    static constexpr index_t GlobalBufferNum  = 1;
    static constexpr bool UsePersistentKernel = Problem::Traits::UsePersistentKernel;

    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        constexpr index_t HotLoopGlobalReads = 2;
        return num_loop >= (HotLoopGlobalReads + PrefetchStages);
    }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        if(num_loop == 1)
        {
            return TailNumber::One;
        }
        if(num_loop % PrefetchStages == 1)
        {
            return TailNumber::Three;
        }
        else
        {
            return TailNumber::Two;
        }
    }

    template <typename RunFunction>
    CK_TILE_HOST_DEVICE static auto
    TailHandler(const RunFunction& run_func, bool has_hot_loop, TailNumber tail_number)
    {
        // Use amd_wave_read_first_lane to avoid higher resource usage.
        // It forces to store these values in SGPR.
        // Compiler cannot deduce if one path is used for all threads
        const bool has_hot_loop_first_lane      = amd_wave_read_first_lane(has_hot_loop);
        const TailNumber tail_number_first_lane = amd_wave_read_first_lane(tail_number);
        // Handle all the valid cases.
        if(has_hot_loop_first_lane)
        {
            if(tail_number_first_lane == TailNumber::Three)
            {
                return run_func(bool_constant<true>{},
                                integral_constant<TailNumber, TailNumber::Three>{});
            }
            else if(tail_number_first_lane == TailNumber::Two)
            {
                return run_func(bool_constant<true>{},
                                integral_constant<TailNumber, TailNumber::Two>{});
            }
        }
        else
        {
            if(tail_number_first_lane == TailNumber::Three)
            {
                return run_func(bool_constant<false>{},
                                integral_constant<TailNumber, TailNumber::Three>{});
            }
            else if(tail_number_first_lane == TailNumber::Two)
            {
                return run_func(bool_constant<false>{},
                                integral_constant<TailNumber, TailNumber::Two>{});
            }
            else
            {
                return (run_func(bool_constant<false>{},
                                 integral_constant<TailNumber, TailNumber::One>{}));
            }
        }
        // If execution reaches here, it's an invalid tail_number because it wasn't handled above.
#if defined(__HIP_DEVICE_COMPILE__)
        __builtin_unreachable();
#else
        throw std::logic_error(
            "Invalid TailNumber: Only TailNumber::Three and TailNumber::Two are supported");
#endif
    }

    CK_TILE_HOST static constexpr auto GetName() { return "COMPUTE_ASYNC"; }
};

/**
 * @brief Compute optimized pipeline version async; which is based on V4.
 *
 * This pipeline introduces asynchronous load from global memory to LDS,
 * skipping the intermediate loading into pipeline registers.
 */
template <typename Problem, typename Policy = GemmPipelineAgBgCrCompAsyncDefaultPolicy<>>
struct GemmPipelineAgBgCrCompAsync : public BaseGemmPipelineAgBgCrCompAsync<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrCompAsync<Problem>;
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
    using I0        = number<0>;
    using I1        = number<1>;
    using I2        = number<2>;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr bool Async = true;

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return Policy::template GetVectorSizeA<Problem, IsWave32Host>();
    }
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return Policy::template GetVectorSizeB<Problem, IsWave32Host>();
    }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
    static constexpr index_t Preshuffle    = Problem::Preshuffle;

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;

    static_assert(DoubleSmemBuffer == true, "pipeline requires double smem buffer");

    static constexpr auto Scheduler = Problem::Scheduler;

    static constexpr auto is_a_load_tr_v = bool_constant<PipelineImplBase::is_a_load_tr>{};
    static constexpr auto is_b_load_tr_v = bool_constant<PipelineImplBase::is_b_load_tr>{};

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName()
    {
        // clang-format off
        return "COMPUTE_ASYNC";
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t smem_size = Policy::template GetSmemSize<Problem>();
        return 2 * smem_size;
    }

    CK_TILE_HOST_DEVICE static constexpr auto IsTransposeC()
    {
        return Policy::template IsTransposeC<Problem>();
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        // ---------------------------------------------------------------------
        // Asynchronous hot-loop scheduler used by the async direct-to-LDS path.
        // Only the (RowMajor A, ColumnMajor B) layout combination takes this
        // path; other layouts fall back to the synchronous scheduler below.
        // ---------------------------------------------------------------------
        CK_TILE_DEVICE static constexpr auto HotLoopSchedulerAsync()
        {
            constexpr index_t MPerXDL = BlockGemmShape::WarpTile::at(I0{});
            constexpr index_t NPerXDL = BlockGemmShape::WarpTile::at(I1{});
            constexpr index_t KPerXDL = BlockGemmShape::WarpTile::at(I2{});

            constexpr index_t WaveSize = get_warp_size();

            constexpr index_t A_Buffer_Load_Inst_Num =
                MPerBlock * KPerBlock / (BlockSize * GetVectorSizeA());
            constexpr index_t B_Buffer_Load_Inst_Num =
                NPerBlock * KPerBlock / (BlockSize * GetVectorSizeB());

            constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                                (BlockSize / WaveSize) /
                                                (MPerXDL * NPerXDL * KPerXDL);

            constexpr auto num_buffer_load_inst = A_Buffer_Load_Inst_Num + B_Buffer_Load_Inst_Num;
            constexpr auto num_issue            = num_buffer_load_inst;

            static_for<0, num_buffer_load_inst, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0); // MFMA : 1
                __builtin_amdgcn_sched_group_barrier(
                    LLVMSchedGroupMask::DS_READ, 1, 0);                               // DS read : 1
                __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0); // MFMA: 1
                __builtin_amdgcn_sched_group_barrier(
                    LLVMSchedGroupMask::VMEM_READ, 1, 0); // VMEM read :1
                __builtin_amdgcn_sched_group_barrier(
                    LLVMSchedGroupMask::MFMA, C_MFMA_Inst_Num / num_issue - 2, 0); // MFMA : 6
            });
            __builtin_amdgcn_sched_barrier(0);
        }

        // ---------------------------------------------------------------------
        // Synchronous hot-loop scheduler used when the layout combination is
        // not eligible for the async direct-to-LDS path.
        // ---------------------------------------------------------------------
        CK_TILE_DEVICE static constexpr auto HotLoopSchedulerSync()
        {
            constexpr index_t MPerXDL = BlockGemmShape::WarpTile::at(I0{});
            constexpr index_t NPerXDL = BlockGemmShape::WarpTile::at(I1{});
            constexpr index_t KPerXDL = BlockGemmShape::WarpTile::at(I2{});

            constexpr index_t WaveSize = get_warp_size();
            constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
            constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});

            constexpr index_t A_LDS_Read_Width = KPerXDL;
            constexpr index_t B_LDS_Read_Width = KPerXDL;

            constexpr index_t A_Buffer_Load_Inst_Num =
                MPerBlock * KPerBlock / (BlockSize * GetVectorSizeA());
            constexpr index_t B_Buffer_Load_Inst_Num =
                NPerBlock * KPerBlock / (BlockSize * GetVectorSizeB());

            constexpr index_t A_LDS_Write_Inst_Num = MPerBlock * KPerBlock / (BlockSize * KPerXDL);
            constexpr index_t B_LDS_Write_Inst_Num = NPerBlock * KPerBlock / (BlockSize * KPerXDL);

            constexpr index_t A_LDS_Read_Inst_Num =
                WaveNumN * MPerBlock * KPerBlock / (BlockSize * KPerXDL);
            constexpr index_t B_LDS_Read_Inst_Num =
                WaveNumM * NPerBlock * KPerBlock / (BlockSize * KPerXDL);

            constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                                (BlockSize / WaveSize) /
                                                (MPerXDL * NPerXDL * KPerXDL);

            constexpr auto num_ds_read_inst_a =
                A_LDS_Read_Width * sizeof(ADataType) / APackedSize == 16 ? A_LDS_Read_Inst_Num
                                                                         : A_LDS_Read_Inst_Num / 2;
            constexpr auto num_ds_read_inst_b =
                B_LDS_Read_Width * sizeof(BDataType) / BPackedSize == 16 ? B_LDS_Read_Inst_Num
                                                                         : B_LDS_Read_Inst_Num / 2;

            constexpr auto num_ds_read_inst     = num_ds_read_inst_a + num_ds_read_inst_b;
            constexpr auto num_ds_write_inst    = A_LDS_Write_Inst_Num + B_LDS_Write_Inst_Num;
            constexpr auto num_buffer_load_inst = A_Buffer_Load_Inst_Num + B_Buffer_Load_Inst_Num;
            constexpr auto num_issue            = num_buffer_load_inst;

            static_for<0, num_buffer_load_inst, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA : 1
                __builtin_amdgcn_sched_group_barrier(
                    0x100, num_ds_read_inst / num_issue, 0);       // DS read : 2
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA: 1
                __builtin_amdgcn_sched_group_barrier(
                    0x200, num_ds_write_inst / num_issue, 0);      // DS write : 1
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA : 1
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read :1
                __builtin_amdgcn_sched_group_barrier(
                    0x008, C_MFMA_Inst_Num / num_issue - 3, 0); // MFMA : 5
            });
            __builtin_amdgcn_sched_barrier(0);
        }

        // ---------------------------------------------------------------------
        // Synchronous device implementation (sync DRAM→register→LDS→register).
        // Used for layouts that the async direct-to-LDS path does not support.
        // Ported from GemmPipelineAgBgCrCompV4::RunSync.
        // ---------------------------------------------------------------------
        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename AsDramBlockWindowTmp,
                  typename BsDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction>
        CK_TILE_DEVICE auto RunSync(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                    const AElementFunction& a_element_func,
                                    const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                    const BElementFunction& b_element_func,
                                    index_t num_loop,
                                    void* __restrict__ p_smem) const
        {
            using ADramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, AsDramBlockWindowTmp>>;
            using BDramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, BsDramBlockWindowTmp>>;

            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "Data Type conflict on A and B matrix input data type.");

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

            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

            ////////////// LDS desc, window & register /////////////////
            constexpr index_t smem_size         = Policy::template GetSmemSize<Problem>();
            auto&& [a_lds_block0, b_lds_block0] = Base::GetABLdsTensorViews(p_smem);
            auto&& [a_lds_block1, b_lds_block1] =
                Base::GetABLdsTensorViews(static_cast<char*>(p_smem) + smem_size);

            constexpr auto a_lds_shape = []() {
                if constexpr(is_a_load_tr_v())
                    return make_tuple(number<KPerBlock>{}, number<MPerBlock>{});
                else
                    return make_tuple(number<MPerBlock>{}, number<KPerBlock>{});
            }();
            auto a_copy_lds_window0 = make_tile_window(a_lds_block0, a_lds_shape, {0, 0});
            auto a_copy_lds_window1 = make_tile_window(a_lds_block1, a_lds_shape, {0, 0});

            constexpr auto b_lds_shape = []() {
                if constexpr(is_b_load_tr_v())
                    return make_tuple(number<KPerBlock>{}, number<NPerBlock>{});
                else
                    return make_tuple(number<NPerBlock>{}, number<KPerBlock>{});
            }();
            auto b_copy_lds_window0 = make_tile_window(b_lds_block0, b_lds_shape, {0, 0});
            auto b_copy_lds_window1 = make_tile_window(b_lds_block1, b_lds_shape, {0, 0});

            // Block GEMM
            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // DRAM tile windows for A
            auto a_tile_windows = generate_tuple(
                [&](auto idx) {
                    return make_tile_window(
                        a_dram_block_window_tmp[number<idx>{}].get_bottom_tensor_view(),
                        make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                        a_dram_block_window_tmp[number<idx>{}].get_window_origin(),
                        Policy::template MakeADramTileDistribution<Problem>());
                },
                number<AsLayout::size()>{});

            auto elementwise_As_res = load_tile_with_elementwise(a_tile_windows, a_element_func);
            move_tile_window(a_tile_windows, a_dram_tile_window_step);

            // DRAM tile windows for B
            auto b_tile_windows = generate_tuple(
                [&](auto idx) {
                    return make_tile_window(
                        b_dram_block_window_tmp[number<idx>{}].get_bottom_tensor_view(),
                        make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                        b_dram_block_window_tmp[number<idx>{}].get_window_origin(),
                        Policy::template MakeBDramTileDistribution<Problem>());
                },
                number<AsLayout::size()>{});

            auto elementwise_Bs_res = load_tile_with_elementwise(b_tile_windows, b_element_func);
            move_tile_window(b_tile_windows, b_dram_tile_window_step);

            // LDS write 0
            if constexpr(is_a_col_major && !is_a_load_tr_v())
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp, elementwise_As_res);
                Base::LocalPrefill(a_copy_lds_window0, a_shuffle_tmp);
            }
            else
            {
                Base::LocalPrefill(a_copy_lds_window0, elementwise_As_res);
            }
            if constexpr(is_b_row_major && !is_b_load_tr_v())
            {
                auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b_shuffle_tmp, elementwise_Bs_res);
                Base::LocalPrefill(b_copy_lds_window0, b_shuffle_tmp);
            }
            else
            {
                Base::LocalPrefill(b_copy_lds_window0, elementwise_Bs_res);
            }

            // global read 1
            elementwise_As_res = load_tile_with_elementwise(a_tile_windows, a_element_func);
            move_tile_window(a_tile_windows, a_dram_tile_window_step);
            elementwise_Bs_res = load_tile_with_elementwise(b_tile_windows, b_element_func);
            move_tile_window(b_tile_windows, b_dram_tile_window_step);
            block_sync_lds();

            constexpr auto ALdsTileDistr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto BLdsTileDistr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            using ALdsTile = decltype(make_static_distributed_tensor<ADataType>(ALdsTileDistr));
            using BLdsTile = decltype(make_static_distributed_tensor<BDataType>(BLdsTileDistr));
            ALdsTile a_block_tile0, a_block_tile1;
            BLdsTile b_block_tile0, b_block_tile1;

            constexpr auto a_lds_input_tile_distr = [&]() {
                if constexpr(is_a_load_tr_v())
                    return make_static_tile_distribution(
                        typename InputTileDistributionTraits<
                            decltype(BlockGemm::MakeABlockDistributionEncode()),
                            typename Problem::ADataType>::TransposedDstrEncode{});
                else
                    return ALdsTileDistr;
            }();
            constexpr auto b_lds_input_tile_distr = [&]() {
                if constexpr(is_b_load_tr_v())
                    return make_static_tile_distribution(
                        typename InputTileDistributionTraits<
                            decltype(BlockGemm::MakeBBlockDistributionEncode()),
                            typename Problem::BDataType>::TransposedDstrEncode{});
                else
                    return BLdsTileDistr;
            }();
            auto a_lds_ld_window0 =
                make_tile_window(a_lds_block0, a_lds_shape, {0, 0}, a_lds_input_tile_distr);
            auto a_lds_ld_window1 =
                make_tile_window(a_lds_block1, a_lds_shape, {0, 0}, a_lds_input_tile_distr);
            auto b_lds_ld_window0 =
                make_tile_window(b_lds_block0, b_lds_shape, {0, 0}, b_lds_input_tile_distr);
            auto b_lds_ld_window1 =
                make_tile_window(b_lds_block1, b_lds_shape, {0, 0}, b_lds_input_tile_distr);

            static_assert(!is_tile_window_linear_v<decltype(a_lds_ld_window0)> &&
                              !is_tile_window_linear_v<decltype(a_lds_ld_window1)> &&
                              !is_tile_window_linear_v<decltype(b_lds_ld_window0)> &&
                              !is_tile_window_linear_v<decltype(b_lds_ld_window1)>,
                          "LDS windows must not be linear");

            Base::LocalPrefetch(a_block_tile0, a_lds_ld_window0, is_a_load_tr_v);
            Base::LocalPrefetch(b_block_tile0, b_lds_ld_window0, is_b_load_tr_v);

            if constexpr(is_a_col_major && !is_a_load_tr_v())
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp, elementwise_As_res);
                Base::LocalPrefill(a_copy_lds_window1, a_shuffle_tmp);
            }
            else
            {
                Base::LocalPrefill(a_copy_lds_window1, elementwise_As_res);
            }
            if constexpr(is_b_row_major && !is_b_load_tr_v())
            {
                auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b_shuffle_tmp, elementwise_Bs_res);
                Base::LocalPrefill(b_copy_lds_window1, b_shuffle_tmp);
            }
            else
            {
                Base::LocalPrefill(b_copy_lds_window1, elementwise_Bs_res);
            }

            elementwise_As_res = load_tile_with_elementwise(a_tile_windows, a_element_func);
            move_tile_window(a_tile_windows, a_dram_tile_window_step);
            elementwise_Bs_res = load_tile_with_elementwise(b_tile_windows, b_element_func);
            move_tile_window(b_tile_windows, b_dram_tile_window_step);

            if constexpr(HasHotLoop)
            {
                index_t iCounter = amd_wave_read_first_lane(num_loop - 2);
                do
                {
                    // ping
                    {
                        block_sync_lds();
                        Base::LocalPrefetch(a_block_tile1, a_lds_ld_window1, is_a_load_tr_v);
                        Base::LocalPrefetch(b_block_tile1, b_lds_ld_window1, is_b_load_tr_v);

                        if constexpr(is_a_col_major && !is_a_load_tr_v())
                        {
                            auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                                Policy::template MakeShuffledARegTileDistribution<Problem>());
                            transpose_tile2d(a_shuffle_tmp, elementwise_As_res);
                            Base::LocalPrefill(a_copy_lds_window0, a_shuffle_tmp);
                        }
                        else
                        {
                            Base::LocalPrefill(a_copy_lds_window0, elementwise_As_res);
                        }
                        if constexpr(is_b_row_major && !is_b_load_tr_v())
                        {
                            auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                                Policy::template MakeShuffledBRegTileDistribution<Problem>());
                            transpose_tile2d(b_shuffle_tmp, elementwise_Bs_res);
                            Base::LocalPrefill(b_copy_lds_window0, b_shuffle_tmp);
                        }
                        else
                        {
                            Base::LocalPrefill(b_copy_lds_window0, elementwise_Bs_res);
                        }

                        elementwise_As_res =
                            load_tile_with_elementwise(a_tile_windows, a_element_func);
                        move_tile_window(a_tile_windows, a_dram_tile_window_step);
                        elementwise_Bs_res =
                            load_tile_with_elementwise(b_tile_windows, b_element_func);
                        move_tile_window(b_tile_windows, b_dram_tile_window_step);

                        block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
                        HotLoopSchedulerSync();
                    }
                    // pong
                    {
                        block_sync_lds();
                        Base::LocalPrefetch(a_block_tile0, a_lds_ld_window0, is_a_load_tr_v);
                        Base::LocalPrefetch(b_block_tile0, b_lds_ld_window0, is_b_load_tr_v);

                        if constexpr(is_a_col_major && !is_a_load_tr_v())
                        {
                            auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                                Policy::template MakeShuffledARegTileDistribution<Problem>());
                            transpose_tile2d(a_shuffle_tmp, elementwise_As_res);
                            Base::LocalPrefill(a_copy_lds_window1, a_shuffle_tmp);
                        }
                        else
                        {
                            Base::LocalPrefill(a_copy_lds_window1, elementwise_As_res);
                        }
                        if constexpr(is_b_row_major && !is_b_load_tr_v())
                        {
                            auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                                Policy::template MakeShuffledBRegTileDistribution<Problem>());
                            transpose_tile2d(b_shuffle_tmp, elementwise_Bs_res);
                            Base::LocalPrefill(b_copy_lds_window1, b_shuffle_tmp);
                        }
                        else
                        {
                            Base::LocalPrefill(b_copy_lds_window1, elementwise_Bs_res);
                        }
                        block_sync_lds();

                        elementwise_As_res =
                            load_tile_with_elementwise(a_tile_windows, a_element_func);
                        move_tile_window(a_tile_windows, a_dram_tile_window_step);
                        elementwise_Bs_res =
                            load_tile_with_elementwise(b_tile_windows, b_element_func);
                        move_tile_window(b_tile_windows, b_dram_tile_window_step);

                        block_gemm(c_block_tile, a_block_tile1, b_block_tile1);
                        HotLoopSchedulerSync();
                    }
                    iCounter -= 2;
                } while(iCounter > 1);
            }

            // tail 3
            if(TailNum == TailNumber::Three)
            {
                {
                    block_sync_lds();
                    Base::LocalPrefetch(a_block_tile1, a_lds_ld_window1, is_a_load_tr_v);
                    Base::LocalPrefetch(b_block_tile1, b_lds_ld_window1, is_b_load_tr_v);
                    if constexpr(is_a_col_major && !is_a_load_tr_v())
                    {
                        auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                            Policy::template MakeShuffledARegTileDistribution<Problem>());
                        transpose_tile2d(a_shuffle_tmp, elementwise_As_res);
                        Base::LocalPrefill(a_copy_lds_window0, a_shuffle_tmp);
                    }
                    else
                    {
                        Base::LocalPrefill(a_copy_lds_window0, elementwise_As_res);
                    }
                    if constexpr(is_b_row_major && !is_b_load_tr_v())
                    {
                        auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                            Policy::template MakeShuffledBRegTileDistribution<Problem>());
                        transpose_tile2d(b_shuffle_tmp, elementwise_Bs_res);
                        Base::LocalPrefill(b_copy_lds_window0, b_shuffle_tmp);
                    }
                    else
                    {
                        Base::LocalPrefill(b_copy_lds_window0, elementwise_Bs_res);
                    }
                    block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
                }
                {
                    block_sync_lds();
                    Base::LocalPrefetch(a_block_tile0, a_lds_ld_window0, is_a_load_tr_v);
                    Base::LocalPrefetch(b_block_tile0, b_lds_ld_window0, is_b_load_tr_v);
                    block_gemm(c_block_tile, a_block_tile1, b_block_tile1);
                }
                {
                    block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
                    __builtin_amdgcn_sched_barrier(0);
                }
            }
            else if(TailNum == TailNumber::Two)
            {
                {
                    block_sync_lds();
                    Base::LocalPrefetch(a_block_tile1, a_lds_ld_window1, is_a_load_tr_v);
                    Base::LocalPrefetch(b_block_tile1, b_lds_ld_window1, is_b_load_tr_v);
                    block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
                    static_for<0, 8, 1>{}([&](auto) {
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                        __builtin_amdgcn_sched_group_barrier(0x008, 8, 0); // MFMA
                    });
                    __builtin_amdgcn_sched_barrier(0);
                }
                {
                    block_gemm(c_block_tile, a_block_tile1, b_block_tile1);
                    __builtin_amdgcn_sched_barrier(0);
                }
            }
            else if(TailNum == TailNumber::One)
            {
                block_sync_lds();
                block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
                __builtin_amdgcn_sched_barrier(0);
            }
            return c_block_tile;
        }

        // ---------------------------------------------------------------------
        // Asynchronous device implementation (direct DRAM→LDS).
        // Limited to ALayout=RowMajor, BLayout=ColumnMajor; the public
        // dispatcher routes other layouts through RunSync above.
        // ---------------------------------------------------------------------
        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename AsDramBlockWindowTmp,
                  typename BsDramBlockWindowTmp>
        CK_TILE_DEVICE auto RunAsync(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                     const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                     index_t num_loop,
                                     void* __restrict__ p_smem) const
        {
            // TODO support multi-ABD
            static_assert(1 == std::tuple_size_v<AsDramBlockWindowTmp>);
            static_assert(1 == std::tuple_size_v<BsDramBlockWindowTmp>);
            using ADramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, AsDramBlockWindowTmp>>;
            using BDramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, BsDramBlockWindowTmp>>;
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "Data Type conflict on A and B matrix input data type.");

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

            ////////////// global window & register /////////////////
            // A DRAM tile window(s) for async byte-based load
            auto a_tile_windows = generate_tuple(
                [&](auto idx) {
                    return Policy::template MakeAAsyncLoadBytesDramWindow<Problem>(
                        a_dram_block_window_tmp[number<idx>{}]);
                },
                number<AsLayout::size()>{});
            // B DRAM tile window(s) for async byte-based load
            auto b_tile_windows = generate_tuple(
                [&](auto idx) {
                    return Policy::template MakeBAsyncLoadBytesDramWindow<Problem>(
                        b_dram_block_window_tmp[number<idx>{}]);
                },
                number<BsLayout::size()>{});

            // this pipeline has a pair of LDS buffers per logical tile
            constexpr index_t smem_size = Policy::template GetSmemSize<Problem>();
            constexpr index_t a_lds_block_space_size_aligned =
                Policy::template GetSmemSizeA<Problem, ADataType>();

            // Create LDS store tensor views for buffer 0
            auto a_lds_store_block0 =
                Policy::template GetALdsStoreTensorView<Problem, ADataType>(p_smem);
            auto b_lds_store_block0 = Policy::template GetBLdsStoreTensorView<Problem, BDataType>(
                static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));

            // Create LDS store tensor views for buffer 1
            void* p_smem_buffer1 = static_cast<char*>(p_smem) + smem_size;
            auto a_lds_store_block1 =
                Policy::template GetALdsStoreTensorView<Problem, ADataType>(p_smem_buffer1);
            auto b_lds_store_block1 =
                Policy::template GetBLdsStoreTensorView<Problem, BDataType>(static_cast<void*>(
                    static_cast<char*>(p_smem_buffer1) + a_lds_block_space_size_aligned));

            // Create LDS load tensor views for buffer 0
            auto a_lds_load_block0 =
                Policy::template GetALdsLoadTensorView<Problem, ADataType>(p_smem);
            auto b_lds_load_block0 = Policy::template GetBLdsLoadTensorView<Problem, BDataType>(
                static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));

            // Create LDS load tensor views for buffer 1
            auto a_lds_load_block1 =
                Policy::template GetALdsLoadTensorView<Problem, ADataType>(p_smem_buffer1);
            auto b_lds_load_block1 =
                Policy::template GetBLdsLoadTensorView<Problem, BDataType>(static_cast<void*>(
                    static_cast<char*>(p_smem_buffer1) + a_lds_block_space_size_aligned));

            // set up LDS tile shapes
            constexpr auto a_lds_shape = []() {
                if constexpr(is_a_load_tr_v)
                    return make_tuple(number<KPerBlock>{}, number<MPerBlock>{});
                else
                    return make_tuple(number<MPerBlock>{}, number<KPerBlock>{});
            }();

            constexpr auto b_lds_shape = []() {
                if constexpr(is_b_load_tr_v)
                    return make_tuple(number<KPerBlock>{}, number<NPerBlock>{});
                else
                    return make_tuple(number<NPerBlock>{}, number<KPerBlock>{});
            }();

            // LDS tile windows for storing, one per LDS buffer
            auto a_copy_lds_window0 = make_tile_window(a_lds_store_block0, a_lds_shape, {0, 0});

            auto a_copy_lds_window1 = make_tile_window(a_lds_store_block1, a_lds_shape, {0, 0});

            auto b_copy_lds_window0 = make_tile_window(b_lds_store_block0, b_lds_shape, {0, 0});

            auto b_copy_lds_window1 = make_tile_window(b_lds_store_block1, b_lds_shape, {0, 0});

            // initialize DRAM window steps for byte-based windows
            // Note: byte-based windows already account for data type packing
            const auto a_dram_tile_window_step =
                array<index_t, 2>{number<0>{}, number<KPerBlock>{}};
            const auto b_dram_tile_window_step =
                array<index_t, 2>{number<0>{}, number<KPerBlock>{}};

            // Define async load tile lambda
            auto async_load_tile_ = [](auto lds, auto dram) {
                async_load_tile(lds, dram, number<-1>{}, true_type{}, true_type{});
            };

            // read A(0), B(0) from DRAM to LDS window(0)
            // and advance the DRAM windows
            async_load_tile_(a_copy_lds_window0, a_tile_windows[number<0>{}]);
            move_tile_window(a_tile_windows[number<0>{}], a_dram_tile_window_step);
            async_load_tile_(b_copy_lds_window0, b_tile_windows[number<0>{}]);
            move_tile_window(b_tile_windows[number<0>{}], b_dram_tile_window_step);

            // initialize block gemm
            auto block_gemm = BlockGemm();

            // initialize C block tile
            auto c_block_tile = block_gemm.MakeCBlockTile();
            clear_tile(c_block_tile);

            // read A(1), B(1) from DRAM to LDS window(1)
            // and advance the DRAM windows
            async_load_tile_(a_copy_lds_window1, a_tile_windows[number<0>{}]);
            move_tile_window(a_tile_windows[number<0>{}], a_dram_tile_window_step);
            async_load_tile_(b_copy_lds_window1, b_tile_windows[number<0>{}]);
            move_tile_window(b_tile_windows[number<0>{}], b_dram_tile_window_step);

            // tile distribution for the register tiles
            constexpr auto ALdsTileDistr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto BLdsTileDistr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            using ALdsTile = decltype(make_static_distributed_tensor<ADataType>(ALdsTileDistr));
            using BLdsTile = decltype(make_static_distributed_tensor<BDataType>(BLdsTileDistr));

            // register tiles; double buffering -> a register tile corresponds to a LDS tile window
            ALdsTile a_block_tile0, a_block_tile1;
            BLdsTile b_block_tile0, b_block_tile1;

            constexpr auto a_lds_input_tile_distr = [ALdsTileDistr]() {
                if constexpr(is_a_load_tr_v)
                    return make_static_tile_distribution(
                        typename InputTileDistributionTraits<
                            typename decltype(ALdsTileDistr)::DstrEncode,
                            typename Problem::ADataType>::TransposedDstrEncode{});
                else
                    return ALdsTileDistr;
            }();
            constexpr auto b_lds_input_tile_distr = [BLdsTileDistr]() {
                if constexpr(is_b_load_tr_v)
                    return make_static_tile_distribution(
                        typename InputTileDistributionTraits<
                            typename decltype(BLdsTileDistr)::DstrEncode,
                            typename Problem::BDataType>::TransposedDstrEncode{});
                else
                    return BLdsTileDistr;
            }();

            // LDS tile windows for reading;
            // they share the data pointer with the LDS windows for storing
            // but also associate with a distribution to produce a register tile when reading
            auto a_lds_ld_window0 =
                make_tile_window(a_lds_load_block0, a_lds_shape, {0, 0}, a_lds_input_tile_distr);
            auto a_lds_ld_window1 =
                make_tile_window(a_lds_load_block1, a_lds_shape, {0, 0}, a_lds_input_tile_distr);
            auto b_lds_ld_window0 =
                make_tile_window(b_lds_load_block0, b_lds_shape, {0, 0}, b_lds_input_tile_distr);
            auto b_lds_ld_window1 =
                make_tile_window(b_lds_load_block1, b_lds_shape, {0, 0}, b_lds_input_tile_distr);

            static_assert(!(is_tile_window_linear_v<decltype(a_lds_ld_window0)>) &&
                              !(is_tile_window_linear_v<decltype(a_lds_ld_window1)>) &&
                              !(is_tile_window_linear_v<decltype(b_lds_ld_window0)>) &&
                              !(is_tile_window_linear_v<decltype(b_lds_ld_window1)>),
                          "LDS windows must not be linear");

            // write to LDS window(0) must complete before the local prefetch
            block_sync_lds_direct_load();
            // read A(0), B(0) from LDS window(0) to pipeline registers(0)
            Base::LocalPrefetch(a_block_tile0, a_lds_ld_window0, is_a_load_tr_v);
            Base::LocalPrefetch(b_block_tile0, b_lds_ld_window0, is_b_load_tr_v);
            // LDS window(0) contents are overwritten below by global prefetch, need to sync
            block_sync_lds();
            // read A(2), B(2) from DRAM to LDS window(0)
            // and advance the DRAM windows
            if constexpr((!HasHotLoop && (TailNum == TailNumber::Three)) || HasHotLoop)
            {
                async_load_tile_(a_copy_lds_window0, a_tile_windows[number<0>{}]);
                move_tile_window(a_tile_windows[number<0>{}], a_dram_tile_window_step);
                async_load_tile_(b_copy_lds_window0, b_tile_windows[number<0>{}]);
                move_tile_window(b_tile_windows[number<0>{}], b_dram_tile_window_step);
            }

            if constexpr(HasHotLoop)
            {
                // we have had 3 global prefetches so far, indexed (0, 1, 2).
                index_t i_global_read = amd_wave_read_first_lane(3);
                // alternate ping: (read to register tile(1), use register tile(0) as gemm input)
                //           pong: (read to register tile(0), use register tile(1) as gemm input)
                do
                {
                    // ping
                    {
                        // read A(i-1), B(i-1) from LDS window(1) to pipeline registers(1)
                        Base::LocalPrefetch(a_block_tile1, a_lds_ld_window1, is_a_load_tr_v);
                        Base::LocalPrefetch(b_block_tile1, b_lds_ld_window1, is_b_load_tr_v);
                        // LDS window(1) contents are overwritten by global prefetch, need to sync
                        block_sync_lds();
                        // read A(i), B(i) from DRAM to LDS window(1)
                        // and advance the DRAM windows
                        async_load_tile_(a_copy_lds_window1, a_tile_windows[number<0>{}]);
                        move_tile_window(a_tile_windows[number<0>{}], a_dram_tile_window_step);
                        async_load_tile_(b_copy_lds_window1, b_tile_windows[number<0>{}]);
                        move_tile_window(b_tile_windows[number<0>{}], b_dram_tile_window_step);
                        // C(i-3) = A(i-3) @ B(i-3)
                        block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
                        HotLoopSchedulerAsync();
                    }
                    // pong
                    {
                        // write to LDS window(0) must complete before the local prefetch
                        block_sync_lds_direct_load();
                        // read A(i), B(i) from LDS window(0) to pipeline registers(0)
                        Base::LocalPrefetch(a_block_tile0, a_lds_ld_window0, is_a_load_tr_v);
                        Base::LocalPrefetch(b_block_tile0, b_lds_ld_window0, is_b_load_tr_v);
                        // LDS window(0) contents are overwritten by global prefetch, need to sync
                        block_sync_lds();
                        // read A(i+1), B(i+1) from DRAM to LDS window(0)
                        // and advance the DRAM windows
                        async_load_tile_(a_copy_lds_window0, a_tile_windows[number<0>{}]);
                        move_tile_window(a_tile_windows[number<0>{}], a_dram_tile_window_step);
                        async_load_tile_(b_copy_lds_window0, b_tile_windows[number<0>{}]);
                        move_tile_window(b_tile_windows[number<0>{}], b_dram_tile_window_step);
                        // C(i-2) = A(i-2) @ B(i-2)
                        block_gemm(c_block_tile, a_block_tile1, b_block_tile1);
                        HotLoopSchedulerAsync();
                    }
                    i_global_read += 2;
                } while(i_global_read < num_loop);
            }

            // 3 block gemms remaining
            if constexpr(TailNum == TailNumber::Three)
            {
                {
                    // read A(num_loop-1), B(num_loop-1) from LDS window(1) to pipeline registers(1)
                    Base::LocalPrefetch(a_block_tile1, a_lds_ld_window1, is_a_load_tr_v);
                    Base::LocalPrefetch(b_block_tile1, b_lds_ld_window1, is_b_load_tr_v);
                    // C(num_loop-2) = A(num_loop-2) @ B(num_loop-2)
                    block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
                }
                {
                    // write to LDS window(0) must complete before the local prefetch
                    block_sync_lds_direct_load();
                    // read A(num_loop), B(num_loop) from LDS window(0) to pipeline registers(0)
                    Base::LocalPrefetch(a_block_tile0, a_lds_ld_window0, is_a_load_tr_v);
                    Base::LocalPrefetch(b_block_tile0, b_lds_ld_window0, is_b_load_tr_v);
                    // C(num_loop-1) = A(num_loop-1) @ B(num_loop-1)
                    block_gemm(c_block_tile, a_block_tile1, b_block_tile1);
                }
                {
                    // C(num_loop) = A(num_loop) @ B(num_loop)
                    block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
                }
            }
            else if(TailNum == TailNumber::Two)
            // 2 block gemms remaining
            {
                {
                    // read A(num_loop), B(num_loop) from LDS window(1) to pipeline registers(1)
                    Base::LocalPrefetch(a_block_tile1, a_lds_ld_window1, is_a_load_tr_v);
                    Base::LocalPrefetch(b_block_tile1, b_lds_ld_window1, is_b_load_tr_v);
                    // C(num_loop-1) = A(num_loop-1) @ B(num_loop-1)
                    block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
                }
                {
                    // C(num_loop) = A(num_loop) @ B(num_loop)
                    block_gemm(c_block_tile, a_block_tile1, b_block_tile1);
                }
            }
            else if(TailNum == TailNumber::One)
            {
                block_sync_lds();
                block_gemm(c_block_tile, a_block_tile0, b_block_tile0);
                __builtin_amdgcn_sched_barrier(0);
            }
            return c_block_tile;
        }

        // ---------------------------------------------------------------------
        // Public per-stage entry. Dispatches to RunAsync for single-AB-tuple,
        // PassThrough element functions, ALayout=RowMajor, BLayout=ColumnMajor;
        // all other shapes fall back to RunSync because async direct-to-LDS
        // does not perform the register-side transpose those layouts need.
        // ---------------------------------------------------------------------
        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename AsDramBlockWindowTmp,
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
                                       void* __restrict__ p_smem) const
        {
            constexpr bool can_async =
                (std::tuple_size_v<AsDramBlockWindowTmp> == 1) &&
                (std::tuple_size_v<BsDramBlockWindowTmp> == 1) &&
                std::is_same_v<remove_cvref_t<AElementFunction>, element_wise::PassThrough> &&
                std::is_same_v<remove_cvref_t<BElementFunction>, element_wise::PassThrough> &&
                std::is_same_v<ALayout, tensor_layout::gemm::RowMajor> &&
                std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>;
            if constexpr(can_async)
            {
                ignore = a_element_func;
                ignore = b_element_func;
                return RunAsync<HasHotLoop, TailNum>(
                    a_dram_block_window_tmp, b_dram_block_window_tmp, num_loop, p_smem);
            }
            else
            {
                return RunSync<HasHotLoop, TailNum>(a_dram_block_window_tmp,
                                                    a_element_func,
                                                    b_dram_block_window_tmp,
                                                    b_element_func,
                                                    num_loop,
                                                    p_smem);
            }
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
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);
        const auto RunPipeline  = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp,
                a_element_func,
                b_dram_block_window_tmp,
                b_element_func,
                num_loop,
                p_smem);
        };

        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }

    public:
    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const index_t num_loop,
                                   void* __restrict__ p_smem) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);

        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp,
                element_wise::PassThrough{},
                b_dram_block_window_tmp,
                element_wise::PassThrough{},
                num_loop,
                p_smem);
        };

        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction,
              typename std::enable_if_t<!is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);
        const auto RunPipeline  = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                ck_tile::make_tuple(a_dram_block_window_tmp),
                a_element_func,
                ck_tile::make_tuple(b_dram_block_window_tmp),
                b_element_func,
                num_loop,
                p_smem);
        };

        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }

    public:
    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename std::enable_if_t<!is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const index_t num_loop,
                                   void* __restrict__ p_smem) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);

        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                ck_tile::make_tuple(a_dram_block_window_tmp),
                element_wise::PassThrough{},
                ck_tile::make_tuple(b_dram_block_window_tmp),
                element_wise::PassThrough{},
                num_loop,
                p_smem);
        };

        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});
        return concat('_', "pipeline_AgBgCrCompAsync", 
                      concat('x', MPerBlock, NPerBlock, KPerBlock),  BlockSize,
                      concat('x', WaveNumM, WaveNumN),
                      concat('x', kPadM, kPadN, kPadK));
        // clang-format on
    }
};
} // namespace ck_tile
