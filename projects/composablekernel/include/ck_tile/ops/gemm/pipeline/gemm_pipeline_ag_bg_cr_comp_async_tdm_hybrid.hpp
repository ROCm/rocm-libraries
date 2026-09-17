// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_tdm_hybrid_policy.hpp"

namespace ck_tile {

// A Tile Window: global memory
// B Tile Window: global memory
// C Distributed tensor: register
//
// Task 9 (docs_gfx1250/GFX1250_WRW_REMAINING_TASKS.md): pairs `global_load_async` for A
// with TDM (tensor data mover) for B in a single compute-optimized pipeline.
//
// This is a ping-pong (double-buffered) pipeline built by taking
// `GemmPipelineAgBgCrCompAsync` as the base shape (same PrefetchStages/UnrollHotLoop/
// TailNumber cadence, reused unchanged via `BaseGemmPipelineAgBgCrCompAsync`) and
// replacing every B-operand `GlobalPrefetchAsync` call with `GlobalPrefetchTDM`, using
// `GemmPipelineAgBgCrCompAsyncTDMHybridDefaultPolicy` (B keeps TDM's own coarse DRAM
// distribution + bank-conflict-avoiding LDS descriptor; A keeps Async's generic
// per-lane distribution, which for bf16 on gfx125 is exactly what
// `GemmPipelineAgBgCrCompAsyncDefaultPolicy` already reduces to).
//
// Synchronization: TDM completion is tracked by a *separate* hardware counter
// (`tensorcnt`, waited via `s_wait_tensorcnt<0>()`) from the async copy's completion
// (`asynccnt`, waited via `block_sync_lds_direct_load()`). The original
// `GemmPipelineAgBgCrCompAsync` loop only calls `block_sync_lds_direct_load()` at two
// of its four LDS-buffer read points, relying on `asynccnt` being a single hardware
// counter that transitively drains *all* outstanding async ops (both buffers) whenever
// it is waited on once -- so the untouched two read points are safe without their own
// explicit wait. That transitivity does not hold across the two independent counters:
// `s_wait_tensorcnt<0>()` says nothing about `asynccnt` and vice versa. So this pipeline
// explicitly waits `tensorcnt` at *every* point that reads a buffer TDM last wrote
// (all four read points), while `asynccnt` keeps the original two-point placement.
// Neither TDM nor async ever has more than one outstanding op per buffer in this
// design (every reload is issued only after the previous occupant's reads have been
// synced via `block_sync_lds()`), so `s_wait_tensorcnt<0>()` (full drain) is exactly
// the right primitive at each of those points, not merely a throttle.
//
// Not implemented (explicitly out of scope for this first version -- see Task 9 in
// the roadmap doc for the reasoning):
//  - MX microscaling (no `Policy::MakeMX_Scale{A,B}_DramTileDistribution`).
//  - TDM cluster-launch / cross-CU multicast for B (`UseClusterLaunch` hardcoded false).
//  - K-subtiling (`sub_tile_num` hardcoded to 1 by the policy's `GetBlockGemm`).
template <typename Problem, typename Policy = GemmPipelineAgBgCrCompAsyncTDMHybridDefaultPolicy>
struct GemmPipelineAgBgCrCompAsyncTDMHybrid : public BaseGemmPipelineAgBgCrCompAsync<Problem>
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

    using I0 = number<0>;
    using I1 = number<1>;
    using I2 = number<2>;

    static constexpr bool LargeTensors = Problem::LargeTensors;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    // A operand is async-loaded; B is TDM-loaded (see file header comment).
    static constexpr bool Async = true;

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return Policy::template GetVectorSizeA<Problem, IsWave32Host>();
    }
    // B goes through TDM: the hardware descriptor handles vectorization internally,
    // same rationale as GemmPipelineAgBgCrCompTDMV1::GetVectorSizeB.
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return 1;
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

    using BlockWarps               = typename BlockGemmShape::BlockWarps;
    using WarpTile                 = typename BlockGemmShape::WarpTile;
    static constexpr index_t MWarp = BlockWarps::at(I0{});
    static constexpr index_t NWarp = BlockWarps::at(I1{});

    // Not used for K-subtiling by this pipeline (sub_tile_num == 1 unconditionally, see
    // the policy's GetBlockGemm), kept only because ScaleKDimPerBlock below is
    // referenced unconditionally by the (always-null in this hybrid) MX-scale plumbing
    // inherited structurally from GemmPipelineAgBgCrCompAsync.
    static constexpr index_t MPerXdl      = WarpTile::at(I0{});
    static constexpr index_t NPerXdl      = WarpTile::at(I1{});
    static constexpr index_t KPerXdl      = WarpTile::at(I2{});
    static constexpr index_t KIterPerWarp = KPerBlock / KPerXdl;
    static constexpr index_t KXdlPackEff  = 1;

    static constexpr index_t ScaleBlockSize    = 32;
    static constexpr index_t ScaleKDimPerBlock = KPerBlock / ScaleBlockSize / KXdlPackEff;

    // No TDM cluster-launch / multicast support in this first hybrid (only B is TDM,
    // so multicast would only ever help B).
    static constexpr bool UseClusterLaunch = false;

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName()
    {
        return "COMPUTE_ASYNC_TDM_HYBRID";
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

        CK_TILE_DEVICE static constexpr auto HotLoopScheduler()
        {
            constexpr index_t MPerXDL = BlockGemmShape::WarpTile::at(I0{});
            constexpr index_t NPerXDL = BlockGemmShape::WarpTile::at(I1{});
            constexpr index_t KPerXDL = BlockGemmShape::WarpTile::at(I2{});

            constexpr index_t WaveSize = get_warp_size();

            // Only A issues real buffer_load/VMEM instructions in this hybrid; B goes
            // through the TDM hardware mover (a different instruction class entirely,
            // not meaningfully accounted for by this VMEM_READ-based scheduling hint).
            constexpr index_t A_Buffer_Load_Inst_Num =
                MPerBlock * KPerBlock / (BlockSize * GetVectorSizeA());

            constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                                (BlockSize / WaveSize) /
                                                (MPerXDL * NPerXDL * KPerXDL);

            constexpr auto num_buffer_load_inst = A_Buffer_Load_Inst_Num;
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

        // Waits both hardware counters this hybrid pipeline uses: TDM's `tensorcnt`
        // (B's most recent LDS write) and async's `asynccnt` (A's most recent LDS
        // write, drained inside block_sync_lds_direct_load along with the workgroup
        // barrier). See this file's header comment for why both are needed here.
        CK_TILE_DEVICE static void WaitAAsyncBTdm()
        {
            s_wait_tensorcnt<0>();
            block_sync_lds_direct_load();
        }

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename AsDramBlockWindowTmp,
                  typename BsDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction,
                  typename ScaleADramBlockWindow,
                  typename ScaleBDramBlockWindow,
                  typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                                is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                            bool>* = nullptr>
        CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       ScaleADramBlockWindow& scale_a_dram_window,
                                       ScaleBDramBlockWindow& scale_b_dram_window,
                                       index_t num_loop,
                                       void* __restrict__ p_smem) const
        {
            using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;

            // TODO support multi-ABD
            static_assert(1 == std::tuple_size_v<AsDramBlockWindowTmp>);
            static_assert(1 == std::tuple_size_v<BsDramBlockWindowTmp>);
            using ADramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, AsDramBlockWindowTmp>>;
            using BDramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, BsDramBlockWindowTmp>>;
            // TODO currently fused elementwise are not supported
            static_assert(std::is_same_v<remove_cvref_t<decltype(a_element_func)>,
                                         element_wise::PassThrough>);
            static_assert(std::is_same_v<remove_cvref_t<decltype(b_element_func)>,
                                         element_wise::PassThrough>);
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

            // ---- TDM config for B (mirrors GemmPipelineAgBgCrCompTDMV1's own setup) ----
            TDMConfig tdm_config_b;
            constexpr auto LdsPaddingConfigB =
                Policy::template GetLdsPaddingConfig<Problem, false>();
            constexpr auto IsBPadding            = LdsPaddingConfigB[I0{}];
            constexpr auto BPaddingAmount        = LdsPaddingConfigB[I1{}];
            constexpr auto BPaddingInterval      = LdsPaddingConfigB[I2{}];
            tdm_config_b.pad_enable              = IsBPadding;
            tdm_config_b.pad_config.pad_amount   = BPaddingAmount;
            tdm_config_b.pad_config.pad_interval = BPaddingInterval;
            // workgroup_mask left at its default-constructed 0: UseClusterLaunch is
            // false, so no multicast is ever requested (matches
            // GemmPipelineAgBgCrCompTDMV1's own behavior when UseClusterLaunch==false).

            ////////////// global window & register /////////////////
            // A DRAM tile window(s) for load
            auto a_tile_windows = generate_tuple(
                [&](auto idx) {
                    return make_tile_window(
                        a_dram_block_window_tmp[number<idx>{}].get_bottom_tensor_view(),
                        make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                        a_dram_block_window_tmp[number<idx>{}].get_window_origin(),
                        Policy::template MakeADramTileDistribution<Problem>());
                },
                number<AsLayout::size()>{});

            // for XOR swizzle: policy makes async global-to-LDS stores match LDS reads
            // (bf16 never takes that branch on gfx125 -- see the policy header comment)
            auto a_async_tile_windows = generate_tuple(
                [&](auto idx) {
                    return make_tile_window(Policy::template MakeAsyncLoadADramWindow<Problem>(
                                                a_tile_windows[number<idx>{}]),
                                            Policy::template MakeADramTileDistribution<Problem>());
                },
                number<AsLayout::size()>{});

            // this pipeline has a pair of LDS buffers per logical tile
            constexpr index_t smem_size         = Policy::template GetSmemSize<Problem>();
            auto&& [a_lds_block0, b_lds_block0] = Base::GetABLdsTensorViews(p_smem);
            auto&& [a_lds_block1, b_lds_block1] =
                Base::GetABLdsTensorViews(static_cast<char*>(p_smem) + smem_size);

            // set up LDS tile shapes
            constexpr auto a_lds_shape = []() {
                if constexpr(is_a_load_tr_v)
                    return make_tuple(number<KPerBlock>{}, number<MPerBlock>{});
                else
                    return make_tuple(number<MPerBlock>{}, number<KPerBlock>{});
            }();

            // LDS tile windows for storing A, one per LDS buffer
            auto a_copy_lds_window0 = make_tile_window(a_lds_block0, a_lds_shape, {0, 0});
            auto a_copy_lds_window1 = make_tile_window(a_lds_block1, a_lds_shape, {0, 0});

            // ---- B: build (dram_window, {(copy_lds_window, gemm_window)} x 2) the same
            // way GemmPipelineAgBgCrCompTDMV1 itself does, via the shared generic
            // Base::GetBWindows helper (Policy::MakeBDramTileDistribution /
            // MakeBLdsBlockDescriptor route to TDM's coarse formulas for this policy). ----
            constexpr auto b_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());
            auto&& [b_copy_dram_window, b_lds_windows] =
                Base::GetBWindows(b_dram_block_window_tmp[number<0>{}],
                                  make_tuple(b_lds_block0, b_lds_block1),
                                  b_lds_load_tile_distr);
            auto& b_copy_lds_window0 = b_lds_windows[I0{}].template at<0>();
            auto& b_lds_ld_window0   = b_lds_windows[I0{}].template at<1>();
            auto& b_copy_lds_window1 = b_lds_windows[I1{}].template at<0>();
            auto& b_lds_ld_window1   = b_lds_windows[I1{}].template at<1>();

            // initialize DRAM window steps, used to advance the DRAM windows
            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

            // MX-scale plumbing: always null_tensor in this hybrid (no MX scale
            // support -- see file header comment). This exact null-tensor code shape is
            // already exercised, unmodified, by the existing non-scale bf16
            // test_ck_tile_gemm_pipeline_comp_async_wmma test of the donor Async
            // pipeline, so it is proven inert rather than a newly-trusted path.
            using ScaleATileType = decltype(load_tile(scale_a_dram_window));
            using ScaleBTileType = decltype(load_tile(scale_b_dram_window));
            ScaleATileType scale_a_tile_ping, scale_a_tile_pong;
            ScaleBTileType scale_b_tile_ping, scale_b_tile_pong;

            constexpr auto scale_a_dram_tile_window_step = make_array(0, ScaleKDimPerBlock);
            constexpr auto scale_b_dram_tile_window_step = make_array(0, ScaleKDimPerBlock);

            auto load_scales_from_dram = [&](auto& scale_a, auto& scale_b) {
                scale_a = load_tile(scale_a_dram_window);
                scale_b = load_tile(scale_b_dram_window);
                move_tile_window(scale_a_dram_window, scale_a_dram_tile_window_step);
                move_tile_window(scale_b_dram_window, scale_b_dram_tile_window_step);
            };

            // read A(0) from DRAM to LDS window(0) via async copy; B(0) via TDM.
            Base::GlobalPrefetchAsync(
                a_copy_lds_window0, a_async_tile_windows[number<0>{}], a_dram_tile_window_step);
            Base::GlobalPrefetchTDM(
                tdm_config_b, b_copy_lds_window0, b_copy_dram_window, b_dram_tile_window_step);

            // initialize block gemm
            auto block_gemm = BlockGemm();

            // initialize C block tile
            auto c_block_tile = block_gemm.MakeCBlockTile();
            clear_tile(c_block_tile);

            // read A(1)/B(1) into window(1)
            Base::GlobalPrefetchAsync(
                a_copy_lds_window1, a_async_tile_windows[number<0>{}], a_dram_tile_window_step);
            Base::GlobalPrefetchTDM(
                tdm_config_b, b_copy_lds_window1, b_copy_dram_window, b_dram_tile_window_step);

            // tile distribution for the register tiles
            constexpr auto ALdsTileDistr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            // (B's register-tile distribution is b_lds_load_tile_distr, computed above
            // for Base::GetBWindows -- reused directly, no separate alias needed.)

            using ALdsTile = decltype(make_static_distributed_tensor<ADataType>(ALdsTileDistr));
            using BLdsTile =
                decltype(make_static_distributed_tensor<BDataType>(b_lds_load_tile_distr));

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

            // LDS tile windows for reading A;
            // they share the data pointer with the LDS windows for storing
            // but also associate with a distribution to produce a register tile when reading.
            // (B's read windows, b_lds_ld_window0/1, are already produced above by
            // Base::GetBWindows with the correct -- possibly transposed -- distribution.)
            auto a_lds_ld_window0 =
                make_tile_window(a_lds_block0, a_lds_shape, {0, 0}, a_lds_input_tile_distr);
            auto a_lds_ld_window1 =
                make_tile_window(a_lds_block1, a_lds_shape, {0, 0}, a_lds_input_tile_distr);

            static_assert(!(is_tile_window_linear_v<decltype(a_lds_ld_window0)>) &&
                              !(is_tile_window_linear_v<decltype(a_lds_ld_window1)>) &&
                              !(is_tile_window_linear_v<decltype(b_lds_ld_window0)>) &&
                              !(is_tile_window_linear_v<decltype(b_lds_ld_window1)>),
                          "LDS windows must not be linear");

            // write to LDS window(0) must complete before the local prefetch
            WaitAAsyncBTdm();
            // read A(0), B(0) from LDS window(0) to pipeline registers(0)
            Base::LocalPrefetch(a_block_tile0, a_lds_ld_window0, is_a_load_tr_v);
            Base::LocalPrefetch(b_block_tile0, b_lds_ld_window0, is_b_load_tr_v);
            // LDS window(0) contents are overwritten below by global prefetch, need to sync
            block_sync_lds();
            // read A(2), B(2) from DRAM to LDS window(0)
            // and advance the DRAM windows
            if constexpr((!HasHotLoop && (TailNum == TailNumber::Three)) || HasHotLoop)
            {
                Base::GlobalPrefetchAsync(
                    a_copy_lds_window0, a_async_tile_windows[number<0>{}], a_dram_tile_window_step);
                Base::GlobalPrefetchTDM(
                    tdm_config_b, b_copy_lds_window0, b_copy_dram_window, b_dram_tile_window_step);
            }

            // Load scales for iteration 0 (ping)
            load_scales_from_dram(scale_a_tile_ping, scale_b_tile_ping);

            // Load scales for iteration 1 (pong) if needed
            if(num_loop > 1)
            {
                load_scales_from_dram(scale_a_tile_pong, scale_b_tile_pong);
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
                        // B's TDM write into window(1) (issued last round / during
                        // priming) is on its own counter -- drain it before reading.
                        s_wait_tensorcnt<0>();
                        Base::LocalPrefetch(b_block_tile1, b_lds_ld_window1, is_b_load_tr_v);
                        // LDS window(1) contents are overwritten by global prefetch, need to sync
                        block_sync_lds();
                        // read A(i), B(i) from DRAM to LDS window(1)
                        // and advance the DRAM windows
                        Base::GlobalPrefetchAsync(a_copy_lds_window1,
                                                  a_async_tile_windows[number<0>{}],
                                                  a_dram_tile_window_step);
                        Base::GlobalPrefetchTDM(tdm_config_b,
                                                b_copy_lds_window1,
                                                b_copy_dram_window,
                                                b_dram_tile_window_step);
                        // C(i-3) = A(i-3) @ B(i-3)
                        block_gemm(c_block_tile,
                                   a_block_tile0,
                                   b_block_tile0,
                                   scale_a_tile_ping,
                                   scale_b_tile_ping);
                        HotLoopScheduler();
                        // Load next scales after using current scales above
                        load_scales_from_dram(scale_a_tile_ping, scale_b_tile_ping);
                    }
                    // pong
                    {
                        // write to LDS window(0) must complete before the local prefetch
                        WaitAAsyncBTdm();
                        // read A(i), B(i) from LDS window(0) to pipeline registers(0)
                        Base::LocalPrefetch(a_block_tile0, a_lds_ld_window0, is_a_load_tr_v);
                        Base::LocalPrefetch(b_block_tile0, b_lds_ld_window0, is_b_load_tr_v);
                        // LDS window(0) contents are overwritten by global prefetch, need to sync
                        block_sync_lds();
                        // read A(i+1), B(i+1) from DRAM to LDS window(0)
                        // and advance the DRAM windows
                        Base::GlobalPrefetchAsync(a_copy_lds_window0,
                                                  a_async_tile_windows[number<0>{}],
                                                  a_dram_tile_window_step);
                        Base::GlobalPrefetchTDM(tdm_config_b,
                                                b_copy_lds_window0,
                                                b_copy_dram_window,
                                                b_dram_tile_window_step);
                        // C(i-2) = A(i-2) @ B(i-2)
                        block_gemm(c_block_tile,
                                   a_block_tile1,
                                   b_block_tile1,
                                   scale_a_tile_pong,
                                   scale_b_tile_pong);
                        HotLoopScheduler();
                        // Load next scales after using current scales above
                        load_scales_from_dram(scale_a_tile_pong, scale_b_tile_pong);
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
                    s_wait_tensorcnt<0>();
                    Base::LocalPrefetch(b_block_tile1, b_lds_ld_window1, is_b_load_tr_v);
                    // C(num_loop-2) = A(num_loop-2) @ B(num_loop-2)
                    block_gemm(c_block_tile,
                               a_block_tile0,
                               b_block_tile0,
                               scale_a_tile_ping,
                               scale_b_tile_ping);
                    // load last scales to ping for the last iteration to ping buffers
                    load_scales_from_dram(scale_a_tile_ping, scale_b_tile_ping);
                }
                {
                    // write to LDS window(0) must complete before the local prefetch
                    WaitAAsyncBTdm();
                    // read A(num_loop), B(num_loop) from LDS window(0) to pipeline registers(0)
                    Base::LocalPrefetch(a_block_tile0, a_lds_ld_window0, is_a_load_tr_v);
                    Base::LocalPrefetch(b_block_tile0, b_lds_ld_window0, is_b_load_tr_v);
                    // C(num_loop-1) = A(num_loop-1) @ B(num_loop-1)
                    block_gemm(c_block_tile,
                               a_block_tile1,
                               b_block_tile1,
                               scale_a_tile_pong,
                               scale_b_tile_pong);
                }
                {
                    // C(num_loop) = A(num_loop) @ B(num_loop)
                    block_gemm(c_block_tile,
                               a_block_tile0,
                               b_block_tile0,
                               scale_a_tile_ping,
                               scale_b_tile_ping);
                }
            }
            else if(TailNum == TailNumber::Two)
            // 2 block gemms remaining
            {
                {
                    // read A(num_loop), B(num_loop) from LDS window(1) to pipeline registers(1)
                    Base::LocalPrefetch(a_block_tile1, a_lds_ld_window1, is_a_load_tr_v);
                    s_wait_tensorcnt<0>();
                    Base::LocalPrefetch(b_block_tile1, b_lds_ld_window1, is_b_load_tr_v);
                    // C(num_loop-1) = A(num_loop-1) @ B(num_loop-1)
                    block_gemm(c_block_tile,
                               a_block_tile0,
                               b_block_tile0,
                               scale_a_tile_ping,
                               scale_b_tile_ping);
                }
                {
                    // C(num_loop) = A(num_loop) @ B(num_loop)
                    block_gemm(c_block_tile,
                               a_block_tile1,
                               b_block_tile1,
                               scale_a_tile_pong,
                               scale_b_tile_pong);
                }
            }
            else if(TailNum == TailNumber::One)
            {
                block_sync_lds();
                block_gemm(c_block_tile,
                           a_block_tile0,
                           b_block_tile0,
                           scale_a_tile_ping,
                           scale_b_tile_ping);
                __builtin_amdgcn_sched_barrier(0);
            }
            return c_block_tile;
        }
    };

    using NullTileWindowType =
        decltype(make_null_tile_window(make_tuple(number<0>{}, number<0>{})));

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
        auto scale_a_dram_window = NullTileWindowType{};
        auto scale_b_dram_window = NullTileWindowType{};

        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);
        const auto RunPipeline  = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp,
                a_element_func,
                b_dram_block_window_tmp,
                b_element_func,
                scale_a_dram_window,
                scale_b_dram_window,
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
        auto scale_a_dram_window = NullTileWindowType{};
        auto scale_b_dram_window = NullTileWindowType{};

        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);

        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp,
                element_wise::PassThrough{},
                b_dram_block_window_tmp,
                element_wise::PassThrough{},
                scale_a_dram_window,
                scale_b_dram_window,
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
        auto scale_a_dram_window = NullTileWindowType{};
        auto scale_b_dram_window = NullTileWindowType{};

        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);
        const auto RunPipeline  = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                ck_tile::make_tuple(a_dram_block_window_tmp),
                a_element_func,
                ck_tile::make_tuple(b_dram_block_window_tmp),
                b_element_func,
                scale_a_dram_window,
                scale_b_dram_window,
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
        auto scale_a_dram_window = NullTileWindowType{};
        auto scale_b_dram_window = NullTileWindowType{};

        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);

        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                ck_tile::make_tuple(a_dram_block_window_tmp),
                element_wise::PassThrough{},
                ck_tile::make_tuple(b_dram_block_window_tmp),
                element_wise::PassThrough{},
                scale_a_dram_window,
                scale_b_dram_window,
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
        return concat('_', "pipeline_AgBgCrCompAsyncTDMHybrid", 
                      concat('x', MPerBlock, NPerBlock, KPerBlock),  BlockSize,
                      concat('x', WaveNumM, WaveNumN),
                      concat('x', kPadM, kPadN, kPadK));
        // clang-format on
    }
};
} // namespace ck_tile
