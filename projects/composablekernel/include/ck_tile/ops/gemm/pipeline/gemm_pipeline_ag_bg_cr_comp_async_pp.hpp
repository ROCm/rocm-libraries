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
struct BaseGemmPipelineAgBgCrCompAsyncPP
{
    // One tile is in flight (global -> LDS) while the previous one is computed.
    static constexpr index_t PrefetchStages  = 1;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 1;
    static constexpr index_t UnrollHotLoop   = 2;

    // The hot loop consumes two tiles (one per LDS buffer) per trip and always leaves one or two
    // tiles for the tail, so it only runs when there are more than two tiles.
    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > UnrollHotLoop;
    }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        return num_loop % UnrollHotLoop == 1 ? TailNumber::One : TailNumber::Two;
    }

    template <typename RunFunction>
    CK_TILE_HOST_DEVICE static auto
    TailHandler(const RunFunction& run_func, bool has_hot_loop, TailNumber tail_number)
    {
        // Keep the dispatch values in SGPRs.
        const bool has_hot_loop_first_lane      = amd_wave_read_first_lane(has_hot_loop);
        const TailNumber tail_number_first_lane = amd_wave_read_first_lane(tail_number);
        if(has_hot_loop_first_lane)
        {
            if(tail_number_first_lane == TailNumber::One)
            {
                return run_func(bool_constant<true>{},
                                integral_constant<TailNumber, TailNumber::One>{});
            }
            else
            {
                return run_func(bool_constant<true>{},
                                integral_constant<TailNumber, TailNumber::Two>{});
            }
        }
        else
        {
            if(tail_number_first_lane == TailNumber::One)
            {
                return run_func(bool_constant<false>{},
                                integral_constant<TailNumber, TailNumber::One>{});
            }
            else
            {
                return run_func(bool_constant<false>{},
                                integral_constant<TailNumber, TailNumber::Two>{});
            }
        }
    }

    CK_TILE_HOST static constexpr auto GetName() { return "COMPUTE_ASYNC_PP"; }
};

/**
 * @brief Compute pipeline with async global-to-LDS loads in a single-barrier ping-pong order.
 *
 * Two LDS buffers; tile i lives in buffer i % 2. Per K tile:
 *
 *   block_sync_lds_async_ds<0>()      // tile i landed in LDS; all reads of the other buffer
 *                                     // (tile i - 1) retired on every wave
 *   async prefetch tile i + 1 -> other buffer   (skipped for the last tile)
 *   LDS -> registers for tile i; block gemm
 *
 * and a final block_sync_lds() before the epilogue reuses LDS. Unlike CompAsync and
 * CompAsyncV2 this order never waits on a partial ASYNCcnt (so it does not rely on in-order
 * completion), has one barrier per tile, and never prefetches past num_loop, so it needs no
 * K padding to stay in bounds. When kPadM/kPadN/kPadK are all false the async loads are issued
 * without the per-lane range check (gfx125: no v_cmp, branch or zero-fill DS store); the
 * caller must then pass exact tile multiples, which IsSupportedArgument enforces. Each tile is
 * read from LDS and multiplied whole (the default policy has sub-tiling disabled).
 *
 * This is the ordering of the rocKE direct-to-LDS GEMM (rocm-libraries #12658/#12810).
 */
template <typename Problem, typename Policy = GemmPipelineAgBgCrCompAsyncDefaultPolicy<>>
struct GemmPipelineAgBgCrCompAsyncPP : public BaseGemmPipelineAgBgCrCompAsyncPP<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrCompAsyncPP<Problem>;
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

    static constexpr bool UsePersistentKernel = Problem::Traits::UsePersistentKernel;

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

    // Without any padding every async load is in bounds (given IsSupportedArgument), so the
    // per-lane range check and zero-fill are dropped.
    static constexpr bool kAsyncOobCheck = kPadM || kPadN || kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;

    static_assert(DoubleSmemBuffer == true, "pipeline requires double smem buffer");

    static constexpr auto Scheduler = Problem::Scheduler;

    static constexpr auto is_a_load_tr_v = bool_constant<PipelineImplBase::is_a_load_tr>{};
    static constexpr auto is_b_load_tr_v = bool_constant<PipelineImplBase::is_b_load_tr>{};

    // Whole-pass checks: every lane of the block issues the same number of whole-vector async
    // copies per tile, and a copy never straddles the end of a contiguous row.
    static constexpr bool is_a_k_contiguous =
        std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>;
    static constexpr bool is_b_k_contiguous =
        std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>;

    static constexpr index_t AContiguousPerBlock = is_a_k_contiguous ? KPerBlock : MPerBlock;
    static constexpr index_t BContiguousPerBlock = is_b_k_contiguous ? KPerBlock : NPerBlock;

    static_assert((MPerBlock * KPerBlock) % (BlockSize * GetVectorSizeA()) == 0 &&
                      MPerBlock * KPerBlock >= BlockSize * GetVectorSizeA(),
                  "A tile must be a whole number of block-wide async copy passes!");
    static_assert((NPerBlock * KPerBlock) % (BlockSize * GetVectorSizeB()) == 0 &&
                      NPerBlock * KPerBlock >= BlockSize * GetVectorSizeB(),
                  "B tile must be a whole number of block-wide async copy passes!");
    static_assert(AContiguousPerBlock % GetVectorSizeA() == 0 &&
                      BContiguousPerBlock % GetVectorSizeB() == 0,
                  "Tile contiguous extent must be a multiple of the async copy vector!");
    static_assert(Policy::template IsSupportedAsyncVectorWidth<ADataType, GetVectorSizeA()> &&
                      Policy::template IsSupportedAsyncVectorWidth<BDataType, GetVectorSizeB()>,
                  "Async copy vector must be 4, 12 or 16 bytes!");
    static_assert(AContiguousPerBlock * sizeof(ADataType) / APackedSize % 16 == 0 &&
                      BContiguousPerBlock * sizeof(BDataType) / BPackedSize % 16 == 0,
                  "Tile rows must be a multiple of 16 bytes!");

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName()
    {
        // clang-format off
        return "COMPUTE_ASYNC_PP";
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

    /**
     * @brief Pipeline-specific argument checks, on top of the kernel's own.
     *
     * With no padding the async loads are unchecked (gfx125 has no buffer clamp), so M, N and
     * K / k_batch must be exact multiples of the block tile whatever the layouts. The kernel only
     * rejects a tail in a dimension one of its A/B/C layout checks covers, so it misses an M
     * tail with row-major A and C, an N tail with column-major B and C, and a K tail with
     * column-major A and row-major B. Checking all three here is redundant for the covered
     * cases and keeps the condition in one place. The contiguous-dimension async vector
     * alignment is already checked by the kernel for every A/B layout.
     */
    CK_TILE_HOST static bool IsSupportedArgument(index_t M, index_t N, index_t K, index_t k_batch)
    {
        if constexpr(!kAsyncOobCheck)
        {
            if(M % MPerBlock != 0 || N % NPerBlock != 0 || K % (KPerBlock * k_batch) != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("CompAsyncPP without padding needs M, N and K / k_batch to be "
                                  "multiples of the block tile!");
                }
                return false;
            }
        }
        else
        {
            ignore = M;
            ignore = N;
            ignore = K;
            ignore = k_batch;
        }
        return true;
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        template <typename DstBlockWindow, typename SrcTileWindow, typename DramTileWindowStep>
        CK_TILE_DEVICE void
        GlobalPrefetchAsync(DstBlockWindow& dst_block_window,
                            SrcTileWindow& dram_tile_window,
                            const DramTileWindowStep& dram_tile_window_step) const
        {
            async_load_tile(
                dst_block_window, dram_tile_window, number<-1>{}, bool_constant<kAsyncOobCheck>{});
            move_tile_window(dram_tile_window, dram_tile_window_step);
        }

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
            // B DRAM window(s) for load
            auto b_tile_windows = generate_tuple(
                [&](auto idx) {
                    return make_tile_window(
                        b_dram_block_window_tmp[number<idx>{}].get_bottom_tensor_view(),
                        make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                        b_dram_block_window_tmp[number<idx>{}].get_window_origin(),
                        Policy::template MakeBDramTileDistribution<Problem>());
                },
                number<BsLayout::size()>{});

            // for XOR swizzle: policy makes async global-to-LDS stores match LDS reads
            // otherwise: no change to view
            auto a_async_tile_window = make_tile_window(
                Policy::template MakeAsyncLoadADramWindow<Problem>(a_tile_windows[number<0>{}]),
                Policy::template MakeADramTileDistribution<Problem>());
            auto b_async_tile_window = make_tile_window(
                Policy::template MakeAsyncLoadBDramWindow<Problem>(b_tile_windows[number<0>{}]),
                Policy::template MakeBDramTileDistribution<Problem>());

            // a pair of LDS buffers; tile i goes to buffer i % 2
            constexpr index_t smem_size         = Policy::template GetSmemSize<Problem>();
            auto&& [a_lds_block0, b_lds_block0] = Base::GetABLdsTensorViews(p_smem);
            auto&& [a_lds_block1, b_lds_block1] =
                Base::GetABLdsTensorViews(static_cast<char*>(p_smem) + smem_size);

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

            // LDS tile windows for the async stores, one per LDS buffer
            auto a_copy_lds_windows =
                make_tuple(make_tile_window(a_lds_block0, a_lds_shape, {0, 0}),
                           make_tile_window(a_lds_block1, a_lds_shape, {0, 0}));
            auto b_copy_lds_windows =
                make_tuple(make_tile_window(b_lds_block0, b_lds_shape, {0, 0}),
                           make_tile_window(b_lds_block1, b_lds_shape, {0, 0}));

            // initialize DRAM window steps, used to advance the DRAM windows
            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

            // tile distribution for the register tiles
            constexpr auto ALdsTileDistr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto BLdsTileDistr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

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

            // LDS tile windows for reading, sharing the data pointers of the store windows
            auto a_lds_ld_windows = make_tuple(
                make_tile_window(a_lds_block0, a_lds_shape, {0, 0}, a_lds_input_tile_distr),
                make_tile_window(a_lds_block1, a_lds_shape, {0, 0}, a_lds_input_tile_distr));
            auto b_lds_ld_windows = make_tuple(
                make_tile_window(b_lds_block0, b_lds_shape, {0, 0}, b_lds_input_tile_distr),
                make_tile_window(b_lds_block1, b_lds_shape, {0, 0}, b_lds_input_tile_distr));

            using ALdsTile = decltype(make_static_distributed_tensor<ADataType>(ALdsTileDistr));
            using BLdsTile = decltype(make_static_distributed_tensor<BDataType>(BLdsTileDistr));

            ALdsTile a_block_tile;
            BLdsTile b_block_tile;

            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            // One K tile computed out of LDS buffer `buf`; when `prefetch_next`, the next tile is
            // first issued into the other buffer.
            auto run_tile = [&](auto buf, auto prefetch_next) {
                constexpr index_t cur = decltype(buf)::value;
                constexpr index_t nxt = 1 - cur;
                // This tile's async copies (and any zero-fill DS stores) have landed, and every
                // wave has retired its reads of buffer `nxt` from the previous tile.
                block_sync_lds_async_ds<0>();
                if constexpr(decltype(prefetch_next)::value)
                {
                    GlobalPrefetchAsync(a_copy_lds_windows[number<nxt>{}],
                                        a_async_tile_window,
                                        a_dram_tile_window_step);
                    GlobalPrefetchAsync(b_copy_lds_windows[number<nxt>{}],
                                        b_async_tile_window,
                                        b_dram_tile_window_step);
                }
                Base::LocalPrefetch(a_block_tile, a_lds_ld_windows[number<cur>{}], is_a_load_tr_v);
                Base::LocalPrefetch(b_block_tile, b_lds_ld_windows[number<cur>{}], is_b_load_tr_v);
                block_gemm(c_block_tile, a_block_tile, b_block_tile);
            };

            // prologue: tile 0 -> buffer 0
            GlobalPrefetchAsync(
                a_copy_lds_windows[I0{}], a_async_tile_window, a_dram_tile_window_step);
            GlobalPrefetchAsync(
                b_copy_lds_windows[I0{}], b_async_tile_window, b_dram_tile_window_step);

            clear_tile(c_block_tile);

            if constexpr(HasHotLoop)
            {
                // Each trip computes tiles i (buffer 0) and i + 1 (buffer 1) and prefetches
                // i + 1 and i + 2; the one or two tiles left go to the tail.
                constexpr index_t tail_tiles = TailNum == TailNumber::One ? 1 : 2;
                index_t i_tile               = 0;
                do
                {
                    run_tile(I0{}, true_type{});
                    run_tile(I1{}, true_type{});
                    i_tile += 2;
                } while(i_tile < num_loop - tail_tiles);
            }

            if constexpr(TailNum == TailNumber::Two)
            {
                run_tile(I0{}, true_type{});
                run_tile(I1{}, false_type{});
            }
            else
            {
                run_tile(I0{}, false_type{});
            }

            // The epilogue may reuse LDS; every wave must be done reading the last tile.
            block_sync_lds();

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
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);

        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
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
        return operator()(a_dram_block_window_tmp,
                          element_wise::PassThrough{},
                          b_dram_block_window_tmp,
                          element_wise::PassThrough{},
                          num_loop,
                          p_smem);
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});
        return concat('_', "pipeline_AgBgCrCompAsyncPP",
                      concat('x', MPerBlock, NPerBlock, KPerBlock),  BlockSize,
                      concat('x', WaveNumM, WaveNumN),
                      concat('x', kPadM, kPadN, kPadK));
        // clang-format on
    }
};
} // namespace ck_tile
