// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_producer_consumer_policy.hpp"

namespace ck_tile {

/**
 * @brief TDM GEMM pipeline whose loader waves run ahead of its compute waves, synchronised over
 * hardware named barriers (gfx1250).
 *
 * The workgroup is split by role. Waves [0, NumConsumerWaves) are consumers: BlockGemm maps the
 * C tile onto them exactly as in any other pipeline, and they only read LDS and multiply. The
 * next two waves are producers, one per operand, and only issue that operand's TDM transfers.
 * Between them sits a ring of NumSlots LDS buffers:
 *
 *     producer A --TDM--> +--------+--------+--------+
 *                         | slot 0 | slot 1 | slot 2 | --ds_read--> consumers --> C tile
 *     producer B --TDM--> +--------+--------+--------+
 *
 * A producer waits only until the slot it is about to overwrite has been drained, and the
 * consumers only until the slot they are about to read has been filled, so the loaders run up
 * to NumSlots K steps ahead instead of meeting the math at a workgroup barrier every step, as
 * the symmetric TDM pipelines do. The handshake itself lives in named_barrier.hpp; this pipeline
 * supplies only what a step does on each side.
 *
 * The kernel owns the barriers: it declares `using barrier_pipeline = BarrierPipeline;`, passes
 * the token of BarrierPipeline::init<Kernel>() to operator(), launches LaunchBlockSize threads
 * and runs the epilogue on the consumer waves only. UniversalGemmKernel does all of this for any
 * pipeline exposing these members. The producers return an empty accumulator.
 *
 * Other targets compile an inert fallback that returns zeros, so that multi-target builds still
 * compile; UniversalGemmKernel::IsSupportedArgument rejects those devices.
 */
template <typename Problem, typename Policy = GemmPipelineAgBgCrCompTDMProducerConsumerPolicy<>>
struct GemmPipelineAgBgCrCompTDMProducerConsumer
{
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

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t NumSlots   = Policy::NumSlots;
    static constexpr index_t PublishLag = Policy::PublishLag;

    /// BlockGemm maps the C tile onto these waves; the producers are launched in addition.
    static constexpr index_t NumConsumerWaves = BlockGemmShape::NumWarps;
    /// One producer per operand: wave NumConsumerWaves loads A, the wave after it loads B.
    static constexpr index_t NumProducerWaves = 2;

    /// The named barriers the kernel must arm for this pipeline.
    using BarrierPipeline =
        named_barrier_pipeline<ring_spec<NumSlots, NumProducerWaves, NumConsumerWaves>>;

    // Wavelet interface: BlockGemm and the epilogue see BlockSize; the kernel launches more.
    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t LaunchBlockSize =
        (NumConsumerWaves + NumProducerWaves) * get_warp_size();
    static constexpr bool IsWavelet = true;

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr index_t NumWaveGroups    = Problem::NumWaveGroups;
    static constexpr index_t Preshuffle       = Problem::Preshuffle;
    static constexpr bool UsePersistentKernel = Problem::Traits::UsePersistentKernel;

    // TDM vectorises internally and handles ragged extents, so launch checks do not apply.
    static constexpr bool skipCheckValidLaunchParams = true;

    static constexpr index_t KSubTileNum = Policy::template GetPipelineSubTileNum<Problem>().value;

    static constexpr auto is_a_load_tr_v = bool_constant<PipelineImplBase::is_a_load_tr>{};
    static constexpr auto is_b_load_tr_v = bool_constant<PipelineImplBase::is_b_load_tr>{};

    static_assert(NumWaveGroups == 1, "the producer/consumer split replaces wave groups");
    static_assert(!Preshuffle, "preshuffled B is not supported");
    static_assert(!Policy::template isClusterLaunch<Problem>(),
                  "cluster multicast is not supported: a peer's transfer into this workgroup's "
                  "LDS is invisible to the tensor counter a producer publishes on");
    static_assert(Policy::DataCachePrefetchA == DataCachePrefetchKind::None &&
                      Policy::DataCachePrefetchB == DataCachePrefetchKind::None,
                  "data cache prefetch is not supported");
    static_assert(!std::is_same_v<BDataType, pk_int4_t>, "Not implemented");

    CK_TILE_DEVICE static bool IsMathWave() { return get_warp_id() < NumConsumerWaves; }

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return 1;
    }
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return 1;
    }
    static constexpr index_t GetVectorSizeC() { return 1; }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return NumSlots * Policy::template GetSmemSize<Problem>();
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AgBgCrCompTDMProducerConsumer",
                      concat('x', MPerBlock, NPerBlock, KPerBlock), LaunchBlockSize,
                      concat('x', BlockGemmShape::BlockWarps::at(number<0>{}),
                                  BlockGemmShape::BlockWarps::at(number<1>{})),
                      concat('x', kPadM, kPadN, kPadK),
                      concat('x', NumSlots, PublishLag));
        // clang-format on
    }

    template <typename AsDramBlockWindowTmp,
              typename AElementFunction,
              typename BsDramBlockWindowTmp,
              typename BElementFunction>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* __restrict__ p_smem,
                                   typename BarrierPipeline::token barriers) const
    {
        static_assert(1 == std::tuple_size_v<AsDramBlockWindowTmp> &&
                          1 == std::tuple_size_v<BsDramBlockWindowTmp>,
                      "one A and one B tensor");
        using ADramBlockWindowTmp = remove_cvref_t<std::tuple_element_t<0, AsDramBlockWindowTmp>>;
        using BDramBlockWindowTmp = remove_cvref_t<std::tuple_element_t<0, BsDramBlockWindowTmp>>;

        static_assert(std::is_same_v<AElementFunction, element_wise::PassThrough> &&
                          std::is_same_v<BElementFunction, element_wise::PassThrough>,
                      "TDM moves global memory to LDS unmodified; no element-wise function");
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
            "Data Type conflict on A and B matrix input data type.");

        constexpr bool is_a_col_major = std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
        constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;
        static_assert(is_a_col_major
                          ? (KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                             MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}])
                          : (MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                             KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}]),
                      "A block window has incorrect lengths for defined ALayout!");
        static_assert(is_b_row_major
                          ? (KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                             NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[number<1>{}])
                          : (NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                             KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[number<1>{}]),
                      "B block window has incorrect lengths for defined BLayout!");

        ignore = a_element_func;
        ignore = b_element_func;

        using CBlockTile = decltype(BlockGemm::MakeCBlockTile());

        if constexpr(BarrierPipeline::kIsSupported)
        {
            const auto lds_views =
                PipelineImplBase{}.template GetABLdsTensorViews<NumSlots>(p_smem);

            // Wave-uniform: every ring call below executes once per wave, whatever EXEC holds.
            const index_t warp_id = get_warp_id();
            if(warp_id < NumConsumerWaves)
            {
                return RunConsumers(
                    lds_views.at(number<0>{}), lds_views.at(number<1>{}), num_loop, barriers);
            }
            if(warp_id == NumConsumerWaves)
            {
                RunProducer<0>(a_dram_block_window_tmp[number<0>{}],
                               lds_views.at(number<0>{}),
                               num_loop,
                               barriers);
            }
            else
            {
                RunProducer<1>(b_dram_block_window_tmp[number<0>{}],
                               lds_views.at(number<1>{}),
                               num_loop,
                               barriers);
            }
            return CBlockTile{};
        }
        else
        {
            ignore = a_dram_block_window_tmp;
            ignore = b_dram_block_window_tmp;
            ignore = num_loop;
            ignore = p_smem;
            ignore = barriers;
            return CBlockTile{};
        }
    }

    private:
    template <bool IsA>
    CK_TILE_DEVICE static TDMConfig MakeTdmConfig()
    {
        constexpr auto padding = Policy::template GetLdsPaddingConfig<Problem, IsA>();

        TDMConfig config;
        config.pad_enable              = padding[number<0>{}];
        config.pad_config.pad_amount   = padding[number<1>{}];
        config.pad_config.pad_interval = padding[number<2>{}];
        // workgroup_mask stays 0: a non-zero mask makes the transfer wait for cluster peers.
        return config;
    }

    // One producer wave: streams operand Operand (0 = A, 1 = B) into the ring, one TDM per step.
    template <index_t Operand, typename DramBlockWindowTmp, typename LdsViews>
    CK_TILE_DEVICE static void RunProducer(const DramBlockWindowTmp& dram_block_window_tmp,
                                           const LdsViews& lds_views,
                                           index_t num_loop,
                                           typename BarrierPipeline::token barriers)
    {
        constexpr bool IsA = (Operand == 0);
        const PipelineImplBase impl{};

        auto dram_window = [&]() {
            if constexpr(IsA)
                return impl.CopyADramWindow(dram_block_window_tmp);
            else
                return impl.CopyBDramWindow(dram_block_window_tmp);
        }();

        // Only the copy windows are used: TDM needs no per-thread distribution in LDS.
        auto lds_windows = generate_tuple(
            [&](auto slot) {
                if constexpr(IsA)
                    return impl.MakeALdsWindows(lds_views[slot], ALdsLoadTileDistr{})
                        .at(number<0>{});
                else
                    return impl.MakeBLdsWindows(lds_views[slot], BLdsLoadTileDistr{})
                        .at(number<0>{});
            },
            number<NumSlots>{});

        constexpr bool k_is_dim0  = IsA ? std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>
                                        : std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;
        using DramStep            = typename decltype(dram_window)::BottomTensorIndex;
        constexpr DramStep k_step = k_is_dim0 ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

        const TDMConfig tdm_config = MakeTdmConfig<IsA>();

        using Ring = typename BarrierPipeline::template ring<0>;
        Ring::template producer<Operand>::template run<PublishLag>(
            barriers,
            num_loop,
            [&](auto slot, index_t) {
                impl.GlobalPrefetchTDM(tdm_config, lds_windows[slot], dram_window, k_step);
            },
            // Each fill is one TENSOR_LOAD_TO_LDS, retiring in issue order.
            [](auto in_flight) { s_wait_tensorcnt<decltype(in_flight)::value>(); });
    }

    // The consumer waves: multiply each slot as it fills, and hand it back once read.
    template <typename ALdsViews, typename BLdsViews>
    CK_TILE_DEVICE static auto RunConsumers(const ALdsViews& a_lds_views,
                                            const BLdsViews& b_lds_views,
                                            index_t num_loop,
                                            typename BarrierPipeline::token barriers)
    {
        const PipelineImplBase impl{};

        // One window per slot; every step leaves its slot's window where it found it.
        auto a_windows = generate_tuple(
            [&](auto slot) {
                return impl.MakeALdsWindows(a_lds_views[slot], ALdsLoadTileDistr{}).at(number<1>{});
            },
            number<NumSlots>{});
        auto b_windows = generate_tuple(
            [&](auto slot) {
                return impl.MakeBLdsWindows(b_lds_views[slot], BLdsLoadTileDistr{}).at(number<1>{});
            },
            number<NumSlots>{});

        // K sub-tiles double-buffer through registers: one is read while the other multiplies.
        using ALdsTile = decltype(make_static_distributed_tensor<ADataType>(ALdsLoadTileDistr{}));
        using BLdsTile = decltype(make_static_distributed_tensor<BDataType>(BLdsLoadTileDistr{}));
        ALdsTile a_tiles[2];
        BLdsTile b_tiles[2];

        auto block_gemm   = BlockGemm();
        auto c_block_tile = block_gemm.MakeCBlockTile();
        clear_tile(c_block_tile);

        using Ring = typename BarrierPipeline::template ring<0>;
        Ring::consumer::run(barriers, num_loop, [&](auto slot, index_t) {
            auto& a_window = a_windows[slot];
            auto& b_window = b_windows[slot];

            block_gemm.template LocalPrefetch<KSubTileNum == 1 ? WindowSlideMode::Stay
                                                               : WindowSlideMode::Move>(
                a_tiles[0], b_tiles[0], a_window, b_window, is_a_load_tr_v, is_b_load_tr_v);
            static_for<0, KSubTileNum - 1, 1>{}([&](auto i) {
                constexpr index_t kNext = (i.value + 1) % 2;
                constexpr auto kSlideTo = (i.value + 1 == KSubTileNum - 1) ? WindowSlideMode::Reset
                                                                           : WindowSlideMode::Move;
                block_gemm.template LocalPrefetch<kSlideTo>(a_tiles[kNext],
                                                            b_tiles[kNext],
                                                            a_window,
                                                            b_window,
                                                            is_a_load_tr_v,
                                                            is_b_load_tr_v);
                block_gemm(c_block_tile, a_tiles[i.value % 2], b_tiles[i.value % 2]);
            });
            block_gemm(
                c_block_tile, a_tiles[(KSubTileNum - 1) % 2], b_tiles[(KSubTileNum - 1) % 2]);

            // The ring releases the slot the moment this returns, so retire its reads first.
            s_wait_dscnt<0>();
        });
        return c_block_tile;
    }

    using ALdsLoadTileDistr =
        decltype(make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode()));
    using BLdsLoadTileDistr =
        decltype(make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode()));
};

} // namespace ck_tile
