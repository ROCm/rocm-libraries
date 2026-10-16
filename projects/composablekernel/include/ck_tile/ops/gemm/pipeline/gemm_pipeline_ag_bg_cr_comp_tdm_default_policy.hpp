// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/utility/data_cache_prefetch.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {

namespace detail {
// Resolves a problem/trait type's microscaling block size: yields T::ScaleBlockSize
// when the member exists, otherwise falls back to the legacy default of 32 (so
// problems that predate the ScaleBlockSize member keep compiling unchanged).
template <typename T, typename = void>
struct scale_block_size_or_default
{
    static constexpr index_t value = 32;
};
template <typename T>
struct scale_block_size_or_default<T, std::void_t<decltype(T::ScaleBlockSize)>>
{
    static constexpr index_t value = T::ScaleBlockSize;
};
} // namespace detail

enum class MultiCastDirection
{
    kM,
    kN,
    kMN
};

// Default policy for GemmPipelineAgBgCrCompTDM
template <bool WaveSpecialized                               = false,
          ck_tile::DataCachePrefetchKind DataCachePrefetchA_ = ck_tile::DataCachePrefetchKind::None,
          ck_tile::DataCachePrefetchKind DataCachePrefetchB_ = ck_tile::DataCachePrefetchKind::None>
struct GemmPipelineAgBgCrCompTDMDefaultPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompTDMDefaultPolicy<WaveSpecialized,
                                                                            DataCachePrefetchA_,
                                                                            DataCachePrefetchB_>>
{
    using Base =
        UniversalGemmBasePolicy<GemmPipelineAgBgCrCompTDMDefaultPolicy<WaveSpecialized,
                                                                       DataCachePrefetchA_,
                                                                       DataCachePrefetchB_>>;

    static constexpr ck_tile::DataCachePrefetchKind DataCachePrefetchA = DataCachePrefetchA_;
    static constexpr ck_tile::DataCachePrefetchKind DataCachePrefetchB = DataCachePrefetchB_;

    template <typename Problem>
    using LdsADataType = typename Problem::ADataType;

    template <typename Problem>
    using LdsBDataType = typename Problem::BDataType;

    static constexpr index_t VecByteSize = 16;

    // Get the number of data elements in one unpadded, folded LDS row.
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsRowElements()
    {
        using DataType = std::conditional_t<IsA, LdsADataType<Problem>, LdsBDataType<Problem>>;
        constexpr index_t PackedSize = numeric_traits<DataType>::PackedSize;
        constexpr index_t KPerBlock  = Problem::BlockGemmShape::kK;

        // Bytes in one pass over all LDS banks (one dword per bank)
        constexpr index_t LdsBanksWidth = get_n_lds_banks() * get_n_dwords_per_128b();

        // Num. logical M/N rows whose K payload fits into LdsBanksWidth
        constexpr index_t LdsLayerRequired =
            LdsBanksWidth / KPerBlock / index_t{sizeof(DataType)} * PackedSize;

        // Fold at least one logical M/N row
        constexpr index_t LdsLayer = max(1, LdsLayerRequired);

        // Return unpadded payload length of the folded LDS row
        return KPerBlock * LdsLayer;
    }

    // TDM can only encode a power-of-two padding interval.
    // When padding falls inside an LDS row, pick an interval which divides both
    // the row and one pipeline sub-tile.
    // In gfx125, tile_window::load_with_offset() computes:
    // offset(origin+d) = offset(origin) + offset(d)
    // which only holds when every window origin (each K sub-tile) starts an interval.
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfig()
    {
        constexpr auto config = Base::template GetLdsPaddingConfig<Problem, IsA>();

        constexpr bool is_tr_load =
            IsA ? Base::template is_a_load_tr<Problem> : Base::template is_b_load_tr<Problem>;
        if constexpr(is_tr_load || !static_cast<bool>(config[Base::I0]))
        {
            // Transpose-load or disabled padding: stop here
            return config;
        }
        else
        {
            using DataType = std::conditional_t<IsA, LdsADataType<Problem>, LdsBDataType<Problem>>;
            constexpr index_t PackedSize = numeric_traits<DataType>::PackedSize;

            // Unpadded LDS row length in dwords
            constexpr index_t row_dwords = GetLdsRowElements<Problem, IsA>() *
                                           index_t{sizeof(DataType)} / PackedSize / sizeof(int32_t);

            // K length processed by 1 pipeline sub-tile, in dwords
            constexpr index_t sub_tile_dwords =
                Problem::BlockGemmShape::kK / GetPipelineSubTileNum<Problem>().value *
                index_t{sizeof(DataType)} / PackedSize / sizeof(int32_t);

            // x & -x selects the largest power-of-two divisor; the TDM field caps it at 256.
            constexpr index_t row_interval_dwords = min(row_dwords & -row_dwords, index_t{256});
            // Pads inside the row: each K sub-tile must also start on an interval.
            constexpr index_t interval_dwords =
                row_interval_dwords < row_dwords
                    ? min(row_interval_dwords, sub_tile_dwords & -sub_tile_dwords)
                    : row_interval_dwords;

            // Do not place padding inside a 128-bit LDS access.
            if constexpr(interval_dwords < VecByteSize / index_t{sizeof(int32_t)})
            {
                return make_tuple(number<false>{}, number<0>{}, number<0>{});
            }
            else
            {
                return make_tuple(
                    number<true>{},                                     // I0: padding enabled?
                    config[Base::I1],                                   // I1: amount P-1
                    number<integer_log2_floor(interval_dwords) - 1>{}); // I2: interval log2(I)-1
            }
        }
    }

    // LDS descriptor (MN x K) for the non-transposed layout.
    // It places the pads exactly where TDM writes them (see GetLdsPaddingConfig).
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsBlockDescriptor()
    {
        using DataType = std::conditional_t<IsA, LdsADataType<Problem>, LdsBDataType<Problem>>;
        constexpr index_t MNPerBlock =
            IsA ? Problem::BlockGemmShape::kM : Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        constexpr auto config        = GetLdsPaddingConfig<Problem, IsA>();
        constexpr auto IsNeedPadding = config[Base::I0];

        // -1 makes PaddingDataAmount equal 0 when padding is disabled
        constexpr auto PaddingAmount    = IsNeedPadding ? config[Base::I1] : -1;
        constexpr index_t PackedSize    = numeric_traits<DataType>::PackedSize;
        constexpr index_t DataTypeSize  = sizeof(DataType);
        constexpr index_t VectorLen     = VecByteSize / DataTypeSize * PackedSize;
        constexpr index_t RowElements   = GetLdsRowElements<Problem, IsA>();
        constexpr index_t LdsLayer      = RowElements / KPerBlock;
        constexpr index_t BytesPerDword = sizeof(int32_t);

        // Pad & interval: convert from dwords to logical elements
        constexpr index_t PaddingDataAmount =
            (PaddingAmount + 1) * BytesPerDword / DataTypeSize * PackedSize;
        constexpr index_t IntervalElements =
            IsNeedPadding
                ? (index_t{1} << (config[Base::I2] + 1)) * BytesPerDword / DataTypeSize * PackedSize
                : RowElements;

        if constexpr(IntervalElements == RowElements)
        {
            // Fold LdsLayer M/N rows into one LDS row of RowElements + pad, then unfold back to
            // (MN, K).

            // Physical layout: each LDS row holds LdsLayer logical M/N rows, then 1 pad
            constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<MNPerBlock / LdsLayer>{},
                           number<KPerBlock / VectorLen * LdsLayer>{},
                           number<VectorLen>{}),
                make_tuple(
                    number<RowElements + PaddingDataAmount>{}, number<VectorLen>{}, number<1>{}),
                number<VectorLen>{},
                number<1>{});

            constexpr auto lds_block_desc_1 = transform_tensor_descriptor(
                lds_block_desc_0,
                make_tuple(make_pass_through_transform(number<MNPerBlock / LdsLayer>{}),
                           make_unmerge_transform(
                               make_tuple(number<LdsLayer>{}, number<KPerBlock / VectorLen>{})),
                           make_pass_through_transform(number<VectorLen>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

            // Merge into logical coordinates (MN, K)
            return transform_tensor_descriptor(
                lds_block_desc_1,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(number<MNPerBlock / LdsLayer>{}, number<LdsLayer>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<KPerBlock / VectorLen>{}, number<VectorLen>{}))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else
        {
            static_assert(KPerBlock % IntervalElements == 0 && IntervalElements % VectorLen == 0,
                          "TDM pad interval must split K into whole 128-bit accesses");

            // f(q) = q + P * floor(q / I) where:
            // P = config[Base::I1] + 1 (dwords)
            // I = 2^(config[Base::I2] + 1) (dwords)
            // Each logical row holds whole intervals, so for row mn and offset k in that row, this
            // equals:
            // f(mn,k) = mn * KIntervals * (I + P) + k + P * floor(k / I).

            constexpr index_t KIntervals = KPerBlock / IntervalElements;

            // Physical layout: interval, then pad
            constexpr auto lds_block_desc = make_naive_tensor_descriptor(
                make_tuple(number<MNPerBlock>{},
                           number<KIntervals>{},
                           number<IntervalElements / VectorLen>{},
                           number<VectorLen>{}),
                make_tuple(number<KIntervals*(IntervalElements + PaddingDataAmount)>{},
                           number<IntervalElements + PaddingDataAmount>{},
                           number<VectorLen>{},
                           number<1>{}),
                number<VectorLen>{},
                number<1>{});

            // Merge back into logical coordinates (MN, K)
            return transform_tensor_descriptor(
                lds_block_desc,
                make_tuple(make_pass_through_transform(number<MNPerBlock>{}),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<KIntervals>{},
                                          number<IntervalElements / VectorLen>{},
                                          number<VectorLen>{}))),
                make_tuple(sequence<0>{}, sequence<1, 2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    // currently implement basic situation: the tile is divided into same parts
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        constexpr index_t BlockSize = Problem::kBlockSize;
        // for wave specialized policy, only one wave per workgroup will load A / B matrix from DRAM
        // to LDS
        constexpr index_t warpNum = WaveSpecialized ? 1 : (BlockSize / get_warp_size());

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        using ALayout =
            remove_cvref_t<std::tuple_element_t<number<0>{}, problem_as_layout_t<Problem>>>;

        // Tile : MPerBlock X KPerBlock
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            if constexpr(!WaveSpecialized)
            {
                static_assert(MPerBlock % warpNum == 0, "MPerBlock should be divided by warpNum");
            }
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<warpNum, MPerBlock / warpNum>, sequence<KPerBlock>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{},
                bool_constant<true>{});
        }
        // Tile : KPerBlock * MPerBlock
        else
        {
            if constexpr(!WaveSpecialized)
            {
                static_assert(KPerBlock % warpNum == 0, "KPerBlock should be divided by warpNum");
            }
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<warpNum, KPerBlock / warpNum>, sequence<MPerBlock>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{},
                bool_constant<true>{});
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        constexpr index_t BlockSize = Problem::kBlockSize;
        // for wave specialized policy, only one wave per workgroup will load A / B matrix from DRAM
        // to LDS
        constexpr index_t warpNum = WaveSpecialized ? 1 : (BlockSize / get_warp_size());

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        using BLayout =
            remove_cvref_t<std::tuple_element_t<number<0>{}, problem_bs_layout_t<Problem>>>;

        // Tile : KPerBlock X NPerBlock
        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            if constexpr(!WaveSpecialized)
            {
                static_assert(KPerBlock % warpNum == 0, "KPerBlock should be divided by warpNum");
            }
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<warpNum, KPerBlock / warpNum>, sequence<NPerBlock>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{},
                bool_constant<true>{});
        }
        // Tile : NPerBlock * KPerBlock
        else
        {
            if constexpr(!WaveSpecialized)
            {
                static_assert(NPerBlock % warpNum == 0, "NPerBlock should be divided by warpNum");
            }
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<warpNum, NPerBlock / warpNum>, sequence<KPerBlock>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{},
                bool_constant<true>{});
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        if constexpr(Base::template is_a_load_tr<Problem>)
        {
            return Base::template MakeALdsBlockDescriptorForTrLoad<Problem>();
        }
        else
        {
            return MakeLdsBlockDescriptor<Problem, true>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        if constexpr(Base::template is_b_load_tr<Problem>)
        {
            return Base::template MakeBLdsBlockDescriptorForTrLoad<Problem>();
        }
        else
        {
            return MakeLdsBlockDescriptor<Problem, false>();
        }
    }

    template <MultiCastDirection Direction, typename Problem>
    CK_TILE_DEVICE static uint16_t GetTDMWorkgroupMask(dim3 block_id_in_cluster)
    {
        constexpr index_t MCluster = Problem::BlockGemmShape::kclusterM;
        constexpr index_t NCluster = Problem::BlockGemmShape::kclusterN;

        auto is_participant = [&](auto i_m, auto i_n) {
            if constexpr(Direction == MultiCastDirection::kM)
            {
                return i_m == block_id_in_cluster.x;
            }
            else if constexpr(Direction == MultiCastDirection::kN)
            {
                return (i_n == block_id_in_cluster.y);
            }
            else // Direction == MultiCastDirection::kMN
            {
                return (i_m == block_id_in_cluster.x) || (i_n == block_id_in_cluster.y);
            }
        };

        // Iterate over all possible (m, n) block coordinates in the cluster. If the current (m,
        // n) block is a participant according to the multicast direction, set the corresponding
        // bit in the mask. for matmul AxB, A broadcasts from M direction, B broadcasts from N
        // direction.
        uint16_t block_id_mask = 0;
        static_for<0, NCluster, 1>{}([&](auto n) {
            static_for<0, MCluster, 1>{}([&](auto m) {
                if(is_participant(m, n))
                {
                    block_id_mask |= (1 << (n * MCluster + m));
                }
            });
        });
        return block_id_mask;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetEstimatedVgprCount()
    {
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using BDataType = remove_cvref_t<typename Problem::BDataType>;
        using CDataType = remove_cvref_t<typename Problem::CDataType>;

        constexpr index_t MWarps       = Problem::BlockGemmShape::BlockWarps::at(Base::I0);
        constexpr index_t NWarps       = Problem::BlockGemmShape::BlockWarps::at(Base::I1);
        constexpr index_t warpSize     = get_warp_size();
        constexpr index_t BlockSize    = Problem::kBlockSize;
        constexpr index_t BytesPerVGPR = 4;
        constexpr index_t AccVGPRNum =
            sizeof(CDataType) * MPerBlock * NPerBlock / BlockSize / BytesPerVGPR;

        // this is used to calculate DoubleBufferFactor which is 2.5; this is to make sure float
        // calculation in constexpr is avoided
        constexpr index_t DoubleBufferNumerator   = 5;
        constexpr index_t DoubleBufferDenominator = 2;

        constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
        constexpr index_t BPackedSize = numeric_traits<BDataType>::PackedSize;

        constexpr index_t ALoadVGPRNum = sizeof(ADataType) / APackedSize * MPerBlock * KPerBlock /
                                         MWarps / warpSize / BytesPerVGPR * DoubleBufferNumerator /
                                         DoubleBufferDenominator;

        constexpr index_t BLoadVGPRNum = sizeof(BDataType) / BPackedSize * NPerBlock * KPerBlock /
                                         NWarps / warpSize / BytesPerVGPR * DoubleBufferNumerator /
                                         DoubleBufferDenominator;

        constexpr index_t TotalInputVGPRNum = ALoadVGPRNum + BLoadVGPRNum;

        return make_tuple(number<AccVGPRNum>{}, number<TotalInputVGPRNum>{});
    }

    // this function is used to get SubTile Number
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetPipelineSubTileNum()
    {
        constexpr index_t KPerBlock        = Problem::BlockGemmShape::kK;
        constexpr index_t KPerTile         = Problem::BlockGemmShape::WarpTile::at(Base::I2);
        constexpr index_t max_sub_tile_num = KPerBlock / KPerTile;

        constexpr auto estimated_vgpr = GetEstimatedVgprCount<Problem>();

        constexpr auto acc_vgpr_num   = estimated_vgpr.at(number<0>{});
        constexpr auto input_vgpr_num = estimated_vgpr.at(number<1>{});

        constexpr index_t vgpr_capacity = get_max_vgpr_count();
        // sub tile number; have 1, 2, 4 choices
        constexpr index_t sub_tile_num = ((input_vgpr_num + acc_vgpr_num) <= vgpr_capacity) ? 1
                                         : ((input_vgpr_num / 2 + acc_vgpr_num) <= vgpr_capacity)
                                             ? 2
                                             : 4;

        return number<min(sub_tile_num, max_sub_tile_num)>{};
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeScaleADramTileDistribution()
    {
        using TileShape  = typename Problem::BlockGemmShape;
        using BlockWarps = typename TileShape::BlockWarps;

        constexpr index_t MWarps     = BlockWarps::at(Base::I0);
        constexpr index_t NWarps     = BlockWarps::at(Base::I1);
        constexpr index_t kMPerBlock = TileShape::kM;
        constexpr index_t kKPerBlock = TileShape::kK;

        constexpr index_t ScaleSize = detail::scale_block_size_or_default<Problem>::value;
        // PackSize = number of e8m0 bytes per int32_t element (always 4).
        // scale32: 1 int32_t per thread (4 bytes); scale16: 2 int32_t per thread (8 bytes ->
        // int64_t).
        constexpr index_t PackSize = 4;

        // WarpM: unique M-lanes per warp (equals WarpTile::M for WMMA).
        // For scale32, WarpM == WaveSize (all lanes are unique in M).
        // For scale16, WarpM < WaveSize: the remaining WaveSize/WarpM lanes
        // repeat the same M-rows (they cover different N columns), so they
        // must be replicated in the RS rather than treated as distinct M-positions.
        constexpr index_t WarpM        = TileShape::WarpTile::at(Base::I0);
        constexpr index_t NWithinWarp  = get_warp_size() / WarpM;
        constexpr index_t MIterPerWarp = kMPerBlock / MWarps / WarpM;
        static_assert(WarpM == 16 || WarpM == 32,
                      "scale tile distribution supports only WarpTile::M of 16 or 32");
        static_assert(get_warp_size() % WarpM == 0, "WarpTile::M must divide the wavefront size");

        // H0 = <MIterPerWarp, MWarps, WarpM> matching A block distribution order.
        // M-index = mIter * (MWarps * WarpM) + m_warp * WarpM + lane_m
        // P1 (lane_id) decomposes as: lane_id = n_within_warp + NWithinWarp * lane_m
        // so P1[0]=NWithinWarp (least sig, replication) and P1[1]=WarpM (most sig, M-position).
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarps, NWithinWarp>,
                                       tuple<sequence<MIterPerWarp, MWarps, WarpM>,
                                             sequence<kKPerBlock / ScaleSize / PackSize, 1>>,
                                       tuple<sequence<1, 0>, sequence<0, 1>>,
                                       tuple<sequence<1, 0>, sequence<1, 2>>,
                                       sequence<1, 2, 2>,
                                       sequence<0, 0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeScaleBDramTileDistribution()
    {
        using TileShape  = typename Problem::BlockGemmShape;
        using BlockWarps = typename TileShape::BlockWarps;

        constexpr index_t MWarps     = BlockWarps::at(Base::I0);
        constexpr index_t NWarps     = BlockWarps::at(Base::I1);
        constexpr index_t kNPerBlock = TileShape::kN;
        constexpr index_t kKPerBlock = TileShape::kK;

        constexpr index_t ScaleSize = detail::scale_block_size_or_default<Problem>::value;
        constexpr index_t PackSize  = 4;

        // WarpN: unique N-lanes per warp (equals WarpTile::N for WMMA).
        // For scale16, WaveSize/WarpN lanes repeat the same N-rows (covering
        // different M columns) and must be replicated in RS.
        constexpr index_t WarpN        = TileShape::WarpTile::at(Base::I1);
        constexpr index_t MWithinWarp  = get_warp_size() / WarpN;
        constexpr index_t NIterPerWarp = kNPerBlock / NWarps / WarpN;
        static_assert(WarpN == 16 || WarpN == 32,
                      "scale tile distribution supports only WarpTile::N of 16 or 32");
        static_assert(get_warp_size() % WarpN == 0, "WarpTile::N must divide the wavefront size");

        // H0 = <NIterPerWarp, NWarps, WarpN> matching B block distribution order.
        // N-index = nIter * (NWarps * WarpN) + n_warp * WarpN + lane_n
        // P1 (lane_id) decomposes as: lane_id = m_within_warp + MWithinWarp * lane_n
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<MWarps, MWithinWarp>,
                                       tuple<sequence<NIterPerWarp, NWarps, WarpN>,
                                             sequence<kKPerBlock / ScaleSize / PackSize, 1>>,
                                       tuple<sequence<0, 1>, sequence<0, 1>>,
                                       tuple<sequence<0, 1>, sequence<1, 2>>,
                                       sequence<1, 2, 2>,
                                       sequence<0, 0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        constexpr auto pipeline_tune_params = GetPipelineSubTileNum<Problem>();
        constexpr index_t sub_tile_num      = pipeline_tune_params.value;
        constexpr index_t kScaleBlockSize   = detail::scale_block_size_or_default<Problem>::value;

#if defined(__gfx125__)
        // Compute WGAttrNumAccess for a single operand (A or B).
        //
        // For normal types:
        //   vec_size       = instruction K-pack (kAKPack / kBKPack)
        //   total_elements = K_warp_tile / 2  (2 lanes share K)
        //
        // For packed types (f4/f6) with transpose load:
        //   vec_size       = PackedSize * elements_per_vgpr  (one VGPR worth of packed data)
        //   total_elements = instruction K-pack              (kAKPack / kBKPack)
        constexpr auto compute_num_access = []<typename DataType>(bool is_load_tr,
                                                                  index_t instr_kpack,
                                                                  index_t k_warp_tile) constexpr {
            constexpr index_t packed_size   = numeric_traits<DataType>::PackedSize;
            constexpr index_t bits_per_elem = sizeof(DataType) * 8 / packed_size;
            // in gfx1250 always use double vgpr for fp4 and fp8 in tr load
            constexpr index_t elems_per_dvgpr = 64 / bits_per_elem;

            const bool is_packed =
                is_load_tr && ((packed_size > 1) || ((bits_per_elem == 8) && (instr_kpack > 8)));
            const auto vec_size       = is_packed ? elems_per_dvgpr : instr_kpack;
            const auto total_elements = is_packed ? instr_kpack : (k_warp_tile / 2);
            const auto ratio          = total_elements / vec_size;

            // Map the ratio to WGAttrNumAccessEnum;
            // is_packed selects between Packed* and non-Packed variants.
            switch(ratio)
            {
            case 1:
                return is_packed ? WGAttrNumAccessEnum::PackedSingle : WGAttrNumAccessEnum::Single;
            case 2:
                return is_packed ? WGAttrNumAccessEnum::PackedDouble : WGAttrNumAccessEnum::Double;
            case 4: return is_packed ? WGAttrNumAccessEnum::PackedQuad : WGAttrNumAccessEnum::Quad;
            case 8: return is_packed ? WGAttrNumAccessEnum::PackedOcta : WGAttrNumAccessEnum::Octa;
            default: return WGAttrNumAccessEnum::Invalid;
            }
        };

        // Probe the default warp gemm to get instruction K-pack sizes.
        // For scale16, probe the scale16 type directly; otherwise use the standard dispatcher.
        using WarpGemmProbe      = WarpGemmDispatcher<typename Problem::ADataType,
                                                      typename Problem::BDataType,
                                                      typename Problem::CDataType,
                                                      WarpTile::at(Base::I0),
                                                      WarpTile::at(Base::I1),
                                                      WarpTile::at(Base::I2),
                                                      Problem::TransposeC,
                                                      false,
                                                      false,
                                                      WGAttrNumAccessEnum::Default,
                                                      WGAttrNumAccessEnum::Default,
                                                      (kScaleBlockSize == 16)>;
        constexpr index_t k_warp = WarpTile::at(Base::I2);

        constexpr auto a_wg_attr_num_access =
            compute_num_access.template operator()<typename Problem::ADataType>(
                Base::template is_a_load_tr<Problem>, WarpGemmProbe::kAKPack, k_warp);

        constexpr auto b_wg_attr_num_access =
            compute_num_access.template operator()<typename Problem::BDataType>(
                Base::template is_b_load_tr<Problem>, WarpGemmProbe::kBKPack, k_warp);
#else
        constexpr auto a_wg_attr_num_access = WGAttrNumAccessEnum::Default;
        constexpr auto b_wg_attr_num_access = WGAttrNumAccessEnum::Default;
#endif
        using WarpGemm = WarpGemmDispatcher<typename Problem::ADataType,
                                            typename Problem::BDataType,
                                            typename Problem::CDataType, // AccDataType
                                            WarpTile::at(Base::I0),
                                            WarpTile::at(Base::I1),
                                            WarpTile::at(Base::I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            a_wg_attr_num_access,
                                            b_wg_attr_num_access,
                                            (kScaleBlockSize == 16)>;

        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm,
                                                                    sub_tile_num>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};

// Type aliases for backward compatibility
using GemmPipelineAgBgCrCompTDMWaveSpecializedPolicy = GemmPipelineAgBgCrCompTDMDefaultPolicy<true>;

} // namespace ck_tile
