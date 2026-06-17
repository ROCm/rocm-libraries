// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// [v3 Phase 1] Sync-load variant of MXFlatmmPipelineAgBgCrPolicy.
// fp6-only (locked by static_assert below). Differences from baseline:
//   - D1: LDS xor_transform on <M3, K1> replaced with pass_through (identity).
//         TODO[v3 P3.2]: revisit swizzle for bank-conflict perf recovery
//         once correctness is established.
//   - D2: Aload_num constant added (= Aload_num_perK * KIterPerWarp), so the
//         new pipeline's s_waitcnt formula can count A loads which the async
//         baseline excludes (async DRAM→LDS uses a separate lds-direct counter).
// Baseline source is physically untouched (additive principle).

#pragma once

#include "ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp"

namespace ck_tile {

namespace detail {
template <typename Problem>
struct MXFlatmmPipelineAgBgCrPolicySync : UniversalFlatmmPipelineAgBgCrPolicy
{
    static_assert(std::is_same_v<remove_cvref_t<typename Problem::ADataType>, pk_fp6x16_t>,
                  "MXFlatmmPipelineAgBgCrPolicySync is locked to pk_fp6x16_t (fp6 only)");
    static_assert(std::is_same_v<remove_cvref_t<typename Problem::BDataType>, pk_fp6x16_t>,
                  "MXFlatmmPipelineAgBgCrPolicySync is locked to pk_fp6x16_t (fp6 only)");

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr index_t kDramLoadPackBytes = 128;
    static constexpr index_t DWORDx4            = 16;
    static constexpr index_t DWORDx3            = 12;

    static constexpr int MXdlPack = 2;
    static constexpr int NXdlPack = 2;
    static constexpr int KXdlPack = 2;

    private:
    using ADataType                      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType                      = remove_cvref_t<typename Problem::BDataType>;
    static constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
    static constexpr index_t BPackedSize = numeric_traits<BDataType>::PackedSize;

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    static_assert(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>);

    using TileShape                    = typename Problem::BlockGemmShape;
    using BlockWarps                   = typename TileShape::BlockWarps;
    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t WaveSize  = get_warp_size();
    static constexpr index_t WaveNum   = BlockSize / WaveSize;

    static constexpr index_t MPerBlock = TileShape::kM;
    static constexpr index_t NPerBlock = TileShape::kN;
    static constexpr index_t KPerBlock = TileShape::kK;
    static constexpr index_t MWarps    = BlockWarps::at(I0);
    static constexpr index_t NWarps    = BlockWarps::at(I1);
    static_assert(WaveNum == MWarps * NWarps, "Block warps do not match block size");

    static constexpr index_t MPerXdl = TileShape::WarpTile::at(I0);
    static constexpr index_t NPerXdl = TileShape::WarpTile::at(I1);
    static constexpr index_t KPerXdl = TileShape::WarpTile::at(I2);
    // 16x16x128 or 32x32x64 scaled-f8f6f4 XDL (both square M==N).
    static_assert((MPerXdl == 16 && NPerXdl == 16) || (MPerXdl == 32 && NPerXdl == 32));
    static constexpr index_t K_Lane   = get_warp_size() / MPerXdl; // 4 (16x16) | 2 (32x32)
    static constexpr index_t K_Thread = KPerXdl / K_Lane;          // 32

    public:
    static constexpr index_t AK1 = DWORDx4 * APackedSize;
    static constexpr index_t BK1 = DWORDx4 * BPackedSize;

    CK_TILE_HOST_DEVICE static constexpr auto GetBlockFlatmm()
    {
        using WarpTile          = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm          = WarpGemmDispatcher< //
            ADataType,
            BDataType,
            typename Problem::CDataType,
            WarpTile::at(I0),
            WarpTile::at(I1),
            WarpTile::at(I2),
            Problem::TransposeC>;
        using BlockFlatmmPolicy = BlockFlatmmASmemBSmemCRegV1CustomPolicy< //
            ADataType,
            BDataType,
            typename Problem::CDataType,
            BlockWarps,
            WarpGemm>;
        return BlockFlatmmASmemBSmemCRegV1<Problem, BlockFlatmmPolicy>{};
    }

    CK_TILE_DEVICE static constexpr auto MakeMX_ABytesDramTileDistribution()
    {
        // 3×contiguous dwordx4 per load (48 bytes), K0=1 (no repeat).
        constexpr index_t K2 = 3 * DWORDx4; // 48 bytes
        constexpr index_t K1 = std::is_same_v<ADataType, pk_fp6x16_t>
                                   ? kDramLoadPackBytes / DWORDx4 / 2  // 4 for fp6
                                   : kDramLoadPackBytes / DWORDx4;     // 8 for others
        constexpr index_t K0 =
            KPerBlock / APackedSize * sizeof(ADataType) / (K1 * K2);

        constexpr index_t M2 = WaveSize / K1;
        constexpr index_t M1 = BlockSize / WaveSize; // 4
        constexpr index_t M0 = MPerBlock / (M2 * M1);
        static_assert(M0 * M1 * M2 == MPerBlock, "M0, M1, M2 must cover whole MPerBlock!");
        static_assert(K0 * K1 * K2 == KPerBlock / APackedSize * sizeof(ADataType),
                      "K0, K1, K2 must cover whole KPerBlock!");

        return make_static_tile_distribution(
            tile_distribution_encoding< //
                sequence<1>,
                tuple<sequence<M0, M1, M2>, sequence<K0, K1, K2>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 1>>,
                sequence<1, 2, 2>, // Y: M0, K0, K2(=vector)
                sequence<0, 0, 2>>{});
    }

    // Sync path: plain byte DRAM window without XOR swizzle.
    // Async path uses XOR on <M1,K1> for bank-conflict reduction under direct-to-LDS writes;
    // sync path does buffer_load→VGPR→ds_write, so the XOR is not needed here.
    // TODO[v3 P3.2]: re-derive correct swizzle for ds_write bank layout.
    template <typename WindowTmp>
    CK_TILE_DEVICE static constexpr auto
    MakeMX_ASyncLoadBytesDramWindow(const WindowTmp& window_tmp)
    {
        constexpr auto ndims = std::decay_t<decltype(window_tmp)>::get_num_of_dimension();
        static_assert(ndims == 2, "only support 2D tensor");
        auto&& tensor_view_tmp  = window_tmp.get_bottom_tensor_view();
        const auto [rows, cols] = tensor_view_tmp.get_tensor_descriptor().get_lengths();

        // fp6: K2=48 (3×dwordx4) so window's safe vector length = 48,
        // triggering the N==48 inline-asm path (3× buffer_load_dwordx4).
        constexpr index_t K2 = std::is_same_v<ADataType, pk_fp6x16_t> ? 3 * DWORDx4 : DWORDx4;
        constexpr index_t K1 = std::is_same_v<ADataType, pk_fp6x16_t>
                                   ? 4   // K1*K2=192, 192/12=16 (exact)
                                   : kDramLoadPackBytes / DWORDx4; // 8
        const index_t K0     = cols / (K1 * K2 / sizeof(ADataType) * APackedSize);
        const auto col_lens  = make_tuple(K0, number<K1>{}, number<K2>{});

        constexpr index_t M1 = 4;
        const index_t M0     = integer_divide_ceil(rows, M1);
        const auto row_lens  = make_tuple(M0, number<M1>{});

        const auto d0 = make_naive_tensor_descriptor_packed(container_concat(row_lens, col_lens));
        const auto desc_0 = decltype(d0)(
            d0.get_transforms(),
            tensor_view_tmp.get_tensor_descriptor().get_element_space_size());
        // D1: pass_through on <M1,K1> — no XOR swizzle for sync path.
        const auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(M0),
                       make_pass_through_transform(number<M1>{}),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(number<K1>{}),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}));
        const auto desc = transform_tensor_descriptor(
            desc_1,
            make_tuple(make_merge_transform_v3_division_mod(row_lens),
                       make_merge_transform_v3_division_mod(col_lens)),
            make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        auto&& byte_ptr = reinterpret_cast<const uint8_t*>(&(tensor_view_tmp.get_buffer_view()(0)));
        auto&& byte_tensor_view = make_tensor_view<address_space_enum::global>(byte_ptr, desc);

        auto&& origin_tmp = window_tmp.get_window_origin();
        // Window K must match distribution K = KPerBlock/APackedSize*sizeof(ADataType).
        // For fp6: 512/16*12 = 384. The old formula "KPerBlock / (APackedSize/sizeof)" truncates
        // for fp6 (16/12=1 in integer) giving 512 instead of 384 → buffer size mismatch.
        constexpr index_t kKBytesPerBlock = KPerBlock / APackedSize * sizeof(ADataType);
        return make_tile_window(
            byte_tensor_view,
            make_tuple(number<MPerBlock>{}, number<kKBytesPerBlock>{}),
            {origin_tmp[0], static_cast<index_t>(origin_tmp[1] * sizeof(ADataType) / APackedSize)},
            MakeMX_ABytesDramTileDistribution());
    }

    CK_TILE_DEVICE static constexpr auto MakeMX_ALdsBytesBlockDescriptor()
    {
        constexpr index_t K2     = std::is_same_v<ADataType, pk_fp6x16_t> ? DWORDx3 : AK1 / APackedSize;
        constexpr index_t K2_Pad = std::is_same_v<ADataType, pk_fp6x16_t> ? DWORDx3 : 16;
        constexpr index_t K1     = kDramLoadPackBytes / DWORDx4; // 8
        constexpr index_t K0     = std::is_same_v<ADataType, pk_fp6x16_t>
                                       ? KPerBlock / (K1 * K2 / sizeof(ADataType) * APackedSize)
                                       : KPerBlock / (K1 * AK1); // KPerBlock/256
        static_assert(K0 * K1 * K2 / sizeof(ADataType) * APackedSize == KPerBlock,
                      "K0, K1, K2 must cover whole KPerBlock!");

        constexpr index_t M3 = 4;                   // so that we can use imm offset to load lds
        constexpr index_t M2 = WaveSize / K1 / M3;  // 2
        constexpr index_t M1 = MPerXdl / (M2 * M3); // 2
        constexpr index_t M0 = MPerBlock / (M1 * M2 * M3); // MPerBlock/16
        static_assert(M0 * M1 * M2 * M3 == MPerBlock, "M0, M1, M2, M3 must cover whole MPerBlock!");

        constexpr index_t Pad = 4 * K2;

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor( //
            make_tuple(number<M0>{},
                       number<K0>{},
                       number<M1>{},
                       number<M2>{},
                       number<M3>{},
                       number<K1>{},
                       number<K2>{}),
            make_tuple(number<K0*(M1 * (M2 * M3 * K1 * K2_Pad) + (M1 - 1) * Pad)>{},
                       number<M1*(M2 * M3 * K1 * K2_Pad) + (M1 - 1) * Pad>{},
                       number<M2 * M3 * K1 * K2_Pad + Pad>{},
                       number<M3 * K1 * K2_Pad>{},
                       number<K1 * K2_Pad>{},
                       number<K2_Pad>{},
                       number<1>{}),
            number<std::is_same_v<ADataType, pk_fp6x16_t> ? 3 * DWORDx4 : K2>{},
            number<1>{});

        // [v3 P1.1 / D1] identity transform on <M3, K1> (sync path).
        // Baseline used make_xor_transform here to reduce ds_read bank conflicts
        // under the async DRAM-direct-to-LDS write pattern. Sync writes (ds_write_b128
        // from per-thread VGPRs) hit banks differently — re-derive the right swizzle
        // in P3.2 perf-recovery once correctness is established. For now use
        // pass-through (no swizzle) to keep mental model simple while we get
        // sync working.
        constexpr auto a_lds_block_desc_1 = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(M0),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(M1),
                       make_pass_through_transform(M2),
                       make_pass_through_transform(number<M3>{}),  // D1: was xor on <M3,K1>
                       make_pass_through_transform(number<K1>{}),  // D1: was xor on <M3,K1>
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{},
                       sequence<1>{},
                       sequence<2>{},
                       sequence<3>{},
                       sequence<4>{},   // D1: was sequence<4,5> for xor merge
                       sequence<5>{},   // D1: was sequence<4,5> for xor merge
                       sequence<6>{}),
            make_tuple(sequence<0>{},
                       sequence<1>{},
                       sequence<2>{},
                       sequence<3>{},
                       sequence<4>{},   // D1: was sequence<4,5> for xor merge
                       sequence<5>{},   // D1: was sequence<4,5> for xor merge
                       sequence<6>{}));
        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_1,
            make_tuple(make_merge_transform_v3_division_mod(
                           make_tuple(number<M0>{}, number<M1>{}, number<M2>{}, number<M3>{})),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<K0>{}, number<K1>{}, number<K2>{}))),
            make_tuple(sequence<0, 2, 3, 4>{}, sequence<1, 5, 6>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        // return a_lds_block_desc_permuted;
        return a_lds_block_desc;
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ALDSBytes_TileDistribution()
    {
        static_assert(BlockWarps::at(I0) == 1, "requires Wave_M == 1");

        if constexpr(std::is_same_v<ADataType, pk_fp4_t>)
            return make_static_tile_distribution(
                tile_distribution_encoding< //
                    sequence<NWarps>,
                    tuple<sequence<MWarps, MXdlPack, MPerXdl>, sequence<K_Lane, AK1 / APackedSize>>,
                    tuple<sequence<1, 0>, sequence<2, 1>>,
                    tuple<sequence<0, 0>, sequence<0, 2>>,
                    sequence<2>,
                    sequence<1>>{});
        else if constexpr(std::is_same_v<ADataType, fp8_t>)
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<NWarps>,
                    tuple<sequence<MWarps, MXdlPack, MPerXdl>,
                          sequence<K_Thread / AK1, K_Lane, AK1 / APackedSize>>,
                    tuple<sequence<1, 0>, sequence<2, 1>>,
                    tuple<sequence<0, 0>, sequence<1, 2>>,
                    sequence<2, 2>,
                    sequence<0, 2>>{});
        else if constexpr(std::is_same_v<ADataType, pk_fp6x16_t>)
            // K_Lane=WaveSize/MPerXdl (4 for 16x16, 2 for 32x32)
            return make_static_tile_distribution(
                tile_distribution_encoding< //
                    sequence<NWarps>,
                    tuple<sequence<MWarps, MXdlPack, MPerXdl>,
                          sequence<K_Lane, KPerXdl / (K_Lane * APackedSize), DWORDx3>>,
                    tuple<sequence<1, 0>, sequence<2, 1>>,
                    tuple<sequence<0, 0>, sequence<0, 2>>,
                    sequence<2, 2>,
                    sequence<1, 2>>{});
        else
            static_assert(false, "unsupported datatype");
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_BFlatBytesDramTileDistribution()
    {
        constexpr index_t K1          = WaveSize; // threads cnt in K dim
        constexpr index_t KWavePerBlk = 1;
        constexpr index_t K0          = KWavePerBlk;

        constexpr index_t WaveRepeat = WaveNum / TileShape::flatNPerWarp;

        if constexpr(std::is_same_v<BDataType, pk_fp4_t>)
            return make_static_tile_distribution(
                tile_distribution_encoding< //
                    sequence<WaveRepeat>,
                    tuple<sequence<NWarps, NXdlPack>,           // 4 2
                          sequence<K0, K1, BK1 / BPackedSize>>, // 1 64 16
                    tuple<sequence<0, 1, 2>, sequence<2>>,
                    tuple<sequence<0, 0, 0>, sequence<1>>,
                    sequence<2>,
                    sequence<2>>{});
        else if constexpr(std::is_same_v<BDataType, fp8_t>)
            return make_static_tile_distribution(
                tile_distribution_encoding< //
                    sequence<WaveRepeat>,
                    tuple<sequence<NWarps, NXdlPack>,                           // 4 2
                          sequence<K_Thread / BK1, K0, K1, BK1 / BPackedSize>>, // 2 1 64 16
                    tuple<sequence<0, 1, 2>, sequence<2>>,
                    tuple<sequence<0, 0, 1>, sequence<2>>,
                    sequence<2, 2>,
                    sequence<0, 3>>{});
        else if constexpr(std::is_same_v<BDataType, pk_fp6x16_t>)
            return make_static_tile_distribution(
                tile_distribution_encoding< //
                    sequence<WaveRepeat>,
                    tuple<sequence<NWarps, NXdlPack>, // 4 2
                          sequence<K0,
                                   K1,
                                   K_Thread * sizeof(BDataType) / (DWORDx3 * BPackedSize),
                                   DWORDx3>>, // 64 1 2 12
                    tuple<sequence<0, 1, 2>, sequence<2>>,
                    tuple<sequence<0, 0, 0>, sequence<1>>,
                    sequence<2, 2>,
                    sequence<2, 3>>{});
        else
            static_assert(false, "unsupported datatype");
    }

    template <typename WindowTmp>
    CK_TILE_HOST_DEVICE static constexpr auto
    MakeMX_BFlatBytesDramWindow(const WindowTmp& window_tmp)
    {
        constexpr auto M_Warp_Tile  = Problem::BlockGemmShape::WarpTile::at(I1);
        constexpr auto flatNPerWarp = Problem::BlockGemmShape::flatNPerWarp;
        constexpr auto flatKPerWarp = Problem::BlockGemmShape::flatKPerWarp;

        static_assert(std::decay_t<decltype(window_tmp)>::get_num_of_dimension() == 2);
        auto&& tensor_view_tmp          = window_tmp.get_bottom_tensor_view();
        const auto [flat_n, flat_k]     = tensor_view_tmp.get_tensor_descriptor().get_lengths();
        constexpr auto flat_k_per_block = KPerBlock * M_Warp_Tile;
        auto&& byte_tensor_desc         = transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(
                make_tuple(flat_n,
                           flat_k / flat_k_per_block,
                           number<flat_k_per_block / BPackedSize * sizeof(BDataType)>{})),
            make_tuple(make_pass_through_transform(flat_n),
                       make_merge_transform_v3_division_mod(make_tuple(
                           flat_k / flat_k_per_block,
                           number<flat_k_per_block / BPackedSize * sizeof(BDataType)>{}))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        auto&& byte_ptr = reinterpret_cast<const uint8_t*>(&(tensor_view_tmp.get_buffer_view()(0)));
        auto&& byte_tensor_view =
            make_tensor_view<address_space_enum::global>(byte_ptr, byte_tensor_desc);
        auto&& origin_tmp = window_tmp.get_window_origin();
        auto origin_n     = origin_tmp[0];
        auto origin_k     = static_cast<int>(origin_tmp[1] * sizeof(BDataType) / BPackedSize);
        return make_tile_window(
            byte_tensor_view,
            make_tuple(number<flatNPerWarp>{},
                       number<flatKPerWarp * sizeof(BDataType) / BPackedSize>{}),
            {origin_n, origin_k},
            MakeMX_BFlatBytesDramTileDistribution());
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleA_DramTileDistribution()
    {
        constexpr index_t M_Lanes = TileShape::WarpTile::at(I0);
        constexpr index_t K_Lanes = 64 / M_Lanes;

        // Y dimension (M) decomposition
        constexpr index_t Y2 = M_Lanes;
        constexpr index_t Y1 = MWarps;
        constexpr index_t Y0 = MPerBlock / (MXdlPack * Y1 * Y2);

        // X dimension (K) decomposition
        constexpr index_t X0 = K_Lanes;
        constexpr index_t X1 = 1; // packed 2x2 E8M0 data into 1 int32_t for load

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarps>, // repeat NWarps
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>,
                                       tuple<sequence<1, 0>, sequence<2, 1>>,
                                       tuple<sequence<1, 0>, sequence<0, 2>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleB_DramTileDistribution()
    {
        constexpr index_t N_Lanes = TileShape::WarpTile::at(I1);
        constexpr index_t K_Lanes = 64 / N_Lanes;

        // Y dimension (M) decomposition
        constexpr index_t Y2 = N_Lanes;
        constexpr index_t Y1 = NWarps;
        constexpr index_t Y0 = NPerBlock / (NXdlPack * Y1 * Y2);

        // X dimension (K) decomposition
        constexpr index_t X0 = K_Lanes;
        constexpr index_t X1 = 1; // packed 2x2 E8M0 data into 1 int32_t for load

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<MWarps>, // ?
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>,
                                       tuple<sequence<0, 1>, sequence<2, 1>>,
                                       tuple<sequence<0, 1>, sequence<0, 2>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleA_FlatDramTileDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarps>,                      // ?
                                       tuple<sequence<MWarps, MPerXdl>,       // second direction
                                             sequence<K_Lane, 1>>,            // first direction
                                       tuple<sequence<1, 0>, sequence<2, 1>>, // which direction
                                       tuple<sequence<0, 0>, sequence<0, 1>>, // which index
                                       // <repeat, vec_load>
                                       sequence<2>,
                                       sequence<1>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleB_FlatDramTileDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<MWarps>,                      // ?
                                       tuple<sequence<NWarps, NPerXdl>,       // second direction
                                             sequence<K_Lane, 1>>,            // first direction
                                       tuple<sequence<0, 1>, sequence<2, 1>>, // which direction
                                       tuple<sequence<0, 0>, sequence<0, 1>>, // which index
                                       // <repeat, vec_load>
                                       sequence<2>,
                                       sequence<1>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeA()
    {
        if constexpr(!std::is_same_v<ADataType, pk_fp6x16_t>)
        {
            return sizeof(ADataType) * MakeMX_ALdsBytesBlockDescriptor().get_element_space_size();
        }
        else
        {
            return MakeMX_ALdsBytesBlockDescriptor().get_element_space_size();
        }
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return GetSmemSizeA(); }
};
} // namespace detail

struct MXFlatmmPipelineAgBgCrPolicySync
{

#define FORWARD_METHOD_(method)                                                                    \
    template <typename Problem, typename... Args>                                                  \
    CK_TILE_HOST_DEVICE static constexpr auto method(Args&&... args)                               \
    {                                                                                              \
        return detail::MXFlatmmPipelineAgBgCrPolicySync<Problem>::method(std::forward<Args>(args)...); \
    }

    FORWARD_METHOD_(GetBlockFlatmm);
    FORWARD_METHOD_(MakeMX_ASyncLoadBytesDramWindow);
    FORWARD_METHOD_(MakeMX_ABytesDramTileDistribution);
    FORWARD_METHOD_(MakeMX_ALdsBytesBlockDescriptor);
    FORWARD_METHOD_(MakeMX_ALDSBytes_TileDistribution);
    FORWARD_METHOD_(MakeMX_BFlatBytesDramTileDistribution);
    FORWARD_METHOD_(MakeMX_BFlatBytesDramWindow);
    FORWARD_METHOD_(MakeMX_ScaleA_DramTileDistribution);
    FORWARD_METHOD_(MakeMX_ScaleB_DramTileDistribution);
    FORWARD_METHOD_(MakeMX_ScaleA_FlatDramTileDistribution);
    FORWARD_METHOD_(MakeMX_ScaleB_FlatDramTileDistribution);
    FORWARD_METHOD_(GetSmemSizeA);
    FORWARD_METHOD_(GetSmemSize);

#undef FORWARD_METHOD_
};

} // namespace ck_tile
