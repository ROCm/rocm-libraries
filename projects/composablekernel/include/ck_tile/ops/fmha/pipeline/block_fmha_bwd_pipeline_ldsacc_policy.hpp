// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"

namespace ck_tile {

// Policy for the bwd pipeline that keeps the dK and dV accumulators in LDS
// instead of registers.
//
// Why: on gfx1250 (wave32, 1024 VGPRs/SIMD) the two fp32 accumulators are
// kN0*headdim floats each, i.e. kN0*headdim/kBlockSize per lane. At kN0=64,
// headdim=128, kBlockSize=128 that is 64 VGPRs apiece -- 128 of the 597 the
// kernel needs, which is exactly what pushes it from 2 waves/SIMD down to 1.
// Measured: headdim 64 -> 377 VGPRs, occupancy 2, 226 TFLOPS;
//           headdim 128 -> 597 VGPRs, occupancy 1, 120 TFLOPS.
// LDS is nearly free here (36 KiB of the 320 KiB a gfx1250 workgroup may take),
// so the accumulators are the one large thing that can move without cost.
//
// Everything else is inherited unchanged; this policy only adds the two
// accumulator descriptors and re-does the smem budget.
struct BlockFmhaBwdPipelineLdsAccPolicy : BlockFmhaBwdPipelineDefaultPolicy
{
    // Padding, in floats, added to the leading dimension of each accumulator.
    //
    // The gemm C fragment hands lane L the values at rows m = j + 8*(L>>4)
    // (j = 0..7) and column n = L&15, so the two half-waves of a wave32 touch
    // rows that are 8 apart. With an unpadded stride those two rows land on the
    // same LDS banks whenever 8*headdim is a multiple of the 64 banks -- true
    // for every headdim we build (32/64/128/256), giving a 2-way conflict on
    // every access. A 4-float pad makes 8*(headdim+4) mod 64 == 32, which
    // separates the two half-waves by 32 banks, and keeps the row start
    // 16-byte aligned.
    static constexpr index_t kAccLdsPad = 4;

    // ---- K: one plain box, K^T read back by ds_load_tr ----------------------
    //
    // K used to be materialised twice: once as-is for gemm_0, and once through
    // shuffle_tile into a second LDS copy so gemm_4 could read K^T with a plain
    // load_tile. gfx1250 has ds_load_tr16_b128, so the second copy and the
    // shuffle are both unnecessary -- one box, read straight for gemm_0 and
    // transposed for gemm_4.
    //
    // The box has to be plain: ds_load_tr reads a hardware-fixed physical
    // pattern, so a descriptor-level XOR has no opportunity to cancel the way it
    // does on the per-element load_tile path. That is also exactly what TDM
    // needs, so the two changes want the same layout.
    //
    // Verified on gfx1250 against the shuffle path as reference: both produce
    // K^T with extent [0..kQKHeaddim-1]x[0..kN0-1] and zero mismatches.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsWriteBlockDescriptor()
    {
        using KDataType              = typename Problem::KDataType;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPack     = 16 / sizeof(KDataType);

        return make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
            make_tuple(number<kKPerBlock>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeK()
    {
        return sizeof(typename Problem::KDataType) *
               MakeKLdsWriteBlockDescriptor<Problem>().get_element_space_size();
    }

    // The second K copy is gone, so nothing needs the shuffled staging area.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeKT()
    {
        return 0;
    }

    // Same encoding the base policy builds for gemm_4's B operand, wrapped so
    // that load_tile_transpose fills it. Identical logical content, different
    // physical arrangement.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKTRegBlockDescriptor()
    {
        using BlockGemm = remove_cvref_t<decltype(GetSGradKTBlockGemm<Problem>())>;
        using WarpGemm  = typename BlockGemm::WarpGemm;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto kt_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<MWarp>,
            tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>, // 2 4, 4
            tuple<sequence<0, 1>>,
            tuple<sequence<0, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto kt_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            kt_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        auto output =
            make_static_tile_distribution(typename InputTileDistributionTraits<
                                          decltype(kt_block_dstr_encode),
                                          typename Problem::KDataType>::TransposedDstrEncode{});
        return output;
    }

    // K is moved global->LDS by TDM, same three requirements as V: a plain box
    // (above), a trivial tile-major DRAM walk (here) and no writer-side padding.
    // The inherited distribution scatters each row across lanes so load_tile can
    // assemble a register tile; TDM builds no register tile and just needs the
    // box walked in order.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        constexpr index_t kSeq     = Problem::BlockFmhaShape::kN0;
        constexpr index_t kHeaddim = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t warpNum  = Problem::BlockFmhaShape::NumWarps;

        static_assert(kSeq % warpNum == 0,
                      "K kN0 must be divisible by the warp count for a tile-major K dist");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<warpNum, kSeq / warpNum>,
                                             sequence<kHeaddim>>,
                                       tuple<sequence<1>>,
                                       tuple<sequence<0>>,
                                       sequence<1, 2>,
                                       sequence<1, 0>>{},
            bool_constant<true>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigK()
    {
        return make_tuple(number<false>{}, number<0>{}, number<0>{});
    }

    // ---- dO: one plain box, dO^T read back by ds_load_tr --------------------
    //
    // Exactly the K story one operand over: dO was materialised twice, the
    // second copy produced by shuffle_tile purely so gemm_1 could read dO^T with
    // a plain load_tile. ds_load_tr16_b128 does it in hardware, so the shuffle,
    // its staging tile and the second copy all go.
    //
    // Unlike K, dO is re-loaded every Q iteration, so the shuffle this removes
    // was in the hot loop.
    //
    // Verified on gfx1250 against the shuffle path as reference: both produce
    // dO^T with zero mismatches over the whole tile.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradLdsBlockDescriptor()
    {
        using OGradDataType          = typename Problem::OGradDataType;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kKPack     = 16 / sizeof(OGradDataType);

        return make_naive_tensor_descriptor(
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
            make_tuple(number<kKPerBlock>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeOGrad()
    {
        return sizeof(typename Problem::OGradDataType) *
               MakeOGradLdsBlockDescriptor<Problem>().get_element_space_size();
    }

    // the shuffled staging area is gone
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeOGradT()
    {
        return 0;
    }

    // Same encoding the base policy builds for gemm_1's B operand, wrapped so
    // that load_tile_transpose fills it.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeOGradTRegSliceBlockDescriptor()
    {
        using BlockGemm = remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;
        using WarpGemm  = typename BlockGemm::WarpGemm;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kVHeaddim;
        // constexpr index_t kNPerBlock = 32;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto dot_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto dot_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            dot_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});
        // CK_PRINT<typename WarpGemm::BWarpDstrEncoding>();
        // CK_PRINT<decltype(dot_block_dstr_encode)>();

        return make_static_tile_distribution(
            typename InputTileDistributionTraits<
                decltype(dot_block_dstr_encode),
                typename Problem::OGradDataType>::TransposedDstrEncode{});
    }

    // ---- Q: one plain box, Q^T read back by ds_load_tr ----------------------
    //
    // Same treatment as K and dO. Q was materialised twice -- once as-is for
    // gemm_0 and once through shuffle_tile into a second LDS copy so gemm_3
    // could read Q^T with a plain load_tile. ds_load_tr16_b128 removes the need
    // for both, and Q's shuffle sits in the Q loop, so it ran once per iteration
    // rather than once per block the way K's did.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        using QDataType              = typename Problem::QDataType;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPack     = 16 / sizeof(QDataType);

        return make_naive_tensor_descriptor(
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
            make_tuple(number<kKPerBlock>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeQ()
    {
        return sizeof(typename Problem::QDataType) *
               MakeQLdsBlockDescriptor<Problem>().get_element_space_size();
    }

    // The shuffled Q copy is gone.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeQT()
    {
        return 0;
    }

    // gemm_3's B operand, wrapped so load_tile_transpose fills it -- the base
    // policy returns the same encoding unwrapped, for a plain read off the
    // pre-shuffled copy.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeQTRegSliceBlockDescriptor()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm3BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm3BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK3;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto qt_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto qt_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            qt_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        return make_static_tile_distribution(
            typename InputTileDistributionTraits<
                decltype(qt_block_dstr_encode),
                typename Problem::QDataType>::TransposedDstrEncode{});
    }

    // ---- V staged into LDS by TDM -------------------------------------------
    //
    // V used to go global -> registers (load_tile) -> LDS (store_tile). TDM
    // writes global -> LDS directly, so the register round-trip disappears.
    //
    // The inherited descriptor (MakeXLdsBlockDescriptor) carries an XOR swizzle.
    // That is fine when store_tile writes it, because the reader shares the same
    // descriptor and the two XORs cancel. TDM does not go through the descriptor
    // at all -- it writes a single plain box -- so the reader's XOR would no
    // longer cancel against anything. V therefore gets a plain row-major
    // descriptor, which is also what the fwd TDM policy uses for V.
    //
    // Element space is unchanged (kN0 * kVHeaddim), so every smem size and
    // offset computed from this descriptor stays put.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsWriteBlockDescriptor()
    {
        using VDataType              = typename Problem::VDataType;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kKPack     = 16 / sizeof(VDataType);

        return make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
            make_tuple(number<kKPerBlock>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
    }

    // V's DRAM-side distribution for TDM. The inherited one is shaped for
    // load_tile, which spreads each row across lanes so a register tile comes
    // out in the gemm's operand order. TDM does not build a register tile at
    // all -- it needs a trivial tile-major walk of the box it is about to write.
    // This mirrors MakeVDramTileDistribution in the fwd TDM policy.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        constexpr index_t kSeq     = Problem::BlockFmhaShape::kN0;        // V rows
        constexpr index_t kHeaddim = Problem::BlockFmhaShape::kVHeaddim;  // V cols
        constexpr index_t warpNum  = Problem::BlockFmhaShape::NumWarps;

        static_assert(kSeq % warpNum == 0,
                      "V kN0 must be divisible by the warp count for a tile-major V dist");

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,                                 // R: nothing replicated
                tuple<sequence<warpNum, kSeq / warpNum>,    // X[0]: rows, split across warps
                      sequence<kHeaddim>>,                  // X[1]: one full row per thread
                tuple<sequence<1>>,                         // PsToRH major
                tuple<sequence<0>>,                         // PsToRH minor
                sequence<1, 2>,                             // YsToD major
                sequence<1, 0>>{},                          // YsToD minor
            bool_constant<true>{});                         // warp-level parallel only
    }

    // TDM LDS padding for V: disabled, mirroring the fwd TDM policy. Enabling it
    // on the writer alone would misalign the reader, which shares the descriptor
    // above. Padding is bank-conflict avoidance, not correctness.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigV()
    {
        return make_tuple(number<false>{}, number<0>{}, number<0>{});
    }

    // GetSmemSizeV lives in the base and would otherwise call the base's
    // descriptor, not the override above. Same value either way, but keep them
    // from drifting apart.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeV()
    {
        return MakeVLdsWriteBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::VDataType);
    }

    // dK accumulator: [kN0, kQKHeaddim] fp32, row major with the pad above.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKGradAccLdsBlockDescriptor()
    {
        constexpr index_t kN0        = Problem::BlockFmhaShape::kN0;
        constexpr index_t kQKHeaddim = Problem::BlockFmhaShape::kQKHeaddim;

        return make_naive_tensor_descriptor(
            make_tuple(number<kN0>{}, number<kQKHeaddim>{}),
            make_tuple(number<kQKHeaddim + kAccLdsPad>{}, number<1>{}),
            number<1>{},
            number<1>{});
    }

    // dV accumulator: [kN0, kVHeaddim] fp32.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVGradAccLdsBlockDescriptor()
    {
        constexpr index_t kN0       = Problem::BlockFmhaShape::kN0;
        constexpr index_t kVHeaddim = Problem::BlockFmhaShape::kVHeaddim;

        return make_naive_tensor_descriptor(
            make_tuple(number<kN0>{}, number<kVHeaddim>{}),
            make_tuple(number<kVHeaddim + kAccLdsPad>{}, number<1>{}),
            number<1>{},
            number<1>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeKGradAcc()
    {
        return sizeof(typename Problem::AccDataType) *
               MakeKGradAccLdsBlockDescriptor<Problem>().get_element_space_size();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeVGradAcc()
    {
        return sizeof(typename Problem::AccDataType) *
               MakeVGradAccLdsBlockDescriptor<Problem>().get_element_space_size();
    }

    // Offset of the accumulator block inside the workgroup's smem.
    //
    // The staged regions (K/KT, V, and the Q/dO/LSE/D/dS set) alias each other
    // and are sized as a max over phases, because each is dead by the time the
    // next phase starts. The accumulators are live for the whole Q loop, so
    // they cannot join that max -- they sit after it, and every existing offset
    // in the pipeline stays exactly as it was.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeStaged()
    {
        return BlockFmhaBwdPipelineDefaultPolicy::template GetSmemSize<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetKGradAccSmemOffset()
    {
        return GetSmemSizeStaged<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVGradAccSmemOffset()
    {
        return GetSmemSizeStaged<Problem>() + GetSmemSizeKGradAcc<Problem>();
    }

    // V gets its own region rather than aliasing K/KT.
    //
    // In the staged layout V and K/KT share offset 0 and are separated only by
    // time: V could not be written until K and KT had been read back out, which
    // is why V used to be parked in registers first. That ordering constraint is
    // fatal for TDM -- issuing the load late and waiting on TENSORcnt right
    // afterwards exposes the whole global->LDS latency (measured 0.39 TFLOPS
    // against a 48.65 baseline), and issuing it early lands the V box on top of
    // K. Giving V its own kN0*kVHeaddim*sizeof(V) bytes lets the TDM issue sit at
    // the top of the prologue and the TENSORcnt wait sit just before the first
    // read, so the transfer overlaps the K staging that follows it.
    //
    // Cost at headdim 128 is 16 KiB on top of ~102 KiB, well inside the 320 KiB
    // a gfx1250 workgroup may take, and not enough to change occupancy.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVSmemOffset()
    {
        return GetSmemSizeStaged<Problem>() + GetSmemSizeKGradAcc<Problem>() +
               GetSmemSizeVGradAcc<Problem>();
    }

    // Q and dO DRAM distributions for TDM: trivial tile-major, same reasoning as
    // K and V. The inherited ones scatter each row across lanes so load_tile can
    // assemble a register tile; TDM builds no register tile.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeQDramTileDistribution()
    {
        constexpr index_t kRows    = Problem::BlockFmhaShape::kM0;
        constexpr index_t kCols    = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t warpNum  = Problem::BlockFmhaShape::NumWarps;
        static_assert(kRows % warpNum == 0, "kM0 must divide by the warp count");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<warpNum, kRows / warpNum>,
                                             sequence<kCols>>,
                                       tuple<sequence<1>>,
                                       tuple<sequence<0>>,
                                       sequence<1, 2>,
                                       sequence<1, 0>>{},
            bool_constant<true>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeOGradDramTileDistribution()
    {
        constexpr index_t kRows    = Problem::BlockFmhaShape::kM0;
        constexpr index_t kCols    = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t warpNum  = Problem::BlockFmhaShape::NumWarps;
        static_assert(kRows % warpNum == 0, "kM0 must divide by the warp count");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<warpNum, kRows / warpNum>,
                                             sequence<kCols>>,
                                       tuple<sequence<1>>,
                                       tuple<sequence<0>>,
                                       sequence<1, 2>,
                                       sequence<1, 0>>{},
            bool_constant<true>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigQ()
    {
        return make_tuple(number<false>{}, number<0>{}, number<0>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigOGrad()
    {
        return make_tuple(number<false>{}, number<0>{}, number<0>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return GetSmemSizeStaged<Problem>() + GetSmemSizeKGradAcc<Problem>() +
               GetSmemSizeVGradAcc<Problem>() + GetSmemSizeV<Problem>();
    }
};

} // namespace ck_tile
