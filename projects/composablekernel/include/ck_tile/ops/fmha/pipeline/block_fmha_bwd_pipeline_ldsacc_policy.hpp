// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"

namespace ck_tile {

#ifndef CK_TILE_FMHA_BWD_PREFETCH_QDO
#define CK_TILE_FMHA_BWD_PREFETCH_QDO 1
#endif

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

    // Padding, in elements, added to the leading dimension of the four operand
    // boxes that TDM writes and ds_load reads back.
    //
    // At headdim 128 with bf16 a row is 128 * 2 / 4 = 64 dwords, and gfx1250 has
    // 64 LDS banks, so an unpadded row stride puts *every* row on bank 0 and the
    // reader conflicts across the full width. One 128-bit unit of pad per row
    // (8 elements = 16 B = 4 dwords) makes the stride 68 dwords, spreading
    // consecutive rows over 64 / gcd(68, 64) = 16 banks, 2 lanes each -- the
    // floor for a wave32 ds_read_b128, which moves 32 * 4 = 128 dwords -- and a
    // whole number of 16 B units keeps the row start aligned for ds_read_b128 /
    // ds_load_tr16_b128.
    //
    // This is the same failure kAccLdsPad fixes for the accumulators, arrived at
    // from the other direction: there the conflict comes from the C fragment's
    // lane mapping, here from the row stride landing on a multiple of the bank
    // count. The value is derived the way the gemm pipeline derives it (see
    // GetLdsPaddingConfig in gemm_universal_pipeline_ag_bg_cr_policy.hpp).
    //
    // XOR swizzling is not an option for these four: TDM writes a plain box
    // without going through the descriptor, so a reader-side XOR would have
    // nothing to cancel against. dS, which IS written through its descriptor by
    // store_tile, keeps its XOR -- measured better there than padding.
    static constexpr index_t kOperandLdsPad = 8;

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
            make_tuple(number<kKPerBlock + kOperandLdsPad>{}, number<1>{}),
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

    // LSE/D are 1-D over seqlen_q and, after this change, reach LDS by TDM just
    // like K/V/Q/dO.  The inherited MakeLSEDDramTileDistribution scatters kM0
    // across the lanes of a warp so load_tile can assemble a register tile; its
    // ys length is therefore kM0/warp_size = 2, which as a TDM box would be 8 B.
    // TDM builds no register tile and just needs the box walked in order, so
    // partition by warp only (IsWarpLevelParallelOnly) and let ys span the whole
    // per-warp run.  This mirrors MakeKDramTileDistribution one rank down.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeLSEDDramTdmDistribution()
    {
        constexpr index_t kSeq    = Problem::BlockFmhaShape::kM0;
        constexpr index_t warpNum = Problem::BlockFmhaShape::NumWarps;

        static_assert(kSeq % warpNum == 0,
                      "LSE/D kM0 must be divisible by the warp count for a tile-major dist");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<warpNum, kSeq / warpNum>>,
                                       tuple<sequence<1>>,
                                       tuple<sequence<0>>,
                                       sequence<1>,
                                       sequence<1>>{},
            bool_constant<true>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigK()
    {
        // TDM encodes pad_amount as (dwords of padding - 1) and pad_interval as
        // (log2 of the dwords written between pads - 1). One row is exactly one
        // interval here, so each row gets kOperandLdsPad elements appended --
        // matching the descriptor stride above. The two MUST agree: TDM writes
        // the box, the descriptor reads it, and nothing checks them against
        // each other.
        using DT                         = typename Problem::KDataType;
        constexpr index_t kKPerBlock     = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kBytesPerDword = 4;

        constexpr auto log2_floor = [](index_t x) constexpr {
            index_t r = 0;
            while(x > 1)
            {
                x >>= 1;
                r++;
            }
            return r;
        };

        constexpr index_t pad_dwords = kOperandLdsPad * sizeof(DT) / kBytesPerDword;
        constexpr index_t row_dwords = kKPerBlock * sizeof(DT) / kBytesPerDword;
        static_assert(pad_dwords * kBytesPerDword == kOperandLdsPad * sizeof(DT),
                      "operand LDS pad must be a whole number of dwords");
        static_assert(pad_dwords >= 1, "operand LDS pad must be at least one dword");

        return make_tuple(number<true>{},
                          number<pad_dwords - 1>{},
                          number<log2_floor(row_dwords) - 1>{});
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
            make_tuple(number<kKPerBlock + kOperandLdsPad>{}, number<1>{}),
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
            make_tuple(number<kKPerBlock + kOperandLdsPad>{}, number<1>{}),
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

    // dS box transposed to [kN0][kM0], for THIS pipeline only.
    //
    // v_wmma_f32_16x16x32_bf16 hands each lane 8 C values down a *column* (the M
    // direction). With the box N-contiguous those 8 land in 8 different rows,
    // kN0*sizeof(bf16) = 256 B apart, so the compiler cannot merge them -- 64
    // ds_store_b16/_d16_hi per thread per iteration. M-contiguous instead and the
    // same run collapses to ds_store_b128.
    //
    // This MUST stay an override: the base descriptor is shared by five other bwd
    // pipelines, and only this one reads dS back through load_tile_transpose. A
    // transposed box with a plain reader silently produces wrong dQ -- caught on
    // d=32/64/100/127, which route through the register-resident pipeline.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSGradLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPack     = GetSmemKPackSGrad<Problem>();

        return MakeXLdsBlockDescriptor<kMPerBlock, kKPerBlock, kKPack>();
    }

    // gemm_4's A operand, wrapped so load_tile_transpose fills it -- the same
    // trick the Q^T reader above uses.  Route (b): dS is stored into an
    // M-contiguous [kN0][kM0] box so gemm_2's C fragment (8 values down a
    // column) lands contiguously and the store collapses to ds_store_b128;
    // gemm_4 then reads it back through ds_load_tr16_b128.
    //
    // kKPerBlock is kN0, not kK4: a kK4-wide window cannot be transposed here
    // because kK4 == WarpGemm::kK, so KIterPerWarp collapses to 1 and the
    // encoding comes out a lane-mapping factor short (is_sequence_suffix
    // underflows on <2,1> against the required <2,2,1>).  The whole box is read
    // once and sliced in registers instead, exactly as kt_reg_tensor is.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeSGradRegSliceBlockDescriptor()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetSGradKTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<1>{});

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto ds_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto ds_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            ds_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        return make_static_tile_distribution(
            typename InputTileDistributionTraits<
                decltype(ds_block_dstr_encode),
                typename Problem::GemmDataType>::TransposedDstrEncode{});
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
            make_tuple(number<kKPerBlock + kOperandLdsPad>{}, number<1>{}),
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
        // TDM encodes pad_amount as (dwords of padding - 1) and pad_interval as
        // (log2 of the dwords written between pads - 1). One row is exactly one
        // interval here, so each row gets kOperandLdsPad elements appended --
        // matching the descriptor stride above. The two MUST agree: TDM writes
        // the box, the descriptor reads it, and nothing checks them against
        // each other.
        using DT                         = typename Problem::VDataType;
        constexpr index_t kKPerBlock     = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kBytesPerDword = 4;

        constexpr auto log2_floor = [](index_t x) constexpr {
            index_t r = 0;
            while(x > 1)
            {
                x >>= 1;
                r++;
            }
            return r;
        };

        constexpr index_t pad_dwords = kOperandLdsPad * sizeof(DT) / kBytesPerDword;
        constexpr index_t row_dwords = kKPerBlock * sizeof(DT) / kBytesPerDword;
        static_assert(pad_dwords * kBytesPerDword == kOperandLdsPad * sizeof(DT),
                      "operand LDS pad must be a whole number of dwords");
        static_assert(pad_dwords >= 1, "operand LDS pad must be at least one dword");

        return make_tuple(number<true>{},
                          number<pad_dwords - 1>{},
                          number<log2_floor(row_dwords) - 1>{});
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
        // Recomputed from THIS policy's sizes rather than delegating.
        //
        // The base's GetSmemSize resolves GetSmemSizeK/V/Q/OGrad *inside the
        // base class*, so it cannot see the padded descriptors above. Delegating
        // to it would undersize the region and let the padded boxes overrun
        // whatever follows them, with nothing reporting an error. The internal
        // offsets are fine either way -- the pipeline places LSE/D/dS through
        // GetStagedTailOffset, which does use the derived sizes.
        using Base = BlockFmhaBwdPipelineDefaultPolicy;

        constexpr index_t stage0_0 = GetSmemSizeK<Problem>() + GetSmemSizeKT<Problem>();
        constexpr index_t stage0_1 = GetSmemSizeV<Problem>();
        // Q^T and dO^T are read out of the Q and dO boxes, so this policy
        // reports 0 for them and there is nothing to reserve. Taking the base's
        // 16,384 each instead -- as this did originally -- reserved 32,768 B
        // that the layout never addresses.
        constexpr index_t stage1 =
            GetSmemSizeQT<Problem>() + GetSmemSizeQ<Problem>() +
            GetSmemSizeOGradT<Problem>() + GetSmemSizeOGrad<Problem>() +
            Base::template GetSmemSizeLSE<Problem>() + Base::template GetSmemSizeD<Problem>() +
            max(Base::template GetSmemSizeBias<Problem>(),
                Base::template GetSmemSizeSGrad<Problem>());

        constexpr index_t total = max(stage0_0, stage0_1, stage1);
        // The old assert compared against the base policy's total, which does
        // not describe the layout actually built here; it passed for the wrong
        // reason and would now fail for the wrong reason too. Check the thing
        // that binds: every stage must fit in the region we hand out.
        static_assert(total >= stage0_0 && total >= stage0_1 && total >= stage1,
                      "staged region must cover every stage");
        return total;
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
        // TDM encodes pad_amount as (dwords of padding - 1) and pad_interval as
        // (log2 of the dwords written between pads - 1). One row is exactly one
        // interval here, so each row gets kOperandLdsPad elements appended --
        // matching the descriptor stride above. The two MUST agree: TDM writes
        // the box, the descriptor reads it, and nothing checks them against
        // each other.
        using DT                         = typename Problem::QDataType;
        constexpr index_t kKPerBlock     = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kBytesPerDword = 4;

        constexpr auto log2_floor = [](index_t x) constexpr {
            index_t r = 0;
            while(x > 1)
            {
                x >>= 1;
                r++;
            }
            return r;
        };

        constexpr index_t pad_dwords = kOperandLdsPad * sizeof(DT) / kBytesPerDword;
        constexpr index_t row_dwords = kKPerBlock * sizeof(DT) / kBytesPerDword;
        static_assert(pad_dwords * kBytesPerDword == kOperandLdsPad * sizeof(DT),
                      "operand LDS pad must be a whole number of dwords");
        static_assert(pad_dwords >= 1, "operand LDS pad must be at least one dword");

        return make_tuple(number<true>{},
                          number<pad_dwords - 1>{},
                          number<log2_floor(row_dwords) - 1>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigOGrad()
    {
        // TDM encodes pad_amount as (dwords of padding - 1) and pad_interval as
        // (log2 of the dwords written between pads - 1). One row is exactly one
        // interval here, so each row gets kOperandLdsPad elements appended --
        // matching the descriptor stride above. The two MUST agree: TDM writes
        // the box, the descriptor reads it, and nothing checks them against
        // each other.
        using DT                         = typename Problem::OGradDataType;
        constexpr index_t kKPerBlock     = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kBytesPerDword = 4;

        constexpr auto log2_floor = [](index_t x) constexpr {
            index_t r = 0;
            while(x > 1)
            {
                x >>= 1;
                r++;
            }
            return r;
        };

        constexpr index_t pad_dwords = kOperandLdsPad * sizeof(DT) / kBytesPerDword;
        constexpr index_t row_dwords = kKPerBlock * sizeof(DT) / kBytesPerDword;
        static_assert(pad_dwords * kBytesPerDword == kOperandLdsPad * sizeof(DT),
                      "operand LDS pad must be a whole number of dwords");
        static_assert(pad_dwords >= 1, "operand LDS pad must be at least one dword");

        return make_tuple(number<true>{},
                          number<pad_dwords - 1>{},
                          number<log2_floor(row_dwords) - 1>{});
    }

    // Double-buffering Q/dO is a trade, not a free win: measured +3.2% on nomask
    // at a low core clock and ~0 at boost, against about -1.3% on causal across
    // four batches on two machines. Causal has the mask VALU to cover the
    // transfer already, so only the unmasked instance takes it -- and only that
    // instance pays the 36,800 B.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseQDOPrefetch()
    {
        return CK_TILE_FMHA_BWD_PREFETCH_QDO && !Problem::FmhaMask::IsMasking;
    }

    // Base of the second Q/dO pair, used when the pipeline double-buffers them.
    // Appended past everything else so the existing layout is byte-identical.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetQPrefetchSmemOffset()
    {
        return GetSmemSizeStaged<Problem>() + GetSmemSizeKGradAcc<Problem>() +
               GetSmemSizeVGradAcc<Problem>() + GetSmemSizeV<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetOGradPrefetchSmemOffset()
    {
        return GetQPrefetchSmemOffset<Problem>() + GetSmemSizeQ<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t single = GetSmemSizeStaged<Problem>() +
                                   GetSmemSizeKGradAcc<Problem>() +
                                   GetSmemSizeVGradAcc<Problem>() + GetSmemSizeV<Problem>();
        return single + (UseQDOPrefetch<Problem>()
                             ? GetSmemSizeQ<Problem>() + GetSmemSizeOGrad<Problem>()
                             : 0);
    }
};

} // namespace ck_tile
