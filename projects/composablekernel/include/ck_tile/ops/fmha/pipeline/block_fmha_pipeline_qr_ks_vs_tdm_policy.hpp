// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2.hpp"

// can remove all bank conflicts, but drop the performance for some cases
// Probably it is limited by compiler optimization.
#define CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD 0
namespace ck_tile {
// This pipeline is qkv all located in LDS, targeting gfx1250
struct BlockFmhaPipelineQRKSVSTdmDefaultPolicy
    : BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                          /* AsyncCopy = */ true,
                                          /* NumPrefetchK = */ 1,
                                          /* NumPrefetchV = */ 1>
{
    using BasePolicy = BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                                           /* AsyncCopy = */ true,
                                                           /* NumPrefetchK = */ 1,
                                                           /* NumPrefetchV = */ 1>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::QDataType);

        // this should align with MakeQDramTileDistribution()
        constexpr index_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        return min(ElemPerThread, MaxVectorSize);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentOacc()
    {
        using OaccDataType = remove_cvref_t<typename Problem::OaccDataType>;

        return static_cast<index_t>(16 / sizeof(OaccDataType));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK()
    {
        // gfx1250 wave32 + d>=128 needs KVector >= 4 to satisfy
        // base async distribution static_assert (WarpSize*KVector >= kKPerBlock).
        // Choose b128 (dwordx4) as starting point; b64 fallback if perf problems.
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
#if defined(__gfx125__)
        constexpr index_t MaxLoadSizeInBytes = 16; // dwordx4 = b128
#else
        constexpr index_t MaxLoadSizeInBytes = 4; // dword (matches base async default)
#endif
        return MaxLoadSizeInBytes * numeric_traits<KDataType>::PackedSize / sizeof(KDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV()
    {
        // Same rationale as GetAlignmentK; V is loaded with async copy too.
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
#if defined(__gfx125__)
        constexpr index_t MaxLoadSizeInBytes = 16; // dwordx4 = b128
#else
        constexpr index_t MaxLoadSizeInBytes = 4; // dword
#endif
        return MaxLoadSizeInBytes * numeric_traits<VDataType>::PackedSize / sizeof(VDataType);
    }

    // Step B2 (λ-Q) — trivial tile-major Q dram dist, mirrors (λ-K) at L168-190.
    // Reviewer trace finding: Q LDS dump shows tid 1 全 -23.203 sentinel
    // (uninit fp16 LDS) = TDM box doesn't cover plain row-major LDS
    // because old 5D async-style dist was tailored for per-thread scatter
    // writes (async_load_tile design). Under TDM box-major write, that
    // mismatches the row-major Q LDS desc, leaving sentinel patches.
    //
    // The trivial tile-major form (mirror K λ) makes each thread's
    // per-call footprint exactly one contiguous (kMPerBlock/warpNum × kKPerBlock)
    // tile, so:
    //   * box-major TDM write lands at the row-major LDS strip the reader
    //     expects;
    //   * BlockGemm 0 (QK GEMM, Q is A operand) reg dist remains unchanged.
    //
    // Combined with (β')-Q-fix (reader Xor=false plain) — full K-side recipe
    // applied to Q. See B2 reviewer pivot post-(β')-Q-fail report for context.
    //
    // Keep the `BypassLDS` template param to preserve existing call sites
    // that bypass LDS (not used by TDM path but kept for symmetry).
    template <typename Problem, bool BypassLDS = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQDramTileDistribution()
    {
        if constexpr(!BypassLDS)
        {
            constexpr index_t kBlockSize = Problem::kBlockSize;
            constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;
            constexpr index_t warpNum    = kBlockSize / get_warp_size();

            static_assert(kMPerBlock % warpNum == 0,
                          "kMPerBlock must be divisible by warpNum for trivial tile-major Q dist");

            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,                                    // R: empty
                    tuple<sequence<warpNum, kMPerBlock / warpNum>, // X[0]: M-axis, warp split
                          sequence<kKPerBlock>>, // X[1]: K-axis, single full vector
                    tuple<sequence<1>>,          // PsToRH (warp dim mapping)
                    tuple<sequence<0>>,          // PsToRH_lid
                    sequence<1, 2>,              // YsToD outer
                    sequence<1, 0>>{},           // YsToD inner
                bool_constant<true>{});          // IsWarpLevelParallelOnly
        }
        else
        {
            using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
            constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

            constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<0>{});
            constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{});

            constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

            constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
            constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

            constexpr auto q_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<NWarp>,
                tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                tuple<sequence<1, 0>>,
                tuple<sequence<1, 0>>,
                sequence<2, 1>,
                sequence<0, 0>>{};

            constexpr auto q_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                q_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

            constexpr auto q_block_dstr = make_static_tile_distribution(q_block_dstr_encode);

            return q_block_dstr;
        }
    }

    // Step B2 (λ-1) — trivial tile-major K dram dist, mirrors GEMM v1
    // ColMajor-B `MakeBDramTileDistribution`
    // (gemm_pipeline_ag_bg_cr_comp_tdm_default_policy.hpp:117-126).
    //
    // Why this is needed: TDM hardware writes to LDS in **box-major order**
    // (each thread writes a contiguous box of shape `box_dim` starting at
    // `lds_coord`). Our K LDS descriptor is naive 2D (kN0, kK0) row-major
    // (MakeKLdsBlockDescriptor). The ds_load reader (MakeKRegTileDistribution)
    // is built from `WarpGemm::BWarpDstrEncoding` and assumes that
    // contiguous-row interface.
    //
    // The B1 distribution was tailored for `async_load_tile` (per-thread
    // scatter writes), so each thread's per-call footprint was *not* a
    // contiguous LDS row. Under TDM, that mismatched the box-major writer
    // and the row-major reader, producing 100%-wrong but deterministic
    // garbage (B2 H3 root cause).
    //
    // The trivial tile-major form below makes thread-i's per-call footprint
    // exactly one contiguous (kN0/warpNum × kK0) tile, so:
    //   * box-major TDM write lands at the row-major LDS strip the reader
    //     expects;
    //   * BlockGemm 0 (QK GEMM, K is B operand) reg dist remains unchanged.
    //
    // Keep the `LoadOnce` template param to preserve the existing
    // `MakeKDramTileDistribution<Problem, true>` call site (which uses
    // kSubQKHeaddim instead of kK0).
    template <typename Problem, bool LoadOnce = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock =
            LoadOnce ? Problem::BlockFmhaShape::kSubQKHeaddim : Problem::BlockFmhaShape::kK0;
        constexpr index_t warpNum = kBlockSize / get_warp_size();

        static_assert(kNPerBlock % warpNum == 0,
                      "kNPerBlock must be divisible by warpNum for trivial tile-major K dist");

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,                                    // R: empty
                tuple<sequence<warpNum, kNPerBlock / warpNum>, // X[0]: N-axis, warp split
                      sequence<kKPerBlock>>, // X[1]: K-axis, single full vector per thread
                tuple<sequence<1>>,          // PsToRH (warp dim mapping)
                tuple<sequence<0>>,          // PsToRH_lid
                sequence<1, 2>,              // YsToD outer
                sequence<1, 0>>{},           // YsToD inner
            bool_constant<true>{});          // IsWarpLevelParallelOnly
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQRegTileDistribution()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{});

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        // Read M first, then K
        // This is the same data consume order as BlockGEMM
        constexpr auto q_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto q_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            q_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        constexpr auto q_block_dstr = make_static_tile_distribution(q_block_dstr_encode);

        return q_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackQ()
    {
        // TODO: this is for 3d layout
        using QDataType = remove_cvref_t<typename Problem::QDataType>;
        return static_cast<index_t>(16 / sizeof(QDataType));
    }

    // Step B2 (β')-Q + Task A cleanup extension: Q LDS desc `Xor` template
    // parameter removed. Same reasoning as K desc cleanup (TDM box-major
    // write doesn't support swizzle, so XOR'd Q LDS layout was unreachable
    // from TDM path, the only Q writer in this tdm pipeline). Old
    // `if constexpr(Xor) { ... } else { ... }` branch had only the `else`
    // (plain row-major) executed by TDM-compatible call sites; XOR'd branch
    // was dead code that caused (β')-Q-fix bug (smoking gun: B2 Q dump
    // 0/32 thread match B1 + tid 1 全 sentinel-like -23.203 = uninit LDS
    // pattern, because Xor=true reader expected XOR'd bytes that TDM
    // writer never wrote). Fix = align reader with writer (both plain).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

        constexpr index_t kKPack = GetSmemKPackQ<Problem>();

        return make_naive_tensor_descriptor(make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                                            make_tuple(number<kKPerBlock>{}, number<1>{}),
                                            number<kKPack>{},
                                            number<1>{});
    }

    // Step B2 cleanup (Task A): K LDS desc `Xor` template parameter removed.
    // Rationale: TDM box-major write doesn't support swizzle, so XOR'd K LDS
    // layout was unreachable from TDM path (the only K writer in this tdm
    // pipeline). Old `if constexpr(Xor) { ... } else { ... }` branch had only
    // the `else` (plain row-major) executed; XOR'd branch was dead code.
    // Removing the param cuts ~80 lines of unreachable code and simplifies
    // the call site signature for readability. Runtime equivalent.
    template <typename Problem, bool LoadOnce = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock =
            LoadOnce ? Problem::BlockFmhaShape::kSubQKHeaddim : Problem::BlockFmhaShape::kK0;

        constexpr index_t kKPack = GetSmemKPackK<Problem>();

        return make_naive_tensor_descriptor(make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                                            make_tuple(number<kKPerBlock>{}, number<1>{}),
                                            number<kKPack>{},
                                            number<1>{});
    }

    template <typename Problem, bool Xor = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t kKPack = GetSmemKPackV<Problem>();

        constexpr auto v_lds_block_desc = [&]() {
            if constexpr(Xor)
            {
                constexpr auto XorGroupSize =
                    Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{});

#if CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD
                constexpr auto LDSLayerSize  = 256 / sizeof(typename Problem::VDataType);
                constexpr auto XorLengthFold = LDSLayerSize / kNPerBlock;

                if constexpr(XorLengthFold > 1)
                {
                    constexpr auto v_lds_block_desc_naive = make_naive_tensor_descriptor(
                        make_tuple(number<kKPerBlock / XorLengthFold>{},
                                   number<LDSLayerSize / XorGroupSize>{},
                                   number<XorGroupSize>{}),
                        make_tuple(number<LDSLayerSize>{}, number<XorGroupSize>{}, number<1>{}),
                        number<kKPack>{},
                        number<1>{});

                    constexpr auto v_lds_block_desc_permuted = transform_tensor_descriptor(
                        v_lds_block_desc_naive,
                        make_tuple(
                            make_xor_transform(make_tuple(number<kKPerBlock / XorLengthFold>{},
                                                          number<LDSLayerSize / XorGroupSize>{})),
                            make_pass_through_transform(number<XorGroupSize>{})),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}));

                    constexpr auto v_lds_block_desc_tmp = transform_tensor_descriptor(
                        v_lds_block_desc_permuted,
                        make_tuple(
                            make_pass_through_transform(number<kKPerBlock / XorLengthFold>{}),
                            make_unmerge_transform(make_tuple(number<XorLengthFold>{},
                                                              number<kNPerBlock / XorGroupSize>{})),
                            make_pass_through_transform(number<XorGroupSize>{})),
                        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                        make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

                    return transform_tensor_descriptor(
                        v_lds_block_desc_tmp,
                        make_tuple(
                            make_merge_transform_v3_division_mod(make_tuple(
                                number<kKPerBlock / XorLengthFold>{}, number<XorLengthFold>{})),
                            make_merge_transform_v3_division_mod(make_tuple(
                                number<kNPerBlock / XorGroupSize>{}, number<XorGroupSize>{}))),
                        make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                        make_tuple(sequence<0>{}, sequence<1>{}));
                }
                else
#endif // CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD
                {
                    constexpr auto v_lds_block_desc_naive = make_naive_tensor_descriptor(
                        make_tuple(number<kKPerBlock>{},
                                   number<kNPerBlock / XorGroupSize>{},
                                   number<XorGroupSize>{}),
                        make_tuple(number<kNPerBlock>{}, number<XorGroupSize>{}, number<1>{}),
                        number<kKPack>{},
                        number<1>{});

                    constexpr auto v_lds_block_desc_permuted = transform_tensor_descriptor(
                        v_lds_block_desc_naive,
                        make_tuple(make_xor_transform(make_tuple(
                                       number<kKPerBlock>{}, number<kNPerBlock / XorGroupSize>{})),
                                   make_pass_through_transform(number<XorGroupSize>{})),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}));

                    return transform_tensor_descriptor(
                        v_lds_block_desc_permuted,
                        make_tuple(
                            make_pass_through_transform(number<kKPerBlock>{}),
                            make_merge_transform_v3_division_mod(make_tuple(
                                number<kNPerBlock / XorGroupSize>{}, number<XorGroupSize>{}))),
                        make_tuple(sequence<0>{}, sequence<1, 2>{}),
                        make_tuple(sequence<0>{}, sequence<1>{}));
                }
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(number<kKPerBlock>{}, number<kNPerBlock>{}),
                    make_tuple(number<kNPerBlock>{}, number<1>{}),
                    number<kKPack>{},
                    number<1>{});
            }
        }();

        return v_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::QDataType,
                             typename Problem::KDataType,
                             typename Problem::SaccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN0,
                                                    Problem::BlockFmhaShape::kK0>,
                                           typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

        using WarpGemm = WarpGemmDispatcher<typename Problem::QDataType,
                                            typename Problem::KDataType,
                                            typename Problem::SaccDataType,
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}),
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}),
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}),
                                            true>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV2CustomPolicy<typename Problem::QDataType,
                                                typename Problem::KDataType,
                                                typename Problem::SaccDataType,
                                                typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                WarpGemm,
                                                GemmLoopOrder::MNK>;

        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetPVBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::PDataType,
                             typename Problem::VDataType,
                             typename Problem::OaccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN1,
                                                    Problem::BlockFmhaShape::kK1>,
                                           typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm1WarpTile>>;

        using WarpGemm =
            WarpGemmDispatcher<typename Problem::PDataType,
                               typename Problem::VDataType,
                               typename Problem::OaccDataType,
                               Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}),
                               Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}),
                               Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}),
                               true,
                               false,
                               false,
                               ((Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}) == 16 &&
                                 Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}) == 32) ||
                                (Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}) == 32 &&
                                 Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}) == 16))
                                   ? WGAttrNumAccessEnum::Double
                                   : WGAttrNumAccessEnum::Single>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV2CustomPolicy<typename Problem::PDataType,
                                                typename Problem::VDataType,
                                                typename Problem::OaccDataType,
                                                typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                WarpGemm,
                                                GemmLoopOrder::KMN>;

        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKRegTileDistribution()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        // Read N first, then K
        // This is the same data consume order as BlockGEMM
        constexpr auto k_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto k_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            k_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        constexpr auto k_block_dstr = make_static_tile_distribution(k_block_dstr_encode);

        return k_block_dstr;
    }

    // Step B2 (h hybrid): V goes back to async_load + ds_load_tr (B1 path).
    // V dram dist reverted to B1 5D (mirrors
    // block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp:615-641).
    // The B2 (λ-2) trivial tile-major attempt produced element-permutation
    // garbage because TDM box-major write writes plain row-major LDS but
    // ds_load_tr_b128 expects a thread-permuted layout (5D dist is the
    // reverse-engineered match). TDM box-major + thread-permuted layout are
    // physically incompatible under simple dist changes — see (h) decision
    // doc swe-status-lambda-pivot.md.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::VDataType);

        constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        constexpr index_t kMaxVecLoad = min(ElemPerThread, MaxVectorSize);

        constexpr index_t NPerThread     = kMaxVecLoad;
        constexpr index_t NThreads       = kNPerBlock / NPerThread;
        constexpr index_t KThreadPerWarp = get_warp_size() / NThreads;
        constexpr index_t NumWarps       = kBlockSize / get_warp_size();
        constexpr index_t KPerThread     = kKPerBlock / (KThreadPerWarp * NumWarps);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<KPerThread, NumWarps, KThreadPerWarp>,
                                             sequence<NThreads, NPerThread>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakePRegTileDistribution()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<1>{});

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        // Read M first, then K
        // This is the same data consume order as BlockGEMM
        constexpr auto p_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<2, 1>,
                                       sequence<0, 0>>{};

        constexpr auto p_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            p_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        constexpr auto p_block_dstr = make_static_tile_distribution(p_block_dstr_encode);

        return p_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVRegTileDistribution()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        // Read N first, then K
        // This is the same data consume order as BlockGEMM
        constexpr auto v_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<2, 1>,
                                       sequence<0, 0>>{};

        constexpr auto v_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            v_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        constexpr auto v_block_dstr =
            make_static_tile_distribution(typename InputTileDistributionTraits<
                                          decltype(v_block_dstr_encode),
                                          typename Problem::VDataType>::TransposedDstrEncode{});

        return v_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemNPackS()
    {
        using SDataType = remove_cvref_t<typename Problem::SaccDataType>;
        return static_cast<index_t>(16 / sizeof(SDataType));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kNPack     = GetSmemNPackS<Problem>();

        constexpr auto s_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock / kNPack>{}, number<kMPerBlock>{}, number<kNPack>{}),
            make_tuple(number<(kMPerBlock + 1) * kNPack>{}, number<kNPack>{}, number<1>{}),
            number<kNPack>{},
            number<1>{});

        constexpr auto s_lds_block_desc = transform_tensor_descriptor(
            s_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(number<kMPerBlock>{}),
                make_merge_transform(make_tuple(number<kNPerBlock / kNPack>{}, number<kNPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return s_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetKVBlockGemm<Problem>())>;

        constexpr auto config   = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG                = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kTileK     = Problem::BlockFmhaShape::kN0;

        // K2 is equal to Impl::kABKPerLane * kKIterPerWarpGemm
        constexpr index_t K3 = WG::kK / WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K2 = WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K1 = kKPerBlock / (K2 * K3);
        constexpr index_t K0 = kTileK / kKPerBlock;
        constexpr index_t M2 = WG::WarpGemmAttribute::Impl::kAMLane;
        constexpr index_t M1 = MWarp;
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        constexpr auto s2_block_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1, K2, K3>>,
                                       tuple<sequence<1, 0>, sequence<2, 1>>,
                                       tuple<sequence<1, 0>, sequence<2, 2>>,
                                       sequence<1, 2, 2, 2>,
                                       sequence<0, 0, 1, 3>>{};

        constexpr auto s2_block_dstr = make_static_tile_distribution(s2_block_dstr_encoding);

        return s2_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeQ()
    {
        return MakeQLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::QDataType);
    }

    template <typename Problem, bool LoadOnce = false>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeK()
    {
        return MakeKLdsBlockDescriptor<Problem, LoadOnce>().get_element_space_size() *
               sizeof(typename Problem::KDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeV()
    {
        return MakeVLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::VDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeS()
    {
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{});

        return NWarp > 1 ? MakeSLdsBlockDescriptor<Problem>().get_element_space_size() *
                               sizeof(typename Problem::SaccDataType)
                         : 0;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        constexpr ck_tile::index_t kM0 = Problem::BlockFmhaShape::kM0;
        if constexpr(kM0 > 64)
        {
            // Prefill: same layout as qr_async_trload kernel allocations.
            // Two K buffers (ping/pong) + two V buffers (ping/pong).
            return 2 * GetSmemSizeK<Problem>() + 2 * GetSmemSizeV<Problem>();
        }
        else
        {
            // Decode: single buffer; Q, K, S, V laid out sequentially.
            return max(GetSmemSizeQ<Problem>(),
                       GetSmemSizeK<Problem>() + GetSmemSizeS<Problem>() + GetSmemSizeV<Problem>());
        }
    }

    // -------------------------------------------------------------------------
    // TDM LDS padding config (Step B2)
    //
    // Mirrors the formula in
    //   gemm_universal_pipeline_ag_bg_cr_policy.hpp:1131 GetLdsPaddingConfig
    // adapted to fmha shape names. Returns a tuple (IsPadding, PadAmount,
    // PadInterval) suitable for filling TDMConfig::pad_config (see
    // amd_tdm_descriptor.hpp). Field semantics:
    //   * pad_amount:  N -> add (N+1) dwords padding
    //   * pad_interval N -> insert padding every 2^(N+1) dwords
    //
    // Branch selection rationale (do NOT silently change without re-verifying):
    //   * Q : non-tr-load   (output via plain ds_load, like K)  -> "else" branch
    //   * K : non-tr-load   (output via plain ds_load)          -> "else" branch
    //   * V : tr-load       (output via ds_load_tr_b128)        -> "if"   branch
    // -------------------------------------------------------------------------

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigQ()
    {
        // Step B2 (λ-Q) disambig: Q padding temporarily DISABLED.
        // Same rationale as GetLdsPaddingConfigK/V (L781-802): H4' verified
        // swizzle/padding is not the root cause of B2 garbage. K/V padding
        // were disabled in (λ-1)/(λ-2) phases but Q padding was NOT
        // synchronously disabled — historical asymmetry uncovered by
        // mentor's Q4 padding hypothesis + QA Q dump sentinel signal
        // (tid 1 全 -23.203 fp16 uninit).
        //
        // Mechanism: Q TDM writes WITH padding (every pad_interval dwords
        // skip pad_amount dwords) into a Q LDS desc that is plain row-major
        // (no padding) — writer/reader byte-position mismatch + skipped
        // bytes hold sentinel.
        //
        // (λ-Q) trivial tile-major Q dram dist alone improved max_err
        // 0.36→0.12 (3x improvement, tid 0 fully fixed) but tid 1 sentinel
        // remained — confirming padding is independent root cause #2.
        // This Q padding disable mirrors K/V pattern → expects max_err < 0.001.
        //
        // Re-enable after (λ-Q)+padding-disable+ABC pass during perf phase.
        return make_tuple(number<false>{}, number<0>{}, number<0>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigK()
    {
        // Step B2 (λ-1) disambig: K padding temporarily DISABLED.
        // Rationale: H4' already verified swizzle/padding is not the root
        // cause of B2 garbage. Disabling here narrows the variable set for
        // step-1 verify (only K dram dist changes, not padding).
        // Re-enable after (λ-1)+(λ-2) ABC pass. Original implementation
        // mirrors gemm_universal_pipeline_ag_bg_cr_policy.hpp:1131; restore
        // by reverting this hunk.
        return make_tuple(number<false>{}, number<0>{}, number<0>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigV()
    {
        // Step B2 (λ-2) disambig: V padding temporarily DISABLED.
        // Same rationale as GetLdsPaddingConfigK above (H4' verified swizzle/
        // padding is not the root cause of B2 garbage; disabling here narrows
        // the variable set during step-5 verify so only V dram dist + dual
        // view changes are in flight). Re-enable after (λ-1)+(λ-2) ABC pass.
        return make_tuple(number<false>{}, number<0>{}, number<0>{});
    }
};

} // namespace ck_tile
