// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

#include <type_traits>

// Set to 1 by every build rule that compiles a page-tile instance with
// -mllvm -amdgpu-expert-scheduling-mode (one of the scheduled-TDM FMHA flags that
// fmha_batch_prefill_qr_tdm_page_tile_kernel.hpp requires). The page tile was validated and
// measured only in that mode, where the compiler places the VALU dependence waits itself
// (WAVE_SCHED_MODE). An -mllvm option is invisible to the preprocessor, so the build states it with
// this define, and both page-tile pipelines refuse to compile without it.
#ifndef CK_TILE_FMHA_PAGE_TILE_EXPERT_SCHEDULING
#define CK_TILE_FMHA_PAGE_TILE_EXPERT_SCHEDULING 0
#endif

// 1 when the compiler protects the WMMA operand WAR of the scheduled-TDM hazard note
// (block_fmha_pipeline_qr_ks_vs_tdm_sched_policy.hpp) itself in expert scheduling mode: it places a
// va_vdst wait before every load that overwrites a source of a WMMA still in flight. Clang 24 does,
// clang 23 does not. The page tiles then keep the guard's register allocation (the grouped-drain
// placement and the keep-alive uses) but not its own full drains and RAW fences
// (kFmhaBatchPrefillPageTileGuardWaits). The compiler's waits count on WMMAs completing in issue
// order, which a gfx1250 probe agrees with: every older WMMA result was written once va_vdst(n)
// covered it, while a read with no wait saw the old register; and no WMMA read a source that a
// later LDS load overwrote, even with no wait between them.
#ifndef CK_TILE_FMHA_PAGE_TILE_COMPILER_GUARDS_WMMA_OPERANDS
#if __clang_major__ >= 24
#define CK_TILE_FMHA_PAGE_TILE_COMPILER_GUARDS_WMMA_OPERANDS 1
#else
#define CK_TILE_FMHA_PAGE_TILE_COMPILER_GUARDS_WMMA_OPERANDS 0
#endif
#endif

namespace ck_tile {

// Waits until every asynchronous global-to-LDS load of the wave has completed, marked or not. The
// page tiles publish the next tile's page ids with amd_async_global_load_to_lds, which on gfx1250
// issues the DMA as inline assembly without an asyncmark. ck_tile::s_wait_asynccnt becomes
// wait_asyncmark when the compiler has the asyncmark builtins, and that waits for marked DMAs
// only: clang 24 then emitted no wait before the barrier that publishes the ids, so waves read an
// id the DMA had not yet written (a wrong page: launch-to-launch different output, page faults).
// Clang 23 emitted s_wait_asynccnt 0 there. The hardware counter covers every DMA.
CK_TILE_DEVICE void fmha_page_tile_wait_page_id_dma()
{
#if defined(__gfx125__)
    __builtin_amdgcn_s_wait_asynccnt(0);
#else
    s_wait_asynccnt<0>();
#endif
}

// A page-tile barrier that waits for at most lgkmcnt outstanding LDS operations. block_sync_lds
// drops its workgroup LDS fences for a nonzero count, so LDS loads stay in flight across the
// barrier. Fenced keeps the fences (block_sync_lds of every count before that change): the
// release fence drains the LDS loads (s_wait_dscnt 0), so no load result stays live across the
// barrier. Tiles whose register pressure sits at an occupancy limit use it
// (fmha_page_tile_fenced_barriers_v).
template <bool Fenced, index_t lgkmcnt>
CK_TILE_DEVICE void fmha_page_tile_block_sync_lds()
{
    if constexpr(Fenced && lgkmcnt != 0)
    {
        s_waitcnt_lgkm<lgkmcnt>();
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
#if defined(__gfx12__)
        __builtin_amdgcn_s_barrier_signal(-1);
        __builtin_amdgcn_s_barrier_wait(-1);
#else
        __builtin_amdgcn_s_barrier();
#endif
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
    }
    else
    {
        block_sync_lds<lgkmcnt>();
    }
}

// s_wait_tensorcnt_barrier with the barrier of fmha_page_tile_block_sync_lds.
template <bool Fenced, index_t tensorcnt, index_t lgkmcnt = waitcnt_arg::kMaxLgkmCnt>
CK_TILE_DEVICE void fmha_page_tile_tensorcnt_barrier()
{
    s_wait_tensorcnt<tensorcnt>();
    fmha_page_tile_block_sync_lds<Fenced, lgkmcnt>();
}

// Whether the page tiles issue the WMMA operand guard's own waits (full drains, RAW fences): only
// when the compiler does not protect the hazard
// (CK_TILE_FMHA_PAGE_TILE_COMPILER_GUARDS_WMMA_OPERANDS).
inline constexpr bool kFmhaBatchPrefillPageTileGuardWaits =
    !CK_TILE_FMHA_PAGE_TILE_COMPILER_GUARDS_WMMA_OPERANDS;

// Independent page-size / tile-size geometry for gfx1250 batch prefill.
//
// The logical KV tile stays N128 (with M128 or M64 query tiles); only the physical KV allocation
// page changes. This header owns the compile-time page geometry, the per-wave
// owned page metadata and the ordinary linear page-local TDM writer. It does
// not own attention computation, and it deliberately has no runtime page switch,
// no gather path and no shuffled-layout branch.

// Fragments one wave issues per K/V tile: one 32-row box for P>=64, two 16-row boxes for P16.
template <index_t PageSize>
inline constexpr index_t kFmhaBatchPrefillPageFragmentsPerWave = PageSize == 16 ? 2 : 1;

// The page-size gate of the page tile: the pipeline, its wrapper and the kernel instantiate these
// traits. Supported pages satisfy P % N == 0 or N % P == 0 (N = 128): P256/P1024 slice one page,
// P128 maps one page to one tile, P64 two pages to one tile, P16 bundles eight independently
// mapped fragments.
template <index_t PageSize, index_t HeadDim>
struct FmhaBatchPrefillPageTileTraits
{
    static_assert(PageSize == 16 || PageSize == 64 || PageSize == 128 || PageSize == 256 ||
                      PageSize == 1024,
                  "page size must be 16, 64, 128, 256 or 1024");
    static_assert(HeadDim == 64 || HeadDim == 128, "head dim must be 64 or 128");

    static constexpr index_t kTileRows         = 128;
    static constexpr index_t kWaves            = 4;
    static constexpr index_t kRowsPerWave      = 32;
    static constexpr index_t kFragmentsPerWave = kFmhaBatchPrefillPageFragmentsPerWave<PageSize>;
    static constexpr index_t kFragmentRows     = PageSize == 16 ? 16 : 32;
    static constexpr index_t kMetadataBytes    = PageSize == 16 ? 32 : 16;
    // Native tensor issues per wave per K/V load group.
    static constexpr index_t kKPrefetchTensorCount = kFragmentsPerWave;
    static constexpr index_t kVPrefetchTensorCount = kFragmentsPerWave;

    static_assert(PageSize % 128 == 0 || 128 % PageSize == 0,
                  "supported pages satisfy P % N == 0 or N % P == 0");
    static_assert(kWaves * kRowsPerWave == kTileRows);
    static_assert(kFragmentsPerWave * kFragmentRows == kRowsPerWave);
};

// KV tile order of a query tile with sink tokens (StreamingLLM): the loop first visits the sink
// tiles from origin 0 and then jumps to the window start, so tile i starts at 128*i below
// sink_tiles and at resume_origin + 128*(i - sink_tiles) from there. Without sink tokens the
// tiles are contiguous from the first origin.
struct FmhaPageTileSinkTiles
{
    index_t sink_tiles;
    index_t resume_origin;
};

// One owned fragment's page metadata.
struct FmhaBatchPrefillPageFragmentId
{
    index_t physical_page = 0;
    index_t valid_rows    = 0;
};

// Wave-local owned metadata.
//   P>=128: owned[0] carries the one shared physical id and the logical tile's
//           valid rows (0..128).
//   P64   : owned[0] carries the id of the page holding this wave's 32-row slice (waves 0-1
//           the tile's first page, waves 2-3 its second) and the logical tile's valid rows.
//   P16   : owned[r] is fragment f = 2*w + r with valid_rows 0..16.
template <index_t PageSize>
struct FmhaBatchPrefillPageTileIds
{
    static constexpr index_t kFragmentsPerWave = kFmhaBatchPrefillPageFragmentsPerWave<PageSize>;

    array<FmhaBatchPrefillPageFragmentId, kFragmentsPerWave> owned{};
};

namespace detail {

// Per-wave page source distribution (P16 16-row fragments, P64 32-row slices). One warp
// coordinate factor of length 1 keeps the ordinary source coordinate (0,0) in all four waves:
// each wave rebases its own physical page pointer, so no 32*w or 16*w source offset may enter
// the view.
template <index_t Rows, index_t BoxWidth>
CK_TILE_HOST_DEVICE constexpr auto MakePageFragmentDistribution()
{
    return make_static_tile_distribution(
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<1, Rows>, sequence<BoxWidth>>,
                                   tuple<sequence<1>>,
                                   tuple<sequence<0>>,
                                   sequence<1, 2>,
                                   sequence<1, 0>>{},
        bool_constant<true>{});
}

// P>=128 keeps the existing four-wave 32-row split (one page base for the whole tile).
template <index_t BoxWidth>
CK_TILE_HOST_DEVICE constexpr auto MakePageN32RowDistribution()
{
    return make_static_tile_distribution(
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<4, 32>, sequence<BoxWidth>>,
                                   tuple<sequence<1>>,
                                   tuple<sequence<0>>,
                                   sequence<1, 2>,
                                   sequence<1, 0>>{},
        bool_constant<true>{});
}

// Box width of the LDS write view fixes the operand: K is D+8, V is D.
template <typename LdsWindow, index_t HeadDim>
CK_TILE_HOST_DEVICE constexpr index_t CheckBoxWidth()
{
    using LdsLengths            = typename remove_cvref_t<LdsWindow>::WindowLengths;
    constexpr index_t kBoxWidth = LdsLengths{}.at(number<1>{});
    static_assert(kBoxWidth == HeadDim + 8 || kBoxWidth == HeadDim,
                  "page tile writer accepts the K (D+8) or V (D) LDS write view only");
    return kBoxWidth;
}

} // namespace detail

// Ordinary page-local writer. It issues the fixed number of first-class rank-2
// TDM boxes the schedule expects (also for absent fragments, which read no global
// row and rely on the descriptor's out-of-bounds fill) and leaves inserted V
// padding bytes untouched.
template <index_t PageSize, index_t HeadDim>
struct FmhaBatchPrefillLinearPageTileWriter
{
    using Traits = FmhaBatchPrefillPageTileTraits<PageSize, HeadDim>;

    static_assert(Traits::kTileRows == 128);
    static_assert(Traits::kWaves == 4);
    static_assert(Traits::kMetadataBytes % 16 == 0, "pending metadata slot alignment contract");

    static constexpr index_t kWaves            = Traits::kWaves;
    static constexpr index_t kRowsPerWave      = Traits::kRowsPerWave;
    static constexpr index_t kFragmentsPerWave = Traits::kFragmentsPerWave;
    static constexpr index_t kFragmentRows     = Traits::kFragmentRows;
    static constexpr index_t kPageSize         = PageSize;
    static constexpr index_t kHeadDim          = HeadDim;
    // Pages smaller than the tile: every wave rebases its own page (P16 per fragment, P64 per
    // 32-row slice). Whole-tile pages share one base and the four-wave split.
    static constexpr bool kPerWavePage = PageSize < 128;

    // This wave's owned fragment valid rows, in physical page coordinates.
    CK_TILE_HOST_DEVICE static constexpr index_t fragment_valid_rows(
        index_t wave, index_t fragment, const FmhaBatchPrefillPageTileIds<PageSize>& ids)
    {
        if constexpr(PageSize == 16)
        {
            (void)wave;
            return clamp(ids.owned[fragment].valid_rows, index_t{0}, index_t{kFragmentRows});
        }
        else
        {
            (void)fragment;
            return clamp(
                ids.owned[0].valid_rows - kRowsPerWave * wave, index_t{0}, index_t{kRowsPerWave});
        }
    }

    // Destination row inside the logical 128-row K/V LDS tile.
    CK_TILE_HOST_DEVICE static constexpr index_t fragment_destination_row(index_t wave,
                                                                          index_t fragment)
    {
        return kRowsPerWave * wave + kFragmentRows * fragment;
    }

    // WORKGROUP COLLECTIVE: for P16 it contains a workgroup barrier between fragments,
    // so every wave of the workgroup must call it the same number of times, from
    // wave-uniform control flow. Callers assert kWaves against their own geometry
    // (pipeline core) and the kernel pins the block to kWaves wave32 waves, so the
    // early return for wave >= kWaves below is unreachable in supported launches.
    // Requires the gfx125 TDM feature; on any other device target the TDM builtins
    // compile to nothing, so instantiating it there is an error.
    template <typename T,
              index_t Alignment,
              typename TDMConfig,
              typename LdsWindow,
              bool RematerializeStride>
    CK_TILE_DEVICE static void Load(const T* pool_head_ptr,
                                    long_index_t page_stride,
                                    index_t token_stride,
                                    index_t logical_tile_origin,
                                    const FmhaBatchPrefillPageTileIds<PageSize>& ids,
                                    const TDMConfig& config,
                                    LdsWindow&& lds_window,
                                    number<Alignment>,
                                    bool_constant<RematerializeStride>)
    {
        static_assert(std::is_same_v<remove_cvref_t<T>, bf16_t> ||
                          std::is_same_v<remove_cvref_t<T>, half_t>,
                      "the page tile writer instantiates BF16 and FP16");
        static_assert(Alignment > 0 && 64 % Alignment == 0, "unsupported global alignment");
        static_assert(sizeof(T) == 2, "16-bit element size contract");
#if defined(__HIP_DEVICE_COMPILE__)
        static_assert(CK_TILE_ENABLE_TDM_FEATURE,
                      "the page tile writer requires the gfx125 tensor data mover");
#endif

        constexpr index_t kBoxWidth = detail::CheckBoxWidth<LdsWindow, HeadDim>();

        const index_t wave = get_warp_id();
        if(wave >= kWaves)
            return;

        const auto lds_view   = lds_window.get_bottom_tensor_view();
        const auto lds_origin = lds_window.get_window_origin();

        // Within-page token offset of this wave's slice: the tile's offset in its page for
        // P>=128, the slice's offset in its page for P64 (tiles start on page boundaries).
        const index_t slice_row = PageSize >= 128
                                      ? static_cast<index_t>(logical_tile_origin % PageSize)
                                  : PageSize == 64 ? (kRowsPerWave * wave) % PageSize
                                                   : 0;

        static_for<0, kFragmentsPerWave, 1>{}([&](auto i_fragment) {
            constexpr index_t kFragment = decltype(i_fragment)::value;

            // Per-wave real rows drive the absent-fragment decision and the
            // P16 per-fragment extent. For P>=128 the TDM box spans the whole
            // logical tile, so the source view must carry the tile-level extent
            // R; using the per-wave count would clamp waves 1..3 to zero rows.
            const index_t wave_rows = fragment_valid_rows(wave, kFragment, ids);
            const index_t tile_rows = clamp(ids.owned[0].valid_rows, index_t{0}, index_t{128});
            const index_t view_rows = kPerWavePage ? wave_rows : tile_rows;
            const index_t pad_rows  = kPerWavePage ? kFragmentRows : index_t{128};
            // Retire the previous fragment's tensor traffic before the next fragment's
            // descriptor is issued. P16 only: a P16 wave emits two 16-row boxes per operand
            // (fragments 2w and 2w+1, each with its own page base); P>=128 has one fragment
            // per wave, so this compiles out.
            //
            // Without this step, with both boxes in flight and more than one workgroup resident
            // per WGP, the output can differ between launches on gfx1250, independently of page
            // ids, page mapping and base addresses; with the fragments serialized it does not.
            // The mechanism is not established. This is separate from the three-workgroup
            // residency limit of the kernel's launch contract (GetLaunchDynamicSmemSize).
            //
            // What each step covers (documented ISA behaviour):
            //  - s_wait_tensorcnt 0: the fragment's TDM transfer, including its writes to
            //    LDS, is complete. Tensor instructions are tracked only by TENSORcnt and are
            //    unordered with respect to other memory instructions.
            //  - s_wait_dscnt 0: drains the wave's own outstanding ordinary LDS operations;
            //    it does not track TDM-to-LDS writes.
            //  - barrier: the workgroup is at a common epoch before the next fragment
            //    publishes into the same LDS tile.
            if constexpr(PageSize == 16)
            {
                if constexpr(kFragment != 0)
                {
                    s_wait_tensorcnt<0>();
                    s_wait_dscnt<0>();
                    block_sync_lds<0>();
                }
            }
            // An absent or empty fragment keeps a valid page-0 pointer and a zero
            // real extent: the descriptor still issues and reads no global row. Its
            // padded box rows rely on the descriptor's out-of-bounds fill (documented as
            // zero fill; not independently measured on this toolchain). For K the
            // mask then discards those rows (their scores become -inf); for V it does NOT -
            // V rows are multiplied by P, so a non-zero (e.g. NaN) row would propagate -
            // so correctness of the V tail depends on that zero fill.
            const index_t present_rows = kPerWavePage ? wave_rows : tile_rows;
            // In an SGPR (the row count comes from a clamp, which selects v_med3_i32): the page
            // address arithmetic below stays on the scalar unit.
            const index_t physical_page = amd_wave_read_first_lane(
                present_rows > 0 ? ids.owned[kFragment].physical_page : index_t{0});

            // Full 64-bit pointer rebase; never a 32-bit pool-relative offset. The
            // P>=128 distribution's (4,32) warp factor already contributes the
            // 32*wave source rows, so the rebase adds the slice offset only. P16
            // fragments always start at physical row 0 of their own page; a P64 slice
            // starts at row 0 or 32 of its page.
            const long_index_t page_elements =
                static_cast<long_index_t>(physical_page) * page_stride;
            long_index_t within_page = 0;
            if constexpr(PageSize >= 64)
            {
                within_page =
                    static_cast<long_index_t>(slice_row) * static_cast<long_index_t>(token_stride);
            }
            const T* fragment_src = pool_head_ptr + page_elements + within_page;

            // Real pre-padding global extent: the tile's real rows and the real D
            // columns. The padded row extent only reaches the box; the K column
            // tail is left to descriptor zero-fill and the V inserted padding is
            // skipped, so no neighbouring page is ever read. The slice bound
            // x + R <= P is checked before rebasing and is never this extent.
            const auto src_naive = make_naive_tensor_view<address_space_enum::global>(
                fragment_src,
                make_tuple(view_rows, index_t{HeadDim}),
                make_tuple(token_stride, index_t{1}),
                number<Alignment>{},
                number<1>{});
            const auto src_view = pad_tensor_view(src_naive,
                                                  make_tuple(number<pad_rows>{}, number<HeadDim>{}),
                                                  sequence<true, false>{});

            if constexpr(kPerWavePage)
            {
                const auto fragment_lds = make_tile_window(
                    lds_view,
                    make_tuple(number<kFragmentRows>{}, number<kBoxWidth>{}),
                    {lds_origin.at(number<0>{}) + fragment_destination_row(wave, kFragment),
                     lds_origin.at(number<1>{})});

                const auto src_window = make_tile_window(
                    src_view,
                    make_tuple(number<kFragmentRows>{}, number<kBoxWidth>{}),
                    {0, 0},
                    detail::MakePageFragmentDistribution<kFragmentRows, kBoxWidth>());

                if constexpr(RematerializeStride)
                {
                    // The D128 dense fragment forwards the stride rematerialization
                    // flag to the existing ordinary tile_window path.
                    src_window.tdm_load_to_lds(
                        config,
                        fragment_lds,
                        make_null_tile_window(src_window.get_window_lengths()),
                        number<-1>{},
                        bool_constant<false>{},
                        bool_constant<true>{});
                }
                else
                {
                    load_tile_tdm(config, fragment_lds, src_window);
                }
            }
            else
            {
                const auto src_window =
                    make_tile_window(src_view,
                                     make_tuple(number<128>{}, number<kBoxWidth>{}),
                                     {0, 0},
                                     detail::MakePageN32RowDistribution<kBoxWidth>());

                if constexpr(RematerializeStride)
                {
                    src_window.tdm_load_to_lds(
                        config,
                        lds_window,
                        make_null_tile_window(src_window.get_window_lengths()),
                        number<-1>{},
                        bool_constant<false>{},
                        bool_constant<true>{});
                }
                else
                {
                    load_tile_tdm(config, lds_window, src_window);
                }
            }
        });
    }
};

} // namespace ck_tile
