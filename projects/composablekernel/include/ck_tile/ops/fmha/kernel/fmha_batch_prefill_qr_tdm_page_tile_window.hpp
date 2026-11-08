// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/detail/fmha_batch_prefill_page_tile.hpp"

#include <type_traits>

// Vectorized KV lean issue (P >= 64; the D64 lean loop and P64 D128): 1 (default) derives the
// per-wave TDM descriptor inputs once (as the linear tdm_issue_lean) instead of from the warp
// index and the page-id queue value on every tile; 0 keeps tdm_issue_vectorized.
#ifndef CK_TILE_FMHA_TDM_VEC_LEAN_ISSUE
#define CK_TILE_FMHA_TDM_VEC_LEAN_ISSUE 1
#endif

namespace ck_tile {

// The page-tile schedule reads page metadata once for each logical tile and supplies
// the same value to its K and V transfers. The window keeps logical tile origins
// separate from physical page addresses and forms ordinary rank-2 page-local TDM
// descriptors through the shared page tile writer.
//
// PageSize is the physical allocation page; the logical tile remains N128. The
// window holds only this wave's owned page metadata, never all fragments.

// KV cache layout of a page-tile window: linear [page][token][D] rows, or the vectorized
// (16-bit, 8-wide) K page-head [D/8][P][8] / V page-head [P/8][D][8].
struct FmhaPageTileKVKind
{
    static constexpr index_t kLinear      = 0;
    static constexpr index_t kVectorizedK = 1;
    static constexpr index_t kVectorizedV = 2;
};

template <typename DataType_,
          typename WindowLengths_,
          typename Distribution_,
          index_t Alignment_,
          index_t PageSize_,
          index_t HeadDim_,
          index_t KVKind_      = FmhaPageTileKVKind::kLinear,
          bool ScalarRowClamp_ = false> // see FmhaTdmSchedTuningOptions
struct FmhaBatchPrefillQrTdmPageTileWindow
{
    using DataType          = DataType_;
    using WindowLengths     = WindowLengths_;
    using Distribution      = Distribution_;
    using BottomTensorIndex = array<index_t, 2>;
    using PageIds           = FmhaBatchPrefillPageTileIds<PageSize_>;

    static constexpr index_t kPageSize = PageSize_;
    static constexpr index_t kHeadDim  = HeadDim_;
    static constexpr index_t kKVKind   = KVKind_;
    static constexpr index_t kWaves = FmhaBatchPrefillPageTileTraits<PageSize_, HeadDim_>::kWaves;
    static constexpr index_t kFragmentsPerWave = PageIds::kFragmentsPerWave;
    static constexpr index_t kFragmentRows =
        FmhaBatchPrefillPageTileTraits<PageSize_, HeadDim_>::kFragmentRows;
    static constexpr index_t kSliceRows =
        FmhaBatchPrefillPageTileTraits<PageSize_, HeadDim_>::kRowsPerWave;
    // Pages per logical 128-row tile on the scalar queue: two for P64, one for P>=128.
    static constexpr index_t kQueuePages = PageSize_ < 128 ? 128 / PageSize_ : 1;

    const DataType* pool_ptr_ = nullptr;
    index_t seqlen_k_         = 0;
    index_t token_stride_     = 0;
    const int32_t* page_row_  = nullptr;
    long_index_t page_stride_ = 0;
    index_t page_capacity_    = 0;
    index_t total_pages_      = 0;
    PageIds page_id_{};
    BottomTensorIndex origin_ = {0, 0};

    CK_TILE_DEVICE constexpr auto get_window_lengths() const { return WindowLengths{}; }
    CK_TILE_DEVICE constexpr auto get_window_origin() const { return origin_; }
    CK_TILE_DEVICE void move(const BottomTensorIndex& step) { origin_ += step; }
    CK_TILE_DEVICE void set_page_id(const PageIds& page_id) { page_id_ = page_id; }

    // Logical tile origin of this tile, in logical KV tokens.
    CK_TILE_DEVICE long_index_t logical_tile_origin_of(index_t tile_index,
                                                       index_t first_origin) const
    {
        return static_cast<long_index_t>(first_origin) +
               static_cast<long_index_t>(tile_index) * 128;
    }

    // With sink tokens (StreamingLLM) the key loop visits the sink tiles from first_origin and
    // then jumps to the window start: tile i starts at resume_origin + 128 * (i - sink_tiles)
    // from sink_tiles on. Instances without sinks use the contiguous overload above.
    CK_TILE_DEVICE long_index_t logical_tile_origin_of(index_t tile_index,
                                                       index_t first_origin,
                                                       const FmhaPageTileSinkTiles& order) const
    {
        if(tile_index >= order.sink_tiles)
            return static_cast<long_index_t>(order.resume_origin) +
                   static_cast<long_index_t>(tile_index - order.sink_tiles) * 128;
        return logical_tile_origin_of(tile_index, first_origin);
    }

    // Physical page lookup for one owned fragment.
    // P>=128: fragment 0 only, table column t / P.
    // P64   : fragment 0 only, table column (t + 32*w) / 64; valid_rows stays tile-level.
    // P16   : fragment f = 2*w + r, table column (t + 16*f) / 16.
    CK_TILE_DEVICE FmhaBatchPrefillPageFragmentId load_fragment_id(long_index_t tile_origin,
                                                                   index_t wave,
                                                                   index_t fragment) const
    {
        FmhaBatchPrefillPageFragmentId result{};
        if constexpr(PageSize_ == 16)
        {
            const index_t f = 2 * wave + fragment;
            const long_index_t fragment_origin =
                tile_origin + static_cast<long_index_t>(f) * kFragmentRows;
            const index_t valid_rows = static_cast<index_t>(
                clamp<long_index_t>(seqlen_k_ - fragment_origin, 0, kFragmentRows));
            if(valid_rows <= 0)
                return result; // absent: no table read, dummy id, zero rows
            const index_t logical_page = static_cast<index_t>(fragment_origin / kFragmentRows);
            if(logical_page >= page_capacity_ || page_row_ == nullptr)
                __builtin_trap();
            const index_t physical_page = page_row_[logical_page];
            if(physical_page < 0 || physical_page >= total_pages_)
                __builtin_trap();
            result.physical_page = physical_page;
            result.valid_rows    = valid_rows;
        }
        else if constexpr(PageSize_ == 64)
        {
            (void)fragment;
            const index_t valid_rows =
                static_cast<index_t>(clamp<long_index_t>(seqlen_k_ - tile_origin, 0, 128));
            const index_t page_in_tile = kSliceRows * wave / PageSize_;
            result.valid_rows          = valid_rows;
            // A wave whose page holds no valid row keeps the dummy id: no table read.
            if(valid_rows <= page_in_tile * PageSize_)
                return result;
            const index_t logical_page =
                static_cast<index_t>(tile_origin / PageSize_) + page_in_tile;
            if(logical_page >= page_capacity_ || page_row_ == nullptr)
                __builtin_trap();
            const index_t physical_page = page_row_[logical_page];
            if(physical_page < 0 || physical_page >= total_pages_)
                __builtin_trap();
            result.physical_page = physical_page;
        }
        else
        {
            (void)wave;
            (void)fragment;
            const index_t valid_rows =
                static_cast<index_t>(clamp<long_index_t>(seqlen_k_ - tile_origin, 0, 128));
            if(valid_rows <= 0)
                return result;
            const index_t logical_page = static_cast<index_t>(tile_origin / PageSize_);
            if(logical_page >= page_capacity_ || page_row_ == nullptr)
                __builtin_trap();
            const index_t physical_page = page_row_[logical_page];
            if(physical_page < 0 || physical_page >= total_pages_)
                __builtin_trap();
            result.physical_page = physical_page;
            result.valid_rows    = valid_rows;
        }
        return result;
    }

    // Synchronous prologue lookup for the j / j+1 / j+2 metadata values.
    // P>=128 returns the one shared id; P64 this wave's page id; P16 this wave's two owned ids.
    CK_TILE_DEVICE PageIds prefetch_page_id(index_t tile_index,
                                            index_t num_total_loop,
                                            index_t first_origin) const
    {
        PageIds result{};
        if(tile_index < 0 || tile_index >= num_total_loop)
            return result;
        const index_t wave = get_warp_id();
        if(wave >= kWaves)
            return result;
        const long_index_t tile_origin = logical_tile_origin_of(tile_index, first_origin);
        static_for<0, kFragmentsPerWave, 1>{}([&](auto i_fragment) {
            result.owned[decltype(i_fragment)::value] =
                load_fragment_id(tile_origin, wave, decltype(i_fragment)::value);
        });
        return result;
    }

    // Sink-order prologue lookup (see logical_tile_origin_of).
    CK_TILE_DEVICE PageIds prefetch_page_id(index_t tile_index,
                                            index_t num_total_loop,
                                            index_t first_origin,
                                            const FmhaPageTileSinkTiles& order) const
    {
        PageIds result{};
        if(tile_index < 0 || tile_index >= num_total_loop)
            return result;
        const index_t wave = get_warp_id();
        if(wave >= kWaves)
            return result;
        const long_index_t tile_origin = logical_tile_origin_of(tile_index, first_origin, order);
        static_for<0, kFragmentsPerWave, 1>{}([&](auto i_fragment) {
            result.owned[decltype(i_fragment)::value] =
                load_fragment_id(tile_origin, wave, decltype(i_fragment)::value);
        });
        return result;
    }

    // P16 steady-state metadata publication. One masked wave0 async b32 writes up
    // to eight owned physical ids into the 32-byte pending slot. The caller waits
    // for asynccnt 0 and crosses a workgroup barrier before any wave reads a slot.
    //
    // The API's false branch performs a synchronous LDS zero store rather than a
    // no-op, so every lane that participates must pass is_valid_element=true and
    // the slot element must stay inside the 32-byte slot.
    CK_TILE_DEVICE void prefetch_page_id_to_lds_p16(long_index_t tile_origin,
                                                    CK_TILE_LDS_ADDR int32_t* id_slot) const
    {
        if constexpr(PageSize_ != 16)
            return;
        // Number of owned fragments in this lookahead tile that contain at least
        // one valid row. A partially valid tail fragment is NOT absent: it must
        // still publish its physical id, or the tail rows would read a stale or
        // dummy page. Round up, never truncate.
        const long_index_t remaining =
            clamp<long_index_t>(seqlen_k_ - tile_origin, 0, 8 * kFragmentRows);
        const index_t valid_pages =
            static_cast<index_t>((remaining + kFragmentRows - 1) / kFragmentRows);
        if(valid_pages <= 0)
            return; // absent lookahead: no table pointer is formed at all
        if(get_warp_id() != 0)
            return;
        const index_t lane = get_lane_id();
        if(lane < valid_pages && lane < 8)
        {
            const long_index_t first_page = tile_origin / kFragmentRows;
            if(first_page < 0 || first_page >= page_capacity_ || page_row_ == nullptr)
                __builtin_trap();
            const auto* first_id = page_row_ + static_cast<long_index_t>(first_page);
            // The active lane set and the published row count must agree with the
            // per-fragment validity the consumer derives, including a valid
            // prefix of 1..16 rows in the final fragment.
            // The pointer is uniform across active lanes; the lane-dependent
            // global element offset carries the fragment index.
            amd_async_global_load_to_lds<int32_t, 1, 0, true>(
                id_slot + lane,
                reinterpret_cast<const CK_TILE_GLOBAL_ADDR int32_t*>(first_id),
                lane,
                true);
        }
    }

    // ---- P16 page-id queue (P>=64 uses the *_scalar queue below) ----
    // Returns the owned valid-row metadata for the pending tile with dummy ids.
    CK_TILE_DEVICE PageIds prefetch_page_id_to_lds(index_t tile_index,
                                                   index_t num_total_loop,
                                                   index_t first_origin,
                                                   CK_TILE_LDS_ADDR int32_t* id_slot) const
    {
        static_assert(PageSize_ == 16, "the struct page-id queue serves P16 only");
        PageIds result{};
        if(tile_index < 0 || tile_index >= num_total_loop)
            return result;
        const index_t wave = get_warp_id();
        if(wave >= kWaves)
            return result;
        const long_index_t tile_origin = logical_tile_origin_of(tile_index, first_origin);
        static_for<0, kFragmentsPerWave, 1>{}([&](auto i_fragment) {
            FmhaBatchPrefillPageFragmentId valid{};
            const index_t f = 2 * wave + decltype(i_fragment)::value;
            const long_index_t fragment_origin =
                tile_origin + static_cast<long_index_t>(f) * kFragmentRows;
            valid.valid_rows = static_cast<index_t>(
                clamp<long_index_t>(seqlen_k_ - fragment_origin, 0, kFragmentRows));
            result.owned[decltype(i_fragment)::value] = valid; // dummy id 0
        });
        prefetch_page_id_to_lds_p16(tile_origin, id_slot);
        return result;
    }

    // Sink-order P16 queue publication (see logical_tile_origin_of).
    CK_TILE_DEVICE PageIds prefetch_page_id_to_lds(index_t tile_index,
                                                   index_t num_total_loop,
                                                   index_t first_origin,
                                                   CK_TILE_LDS_ADDR int32_t* id_slot,
                                                   const FmhaPageTileSinkTiles& order) const
    {
        static_assert(PageSize_ == 16, "the struct page-id queue serves P16 only");
        PageIds result{};
        if(tile_index < 0 || tile_index >= num_total_loop)
            return result;
        const index_t wave = get_warp_id();
        if(wave >= kWaves)
            return result;
        const long_index_t tile_origin = logical_tile_origin_of(tile_index, first_origin, order);
        static_for<0, kFragmentsPerWave, 1>{}([&](auto i_fragment) {
            FmhaBatchPrefillPageFragmentId valid{};
            const index_t f = 2 * wave + decltype(i_fragment)::value;
            const long_index_t fragment_origin =
                tile_origin + static_cast<long_index_t>(f) * kFragmentRows;
            valid.valid_rows = static_cast<index_t>(
                clamp<long_index_t>(seqlen_k_ - fragment_origin, 0, kFragmentRows));
            result.owned[decltype(i_fragment)::value] = valid; // dummy id 0
        });
        prefetch_page_id_to_lds_p16(tile_origin, id_slot);
        return result;
    }

    // Read this wave's owned raw ids before PV. No validation happens here.
    CK_TILE_DEVICE PageIds load_page_id_from_lds(const PageIds& valid_metadata,
                                                 CK_TILE_LDS_ADDR int32_t* id_slot) const
    {
        static_assert(PageSize_ == 16, "the struct page-id queue serves P16 only");
        PageIds result     = valid_metadata;
        const index_t wave = get_warp_id();
        if(wave >= kWaves)
            return result;
        static_for<0, kFragmentsPerWave, 1>{}([&](auto i_fragment) {
            constexpr index_t kFragment = decltype(i_fragment)::value;
            if(result.owned[kFragment].valid_rows > 0)
                result.owned[kFragment].physical_page = id_slot[2 * wave + kFragment];
        });
        return result;
    }

    // Validate and complete the queue value after PV.
    CK_TILE_DEVICE PageIds complete_page_id_from_value(const PageIds& valid_metadata,
                                                       const PageIds& raw_ids) const
    {
        static_assert(PageSize_ == 16, "the struct page-id queue serves P16 only");
        PageIds result = valid_metadata;
        static_for<0, kFragmentsPerWave, 1>{}([&](auto i_fragment) {
            constexpr index_t kFragment = decltype(i_fragment)::value;
            const index_t physical_page = raw_ids.owned[kFragment].physical_page;
            if(result.owned[kFragment].valid_rows > 0)
            {
                if(physical_page < 0 || physical_page >= total_pages_)
                    __builtin_trap();
                result.owned[kFragment].physical_page = physical_page;
            }
            else
            {
                result.owned[kFragment].physical_page = 0;
            }
        });
        return result;
    }

    // ---- P>=64 scalar page-id queue ----
    // P>=128: one shared physical page per logical tile. P64: the tile's two page ids, one
    // per slot; each wave reads the one holding its 32-row slice. publish returns the tile's
    // valid rows; read returns this wave's raw slot id; complete validates it.
    CK_TILE_DEVICE index_t prefetch_page_id_to_lds_scalar(index_t tile_index,
                                                          index_t num_total_loop,
                                                          index_t first_origin,
                                                          CK_TILE_LDS_ADDR int32_t* id_slot) const
    {
        if constexpr(PageSize_ == 16)
            return 0;
        if(tile_index < 0 || tile_index >= num_total_loop)
            return 0;
        // No per-tile bounds traps. check_loop_invariants() checks first_origin >= 0 and
        // 128-aligned before the loop. tile_index < num_total_loop (precondition: num_total_loop
        // is derived from this window's seqlen_k_ through the mask tile range) puts
        // logical_origin inside seqlen_k_. The kernel prologue checks page_count <=
        // page_capacity_, positive strides and num_total_pages > 0. The loaded physical id is
        // range-checked in complete_page_id_from_value_scalar.
        const long_index_t logical_origin =
            static_cast<long_index_t>(first_origin) + static_cast<long_index_t>(tile_index) * 128;
        const index_t logical_page = static_cast<index_t>(logical_origin / PageSize_);
        if constexpr(kQueuePages > 1)
        {
            // One lane per page that holds a valid row (lane 1 only when the tile reaches
            // its second page), so no table entry past the sequence is read. The pointer is
            // uniform across the active lanes; the lane offset selects the page.
            const index_t valid_tokens =
                min(index_t{128}, seqlen_k_ - static_cast<index_t>(logical_origin));
            const index_t lane = get_thread_id();
            if(lane < kQueuePages && lane * PageSize_ < valid_tokens)
            {
                amd_async_global_load_to_lds<int32_t, 1, 0, true>(
                    id_slot + lane,
                    reinterpret_cast<const CK_TILE_GLOBAL_ADDR int32_t*>(page_row_ + logical_page),
                    lane,
                    true);
            }
            return valid_tokens;
        }
        // P>=128 keeps its original statement order (its code objects are compared bit for bit).
        if(get_thread_id() == 0)
        {
            amd_async_global_load_to_lds<int32_t, 1>(
                id_slot,
                reinterpret_cast<const CK_TILE_GLOBAL_ADDR int32_t*>(page_row_ + logical_page),
                0,
                true);
        }
        return min(index_t{128}, seqlen_k_ - static_cast<index_t>(logical_origin));
    }

    // Sink-order scalar queue publication (see logical_tile_origin_of); same body as above with
    // the tile origin taken from the sink order.
    CK_TILE_DEVICE index_t prefetch_page_id_to_lds_scalar(index_t tile_index,
                                                          index_t num_total_loop,
                                                          index_t first_origin,
                                                          CK_TILE_LDS_ADDR int32_t* id_slot,
                                                          const FmhaPageTileSinkTiles& order) const
    {
        if constexpr(PageSize_ == 16)
            return 0;
        if(tile_index < 0 || tile_index >= num_total_loop)
            return 0;
        const long_index_t logical_origin = logical_tile_origin_of(tile_index, first_origin, order);
        const index_t logical_page        = static_cast<index_t>(logical_origin / PageSize_);
        const index_t valid_tokens =
            min(index_t{128}, seqlen_k_ - static_cast<index_t>(logical_origin));
        const index_t lane = get_thread_id();
        if(lane < kQueuePages && lane * PageSize_ < valid_tokens)
        {
            amd_async_global_load_to_lds<int32_t, 1, 0, true>(
                id_slot + lane,
                reinterpret_cast<const CK_TILE_GLOBAL_ADDR int32_t*>(page_row_ + logical_page),
                lane,
                true);
        }
        return valid_tokens;
    }

    CK_TILE_DEVICE index_t load_page_id_from_lds_scalar(index_t valid_tokens,
                                                        CK_TILE_LDS_ADDR int32_t* id_slot) const
    {
        if constexpr(kQueuePages > 1)
        {
            // A page without a valid row was not published: report the dummy id 0.
            const index_t page_in_tile = kSliceRows * get_warp_id() / PageSize_;
            if(valid_tokens <= page_in_tile * PageSize_)
                return 0;
            return id_slot[page_in_tile];
        }
        if(valid_tokens <= 0)
            return 0;
        return *id_slot;
    }

    CK_TILE_DEVICE PageIds complete_page_id_from_value_scalar(index_t valid_tokens,
                                                              index_t physical_page) const
    {
        PageIds result{};
        if(valid_tokens <= 0)
            return result;
        // One unsigned compare covers both physical_page < 0 and >= total_pages_, given
        // total_pages_ > 0 (kernel prologue).
        if(static_cast<uint32_t>(physical_page) >= static_cast<uint32_t>(total_pages_))
            __builtin_trap();
        result.owned[0].physical_page = physical_page;
        result.owned[0].valid_rows    = valid_tokens;
        return result;
    }

    // Loop-invariant preconditions of the steady-state page-id queue and TDM issue, checked
    // once before the KV loop instead of on every tile. The window origin starts at
    // first_origin and only moves by whole 128-token tiles (the page-tile pipeline has no
    // sinks).
    CK_TILE_DEVICE void check_loop_invariants(index_t first_origin) const
    {
        if(seqlen_k_ <= 0 || first_origin < 0 || first_origin % 128 != 0 || page_row_ == nullptr)
            __builtin_trap();
    }

    // ---- vectorized KV layout issue (16-bit, x = 8) ----
    // One descriptor per wave slice (two per P16 wave).
    //   K rank 2: dims (rows*8, D/8), stride P*8: copies the page's column vectors of the
    //     fragment as they are, into the [D/8][G][8] K LDS tile of the vectorized layout.
    //   V rank 2: dims (D*8, groups of 8 rows), stride D*8: copies the page's groups as they
    //     are, into the [rows/8][D][8] V LDS tile of the vectorized layout (plain b128 reads).
    // V rows past the valid count never reach LDS: a partial last group is copied by a second,
    // serialized descriptor (see the V branch).
    template <typename TDMConfig, typename LdsWindow>
    CK_TILE_DEVICE void tdm_issue_vectorized(const TDMConfig& config,
                                             const LdsWindow& lds_window) const
    {
        static_assert(KVKind_ == FmhaPageTileKVKind::kVectorizedK ||
                      KVKind_ == FmhaPageTileKVKind::kVectorizedV);
        static_assert(!std::is_void_v<Distribution>);
        static_assert(sizeof(DataType) == 2, "the vectorized layout is 8 x 16-bit wide");
        using Writer           = FmhaBatchPrefillLinearPageTileWriter<PageSize_, HeadDim_>;
        constexpr index_t kVec = 8;
        constexpr auto raw_box =
            to_sequence(Distribution{}.get_ys_to_d_descriptor().get_lengths()).reverse();
        static_assert(raw_box.size() == 2);
        constexpr index_t kBoxCols = raw_box[number<0>{}];
        static_assert(kBoxCols % kVec == 0 && HeadDim_ % kVec == 0 && PageSize_ % kVec == 0 &&
                      kFragmentRows % kVec == 0);

        const index_t wave = get_warp_id();
        if(wave >= kWaves)
            return;
        const long_index_t tile_origin = logical_tile_origin_of(0, origin_[0]);
        const auto lds_view            = lds_window.get_bottom_tensor_view();
        const auto lds_origin          = lds_window.get_window_origin();

        static_for<0, kFragmentsPerWave, 1>{}([&](auto i_fragment) {
            constexpr index_t kFragment = decltype(i_fragment)::value;
            // Same P16 fragment serialization as the linear writer.
            if constexpr(PageSize_ == 16 && kFragment != 0)
            {
                s_wait_tensorcnt<0>();
                s_wait_dscnt<0>();
                block_sync_lds<0>();
            }
            const index_t rows          = Writer::fragment_valid_rows(wave, kFragment, page_id_);
            const index_t physical_page = rows > 0 ? page_id_.owned[kFragment].physical_page : 0;
            // First page-local token of the fragment.
            index_t t0 = 0;
            if constexpr(PageSize_ == 64)
                t0 = (kSliceRows * wave) % PageSize_;
            else if constexpr(PageSize_ >= 128)
                t0 = static_cast<index_t>(tile_origin % PageSize_) + kSliceRows * wave;
            const auto lds_coord = make_tensor_coordinate(
                lds_view.get_tensor_descriptor(),
                make_multi_index(lds_origin.at(number<0>{}) +
                                     Writer::fragment_destination_row(wave, kFragment),
                                 lds_origin.at(number<1>{})));
            auto* lds_ptr = lds_view.get_buffer_view().p_data_ + lds_coord.get_offset();
            const DataType* page_ptr =
                pool_ptr_ + static_cast<long_index_t>(physical_page) * page_stride_;
            if constexpr(KVKind_ == FmhaPageTileKVKind::kVectorizedK)
            {
                // [D/8][rows][8]: per column vector, the fragment's tokens are contiguous in the
                // page (rows*8 elements), so each descriptor row is a long contiguous run.
                const uint32_t dims[2]    = {static_cast<uint32_t>(rows * kVec),
                                             static_cast<uint32_t>(HeadDim_ / kVec)};
                const uint64_t strides[2] = {static_cast<uint64_t>(PageSize_) * kVec, 1};
                constexpr uint16_t box[2] = {kFragmentRows * kVec, HeadDim_ / kVec};
                amd_tdm_load<amd_buffer_coherence_enum::coherence_default>(
                    createTDMDescriptor<remove_cvref_t<DataType>, 2>(
                        page_ptr + static_cast<long_index_t>(t0) * kVec,
                        lds_ptr,
                        dims,
                        strides,
                        box,
                        config));
            }
            else
            {
                // Whole groups of 8 rows: [groups][D][8] in LDS (the V LDS tile of the
                // vectorized layout). A partial last group (tail tile only) cannot be bounded by
                // this descriptor: it is left out here (zero filled as out of bounds) and, once
                // that transfer has landed, copied by a rank-3 descriptor whose token extent
                // zero fills the rest of the group - V rows past seqlen_k never reach LDS.
                static_assert(kBoxCols == HeadDim_, "the V box spans the head width");
                const index_t full_groups = rows / kVec;
                const index_t tail_rows   = rows % kVec;
                const DataType* slice_ptr =
                    page_ptr + static_cast<long_index_t>(t0 / kVec) * HeadDim_ * kVec;
                const uint32_t dims[2]    = {static_cast<uint32_t>(HeadDim_ * kVec),
                                             static_cast<uint32_t>(full_groups)};
                const uint64_t strides[2] = {static_cast<uint64_t>(HeadDim_) * kVec, 1};
                constexpr uint16_t box[2] = {HeadDim_ * kVec, kFragmentRows / kVec};
                amd_tdm_load<amd_buffer_coherence_enum::coherence_default>(
                    createTDMDescriptor<remove_cvref_t<DataType>, 2>(
                        slice_ptr, lds_ptr, dims, strides, box, config));
                if(tail_rows != 0)
                {
                    // Serialized after the group transfer (both write the tail group's LDS). The
                    // schedule's partial tensor waits still hold: this transfer takes the group
                    // transfer's place in issue order, and everything older has completed.
                    s_wait_tensorcnt<0>();
                    const uint32_t tail_dims[3] = {
                        static_cast<uint32_t>(tail_rows), static_cast<uint32_t>(HeadDim_), 1};
                    const uint64_t tail_strides[3] = {
                        kVec, static_cast<uint64_t>(HeadDim_) * kVec, 0};
                    constexpr uint16_t tail_box[3] = {kVec, HeadDim_, 1};
                    // The page holds the groups unpadded; the LDS position comes from the
                    // V LDS descriptor (which may pad).
                    const long_index_t tail_offset =
                        static_cast<long_index_t>(full_groups) * HeadDim_ * kVec;
                    const auto tail_coord = make_tensor_coordinate(
                        lds_view.get_tensor_descriptor(),
                        make_multi_index(lds_origin.at(number<0>{}) +
                                             Writer::fragment_destination_row(wave, kFragment) +
                                             full_groups * kVec,
                                         lds_origin.at(number<1>{})));
                    amd_tdm_load<amd_buffer_coherence_enum::coherence_default>(
                        createTDMDescriptor<remove_cvref_t<DataType>, 3>(
                            slice_ptr + tail_offset,
                            lds_view.get_buffer_view().p_data_ + tail_coord.get_offset(),
                            tail_dims,
                            tail_strides,
                            tail_box,
                            config));
                }
            }
        });
    }

#if CK_TILE_FMHA_TDM_VEC_LEAN_ISSUE
    // tdm_issue_vectorized for one fragment per wave (P >= 64) from the per-wave state of
    // make_lean_issue_state: per tile only the page address and the valid rows change, and the
    // rows are wave-uniform so the descriptor and the group/tail split stay scalar.
    // clamp(x, 0, Hi) of a wave-uniform value, kept in SALU. LLVM selects v_med3_i32 for an i32
    // clamp even when the inputs are SGPRs, which costs a v_readfirstlane or a VCC branch per use
    // in the steady loop.
    template <index_t Hi>
    CK_TILE_DEVICE static index_t ScalarClamp(index_t x)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        index_t r;
        asm("s_max_i32 %0, %1, 0\n\ts_min_i32 %0, %0, %2" : "=&s"(r) : "s"(x), "n"(Hi) : "scc");
        return r;
#else
        return clamp(x, index_t{0}, index_t{Hi});
#endif
    }

    template <typename TDMConfig, typename LdsWindow, typename State>
    CK_TILE_DEVICE void tdm_issue_vectorized_lean(const TDMConfig& config,
                                                  const LdsWindow& lds_window,
                                                  const State& state) const
    {
        static_assert(PageSize_ >= 64 && kFragmentsPerWave == 1);
        static_assert(sizeof(DataType) == 2, "the vectorized layout is 8 x 16-bit wide");
        using Writer           = FmhaBatchPrefillLinearPageTileWriter<PageSize_, HeadDim_>;
        constexpr index_t kVec = 8;
        const index_t wave     = state.wave_row0;
        const index_t rows     = [&]() {
            // P >= 128: from the scalar tile origin (no sinks on the lean loop). Reading the
            // queue value into a scalar instead spills on P128 D64 (VGPR limit ~320). A
            // look-ahead past the loop end can get rows > 0 here while its queue entry is empty;
            // it then loads the dummy page 0, which nothing reads.
            if constexpr(ScalarRowClamp_)
            {
                if constexpr(PageSize_ >= 128)
                    return ScalarClamp<kSliceRows>(seqlen_k_ - static_cast<index_t>(origin_[0]) -
                                                   kSliceRows * wave);
                else
                    return ScalarClamp<Writer::kRowsPerWave>(
                        amd_wave_read_first_lane(page_id_.owned[0].valid_rows) -
                        Writer::kRowsPerWave * wave);
            }
            else if constexpr(PageSize_ >= 128)
                return clamp(seqlen_k_ - static_cast<index_t>(origin_[0]) - kSliceRows * wave,
                             index_t{0},
                             index_t{kSliceRows});
            else
                return amd_wave_read_first_lane(Writer::fragment_valid_rows(wave, 0, page_id_));
        }();
        const index_t physical_page = rows > 0 ? page_id_.owned[0].physical_page : 0;
        // Tiles start on 128-token boundaries: only pages larger than a tile add the tile's
        // offset inside its page (as tdm_issue_lean).
        index_t t0 = kSliceRows * wave;
        if constexpr(PageSize_ == 64)
            t0 = t0 % PageSize_;
        else if constexpr(PageSize_ > 128)
            t0 += static_cast<index_t>(origin_[0] % PageSize_);
        auto* lds_base = lds_window.get_bottom_tensor_view().get_buffer_view().p_data_;
        auto* lds_ptr  = lds_base + state.lds_elem_offset;
        const DataType* page_ptr =
            pool_ptr_ + static_cast<long_index_t>(physical_page) * page_stride_;
        if constexpr(KVKind_ == FmhaPageTileKVKind::kVectorizedK)
        {
            const uint32_t dims[2]    = {static_cast<uint32_t>(rows * kVec),
                                         static_cast<uint32_t>(HeadDim_ / kVec)};
            const uint64_t strides[2] = {static_cast<uint64_t>(PageSize_) * kVec, 1};
            constexpr uint16_t box[2] = {kFragmentRows * kVec, HeadDim_ / kVec};
            amd_tdm_load<amd_buffer_coherence_enum::coherence_default>(
                createTDMDescriptor<remove_cvref_t<DataType>, 2>(
                    page_ptr + static_cast<long_index_t>(t0) * kVec,
                    lds_ptr,
                    dims,
                    strides,
                    box,
                    config));
        }
        else
        {
            const index_t full_groups = rows / kVec;
            const index_t tail_rows   = rows % kVec;
            const DataType* slice_ptr =
                page_ptr + static_cast<long_index_t>(t0 / kVec) * HeadDim_ * kVec;
            const uint32_t dims[2]    = {static_cast<uint32_t>(HeadDim_ * kVec),
                                         static_cast<uint32_t>(full_groups)};
            const uint64_t strides[2] = {static_cast<uint64_t>(HeadDim_) * kVec, 1};
            constexpr uint16_t box[2] = {HeadDim_ * kVec, kFragmentRows / kVec};
            amd_tdm_load<amd_buffer_coherence_enum::coherence_default>(
                createTDMDescriptor<remove_cvref_t<DataType>, 2>(
                    slice_ptr, lds_ptr, dims, strides, box, config));
            if(tail_rows != 0)
            {
                // Same serialized partial-group copy as tdm_issue_vectorized (tail tile only).
                s_wait_tensorcnt<0>();
                const auto lds_view         = lds_window.get_bottom_tensor_view();
                const auto lds_origin       = lds_window.get_window_origin();
                const uint32_t tail_dims[3] = {
                    static_cast<uint32_t>(tail_rows), static_cast<uint32_t>(HeadDim_), 1};
                const uint64_t tail_strides[3] = {kVec, static_cast<uint64_t>(HeadDim_) * kVec, 0};
                constexpr uint16_t tail_box[3] = {kVec, HeadDim_, 1};
                const long_index_t tail_offset =
                    static_cast<long_index_t>(full_groups) * HeadDim_ * kVec;
                const auto tail_coord = make_tensor_coordinate(
                    lds_view.get_tensor_descriptor(),
                    make_multi_index(lds_origin.at(number<0>{}) +
                                         Writer::fragment_destination_row(wave, 0) +
                                         full_groups * kVec,
                                     lds_origin.at(number<1>{})));
                amd_tdm_load<amd_buffer_coherence_enum::coherence_default>(
                    createTDMDescriptor<remove_cvref_t<DataType>, 3>(slice_ptr + tail_offset,
                                                                     lds_base +
                                                                         tail_coord.get_offset(),
                                                                     tail_dims,
                                                                     tail_strides,
                                                                     tail_box,
                                                                     config));
            }
        }
    }
#endif

    // ---- lean whole-tile TDM issue (P >= 64) ----
    // The per-wave inputs of a page-tile TDM descriptor that do not change between tiles: the
    // wave's (row, column) origin inside the 128-row tile and its LDS element offset. The
    // generic issue path re-derives them from the warp index for every tile, which puts a
    // v_readfirstlane -> SALU dependency chain into the steady loop.
    struct LeanIssueState
    {
        index_t wave_row0       = 0;
        index_t wave_col0       = 0;
        index_t lds_elem_offset = 0;
    };

    template <typename LdsWindow>
    CK_TILE_DEVICE LeanIssueState make_lean_issue_state(const LdsWindow& lds_window) const
    {
        static_assert(PageSize_ >= 64, "the lean issue covers one page per wave slice only");
        static_assert(!std::is_void_v<Distribution>);
#if CK_TILE_FMHA_TDM_VEC_LEAN_ISSUE
        if constexpr(KVKind_ != FmhaPageTileKVKind::kLinear)
        {
            // wave_row0: the wave index; lds_elem_offset: the wave slice's LDS destination
            // (same coordinate tdm_issue_vectorized forms per tile).
            using Writer     = FmhaBatchPrefillLinearPageTileWriter<PageSize_, HeadDim_>;
            const index_t w  = get_warp_id();
            const auto view  = lds_window.get_bottom_tensor_view();
            const auto o     = lds_window.get_window_origin();
            const auto coord = make_tensor_coordinate(
                view.get_tensor_descriptor(),
                make_multi_index(o.at(number<0>{}) + Writer::fragment_destination_row(w, 0),
                                 o.at(number<1>{})));
            LeanIssueState state;
            state.wave_row0       = amd_wave_read_first_lane(w);
            state.lds_elem_offset = amd_wave_read_first_lane(coord.get_offset());
            return state;
        }
#endif
        constexpr auto distribution = Distribution{};
        // Same thread coordinate as tile_window_with_static_distribution::prepare_coords.
        const auto coord =
            make_tensor_adaptor_coordinate(distribution.get_ps_ys_to_xs_adaptor(),
                                           container_concat(get_partition_index(distribution),
                                                            multi_index<Distribution::NDimY>{0}));
        const auto origin = coord.get_bottom_index();
        const auto lds_coord =
            make_tensor_coordinate(lds_window.get_bottom_tensor_view().get_tensor_descriptor(),
                                   lds_window.get_window_origin() + origin);
        LeanIssueState state;
        state.wave_row0       = amd_wave_read_first_lane(origin[number<0>{}]);
        state.wave_col0       = amd_wave_read_first_lane(origin[number<1>{}]);
        state.lds_elem_offset = amd_wave_read_first_lane(lds_coord.get_offset());
        return state;
    }

    // Issues the same descriptor as tdm_load_to_lds (same packer, same field values) from the
    // precomputed per-wave state: per tile only the page address and the valid rows change.
    template <typename TDMConfig, typename LdsWindow>
    CK_TILE_DEVICE void tdm_issue_lean(const TDMConfig& config,
                                       const LdsWindow& lds_window,
                                       const LeanIssueState& state) const
    {
        static_assert(PageSize_ >= 64, "the lean issue covers one page per wave slice only");
        if constexpr(KVKind_ != FmhaPageTileKVKind::kLinear)
        {
#if CK_TILE_FMHA_TDM_VEC_LEAN_ISSUE
            tdm_issue_vectorized_lean(config, lds_window, state);
#else
            (void)state;
            tdm_issue_vectorized(config, lds_window);
#endif
            return;
        }
        constexpr auto raw_box =
            to_sequence(Distribution{}.get_ys_to_d_descriptor().get_lengths()).reverse();
        static_assert(raw_box.size() == 2);
        constexpr uint16_t box[2] = {static_cast<uint16_t>(raw_box[number<0>{}]),
                                     static_cast<uint16_t>(raw_box[number<1>{}])};

        const index_t valid_tokens = clamp(page_id_.owned[0].valid_rows, index_t{0}, index_t{128});
        const index_t physical_page =
            valid_tokens > 0 ? page_id_.owned[0].physical_page : index_t{0};
        // P > 128 pages hold several tiles; the tile starts at origin % P inside its page.
        // P64: the wave's page (owned[0]) holds its slice at row wave_row0 % 64.
        const index_t slice_row = PageSize_ > 128 ? origin_[0] % PageSize_ : index_t{0};
        long_index_t element_offset;
        if constexpr(PageSize_ < 128)
            element_offset =
                static_cast<long_index_t>(physical_page) * page_stride_ +
                static_cast<long_index_t>(state.wave_row0 % PageSize_) * token_stride_ +
                state.wave_col0;
        else
            element_offset =
                static_cast<long_index_t>(physical_page) * page_stride_ +
                static_cast<long_index_t>(slice_row + state.wave_row0) * token_stride_ +
                state.wave_col0;
        // Remaining extents measured from the wave's origin, columns first (TDM order).
        const uint32_t dims[2] = {
            static_cast<uint32_t>(max(index_t{0}, HeadDim_ - state.wave_col0)),
            static_cast<uint32_t>(max(index_t{0}, valid_tokens - state.wave_row0))};
        const uint64_t strides[2] = {static_cast<uint64_t>(max(token_stride_, index_t{1})), 1};
        auto* lds_ptr =
            lds_window.get_bottom_tensor_view().get_buffer_view().p_data_ + state.lds_elem_offset;
        amd_tdm_load<amd_buffer_coherence_enum::coherence_default>(
            createTDMDescriptor<remove_cvref_t<DataType>, 2>(
                pool_ptr_ + element_offset, lds_ptr, dims, strides, box, config));
    }

    // Ordinary page-local TDM issue through the shared page tile writer. The
    // gather window must be null: no gather, full-tile or shuffled path exists.
    template <typename TDMConfig,
              typename LdsWindow,
              typename GatherIndexView,
              index_t IAccess          = -1,
              bool FullTile            = false,
              bool RematerializeStride = false>
    CK_TILE_DEVICE void tdm_load_to_lds(const TDMConfig& config,
                                        LdsWindow&& lds_window,
                                        const GatherIndexView& gather_index_view,
                                        number<IAccess>                    = {},
                                        bool_constant<FullTile>            = {},
                                        bool_constant<RematerializeStride> = {}) const
    {
        static_assert(IAccess == -1, "page tile writer has no gather access count");
        static_assert(!FullTile, "page tile writer keeps descriptor bounds checks");
        static_assert(is_null_tile_window_v<remove_cvref_t<GatherIndexView>>,
                      "page tile writer supports the ordinary linear page path only");
        (void)gather_index_view;
        if constexpr(KVKind_ != FmhaPageTileKVKind::kLinear)
        {
            tdm_issue_vectorized(config, lds_window);
            return;
        }

        static_assert(WindowLengths{}.at(number<0>{}) <= 128);
        static_assert(WindowLengths{}.at(number<1>{}) <= 64);
        static_assert(Alignment_ > 0 && 64 % Alignment_ == 0);

        // Missing terminal prefetches use physical page zero and a zero-row
        // descriptor. They still issue the scheduled TDM for tensor waits.
        // The origin preconditions (seqlen_k > 0, origin >= 0, 128-aligned) are checked once by
        // check_loop_invariants(). Column 0 holds by construction: both windows are created and
        // moved with column 0, and the issue below uses only origin_[0].

        // The core already supplies the correct ordinary DRAM distribution and window
        // lengths, and this path must FORWARD them instead of rebuilding one. It applies
        // to PageSize_ == 128 ONLY (page == tile): for P256/P1024 the physical page holds
        // several logical tiles, so the pointer must also be rebased by the within-page
        // slice offset (logical_tile_origin % P). That rebase lives in the helper writer,
        // which is why the verbatim path is restricted here.
        if constexpr(PageSize_ == 128)
        {
            static_assert(!std::is_void_v<Distribution>);
            const index_t valid_tokens =
                clamp(page_id_.owned[0].valid_rows, index_t{0}, index_t{128});
            const index_t physical_page =
                valid_tokens > 0 ? page_id_.owned[0].physical_page : index_t{0};
            const long_index_t element_offset =
                static_cast<long_index_t>(physical_page) * page_stride_;
            const DataType* page_ptr = pool_ptr_ + element_offset;
            // The column extent is the full head dimension.
            const auto page_naive = make_naive_tensor_view<address_space_enum::global>(
                page_ptr,
                make_tuple(valid_tokens, index_t{HeadDim_}),
                make_tuple(token_stride_, index_t{1}),
                number<Alignment_>{},
                number<1>{});
            static_assert(decltype(page_naive)::get_num_of_dimension() == 2);
            const auto page_view = pad_tensor_view(
                page_naive, make_tuple(number<128>{}, number<HeadDim_>{}), sequence<true, false>{});
            const auto page_window =
                make_tile_window(page_view, WindowLengths{}, {0, 0}, Distribution{});
            load_tile_tdm(config, lds_window, page_window);
        }
        else
        {
            const long_index_t tile_origin = logical_tile_origin_of(0, origin_[0]);
            FmhaBatchPrefillLinearPageTileWriter<PageSize_, HeadDim_>::Load(
                pool_ptr_,
                page_stride_,
                token_stride_,
                static_cast<index_t>(tile_origin),
                page_id_,
                config,
                lds_window,
                number<Alignment_>{},
                bool_constant<RematerializeStride>{});
        }
    }
};

template <typename T,
          typename Lengths,
          typename OldDistribution,
          index_t Alignment,
          index_t PageSize,
          index_t HeadDim,
          index_t KVKind,
          bool ScalarRowClamp,
          typename NewDistribution>
CK_TILE_DEVICE auto
make_tile_window(const FmhaBatchPrefillQrTdmPageTileWindow<T,
                                                           Lengths,
                                                           OldDistribution,
                                                           Alignment,
                                                           PageSize,
                                                           HeadDim,
                                                           KVKind,
                                                           ScalarRowClamp>& source,
                 const array<index_t, 2>& origin,
                 const NewDistribution&)
{
    return FmhaBatchPrefillQrTdmPageTileWindow<T,
                                               Lengths,
                                               NewDistribution,
                                               Alignment,
                                               PageSize,
                                               HeadDim,
                                               KVKind,
                                               ScalarRowClamp>{source.pool_ptr_,
                                                               source.seqlen_k_,
                                                               source.token_stride_,
                                                               source.page_row_,
                                                               source.page_stride_,
                                                               source.page_capacity_,
                                                               source.total_pages_,
                                                               source.page_id_,
                                                               origin};
}

template <typename T,
          typename Lengths,
          typename Distribution,
          index_t Alignment,
          index_t PageSize,
          index_t HeadDim,
          index_t KVKind,
          bool ScalarRowClamp>
CK_TILE_DEVICE void move_tile_window(
    FmhaBatchPrefillQrTdmPageTileWindow<T,
                                        Lengths,
                                        Distribution,
                                        Alignment,
                                        PageSize,
                                        HeadDim,
                                        KVKind,
                                        ScalarRowClamp>& window,
    const typename FmhaBatchPrefillQrTdmPageTileWindow<T,
                                                       Lengths,
                                                       Distribution,
                                                       Alignment,
                                                       PageSize,
                                                       HeadDim,
                                                       KVKind,
                                                       ScalarRowClamp>::BottomTensorIndex& step)
{
    window.move(step);
}

template <typename TDMConfig,
          typename LdsWindow,
          typename T,
          typename Lengths,
          typename Distribution,
          index_t Alignment,
          index_t PageSize,
          index_t HeadDim,
          index_t KVKind,
          bool ScalarRowClamp>
CK_TILE_DEVICE void load_tile_tdm(const TDMConfig& config,
                                  LdsWindow&& lds_window,
                                  const FmhaBatchPrefillQrTdmPageTileWindow<T,
                                                                            Lengths,
                                                                            Distribution,
                                                                            Alignment,
                                                                            PageSize,
                                                                            HeadDim,
                                                                            KVKind,
                                                                            ScalarRowClamp>& source)
{
    using WindowType = FmhaBatchPrefillQrTdmPageTileWindow<T,
                                                           Lengths,
                                                           Distribution,
                                                           Alignment,
                                                           PageSize,
                                                           HeadDim,
                                                           KVKind,
                                                           ScalarRowClamp>;
    source.tdm_load_to_lds(config,
                           lds_window,
                           make_null_tile_window(WindowType{}.get_window_lengths()),
                           number<-1>{},
                           bool_constant<false>{},
                           bool_constant<false>{});
}

} // namespace ck_tile
