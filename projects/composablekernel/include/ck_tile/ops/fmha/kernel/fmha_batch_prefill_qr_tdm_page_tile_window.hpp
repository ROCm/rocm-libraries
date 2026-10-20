// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/detail/fmha_batch_prefill_page_tile.hpp"

#include <type_traits>

namespace ck_tile {

// The page-tile schedule reads page metadata once for each logical tile and supplies
// the same value to its K and V transfers. The window keeps logical tile origins
// separate from physical page addresses and forms ordinary rank-2 page-local TDM
// descriptors through the shared page tile writer.
//
// PageSize is the physical allocation page; the logical tile remains N128. The
// window holds only this wave's owned page metadata, never all fragments.

template <typename DataType_,
          typename WindowLengths_,
          typename Distribution_,
          index_t Alignment_,
          index_t PageSize_,
          index_t HeadDim_>
struct FmhaBatchPrefillQrTdmPageTileWindow
{
    using DataType          = DataType_;
    using WindowLengths     = WindowLengths_;
    using Distribution      = Distribution_;
    using BottomTensorIndex = array<index_t, 2>;
    using PageIds           = FmhaBatchPrefillPageTileIds<PageSize_>;

    static constexpr index_t kPageSize = PageSize_;
    static constexpr index_t kHeadDim  = HeadDim_;
    static constexpr index_t kWaves = FmhaBatchPrefillPageTileTraits<PageSize_, HeadDim_>::kWaves;
    static constexpr index_t kFragmentsPerWave = PageIds::kFragmentsPerWave;
    static constexpr index_t kFragmentRows =
        FmhaBatchPrefillPageTileTraits<PageSize_, HeadDim_>::kFragmentRows;

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

    // Physical page lookup for one owned fragment.
    // P>=128: fragment 0 only, table column t / P.
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
    // P>=128 returns the one shared id; P16 returns this wave's two owned ids.
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

    // ---- P16 page-id queue (P>=128 uses the *_scalar queue below) ----
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

    // ---- P>=128 scalar page-id queue (one shared physical page per logical tile) ----
    // P>=128 only. publish returns valid rows; read returns the raw slot id;
    // complete validates and yields one shared PageIds with that id.
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

    CK_TILE_DEVICE index_t load_page_id_from_lds_scalar(index_t valid_tokens,
                                                        CK_TILE_LDS_ADDR int32_t* id_slot) const
    {
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

    // ---- lean whole-tile TDM issue (P >= 128) ----
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
        static_assert(PageSize_ >= 128, "the lean issue covers whole-tile pages only");
        static_assert(!std::is_void_v<Distribution>);
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
        static_assert(PageSize_ >= 128, "the lean issue covers whole-tile pages only");
        constexpr auto raw_box =
            to_sequence(Distribution{}.get_ys_to_d_descriptor().get_lengths()).reverse();
        static_assert(raw_box.size() == 2);
        constexpr uint16_t box[2] = {static_cast<uint16_t>(raw_box[number<0>{}]),
                                     static_cast<uint16_t>(raw_box[number<1>{}])};

        const index_t valid_tokens = clamp(page_id_.owned[0].valid_rows, index_t{0}, index_t{128});
        const index_t physical_page =
            valid_tokens > 0 ? page_id_.owned[0].physical_page : index_t{0};
        // P > 128 pages hold several tiles; the tile starts at origin % P inside its page.
        const index_t slice_row = PageSize_ > 128 ? origin_[0] % PageSize_ : index_t{0};
        const long_index_t element_offset =
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
          typename NewDistribution>
CK_TILE_DEVICE auto make_tile_window(const FmhaBatchPrefillQrTdmPageTileWindow<T,
                                                                               Lengths,
                                                                               OldDistribution,
                                                                               Alignment,
                                                                               PageSize,
                                                                               HeadDim>& source,
                                     const array<index_t, 2>& origin,
                                     const NewDistribution&)
{
    return FmhaBatchPrefillQrTdmPageTileWindow<T,
                                               Lengths,
                                               NewDistribution,
                                               Alignment,
                                               PageSize,
                                               HeadDim>{source.pool_ptr_,
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
          index_t HeadDim>
CK_TILE_DEVICE void move_tile_window(
    FmhaBatchPrefillQrTdmPageTileWindow<T, Lengths, Distribution, Alignment, PageSize, HeadDim>&
        window,
    const typename FmhaBatchPrefillQrTdmPageTileWindow<T,
                                                       Lengths,
                                                       Distribution,
                                                       Alignment,
                                                       PageSize,
                                                       HeadDim>::BottomTensorIndex& step)
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
          index_t HeadDim>
CK_TILE_DEVICE void load_tile_tdm(const TDMConfig& config,
                                  LdsWindow&& lds_window,
                                  const FmhaBatchPrefillQrTdmPageTileWindow<T,
                                                                            Lengths,
                                                                            Distribution,
                                                                            Alignment,
                                                                            PageSize,
                                                                            HeadDim>& source)
{
    using WindowType =
        FmhaBatchPrefillQrTdmPageTileWindow<T, Lengths, Distribution, Alignment, PageSize, HeadDim>;
    source.tdm_load_to_lds(config,
                           lds_window,
                           make_null_tile_window(WindowType{}.get_window_lengths()),
                           number<-1>{},
                           bool_constant<false>{},
                           bool_constant<false>{});
}

} // namespace ck_tile
