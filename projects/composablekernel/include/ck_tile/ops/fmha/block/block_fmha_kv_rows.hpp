// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// K/V DRAM windows of dense attention advance linearly along the sequence.
struct BlockFmhaDenseKVRows
{
    static constexpr bool kIsPaged      = false;
    static constexpr bool kIsGather     = false;
    static constexpr bool kIsVectorized = false;

    template <typename Window>
    static constexpr index_t kTdmInsts = 1;
};

// Paged K/V (batch_prefill), in one of two modes:
// - page view (kGatherWarpRows == 0): every warp reads its rows of a K/V tile from the page holding
//   them, through a view of that page cut at the sequence end, so TDM zero-fills the rows past
//   seqlen_k instead of reading them. A warp's rows must not straddle two pages.
// - gather (kGatherWarpRows > 0, pages smaller than a warp's rows): the window origin is the
//   sequence row, and load_gather() reads every 8 rows of a warp with one TDM gather whose row
//   indices come from the page table. Rows past seqlen_k get an out-of-range index, which TDM
//   zero-fills. Needs each K/V page stride to be a multiple of its row stride.
// MakeKView/MakeVView(physical_page, rows) return the view of the first `rows` rows of a
// page, with the type of the K/V views the pipeline was called with.
// kPageSize 0 (pagedkv_prefill) takes the page size at run time, and the sequence may then
// start `row_offset` rows into its first page (gappy paged K/V); page view needs the page
// size and row_offset to be multiples of a warp's K/V rows.
// kVectorized: aiter's vectorized KV cache (batch_prefill), see load_vectorized().
// MakeKView/MakeVView then give (rows, hdim) views whose data pointer is the page and whose
// lengths bound the TDM boxes; their strides are not used. With kGatherWarpRows > 0 (pages
// smaller than a warp's rows, a multiple of the vector) each vector of rows is loaded from
// its own page, and the gather row indices hold physical page ids.
template <index_t kPageSize, // 0: page_size at run time
          typename MakeKView,
          typename MakeVView,
          index_t kGatherWarpRows = 0, // gather: K/V rows per warp; 0: page view
          bool kVectorized        = false>
struct BlockFmhaPagedKVRows
{
    static constexpr bool kGather       = kGatherWarpRows > 0;
    static constexpr bool kIsPaged      = true;
    static constexpr bool kIsGather     = kGather;
    static constexpr bool kIsVectorized = kVectorized;
    static constexpr index_t kPageSize_ = kPageSize;
    // One TDM gather takes eight 32-bit row indices; a vectorized KV cache in gather mode
    // issues one box per vector of rows, 8 rows for fp16/bf16.
    static constexpr index_t kGatherRows = 8;

    CK_TILE_HOST_DEVICE BlockFmhaPagedKVRows(const index_t* page_idx_,
                                             index_t max_page_,
                                             index_t seqlen_k_,
                                             MakeKView make_k_view_,
                                             MakeVView make_v_view_,
                                             index_t k_page_rows_ = 0,
                                             index_t v_page_rows_ = 0,
                                             index_t page_size_   = kPageSize,
                                             index_t row_offset_  = 0)
        : page_idx(page_idx_),
          max_page(max_page_),
          seqlen_k(seqlen_k_),
          make_k_view(make_k_view_),
          make_v_view(make_v_view_),
          k_page_rows(k_page_rows_),
          v_page_rows(v_page_rows_),
          page_size(page_size_),
          row_offset(row_offset_),
          page_shift(page_size_ > 0 && (page_size_ & (page_size_ - 1)) == 0
                         ? __builtin_ctz(static_cast<unsigned>(page_size_))
                         : -1)
    {
    }

    const index_t* page_idx; // this sequence's physical page ids
    index_t max_page;        // last valid index into page_idx
    index_t seqlen_k;
    MakeKView make_k_view;
    MakeVView make_v_view;
    // gather: K/V page stride in units of the K/V row stride
    index_t k_page_rows = 0;
    index_t v_page_rows = 0;
    // kPageSize == 0 only: rows per page, and the page row of sequence row 0
    index_t page_size  = kPageSize;
    index_t row_offset = 0;
    // kPageSize == 0: log2 of a power-of-two page size (pages are found by shifts), else -1
    index_t page_shift = -1;

    CK_TILE_DEVICE index_t rows_per_page() const
    {
        if constexpr(kPageSize > 0)
            return kPageSize;
        else
            return page_size;
    }
    // Page index and row within the page of page row `r`.
    CK_TILE_DEVICE index_t page_of(index_t r) const
    {
        if constexpr(kPageSize > 0)
            return r / kPageSize;
        else
            return page_shift >= 0 ? r >> page_shift : r / page_size;
    }
    CK_TILE_DEVICE index_t row_in_page(index_t r) const
    {
        if constexpr(kPageSize > 0)
            return r % kPageSize;
        else
            return page_shift >= 0 ? r & (page_size - 1) : r % page_size;
    }
    // Row of the sequence's pages (from the start of its first page) holding sequence row `row`.
    CK_TILE_DEVICE index_t page_row(index_t row) const
    {
        if constexpr(kPageSize > 0)
            return row;
        else
            return row + row_offset;
    }

    // Consecutive K/V tiles mostly stay in one page: reload only when the page changes.
    mutable index_t cached_page_id = -1;
    mutable index_t cached_page    = 0;

    // gather: lane j holds the row index of this warp's row j of the current K/V tile. Each
    // move_to() also fetches the indices of the tile one move further along (same step), so
    // the page table read of a regular walk lands long before its TDM gathers are issued.
    static_assert(kGatherWarpRows % 8 == 0 && kGatherWarpRows <= 32);
    struct GatherRows
    {
        index_t last_row = 0;  // seq row of this warp's first row, current tile
        index_t next_row = -1; // the same for the prefetched tile
        uint32_t cur     = 0;
        uint32_t next    = 0;
    };
    mutable GatherRows k_gather;
    mutable GatherRows v_gather;

    // TDM instructions one K/V tile load issues per wave (each K/V warp box is
    // kGatherWarpRows rows in gather mode).
    template <typename Window>
    static constexpr index_t kTdmInsts = kGather ? kGatherWarpRows / kGatherRows : 1;

    // Window row of this warp's rows.
    template <typename Window>
    CK_TILE_DEVICE static index_t warp_row_of(const Window& window)
    {
        const auto dstr = window.get_tile_distribution();
        using Dstr      = remove_cvref_t<decltype(dstr)>;
        return __builtin_amdgcn_readfirstlane(
            make_tensor_adaptor_coordinate(
                dstr.get_ps_ys_to_xs_adaptor(),
                container_concat(Dstr::get_partition_index(),
                                 array<index_t, Dstr::get_num_of_dimension_y()>{0}))
                .get_bottom_index()[number<0>{}]);
    }

    // Physical page holding sequence row `row`.
    CK_TILE_DEVICE index_t physical_page(index_t row) const
    {
        return __builtin_amdgcn_readfirstlane(
            page_idx[__builtin_amdgcn_readfirstlane(min(page_of(page_row(row)), max_page))]);
    }

    // Point `window` at sequence row `row`.
    template <bool kIsK, typename Window>
    CK_TILE_DEVICE void move_to(Window& window, index_t row) const
    {
        if constexpr(kGather)
        {
            auto origin         = window.get_window_origin();
            origin(number<0>{}) = row;
            window.set_window_origin(origin);

            const index_t seq_row   = __builtin_amdgcn_readfirstlane(row + warp_row_of(window));
            const index_t page_rows = kIsK ? k_page_rows : v_page_rows;
            auto& g                 = kIsK ? k_gather : v_gather;
            // One vector load reads the page ids of all rows, lane j for row j.
            const auto fetch = [&](index_t first_row) {
                const index_t r    = first_row + static_cast<index_t>(__lane_id());
                const index_t pr   = page_row(r);
                const index_t page = page_idx[min(page_of(pr), max_page)];
                if constexpr(kVectorized)
                {
                    ignore = page_rows;
                    return static_cast<uint32_t>(page);
                }
                else // unsigned: TDM row indices span 32 bits
                    return r < seqlen_k ? static_cast<uint32_t>(page) * page_rows + row_in_page(pr)
                                        : 0xffffffffu;
            };
            if(seq_row == g.next_row)
                g.cur = g.next;
            else
                g.cur = fetch(max(seq_row, index_t{0}));
            g.next_row = __builtin_amdgcn_readfirstlane(2 * seq_row - g.last_row);
            g.last_row = seq_row;
            g.next     = fetch(max(g.next_row, index_t{0}));
        }
        else
        {
            const index_t warp_row      = warp_row_of(window);
            const index_t warp_page_row = page_row(row + warp_row);
            const index_t psize         = rows_per_page();

            // A wave-uniform page index keeps the page table read on the scalar path.
            const index_t page_id =
                __builtin_amdgcn_readfirstlane(min(page_of(warp_page_row), max_page));
            if(page_id != cached_page_id)
            {
                cached_page    = __builtin_amdgcn_readfirstlane(page_idx[page_id]);
                cached_page_id = page_id;
            }
            const index_t page_start = page_id * psize;
            const index_t rows       = max(index_t{0}, min(psize, page_row(seqlen_k) - page_start));

            auto origin         = window.get_window_origin();
            origin(number<0>{}) = warp_page_row - page_start - warp_row;
            if constexpr(kIsK)
                window = make_tile_window(make_k_view(cached_page, rows),
                                          window.get_window_lengths(),
                                          origin,
                                          window.get_tile_distribution());
            else
                window = make_tile_window(make_v_view(cached_page, rows),
                                          window.get_window_lengths(),
                                          origin,
                                          window.get_tile_distribution());
        }
    }

    // gather: TDM-load this warp's rows of the K/V tile at `window` (origin = sequence row,
    // column) into `lds_window`, in the layout (LDS padding included) a box load writes.
    template <bool kIsK, typename LdsWindow, typename Window>
    CK_TILE_DEVICE void
    load_gather(const TDMConfig& config, LdsWindow& lds_window, const Window& window) const
    {
        static_assert(kGather && !kVectorized);
        constexpr auto ys_to_d  = typename Window::TileDstr{}.get_ys_to_d_descriptor();
        constexpr index_t kRows = ys_to_d.get_lengths()[number<0>{}];
        static_assert(ys_to_d.get_num_of_dimension() == 2 && kRows == kGatherWarpRows,
                      "gather: one (kGatherWarpRows x cols) box per warp");
#if CK_TILE_ENABLE_TDM_FEATURE
        using DataType          = remove_cvref_t<typename Window::DataType>;
        constexpr index_t kCols = ys_to_d.get_lengths()[number<1>{}];
        const index_t warp_row  = warp_row_of(window);
        const index_t col       = window.get_window_origin()[number<1>{}];

        const auto& view         = window.get_bottom_tensor_view();
        const auto& desc         = view.get_tensor_descriptor();
        const index_t hdim       = desc.get_length(number<1>{});
        const index_t row_step   = desc.calculate_offset(make_multi_index(1, 0));
        const DataType* gptr     = view.get_buffer_view().p_data_ + col;
        const uint32_t row_index = (kIsK ? k_gather : v_gather).cur;

        const auto& lds_view = lds_window.get_bottom_tensor_view();
        const auto& lds_desc = lds_view.get_tensor_descriptor();

        TDM_GROUP1 g1;
        g1.workgroup_mask = config.workgroup_mask;
        g1.data_size      = sizeof(DataType) == 4 ? 2 : (sizeof(DataType) == 2 ? 1 : 0);
        g1.pad_enable     = config.pad_enable;
        g1.pad_interval   = config.pad_config.pad_interval;
        g1.pad_amount     = config.pad_config.pad_amount;
        g1.tensorDim0(max(index_t{0}, hdim - col));
        // Every real row index is below this bound; 0xffffffff marks a row past seqlen_k.
        g1.tensorDim1(0xffffffffu);
        g1.tensorDim0Stride(row_step);
        g1.tileDim(0, kCols);
        g1.tileDim(1, kGatherRows);
        const auto group1 = amd_wave_read_first_lane(g1.bitfield);

        static_for<0, kRows / kGatherRows, 1>{}([&](auto i_gather) {
            TDM_GROUP2 g2;
            TDM_GROUP3 g3;
            static_for<0, kGatherRows, 1>{}([&](auto j) {
                const uint32_t idx = __builtin_amdgcn_readlane(
                    row_index, static_cast<int>(i_gather * kGatherRows + j));
                if constexpr(j < 4)
                    g2.mode1.setRowIndex32(j, idx);
                else
                    g3.mode1.setRowIndex32(j - 4, idx);
            });

            auto lds_idx         = lds_window.get_window_origin();
            lds_idx(number<0>{}) = lds_idx[number<0>{}] + warp_row + i_gather * kGatherRows;
            CK_TILE_LDS_ADDR DataType* smem =
                lds_view.get_buffer_view().p_data_ + lds_desc.calculate_offset(lds_idx);

            TDM_GROUP0 g0(reinterpret_cast<uintptr_t>(smem),
                          reinterpret_cast<uintptr_t>(gptr),
                          static_cast<uint32_t>(TDMGatherIndexSize::Row32bit_Index),
                          1u);
            int32x8_t zeros = {};
            __builtin_amdgcn_tensor_load_to_lds(
                bit_cast<uint32x4_t>(amd_wave_read_first_lane(g0.bitfield)),
                group1,
                amd_wave_read_first_lane(g2.bitfield),
                amd_wave_read_first_lane(g3.bitfield),
                zeros,
                0);
        });
#else
        ignore = config;
        ignore = lds_window;
        ignore = window;
#endif
    }

    // vectorized: TDM-load this warp's rows of the K/V tile at `window` from aiter's vectorized
    // KV cache (page view: origin = page row, column; gather mode: origin = sequence row,
    // column, one box per vector of rows). A K page is [D/v][page][v] (v = 16 bytes of
    // elements) and lands row-major in LDS, as a box load writes it; a V page is [page/v][D][v]
    // and lands as [rows/v][kN1][v] (no LDS padding), which the pipeline reads with plain
    // loads (see the pipeline's vectorized V read).
    template <bool kIsK, typename LdsWindow, typename Window>
    CK_TILE_DEVICE void
    load_vectorized(const TDMConfig& config, LdsWindow& lds_window, const Window& window) const
    {
        static_assert(kVectorized);
        using DataType          = remove_cvref_t<typename Window::DataType>;
        constexpr index_t kVec  = 16 / sizeof(DataType);
        constexpr auto ys_to_d  = typename Window::TileDstr{}.get_ys_to_d_descriptor();
        constexpr index_t kRows = ys_to_d.get_lengths()[number<0>{}];
        constexpr index_t kCols = ys_to_d.get_lengths()[number<1>{}];
        static_assert(ys_to_d.get_num_of_dimension() == 2 && kRows % kVec == 0 && kCols % kVec == 0,
                      "vectorized: one (rows x cols) box per warp, whole vectors");
        static_assert(!kGather || (kRows == kGatherWarpRows && kVec == kGatherRows),
                      "vectorized gather: one box per vector of a warp's rows");
#if CK_TILE_ENABLE_TDM_FEATURE
        const index_t warp_row = warp_row_of(window);
        const auto origin      = window.get_window_origin();
        const index_t col      = origin[number<1>{}];
        const auto& view       = window.get_bottom_tensor_view();
        const index_t hdim     = view.get_tensor_descriptor().get_length(number<1>{});
        const index_t psize    = rows_per_page();

        const auto& lds_view = lds_window.get_bottom_tensor_view();
        const auto& lds_desc = lds_view.get_tensor_descriptor();

        // One box of `kBoxRows` rows from page row `row` of the page at `page_ptr`, of which
        // `rows_left` are valid, to the LDS rows from the warp's row `lds_row`.
        const auto load_box = [&](auto box_rows,
                                  const DataType* page_ptr,
                                  index_t row,
                                  index_t rows_left,
                                  index_t lds_row) {
            constexpr index_t kBoxRows = decltype(box_rows)::value;
            TDM_GROUP1 g1;
            g1.workgroup_mask = config.workgroup_mask;
            g1.data_size      = sizeof(DataType) == 4 ? 2 : (sizeof(DataType) == 2 ? 1 : 0);
            TDM_GROUP2 g2;
            const DataType* gptr;
            CK_TILE_LDS_ADDR DataType* smem;
            if constexpr(kIsK)
            {
                // (v, D/v, rows) with strides (1, page * v, v): row-major rows of D in LDS
                gptr            = page_ptr + row * kVec + (col / kVec) * psize * kVec;
                g1.pad_enable   = config.pad_enable;
                g1.pad_interval = config.pad_config.pad_interval;
                g1.pad_amount   = config.pad_config.pad_amount;
                g1.tensorDim0(kVec);
                g1.tensorDim1(max(index_t{0}, hdim - col) / kVec);
                g1.tensorDim0Stride(psize * kVec);
                g1.tensorDim1Stride(kVec);
                g1.tileDim(0, kVec);
                g1.tileDim(1, kCols / kVec);
                g1.tileDim(2, kBoxRows);
                g2.mode0.tensor_dim2 = rows_left;
                auto lds_idx         = lds_window.get_window_origin();
                lds_idx(number<0>{}) = lds_idx[number<0>{}] + lds_row;
                smem = lds_view.get_buffer_view().p_data_ + lds_desc.calculate_offset(lds_idx);
            }
            else
            {
                // (v, D, rows/v) with strides (1, v, D * v): [rows/v][kCols][v] in LDS
                gptr          = page_ptr + (row / kVec) * hdim * kVec + col * kVec;
                g1.pad_enable = 0;
                g1.tensorDim0(kVec);
                g1.tensorDim1(max(index_t{0}, hdim - col));
                g1.tensorDim0Stride(kVec);
                g1.tensorDim1Stride(hdim * kVec);
                g1.tileDim(0, kVec);
                g1.tileDim(1, kCols);
                g1.tileDim(2, kBoxRows / kVec);
                g2.mode0.tensor_dim2 = integer_divide_ceil(rows_left, kVec);
                smem                 = lds_view.get_buffer_view().p_data_ +
                       lds_desc.calculate_offset(lds_window.get_window_origin()) + lds_row * kCols;
            }
            TDM_GROUP0 g0(
                reinterpret_cast<uintptr_t>(smem), reinterpret_cast<uintptr_t>(gptr), 0u, 0u);
            TDM_GROUP3 g3;
            int32x8_t zeros = {};
            __builtin_amdgcn_tensor_load_to_lds(
                bit_cast<uint32x4_t>(amd_wave_read_first_lane(g0.bitfield)),
                amd_wave_read_first_lane(g1.bitfield),
                amd_wave_read_first_lane(g2.bitfield),
                amd_wave_read_first_lane(g3.bitfield),
                zeros,
                0);
        };

        if constexpr(kGather)
        {
            const index_t seq_row = origin[number<0>{}] + warp_row;
            const uint32_t pages  = (kIsK ? k_gather : v_gather).cur;
            static_for<0, kRows / kVec, 1>{}([&](auto i_box) {
                const index_t r    = seq_row + i_box * kVec;
                const index_t page = static_cast<index_t>(
                    __builtin_amdgcn_readlane(pages, static_cast<int>(i_box * kVec)));
                const DataType* page_ptr = [&]() {
                    if constexpr(kIsK)
                        return make_k_view(page, psize).get_buffer_view().p_data_;
                    else
                        return make_v_view(page, psize).get_buffer_view().p_data_;
                }();
                load_box(number<kVec>{},
                         page_ptr,
                         row_in_page(page_row(r)),
                         max(index_t{0}, min(kVec, seqlen_k - r)),
                         warp_row + i_box * kVec);
            });
        }
        else
        {
            const index_t row  = origin[number<0>{}] + warp_row; // page row
            const index_t rows = view.get_tensor_descriptor().get_length(number<0>{});
            load_box(number<kRows>{},
                     view.get_buffer_view().p_data_,
                     row,
                     max(index_t{0}, rows - row),
                     warp_row);
        }
#else
        ignore = config;
        ignore = lds_window;
        ignore = window;
#endif
    }
};

} // namespace ck_tile
