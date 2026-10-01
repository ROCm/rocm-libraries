/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

// Bitmap based path for the csrgemm rows that do not fit an LDS hash table.
//
// Such a row needs an occupancy table over its columns, which LDS cannot hold. Here it lives in
// global memory. One pass of atomic Or marks every product, and a population count over the
// words gives the distinct column count; an exclusive scan of the counts turns a column into its
// offset in C, so columns and values are written in ascending order without anything being
// sorted. The cost is one walk over the products, a second one in the calc stage to accumulate
// the values, plus a few sweeps over the bitmap.
//
// A row's bitmap covers only the span of columns the row can reach, bounded by the first and last
// column of the sorted rows it draws from, so a band of columns costs its width rather than the
// column count of C. The bitmaps of a group are packed one after the other and walked in passes;
// a row whose columns are spread thinly over a wide span still pays for the whole span. A row
// whose span device memory cannot hold whole is walked in tiles of its span, a pass each. Every
// tile walks all the products of the row again, so this is a way to finish, not a fast path.
//
// The path allocates its device memory itself, sized from the group it is handed, so the
// temporary buffer of the stages does not grow for it. This header holds that workspace and the
// declarations of the drivers. The kernels are in csrgemm_bitmap_device.h, and the drivers and the
// workspace are defined and instantiated in rocsparse_csrgemm_bitmap.cpp.

#pragma once

#include "rocsparse_handle.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rocsparse
{
    // Most a pass takes for its bitmap and any rank index when the group's rows need more; a
    // group that needs less gets exactly what it needs. Passes cost a fixed number of launches
    // each and the words are walked once however they are split, so beyond this more passes cost
    // little. The widest row of the group is taken whole even when it needs more, unless device
    // memory cannot hold it.
    inline constexpr size_t csrgemm_bitmap_pass_bytes_max = size_t{256} << 20;

    // Fewest words a pass is halved to when device memory runs out; only the last tile of a row
    // may be narrower. Narrower tiles save too little memory to be worth their launches, and a
    // tile this wide is still marked in LDS.
    inline constexpr int64_t csrgemm_bitmap_tile_words_min = 4096;

    // Most tiles a row is walked in. Each one walks all the products of the row, so a row that
    // would need more fails for lack of memory rather than taking ages.
    inline constexpr int64_t csrgemm_bitmap_tiles_max = 4096;

    // Device memory of the bitmap path, allocated on the handle's stream and freed with the
    // workspace. The layout of a group is sized by its row count, and the bitmap and any rank
    // index of a pass once the spans of its rows are known. A stage keeps one workspace for all the
    // groups it hands to the path, and a pass reuses the blocks of the passes before it when they
    // are large enough. An allocation of a pass larger than
    // alloc_bytes_max fails as if device memory ran out, so that a test can have rows walked in
    // tiles without exhausting device memory.
    template <typename I, typename J>
    class csrgemm_bitmap_workspace
    {
    public:
        explicit csrgemm_bitmap_workspace(rocsparse_handle handle,
                                          size_t           pass_bytes_max
                                          = rocsparse::csrgemm_bitmap_pass_bytes_max,
                                          size_t alloc_bytes_max
                                          = std::numeric_limits<size_t>::max());
        ~csrgemm_bitmap_workspace();

        csrgemm_bitmap_workspace(const csrgemm_bitmap_workspace&)            = delete;
        csrgemm_bitmap_workspace& operator=(const csrgemm_bitmap_workspace&) = delete;

        // The entry offsets, word offsets and first words of nslot rows, and the scan over the
        // offsets.
        rocsparse_status reserve_layout(J nslot);

        // The bitmap of a pass over a group whose rows take total_words words and whose widest
        // row takes row_words, and its rank index when with_rank is set; a pass reserved without
        // one is reallocated for a stage that needs it. When the allocation fails for lack of
        // memory the pass is halved, down to row_words; when even that fails, the rows wider than
        // the pass are walked in tiles of it, and the pass goes on halving down to
        // csrgemm_bitmap_tile_words_min. Out of memory is returned when that fails too, or when
        // the widest row would take more than csrgemm_bitmap_tiles_max tiles.
        rocsparse_status reserve_pass(int64_t row_words, int64_t total_words, bool with_rank);

        I*       entry_offset     = nullptr;
        int64_t* word_offset      = nullptr;
        J*       word_first       = nullptr;
        void*    offset_scan      = nullptr;
        size_t   offset_scan_size = 0;

        int64_t   pass_words     = 0;
        uint32_t* bitmap         = nullptr;
        I*        rank           = nullptr;
        void*     rank_scan      = nullptr;
        size_t    rank_scan_size = 0;

    private:
        rocsparse_handle handle_;
        size_t           pass_bytes_max_;
        size_t           alloc_bytes_max_;
        J                layout_rows_   = 0;
        void*            layout_memory_ = nullptr;
        void*            pass_memory_   = nullptr;
    };

    // Group 10 of the nnz stage: the distinct column count of every row, from the occupancy of its
    // products and, with D, of its row of D.
    template <typename I, typename J>
    rocsparse_status csrgemm_nnz_bitmap(rocsparse_handle                handle,
                                        csrgemm_bitmap_workspace<I, J>& workspace,
                                        J                               n,
                                        J                               nslot,
                                        const J*                        d_group_offset,
                                        const J*                        d_perm,
                                        const I*                        csr_row_ptr_A,
                                        const J*                        csr_col_ind_A,
                                        rocsparse_index_base            idx_base_A,
                                        const I*                        csr_row_ptr_B,
                                        const J*                        csr_col_ind_B,
                                        rocsparse_index_base            idx_base_B,
                                        bool                            add,
                                        const I*                        csr_row_ptr_D,
                                        const J*                        csr_col_ind_D,
                                        rocsparse_index_base            idx_base_D,
                                        I*                              row_nnz);

    // Group 10 of the calc stage, which owes both the columns and the values of C.
    template <typename I, typename J, typename T>
    rocsparse_status csrgemm_calc_bitmap(rocsparse_handle                handle,
                                         csrgemm_bitmap_workspace<I, J>& workspace,
                                         J                               m,
                                         J                               n,
                                         J                               nslot,
                                         const J*                        d_group_offset,
                                         const J*                        d_perm,
                                         const T*                        alpha_device_host,
                                         const I*                        csr_row_ptr_A,
                                         const J*                        csr_col_ind_A,
                                         const T*                        csr_val_A,
                                         rocsparse_index_base            idx_base_A,
                                         const I*                        csr_row_ptr_B,
                                         const J*                        csr_col_ind_B,
                                         const T*                        csr_val_B,
                                         rocsparse_index_base            idx_base_B,
                                         bool                            add,
                                         const T*                        beta_device_host,
                                         const I*                        csr_row_ptr_D,
                                         const J*                        csr_col_ind_D,
                                         const T*                        csr_val_D,
                                         rocsparse_index_base            idx_base_D,
                                         const I*                        csr_row_ptr_C,
                                         J*                              csr_col_ind_C,
                                         T*                              csr_val_C,
                                         rocsparse_index_base            idx_base_C);

    // Group 10 of the symbolic stage. The row pointers of C are known, so only its columns are
    // owed: the calc stage without the values.
    template <typename I, typename J>
    rocsparse_status csrgemm_symbolic_bitmap(rocsparse_handle                handle,
                                             csrgemm_bitmap_workspace<I, J>& workspace,
                                             J                               n,
                                             J                               nslot,
                                             const J*                        d_group_offset,
                                             const J*                        d_perm,
                                             const I*                        csr_row_ptr_A,
                                             const J*                        csr_col_ind_A,
                                             rocsparse_index_base            idx_base_A,
                                             const I*                        csr_row_ptr_B,
                                             const J*                        csr_col_ind_B,
                                             rocsparse_index_base            idx_base_B,
                                             bool                            add,
                                             const I*                        csr_row_ptr_D,
                                             const J*                        csr_col_ind_D,
                                             rocsparse_index_base            idx_base_D,
                                             const I*                        csr_row_ptr_C,
                                             J*                              csr_col_ind_C,
                                             rocsparse_index_base            idx_base_C);

    // Rows of the numeric stage that do not fit an LDS hash table: group 10, and the rows of groups
    // 6 to 9 whose table outgrows LDS for a wide value type. The pattern of C is given, so it is
    // marked instead of being rebuilt from the products: the rank of a column is then its place in
    // C by construction, and the products are walked once, to be accumulated. This relies on C
    // holding the sorted pattern the symbolic stage produced; a product whose column C does not
    // hold is dropped.
    template <typename I, typename J, typename T>
    rocsparse_status csrgemm_numeric_bitmap(rocsparse_handle                handle,
                                            csrgemm_bitmap_workspace<I, J>& workspace,
                                            J                               m,
                                            J                               n,
                                            J                               nslot,
                                            const J*                        d_group_offset,
                                            const J*                        d_perm,
                                            const T*                        alpha_device_host,
                                            const I*                        csr_row_ptr_A,
                                            const J*                        csr_col_ind_A,
                                            const T*                        csr_val_A,
                                            rocsparse_index_base            idx_base_A,
                                            const I*                        csr_row_ptr_B,
                                            const J*                        csr_col_ind_B,
                                            const T*                        csr_val_B,
                                            rocsparse_index_base            idx_base_B,
                                            bool                            add,
                                            const T*                        beta_device_host,
                                            const I*                        csr_row_ptr_D,
                                            const J*                        csr_col_ind_D,
                                            const T*                        csr_val_D,
                                            rocsparse_index_base            idx_base_D,
                                            const I*                        csr_row_ptr_C,
                                            const J*                        csr_col_ind_C,
                                            T*                              csr_val_C,
                                            rocsparse_index_base            idx_base_C);
}
