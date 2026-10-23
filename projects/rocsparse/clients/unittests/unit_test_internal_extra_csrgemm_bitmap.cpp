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

//
// Device unit tests for the csrgemm bitmap path (library/src/extra/rocsparse_csrgemm_bitmap.cpp).
//
// The four stage drivers are called directly, bypassing the row grouping, so a test chooses
// which rows form the group (through the permutation and its offset) and, through the cap of the
// workspace, how many words a pass holds. That reaches what the public API cannot steer: groups
// walked in several passes, rows outside the group left untouched, a workspace sized to the
// group and shared between groups, and, through a limit on its allocations, rows walked in tiles
// as they are when device memory runs out.
//
#include "unit_test_utils.hpp"

#include "../../library/src/extra/rocsparse_csrgemm_bitmap.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using rocsparse_ut::device_vector;
using rocsparse_ut::HandleTest;
using rocsparse_ut::to_host;

namespace
{
    constexpr rocsparse_index_base zero = rocsparse_index_base_zero;
    constexpr rocsparse_index_base one  = rocsparse_index_base_one;

    struct bases
    {
        rocsparse_index_base A, B, D, C;
    };

    constexpr bases base_cases[]
        = {{zero, zero, zero, zero}, {one, one, zero, one}, {one, zero, one, zero}};

    std::string to_string(const bases& b)
    {
        return "bases A" + std::to_string(b.A) + " B" + std::to_string(b.B) + " D"
               + std::to_string(b.D) + " C" + std::to_string(b.C);
    }

    // Passes of five, two and one rows spanning all n columns. A row of full width takes a pass
    // of its own at one; the rows of a band hold far fewer words, so several share a pass even
    // then.
    constexpr int64_t pass_row_widths[] = {5, 2, 1};

    int64_t row_words(int64_t n)
    {
        return (n + 31) / 32;
    }

    // Bytes a word of a pass takes: its bitmap word, and its rank unless the stage is nnz.
    template <typename I>
    size_t word_bytes(bool with_rank)
    {
        return sizeof(uint32_t) + (with_rank ? sizeof(I) : 0);
    }

    // The cap of a workspace whose pass holds width rows of full width.
    template <typename I>
    size_t pass_bytes(int64_t width, int64_t n, bool with_rank = true)
    {
        return static_cast<size_t>(width * row_words(n)) * word_bytes<I>(with_rank);
    }

    // What a check runs a workspace with: passes of width rows of full width, and with
    // tile_words set, an allocation limit under which the pass halves down to tile_words words.
    // Rows wider than that are then walked in tiles, just as when device memory runs out.
    struct pass_setting
    {
        int64_t width;
        int64_t tile_words;
    };

    // An allocation limit that the pass of tile_words words fits and the one twice as large does
    // not.
    template <typename I>
    size_t alloc_bytes(int64_t tile_words, bool with_rank = true)
    {
        return static_cast<size_t>(tile_words) * word_bytes<I>(with_rank) * 3 / 2;
    }

    // Untiled, passes of five, two and one rows of full width. Tiled, tiles of the smallest size
    // there is, which is marked in LDS, and of twice that, marked in global memory; a problem
    // whose rows are no wider than a tile gets none.
    std::vector<pass_setting> pass_settings(bool tiled, int64_t n)
    {
        std::vector<pass_setting> settings;

        if(!tiled)
        {
            for(const int64_t width : pass_row_widths)
            {
                settings.push_back({width, 0});
            }
            return settings;
        }

        for(const int64_t tile_words : {rocsparse::csrgemm_bitmap_tile_words_min,
                                        2 * rocsparse::csrgemm_bitmap_tile_words_min})
        {
            if(row_words(n) > tile_words)
            {
                settings.push_back({5, tile_words});
            }
        }
        return settings;
    }

    std::string to_string(const pass_setting& s)
    {
        return "pass of " + std::to_string(s.width) + " rows"
               + (s.tile_words > 0 ? ", tiles of " + std::to_string(s.tile_words) + " words"
                                   : std::string());
    }

    template <typename I, typename J>
    rocsparse::csrgemm_bitmap_workspace<I, J> make_workspace(rocsparse_handle    handle,
                                                             const pass_setting& s,
                                                             int64_t             n,
                                                             bool                with_rank = true)
    {
        return rocsparse::csrgemm_bitmap_workspace<I, J>(
            handle,
            pass_bytes<I>(s.width, n, with_rank),
            s.tile_words > 0 ? alloc_bytes<I>(s.tile_words, with_rank)
                             : std::numeric_limits<size_t>::max());
    }

    // A tiled setting has to have halved the pass down to its tile.
    template <typename I, typename J>
    void expect_tiles(const rocsparse::csrgemm_bitmap_workspace<I, J>& workspace,
                      const pass_setting&                              s)
    {
        if(s.tile_words > 0)
        {
            EXPECT_EQ(workspace.pass_words, s.tile_words) << "words of a tile";
        }
    }

    template <typename T>
    T value(int re, int im)
    {
        if constexpr(std::is_same_v<
                         T,
                         rocsparse_float_complex> || std::is_same_v<T, rocsparse_double_complex>)
        {
            return T(re, im);
        }
        else
        {
            return static_cast<T>(re);
        }
    }

    using entries = std::vector<std::vector<std::pair<int64_t, int>>>;

    template <typename I, typename J, typename T>
    struct host_csr
    {
        std::vector<I>       ptr;
        std::vector<J>       ind;
        std::vector<T>       val;
        rocsparse_index_base base;
    };

    template <typename I, typename J, typename T>
    host_csr<I, J, T> make_csr(const entries& rows, rocsparse_index_base base)
    {
        host_csr<I, J, T> csr;

        csr.base = base;
        csr.ptr.push_back(static_cast<I>(base));

        for(const auto& row : rows)
        {
            for(const auto& [col, v] : row)
            {
                csr.ind.push_back(static_cast<J>(col + base));
                csr.val.push_back(value<T>(v, v % 3 - 1));
            }

            csr.ptr.push_back(static_cast<I>(csr.ind.size()) + static_cast<I>(base));
        }

        return csr;
    }

    template <typename I, typename J, typename T>
    struct device_csr
    {
        device_vector<I>     ptr;
        device_vector<J>     ind;
        device_vector<T>     val;
        rocsparse_index_base base;

        explicit device_csr(const host_csr<I, J, T>& h)
            : ptr(h.ptr)
            , ind(h.ind)
            , val(h.val)
            , base(h.base)
        {
        }
    };

    // The group is perm[offset, offset + nslot).
    template <typename I, typename J, typename T>
    struct problem
    {
        J                 m;
        J                 n;
        std::vector<J>    perm;
        J                 offset;
        J                 nslot;
        host_csr<I, J, T> A;
        host_csr<I, J, T> B;
        host_csr<I, J, T> D;
        bases             b;

        std::vector<bool> in_group() const
        {
            std::vector<bool> group(m, false);
            for(J slot = 0; slot < nslot; ++slot)
            {
                group[perm[offset + slot]] = true;
            }
            return group;
        }
    };

    // Nine rows of which five, listed out of order at offset 2 of the permutation, form the group.
    // The group holds an empty row of A, products landing on the same column, columns of D inside
    // and outside the products, and columns in the first and last word of a row of 2^20 columns.
    template <typename I, typename J, typename T>
    problem<I, J, T> make_problem(const bases& b)
    {
        const int64_t n = int64_t{1} << 20;

        entries rows_B(4);
        for(int64_t col = 0, i = 0; col < n; col += 37, ++i)
        {
            rows_B[0].push_back({col, static_cast<int>(i % 5) + 1});
        }
        for(int64_t col = 3, i = 0; col < n; col += 41, ++i)
        {
            rows_B[1].push_back({col, static_cast<int>(i % 4) + 1});
        }
        rows_B[2] = {{0, 1}, {1, 2}, {n - 2, 3}, {n - 1, 4}};

        const entries rows_A = {{{0, 1}},
                                {{1, -1}, {2, 4}},
                                {{2, 1}},
                                {{0, 2}},
                                {},
                                {{1, 1}},
                                {{3, 1}},
                                {{0, 2}, {1, 1}},
                                {{0, 1}, {1, 1}, {2, 1}, {3, 5}}};

        const entries rows_D = {{{5, 1}},
                                {},
                                {{0, 1}, {1000, 2}, {n - 1, 3}},
                                {},
                                {{10, 1}, {500000, 2}},
                                {},
                                {},
                                {{0, 4}, {1, 5}, {n - 3, 6}},
                                {}};

        problem<I, J, T> p;

        p.m      = 9;
        p.n      = static_cast<J>(n);
        p.perm   = {0, 3, 7, 1, 4, 8, 2, 5, 6};
        p.offset = 2;
        p.nslot  = 5;
        p.A      = make_csr<I, J, T>(rows_A, b.A);
        p.B      = make_csr<I, J, T>(rows_B, b.B);
        p.D      = make_csr<I, J, T>(rows_D, b.D);
        p.b      = b;

        return p;
    }

    // Twelve rows over 5000 columns, not a whole number of words, of which ten form the group.
    // Every row of B but the last is a band of ten columns at its own offset; the last is empty,
    // the row before it also holds the last column of C, and row 4 continues with 100
    // consecutive columns, so its lanes take several columns across words. The group holds rows
    // spanning one band, two neighbouring bands, the whole width, an empty row of A with only D,
    // a row whose one entry selects the empty row of B, a row of D reaching the last column, and
    // a row of A selecting the same row of B twice.
    template <typename I, typename J, typename T>
    problem<I, J, T> make_band_problem(const bases& b)
    {
        const int64_t n = 5000;

        entries rows_B(11);
        for(int64_t k = 0; k < 10; ++k)
        {
            for(int64_t j = 0; j < 10; ++j)
            {
                rows_B[k].push_back({37 + 480 * k + 3 * j, static_cast<int>((j + k) % 4) + 1});
            }
        }
        rows_B[9].push_back({n - 1, 2});
        for(int64_t col = 1990; col < 2090; ++col)
        {
            rows_B[4].push_back({col, static_cast<int>(col % 3) + 1});
        }

        const entries rows_A = {{{0, 1}},
                                {{1, 2}, {2, 1}},
                                {{2, 3}},
                                {{3, 1}},
                                {},
                                {{10, 1}},
                                {{5, 1}, {5, 2}},
                                {{9, 1}},
                                {{0, 1}, {9, 2}},
                                {{4, 1}},
                                {{6, 2}},
                                {{7, 1}, {8, 1}}};

        const entries rows_D = {{},
                                {{1000, 1}},
                                {},
                                {{n - 1, 2}},
                                {{100, 1}, {200, 3}},
                                {},
                                {},
                                {},
                                {{2000, 1}},
                                {},
                                {{37 + 480 * 6 + 1, 4}},
                                {}};

        problem<I, J, T> p;

        p.m      = 12;
        p.n      = static_cast<J>(n);
        p.perm   = {11, 8, 0, 5, 1, 3, 10, 4, 6, 2, 9, 7};
        p.offset = 1;
        p.nslot  = 10;
        p.A      = make_csr<I, J, T>(rows_A, b.A);
        p.B      = make_csr<I, J, T>(rows_B, b.B);
        p.D      = make_csr<I, J, T>(rows_D, b.D);
        p.b      = b;

        return p;
    }

    // Eight rows over 2^20 columns of which seven form the group, alternating rows marked in LDS
    // with rows marked in global memory so that passes hold both: narrow bands near the start, at
    // column 300000 and near the end, a row across the first two bands, one of full width, one
    // from the first band to the last, an empty row of A with only D, and a band row whose D
    // widens it past LDS. Tiled, the widest rows are walked in tiles and the others still share
    // passes; the row across two bands spans 9351 words, so in tiles of 8192 its products end in
    // a tile narrow enough for LDS.
    template <typename I, typename J, typename T>
    problem<I, J, T> make_mixed_problem(const bases& b)
    {
        const int64_t n = int64_t{1} << 20;

        entries rows_B(4);
        for(int64_t j = 0; j < 40; ++j)
        {
            rows_B[0].push_back({1000 + 3 * j, static_cast<int>(j % 4) + 1});
            rows_B[1].push_back({300000 + 5 * j, static_cast<int>(j % 3) + 1});
        }
        rows_B[2] = {{0, 1}, {n / 2, 2}, {n - 1, 3}};
        rows_B[3] = {{n - 4000, 2}, {n - 3999, 1}, {n - 3000, 4}};

        const entries rows_A = {{{0, 1}},
                                {{0, 2}, {1, 1}},
                                {{2, 1}},
                                {{3, 1}},
                                {},
                                {{0, 1}, {3, 3}},
                                {{1, 1}},
                                {{2, 1}}};

        const entries rows_D = {{{1001, 5}},
                                {{100000, 1}},
                                {{1, 2}},
                                {{n - 1, 1}},
                                {{5000, 1}, {6000, 2}},
                                {},
                                {{150000, 3}},
                                {{7, 1}}};

        problem<I, J, T> p;

        p.m      = 8;
        p.n      = static_cast<J>(n);
        p.perm   = {7, 2, 0, 5, 3, 6, 1, 4};
        p.offset = 1;
        p.nslot  = 7;
        p.A      = make_csr<I, J, T>(rows_A, b.A);
        p.B      = make_csr<I, J, T>(rows_B, b.B);
        p.D      = make_csr<I, J, T>(rows_D, b.D);
        p.b      = b;

        return p;
    }

    template <typename I, typename J, typename T>
    std::vector<problem<I, J, T>> make_problems(const bases& b)
    {
        return {make_problem<I, J, T>(b),
                make_band_problem<I, J, T>(b),
                make_mixed_problem<I, J, T>(b)};
    }

    // Every row in the group, each taking one row of B through its first, middle and last column.
    template <typename I, typename J, typename T>
    problem<I, J, T> make_column_problem(J m, int64_t n)
    {
        const entries rows_A(static_cast<size_t>(m), std::vector<std::pair<int64_t, int>>{{0, 1}});
        const entries rows_B = {{{0, 1}, {n / 2, 2}, {n - 1, 3}}};
        const entries rows_D(static_cast<size_t>(m));

        problem<I, J, T> p;

        p.m      = m;
        p.n      = static_cast<J>(n);
        p.offset = 0;
        p.nslot  = m;
        p.b      = {zero, zero, zero, zero};
        p.A      = make_csr<I, J, T>(rows_A, zero);
        p.B      = make_csr<I, J, T>(rows_B, zero);
        p.D      = make_csr<I, J, T>(rows_D, zero);

        for(J row = 0; row < m; ++row)
        {
            p.perm.push_back(row);
        }

        return p;
    }

    template <typename T>
    struct ref_row
    {
        std::vector<int64_t> cols;
        std::vector<T>       vals;
    };

    // C = alpha * A * B (+ beta * D), row by row, with sorted columns.
    template <typename I, typename J, typename T>
    std::vector<ref_row<T>> reference(const problem<I, J, T>& p, bool add, T alpha, T beta)
    {
        std::vector<ref_row<T>> rows(p.m);
        std::vector<T>          acc(p.n, value<T>(0, 0));
        std::vector<char>       hit(p.n, 0);

        for(J r = 0; r < p.m; ++r)
        {
            std::vector<int64_t> touched;

            const auto put = [&](int64_t col, T v) {
                if(!hit[col])
                {
                    hit[col] = 1;
                    touched.push_back(col);
                }
                acc[col] += v;
            };

            for(I a = p.A.ptr[r] - p.A.base; a < p.A.ptr[r + 1] - p.A.base; ++a)
            {
                const int64_t row_B = p.A.ind[a] - p.A.base;

                for(I k = p.B.ptr[row_B] - p.B.base; k < p.B.ptr[row_B + 1] - p.B.base; ++k)
                {
                    put(p.B.ind[k] - p.B.base, alpha * p.A.val[a] * p.B.val[k]);
                }
            }

            if(add)
            {
                for(I d = p.D.ptr[r] - p.D.base; d < p.D.ptr[r + 1] - p.D.base; ++d)
                {
                    put(p.D.ind[d] - p.D.base, beta * p.D.val[d]);
                }
            }

            std::sort(touched.begin(), touched.end());

            for(const int64_t col : touched)
            {
                rows[r].cols.push_back(col);
                rows[r].vals.push_back(acc[col]);
                acc[col] = value<T>(0, 0);
                hit[col] = 0;
            }
        }

        return rows;
    }

    // C as the drivers must leave it. Rows outside the group hold two sentinel entries that must
    // survive; with filled unset the group rows hold sentinels too, which is what C starts as.
    template <typename I, typename J, typename T>
    struct host_c
    {
        std::vector<I> ptr;
        std::vector<J> ind;
        std::vector<T> val;
    };

    template <typename I, typename J, typename T>
    host_c<I, J, T>
        make_c(const problem<I, J, T>& p, const std::vector<ref_row<T>>& ref, bool filled)
    {
        const std::vector<bool>    group  = p.in_group();
        const rocsparse_index_base base_C = p.b.C;

        host_c<I, J, T> c;

        c.ptr.push_back(static_cast<I>(base_C));

        for(J r = 0; r < p.m; ++r)
        {
            const size_t len = group[r] ? ref[r].cols.size() : 2;

            for(size_t j = 0; j < len; ++j)
            {
                const bool real_entry = group[r] && filled;

                c.ind.push_back(real_entry ? static_cast<J>(ref[r].cols[j] + base_C)
                                           : static_cast<J>(-5));
                c.val.push_back(real_entry ? ref[r].vals[j] : value<T>(99, 7));
            }

            c.ptr.push_back(static_cast<I>(c.ind.size()) + static_cast<I>(base_C));
        }

        return c;
    }

    template <typename V>
    void expect_same(const std::vector<V>& got, const std::vector<V>& want, const char* what)
    {
        ASSERT_EQ(got.size(), want.size()) << what;

        size_t bad   = 0;
        size_t first = 0;

        for(size_t i = 0; i < got.size(); ++i)
        {
            if(!(got[i] == want[i]))
            {
                if(bad++ == 0)
                {
                    first = i;
                }
            }
        }

        if(bad != 0)
        {
            ADD_FAILURE() << what << ": " << bad << " mismatches, first at " << first << ": got "
                          << got[first] << ", want " << want[first];
        }
    }

    template <typename I, typename J, typename T>
    rocsparse_status run_nnz(rocsparse_handle                           handle,
                             rocsparse::csrgemm_bitmap_workspace<I, J>& workspace,
                             const problem<I, J, T>&                    p,
                             bool                                       add,
                             const device_vector<J>&                    d_offset,
                             const device_vector<J>&                    d_perm,
                             const device_csr<I, J, T>&                 A,
                             const device_csr<I, J, T>&                 B,
                             const device_csr<I, J, T>&                 D,
                             I*                                         row_nnz)
    {
        return rocsparse::csrgemm_nnz_bitmap<I, J>(handle,
                                                   workspace,
                                                   p.n,
                                                   p.nslot,
                                                   d_offset.ptr,
                                                   d_perm.ptr,
                                                   A.ptr.ptr,
                                                   A.ind.ptr,
                                                   A.base,
                                                   B.ptr.ptr,
                                                   B.ind.ptr,
                                                   B.base,
                                                   add,
                                                   D.ptr.ptr,
                                                   D.ind.ptr,
                                                   D.base,
                                                   row_nnz);
    }

    template <typename I, typename J, typename T>
    rocsparse_status run_symbolic(rocsparse_handle                           handle,
                                  rocsparse::csrgemm_bitmap_workspace<I, J>& workspace,
                                  const problem<I, J, T>&                    p,
                                  bool                                       add,
                                  const device_vector<J>&                    d_offset,
                                  const device_vector<J>&                    d_perm,
                                  const device_csr<I, J, T>&                 A,
                                  const device_csr<I, J, T>&                 B,
                                  const device_csr<I, J, T>&                 D,
                                  const I*                                   csr_row_ptr_C,
                                  J*                                         csr_col_ind_C)
    {
        return rocsparse::csrgemm_symbolic_bitmap<I, J>(handle,
                                                        workspace,
                                                        p.n,
                                                        p.nslot,
                                                        d_offset.ptr,
                                                        d_perm.ptr,
                                                        A.ptr.ptr,
                                                        A.ind.ptr,
                                                        A.base,
                                                        B.ptr.ptr,
                                                        B.ind.ptr,
                                                        B.base,
                                                        add,
                                                        D.ptr.ptr,
                                                        D.ind.ptr,
                                                        D.base,
                                                        csr_row_ptr_C,
                                                        csr_col_ind_C,
                                                        p.b.C);
    }

    // The stages that write the values of C: calc writes its columns too, numeric takes them as
    // given.
    enum class value_stage
    {
        calc,
        numeric
    };

    template <typename I, typename J, typename T>
    rocsparse_status run_values(rocsparse_handle                           handle,
                                rocsparse::csrgemm_bitmap_workspace<I, J>& workspace,
                                const problem<I, J, T>&                    p,
                                value_stage                                stage,
                                bool                                       add,
                                const T*                                   alpha,
                                const T*                                   beta,
                                const device_vector<J>&                    d_offset,
                                const device_vector<J>&                    d_perm,
                                const device_csr<I, J, T>&                 A,
                                const device_csr<I, J, T>&                 B,
                                const device_csr<I, J, T>&                 D,
                                const I*                                   csr_row_ptr_C,
                                J*                                         csr_col_ind_C,
                                T*                                         csr_val_C)
    {
        if(stage == value_stage::numeric)
        {
            return rocsparse::csrgemm_numeric_bitmap<I, J, T>(handle,
                                                              workspace,
                                                              p.m,
                                                              p.n,
                                                              p.nslot,
                                                              d_offset.ptr,
                                                              d_perm.ptr,
                                                              alpha,
                                                              A.ptr.ptr,
                                                              A.ind.ptr,
                                                              A.val.ptr,
                                                              A.base,
                                                              B.ptr.ptr,
                                                              B.ind.ptr,
                                                              B.val.ptr,
                                                              B.base,
                                                              add,
                                                              beta,
                                                              D.ptr.ptr,
                                                              D.ind.ptr,
                                                              D.val.ptr,
                                                              D.base,
                                                              csr_row_ptr_C,
                                                              csr_col_ind_C,
                                                              csr_val_C,
                                                              p.b.C);
        }

        return rocsparse::csrgemm_calc_bitmap<I, J, T>(handle,
                                                       workspace,
                                                       p.m,
                                                       p.n,
                                                       p.nslot,
                                                       d_offset.ptr,
                                                       d_perm.ptr,
                                                       alpha,
                                                       A.ptr.ptr,
                                                       A.ind.ptr,
                                                       A.val.ptr,
                                                       A.base,
                                                       B.ptr.ptr,
                                                       B.ind.ptr,
                                                       B.val.ptr,
                                                       B.base,
                                                       add,
                                                       beta,
                                                       D.ptr.ptr,
                                                       D.ind.ptr,
                                                       D.val.ptr,
                                                       D.base,
                                                       csr_row_ptr_C,
                                                       csr_col_ind_C,
                                                       csr_val_C,
                                                       p.b.C);
    }

    // The distinct column count of every group row, written at its row and nowhere else.
    template <typename I, typename J>
    void check_nnz(rocsparse_handle handle, bool tiled = false)
    {
        for(const bases& b : base_cases)
        {
            for(const problem<I, J, float>& p : make_problems<I, J, float>(b))
            {
                for(const pass_setting& s : pass_settings(tiled, p.n))
                {
                    for(const bool add : {false, true})
                    {
                        SCOPED_TRACE("n " + std::to_string(p.n) + ", " + to_string(s) + ", "
                                     + to_string(b) + (add ? ", with D" : ", without D"));

                        const auto ref = reference(p, add, 1.0f, 1.0f);

                        device_csr<I, J, float> A(p.A), B(p.B), D(p.D);
                        device_vector<J>        d_perm(p.perm), d_offset(std::vector<J>{p.offset});
                        device_vector<I>        d_row_nnz(std::vector<I>(p.m, static_cast<I>(-7)));
                        auto workspace = make_workspace<I, J>(handle, s, p.n, false);
                        ASSERT_TRUE(d_row_nnz.ptr && d_perm.ptr && d_offset.ptr);

                        ASSERT_EQ(run_nnz(handle,
                                          workspace,
                                          p,
                                          add,
                                          d_offset,
                                          d_perm,
                                          A,
                                          B,
                                          D,
                                          d_row_nnz.ptr),
                                  rocsparse_status_success);
                        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                        expect_tiles(workspace, s);
                        EXPECT_EQ(workspace.rank, nullptr) << "rank index of the nnz stage";

                        const std::vector<bool> group = p.in_group();

                        std::vector<I> want(p.m, static_cast<I>(-7));
                        for(J r = 0; r < p.m; ++r)
                        {
                            if(group[r])
                            {
                                want[r] = static_cast<I>(ref[r].cols.size());
                            }
                        }

                        expect_same(to_host(d_row_nnz), want, "row nnz");
                    }
                }
            }
        }
    }

    template <typename I, typename J>
    void check_symbolic(rocsparse_handle handle, bool tiled = false)
    {
        for(const bases& b : base_cases)
        {
            for(const problem<I, J, float>& p : make_problems<I, J, float>(b))
            {
                for(const pass_setting& s : pass_settings(tiled, p.n))
                {
                    for(const bool add : {false, true})
                    {
                        SCOPED_TRACE("n " + std::to_string(p.n) + ", " + to_string(s) + ", "
                                     + to_string(b) + (add ? ", with D" : ", without D"));

                        const auto                ref     = reference(p, add, 1.0f, 1.0f);
                        const host_c<I, J, float> initial = make_c(p, ref, false);
                        const host_c<I, J, float> want    = make_c(p, ref, true);

                        device_csr<I, J, float> A(p.A), B(p.B), D(p.D);
                        device_vector<J>        d_perm(p.perm), d_offset(std::vector<J>{p.offset});
                        device_vector<I>        d_ptr_C(initial.ptr);
                        device_vector<J>        d_ind_C(initial.ind);
                        auto                    workspace = make_workspace<I, J>(handle, s, p.n);
                        ASSERT_TRUE(d_ptr_C.ptr && d_ind_C.ptr);

                        ASSERT_EQ(run_symbolic(handle,
                                               workspace,
                                               p,
                                               add,
                                               d_offset,
                                               d_perm,
                                               A,
                                               B,
                                               D,
                                               d_ptr_C.ptr,
                                               d_ind_C.ptr),
                                  rocsparse_status_success);
                        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                        expect_tiles(workspace, s);

                        expect_same(to_host(d_ind_C), want.ind, "columns of C");
                    }
                }
            }
        }
    }

    template <typename I, typename J, typename T>
    void check_values(rocsparse_handle handle, value_stage stage, bool tiled = false)
    {
        const T alpha = value<T>(2, 1);
        const T beta  = value<T>(3, -1);

        device_vector<T> d_alpha(std::vector<T>{alpha}), d_beta(std::vector<T>{beta});
        ASSERT_TRUE(d_alpha.ptr && d_beta.ptr);

        for(const bases& b : base_cases)
        {
            for(const problem<I, J, T>& p : make_problems<I, J, T>(b))
            {
                for(const pass_setting& s : pass_settings(tiled, p.n))
                {
                    for(const bool add : {false, true})
                    {
                        for(const rocsparse_pointer_mode mode :
                            {rocsparse_pointer_mode_host, rocsparse_pointer_mode_device})
                        {
                            SCOPED_TRACE("n " + std::to_string(p.n) + ", " + to_string(s) + ", "
                                         + to_string(b) + (add ? ", with D" : ", without D")
                                         + (mode == rocsparse_pointer_mode_host
                                                ? ", host scalars"
                                                : ", device scalars"));

                            const auto            ref     = reference(p, add, alpha, beta);
                            const host_c<I, J, T> initial = make_c(p, ref, false);
                            const host_c<I, J, T> want    = make_c(p, ref, true);

                            device_csr<I, J, T> A(p.A), B(p.B), D(p.D);
                            device_vector<J>    d_perm(p.perm), d_offset(std::vector<J>{p.offset});
                            device_vector<I>    d_ptr_C(initial.ptr);
                            device_vector<J>    d_ind_C(stage == value_stage::numeric ? want.ind
                                                                                      : initial.ind);
                            device_vector<T>    d_val_C(initial.val);
                            auto                workspace = make_workspace<I, J>(handle, s, p.n);
                            ASSERT_TRUE(d_ptr_C.ptr && d_ind_C.ptr && d_val_C.ptr);

                            const bool on_device = mode == rocsparse_pointer_mode_device;

                            ASSERT_EQ(rocsparse_set_pointer_mode(handle, mode),
                                      rocsparse_status_success);

                            const rocsparse_status status
                                = run_values(handle,
                                             workspace,
                                             p,
                                             stage,
                                             add,
                                             on_device ? d_alpha.ptr : &alpha,
                                             on_device ? d_beta.ptr : &beta,
                                             d_offset,
                                             d_perm,
                                             A,
                                             B,
                                             D,
                                             d_ptr_C.ptr,
                                             d_ind_C.ptr,
                                             d_val_C.ptr);

                            ASSERT_EQ(
                                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
                                rocsparse_status_success);
                            ASSERT_EQ(status, rocsparse_status_success);
                            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                            expect_tiles(workspace, s);

                            expect_same(to_host(d_ind_C), want.ind, "columns of C");
                            expect_same(to_host(d_val_C), want.val, "values of C");
                        }
                    }
                }
            }
        }
    }

    // nnz, then symbolic on the same workspace, over a group of m rows of full width
    template <typename I, typename J>
    void run_column_problem(rocsparse_handle                           handle,
                            rocsparse::csrgemm_bitmap_workspace<I, J>& workspace,
                            J                                          m,
                            int64_t                                    n)
    {
        const problem<I, J, float> p       = make_column_problem<I, J, float>(m, n);
        const auto                 ref     = reference(p, false, 1.0f, 1.0f);
        const host_c<I, J, float>  initial = make_c(p, ref, false);
        const host_c<I, J, float>  want    = make_c(p, ref, true);

        device_csr<I, J, float> A(p.A), B(p.B), D(p.D);
        device_vector<J>        d_perm(p.perm), d_offset(std::vector<J>{p.offset});
        device_vector<I>        d_row_nnz(std::vector<I>(p.m, static_cast<I>(-7)));
        device_vector<I>        d_ptr_C(initial.ptr);
        device_vector<J>        d_ind_C(initial.ind);
        ASSERT_TRUE(d_perm.ptr && d_offset.ptr && d_row_nnz.ptr && d_ptr_C.ptr && d_ind_C.ptr);

        ASSERT_EQ(run_nnz(handle, workspace, p, false, d_offset, d_perm, A, B, D, d_row_nnz.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(
            run_symbolic(
                handle, workspace, p, false, d_offset, d_perm, A, B, D, d_ptr_C.ptr, d_ind_C.ptr),
            rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        expect_same(to_host(d_row_nnz), std::vector<I>(p.m, static_cast<I>(3)), "row nnz");
        expect_same(to_host(d_ind_C), want.ind, "columns of C");
    }

    // A pass holds the whole group when the cap allows, the cap when the group needs more, and the
    // widest row when that alone exceeds the cap.
    template <typename I, typename J>
    void check_pass_sizing(rocsparse_handle handle)
    {
        struct shape
        {
            J       m;
            int64_t n;
            size_t  pass_bytes_max;
            int64_t want_words;
        };

        const size_t  cap   = rocsparse::csrgemm_bitmap_pass_bytes_max;
        const int64_t wide  = int64_t{1} << 20;
        const int64_t words = row_words(wide);

        const shape shapes[] = {{1, 100, cap, row_words(100)},
                                {5, 100, cap, 5 * row_words(100)},
                                {200, wide, cap, 200 * words},
                                {200, wide, pass_bytes<I>(5, wide), 5 * words},
                                {3, wide, 1024, words},
                                {1, int64_t{1} << 25, cap, row_words(int64_t{1} << 25)}};

        for(const shape& sh : shapes)
        {
            SCOPED_TRACE("m " + std::to_string(sh.m) + ", n " + std::to_string(sh.n) + ", cap "
                         + std::to_string(sh.pass_bytes_max));

            rocsparse::csrgemm_bitmap_workspace<I, J> workspace(handle, sh.pass_bytes_max);

            run_column_problem<I, J>(handle, workspace, sh.m, sh.n);

            EXPECT_EQ(workspace.pass_words, sh.want_words);
            EXPECT_EQ(reinterpret_cast<uintptr_t>(workspace.bitmap) % 256, 0u);
            EXPECT_EQ(reinterpret_cast<uintptr_t>(workspace.rank) % 256, 0u);
        }
    }

    // One workspace handed groups of growing and shrinking size keeps a pass for the largest.
    template <typename I, typename J>
    void check_shared_workspace(rocsparse_handle handle)
    {
        const int64_t n = int64_t{1} << 20;

        rocsparse::csrgemm_bitmap_workspace<I, J> workspace(handle);

        run_column_problem<I, J>(handle, workspace, J{3}, n);
        EXPECT_EQ(workspace.pass_words, 3 * row_words(n));

        run_column_problem<I, J>(handle, workspace, J{200}, n);
        EXPECT_EQ(workspace.pass_words, 200 * row_words(n));

        run_column_problem<I, J>(handle, workspace, J{1}, 100);
        EXPECT_EQ(workspace.pass_words, 200 * row_words(n));
    }
}

class internal_extra_csrgemm_bitmap : public HandleTest
{
};

TEST_F(internal_extra_csrgemm_bitmap, nnz_i32_i32)
{
    check_nnz<int32_t, int32_t>(handle);
}
TEST_F(internal_extra_csrgemm_bitmap, nnz_i64_i32)
{
    check_nnz<int64_t, int32_t>(handle);
}
TEST_F(internal_extra_csrgemm_bitmap, nnz_i64_i64)
{
    check_nnz<int64_t, int64_t>(handle);
}

TEST_F(internal_extra_csrgemm_bitmap, symbolic_i32_i32)
{
    check_symbolic<int32_t, int32_t>(handle);
}
TEST_F(internal_extra_csrgemm_bitmap, symbolic_i64_i32)
{
    check_symbolic<int64_t, int32_t>(handle);
}
TEST_F(internal_extra_csrgemm_bitmap, symbolic_i64_i64)
{
    check_symbolic<int64_t, int64_t>(handle);
}

TEST_F(internal_extra_csrgemm_bitmap, calc_f32)
{
    check_values<int32_t, int32_t, float>(handle, value_stage::calc);
}
TEST_F(internal_extra_csrgemm_bitmap, calc_f64)
{
    check_values<int32_t, int32_t, double>(handle, value_stage::calc);
}
TEST_F(internal_extra_csrgemm_bitmap, calc_c32)
{
    check_values<int32_t, int32_t, rocsparse_float_complex>(handle, value_stage::calc);
}
TEST_F(internal_extra_csrgemm_bitmap, calc_c64)
{
    check_values<int32_t, int32_t, rocsparse_double_complex>(handle, value_stage::calc);
}
TEST_F(internal_extra_csrgemm_bitmap, calc_f64_i64_i64)
{
    check_values<int64_t, int64_t, double>(handle, value_stage::calc);
}

TEST_F(internal_extra_csrgemm_bitmap, numeric_f32)
{
    check_values<int32_t, int32_t, float>(handle, value_stage::numeric);
}
TEST_F(internal_extra_csrgemm_bitmap, numeric_f64)
{
    check_values<int32_t, int32_t, double>(handle, value_stage::numeric);
}
TEST_F(internal_extra_csrgemm_bitmap, numeric_c32)
{
    check_values<int32_t, int32_t, rocsparse_float_complex>(handle, value_stage::numeric);
}
TEST_F(internal_extra_csrgemm_bitmap, numeric_c64)
{
    check_values<int32_t, int32_t, rocsparse_double_complex>(handle, value_stage::numeric);
}
TEST_F(internal_extra_csrgemm_bitmap, numeric_f64_i64_i64)
{
    check_values<int64_t, int64_t, double>(handle, value_stage::numeric);
}

TEST_F(internal_extra_csrgemm_bitmap, pass_sizing_i32_i32)
{
    check_pass_sizing<int32_t, int32_t>(handle);
}
TEST_F(internal_extra_csrgemm_bitmap, pass_sizing_i64_i64)
{
    check_pass_sizing<int64_t, int64_t>(handle);
}

TEST_F(internal_extra_csrgemm_bitmap, shared_workspace_i32_i32)
{
    check_shared_workspace<int32_t, int32_t>(handle);
}
TEST_F(internal_extra_csrgemm_bitmap, shared_workspace_i64_i32)
{
    check_shared_workspace<int64_t, int32_t>(handle);
}

TEST_F(internal_extra_csrgemm_bitmap, nnz_tiled_i32_i32)
{
    check_nnz<int32_t, int32_t>(handle, true);
}
TEST_F(internal_extra_csrgemm_bitmap, nnz_tiled_i64_i64)
{
    check_nnz<int64_t, int64_t>(handle, true);
}
TEST_F(internal_extra_csrgemm_bitmap, symbolic_tiled_i32_i32)
{
    check_symbolic<int32_t, int32_t>(handle, true);
}
TEST_F(internal_extra_csrgemm_bitmap, symbolic_tiled_i64_i64)
{
    check_symbolic<int64_t, int64_t>(handle, true);
}
TEST_F(internal_extra_csrgemm_bitmap, calc_tiled_f32)
{
    check_values<int32_t, int32_t, float>(handle, value_stage::calc, true);
}
TEST_F(internal_extra_csrgemm_bitmap, calc_tiled_c64)
{
    check_values<int32_t, int32_t, rocsparse_double_complex>(handle, value_stage::calc, true);
}
TEST_F(internal_extra_csrgemm_bitmap, calc_tiled_f64_i64_i64)
{
    check_values<int64_t, int64_t, double>(handle, value_stage::calc, true);
}
TEST_F(internal_extra_csrgemm_bitmap, numeric_tiled_f32)
{
    check_values<int32_t, int32_t, float>(handle, value_stage::numeric, true);
}
TEST_F(internal_extra_csrgemm_bitmap, numeric_tiled_c64)
{
    check_values<int32_t, int32_t, rocsparse_double_complex>(handle, value_stage::numeric, true);
}
TEST_F(internal_extra_csrgemm_bitmap, numeric_tiled_f64_i64_i64)
{
    check_values<int64_t, int64_t, double>(handle, value_stage::numeric, true);
}

// One row through the first, middle and last of 2^36 columns: a span of 2^31 words, past a
// 32-bit count, walked in 512 tiles of 2^22 words of which only the first, the middle and the last
// hold a column. The allocation limit makes the pass halve down to the tile, so the test does not
// depend on how much memory the device has.
TEST_F(internal_extra_csrgemm_bitmap, tiled_row_i64_i64)
{
    using I = int64_t;
    using J = int64_t;
    using T = double;

    const int64_t          n          = int64_t{1} << 36;
    const int64_t          tile_words = int64_t{1} << 22;
    const problem<I, J, T> p          = make_column_problem<I, J, T>(J{1}, n);
    const std::vector<J>   want_ind{0, n / 2, n - 1};
    const std::vector<T>   want_val{1.0, 2.0, 3.0};
    const T                alpha = 1.0;

    device_csr<I, J, T> A(p.A), B(p.B), D(p.D);
    device_vector<J>    d_perm(p.perm), d_offset(std::vector<J>{p.offset});
    device_vector<I>    d_row_nnz(std::vector<I>{-7});
    device_vector<I>    d_ptr_C(std::vector<I>{0, 3});
    device_vector<J>    d_ind_C(std::vector<J>(3, -5));
    device_vector<J>    d_ind_calc(std::vector<J>(3, -5));
    device_vector<T>    d_val_calc(std::vector<T>(3, 99.0));
    device_vector<T>    d_val_numeric(std::vector<T>(3, 99.0));
    ASSERT_TRUE(d_perm.ptr && d_offset.ptr && d_row_nnz.ptr && d_ptr_C.ptr && d_ind_C.ptr
                && d_ind_calc.ptr && d_val_calc.ptr && d_val_numeric.ptr);

    {
        rocsparse::csrgemm_bitmap_workspace<I, J> count_workspace(
            handle, rocsparse::csrgemm_bitmap_pass_bytes_max, alloc_bytes<I>(tile_words, false));

        ASSERT_EQ(
            run_nnz(handle, count_workspace, p, false, d_offset, d_perm, A, B, D, d_row_nnz.ptr),
            rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(count_workspace.pass_words, tile_words);
        expect_same(to_host(d_row_nnz), std::vector<I>{3}, "row nnz");
    }

    rocsparse::csrgemm_bitmap_workspace<I, J> workspace(
        handle, rocsparse::csrgemm_bitmap_pass_bytes_max, alloc_bytes<I>(tile_words));

    ASSERT_EQ(run_symbolic(
                  handle, workspace, p, false, d_offset, d_perm, A, B, D, d_ptr_C.ptr, d_ind_C.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    expect_same(to_host(d_ind_C), want_ind, "columns of C, symbolic");

    ASSERT_EQ(run_values(handle,
                         workspace,
                         p,
                         value_stage::calc,
                         false,
                         &alpha,
                         &alpha,
                         d_offset,
                         d_perm,
                         A,
                         B,
                         D,
                         d_ptr_C.ptr,
                         d_ind_calc.ptr,
                         d_val_calc.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    expect_same(to_host(d_ind_calc), want_ind, "columns of C, calc");
    expect_same(to_host(d_val_calc), want_val, "values of C, calc");

    ASSERT_EQ(run_values(handle,
                         workspace,
                         p,
                         value_stage::numeric,
                         false,
                         &alpha,
                         &alpha,
                         d_offset,
                         d_perm,
                         A,
                         B,
                         D,
                         d_ptr_C.ptr,
                         d_ind_C.ptr,
                         d_val_numeric.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    expect_same(to_host(d_val_numeric), want_val, "values of C, numeric");
}

// Where tiling stops: a pass that cannot hold the smallest tile, and a row that would take more
// than the most tiles, fail for lack of memory; a row of exactly the most tiles is walked. The
// workspace first holds the pass of a row of one tile, so a failure frees it and leaves nothing
// behind.
TEST_F(internal_extra_csrgemm_bitmap, tile_limits_i32_i32)
{
    using I = int32_t;
    using J = int32_t;
    using T = float;

    const int64_t tile_min  = rocsparse::csrgemm_bitmap_tile_words_min;
    const int64_t tiles_max = rocsparse::csrgemm_bitmap_tiles_max;
    const int64_t wide      = tiles_max * 2 * tile_min * 32;

    struct limit_case
    {
        int64_t          n;
        int64_t          tile_words;
        rocsparse_status want;
    };

    const limit_case cases[] = {{int64_t{1} << 20, tile_min / 2, rocsparse_status_memory_error},
                                {wide, tile_min, rocsparse_status_memory_error},
                                {wide, 2 * tile_min, rocsparse_status_success}};

    for(const limit_case& c : cases)
    {
        SCOPED_TRACE("n " + std::to_string(c.n) + ", tiles of " + std::to_string(c.tile_words)
                     + " words");

        rocsparse::csrgemm_bitmap_workspace<I, J> workspace(
            handle, rocsparse::csrgemm_bitmap_pass_bytes_max, alloc_bytes<I>(c.tile_words, false));

        {
            const problem<I, J, T> narrow = make_column_problem<I, J, T>(J{1}, c.tile_words * 32);

            device_csr<I, J, T> A(narrow.A), B(narrow.B), D(narrow.D);
            device_vector<J>    d_perm(narrow.perm), d_offset(std::vector<J>{narrow.offset});
            device_vector<I>    d_row_nnz(std::vector<I>{-7});
            ASSERT_TRUE(d_perm.ptr && d_offset.ptr && d_row_nnz.ptr);

            ASSERT_EQ(
                run_nnz(handle, workspace, narrow, false, d_offset, d_perm, A, B, D, d_row_nnz.ptr),
                rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            ASSERT_EQ(workspace.pass_words, c.tile_words);
            expect_same(to_host(d_row_nnz), std::vector<I>{3}, "row nnz, one tile");
        }

        const problem<I, J, T> p = make_column_problem<I, J, T>(J{1}, c.n);

        device_csr<I, J, T> A(p.A), B(p.B), D(p.D);
        device_vector<J>    d_perm(p.perm), d_offset(std::vector<J>{p.offset});
        device_vector<I>    d_row_nnz(std::vector<I>{-7});
        ASSERT_TRUE(d_perm.ptr && d_offset.ptr && d_row_nnz.ptr);

        EXPECT_EQ(run_nnz(handle, workspace, p, false, d_offset, d_perm, A, B, D, d_row_nnz.ptr),
                  c.want);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        if(c.want == rocsparse_status_success)
        {
            EXPECT_EQ(workspace.pass_words, c.tile_words);
            expect_same(to_host(d_row_nnz), std::vector<I>{3}, "row nnz");
        }
        else
        {
            EXPECT_EQ(workspace.pass_words, 0);
            EXPECT_EQ(workspace.bitmap, nullptr);
            EXPECT_EQ(workspace.rank, nullptr);
            EXPECT_EQ(workspace.rank_scan, nullptr);
            EXPECT_EQ(workspace.rank_scan_size, 0u);
            EXPECT_EQ(hipGetLastError(), hipSuccess);
        }
    }
}

// An empty group returns before it allocates, or touches the permutation or any matrix.
TEST_F(internal_extra_csrgemm_bitmap, empty_group_is_a_no_op)
{
    using I = int32_t;
    using J = int32_t;
    using T = float;

    rocsparse::csrgemm_bitmap_workspace<I, J> none(handle);
    const T                                   alpha = 1.0f;

    EXPECT_EQ((rocsparse::csrgemm_nnz_bitmap<I, J>(handle,
                                                   none,
                                                   J{100},
                                                   J{0},
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   zero,
                                                   nullptr,
                                                   nullptr,
                                                   zero,
                                                   false,
                                                   nullptr,
                                                   nullptr,
                                                   zero,
                                                   nullptr)),
              rocsparse_status_success);
    EXPECT_EQ((rocsparse::csrgemm_symbolic_bitmap<I, J>(handle,
                                                        none,
                                                        J{100},
                                                        J{0},
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        zero,
                                                        nullptr,
                                                        nullptr,
                                                        zero,
                                                        false,
                                                        nullptr,
                                                        nullptr,
                                                        zero,
                                                        nullptr,
                                                        nullptr,
                                                        zero)),
              rocsparse_status_success);
    EXPECT_EQ((rocsparse::csrgemm_calc_bitmap<I, J, T>(handle,
                                                       none,
                                                       J{1},
                                                       J{100},
                                                       J{0},
                                                       nullptr,
                                                       nullptr,
                                                       &alpha,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       zero,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       zero,
                                                       false,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       zero,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       zero)),
              rocsparse_status_success);
    EXPECT_EQ((rocsparse::csrgemm_numeric_bitmap<I, J, T>(handle,
                                                          none,
                                                          J{1},
                                                          J{100},
                                                          J{0},
                                                          nullptr,
                                                          nullptr,
                                                          &alpha,
                                                          nullptr,
                                                          nullptr,
                                                          nullptr,
                                                          zero,
                                                          nullptr,
                                                          nullptr,
                                                          nullptr,
                                                          zero,
                                                          false,
                                                          nullptr,
                                                          nullptr,
                                                          nullptr,
                                                          nullptr,
                                                          zero,
                                                          nullptr,
                                                          nullptr,
                                                          nullptr,
                                                          zero)),
              rocsparse_status_success);

    EXPECT_EQ(none.entry_offset, nullptr);
    EXPECT_EQ(none.bitmap, nullptr);
    EXPECT_EQ(none.pass_words, 0);
}
