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

// check_matrix_csr / check_matrix_gebsr / check_matrix_coo: rows of different lengths (the
// CSR checker picks a lane-group width from the average row length), unsorted block
// columns, and the value types of the COO checker.

#include "unit_test_device_conv_common.hpp"

#include <limits>

using namespace convut;

namespace
{
    using ConvCheck = ConvTest;

    constexpr auto gen_type = rocsparse_matrix_type_general;
    constexpr auto sym_type = rocsparse_matrix_type_symmetric;
    constexpr auto sorted   = rocsparse_storage_mode_sorted;
    constexpr auto unsorted = rocsparse_storage_mode_unsorted;
    constexpr auto ds_ok    = rocsparse_data_status_success;
    constexpr auto nan_f    = std::numeric_limits<float>::quiet_NaN();

    // Buffer size query, then check. call(f, tail...) invokes a routine of the family
    // with the matrix arguments followed by the tail. A rejected matrix (expect_ok =
    // false) must make the check fail; the buffer size query is allowed to reject it.
    template <typename Call, typename SizeFn, typename CheckFn>
    rocsparse_data_status
        run_checker(Call call, SizeFn size_fn, CheckFn check_fn, bool expect_ok = true)
    {
        size_t       size = 0;
        auto         ds   = bad_ds;
        const auto   st1  = call(size_fn, &size);
        DevBuf<char> buffer(size);
        const auto   st2 = call(check_fn, &ds, buffer.get());
        if(expect_ok)
        {
            EXPECT_EQ(st1, st_ok);
            EXPECT_EQ(st2, st_ok);
        }
        else
        {
            EXPECT_REJECTED(st2);
        }
        return ds;
    }

    // ------------------------------------------------------------------
    // CSR
    // ------------------------------------------------------------------
    using Csr = HostCsr<float>;

    // `m` rows with `k` consecutive columns each, row r starting at 3 r.
    Csr make_csr(I m, I n, I k, int base)
    {
        Csr a{m, n, base, {base}, {}, {}};
        for(I r = 0; r < m; ++r)
        {
            for(I j = 0; j < k; ++j)
            {
                a.ind.push_back(base + 3 * r + j);
                a.val.push_back(1.0f + j);
            }
            a.ptr.push_back(base + a.ind.size());
        }
        return a;
    }

    rocsparse_data_status check_csr(rocsparse_handle       handle,
                                    const Csr&             a,
                                    int                    base,
                                    rocsparse_matrix_type  type,
                                    rocsparse_storage_mode storage,
                                    bool                   ok = true)
    {
        DevBuf<I>     d_ptr(a.ptr), d_ind(a.ind);
        DevBuf<float> d_val(a.val);
        const auto    call = [&](auto f, auto... tail) {
            return f(handle,
                     a.m,
                     a.n,
                     a.nnz(),
                     d_val.get(),
                     d_ptr.get(),
                     d_ind.get(),
                     base_of(base),
                     type,
                     rocsparse_fill_mode_lower,
                     storage,
                     tail...);
        };
        return run_checker(
            call, rocsparse_scheck_matrix_csr_buffer_size, rocsparse_scheck_matrix_csr, ok);
    }

    TEST_F(ConvCheck, CheckCsrRowLengths)
    {
        // The average row length is 6, 12, 50 and 100 entries.
        for(I k : {6, 12, 50, 100})
            for(int base : {0, 1})
            {
                SCOPED_TRACE("row length " + std::to_string(k) + " base " + std::to_string(base));
                const I   m = 5, n = 200;
                const Csr good  = make_csr(m, n, k, base);
                const I   last1 = good.ptr[2] - base - 1, last2 = good.ptr[3] - base - 1,
                        last3 = good.ptr[4] - base - 1;
                EXPECT_EQ(check_csr(handle, good, base, gen_type, sorted), ds_ok);
                EXPECT_EQ(check_csr(handle, good, base, gen_type, unsorted), ds_ok);

                Csr a        = good;
                a.val[last2] = nan_f;
                EXPECT_EQ(check_csr(handle, a, base, gen_type, sorted), rocsparse_data_status_nan);
                a.val[last2] = std::numeric_limits<float>::infinity();
                EXPECT_EQ(check_csr(handle, a, base, gen_type, sorted), rocsparse_data_status_inf);
                // A column index past the last column, at the end of a row.
                a            = good;
                a.ind[last1] = base + n;
                EXPECT_EQ(check_csr(handle, a, base, gen_type, sorted),
                          rocsparse_data_status_invalid_index);
                // The last two columns of a row swapped.
                a = good;
                std::swap(a.ind[last3], a.ind[last3 - 1]);
                EXPECT_EQ(check_csr(handle, a, base, gen_type, sorted),
                          rocsparse_data_status_invalid_sorting);
                EXPECT_EQ(check_csr(handle, a, base, gen_type, unsorted), ds_ok);
            }
        // A non-general matrix type needs a square matrix.
        check_csr(handle, make_csr(5, 200, 6, 0), 0, sym_type, sorted, false);
    }

    // ------------------------------------------------------------------
    // GEBSR: 3 x 4 blocks of 2 x 3; block row r has the block columns 0..r+1.
    // ------------------------------------------------------------------
    struct Gebsr
    {
        I                  mb = 3, nb = 4, rdim = 2, cdim = 3;
        std::vector<I>     ptr, ind;
        std::vector<float> val;
    };

    Gebsr make_gebsr(int base)
    {
        Gebsr a;
        a.ptr.push_back(base);
        for(I r = 0; r < a.mb; ++r)
        {
            for(I c = 0; c <= r + 1; ++c)
                a.ind.push_back(base + c);
            a.ptr.push_back(base + a.ind.size());
        }
        a.val.assign(a.ind.size() * a.rdim * a.cdim, 1.0f);
        return a;
    }

    rocsparse_data_status check_gebsr(rocsparse_handle       handle,
                                      const Gebsr&           a,
                                      int                    base,
                                      rocsparse_direction    dir,
                                      rocsparse_matrix_type  type,
                                      rocsparse_storage_mode storage,
                                      bool                   ok = true)
    {
        DevBuf<I>     d_ptr(a.ptr), d_ind(a.ind);
        DevBuf<float> d_val(a.val);
        const auto    call = [&](auto f, auto... tail) {
            return f(handle,
                     dir,
                     a.mb,
                     a.nb,
                     I(a.ind.size()),
                     a.rdim,
                     a.cdim,
                     d_val.get(),
                     d_ptr.get(),
                     d_ind.get(),
                     base_of(base),
                     type,
                     rocsparse_fill_mode_lower,
                     storage,
                     tail...);
        };
        return run_checker(
            call, rocsparse_scheck_matrix_gebsr_buffer_size, rocsparse_scheck_matrix_gebsr, ok);
    }

    TEST_F(ConvCheck, CheckGebsr)
    {
        for(int base : {0, 1})
            for(rocsparse_direction dir : {dir_r, dir_c})
            {
                SCOPED_TRACE("base " + std::to_string(base) + " dir " + std::to_string(dir));
                const Gebsr good  = make_gebsr(base);
                const auto  check = [&](const Gebsr& a, rocsparse_storage_mode s) {
                    return check_gebsr(handle, a, base, dir, gen_type, s);
                };
                EXPECT_EQ(check(good, sorted), ds_ok);
                EXPECT_EQ(check(good, unsorted), ds_ok);

                // Unsorted block columns of the third block row.
                Gebsr a = good;
                std::swap(a.ind[a.ptr[2] - base], a.ind[a.ptr[2] - base + 2]);
                EXPECT_EQ(check(a, sorted), rocsparse_data_status_invalid_sorting);
                EXPECT_EQ(check(a, unsorted), ds_ok);

                a        = good;
                a.val[7] = nan_f;
                EXPECT_EQ(check(a, unsorted), rocsparse_data_status_nan);
                a        = good;
                a.ind[1] = base + a.nb;
                EXPECT_EQ(check(a, sorted), rocsparse_data_status_invalid_index);
            }
        // A non-general matrix needs equal block dimensions and a square block matrix.
        check_gebsr(handle, make_gebsr(0), 0, dir_r, sym_type, sorted, false);
    }

    // ------------------------------------------------------------------
    // COO
    // ------------------------------------------------------------------
    template <typename T>
    rocsparse_data_status check_coo(rocsparse_handle       handle,
                                    I                      m,
                                    I                      n,
                                    const std::vector<T>&  val,
                                    const std::vector<I>&  row,
                                    const std::vector<I>&  col,
                                    int                    base,
                                    rocsparse_matrix_type  type,
                                    rocsparse_storage_mode storage,
                                    bool                   ok = true)
    {
        DevBuf<I>  d_row(row), d_col(col);
        DevBuf<T>  d_val(val);
        const auto call = [&](auto f, auto... tail) {
            return f(handle,
                     m,
                     n,
                     I(val.size()),
                     d_val.get(),
                     d_row.get(),
                     d_col.get(),
                     base_of(base),
                     type,
                     rocsparse_fill_mode_lower,
                     storage,
                     tail...);
        };
        return run_checker(call,
                           pick<T>(rocsparse_scheck_matrix_coo_buffer_size,
                                   rocsparse_dcheck_matrix_coo_buffer_size,
                                   rocsparse_ccheck_matrix_coo_buffer_size,
                                   rocsparse_zcheck_matrix_coo_buffer_size),
                           pick<T>(rocsparse_scheck_matrix_coo,
                                   rocsparse_dcheck_matrix_coo,
                                   rocsparse_ccheck_matrix_coo,
                                   rocsparse_zcheck_matrix_coo),
                           ok);
    }

    template <typename T>
    void run_coo(rocsparse_handle handle)
    {
        const I m = 4, n = 5;
        for(int base : {0, 1})
        {
            SCOPED_TRACE("base " + std::to_string(base));
            // Row-major sorted entries.
            const std::vector<I> row = {base, base, base + 1, base + 2, base + 2, base + 3};
            const std::vector<I> col = {base, base + 3, base + 1, base, base + 4, base + 2};
            std::vector<T>       val;
            for(int k = 0; k < 6; ++k)
                val.push_back(make_val<T>(1 + k, -k));
            const auto check
                = [&](const std::vector<T>& v, const std::vector<I>& c, rocsparse_storage_mode s) {
                      return check_coo<T>(handle, m, n, v, row, c, base, gen_type, s);
                  };
            EXPECT_EQ(check(val, col, sorted), ds_ok);

            auto v = val;
            v[4]   = make_special<T>(true);
            EXPECT_EQ(check(v, col, sorted), rocsparse_data_status_nan);

            auto c = col;
            c[5]   = base + n;
            EXPECT_EQ(check(val, c, sorted), rocsparse_data_status_invalid_index);
            // The entries of the first row swapped.
            c = col;
            std::swap(c[0], c[1]);
            EXPECT_EQ(check(val, c, sorted), rocsparse_data_status_invalid_sorting);
            EXPECT_EQ(check(val, c, unsorted), ds_ok);
        }
    }

    TEST_F(ConvCheck, CheckCoo)
    {
        run_coo<float>(handle);
        run_coo<rocsparse_double_complex>(handle);
        // A non-general matrix type needs a square matrix.
        check_coo<float>(handle, 3, 5, {1.0f, 2.0f}, {0, 1}, {0, 3}, 0, sym_type, sorted, false);
    }
} // namespace
