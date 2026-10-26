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

// HYB matrices: csr2hyb (every partition type, both index bases, the four value types,
// conversion into an already filled HYB matrix), hyb2csr, and
// rocsparse_check_matrix_hyb / _buffer_size on valid and defective matrices.
//
// A HYB matrix is an ELL part (the first ell_width entries of every row) and a COO part
// (the remaining entries). The defects of the check tests are placed in the COO part:
// rocsparse_check_matrix_hyb checks the ELL part, then the COO part, and the second check
// overwrites the data status of the first one, so a defect in the ELL part is not
// reported (library behavior that is not pinned here). Only general matrices are checked:
// the ELL checker rejects the other matrix types with rocsparse_status_invalid_value
// although the HYB documentation lists them. Duplicate entries are not reported by the
// COO checker either and are not tested.

#include "unit_test_device_conv_common.hpp"

#include <algorithm>
#include <map>

using namespace convut;

namespace
{
    using ConvHyb = ConvTest;

    // Row r has 1 + (3 r) % 5 entries at the columns (r + 2 k) % n (sorted, unique).
    template <typename T>
    HostCsr<T> make_csr(int m, int n, int base)
    {
        HostCsr<T> a;
        a.m    = m;
        a.n    = n;
        a.base = base;
        a.ptr.push_back(base);
        for(int r = 0, k = 0; r < m; ++r)
        {
            std::vector<int> cols;
            for(int e = 0; e < std::min(n, 1 + (3 * r) % 5); ++e)
                cols.push_back((r + 2 * e) % n);
            std::sort(cols.begin(), cols.end());
            cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
            for(int c : cols)
            {
                a.ind.push_back(c + base);
                a.val.push_back(make_val<T>(1 + k, -(k % 3)));
                ++k;
            }
            a.ptr.push_back(base + a.ind.size());
        }
        return a;
    }

    template <typename T>
    std::map<std::pair<int, int>, T> to_map(const std::vector<I>& ptr,
                                            const std::vector<I>& ind,
                                            const std::vector<T>& val,
                                            int                   m,
                                            int                   base)
    {
        std::map<std::pair<int, int>, T> out;
        for(int r = 0; r < m; ++r)
            for(I k = ptr[r] - base; k < ptr[r + 1] - base; ++k)
                out[{r, ind[k] - base}] = val[k];
        return out;
    }

    constexpr std::pair<rocsparse_hyb_partition, I> s_partitions[]
        = {{rocsparse_hyb_partition_auto, 0},
           {rocsparse_hyb_partition_user, 0},
           {rocsparse_hyb_partition_user, 2},
           {rocsparse_hyb_partition_user, 5},
           {rocsparse_hyb_partition_max, 0}};

    template <typename T>
    rocsparse_status csr2hyb(rocsparse_handle        handle,
                             const DevCsr<T>&        d,
                             rocsparse_hyb_mat       hyb,
                             rocsparse_hyb_partition part,
                             I                       width)
    {
        return pick<T>(
            rocsparse_scsr2hyb, rocsparse_dcsr2hyb, rocsparse_ccsr2hyb, rocsparse_zcsr2hyb)(
            handle, d.a.m, d.a.n, d.descr, d.val, d.ptr, d.ind, hyb, width, part);
    }

    rocsparse_data_status check_hyb(rocsparse_handle       handle,
                                    rocsparse_hyb_mat      hyb,
                                    int                    base,
                                    rocsparse_storage_mode storage)
    {
        const auto general = rocsparse_matrix_type_general;
        const auto lower   = rocsparse_fill_mode_lower;
        size_t     size    = 0;
        const auto b       = base_of(base);
        EXPECT_EQ(
            rocsparse_check_matrix_hyb_buffer_size(handle, hyb, b, general, lower, storage, &size),
            st_ok);
        DevBuf<char> buffer(size);
        auto         ds = static_cast<rocsparse_data_status>(-1);
        EXPECT_EQ(rocsparse_check_matrix_hyb(handle, hyb, b, general, lower, storage, &ds, buffer),
                  st_ok);
        return ds;
    }

    // Converts the matrix into `hyb`; hyb2csr gives it back and the check accepts it for
    // both storage modes.
    template <typename T>
    void convert_and_verify(rocsparse_handle        handle,
                            const DevCsr<T>&        d,
                            rocsparse_hyb_mat       hyb,
                            rocsparse_hyb_partition part,
                            I                       width)
    {
        const auto& a = d.a;
        SCOPED_TRACE("base " + std::to_string(a.base) + " partition " + std::to_string(part)
                     + " width " + std::to_string(width));
        ASSERT_EQ(csr2hyb(handle, d, hyb, part, width), st_ok);

        size_t size = 0;
        ASSERT_EQ(rocsparse_hyb2csr_buffer_size(handle, d.descr, hyb, d.ptr, &size), st_ok);
        DevBuf<char> buffer(size);
        DevBuf<I>    o_ptr(a.m + 1), o_ind(a.nnz());
        DevBuf<T>    o_val(a.nnz());
        ASSERT_EQ(
            pick<T>(rocsparse_shyb2csr, rocsparse_dhyb2csr, rocsparse_chyb2csr, rocsparse_zhyb2csr)(
                handle, d.descr, hyb, o_val, o_ptr, o_ind, buffer),
            st_ok);
        const auto h_ptr = o_ptr.read();
        ASSERT_EQ(h_ptr[a.m] - a.base, a.nnz());
        EXPECT_EQ(to_map<T>(h_ptr, o_ind.read(), o_val.read(), a.m, a.base),
                  to_map<T>(a.ptr, a.ind, a.val, a.m, a.base));

        for(auto storage : {rocsparse_storage_mode_sorted, rocsparse_storage_mode_unsorted})
            EXPECT_EQ(check_hyb(handle, hyb, a.base, storage), rocsparse_data_status_success);
    }

    template <typename T>
    void run_roundtrip(rocsparse_handle handle)
    {
        for(int base : {0, 1})
        {
            const DevCsr<T> d(make_csr<T>(8, 8, base));
            // Every partition into a new HYB matrix.
            for(const auto& [part, width] : s_partitions)
            {
                HybMat hyb;
                ASSERT_EQ(rocsparse_create_hyb_mat(&hyb), st_ok);
                convert_and_verify<T>(handle, d, hyb, part, width);
            }
            // Converting into a HYB matrix that already holds a matrix replaces it; the old
            // ELL and COO parts are freed and the new ones allocated.
            HybMat hyb;
            ASSERT_EQ(rocsparse_create_hyb_mat(&hyb), st_ok);
            for(const auto& [part, width] : {s_partitions[0], s_partitions[2], s_partitions[0]})
                convert_and_verify<T>(handle, d, hyb, part, width);
        }
    }

    TEST_F(ConvHyb, HybRoundTrip)
    {
        run_roundtrip<float>(handle);
        run_roundtrip<double>(handle);
        run_roundtrip<rocsparse_float_complex>(handle);
        run_roundtrip<rocsparse_double_complex>(handle);
    }

    // 3 x 4: row 0 has one entry, row 1 three, row 2 one. With an ELL width of one the COO
    // part holds the entries 2 and 3 (row 1, the columns 1 and 2).
    template <typename T>
    HostCsr<T> defect_matrix(int base)
    {
        HostCsr<T> a;
        a.m    = 3;
        a.n    = 4;
        a.base = base;
        a.ptr  = {base, base + 1, base + 4, base + 5};
        a.ind  = {base, base, base + 1, base + 2, base + 3};
        for(int k = 0; k < 5; ++k)
            a.val.push_back(make_val<T>(1 + k, -(k % 3)));
        return a;
    }

    template <typename T>
    void run_defects(rocsparse_handle handle)
    {
        for(int base : {0, 1})
        {
            SCOPED_TRACE("base " + std::to_string(base));
            const auto check = [&](const HostCsr<T>& a, rocsparse_storage_mode storage) {
                const DevCsr<T> d(a);
                HybMat          hyb;
                EXPECT_EQ(rocsparse_create_hyb_mat(&hyb), st_ok);
                EXPECT_EQ(csr2hyb(handle, d, hyb, rocsparse_hyb_partition_user, 1), st_ok);
                return check_hyb(handle, hyb, base, storage);
            };
            const auto sorted = rocsparse_storage_mode_sorted;
            auto       a      = defect_matrix<T>(base);
            EXPECT_EQ(check(a, sorted), rocsparse_data_status_success);
            a.val[2] = make_special<T>(true);
            EXPECT_EQ(check(a, sorted), rocsparse_data_status_nan);
            a        = defect_matrix<T>(base);
            a.val[3] = make_special<T>(false);
            EXPECT_EQ(check(a, sorted), rocsparse_data_status_inf);
            // A column index past the last column.
            a        = defect_matrix<T>(base);
            a.ind[3] = base + a.n;
            EXPECT_EQ(check(a, sorted), rocsparse_data_status_invalid_index);
            // Unsorted columns are an error for sorted storage only.
            a = defect_matrix<T>(base);
            std::swap(a.ind[2], a.ind[3]);
            EXPECT_EQ(check(a, sorted), rocsparse_data_status_invalid_sorting);
            EXPECT_EQ(check(a, rocsparse_storage_mode_unsorted), rocsparse_data_status_success);
        }
    }

    TEST_F(ConvHyb, HybDefects)
    {
        run_defects<float>(handle);
        run_defects<rocsparse_double_complex>(handle);
    }
} // namespace
