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

// csr2dense / csc2dense / dense2csr / dense2csc: every value type, both index bases, empty
// matrices and the pointer checks; rocsparse_sparse_to_dense for the CSR, CSC and COO
// formats, both index types, both index bases and both orders of the dense matrix.

#include "unit_test_device_conv_common.hpp"

#include <cstdint>

using namespace convut;

namespace
{
    using ConvDense = ConvTest;

    // The test matrix is square (so that a CSC pointer array is also long enough for the
    // checks that read its last element); the dense matrix is column major.
    constexpr I s_m = 7, s_n = 7, s_ld = 9;

    // CSR or CSC arrays of the dense matrix, and the number of entries per row / column.
    template <typename T>
    struct Compressed
    {
        std::vector<I> ptr, ind, per;
        std::vector<T> val;
    };

    template <typename T>
    Compressed<T> compress(const std::vector<T>& dense, bool csc, int base)
    {
        Compressed<T> c;
        c.ptr.push_back(base);
        for(I p = 0; p < s_m; ++p)
        {
            I cnt = 0;
            for(I q = 0; q < s_n; ++q)
            {
                const T& v = dense[csc ? q + p * s_ld : p + q * s_ld];
                if(v != T{})
                {
                    c.ind.push_back(q + base);
                    c.val.push_back(v);
                    ++cnt;
                }
            }
            c.per.push_back(cnt);
            c.ptr.push_back(base + c.ind.size());
        }
        return c;
    }

    template <typename T>
    void run_dense(rocsparse_handle handle)
    {
        std::vector<T> dense(s_ld * s_n, T{});
        for(I j = 0, k = 1; j < s_n; ++j)
            for(I i = 0; i < s_m; ++i)
                if((3 * i + 2 * j) % 4 == 0 || i == j)
                {
                    dense[i + j * s_ld] = make_val<T>(k, 1 - k);
                    ++k;
                }

        const auto csr2dense = pick<T>(
            rocsparse_scsr2dense, rocsparse_dcsr2dense, rocsparse_ccsr2dense, rocsparse_zcsr2dense);
        const auto csc2dense = pick<T>(
            rocsparse_scsc2dense, rocsparse_dcsc2dense, rocsparse_ccsc2dense, rocsparse_zcsc2dense);
        const auto dense2csr = pick<T>(
            rocsparse_sdense2csr, rocsparse_ddense2csr, rocsparse_cdense2csr, rocsparse_zdense2csr);
        const auto dense2csc = pick<T>(
            rocsparse_sdense2csc, rocsparse_ddense2csc, rocsparse_cdense2csc, rocsparse_zdense2csc);

        for(int base : {0, 1})
            for(bool csc : {false, true})
            {
                SCOPED_TRACE("base " + std::to_string(base) + " csc " + std::to_string(csc));
                const auto c   = compress(dense, csc, base);
                const I    nnz = c.ind.size();
                MatDescr   descr(base_of(base));
                DevBuf<I>  d_ptr(c.ptr), d_ind(c.ind), d_per(c.per);
                DevBuf<T>  d_val(c.val), d_a(dense);

                // Sparse to dense; every pointer argument is required.
                const auto to_dense = [&](const T* v, const I* p, const I* i, T* a) {
                    return (csc ? csc2dense : csr2dense)(handle, s_m, s_n, descr, v, p, i, a, s_ld);
                };
                DevBuf<T> o_a(std::vector<T>(s_ld * s_n, make_val<T>(-99)));
                ASSERT_EQ(to_dense(d_val, d_ptr, d_ind, o_a), st_ok);
                const auto got = o_a.read();
                for(I j = 0; j < s_n; ++j)
                    for(I i = 0; i < s_m; ++i)
                        EXPECT_EQ(got[i + j * s_ld], dense[i + j * s_ld]) << i << ", " << j;
                expect_null_args(to_dense, d_val.get(), d_ptr.get(), d_ind.get(), o_a.get());

                // Dense to sparse.
                DevBuf<I>  o_ptr(s_m + 1), o_ind(nnz);
                DevBuf<T>  o_val(nnz);
                const auto to_sparse = [&](const T* a, const I* nz, T* v, I* p, I* i) {
                    return (csc ? dense2csc
                                : dense2csr)(handle, s_m, s_n, descr, a, s_ld, nz, v, p, i);
                };
                ASSERT_EQ(to_sparse(d_a, d_per, o_val, o_ptr, o_ind), st_ok);
                EXPECT_EQ(o_ptr.read(), c.ptr);
                EXPECT_EQ(o_ind.read(nnz), c.ind);
                EXPECT_EQ(o_val.read(nnz), c.val);
                expect_null_args(
                    to_sparse, d_a.get(), d_per.get(), o_val.get(), o_ptr.get(), o_ind.get());
            }

        // Empty matrices.
        MatDescr              descr(base0);
        DevBuf<T>             d_a(16), d_v(16);
        DevBuf<I>             d_p(8), d_c(8), d_nz(8);
        const std::pair<I, I> dims[] = {{0, 4}, {4, 0}};
        for(const auto& [m, n] : dims)
        {
            SCOPED_TRACE("m " + std::to_string(m) + " n " + std::to_string(n));
            const I ld = std::max<I>(m, 1);
            EXPECT_EQ(csr2dense(handle, m, n, descr, d_v, d_p, d_c, d_a, ld), st_ok);
            EXPECT_EQ(csc2dense(handle, m, n, descr, d_v, d_p, d_c, d_a, ld), st_ok);
            EXPECT_EQ(dense2csr(handle, m, n, descr, d_a, ld, d_nz, d_v, d_p, d_c), st_ok);
            EXPECT_EQ(dense2csc(handle, m, n, descr, d_a, ld, d_nz, d_v, d_p, d_c), st_ok);
        }
    }

    TEST_F(ConvDense, Dense)
    {
        run_dense<float>(handle);
        run_dense<double>(handle);
        run_dense<rocsparse_float_complex>(handle);
        run_dense<rocsparse_double_complex>(handle);
    }

    // ------------------------------------------------------------------
    // rocsparse_sparse_to_dense
    // ------------------------------------------------------------------

    constexpr I s_sm = 6, s_sn = 5;

    // P: type of the row / column pointers, J: type of the indices.
    template <typename P, typename J>
    void run_sparse_to_dense(rocsparse_handle handle, rocsparse_format format)
    {
        // Column major, an entry is stored when (3 i + 2 j) % 4 == 0.
        std::vector<float> dense(s_sm * s_sn, 0.0f);
        for(I j = 0; j < s_sn; ++j)
            for(I i = 0; i < s_sm; ++i)
                if((3 * i + 2 * j) % 4 == 0)
                    dense[i + j * s_sm] = 1.0f + i + 10.0f * j;

        const bool csc = format == rocsparse_format_csc;
        for(int base : {0, 1})
            for(rocsparse_order order : {rocsparse_order_column, rocsparse_order_row})
            {
                SCOPED_TRACE("format " + std::to_string(format) + " base " + std::to_string(base)
                             + " index bytes " + std::to_string(sizeof(P)) + "/"
                             + std::to_string(sizeof(J)) + " order " + std::to_string(order));
                std::vector<P>     ptr = {static_cast<P>(base)}; // row or column pointers
                std::vector<J>     ind, rows; // minor indices; COO row indices
                std::vector<float> val;
                for(I p = 0; p < (csc ? s_sn : s_sm); ++p)
                {
                    for(I q = 0; q < (csc ? s_sm : s_sn); ++q)
                    {
                        const float v = dense[csc ? q + p * s_sm : p + q * s_sm];
                        if(v != 0.0f)
                        {
                            ind.push_back(q + base);
                            rows.push_back(p + base);
                            val.push_back(v);
                        }
                    }
                    ptr.push_back(static_cast<P>(ind.size()) + base);
                }
                const int64_t nnz = val.size();
                ASSERT_GT(nnz, 0);

                DevBuf<P>     d_ptr(ptr);
                DevBuf<J>     d_ind(ind), d_rows(rows);
                DevBuf<float> d_val(val);
                const auto    pt = it_of<P>(), jt = it_of<J>();
                const auto    b = base_of(base);
                Spmat         a;
                if(format == rocsparse_format_csr)
                    ASSERT_EQ(rocsparse_create_csr_descr(
                                  &a, s_sm, s_sn, nnz, d_ptr, d_ind, d_val, pt, jt, b, f32),
                              st_ok);
                else if(csc)
                    ASSERT_EQ(rocsparse_create_csc_descr(
                                  &a, s_sm, s_sn, nnz, d_ptr, d_ind, d_val, pt, jt, b, f32),
                              st_ok);
                else // COO has a single index type
                    ASSERT_EQ(rocsparse_create_coo_descr(
                                  &a, s_sm, s_sn, nnz, d_rows, d_ind, d_val, jt, b, f32),
                              st_ok);

                // The dense output, in the requested order.
                const bool         col_order = order == rocsparse_order_column;
                const int64_t      ld        = col_order ? s_sm : s_sn;
                std::vector<float> expected(s_sm * s_sn);
                for(I j = 0; j < s_sn; ++j)
                    for(I i = 0; i < s_sm; ++i)
                        expected[col_order ? i + j * ld : j + i * ld] = dense[i + j * s_sm];
                DevBuf<float> d_dense(std::vector<float>(expected.size(), -99.0f));
                Dnmat         dn;
                ASSERT_EQ(rocsparse_create_dnmat_descr(&dn, s_sm, s_sn, ld, d_dense, f32, order),
                          st_ok);

                size_t         size = 0;
                constexpr auto alg  = rocsparse_sparse_to_dense_alg_default;
                ASSERT_EQ(rocsparse_sparse_to_dense(handle, a, dn, alg, &size, nullptr), st_ok);
                DevBuf<char> buffer(size);
                ASSERT_EQ(rocsparse_sparse_to_dense(handle, a, dn, alg, &size, buffer), st_ok);
                EXPECT_EQ(d_dense.read(), expected);
            }
    }

    TEST_F(ConvDense, SparseToDense)
    {
        for(auto format : {rocsparse_format_csr, rocsparse_format_csc})
        {
            run_sparse_to_dense<int32_t, int32_t>(handle, format);
            run_sparse_to_dense<int64_t, int64_t>(handle, format);
            run_sparse_to_dense<int64_t, int32_t>(handle, format);
        }
        run_sparse_to_dense<int32_t, int32_t>(handle, rocsparse_format_coo);
        run_sparse_to_dense<int64_t, int64_t>(handle, rocsparse_format_coo);
    }
} // namespace
