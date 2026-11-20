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
// formats, both index types, both index bases and both orders of the dense matrix;
// rocsparse_dense_to_sparse and rocsparse_sparse_to_dense with mismatched dimensions or an
// unsupported sparse format.

#include "rocsparse_spmat_descr.hpp"
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

    // rocsparse_dense_to_sparse sizes its buffer from the sparse descriptor and
    // writes with the dense matrix dimensions. Each call differs in only one
    // dimension, so a check of rows alone or columns alone still fails a case.
    void expect_dense_to_sparse_dim_mismatch(rocsparse_handle handle,
                                             rocsparse_format format,
                                             bool             row_mismatch,
                                             bool             sparse_to_dense = false)
    {
        const int64_t sparse_rows = 2;
        const int64_t sparse_cols = 3;
        const int64_t dense_rows  = sparse_rows + (row_mismatch ? 1 : 0);
        const int64_t dense_cols  = sparse_cols + (row_mismatch ? 0 : 1);
        const int32_t sentinel_i  = 0x5a5a5a5a;
        const float   sentinel_v  = 42.0f;
        const size_t  sentinel_b  = 0xA5A5A5A5u;
        const int64_t nnz         = 2;

        const std::vector<int32_t> idx(8, sentinel_i);
        const std::vector<float>   val(8, sentinel_v);
        DevBuf<int32_t>            d_idx(idx);
        DevBuf<int32_t>            d_col(idx);
        DevBuf<float>              d_val(val);
        DevBuf<float>              d_dense(
            std::vector<float>(static_cast<size_t>(dense_rows * dense_cols), 1.0f));

        Dnmat mat_A;
        ASSERT_EQ(rocsparse_create_dnmat_descr(&mat_A,
                                               dense_rows,
                                               dense_cols,
                                               dense_rows,
                                               d_dense.get(),
                                               f32,
                                               rocsparse_order_column),
                  st_ok);

        Spmat mat_B;
        if(format == rocsparse_format_csr)
        {
            ASSERT_EQ(rocsparse_create_csr_descr(&mat_B,
                                                 sparse_rows,
                                                 sparse_cols,
                                                 nnz,
                                                 d_idx.get(),
                                                 d_col.get(),
                                                 d_val.get(),
                                                 i32,
                                                 i32,
                                                 base0,
                                                 f32),
                      st_ok);
        }
        else if(format == rocsparse_format_csc)
        {
            ASSERT_EQ(rocsparse_create_csc_descr(&mat_B,
                                                 sparse_rows,
                                                 sparse_cols,
                                                 nnz,
                                                 d_col.get(),
                                                 d_idx.get(),
                                                 d_val.get(),
                                                 i32,
                                                 i32,
                                                 base0,
                                                 f32),
                      st_ok);
        }
        else if(format == rocsparse_format_coo)
        {
            ASSERT_EQ(rocsparse_create_coo_descr(&mat_B,
                                                 sparse_rows,
                                                 sparse_cols,
                                                 nnz,
                                                 d_idx.get(),
                                                 d_col.get(),
                                                 d_val.get(),
                                                 i32,
                                                 base0,
                                                 f32),
                      st_ok);
        }
        else if(format == rocsparse_format_ell)
        {
            ASSERT_EQ(
                rocsparse_create_ell_descr(
                    &mat_B, sparse_rows, sparse_cols, d_col.get(), d_val.get(), 2, i32, base0, f32),
                st_ok);
        }
        else if(format == rocsparse_format_bell)
        {
            ASSERT_EQ(rocsparse_create_bell_descr(&mat_B,
                                                  sparse_rows,
                                                  sparse_cols,
                                                  dir_r,
                                                  2,
                                                  2,
                                                  d_col.get(),
                                                  d_val.get(),
                                                  i32,
                                                  base0,
                                                  f32),
                      st_ok);
        }
        else
        {
            FAIL() << "unexpected format";
        }

        const int64_t nnz_before       = mat_B.get()->nnz;
        const int64_t ell_cols_before  = mat_B.get()->ell_cols;
        const int64_t ell_width_before = mat_B.get()->ell_width;

        size_t buffer_size = sentinel_b;
        if(sparse_to_dense)
        {
            ASSERT_EQ(rocsparse_sparse_to_dense(handle,
                                                mat_B,
                                                mat_A,
                                                rocsparse_sparse_to_dense_alg_default,
                                                &buffer_size,
                                                nullptr),
                      st_size)
                << "buffer-size query";
        }
        else
        {
            ASSERT_EQ(rocsparse_dense_to_sparse(handle,
                                                mat_A,
                                                mat_B,
                                                rocsparse_dense_to_sparse_alg_default,
                                                &buffer_size,
                                                nullptr),
                      st_size)
                << "buffer-size query";
        }
        EXPECT_EQ(buffer_size, sentinel_b);

        DevBuf<char> temp(64);
        size_t       compute_buffer_size = sentinel_b;
        if(sparse_to_dense)
        {
            ASSERT_EQ(rocsparse_sparse_to_dense(handle,
                                                mat_B,
                                                mat_A,
                                                rocsparse_sparse_to_dense_alg_default,
                                                &compute_buffer_size,
                                                temp.get()),
                      st_size)
                << "compute";
        }
        else
        {
            ASSERT_EQ(rocsparse_dense_to_sparse(handle,
                                                mat_A,
                                                mat_B,
                                                rocsparse_dense_to_sparse_alg_default,
                                                &compute_buffer_size,
                                                temp.get()),
                      st_size)
                << "compute";
        }
        EXPECT_EQ(compute_buffer_size, sentinel_b);

        if(sparse_to_dense)
        {
            ASSERT_EQ(rocsparse_sparse_to_dense(handle,
                                                mat_B,
                                                mat_A,
                                                rocsparse_sparse_to_dense_alg_default,
                                                nullptr,
                                                temp.get()),
                      st_size)
                << "compute with buffer";
        }
        else
        {
            ASSERT_EQ(rocsparse_dense_to_sparse(handle,
                                                mat_A,
                                                mat_B,
                                                rocsparse_dense_to_sparse_alg_default,
                                                nullptr,
                                                temp.get()),
                      st_size)
                << "analysis";
        }

        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(mat_B.get()->nnz, nnz_before);
        EXPECT_EQ(mat_B.get()->ell_cols, ell_cols_before);
        EXPECT_EQ(mat_B.get()->ell_width, ell_width_before);
        EXPECT_EQ(d_idx.read(), idx);
        EXPECT_EQ(d_col.read(), idx);
        EXPECT_EQ(d_val.read(), val);
        if(sparse_to_dense)
        {
            EXPECT_EQ(d_dense.read(),
                      std::vector<float>(static_cast<size_t>(dense_rows * dense_cols), 1.0f));
        }
    }

    TEST_F(ConvDense, DenseToSparseDimMismatchCsr)
    {
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_csr, true);
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_csr, false);
    }

    TEST_F(ConvDense, DenseToSparseDimMismatchCsc)
    {
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_csc, true);
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_csc, false);
    }

    TEST_F(ConvDense, DenseToSparseDimMismatchCoo)
    {
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_coo, true);
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_coo, false);
    }

    // dense_to_sparse has no ELL conversion. This still guards the dimension
    // check, which runs before the not_implemented return.
    TEST_F(ConvDense, DenseToSparseDimMismatchEll)
    {
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_ell, true);
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_ell, false);
    }

    TEST_F(ConvDense, DenseToSparseDimMismatchBell)
    {
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_bell, true);
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_bell, false);
    }

    TEST_F(ConvDense, SparseToDenseDimMismatchCsr)
    {
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_csr, true, true);
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_csr, false, true);
    }

    TEST_F(ConvDense, SparseToDenseDimMismatchCsc)
    {
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_csc, true, true);
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_csc, false, true);
    }

    TEST_F(ConvDense, SparseToDenseDimMismatchCoo)
    {
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_coo, true, true);
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_coo, false, true);
    }

    TEST_F(ConvDense, SparseToDenseDimMismatchEll)
    {
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_ell, true, true);
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_ell, false, true);
    }

    TEST_F(ConvDense, SparseToDenseDimMismatchBell)
    {
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_bell, true, true);
        expect_dense_to_sparse_dim_mismatch(handle, rocsparse_format_bell, false, true);
    }

    // The ELL descriptor matches the dense dimensions, so only the format is unsupported.
    // Every stage must fail and leave buffer_size, the descriptor and its arrays unchanged.
    TEST_F(ConvDense, DenseToSparseEllNotImplemented)
    {
        constexpr auto             alg        = rocsparse_dense_to_sparse_alg_default;
        constexpr auto             st_ni      = rocsparse_status_not_implemented;
        const size_t               sentinel_b = 0xA5A5A5A5u;
        const std::vector<int32_t> idx(6, 0x5a5a5a5a);
        const std::vector<float>   val(6, 42.0f);
        DevBuf<int32_t>            d_col(idx);
        DevBuf<float>              d_val(val);
        DevBuf<float>              d_dense(std::vector<float>(6, 1.0f));

        Dnmat mat_A;
        ASSERT_EQ(rocsparse_create_dnmat_descr(
                      &mat_A, 3, 2, 3, d_dense.get(), f32, rocsparse_order_column),
                  st_ok);
        Spmat mat_B;
        ASSERT_EQ(
            rocsparse_create_ell_descr(&mat_B, 3, 2, d_col.get(), d_val.get(), 2, i32, base0, f32),
            st_ok);

        size_t buffer_size = sentinel_b;
        EXPECT_EQ(rocsparse_dense_to_sparse(handle, mat_A, mat_B, alg, &buffer_size, nullptr),
                  st_ni)
            << "buffer-size query";
        EXPECT_EQ(buffer_size, sentinel_b);

        DevBuf<char> temp(64);
        EXPECT_EQ(rocsparse_dense_to_sparse(handle, mat_A, mat_B, alg, nullptr, temp.get()), st_ni)
            << "analysis";

        size_t compute_buffer_size = sentinel_b;
        EXPECT_EQ(
            rocsparse_dense_to_sparse(handle, mat_A, mat_B, alg, &compute_buffer_size, temp.get()),
            st_ni)
            << "compute";
        EXPECT_EQ(compute_buffer_size, sentinel_b);

        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(mat_B.get()->nnz, 6);
        EXPECT_EQ(mat_B.get()->ell_width, 2);
        EXPECT_EQ(d_col.read(), idx);
        EXPECT_EQ(d_val.read(), val);
    }
} // namespace
