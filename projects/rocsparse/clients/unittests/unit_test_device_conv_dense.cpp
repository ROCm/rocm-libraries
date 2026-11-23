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
// rocsparse_dense_to_sparse and rocsparse_sparse_to_dense with mismatched dimensions in
// either direction, BSR shapes compared in scalar units, or an unsupported sparse format.

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

    // ------------------------------------------------------------------
    // rocsparse_dense_to_sparse and rocsparse_sparse_to_dense: dimension and format checks
    // ------------------------------------------------------------------

    constexpr auto st_ni = rocsparse_status_not_implemented;

    // For BSR, sparse_rows and sparse_cols count blocks.
    struct DimCase
    {
        rocsparse_format format;
        int64_t          dense_rows, dense_cols;
        int64_t          sparse_rows, sparse_cols;
        int64_t          block_dim = 2;
    };

    // A 2x3 sparse matrix with a dense matrix one row or one column larger or smaller. Each
    // case differs in one dimension and one direction, so a check of rows alone, of columns
    // alone, or of one direction alone still fails a case.
    std::vector<DimCase> dim_mismatches(rocsparse_format format)
    {
        return {
            {format, 3, 3, 2, 3}, {format, 2, 4, 2, 3}, {format, 1, 3, 2, 3}, {format, 2, 2, 2, 3}};
    }

    // Sentinel-filled arrays, large enough for every case. A rejected call must leave them
    // unchanged.
    struct DimArrays
    {
        const std::vector<int32_t> idx   = std::vector<int32_t>(8, 0x5a5a5a5a);
        const std::vector<float>   val   = std::vector<float>(8, 42.0f);
        const std::vector<float>   dense = std::vector<float>(24, 1.0f);
        DevBuf<int32_t>            d_idx{idx}, d_col{idx};
        DevBuf<float>              d_val{val}, d_dense{dense};

        void expect_unchanged() const
        {
            EXPECT_EQ(d_idx.read(), idx);
            EXPECT_EQ(d_col.read(), idx);
            EXPECT_EQ(d_val.read(), val);
            EXPECT_EQ(d_dense.read(), dense);
        }
    };

    void create_dim_case(const DimCase& c, DimArrays& a, Dnmat& dense, Spmat& sparse)
    {
        ASSERT_EQ(rocsparse_create_dnmat_descr(&dense,
                                               c.dense_rows,
                                               c.dense_cols,
                                               c.dense_rows,
                                               a.d_dense.get(),
                                               f32,
                                               rocsparse_order_column),
                  st_ok);

        const int64_t    rows = c.sparse_rows, cols = c.sparse_cols, nnz = 2;
        rocsparse_status status = rocsparse_status_invalid_value;
        switch(c.format)
        {
        case rocsparse_format_csr:
            status = rocsparse_create_csr_descr(
                &sparse, rows, cols, nnz, a.d_idx, a.d_col, a.d_val, i32, i32, base0, f32);
            break;
        case rocsparse_format_csc:
            status = rocsparse_create_csc_descr(
                &sparse, rows, cols, nnz, a.d_col, a.d_idx, a.d_val, i32, i32, base0, f32);
            break;
        case rocsparse_format_coo:
            status = rocsparse_create_coo_descr(
                &sparse, rows, cols, nnz, a.d_idx, a.d_col, a.d_val, i32, base0, f32);
            break;
        case rocsparse_format_ell:
            status = rocsparse_create_ell_descr(
                &sparse, rows, cols, a.d_col, a.d_val, 2, i32, base0, f32);
            break;
        case rocsparse_format_bell:
            status = rocsparse_create_bell_descr(&sparse,
                                                 rows,
                                                 cols,
                                                 dir_r,
                                                 c.block_dim,
                                                 c.block_dim,
                                                 a.d_col,
                                                 a.d_val,
                                                 i32,
                                                 base0,
                                                 f32);
            break;
        case rocsparse_format_bsr:
            status = rocsparse_create_bsr_descr(&sparse,
                                                rows,
                                                cols,
                                                nnz,
                                                dir_r,
                                                c.block_dim,
                                                a.d_idx,
                                                a.d_col,
                                                a.d_val,
                                                i32,
                                                i32,
                                                base0,
                                                f32);
            break;
        default:
            break;
        }
        ASSERT_EQ(status, st_ok);
    }

    std::string describe(const DimCase& c)
    {
        return "format " + std::to_string(c.format) + " dense " + std::to_string(c.dense_rows) + "x"
               + std::to_string(c.dense_cols) + " sparse " + std::to_string(c.sparse_rows) + "x"
               + std::to_string(c.sparse_cols) + " block_dim " + std::to_string(c.block_dim);
    }

    // Every stage of rocsparse_dense_to_sparse must return `want` and leave buffer_size,
    // the descriptor and the arrays unchanged.
    void expect_dense_to_sparse(rocsparse_handle handle, const DimCase& c, rocsparse_status want)
    {
        SCOPED_TRACE(describe(c));
        DimArrays a;
        Dnmat     dense;
        Spmat     sparse;
        create_dim_case(c, a, dense, sparse);
        if(::testing::Test::HasFatalFailure())
            return;
        const int64_t nnz = sparse.get()->nnz, ell_cols = sparse.get()->ell_cols,
                      ell_width = sparse.get()->ell_width;

        constexpr auto alg      = rocsparse_dense_to_sparse_alg_default;
        const size_t   sentinel = 0xA5A5A5A5u;
        DevBuf<char>   temp(64);
        size_t         buffer_size = sentinel;
        EXPECT_EQ(rocsparse_dense_to_sparse(handle, dense, sparse, alg, &buffer_size, nullptr),
                  want)
            << "buffer-size query";
        EXPECT_EQ(buffer_size, sentinel);
        EXPECT_EQ(rocsparse_dense_to_sparse(handle, dense, sparse, alg, nullptr, temp.get()), want)
            << "analysis";
        EXPECT_EQ(rocsparse_dense_to_sparse(handle, dense, sparse, alg, &buffer_size, temp.get()),
                  want)
            << "conversion";
        EXPECT_EQ(buffer_size, sentinel);

        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(sparse.get()->nnz, nnz);
        EXPECT_EQ(sparse.get()->ell_cols, ell_cols);
        EXPECT_EQ(sparse.get()->ell_width, ell_width);
        a.expect_unchanged();
    }

    // rocsparse_sparse_to_dense has no analysis stage: without temp_buffer it is the
    // buffer-size query, with it the conversion. A rejected stage must leave buffer_size,
    // the dense matrix and the sparse arrays unchanged.
    void expect_sparse_to_dense(rocsparse_handle handle,
                                const DimCase&   c,
                                rocsparse_status want_buffer_size,
                                rocsparse_status want_conversion)
    {
        SCOPED_TRACE(describe(c));
        DimArrays a;
        Dnmat     dense;
        Spmat     sparse;
        create_dim_case(c, a, dense, sparse);
        if(::testing::Test::HasFatalFailure())
            return;

        constexpr auto alg      = rocsparse_sparse_to_dense_alg_default;
        const size_t   sentinel = 0xA5A5A5A5u;
        DevBuf<char>   temp(64);
        size_t         buffer_size = sentinel;
        EXPECT_EQ(rocsparse_sparse_to_dense(handle, sparse, dense, alg, &buffer_size, nullptr),
                  want_buffer_size)
            << "buffer-size query";
        if(want_buffer_size != st_ok)
        {
            EXPECT_EQ(buffer_size, sentinel);
        }
        EXPECT_EQ(rocsparse_sparse_to_dense(handle, sparse, dense, alg, nullptr, temp.get()),
                  want_conversion)
            << "conversion";

        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        a.expect_unchanged();
    }

    void expect_dim_mismatches(rocsparse_handle handle, rocsparse_format format, bool to_dense)
    {
        for(const DimCase& c : dim_mismatches(format))
        {
            if(to_dense)
                expect_sparse_to_dense(handle, c, st_size, st_size);
            else
                expect_dense_to_sparse(handle, c, st_size);
            if(::testing::Test::HasFatalFailure())
                return;
        }
    }

    TEST_F(ConvDense, DenseToSparseDimMismatchCsr)
    {
        expect_dim_mismatches(handle, rocsparse_format_csr, false);
    }

    TEST_F(ConvDense, DenseToSparseDimMismatchCsc)
    {
        expect_dim_mismatches(handle, rocsparse_format_csc, false);
    }

    TEST_F(ConvDense, DenseToSparseDimMismatchCoo)
    {
        expect_dim_mismatches(handle, rocsparse_format_coo, false);
    }

    // dense_to_sparse has no ELL conversion. This still guards the dimension
    // check, which runs before the not_implemented return.
    TEST_F(ConvDense, DenseToSparseDimMismatchEll)
    {
        expect_dim_mismatches(handle, rocsparse_format_ell, false);
    }

    // The last case is a 3x3 dense matrix with a 4x4 descriptor rounded up to whole blocks.
    TEST_F(ConvDense, DenseToSparseDimMismatchBell)
    {
        expect_dim_mismatches(handle, rocsparse_format_bell, false);
        expect_dense_to_sparse(handle, {rocsparse_format_bell, 3, 3, 4, 4}, st_size);
    }

    TEST_F(ConvDense, SparseToDenseDimMismatchCsr)
    {
        expect_dim_mismatches(handle, rocsparse_format_csr, true);
    }

    TEST_F(ConvDense, SparseToDenseDimMismatchCsc)
    {
        expect_dim_mismatches(handle, rocsparse_format_csc, true);
    }

    TEST_F(ConvDense, SparseToDenseDimMismatchCoo)
    {
        expect_dim_mismatches(handle, rocsparse_format_coo, true);
    }

    TEST_F(ConvDense, SparseToDenseDimMismatchEll)
    {
        expect_dim_mismatches(handle, rocsparse_format_ell, true);
    }

    TEST_F(ConvDense, SparseToDenseDimMismatchBell)
    {
        expect_dim_mismatches(handle, rocsparse_format_bell, true);
        expect_sparse_to_dense(handle, {rocsparse_format_bell, 3, 3, 4, 4}, st_size, st_size);
    }

    // 2x2 blocks of size 2 make a 4x4 matrix. Neither routine converts BSR, so the matching
    // shape fails with not_implemented. The dense matrices that equal the block counts, end
    // in a partial block, or hold a different number of whole blocks are size errors.
    std::vector<DimCase> bsr_mismatches()
    {
        constexpr auto bsr = rocsparse_format_bsr;
        return {{bsr, 2, 2, 2, 2}, {bsr, 5, 4, 2, 2}, {bsr, 4, 5, 2, 2}, {bsr, 6, 4, 2, 2}};
    }

    TEST_F(ConvDense, DenseToSparseBsrScalarShape)
    {
        expect_dense_to_sparse(handle, {rocsparse_format_bsr, 4, 4, 2, 2}, st_ni);
        for(const DimCase& c : bsr_mismatches())
            expect_dense_to_sparse(handle, c, st_size);
    }

    // The buffer-size query of rocsparse_sparse_to_dense succeeds for every format.
    TEST_F(ConvDense, SparseToDenseBsrScalarShape)
    {
        expect_sparse_to_dense(handle, {rocsparse_format_bsr, 4, 4, 2, 2}, st_ok, st_ni);
        for(const DimCase& c : bsr_mismatches())
            expect_sparse_to_dense(handle, c, st_size, st_size);
    }

    // The ELL descriptor matches the dense dimensions, so only the format is unsupported.
    // Every stage must fail and leave buffer_size, the descriptor and its arrays unchanged.
    TEST_F(ConvDense, DenseToSparseEllNotImplemented)
    {
        constexpr auto             alg        = rocsparse_dense_to_sparse_alg_default;
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
