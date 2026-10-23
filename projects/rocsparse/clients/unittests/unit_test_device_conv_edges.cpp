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

// Conversion routines: inverse permutation, nnz counting of a dense matrix, compression of
// CSR matrices, ELL width, CSR2CSC, and CSR2BSR of matrices without entries. The matrices
// are small and the expectations come from host references that do not depend on the
// wavefront size. The routines that return a count are called with host and device pointer
// modes.

#include "unit_test_device_conv_common.hpp"

#include <cmath>
#include <string>

using namespace convut;

namespace
{
    using ConvEdges = ConvTest;

    // m rows, 2 + (r % 3) entries per row at the columns (r + k) % n, sorted. The values are
    // 0.5 or 2 + k, so that a tolerance of 1 splits them without ambiguity.
    HostCsr<float> make_csr(I m, I n, int base)
    {
        HostCsr<float> a;
        a.m    = m;
        a.n    = n;
        a.base = base;
        a.ptr.push_back(base);
        for(I r = 0; r < m; ++r)
        {
            std::vector<std::pair<I, float>> row;
            for(int k = 0; k < std::min<int>(2 + (r % 3), n); ++k)
                row.emplace_back((r + k) % n,
                                 ((r + k) % 2 == 0) ? (r % 2 ? -2.0f - k : 2.0f + k) : 0.5f);
            std::sort(row.begin(), row.end());
            for(const auto& e : row)
            {
                a.ind.push_back(e.first + base);
                a.val.push_back(e.second);
            }
            a.ptr.push_back(base + a.ind.size());
        }
        return a;
    }

    TEST_F(ConvEdges, InversePermutation)
    {
        // 701 is prime, so i -> (7 i + 3) % 701 is a permutation. More than one block.
        const I n = 701;
        for(int base : {0, 1})
        {
            std::vector<I> p(n), want(n);
            for(I i = 0; i < n; ++i)
            {
                p[i]              = (7 * i + 3) % n + base;
                want[p[i] - base] = i + base;
            }
            DevBuf<I> d_p(p), d_q(n);
            ASSERT_EQ(rocsparse_inverse_permutation(handle, n, d_p, d_q, base_of(base)), st_ok);
            EXPECT_EQ(d_q.read(n), want) << "base " << base;
        }

        DevBuf<I> d_p(1), d_q(1);
        EXPECT_EQ(rocsparse_inverse_permutation(handle, 0, d_p, d_q, base0), st_ok);
        EXPECT_EQ(rocsparse_inverse_permutation(handle, -1, d_p, d_q, base0), st_size);
        EXPECT_EQ(rocsparse_inverse_permutation(
                      handle, 1, d_p, d_q, static_cast<rocsparse_index_base>(2)),
                  st_val);
        expect_null_args(
            [&](I size, const I* p, I* q, rocsparse_index_base base) {
                return rocsparse_inverse_permutation(handle, size, p, q, base);
            },
            I(1),
            d_p.get(),
            d_q.get(),
            base0);
    }

    // rocsparse_snnz: the entries of a dense matrix per row or column and in total.
    TEST_F(ConvEdges, NnzDense)
    {
        for(auto shape : {std::pair<I, I>{6, 5}, {0, 4}, {4, 0}, {0, 0}})
        {
            const I m = shape.first, n = shape.second, ld = m + 1;
            SCOPED_TRACE("m " + std::to_string(m) + " n " + std::to_string(n));
            std::vector<float> A(size_t(ld) * n, 0.0f);
            std::vector<I>     rows(m, 0), cols(n, 0);
            I                  total = 0;
            for(I j = 0; j < n; ++j)
                for(I i = 0; i < m; ++i)
                {
                    if((i * 3 + j * 2) % 4 == 0)
                    {
                        A[i + j * ld] = 1.0f + i + j;
                        ++rows[i];
                        ++cols[j];
                        ++total;
                    }
                }
            DevBuf<float> d_A(A);
            MatDescr      descr;
            for(bool device_mode : {false, true})
                for(rocsparse_direction dir : {dir_r, dir_c})
                {
                    const auto& want = (dir == dir_r) ? rows : cols;
                    DevBuf<I>   d_cnt(std::vector<I>(8, 77));
                    EXPECT_EQ(nnz_via(handle,
                                      device_mode,
                                      [&](I* t) {
                                          return rocsparse_snnz(
                                              handle, dir, m, n, descr, d_A, ld, d_cnt, t);
                                      }),
                              total)
                        << "device " << device_mode << " column " << (dir == dir_c);
                    EXPECT_EQ(d_cnt.read(want.size()), want);
                }
        }

        const I       m = 6, n = 5, ld = 7;
        DevBuf<float> d_A(size_t(ld) * n);
        DevBuf<I>     d_cnt(8), d_total(1);
        MatDescr      descr;
        const auto    nnz = [&](I rows, I lda, const float* A, I* cnt, I* total) {
            return rocsparse_snnz(handle, dir_r, rows, n, descr, A, lda, cnt, total);
        };
        expect_null_args(
            [&](const float* A, I* cnt, I* total) { return nnz(m, ld, A, cnt, total); },
            d_A.get(),
            d_cnt.get(),
            d_total.get());
        EXPECT_EQ(nnz(-1, ld, d_A, d_cnt, d_total), st_size);
        EXPECT_EQ(nnz(m, m - 1, d_A, d_cnt, d_total), st_size);
    }

    // rocsparse_snnz_compress and rocsparse_scsr2csr_compress: the entries with a magnitude
    // above the tolerance are kept.
    TEST_F(ConvEdges, Csr2CsrCompress)
    {
        const float tol = 1.0f;
        const I     m = 13, n = 9;
        for(int base : {0, 1})
        {
            SCOPED_TRACE("base " + std::to_string(base));
            const auto a    = make_csr(m, n, base);
            const auto want = keep_entries(a, base, [&](float v) { return std::abs(v) > tol; });
            ASSERT_GT(want.nnz(), 0);
            ASSERT_LT(want.nnz(), a.nnz());
            std::vector<I> per_row(m);
            for(I r = 0; r < m; ++r)
                per_row[r] = want.ptr[r + 1] - want.ptr[r];
            DevCsr<float> d(a);

            DevBuf<I>  d_per_row(m);
            const auto count = [&](bool device_mode, float t) {
                return nnz_via(handle, device_mode, [&](I* total) {
                    return rocsparse_snnz_compress(
                        handle, m, d.descr, d.val, d.ptr, d_per_row, total, t);
                });
            };
            const auto compress = [&](float t, float* val_c, I* ptr_c, I* ind_c) {
                return rocsparse_scsr2csr_compress(handle,
                                                   m,
                                                   n,
                                                   d.descr,
                                                   d.val,
                                                   d.ptr,
                                                   d.ind,
                                                   a.nnz(),
                                                   d_per_row,
                                                   val_c,
                                                   ptr_c,
                                                   ind_c,
                                                   t);
            };
            for(bool device_mode : {false, true})
            {
                ASSERT_EQ(count(device_mode, tol), want.nnz());
                EXPECT_EQ(d_per_row.read(m), per_row);
            }
            DevBuf<I>     ptr_c(m + 1), ind_c(want.nnz());
            DevBuf<float> val_c(want.nnz());
            ASSERT_EQ(compress(tol, val_c, ptr_c, ind_c), st_ok);
            expect_csr(ptr_c, ind_c, val_c, want);
            // The compressed arrays are required when entries are kept.
            EXPECT_EQ(compress(tol, nullptr, ptr_c, ind_c), st_ptr);
            EXPECT_EQ(compress(tol, val_c, ptr_c, nullptr), st_ptr);

            // A tolerance above every magnitude keeps nothing.
            EXPECT_EQ(count(false, 100.0f), 0);
            EXPECT_EQ(compress(100.0f, nullptr, ptr_c, nullptr), st_ok);
            EXPECT_EQ(ptr_c.read(m + 1), std::vector<I>(m + 1, base));

            // A negative tolerance is invalid.
            I total = 0;
            EXPECT_EQ(
                rocsparse_snnz_compress(handle, m, d.descr, d.val, d.ptr, d_per_row, &total, -1.0f),
                st_val);
        }
    }

    TEST_F(ConvEdges, Csr2CsrCompressEmpty)
    {
        MatDescr  descr;
        DevBuf<I> ptr(std::vector<I>(8, 0)), ind(8), per_row(std::vector<I>(8, 0)), ptr_c(8),
            ind_c(8);
        DevBuf<float> val(8), val_c(8);
        for(auto shape : {std::pair<I, I>{0, 4}, {4, 0}})
            EXPECT_EQ(rocsparse_scsr2csr_compress(handle,
                                                  shape.first,
                                                  shape.second,
                                                  descr,
                                                  val,
                                                  ptr,
                                                  ind,
                                                  0,
                                                  per_row,
                                                  val_c,
                                                  ptr_c,
                                                  ind_c,
                                                  1.0f),
                      st_ok);
        EXPECT_EQ(nnz_via(handle,
                          false,
                          [&](I* total) {
                              return rocsparse_snnz_compress(
                                  handle, 0, descr, val, ptr, per_row, total, 1.0f);
                          }),
                  0);
    }

    TEST_F(ConvEdges, Csr2EllWidth)
    {
        const auto a     = make_csr(11, 8, 0);
        I          width = 0;
        for(I r = 0; r < a.m; ++r)
            width = std::max(width, a.ptr[r + 1] - a.ptr[r]);
        MatDescr  csr_descr, ell_descr;
        DevBuf<I> d_ptr(a.ptr);
        for(bool device_mode : {false, true})
            for(auto rows_want : {std::pair<I, I>{a.m, width}, {0, 0}})
                EXPECT_EQ(nnz_via(handle,
                                  device_mode,
                                  [&](I* w) {
                                      return rocsparse_csr2ell_width(
                                          handle, rows_want.first, csr_descr, d_ptr, ell_descr, w);
                                  }),
                          rows_want.second)
                    << "device " << device_mode << " rows " << rows_want.first;
    }

    // csr2csc, symbolic and numeric action: the column pointers and the row indices sorted
    // by row; the values for the numeric action.
    TEST_F(ConvEdges, Csr2Csc)
    {
        const I m = 17, n = 10;
        for(int base : {0, 1})
        {
            const auto a   = make_csr(m, n, base);
            const I    nnz = a.nnz();

            std::vector<I>     cptr(n + 1, base), rind(nnz);
            std::vector<float> cval(nnz);
            for(I e = 0; e < nnz; ++e)
                ++cptr[a.ind[e] - base + 1];
            for(I c = 0; c < n; ++c)
                cptr[c + 1] += cptr[c] - base;
            std::vector<I> next(cptr.begin(), cptr.end() - 1);
            for(I r = 0; r < m; ++r)
                for(I e = a.ptr[r] - base; e < a.ptr[r + 1] - base; ++e)
                {
                    const I pos = next[a.ind[e] - base]++ - base;
                    rind[pos]   = r + base;
                    cval[pos]   = a.val[e];
                }

            DevCsr<float> d(a);
            for(rocsparse_action action : {rocsparse_action_symbolic, rocsparse_action_numeric})
            {
                const bool numeric = action == rocsparse_action_numeric;
                SCOPED_TRACE("base " + std::to_string(base) + " numeric "
                             + std::to_string(numeric));
                size_t size = 0;
                ASSERT_EQ(
                    rocsparse_csr2csc_buffer_size(handle, m, n, nnz, d.ptr, d.ind, action, &size),
                    st_ok);
                DevBuf<char>  buffer(size);
                DevBuf<I>     d_cptr(n + 1), d_rind(nnz);
                DevBuf<float> d_cval(nnz);
                const auto    run = [&](void* temp) {
                    return rocsparse_scsr2csc(handle,
                                              m,
                                              n,
                                              nnz,
                                              d.val,
                                              d.ptr,
                                              d.ind,
                                              d_cval,
                                              d_rind,
                                              d_cptr,
                                              action,
                                              base_of(base),
                                              temp);
                };
                ASSERT_EQ(run(buffer), st_ok);
                EXPECT_EQ(d_cptr.read(n + 1), cptr);
                EXPECT_EQ(d_rind.read(nnz), rind);
                if(numeric)
                    EXPECT_EQ(d_cval.read(nnz), cval);
                // The temporary buffer is required.
                EXPECT_EQ(run(nullptr), st_ptr);
            }
        }
    }

    TEST_F(ConvEdges, Csr2CscEmpty)
    {
        for(int base : {0, 1})
            for(rocsparse_action action : {rocsparse_action_symbolic, rocsparse_action_numeric})
                for(auto shape : {std::pair<I, I>{4, 5}, {0, 5}})
                {
                    const I m = shape.first, n = shape.second;
                    SCOPED_TRACE("base " + std::to_string(base) + " m " + std::to_string(m));
                    // No entries: the column pointers are all the index base.
                    DevBuf<I> d_ptr(std::vector<I>(m + 1, base)), d_ind(1),
                        d_cptr(std::vector<I>(n + 1, 77)), d_rind(1);
                    DevBuf<float> d_val(1), d_cval(1);
                    size_t        size = 123;
                    ASSERT_EQ(
                        rocsparse_csr2csc_buffer_size(handle, m, n, 0, d_ptr, d_ind, action, &size),
                        st_ok);
                    DevBuf<char> d_buffer(size);
                    ASSERT_EQ(rocsparse_scsr2csc(handle,
                                                 m,
                                                 n,
                                                 0,
                                                 d_val,
                                                 d_ptr,
                                                 d_ind,
                                                 d_cval,
                                                 d_rind,
                                                 d_cptr,
                                                 action,
                                                 base_of(base),
                                                 d_buffer),
                              st_ok);
                    EXPECT_EQ(d_cptr.read(n + 1), std::vector<I>(n + 1, base));
                }
    }

    // csr2bsr_nnz and scsr2bsr of matrices without entries: no block, all the block row
    // pointers are the index base. With entries, the column indices and values of the CSR
    // matrix are required, and the result matches a host reference.
    TEST_F(ConvEdges, Csr2BsrEdges)
    {
        struct Shape
        {
            I m, n, block_dim;
        };
        for(int base : {0, 1})
        {
            MatDescr csr_descr(base_of(base)), bsr_descr(base_of(base));
            for(Shape s :
                {Shape{6, 6, 1}, Shape{6, 6, 2}, Shape{6, 6, 4}, Shape{0, 4, 2}, Shape{4, 0, 2}})
                for(bool device_mode : {false, true})
                {
                    SCOPED_TRACE("base " + std::to_string(base) + " m " + std::to_string(s.m)
                                 + " n " + std::to_string(s.n) + " block_dim "
                                 + std::to_string(s.block_dim) + " device "
                                 + std::to_string(device_mode));
                    const I   mb = (s.m + s.block_dim - 1) / s.block_dim;
                    DevBuf<I> d_ptr(std::vector<I>(s.m + 1, base)), d_ind(1),
                        d_bptr(std::vector<I>(mb + 1, 77)), d_bind(1);
                    EXPECT_EQ(nnz_via(handle,
                                      device_mode,
                                      [&](I* t) {
                                          return rocsparse_csr2bsr_nnz(handle,
                                                                       dir_r,
                                                                       s.m,
                                                                       s.n,
                                                                       csr_descr,
                                                                       d_ptr,
                                                                       d_ind,
                                                                       s.block_dim,
                                                                       bsr_descr,
                                                                       d_bptr,
                                                                       t);
                                      }),
                              0);
                    EXPECT_EQ(d_bptr.read(mb + 1), std::vector<I>(mb + 1, base));
                    DevBuf<float> d_val(1), d_bval(1);
                    EXPECT_EQ(rocsparse_scsr2bsr(handle,
                                                 dir_r,
                                                 s.m,
                                                 s.n,
                                                 csr_descr,
                                                 d_val,
                                                 d_ptr,
                                                 d_ind,
                                                 s.block_dim,
                                                 bsr_descr,
                                                 d_bval,
                                                 d_bptr,
                                                 d_bind),
                              st_ok);
                }

            const I m = 6, n = 6, block_dim = 2, mb = 3;
            auto    a = make_csr(m, n, base);
            for(size_t k = 0; k < a.val.size(); ++k)
                a.val[k] = 1.0f + k;
            // Host BSR reference with row-major blocks (block_dim divides m and n).
            std::vector<float> dense(m * n, 0.0f), bval;
            for(I r = 0; r < m; ++r)
                for(I k = a.ptr[r] - base; k < a.ptr[r + 1] - base; ++k)
                    dense[r * n + a.ind[k] - base] = a.val[k];
            std::vector<I> bptr{base}, bind;
            for(I ib = 0; ib < mb; ++ib)
            {
                for(I jb = 0; jb < n / block_dim; ++jb)
                {
                    std::vector<float> blk;
                    for(I ii = 0; ii < block_dim; ++ii)
                        for(I jj = 0; jj < block_dim; ++jj)
                            blk.push_back(dense[(ib * block_dim + ii) * n + jb * block_dim + jj]);
                    if(std::any_of(blk.begin(), blk.end(), [](float v) { return v != 0.0f; }))
                    {
                        bind.push_back(jb + base);
                        bval.insert(bval.end(), blk.begin(), blk.end());
                    }
                }
                bptr.push_back(base + bind.size());
            }
            DevCsr<float> d(a);
            DevBuf<I>     d_bptr(std::vector<I>(mb + 1, 77));
            const auto    nnz = [&](const I* ind, I* bnnz) {
                return rocsparse_csr2bsr_nnz(
                    handle, dir_r, m, n, csr_descr, d.ptr, ind, block_dim, bsr_descr, d_bptr, bnnz);
            };
            I bnnz = 0;
            EXPECT_EQ(nnz(nullptr, &bnnz), st_ptr);
            ASSERT_EQ(nnz(d.ind, &bnnz), st_ok);
            ASSERT_EQ(bnnz, I(bind.size()));
            EXPECT_EQ(d_bptr.read(mb + 1), bptr);
            DevBuf<I>     d_bind(bnnz);
            DevBuf<float> d_bval(std::vector<float>(bval.size(), -99.0f));
            const auto    compute = [&](const float* val, const I* ind) {
                return rocsparse_scsr2bsr(handle,
                                          dir_r,
                                          m,
                                          n,
                                          csr_descr,
                                          val,
                                          d.ptr,
                                          ind,
                                          block_dim,
                                          bsr_descr,
                                          d_bval,
                                          d_bptr,
                                          d_bind);
            };
            EXPECT_EQ(compute(nullptr, d.ind), st_ptr);
            EXPECT_EQ(compute(d.val, nullptr), st_ptr);
            ASSERT_EQ(compute(d.val, d.ind), st_ok);
            EXPECT_EQ(d_bind.read(), bind);
            EXPECT_EQ(d_bval.read(), bval);
        }
    }
} // namespace
