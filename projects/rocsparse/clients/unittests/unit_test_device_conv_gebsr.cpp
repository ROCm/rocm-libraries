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

// General block CSR (gebsr) and block / ELL conversions: gebsr2csr, gebsr2gebsr,
// csr2gebsr, bsrpad_value, gebsr2gebsc, bsr2csr, ell2csr and hyb2csr. The main
// conversions are checked on the full output: the result is expanded to a dense matrix
// on the host and compared with the dense matrix the input describes.

#include "unit_test_device_conv_common.hpp"

#include <optional>
#include <string>

using namespace convut;

namespace
{
    using ConvGebsr = ConvTest;

    // Runs f with a value of the k-th (modulo 4) value type.
    template <typename F>
    void with_type(int k, F f)
    {
        switch(k % 4)
        {
        case 0:
            return f(float{});
        case 1:
            return f(double{});
        case 2:
            return f(rocsparse_float_complex{});
        default:
            return f(rocsparse_double_complex{});
        }
    }

    // Deterministic nonzero values (1.0 .. 16.0, exact in float); complex values get an
    // imaginary part.
    template <typename T>
    T pv(size_t i)
    {
        const float v = 1.0f + static_cast<float>((i * 7) % 61) * 0.25f;
        if constexpr(is_complex_v<T>)
        {
            return T(v, -v);
        }
        else
        {
            return static_cast<T>(v);
        }
    }

    // Host gebsr matrix. Each block holds rbd x cbd values, row major for
    // rocsparse_direction_row and column major otherwise.
    template <typename T>
    struct Gebsr
    {
        rocsparse_direction dir;
        I                   mb, nb, rbd, cbd, base;
        std::vector<I>      ptr, ind;
        std::vector<T>      val;

        size_t at(size_t k, I ii, I jj) const
        {
            return k * rbd * cbd + (dir == dir_r ? ii * cbd + jj : jj * rbd + ii);
        }
    };

    // 3 x 3 blocks; block row 1 is empty, row 0 has the block columns {0, 2} and row 2
    // the block columns {1, 2}.
    template <typename T>
    Gebsr<T> make_gebsr(rocsparse_direction dir, I rbd, I cbd, I base)
    {
        Gebsr<T> g{dir, 3, 3, rbd, cbd, base, {0, 2, 2, 4}, {0, 2, 1, 2}, {}};
        for(I& p : g.ptr)
            p += base;
        for(I& c : g.ind)
            c += base;
        g.val.resize(size_t(4) * rbd * cbd);
        for(size_t k = 0; k < 4; ++k)
            for(I ii = 0; ii < rbd; ++ii)
                for(I jj = 0; jj < cbd; ++jj)
                    g.val[g.at(k, ii, jj)] = pv<T>(k * rbd * cbd + ii * cbd + jj);
        return g;
    }

    // Row-major dense expansion of a gebsr matrix, zero padded to rows x cols.
    template <typename T>
    std::vector<T> to_dense(const Gebsr<T>& g, I rows, I cols)
    {
        std::vector<T> dense(size_t(rows) * cols, T{});
        for(I ib = 0; ib < g.mb; ++ib)
            for(I k = g.ptr[ib] - g.base; k < g.ptr[ib + 1] - g.base; ++k)
                for(I ii = 0; ii < g.rbd; ++ii)
                    for(I jj = 0; jj < g.cbd; ++jj)
                    {
                        const I row = ib * g.rbd + ii, col = (g.ind[k] - g.base) * g.cbd + jj;
                        EXPECT_LT(row, rows);
                        EXPECT_LT(col, cols);
                        if(row < rows && col < cols)
                            dense[size_t(row) * cols + col] = g.val[g.at(k, ii, jj)];
                    }
        return dense;
    }

    // Reads a device gebsr matrix and compares it with the expected row-major dense
    // matrix (rows x cols, multiples of the block dimensions): the block row pointer,
    // strictly increasing in-range block columns, and the content.
    template <typename T>
    void expect_gebsr(rocsparse_direction   dir,
                      I                     rbd,
                      I                     cbd,
                      I                     base,
                      const DevBuf<I>&      ptr,
                      const DevBuf<I>&      ind,
                      const DevBuf<T>&      val,
                      const std::vector<T>& expected,
                      I                     rows,
                      I                     cols)
    {
        Gebsr<T> g{dir, rows / rbd, cols / cbd, rbd, cbd, base, ptr.read(rows / rbd + 1), {}, {}};
        const size_t nnzb = g.ptr.back() - base;
        g.ind             = ind.read(nnzb);
        g.val             = val.read(nnzb * rbd * cbd);
        ASSERT_EQ(g.ptr[0], base);
        for(I ib = 0; ib < g.mb; ++ib)
        {
            I count = 0;
            for(I jb = 0; jb < g.nb; ++jb)
            {
                bool any = false;
                for(I ii = 0; ii < rbd; ++ii)
                    for(I jj = 0; jj < cbd; ++jj)
                        any = any || expected[size_t(ib * rbd + ii) * cols + jb * cbd + jj] != T{};
                count += any;
            }
            EXPECT_EQ(g.ptr[ib + 1] - g.ptr[ib], count) << "block row " << ib;
            for(I k = g.ptr[ib] - base; k < g.ptr[ib + 1] - base; ++k)
            {
                EXPECT_GE(g.ind[k] - base, 0);
                EXPECT_LT(g.ind[k] - base, g.nb);
                EXPECT_TRUE(k == g.ptr[ib] - base || g.ind[k] > g.ind[k - 1]) << "block row " << ib;
            }
        }
        EXPECT_EQ(to_dense(g, rows, cols), expected);
    }

    // Number of nonzero blocks of a row-major dense matrix.
    template <typename T>
    I count_blocks(const std::vector<T>& dense, I rows, I cols, I rbd, I cbd)
    {
        I count = 0;
        for(I ib = 0; ib < rows / rbd; ++ib)
            for(I jb = 0; jb < cols / cbd; ++jb)
            {
                bool any = false;
                for(I ii = 0; ii < rbd; ++ii)
                    for(I jj = 0; jj < cbd; ++jj)
                        any = any || dense[size_t(ib * rbd + ii) * cols + jb * cbd + jj] != T{};
                count += any;
            }
        return count;
    }

    // Dense m x n matrix with an irregular pattern (an empty row, rows of different
    // lengths); the stored values are nonzero.
    template <typename T>
    std::vector<T> make_csr_dense(I m, I n)
    {
        std::vector<T> dense(size_t(m) * n, T{});
        for(I i = 0; i < m; ++i)
            for(I j = 0; j < n && i % 11 != 5; ++j)
                if((i * 3 + j * 5) % 7 == 0 || (i + j) % 13 == 0)
                    dense[size_t(i) * n + j] = pv<T>(size_t(i) * n + j);
        return dense;
    }

    template <typename T>
    HostCsr<T> dense_to_csr(const std::vector<T>& dense, I m, I n, I base)
    {
        HostCsr<T> c{m, n, static_cast<int>(base), {base}, {}, {}};
        for(I i = 0; i < m; ++i)
        {
            for(I j = 0; j < n; ++j)
                if(dense[size_t(i) * n + j] != T{})
                {
                    c.ind.push_back(j + base);
                    c.val.push_back(dense[size_t(i) * n + j]);
                }
            c.ptr.push_back(c.val.size() + base);
        }
        return c;
    }

    // Device csr matrix against the expected row-major dense matrix: the row pointer
    // starts at the base, the columns of each row increase, and the content matches.
    template <typename T>
    void expect_csr_dense(const DevBuf<I>&      ptr,
                          const DevBuf<I>&      ind,
                          const DevBuf<T>&      val,
                          I                     base,
                          I                     m,
                          I                     n,
                          const std::vector<T>& expected)
    {
        const size_t nnz = expected.size() - std::count(expected.begin(), expected.end(), T{});
        const auto   p = ptr.read(m + 1), j = ind.read(nnz);
        const auto   v = val.read(nnz);
        ASSERT_EQ(p[0], base);
        ASSERT_EQ(p[m] - base, nnz);
        std::vector<T> dense(size_t(m) * n, T{});
        for(I i = 0; i < m; ++i)
            for(I k = p[i] - base; k < p[i + 1] - base; ++k)
            {
                EXPECT_TRUE(k == p[i] - base || j[k] > j[k - 1]) << "row " << i;
                dense[size_t(i) * n + j[k] - base] = v[k];
            }
        EXPECT_EQ(dense, expected);
    }

    std::string
        trace(rocsparse_direction dir, I rbd, I cbd, I base_a, I base_c, bool dev, bool shrink)
    {
        return std::string(dir == dir_r ? "row" : "column") + " block " + std::to_string(rbd) + "x"
               + std::to_string(cbd) + " bases " + std::to_string(base_a) + std::to_string(base_c)
               + " device_mode " + std::to_string(dev) + " shrink " + std::to_string(shrink);
    }

    // ------------------------------------------------------------------------------
    // gebsr2csr
    // ------------------------------------------------------------------------------

    template <typename T>
    void run_gebsr2csr(
        rocsparse_handle handle, rocsparse_direction dir, I rbd, I cbd, I bsr_base, I csr_base)
    {
        SCOPED_TRACE(trace(dir, rbd, cbd, bsr_base, csr_base, false, false));
        const auto g = make_gebsr<T>(dir, rbd, cbd, bsr_base);
        const I    m = g.mb * rbd, n = g.nb * cbd, nnz = 4 * rbd * cbd;
        MatDescr   bsr_descr(base_of(bsr_base)), csr_descr(base_of(csr_base));
        DevBuf<I>  bsr_ptr(g.ptr), bsr_ind(g.ind), ptr(m + 1), ind(nnz);
        DevBuf<T>  bsr_val(g.val), val(nnz);
        ASSERT_EQ(pick<T>(rocsparse_sgebsr2csr,
                          rocsparse_dgebsr2csr,
                          rocsparse_cgebsr2csr,
                          rocsparse_zgebsr2csr)(handle,
                                                dir,
                                                g.mb,
                                                g.nb,
                                                bsr_descr,
                                                bsr_val,
                                                bsr_ptr,
                                                bsr_ind,
                                                rbd,
                                                cbd,
                                                csr_descr,
                                                val,
                                                ptr,
                                                ind),
                  st_ok);
        expect_csr_dense(ptr, ind, val, csr_base, m, n, to_dense(g, m, n));
    }

    // gebsr2csr for every pair of dispatch buckets (<= 2, 4, 8, 16, 32, 64, 128) of the
    // row and the column block dimension (the seven diagonal pairs included), both block
    // directions. The index bases and value types are varied over the grid.
    TEST_F(ConvGebsr, Gebsr2CsrBucketGrid)
    {
        int combo = 0;
        for(rocsparse_direction dir : {dir_r, dir_c})
            for(I rbd : {1, 3, 6, 12, 24, 48, 96})
                for(I cbd : {2, 4, 8, 16, 32, 64, 128})
                {
                    const int c = combo++;
                    with_type(c / 4, [&](auto t) {
                        run_gebsr2csr<decltype(t)>(handle, dir, rbd, cbd, c % 2, (c / 2) % 2);
                    });
                    if(HasFatalFailure())
                        return;
                }
    }

    // Equal row and column block dimensions are converted through the BSR code path.
    TEST_F(ConvGebsr, Gebsr2CsrEqualBlockDims)
    {
        for(rocsparse_direction dir : {dir_r, dir_c})
            for(I bd : {1, 3, 24, 96})
                run_gebsr2csr<double>(handle, dir, bd, bd, bd % 2, dir == dir_c);
    }

    TEST_F(ConvGebsr, Gebsr2CsrEmptyAndArguments)
    {
        const I rbd = 3, cbd = 5;

        // Empty matrices: the row pointer is filled with the index base.
        for(I base : {0, 1})
            for(I mb_nb : {0, 1})
            {
                const I   mb = mb_nb ? 2 : 0, nb = mb_nb ? 0 : 3, m = mb * rbd;
                MatDescr  bsr_descr, csr_descr(base_of(base));
                DevBuf<I> bsr_ptr(std::vector<I>(mb + 1, 0)), ptr(std::vector<I>(m + 1, -3));
                EXPECT_EQ(rocsparse_sgebsr2csr(handle,
                                               dir_r,
                                               mb,
                                               nb,
                                               bsr_descr,
                                               nullptr,
                                               bsr_ptr,
                                               nullptr,
                                               rbd,
                                               cbd,
                                               csr_descr,
                                               nullptr,
                                               ptr,
                                               nullptr),
                          st_ok);
                EXPECT_EQ(ptr.read(m + 1), std::vector<I>(m + 1, base));
            }

        const auto    g = make_gebsr<float>(dir_r, rbd, cbd, 0);
        const I       m = g.mb * rbd, nnz = 4 * rbd * cbd;
        MatDescr      bsr_descr, csr_descr;
        DevBuf<I>     bsr_ptr(g.ptr), bsr_ind(g.ind), ptr(m + 1), ind(nnz);
        DevBuf<float> bsr_val(g.val), val(nnz);
        const auto    call = [&](rocsparse_handle h, rocsparse_direction dir, I rb, I cb) {
            return rocsparse_sgebsr2csr(h,
                                        dir,
                                        g.mb,
                                        g.nb,
                                        bsr_descr,
                                        bsr_val,
                                        bsr_ptr,
                                        bsr_ind,
                                        rb,
                                        cb,
                                        csr_descr,
                                        val,
                                        ptr,
                                        ind);
        };
        ASSERT_EQ(call(handle, dir_r, rbd, cbd), st_ok);
        expect_csr_dense(ptr, ind, val, 0, m, g.nb * cbd, to_dense(g, m, g.nb * cbd));
        EXPECT_EQ(call(nullptr, dir_r, rbd, cbd), rocsparse_status_invalid_handle);
        // An invalid direction is not documented; the library convention is invalid_value.
        EXPECT_STATUS_IN(
            call(handle, static_cast<rocsparse_direction>(77), rbd, cbd), st_val, st_size);
        EXPECT_EQ(call(handle, dir_r, 0, cbd), st_size);
        EXPECT_EQ(call(handle, dir_r, rbd, 0), st_size);
        // The arrays are required: the row pointer announces nnzb blocks.
        expect_null_args(
            [&](const float* bv, const I* bi, float* cv, I* ci) {
                return rocsparse_sgebsr2csr(handle,
                                            dir_r,
                                            g.mb,
                                            g.nb,
                                            bsr_descr,
                                            bv,
                                            bsr_ptr,
                                            bi,
                                            rbd,
                                            cbd,
                                            csr_descr,
                                            cv,
                                            ptr,
                                            ci);
            },
            bsr_val.get(),
            bsr_ind.get(),
            val.get(),
            ind.get());

        // Only general matrices are supported (the status of the refusal is not documented).
        for(rocsparse_mat_descr d : {bsr_descr.get(), csr_descr.get()})
        {
            ASSERT_EQ(rocsparse_set_mat_type(d, rocsparse_matrix_type_symmetric), st_ok);
            EXPECT_REJECTED(call(handle, dir_r, rbd, cbd));
            ASSERT_EQ(rocsparse_set_mat_type(d, rocsparse_matrix_type_general), st_ok);
        }
    }

    // ------------------------------------------------------------------------------
    // gebsr2gebsr
    // ------------------------------------------------------------------------------

    template <typename T>
    void run_gebsr2gebsr(rocsparse_handle    handle,
                         rocsparse_direction dir,
                         I                   rbd_a,
                         I                   cbd_a,
                         I                   rbd_c,
                         I                   cbd_c,
                         I                   base_a,
                         I                   base_c,
                         bool                device_mode,
                         bool                shrink)
    {
        SCOPED_TRACE(trace(dir, rbd_c, cbd_c, base_a, base_c, device_mode, shrink)
                     + " A=" + std::to_string(rbd_a) + "x" + std::to_string(cbd_a));
        const auto a = make_gebsr<T>(dir, rbd_a, cbd_a, base_a);
        const I    m = a.mb * rbd_a, n = a.nb * cbd_a;
        const I    mb_c = (m + rbd_c - 1) / rbd_c, nb_c = (n + cbd_c - 1) / cbd_c;
        const I    rows = mb_c * rbd_c, cols = nb_c * cbd_c, nnzb = 4;
        MatDescr   descr_a(base_of(base_a)), descr_c(base_of(base_c));
        DevBuf<I>  ptr_a(a.ptr), ind_a(a.ind), ptr_c(mb_c + 1);
        DevBuf<T>  val_a(a.val);

        size_t buffer_size = 0;
        ASSERT_EQ(pick<T>(rocsparse_sgebsr2gebsr_buffer_size,
                          rocsparse_dgebsr2gebsr_buffer_size,
                          rocsparse_cgebsr2gebsr_buffer_size,
                          rocsparse_zgebsr2gebsr_buffer_size)(handle,
                                                              dir,
                                                              a.mb,
                                                              a.nb,
                                                              nnzb,
                                                              descr_a,
                                                              val_a,
                                                              ptr_a,
                                                              ind_a,
                                                              rbd_a,
                                                              cbd_a,
                                                              rbd_c,
                                                              cbd_c,
                                                              &buffer_size),
                  st_ok);
        // The size is the library's choice; the routines run with exactly that size.
        DevBuf<char>                  temp(buffer_size);
        std::optional<NoHandleBuffer> no_buffer;
        if(shrink)
            no_buffer.emplace(handle);

        const I nnzb_c = nnz_via(handle, device_mode, [&](I* nnz) {
            return rocsparse_gebsr2gebsr_nnz(handle,
                                             dir,
                                             a.mb,
                                             a.nb,
                                             nnzb,
                                             descr_a,
                                             ptr_a,
                                             ind_a,
                                             rbd_a,
                                             cbd_a,
                                             descr_c,
                                             ptr_c,
                                             rbd_c,
                                             cbd_c,
                                             nnz,
                                             temp);
        });

        // The same dense matrix padded with zeros to the output block grid. All stored
        // values are nonzero, so the blocks the conversion keeps are those with a nonzero.
        std::vector<T> expected(size_t(rows) * cols, T{});
        const auto     dense_a = to_dense(a, m, n);
        for(I i = 0; i < m; ++i)
            std::copy_n(&dense_a[size_t(i) * n], n, &expected[size_t(i) * cols]);
        const I expected_nnzb = count_blocks(expected, rows, cols, rbd_c, cbd_c);
        ASSERT_GT(expected_nnzb, 0);
        ASSERT_EQ(nnzb_c, expected_nnzb);

        DevBuf<I> ind_c(nnzb_c);
        DevBuf<T> val_c(std::vector<T>(size_t(nnzb_c) * rbd_c * cbd_c, make_val<T>(-99)));
        ASSERT_EQ(pick<T>(rocsparse_sgebsr2gebsr,
                          rocsparse_dgebsr2gebsr,
                          rocsparse_cgebsr2gebsr,
                          rocsparse_zgebsr2gebsr)(handle,
                                                  dir,
                                                  a.mb,
                                                  a.nb,
                                                  nnzb,
                                                  descr_a,
                                                  val_a,
                                                  ptr_a,
                                                  ind_a,
                                                  rbd_a,
                                                  cbd_a,
                                                  descr_c,
                                                  val_c,
                                                  ptr_c,
                                                  ind_c,
                                                  rbd_c,
                                                  cbd_c,
                                                  temp),
                  st_ok);
        expect_gebsr(dir, rbd_c, cbd_c, base_c, ptr_c, ind_c, val_c, expected, rows, cols);
    }

    // gebsr2gebsr over a grid of output block sizes: the fast kernels (row block
    // dimension <= 32, per power of two bucket) and the CSR based path (> 32), both
    // directions. For every output row block dimension the 16 runs (direction, input
    // block size, output column block dimension) use all 16 combinations of (base of A,
    // base of C, host/device pointer mode, large/small handle buffer) and cycle through
    // the value types.
    TEST_F(ConvGebsr, Gebsr2GebsrGrid)
    {
        int dir_index = 0;
        for(rocsparse_direction dir : {dir_r, dir_c})
        {
            int dims_index = 0;
            for(auto dims_a : {std::pair<I, I>{3, 2}, std::pair<I, I>{4, 5}})
            {
                for(I rbd_c : {1, 2, 3, 7, 13, 25, 40})
                {
                    int cbd_index = 0;
                    for(I cbd_c : {2, 5, 16, 33})
                    {
                        const int combo = cbd_index + 4 * (dims_index + 2 * dir_index);
                        with_type(combo + rbd_c, [&](auto t) {
                            run_gebsr2gebsr<decltype(t)>(handle,
                                                         dir,
                                                         dims_a.first,
                                                         dims_a.second,
                                                         rbd_c,
                                                         cbd_c,
                                                         combo % 2,
                                                         (combo / 2) % 2,
                                                         (combo / 4) % 2,
                                                         (combo / 8) % 2);
                        });
                        ++cbd_index;
                        if(HasFatalFailure())
                            return;
                    }
                }
                ++dims_index;
            }
            ++dir_index;
        }
    }

    // gebsr2gebsr without blocks: buffer size and nnz of the two paths and the compute
    // quick return.
    TEST_F(ConvGebsr, Gebsr2GebsrEmpty)
    {
        const I rbd_a = 3, cbd_a = 2, cbd_c = 3, mb = 2, nb = 3;
        for(I rbd_c : {4, 40})
        {
            // Without block rows or without block columns: the output row pointer is the
            // index base for every output block row; the rest of the array is untouched.
            for(int no_cols = 0; no_cols < 2; ++no_cols)
            {
                const I   mb_e = no_cols ? mb : 0, nb_e = no_cols ? 0 : nb;
                const I   mb_c = (mb_e * rbd_a + rbd_c - 1) / rbd_c;
                MatDescr  descr_a, descr_c;
                DevBuf<I> ptr_a(std::vector<I>(mb_e + 1, 0)), ptr_c(std::vector<I>(8, -4));
                size_t    buffer_size = 1;
                ASSERT_EQ(rocsparse_sgebsr2gebsr_buffer_size(handle,
                                                             dir_r,
                                                             mb_e,
                                                             nb_e,
                                                             0,
                                                             descr_a,
                                                             nullptr,
                                                             ptr_a,
                                                             nullptr,
                                                             rbd_a,
                                                             cbd_a,
                                                             rbd_c,
                                                             cbd_c,
                                                             &buffer_size),
                          st_ok);
                // At least one byte: a null buffer is rejected on the CSR based path.
                DevBuf<char> temp(buffer_size);
                I            nnz_total = -1;
                EXPECT_EQ(rocsparse_gebsr2gebsr_nnz(handle,
                                                    dir_r,
                                                    mb_e,
                                                    nb_e,
                                                    0,
                                                    descr_a,
                                                    ptr_a,
                                                    nullptr,
                                                    rbd_a,
                                                    cbd_a,
                                                    descr_c,
                                                    ptr_c,
                                                    rbd_c,
                                                    cbd_c,
                                                    &nnz_total,
                                                    temp),
                          st_ok);
                EXPECT_EQ(nnz_total, 0);
                std::vector<I> want(8, -4);
                std::fill_n(want.begin(), mb_c + 1, 0);
                EXPECT_EQ(ptr_c.read(8), want);
                EXPECT_EQ(rocsparse_sgebsr2gebsr(handle,
                                                 dir_r,
                                                 mb_e,
                                                 nb_e,
                                                 0,
                                                 descr_a,
                                                 nullptr,
                                                 ptr_a,
                                                 nullptr,
                                                 rbd_a,
                                                 cbd_a,
                                                 descr_c,
                                                 nullptr,
                                                 ptr_c,
                                                 nullptr,
                                                 rbd_c,
                                                 cbd_c,
                                                 temp),
                          st_ok);
            }

            // Block rows and columns but no blocks: the output row pointer is the base and
            // no entries are reported.
            for(I base_c : {0, 1})
                for(bool device_mode : {false, true})
                {
                    const I   mb_c = (mb * rbd_a + rbd_c - 1) / rbd_c;
                    MatDescr  descr_a, descr_c(base_of(base_c));
                    DevBuf<I> ptr_a(std::vector<I>(mb + 1, 0)), ptr_c(std::vector<I>(mb_c + 1, -4));
                    DevBuf<char> temp(size_t(1) << 12);
                    const I      nnz = nnz_via(handle, device_mode, [&](I* p) {
                        return rocsparse_gebsr2gebsr_nnz(handle,
                                                         dir_r,
                                                         mb,
                                                         nb,
                                                         0,
                                                         descr_a,
                                                         ptr_a,
                                                         nullptr,
                                                         rbd_a,
                                                         cbd_a,
                                                         descr_c,
                                                         ptr_c,
                                                         rbd_c,
                                                         cbd_c,
                                                         p,
                                                         temp);
                    });
                    EXPECT_EQ(nnz, 0);
                    EXPECT_EQ(ptr_c.read(mb_c + 1), std::vector<I>(mb_c + 1, base_c));
                }
        }
    }

    // ------------------------------------------------------------------------------
    // csr2gebsr
    // ------------------------------------------------------------------------------

    template <typename T>
    void run_csr2gebsr(rocsparse_handle    handle,
                       rocsparse_direction dir,
                       I                   rbd,
                       I                   cbd,
                       I                   csr_base,
                       I                   bsr_base,
                       bool                device_mode,
                       bool                shrink)
    {
        SCOPED_TRACE(trace(dir, rbd, cbd, csr_base, bsr_base, device_mode, shrink));
        const I    m = 37, n = 29, mb = (m + rbd - 1) / rbd, nb = (n + cbd - 1) / cbd;
        const auto dense = make_csr_dense<T>(m, n);
        const auto csr   = dense_to_csr(dense, m, n, csr_base);
        MatDescr   csr_descr(base_of(csr_base)), bsr_descr(base_of(bsr_base));
        DevBuf<I>  ptr(csr.ptr), ind(csr.ind), bsr_ptr(mb + 1);
        DevBuf<T>  val(csr.val);

        size_t buffer_size = 0;
        ASSERT_EQ(pick<T>(rocsparse_scsr2gebsr_buffer_size,
                          rocsparse_dcsr2gebsr_buffer_size,
                          rocsparse_ccsr2gebsr_buffer_size,
                          rocsparse_zcsr2gebsr_buffer_size)(
                      handle, dir, m, n, csr_descr, val, ptr, ind, rbd, cbd, &buffer_size),
                  st_ok);
        DevBuf<char>                  temp(buffer_size);
        std::optional<NoHandleBuffer> no_buffer;
        if(shrink)
            no_buffer.emplace(handle);
        const I nnzb = nnz_via(handle, device_mode, [&](I* nnz) {
            return rocsparse_csr2gebsr_nnz(
                handle, dir, m, n, csr_descr, ptr, ind, bsr_descr, bsr_ptr, rbd, cbd, nnz, temp);
        });

        // The blocks to keep are those that hold at least one stored entry.
        std::vector<T> expected(size_t(mb * rbd) * (nb * cbd), T{});
        for(I i = 0; i < m; ++i)
            std::copy_n(&dense[size_t(i) * n], n, &expected[size_t(i) * nb * cbd]);
        const I expected_nnzb = count_blocks(expected, mb * rbd, nb * cbd, rbd, cbd);
        ASSERT_GT(expected_nnzb, 0);
        ASSERT_EQ(nnzb, expected_nnzb);

        DevBuf<I> bsr_ind(nnzb);
        DevBuf<T> bsr_val(std::vector<T>(size_t(nnzb) * rbd * cbd, make_val<T>(-99)));
        ASSERT_EQ(pick<T>(rocsparse_scsr2gebsr,
                          rocsparse_dcsr2gebsr,
                          rocsparse_ccsr2gebsr,
                          rocsparse_zcsr2gebsr)(handle,
                                                dir,
                                                m,
                                                n,
                                                csr_descr,
                                                val,
                                                ptr,
                                                ind,
                                                bsr_descr,
                                                bsr_val,
                                                bsr_ptr,
                                                bsr_ind,
                                                rbd,
                                                cbd,
                                                temp),
                  st_ok);
        expect_gebsr(
            dir, rbd, cbd, bsr_base, bsr_ptr, bsr_ind, bsr_val, expected, mb * rbd, nb * cbd);
    }

    // csr2gebsr over a grid of block sizes covering the buckets of the kernel dispatch
    // for both directions (square blocks go through the BSR path, a row block dimension
    // above 64 takes the fallback). The index bases, pointer modes, handle buffer
    // situations and value types are varied across the grid; the row block dimension 1
    // (for even column block dimensions) and the largest ones always run without the
    // handle buffer, which makes the temporary storage of their kernels come from the
    // allocator.
    TEST_F(ConvGebsr, Csr2GebsrGrid)
    {
        int        combo = 0;
        const auto run   = [&](rocsparse_direction dir, I rbd, I cbd) {
            const int c = combo++;
            with_type(c / 16 + c, [&](auto t) {
                run_csr2gebsr<decltype(t)>(handle,
                                           dir,
                                           rbd,
                                           cbd,
                                           c % 2,
                                           (c / 2) % 2,
                                           (c / 4) % 2,
                                           (c / 8) % 2 || (rbd == 1 && cbd % 2 == 0) || rbd >= 48);
            });
        };
        for(I rbd : {1, 2, 3, 5, 9, 17, 40, 48, 70})
            for(I cbd : {1, 3, 6, 12, 24, 40})
                run(dir_r, rbd, cbd);
        for(I rbd : {2, 4, 9, 48})
            for(I cbd : {3, 4, 24, 40})
                run(dir_c, rbd, cbd);
        run(dir_r, 4, 4);
    }

    TEST_F(ConvGebsr, Csr2GebsrEmptyAndArguments)
    {
        const I    rbd = 3, cbd = 4;
        MatDescr   csr_descr, bsr_descr;
        const auto nnz_call = [&](rocsparse_direction dir,
                                  I                   m,
                                  I                   n,
                                  const I*            ptr,
                                  const I*            ind,
                                  I*                  bsr_ptr,
                                  I                   rb,
                                  I*                  nnz,
                                  void*               temp) {
            return rocsparse_csr2gebsr_nnz(
                handle, dir, m, n, csr_descr, ptr, ind, bsr_descr, bsr_ptr, rb, cbd, nnz, temp);
        };

        // m == 0 or n == 0: nothing is stored and the compute routine returns at once.
        for(I base : {0, 1})
            for(bool device_mode : {false, true})
                for(int shape = 0; shape < 2; ++shape)
                {
                    const I   m = shape ? 7 : 0, n = shape ? 0 : 5, mb = (m + rbd - 1) / rbd;
                    MatDescr  csr(base_of(0)), bsr(base_of(base));
                    DevBuf<I> csr_ptr(std::vector<I>(m + 1, 0)),
                        bsr_ptr(std::vector<I>(mb + 1, -2));
                    const I nnz = nnz_via(handle, device_mode, [&](I* p) {
                        return rocsparse_csr2gebsr_nnz(handle,
                                                       dir_r,
                                                       m,
                                                       n,
                                                       csr,
                                                       csr_ptr,
                                                       nullptr,
                                                       bsr,
                                                       bsr_ptr,
                                                       rbd,
                                                       cbd,
                                                       p,
                                                       nullptr);
                    });
                    EXPECT_EQ(nnz, 0);
                    EXPECT_EQ(bsr_ptr.read(mb + 1), std::vector<I>(mb + 1, base));
                    EXPECT_EQ(rocsparse_scsr2gebsr(handle,
                                                   dir_r,
                                                   m,
                                                   n,
                                                   csr,
                                                   nullptr,
                                                   csr_ptr,
                                                   nullptr,
                                                   bsr,
                                                   nullptr,
                                                   bsr_ptr,
                                                   nullptr,
                                                   rbd,
                                                   cbd,
                                                   nullptr),
                              st_ok);
                }
        // The size query of an empty matrix succeeds.
        {
            DevBuf<I> csr_ptr(std::vector<I>{0});
            size_t    buffer_size = 99;
            EXPECT_EQ(rocsparse_scsr2gebsr_buffer_size(handle,
                                                       dir_r,
                                                       0,
                                                       0,
                                                       csr_descr,
                                                       nullptr,
                                                       csr_ptr,
                                                       nullptr,
                                                       rbd,
                                                       cbd,
                                                       &buffer_size),
                      st_ok);
        }

        // Missing arrays while the row pointer announces entries.
        const I            m = 37, n = 29, mb = (m + rbd - 1) / rbd, nb = (n + cbd - 1) / cbd;
        const auto         dense = make_csr_dense<float>(m, n);
        const auto         csr   = dense_to_csr(dense, m, n, 0);
        std::vector<float> padded(size_t(mb * rbd) * (nb * cbd), 0.0f);
        for(I i = 0; i < m; ++i)
            std::copy_n(&dense[size_t(i) * n], n, &padded[size_t(i) * nb * cbd]);
        DevBuf<I>     ptr(csr.ptr), ind(csr.ind), bsr_ptr(mb + 1);
        DevBuf<float> val(csr.val);
        DevBuf<char>  temp(size_t(1) << 16);
        I             nnz = 0;
        EXPECT_EQ(nnz_call(dir_r, m, n, ptr, nullptr, bsr_ptr, rbd, &nnz, temp), st_ptr);
        size_t buffer_size = 0;
        EXPECT_EQ(
            rocsparse_scsr2gebsr_buffer_size(
                handle, dir_r, m, n, csr_descr, nullptr, ptr, nullptr, rbd, cbd, &buffer_size),
            st_ptr);
        EXPECT_STATUS_IN(
            nnz_call(
                static_cast<rocsparse_direction>(99), m, n, ptr, ind, bsr_ptr, rbd, &nnz, temp),
            st_val,
            st_size);
        ASSERT_EQ(nnz_call(dir_r, m, n, ptr, ind, bsr_ptr, rbd, &nnz, temp), st_ok);
        ASSERT_EQ(nnz, count_blocks(padded, mb * rbd, nb * cbd, rbd, cbd));

        DevBuf<I>     bsr_ind(nnz);
        DevBuf<float> bsr_val(size_t(nnz) * rbd * cbd);
        const auto    compute
            = [&](rocsparse_direction dir, const float* cv, const I* ci, I rb, I cb) {
                  return rocsparse_scsr2gebsr(handle,
                                              dir,
                                              m,
                                              n,
                                              csr_descr,
                                              cv,
                                              ptr,
                                              ci,
                                              bsr_descr,
                                              bsr_val,
                                              bsr_ptr,
                                              bsr_ind,
                                              rb,
                                              cb,
                                              temp);
              };
        EXPECT_EQ(compute(dir_r, nullptr, ind, rbd, cbd), st_ptr);
        EXPECT_EQ(compute(dir_r, val, nullptr, rbd, cbd), st_ptr);
        EXPECT_EQ(compute(dir_r, val, ind, 0, cbd), st_size);
        EXPECT_EQ(compute(dir_r, val, ind, rbd, 0), st_size);
        // The library convention for an invalid enumeration is invalid_value; the header
        // does not document it.
        EXPECT_STATUS_IN(
            compute(static_cast<rocsparse_direction>(55), val, ind, rbd, cbd), st_val, st_size);
    }
}

namespace
{
    // ------------------------------------------------------------------------------
    // bsrpad_value
    // ------------------------------------------------------------------------------

    // m = 5 rows in 3 block rows of 2: the last block row has one row outside the
    // matrix. The block rows hold the block columns {0}, {1} and {0, 2} (or {2, 0}).
    TEST_F(ConvGebsr, BsrPadValue)
    {
        const I     m = 5, mb = 3, bd = 2, nnzb = 4;
        const float value = 7.0f;
        for(I base : {0, 1})
            for(bool unsorted : {false, true})
            {
                SCOPED_TRACE("base " + std::to_string(base) + " unsorted "
                             + std::to_string(unsorted));
                MatDescr descr(base_of(base));
                ASSERT_EQ(rocsparse_set_mat_storage_mode(descr,
                                                         unsorted ? rocsparse_storage_mode_unsorted
                                                                  : rocsparse_storage_mode_sorted),
                          st_ok);
                DevBuf<I>                d_ptr(std::vector<I>{base, base + 1, base + 2, base + 4});
                DevBuf<I>                d_ind(std::vector<I>{
                    base, base + 1, unsorted ? base + 2 : base, unsorted ? base : base + 2});
                const std::vector<float> val(nnzb * bd * bd, 1.0f);
                DevBuf<float>            d_val(val);
                ASSERT_EQ(rocsparse_sbsrpad_value(
                              handle, m, mb, nnzb, bd, value, descr, d_val, d_ptr, d_ind),
                          st_ok);

                // Only the entry (1, 1) of the diagonal block of the last block row is set.
                std::vector<float> expected                         = val;
                expected[(unsorted ? 2 : 3) * bd * bd + 1 * bd + 1] = value;
                EXPECT_EQ(d_val.read(), expected);
                // The values are required.
                EXPECT_EQ(rocsparse_sbsrpad_value(handle,
                                                  m,
                                                  mb,
                                                  nnzb,
                                                  bd,
                                                  value,
                                                  descr,
                                                  static_cast<float*>(nullptr),
                                                  d_ptr,
                                                  d_ind),
                          st_ptr);
            }
    }

    TEST_F(ConvGebsr, BsrPadValueNothingToPad)
    {
        const I       bd = 2;
        MatDescr      descr;
        DevBuf<I>     d_ind(std::vector<I>{0, 1, 2});
        DevBuf<float> d_val(std::vector<float>(3 * bd * bd, 1.0f));
        // m = mb * block_dim: no row outside the matrix; no block rows; no blocks.
        for(const auto& c : {std::array<I, 3>{6, 3, 3}, {0, 0, 0}, {5, 3, 0}})
        {
            DevBuf<I> d_ptr(c[2] ? std::vector<I>{0, 1, 2, 3} : std::vector<I>(4, 0));
            EXPECT_EQ(rocsparse_sbsrpad_value(
                          handle, c[0], c[1], c[2], bd, 7.0f, descr, d_val, d_ptr, d_ind),
                      st_ok);
            EXPECT_EQ(d_val.read(), std::vector<float>(3 * bd * bd, 1.0f));
        }
    }

    // ------------------------------------------------------------------------------
    // gebsr2gebsc
    // ------------------------------------------------------------------------------

    TEST_F(ConvGebsr, Gebsr2Gebsc)
    {
        const I mb = 3, nb = 4, rdim = 2, cdim = 3;
        for(I base : {0, 1})
        {
            // Block row r holds the block columns 0 .. r + 1.
            std::vector<I> ptr = {base}, ind;
            for(I r = 0; r < mb; ++r)
            {
                for(I c = 0; c <= r + 1; ++c)
                    ind.push_back(base + c);
                ptr.push_back(base + static_cast<I>(ind.size()));
            }
            const I nnzb = ind.size();

            // Host transpose of the block pattern.
            std::vector<I> cptr(nb + 1, base), rind(nnzb);
            for(I e = 0; e < nnzb; ++e)
                ++cptr[ind[e] - base + 1];
            for(I c = 0; c < nb; ++c)
                cptr[c + 1] += cptr[c] - base;
            std::vector<I> next(cptr.begin(), cptr.end() - 1), perm(nnzb);
            for(I r = 0; r < mb; ++r)
                for(I e = ptr[r] - base; e < ptr[r + 1] - base; ++e)
                {
                    const I pos = next[ind[e] - base]++ - base;
                    rind[pos]   = r + base;
                    perm[pos]   = e;
                }
            // Each block is copied unchanged to its transposed position.
            const I            bs = rdim * cdim;
            std::vector<float> val(nnzb * bs), cval(nnzb * bs);
            for(I t = 0; t < nnzb * bs; ++t)
                val[t] = 1.0f + t;
            for(I k = 0; k < nnzb; ++k)
                std::copy_n(&val[perm[k] * bs], bs, &cval[k * bs]);

            DevBuf<I>     d_ptr(ptr), d_ind(ind);
            DevBuf<float> d_val(val);
            size_t        size = 0;
            ASSERT_EQ(rocsparse_sgebsr2gebsc_buffer_size(
                          handle, mb, nb, nnzb, d_val, d_ptr, d_ind, rdim, cdim, &size),
                      st_ok);
            for(rocsparse_action action : {rocsparse_action_symbolic, rocsparse_action_numeric})
            {
                SCOPED_TRACE("base " + std::to_string(base) + " action " + std::to_string(action));
                DevBuf<char>  d_buffer(size);
                DevBuf<I>     d_cptr(nb + 1), d_rind(nnzb);
                DevBuf<float> d_cval(nnzb * rdim * cdim);
                ASSERT_EQ(rocsparse_sgebsr2gebsc(handle,
                                                 mb,
                                                 nb,
                                                 nnzb,
                                                 d_val,
                                                 d_ptr,
                                                 d_ind,
                                                 rdim,
                                                 cdim,
                                                 d_cval,
                                                 d_rind,
                                                 d_cptr,
                                                 action,
                                                 base_of(base),
                                                 d_buffer),
                          st_ok);
                EXPECT_EQ(d_cptr.read(nb + 1), cptr);
                EXPECT_EQ(d_rind.read(nnzb), rind);
                if(action == rocsparse_action_numeric)
                    EXPECT_EQ(d_cval.read(nnzb * bs), cval);
            }
        }
    }

    TEST_F(ConvGebsr, Gebsr2GebscEmpty)
    {
        const I rdim = 2, cdim = 3;
        for(I base : {0, 1})
            for(const auto& d : {std::pair<I, I>(3, 4), {0, 4}, {3, 0}})
            {
                const I mb = d.first, nb = d.second;
                SCOPED_TRACE("base " + std::to_string(base) + " mb " + std::to_string(mb) + " nb "
                             + std::to_string(nb));
                DevBuf<I> d_ptr(std::vector<I>(mb + 1, base)), d_ind(0),
                    d_cptr(std::vector<I>(nb + 1, 77));
                DevBuf<float> d_val(0), d_cval(0);
                size_t        size = 0;
                ASSERT_EQ(rocsparse_sgebsr2gebsc_buffer_size(
                              handle, mb, nb, 0, d_val, d_ptr, d_ind, rdim, cdim, &size),
                          st_ok);
                DevBuf<char> d_buffer(size);
                ASSERT_EQ(rocsparse_sgebsr2gebsc(handle,
                                                 mb,
                                                 nb,
                                                 0,
                                                 d_val,
                                                 d_ptr,
                                                 d_ind,
                                                 rdim,
                                                 cdim,
                                                 d_cval,
                                                 d_ind,
                                                 d_cptr,
                                                 rocsparse_action_numeric,
                                                 base_of(base),
                                                 d_buffer),
                          st_ok);
                EXPECT_EQ(d_cptr.read(nb + 1), std::vector<I>(nb + 1, base));
            }
    }

    // ------------------------------------------------------------------------------
    // bsr2csr
    // ------------------------------------------------------------------------------

    TEST_F(ConvGebsr, Bsr2CsrEdges)
    {
        const I bd = 2;
        for(I base : {0, 1})
        {
            MatDescr bsr_descr(base_of(base)), csr_descr(base_of(base));

            // No block row or no block column: the row pointers are all the index base.
            for(const auto& d : {std::pair<I, I>(0, 3), {3, 0}})
            {
                const I mb = d.first, nb = d.second;
                SCOPED_TRACE("base " + std::to_string(base) + " mb " + std::to_string(mb) + " nb "
                             + std::to_string(nb));
                DevBuf<I> d_bptr(std::vector<I>(mb + 1, base)),
                    d_ptr(std::vector<I>(mb * bd + 1, 77)), d_ind(0);
                DevBuf<float> d_val(0);
                ASSERT_EQ(rocsparse_sbsr2csr(handle,
                                             dir_r,
                                             mb,
                                             nb,
                                             bsr_descr,
                                             d_val,
                                             d_bptr,
                                             d_ind,
                                             bd,
                                             csr_descr,
                                             d_val,
                                             d_ptr,
                                             d_ind),
                          st_ok);
                EXPECT_EQ(d_ptr.read(mb * bd + 1), std::vector<I>(mb * bd + 1, base));
            }

            // With blocks, the output arrays are required.
            const I   mb = 2, nb = 2, nnzb = 2;
            DevBuf<I> d_bptr(std::vector<I>{base, base + 1, base + 2}),
                d_bind(std::vector<I>{base, base + 1});
            DevBuf<float> d_bval(std::vector<float>(nnzb * bd * bd, 1.0f));
            DevBuf<I>     d_ptr(mb * bd + 1), d_ind(nnzb * bd * bd);
            DevBuf<float> d_val(nnzb * bd * bd);
            expect_null_args(
                [&](float* val, I* ind) {
                    return rocsparse_sbsr2csr(handle,
                                              dir_r,
                                              mb,
                                              nb,
                                              bsr_descr,
                                              d_bval,
                                              d_bptr,
                                              d_bind,
                                              bd,
                                              csr_descr,
                                              val,
                                              d_ptr,
                                              ind);
                },
                d_val.get(),
                d_ind.get());
        }
    }

    // ------------------------------------------------------------------------------
    // ell2csr_nnz / sell2csr
    // ------------------------------------------------------------------------------

    TEST_F(ConvGebsr, Ell2Csr)
    {
        const I m = 5, n = 6, width = 3;
        const I counts[m] = {2, 0, 3, 1, 3};
        for(I base : {0, 1})
        {
            // ELL arrays are column major, padded with -1.
            std::vector<I>     e_ind(m * width, -1), ptr = {base}, ind;
            std::vector<float> e_val(m * width, 0.0f), val;
            for(I i = 0; i < m; ++i)
            {
                for(I k = 0; k < counts[i]; ++k)
                {
                    e_ind[i + k * m] = (i + 2 * k) % n + base;
                    e_val[i + k * m] = 1.0f + i * 10 + k;
                    ind.push_back(e_ind[i + k * m]);
                    val.push_back(e_val[i + k * m]);
                }
                ptr.push_back(base + static_cast<I>(ind.size()));
            }
            const I nnz = ind.size();

            MatDescr      ell_descr(base_of(base)), csr_descr(base_of(base));
            DevBuf<I>     d_eind(e_ind);
            DevBuf<float> d_eval(e_val);
            for(bool device_mode : {false, true})
            {
                SCOPED_TRACE("base " + std::to_string(base) + " device mode "
                             + std::to_string(device_mode));
                DevBuf<I> d_ptr(m + 1);
                ASSERT_EQ(
                    nnz_via(handle,
                            device_mode,
                            [&](I* t) {
                                return rocsparse_ell2csr_nnz(
                                    handle, m, n, ell_descr, width, d_eind, csr_descr, d_ptr, t);
                            }),
                    nnz);
                EXPECT_EQ(d_ptr.read(m + 1), ptr);

                DevBuf<I>     d_ind(nnz);
                DevBuf<float> d_val(nnz);
                ASSERT_EQ(rocsparse_sell2csr(handle,
                                             m,
                                             n,
                                             ell_descr,
                                             width,
                                             d_eval,
                                             d_eind,
                                             csr_descr,
                                             d_val,
                                             d_ptr,
                                             d_ind),
                          st_ok);
                EXPECT_EQ(d_ind.read(nnz), ind);
                EXPECT_EQ(d_val.read(nnz), val);
            }
        }
    }

    TEST_F(ConvGebsr, Ell2CsrEmpty)
    {
        MatDescr      ell_descr;
        DevBuf<I>     d_eind(16), d_ind(8);
        DevBuf<float> d_eval(16), d_val(8);
        for(const auto& c : {std::array<I, 4>{0, 4, 2, 0}, {4, 0, 2, 1}, {4, 4, 0, 0}})
        {
            SCOPED_TRACE("m " + std::to_string(c[0]) + " n " + std::to_string(c[1]) + " width "
                         + std::to_string(c[2]) + " base " + std::to_string(c[3]));
            MatDescr  csr_descr(base_of(c[3]));
            DevBuf<I> d_ptr(std::vector<I>(8, 77));
            I         nnz = -1;
            EXPECT_EQ(rocsparse_ell2csr_nnz(
                          handle, c[0], c[1], ell_descr, c[2], d_eind, csr_descr, d_ptr, &nnz),
                      st_ok);
            EXPECT_EQ(rocsparse_sell2csr(handle,
                                         c[0],
                                         c[1],
                                         ell_descr,
                                         c[2],
                                         d_eval,
                                         d_eind,
                                         csr_descr,
                                         d_val,
                                         d_ptr,
                                         d_ind),
                      st_ok);
        }
    }

    // ------------------------------------------------------------------------------
    // hyb2csr of an empty HYB matrix
    // ------------------------------------------------------------------------------

    TEST_F(ConvGebsr, Hyb2CsrEmpty)
    {
        for(I base : {0, 1})
        {
            MatDescr descr(base_of(base));
            HybMat   hyb;
            ASSERT_EQ(rocsparse_create_hyb_mat(&hyb), st_ok);

            // No rows: csr2hyb clears the HYB matrix.
            DevBuf<I>     d_ptr(std::vector<I>(1, base)), d_ind(0), o_ptr(std::vector<I>(2, 77));
            DevBuf<float> d_val(0);
            ASSERT_EQ(
                rocsparse_scsr2hyb(
                    handle, 0, 4, descr, d_val, d_ptr, d_ind, hyb, 0, rocsparse_hyb_partition_auto),
                st_ok);
            size_t size = 123;
            ASSERT_EQ(rocsparse_hyb2csr_buffer_size(handle, descr, hyb, d_ptr, &size), st_ok);
            DevBuf<char> d_buffer(size);
            ASSERT_EQ(rocsparse_shyb2csr(handle, descr, hyb, d_val, o_ptr, d_ind, d_buffer), st_ok);
            EXPECT_EQ(o_ptr.read(1)[0], base);
        }
    }
} // namespace
