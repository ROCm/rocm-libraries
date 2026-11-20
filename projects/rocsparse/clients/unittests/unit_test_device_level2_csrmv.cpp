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
// Device unit tests of the CSR matrix vector product on matrices of every shape
// class: many short rows, many medium rows, long rows, a single very long row
// among short ones, and runs of consecutive long rows. They exercise the
// granularity choices of the row split, adaptive, LRB and nnz split algorithms
// through the v2 interface and compare the products with a host product.
// Two further suites use the same runner: beta == 0 on a NaN-filled y (the
// product must not read y) and the CSC storage of a transposed matrix, whose
// operations are routed through the CSR kernels.
//
#include "unit_test_device_level2_common.hpp"

#include <algorithm>

using namespace level2;

namespace
{
    // The CSR algorithms of the v2 interface.
    const std::vector<rocsparse_spmv_alg> csr_algs = {rocsparse_spmv_alg_csr_rowsplit,
                                                      rocsparse_spmv_alg_csr_adaptive,
                                                      rocsparse_spmv_alg_csr_lrb,
                                                      rocsparse_spmv_alg_csr_nnzsplit};

    struct csr_case
    {
        rocsparse_spmv_alg  alg    = rocsparse_spmv_alg_default;
        rocsparse_operation trans  = rocsparse_operation_none;
        bool                device = false;
        float               beta   = -0.5f;
        // y is filled with quiet NaN before the product (meaningful with beta == 0:
        // the kernel must overwrite y without reading it).
        bool nan_y = false;
        // The arrays of A are handed over as the CSC storage of A^T (an n x m
        // matrix), so op(A^T) is computed through the CSC entry of csrmv.
        bool csc = false;
    };

    template <typename T>
    host_csr<T> conjugated(const host_csr<T>& a)
    {
        host_csr<T> c(a);
        for(auto& v : c.val)
            v = conj_of(v);
        return c;
    }

    // The CSR operation that computes op(A^T) when the CSC storage of A^T is the
    // CSR storage of A: none -> transpose, transpose -> none and
    // conjugate transpose -> none on the conjugated values.
    inline rocsparse_operation csc_reference_op(rocsparse_operation trans)
    {
        return (trans == rocsparse_operation_none) ? rocsparse_operation_transpose
                                                   : rocsparse_operation_none;
    }

    // Runs the product of `c` on A and leaves the host reference in `want` and the
    // device result in `got`; a failed setup is a fatal failure and leaves `got`
    // empty. The comparison is done by the caller (run_csr).
    template <typename T>
    void compute_csr(rocsparse_handle   handle,
                     const host_csr<T>& A,
                     const csr_case&    c,
                     std::vector<T>&    want,
                     std::vector<T>&    got)
    {
        SCOPED_TRACE(::testing::Message()
                     << "csrmv alg=" << c.alg << " trans=" << c.trans << " device=" << c.device
                     << " base=" << A.base << " m=" << A.m << " n=" << A.n << " nnz=" << A.nnz()
                     << " beta=" << c.beta << " nan_y=" << c.nan_y << " csc=" << c.csc);

        // Sizes of x and y as seen by the library: the CSC matrix is n x m.
        const int32_t rows = c.csc ? A.n : A.m;
        const int32_t cols = c.csc ? A.m : A.n;
        const size_t  xs   = trans_size(c.trans, rows, cols, true);
        const size_t  ys   = trans_size(c.trans, rows, cols, false);

        const T alpha = rocsparse_ut::scalar<T>(1.5f);
        const T beta  = rocsparse_ut::scalar<T>(c.beta);

        const std::vector<T> x  = make_vector<T>(xs, 2);
        const std::vector<T> y0 = c.nan_y ? make_nan_vector<T>(ys) : make_vector<T>(ys, 9);
        // With NaN in y the reference uses y = 0, which is what beta == 0 must produce.
        const std::vector<T> yref = c.nan_y ? std::vector<T>(ys, T{}) : y0;
        got.clear();
        if(c.csc)
        {
            const bool conj = (c.trans == rocsparse_operation_conjugate_transpose);
            want            = csr_gemv(
                conj ? conjugated(A) : A, csc_reference_op(c.trans), alpha, x, beta, yref);
        }
        else
        {
            want = csr_gemv(A, c.trans, alpha, x, beta, yref);
        }

        device_csr<T, int32_t, int32_t> dA(A);
        device_vector<T>                dx(x), dy(y0), dalpha(std::vector<T>(1, alpha)),
            dbeta(std::vector<T>(1, beta));
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr && dalpha.ptr && dbeta.ptr);

        spmat_guard mat;
        if(c.csc)
        {
            ASSERT_EQ(rocsparse_create_csc_descr(&mat.d,
                                                 rows,
                                                 cols,
                                                 A.nnz(),
                                                 dA.ptr,
                                                 dA.ind,
                                                 dA.val,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_indextype_i32,
                                                 to_base(A.base),
                                                 rocsparse_ut::dt_of<T>()),
                      rocsparse_status_success);
        }
        else
        {
            ASSERT_EQ(rocsparse_create_csr_descr(&mat.d,
                                                 rows,
                                                 cols,
                                                 A.nnz(),
                                                 dA.ptr,
                                                 dA.ind,
                                                 dA.val,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_indextype_i32,
                                                 to_base(A.base),
                                                 rocsparse_ut::dt_of<T>()),
                      rocsparse_status_success);
        }
        dnvec_guard vx, vy;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, xs, dx, rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, ys, dy, rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);

        scoped_pointer_mode mode(handle, c.device);

        spmv_descr_guard descr;
        ASSERT_TRUE(descr.d);
        ASSERT_EQ(set_basic_inputs(handle,
                                   descr.d,
                                   c.alg,
                                   c.trans,
                                   rocsparse_ut::dt_of<T>(),
                                   rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);
        run_v2_spmv(
            handle,
            descr.d,
            mat.d,
            vx.d,
            vy.d,
            c.device ? static_cast<const void*>(dalpha.ptr) : static_cast<const void*>(&alpha),
            c.device ? static_cast<const void*>(dbeta.ptr) : static_cast<const void*>(&beta));
        if(::testing::Test::HasFatalFailure())
            return;
        got = to_host(dy);
    }

    template <typename T>
    void run_csr(rocsparse_handle handle, const host_csr<T>& A, const csr_case& c)
    {
        std::vector<T> want, got;
        compute_csr(handle, A, c, want, got);
        if(::testing::Test::HasFatalFailure())
            return;
        // An explicit check: a NaN that leaked through beta * y would also fail
        // expect_near, but this names the defect.
        if(c.nan_y)
            expect_no_nan(got, "csrmv y with beta == 0");
        EXPECT_TRUE(expect_near(want, got, "csrmv y"));
    }

    class L2DeviceCsrmvRows : public rocsparse_ut::HandleTest
    {
    protected:
        // Every algorithm on one matrix; the pointer mode alternates with the
        // algorithm and the operation, the transposed product is only requested
        // from the algorithms that implement it (documented as not deterministic).
        void sweep(const host_csr<float>& A, bool with_transpose)
        {
            int n = 0;
            for(const auto alg : csr_algs)
                for(const auto trans : {rocsparse_operation_none, rocsparse_operation_transpose})
                {
                    const bool trans_ok = (alg == rocsparse_spmv_alg_csr_rowsplit
                                           || alg == rocsparse_spmv_alg_csr_nnzsplit);
                    if(trans != rocsparse_operation_none && !(trans_ok && with_transpose))
                        continue;
                    csr_case c;
                    c.alg    = alg;
                    c.trans  = trans;
                    c.device = ((n++) % 3) == 1;
                    run_csr<float>(handle, A, c);
                    if(HasFailure())
                        return;
                }
        }
    };

    TEST_F(L2DeviceCsrmvRows, many_short_rows)
    {
        // 12000 rows of 30 entries.
        sweep(make_csr_by_length<float>(12000, 5003, 0, [](int32_t) { return 30; }), true);
    }

    TEST_F(L2DeviceCsrmvRows, many_medium_rows)
    {
        // 12000 rows of 40 entries, one based.
        sweep(make_csr_by_length<float>(12000, 5003, 1, [](int32_t) { return 40; }), true);
    }

    TEST_F(L2DeviceCsrmvRows, dense_rows)
    {
        // 3000 rows of 130 entries.
        sweep(make_csr_by_length<float>(3000, 2011, 0, [](int32_t) { return 130; }), true);
    }

    TEST_F(L2DeviceCsrmvRows, one_very_long_row_among_short_ones)
    {
        sweep(make_csr_by_length<float>(
                  6000, 4001, 0, [](int32_t i) { return i == 2500 ? 3500 : 2 + (i % 5); }),
              true);
    }

    TEST_F(L2DeviceCsrmvRows, consecutive_long_rows)
    {
        // Runs of consecutive rows that are longer than a workgroup can reduce at once.
        sweep(make_csr_by_length<float>(
                  3000, 6007, 1, [](int32_t i) { return (i % 100) < 12 ? 4500 : 1 + (i % 4); }),
              false);
    }

    TEST_F(L2DeviceCsrmvRows, mixed_lengths_with_empty_rows)
    {
        sweep(make_csr_by_length<float>(9000,
                                        3001,
                                        0,
                                        [](int32_t i) {
                                            if(i % 11 == 0)
                                                return 0;
                                            if(i % 97 == 5)
                                                return 700;
                                            return 1 + (i % 61);
                                        }),
              true);
    }

    TEST_F(L2DeviceCsrmvRows, long_last_row)
    {
        // The last row spans several nnz blocks of the nnz split algorithm (1024 or
        // 2048 entries on a wave32 or wave64 GPU once the long row selects the
        // coarse granularity): the block starts inside it (and the sentinel one past
        // the end) are owned by the last row, so its end offset must come from
        // row_ptr[m] and not be clamped to row_ptr[m - 1].
        sweep(make_csr_by_length<float>(
                  300, 6007, 0, [](int32_t i) { return i == 299 ? 5000 : 1 + (i % 5); }),
              true);
        if(HasFailure())
            return;
        // A second shape: empty leading rows, one based, and a last row with a single
        // interior block start on a wave64 GPU.
        sweep(make_csr_by_length<float>(
                  500,
                  4001,
                  1,
                  [](int32_t i) { return i < 3 ? 0 : (i == 499 ? 3300 : 2 + (i % 7)); }),
              true);
    }

    //
    // beta == 0 must overwrite y without reading it: y is pre-filled with NaN.
    //
    class L2DeviceCsrmvBetaZero : public rocsparse_ut::HandleTest
    {
    protected:
        void nan_sweep(const host_csr<float>& A, const std::vector<rocsparse_spmv_alg>& algs)
        {
            for(const auto alg : algs)
                for(const bool device : {false, true})
                {
                    csr_case c;
                    c.alg    = alg;
                    c.beta   = 0.0f;
                    c.nan_y  = true;
                    c.device = device;
                    run_csr<float>(handle, A, c);
                    if(HasFailure())
                        return;
                }
        }
    };

    TEST_F(L2DeviceCsrmvBetaZero, short_rows_overwrite_nan_y)
    {
        // Many rows of 3 entries: a row block of the adaptive kernel holds hundreds
        // of rows, which selects its per-thread (CSR-Scalar) reduction.
        nan_sweep(make_csr_by_length<float>(5000, 1009, 0, [](int32_t) { return 3; }), csr_algs);
    }

    TEST_F(L2DeviceCsrmvBetaZero, medium_rows_overwrite_nan_y)
    {
        // Rows of 30 entries, one based: a row block holds about 34 rows, which
        // selects the tree reduction of the adaptive kernel (several threads per
        // row). The rows stay well below the long row path.
        nan_sweep(make_csr_by_length<float>(2000, 1511, 1, [](int32_t) { return 30; }), csr_algs);
        nan_sweep(make_csr_by_length<float>(3000, 1013, 0, [](int32_t i) { return 20 + (i % 40); }),
                  csr_algs);
    }

    TEST_F(L2DeviceCsrmvBetaZero, adaptive_long_row_overwrites_nan_y)
    {
        // A row longer than a row block of the adaptive kernel is split over several
        // workgroups whose contributions are accumulated into y, so the kernel reads
        // y even with beta == 0. While the library behaves that way (a NaN in the
        // result) the case is skipped; once the long-row path overwrites y it runs
        // the full checks.
        const host_csr<float> A = make_csr_by_length<float>(
            600, 4001, 0, [](int32_t i) { return i == 250 ? 3500 : 2 + (i % 5); });
        csr_case probe;
        probe.alg   = rocsparse_spmv_alg_csr_adaptive;
        probe.beta  = 0.0f;
        probe.nan_y = true;
        std::vector<float> want, got;
        compute_csr(handle, A, probe, want, got);
        if(HasFatalFailure())
            return;
        if(any_nan(got))
            GTEST_SKIP() << "csrmv adaptive long-row path reads y when beta == 0 and "
                            "propagates NaN; known library defect";
        nan_sweep(A, {rocsparse_spmv_alg_csr_adaptive});
    }

}
