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
//
#include "unit_test_device_level2_common.hpp"

#include <algorithm>
#include <limits>
#include <utility>

using namespace level2;

namespace
{
    struct csr_case
    {
        rocsparse_spmv_alg  alg    = rocsparse_spmv_alg_default;
        rocsparse_operation trans  = rocsparse_operation_none;
        bool                device = false;
        // beta == 0 with y starting as NaN; y is write-only then.
        bool beta_zero_nan_y = false;
    };

    template <typename T>
    void run_csr(rocsparse_handle handle, const host_csr<T>& A, const csr_case& c)
    {
        SCOPED_TRACE(::testing::Message()
                     << "csrmv alg=" << c.alg << " trans=" << c.trans << " device=" << c.device
                     << " base=" << A.base << " m=" << A.m << " n=" << A.n << " nnz=" << A.nnz());

        const size_t xs = trans_size(c.trans, A.m, A.n, true);
        const size_t ys = trans_size(c.trans, A.m, A.n, false);

        const T alpha = rocsparse_ut::scalar<T>(1.5f);
        const T beta  = rocsparse_ut::scalar<T>(c.beta_zero_nan_y ? 0.0f : -0.5f);

        const std::vector<T> x    = make_vector<T>(xs, 2);
        const std::vector<T> y0   = c.beta_zero_nan_y
                                        ? std::vector<T>(ys, std::numeric_limits<T>::quiet_NaN())
                                        : make_vector<T>(ys, 9);
        const std::vector<T> want = csr_gemv(
            A, c.trans, alpha, x, beta, c.beta_zero_nan_y ? std::vector<T>(ys, T(0)) : y0);

        device_csr<T, int32_t, int32_t> dA(A);
        device_vector<T>                dx(x), dy(y0), dalpha(std::vector<T>(1, alpha)),
            dbeta(std::vector<T>(1, beta));
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr && dalpha.ptr && dbeta.ptr);

        spmat_guard mat;
        ASSERT_EQ(rocsparse_create_csr_descr(&mat.d,
                                             A.m,
                                             A.n,
                                             A.nnz(),
                                             dA.ptr,
                                             dA.ind,
                                             dA.val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             to_base(A.base),
                                             rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);
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
        EXPECT_TRUE(expect_near(want, to_host(dy), "csrmv y"));
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
            for(const auto alg : {rocsparse_spmv_alg_csr_rowsplit,
                                  rocsparse_spmv_alg_csr_adaptive,
                                  rocsparse_spmv_alg_csr_lrb,
                                  rocsparse_spmv_alg_csr_nnzsplit})
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

    TEST_F(L2DeviceCsrmvRows, beta_zero_ignores_nan_in_y)
    {
        // With beta == 0, y is write-only. The rows of 10000 entries are shared by several
        // workgroups in the adaptive and LRB algorithms, which add their parts to y atomically;
        // the NaN y starts with must not end up in them.
        const auto A = make_csr_by_length<float>(
            2000, 12007, 0, [](int32_t i) { return (i % 250) == 7 ? 10000 : 1 + (i % 4); });
        for(const auto alg : {rocsparse_spmv_alg_default,
                              rocsparse_spmv_alg_csr_rowsplit,
                              rocsparse_spmv_alg_csr_adaptive,
                              rocsparse_spmv_alg_csr_lrb,
                              rocsparse_spmv_alg_csr_nnzsplit})
            for(const bool device : {false, true})
            {
                csr_case c;
                c.alg             = alg;
                c.device          = device;
                c.beta_zero_nan_y = true;
                run_csr<float>(handle, A, c);
                if(HasFailure())
                    return;
            }
    }

    TEST_F(L2DeviceCsrmvRows, algorithms_share_one_descriptor)
    {
        // The preprocess stage keeps its analysis in the matrix descriptor. Preprocessing with
        // another algorithm or operation replaces it, and compute refuses an analysis that was
        // built for another algorithm.
        const auto A = make_csr_by_length<float>(
            2000, 12007, 0, [](int32_t i) { return (i % 250) == 7 ? 10000 : 1 + (i % 4); });
        device_csr<float, int32_t, int32_t> dA(A);
        ASSERT_TRUE(dA.ok());
        spmat_guard mat;
        ASSERT_EQ(rocsparse_create_csr_descr(&mat.d,
                                             A.m,
                                             A.n,
                                             A.nnz(),
                                             dA.ptr,
                                             dA.ind,
                                             dA.val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             to_base(A.base),
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        scoped_pointer_mode mode(handle, false);

        const float alpha = 1.5f;
        const float beta  = -0.5f;

        // Runs one product on fresh x and y, with or without the preprocess stage, and checks
        // the status of the compute stage and, if it succeeds, y.
        const auto run = [&](rocsparse_spmv_alg  alg,
                             rocsparse_operation trans,
                             bool                preprocess,
                             rocsparse_status    want = rocsparse_status_success) {
            SCOPED_TRACE(::testing::Message() << "alg=" << alg << " trans=" << trans);
            const std::vector<float> x  = make_vector<float>(trans_size(trans, A.m, A.n, true), 2);
            const std::vector<float> y0 = make_vector<float>(trans_size(trans, A.m, A.n, false), 9);
            device_vector<float>     dx(x), dy(y0);
            dnvec_guard              vx, vy;
            ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, x.size(), dx, rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, y0.size(), dy, rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            const auto spmv = [&](rocsparse_spmv_stage stage, size_t* bytes, void* buffer) {
                return rocsparse_spmv(handle,
                                      trans,
                                      &alpha,
                                      mat.d,
                                      vx.d,
                                      &beta,
                                      vy.d,
                                      rocsparse_datatype_f32_r,
                                      alg,
                                      stage,
                                      bytes,
                                      buffer);
            };
            size_t bytes = 0;
            ASSERT_EQ(spmv(rocsparse_spmv_stage_buffer_size, &bytes, nullptr),
                      rocsparse_status_success);
            device_vector<char> buffer(std::max<size_t>(bytes, 1));
            if(preprocess)
                ASSERT_EQ(spmv(rocsparse_spmv_stage_preprocess, &bytes, buffer),
                          rocsparse_status_success);
            ASSERT_EQ(spmv(rocsparse_spmv_stage_compute, &bytes, buffer), want);
            if(want == rocsparse_status_success)
                EXPECT_TRUE(
                    expect_near(csr_gemv(A, trans, alpha, x, beta, y0), to_host(dy), "spmv y"));
        };

        const auto op_n = rocsparse_operation_none;
        const auto op_t = rocsparse_operation_transpose;
        const std::pair<rocsparse_spmv_alg, rocsparse_operation> steps[]
            = {{rocsparse_spmv_alg_csr_adaptive, op_n},
               {rocsparse_spmv_alg_csr_lrb, op_n},
               {rocsparse_spmv_alg_csr_nnzsplit, op_n},
               {rocsparse_spmv_alg_csr_rowsplit, op_n},
               {rocsparse_spmv_alg_csr_adaptive, op_n},
               {rocsparse_spmv_alg_csr_nnzsplit, op_t},
               {rocsparse_spmv_alg_csr_lrb, op_n},
               {rocsparse_spmv_alg_default, op_n},
               {rocsparse_spmv_alg_csr_adaptive, op_t},
               {rocsparse_spmv_alg_csr_adaptive, op_n},
               {rocsparse_spmv_alg_csr_nnzsplit, op_n}};
        for(const auto& [alg, trans] : steps)
        {
            run(alg, trans, true);
            if(HasFailure())
                return;
        }

        // The descriptor holds the nnz-split analysis now, which LRB cannot use.
        run(rocsparse_spmv_alg_csr_lrb, op_n, false, rocsparse_status_invalid_value);
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
}
