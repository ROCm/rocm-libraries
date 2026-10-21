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
// rocsparse_?gemmi: C = alpha * A * B^T + beta * C with a dense A and a sparse
// B. Covers the launch configurations for every number of rows m, the quick
// returns (k == 0, alpha == 0, empty C) and host / device scalars. The whole C
// matrix, including its leading-dimension padding, is compared with a host
// reference.
//
#include "unit_test_extra_common.hpp"

using namespace extra_ut;

namespace
{
    struct GemmiCase
    {
        int64_t m      = 7;
        int64_t n      = 6;
        int64_t k      = 5;
        cplx    alpha  = cplx(2.0, 0.0);
        cplx    beta   = cplx(-1.0, 0.0);
        bool    device = false;
        int64_t pad    = 2;
    };

    template <typename T>
    void run_gemmi(rocsparse_handle handle, const GemmiCase& c)
    {
        SCOPED_TRACE("m=" + std::to_string(c.m) + " n=" + std::to_string(c.n)
                     + " k=" + std::to_string(c.k) + " device=" + std::to_string(c.device));
        Rng         rng(static_cast<uint64_t>(c.m * 7 + c.n * 3 + c.k) + 11);
        const Csr   B  = random_csr<T>(c.n, c.k, 4, rng); // n x k, op(B) = B^T is k x n
        const Dense Ad = random_dense<T>(c.m, c.k, rng);
        const Dense C0 = random_dense<T>(c.m, c.n, rng);

        const int64_t lda = min_ld(Ad, rocsparse_order_column) + c.pad;
        const int64_t ldc = min_ld(C0, rocsparse_order_column) + c.pad;

        DevBuf<T>             dA  = up_raw<T>(store_dense<T>(Ad, rocsparse_order_column, lda));
        DevBuf<T>             dC  = up_raw<T>(store_dense<T>(C0, rocsparse_order_column, ldc));
        DevBuf<rocsparse_int> ptr = up_idx<rocsparse_int>(B.ptr);
        DevBuf<rocsparse_int> col = up_idx<rocsparse_int>(B.col);
        DevBuf<T>             val = up_val<T>(B.val);
        ASSERT_TRUE(dA.ptr && dC.ptr && ptr.ptr && col.ptr && val.ptr);

        MatDescrH        descr;
        Scalar<T>        alpha(c.alpha, c.device);
        Scalar<T>        beta(c.beta, c.device);
        PointerModeGuard mode(handle, mode_of(c.device));

        ASSERT_EQ(Api<T>::gemmi(handle,
                                rocsparse_operation_none,
                                rocsparse_operation_transpose,
                                static_cast<rocsparse_int>(c.m),
                                static_cast<rocsparse_int>(c.n),
                                static_cast<rocsparse_int>(c.k),
                                static_cast<rocsparse_int>(B.nnz()),
                                alpha.typed(),
                                static_cast<const T*>(dA.ptr),
                                static_cast<rocsparse_int>(lda),
                                descr.d,
                                static_cast<const T*>(val.ptr),
                                static_cast<const rocsparse_int*>(ptr.ptr),
                                static_cast<const rocsparse_int*>(col.ptr),
                                beta.typed(),
                                dC.ptr,
                                static_cast<rocsparse_int>(ldc)),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const Dense want = axpby(to_c<T>(alpha.h),
                                 matmul(Ad, apply_op(dense_of(B), rocsparse_operation_transpose)),
                                 to_c<T>(beta.h),
                                 C0);
        expect_dense_equal<T>(dC.host(), want, rocsparse_order_column, ldc, "C");
        expect_dense_equal<T>(dA.host(), Ad, rocsparse_order_column, lda, "A");
    }

    struct ExtraGemmi : public rocsparse_ut::HandleTest
    {
    };
}

// The thread-block size depends on m: <= 64, <= 128 and larger.
TEST_F(ExtraGemmi, rows_float)
{
    for(int64_t m : {1, 31, 64, 65, 100, 128, 129, 300})
        for(bool device : {false, true})
        {
            GemmiCase c;
            c.m      = m;
            c.n      = 9;
            c.k      = 11;
            c.device = device;
            run_gemmi<float>(handle, c);
            if(HasFatalFailure())
                return;
        }
}

// k == 0 and alpha == 0 leave C = beta * C. (ldc == m here: these paths scale
// the m * n leading elements of C.)
TEST_F(ExtraGemmi, quick_returns)
{
    for(bool device : {false, true})
    {
        GemmiCase c;
        c.device = device;
        c.pad    = 0;
        c.k      = 0;
        run_gemmi<float>(handle, c);
        if(HasFatalFailure())
            return;
        c.k     = 4;
        c.alpha = cplx(0.0, 0.0);
        run_gemmi<float>(handle, c);
        if(HasFatalFailure())
            return;
        c.beta = cplx(0.0, 0.0);
        run_gemmi<double>(handle, c);
        if(HasFatalFailure())
            return;
    }
}

// m == 0 or n == 0: nothing to compute.
TEST_F(ExtraGemmi, empty_C)
{
    for(bool device : {false, true})
    {
        GemmiCase c;
        c.device = device;
        c.m      = 0;
        run_gemmi<float>(handle, c);
        if(HasFatalFailure())
            return;
        c.m = 5;
        c.n = 0;
        run_gemmi<double>(handle, c);
        if(HasFatalFailure())
            return;
    }
}
