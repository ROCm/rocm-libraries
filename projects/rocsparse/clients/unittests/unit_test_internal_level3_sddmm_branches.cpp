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
// rocsparse_sddmm with rocsparse_sddmm_alg_dense on CSR, CSC, ELL and COO-AoS.
// alg_dense converts C to dense (ell2dense / coo2dense_aos / csx2dense) and
// then calls rocBLAS gemm. A build without rocBLAS returns not_implemented
// after that conversion; the post-gemm sample kernel runs only when rocBLAS is
// linked, so without it the test skips after running the conversions.
//
#include "unit_test_extra_common.hpp"

using namespace extra_ut;
using namespace rocsparse_ut;

namespace
{
    constexpr auto I32 = rocsparse_indextype_i32;
    constexpr auto F32 = rocsparse_datatype_f32_r;
    constexpr auto ZB  = rocsparse_index_base_zero;

    constexpr float alpha = 2.0f;
    constexpr float beta  = 0.5f;

    // Column-major A (m x k) and B (k x n) with distinct small integers, so every
    // (row, col) gives a different exact product in float.
    float a_at(int64_t i, int64_t p)
    {
        return float(1 + i + 2 * p);
    }
    float b_at(int64_t p, int64_t j)
    {
        return float(1 + 3 * p + j);
    }

    // With null_buffer the sddmm call is first issued without a buffer, which
    // must be rejected before gemm. Returns the status of the real call.
    rocsparse_status run_dense(rocsparse_handle      handle,
                               rocsparse_spmat_descr C,
                               int64_t               m,
                               int64_t               n,
                               int64_t               k,
                               bool                  null_buffer)
    {
        std::vector<float> hA(size_t(m * k)), hB(size_t(k * n));
        for(int64_t p = 0; p < k; ++p)
        {
            for(int64_t i = 0; i < m; ++i)
                hA[i + p * m] = a_at(i, p);
            for(int64_t j = 0; j < n; ++j)
                hB[p + j * k] = b_at(p, j);
        }
        DevBuf<float> A(hA);
        DevBuf<float> B(hB);
        if(!A.ptr || !B.ptr)
            return rocsparse_status_memory_error;
        DnmatH mA, mB;
        auto st = rocsparse_create_dnmat_descr(&mA.p, m, k, m, A.ptr, F32, rocsparse_order_column);
        if(st == rocsparse_status_success)
            st = rocsparse_create_dnmat_descr(&mB.p, k, n, k, B.ptr, F32, rocsparse_order_column);
        if(st != rocsparse_status_success)
            return st;

        const auto N     = rocsparse_operation_none;
        const auto alg   = rocsparse_sddmm_alg_dense;
        size_t     bytes = 0;
        st               = rocsparse_sddmm_buffer_size(
            handle, N, N, &alpha, mA.p, mB.p, &beta, C, F32, alg, &bytes);
        if(st != rocsparse_status_success)
            return st;
        if(null_buffer)
        {
            EXPECT_GT(bytes, size_t(0));
            EXPECT_EQ(
                rocsparse_sddmm(handle, N, N, &alpha, mA.p, mB.p, &beta, C, F32, alg, nullptr),
                rocsparse_status_invalid_pointer);
        }
        DevBuf<char> tmp{std::vector<char>(bytes)};
        if(!tmp.ptr)
            return rocsparse_status_memory_error;
        st = rocsparse_sddmm_preprocess(
            handle, N, N, &alpha, mA.p, mB.p, &beta, C, F32, alg, tmp.ptr);
        if(st == rocsparse_status_success)
            st = rocsparse_sddmm(handle, N, N, &alpha, mA.p, mB.p, &beta, C, F32, alg, tmp.ptr);
        if(st == rocsparse_status_success)
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        return st;
    }

    // Runs alg_dense on C, whose stored value v sits at (row[v], col[v]) and
    // started at c0[v]; row[v] < 0 marks ELL padding. On success every stored
    // value must be alpha * (A * B)(row, col) + beta * c0. Returns false when the
    // gemm is not available (no rocBLAS), i.e. no value was checked.
    bool check_dense(rocsparse_handle            handle,
                     rocsparse_spmat_descr       C,
                     const float*                val,
                     int64_t                     m,
                     int64_t                     n,
                     int64_t                     k,
                     const std::vector<int32_t>& row,
                     const std::vector<int32_t>& col,
                     const std::vector<float>&   c0,
                     bool                        null_buffer = false)
    {
        const auto st = run_dense(handle, C, m, n, k, null_buffer);
#ifdef ROCSPARSE_WITH_ROCBLAS
        EXPECT_EQ(st, rocsparse_status_success);
#else
        EXPECT_TRUE(st == rocsparse_status_success || st == rocsparse_status_not_implemented) << st;
#endif
        if(st != rocsparse_status_success)
            return false;
        const auto host = to_host(val, c0.size());
        for(size_t v = 0; v < c0.size(); ++v)
        {
            if(row[v] < 0)
                continue;
            float ab = 0.0f;
            for(int64_t p = 0; p < k; ++p)
                ab += a_at(row[v], p) * b_at(p, col[v]);
            EXPECT_FLOAT_EQ(host[v], alpha * ab + beta * c0[v]) << "index " << v;
        }
        return true;
    }

    // Distinct even initial values, so beta * c0 is exact and no result equals c0.
    std::vector<float> initial_values(size_t n)
    {
        std::vector<float> c0(n);
        for(size_t v = 0; v < n; ++v)
            c0[v] = 2.0f * float(v + 1);
        return c0;
    }

    class SddmmBranches : public HandleTest
    {
    };
} // namespace

TEST_F(SddmmBranches, alg_dense)
{
    bool checked = true;

    // Non-symmetric 3x4 pattern, k = 3, in CSR (with the null-buffer check),
    // CSC, ELL and COO-AoS:
    //   row 0: cols 1, 3   row 1: col 0   row 2: cols 2, 3
    constexpr int64_t m = 3, n = 4, k = 3;
    {
        const std::vector<int32_t> row{0, 0, 1, 2, 2}, col{1, 3, 0, 2, 3};
        const auto                 c0 = initial_values(row.size());
        DevBuf<int32_t>            ptr(std::vector<int32_t>{0, 2, 3, 5}), ind(col);
        DevBuf<float>              val(c0);
        SpmatH                     C;
        ASSERT_EQ(
            rocsparse_create_csr_descr(&C.p, m, n, 5, ptr.ptr, ind.ptr, val.ptr, I32, I32, ZB, F32),
            rocsparse_status_success);
        checked &= check_dense(handle, C.p, val.ptr, m, n, k, row, col, c0, true);
    }
    {
        const std::vector<int32_t> row{1, 0, 2, 0, 2}, col{0, 1, 2, 3, 3};
        const auto                 c0 = initial_values(row.size());
        DevBuf<int32_t>            ptr(std::vector<int32_t>{0, 1, 2, 3, 5}), ind(row);
        DevBuf<float>              val(c0);
        SpmatH                     C;
        ASSERT_EQ(
            rocsparse_create_csc_descr(&C.p, m, n, 5, ptr.ptr, ind.ptr, val.ptr, I32, I32, ZB, F32),
            rocsparse_status_success);
        checked &= check_dense(handle, C.p, val.ptr, m, n, k, row, col, c0);
    }
    {
        // Width 2, entry (i, p) stored at p * m + i; row 1 is padded with -1.
        const std::vector<int32_t> row{0, 1, 2, 0, -1, 2}, col{1, 0, 2, 3, -1, 3};
        const auto                 c0 = initial_values(row.size());
        DevBuf<int32_t>            ind(col);
        DevBuf<float>              val(c0);
        SpmatH                     C;
        ASSERT_EQ(rocsparse_create_ell_descr(&C.p, m, n, ind.ptr, val.ptr, 2, I32, ZB, F32),
                  rocsparse_status_success);
        checked &= check_dense(handle, C.p, val.ptr, m, n, k, row, col, c0);
    }
    {
        const std::vector<int32_t> row{0, 0, 1, 2, 2}, col{1, 3, 0, 2, 3};
        const auto                 c0 = initial_values(row.size());
        std::vector<int32_t>       rc;
        for(size_t v = 0; v < row.size(); ++v)
            rc.insert(rc.end(), {row[v], col[v]});
        DevBuf<int32_t> aos(rc);
        DevBuf<float>   val(c0);
        SpmatH          C;
        ASSERT_EQ(rocsparse_create_coo_aos_descr(&C.p, m, n, 5, aos.ptr, val.ptr, I32, ZB, F32),
                  rocsparse_status_success);
        checked &= check_dense(handle, C.p, val.ptr, m, n, k, row, col, c0);
    }

    // nnz == 0: the buffer-size query hits the dense quick return.
    {
        DevBuf<int32_t> zptr(std::vector<int32_t>{0, 0});
        DevBuf<float>   one_a(std::vector<float>{1.0f});
        SpmatH          C;
        DnmatH          mA, mB;
        ASSERT_EQ(rocsparse_create_csr_descr(
                      &C.p, 1, 1, 0, zptr.ptr, zptr.ptr, one_a.ptr, I32, I32, ZB, F32),
                  rocsparse_status_success);
        ASSERT_EQ(
            rocsparse_create_dnmat_descr(&mA.p, 1, 1, 1, one_a.ptr, F32, rocsparse_order_column),
            rocsparse_status_success);
        ASSERT_EQ(
            rocsparse_create_dnmat_descr(&mB.p, 1, 1, 1, one_a.ptr, F32, rocsparse_order_column),
            rocsparse_status_success);
        size_t bytes = 99;
        ASSERT_EQ(rocsparse_sddmm_buffer_size(handle,
                                              rocsparse_operation_none,
                                              rocsparse_operation_none,
                                              &alpha,
                                              mA.p,
                                              mB.p,
                                              &beta,
                                              C.p,
                                              F32,
                                              rocsparse_sddmm_alg_dense,
                                              &bytes),
                  rocsparse_status_success);
        EXPECT_EQ(bytes, size_t(0));
    }

    // Sample-kernel width ladder: avg_nnz = nnz / m, one row of n entries.
    for(int w : {1, 2, 3, 6, 10, 20, 40})
    {
        std::vector<int32_t> cols(w);
        for(int c = 0; c < w; ++c)
            cols[c] = c;
        const auto      c0 = initial_values(size_t(w));
        DevBuf<int32_t> dptr(std::vector<int32_t>{0, w}), dcol(cols);
        DevBuf<float>   dval(c0);
        SpmatH          C;
        ASSERT_EQ(rocsparse_create_csr_descr(
                      &C.p, 1, w, w, dptr.ptr, dcol.ptr, dval.ptr, I32, I32, ZB, F32),
                  rocsparse_status_success);
        SCOPED_TRACE(w);
        checked &= check_dense(
            handle, C.p, dval.ptr, 1, w, 2, std::vector<int32_t>(size_t(w), 0), cols, c0);
    }

    if(!checked)
        GTEST_SKIP() << "alg_dense returned not_implemented (no rocBLAS): values not checked";
}
