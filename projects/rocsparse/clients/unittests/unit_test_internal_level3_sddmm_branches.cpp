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
// linked.
//
#include "unit_test_extra_common.hpp"

using namespace extra_ut;
using namespace rocsparse_ut;

namespace
{
    constexpr auto I32 = rocsparse_indextype_i32;
    constexpr auto F32 = rocsparse_datatype_f32_r;
    constexpr auto ZB  = rocsparse_index_base_zero;

    // A and B are m x k and k x n dense ones, so every sampled entry equals k.
    // With null_buffer the sddmm call is first issued without a buffer, which
    // must be rejected before gemm. Returns the status of the real call.
    rocsparse_status run_dense(rocsparse_handle      handle,
                               rocsparse_spmat_descr C,
                               int64_t               m,
                               int64_t               n,
                               int64_t               k,
                               bool                  null_buffer = false)
    {
        DevBuf<float> A(std::vector<float>(size_t(m * k), 1.0f));
        DevBuf<float> B(std::vector<float>(size_t(k * n), 1.0f));
        if(!A.ptr || !B.ptr)
            return rocsparse_status_memory_error;
        DnmatH mA, mB;
        auto st = rocsparse_create_dnmat_descr(&mA.p, m, k, m, A.ptr, F32, rocsparse_order_column);
        if(st == rocsparse_status_success)
            st = rocsparse_create_dnmat_descr(&mB.p, k, n, k, B.ptr, F32, rocsparse_order_column);
        if(st != rocsparse_status_success)
            return st;

        const float one = 1.0f, zero = 0.0f;
        const auto  N     = rocsparse_operation_none;
        const auto  alg   = rocsparse_sddmm_alg_dense;
        size_t      bytes = 0;
        st                = rocsparse_sddmm_buffer_size(
            handle, N, N, &one, mA.p, mB.p, &zero, C, F32, alg, &bytes);
        if(st != rocsparse_status_success)
            return st;
        if(null_buffer)
        {
            EXPECT_GT(bytes, size_t(0));
            EXPECT_EQ(rocsparse_sddmm(handle, N, N, &one, mA.p, mB.p, &zero, C, F32, alg, nullptr),
                      rocsparse_status_invalid_pointer);
        }
        DevBuf<char> tmp{std::vector<char>(bytes)};
        if(!tmp.ptr)
            return rocsparse_status_memory_error;
        st = rocsparse_sddmm_preprocess(
            handle, N, N, &one, mA.p, mB.p, &zero, C, F32, alg, tmp.ptr);
        if(st == rocsparse_status_success)
            st = rocsparse_sddmm(handle, N, N, &one, mA.p, mB.p, &zero, C, F32, alg, tmp.ptr);
        if(st == rocsparse_status_success)
            (void)hipDeviceSynchronize();
        return st;
    }

    // Runs alg_dense on C and, when it succeeds, checks that all n stored
    // values equal `want`.
    void check_dense(rocsparse_handle      handle,
                     rocsparse_spmat_descr C,
                     const float*          val,
                     int64_t               m,
                     int64_t               n,
                     int64_t               k,
                     size_t                nvals,
                     float                 want,
                     bool                  null_buffer = false)
    {
        const auto st = run_dense(handle, C, m, n, k, null_buffer);
        EXPECT_TRUE(st == rocsparse_status_success || st == rocsparse_status_not_implemented) << st;
        if(st == rocsparse_status_success)
        {
            const auto host = to_host(val, nvals);
            for(size_t i = 0; i < nvals; ++i)
                EXPECT_FLOAT_EQ(host[i], want) << "index " << i;
        }
    }

    class SddmmBranches : public HandleTest
    {
    };
} // namespace

TEST_F(SddmmBranches, alg_dense)
{
    const std::vector<int32_t> diag{0, 1, 2};
    const std::vector<float>   ones(3, 1.0f);

    // 3x3 diagonal in CSR (with the null-buffer check), CSC, ELL and COO-AoS.
    DevBuf<int32_t> ptr(std::vector<int32_t>{0, 1, 2, 3}), ind(diag),
        aos(std::vector<int32_t>{0, 0, 1, 1, 2, 2});
    {
        DevBuf<float> val(ones);
        SpmatH        C;
        ASSERT_EQ(
            rocsparse_create_csr_descr(&C.p, 3, 3, 3, ptr.ptr, ind.ptr, val.ptr, I32, I32, ZB, F32),
            rocsparse_status_success);
        check_dense(handle, C.p, val.ptr, 3, 3, 3, 3, 3.0f, true);
    }
    {
        DevBuf<float> val(ones);
        SpmatH        C;
        ASSERT_EQ(
            rocsparse_create_csc_descr(&C.p, 3, 3, 3, ptr.ptr, ind.ptr, val.ptr, I32, I32, ZB, F32),
            rocsparse_status_success);
        check_dense(handle, C.p, val.ptr, 3, 3, 3, 3, 3.0f);
    }
    {
        DevBuf<float> val(ones);
        SpmatH        C;
        ASSERT_EQ(rocsparse_create_ell_descr(&C.p, 3, 3, ind.ptr, val.ptr, 1, I32, ZB, F32),
                  rocsparse_status_success);
        check_dense(handle, C.p, val.ptr, 3, 3, 3, 3, 3.0f);
    }
    {
        DevBuf<float> val(ones);
        SpmatH        C;
        ASSERT_EQ(rocsparse_create_coo_aos_descr(&C.p, 3, 3, 3, aos.ptr, val.ptr, I32, ZB, F32),
                  rocsparse_status_success);
        check_dense(handle, C.p, val.ptr, 3, 3, 3, 3, 3.0f);
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
        const float one = 1.0f, zero = 0.0f;
        size_t      bytes = 99;
        ASSERT_EQ(rocsparse_sddmm_buffer_size(handle,
                                              rocsparse_operation_none,
                                              rocsparse_operation_none,
                                              &one,
                                              mA.p,
                                              mB.p,
                                              &zero,
                                              C.p,
                                              F32,
                                              rocsparse_sddmm_alg_dense,
                                              &bytes),
                  rocsparse_status_success);
        EXPECT_EQ(bytes, size_t(0));
    }

    // Sample-kernel width ladder: avg_nnz = nnz / m, one row of n entries.
    for(int n : {1, 2, 3, 6, 10, 20, 40})
    {
        std::vector<int32_t> cols(n);
        for(int c = 0; c < n; ++c)
            cols[c] = c;
        DevBuf<int32_t> dptr(std::vector<int32_t>{0, n}), dcol(cols);
        DevBuf<float>   dval(std::vector<float>(n, 1.0f));
        SpmatH          C;
        ASSERT_EQ(rocsparse_create_csr_descr(
                      &C.p, 1, n, n, dptr.ptr, dcol.ptr, dval.ptr, I32, I32, ZB, F32),
                  rocsparse_status_success);
        SCOPED_TRACE(n);
        check_dense(handle, C.p, dval.ptr, 1, n, 1, n, 1.0f);
    }
}
