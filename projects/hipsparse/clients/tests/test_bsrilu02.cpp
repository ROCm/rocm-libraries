/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "test.hpp"
#include "testing_bsrilu02.hpp"

#include <cstring>
#include <limits>

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
TEST_ROUTINE(bsrilu02,
             precond,
             arg.M,
             arg.block_dim,
             arg.numericboost,
             arg.boosttol,
             arg.boostval,
             arg.boostvali,
             arg.dirA,
             arg.solve_policy,
             arg.baseA);
#endif

#if(!defined(CUDART_VERSION))
// Known bug: rocSPARSE computes block value offsets in int32, so nnzb * block_dim^2 > INT_MAX
// overflows. Kept strict; the 'known_bug' suite name excludes it from PR gating.
TEST(known_bug_bsrilu02, zbsrilu02_nnzb_block_dim_squared_past_int_max)
{
    const int    block_dim = 64;
    const int    mb        = 524289;
    const int    nnzb      = mb;
    const size_t bb        = size_t(block_dim) * block_dim;
    const size_t nval      = size_t(nnzb) * bb;
    ASSERT_GT(nval, size_t(std::numeric_limits<int>::max()));

    size_t free_mem = 0, total_mem = 0;
    ASSERT_EQ(hipMemGetInfo(&free_mem, &total_mem), hipSuccess);
    if(free_mem < sizeof(hipDoubleComplex) * nval + (size_t(1) << 32))
    {
        GTEST_SKIP() << "Not enough free device memory";
    }

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;
    std::unique_ptr<descr_struct>  unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t            descr = unique_ptr_descr->descr;

    // Block diagonal matrix
    std::vector<int> hptr(mb + 1), hind(nnzb);
    for(int i = 0; i < mb; ++i)
    {
        hptr[i] = hind[i] = i;
    }
    hptr[mb] = nnzb;

    auto dptr_m = hipsparse_unique_ptr{device_malloc(sizeof(int) * (mb + 1)), device_free};
    auto dind_m = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnzb), device_free};
    auto dval_m = hipsparse_unique_ptr{device_malloc(sizeof(hipDoubleComplex) * nval), device_free};
    int*              dptr = (int*)dptr_m.get();
    int*              dind = (int*)dind_m.get();
    hipDoubleComplex* dval = (hipDoubleComplex*)dval_m.get();
    ASSERT_NE(dval, nullptr);
    CHECK_HIP_ERROR(hipMemcpy(dptr, hptr.data(), sizeof(int) * (mb + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dind, hind.data(), sizeof(int) * nnzb, hipMemcpyHostToDevice));

    // Every block is the tridiagonal A = L*U with L = I + 0.5*subdiag, U = 2*I + superdiag.
    // Its in-place LU has no fill-in and is exact: 0.5 below, 2 on, 1 above the diagonal.
    const size_t                  chunk = 8192;
    std::vector<hipDoubleComplex> hin(chunk * bb), hexp(chunk * bb), hout(chunk * bb);

    for(hipsparseDirection_t dir : {HIPSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_COLUMN})
    {
        for(hipsparseSolvePolicy_t policy :
            {HIPSPARSE_SOLVE_POLICY_NO_LEVEL, HIPSPARSE_SOLVE_POLICY_USE_LEVEL})
        {
            SCOPED_TRACE(testing::Message() << "dir=" << dir << " policy=" << policy);

            for(int r = 0; r < block_dim; ++r)
            {
                for(int c = 0; c < block_dim; ++c)
                {
                    size_t k = (dir == HIPSPARSE_DIRECTION_ROW) ? size_t(r) * block_dim + c
                                                                : size_t(c) * block_dim + r;
                    double a = 0.0, f = 0.0;
                    if(r == c)
                    {
                        a = (r == 0) ? 2.0 : 2.5;
                        f = 2.0;
                    }
                    else if(c == r + 1)
                    {
                        a = f = 1.0;
                    }
                    else if(r == c + 1)
                    {
                        a = 1.0;
                        f = 0.5;
                    }
                    for(size_t b = 0; b < chunk; ++b)
                    {
                        hin[b * bb + k]  = make_hipDoubleComplex(a, 0.0);
                        hexp[b * bb + k] = make_hipDoubleComplex(f, 0.0);
                    }
                }
            }
            for(size_t first = 0; first < size_t(nnzb); first += chunk)
            {
                size_t n = std::min(chunk, size_t(nnzb) - first);
                CHECK_HIP_ERROR(hipMemcpy(dval + first * bb,
                                          hin.data(),
                                          sizeof(hipDoubleComplex) * n * bb,
                                          hipMemcpyHostToDevice));
            }

            std::unique_ptr<bsrilu02_struct> unique_ptr_info(new bsrilu02_struct);
            bsrilu02Info_t                   info = unique_ptr_info->info;

            int size;
            ASSERT_EQ(hipsparseZbsrilu02_bufferSize(
                          handle, dir, mb, nnzb, descr, dval, dptr, dind, block_dim, info, &size),
                      HIPSPARSE_STATUS_SUCCESS);
            auto  buf_m = hipsparse_unique_ptr{device_malloc(size), device_free};
            void* buf   = buf_m.get();

            CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
            ASSERT_EQ(hipsparseZbsrilu02_analysis(
                          handle, dir, mb, nnzb, descr, dval, dptr, dind, block_dim, info, policy, buf),
                      HIPSPARSE_STATUS_SUCCESS);
            ASSERT_EQ(hipsparseZbsrilu02(
                          handle, dir, mb, nnzb, descr, dval, dptr, dind, block_dim, info, policy, buf),
                      HIPSPARSE_STATUS_SUCCESS);

            int position;
            EXPECT_EQ(hipsparseXbsrilu02_zeroPivot(handle, info, &position),
                      HIPSPARSE_STATUS_SUCCESS);
            EXPECT_EQ(position, -1);

            size_t bad_chunks = 0;
            for(size_t first = 0; first < size_t(nnzb); first += chunk)
            {
                size_t n = std::min(chunk, size_t(nnzb) - first);
                CHECK_HIP_ERROR(hipMemcpy(hout.data(),
                                          dval + first * bb,
                                          sizeof(hipDoubleComplex) * n * bb,
                                          hipMemcpyDeviceToHost));
                if(std::memcmp(hout.data(), hexp.data(), sizeof(hipDoubleComplex) * n * bb) != 0)
                {
                    ADD_FAILURE() << "factorization mismatch in blocks [" << first << ", "
                                  << first + n << ")";
                    ++bad_chunks;
                }
            }
            EXPECT_EQ(bad_chunks, size_t(0));
        }
    }
}
#endif
