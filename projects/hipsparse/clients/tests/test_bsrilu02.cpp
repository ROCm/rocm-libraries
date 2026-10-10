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

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

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
// nnzb * block_dim^2 is just past INT_MAX, so 32-bit value offsets inside rocSPARSE would
// corrupt or skip the last blocks. Registered as known_bug until that is fixed.
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
    const size_t required = sizeof(hipDoubleComplex) * nval + (size_t(1) << 32);
    if(free_mem < required)
    {
        GTEST_SKIP() << "Requires " << required << " bytes of free device memory";
    }

    std::unique_ptr<handle_struct>   unique_ptr_handle(new handle_struct);
    hipsparseHandle_t                handle = unique_ptr_handle->handle;
    std::unique_ptr<descr_struct>    unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t              descr = unique_ptr_descr->descr;
    std::unique_ptr<bsrilu02_struct> unique_ptr_info(new bsrilu02_struct);
    bsrilu02Info_t                   info = unique_ptr_info->info;

    // Block diagonal matrix
    std::vector<int> hrow_ptr(mb + 1), hcol_ind(nnzb);
    for(int i = 0; i < mb; ++i)
    {
        hrow_ptr[i] = hcol_ind[i] = i;
    }
    hrow_ptr[mb] = nnzb;

    // Each block is L * U (row-major) with L = I + 0.5 * subdiagonal, U = 2 * I + superdiagonal.
    // The in-place LU has no fill-in: 0.5 below, 2 on, and 1 above the diagonal (all exact).
    std::vector<hipDoubleComplex> hblock(bb), hexpect(bb);
    for(int r = 0; r < block_dim; ++r)
    {
        for(int c = 0; c < block_dim; ++c)
        {
            double a = 0, lu = 0;
            if(r == c)
            {
                a  = (r == 0) ? 2.0 : 2.5;
                lu = 2.0;
            }
            else if(c == r + 1)
            {
                a = lu = 1.0;
            }
            else if(r == c + 1)
            {
                a  = 1.0;
                lu = 0.5;
            }
            hblock[r * block_dim + c]  = make_hipDoubleComplex(a, 0.0);
            hexpect[r * block_dim + c] = make_hipDoubleComplex(lu, 0.0);
        }
    }

    auto drow_ptr_m = hipsparse_unique_ptr{device_malloc(sizeof(int) * (mb + 1)), device_free};
    auto dcol_ind_m = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnzb), device_free};
    auto dval_m = hipsparse_unique_ptr{device_malloc(sizeof(hipDoubleComplex) * nval), device_free};
    int*              drow_ptr = (int*)drow_ptr_m.get();
    int*              dcol_ind = (int*)dcol_ind_m.get();
    hipDoubleComplex* dval     = (hipDoubleComplex*)dval_m.get();

    CHECK_HIP_ERROR(
        hipMemcpy(drow_ptr, hrow_ptr.data(), sizeof(int) * (mb + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcol_ind, hcol_ind.data(), sizeof(int) * nnzb, hipMemcpyHostToDevice));
    for(int b = 0; b < nnzb; ++b)
    {
        CHECK_HIP_ERROR(hipMemcpy(dval + size_t(b) * bb,
                                  hblock.data(),
                                  sizeof(hipDoubleComplex) * bb,
                                  hipMemcpyHostToDevice));
    }

    // hipSPARSE ignores the policy and always passes rocsparse_solve_policy_auto.
    const hipsparseDirection_t   dir    = HIPSPARSE_DIRECTION_ROW;
    const hipsparseSolvePolicy_t policy = HIPSPARSE_SOLVE_POLICY_NO_LEVEL;

    int buffer_size;
    ASSERT_EQ(hipsparseZbsrilu02_bufferSize(
                  handle, dir, mb, nnzb, descr, dval, drow_ptr, dcol_ind, block_dim, info,
                  &buffer_size),
              HIPSPARSE_STATUS_SUCCESS);
    auto  dbuffer_m = hipsparse_unique_ptr{device_malloc(buffer_size), device_free};
    void* dbuffer   = dbuffer_m.get();

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    ASSERT_EQ(hipsparseZbsrilu02_analysis(handle, dir, mb, nnzb, descr, dval, drow_ptr, dcol_ind,
                                          block_dim, info, policy, dbuffer),
              HIPSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(hipsparseZbsrilu02(handle, dir, mb, nnzb, descr, dval, drow_ptr, dcol_ind,
                                 block_dim, info, policy, dbuffer),
              HIPSPARSE_STATUS_SUCCESS);

    int position;
    EXPECT_EQ(hipsparseXbsrilu02_zeroPivot(handle, info, &position), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(position, -1);

    // First block, a middle block, and the last two (the last starts at value offset 2^31)
    for(int b : {0, nnzb / 2, nnzb - 2, nnzb - 1})
    {
        SCOPED_TRACE(testing::Message() << "block " << b);
        std::vector<hipDoubleComplex> hres(bb);
        CHECK_HIP_ERROR(hipMemcpy(hres.data(),
                                  dval + size_t(b) * bb,
                                  sizeof(hipDoubleComplex) * bb,
                                  hipMemcpyDeviceToHost));
        size_t mismatches = 0;
        for(size_t k = 0; k < bb; ++k)
        {
            mismatches += (std::abs(hipCreal(hres[k]) - hipCreal(hexpect[k])) > 1e-12
                           || std::abs(hipCimag(hres[k]) - hipCimag(hexpect[k])) > 1e-12);
        }
        EXPECT_EQ(mismatches, size_t(0));
    }
}
#endif
