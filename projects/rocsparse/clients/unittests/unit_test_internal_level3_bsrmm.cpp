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
// Host-path unit tests for the block-sparse mm C API:
//   rocsparse_Xbsrmm and rocsparse_Xgebsrmm.
//
// These entry points (library/src/level3/rocsparse_bsrmm.cpp / rocsparse_gebsrmm*)
// were uncovered. A tiny 2x2 block-identity A (1x1 blocks) times a dense 2x2 B is
// run through the public API (so host dispatch/validation/orchestration is
// counted), plus bad-argument branches.
//
#include "unit_test_utils.hpp"

#include "rocsparse.h"

using namespace rocsparse_ut;

namespace
{
    struct MatDescr
    {
        rocsparse_mat_descr d = nullptr;
        MatDescr()
        {
            (void)rocsparse_create_mat_descr(&d);
        }
        ~MatDescr()
        {
            if(d)
                (void)rocsparse_destroy_mat_descr(d);
        }
    };
}

// ===========================================================================
// bsrmm : C = alpha * A_bsr * B + beta * C
// ===========================================================================
class BsrMM : public HandleTest
{
};

// bsrmm C = alpha*A*B + beta*C with A = 2x2 block identity (1x1 blocks),
// B = [[1,2],[3,4]] (column-major {1,3,2,4}), alpha=1, beta=0. Since A = I the
// result is exactly B, so C (column-major) must equal {1,3,2,4}.
TEST_F(BsrMM, identity_times_dense)
{
    const rocsparse_direction dir       = rocsparse_direction_row;
    const rocsparse_int       mb        = 2;
    const rocsparse_int       kb        = 2;
    const rocsparse_int       n         = 2;
    const rocsparse_int       nnzb      = 2;
    const rocsparse_int       block_dim = 1;

    device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f}};
    // B column-major 2x2 = [[1,2],[3,4]] -> {1,3,2,4}
    device_vector<float> B{std::vector<float>{1.0f, 3.0f, 2.0f, 4.0f}};
    device_vector<float> C{std::vector<float>(4, 0.0f)};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr && B.ptr && C.ptr);

    MatDescr descr;
    ASSERT_TRUE(descr.d);

    const float alpha = 1.0f, beta = 0.0f;
    ASSERT_EQ(rocsparse_sbsrmm(handle,
                               dir,
                               rocsparse_operation_none,
                               rocsparse_operation_none,
                               mb,
                               n,
                               kb,
                               nnzb,
                               &alpha,
                               descr.d,
                               val,
                               row_ptr,
                               col_ind,
                               block_dim,
                               B,
                               2,
                               &beta,
                               C,
                               2),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // A = I so C = B (column-major {1,3,2,4}).
    auto hc = to_host(C.ptr, 4);
    EXPECT_FLOAT_EQ(hc[0], 1.0f);
    EXPECT_FLOAT_EQ(hc[1], 3.0f);
    EXPECT_FLOAT_EQ(hc[2], 2.0f);
    EXPECT_FLOAT_EQ(hc[3], 4.0f);
}

TEST_F(BsrMM, bad_args)
{
    device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f}};
    device_vector<float>   B{std::vector<float>{1.0f, 3.0f, 2.0f, 4.0f}};
    device_vector<float>   C{std::vector<float>(4, 0.0f)};
    MatDescr               descr;
    const float            alpha = 1.0f, beta = 0.0f;

    EXPECT_EQ(rocsparse_sbsrmm(nullptr,
                               rocsparse_direction_row,
                               rocsparse_operation_none,
                               rocsparse_operation_none,
                               2,
                               2,
                               2,
                               2,
                               &alpha,
                               descr.d,
                               val,
                               row_ptr,
                               col_ind,
                               1,
                               B,
                               2,
                               &beta,
                               C,
                               2),
              rocsparse_status_invalid_handle);

    // Invalid direction enum.
    EXPECT_EQ(rocsparse_sbsrmm(handle,
                               static_cast<rocsparse_direction>(-1),
                               rocsparse_operation_none,
                               rocsparse_operation_none,
                               2,
                               2,
                               2,
                               2,
                               &alpha,
                               descr.d,
                               val,
                               row_ptr,
                               col_ind,
                               1,
                               B,
                               2,
                               &beta,
                               C,
                               2),
              rocsparse_status_invalid_value);
}

// ===========================================================================
// gebsrmm : general block-sparse mm (row_block_dim == col_block_dim == 1 here)
// ===========================================================================
class GebsrMM : public HandleTest
{
};

// gebsrmm C = alpha*A*B + beta*C, same tiny identity problem as BsrMM but via
// the general block API (row_block_dim == col_block_dim == 1). C == B.
TEST_F(GebsrMM, identity_times_dense)
{
    const rocsparse_direction dir = rocsparse_direction_row;

    device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f}};
    device_vector<float>   B{std::vector<float>{1.0f, 3.0f, 2.0f, 4.0f}};
    device_vector<float>   C{std::vector<float>(4, 0.0f)};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr && B.ptr && C.ptr);

    MatDescr descr;
    ASSERT_TRUE(descr.d);

    const float alpha = 1.0f, beta = 0.0f;
    ASSERT_EQ(rocsparse_sgebsrmm(handle,
                                 dir,
                                 rocsparse_operation_none,
                                 rocsparse_operation_none,
                                 2,
                                 2,
                                 2,
                                 2,
                                 &alpha,
                                 descr.d,
                                 val,
                                 row_ptr,
                                 col_ind,
                                 1,
                                 1,
                                 B,
                                 2,
                                 &beta,
                                 C,
                                 2),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // A = I so C = B (column-major {1,3,2,4}).
    auto hc = to_host(C.ptr, 4);
    EXPECT_FLOAT_EQ(hc[0], 1.0f);
    EXPECT_FLOAT_EQ(hc[1], 3.0f);
    EXPECT_FLOAT_EQ(hc[2], 2.0f);
    EXPECT_FLOAT_EQ(hc[3], 4.0f);
}

TEST_F(GebsrMM, bad_args)
{
    device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f}};
    device_vector<float>   B{std::vector<float>{1.0f, 3.0f, 2.0f, 4.0f}};
    device_vector<float>   C{std::vector<float>(4, 0.0f)};
    MatDescr               descr;
    const float            alpha = 1.0f, beta = 0.0f;

    EXPECT_EQ(rocsparse_sgebsrmm(nullptr,
                                 rocsparse_direction_row,
                                 rocsparse_operation_none,
                                 rocsparse_operation_none,
                                 2,
                                 2,
                                 2,
                                 2,
                                 &alpha,
                                 descr.d,
                                 val,
                                 row_ptr,
                                 col_ind,
                                 1,
                                 1,
                                 B,
                                 2,
                                 &beta,
                                 C,
                                 2),
              rocsparse_status_invalid_handle);
}
