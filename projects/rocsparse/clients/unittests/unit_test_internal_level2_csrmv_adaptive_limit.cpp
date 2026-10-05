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
// Unit tests for the csrmv adaptive dispatch-limit check.
//
// The adaptive kernels need one workgroup per row block in a single dispatch,
// so csrmv analysis and csrmv reject a matrix whose row-block count exceeds
// get_grid_size_x instead of clamping the grid. The tests shrink
// handle->properties.maxGridSize[0] to one below the row-block count of a small
// matrix, check that analysis and compute return rocsparse_status_not_implemented
// and leave y untouched, then check that at exactly the row-block count both
// succeed and match a host reference.
//
#include "unit_test_utils.hpp"

#include "rocsparse.h"
#include "rocsparse_csrmv_info.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_mat_info.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    constexpr rocsparse_int M = 16384;

    // Lower bidiagonal: row 0 holds (0, 0); row i > 0 holds (i, i - 1) and (i, i).
    struct Bidiagonal
    {
        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<float>         val;

        Bidiagonal()
        {
            row_ptr.push_back(0);
            for(rocsparse_int i = 0; i < M; ++i)
            {
                if(i > 0)
                {
                    col_ind.push_back(i - 1);
                    val.push_back(1.0f);
                }
                col_ind.push_back(i);
                val.push_back(2.0f);
                row_ptr.push_back(static_cast<rocsparse_int>(col_ind.size()));
            }
        }

        rocsparse_int nnz() const
        {
            return static_cast<rocsparse_int>(val.size());
        }
    };

    // y = alpha * op(A) * x + beta * y, where a symmetric A also applies the
    // transpose of every stored off-diagonal entry.
    std::vector<float> host_reference(const Bidiagonal&         A,
                                      bool                      symmetric,
                                      float                     alpha,
                                      const std::vector<float>& x,
                                      float                     beta,
                                      const std::vector<float>& y0)
    {
        std::vector<float> ax(M, 0.0f);
        for(rocsparse_int i = 0; i < M; ++i)
        {
            for(rocsparse_int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
            {
                const rocsparse_int j = A.col_ind[k];
                ax[i] += A.val[k] * x[j];
                if(symmetric && j != i)
                {
                    ax[j] += A.val[k] * x[i];
                }
            }
        }

        std::vector<float> y(M);
        for(rocsparse_int i = 0; i < M; ++i)
        {
            y[i] = alpha * ax[i] + beta * y0[i];
        }
        return y;
    }

    // Restores maxGridSize[0] even when an assertion returns early.
    struct GridLimitGuard
    {
        rocsparse_handle handle;
        int              saved;

        explicit GridLimitGuard(rocsparse_handle h)
            : handle(h)
            , saved(h->properties.maxGridSize[0])
        {
        }
        ~GridLimitGuard()
        {
            handle->properties.maxGridSize[0] = saved;
        }
        void set(int64_t limit)
        {
            handle->properties.maxGridSize[0] = static_cast<int>(limit);
        }
        void restore()
        {
            handle->properties.maxGridSize[0] = saved;
        }
    };

    class CsrmvAdaptiveLimit : public HandleTest
    {
    protected:
        void run(rocsparse_matrix_type type)
        {
            const bool       symmetric = (type == rocsparse_matrix_type_symmetric);
            const Bidiagonal A;

            device_vector<rocsparse_int> d_row_ptr{A.row_ptr};
            device_vector<rocsparse_int> d_col_ind{A.col_ind};
            device_vector<float>         d_val{A.val};
            ASSERT_TRUE(d_row_ptr.ptr && d_col_ind.ptr && d_val.ptr);

            std::vector<float> hx(M), hy0(M);
            for(rocsparse_int i = 0; i < M; ++i)
            {
                hx[i]  = static_cast<float>(i % 7 + 1);
                hy0[i] = static_cast<float>(i % 5 - 2);
            }
            device_vector<float> d_x{hx};
            device_vector<float> d_y{hy0};
            ASSERT_TRUE(d_x.ptr && d_y.ptr);

            rocsparse_mat_descr descr = nullptr;
            rocsparse_mat_info  info  = nullptr;
            ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
            ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);
            ASSERT_EQ(rocsparse_set_mat_type(descr, type), rocsparse_status_success);
            if(symmetric)
            {
                ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower),
                          rocsparse_status_success);
            }

            const float alpha = 2.0f;
            const float beta  = 0.5f;

            auto analysis = [&]() {
                return rocsparse_scsrmv_analysis(handle,
                                                 rocsparse_operation_none,
                                                 M,
                                                 M,
                                                 A.nnz(),
                                                 descr,
                                                 d_val,
                                                 d_row_ptr,
                                                 d_col_ind,
                                                 info);
            };
            auto compute = [&]() {
                return rocsparse_scsrmv(handle,
                                        rocsparse_operation_none,
                                        M,
                                        M,
                                        A.nnz(),
                                        &alpha,
                                        descr,
                                        d_val,
                                        d_row_ptr,
                                        d_col_ind,
                                        info,
                                        d_x,
                                        &beta,
                                        d_y);
            };

            GridLimitGuard guard(handle);

            // Learn the row-block count from an unrestricted analysis.
            ASSERT_EQ(analysis(), rocsparse_status_success);
            ASSERT_NE(info->get_csrmv_info(), nullptr);
            const int64_t nblocks = static_cast<int64_t>(info->get_csrmv_info()->adaptive.size) - 1;
            ASSERT_GE(nblocks, 2);

            // One below the row-block count: analysis fails and stores no info.
            guard.set(nblocks - 1);
            EXPECT_EQ(analysis(), rocsparse_status_not_implemented);
            EXPECT_EQ(info->get_csrmv_info(), nullptr);

            // Analyse within the limit, then shrink it: compute fails, y untouched.
            guard.restore();
            ASSERT_EQ(analysis(), rocsparse_status_success);
            guard.set(nblocks - 1);
            EXPECT_EQ(compute(), rocsparse_status_not_implemented);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            EXPECT_EQ(to_host(d_y), hy0);

            // Exactly the row-block count: compute succeeds and matches the host.
            guard.set(nblocks);
            ASSERT_EQ(compute(), rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            EXPECT_EQ(to_host(d_y), host_reference(A, symmetric, alpha, hx, beta, hy0));

            EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
            EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
        }
    };
}

TEST_F(CsrmvAdaptiveLimit, general)
{
    run(rocsparse_matrix_type_general);
}

TEST_F(CsrmvAdaptiveLimit, symmetric)
{
    run(rocsparse_matrix_type_symmetric);
}
