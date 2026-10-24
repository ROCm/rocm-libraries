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
// Default CSR SpMV selects nnz-split only for a poor adaptive partition, and
// only when the caller asked for the default algorithm. These tests run the
// public analysis and read the algorithm stored on the csrmv info.
//
#include "unit_test_utils.hpp"

#include "../../library/src/level2/rocsparse_csrmv.hpp"
#include "rocsparse_mat_info.hpp"
#include "rocsparse_spmat_descr.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    struct HostCsr
    {
        int64_t              m = 0;
        int64_t              n = 0;
        std::vector<int32_t> row_ptr;
        std::vector<int32_t> col_ind;
        std::vector<float>   val;

        int64_t nnz() const
        {
            return static_cast<int64_t>(col_ind.size());
        }
    };

    void add_row(HostCsr& a, int32_t len)
    {
        for(int32_t k = 0; k < len; ++k)
        {
            a.col_ind.push_back(k);
            a.val.push_back(1.0f);
        }
        a.row_ptr.push_back(static_cast<int32_t>(a.col_ind.size()));
        ++a.m;
    }

    // One nonzero at each end and a long empty gap between them. Adaptive keeps
    // that gap in one row block.
    HostCsr scattered_gap()
    {
        HostCsr a;
        a.n = 1;
        a.row_ptr.push_back(0);
        add_row(a, 1);
        for(int i = 0; i < 20000; ++i)
        {
            add_row(a, 0);
        }
        add_row(a, 1);
        return a;
    }

    // Every row is short and populated, so one adaptive block holds the matrix.
    HostCsr populated()
    {
        HostCsr a;
        a.n = 8;
        a.row_ptr.push_back(0);
        for(int i = 0; i < 64; ++i)
        {
            add_row(a, 8);
        }
        return a;
    }

    // A just-long row, then two short rows, repeated. Adaptive cuts a block on
    // every long-to-short transition.
    HostCsr alternating()
    {
        HostCsr a;
        a.n = 160;
        a.row_ptr.push_back(0);
        for(int g = 0; g < 8; ++g)
        {
            add_row(a, 160);
            add_row(a, 3);
            add_row(a, 3);
        }
        return a;
    }

    // Long rows, then an empty suffix adaptive already skips.
    HostCsr long_then_empty()
    {
        HostCsr a;
        a.n = 200;
        a.row_ptr.push_back(0);
        for(int i = 0; i < 4; ++i)
        {
            add_row(a, 200);
        }
        for(int i = 0; i < 20000; ++i)
        {
            add_row(a, 0);
        }
        return a;
    }

    std::vector<float> host_spmv(const HostCsr& a)
    {
        std::vector<float> y(static_cast<size_t>(a.m), 0.0f);
        for(int64_t row = 0; row < a.m; ++row)
        {
            float sum = 0.0f;
            for(int32_t k = a.row_ptr[static_cast<size_t>(row)];
                k < a.row_ptr[static_cast<size_t>(row) + 1];
                ++k)
            {
                sum += a.val[static_cast<size_t>(k)];
            }
            y[static_cast<size_t>(row)] = sum;
        }
        return y;
    }

    struct Selection
    {
        bool               has_info        = false;
        int32_t            resolved        = -1;
        bool               adaptive_blocks = false;
        bool               nnzsplit_tuned  = false;
        std::vector<float> y;
    };

    Selection run_spmv(rocsparse_handle    handle,
                       const HostCsr&      a,
                       rocsparse_spmv_alg  alg,
                       rocsparse_operation op,
                       bool                as_csc,
                       bool                check_y)
    {
        device_vector<int32_t> d_ptr{a.row_ptr};
        device_vector<int32_t> d_ind{a.col_ind};
        device_vector<float>   d_val{a.val};
        const int64_t          rows = as_csc ? a.n : a.m;
        const int64_t          cols = as_csc ? a.m : a.n;
        const int64_t          xlen = (op == rocsparse_operation_none) ? cols : rows;
        const int64_t          ylen = (op == rocsparse_operation_none) ? rows : cols;
        device_vector<float>   d_x{std::vector<float>(static_cast<size_t>(xlen), 1.0f)};
        device_vector<float>   d_y{std::vector<float>(static_cast<size_t>(ylen), 0.0f)};
        if(!d_ptr.ptr || !d_ind.ptr || !d_val.ptr || !d_x.ptr || !d_y.ptr)
        {
            ADD_FAILURE() << "device allocation failed";
            return {};
        }

        auto require = [](rocsparse_status status) {
            EXPECT_EQ(status, rocsparse_status_success);
            return status == rocsparse_status_success;
        };

        rocsparse_spmat_descr mat = nullptr;
        if(as_csc)
        {
            // The host "row" pointers are the CSC column pointers, so the CSR
            // view of A^T has the same empty-line pattern.
            if(!require(rocsparse_create_csc_descr(&mat,
                                                   a.n,
                                                   a.m,
                                                   a.nnz(),
                                                   d_ptr.ptr,
                                                   d_ind.ptr,
                                                   d_val.ptr,
                                                   rocsparse_indextype_i32,
                                                   rocsparse_indextype_i32,
                                                   rocsparse_index_base_zero,
                                                   rocsparse_datatype_f32_r)))
            {
                return {};
            }
        }
        else
        {
            if(!require(rocsparse_create_csr_descr(&mat,
                                                   a.m,
                                                   a.n,
                                                   a.nnz(),
                                                   d_ptr.ptr,
                                                   d_ind.ptr,
                                                   d_val.ptr,
                                                   rocsparse_indextype_i32,
                                                   rocsparse_indextype_i32,
                                                   rocsparse_index_base_zero,
                                                   rocsparse_datatype_f32_r)))
            {
                return {};
            }
        }

        rocsparse_dnvec_descr dx = nullptr;
        rocsparse_dnvec_descr dy = nullptr;
        if(!require(rocsparse_create_dnvec_descr(&dx, xlen, d_x.ptr, rocsparse_datatype_f32_r))
           || !require(rocsparse_create_dnvec_descr(&dy, ylen, d_y.ptr, rocsparse_datatype_f32_r)))
        {
            EXPECT_EQ(rocsparse_destroy_dnvec_descr(dx), rocsparse_status_success);
            EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
            return {};
        }

        const float alpha       = 1.0f;
        const float beta        = 0.0f;
        size_t      buffer_size = 0;
        if(!require(rocsparse_spmv(handle,
                                   op,
                                   &alpha,
                                   mat,
                                   dx,
                                   &beta,
                                   dy,
                                   rocsparse_datatype_f32_r,
                                   alg,
                                   rocsparse_spmv_stage_preprocess,
                                   &buffer_size,
                                   nullptr)))
        {
            EXPECT_EQ(rocsparse_destroy_dnvec_descr(dx), rocsparse_status_success);
            EXPECT_EQ(rocsparse_destroy_dnvec_descr(dy), rocsparse_status_success);
            EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
            return {};
        }

        Selection            out;
        rocsparse_csrmv_info info = mat->info->get_csrmv_info();
        if(info != nullptr)
        {
            out.has_info        = true;
            out.resolved        = info->resolved_alg;
            out.adaptive_blocks = info->adaptive.row_blocks != nullptr;
            out.nnzsplit_tuned  = info->nnzsplit.block_size != 0;
        }

        if(check_y
           && require(rocsparse_spmv(handle,
                                     op,
                                     &alpha,
                                     mat,
                                     dx,
                                     &beta,
                                     dy,
                                     rocsparse_datatype_f32_r,
                                     alg,
                                     rocsparse_spmv_stage_compute,
                                     &buffer_size,
                                     nullptr)))
        {
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
            out.y = to_host(d_y.ptr, static_cast<size_t>(ylen));
        }

        EXPECT_EQ(rocsparse_destroy_dnvec_descr(dx), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_dnvec_descr(dy), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
        return out;
    }

    void expect_y(const std::vector<float>& y, const std::vector<float>& ref)
    {
        ASSERT_EQ(y.size(), ref.size());
        for(size_t i = 0; i < y.size(); ++i)
        {
            EXPECT_FLOAT_EQ(y[i], ref[i]) << "y[" << i << "]";
        }
    }
}

class CsrmvDefaultDispatch : public HandleTest
{
};

TEST_F(CsrmvDefaultDispatch, default_selects_nnzsplit_for_internal_empty_gap)
{
    const HostCsr a = scattered_gap();
    const auto    got
        = run_spmv(handle, a, rocsparse_spmv_alg_default, rocsparse_operation_none, false, true);
    EXPECT_TRUE(got.has_info);
    EXPECT_EQ(got.resolved, static_cast<int32_t>(rocsparse::csrmv_alg_nnzsplit));
    EXPECT_TRUE(got.nnzsplit_tuned);
    EXPECT_FALSE(got.adaptive_blocks);
    expect_y(got.y, host_spmv(a));
}

TEST_F(CsrmvDefaultDispatch, explicit_adaptive_is_not_replaced)
{
    const HostCsr a   = scattered_gap();
    const auto    got = run_spmv(
        handle, a, rocsparse_spmv_alg_csr_adaptive, rocsparse_operation_none, false, true);
    EXPECT_TRUE(got.has_info);
    EXPECT_EQ(got.resolved, static_cast<int32_t>(rocsparse::csrmv_alg_default));
    EXPECT_TRUE(got.adaptive_blocks);
    EXPECT_FALSE(got.nnzsplit_tuned);
    expect_y(got.y, host_spmv(a));
}

TEST_F(CsrmvDefaultDispatch, default_keeps_adaptive_for_populated_rows)
{
    const HostCsr a = populated();
    const auto    got
        = run_spmv(handle, a, rocsparse_spmv_alg_default, rocsparse_operation_none, false, true);
    EXPECT_TRUE(got.has_info);
    EXPECT_EQ(got.resolved, static_cast<int32_t>(rocsparse::csrmv_alg_adaptive));
    EXPECT_TRUE(got.adaptive_blocks);
    EXPECT_FALSE(got.nnzsplit_tuned);
    expect_y(got.y, host_spmv(a));
}

TEST_F(CsrmvDefaultDispatch, default_selects_nnzsplit_for_alternating_row_lengths)
{
    const HostCsr a = alternating();
    const auto    got
        = run_spmv(handle, a, rocsparse_spmv_alg_default, rocsparse_operation_none, false, true);
    EXPECT_TRUE(got.has_info);
    EXPECT_EQ(got.resolved, static_cast<int32_t>(rocsparse::csrmv_alg_nnzsplit));
    EXPECT_TRUE(got.nnzsplit_tuned);
    EXPECT_FALSE(got.adaptive_blocks);
    expect_y(got.y, host_spmv(a));
}

TEST_F(CsrmvDefaultDispatch, default_keeps_adaptive_when_empty_suffix_is_skipped)
{
    const HostCsr a = long_then_empty();
    const auto    got
        = run_spmv(handle, a, rocsparse_spmv_alg_default, rocsparse_operation_none, false, true);
    EXPECT_TRUE(got.has_info);
    EXPECT_EQ(got.resolved, static_cast<int32_t>(rocsparse::csrmv_alg_adaptive));
    EXPECT_TRUE(got.adaptive_blocks);
    EXPECT_FALSE(got.nnzsplit_tuned);
    expect_y(got.y, host_spmv(a));
}

TEST_F(CsrmvDefaultDispatch, transpose_default_stays_on_rowsplit)
{
    const HostCsr a   = scattered_gap();
    const auto    got = run_spmv(
        handle, a, rocsparse_spmv_alg_default, rocsparse_operation_transpose, false, false);
    EXPECT_FALSE(got.has_info);
}

TEST_F(CsrmvDefaultDispatch, transposed_csc_default_uses_the_csr_rule)
{
    const HostCsr a   = scattered_gap();
    const auto    got = run_spmv(
        handle, a, rocsparse_spmv_alg_default, rocsparse_operation_transpose, true, false);
    EXPECT_TRUE(got.has_info);
    EXPECT_EQ(got.resolved, static_cast<int32_t>(rocsparse::csrmv_alg_nnzsplit));
    EXPECT_TRUE(got.nnzsplit_tuned);
}
