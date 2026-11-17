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
// Regression tests for the grid.x extents of the csrmm merge path
// (AISPARSE-671) and nnz split (AISPARSE-672) algorithms.
//
// FOCUS. Both algorithms size grid.x from a 64-bit count and assigned it straight
// into dim3, which is unsigned int:
//
//   * merge path: block_count = (m + nnz - 1) / 256 + 1, a uint64_t. Because it
//     derives from m + nnz rather than from a single index it is genuinely wider
//     than 32 bits.
//   * nnz split: nblocks = (nnz - 1) / 256 + 1 from an int64_t nnz, plus the
//     dense-column extent n for the final block reduction.
//
// The kernels then read the block index back into a plain int and indexed blocks
// directly, so the value was truncated twice: once narrowing into dim3 and again
// when the kernel read it. The fix clamps every grid.x with
// rocsparse::get_grid_size_x, i.e. to min(maxGridSize[0], (2^32 - 1) / blockDim.x)
// blocks, grid-strides the kernels over the full count, and widens the block
// index to 64 bits.
//
// WHY NOT TEST THE REAL THRESHOLD. The limit that binds is the work-item count,
// gridDim.x * blockDim.x < 2^32, not maxGridSize[0]. The nnz split grids run one
// 256-thread block per 256 non-zeros, so they clamp once nnz exceeds
// 16,777,215 * 256, about 4.3e9 non-zeros. The 1024-thread block reduction runs
// one block per dense column and clamps at n >= 4,194,304, which is easy to
// allocate. The merge path kernels clamp from m + nnz of about 4.3e9 (256 lanes
// per merge block) to 1.4e11 (8 lanes). These tests shrink the limit with
// ScopedMaxGridSizeX for speed, not because the real threshold is unreachable:
// a small problem then runs through the same code path in milliseconds, and the
// grid-stride loops carry essentially all of the work. This is the AISPARSE-702
// idiom.
//
// WHAT MAKES EACH CASE LOAD-BEARING. The matrix is sized so that the unclamped
// grid is many blocks wide for every kernel under test:
//
//   * nnz = 8704 over m = 512 rows gives nblocks = 34 for the nnz split kernels.
//   * m + nnz = 9216 gives block_count = 36 for the merge path, and the merge
//     kernels that assign BLOCKSIZE / WF_SIZE merge blocks per block still need
//     ceil(36 / 16) = 3 blocks in the narrowest configuration (n <= 16).
//
// With the limit shrunk to clamped_grid_x = 2 every one of those grids is
// clamped, so a kernel that does not grid-stride drops most of its work and the
// result is visibly wrong. The nn and nt kernel families are selected by the B
// layout (column order reaches csrmmnn_*, row order reaches csrmmnt_*), and n is
// varied to reach both the "main" and the "remainder" kernels of each family.
//
// INDEX TYPES. Every grid case runs once per row pointer / column index pair
// that CSR spmm accepts (i32/i32, i64/i32, i64/i64), so each instantiation of
// the kernels and of the host dispatch is reached.
//
// EXACT ARITHMETIC. The reference is computed on the host in the same type. All
// matrix and dense values are small integers, so every product and partial sum
// is exactly representable in double and the result does not depend on the order
// in which the atomics land. That is what allows an exact comparison against a
// grid whose block count, and therefore whose accumulation order, differs.
//
// NO DEVICE-MEMORY GUARD and no size-based skip: the largest allocation in this
// file is under 3 MB (the batched n = 300 C matrix), so every case runs on every
// GPU.
//
// TARGET: rocsparse-unit-test-device. These tests drive the public
// rocsparse_spmm entry point, so nothing hidden in librocsparse is needed, but
// they do need a real device and the complete handle type.
//
#include "unit_test_utils.hpp"

// ScopedMaxGridSizeX: shrinks handle->properties.maxGridSize[0], the limit the
// csrmm merge path and nnz split launches now clamp grid.x against.
#include "unit_test_grid_clamp.hpp"

#include "rocsparse.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    // Sized so that every grid under test is several blocks wide before clamping;
    // see the file header for the resulting block counts.
    // Shared column count. Every matrix below has the same k so one dense B
    // generator serves them all.
    constexpr int32_t mat_k = 320;

    // Default shape. 512 rows of 17 entries gives nnz = 8704, so the nnz split
    // kernels want nblocks = 34 blocks and the merge path wants
    // block_count = (512 + 8704 - 1) / 256 + 1 = 36.
    constexpr int32_t default_m           = 512;
    constexpr int32_t default_nnz_per_row = 17;

    // Second shape, used only to reach the analysis launch. The grid of
    // csrmmnn_nnz_split_compute_row_limits is ceil(nblocks / 256), which is a single
    // block for the default shape and therefore never clamped. 1024 rows of 140
    // entries gives nnz = 143360 and nblocks = 560, so that grid is 3 blocks wide
    // and the clamp below actually bites.
    constexpr int32_t many_blocks_m           = 1024;
    constexpr int32_t many_blocks_nnz_per_row = 140;

    // Far below the 3 to 36 blocks the grids above ask for, so the grid-stride
    // loops carry almost all of the work.
    constexpr int clamped_grid_x = 2;

    // batch_count_C of the batched cases.
    constexpr int32_t batched_count = 2;

    // Row pointer and column index types of A. The host matrix is always built in
    // int32_t and widened on upload.
    struct IndexTypes
    {
        rocsparse_indextype row;
        rocsparse_indextype col;
    };

    std::string index_types_name(const ::testing::TestParamInfo<IndexTypes>& info)
    {
        const auto name = [](rocsparse_indextype t) {
            return (t == rocsparse_indextype_i64) ? std::string("i64") : std::string("i32");
        };
        return name(info.param.row) + "_" + name(info.param.col);
    }

    // A small banded pattern with exactly nnz_per_row entries per row, and small
    // integer values so that the host reference is bit-exact against any
    // accumulation order. A batched matrix stores batch_count strided copies, each
    // with its own column pattern and values, so a kernel that reads the wrong
    // batch of A produces a visibly wrong result. nnz is the per-batch count.
    struct CsrMatrix
    {
        int32_t              m           = 0;
        int32_t              k           = mat_k;
        int64_t              nnz         = 0;
        int32_t              batch_count = 1;
        std::vector<int32_t> row_ptr;
        std::vector<int32_t> col_ind;
        std::vector<double>  val;
    };

    // row_length(i, batch) is the number of entries in row i of that batch. Every
    // batch must hold the same total, which becomes a.nnz.
    template <typename F>
    CsrMatrix make_matrix_with_rows(int32_t m, int32_t batch_count, F row_length)
    {
        CsrMatrix a;
        a.m           = m;
        a.batch_count = batch_count;

        for(int32_t batch = 0; batch < batch_count; ++batch)
        {
            a.row_ptr.push_back(0);
            for(int32_t i = 0; i < m; ++i)
            {
                const int32_t len = row_length(i, batch);
                // Row start walks across the columns; the entries of a row are then
                // consecutive, which keeps the column indices strictly increasing as
                // csrmm requires, and keeps the largest index below mat_k.
                const int32_t base = (i * 7 + batch * 3) % (mat_k - len);
                for(int32_t j = 0; j < len; ++j)
                {
                    a.col_ind.push_back(base + j);
                    a.val.push_back(static_cast<double>(1 + ((i + j + batch) % 5)));
                }
                a.row_ptr.push_back(a.row_ptr.back() + len);
            }
        }
        a.nnz = a.row_ptr[m];
        return a;
    }

    CsrMatrix make_matrix(int32_t m, int32_t nnz_per_row, int32_t batch_count = 1)
    {
        return make_matrix_with_rows(
            m, batch_count, [nnz_per_row](int32_t, int32_t) { return nnz_per_row; });
    }

    // Row lengths cycle through 1 to 33. Odd batches take the cycle in reverse row
    // order, which keeps nnz per batch equal but gives each batch its own row_ptr,
    // so the merge path coordinates of one batch are wrong for the other.
    CsrMatrix make_varying_rows_matrix(int32_t m, int32_t batch_count)
    {
        return make_matrix_with_rows(m, batch_count, [m](int32_t i, int32_t batch) {
            const int32_t row = (batch % 2 == 0) ? i : m - 1 - i;
            return 1 + row % 33;
        });
    }

    // Dense B, small integers. ld is the leading dimension implied by order.
    std::vector<double> make_dense(int64_t count)
    {
        std::vector<double> b(count);
        for(int64_t i = 0; i < count; ++i)
        {
            b[i] = static_cast<double>(1 + (i % 7));
        }
        return b;
    }

    // C = alpha * A * B + beta * C on the host, in the same type and with the same
    // layouts the device call uses. order_B selects how the dense_B buffer is
    // interpreted; C is always column order here. B and C hold a.batch_count
    // tightly packed batches, of k * n and m * n elements.
    std::vector<double> host_csrmm(const CsrMatrix&           a,
                                   int32_t                    n,
                                   const std::vector<double>& b,
                                   rocsparse_order            order_B,
                                   double                     alpha,
                                   double                     beta,
                                   const std::vector<double>& c_in)
    {
        std::vector<double> c = c_in;
        for(int32_t batch = 0; batch < a.batch_count; ++batch)
        {
            const int32_t* row_ptr = a.row_ptr.data() + static_cast<int64_t>(a.m + 1) * batch;
            const int32_t* col_ind = a.col_ind.data() + a.nnz * batch;
            const double*  val     = a.val.data() + a.nnz * batch;
            const double*  b_batch = b.data() + static_cast<int64_t>(a.k) * n * batch;
            double*        c_batch = c.data() + static_cast<int64_t>(a.m) * n * batch;

            for(int32_t i = 0; i < a.m; ++i)
            {
                for(int32_t j = 0; j < n; ++j)
                {
                    double sum = 0.0;
                    for(int32_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p)
                    {
                        const int32_t col = col_ind[p];
                        const double  bv  = (order_B == rocsparse_order_column)
                                                ? b_batch[static_cast<int64_t>(col)
                                                        + static_cast<int64_t>(mat_k) * j]
                                                : b_batch[static_cast<int64_t>(col) * n + j];
                        sum += val[p] * bv;
                    }
                    const int64_t ci = static_cast<int64_t>(i) + static_cast<int64_t>(a.m) * j;
                    c_batch[ci]      = alpha * sum + beta * c_batch[ci];
                }
            }
        }
        return c;
    }

    // Index of the first element of `got` that differs from `want`, or -1. One
    // precise failure instead of hundreds of thousands of EXPECT_EQ macros.
    int64_t first_mismatch(const std::vector<double>& got, const std::vector<double>& want)
    {
        for(size_t i = 0; i < got.size(); ++i)
        {
            if(got[i] != want[i])
            {
                return static_cast<int64_t>(i);
            }
        }
        return -1;
    }

    // Owns the spmm descriptors so that each one created is destroyed on every
    // return path, including a failure part way through creating them.
    struct SpmmDescrs
    {
        rocsparse_spmat_descr mat_a = nullptr;
        rocsparse_dnmat_descr mat_b = nullptr;
        rocsparse_dnmat_descr mat_c = nullptr;

        SpmmDescrs() = default;

        SpmmDescrs(const SpmmDescrs&) = delete;

        SpmmDescrs& operator=(const SpmmDescrs&) = delete;

        ~SpmmDescrs()
        {
            if(mat_a != nullptr)
            {
                (void)rocsparse_destroy_spmat_descr(mat_a);
            }
            if(mat_b != nullptr)
            {
                (void)rocsparse_destroy_dnmat_descr(mat_b);
            }
            if(mat_c != nullptr)
            {
                (void)rocsparse_destroy_dnmat_descr(mat_c);
            }
        }
    };

    // Run rocsparse_spmm (buffer size / preprocess / compute) for the given
    // algorithm and B layout, and return C. When clamp_grid_x is true the grid.x
    // limit is shrunk for the whole sequence, so the analysis launch is clamped
    // too, not just the compute launches.
    //
    // Returns an empty vector and records a gtest failure on any API error.
    template <typename I, typename J>
    std::vector<double> device_csrmm_typed(rocsparse_handle           handle,
                                           const CsrMatrix&           a,
                                           int32_t                    n,
                                           rocsparse_spmm_alg         alg,
                                           rocsparse_order            order_B,
                                           bool                       clamp_grid_x,
                                           const std::vector<double>& b,
                                           const std::vector<double>& c_in)
    {
        const double alpha = 2.0;
        const double beta  = 3.0;

        constexpr rocsparse_indextype row_type
            = (sizeof(I) == 8) ? rocsparse_indextype_i64 : rocsparse_indextype_i32;
        constexpr rocsparse_indextype col_type
            = (sizeof(J) == 8) ? rocsparse_indextype_i64 : rocsparse_indextype_i32;

        device_vector<I>      d_row_ptr(std::vector<I>(a.row_ptr.begin(), a.row_ptr.end()));
        device_vector<J>      d_col_ind(std::vector<J>(a.col_ind.begin(), a.col_ind.end()));
        device_vector<double> d_val(a.val);
        device_vector<double> d_b(b);
        device_vector<double> d_c(c_in);

        if(d_row_ptr.ptr == nullptr || d_col_ind.ptr == nullptr || d_val.ptr == nullptr
           || d_b.ptr == nullptr || d_c.ptr == nullptr)
        {
            ADD_FAILURE() << "device allocation failed";
            return {};
        }

        SpmmDescrs             descrs;
        rocsparse_spmat_descr& mat_a = descrs.mat_a;
        rocsparse_dnmat_descr& mat_b = descrs.mat_b;
        rocsparse_dnmat_descr& mat_c = descrs.mat_c;

        const int64_t ldb = (order_B == rocsparse_order_column) ? a.k : n;

        if(rocsparse_create_csr_descr(&mat_a,
                                      a.m,
                                      a.k,
                                      a.nnz,
                                      d_row_ptr.ptr,
                                      d_col_ind.ptr,
                                      d_val.ptr,
                                      row_type,
                                      col_type,
                                      rocsparse_index_base_zero,
                                      rocsparse_datatype_f64_r)
               != rocsparse_status_success
           || rocsparse_create_dnmat_descr(
                  &mat_b, a.k, n, ldb, d_b.ptr, rocsparse_datatype_f64_r, order_B)
                  != rocsparse_status_success
           || rocsparse_create_dnmat_descr(
                  &mat_c, a.m, n, a.m, d_c.ptr, rocsparse_datatype_f64_r, rocsparse_order_column)
                  != rocsparse_status_success)
        {
            ADD_FAILURE() << "descriptor creation failed";
            return {};
        }

        if(a.batch_count > 1
           && (rocsparse_csr_set_strided_batch(mat_a, a.batch_count, a.m + 1, a.nnz)
                   != rocsparse_status_success
               || rocsparse_dnmat_set_strided_batch(
                      mat_b, a.batch_count, static_cast<int64_t>(a.k) * n)
                      != rocsparse_status_success
               || rocsparse_dnmat_set_strided_batch(
                      mat_c, a.batch_count, static_cast<int64_t>(a.m) * n)
                      != rocsparse_status_success))
        {
            ADD_FAILURE() << "strided batch setup failed";
            return {};
        }

        std::vector<double> out;
        {
            // Scoped so the limit is restored before the descriptors are destroyed
            // even if an assertion below returns early.
            std::unique_ptr<ScopedMaxGridSizeX> clamp;
            if(clamp_grid_x)
            {
                clamp.reset(new ScopedMaxGridSizeX(handle, clamped_grid_x));
            }

            size_t           buffer_size = 0;
            rocsparse_status status      = rocsparse_spmm(handle,
                                                     rocsparse_operation_none,
                                                     rocsparse_operation_none,
                                                     &alpha,
                                                     mat_a,
                                                     mat_b,
                                                     &beta,
                                                     mat_c,
                                                     rocsparse_datatype_f64_r,
                                                     alg,
                                                     rocsparse_spmm_stage_buffer_size,
                                                     &buffer_size,
                                                     nullptr);

            void* d_buffer = nullptr;
            if(status == rocsparse_status_success && buffer_size > 0)
            {
                if(hipMalloc(&d_buffer, buffer_size) != hipSuccess)
                {
                    ADD_FAILURE() << "temp buffer allocation of " << buffer_size << " bytes failed";
                    status = rocsparse_status_memory_error;
                }
            }

            if(status == rocsparse_status_success)
            {
                status = rocsparse_spmm(handle,
                                        rocsparse_operation_none,
                                        rocsparse_operation_none,
                                        &alpha,
                                        mat_a,
                                        mat_b,
                                        &beta,
                                        mat_c,
                                        rocsparse_datatype_f64_r,
                                        alg,
                                        rocsparse_spmm_stage_preprocess,
                                        &buffer_size,
                                        d_buffer);
            }

            if(status == rocsparse_status_success)
            {
                status = rocsparse_spmm(handle,
                                        rocsparse_operation_none,
                                        rocsparse_operation_none,
                                        &alpha,
                                        mat_a,
                                        mat_b,
                                        &beta,
                                        mat_c,
                                        rocsparse_datatype_f64_r,
                                        alg,
                                        rocsparse_spmm_stage_compute,
                                        &buffer_size,
                                        d_buffer);
            }

            if(status == rocsparse_status_success)
            {
                if(hipDeviceSynchronize() != hipSuccess)
                {
                    ADD_FAILURE() << "hipDeviceSynchronize failed";
                    status = rocsparse_status_internal_error;
                }
            }

            if(status == rocsparse_status_success)
            {
                out = to_host(d_c);
            }
            else
            {
                ADD_FAILURE() << "rocsparse_spmm returned status " << status;
            }

            if(d_buffer != nullptr)
            {
                (void)hipFree(d_buffer);
            }
        }

        return out;
    }

    std::vector<double> device_csrmm(rocsparse_handle           handle,
                                     IndexTypes                 idx,
                                     const CsrMatrix&           a,
                                     int32_t                    n,
                                     rocsparse_spmm_alg         alg,
                                     rocsparse_order            order_B,
                                     bool                       clamp_grid_x,
                                     const std::vector<double>& b,
                                     const std::vector<double>& c_in)
    {
        if(idx.row == rocsparse_indextype_i32 && idx.col == rocsparse_indextype_i32)
        {
            return device_csrmm_typed<int32_t, int32_t>(
                handle, a, n, alg, order_B, clamp_grid_x, b, c_in);
        }
        if(idx.row == rocsparse_indextype_i64 && idx.col == rocsparse_indextype_i32)
        {
            return device_csrmm_typed<int64_t, int32_t>(
                handle, a, n, alg, order_B, clamp_grid_x, b, c_in);
        }
        if(idx.row == rocsparse_indextype_i64 && idx.col == rocsparse_indextype_i64)
        {
            return device_csrmm_typed<int64_t, int64_t>(
                handle, a, n, alg, order_B, clamp_grid_x, b, c_in);
        }
        ADD_FAILURE() << "unsupported index types";
        return {};
    }

    // Shared body: compute C on the device with grid.x clamped to a couple of
    // blocks and compare against the host reference.
    void check_clamped_grid_matches_host(rocsparse_handle   handle,
                                         IndexTypes         idx,
                                         const CsrMatrix&   a,
                                         int32_t            n,
                                         rocsparse_spmm_alg alg,
                                         rocsparse_order    order_B)
    {
        const std::vector<double> b = make_dense(static_cast<int64_t>(a.k) * n
                                                 * a.batch_count); // same buffer either layout
        const std::vector<double> c_in(static_cast<size_t>(a.m) * n * a.batch_count, 1.0);

        const std::vector<double> want = host_csrmm(a, n, b, order_B, 2.0, 3.0, c_in);

        const std::vector<double> got
            = device_csrmm(handle, idx, a, n, alg, order_B, /*clamp_grid_x=*/true, b, c_in);
        ASSERT_EQ(got.size(), want.size());

        const int64_t bad      = first_mismatch(got, want);
        const int64_t in_batch = bad % (static_cast<int64_t>(a.m) * n);
        EXPECT_EQ(bad, -1) << "csrmm did not cover the full block count with grid.x clamped to "
                           << clamped_grid_x << " blocks: first wrong element at linear index "
                           << bad << " (batch " << (bad / (static_cast<int64_t>(a.m) * n))
                           << ", row " << (in_batch % a.m) << ", column " << (in_batch / a.m)
                           << "), got " << (bad >= 0 ? got[bad] : 0.0) << " want "
                           << (bad >= 0 ? want[bad] : 0.0) << ". n = " << n << ", m = " << a.m
                           << ", nnz = " << a.nnz << ", batch_count = " << a.batch_count;
    }

    // The default shape, built once per test.
    CsrMatrix default_matrix()
    {
        return make_matrix(default_m, default_nnz_per_row);
    }

    class CsrmmGrids : public HandleTest, public ::testing::WithParamInterface<IndexTypes>
    {
    };

    using CsrmmMergeRange = HandleTest;
}

INSTANTIATE_TEST_SUITE_P(
    index_types,
    CsrmmGrids,
    ::testing::Values(IndexTypes{rocsparse_indextype_i32, rocsparse_indextype_i32},
                      IndexTypes{rocsparse_indextype_i64, rocsparse_indextype_i32},
                      IndexTypes{rocsparse_indextype_i64, rocsparse_indextype_i64}),
    index_types_name);

// ---------------------------------------------------------------------------
// Merge path (AISPARSE-671)
//
// n selects the WF_SIZE branch of the dispatch, which in turn sets how many
// merge blocks each launched block covers, so the two shapes of the fix (one
// merge block per block, and BLOCKSIZE / WF_SIZE merge blocks per block) are
// both exercised.
// ---------------------------------------------------------------------------

// csrmmnn_merge_path_kernel, narrowest branch: n <= 16 means 16 merge blocks per
// launched block, so the grid is only ceil(36 / 16) = 3 blocks wide. This is the
// tightest case in the file and the one a missing stride loop is least likely to
// break, which is exactly why it is worth pinning.
TEST_P(CsrmmGrids, merge_path_nn_grid_stride_n16)
{
    check_clamped_grid_matches_host(handle,
                                    GetParam(),
                                    default_matrix(),
                                    16,
                                    rocsparse_spmm_alg_csr_merge_path,
                                    rocsparse_order_column);
}

// csrmmnn_merge_path_kernel, widest branch: n > 64 means one merge block per
// launched block, so the unclamped grid is the full 36 blocks.
TEST_P(CsrmmGrids, merge_path_nn_grid_stride_n80)
{
    check_clamped_grid_matches_host(handle,
                                    GetParam(),
                                    default_matrix(),
                                    80,
                                    rocsparse_spmm_alg_csr_merge_path,
                                    rocsparse_order_column);
}

// csrmmnt_merge_path_main_kernel: one merge block per launched block, grid.x is
// block_count itself. n = 256 makes main cover all columns with no remainder.
TEST_P(CsrmmGrids, merge_path_nt_main_grid_stride)
{
    check_clamped_grid_matches_host(handle,
                                    GetParam(),
                                    default_matrix(),
                                    256,
                                    rocsparse_spmm_alg_csr_merge_path,
                                    rocsparse_order_row);
}

// csrmmnt_merge_path_remainder_kernel: n = 300 leaves a remainder of 44 columns
// after the 256-column main pass, so both the main and the remainder launch run.
TEST_P(CsrmmGrids, merge_path_nt_remainder_grid_stride)
{
    check_clamped_grid_matches_host(handle,
                                    GetParam(),
                                    default_matrix(),
                                    300,
                                    rocsparse_spmm_alg_csr_merge_path,
                                    rocsparse_order_row);
}

// csrmmnt_merge_path_remainder_kernel only: n < 64 skips the main pass entirely.
TEST_P(CsrmmGrids, merge_path_nt_remainder_only_grid_stride)
{
    check_clamped_grid_matches_host(handle,
                                    GetParam(),
                                    default_matrix(),
                                    24,
                                    rocsparse_spmm_alg_csr_merge_path,
                                    rocsparse_order_row);
}

// ---------------------------------------------------------------------------
// nnz split (AISPARSE-672)
// ---------------------------------------------------------------------------

// csrmmnn_nnz_split_main_kernel plus csrmmnn_general_block_reduce. This is the
// case that also covers the block reduction buffer strides: the producing
// kernels used to take the row stride of row_block_red / val_block_red from
// hipGridDim_x, which only equals nblocks while the grid is unclamped, while the
// consumer always used nblocks. Clamping grid.x without fixing that would read
// the reduction buffers at the wrong offsets.
TEST_P(CsrmmGrids, nnz_split_nn_main_grid_stride)
{
    check_clamped_grid_matches_host(handle,
                                    GetParam(),
                                    default_matrix(),
                                    32,
                                    rocsparse_spmm_alg_csr_nnz_split,
                                    rocsparse_order_column);
}

// csrmmnn_nnz_split_remainder_kernel as well: n = 35 leaves 3 columns after the
// 8-wide main pass.
TEST_P(CsrmmGrids, nnz_split_nn_remainder_grid_stride)
{
    check_clamped_grid_matches_host(handle,
                                    GetParam(),
                                    default_matrix(),
                                    35,
                                    rocsparse_spmm_alg_csr_nnz_split,
                                    rocsparse_order_column);
}

// csrmmnt_nnz_split_main_kernel, grid.x sized directly from nnz.
TEST_P(CsrmmGrids, nnz_split_nt_main_grid_stride)
{
    check_clamped_grid_matches_host(handle,
                                    GetParam(),
                                    default_matrix(),
                                    64,
                                    rocsparse_spmm_alg_csr_nnz_split,
                                    rocsparse_order_row);
}

// csrmmnt_nnz_split_remainder_kernel as well.
TEST_P(CsrmmGrids, nnz_split_nt_remainder_grid_stride)
{
    check_clamped_grid_matches_host(handle,
                                    GetParam(),
                                    default_matrix(),
                                    70,
                                    rocsparse_spmm_alg_csr_nnz_split,
                                    rocsparse_order_row);
}

// ---------------------------------------------------------------------------
// Analysis launch (AISPARSE-672)
//
// csrmmnn_nnz_split_compute_row_limits is launched with one thread per nnz block,
// so its grid is ceil(nblocks / 256): a single block for the default shape, which
// means the clamp there can never bite and the stride loop never runs. The
// many-blocks shape has nblocks = 560, so that grid is 3 blocks wide and clamping
// it to 2 forces the stride loop. row_limits feeds the dichotomic search in every
// compute kernel, so a gap in it shows up as wrong rows rather than as a missing
// tail.
//
// This shape also drives the nn compute kernels and the block reduction over 560
// nnz blocks instead of 34, which is a much longer stride loop than the cases
// above.
// ---------------------------------------------------------------------------

TEST_P(CsrmmGrids, nnz_split_analysis_grid_stride)
{
    const CsrMatrix a = make_matrix(many_blocks_m, many_blocks_nnz_per_row);
    ASSERT_GT((a.nnz - 1) / 256 + 1, 2 * 256)
        << "the shape must need more than two blocks of row-limit threads for this "
           "test to reach the analysis stride loop";

    check_clamped_grid_matches_host(
        handle, GetParam(), a, 8, rocsparse_spmm_alg_csr_nnz_split, rocsparse_order_column);
}

TEST_P(CsrmmGrids, merge_path_many_blocks_grid_stride)
{
    const CsrMatrix a = make_matrix(many_blocks_m, many_blocks_nnz_per_row);

    check_clamped_grid_matches_host(
        handle, GetParam(), a, 8, rocsparse_spmm_alg_csr_merge_path, rocsparse_order_column);
}

// ---------------------------------------------------------------------------
// Batched, batch_count_C = 2
//
// The batch dimension runs on grid.y, which is not clamped here, so these cases
// check that the per-batch offsets survive the grid.x stride loops. A, B and C
// are all batched, and each batch of A has its own column pattern and values,
// so the merge path computes a separate coordinate set per batch
// (offsets_batch_stride_A != 0), and the nnz split reduction reads its own
// nblocks * n slice of val_block_red per batch. Reading the wrong batch
// anywhere gives a wrong result.
// ---------------------------------------------------------------------------

TEST_P(CsrmmGrids, batched_merge_path_nn_grid_stride_n16)
{
    check_clamped_grid_matches_host(handle,
                                    GetParam(),
                                    make_matrix(default_m, default_nnz_per_row, batched_count),
                                    16,
                                    rocsparse_spmm_alg_csr_merge_path,
                                    rocsparse_order_column);
}

TEST_P(CsrmmGrids, batched_merge_path_nn_grid_stride_n80)
{
    check_clamped_grid_matches_host(handle,
                                    GetParam(),
                                    make_matrix(default_m, default_nnz_per_row, batched_count),
                                    80,
                                    rocsparse_spmm_alg_csr_merge_path,
                                    rocsparse_order_column);
}

TEST_P(CsrmmGrids, batched_merge_path_nt_remainder_grid_stride)
{
    check_clamped_grid_matches_host(handle,
                                    GetParam(),
                                    make_matrix(default_m, default_nnz_per_row, batched_count),
                                    300,
                                    rocsparse_spmm_alg_csr_merge_path,
                                    rocsparse_order_row);
}

TEST_P(CsrmmGrids, batched_nnz_split_nn_remainder_grid_stride)
{
    check_clamped_grid_matches_host(handle,
                                    GetParam(),
                                    make_matrix(default_m, default_nnz_per_row, batched_count),
                                    35,
                                    rocsparse_spmm_alg_csr_nnz_split,
                                    rocsparse_order_column);
}

TEST_P(CsrmmGrids, batched_nnz_split_nt_remainder_grid_stride)
{
    check_clamped_grid_matches_host(handle,
                                    GetParam(),
                                    make_matrix(default_m, default_nnz_per_row, batched_count),
                                    70,
                                    rocsparse_spmm_alg_csr_nnz_split,
                                    rocsparse_order_row);
}

// ---------------------------------------------------------------------------
// Batched, per-batch row lengths
//
// make_matrix gives every batch the same row_ptr, so the per-batch merge
// coordinate sets above are identical and a kernel that read batch 0's set for
// every batch would still pass. Here the batches share nnz but not row_ptr, so
// each merge path kernel must read the coordinate set of its own batch.
// ---------------------------------------------------------------------------

TEST_P(CsrmmGrids, batched_varying_rows_merge_path_nn_grid_stride_n16)
{
    check_clamped_grid_matches_host(handle,
                                    GetParam(),
                                    make_varying_rows_matrix(default_m, batched_count),
                                    16,
                                    rocsparse_spmm_alg_csr_merge_path,
                                    rocsparse_order_column);
}

TEST_P(CsrmmGrids, batched_varying_rows_merge_path_nn_grid_stride_n80)
{
    check_clamped_grid_matches_host(handle,
                                    GetParam(),
                                    make_varying_rows_matrix(default_m, batched_count),
                                    80,
                                    rocsparse_spmm_alg_csr_merge_path,
                                    rocsparse_order_column);
}

// Main and remainder kernels.
TEST_P(CsrmmGrids, batched_varying_rows_merge_path_nt_remainder_grid_stride)
{
    check_clamped_grid_matches_host(handle,
                                    GetParam(),
                                    make_varying_rows_matrix(default_m, batched_count),
                                    300,
                                    rocsparse_spmm_alg_csr_merge_path,
                                    rocsparse_order_row);
}

// Remainder kernel only.
TEST_P(CsrmmGrids, batched_varying_rows_merge_path_nt_remainder_only_grid_stride)
{
    check_clamped_grid_matches_host(handle,
                                    GetParam(),
                                    make_varying_rows_matrix(default_m, batched_count),
                                    24,
                                    rocsparse_spmm_alg_csr_merge_path,
                                    rocsparse_order_row);
}

// ---------------------------------------------------------------------------
// Control: the same problems with the real, unclamped grid.x limit. If one of
// these fails then the fix broke the ordinary path and the clamped cases above
// prove nothing.
// ---------------------------------------------------------------------------

TEST_P(CsrmmGrids, unclamped_grid_matches_host)
{
    const CsrMatrix a = default_matrix();

    struct Case
    {
        int32_t            n;
        rocsparse_spmm_alg alg;
        rocsparse_order    order_B;
    };

    // Exactly the cases the clamped tests above use, so a failure here separates a
    // pre-existing defect at this shape from one introduced by the clamp.
    const Case cases[] = {
        {16, rocsparse_spmm_alg_csr_merge_path, rocsparse_order_column},
        {80, rocsparse_spmm_alg_csr_merge_path, rocsparse_order_column},
        {256, rocsparse_spmm_alg_csr_merge_path, rocsparse_order_row},
        {300, rocsparse_spmm_alg_csr_merge_path, rocsparse_order_row},
        {24, rocsparse_spmm_alg_csr_merge_path, rocsparse_order_row},
        {32, rocsparse_spmm_alg_csr_nnz_split, rocsparse_order_column},
        {35, rocsparse_spmm_alg_csr_nnz_split, rocsparse_order_column},
        {64, rocsparse_spmm_alg_csr_nnz_split, rocsparse_order_row},
        {70, rocsparse_spmm_alg_csr_nnz_split, rocsparse_order_row},
    };

    const CsrMatrix many = make_matrix(many_blocks_m, many_blocks_nnz_per_row);

    for(const Case& c : cases)
    {
        const std::vector<double> b = make_dense(static_cast<int64_t>(a.k) * c.n);
        const std::vector<double> c_in(static_cast<size_t>(a.m) * c.n, 1.0);

        const std::vector<double> want = host_csrmm(a, c.n, b, c.order_B, 2.0, 3.0, c_in);
        const std::vector<double> got  = device_csrmm(
            handle, GetParam(), a, c.n, c.alg, c.order_B, /*clamp_grid_x=*/false, b, c_in);
        ASSERT_EQ(got.size(), want.size());

        EXPECT_EQ(first_mismatch(got, want), -1)
            << "csrmm is wrong with an unclamped grid: n = " << c.n << ", alg = " << c.alg
            << ", order_B = " << c.order_B;
    }

    // The many-blocks shape too, for both algorithms.
    for(rocsparse_spmm_alg alg :
        {rocsparse_spmm_alg_csr_nnz_split, rocsparse_spmm_alg_csr_merge_path})
    {
        const int32_t             n = 8;
        const std::vector<double> b = make_dense(static_cast<int64_t>(many.k) * n);
        const std::vector<double> c_in(static_cast<size_t>(many.m) * n, 1.0);

        const std::vector<double> want
            = host_csrmm(many, n, b, rocsparse_order_column, 2.0, 3.0, c_in);
        const std::vector<double> got = device_csrmm(handle,
                                                     GetParam(),
                                                     many,
                                                     n,
                                                     alg,
                                                     rocsparse_order_column,
                                                     /*clamp_grid_x=*/false,
                                                     b,
                                                     c_in);
        ASSERT_EQ(got.size(), want.size());

        EXPECT_EQ(first_mismatch(got, want), -1)
            << "csrmm is wrong with an unclamped grid on the many-blocks shape: alg = " << alg;
    }

    // And the batched cases. The merge path also runs with per-batch row lengths;
    // the nnz split shares one row_limits array across batches, so it is only
    // given batches that share row_ptr.
    const Case batched_cases[] = {
        {16, rocsparse_spmm_alg_csr_merge_path, rocsparse_order_column},
        {80, rocsparse_spmm_alg_csr_merge_path, rocsparse_order_column},
        {300, rocsparse_spmm_alg_csr_merge_path, rocsparse_order_row},
        {24, rocsparse_spmm_alg_csr_merge_path, rocsparse_order_row},
        {35, rocsparse_spmm_alg_csr_nnz_split, rocsparse_order_column},
        {70, rocsparse_spmm_alg_csr_nnz_split, rocsparse_order_row},
    };

    const CsrMatrix shared_rows  = make_matrix(default_m, default_nnz_per_row, batched_count);
    const CsrMatrix varying_rows = make_varying_rows_matrix(default_m, batched_count);

    for(const Case& c : batched_cases)
    {
        for(const CsrMatrix* matrix : {&shared_rows, &varying_rows})
        {
            if(matrix == &varying_rows && c.alg != rocsparse_spmm_alg_csr_merge_path)
            {
                continue;
            }

            const CsrMatrix&          batched = *matrix;
            const std::vector<double> b
                = make_dense(static_cast<int64_t>(batched.k) * c.n * batched.batch_count);
            const std::vector<double> c_in(
                static_cast<size_t>(batched.m) * c.n * batched.batch_count, 1.0);

            const std::vector<double> want = host_csrmm(batched, c.n, b, c.order_B, 2.0, 3.0, c_in);
            const std::vector<double> got  = device_csrmm(handle,
                                                         GetParam(),
                                                         batched,
                                                         c.n,
                                                         c.alg,
                                                         c.order_B,
                                                         /*clamp_grid_x=*/false,
                                                         b,
                                                         c_in);
            ASSERT_EQ(got.size(), want.size());

            EXPECT_EQ(first_mismatch(got, want), -1)
                << "batched csrmm is wrong with an unclamped grid: n = " << c.n
                << ", alg = " << c.alg << ", order_B = " << c.order_B << ", nnz = " << batched.nnz;
        }
    }
}

// ---------------------------------------------------------------------------
// Merge path coordinate range (AISPARSE-671)
//
// The merge coordinates are coordinate_t<uint32_t>, so every merge path stage
// rejects m or nnz above UINT32_MAX with rocsparse_status_not_implemented and
// accepts them at UINT32_MAX. Only the sizes are checked before the rejection,
// so A, B and C point at one small dummy allocation that is never read.
// Accepted sizes run only the buffer size stage, which does not touch memory
// either.
// ---------------------------------------------------------------------------

namespace
{
    rocsparse_status merge_path_stage_status(rocsparse_handle     handle,
                                             int64_t              m,
                                             int64_t              nnz,
                                             rocsparse_spmm_stage stage)
    {
        // Wide enough that nnz fits m * k, which descriptor creation checks.
        const int64_t     k = std::max<int64_t>(8, (nnz - 1) / m + 1);
        constexpr int64_t n = 1;

        device_vector<int64_t> dummy(std::vector<int64_t>(64, 0));
        if(dummy.ptr == nullptr)
        {
            ADD_FAILURE() << "device allocation failed";
            return rocsparse_status_memory_error;
        }

        SpmmDescrs descrs;
        if(rocsparse_create_csr_descr(&descrs.mat_a,
                                      m,
                                      k,
                                      nnz,
                                      dummy.ptr,
                                      dummy.ptr,
                                      dummy.ptr,
                                      rocsparse_indextype_i64,
                                      rocsparse_indextype_i64,
                                      rocsparse_index_base_zero,
                                      rocsparse_datatype_f64_r)
               != rocsparse_status_success
           || rocsparse_create_dnmat_descr(&descrs.mat_b,
                                           k,
                                           n,
                                           k,
                                           dummy.ptr,
                                           rocsparse_datatype_f64_r,
                                           rocsparse_order_column)
                  != rocsparse_status_success
           || rocsparse_create_dnmat_descr(&descrs.mat_c,
                                           m,
                                           n,
                                           m,
                                           dummy.ptr,
                                           rocsparse_datatype_f64_r,
                                           rocsparse_order_column)
                  != rocsparse_status_success)
        {
            ADD_FAILURE() << "descriptor creation failed";
            return rocsparse_status_internal_error;
        }

        const double alpha       = 2.0;
        const double beta        = 3.0;
        size_t       buffer_size = sizeof(int64_t) * 64;
        return rocsparse_spmm(handle,
                              rocsparse_operation_none,
                              rocsparse_operation_none,
                              &alpha,
                              descrs.mat_a,
                              descrs.mat_b,
                              &beta,
                              descrs.mat_c,
                              rocsparse_datatype_f64_r,
                              rocsparse_spmm_alg_csr_merge_path,
                              stage,
                              &buffer_size,
                              (stage == rocsparse_spmm_stage_buffer_size) ? nullptr : dummy.ptr);
    }
}

TEST_F(CsrmmMergeRange, rejects_m_or_nnz_above_uint32_max)
{
    constexpr int64_t over = int64_t(UINT32_MAX) + 1;

    for(rocsparse_spmm_stage stage : {rocsparse_spmm_stage_buffer_size,
                                      rocsparse_spmm_stage_preprocess,
                                      rocsparse_spmm_stage_compute})
    {
        EXPECT_EQ(merge_path_stage_status(handle, over, 8, stage), rocsparse_status_not_implemented)
            << "m = 2^32, stage " << stage;
        EXPECT_EQ(merge_path_stage_status(handle, 8, over, stage), rocsparse_status_not_implemented)
            << "nnz = 2^32, stage " << stage;
    }
}

TEST_F(CsrmmMergeRange, accepts_m_and_nnz_at_uint32_max)
{
    constexpr int64_t at = int64_t(UINT32_MAX);

    EXPECT_EQ(merge_path_stage_status(handle, at, 8, rocsparse_spmm_stage_buffer_size),
              rocsparse_status_success);
    EXPECT_EQ(merge_path_stage_status(handle, 8, at, rocsparse_spmm_stage_buffer_size),
              rocsparse_status_success);
}
