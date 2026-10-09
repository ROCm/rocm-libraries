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
// Forced-clamp tests for the bsrmm grid.x / grid.y extents and for the grid.x
// extent of the coomm segmented block reduction (AISPARSE-665/704).
//
// bsrmm sizes grid.x from the block-row count and grid.y from the number of
// dense column panels. Both are now clamped, grid.x with
// rocsparse::get_grid_size_x (min(maxGridSize[0], (2^32 - 1) / blockDim.x)) and
// grid.y with rocsparse::get_grid_size_y (maxGridSize[1], 65535), and the
// kernels grid-stride over both. The block reduction ran one block per dense
// column with grid.x = n; it now clamps grid.x and grid-strides over columns.
//
// At real sizes the clamps need more than 65535 column panels or millions of
// block rows. These tests shrink handle->properties.maxGridSize[0] and [1] for
// the compute stage instead, so a small problem runs through the clamped,
// looping path, and compare C against a host reference.
//
// DISPATCH. rocsparse_spmm with a BSR matrix reaches bsrmm, which picks the
// kernel from block_dim (trans_A = none):
//
//   block_dim == 2      bsrmmnn_small_blockdim_kernel (nn only; the nt small
//                       kernel is not clamped by this change and not tested)
//   3 <= block_dim <= 32 bsrmm_large_blockdim_kernel_ext, tuned as
//                       4x16 (<= 4), 8x8 (<= 8), 16x16 (<= 16), 32x32 (<= 32)
//   block_dim > 32      bsrmm_general_blockdim_kernel
//
// "nn" is B in column order, "nt" is B in row order; large_ext and general
// serve both. Every shape below is sized so that the unclamped grid.x is well
// over 7 blocks and grid.y is at least 4 panels, so each of the limits used
// forces the corresponding stride loop. Some block rows are empty, so the
// grid-stride must still write beta * C for them.
//
// EXACT ARITHMETIC. All values are small integers, so every product and
// partial sum is exact in float and double, independent of accumulation order.
// The comparison still uses a relative tolerance of a few ulps.
//
// TARGET: rocsparse-unit-test-device. The tests drive the public
// rocsparse_spmm entry point and need the complete handle type to reach
// handle->properties.
//
#include "unit_test_utils.hpp"

#include "rocsparse_handle.hpp"

#include "rocsparse.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    // Shrinks the grid.x / grid.y limits that get_grid_size_x / get_grid_size_y
    // clamp against (0 leaves an axis alone), and restores them on scope exit
    // so a failed assertion cannot leak into the next test.
    struct ScopedMaxGridSizeXY
    {
        rocsparse_handle handle;
        int              saved_x;
        int              saved_y;

        ScopedMaxGridSizeXY(rocsparse_handle h, int limit_x, int limit_y)
            : handle(h)
            , saved_x(h->properties.maxGridSize[0])
            , saved_y(h->properties.maxGridSize[1])
        {
            if(limit_x > 0)
            {
                handle->properties.maxGridSize[0] = limit_x;
            }
            if(limit_y > 0)
            {
                handle->properties.maxGridSize[1] = limit_y;
            }
        }

        ~ScopedMaxGridSizeXY()
        {
            handle->properties.maxGridSize[0] = saved_x;
            handle->properties.maxGridSize[1] = saved_y;
        }

        ScopedMaxGridSizeXY(const ScopedMaxGridSizeXY&) = delete;

        ScopedMaxGridSizeXY& operator=(const ScopedMaxGridSizeXY&) = delete;
    };

    struct GridLimits
    {
        int x; // 0 = unclamped
        int y; // 0 = unclamped
    };

    // x only, y only, and both.
    constexpr GridLimits bsrmm_limits[]
        = {{1, 0}, {3, 0}, {7, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 1}, {2, 3}, {7, 2}};

    constexpr GridLimits reduce_limits[] = {{1, 0}, {2, 0}, {3, 0}, {7, 0}};

    constexpr GridLimits unclamped[] = {{0, 0}};

    constexpr double alpha_value = 2.0;
    constexpr double beta_value  = -3.0;

    // A sparse matrix on the host, in the storage of its format, plus its
    // entries as (row, col, val) triplets for the reference.
    struct HostSparse
    {
        rocsparse_format    format    = rocsparse_format_csr;
        int64_t             m         = 0;
        int64_t             k         = 0;
        int64_t             nnz       = 0; // stored entries (nnzb for BSR)
        int64_t             mb        = 0;
        int64_t             kb        = 0;
        int64_t             block_dim = 1;
        rocsparse_direction dir       = rocsparse_direction_row;

        std::vector<int64_t> row; // row_ptr (CSR/BSR) or row_ind (COO)
        std::vector<int64_t> col;
        std::vector<double>  val;

        std::vector<int64_t> t_row;
        std::vector<int64_t> t_col;
        std::vector<double>  t_val;
    };

    // Block row i holds 0 to 3 blocks (every fifth block row is empty).
    HostSparse make_bsr(int64_t mb, int64_t kb, int64_t block_dim, rocsparse_direction dir)
    {
        HostSparse a;
        a.format    = rocsparse_format_bsr;
        a.mb        = mb;
        a.kb        = kb;
        a.block_dim = block_dim;
        a.m         = mb * block_dim;
        a.k         = kb * block_dim;
        a.dir       = dir;
        a.row.push_back(0);

        const int64_t bd2 = block_dim * block_dim;
        for(int64_t i = 0; i < mb; ++i)
        {
            const int64_t nblk = (i % 5 == 4) ? 0 : 1 + i % 3;

            std::vector<int64_t> cols;
            for(int64_t t = 0; t < nblk; ++t)
            {
                cols.push_back((i * 5 + t * 7) % kb);
            }
            std::sort(cols.begin(), cols.end());
            cols.erase(std::unique(cols.begin(), cols.end()), cols.end());

            for(const int64_t bc : cols)
            {
                const int64_t blk = static_cast<int64_t>(a.col.size());
                a.col.push_back(bc);
                a.val.resize((blk + 1) * bd2);
                for(int64_t r = 0; r < block_dim; ++r)
                {
                    for(int64_t c = 0; c < block_dim; ++c)
                    {
                        const double  v   = static_cast<double>((i + bc + 3 * r + c) % 5) - 2.0;
                        const int64_t off = (dir == rocsparse_direction_row) ? r * block_dim + c
                                                                             : c * block_dim + r;
                        a.val[blk * bd2 + off] = v;
                        a.t_row.push_back(i * block_dim + r);
                        a.t_col.push_back(bc * block_dim + c);
                        a.t_val.push_back(v);
                    }
                }
            }
            a.row.push_back(static_cast<int64_t>(a.col.size()));
        }
        a.nnz = static_cast<int64_t>(a.col.size());
        return a;
    }

    // nnz_per_row consecutive entries per row. The same triplets serve as COO
    // (sorted by row) or CSR.
    HostSparse make_coo_or_csr(rocsparse_format format, int64_t m, int64_t k, int64_t nnz_per_row)
    {
        HostSparse a;
        a.format = format;
        a.m      = m;
        a.k      = k;
        if(format == rocsparse_format_csr)
        {
            a.row.push_back(0);
        }
        for(int64_t i = 0; i < m; ++i)
        {
            const int64_t base = (i * 7) % (k - nnz_per_row + 1);
            for(int64_t j = 0; j < nnz_per_row; ++j)
            {
                const double v = static_cast<double>((i + j) % 5) - 2.0;
                if(format == rocsparse_format_coo)
                {
                    a.row.push_back(i);
                }
                a.col.push_back(base + j);
                a.val.push_back(v);
                a.t_row.push_back(i);
                a.t_col.push_back(base + j);
                a.t_val.push_back(v);
            }
            if(format == rocsparse_format_csr)
            {
                a.row.push_back(static_cast<int64_t>(a.col.size()));
            }
        }
        a.nnz = static_cast<int64_t>(a.col.size());
        return a;
    }

    std::vector<double> make_dense(int64_t count, int64_t salt)
    {
        std::vector<double> v(count);
        for(int64_t i = 0; i < count; ++i)
        {
            v[i] = static_cast<double>((i + salt) % 7) - 3.0;
        }
        return v;
    }

    // C = alpha * A * B + beta * C per batch. B is k x n in order_B; C is m x n
    // in column order. A is shared by every batch.
    std::vector<double> host_spmm(const HostSparse&          a,
                                  int64_t                    n,
                                  int64_t                    batch_count,
                                  const std::vector<double>& b,
                                  rocsparse_order            order_B,
                                  const std::vector<double>& c_in)
    {
        std::vector<double> c = c_in;
        for(int64_t batch = 0; batch < batch_count; ++batch)
        {
            const double* bb = b.data() + a.k * n * batch;
            double*       cb = c.data() + a.m * n * batch;

            std::vector<double> acc(a.m * n, 0.0);
            for(size_t e = 0; e < a.t_val.size(); ++e)
            {
                const int64_t r = a.t_row[e];
                const int64_t q = a.t_col[e];
                for(int64_t j = 0; j < n; ++j)
                {
                    const double bv
                        = (order_B == rocsparse_order_column) ? bb[q + a.k * j] : bb[q * n + j];
                    acc[r + a.m * j] += a.t_val[e] * bv;
                }
            }
            for(int64_t i = 0; i < a.m * n; ++i)
            {
                cb[i] = alpha_value * acc[i] + beta_value * cb[i];
            }
        }
        return c;
    }

    template <typename U>
    std::vector<U> convert(const std::vector<int64_t>& v)
    {
        return std::vector<U>(v.begin(), v.end());
    }

    template <typename T>
    std::vector<T> convert_values(const std::vector<double>& v)
    {
        std::vector<T> out(v.size());
        std::transform(v.begin(), v.end(), out.begin(), [](double d) { return static_cast<T>(d); });
        return out;
    }

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

    // C = alpha * A * B + beta * C through rocsparse_spmm (buffer size,
    // preprocess, compute). The grid limits are shrunk for the compute stage,
    // which is where the launches under test live.
    //
    // Returns an empty vector and records a gtest failure on any API error.
    template <typename T, typename I, typename J>
    std::vector<double> device_spmm(rocsparse_handle           handle,
                                    const HostSparse&          a,
                                    int64_t                    n,
                                    int64_t                    batch_count,
                                    rocsparse_spmm_alg         alg,
                                    rocsparse_order            order_B,
                                    const std::vector<double>& b,
                                    const std::vector<double>& c_in,
                                    GridLimits                 limits)
    {
        // COO has a single index type, I.
        device_vector<I> d_row(convert<I>(a.row));
        device_vector<J> d_col_j(a.format == rocsparse_format_coo ? std::vector<J>{0}
                                                                  : convert<J>(a.col));
        device_vector<I> d_col_i(a.format == rocsparse_format_coo ? convert<I>(a.col)
                                                                  : std::vector<I>{0});
        device_vector<T> d_val(convert_values<T>(a.val));
        device_vector<T> d_b(convert_values<T>(b));
        device_vector<T> d_c(convert_values<T>(c_in));

        if(d_row.ptr == nullptr || d_col_j.ptr == nullptr || d_col_i.ptr == nullptr
           || d_val.ptr == nullptr || d_b.ptr == nullptr || d_c.ptr == nullptr)
        {
            ADD_FAILURE() << "device allocation failed";
            return {};
        }

        SpmmDescrs       descrs;
        rocsparse_status status = rocsparse_status_success;
        switch(a.format)
        {
        case rocsparse_format_bsr:
            status = rocsparse_create_bsr_descr(&descrs.mat_a,
                                                a.mb,
                                                a.kb,
                                                a.nnz,
                                                a.dir,
                                                a.block_dim,
                                                d_row.ptr,
                                                d_col_j.ptr,
                                                d_val.ptr,
                                                it_of<I>(),
                                                it_of<J>(),
                                                rocsparse_index_base_zero,
                                                dt_of<T>());
            break;
        case rocsparse_format_coo:
            status = rocsparse_create_coo_descr(&descrs.mat_a,
                                                a.m,
                                                a.k,
                                                a.nnz,
                                                d_row.ptr,
                                                d_col_i.ptr,
                                                d_val.ptr,
                                                it_of<I>(),
                                                rocsparse_index_base_zero,
                                                dt_of<T>());
            break;
        default:
            status = rocsparse_create_csr_descr(&descrs.mat_a,
                                                a.m,
                                                a.k,
                                                a.nnz,
                                                d_row.ptr,
                                                d_col_j.ptr,
                                                d_val.ptr,
                                                it_of<I>(),
                                                it_of<J>(),
                                                rocsparse_index_base_zero,
                                                dt_of<T>());
            break;
        }

        const int64_t ldb = (order_B == rocsparse_order_column) ? a.k : n;
        if(status != rocsparse_status_success
           || rocsparse_create_dnmat_descr(&descrs.mat_b, a.k, n, ldb, d_b.ptr, dt_of<T>(), order_B)
                  != rocsparse_status_success
           || rocsparse_create_dnmat_descr(
                  &descrs.mat_c, a.m, n, a.m, d_c.ptr, dt_of<T>(), rocsparse_order_column)
                  != rocsparse_status_success)
        {
            ADD_FAILURE() << "descriptor creation failed";
            return {};
        }

        if(batch_count > 1
           && (rocsparse_dnmat_set_strided_batch(descrs.mat_b, batch_count, a.k * n)
                   != rocsparse_status_success
               || rocsparse_dnmat_set_strided_batch(descrs.mat_c, batch_count, a.m * n)
                      != rocsparse_status_success))
        {
            ADD_FAILURE() << "strided batch setup failed";
            return {};
        }

        const T alpha = static_cast<T>(alpha_value);
        const T beta  = static_cast<T>(beta_value);

        auto spmm = [&](rocsparse_spmm_stage stage, size_t* buffer_size, void* buffer) {
            return rocsparse_spmm(handle,
                                  rocsparse_operation_none,
                                  rocsparse_operation_none,
                                  &alpha,
                                  descrs.mat_a,
                                  descrs.mat_b,
                                  &beta,
                                  descrs.mat_c,
                                  dt_of<T>(),
                                  alg,
                                  stage,
                                  buffer_size,
                                  buffer);
        };

        size_t buffer_size = 0;
        status             = spmm(rocsparse_spmm_stage_buffer_size, &buffer_size, nullptr);
        if(status != rocsparse_status_success)
        {
            ADD_FAILURE() << "rocsparse_spmm buffer_size returned status " << status;
            return {};
        }

        device_vector<char> d_buffer(buffer_size > 0 ? buffer_size : size_t(1));
        if(d_buffer.ptr == nullptr)
        {
            ADD_FAILURE() << "temp buffer allocation of " << buffer_size << " bytes failed";
            return {};
        }

        status = spmm(rocsparse_spmm_stage_preprocess, &buffer_size, d_buffer.ptr);
        if(status != rocsparse_status_success)
        {
            ADD_FAILURE() << "rocsparse_spmm preprocess returned status " << status;
            return {};
        }

        {
            ScopedMaxGridSizeXY clamp(handle, limits.x, limits.y);
            status = spmm(rocsparse_spmm_stage_compute, &buffer_size, d_buffer.ptr);
        }
        if(status != rocsparse_status_success)
        {
            ADD_FAILURE() << "rocsparse_spmm compute returned status " << status;
            return {};
        }

        if(hipDeviceSynchronize() != hipSuccess)
        {
            ADD_FAILURE() << "hipDeviceSynchronize failed";
            return {};
        }

        const std::vector<T> c = to_host(d_c);
        return std::vector<double>(c.begin(), c.end());
    }

    template <typename T>
    bool near(double got, double want)
    {
        const double tol = 4.0 * std::numeric_limits<T>::epsilon() * std::max(1.0, std::abs(want));
        return std::abs(got - want) <= tol;
    }

    std::string describe_limits(GridLimits limits)
    {
        std::ostringstream os;
        os << "maxGridSize[0] = ";
        if(limits.x > 0)
        {
            os << limits.x;
        }
        else
        {
            os << "unclamped";
        }
        os << ", maxGridSize[1] = ";
        if(limits.y > 0)
        {
            os << limits.y;
        }
        else
        {
            os << "unclamped";
        }
        return os.str();
    }

    template <typename T, typename I, typename J, size_t N>
    void check_spmm(rocsparse_handle   handle,
                    const HostSparse&  a,
                    int64_t            n,
                    int64_t            batch_count,
                    rocsparse_spmm_alg alg,
                    rocsparse_order    order_B,
                    const GridLimits (&limits)[N],
                    const char* what)
    {
        const std::vector<double> b    = make_dense(a.k * n * batch_count, 1);
        const std::vector<double> c_in = make_dense(a.m * n * batch_count, 4);
        const std::vector<double> want = host_spmm(a, n, batch_count, b, order_B, c_in);

        for(const GridLimits& l : limits)
        {
            const std::vector<double> got
                = device_spmm<T, I, J>(handle, a, n, batch_count, alg, order_B, b, c_in, l);
            ASSERT_EQ(got.size(), want.size());

            int64_t bad   = 0;
            int64_t first = -1;
            for(size_t i = 0; i < got.size(); ++i)
            {
                if(!near<T>(got[i], want[i]))
                {
                    if(bad++ == 0)
                    {
                        first = static_cast<int64_t>(i);
                    }
                }
            }

            const int64_t in_batch = first % (a.m * n);
            EXPECT_EQ(bad, 0) << what << ", " << describe_limits(l)
                              << ", block_dim = " << a.block_dim << ", n = " << n << ", order_B = "
                              << (order_B == rocsparse_order_column ? "column" : "row")
                              << ", batch_count = " << batch_count << ": " << bad << " of "
                              << got.size() << " entries wrong; first at row " << (in_batch % a.m)
                              << ", column " << (in_batch / a.m) << ", batch "
                              << (first / (a.m * n)) << ", got " << (first >= 0 ? got[first] : 0.0)
                              << " want " << (first >= 0 ? want[first] : 0.0);
        }
    }

    // bsrmm, sized so every path's grid is > 7 blocks in x and >= 4 panels in y:
    //   small       mb = 100: grid.x = ceil(200 / 8) = 25, n = 40: 5 panels of 8
    //   large_ext   mb = 24 : grid.x = 24, n = 200: 7, 13, 7 and 4 panels for
    //               the 4x16, 8x8, 16x16 and 32x32 configs (32, 16, 32 and 64
    //               columns per panel)
    //   general     mb = 12 : grid.x = 12, n = 100: 4 panels of 32
    template <typename T, typename I, typename J, size_t N>
    void check_bsrmm(rocsparse_handle    handle,
                     int64_t             block_dim,
                     rocsparse_direction dir,
                     rocsparse_order     order_B,
                     const GridLimits (&limits)[N],
                     const char* what)
    {
        int64_t mb = 24;
        int64_t n  = 200;
        if(block_dim == 2)
        {
            mb = 100;
            n  = 40;
        }
        else if(block_dim > 32)
        {
            mb = 12;
            n  = 100;
        }
        const HostSparse a = make_bsr(mb, mb, block_dim, dir);
        check_spmm<T, I, J>(handle, a, n, 1, rocsparse_spmm_alg_bsr, order_B, limits, what);
    }

    // coomm segmented: 2000 rows of 5 entries is nnz = 10000, i.e. 10 segments of
    // 1024 for the block reduction to combine. n = 20 runs the WF_SIZE = 8 main
    // kernel on 16 columns and the remainder kernel on 4, and the block
    // reduction wants one block per column, 20 blocks.
    template <typename T, typename I, size_t N>
    void check_coomm(rocsparse_handle handle,
                     rocsparse_order  order_B,
                     int64_t          batch_count,
                     const GridLimits (&limits)[N])
    {
        const HostSparse a = make_coo_or_csr(rocsparse_format_coo, 2000, 300, 5);
        check_spmm<T, I, I>(handle,
                            a,
                            20,
                            batch_count,
                            rocsparse_spmm_alg_coo_segmented,
                            order_B,
                            limits,
                            "coommnn_general_block_reduce");
    }

    using BsrmmGrids       = HandleTest;
    using BlockReduceGrids = HandleTest;
}

// ---------------------------------------------------------------------------
// bsrmm small (block_dim = 2, nn)
// ---------------------------------------------------------------------------

TEST_F(BsrmmGrids, small_nn_clamped_f32_i32)
{
    check_bsrmm<float, int32_t, int32_t>(handle,
                                         2,
                                         rocsparse_direction_row,
                                         rocsparse_order_column,
                                         bsrmm_limits,
                                         "bsrmmnn_small_blockdim_kernel");
}

TEST_F(BsrmmGrids, small_nn_clamped_f64_i64)
{
    check_bsrmm<double, int64_t, int64_t>(handle,
                                          2,
                                          rocsparse_direction_column,
                                          rocsparse_order_column,
                                          bsrmm_limits,
                                          "bsrmmnn_small_blockdim_kernel");
}

// ---------------------------------------------------------------------------
// bsrmm large_ext (3 <= block_dim <= 32), all four tuned configs, nn and nt
// ---------------------------------------------------------------------------

TEST_F(BsrmmGrids, large_ext_4x16_clamped)
{
    for(const rocsparse_order order_B : {rocsparse_order_column, rocsparse_order_row})
    {
        check_bsrmm<float, int32_t, int32_t>(handle,
                                             3,
                                             rocsparse_direction_row,
                                             order_B,
                                             bsrmm_limits,
                                             "bsrmm_large_blockdim_kernel_ext<4, 16, 2>");
        check_bsrmm<double, int64_t, int64_t>(handle,
                                              4,
                                              rocsparse_direction_column,
                                              order_B,
                                              bsrmm_limits,
                                              "bsrmm_large_blockdim_kernel_ext<4, 16, 2>");
    }
}

TEST_F(BsrmmGrids, large_ext_8x8_clamped)
{
    for(const rocsparse_order order_B : {rocsparse_order_column, rocsparse_order_row})
    {
        check_bsrmm<float, int32_t, int32_t>(handle,
                                             7,
                                             rocsparse_direction_column,
                                             order_B,
                                             bsrmm_limits,
                                             "bsrmm_large_blockdim_kernel_ext<8, 8, 2>");
        check_bsrmm<double, int64_t, int64_t>(handle,
                                              8,
                                              rocsparse_direction_row,
                                              order_B,
                                              bsrmm_limits,
                                              "bsrmm_large_blockdim_kernel_ext<8, 8, 2>");
    }
}

TEST_F(BsrmmGrids, large_ext_16x16_clamped)
{
    for(const rocsparse_order order_B : {rocsparse_order_column, rocsparse_order_row})
    {
        check_bsrmm<float, int32_t, int32_t>(handle,
                                             13,
                                             rocsparse_direction_row,
                                             order_B,
                                             bsrmm_limits,
                                             "bsrmm_large_blockdim_kernel_ext<16, 16, 2>");
        check_bsrmm<double, int64_t, int64_t>(handle,
                                              16,
                                              rocsparse_direction_column,
                                              order_B,
                                              bsrmm_limits,
                                              "bsrmm_large_blockdim_kernel_ext<16, 16, 2>");
    }
}

TEST_F(BsrmmGrids, large_ext_32x32_clamped)
{
    for(const rocsparse_order order_B : {rocsparse_order_column, rocsparse_order_row})
    {
        check_bsrmm<float, int32_t, int32_t>(handle,
                                             29,
                                             rocsparse_direction_column,
                                             order_B,
                                             bsrmm_limits,
                                             "bsrmm_large_blockdim_kernel_ext<32, 32, 2>");
        check_bsrmm<double, int64_t, int64_t>(handle,
                                              32,
                                              rocsparse_direction_row,
                                              order_B,
                                              bsrmm_limits,
                                              "bsrmm_large_blockdim_kernel_ext<32, 32, 2>");
    }
}

// ---------------------------------------------------------------------------
// bsrmm general (block_dim > 32), nn and nt
// ---------------------------------------------------------------------------

TEST_F(BsrmmGrids, general_clamped)
{
    for(const rocsparse_order order_B : {rocsparse_order_column, rocsparse_order_row})
    {
        check_bsrmm<float, int32_t, int32_t>(handle,
                                             37,
                                             rocsparse_direction_row,
                                             order_B,
                                             bsrmm_limits,
                                             "bsrmm_general_blockdim_kernel");
        check_bsrmm<double, int64_t, int64_t>(handle,
                                              40,
                                              rocsparse_direction_column,
                                              order_B,
                                              bsrmm_limits,
                                              "bsrmm_general_blockdim_kernel");
    }
}

// Control: every bsrmm shape above with the real grid limits. If this fails
// the clamped cases prove nothing.
TEST_F(BsrmmGrids, unclamped_grid_matches_host)
{
    const struct
    {
        int64_t             block_dim;
        rocsparse_direction dir;
    } shapes[] = {{2, rocsparse_direction_row},
                  {3, rocsparse_direction_row},
                  {7, rocsparse_direction_column},
                  {13, rocsparse_direction_row},
                  {29, rocsparse_direction_column},
                  {37, rocsparse_direction_row}};

    for(const auto& s : shapes)
    {
        for(const rocsparse_order order_B : {rocsparse_order_column, rocsparse_order_row})
        {
            check_bsrmm<float, int32_t, int32_t>(
                handle, s.block_dim, s.dir, order_B, unclamped, "bsrmm");
            check_bsrmm<double, int64_t, int64_t>(
                handle, s.block_dim, s.dir, order_B, unclamped, "bsrmm");
        }
    }
}

// ---------------------------------------------------------------------------
// coomm segmented block reduction, grid.x clamped. B in column order and in
// row order (the transposed main kernels), and batched C so the per-batch
// nblocks * n offsets into the reduction buffers are exercised.
// ---------------------------------------------------------------------------

TEST_F(BlockReduceGrids, coomm_segmented_clamped_f32_i32)
{
    check_coomm<float, int32_t>(handle, rocsparse_order_column, 1, reduce_limits);
    check_coomm<float, int32_t>(handle, rocsparse_order_row, 1, reduce_limits);
}

TEST_F(BlockReduceGrids, coomm_segmented_clamped_f64_i64)
{
    check_coomm<double, int64_t>(handle, rocsparse_order_column, 1, reduce_limits);
    check_coomm<double, int64_t>(handle, rocsparse_order_row, 1, reduce_limits);
}

TEST_F(BlockReduceGrids, coomm_segmented_batched_clamped)
{
    check_coomm<float, int32_t>(handle, rocsparse_order_column, 2, reduce_limits);
    check_coomm<double, int64_t>(handle, rocsparse_order_row, 2, reduce_limits);
}

TEST_F(BlockReduceGrids, unclamped_grid_matches_host)
{
    for(const rocsparse_order order_B : {rocsparse_order_column, rocsparse_order_row})
    {
        check_coomm<float, int32_t>(handle, order_B, 1, unclamped);
        check_coomm<double, int64_t>(handle, order_B, 2, unclamped);
    }
}
