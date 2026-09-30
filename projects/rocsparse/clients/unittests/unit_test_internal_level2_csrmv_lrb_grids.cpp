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
// Forced-clamp tests for the grid.x extents of the csrmv LRB kernels and of the
// adaptive partial_scale_y kernel (AISPARSE-662).
//
// Every LRB compute launch clamps grid.x with rocsparse::get_grid_size_x, i.e.
// to min(maxGridSize[0], (2^32 - 1) / blockDim.x) blocks. When the clamp binds,
// the host launches the GRID_STRIDE variant of the kernel, which grid-strides
// over the full logical block count; otherwise it launches the straight-line
// variant, one block per block index. The long-rows grid is further
// rounded down to a whole multiple of num_wgs_per_row, so that all workgroups
// cooperating on one row run in the same grid-stride wave (the spin-wait
// hand-off needs that). At real sizes the clamp needs more than 16.7M blocks.
// These tests shrink handle->properties.maxGridSize[0] to a few blocks for the
// compute stage instead, so a small matrix runs through the clamped, looping
// path, and compare y against a host reference.
//
// BINS. LRB puts row i in bin ceil(log2(nnz_i)) (0 for an empty row) and
// launches one kernel per non-empty bin:
//
//   bins 0-1    csrmvn_lrb_short_rows_kernel      ceil(rows / 256) blocks
//   bins 2-4    csrmvn_lrb_short_rows_2_kernel    ceil(rows / (1024 >> bin))
//   bins 5-8    csrmvn_lrb_medium_rows_warp_reduce_kernel, one wavefront per row
//   bins 9-10   csrmvn_lrb_medium_rows_kernel     one block per row
//   bins 11+    csrmvn_lrb_long_rows_kernel       num_wgs_per_row blocks per row
//
// The matrix below populates bins 0 to 14 with enough rows that every one of
// those grids is wider than 7 blocks, so each of the clamped limits 1, 2, 3 and
// 7 forces the stride loop. For long rows num_wgs_per_row is 3, 6, 11 and 22
// for bins 11 to 14, so the limits cover both "one row per wave" (grid raised
// to num_wgs_per_row) and "two rows per wave" (limit 7, bins 11 and 12).
// Results are checked per bin, so a failure names the kernel that dropped rows.
//
// PARTIAL_SCALE_Y. The adaptive analysis trims leading and trailing runs of
// more than 32 * 256 empty rows out of the adaptive kernel; y is then scaled by
// beta over [0, first_row) and [last_row, m) in partial_scale_y_kernel, which
// only runs for beta != 1. The adaptive matrix has 20000 empty rows on each
// side, so that kernel needs about 40000 threads, i.e. well over 7 blocks.
//
// EXACT ARITHMETIC. Values are small integers, so every product and partial
// sum is exactly representable in float and double and the long-rows atomics
// cannot change the result. The comparison still uses a relative tolerance of a
// few ulps.
//
// TARGET: rocsparse-unit-test-device. The tests drive the public rocsparse_spmv
// entry point and need the complete handle type to reach handle->properties.
//
#include "unit_test_utils.hpp"

#include "rocsparse_handle.hpp"

#include "rocsparse.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    // Shrinks the grid.x limit that get_grid_size_x clamps against, and restores
    // it on scope exit so a failed assertion cannot leak into the next test.
    struct ScopedMaxGridSizeX
    {
        rocsparse_handle handle;
        int              saved;

        ScopedMaxGridSizeX(rocsparse_handle h, int limit)
            : handle(h)
            , saved(h->properties.maxGridSize[0])
        {
            handle->properties.maxGridSize[0] = limit;
        }

        ~ScopedMaxGridSizeX()
        {
            handle->properties.maxGridSize[0] = saved;
        }

        ScopedMaxGridSizeX(const ScopedMaxGridSizeX&) = delete;

        ScopedMaxGridSizeX& operator=(const ScopedMaxGridSizeX&) = delete;
    };

    constexpr int clamped_limits[] = {1, 2, 3, 7};

    constexpr int max_bin = 32;

    // Host CSR matrix. Converted to the index / value types under test on upload.
    struct HostCsr
    {
        int64_t              m   = 0;
        int64_t              n   = 0;
        int64_t              nnz = 0;
        std::vector<int64_t> row_ptr;
        std::vector<int64_t> col_ind;
        std::vector<double>  val;
    };

    int lrb_bin(int64_t row_len)
    {
        int bin = 0;
        while((int64_t(1) << bin) < row_len)
        {
            ++bin;
        }
        return bin;
    }

    const char* lrb_kernel_of_bin(int bin)
    {
        if(bin < 2)
        {
            return "csrmvn_lrb_short_rows_kernel";
        }
        if(bin < 5)
        {
            return "csrmvn_lrb_short_rows_2_kernel";
        }
        if(bin < 9)
        {
            return "csrmvn_lrb_medium_rows_warp_reduce_kernel";
        }
        if(bin < 11)
        {
            return "csrmvn_lrb_medium_rows_kernel";
        }
        return "csrmvn_lrb_long_rows_kernel";
    }

    // Builds a CSR matrix from a list of row lengths. Each row holds consecutive
    // column indices starting at a row-dependent offset, with values in [-2, 2].
    HostCsr make_matrix(const std::vector<int64_t>& row_len, int64_t n)
    {
        HostCsr a;
        a.m = static_cast<int64_t>(row_len.size());
        a.n = n;
        a.row_ptr.assign(a.m + 1, 0);
        for(int64_t i = 0; i < a.m; ++i)
        {
            a.row_ptr[i + 1] = a.row_ptr[i] + row_len[i];
        }
        a.nnz = a.row_ptr[a.m];
        a.col_ind.reserve(a.nnz);
        a.val.reserve(a.nnz);
        for(int64_t i = 0; i < a.m; ++i)
        {
            const int64_t base = (i * 131) % (n - row_len[i] + 1);
            for(int64_t k = 0; k < row_len[i]; ++k)
            {
                a.col_ind.push_back(base + k);
                a.val.push_back(static_cast<double>((i + k) % 5) - 2.0);
            }
        }
        return a;
    }

    // Rows spread over LRB bins 0 to 14 (and some empty rows), shuffled so that
    // the rows of a bin are not contiguous in the matrix.
    HostCsr make_lrb_matrix()
    {
        struct BinRows
        {
            int64_t min_len;
            int64_t max_len;
            int64_t rows;
        };

        const BinRows bins[] = {
            {0, 0, 64}, // bin 0, empty rows
            {1, 1, 3000}, // bin 0, 12 blocks
            {2, 2, 3000}, // bin 1, 12 blocks
            {3, 4, 4000}, // bin 2, 16 blocks
            {5, 8, 2000}, // bin 3, 16 blocks
            {9, 16, 2000}, // bin 4, 32 blocks
            {17, 32, 200}, // bin 5, 50 blocks of 4 wavefronts
            {33, 64, 200}, // bin 6
            {65, 128, 200}, // bin 7
            {129, 256, 200}, // bin 8
            {257, 512, 40}, // bin 9, 40 blocks
            {513, 1024, 40}, // bin 10
            {1025, 2048, 12}, // bin 11, 3 blocks per row, 36 blocks
            {2049, 4096, 8}, // bin 12, 6 blocks per row, 48 blocks
            {4097, 8192, 6}, // bin 13, 11 blocks per row, 66 blocks
            {8193, 16384, 4}, // bin 14, 22 blocks per row, 88 blocks
        };

        std::vector<int64_t> row_len;
        for(const BinRows& b : bins)
        {
            const int64_t span = b.max_len - b.min_len + 1;
            for(int64_t r = 0; r < b.rows; ++r)
            {
                // Hit both ends of the bin, including the exact power of two.
                row_len.push_back(r == 0 ? b.max_len : b.min_len + (r * 7) % span);
            }
        }

        std::mt19937 gen(662);
        std::shuffle(row_len.begin(), row_len.end(), gen);

        return make_matrix(row_len, 20000);
    }

    // 20000 empty rows, 3000 rows of 1 to 64 entries, then 20000 empty rows.
    HostCsr make_adaptive_matrix()
    {
        constexpr int64_t empty = 20000;
        constexpr int64_t body  = 3000;

        std::vector<int64_t> row_len(empty + body + empty, 0);
        for(int64_t i = 0; i < body; ++i)
        {
            row_len[empty + i] = 1 + (i * 13) % 64;
        }
        return make_matrix(row_len, 4096);
    }

    std::vector<double> make_x(int64_t n)
    {
        std::vector<double> x(n);
        for(int64_t i = 0; i < n; ++i)
        {
            x[i] = static_cast<double>(i % 7) - 3.0;
        }
        return x;
    }

    std::vector<double> make_y(int64_t m)
    {
        std::vector<double> y(m);
        for(int64_t i = 0; i < m; ++i)
        {
            y[i] = static_cast<double>(1 + i % 3);
        }
        return y;
    }

    std::vector<double> host_csrmv(const HostCsr&             a,
                                   double                     alpha,
                                   const std::vector<double>& x,
                                   double                     beta,
                                   std::vector<double>        y)
    {
        for(int64_t i = 0; i < a.m; ++i)
        {
            double sum = 0.0;
            for(int64_t p = a.row_ptr[i]; p < a.row_ptr[i + 1]; ++p)
            {
                sum += a.val[p] * x[a.col_ind[p]];
            }
            y[i] = alpha * sum + beta * y[i];
        }
        return y;
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

    struct SpmvDescrs
    {
        rocsparse_spmat_descr mat = nullptr;
        rocsparse_dnvec_descr x   = nullptr;
        rocsparse_dnvec_descr y   = nullptr;

        SpmvDescrs() = default;

        SpmvDescrs(const SpmvDescrs&) = delete;

        SpmvDescrs& operator=(const SpmvDescrs&) = delete;

        ~SpmvDescrs()
        {
            if(mat != nullptr)
            {
                (void)rocsparse_destroy_spmat_descr(mat);
            }
            if(x != nullptr)
            {
                (void)rocsparse_destroy_dnvec_descr(x);
            }
            if(y != nullptr)
            {
                (void)rocsparse_destroy_dnvec_descr(y);
            }
        }
    };

    // y = alpha * A * x + beta * y through rocsparse_spmv (buffer size,
    // preprocess, compute). A non-zero clamp_limit shrinks maxGridSize[0] for
    // the compute stage, which is where the launches under test live.
    //
    // Returns an empty vector and records a gtest failure on any API error.
    template <typename T, typename I, typename J>
    std::vector<double> device_csrmv(rocsparse_handle           handle,
                                     const HostCsr&             a,
                                     rocsparse_spmv_alg         alg,
                                     double                     alpha_d,
                                     const std::vector<double>& x,
                                     double                     beta_d,
                                     const std::vector<double>& y_in,
                                     int                        clamp_limit)
    {
        device_vector<I> d_row_ptr(convert<I>(a.row_ptr));
        device_vector<J> d_col_ind(convert<J>(a.col_ind));
        device_vector<T> d_val(convert_values<T>(a.val));
        device_vector<T> d_x(convert_values<T>(x));
        device_vector<T> d_y(convert_values<T>(y_in));

        if(d_row_ptr.ptr == nullptr || d_col_ind.ptr == nullptr || d_val.ptr == nullptr
           || d_x.ptr == nullptr || d_y.ptr == nullptr)
        {
            ADD_FAILURE() << "device allocation failed";
            return {};
        }

        SpmvDescrs descrs;
        if(rocsparse_create_csr_descr(&descrs.mat,
                                      a.m,
                                      a.n,
                                      a.nnz,
                                      d_row_ptr.ptr,
                                      d_col_ind.ptr,
                                      d_val.ptr,
                                      it_of<I>(),
                                      it_of<J>(),
                                      rocsparse_index_base_zero,
                                      dt_of<T>())
               != rocsparse_status_success
           || rocsparse_create_dnvec_descr(&descrs.x, a.n, d_x.ptr, dt_of<T>())
                  != rocsparse_status_success
           || rocsparse_create_dnvec_descr(&descrs.y, a.m, d_y.ptr, dt_of<T>())
                  != rocsparse_status_success)
        {
            ADD_FAILURE() << "descriptor creation failed";
            return {};
        }

        const T alpha = static_cast<T>(alpha_d);
        const T beta  = static_cast<T>(beta_d);

        auto spmv = [&](rocsparse_spmv_stage stage, size_t* buffer_size, void* buffer) {
            return rocsparse_spmv(handle,
                                  rocsparse_operation_none,
                                  &alpha,
                                  descrs.mat,
                                  descrs.x,
                                  &beta,
                                  descrs.y,
                                  dt_of<T>(),
                                  alg,
                                  stage,
                                  buffer_size,
                                  buffer);
        };

        size_t           buffer_size = 0;
        rocsparse_status status = spmv(rocsparse_spmv_stage_buffer_size, &buffer_size, nullptr);
        if(status != rocsparse_status_success)
        {
            ADD_FAILURE() << "rocsparse_spmv buffer_size returned status " << status;
            return {};
        }

        device_vector<char> d_buffer(buffer_size > 0 ? buffer_size : size_t(1));
        if(d_buffer.ptr == nullptr)
        {
            ADD_FAILURE() << "temp buffer allocation of " << buffer_size << " bytes failed";
            return {};
        }

        status = spmv(rocsparse_spmv_stage_preprocess, &buffer_size, d_buffer.ptr);
        if(status != rocsparse_status_success)
        {
            ADD_FAILURE() << "rocsparse_spmv preprocess returned status " << status;
            return {};
        }

        {
            std::unique_ptr<ScopedMaxGridSizeX> clamp;
            if(clamp_limit > 0)
            {
                clamp.reset(new ScopedMaxGridSizeX(handle, clamp_limit));
            }
            status = spmv(rocsparse_spmv_stage_compute, &buffer_size, d_buffer.ptr);
        }
        if(status != rocsparse_status_success)
        {
            ADD_FAILURE() << "rocsparse_spmv compute returned status " << status;
            return {};
        }

        if(hipDeviceSynchronize() != hipSuccess)
        {
            ADD_FAILURE() << "hipDeviceSynchronize failed";
            return {};
        }

        const std::vector<T> y = to_host(d_y);
        return std::vector<double>(y.begin(), y.end());
    }

    template <typename T>
    bool near(double got, double want)
    {
        const double tol = 4.0 * std::numeric_limits<T>::epsilon() * std::max(1.0, std::abs(want));
        return std::abs(got - want) <= tol;
    }

    // Compares y per LRB bin, so a failure names the kernel that owns the rows.
    template <typename T>
    void expect_lrb_match(const HostCsr&             a,
                          const std::vector<double>& got,
                          const std::vector<double>& want,
                          const std::string&         what)
    {
        ASSERT_EQ(got.size(), want.size());

        std::vector<int64_t> rows(max_bin, 0);
        std::vector<int64_t> bad(max_bin, 0);
        std::vector<int64_t> first_bad(max_bin, -1);
        for(int64_t i = 0; i < a.m; ++i)
        {
            const int bin = lrb_bin(a.row_ptr[i + 1] - a.row_ptr[i]);
            ++rows[bin];
            if(!near<T>(got[i], want[i]))
            {
                if(bad[bin]++ == 0)
                {
                    first_bad[bin] = i;
                }
            }
        }

        for(int bin = 0; bin < max_bin; ++bin)
        {
            if(rows[bin] == 0)
            {
                continue;
            }
            const int64_t i = first_bad[bin];
            EXPECT_EQ(bad[bin], 0)
                << what << ": " << lrb_kernel_of_bin(bin) << " (bin " << bin << ") got " << bad[bin]
                << " of " << rows[bin] << " rows wrong; first is row " << i << " with "
                << (i >= 0 ? a.row_ptr[i + 1] - a.row_ptr[i] : 0) << " entries, got "
                << (i >= 0 ? got[i] : 0.0) << " want " << (i >= 0 ? want[i] : 0.0);
        }
    }

    template <typename T>
    void expect_adaptive_match(const std::vector<double>& got,
                               const std::vector<double>& want,
                               const std::string&         what)
    {
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
        EXPECT_EQ(bad, 0) << what << ": " << bad << " of " << got.size()
                          << " rows wrong; first is row " << first << ", got "
                          << (first >= 0 ? got[first] : 0.0) << " want "
                          << (first >= 0 ? want[first] : 0.0);
    }

    std::string describe(const char* alg, int clamp_limit, double beta)
    {
        std::ostringstream os;
        os << alg << ", maxGridSize[0] = ";
        if(clamp_limit > 0)
        {
            os << clamp_limit;
        }
        else
        {
            os << "unclamped";
        }
        os << ", beta = " << beta;
        return os.str();
    }

    constexpr double lrb_alpha = 2.0;
    constexpr double lrb_beta  = -3.0;

    template <typename T, typename I, typename J>
    void check_lrb(rocsparse_handle handle, bool clamped)
    {
        const HostCsr             a    = make_lrb_matrix();
        const std::vector<double> x    = make_x(a.n);
        const std::vector<double> y    = make_y(a.m);
        const std::vector<double> want = host_csrmv(a, lrb_alpha, x, lrb_beta, y);

        const std::vector<int> limits
            = clamped ? std::vector<int>(std::begin(clamped_limits), std::end(clamped_limits))
                      : std::vector<int>{0};
        for(const int limit : limits)
        {
            const std::vector<double> got = device_csrmv<T, I, J>(
                handle, a, rocsparse_spmv_alg_csr_lrb, lrb_alpha, x, lrb_beta, y, limit);
            expect_lrb_match<T>(a, got, want, describe("lrb", limit, lrb_beta));
        }
    }

    template <typename T, typename I, typename J>
    void check_adaptive_partial_scale_y(rocsparse_handle handle, bool clamped)
    {
        const HostCsr             a     = make_adaptive_matrix();
        const std::vector<double> x     = make_x(a.n);
        const std::vector<double> y     = make_y(a.m);
        const double              alpha = 2.0;

        const std::vector<int> limits
            = clamped ? std::vector<int>(std::begin(clamped_limits), std::end(clamped_limits))
                      : std::vector<int>{0};

        // beta = 0 takes the store-zero branch of partial_scale_y, beta = -3 the
        // multiply branch. beta = 1 skips the kernel entirely.
        for(const double beta : {-3.0, 0.0})
        {
            const std::vector<double> want = host_csrmv(a, alpha, x, beta, y);
            for(const int limit : limits)
            {
                const std::vector<double> got = device_csrmv<T, I, J>(
                    handle, a, rocsparse_spmv_alg_csr_adaptive, alpha, x, beta, y, limit);
                expect_adaptive_match<T>(got, want, describe("adaptive", limit, beta));
            }
        }
    }

    using CsrmvLrbGrids      = HandleTest;
    using CsrmvAdaptiveGrids = HandleTest;
}

// ---------------------------------------------------------------------------
// LRB, grid.x clamped to 1, 2, 3 and 7 blocks for every kernel.
// ---------------------------------------------------------------------------

TEST_F(CsrmvLrbGrids, clamped_grid_stride_f32_i32)
{
    check_lrb<float, int32_t, int32_t>(handle, true);
}

TEST_F(CsrmvLrbGrids, clamped_grid_stride_f64_i32)
{
    check_lrb<double, int32_t, int32_t>(handle, true);
}

TEST_F(CsrmvLrbGrids, clamped_grid_stride_f32_i64)
{
    check_lrb<float, int64_t, int64_t>(handle, true);
}

TEST_F(CsrmvLrbGrids, clamped_grid_stride_f64_i64)
{
    check_lrb<double, int64_t, int64_t>(handle, true);
}

// Control: the same problems with the real grid.x limit, which run the
// straight-line variants. If this fails the clamped cases above prove nothing.
TEST_F(CsrmvLrbGrids, unclamped_grid_matches_host)
{
    check_lrb<float, int32_t, int32_t>(handle, false);
    check_lrb<double, int32_t, int32_t>(handle, false);
    check_lrb<float, int64_t, int64_t>(handle, false);
    check_lrb<double, int64_t, int64_t>(handle, false);
}

// ---------------------------------------------------------------------------
// Adaptive partial_scale_y, grid.x clamped to 1, 2, 3 and 7 blocks.
// ---------------------------------------------------------------------------

TEST_F(CsrmvAdaptiveGrids, partial_scale_y_clamped_f32_i32)
{
    check_adaptive_partial_scale_y<float, int32_t, int32_t>(handle, true);
}

TEST_F(CsrmvAdaptiveGrids, partial_scale_y_clamped_f64_i32)
{
    check_adaptive_partial_scale_y<double, int32_t, int32_t>(handle, true);
}

TEST_F(CsrmvAdaptiveGrids, partial_scale_y_clamped_f32_i64)
{
    check_adaptive_partial_scale_y<float, int64_t, int64_t>(handle, true);
}

TEST_F(CsrmvAdaptiveGrids, partial_scale_y_clamped_f64_i64)
{
    check_adaptive_partial_scale_y<double, int64_t, int64_t>(handle, true);
}

TEST_F(CsrmvAdaptiveGrids, unclamped_grid_matches_host)
{
    check_adaptive_partial_scale_y<float, int32_t, int32_t>(handle, false);
    check_adaptive_partial_scale_y<double, int32_t, int32_t>(handle, false);
    check_adaptive_partial_scale_y<float, int64_t, int64_t>(handle, false);
    check_adaptive_partial_scale_y<double, int64_t, int64_t>(handle, false);
}
