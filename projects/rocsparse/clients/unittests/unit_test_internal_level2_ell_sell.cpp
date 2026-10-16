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
// Device unit tests for two level-2 ELL / sliced-ELL code paths that are
// otherwise only reached by YAML integration tests:
//
//  - ELL triangular solve (ellsv), reached through the generic
//    rocsparse_sptrsv descriptor API for rocsparse_format_ell
//    (library/src/level2/rocsparse_sptrsv.cpp). clients/tests/test_sptrsv_ell.yaml
//    only drives it with transA none; this file adds a tiny device-level
//    check of the numeric solution, plus a unit-diagonal case and an
//    explicit-zero-pivot case. Gated the same way the library gates ELL
//    TRSV support (BUILD_WITH_ELL_TRSV / ROCSPARSE_WITH_ELL_TRSV): when that
//    symbol is not defined the test is skipped with a clear message instead
//    of being compiled out, so it still shows up (as skipped) in a
//    BUILD_WITH_ELL_TRSV=OFF build.
//
//  - Sliced ELL SpMV (sellmv), reached through rocsparse_v2_spmv for
//    rocsparse_format_sell (library/src/level2/rocsparse_sellmv_impl.cpp,
//    rocsparse_sellmv.cpp, sellmv_device.h). The dispatch in
//    rocsparse_sellmv_impl.cpp picks the small-slice kernel
//    (sellmvn_device / sellmvt_device) for sell_slice_size <= 128, the
//    large-slice kernel (sellmvn_large_slice_device /
//    sellmvt_large_slice_device) above that, and falls back to csrmv when
//    sell_slice_size == 1. This file exercises all three, in both
//    directions, including a transpose case whose matrix size is not a
//    multiple of the slice size (a partial last slice).
//
//    NOTE: on current develop, sellmvt_device reads x[row] before checking
//    row < m (the bounds check only guards the start/end offsets used by the
//    accumulation loop, which is why padding threads never touch
//    sell_col_ind / sell_val). That stray read of x is harmless here -- the
//    loaded value is never used because the padding thread's own start/end
//    range is empty -- but it means the transpose/partial-last-slice test
//    below must size x so that the read stays in-bounds for every slice
//    (see build_bidiagonal_sell). This fix is already implemented in open PR
//    https://github.com/ROCm/rocm-libraries/pull/13339 (not merged at the
//    time of writing); this test does not attempt to change
//    sellmv_device.h, it only asserts the numeric result, which is correct
//    either way.
//
#include "unit_test_utils.hpp"

#include "rocsparse.h"

#include <cmath>

using namespace rocsparse_ut;

namespace
{
    // Loose-enough tolerance for the float computations below. Most of the
    // matrices are built from small powers of two so the host reference and
    // the device result usually match exactly; this only guards against
    // summation-order differences.
    constexpr double kTol = 1e-4;

    void expect_vectors_near(const std::vector<float>& expected,
                             const std::vector<float>& actual,
                             double                    tol = kTol)
    {
        ASSERT_EQ(expected.size(), actual.size());
        for(size_t i = 0; i < expected.size(); ++i)
        {
            EXPECT_NEAR(static_cast<double>(actual[i]), static_cast<double>(expected[i]), tol)
                << "index " << i;
        }
    }

    int64_t count_nonneg(const std::vector<int32_t>& v)
    {
        int64_t c = 0;
        for(int32_t e : v)
        {
            if(e >= 0)
            {
                ++c;
            }
        }
        return c;
    }

// ===========================================================================
// ellsv : ELL triangular solve via the generic rocsparse_sptrsv descriptor.
// Only built / exercised when the library is built with ELL TRSV support.
// ===========================================================================
#ifdef ROCSPARSE_WITH_ELL_TRSV

    // Host reference for a tiny lower-triangular ELL solve, matching the
    // semantics of rocsparse::ellsv_device (library/src/level2/ellsv_device.h):
    //  - a unit diagonal entry is ignored for the division even if one is
    //    stored in the ELL data,
    //  - an explicit zero on a non-unit diagonal is treated as 1 after
    //    recording the pivot (so the numeric result still comes out
    //    deterministic, matching what the kernel computes),
    //  - only the strictly-lower entries (plus the diagonal) contribute.
    // ell_col_ind / ell_val are laid out exactly like rocsparse_create_ell_descr
    // expects: column-major, ELL_IND(row, slot, m, width) = slot * m + row.
    std::vector<float> host_ellsv_lower(int64_t                      m,
                                        int64_t                      n,
                                        int64_t                      ell_width,
                                        const std::vector<int32_t>& ell_col_ind,
                                        const std::vector<float>&   ell_val,
                                        const std::vector<float>&   x,
                                        float                        alpha,
                                        rocsparse_diag_type          diag_type,
                                        int64_t*                     zero_pivot)
    {
        std::vector<float> y(static_cast<size_t>(m), 0.0f);
        *zero_pivot = -1;

        for(int64_t row = 0; row < m; ++row)
        {
            float sum      = alpha * x[static_cast<size_t>(row)];
            float diag_val = 1.0f;

            for(int64_t p = 0; p < ell_width; ++p)
            {
                const int64_t idx = p * m + row;
                const int32_t col = ell_col_ind[static_cast<size_t>(idx)];
                if(col < 0 || col >= n)
                {
                    continue;
                }
                float val = ell_val[static_cast<size_t>(idx)];

                if(col == row)
                {
                    if(diag_type == rocsparse_diag_type_non_unit)
                    {
                        if(val == 0.0f)
                        {
                            if(*zero_pivot == -1 || row < *zero_pivot)
                            {
                                *zero_pivot = row;
                            }
                            val = 1.0f;
                        }
                        diag_val = 1.0f / val;
                    }
                    continue;
                }

                // Lower triangular solve: only strictly-lower entries feed
                // back already-solved y values.
                if(col > row)
                {
                    continue;
                }
                sum -= val * y[static_cast<size_t>(col)];
            }

            y[static_cast<size_t>(row)]
                = (diag_type == rocsparse_diag_type_non_unit) ? sum * diag_val : sum;
        }
        return y;
    }

    // Runs buffer_size / analysis / compute on sptrsv_descr for a lower
    // triangular ELL matrix A, operation none, alpha on the host. Returns the
    // zero-pivot position reported after analysis and after compute (-1 if
    // none was reported at that stage).
    void run_ellsv(rocsparse_handle       handle,
                   rocsparse_spmat_descr  A,
                   rocsparse_dnvec_descr  x,
                   rocsparse_dnvec_descr  y,
                   float                  alpha,
                   int64_t*               analysis_pivot,
                   int64_t*               solve_pivot)
    {
        rocsparse_sptrsv_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_sptrsv_descr(&descr), rocsparse_status_success);

        const rocsparse_operation  trans = rocsparse_operation_none;
        const rocsparse_sptrsv_alg alg   = rocsparse_sptrsv_alg_default;
        const rocsparse_datatype   dt    = rocsparse_datatype_f32_r;

        ASSERT_EQ(rocsparse_sptrsv_set_input(
                      handle, descr, rocsparse_sptrsv_input_operation, &trans, sizeof(trans), nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(
            rocsparse_sptrsv_set_input(handle, descr, rocsparse_sptrsv_input_alg, &alg, sizeof(alg), nullptr),
            rocsparse_status_success);
        ASSERT_EQ(rocsparse_sptrsv_set_input(handle,
                                             descr,
                                             rocsparse_sptrsv_input_scalar_datatype,
                                             &dt,
                                             sizeof(dt),
                                             nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_sptrsv_set_input(handle,
                                             descr,
                                             rocsparse_sptrsv_input_compute_datatype,
                                             &dt,
                                             sizeof(dt),
                                             nullptr),
                  rocsparse_status_success);
        // Mandatory: the descriptor's analysis policy defaults to an invalid
        // sentinel, so rocsparse::ellsv_analysis's ROCSPARSE_CHECKARG_ENUM on it
        // fails with rocsparse_status_invalid_value unless this is set.
        const rocsparse_analysis_policy apol = rocsparse_analysis_policy_force;
        ASSERT_EQ(rocsparse_sptrsv_set_input(handle,
                                             descr,
                                             rocsparse_sptrsv_input_analysis_policy,
                                             &apol,
                                             sizeof(apol),
                                             nullptr),
                  rocsparse_status_success);

        size_t bs_analysis = 0;
        ASSERT_EQ(rocsparse_sptrsv_buffer_size(
                      handle, descr, A, x, y, rocsparse_sptrsv_stage_analysis, &bs_analysis, nullptr),
                  rocsparse_status_success);
        device_vector<char> buf_a{bs_analysis ? bs_analysis : size_t(1)};
        ASSERT_TRUE(buf_a.ptr);
        ASSERT_EQ(rocsparse_sptrsv(handle,
                                   descr,
                                   A,
                                   x,
                                   y,
                                   rocsparse_sptrsv_stage_analysis,
                                   bs_analysis,
                                   bs_analysis ? buf_a.ptr : nullptr,
                                   nullptr),
                  rocsparse_status_success);

        int64_t          pivot        = -1;
        rocsparse_status pivot_status = rocsparse_sptrsv_get_output(
            handle, descr, rocsparse_sptrsv_output_zero_pivot_position, &pivot, sizeof(pivot), nullptr);
        ASSERT_TRUE(pivot_status == rocsparse_status_success
                    || pivot_status == rocsparse_status_zero_pivot);
        *analysis_pivot = pivot;

        const void* alpha_ptr = &alpha;
        ASSERT_EQ(rocsparse_sptrsv_set_input(handle,
                                             descr,
                                             rocsparse_sptrsv_input_scalar_alpha,
                                             alpha_ptr,
                                             sizeof(alpha_ptr),
                                             nullptr),
                  rocsparse_status_success);

        size_t bs_compute = 0;
        ASSERT_EQ(rocsparse_sptrsv_buffer_size(
                      handle, descr, A, x, y, rocsparse_sptrsv_stage_compute, &bs_compute, nullptr),
                  rocsparse_status_success);
        device_vector<char> buf_c{bs_compute ? bs_compute : size_t(1)};
        ASSERT_TRUE(buf_c.ptr);
        ASSERT_EQ(rocsparse_sptrsv(handle,
                                   descr,
                                   A,
                                   x,
                                   y,
                                   rocsparse_sptrsv_stage_compute,
                                   bs_compute,
                                   bs_compute ? buf_c.ptr : nullptr,
                                   nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        pivot        = -1;
        pivot_status = rocsparse_sptrsv_get_output(
            handle, descr, rocsparse_sptrsv_output_zero_pivot_position, &pivot, sizeof(pivot), nullptr);
        ASSERT_TRUE(pivot_status == rocsparse_status_success
                    || pivot_status == rocsparse_status_zero_pivot);
        *solve_pivot = pivot;

        EXPECT_EQ(rocsparse_destroy_sptrsv_descr(descr), rocsparse_status_success);
    }

#endif // ROCSPARSE_WITH_ELL_TRSV

    // ===========================================================================
    // sellmv : Sliced ELL SpMV helpers, used regardless of ROCSPARSE_WITH_ELL_TRSV.
    // ===========================================================================

    // Host reference for a sliced-ELL SpMV: y := alpha * op(A) * x + beta * y.
    // Mirrors the mathematical semantics of sellmv_device.h without any of the
    // thread/slice tricks, so it independently checks the kernels' output.
    // sell_col_ind / sell_val layout: for slice s spanning
    // [slice_offsets[s], slice_offsets[s+1]) with depth = (end - start) /
    // sell_slice_size, slot p, local row r (0 <= r < sell_slice_size), the
    // entry lives at slice_offsets[s] + p * sell_slice_size + r.
    std::vector<float> host_sellmv(int64_t                      m,
                                   int64_t                      n,
                                   int64_t                      sell_slice_size,
                                   const std::vector<int32_t>& slice_offsets,
                                   const std::vector<int32_t>& col_ind,
                                   const std::vector<float>&   val,
                                   const std::vector<float>&   x,
                                   float                        alpha,
                                   float                        beta,
                                   const std::vector<float>&   y_init,
                                   rocsparse_operation          trans)
    {
        const int64_t       out_size = (trans == rocsparse_operation_none) ? m : n;
        std::vector<float> result(static_cast<size_t>(out_size));
        for(int64_t i = 0; i < out_size; ++i)
        {
            result[static_cast<size_t>(i)] = beta * y_init[static_cast<size_t>(i)];
        }

        const int64_t nslices = (m + sell_slice_size - 1) / sell_slice_size;
        for(int64_t s = 0; s < nslices; ++s)
        {
            const int64_t start = slice_offsets[static_cast<size_t>(s)];
            const int64_t end   = slice_offsets[static_cast<size_t>(s + 1)];
            const int64_t depth = (end - start) / sell_slice_size;

            for(int64_t r = 0; r < sell_slice_size; ++r)
            {
                const int64_t row = s * sell_slice_size + r;
                if(row >= m)
                {
                    continue;
                }
                for(int64_t p = 0; p < depth; ++p)
                {
                    const int64_t idx = start + p * sell_slice_size + r;
                    const int32_t col = col_ind[static_cast<size_t>(idx)];
                    if(col < 0)
                    {
                        continue;
                    }
                    const float v = val[static_cast<size_t>(idx)];
                    if(trans == rocsparse_operation_none)
                    {
                        result[static_cast<size_t>(row)] += alpha * v * x[static_cast<size_t>(col)];
                    }
                    else
                    {
                        result[static_cast<size_t>(col)] += alpha * v * x[static_cast<size_t>(row)];
                    }
                }
            }
        }
        return result;
    }

    // x[i] = i + 1. A constant x would make the non-transposed result the
    // plain sum of a row's values, independent of which column each value
    // claims, so a scrambled col_ind would go unnoticed. All values are small
    // integers, so the float arithmetic stays exact.
    std::vector<float> make_varying_x(int64_t size)
    {
        std::vector<float> x(static_cast<size_t>(size));
        for(int64_t i = 0; i < size; ++i)
        {
            x[static_cast<size_t>(i)] = static_cast<float>(i + 1);
        }
        return x;
    }

    // Builds a tiny general (non-triangular) sliced-ELL matrix with a main
    // diagonal (value row+1) and a sub-diagonal (value row, for row >= 1),
    // depth 2, padded with -1 / 0 where a slot has no entry (row 0's missing
    // sub-diagonal, and every slot in a slice's padding rows). Suitable for
    // any sell_slice_size >= 2; for sell_slice_size == 1 use
    // build_bidiagonal_sell_csr_exact instead (slice_size == 1 is dispatched
    // straight to the CSR fallback, which does not skip padded entries).
    void build_bidiagonal_sell(int64_t                m,
                               int64_t                sell_slice_size,
                               std::vector<int32_t>& slice_offsets,
                               std::vector<int32_t>& col_ind,
                               std::vector<float>&   val)
    {
        constexpr int64_t depth   = 2;
        const int64_t     nslices = (m + sell_slice_size - 1) / sell_slice_size;

        slice_offsets.assign(static_cast<size_t>(nslices + 1), 0);
        for(int64_t s = 0; s <= nslices; ++s)
        {
            slice_offsets[static_cast<size_t>(s)] = static_cast<int32_t>(s * sell_slice_size * depth);
        }

        const int64_t colval_size = nslices * sell_slice_size * depth;
        col_ind.assign(static_cast<size_t>(colval_size), -1);
        val.assign(static_cast<size_t>(colval_size), 0.0f);

        for(int64_t s = 0; s < nslices; ++s)
        {
            const int64_t start = slice_offsets[static_cast<size_t>(s)];
            for(int64_t r = 0; r < sell_slice_size; ++r)
            {
                const int64_t row = s * sell_slice_size + r;
                if(row >= m)
                {
                    continue;
                }
                // slot 0: main diagonal.
                const int64_t idx0            = start + 0 * sell_slice_size + r;
                col_ind[static_cast<size_t>(idx0)] = static_cast<int32_t>(row);
                val[static_cast<size_t>(idx0)]     = static_cast<float>(row + 1);

                // slot 1: sub-diagonal (none for row 0).
                if(row >= 1)
                {
                    const int64_t idx1            = start + 1 * sell_slice_size + r;
                    col_ind[static_cast<size_t>(idx1)] = static_cast<int32_t>(row - 1);
                    val[static_cast<size_t>(idx1)]     = static_cast<float>(row);
                }
            }
        }
    }

    // Same bidiagonal pattern, but as an exact CSR structure (slice_offsets
    // behaves as a CSR row pointer, no padding) for the sell_slice_size == 1
    // dispatch, which routes straight through rocsparse::csrmv_template.
    void build_bidiagonal_sell_csr_exact(int64_t                m,
                                         std::vector<int32_t>& slice_offsets,
                                         std::vector<int32_t>& col_ind,
                                         std::vector<float>&   val)
    {
        slice_offsets.assign(static_cast<size_t>(m + 1), 0);
        col_ind.clear();
        val.clear();

        int32_t offset = 0;
        for(int64_t row = 0; row < m; ++row)
        {
            slice_offsets[static_cast<size_t>(row)] = offset;
            if(row >= 1)
            {
                col_ind.push_back(static_cast<int32_t>(row - 1));
                val.push_back(static_cast<float>(row));
                ++offset;
            }
            col_ind.push_back(static_cast<int32_t>(row));
            val.push_back(static_cast<float>(row + 1));
            ++offset;
        }
        slice_offsets[static_cast<size_t>(m)] = offset;
    }

    // Creates the SELL spmat descriptor from host data, runs the two-stage
    // (analysis, compute) rocsparse_v2_spmv pipeline with alg = sell, and
    // checks the result against `expected`.
    void run_v2_spmv_sell(rocsparse_handle            handle,
                          rocsparse_operation         trans,
                          int64_t                     m,
                          int64_t                     n,
                          const std::vector<int32_t>& slice_offsets,
                          const std::vector<int32_t>& col_ind,
                          const std::vector<float>&   val,
                          int64_t                     sell_slice_size,
                          const std::vector<float>&   hx,
                          float                        alpha,
                          float                        beta,
                          const std::vector<float>&   hy_init,
                          const std::vector<float>&   expected)
    {
        device_vector<int32_t> d_offs{slice_offsets};
        device_vector<int32_t> d_cols{col_ind};
        device_vector<float>   d_vals{val};
        ASSERT_TRUE(d_offs.ptr && d_cols.ptr && d_vals.ptr);

        const int64_t nnz = count_nonneg(col_ind);
        ASSERT_GT(nnz, 0);

        rocsparse_spmat_descr A = nullptr;
        ASSERT_EQ(rocsparse_create_sell_descr(&A,
                                              m,
                                              n,
                                              nnz,
                                              sell_slice_size,
                                              static_cast<int64_t>(col_ind.size()),
                                              d_offs.ptr,
                                              d_cols.ptr,
                                              d_vals.ptr,
                                              rocsparse_indextype_i32,
                                              rocsparse_indextype_i32,
                                              rocsparse_index_base_zero,
                                              rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        const int64_t x_size = (trans == rocsparse_operation_none) ? n : m;
        const int64_t y_size = (trans == rocsparse_operation_none) ? m : n;
        ASSERT_EQ(static_cast<int64_t>(hx.size()), x_size);
        ASSERT_EQ(static_cast<int64_t>(hy_init.size()), y_size);
        ASSERT_EQ(static_cast<int64_t>(expected.size()), y_size);

        device_vector<float> x{hx};
        device_vector<float> y{hy_init};
        ASSERT_TRUE(x.ptr && y.ptr);

        rocsparse_dnvec_descr dx = nullptr, dy = nullptr;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&dx, x_size, x.ptr, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&dy, y_size, y.ptr, rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        rocsparse_spmv_descr spmv_descr = nullptr;
        ASSERT_EQ(rocsparse_create_spmv_descr(&spmv_descr), rocsparse_status_success);

        const rocsparse_spmv_alg alg = rocsparse_spmv_alg_sell;
        const rocsparse_datatype dt  = rocsparse_datatype_f32_r;
        ASSERT_EQ(rocsparse_spmv_set_input(
                      handle, spmv_descr, rocsparse_spmv_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spmv_set_input(
                      handle, spmv_descr, rocsparse_spmv_input_operation, &trans, sizeof(trans), nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spmv_set_input(handle,
                                           spmv_descr,
                                           rocsparse_spmv_input_scalar_datatype,
                                           &dt,
                                           sizeof(dt),
                                           nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spmv_set_input(handle,
                                           spmv_descr,
                                           rocsparse_spmv_input_compute_datatype,
                                           &dt,
                                           sizeof(dt),
                                           nullptr),
                  rocsparse_status_success);

        size_t bs_analysis = 0;
        ASSERT_EQ(rocsparse_v2_spmv_buffer_size(handle,
                                                spmv_descr,
                                                A,
                                                dx,
                                                dy,
                                                rocsparse_v2_spmv_stage_analysis,
                                                &bs_analysis,
                                                nullptr),
                  rocsparse_status_success);
        device_vector<char> buf_a{bs_analysis ? bs_analysis : size_t(1)};
        ASSERT_TRUE(buf_a.ptr);
        ASSERT_EQ(rocsparse_v2_spmv(handle,
                                    spmv_descr,
                                    &alpha,
                                    A,
                                    dx,
                                    &beta,
                                    dy,
                                    rocsparse_v2_spmv_stage_analysis,
                                    bs_analysis,
                                    bs_analysis ? buf_a.ptr : nullptr,
                                    nullptr),
                  rocsparse_status_success);

        size_t bs_compute = 0;
        ASSERT_EQ(rocsparse_v2_spmv_buffer_size(handle,
                                                spmv_descr,
                                                A,
                                                dx,
                                                dy,
                                                rocsparse_v2_spmv_stage_compute,
                                                &bs_compute,
                                                nullptr),
                  rocsparse_status_success);
        device_vector<char> buf_c{bs_compute ? bs_compute : size_t(1)};
        ASSERT_TRUE(buf_c.ptr);
        ASSERT_EQ(rocsparse_v2_spmv(handle,
                                    spmv_descr,
                                    &alpha,
                                    A,
                                    dx,
                                    &beta,
                                    dy,
                                    rocsparse_v2_spmv_stage_compute,
                                    bs_compute,
                                    bs_compute ? buf_c.ptr : nullptr,
                                    nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        expect_vectors_near(expected, to_host(y));

        EXPECT_EQ(rocsparse_destroy_spmv_descr(spmv_descr), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_dnvec_descr(dx), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_dnvec_descr(dy), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    }

} // namespace

// ===========================================================================
// ellsv : ELL triangular solve (rocsparse_sptrsv on rocsparse_format_ell).
// ===========================================================================
class EllSv : public HandleTest
{
};

#ifdef ROCSPARSE_WITH_ELL_TRSV

// Lower triangular, non-unit diagonal, no pivot. 3x3:
//   [2       ]   [y0]   [4 ]
//   [1  4    ] * [y1] = [12]
//   [  2  8  ]   [y2]   [40]
// All constants are small powers of two, so the float arithmetic is exact.
TEST_F(EllSv, lower_non_unit)
{
    const int64_t m = 3, n = 3, width = 2;
    // ELL_IND(row, slot, m, width) = slot * m + row.
    std::vector<int32_t> col_ind{0, 0, 1, -1, 1, 2};
    std::vector<float>   val{2, 1, 2, 0, 4, 8};

    device_vector<int32_t> d_col{col_ind};
    device_vector<float>   d_val{val};
    ASSERT_TRUE(d_col.ptr && d_val.ptr);

    rocsparse_spmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_ell_descr(
                  &A, m, n, d_col.ptr, d_val.ptr, width, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_datatype_f32_r),
              rocsparse_status_success);

    const rocsparse_fill_mode   fill_mode = rocsparse_fill_mode_lower;
    const rocsparse_diag_type   diag_type = rocsparse_diag_type_non_unit;
    const rocsparse_matrix_type mat_type  = rocsparse_matrix_type_general;
    ASSERT_EQ(rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &fill_mode, sizeof(fill_mode)),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &diag_type, sizeof(diag_type)),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_spmat_set_attribute(A, rocsparse_spmat_matrix_type, &mat_type, sizeof(mat_type)),
              rocsparse_status_success);

    std::vector<float> hx{4, 12, 40};
    const float         alpha = 1.0f;

    device_vector<float> x{hx};
    device_vector<float> y{std::vector<float>(static_cast<size_t>(m), 0.0f)};
    ASSERT_TRUE(x.ptr && y.ptr);

    rocsparse_dnvec_descr dx = nullptr, dy = nullptr;
    ASSERT_EQ(rocsparse_create_dnvec_descr(&dx, n, x.ptr, rocsparse_datatype_f32_r), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnvec_descr(&dy, m, y.ptr, rocsparse_datatype_f32_r), rocsparse_status_success);

    int64_t analysis_pivot = -2, solve_pivot = -2;
    run_ellsv(handle, A, dx, dy, alpha, &analysis_pivot, &solve_pivot);
    EXPECT_EQ(analysis_pivot, -1);
    EXPECT_EQ(solve_pivot, -1);

    int64_t             ref_pivot = -1;
    std::vector<float> expected
        = host_ellsv_lower(m, n, width, col_ind, val, hx, alpha, diag_type, &ref_pivot);
    EXPECT_EQ(ref_pivot, -1);
    expect_vectors_near(expected, to_host(y));

    EXPECT_EQ(rocsparse_destroy_dnvec_descr(dx), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnvec_descr(dy), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
}

// Lower triangular, unit diagonal. The stored diagonal entries carry a
// deliberately wrong value (999) that must be ignored by the solve. 3x3:
//   [1          ]   [y0]   [2]
//   [0.5  1     ] * [y1] = [4]
//   [    0.25  1]   [y2]   [8]
TEST_F(EllSv, lower_unit_diagonal)
{
    const int64_t m = 3, n = 3, width = 2;
    std::vector<int32_t> col_ind{-1, 0, 1, 0, 1, 2};
    std::vector<float>   val{0, 0.5f, 0.25f, 999, 999, 999};

    device_vector<int32_t> d_col{col_ind};
    device_vector<float>   d_val{val};
    ASSERT_TRUE(d_col.ptr && d_val.ptr);

    rocsparse_spmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_ell_descr(
                  &A, m, n, d_col.ptr, d_val.ptr, width, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_datatype_f32_r),
              rocsparse_status_success);

    const rocsparse_fill_mode   fill_mode = rocsparse_fill_mode_lower;
    const rocsparse_diag_type   diag_type = rocsparse_diag_type_unit;
    const rocsparse_matrix_type mat_type  = rocsparse_matrix_type_general;
    ASSERT_EQ(rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &fill_mode, sizeof(fill_mode)),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &diag_type, sizeof(diag_type)),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_spmat_set_attribute(A, rocsparse_spmat_matrix_type, &mat_type, sizeof(mat_type)),
              rocsparse_status_success);

    std::vector<float> hx{2, 4, 8};
    const float         alpha = 1.0f;

    device_vector<float> x{hx};
    device_vector<float> y{std::vector<float>(static_cast<size_t>(m), 0.0f)};
    ASSERT_TRUE(x.ptr && y.ptr);

    rocsparse_dnvec_descr dx = nullptr, dy = nullptr;
    ASSERT_EQ(rocsparse_create_dnvec_descr(&dx, n, x.ptr, rocsparse_datatype_f32_r), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnvec_descr(&dy, m, y.ptr, rocsparse_datatype_f32_r), rocsparse_status_success);

    int64_t analysis_pivot = -2, solve_pivot = -2;
    run_ellsv(handle, A, dx, dy, alpha, &analysis_pivot, &solve_pivot);
    EXPECT_EQ(analysis_pivot, -1);
    EXPECT_EQ(solve_pivot, -1);

    int64_t             ref_pivot = -1;
    std::vector<float> expected
        = host_ellsv_lower(m, n, width, col_ind, val, hx, alpha, diag_type, &ref_pivot);
    EXPECT_EQ(ref_pivot, -1);
    // y = {2, 3, 7.25}: the garbage diagonal values must not leak in.
    expect_vectors_near(expected, to_host(y));

    EXPECT_EQ(rocsparse_destroy_dnvec_descr(dx), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnvec_descr(dy), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
}

// Lower triangular, non-unit diagonal, explicit zero on the diagonal of row 1
// (a numeric, not symbolic, zero pivot: the entry is present, just zero
// valued). The analysis stage only checks for a *missing* diagonal entry, so
// it must report no pivot; the compute stage must report the zero pivot and
// still produce the same "treat the zero as 1" numeric result the kernel
// computes. 3x3:
//   [2       ]   [y0]   [4 ]
//   [1  0    ] * [y1] = [10]
//   [  3  4  ]   [y2]   [20]
TEST_F(EllSv, zero_pivot)
{
    const int64_t m = 3, n = 3, width = 2;
    std::vector<int32_t> col_ind{0, 0, 1, -1, 1, 2};
    std::vector<float>   val{2, 1, 3, 0, 0, 4};

    device_vector<int32_t> d_col{col_ind};
    device_vector<float>   d_val{val};
    ASSERT_TRUE(d_col.ptr && d_val.ptr);

    rocsparse_spmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_ell_descr(
                  &A, m, n, d_col.ptr, d_val.ptr, width, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_datatype_f32_r),
              rocsparse_status_success);

    const rocsparse_fill_mode   fill_mode = rocsparse_fill_mode_lower;
    const rocsparse_diag_type   diag_type = rocsparse_diag_type_non_unit;
    const rocsparse_matrix_type mat_type  = rocsparse_matrix_type_general;
    ASSERT_EQ(rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &fill_mode, sizeof(fill_mode)),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &diag_type, sizeof(diag_type)),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_spmat_set_attribute(A, rocsparse_spmat_matrix_type, &mat_type, sizeof(mat_type)),
              rocsparse_status_success);

    std::vector<float> hx{4, 10, 20};
    const float         alpha = 1.0f;

    device_vector<float> x{hx};
    device_vector<float> y{std::vector<float>(static_cast<size_t>(m), 0.0f)};
    ASSERT_TRUE(x.ptr && y.ptr);

    rocsparse_dnvec_descr dx = nullptr, dy = nullptr;
    ASSERT_EQ(rocsparse_create_dnvec_descr(&dx, n, x.ptr, rocsparse_datatype_f32_r), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnvec_descr(&dy, m, y.ptr, rocsparse_datatype_f32_r), rocsparse_status_success);

    int64_t analysis_pivot = -2, solve_pivot = -2;
    run_ellsv(handle, A, dx, dy, alpha, &analysis_pivot, &solve_pivot);
    // The diagonal entry is present (just zero), so analysis finds no
    // *missing* diagonal; the zero is only caught numerically during solve.
    EXPECT_EQ(analysis_pivot, -1);
    EXPECT_EQ(solve_pivot, 1);

    int64_t             ref_pivot = -1;
    std::vector<float> expected
        = host_ellsv_lower(m, n, width, col_ind, val, hx, alpha, diag_type, &ref_pivot);
    EXPECT_EQ(ref_pivot, 1);
    // y = {2, 8, -1}, using the library's "treat a zero diagonal as 1" rule.
    expect_vectors_near(expected, to_host(y));

    EXPECT_EQ(rocsparse_destroy_dnvec_descr(dx), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnvec_descr(dy), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
}

#else // !ROCSPARSE_WITH_ELL_TRSV

TEST_F(EllSv, skipped_ell_trsv_not_built)
{
    GTEST_SKIP() << "ELL triangular solve support not built "
                    "(BUILD_WITH_ELL_TRSV=OFF / ROCSPARSE_WITH_ELL_TRSV not defined); "
                    "rocsparse_sptrsv on rocsparse_format_ell returns rocsparse_status_not_implemented.";
}

#endif // ROCSPARSE_WITH_ELL_TRSV

// ===========================================================================
// sellmv : Sliced ELL SpMV (rocsparse_v2_spmv on rocsparse_format_sell).
// ===========================================================================
class SellMv : public HandleTest
{
};

// Non-transpose, slice size 32 (<=128, small-slice kernel sellmvn_device).
// rocsparse_create_sell_descr requires sell_slice_size <= rows, so m = n =
// slice = 32: a single, exactly-full slice.
TEST_F(SellMv, none_small_slice)
{
    const int64_t m = 32, n = 32, slice = 32;
    std::vector<int32_t> offs, cols;
    std::vector<float>   vals;
    build_bidiagonal_sell(m, slice, offs, cols, vals);

    std::vector<float> hx = make_varying_x(n);
    std::vector<float> hy0(static_cast<size_t>(m), 0.0f);
    const float          alpha = 2.0f, beta = 0.0f;

    std::vector<float> expected
        = host_sellmv(m, n, slice, offs, cols, vals, hx, alpha, beta, hy0, rocsparse_operation_none);

    run_v2_spmv_sell(
        handle, rocsparse_operation_none, m, n, offs, cols, vals, slice, hx, alpha, beta, hy0, expected);
}

// Transpose, slice size 32 (small-slice kernel sellmvt_device), m = 40 is not
// a multiple of 32 so the second (and last) slice is only 8 rows out of 32 --
// a partial last slice. This is exactly the configuration where
// sellmvt_device's padding threads (local rows 8..31 of slice 1, global rows
// 40..63) read past the end of x; see the file comment at the top. The
// padding threads' loop range is empty, so the numeric result is unaffected.
TEST_F(SellMv, transpose_small_slice_partial_last_slice)
{
    const int64_t m = 40, n = 40, slice = 32;
    std::vector<int32_t> offs, cols;
    std::vector<float>   vals;
    build_bidiagonal_sell(m, slice, offs, cols, vals);

    std::vector<float> hx = make_varying_x(m);
    std::vector<float> hy0(static_cast<size_t>(n), 0.0f);
    const float          alpha = 1.0f, beta = 0.0f;

    std::vector<float> expected
        = host_sellmv(m, n, slice, offs, cols, vals, hx, alpha, beta, hy0, rocsparse_operation_transpose);

    run_v2_spmv_sell(handle,
                     rocsparse_operation_transpose,
                     m,
                     n,
                     offs,
                     cols,
                     vals,
                     slice,
                     hx,
                     alpha,
                     beta,
                     hy0,
                     expected);
}

// Non-transpose, slice size 256 (>128, large-slice kernel
// sellmvn_large_slice_device). rocsparse_create_sell_descr requires
// sell_slice_size <= rows, which rules out keeping this one under 64 rows (as
// the other cases are); m = n = slice = 256 is the smallest matrix that still
// reaches this kernel, one exactly-full slice.
TEST_F(SellMv, none_large_slice)
{
    const int64_t m = 256, n = 256, slice = 256;
    std::vector<int32_t> offs, cols;
    std::vector<float>   vals;
    build_bidiagonal_sell(m, slice, offs, cols, vals);

    std::vector<float> hx = make_varying_x(n);
    std::vector<float> hy0(static_cast<size_t>(m), 0.0f);
    const float          alpha = 1.0f, beta = 0.0f;

    std::vector<float> expected
        = host_sellmv(m, n, slice, offs, cols, vals, hx, alpha, beta, hy0, rocsparse_operation_none);

    run_v2_spmv_sell(
        handle, rocsparse_operation_none, m, n, offs, cols, vals, slice, hx, alpha, beta, hy0, expected);
}

// Transpose, slice size 256 (large-slice kernel sellmvt_large_slice_device).
// Same sell_slice_size <= rows constraint as none_large_slice above.
TEST_F(SellMv, transpose_large_slice)
{
    const int64_t m = 256, n = 256, slice = 256;
    std::vector<int32_t> offs, cols;
    std::vector<float>   vals;
    build_bidiagonal_sell(m, slice, offs, cols, vals);

    std::vector<float> hx = make_varying_x(m);
    std::vector<float> hy0(static_cast<size_t>(n), 0.0f);
    const float          alpha = 1.0f, beta = 0.0f;

    std::vector<float> expected
        = host_sellmv(m, n, slice, offs, cols, vals, hx, alpha, beta, hy0, rocsparse_operation_transpose);

    run_v2_spmv_sell(handle,
                     rocsparse_operation_transpose,
                     m,
                     n,
                     offs,
                     cols,
                     vals,
                     slice,
                     hx,
                     alpha,
                     beta,
                     hy0,
                     expected);
}

// Slice size 1: rocsparse_sellmv_impl dispatches this straight to the CSR
// fallback (rocsparse::csrmv_template on the slice-offsets-as-row-pointer
// data), bypassing sellmv_device.h entirely.
TEST_F(SellMv, slice_size_one_csr_fallback)
{
    const int64_t m = 5, n = 5, slice = 1;
    std::vector<int32_t> offs, cols;
    std::vector<float>   vals;
    build_bidiagonal_sell_csr_exact(m, offs, cols, vals);

    std::vector<float> hx = make_varying_x(n);
    std::vector<float> hy0(static_cast<size_t>(m), 0.0f);
    const float          alpha = 1.0f, beta = 0.0f;

    std::vector<float> expected
        = host_sellmv(m, n, slice, offs, cols, vals, hx, alpha, beta, hy0, rocsparse_operation_none);

    run_v2_spmv_sell(
        handle, rocsparse_operation_none, m, n, offs, cols, vals, slice, hx, alpha, beta, hy0, expected);
}

// beta != 0, so y is read as well as written and the kernels take their
// fma(beta, y[row], ...) path instead of the plain store. Non-transpose,
// slice size 32 (small-slice kernel). All constants are small integers, so
// the float arithmetic is exact.
TEST_F(SellMv, none_small_slice_nonzero_beta)
{
    const int64_t m = 32, n = 32, slice = 32;
    std::vector<int32_t> offs, cols;
    std::vector<float>   vals;
    build_bidiagonal_sell(m, slice, offs, cols, vals);

    std::vector<float> hx = make_varying_x(n);
    std::vector<float> hy0(static_cast<size_t>(m));
    for(int64_t i = 0; i < m; ++i)
    {
        hy0[static_cast<size_t>(i)] = static_cast<float>(2 * (i + 1));
    }
    const float alpha = 2.0f, beta = 3.0f;

    std::vector<float> expected
        = host_sellmv(m, n, slice, offs, cols, vals, hx, alpha, beta, hy0, rocsparse_operation_none);

    run_v2_spmv_sell(
        handle, rocsparse_operation_none, m, n, offs, cols, vals, slice, hx, alpha, beta, hy0, expected);
}
