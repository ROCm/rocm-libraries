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

// Device unit tests that launch preconditioner kernels the public clients do
// not select on a wave32 GPU with the usual tiny matrices.
//
// csrilu0 binsearch:      wavefront 32 and max row nnz >= 512
//                         (library/src/precond/csrilu0/rocsparse_csrilu0_kernel_launch.cpp)
// csrildlt0 binsearch:    max row nnz > 512
//                         (library/src/precond/csrildlt0/rocsparse_csrildlt0_kernel_launch.cpp)
// bsric0 kernel 2_8:      wavefront != 32, block_dim <= 8, 32 < max nnzb <= 128
//                         (library/src/precond/bsric0/rocsparse_bsric0_kernel_launch.cpp)
//                         gfx1201 is wave32, so the numeric call temporarily reports
//                         wavefront 64. The 2_8 kernel launches an 8x8 thread block
//                         and does not use a 64-lane reduction.
// itilu0 sync / fusion / async inplace: kernel variant is selected from nnz/m
//                         (thresholds 2, 4, 8, 16, 32, 64, and above).
//
// Not reachable from this GPU, so these tests do not claim them:
//   * SLEEP=true instantiations (gfx908 with asic_rev < 2 only).
//   * csrilu0 / bsric0 int64 index dispatch. The public entry points pass
//     rocsparse_int, which is int32 in this build. csrildlt0 does take int64
//     through rocsparse_create_csr_descr, and those layouts are covered below.

#include "unit_test_utils.hpp"

#include "rocsparse_handle.hpp"
#include "rocsparse_mat_info.hpp"
#include "rocsparse_spmat_descr.hpp"

#include <type_traits>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    struct ScopedWavefrontSize
    {
        rocsparse_handle handle;
        int              saved;

        ScopedWavefrontSize(rocsparse_handle h, int wavefront)
            : handle(h)
            , saved(h->wavefront_size)
        {
            handle->wavefront_size = wavefront;
        }

        ~ScopedWavefrontSize()
        {
            handle->wavefront_size = saved;
        }

        ScopedWavefrontSize(const ScopedWavefrontSize&)            = delete;
        ScopedWavefrontSize& operator=(const ScopedWavefrontSize&) = delete;
    };

    struct csr_pattern
    {
        int32_t              m = 0;
        std::vector<int32_t> row_ptr;
        std::vector<int32_t> col_ind;
    };

    // Lower-triangular CSR whose last row holds `fat` entries (columns 1..fat).
    // fat >= 513 selects csrilu0 binsearch on wave32 (max nnz >= 512) and
    // csrildlt0 binsearch (max nnz > 512). fat >= 1024 also selects csrilu0
    // binsearch when the handle reports wavefront 64 (max nnz >= 1024).
    // Earlier rows plant a binary-search hit (column 1 is in the fat row), a
    // miss (column 0 is not), and a structural zero (row 5 has no diagonal).
    csr_pattern make_binsearch_pattern(int32_t fat = 513)
    {
        const int32_t                     m = fat + 1;
        std::vector<std::vector<int32_t>> rows(m);

        rows[0] = {0};
        rows[1] = {0, 1};
        rows[2] = {1, 2};
        rows[3] = {0, 3};
        for(int32_t r = 4; r < m - 1; ++r)
            rows[r] = {r};
        rows[5] = {4};

        for(int32_t c = m - fat; c < m; ++c)
            rows[m - 1].push_back(c);

        csr_pattern p;
        p.m = m;
        p.row_ptr.push_back(0);
        for(int32_t r = 0; r < m; ++r)
        {
            for(int32_t c : rows[r])
                p.col_ind.push_back(c);
            p.row_ptr.push_back(static_cast<int32_t>(p.col_ind.size()));
        }
        return p;
    }

    template <typename T>
    std::vector<T> pattern_values(const csr_pattern& p, bool zero_first_diag)
    {
        std::vector<T> v(p.col_ind.size());
        for(int32_t r = 0; r < p.m; ++r)
        {
            for(int32_t k = p.row_ptr[r]; k < p.row_ptr[r + 1]; ++k)
            {
                const bool diag = (p.col_ind[k] == r);
                if(diag && zero_first_diag && r == 0)
                    v[k] = scalar<T>(0.0f);
                else if(diag)
                    v[k] = scalar<T>(4.0f);
                else
                    v[k] = scalar<T>(0.1f);
            }
        }
        return v;
    }

    template <typename T>
    rocsparse_status csrilu0_buffer_size(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         rocsparse_int             nnz,
                                         const rocsparse_mat_descr descr,
                                         const T*                  val,
                                         const rocsparse_int*      row_ptr,
                                         const rocsparse_int*      col_ind,
                                         rocsparse_mat_info        info,
                                         size_t*                   buffer_size)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scsrilu0_buffer_size(
                handle, m, nnz, descr, val, row_ptr, col_ind, info, buffer_size);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcsrilu0_buffer_size(
                handle, m, nnz, descr, val, row_ptr, col_ind, info, buffer_size);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccsrilu0_buffer_size(
                handle, m, nnz, descr, val, row_ptr, col_ind, info, buffer_size);
        else
            return rocsparse_zcsrilu0_buffer_size(
                handle, m, nnz, descr, val, row_ptr, col_ind, info, buffer_size);
    }

    template <typename T>
    rocsparse_status csrilu0_analysis(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             nnz,
                                      const rocsparse_mat_descr descr,
                                      const T*                  val,
                                      const rocsparse_int*      row_ptr,
                                      const rocsparse_int*      col_ind,
                                      rocsparse_mat_info        info,
                                      void*                     buffer)
    {
        const auto analysis = rocsparse_analysis_policy_reuse;
        const auto solve    = rocsparse_solve_policy_auto;
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scsrilu0_analysis(
                handle, m, nnz, descr, val, row_ptr, col_ind, info, analysis, solve, buffer);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcsrilu0_analysis(
                handle, m, nnz, descr, val, row_ptr, col_ind, info, analysis, solve, buffer);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccsrilu0_analysis(
                handle, m, nnz, descr, val, row_ptr, col_ind, info, analysis, solve, buffer);
        else
            return rocsparse_zcsrilu0_analysis(
                handle, m, nnz, descr, val, row_ptr, col_ind, info, analysis, solve, buffer);
    }

    template <typename T>
    rocsparse_status csrilu0_compute(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             nnz,
                                     const rocsparse_mat_descr descr,
                                     T*                        val,
                                     const rocsparse_int*      row_ptr,
                                     const rocsparse_int*      col_ind,
                                     rocsparse_mat_info        info,
                                     void*                     buffer)
    {
        const auto solve = rocsparse_solve_policy_auto;
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scsrilu0(
                handle, m, nnz, descr, val, row_ptr, col_ind, info, solve, buffer);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcsrilu0(
                handle, m, nnz, descr, val, row_ptr, col_ind, info, solve, buffer);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccsrilu0(
                handle, m, nnz, descr, val, row_ptr, col_ind, info, solve, buffer);
        else
            return rocsparse_zcsrilu0(
                handle, m, nnz, descr, val, row_ptr, col_ind, info, solve, buffer);
    }

    template <typename T>
    using real_of = std::conditional_t<std::is_same_v<T, double> || std::is_same_v<T, rocsparse_double_complex>,
                                       double,
                                       float>;

    template <typename T>
    rocsparse_status csrilu0_boost(rocsparse_handle   handle,
                                   rocsparse_mat_info info,
                                   int                enable,
                                   const real_of<T>*  tol,
                                   const T*           val)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scsrilu0_numeric_boost(handle, info, enable, tol, val);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcsrilu0_numeric_boost(handle, info, enable, tol, val);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccsrilu0_numeric_boost(handle, info, enable, tol, val);
        else
            return rocsparse_zcsrilu0_numeric_boost(handle, info, enable, tol, val);
    }

    // boost_mode: 0 off, 1 on with a zero tolerance, 2 on with a huge tolerance.
    // singular_tol < 0 leaves the library default in place.
    // wavefront == 0 keeps the device wavefront; otherwise the numeric call reports that size.
    template <typename T>
    void run_csrilu0(rocsparse_handle handle,
                     bool             zero_first_diag,
                     int              boost_mode,
                     double           singular_tol,
                     int              wavefront,
                     int32_t          fat = 513)
    {
        const csr_pattern         pattern = make_binsearch_pattern(fat);
        const std::vector<T>      host_val = pattern_values<T>(pattern, zero_first_diag);
        std::vector<rocsparse_int> row(pattern.row_ptr.begin(), pattern.row_ptr.end());
        std::vector<rocsparse_int> col(pattern.col_ind.begin(), pattern.col_ind.end());
        const rocsparse_int        m   = pattern.m;
        const rocsparse_int        nnz = static_cast<rocsparse_int>(col.size());
        ASSERT_GE(row[m] - row[m - 1], fat);

        device_vector<rocsparse_int> drow{row};
        device_vector<rocsparse_int> dcol{col};
        device_vector<T>             dval{host_val};
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr);

        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_non_unit),
                  rocsparse_status_success);

        rocsparse_mat_info info = nullptr;
        ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

        size_t buffer_size = 0;
        ASSERT_EQ(csrilu0_buffer_size<T>(
                      handle, m, nnz, descr, dval, drow, dcol, info, &buffer_size),
                  rocsparse_status_success);
        ASSERT_GT(buffer_size, 0u);
        device_vector<char> buffer{buffer_size};
        ASSERT_TRUE(buffer.ptr);

        ASSERT_EQ(csrilu0_analysis<T>(handle, m, nnz, descr, dval, drow, dcol, info, buffer),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        if(singular_tol >= 0.0)
        {
            ASSERT_EQ(rocsparse_csrilu0_set_tolerance(handle, info, singular_tol),
                      rocsparse_status_success);
        }

        real_of<T> boost_tol = static_cast<real_of<T>>(boost_mode == 2 ? 1.0e20 : 0.0);
        T          boost_val = scalar<T>(1.0f);
        if(boost_mode != 0)
        {
            ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
                      rocsparse_status_success);
            ASSERT_EQ(csrilu0_boost<T>(handle, info, 1, &boost_tol, &boost_val),
                      rocsparse_status_success);
        }

        rocsparse_status st;
        if(wavefront > 0)
        {
            ScopedWavefrontSize guard(handle, wavefront);
            st = csrilu0_compute<T>(handle, m, nnz, descr, dval, drow, dcol, info, buffer);
        }
        else
        {
            st = csrilu0_compute<T>(handle, m, nnz, descr, dval, drow, dcol, info, buffer);
        }
        if(zero_first_diag)
        {
            EXPECT_TRUE(st == rocsparse_status_success || st == rocsparse_status_zero_pivot) << st;
        }
        else
        {
            ASSERT_EQ(st, rocsparse_status_success);
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        EXPECT_EQ(rocsparse_csrilu0_clear(handle, info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    template <typename I>
    std::vector<I> cast_index(const std::vector<int32_t>& src)
    {
        return std::vector<I>(src.begin(), src.end());
    }

    template <typename T, typename I, typename J>
    void run_csrildlt0(rocsparse_handle handle,
                       bool             zero_first_diag,
                       bool             boost,
                       bool             huge_singularity_tol,
                       bool             copy_diag,
                       int              wavefront)
    {
        const csr_pattern    pattern  = make_binsearch_pattern();
        const std::vector<T> host_val = pattern_values<T>(pattern, zero_first_diag);
        const std::vector<I> row      = cast_index<I>(pattern.row_ptr);
        const std::vector<J> col      = cast_index<J>(pattern.col_ind);
        const int64_t        m        = pattern.m;
        const int64_t        nnz      = static_cast<int64_t>(col.size());

        device_vector<I> drow{row};
        device_vector<J> dcol{col};
        device_vector<T> dval{host_val};
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr);

        using real_t = real_of<T>;
        device_vector<real_t> ddiag{copy_diag ? static_cast<size_t>(m) : size_t{1}};
        ASSERT_TRUE(ddiag.ptr);

        rocsparse_spmat_descr A = nullptr;
        ASSERT_EQ(rocsparse_create_csr_descr(&A,
                                             m,
                                             m,
                                             nnz,
                                             drow.ptr,
                                             dcol.ptr,
                                             dval.ptr,
                                             it_of<I>(),
                                             it_of<J>(),
                                             rocsparse_index_base_zero,
                                             dt_of<T>()),
                  rocsparse_status_success);

        rocsparse_spildlt0_descr descr = nullptr;
        ASSERT_EQ(rocsparse_spildlt0_descr_create(handle, &descr, nullptr), rocsparse_status_success);

        const rocsparse_spildlt0_alg alg = rocsparse_spildlt0_alg_default;
        ASSERT_EQ(rocsparse_spildlt0_set_input(handle,
                                               descr,
                                               rocsparse_spildlt0_input_alg,
                                               &alg,
                                               sizeof(alg),
                                               nullptr),
                  rocsparse_status_success);
        const rocsparse_datatype compute_dt = dt_of<T>();
        ASSERT_EQ(rocsparse_spildlt0_set_input(handle,
                                               descr,
                                               rocsparse_spildlt0_input_compute_datatype,
                                               &compute_dt,
                                               sizeof(compute_dt),
                                               nullptr),
                  rocsparse_status_success);
        const rocsparse_analysis_policy policy = rocsparse_analysis_policy_reuse;
        ASSERT_EQ(rocsparse_spildlt0_set_input(handle,
                                               descr,
                                               rocsparse_spildlt0_input_analysis_policy,
                                               &policy,
                                               sizeof(policy),
                                               nullptr),
                  rocsparse_status_success);

        const double singularity_tol = huge_singularity_tol ? 1.0e6 : 1.0e-12;
        ASSERT_EQ(rocsparse_spildlt0_set_input(handle,
                                               descr,
                                               rocsparse_spildlt0_input_singularity_tolerance,
                                               &singularity_tol,
                                               sizeof(void*),
                                               nullptr),
                  rocsparse_status_success);

        const int32_t boost_enable = 0;
        ASSERT_EQ(rocsparse_spildlt0_set_input(handle,
                                               descr,
                                               rocsparse_spildlt0_input_boost_enable,
                                               &boost_enable,
                                               sizeof(boost_enable),
                                               nullptr),
                  rocsparse_status_success);

        if(copy_diag)
        {
            void* diag_ptr = ddiag.ptr;
            ASSERT_EQ(rocsparse_spildlt0_set_input(handle,
                                                   descr,
                                                   rocsparse_spildlt0_input_diag,
                                                   &diag_ptr,
                                                   sizeof(void*),
                                                   nullptr),
                      rocsparse_status_success);
        }

        // The numeric kernel reads boost from the matrix info, not from the
        // descriptor. Poke the info the launch actually consults.
        real_t boost_tol = boost ? static_cast<real_t>(1.0e20) : static_cast<real_t>(0);
        T      boost_val = scalar<T>(1.0f);
        if(boost)
        {
            auto* numeric = A->info->get_boost();
            numeric->set_enable(1);
            numeric->set_tol_pointer_mode(rocsparse_pointer_mode_host);
            numeric->set_val_pointer_mode(rocsparse_pointer_mode_host);
            numeric->set_tol_datatype(std::is_same_v<real_t, double> ? rocsparse_datatype_f64_r
                                                                     : rocsparse_datatype_f32_r);
            numeric->set_tol(&boost_tol);
            numeric->set_val(&boost_val);
        }

        size_t buffer_size = 0;
        ASSERT_EQ(rocsparse_spildlt0_buffer_size(handle,
                                                 descr,
                                                 A,
                                                 A,
                                                 rocsparse_spildlt0_stage_analysis,
                                                 &buffer_size,
                                                 nullptr),
                  rocsparse_status_success);
        ASSERT_GT(buffer_size, 0u);
        device_vector<char> buffer{buffer_size};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(rocsparse_spildlt0(handle,
                                     descr,
                                     A,
                                     A,
                                     rocsparse_spildlt0_stage_analysis,
                                     buffer_size,
                                     buffer.ptr,
                                     nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        ASSERT_EQ(rocsparse_spildlt0_buffer_size(handle,
                                                 descr,
                                                 A,
                                                 A,
                                                 rocsparse_spildlt0_stage_compute,
                                                 &buffer_size,
                                                 nullptr),
                  rocsparse_status_success);
        device_vector<char> compute_buffer{buffer_size ? buffer_size : size_t{1}};
        ASSERT_TRUE(compute_buffer.ptr);

        rocsparse_status st;
        if(wavefront > 0)
        {
            ScopedWavefrontSize guard(handle, wavefront);
            st = rocsparse_spildlt0(handle,
                                    descr,
                                    A,
                                    A,
                                    rocsparse_spildlt0_stage_compute,
                                    buffer_size,
                                    compute_buffer.ptr,
                                    nullptr);
        }
        else
        {
            st = rocsparse_spildlt0(handle,
                                    descr,
                                    A,
                                    A,
                                    rocsparse_spildlt0_stage_compute,
                                    buffer_size,
                                    compute_buffer.ptr,
                                    nullptr);
        }
        ASSERT_EQ(st, rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        EXPECT_EQ(rocsparse_spildlt0_descr_destroy(handle, descr, nullptr), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    }

    struct bsr_pattern
    {
        int32_t              mb        = 0;
        int32_t              block_dim = 0;
        int32_t              nnzb      = 0;
        std::vector<int32_t> row_ptr;
        std::vector<int32_t> col_ind;
    };

    // Lower-triangular BSR. Block-rows 0..mb-2 are a single diagonal block.
    // The last block-row holds `fat_blocks` blocks, which is max_nnzb.
    bsr_pattern make_bsr(int32_t mb, int32_t block_dim, int32_t fat_blocks)
    {
        bsr_pattern p;
        p.mb        = mb;
        p.block_dim = block_dim;
        p.row_ptr.push_back(0);
        for(int32_t r = 0; r < mb - 1; ++r)
        {
            p.col_ind.push_back(r);
            p.row_ptr.push_back(static_cast<int32_t>(p.col_ind.size()));
        }
        for(int32_t c = mb - fat_blocks; c < mb; ++c)
            p.col_ind.push_back(c);
        p.row_ptr.push_back(static_cast<int32_t>(p.col_ind.size()));
        p.nnzb = static_cast<int32_t>(p.col_ind.size());
        return p;
    }

    template <typename T>
    std::vector<T> bsr_values(const bsr_pattern& p, rocsparse_direction dir)
    {
        const int32_t  b = p.block_dim;
        std::vector<T> v(static_cast<size_t>(p.nnzb) * static_cast<size_t>(b) * b, scalar<T>(0.0f));
        for(int32_t r = 0; r < p.mb; ++r)
        {
            for(int32_t k = p.row_ptr[r]; k < p.row_ptr[r + 1]; ++k)
            {
                const int32_t c     = p.col_ind[k];
                T*            block = v.data() + static_cast<size_t>(k) * b * b;
                for(int32_t i = 0; i < b; ++i)
                {
                    for(int32_t j = 0; j < b; ++j)
                    {
                        const size_t idx = (dir == rocsparse_direction_row)
                                               ? static_cast<size_t>(i) * b + j
                                               : static_cast<size_t>(j) * b + i;
                        if(c == r && i == j)
                            block[idx] = scalar<T>(4.0f);
                        else if(c != r)
                            block[idx] = scalar<T>(0.01f);
                    }
                }
            }
        }
        return v;
    }

    template <typename T>
    rocsparse_status bsric0_buffer_size(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_int             mb,
                                        rocsparse_int             nnzb,
                                        const rocsparse_mat_descr descr,
                                        const T*                  val,
                                        const rocsparse_int*      row_ptr,
                                        const rocsparse_int*      col_ind,
                                        rocsparse_int             block_dim,
                                        rocsparse_mat_info        info,
                                        size_t*                   buffer_size)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_sbsric0_buffer_size(
                handle, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, buffer_size);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dbsric0_buffer_size(
                handle, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, buffer_size);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_cbsric0_buffer_size(
                handle, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, buffer_size);
        else
            return rocsparse_zbsric0_buffer_size(
                handle, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, buffer_size);
    }

    template <typename T>
    rocsparse_status bsric0_analysis(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_int             mb,
                                     rocsparse_int             nnzb,
                                     const rocsparse_mat_descr descr,
                                     const T*                  val,
                                     const rocsparse_int*      row_ptr,
                                     const rocsparse_int*      col_ind,
                                     rocsparse_int             block_dim,
                                     rocsparse_mat_info        info,
                                     void*                     buffer)
    {
        const auto analysis = rocsparse_analysis_policy_reuse;
        const auto solve    = rocsparse_solve_policy_auto;
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_sbsric0_analysis(handle,
                                              dir,
                                              mb,
                                              nnzb,
                                              descr,
                                              val,
                                              row_ptr,
                                              col_ind,
                                              block_dim,
                                              info,
                                              analysis,
                                              solve,
                                              buffer);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dbsric0_analysis(handle,
                                              dir,
                                              mb,
                                              nnzb,
                                              descr,
                                              val,
                                              row_ptr,
                                              col_ind,
                                              block_dim,
                                              info,
                                              analysis,
                                              solve,
                                              buffer);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_cbsric0_analysis(handle,
                                              dir,
                                              mb,
                                              nnzb,
                                              descr,
                                              val,
                                              row_ptr,
                                              col_ind,
                                              block_dim,
                                              info,
                                              analysis,
                                              solve,
                                              buffer);
        else
            return rocsparse_zbsric0_analysis(handle,
                                              dir,
                                              mb,
                                              nnzb,
                                              descr,
                                              val,
                                              row_ptr,
                                              col_ind,
                                              block_dim,
                                              info,
                                              analysis,
                                              solve,
                                              buffer);
    }

    template <typename T>
    rocsparse_status bsric0_compute(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             mb,
                                    rocsparse_int             nnzb,
                                    const rocsparse_mat_descr descr,
                                    T*                        val,
                                    const rocsparse_int*      row_ptr,
                                    const rocsparse_int*      col_ind,
                                    rocsparse_int             block_dim,
                                    rocsparse_mat_info        info,
                                    void*                     buffer)
    {
        const auto solve = rocsparse_solve_policy_auto;
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_sbsric0(
                handle, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, solve, buffer);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dbsric0(
                handle, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, solve, buffer);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_cbsric0(
                handle, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, solve, buffer);
        else
            return rocsparse_zbsric0(
                handle, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, solve, buffer);
    }

    // fat_blocks in (32, 64] selects the MX_NNZB=64 instantiation; (64, 128] selects 128.
    template <typename T>
    void run_bsric0_2_8(rocsparse_handle    handle,
                        rocsparse_direction dir,
                        int32_t             mb,
                        int32_t             fat_blocks)
    {
        const int32_t     block_dim = 4;
        const bsr_pattern pattern   = make_bsr(mb, block_dim, fat_blocks);
        ASSERT_GT(fat_blocks, 32);
        ASSERT_LE(fat_blocks, 128);
        ASSERT_EQ(pattern.row_ptr[mb] - pattern.row_ptr[mb - 1], fat_blocks);

        const std::vector<T>       host_val = bsr_values<T>(pattern, dir);
        std::vector<rocsparse_int> row(pattern.row_ptr.begin(), pattern.row_ptr.end());
        std::vector<rocsparse_int> col(pattern.col_ind.begin(), pattern.col_ind.end());

        device_vector<rocsparse_int> drow{row};
        device_vector<rocsparse_int> dcol{col};
        device_vector<T>             dval{host_val};
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr);

        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_non_unit),
                  rocsparse_status_success);

        rocsparse_mat_info info = nullptr;
        ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

        size_t buffer_size = 0;
        ASSERT_EQ(bsric0_buffer_size<T>(handle,
                                        dir,
                                        pattern.mb,
                                        pattern.nnzb,
                                        descr,
                                        dval,
                                        drow,
                                        dcol,
                                        block_dim,
                                        info,
                                        &buffer_size),
                  rocsparse_status_success);
        ASSERT_GT(buffer_size, 0u);
        device_vector<char> buffer{buffer_size};
        ASSERT_TRUE(buffer.ptr);

        // Analysis must see the real wavefront. Only the numeric kernel selection
        // consults the overridden size.
        ASSERT_EQ(bsric0_analysis<T>(handle,
                                     dir,
                                     pattern.mb,
                                     pattern.nnzb,
                                     descr,
                                     dval,
                                     drow,
                                     dcol,
                                     block_dim,
                                     info,
                                     buffer),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        {
            ScopedWavefrontSize guard(handle, 64);
            ASSERT_EQ(bsric0_compute<T>(handle,
                                        dir,
                                        pattern.mb,
                                        pattern.nnzb,
                                        descr,
                                        dval,
                                        drow,
                                        dcol,
                                        block_dim,
                                        info,
                                        buffer),
                      rocsparse_status_success);
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        EXPECT_EQ(rocsparse_bsric0_clear(handle, info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    void make_dense_csr(int                       m,
                        std::vector<rocsparse_int>& row_ptr,
                        std::vector<rocsparse_int>& col_ind,
                        std::vector<double>&        val)
    {
        row_ptr.resize(static_cast<size_t>(m) + 1);
        col_ind.resize(static_cast<size_t>(m) * m);
        val.resize(col_ind.size());
        rocsparse_int nnz = 0;
        for(int r = 0; r < m; ++r)
        {
            row_ptr[r] = nnz;
            for(int c = 0; c < m; ++c)
            {
                col_ind[nnz] = c;
                val[nnz]     = (r == c) ? 4.0 : 0.05;
                ++nnz;
            }
        }
        row_ptr[m] = nnz;
    }

    // Tridiagonal, not a pure diagonal. sync_split splits into unit L and U;
    // a matrix with only stored diagonals leaves both triangles empty and
    // csxsldu returns internal_error. nnz/m stays in the <= 2 bucket.
    void make_tridiag_csr(int                         m,
                          std::vector<rocsparse_int>& row_ptr,
                          std::vector<rocsparse_int>& col_ind,
                          std::vector<double>&        val)
    {
        row_ptr.clear();
        col_ind.clear();
        val.clear();
        row_ptr.push_back(0);
        for(int r = 0; r < m; ++r)
        {
            if(r > 0)
            {
                col_ind.push_back(r - 1);
                val.push_back(0.1);
            }
            col_ind.push_back(r);
            val.push_back(4.0);
            if(r + 1 < m)
            {
                col_ind.push_back(r + 1);
                val.push_back(0.1);
            }
            row_ptr.push_back(static_cast<rocsparse_int>(col_ind.size()));
        }
    }

    template <typename T>
    rocsparse_status itilu0_compute_ex(rocsparse_handle     handle,
                                       rocsparse_itilu0_alg alg,
                                       rocsparse_int        option,
                                       rocsparse_int*       nmaxiter,
                                       rocsparse_int        nfreeiter,
                                       rocsparse_int        m,
                                       rocsparse_int        nnz,
                                       const rocsparse_int* row_ptr,
                                       const rocsparse_int* col_ind,
                                       const T*             val,
                                       T*                   ilu,
                                       rocsparse_index_base base,
                                       size_t               buffer_size,
                                       void*                buffer)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scsritilu0_compute_ex(handle,
                                                   alg,
                                                   option,
                                                   nmaxiter,
                                                   nfreeiter,
                                                   1.0e-6f,
                                                   m,
                                                   nnz,
                                                   row_ptr,
                                                   col_ind,
                                                   val,
                                                   ilu,
                                                   base,
                                                   buffer_size,
                                                   buffer);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcsritilu0_compute_ex(handle,
                                                   alg,
                                                   option,
                                                   nmaxiter,
                                                   nfreeiter,
                                                   1.0e-10,
                                                   m,
                                                   nnz,
                                                   row_ptr,
                                                   col_ind,
                                                   val,
                                                   ilu,
                                                   base,
                                                   buffer_size,
                                                   buffer);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccsritilu0_compute_ex(handle,
                                                   alg,
                                                   option,
                                                   nmaxiter,
                                                   nfreeiter,
                                                   1.0e-6f,
                                                   m,
                                                   nnz,
                                                   row_ptr,
                                                   col_ind,
                                                   val,
                                                   ilu,
                                                   base,
                                                   buffer_size,
                                                   buffer);
        else
            return rocsparse_zcsritilu0_compute_ex(handle,
                                                   alg,
                                                   option,
                                                   nmaxiter,
                                                   nfreeiter,
                                                   1.0e-10,
                                                   m,
                                                   nnz,
                                                   row_ptr,
                                                   col_ind,
                                                   val,
                                                   ilu,
                                                   base,
                                                   buffer_size,
                                                   buffer);
    }

    // dense == false builds a tridiagonal (integer mean nnz per row <= 2).
    // Otherwise a dense matrix, whose integer mean nnz per row equals m.
    void run_itilu0(rocsparse_handle     handle,
                    rocsparse_itilu0_alg alg,
                    int                  m,
                    bool                 dense,
                    rocsparse_int        option,
                    rocsparse_int        nfreeiter,
                    int                  wavefront,
                    bool                 query_history)
    {
        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<double>        val;
        if(dense)
            make_dense_csr(m, row_ptr, col_ind, val);
        else
            make_tridiag_csr(m, row_ptr, col_ind, val);

        const rocsparse_int        nnz  = static_cast<rocsparse_int>(col_ind.size());
        const rocsparse_index_base base = rocsparse_index_base_zero;
        rocsparse_int              nmaxiter = 2;

        device_vector<rocsparse_int> drow{row_ptr};
        device_vector<rocsparse_int> dcol{col_ind};
        device_vector<double>        dval{val};
        device_vector<double>        dilu{val.size()};
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr && dilu.ptr);

        size_t buffer_size = 0;
        rocsparse_status st = rocsparse_csritilu0_buffer_size(handle,
                                                              alg,
                                                              option,
                                                              nmaxiter,
                                                              m,
                                                              nnz,
                                                              drow,
                                                              dcol,
                                                              base,
                                                              rocsparse_datatype_f64_r,
                                                              &buffer_size);
        if(st == rocsparse_status_not_implemented)
            GTEST_SKIP() << "csritilu0 not implemented";
        ASSERT_EQ(st, rocsparse_status_success);
        ASSERT_GT(buffer_size, 0u);
        device_vector<char> buffer{buffer_size};
        ASSERT_TRUE(buffer.ptr);

        st = rocsparse_csritilu0_preprocess(handle,
                                            alg,
                                            option,
                                            nmaxiter,
                                            m,
                                            nnz,
                                            drow,
                                            dcol,
                                            base,
                                            rocsparse_datatype_f64_r,
                                            buffer_size,
                                            buffer.ptr);
        if(st == rocsparse_status_not_implemented)
            GTEST_SKIP() << "csritilu0 preprocess not implemented";
        ASSERT_EQ(st, rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        if(wavefront > 0)
        {
            ScopedWavefrontSize guard(handle, wavefront);
            st = itilu0_compute_ex<double>(handle,
                                           alg,
                                           option,
                                           &nmaxiter,
                                           nfreeiter,
                                           m,
                                           nnz,
                                           drow,
                                           dcol,
                                           dval,
                                           dilu,
                                           base,
                                           buffer_size,
                                           buffer.ptr);
        }
        else
        {
            st = itilu0_compute_ex<double>(handle,
                                           alg,
                                           option,
                                           &nmaxiter,
                                           nfreeiter,
                                           m,
                                           nnz,
                                           drow,
                                           dcol,
                                           dval,
                                           dilu,
                                           base,
                                           buffer_size,
                                           buffer.ptr);
        }
        ASSERT_EQ(st, rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        if(query_history)
        {
            rocsparse_int           niter = 0;
            std::vector<double>     hist(static_cast<size_t>(nmaxiter) * 2, 0.0);
            ASSERT_EQ(rocsparse_dcsritilu0_history(
                          handle, alg, &niter, hist.data(), buffer_size, buffer.ptr),
                      rocsparse_status_success);
        }
    }

    template <typename T>
    void run_itilu0_small(rocsparse_handle handle, rocsparse_itilu0_alg alg)
    {
        const int                  m = 4;
        const rocsparse_index_base base = rocsparse_index_base_zero;
        const rocsparse_int        option = rocsparse_itilu0_option_stopping_criteria
                                     | rocsparse_itilu0_option_compute_nrm_residual
                                     | rocsparse_itilu0_option_compute_nrm_correction
                                     | rocsparse_itilu0_option_convergence_history;
        rocsparse_int nmaxiter = 2;

        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<double>        val_d;
        make_tridiag_csr(m, row_ptr, col_ind, val_d);
        std::vector<T> val(val_d.size());
        for(size_t i = 0; i < val_d.size(); ++i)
            val[i] = scalar<T>(val_d[i]);
        const rocsparse_int nnz = static_cast<rocsparse_int>(col_ind.size());

        device_vector<rocsparse_int> drow{row_ptr};
        device_vector<rocsparse_int> dcol{col_ind};
        device_vector<T>             dval{val};
        device_vector<T>             dilu{(size_t)nnz};
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr && dilu.ptr);

        size_t buffer_size = 0;
        ASSERT_EQ(rocsparse_csritilu0_buffer_size(handle,
                                                  alg,
                                                  option,
                                                  nmaxiter,
                                                  m,
                                                  nnz,
                                                  drow,
                                                  dcol,
                                                  base,
                                                  dt_of<T>(),
                                                  &buffer_size),
                  rocsparse_status_success);
        device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(rocsparse_csritilu0_preprocess(handle,
                                                 alg,
                                                 option,
                                                 nmaxiter,
                                                 m,
                                                 nnz,
                                                 drow,
                                                 dcol,
                                                 base,
                                                 dt_of<T>(),
                                                 buffer_size,
                                                 buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(itilu0_compute_ex<T>(handle,
                                       alg,
                                       option,
                                       &nmaxiter,
                                       1,
                                       m,
                                       nnz,
                                       drow,
                                       dcol,
                                       dval,
                                       dilu,
                                       base,
                                       buffer_size,
                                       buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    }

    const rocsparse_int k_itilu_options = rocsparse_itilu0_option_stopping_criteria
                                          | rocsparse_itilu0_option_compute_nrm_residual
                                          | rocsparse_itilu0_option_compute_nrm_correction
                                          | rocsparse_itilu0_option_convergence_history;

    void run_itilu_buckets(rocsparse_handle handle, rocsparse_itilu0_alg alg)
    {
        // Tridiagonal mean nnz/row <= 2, then dense m so mean == m hits each threshold.
        run_itilu0(handle, alg, 4, false, k_itilu_options, 1, 0, true);
        const int dense_sizes[] = {4, 8, 16, 32, 64, 65};
        for(int m : dense_sizes)
            run_itilu0(handle, alg, m, true, k_itilu_options, 1, 0, false);
    }
}

class PrecondCsrilu0Binsearch : public HandleTest
{
};

TEST_F(PrecondCsrilu0Binsearch, f32_healthy)
{
    run_csrilu0<float>(handle, false, 0, -1.0, 0);
}

TEST_F(PrecondCsrilu0Binsearch, f64_healthy)
{
    run_csrilu0<double>(handle, false, 0, -1.0, 0);
}

TEST_F(PrecondCsrilu0Binsearch, f32c_healthy)
{
    run_csrilu0<rocsparse_float_complex>(handle, false, 0, -1.0, 0);
}

TEST_F(PrecondCsrilu0Binsearch, f64c_healthy)
{
    run_csrilu0<rocsparse_double_complex>(handle, false, 0, -1.0, 0);
}

TEST_F(PrecondCsrilu0Binsearch, f32_singular_and_zero)
{
    run_csrilu0<float>(handle, false, 0, 1.0e6, 0);
    run_csrilu0<float>(handle, true, 0, 0.0, 0);
}

TEST_F(PrecondCsrilu0Binsearch, f32_boost_both_sides)
{
    run_csrilu0<float>(handle, false, 1, -1.0, 0);
    run_csrilu0<float>(handle, false, 2, -1.0, 0);
}

TEST_F(PrecondCsrilu0Binsearch, f64_boost)
{
    run_csrilu0<double>(handle, false, 2, -1.0, 0);
}

TEST_F(PrecondCsrilu0Binsearch, wave64)
{
    // max nnz 1024 is the wave64 binsearch threshold. 513 would fall through to hash.
    run_csrilu0<float>(handle, false, 0, -1.0, 64, 1024);
    run_csrilu0<double>(handle, false, 0, -1.0, 64, 1024);
    run_csrilu0<rocsparse_float_complex>(handle, false, 0, -1.0, 64, 1024);
    run_csrilu0<rocsparse_double_complex>(handle, false, 0, -1.0, 64, 1024);
}

class PrecondCsrildlt0Binsearch : public HandleTest
{
};

TEST_F(PrecondCsrildlt0Binsearch, f32_layouts)
{
    run_csrildlt0<float, int32_t, int32_t>(handle, false, false, false, true, 0);
    run_csrildlt0<float, int64_t, int32_t>(handle, false, false, false, false, 0);
    run_csrildlt0<float, int64_t, int64_t>(handle, false, false, false, false, 0);
    run_csrildlt0<float, int32_t, int32_t>(handle, false, true, false, false, 0);
    run_csrildlt0<float, int32_t, int32_t>(handle, true, false, true, false, 0);
}

TEST_F(PrecondCsrildlt0Binsearch, f64_layouts)
{
    run_csrildlt0<double, int32_t, int32_t>(handle, false, false, false, false, 0);
    run_csrildlt0<double, int64_t, int32_t>(handle, false, false, false, false, 0);
    run_csrildlt0<double, int64_t, int64_t>(handle, false, false, false, false, 0);
}

TEST_F(PrecondCsrildlt0Binsearch, f32c_layouts)
{
    run_csrildlt0<rocsparse_float_complex, int32_t, int32_t>(handle, false, false, false, false, 0);
    run_csrildlt0<rocsparse_float_complex, int64_t, int32_t>(handle, false, false, false, false, 0);
    run_csrildlt0<rocsparse_float_complex, int64_t, int64_t>(handle, false, false, false, false, 0);
}

TEST_F(PrecondCsrildlt0Binsearch, f64c_layouts)
{
    run_csrildlt0<rocsparse_double_complex, int32_t, int32_t>(handle, false, false, false, false, 0);
    run_csrildlt0<rocsparse_double_complex, int64_t, int32_t>(handle, false, false, false, false, 0);
    run_csrildlt0<rocsparse_double_complex, int64_t, int64_t>(handle, false, false, false, false, 0);
}

TEST_F(PrecondCsrildlt0Binsearch, wave64)
{
    run_csrildlt0<float, int32_t, int32_t>(handle, false, false, false, false, 64);
    run_csrildlt0<double, int32_t, int32_t>(handle, false, false, false, false, 64);
    run_csrildlt0<rocsparse_float_complex, int32_t, int32_t>(handle, false, false, false, false, 64);
    run_csrildlt0<rocsparse_double_complex, int32_t, int32_t>(handle, false, false, false, false, 64);
    run_csrildlt0<double, int64_t, int32_t>(handle, false, false, false, false, 64);
    run_csrildlt0<double, int64_t, int64_t>(handle, false, false, false, false, 64);
}

class PrecondBsric0Kernel28 : public HandleTest
{
};

TEST_F(PrecondBsric0Kernel28, nnzb64_row_f32)
{
    run_bsric0_2_8<float>(handle, rocsparse_direction_row, 40, 40);
}

TEST_F(PrecondBsric0Kernel28, nnzb64_row_f64)
{
    run_bsric0_2_8<double>(handle, rocsparse_direction_row, 40, 40);
}

TEST_F(PrecondBsric0Kernel28, nnzb64_row_f32c)
{
    run_bsric0_2_8<rocsparse_float_complex>(handle, rocsparse_direction_row, 40, 40);
}

TEST_F(PrecondBsric0Kernel28, nnzb64_row_f64c)
{
    run_bsric0_2_8<rocsparse_double_complex>(handle, rocsparse_direction_row, 40, 40);
}

TEST_F(PrecondBsric0Kernel28, nnzb64_column)
{
    run_bsric0_2_8<float>(handle, rocsparse_direction_column, 40, 40);
}

TEST_F(PrecondBsric0Kernel28, nnzb128_row)
{
    run_bsric0_2_8<float>(handle, rocsparse_direction_row, 80, 70);
}

class PrecondItilu0Device : public HandleTest
{
};

TEST_F(PrecondItilu0Device, sync_split_mean_buckets)
{
    run_itilu_buckets(handle, rocsparse_itilu0_alg_sync_split);
}

TEST_F(PrecondItilu0Device, sync_fusion_mean_buckets)
{
    run_itilu_buckets(handle, rocsparse_itilu0_alg_sync_split_fusion);
}

TEST_F(PrecondItilu0Device, async_inplace_mean_buckets)
{
    run_itilu_buckets(handle, rocsparse_itilu0_alg_async_inplace);
}

TEST_F(PrecondItilu0Device, async_inplace_coo)
{
    const rocsparse_int option = k_itilu_options | rocsparse_itilu0_option_coo_format;
    run_itilu0(handle, rocsparse_itilu0_alg_async_inplace, 4, false, option, 1, 0, true);
    run_itilu0(handle, rocsparse_itilu0_alg_async_inplace, 8, true, option, 1, 0, false);
}

TEST_F(PrecondItilu0Device, sync_split_verbose)
{
    const rocsparse_int option = k_itilu_options | rocsparse_itilu0_option_verbose;
    run_itilu0(handle, rocsparse_itilu0_alg_sync_split, 4, true, option, 1, 0, true);
}

TEST_F(PrecondItilu0Device, sync_split_complex)
{
    run_itilu0_small<rocsparse_float_complex>(handle, rocsparse_itilu0_alg_sync_split);
    run_itilu0_small<rocsparse_double_complex>(handle, rocsparse_itilu0_alg_sync_split);
    run_itilu0_small<float>(handle, rocsparse_itilu0_alg_sync_split_fusion);
}

TEST_F(PrecondItilu0Device, sync_split_wave64)
{
    run_itilu0(handle, rocsparse_itilu0_alg_sync_split, 65, true, k_itilu_options, 0, 64, false);
}

TEST_F(PrecondItilu0Device, empty_matrix_buffer_size)
{
    device_vector<rocsparse_int> row{std::vector<rocsparse_int>{0}};
    device_vector<rocsparse_int> col{std::vector<rocsparse_int>{0}};
    ASSERT_TRUE(row.ptr && col.ptr);
    size_t     buffer_size = 99;
    const auto st          = rocsparse_csritilu0_buffer_size(handle,
                                                    rocsparse_itilu0_alg_sync_split,
                                                    0,
                                                    1,
                                                    0,
                                                    0,
                                                    row,
                                                    col,
                                                    rocsparse_index_base_zero,
                                                    rocsparse_datatype_f64_r,
                                                    &buffer_size);
    EXPECT_EQ(st, rocsparse_status_success);
    EXPECT_EQ(buffer_size, 0u);
}
