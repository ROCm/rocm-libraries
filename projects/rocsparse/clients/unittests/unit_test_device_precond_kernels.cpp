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

#include <algorithm>
#include <cmath>
#include <complex>
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

        ScopedWavefrontSize(const ScopedWavefrontSize&) = delete;
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
    using real_of = std::conditional_t<
        std::is_same_v<T, double> || std::is_same_v<T, rocsparse_double_complex>,
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

    template <typename T>
    double part_re(T v)
    {
        return static_cast<double>(std::real(v));
    }

    template <typename T>
    double part_im(T v)
    {
        return static_cast<double>(std::imag(v));
    }

    template <typename T>
    bool is_zero(T v)
    {
        return part_re(v) == 0.0 && part_im(v) == 0.0;
    }

    template <typename T>
    T conj_of(T v)
    {
        if constexpr(std::is_floating_point_v<T>)
            return v;
        else
            return std::conj(v);
    }

    template <typename T>
    T from_real(double x)
    {
        if constexpr(std::is_same_v<T, float>)
            return static_cast<float>(x);
        else if constexpr(std::is_same_v<T, double>)
            return x;
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_float_complex(static_cast<float>(x), 0.0f);
        else
            return rocsparse_double_complex(x, 0.0);
    }

    template <typename T>
    double close_tol()
    {
        if constexpr(std::is_same_v<T, float> || std::is_same_v<T, rocsparse_float_complex>)
            return 1.0e-4;
        else
            return 1.0e-8;
    }

    template <typename T>
    void expect_close(const std::vector<T>& got, const std::vector<T>& want, double abs_tol = 0.0)
    {
        ASSERT_EQ(got.size(), want.size());
        const double tol = close_tol<T>();
        for(size_t i = 0; i < got.size(); ++i)
        {
            const double scale
                = std::max(1.0, std::max(std::abs(part_re(want[i])), std::abs(part_im(want[i]))));
            EXPECT_NEAR(part_re(got[i]), part_re(want[i]), tol * scale + abs_tol) << "re " << i;
            EXPECT_NEAR(part_im(got[i]), part_im(want[i]), tol * scale + abs_tol) << "im " << i;
        }
    }

    // Mirrors csrilu0_device_binsearch, including the break on an exact zero pivot
    // and the replacement of a tiny pivot when numeric boost is on.
    template <typename T>
    std::vector<T> host_csrilu0(const csr_pattern& p, std::vector<T> val, int boost_mode)
    {
        const int32_t        m = p.m;
        std::vector<int32_t> diag(static_cast<size_t>(m), -1);
        for(int32_t r = 0; r < m; ++r)
        {
            for(int32_t k = p.row_ptr[r]; k < p.row_ptr[r + 1]; ++k)
            {
                if(p.col_ind[k] == r)
                    diag[static_cast<size_t>(r)] = k;
            }
        }

        const bool   boost     = boost_mode != 0;
        const double boost_tol = (boost_mode == 2) ? 1.0e20 : 0.0;
        const T      boost_val = scalar<T>(1.0f);

        for(int32_t row = 0; row < m; ++row)
        {
            const int32_t row_begin = p.row_ptr[row];
            const int32_t row_end   = p.row_ptr[row + 1];
            const int32_t row_diag  = diag[static_cast<size_t>(row)];
            bool          stop      = false;
            for(int32_t j = row_begin; j < row_diag && !stop; ++j)
            {
                const int32_t local_col  = p.col_ind[j];
                const int32_t local_end  = p.row_ptr[local_col + 1];
                int32_t       local_diag = diag[static_cast<size_t>(local_col)];
                if(local_diag == -1)
                    local_diag = local_end - 1;
                const T diag_val = val[static_cast<size_t>(local_diag)];
                if(is_zero(diag_val))
                {
                    stop = true;
                    break;
                }
                const T local_val           = val[static_cast<size_t>(j)] / diag_val;
                val[static_cast<size_t>(j)] = local_val;
                for(int32_t k = local_diag + 1; k < local_end; ++k)
                {
                    const int32_t col_k = p.col_ind[k];
                    for(int32_t t = j + 1; t < row_end; ++t)
                    {
                        if(p.col_ind[t] == col_k)
                        {
                            val[static_cast<size_t>(t)] = val[static_cast<size_t>(t)]
                                                          - local_val * val[static_cast<size_t>(k)];
                            break;
                        }
                        if(p.col_ind[t] > col_k)
                            break;
                    }
                }
            }
            if(row_diag >= 0 && boost)
            {
                const T      pivot = val[static_cast<size_t>(row_diag)];
                const double mag
                    = std::sqrt(part_re(pivot) * part_re(pivot) + part_im(pivot) * part_im(pivot));
                if(mag <= boost_tol)
                    val[static_cast<size_t>(row_diag)] = boost_val;
            }
        }
        return val;
    }

    // Mirrors csrildlt0_device_binsearch. A missing diagonal is a zero pivot and
    // stops the rest of that row's strictly-lower updates; the diagonal is still published.
    template <typename T>
    std::vector<T> host_csrildlt0(const csr_pattern& p, std::vector<T> val, bool boost)
    {
        using real_t           = real_of<T>;
        const int32_t        m = p.m;
        std::vector<int32_t> diag(static_cast<size_t>(m), -1);
        for(int32_t r = 0; r < m; ++r)
        {
            for(int32_t k = p.row_ptr[r]; k < p.row_ptr[r + 1]; ++k)
            {
                if(p.col_ind[k] == r)
                    diag[static_cast<size_t>(r)] = k;
            }
        }

        const real_t boost_tol = boost ? static_cast<real_t>(1.0e20) : static_cast<real_t>(0);
        const real_t boost_val = static_cast<real_t>(1);

        for(int32_t row = 0; row < m; ++row)
        {
            const int32_t row_begin = p.row_ptr[row];
            const int32_t row_end   = p.row_ptr[row + 1];
            const int32_t row_diag  = diag[static_cast<size_t>(row)];
            real_t        diag_sum  = static_cast<real_t>(0);
            bool          stop      = false;
            for(int32_t j = row_begin; j < row_diag && !stop; ++j)
            {
                const int32_t local_col      = p.col_ind[j];
                const int32_t local_begin    = p.row_ptr[local_col];
                const int32_t local_diag_pos = diag[static_cast<size_t>(local_col)];
                const real_t  d_j
                    = (local_diag_pos >= 0)
                          ? static_cast<real_t>(part_re(val[static_cast<size_t>(local_diag_pos)]))
                          : static_cast<real_t>(0);
                if(d_j == static_cast<real_t>(0))
                {
                    stop = true;
                    break;
                }
                const real_t inv_d_j   = static_cast<real_t>(1) / d_j;
                T            local_sum = scalar<T>(0.0f);
                for(int32_t k = local_begin; k < local_diag_pos; ++k)
                {
                    const int32_t col_k = p.col_ind[k];
                    for(int32_t t = row_begin; t < row_end; ++t)
                    {
                        if(p.col_ind[t] == col_k)
                        {
                            const int32_t dk_pos = diag[static_cast<size_t>(col_k)];
                            const real_t  d_k    = (dk_pos >= 0) ? static_cast<real_t>(
                                                   part_re(val[static_cast<size_t>(dk_pos)]))
                                                                 : static_cast<real_t>(0);
                            const T       lk     = val[static_cast<size_t>(k)];
                            local_sum
                                = local_sum
                                  + val[static_cast<size_t>(t)]
                                        * (from_real<T>(static_cast<double>(d_k)) * conj_of(lk));
                            break;
                        }
                        if(p.col_ind[t] > col_k)
                            break;
                    }
                }
                const T local_val = (val[static_cast<size_t>(j)] - local_sum)
                                    * from_real<T>(static_cast<double>(inv_d_j));
                val[static_cast<size_t>(j)] = local_val;
                const real_t re             = static_cast<real_t>(part_re(local_val));
                const real_t im             = static_cast<real_t>(part_im(local_val));
                diag_sum                    = std::fma(re * re + im * im, d_j, diag_sum);
            }
            if(row_diag >= 0)
            {
                real_t d_i
                    = static_cast<real_t>(part_re(val[static_cast<size_t>(row_diag)])) - diag_sum;
                if(boost && std::abs(d_i) <= boost_tol)
                    d_i = boost_val;
                val[static_cast<size_t>(row_diag)] = from_real<T>(static_cast<double>(d_i));
            }
        }
        return val;
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
        const csr_pattern          pattern  = make_binsearch_pattern(fat);
        const std::vector<T>       host_val = pattern_values<T>(pattern, zero_first_diag);
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
        ASSERT_EQ(
            csrilu0_buffer_size<T>(handle, m, nnz, descr, dval, drow, dcol, info, &buffer_size),
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

        ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
                  rocsparse_status_success);
        rocsparse_int          pivot    = -2;
        const rocsparse_status pivot_st = rocsparse_csrilu0_zero_pivot(handle, info, &pivot);
        if(zero_first_diag)
        {
            EXPECT_EQ(pivot_st, rocsparse_status_zero_pivot);
            EXPECT_EQ(pivot, static_cast<rocsparse_int>(0));
        }
        else
        {
            // Row 5 of this pattern has no diagonal, so analysis records that pivot.
            EXPECT_EQ(pivot_st, rocsparse_status_zero_pivot);
            EXPECT_EQ(pivot, static_cast<rocsparse_int>(5));
            const auto got = to_host(dval);
            expect_close(got, host_csrilu0<T>(pattern, host_val, boost_mode));
            if(boost_mode == 0)
            {
                // Entries worked out by hand, so a mistake copied into both the
                // kernel and host_csrilu0 still fails. Diagonals are 4 and the
                // off-diagonals 0.1; the first rows only have a diagonal to the
                // right, so each lower entry is 0.1 / 4 and no row is updated.
                const auto expect_entry = [&](int32_t pos, double want) {
                    ASSERT_LT(static_cast<size_t>(pos), got.size());
                    EXPECT_NEAR(part_re(got[static_cast<size_t>(pos)]), want, 10 * close_tol<T>())
                        << "entry " << pos;
                    EXPECT_NEAR(part_im(got[static_cast<size_t>(pos)]), 0.0, close_tol<T>())
                        << "entry " << pos;
                };
                expect_entry(pattern.row_ptr[1], 0.025);
                expect_entry(pattern.row_ptr[1] + 1, 4.0);
                expect_entry(pattern.row_ptr[2], 0.025);
                expect_entry(pattern.row_ptr[3], 0.025);
                expect_entry(pattern.row_ptr[m - 1], 0.025);
                expect_entry(pattern.row_ptr[m] - 1, 4.0);
            }
        }

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
        ASSERT_EQ(rocsparse_spildlt0_descr_create(handle, &descr, nullptr),
                  rocsparse_status_success);

        const rocsparse_spildlt0_alg alg = rocsparse_spildlt0_alg_default;
        ASSERT_EQ(rocsparse_spildlt0_set_input(
                      handle, descr, rocsparse_spildlt0_input_alg, &alg, sizeof(alg), nullptr),
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
        ASSERT_EQ(
            rocsparse_spildlt0_buffer_size(
                handle, descr, A, A, rocsparse_spildlt0_stage_analysis, &buffer_size, nullptr),
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

        ASSERT_EQ(rocsparse_spildlt0_buffer_size(
                      handle, descr, A, A, rocsparse_spildlt0_stage_compute, &buffer_size, nullptr),
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

        const std::vector<T> factor = host_csrildlt0<T>(pattern, host_val, boost);
        // Wavefront 64 reduces the fat row in a different order than the host sum.
        expect_close(to_host(dval), factor, wavefront > 0 ? 2.0e-3 : 0.0);
        if(copy_diag)
        {
            const auto got_diag = to_host(ddiag.ptr, static_cast<size_t>(m));
            ASSERT_EQ(got_diag.size(), static_cast<size_t>(m));
            for(int32_t r = 0; r < pattern.m; ++r)
            {
                double want = 0.0;
                for(int32_t k = pattern.row_ptr[r]; k < pattern.row_ptr[r + 1]; ++k)
                {
                    if(pattern.col_ind[k] == r)
                        want = part_re(factor[static_cast<size_t>(k)]);
                }
                EXPECT_NEAR(
                    static_cast<double>(got_diag[static_cast<size_t>(r)]), want, close_tol<T>())
                    << "diag " << r;
            }
        }

        EXPECT_EQ(rocsparse_spildlt0_descr_destroy(handle, descr, nullptr),
                  rocsparse_status_success);
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
            return rocsparse_sbsric0(handle,
                                     dir,
                                     mb,
                                     nnzb,
                                     descr,
                                     val,
                                     row_ptr,
                                     col_ind,
                                     block_dim,
                                     info,
                                     solve,
                                     buffer);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dbsric0(handle,
                                     dir,
                                     mb,
                                     nnzb,
                                     descr,
                                     val,
                                     row_ptr,
                                     col_ind,
                                     block_dim,
                                     info,
                                     solve,
                                     buffer);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_cbsric0(handle,
                                     dir,
                                     mb,
                                     nnzb,
                                     descr,
                                     val,
                                     row_ptr,
                                     col_ind,
                                     block_dim,
                                     info,
                                     solve,
                                     buffer);
        else
            return rocsparse_zbsric0(handle,
                                     dir,
                                     mb,
                                     nnzb,
                                     descr,
                                     val,
                                     row_ptr,
                                     col_ind,
                                     block_dim,
                                     info,
                                     solve,
                                     buffer);
    }

    // Incomplete Cholesky on the scalar CSR expansion of the BSR pattern.
    // Diagonal blocks store their zeros, so those entries are part of the pattern.
    template <typename T>
    std::vector<T>
        host_bsric0(const bsr_pattern& p, rocsparse_direction dir, std::vector<T> bsr_val)
    {
        const int32_t        b = p.block_dim;
        const int32_t        n = p.mb * b;
        std::vector<int32_t> ptr(static_cast<size_t>(n) + 1, 0);
        std::vector<int32_t> col;
        std::vector<T>       val;
        col.reserve(bsr_val.size());
        val.reserve(bsr_val.size());
        for(int32_t r = 0; r < p.mb; ++r)
        {
            for(int32_t i = 0; i < b; ++i)
            {
                ptr[static_cast<size_t>(r * b + i)] = static_cast<int32_t>(col.size());
                for(int32_t k = p.row_ptr[r]; k < p.row_ptr[r + 1]; ++k)
                {
                    const int32_t c     = p.col_ind[k];
                    const T*      block = bsr_val.data() + static_cast<size_t>(k) * b * b;
                    for(int32_t j = 0; j < b; ++j)
                    {
                        const size_t idx = (dir == rocsparse_direction_row)
                                               ? static_cast<size_t>(i) * b + j
                                               : static_cast<size_t>(j) * b + i;
                        col.push_back(c * b + j);
                        val.push_back(block[idx]);
                    }
                }
            }
        }
        ptr[static_cast<size_t>(n)] = static_cast<int32_t>(col.size());

        std::vector<int32_t> diag(static_cast<size_t>(n), -1);
        for(int32_t i = 0; i < n; ++i)
        {
            for(int32_t k = ptr[static_cast<size_t>(i)]; k < ptr[static_cast<size_t>(i) + 1]; ++k)
            {
                if(col[static_cast<size_t>(k)] == i)
                    diag[static_cast<size_t>(i)] = k;
            }
        }

        for(int32_t i = 0; i < n; ++i)
        {
            const int32_t begin = ptr[static_cast<size_t>(i)];
            const int32_t dend  = diag[static_cast<size_t>(i)];
            for(int32_t j = begin; j < dend; ++j)
            {
                const int32_t colj   = col[static_cast<size_t>(j)];
                T             sum    = scalar<T>(0.0f);
                int32_t       a      = begin;
                int32_t       bk     = ptr[static_cast<size_t>(colj)];
                const int32_t bk_end = diag[static_cast<size_t>(colj)];
                while(a < j && bk < bk_end)
                {
                    if(col[static_cast<size_t>(a)] == col[static_cast<size_t>(bk)])
                    {
                        sum = sum
                              + val[static_cast<size_t>(a)] * conj_of(val[static_cast<size_t>(bk)]);
                        ++a;
                        ++bk;
                    }
                    else if(col[static_cast<size_t>(a)] < col[static_cast<size_t>(bk)])
                    {
                        ++a;
                    }
                    else
                    {
                        ++bk;
                    }
                }
                T piv = val[static_cast<size_t>(diag[static_cast<size_t>(colj)])];
                if(is_zero(piv))
                    piv = scalar<T>(1.0f);
                val[static_cast<size_t>(j)] = (val[static_cast<size_t>(j)] - sum) / piv;
            }
            T sum = scalar<T>(0.0f);
            for(int32_t j = begin; j < dend; ++j)
                sum = sum + val[static_cast<size_t>(j)] * conj_of(val[static_cast<size_t>(j)]);
            const T      remain = val[static_cast<size_t>(dend)] - sum;
            const double mag
                = std::sqrt(part_re(remain) * part_re(remain) + part_im(remain) * part_im(remain));
            val[static_cast<size_t>(dend)] = from_real<T>(std::sqrt(mag));
        }

        std::vector<T> out(bsr_val.size());
        for(int32_t r = 0; r < p.mb; ++r)
        {
            for(int32_t k = p.row_ptr[r]; k < p.row_ptr[r + 1]; ++k)
            {
                const int32_t c     = p.col_ind[k];
                T*            block = out.data() + static_cast<size_t>(k) * b * b;
                for(int32_t i = 0; i < b; ++i)
                {
                    const int32_t srow = r * b + i;
                    for(int32_t j = 0; j < b; ++j)
                    {
                        const int32_t scol  = c * b + j;
                        T             found = scalar<T>(0.0f);
                        for(int32_t t = ptr[static_cast<size_t>(srow)];
                            t < ptr[static_cast<size_t>(srow) + 1];
                            ++t)
                        {
                            if(col[static_cast<size_t>(t)] == scol)
                            {
                                found = val[static_cast<size_t>(t)];
                                break;
                            }
                        }
                        const size_t idx = (dir == rocsparse_direction_row)
                                               ? static_cast<size_t>(i) * b + j
                                               : static_cast<size_t>(j) * b + i;
                        block[idx]       = found;
                    }
                }
            }
        }
        return out;
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
        expect_close(to_host(dval), host_bsric0<T>(pattern, dir, host_val));

        EXPECT_EQ(rocsparse_bsric0_clear(handle, info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    void make_dense_csr(int                         m,
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

    // Exact ILU(0) of a sorted CSR matrix, stored in place (unit L below, U on
    // and above the diagonal). The iterative kernels converge to this.
    template <typename T>
    std::vector<T> host_exact_ilu0(const std::vector<rocsparse_int>& row_ptr,
                                   const std::vector<rocsparse_int>& col_ind,
                                   std::vector<T>                    val)
    {
        const size_t m = row_ptr.size() - 1;
        for(size_t i = 0; i < m; ++i)
        {
            for(rocsparse_int p = row_ptr[i]; p < row_ptr[i + 1] && col_ind[p] < (rocsparse_int)i;
                ++p)
            {
                const size_t  k = static_cast<size_t>(col_ind[p]);
                rocsparse_int d = -1;
                for(rocsparse_int q = row_ptr[k]; q < row_ptr[k + 1]; ++q)
                {
                    if(col_ind[q] == (rocsparse_int)k)
                        d = q;
                }
                val[p] = val[p] / val[d];
                for(rocsparse_int q = p + 1; q < row_ptr[i + 1]; ++q)
                {
                    for(rocsparse_int r = row_ptr[k]; r < row_ptr[k + 1]; ++r)
                    {
                        if(col_ind[r] == col_ind[q])
                            val[q] = val[q] - val[p] * val[r];
                    }
                }
            }
        }
        return val;
    }

    template <typename T>
    double max_abs_diff(const std::vector<T>& a, const std::vector<T>& b)
    {
        double e = 0.0;
        for(size_t i = 0; i < a.size(); ++i)
        {
            e = std::max(e, std::abs(part_re(a[i]) - part_re(b[i])));
            e = std::max(e, std::abs(part_im(a[i]) - part_im(b[i])));
        }
        return e;
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

        const rocsparse_int        nnz      = static_cast<rocsparse_int>(col_ind.size());
        const rocsparse_index_base base     = rocsparse_index_base_zero;
        rocsparse_int              nmaxiter = 2;

        device_vector<rocsparse_int> drow{row_ptr};
        device_vector<rocsparse_int> dcol{col_ind};
        device_vector<double>        dval{val};
        // ilu0 is also the starting iterate (L0 = lower(ilu0), U0 = upper(ilu0)).
        device_vector<double> dilu{val};
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr && dilu.ptr);

        size_t           buffer_size = 0;
        rocsparse_status st          = rocsparse_csritilu0_buffer_size(handle,
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

        const auto factor = to_host(dilu);
        ASSERT_EQ(factor.size(), val.size());
        for(size_t i = 0; i < factor.size(); ++i)
            EXPECT_TRUE(std::isfinite(factor[i])) << i;
        {
            // Two sweeps do not reach ILU(0) on the larger dense matrices, but
            // they must land closer to it than the starting matrix does, and
            // on the small ones they are very close.
            const auto   exact   = host_exact_ilu0<double>(row_ptr, col_ind, val);
            const double err_in  = max_abs_diff(val, exact);
            const double err_out = max_abs_diff(factor, exact);
            EXPECT_LT(err_out, err_in) << "m=" << m;
            if(m <= 8)
                EXPECT_LT(err_out, 1.0e-4) << "m=" << m;
        }

        if(query_history)
        {
            rocsparse_int       niter = 0;
            std::vector<double> hist(static_cast<size_t>(nmaxiter) * 2, 0.0);
            ASSERT_EQ(rocsparse_dcsritilu0_history(
                          handle, alg, &niter, hist.data(), buffer_size, buffer.ptr),
                      rocsparse_status_success);
            EXPECT_GT(niter, 0);
            EXPECT_LE(niter, nmaxiter);
            // The residual norms follow the niter correction norms.
            if(niter >= 2)
            {
                const double first = hist[static_cast<size_t>(niter)];
                const double last  = hist[static_cast<size_t>(niter) * 2 - 1];
                EXPECT_TRUE(std::isfinite(first));
                EXPECT_TRUE(std::isfinite(last));
                EXPECT_LT(last, first);
            }
        }
    }

    template <typename T>
    void run_itilu0_small(rocsparse_handle handle, rocsparse_itilu0_alg alg)
    {
        const int                  m      = 4;
        const rocsparse_index_base base   = rocsparse_index_base_zero;
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
        device_vector<T>             dilu{val};
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr && dilu.ptr);

        size_t buffer_size = 0;
        ASSERT_EQ(
            rocsparse_csritilu0_buffer_size(
                handle, alg, option, nmaxiter, m, nnz, drow, dcol, base, dt_of<T>(), &buffer_size),
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
        const auto factor = to_host(dilu);
        ASSERT_EQ(factor.size(), val.size());
        for(size_t i = 0; i < factor.size(); ++i)
        {
            EXPECT_TRUE(std::isfinite(part_re(factor[i]))) << "re " << i;
            EXPECT_TRUE(std::isfinite(part_im(factor[i]))) << "im " << i;
        }
        const auto   exact   = host_exact_ilu0<T>(row_ptr, col_ind, val);
        const double err_out = max_abs_diff(factor, exact);
        EXPECT_LT(err_out, max_abs_diff(val, exact));
        EXPECT_LT(err_out, 1.0e-4);
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
    run_csrildlt0<rocsparse_double_complex, int32_t, int32_t>(
        handle, false, false, false, false, 0);
    run_csrildlt0<rocsparse_double_complex, int64_t, int32_t>(
        handle, false, false, false, false, 0);
    run_csrildlt0<rocsparse_double_complex, int64_t, int64_t>(
        handle, false, false, false, false, 0);
}

TEST_F(PrecondCsrildlt0Binsearch, wave64)
{
    run_csrildlt0<float, int32_t, int32_t>(handle, false, false, false, false, 64);
    run_csrildlt0<double, int32_t, int32_t>(handle, false, false, false, false, 64);
    run_csrildlt0<rocsparse_float_complex, int32_t, int32_t>(
        handle, false, false, false, false, 64);
    run_csrildlt0<rocsparse_double_complex, int32_t, int32_t>(
        handle, false, false, false, false, 64);
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
