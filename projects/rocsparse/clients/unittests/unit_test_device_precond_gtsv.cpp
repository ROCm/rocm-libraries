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

// Tridiagonal solvers: gtsv_no_pivot, gtsv_no_pivot_strided_batch, gtsv and
// gtsv_interleaved_batch. The matrix size selects the kernel: the Thomas /
// Cramer kernels for m <= 16, PCR kernels in a wavefront or shared memory up
// to 1024 rows, the SPIKE decomposition above (with a recursion once the
// reduced system is itself larger than 1024 rows), and, for gtsv, a SPIKE
// grid whose size follows m. Each solve is checked against a known solution.

#include "unit_test_device_precond_common.hpp"

#include <string>

using namespace precond_ut;

namespace
{
    // Typed entry points: the argument lists follow the public routines.
    template <typename T, typename... A>
    rocsparse_status gtsv_no_pivot_buffer_size(A&&... args)
    {
        return PRECOND_CALL(T, gtsv_no_pivot_buffer_size, std::forward<A>(args)...);
    }

    template <typename T, typename... A>
    rocsparse_status gtsv_no_pivot(A&&... args)
    {
        return PRECOND_CALL(T, gtsv_no_pivot, std::forward<A>(args)...);
    }

    template <typename T, typename... A>
    rocsparse_status gtsv_pivot_buffer_size(A&&... args)
    {
        return PRECOND_CALL(T, gtsv_buffer_size, std::forward<A>(args)...);
    }

    template <typename T, typename... A>
    rocsparse_status gtsv_pivot(A&&... args)
    {
        return PRECOND_CALL(T, gtsv, std::forward<A>(args)...);
    }

    template <typename T, typename... A>
    rocsparse_status strided_buffer_size(A&&... args)
    {
        return PRECOND_CALL(T, gtsv_no_pivot_strided_batch_buffer_size, std::forward<A>(args)...);
    }

    template <typename T, typename... A>
    rocsparse_status strided_solve(A&&... args)
    {
        return PRECOND_CALL(T, gtsv_no_pivot_strided_batch, std::forward<A>(args)...);
    }

    template <typename T, typename... A>
    rocsparse_status interleaved_buffer_size(A&&... args)
    {
        return PRECOND_CALL(T, gtsv_interleaved_batch_buffer_size, std::forward<A>(args)...);
    }

    template <typename T, typename... A>
    rocsparse_status interleaved_solve(A&&... args)
    {
        return PRECOND_CALL(T, gtsv_interleaved_batch, std::forward<A>(args)...);
    }

    // One tridiagonal system with a known solution. By default the matrix is
    // strictly diagonally dominant, so solving without pivoting is stable.
    //
    // With `swaps`, every third row (i % 3 == 1) has |dl| (3.0 .. 3.4) larger than
    // |d| (0.5 .. 0.7) and |du| (0.8 .. 1.1): partial pivoting exchanges that row
    // with the one above it. The sub-diagonal of those rows has the opposite sign of
    // the others; this pattern was chosen because its LAPACK partial-pivoting
    // solution reproduces the known solution to float accuracy (relative error
    // 1e-6) for every size used by the tests, i.e. the system is well conditioned.
    template <typename T>
    struct system_t
    {
        std::vector<T> dl, d, du;
        std::vector<T> x; // known solution, m * n (column-major, leading dimension m)
        std::vector<T> b; // right-hand side A * x
        int32_t        swap_rows = 0; // rows with |dl| > |d|
    };

    template <typename T>
    system_t<T> make_system(int32_t m, int32_t n, int seed, bool swaps = false)
    {
        system_t<T> s;
        s.dl.resize(m);
        s.d.resize(m);
        s.du.resize(m);
        for(int32_t i = 0; i < m; ++i)
        {
            const double im_part = is_cplx<T> ? 0.1 * (((i + seed) % 3) - 1) : 0.0;
            double       dl      = -(1.0 + 0.1 * ((i * 7 + seed) % 5));
            double       d       = 5.0 + 0.2 * (i % 3);
            if(swaps && i > 0 && i % 3 == 1)
            {
                dl = 3.0 + 0.1 * ((i * 7 + seed) % 5);
                d  = 0.5 + 0.1 * (i % 3);
                ++s.swap_rows;
            }
            s.dl[i] = (i == 0) ? make_value<T>(0.0) : make_value<T>(dl, im_part);
            s.du[i] = (i == m - 1) ? make_value<T>(0.0)
                                   : make_value<T>(-(0.8 + 0.1 * ((i * 3 + seed) % 4)), -im_part);
            s.d[i]  = make_value<T>(d, im_part);
        }
        s.x.resize(static_cast<size_t>(m) * n);
        s.b.resize(static_cast<size_t>(m) * n);
        for(int32_t c = 0; c < n; ++c)
        {
            for(int32_t i = 0; i < m; ++i)
            {
                s.x[static_cast<size_t>(c) * m + i]
                    = make_value<T>(1.0 + 0.01 * ((i * 13 + c * 7 + seed) % 17),
                                    is_cplx<T> ? 0.02 * ((i + c) % 5) : 0.0);
            }
        }
        for(int32_t c = 0; c < n; ++c)
        {
            for(int32_t i = 0; i < m; ++i)
            {
                std::complex<double> v = to_cd(s.d[i]) * to_cd(s.x[static_cast<size_t>(c) * m + i]);
                if(i > 0)
                    v += to_cd(s.dl[i]) * to_cd(s.x[static_cast<size_t>(c) * m + i - 1]);
                if(i + 1 < m)
                    v += to_cd(s.du[i]) * to_cd(s.x[static_cast<size_t>(c) * m + i + 1]);
                s.b[static_cast<size_t>(c) * m + i] = from_cd<T>(v);
            }
        }
        return s;
    }

    // Absolute error of the solution entries (about 1.0 .. 1.2), and residual relative to
    // max|b|: about ten times the largest measured on gfx1201 over all tests of this
    // file. Solution error: float/cfloat 1.31e-6 (the systems that force row swaps),
    // double/cdouble 2.7e-15. Residual / max|b|: float/cfloat 1.36e-6, double/cdouble
    // 2.25e-15.
    template <typename T>
    double solve_tolerance()
    {
        return is_single_precision<T> ? 1.5e-5 : 3.0e-14;
    }

    template <typename T>
    void expect_solution(const std::vector<T>& got,
                         const std::vector<T>& want,
                         const std::string&    what)
    {
        ASSERT_EQ(got.size(), want.size()) << what;
        const double tol = solve_tolerance<T>();
        double       err = 0.0;
        for(size_t i = 0; i < got.size(); ++i)
        {
            err = std::max(err, std::abs(re(got[i]) - re(want[i])));
            err = std::max(err, std::abs(im(got[i]) - im(want[i])));
        }
        EXPECT_LT(err, tol) << what;
    }

    enum class Solver
    {
        no_pivot,
        pivot
    };

    // max_i |b_i - (A x)_i| over every right-hand side, relative to max |b|: the
    // residual check does not depend on the known solution.
    template <typename T>
    void expect_small_solve_residual(const system_t<T>&    s,
                                     const std::vector<T>& sol,
                                     int32_t               m,
                                     int32_t               n)
    {
        double res = 0.0, norm_b = 0.0;
        for(int32_t c = 0; c < n; ++c)
        {
            for(int32_t i = 0; i < m; ++i)
            {
                const size_t         k  = static_cast<size_t>(c) * m + i;
                std::complex<double> ax = to_cd(s.d[i]) * to_cd(sol[k]);
                if(i > 0)
                    ax += to_cd(s.dl[i]) * to_cd(sol[k - 1]);
                if(i + 1 < m)
                    ax += to_cd(s.du[i]) * to_cd(sol[k + 1]);
                res    = std::max(res, std::abs(to_cd(s.b[k]) - ax));
                norm_b = std::max(norm_b, std::abs(to_cd(s.b[k])));
            }
        }
        EXPECT_LE(res, solve_tolerance<T>() * norm_b);
    }

    // Multi-RHS solve through gtsv_no_pivot or gtsv; ldb >= m. `swaps` makes the
    // systems need row exchanges (see make_system).
    template <Solver S, typename T>
    void run_multi_rhs(
        rocsparse_handle handle, int32_t m, int32_t n, int32_t ldb_extra = 0, bool swaps = false)
    {
        SCOPED_TRACE(std::string(S == Solver::no_pivot ? "no_pivot" : "pivot") + " m="
                     + std::to_string(m) + " n=" + std::to_string(n) + (swaps ? " swaps" : ""));
        const system_t<T> s = make_system<T>(m, n, m % 7, swaps);
        if(swaps)
            ASSERT_GT(s.swap_rows, 0);
        const int32_t  ldb = m + ldb_extra;
        std::vector<T> hb(static_cast<size_t>(ldb) * n, make_value<T>(0.0));
        for(int32_t c = 0; c < n; ++c)
            for(int32_t i = 0; i < m; ++i)
                hb[static_cast<size_t>(c) * ldb + i] = s.b[static_cast<size_t>(c) * m + i];

        device_vector<T> ddl{s.dl}, dd{s.d}, ddu{s.du}, dB{hb};
        ASSERT_TRUE(ddl.ptr && dd.ptr && ddu.ptr && dB.ptr);
        size_t size = 0;
        if constexpr(S == Solver::no_pivot)
            ASSERT_EQ(gtsv_no_pivot_buffer_size<T>(handle, m, n, ddl, dd, ddu, dB, ldb, &size),
                      rocsparse_status_success);
        else
            ASSERT_EQ(gtsv_pivot_buffer_size<T>(handle, m, n, ddl, dd, ddu, dB, ldb, &size),
                      rocsparse_status_success);
        device_vector<char> buffer{size ? size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        if constexpr(S == Solver::no_pivot)
            ASSERT_EQ(gtsv_no_pivot<T>(handle, m, n, ddl, dd, ddu, dB, ldb, buffer.ptr),
                      rocsparse_status_success);
        else
            ASSERT_EQ(gtsv_pivot<T>(handle, m, n, ddl, dd, ddu, dB, ldb, buffer.ptr),
                      rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const auto     got = to_host(dB);
        std::vector<T> sol(static_cast<size_t>(m) * n);
        for(int32_t c = 0; c < n; ++c)
            for(int32_t i = 0; i < m; ++i)
                sol[static_cast<size_t>(c) * m + i] = got[static_cast<size_t>(c) * ldb + i];
        expect_solution(sol, s.x, "solution");
        expect_small_solve_residual(s, sol, m, n);
        // The padding rows between m and ldb are left untouched.
        for(int32_t c = 0; c < n; ++c)
            for(int32_t i = m; i < ldb; ++i)
            {
                EXPECT_EQ(re(got[static_cast<size_t>(c) * ldb + i]), 0.0);
                EXPECT_EQ(im(got[static_cast<size_t>(c) * ldb + i]), 0.0);
            }
    }

    // gtsv_no_pivot_strided_batch: batch_count systems of order m, batch_stride apart.
    template <typename T>
    void run_strided(rocsparse_handle handle, int32_t m, int32_t batch_count, int32_t extra)
    {
        SCOPED_TRACE("strided m=" + std::to_string(m) + " batch=" + std::to_string(batch_count));
        const int32_t  stride = m + extra;
        std::vector<T> dl(static_cast<size_t>(stride) * batch_count, make_value<T>(0.0));
        std::vector<T> d = dl, du = dl, b = dl;
        std::vector<std::vector<T>> want;
        for(int32_t k = 0; k < batch_count; ++k)
        {
            const system_t<T> s = make_system<T>(m, 1, k + 1);
            for(int32_t i = 0; i < m; ++i)
            {
                dl[static_cast<size_t>(k) * stride + i] = s.dl[i];
                d[static_cast<size_t>(k) * stride + i]  = s.d[i];
                du[static_cast<size_t>(k) * stride + i] = s.du[i];
                b[static_cast<size_t>(k) * stride + i]  = s.b[i];
            }
            want.push_back(s.x);
        }
        device_vector<T> ddl{dl}, dd{d}, ddu{du}, dx{b};
        ASSERT_TRUE(ddl.ptr && dd.ptr && ddu.ptr && dx.ptr);
        size_t size = 0;
        ASSERT_EQ(strided_buffer_size<T>(handle, m, ddl, dd, ddu, dx, batch_count, stride, &size),
                  rocsparse_status_success);
        device_vector<char> buffer{size ? size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(strided_solve<T>(handle, m, ddl, dd, ddu, dx, batch_count, stride, buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        const auto got = to_host(dx);
        for(int32_t k = 0; k < batch_count; ++k)
        {
            const std::vector<T> sol(got.begin() + static_cast<size_t>(k) * stride,
                                     got.begin() + static_cast<size_t>(k) * stride + m);
            expect_solution(sol, want[k], "batch " + std::to_string(k));
            for(int32_t i = m; i < stride; ++i)
            {
                EXPECT_EQ(re(got[static_cast<size_t>(k) * stride + i]), 0.0);
                EXPECT_EQ(im(got[static_cast<size_t>(k) * stride + i]), 0.0);
            }
        }
    }

    // Sizes of the Thomas / Cramer kernels, the wavefront and shared-memory PCR
    // kernels (each bucket boundary and the value just above it).
    const int32_t k_small_sizes[]
        = {2,  3,  4,  5,  6,  7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  24,   31,
           32, 33, 63, 64, 65, 100, 128, 129, 255, 256, 257, 400, 511, 512, 513, 800, 1023, 1024};

    // Above 1024 rows the SPIKE decomposition runs; num_spikes = 2 * ceil(m / 256) picks the
    // reduced solver (4..1024 spikes) and a reduced system above 1024 recurses.
    const int32_t k_medium_sizes[] = {1025, 2048, 3000, 5000, 9000, 20000, 40000, 70000, 135000};
}

class PrecondTridiagonal : public HandleTest
{
};

TEST_F(PrecondTridiagonal, no_pivot_small_sizes)
{
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        for(const int32_t m : k_small_sizes)
        {
            run_multi_rhs<Solver::no_pivot, T>(handle, m, 1);
            run_multi_rhs<Solver::no_pivot, T>(handle, m, 11);
        }
    });
}

TEST_F(PrecondTridiagonal, no_pivot_padded_leading_dimension)
{
    run_multi_rhs<Solver::no_pivot, double>(handle, 20, 3, 5);
    run_multi_rhs<Solver::no_pivot, float>(handle, 700, 4, 9);
    run_multi_rhs<Solver::no_pivot, double>(handle, 3000, 3, 17);
}

TEST_F(PrecondTridiagonal, no_pivot_medium_sizes)
{
    for(const int32_t m : k_medium_sizes)
    {
        run_multi_rhs<Solver::no_pivot, float>(handle, m, 2);
        run_multi_rhs<Solver::no_pivot, double>(handle, m, 1);
    }
    run_multi_rhs<Solver::no_pivot, rocsparse_float_complex>(handle, 3000, 3);
    run_multi_rhs<Solver::no_pivot, rocsparse_double_complex>(handle, 5000, 2);
    run_multi_rhs<Solver::no_pivot, rocsparse_double_complex>(handle, 135000, 1);
}

TEST_F(PrecondTridiagonal, pivot_spike_grid_sizes)
{
    // gtsv rounds the SPIKE grid up to a power of two: 16 .. 256 blocks of 256 threads.
    for(const int32_t m : {8000, 16000, 30000, 60000, 120000})
    {
        run_multi_rhs<Solver::pivot, double>(handle, m, 2);
    }
    // Systems that need row exchanges, one per value type.
    run_multi_rhs<Solver::pivot, double>(handle, 16000, 1, 0, true);
    run_multi_rhs<Solver::pivot, float>(handle, 30000, 1, 0, true);
    run_multi_rhs<Solver::pivot, rocsparse_float_complex>(handle, 8000, 1, 0, true);
    run_multi_rhs<Solver::pivot, rocsparse_double_complex>(handle, 9000, 1, 0, true);
}

// Above 262144 rows the SPIKE grid would exceed 512 blocks: the block dimension doubles.
TEST_F(PrecondTridiagonal, pivot_large_block_dimension)
{
    run_multi_rhs<Solver::pivot, double>(handle, 270000, 1, 0, true);
    run_multi_rhs<Solver::pivot, float>(handle, 700000, 1);
}

TEST_F(PrecondTridiagonal, strided_batch_small_sizes)
{
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        for(const int32_t m : k_small_sizes)
        {
            run_strided<T>(handle, m, 3, 0);
        }
        run_strided<T>(handle, 17, 5, 4);
    });
}

TEST_F(PrecondTridiagonal, strided_batch_medium_sizes)
{
    for(const int32_t m : k_medium_sizes)
    {
        run_strided<float>(handle, m, 2, 3);
        run_strided<double>(handle, m, 1, 0);
    }
    run_strided<rocsparse_float_complex>(handle, 3000, 2, 0);
    run_strided<rocsparse_double_complex>(handle, 135000, 1, 0);
}

TEST_F(PrecondTridiagonal, no_pivot_argument_validation)
{
    const int32_t          m = 8;
    const system_t<double> s = make_system<double>(m, 1, 1);
    device_vector<double>  ddl{s.dl}, dd{s.d}, ddu{s.du}, dB{s.b};
    ASSERT_TRUE(ddl.ptr && dd.ptr && ddu.ptr && dB.ptr);
    size_t size = 7;

    EXPECT_EQ(gtsv_no_pivot_buffer_size<double>(handle, 1, 1, ddl, dd, ddu, dB, 8, &size),
              rocsparse_status_invalid_size);
    EXPECT_EQ(gtsv_no_pivot_buffer_size<double>(handle, m, -1, ddl, dd, ddu, dB, 8, &size),
              rocsparse_status_invalid_size);
    EXPECT_EQ(gtsv_no_pivot_buffer_size<double>(handle, m, 1, ddl, dd, ddu, dB, 4, &size),
              rocsparse_status_invalid_size);
    EXPECT_EQ(gtsv_no_pivot_buffer_size<double>(handle, m, 1, nullptr, dd, ddu, dB, 8, &size),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(gtsv_no_pivot_buffer_size<double>(handle, m, 1, ddl, dd, ddu, dB, 8, nullptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(gtsv_no_pivot_buffer_size<double>(nullptr, m, 1, ddl, dd, ddu, dB, 8, &size),
              rocsparse_status_invalid_handle);
    // Empty, small and large systems (the size of the workspace is not specified).
    EXPECT_EQ(gtsv_no_pivot_buffer_size<double>(
                  handle, m, 0, nullptr, nullptr, nullptr, nullptr, 8, &size),
              rocsparse_status_success);
    EXPECT_EQ(gtsv_no_pivot_buffer_size<double>(handle, 1024, 1, ddl, dd, ddu, dB, 1024, &size),
              rocsparse_status_success);
    size = 0;
    EXPECT_EQ(gtsv_no_pivot_buffer_size<double>(handle, 2000, 1, ddl, dd, ddu, dB, 2000, &size),
              rocsparse_status_success);

    EXPECT_EQ(gtsv_no_pivot<double>(handle, 1, 1, ddl, dd, ddu, dB, 8, nullptr),
              rocsparse_status_invalid_size);
    EXPECT_EQ(gtsv_no_pivot<double>(handle, m, 1, ddl, dd, ddu, dB, 4, nullptr),
              rocsparse_status_invalid_size);
    EXPECT_EQ(gtsv_no_pivot<double>(handle, m, 1, nullptr, dd, ddu, dB, 8, nullptr),
              rocsparse_status_invalid_pointer);
    // A system that asks for a workspace cannot be solved without one.
    if(size > 0)
        EXPECT_EQ(gtsv_no_pivot<double>(handle, 2000, 1, ddl, dd, ddu, dB, 2000, nullptr),
                  rocsparse_status_invalid_pointer);
    EXPECT_EQ(gtsv_no_pivot<double>(handle, m, 0, nullptr, nullptr, nullptr, nullptr, 8, nullptr),
              rocsparse_status_success);
    EXPECT_EQ(gtsv_no_pivot<double>(nullptr, m, 1, ddl, dd, ddu, dB, 8, nullptr),
              rocsparse_status_invalid_handle);
}

TEST_F(PrecondTridiagonal, strided_batch_argument_validation)
{
    const int32_t          m = 8;
    const system_t<double> s = make_system<double>(m, 1, 1);
    device_vector<double>  ddl{s.dl}, dd{s.d}, ddu{s.du}, dx{s.b};
    ASSERT_TRUE(ddl.ptr && dd.ptr && ddu.ptr && dx.ptr);
    size_t size = 7;

    EXPECT_EQ(strided_buffer_size<double>(handle, 1, ddl, dd, ddu, dx, 1, 8, &size),
              rocsparse_status_invalid_size);
    EXPECT_EQ(strided_buffer_size<double>(handle, m, ddl, dd, ddu, dx, 1, 4, &size),
              rocsparse_status_invalid_size);
    EXPECT_EQ(strided_buffer_size<double>(handle, m, ddl, dd, ddu, dx, -1, 8, &size),
              rocsparse_status_invalid_size);
    EXPECT_EQ(strided_buffer_size<double>(handle, m, nullptr, dd, ddu, dx, 1, 8, &size),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(strided_buffer_size<double>(handle, m, ddl, dd, ddu, dx, 1, 8, nullptr),
              rocsparse_status_invalid_pointer);
    // Empty, small and large systems (the size of the workspace is not specified).
    EXPECT_EQ(
        strided_buffer_size<double>(handle, m, nullptr, nullptr, nullptr, nullptr, 0, 8, &size),
        rocsparse_status_success);
    EXPECT_EQ(strided_buffer_size<double>(handle, 1024, ddl, dd, ddu, dx, 1, 1024, &size),
              rocsparse_status_success);
    EXPECT_EQ(strided_buffer_size<double>(handle, 1025, ddl, dd, ddu, dx, 2, 1025, &size),
              rocsparse_status_success);
    size = 0;
    EXPECT_EQ(strided_buffer_size<double>(handle, 2000, ddl, dd, ddu, dx, 1, 2000, &size),
              rocsparse_status_success);

    EXPECT_EQ(strided_solve<double>(handle, 1, ddl, dd, ddu, dx, 1, 8, nullptr),
              rocsparse_status_invalid_size);
    EXPECT_EQ(strided_solve<double>(handle, m, nullptr, dd, ddu, dx, 1, 8, nullptr),
              rocsparse_status_invalid_pointer);
    // A system that asks for a workspace cannot be solved without one.
    if(size > 0)
        EXPECT_EQ(strided_solve<double>(handle, 2000, ddl, dd, ddu, dx, 1, 2000, nullptr),
                  rocsparse_status_invalid_pointer);
    EXPECT_EQ(strided_solve<double>(handle, m, nullptr, nullptr, nullptr, nullptr, 0, 8, nullptr),
              rocsparse_status_success);
    EXPECT_EQ(strided_solve<double>(nullptr, m, ddl, dd, ddu, dx, 1, 8, nullptr),
              rocsparse_status_invalid_handle);
}

// ---------------------------------------------------------------------------
// Interleaved batch: element i of system b lives at i * batch_stride + b.
// ---------------------------------------------------------------------------
namespace
{
    template <typename T>
    void run_interleaved(rocsparse_handle               handle,
                         rocsparse_gtsv_interleaved_alg alg,
                         int32_t                        m,
                         int32_t                        batch_count,
                         int32_t                        extra)
    {
        SCOPED_TRACE("interleaved alg=" + std::to_string(static_cast<int>(alg))
                     + " m=" + std::to_string(m) + " batch=" + std::to_string(batch_count));
        const int32_t               stride = batch_count + extra;
        const size_t                total  = static_cast<size_t>(m) * stride;
        std::vector<T>              dl(total, make_value<T>(0.0)), d = dl, du = dl, b = dl;
        std::vector<std::vector<T>> want;
        for(int32_t k = 0; k < batch_count; ++k)
        {
            const system_t<T> s = make_system<T>(m, 1, k + 1);
            for(int32_t i = 0; i < m; ++i)
            {
                const size_t pos = static_cast<size_t>(i) * stride + k;
                dl[pos]          = s.dl[i];
                d[pos]           = s.d[i];
                du[pos]          = s.du[i];
                b[pos]           = s.b[i];
            }
            want.push_back(s.x);
        }
        device_vector<T> ddl{dl}, dd{d}, ddu{du}, dx{b};
        ASSERT_TRUE(ddl.ptr && dd.ptr && ddu.ptr && dx.ptr);
        size_t size = 0;
        ASSERT_EQ(interleaved_buffer_size<T>(
                      handle, alg, m, ddl, dd, ddu, dx, batch_count, stride, &size),
                  rocsparse_status_success);
        ASSERT_GT(size, 0u);
        device_vector<char> buffer{size};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(
            interleaved_solve<T>(handle, alg, m, ddl, dd, ddu, dx, batch_count, stride, buffer.ptr),
            rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        const auto got = to_host(dx);
        for(int32_t k = 0; k < batch_count; ++k)
        {
            std::vector<T> sol(m);
            for(int32_t i = 0; i < m; ++i)
                sol[i] = got[static_cast<size_t>(i) * stride + k];
            expect_solution(sol, want[k], "batch " + std::to_string(k));
        }
    }
}

TEST_F(PrecondTridiagonal, interleaved_batch_algorithms)
{
    const rocsparse_gtsv_interleaved_alg algs[] = {rocsparse_gtsv_interleaved_alg_default,
                                                   rocsparse_gtsv_interleaved_alg_thomas,
                                                   rocsparse_gtsv_interleaved_alg_lu,
                                                   rocsparse_gtsv_interleaved_alg_qr};
    for(const auto alg : algs)
    {
        for_all_types([&](auto tag) {
            using T = decltype(tag);
            run_interleaved<T>(handle, alg, 2, 5, 0);
            run_interleaved<T>(handle, alg, 9, 130, 3);
        });
        run_interleaved<double>(handle, alg, 33, 257, 0);
    }
}

TEST_F(PrecondTridiagonal, interleaved_batch_argument_validation)
{
    const int32_t          m = 4;
    const system_t<double> s = make_system<double>(m, 1, 1);
    device_vector<double>  ddl{s.dl}, dd{s.d}, ddu{s.du}, dx{s.b};
    ASSERT_TRUE(ddl.ptr && dd.ptr && ddu.ptr && dx.ptr);
    size_t size = 7;
    EXPECT_EQ(interleaved_buffer_size<double>(handle,
                                              static_cast<rocsparse_gtsv_interleaved_alg>(99),
                                              m,
                                              ddl,
                                              dd,
                                              ddu,
                                              dx,
                                              1,
                                              1,
                                              &size),
              rocsparse_status_invalid_value);
    // A batch of zero systems needs no workspace and nothing is solved.
    for(const auto alg : {rocsparse_gtsv_interleaved_alg_default,
                          rocsparse_gtsv_interleaved_alg_thomas,
                          rocsparse_gtsv_interleaved_alg_lu,
                          rocsparse_gtsv_interleaved_alg_qr})
    {
        size = 7;
        EXPECT_EQ(interleaved_buffer_size<double>(
                      handle, alg, m, nullptr, nullptr, nullptr, nullptr, 0, 1, &size),
                  rocsparse_status_success);
        EXPECT_EQ(size, 0u);
        EXPECT_EQ(interleaved_solve<double>(
                      handle, alg, m, nullptr, nullptr, nullptr, nullptr, 0, 1, nullptr),
                  rocsparse_status_success);
    }
}
