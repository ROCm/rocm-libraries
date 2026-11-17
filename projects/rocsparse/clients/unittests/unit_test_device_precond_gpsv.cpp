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

// Pentadiagonal QR. A uniform scale of the five bands and the right-hand side does
// not change the solution, but squaring the raw entries overflows or underflows once
// they leave the square root of the format range. These solves are the order-4
// witness of that failure, plus order 17 so the general backsolve loop runs. The
// result is compared to the known solution, not to a host QR that still squares the
// entries.
//
// Two paths are covered:
//  - the public routine: Householder QR for real types, Givens QR for complex types, and
//  - the Givens kernel launched directly for all four types.

#include "unit_test_device_precond_common.hpp"

#include "../../library/src/precond/gpsv/gpsv_interleaved_batch_device.h"

#include <sstream>
#include <string>

using namespace precond_ut;

namespace
{
    // The five bands of an order m system with every entry multiplied by `scale`, a
    // known solution, and the right-hand side A * x. Real types drop the imaginary
    // parts.
    template <typename T>
    struct scaled_system_t
    {
        std::vector<T> ds, dl, d, du, dw, rhs, want;
    };

    // A true member drops that whole off-diagonal band. The diagonal is never dropped.
    struct zero_bands_t
    {
        bool ds;
        bool dl;
        bool du;
        bool dw;
    };

    template <typename T>
    scaled_system_t<T> make_scaled_system(int32_t m, double scale, zero_bands_t zero = {})
    {
        scaled_system_t<T> s;
        s.ds.resize(m);
        s.dl.resize(m);
        s.d.resize(m);
        s.du.resize(m);
        s.dw.resize(m);
        s.rhs.resize(m);
        s.want.resize(m);
        for(int32_t j = 0; j < m; ++j)
        {
            s.ds[j]   = (!zero.ds && j >= 2) ? make_value<T>(-0.2 * scale, 0.1 * scale)
                                             : make_value<T>(0.0);
            s.dl[j]   = (!zero.dl && j >= 1) ? make_value<T>(-0.5 * scale, 0.2 * scale)
                                             : make_value<T>(0.0);
            s.d[j]    = make_value<T>(4.0 * scale, 0.3 * scale);
            s.du[j]   = (!zero.du && j < m - 1) ? make_value<T>(0.75 * scale, -0.1 * scale)
                                                : make_value<T>(0.0);
            s.dw[j]   = (!zero.dw && j < m - 2) ? make_value<T>(0.3 * scale, 0.15 * scale)
                                                : make_value<T>(0.0);
            s.want[j] = make_value<T>(static_cast<double>(j + 1) / 7.0,
                                      0.2 * static_cast<double>(j + 1) / 7.0);
        }
        for(int32_t j = 0; j < m; ++j)
        {
            std::complex<double> b = to_cd(s.d[j]) * to_cd(s.want[j]);
            if(j >= 1)
                b += to_cd(s.dl[j]) * to_cd(s.want[j - 1]);
            if(j >= 2)
                b += to_cd(s.ds[j]) * to_cd(s.want[j - 2]);
            if(j + 1 < m)
                b += to_cd(s.du[j]) * to_cd(s.want[j + 1]);
            if(j + 2 < m)
                b += to_cd(s.dw[j]) * to_cd(s.want[j + 2]);
            s.rhs[j] = from_cd<T>(b);
        }
        return s;
    }

    // One sentinel on either side of the m unknowns. The solver writes only the
    // unknowns, so both sentinels must stay at 77.
    template <typename T>
    std::vector<T> pad(const std::vector<T>& v)
    {
        std::vector<T> out(v.size() + 2, make_value<T>(77.0, 0.0));
        std::copy(v.begin(), v.end(), out.begin() + 1);
        return out;
    }

    template <typename T>
    void expect_scaled_solution(const std::vector<T>& got_padded, const std::vector<T>& want)
    {
        ASSERT_EQ(got_padded.size(), want.size() + 2);
        EXPECT_DOUBLE_EQ(re(got_padded.front()), 77.0);
        EXPECT_DOUBLE_EQ(im(got_padded.front()), 0.0);
        EXPECT_DOUBLE_EQ(re(got_padded.back()), 77.0);
        EXPECT_DOUBLE_EQ(im(got_padded.back()), 0.0);

        std::vector<T> sol(got_padded.begin() + 1, got_padded.end() - 1);
        if(!expect_finite(sol, "solution"))
            return;
        const double tol = is_single_precision<T> ? 1.0e-4 : 1.0e-8;
        for(size_t j = 0; j < want.size(); ++j)
        {
            EXPECT_NEAR(re(sol[j]), re(want[j]), tol) << "x[" << j << "]";
            EXPECT_NEAR(im(sol[j]), im(want[j]), tol) << "x[" << j << "]";
        }
    }

    // Solves through the public routine.
    template <typename T>
    void run_uniform_scale(rocsparse_handle               handle,
                           rocsparse_gpsv_interleaved_alg alg,
                           int32_t                        m,
                           double                         scale,
                           zero_bands_t                   zero = {})
    {
        std::ostringstream trace;
        trace << "api alg=" << static_cast<int>(alg) << " m=" << m << " scale=" << std::scientific
              << scale;
        if(zero.ds)
            trace << " ds=0";
        if(zero.dl)
            trace << " dl=0";
        if(zero.du)
            trace << " du=0";
        if(zero.dw)
            trace << " dw=0";
        SCOPED_TRACE(trace.str());

        const auto       s = make_scaled_system<T>(m, scale, zero);
        device_vector<T> dds{pad(s.ds)}, ddl{pad(s.dl)}, dd{pad(s.d)}, ddu{pad(s.du)},
            ddw{pad(s.dw)}, dx{pad(s.rhs)};
        ASSERT_TRUE(dds.ptr && ddl.ptr && dd.ptr && ddu.ptr && ddw.ptr && dx.ptr);

        const rocsparse_int batch_count  = 1;
        const rocsparse_int batch_stride = 1;
        const rocsparse_int order        = m;
        size_t              buffer_size  = 0;
        ASSERT_EQ(PRECOND_CALL(T,
                               gpsv_interleaved_batch_buffer_size,
                               handle,
                               alg,
                               order,
                               dds.ptr + 1,
                               ddl.ptr + 1,
                               dd.ptr + 1,
                               ddu.ptr + 1,
                               ddw.ptr + 1,
                               dx.ptr + 1,
                               batch_count,
                               batch_stride,
                               &buffer_size),
                  rocsparse_status_success);
        ASSERT_GT(buffer_size, 0u);

        device_vector<char> buffer{buffer_size};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(PRECOND_CALL(T,
                               gpsv_interleaved_batch,
                               handle,
                               alg,
                               order,
                               dds.ptr + 1,
                               ddl.ptr + 1,
                               dd.ptr + 1,
                               ddu.ptr + 1,
                               ddw.ptr + 1,
                               dx.ptr + 1,
                               batch_count,
                               batch_stride,
                               buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        expect_scaled_solution(to_host(dx), s.want);
    }

    // Launches the Givens kernel directly, whatever T is.
    template <typename T>
    void run_givens_kernel(int32_t m, double scale)
    {
        std::ostringstream trace;
        trace << "givens kernel m=" << m << " scale=" << std::scientific << scale;
        SCOPED_TRACE(trace.str());

        const auto       s = make_scaled_system<T>(m, scale);
        device_vector<T> dds{pad(s.ds)}, ddl{pad(s.dl)}, dd{pad(s.d)}, ddu{pad(s.du)},
            ddw{pad(s.dw)}, dx{pad(s.rhs)};
        // The kernel reads r3 before it writes it, so both start at zero.
        device_vector<T> dr3{std::vector<T>(m, make_value<T>(0.0))};
        device_vector<T> dr4{std::vector<T>(m, make_value<T>(0.0))};
        ASSERT_TRUE(dds.ptr && ddl.ptr && dd.ptr && ddu.ptr && ddw.ptr && dx.ptr && dr3.ptr
                    && dr4.ptr);

        const rocsparse_int batch_count  = 1;
        const rocsparse_int batch_stride = 1;
        hipLaunchKernelGGL((rocsparse::gpsv_interleaved_batch_givens_qr_kernel<128, T>),
                           dim3(1),
                           dim3(128),
                           0,
                           0,
                           static_cast<rocsparse_int>(m),
                           batch_count,
                           batch_stride,
                           dds.ptr + 1,
                           ddl.ptr + 1,
                           dd.ptr + 1,
                           ddu.ptr + 1,
                           ddw.ptr + 1,
                           dr3.ptr,
                           dr4.ptr,
                           dx.ptr + 1);
        ASSERT_EQ(hipGetLastError(), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        expect_scaled_solution(to_host(dx), s.want);
    }
}

class PrecondGpsvInterleaved : public HandleTest
{
};

TEST_F(PrecondGpsvInterleaved, uniform_scale_keeps_solution)
{
    const rocsparse_gpsv_interleaved_alg algs[]
        = {rocsparse_gpsv_interleaved_alg_default, rocsparse_gpsv_interleaved_alg_qr};
    const int32_t orders[] = {4, 17};
    for(const auto alg : algs)
    {
        for(const int32_t m : orders)
        {
            run_uniform_scale<float>(handle, alg, m, 1.0);
            run_uniform_scale<float>(handle, alg, m, 1.0e20);
            run_uniform_scale<float>(handle, alg, m, 1.0e-20);
            run_uniform_scale<rocsparse_float_complex>(handle, alg, m, 1.0);
            run_uniform_scale<rocsparse_float_complex>(handle, alg, m, 1.0e20);
            run_uniform_scale<rocsparse_float_complex>(handle, alg, m, 1.0e-20);

            run_uniform_scale<double>(handle, alg, m, 1.0);
            run_uniform_scale<double>(handle, alg, m, 1.0e200);
            run_uniform_scale<double>(handle, alg, m, 1.0e-200);
            run_uniform_scale<rocsparse_double_complex>(handle, alg, m, 1.0);
            run_uniform_scale<rocsparse_double_complex>(handle, alg, m, 1.0e200);
            run_uniform_scale<rocsparse_double_complex>(handle, alg, m, 1.0e-200);
        }
    }

    // Every subset of {ds, dl, du, dw} is dropped. The diagonal stays nonzero, so the
    // system stays nonsingular and the solution is known. Bits, from low to high, are
    // ds, dl, du, dw. ds = dl = 0 makes the first complex Givens rotation (0, 0), which
    // used to divide by a zero hypotenuse, and skips the real Householder reflector.
    for(const auto alg : algs)
    {
        for(int mask = 0; mask < 16; ++mask)
        {
            const zero_bands_t zero{static_cast<bool>(mask & 1),
                                    static_cast<bool>(mask & 2),
                                    static_cast<bool>(mask & 4),
                                    static_cast<bool>(mask & 8)};

            run_uniform_scale<float>(handle, alg, 4, 1.0, zero);
            run_uniform_scale<double>(handle, alg, 4, 1.0, zero);
            run_uniform_scale<rocsparse_float_complex>(handle, alg, 4, 1.0, zero);
            run_uniform_scale<rocsparse_double_complex>(handle, alg, 4, 1.0, zero);
        }
    }
}

// The scales sit outside sqrt(FLT_MAX) ~ 1.8e19 and sqrt(FLT_MIN) ~ 1.1e-19 for the
// single precision types, and outside the same limits of double for the others.
TEST_F(PrecondGpsvInterleaved, givens_kernel_uniform_scale_keeps_solution)
{
    for(const int32_t m : {4, 17})
    {
        run_givens_kernel<float>(m, 1.0);
        run_givens_kernel<float>(m, 1.0e20);
        run_givens_kernel<float>(m, 1.0e-20);
        run_givens_kernel<rocsparse_float_complex>(m, 1.0);
        run_givens_kernel<rocsparse_float_complex>(m, 1.0e20);
        run_givens_kernel<rocsparse_float_complex>(m, 1.0e-20);

        run_givens_kernel<double>(m, 1.0);
        run_givens_kernel<double>(m, 1.0e200);
        run_givens_kernel<double>(m, 1.0e-200);
        run_givens_kernel<rocsparse_double_complex>(m, 1.0);
        run_givens_kernel<rocsparse_double_complex>(m, 1.0e200);
        run_givens_kernel<rocsparse_double_complex>(m, 1.0e-200);
    }
}
