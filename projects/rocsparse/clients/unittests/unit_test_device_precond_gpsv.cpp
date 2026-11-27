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
//
// Pentadiagonal interleaved-batch solve. The complex QR kernel used to update
// row m - 4 unconditionally. For the smallest accepted order, m == 3, that row
// is index -batch_stride, so the kernel wrote one value before x. Each array
// passed to the solve is preceded by a sentinel prefix of batch_stride entries
// so that write stays inside the allocation and is observable.

#include "unit_test_device_precond_common.hpp"

#include "../../library/src/precond/gpsv/gpsv_interleaved_batch_device.h"

#include <limits>
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

    template <typename T>
    void expect_scaled_solution(const std::vector<T>& got_padded, const std::vector<T>& want)
    {
        ASSERT_EQ(got_padded.size(), want.size() + 2);
        // The solver writes only the unknowns, so both sentinels must stay as they were.
        expect_padding_untouched(got_padded);

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

    template <typename T, typename... A>
    rocsparse_status gpsv_buffer_size(A&&... args)
    {
        return PRECOND_CALL(T, gpsv_interleaved_batch_buffer_size, std::forward<A>(args)...);
    }

    template <typename T, typename... A>
    rocsparse_status gpsv_solve(A&&... args)
    {
        return PRECOND_CALL(T, gpsv_interleaved_batch, std::forward<A>(args)...);
    }

    // Solves a system through the public routine. `got_padded` is x with its sentinels.
    template <typename T>
    void solve_public_padded(rocsparse_handle               handle,
                             rocsparse_gpsv_interleaved_alg alg,
                             const scaled_system_t<T>&      s,
                             std::vector<T>&                got_padded)
    {
        const rocsparse_int order = static_cast<rocsparse_int>(s.d.size());
        device_vector<T>    dds{pad_with_sentinel(s.ds)}, ddl{pad_with_sentinel(s.dl)},
            dd{pad_with_sentinel(s.d)}, ddu{pad_with_sentinel(s.du)}, ddw{pad_with_sentinel(s.dw)},
            dx{pad_with_sentinel(s.rhs)};
        ASSERT_TRUE(dds.ptr && ddl.ptr && dd.ptr && ddu.ptr && ddw.ptr && dx.ptr);

        const rocsparse_int batch_count  = 1;
        const rocsparse_int batch_stride = 1;
        size_t              buffer_size  = 0;
        ASSERT_EQ(gpsv_buffer_size<T>(handle,
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
        ASSERT_EQ(gpsv_solve<T>(handle,
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

        got_padded = to_host(dx);
    }

    // Solves a system through the public routine and compares with the known solution.
    template <typename T>
    void solve_public(rocsparse_handle               handle,
                      rocsparse_gpsv_interleaved_alg alg,
                      const scaled_system_t<T>&      s)
    {
        std::vector<T> got_padded;
        solve_public_padded<T>(handle, alg, s, got_padded);
        if(!got_padded.empty())
            expect_scaled_solution(got_padded, s.want);
    }

    // Solves the uniformly scaled system through the public routine.
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

        solve_public<T>(handle, alg, make_scaled_system<T>(m, scale, zero));
    }

    // Launches the Givens kernel directly, whatever T is.
    template <typename T>
    void run_givens_kernel(int32_t m, double scale)
    {
        std::ostringstream trace;
        trace << "givens kernel m=" << m << " scale=" << std::scientific << scale;
        SCOPED_TRACE(trace.str());

        const auto       s = make_scaled_system<T>(m, scale);
        device_vector<T> dds{pad_with_sentinel(s.ds)}, ddl{pad_with_sentinel(s.dl)},
            dd{pad_with_sentinel(s.d)}, ddu{pad_with_sentinel(s.du)}, ddw{pad_with_sentinel(s.dw)},
            dx{pad_with_sentinel(s.rhs)};
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

    // Storage is [prefix | m * stride | suffix]. The pointer handed to the
    // routine is the first entry after the prefix. A write at index
    // -stride + gid lands on prefix entry gid.
    struct band_t
    {
        int32_t             m            = 0;
        int32_t             batch_count  = 0;
        int32_t             batch_stride = 0;
        int32_t             prefix       = 0;
        std::vector<double> ds, dl, d, du, dw; // real parts of one batch, length m
        std::vector<double> ds_i, dl_i, d_i, du_i, dw_i;
    };

    // Diagonally dominant pentadiagonal system. `offdiag` false keeps only the
    // main diagonal. Imaginary parts are zero for real types.
    template <typename T>
    band_t make_band(int32_t m, int32_t batch, bool offdiag)
    {
        band_t a;
        a.m = m;
        a.ds.resize(m);
        a.dl.resize(m);
        a.d.assign(m, 4.0 + 0.25 * batch);
        a.du.resize(m);
        a.dw.resize(m);
        a.ds_i.resize(m);
        a.dl_i.resize(m);
        a.d_i.assign(m, offdiag && is_cplx<T> ? 0.3 : 0.0);
        a.du_i.resize(m);
        a.dw_i.resize(m);
        for(int32_t i = 0; i < m; ++i)
        {
            const bool lower2 = offdiag && i >= 2;
            const bool lower1 = offdiag && i >= 1;
            const bool upper1 = offdiag && i + 1 < m;
            const bool upper2 = offdiag && i + 2 < m;
            a.ds[i]           = lower2 ? -0.2 : 0.0;
            a.dl[i]           = lower1 ? -0.5 : 0.0;
            a.du[i]           = upper1 ? 0.75 : 0.0;
            a.dw[i]           = upper2 ? 0.3 : 0.0;
            a.ds_i[i]         = lower2 && is_cplx<T> ? 0.1 : 0.0;
            a.dl_i[i]         = lower1 && is_cplx<T> ? 0.2 : 0.0;
            a.du_i[i]         = upper1 && is_cplx<T> ? -0.1 : 0.0;
            a.dw_i[i]         = upper2 && is_cplx<T> ? 0.15 : 0.0;
        }
        return a;
    }

    template <typename T>
    T at(const std::vector<double>& re, const std::vector<double>& im, int32_t i)
    {
        return make_value<T>(re[i], im[i]);
    }

    // Known solution (j + 1) / 7, with a matching imaginary part when the matrix
    // has one. Right-hand side is A * x for that solution.
    template <typename T>
    void rhs_and_solution(const band_t& a, std::vector<T>& x, std::vector<T>& b)
    {
        x.resize(a.m);
        b.resize(a.m);
        for(int32_t i = 0; i < a.m; ++i)
        {
            const double xi = (is_cplx<T> && a.d_i[i] != 0.0) ? 0.2 * (i + 1) / 7.0 : 0.0;
            x[i]            = make_value<T>((i + 1) / 7.0, xi);
        }
        for(int32_t i = 0; i < a.m; ++i)
        {
            std::complex<double> rhs = to_cd(at<T>(a.d, a.d_i, i)) * to_cd(x[i]);
            if(i >= 1)
                rhs += to_cd(at<T>(a.dl, a.dl_i, i)) * to_cd(x[i - 1]);
            if(i >= 2)
                rhs += to_cd(at<T>(a.ds, a.ds_i, i)) * to_cd(x[i - 2]);
            if(i + 1 < a.m)
                rhs += to_cd(at<T>(a.du, a.du_i, i)) * to_cd(x[i + 1]);
            if(i + 2 < a.m)
                rhs += to_cd(at<T>(a.dw, a.dw_i, i)) * to_cd(x[i + 2]);
            b[i] = from_cd<T>(rhs);
        }
    }

    template <typename T>
    bool is_sentinel(T v)
    {
        return re(v) == padding_sentinel && im(v) == 0.0;
    }

    template <typename T>
    void expect_untouched(const std::vector<T>& got, const band_t& a, const std::string& what)
    {
        const int32_t stride = a.batch_stride;
        for(int32_t i = 0; i < a.prefix; ++i)
            EXPECT_TRUE(is_sentinel(got[i])) << what << " prefix[" << i << "]";
        const int32_t end = a.prefix + a.m * stride;
        for(int32_t i = end; i < end + stride; ++i)
            EXPECT_TRUE(is_sentinel(got[i])) << what << " suffix[" << (i - end) << "]";
        for(int32_t row = 0; row < a.m; ++row)
            for(int32_t b = a.batch_count; b < stride; ++b)
                EXPECT_TRUE(is_sentinel(got[a.prefix + row * stride + b]))
                    << what << " padding row " << row << " batch " << b;
    }

    template <typename T>
    void run_case(rocsparse_handle               handle,
                  rocsparse_gpsv_interleaved_alg alg,
                  int32_t                        m,
                  int32_t                        batch_count,
                  int32_t                        batch_stride,
                  bool                           offdiag,
                  bool                           check_solution)
    {
        SCOPED_TRACE(std::string(is_cplx<T> ? (is_single_precision<T> ? "c" : "z")
                                            : (is_single_precision<T> ? "s" : "d"))
                     + " alg=" + std::to_string(static_cast<int>(alg)) + " m=" + std::to_string(m)
                     + " batch=" + std::to_string(batch_count) + " stride="
                     + std::to_string(batch_stride) + (offdiag ? " pentadiagonal" : " diagonal"));

        band_t layout;
        layout.m            = m;
        layout.batch_count  = batch_count;
        layout.batch_stride = batch_stride;
        layout.prefix       = batch_stride;

        const size_t   n = static_cast<size_t>(layout.prefix + m * batch_stride + batch_stride);
        std::vector<T> hds(n, make_value<T>(padding_sentinel));
        std::vector<T> hdl = hds, hd = hds, hdu = hds, hdw = hds, hx = hds;
        std::vector<std::vector<T>> want(batch_count);

        for(int32_t b = 0; b < batch_count; ++b)
        {
            const band_t   band = make_band<T>(m, b, offdiag);
            std::vector<T> x, rhs;
            rhs_and_solution<T>(band, x, rhs);
            want[b] = x;
            for(int32_t i = 0; i < m; ++i)
            {
                const size_t pos = static_cast<size_t>(layout.prefix + i * batch_stride + b);
                hds[pos]         = at<T>(band.ds, band.ds_i, i);
                hdl[pos]         = at<T>(band.dl, band.dl_i, i);
                hd[pos]          = at<T>(band.d, band.d_i, i);
                hdu[pos]         = at<T>(band.du, band.du_i, i);
                hdw[pos]         = at<T>(band.dw, band.dw_i, i);
                hx[pos]          = rhs[i];
            }
        }

        device_vector<T> dds{hds}, ddl{hdl}, dd{hd}, ddu{hdu}, ddw{hdw}, dx{hx};
        ASSERT_TRUE(dds.ptr && ddl.ptr && dd.ptr && ddu.ptr && ddw.ptr && dx.ptr);
        T* pds = dds.ptr + layout.prefix;
        T* pdl = ddl.ptr + layout.prefix;
        T* pd  = dd.ptr + layout.prefix;
        T* pdu = ddu.ptr + layout.prefix;
        T* pdw = ddw.ptr + layout.prefix;
        T* px  = dx.ptr + layout.prefix;

        size_t size = 0;
        ASSERT_EQ(gpsv_buffer_size<T>(
                      handle, alg, m, pds, pdl, pd, pdu, pdw, px, batch_count, batch_stride, &size),
                  rocsparse_status_success);
        device_vector<char> buffer{size ? size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(
            gpsv_solve<T>(
                handle, alg, m, pds, pdl, pd, pdu, pdw, px, batch_count, batch_stride, buffer.ptr),
            rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const auto got_x = to_host(dx);
        expect_untouched(got_x, layout, "x");
        expect_untouched(to_host(dds), layout, "ds");
        expect_untouched(to_host(ddl), layout, "dl");
        expect_untouched(to_host(dd), layout, "d");
        expect_untouched(to_host(ddu), layout, "du");
        expect_untouched(to_host(ddw), layout, "dw");

        if(!check_solution)
            return;

        for(int32_t b = 0; b < batch_count; ++b)
        {
            std::vector<T> sol(m);
            for(int32_t i = 0; i < m; ++i)
                sol[i] = got_x[static_cast<size_t>(layout.prefix + i * batch_stride + b)];
            expect_close(sol, want[b], is_single_precision<T> ? 1.0e-4 : 1.0e-10);
        }
    }
}

class PrecondGpsvInterleaved : public HandleTest
{
};

// Float scales of 1e-20 only reach the denormal range of the squared band entries, which
// the unscaled real kernel still solves. 1e-25 squares to 1e-50, which flushes to zero.
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
            run_uniform_scale<float>(handle, alg, m, 1.0e-25);
            run_uniform_scale<rocsparse_float_complex>(handle, alg, m, 1.0);
            run_uniform_scale<rocsparse_float_complex>(handle, alg, m, 1.0e20);
            run_uniform_scale<rocsparse_float_complex>(handle, alg, m, 1.0e-20);
            run_uniform_scale<rocsparse_float_complex>(handle, alg, m, 1.0e-25);

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
        run_givens_kernel<float>(m, 1.0e-25);
        run_givens_kernel<rocsparse_float_complex>(m, 1.0);
        run_givens_kernel<rocsparse_float_complex>(m, 1.0e20);
        run_givens_kernel<rocsparse_float_complex>(m, 1.0e-20);
        run_givens_kernel<rocsparse_float_complex>(m, 1.0e-25);

        run_givens_kernel<double>(m, 1.0);
        run_givens_kernel<double>(m, 1.0e200);
        run_givens_kernel<double>(m, 1.0e-200);
        run_givens_kernel<rocsparse_double_complex>(m, 1.0);
        run_givens_kernel<rocsparse_double_complex>(m, 1.0e200);
        run_givens_kernel<rocsparse_double_complex>(m, 1.0e-200);
    }
}

namespace
{
    // The rows are on very different scales: row 0 is of order `big`, row 1 is of order
    // `small`, and the coupling between them is `small`. A Householder reflector that
    // is built from the squared lower coefficient of the row-0 column sees
    // (small / big)^2, which is zero in the format, but row 1 still depends on x0.
    // The solution is all ones.
    template <typename T>
    scaled_system_t<T> make_mixed_magnitude_system(double big, double small)
    {
        scaled_system_t<T> s;
        s.ds = std::vector<T>(4, make_value<T>(0.0));
        s.du = std::vector<T>(4, make_value<T>(0.0));
        s.dw = std::vector<T>(4, make_value<T>(0.0));
        s.dl = {make_value<T>(0.0), make_value<T>(small), make_value<T>(0.0), make_value<T>(0.0)};
        s.d  = {
            make_value<T>(big), make_value<T>(2.0 * small), make_value<T>(1.0), make_value<T>(1.0)};
        s.rhs = {
            make_value<T>(big), make_value<T>(3.0 * small), make_value<T>(1.0), make_value<T>(1.0)};
        s.want = std::vector<T>(4, make_value<T>(1.0));
        return s;
    }

    // A diagonal system whose last unknown is finite, but only just: H / (0.6 + 0.6i) with
    // H + iH on the right-hand side is the real number H / 0.6. A division that adds the
    // two products of its numerator before the final scaling overflows on the way.
    template <typename T>
    scaled_system_t<T> make_large_quotient_system(double half_max)
    {
        scaled_system_t<T> s;
        s.ds = std::vector<T>(4, make_value<T>(0.0));
        s.dl = std::vector<T>(4, make_value<T>(0.0));
        s.du = std::vector<T>(4, make_value<T>(0.0));
        s.dw = std::vector<T>(4, make_value<T>(0.0));
        s.d = {make_value<T>(1.0), make_value<T>(1.0), make_value<T>(1.0), make_value<T>(0.6, 0.6)};
        s.rhs  = {make_value<T>(1.0),
                 make_value<T>(1.0),
                 make_value<T>(1.0),
                 make_value<T>(half_max, half_max)};
        s.want = {make_value<T>(1.0),
                  make_value<T>(1.0),
                  make_value<T>(1.0),
                  make_value<T>(half_max / 0.6)};
        return s;
    }
}

// Householder QR of the real routines and Givens QR of the complex ones.
TEST_F(PrecondGpsvInterleaved, mixed_row_scales_keep_solution)
{
    const rocsparse_gpsv_interleaved_alg algs[]
        = {rocsparse_gpsv_interleaved_alg_default, rocsparse_gpsv_interleaved_alg_qr};
    for(const auto alg : algs)
    {
        SCOPED_TRACE("alg=" + std::to_string(static_cast<int>(alg)));
        solve_public<float>(handle, alg, make_mixed_magnitude_system<float>(1.0e10, 1.0e-15));
        solve_public<rocsparse_float_complex>(
            handle, alg, make_mixed_magnitude_system<rocsparse_float_complex>(1.0e10, 1.0e-15));
        solve_public<double>(handle, alg, make_mixed_magnitude_system<double>(1.0e100, 1.0e-100));
        solve_public<rocsparse_double_complex>(
            handle, alg, make_mixed_magnitude_system<rocsparse_double_complex>(1.0e100, 1.0e-100));
    }
}

// Only the complex backsolve divides by a complex number.
TEST_F(PrecondGpsvInterleaved, complex_division_keeps_large_finite_quotient)
{
    const rocsparse_gpsv_interleaved_alg algs[]
        = {rocsparse_gpsv_interleaved_alg_default, rocsparse_gpsv_interleaved_alg_qr};
    for(const auto alg : algs)
    {
        SCOPED_TRACE("alg=" + std::to_string(static_cast<int>(alg)));

        std::vector<rocsparse_float_complex> got_c;
        solve_public_padded<rocsparse_float_complex>(
            handle,
            alg,
            make_large_quotient_system<rocsparse_float_complex>(
                0.5 * std::numeric_limits<float>::max()),
            got_c);
        ASSERT_EQ(got_c.size(), size_t(6));
        expect_padding_untouched(got_c);
        expect_close(std::vector<rocsparse_float_complex>(got_c.begin() + 1, got_c.end() - 1),
                     make_large_quotient_system<rocsparse_float_complex>(
                         0.5 * std::numeric_limits<float>::max())
                         .want);

        std::vector<rocsparse_double_complex> got_z;
        solve_public_padded<rocsparse_double_complex>(
            handle,
            alg,
            make_large_quotient_system<rocsparse_double_complex>(
                0.5 * std::numeric_limits<double>::max()),
            got_z);
        ASSERT_EQ(got_z.size(), size_t(6));
        expect_padding_untouched(got_z);
        expect_close(std::vector<rocsparse_double_complex>(got_z.begin() + 1, got_z.end() - 1),
                     make_large_quotient_system<rocsparse_double_complex>(
                         0.5 * std::numeric_limits<double>::max())
                         .want);
    }
}

class PrecondPentadiagonal : public HandleTest
{
};

TEST_F(PrecondPentadiagonal, order3_does_not_write_before_x)
{
    const rocsparse_gpsv_interleaved_alg algs[]
        = {rocsparse_gpsv_interleaved_alg_default, rocsparse_gpsv_interleaved_alg_qr};

    for(const auto alg : algs)
    {
        // The reported case: one complex order-3 system, both algorithms.
        run_case<rocsparse_float_complex>(handle, alg, 3, 1, 1, true, true);
        run_case<rocsparse_double_complex>(handle, alg, 3, 1, 1, true, true);
        // Several batches, and a stride longer than the batch, so every gid
        // writes before x when the m - 4 update is not guarded.
        run_case<rocsparse_float_complex>(handle, alg, 3, 3, 5, true, true);
        run_case<rocsparse_double_complex>(handle, alg, 3, 3, 5, true, true);
        // A diagonal complex order-3 matrix takes the same back-substitution.
        // Its Givens rotations have a zero radius, which is a separate defect,
        // so only the memory around x and the diagonals is checked here.
        run_case<rocsparse_float_complex>(handle, alg, 3, 1, 1, false, false);
        run_case<rocsparse_double_complex>(handle, alg, 3, 2, 2, false, false);
    }

    // Real precisions use the Householder kernel, whose back-substitution is
    // already bounded. Order 3 and the order-4 row that the complex kernel
    // updates explicitly must still solve, and must leave the prefix alone.
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        run_case<T>(handle, rocsparse_gpsv_interleaved_alg_default, 3, 2, 4, true, true);
        run_case<T>(handle, rocsparse_gpsv_interleaved_alg_qr, 4, 2, 2, true, true);
        if constexpr(!is_cplx<T>)
            run_case<T>(handle, rocsparse_gpsv_interleaved_alg_default, 3, 1, 1, false, true);
    });
}
