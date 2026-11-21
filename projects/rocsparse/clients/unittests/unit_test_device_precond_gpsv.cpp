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

// Pentadiagonal interleaved-batch solve (gpsv_interleaved_batch): the write-before-x
// regression of the complex QR kernel, and the sign of the real Householder reflector.
//
// The complex QR kernel used to update row m - 4 unconditionally. For the smallest
// accepted order, m == 3, that row is index -batch_stride, so the kernel wrote one
// value before x. Each array passed to the solve is preceded by a sentinel prefix of
// batch_stride entries so that write stays inside the allocation and is observable.

#include "unit_test_device_precond_common.hpp"

#include <string>

using namespace precond_ut;

namespace
{
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
    // main diagonal. Imaginary parts are zero for real types. `diag` (when not zero)
    // replaces the main diagonal 4 + 0.25 * batch.
    template <typename T>
    band_t make_band(int32_t m, int32_t batch, bool offdiag, double diag = 0.0)
    {
        band_t a;
        a.m = m;
        a.ds.resize(m);
        a.dl.resize(m);
        a.d.assign(m, diag != 0.0 ? diag : 4.0 + 0.25 * batch);
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
        return re(v) == 77.0 && im(v) == 0.0;
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

    // `diag` (when not zero) is the main diagonal of every system, see make_band.
    template <typename T>
    void run_case(rocsparse_handle               handle,
                  rocsparse_gpsv_interleaved_alg alg,
                  int32_t                        m,
                  int32_t                        batch_count,
                  int32_t                        batch_stride,
                  bool                           offdiag,
                  bool                           check_solution,
                  double                         diag = 0.0)
    {
        SCOPED_TRACE(std::string(is_cplx<T> ? (is_single_precision<T> ? "c" : "z")
                                            : (is_single_precision<T> ? "s" : "d"))
                     + " alg=" + std::to_string(static_cast<int>(alg)) + " m=" + std::to_string(m)
                     + " batch=" + std::to_string(batch_count) + " stride="
                     + std::to_string(batch_stride) + (offdiag ? " pentadiagonal" : " diagonal")
                     + (diag != 0.0 ? " diag=" + std::to_string(diag) : ""));

        band_t layout;
        layout.m            = m;
        layout.batch_count  = batch_count;
        layout.batch_stride = batch_stride;
        layout.prefix       = batch_stride;

        const size_t   n = static_cast<size_t>(layout.prefix + m * batch_stride + batch_stride);
        std::vector<T> hds(n, make_value<T>(77.0));
        std::vector<T> hdl = hds, hd = hds, hdu = hds, hdw = hds, hx = hds;
        std::vector<std::vector<T>> want(batch_count);

        for(int32_t b = 0; b < batch_count; ++b)
        {
            const band_t   band = make_band<T>(m, b, offdiag, diag);
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

// The Householder kernel of the real precisions reflects the column (d_i, dl_i+1, ds_i+2)
// onto -sign(d_i) * |column| e_1, i.e. the first component of the reflector is
//   a_ii = d_i + sign(d_i) * sqrt(d_i^2 + dl_i+1^2 + ds_i+2^2),
// the sign that adds the two terms. The other sign cancels, and cancels completely once
// the off-diagonal part is below the rounding of d_i^2: for d_i = 2^13 (float) the sum
// 2^26 + 0.29 rounds to 2^26 (ulp 8), its square root is exactly 2^13 and the cancelling
// choice gives a_ii = 2^13 - 2^13 = 0, so dividing the reflector by a_ii produces infinite
// and then NaN entries. The stable choice gives a_ii = 2^14 and an accurate solve of a
// system that is almost diagonal. In double, d_i = 2^27 plays the same role (2^54 + 0.29
// rounds to 2^54, ulp 4).
TEST_F(PrecondPentadiagonal, householder_reflector_sign_avoids_cancellation)
{
    const double diag_f = 8192.0; // 2^13: 2^26 + 0.29 rounds to 2^26 in float
    const double diag_d = 134217728.0; // 2^27: 2^54 + 0.29 rounds to 2^54 in double
    // The dispatcher ignores `alg` for the real types (both take the Householder
    // kernel); both values are passed so the test survives a dispatcher that honours it.
    for(const auto alg :
        {rocsparse_gpsv_interleaved_alg_default, rocsparse_gpsv_interleaved_alg_qr})
    {
        run_case<float>(handle, alg, 3, 1, 1, true, true, diag_f);
        run_case<float>(handle, alg, 7, 3, 4, true, true, diag_f);
        run_case<double>(handle, alg, 3, 1, 1, true, true, diag_d);
        run_case<double>(handle, alg, 7, 3, 4, true, true, diag_d);
    }
}
