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

// Iterative ILU(0) (csritilu0) coverage:
//  * the (deprecated) rocsparse_Xcsritilu0_compute entry points, for every
//    algorithm and value type, checked against the exact ILU(0) factor;
//  * the async_split driver of the "x" interface, driven directly through
//    csritilu0x_compute_impl (which has no public wrapper) with hand-built L
//    (rows), U (columns) and D structures for every nnz-per-row bucket of the
//    sweep kernels.

#include "unit_test_device_precond_common.hpp"

#include "rocsparse_csritilu0x_buffer_size.hpp"
#include "rocsparse_csritilu0x_compute.hpp"
#include "rocsparse_csritilu0x_preprocess.hpp"

#include <algorithm>
#include <string>
#include <utility>

using namespace precond_ut;

namespace
{
    // Diagonally dominant band matrix with half bandwidth `w` (w >= m gives a
    // dense matrix). The strong diagonal makes the fixed-point iterations converge fast.
    template <typename T>
    csr_matrix<T> make_band_matrix(int32_t m, int32_t w)
    {
        csr_matrix<T> a;
        a.n = m;
        a.ptr.push_back(0);
        for(int32_t i = 0; i < m; ++i)
        {
            for(int32_t j = std::max(0, i - w); j <= std::min(m - 1, i + w); ++j)
            {
                a.ind.push_back(j);
                if(i == j)
                {
                    a.val.push_back(make_value<T>(4.0 + 1.6 * std::min(w, m)));
                }
                else
                {
                    const double r = -0.3 * (1.0 + 0.1 * ((i + 2 * j) % 3));
                    const double c = is_cplx<T> ? 0.05 * ((i - j) % 4) : 0.0;
                    a.val.push_back(make_value<T>(r * (j < i ? 1.1 : 1.0), c));
                }
            }
            a.ptr.push_back(static_cast<int32_t>(a.ind.size()));
        }
        return a;
    }

    // ------------------------------------------------------------------
    // Public (deprecated) compute entry points
    // ------------------------------------------------------------------
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    template <typename T, typename... A>
    rocsparse_status plain_compute(A&&... args)
    {
        return PRECOND_CALL(T, csritilu0_compute, std::forward<A>(args)...);
    }
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

    template <typename T, typename... A>
    rocsparse_status history(A&&... args)
    {
        return PRECOND_CALL(T, csritilu0_history, std::forward<A>(args)...);
    }

    template <typename T, typename... A>
    rocsparse_status compute_ex(A&&... args)
    {
        return PRECOND_CALL(T, csritilu0_compute_ex, std::forward<A>(args)...);
    }

    // The sweeps stop at a fixed count or on the residual criterion; the error left in
    // the factor differs (see the measured values in the tests).
    constexpr bool is_stopping(rocsparse_int options)
    {
        return (options & rocsparse_itilu0_option_stopping_criteria) != 0;
    }

    // Relative (per entry) tolerance on the factor, about 10x the largest error measured on
    // gfx1201 over every algorithm, type and matrix of this file:
    //  * fixed sweep count (60-80 sweeps, the iteration has converged to rounding):
    //    float/cfloat <= 1.6e-7, double/cdouble <= 8.7e-16 -> 1.6e-6, 1e-14 (the double
    //    bound is only 10x the rounding level of a direct factorization);
    //  * stopping criterion (tol 1e-6 float, 1e-12 double): the error is the residual
    //    left at the stop: float <= 2.1e-5 (async_inplace; <= 5e-7 for the others),
    //    double <= 2.4e-12 -> 2.1e-4, 2.4e-11.
    //
    // The converged factor still differs from the host one by the rounding of the
    // row sums, which grows with the row length and depends on the summation order
    // (wave32 and wave64 kernels, async scheduling), so the bound never goes below
    // eps * row_length (dense 100 x 100 rows: 2.2e-14 in double against a measured
    // 1.5e-15 on gfx1201).
    template <typename T>
    double iterative_tolerance(bool stopping, int32_t row_length)
    {
        const double fixed = is_single_precision<T> ? (stopping ? 2.1e-4 : 1.6e-6)
                                                    : (stopping ? 2.4e-11 : 1.0e-14);
        return std::max(fixed, epsilon_of<T>() * static_cast<double>(row_length));
    }

    // Factorization of `a` through rocsparse_Xcsritilu0_compute, compared with
    // the exact ILU(0) factor.
    template <typename T>
    void check_plain_compute(rocsparse_handle     handle,
                             rocsparse_itilu0_alg alg,
                             const csr_matrix<T>& a,
                             rocsparse_int        options,
                             rocsparse_int        nmaxiter_in,
                             const std::string&   what)
    {
        SCOPED_TRACE(what);
        const rocsparse_int        m        = a.n;
        const rocsparse_int        nnz      = a.nnz();
        const rocsparse_index_base base     = rocsparse_index_base_zero;
        rocsparse_int              nmaxiter = nmaxiter_in;

        device_vector<rocsparse_int> drow{as_rocsparse_int(a.ptr)};
        device_vector<rocsparse_int> dcol{as_rocsparse_int(a.ind)};
        device_vector<T>             dval{a.val};
        device_vector<T>             dilu{a.val}; // starting iterate
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr && dilu.ptr);

        size_t buffer_size = 0;
        ASSERT_EQ(
            rocsparse_csritilu0_buffer_size(
                handle, alg, options, nmaxiter, m, nnz, drow, dcol, base, dt_of<T>(), &buffer_size),
            rocsparse_status_success);
        device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(rocsparse_csritilu0_preprocess(handle,
                                                 alg,
                                                 options,
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
        const real_of<T> tol = std::is_same_v<real_of<T>, float> ? 1.0e-6f : 1.0e-12;
        ASSERT_EQ(plain_compute<T>(handle,
                                   alg,
                                   options,
                                   &nmaxiter,
                                   tol,
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
        EXPECT_GE(nmaxiter, 1);
        EXPECT_LE(nmaxiter, nmaxiter_in);
        expect_close(to_host(dilu),
                     host_ilu0(a),
                     iterative_tolerance<T>(is_stopping(options), max_row_length(a)));

        if((options & rocsparse_itilu0_option_convergence_history) != 0)
        {
            // The log holds `niter` residual norms (after `niter` entries reserved for the
            // correction norms, which were not requested).
            rocsparse_int           niter = -1;
            std::vector<real_of<T>> log(2 * static_cast<size_t>(nmaxiter_in), real_of<T>(-1));
            ASSERT_EQ(history<T>(handle, alg, &niter, log.data(), buffer_size, buffer.ptr),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            // The history is what the compute call reported, each residual norm is a finite
            // non-negative number, the iteration reduced the residual, and a stopping run
            // ended early on one of the library's stopping rules.
            EXPECT_EQ(niter, nmaxiter);
            ASSERT_GE(niter, 1);
            ASSERT_LE(niter, nmaxiter_in);
            for(rocsparse_int i = 0; i < niter; ++i)
            {
                EXPECT_TRUE(std::isfinite(log[niter + i])) << "iteration " << i;
                EXPECT_GE(log[niter + i], real_of<T>(0)) << "iteration " << i;
            }
            EXPECT_LT(log[2 * niter - 1], log[niter]);
            if(is_stopping(options))
            {
                // The drivers stop either on the tolerance or on stagnation: after more than
                // 4 sweeps, when the relative change of the residual is below 1e-5 (float)
                // or 1e-15 (double) and the residual is below a limit that depends on the
                // driver: < 1e-5 for async_inplace (and default), <= 10 * tol for the x
                // drivers behind async_split, sync_split and sync_split_fusion.
                EXPECT_LT(nmaxiter, nmaxiter_in);
                const real_of<T> last    = log[2 * niter - 1];
                const real_of<T> delta   = std::abs(last - log[2 * niter - 2]);
                const real_of<T> inc     = std::is_same_v<real_of<T>, float> ? 1.0e-5f : 1.0e-15;
                const bool       inplace = alg == rocsparse_itilu0_alg_default
                                     || alg == rocsparse_itilu0_alg_async_inplace;
                const bool stagnated
                    = niter > 4
                      && (inplace ? (delta < inc * last && static_cast<double>(last) < 1.0e-5)
                                  : (delta <= inc * last && last <= tol * 10));
                EXPECT_TRUE(last <= tol || stagnated)
                    << "residual " << last << " after " << niter << " sweeps";
            }
        }
    }

    // ------------------------------------------------------------------
    // x interface: async_split driven directly
    // ------------------------------------------------------------------
    // L (strictly lower, by rows), U (strictly upper, by columns) and D of a CSR
    // matrix, with the position of every entry in the CSR arrays.
    template <typename T>
    struct split_matrix
    {
        std::vector<rocsparse_int> lptr, lind, lpos;
        std::vector<rocsparse_int> uptr, uind, upos;
        std::vector<rocsparse_int> dpos;

        explicit split_matrix(const csr_matrix<T>& a)
        {
            const int32_t n = a.n;
            lptr.assign(1, 0);
            dpos.assign(static_cast<size_t>(n), -1);
            std::vector<std::vector<std::pair<int32_t, int32_t>>> cols(static_cast<size_t>(n));
            for(int32_t i = 0; i < n; ++i)
            {
                for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                {
                    const int32_t j = a.ind[p];
                    if(j < i)
                    {
                        lind.push_back(j);
                        lpos.push_back(p);
                    }
                    else if(j > i)
                    {
                        cols[static_cast<size_t>(j)].push_back({i, p});
                    }
                    else
                    {
                        dpos[static_cast<size_t>(i)] = p;
                    }
                }
                lptr.push_back(static_cast<rocsparse_int>(lind.size()));
            }
            uptr.assign(1, 0);
            for(int32_t j = 0; j < n; ++j)
            {
                for(const auto& e : cols[static_cast<size_t>(j)])
                {
                    uind.push_back(e.first);
                    upos.push_back(e.second);
                }
                uptr.push_back(static_cast<rocsparse_int>(uind.size()));
            }
        }
    };

    template <typename T>
    std::vector<T> gather(const std::vector<T>& v, const std::vector<rocsparse_int>& pos)
    {
        std::vector<T> out;
        out.reserve(pos.size());
        for(const rocsparse_int p : pos)
            out.push_back(v[static_cast<size_t>(p)]);
        return out;
    }

    struct x_options
    {
        rocsparse_int options   = rocsparse_itilu0_option_compute_nrm_residual;
        rocsparse_int nmaxiter  = 80;
        rocsparse_int nfreeiter = 0;
    };

    template <typename T>
    void check_async_split_x(rocsparse_handle     handle,
                             const csr_matrix<T>& a,
                             const x_options&     opt,
                             const std::string&   what)
    {
        SCOPED_TRACE(what);
        using I = rocsparse_int;
        using J = rocsparse_int;

        const split_matrix<T> split(a);
        const I               m    = a.n;
        const I               nnz  = a.nnz();
        const I               lnnz = static_cast<I>(split.lind.size());
        const I               unnz = static_cast<I>(split.uind.size());
        // The starting iterate is the matching part of A.
        const std::vector<T> l0 = gather(a.val, split.lpos);
        const std::vector<T> u0 = gather(a.val, split.upos);
        const std::vector<T> d0 = gather(a.val, split.dpos);

        device_vector<I> dptr{std::vector<I>(a.ptr.begin(), a.ptr.end())};
        device_vector<J> dind{std::vector<J>(a.ind.begin(), a.ind.end())};
        device_vector<T> dval{a.val};
        device_vector<I> dlptr{split.lptr};
        device_vector<J> dlind{split.lind};
        device_vector<T> dlval{l0};
        device_vector<I> duptr{split.uptr};
        device_vector<J> duind{split.uind};
        device_vector<T> duval{u0};
        device_vector<T> ddval{d0};
        ASSERT_TRUE(dptr.ptr && dind.ptr && dval.ptr && dlptr.ptr && dlind.ptr && dlval.ptr
                    && duptr.ptr && duind.ptr && duval.ptr && ddval.ptr);

        const auto ldiag = rocsparse_diag_type_unit;
        const auto udiag = rocsparse_diag_type_non_unit;
        const auto ldir  = rocsparse_direction_row;
        const auto udir  = rocsparse_direction_column;
        const auto base  = rocsparse_index_base_zero;
        const auto alg   = rocsparse_itilu0_alg_async_split;

        size_t buffer_size = 0;
        ASSERT_EQ((rocsparse::csritilu0x_buffer_size_impl<I, J>(handle,
                                                                alg,
                                                                opt.options,
                                                                opt.nmaxiter,
                                                                m,
                                                                nnz,
                                                                dptr.ptr,
                                                                dptr.ptr + 1,
                                                                dind.ptr,
                                                                base,
                                                                ldiag,
                                                                ldir,
                                                                udiag,
                                                                udir,
                                                                dt_of<T>(),
                                                                &buffer_size)),
                  rocsparse_status_success);
        device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);

        ASSERT_EQ((rocsparse::csritilu0x_preprocess_impl<I, J>(handle,
                                                               alg,
                                                               opt.options,
                                                               opt.nmaxiter,
                                                               m,
                                                               nnz,
                                                               dptr.ptr,
                                                               dptr.ptr + 1,
                                                               dind.ptr,
                                                               base,
                                                               ldiag,
                                                               ldir,
                                                               lnnz,
                                                               dlptr.ptr,
                                                               dlptr.ptr + 1,
                                                               dlind.ptr,
                                                               base,
                                                               udiag,
                                                               udir,
                                                               unnz,
                                                               duptr.ptr,
                                                               duptr.ptr + 1,
                                                               duind.ptr,
                                                               base,
                                                               dt_of<T>(),
                                                               buffer_size,
                                                               buffer.ptr)),
                  rocsparse_status_success);

        J                nsweeps = opt.nmaxiter;
        const real_of<T> tol     = std::is_same_v<real_of<T>, float> ? 1.0e-6f : 1.0e-12;
        const bool       verbose = (opt.options & rocsparse_itilu0_option_verbose) != 0;
        if(verbose)
        {
            testing::internal::CaptureStdout();
        }
        const rocsparse_status status = rocsparse::csritilu0x_compute_impl<T, I, J>(handle,
                                                                                    alg,
                                                                                    opt.options,
                                                                                    &nsweeps,
                                                                                    opt.nfreeiter,
                                                                                    tol,
                                                                                    m,
                                                                                    nnz,
                                                                                    dptr.ptr,
                                                                                    dptr.ptr + 1,
                                                                                    dind.ptr,
                                                                                    dval.ptr,
                                                                                    base,
                                                                                    ldiag,
                                                                                    ldir,
                                                                                    lnnz,
                                                                                    dlptr.ptr,
                                                                                    dlptr.ptr + 1,
                                                                                    dlind.ptr,
                                                                                    dlval.ptr,
                                                                                    base,
                                                                                    udiag,
                                                                                    udir,
                                                                                    unnz,
                                                                                    duptr.ptr,
                                                                                    duptr.ptr + 1,
                                                                                    duind.ptr,
                                                                                    duval.ptr,
                                                                                    base,
                                                                                    ddval.ptr,
                                                                                    buffer_size,
                                                                                    buffer.ptr);
        std::string            log_text;
        if(verbose)
        {
            log_text = testing::internal::GetCapturedStdout();
        }
        ASSERT_EQ(status, rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_GE(nsweeps, 1);
        EXPECT_LE(nsweeps, opt.nmaxiter);

        if(verbose)
        {
            // One header line per sweep. The residual and rate columns of this driver are
            // not meaningful (residual 0, rate nan), so only the table is checked.
            size_t headers = 0;
            for(size_t at = log_text.find("residual"); at != std::string::npos;
                at        = log_text.find("residual", at + 1))
                ++headers;
            EXPECT_EQ(headers, static_cast<size_t>(nsweeps));
        }

        const std::vector<T> ref = host_ilu0(a);
        const double rel_tol = iterative_tolerance<T>(is_stopping(opt.options), max_row_length(a));
        expect_close(to_host(dlval), gather(ref, split.lpos), rel_tol);
        expect_close(to_host(duval), gather(ref, split.upos), rel_tol);
        expect_close(to_host(ddval), gather(ref, split.dpos), rel_tol);
    }

    // The four algorithms of the public interface.
    constexpr rocsparse_itilu0_alg algorithms[] = {rocsparse_itilu0_alg_default,
                                                   rocsparse_itilu0_alg_async_inplace,
                                                   rocsparse_itilu0_alg_async_split,
                                                   rocsparse_itilu0_alg_sync_split};

    struct run_out
    {
        rocsparse_status compute_status = rocsparse_status_internal_error;
        rocsparse_int    nmaxiter       = 0;
        std::string      log;
    };

    // Buffer size, analysis and compute of `a` through the public interface with
    // `options`, keeping the buffer for later history queries. The matrix of the
    // device arrays is `a`; `ilu` receives the factor.
    template <typename T>
    struct itilu0_run
    {
        csr_matrix<T>                a;
        device_vector<rocsparse_int> drow;
        device_vector<rocsparse_int> dcol;
        device_vector<T>             dval;
        device_vector<T>             dilu;
        device_vector<char>          buffer;
        size_t                       buffer_size = 0;
        rocsparse_int                options     = 0;
        rocsparse_int                nmaxiter_in = 0;
        rocsparse_itilu0_alg         alg         = rocsparse_itilu0_alg_default;

        void prepare(rocsparse_handle     handle,
                     rocsparse_itilu0_alg alg_,
                     const csr_matrix<T>& a_,
                     rocsparse_int        options_,
                     rocsparse_int        nmaxiter_)
        {
            a           = a_;
            alg         = alg_;
            options     = options_;
            nmaxiter_in = nmaxiter_;
            upload(drow, as_rocsparse_int(a.ptr));
            upload(dcol, as_rocsparse_int(a.ind));
            upload(dval, a.val);
            upload(dilu, a.val); // starting iterate
            ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr && dilu.ptr);
            ASSERT_EQ(rocsparse_csritilu0_buffer_size(handle,
                                                      alg,
                                                      options,
                                                      nmaxiter_,
                                                      a.n,
                                                      a.nnz(),
                                                      drow,
                                                      dcol,
                                                      rocsparse_index_base_zero,
                                                      dt_of<T>(),
                                                      &buffer_size),
                      rocsparse_status_success);
            upload(buffer, std::vector<char>(buffer_size ? buffer_size : size_t{1}));
            ASSERT_TRUE(buffer.ptr);
            ASSERT_EQ(rocsparse_csritilu0_preprocess(handle,
                                                     alg,
                                                     options,
                                                     nmaxiter_,
                                                     a.n,
                                                     a.nnz(),
                                                     drow,
                                                     dcol,
                                                     rocsparse_index_base_zero,
                                                     dt_of<T>(),
                                                     buffer_size,
                                                     buffer.ptr),
                      rocsparse_status_success);
        }

        run_out compute(rocsparse_handle handle, real_of<T> tol, rocsparse_int nfreeiter = 0)
        {
            run_out       out;
            rocsparse_int nmaxiter = nmaxiter_in;
            const bool    verbose  = (options & rocsparse_itilu0_option_verbose) != 0;
            if(verbose)
                testing::internal::CaptureStdout();
            out.compute_status = compute_ex<T>(handle,
                                               alg,
                                               options,
                                               &nmaxiter,
                                               nfreeiter,
                                               tol,
                                               a.n,
                                               a.nnz(),
                                               drow,
                                               dcol,
                                               dval,
                                               dilu,
                                               rocsparse_index_base_zero,
                                               buffer_size,
                                               buffer.ptr);
            if(verbose)
                out.log = testing::internal::GetCapturedStdout();
            (void)hipDeviceSynchronize();
            out.nmaxiter = nmaxiter;
            return out;
        }
    };

    constexpr rocsparse_int nrm_residual   = rocsparse_itilu0_option_compute_nrm_residual;
    constexpr rocsparse_int nrm_correction = rocsparse_itilu0_option_compute_nrm_correction;
    constexpr rocsparse_int history_opt    = rocsparse_itilu0_option_convergence_history;
    constexpr rocsparse_int stopping       = rocsparse_itilu0_option_stopping_criteria;
    constexpr rocsparse_int verbose_opt    = rocsparse_itilu0_option_verbose;

    template <typename T>
    real_of<T> default_tol()
    {
        return std::is_same_v<real_of<T>, float> ? 1.0e-6f : 1.0e-12;
    }

    // The sync_split algorithm of the extended interface: buffer size, analysis and
    // compute with hand-built L, U and D, checked against a host ILU(0). The residual norm
    // is always requested.
    template <typename T>
    void check_sync_split_x(rocsparse_handle     handle,
                            const csr_matrix<T>& a,
                            rocsparse_int        options,
                            rocsparse_int        nmaxiter,
                            const std::string&   what,
                            rocsparse_itilu0_alg alg = rocsparse_itilu0_alg_sync_split)
    {
        SCOPED_TRACE(what);
        using I = rocsparse_int;
        using J = rocsparse_int;

        const split_matrix<T> split(a);
        const I               m    = a.n;
        const I               nnz  = a.nnz();
        const I               lnnz = static_cast<I>(split.lind.size());
        const I               unnz = static_cast<I>(split.uind.size());

        device_vector<I> dptr{std::vector<I>(a.ptr.begin(), a.ptr.end())};
        device_vector<J> dind{std::vector<J>(a.ind.begin(), a.ind.end())};
        device_vector<T> dval{a.val};
        device_vector<I> dlptr{split.lptr};
        device_vector<J> dlind{split.lind};
        device_vector<T> dlval{gather(a.val, split.lpos)};
        device_vector<I> duptr{split.uptr};
        device_vector<J> duind{split.uind};
        device_vector<T> duval{gather(a.val, split.upos)};
        device_vector<T> ddval{gather(a.val, split.dpos)};
        ASSERT_TRUE(dptr.ptr && dind.ptr && dval.ptr && dlptr.ptr && dlind.ptr && dlval.ptr
                    && duptr.ptr && duind.ptr && duval.ptr && ddval.ptr);

        // The diagonal is stored apart (D) only when both factors have a unit diagonal.
        const auto ldiag = rocsparse_diag_type_unit;
        const auto udiag = rocsparse_diag_type_unit;
        const auto ldir  = rocsparse_direction_row;
        const auto udir  = rocsparse_direction_column;
        const auto base  = rocsparse_index_base_zero;

        size_t buffer_size = 0;
        ASSERT_EQ((rocsparse::csritilu0x_buffer_size_impl<I, J>(handle,
                                                                alg,
                                                                options,
                                                                nmaxiter,
                                                                m,
                                                                nnz,
                                                                dptr.ptr,
                                                                dptr.ptr + 1,
                                                                dind.ptr,
                                                                base,
                                                                ldiag,
                                                                ldir,
                                                                udiag,
                                                                udir,
                                                                dt_of<T>(),
                                                                &buffer_size)),
                  rocsparse_status_success);
        device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ((rocsparse::csritilu0x_preprocess_impl<I, J>(handle,
                                                               alg,
                                                               options,
                                                               nmaxiter,
                                                               m,
                                                               nnz,
                                                               dptr.ptr,
                                                               dptr.ptr + 1,
                                                               dind.ptr,
                                                               base,
                                                               ldiag,
                                                               ldir,
                                                               lnnz,
                                                               dlptr.ptr,
                                                               dlptr.ptr + 1,
                                                               dlind.ptr,
                                                               base,
                                                               udiag,
                                                               udir,
                                                               unnz,
                                                               duptr.ptr,
                                                               duptr.ptr + 1,
                                                               duind.ptr,
                                                               base,
                                                               dt_of<T>(),
                                                               buffer_size,
                                                               buffer.ptr)),
                  rocsparse_status_success);

        J          nsweeps = nmaxiter;
        const bool verbose = (options & verbose_opt) != 0;
        if(verbose)
            testing::internal::CaptureStdout();
        const auto  status = rocsparse::csritilu0x_compute_impl<T, I, J>(handle,
                                                                        alg,
                                                                        options,
                                                                        &nsweeps,
                                                                        0,
                                                                        default_tol<T>(),
                                                                        m,
                                                                        nnz,
                                                                        dptr.ptr,
                                                                        dptr.ptr + 1,
                                                                        dind.ptr,
                                                                        dval.ptr,
                                                                        base,
                                                                        ldiag,
                                                                        ldir,
                                                                        lnnz,
                                                                        dlptr.ptr,
                                                                        dlptr.ptr + 1,
                                                                        dlind.ptr,
                                                                        dlval.ptr,
                                                                        base,
                                                                        udiag,
                                                                        udir,
                                                                        unnz,
                                                                        duptr.ptr,
                                                                        duptr.ptr + 1,
                                                                        duind.ptr,
                                                                        duval.ptr,
                                                                        base,
                                                                        ddval.ptr,
                                                                        buffer_size,
                                                                        buffer.ptr);
        std::string log_text;
        if(verbose)
            log_text = testing::internal::GetCapturedStdout();
        ASSERT_EQ(status, rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_GE(nsweeps, 1);
        EXPECT_LE(nsweeps, nmaxiter);
        if(verbose)
            EXPECT_NE(log_text.find("iter"), std::string::npos);

        const std::vector<T> ref = host_ilu0(a);
        const double rel_tol = iterative_tolerance<T>((options & stopping) != 0, max_row_length(a));
        expect_close(to_host(dlval), gather(ref, split.lpos), rel_tol);
        expect_close(to_host(duval), gather(ref, split.upos), rel_tol);
        expect_close(to_host(ddval), gather(ref, split.dpos), rel_tol);
    }

    void check_extended_interface_arguments(rocsparse_handle handle, rocsparse_itilu0_alg alg)
    {
        using I = rocsparse_int;
        using J = rocsparse_int;

        // Tridiagonal 4 x 4 matrix with its strictly lower part by rows (L) and strictly
        // upper part by columns (U).
        const std::vector<I>      ptr{0, 2, 5, 8, 10};
        const std::vector<J>      ind{0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
        const std::vector<I>      lptr{0, 0, 1, 2, 3};
        const std::vector<J>      lind{0, 1, 2};
        const std::vector<I>      uptr{0, 0, 1, 2, 3};
        const std::vector<J>      uind{0, 1, 2};
        const std::vector<double> val{4, -1, -1, 4, -1, -1, 4, -1, -1, 4};
        const std::vector<double> lval{-0.25, -0.25, -0.25};
        const std::vector<double> uval{-1, -1, -1};
        const std::vector<double> dval{4, 4, 4, 4};

        device_vector<I>      dptr{ptr};
        device_vector<J>      dind{ind};
        device_vector<double> dv{val};
        device_vector<I>      dlptr{lptr};
        device_vector<J>      dlind{lind};
        device_vector<double> dlval{lval};
        device_vector<I>      duptr{uptr};
        device_vector<J>      duind{uind};
        device_vector<double> duval{uval};
        device_vector<double> ddval{dval};
        ASSERT_TRUE(dptr.ptr && dind.ptr && dv.ptr && dlptr.ptr && dlind.ptr && dlval.ptr
                    && duptr.ptr && duind.ptr && duval.ptr && ddval.ptr);

        const auto ldiag = rocsparse_diag_type_unit;
        const auto udiag = rocsparse_diag_type_non_unit;
        const auto ldir  = rocsparse_direction_row;
        const auto udir  = rocsparse_direction_column;
        const auto base  = rocsparse_index_base_zero;
        const I    m = 4, nnz = 10, lnnz = 3, unnz = 3;

        const auto bad_diag = static_cast<rocsparse_diag_type>(-1);
        const auto bad_dir  = static_cast<rocsparse_direction>(-1);
        const auto bad_base = static_cast<rocsparse_index_base>(-1);

        auto buffer_size_call = [&](rocsparse_index_base b,
                                    rocsparse_diag_type  ld,
                                    rocsparse_direction  ldr,
                                    rocsparse_diag_type  ud,
                                    rocsparse_direction  udr,
                                    size_t*              size) {
            return rocsparse::csritilu0x_buffer_size_impl<I, J>(handle,
                                                                alg,
                                                                0,
                                                                10,
                                                                m,
                                                                nnz,
                                                                dptr.ptr,
                                                                dptr.ptr + 1,
                                                                dind.ptr,
                                                                b,
                                                                ld,
                                                                ldr,
                                                                ud,
                                                                udr,
                                                                rocsparse_datatype_f64_r,
                                                                size);
        };

        size_t buffer_size = 0;
        ASSERT_EQ(buffer_size_call(base, ldiag, ldir, udiag, udir, &buffer_size),
                  rocsparse_status_success);
        device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);

        // Buffer size.
        size_t tmp = 0;
        EXPECT_NE(buffer_size_call(bad_base, ldiag, ldir, udiag, udir, &tmp),
                  rocsparse_status_success);
        EXPECT_NE(buffer_size_call(base, bad_diag, ldir, udiag, udir, &tmp),
                  rocsparse_status_success);
        EXPECT_NE(buffer_size_call(base, ldiag, bad_dir, udiag, udir, &tmp),
                  rocsparse_status_success);
        EXPECT_NE(buffer_size_call(base, ldiag, ldir, bad_diag, udir, &tmp),
                  rocsparse_status_success);
        EXPECT_NE(buffer_size_call(base, ldiag, ldir, udiag, bad_dir, &tmp),
                  rocsparse_status_success);

        // Analysis.
        auto preprocess_call = [&](rocsparse_index_base b,
                                   rocsparse_index_base lb,
                                   rocsparse_index_base ub,
                                   rocsparse_diag_type  ld,
                                   rocsparse_direction  ldr,
                                   rocsparse_diag_type  ud,
                                   rocsparse_direction  udr,
                                   const J*             lind_ptr,
                                   const J*             uind_ptr) {
            return rocsparse::csritilu0x_preprocess_impl<I, J>(handle,
                                                               alg,
                                                               0,
                                                               10,
                                                               m,
                                                               nnz,
                                                               dptr.ptr,
                                                               dptr.ptr + 1,
                                                               dind.ptr,
                                                               b,
                                                               ld,
                                                               ldr,
                                                               lnnz,
                                                               dlptr.ptr,
                                                               dlptr.ptr + 1,
                                                               lind_ptr,
                                                               lb,
                                                               ud,
                                                               udr,
                                                               unnz,
                                                               duptr.ptr,
                                                               duptr.ptr + 1,
                                                               uind_ptr,
                                                               ub,
                                                               rocsparse_datatype_f64_r,
                                                               buffer_size,
                                                               buffer.ptr);
        };
        EXPECT_NE(
            preprocess_call(bad_base, base, base, ldiag, ldir, udiag, udir, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            preprocess_call(base, bad_base, base, ldiag, ldir, udiag, udir, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            preprocess_call(base, base, bad_base, ldiag, ldir, udiag, udir, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            preprocess_call(base, base, base, bad_diag, ldir, udiag, udir, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            preprocess_call(base, base, base, ldiag, bad_dir, udiag, udir, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            preprocess_call(base, base, base, ldiag, ldir, bad_diag, udir, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            preprocess_call(base, base, base, ldiag, ldir, udiag, bad_dir, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        // A non-empty U or L needs its column indices.
        EXPECT_EQ(preprocess_call(base, base, base, ldiag, ldir, udiag, udir, dlind.ptr, nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(preprocess_call(base, base, base, ldiag, ldir, udiag, udir, nullptr, duind.ptr),
                  rocsparse_status_invalid_pointer);

        // Compute (a valid analysis comes first for the calls that get past the checks).
        ASSERT_EQ(preprocess_call(base, base, base, ldiag, ldir, udiag, udir, dlind.ptr, duind.ptr),
                  rocsparse_status_success);
        auto compute_call = [&](rocsparse_index_base b,
                                rocsparse_index_base lb,
                                rocsparse_index_base ub,
                                rocsparse_diag_type  ld,
                                rocsparse_direction  ldr,
                                rocsparse_diag_type  ud,
                                rocsparse_direction  udr,
                                I                    nz,
                                const J*             lind_ptr,
                                const J*             uind_ptr) {
            J nsweeps = 5;
            return rocsparse::csritilu0x_compute_impl<double, I, J>(
                handle,
                alg,
                rocsparse_itilu0_option_compute_nrm_residual,
                &nsweeps,
                0,
                1.0e-12,
                m,
                nz,
                dptr.ptr,
                dptr.ptr + 1,
                dind.ptr,
                dv.ptr,
                b,
                ld,
                ldr,
                lnnz,
                dlptr.ptr,
                dlptr.ptr + 1,
                lind_ptr,
                dlval.ptr,
                lb,
                ud,
                udr,
                unnz,
                duptr.ptr,
                duptr.ptr + 1,
                uind_ptr,
                duval.ptr,
                ub,
                ddval.ptr,
                buffer_size,
                buffer.ptr);
        };
        EXPECT_NE(
            compute_call(bad_base, base, base, ldiag, ldir, udiag, udir, nnz, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            compute_call(base, bad_base, base, ldiag, ldir, udiag, udir, nnz, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            compute_call(base, base, bad_base, ldiag, ldir, udiag, udir, nnz, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            compute_call(base, base, base, bad_diag, ldir, udiag, udir, nnz, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            compute_call(base, base, base, ldiag, bad_dir, udiag, udir, nnz, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            compute_call(base, base, base, ldiag, ldir, bad_diag, udir, nnz, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_NE(
            compute_call(base, base, base, ldiag, ldir, udiag, bad_dir, nnz, dlind.ptr, duind.ptr),
            rocsparse_status_success);
        EXPECT_EQ(compute_call(base, base, base, ldiag, ldir, udiag, udir, nnz, dlind.ptr, nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(compute_call(base, base, base, ldiag, ldir, udiag, udir, nnz, nullptr, duind.ptr),
                  rocsparse_status_invalid_pointer);
        // Only the row-wise L and the column-wise U are supported.
        EXPECT_NE(compute_call(base,
                               base,
                               base,
                               ldiag,
                               rocsparse_direction_column,
                               udiag,
                               udir,
                               nnz,
                               dlind.ptr,
                               duind.ptr),
                  rocsparse_status_success);
        EXPECT_NE(compute_call(base,
                               base,
                               base,
                               ldiag,
                               ldir,
                               udiag,
                               rocsparse_direction_row,
                               nnz,
                               dlind.ptr,
                               duind.ptr),
                  rocsparse_status_success);
    }
}

class PrecondItilu0 : public HandleTest
{
};

TEST_F(PrecondItilu0, compute_algorithms_and_types)
{
    const rocsparse_int nrm  = rocsparse_itilu0_option_compute_nrm_residual;
    const rocsparse_int stop = rocsparse_itilu0_option_stopping_criteria
                               | rocsparse_itilu0_option_compute_nrm_residual
                               | rocsparse_itilu0_option_convergence_history;
    for(const auto alg : algorithms)
    {
        for_all_types([&](auto tag) {
            using T = decltype(tag);
            // Fixed number of sweeps, then with the stopping criterion.
            check_plain_compute<T>(
                handle, alg, make_band_matrix<T>(24, 2), nrm, 60, "alg " + std::to_string(alg));
            check_plain_compute<T>(handle,
                                   alg,
                                   make_band_matrix<T>(24, 3),
                                   stop,
                                   60,
                                   "stop alg " + std::to_string(alg));
        });
    }
}

TEST_F(PrecondItilu0, compute_quick_returns_and_argument_checks)
{
    const rocsparse_index_base   base     = rocsparse_index_base_zero;
    rocsparse_int                nmaxiter = 5;
    device_vector<rocsparse_int> dptr{std::vector<rocsparse_int>{0, 1}};
    device_vector<rocsparse_int> dind{std::vector<rocsparse_int>{0}};
    device_vector<double>        dval{std::vector<double>{2.0}};
    device_vector<double>        dilu{std::vector<double>{2.0}};
    ASSERT_TRUE(dptr.ptr && dind.ptr && dval.ptr && dilu.ptr);

    const auto alg = rocsparse_itilu0_alg_sync_split;
    // m == 0: success and no sweep performed.
    EXPECT_EQ(plain_compute<double>(handle,
                                    alg,
                                    0,
                                    &nmaxiter,
                                    1.0e-10,
                                    0,
                                    0,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    base,
                                    0,
                                    nullptr),
              rocsparse_status_success);
    EXPECT_EQ(nmaxiter, 0);
    // No entries in a non-empty matrix: the first pivot is missing.
    nmaxiter = 5;
    EXPECT_EQ(
        plain_compute<double>(
            handle, alg, 0, &nmaxiter, 1.0e-10, 3, 0, dptr, dind, dval, dilu, base, 0, nullptr),
        rocsparse_status_zero_pivot);
    EXPECT_EQ(nmaxiter, 0);

    // Argument validation.
    nmaxiter = 5;
    EXPECT_EQ(
        plain_compute<double>(
            nullptr, alg, 0, &nmaxiter, 1.0e-10, 1, 1, dptr, dind, dval, dilu, base, 0, nullptr),
        rocsparse_status_invalid_handle);
    EXPECT_EQ(plain_compute<double>(handle,
                                    static_cast<rocsparse_itilu0_alg>(-5),
                                    0,
                                    &nmaxiter,
                                    1.0e-10,
                                    1,
                                    1,
                                    dptr,
                                    dind,
                                    dval,
                                    dilu,
                                    base,
                                    0,
                                    nullptr),
              rocsparse_status_invalid_value);
    EXPECT_EQ(
        plain_compute<double>(
            handle, alg, -1, &nmaxiter, 1.0e-10, 1, 1, dptr, dind, dval, dilu, base, 0, nullptr),
        rocsparse_status_invalid_value);
    EXPECT_EQ(plain_compute<double>(
                  handle, alg, 0, nullptr, 1.0e-10, 1, 1, dptr, dind, dval, dilu, base, 0, nullptr),
              rocsparse_status_invalid_pointer);
    // A negative tolerance is rejected (the status is not documented).
    EXPECT_NE(plain_compute<double>(
                  handle, alg, 0, &nmaxiter, -1.0, 1, 1, dptr, dind, dval, dilu, base, 0, nullptr),
              rocsparse_status_success);
    EXPECT_EQ(
        plain_compute<double>(
            handle, alg, 0, &nmaxiter, 1.0e-10, -1, 1, dptr, dind, dval, dilu, base, 0, nullptr),
        rocsparse_status_invalid_size);
    EXPECT_EQ(
        plain_compute<double>(
            handle, alg, 0, &nmaxiter, 1.0e-10, 1, -1, dptr, dind, dval, dilu, base, 0, nullptr),
        rocsparse_status_invalid_size);
    EXPECT_EQ(
        plain_compute<double>(
            handle, alg, 0, &nmaxiter, 1.0e-10, 1, 1, nullptr, dind, dval, dilu, base, 0, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(
        plain_compute<double>(
            handle, alg, 0, &nmaxiter, 1.0e-10, 1, 1, dptr, nullptr, dval, dilu, base, 0, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(
        plain_compute<double>(
            handle, alg, 0, &nmaxiter, 1.0e-10, 1, 1, dptr, dind, nullptr, dilu, base, 0, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(
        plain_compute<double>(
            handle, alg, 0, &nmaxiter, 1.0e-10, 1, 1, dptr, dind, dval, nullptr, base, 0, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(plain_compute<double>(handle,
                                    alg,
                                    0,
                                    &nmaxiter,
                                    1.0e-10,
                                    1,
                                    1,
                                    dptr,
                                    dind,
                                    dval,
                                    dilu,
                                    static_cast<rocsparse_index_base>(-1),
                                    0,
                                    nullptr),
              rocsparse_status_invalid_value);
    EXPECT_EQ(
        plain_compute<double>(
            handle, alg, 0, &nmaxiter, 1.0e-10, 1, 1, dptr, dind, dval, dilu, base, 16, nullptr),
        rocsparse_status_invalid_pointer);
}

TEST_F(PrecondItilu0, x_async_split_nnz_per_row_buckets)
{
    // Same selection as the plain driver (mean nnz per row, integer division): half
    // bandwidths 1, 2, 3, 6, 12, 24 and a dense matrix give the means 2, 4, 6, 12, 22, 39
    // and 100, i.e. the widths 2, 4, 8, 16, 32, 64 and the wavefront size.
    const std::pair<int32_t, int32_t> cases[]
        = {{1, 2}, {2, 4}, {3, 6}, {6, 12}, {12, 22}, {24, 39}, {100, 100}};
    x_options opt;
    for(const auto& [w, mean] : cases)
    {
        const int32_t m = (w >= 100) ? 100 : 64;
        const auto    a = make_band_matrix<double>(m, w);
        ASSERT_EQ(a.nnz() / a.n, mean);
        check_async_split_x<double>(handle, a, opt, "w=" + std::to_string(w));
    }
}

TEST_F(PrecondItilu0, x_async_split_options)
{
    // Residual norm, stopping criterion, convergence history and free sweeps.
    x_options opt;
    opt.options = rocsparse_itilu0_option_stopping_criteria
                  | rocsparse_itilu0_option_compute_nrm_residual
                  | rocsparse_itilu0_option_convergence_history;
    check_async_split_x<double>(handle, make_band_matrix<double>(48, 3), opt, "stopping");
    check_async_split_x<float>(handle, make_band_matrix<float>(48, 3), opt, "stopping float");

    opt.nfreeiter = 2;
    check_async_split_x<double>(handle, make_band_matrix<double>(48, 2), opt, "free sweeps");

    // Residual norm only, with the iteration log captured and checked (one table per sweep).
    opt.nfreeiter = 0;
    opt.nmaxiter  = 30;
    opt.options   = rocsparse_itilu0_option_compute_nrm_residual | rocsparse_itilu0_option_verbose;
    check_async_split_x<double>(handle, make_band_matrix<double>(32, 2), opt, "verbose");
}

// The stopping criterion with a tolerance the residual cannot reach in a sensible
// number of sweeps: the iteration ends before the sweep limit only through the
// stagnation test, or runs to the limit. Either way it succeeds with a converged factor.
TEST_F(PrecondItilu0, stopping_criterion_with_unreachable_tolerance)
{
    for(const auto alg : algorithms)
    {
        for_all_types([&](auto tag) {
            using T = decltype(tag);
            SCOPED_TRACE("alg " + std::to_string(alg));
            const auto a = make_band_matrix<T>(40, 2);
            for(const bool verbose : {false, true})
            {
                itilu0_run<T> r;
                r.prepare(handle,
                          alg,
                          a,
                          stopping | nrm_residual | nrm_correction | (verbose ? verbose_opt : 0),
                          200);
                const auto out = r.compute(handle, real_of<T>(1.0e-30));
                ASSERT_EQ(out.compute_status, rocsparse_status_success);
                EXPECT_GE(out.nmaxiter, 1);
                EXPECT_LE(out.nmaxiter, 200);
                expect_close(to_host(r.dilu), host_ilu0(a), iterative_tolerance<T>(true, 5));
            }
        });
    }
}

// The history can only be queried if the history option was given to the compute
// call.
TEST_F(PrecondItilu0, history_without_the_history_option)
{
    for(const auto alg : algorithms)
    {
        SCOPED_TRACE("alg " + std::to_string(alg));
        const auto         a = make_band_matrix<double>(16, 2);
        itilu0_run<double> r;
        r.prepare(handle, alg, a, nrm_residual, 10);
        const auto out = r.compute(handle, 1.0e-12);
        ASSERT_EQ(out.compute_status, rocsparse_status_success);

        rocsparse_int       niter = -1;
        std::vector<double> log(20, -1.0);
        testing::internal::CaptureStderr();
        const rocsparse_status status
            = history<double>(handle, alg, &niter, log.data(), r.buffer_size, r.buffer.ptr);
        (void)testing::internal::GetCapturedStderr();
        // Either the library reports the missing history or it has nothing to report.
        EXPECT_TRUE(status == rocsparse_status_internal_error
                    || (status == rocsparse_status_success && niter == 0))
            << "status " << status << " niter " << niter;
    }
}

// Quick returns on empty problems and argument checks of the buffer size and the
// analysis.
TEST_F(PrecondItilu0, empty_problems)
{
    const rocsparse_index_base   base = rocsparse_index_base_zero;
    device_vector<rocsparse_int> dptr{std::vector<rocsparse_int>{0, 0, 0, 0}};
    device_vector<rocsparse_int> dind{std::vector<rocsparse_int>{0}};
    ASSERT_TRUE(dptr.ptr && dind.ptr);
    for(const auto alg : algorithms)
    {
        SCOPED_TRACE("alg " + std::to_string(alg));
        size_t buffer_size = 12345;
        // No rows: nothing to do.
        EXPECT_EQ(
            rocsparse_csritilu0_buffer_size(
                handle, alg, 0, 10, 0, 0, dptr, dind, base, rocsparse_datatype_f64_r, &buffer_size),
            rocsparse_status_success);
        EXPECT_EQ(buffer_size, 0u);
        EXPECT_EQ(
            rocsparse_csritilu0_preprocess(
                handle, alg, 0, 10, 0, 0, dptr, dind, base, rocsparse_datatype_f64_r, 0, nullptr),
            rocsparse_status_success);

        // Rows but no entries: the diagonal is missing (documented as a zero pivot).
        buffer_size = 12345;
        EXPECT_EQ(
            rocsparse_csritilu0_buffer_size(
                handle, alg, 0, 10, 3, 0, dptr, dind, base, rocsparse_datatype_f64_r, &buffer_size),
            rocsparse_status_zero_pivot);
        EXPECT_EQ(
            rocsparse_csritilu0_preprocess(
                handle, alg, 0, 10, 3, 0, dptr, dind, base, rocsparse_datatype_f64_r, 0, nullptr),
            rocsparse_status_zero_pivot);

        // Invalid enumerations.
        EXPECT_EQ(rocsparse_csritilu0_buffer_size(handle,
                                                  alg,
                                                  0,
                                                  10,
                                                  3,
                                                  1,
                                                  dptr,
                                                  dind,
                                                  static_cast<rocsparse_index_base>(-1),
                                                  rocsparse_datatype_f64_r,
                                                  &buffer_size),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_csritilu0_preprocess(handle,
                                                 alg,
                                                 0,
                                                 10,
                                                 3,
                                                 1,
                                                 dptr,
                                                 dind,
                                                 static_cast<rocsparse_index_base>(-1),
                                                 rocsparse_datatype_f64_r,
                                                 0,
                                                 nullptr),
                  rocsparse_status_invalid_value);
    }
}

// The extended interface checks its enumerations (the public interface fixes them).
// A bad enumeration, a missing pointer for a non-empty L or U, or an unsupported
// direction must be rejected before any work is done.
TEST_F(PrecondItilu0, extended_interface_rejects_invalid_arguments)
{
    for(const auto alg : {rocsparse_itilu0_alg_async_split, rocsparse_itilu0_alg_sync_split})
    {
        SCOPED_TRACE("alg " + std::to_string(alg));
        check_extended_interface_arguments(handle, alg);
    }
}

TEST_F(PrecondItilu0, extended_interface_buffer_size_empty_and_unsupported_types)
{
    using I = rocsparse_int;
    using J = rocsparse_int;
    const std::vector<I> ptr{0, 2, 5, 8, 10};
    device_vector<I>     dptr{ptr};
    device_vector<J>     dind{std::vector<J>{0, 1, 0, 1, 2, 1, 2, 3, 2, 3}};
    ASSERT_TRUE(dptr.ptr && dind.ptr);
    for(const auto alg : {rocsparse_itilu0_alg_async_split, rocsparse_itilu0_alg_sync_split})
    {
        SCOPED_TRACE("alg " + std::to_string(alg));
        auto size_of = [&](I m, I nnz, rocsparse_datatype dtype, size_t* size) {
            return rocsparse::csritilu0x_buffer_size_impl<I, J>(
                handle,
                alg,
                rocsparse_itilu0_option_compute_nrm_residual,
                10,
                m,
                nnz,
                dptr.ptr,
                dptr.ptr + 1,
                dind.ptr,
                rocsparse_index_base_zero,
                rocsparse_diag_type_unit,
                rocsparse_direction_row,
                rocsparse_diag_type_non_unit,
                rocsparse_direction_column,
                dtype,
                size);
        };
        size_t size = 777;
        // Nothing to factorize: no buffer is needed.
        EXPECT_EQ(size_of(0, 0, rocsparse_datatype_f64_r, &size), rocsparse_status_success);
        EXPECT_EQ(size, 0u);
        // A non-empty matrix without entries has no diagonal: only require a clean return.
        size = 777;
        (void)size_of(4, 0, rocsparse_datatype_f64_r, &size);
        // The four floating point types need a buffer, the other types are not supported.
        for(const auto dtype : {rocsparse_datatype_f32_r,
                                rocsparse_datatype_f64_r,
                                rocsparse_datatype_f32_c,
                                rocsparse_datatype_f64_c})
        {
            size = 0;
            EXPECT_EQ(size_of(4, 10, dtype, &size), rocsparse_status_success);
            EXPECT_GT(size, 0u);
        }
        for(const auto dtype : {rocsparse_datatype_i8_r,
                                rocsparse_datatype_u8_r,
                                rocsparse_datatype_i32_r,
                                rocsparse_datatype_u32_r,
                                rocsparse_datatype_f16_r,
                                rocsparse_datatype_bf16_r})
        {
            EXPECT_NE(size_of(4, 10, dtype, &size), rocsparse_status_success);
        }
    }
}

TEST_F(PrecondItilu0, sync_split_extended_interface_sweeps)
{
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        // Fixed sweeps, then the stopping criterion with the verbose table and the
        // correction norms.
        check_sync_split_x<T>(handle, make_band_matrix<T>(24, 3), nrm_residual, 60, "fixed");
        check_sync_split_x<T>(handle,
                              make_band_matrix<T>(24, 3),
                              stopping | nrm_residual | nrm_correction | history_opt | verbose_opt,
                              60,
                              "stopping");
    });
    // The sweep kernel width follows the mean number of entries per row.
    for(const auto& [w, m] :
        {std::pair<int32_t, int32_t>{1, 64}, {2, 64}, {6, 64}, {12, 64}, {24, 64}, {100, 100}})
    {
        check_sync_split_x<double>(
            handle, make_band_matrix<double>(m, w), nrm_residual, 80, "w=" + std::to_string(w));
    }
}

// The factorization does not depend on where the temporary storage of the library lives.
TEST_F(PrecondItilu0, temporary_storage_when_the_handle_buffer_is_small)
{
    ScopedOverride<size_t> shrink(handle->buffer_size, 1);
    for(const auto alg : algorithms)
    {
        for_all_types([&](auto tag) {
            using T = decltype(tag);
            SCOPED_TRACE("alg " + std::to_string(alg));
            const auto    a = make_band_matrix<T>(40, 4);
            itilu0_run<T> r;
            r.prepare(handle, alg, a, nrm_residual, 40);
            const auto out = r.compute(handle, default_tol<T>());
            ASSERT_EQ(out.compute_status, rocsparse_status_success);
            expect_close(to_host(r.dilu), host_ilu0(a), iterative_tolerance<T>(false, 9));
        });
    }
}

// The sweep kernel of the public interface follows the mean number of entries per row
// (2, 4, 8, 16, 32, 64 and above): a tridiagonal matrix and dense matrices of order 4 to
// 65, with a free sweep, the COO option and the verbose table.
TEST_F(PrecondItilu0, public_interface_sweep_kernels)
{
    const auto check = [&](rocsparse_itilu0_alg alg, int32_t m, int32_t w, rocsparse_int extra) {
        SCOPED_TRACE("alg " + std::to_string(alg) + " m " + std::to_string(m) + " w "
                     + std::to_string(w));
        const auto         a = make_band_matrix<double>(m, w);
        itilu0_run<double> r;
        r.prepare(
            handle, alg, a, stopping | nrm_residual | nrm_correction | history_opt | extra, 80);
        const auto out = r.compute(handle, default_tol<double>(), 1);
        ASSERT_EQ(out.compute_status, rocsparse_status_success);
        // The log holds the correction norms, then the residual norms, of every sweep.
        rocsparse_int       niter = -1;
        std::vector<double> log(2 * 80, -1.0);
        ASSERT_EQ(history<double>(handle, alg, &niter, log.data(), r.buffer_size, r.buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(niter, out.nmaxiter);
        expect_close(
            to_host(r.dilu), host_ilu0(a), iterative_tolerance<double>(true, max_row_length(a)));
        ASSERT_GE(niter, 1);
        ASSERT_LE(niter, 80);
        for(rocsparse_int i = 0; i < 2 * niter; ++i)
        {
            EXPECT_TRUE(std::isfinite(log[i])) << "entry " << i;
            EXPECT_GE(log[i], 0.0) << "entry " << i;
        }
        if(niter > 1)
            EXPECT_LT(log[2 * niter - 1], log[niter]);
        if(niter < 80)
            EXPECT_LE(log[2 * niter - 1], default_tol<double>());
    };
    const auto inplace = rocsparse_itilu0_alg_async_inplace;
    check(inplace, 4, 1, 0);
    for(const int32_t m : {4, 8, 16, 32, 64, 65})
        check(inplace, m, m, 0);
    check(inplace, 4, 1, rocsparse_itilu0_option_coo_format);
    check(inplace, 8, 8, rocsparse_itilu0_option_coo_format);
    check(rocsparse_itilu0_alg_sync_split, 4, 4, verbose_opt);
    check(rocsparse_itilu0_alg_sync_split, 65, 65, 0);
}

// The extended interface only implements the split algorithms; the others are refused
// when computing, and the unsupported value types are refused when sizing the buffer.
TEST_F(PrecondItilu0, extended_interface_with_the_unimplemented_algorithms)
{
    using I = rocsparse_int;
    using J = rocsparse_int;
    const std::vector<I>  ptr{0, 2, 5, 8, 10};
    const std::vector<J>  ind{0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    const std::vector<I>  lptr{0, 0, 1, 2, 3};
    const std::vector<J>  lind{0, 1, 2};
    const std::vector<I>  uptr{0, 0, 1, 2, 3};
    const std::vector<J>  uind{0, 1, 2};
    device_vector<I>      dptr{ptr};
    device_vector<J>      dind{ind};
    device_vector<I>      dlptr{lptr};
    device_vector<J>      dlind{lind};
    device_vector<I>      duptr{uptr};
    device_vector<J>      duind{uind};
    device_vector<double> dval{std::vector<double>{4, -1, -1, 4, -1, -1, 4, -1, -1, 4}};
    device_vector<double> dlval{std::vector<double>(3, 0.0)};
    device_vector<double> duval{std::vector<double>(3, 0.0)};
    device_vector<double> ddval{std::vector<double>(4, 0.0)};
    ASSERT_TRUE(dptr.ptr && dind.ptr && dlptr.ptr && dlind.ptr && duptr.ptr && duind.ptr && dval.ptr
                && dlval.ptr && duval.ptr && ddval.ptr);

    for(const auto alg : {rocsparse_itilu0_alg_default, rocsparse_itilu0_alg_async_inplace})
    {
        SCOPED_TRACE("alg " + std::to_string(alg));

        size_t size = 0;
        for(const auto dtype : {rocsparse_datatype_i8_r,
                                rocsparse_datatype_u8_r,
                                rocsparse_datatype_i32_r,
                                rocsparse_datatype_u32_r,
                                rocsparse_datatype_f16_r,
                                rocsparse_datatype_bf16_r})
        {
            EXPECT_NE((rocsparse::csritilu0x_buffer_size_impl<I, J>(
                          handle,
                          alg,
                          rocsparse_itilu0_option_compute_nrm_residual,
                          10,
                          4,
                          10,
                          dptr.ptr,
                          dptr.ptr + 1,
                          dind.ptr,
                          rocsparse_index_base_zero,
                          rocsparse_diag_type_unit,
                          rocsparse_direction_row,
                          rocsparse_diag_type_unit,
                          rocsparse_direction_column,
                          dtype,
                          &size)),
                      rocsparse_status_success);
        }

        J nsweeps = 10;
        EXPECT_NE((rocsparse::csritilu0x_compute_impl<double, I, J>(handle,
                                                                    alg,
                                                                    0,
                                                                    &nsweeps,
                                                                    0,
                                                                    1.0e-12,
                                                                    4,
                                                                    10,
                                                                    dptr.ptr,
                                                                    dptr.ptr + 1,
                                                                    dind.ptr,
                                                                    dval.ptr,
                                                                    rocsparse_index_base_zero,
                                                                    rocsparse_diag_type_unit,
                                                                    rocsparse_direction_row,
                                                                    3,
                                                                    dlptr.ptr,
                                                                    dlptr.ptr + 1,
                                                                    dlind.ptr,
                                                                    dlval.ptr,
                                                                    rocsparse_index_base_zero,
                                                                    rocsparse_diag_type_unit,
                                                                    rocsparse_direction_column,
                                                                    3,
                                                                    duptr.ptr,
                                                                    duptr.ptr + 1,
                                                                    duind.ptr,
                                                                    duval.ptr,
                                                                    rocsparse_index_base_zero,
                                                                    ddval.ptr,
                                                                    0,
                                                                    nullptr)),
                  rocsparse_status_success);
    }
}
