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
// Device unit tests of the Jacobi iterative triangular solvers
// rocsparse_Xcsritsv_* and rocsparse_spitsv.
//
// The iteration solves op(A) y = alpha * x for a triangular system. For the
// diagonally dominant (or, for unit diagonals, strictly small off-diagonal)
// matrices used here it converges to the exact solution after at most m
// iterations, so every result is compared with a host substitution.
//
#include "unit_test_device_level2_common.hpp"

#include <set>
#include <sstream>
#include <string>
#include <tuple>

using namespace level2;

namespace
{
    // The legacy solve (no free iterations) or, with `ex`, rocsparse_csritsv_solve_ex.
    template <typename T, typename... Args>
    rocsparse_status itsv_solve(
        bool ex, rocsparse_handle handle, rocsparse_int* nmax, rocsparse_int nfree, Args&&... args)
    {
        return ex ? csritsv_solve_ex<T>(handle, nmax, nfree, std::forward<Args>(args)...)
                  : csritsv_solve<T>(handle, nmax, std::forward<Args>(args)...);
    }

    struct itsv_case
    {
        rocsparse_matrix_type mtype   = rocsparse_matrix_type_general;
        rocsparse_fill_mode   fill    = rocsparse_fill_mode_lower;
        rocsparse_diag_type   diag    = rocsparse_diag_type_non_unit;
        rocsparse_operation   trans   = rocsparse_operation_none;
        int                   base    = 0;
        bool                  device  = false;
        int32_t               nfree   = 0;
        bool                  tol     = true;
        bool                  hist    = true;
        bool                  ex      = true;
        int32_t               m       = 20;
        int32_t               per_row = 3;
        // Analysis calls made before the solve (each one with its own policy).
        std::vector<rocsparse_analysis_policy> policies = {rocsparse_analysis_policy_force};
    };

    // The iteration controls that do not need to be crossed with the matrix
    // description: they are derived from a running counter instead.
    inline void vary_controls(itsv_case& c, int n)
    {
        c.nfree = n % 3;
        c.tol   = (n % 4) != 3; // a tolerance is given in three cases out of four
        c.hist  = (n % 3) != 1;
        c.ex    = c.nfree != 0 || (n % 2) == 0; // free iterations need solve_ex
    }

    // Checks the convergence history and the iteration count written by a solve.
    template <typename R>
    void check_history(const itsv_case&      c,
                       const std::vector<R>& history,
                       rocsparse_int         nmax,
                       rocsparse_int         nmax_in,
                       R                     tol)
    {
        const bool stopped = c.tol && nmax < nmax_in;
        if(!stopped)
        {
            // No tolerance, or never reached: all the iterations ran.
            EXPECT_EQ(nmax, nmax_in);
        }
        if(!c.hist)
            return;
        // The first entry is the norm after the first step: a non zero vector.
        EXPECT_GT(history[0], R(0));
        // Entries 0 .. nmax were recorded (all of them if the loop never stopped).
        const rocsparse_int recorded = stopped ? nmax + 1 : nmax_in;
        for(rocsparse_int i = 0; i < recorded; ++i)
            EXPECT_GE(history[i], R(0)) << "history[" << i << "] was not recorded";
        if(c.tol)
        {
            // The loop stops at the first entry within the tolerance and not before.
            const rocsparse_int above = stopped ? nmax : nmax_in;
            for(rocsparse_int i = 0; i < above; ++i)
                EXPECT_GT(history[i], tol) << "history[" << i << "]";
            if(stopped)
                EXPECT_LE(history[nmax], tol);
        }
    }

    inline std::string describe(const itsv_case& c)
    {
        std::ostringstream s;
        s << "csritsv type=" << c.mtype << " fill=" << c.fill << " diag=" << c.diag
          << " trans=" << c.trans << " base=" << c.base << " device=" << c.device
          << " nfree=" << c.nfree << " tol=" << c.tol << " hist=" << c.hist << " ex=" << c.ex;
        return s.str();
    }

    template <typename T>
    host_csr<T> make_itsv_matrix(const itsv_case& c)
    {
        const bool general  = (c.mtype == rocsparse_matrix_type_general);
        const bool lower    = (c.fill == rocsparse_fill_mode_lower);
        const bool non_unit = (c.diag == rocsparse_diag_type_non_unit);
        // A general matrix stores both triangles (and a diagonal that the unit
        // variant must ignore); a triangular matrix stores only its triangle.
        host_csr<T> A = make_triangular_pattern<T>(c.m,
                                                   c.per_row,
                                                   general ? true : lower,
                                                   general ? true : !lower,
                                                   non_unit || general,
                                                   c.base);
        if(!non_unit)
        {
            // Small off-diagonal entries keep the Jacobi iteration contractive.
            for(auto& v : A.val)
                v = v * rocsparse_ut::scalar<T>(0.125f);
        }
        return A;
    }

    void set_descr(const mat_descr& descr, const itsv_case& c)
    {
        ASSERT_EQ(rocsparse_set_mat_type(descr, c.mtype), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, c.fill), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_diag_type(descr, c.diag), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_index_base(descr, to_base(c.base)), rocsparse_status_success);
    }

    template <typename T>
    void run_csritsv(rocsparse_handle handle, const itsv_case& c)
    {
        using R = typename real_of<T>::type;
        SCOPED_TRACE(describe(c));

        const host_csr<T>    A     = make_itsv_matrix<T>(c);
        const int32_t        m     = A.m;
        const T              alpha = rocsparse_ut::scalar<T>(2.0f);
        const std::vector<T> x     = make_vector<T>(m, 3);
        const std::vector<T> want  = host_triangular_solve(A, c.fill, c.diag, c.trans, alpha, x);

        mat_descr descr;
        mat_info  info;
        ASSERT_TRUE(descr.d && info.d);
        set_descr(descr, c);
        if(::testing::Test::HasFatalFailure())
            return;

        device_csr<T>    dA(A);
        device_vector<T> dx(x), dy(std::vector<T>(m, T{})), dalpha(std::vector<T>(1, alpha));
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr && dalpha.ptr);

        size_t bytes = 0;
        ASSERT_EQ(csritsv_buffer_size<T>(
                      handle, c.trans, m, A.nnz(), descr, dA.val, dA.ptr, dA.ind, info, &bytes),
                  rocsparse_status_success);
        ASSERT_GT(bytes, size_t{0});
        device_vector<char> dbuf(bytes);
        ASSERT_TRUE(dbuf.ptr);

        for(const auto policy : c.policies)
        {
            ASSERT_EQ(csritsv_analysis<T>(handle,
                                          c.trans,
                                          m,
                                          A.nnz(),
                                          descr,
                                          dA.val,
                                          dA.ptr,
                                          dA.ind,
                                          info,
                                          policy,
                                          rocsparse_solve_policy_auto,
                                          dbuf),
                      rocsparse_status_success);
        }

        const rocsparse_int nmax_in = 60;
        rocsparse_int       nmax    = nmax_in;
        const R             tol     = is_single<T>() ? R(1.0e-5) : R(1.0e-13);
        std::vector<R>      history(nmax_in, R(-1));
        {
            scoped_pointer_mode mode(handle, c.device);
            const T*            palpha = c.device ? dalpha.ptr : &alpha;
            ASSERT_EQ(itsv_solve<T>(c.ex,
                                    handle,
                                    &nmax,
                                    c.nfree,
                                    c.tol ? &tol : nullptr,
                                    c.hist ? history.data() : nullptr,
                                    c.trans,
                                    m,
                                    A.nnz(),
                                    palpha,
                                    descr,
                                    dA.val,
                                    dA.ptr,
                                    dA.ind,
                                    info,
                                    dx,
                                    dy,
                                    rocsparse_solve_policy_auto,
                                    dbuf),
                      rocsparse_status_success);
        }
        EXPECT_TRUE(expect_near(want, to_host(dy), "csritsv y"));

        check_history(c, history, nmax, nmax_in, tol);

        rocsparse_int position = -2;
        EXPECT_EQ(rocsparse_csritsv_zero_pivot(handle, descr, info, &position),
                  rocsparse_status_success);
        EXPECT_EQ(position, -1);
        EXPECT_EQ(rocsparse_csritsv_clear(handle, descr, info), rocsparse_status_success);
    }

    class L2DeviceCsritsv : public rocsparse_ut::HandleTest
    {
    protected:
        // Every (matrix type, fill mode, diagonal, base, pointer mode) combination runs
        // for each precision. `all_trans` runs every operation for each of them; otherwise
        // the operation cycles through the three with the combination counter.
        template <typename T>
        void sweep(bool all_trans)
        {
            const rocsparse_operation                     ops[] = {rocsparse_operation_none,
                                               rocsparse_operation_transpose,
                                               rocsparse_operation_conjugate_transpose};
            std::set<std::tuple<int, int, int, int, int>> seen;
            int                                           n = 0;
            for(const auto mtype :
                {rocsparse_matrix_type_general, rocsparse_matrix_type_triangular})
                for(const auto fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
                    for(const auto diag : {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit})
                        for(const int base : {0, 1})
                            for(const bool device : {false, true})
                            {
                                seen.insert(std::make_tuple(static_cast<int>(mtype),
                                                            static_cast<int>(fill),
                                                            static_cast<int>(diag),
                                                            base,
                                                            device ? 1 : 0));
                                for(int o = 0; o < 3; ++o)
                                {
                                    if(!all_trans && o != n % 3)
                                        continue;
                                    itsv_case c;
                                    c.mtype  = mtype;
                                    c.fill   = fill;
                                    c.diag   = diag;
                                    c.trans  = ops[o];
                                    c.base   = base;
                                    c.device = device;
                                    vary_controls(c, n++);
                                    run_csritsv<T>(handle, c);
                                    if(::testing::Test::HasFailure())
                                        return;
                                }
                            }
            EXPECT_EQ(seen.size(), size_t{32});
        }
    };

    TEST_F(L2DeviceCsritsv, solve_complex)
    {
        sweep<rocsparse_float_complex>(false);
        sweep<rocsparse_double_complex>(false);
    }

    TEST_F(L2DeviceCsritsv, solve_real)
    {
        sweep<float>(false);
        sweep<double>(false);
    }

    TEST_F(L2DeviceCsritsv, solve_both_apis_and_reuse_policy)
    {
        // The legacy solve entry (no free iterations) next to solve_ex, each one after
        // repeated analyses: a force analysis followed by reuse analyses (which keep
        // the first one), and reuse analyses on an info object never analysed.
        const std::vector<std::vector<rocsparse_analysis_policy>> sequences
            = {{rocsparse_analysis_policy_force,
                rocsparse_analysis_policy_reuse,
                rocsparse_analysis_policy_reuse,
                rocsparse_analysis_policy_force},
               {rocsparse_analysis_policy_reuse, rocsparse_analysis_policy_reuse}};
        for(const bool ex : {false, true})
            for(const auto& sequence : sequences)
                for(const auto diag : {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit})
                {
                    itsv_case c;
                    c.ex       = ex;
                    c.fill     = rocsparse_fill_mode_upper;
                    c.diag     = diag;
                    c.mtype    = rocsparse_matrix_type_triangular;
                    c.policies = sequence;
                    run_csritsv<double>(handle, c);
                    if(::testing::Test::HasFailure())
                        return;
                }
    }

    // A matrix whose diagonal entry of row `p` is missing (`zero_value` false) or
    // stored as an explicit zero (`zero_value` true).
    template <typename T>
    void run_zero_pivot(rocsparse_handle      handle,
                        rocsparse_matrix_type mtype,
                        rocsparse_fill_mode   fill,
                        bool                  zero_value,
                        int32_t               p,
                        int                   base)
    {
        SCOPED_TRACE(::testing::Message()
                     << "zero pivot type=" << mtype << " fill=" << fill << " value=" << zero_value
                     << " p=" << p << " base=" << base);
        itsv_case c;
        c.mtype         = mtype;
        c.fill          = fill;
        c.base          = base;
        host_csr<T>   A = make_itsv_matrix<T>(c);
        const int32_t m = A.m;

        if(zero_value)
        {
            for(int32_t k = A.ptr[p] - A.base; k < A.ptr[p + 1] - A.base; ++k)
                if(A.ind[k] - A.base == p)
                    A.val[k] = T{};
        }
        else
        {
            host_csr<T> B;
            B.m    = A.m;
            B.n    = A.n;
            B.base = A.base;
            B.ptr.push_back(A.base);
            for(int32_t i = 0; i < A.m; ++i)
            {
                for(int32_t k = A.ptr[i] - A.base; k < A.ptr[i + 1] - A.base; ++k)
                {
                    if(i == p && A.ind[k] - A.base == p)
                        continue;
                    B.ind.push_back(A.ind[k]);
                    B.val.push_back(A.val[k]);
                }
                B.ptr.push_back(static_cast<int32_t>(B.ind.size()) + A.base);
            }
            A = B;
        }

        mat_descr descr;
        mat_info  info;
        ASSERT_TRUE(descr.d && info.d);
        set_descr(descr, c);
        const std::vector<T> x     = make_vector<T>(m, 3);
        const std::vector<T> y0    = make_vector<T>(m, 8);
        const T              alpha = rocsparse_ut::scalar<T>(1.0f);
        device_csr<T>        dA(A);
        device_vector<T>     dx(x), dy(y0);
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr);

        size_t bytes = 0;
        ASSERT_EQ(csritsv_buffer_size<T>(
                      handle, c.trans, m, A.nnz(), descr, dA.val, dA.ptr, dA.ind, info, &bytes),
                  rocsparse_status_success);
        device_vector<char> dbuf(bytes);
        ASSERT_TRUE(dbuf.ptr);
        ASSERT_EQ(csritsv_analysis<T>(handle,
                                      c.trans,
                                      m,
                                      A.nnz(),
                                      descr,
                                      dA.val,
                                      dA.ptr,
                                      dA.ind,
                                      info,
                                      rocsparse_analysis_policy_force,
                                      rocsparse_solve_policy_auto,
                                      dbuf),
                  rocsparse_status_success);

        using R            = typename real_of<T>::type;
        rocsparse_int nmax = 30;
        const R       tol  = R(1.0e-4);
        ASSERT_EQ(itsv_solve<T>(true,
                                handle,
                                &nmax,
                                0,
                                &tol,
                                nullptr,
                                c.trans,
                                m,
                                A.nnz(),
                                &alpha,
                                descr,
                                dA.val,
                                dA.ptr,
                                dA.ind,
                                info,
                                dx,
                                dy,
                                rocsparse_solve_policy_auto,
                                dbuf),
                  rocsparse_status_success);

        // The solve is skipped when a pivot is zero: y is not modified.
        EXPECT_TRUE(expect_near(y0, to_host(dy), "csritsv y after zero pivot"));

        rocsparse_int position = -2;
        EXPECT_EQ(rocsparse_csritsv_zero_pivot(handle, descr, info, &position),
                  rocsparse_status_zero_pivot);
        EXPECT_EQ(position, p + base);
        EXPECT_EQ(rocsparse_csritsv_clear(handle, descr, info), rocsparse_status_success);
    }

    TEST_F(L2DeviceCsritsv, zero_pivot)
    {
        for(const auto mtype : {rocsparse_matrix_type_general, rocsparse_matrix_type_triangular})
            for(const auto fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
                for(const bool zero_value : {false, true})
                {
                    run_zero_pivot<float>(handle, mtype, fill, zero_value, 7, 0);
                    run_zero_pivot<double>(handle, mtype, fill, zero_value, 3, 1);
                    run_zero_pivot<rocsparse_double_complex>(
                        handle, mtype, fill, zero_value, 11, 0);
                    if(::testing::Test::HasFailure())
                        return;
                }
    }

    TEST_F(L2DeviceCsritsv, empty_matrices)
    {
        mat_descr descr;
        mat_info  info;
        ASSERT_TRUE(descr.d && info.d);
        ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general),
                  rocsparse_status_success);

        // m == 0.
        {
            size_t bytes = 99;
            EXPECT_EQ(csritsv_buffer_size<float>(handle,
                                                 rocsparse_operation_none,
                                                 0,
                                                 0,
                                                 descr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 info,
                                                 &bytes),
                      rocsparse_status_success);
            EXPECT_EQ(csritsv_analysis<float>(handle,
                                              rocsparse_operation_none,
                                              0,
                                              0,
                                              descr,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              info,
                                              rocsparse_analysis_policy_force,
                                              rocsparse_solve_policy_auto,
                                              nullptr),
                      rocsparse_status_success);
            rocsparse_int nmax  = 5;
            const float   alpha = 1.0f;
            for(const bool ex : {false, true})
                EXPECT_EQ(itsv_solve<float>(ex,
                                            handle,
                                            &nmax,
                                            0,
                                            nullptr,
                                            nullptr,
                                            rocsparse_operation_none,
                                            0,
                                            0,
                                            &alpha,
                                            descr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            info,
                                            nullptr,
                                            nullptr,
                                            rocsparse_solve_policy_auto,
                                            nullptr),
                          rocsparse_status_success);
            rocsparse_int position = -2;
            EXPECT_EQ(rocsparse_csritsv_zero_pivot(handle, descr, info, &position),
                      rocsparse_status_success);
            EXPECT_EQ(rocsparse_csritsv_clear(handle, descr, info), rocsparse_status_success);
        }

        // nnz == 0 with m > 0: a unit diagonal solves to alpha * x, a non unit
        // diagonal is a (structurally) zero pivot at the first row.
        const int32_t            m     = 9;
        const std::vector<float> x     = make_vector<float>(m, 6);
        const float              alpha = 2.0f;
        host_csr<float>          A;
        A.m = A.n = m;
        A.ptr.assign(m + 1, 0);
        std::vector<float> scaled(x);
        for(auto& v : scaled)
            v *= alpha;

        for(const auto diag : {rocsparse_diag_type_unit, rocsparse_diag_type_non_unit})
            for(const auto mtype :
                {rocsparse_matrix_type_general, rocsparse_matrix_type_triangular})
            {
                SCOPED_TRACE(::testing::Message() << "nnz=0 diag=" << diag << " type=" << mtype);
                mat_descr d2;
                mat_info  i2;
                ASSERT_TRUE(d2.d && i2.d);
                ASSERT_EQ(rocsparse_set_mat_type(d2, mtype), rocsparse_status_success);
                ASSERT_EQ(rocsparse_set_mat_diag_type(d2, diag), rocsparse_status_success);

                device_csr<float>    dA(A);
                device_vector<float> dx(x);
                ASSERT_TRUE(dA.ok() && dx.ptr);
                size_t bytes = 0;
                ASSERT_EQ(csritsv_buffer_size<float>(handle,
                                                     rocsparse_operation_none,
                                                     m,
                                                     0,
                                                     d2,
                                                     dA.val,
                                                     dA.ptr,
                                                     dA.ind,
                                                     i2,
                                                     &bytes),
                          rocsparse_status_success);
                device_vector<char> dbuf(std::max<size_t>(bytes, 16));
                ASSERT_TRUE(dbuf.ptr);
                ASSERT_EQ(csritsv_analysis<float>(handle,
                                                  rocsparse_operation_none,
                                                  m,
                                                  0,
                                                  d2,
                                                  dA.val,
                                                  dA.ptr,
                                                  dA.ind,
                                                  i2,
                                                  rocsparse_analysis_policy_force,
                                                  rocsparse_solve_policy_auto,
                                                  dbuf),
                          rocsparse_status_success);

                const float tol = 1.0e-6f;
                // y starts at zero, so one step is needed.
                {
                    device_vector<float> dy(std::vector<float>(m, 0.0f));
                    rocsparse_int        nmax = 5;
                    ASSERT_EQ(itsv_solve<float>(true,
                                                handle,
                                                &nmax,
                                                0,
                                                &tol,
                                                nullptr,
                                                rocsparse_operation_none,
                                                m,
                                                0,
                                                &alpha,
                                                d2,
                                                dA.val,
                                                dA.ptr,
                                                dA.ind,
                                                i2,
                                                dx,
                                                dy,
                                                rocsparse_solve_policy_auto,
                                                dbuf),
                              rocsparse_status_success);
                    if(diag == rocsparse_diag_type_unit)
                    {
                        EXPECT_EQ(nmax, 1);
                        EXPECT_TRUE(expect_near(scaled, to_host(dy), "csritsv unit nnz=0 y"));
                    }
                    else
                    {
                        EXPECT_TRUE(expect_near(
                            std::vector<float>(m, 0.0f), to_host(dy), "csritsv non unit nnz=0 y"));
                    }
                }
                // y already holds the solution: no step is needed.
                if(diag == rocsparse_diag_type_unit)
                {
                    device_vector<float> dy(scaled);
                    rocsparse_int        nmax = 5;
                    for(const bool history : {false, true})
                    {
                        std::vector<float> hist(5, -1.0f);
                        ASSERT_EQ(itsv_solve<float>(true,
                                                    handle,
                                                    &nmax,
                                                    0,
                                                    &tol,
                                                    history ? hist.data() : nullptr,
                                                    rocsparse_operation_none,
                                                    m,
                                                    0,
                                                    &alpha,
                                                    d2,
                                                    dA.val,
                                                    dA.ptr,
                                                    dA.ind,
                                                    i2,
                                                    dx,
                                                    dy,
                                                    rocsparse_solve_policy_auto,
                                                    dbuf),
                                  rocsparse_status_success);
                        EXPECT_EQ(nmax, 0);
                        EXPECT_TRUE(expect_near(scaled, to_host(dy), "csritsv unit nnz=0 kept y"));
                        if(history)
                            EXPECT_EQ(hist[0], 0.0f);
                        nmax = 5;
                    }
                }

                rocsparse_int          position = -2;
                const rocsparse_status st = rocsparse_csritsv_zero_pivot(handle, d2, i2, &position);
                if(diag == rocsparse_diag_type_unit)
                {
                    EXPECT_EQ(st, rocsparse_status_success);
                }
                else
                {
                    EXPECT_EQ(st, rocsparse_status_zero_pivot);
                    EXPECT_EQ(position, 0);
                }
            }
    }

    TEST_F(L2DeviceCsritsv, argument_checks)
    {
        const int32_t         m = 6;
        itsv_case             c;
        const host_csr<float> A = make_itsv_matrix<float>([&] {
            itsv_case k;
            k.m = m;
            return k;
        }());
        mat_descr             descr;
        mat_info              info;
        ASSERT_TRUE(descr.d && info.d);
        device_csr<float>    dA(A);
        device_vector<float> dx(make_vector<float>(m, 1)), dy(make_vector<float>(m, 2));
        size_t               bytes = 0;
        ASSERT_EQ(csritsv_buffer_size<float>(
                      handle, c.trans, m, A.nnz(), descr, dA.val, dA.ptr, dA.ind, info, &bytes),
                  rocsparse_status_success);
        device_vector<char> dbuf(bytes);
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr && dbuf.ptr);

        mat_descr sym, unsorted;
        ASSERT_EQ(rocsparse_set_mat_type(sym, rocsparse_matrix_type_symmetric),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_storage_mode(unsorted, rocsparse_storage_mode_unsorted),
                  rocsparse_status_success);
        const auto invalid_pointer = rocsparse_status_invalid_pointer;

        {
            enum
            {
                TRANS,
                M,
                NNZ,
                DESCR,
                VAL,
                PTR,
                IND,
                INFO,
                BYTES
            };
            const arg_checker bad(
                [&](auto... a) { return csritsv_buffer_size<float>(handle, a...); },
                rocsparse_operation_none,
                m,
                A.nnz(),
                descr.d,
                dA.val.ptr,
                dA.ptr.ptr,
                dA.ind.ptr,
                info.d,
                &bytes);
            bad.check_invalid<M>(-1);
            bad.check<BYTES>(nullptr, invalid_pointer);
            bad.check_invalid<TRANS>(static_cast<rocsparse_operation>(77));
            bad.check<PTR>(nullptr, invalid_pointer);
            bad.check<DESCR>(sym.d, rocsparse_status_not_implemented);
            bad.check_fails<DESCR>(unsorted.d);
        }
        {
            enum
            {
                M,
                NNZ,
                DESCR,
                VAL,
                PTR,
                IND,
                INFO,
                ANALYSIS,
                SOLVE,
                BUFFER
            };
            const arg_checker bad(
                [&](auto... a) {
                    return csritsv_analysis<float>(handle, rocsparse_operation_none, a...);
                },
                m,
                A.nnz(),
                descr.d,
                dA.val.ptr,
                dA.ptr.ptr,
                dA.ind.ptr,
                info.d,
                rocsparse_analysis_policy_force,
                rocsparse_solve_policy_auto,
                dbuf.ptr);
            bad.check<INFO>(nullptr, invalid_pointer);
            bad.check_invalid<ANALYSIS>(static_cast<rocsparse_analysis_policy>(55));
            bad.check_invalid<SOLVE>(static_cast<rocsparse_solve_policy>(55));
            bad.check<BUFFER>(nullptr, invalid_pointer);
        }

        // ---- solve without analysis ----
        rocsparse_int nmax  = 3;
        const float   alpha = 1.0f;
        for(const bool ex : {false, true})
        {
            mat_info fresh;
            ASSERT_TRUE(fresh.d);
            EXPECT_EQ(itsv_solve<float>(ex,
                                        handle,
                                        &nmax,
                                        0,
                                        nullptr,
                                        nullptr,
                                        rocsparse_operation_none,
                                        m,
                                        A.nnz(),
                                        &alpha,
                                        descr,
                                        dA.val,
                                        dA.ptr,
                                        dA.ind,
                                        fresh,
                                        dx,
                                        dy,
                                        rocsparse_solve_policy_auto,
                                        dbuf),
                      invalid_pointer);
        }

        ASSERT_EQ(csritsv_analysis<float>(handle,
                                          rocsparse_operation_none,
                                          m,
                                          A.nnz(),
                                          descr,
                                          dA.val,
                                          dA.ptr,
                                          dA.ind,
                                          info,
                                          rocsparse_analysis_policy_force,
                                          rocsparse_solve_policy_auto,
                                          dbuf),
                  rocsparse_status_success);

        // ---- solve and solve_ex ----
        enum
        {
            NMAX,
            NFREE,
            TOL,
            HIST,
            TRANS,
            M,
            NNZ,
            ALPHA,
            DESCR,
            VAL,
            PTR,
            IND,
            INFO,
            X,
            Y,
            POLICY,
            BUFFER
        };
        const float* no_tol  = nullptr;
        float*       no_hist = nullptr;
        for(const bool ex : {false, true})
        {
            const arg_checker bad([&](auto... a) { return itsv_solve<float>(ex, handle, a...); },
                                  &nmax,
                                  0,
                                  no_tol,
                                  no_hist,
                                  rocsparse_operation_none,
                                  m,
                                  A.nnz(),
                                  &alpha,
                                  descr.d,
                                  dA.val.ptr,
                                  dA.ptr.ptr,
                                  dA.ind.ptr,
                                  info.d,
                                  dx.ptr,
                                  dy.ptr,
                                  rocsparse_solve_policy_auto,
                                  dbuf.ptr);
            bad.check<NFREE>(0, rocsparse_status_success);
            bad.check_invalid<TRANS>(static_cast<rocsparse_operation>(77));
            bad.check_invalid<POLICY>(static_cast<rocsparse_solve_policy>(55));
            bad.check_invalid<M>(-1);
            bad.check<PTR>(nullptr, invalid_pointer);
            bad.check<BUFFER>(nullptr, invalid_pointer);
            bad.check<NMAX>(nullptr, invalid_pointer);
            bad.check<ALPHA>(nullptr, invalid_pointer);
            bad.check<X>(nullptr, invalid_pointer);
            bad.check<Y>(nullptr, invalid_pointer);
            bad.check<DESCR>(nullptr, invalid_pointer);
            bad.check<DESCR>(sym.d, rocsparse_status_not_implemented);
            bad.check_fails<DESCR>(unsorted.d);
            if(ex)
                bad.check_invalid<NFREE>(-1);
        }
    }

    // ----- generic API --------------------------------------------------------------
    inline const char* index_name(rocsparse_indextype t)
    {
        return t == rocsparse_indextype_i32 ? "i32" : "i64";
    }

    template <typename T, typename I, typename J>
    void run_spitsv(rocsparse_handle handle, const itsv_case& c)
    {
        using R = typename real_of<T>::type;
        SCOPED_TRACE(describe(c) + " spitsv " + index_name(rocsparse_ut::it_of<I>()) + "/"
                     + index_name(rocsparse_ut::it_of<J>()));

        const host_csr<T>    A     = make_itsv_matrix<T>(c);
        const int32_t        m     = A.m;
        const T              alpha = rocsparse_ut::scalar<T>(2.0f);
        const std::vector<T> x     = make_vector<T>(m, 3);
        const std::vector<T> want  = host_triangular_solve(A, c.fill, c.diag, c.trans, alpha, x);

        device_csr<T, I, J> dA(A);
        device_vector<T>    dx(x), dy(std::vector<T>(m, T{})), dalpha(std::vector<T>(1, alpha));
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr && dalpha.ptr);

        spmat_guard mat;
        ASSERT_EQ(rocsparse_create_csr_descr(&mat.d,
                                             m,
                                             m,
                                             A.nnz(),
                                             dA.ptr,
                                             dA.ind,
                                             dA.val,
                                             rocsparse_ut::it_of<I>(),
                                             rocsparse_ut::it_of<J>(),
                                             to_base(c.base),
                                             rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);
        dnvec_guard vx, vy;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, m, dx, rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, m, dy, rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spmat_set_attribute(
                      mat.d, rocsparse_spmat_fill_mode, &c.fill, sizeof(c.fill)),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spmat_set_attribute(
                      mat.d, rocsparse_spmat_diag_type, &c.diag, sizeof(c.diag)),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spmat_set_attribute(
                      mat.d, rocsparse_spmat_matrix_type, &c.mtype, sizeof(c.mtype)),
                  rocsparse_status_success);

        const rocsparse_int nmax_in = 60;
        rocsparse_int       nmax    = nmax_in;
        const R             tol     = is_single<T>() ? R(1.0e-5) : R(1.0e-13);
        std::vector<R>      history(nmax_in, R(-1));

        scoped_pointer_mode mode(handle, c.device);
        const void*         palpha
            = c.device ? static_cast<const void*>(dalpha.ptr) : static_cast<const void*>(&alpha);

        auto call = [&](rocsparse_spitsv_stage stage, size_t* bytes, void* buffer) {
            return rocsparse_spitsv(handle,
                                    &nmax,
                                    c.tol ? &tol : nullptr,
                                    c.hist ? history.data() : nullptr,
                                    c.trans,
                                    palpha,
                                    mat.d,
                                    vx.d,
                                    vy.d,
                                    rocsparse_ut::dt_of<T>(),
                                    rocsparse_spitsv_alg_default,
                                    stage,
                                    bytes,
                                    buffer);
        };

        size_t bytes = 0;
        ASSERT_EQ(call(rocsparse_spitsv_stage_buffer_size, &bytes, nullptr),
                  rocsparse_status_success);
        ASSERT_GT(bytes, size_t{0});
        device_vector<char> dbuf(bytes);
        ASSERT_TRUE(dbuf.ptr);
        ASSERT_EQ(call(rocsparse_spitsv_stage_preprocess, nullptr, dbuf), rocsparse_status_success);
        ASSERT_EQ(call(rocsparse_spitsv_stage_compute, nullptr, dbuf), rocsparse_status_success);
        EXPECT_TRUE(expect_near(want, to_host(dy), "spitsv y"));
        check_history(c, history, nmax, nmax_in, tol);
    }

    class L2DeviceSpitsv : public rocsparse_ut::HandleTest
    {
    };

    TEST_F(L2DeviceSpitsv, types_and_modes)
    {
        // Every (matrix type, fill mode, diagonal, base, pointer mode) combination for
        // each of four type / index type combinations; the operation alternates.
        std::set<std::tuple<int, int, int, int, int>> seen;
        int                                           n = 0;
        for(const auto mtype : {rocsparse_matrix_type_general, rocsparse_matrix_type_triangular})
            for(const auto fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
                for(const auto diag : {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit})
                    for(const int base : {0, 1})
                        for(const bool device : {false, true})
                        {
                            seen.insert(std::make_tuple(static_cast<int>(mtype),
                                                        static_cast<int>(fill),
                                                        static_cast<int>(diag),
                                                        base,
                                                        device ? 1 : 0));
                            itsv_case c;
                            c.mtype  = mtype;
                            c.fill   = fill;
                            c.diag   = diag;
                            c.trans  = (n % 2 == 0) ? rocsparse_operation_none
                                                    : rocsparse_operation_transpose;
                            c.base   = base;
                            c.device = device;
                            c.tol    = (n % 4) != 3;
                            c.hist   = (n % 3) != 1;
                            ++n;
                            run_spitsv<float, int32_t, int32_t>(handle, c);
                            run_spitsv<double, int64_t, int32_t>(handle, c);
                            run_spitsv<rocsparse_float_complex, int64_t, int64_t>(handle, c);
                            run_spitsv<rocsparse_double_complex, int32_t, int32_t>(handle, c);
                            if(::testing::Test::HasFailure())
                                return;
                        }
        EXPECT_EQ(seen.size(), size_t{32});
    }

    TEST_F(L2DeviceSpitsv, argument_checks)
    {
        const int32_t m = 6;
        itsv_case     c;
        c.m                                   = m;
        const host_csr<float>               A = make_itsv_matrix<float>(c);
        device_csr<float, int32_t, int32_t> dA(A);
        device_vector<float> dx(make_vector<float>(m, 1)), dy(make_vector<float>(m, 2));
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr);

        spmat_guard mat;
        ASSERT_EQ(rocsparse_create_csr_descr(&mat.d,
                                             m,
                                             m,
                                             A.nnz(),
                                             dA.ptr,
                                             dA.ind,
                                             dA.val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        dnvec_guard vx, vy;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, m, dx, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, m, dy, rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        rocsparse_int nmax  = 4;
        const float   alpha = 1.0f;
        size_t        bytes = 0;
        auto          call  = [&](rocsparse_handle       h,
                        rocsparse_int*         p_nmax,
                        rocsparse_operation    trans,
                        const void*            p_alpha,
                        rocsparse_spmat_descr  a,
                        rocsparse_dnvec_descr  px,
                        rocsparse_dnvec_descr  py,
                        rocsparse_datatype     compute,
                        rocsparse_spitsv_alg   alg,
                        rocsparse_spitsv_stage stage,
                        size_t*                p_bytes) {
            return rocsparse_spitsv(h,
                                    p_nmax,
                                    nullptr,
                                    nullptr,
                                    trans,
                                    p_alpha,
                                    a,
                                    px,
                                    py,
                                    compute,
                                    alg,
                                    stage,
                                    p_bytes,
                                    nullptr);
        };
        const auto f32 = rocsparse_datatype_f32_r;
        const auto sz  = rocsparse_spitsv_stage_buffer_size;
        const auto alg = rocsparse_spitsv_alg_default;

        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       vx.d,
                       vy.d,
                       f32,
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_success);
        EXPECT_EQ(call(nullptr,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       vx.d,
                       vy.d,
                       f32,
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       nullptr,
                       vx.d,
                       vy.d,
                       f32,
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       nullptr,
                       vy.d,
                       f32,
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       vx.d,
                       nullptr,
                       f32,
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       nullptr,
                       mat.d,
                       vx.d,
                       vy.d,
                       f32,
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(call(handle,
                       nullptr,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       vx.d,
                       vy.d,
                       f32,
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       vx.d,
                       vy.d,
                       f32,
                       alg,
                       sz,
                       nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(call(handle,
                       &nmax,
                       static_cast<rocsparse_operation>(77),
                       &alpha,
                       mat.d,
                       vx.d,
                       vy.d,
                       f32,
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       vx.d,
                       vy.d,
                       static_cast<rocsparse_datatype>(77),
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       vx.d,
                       vy.d,
                       f32,
                       static_cast<rocsparse_spitsv_alg>(77),
                       sz,
                       &bytes),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       vx.d,
                       vy.d,
                       f32,
                       alg,
                       static_cast<rocsparse_spitsv_stage>(77),
                       &bytes),
                  rocsparse_status_invalid_value);
        // Mixed precision is not supported.
        EXPECT_EQ(call(handle,
                       &nmax,
                       rocsparse_operation_none,
                       &alpha,
                       mat.d,
                       vx.d,
                       vy.d,
                       rocsparse_datatype_f64_r,
                       alg,
                       sz,
                       &bytes),
                  rocsparse_status_not_implemented);

        // Index type combinations that the solver does not provide.
        {
            device_csr<float, int32_t, int64_t> dB(A);
            ASSERT_TRUE(dB.ok());
            spmat_guard mixed;
            ASSERT_EQ(rocsparse_create_csr_descr(&mixed.d,
                                                 m,
                                                 m,
                                                 A.nnz(),
                                                 dB.ptr,
                                                 dB.ind,
                                                 dB.val,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_indextype_i64,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            EXPECT_EQ(call(handle,
                           &nmax,
                           rocsparse_operation_none,
                           &alpha,
                           mixed.d,
                           vx.d,
                           vy.d,
                           f32,
                           alg,
                           sz,
                           &bytes),
                      rocsparse_status_not_implemented);
        }
        // Types other than the four floating point precisions.
        {
            spmat_guard                         half;
            device_csr<float, int32_t, int32_t> dC(A);
            ASSERT_TRUE(dC.ok());
            ASSERT_EQ(rocsparse_create_csr_descr(&half.d,
                                                 m,
                                                 m,
                                                 A.nnz(),
                                                 dC.ptr,
                                                 dC.ind,
                                                 dC.val,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            dnvec_guard hx, hy;
            ASSERT_EQ(rocsparse_create_dnvec_descr(&hx.d, m, dx, rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_create_dnvec_descr(&hy.d, m, dy, rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            // compute type i32 differs from the matrix type: also not implemented.
            EXPECT_EQ(call(handle,
                           &nmax,
                           rocsparse_operation_none,
                           &alpha,
                           half.d,
                           hx.d,
                           hy.d,
                           rocsparse_datatype_i32_r,
                           alg,
                           sz,
                           &bytes),
                      rocsparse_status_not_implemented);
        }
    }
}
