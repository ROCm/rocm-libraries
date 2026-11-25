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

// Scenarios of the triangular analysis data that csrilu0, csric0, csrsv and csrsm share
// on one mat_info: the ILU(0) and IC(0) workflows with either triangle analysed first
// and both analysis policies, each routine reusing the analysis of another, forced
// analyses of shared data, calls without an analysis, new values or a new pattern on an
// analysed mat_info, clear and copy of a mat_info whose data is shared, and the zero
// pivots reported along the way. Every byte of the buffer is set to 0xff before every
// call. Only the public API is used, so the tests keep holding when the way the analysis
// data is stored or shared changes. The value type is double throughout: the sharing
// does not depend on it.
//
// The expectations pin what the library returns today, including values the
// documentation leaves open (such as the zero pivot right after an analysis). Where the
// library is wrong and its page is right the expectation is a known_defect: it holds
// both values and fails with its own message once the documented value is returned, so
// the fix updates the expectation on purpose. Where the library does something on
// purpose that its page omits, contradicts or states inconsistently, the page is what
// needs fixing: the expectation is today's value, with a comment naming the page.
// Sequences that crash or use freed memory are DISABLED_ tests asserting the documented
// behaviour or, where the page is silent, that of the sibling routines; run them with
// --gtest_also_run_disabled_tests, those using freed memory under AddressSanitizer
// (without it the freed memory is usually handed out again and they may pass). A kernel
// dereferencing a pivot that was never allocated faults on the device, after which the
// process can hang instead of aborting: run them one at a time under a timeout.

#include "unit_test_device_precond_triangular_common.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

using namespace triangular_scenarios;

namespace
{
    using triangular_scenarios::name;

    // The matrix without its entry (row, col).
    csr_matrix<double> without_entry(csr_matrix<double> a, int32_t row, int32_t col)
    {
        for(int32_t p = a.ptr[row]; p < a.ptr[row + 1]; ++p)
        {
            if(a.ind[p] == col)
            {
                a.ind.erase(a.ind.begin() + p);
                a.val.erase(a.val.begin() + p);
                for(int32_t r = row + 1; r <= a.n; ++r)
                    --a.ptr[r];
                break;
            }
        }
        return a;
    }

    // The nonzero entries of the rows of `dense`, so a zero leaves the entry out.
    csr_matrix<double> from_dense(const std::vector<std::vector<double>>& dense)
    {
        csr_matrix<double> a;
        a.n = static_cast<int32_t>(dense.size());
        a.ptr.push_back(0);
        for(const auto& row : dense)
        {
            for(int32_t j = 0; j < a.n; ++j)
            {
                if(row[j] != 0.0)
                {
                    a.ind.push_back(j);
                    a.val.push_back(row[j]);
                }
            }
            a.ptr.push_back(static_cast<int32_t>(a.ind.size()));
        }
        return a;
    }

    // The same pattern with every diagonal entry multiplied by `factor`.
    csr_matrix<double> with_scaled_diagonal(csr_matrix<double> a, double factor)
    {
        for(int32_t i = 0; i < a.n; ++i)
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                if(a.ind[p] == i)
                    a.val[p] *= factor;
        return a;
    }

    // A matrix on the device (indices in `base`) with the descriptors of the ILU(0) and
    // IC(0) workflows on one mat_info: A (general, lower, non unit) for csrilu0, csric0
    // and the non unit lower solves of csrsv and csrsm (with A^T for the second solve of
    // IC(0)), L (lower, unit) and U (upper, non unit) for csrsv. The zero pivots are
    // queried in device memory when `device_query` is set. csrsv keeps one zero pivot
    // per mat_info for L and U: rocsparse_csrsv_zero_pivot does not use its descriptor.
    struct triangular_problem
    {
        rocsparse_handle             handle;
        csr_matrix<double>           a;
        rocsparse_index_base         base;
        bool                         device_query;
        device_vector<rocsparse_int> row, col;
        device_vector<double>        val;
        device_vector<char>          buffer;
        rocsparse_mat_descr          A    = nullptr;
        rocsparse_mat_descr          L    = nullptr;
        rocsparse_mat_descr          U    = nullptr;
        rocsparse_mat_info           info = nullptr;
        // The original mat_info once `info` has been replaced by a copy of it.
        rocsparse_mat_info original = nullptr;

        triangular_problem(rocsparse_handle          h,
                           const csr_matrix<double>& m,
                           rocsparse_index_base      b,
                           bool                      device = false)
            : handle(h)
            , a(m)
            , base(b)
            , device_query(device)
        {
        }

        triangular_problem(const triangular_problem&) = delete;
        triangular_problem& operator=(const triangular_problem&) = delete;

        ~triangular_problem()
        {
            for(auto i : {info, original})
                if(i)
                    (void)rocsparse_destroy_mat_info(i);
            for(auto descr : {A, L, U})
                if(descr)
                    (void)rocsparse_destroy_mat_descr(descr);
        }

        void init()
        {
            A = make_descr(base, rocsparse_fill_mode_lower, rocsparse_diag_type_non_unit);
            L = make_descr(base, rocsparse_fill_mode_lower, rocsparse_diag_type_unit);
            U = make_descr(base, rocsparse_fill_mode_upper, rocsparse_diag_type_non_unit);
            ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);
            ASSERT_NO_FATAL_FAILURE(upload_matrix());
        }

        // Replaces the matrix, pattern included, keeping the descriptors and the
        // mat_info with whatever it holds.
        void replace_matrix(const csr_matrix<double>& m)
        {
            a = m;
            ASSERT_NO_FATAL_FAILURE(upload_matrix());
        }

        void upload_matrix()
        {
            const rocsparse_int        shift = (base == rocsparse_index_base_one) ? 1 : 0;
            std::vector<rocsparse_int> hptr  = as_rocsparse_int(a.ptr);
            std::vector<rocsparse_int> hind  = as_rocsparse_int(a.ind);
            for(auto& v : hptr)
                v += shift;
            for(auto& v : hind)
                v += shift;
            upload(row, hptr);
            upload(col, hind);
            upload(val, a.val);
            ASSERT_TRUE(row.ptr && col.ptr && val.ptr);

            size_t size = 0;
            ASSERT_EQ(
                rocsparse_dcsrilu0_buffer_size(handle, a.n, a.nnz(), A, val, row, col, info, &size),
                rocsparse_status_success);
            ASSERT_NO_FATAL_FAILURE(reserve(size));
            ASSERT_EQ(
                rocsparse_dcsric0_buffer_size(handle, a.n, a.nnz(), A, val, row, col, info, &size),
                rocsparse_status_success);
            ASSERT_NO_FATAL_FAILURE(reserve(size));
            for(auto descr : {A, L, U})
            {
                for(auto op : {rocsparse_operation_none, rocsparse_operation_transpose})
                {
                    ASSERT_EQ(rocsparse_dcsrsv_buffer_size(
                                  handle, op, a.n, a.nnz(), descr, val, row, col, info, &size),
                              rocsparse_status_success);
                    ASSERT_NO_FATAL_FAILURE(reserve(size));
                }
            }
        }

        void reserve(size_t size)
        {
            if(size > buffer.n)
                upload(buffer, std::vector<char>(size, static_cast<char>(0xff)));
            ASSERT_NE(buffer.ptr, nullptr);
        }

        void poison()
        {
            triangular_scenarios::poison(handle, buffer);
        }

        // Replaces the values of the matrix on the device (csrilu0 factors them in
        // place); `values` defaults to those of `a`.
        void set_values(const std::vector<double>* values = nullptr)
        {
            const std::vector<double>& v = values ? *values : a.val;
            ASSERT_EQ(v.size(), a.val.size());
            ASSERT_EQ(
                hipMemcpy(val.ptr, v.data(), sizeof(double) * v.size(), hipMemcpyHostToDevice),
                hipSuccess);
        }

        std::vector<double> values() const
        {
            return to_host(val);
        }

        rocsparse_status ilu0_analysis(rocsparse_analysis_policy policy)
        {
            poison();
            return rocsparse_dcsrilu0_analysis(handle,
                                               a.n,
                                               a.nnz(),
                                               A,
                                               val,
                                               row,
                                               col,
                                               info,
                                               policy,
                                               rocsparse_solve_policy_auto,
                                               buffer);
        }

        rocsparse_status ilu0()
        {
            poison();
            return rocsparse_dcsrilu0(
                handle, a.n, a.nnz(), A, val, row, col, info, rocsparse_solve_policy_auto, buffer);
        }

        rocsparse_status ic0_analysis(rocsparse_analysis_policy policy)
        {
            poison();
            return rocsparse_dcsric0_analysis(handle,
                                              a.n,
                                              a.nnz(),
                                              A,
                                              val,
                                              row,
                                              col,
                                              info,
                                              policy,
                                              rocsparse_solve_policy_auto,
                                              buffer);
        }

        rocsparse_status ic0()
        {
            poison();
            return rocsparse_dcsric0(
                handle, a.n, a.nnz(), A, val, row, col, info, rocsparse_solve_policy_auto, buffer);
        }

        rocsparse_status sv_analysis(rocsparse_mat_descr       descr,
                                     rocsparse_analysis_policy policy,
                                     rocsparse_operation       op = rocsparse_operation_none)
        {
            poison();
            return rocsparse_dcsrsv_analysis(handle,
                                             op,
                                             a.n,
                                             a.nnz(),
                                             descr,
                                             val,
                                             row,
                                             col,
                                             info,
                                             policy,
                                             rocsparse_solve_policy_auto,
                                             buffer);
        }

        rocsparse_status sv_solve(rocsparse_mat_descr descr,
                                  const double*       x,
                                  double*             y,
                                  rocsparse_operation op = rocsparse_operation_none)
        {
            poison();
            const double one = 1.0;
            return rocsparse_dcsrsv_solve(handle,
                                          op,
                                          a.n,
                                          a.nnz(),
                                          &one,
                                          descr,
                                          val,
                                          row,
                                          col,
                                          info,
                                          x,
                                          y,
                                          rocsparse_solve_policy_auto,
                                          buffer);
        }

        // csrilu0_analysis, the analyses of L and U in the order `o`, then csrilu0.
        void ilu0_workflow(order o, rocsparse_analysis_policy policy)
        {
            ASSERT_EQ(ilu0_analysis(policy), rocsparse_status_success);
            const auto first  = (o == order::lower_first) ? L : U;
            const auto second = (o == order::lower_first) ? U : L;
            ASSERT_EQ(sv_analysis(first, policy), rocsparse_status_success);
            ASSERT_EQ(sv_analysis(second, policy), rocsparse_status_success);
            ASSERT_EQ(ilu0(), rocsparse_status_success);
        }

        // Solves with L, then with U, and returns the result.
        std::vector<double> lu_solve(const std::vector<double>& b)
        {
            device_vector<double> x(b), y(sentinel(b.size())), z(sentinel(b.size()));
            EXPECT_EQ(sv_solve(L, x, y), rocsparse_status_success);
            EXPECT_EQ(sv_solve(U, y, z), rocsparse_status_success);
            return to_host(z);
        }

        // csric0_analysis, the analyses of A and A^T in the order `o` (lower_first
        // analyses A first), then csric0.
        void ic0_workflow(order o, rocsparse_analysis_policy policy)
        {
            ASSERT_EQ(ic0_analysis(policy), rocsparse_status_success);
            const auto first  = (o == order::lower_first) ? rocsparse_operation_none
                                                          : rocsparse_operation_transpose;
            const auto second = (o == order::lower_first) ? rocsparse_operation_transpose
                                                          : rocsparse_operation_none;
            ASSERT_EQ(sv_analysis(A, policy, first), rocsparse_status_success);
            ASSERT_EQ(sv_analysis(A, policy, second), rocsparse_status_success);
            ASSERT_EQ(ic0(), rocsparse_status_success);
        }

        // Solves with the lower triangle of the values, then with its transpose.
        std::vector<double> llt_solve(const std::vector<double>& b)
        {
            device_vector<double> x(b), y(sentinel(b.size())), z(sentinel(b.size()));
            EXPECT_EQ(sv_solve(A, x, y), rocsparse_status_success);
            EXPECT_EQ(sv_solve(A, y, z, rocsparse_operation_transpose), rocsparse_status_success);
            return to_host(z);
        }

        // Replaces the mat_info by a copy of it. The original is destroyed or, when
        // `keep_original` is set, kept in `original`.
        rocsparse_status replace_info_by_copy(bool keep_original)
        {
            rocsparse_mat_info copy   = nullptr;
            rocsparse_status   status = rocsparse_create_mat_info(&copy);
            if(status != rocsparse_status_success)
                return status;
            status = rocsparse_copy_mat_info(copy, info);
            if(status != rocsparse_status_success)
            {
                (void)rocsparse_destroy_mat_info(copy);
                return status;
            }
            if(keep_original)
                original = info;
            else
                status = rocsparse_destroy_mat_info(info);
            info = copy;
            return status;
        }

        int32_t csrsv_pivot(rocsparse_mat_descr descr)
        {
            return zero_pivot([&](rocsparse_int* p) {
                return rocsparse_csrsv_zero_pivot(handle, descr, info, p);
            });
        }

        int32_t csrilu0_pivot()
        {
            return zero_pivot(
                [&](rocsparse_int* p) { return rocsparse_csrilu0_zero_pivot(handle, info, p); });
        }

        int32_t csric0_pivot()
        {
            return zero_pivot(
                [&](rocsparse_int* p) { return rocsparse_csric0_zero_pivot(handle, info, p); });
        }

        int32_t csrsm_pivot()
        {
            return zero_pivot(
                [&](rocsparse_int* p) { return rocsparse_csrsm_zero_pivot(handle, info, p); });
        }

        // The singular pivot is reported with rocsparse_status_success either way.
        int32_t csrilu0_singular_pivot()
        {
            return singular_pivot([&](rocsparse_int* p) {
                return rocsparse_csrilu0_singular_pivot(handle, info, p);
            });
        }

        int32_t csric0_singular_pivot()
        {
            return singular_pivot(
                [&](rocsparse_int* p) { return rocsparse_csric0_singular_pivot(handle, info, p); });
        }

        template <typename F>
        int32_t singular_pivot(F&& call)
        {
            rocsparse_status status   = rocsparse_status_internal_error;
            const auto       position = query_position(handle, device_query, [&](rocsparse_int* p) {
                status = call(p);
                return status;
            });
            EXPECT_EQ(status, rocsparse_status_success) << describe(status);
            return row_of(position, base, a.n);
        }

        template <typename F>
        int32_t zero_pivot(F&& call)
        {
            return zero_pivot_row(handle, base, a.n, device_query, call);
        }
    };

    struct csrsm_result
    {
        rocsparse_status    analysis = rocsparse_status_internal_error;
        rocsparse_status    solve    = rocsparse_status_internal_error;
        std::vector<double> x;
    };

    // csrsm with alpha 1 on the lower triangle of A, for `nrhs` right-hand sides in B.
    rocsparse_status csrsm_buffer_size(triangular_problem& p,
                                       rocsparse_operation trans_A,
                                       rocsparse_operation trans_B,
                                       int32_t             nrhs,
                                       const double*       B,
                                       int32_t             ldb,
                                       size_t*             size)
    {
        const double alpha = 1.0;
        return rocsparse_dcsrsm_buffer_size(p.handle,
                                            trans_A,
                                            trans_B,
                                            p.a.n,
                                            nrhs,
                                            p.a.nnz(),
                                            &alpha,
                                            p.A,
                                            p.val,
                                            p.row,
                                            p.col,
                                            B,
                                            ldb,
                                            p.info,
                                            rocsparse_solve_policy_auto,
                                            size);
    }

    rocsparse_status csrsm_analysis(triangular_problem&       p,
                                    rocsparse_operation       trans_A,
                                    rocsparse_operation       trans_B,
                                    int32_t                   nrhs,
                                    const double*             B,
                                    int32_t                   ldb,
                                    rocsparse_analysis_policy policy)
    {
        p.poison();
        const double alpha = 1.0;
        return rocsparse_dcsrsm_analysis(p.handle,
                                         trans_A,
                                         trans_B,
                                         p.a.n,
                                         nrhs,
                                         p.a.nnz(),
                                         &alpha,
                                         p.A,
                                         p.val,
                                         p.row,
                                         p.col,
                                         B,
                                         ldb,
                                         p.info,
                                         policy,
                                         rocsparse_solve_policy_auto,
                                         p.buffer);
    }

    rocsparse_status csrsm_solve(triangular_problem& p,
                                 rocsparse_operation trans_A,
                                 rocsparse_operation trans_B,
                                 int32_t             nrhs,
                                 double*             B,
                                 int32_t             ldb)
    {
        p.poison();
        const double alpha = 1.0;
        return rocsparse_dcsrsm_solve(p.handle,
                                      trans_A,
                                      trans_B,
                                      p.a.n,
                                      nrhs,
                                      p.a.nnz(),
                                      &alpha,
                                      p.A,
                                      p.val,
                                      p.row,
                                      p.col,
                                      B,
                                      ldb,
                                      p.info,
                                      rocsparse_solve_policy_auto,
                                      p.buffer);
    }

    // The buffer of csrsm with `nrhs` right-hand sides on the lower triangle of A.
    void reserve_csrsm(triangular_problem& p, int32_t nrhs, const double* B)
    {
        const auto none = rocsparse_operation_none;
        size_t     size = 0;
        ASSERT_EQ(csrsm_buffer_size(p, none, none, nrhs, B, p.a.n, &size),
                  rocsparse_status_success);
        ASSERT_NO_FATAL_FAILURE(p.reserve(size));
    }

    rocsparse_status
        csrsm_analysis(triangular_problem& p, int32_t nrhs, rocsparse_analysis_policy policy)
    {
        const auto            none = rocsparse_operation_none;
        device_vector<double> dB(right_hand_side(p.a.n, nrhs));
        reserve_csrsm(p, nrhs, dB);
        if(::testing::Test::HasFatalFailure())
            return rocsparse_status_internal_error;
        return csrsm_analysis(p, none, none, nrhs, dB, p.a.n, policy);
    }

    // Solves in place with the lower triangle of A for the `nrhs` columns of B.
    rocsparse_status csrsm_solve(triangular_problem& p, int32_t nrhs, double* B)
    {
        const auto none = rocsparse_operation_none;
        return csrsm_solve(p, none, none, nrhs, B, p.a.n);
    }

    // csrsm with `nrhs` right-hand sides on the lower triangle of A (non unit), alpha 1.
    void run_csrsm(triangular_problem&       p,
                   int32_t                   nrhs,
                   rocsparse_analysis_policy policy,
                   csrsm_result&             out)
    {
        out.analysis = csrsm_analysis(p, nrhs, policy);
        if(out.analysis != rocsparse_status_success)
            return;
        device_vector<double> dB(right_hand_side(p.a.n, nrhs));
        out.solve = csrsm_solve(p, nrhs, dB);
        out.x     = to_host(dB);
    }

    // The routines that analyse the lower triangle of A on the mat_info; csrsm with two
    // right-hand sides (with one it runs on csrsv).
    enum class routine
    {
        csrsv,
        csrsm,
        csrilu0,
        csric0
    };

    const char* name(routine r)
    {
        switch(r)
        {
        case routine::csrsv:
            return "csrsv";
        case routine::csrsm:
            return "csrsm";
        case routine::csrilu0:
            return "csrilu0";
        case routine::csric0:
            return "csric0";
        }
        return "?";
    }

    rocsparse_status analyse(triangular_problem& p, routine r, rocsparse_analysis_policy policy)
    {
        switch(r)
        {
        case routine::csrsv:
            return p.sv_analysis(p.A, policy);
        case routine::csrsm:
            return csrsm_analysis(p, 2, policy);
        case routine::csrilu0:
            return p.ilu0_analysis(policy);
        case routine::csric0:
            return p.ic0_analysis(policy);
        }
        return rocsparse_status_internal_error;
    }

    // Runs `r` after its analysis (the factorization, or the solve with the lower
    // triangle of A) and compares the result and its zero pivot with the host.
    void check_compute(triangular_problem& p, routine r)
    {
        const auto tol = tolerance<double>(max_row_length(p.a));
        switch(r)
        {
        case routine::csrsv:
        {
            const auto            b = right_hand_side(p.a.n);
            device_vector<double> x(b), y(sentinel(p.a.n));
            ASSERT_EQ(p.sv_solve(p.A, x, y), rocsparse_status_success);
            expect_close(to_host(y), host_lower_solve(p.a, b, 1), tol);
            check(p.csrsv_pivot(p.A), is(no_pivot), "csrsv_zero_pivot after the solve");
            return;
        }
        case routine::csrsm:
        {
            const auto            b = right_hand_side(p.a.n, 2);
            device_vector<double> dB(b);
            ASSERT_EQ(csrsm_solve(p, 2, dB), rocsparse_status_success);
            expect_close(to_host(dB), host_lower_solve(p.a, b, 2), tol);
            check(p.csrsm_pivot(), is(no_pivot), "csrsm_zero_pivot after the solve");
            return;
        }
        case routine::csrilu0:
            ASSERT_EQ(p.ilu0(), rocsparse_status_success);
            expect_close(p.values(), host_ilu0(p.a), tol);
            check(p.csrilu0_pivot(), is(no_pivot), "csrilu0_zero_pivot");
            return;
        case routine::csric0:
            ASSERT_EQ(p.ic0(), rocsparse_status_success);
            expect_close(
                keep_lower_only(p.a, p.values()), keep_lower_only(p.a, host_ic0(p.a)), tol);
            check(p.csric0_pivot(), is(no_pivot), "csric0_zero_pivot");
            return;
        }
    }

    // `consumer` analysed with the reuse policy after a forced analysis of `producer`,
    // then run on its own: it computes what it computes after an analysis of its own.
    void check_reuse(rocsparse_handle handle, routine producer, routine consumer)
    {
        SCOPED_TRACE(::testing::Message() << name(consumer) << " reusing " << name(producer));
        triangular_problem p(handle, make_grid_matrix<double>(4, false), rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_EQ(analyse(p, producer, rocsparse_analysis_policy_force), rocsparse_status_success);
        ASSERT_EQ(analyse(p, consumer, rocsparse_analysis_policy_reuse), rocsparse_status_success);
        ASSERT_NO_FATAL_FAILURE(check_compute(p, consumer));
    }

    // csrsm after a forced analysis of the same triangle by csrsv, csrilu0 or csric0.
    void check_csrsm_after(rocsparse_handle                     handle,
                           routine                              producer,
                           int32_t                              nrhs,
                           rocsparse_analysis_policy            policy,
                           const expectation<rocsparse_status>& solve)
    {
        const auto a = make_grid_matrix<double>(4, producer != routine::csric0);
        for(const auto base : bases)
        {
            SCOPED_TRACE(::testing::Message() << "after " << name(producer) << ", " << name(base)
                                              << ", nrhs " << nrhs << ", " << name(policy));
            triangular_problem p(handle, a, base);
            ASSERT_NO_FATAL_FAILURE(p.init());
            ASSERT_EQ(analyse(p, producer, rocsparse_analysis_policy_force),
                      rocsparse_status_success);
            csrsm_result out;
            ASSERT_NO_FATAL_FAILURE(run_csrsm(p, nrhs, policy, out));
            ASSERT_EQ(out.analysis, rocsparse_status_success);
            check(out.solve, solve, "csrsm_solve");
            if(out.solve == rocsparse_status_success)
            {
                expect_close(out.x,
                             host_lower_solve(a, right_hand_side(a.n, nrhs), nrhs),
                             tolerance<double>(max_row_length(a)));
                check(p.csrsm_pivot(), is(no_pivot), "csrsm_zero_pivot after the solve");
            }
        }
    }

    // csrilu0 and the solves with L and U once csrilu0 and csrsv have analysed both.
    void check_ilu0_workflow_after_forced_analysis(triangular_problem& p)
    {
        const auto b = right_hand_side(p.a.n);
        ASSERT_EQ(p.ilu0(), rocsparse_status_success);
        const auto lu = p.values();
        expect_close(lu, host_ilu0(p.a), tolerance<double>(max_row_length(p.a)));
        expect_close(
            p.lu_solve(b), host_lu_solve(p.a, lu, b), tolerance<double>(max_row_length(p.a)));
    }

    // The pivot queries, tolerance and clear of the factorization `r` (csrilu0 or
    // csric0).
    rocsparse_status factorize(triangular_problem& p, routine r)
    {
        return (r == routine::csrilu0) ? p.ilu0() : p.ic0();
    }

    int32_t factorization_pivot(triangular_problem& p, routine r)
    {
        return (r == routine::csrilu0) ? p.csrilu0_pivot() : p.csric0_pivot();
    }

    int32_t factorization_singular_pivot(triangular_problem& p, routine r)
    {
        return (r == routine::csrilu0) ? p.csrilu0_singular_pivot() : p.csric0_singular_pivot();
    }

    rocsparse_status set_tolerance(triangular_problem& p, routine r, double tol)
    {
        return (r == routine::csrilu0) ? rocsparse_csrilu0_set_tolerance(p.handle, p.info, tol)
                                       : rocsparse_csric0_set_tolerance(p.handle, p.info, tol);
    }

    double get_tolerance(triangular_problem& p, routine r)
    {
        double tol = -1.0;
        EXPECT_EQ((r == routine::csrilu0) ? rocsparse_csrilu0_get_tolerance(p.handle, p.info, &tol)
                                          : rocsparse_csric0_get_tolerance(p.handle, p.info, &tol),
                  rocsparse_status_success);
        return tol;
    }

    rocsparse_status clear(triangular_problem& p, routine r)
    {
        return (r == routine::csrilu0) ? rocsparse_csrilu0_clear(p.handle, p.info)
                                       : rocsparse_csric0_clear(p.handle, p.info);
    }

    constexpr routine factorizations[] = {routine::csrilu0, routine::csric0};
}

class PrecondTriangularScenarios : public HandleTest
{
};

// ---------------------------------------------------------------------------
// csrsv on its own
// ---------------------------------------------------------------------------

// A missing diagonal entry is a zero pivot of a non unit triangle, reported after the
// analysis and after the solve; with a unit diagonal it is not a pivot.
TEST_F(PrecondTriangularScenarios, csrsv_structural_pivot)
{
    const int32_t r = 4;
    const auto    a = without_diagonal(make_grid_matrix<double>(3, false), r);
    for(const auto base : bases)
    {
        for(const bool device_query : {false, true})
        {
            for(const auto fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
            {
                for(const auto diag : {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit})
                {
                    SCOPED_TRACE(::testing::Message()
                                 << name(base) << ", device query " << device_query << ", "
                                 << name(fill) << ", " << name(diag));
                    triangular_problem p(handle, a, base, device_query);
                    ASSERT_NO_FATAL_FAILURE(p.init());
                    const auto descr = (fill == rocsparse_fill_mode_lower) ? p.L : p.U;
                    ASSERT_EQ(rocsparse_set_mat_diag_type(descr, diag), rocsparse_status_success);
                    const int32_t want = (diag == rocsparse_diag_type_non_unit) ? r : no_pivot;

                    ASSERT_EQ(p.sv_analysis(descr, rocsparse_analysis_policy_force),
                              rocsparse_status_success);
                    check(p.csrsv_pivot(descr), is(want), "csrsv_zero_pivot after the analysis");

                    device_vector<double> x(right_hand_side(a.n)), y(a.n);
                    ASSERT_EQ(p.sv_solve(descr, x, y), rocsparse_status_success);
                    check(p.csrsv_pivot(descr), is(want), "csrsv_zero_pivot after the solve");
                }
            }
        }
    }
}

// A stored zero on the diagonal is a numerical zero pivot: found by the solve of a non
// unit triangle, not by the analysis, and gone again once the values are restored.
TEST_F(PrecondTriangularScenarios, csrsv_numerical_pivot)
{
    const int32_t r     = 5;
    const auto    clean = make_grid_matrix<double>(3, false);
    const auto    zero  = with_zero_diagonal(clean, r).val;
    for(const auto base : bases)
    {
        for(const auto fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
        {
            for(const auto diag : {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit})
            {
                SCOPED_TRACE(::testing::Message()
                             << name(base) << ", " << name(fill) << ", " << name(diag));
                triangular_problem p(handle, clean, base);
                ASSERT_NO_FATAL_FAILURE(p.init());
                const auto descr = (fill == rocsparse_fill_mode_lower) ? p.L : p.U;
                ASSERT_EQ(rocsparse_set_mat_diag_type(descr, diag), rocsparse_status_success);
                ASSERT_NO_FATAL_FAILURE(p.set_values(&zero));

                ASSERT_EQ(p.sv_analysis(descr, rocsparse_analysis_policy_force),
                          rocsparse_status_success);
                check(p.csrsv_pivot(descr), is(no_pivot), "csrsv_zero_pivot after the analysis");

                device_vector<double> x(right_hand_side(clean.n)), y(clean.n);
                ASSERT_EQ(p.sv_solve(descr, x, y), rocsparse_status_success);
                check(p.csrsv_pivot(descr),
                      is(diag == rocsparse_diag_type_non_unit ? r : no_pivot),
                      "csrsv_zero_pivot after the solve");

                ASSERT_NO_FATAL_FAILURE(p.set_values());
                ASSERT_EQ(p.sv_solve(descr, x, y), rocsparse_status_success);
                check(p.csrsv_pivot(descr),
                      is(no_pivot),
                      "csrsv_zero_pivot after the solve with the restored values");
            }
        }
    }
}

// A reused analysis keeps the diagonal type it was made with: after a unit diagonal
// analysis of L, a non unit analysis of the same triangle with the reuse policy does
// not record the missing diagonal, and the solve does not report it either.
TEST_F(PrecondTriangularScenarios, csrsv_reanalysis_with_another_diagonal_type)
{
    const int32_t r = 4;
    const auto    a = without_diagonal(make_grid_matrix<double>(3, false), r);
    for(const auto base : bases)
    {
        for(const auto policy : policies)
        {
            SCOPED_TRACE(::testing::Message() << name(base) << ", " << name(policy));
            triangular_problem p(handle, a, base);
            ASSERT_NO_FATAL_FAILURE(p.init());
            ASSERT_EQ(p.sv_analysis(p.L, rocsparse_analysis_policy_force),
                      rocsparse_status_success);
            ASSERT_EQ(p.sv_analysis(p.A, policy), rocsparse_status_success);
            check(p.csrsv_pivot(p.A),
                  is(policy == rocsparse_analysis_policy_reuse ? no_pivot : r),
                  "csrsv_zero_pivot after the non unit analysis");

            device_vector<double> x(right_hand_side(a.n)), y(a.n);
            ASSERT_EQ(p.sv_solve(p.A, x, y), rocsparse_status_success);
            check(p.csrsv_pivot(p.A),
                  (policy == rocsparse_analysis_policy_reuse)
                      ? known_defect(no_pivot,
                                     r,
                                     "a reused unit diagonal analysis does not record the "
                                     "structural pivot of the non unit solve")
                      : is(r),
                  "csrsv_zero_pivot after the non unit solve");
        }
    }
}

// ---------------------------------------------------------------------------
// The ILU(0) workflow on one mat_info
// ---------------------------------------------------------------------------

// csrilu0_analysis, the analyses of L and U in either order, csrilu0, then the solves
// with L and U: the factor and the preconditioned vector match the host, and no zero
// pivot is reported.
TEST_F(PrecondTriangularScenarios, ilu0_workflow_values)
{
    const auto a = make_grid_matrix<double>(4, true);
    const auto b = right_hand_side(a.n);
    for(const auto base : bases)
    {
        for(const auto o : orders)
        {
            for(const auto policy : policies)
            {
                SCOPED_TRACE(::testing::Message()
                             << name(base) << ", " << name(o) << ", " << name(policy));
                triangular_problem p(handle, a, base);
                ASSERT_NO_FATAL_FAILURE(p.init());
                ASSERT_NO_FATAL_FAILURE(p.ilu0_workflow(o, policy));
                check(p.csrsv_pivot(p.U), is(no_pivot), "csrsv_zero_pivot after the analyses");
                check(p.csrilu0_pivot(), is(no_pivot), "csrilu0_zero_pivot");

                const auto lu = p.values();
                expect_close(lu, host_ilu0(a), tolerance<double>(max_row_length(a)));
                expect_close(
                    p.lu_solve(b), host_lu_solve(a, lu, b), tolerance<double>(max_row_length(a)));
                check(p.csrsv_pivot(p.U), is(no_pivot), "csrsv_zero_pivot after the solves");
                check(p.csrilu0_pivot(), is(no_pivot), "csrilu0_zero_pivot after the solves");
            }
        }
    }
}

// New values on the analysed pattern: csrilu0 and the solves run again without a new
// analysis and match the host for the new values.
TEST_F(PrecondTriangularScenarios, ilu0_workflow_new_values)
{
    const auto a     = make_grid_matrix<double>(4, true);
    const auto a_new = with_scaled_diagonal(a, 1.5);
    const auto b     = right_hand_side(a.n);
    for(const auto base : bases)
    {
        for(const auto o : orders)
        {
            for(const auto policy : policies)
            {
                SCOPED_TRACE(::testing::Message()
                             << name(base) << ", " << name(o) << ", " << name(policy));
                triangular_problem p(handle, a, base);
                ASSERT_NO_FATAL_FAILURE(p.init());
                ASSERT_NO_FATAL_FAILURE(p.ilu0_workflow(o, policy));
                (void)p.lu_solve(b);

                ASSERT_NO_FATAL_FAILURE(p.set_values(&a_new.val));
                ASSERT_EQ(p.ilu0(), rocsparse_status_success);
                const auto lu = p.values();
                expect_close(lu, host_ilu0(a_new), tolerance<double>(max_row_length(a)));
                expect_close(p.lu_solve(b),
                             host_lu_solve(a_new, lu, b),
                             tolerance<double>(max_row_length(a)));
                check(p.csrilu0_pivot(), is(no_pivot), "csrilu0_zero_pivot");
                check(p.csrsv_pivot(p.U), is(no_pivot), "csrsv_zero_pivot");
            }
        }
    }
}

// The workflow on a matrix whose row r has no diagonal entry: r is a zero pivot of A
// and of U, not of the unit diagonal L. After the solve with U, csrsv_zero_pivot must
// report r ("structural or numerical zero found during the solve").
TEST_F(PrecondTriangularScenarios, ilu0_workflow_structural_pivot)
{
    const int32_t r = 4;
    const auto    a = without_diagonal(make_grid_matrix<double>(3, false), r);
    const auto    b = right_hand_side(a.n);
    for(const auto base : bases)
    {
        for(const bool device_query : {false, true})
        {
            for(const auto o : orders)
            {
                for(const auto policy : policies)
                {
                    SCOPED_TRACE(::testing::Message()
                                 << name(base) << ", device query " << device_query << ", "
                                 << name(o) << ", " << name(policy));
                    triangular_problem p(handle, a, base, device_query);
                    ASSERT_NO_FATAL_FAILURE(p.init());

                    ASSERT_EQ(p.ilu0_analysis(policy), rocsparse_status_success);
                    check(p.csrilu0_pivot(), is(r), "csrilu0_zero_pivot after csrilu0_analysis");

                    const std::string after_l = "csrsv_zero_pivot after the analysis of L";
                    const std::string after_u = "csrsv_zero_pivot after the analysis of U";
                    if(o == order::lower_first)
                    {
                        // With reuse, L takes the analysis of csrilu0 and csrsv has no
                        // zero pivot yet; the analysis of U creates it.
                        ASSERT_EQ(p.sv_analysis(p.L, policy), rocsparse_status_success);
                        check(p.csrsv_pivot(p.L), is(no_pivot), after_l);
                        ASSERT_EQ(p.sv_analysis(p.U, policy), rocsparse_status_success);
                        check(p.csrsv_pivot(p.U), is(r), after_u);
                    }
                    else
                    {
                        ASSERT_EQ(p.sv_analysis(p.U, policy), rocsparse_status_success);
                        check(p.csrsv_pivot(p.U), is(r), after_u);
                        // The zero pivot of csrsv is shared by L and U: a reused analysis
                        // of L leaves the one of U, a new one resets it.
                        ASSERT_EQ(p.sv_analysis(p.L, policy), rocsparse_status_success);
                        check(p.csrsv_pivot(p.L),
                              is(policy == rocsparse_analysis_policy_reuse ? r : no_pivot),
                              after_l);
                    }

                    ASSERT_EQ(p.ilu0(), rocsparse_status_success);
                    check(p.csrilu0_pivot(), is(r), "csrilu0_zero_pivot after csrilu0");

                    // The solve kernel never sees a missing diagonal: the solve with U
                    // reports r only through the structural pivot of the analysis.
                    device_vector<double> x(b), y(a.n), z(a.n);
                    ASSERT_EQ(p.sv_solve(p.L, x, y), rocsparse_status_success);
                    check(p.csrsv_pivot(p.L),
                          is(no_pivot),
                          "csrsv_zero_pivot after the solve with L");
                    ASSERT_EQ(p.sv_solve(p.U, y, z), rocsparse_status_success);
                    check(p.csrsv_pivot(p.U),
                          known_defect(no_pivot,
                                       r,
                                       "the solve with the unit diagonal L, or a new analysis of "
                                       "L, resets the structural pivot shared with U"),
                          "csrsv_zero_pivot after the solve with U");
                    check(p.csrilu0_pivot(), is(r), "csrilu0_zero_pivot after the solves");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The IC(0) workflow on one mat_info
// ---------------------------------------------------------------------------

// csric0_analysis, the analyses of A and A^T in either order, csric0, then the solves
// with L and L^T: the factor and the preconditioned vector match the host, and no zero
// pivot is reported.
TEST_F(PrecondTriangularScenarios, ic0_workflow_values)
{
    const auto a   = make_grid_matrix<double>(4, false);
    const auto b   = right_hand_side(a.n);
    const auto tol = tolerance<double>(max_row_length(a));
    for(const auto base : bases)
    {
        for(const auto o : orders)
        {
            for(const auto policy : policies)
            {
                SCOPED_TRACE(::testing::Message()
                             << name(base) << ", " << name(o) << ", " << name(policy));
                triangular_problem p(handle, a, base);
                ASSERT_NO_FATAL_FAILURE(p.init());
                ASSERT_NO_FATAL_FAILURE(p.ic0_workflow(o, policy));
                check(p.csric0_pivot(), is(no_pivot), "csric0_zero_pivot");

                const auto l = p.values();
                expect_close(keep_lower_only(a, l), keep_lower_only(a, host_ic0(a)), tol);
                expect_close(p.llt_solve(b), host_llt_solve(a, l, b), tol);
                check(p.csrsv_pivot(p.A), is(no_pivot), "csrsv_zero_pivot after the solves");
                check(p.csric0_pivot(), is(no_pivot), "csric0_zero_pivot after the solves");
            }
        }
    }
}

// csric0 needs its analysis: it returns rocsparse_status_invalid_pointer without one and
// after csric0_clear, leaving the values alone. csric0_clear keeps the analyses of the
// solves.
TEST_F(PrecondTriangularScenarios, csric0_needs_its_analysis)
{
    const auto a   = make_grid_matrix<double>(4, false);
    const auto b   = right_hand_side(a.n);
    const auto tol = tolerance<double>(max_row_length(a));
    for(const auto policy : policies)
    {
        SCOPED_TRACE(name(policy));
        triangular_problem p(handle, a, rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        check(p.ic0(), is(rocsparse_status_invalid_pointer), "csric0 without an analysis");
        EXPECT_EQ(p.values(), a.val);

        ASSERT_NO_FATAL_FAILURE(p.ic0_workflow(order::lower_first, policy));
        const auto l = p.values();
        ASSERT_EQ(rocsparse_csric0_clear(handle, p.info), rocsparse_status_success);
        check(p.csric0_pivot(), is(no_pivot), "csric0_zero_pivot after csric0_clear");
        check(p.ic0(), is(rocsparse_status_invalid_pointer), "csric0 after csric0_clear");
        EXPECT_EQ(p.values(), l);
        expect_close(p.llt_solve(b), host_llt_solve(a, l, b), tol);
    }
}

// ---------------------------------------------------------------------------
// csrsm after the analysis of another routine
// ---------------------------------------------------------------------------

TEST_F(PrecondTriangularScenarios, csrsm_after_another_analysis)
{
    for(const auto producer : {routine::csrsv, routine::csrilu0, routine::csric0})
    {
        check_csrsm_after(
            handle, producer, 1, rocsparse_analysis_policy_force, is(rocsparse_status_success));
        check_csrsm_after(handle,
                          producer,
                          1,
                          rocsparse_analysis_policy_reuse,
                          known_defect(rocsparse_status_invalid_pointer,
                                       rocsparse_status_success,
                                       "with one right-hand side the reused analysis is "
                                       "stored for csrsv, so the solve of csrsm finds none"));
        check_csrsm_after(
            handle, producer, 2, rocsparse_analysis_policy_force, is(rocsparse_status_success));
#ifndef ROCSPARSE_WITH_TRSM_REFACTORING
        check_csrsm_after(
            handle, producer, 2, rocsparse_analysis_policy_reuse, is(rocsparse_status_success));
#endif
    }
}

// The zero pivot of csrsm is its own, with one right-hand side too (where csrsm runs
// on csrsv): csrsv_zero_pivot does not report it.
TEST_F(PrecondTriangularScenarios, csrsm_structural_pivot)
{
    const int32_t r = 4;
    const auto    a = without_diagonal(make_grid_matrix<double>(3, false), r);
    for(const auto base : bases)
    {
        for(const int32_t nrhs : {1, 2})
        {
            SCOPED_TRACE(::testing::Message() << name(base) << ", nrhs " << nrhs);
            triangular_problem p(handle, a, base);
            ASSERT_NO_FATAL_FAILURE(p.init());
            csrsm_result out;
            ASSERT_NO_FATAL_FAILURE(run_csrsm(p, nrhs, rocsparse_analysis_policy_force, out));
            ASSERT_EQ(out.analysis, rocsparse_status_success);
            ASSERT_EQ(out.solve, rocsparse_status_success);
            check(p.csrsm_pivot(), is(r), "csrsm_zero_pivot after the solve");
            check(p.csrsv_pivot(p.A), is(no_pivot), "csrsv_zero_pivot after the csrsm solve");
        }
    }
}

// ---------------------------------------------------------------------------
// Clear and copy of a mat_info whose analysis data is shared
// ---------------------------------------------------------------------------

// Clearing one routine keeps the analysis data that the other routines use; the
// cleared routine needs a new analysis.
TEST_F(PrecondTriangularScenarios, clear_keeps_the_other_routines)
{
    const auto a = make_grid_matrix<double>(4, true);
    const auto b = right_hand_side(a.n);
    for(const auto base : bases)
    {
        for(const auto policy : policies)
        {
            for(const bool clear_csrsv : {true, false})
            {
                SCOPED_TRACE(::testing::Message()
                             << name(base) << ", " << name(policy) << ", clear "
                             << (clear_csrsv ? "csrsv" : "csrilu0"));
                triangular_problem p(handle, a, base);
                ASSERT_NO_FATAL_FAILURE(p.init());
                ASSERT_NO_FATAL_FAILURE(p.ilu0_workflow(order::lower_first, policy));
                const auto lu = p.values();

                if(clear_csrsv)
                {
                    ASSERT_EQ(rocsparse_csrsv_clear(handle, p.L, p.info), rocsparse_status_success);
                    check(p.csrsv_pivot(p.L), is(no_pivot), "csrsv_zero_pivot after csrsv_clear");
                    // csrsv_clear drops the analyses of both triangles.
                    device_vector<double> x(b), y(a.n);
                    check(p.sv_solve(p.U, x, y),
                          is(rocsparse_status_invalid_pointer),
                          "csrsv_solve with U after csrsv_clear");
                    check(p.sv_solve(p.L, x, y),
                          is(rocsparse_status_invalid_pointer),
                          "csrsv_solve with L after csrsv_clear");

                    ASSERT_NO_FATAL_FAILURE(p.set_values());
                    ASSERT_EQ(p.ilu0(), rocsparse_status_success);
                    expect_close(p.values(), lu, tolerance<double>(max_row_length(a)));
                    ASSERT_EQ(p.sv_analysis(p.L, policy), rocsparse_status_success);
                    ASSERT_EQ(p.sv_analysis(p.U, policy), rocsparse_status_success);
                }
                else
                {
                    ASSERT_EQ(rocsparse_csrilu0_clear(handle, p.info), rocsparse_status_success);
                    check(
                        p.csrilu0_pivot(), is(no_pivot), "csrilu0_zero_pivot after csrilu0_clear");
                }
                expect_close(
                    p.lu_solve(b), host_lu_solve(a, lu, b), tolerance<double>(max_row_length(a)));
                check(p.csrsv_pivot(p.U), is(no_pivot), "csrsv_zero_pivot after the solves");
            }
        }
    }
}

// csrsm_clear drops the analysis of csrsm, with one right-hand side too.
TEST_F(PrecondTriangularScenarios, csrsm_clear_needs_a_new_analysis)
{
    const auto a = make_grid_matrix<double>(4, true);
    for(const auto base : bases)
    {
        SCOPED_TRACE(name(base));
        triangular_problem p(handle, a, base);
        ASSERT_NO_FATAL_FAILURE(p.init());
        csrsm_result out;
        ASSERT_NO_FATAL_FAILURE(run_csrsm(p, 1, rocsparse_analysis_policy_force, out));
        ASSERT_EQ(out.solve, rocsparse_status_success);
        ASSERT_EQ(rocsparse_csrsm_clear(handle, p.info), rocsparse_status_success);

        device_vector<double> dB(right_hand_side(a.n));
        check(csrsm_solve(p, 1, dB),
              is(rocsparse_status_invalid_pointer),
              "csrsm_solve after csrsm_clear");
        check(p.csrsm_pivot(), is(no_pivot), "csrsm_zero_pivot after csrsm_clear");
    }
}

// A copy of the mat_info carries the analysis data and the zero pivots, and stays usable
// for solves once the original is destroyed.
TEST_F(PrecondTriangularScenarios, copy_outlives_the_original)
{
    const int32_t r     = 4;
    const auto    clean = make_grid_matrix<double>(3, true);
    const auto    b     = right_hand_side(clean.n);
    for(const bool missing_diagonal : {false, true})
    {
        const auto    a    = missing_diagonal ? without_diagonal(clean, r) : clean;
        const int32_t want = missing_diagonal ? r : no_pivot;
        for(const auto base : bases)
        {
            for(const auto policy : policies)
            {
                SCOPED_TRACE(::testing::Message() << "missing diagonal " << missing_diagonal << ", "
                                                  << name(base) << ", " << name(policy));
                triangular_problem p(handle, a, base);
                ASSERT_NO_FATAL_FAILURE(p.init());
                ASSERT_NO_FATAL_FAILURE(p.ilu0_workflow(order::lower_first, policy));
                const auto lu = p.values();

                ASSERT_EQ(p.replace_info_by_copy(false), rocsparse_status_success);
                check(p.csrilu0_pivot(), is(want), "csrilu0_zero_pivot of the copy");
                check(p.csrsv_pivot(p.U), is(want), "csrsv_zero_pivot of the copy");
                if(!missing_diagonal)
                {
                    expect_close(p.lu_solve(b),
                                 host_lu_solve(a, lu, b),
                                 tolerance<double>(max_row_length(a)));
                    check(p.csrsv_pivot(p.U), is(no_pivot), "csrsv_zero_pivot after the solves");
                }
            }
        }
    }
}

// The copy does not share analysis data with the original: a new analysis on the copy
// leaves the original untouched and the copy computes the whole workflow again.
TEST_F(PrecondTriangularScenarios, copy_is_independent_of_the_original)
{
    const auto a = make_grid_matrix<double>(4, true);
    const auto b = right_hand_side(a.n);
    for(const auto base : bases)
    {
        for(const auto policy : policies)
        {
            SCOPED_TRACE(::testing::Message() << name(base) << ", " << name(policy));
            triangular_problem p(handle, a, base);
            ASSERT_NO_FATAL_FAILURE(p.init());
            ASSERT_NO_FATAL_FAILURE(p.ilu0_workflow(order::lower_first, policy));
            const auto lu = p.values();

            ASSERT_EQ(p.replace_info_by_copy(true), rocsparse_status_success);
            ASSERT_EQ(p.sv_analysis(p.L, rocsparse_analysis_policy_force),
                      rocsparse_status_success);
            ASSERT_NO_FATAL_FAILURE(p.set_values());
            ASSERT_EQ(p.ilu0(), rocsparse_status_success);
            expect_close(p.values(), lu, tolerance<double>(max_row_length(a)));
            expect_close(
                p.lu_solve(b), host_lu_solve(a, lu, b), tolerance<double>(max_row_length(a)));

            std::swap(p.info, p.original);
            expect_close(
                p.lu_solve(b), host_lu_solve(a, lu, b), tolerance<double>(max_row_length(a)));
            check(p.csrilu0_pivot(), is(no_pivot), "csrilu0_zero_pivot of the original");
        }
    }
}

// The tolerance of csrilu0 is copied by value: changing the one of the original does not
// change the singular pivot found on the copy.
TEST_F(PrecondTriangularScenarios, copy_has_its_own_tolerance)
{
    const auto a = make_grid_matrix<double>(4, true);
    for(const auto base : bases)
    {
        SCOPED_TRACE(name(base));
        triangular_problem p(handle, a, base);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_NO_FATAL_FAILURE(
            p.ilu0_workflow(order::lower_first, rocsparse_analysis_policy_force));
        ASSERT_EQ(p.replace_info_by_copy(true), rocsparse_status_success);

        // Every pivot of the grid matrix is below 10.
        ASSERT_EQ(rocsparse_csrilu0_set_tolerance(handle, p.original, 10.0),
                  rocsparse_status_success);
        ASSERT_NO_FATAL_FAILURE(p.set_values());
        ASSERT_EQ(p.ilu0(), rocsparse_status_success);
        double tol = -1.0;
        ASSERT_EQ(rocsparse_csrilu0_get_tolerance(handle, p.info, &tol), rocsparse_status_success);
        EXPECT_EQ(tol, 0.0);
        check(p.csrilu0_singular_pivot(),
              known_defect(int32_t{0},
                           no_pivot,
                           "copy_mat_info copies the pointer to the tolerance, so the copy reads "
                           "the tolerance of the original"),
              "csrilu0_singular_pivot of the copy");
    }
}

// copy_mat_info into itself, and into a mat_info analysed for another matrix. The copy
// into itself is rejected on purpose (copy_mat_info checks src == dest) as an invalid
// pointer; the page lists invalid_pointer only for an invalid src or dest and needs to
// say so. The page asks only for an initialized src and dest, so the other copy is to
// succeed and replace what the destination held, as in copy_replaces_the_destination.
// It stops with an internal error at the first analysis of another size instead,
// leaving what it copied before that in the destination.
TEST_F(PrecondTriangularScenarios, copy_mat_info_status)
{
    triangular_problem small(handle, make_grid_matrix<double>(3, true), rocsparse_index_base_zero);
    triangular_problem large(handle, make_grid_matrix<double>(4, true), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(small.init());
    ASSERT_NO_FATAL_FAILURE(large.init());
    ASSERT_EQ(small.sv_analysis(small.L, rocsparse_analysis_policy_force),
              rocsparse_status_success);
    ASSERT_EQ(large.sv_analysis(large.L, rocsparse_analysis_policy_force),
              rocsparse_status_success);

    check(rocsparse_copy_mat_info(small.info, small.info),
          is(rocsparse_status_invalid_pointer),
          "copy_mat_info into itself");
    check(rocsparse_copy_mat_info(small.info, large.info),
          known_defect(rocsparse_status_internal_error,
                       rocsparse_status_success,
                       "a copy into a mat_info analysed for another size fails part way through"),
          "copy_mat_info into a mat_info analysed for another matrix");
}

// destroy_mat_info of a null pointer returns early with success, a no-op as free is; the
// page lists invalid_pointer for an invalid info and needs to exempt the null pointer.
TEST_F(PrecondTriangularScenarios, destroy_mat_info_of_a_null_pointer)
{
    check(rocsparse_destroy_mat_info(nullptr),
          is(rocsparse_status_success),
          "destroy_mat_info(nullptr)");
}

// The copy replaces what the destination held: an analysis that only the destination
// had is gone.
TEST_F(PrecondTriangularScenarios, copy_replaces_the_destination)
{
    const auto         a = make_grid_matrix<double>(4, true);
    triangular_problem dest(handle, a, rocsparse_index_base_zero);
    triangular_problem src(handle, a, rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(dest.init());
    ASSERT_NO_FATAL_FAILURE(src.init());
    ASSERT_NO_FATAL_FAILURE(
        dest.ilu0_workflow(order::lower_first, rocsparse_analysis_policy_force));
    ASSERT_EQ(src.ilu0_analysis(rocsparse_analysis_policy_force), rocsparse_status_success);

    ASSERT_EQ(rocsparse_copy_mat_info(dest.info, src.info), rocsparse_status_success);
    const auto            b = right_hand_side(a.n);
    device_vector<double> x(b), y(a.n);
    check(dest.sv_solve(dest.L, x, y),
          known_defect(rocsparse_status_success,
                       rocsparse_status_invalid_pointer,
                       "copy_mat_info keeps the analyses of the destination that the source "
                       "does not have"),
          "csrsv_solve with L on a copy of a mat_info without a csrsv analysis");
}

// ---------------------------------------------------------------------------
// Analyses, clear and a new pattern
// ---------------------------------------------------------------------------

// A solve needs the analysis of its own triangle and operation: with the analysis of
// another one it returns rocsparse_status_invalid_pointer and leaves y alone.
TEST_F(PrecondTriangularScenarios, solve_needs_the_analysis_of_its_triangle)
{
    const auto a = make_grid_matrix<double>(4, true);
    const auto b = right_hand_side(a.n);
    for(const auto policy : policies)
    {
        SCOPED_TRACE(name(policy));
        triangular_problem p(handle, a, rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_EQ(p.sv_analysis(p.A, policy), rocsparse_status_success);
        ASSERT_EQ(p.sv_analysis(p.L, policy), rocsparse_status_success);

        const std::vector<double> untouched(a.n, 7.0);
        device_vector<double>     x(b), y(untouched);
        check(p.sv_solve(p.A, x, y, rocsparse_operation_transpose),
              is(rocsparse_status_invalid_pointer),
              "csrsv_solve with A^T after the analysis of A");
        EXPECT_EQ(to_host(y), untouched);
        check(p.sv_solve(p.U, x, y),
              is(rocsparse_status_invalid_pointer),
              "csrsv_solve with U after the analysis of L");
        EXPECT_EQ(to_host(y), untouched);
    }
}

// Clearing csrsv keeps the analysis of csrsm and the other way round, also when one
// reused the analysis of the other. The analysis of U first gives csrsv its zero pivot,
// which a reused analysis does not create.
TEST_F(PrecondTriangularScenarios, csrsv_and_csrsm_clear_keep_each_other)
{
    const auto a   = make_grid_matrix<double>(4, true);
    const auto b   = right_hand_side(a.n);
    const auto b2  = right_hand_side(a.n, 2);
    const auto tol = tolerance<double>(max_row_length(a));
    for(const auto policy : policies)
    {
        for(const bool clear_csrsv : {true, false})
        {
            SCOPED_TRACE(::testing::Message()
                         << name(policy) << ", clear " << (clear_csrsv ? "csrsv" : "csrsm"));
            triangular_problem p(handle, a, rocsparse_index_base_zero);
            ASSERT_NO_FATAL_FAILURE(p.init());
            ASSERT_EQ(p.sv_analysis(p.U, rocsparse_analysis_policy_force),
                      rocsparse_status_success);
            ASSERT_EQ(csrsm_analysis(p, 2, rocsparse_analysis_policy_force),
                      rocsparse_status_success);
            ASSERT_EQ(p.sv_analysis(p.A, policy), rocsparse_status_success);

            if(clear_csrsv)
            {
                ASSERT_EQ(rocsparse_csrsv_clear(handle, p.A, p.info), rocsparse_status_success);
                device_vector<double> dB(b2);
                ASSERT_EQ(csrsm_solve(p, 2, dB), rocsparse_status_success);
                expect_close(to_host(dB), host_lower_solve(a, b2, 2), tol);
            }
            else
            {
                ASSERT_EQ(rocsparse_csrsm_clear(handle, p.info), rocsparse_status_success);
                device_vector<double> x(b), y(a.n);
                ASSERT_EQ(p.sv_solve(p.A, x, y), rocsparse_status_success);
                expect_close(to_host(y), host_lower_solve(a, b, 1), tol);
            }
        }
    }
}

// After csrilu0_clear, a forced csrilu0 analysis gives csrilu0 an analysis of its own
// again, also when csrsv reused the cleared one.
TEST_F(PrecondTriangularScenarios, csrilu0_new_analysis_after_csrilu0_clear)
{
    const auto a   = make_grid_matrix<double>(4, true);
    const auto b   = right_hand_side(a.n);
    const auto tol = tolerance<double>(max_row_length(a));
    for(const auto policy : policies)
    {
        SCOPED_TRACE(name(policy));
        triangular_problem p(handle, a, rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_NO_FATAL_FAILURE(p.ilu0_workflow(order::lower_first, policy));
        ASSERT_EQ(rocsparse_csrilu0_clear(handle, p.info), rocsparse_status_success);

        ASSERT_NO_FATAL_FAILURE(p.set_values());
        ASSERT_EQ(p.ilu0_analysis(rocsparse_analysis_policy_force), rocsparse_status_success);
        ASSERT_EQ(p.ilu0(), rocsparse_status_success);
        const auto lu = p.values();
        expect_close(lu, host_ilu0(a), tol);
        check(p.csrilu0_pivot(), is(no_pivot), "csrilu0_zero_pivot");
        expect_close(p.lu_solve(b), host_lu_solve(a, lu, b), tol);
    }
}

// The analysis belongs to the pattern it was made for. A new pattern on the mat_info
// needs csrilu0_clear and csrsv_clear first, or forced analyses of every routine when
// none shares an analysis (after a reuse the forced analysis frees data in use).
TEST_F(PrecondTriangularScenarios, ilu0_workflow_new_pattern)
{
    const auto a     = make_grid_matrix<double>(4, true);
    const auto a_new = without_entry(without_entry(a, 5, 4), 4, 5);
    const auto b     = right_hand_side(a.n);
    const auto tol   = tolerance<double>(max_row_length(a));
    for(const auto o : orders)
    {
        for(const auto policy : policies)
        {
            for(const bool clear : {true, false})
            {
                if(!clear && policy == rocsparse_analysis_policy_reuse)
                    continue;
                SCOPED_TRACE(::testing::Message() << name(o) << ", " << name(policy)
                                                  << (clear ? ", cleared" : ", forced"));
                triangular_problem p(handle, a, rocsparse_index_base_zero);
                ASSERT_NO_FATAL_FAILURE(p.init());
                ASSERT_NO_FATAL_FAILURE(p.ilu0_workflow(o, policy));
                (void)p.lu_solve(b);
                if(clear)
                {
                    ASSERT_EQ(rocsparse_csrilu0_clear(handle, p.info), rocsparse_status_success);
                    ASSERT_EQ(rocsparse_csrsv_clear(handle, p.L, p.info), rocsparse_status_success);
                }

                ASSERT_NO_FATAL_FAILURE(p.replace_matrix(a_new));
                ASSERT_NO_FATAL_FAILURE(p.ilu0_workflow(o, policy));
                const auto lu = p.values();
                expect_close(lu, host_ilu0(a_new), tol);
                check(p.csrilu0_pivot(), is(no_pivot), "csrilu0_zero_pivot");
                expect_close(p.lu_solve(b), host_lu_solve(a_new, lu, b), tol);
                check(p.csrsv_pivot(p.U), is(no_pivot), "csrsv_zero_pivot after the solves");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Zero pivots of the solves
// ---------------------------------------------------------------------------

// Before any analysis no routine reports a zero or singular pivot.
TEST_F(PrecondTriangularScenarios, pivots_of_a_fresh_mat_info)
{
    for(const auto base : bases)
    {
        for(const bool device_query : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << name(base) << ", device query " << device_query);
            triangular_problem p(handle, make_grid_matrix<double>(3, false), base, device_query);
            ASSERT_NO_FATAL_FAILURE(p.init());
            check(p.csrsv_pivot(p.L), is(no_pivot), "csrsv_zero_pivot");
            check(p.csrsm_pivot(), is(no_pivot), "csrsm_zero_pivot");
            check(p.csrilu0_pivot(), is(no_pivot), "csrilu0_zero_pivot");
            check(p.csric0_pivot(), is(no_pivot), "csric0_zero_pivot");
            check(p.csrilu0_singular_pivot(), is(no_pivot), "csrilu0_singular_pivot");
            check(p.csric0_singular_pivot(), is(no_pivot), "csric0_singular_pivot");
        }
    }
}

// With a structural and a numerical zero pivot, the solve reports the first one,
// whichever kind it is, for the triangle and for its transpose. The analysis only finds
// the structural one.
TEST_F(PrecondTriangularScenarios, csrsv_first_of_several_zero_pivots)
{
    const auto                        clean   = make_grid_matrix<double>(3, false);
    const std::pair<int32_t, int32_t> cases[] = {{2, 6}, {6, 2}};
    for(const auto& [numerical, structural] : cases)
    {
        const auto a = without_diagonal(with_zero_diagonal(clean, numerical), structural);
        for(const auto base : bases)
        {
            for(const auto op : {rocsparse_operation_none, rocsparse_operation_transpose})
            {
                SCOPED_TRACE(::testing::Message()
                             << "numerical zero " << numerical << ", structural zero " << structural
                             << ", " << name(base) << ", "
                             << (op == rocsparse_operation_none ? "L" : "L^T"));
                triangular_problem p(handle, a, base);
                ASSERT_NO_FATAL_FAILURE(p.init());
                ASSERT_EQ(p.sv_analysis(p.A, rocsparse_analysis_policy_force, op),
                          rocsparse_status_success);
                check(p.csrsv_pivot(p.A), is(structural), "csrsv_zero_pivot after the analysis");
                device_vector<double> x(right_hand_side(a.n)), y(a.n);
                ASSERT_EQ(p.sv_solve(p.A, x, y, op), rocsparse_status_success);
                check(p.csrsv_pivot(p.A),
                      is(std::min(numerical, structural)),
                      "csrsv_zero_pivot after the solve");
            }
        }
    }
}

// csrsv keeps one zero pivot for both triangles of the mat_info, and the documentation
// makes it the pivot of the matrix: the structural zero of the non unit lower triangle
// is still reported after the analysis and the solve of a unit upper triangle.
TEST_F(PrecondTriangularScenarios, csrsv_pivot_of_one_triangle_survives_the_other)
{
    const int32_t r = 4;
    const auto    a = without_diagonal(make_grid_matrix<double>(3, false), r);
    const auto    b = right_hand_side(a.n);
    for(const auto base : bases)
    {
        SCOPED_TRACE(name(base));
        triangular_problem p(handle, a, base);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_EQ(rocsparse_set_mat_diag_type(p.U, rocsparse_diag_type_unit),
                  rocsparse_status_success);
        ASSERT_EQ(p.sv_analysis(p.A, rocsparse_analysis_policy_force), rocsparse_status_success);
        check(p.csrsv_pivot(p.A), is(r), "csrsv_zero_pivot after the analysis of L");
        ASSERT_EQ(p.sv_analysis(p.U, rocsparse_analysis_policy_force), rocsparse_status_success);
        check(p.csrsv_pivot(p.A),
              known_defect(no_pivot,
                           r,
                           "the analysis of the unit upper triangle resets the zero pivot "
                           "that csrsv keeps for both triangles"),
              "csrsv_zero_pivot after the analysis of U");
        device_vector<double> x(b), y(a.n), z(a.n);
        ASSERT_EQ(p.sv_solve(p.U, x, y), rocsparse_status_success);
        ASSERT_EQ(p.sv_solve(p.A, y, z), rocsparse_status_success);
        check(p.csrsv_pivot(p.A),
              known_defect(no_pivot, r, "the solve of L does not record its structural zero again"),
              "csrsv_zero_pivot after the solves with U and L");
    }
}

// With a unit diagonal csrsm ignores the diagonal, missing or zero: no zero pivot, and
// the solution of the unit triangle.
TEST_F(PrecondTriangularScenarios, csrsm_unit_diagonal)
{
    const int32_t r     = 4;
    const auto    clean = make_grid_matrix<double>(3, false);
    for(const bool missing : {true, false})
    {
        const auto a = missing ? without_diagonal(clean, r) : with_zero_diagonal(clean, r);
        for(const auto base : bases)
        {
            SCOPED_TRACE(::testing::Message()
                         << (missing ? "missing" : "zero") << " diagonal, " << name(base));
            triangular_problem p(handle, a, base);
            ASSERT_NO_FATAL_FAILURE(p.init());
            ASSERT_EQ(rocsparse_set_mat_diag_type(p.A, rocsparse_diag_type_unit),
                      rocsparse_status_success);
            csrsm_result out;
            ASSERT_NO_FATAL_FAILURE(run_csrsm(p, 2, rocsparse_analysis_policy_force, out));
            ASSERT_EQ(out.analysis, rocsparse_status_success);
            ASSERT_EQ(out.solve, rocsparse_status_success);
            expect_close(out.x,
                         host_lower_solve(a, right_hand_side(a.n, 2), 2, rocsparse_diag_type_unit),
                         tolerance<double>(max_row_length(a)));
            check(p.csrsm_pivot(), is(no_pivot), "csrsm_zero_pivot");
        }
    }
}

// ---------------------------------------------------------------------------
// Pivots, unit diagonal, tolerance and boost of the factorizations
// ---------------------------------------------------------------------------

// A numerical zero pivot belongs to the values factorized: a factorization of new
// values without one reports none, and so does a new analysis, which only finds
// structural zeros.
TEST_F(PrecondTriangularScenarios, numerical_pivot_of_older_values)
{
    for(const auto r : factorizations)
    {
        // The second pivot of each singular matrix cancels exactly.
        const auto singular = (r == routine::csrilu0)
                                  ? from_dense({{2, 1, 0}, {1, 0.5, 1}, {0, 1, 2}})
                                  : from_dense({{1, 1}, {1, 1}});
        const auto regular = (r == routine::csrilu0) ? from_dense({{2, 1, 0}, {1, 2, 1}, {0, 1, 2}})
                                                     : from_dense({{2, 1}, {1, 2}});
        for(const auto base : bases)
        {
            SCOPED_TRACE(::testing::Message() << name(r) << ", " << name(base));
            triangular_problem p(handle, singular, base);
            ASSERT_NO_FATAL_FAILURE(p.init());
            ASSERT_EQ(analyse(p, r, rocsparse_analysis_policy_force), rocsparse_status_success);
            ASSERT_EQ(factorize(p, r), rocsparse_status_success);
            check(factorization_pivot(p, r), is(1), "zero pivot of the singular values");

            ASSERT_NO_FATAL_FAILURE(p.set_values(&regular.val));
            ASSERT_EQ(factorize(p, r), rocsparse_status_success);
            check(factorization_pivot(p, r),
                  is(no_pivot),
                  "zero pivot after the factorization of regular values");

            ASSERT_NO_FATAL_FAILURE(p.set_values());
            ASSERT_EQ(factorize(p, r), rocsparse_status_success);
            ASSERT_NO_FATAL_FAILURE(p.set_values(&regular.val));
            ASSERT_EQ(analyse(p, r, rocsparse_analysis_policy_force), rocsparse_status_success);
            check(factorization_pivot(p, r),
                  known_defect(1,
                               no_pivot,
                               "the analysis keeps the numerical zero pivot of the last "
                               "factorization"),
                  "zero pivot after a new analysis of regular values");
            check(factorization_singular_pivot(p, r),
                  known_defect(1,
                               no_pivot,
                               "the analysis keeps the numerical zero pivot of the last "
                               "factorization"),
                  "singular pivot after a new analysis of regular values");
        }
    }
}

// The csrilu0 and csric0 pages say a unit diagonal makes the routine "interpret the
// matrix A as having ones on its diagonal (even if no non-zero exists in the sparsity
// pattern)". Read as covering the missing entries only, a missing diagonal entry is
// taken as 1 (the factor has no entry for it): no zero pivot, and the factor of the
// matrix with that 1 added, `missing_only`. Read as covering every row, the matrices
// below become [[1, 1, 0], [1, 1, 1], [0, 1, 1]] and [[1, .], [1, 1]], whose
// factorizations have a numerical zero pivot at row 1. Today's factor matches neither
// reading; the zero pivot expectations are today's values.
TEST_F(PrecondTriangularScenarios, unit_diagonal_fills_a_missing_diagonal_of_the_factorization)
{
    for(const auto r : factorizations)
    {
        // Row 1 has no diagonal entry.
        const auto a = (r == routine::csrilu0) ? from_dense({{4, 1, 0}, {1, 0, 1}, {0, 1, 4}})
                                               : from_dense({{4, 0}, {1, 0}});
        const std::vector<double> missing_only
            = (r == routine::csrilu0) ? std::vector<double>{4, 1, 0.25, 1, 4.0 / 3, 8.0 / 3}
                                      : std::vector<double>{2, 0.5};
        for(const auto base : bases)
        {
            SCOPED_TRACE(::testing::Message() << name(r) << ", " << name(base));
            triangular_problem p(handle, a, base);
            ASSERT_NO_FATAL_FAILURE(p.init());
            ASSERT_EQ(rocsparse_set_mat_diag_type(p.A, rocsparse_diag_type_unit),
                      rocsparse_status_success);
            ASSERT_EQ(analyse(p, r, rocsparse_analysis_policy_force), rocsparse_status_success);
            check(factorization_pivot(p, r), is(no_pivot), "zero pivot after the analysis");
            ASSERT_EQ(factorize(p, r), rocsparse_status_success);
            check(factorization_pivot(p, r), is(no_pivot), "zero pivot after the factorization");
            check(all_close(p.values(), missing_only, 1e-12),
                  known_defect(false,
                               true,
                               "the factorization ignores the diagonal type: the row without a "
                               "diagonal entry is not factorized, and the rows below divide by "
                               "another of its entries"),
                  "factor with the missing diagonal taken as 1");
        }
    }
}

// With a unit diagonal the stored diagonal values are factorized as with a non unit
// diagonal. The csrilu0 and csric0 pages read as replacing them by ones too ("having
// ones on its diagonal"), which would factorize [[4, 2], [2, 4]] as [[1, 2], [2, 1]]
// (u11 = -3 for csrilu0); the pages need to say the stored diagonal is kept.
TEST_F(PrecondTriangularScenarios, unit_diagonal_keeps_the_stored_diagonal_of_the_factorization)
{
    const auto a = from_dense({{4, 2}, {2, 4}});
    for(const auto r : factorizations)
    {
        SCOPED_TRACE(name(r));
        std::vector<double> factor[2];
        for(const auto diag : {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit})
        {
            triangular_problem p(handle, a, rocsparse_index_base_zero);
            ASSERT_NO_FATAL_FAILURE(p.init());
            ASSERT_EQ(rocsparse_set_mat_diag_type(p.A, diag), rocsparse_status_success);
            ASSERT_EQ(analyse(p, r, rocsparse_analysis_policy_force), rocsparse_status_success);
            ASSERT_EQ(factorize(p, r), rocsparse_status_success);
            factor[diag == rocsparse_diag_type_unit] = p.values();
        }
        check(factor[1] == factor[0],
              is(true),
              "the factor with a unit diagonal equals the one with a non unit diagonal");
    }
}

// The tolerance of the singular pivot is 0 until it is set, is used by the
// factorization after the analysis that followed it, and is reset to 0 by the clear of
// the routine (which the documentation does not mention).
TEST_F(PrecondTriangularScenarios, tolerance_until_the_clear)
{
    for(const auto r : factorizations)
    {
        SCOPED_TRACE(name(r));
        triangular_problem p(handle, make_grid_matrix<double>(3, false), rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        check(get_tolerance(p, r), is(0.0), "tolerance of a fresh mat_info");

        // Every pivot of the grid matrix is below 10.
        ASSERT_EQ(set_tolerance(p, r, 10.0), rocsparse_status_success);
        ASSERT_EQ(analyse(p, r, rocsparse_analysis_policy_force), rocsparse_status_success);
        ASSERT_EQ(factorize(p, r), rocsparse_status_success);
        check(factorization_singular_pivot(p, r), is(0), "singular pivot with tolerance 10");

        ASSERT_EQ(clear(p, r), rocsparse_status_success);
        check(get_tolerance(p, r), is(0.0), "tolerance after the clear");
        ASSERT_NO_FATAL_FAILURE(p.set_values());
        ASSERT_EQ(analyse(p, r, rocsparse_analysis_policy_force), rocsparse_status_success);
        ASSERT_EQ(factorize(p, r), rocsparse_status_success);
        check(factorization_singular_pivot(p, r), is(no_pivot), "singular pivot after the clear");
    }
}

// A pivot with |pivot| <= boost_tol becomes |boost_val| with the sign of the pivot, and
// disabling the boost on the same mat_info leaves the next factorization alone. The sign
// is kept on purpose, to keep the inertia of A; the csrilu0_numeric_boost page, which
// gives the boosted pivot as boost_val, needs to say so.
TEST_F(PrecondTriangularScenarios, csrilu0_numeric_boost)
{
    const auto   a         = from_dense({{-1e-3, 1}, {1, 4}});
    const double boost_tol = 1e-2;
    const double boost_val = 2.0;
    for(const auto base : bases)
    {
        SCOPED_TRACE(name(base));
        triangular_problem p(handle, a, base);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_EQ(p.ilu0_analysis(rocsparse_analysis_policy_force), rocsparse_status_success);
        ASSERT_EQ(rocsparse_dcsrilu0_numeric_boost(handle, p.info, 1, &boost_tol, &boost_val),
                  rocsparse_status_success);
        ASSERT_EQ(p.ilu0(), rocsparse_status_success);
        check(p.values()[0], is(-2.0), "the boosted pivot");

        ASSERT_NO_FATAL_FAILURE(p.set_values());
        ASSERT_EQ(rocsparse_dcsrilu0_numeric_boost(handle, p.info, 0, &boost_tol, &boost_val),
                  rocsparse_status_success);
        ASSERT_EQ(p.ilu0(), rocsparse_status_success);
        check(p.values()[0], is(-1e-3), "the pivot with the boost disabled");
    }
}

// ---------------------------------------------------------------------------
// Statuses and buffer sizes
// ---------------------------------------------------------------------------

namespace
{
    constexpr rocsparse_operation conj_transpose = rocsparse_operation_conjugate_transpose;

    size_t csrsv_buffer_size(triangular_problem& p, rocsparse_operation op)
    {
        size_t size = 0;
        EXPECT_EQ(rocsparse_dcsrsv_buffer_size(
                      p.handle, op, p.a.n, p.a.nnz(), p.A, p.val, p.row, p.col, p.info, &size),
                  rocsparse_status_success);
        return size;
    }
}

// csrsv_zero_pivot does not read the matrix descriptor, and the documentation lists no
// status for a null one. hipsparseXcsrsv2_zeroPivot calls it with a null descriptor, so
// rejecting one breaks hipSPARSE: the page needs to say that the descriptor is not read.
TEST_F(PrecondTriangularScenarios, csrsv_zero_pivot_without_a_matrix_descriptor)
{
    triangular_problem p(handle, make_grid_matrix<double>(3, true), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    rocsparse_int position = -7;
    check(rocsparse_csrsv_zero_pivot(handle, nullptr, p.info, &position),
          is(rocsparse_status_success),
          "csrsv_zero_pivot with a null descr");
    EXPECT_EQ(position, -1);
}

// The csrsv, csrilu0 and csric0 buffer size pages say that the sizes are identical on
// one pattern, so that one buffer can be shared, and name no operation. That holds for
// csrsv with op none; with op transpose csrsv needs more, for the transposed matrix, so
// a buffer sized by csrilu0_buffer_size or csric0_buffer_size is too small for it (as
// in the IC(0) workflow, which solves with A^T). The pages need to restrict the
// statement to op none. The matrix is large enough for the sizes to differ: with 16 rows
// every size is the same minimum.
TEST_F(PrecondTriangularScenarios, buffer_size_of_the_solve_and_the_factorizations)
{
    triangular_problem p(handle, make_grid_matrix<double>(32, true), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    size_t ilu0 = 0, ic0 = 0;
    ASSERT_EQ(rocsparse_dcsrilu0_buffer_size(
                  handle, p.a.n, p.a.nnz(), p.A, p.val, p.row, p.col, p.info, &ilu0),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_dcsric0_buffer_size(
                  handle, p.a.n, p.a.nnz(), p.A, p.val, p.row, p.col, p.info, &ic0),
              rocsparse_status_success);
    const size_t none      = csrsv_buffer_size(p, rocsparse_operation_none);
    const size_t transpose = csrsv_buffer_size(p, rocsparse_operation_transpose);

    check(ilu0 == none,
          is(true),
          compared("csrilu0_buffer_size equals csrsv_buffer_size (op none)", ilu0, none));
    check(ic0 == none,
          is(true),
          compared("csric0_buffer_size equals csrsv_buffer_size (op none)", ic0, none));
    check(transpose > none,
          is(true),
          compared("csrsv_buffer_size is larger for op transpose", transpose, none));
}

// The csrsm buffer is larger for a transposed A, and for a B that is not transposed (on
// a matrix large enough for the sizes to differ).
TEST_F(PrecondTriangularScenarios, csrsm_buffer_size_by_operation)
{
    const int32_t      nrhs = 4;
    triangular_problem p(handle, make_grid_matrix<double>(32, true), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    device_vector<double> B(right_hand_side(p.a.n, nrhs));
    const auto            size = [&](rocsparse_operation trans_A, rocsparse_operation trans_B) {
        size_t s = 0;
        EXPECT_EQ(csrsm_buffer_size(p,
                                    trans_A,
                                    trans_B,
                                    nrhs,
                                    B,
                                    (trans_B == rocsparse_operation_none) ? p.a.n : nrhs,
                                    &s),
                  rocsparse_status_success);
        return s;
    };
    const auto N = rocsparse_operation_none;
    const auto T = rocsparse_operation_transpose;

    check(size(T, N) > size(N, N),
          is(true),
          compared("csrsm_buffer_size: trans_A T against N", size(T, N), size(N, N)));
    check(size(N, N) > size(N, T),
          is(true),
          compared("csrsm_buffer_size: trans_B N against T", size(N, N), size(N, T)));
}

// With m == 0 the buffer size, the analysis and the solve of csrsv return success
// before checking the handle.
TEST_F(PrecondTriangularScenarios, csrsv_empty_matrix_without_a_handle)
{
    triangular_problem p(handle, make_grid_matrix<double>(3, true), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    const auto expected = known_defect(rocsparse_status_success,
                                       rocsparse_status_invalid_handle,
                                       "m == 0 returns before the handle is checked");
    const auto none     = rocsparse_operation_none;
    const auto solve    = rocsparse_solve_policy_auto;

    size_t size = 0;
    check(
        rocsparse_dcsrsv_buffer_size(nullptr, none, 0, 0, p.A, p.val, p.row, p.col, p.info, &size),
        expected,
        "csrsv_buffer_size with a null handle");
    check(rocsparse_dcsrsv_analysis(nullptr,
                                    none,
                                    0,
                                    0,
                                    p.A,
                                    p.val,
                                    p.row,
                                    p.col,
                                    p.info,
                                    rocsparse_analysis_policy_force,
                                    solve,
                                    p.buffer),
          expected,
          "csrsv_analysis with a null handle");
    const double          one = 1.0;
    device_vector<double> x(right_hand_side(p.a.n)), y(p.a.n);
    check(rocsparse_dcsrsv_solve(
              nullptr, none, 0, 0, &one, p.A, p.val, p.row, p.col, p.info, x, y, solve, p.buffer),
          expected,
          "csrsv_solve with a null handle");
}

// The csrsv page contradicts itself: it defines op(A) = A^H for conjugate_transpose and
// lists not_implemented for it. csrsv implements it and solves with A^H (with double
// values, the solve with the transpose); the page needs reconciling with the routine.
TEST_F(PrecondTriangularScenarios, csrsv_conjugate_transpose)
{
    triangular_problem p(handle, make_grid_matrix<double>(3, true), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    const auto expected = is(rocsparse_status_success);
    size_t     size     = 0;
    check(rocsparse_dcsrsv_buffer_size(
              handle, conj_transpose, p.a.n, p.a.nnz(), p.A, p.val, p.row, p.col, p.info, &size),
          expected,
          "csrsv_buffer_size");
    ASSERT_NO_FATAL_FAILURE(p.reserve(size));
    check(p.sv_analysis(p.A, rocsparse_analysis_policy_force, conj_transpose),
          expected,
          "csrsv_analysis");
    const auto            b = right_hand_side(p.a.n);
    device_vector<double> x(b), y(sentinel(p.a.n));
    check(p.sv_solve(p.A, x, y, conj_transpose), expected, "csrsv_solve");
    expect_close(
        to_host(y), host_lower_transposed_solve(p.a, b, 1), tolerance<double>(max_row_length(p.a)));
}

// The same for trans_A and for trans_B of csrsm: its page defines A^H, B^H and X^H and
// lists not_implemented for them, and csrsm solves with them.
TEST_F(PrecondTriangularScenarios, csrsm_conjugate_transpose)
{
    const int32_t nrhs     = 2;
    const auto    expected = is(rocsparse_status_success);
    for(const bool on_A : {true, false})
    {
        SCOPED_TRACE(on_A ? "trans_A" : "trans_B");
        const auto         trans_A = on_A ? conj_transpose : rocsparse_operation_none;
        const auto         trans_B = on_A ? rocsparse_operation_none : conj_transpose;
        triangular_problem p(handle, make_grid_matrix<double>(3, true), rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        const int32_t         ldb = on_A ? p.a.n : nrhs;
        const auto            b   = right_hand_side(p.a.n, nrhs);
        device_vector<double> B(b);
        size_t                size = 0;
        check(csrsm_buffer_size(p, trans_A, trans_B, nrhs, B, ldb, &size),
              expected,
              "csrsm_buffer_size");
        ASSERT_NO_FATAL_FAILURE(p.reserve(size));
        check(csrsm_analysis(p, trans_A, trans_B, nrhs, B, ldb, rocsparse_analysis_policy_force),
              expected,
              "csrsm_analysis");
        check(csrsm_solve(p, trans_A, trans_B, nrhs, B, ldb), expected, "csrsm_solve");
        // With trans_B, B and X are nrhs x m: op(X) = A^-1 op(B) is their transpose.
        const auto want = on_A ? host_lower_transposed_solve(p.a, b, nrhs)
                               : transposed(host_lower_solve(p.a, transposed(b, nrhs, p.a.n), nrhs),
                                            p.a.n,
                                            nrhs);
        expect_close(to_host(B), want, tolerance<double>(max_row_length(p.a)));
    }
}

// The csrsv page rejects a matrix type other than general with not_implemented; csrsv
// accepts the triangular type on purpose (its checks name it next to general) and
// rejects symmetric and hermitian. The page needs to list the triangular type.
TEST_F(PrecondTriangularScenarios, csrsv_triangular_matrix_type)
{
    triangular_problem p(handle, make_grid_matrix<double>(3, true), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    rocsparse_mat_descr T = make_descr(
        rocsparse_index_base_zero, rocsparse_fill_mode_lower, rocsparse_diag_type_non_unit);
    ASSERT_EQ(rocsparse_set_mat_type(T, rocsparse_matrix_type_triangular),
              rocsparse_status_success);
    const auto expected = is(rocsparse_status_success);
    size_t     size     = 0;
    check(rocsparse_dcsrsv_buffer_size(handle,
                                       rocsparse_operation_none,
                                       p.a.n,
                                       p.a.nnz(),
                                       T,
                                       p.val,
                                       p.row,
                                       p.col,
                                       p.info,
                                       &size),
          expected,
          "csrsv_buffer_size");
    ASSERT_NO_FATAL_FAILURE(p.reserve(size));
    check(p.sv_analysis(T, rocsparse_analysis_policy_force), expected, "csrsv_analysis");
    const auto            b = right_hand_side(p.a.n);
    device_vector<double> x(b), y(sentinel(p.a.n));
    check(p.sv_solve(T, x, y), expected, "csrsv_solve");
    expect_close(to_host(y), host_lower_solve(p.a, b, 1), tolerance<double>(max_row_length(p.a)));
    (void)rocsparse_destroy_mat_descr(T);
}

// ldb below m (trans_B none) or below nrhs (trans_B transpose) is invalid_size.
TEST_F(PrecondTriangularScenarios, csrsm_leading_dimension_too_small)
{
    const int32_t nrhs = 2;
    for(const auto trans_B : {rocsparse_operation_none, rocsparse_operation_transpose})
    {
        SCOPED_TRACE(trans_B == rocsparse_operation_none ? "trans_B none" : "trans_B transpose");
        const auto         N = rocsparse_operation_none;
        triangular_problem p(handle, make_grid_matrix<double>(3, true), rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        const int32_t         ldb = (trans_B == N) ? p.a.n : nrhs;
        device_vector<double> B(right_hand_side(p.a.n, nrhs));
        size_t                size = 0;
        check(csrsm_buffer_size(p, N, trans_B, nrhs, B, ldb - 1, &size),
              is(rocsparse_status_invalid_size),
              "csrsm_buffer_size");
        ASSERT_EQ(csrsm_buffer_size(p, N, trans_B, nrhs, B, ldb, &size), rocsparse_status_success);
        ASSERT_NO_FATAL_FAILURE(p.reserve(size));
        const auto force = rocsparse_analysis_policy_force;
        check(csrsm_analysis(p, N, trans_B, nrhs, B, ldb - 1, force),
              is(rocsparse_status_invalid_size),
              "csrsm_analysis");
        ASSERT_EQ(csrsm_analysis(p, N, trans_B, nrhs, B, ldb, force), rocsparse_status_success);
        check(csrsm_solve(p, N, trans_B, nrhs, B, ldb - 1),
              is(rocsparse_status_invalid_size),
              "csrsm_solve");
    }
}

// ---------------------------------------------------------------------------
// Sequences that crash or use freed memory
// ---------------------------------------------------------------------------

// csrilu0_analysis reuses the analysis of L made by csrsv; csrilu0 then reads the zero
// pivot of csrilu0, which the reused analysis never allocated.
TEST_F(PrecondTriangularScenarios, DISABLED_csrilu0_reusing_a_csrsv_analysis)
{
    const auto a = make_grid_matrix<double>(4, true);
    for(const auto base : bases)
    {
        SCOPED_TRACE(name(base));
        triangular_problem p(handle, a, base);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_EQ(p.sv_analysis(p.L, rocsparse_analysis_policy_force), rocsparse_status_success);
        ASSERT_EQ(p.ilu0_analysis(rocsparse_analysis_policy_reuse), rocsparse_status_success);
        ASSERT_EQ(p.ilu0(), rocsparse_status_success);
        expect_close(p.values(), host_ilu0(a), tolerance<double>(max_row_length(a)));
        check(p.csrilu0_pivot(), is(no_pivot), "csrilu0_zero_pivot");
    }
}

// The analysis of L reuses the one of csrilu0; the solve with L then resets the zero
// pivot of csrsv, which the reused analysis never allocated.
TEST_F(PrecondTriangularScenarios, DISABLED_csrsv_solve_after_a_reused_analysis)
{
    const auto a = make_grid_matrix<double>(4, true);
    const auto b = right_hand_side(a.n);
    for(const auto base : bases)
    {
        SCOPED_TRACE(name(base));
        triangular_problem p(handle, a, base);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_EQ(p.ilu0_analysis(rocsparse_analysis_policy_force), rocsparse_status_success);
        ASSERT_EQ(p.ilu0(), rocsparse_status_success);
        ASSERT_EQ(p.sv_analysis(p.L, rocsparse_analysis_policy_reuse), rocsparse_status_success);
        device_vector<double> x(b), y(sentinel(a.n));
        ASSERT_EQ(p.sv_solve(p.L, x, y), rocsparse_status_success);
        expect_close(to_host(y),
                     host_lower_solve(with_values(a, p.values()), b, 1, rocsparse_diag_type_unit),
                     tolerance<double>(max_row_length(a)));
        check(p.csrsv_pivot(p.L), is(no_pivot), "csrsv_zero_pivot after the solve with L");
    }
}

#ifdef ROCSPARSE_WITH_TRSM_REFACTORING
// csrsm with two right-hand sides reuses the analysis of csrsv; the solve then reads the
// zero pivot of csrsm, which the reused analysis never allocated.
TEST_F(PrecondTriangularScenarios, DISABLED_csrsm_reusing_a_csrsv_analysis)
{
    check_csrsm_after(
        handle, routine::csrsv, 2, rocsparse_analysis_policy_reuse, is(rocsparse_status_success));
}

// The same with the analysis of csrilu0 or csric0.
TEST_F(PrecondTriangularScenarios, DISABLED_csrsm_reusing_a_csrilu0_analysis)
{
    ASSERT_NO_FATAL_FAILURE(check_reuse(handle, routine::csrilu0, routine::csrsm));
}

TEST_F(PrecondTriangularScenarios, DISABLED_csrsm_reusing_a_csric0_analysis)
{
    ASSERT_NO_FATAL_FAILURE(check_reuse(handle, routine::csric0, routine::csrsm));
}
#endif

// Each reuse hit below copies the analysis of another routine and returns before
// allocating the zero pivot of the routine that reuses it, which then faults on it.
TEST_F(PrecondTriangularScenarios, DISABLED_csrilu0_reusing_a_csrsm_analysis)
{
    ASSERT_NO_FATAL_FAILURE(check_reuse(handle, routine::csrsm, routine::csrilu0));
}

TEST_F(PrecondTriangularScenarios, DISABLED_csrilu0_reusing_a_csric0_analysis)
{
    ASSERT_NO_FATAL_FAILURE(check_reuse(handle, routine::csric0, routine::csrilu0));
}

TEST_F(PrecondTriangularScenarios, DISABLED_csric0_reusing_a_csrsv_analysis)
{
    ASSERT_NO_FATAL_FAILURE(check_reuse(handle, routine::csrsv, routine::csric0));
}

TEST_F(PrecondTriangularScenarios, DISABLED_csric0_reusing_a_csrsm_analysis)
{
    ASSERT_NO_FATAL_FAILURE(check_reuse(handle, routine::csrsm, routine::csric0));
}

TEST_F(PrecondTriangularScenarios, DISABLED_csric0_reusing_a_csrilu0_analysis)
{
    ASSERT_NO_FATAL_FAILURE(check_reuse(handle, routine::csrilu0, routine::csric0));
}

TEST_F(PrecondTriangularScenarios, DISABLED_csrsv_reusing_a_csric0_analysis)
{
    ASSERT_NO_FATAL_FAILURE(check_reuse(handle, routine::csric0, routine::csrsv));
}

TEST_F(PrecondTriangularScenarios, DISABLED_csrsv_reusing_a_csrsm_analysis)
{
    ASSERT_NO_FATAL_FAILURE(check_reuse(handle, routine::csrsm, routine::csrsv));
}

// After csrilu0_clear the reuse policy takes the analysis that csrsv kept, again
// without the zero pivot of csrilu0.
TEST_F(PrecondTriangularScenarios, DISABLED_csrilu0_reused_analysis_after_csrilu0_clear)
{
    const auto         a = make_grid_matrix<double>(4, true);
    triangular_problem p(handle, a, rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    ASSERT_NO_FATAL_FAILURE(p.ilu0_workflow(order::lower_first, rocsparse_analysis_policy_reuse));
    ASSERT_EQ(rocsparse_csrilu0_clear(handle, p.info), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(p.set_values());
    ASSERT_EQ(p.ilu0_analysis(rocsparse_analysis_policy_reuse), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(check_compute(p, routine::csrilu0));
}

// With no analysis of (none, lower), the reuse policy of csrilu0 falls back to the
// transposed analysis of the upper triangle, which only matches ILU(0) on a structurally
// symmetric pattern; on this one the factor differs (once the missing zero pivot of the
// reuse hit no longer faults first).
TEST_F(PrecondTriangularScenarios, DISABLED_csrilu0_reusing_a_transposed_analysis)
{
    const auto         a = without_entry(make_grid_matrix<double>(4, true), 5, 4);
    triangular_problem p(handle, a, rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    ASSERT_EQ(p.sv_analysis(p.U, rocsparse_analysis_policy_force, rocsparse_operation_transpose),
              rocsparse_status_success);
    ASSERT_EQ(p.ilu0_analysis(rocsparse_analysis_policy_reuse), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(check_compute(p, routine::csrilu0));
}

// The documented call sequence starts with the analysis, and the pages list no status
// for a call without one; these calls crash on the host. The expectation is
// rocsparse_status_invalid_pointer, what csric0 (csric0_needs_its_analysis) and the BSR
// routines return without an analysis.
TEST_F(PrecondTriangularScenarios, DISABLED_csrilu0_without_an_analysis)
{
    triangular_problem p(handle, make_grid_matrix<double>(4, true), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    check(p.ilu0(), is(rocsparse_status_invalid_pointer), "csrilu0 without an analysis");
}

TEST_F(PrecondTriangularScenarios, DISABLED_csrsm_without_an_analysis)
{
    triangular_problem p(handle, make_grid_matrix<double>(4, true), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    device_vector<double> dB(right_hand_side(p.a.n, 2));
    ASSERT_NO_FATAL_FAILURE(reserve_csrsm(p, 2, dB));
    check(csrsm_solve(p, 2, dB),
          is(rocsparse_status_invalid_pointer),
          "csrsm_solve with two right-hand sides without an analysis");
}

TEST_F(PrecondTriangularScenarios, DISABLED_csrsm_after_csrsm_clear)
{
    triangular_problem p(handle, make_grid_matrix<double>(4, true), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    csrsm_result out;
    ASSERT_NO_FATAL_FAILURE(run_csrsm(p, 2, rocsparse_analysis_policy_force, out));
    ASSERT_EQ(out.solve, rocsparse_status_success);
    ASSERT_EQ(rocsparse_csrsm_clear(handle, p.info), rocsparse_status_success);
    device_vector<double> dB(right_hand_side(p.a.n, 2));
    check(csrsm_solve(p, 2, dB),
          is(rocsparse_status_invalid_pointer),
          "csrsm_solve with two right-hand sides after csrsm_clear");
}

// After csrilu0_clear, csrilu0 dereferences the analysis it no longer has (csric0
// returns rocsparse_status_invalid_pointer in that case).
TEST_F(PrecondTriangularScenarios, DISABLED_csrilu0_after_csrilu0_clear)
{
    const auto         a = make_grid_matrix<double>(4, true);
    triangular_problem p(handle, a, rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    ASSERT_NO_FATAL_FAILURE(p.ilu0_workflow(order::lower_first, rocsparse_analysis_policy_force));
    ASSERT_EQ(rocsparse_csrilu0_clear(handle, p.info), rocsparse_status_success);
    check(p.ilu0(), is(rocsparse_status_invalid_pointer), "csrilu0 after csrilu0_clear");
}

// csrsv reuses the analysis of L made by csrilu0, then a forced csrsv analysis of L
// deletes it while csrilu0 still uses it.
TEST_F(PrecondTriangularScenarios, DISABLED_forced_csrsv_analysis_after_reusing_csrilu0)
{
    triangular_problem p(handle, make_grid_matrix<double>(4, true), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    ASSERT_EQ(p.ilu0_analysis(rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_EQ(p.sv_analysis(p.L, rocsparse_analysis_policy_reuse), rocsparse_status_success);
    ASSERT_EQ(p.sv_analysis(p.U, rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_EQ(p.sv_analysis(p.L, rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(check_ilu0_workflow_after_forced_analysis(p));
}

// csrilu0 reuses the analysis of L made by csrsv, then a forced csrilu0 analysis deletes
// it while csrsv still uses it.
TEST_F(PrecondTriangularScenarios, DISABLED_forced_csrilu0_analysis_after_reusing_csrsv)
{
    triangular_problem p(handle, make_grid_matrix<double>(4, true), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    ASSERT_EQ(p.sv_analysis(p.L, rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_EQ(p.sv_analysis(p.U, rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_EQ(p.ilu0_analysis(rocsparse_analysis_policy_reuse), rocsparse_status_success);
    ASSERT_EQ(p.ilu0_analysis(rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(check_ilu0_workflow_after_forced_analysis(p));
}

// The same with csric0: csrsv reuses the analysis of A made by csric0, then a forced
// csrsv analysis of A deletes it while csric0 still uses it.
TEST_F(PrecondTriangularScenarios, DISABLED_forced_csrsv_analysis_after_reusing_csric0)
{
    const auto         a = make_grid_matrix<double>(4, false);
    triangular_problem p(handle, a, rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    ASSERT_EQ(p.ic0_analysis(rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_EQ(p.sv_analysis(p.A, rocsparse_analysis_policy_reuse), rocsparse_status_success);
    ASSERT_EQ(p.sv_analysis(p.A, rocsparse_analysis_policy_force, rocsparse_operation_transpose),
              rocsparse_status_success);
    ASSERT_EQ(p.sv_analysis(p.A, rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(check_compute(p, routine::csric0));
}

// csric0 reuses the analysis of A made by csrsv, then a forced csric0 analysis deletes
// it while csrsv still uses it.
TEST_F(PrecondTriangularScenarios, DISABLED_forced_csric0_analysis_after_reusing_csrsv)
{
    const auto         a = make_grid_matrix<double>(4, false);
    triangular_problem p(handle, a, rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    ASSERT_EQ(p.sv_analysis(p.A, rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_EQ(p.sv_analysis(p.A, rocsparse_analysis_policy_force, rocsparse_operation_transpose),
              rocsparse_status_success);
    ASSERT_EQ(p.ic0_analysis(rocsparse_analysis_policy_reuse), rocsparse_status_success);
    ASSERT_EQ(p.ic0_analysis(rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_EQ(p.ic0(), rocsparse_status_success);
    const auto l = p.values();
    expect_close(keep_lower_only(a, l),
                 keep_lower_only(a, host_ic0(a)),
                 tolerance<double>(max_row_length(a)));
    expect_close(p.llt_solve(right_hand_side(a.n)),
                 host_llt_solve(a, l, right_hand_side(a.n)),
                 tolerance<double>(max_row_length(a)));
}

// csrsv reuses the analysis of A made by csrsm, then a forced csrsm analysis deletes it
// while csrsv still uses it. The analysis of U first gives csrsv its zero pivot.
TEST_F(PrecondTriangularScenarios, DISABLED_forced_csrsm_analysis_after_csrsv_reused_it)
{
    triangular_problem p(handle, make_grid_matrix<double>(4, true), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    ASSERT_EQ(p.sv_analysis(p.U, rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_EQ(csrsm_analysis(p, 2, rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_EQ(p.sv_analysis(p.A, rocsparse_analysis_policy_reuse), rocsparse_status_success);
    ASSERT_EQ(csrsm_analysis(p, 2, rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(check_compute(p, routine::csrsv));
    ASSERT_NO_FATAL_FAILURE(check_compute(p, routine::csrsm));
}

// The copy reads the tolerance of csrilu0 through a pointer into the original, which
// csrilu0 on the copy dereferences once the original is destroyed.
TEST_F(PrecondTriangularScenarios, DISABLED_csrilu0_on_a_copy_after_destroying_the_original)
{
    const auto         a = make_grid_matrix<double>(4, true);
    triangular_problem p(handle, a, rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    ASSERT_NO_FATAL_FAILURE(p.ilu0_workflow(order::lower_first, rocsparse_analysis_policy_force));
    const auto lu = p.values();
    ASSERT_EQ(p.replace_info_by_copy(false), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(p.set_values());
    ASSERT_EQ(p.ilu0(), rocsparse_status_success);
    expect_close(p.values(), lu, tolerance<double>(max_row_length(a)));
    check(p.csrilu0_singular_pivot(), is(no_pivot), "csrilu0_singular_pivot of the copy");
}

// With m == 0 csrsv_buffer_size writes the size before checking the pointer, and a null
// buffer_size crashes on the host. csrsm does the same with m == 0 or nrhs == 0.
TEST_F(PrecondTriangularScenarios, DISABLED_csrsv_buffer_size_into_a_null_pointer_without_rows)
{
    triangular_problem p(handle, make_grid_matrix<double>(3, true), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    check(rocsparse_dcsrsv_buffer_size(
              handle, rocsparse_operation_none, 0, 0, p.A, p.val, p.row, p.col, p.info, nullptr),
          is(rocsparse_status_invalid_pointer),
          "csrsv_buffer_size with m == 0 and a null buffer_size");
}

TEST_F(PrecondTriangularScenarios, DISABLED_csrsm_buffer_size_into_a_null_pointer_without_rows)
{
    triangular_problem p(handle, make_grid_matrix<double>(3, true), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    device_vector<double> B(right_hand_side(p.a.n, 2));
    const double          alpha = 1.0;
    check(rocsparse_dcsrsm_buffer_size(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       0,
                                       2,
                                       0,
                                       &alpha,
                                       p.A,
                                       p.val,
                                       p.row,
                                       p.col,
                                       B,
                                       1,
                                       p.info,
                                       rocsparse_solve_policy_auto,
                                       nullptr),
          is(rocsparse_status_invalid_pointer),
          "csrsm_buffer_size with m == 0 and a null buffer_size");
}

TEST_F(PrecondTriangularScenarios,
       DISABLED_csrsm_buffer_size_into_a_null_pointer_without_right_hand_sides)
{
    triangular_problem p(handle, make_grid_matrix<double>(3, true), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    device_vector<double> B(right_hand_side(p.a.n, 1));
    check(csrsm_buffer_size(
              p, rocsparse_operation_none, rocsparse_operation_none, 0, B, p.a.n, nullptr),
          is(rocsparse_status_invalid_pointer),
          "csrsm_buffer_size with nrhs == 0 and a null buffer_size");
}
