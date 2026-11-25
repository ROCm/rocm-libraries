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

// Scenarios of the triangular analysis data that bsrilu0, bsric0, bsrsv and bsrsm share
// on one mat_info: the block ILU(0) and IC(0) workflows, each routine reusing the
// analysis of another, forced analyses of shared data, calls without an analysis, clear,
// a copy that outlives the original, and a new pattern on an analysed mat_info. Only the
// public API is used; the results are compared with the scalar host references on the
// expanded matrix, since the blocks are dense. The conventions are those of
// unit_test_device_precond_triangular_scenarios.cpp: expectations pin what the library
// returns today, known_defect marks where the library is wrong and its page right, a
// comment names the page where the page is what needs fixing, every byte of the buffer
// is set to 0xff before every call, and sequences that crash or use freed memory are
// DISABLED_ tests asserting the documented behaviour (run one at a time under a timeout,
// those using freed memory under AddressSanitizer).

#include "unit_test_device_precond_triangular_common.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

using namespace triangular_scenarios;

namespace
{
    using triangular_scenarios::name;

    const char* name(rocsparse_direction dir)
    {
        return (dir == rocsparse_direction_row) ? "row major blocks" : "column major blocks";
    }

    constexpr rocsparse_direction directions[]
        = {rocsparse_direction_row, rocsparse_direction_column};

    // The matrix without its block (row, col).
    bsr_matrix<double> without_block(bsr_matrix<double> m, int32_t row, int32_t col)
    {
        const size_t size = static_cast<size_t>(m.block_dim) * m.block_dim;
        for(int32_t k = m.ptr[row]; k < m.ptr[row + 1]; ++k)
        {
            if(m.ind[k] == col)
            {
                m.ind.erase(m.ind.begin() + k);
                m.val.erase(m.val.begin() + k * size, m.val.begin() + (k + 1) * size);
                for(int32_t r = row + 1; r <= m.mb; ++r)
                    --m.ptr[r];
                break;
            }
        }
        return m;
    }

    // The matrix with the scalar diagonal entry `i` of the diagonal block of `row` set
    // to zero.
    bsr_matrix<double> with_zero_in_diagonal_block(bsr_matrix<double> m, int32_t row, int32_t i)
    {
        for(int32_t k = m.ptr[row]; k < m.ptr[row + 1]; ++k)
            if(m.ind[k] == row)
                m.val[static_cast<size_t>(k) * m.block_dim * m.block_dim
                      + static_cast<size_t>(i) * m.block_dim + i]
                    = 0.0;
        return m;
    }

    // A block matrix on the device (indices in `base`) with the descriptors of the block
    // ILU(0) and IC(0) workflows on one mat_info: A (general, lower, non unit) for
    // bsrilu0, bsric0 and the non unit lower solves of bsrsv and bsrsm (with A^T for the
    // second solve of IC(0)), L (lower, unit) and U (upper, non unit). bsrsm solves for
    // `nrhs` column major right-hand sides. Zero pivots are block rows.
    struct block_problem
    {
        static constexpr int32_t nrhs = 2;

        rocsparse_handle             handle;
        bsr_matrix<double>           m;
        rocsparse_index_base         base;
        csr_matrix<double>           scalar;
        device_vector<rocsparse_int> row, col;
        device_vector<double>        val;
        device_vector<char>          buffer;
        rocsparse_mat_descr          A    = nullptr;
        rocsparse_mat_descr          L    = nullptr;
        rocsparse_mat_descr          U    = nullptr;
        rocsparse_mat_info           info = nullptr;
        bool                         device_query;

        // The zero pivots are queried in device memory when `device` is set.
        block_problem(rocsparse_handle          h,
                      const bsr_matrix<double>& b,
                      rocsparse_index_base      ib,
                      bool                      device = false)
            : handle(h)
            , m(b)
            , base(ib)
            , device_query(device)
        {
        }

        block_problem(const block_problem&) = delete;
        block_problem& operator=(const block_problem&) = delete;

        ~block_problem()
        {
            if(info)
                (void)rocsparse_destroy_mat_info(info);
            for(auto descr : {A, L, U})
                if(descr)
                    (void)rocsparse_destroy_mat_descr(descr);
        }

        int32_t n() const
        {
            return m.mb * m.block_dim;
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
        void replace_matrix(const bsr_matrix<double>& b)
        {
            m = b;
            ASSERT_NO_FATAL_FAILURE(upload_matrix());
        }

        void upload_matrix()
        {
            scalar                           = expand_bsr(m);
            const rocsparse_int        shift = (base == rocsparse_index_base_one) ? 1 : 0;
            std::vector<rocsparse_int> hptr  = as_rocsparse_int(m.ptr);
            std::vector<rocsparse_int> hind  = as_rocsparse_int(m.ind);
            for(auto& v : hptr)
                v += shift;
            for(auto& v : hind)
                v += shift;
            upload(row, hptr);
            upload(col, hind);
            upload(val, m.val);
            ASSERT_TRUE(row.ptr && col.ptr && val.ptr);

            const int32_t mb   = m.mb;
            const int32_t nnzb = m.nnzb();
            const int32_t bd   = m.block_dim;
            size_t        size = 0;
            ASSERT_EQ(rocsparse_dbsrilu0_buffer_size(
                          handle, m.dir, mb, nnzb, A, val, row, col, bd, info, &size),
                      rocsparse_status_success);
            ASSERT_NO_FATAL_FAILURE(reserve(size));
            ASSERT_EQ(rocsparse_dbsric0_buffer_size(
                          handle, m.dir, mb, nnzb, A, val, row, col, bd, info, &size),
                      rocsparse_status_success);
            ASSERT_NO_FATAL_FAILURE(reserve(size));
            for(auto descr : {A, L, U})
            {
                for(auto op : {rocsparse_operation_none, rocsparse_operation_transpose})
                {
                    ASSERT_EQ(
                        rocsparse_dbsrsv_buffer_size(
                            handle, m.dir, op, mb, nnzb, descr, val, row, col, bd, info, &size),
                        rocsparse_status_success);
                    ASSERT_NO_FATAL_FAILURE(reserve(size));
                    ASSERT_EQ(rocsparse_dbsrsm_buffer_size(handle,
                                                           m.dir,
                                                           op,
                                                           rocsparse_operation_none,
                                                           mb,
                                                           nrhs,
                                                           nnzb,
                                                           descr,
                                                           val,
                                                           row,
                                                           col,
                                                           bd,
                                                           info,
                                                           &size),
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

        // Replaces the values on the device (bsrilu0 and bsric0 factor them in place);
        // `values` defaults to those of `m`.
        void set_values(const std::vector<double>* values = nullptr)
        {
            const std::vector<double>& v = values ? *values : m.val;
            ASSERT_EQ(v.size(), m.val.size());
            ASSERT_EQ(
                hipMemcpy(val.ptr, v.data(), sizeof(double) * v.size(), hipMemcpyHostToDevice),
                hipSuccess);
        }

        std::vector<double> values() const
        {
            return to_host(val);
        }

        // The values on the device, on the pattern of `scalar`.
        std::vector<double> scalar_values() const
        {
            bsr_matrix<double> f = m;
            f.val                = values();
            return expand_bsr(f).val;
        }

        rocsparse_status ilu0_analysis(rocsparse_analysis_policy policy)
        {
            poison();
            return rocsparse_dbsrilu0_analysis(handle,
                                               m.dir,
                                               m.mb,
                                               m.nnzb(),
                                               A,
                                               val,
                                               row,
                                               col,
                                               m.block_dim,
                                               info,
                                               policy,
                                               rocsparse_solve_policy_auto,
                                               buffer);
        }

        rocsparse_status ilu0()
        {
            poison();
            return rocsparse_dbsrilu0(handle,
                                      m.dir,
                                      m.mb,
                                      m.nnzb(),
                                      A,
                                      val,
                                      row,
                                      col,
                                      m.block_dim,
                                      info,
                                      rocsparse_solve_policy_auto,
                                      buffer);
        }

        rocsparse_status ic0_analysis(rocsparse_analysis_policy policy)
        {
            poison();
            return rocsparse_dbsric0_analysis(handle,
                                              m.dir,
                                              m.mb,
                                              m.nnzb(),
                                              A,
                                              val,
                                              row,
                                              col,
                                              m.block_dim,
                                              info,
                                              policy,
                                              rocsparse_solve_policy_auto,
                                              buffer);
        }

        rocsparse_status ic0()
        {
            poison();
            return rocsparse_dbsric0(handle,
                                     m.dir,
                                     m.mb,
                                     m.nnzb(),
                                     A,
                                     val,
                                     row,
                                     col,
                                     m.block_dim,
                                     info,
                                     rocsparse_solve_policy_auto,
                                     buffer);
        }

        rocsparse_status sv_analysis(rocsparse_mat_descr       descr,
                                     rocsparse_analysis_policy policy,
                                     rocsparse_operation       op = rocsparse_operation_none)
        {
            poison();
            return rocsparse_dbsrsv_analysis(handle,
                                             m.dir,
                                             op,
                                             m.mb,
                                             m.nnzb(),
                                             descr,
                                             val,
                                             row,
                                             col,
                                             m.block_dim,
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
            return rocsparse_dbsrsv_solve(handle,
                                          m.dir,
                                          op,
                                          m.mb,
                                          m.nnzb(),
                                          &one,
                                          descr,
                                          val,
                                          row,
                                          col,
                                          m.block_dim,
                                          info,
                                          x,
                                          y,
                                          rocsparse_solve_policy_auto,
                                          buffer);
        }

        rocsparse_status sm_analysis(rocsparse_mat_descr descr, rocsparse_analysis_policy policy)
        {
            poison();
            return rocsparse_dbsrsm_analysis(handle,
                                             m.dir,
                                             rocsparse_operation_none,
                                             rocsparse_operation_none,
                                             m.mb,
                                             nrhs,
                                             m.nnzb(),
                                             descr,
                                             val,
                                             row,
                                             col,
                                             m.block_dim,
                                             info,
                                             policy,
                                             rocsparse_solve_policy_auto,
                                             buffer);
        }

        rocsparse_status sm_solve(rocsparse_mat_descr descr, const double* B, double* X)
        {
            poison();
            const double one = 1.0;
            return rocsparse_dbsrsm_solve(handle,
                                          m.dir,
                                          rocsparse_operation_none,
                                          rocsparse_operation_none,
                                          m.mb,
                                          nrhs,
                                          m.nnzb(),
                                          &one,
                                          descr,
                                          val,
                                          row,
                                          col,
                                          m.block_dim,
                                          info,
                                          B,
                                          n(),
                                          X,
                                          n(),
                                          rocsparse_solve_policy_auto,
                                          buffer);
        }

        // bsrilu0_analysis, the analyses of L and U in the order `o`, then bsrilu0.
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

        // bsric0_analysis, the analyses of A and A^T in the order `o` (lower_first
        // analyses A first), then bsric0.
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

        // Replaces the mat_info by a copy of it and destroys the original.
        rocsparse_status replace_info_by_copy()
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
            status = rocsparse_destroy_mat_info(info);
            info   = copy;
            return status;
        }

        int32_t bsrsv_pivot()
        {
            return zero_pivot(
                [&](rocsparse_int* p) { return rocsparse_bsrsv_zero_pivot(handle, info, p); });
        }

        int32_t bsrsm_pivot()
        {
            return zero_pivot(
                [&](rocsparse_int* p) { return rocsparse_bsrsm_zero_pivot(handle, info, p); });
        }

        int32_t bsrilu0_pivot()
        {
            return zero_pivot(
                [&](rocsparse_int* p) { return rocsparse_bsrilu0_zero_pivot(handle, info, p); });
        }

        int32_t bsric0_pivot()
        {
            return zero_pivot(
                [&](rocsparse_int* p) { return rocsparse_bsric0_zero_pivot(handle, info, p); });
        }

        template <typename F>
        int32_t zero_pivot(F&& call)
        {
            return zero_pivot_row(handle, base, m.mb, device_query, call);
        }
    };

    // The routines that analyse the lower triangle of A (op none) on the mat_info.
    enum class routine
    {
        bsrsv,
        bsrsm,
        bsrilu0,
        bsric0
    };

    const char* name(routine r)
    {
        switch(r)
        {
        case routine::bsrsv:
            return "bsrsv";
        case routine::bsrsm:
            return "bsrsm";
        case routine::bsrilu0:
            return "bsrilu0";
        case routine::bsric0:
            return "bsric0";
        }
        return "?";
    }

    rocsparse_status analyse(block_problem& p, routine r, rocsparse_analysis_policy policy)
    {
        switch(r)
        {
        case routine::bsrsv:
            return p.sv_analysis(p.A, policy);
        case routine::bsrsm:
            return p.sm_analysis(p.A, policy);
        case routine::bsrilu0:
            return p.ilu0_analysis(policy);
        case routine::bsric0:
            return p.ic0_analysis(policy);
        }
        return rocsparse_status_internal_error;
    }

    // Runs `r` after its analysis (the factorization, or the solve with the lower
    // triangle of A) and compares the result and its zero pivot with the host.
    void check_compute(block_problem& p, routine r)
    {
        const auto tol = tolerance<double>(max_row_length(p.scalar));
        switch(r)
        {
        case routine::bsrsv:
        {
            const auto            b = right_hand_side(p.n());
            device_vector<double> x(b), y(sentinel(p.n()));
            ASSERT_EQ(p.sv_solve(p.A, x, y), rocsparse_status_success);
            expect_close(to_host(y), host_lower_solve(p.scalar, b, 1), tol);
            check(p.bsrsv_pivot(), is(no_pivot), "bsrsv_zero_pivot after the solve");
            return;
        }
        case routine::bsrsm:
        {
            const auto            b = right_hand_side(p.n(), block_problem::nrhs);
            device_vector<double> B(b), X(sentinel(b.size()));
            ASSERT_EQ(p.sm_solve(p.A, B, X), rocsparse_status_success);
            expect_close(to_host(X), host_lower_solve(p.scalar, b, block_problem::nrhs), tol);
            check(p.bsrsm_pivot(), is(no_pivot), "bsrsm_zero_pivot after the solve");
            return;
        }
        case routine::bsrilu0:
            ASSERT_EQ(p.ilu0(), rocsparse_status_success);
            expect_close(p.scalar_values(), host_ilu0(p.scalar), tol);
            check(p.bsrilu0_pivot(), is(no_pivot), "bsrilu0_zero_pivot");
            return;
        case routine::bsric0:
            ASSERT_EQ(p.ic0(), rocsparse_status_success);
            expect_close(keep_lower_only(p.scalar, p.scalar_values()),
                         keep_lower_only(p.scalar, host_ic0(p.scalar)),
                         tol);
            check(p.bsric0_pivot(), is(no_pivot), "bsric0_zero_pivot");
            return;
        }
    }

    bsr_matrix<double> test_matrix(rocsparse_direction dir  = rocsparse_direction_row,
                                   bool                skew = false)
    {
        return make_block_matrix<double>(5, 3, dir, skew);
    }

    // `consumer` analysed with the reuse policy after a forced analysis of `producer`,
    // then run on its own: it computes what it computes after an analysis of its own.
    void check_reuse(rocsparse_handle handle, routine producer, routine consumer)
    {
        SCOPED_TRACE(::testing::Message() << name(consumer) << " reusing " << name(producer));
        block_problem p(handle, test_matrix(), rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_EQ(analyse(p, producer, rocsparse_analysis_policy_force), rocsparse_status_success);
        ASSERT_EQ(analyse(p, consumer, rocsparse_analysis_policy_reuse), rocsparse_status_success);
        ASSERT_NO_FATAL_FAILURE(check_compute(p, consumer));
    }

    // A matrix of one block, its entries given row by row.
    bsr_matrix<double> one_block(int32_t block_dim, std::vector<double> values)
    {
        bsr_matrix<double> m;
        m.mb        = 1;
        m.block_dim = block_dim;
        m.dir       = rocsparse_direction_row;
        m.ptr       = {0, 1};
        m.ind       = {0};
        m.val       = std::move(values);
        return m;
    }

    // The analysis, solve and zero pivot of the solve `r` (bsrsv, or bsrsm with
    // block_problem::nrhs columns) with the triangle of `descr`.
    constexpr routine solves[] = {routine::bsrsv, routine::bsrsm};

    int32_t columns(routine r)
    {
        return (r == routine::bsrsm) ? block_problem::nrhs : 1;
    }

    rocsparse_status analyse_triangle(block_problem&            p,
                                      routine                   r,
                                      rocsparse_mat_descr       descr,
                                      rocsparse_analysis_policy policy)
    {
        return (r == routine::bsrsv) ? p.sv_analysis(descr, policy) : p.sm_analysis(descr, policy);
    }

    // Solves for `b` (right_hand_side by default) and returns the solution in `x`.
    rocsparse_status solve(block_problem&             p,
                           routine                    r,
                           rocsparse_mat_descr        descr,
                           std::vector<double>*       x = nullptr,
                           const std::vector<double>* b = nullptr)
    {
        const auto             rhs = b ? *b : right_hand_side(p.n(), columns(r));
        device_vector<double>  B(rhs), X(sentinel(rhs.size()));
        const rocsparse_status status
            = (r == routine::bsrsv) ? p.sv_solve(descr, B, X) : p.sm_solve(descr, B, X);
        if(x)
            *x = to_host(X);
        return status;
    }

    int32_t solve_pivot(block_problem& p, routine r)
    {
        return (r == routine::bsrsv) ? p.bsrsv_pivot() : p.bsrsm_pivot();
    }

    rocsparse_status clear_solve(block_problem& p, routine r)
    {
        return (r == routine::bsrsv) ? rocsparse_bsrsv_clear(p.handle, p.info)
                                     : rocsparse_bsrsm_clear(p.handle, p.info);
    }

    // The factorization `r` (bsrilu0 or bsric0) and its zero pivot.
    constexpr routine factorizations[] = {routine::bsrilu0, routine::bsric0};

    rocsparse_status factorize(block_problem& p, routine r)
    {
        return (r == routine::bsrilu0) ? p.ilu0() : p.ic0();
    }

    rocsparse_status clear_factorization(block_problem& p, routine r)
    {
        return (r == routine::bsrilu0) ? rocsparse_bsrilu0_clear(p.handle, p.info)
                                       : rocsparse_bsric0_clear(p.handle, p.info);
    }

    int32_t factorization_pivot(block_problem& p, routine r)
    {
        return (r == routine::bsrilu0) ? p.bsrilu0_pivot() : p.bsric0_pivot();
    }
}

class PrecondTriangularScenariosBsr : public HandleTest
{
};

// ---------------------------------------------------------------------------
// The block ILU(0) and IC(0) workflows on one mat_info
// ---------------------------------------------------------------------------

// bsrilu0_analysis, the analyses of L and U in either order, bsrilu0, then the solves
// with L and U: the factor and the preconditioned vector match the host, and no zero
// pivot is reported.
TEST_F(PrecondTriangularScenariosBsr, ilu0_workflow_values)
{
    for(const auto dir : directions)
    {
        const auto m = test_matrix(dir, true);
        const auto a = expand_bsr(m);
        const auto b = right_hand_side(a.n);
        for(const auto base : bases)
        {
            for(const auto o : orders)
            {
                for(const auto policy : policies)
                {
                    SCOPED_TRACE(::testing::Message() << name(dir) << ", " << name(base) << ", "
                                                      << name(o) << ", " << name(policy));
                    block_problem p(handle, m, base);
                    ASSERT_NO_FATAL_FAILURE(p.init());
                    ASSERT_NO_FATAL_FAILURE(p.ilu0_workflow(o, policy));
                    check(p.bsrilu0_pivot(), is(no_pivot), "bsrilu0_zero_pivot");

                    const auto lu = p.scalar_values();
                    expect_close(lu, host_ilu0(a), tolerance<double>(max_row_length(a)));
                    expect_close(p.lu_solve(b),
                                 host_lu_solve(a, lu, b),
                                 tolerance<double>(max_row_length(a)));
                    check(p.bsrsv_pivot(), is(no_pivot), "bsrsv_zero_pivot after the solves");
                }
            }
        }
    }
}

// bsric0_analysis, the analyses of A and A^T in either order, bsric0, then the solves
// with L and L^T.
TEST_F(PrecondTriangularScenariosBsr, ic0_workflow_values)
{
    for(const auto dir : directions)
    {
        const auto m = test_matrix(dir);
        const auto a = expand_bsr(m);
        const auto b = right_hand_side(a.n);
        for(const auto base : bases)
        {
            for(const auto o : orders)
            {
                for(const auto policy : policies)
                {
                    SCOPED_TRACE(::testing::Message() << name(dir) << ", " << name(base) << ", "
                                                      << name(o) << ", " << name(policy));
                    block_problem p(handle, m, base);
                    ASSERT_NO_FATAL_FAILURE(p.init());
                    ASSERT_NO_FATAL_FAILURE(p.ic0_workflow(o, policy));
                    check(p.bsric0_pivot(), is(no_pivot), "bsric0_zero_pivot");

                    const auto l = p.scalar_values();
                    expect_close(keep_lower_only(a, l),
                                 keep_lower_only(a, host_ic0(a)),
                                 tolerance<double>(max_row_length(a)));
                    expect_close(p.llt_solve(b),
                                 host_llt_solve(a, l, b),
                                 tolerance<double>(max_row_length(a)));
                    check(p.bsrsv_pivot(), is(no_pivot), "bsrsv_zero_pivot after the solves");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// One routine reusing the analysis of another
// ---------------------------------------------------------------------------

// The pairs that compute correctly today. bsrsv and bsrsm take the analysis of the
// other routines (and run without a zero pivot of their own, which a non unit solve
// without a zero does not touch); bsrilu0 never looks at the analysis of bsrsm, and
// bsric0 at neither bsrsm nor bsrilu0, so these run a full analysis.
TEST_F(PrecondTriangularScenariosBsr, reuse_of_another_analysis)
{
    const std::pair<routine, routine> pairs[] = {{routine::bsrsm, routine::bsrsv},
                                                 {routine::bsrilu0, routine::bsrsv},
                                                 {routine::bsric0, routine::bsrsv},
                                                 {routine::bsrsv, routine::bsrsm},
                                                 {routine::bsrilu0, routine::bsrsm},
                                                 {routine::bsric0, routine::bsrsm},
                                                 {routine::bsrsm, routine::bsrilu0},
                                                 {routine::bsrsm, routine::bsric0},
                                                 {routine::bsrilu0, routine::bsric0}};
    for(const auto& [producer, consumer] : pairs)
        ASSERT_NO_FATAL_FAILURE(check_reuse(handle, producer, consumer));
}

// A reused analysis leaves the routine without a zero pivot of its own: the structural
// zero that the analysis found is reported by the routine that made it only, in host and
// in device memory.
TEST_F(PrecondTriangularScenariosBsr, reused_analysis_loses_the_structural_pivot)
{
    const int32_t r = 2;
    const auto    m = without_block(test_matrix(), r, r);
    const auto    b = right_hand_side(m.mb * m.block_dim, block_problem::nrhs);
    for(const bool device_query : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "device query " << device_query);
        {
            SCOPED_TRACE("bsrsv reusing bsrsm");
            block_problem p(handle, m, rocsparse_index_base_zero, device_query);
            ASSERT_NO_FATAL_FAILURE(p.init());
            ASSERT_EQ(p.sm_analysis(p.A, rocsparse_analysis_policy_force),
                      rocsparse_status_success);
            ASSERT_EQ(p.sv_analysis(p.A, rocsparse_analysis_policy_reuse),
                      rocsparse_status_success);
            device_vector<double> x(b), y(p.n());
            ASSERT_EQ(p.sv_solve(p.A, x, y), rocsparse_status_success);
            check(p.bsrsm_pivot(), is(r), "bsrsm_zero_pivot after its analysis");
            check(p.bsrsv_pivot(),
                  known_defect(no_pivot,
                               r,
                               "the reused analysis does not give bsrsv a zero pivot, so the "
                               "structural zero stays with bsrsm"),
                  "bsrsv_zero_pivot after the solve");
        }
        {
            SCOPED_TRACE("bsrsm reusing bsrsv");
            block_problem p(handle, m, rocsparse_index_base_zero, device_query);
            ASSERT_NO_FATAL_FAILURE(p.init());
            ASSERT_EQ(p.sv_analysis(p.A, rocsparse_analysis_policy_force),
                      rocsparse_status_success);
            ASSERT_EQ(p.sm_analysis(p.A, rocsparse_analysis_policy_reuse),
                      rocsparse_status_success);
            device_vector<double> B(b), X(b.size());
            ASSERT_EQ(p.sm_solve(p.A, B, X), rocsparse_status_success);
            check(p.bsrsv_pivot(), is(r), "bsrsv_zero_pivot after its analysis");
            check(p.bsrsm_pivot(),
                  known_defect(no_pivot,
                               r,
                               "the reused analysis does not give bsrsm a zero pivot, so the "
                               "structural zero stays with bsrsv"),
                  "bsrsm_zero_pivot after the solve");
        }
    }
}

// A unit diagonal solve of bsrsm after a reused analysis resets a zero pivot that was
// never allocated.
TEST_F(PrecondTriangularScenariosBsr, bsrsm_unit_solve_after_a_reused_analysis)
{
    block_problem p(handle, test_matrix(), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    ASSERT_EQ(p.sv_analysis(p.L, rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_EQ(p.sm_analysis(p.L, rocsparse_analysis_policy_reuse), rocsparse_status_success);
    const auto            b = right_hand_side(p.n(), block_problem::nrhs);
    device_vector<double> B(b), X(b.size());
    check(p.sm_solve(p.L, B, X),
          known_defect(rocsparse_status_internal_error,
                       rocsparse_status_success,
                       "the reused analysis does not give bsrsm a zero pivot, which the unit "
                       "diagonal solve resets"),
          "bsrsm_solve with L after reusing the analysis of bsrsv");
    // The failed copy leaves its error behind, which the next rocsparse_create_handle
    // would return.
    (void)hipGetLastError();
}

// ---------------------------------------------------------------------------
// Calls without an analysis, and clear
// ---------------------------------------------------------------------------

// bsrsv, bsrilu0 and bsric0 need their analysis: without one, for another triangle or
// operation, or after their clear they return rocsparse_status_invalid_pointer and leave
// their output alone. A new analysis after the clear works again.
TEST_F(PrecondTriangularScenariosBsr, calls_need_their_analysis)
{
    const auto m = test_matrix();
    for(const auto policy : policies)
    {
        SCOPED_TRACE(name(policy));
        block_problem p(handle, m, rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        const auto                b = right_hand_side(p.n());
        const std::vector<double> untouched(p.n(), 7.0);
        device_vector<double>     x(b), y(untouched);

        check(p.sv_solve(p.A, x, y),
              is(rocsparse_status_invalid_pointer),
              "bsrsv_solve without an analysis");
        check(p.ilu0(), is(rocsparse_status_invalid_pointer), "bsrilu0 without an analysis");
        check(p.ic0(), is(rocsparse_status_invalid_pointer), "bsric0 without an analysis");
        EXPECT_EQ(p.values(), m.val);

        ASSERT_EQ(p.sv_analysis(p.A, policy), rocsparse_status_success);
        check(p.sv_solve(p.A, x, y, rocsparse_operation_transpose),
              is(rocsparse_status_invalid_pointer),
              "bsrsv_solve with A^T after the analysis of A");
        check(p.sv_solve(p.U, x, y),
              is(rocsparse_status_invalid_pointer),
              "bsrsv_solve with U after the analysis of A");
        EXPECT_EQ(to_host(y), untouched);

        ASSERT_EQ(rocsparse_bsrsv_clear(handle, p.info), rocsparse_status_success);
        check(p.sv_solve(p.A, x, y),
              is(rocsparse_status_invalid_pointer),
              "bsrsv_solve after bsrsv_clear");
        check(p.bsrsv_pivot(), is(no_pivot), "bsrsv_zero_pivot after bsrsv_clear");
        EXPECT_EQ(to_host(y), untouched);
        ASSERT_EQ(p.sv_analysis(p.A, policy), rocsparse_status_success);
        ASSERT_NO_FATAL_FAILURE(check_compute(p, routine::bsrsv));

        for(const auto r : factorizations)
        {
            SCOPED_TRACE(name(r));
            ASSERT_NO_FATAL_FAILURE(p.set_values());
            ASSERT_EQ(analyse(p, r, rocsparse_analysis_policy_force), rocsparse_status_success);
            ASSERT_NO_FATAL_FAILURE(check_compute(p, r));
            ASSERT_NO_FATAL_FAILURE(p.set_values());
            ASSERT_EQ(clear_factorization(p, r), rocsparse_status_success);
            check(factorize(p, r),
                  is(rocsparse_status_invalid_pointer),
                  std::string(name(r)) + " after its clear");
            EXPECT_EQ(p.values(), m.val);
        }
    }
}

// Clearing one routine keeps the analysis data that the others use, also when they
// share it.
TEST_F(PrecondTriangularScenariosBsr, clear_keeps_the_other_routines)
{
    for(const auto policy : policies)
    {
        for(const bool clear_bsrsv : {true, false})
        {
            SCOPED_TRACE(::testing::Message()
                         << name(policy) << ", clear " << (clear_bsrsv ? "bsrsv" : "bsrsm"));
            block_problem p(handle, test_matrix(), rocsparse_index_base_zero);
            ASSERT_NO_FATAL_FAILURE(p.init());
            ASSERT_EQ(p.sm_analysis(p.A, rocsparse_analysis_policy_force),
                      rocsparse_status_success);
            ASSERT_EQ(p.sv_analysis(p.A, policy), rocsparse_status_success);
            if(clear_bsrsv)
            {
                ASSERT_EQ(rocsparse_bsrsv_clear(handle, p.info), rocsparse_status_success);
                ASSERT_NO_FATAL_FAILURE(check_compute(p, routine::bsrsm));
            }
            else
            {
                ASSERT_EQ(rocsparse_bsrsm_clear(handle, p.info), rocsparse_status_success);
                ASSERT_NO_FATAL_FAILURE(check_compute(p, routine::bsrsv));
            }
        }
    }

    for(const auto policy : policies)
    {
        SCOPED_TRACE(::testing::Message() << "ILU(0), " << name(policy));
        const auto    m = test_matrix(rocsparse_direction_row, true);
        const auto    b = right_hand_side(m.mb * m.block_dim);
        block_problem p(handle, m, rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_NO_FATAL_FAILURE(p.ilu0_workflow(order::lower_first, policy));
        const auto lu = p.scalar_values();
        ASSERT_EQ(rocsparse_bsrilu0_clear(handle, p.info), rocsparse_status_success);
        expect_close(p.lu_solve(b),
                     host_lu_solve(p.scalar, lu, b),
                     tolerance<double>(max_row_length(p.scalar)));

        block_problem q(handle, m, rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(q.init());
        ASSERT_NO_FATAL_FAILURE(q.ilu0_workflow(order::lower_first, policy));
        ASSERT_EQ(rocsparse_bsrsv_clear(handle, q.info), rocsparse_status_success);
        ASSERT_NO_FATAL_FAILURE(q.set_values());
        ASSERT_NO_FATAL_FAILURE(check_compute(q, routine::bsrilu0));
    }

    for(const auto policy : policies)
    {
        SCOPED_TRACE(::testing::Message() << "IC(0), " << name(policy));
        const auto    m = test_matrix();
        const auto    b = right_hand_side(m.mb * m.block_dim);
        block_problem p(handle, m, rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_NO_FATAL_FAILURE(p.ic0_workflow(order::lower_first, policy));
        const auto l = p.scalar_values();
        ASSERT_EQ(rocsparse_bsric0_clear(handle, p.info), rocsparse_status_success);
        expect_close(p.llt_solve(b),
                     host_llt_solve(p.scalar, l, b),
                     tolerance<double>(max_row_length(p.scalar)));
    }
}

// A copy of the mat_info carries the analysis data that bsrilu0, bsrsv and bsrsm share
// and the zero pivots, and stays usable for solves once the original is destroyed.
TEST_F(PrecondTriangularScenariosBsr, copy_outlives_the_original)
{
    const int32_t r     = 2;
    const auto    clean = test_matrix(rocsparse_direction_row, true);
    const int32_t n     = clean.mb * clean.block_dim;
    const auto    b     = right_hand_side(n);
    const auto    bm    = right_hand_side(n, block_problem::nrhs);
    for(const bool missing_diagonal : {false, true})
    {
        const auto    m    = missing_diagonal ? without_block(clean, r, r) : clean;
        const int32_t want = missing_diagonal ? r : no_pivot;
        for(const auto base : bases)
        {
            for(const auto policy : policies)
            {
                SCOPED_TRACE(::testing::Message() << "missing diagonal " << missing_diagonal << ", "
                                                  << name(base) << ", " << name(policy));
                block_problem p(handle, m, base);
                ASSERT_NO_FATAL_FAILURE(p.init());
                ASSERT_NO_FATAL_FAILURE(p.ilu0_workflow(order::lower_first, policy));
                ASSERT_EQ(p.sm_analysis(p.A, policy), rocsparse_status_success);
                const auto f = with_values(p.scalar, p.scalar_values());

                ASSERT_EQ(p.replace_info_by_copy(), rocsparse_status_success);
                check(p.bsrilu0_pivot(), is(want), "bsrilu0_zero_pivot of the copy");
                check(p.bsrsv_pivot(), is(want), "bsrsv_zero_pivot of the copy");
                if(!missing_diagonal)
                {
                    const auto tol = tolerance<double>(max_row_length(f));
                    expect_close(p.lu_solve(b), host_lu_solve(p.scalar, f.val, b), tol);
                    check(p.bsrsv_pivot(), is(no_pivot), "bsrsv_zero_pivot after the solves");
                    device_vector<double> B(bm), X(sentinel(bm.size()));
                    ASSERT_EQ(p.sm_solve(p.A, B, X), rocsparse_status_success);
                    expect_close(to_host(X), host_lower_solve(f, bm, block_problem::nrhs), tol);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Repeated analyses and a new pattern
// ---------------------------------------------------------------------------

// A second analysis of the same matrix, reused or forced, leaves a working analysis.
TEST_F(PrecondTriangularScenariosBsr, repeated_analysis)
{
    for(const auto r : {routine::bsrsv, routine::bsrsm, routine::bsrilu0, routine::bsric0})
    {
        for(const auto policy : policies)
        {
            SCOPED_TRACE(::testing::Message() << name(r) << ", then " << name(policy));
            block_problem p(handle, test_matrix(), rocsparse_index_base_zero);
            ASSERT_NO_FATAL_FAILURE(p.init());
            ASSERT_EQ(analyse(p, r, rocsparse_analysis_policy_force), rocsparse_status_success);
            ASSERT_EQ(analyse(p, r, policy), rocsparse_status_success);
            ASSERT_NO_FATAL_FAILURE(check_compute(p, r));
        }
    }
}

// The analysis belongs to the pattern it was made for. A new pattern on the mat_info
// needs bsrilu0_clear and bsrsv_clear first, or forced analyses of every routine when
// none shares an analysis.
TEST_F(PrecondTriangularScenariosBsr, ilu0_workflow_new_pattern)
{
    const auto m     = test_matrix(rocsparse_direction_row, true);
    const auto m_new = without_block(without_block(m, 2, 1), 1, 2);
    const auto a_new = expand_bsr(m_new);
    const auto b     = right_hand_side(a_new.n);
    const auto tol   = tolerance<double>(max_row_length(a_new));
    for(const auto policy : policies)
    {
        for(const bool clear : {true, false})
        {
            if(!clear && policy == rocsparse_analysis_policy_reuse)
                continue;
            SCOPED_TRACE(::testing::Message()
                         << name(policy) << (clear ? ", cleared" : ", forced"));
            block_problem p(handle, m, rocsparse_index_base_zero);
            ASSERT_NO_FATAL_FAILURE(p.init());
            ASSERT_NO_FATAL_FAILURE(p.ilu0_workflow(order::lower_first, policy));
            (void)p.lu_solve(b);
            if(clear)
            {
                ASSERT_EQ(rocsparse_bsrilu0_clear(handle, p.info), rocsparse_status_success);
                ASSERT_EQ(rocsparse_bsrsv_clear(handle, p.info), rocsparse_status_success);
            }

            ASSERT_NO_FATAL_FAILURE(p.replace_matrix(m_new));
            ASSERT_NO_FATAL_FAILURE(p.ilu0_workflow(order::lower_first, policy));
            const auto lu = p.scalar_values();
            expect_close(lu, host_ilu0(a_new), tol);
            check(p.bsrilu0_pivot(), is(no_pivot), "bsrilu0_zero_pivot");
            expect_close(p.lu_solve(b), host_lu_solve(a_new, lu, b), tol);
        }
    }
}

// The same for bsrsm with forced analyses.
TEST_F(PrecondTriangularScenariosBsr, bsrsm_new_pattern)
{
    const auto    m = test_matrix();
    block_problem p(handle, m, rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    ASSERT_EQ(p.sm_analysis(p.A, rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(check_compute(p, routine::bsrsm));
    ASSERT_NO_FATAL_FAILURE(p.replace_matrix(without_block(m, 2, 1)));
    ASSERT_EQ(p.sm_analysis(p.A, rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(check_compute(p, routine::bsrsm));
}

// ---------------------------------------------------------------------------
// Zero pivots and unit diagonal of the solves
// ---------------------------------------------------------------------------

// With a missing diagonal block and a zero on the diagonal of another, the solve reports
// the first block row, whichever kind of zero it holds. The analysis only finds the
// missing block.
TEST_F(PrecondTriangularScenariosBsr, first_of_several_zero_pivots)
{
    const auto                        clean   = test_matrix();
    const std::pair<int32_t, int32_t> cases[] = {{1, 3}, {3, 1}};
    for(const auto& [numerical, structural] : cases)
    {
        const auto m = without_block(
            with_zero_in_diagonal_block(clean, numerical, 1), structural, structural);
        for(const auto r : solves)
        {
            for(const auto base : bases)
            {
                SCOPED_TRACE(::testing::Message()
                             << name(r) << ", zero in block row " << numerical << ", block row "
                             << structural << " without its diagonal block, " << name(base));
                block_problem p(handle, m, base);
                ASSERT_NO_FATAL_FAILURE(p.init());
                ASSERT_EQ(analyse_triangle(p, r, p.A, rocsparse_analysis_policy_force),
                          rocsparse_status_success);
                check(solve_pivot(p, r), is(structural), "zero pivot after the analysis");
                ASSERT_EQ(solve(p, r, p.A), rocsparse_status_success);
                check(solve_pivot(p, r),
                      is(std::min(numerical, structural)),
                      "zero pivot after the solve");
            }
        }
    }
}

// The numerical zero pivot belongs to the values solved with: once the zero on the
// diagonal is replaced, the next solve reports none. A reused analysis finds the existing
// one and returns without resetting it; a forced analysis of the new values resets it,
// which is how a caller recovers today.
TEST_F(PrecondTriangularScenariosBsr, numerical_pivot_of_older_values)
{
    const auto regular = test_matrix();
    const auto zero    = with_zero_in_diagonal_block(regular, 1, 1);
    for(const auto r : solves)
    {
        SCOPED_TRACE(name(r));
        block_problem p(handle, zero, rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_EQ(analyse_triangle(p, r, p.A, rocsparse_analysis_policy_force),
                  rocsparse_status_success);
        ASSERT_EQ(solve(p, r, p.A), rocsparse_status_success);
        check(solve_pivot(p, r), is(1), "zero pivot of the solve with the zero");

        ASSERT_NO_FATAL_FAILURE(p.set_values(&regular.val));
        ASSERT_EQ(solve(p, r, p.A), rocsparse_status_success);
        check(solve_pivot(p, r),
              known_defect(1,
                           no_pivot,
                           "the solve only lowers the zero pivot, so the one of the earlier "
                           "values stays"),
              "zero pivot of the solve with the zero replaced");
        ASSERT_EQ(analyse_triangle(p, r, p.A, rocsparse_analysis_policy_reuse),
                  rocsparse_status_success);
        ASSERT_EQ(solve(p, r, p.A), rocsparse_status_success);
        check(solve_pivot(p, r),
              known_defect(1,
                           no_pivot,
                           "the reused analysis keeps the zero pivot, which the solve only "
                           "lowers"),
              "zero pivot after a reused analysis and solve");
        ASSERT_EQ(analyse_triangle(p, r, p.A, rocsparse_analysis_policy_force),
                  rocsparse_status_success);
        ASSERT_EQ(solve(p, r, p.A), rocsparse_status_success);
        check(solve_pivot(p, r), is(no_pivot), "zero pivot after a forced analysis and solve");
    }
}

// bsrsv and bsrsm keep one zero pivot for both triangles of the mat_info, and the
// documentation makes it the pivot of the matrix: the structural zero of the upper
// triangle is still reported after an analysis of the unit lower triangle.
TEST_F(PrecondTriangularScenariosBsr, pivot_of_one_triangle_survives_the_other)
{
    const int32_t row = 2;
    const auto    m   = without_block(test_matrix(), row, row);
    for(const auto r : solves)
    {
        SCOPED_TRACE(name(r));
        block_problem p(handle, m, rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_EQ(analyse_triangle(p, r, p.U, rocsparse_analysis_policy_force),
                  rocsparse_status_success);
        check(solve_pivot(p, r), is(row), "zero pivot after the analysis of U");
        ASSERT_EQ(analyse_triangle(p, r, p.L, rocsparse_analysis_policy_force),
                  rocsparse_status_success);
        ASSERT_EQ(solve(p, r, p.U), rocsparse_status_success);
        check(solve_pivot(p, r),
              known_defect(no_pivot,
                           row,
                           "the analysis of the unit lower triangle resets the zero pivot that "
                           "the routine keeps for both triangles"),
              "zero pivot after the analysis of L and the solve with U");
    }
}

// With a unit diagonal the solve ignores the diagonal, a missing diagonal block or a
// zero on the diagonal: no zero pivot, and the solution of the unit triangle.
TEST_F(PrecondTriangularScenariosBsr, unit_diagonal)
{
    const auto clean = test_matrix();
    for(const bool missing : {true, false})
    {
        const auto m
            = missing ? without_block(clean, 2, 2) : with_zero_in_diagonal_block(clean, 1, 1);
        for(const auto r : solves)
        {
            for(const auto base : bases)
            {
                SCOPED_TRACE(::testing::Message()
                             << name(r) << ", " << (missing ? "missing block" : "zero diagonal")
                             << ", " << name(base));
                block_problem p(handle, m, base);
                ASSERT_NO_FATAL_FAILURE(p.init());
                ASSERT_EQ(analyse_triangle(p, r, p.L, rocsparse_analysis_policy_force),
                          rocsparse_status_success);
                std::vector<double> x;
                ASSERT_EQ(solve(p, r, p.L, &x), rocsparse_status_success);
                expect_close(x,
                             host_lower_solve(p.scalar,
                                              right_hand_side(p.n(), columns(r)),
                                              columns(r),
                                              rocsparse_diag_type_unit),
                             tolerance<double>(max_row_length(p.scalar)));
                check(solve_pivot(p, r), is(no_pivot), "zero pivot");
            }
        }
    }
}

// The zero pivot that a non unit analysis found is reported again by a non unit solve
// that follows a unit diagonal solve.
TEST_F(PrecondTriangularScenariosBsr, unit_solve_keeps_the_non_unit_pivot)
{
    const int32_t row = 2;
    const auto    m   = without_block(test_matrix(), row, row);
    for(const auto r : solves)
    {
        SCOPED_TRACE(name(r));
        block_problem p(handle, m, rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_EQ(analyse_triangle(p, r, p.A, rocsparse_analysis_policy_force),
                  rocsparse_status_success);
        check(solve_pivot(p, r), is(row), "zero pivot after the analysis");
        ASSERT_EQ(rocsparse_set_mat_diag_type(p.A, rocsparse_diag_type_unit),
                  rocsparse_status_success);
        ASSERT_EQ(solve(p, r, p.A), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_diag_type(p.A, rocsparse_diag_type_non_unit),
                  rocsparse_status_success);
        ASSERT_EQ(solve(p, r, p.A), rocsparse_status_success);
        check(solve_pivot(p, r),
              known_defect(no_pivot,
                           row,
                           "the unit diagonal solve resets the zero pivot, and the non unit "
                           "solve does not record the missing block again"),
              "zero pivot after the unit and the non unit solve");
    }
}

// With a unit diagonal the solve takes the diagonal of each diagonal block as 1 and uses
// the rest of its lower triangle: it solves with the scalar lower triangle of the
// expanded matrix, as check_compute assumes throughout. The bsrsv and bsrsm pages
// describe identity blocks on the diagonal instead, under which the solve of
// [[5, 0], [3, 5]] with b = (1, 1) is (1, 1); the pages need to describe the scalar
// solve.
TEST_F(PrecondTriangularScenariosBsr, unit_diagonal_of_a_block)
{
    const auto m = one_block(2, {5, 0, 3, 5});
    for(const auto r : solves)
    {
        SCOPED_TRACE(name(r));
        block_problem p(handle, m, rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_EQ(analyse_triangle(p, r, p.L, rocsparse_analysis_policy_force),
                  rocsparse_status_success);
        const std::vector<double> b(static_cast<size_t>(p.n()) * columns(r), 1.0);
        std::vector<double>       want;
        for(int32_t j = 0; j < columns(r); ++j)
            want.insert(want.end(), {1.0, -2.0});
        std::vector<double> x;
        ASSERT_EQ(solve(p, r, p.L, &x, &b), rocsparse_status_success);
        expect_close(x, want, tolerance<double>(2));
    }
}

// The solve uses the lower triangle of the diagonal block: a zero on its diagonal is a
// zero pivot even when the block is invertible, and a singular block with a nonzero
// diagonal is not. The bsrsv and bsrsm pages ("inverting the diagonal blocks") make a
// block that is not invertible the numerical zero instead, under which [[0, 1], [1, 0]]
// has none and [[1, 1], [1, 1]] has one at 0; the pages need to describe the scalar
// solve.
TEST_F(PrecondTriangularScenariosBsr, numerical_zero_of_a_diagonal_block)
{
    const std::pair<std::vector<double>, int32_t> cases[]
        = {{{0, 1, 1, 0}, 0}, {{1, 1, 1, 1}, no_pivot}};
    for(const auto& [values, want] : cases)
    {
        for(const auto r : solves)
        {
            SCOPED_TRACE(::testing::Message()
                         << name(r) << ", block " << values[0] << " " << values[1] << " / "
                         << values[2] << " " << values[3]);
            block_problem p(handle, one_block(2, values), rocsparse_index_base_zero);
            ASSERT_NO_FATAL_FAILURE(p.init());
            ASSERT_EQ(analyse_triangle(p, r, p.A, rocsparse_analysis_policy_force),
                      rocsparse_status_success);
            ASSERT_EQ(solve(p, r, p.A), rocsparse_status_success);
            check(solve_pivot(p, r), is(want), "zero pivot");
        }
    }
}

// The clear of a solve resets its zero pivot.
TEST_F(PrecondTriangularScenariosBsr, clear_resets_the_pivot)
{
    const int32_t row = 2;
    for(const auto r : solves)
    {
        SCOPED_TRACE(name(r));
        block_problem p(handle, without_block(test_matrix(), row, row), rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_EQ(analyse_triangle(p, r, p.A, rocsparse_analysis_policy_force),
                  rocsparse_status_success);
        check(solve_pivot(p, r), is(row), "zero pivot after the analysis");
        ASSERT_EQ(clear_solve(p, r), rocsparse_status_success);
        check(solve_pivot(p, r), is(no_pivot), "zero pivot after the clear");
    }
}

// ---------------------------------------------------------------------------
// Pivots and boost of the factorizations
// ---------------------------------------------------------------------------

// A missing diagonal block is a zero pivot from the analysis on. The pages make a
// numerical zero a diagonal block that is not invertible (bsrilu0) or not positive
// definite (bsric0). bsrilu0 factorizes the block in place, without pivoting (the BSR
// arrays have nowhere to keep a permutation), so its zero pivot is that of the scalar
// factorization, as for the solves: the bsrilu0 page needs to say so. bsric0 is the one
// that is wrong: the negative pivot of a block that is not positive definite is not
// reported.
TEST_F(PrecondTriangularScenariosBsr, factorization_zero_pivots)
{
    const int32_t row = 2;
    for(const auto r : factorizations)
    {
        SCOPED_TRACE(name(r));
        block_problem p(handle, without_block(test_matrix(), row, row), rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_EQ(analyse(p, r, rocsparse_analysis_policy_force), rocsparse_status_success);
        check(factorization_pivot(p, r), is(row), "zero pivot after the analysis");
    }
    {
        SCOPED_TRACE("bsrilu0 of an invertible block with a zero leading entry");
        block_problem p(handle, one_block(2, {0, 1, 1, 0}), rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_EQ(p.ilu0_analysis(rocsparse_analysis_policy_force), rocsparse_status_success);
        ASSERT_EQ(p.ilu0(), rocsparse_status_success);
        check(p.bsrilu0_pivot(), is(0), "bsrilu0_zero_pivot");
    }
    {
        SCOPED_TRACE("bsric0 of a block that is not positive definite");
        block_problem p(handle, one_block(2, {-4, 0, 0, 1}), rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_EQ(p.ic0_analysis(rocsparse_analysis_policy_force), rocsparse_status_success);
        ASSERT_EQ(p.ic0(), rocsparse_status_success);
        check(p.bsric0_pivot(),
              known_defect(no_pivot,
                           0,
                           "bsric0 takes the square root of the modulus of the pivot, so it "
                           "only reports a pivot that is exactly zero"),
              "bsric0_zero_pivot");
    }
}

// A numerical zero pivot belongs to the values factorized: a factorization of new
// values without one reports none, and so does a new analysis, which only finds
// missing blocks.
TEST_F(PrecondTriangularScenariosBsr, factorization_pivot_of_older_values)
{
    // The second pivot of the singular block cancels exactly.
    const auto singular = one_block(2, {1, 1, 1, 1});
    const auto regular  = one_block(2, {2, 1, 1, 2});
    for(const auto r : factorizations)
    {
        SCOPED_TRACE(name(r));
        block_problem p(handle, singular, rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_EQ(analyse(p, r, rocsparse_analysis_policy_force), rocsparse_status_success);
        ASSERT_EQ(factorize(p, r), rocsparse_status_success);
        check(factorization_pivot(p, r), is(0), "zero pivot of the singular values");

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
              known_defect(0,
                           no_pivot,
                           "the analysis keeps the numerical zero pivot of the last "
                           "factorization"),
              "zero pivot after a new analysis of regular values");
    }
}

// A pivot with |pivot| <= boost_tol becomes |boost_val| with the sign of the pivot, and
// disabling the boost on the same mat_info leaves the next factorization alone. The sign
// is kept on purpose, to keep the inertia of A; the bsrilu0_numeric_boost page, which
// gives the boosted pivot as boost_val, needs to say so.
TEST_F(PrecondTriangularScenariosBsr, bsrilu0_numeric_boost)
{
    const auto    m         = one_block(2, {-1e-3, 1, 1, 4});
    const double  boost_tol = 1e-2;
    const double  boost_val = 2.0;
    block_problem p(handle, m, rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    ASSERT_EQ(p.ilu0_analysis(rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_EQ(rocsparse_dbsrilu0_numeric_boost(handle, p.info, 1, &boost_tol, &boost_val),
              rocsparse_status_success);
    ASSERT_EQ(p.ilu0(), rocsparse_status_success);
    check(p.values()[0], is(-2.0), "the boosted pivot");

    ASSERT_NO_FATAL_FAILURE(p.set_values());
    ASSERT_EQ(rocsparse_dbsrilu0_numeric_boost(handle, p.info, 0, &boost_tol, &boost_val),
              rocsparse_status_success);
    ASSERT_EQ(p.ilu0(), rocsparse_status_success);
    check(p.values()[0], is(-1e-3), "the pivot with the boost disabled");
}

// ---------------------------------------------------------------------------
// Statuses and buffer sizes
// ---------------------------------------------------------------------------

namespace
{
    constexpr rocsparse_operation conj_transpose = rocsparse_operation_conjugate_transpose;

    // The leading dimension of the right-hand sides of bsrsm with `nrhs` columns.
    int32_t bsrsm_ld(const block_problem& p, rocsparse_operation trans_X, int32_t nrhs)
    {
        return (trans_X == rocsparse_operation_none) ? p.n() : nrhs;
    }

    rocsparse_status bsrsm_buffer_size(block_problem&      p,
                                       rocsparse_operation trans_A,
                                       rocsparse_operation trans_X,
                                       int32_t             nrhs,
                                       size_t*             size)
    {
        return rocsparse_dbsrsm_buffer_size(p.handle,
                                            p.m.dir,
                                            trans_A,
                                            trans_X,
                                            p.m.mb,
                                            nrhs,
                                            p.m.nnzb(),
                                            p.A,
                                            p.val,
                                            p.row,
                                            p.col,
                                            p.m.block_dim,
                                            p.info,
                                            size);
    }

    size_t bsrsv_buffer_size(block_problem& p, rocsparse_operation op)
    {
        size_t size = 0;
        EXPECT_EQ(rocsparse_dbsrsv_buffer_size(p.handle,
                                               p.m.dir,
                                               op,
                                               p.m.mb,
                                               p.m.nnzb(),
                                               p.A,
                                               p.val,
                                               p.row,
                                               p.col,
                                               p.m.block_dim,
                                               p.info,
                                               &size),
                  rocsparse_status_success);
        return size;
    }
}

// The bsrilu0 and bsric0 buffer size pages say that their size is identical to that of
// bsrsv on one pattern, so that one buffer can be shared, and name no operation (the
// bsrsv page says nothing of it). That holds for bsrsv with op none; with op transpose
// bsrsv needs more, so a buffer sized by bsrilu0_buffer_size or bsric0_buffer_size is
// too small for it. The bsrilu0 and bsric0 pages need to restrict the statement to op
// none.
TEST_F(PrecondTriangularScenariosBsr, buffer_size_of_the_solve_and_the_factorizations)
{
    for(const auto dir : directions)
    {
        SCOPED_TRACE(name(dir));
        block_problem p(handle, test_matrix(dir), rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        size_t ilu0 = 0, ic0 = 0;
        ASSERT_EQ(rocsparse_dbsrilu0_buffer_size(handle,
                                                 dir,
                                                 p.m.mb,
                                                 p.m.nnzb(),
                                                 p.A,
                                                 p.val,
                                                 p.row,
                                                 p.col,
                                                 p.m.block_dim,
                                                 p.info,
                                                 &ilu0),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_dbsric0_buffer_size(handle,
                                                dir,
                                                p.m.mb,
                                                p.m.nnzb(),
                                                p.A,
                                                p.val,
                                                p.row,
                                                p.col,
                                                p.m.block_dim,
                                                p.info,
                                                &ic0),
                  rocsparse_status_success);
        const size_t none      = bsrsv_buffer_size(p, rocsparse_operation_none);
        const size_t transpose = bsrsv_buffer_size(p, rocsparse_operation_transpose);

        check(ilu0 == none,
              is(true),
              compared("bsrilu0_buffer_size equals bsrsv_buffer_size (op none)", ilu0, none));
        check(ic0 == none,
              is(true),
              compared("bsric0_buffer_size equals bsrsv_buffer_size (op none)", ic0, none));
        check(transpose > none,
              is(true),
              compared("bsrsv_buffer_size is larger for op transpose", transpose, none));
    }
}

// The bsrsm buffer is larger for a transposed A, and for an X that is not transposed.
TEST_F(PrecondTriangularScenariosBsr, bsrsm_buffer_size_by_operation)
{
    const int32_t nrhs = 4;
    block_problem p(handle, test_matrix(), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    const auto size = [&](rocsparse_operation trans_A, rocsparse_operation trans_X) {
        size_t s = 0;
        EXPECT_EQ(bsrsm_buffer_size(p, trans_A, trans_X, nrhs, &s), rocsparse_status_success);
        return s;
    };
    const auto N = rocsparse_operation_none;
    const auto T = rocsparse_operation_transpose;

    check(size(T, N) > size(N, N),
          is(true),
          compared("bsrsm_buffer_size: trans_A T against N", size(T, N), size(N, N)));
    check(size(N, N) > size(N, T),
          is(true),
          compared("bsrsm_buffer_size: trans_X N against T", size(N, N), size(N, T)));
}

// The documentation rejects conjugate_transpose with not_implemented. The analysis and
// the solve of bsrsv do; its buffer size and all three calls of bsrsm accept it (bsrsm
// then solves with A for trans_A, and with X^T for trans_X).
TEST_F(PrecondTriangularScenariosBsr, conjugate_transpose)
{
    const auto expected = known_defect(rocsparse_status_success,
                                       rocsparse_status_not_implemented,
                                       "conjugate_transpose is accepted");
    {
        SCOPED_TRACE("bsrsv");
        block_problem p(handle, test_matrix(), rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        size_t size = 0;
        check(rocsparse_dbsrsv_buffer_size(handle,
                                           p.m.dir,
                                           conj_transpose,
                                           p.m.mb,
                                           p.m.nnzb(),
                                           p.A,
                                           p.val,
                                           p.row,
                                           p.col,
                                           p.m.block_dim,
                                           p.info,
                                           &size),
              expected,
              "bsrsv_buffer_size");
        check(p.sv_analysis(p.A, rocsparse_analysis_policy_force, conj_transpose),
              is(rocsparse_status_not_implemented),
              "bsrsv_analysis");
        device_vector<double> x(right_hand_side(p.n())), y(p.n());
        check(p.sv_solve(p.A, x, y, conj_transpose),
              is(rocsparse_status_not_implemented),
              "bsrsv_solve");
    }
    const int32_t nrhs = 2;
    for(const bool on_A : {true, false})
    {
        SCOPED_TRACE(on_A ? "bsrsm trans_A" : "bsrsm trans_X");
        const auto    trans_A = on_A ? conj_transpose : rocsparse_operation_none;
        const auto    trans_X = on_A ? rocsparse_operation_none : conj_transpose;
        block_problem p(handle, test_matrix(), rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        size_t size = 0;
        check(bsrsm_buffer_size(p, trans_A, trans_X, nrhs, &size), expected, "bsrsm_buffer_size");
        ASSERT_NO_FATAL_FAILURE(p.reserve(size));
        p.poison();
        check(rocsparse_dbsrsm_analysis(handle,
                                        p.m.dir,
                                        trans_A,
                                        trans_X,
                                        p.m.mb,
                                        nrhs,
                                        p.m.nnzb(),
                                        p.A,
                                        p.val,
                                        p.row,
                                        p.col,
                                        p.m.block_dim,
                                        p.info,
                                        rocsparse_analysis_policy_force,
                                        rocsparse_solve_policy_auto,
                                        p.buffer),
              expected,
              "bsrsm_analysis");
        const double          one = 1.0;
        const int32_t         ld  = bsrsm_ld(p, trans_X, nrhs);
        device_vector<double> B(right_hand_side(p.n(), nrhs)), X(p.n() * nrhs);
        p.poison();
        check(rocsparse_dbsrsm_solve(handle,
                                     p.m.dir,
                                     trans_A,
                                     trans_X,
                                     p.m.mb,
                                     nrhs,
                                     p.m.nnzb(),
                                     &one,
                                     p.A,
                                     p.val,
                                     p.row,
                                     p.col,
                                     p.m.block_dim,
                                     p.info,
                                     B,
                                     ld,
                                     X,
                                     ld,
                                     rocsparse_solve_policy_auto,
                                     p.buffer),
              expected,
              "bsrsm_solve");
    }
}

// ---------------------------------------------------------------------------
// Sequences that crash or use freed memory
// ---------------------------------------------------------------------------

// Each reuse hit below copies the analysis of another routine and returns before
// allocating the zero pivot of the routine that reuses it, which then faults on it.
TEST_F(PrecondTriangularScenariosBsr, DISABLED_bsrilu0_reusing_a_bsrsv_analysis)
{
    ASSERT_NO_FATAL_FAILURE(check_reuse(handle, routine::bsrsv, routine::bsrilu0));
}

TEST_F(PrecondTriangularScenariosBsr, DISABLED_bsrilu0_reusing_a_bsric0_analysis)
{
    ASSERT_NO_FATAL_FAILURE(check_reuse(handle, routine::bsric0, routine::bsrilu0));
}

TEST_F(PrecondTriangularScenariosBsr, DISABLED_bsric0_reusing_a_bsrsv_analysis)
{
    ASSERT_NO_FATAL_FAILURE(check_reuse(handle, routine::bsrsv, routine::bsric0));
}

// The unit diagonal solve of bsrsv resets the zero pivot, which the reused analysis
// never allocated.
TEST_F(PrecondTriangularScenariosBsr, DISABLED_bsrsv_unit_solve_after_a_reused_analysis)
{
    block_problem p(handle, test_matrix(), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    ASSERT_EQ(p.ilu0_analysis(rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_EQ(p.sv_analysis(p.L, rocsparse_analysis_policy_reuse), rocsparse_status_success);
    const auto            b = right_hand_side(p.n());
    device_vector<double> x(b), y(sentinel(p.n()));
    ASSERT_EQ(p.sv_solve(p.L, x, y), rocsparse_status_success);
    expect_close(to_host(y),
                 host_lower_solve(p.scalar, b, 1, rocsparse_diag_type_unit),
                 tolerance<double>(max_row_length(p.scalar)));
    check(p.bsrsv_pivot(), is(no_pivot), "bsrsv_zero_pivot after the solve with L");
}

// A zero on the diagonal of a diagonal block is recorded into the zero pivot of bsrsv,
// which the reused analysis never allocated.
TEST_F(PrecondTriangularScenariosBsr, DISABLED_bsrsv_numerical_zero_after_a_reused_analysis)
{
    const int32_t r = 1;
    block_problem p(
        handle, with_zero_in_diagonal_block(test_matrix(), r, 0), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    ASSERT_EQ(p.sm_analysis(p.A, rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_EQ(p.sv_analysis(p.A, rocsparse_analysis_policy_reuse), rocsparse_status_success);
    const auto            b = right_hand_side(p.n());
    device_vector<double> x(b), y(p.n());
    ASSERT_EQ(p.sv_solve(p.A, x, y), rocsparse_status_success);
    check(p.bsrsv_pivot(), is(r), "bsrsv_zero_pivot after the solve");
}

// bsrsm dereferences the analysis it does not have, without an analysis and after
// bsrsm_clear (bsrsv returns rocsparse_status_invalid_pointer in that case).
TEST_F(PrecondTriangularScenariosBsr, DISABLED_bsrsm_without_an_analysis)
{
    block_problem p(handle, test_matrix(), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    const auto            b = right_hand_side(p.n(), block_problem::nrhs);
    device_vector<double> B(b), X(b.size());
    check(p.sm_solve(p.A, B, X),
          is(rocsparse_status_invalid_pointer),
          "bsrsm_solve without an analysis");
}

TEST_F(PrecondTriangularScenariosBsr, DISABLED_bsrsm_after_bsrsm_clear)
{
    block_problem p(handle, test_matrix(), rocsparse_index_base_zero);
    ASSERT_NO_FATAL_FAILURE(p.init());
    ASSERT_EQ(p.sm_analysis(p.A, rocsparse_analysis_policy_force), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(check_compute(p, routine::bsrsm));
    ASSERT_EQ(rocsparse_bsrsm_clear(handle, p.info), rocsparse_status_success);
    const auto            b = right_hand_side(p.n(), block_problem::nrhs);
    device_vector<double> B(b), X(b.size());
    check(p.sm_solve(p.A, B, X),
          is(rocsparse_status_invalid_pointer),
          "bsrsm_solve after bsrsm_clear");
}

namespace
{
    // bsrsv reuses the analysis of `owner`; a forced analysis of `deleter` (the owner or
    // bsrsv) then deletes it while the other routine still uses it, and the destructor
    // of the mat_info deletes it again.
    void check_forced_analysis_of_shared_data(rocsparse_handle handle,
                                              routine          owner,
                                              routine          deleter)
    {
        block_problem p(handle, test_matrix(), rocsparse_index_base_zero);
        ASSERT_NO_FATAL_FAILURE(p.init());
        ASSERT_EQ(analyse(p, owner, rocsparse_analysis_policy_force), rocsparse_status_success);
        ASSERT_EQ(p.sv_analysis(p.A, rocsparse_analysis_policy_reuse), rocsparse_status_success);
        ASSERT_EQ(analyse(p, deleter, rocsparse_analysis_policy_force), rocsparse_status_success);
        ASSERT_NO_FATAL_FAILURE(p.set_values());
        ASSERT_NO_FATAL_FAILURE(
            check_compute(p, deleter == routine::bsrsv ? owner : routine::bsrsv));
    }
}

TEST_F(PrecondTriangularScenariosBsr, DISABLED_forced_bsrsm_analysis_after_bsrsv_reused_it)
{
    check_forced_analysis_of_shared_data(handle, routine::bsrsm, routine::bsrsm);
}

TEST_F(PrecondTriangularScenariosBsr, DISABLED_forced_bsrsv_analysis_after_reusing_bsrsm)
{
    check_forced_analysis_of_shared_data(handle, routine::bsrsm, routine::bsrsv);
}

TEST_F(PrecondTriangularScenariosBsr, DISABLED_forced_bsrilu0_analysis_after_bsrsv_reused_it)
{
    check_forced_analysis_of_shared_data(handle, routine::bsrilu0, routine::bsrilu0);
}

TEST_F(PrecondTriangularScenariosBsr, DISABLED_forced_bsrsv_analysis_after_reusing_bsrilu0)
{
    check_forced_analysis_of_shared_data(handle, routine::bsrilu0, routine::bsrsv);
}

TEST_F(PrecondTriangularScenariosBsr, DISABLED_forced_bsric0_analysis_after_bsrsv_reused_it)
{
    check_forced_analysis_of_shared_data(handle, routine::bsric0, routine::bsric0);
}
