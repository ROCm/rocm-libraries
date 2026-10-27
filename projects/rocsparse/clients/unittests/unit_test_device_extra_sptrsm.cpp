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
// rocsparse_sptrsm: numeric solves of op(A) * Y = alpha * op(X) for CSR, CSC and
// COO matrices (lower / upper, unit / non-unit diagonal, A and X transposed or
// not, row / column ordered dense matrices, host / device alpha, f64 alpha with
// f32 data, strided batch)
// compared with a host substitution on a diagonally dominant matrix. Also the
// singularity outputs of the descriptor and the stage ordering errors.
//
#include "unit_test_extra_common.hpp"

#include <map>

using namespace extra_ut;

#ifdef ROCSPARSE_WITH_TRSM_REFACTORING

namespace
{
    struct SptrsmDescrH
    {
        rocsparse_sptrsm_descr d = nullptr;
        SptrsmDescrH()
        {
            (void)rocsparse_create_sptrsm_descr(&d);
        }
        SptrsmDescrH(const SptrsmDescrH&) = delete;
        SptrsmDescrH& operator=(const SptrsmDescrH&) = delete;
        ~SptrsmDescrH()
        {
            if(d)
                (void)rocsparse_destroy_sptrsm_descr(d);
        }
    };

    struct SptrsmCase
    {
        SFmt                      fmt       = SFmt::csr;
        rocsparse_fill_mode       fill      = rocsparse_fill_mode_lower;
        rocsparse_diag_type       diag      = rocsparse_diag_type_non_unit;
        rocsparse_operation       opA       = rocsparse_operation_none;
        rocsparse_operation       opX       = rocsparse_operation_none;
        rocsparse_order           order_X   = rocsparse_order_column;
        rocsparse_order           order_Y   = rocsparse_order_column;
        rocsparse_analysis_policy policy    = rocsparse_analysis_policy_reuse;
        int64_t                   m         = 23;
        int64_t                   nrhs      = 5;
        int64_t                   batch     = 1; // Y and X batch count (A is not batched)
        bool                      device    = false;
        bool                      garbage   = true; // values in the ignored triangle
        cplx                      alpha     = cplx(2.0, 0.0);
        bool                      f64_alpha = false; // alpha given as f64, converted to T
        // Singular matrices: row whose diagonal is zero (value) or missing.
        int64_t singular_row = -1;
        bool    missing_diag = false;
        // Diagonal solve mode (D * Y = alpha * op(X)), with |d| when `absolute`.
        bool diagonal = false;
        bool absolute = false;
        // Documented as unsupported: not_implemented is accepted.
        bool unsupported = false;
    };

    std::string describe(const SptrsmCase& c)
    {
        return "fmt=" + std::to_string(static_cast<int>(c.fmt))
               + " fill=" + std::to_string(static_cast<int>(c.fill))
               + " diag=" + std::to_string(static_cast<int>(c.diag))
               + " opA=" + std::to_string(static_cast<int>(c.opA))
               + " opX=" + std::to_string(static_cast<int>(c.opX))
               + " oX=" + std::to_string(static_cast<int>(c.order_X))
               + " oY=" + std::to_string(static_cast<int>(c.order_Y)) + " policy="
               + std::to_string(static_cast<int>(c.policy)) + " m=" + std::to_string(c.m)
               + " nrhs=" + std::to_string(c.nrhs) + " batch=" + std::to_string(c.batch)
               + " dev=" + std::to_string(c.device) + " f64_alpha=" + std::to_string(c.f64_alpha)
               + " diagonal=" + std::to_string(c.diagonal)
               + " absolute=" + std::to_string(c.absolute);
    }

    // Diagonally dominant triangular matrix. Off-diagonal entries are small
    // multiples of 1/8 (at most three per row, so that the sum of their moduli
    // stays below the smallest diagonal modulus). `garbage` adds finite entries
    // in the opposite triangle that a triangular solve must not read.
    template <typename T>
    Csr triangular(const SptrsmCase& c, Rng& rng)
    {
        const bool lower = (c.fill == rocsparse_fill_mode_lower);
        const bool unit  = (c.diag == rocsparse_diag_type_unit);
        Csr        A;
        A.m = A.n = c.m;
        A.ptr.assign(static_cast<size_t>(c.m) + 1, 0);
        for(int64_t i = 0; i < c.m; ++i)
        {
            std::map<int64_t, cplx> row;
            const int64_t           lo = lower ? 0 : i + 1;
            const int64_t           hi = lower ? i - 1 : c.m - 1;
            if(hi >= lo)
            {
                const int64_t cnt = rng.range(0, 3);
                for(int64_t q = 0; q < cnt; ++q)
                {
                    const double re = 0.125 * rng.range(-3, 3);
                    const double im = is_complex_t<T>::value ? 0.125 * rng.range(-1, 1) : 0.0;
                    row[rng.range(lo, hi)] = round_to<T>(cplx(re, im));
                }
            }
            if(c.garbage)
            {
                const int64_t glo = lower ? i + 1 : 0;
                const int64_t ghi = lower ? c.m - 1 : i - 1;
                if(ghi >= glo)
                    row[rng.range(glo, ghi)] = round_to<T>(cplx(37.0, 0.0));
            }
            if(!unit && !(c.missing_diag && i == c.singular_row))
            {
                cplx d = round_to<T>(rand_diag<T>(rng));
                if(i == c.singular_row)
                    d = cplx(0.0, 0.0);
                row[i] = d;
            }
            for(const auto& e : row)
            {
                A.col.push_back(e.first);
                A.val.push_back(e.second);
            }
            A.ptr[static_cast<size_t>(i) + 1] = static_cast<int64_t>(A.col.size());
        }
        return A;
    }

    // The matrix the library is supposed to solve with: the selected triangle,
    // and a unit diagonal if requested.
    Dense reference_matrix(const SptrsmCase& c, const Csr& A)
    {
        const bool lower = (c.fill == rocsparse_fill_mode_lower);
        Dense      D(c.m, c.m);
        for(int64_t i = 0; i < c.m; ++i)
        {
            for(int64_t p = A.ptr[static_cast<size_t>(i)]; p < A.ptr[static_cast<size_t>(i) + 1];
                ++p)
            {
                const int64_t j = A.col[static_cast<size_t>(p)];
                if((lower && j <= i) || (!lower && j >= i))
                    D.at(i, j) = A.val[static_cast<size_t>(p)];
            }
            if(c.diag == rocsparse_diag_type_unit)
                D.at(i, i) = cplx(1.0, 0.0);
        }
        return D;
    }

    template <typename T, typename I, typename J>
    void make_matrix(DevSparse<T, I, J>& d, const SptrsmCase& c, const Csr& A)
    {
        make_sparse<T, I, J>(d, c.fmt, A);
        if(::testing::Test::HasFatalFailure())
            return;
        ASSERT_EQ(rocsparse_spmat_set_attribute(
                      d.h.p, rocsparse_spmat_fill_mode, &c.fill, sizeof(c.fill)),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spmat_set_attribute(
                      d.h.p, rocsparse_spmat_diag_type, &c.diag, sizeof(c.diag)),
                  rocsparse_status_success);
    }

    rocsparse_status set_input(rocsparse_handle       handle,
                               rocsparse_sptrsm_descr descr,
                               rocsparse_sptrsm_input which,
                               const void*            data,
                               size_t                 n)
    {
        return rocsparse_sptrsm_set_input(handle, descr, which, data, n, nullptr);
    }

    // Everything needed to run the stages of one solve.
    template <typename T, typename I, typename J = I>
    struct SptrsmProblem
    {
        DevSparse<T, I, J> A;
        DnmatH             X, Y;
        SptrsmDescrH       descr;
        Scalar<T>          alpha;
        Scalar<double>     alpha64;
        DevBuf<T>          dX, dY;
        std::vector<Dense> Xl; // logical right-hand sides, one per batch
        std::vector<Dense> Y0;
        Dense              Aref;
        int64_t            ldx = 0, ldy = 0, bsx = 0, bsy = 0;
        SptrsmProblem(const SptrsmCase& c)
            : alpha(c.alpha, c.device)
            , alpha64(c.alpha, c.device)
        {
        }
    };

    template <typename T, typename I, typename J>
    void setup(SptrsmProblem<T, I, J>& pr, rocsparse_handle handle, const SptrsmCase& c, Rng& rng)
    {
        const rocsparse_datatype dt = rocsparse_ut::dt_of<T>();
        const Csr                A  = triangular<T>(c, rng);
        pr.Aref                     = reference_matrix(c, A);
        make_matrix<T, I, J>(pr.A, c, A);
        if(::testing::Test::HasFatalFailure())
            return;

        const bool    xt = (c.opX != rocsparse_operation_none);
        const int64_t xr = xt ? c.nrhs : c.m; // stored rows of X
        const int64_t xc = xt ? c.m : c.nrhs;
        pr.ldx = std::max<int64_t>(1, (c.order_X == rocsparse_order_column) ? xr : xc) + 2;
        pr.ldy = std::max<int64_t>(1, (c.order_Y == rocsparse_order_column) ? c.m : c.nrhs) + 3;
        const int64_t xlines = (c.order_X == rocsparse_order_column) ? xc : xr;
        const int64_t ylines = (c.order_Y == rocsparse_order_column) ? c.nrhs : c.m;
        pr.bsx               = pr.ldx * xlines;
        pr.bsy               = pr.ldy * ylines;

        std::vector<T> hx, hy;
        for(int64_t b = 0; b < c.batch; ++b)
        {
            pr.Xl.push_back(random_dense<T>(c.m, c.nrhs, rng));
            const Dense Xs = apply_op(pr.Xl.back(), c.opX);
            const auto  sx = store_dense<T>(Xs, c.order_X, pr.ldx);
            hx.insert(hx.end(), sx.begin(), sx.end());
            pr.Y0.push_back(Dense(c.m, c.nrhs));
            const auto sy = store_dense<T>(pr.Y0.back(), c.order_Y, pr.ldy);
            hy.insert(hy.end(), sy.begin(), sy.end());
        }
        pr.dX = up_raw<T>(hx);
        pr.dY = up_raw<T>(hy);
        ASSERT_TRUE(pr.dX.ptr && pr.dY.ptr);

        ASSERT_EQ(rocsparse_create_dnmat_descr(&pr.X.p, xr, xc, pr.ldx, pr.dX.ptr, dt, c.order_X),
                  rocsparse_status_success);
        ASSERT_EQ(
            rocsparse_create_dnmat_descr(&pr.Y.p, c.m, c.nrhs, pr.ldy, pr.dY.ptr, dt, c.order_Y),
            rocsparse_status_success);
        if(c.batch > 1)
        {
            ASSERT_EQ(rocsparse_dnmat_set_strided_batch(pr.X.p, c.batch, pr.bsx),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_dnmat_set_strided_batch(pr.Y.p, c.batch, pr.bsy),
                      rocsparse_status_success);
        }

        ASSERT_EQ(rocsparse_set_pointer_mode(handle, mode_of(c.device)), rocsparse_status_success);
        const rocsparse_sptrsm_alg alg = rocsparse_sptrsm_alg_default;
        ASSERT_EQ(set_input(handle, pr.descr.d, rocsparse_sptrsm_input_alg, &alg, sizeof(alg)),
                  rocsparse_status_success);
        ASSERT_EQ(
            set_input(
                handle, pr.descr.d, rocsparse_sptrsm_input_operation_A, &c.opA, sizeof(c.opA)),
            rocsparse_status_success);
        ASSERT_EQ(
            set_input(
                handle, pr.descr.d, rocsparse_sptrsm_input_operation_X, &c.opX, sizeof(c.opX)),
            rocsparse_status_success);
        const rocsparse_datatype sdt = c.f64_alpha ? rocsparse_datatype_f64_r : dt;
        ASSERT_EQ(
            set_input(
                handle, pr.descr.d, rocsparse_sptrsm_input_scalar_datatype, &sdt, sizeof(sdt)),
            rocsparse_status_success);
        ASSERT_EQ(
            set_input(handle, pr.descr.d, rocsparse_sptrsm_input_compute_datatype, &dt, sizeof(dt)),
            rocsparse_status_success);
        ASSERT_EQ(set_input(handle,
                            pr.descr.d,
                            rocsparse_sptrsm_input_analysis_policy,
                            &c.policy,
                            sizeof(c.policy)),
                  rocsparse_status_success);
        const void* alpha_ptr = c.f64_alpha ? pr.alpha64.get() : pr.alpha.get();
        ASSERT_EQ(set_input(handle,
                            pr.descr.d,
                            rocsparse_sptrsm_input_scalar_alpha,
                            alpha_ptr,
                            sizeof(const void*)),
                  rocsparse_status_success);
#if defined(ROCSPARSE_WITH_DIAGONAL_SOLVE)
        if(c.diagonal)
        {
            const rocsparse_solve_mode        mode     = rocsparse_solve_mode_diagonal;
            const rocsparse_diagonal_modifier modifier = c.absolute
                                                             ? rocsparse_diagonal_modifier_absolute
                                                             : rocsparse_diagonal_modifier_none;
            ASSERT_EQ(
                set_input(
                    handle, pr.descr.d, rocsparse_sptrsm_input_solve_mode, &mode, sizeof(mode)),
                rocsparse_status_success);
            ASSERT_EQ(set_input(handle,
                                pr.descr.d,
                                rocsparse_sptrsm_input_diagonal_modifier,
                                &modifier,
                                sizeof(modifier)),
                      rocsparse_status_success);
        }
#endif
    }

    // buffer size + stage execution. Returns the status of the stage.
    template <typename T, typename I, typename J>
    rocsparse_status run_stage(rocsparse_handle        handle,
                               SptrsmProblem<T, I, J>& pr,
                               rocsparse_sptrsm_stage  stage,
                               DevBuf<char>&           buffer)
    {
        size_t bytes = 0;
        auto   st    = rocsparse_sptrsm_buffer_size(
            handle, pr.descr.d, pr.A.h.p, pr.X.p, pr.Y.p, stage, &bytes, nullptr);
        if(st != rocsparse_status_success)
            return st;
        buffer = DevBuf<char>(bytes);
        return rocsparse_sptrsm(handle,
                                pr.descr.d,
                                pr.A.h.p,
                                pr.X.p,
                                pr.Y.p,
                                stage,
                                bytes,
                                bytes ? buffer.ptr : nullptr,
                                nullptr);
    }

    template <typename T, typename I, typename J = I>
    void run_sptrsm(rocsparse_handle handle, const SptrsmCase& c)
    {
        SCOPED_TRACE(describe(c));
        Rng rng(static_cast<uint64_t>(c.m * 31 + c.nrhs * 7 + c.batch) + 3);

        SptrsmProblem<T, I, J> pr(c);
        setup<T, I, J>(pr, handle, c, rng);
        if(::testing::Test::HasFatalFailure())
            return;

        DevBuf<char> buf_a, buf_c;
        for(const auto stage : {rocsparse_sptrsm_stage_analysis, rocsparse_sptrsm_stage_compute})
        {
            const rocsparse_status st = run_stage<T, I, J>(
                handle, pr, stage, stage == rocsparse_sptrsm_stage_analysis ? buf_a : buf_c);
            if(c.unsupported && st == rocsparse_status_not_implemented)
                return;
            ASSERT_EQ(st, rocsparse_status_success);
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const bool lower
            = (c.fill == rocsparse_fill_mode_lower) == (c.opA == rocsparse_operation_none);
        Dense Aop = apply_op(pr.Aref, c.opA);
        if(c.diagonal)
        {
            Dense D(c.m, c.m);
            for(int64_t i = 0; i < c.m; ++i)
                D.at(i, i) = c.absolute ? cplx(std::abs(Aop.at(i, i)), 0.0) : Aop.at(i, i);
            Aop = D;
        }
        const auto hy = pr.dY.host();
        // A triangular solve is backward stable in any summation order: the computed Y
        // solves (op(A) + E) Y = alpha * op(X) with |E| <= gamma_m |op(A)|, where gamma_m
        // is about m * u (u the unit roundoff, a small multiple of it for complex
        // arithmetic), so |Y - Y_exact| <= gamma_m |op(A)^-1| |op(A)| |Y|. With m <= 40,
        // gamma_m is about 2.4e-6 in single and 4.4e-15 in double precision.
        // For a non-unit diagonal and opA = none, each row has at most three off-diagonal
        // entries of modulus <= sqrt(0.375^2 + 0.125^2) ~ 0.395 and a diagonal of modulus
        // >= 2.5, so the off-diagonal row sum is at most 0.48 of the diagonal and the
        // growth |op(A)^-1| |op(A)| is at most (1 + 0.48) / (1 - 0.48) < 3. The unit
        // diagonal cases and the transposed ones (whose rows are columns of A, which are
        // not limited to three entries) are not diagonally dominant by rows and have no
        // such a priori bound; only m <= 40 and the small off-diagonal entries limit
        // their growth. The tolerances leave about two (single) and five (double) orders
        // of magnitude above gamma_m for that growth and the alpha scaling, and the
        // measured errors are far below them. The diagonal solve mode divides entry by
        // entry. float and float complex (4 and 8 bytes) are single precision, the rest
        // double.
        const double tol
            = (sizeof(T) == 4 || sizeof(T) == 8 && is_complex_t<T>::value) ? 2e-4 : 1e-9;
        for(int64_t b = 0; b < c.batch; ++b)
        {
            Dense rhs = pr.Xl[static_cast<size_t>(b)];
            for(auto& v : rhs.v)
                v *= to_c<T>(pr.alpha.h);
            const Dense want = solve_triangular(Aop, lower, rhs);
            for(const cplx& v : want.v)
                ASSERT_TRUE(std::isfinite(v.real()) && std::isfinite(v.imag()));
            const std::vector<T> yb(hy.begin() + b * pr.bsy, hy.begin() + (b + 1) * pr.bsy);
            expect_dense_equal<T>(yb, want, c.order_Y, pr.ldy, "Y", cplx(-77.0, 0.0), tol);
        }

        // Input X is not modified.
        const auto hx = pr.dX.host();
        for(int64_t b = 0; b < c.batch; ++b)
        {
            const std::vector<T> xb(hx.begin() + b * pr.bsx, hx.begin() + (b + 1) * pr.bsx);
            expect_dense_equal<T>(
                xb, apply_op(pr.Xl[static_cast<size_t>(b)], c.opX), c.order_X, pr.ldx, "X");
        }

        // A non-singular system reports no singularity. (The outputs hold one
        // value per batch entry, so they are only queried without batching.)
        if(c.batch != 1)
            return;
        ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
                  rocsparse_status_success);
        rocsparse_singularity sing = rocsparse_singularity_symbolic;
        ASSERT_EQ(rocsparse_sptrsm_get_output(handle,
                                              pr.descr.d,
                                              rocsparse_sptrsm_output_singularity,
                                              &sing,
                                              sizeof(sing),
                                              nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(sing, rocsparse_singularity_none);
        int64_t pivot = 0;
        EXPECT_EQ(rocsparse_sptrsm_get_output(handle,
                                              pr.descr.d,
                                              rocsparse_sptrsm_output_zero_pivot_position,
                                              &pivot,
                                              sizeof(pivot),
                                              nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(pivot, -1);
    }

    struct ExtraSptrsm : public rocsparse_ut::HandleTest
    {
    };

    const rocsparse_order all_orders[] = {rocsparse_order_column, rocsparse_order_row};
}

namespace
{
    void solve_grid(rocsparse_handle handle, SFmt f, bool mixed_orders)
    {
        for(rocsparse_fill_mode fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
            for(rocsparse_diag_type diag : {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit})
                for(rocsparse_operation opA :
                    {rocsparse_operation_none, rocsparse_operation_transpose})
                    for(rocsparse_operation opX :
                        {rocsparse_operation_none, rocsparse_operation_transpose})
                        for(rocsparse_order oX : all_orders)
                            for(rocsparse_order oY : all_orders)
                            {
                                if(!mixed_orders && oX != oY)
                                    continue;
                                SptrsmCase c;
                                c.fmt     = f;
                                c.fill    = fill;
                                c.diag    = diag;
                                c.opA     = opA;
                                c.opX     = opX;
                                c.order_X = oX;
                                c.order_Y = oY;
                                run_sptrsm<float, int32_t>(handle, c);
                                if(::testing::Test::HasFatalFailure())
                                    return;
                            }
    }
}

TEST_F(ExtraSptrsm, solve_grid_csr_float)
{
    solve_grid(handle, SFmt::csr, true);
}

TEST_F(ExtraSptrsm, solve_types_policies_and_scalars)
{
    for(SFmt f : {SFmt::csr, SFmt::csc, SFmt::coo})
        for(rocsparse_analysis_policy policy :
            {rocsparse_analysis_policy_reuse, rocsparse_analysis_policy_force})
            for(bool device : {false, true})
            {
                SptrsmCase c;
                c.fmt    = f;
                c.policy = policy;
                c.device = device;
                c.fill   = (f == SFmt::coo) ? rocsparse_fill_mode_upper : rocsparse_fill_mode_lower;
                run_sptrsm<double, int32_t>(handle, c);
                if(HasFatalFailure())
                    return;
                run_sptrsm<double, int64_t>(handle, c);
                if(HasFatalFailure())
                    return;
                c.f64_alpha = true;
                run_sptrsm<float, int32_t>(handle, c);
                if(HasFatalFailure())
                    return;
                c.f64_alpha = false;
                c.alpha     = cplx(0.5, -1.0);
                run_sptrsm<rocsparse_float_complex, int32_t>(handle, c);
                if(HasFatalFailure())
                    return;
                run_sptrsm<rocsparse_double_complex, int64_t>(handle, c);
                if(HasFatalFailure())
                    return;
            }
}

TEST_F(ExtraSptrsm, right_hand_side_counts)
{
    for(int64_t nrhs : {1, 2, 33, 70})
        for(rocsparse_order oY : all_orders)
        {
            SptrsmCase c;
            c.nrhs    = nrhs;
            c.order_Y = oY;
            c.m       = 40;
            run_sptrsm<float, int32_t>(handle, c);
            if(HasFatalFailure())
                return;
        }
}

// Strided batches of X and Y with one and several right-hand sides, every data
// type and both orders. A conjugate transposed X is documented as unsupported.
TEST_F(ExtraSptrsm, strided_batch_types_and_orders)
{
    for(int64_t nrhs : {1, 3})
        for(rocsparse_order o : all_orders)
        {
            SptrsmCase c;
            c.batch   = 3;
            c.nrhs    = nrhs;
            c.m       = 17;
            c.order_X = o;
            c.order_Y = o;
            run_sptrsm<float, int32_t>(handle, c);
            if(HasFatalFailure())
                return;
            c.opA = rocsparse_operation_transpose;
            run_sptrsm<double, int64_t>(handle, c);
            if(HasFatalFailure())
                return;
            c.opA   = rocsparse_operation_none;
            c.alpha = cplx(0.5, -1.0);
            run_sptrsm<rocsparse_float_complex, int32_t>(handle, c);
            if(HasFatalFailure())
                return;
            c.opX = rocsparse_operation_transpose;
            run_sptrsm<rocsparse_double_complex, int32_t>(handle, c);
            if(HasFatalFailure())
                return;
            c.opX         = rocsparse_operation_conjugate_transpose;
            c.unsupported = true;
            run_sptrsm<rocsparse_float_complex, int32_t>(handle, c);
            if(HasFatalFailure())
                return;
            run_sptrsm<rocsparse_double_complex, int64_t>(handle, c);
            if(HasFatalFailure())
                return;
        }
}

#if defined(ROCSPARSE_WITH_DIAGONAL_SOLVE)
// Diagonal solve mode for CSR and CSC matrices, every data type, both index
// types, with and without the absolute value of the diagonal.
TEST_F(ExtraSptrsm, diagonal_solve_types_and_formats)
{
    for(SFmt f : {SFmt::csr, SFmt::csc})
        for(rocsparse_operation opA : {rocsparse_operation_none, rocsparse_operation_transpose})
        {
            SptrsmCase c;
            c.fmt      = f;
            c.opA      = opA;
            c.diagonal = true;
            c.absolute = true;
            run_sptrsm<double, int64_t>(handle, c);
            if(HasFatalFailure())
                return;
            c.batch   = 2;
            c.order_Y = rocsparse_order_row;
            run_sptrsm<float, int32_t>(handle, c);
            if(HasFatalFailure())
                return;
            c.absolute = false;
            c.batch    = 1;
            c.alpha    = cplx(0.5, -1.0);
            run_sptrsm<rocsparse_float_complex, int32_t>(handle, c);
            if(HasFatalFailure())
                return;
            c.opX = rocsparse_operation_transpose;
            run_sptrsm<rocsparse_double_complex, int64_t>(handle, c);
            if(HasFatalFailure())
                return;
            // 64-bit offsets with 32-bit indices.
            c.opX   = rocsparse_operation_none;
            c.alpha = cplx(-1.5, 0.0);
            run_sptrsm<double, int64_t, int32_t>(handle, c);
            if(HasFatalFailure())
                return;
        }
}
#else
TEST_F(ExtraSptrsm, diagonal_solve_types_and_formats)
{
    GTEST_SKIP() << "built without ROCSPARSE_WITH_DIAGONAL_SOLVE";
}
#endif

// ---------------------------------------------------------------------------
// Singular systems: a zero or a missing diagonal entry is reported by the
// descriptor outputs.
// ---------------------------------------------------------------------------
namespace
{
    void expect_singular(rocsparse_handle handle, const SptrsmCase& c, rocsparse_singularity kind)
    {
        SCOPED_TRACE(describe(c));
        Rng                           rng(99);
        SptrsmProblem<float, int32_t> pr(c);
        setup<float, int32_t>(pr, handle, c, rng);
        if(::testing::Test::HasFatalFailure())
            return;
        DevBuf<char> buf_a, buf_c;
        ASSERT_EQ((run_stage<float, int32_t>(handle, pr, rocsparse_sptrsm_stage_analysis, buf_a)),
                  rocsparse_status_success);
        ASSERT_EQ((run_stage<float, int32_t>(handle, pr, rocsparse_sptrsm_stage_compute, buf_c)),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        // Host pointers.
        ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
                  rocsparse_status_success);
        rocsparse_singularity sing = rocsparse_singularity_none;
        ASSERT_EQ(rocsparse_sptrsm_get_output(handle,
                                              pr.descr.d,
                                              rocsparse_sptrsm_output_singularity,
                                              &sing,
                                              sizeof(sing),
                                              nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(sing, kind);
        int64_t position = -2;
        ASSERT_EQ(rocsparse_sptrsm_get_output(handle,
                                              pr.descr.d,
                                              rocsparse_sptrsm_output_singularity_position,
                                              &position,
                                              sizeof(position),
                                              nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(position, c.singular_row);

        // Device pointers.
        ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
                  rocsparse_status_success);
        DevBuf<rocsparse_singularity> dsing(1);
        DevBuf<int64_t>               dpos(1);
        ASSERT_TRUE(dsing.ptr && dpos.ptr);
        ASSERT_EQ(rocsparse_sptrsm_get_output(handle,
                                              pr.descr.d,
                                              rocsparse_sptrsm_output_singularity,
                                              dsing.ptr,
                                              sizeof(rocsparse_singularity),
                                              nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_sptrsm_get_output(handle,
                                              pr.descr.d,
                                              rocsparse_sptrsm_output_singularity_position,
                                              dpos.ptr,
                                              sizeof(int64_t),
                                              nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(dsing.host()[0], kind);
        EXPECT_EQ(dpos.host()[0], c.singular_row);
        ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
                  rocsparse_status_success);

        // The legacy zero pivot query reports a zero pivot (as the status or as
        // a position) and agrees with the singularity position.
        int64_t zp = -2;
        auto    st = rocsparse_sptrsm_get_output(handle,
                                              pr.descr.d,
                                              rocsparse_sptrsm_output_zero_pivot_position,
                                              &zp,
                                              sizeof(zp),
                                              nullptr);
        EXPECT_TRUE(st == rocsparse_status_zero_pivot || st == rocsparse_status_success);
        EXPECT_EQ(zp, c.singular_row);
    }
}

TEST_F(ExtraSptrsm, singular_missing_diagonal)
{
    for(SFmt f : {SFmt::csr, SFmt::coo})
    {
        SptrsmCase c;
        c.fmt          = f;
        c.singular_row = 4;
        c.missing_diag = true;
        expect_singular(handle, c, rocsparse_singularity_symbolic);
        if(HasFatalFailure())
            return;
    }
}

TEST_F(ExtraSptrsm, stage_ordering)
{
    SptrsmCase                    c;
    Rng                           rng(5);
    SptrsmProblem<float, int32_t> pr(c);
    setup<float, int32_t>(pr, handle, c, rng);
    if(HasFatalFailure())
        return;
    DevBuf<char> buf;
    // The compute stage needs a previous analysis stage.
    EXPECT_NE((run_stage<float, int32_t>(handle, pr, rocsparse_sptrsm_stage_compute, buf)),
              rocsparse_status_success);
    ASSERT_EQ((run_stage<float, int32_t>(handle, pr, rocsparse_sptrsm_stage_analysis, buf)),
              rocsparse_status_success);
    // The analysis stage is executed once.
    DevBuf<char> buf2;
    EXPECT_NE((run_stage<float, int32_t>(handle, pr, rocsparse_sptrsm_stage_analysis, buf2)),
              rocsparse_status_success);
    ASSERT_EQ((run_stage<float, int32_t>(handle, pr, rocsparse_sptrsm_stage_compute, buf2)),
              rocsparse_status_success);
    // ... and cannot follow a compute stage.
    DevBuf<char> buf3;
    EXPECT_NE((run_stage<float, int32_t>(handle, pr, rocsparse_sptrsm_stage_analysis, buf3)),
              rocsparse_status_success);
}

// Formats without a triangular solve are rejected.
TEST_F(ExtraSptrsm, unsupported_formats)
{
    const rocsparse_datatype dt = rocsparse_datatype_f32_r;
    DevBuf<int32_t>          ind(std::vector<int32_t>(8, 0));
    DevBuf<float>            val(std::vector<float>(16, 1.0f));
    ASSERT_TRUE(ind.ptr && val.ptr);

    SpmatH A;
    ASSERT_EQ(rocsparse_create_ell_descr(&A.p,
                                         4,
                                         4,
                                         ind.ptr,
                                         val.ptr,
                                         2,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         dt),
              rocsparse_status_success);
    DevBuf<float> dx(std::vector<float>(16, 1.0f));
    DevBuf<float> dy(std::vector<float>(16, 0.0f));
    DnmatH        X, Y;
    ASSERT_EQ(rocsparse_create_dnmat_descr(&X.p, 4, 4, 4, dx.ptr, dt, rocsparse_order_column),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnmat_descr(&Y.p, 4, 4, 4, dy.ptr, dt, rocsparse_order_column),
              rocsparse_status_success);
    SptrsmDescrH descr;
    const float  alpha = 1.0f;
    ASSERT_EQ(
        rocsparse_sptrsm_set_input(
            handle, descr.d, rocsparse_sptrsm_input_scalar_datatype, &dt, sizeof(dt), nullptr),
        rocsparse_status_success);
    ASSERT_EQ(
        rocsparse_sptrsm_set_input(
            handle, descr.d, rocsparse_sptrsm_input_compute_datatype, &dt, sizeof(dt), nullptr),
        rocsparse_status_success);
    ASSERT_EQ(rocsparse_sptrsm_set_input(handle,
                                         descr.d,
                                         rocsparse_sptrsm_input_scalar_alpha,
                                         &alpha,
                                         sizeof(const void*),
                                         nullptr),
              rocsparse_status_success);

    size_t bytes = 0;
    EXPECT_NE(rocsparse_sptrsm_buffer_size(
                  handle, descr.d, A.p, X.p, Y.p, rocsparse_sptrsm_stage_analysis, &bytes, nullptr),
              rocsparse_status_success);
    EXPECT_NE(
        rocsparse_sptrsm(
            handle, descr.d, A.p, X.p, Y.p, rocsparse_sptrsm_stage_analysis, 0, nullptr, nullptr),
        rocsparse_status_success);
}

#endif // ROCSPARSE_WITH_TRSM_REFACTORING
