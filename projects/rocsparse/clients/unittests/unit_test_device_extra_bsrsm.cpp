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
// rocsparse_?bsrsm_*: block triangular solves op(A) * X = alpha * op(B) for
// every block dimension class of the solve kernels, both block directions,
// lower / upper, unit / non-unit diagonal, A and X transposed or not, host and
// device scalars, plus the zero pivot query. The matrices are diagonally
// dominant; the reference is a scalar substitution on the expanded matrix and
// the whole X, including its padding, is compared.
//
#include "unit_test_extra_common.hpp"

#include <set>
#include <type_traits>

using namespace extra_ut;

namespace
{
    struct BsrsmCase
    {
        int64_t             bd     = 4;
        int64_t             mb     = 5;
        int64_t             nrhs   = 4;
        rocsparse_direction dir    = rocsparse_direction_row;
        rocsparse_fill_mode fill   = rocsparse_fill_mode_lower;
        rocsparse_diag_type diag   = rocsparse_diag_type_non_unit;
        rocsparse_operation opA    = rocsparse_operation_none;
        rocsparse_operation opX    = rocsparse_operation_none;
        bool                device = false;
        cplx                alpha  = cplx(2.0, 0.0);
        // Block row whose diagonal block starts with a zero pivot, -1 for none.
        int64_t singular_block = -1;
    };

    std::string describe(const BsrsmCase& c)
    {
        return "bd=" + std::to_string(c.bd) + " mb=" + std::to_string(c.mb) + " nrhs="
               + std::to_string(c.nrhs) + " dir=" + std::to_string(static_cast<int>(c.dir))
               + " fill=" + std::to_string(static_cast<int>(c.fill))
               + " diag=" + std::to_string(static_cast<int>(c.diag))
               + " opA=" + std::to_string(static_cast<int>(c.opA)) + " opX="
               + std::to_string(static_cast<int>(c.opX)) + " dev=" + std::to_string(c.device);
    }

    // Block triangular matrix, diagonally dominant on the scalar level. The solved
    // triangle of a diagonal block is random (only the diagonal for a unit
    // diagonal, where the stored diagonal is ignored); its other triangle holds 37,
    // which the solve must not read.
    template <typename T>
    Bsr block_triangular(const BsrsmCase& c, Rng& rng)
    {
        const bool lower = (c.fill == rocsparse_fill_mode_lower);
        const bool unit  = (c.diag == rocsparse_diag_type_unit);
        Bsr        B;
        B.mb = B.nb = c.mb;
        B.bd        = c.bd;
        B.ptr.assign(static_cast<size_t>(c.mb) + 1, 0);
        const double scale = 1.0 / (64.0 * static_cast<double>(c.bd));
        for(int64_t i = 0; i < c.mb; ++i)
        {
            std::set<int64_t> cols;
            const int64_t     lo = lower ? 0 : i + 1;
            const int64_t     hi = lower ? i - 1 : c.mb - 1;
            if(hi >= lo)
            {
                const int64_t cnt = rng.range(0, 2);
                for(int64_t q = 0; q < cnt; ++q)
                    cols.insert(rng.range(lo, hi));
            }
            cols.insert(i); // the diagonal block, also needed for a unit diagonal
            // A block of 37 in the other triangle, which the solve must not read.
            const int64_t ignored = lower ? c.mb - 1 : 0;
            if(lower ? ignored > i : ignored < i)
                cols.insert(ignored);
            for(int64_t j : cols)
            {
                B.col.push_back(j);
                if(lower ? j > i : j < i)
                {
                    B.val.insert(B.val.end(), c.bd * c.bd, round_to<T>(cplx(37.0, 0.0)));
                    continue;
                }
                for(int64_t r = 0; r < c.bd; ++r)
                    for(int64_t k = 0; k < c.bd; ++k)
                    {
                        const double re = scale * rng.range(-4, 4);
                        cplx         v(re, is_complex_t<T>::value ? scale * rng.range(-4, 4) : 0.0);
                        if(j == i)
                        {
                            // The other triangle of the diagonal block holds 37, which
                            // the solve must not read.
                            if(r == k)
                                v = unit ? cplx(11.0, 0.0) : rand_diag<T>(rng);
                            else if((lower && k > r) || (!lower && k < r))
                                v = cplx(37.0, 0.0);
                            else if(unit)
                                v = cplx(0.0, 0.0);
                            if(j == c.singular_block && r == 0 && k == 0)
                                v = cplx(0.0, 0.0);
                        }
                        B.val.push_back(round_to<T>(v));
                    }
            }
            B.ptr[static_cast<size_t>(i) + 1] = static_cast<int64_t>(B.col.size());
        }
        return B;
    }

    // Scalar matrix the solve is defined on.
    Dense reference_matrix(const BsrsmCase& c, const Bsr& B)
    {
        const bool    lower = (c.fill == rocsparse_fill_mode_lower);
        const int64_t n     = c.mb * c.bd;
        Dense         D(n, n);
        for(int64_t i = 0; i < c.mb; ++i)
            for(int64_t p = B.ptr[static_cast<size_t>(i)]; p < B.ptr[static_cast<size_t>(i) + 1];
                ++p)
                for(int64_t r = 0; r < c.bd; ++r)
                    for(int64_t k = 0; k < c.bd; ++k)
                    {
                        const int64_t row = i * c.bd + r;
                        const int64_t col = B.col[static_cast<size_t>(p)] * c.bd + k;
                        if((lower && col <= row) || (!lower && col >= row))
                            D.at(row, col)
                                = B.val[static_cast<size_t>(p * c.bd * c.bd + r * c.bd + k)];
                    }
        if(c.diag == rocsparse_diag_type_unit)
            for(int64_t i = 0; i < n; ++i)
                D.at(i, i) = cplx(1.0, 0.0);
        return D;
    }

    template <typename T>
    void run_bsrsm(rocsparse_handle handle, const BsrsmCase& c)
    {
        SCOPED_TRACE(describe(c));
        Rng           rng(static_cast<uint64_t>(c.bd * 101 + c.mb * 13 + c.nrhs) + 9);
        const Bsr     B = block_triangular<T>(c, rng);
        const int64_t n = c.mb * c.bd;

        const Dense   Bl = random_dense<T>(n, c.nrhs, rng); // logical right-hand side
        const Dense   Bs = apply_op(Bl, c.opX); // stored operand
        const int64_t ld = Bs.rows + 3; // column major, padded
        DevBuf<T>     dB = up_raw<T>(store_dense<T>(Bs, rocsparse_order_column, ld));
        DevBuf<T>     dX
            = up_raw<T>(store_dense<T>(Dense(Bs.rows, Bs.cols), rocsparse_order_column, ld));

        DevBuf<rocsparse_int> ptr = up_idx<rocsparse_int>(B.ptr),
                              ind = up_idx<rocsparse_int>(B.col);
        DevBuf<T> val             = up_raw<T>(bsr_pack<T>(B, c.dir));
        ASSERT_TRUE(dB.ptr && dX.ptr && ptr.ptr && ind.ptr && val.ptr);

        MatDescrH descr;
        ASSERT_EQ(rocsparse_set_mat_fill_mode(descr.d, c.fill), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_diag_type(descr.d, c.diag), rocsparse_status_success);
        MatInfoH info;
        ASSERT_NE(info.i, nullptr);

        Scalar<T>        alpha(c.alpha, c.device);
        PointerModeGuard mode(handle, mode_of(c.device));

        const rocsparse_int mb   = static_cast<rocsparse_int>(c.mb);
        const rocsparse_int nrhs = static_cast<rocsparse_int>(c.nrhs);
        const rocsparse_int nnzb = static_cast<rocsparse_int>(B.nnzb());
        const rocsparse_int bd   = static_cast<rocsparse_int>(c.bd);

        size_t bytes = 0;
        ASSERT_EQ(Api<T>::bsrsm_buffer_size(handle,
                                            c.dir,
                                            c.opA,
                                            c.opX,
                                            mb,
                                            nrhs,
                                            nnzb,
                                            descr.d,
                                            static_cast<const T*>(val.ptr),
                                            static_cast<const rocsparse_int*>(ptr.ptr),
                                            static_cast<const rocsparse_int*>(ind.ptr),
                                            bd,
                                            info.i,
                                            &bytes),
                  rocsparse_status_success);
        DevBuf<char> buffer(bytes);
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(Api<T>::bsrsm_analysis(handle,
                                         c.dir,
                                         c.opA,
                                         c.opX,
                                         mb,
                                         nrhs,
                                         nnzb,
                                         descr.d,
                                         static_cast<const T*>(val.ptr),
                                         static_cast<const rocsparse_int*>(ptr.ptr),
                                         static_cast<const rocsparse_int*>(ind.ptr),
                                         bd,
                                         info.i,
                                         rocsparse_analysis_policy_force,
                                         rocsparse_solve_policy_auto,
                                         buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(Api<T>::bsrsm_solve(handle,
                                      c.dir,
                                      c.opA,
                                      c.opX,
                                      mb,
                                      nrhs,
                                      nnzb,
                                      alpha.typed(),
                                      descr.d,
                                      static_cast<const T*>(val.ptr),
                                      static_cast<const rocsparse_int*>(ptr.ptr),
                                      static_cast<const rocsparse_int*>(ind.ptr),
                                      bd,
                                      info.i,
                                      static_cast<const T*>(dB.ptr),
                                      static_cast<rocsparse_int>(ld),
                                      dX.ptr,
                                      static_cast<rocsparse_int>(ld),
                                      rocsparse_solve_policy_auto,
                                      buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        // Zero pivot query (host or device memory, following the pointer mode).
        rocsparse_int         position = -2;
        DevBuf<rocsparse_int> dpos(1);
        ASSERT_TRUE(dpos.ptr);
        rocsparse_status st
            = rocsparse_bsrsm_zero_pivot(handle, info.i, c.device ? dpos.ptr : &position);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        if(c.device)
            position = dpos.host()[0];
        if(c.singular_block >= 0)
        {
            EXPECT_EQ(st, rocsparse_status_zero_pivot);
            EXPECT_EQ(position, c.singular_block);
            return;
        }
        EXPECT_EQ(st, rocsparse_status_success);
        EXPECT_EQ(position, -1);

        const bool lower
            = (c.fill == rocsparse_fill_mode_lower) == (c.opA == rocsparse_operation_none);
        const Dense Aop = apply_op(reference_matrix(c, B), c.opA);
        Dense       rhs = Bl;
        for(auto& v : rhs.v)
            v *= to_c<T>(alpha.h);
        const Dense want = solve_triangular(Aop, lower, rhs);
        for(const cplx& v : want.v)
            ASSERT_TRUE(std::isfinite(v.real()) && std::isfinite(v.imag()));
        // float and float complex
        constexpr bool single
            = std::is_same<T, float>::value || std::is_same<T, rocsparse_float_complex>::value;
        const double tol = single ? 1e-5 : 1e-9;
        expect_dense_equal<T>(dX.host(),
                              apply_op(want, c.opX),
                              rocsparse_order_column,
                              ld,
                              "X",
                              cplx(-77.0, 0.0),
                              tol);
        // B is an input only.
        expect_dense_equal<T>(dB.host(), Bs, rocsparse_order_column, ld, "B");
    }

    struct ExtraBsrsm : public rocsparse_ut::HandleTest
    {
    };
}

// Block dimensions of every kernel class (<= 4, <= 8, <= 16, <= 32) with all
// direction / fill / operation combinations.
TEST_F(ExtraBsrsm, grid_float)
{
    for(int64_t bd : {2, 5, 8, 9, 16, 17, 32})
        for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
            for(rocsparse_fill_mode fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
                for(rocsparse_operation opA :
                    {rocsparse_operation_none, rocsparse_operation_transpose})
                    for(rocsparse_operation opX :
                        {rocsparse_operation_none, rocsparse_operation_transpose})
                    {
                        BsrsmCase c;
                        c.bd   = bd;
                        c.mb   = (bd > 8) ? 3 : 5;
                        c.dir  = dir;
                        c.fill = fill;
                        c.opA  = opA;
                        c.opX  = opX;
                        run_bsrsm<float>(handle, c);
                        if(HasFatalFailure())
                            return;
                    }
}

TEST_F(ExtraBsrsm, unit_diagonal_and_scalars)
{
    for(int64_t bd : {3, 8, 20})
        for(bool device : {false, true})
            for(rocsparse_operation opA : {rocsparse_operation_none, rocsparse_operation_transpose})
            {
                BsrsmCase c;
                c.bd     = bd;
                c.mb     = 4;
                c.diag   = rocsparse_diag_type_unit;
                c.opA    = opA;
                c.device = device;
                c.fill   = device ? rocsparse_fill_mode_upper : rocsparse_fill_mode_lower;
                run_bsrsm<double>(handle, c);
                if(HasFatalFailure())
                    return;
                c.alpha = cplx(0.5, -1.0);
                run_bsrsm<rocsparse_float_complex>(handle, c);
                if(HasFatalFailure())
                    return;
                run_bsrsm<rocsparse_double_complex>(handle, c);
                if(HasFatalFailure())
                    return;
            }
}

TEST_F(ExtraBsrsm, zero_pivot)
{
    for(bool device : {false, true})
        for(rocsparse_fill_mode fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
        {
            BsrsmCase c;
            c.bd             = 4;
            c.mb             = 6;
            c.singular_block = 2;
            c.device         = device;
            c.fill           = fill;
            run_bsrsm<float>(handle, c);
            if(HasFatalFailure())
                return;
        }
}
