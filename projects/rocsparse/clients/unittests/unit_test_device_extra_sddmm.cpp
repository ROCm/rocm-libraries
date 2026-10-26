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
// Device unit tests for the generic rocsparse_sddmm API (library/src/level3/
// rocsparse_sddmm*.cpp). The default algorithm is run for the CSR, CSC, COO,
// COO (AoS) and ELL formats with every combination of operation and storage
// order of the dense matrices A and B, host and device scalars and a sweep
// of the inner dimension k that selects the kernel variant. Every value of
// the sparse output C (including the padding of ELL, which must stay
// untouched) is compared with a host reference
//     C = alpha * (op(A) * op(B)) sampled at the pattern of C + beta * C.
// The dense algorithm needs rocBLAS for its computation, so only the parts
// that run before any rocBLAS call are tested (buffer size formula, empty C,
// missing buffer); they behave the same whether or not rocBLAS is built in.
// The ELL padding holds a non-zero sentinel that must stay untouched.
//
#include "unit_test_extra_common.hpp"

using namespace extra_ut;

namespace
{
    enum class SddmmFmt
    {
        csr,
        csc,
        coo,
        coo_aos,
        ell
    };

    struct SddmmCase
    {
        SddmmFmt             fmt     = SddmmFmt::csr;
        rocsparse_operation  opA     = rocsparse_operation_none;
        rocsparse_operation  opB     = rocsparse_operation_none;
        rocsparse_order      order_A = rocsparse_order_column;
        rocsparse_order      order_B = rocsparse_order_column;
        int64_t              m       = 9;
        int64_t              n       = 8;
        int64_t              k       = 5;
        bool                 device  = false;
        cplx                 alpha   = cplx(2.0, 0.0);
        cplx                 beta    = cplx(-0.5, 0.0);
        int64_t              ld_pad  = 1;
        bool                 empty_C = false;
        uint64_t             seed    = 1;
        rocsparse_index_base base    = rocsparse_index_base_zero;
        // Only check the dense algorithm entry points that run before any
        // rocBLAS call instead of computing.
        bool dense_checks = false;
    };

    std::string describe(const SddmmCase& c)
    {
        return "fmt=" + std::to_string(static_cast<int>(c.fmt))
               + " opA=" + std::to_string(static_cast<int>(c.opA))
               + " opB=" + std::to_string(static_cast<int>(c.opB))
               + " order_A=" + std::to_string(static_cast<int>(c.order_A)) + " order_B="
               + std::to_string(static_cast<int>(c.order_B)) + " m=" + std::to_string(c.m)
               + " n=" + std::to_string(c.n) + " k=" + std::to_string(c.k) + " device="
               + std::to_string(c.device) + " base=" + std::to_string(static_cast<int>(c.base));
    }

    // One stored entry of C in the order of the device value array.
    struct Entry
    {
        int64_t row, col; // -1 for ELL padding (value: the padding sentinel)
        cplx    val;
    };

    // Dense algorithm: the parts that do not need rocBLAS.
    //  - buffer size: room for a dense m x n matrix of C's type when C has
    //    entries,
    //  - preprocess: nothing to do,
    //  - compute with an empty C: success without looking at the buffer,
    //  - compute with a non-empty C and no buffer: invalid pointer (checked
    //    before the dense conversion and the matrix multiplication).
    template <typename T>
    void dense_algorithm_checks(rocsparse_handle      handle,
                                const SddmmCase&      c,
                                const void*           alpha,
                                const void*           beta,
                                rocsparse_dnmat_descr mA,
                                rocsparse_dnmat_descr mB,
                                rocsparse_spmat_descr sC,
                                int64_t               nnz)
    {
        const rocsparse_datatype dt  = rocsparse_ut::dt_of<T>();
        const auto               alg = rocsparse_sddmm_alg_dense;

        size_t bsz = 12345;
        ASSERT_EQ(rocsparse_sddmm_buffer_size(
                      handle, c.opA, c.opB, alpha, mA, mB, beta, sC, dt, alg, &bsz),
                  rocsparse_status_success);
        const size_t dense_bytes = sizeof(T) * static_cast<size_t>(c.m) * static_cast<size_t>(c.n);
        if(nnz > 0)
            ASSERT_GE(bsz, dense_bytes);

        DevBuf<char> buffer(bsz);
        ASSERT_TRUE(buffer.ptr);
        EXPECT_EQ(rocsparse_sddmm_preprocess(
                      handle, c.opA, c.opB, alpha, mA, mB, beta, sC, dt, alg, buffer.ptr),
                  rocsparse_status_success);

        if(nnz == 0)
        {
            EXPECT_EQ(
                rocsparse_sddmm(handle, c.opA, c.opB, alpha, mA, mB, beta, sC, dt, alg, nullptr),
                rocsparse_status_success);
            EXPECT_EQ(
                rocsparse_sddmm(handle, c.opA, c.opB, alpha, mA, mB, beta, sC, dt, alg, buffer.ptr),
                rocsparse_status_success);
        }
        else
        {
            EXPECT_EQ(
                rocsparse_sddmm(handle, c.opA, c.opB, alpha, mA, mB, beta, sC, dt, alg, nullptr),
                rocsparse_status_invalid_pointer);
        }
    }

    template <typename T, typename I, typename J>
    void run_sddmm(rocsparse_handle handle, const SddmmCase& c)
    {
        SCOPED_TRACE(describe(c));
        const rocsparse_datatype dt = rocsparse_ut::dt_of<T>();
        Rng                      rng(c.seed * 31 + static_cast<uint64_t>(c.m * 7 + c.n * 3 + c.k));

        // Pattern and initial values of C.
        Csr P = c.empty_C ? csr_with_row_lengths<T>(
                    c.m, c.n, std::vector<int64_t>(static_cast<size_t>(c.m), 0), rng)
                          : random_csr<T>(c.m, c.n, std::min<int64_t>(c.n, 4), rng);
        for(auto& v : P.val)
            v = round_to<T>(rand_val<T>(rng));

        // Non-zero value stored in the ELL padding slots: the library must
        // neither write nor scale them.
        const cplx pad_sentinel(-77.0, 0.0);

        // Entries in the order of the value array of the chosen format.
        std::vector<Entry> entries;
        int64_t            ell_width = 0;
        switch(c.fmt)
        {
        case SddmmFmt::csr:
        case SddmmFmt::coo:
        case SddmmFmt::coo_aos:
        {
            for(int64_t i = 0; i < P.m; ++i)
                for(int64_t p = P.ptr[static_cast<size_t>(i)];
                    p < P.ptr[static_cast<size_t>(i) + 1];
                    ++p)
                    entries.push_back(
                        {i, P.col[static_cast<size_t>(p)], P.val[static_cast<size_t>(p)]});
            break;
        }
        case SddmmFmt::csc:
        {
            const Csr Pt = transpose(P);
            for(int64_t j = 0; j < Pt.m; ++j)
                for(int64_t p = Pt.ptr[static_cast<size_t>(j)];
                    p < Pt.ptr[static_cast<size_t>(j) + 1];
                    ++p)
                    entries.push_back(
                        {Pt.col[static_cast<size_t>(p)], j, Pt.val[static_cast<size_t>(p)]});
            break;
        }
        case SddmmFmt::ell:
        {
            for(int64_t i = 0; i < P.m; ++i)
                ell_width = std::max(
                    ell_width, P.ptr[static_cast<size_t>(i) + 1] - P.ptr[static_cast<size_t>(i)]);
            entries.assign(static_cast<size_t>(P.m * ell_width), Entry{-1, -1, pad_sentinel});
            for(int64_t i = 0; i < P.m; ++i)
            {
                const int64_t b = P.ptr[static_cast<size_t>(i)];
                for(int64_t w = 0; w < P.ptr[static_cast<size_t>(i) + 1] - b; ++w)
                    entries[static_cast<size_t>(w * P.m + i)]
                        = {i, P.col[static_cast<size_t>(b + w)], P.val[static_cast<size_t>(b + w)]};
            }
            break;
        }
        }

        // Dense operands.
        const Dense   A_logical = random_dense<T>(c.m, c.k, rng);
        const Dense   B_logical = random_dense<T>(c.k, c.n, rng);
        const Dense   A_stored  = apply_op(A_logical, c.opA);
        const Dense   B_stored  = apply_op(B_logical, c.opB);
        const int64_t ldA       = min_ld(A_stored, c.order_A) + c.ld_pad;
        const int64_t ldB       = min_ld(B_stored, c.order_B) + c.ld_pad;
        DevBuf<T>     dA        = up_raw<T>(store_dense<T>(A_stored, c.order_A, ldA));
        DevBuf<T>     dB        = up_raw<T>(store_dense<T>(B_stored, c.order_B, ldB));
        ASSERT_TRUE(dA.ptr && dB.ptr);

        // Sparse C.
        std::vector<cplx> init_vals(entries.size());
        for(size_t i = 0; i < entries.size(); ++i)
            init_vals[i] = entries[i].val;
        DevBuf<T> dval = up_val<T>(init_vals);
        ASSERT_TRUE(dval.ptr);

        const int64_t        nnz = static_cast<int64_t>(entries.size());
        const int64_t        b   = static_cast<int64_t>(c.base);
        DevBuf<I>            d_ptr_or_row;
        DevBuf<J>            d_idx;
        DevBuf<I>            d_idx_i;
        SpmatH               sC;
        std::vector<int64_t> v_a, v_b;
        switch(c.fmt)
        {
        case SddmmFmt::csr:
        {
            d_ptr_or_row = up_idx<I>(P.ptr, b);
            d_idx        = up_idx<J>(P.col, b);
            ASSERT_TRUE(d_ptr_or_row.ptr && d_idx.ptr);
            ASSERT_EQ(rocsparse_create_csr_descr(&sC.p,
                                                 c.m,
                                                 c.n,
                                                 nnz,
                                                 d_ptr_or_row.ptr,
                                                 d_idx.ptr,
                                                 dval.ptr,
                                                 rocsparse_ut::it_of<I>(),
                                                 rocsparse_ut::it_of<J>(),
                                                 c.base,
                                                 dt),
                      rocsparse_status_success);
            break;
        }
        case SddmmFmt::csc:
        {
            const Csr Pt = transpose(P);
            d_ptr_or_row = up_idx<I>(Pt.ptr, b);
            d_idx        = up_idx<J>(Pt.col, b);
            ASSERT_TRUE(d_ptr_or_row.ptr && d_idx.ptr);
            ASSERT_EQ(rocsparse_create_csc_descr(&sC.p,
                                                 c.m,
                                                 c.n,
                                                 nnz,
                                                 d_ptr_or_row.ptr,
                                                 d_idx.ptr,
                                                 dval.ptr,
                                                 rocsparse_ut::it_of<I>(),
                                                 rocsparse_ut::it_of<J>(),
                                                 c.base,
                                                 dt),
                      rocsparse_status_success);
            break;
        }
        case SddmmFmt::coo:
        {
            for(const Entry& e : entries)
            {
                v_a.push_back(e.row);
                v_b.push_back(e.col);
            }
            d_ptr_or_row = up_idx<I>(v_a, b);
            d_idx_i      = up_idx<I>(v_b, b);
            ASSERT_TRUE(d_ptr_or_row.ptr && d_idx_i.ptr);
            ASSERT_EQ(rocsparse_create_coo_descr(&sC.p,
                                                 c.m,
                                                 c.n,
                                                 nnz,
                                                 d_ptr_or_row.ptr,
                                                 d_idx_i.ptr,
                                                 dval.ptr,
                                                 rocsparse_ut::it_of<I>(),
                                                 c.base,
                                                 dt),
                      rocsparse_status_success);
            break;
        }
        case SddmmFmt::coo_aos:
        {
            for(const Entry& e : entries)
            {
                v_a.push_back(e.row);
                v_a.push_back(e.col);
            }
            d_ptr_or_row = up_idx<I>(v_a, b);
            ASSERT_TRUE(d_ptr_or_row.ptr);
            ASSERT_EQ(rocsparse_create_coo_aos_descr(&sC.p,
                                                     c.m,
                                                     c.n,
                                                     nnz,
                                                     d_ptr_or_row.ptr,
                                                     dval.ptr,
                                                     rocsparse_ut::it_of<I>(),
                                                     c.base,
                                                     dt),
                      rocsparse_status_success);
            break;
        }
        case SddmmFmt::ell:
        {
            // Column-major ELL, padding marked with -1 (zero based) / 0 (one based).
            for(const Entry& e : entries)
                v_a.push_back(e.col);
            d_ptr_or_row = up_idx<I>(v_a, b);
            ASSERT_TRUE(d_ptr_or_row.ptr);
            ASSERT_EQ(rocsparse_create_ell_descr(&sC.p,
                                                 c.m,
                                                 c.n,
                                                 d_ptr_or_row.ptr,
                                                 dval.ptr,
                                                 ell_width,
                                                 rocsparse_ut::it_of<I>(),
                                                 c.base,
                                                 dt),
                      rocsparse_status_success);
            break;
        }
        }

        DnmatH mA, mB;
        ASSERT_EQ(rocsparse_create_dnmat_descr(
                      &mA.p, A_stored.rows, A_stored.cols, ldA, dA.ptr, dt, c.order_A),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnmat_descr(
                      &mB.p, B_stored.rows, B_stored.cols, ldB, dB.ptr, dt, c.order_B),
                  rocsparse_status_success);

        Scalar<T>        alpha(c.alpha, c.device);
        Scalar<T>        beta(c.beta, c.device);
        PointerModeGuard mode(handle, mode_of(c.device));

        if(c.dense_checks)
        {
            ASSERT_NO_FATAL_FAILURE((dense_algorithm_checks<T>(
                handle, c, alpha.get(), beta.get(), mA.p, mB.p, sC.p, nnz)));
            return;
        }

        size_t bsz = 0;
        ASSERT_EQ(rocsparse_sddmm_buffer_size(handle,
                                              c.opA,
                                              c.opB,
                                              alpha.get(),
                                              mA.p,
                                              mB.p,
                                              beta.get(),
                                              sC.p,
                                              dt,
                                              rocsparse_sddmm_alg_default,
                                              &bsz),
                  rocsparse_status_success);
        DevBuf<char> buffer(bsz);
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(rocsparse_sddmm_preprocess(handle,
                                             c.opA,
                                             c.opB,
                                             alpha.get(),
                                             mA.p,
                                             mB.p,
                                             beta.get(),
                                             sC.p,
                                             dt,
                                             rocsparse_sddmm_alg_default,
                                             buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_sddmm(handle,
                                  c.opA,
                                  c.opB,
                                  alpha.get(),
                                  mA.p,
                                  mB.p,
                                  beta.get(),
                                  sC.p,
                                  dt,
                                  rocsparse_sddmm_alg_default,
                                  buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        // Reference.
        const Dense       prod = matmul(A_logical, B_logical);
        std::vector<cplx> want(entries.size());
        for(size_t i = 0; i < entries.size(); ++i)
        {
            const Entry& e = entries[i];
            want[i]        = (e.row < 0)
                                 ? e.val
                                 : to_c<T>(alpha.h) * prod.at(e.row, e.col) + to_c<T>(beta.h) * e.val;
        }
        expect_vec_equal<T>(dval.host(), want, "C val");
        // The dense inputs are not modified.
        expect_dense_equal<T>(dA.host(), A_stored, c.order_A, ldA, "A");
        expect_dense_equal<T>(dB.host(), B_stored, c.order_B, ldB, "B");
    }

    template <typename T, typename I, typename J>
    void sddmm_grid(rocsparse_handle                        handle,
                    const std::vector<SddmmFmt>&            fmts,
                    const std::vector<rocsparse_operation>& opAs,
                    const std::vector<rocsparse_operation>& opBs,
                    bool                                    device,
                    uint64_t                                seed)
    {
        const rocsparse_order orders[] = {rocsparse_order_column, rocsparse_order_row};
        for(SddmmFmt f : fmts)
            for(rocsparse_operation opA : opAs)
                for(rocsparse_operation opB : opBs)
                    for(rocsparse_order oA : orders)
                        for(rocsparse_order oB : orders)
                        {
                            SddmmCase c;
                            c.fmt     = f;
                            c.opA     = opA;
                            c.opB     = opB;
                            c.order_A = oA;
                            c.order_B = oB;
                            c.device  = device;
                            c.seed    = seed++;
                            run_sddmm<T, I, J>(handle, c);
                            if(::testing::Test::HasFatalFailure())
                                return;
                        }
    }

    const std::vector<SddmmFmt> all_fmts
        = {SddmmFmt::csr, SddmmFmt::csc, SddmmFmt::coo, SddmmFmt::coo_aos, SddmmFmt::ell};
    const std::vector<rocsparse_operation> ops
        = {rocsparse_operation_none, rocsparse_operation_transpose};

    struct ExtraSddmm : public rocsparse_ut::HandleTest
    {
    };
}

// The inner dimension selects the kernel variant.
TEST_F(ExtraSddmm, inner_dimension_sweep)
{
    const int64_t ks[] = {1, 2, 3, 4, 5, 8, 9, 16, 17, 33, 64, 70};
    uint64_t      seed = 600;
    for(SddmmFmt f : all_fmts)
        for(int64_t k : ks)
            for(bool device : {false, true})
            {
                SddmmCase c;
                c.fmt     = f;
                c.k       = k;
                c.device  = device;
                c.seed    = seed++;
                c.order_A = (k % 2 == 0) ? rocsparse_order_row : rocsparse_order_column;
                c.order_B = (k % 3 == 0) ? rocsparse_order_row : rocsparse_order_column;
                run_sddmm<float, int32_t, int32_t>(handle, c);
                if(HasFatalFailure())
                    return;
            }
}

TEST_F(ExtraSddmm, operations_and_orders)
{
    sddmm_grid<float, int32_t, int32_t>(handle, all_fmts, ops, ops, false, 700);
    if(HasFatalFailure())
        return;
    sddmm_grid<rocsparse_double_complex, int64_t, int64_t>(
        handle, {SddmmFmt::csr, SddmmFmt::ell}, ops, ops, true, 800);
}

TEST_F(ExtraSddmm, scalars_and_empty_pattern)
{
    const cplx alphas[] = {cplx(0.0, 0.0), cplx(1.0, 0.0), cplx(-2.0, 0.0)};
    const cplx betas[]  = {cplx(0.0, 0.0), cplx(1.0, 0.0), cplx(3.0, 0.0)};
    uint64_t   seed     = 900;
    for(SddmmFmt f : all_fmts)
    {
        for(bool device : {false, true})
            for(const cplx& a : alphas)
                for(const cplx& b : betas)
                {
                    SddmmCase c;
                    c.fmt    = f;
                    c.device = device;
                    c.alpha  = a;
                    c.beta   = b;
                    c.seed   = seed++;
                    run_sddmm<double, int32_t, int32_t>(handle, c);
                    if(HasFatalFailure())
                        return;
                }
        SddmmCase c;
        c.fmt     = f;
        c.empty_C = true;
        c.seed    = seed++;
        run_sddmm<float, int32_t, int32_t>(handle, c);
        if(HasFatalFailure())
            return;
    }
}

TEST_F(ExtraSddmm, dense_algorithm_without_blas)
{
    const int64_t dims[][2] = {{9, 8}, {16, 16}, {1, 1}, {33, 31}, {64, 20}};
    uint64_t      seed      = 1300;
    for(SddmmFmt f : all_fmts)
        for(const auto& d : dims)
            for(bool empty : {false, true})
            {
                SddmmCase c;
                c.fmt          = f;
                c.m            = d[0];
                c.n            = d[1];
                c.empty_C      = empty;
                c.dense_checks = true;
                c.seed         = seed++;
                run_sddmm<float, int32_t, int32_t>(handle, c);
                if(HasFatalFailure())
                    return;
                run_sddmm<double, int32_t, int32_t>(handle, c);
                if(HasFatalFailure())
                    return;
                run_sddmm<rocsparse_float_complex, int32_t, int32_t>(handle, c);
                if(HasFatalFailure())
                    return;
                run_sddmm<rocsparse_double_complex, int64_t, int64_t>(handle, c);
                if(HasFatalFailure())
                    return;
            }
}

TEST_F(ExtraSddmm, argument_errors)
{
    const rocsparse_datatype dt = rocsparse_datatype_f32_r;
    Rng                      rng(5);
    const Csr                P    = random_csr<float>(6, 5, 3, rng);
    DevBuf<int32_t>          dptr = up_idx<int32_t>(P.ptr);
    DevBuf<int32_t>          dcol = up_idx<int32_t>(P.col);
    DevBuf<float>            dval = up_val<float>(P.val);
    SpmatH                   sC;
    ASSERT_EQ(rocsparse_create_csr_descr(&sC.p,
                                         6,
                                         5,
                                         P.nnz(),
                                         dptr.ptr,
                                         dcol.ptr,
                                         dval.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         dt),
              rocsparse_status_success);
    DevBuf<float> dA(size_t{6 * 4}), dB(size_t{4 * 5});
    DnmatH        mA, mB;
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mA.p, 6, 4, 6, dA.ptr, dt, rocsparse_order_column),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mB.p, 4, 5, 4, dB.ptr, dt, rocsparse_order_column),
              rocsparse_status_success);

    const float  alpha = 1.f, beta = 0.f;
    const auto   N   = rocsparse_operation_none;
    const auto   CT  = rocsparse_operation_conjugate_transpose;
    const auto   ALG = rocsparse_sddmm_alg_default;
    size_t       bsz = 0;
    DevBuf<char> buf(size_t{4096});

    // Argument positions shared by the three entry points.
    enum
    {
        HANDLE,
        OP_A,
        OP_B,
        ALPHA,
        A,
        B,
        BETA,
        C,
        COMPUTE_TYPE,
        ALGORITHM,
        LAST // buffer (sddmm, preprocess) or size (buffer_size)
    };
    const auto sddmm = make_call(
        rocsparse_sddmm, handle, N, N, &alpha, mA.p, mB.p, &beta, sC.p, dt, ALG, buf.ptr);
    const auto buffer_size = make_call(
        rocsparse_sddmm_buffer_size, handle, N, N, &alpha, mA.p, mB.p, &beta, sC.p, dt, ALG, &bsz);
    const auto preprocess = make_call(rocsparse_sddmm_preprocess,
                                      handle,
                                      N,
                                      N,
                                      &alpha,
                                      mA.p,
                                      mB.p,
                                      &beta,
                                      sC.p,
                                      dt,
                                      ALG,
                                      buf.ptr);

    // The three entry points share their checks.
    auto check = [&](const auto& call) {
        EXPECT_EQ(call.template set<HANDLE>(nullptr)(), rocsparse_status_invalid_handle);
        EXPECT_EQ(call.template set<OP_A>(static_cast<rocsparse_operation>(99))(),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(call.template set<OP_B>(static_cast<rocsparse_operation>(99))(),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(call.template set<OP_A>(CT)(), rocsparse_status_not_implemented);
        EXPECT_EQ(call.template set<OP_B>(CT)(), rocsparse_status_not_implemented);
        EXPECT_EQ(call.template set<ALPHA>(nullptr)(), rocsparse_status_invalid_pointer);
        EXPECT_EQ(call.template set<BETA>(nullptr)(), rocsparse_status_invalid_pointer);
        EXPECT_EQ(call.template set<A>(nullptr)(), rocsparse_status_invalid_pointer);
        EXPECT_EQ(call.template set<B>(nullptr)(), rocsparse_status_invalid_pointer);
        EXPECT_EQ(call.template set<C>(nullptr)(), rocsparse_status_invalid_pointer);
    };
    check(sddmm);
    check(buffer_size);
    check(preprocess);
    EXPECT_EQ(buffer_size.set<LAST>(nullptr)(), rocsparse_status_invalid_pointer);

    // Compute type different from the type of the operands, invalid algorithm.
    EXPECT_EQ(sddmm.set<COMPUTE_TYPE>(rocsparse_datatype_f64_r)(), rocsparse_status_invalid_value);
    EXPECT_EQ(sddmm.set<ALGORITHM>(static_cast<rocsparse_sddmm_alg>(99))(),
              rocsparse_status_invalid_value);
}
