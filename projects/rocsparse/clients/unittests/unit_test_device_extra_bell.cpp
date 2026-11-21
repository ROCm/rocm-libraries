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
// Generic spmm with a blocked ELL matrix that has no stored blocks (the
// product is zero, so C = beta * C), with a blocked ELL matrix built by
// dense_to_sparse, with a hand-built complex matrix under every operation of
// B (the conjugate transpose has to conjugate B), and with formats spmm does
// not support.
//
#include "unit_test_extra_common.hpp"

using namespace extra_ut;

namespace
{
    struct ExtraBell : public rocsparse_ut::HandleTest
    {
    };

    struct BellRun
    {
        rocsparse_operation opA    = rocsparse_operation_none;
        int64_t             mb     = 3;
        int64_t             kb     = 2;
        int64_t             bd     = 2;
        int64_t             cols   = 0;
        int64_t             n      = 5;
        bool                device = false;
    };

    void run_empty_bell(rocsparse_handle handle, const BellRun& r)
    {
        SCOPED_TRACE("opA=" + std::to_string(static_cast<int>(r.opA))
                     + " mb=" + std::to_string(r.mb) + " kb=" + std::to_string(r.kb)
                     + " n=" + std::to_string(r.n) + " cols=" + std::to_string(r.cols));
        const int64_t m = r.mb * r.bd, k = r.kb * r.bd;
        const int64_t Mc = (r.opA == rocsparse_operation_none) ? m : k;
        const int64_t Kc = (r.opA == rocsparse_operation_none) ? k : m;

        Rng           rng(17);
        const Dense   Bl  = random_dense<float>(Kc, r.n, rng);
        const Dense   C0  = random_dense<float>(Mc, r.n, rng);
        const int64_t ldb = Kc + 1, ldc = Mc + 2;

        DevBuf<float> dB = up_raw<float>(store_dense<float>(Bl, rocsparse_order_column, ldb));
        DevBuf<float> dC = up_raw<float>(store_dense<float>(C0, rocsparse_order_column, ldc));
        // The arrays are never read, they only have to be valid pointers.
        DevBuf<int32_t> ind = up_idx<int32_t>(std::vector<int64_t>(4, 0));
        DevBuf<float>   val = up_raw<float>(std::vector<float>(4, 0.0f));
        ASSERT_TRUE(dB.ptr && dC.ptr && ind.ptr && val.ptr);

        SpmatH sp;
        ASSERT_EQ(rocsparse_create_bell_descr(&sp.p,
                                              m,
                                              k,
                                              rocsparse_direction_row,
                                              r.bd,
                                              r.cols,
                                              ind.ptr,
                                              val.ptr,
                                              rocsparse_indextype_i32,
                                              rocsparse_index_base_zero,
                                              rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        DnmatH mB, mC;
        ASSERT_EQ(
            rocsparse_create_dnmat_descr(
                &mB.p, Kc, r.n, ldb, dB.ptr, rocsparse_datatype_f32_r, rocsparse_order_column),
            rocsparse_status_success);
        ASSERT_EQ(
            rocsparse_create_dnmat_descr(
                &mC.p, Mc, r.n, ldc, dC.ptr, rocsparse_datatype_f32_r, rocsparse_order_column),
            rocsparse_status_success);

        Scalar<float>    alpha(cplx(2.0, 0.0), r.device);
        Scalar<float>    beta(cplx(-0.5, 0.0), r.device);
        PointerModeGuard mode(handle, mode_of(r.device));

        size_t bsz  = 0;
        auto   spmm = [&](rocsparse_spmm_stage stage, void* buffer) {
            return rocsparse_spmm(handle,
                                  r.opA,
                                  rocsparse_operation_none,
                                  alpha.get(),
                                  sp.p,
                                  mB.p,
                                  beta.get(),
                                  mC.p,
                                  rocsparse_datatype_f32_r,
                                  rocsparse_spmm_alg_bell,
                                  stage,
                                  &bsz,
                                  buffer);
        };
        ASSERT_EQ(spmm(rocsparse_spmm_stage_buffer_size, nullptr), rocsparse_status_success);
        DevBuf<char> buffer(bsz > 0 ? bsz : 1);
        ASSERT_TRUE(buffer.ptr);
        for(rocsparse_spmm_stage stage :
            {rocsparse_spmm_stage_preprocess, rocsparse_spmm_stage_compute})
            ASSERT_EQ(spmm(stage, buffer.ptr), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const Dense want = axpby(0.0, Dense(Mc, r.n), -0.5, C0);
        expect_dense_equal<float>(dC.host(), want, rocsparse_order_column, ldc, "C");
        expect_dense_equal<float>(dB.host(), Bl, rocsparse_order_column, ldb, "B");
    }
}

TEST_F(ExtraBell, no_stored_blocks_scales_C)
{
    for(rocsparse_operation opA : {rocsparse_operation_none, rocsparse_operation_transpose})
        for(bool device : {false, true})
        {
            BellRun r;
            r.opA    = opA;
            r.device = device;
            // The transposed quick return scales a k x n block, so only
            // a square block grid keeps it inside C (AISPARSE-924).
            if(opA == rocsparse_operation_transpose)
                r.kb = r.mb;
            run_empty_bell(handle, r);
            if(HasFatalFailure())
                return;
        }
}

namespace
{
    // rocsparse_dense_to_sparse into a blocked ELL matrix (not documented as supported,
    // the library implements it), then spmm with the result is compared with the dense
    // product.
    template <typename T, typename I>
    void
        run_dense_to_bell(rocsparse_handle handle, rocsparse_order order, rocsparse_index_base base)
    {
        SCOPED_TRACE("order=" + std::to_string(static_cast<int>(order))
                     + " base=" + std::to_string(static_cast<int>(base)));
        const rocsparse_datatype dt = rocsparse_ut::dt_of<T>();
        const int64_t            bd = 2, mb = 3, nb = 4, m = mb * bd, k = nb * bd, n = 3;
        // Stored blocks (block row, block column); block row 2 has the most (3).
        const int64_t blocks[][2] = {{0, 0}, {0, 2}, {1, 3}, {2, 1}, {2, 2}, {2, 3}};
        Rng           rng(23);
        Dense         A(m, k);
        for(const auto& b : blocks)
            for(int64_t r = 0; r < bd; ++r)
                for(int64_t c = 0; c < bd; ++c)
                    if(r != c || b[1] != 2)
                        A.at(b[0] * bd + r, b[1] * bd + c) = round_to<T>(rand_val<T>(rng));
        const int64_t lda = (order == rocsparse_order_column ? m : k) + 1;
        DevBuf<T>     dA  = up_raw<T>(store_dense<T>(A, order, lda));
        DnmatH        mA;
        ASSERT_EQ(rocsparse_create_dnmat_descr(&mA.p, m, k, lda, dA.ptr, dt, order),
                  rocsparse_status_success);

        DevBuf<I> ind0(size_t{1});
        DevBuf<T> val0(size_t{1});
        SpmatH    sp;
        ASSERT_EQ(rocsparse_create_bell_descr(&sp.p,
                                              m,
                                              k,
                                              rocsparse_direction_row,
                                              bd,
                                              0,
                                              ind0.ptr,
                                              val0.ptr,
                                              rocsparse_ut::it_of<I>(),
                                              base,
                                              dt),
                  rocsparse_status_success);
        const auto alg = rocsparse_dense_to_sparse_alg_default;
        size_t     bsz = 0;
        ASSERT_EQ(rocsparse_dense_to_sparse(handle, mA.p, sp.p, alg, &bsz, nullptr),
                  rocsparse_status_success);
        DevBuf<char> buffer(bsz);
        ASSERT_EQ(rocsparse_dense_to_sparse(handle, mA.p, sp.p, alg, nullptr, buffer.ptr),
                  rocsparse_status_success);

        int64_t              rows = 0, cols = 0, block_dim = 0, ell_cols = -1;
        rocsparse_direction  dir;
        void*                pind = nullptr;
        void*                pval = nullptr;
        rocsparse_indextype  it;
        rocsparse_index_base got_base;
        rocsparse_datatype   got_dt;
        ASSERT_EQ(rocsparse_bell_get(sp.p,
                                     &rows,
                                     &cols,
                                     &dir,
                                     &block_dim,
                                     &ell_cols,
                                     &pind,
                                     &pval,
                                     &it,
                                     &got_base,
                                     &got_dt),
                  rocsparse_status_success);
        ASSERT_EQ(ell_cols % bd, 0);
        ASSERT_GE(ell_cols, 3 * bd);
        DevBuf<I> ind = up_idx<I>(std::vector<int64_t>(mb * (ell_cols / bd), -5));
        DevBuf<T> val = up_val<T>(std::vector<cplx>(m * ell_cols, cplx(-66.0, 0.0)));
        ASSERT_EQ(rocsparse_bell_set_pointers(sp.p, ind.ptr, val.ptr), rocsparse_status_success);
        ASSERT_EQ(rocsparse_dense_to_sparse(handle, mA.p, sp.p, alg, &bsz, buffer.ptr),
                  rocsparse_status_success);

        const Dense   X   = random_dense<T>(k, n, rng);
        const Dense   C0  = random_dense<T>(m, n, rng);
        const int64_t ldc = m + 2;
        DevBuf<T>     dX  = up_raw<T>(store_dense<T>(X, rocsparse_order_column, k));
        DevBuf<T>     dC  = up_raw<T>(store_dense<T>(C0, rocsparse_order_column, ldc));
        DnmatH        mX, mC;
        ASSERT_EQ(rocsparse_create_dnmat_descr(&mX.p, k, n, k, dX.ptr, dt, rocsparse_order_column),
                  rocsparse_status_success);
        ASSERT_EQ(
            rocsparse_create_dnmat_descr(&mC.p, m, n, ldc, dC.ptr, dt, rocsparse_order_column),
            rocsparse_status_success);
        const T alpha = from_c<T>(cplx(1.5, -0.5));
        const T beta  = from_c<T>(cplx(-0.5, 0.25));
        size_t  sbsz  = 0;
        auto    spmm  = [&](rocsparse_spmm_stage stage, void* buffer) {
            return rocsparse_spmm(handle,
                                  rocsparse_operation_none,
                                  rocsparse_operation_none,
                                  &alpha,
                                  sp.p,
                                  mX.p,
                                  &beta,
                                  mC.p,
                                  dt,
                                  rocsparse_spmm_alg_bell,
                                  stage,
                                  &sbsz,
                                  buffer);
        };
        ASSERT_EQ(spmm(rocsparse_spmm_stage_buffer_size, nullptr), rocsparse_status_success);
        DevBuf<char> sbuf(sbsz > 0 ? sbsz : 1);
        ASSERT_TRUE(sbuf.ptr);
        for(rocsparse_spmm_stage stage :
            {rocsparse_spmm_stage_preprocess, rocsparse_spmm_stage_compute})
            ASSERT_EQ(spmm(stage, sbuf.ptr), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        const Dense want = axpby(to_c<T>(alpha), matmul(A, X), to_c<T>(beta), C0);
        expect_dense_equal<T>(
            dC.host(), want, rocsparse_order_column, ldc, "C", cplx(-77.0, 0.0), 1e-5);
    }
}

TEST_F(ExtraBell, dense_to_sparse_blocked_ell_used_by_spmm)
{
    for(rocsparse_order order : {rocsparse_order_column, rocsparse_order_row})
    {
        run_dense_to_bell<float, int32_t>(handle, order, rocsparse_index_base_zero);
        if(HasFatalFailure())
            return;
        run_dense_to_bell<rocsparse_double_complex, int64_t>(
            handle, order, rocsparse_index_base_one);
        if(HasFatalFailure())
            return;
    }
}

namespace
{
    // Hand-built Blocked-ELL matrix (cuSPARSE layout: block column indices
    // row-major per block row, values row-major per block row with the blocks
    // of a block row side by side) times a dense B under every operation of B.
    // Complex values, so a conjugate transpose of B differs from a transpose.
    template <typename T, typename I>
    void run_bell_ops(rocsparse_handle     handle,
                      rocsparse_operation  opB,
                      rocsparse_order      order_B,
                      rocsparse_order      order_C,
                      rocsparse_index_base base,
                      bool                 device)
    {
        SCOPED_TRACE("opB=" + std::to_string(static_cast<int>(opB))
                     + " order_B=" + std::to_string(static_cast<int>(order_B))
                     + " order_C=" + std::to_string(static_cast<int>(order_C))
                     + " base=" + std::to_string(static_cast<int>(base))
                     + " device=" + std::to_string(device));
        const rocsparse_datatype dt = rocsparse_ut::dt_of<T>();
        const int64_t            bd = 2, mb = 3, kb = 3, ellw = 2, n = 5;
        const int64_t            m = mb * bd, k = kb * bd, ell_cols = ellw * bd;
        // Block columns of every block row, -1 marks an unused (padded) slot.
        const int64_t blocks[3][2] = {{0, 2}, {1, -1}, {2, 0}};

        Rng                  rng(29);
        Dense                A(m, k);
        std::vector<int64_t> ind(static_cast<size_t>(mb * ellw), -1);
        std::vector<cplx>    val(static_cast<size_t>(m * ell_cols), cplx(0.0, 0.0));
        for(int64_t br = 0; br < mb; ++br)
            for(int64_t ei = 0; ei < ellw; ++ei)
            {
                const int64_t bc = blocks[br][ei];
                if(bc < 0)
                    continue;
                ind[static_cast<size_t>(br * ellw + ei)] = bc;
                for(int64_t r = 0; r < bd; ++r)
                    for(int64_t c = 0; c < bd; ++c)
                    {
                        const cplx v = round_to<T>(rand_val<T>(rng));
                        val[static_cast<size_t>((br * bd + r) * ell_cols + ei * bd + c)] = v;
                        A.at(br * bd + r, bc * bd + c)                                   = v;
                    }
            }
        const Dense   Bl  = random_dense<T>(k, n, rng);
        const Dense   Bs  = apply_op(Bl, opB);
        const Dense   C0  = random_dense<T>(m, n, rng);
        const int64_t ldb = min_ld(Bs, order_B) + 1;
        const int64_t ldc = min_ld(C0, order_C) + 2;

        // up_idx adds the base, so the -1 padding slot becomes base - 1 as the
        // library expects for a one-based matrix.
        DevBuf<I> dind = up_idx<I>(ind, static_cast<int64_t>(base));
        DevBuf<T> dval = up_val<T>(val);
        DevBuf<T> dB   = up_raw<T>(store_dense<T>(Bs, order_B, ldb));
        DevBuf<T> dC   = up_raw<T>(store_dense<T>(C0, order_C, ldc));
        ASSERT_TRUE(dind.ptr && dval.ptr && dB.ptr && dC.ptr);

        SpmatH sp;
        ASSERT_EQ(rocsparse_create_bell_descr(&sp.p,
                                              m,
                                              k,
                                              rocsparse_direction_row,
                                              bd,
                                              ell_cols,
                                              dind.ptr,
                                              dval.ptr,
                                              rocsparse_ut::it_of<I>(),
                                              base,
                                              dt),
                  rocsparse_status_success);
        DnmatH mB, mC;
        ASSERT_EQ(rocsparse_create_dnmat_descr(&mB.p, Bs.rows, Bs.cols, ldb, dB.ptr, dt, order_B),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnmat_descr(&mC.p, m, n, ldc, dC.ptr, dt, order_C),
                  rocsparse_status_success);

        Scalar<T>        alpha(cplx(2.0, -1.0), device);
        Scalar<T>        beta(cplx(-1.0, 0.5), device);
        PointerModeGuard mode(handle, mode_of(device));

        size_t bsz  = 0;
        auto   spmm = [&](rocsparse_spmm_stage stage, void* buffer) {
            return rocsparse_spmm(handle,
                                  rocsparse_operation_none,
                                  opB,
                                  alpha.get(),
                                  sp.p,
                                  mB.p,
                                  beta.get(),
                                  mC.p,
                                  dt,
                                  rocsparse_spmm_alg_bell,
                                  stage,
                                  &bsz,
                                  buffer);
        };
        ASSERT_EQ(spmm(rocsparse_spmm_stage_buffer_size, nullptr), rocsparse_status_success);
        DevBuf<char> buffer(bsz > 0 ? bsz : 1);
        ASSERT_TRUE(buffer.ptr);
        for(rocsparse_spmm_stage stage :
            {rocsparse_spmm_stage_preprocess, rocsparse_spmm_stage_compute})
            ASSERT_EQ(spmm(stage, buffer.ptr), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const Dense want = axpby(to_c<T>(alpha.h), matmul(A, Bl), to_c<T>(beta.h), C0);
        expect_dense_equal<T>(dC.host(), want, order_C, ldc, "C");
        expect_dense_equal<T>(dB.host(), Bs, order_B, ldb, "B");
    }
}

// Every operation of a complex B (the conjugate transpose has to conjugate B),
// both storage orders of B and C, both index bases and both scalar modes.
TEST_F(ExtraBell, complex_B_operations)
{
    const rocsparse_operation ops[]    = {rocsparse_operation_none,
                                          rocsparse_operation_transpose,
                                          rocsparse_operation_conjugate_transpose};
    const rocsparse_order     orders[] = {rocsparse_order_column, rocsparse_order_row};
    for(rocsparse_operation opB : ops)
        for(rocsparse_order oB : orders)
            for(rocsparse_order oC : orders)
            {
                run_bell_ops<rocsparse_float_complex, int32_t>(
                    handle, opB, oB, oC, rocsparse_index_base_zero, false);
                if(HasFatalFailure())
                    return;
                run_bell_ops<rocsparse_double_complex, int64_t>(
                    handle, opB, oB, oC, rocsparse_index_base_one, true);
                if(HasFatalFailure())
                    return;
            }
}

TEST_F(ExtraBell, unsupported_formats_are_rejected)
{
    const int64_t   m = 4, n = 3;
    DevBuf<int32_t> ind = up_idx<int32_t>(std::vector<int64_t>(m * 2, 0));
    DevBuf<float>   val = up_raw<float>(std::vector<float>(m * 2, 1.0f));
    DevBuf<float>   dB  = up_raw<float>(std::vector<float>(m * n, 1.0f));
    DevBuf<float>   dC  = up_raw<float>(std::vector<float>(m * n, 1.0f));
    ASSERT_TRUE(ind.ptr && val.ptr && dB.ptr && dC.ptr);

    SpmatH sp;
    ASSERT_EQ(rocsparse_create_ell_descr(&sp.p,
                                         m,
                                         m,
                                         ind.ptr,
                                         val.ptr,
                                         2,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    DnmatH mB, mC;
    ASSERT_EQ(rocsparse_create_dnmat_descr(
                  &mB.p, m, n, m, dB.ptr, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnmat_descr(
                  &mC.p, m, n, m, dC.ptr, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);

    const float alpha = 1.0f, beta = 0.0f;
    size_t      bsz = 0;
    // Only the generic spmm formats are supported, so the call must report an error.
    EXPECT_NE(rocsparse_spmm(handle,
                             rocsparse_operation_none,
                             rocsparse_operation_none,
                             &alpha,
                             sp.p,
                             mB.p,
                             &beta,
                             mC.p,
                             rocsparse_datatype_f32_r,
                             rocsparse_spmm_alg_default,
                             rocsparse_spmm_stage_buffer_size,
                             &bsz,
                             nullptr),
              rocsparse_status_success);
}
