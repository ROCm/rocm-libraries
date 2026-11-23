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
// product is zero, so C = beta * C) or a few stored blocks, including a row
// count that the block dimension does not divide (also with a block wider
// than the kernel tile), i64 indices in base one and a C without padding,
// with a transposed or batched blocked ELL product that spmm rejects, with a
// blocked ELL matrix built by dense_to_sparse, and with formats spmm does not
// support.
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
        rocsparse_operation  opA      = rocsparse_operation_none;
        int64_t              rows     = 6;
        int64_t              kb       = 2;
        int64_t              bd       = 2;
        int64_t              ell_cols = 0;
        int64_t              n        = 5;
        rocsparse_order      order    = rocsparse_order_column;
        bool                 device   = false;
        rocsparse_indextype  itype    = rocsparse_indextype_i32;
        rocsparse_index_base base     = rocsparse_index_base_zero;
        int64_t              ldc_pad  = 2;
    };

    // A blocked ELL A of `rows` scalar rows (ceil(rows / bd) block rows, so the
    // last block row is partial when bd does not divide rows) and kb block
    // columns, with ell_cols / bd stored blocks per block row and block column
    // indices of type itype in base `base`. Ad is A on the host. B and C, in
    // `order`, are those of op(A) * B and hold `slabs` copies of their
    // storage so that a strided batch stays in bounds. The leading
    // dimension of C is ldc_pad more than its rows (column order) or columns
    // (row order). C's slabs are followed by a guard of bd * ldc sentinels, so
    // a write to a row past the last row of C stays in the buffer and is seen
    // even without padding. c_size is the storage of one slab of C.
    struct BellProblem
    {
        int64_t            Mc = 0, Kc = 0, ldb = 0, ldc = 0, c_size = 0;
        Dense              Ad, Bl, C0;
        std::vector<float> hC;
        DevBuf<float>      dB, dC;
        DevBuf<int32_t>    ind;
        DevBuf<int64_t>    ind64;
        DevBuf<float>      val;
        SpmatH             sp;
        DnmatH             mB, mC;
    };

    void make_bell_problem(const BellRun& r, BellProblem& p, int64_t slabs = 1)
    {
        const int64_t m = r.rows, k = r.kb * r.bd;
        const int64_t mb = (m + r.bd - 1) / r.bd, width = r.ell_cols / r.bd;
        const bool    col = (r.order == rocsparse_order_column);
        p.Mc              = (r.opA == rocsparse_operation_none) ? m : k;
        p.Kc              = (r.opA == rocsparse_operation_none) ? k : m;
        p.ldb             = (col ? p.Kc : r.n) + 1;
        p.ldc             = (col ? p.Mc : r.n) + r.ldc_pad;

        Rng rng(17);
        p.Bl                  = random_dense<float>(p.Kc, r.n, rng);
        p.C0                  = random_dense<float>(p.Mc, r.n, rng);
        std::vector<float> hB = store_dense<float>(p.Bl, r.order, p.ldb);
        p.hC                  = store_dense<float>(p.C0, r.order, p.ldc);
        p.c_size              = static_cast<int64_t>(p.hC.size());
        hB.resize(slabs * hB.size(), -77.0f);
        p.hC.resize(static_cast<size_t>(slabs * p.c_size + r.bd * p.ldc), -77.0f);
        p.dB = up_raw<float>(hB);
        p.dC = up_raw<float>(p.hC);

        // Block row i stores block columns i, i + 1, ... (mod kb) in increasing
        // order. ell_val holds rows * ell_cols values, so the rows of a partial last
        // block row past `rows` have no storage.
        std::vector<int64_t> hind(mb * width);
        std::vector<cplx>    hval(m * r.ell_cols, cplx(0.0, 0.0));
        p.Ad = Dense(m, k);
        for(int64_t i = 0; i < mb; ++i)
        {
            const auto first = hind.begin() + i * width;
            for(int64_t e = 0; e < width; ++e)
                first[e] = (i + e) % r.kb;
            std::sort(first, first + width);
            for(int64_t e = 0; e < width; ++e)
                for(int64_t row = i * r.bd; row < std::min(m, (i + 1) * r.bd); ++row)
                    for(int64_t c = 0; c < r.bd; ++c)
                    {
                        const cplx v = round_to<float>(rand_val<float>(rng));
                        hval[row * r.ell_cols + e * r.bd + c] = v;
                        p.Ad.at(row, first[e] * r.bd + c)     = v;
                    }
        }
        ASSERT_TRUE(r.itype == rocsparse_indextype_i32 || r.itype == rocsparse_indextype_i64);
        void* ind = nullptr;
        if(r.itype == rocsparse_indextype_i64)
        {
            p.ind64 = up_idx<int64_t>(hind, static_cast<int64_t>(r.base));
            ind     = p.ind64.ptr;
        }
        else
        {
            p.ind = up_idx<int32_t>(hind, static_cast<int64_t>(r.base));
            ind   = p.ind.ptr;
        }
        p.val = up_val<float>(hval);
        ASSERT_TRUE(p.dB.ptr && p.dC.ptr && ind && p.val.ptr);

        ASSERT_EQ(rocsparse_create_bell_descr(&p.sp.p,
                                              m,
                                              k,
                                              rocsparse_direction_row,
                                              r.bd,
                                              r.ell_cols,
                                              ind,
                                              p.val.ptr,
                                              r.itype,
                                              r.base,
                                              rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnmat_descr(
                      &p.mB.p, p.Kc, r.n, p.ldb, p.dB.ptr, rocsparse_datatype_f32_r, r.order),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnmat_descr(
                      &p.mC.p, p.Mc, r.n, p.ldc, p.dC.ptr, rocsparse_datatype_f32_r, r.order),
                  rocsparse_status_success);
    }

    rocsparse_status bell_spmm(rocsparse_handle     handle,
                               rocsparse_operation  opA,
                               const void*          alpha,
                               BellProblem&         p,
                               const void*          beta,
                               rocsparse_spmm_stage stage,
                               size_t*              bsz,
                               void*                buffer)
    {
        return rocsparse_spmm(handle,
                              opA,
                              rocsparse_operation_none,
                              alpha,
                              p.sp.p,
                              p.mB.p,
                              beta,
                              p.mC.p,
                              rocsparse_datatype_f32_r,
                              rocsparse_spmm_alg_bell,
                              stage,
                              bsz,
                              buffer);
    }

    // trans_A == none: C = alpha * A * B + beta * C on every stage, compared
    // with the host product. With no stored blocks that is C = beta * C.
    void run_bell_product(rocsparse_handle handle, const BellRun& r)
    {
        SCOPED_TRACE("rows=" + std::to_string(r.rows) + " kb=" + std::to_string(r.kb)
                     + " bd=" + std::to_string(r.bd) + " n=" + std::to_string(r.n)
                     + " ell_cols=" + std::to_string(r.ell_cols)
                     + " order=" + std::to_string(static_cast<int>(r.order))
                     + " device=" + std::to_string(r.device));
        SCOPED_TRACE("itype=" + std::to_string(static_cast<int>(r.itype))
                     + " base=" + std::to_string(static_cast<int>(r.base))
                     + " ldc_pad=" + std::to_string(r.ldc_pad));
        ASSERT_EQ(r.opA, rocsparse_operation_none);
        BellProblem p;
        ASSERT_NO_FATAL_FAILURE(make_bell_problem(r, p));

        Scalar<float>    alpha(cplx(2.0, 0.0), r.device);
        Scalar<float>    beta(cplx(-0.5, 0.0), r.device);
        PointerModeGuard mode(handle, mode_of(r.device));

        size_t bsz  = 0;
        auto   spmm = [&](rocsparse_spmm_stage stage, void* buffer) {
            return bell_spmm(handle, r.opA, alpha.get(), p, beta.get(), stage, &bsz, buffer);
        };
        // The guard after C keeps its sentinel on every stage.
        auto expect_guard = [&](rocsparse_spmm_stage stage) {
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            const std::vector<float> hC  = p.dC.host();
            const auto               bad = std::count_if(
                hC.begin() + p.c_size, hC.end(), [](float v) { return v != -77.0f; });
            EXPECT_EQ(bad, 0) << "guard after C was written by stage " << static_cast<int>(stage);
        };
        ASSERT_EQ(spmm(rocsparse_spmm_stage_buffer_size, nullptr), rocsparse_status_success);
        ASSERT_NO_FATAL_FAILURE(expect_guard(rocsparse_spmm_stage_buffer_size));
        DevBuf<char> buffer(bsz > 0 ? bsz : 1);
        ASSERT_TRUE(buffer.ptr);
        for(rocsparse_spmm_stage stage :
            {rocsparse_spmm_stage_preprocess, rocsparse_spmm_stage_compute})
        {
            ASSERT_EQ(spmm(stage, buffer.ptr), rocsparse_status_success);
            ASSERT_NO_FATAL_FAILURE(expect_guard(stage));
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const Dense want = axpby(2.0, matmul(p.Ad, p.Bl), -0.5, p.C0);
        // Without stored blocks C = beta * C is exact.
        const double tol = (r.ell_cols == 0) ? 0.0 : 1e-5;
        expect_dense_equal<float>(p.dC.host(), want, r.order, p.ldc, "C", cplx(-77.0, 0.0), tol);
        expect_dense_equal<float>(p.dB.host(), p.Bl, r.order, p.ldb, "B");
    }
}

TEST_F(ExtraBell, no_stored_blocks_scales_C)
{
    // trans_A == none: an empty blocked ELL product is C = beta * C.
    // Host and device pointer modes both take that path. With 5 rows and
    // block dimension 2 the last block row is partial, and 1 row is less than
    // one block; every row of C is still scaled. Transposed A is covered by
    // transpose_is_not_implemented_on_every_stage.
    for(int64_t rows : {6, 5, 1})
        for(rocsparse_order order : {rocsparse_order_column, rocsparse_order_row})
            for(bool device : {false, true})
            {
                BellRun r;
                r.rows   = rows;
                r.order  = order;
                r.device = device;
                run_bell_product(handle, r);
                if(HasFatalFailure())
                    return;
            }
}

TEST_F(ExtraBell, stored_blocks_match_host_product)
{
    // trans_A == none with one or two stored blocks per block row, for B and C
    // in column and in row order. With 5 rows and block dimension 2 the last
    // block row is partial, and 1 row is less than one block: every row of C
    // holds the product, and the padding of C stays untouched.
    for(int64_t rows : {6, 5, 1})
        for(int64_t blocks : {1, 2})
            for(rocsparse_order order : {rocsparse_order_column, rocsparse_order_row})
            {
                BellRun r;
                r.rows     = rows;
                r.ell_cols = blocks * r.bd;
                r.order    = order;
                run_bell_product(handle, r);
                if(HasFatalFailure())
                    return;
            }
}

TEST_F(ExtraBell, stored_blocks_i64_base_one_tight_ldc)
{
    // As stored_blocks_match_host_product, with int64_t block column indices
    // in base one and ldc equal to the rows (column order) or columns (row
    // order) of C, so C has no padding. With 5 rows and 1 row the last block
    // row is partial; in row order a write to its row past the last row of C
    // lands in the guard after C.
    for(int64_t rows : {6, 5, 1})
        for(int64_t blocks : {1, 2})
            for(rocsparse_order order : {rocsparse_order_row, rocsparse_order_column})
            {
                BellRun r;
                r.rows     = rows;
                r.ell_cols = blocks * r.bd;
                r.order    = order;
                r.itype    = rocsparse_indextype_i64;
                r.base     = rocsparse_index_base_one;
                r.ldc_pad  = 0;
                run_bell_product(handle, r);
                if(HasFatalFailure())
                    return;
            }
}

TEST_F(ExtraBell, partial_block_row_wider_than_tile)
{
    // A block dimension of 33 is wider than the widest kernel tile (32), so a
    // thread handles rows r and r + 32 of its block row. With 41 rows the last
    // block row holds 8 rows, and on the second pass every thread has to skip
    // its row. C has no padding: a write to a row past the last row of C lands
    // in the next column of C (column order) or in the guard after C.
    for(rocsparse_order order : {rocsparse_order_column, rocsparse_order_row})
    {
        BellRun r;
        r.rows     = 41;
        r.bd       = 33;
        r.kb       = 2;
        r.ell_cols = r.bd;
        r.order    = order;
        r.ldc_pad  = 0;
        run_bell_product(handle, r);
        if(HasFatalFailure())
            return;
    }
}

TEST_F(ExtraBell, transpose_is_not_implemented_on_every_stage)
{
    // The unsupported-transpose check has to run before the empty-product
    // quick return, or a matrix with no stored blocks would succeed while
    // the same call with stored blocks returns not_implemented.
    const rocsparse_operation ops[]
        = {rocsparse_operation_transpose, rocsparse_operation_conjugate_transpose};
    for(rocsparse_operation opA : ops)
    {
        BellRun r;
        r.opA = opA;
        r.kb  = 5;
        // ell_cols == bd stores one block per block row.
        for(int64_t ell_cols : {int64_t{0}, r.bd})
        {
            r.ell_cols = ell_cols;
            SCOPED_TRACE("opA=" + std::to_string(static_cast<int>(opA))
                         + " ell_cols=" + std::to_string(ell_cols));
            BellProblem p;
            ASSERT_NO_FATAL_FAILURE(make_bell_problem(r, p));

            const float alpha = 2.0f, beta = -0.5f;
            size_t      bsz = 0;
            // Blocked ELL needs no buffer, so a stage that wrongly succeeds
            // does not keep the later stages from running.
            for(rocsparse_spmm_stage stage : {rocsparse_spmm_stage_buffer_size,
                                              rocsparse_spmm_stage_preprocess,
                                              rocsparse_spmm_stage_compute})
            {
                SCOPED_TRACE("stage=" + std::to_string(static_cast<int>(stage)));
                EXPECT_EQ(bell_spmm(handle, opA, &alpha, p, &beta, stage, &bsz, nullptr),
                          rocsparse_status_not_implemented);
            }
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            expect_dense_equal<float>(p.dC.host(), p.C0, rocsparse_order_column, p.ldc, "C");
        }
    }
}

TEST_F(ExtraBell, no_stored_blocks_batched_operands_are_rejected)
{
    // Blocked ELL spmm is not batched. An empty product with a batch count
    // other than 1 on A, B or C returns invalid_value from compute and leaves
    // C unchanged, as a product with stored blocks does. Buffer size and
    // preprocess do not see the batch counts and succeed for both.
    enum class Batched
    {
        A,
        B,
        C_stride0,
        C_strided
    };
    const std::pair<Batched, const char*> cases[] = {{Batched::A, "A"},
                                                     {Batched::B, "B"},
                                                     {Batched::C_stride0, "C, stride 0"},
                                                     {Batched::C_strided, "C, stride ldc * n"}};
    for(const auto& c : cases)
    {
        SCOPED_TRACE(std::string("batched ") + c.second);
        BellRun     r;
        BellProblem p;
        ASSERT_NO_FATAL_FAILURE(make_bell_problem(r, p, 2));
        // A scaled C would hold -2 with beta = -0.5. With a zero stride both
        // batches scale the same C.
        std::fill(p.hC.begin(), p.hC.end(), 4.0f);
        ASSERT_EQ(
            hipMemcpy(p.dC.ptr, p.hC.data(), p.hC.size() * sizeof(float), hipMemcpyHostToDevice),
            hipSuccess);
        switch(c.first)
        {
        case Batched::A:
            ASSERT_EQ(rocsparse_spmat_set_strided_batch(p.sp.p, 2), rocsparse_status_success);
            break;
        case Batched::B:
            ASSERT_EQ(rocsparse_dnmat_set_strided_batch(p.mB.p, 2, p.ldb * r.n),
                      rocsparse_status_success);
            break;
        case Batched::C_stride0:
            ASSERT_EQ(rocsparse_dnmat_set_strided_batch(p.mC.p, 2, 0), rocsparse_status_success);
            break;
        case Batched::C_strided:
            ASSERT_EQ(rocsparse_dnmat_set_strided_batch(p.mC.p, 2, p.ldc * r.n),
                      rocsparse_status_success);
            break;
        }

        const float alpha = 2.0f, beta = -0.5f;
        size_t      bsz  = 0;
        auto        spmm = [&](rocsparse_spmm_stage stage, void* buffer) {
            return bell_spmm(handle, r.opA, &alpha, p, &beta, stage, &bsz, buffer);
        };
        ASSERT_EQ(spmm(rocsparse_spmm_stage_buffer_size, nullptr), rocsparse_status_success);
        DevBuf<char> buffer(bsz > 0 ? bsz : 1);
        ASSERT_TRUE(buffer.ptr);
        EXPECT_EQ(spmm(rocsparse_spmm_stage_preprocess, buffer.ptr), rocsparse_status_success);
        EXPECT_EQ(spmm(rocsparse_spmm_stage_compute, buffer.ptr), rocsparse_status_invalid_value);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(p.dC.host(), p.hC) << "C was written";
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
