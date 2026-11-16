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
// Device unit tests for the sparse-times-dense multiplication paths:
//   library/src/level3/rocsparse_spmm.cpp          (CSR / CSC / COO dispatch)
//   library/src/level3/rocsparse_cscmm_impl.cpp    (CSC -> CSR operation flip)
//   library/src/level3/rocsparse_csrmm_impl.cpp    (algorithm / transpose dispatch,
//                                                   quick return, argument checks)
//   library/src/level3/rocsparse_csrmm_template_row_split.cpp
//                                                  (row-split kernels for every
//                                                   dense column count)
//
// All tests go through the public API (rocsparse_spmm, rocsparse_?csrmm). The
// complete C matrix is compared with a host reference computed in double
// complex precision, including the untouched leading-dimension padding.
//
#include "unit_test_extra_common.hpp"

using namespace extra_ut;

namespace
{
    struct SpmmCase
    {
        SFmt                 fmt        = SFmt::csr;
        rocsparse_spmm_alg   alg        = rocsparse_spmm_alg_default;
        rocsparse_operation  opA        = rocsparse_operation_none;
        rocsparse_operation  opB        = rocsparse_operation_none;
        rocsparse_order      order_B    = rocsparse_order_column;
        rocsparse_order      order_C    = rocsparse_order_column;
        int64_t              m          = 9; // rows of the stored A
        int64_t              k          = 7; // columns of the stored A
        int64_t              n          = 5; // columns of op(B) / C
        int64_t              max_nnz    = 4; // max non-zeros per row
        bool                 device     = false;
        cplx                 alpha      = cplx(2.0, 0.0);
        cplx                 beta       = cplx(-1.0, 0.0);
        bool                 preprocess = true;
        bool                 long_rows  = false; // one row touching all columns, one empty row
        int64_t              ld_pad     = 3;
        rocsparse_index_base base       = rocsparse_index_base_zero;
        // BSR only: block dimension and block direction; m and k count blocks.
        int64_t             bd  = 3;
        rocsparse_direction dir = rocsparse_direction_row;
    };

    std::string describe(const SpmmCase& c)
    {
        return "fmt=" + std::to_string(static_cast<int>(c.fmt))
               + " alg=" + std::to_string(static_cast<int>(c.alg))
               + " opA=" + std::to_string(static_cast<int>(c.opA))
               + " opB=" + std::to_string(static_cast<int>(c.opB))
               + " oB=" + std::to_string(static_cast<int>(c.order_B))
               + " oC=" + std::to_string(static_cast<int>(c.order_C)) + " m=" + std::to_string(c.m)
               + " k=" + std::to_string(c.k) + " n=" + std::to_string(c.n) + " dev="
               + std::to_string(c.device) + " base=" + std::to_string(static_cast<int>(c.base))
               + " bd=" + std::to_string(c.bd);
    }

    // Scalar (point) CSR of a BSR matrix.
    Csr expand_bsr(const Bsr& B)
    {
        Csr A;
        A.m = B.mb * B.bd;
        A.n = B.nb * B.bd;
        A.ptr.assign(static_cast<size_t>(A.m) + 1, 0);
        for(int64_t i = 0; i < B.mb; ++i)
            for(int64_t r = 0; r < B.bd; ++r)
            {
                for(int64_t p = B.ptr[static_cast<size_t>(i)];
                    p < B.ptr[static_cast<size_t>(i) + 1];
                    ++p)
                    for(int64_t c = 0; c < B.bd; ++c)
                    {
                        A.col.push_back(B.col[static_cast<size_t>(p)] * B.bd + c);
                        A.val.push_back(B.val[static_cast<size_t>(p * B.bd * B.bd + r * B.bd + c)]);
                    }
                A.ptr[static_cast<size_t>(i * B.bd + r) + 1] = static_cast<int64_t>(A.col.size());
            }
        return A;
    }

    // Runs the three stages of rocsparse_spmm and compares the whole C.
    template <typename T, typename I, typename J>
    void run_spmm(rocsparse_handle handle, const SpmmCase& c)
    {
        SCOPED_TRACE(describe(c));
        const rocsparse_datatype dt = rocsparse_ut::dt_of<T>();

        Rng rng(static_cast<uint64_t>(c.m * 131 + c.k * 17 + c.n) + 5);

        // Host operands.
        Bsr bsr_host;
        Csr A;
        if(c.fmt == SFmt::bsr)
        {
            bsr_host = random_bsr<T>(c.m, c.k, c.bd, c.max_nnz, rng);
            A        = expand_bsr(bsr_host);
        }
        else
        {
            std::vector<int64_t> len(static_cast<size_t>(c.m));
            for(auto& l : len)
                l = rng.range(0, c.max_nnz);
            if(c.long_rows && c.m > 2)
            {
                len[1] = c.k;
                len[2] = 0;
            }
            A = csr_with_row_lengths<T>(c.m, c.k, len, rng);
        }

        const bool    nonT = (c.opA == rocsparse_operation_none);
        const int64_t M    = nonT ? A.m : A.n; // rows of op(A) and of C
        const int64_t K    = nonT ? A.n : A.m; // columns of op(A), rows of op(B)

        const Dense opB_logical = random_dense<T>(K, c.n, rng);
        // apply_op is an involution, so applying it gives the stored operand.
        const Dense B_stored = apply_op(opB_logical, c.opB);
        const Dense C0       = random_dense<T>(M, c.n, rng);

        const int64_t ldb = min_ld(B_stored, c.order_B) + c.ld_pad;
        const int64_t ldc = min_ld(C0, c.order_C) + c.ld_pad;

        DevBuf<T> dB = up_raw<T>(store_dense<T>(B_stored, c.order_B, ldb));
        DevBuf<T> dC = up_raw<T>(store_dense<T>(C0, c.order_C, ldc));
        ASSERT_TRUE(dB.ptr && dC.ptr);

        DevSparse<T, I, J> sp;
        if(c.fmt == SFmt::bsr)
        {
            const int64_t b = static_cast<int64_t>(c.base);
            sp.ptr          = up_idx<I>(bsr_host.ptr, b);
            sp.ind          = up_idx<J>(bsr_host.col, b);
            sp.val          = up_raw<T>(bsr_pack<T>(bsr_host, c.dir));
            ASSERT_TRUE(sp.ptr.ptr && sp.ind.ptr && sp.val.ptr);
            ASSERT_EQ(rocsparse_create_bsr_descr(&sp.h.p,
                                                 bsr_host.mb,
                                                 bsr_host.nb,
                                                 bsr_host.nnzb(),
                                                 c.dir,
                                                 bsr_host.bd,
                                                 sp.ptr.ptr,
                                                 sp.ind.ptr,
                                                 sp.val.ptr,
                                                 rocsparse_ut::it_of<I>(),
                                                 rocsparse_ut::it_of<J>(),
                                                 c.base,
                                                 dt),
                      rocsparse_status_success);
        }
        else
        {
            make_sparse<T, I, J>(sp, c.fmt, A, c.base);
        }
        if(::testing::Test::HasFatalFailure())
            return;

        DnmatH mB, mC;
        ASSERT_EQ(rocsparse_create_dnmat_descr(
                      &mB.p, B_stored.rows, B_stored.cols, ldb, dB.ptr, dt, c.order_B),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnmat_descr(&mC.p, M, c.n, ldc, dC.ptr, dt, c.order_C),
                  rocsparse_status_success);

        Scalar<T>        alpha(c.alpha, c.device);
        Scalar<T>        beta(c.beta, c.device);
        PointerModeGuard mode(handle, mode_of(c.device));

        auto stage = [&](rocsparse_spmm_stage st, size_t* bsz, void* buffer) {
            return rocsparse_spmm(handle,
                                  c.opA,
                                  c.opB,
                                  alpha.get(),
                                  sp.h.p,
                                  mB.p,
                                  beta.get(),
                                  mC.p,
                                  dt,
                                  c.alg,
                                  st,
                                  bsz,
                                  buffer);
        };
        auto proceed = [&](rocsparse_status st) {
            EXPECT_EQ(st, rocsparse_status_success);
            return st == rocsparse_status_success;
        };

        size_t bsz = 0;
        if(!proceed(stage(rocsparse_spmm_stage_buffer_size, &bsz, nullptr)))
            return;
        DevBuf<char> buffer(bsz);
        ASSERT_TRUE(buffer.ptr);
        if(c.preprocess && !proceed(stage(rocsparse_spmm_stage_preprocess, &bsz, buffer.ptr)))
            return;
        if(!proceed(stage(rocsparse_spmm_stage_compute, &bsz, buffer.ptr)))
            return;
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const Dense want = axpby(to_c<T>(alpha.h),
                                 matmul(apply_op(dense_of(A), c.opA), opB_logical),
                                 to_c<T>(beta.h),
                                 C0);
        expect_dense_equal<T>(dC.host(), want, c.order_C, ldc, "C");
        // B is an input only.
        expect_dense_equal<T>(dB.host(), B_stored, c.order_B, ldb, "B");
    }

    const rocsparse_operation all_ops[]    = {rocsparse_operation_none,
                                           rocsparse_operation_transpose,
                                           rocsparse_operation_conjugate_transpose};
    const rocsparse_order     all_orders[] = {rocsparse_order_column, rocsparse_order_row};

    const rocsparse_spmm_alg csr_algs[] = {rocsparse_spmm_alg_default,
                                           rocsparse_spmm_alg_csr,
                                           rocsparse_spmm_alg_csr_row_split,
                                           rocsparse_spmm_alg_csr_nnz_split,
                                           rocsparse_spmm_alg_csr_merge_path};

    // Dense column counts: the main kernels of the sparse-times-dense routines
    // are used for 64, 128, 192 and 256 columns at a time, the remainder
    // kernels for what is left (1, 2, 3-4, 5-8, 9-16, 17-32 and 33-64 columns).
    const std::vector<int64_t> n_list
        = {1,  2,  3,   4,   5,   8,   9,   16,  17,  32,  33,  64,  65,  66,  68, 72,
           80, 96, 127, 128, 129, 130, 132, 144, 192, 193, 196, 256, 257, 260, 300};
    const std::vector<int64_t> n_one = {5};

    // Grid of algorithms, operations, orders and dense column counts. Without
    // `both_orders` B and C share their storage order.
    template <typename T, typename I, typename J>
    void sweep(rocsparse_handle                        handle,
               SFmt                                    fmt,
               const std::vector<rocsparse_spmm_alg>&  algs,
               const std::vector<rocsparse_operation>& opAs,
               const std::vector<rocsparse_operation>& opBs,
               const std::vector<int64_t>&             ns,
               bool                                    device,
               bool                                    both_orders = true)
    {
        for(rocsparse_spmm_alg alg : algs)
            for(rocsparse_operation opA : opAs)
                for(rocsparse_operation opB : opBs)
                    for(rocsparse_order oB : all_orders)
                        for(rocsparse_order oC : all_orders)
                            for(int64_t n : ns)
                            {
                                if(!both_orders && oB != oC)
                                    continue;
                                SpmmCase c;
                                c.fmt       = fmt;
                                c.alg       = alg;
                                c.opA       = opA;
                                c.opB       = opB;
                                c.order_B   = oB;
                                c.order_C   = oC;
                                c.n         = n;
                                c.m         = 19;
                                c.k         = 13;
                                c.max_nnz   = 5;
                                c.device    = device;
                                c.long_rows = true;
                                run_spmm<T, I, J>(handle, c);
                                if(::testing::Test::HasFatalFailure())
                                    return;
                            }
    }

    const std::vector<rocsparse_spmm_alg>  csr_alg_list(std::begin(csr_algs), std::end(csr_algs));
    const std::vector<rocsparse_operation> op_list(std::begin(all_ops), std::end(all_ops));

    const std::vector<rocsparse_operation> op_none = {rocsparse_operation_none};
    const std::vector<rocsparse_operation> op_nt
        = {rocsparse_operation_none, rocsparse_operation_transpose};

    struct ExtraSpmm : public rocsparse_ut::HandleTest
    {
    };
}

// ---------------------------------------------------------------------------
// CSR and CSC (CSC goes through rocsparse_cscmm_impl.cpp, an operation flip):
// every algorithm, A / B operation and storage order.
// ---------------------------------------------------------------------------
TEST_F(ExtraSpmm, csr_csc_grid)
{
    for(SFmt f : {SFmt::csr, SFmt::csc})
    {
        sweep<float, int32_t, int32_t>(handle, f, csr_alg_list, op_list, op_list, n_one, false);
        sweep<double, int32_t, int32_t>(
            handle, f, csr_alg_list, op_list, op_nt, n_one, true, false);
        const std::vector<rocsparse_spmm_alg> algs = {
            rocsparse_spmm_alg_default,
            rocsparse_spmm_alg_csr_row_split,
            f == SFmt::csr ? rocsparse_spmm_alg_csr_merge_path : rocsparse_spmm_alg_csr_nnz_split};
        sweep<rocsparse_float_complex, int32_t, int32_t>(
            handle, f, algs, op_list, op_list, n_one, false, false);
        sweep<rocsparse_double_complex, int32_t, int32_t>(
            handle, f, algs, op_list, op_list, n_one, true, false);
    }
}

TEST_F(ExtraSpmm, index_types)
{
    const std::vector<rocsparse_spmm_alg> algs
        = {rocsparse_spmm_alg_default, rocsparse_spmm_alg_csr_row_split};
    for(SFmt f : {SFmt::csr, SFmt::csc})
    {
        sweep<float, int64_t, int32_t>(handle, f, algs, op_nt, op_none, n_one, false, false);
        sweep<double, int64_t, int64_t>(handle, f, algs, op_nt, op_none, n_one, false, false);
        sweep<double, int32_t, int32_t>(handle, f, algs, op_nt, op_none, n_one, false, false);
        sweep<rocsparse_float_complex, int64_t, int64_t>(
            handle, f, algs, op_nt, op_none, n_one, false, false);
    }
}

// ---------------------------------------------------------------------------
// Scalars: alpha / beta combinations (host and device), including the
// alpha == 0 && beta == 1 host quick return.
// ---------------------------------------------------------------------------
TEST_F(ExtraSpmm, scalar_combinations)
{
    const cplx alphas[] = {cplx(0.0, 0.0), cplx(1.0, 0.0), cplx(-2.0, 0.0), cplx(0.5, 0.0)};
    const cplx betas[]  = {cplx(0.0, 0.0), cplx(1.0, 0.0), cplx(3.0, 0.0)};
    for(SFmt f : {SFmt::csr, SFmt::csc})
        for(bool device : {false, true})
            for(rocsparse_operation opA : {rocsparse_operation_none, rocsparse_operation_transpose})
                for(const cplx& a : alphas)
                    for(const cplx& b : betas)
                    {
                        SpmmCase c;
                        c.fmt    = f;
                        c.opA    = opA;
                        c.alpha  = a;
                        c.beta   = b;
                        c.device = device;
                        run_spmm<float, int32_t, int32_t>(handle, c);
                        if(HasFatalFailure())
                            return;
                    }
}

// ---------------------------------------------------------------------------
// beta == 0 for every format, algorithm and kernel family.
// ---------------------------------------------------------------------------
TEST_F(ExtraSpmm, beta_zero)
{
    struct FmtAlgs
    {
        SFmt                            fmt;
        std::vector<rocsparse_spmm_alg> algs;
    };
    const std::vector<FmtAlgs> sets
        = {{SFmt::csr, csr_alg_list},
           {SFmt::csc, {rocsparse_spmm_alg_default, rocsparse_spmm_alg_csr_row_split}},
           {SFmt::coo,
            {rocsparse_spmm_alg_default,
             rocsparse_spmm_alg_coo_segmented,
             rocsparse_spmm_alg_coo_atomic,
             rocsparse_spmm_alg_coo_segmented_atomic}}};
    for(const FmtAlgs& fa : sets)
        for(rocsparse_spmm_alg alg : fa.algs)
            for(bool device : {false, true})
                for(rocsparse_operation opA :
                    {rocsparse_operation_none, rocsparse_operation_transpose})
                {
                    SpmmCase c;
                    c.fmt       = fa.fmt;
                    c.alg       = alg;
                    c.opA       = opA;
                    c.device    = device;
                    c.beta      = cplx(0.0, 0.0);
                    c.long_rows = true;
                    run_spmm<float, int32_t, int32_t>(handle, c);
                    if(HasFatalFailure())
                        return;
                    run_spmm<rocsparse_double_complex, int32_t, int32_t>(handle, c);
                    if(HasFatalFailure())
                        return;
                }
}

// ---------------------------------------------------------------------------
// Row-split csrmm for every dense column count n: the kernel selection chain
// in csrmm_template_row_split.cpp depends on n (loop width, remainder, ...).
// ---------------------------------------------------------------------------
TEST_F(ExtraSpmm, row_split_column_sweep)
{
    std::vector<int64_t> ns;
    for(int64_t n = 1; n <= 72; ++n)
        ns.push_back(n);
    for(int64_t n : {80, 96, 100, 127, 128, 129})
        ns.push_back(n);
    for(int64_t n : ns)
        for(rocsparse_order oB : all_orders)
        {
            SpmmCase c;
            c.alg     = rocsparse_spmm_alg_csr_row_split;
            c.n       = n;
            c.m       = 6;
            c.k       = 5;
            c.order_B = oB;
            c.order_C = oB;
            c.ld_pad  = 1;
            run_spmm<float, int32_t, int32_t>(handle, c);
            if(HasFatalFailure())
                return;
        }
}

// ---------------------------------------------------------------------------
// Legacy csrmm entry point (column major): numerical check for the operation
// combinations plus the argument checks of csrmm_impl.cpp.
// ---------------------------------------------------------------------------
TEST_F(ExtraSpmm, legacy_csrmm_values_and_errors)
{
    Rng                   rng(21);
    const int64_t         m = 8, k = 6, n = 4;
    const Csr             A = random_csr<float>(m, k, 4, rng);
    MatDescrH             descr;
    DevBuf<rocsparse_int> dptr = up_idx<rocsparse_int>(A.ptr);
    DevBuf<rocsparse_int> dcol = up_idx<rocsparse_int>(A.col);
    DevBuf<float>         dval = up_val<float>(A.val);
    // The same matrix with index base one.
    MatDescrH             descr1(rocsparse_index_base_one);
    DevBuf<rocsparse_int> dptr1 = up_idx<rocsparse_int>(A.ptr, 1);
    DevBuf<rocsparse_int> dcol1 = up_idx<rocsparse_int>(A.col, 1);

    for(int one_based = 0; one_based < 2; ++one_based)
        for(rocsparse_operation opA : {rocsparse_operation_none, rocsparse_operation_transpose})
            for(rocsparse_operation opB : {rocsparse_operation_none, rocsparse_operation_transpose})
            {
                const int64_t M    = (opA == rocsparse_operation_none) ? m : k;
                const int64_t K    = (opA == rocsparse_operation_none) ? k : m;
                const Dense   opBl = random_dense<float>(K, n, rng);
                const Dense   Bs   = apply_op(opBl, opB);
                const Dense   C0   = random_dense<float>(M, n, rng);
                const int64_t ldb  = min_ld(Bs, rocsparse_order_column) + 2;
                const int64_t ldc  = min_ld(C0, rocsparse_order_column) + 2;
                DevBuf<float> dB
                    = up_raw<float>(store_dense<float>(Bs, rocsparse_order_column, ldb));
                DevBuf<float> dC
                    = up_raw<float>(store_dense<float>(C0, rocsparse_order_column, ldc));
                const float alpha = 1.5f, beta = 0.5f;
                ASSERT_EQ(rocsparse_scsrmm(handle,
                                           opA,
                                           opB,
                                           static_cast<rocsparse_int>(m),
                                           static_cast<rocsparse_int>(n),
                                           static_cast<rocsparse_int>(k),
                                           static_cast<rocsparse_int>(A.nnz()),
                                           &alpha,
                                           one_based ? descr1.d : descr.d,
                                           dval.ptr,
                                           one_based ? dptr1.ptr : dptr.ptr,
                                           one_based ? dcol1.ptr : dcol.ptr,
                                           dB.ptr,
                                           static_cast<rocsparse_int>(ldb),
                                           &beta,
                                           dC.ptr,
                                           static_cast<rocsparse_int>(ldc)),
                          rocsparse_status_success);
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                const Dense want = axpby(
                    cplx(alpha, 0), matmul(apply_op(dense_of(A), opA), opBl), cplx(beta, 0), C0);
                expect_dense_equal<float>(dC.host(), want, rocsparse_order_column, ldc, "C");
            }

    // argument checks
    DevBuf<float> dB(size_t{256}), dC(size_t{256});
    const float   one = 1.f, zero = 0.f;
    // Argument positions of rocsparse_scsrmm.
    enum
    {
        HANDLE,
        OP_A,
        OP_B,
        M,
        N,
        K,
        NNZ,
        ALPHA,
        DESCR,
        VAL,
        PTR,
        COL,
        B,
        LDB,
        BETA,
        C,
        LDC
    };
    const rocsparse_int nnz  = static_cast<rocsparse_int>(A.nnz());
    const auto          call = make_call(rocsparse_scsrmm,
                                handle,
                                rocsparse_operation_none,
                                rocsparse_operation_none,
                                rocsparse_int(8),
                                rocsparse_int(4),
                                rocsparse_int(6),
                                nnz,
                                &one,
                                descr.d,
                                dval.ptr,
                                dptr.ptr,
                                dcol.ptr,
                                dB.ptr,
                                rocsparse_int(8),
                                &zero,
                                dC.ptr,
                                rocsparse_int(8));
    const auto          T_   = rocsparse_operation_transpose;
    EXPECT_EQ(call.set<HANDLE>(nullptr)(), rocsparse_status_invalid_handle);
    EXPECT_EQ(call.set<ALPHA>(nullptr)(), rocsparse_status_invalid_pointer);
    EXPECT_EQ(call.set<BETA>(nullptr)(), rocsparse_status_invalid_pointer);
    EXPECT_EQ(call.set<B>(nullptr)(), rocsparse_status_invalid_pointer);
    EXPECT_EQ(call.set<C>(nullptr)(), rocsparse_status_invalid_pointer);
    EXPECT_EQ(call.set<M>(-1)(), rocsparse_status_invalid_size);
    EXPECT_EQ(call.set<N>(-1)(), rocsparse_status_invalid_size);
    EXPECT_EQ(call.set<K>(-1)(), rocsparse_status_invalid_size);
    // ldb / ldc too small for each operation combination
    EXPECT_EQ(call.set<LDB>(5)(), rocsparse_status_invalid_size);
    EXPECT_EQ(call.set<LDB>(6).set<LDC>(7)(), rocsparse_status_invalid_size);
    EXPECT_EQ(call.set<OP_B>(T_).set<LDB>(3)(), rocsparse_status_invalid_size);
    EXPECT_EQ(call.set<OP_A>(T_).set<LDB>(5).set<LDC>(6)(), rocsparse_status_invalid_size);
    EXPECT_EQ(call.set<OP_A>(T_).set<LDC>(5)(), rocsparse_status_invalid_size);
    EXPECT_EQ(call.set<OP_A>(T_).set<OP_B>(T_).set<LDB>(3).set<LDC>(6)(),
              rocsparse_status_invalid_size);
}

// ---------------------------------------------------------------------------
// CSR nnz-split and merge-path: every dense column count.
// ---------------------------------------------------------------------------
TEST_F(ExtraSpmm, csr_nnz_split_merge_dense_columns)
{
    sweep<float, int32_t, int32_t>(
        handle,
        SFmt::csr,
        {rocsparse_spmm_alg_csr_nnz_split, rocsparse_spmm_alg_csr_merge_path},
        op_none,
        op_nt,
        n_list,
        false,
        false);
}

// ---------------------------------------------------------------------------
// COO: every dense column count for every algorithm.
// ---------------------------------------------------------------------------
TEST_F(ExtraSpmm, coo_dense_columns_float)
{
    sweep<float, int32_t, int32_t>(handle,
                                   SFmt::coo,
                                   {rocsparse_spmm_alg_coo_segmented,
                                    rocsparse_spmm_alg_coo_atomic,
                                    rocsparse_spmm_alg_coo_segmented_atomic},
                                   op_none,
                                   op_nt,
                                   n_list,
                                   false,
                                   false);
}

// ---------------------------------------------------------------------------
// Empty operands: the product vanishes, so C = beta * C.
// ---------------------------------------------------------------------------
TEST_F(ExtraSpmm, empty_operands_scale_C)
{
    struct Dims
    {
        int64_t m, k, n;
    };
    const Dims dims[] = {{6, 0, 4}, {0, 5, 4}, {6, 5, 0}};
    for(SFmt f : {SFmt::csr, SFmt::csc, SFmt::coo, SFmt::bsr})
        for(const Dims& d : dims)
            for(bool device : {false, true})
                for(rocsparse_operation opA :
                    {rocsparse_operation_none, rocsparse_operation_transpose})
                    for(double beta : {2.0, 0.0})
                    {
                        // BSR matrices are only supported without transposition.
                        if(f == SFmt::bsr && opA != rocsparse_operation_none)
                            continue;
                        SpmmCase c;
                        c.fmt    = f;
                        c.m      = d.m;
                        c.k      = d.k;
                        c.n      = d.n;
                        c.opA    = opA;
                        c.device = device;
                        c.beta   = cplx(beta, 0.0);
                        c.bd     = 2;
                        run_spmm<float, int32_t, int32_t>(handle, c);
                        if(HasFatalFailure())
                            return;
                    }
}

TEST_F(ExtraSpmm, alpha_zero_beta_one_is_a_no_op)
{
    for(SFmt f : {SFmt::coo, SFmt::bsr})
        for(rocsparse_operation opA : {rocsparse_operation_none, rocsparse_operation_transpose})
        {
            if(f == SFmt::bsr && opA != rocsparse_operation_none)
                continue;
            SpmmCase c;
            c.fmt   = f;
            c.opA   = opA;
            c.alpha = cplx(0.0, 0.0);
            c.beta  = cplx(1.0, 0.0);
            run_spmm<double, int32_t, int32_t>(handle, c);
            if(HasFatalFailure())
                return;
        }
}
