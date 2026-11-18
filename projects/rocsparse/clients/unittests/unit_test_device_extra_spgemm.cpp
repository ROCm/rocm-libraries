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
// Generic rocsparse_spgemm on CSR and BSR matrices:
//
//   C = alpha * A * B + beta * D     (alpha and beta given)
//   C = alpha * A * B                (beta == nullptr)
//   C = beta * D                     (alpha == nullptr)
//
// through the stages buffer_size / nnz / compute, and for CSR also through
// buffer_size / nnz / symbolic / numeric (the numeric stage is repeated with
// different scalars). Index types (i32/i32, i64/i32, i64/i64), value types,
// index bases of every operand, host and device scalars, empty operands and
// empty dimensions are covered. The whole of C (row pointer, column indices,
// values) is compared with a host reference.
//
#include "unit_test_extra_common.hpp"

#include <memory>
#include <sstream>

using namespace extra_ut;

namespace
{
    struct Case
    {
        bool                bsr  = false;
        int64_t             bd   = 1;
        rocsparse_direction dir  = rocsparse_direction_row;
        Mode                mode = Mode::multadd;
        int64_t             mb = 7, nb = 9, kb = 6;
        int64_t             max_row = 3;
        bool                device  = false;
        // Index bases of A, B, C, D (independent of each other).
        rocsparse_index_base bA = rocsparse_index_base_zero, bB = rocsparse_index_base_zero,
                             bC = rocsparse_index_base_zero, bD = rocsparse_index_base_zero;
        bool     split   = false; // CSR only: symbolic + numeric stages
        bool     empty_A = false, empty_B = false, empty_D = false;
        uint64_t seed = 1;
    };

    std::string describe(const Case& c)
    {
        return std::string(c.bsr ? "bsr" : "csr") + " mode=" + std::to_string(int(c.mode))
               + " mb=" + std::to_string(c.mb) + " nb=" + std::to_string(c.nb)
               + " kb=" + std::to_string(c.kb) + " bd=" + std::to_string(c.bd)
               + " dir=" + std::to_string(int(c.dir)) + " dev=" + std::to_string(c.device)
               + " split=" + std::to_string(c.split) + " bases=" + std::to_string(int(c.bA))
               + std::to_string(int(c.bB)) + std::to_string(int(c.bC)) + std::to_string(int(c.bD));
    }

    // An operand: host matrix, device arrays and the generic descriptor.
    template <typename T, typename I, typename J>
    struct Operand
    {
        Bsr       h;
        DevBuf<I> p;
        DevBuf<J> c;
        DevBuf<T> v;
        SpmatH    d;

        void create(const Bsr& host, bool bsr, rocsparse_direction dir, rocsparse_index_base base)
        {
            h = host;
            p = up_idx<I>(h.ptr, static_cast<int64_t>(base));
            c = up_idx<J>(h.col, static_cast<int64_t>(base));
            v = up_raw<T>(bsr_pack<T>(h, dir));
            ASSERT_TRUE(p.ptr && c.ptr && v.ptr);
            if(bsr)
            {
                ASSERT_EQ(rocsparse_create_bsr_descr(&d.p,
                                                     h.mb,
                                                     h.nb,
                                                     h.nnzb(),
                                                     dir,
                                                     h.bd,
                                                     p.ptr,
                                                     c.ptr,
                                                     v.ptr,
                                                     rocsparse_ut::it_of<I>(),
                                                     rocsparse_ut::it_of<J>(),
                                                     base,
                                                     rocsparse_ut::dt_of<T>()),
                          rocsparse_status_success);
            }
            else
            {
                ASSERT_EQ(rocsparse_create_csr_descr(&d.p,
                                                     h.mb,
                                                     h.nb,
                                                     h.nnzb(),
                                                     p.ptr,
                                                     c.ptr,
                                                     v.ptr,
                                                     rocsparse_ut::it_of<I>(),
                                                     rocsparse_ut::it_of<J>(),
                                                     base,
                                                     rocsparse_ut::dt_of<T>()),
                          rocsparse_status_success);
            }
        }
    };

    // The generic call with a given stage.
    template <typename T>
    rocsparse_status spgemm_call(rocsparse_handle            handle,
                                 const void*                 alpha,
                                 rocsparse_const_spmat_descr A,
                                 rocsparse_const_spmat_descr B,
                                 const void*                 beta,
                                 rocsparse_const_spmat_descr D,
                                 rocsparse_spmat_descr       C,
                                 rocsparse_spgemm_stage      stage,
                                 size_t*                     bs,
                                 void*                       tmp)
    {
        return rocsparse_spgemm(handle,
                                rocsparse_operation_none,
                                rocsparse_operation_none,
                                alpha,
                                A,
                                B,
                                beta,
                                D,
                                C,
                                rocsparse_ut::dt_of<T>(),
                                rocsparse_spgemm_alg_default,
                                stage,
                                bs,
                                tmp);
    }

    template <typename T, typename I, typename J>
    void run(rocsparse_handle handle, const Case& c)
    {
        SCOPED_TRACE(describe(c));
        const bool    mul = (c.mode != Mode::add);
        const bool    add = (c.mode != Mode::mult);
        const int64_t bd  = c.bsr ? c.bd : 1;

        Rng rng(c.seed * 977 + static_cast<uint64_t>(c.mb * 13 + c.nb * 7 + c.kb + bd));

        Bsr A = random_bsr<T>(c.mb, c.kb, bd, c.max_row, rng);
        Bsr B = random_bsr<T>(c.kb, c.nb, bd, c.max_row, rng);
        Bsr D = random_bsr<T>(c.mb, c.nb, bd, c.max_row, rng);
        if(c.empty_A)
            A = empty_bsr(c.mb, c.kb, bd);
        if(c.empty_B)
            B = empty_bsr(c.kb, c.nb, bd);
        if(c.empty_D)
            D = empty_bsr(c.mb, c.nb, bd);
        // Operands that do not take part in the operation are empty.
        const Bsr Ause = mul ? A : empty_bsr(c.mb, c.kb, bd);
        const Bsr Buse = mul ? B : empty_bsr(c.kb, c.nb, bd);
        const Bsr Duse = add ? D : empty_bsr(c.mb, c.nb, bd);

        Operand<T, I, J> oA, oB, oD;
        ASSERT_NO_FATAL_FAILURE(oA.create(Ause, c.bsr, c.dir, c.bA));
        ASSERT_NO_FATAL_FAILURE(oB.create(Buse, c.bsr, c.dir, c.bB));
        ASSERT_NO_FATAL_FAILURE(oD.create(Duse, c.bsr, c.dir, c.bD));

        DevBuf<I> pC(std::vector<I>(static_cast<size_t>(c.mb) + 1, I(-7)));
        ASSERT_TRUE(pC.ptr);
        SpmatH C;
        if(c.bsr)
        {
            ASSERT_EQ(rocsparse_create_bsr_descr(&C.p,
                                                 c.mb,
                                                 c.nb,
                                                 0,
                                                 c.dir,
                                                 bd,
                                                 pC.ptr,
                                                 nullptr,
                                                 nullptr,
                                                 rocsparse_ut::it_of<I>(),
                                                 rocsparse_ut::it_of<J>(),
                                                 c.bC,
                                                 rocsparse_ut::dt_of<T>()),
                      rocsparse_status_success);
        }
        else
        {
            ASSERT_EQ(rocsparse_create_csr_descr(&C.p,
                                                 c.mb,
                                                 c.nb,
                                                 0,
                                                 pC.ptr,
                                                 nullptr,
                                                 nullptr,
                                                 rocsparse_ut::it_of<I>(),
                                                 rocsparse_ut::it_of<J>(),
                                                 c.bC,
                                                 rocsparse_ut::dt_of<T>()),
                      rocsparse_status_success);
        }

        PointerModeGuard mode(handle, mode_of(c.device));

        auto scalars = [&](cplx a, cplx b) {
            return std::make_pair(std::make_unique<Scalar<T>>(a, c.device),
                                  std::make_unique<Scalar<T>>(b, c.device));
        };
        auto s1 = scalars(cplx(2.0, 1.0), cplx(-1.0, 0.5));

        auto call = [&](const Scalar<T>&       a,
                        const Scalar<T>&       b,
                        rocsparse_spgemm_stage stage,
                        size_t*                bs,
                        void*                  tmp) {
            return spgemm_call<T>(handle,
                                  mul ? a.get() : nullptr,
                                  oA.d.p,
                                  oB.d.p,
                                  add ? b.get() : nullptr,
                                  oD.d.p,
                                  C.p,
                                  stage,
                                  bs,
                                  tmp);
        };

        size_t bsz = 0;
        ASSERT_EQ(call(*s1.first, *s1.second, rocsparse_spgemm_stage_buffer_size, &bsz, nullptr),
                  rocsparse_status_success);
        DevBuf<char> buffer(bsz);
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(call(*s1.first, *s1.second, rocsparse_spgemm_stage_nnz, &bsz, buffer.ptr),
                  rocsparse_status_success);

        const Bsr want1 = bsr_gemm(
            mul, to_c<T>(s1.first->h), Ause, Buse, add, to_c<T>(s1.second->h), Duse, c.mb, c.nb);

        int64_t rows = 0, cols = 0, nnz = -1;
        ASSERT_EQ(rocsparse_spmat_get_size(C.p, &rows, &cols, &nnz), rocsparse_status_success);
        ASSERT_EQ(nnz, want1.nnzb());

        DevBuf<J> cC(static_cast<size_t>(nnz));
        DevBuf<T> vC(static_cast<size_t>(nnz * bd * bd));
        ASSERT_TRUE(cC.ptr && vC.ptr);
        if(c.bsr)
            ASSERT_EQ(rocsparse_bsr_set_pointers(C.p, pC.ptr, cC.ptr, vC.ptr),
                      rocsparse_status_success);
        else
            ASSERT_EQ(rocsparse_csr_set_pointers(C.p, pC.ptr, cC.ptr, vC.ptr),
                      rocsparse_status_success);

        auto check = [&](const Bsr& want) {
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            expect_bsr_equal<T, I, J>(
                pC.host(), cC.host(), vC.host(), want, c.dir, static_cast<int64_t>(c.bC));
        };

        if(!c.split)
        {
            ASSERT_EQ(call(*s1.first, *s1.second, rocsparse_spgemm_stage_compute, &bsz, buffer.ptr),
                      rocsparse_status_success);
            ASSERT_NO_FATAL_FAILURE(check(want1));
            return;
        }

        // Symbolic stage: row pointer and column indices of C only.
        ASSERT_EQ(call(*s1.first, *s1.second, rocsparse_spgemm_stage_symbolic, &bsz, buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        {
            // The pattern is already final, the values are not required.
            std::vector<int64_t> cols_want = want1.col;
            std::vector<J>       got       = cC.host();
            std::vector<I>       gp        = pC.host();
            ASSERT_GE(gp.size(), want1.ptr.size());
            for(int64_t i = 0; i < c.mb; ++i)
            {
                std::set<int64_t> g, w;
                for(int64_t q = static_cast<int64_t>(gp[static_cast<size_t>(i)] - c.bC);
                    q < static_cast<int64_t>(gp[static_cast<size_t>(i) + 1] - c.bC);
                    ++q)
                    g.insert(static_cast<int64_t>(got[static_cast<size_t>(q)]) - c.bC);
                for(int64_t q = want1.ptr[static_cast<size_t>(i)];
                    q < want1.ptr[static_cast<size_t>(i) + 1];
                    ++q)
                    w.insert(want1.col[static_cast<size_t>(q)]);
                ASSERT_EQ(g, w) << "symbolic pattern of row " << i;
            }
        }

        // Numeric stage, repeated with different scalars.
        ASSERT_EQ(call(*s1.first, *s1.second, rocsparse_spgemm_stage_numeric, &bsz, buffer.ptr),
                  rocsparse_status_success);
        ASSERT_NO_FATAL_FAILURE(check(want1));

        auto s2 = scalars(cplx(-3.0, 0.0), cplx(2.0, -1.0));
        ASSERT_EQ(call(*s2.first, *s2.second, rocsparse_spgemm_stage_numeric, &bsz, buffer.ptr),
                  rocsparse_status_success);
        const Bsr want2 = bsr_gemm(
            mul, to_c<T>(s2.first->h), Ause, Buse, add, to_c<T>(s2.second->h), Duse, c.mb, c.nb);
        ASSERT_NO_FATAL_FAILURE(check(want2));
    }

    const std::vector<Mode> kModes = {Mode::multadd, Mode::mult, Mode::add};

    template <typename T, typename I, typename J>
    void sweep_bsr(rocsparse_handle handle, const std::vector<int64_t>& dims)
    {
        for(int64_t bd : dims)
            for(Mode m : kModes)
                for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
                {
                    Case c;
                    c.bsr  = true;
                    c.bd   = bd;
                    c.mode = m;
                    c.dir  = dir;
                    run<T, I, J>(handle, c);
                    if(::testing::Test::HasFatalFailure())
                        return;
                }
    }

    template <typename T, typename I, typename J>
    void sweep_degenerate(rocsparse_handle handle, bool bsr)
    {
        for(Mode m : kModes)
            for(int which = 0; which < 7; ++which)
            {
                Case c;
                c.bsr  = bsr;
                c.bd   = bsr ? 2 : 1;
                c.mode = m;
                switch(which)
                {
                case 0:
                    c.empty_A = true;
                    break;
                case 1:
                    c.empty_B = true;
                    break;
                case 2:
                    c.empty_D = true;
                    break;
                case 3:
                    c.kb = 0;
                    break;
                case 4:
                    c.mb = 0;
                    break;
                case 5:
                    c.nb = 0;
                    break;
                case 6:
                    c.empty_A = c.empty_D = true;
                    break;
                }
                for(bool dev : {false, true})
                    for(bool split : {false, true})
                    {
                        if(split && bsr)
                            continue;
                        c.device = dev;
                        c.split  = split;
                        run<T, I, J>(handle, c);
                        if(::testing::Test::HasFatalFailure())
                            return;
                    }
            }
    }

    struct ExtraSpgemm : public rocsparse_ut::HandleTest
    {
    };
}

TEST_F(ExtraSpgemm, csr_degenerate)
{
    sweep_degenerate<float, int32_t, int32_t>(handle, false);
    sweep_degenerate<double, int64_t, int32_t>(handle, false);
}

TEST_F(ExtraSpgemm, bsr_block_dims_float)
{
    sweep_bsr<float, int32_t, int32_t>(handle, {2, 3, 4, 5, 8, 9, 16, 17});
}

TEST_F(ExtraSpgemm, bsr_degenerate)
{
    sweep_degenerate<float, int32_t, int32_t>(handle, true);
    sweep_degenerate<rocsparse_double_complex, int64_t, int64_t>(handle, true);
}

// Rows with many intermediate products go through the larger hashing groups
// and, beyond 65536 products, the bitmap algorithm.
namespace
{
    template <typename T>
    void long_rows(rocsparse_handle handle, Mode mode, bool sorted_B)
    {
        const bool    mul = (mode != Mode::add);
        const bool    add = (mode != Mode::mult);
        const int64_t m = 5, k = 300, n = 3000;
        Rng           rng(11);

        // A: first row touches every row of B, the others a few; B rows of 260
        // entries: row 0 of C has 78000 intermediate products.
        std::vector<int64_t> la = {300, 150, 40, 3, 0};
        std::vector<int64_t> lb(static_cast<size_t>(k), 260);
        std::vector<int64_t> ld = {2000, 5, 0, 100, 40};
        const Csr            A  = csr_with_row_lengths<T>(m, k, la, rng);
        const Csr            B  = csr_with_row_lengths<T>(k, n, lb, rng);
        const Csr            D  = csr_with_row_lengths<T>(m, n, ld, rng);
        Csr                  E0, E1, E2;
        E0.m          = m;
        E0.n          = k;
        E0.ptr        = std::vector<int64_t>(static_cast<size_t>(m) + 1, 0);
        E1.m          = k;
        E1.n          = n;
        E1.ptr        = std::vector<int64_t>(static_cast<size_t>(k) + 1, 0);
        E2.m          = m;
        E2.n          = n;
        E2.ptr        = std::vector<int64_t>(static_cast<size_t>(m) + 1, 0);
        const Csr& Au = mul ? A : E0;
        const Csr& Bu = mul ? B : E1;
        const Csr& Du = add ? D : E2;

        const Bsr want = bsr_gemm(mul,
                                  cplx(2.0, 0.0),
                                  bsr_from_csr(Au),
                                  bsr_from_csr(Bu),
                                  add,
                                  cplx(-1.0, 0.0),
                                  bsr_from_csr(Du),
                                  m,
                                  n);

        Operand<T, int32_t, int32_t> oA, oB, oD;
        ASSERT_NO_FATAL_FAILURE(
            oA.create(bsr_from_csr(Au), false, rocsparse_direction_row, rocsparse_index_base_zero));
        ASSERT_NO_FATAL_FAILURE(
            oB.create(bsr_from_csr(Bu), false, rocsparse_direction_row, rocsparse_index_base_zero));
        ASSERT_NO_FATAL_FAILURE(
            oD.create(bsr_from_csr(Du), false, rocsparse_direction_row, rocsparse_index_base_zero));
        if(!sorted_B)
        {
            const rocsparse_storage_mode unsorted = rocsparse_storage_mode_unsorted;
            ASSERT_EQ(rocsparse_spmat_set_attribute(
                          oB.d.p, rocsparse_spmat_storage_mode, &unsorted, sizeof(unsorted)),
                      rocsparse_status_success);
        }

        DevBuf<int32_t> pC(std::vector<int32_t>(static_cast<size_t>(m) + 1, -7));
        SpmatH          C;
        ASSERT_EQ(rocsparse_create_csr_descr(&C.p,
                                             m,
                                             n,
                                             0,
                                             pC.ptr,
                                             nullptr,
                                             nullptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);

        const Scalar<T> alpha(cplx(2.0, 0.0), false), beta(cplx(-1.0, 0.0), false);
        const void*     a   = mul ? alpha.get() : nullptr;
        const void*     b   = add ? beta.get() : nullptr;
        size_t          bsz = 0;
        ASSERT_EQ(spgemm_call<T>(handle,
                                 a,
                                 oA.d.p,
                                 oB.d.p,
                                 b,
                                 oD.d.p,
                                 C.p,
                                 rocsparse_spgemm_stage_buffer_size,
                                 &bsz,
                                 nullptr),
                  rocsparse_status_success);
        DevBuf<char> buffer(bsz);
        ASSERT_TRUE(buffer.ptr);
        const rocsparse_status st = spgemm_call<T>(handle,
                                                   a,
                                                   oA.d.p,
                                                   oB.d.p,
                                                   b,
                                                   oD.d.p,
                                                   C.p,
                                                   rocsparse_spgemm_stage_nnz,
                                                   &bsz,
                                                   buffer.ptr);
        if(!sorted_B && mul)
        {
            // A row of C needs the bitmap algorithm, which requires sorted B.
            ASSERT_EQ(st, rocsparse_status_requires_sorted_storage);
            return;
        }
        ASSERT_EQ(st, rocsparse_status_success);
        int64_t rows = 0, cols = 0, nnz = -1;
        ASSERT_EQ(rocsparse_spmat_get_size(C.p, &rows, &cols, &nnz), rocsparse_status_success);
        ASSERT_EQ(nnz, want.nnzb());
        DevBuf<int32_t> cC(static_cast<size_t>(nnz));
        DevBuf<T>       vC(static_cast<size_t>(nnz));
        ASSERT_TRUE(cC.ptr && vC.ptr);
        ASSERT_EQ(rocsparse_csr_set_pointers(C.p, pC.ptr, cC.ptr, vC.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(spgemm_call<T>(handle,
                                 a,
                                 oA.d.p,
                                 oB.d.p,
                                 b,
                                 oD.d.p,
                                 C.p,
                                 rocsparse_spgemm_stage_compute,
                                 &bsz,
                                 buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        expect_bsr_equal<T, int32_t, int32_t>(
            pC.host(), cC.host(), vC.host(), want, rocsparse_direction_row, 0);
    }
}

TEST_F(ExtraSpgemm, long_rows_sorted_and_unsorted_B)
{
    for(Mode m : kModes)
        for(bool sorted_B : {true, false})
        {
            long_rows<float>(handle, m, sorted_B);
            if(HasFatalFailure())
                return;
        }
}

TEST_F(ExtraSpgemm, mixed_index_bases)
{
    const rocsparse_index_base Z = rocsparse_index_base_zero, O = rocsparse_index_base_one;
    const rocsparse_index_base bases[][4] = {{O, Z, O, Z}, {Z, O, Z, O}, {O, O, O, Z}};
    for(const auto& b : bases)
        for(bool bsr : {false, true})
            for(bool split : {false, true})
            {
                if(split && bsr)
                    continue;
                Case c;
                c.bsr   = bsr;
                c.bd    = 2;
                c.split = split;
                c.bA    = b[0];
                c.bB    = b[1];
                c.bC    = b[2];
                c.bD    = b[3];
                run<float, int32_t, int32_t>(handle, c);
                if(HasFatalFailure())
                    return;
                run<double, int64_t, int32_t>(handle, c);
                if(HasFatalFailure())
                    return;
            }
}

// One descriptor passed as both C and D, which the documentation allows when C
// and D have the same sparsity pattern, i.e. when the pattern of A * B is
// contained in the pattern of D. The nnz stage then has to leave the row
// pointer of D as it is, and compute / numeric have to read the values of D
// before they overwrite them.
namespace
{
    enum class Stages
    {
        nnz_compute,
        compute, // the nnz stage is skipped, the pattern of C being known
        symbolic_numeric
    };

    struct AliasCase
    {
        bool                 bsr    = false;
        rocsparse_direction  dir    = rocsparse_direction_row;
        rocsparse_index_base bAB    = rocsparse_index_base_zero;
        rocsparse_index_base bCD    = rocsparse_index_base_zero;
        cplx                 alpha  = cplx(1.0, 0.0);
        cplx                 beta   = cplx(1.0, 0.0);
        bool                 device = false;
        Stages               stages = Stages::nnz_compute;
    };

    std::string describe(const AliasCase& c)
    {
        std::ostringstream s;
        s << (c.bsr ? "bsr" : "csr") << " dir=" << int(c.dir) << " bases=" << int(c.bAB)
          << int(c.bCD) << " alpha=" << c.alpha << " beta=" << c.beta << " dev=" << c.device
          << " stages=" << int(c.stages);
        return s.str();
    }

    Bsr bsr_of(int64_t                    mb,
               int64_t                    nb,
               int64_t                    bd,
               std::vector<int64_t>       ptr,
               std::vector<int64_t>       col,
               const std::vector<double>& val)
    {
        Bsr M;
        M.mb  = mb;
        M.nb  = nb;
        M.bd  = bd;
        M.ptr = std::move(ptr);
        M.col = std::move(col);
        for(double v : val)
            M.val.push_back(cplx(v, 0.0));
        return M;
    }

    // A D whose pattern is the pattern of A * B plus up to extra_row_nnzb random
    // blocks per row, with random values.
    template <typename T>
    Bsr alias_d(const Bsr& A, const Bsr& B, int64_t extra_row_nnzb, Rng& rng)
    {
        const Bsr P = bsr_gemm(true,
                               cplx(1.0, 0.0),
                               A,
                               B,
                               false,
                               cplx(0.0, 0.0),
                               empty_bsr(A.mb, B.nb, A.bd),
                               A.mb,
                               B.nb);
        const Bsr X = random_bsr<T>(A.mb, B.nb, A.bd, extra_row_nnzb, rng);
        Bsr       D = bsr_geam(cplx(0.0, 0.0), P, cplx(1.0, 0.0), X);
        for(auto& v : D.val)
            v = round_to<T>(rand_val<T>(rng));
        return D;
    }

    template <typename T, typename I, typename J>
    void run_alias(
        rocsparse_handle handle, const AliasCase& c, const Bsr& A, const Bsr& B, const Bsr& D)
    {
        SCOPED_TRACE(describe(c));
        const Bsr want = bsr_gemm(true, c.alpha, A, B, true, c.beta, D, D.mb, D.nb);
        ASSERT_EQ(want.ptr, D.ptr) << "the case does not satisfy the precondition of the alias";
        ASSERT_EQ(want.col, D.col) << "the case does not satisfy the precondition of the alias";

        Operand<T, I, J> oA, oB, oCD;
        ASSERT_NO_FATAL_FAILURE(oA.create(A, c.bsr, c.dir, c.bAB));
        ASSERT_NO_FATAL_FAILURE(oB.create(B, c.bsr, c.dir, c.bAB));
        ASSERT_NO_FATAL_FAILURE(oCD.create(D, c.bsr, c.dir, c.bCD));

        PointerModeGuard mode(handle, mode_of(c.device));
        const Scalar<T>  alpha(c.alpha, c.device), beta(c.beta, c.device);
        auto             call = [&](rocsparse_spgemm_stage stage, size_t* bs, void* tmp) {
            return spgemm_call<T>(
                handle, alpha.get(), oA.d.p, oB.d.p, beta.get(), oCD.d.p, oCD.d.p, stage, bs, tmp);
        };

        size_t bsz = 0;
        ASSERT_EQ(call(rocsparse_spgemm_stage_buffer_size, &bsz, nullptr),
                  rocsparse_status_success);
        DevBuf<char> buffer(bsz);
        ASSERT_TRUE(buffer.ptr);

        if(c.stages != Stages::compute)
        {
            ASSERT_EQ(call(rocsparse_spgemm_stage_nnz, &bsz, buffer.ptr), rocsparse_status_success);
            int64_t rows = 0, cols = 0, nnz = -1;
            ASSERT_EQ(rocsparse_spmat_get_size(oCD.d.p, &rows, &cols, &nnz),
                      rocsparse_status_success);
            ASSERT_EQ(nnz, D.nnzb());
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            // Fatal: with a wrong row pointer, compute writes past the arrays of D.
            const std::vector<I> ptr = oCD.p.host();
            for(size_t i = 0; i < D.ptr.size(); ++i)
                ASSERT_EQ(static_cast<int64_t>(ptr[i]) - c.bCD, D.ptr[i])
                    << "row pointer entry " << i << " after the nnz stage";
        }

        if(c.stages == Stages::symbolic_numeric)
        {
            ASSERT_EQ(call(rocsparse_spgemm_stage_symbolic, &bsz, buffer.ptr),
                      rocsparse_status_success);
            ASSERT_EQ(call(rocsparse_spgemm_stage_numeric, &bsz, buffer.ptr),
                      rocsparse_status_success);
        }
        else
        {
            ASSERT_EQ(call(rocsparse_spgemm_stage_compute, &bsz, buffer.ptr),
                      rocsparse_status_success);
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        expect_bsr_equal<T, I, J>(
            oCD.p.host(), oCD.c.host(), oCD.v.host(), want, c.dir, static_cast<int64_t>(c.bCD));
    }

    const std::vector<Stages> kCsrStages
        = {Stages::nnz_compute, Stages::compute, Stages::symbolic_numeric};
    const std::vector<Stages> kBsrStages = {Stages::nnz_compute, Stages::compute};
}

// pattern(A * B) = {(0,0), (0,2)}; D adds (0,1) and (1,3), so C = D in pattern.
TEST_F(ExtraSpgemm, alias_c_d)
{
    const Bsr A = bsr_of(2, 3, 1, {0, 2, 2}, {0, 1}, {1, 1});
    const Bsr B = bsr_of(3, 4, 1, {0, 2, 3, 3}, {0, 2, 0}, {1, 1, 1});
    const Bsr D = bsr_of(2, 4, 1, {0, 3, 4}, {0, 1, 2, 3}, {1, 1, 1, 1});

    const rocsparse_index_base Z = rocsparse_index_base_zero, O = rocsparse_index_base_one;
    const rocsparse_index_base bases[][2] = {{Z, Z}, {O, O}, {O, Z}, {Z, O}};
    for(Stages s : kCsrStages)
        for(bool dev : {false, true})
            for(const auto& b : bases)
            {
                AliasCase c;
                c.stages = s;
                c.device = dev;
                c.bAB    = b[0];
                c.bCD    = b[1];
                run_alias<float, int32_t, int32_t>(handle, c, A, B, D);
                if(HasFatalFailure())
                    return;
                run_alias<double, int64_t, int32_t>(handle, c, A, B, D);
                if(HasFatalFailure())
                    return;
                run_alias<float, int64_t, int64_t>(handle, c, A, B, D);
                if(HasFatalFailure())
                    return;
            }

    // Zero passed as a non-null scalar still takes the multiply-add path.
    const double scalars[][2] = {{2, -1}, {1, 0}, {0, 1}};
    // D with an empty last row.
    const Bsr D2 = bsr_of(2, 4, 1, {0, 3, 3}, {0, 1, 2}, {1, 2, 3});
    for(Stages s : kCsrStages)
    {
        for(const auto& ab : scalars)
        {
            AliasCase c;
            c.stages = s;
            c.alpha  = cplx(ab[0], 0.0);
            c.beta   = cplx(ab[1], 0.0);
            run_alias<float, int32_t, int32_t>(handle, c, A, B, D);
            if(HasFatalFailure())
                return;
        }
        AliasCase c;
        c.stages = s;
        run_alias<float, int32_t, int32_t>(handle, c, A, B, D2);
        if(HasFatalFailure())
            return;
    }
}

TEST_F(ExtraSpgemm, alias_c_d_random)
{
    const rocsparse_index_base Z = rocsparse_index_base_zero, O = rocsparse_index_base_one;
    const rocsparse_index_base bases[][2] = {{Z, Z}, {O, O}, {O, Z}};
    for(uint64_t seed = 1; seed <= 3; ++seed)
        for(int64_t bd : {1, 2, 3})
        {
            Rng        rng(seed * 131 + static_cast<uint64_t>(bd));
            const Bsr  A   = random_bsr<float>(7, 6, bd, 3, rng);
            const Bsr  B   = random_bsr<float>(6, 9, bd, 3, rng);
            const Bsr  D   = alias_d<float>(A, B, 2, rng);
            const bool bsr = bd > 1;
            for(Stages s : bsr ? kBsrStages : kCsrStages)
                for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
                    for(const auto& b : bases)
                    {
                        if(!bsr && dir == rocsparse_direction_column)
                            continue;
                        AliasCase c;
                        c.bsr    = bsr;
                        c.dir    = dir;
                        c.bAB    = b[0];
                        c.bCD    = b[1];
                        c.alpha  = cplx(2.0, 0.0);
                        c.beta   = cplx(-1.0, 0.0);
                        c.device = (seed == 2);
                        c.stages = s;
                        run_alias<float, int32_t, int32_t>(handle, c, A, B, D);
                        if(HasFatalFailure())
                            return;
                        run_alias<double, int64_t, int64_t>(handle, c, A, B, D);
                        if(HasFatalFailure())
                            return;
                    }
        }
}

// The CSR witness of alias_c_d with 2 x 2 identity blocks.
TEST_F(ExtraSpgemm, alias_c_d_bsr)
{
    auto identity = [](int64_t nnzb) {
        std::vector<double> v;
        for(int64_t p = 0; p < nnzb; ++p)
            v.insert(v.end(), {1, 0, 0, 1});
        return v;
    };
    const Bsr A = bsr_of(2, 3, 2, {0, 2, 2}, {0, 1}, identity(2));
    const Bsr B = bsr_of(3, 4, 2, {0, 2, 3, 3}, {0, 2, 0}, identity(3));
    const Bsr D = bsr_of(2, 4, 2, {0, 3, 4}, {0, 1, 2, 3}, identity(4));
    for(Stages s : kBsrStages)
        for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
        {
            AliasCase c;
            c.bsr    = true;
            c.dir    = dir;
            c.stages = s;
            run_alias<float, int32_t, int32_t>(handle, c, A, B, D);
            if(HasFatalFailure())
                return;
            run_alias<rocsparse_double_complex, int64_t, int32_t>(handle, c, A, B, D);
            if(HasFatalFailure())
                return;
        }
}

// One row of C with n + extra entries: A = [1], B is a row of n ones, D holds
// j % 7 + 1 at the n columns of B and at extra more. A row with more entries
// than the hash table of its group fits in shared memory goes through the
// bitmap algorithm (with 64 KiB and 32-bit column indices, beyond 4096 entries
// for float and 2048 for double complex; on every device beyond 32768; in the
// nnz stage, which counts products and entries of D, beyond 8192 with 64 KiB
// and 65536 on every device); the shorter rows stay in the hash groups.
namespace
{
    template <typename T>
    void alias_long_row(rocsparse_handle handle, int64_t n, int64_t extra, cplx alpha, cplx beta)
    {
        const int64_t        w = n + extra;
        std::vector<int64_t> colB(static_cast<size_t>(n)), colD(static_cast<size_t>(w));
        std::vector<double>  valB(static_cast<size_t>(n), 1.0), valD(static_cast<size_t>(w));
        for(int64_t j = 0; j < w; ++j)
        {
            colD[static_cast<size_t>(j)] = j;
            valD[static_cast<size_t>(j)] = static_cast<double>(j % 7 + 1);
            if(j < n)
                colB[static_cast<size_t>(j)] = j;
        }
        const Bsr A = bsr_of(1, 1, 1, {0, 1}, {0}, {1});
        const Bsr B = bsr_of(1, w, 1, {0, n}, colB, valB);
        const Bsr D = bsr_of(1, w, 1, {0, w}, colD, valD);
        for(Stages s : kCsrStages)
            for(bool dev : {false, true})
            {
                AliasCase c;
                c.alpha  = alpha;
                c.beta   = beta;
                c.device = dev;
                c.stages = s;
                run_alias<T, int32_t, int32_t>(handle, c, A, B, D);
                if(::testing::Test::HasFatalFailure())
                    return;
            }
    }
}

TEST_F(ExtraSpgemm, alias_c_d_long_row)
{
    const cplx one(1.0, 0.0), two(2.0, 0.0);
    alias_long_row<float>(handle, 1000, 0, one, two);
    if(HasFatalFailure())
        return;
    alias_long_row<float>(handle, 5000, 0, one, two);
    if(HasFatalFailure())
        return;
    alias_long_row<float>(handle, 4000, 1000, one, two);
    if(HasFatalFailure())
        return;
    alias_long_row<float>(handle, 33000, 0, one, two);
    if(HasFatalFailure())
        return;
    alias_long_row<rocsparse_double_complex>(handle, 2048, 0, one, cplx(2.0, -1.0));
    if(HasFatalFailure())
        return;
    alias_long_row<rocsparse_double_complex>(handle, 2049, 0, one, cplx(2.0, -1.0));
}

// D misses (0,2) of the pattern of A * B, so C cannot have the pattern of D: the
// nnz stage rejects the alias and leaves D as it is.
TEST_F(ExtraSpgemm, alias_c_d_pattern_mismatch)
{
    for(int64_t bd : {1, 2})
    {
        const int64_t       bb = bd * bd;
        std::vector<double> blk(static_cast<size_t>(bb), 0.0);
        for(int64_t r = 0; r < bd; ++r)
            blk[static_cast<size_t>(r * bd + r)] = 1.0;
        auto vals = [&](int64_t nnzb) {
            std::vector<double> v;
            for(int64_t p = 0; p < nnzb; ++p)
                v.insert(v.end(), blk.begin(), blk.end());
            return v;
        };
        const Bsr  A   = bsr_of(2, 3, bd, {0, 2, 2}, {0, 1}, vals(2));
        const Bsr  B   = bsr_of(3, 4, bd, {0, 2, 3, 3}, {0, 2, 0}, vals(3));
        const Bsr  D   = bsr_of(2, 4, bd, {0, 2, 3}, {0, 1, 3}, vals(3));
        const bool bsr = bd > 1;
        SCOPED_TRACE(bsr ? "bsr" : "csr");

        Operand<float, int32_t, int32_t> oA, oB, oCD;
        ASSERT_NO_FATAL_FAILURE(
            oA.create(A, bsr, rocsparse_direction_row, rocsparse_index_base_zero));
        ASSERT_NO_FATAL_FAILURE(
            oB.create(B, bsr, rocsparse_direction_row, rocsparse_index_base_zero));
        ASSERT_NO_FATAL_FAILURE(
            oCD.create(D, bsr, rocsparse_direction_row, rocsparse_index_base_zero));

        const float alpha = 1.0f, beta = 1.0f;
        size_t      bsz = 0;
        ASSERT_EQ(spgemm_call<float>(handle,
                                     &alpha,
                                     oA.d.p,
                                     oB.d.p,
                                     &beta,
                                     oCD.d.p,
                                     oCD.d.p,
                                     rocsparse_spgemm_stage_buffer_size,
                                     &bsz,
                                     nullptr),
                  rocsparse_status_success);
        DevBuf<char> buffer(bsz);
        ASSERT_TRUE(buffer.ptr);
        EXPECT_EQ(spgemm_call<float>(handle,
                                     &alpha,
                                     oA.d.p,
                                     oB.d.p,
                                     &beta,
                                     oCD.d.p,
                                     oCD.d.p,
                                     rocsparse_spgemm_stage_nnz,
                                     &bsz,
                                     buffer.ptr),
                  rocsparse_status_invalid_value);

        int64_t rows = 0, cols = 0, nnz = -1;
        ASSERT_EQ(rocsparse_spmat_get_size(oCD.d.p, &rows, &cols, &nnz), rocsparse_status_success);
        EXPECT_EQ(nnz, D.nnzb());
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        expect_idx_equal<int32_t>(oCD.p.host(), D.ptr, 0, "row pointer of D");
    }
}

// Without beta, or with an empty D, D does not take part and the pattern of C
// is not checked against it: C comes out as alpha * A * B. hipSPARSE SpGEMM
// relies on this, passing its C as D with a null beta.
TEST_F(ExtraSpgemm, alias_c_d_without_d)
{
    const Bsr A = bsr_of(2, 3, 1, {0, 2, 2}, {0, 1}, {1, 1});
    const Bsr B = bsr_of(3, 4, 1, {0, 2, 3, 3}, {0, 2, 0}, {1, 2, 3});
    const Bsr want
        = bsr_gemm(true, cplx(2.0, 0.0), A, B, false, cplx(0.0, 0.0), empty_bsr(2, 4, 1), 2, 4);

    Operand<float, int32_t, int32_t> oA, oB;
    ASSERT_NO_FATAL_FAILURE(
        oA.create(A, false, rocsparse_direction_row, rocsparse_index_base_zero));
    ASSERT_NO_FATAL_FAILURE(
        oB.create(B, false, rocsparse_direction_row, rocsparse_index_base_zero));

    const float alpha = 2.0f, beta = 1.0f;
    for(bool null_beta : {true, false})
    {
        SCOPED_TRACE(null_beta ? "beta null" : "D empty");

        // Without beta, the pattern of D is one that A * B does not fit in.
        const Bsr D = null_beta ? bsr_of(2, 4, 1, {0, 1, 1}, {3}, {5}) : empty_bsr(2, 4, 1);
        Operand<float, int32_t, int32_t> oCD;
        ASSERT_NO_FATAL_FAILURE(
            oCD.create(D, false, rocsparse_direction_row, rocsparse_index_base_zero));

        auto call = [&](rocsparse_spgemm_stage stage, size_t* bs, void* tmp) {
            return spgemm_call<float>(handle,
                                      &alpha,
                                      oA.d.p,
                                      oB.d.p,
                                      null_beta ? nullptr : &beta,
                                      oCD.d.p,
                                      oCD.d.p,
                                      stage,
                                      bs,
                                      tmp);
        };

        size_t bsz = 0;
        ASSERT_EQ(call(rocsparse_spgemm_stage_buffer_size, &bsz, nullptr),
                  rocsparse_status_success);
        DevBuf<char> buffer(bsz);
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(call(rocsparse_spgemm_stage_nnz, &bsz, buffer.ptr), rocsparse_status_success);

        int64_t rows = 0, cols = 0, nnz = -1;
        ASSERT_EQ(rocsparse_spmat_get_size(oCD.d.p, &rows, &cols, &nnz), rocsparse_status_success);
        ASSERT_EQ(nnz, want.nnzb());
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        expect_idx_equal<int32_t>(oCD.p.host(), want.ptr, 0, "row pointer of C");

        DevBuf<int32_t> cC(static_cast<size_t>(nnz));
        DevBuf<float>   vC(static_cast<size_t>(nnz));
        ASSERT_TRUE(cC.ptr && vC.ptr);
        ASSERT_EQ(rocsparse_csr_set_pointers(oCD.d.p, oCD.p.ptr, cC.ptr, vC.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(call(rocsparse_spgemm_stage_compute, &bsz, buffer.ptr), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        expect_bsr_equal<float, int32_t, int32_t>(
            oCD.p.host(), cC.host(), vC.host(), want, rocsparse_direction_row, 0);
    }
}

// The arrays of D passed as C through a second descriptor with the other index
// base. A row pointer cannot be in both bases, so every stage rejects the call
// and leaves D as it is, although the patterns match. Without beta, or with an
// empty D, D does not take part and the call is accepted.
TEST_F(ExtraSpgemm, alias_c_d_base_mismatch)
{
    const rocsparse_index_base Z = rocsparse_index_base_zero, O = rocsparse_index_base_one;
    const rocsparse_index_base bases[][2] = {{Z, O}, {O, Z}};
    for(int64_t bd : {1, 2})
        for(const auto& b : bases)
        {
            const int64_t       bb = bd * bd;
            std::vector<double> blk(static_cast<size_t>(bb), 0.0);
            for(int64_t r = 0; r < bd; ++r)
                blk[static_cast<size_t>(r * bd + r)] = 1.0;
            auto vals = [&](int64_t nnzb) {
                std::vector<double> v;
                for(int64_t p = 0; p < nnzb; ++p)
                    v.insert(v.end(), blk.begin(), blk.end());
                return v;
            };
            const Bsr  A   = bsr_of(2, 3, bd, {0, 2, 2}, {0, 1}, vals(2));
            const Bsr  B   = bsr_of(3, 4, bd, {0, 2, 3, 3}, {0, 2, 0}, vals(3));
            const Bsr  D   = bsr_of(2, 4, bd, {0, 3, 4}, {0, 1, 2, 3}, vals(4));
            const bool bsr = bd > 1;
            SCOPED_TRACE(std::string(bsr ? "bsr" : "csr") + " bases=" + std::to_string(int(b[0]))
                         + std::to_string(int(b[1])));

            Operand<float, int32_t, int32_t> oA, oB, oD, oE;
            ASSERT_NO_FATAL_FAILURE(oA.create(A, bsr, rocsparse_direction_row, Z));
            ASSERT_NO_FATAL_FAILURE(oB.create(B, bsr, rocsparse_direction_row, Z));
            ASSERT_NO_FATAL_FAILURE(oD.create(D, bsr, rocsparse_direction_row, b[1]));
            ASSERT_NO_FATAL_FAILURE(
                oE.create(empty_bsr(2, 4, bd), bsr, rocsparse_direction_row, b[1]));

            // C over the arrays of M, in the base of C.
            auto alias = [&](SpmatH& C, const Operand<float, int32_t, int32_t>& M) {
                if(bsr)
                    ASSERT_EQ(rocsparse_create_bsr_descr(&C.p,
                                                         M.h.mb,
                                                         M.h.nb,
                                                         M.h.nnzb(),
                                                         rocsparse_direction_row,
                                                         bd,
                                                         M.p.ptr,
                                                         M.c.ptr,
                                                         M.v.ptr,
                                                         rocsparse_indextype_i32,
                                                         rocsparse_indextype_i32,
                                                         b[0],
                                                         rocsparse_datatype_f32_r),
                              rocsparse_status_success);
                else
                    ASSERT_EQ(rocsparse_create_csr_descr(&C.p,
                                                         M.h.mb,
                                                         M.h.nb,
                                                         M.h.nnzb(),
                                                         M.p.ptr,
                                                         M.c.ptr,
                                                         M.v.ptr,
                                                         rocsparse_indextype_i32,
                                                         rocsparse_indextype_i32,
                                                         b[0],
                                                         rocsparse_datatype_f32_r),
                              rocsparse_status_success);
            };
            SpmatH C, CE;
            ASSERT_NO_FATAL_FAILURE(alias(C, oD));
            ASSERT_NO_FATAL_FAILURE(alias(CE, oE));

            const float alpha = 1.0f, beta = 1.0f;
            auto        call = [&](const void*                 be,
                            rocsparse_const_spmat_descr Dd,
                            rocsparse_spmat_descr       Cd,
                            rocsparse_spgemm_stage      stage,
                            size_t*                     bs,
                            void*                       tmp) {
                return spgemm_call<float>(
                    handle, &alpha, oA.d.p, oB.d.p, be, Dd, Cd, stage, bs, tmp);
            };

            // The buffer of the documented alias, so that a missing check cannot overrun it.
            size_t bsz = 0;
            ASSERT_EQ(
                call(&beta, oD.d.p, oD.d.p, rocsparse_spgemm_stage_buffer_size, &bsz, nullptr),
                rocsparse_status_success);
            DevBuf<char> buffer(bsz);
            ASSERT_TRUE(buffer.ptr);

            const std::vector<int32_t> ptr = oD.p.host(), col = oD.c.host();
            const std::vector<float>   val = oD.v.host();

            std::vector<rocsparse_spgemm_stage> stages = {rocsparse_spgemm_stage_buffer_size,
                                                          rocsparse_spgemm_stage_nnz,
                                                          rocsparse_spgemm_stage_compute};
            if(!bsr)
            {
                stages.push_back(rocsparse_spgemm_stage_symbolic);
                stages.push_back(rocsparse_spgemm_stage_numeric);
            }
            for(rocsparse_spgemm_stage s : stages)
            {
                size_t bs = bsz;
                EXPECT_EQ(call(&beta,
                               oD.d.p,
                               C.p,
                               s,
                               &bs,
                               s == rocsparse_spgemm_stage_buffer_size ? nullptr : buffer.ptr),
                          rocsparse_status_invalid_value)
                    << "stage " << int(s);
            }
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            EXPECT_EQ(oD.p.host(), ptr) << "row pointer of D";
            EXPECT_EQ(oD.c.host(), col) << "column indices of D";
            EXPECT_EQ(oD.v.host(), val) << "values of D";

            size_t bs = 0;
            EXPECT_EQ(call(nullptr, oD.d.p, C.p, rocsparse_spgemm_stage_buffer_size, &bs, nullptr),
                      rocsparse_status_success)
                << "beta null";
            EXPECT_EQ(call(&beta, oE.d.p, CE.p, rocsparse_spgemm_stage_buffer_size, &bs, nullptr),
                      rocsparse_status_success)
                << "D empty";
        }
}
