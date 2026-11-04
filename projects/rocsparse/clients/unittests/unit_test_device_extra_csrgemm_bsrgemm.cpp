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
// Device unit tests for the legacy sparse-sparse multiplication entry points
//   rocsparse_?csrgemm_buffer_size / rocsparse_csrgemm_nnz / rocsparse_?csrgemm
//   rocsparse_?bsrgemm_buffer_size / rocsparse_bsrgemm_nnzb / rocsparse_?bsrgemm
// in all three operating modes selected by the scalars:
//   C = alpha * A * B + beta * D   (alpha and beta given)
//   C = alpha * A * B              (beta == nullptr)
//   C = beta * D                   (alpha == nullptr)
//
// For every case the whole of C (row pointer, column indices, block values) is
// compared with a host reference.
//
#include "unit_test_extra_common.hpp"

using namespace extra_ut;

namespace
{
    struct GemmCase
    {
        bool                bsr  = true; // BSR entry points, otherwise CSR (bd == 1)
        Mode                mode = Mode::multadd;
        int64_t             mb   = 6;
        int64_t             nb   = 7;
        int64_t             kb   = 5;
        int64_t             bd   = 2;
        rocsparse_direction dir  = rocsparse_direction_row;
        // Index bases of the matrices A, B, C and D (independent of each other).
        rocsparse_index_base base_A       = rocsparse_index_base_zero;
        rocsparse_index_base base_B       = rocsparse_index_base_zero;
        rocsparse_index_base base_C       = rocsparse_index_base_zero;
        rocsparse_index_base base_D       = rocsparse_index_base_zero;
        bool                 device       = false;
        cplx                 alpha        = cplx(2.0, 1.0);
        cplx                 beta         = cplx(-1.0, 0.5);
        int64_t              max_row_nnzb = 3;
        bool                 empty_A      = false;
        bool                 empty_B      = false;
        bool                 empty_D      = false;
        uint64_t             seed         = 1;
    };

    std::string describe(const GemmCase& c)
    {
        return std::string(c.bsr ? "bsr" : "csr")
               + " mode=" + std::to_string(static_cast<int>(c.mode)) + " mb=" + std::to_string(c.mb)
               + " nb=" + std::to_string(c.nb) + " kb=" + std::to_string(c.kb)
               + " bd=" + std::to_string(c.bd) + " dir=" + std::to_string(c.dir) + " bases="
               + std::to_string(c.base_A) + std::to_string(c.base_B) + std::to_string(c.base_C)
               + std::to_string(c.base_D) + " dev=" + std::to_string(c.device);
    }

    template <typename T>
    void run_gemm(rocsparse_handle handle, const GemmCase& c)
    {
        SCOPED_TRACE(describe(c));
        const rocsparse_operation N   = rocsparse_operation_none;
        const bool                mul = (c.mode != Mode::add);
        const bool                add = (c.mode != Mode::mult);
        const int64_t             bd  = c.bsr ? c.bd : 1;

        Rng rng(c.seed * 977 + static_cast<uint64_t>(c.mb * 13 + c.nb * 7 + c.kb + bd));

        Bsr A = random_bsr<T>(c.mb, c.kb, bd, c.max_row_nnzb, rng);
        Bsr B = random_bsr<T>(c.kb, c.nb, bd, c.max_row_nnzb, rng);
        Bsr D = random_bsr<T>(c.mb, c.nb, bd, c.max_row_nnzb, rng);
        if(c.empty_A)
            A = empty_bsr(c.mb, c.kb, bd);
        if(c.empty_B)
            B = empty_bsr(c.kb, c.nb, bd);
        if(c.empty_D)
            D = empty_bsr(c.mb, c.nb, bd);
        // The unused operands are not passed at all.
        const Bsr Ause = mul ? A : empty_bsr(c.mb, c.kb, bd);
        const Bsr Buse = mul ? B : empty_bsr(c.kb, c.nb, bd);
        const Bsr Duse = add ? D : empty_bsr(c.mb, c.nb, bd);

        const Scalar<T> alpha(c.alpha, c.device), beta(c.beta, c.device);
        const Bsr       want
            = bsr_gemm(mul, to_c<T>(alpha.h), Ause, Buse, add, to_c<T>(beta.h), Duse, c.mb, c.nb);

        MatDescrH dA(c.base_A), dB(c.base_B), dC(c.base_C), dD(c.base_D);
        MatInfoH  info;

        // Device operands. Unused operands are passed as null pointers.
        DevBuf<rocsparse_int> pA, iA, pB, iB, pD, iD;
        DevBuf<T>             vA, vB, vD;
        if(mul)
        {
            pA = up_idx<rocsparse_int>(A.ptr, c.base_A);
            iA = up_idx<rocsparse_int>(A.col, c.base_A);
            vA = up_raw<T>(bsr_pack<T>(A, c.dir));
            pB = up_idx<rocsparse_int>(B.ptr, c.base_B);
            iB = up_idx<rocsparse_int>(B.col, c.base_B);
            vB = up_raw<T>(bsr_pack<T>(B, c.dir));
        }
        if(add)
        {
            pD = up_idx<rocsparse_int>(D.ptr, c.base_D);
            iD = up_idx<rocsparse_int>(D.col, c.base_D);
            vD = up_raw<T>(bsr_pack<T>(D, c.dir));
        }
        DevBuf<rocsparse_int> pC(std::vector<rocsparse_int>(static_cast<size_t>(c.mb) + 1, -7));
        ASSERT_TRUE(pC.ptr);

        PointerModeGuard mode(handle, mode_of(c.device));

        const T* a_ptr = mul ? alpha.typed() : nullptr;
        const T* b_ptr = add ? beta.typed() : nullptr;

        const rocsparse_int mb = static_cast<rocsparse_int>(c.mb);
        const rocsparse_int nb = static_cast<rocsparse_int>(c.nb);
        const rocsparse_int kb = static_cast<rocsparse_int>(c.kb);
        const rocsparse_int bs = static_cast<rocsparse_int>(bd);
        const rocsparse_int nA = mul ? static_cast<rocsparse_int>(A.nnzb()) : 0;
        const rocsparse_int nB = mul ? static_cast<rocsparse_int>(B.nnzb()) : 0;
        const rocsparse_int nD = add ? static_cast<rocsparse_int>(D.nnzb()) : 0;

        size_t bsz = 0;
        if(c.bsr)
        {
            ASSERT_EQ(Api<T>::bsrgemm_buffer_size(handle,
                                                  c.dir,
                                                  N,
                                                  N,
                                                  mb,
                                                  nb,
                                                  kb,
                                                  bs,
                                                  a_ptr,
                                                  dA.d,
                                                  nA,
                                                  pA.ptr,
                                                  iA.ptr,
                                                  dB.d,
                                                  nB,
                                                  pB.ptr,
                                                  iB.ptr,
                                                  b_ptr,
                                                  dD.d,
                                                  nD,
                                                  pD.ptr,
                                                  iD.ptr,
                                                  info.i,
                                                  &bsz),
                      rocsparse_status_success);
        }
        else
        {
            ASSERT_EQ(Api<T>::csrgemm_buffer_size(handle,
                                                  N,
                                                  N,
                                                  mb,
                                                  nb,
                                                  kb,
                                                  a_ptr,
                                                  dA.d,
                                                  nA,
                                                  pA.ptr,
                                                  iA.ptr,
                                                  dB.d,
                                                  nB,
                                                  pB.ptr,
                                                  iB.ptr,
                                                  b_ptr,
                                                  dD.d,
                                                  nD,
                                                  pD.ptr,
                                                  iD.ptr,
                                                  info.i,
                                                  &bsz),
                      rocsparse_status_success);
        }
        DevBuf<char> buffer(bsz);
        ASSERT_TRUE(buffer.ptr);

        rocsparse_int         nnzb_host = -1;
        DevBuf<rocsparse_int> nnzb_dev(size_t{1});
        rocsparse_int*        nnzb_ptr = c.device ? nnzb_dev.ptr : &nnzb_host;
        if(c.bsr)
        {
            ASSERT_EQ(rocsparse_bsrgemm_nnzb(handle,
                                             c.dir,
                                             N,
                                             N,
                                             mb,
                                             nb,
                                             kb,
                                             bs,
                                             dA.d,
                                             nA,
                                             pA.ptr,
                                             iA.ptr,
                                             dB.d,
                                             nB,
                                             pB.ptr,
                                             iB.ptr,
                                             dD.d,
                                             nD,
                                             pD.ptr,
                                             iD.ptr,
                                             dC.d,
                                             pC.ptr,
                                             nnzb_ptr,
                                             info.i,
                                             buffer.ptr),
                      rocsparse_status_success);
        }
        else
        {
            ASSERT_EQ(rocsparse_csrgemm_nnz(handle,
                                            N,
                                            N,
                                            mb,
                                            nb,
                                            kb,
                                            dA.d,
                                            nA,
                                            pA.ptr,
                                            iA.ptr,
                                            dB.d,
                                            nB,
                                            pB.ptr,
                                            iB.ptr,
                                            dD.d,
                                            nD,
                                            pD.ptr,
                                            iD.ptr,
                                            dC.d,
                                            pC.ptr,
                                            nnzb_ptr,
                                            info.i,
                                            buffer.ptr),
                      rocsparse_status_success);
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        const int64_t nnzb_C = c.device ? nnzb_dev.host()[0] : nnzb_host;
        ASSERT_EQ(nnzb_C, want.nnzb());

        DevBuf<rocsparse_int> iC(static_cast<size_t>(nnzb_C));
        DevBuf<T>             vC(static_cast<size_t>(nnzb_C * bd * bd));
        ASSERT_TRUE(iC.ptr && vC.ptr);
        if(c.bsr)
        {
            ASSERT_EQ(Api<T>::bsrgemm(handle,
                                      c.dir,
                                      N,
                                      N,
                                      mb,
                                      nb,
                                      kb,
                                      bs,
                                      a_ptr,
                                      dA.d,
                                      nA,
                                      vA.ptr,
                                      pA.ptr,
                                      iA.ptr,
                                      dB.d,
                                      nB,
                                      vB.ptr,
                                      pB.ptr,
                                      iB.ptr,
                                      b_ptr,
                                      dD.d,
                                      nD,
                                      vD.ptr,
                                      pD.ptr,
                                      iD.ptr,
                                      dC.d,
                                      vC.ptr,
                                      pC.ptr,
                                      iC.ptr,
                                      info.i,
                                      buffer.ptr),
                      rocsparse_status_success);
        }
        else
        {
            ASSERT_EQ(Api<T>::csrgemm(handle,
                                      N,
                                      N,
                                      mb,
                                      nb,
                                      kb,
                                      a_ptr,
                                      dA.d,
                                      nA,
                                      vA.ptr,
                                      pA.ptr,
                                      iA.ptr,
                                      dB.d,
                                      nB,
                                      vB.ptr,
                                      pB.ptr,
                                      iB.ptr,
                                      b_ptr,
                                      dD.d,
                                      nD,
                                      vD.ptr,
                                      pD.ptr,
                                      iD.ptr,
                                      dC.d,
                                      vC.ptr,
                                      pC.ptr,
                                      iC.ptr,
                                      info.i,
                                      buffer.ptr),
                      rocsparse_status_success);
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        expect_bsr_equal<T>(pC.host(), iC.host(), vC.host(), want, c.dir, c.base_C, !c.bsr);
    }

    template <typename T>
    void run_modes(rocsparse_handle handle, bool bsr, const std::vector<int64_t>& dims, bool full)
    {
        for(int64_t bd : dims)
            for(Mode m : {Mode::multadd, Mode::mult, Mode::add})
                for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
                {
                    if(!bsr && dir == rocsparse_direction_column)
                        continue;
                    GemmCase c;
                    c.bsr  = bsr;
                    c.bd   = bd;
                    c.mode = m;
                    c.dir  = dir;
                    run_gemm<T>(handle, c);
                    if(::testing::Test::HasFatalFailure())
                        return;
                    if(!full)
                        continue;
                    c.device = true;
                    run_gemm<T>(handle, c);
                    c.device = false;
                    // all one based, then every base different from its
                    // neighbours (mixed bases)
                    c.base_A = c.base_B = c.base_C = c.base_D = rocsparse_index_base_one;
                    run_gemm<T>(handle, c);
                    c.base_A = rocsparse_index_base_one;
                    c.base_B = rocsparse_index_base_zero;
                    c.base_C = rocsparse_index_base_one;
                    c.base_D = rocsparse_index_base_zero;
                    run_gemm<T>(handle, c);
                    c.base_A = rocsparse_index_base_zero;
                    c.base_B = rocsparse_index_base_one;
                    c.base_C = rocsparse_index_base_zero;
                    c.base_D = rocsparse_index_base_one;
                    run_gemm<T>(handle, c);
                    c.base_A = c.base_B = c.base_C = c.base_D = rocsparse_index_base_zero;
                    c.empty_A                                 = true;
                    run_gemm<T>(handle, c);
                    c.empty_A = false;
                    c.empty_B = true;
                    run_gemm<T>(handle, c);
                    c.empty_B = false;
                    c.empty_D = true;
                    run_gemm<T>(handle, c);
                    c.empty_D = false;
                    c.kb      = 0; // inner dimension is empty
                    run_gemm<T>(handle, c);
                    c.kb = 5;
                    c.mb = 0; // no rows
                    run_gemm<T>(handle, c);
                    c.mb = 6;
                    c.nb = 0; // no columns
                    run_gemm<T>(handle, c);
                    if(::testing::Test::HasFatalFailure())
                        return;
                }
    }

    struct ExtraGemm : public rocsparse_ut::HandleTest
    {
    };
}

TEST_F(ExtraGemm, csrgemm_modes_float)
{
    run_modes<float>(handle, false, {1}, true);
}

TEST_F(ExtraGemm, bsrgemm_modes_float)
{
    // Every block dimension class with the default options; the scalar
    // pointer mode, index base, empty operand and empty dimension variants
    // (which do not depend on the block dimension class) for three of them.
    run_modes<float>(handle, true, {1, 2, 3, 4, 5, 8, 9, 16, 17, 32}, false);
    run_modes<float>(handle, true, {2, 5, 16}, true);
}

// ---------------------------------------------------------------------------
// Argument checks of the legacy csrgemm / bsrgemm entry points (buffer size
// and compute), one bad argument at a time, for every mode of the operation
// (multadd, mult, add). The unmodified call has to succeed and gives the
// right C before and after the failing calls.
// ---------------------------------------------------------------------------
namespace
{
    struct GemmArgs
    {
        bool                 bsr     = true;
        rocsparse_handle     handle  = nullptr;
        rocsparse_direction  dir     = rocsparse_direction_row;
        rocsparse_operation  trans_A = rocsparse_operation_none;
        rocsparse_operation  trans_B = rocsparse_operation_none;
        rocsparse_int        mb = 0, nb = 0, kb = 0, bd = 1;
        const float *        alpha = nullptr, *beta = nullptr;
        rocsparse_mat_descr  dA = nullptr, dB = nullptr, dC = nullptr, dD = nullptr;
        rocsparse_int        nA = 0, nB = 0, nD = 0;
        const float *        vA = nullptr, *vB = nullptr, *vD = nullptr;
        const rocsparse_int *pA = nullptr, *iA = nullptr, *pB = nullptr, *iB = nullptr,
                            *pD = nullptr, *iD = nullptr;
        float*               vC     = nullptr;
        const rocsparse_int* pC     = nullptr;
        rocsparse_int*       iC     = nullptr;
        rocsparse_mat_info   info   = nullptr;
        void*                buffer = nullptr;
    };

    rocsparse_status gemm_buffer_size(const GemmArgs& g, size_t* size)
    {
        if(g.bsr)
            return rocsparse_sbsrgemm_buffer_size(g.handle,
                                                  g.dir,
                                                  g.trans_A,
                                                  g.trans_B,
                                                  g.mb,
                                                  g.nb,
                                                  g.kb,
                                                  g.bd,
                                                  g.alpha,
                                                  g.dA,
                                                  g.nA,
                                                  g.pA,
                                                  g.iA,
                                                  g.dB,
                                                  g.nB,
                                                  g.pB,
                                                  g.iB,
                                                  g.beta,
                                                  g.dD,
                                                  g.nD,
                                                  g.pD,
                                                  g.iD,
                                                  g.info,
                                                  size);
        return rocsparse_scsrgemm_buffer_size(g.handle,
                                              g.trans_A,
                                              g.trans_B,
                                              g.mb,
                                              g.nb,
                                              g.kb,
                                              g.alpha,
                                              g.dA,
                                              g.nA,
                                              g.pA,
                                              g.iA,
                                              g.dB,
                                              g.nB,
                                              g.pB,
                                              g.iB,
                                              g.beta,
                                              g.dD,
                                              g.nD,
                                              g.pD,
                                              g.iD,
                                              g.info,
                                              size);
    }

    rocsparse_status gemm_nnz(const GemmArgs& g, rocsparse_int* nnz)
    {
        if(g.bsr)
            return rocsparse_bsrgemm_nnzb(g.handle,
                                          g.dir,
                                          g.trans_A,
                                          g.trans_B,
                                          g.mb,
                                          g.nb,
                                          g.kb,
                                          g.bd,
                                          g.dA,
                                          g.nA,
                                          g.pA,
                                          g.iA,
                                          g.dB,
                                          g.nB,
                                          g.pB,
                                          g.iB,
                                          g.dD,
                                          g.nD,
                                          g.pD,
                                          g.iD,
                                          g.dC,
                                          g.pC ? const_cast<rocsparse_int*>(g.pC) : nullptr,
                                          nnz,
                                          g.info,
                                          g.buffer);
        return rocsparse_csrgemm_nnz(g.handle,
                                     g.trans_A,
                                     g.trans_B,
                                     g.mb,
                                     g.nb,
                                     g.kb,
                                     g.dA,
                                     g.nA,
                                     g.pA,
                                     g.iA,
                                     g.dB,
                                     g.nB,
                                     g.pB,
                                     g.iB,
                                     g.dD,
                                     g.nD,
                                     g.pD,
                                     g.iD,
                                     g.dC,
                                     g.pC ? const_cast<rocsparse_int*>(g.pC) : nullptr,
                                     nnz,
                                     g.info,
                                     g.buffer);
    }

    rocsparse_status gemm_compute(const GemmArgs& g)
    {
        if(g.bsr)
            return rocsparse_sbsrgemm(g.handle,
                                      g.dir,
                                      g.trans_A,
                                      g.trans_B,
                                      g.mb,
                                      g.nb,
                                      g.kb,
                                      g.bd,
                                      g.alpha,
                                      g.dA,
                                      g.nA,
                                      g.vA,
                                      g.pA,
                                      g.iA,
                                      g.dB,
                                      g.nB,
                                      g.vB,
                                      g.pB,
                                      g.iB,
                                      g.beta,
                                      g.dD,
                                      g.nD,
                                      g.vD,
                                      g.pD,
                                      g.iD,
                                      g.dC,
                                      g.vC,
                                      g.pC,
                                      g.iC,
                                      g.info,
                                      g.buffer);
        return rocsparse_scsrgemm(g.handle,
                                  g.trans_A,
                                  g.trans_B,
                                  g.mb,
                                  g.nb,
                                  g.kb,
                                  g.alpha,
                                  g.dA,
                                  g.nA,
                                  g.vA,
                                  g.pA,
                                  g.iA,
                                  g.dB,
                                  g.nB,
                                  g.vB,
                                  g.pB,
                                  g.iB,
                                  g.beta,
                                  g.dD,
                                  g.nD,
                                  g.vD,
                                  g.pD,
                                  g.iD,
                                  g.dC,
                                  g.vC,
                                  g.pC,
                                  g.iC,
                                  g.info,
                                  g.buffer);
    }

    void gemm_argument_errors(rocsparse_handle handle, bool bsr, Mode mode)
    {
        const bool    mul = (mode != Mode::add);
        const bool    add = (mode != Mode::mult);
        const int64_t bd  = bsr ? 3 : 1;
        const int64_t mb = 5, nb = 6, kb = 4;
        SCOPED_TRACE(std::string(bsr ? "bsr" : "csr")
                     + " mode=" + std::to_string(static_cast<int>(mode)));

        Rng        rng(40 + static_cast<uint64_t>(mode) + (bsr ? 10 : 0));
        const Bsr  A  = random_bsr<float>(mb, kb, bd, 3, rng);
        const Bsr  B  = random_bsr<float>(kb, nb, bd, 3, rng);
        const Bsr  D  = random_bsr<float>(mb, nb, bd, 3, rng);
        const Bsr  Z  = empty_bsr(mb, nb, bd);
        const Bsr  Za = empty_bsr(mb, kb, bd);
        const Bsr  Zb = empty_bsr(kb, nb, bd);
        const cplx alpha_c(2.0, 0.0), beta_c(-1.0, 0.0);
        const Bsr  want
            = bsr_gemm(mul, alpha_c, mul ? A : Za, mul ? B : Zb, add, beta_c, add ? D : Z, mb, nb);

        const rocsparse_direction dir = rocsparse_direction_row;
        DevBuf<rocsparse_int> pA = up_idx<rocsparse_int>(A.ptr), iA = up_idx<rocsparse_int>(A.col);
        DevBuf<rocsparse_int> pB = up_idx<rocsparse_int>(B.ptr), iB = up_idx<rocsparse_int>(B.col);
        DevBuf<rocsparse_int> pD = up_idx<rocsparse_int>(D.ptr), iD = up_idx<rocsparse_int>(D.col);
        DevBuf<float>         vA = up_raw<float>(bsr_pack<float>(A, dir));
        DevBuf<float>         vB = up_raw<float>(bsr_pack<float>(B, dir));
        DevBuf<float>         vD = up_raw<float>(bsr_pack<float>(D, dir));
        DevBuf<rocsparse_int> pC(std::vector<rocsparse_int>(static_cast<size_t>(mb) + 1, -7));
        MatDescrH             dA, dB, dC, dD;
        MatInfoH              info;
        const float           alpha = 2.0f, beta = -1.0f;

        GemmArgs g;
        g.bsr    = bsr;
        g.handle = handle;
        g.dir    = dir;
        g.mb     = static_cast<rocsparse_int>(mb);
        g.nb     = static_cast<rocsparse_int>(nb);
        g.kb     = static_cast<rocsparse_int>(kb);
        g.bd     = static_cast<rocsparse_int>(bd);
        g.dA     = dA.d;
        g.dB     = dB.d;
        g.dC     = dC.d;
        g.dD     = dD.d;
        g.info   = info.i;
        g.pC     = pC.ptr;
        if(mul)
        {
            g.alpha = &alpha;
            g.nA    = static_cast<rocsparse_int>(A.nnzb());
            g.nB    = static_cast<rocsparse_int>(B.nnzb());
            g.vA    = vA.ptr;
            g.pA    = pA.ptr;
            g.iA    = iA.ptr;
            g.vB    = vB.ptr;
            g.pB    = pB.ptr;
            g.iB    = iB.ptr;
        }
        if(add)
        {
            g.beta = &beta;
            g.nD   = static_cast<rocsparse_int>(D.nnzb());
            g.vD   = vD.ptr;
            g.pD   = pD.ptr;
            g.iD   = iD.ptr;
        }

        size_t bsz = 0;
        ASSERT_EQ(gemm_buffer_size(g, &bsz), rocsparse_status_success);
        DevBuf<char> buffer(bsz);
        ASSERT_TRUE(buffer.ptr);
        g.buffer = buffer.ptr;

        rocsparse_int nnzb = -1;
        ASSERT_EQ(gemm_nnz(g, &nnzb), rocsparse_status_success);
        ASSERT_EQ(static_cast<int64_t>(nnzb), want.nnzb());
        DevBuf<rocsparse_int> iC(static_cast<size_t>(nnzb));
        DevBuf<float>         vC(static_cast<size_t>(nnzb * bd * bd));
        ASSERT_TRUE(iC.ptr && vC.ptr);
        g.iC = iC.ptr;
        g.vC = vC.ptr;

        auto expect_status = [&](const char* what, Expected expected, auto&& mutate) {
            GemmArgs bad = g;
            mutate(bad);
            EXPECT_TRUE(status_is(gemm_compute(bad), expected)) << "compute: " << what;
        };
        auto expect_bs_status = [&](const char* what, Expected expected, auto&& mutate) {
            GemmArgs bad = g;
            mutate(bad);
            size_t size = 0;
            EXPECT_TRUE(status_is(gemm_buffer_size(bad, &size), expected))
                << "buffer size: " << what;
        };

        const auto inv_h = rocsparse_status_invalid_handle;
        const auto inv_p = rocsparse_status_invalid_pointer;
        const auto inv_s = rocsparse_status_invalid_size;
        const auto not_i = rocsparse_status_not_implemented;

        // ---- both entry points ----
        // For multadd, a null alpha or beta selects the other mode in the buffer
        // size call; in the add mode the operands A and B are not used at all.
        auto common = [&](auto&& check, bool buffer_size_call) {
            check("handle", inv_h, [](GemmArgs& a) { a.handle = nullptr; });
            check("mb", inv_s, [](GemmArgs& a) { a.mb = -1; });
            check("nb", inv_s, [](GemmArgs& a) { a.nb = -1; });
            if(mul)
            {
                check("trans_A", not_i, [](GemmArgs& a) {
                    a.trans_A = rocsparse_operation_transpose;
                });
                check("trans_B", not_i, [](GemmArgs& a) {
                    a.trans_B = rocsparse_operation_transpose;
                });
                check("kb", inv_s, [](GemmArgs& a) { a.kb = -1; });
                if(!(buffer_size_call && mode == Mode::multadd))
                    check("alpha", inv_p, [](GemmArgs& a) { a.alpha = nullptr; });
                check("descr_A", inv_p, [](GemmArgs& a) { a.dA = nullptr; });
                check("descr_B", inv_p, [](GemmArgs& a) { a.dB = nullptr; });
                check("nnz_A", inv_s, [](GemmArgs& a) { a.nA = -1; });
                check("nnz_B", inv_s, [](GemmArgs& a) { a.nB = -1; });
                check("row_ptr_A", inv_p, [](GemmArgs& a) { a.pA = nullptr; });
                check("col_ind_A", inv_p, [](GemmArgs& a) { a.iA = nullptr; });
                check("row_ptr_B", inv_p, [](GemmArgs& a) { a.pB = nullptr; });
                check("col_ind_B", inv_p, [](GemmArgs& a) { a.iB = nullptr; });
            }
            if(add)
            {
                if(!(buffer_size_call && mode == Mode::multadd))
                    check("beta", inv_p, [](GemmArgs& a) { a.beta = nullptr; });
                check("descr_D", inv_p, [](GemmArgs& a) { a.dD = nullptr; });
                check("nnz_D", inv_s, [](GemmArgs& a) { a.nD = -1; });
                check("row_ptr_D", inv_p, [](GemmArgs& a) { a.pD = nullptr; });
                check("col_ind_D", inv_p, [](GemmArgs& a) { a.iD = nullptr; });
            }
            check("info", inv_p, [](GemmArgs& a) { a.info = nullptr; });
        };
        common(expect_bs_status, true);
        common(expect_status, false);
        if(bsr)
        {
            expect_bs_status("block_dim", inv_s, [](GemmArgs& a) { a.bd = 0; });
            expect_status("block_dim", inv_s, [](GemmArgs& a) { a.bd = 0; });
            // A bad direction is not documented: any error.
            const auto bad_dir = static_cast<rocsparse_direction>(99);
            expect_bs_status("direction", any_error, [&](GemmArgs& a) { a.dir = bad_dir; });
            expect_status("direction", any_error, [&](GemmArgs& a) { a.dir = bad_dir; });
        }

        // ---- compute only ----
        expect_status("val_A", inv_p, [&](GemmArgs& a) {
            if(mul)
                a.vA = nullptr;
            else
                a.vD = nullptr;
        });
        if(bsr || mode != Mode::add) // the csrgemm scaling needs no buffer
            expect_status("temp_buffer", inv_p, [](GemmArgs& a) { a.buffer = nullptr; });
        expect_status("row_ptr_C", inv_p, [](GemmArgs& a) { a.pC = nullptr; });
        // C values / column indices missing while the row pointer announces entries.
        expect_status("val_C", inv_p, [](GemmArgs& a) { a.vC = nullptr; });
        expect_status("col_ind_C", inv_p, [](GemmArgs& a) { a.iC = nullptr; });
        expect_status("descr_C", any_error, [](GemmArgs& a) { a.dC = nullptr; });
        {
            MatDescrH sym;
            ASSERT_EQ(rocsparse_set_mat_type(sym.d, rocsparse_matrix_type_symmetric),
                      rocsparse_status_success);
            for(int which = 0; which < 4; ++which)
            {
                const char* names[] = {"type A", "type B", "type C", "type D"};
                if((which < 2 && !mul) || (which == 3 && !add))
                    continue;
                expect_status(names[which], not_i, [&](GemmArgs& a) {
                    (which == 0 ? a.dA : which == 1 ? a.dB : which == 2 ? a.dC : a.dD) = sym.d;
                });
            }
        }

        // The unmodified call still works and computes the right C.
        ASSERT_EQ(gemm_compute(g), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        expect_bsr_equal<float>(pC.host(), iC.host(), vC.host(), want, dir, 0, !bsr);
    }
}

TEST_F(ExtraGemm, argument_errors)
{
    for(bool bsr : {true, false})
        for(Mode m : {Mode::multadd, Mode::mult, Mode::add})
        {
            gemm_argument_errors(handle, bsr, m);
            if(HasFatalFailure())
                return;
        }
}
