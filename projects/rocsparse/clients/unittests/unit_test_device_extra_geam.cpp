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
// Addition of sparse matrices (rocsparse_?csrgeam, rocsparse_spgeam): empty
// operands (no rows, no columns, no non-zeros) and missing output arrays. Every successful call is
// checked against the expected (empty or small) result.
//
#include "unit_test_extra_common.hpp"

using namespace extra_ut;

namespace
{
    struct ExtraGeam : public rocsparse_ut::HandleTest
    {
    };

    // CSR arrays of an m x n matrix without entries, plus a (possibly
    // non-empty) matrix when `entries` is set.
    template <typename T>
    struct LegacyOperand
    {
        Csr                   h;
        DevBuf<rocsparse_int> ptr, ind;
        DevBuf<T>             val;
        void                  build(int64_t m, int64_t n, bool entries, Rng& rng)
        {
            h   = entries ? random_csr<T>(m, n, 3, rng)
                          : csr_with_row_lengths<T>(m, n, std::vector<int64_t>(m, 0), rng);
            ptr = up_idx<rocsparse_int>(h.ptr);
            ind = up_idx<rocsparse_int>(h.col);
            val = up_val<T>(h.val);
        }
    };

    template <typename T>
    void legacy_geam(rocsparse_handle handle, int64_t m, int64_t n, bool a_entries, bool b_entries)
    {
        SCOPED_TRACE("m=" + std::to_string(m) + " n=" + std::to_string(n)
                     + " a=" + std::to_string(a_entries) + " b=" + std::to_string(b_entries));
        Rng              rng(static_cast<uint64_t>(m * 17 + n) + 1);
        LegacyOperand<T> A, B;
        A.build(m, n, a_entries, rng);
        B.build(m, n, b_entries, rng);
        ASSERT_TRUE(A.ptr.ptr && A.ind.ptr && A.val.ptr && B.ptr.ptr && B.ind.ptr && B.val.ptr);

        MatDescrH        dA, dB, dC;
        PointerModeGuard mode(handle, rocsparse_pointer_mode_host);
        const T          alpha = from_c<T>(cplx(2.0, 0.0));
        const T          beta  = from_c<T>(cplx(-1.0, 0.0));

        DevBuf<rocsparse_int> ptrC(std::vector<rocsparse_int>(static_cast<size_t>(m) + 1, -7));
        ASSERT_TRUE(ptrC.ptr);
        rocsparse_int nnz_C = -1;
        ASSERT_EQ(rocsparse_csrgeam_nnz(handle,
                                        static_cast<rocsparse_int>(m),
                                        static_cast<rocsparse_int>(n),
                                        dA.d,
                                        static_cast<rocsparse_int>(A.h.nnz()),
                                        A.ptr.ptr,
                                        A.ind.ptr,
                                        dB.d,
                                        static_cast<rocsparse_int>(B.h.nnz()),
                                        B.ptr.ptr,
                                        B.ind.ptr,
                                        dC.d,
                                        ptrC.ptr,
                                        &nnz_C),
                  rocsparse_status_success);

        // Expected result.
        const Bsr want
            = bsr_geam(to_c<T>(alpha), bsr_from_csr(A.h), to_c<T>(beta), bsr_from_csr(B.h));
        ASSERT_EQ(nnz_C, want.nnzb());
        if(m > 0)
        {
            const auto hp = ptrC.host();
            expect_idx_equal<rocsparse_int>(hp, want.ptr, 0, "C row_ptr");
        }

        const T               sentinel = from_c<T>(cplx(-77.0, 0.0));
        DevBuf<rocsparse_int> indC(
            std::vector<rocsparse_int>(static_cast<size_t>(std::max<int64_t>(nnz_C, 1)), -5));
        DevBuf<T> valC(std::vector<T>(static_cast<size_t>(std::max<int64_t>(nnz_C, 1)), sentinel));
        ASSERT_TRUE(indC.ptr && valC.ptr);

        ASSERT_EQ(Api<T>::csrgeam(handle,
                                  static_cast<rocsparse_int>(m),
                                  static_cast<rocsparse_int>(n),
                                  &alpha,
                                  dA.d,
                                  static_cast<rocsparse_int>(A.h.nnz()),
                                  static_cast<const T*>(A.val.ptr),
                                  static_cast<const rocsparse_int*>(A.ptr.ptr),
                                  static_cast<const rocsparse_int*>(A.ind.ptr),
                                  &beta,
                                  dB.d,
                                  static_cast<rocsparse_int>(B.h.nnz()),
                                  static_cast<const T*>(B.val.ptr),
                                  static_cast<const rocsparse_int*>(B.ptr.ptr),
                                  static_cast<const rocsparse_int*>(B.ind.ptr),
                                  dC.d,
                                  valC.ptr,
                                  static_cast<const rocsparse_int*>(ptrC.ptr),
                                  indC.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        if(nnz_C > 0)
        {
            std::vector<rocsparse_int> hc = indC.host();
            std::vector<T>             hv = valC.host();
            sort_rows(ptrC.host(), 0, hc, hv);
            expect_idx_equal<rocsparse_int>(hc, want.col, 0, "C col_ind");
            expect_vec_equal<T>(hv, want.val, "C val");
        }
        else
        {
            // Nothing to write: the output arrays are left alone.
            EXPECT_EQ(indC.host()[0], -5);
            EXPECT_EQ(to_c<T>(valC.host()[0]), cplx(-77.0, 0.0));
        }

        // The output arrays are required as soon as C has entries.
        if(nnz_C > 0)
        {
            EXPECT_EQ(Api<T>::csrgeam(handle,
                                      static_cast<rocsparse_int>(m),
                                      static_cast<rocsparse_int>(n),
                                      &alpha,
                                      dA.d,
                                      static_cast<rocsparse_int>(A.h.nnz()),
                                      static_cast<const T*>(A.val.ptr),
                                      static_cast<const rocsparse_int*>(A.ptr.ptr),
                                      static_cast<const rocsparse_int*>(A.ind.ptr),
                                      &beta,
                                      dB.d,
                                      static_cast<rocsparse_int>(B.h.nnz()),
                                      static_cast<const T*>(B.val.ptr),
                                      static_cast<const rocsparse_int*>(B.ptr.ptr),
                                      static_cast<const rocsparse_int*>(B.ind.ptr),
                                      dC.d,
                                      static_cast<T*>(nullptr),
                                      static_cast<const rocsparse_int*>(ptrC.ptr),
                                      indC.ptr),
                      rocsparse_status_invalid_pointer);
            EXPECT_EQ(Api<T>::csrgeam(handle,
                                      static_cast<rocsparse_int>(m),
                                      static_cast<rocsparse_int>(n),
                                      &alpha,
                                      dA.d,
                                      static_cast<rocsparse_int>(A.h.nnz()),
                                      static_cast<const T*>(A.val.ptr),
                                      static_cast<const rocsparse_int*>(A.ptr.ptr),
                                      static_cast<const rocsparse_int*>(A.ind.ptr),
                                      &beta,
                                      dB.d,
                                      static_cast<rocsparse_int>(B.h.nnz()),
                                      static_cast<const T*>(B.val.ptr),
                                      static_cast<const rocsparse_int*>(B.ptr.ptr),
                                      static_cast<const rocsparse_int*>(B.ind.ptr),
                                      dC.d,
                                      valC.ptr,
                                      static_cast<const rocsparse_int*>(ptrC.ptr),
                                      static_cast<rocsparse_int*>(nullptr)),
                      rocsparse_status_invalid_pointer);
        }
    }
}

TEST_F(ExtraGeam, legacy_csrgeam_empty_and_missing_outputs)
{
    struct Cfg
    {
        int64_t m, n;
        bool    a, b;
    };
    const Cfg cfgs[] = {{0, 5, false, false},
                        {4, 0, false, false},
                        {5, 6, false, false},
                        {5, 6, true, false},
                        {5, 6, false, true},
                        {5, 6, true, true}};
    for(const Cfg& c : cfgs)
    {
        legacy_geam<float>(handle, c.m, c.n, c.a, c.b);
        if(HasFatalFailure())
            return;
        legacy_geam<rocsparse_double_complex>(handle, c.m, c.n, c.a, c.b);
        if(HasFatalFailure())
            return;
    }
}

namespace
{
    using stage_t = rocsparse_spgeam_stage;

    // C = alpha * A + beta * B where A and B have no rows / no columns / no
    // entries: C has no entries. Combined and split stage sequences.
    // C is always supplied up front: rocsparse_spgeam analysis without C and
    // with empty operands writes through a null pointer (library bug, AISPARSE-926).
    template <typename T>
    void empty_spgeam(rocsparse_handle handle, int64_t m, int64_t n, bool split)
    {
        SCOPED_TRACE("m=" + std::to_string(m) + " n=" + std::to_string(n)
                     + " split=" + std::to_string(split));
        const rocsparse_datatype dt = rocsparse_ut::dt_of<T>();
        Rng                      rng(7);
        const Csr                E
            = csr_with_row_lengths<T>(m, n, std::vector<int64_t>(static_cast<size_t>(m), 0), rng);

        DevBuf<int32_t> pA = up_idx<int32_t>(E.ptr), cA = up_idx<int32_t>(E.col);
        DevBuf<int32_t> pB = up_idx<int32_t>(E.ptr), cB = up_idx<int32_t>(E.col);
        DevBuf<T>       vA = up_val<T>(E.val), vB = up_val<T>(E.val);
        ASSERT_TRUE(pA.ptr && cA.ptr && vA.ptr && pB.ptr && cB.ptr && vB.ptr);

        SpmatH sA, sB, sC;
        for(auto* e : {&sA, &sB})
        {
            const bool first = (e == &sA);
            ASSERT_EQ(rocsparse_create_csr_descr(&e->p,
                                                 m,
                                                 n,
                                                 0,
                                                 first ? pA.ptr : pB.ptr,
                                                 first ? cA.ptr : cB.ptr,
                                                 first ? vA.ptr : vB.ptr,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 dt),
                      rocsparse_status_success);
        }

        DevBuf<int32_t> pC(std::vector<int32_t>(static_cast<size_t>(m) + 1, -7)), cC(1);
        DevBuf<T>       vC(1);
        ASSERT_TRUE(pC.ptr && cC.ptr && vC.ptr);
        ASSERT_EQ(rocsparse_create_csr_descr(&sC.p,
                                             m,
                                             n,
                                             0,
                                             pC.ptr,
                                             cC.ptr,
                                             vC.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             dt),
                  rocsparse_status_success);

        PointerModeGuard mode(handle, rocsparse_pointer_mode_host);
        const T          alpha = from_c<T>(cplx(2.0, 0.0));
        const T          beta  = from_c<T>(cplx(-1.0, 0.0));
        SpgeamDescrH     descr;
        ASSERT_NE(descr.d, nullptr);
        spgeam_set_inputs(
            handle, descr.d, rocsparse_ut::dt_of<T>(), rocsparse_ut::dt_of<T>(), &alpha, &beta);
        if(::testing::Test::HasFatalFailure())
            return;

        auto run = [&](stage_t stage) {
            size_t bs = 0;
            ASSERT_EQ(rocsparse_spgeam_buffer_size(
                          handle, descr.d, sA.p, sB.p, sC.p, stage, &bs, nullptr),
                      rocsparse_status_success);
            DevBuf<char> tmp(bs);
            ASSERT_EQ(
                rocsparse_spgeam(
                    handle, descr.d, sA.p, sB.p, sC.p, stage, bs, bs ? tmp.ptr : nullptr, nullptr),
                rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        };
        auto analysis_nnz = [&]() {
            int64_t nnz = -1, r = 0, c = 0;
            ASSERT_EQ(rocsparse_spmat_get_size(sC.p, &r, &c, &nnz), rocsparse_status_success);
            ASSERT_EQ(nnz, 0);
        };

        if(!split)
        {
            run(rocsparse_spgeam_stage_analysis);
            if(::testing::Test::HasFatalFailure())
                return;
            analysis_nnz();
            if(::testing::Test::HasFatalFailure())
                return;
            run(rocsparse_spgeam_stage_compute);
        }
        else
        {
            run(rocsparse_spgeam_stage_symbolic_analysis);
            if(::testing::Test::HasFatalFailure())
                return;
            analysis_nnz();
            if(::testing::Test::HasFatalFailure())
                return;
            run(rocsparse_spgeam_stage_symbolic_compute);
            if(::testing::Test::HasFatalFailure())
                return;
            run(rocsparse_spgeam_stage_numeric_analysis);
            if(::testing::Test::HasFatalFailure())
                return;
            run(rocsparse_spgeam_stage_numeric_compute);
        }
        if(::testing::Test::HasFatalFailure())
            return;
        // C has no entries: every row pointer, each pre-filled with -7, must be
        // overwritten with 0 (the index base).
        EXPECT_EQ(pC.host(), std::vector<int32_t>(static_cast<size_t>(m) + 1, 0));
    }
}

TEST_F(ExtraGeam, spgeam_empty_operands)
{
    struct Dims
    {
        int64_t m, n;
    };
    for(const Dims& d : {Dims{0, 4}, Dims{4, 0}, Dims{6, 5}})
        for(bool split : {false, true})
        {
            empty_spgeam<float>(handle, d.m, d.n, split);
            if(HasFatalFailure())
                return;
            empty_spgeam<rocsparse_double_complex>(handle, d.m, d.n, split);
            if(HasFatalFailure())
                return;
        }
}
