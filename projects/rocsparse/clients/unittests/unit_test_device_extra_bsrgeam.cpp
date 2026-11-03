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
// Device unit tests for library/src/extra/rocsparse_bsrgeam.cpp.
//
// rocsparse_bsrgeam_nnzb + rocsparse_?bsrgeam are run for every block
// dimension class handled by the dispatch (1, 2, 3, 4, <= 8, <= 16, <= 32),
// both block directions, the zero/one index bases of A, B and C, host and
// device scalars. The complete C (row pointer, column indices, all block
// values) is compared against a host reference. The argument checks are
// exercised with one bad argument at a time.
//
#include "unit_test_extra_common.hpp"

using namespace extra_ut;

namespace
{
    struct GeamCase
    {
        int64_t              mb           = 6;
        int64_t              nb           = 5;
        int64_t              bd           = 2;
        rocsparse_direction  dir          = rocsparse_direction_row;
        rocsparse_index_base base_A       = rocsparse_index_base_zero;
        rocsparse_index_base base_B       = rocsparse_index_base_zero;
        rocsparse_index_base base_C       = rocsparse_index_base_zero;
        bool                 device       = false;
        cplx                 alpha        = cplx(2.0, 1.0);
        cplx                 beta         = cplx(-1.0, 0.5);
        int64_t              max_row_nnzb = 3;
        bool                 empty_A      = false;
        bool                 empty_B      = false;
    };

    std::string describe(const GeamCase& c)
    {
        return "mb=" + std::to_string(c.mb) + " nb=" + std::to_string(c.nb)
               + " bd=" + std::to_string(c.bd) + " dir=" + std::to_string(c.dir)
               + " bases=" + std::to_string(c.base_A) + std::to_string(c.base_B)
               + std::to_string(c.base_C) + " dev=" + std::to_string(c.device);
    }

    template <typename T>
    void run_bsrgeam(rocsparse_handle handle, const GeamCase& c)
    {
        SCOPED_TRACE(describe(c));
        Rng rng(static_cast<uint64_t>(c.mb * 7 + c.bd * 31 + c.nb) + 100);

        Bsr A = random_bsr<T>(c.mb, c.nb, c.bd, c.max_row_nnzb, rng);
        Bsr B = random_bsr<T>(c.mb, c.nb, c.bd, c.max_row_nnzb, rng);
        if(c.empty_A)
            A = empty_bsr(c.mb, c.nb, c.bd);
        if(c.empty_B)
            B = empty_bsr(c.mb, c.nb, c.bd);

        const Scalar<T> alpha(c.alpha, c.device), beta(c.beta, c.device);
        const Bsr       want = bsr_geam(to_c<T>(alpha.h), A, to_c<T>(beta.h), B);

        MatDescrH dA(c.base_A), dB(c.base_B), dC(c.base_C);

        DevBuf<rocsparse_int> pA = up_idx<rocsparse_int>(A.ptr, c.base_A);
        DevBuf<rocsparse_int> iA = up_idx<rocsparse_int>(A.col, c.base_A);
        DevBuf<T>             vA = up_raw<T>(bsr_pack<T>(A, c.dir));
        DevBuf<rocsparse_int> pB = up_idx<rocsparse_int>(B.ptr, c.base_B);
        DevBuf<rocsparse_int> iB = up_idx<rocsparse_int>(B.col, c.base_B);
        DevBuf<T>             vB = up_raw<T>(bsr_pack<T>(B, c.dir));
        DevBuf<rocsparse_int> pC(std::vector<rocsparse_int>(static_cast<size_t>(c.mb) + 1, -7));
        ASSERT_TRUE(pA.ptr && iA.ptr && vA.ptr && pB.ptr && iB.ptr && vB.ptr && pC.ptr);

        PointerModeGuard mode(handle, mode_of(c.device));

        // nnzb of C (host or device pointer, following the pointer mode)
        rocsparse_int         nnzb_host = -1;
        DevBuf<rocsparse_int> nnzb_dev(size_t{1});
        ASSERT_EQ(rocsparse_bsrgeam_nnzb(handle,
                                         c.dir,
                                         static_cast<rocsparse_int>(c.mb),
                                         static_cast<rocsparse_int>(c.nb),
                                         static_cast<rocsparse_int>(c.bd),
                                         dA.d,
                                         static_cast<rocsparse_int>(A.nnzb()),
                                         pA.ptr,
                                         iA.ptr,
                                         dB.d,
                                         static_cast<rocsparse_int>(B.nnzb()),
                                         pB.ptr,
                                         iB.ptr,
                                         dC.d,
                                         pC.ptr,
                                         c.device ? nnzb_dev.ptr : &nnzb_host),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        const int64_t nnzb_C = c.device ? nnzb_dev.host()[0] : nnzb_host;
        ASSERT_EQ(nnzb_C, want.nnzb());
        expect_idx_equal<rocsparse_int>(pC.host(), want.ptr, c.base_C, "row_ptr_C");

        DevBuf<rocsparse_int> iC(static_cast<size_t>(nnzb_C));
        DevBuf<T>             vC(static_cast<size_t>(nnzb_C * c.bd * c.bd));
        ASSERT_TRUE(iC.ptr && vC.ptr);
        ASSERT_EQ(Api<T>::bsrgeam(handle,
                                  c.dir,
                                  static_cast<rocsparse_int>(c.mb),
                                  static_cast<rocsparse_int>(c.nb),
                                  static_cast<rocsparse_int>(c.bd),
                                  alpha.typed(),
                                  dA.d,
                                  static_cast<rocsparse_int>(A.nnzb()),
                                  vA.ptr,
                                  pA.ptr,
                                  iA.ptr,
                                  beta.typed(),
                                  dB.d,
                                  static_cast<rocsparse_int>(B.nnzb()),
                                  vB.ptr,
                                  pB.ptr,
                                  iB.ptr,
                                  dC.d,
                                  vC.ptr,
                                  pC.ptr,
                                  iC.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        if(nnzb_C > 0)
        {
            std::vector<rocsparse_int> hc = iC.host();
            std::vector<T>             hv = vC.host();
            sort_rows(pC.host(), c.base_C, hc, hv, c.bd * c.bd);
            expect_idx_equal<rocsparse_int>(hc, want.col, c.base_C, "col_ind_C");
            // expected values in the device layout of the block direction
            const std::vector<T> packed = bsr_pack<T>(want, c.dir);
            std::vector<cplx>    want_dev(packed.size());
            for(size_t i = 0; i < packed.size(); ++i)
                want_dev[i] = to_c<T>(packed[i]);
            expect_vec_equal<T>(hv, want_dev, "val_C");
        }
        // inputs are unchanged
        expect_idx_equal<rocsparse_int>(pA.host(), A.ptr, c.base_A, "row_ptr_A");
        expect_idx_equal<rocsparse_int>(iB.host(), B.col, c.base_B, "col_ind_B");
    }

    template <typename T>
    void run_dims(rocsparse_handle handle, const std::vector<int64_t>& dims, bool all_options)
    {
        for(int64_t bd : dims)
            for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
            {
                GeamCase c;
                c.bd  = bd;
                c.dir = dir;
                run_bsrgeam<T>(handle, c);
                if(::testing::Test::HasFatalFailure())
                    return;
                if(!all_options)
                    continue;
                c.device = true;
                run_bsrgeam<T>(handle, c);
                c.device = false;
                c.base_A = c.base_B = c.base_C = rocsparse_index_base_one;
                run_bsrgeam<T>(handle, c);
                c.base_A = rocsparse_index_base_one;
                c.base_B = rocsparse_index_base_zero;
                c.base_C = rocsparse_index_base_one;
                run_bsrgeam<T>(handle, c);
                c.base_C  = rocsparse_index_base_zero;
                c.empty_B = true;
                run_bsrgeam<T>(handle, c);
                c.empty_B = false;
                c.empty_A = true;
                run_bsrgeam<T>(handle, c);
                if(::testing::Test::HasFatalFailure())
                    return;
            }
    }

    struct ExtraBsrgeam : public rocsparse_ut::HandleTest
    {
    };
}

TEST_F(ExtraBsrgeam, block_dims_float)
{
    run_dims<float>(handle, {1, 2, 3, 4, 5, 8, 9, 16, 17, 32}, true);
}

// Quick returns: C is empty, nothing to compute.
//  - rocsparse_bsrgeam_nnzb reports nnzb_C = 0 (host or device pointer) and
//    fills the row pointer of C with the index base of C (the row pointer has
//    mb + 1 entries, the sentinel value is overwritten),
//  - rocsparse_sbsrgeam returns success and does not touch C at all (row
//    pointer, column indices and values keep their sentinel values).
TEST_F(ExtraBsrgeam, quick_returns)
{
    struct Dim
    {
        rocsparse_int mb, nb;
    };
    // both inputs empty, all rows empty; mb == 0 and nb == 0 (an empty
    // matrix has no entries).
    const Dim dims[] = {{2, 2}, {3, 0}, {0, 2}};

    for(bool device : {false, true})
        for(rocsparse_index_base base : {rocsparse_index_base_zero, rocsparse_index_base_one})
            for(const Dim& d : dims)
            {
                SCOPED_TRACE(testing::Message() << "device=" << device << " base=" << base
                                                << " mb=" << d.mb << " nb=" << d.nb);
                const rocsparse_int sentinel = 7;
                MatDescrH           descr(base);
                // A and B are empty: row pointers hold the index base.
                DevBuf<rocsparse_int> ptr
                    = up_idx<rocsparse_int>(std::vector<int64_t>(size_t(d.mb) + 1, 0), base);
                DevBuf<rocsparse_int> col(std::vector<rocsparse_int>{-5});
                DevBuf<float>         val(std::vector<float>{-5.f});
                DevBuf<rocsparse_int> pC(std::vector<rocsparse_int>(size_t(d.mb) + 1, sentinel));
                DevBuf<rocsparse_int> cC(std::vector<rocsparse_int>{sentinel});
                DevBuf<float>         vC(std::vector<float>{-7.f});
                PointerModeGuard      mode(handle, mode_of(device));
                const Scalar<float>   alpha(cplx(1.0, 0.0), device), beta(cplx(1.0, 0.0), device);

                rocsparse_int         nnzb_host = -1;
                DevBuf<rocsparse_int> nnzb_dev(std::vector<rocsparse_int>{-1});
                ASSERT_EQ(rocsparse_bsrgeam_nnzb(handle,
                                                 rocsparse_direction_row,
                                                 d.mb,
                                                 d.nb,
                                                 2,
                                                 descr.d,
                                                 0,
                                                 ptr.ptr,
                                                 col.ptr,
                                                 descr.d,
                                                 0,
                                                 ptr.ptr,
                                                 col.ptr,
                                                 descr.d,
                                                 pC.ptr,
                                                 device ? nnzb_dev.ptr : &nnzb_host),
                          rocsparse_status_success);
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                EXPECT_EQ(device ? nnzb_dev.host()[0] : nnzb_host, 0);
                expect_idx_equal<rocsparse_int>(pC.host(),
                                                std::vector<int64_t>(size_t(d.mb) + 1, 0),
                                                base,
                                                "row_ptr_C after nnzb");

                // The compute routine has nothing to do and leaves C alone.
                EXPECT_EQ(rocsparse_sbsrgeam(handle,
                                             rocsparse_direction_row,
                                             d.mb,
                                             d.nb,
                                             2,
                                             alpha.typed(),
                                             descr.d,
                                             0,
                                             val.ptr,
                                             ptr.ptr,
                                             col.ptr,
                                             beta.typed(),
                                             descr.d,
                                             0,
                                             val.ptr,
                                             ptr.ptr,
                                             col.ptr,
                                             descr.d,
                                             vC.ptr,
                                             pC.ptr,
                                             cC.ptr),
                          rocsparse_status_success);
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                expect_idx_equal<rocsparse_int>(pC.host(),
                                                std::vector<int64_t>(size_t(d.mb) + 1, 0),
                                                base,
                                                "row_ptr_C after bsrgeam");
                EXPECT_EQ(cC.host()[0], sentinel);
                EXPECT_EQ(vC.host()[0], -7.f);
            }
}

// One invalid argument at a time.
TEST_F(ExtraBsrgeam, argument_errors)
{
    MatDescrH             ok, general_b, sym, unsorted;
    DevBuf<rocsparse_int> ptr(std::vector<rocsparse_int>{0, 1, 2});
    DevBuf<rocsparse_int> col(std::vector<rocsparse_int>{0, 1});
    DevBuf<float>         val(std::vector<float>(8, 1.f));
    DevBuf<rocsparse_int> pC(size_t{3});
    DevBuf<rocsparse_int> cC(size_t{8});
    DevBuf<float>         vC(size_t{32});
    ASSERT_EQ(rocsparse_set_mat_type(sym.d, rocsparse_matrix_type_symmetric),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_storage_mode(unsorted.d, rocsparse_storage_mode_unsorted),
              rocsparse_status_success);
    const float alpha = 1.f, beta = 1.f;

    // Argument positions of rocsparse_sbsrgeam.
    enum
    {
        HANDLE,
        DIR,
        MB,
        NB,
        BD,
        ALPHA,
        DESCR_A,
        NNZB_A,
        VAL_A,
        PTR_A,
        COL_A,
        BETA,
        DESCR_B,
        NNZB_B,
        VAL_B,
        PTR_B,
        COL_B,
        DESCR_C,
        VAL_C,
        PTR_C,
        COL_C
    };
    const rocsparse_int two  = 2;
    const auto          call = make_call(rocsparse_sbsrgeam,
                                handle,
                                rocsparse_direction_row,
                                two,
                                two,
                                two,
                                &alpha,
                                ok.d,
                                two,
                                val.ptr,
                                ptr.ptr,
                                col.ptr,
                                &beta,
                                ok.d,
                                two,
                                val.ptr,
                                ptr.ptr,
                                col.ptr,
                                ok.d,
                                vC.ptr,
                                pC.ptr,
                                cC.ptr);

    // The setup itself is valid: build C row pointers with the nnz routine.
    rocsparse_int nnzb = 0;
    ASSERT_EQ(rocsparse_bsrgeam_nnzb(handle,
                                     rocsparse_direction_row,
                                     2,
                                     2,
                                     2,
                                     ok.d,
                                     2,
                                     ptr.ptr,
                                     col.ptr,
                                     ok.d,
                                     2,
                                     ptr.ptr,
                                     col.ptr,
                                     ok.d,
                                     pC.ptr,
                                     &nnzb),
              rocsparse_status_success);
    ASSERT_EQ(nnzb, 2);
    EXPECT_EQ(call(), rocsparse_status_success);

    const rocsparse_status ptr_err  = rocsparse_status_invalid_pointer;
    const rocsparse_status size_err = rocsparse_status_invalid_size;
    EXPECT_EQ(call.set<HANDLE>(nullptr)(), rocsparse_status_invalid_handle);
    EXPECT_EQ(call.set<DESCR_A>(nullptr)(), ptr_err);
    EXPECT_EQ(call.set<DESCR_B>(nullptr)(), ptr_err);
    EXPECT_EQ(call.set<DESCR_C>(nullptr)(), ptr_err);
    EXPECT_EQ(call.set<DESCR_A>(sym.d)(), rocsparse_status_not_implemented);
    EXPECT_EQ(call.set<DESCR_B>(sym.d)(), rocsparse_status_not_implemented);
    EXPECT_EQ(call.set<DESCR_C>(sym.d)(), rocsparse_status_not_implemented);
    // Not documented: unsorted operands, a bad direction, block_dim == 0.
    EXPECT_TRUE(status_is(call.set<DESCR_A>(unsorted.d)(), any_error));
    EXPECT_TRUE(status_is(call.set<DESCR_B>(unsorted.d)(), any_error));
    EXPECT_TRUE(status_is(call.set<DESCR_C>(unsorted.d)(), any_error));
    EXPECT_TRUE(status_is(call.set<DIR>(static_cast<rocsparse_direction>(77))(), any_error));
    EXPECT_TRUE(status_is(call.set<BD>(0)(), any_error));
    EXPECT_EQ(call.set<MB>(-1)(), size_err);
    EXPECT_EQ(call.set<NB>(-1)(), size_err);
    EXPECT_EQ(call.set<NNZB_A>(-1)(), size_err);
    EXPECT_EQ(call.set<NNZB_B>(-1)(), size_err);
    EXPECT_EQ(call.set<ALPHA>(nullptr)(), ptr_err);
    EXPECT_EQ(call.set<BETA>(nullptr)(), ptr_err);
    EXPECT_EQ(call.set<VAL_A>(nullptr)(), ptr_err);
    EXPECT_EQ(call.set<PTR_A>(nullptr)(), ptr_err);
    EXPECT_EQ(call.set<COL_A>(nullptr)(), ptr_err);
    EXPECT_EQ(call.set<VAL_B>(nullptr)(), ptr_err);
    EXPECT_EQ(call.set<PTR_B>(nullptr)(), ptr_err);
    EXPECT_EQ(call.set<COL_B>(nullptr)(), ptr_err);
    EXPECT_EQ(call.set<PTR_C>(nullptr)(), ptr_err);
    // C values / columns are required once the row pointer says nnzb_C > 0
    EXPECT_EQ(call.set<VAL_C>(nullptr)(), ptr_err);
    EXPECT_EQ(call.set<COL_C>(nullptr)(), ptr_err);
}
