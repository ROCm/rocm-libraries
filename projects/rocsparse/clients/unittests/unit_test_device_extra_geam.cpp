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
// operands (no rows, no columns, no non-zeros), missing output arrays, and rows that
// fill whole chunks of the csrgeam fill kernel. Every successful call is checked
// against the expected (empty or small) result.
//
#include "unit_test_extra_common.hpp"

#include "rocsparse_spgeam_descr.hpp"

#include <optional>

using namespace extra_ut;

namespace
{
    struct ExtraGeam : public rocsparse_ut::HandleTest
    {
    };

    // C = alpha * A + beta * B through rocsparse_csrgeam_nnz and rocsparse_?csrgeam, checked
    // against the host reference bsr_geam. A and B are zero based, C has the base base_C.
    // The output arrays start filled with sentinels, so an entry the kernel does not place
    // shows. With check_missing_outputs the compute call is repeated with a null value
    // array and a null column array, which are rejected as soon as C has entries.
    template <typename T>
    void run_legacy_geam(rocsparse_handle handle,
                         const Csr&       A,
                         const Csr&       B,
                         cplx             alpha_c,
                         cplx             beta_c,
                         int              base_C,
                         bool             check_missing_outputs)
    {
        const int64_t m = A.m, n = A.n;
        ASSERT_EQ(B.m, m);
        ASSERT_EQ(B.n, n);
        DevBuf<rocsparse_int> pA = up_idx<rocsparse_int>(A.ptr), iA = up_idx<rocsparse_int>(A.col);
        DevBuf<rocsparse_int> pB = up_idx<rocsparse_int>(B.ptr), iB = up_idx<rocsparse_int>(B.col);
        DevBuf<T>             vA = up_val<T>(A.val), vB = up_val<T>(B.val);
        ASSERT_TRUE(pA.ptr && iA.ptr && vA.ptr && pB.ptr && iB.ptr && vB.ptr);

        MatDescrH        dA, dB, dC(base_C ? rocsparse_index_base_one : rocsparse_index_base_zero);
        PointerModeGuard mode(handle, rocsparse_pointer_mode_host);
        const T          alpha = from_c<T>(alpha_c);
        const T          beta  = from_c<T>(beta_c);

        DevBuf<rocsparse_int> ptrC(std::vector<rocsparse_int>(static_cast<size_t>(m) + 1, -7));
        ASSERT_TRUE(ptrC.ptr);
        rocsparse_int nnz_C = -1;
        ASSERT_EQ(rocsparse_csrgeam_nnz(handle,
                                        static_cast<rocsparse_int>(m),
                                        static_cast<rocsparse_int>(n),
                                        dA.d,
                                        static_cast<rocsparse_int>(A.nnz()),
                                        pA.ptr,
                                        iA.ptr,
                                        dB.d,
                                        static_cast<rocsparse_int>(B.nnz()),
                                        pB.ptr,
                                        iB.ptr,
                                        dC.d,
                                        ptrC.ptr,
                                        &nnz_C),
                  rocsparse_status_success);

        // Expected result.
        const Bsr want = bsr_geam(to_c<T>(alpha), bsr_from_csr(A), to_c<T>(beta), bsr_from_csr(B));
        ASSERT_EQ(nnz_C, want.nnzb());
        if(m > 0)
        {
            expect_idx_equal<rocsparse_int>(ptrC.host(), want.ptr, base_C, "C row_ptr");
        }

        const T               sentinel = from_c<T>(cplx(-77.0, 0.0));
        DevBuf<rocsparse_int> indC(
            std::vector<rocsparse_int>(static_cast<size_t>(std::max<int64_t>(nnz_C, 1)), -5));
        DevBuf<T> valC(std::vector<T>(static_cast<size_t>(std::max<int64_t>(nnz_C, 1)), sentinel));
        ASSERT_TRUE(indC.ptr && valC.ptr);

        const auto geam = [&](T* val_C, rocsparse_int* col_ind_C) {
            return Api<T>::csrgeam(handle,
                                   static_cast<rocsparse_int>(m),
                                   static_cast<rocsparse_int>(n),
                                   &alpha,
                                   dA.d,
                                   static_cast<rocsparse_int>(A.nnz()),
                                   static_cast<const T*>(vA.ptr),
                                   static_cast<const rocsparse_int*>(pA.ptr),
                                   static_cast<const rocsparse_int*>(iA.ptr),
                                   &beta,
                                   dB.d,
                                   static_cast<rocsparse_int>(B.nnz()),
                                   static_cast<const T*>(vB.ptr),
                                   static_cast<const rocsparse_int*>(pB.ptr),
                                   static_cast<const rocsparse_int*>(iB.ptr),
                                   dC.d,
                                   val_C,
                                   static_cast<const rocsparse_int*>(ptrC.ptr),
                                   col_ind_C);
        };
        ASSERT_EQ(geam(valC.ptr, indC.ptr), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        if(nnz_C > 0)
        {
            std::vector<rocsparse_int> hc = indC.host();
            std::vector<T>             hv = valC.host();
            sort_rows(ptrC.host(), base_C, hc, hv);
            expect_idx_equal<rocsparse_int>(hc, want.col, base_C, "C col_ind");
            expect_vec_equal<T>(hv, want.val, "C val");
        }
        else
        {
            // Nothing to write: the output arrays are left alone.
            EXPECT_EQ(indC.host()[0], -5);
            EXPECT_EQ(to_c<T>(valC.host()[0]), cplx(-77.0, 0.0));
        }

        // The output arrays are required as soon as C has entries.
        if(check_missing_outputs && nnz_C > 0)
        {
            EXPECT_EQ(geam(nullptr, indC.ptr), rocsparse_status_invalid_pointer);
            EXPECT_EQ(geam(valC.ptr, nullptr), rocsparse_status_invalid_pointer);
        }
    }

    // An m x n matrix without entries, or a random one when `entries` is set.
    template <typename T>
    Csr legacy_operand(int64_t m, int64_t n, bool entries, Rng& rng)
    {
        return entries ? random_csr<T>(m, n, 3, rng)
                       : csr_with_row_lengths<T>(m, n, std::vector<int64_t>(m, 0), rng);
    }

    template <typename T>
    void legacy_geam(rocsparse_handle handle, int64_t m, int64_t n, bool a_entries, bool b_entries)
    {
        SCOPED_TRACE("m=" + std::to_string(m) + " n=" + std::to_string(n)
                     + " a=" + std::to_string(a_entries) + " b=" + std::to_string(b_entries));
        Rng       rng(static_cast<uint64_t>(m * 17 + n) + 1);
        const Csr A = legacy_operand<T>(m, n, a_entries, rng);
        const Csr B = legacy_operand<T>(m, n, b_entries, rng);
        run_legacy_geam<T>(handle, A, B, cplx(2.0, 0.0), cplx(-1.0, 0.0), 0, true);
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
    // CSR matrix with the given (zero based, sorted) columns in each row and the value
    // scale * (column + 1) at every entry.
    Csr csr_from_rows(int64_t n, const std::vector<std::vector<int64_t>>& rows, double scale)
    {
        Csr A;
        A.m = static_cast<int64_t>(rows.size());
        A.n = n;
        A.ptr.push_back(0);
        for(const auto& r : rows)
        {
            for(int64_t j : r)
            {
                A.col.push_back(j);
                A.val.push_back(cplx(scale * static_cast<double>(j + 1), 0.0));
            }
            A.ptr.push_back(static_cast<int64_t>(A.col.size()));
        }
        return A;
    }

    // C = alpha * A + beta * B on rows whose union of columns fills whole chunks of the fill
    // kernel. The kernel processes a row per wavefront and a chunk of WFSIZE consecutive
    // columns at a time (WFSIZE is the device's wavefront size: 32 on gfx1201, 64 on gfx9):
    // it marks the columns of the chunk in a per-lane table and places them with a ballot,
    // so the lanes above 32 only exist in the 64-lane kernel. Every chunk starts at the
    // row's first column not yet placed. With n = 64, rows 0 to 2 are one full 64-chunk or
    // two full 32-chunks; row 3 puts two entries past lane 32 of a 64-chunk that starts at
    // column 5, or spreads over two sparse 32-chunks; row 4 (columns 30 to 63) is a full
    // 32-chunk plus two entries, or the lower 34 lanes of a 64-chunk. A kernel built for the
    // other wavefront size misplaces the lanes above 32 of rows 0 to 2.
    template <typename T>
    void full_chunk_geam(rocsparse_handle handle, int base_C)
    {
        SCOPED_TRACE(std::string("full chunk base_C=") + std::to_string(base_C));
        const int64_t        n = 64;
        std::vector<int64_t> all(n), even, odd, low, high;
        for(int64_t j = 0; j < n; ++j)
        {
            all[static_cast<size_t>(j)] = j;
            (j % 2 ? odd : even).push_back(j);
            if(j < 40)
                low.push_back(j);
            if(j >= 30)
                high.push_back(j);
        }
        // Row 0: all columns from A alone. Row 1: A and B interleave. Row 2: A and B
        // overlap on [30, 40). Row 3: two sparse entries past lane 32 of a chunk that
        // starts at column 5. Row 4: B alone, the upper half of the columns.
        const Csr A = csr_from_rows(n, {all, even, low, {5, 50}, {}}, 1.0);
        const Csr B = csr_from_rows(n, {{}, odd, high, {50, 60}, high}, -0.5);
        run_legacy_geam<T>(handle, A, B, cplx(2.0, 0.0), cplx(3.0, 0.0), base_C, false);
    }
}

TEST_F(ExtraGeam, legacy_csrgeam_full_wavefront_chunk)
{
    for(int base_C : {0, 1})
    {
        full_chunk_geam<float>(handle, base_C);
        if(HasFatalFailure())
            return;
        full_chunk_geam<double>(handle, base_C);
        if(HasFatalFailure())
            return;
    }
}

namespace
{
    using stage_t = rocsparse_spgeam_stage;

    // The stages that count the non-zeros of C, and run before C exists in the
    // documented order.
    bool is_nnz_analysis(stage_t stage)
    {
        return stage == rocsparse_spgeam_stage_analysis
               || stage == rocsparse_spgeam_stage_symbolic_analysis;
    }

    // Frees a block filled with 0xff bytes into the device's default memory pool,
    // with the pool's release threshold raised so that the block is kept. A later
    // small hipMallocAsync on the handle's stream, like the spgeam descriptor's row
    // pointer, can then reuse it, and a missing fill shows up in C. The threshold
    // is restored on destruction.
    struct PoisonedPool
    {
        hipMemPool_t pool      = nullptr;
        uint64_t     threshold = 0;
        PoisonedPool(rocsparse_handle handle, size_t bytes)
        {
            hipStream_t stream = nullptr;
            int         device = 0;
            EXPECT_EQ(rocsparse_get_stream(handle, &stream), rocsparse_status_success);
            EXPECT_EQ(hipGetDevice(&device), hipSuccess);
            EXPECT_EQ(hipDeviceGetDefaultMemPool(&pool, device), hipSuccess);
            EXPECT_EQ(hipMemPoolGetAttribute(pool, hipMemPoolAttrReleaseThreshold, &threshold),
                      hipSuccess);
            uint64_t keep = UINT64_MAX;
            EXPECT_EQ(hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &keep),
                      hipSuccess);
            void* block = nullptr;
            EXPECT_EQ(hipMallocAsync(&block, bytes, stream), hipSuccess);
            EXPECT_EQ(hipMemsetAsync(block, 0xff, bytes, stream), hipSuccess);
            EXPECT_EQ(hipFreeAsync(block, stream), hipSuccess);
        }
        PoisonedPool(const PoisonedPool&) = delete;
        PoisonedPool& operator=(const PoisonedPool&) = delete;
        ~PoisonedPool()
        {
            EXPECT_EQ(hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &threshold),
                      hipSuccess);
        }
    };

    // C = alpha * A + beta * B where A and B have no rows / no columns / no
    // entries: C has no entries. Combined and split stage sequences.
    // pass_C is false for the documented order: analysis (or symbolic analysis)
    // runs before C is created, and rocsparse_spgeam_get_output reports nnz
    // (AISPARSE-926). C is passed to every later stage, numeric analysis included.
    // I is the row pointer type of A, B and C; C has index base base_C and,
    // with null_row_ptr_C (m == 0 only), no row pointer array.
    template <typename T, typename I>
    void empty_spgeam(rocsparse_handle       handle,
                      int64_t                m,
                      int64_t                n,
                      bool                   split,
                      bool                   pass_C,
                      rocsparse_pointer_mode mode,
                      rocsparse_index_base   base_C,
                      bool                   null_row_ptr_C)
    {
        SCOPED_TRACE("m=" + std::to_string(m) + " n=" + std::to_string(n)
                     + " split=" + std::to_string(split) + " pass_C=" + std::to_string(pass_C)
                     + " device=" + std::to_string(mode == rocsparse_pointer_mode_device) + " i64="
                     + std::to_string(sizeof(I) == 8) + " base_C=" + std::to_string(base_C)
                     + " null_row_ptr_C=" + std::to_string(null_row_ptr_C));
        const rocsparse_datatype dt = rocsparse_ut::dt_of<T>();
        Rng                      rng(7);
        const Csr                E
            = csr_with_row_lengths<T>(m, n, std::vector<int64_t>(static_cast<size_t>(m), 0), rng);

        DevBuf<I>       pA = up_idx<I>(E.ptr), pB = up_idx<I>(E.ptr);
        DevBuf<int32_t> cA = up_idx<int32_t>(E.col), cB = up_idx<int32_t>(E.col);
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
                                                 rocsparse_ut::it_of<I>(),
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 dt),
                      rocsparse_status_success);
        }

        // A supplied C starts with a nonzero count (at most m * n) so that the
        // analysis check sees the count being written.
        const int64_t   nnz_C0 = (pass_C && m * n > 0) ? 1 : 0;
        DevBuf<I>       pC(std::vector<I>(static_cast<size_t>(m) + 1, -7));
        DevBuf<int32_t> cC(1);
        DevBuf<T>       vC(1);
        ASSERT_TRUE(pC.ptr && cC.ptr && vC.ptr);
        ASSERT_EQ(rocsparse_create_csr_descr(&sC.p,
                                             m,
                                             n,
                                             nnz_C0,
                                             null_row_ptr_C ? nullptr : pC.ptr,
                                             cC.ptr,
                                             vC.ptr,
                                             rocsparse_ut::it_of<I>(),
                                             rocsparse_indextype_i32,
                                             base_C,
                                             dt),
                  rocsparse_status_success);

        PointerModeGuard mode_guard(handle, mode);
        const bool       device = (mode == rocsparse_pointer_mode_device);
        Scalar<T>        alpha(cplx(2.0, 0.0), device);
        Scalar<T>        beta(cplx(-1.0, 0.0), device);
        SpgeamDescrH     descr;
        ASSERT_NE(descr.d, nullptr);
        // A fresh descriptor already holds a count of 0.
        descr.d->nnz_C = -1;
        spgeam_set_inputs(handle,
                          descr.d,
                          rocsparse_ut::dt_of<T>(),
                          rocsparse_ut::dt_of<T>(),
                          alpha.get(),
                          beta.get());
        if(::testing::Test::HasFatalFailure())
            return;

        auto mat_C_for = [&](stage_t stage) -> rocsparse_spmat_descr {
            if(!pass_C && is_nnz_analysis(stage))
            {
                return nullptr;
            }
            return sC.p;
        };

        auto run = [&](stage_t stage) {
            rocsparse_spmat_descr       mat_C = mat_C_for(stage);
            std::optional<PoisonedPool> poison;
            if(mat_C == nullptr)
            {
                poison.emplace(handle, (static_cast<size_t>(m) + 1) * sizeof(int64_t));
            }
            size_t bs = 0;
            ASSERT_EQ(rocsparse_spgeam_buffer_size(
                          handle, descr.d, sA.p, sB.p, mat_C, stage, &bs, nullptr),
                      rocsparse_status_success);
            DevBuf<char> tmp(bs);
            ASSERT_EQ(
                rocsparse_spgeam(
                    handle, descr.d, sA.p, sB.p, mat_C, stage, bs, bs ? tmp.ptr : nullptr, nullptr),
                rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        };
        // rocsparse_spgeam_get_output is documented for the flow without C; with C
        // supplied, the count is read from C.
        auto analysis_nnz = [&]() {
            int64_t nnz = -1;
            if(pass_C)
            {
                int64_t r = 0, c = 0;
                ASSERT_EQ(rocsparse_spmat_get_size(sC.p, &r, &c, &nnz), rocsparse_status_success);
            }
            else
            {
                ASSERT_EQ(
                    rocsparse_spgeam_get_output(
                        handle, descr.d, rocsparse_spgeam_output_nnz, &nnz, sizeof(nnz), nullptr),
                    rocsparse_status_success);
            }
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
        if(null_row_ptr_C)
            return;
        // C has no entries: every row pointer, each pre-filled with -7, must be
        // overwritten with the index base of C. Without C at analysis, compute copies
        // the spgeam descriptor's row pointer, which PoisonedPool leaves filled with
        // 0xff bytes unless the analysis writes it.
        EXPECT_EQ(pC.host(), std::vector<I>(static_cast<size_t>(m) + 1, static_cast<I>(base_C)));
    }

    // Every row pointer type and index base of C, host pointer mode.
    void empty_spgeam_C_formats(
        rocsparse_handle handle, int64_t m, int64_t n, bool split, bool pass_C, bool null_row_ptr_C)
    {
        for(rocsparse_index_base base : {rocsparse_index_base_zero, rocsparse_index_base_one})
        {
            empty_spgeam<float, int32_t>(
                handle, m, n, split, pass_C, rocsparse_pointer_mode_host, base, null_row_ptr_C);
            if(::testing::Test::HasFatalFailure())
                return;
            empty_spgeam<float, int64_t>(
                handle, m, n, split, pass_C, rocsparse_pointer_mode_host, base, null_row_ptr_C);
            if(::testing::Test::HasFatalFailure())
                return;
        }
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
            for(bool pass_C : {true, false})
            {
                empty_spgeam<rocsparse_double_complex, int32_t>(handle,
                                                                d.m,
                                                                d.n,
                                                                split,
                                                                pass_C,
                                                                rocsparse_pointer_mode_host,
                                                                rocsparse_index_base_zero,
                                                                false);
                if(HasFatalFailure())
                    return;
                empty_spgeam_C_formats(handle, d.m, d.n, split, pass_C, false);
                if(HasFatalFailure())
                    return;
            }
            // Device pointer mode: the count is still written on the host, to the
            // spgeam descriptor with C omitted and to the C matrix descriptor with
            // C supplied.
            for(bool pass_C : {false, true})
            {
                empty_spgeam<float, int32_t>(handle,
                                             d.m,
                                             d.n,
                                             split,
                                             pass_C,
                                             rocsparse_pointer_mode_device,
                                             rocsparse_index_base_zero,
                                             false);
                if(HasFatalFailure())
                    return;
            }
            // C without rows may be created without a row pointer array.
            if(d.m == 0)
            {
                for(bool pass_C : {false, true})
                {
                    empty_spgeam_C_formats(handle, d.m, d.n, split, pass_C, true);
                    if(HasFatalFailure())
                        return;
                }
            }
        }
}

// A C without rows may have a null row pointer. When A has rows, compute and
// symbolic compute copy m + 1 row pointer entries into C, so a null row pointer
// is rejected instead of skipped.
TEST_F(ExtraGeam, spgeam_null_row_ptr_C_with_rows_rejected)
{
    const int64_t   m = 4, n = 4;
    Rng             rng(7);
    const Csr       E  = csr_with_row_lengths<float>(m, n, std::vector<int64_t>(m, 0), rng);
    DevBuf<int32_t> pA = up_idx<int32_t>(E.ptr), cA = up_idx<int32_t>(E.col);
    DevBuf<float>   vA = up_val<float>(E.val);
    DevBuf<int32_t> cC(1);
    DevBuf<float>   vC(1);
    ASSERT_TRUE(pA.ptr && cA.ptr && vA.ptr && cC.ptr && vC.ptr);
    SpmatH sA, sC;
    ASSERT_EQ(rocsparse_create_csr_descr(&sA.p,
                                         m,
                                         n,
                                         0,
                                         pA.ptr,
                                         cA.ptr,
                                         vA.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_csr_descr(&sC.p,
                                         0,
                                         n,
                                         0,
                                         nullptr,
                                         cC.ptr,
                                         vC.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    PointerModeGuard mode_guard(handle, rocsparse_pointer_mode_host);
    const float      alpha = 2.0f, beta = -1.0f;
    for(stage_t compute : {rocsparse_spgeam_stage_compute, rocsparse_spgeam_stage_symbolic_compute})
    {
        SCOPED_TRACE("compute stage " + std::to_string(compute));
        const stage_t analysis = compute == rocsparse_spgeam_stage_compute
                                     ? rocsparse_spgeam_stage_analysis
                                     : rocsparse_spgeam_stage_symbolic_analysis;
        SpgeamDescrH  descr;
        ASSERT_NE(descr.d, nullptr);
        spgeam_set_inputs(
            handle, descr.d, rocsparse_datatype_f32_r, rocsparse_datatype_f32_r, &alpha, &beta);
        if(HasFatalFailure())
            return;
        for(stage_t stage : {analysis, compute})
        {
            rocsparse_spmat_descr mat_C = stage == analysis ? nullptr : sC.p;
            size_t                bs    = 0;
            ASSERT_EQ(rocsparse_spgeam_buffer_size(
                          handle, descr.d, sA.p, sA.p, mat_C, stage, &bs, nullptr),
                      rocsparse_status_success);
            DevBuf<char> tmp(bs);
            EXPECT_EQ(
                rocsparse_spgeam(
                    handle, descr.d, sA.p, sA.p, mat_C, stage, bs, bs ? tmp.ptr : nullptr, nullptr),
                stage == analysis ? rocsparse_status_success : rocsparse_status_invalid_pointer);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        }
    }
}
