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

// Pruning: prune_csr2csr and prune_dense2csr with a threshold, and
// prune_csr2csr_by_percentage and prune_dense2csr_by_percentage. The results are
// compared with host references. A routine keeps the entries whose magnitude is
// strictly larger than the threshold; with the percentage routines the threshold is the
// element at position ceil(count * percentage / 100) - 1 of the sorted magnitudes of the
// candidate entries (all entries of the dense matrix, the stored entries of the CSR one).

#include "unit_test_device_conv_common.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

using namespace convut;

namespace
{
    using ConvPrune = ConvTest;

    // The single or double precision variant of a routine.
    template <typename T, typename S, typename D>
    auto sd(S s, D d)
    {
        return pick<T>(s, d, s, d);
    }

    // Runs f with a float and with a double value.
    template <typename F>
    void both_types(F f)
    {
        f(float{});
        f(double{});
    }

    // Combinations of index bases, pointer mode and handle buffer situation. Every
    // combination is used with every value type; `shrink` removes the handle buffer so
    // that the routine allocates its own temporary storage.
    struct Combo
    {
        int  base_a, base_c;
        bool device, shrink;
    };
    constexpr Combo s_combos[]
        = {{0, 0, false, false}, {0, 1, true, false}, {1, 1, false, true}, {1, 0, true, true}};

    std::string trace(const Combo& c)
    {
        return "bases " + std::to_string(c.base_a) + std::to_string(c.base_c) + " device "
               + std::to_string(c.device) + " shrink " + std::to_string(c.shrink);
    }

    template <typename T>
    T percentile_threshold(std::vector<T> magnitudes, double percentage)
    {
        if(magnitudes.empty())
            return T(0);
        std::sort(magnitudes.begin(), magnitudes.end());
        const int64_t count = magnitudes.size();
        const int64_t pos   = std::clamp<int64_t>(
            static_cast<int64_t>(std::ceil(count * (percentage / 100.0))) - 1, 0, count - 1);
        return magnitudes[pos];
    }

    template <typename T>
    T threshold_of(const HostCsr<T>& a, double percentage)
    {
        std::vector<T> magnitudes;
        for(T v : a.val)
            magnitudes.push_back(std::abs(v));
        return percentile_threshold(magnitudes, percentage);
    }

    // `rows` rows with `per_row` entries at the columns 0 .. per_row - 1. The values
    // alternate between 0.25 and 1.5 or -1.5.
    template <typename T>
    HostCsr<T> make_rows(I rows, I per_row, int base)
    {
        HostCsr<T> a;
        a.m    = rows;
        a.n    = std::max<I>(per_row, 1);
        a.base = base;
        a.ptr.push_back(base);
        for(I i = 0; i < rows; ++i)
        {
            for(I k = 0; k < per_row; ++k)
            {
                a.ind.push_back(k + base);
                a.val.push_back(
                    static_cast<T>(k % 2 == 0 ? 0.25 : ((i + k) % 4 == 1 ? 1.5 : -1.5)));
            }
            a.ptr.push_back(base + a.ind.size());
        }
        return a;
    }

    // Column major dense matrix with distinct magnitudes 1 .. 97 (and a zero where
    // (3 i + j) % 19 == 0, so that low percentages prune nonzero entries too), optionally
    // with alternating signs. The padding between the columns is -777.
    template <typename T>
    std::vector<T> make_dense(int m, int n, int ld, bool signs)
    {
        std::vector<T> a(size_t(ld) * n, T(-777));
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < m; ++i)
            {
                const int k   = i * n + j;
                const T   mag = static_cast<T>(1 + (k * 17) % 97);
                a[size_t(j) * ld + i]
                    = ((3 * i + j) % 19 == 0) ? T(0) : (signs && k % 2 ? -mag : mag);
            }
        return a;
    }

    // The CSR matrix of the entries of a dense matrix for which keep(value) holds.
    template <typename T, typename P>
    HostCsr<T> dense_keep(const std::vector<T>& a, int m, int n, int ld, int base, P keep)
    {
        HostCsr<T> c;
        c.m    = m;
        c.n    = n;
        c.base = base;
        c.ptr.push_back(base);
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                if(keep(a[size_t(j) * ld + i]))
                {
                    c.ind.push_back(j + base);
                    c.val.push_back(a[size_t(j) * ld + i]);
                }
            }
            c.ptr.push_back(base + c.ind.size());
        }
        return c;
    }

    // ------------------------------------------------------------------------------
    // CSR to CSR
    // ------------------------------------------------------------------------------
    template <typename T>
    void run_prune_csr(rocsparse_handle handle, I per_row, const Combo& c)
    {
        SCOPED_TRACE("per_row " + std::to_string(per_row) + " " + trace(c));
        const auto a    = make_rows<T>(4, per_row, c.base_a);
        const T    thr  = 0.5;
        const auto want = keep_entries(a, c.base_c, [&](T v) { return std::abs(v) > thr; });
        const I    nnz  = a.nnz();

        MatDescr  descr_a(base_of(c.base_a)), descr_c(base_of(c.base_c));
        DevBuf<I> ptr(a.ptr), ind(a.ind), ptr_c(a.m + 1), ind_c(nnz);
        DevBuf<T> val(a.val), val_c(nnz), d_thr(std::vector<T>{thr});

        PointerModeScope pm(handle, s_pointer_modes[c.device]);
        const T*         t    = c.device ? d_thr.get() : &thr;
        size_t           size = 0;
        ASSERT_EQ(sd<T>(rocsparse_sprune_csr2csr_buffer_size,
                        rocsparse_dprune_csr2csr_buffer_size)(handle,
                                                              a.m,
                                                              a.n,
                                                              nnz,
                                                              descr_a,
                                                              val,
                                                              ptr,
                                                              ind,
                                                              t,
                                                              descr_c,
                                                              val_c,
                                                              ptr_c,
                                                              ind_c,
                                                              &size),
                  st_ok);
        DevBuf<char>                  temp(size);
        std::optional<NoHandleBuffer> no_buffer;
        if(c.shrink)
            no_buffer.emplace(handle);
        const I nnz_c = nnz_via(handle, c.device, [&](I* p) {
            return sd<T>(rocsparse_sprune_csr2csr_nnz, rocsparse_dprune_csr2csr_nnz)(
                handle, a.m, a.n, nnz, descr_a, val, ptr, ind, t, descr_c, ptr_c, p, temp);
        });
        ASSERT_EQ(nnz_c, want.nnz());
        ASSERT_EQ(sd<T>(rocsparse_sprune_csr2csr, rocsparse_dprune_csr2csr)(handle,
                                                                            a.m,
                                                                            a.n,
                                                                            nnz,
                                                                            descr_a,
                                                                            val,
                                                                            ptr,
                                                                            ind,
                                                                            t,
                                                                            descr_c,
                                                                            val_c,
                                                                            ptr_c,
                                                                            ind_c,
                                                                            temp),
                  st_ok);
        expect_csr(ptr_c, ind_c, val_c, want);
    }

    // The mean number of entries per row selects the kernel segment (< 4, < 8, < 16,
    // < 32, otherwise).
    TEST_F(ConvPrune, PruneCsr2Csr)
    {
        int i = 0;
        for(I per_row : {1, 5, 10, 20, 32})
        {
            int j = 0;
            both_types([&](auto t) {
                run_prune_csr<decltype(t)>(handle, per_row, s_combos[(i + 2 * j++) % 4]);
            });
            ++i;
        }
    }

    TEST_F(ConvPrune, PruneCsr2CsrQuickReturnsAndArguments)
    {
        MatDescr      descr_a, descr_c;
        DevBuf<I>     ptr(std::vector<I>{0, 0, 0}), ptr_c(3);
        const float   h_thr = 0.5f;
        DevBuf<float> d_thr(std::vector<float>{h_thr});
        DevBuf<char>  tmp(64);

        // No rows, or no columns and no entries: nothing is kept; the compute routine
        // returns at once.
        for(bool device_mode : {false, true})
            for(I m : {0, 2})
            {
                PointerModeScope pm(handle, s_pointer_modes[device_mode]);
                const float*     thr = device_mode ? d_thr.get() : &h_thr;
                const I          n   = m ? 0 : 3;
                EXPECT_EQ(nnz_via(handle,
                                  device_mode,
                                  [&](I* p) {
                                      return rocsparse_sprune_csr2csr_nnz(handle,
                                                                          m,
                                                                          n,
                                                                          0,
                                                                          descr_a,
                                                                          nullptr,
                                                                          ptr,
                                                                          nullptr,
                                                                          thr,
                                                                          descr_c,
                                                                          ptr_c,
                                                                          p,
                                                                          tmp);
                                  }),
                          0);
                EXPECT_EQ(rocsparse_sprune_csr2csr(handle,
                                                   m,
                                                   n,
                                                   0,
                                                   descr_a,
                                                   nullptr,
                                                   ptr,
                                                   nullptr,
                                                   thr,
                                                   descr_c,
                                                   nullptr,
                                                   ptr_c,
                                                   nullptr,
                                                   tmp),
                          st_ok);
            }

        // Entries are kept (zero threshold): the output arrays are required.
        const auto    a = make_rows<float>(4, 1, 0);
        DevBuf<I>     a_ptr(a.ptr), a_ind(a.ind), a_ptr_c(5);
        DevBuf<float> a_val(a.val);
        const float   keep = 0;
        size_t        size = 0;
        ASSERT_EQ(rocsparse_sprune_csr2csr_buffer_size(handle,
                                                       4,
                                                       a.n,
                                                       a.nnz(),
                                                       descr_a,
                                                       a_val,
                                                       a_ptr,
                                                       a_ind,
                                                       &keep,
                                                       descr_c,
                                                       nullptr,
                                                       a_ptr_c,
                                                       nullptr,
                                                       &size),
                  st_ok);
        DevBuf<char> temp(size);
        ASSERT_GT(nnz_via(handle,
                          false,
                          [&](I* p) {
                              return rocsparse_sprune_csr2csr_nnz(handle,
                                                                  4,
                                                                  a.n,
                                                                  a.nnz(),
                                                                  descr_a,
                                                                  a_val,
                                                                  a_ptr,
                                                                  a_ind,
                                                                  &keep,
                                                                  descr_c,
                                                                  a_ptr_c,
                                                                  p,
                                                                  temp);
                          }),
                  0);
        EXPECT_EQ(rocsparse_sprune_csr2csr(handle,
                                           4,
                                           a.n,
                                           a.nnz(),
                                           descr_a,
                                           a_val,
                                           a_ptr,
                                           a_ind,
                                           &keep,
                                           descr_c,
                                           nullptr,
                                           a_ptr_c,
                                           nullptr,
                                           temp),
                  st_ptr);

        // Sorted column indices are required.
        ASSERT_EQ(rocsparse_set_mat_storage_mode(descr_a, rocsparse_storage_mode_unsorted), st_ok);
        I nnz_c = 0;
        EXPECT_REJECTED(rocsparse_sprune_csr2csr_nnz(handle,
                                                     4,
                                                     a.n,
                                                     a.nnz(),
                                                     descr_a,
                                                     a_val,
                                                     a_ptr,
                                                     a_ind,
                                                     &keep,
                                                     descr_c,
                                                     a_ptr_c,
                                                     &nnz_c,
                                                     temp));
    }

    // ------------------------------------------------------------------------------
    // CSR to CSR by percentage
    // ------------------------------------------------------------------------------
    template <typename T>
    void run_pct_csr(rocsparse_handle handle, I per_row, double pct, const Combo& c)
    {
        SCOPED_TRACE("per_row " + std::to_string(per_row) + " pct " + std::to_string(pct) + " "
                     + trace(c));
        auto a = make_rows<T>(4, per_row, c.base_a);
        // Distinct magnitudes, so that every percentage selects its own threshold. Every
        // third entry is negative: the percentage is taken over the absolute values.
        for(size_t k = 0; k < a.val.size(); ++k)
            a.val[k] = static_cast<T>((k % 3 == 1 ? -1 : 1) * static_cast<int>(1 + (k * 7) % 97));
        const T    thr  = threshold_of(a, pct);
        const auto want = keep_entries(a, c.base_a, [&](T v) { return std::abs(v) > thr; });
        const I    nnz  = a.nnz();

        MatDescr  descr(base_of(c.base_a));
        MatInfo   info;
        DevBuf<I> ptr(a.ptr), ind(a.ind), ptr_c(a.m + 1), ind_c(nnz);
        DevBuf<T> val(a.val), val_c(nnz);

        size_t size = 0;
        ASSERT_EQ(sd<T>(rocsparse_sprune_csr2csr_by_percentage_buffer_size,
                        rocsparse_dprune_csr2csr_by_percentage_buffer_size)(handle,
                                                                            a.m,
                                                                            a.n,
                                                                            nnz,
                                                                            descr,
                                                                            val,
                                                                            ptr,
                                                                            ind,
                                                                            static_cast<T>(pct),
                                                                            descr,
                                                                            nullptr,
                                                                            ptr_c,
                                                                            nullptr,
                                                                            info.get(),
                                                                            &size),
                  st_ok);
        DevBuf<char>                  temp(size);
        std::optional<NoHandleBuffer> no_buffer;
        if(c.shrink)
            no_buffer.emplace(handle);
        // The threshold is stored in the temporary buffer by the nnz routine, the compute
        // routine reads it back in either pointer mode.
        const I nnz_c = nnz_via(handle, false, [&](I* p) {
            return sd<T>(rocsparse_sprune_csr2csr_nnz_by_percentage,
                         rocsparse_dprune_csr2csr_nnz_by_percentage)(handle,
                                                                     a.m,
                                                                     a.n,
                                                                     nnz,
                                                                     descr,
                                                                     val,
                                                                     ptr,
                                                                     ind,
                                                                     static_cast<T>(pct),
                                                                     descr,
                                                                     ptr_c,
                                                                     p,
                                                                     info.get(),
                                                                     temp);
        });
        ASSERT_EQ(nnz_c, want.nnz());
        PointerModeScope pm(handle, s_pointer_modes[c.device]);
        ASSERT_EQ(sd<T>(rocsparse_sprune_csr2csr_by_percentage,
                        rocsparse_dprune_csr2csr_by_percentage)(handle,
                                                                a.m,
                                                                a.n,
                                                                nnz,
                                                                descr,
                                                                val,
                                                                ptr,
                                                                ind,
                                                                static_cast<T>(pct),
                                                                descr,
                                                                val_c,
                                                                ptr_c,
                                                                ind_c,
                                                                info.get(),
                                                                temp),
                  st_ok);
        expect_csr(ptr_c, ind_c, val_c, want);
    }

    TEST_F(ConvPrune, PruneCsr2CsrByPercentage)
    {
        int i = 0;
        for(I per_row : {1, 5, 10, 20, 32})
        {
            int j = 0;
            both_types([&](auto t) {
                int p = 0;
                for(double pct : {0.0, 25.0, 100.0})
                    run_pct_csr<decltype(t)>(handle, per_row, pct, s_combos[(i + 2 * j + p++) % 4]);
                ++j;
            });
            ++i;
        }
    }

    TEST_F(ConvPrune, PruneCsr2CsrByPercentageQuickReturnsAndArguments)
    {
        MatDescr     descr_a, descr_c;
        MatInfo      info;
        DevBuf<I>    ptr(std::vector<I>{0, 0, 0}), ptr_c(3);
        DevBuf<char> tmp(64);

        // No rows, or no columns and no entries.
        for(bool device_mode : {false, true})
            for(I m : {0, 2})
            {
                const I n = m ? 0 : 4;
                EXPECT_EQ(nnz_via(handle,
                                  device_mode,
                                  [&](I* p) {
                                      return rocsparse_sprune_csr2csr_nnz_by_percentage(handle,
                                                                                        m,
                                                                                        n,
                                                                                        0,
                                                                                        descr_a,
                                                                                        nullptr,
                                                                                        ptr,
                                                                                        nullptr,
                                                                                        10.f,
                                                                                        descr_c,
                                                                                        ptr_c,
                                                                                        p,
                                                                                        info.get(),
                                                                                        tmp);
                                  }),
                          0);
                EXPECT_EQ(rocsparse_sprune_csr2csr_by_percentage(handle,
                                                                 m,
                                                                 n,
                                                                 0,
                                                                 descr_a,
                                                                 nullptr,
                                                                 ptr,
                                                                 nullptr,
                                                                 0.f,
                                                                 descr_c,
                                                                 nullptr,
                                                                 ptr_c,
                                                                 nullptr,
                                                                 info.get(),
                                                                 tmp),
                          st_ok);
            }

        // A percentage outside of [0, 100] is invalid. The header documents
        // invalid_size, the library returns invalid_value (AISPARSE-904).
        DevBuf<double> val(std::vector<double>{1.0, 2.0});
        DevBuf<I>      rp(std::vector<I>{0, 1, 2}), ci(std::vector<I>{0, 1});
        I              nnz_c = 0;
        size_t         size  = 0;
        EXPECT_STATUS_IN(rocsparse_sprune_csr2csr_nnz_by_percentage(handle,
                                                                    2,
                                                                    2,
                                                                    0,
                                                                    descr_a,
                                                                    nullptr,
                                                                    ptr,
                                                                    nullptr,
                                                                    -1.f,
                                                                    descr_c,
                                                                    ptr_c,
                                                                    &nnz_c,
                                                                    info.get(),
                                                                    tmp),
                         st_val,
                         st_size);
        EXPECT_STATUS_IN(rocsparse_dprune_csr2csr_by_percentage_buffer_size(handle,
                                                                            2,
                                                                            2,
                                                                            2,
                                                                            descr_a,
                                                                            val,
                                                                            rp,
                                                                            ci,
                                                                            150.0,
                                                                            descr_c,
                                                                            nullptr,
                                                                            rp,
                                                                            nullptr,
                                                                            info.get(),
                                                                            &size),
                         st_val,
                         st_size);
    }

    // The CSR routine checks csr_val_C and csr_col_ind_C against the number of kept
    // entries. Keeping nothing (percentage 100) makes null arrays legal.
    TEST_F(ConvPrune, PruneCsr2CsrByPercentageOutputArrays)
    {
        const int m = 8, n = 6;

        // Row i holds the columns j with (i + j) % 2 == 0; the magnitudes are distinct.
        HostCsr<double> a;
        a.m = m;
        a.n = n;
        a.ptr.push_back(0);
        for(int i = 0; i < m; ++i)
        {
            for(int j = (i % 2); j < n; j += 2)
            {
                a.ind.push_back(j);
                a.val.push_back((a.val.size() % 2 ? -1.0 : 1.0) * (1.0 + (a.val.size() * 7) % 31));
            }
            a.ptr.push_back(a.ind.size());
        }
        DevCsr<double> d(a);
        MatDescr       descr_c;
        MatInfo        info;

        for(double pct : {100.0, 50.0})
        {
            SCOPED_TRACE("pct " + std::to_string(pct));
            const double thr  = threshold_of(a, pct);
            const auto   want = keep_entries(a, 0, [&](double v) { return std::abs(v) > thr; });
            DevBuf<I>    ptr_c(m + 1);
            size_t       size = 0;
            ASSERT_EQ(rocsparse_dprune_csr2csr_by_percentage_buffer_size(handle,
                                                                         m,
                                                                         n,
                                                                         a.nnz(),
                                                                         d.descr,
                                                                         d.val,
                                                                         d.ptr,
                                                                         d.ind,
                                                                         pct,
                                                                         descr_c,
                                                                         nullptr,
                                                                         ptr_c,
                                                                         nullptr,
                                                                         info.get(),
                                                                         &size),
                      st_ok);
            DevBuf<char> temp(size);
            ASSERT_EQ(nnz_via(handle,
                              false,
                              [&](I* p) {
                                  return rocsparse_dprune_csr2csr_nnz_by_percentage(handle,
                                                                                    m,
                                                                                    n,
                                                                                    a.nnz(),
                                                                                    d.descr,
                                                                                    d.val,
                                                                                    d.ptr,
                                                                                    d.ind,
                                                                                    pct,
                                                                                    descr_c,
                                                                                    ptr_c,
                                                                                    p,
                                                                                    info.get(),
                                                                                    temp);
                              }),
                      want.nnz());

            DevBuf<I>      ind_c(want.nnz());
            DevBuf<double> val_c(want.nnz());
            const auto     compute = [&](double* cv, I* ci) {
                return rocsparse_dprune_csr2csr_by_percentage(handle,
                                                              m,
                                                              n,
                                                              a.nnz(),
                                                              d.descr,
                                                              d.val,
                                                              d.ptr,
                                                              d.ind,
                                                              pct,
                                                              descr_c,
                                                              cv,
                                                              ptr_c,
                                                              ci,
                                                              info.get(),
                                                              temp);
            };
            if(want.nnz() == 0)
            {
                EXPECT_EQ(compute(nullptr, nullptr), st_ok);
                continue;
            }
            EXPECT_EQ(compute(nullptr, ind_c), st_ptr);
            EXPECT_EQ(compute(val_c, nullptr), st_ptr);
            ASSERT_EQ(compute(val_c, ind_c), st_ok);
            expect_csr(ptr_c, ind_c, val_c, want);
        }
    }

    // ------------------------------------------------------------------------------
    // Dense to CSR
    // ------------------------------------------------------------------------------
    template <typename T>
    void run_dense_threshold(rocsparse_handle handle, int base, bool device_mode)
    {
        SCOPED_TRACE("base " + std::to_string(base) + " device " + std::to_string(device_mode));
        // No entry equals the threshold, whose treatment the header describes both ways.
        const int  m = 9, n = 7, ld = 11;
        const T    thr  = T(40.5);
        const auto A    = make_dense<T>(m, n, ld, true);
        const auto want = dense_keep(A, m, n, ld, base, [&](T v) { return std::abs(v) > thr; });

        MatDescr         descr(base_of(base));
        DevBuf<T>        d_A(A), d_thr(std::vector<T>{thr});
        DevBuf<I>        d_ptr(m + 1);
        PointerModeScope pm(handle, s_pointer_modes[device_mode]);
        const T*         t    = device_mode ? d_thr.get() : &thr;
        size_t           size = 0;
        ASSERT_EQ(
            sd<T>(rocsparse_sprune_dense2csr_buffer_size, rocsparse_dprune_dense2csr_buffer_size)(
                handle, m, n, d_A, ld, t, descr, nullptr, d_ptr, nullptr, &size),
            st_ok);
        DevBuf<char> temp(size);
        const I      nnz = nnz_via(handle, device_mode, [&](I* p) {
            return sd<T>(rocsparse_sprune_dense2csr_nnz, rocsparse_dprune_dense2csr_nnz)(
                handle, m, n, d_A, ld, t, descr, d_ptr, p, temp);
        });
        ASSERT_EQ(nnz, want.nnz());

        DevBuf<I>  d_ind(nnz);
        DevBuf<T>  d_val(nnz);
        const auto compute = [&](T* val, I* ind) {
            return sd<T>(rocsparse_sprune_dense2csr, rocsparse_dprune_dense2csr)(
                handle, m, n, d_A, ld, t, descr, val, d_ptr, ind, temp);
        };
        ASSERT_EQ(compute(d_val, d_ind), st_ok);
        expect_csr(d_ptr, d_ind, d_val, want);
        // The output arrays are required when there are entries.
        EXPECT_EQ(compute(nullptr, d_ind), st_ptr);
        EXPECT_EQ(compute(d_val, nullptr), st_ptr);
    }

    TEST_F(ConvPrune, PruneDense2Csr)
    {
        both_types([&](auto t) {
            for(int base : {0, 1})
                for(bool device_mode : {false, true})
                    run_dense_threshold<decltype(t)>(handle, base, device_mode);
        });
    }

    template <typename T>
    void run_dense_pct(rocsparse_handle handle, double pct, const Combo& c)
    {
        SCOPED_TRACE("pct " + std::to_string(pct) + " " + trace(c));
        const int  m = 9, n = 7, ld = 11;
        const auto A = make_dense<T>(m, n, ld, true);
        // The padding is not part of the matrix.
        auto magnitudes = dense_keep(A, m, n, ld, 0, [](T) { return true; }).val;
        for(T& v : magnitudes)
            v = std::abs(v);
        const T    threshold = percentile_threshold(magnitudes, pct);
        const auto want
            = dense_keep(A, m, n, ld, c.base_a, [&](T v) { return std::abs(v) > threshold; });

        MatDescr         descr(base_of(c.base_a));
        MatInfo          info;
        DevBuf<T>        d_A(A);
        DevBuf<I>        d_ptr(m + 1);
        PointerModeScope pm(handle, s_pointer_modes[c.device]);
        size_t           size = 0;
        ASSERT_EQ(sd<T>(rocsparse_sprune_dense2csr_by_percentage_buffer_size,
                        rocsparse_dprune_dense2csr_by_percentage_buffer_size)(handle,
                                                                              m,
                                                                              n,
                                                                              d_A,
                                                                              ld,
                                                                              static_cast<T>(pct),
                                                                              descr,
                                                                              nullptr,
                                                                              d_ptr,
                                                                              nullptr,
                                                                              info.get(),
                                                                              &size),
                  st_ok);
        DevBuf<char>                  temp(size);
        std::optional<NoHandleBuffer> no_buffer;
        if(c.shrink)
            no_buffer.emplace(handle);
        const I nnz = nnz_via(handle, c.device, [&](I* p) {
            return sd<T>(rocsparse_sprune_dense2csr_nnz_by_percentage,
                         rocsparse_dprune_dense2csr_nnz_by_percentage)(
                handle, m, n, d_A, ld, static_cast<T>(pct), descr, d_ptr, p, info.get(), temp);
        });
        ASSERT_EQ(nnz, want.nnz());
        DevBuf<I> d_ind(nnz);
        DevBuf<T> d_val(nnz);
        ASSERT_EQ(sd<T>(rocsparse_sprune_dense2csr_by_percentage,
                        rocsparse_dprune_dense2csr_by_percentage)(handle,
                                                                  m,
                                                                  n,
                                                                  d_A,
                                                                  ld,
                                                                  static_cast<T>(pct),
                                                                  descr,
                                                                  d_val,
                                                                  d_ptr,
                                                                  d_ind,
                                                                  info.get(),
                                                                  temp),
                  st_ok);
        expect_csr(d_ptr, d_ind, d_val, want);
    }

    TEST_F(ConvPrune, PruneDense2CsrByPercentage)
    {
        both_types([&](auto t) {
            for(double pct : {0.0, 10.0, 40.0, 77.0, 100.0})
                for(const Combo& c : s_combos)
                    run_dense_pct<decltype(t)>(handle, pct, c);
        });
    }

    // No entries: the nnz routines report zero and fill the row pointer with the index
    // base, the compute routines return at once.
    TEST_F(ConvPrune, PruneDense2CsrEmpty)
    {
        for(int base : {0, 1})
            for(bool device_mode : {false, true})
                for(auto shape : {std::pair<I, I>{0, 5}, {4, 0}, {0, 0}})
                {
                    const I m = shape.first, n = shape.second;
                    SCOPED_TRACE("base " + std::to_string(base) + " device "
                                 + std::to_string(device_mode) + " m " + std::to_string(m) + " n "
                                 + std::to_string(n));
                    const I          ld = std::max<I>(m, 1);
                    MatDescr         descr(base_of(base));
                    MatInfo          info;
                    DevBuf<double>   d_A(16);
                    DevBuf<char>     temp(64);
                    DevBuf<double>   d_thr(std::vector<double>{1.0});
                    PointerModeScope pm(handle, s_pointer_modes[device_mode]);
                    const double     thr = 1.0;
                    const double*    t   = device_mode ? d_thr.get() : &thr;
                    DevBuf<I>        d_ptr(std::vector<I>(8, 77));
                    EXPECT_EQ(nnz_via(handle,
                                      device_mode,
                                      [&](I* p) {
                                          return rocsparse_dprune_dense2csr_nnz(
                                              handle, m, n, d_A, ld, t, descr, d_ptr, p, temp);
                                      }),
                              0);
                    EXPECT_EQ(d_ptr.read(m + 1), std::vector<I>(m + 1, base));
                    EXPECT_EQ(rocsparse_dprune_dense2csr(
                                  handle, m, n, d_A, ld, t, descr, nullptr, d_ptr, nullptr, temp),
                              st_ok);

                    EXPECT_EQ(
                        nnz_via(
                            handle,
                            device_mode,
                            [&](I* p) {
                                return rocsparse_dprune_dense2csr_nnz_by_percentage(
                                    handle, m, n, d_A, ld, 50.0, descr, d_ptr, p, info.get(), temp);
                            }),
                        0);
                    EXPECT_EQ(d_ptr.read(m + 1), std::vector<I>(m + 1, base));
                    EXPECT_EQ(rocsparse_dprune_dense2csr_by_percentage(handle,
                                                                       m,
                                                                       n,
                                                                       d_A,
                                                                       ld,
                                                                       50.0,
                                                                       descr,
                                                                       nullptr,
                                                                       d_ptr,
                                                                       nullptr,
                                                                       info.get(),
                                                                       temp),
                              st_ok);
                }
    }

    // Argument checks of the double precision percentage routines.
    TEST_F(ConvPrune, PruneDense2CsrByPercentageArguments)
    {
        const int      m = 9, n = 7;
        const auto     A = make_dense<double>(m, n, m, true);
        MatDescr       descr;
        MatInfo        info;
        DevBuf<double> d_A(A);
        DevBuf<I>      d_ptr(m + 1);

        size_t size = 0;
        ASSERT_EQ(
            rocsparse_dprune_dense2csr_by_percentage_buffer_size(
                handle, m, n, d_A, m, 40.0, descr, nullptr, d_ptr, nullptr, info.get(), &size),
            st_ok);
        DevBuf<char> temp(size);
        const I      nnz = nnz_via(handle, false, [&](I* p) {
            return rocsparse_dprune_dense2csr_nnz_by_percentage(
                handle, m, n, d_A, m, 40.0, descr, d_ptr, p, info.get(), temp);
        });
        ASSERT_GT(nnz, 0);
        DevBuf<I>      d_ind(nnz);
        DevBuf<double> d_val(nnz);

        const auto compute
            = [&](rocsparse_handle h, double pct, I ld, double* val, I* ind, void* buffer) {
                  return rocsparse_dprune_dense2csr_by_percentage(
                      h, m, n, d_A, ld, pct, descr, val, d_ptr, ind, info.get(), buffer);
              };
        // The output arrays are required when there are entries.
        EXPECT_EQ(compute(handle, 40.0, m, nullptr, d_ind, temp), st_ptr);
        EXPECT_EQ(compute(handle, 40.0, m, d_val, nullptr, temp), st_ptr);
        EXPECT_EQ(compute(handle, 40.0, m, nullptr, nullptr, temp), st_ptr);
        EXPECT_EQ(compute(nullptr, 40.0, m, d_val, d_ind, temp), rocsparse_status_invalid_handle);
        EXPECT_EQ(compute(handle, 40.0, m - 1, d_val, d_ind, temp), st_size);
        EXPECT_EQ(compute(handle, 40.0, m, d_val, d_ind, nullptr), st_ptr);
        // The header documents invalid_size for an invalid percentage, the library returns
        // invalid_value (AISPARSE-904).
        EXPECT_STATUS_IN(compute(handle, -1.0, m, d_val, d_ind, temp), st_size, st_val);
        I count = 0;
        EXPECT_STATUS_IN(rocsparse_dprune_dense2csr_nnz_by_percentage(
                             handle, m, n, d_A, m, 101.0, descr, d_ptr, &count, info.get(), temp),
                         st_size,
                         st_val);
        EXPECT_EQ(rocsparse_dprune_dense2csr_nnz_by_percentage(
                      handle, m, n, d_A, m, 40.0, descr, d_ptr, nullptr, info.get(), temp),
                  st_ptr);
        EXPECT_EQ(rocsparse_dprune_dense2csr_nnz_by_percentage(
                      handle, m, n, d_A, m, 40.0, descr, d_ptr, &count, info.get(), nullptr),
                  st_ptr);
        EXPECT_EQ(
            rocsparse_dprune_dense2csr_by_percentage_buffer_size(
                handle, m, n, d_A, m, 40.0, descr, nullptr, d_ptr, nullptr, info.get(), nullptr),
            st_ptr);
    }
} // namespace
