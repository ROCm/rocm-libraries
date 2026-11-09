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

// Sparse sort routines: the generic rocsparse_spsort API (COO / CSR / CSC, 32 and 64 bit
// indices, strided batches, in place and out of place) and the legacy coosort / csrsort /
// cscsort entry points with and without a permutation. Every output (offsets, indices,
// values and the permutation) is compared with a host reference sort.

#include "unit_test_device_conv_common.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

using namespace convut;

namespace
{
    using ConvSort = ConvTest;

    // Unique (row, col) positions of an M x N matrix in a scrambled order.
    std::vector<std::pair<int, int>> scrambled_positions(int M, int N)
    {
        std::vector<std::pair<int, int>> pos;
        for(int i = 0; i < M; ++i)
            for(int j = 0; j < N; ++j)
                if((i * 7 + j * 3) % 4 == 0)
                    pos.emplace_back(i, j);
        std::vector<std::pair<int, int>> out(pos.size());
        for(size_t k = 0; k < pos.size(); ++k)
            out[(k * 10007) % pos.size()] = pos[k];
        return out;
    }

    // Compressed major index structure (zero based) with unsorted minor indices in each
    // major slice; some slices are empty. N and the multiplier 5 are coprime, so the
    // minor indices of a slice are unique.
    struct Compressed
    {
        std::vector<int> offsets, minor;
    };

    Compressed make_compressed(int M, int N)
    {
        Compressed c{{0}, {}};
        for(int r = 0; r < M; ++r)
        {
            for(int k = 0, cnt = (r * 3 + 1) % (N / 2 + 2); k < cnt; ++k)
                c.minor.push_back((r * 5 + k * 5) % N);
            c.offsets.push_back(c.minor.size());
        }
        return c;
    }

#if defined(ROCSPARSE_WITH_SPSORT)
    constexpr auto alg_def = rocsparse_spsort_alg_default;
    constexpr auto alg_rdx = rocsparse_spsort_alg_radix_sort;
    constexpr auto stg_ana = rocsparse_spsort_stage_analysis;
    constexpr auto stg_cmp = rocsparse_spsort_stage_compute;

    // Owns a rocsparse_spsort_descr (its destruction needs the handle).
    class SpsortDescr
    {
        rocsparse_handle       m_handle;
        rocsparse_spsort_descr m_d = nullptr;

    public:
        explicit SpsortDescr(rocsparse_handle h)
            : m_handle(h)
        {
            EXPECT_EQ(rocsparse_spsort_descr_create(h, &m_d, nullptr), st_ok);
        }
        ~SpsortDescr()
        {
            EXPECT_EQ(rocsparse_spsort_descr_destroy(m_handle, m_d, nullptr), st_ok);
        }
        operator rocsparse_spsort_descr() const
        {
            return m_d;
        }
    };

    // Runs one stage of rocsparse_spsort with a buffer of the size the library reports.
    rocsparse_status sort_stage(rocsparse_handle            handle,
                                rocsparse_spsort_descr      descr,
                                rocsparse_const_spmat_descr src,
                                rocsparse_spmat_descr       dst,
                                rocsparse_spsort_stage      stage)
    {
        size_t size = std::numeric_limits<size_t>::max();
        EXPECT_EQ(rocsparse_spsort_buffer_size(handle, descr, src, dst, stage, &size, nullptr),
                  st_ok);
        EXPECT_NE(size, std::numeric_limits<size_t>::max());
        DevBuf<char> buffer(size);
        // The size of the scratch space is an implementation detail; a zero size comes
        // with a null buffer.
        return rocsparse_spsort(
            handle, descr, src, dst, stage, size, size ? buffer.get() : nullptr, nullptr);
    }

    // Sets an input of a spsort descriptor.
    template <typename V>
    rocsparse_status set_input(rocsparse_handle       h,
                               rocsparse_spsort_descr d,
                               rocsparse_spsort_input in,
                               const V&               v)
    {
        return rocsparse_spsort_set_input(h, d, in, &v, sizeof(v), nullptr);
    }

    // Sorts src into dst through the public API (analysis, then compute).
    rocsparse_status run_spsort(rocsparse_handle            handle,
                                rocsparse_const_spmat_descr src,
                                rocsparse_spmat_descr       dst,
                                rocsparse_spsort_alg        alg,
                                const rocsparse_direction*  dir)
    {
        SpsortDescr descr(handle);
        EXPECT_EQ(set_input(handle, descr, rocsparse_spsort_input_alg, alg), st_ok);
        if(dir != nullptr)
        {
            EXPECT_EQ(set_input(handle, descr, rocsparse_spsort_input_direction, *dir), st_ok);
        }
        EXPECT_EQ(sort_stage(handle, descr, src, dst, stg_ana), st_ok);
        const auto status = sort_stage(handle, descr, src, dst, stg_cmp);
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        return status;
    }

    enum class Kind
    {
        coo,
        csr,
        csc
    };

    struct SortCfg
    {
        Kind                 kind;
        rocsparse_direction  dir; // COO only
        int                  base;
        int                  batch;
        int64_t              stride1; // stride of the offsets (CSR / CSC), 0: shared by the batches
        int64_t              stride2; // stride of the indices and the values
        bool                 in_place;
        bool                 broadcast; // a single source matrix sorted into every batch
        rocsparse_spsort_alg alg;
    };

    // Sorts a scrambled COO matrix (P == J) or a CSR / CSC matrix with unsorted slices
    // and compares the offsets / rows, the indices and the values with the host sort.
    template <typename P, typename J, typename T>
    void run_sort(rocsparse_handle handle, const SortCfg& cfg)
    {
        const bool coo = cfg.kind == Kind::coo;
        // COO: M x N, two index arrays of nnz entries. CSR / CSC: the major dimension is
        // M (rows of a CSR, columns of a CSC matrix), the offsets have M + 1 entries.
        const int  M = coo ? 23 : 37, N = coo ? 17 : 13;
        const auto pos = scrambled_positions(M, N);
        const auto cmp = make_compressed(M, N);
        const int  nnz = coo ? pos.size() : cmp.minor.size();
        ASSERT_GT(nnz, 20);
        const int len1 = coo ? nnz : M + 1;

        // The zero based reference arrays (first: rows / offsets, second: columns / minor
        // indices) and the permutation the sort has to apply.
        std::vector<int> first(len1), second(nnz), perm(nnz);
        for(int k = 0; k < len1; ++k)
            first[k] = coo ? pos[k].first : cmp.offsets[k];
        for(int k = 0; k < nnz; ++k)
            second[k] = coo ? pos[k].second : cmp.minor[k];
        std::iota(perm.begin(), perm.end(), 0);
        if(coo)
        {
            std::stable_sort(perm.begin(), perm.end(), [&](int a, int b) {
                return cfg.dir == dir_r ? pos[a] < pos[b]
                                        : std::make_pair(pos[a].second, pos[a].first)
                                              < std::make_pair(pos[b].second, pos[b].first);
            });
        }
        else
        {
            for(int r = 0; r < M; ++r)
                std::stable_sort(perm.begin() + cmp.offsets[r],
                                 perm.begin() + cmp.offsets[r + 1],
                                 [&](int a, int b) { return second[a] < second[b]; });
        }

        const int     Bd = cfg.batch, Bs = cfg.broadcast ? 1 : cfg.batch;
        const int64_t s1_src = cfg.broadcast ? 0 : cfg.stride1,
                      s2_src = cfg.broadcast ? 0 : cfg.stride2;
        const size_t n1_src = (Bs - 1) * s1_src + len1, n2_src = (Bs - 1) * s2_src + nnz;
        const size_t n1_dst = (Bd - 1) * cfg.stride1 + len1, n2_dst = (Bd - 1) * cfg.stride2 + nnz;
        const P      sent_p = static_cast<P>(-5);
        const J      sent_j = static_cast<J>(-5);
        const T      sent_v = make_val<T>(-5, -5);

        std::vector<P> h1(n1_src, sent_p);
        std::vector<J> h2(n2_src, sent_j);
        std::vector<T> h_val(n2_src, sent_v);
        for(int b = 0; b < Bs; ++b)
        {
            for(int k = 0; k < len1 && (b == 0 || s1_src != 0); ++k)
                h1[b * s1_src + k] = static_cast<P>(first[k] + cfg.base);
            for(int k = 0; k < nnz; ++k)
            {
                h2[b * s2_src + k]    = static_cast<J>(second[k] + cfg.base);
                h_val[b * s2_src + k] = make_val<T>(1 + k + 1000 * b, k % 5);
            }
        }

        // Expected target: untouched padding, sorted entries (the offsets / rows are
        // permuted for a COO matrix only).
        std::vector<P> e1    = cfg.in_place ? h1 : std::vector<P>(n1_dst, sent_p);
        std::vector<J> e2    = cfg.in_place ? h2 : std::vector<J>(n2_dst, sent_j);
        std::vector<T> e_val = cfg.in_place ? h_val : std::vector<T>(n2_dst, sent_v);
        for(int b = 0; b < Bd; ++b)
        {
            for(int k = 0; k < len1 && (b == 0 || cfg.stride1 != 0); ++k)
                e1[b * cfg.stride1 + k] = static_cast<P>(first[coo ? perm[k] : k] + cfg.base);
            for(int k = 0; k < nnz; ++k)
            {
                const int p = perm[k], at = b * cfg.stride2 + k;
                e2[at]    = static_cast<J>(second[p] + cfg.base);
                e_val[at] = make_val<T>(1 + p + 1000 * (cfg.broadcast ? 0 : b), p % 5);
            }
        }

        DevBuf<P> d1(h1), t1(cfg.in_place ? std::vector<P>() : std::vector<P>(n1_dst, sent_p));
        DevBuf<J> d2(h2), t2(cfg.in_place ? std::vector<J>() : std::vector<J>(n2_dst, sent_j));
        DevBuf<T> dv(h_val), tv(cfg.in_place ? std::vector<T>() : std::vector<T>(n2_dst, sent_v));
        P*        p1 = cfg.in_place ? d1.get() : t1.get();
        J*        p2 = cfg.in_place ? d2.get() : t2.get();
        T*        pv = cfg.in_place ? dv.get() : tv.get();

        const auto base = base_of(cfg.base);
        const auto it = it_of<J>(), pt = it_of<P>();
        const auto dt = dt_of<T>();
        Spmat      src, dst;
        if(coo)
        {
            ASSERT_EQ(rocsparse_create_coo_descr(&src, M, N, nnz, d1, d2, dv, it, base, dt), st_ok);
            ASSERT_EQ(rocsparse_create_coo_descr(&dst, M, N, nnz, p1, p2, pv, it, base, dt), st_ok);
            ASSERT_EQ(rocsparse_coo_set_strided_batch(src, Bs, s2_src), st_ok);
            ASSERT_EQ(rocsparse_coo_set_strided_batch(dst, Bd, cfg.stride2), st_ok);
        }
        else if(cfg.kind == Kind::csr)
        {
            ASSERT_EQ(rocsparse_create_csr_descr(&src, M, N, nnz, d1, d2, dv, pt, it, base, dt),
                      st_ok);
            ASSERT_EQ(rocsparse_create_csr_descr(&dst, M, N, nnz, p1, p2, pv, pt, it, base, dt),
                      st_ok);
            ASSERT_EQ(rocsparse_csr_set_strided_batch(src, Bs, s1_src, s2_src), st_ok);
            ASSERT_EQ(rocsparse_csr_set_strided_batch(dst, Bd, cfg.stride1, cfg.stride2), st_ok);
        }
        else
        {
            ASSERT_EQ(rocsparse_create_csc_descr(&src, N, M, nnz, d1, d2, dv, pt, it, base, dt),
                      st_ok);
            ASSERT_EQ(rocsparse_create_csc_descr(&dst, N, M, nnz, p1, p2, pv, pt, it, base, dt),
                      st_ok);
            ASSERT_EQ(rocsparse_csc_set_strided_batch(src, Bs, s1_src, s2_src), st_ok);
            ASSERT_EQ(rocsparse_csc_set_strided_batch(dst, Bd, cfg.stride1, cfg.stride2), st_ok);
        }

        ASSERT_EQ(run_spsort(handle, src, dst, cfg.alg, coo ? &cfg.dir : nullptr), st_ok);
        EXPECT_EQ(to_host(p1, n1_dst), e1);
        EXPECT_EQ(to_host(p2, n2_dst), e2);
        EXPECT_EQ(to_host(pv, n2_dst), e_val);
    }

    // In place / out of place, both bases and algorithms, single and strided batches, and
    // a single source sorted into every batch.
    template <typename P, typename J, typename T>
    void run_sort_grid(rocsparse_handle handle, bool coo)
    {
        // kind, dir, base, batch, stride1, stride2, in_place, broadcast, alg
        auto cfgs = [&](Kind k) -> std::vector<SortCfg> {
            if(k == Kind::coo)
                return {{k, dir_r, 0, 1, 0, 0, false, false, alg_def},
                        {k, dir_c, 1, 1, 0, 0, false, false, alg_rdx},
                        {k, dir_r, 1, 1, 0, 0, true, false, alg_rdx},
                        {k, dir_c, 0, 1, 0, 0, true, false, alg_def},
                        {k, dir_r, 0, 3, 130, 130, false, false, alg_def},
                        {k, dir_c, 1, 2, 300, 300, true, false, alg_rdx},
                        {k, dir_r, 1, 3, 125, 125, false, true, alg_def}};
            return {{k, dir_r, 0, 1, 0, 0, false, false, alg_def},
                    {k, dir_r, 1, 1, 0, 0, false, false, alg_rdx},
                    {k, dir_r, 1, 1, 0, 0, true, false, alg_def},
                    {k, dir_r, 0, 1, 0, 0, true, false, alg_rdx},
                    // shared offsets, strided indices and values
                    {k, dir_r, 0, 3, 0, 140, false, false, alg_def},
                    // one offsets array per batch, padded
                    {k, dir_r, 1, 2, 50, 150, true, false, alg_rdx},
                    {k, dir_r, 0, 3, 40, 135, false, false, alg_def},
                    // a single source sorted into every batch
                    {k, dir_r, 1, 3, 0, 140, false, true, alg_def},
                    {k, dir_r, 0, 2, 41, 133, false, true, alg_rdx}};
        };
        if(coo)
        {
            for(const SortCfg& c : cfgs(Kind::coo))
                run_sort<P, P, T>(handle, c);
            return;
        }
        for(Kind k : {Kind::csr, Kind::csc})
            for(const SortCfg& c : cfgs(k))
                run_sort<P, J, T>(handle, c);
    }

    TEST_F(ConvSort, SpsortCoo)
    {
        run_sort_grid<int32_t, int32_t, float>(handle, true);
        run_sort_grid<int64_t, int64_t, rocsparse_double_complex>(handle, true);
    }

    TEST_F(ConvSort, SpsortCsx)
    {
        run_sort_grid<int32_t, int64_t, double>(handle, false);
        run_sort_grid<int64_t, int32_t, rocsparse_float_complex>(handle, false);
        run_sort_grid<int64_t, int64_t, rocsparse_double_complex>(handle, false);
    }

    TEST_F(ConvSort, SpsortDescrInputs)
    {
        rocsparse_spsort_descr tmp = nullptr;
        EXPECT_EQ(rocsparse_spsort_descr_create(nullptr, &tmp, nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_spsort_descr_create(handle, nullptr, nullptr), st_ptr);

        SpsortDescr descr(handle);
        const auto  alg     = alg_rdx;
        const auto  dir     = dir_c;
        const auto  bad_dir = static_cast<rocsparse_direction>(100);
        const auto  in
            = [&](rocsparse_handle       h,
                  rocsparse_spsort_descr d,
                  rocsparse_spsort_input i,
                  const void*            v,
                  size_t n) { return rocsparse_spsort_set_input(h, d, i, v, n, nullptr); };
        constexpr auto in_alg = rocsparse_spsort_input_alg,
                       in_dir = rocsparse_spsort_input_direction;
        // (invalid_handle is not in the list of documented statuses)
        EXPECT_REJECTED(in(nullptr, descr, in_alg, &alg, sizeof(alg)));
        EXPECT_EQ(in(handle, nullptr, in_alg, &alg, sizeof(alg)), st_ptr);
        EXPECT_EQ(in(handle, descr, static_cast<rocsparse_spsort_input>(100), &alg, sizeof(alg)),
                  st_val);
        EXPECT_EQ(in(handle, descr, in_alg, nullptr, sizeof(alg)), st_ptr);
        EXPECT_EQ(in(handle, descr, in_alg, &alg, sizeof(alg) - 1), st_size);
        EXPECT_EQ(in(handle, descr, in_dir, &dir, sizeof(dir) + 1), st_size);
        EXPECT_REJECTED(in(handle, descr, in_dir, &bad_dir, sizeof(bad_dir)));
        EXPECT_EQ(set_input(handle, descr, in_alg, alg), st_ok);
        EXPECT_EQ(set_input(handle, descr, in_dir, dir), st_ok);

        // Once the analysis has run, the inputs can no longer be changed, and the
        // analysis cannot run twice.
        DevBuf<int32_t> d_ptr(std::vector<int32_t>{0, 2, 4}),
            d_ind(std::vector<int32_t>{1, 0, 1, 0});
        DevBuf<float> d_val(std::vector<float>{1, 2, 3, 4});
        Spmat         mat;
        ASSERT_EQ(
            rocsparse_create_csr_descr(&mat, 2, 2, 4, d_ptr, d_ind, d_val, i32, i32, base0, f32),
            st_ok);
        ASSERT_EQ(sort_stage(handle, descr, mat, mat, stg_ana), st_ok);
        EXPECT_EQ(set_input(handle, descr, in_alg, alg), st_val);
        EXPECT_EQ(set_input(handle, descr, in_dir, dir), st_val);
        EXPECT_REJECTED(rocsparse_spsort(handle, descr, mat, mat, stg_ana, 0, nullptr, nullptr));

        EXPECT_EQ(rocsparse_spsort_descr_destroy(nullptr, descr, nullptr),
                  rocsparse_status_invalid_handle);
        // A null descriptor is accepted.
        EXPECT_EQ(rocsparse_spsort_descr_destroy(handle, nullptr, nullptr), st_ok);
    }

    TEST_F(ConvSort, SpsortInvalidArguments)
    {
        const int32_t              M = 3, N = 4, nnz = 5;
        const std::vector<int32_t> ptr = {0, 2, 3, 5}, ind = {3, 1, 2, 2, 0};
        DevBuf<int32_t>            d_ptr(ptr), d_ind(ind), d_ptr2(ptr), d_ind2(ind), d_ival(ind);
        DevBuf<float>              d_val(std::vector<float>{1, 2, 3, 4, 5}),
            d_val2(std::vector<float>{1, 2, 3, 4, 5});
        DevBuf<int64_t> d_ind64(std::vector<int64_t>(ind.begin(), ind.end()));

        const auto csr = [&](Spmat&               d,
                             int32_t              cols,
                             void*                p,
                             void*                i,
                             void*                v,
                             rocsparse_indextype  jt   = i32,
                             rocsparse_index_base base = base0,
                             rocsparse_datatype   dt   = f32) {
            ASSERT_EQ(rocsparse_create_csr_descr(&d, M, cols, nnz, p, i, v, i32, jt, base, dt),
                      st_ok);
        };
        Spmat src, dst;
        csr(src, N, d_ptr, d_ind, d_val);
        csr(dst, N, d_ptr2, d_ind2, d_val2);

        SpsortDescr descr(handle);
        size_t      size = 0;
        const auto  bsz  = [&](rocsparse_spsort_descr      d,
                             rocsparse_const_spmat_descr s,
                             rocsparse_spmat_descr       t,
                             rocsparse_spsort_stage      st,
                             size_t*                     out) {
            return rocsparse_spsort_buffer_size(handle, d, s, t, st, out, nullptr);
        };
        const auto sort
            = [&](rocsparse_spsort_descr      d,
                  rocsparse_const_spmat_descr s,
                  rocsparse_spmat_descr       t,
                  rocsparse_spsort_stage      st,
                  size_t                      n,
                  void* buf) { return rocsparse_spsort(handle, d, s, t, st, n, buf, nullptr); };
        const auto bad_stage = static_cast<rocsparse_spsort_stage>(100);

        // Null arguments.
        EXPECT_EQ(rocsparse_spsort_buffer_size(nullptr, descr, src, dst, stg_cmp, &size, nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(bsz(nullptr, src, dst, stg_cmp, &size), st_ptr);
        EXPECT_EQ(bsz(descr, nullptr, dst, stg_cmp, &size), st_ptr);
        EXPECT_EQ(bsz(descr, src, nullptr, stg_cmp, &size), st_ptr);
        EXPECT_REJECTED(bsz(descr, src, dst, bad_stage, &size));
        EXPECT_EQ(bsz(descr, src, dst, stg_cmp, nullptr), st_ptr);

        // The algorithm has not been set.
        EXPECT_REJECTED(bsz(descr, src, dst, stg_cmp, &size));
        ASSERT_EQ(set_input(handle, descr, rocsparse_spsort_input_alg, alg_def), st_ok);
        EXPECT_EQ(bsz(descr, src, dst, stg_cmp, &size), st_ok);

        // Targets that do not match the source: size, index type, index base, format.
        {
            Spmat other;
            csr(other, N + 1, d_ptr2, d_ind2, d_val2);
            EXPECT_EQ(bsz(descr, src, other, stg_cmp, &size), st_size);
        }
        {
            Spmat other;
            csr(other, N, d_ptr2, d_ind64, d_val2, rocsparse_indextype_i64);
            EXPECT_EQ(bsz(descr, src, other, stg_cmp, &size), st_val);
        }
        {
            Spmat other;
            csr(other, N, d_ptr2, d_ind2, d_val2, i32, base1);
            EXPECT_EQ(bsz(descr, src, other, stg_cmp, &size), st_val);
        }
        {
            Spmat other;
            ASSERT_EQ(rocsparse_create_coo_descr(
                          &other, M, N, nnz, d_ind2, d_ind2, d_val2, i32, base0, f32),
                      st_ok);
            EXPECT_EQ(bsz(descr, src, other, stg_cmp, &size), st_val);
        }
        // Data types that the value gather does not support.
        {
            Spmat isrc, idst;
            csr(isrc, N, d_ptr, d_ind, d_ival, i32, base0, rocsparse_datatype_i32_r);
            csr(idst, N, d_ptr2, d_ind2, d_ival, i32, base0, rocsparse_datatype_i32_r);
            EXPECT_EQ(bsz(descr, isrc, idst, stg_cmp, &size), st_val);
        }

        // Batch counts must agree, and the batches of an overlapping layout are rejected.
        ASSERT_EQ(rocsparse_csr_set_strided_batch(dst, 2, 0, nnz), st_ok);
        // a single source is sorted into every batch
        EXPECT_EQ(bsz(descr, src, dst, stg_cmp, &size), st_ok);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(src, 2, M + 1, nnz), st_ok);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(dst, 3, M + 1, nnz), st_ok);
        EXPECT_EQ(bsz(descr, src, dst, stg_cmp, &size), st_val);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(dst, 2, 1, nnz), st_ok);
        EXPECT_EQ(bsz(descr, src, dst, stg_cmp, &size), st_size);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(dst, 2, M + 1, nnz - 1), st_ok);
        EXPECT_EQ(bsz(descr, src, dst, stg_cmp, &size), st_size);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(src, 1, 0, 0), st_ok);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(dst, 1, 0, 0), st_ok);

        // The stages must run in order and the buffer must be large enough.
        EXPECT_REJECTED(sort(descr, src, dst, stg_cmp, 0, nullptr));
        EXPECT_EQ(sort_stage(handle, descr, src, dst, stg_ana), st_ok);
        ASSERT_EQ(bsz(descr, src, dst, stg_cmp, &size), st_ok);
        DevBuf<char> buffer(size);
        if(size > 0)
        {
            EXPECT_REJECTED(sort(descr, src, dst, stg_cmp, 0, buffer));
            EXPECT_EQ(sort(descr, src, dst, stg_cmp, size, nullptr), st_ptr);
            EXPECT_REJECTED(sort(descr, src, dst, stg_cmp, size - 1, buffer));
        }
        EXPECT_EQ(sort(descr, src, dst, stg_cmp, size, size ? buffer.get() : nullptr), st_ok);
        // The analysis cannot follow the compute stage.
        EXPECT_REJECTED(sort(descr, src, dst, stg_ana, 0, nullptr));
        EXPECT_EQ(d_ptr2.read(), ptr);
        EXPECT_EQ(d_ind2.read(), (std::vector<int32_t>{1, 3, 2, 0, 2}));
        EXPECT_EQ(d_val2.read(), (std::vector<float>{2, 1, 3, 5, 4}));

        // Argument checks of rocsparse_spsort itself.
        EXPECT_EQ(rocsparse_spsort(nullptr, descr, src, dst, stg_cmp, 0, nullptr, nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(sort(nullptr, src, dst, stg_cmp, 0, nullptr), st_ptr);
        EXPECT_EQ(sort(descr, nullptr, dst, stg_cmp, 0, nullptr), st_ptr);
        EXPECT_EQ(sort(descr, src, nullptr, stg_cmp, 0, nullptr), st_ptr);
        EXPECT_REJECTED(sort(descr, src, dst, bad_stage, 0, nullptr));
    }

    // The COO format needs the direction and rejects overlapping batches; the other COO
    // layout is not supported.
    TEST_F(ConvSort, SpsortCooDirectionAndUnsupportedFormats)
    {
        const int32_t   nnz = 3;
        DevBuf<int32_t> d_row(std::vector<int32_t>{1, 0, 1}), d_col(std::vector<int32_t>{0, 1, 1}),
            d_row2(std::vector<int32_t>{1, 0, 1}), d_col2(std::vector<int32_t>{0, 1, 1}),
            d_aos(std::vector<int32_t>{1, 0, 0, 1, 1, 1}), d_aos2(2 * nnz);
        DevBuf<float> d_val(std::vector<float>{1, 2, 3}), d_val2(std::vector<float>{1, 2, 3});
        Spmat         src, dst, aos_src, aos_dst;
        ASSERT_EQ(rocsparse_create_coo_descr(&src, 2, 2, nnz, d_row, d_col, d_val, i32, base0, f32),
                  st_ok);
        ASSERT_EQ(
            rocsparse_create_coo_descr(&dst, 2, 2, nnz, d_row2, d_col2, d_val2, i32, base0, f32),
            st_ok);
        ASSERT_EQ(
            rocsparse_create_coo_aos_descr(&aos_src, 2, 2, nnz, d_aos, d_val, i32, base0, f32),
            st_ok);
        ASSERT_EQ(
            rocsparse_create_coo_aos_descr(&aos_dst, 2, 2, nnz, d_aos2, d_val2, i32, base0, f32),
            st_ok);

        SpsortDescr descr(handle);
        ASSERT_EQ(set_input(handle, descr, rocsparse_spsort_input_alg, alg_def), st_ok);
        size_t     size = 0;
        const auto bsz
            = [&](rocsparse_spmat_descr s, rocsparse_spmat_descr t, rocsparse_spsort_stage st) {
                  return rocsparse_spsort_buffer_size(handle, descr, s, t, st, &size, nullptr);
              };
        EXPECT_EQ(bsz(aos_src, aos_dst, stg_ana), rocsparse_status_not_implemented);
        EXPECT_EQ(rocsparse_spsort(handle, descr, aos_src, aos_dst, stg_ana, 0, nullptr, nullptr),
                  rocsparse_status_not_implemented);

        // Only the COO format uses the direction, and it is mandatory there.
        EXPECT_EQ(bsz(src, dst, stg_cmp), st_val);
        ASSERT_EQ(set_input(handle, descr, rocsparse_spsort_input_direction, dir_r), st_ok);
        EXPECT_EQ(bsz(src, dst, stg_cmp), st_ok);
        // Overlapping batches.
        ASSERT_EQ(rocsparse_coo_set_strided_batch(src, 2, nnz - 1), st_ok);
        ASSERT_EQ(rocsparse_coo_set_strided_batch(dst, 2, nnz - 1), st_ok);
        EXPECT_EQ(bsz(src, dst, stg_cmp), st_size);
        ASSERT_EQ(rocsparse_coo_set_strided_batch(dst, 2, nnz), st_ok);
        EXPECT_EQ(bsz(src, dst, stg_cmp), st_size);
        ASSERT_EQ(rocsparse_coo_set_strided_batch(src, 2, nnz), st_ok);
        EXPECT_EQ(bsz(src, dst, stg_cmp), st_ok);
    }
#endif // ROCSPARSE_WITH_SPSORT

    // ------------------------------------------------------------------
    // Legacy sort routines
    // ------------------------------------------------------------------

    TEST_F(ConvSort, CoosortLegacy)
    {
        const I    M = 23, N = 17;
        const auto pos = scrambled_positions(M, N);
        const I    nnz = pos.size();

        for(bool by_row : {true, false})
            for(bool with_perm : {true, false})
            {
                SCOPED_TRACE("by_row " + std::to_string(by_row) + " perm "
                             + std::to_string(with_perm));
                std::vector<I> h_row(nnz), h_col(nnz), ref(nnz), e_row(nnz), e_col(nnz);
                for(I k = 0; k < nnz; ++k)
                    std::tie(h_row[k], h_col[k]) = pos[k];
                std::iota(ref.begin(), ref.end(), 0);
                std::stable_sort(ref.begin(), ref.end(), [&](I a, I b) {
                    return by_row ? std::make_pair(h_row[a], h_col[a])
                                        < std::make_pair(h_row[b], h_col[b])
                                  : std::make_pair(h_col[a], h_row[a])
                                        < std::make_pair(h_col[b], h_row[b]);
                });
                for(I k = 0; k < nnz; ++k)
                    std::tie(e_row[k], e_col[k]) = pos[ref[k]];

                DevBuf<I> d_row(h_row), d_col(h_col), d_perm(nnz);
                size_t    size = 0;
                ASSERT_EQ(rocsparse_coosort_buffer_size(handle, M, N, nnz, d_row, d_col, &size),
                          st_ok);
                DevBuf<char> d_buf(size);
                if(with_perm)
                {
                    ASSERT_EQ(rocsparse_create_identity_permutation(handle, nnz, d_perm), st_ok);
                }
                const auto sort = by_row ? rocsparse_coosort_by_row : rocsparse_coosort_by_column;
                ASSERT_EQ(
                    sort(
                        handle, M, N, nnz, d_row, d_col, with_perm ? d_perm.get() : nullptr, d_buf),
                    st_ok);
                EXPECT_EQ(d_row.read(), e_row);
                EXPECT_EQ(d_col.read(), e_col);
                if(with_perm)
                {
                    EXPECT_EQ(d_perm.read(), ref);
                }
            }

        // Empty matrices return immediately.
        DevBuf<I>    d_row(std::vector<I>{0}), d_col(std::vector<I>{0});
        DevBuf<char> d_buf(64);
        for(auto sort : {rocsparse_coosort_by_row, rocsparse_coosort_by_column})
        {
            EXPECT_EQ(sort(handle, 0, N, 0, d_row, d_col, nullptr, d_buf), st_ok);
            EXPECT_EQ(sort(handle, M, 0, 0, d_row, d_col, nullptr, d_buf), st_ok);
            EXPECT_EQ(sort(handle, M, N, 0, d_row, d_col, nullptr, d_buf), st_ok);
            // Argument validation.
            EXPECT_EQ(sort(nullptr, M, N, 1, d_row, d_col, nullptr, d_buf),
                      rocsparse_status_invalid_handle);
            EXPECT_EQ(sort(handle, -1, N, 1, d_row, d_col, nullptr, d_buf), st_size);
            EXPECT_EQ(sort(handle, M, -1, 1, d_row, d_col, nullptr, d_buf), st_size);
            expect_null_args(
                [&](I* r, I* c, void* b) { return sort(handle, M, N, 1, r, c, nullptr, b); },
                d_row.get(),
                d_col.get(),
                static_cast<void*>(d_buf.get()));
        }
    }

    TEST_F(ConvSort, CsrCscsortLegacy)
    {
        const I    M = 37, N = 13;
        const auto c   = make_compressed(M, N);
        const I    nnz = c.minor.size();

        for(bool csc : {false, true})
            for(I base : {0, 1})
                for(bool with_perm : {true, false})
                {
                    SCOPED_TRACE("csc " + std::to_string(csc) + " base " + std::to_string(base)
                                 + " perm " + std::to_string(with_perm));
                    MatDescr       descr(base_of(base));
                    std::vector<I> h_off(c.offsets.begin(), c.offsets.end()),
                        h_ind(c.minor.begin(), c.minor.end()), ref(nnz), e_ind(nnz);
                    for(I& v : h_off)
                        v += base;
                    for(I& v : h_ind)
                        v += base;
                    std::iota(ref.begin(), ref.end(), 0);
                    for(I r = 0; r < M; ++r)
                        std::stable_sort(ref.begin() + c.offsets[r],
                                         ref.begin() + c.offsets[r + 1],
                                         [&](I a, I b) { return c.minor[a] < c.minor[b]; });
                    for(I k = 0; k < nnz; ++k)
                        e_ind[k] = c.minor[ref[k]] + base;

                    // CSR has M rows and N columns, CSC has N rows and M columns.
                    const I    rows = csc ? N : M, cols = csc ? M : N;
                    DevBuf<I>  d_off(h_off), d_ind(h_ind), d_perm(nnz);
                    size_t     size = 0;
                    const auto buffer_size
                        = csc ? rocsparse_cscsort_buffer_size : rocsparse_csrsort_buffer_size;
                    ASSERT_EQ(buffer_size(handle, rows, cols, nnz, d_off, d_ind, &size), st_ok);
                    DevBuf<char> d_buf(size);
                    if(with_perm)
                    {
                        ASSERT_EQ(rocsparse_create_identity_permutation(handle, nnz, d_perm),
                                  st_ok);
                    }
                    const auto sort = csc ? rocsparse_cscsort : rocsparse_csrsort;
                    ASSERT_EQ(sort(handle,
                                   rows,
                                   cols,
                                   nnz,
                                   descr,
                                   d_off,
                                   d_ind,
                                   with_perm ? d_perm.get() : nullptr,
                                   d_buf),
                              st_ok);
                    EXPECT_EQ(d_off.read(), h_off);
                    EXPECT_EQ(d_ind.read(), e_ind);
                    if(with_perm)
                    {
                        EXPECT_EQ(d_perm.read(), ref);
                    }
                }
    }
} // namespace
