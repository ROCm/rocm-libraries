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

// Sparse matrices of every format (coo, coo_aos, csr, csc, ell, bsr) built from one dense
// matrix, used to check rocsparse_sparse_to_sparse over all pairs of formats and
// rocsparse_check_spmat on every format. Converted matrices are read back, expanded to a
// dense matrix on the host and compared with the dense matrix the source was built from.

#include "unit_test_device_conv_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>

using namespace convut;

namespace
{
    using ConvSpmat = ConvTest;

    constexpr int64_t M = 12, N = 12;

    constexpr auto fmt_coo = rocsparse_format_coo, fmt_aos = rocsparse_format_coo_aos,
                   fmt_csr = rocsparse_format_csr, fmt_csc = rocsparse_format_csc,
                   fmt_ell = rocsparse_format_ell, fmt_bsr = rocsparse_format_bsr;

    constexpr std::array s_formats = {fmt_coo, fmt_aos, fmt_csr, fmt_csc, fmt_ell, fmt_bsr};

    std::string format_name(rocsparse_format f)
    {
        constexpr std::array names = {"coo", "coo_aos", "csr", "csc", "ell", "bsr"};
        const auto           it    = std::find(s_formats.begin(), s_formats.end(), f);
        return it == s_formats.end() ? "other" : names[it - s_formats.begin()];
    }

    // Host description of a sparse matrix. For a BSR matrix rows / cols are the block
    // dimensions and nnz is the number of blocks; for an ELL matrix nnz is width * rows.
    template <typename T, typename I>
    struct HostMat
    {
        rocsparse_format     format = fmt_csr;
        rocsparse_index_base base   = base0;
        rocsparse_direction  dir    = dir_r;
        int64_t              rows = M, cols = N, nnz = 0, width = 0, bd = 1;
        std::vector<I>       ptr, ind, row;
        std::vector<T>       val;
    };

    // Dense matrix (row-major) with an irregular pattern: an empty row, rows of different
    // lengths and a full diagonal elsewhere.
    template <typename T>
    std::vector<T> make_dense()
    {
        std::vector<T> dense(M * N, T{});
        for(int64_t i = 0; i < M; ++i)
            for(int64_t j = 0; j < N && i != 7; ++j)
                if((i + 2 * j) % 5 == 0 || i == j || (i == 3 && j == 11))
                    dense[i * N + j] = make_val<T>(1 + i * N + j, i - j);
        return dense;
    }

    template <typename T, typename I>
    HostMat<T, I> build(rocsparse_format      format,
                        rocsparse_index_base  base,
                        int64_t               bd,
                        rocsparse_direction   dir,
                        const std::vector<T>& dense)
    {
        HostMat<T, I> h;
        h.format       = format;
        h.base         = base;
        h.dir          = dir;
        const I    b   = static_cast<I>(base);
        const T    z   = T{};
        const bool csc = format == fmt_csc, csx = csc || format == fmt_csr;

        if(format == fmt_ell)
        {
            h.width = 1;
            for(int64_t i = 0; i < M; ++i)
                h.width = std::max<int64_t>(
                    h.width,
                    std::count_if(&dense[i * N], &dense[i * N] + N, [&](T v) { return v != z; }));
            h.nnz = h.width * M;
            // rocSPARSE marks ELL padding with -1 for either index base
            h.ind.assign(h.nnz, static_cast<I>(-1));
            h.val.assign(h.nnz, z);
            for(int64_t i = 0; i < M; ++i)
                for(int64_t j = 0, k = 0; j < N; ++j)
                    if(dense[i * N + j] != z)
                    {
                        h.ind[i + k * M] = I(j) + b;
                        h.val[i + k * M] = dense[i * N + j];
                        ++k;
                    }
        }
        else if(format == fmt_bsr)
        {
            h.bd   = bd;
            h.rows = M / bd;
            h.cols = N / bd;
            h.ptr.push_back(b);
            for(int64_t ib = 0; ib < h.rows; ++ib)
            {
                for(int64_t jb = 0; jb < h.cols; ++jb)
                {
                    const auto at = [&](int64_t r, int64_t c) {
                        return dense[(ib * bd + r) * N + jb * bd + c];
                    };
                    bool any = false;
                    for(int64_t r = 0; r < bd; ++r)
                        for(int64_t c = 0; c < bd; ++c)
                            any = any || at(r, c) != z;
                    if(!any)
                        continue;
                    h.ind.push_back(I(jb) + b);
                    h.val.resize(h.val.size() + bd * bd, z);
                    for(int64_t r = 0; r < bd; ++r)
                        for(int64_t c = 0; c < bd; ++c)
                            h.val[(h.ind.size() - 1) * bd * bd
                                  + (dir == dir_r ? r * bd + c : c * bd + r)]
                                = at(r, c);
                }
                h.ptr.push_back(I(h.ind.size()) + b);
            }
            h.nnz = h.ind.size();
        }
        else
        {
            if(csx)
                h.ptr.push_back(b);
            for(int64_t p = 0; p < (csc ? N : M); ++p)
            {
                for(int64_t q = 0; q < (csc ? M : N); ++q)
                {
                    const int64_t i = csc ? q : p, j = csc ? p : q;
                    if(dense[i * N + j] == z)
                        continue;
                    if(format == fmt_coo)
                        h.row.push_back(I(i) + b);
                    if(csx)
                        h.ind.push_back(I(q) + b);
                    else
                    {
                        h.ind.push_back((format == fmt_coo ? I(j) : I(i)) + b);
                        if(format == fmt_aos)
                            h.ind.push_back(I(j) + b);
                    }
                    h.val.push_back(dense[i * N + j]);
                }
                if(csx)
                    h.ptr.push_back(I(h.val.size()) + b);
            }
            h.nnz = h.val.size();
        }
        return h;
    }

    // Expands a host matrix to a dense row-major M x N matrix.
    template <typename T, typename I>
    std::vector<T> to_dense(const HostMat<T, I>& h)
    {
        std::vector<T> dense(M * N, T{});
        const int64_t  b   = static_cast<int64_t>(h.base);
        const auto     put = [&](int64_t i, int64_t j, const T& v) { dense[i * N + j] = v; };
        if(h.format == fmt_coo || h.format == fmt_aos)
        {
            for(int64_t k = 0; k < h.nnz; ++k)
            {
                if(h.format == fmt_coo)
                    put(h.row[k] - b, h.ind[k] - b, h.val[k]);
                else
                    put(h.ind[2 * k] - b, h.ind[2 * k + 1] - b, h.val[k]);
            }
        }
        else if(h.format == fmt_ell)
        {
            for(int64_t i = 0; i < M; ++i)
                for(int64_t k = 0; k < h.width; ++k)
                    if(h.ind[i + k * M] - b >= 0)
                        put(i, h.ind[i + k * M] - b, h.val[i + k * M]);
        }
        else // csr, csc and bsr (bd = 1 for the first two)
        {
            const int64_t bb = h.bd * h.bd;
            for(int64_t p = 0; p < (h.format == fmt_csc ? N : h.rows); ++p)
                for(int64_t k = h.ptr[p] - b; k < h.ptr[p + 1] - b; ++k)
                    for(int64_t r = 0; r < h.bd; ++r)
                        for(int64_t c = 0; c < h.bd; ++c)
                        {
                            const T& v
                                = h.val[k * bb + (h.dir == dir_r ? r * h.bd + c : c * h.bd + r)];
                            if(h.format == fmt_csc)
                                put(h.ind[k] - b, p, v);
                            else
                                put(p * h.bd + r, (h.ind[k] - b) * h.bd + c, v);
                        }
        }
        return dense;
    }

    // Device arrays of a matrix: an upload of a host matrix, or storage large enough for
    // any conversion, filled with -7 (indices) and -99 (values) so that an entry the
    // conversion does not write (zeros of a partial BSR block, ELL padding) shows.
    template <typename T, typename I>
    struct DevMat
    {
        DevBuf<I> ptr, ind, row;
        DevBuf<T> val;

        explicit DevMat(const HostMat<T, I>& h)
            : ptr(h.ptr)
            , ind(h.ind)
            , row(h.row)
            , val(h.val)
        {
        }
        DevMat()
            : ptr(std::vector<I>(32, I(-7)))
            , ind(std::vector<I>(2 * (M * N + 16), I(-7)))
            , row(std::vector<I>(M * N + 16, I(-7)))
            , val(std::vector<T>(M * N + 16, make_val<T>(-99)))
        {
        }
    };

    template <typename T, typename I>
    rocsparse_status create_descr(const HostMat<T, I>& m, DevMat<T, I>& d, Spmat& out)
    {
        const auto it = it_of<I>();
        const auto dt = dt_of<T>();
        const auto R = m.rows, C = m.cols, Z = m.nnz;
        switch(m.format)
        {
        case fmt_coo:
            return rocsparse_create_coo_descr(&out, R, C, Z, d.row, d.ind, d.val, it, m.base, dt);
        case fmt_aos:
            return rocsparse_create_coo_aos_descr(&out, R, C, Z, d.ind, d.val, it, m.base, dt);
        case fmt_csr:
            return rocsparse_create_csr_descr(
                &out, R, C, Z, d.ptr, d.ind, d.val, it, it, m.base, dt);
        case fmt_csc:
            return rocsparse_create_csc_descr(
                &out, R, C, Z, d.ptr, d.ind, d.val, it, it, m.base, dt);
        case fmt_ell:
            return rocsparse_create_ell_descr(&out, R, C, d.ind, d.val, m.width, it, m.base, dt);
        default:
            return rocsparse_create_bsr_descr(
                &out, R, C, Z, m.dir, m.bd, d.ptr, d.ind, d.val, it, it, m.base, dt);
        }
    }

    template <typename T, typename I>
    rocsparse_status set_pointers(rocsparse_format format, DevMat<T, I>& d, Spmat& out)
    {
        switch(format)
        {
        case fmt_coo:
            return rocsparse_coo_set_pointers(out, d.row, d.ind, d.val);
        case fmt_aos:
            return rocsparse_coo_aos_set_pointers(out, d.ind, d.val);
        case fmt_csr:
            return rocsparse_csr_set_pointers(out, d.ptr, d.ind, d.val);
        case fmt_csc:
            return rocsparse_csc_set_pointers(out, d.ptr, d.ind, d.val);
        case fmt_ell:
            return rocsparse_ell_set_pointers(out, d.ind, d.val);
        default:
            return rocsparse_bsr_set_pointers(out, d.ptr, d.ind, d.val);
        }
    }

    // Descriptor metadata of an (empty) target of the given format.
    template <typename T, typename I>
    HostMat<T, I> target_meta(rocsparse_format     format,
                              rocsparse_index_base base,
                              int64_t              bd,
                              rocsparse_direction  dir,
                              const HostMat<T, I>& source)
    {
        HostMat<T, I> t;
        t.format = format;
        t.base   = base;
        t.dir    = dir;
        t.bd     = format == fmt_bsr ? bd : 1;
        t.rows   = M / t.bd;
        t.cols   = N / t.bd;
        t.width  = (format == fmt_ell && source.format == fmt_ell) ? source.width : 1;
        t.nnz    = format == fmt_ell   ? t.width * M
                   : format == fmt_bsr ? (source.format == fmt_bsr ? source.nnz : 1)
                                       : source.nnz;
        return t;
    }

    // ------------------------------------------------------------------
    // sparse_to_sparse
    // ------------------------------------------------------------------

    constexpr auto s2s_ana = rocsparse_sparse_to_sparse_stage_analysis;
    constexpr auto s2s_cmp = rocsparse_sparse_to_sparse_stage_compute;
    constexpr auto s2s_def = rocsparse_sparse_to_sparse_alg_default;

    // Reads the converted target back to the host.
    template <typename T, typename I>
    HostMat<T, I> read_back(const Spmat&         descr,
                            rocsparse_format     format,
                            const DevMat<T, I>&  d,
                            const HostMat<T, I>& meta)
    {
        HostMat<T, I> h   = meta;
        int64_t       nnz = 0;
        EXPECT_EQ(rocsparse_spmat_get_nnz(descr, &nnz), st_ok);
        h.nnz = nnz;
        if(format == fmt_ell)
        {
            int64_t              rows, cols, width = 0;
            void *               ind, *val;
            rocsparse_indextype  it;
            rocsparse_index_base ib;
            rocsparse_datatype   dt;
            EXPECT_EQ(rocsparse_ell_get(descr, &rows, &cols, &ind, &val, &width, &it, &ib, &dt),
                      st_ok);
            h.width = width;
            h.nnz   = width * M;
        }
        else if(format == fmt_bsr)
        {
            // The number of blocks is taken from the row pointer.
            h.ptr = d.ptr.read(M / h.bd + 1);
            h.nnz = static_cast<int64_t>(h.ptr.back()) - static_cast<int64_t>(h.base);
        }
        else if(format == fmt_csr || format == fmt_csc)
            h.ptr = d.ptr.read((format == fmt_csr ? M : N) + 1);
        const int64_t scalars = h.nnz * h.bd * h.bd;
        h.row                 = format == fmt_coo ? d.row.read(h.nnz) : std::vector<I>();
        h.ind                 = d.ind.read(format == fmt_aos ? 2 * h.nnz : h.nnz);
        h.val                 = d.val.read(scalars);
        return h;
    }

    // Analysis, then compute of one conversion; the user sets the target arrays between
    // the two stages as the API requires. Returns false on failure.
    template <typename T, typename I>
    bool convert(rocsparse_handle            handle,
                 rocsparse_const_spmat_descr source,
                 Spmat&                      target,
                 rocsparse_format            tf,
                 DevMat<T, I>&               dev)
    {
        S2sDescr s2s;
        EXPECT_EQ(rocsparse_create_sparse_to_sparse_descr(&s2s, source, target, s2s_def), st_ok);
        if(s2s.get() == nullptr)
            return false;
        // Needed by the conversions going through an intermediate CSR matrix, harmless for
        // the others.
        EXPECT_EQ(rocsparse_sparse_to_sparse_permissive(s2s), st_ok);
        for(auto stage : {s2s_ana, s2s_cmp})
        {
            size_t size = 0;
            EXPECT_EQ(
                rocsparse_sparse_to_sparse_buffer_size(handle, s2s, source, target, stage, &size),
                st_ok);
            DevBuf<char> buffer(size);
            const auto   st
                = rocsparse_sparse_to_sparse(handle, s2s, source, target, stage, size, buffer);
            EXPECT_EQ(st, st_ok);
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
            if(st != st_ok)
                return false;
            if(stage == s2s_ana)
                EXPECT_EQ(set_pointers(tf, dev, target), st_ok);
        }
        return true;
    }

    // Number of bd x bd blocks that contain a nonzero (bd = 1: the nonzeros).
    template <typename T>
    int64_t count_blocks(const std::vector<T>& dense, int64_t bd)
    {
        int64_t cnt = 0;
        for(int64_t ib = 0; ib < M / bd; ++ib)
            for(int64_t jb = 0; jb < N / bd; ++jb)
            {
                bool any = false;
                for(int64_t r = 0; r < bd; ++r)
                    for(int64_t c = 0; c < bd; ++c)
                        any = any || dense[(ib * bd + r) * N + jb * bd + c] != T{};
                cnt += any;
            }
        return cnt;
    }

    struct PairCfg
    {
        rocsparse_index_base base_s;
        int64_t              bd_s;
        rocsparse_direction  dir_s;
        rocsparse_index_base base_t;
        int64_t              bd_t;
        rocsparse_direction  dir_t;
    };

    // Converts the matrix between every pair of formats and compares the result with the
    // dense matrix and the number of stored entries.
    template <typename T, typename I>
    void run_all_pairs(rocsparse_handle handle, const PairCfg& cfg)
    {
        const std::vector<T> dense = make_dense<T>();
        for(rocsparse_format sf : s_formats)
        {
            const HostMat<T, I> src = build<T, I>(sf, cfg.base_s, cfg.bd_s, cfg.dir_s, dense);
            ASSERT_EQ(to_dense(src), dense) << "builder " << format_name(sf);
            for(rocsparse_format tf : s_formats)
            {
                // BSR -> BSR only between equal block dimensions and directions.
                if(sf == fmt_bsr && tf == fmt_bsr
                   && (cfg.bd_s != cfg.bd_t || cfg.dir_s != cfg.dir_t))
                    continue;
                SCOPED_TRACE(format_name(sf) + " -> " + format_name(tf)
                             + " base=" + std::to_string(cfg.base_s) + "->"
                             + std::to_string(cfg.base_t) + " bd=" + std::to_string(cfg.bd_s) + "->"
                             + std::to_string(cfg.bd_t) + " dir=" + std::to_string(cfg.dir_s) + "->"
                             + std::to_string(cfg.dir_t));
                DevMat<T, I> dsrc(src), dtgt;
                Spmat        source, target;
                ASSERT_EQ(create_descr(src, dsrc, source), st_ok);
                const auto meta = target_meta<T, I>(tf, cfg.base_t, cfg.bd_t, cfg.dir_t, src);
                ASSERT_EQ(create_descr(meta, dtgt, target), st_ok);
                if(!convert<T, I>(handle, source, target, tf, dtgt))
                    continue;

                const auto out = read_back<T, I>(target, tf, dtgt, meta);
                EXPECT_EQ(to_dense(out), dense) << "nnz=" << out.nnz << " width=" << out.width;
                if(tf == fmt_ell)
                {
                    // Padding is column -1 with a zero value, every other column is valid.
                    for(size_t k = 0; k < out.ind.size(); ++k)
                    {
                        const int64_t col = static_cast<int64_t>(out.ind[k]);
                        if(col == -1)
                            EXPECT_EQ(out.val[k], T{}) << "ELL padding value " << k;
                        else
                            EXPECT_TRUE(col >= cfg.base_t && col < N + cfg.base_t)
                                << "ELL column " << k << " = " << col;
                    }
                }
                // A BSR source stores explicit zeros inside its blocks and the library is
                // free to keep or drop them: only a lower bound is checked then.
                int64_t expected = count_blocks(dense, tf == fmt_bsr ? cfg.bd_t : 1);
                if(tf == fmt_ell)
                {
                    expected = 1;
                    for(int64_t i = 0; i < M; ++i)
                        expected = std::max<int64_t>(
                            expected, std::count_if(&dense[i * N], &dense[i * N] + N, [](T v) {
                                return v != T{};
                            }));
                    expected *= M;
                }
                if(sf == fmt_bsr)
                    EXPECT_GE(out.nnz, expected)
                        << "stored entries of the " << format_name(tf) << " target";
                else
                    EXPECT_EQ(out.nnz, expected)
                        << "stored entries of the " << format_name(tf) << " target";
            }
        }
    }

    // Different block sizes / directions go through the other formats; the source and
    // the target use the same index base.
    TEST_F(ConvSpmat, SparseToSparseMatrix)
    {
        using cf = rocsparse_float_complex;
        using cd = rocsparse_double_complex;
        run_all_pairs<double, int32_t>(handle, {base0, 2, dir_r, base0, 2, dir_r});
        run_all_pairs<float, int64_t>(handle, {base1, 3, dir_c, base1, 3, dir_c});
        run_all_pairs<cf, int32_t>(handle, {base1, 4, dir_r, base1, 4, dir_r});
        run_all_pairs<cd, int64_t>(handle, {base0, 2, dir_c, base0, 2, dir_c});
        // different block size and direction
        run_all_pairs<double, int32_t>(handle, {base0, 2, dir_r, base0, 3, dir_c});
        // same block size, direction change only
        run_all_pairs<float, int32_t>(handle, {base0, 3, dir_r, base0, 3, dir_c});
        // larger block size, complex values, base one
        run_all_pairs<cf, int64_t>(handle, {base1, 4, dir_c, base1, 2, dir_r});
    }

    // The first error of the descriptor creation, the buffer size query and the analysis
    // stage of a conversion (success if there is none): a refused conversion may be
    // reported by any of them.
    rocsparse_status analysis_status(rocsparse_handle            handle,
                                     rocsparse_const_spmat_descr source,
                                     rocsparse_spmat_descr       target,
                                     bool                        permissive = false)
    {
        S2sDescr s2s;
        size_t   size = 0;
        auto     st   = rocsparse_create_sparse_to_sparse_descr(&s2s, source, target, s2s_def);
        if(st == st_ok && permissive)
            st = rocsparse_sparse_to_sparse_permissive(s2s);
        if(st == st_ok)
            st = rocsparse_sparse_to_sparse_buffer_size(
                handle, s2s, source, target, s2s_ana, &size);
        if(st == st_ok)
        {
            DevBuf<char> buffer(size);
            st = rocsparse_sparse_to_sparse(handle, s2s, source, target, s2s_ana, size, buffer);
        }
        return st;
    }

    template <typename T, typename J>
    rocsparse_status analysis_status(rocsparse_handle     handle,
                                     const HostMat<T, J>& sm,
                                     const HostMat<T, J>& tm,
                                     bool                 permissive)
    {
        DevMat<T, J> ds(sm), dt;
        Spmat        source, target;
        EXPECT_EQ(create_descr(sm, ds, source), st_ok);
        EXPECT_EQ(create_descr(tm, dt, target), st_ok);
        return analysis_status(handle, source, target, permissive);
    }

    // The source and the target descriptor must use the same index base, and conversions
    // that need an intermediate CSR matrix are refused unless the descriptor is made
    // permissive. The header does not document the status of a refused conversion.
    TEST_F(ConvSpmat, SparseToSparseRefused)
    {
        const auto dense = make_dense<double>();
        for(rocsparse_format sf : s_formats)
            for(rocsparse_format tf : s_formats)
            {
                SCOPED_TRACE(format_name(sf) + " -> " + format_name(tf));
                const auto src = build<double, int32_t>(sf, base0, 2, dir_r, dense);
                const auto tgt = target_meta<double, int32_t>(tf, base1, 2, dir_r, src);
                EXPECT_REJECTED((analysis_status<double, int32_t>(handle, src, tgt, true)));
            }
        const auto src = build<double, int32_t>(fmt_coo, base0, 2, dir_r, dense);
        const auto tgt = target_meta<double, int32_t>(fmt_ell, base0, 2, dir_r, src);
        EXPECT_REJECTED((analysis_status<double, int32_t>(handle, src, tgt, false)));
    }

    // Blocked ELL is documented as not supported by sparse_to_sparse, in either
    // direction. The header does not say which status is returned: only an error is
    // required.
    TEST_F(ConvSpmat, SparseToSparseBellNotImplemented)
    {
        using T = float;
        using I = int32_t;
        DevBuf<I>  bell_ind(M * N);
        DevBuf<T>  bell_val(M * N);
        const auto dense     = make_dense<T>();
        const auto status_of = [&](rocsparse_const_spmat_descr s, rocsparse_spmat_descr t) {
            return analysis_status(handle, s, t);
        };
        const auto make_bell = [&](Spmat& out) {
            ASSERT_EQ(rocsparse_create_bell_descr(
                          &out, M, N, dir_r, 2, 2, bell_ind, bell_val, i32, base0, f32),
                      st_ok);
        };
        Spmat b0, b1;
        make_bell(b0);
        make_bell(b1);
        EXPECT_REJECTED(status_of(b0, b1));
        for(rocsparse_format f : s_formats)
        {
            SCOPED_TRACE("format " + format_name(f));
            const auto   h = build<T, I>(f, base0, 2, dir_r, dense);
            DevMat<T, I> dev(h);
            Spmat        other;
            ASSERT_EQ(create_descr(h, dev, other), st_ok);
            EXPECT_REJECTED(status_of(other, b0));
            EXPECT_REJECTED(status_of(b0, other));
        }
    }

    // ------------------------------------------------------------------
    // check_spmat
    // ------------------------------------------------------------------

    struct Attributes
    {
        rocsparse_matrix_type  type    = rocsparse_matrix_type_general;
        rocsparse_fill_mode    fill    = rocsparse_fill_mode_lower;
        rocsparse_storage_mode storage = rocsparse_storage_mode_sorted;
    };

    constexpr auto chk_size = rocsparse_check_spmat_stage_buffer_size;
    constexpr auto chk_cmp  = rocsparse_check_spmat_stage_compute;

    // Runs both stages of rocsparse_check_spmat; returns the status of the compute stage
    // and the reported data status.
    rocsparse_status run_check(rocsparse_handle       handle,
                               rocsparse_spmat_descr  descr,
                               const Attributes&      at,
                               rocsparse_data_status* ds)
    {
        const auto set = [&](rocsparse_spmat_attribute a, const auto& v) {
            EXPECT_EQ(rocsparse_spmat_set_attribute(descr, a, &v, sizeof(v)), st_ok);
        };
        set(rocsparse_spmat_matrix_type, at.type);
        set(rocsparse_spmat_fill_mode, at.fill);
        set(rocsparse_spmat_storage_mode, at.storage);
        size_t size = 0;
        auto   st   = rocsparse_check_spmat(handle, descr, ds, chk_size, &size, nullptr);
        if(st != st_ok)
            return st;
        DevBuf<char> temp(size);
        st = rocsparse_check_spmat(handle, descr, ds, chk_cmp, &size, temp);
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        return st;
    }

    template <typename T, typename I>
    rocsparse_data_status
        check_matrix(rocsparse_handle handle, const HostMat<T, I>& h, const Attributes& at = {})
    {
        DevMat<T, I> d(h);
        Spmat        descr;
        EXPECT_EQ(create_descr(h, d, descr), st_ok);
        auto ds = bad_ds; // a value that no check reports
        EXPECT_EQ(run_check(handle, descr, at, &ds), st_ok);
        return ds;
    }

    enum class Defect
    {
        nan,
        inf,
        out_of_range,
        offsets,
        duplicate,
        unsorted
    };

    constexpr std::array<rocsparse_data_status, 6> s_defect_status
        = {rocsparse_data_status_nan,
           rocsparse_data_status_inf,
           rocsparse_data_status_invalid_index,
           rocsparse_data_status_invalid_offset_ptr,
           rocsparse_data_status_duplicate_entry,
           rocsparse_data_status_invalid_sorting};

    // Applies one defect to a matrix.
    template <typename T, typename I>
    void apply_defect(HostMat<T, I>& h, Defect defect)
    {
        const I b   = static_cast<I>(h.base);
        auto&   idx = h.ind;
        switch(defect)
        {
        case Defect::nan:
        case Defect::inf:
            h.val[3] = make_special<T>(defect == Defect::nan);
            break;
        case Defect::out_of_range:
            if(h.format == fmt_coo)
                h.row[0] = I(M) + b;
            else
                idx[0] = I(h.format == fmt_csc ? M : h.format == fmt_bsr ? h.cols : N) + b;
            break;
        case Defect::offsets: // the pointer of the third row / column drops below the previous one
            h.ptr[3] = h.ptr[2] - 1;
            break;
        case Defect::duplicate: // the first two entries of the first row / column become identical
            idx[1] = idx[0];
            break;
        case Defect::unsorted:
            std::swap(idx[0], idx[1]);
            break;
        }
    }

    // Every format of the given type and index type is accepted (both bases, both block
    // directions), and each malformed matrix, with exactly one defect, reports the
    // matching data status.
    template <typename T, typename I>
    void check_formats(rocsparse_handle handle, rocsparse_index_base defect_base)
    {
        const auto dense = make_dense<T>();
        for(rocsparse_format f : {fmt_coo, fmt_csr, fmt_csc, fmt_ell, fmt_bsr})
            for(rocsparse_index_base base : {base0, base1})
                for(rocsparse_direction dir : {dir_r, dir_c})
                {
                    if(f != fmt_bsr && dir == dir_c)
                        continue;
                    SCOPED_TRACE(format_name(f) + " base=" + std::to_string(base)
                                 + " dir=" + std::to_string(dir));
                    EXPECT_EQ((check_matrix<T, I>(handle, build<T, I>(f, base, 3, dir, dense))),
                              rocsparse_data_status_success);
                }

        using D = Defect;
        const std::vector<D> all
            = {D::nan, D::inf, D::out_of_range, D::offsets, D::duplicate, D::unsorted};
        const std::vector<std::pair<rocsparse_format, std::vector<D>>> cases
            = {{fmt_csr, all},
               {fmt_csc, all},
               {fmt_coo, {D::nan, D::inf, D::out_of_range, D::unsorted}},
               {fmt_ell, {D::nan, D::inf, D::out_of_range}},
               {fmt_bsr, {D::nan, D::inf, D::out_of_range, D::offsets}}};
        for(const auto& [f, defects] : cases)
            for(D defect : defects)
            {
                SCOPED_TRACE(format_name(f)
                             + " defect=" + std::to_string(static_cast<int>(defect)));
                auto h = build<T, I>(f, defect_base, 3, dir_r, dense);
                apply_defect(h, defect);
                EXPECT_EQ((check_matrix<T, I>(handle, h)),
                          s_defect_status[static_cast<int>(defect)]);
            }
    }

    TEST_F(ConvSpmat, CheckSpmatFormatsAndDefects)
    {
        check_formats<float, int32_t>(handle, base0);
        check_formats<rocsparse_double_complex, int64_t>(handle, base1);
        check_formats<rocsparse_float_complex, int32_t>(handle, base1);
    }

    // Fill mode of a triangular / symmetric matrix. The full dense matrix has entries on
    // both sides of the diagonal, so it is invalid for either fill mode. A matrix that
    // only holds the lower (upper) triangle is valid for the lower (upper) fill mode and
    // invalid for the opposite one.
    TEST_F(ConvSpmat, CheckSpmatFillMode)
    {
        const auto full = make_dense<double>();
        // Keeps the entries on and below (lower) or on and above (upper) the diagonal.
        const auto triangle = [&](rocsparse_fill_mode fill) {
            auto t = full;
            for(int64_t i = 0; i < M; ++i)
                for(int64_t j = 0; j < N; ++j)
                    if(fill == rocsparse_fill_mode_lower ? j > i : j < i)
                        t[i * N + j] = 0.0;
            return t;
        };
        const auto lower = triangle(rocsparse_fill_mode_lower),
                   upper = triangle(rocsparse_fill_mode_upper);
        ASSERT_NE(lower, full);
        ASSERT_NE(upper, full);
        ASSERT_NE(lower, upper);

        for(rocsparse_format f : {fmt_csr, fmt_coo})
            for(auto type : {rocsparse_matrix_type_symmetric, rocsparse_matrix_type_triangular})
                for(auto fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
                {
                    SCOPED_TRACE(format_name(f) + " type=" + std::to_string(type)
                                 + " fill=" + std::to_string(fill));
                    const auto status_of = [&](const std::vector<double>& dense) {
                        return check_matrix<double, int32_t>(
                            handle,
                            build<double, int32_t>(f, base0, 3, dir_r, dense),
                            {type, fill});
                    };
                    EXPECT_EQ(status_of(full), rocsparse_data_status_invalid_fill);

                    const auto s_lower = status_of(lower), s_upper = status_of(upper);
                    const bool want_lower = fill == rocsparse_fill_mode_lower;
                    EXPECT_EQ(want_lower ? s_lower : s_upper, rocsparse_data_status_success);
                    EXPECT_EQ(want_lower ? s_upper : s_lower, rocsparse_data_status_invalid_fill);
                }
    }

    // Row / column pointers in 64 bit with 32 bit indices are supported, the opposite is
    // not documented; and a matrix that declares unsorted storage may hold unsorted
    // indices.
    TEST_F(ConvSpmat, CheckSpmatIndexTypesAndStorage)
    {
        const auto dense = make_dense<float>();
        for(rocsparse_format f : {fmt_csr, fmt_csc, fmt_bsr})
        {
            SCOPED_TRACE(format_name(f));
            const auto      h64 = build<float, int64_t>(f, base0, 3, dir_r, dense);
            const auto      h32 = build<float, int32_t>(f, base0, 3, dir_r, dense);
            DevBuf<int64_t> d_ptr64(h64.ptr), d_ind64(h64.ind);
            DevBuf<int32_t> d_ptr32(h32.ptr), d_ind32(h32.ind);
            DevBuf<float>   d_val(h32.val);

            const auto create =
                [&](Spmat& out, void* p, void* i, rocsparse_indextype pt, rocsparse_indextype it) {
                    const auto R = h32.rows, C = h32.cols, Z = h32.nnz;
                    const auto B = h32.base;
                    void*      V = d_val.get();
                    if(f == fmt_csr)
                        return rocsparse_create_csr_descr(&out, R, C, Z, p, i, V, pt, it, B, f32);
                    if(f == fmt_csc)
                        return rocsparse_create_csc_descr(&out, R, C, Z, p, i, V, pt, it, B, f32);
                    return rocsparse_create_bsr_descr(
                        &out, R, C, Z, h32.dir, h32.bd, p, i, V, pt, it, B, f32);
                };
            constexpr auto i64 = rocsparse_indextype_i64;
            // (64 bit pointers, 32 bit indices) and (64, 64) are supported; (32, 64) is
            // not documented as supported or unsupported.
            const std::
                array<std::tuple<void*, void*, rocsparse_indextype, rocsparse_indextype, bool>, 3>
                    combos = {{{d_ptr64.get(), d_ind32.get(), i64, i32, true},
                               {d_ptr32.get(), d_ind64.get(), i32, i64, false},
                               {d_ptr64.get(), d_ind64.get(), i64, i64, true}}};
            for(const auto& [p, i, pt, it, required] : combos)
            {
                Spmat descr;
                ASSERT_EQ(create(descr, p, i, pt, it), st_ok);
                auto       ds = bad_ds;
                const auto st = run_check(handle, descr, {}, &ds);
                if(required)
                    EXPECT_EQ(st, st_ok);
                else
                    EXPECT_STATUS_IN(st, st_ok, rocsparse_status_not_implemented);
                if(st == st_ok)
                    EXPECT_EQ(ds, rocsparse_data_status_success);
            }

            if(f != fmt_bsr)
            {
                auto h = h32;
                apply_defect(h, Defect::unsorted);
                Attributes unsorted;
                unsorted.storage = rocsparse_storage_mode_unsorted;
                EXPECT_EQ((check_matrix<float, int32_t>(handle, h, unsorted)),
                          rocsparse_data_status_success);
            }
        }
    }

    // Formats and value types the routine does not support, and invalid arguments.
    TEST_F(ConvSpmat, CheckSpmatUnsupportedAndArguments)
    {
        const auto csr = build<float, int32_t>(fmt_csr, base0, 3, dir_r, make_dense<float>());
        DevMat<float, int32_t> d_csr(csr);
        Spmat                  descr;
        ASSERT_EQ(create_descr(csr, d_csr, descr), st_ok);
        const auto ok_or_ni = [&](Spmat& other) {
            auto ds = rocsparse_data_status_success;
            EXPECT_STATUS_IN(
                run_check(handle, other, {}, &ds), st_ok, rocsparse_status_not_implemented);
        };

        // Value types that are not supported.
        for(auto type : {rocsparse_datatype_i8_r,
                         rocsparse_datatype_u8_r,
                         rocsparse_datatype_i32_r,
                         rocsparse_datatype_u32_r})
        {
            Spmat other;
            ASSERT_EQ(rocsparse_create_csr_descr(&other,
                                                 csr.rows,
                                                 csr.cols,
                                                 csr.nnz,
                                                 d_csr.ptr,
                                                 d_csr.ind,
                                                 d_csr.val,
                                                 i32,
                                                 i32,
                                                 csr.base,
                                                 type),
                      st_ok);
            ok_or_ni(other);
        }
        // Coordinate (AoS) and blocked ELL matrices are not supported.
        const auto aos = build<float, int32_t>(fmt_aos, base0, 3, dir_r, make_dense<float>());
        DevMat<float, int32_t> d_aos(aos);
        DevBuf<int32_t>        ind(M * N);
        DevBuf<float>          val(M * N);
        Spmat                  aos_descr, bell;
        ASSERT_EQ(create_descr(aos, d_aos, aos_descr), st_ok);
        ASSERT_EQ(rocsparse_create_bell_descr(&bell, M, N, dir_r, 2, 2, ind, val, i32, base0, f32),
                  st_ok);
        ok_or_ni(aos_descr);
        ok_or_ni(bell);

        // Invalid arguments.
        auto       ds   = rocsparse_data_status_success;
        size_t     size = 0;
        const auto chk
            = [&](rocsparse_handle            h,
                  rocsparse_const_spmat_descr d,
                  rocsparse_data_status*      s,
                  rocsparse_check_spmat_stage stage,
                  size_t* sz) { return rocsparse_check_spmat(h, d, s, stage, sz, nullptr); };
        EXPECT_EQ(chk(nullptr, descr, &ds, chk_size, &size), rocsparse_status_invalid_handle);
        EXPECT_EQ(chk(handle, nullptr, &ds, chk_size, &size), st_ptr);
        EXPECT_EQ(chk(handle, descr, nullptr, chk_size, &size), st_ptr);
        EXPECT_EQ(chk(handle, descr, &ds, static_cast<rocsparse_check_spmat_stage>(99), &size),
                  st_val);
        EXPECT_EQ(chk(handle, descr, &ds, chk_cmp, nullptr), st_ptr);

        // Batched matrices are not supported; the header does not say which status is
        // returned.
        Spmat batched;
        ASSERT_EQ(create_descr(csr, d_csr, batched), st_ok);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(batched, 2, csr.rows + 1, csr.nnz), st_ok);
        EXPECT_REJECTED(chk(handle, batched, &ds, chk_size, &size));
    }
} // namespace
