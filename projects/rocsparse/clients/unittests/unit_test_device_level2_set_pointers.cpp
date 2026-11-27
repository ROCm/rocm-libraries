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
// Device unit tests of rocsparse_{csr,csc,coo,bsr,ell}_set_pointers followed by
// SpMV. A matrix is analyzed and multiplied, then its pointers are replaced
// (different sparsity pattern, same pattern with new values, and the same
// arrays rewritten in place), analyzed again and multiplied again. The
// products are compared with a host product and with a freshly created
// descriptor over the same arrays, through rocsparse_spmv and through
// rocsparse_v2_spmv. The compute stage that follows set_pointers without a new
// preprocessing is checked as well: it must never use the analysis of the old
// pattern. SpMV keeps no analysis for ELL, so the ELL tests only guard
// set_pointers followed by SpMV.
//
#include "unit_test_device_level2_common.hpp"

#include "rocsparse_mat_info.hpp"
#include "rocsparse_spmat_descr.hpp"

#include <algorithm>
#include <functional>
#include <memory>

using namespace level2;

namespace
{
    enum class fmt
    {
        csr,
        csc,
        coo,
        bsr,
        ell
    };

    const char* fmt_name(fmt f)
    {
        switch(f)
        {
        case fmt::csr:
            return "csr";
        case fmt::csc:
            return "csc";
        case fmt::coo:
            return "coo";
        case fmt::bsr:
            return "bsr";
        case fmt::ell:
            return "ell";
        }
        return "?";
    }

    // One matrix in the arrays that the descriptor of its format takes:
    // csr / bsr: row pointer, column indices; csc: column pointer, row indices;
    // coo: row indices, column indices; ell: no first array, column indices.
    template <typename T, typename I>
    struct problem
    {
        fmt                 format    = fmt::csr;
        int64_t             rows      = 0;
        int64_t             cols      = 0;
        int64_t             nnz       = 0;
        int                 base      = 0;
        int64_t             block_dim = 1;
        rocsparse_direction dir       = rocsparse_direction_row;
        int64_t             ell_width = 0;
        std::vector<I>      first;
        std::vector<I>      second;
        std::vector<T>      val;

        std::function<std::vector<T>(
            rocsparse_operation, T, const std::vector<T>&, T, const std::vector<T>&)>
            gemv;
    };

    template <typename T, typename I>
    problem<T, I> from_csr(fmt f, const host_csr<T>& h)
    {
        problem<T, I> p;
        p.format = f;
        p.rows   = h.m;
        p.cols   = h.n;
        p.nnz    = h.nnz();
        p.base   = h.base;
        switch(f)
        {
        case fmt::csr:
        {
            p.first  = convert_vector<I>(h.ptr);
            p.second = convert_vector<I>(h.ind);
            p.val    = h.val;
            break;
        }
        case fmt::coo:
        {
            const host_coo<T> c = to_coo(h);
            p.first             = convert_vector<I>(c.row);
            p.second            = convert_vector<I>(c.col);
            p.val               = c.val;
            break;
        }
        case fmt::csc:
        {
            const host_csr<T> c = to_csc_arrays(h);
            p.first             = convert_vector<I>(c.ptr);
            p.second            = convert_vector<I>(c.ind);
            p.val               = c.val;
            break;
        }
        case fmt::ell:
        {
            const host_ell<T> e = to_ell(h);
            p.first             = std::vector<I>(1, 0);
            p.second            = convert_vector<I>(e.col);
            p.val               = e.val;
            p.ell_width         = e.width;
            break;
        }
        case fmt::bsr:
            break;
        }
        p.gemv = [h](rocsparse_operation   op,
                     T                     alpha,
                     const std::vector<T>& x,
                     T                     beta,
                     const std::vector<T>& y0) { return csr_gemv(h, op, alpha, x, beta, y0); };
        return p;
    }

    // A BSR matrix whose block row i holds count[i] distinct blocks, starting at
    // block column (5 * i + shift) mod nb.
    template <typename T, typename I>
    problem<T, I> make_bsr_problem(int32_t                     mb,
                                   int32_t                     nb,
                                   int32_t                     bd,
                                   rocsparse_direction         dir,
                                   int                         base,
                                   const std::vector<int32_t>& count,
                                   int32_t                     shift,
                                   int64_t                     seed)
    {
        host_bsr<T> a;
        a.mb  = mb;
        a.nb  = nb;
        a.rbd = bd;
        a.cbd = bd;
        a.dir = dir;
        a.ptr.push_back(0);
        for(int32_t i = 0; i < mb; ++i)
        {
            std::vector<int32_t> cols;
            const int32_t        stride = std::max<int32_t>(1, nb / std::max<int32_t>(count[i], 1));
            for(int32_t k = 0; k < count[i]; ++k)
                cols.push_back((((5 * i + shift) % nb) + k * stride) % nb);
            std::sort(cols.begin(), cols.end());
            for(const int32_t c : cols)
            {
                a.ind.push_back(c);
                for(int32_t e = 0; e < bd * bd; ++e)
                    a.val.push_back(value_of<T>(seed++));
            }
            a.ptr.push_back(static_cast<int32_t>(a.ind.size()));
        }
        for(auto& q : a.ptr)
            q += base;
        for(auto& c : a.ind)
            c += base;
        a.base = base;

        problem<T, I> p;
        p.format    = fmt::bsr;
        p.rows      = static_cast<int64_t>(mb) * bd;
        p.cols      = static_cast<int64_t>(nb) * bd;
        p.nnz       = a.nnzb();
        p.base      = base;
        p.block_dim = bd;
        p.dir       = dir;
        p.first     = convert_vector<I>(a.ptr);
        p.second    = convert_vector<I>(a.ind);
        p.val       = a.val;
        p.gemv      = [a, mb](rocsparse_operation,
                         T                     alpha,
                         const std::vector<T>& x,
                         T                     beta,
                         const std::vector<T>& y0) {
            return bsr_gemv(a, all_rows(mb), {}, alpha, x, beta, y0);
        };
        return p;
    }

    template <typename T, typename I>
    struct device_problem
    {
        device_vector<I> first;
        device_vector<I> second;
        device_vector<T> val;

        explicit device_problem(const problem<T, I>& p)
            : first(p.first)
            , second(p.second)
            , val(p.val)
        {
        }

        // Rewrites the arrays in place; the sizes of both problems are equal.
        void overwrite(const problem<T, I>& p)
        {
            ASSERT_EQ(p.first.size(), first.n);
            ASSERT_EQ(p.second.size(), second.n);
            ASSERT_EQ(p.val.size(), val.n);
            UT_CHECK_HIP(hipMemcpy(
                first.ptr, p.first.data(), p.first.size() * sizeof(I), hipMemcpyHostToDevice));
            UT_CHECK_HIP(hipMemcpy(
                second.ptr, p.second.data(), p.second.size() * sizeof(I), hipMemcpyHostToDevice));
            UT_CHECK_HIP(
                hipMemcpy(val.ptr, p.val.data(), p.val.size() * sizeof(T), hipMemcpyHostToDevice));
        }
    };

    template <typename T, typename I>
    void create(spmat_guard& g, const problem<T, I>& p, device_problem<T, I>& d)
    {
        const auto       it = rocsparse_ut::it_of<I>();
        const auto       dt = rocsparse_ut::dt_of<T>();
        const auto       ib = to_base(p.base);
        rocsparse_status st = rocsparse_status_internal_error;
        switch(p.format)
        {
        case fmt::csr:
            st = rocsparse_create_csr_descr(
                &g.d, p.rows, p.cols, p.nnz, d.first.ptr, d.second.ptr, d.val.ptr, it, it, ib, dt);
            break;
        case fmt::csc:
            st = rocsparse_create_csc_descr(
                &g.d, p.rows, p.cols, p.nnz, d.first.ptr, d.second.ptr, d.val.ptr, it, it, ib, dt);
            break;
        case fmt::coo:
            st = rocsparse_create_coo_descr(
                &g.d, p.rows, p.cols, p.nnz, d.first.ptr, d.second.ptr, d.val.ptr, it, ib, dt);
            break;
        case fmt::bsr:
            st = rocsparse_create_bsr_descr(&g.d,
                                            p.rows / p.block_dim,
                                            p.cols / p.block_dim,
                                            p.nnz,
                                            p.dir,
                                            p.block_dim,
                                            d.first.ptr,
                                            d.second.ptr,
                                            d.val.ptr,
                                            it,
                                            it,
                                            ib,
                                            dt);
            break;
        case fmt::ell:
            st = rocsparse_create_ell_descr(
                &g.d, p.rows, p.cols, d.second.ptr, d.val.ptr, p.ell_width, it, ib, dt);
            break;
        }
        ASSERT_EQ(st, rocsparse_status_success);
    }

    template <typename T, typename I>
    rocsparse_status set_arrays(spmat_guard& g, fmt f, I* first, I* second, T* val)
    {
        switch(f)
        {
        case fmt::csr:
            return rocsparse_csr_set_pointers(g.d, first, second, val);
        case fmt::csc:
            return rocsparse_csc_set_pointers(g.d, first, second, val);
        case fmt::coo:
            return rocsparse_coo_set_pointers(g.d, first, second, val);
        case fmt::bsr:
            return rocsparse_bsr_set_pointers(g.d, first, second, val);
        case fmt::ell:
            return rocsparse_ell_set_pointers(g.d, second, val);
        }
        return rocsparse_status_internal_error;
    }

    template <typename T, typename I>
    rocsparse_status set_pointers(spmat_guard& g, const problem<T, I>& p, device_problem<T, I>& d)
    {
        return set_arrays(g, p.format, d.first.ptr, d.second.ptr, d.val.ptr);
    }

    struct config
    {
        rocsparse_spmv_alg     alg     = rocsparse_spmv_alg_default;
        rocsparse_operation    trans   = rocsparse_operation_none;
        bool                   device  = false;
        bool                   v2      = false;
        rocsparse_storage_mode storage = rocsparse_storage_mode_sorted;
    };

    void set_storage(spmat_guard& g, rocsparse_storage_mode storage)
    {
        ASSERT_EQ(rocsparse_spmat_set_attribute(
                      g.d, rocsparse_spmat_storage_mode, &storage, sizeof(storage)),
                  rocsparse_status_success);
    }

    // Entries of the y allocation before and after the vector, which no stage may write.
    constexpr size_t y_guard = 16;

    template <typename T>
    T guard_value()
    {
        return case_scalar<T>(-1234.5f, 0.0f);
    }

    template <typename T>
    std::vector<T> with_guards(const std::vector<T>& y)
    {
        std::vector<T> g(y.size() + 2 * y_guard, guard_value<T>());
        std::copy(y.begin(), y.end(), g.begin() + y_guard);
        return g;
    }

    // x, y and the SpMV stages of one matrix descriptor, through the v1 or the v2 interface.
    template <typename T>
    class session
    {
        rocsparse_handle                     m_handle;
        config                               m_cfg;
        rocsparse_spmat_descr                m_mat;
        T                                    m_alpha;
        T                                    m_beta;
        std::vector<T>                       m_x;
        std::vector<T>                       m_y0;
        device_vector<T>                     m_dx, m_dy, m_dalpha, m_dbeta;
        dnvec_guard                          m_vx, m_vy;
        std::unique_ptr<spmv_descr_guard>    m_descr;
        bool                                 m_analyzed = false;
        size_t                               m_bytes    = 0;
        std::unique_ptr<device_vector<char>> m_buffer;

        const void* alpha() const
        {
            return m_cfg.device ? static_cast<const void*>(m_dalpha.ptr)
                                : static_cast<const void*>(&m_alpha);
        }
        const void* beta() const
        {
            return m_cfg.device ? static_cast<const void*>(m_dbeta.ptr)
                                : static_cast<const void*>(&m_beta);
        }

        // A v2 descriptor takes the analysis stage once, so a matrix whose pointers were
        // replaced is analyzed with a new one.
        void new_v2_descr()
        {
            m_descr.reset(new spmv_descr_guard());
            EXPECT_TRUE(m_descr->d != nullptr);
            EXPECT_EQ(set_basic_inputs(m_handle,
                                       m_descr->d,
                                       m_cfg.alg,
                                       m_cfg.trans,
                                       rocsparse_ut::dt_of<T>(),
                                       rocsparse_ut::dt_of<T>()),
                      rocsparse_status_success);
            m_analyzed = false;
        }

        rocsparse_status v1(rocsparse_spmv_stage stage, void* buffer)
        {
            return rocsparse_spmv(m_handle,
                                  m_cfg.trans,
                                  alpha(),
                                  m_mat,
                                  m_vx.d,
                                  beta(),
                                  m_vy.d,
                                  rocsparse_ut::dt_of<T>(),
                                  m_cfg.alg,
                                  stage,
                                  &m_bytes,
                                  buffer);
        }

        rocsparse_status v2_buffer_size(rocsparse_v2_spmv_stage stage, size_t* bytes)
        {
            return rocsparse_v2_spmv_buffer_size(
                m_handle, m_descr->d, m_mat, m_vx.d, m_vy.d, stage, bytes, nullptr);
        }

        rocsparse_status v2(rocsparse_v2_spmv_stage stage, size_t bytes, void* buffer)
        {
            return rocsparse_v2_spmv(m_handle,
                                     m_descr->d,
                                     alpha(),
                                     m_mat,
                                     m_vx.d,
                                     beta(),
                                     m_vy.d,
                                     stage,
                                     bytes,
                                     buffer,
                                     nullptr);
        }

    public:
        session(rocsparse_handle      handle,
                const config&         c,
                rocsparse_spmat_descr mat,
                int64_t               m,
                int64_t               n)
            : m_handle(handle)
            , m_cfg(c)
            , m_mat(mat)
            , m_alpha(case_scalar<T>(1.5f, 0.0f))
            , m_beta(case_scalar<T>(-0.5f, 1.0f))
            , m_x(make_vector<T>(trans_size(c.trans, m, n, true), 2))
            , m_y0(make_vector<T>(trans_size(c.trans, m, n, false), 9))
            , m_dx(m_x)
            , m_dy(with_guards(m_y0))
            , m_dalpha(std::vector<T>(1, m_alpha))
            , m_dbeta(std::vector<T>(1, m_beta))
        {
            EXPECT_EQ(rocsparse_create_dnvec_descr(
                          &m_vx.d, m_x.size(), m_dx.ptr, rocsparse_ut::dt_of<T>()),
                      rocsparse_status_success);
            EXPECT_EQ(rocsparse_create_dnvec_descr(
                          &m_vy.d, m_y0.size(), m_dy.ptr + y_guard, rocsparse_ut::dt_of<T>()),
                      rocsparse_status_success);
            if(m_cfg.v2)
            {
                new_v2_descr();
            }
        }

        const std::vector<T>& y0() const
        {
            return m_y0;
        }

        // Runs the stages on another matrix descriptor with the same sizes.
        void set_matrix(rocsparse_spmat_descr mat)
        {
            m_mat = mat;
        }

        // Analysis stage of the current v2 descriptor. preprocess() replaces that
        // descriptor once it has been analyzed, so this is the same-descriptor path.
        rocsparse_status analyze_same_descr()
        {
            size_t                 bytes = 0;
            const rocsparse_status sized = v2_buffer_size(rocsparse_v2_spmv_stage_analysis, &bytes);
            if(sized != rocsparse_status_success)
            {
                return sized;
            }
            device_vector<char> buffer(std::max<size_t>(bytes, 1));
            return v2(rocsparse_v2_spmv_stage_analysis, bytes, bytes ? buffer.ptr : nullptr);
        }

        // The v1 preprocess stage alone, with the buffer of the last preprocess().
        rocsparse_status preprocess_stage()
        {
            return v1(rocsparse_spmv_stage_preprocess, m_buffer ? m_buffer->ptr : nullptr);
        }

        // Replaces the v2 descriptor and queries the size of its analysis buffer, so
        // that analysis_stage() can run without allocating, for instance during a capture.
        rocsparse_status new_descr_analysis_buffer_size(size_t* bytes)
        {
            new_v2_descr();
            m_analyzed = true;
            return v2_buffer_size(rocsparse_v2_spmv_stage_analysis, bytes);
        }

        // The v2 analysis stage alone.
        rocsparse_status analysis_stage(size_t bytes, void* buffer)
        {
            return v2(rocsparse_v2_spmv_stage_analysis, bytes, buffer);
        }

        // Destroys the rocsparse_spmv_descr (v2).
        rocsparse_status destroy_v2_descr()
        {
            const rocsparse_status st = rocsparse_destroy_spmv_descr(m_descr->d);
            m_descr->d                = nullptr;
            return st;
        }

        // Replaces the scalars and the vectors; the sizes are those of the constructor.
        void set_data(T alpha, T beta, const std::vector<T>& x, const std::vector<T>& y0)
        {
            ASSERT_EQ(x.size(), m_x.size());
            ASSERT_EQ(y0.size(), m_y0.size());
            m_alpha = alpha;
            m_beta  = beta;
            m_x     = x;
            m_y0    = y0;
            ASSERT_EQ(hipMemcpy(m_dx.ptr, x.data(), x.size() * sizeof(T), hipMemcpyHostToDevice),
                      hipSuccess);
            ASSERT_EQ(hipMemcpy(m_dalpha.ptr, &alpha, sizeof(T), hipMemcpyHostToDevice),
                      hipSuccess);
            ASSERT_EQ(hipMemcpy(m_dbeta.ptr, &beta, sizeof(T), hipMemcpyHostToDevice), hipSuccess);
        }

        template <typename P>
        std::vector<T> want(const P& p) const
        {
            return p.gemv(m_cfg.trans, m_alpha, m_x, m_beta, m_y0);
        }

        // y; the entries around it must still hold the guard value.
        std::vector<T> y() const
        {
            const std::vector<T> all = to_host(m_dy);
            size_t               bad = 0;
            for(size_t k = 0; k < y_guard; ++k)
            {
                bad += !(all[k] == guard_value<T>());
                bad += !(all[all.size() - 1 - k] == guard_value<T>());
            }
            EXPECT_EQ(bad, 0u) << "entries around y were written";
            return std::vector<T>(all.begin() + y_guard, all.end() - y_guard);
        }

        // Buffer size query and preprocessing (v1) or analysis (v2) of the current pointers.
        void preprocess()
        {
            if(m_cfg.v2)
            {
                if(m_analyzed)
                {
                    new_v2_descr();
                }
                m_analyzed   = true;
                size_t bytes = 0;
                ASSERT_EQ(v2_buffer_size(rocsparse_v2_spmv_stage_analysis, &bytes),
                          rocsparse_status_success);
                device_vector<char> buffer(std::max<size_t>(bytes, 1));
                ASSERT_EQ(v2(rocsparse_v2_spmv_stage_analysis, bytes, bytes ? buffer.ptr : nullptr),
                          rocsparse_status_success);
                return;
            }
            m_bytes = 0;
            ASSERT_EQ(v1(rocsparse_spmv_stage_buffer_size, nullptr), rocsparse_status_success);
            m_buffer.reset(new device_vector<char>(std::max<size_t>(m_bytes, 1)));
            ASSERT_EQ(v1(rocsparse_spmv_stage_preprocess, m_buffer->ptr), rocsparse_status_success);
        }

        // Compute stage on the initial y.
        rocsparse_status compute()
        {
            const hipError_t copy = hipMemcpy(
                m_dy.ptr + y_guard, m_y0.data(), m_y0.size() * sizeof(T), hipMemcpyHostToDevice);
            EXPECT_EQ(copy, hipSuccess);
            if(m_cfg.v2)
            {
                size_t                 bytes = 0;
                const rocsparse_status st = v2_buffer_size(rocsparse_v2_spmv_stage_compute, &bytes);
                if(st != rocsparse_status_success)
                    return st;
                device_vector<char> buffer(std::max<size_t>(bytes, 1));
                return v2(rocsparse_v2_spmv_stage_compute, bytes, bytes ? buffer.ptr : nullptr);
            }
            return v1(rocsparse_spmv_stage_compute, m_buffer ? m_buffer->ptr : nullptr);
        }
    };

    template <typename T, typename I>
    struct scenario
    {
        problem<T, I> p1; // the matrix the descriptor is created with
        problem<T, I> p2; // another pattern with the same dimensions and number of entries
        problem<T, I> p3; // the pattern of p1 with other values
    };

    // Rows of p1 are empty at the top and long at the bottom; p2 is p1 upside down.
    inline int32_t row_length(int32_t i, int32_t m)
    {
        if(i < 2)
            return 0;
        if(i >= m - 4)
            return 250;
        if(i % 7 == 0)
            return 0;
        return (i % 97 == 5) ? 600 : 1 + (i % 23);
    }

    // Row lengths of p1 and of p2 for COO: p2 has the same number of entries, but
    // a row longer than 10 * 256, above which the atomic COO kernel processes two
    // entries per thread; the longest row of p1 has 600 entries.
    constexpr int32_t coo_long_row = 2800;

    inline std::vector<int32_t> coo_row_lengths(int32_t m, bool second)
    {
        std::vector<int32_t> len(m);
        for(int32_t i = 0; i < m; ++i)
            len[i] = row_length(second ? m - 1 - i : i, m);
        if(!second)
            return len;
        const int32_t r     = m / 2;
        int32_t       extra = coo_long_row - len[r];
        len[r]              = coo_long_row;
        for(int32_t i = 0; extra > 0; i = (i + 1) % m)
        {
            if(i != r && len[i] > 0)
            {
                --len[i];
                --extra;
            }
        }
        return len;
    }

    // block_dim only applies to BSR.
    template <typename T, typename I>
    scenario<T, I> make_scenario(fmt f, int base, int32_t block_dim = 1)
    {
        scenario<T, I> s;
        if(f == fmt::bsr)
        {
            const int32_t mb = 96;
            const int32_t nb = 80;
            const int32_t bd = block_dim;

            std::vector<int32_t> count1(mb), count2(mb);
            for(int32_t i = 0; i < mb; ++i)
                count1[i] = (i == 0 || i % 9 == 4) ? 0 : 1 + (i % 5);
            for(int32_t i = 0; i < mb; ++i)
                count2[i] = count1[mb - 1 - i];

            s.p1 = make_bsr_problem<T, I>(mb, nb, bd, rocsparse_direction_row, base, count1, 0, 0);
            s.p2 = make_bsr_problem<T, I>(mb, nb, bd, rocsparse_direction_row, base, count2, 3, 11);
            s.p3 = make_bsr_problem<T, I>(
                mb, nb, bd, rocsparse_direction_row, base, count1, 0, 4001);
            return s;
        }

        const int32_t m = 1500;
        const int32_t n = (f == fmt::coo) ? 3000 : 1300;
        host_csr<T>   h1, h2;
        if(f == fmt::coo)
        {
            const std::vector<int32_t> len1 = coo_row_lengths(m, false);
            const std::vector<int32_t> len2 = coo_row_lengths(m, true);
            h1 = make_csr_by_length<T>(m, n, base, [&](int32_t i) { return len1[i]; });
            h2 = make_csr_by_length<T>(m, n, base, [&](int32_t i) { return len2[i]; });
        }
        else
        {
            h1 = make_csr_by_length<T>(m, n, base, [&](int32_t i) { return row_length(i, m); });
            h2 = make_csr_by_length<T>(
                m, n, base, [&](int32_t i) { return row_length(m - 1 - i, m); });
        }
        host_csr<T> h3 = h1;
        for(size_t k = 0; k < h3.val.size(); ++k)
            h3.val[k] = value_of<T>(static_cast<int64_t>(k) * 3 + 1001, true);

        s.p1 = from_csr<T, I>(f, h1);
        s.p2 = from_csr<T, I>(f, h2);
        s.p3 = from_csr<T, I>(f, h3);
        return s;
    }

    // Whether the CSR kernels run non-transposed: CSC runs them on the flipped operation.
    inline bool runs_csr_none(fmt f, rocsparse_operation trans)
    {
        return (f == fmt::csr) == (trans == rocsparse_operation_none);
    }

    // Whether the rocsparse_spmv_descr holds an analysis of the matrix arrays, so that
    // its compute stage after set_pointers returns rocsparse_status_invalid_value, as
    // listed in the rocsparse_v2_spmv documentation. The COO analysis is kept with the
    // matrix descriptor, not with the SpMV descriptor.
    inline bool v2_refuses(fmt f, const config& c, int32_t block_dim)
    {
        switch(f)
        {
        case fmt::csr:
        case fmt::csc:
            // The default resolves to rowsplit, which keeps no analysis, unless the CSR
            // kernels run non-transposed.
            return c.alg != rocsparse_spmv_alg_csr_rowsplit
                   && (c.alg != rocsparse_spmv_alg_default || runs_csr_none(f, c.trans));
        case fmt::bsr:
            return block_dim == 1 && c.storage == rocsparse_storage_mode_sorted;
        case fmt::coo:
        case fmt::ell:
            return false;
        }
        return false;
    }

    // Returns the number of v2 compute stages refused after set_pointers.
    template <typename T, typename I>
    int run_case(rocsparse_handle handle, fmt f, const config& c, int base, int32_t block_dim)
    {
        SCOPED_TRACE(::testing::Message()
                     << fmt_name(f) << " alg=" << c.alg << " trans=" << c.trans
                     << " device=" << c.device << " v2=" << c.v2 << " base=" << base << " i64="
                     << (sizeof(I) == 8) << " block_dim=" << block_dim << " storage=" << c.storage);

        int                  refused = 0;
        const bool           refuses = c.v2 && v2_refuses(f, c, block_dim);
        const scenario<T, I> sc      = make_scenario<T, I>(f, base, block_dim);
        device_problem<T, I> d1(sc.p1), d2(sc.p2), d3(sc.p3);

        spmat_guard mat;
        create(mat, sc.p1, d1);
        if(::testing::Test::HasFatalFailure())
            return refused;
        set_storage(mat, c.storage);
        if(::testing::Test::HasFatalFailure())
            return refused;
        session<T> s(handle, c, mat.d, sc.p1.rows, sc.p1.cols);

        auto product = [&](const problem<T, I>& p, const char* what) {
            ASSERT_EQ(s.compute(), rocsparse_status_success) << what;
            EXPECT_TRUE(expect_near(s.want(p), s.y(), what));
        };

        // The product of a freshly created descriptor over the same arrays.
        auto same_as_fresh
            = [&](const problem<T, I>& p, device_problem<T, I>& d, const char* what) {
                  spmat_guard fm;
                  create(fm, p, d);
                  if(::testing::Test::HasFatalFailure())
                      return;
                  set_storage(fm, c.storage);
                  if(::testing::Test::HasFatalFailure())
                      return;
                  session<T> fs(handle, c, fm.d, p.rows, p.cols);
                  fs.preprocess();
                  if(::testing::Test::HasFatalFailure())
                      return;
                  ASSERT_EQ(fs.compute(), rocsparse_status_success) << what;
                  EXPECT_TRUE(expect_near(fs.y(), s.y(), what));
              };

        // set_pointers, a compute stage without a new preprocessing, then the
        // preprocessing and the compute stage.
        auto replace = [&](const problem<T, I>& p, device_problem<T, I>& d, const char* what) {
            SCOPED_TRACE(what);
            ASSERT_EQ(set_pointers(mat, p, d), rocsparse_status_success);

            // The analysis of the previous pointers must not be used. v1 computes the
            // product without it; v2 refuses when its descriptor holds that analysis.
            const rocsparse_status st  = s.compute();
            const auto             got = s.y();
            if(!refuses)
            {
                EXPECT_EQ(st, rocsparse_status_success) << "compute without preprocessing";
                EXPECT_TRUE(expect_near(s.want(p), got, "compute without preprocessing"));
            }
            else
            {
                EXPECT_EQ(st, rocsparse_status_invalid_value) << "compute without analysis";
                EXPECT_TRUE(expect_near(s.y0(), got, "y after the refused compute"));
                refused += (st == rocsparse_status_invalid_value);
            }
            if(::testing::Test::HasFailure())
                return;

            s.preprocess();
            if(::testing::Test::HasFatalFailure())
                return;
            product(p, "product after the new preprocessing");
            if(::testing::Test::HasFailure())
                return;
            same_as_fresh(p, d, "product of a fresh descriptor");
        };

        s.preprocess();
        if(::testing::Test::HasFatalFailure())
            return refused;
        product(sc.p1, "initial product");
        if(::testing::Test::HasFailure())
            return refused;

        // Another sparsity pattern.
        replace(sc.p2, d2, "another pattern");
        if(::testing::Test::HasFailure())
            return refused;

        // The pattern of the first matrix with other values.
        replace(sc.p3, d3, "same pattern, other values");
        if(::testing::Test::HasFailure())
            return refused;

        // The same arrays rewritten with another pattern, then set again.
        replace(sc.p2, d2, "arrays of p2");
        if(::testing::Test::HasFailure())
            return refused;
        d2.overwrite(sc.p1);
        if(::testing::Test::HasFatalFailure())
            return refused;
        replace(sc.p1, d2, "same arrays rewritten in place");
        if(::testing::Test::HasFailure())
            return refused;

        // Back to the arrays of the first matrix.
        replace(sc.p1, d1, "back to the first arrays");
        return refused;
    }

    struct alg_trans
    {
        rocsparse_spmv_alg  alg;
        rocsparse_operation trans;
    };

    // Every CSR and CSC algorithm with every operation it computes: adaptive and LRB
    // compute only when the CSR kernels run non-transposed.
    template <typename T>
    std::vector<alg_trans> cases_of(fmt f)
    {
        constexpr auto none   = rocsparse_operation_none;
        constexpr auto tr     = rocsparse_operation_transpose;
        const bool is_complex = !std::is_same<T, float>::value && !std::is_same<T, double>::value;
        std::vector<alg_trans> v;
        switch(f)
        {
        case fmt::csr:
        case fmt::csc:
        {
            std::vector<rocsparse_operation> ops = {none, tr};
            if(is_complex)
                ops.push_back(rocsparse_operation_conjugate_transpose);
            for(const auto trans : ops)
                for(const auto alg : {rocsparse_spmv_alg_default,
                                      rocsparse_spmv_alg_csr_adaptive,
                                      rocsparse_spmv_alg_csr_rowsplit,
                                      rocsparse_spmv_alg_csr_lrb,
                                      rocsparse_spmv_alg_csr_nnzsplit})
                {
                    const bool needs_csr_none = alg == rocsparse_spmv_alg_csr_adaptive
                                                || alg == rocsparse_spmv_alg_csr_lrb;
                    if(!needs_csr_none || runs_csr_none(f, trans))
                        v.push_back({alg, trans});
                }
            break;
        }
        case fmt::coo:
        {
            for(const auto alg :
                {rocsparse_spmv_alg_default, rocsparse_spmv_alg_coo, rocsparse_spmv_alg_coo_atomic})
                v.push_back({alg, none});
            v.push_back({rocsparse_spmv_alg_default, tr});
            break;
        }
        case fmt::bsr:
        {
            v.push_back({rocsparse_spmv_alg_default, none});
            v.push_back({rocsparse_spmv_alg_bsr, none});
            break;
        }
        case fmt::ell:
        {
            v.push_back({rocsparse_spmv_alg_default, none});
            v.push_back({rocsparse_spmv_alg_ell, none});
            break;
        }
        }
        return v;
    }

    struct layout
    {
        int32_t                block_dim;
        rocsparse_storage_mode storage;
    };

    // SpMV keeps a BSR analysis only for block_dim == 1 in sorted storage.
    inline std::vector<layout> layouts_of(fmt f)
    {
        constexpr auto sorted = rocsparse_storage_mode_sorted;
        if(f == fmt::bsr)
            return {{1, sorted}, {4, sorted}, {1, rocsparse_storage_mode_unsorted}};
        return {{1, sorted}};
    }

    template <typename T, typename I>
    int sweep_index(rocsparse_handle handle, fmt f, bool v2)
    {
        int refused = 0;
        for(const layout& l : layouts_of(f))
            for(const int base : {0, 1})
                for(const bool device : {false, true})
                    for(const auto& at : cases_of<T>(f))
                    {
                        scoped_pointer_mode mode(handle, device);
                        config              c;
                        c.alg     = at.alg;
                        c.trans   = at.trans;
                        c.device  = device;
                        c.v2      = v2;
                        c.storage = l.storage;
                        refused += run_case<T, I>(handle, f, c, base, l.block_dim);
                        if(::testing::Test::HasFailure())
                            return refused;
                    }
        return refused;
    }

    template <typename T>
    void sweep(rocsparse_handle handle, fmt f, bool v2)
    {
        int refused = sweep_index<T, int32_t>(handle, f, v2);
        if(!::testing::Test::HasFailure())
            refused += sweep_index<T, int64_t>(handle, f, v2);
        if(v2 && (f == fmt::csr || f == fmt::csc || f == fmt::bsr)
           && !::testing::Test::HasFailure())
        {
            EXPECT_GT(refused, 0) << "no v2 compute stage was refused after set_pointers";
        }
    }

    class L2DeviceSpmvSetPointers : public rocsparse_ut::HandleTest
    {
    };

#define SET_POINTERS_TEST(NAME, T, V2, FMT) \
    TEST_F(L2DeviceSpmvSetPointers, NAME)   \
    {                                       \
        sweep<T>(handle, FMT, V2);          \
    }

#define SET_POINTERS_TESTS(NAME, FMT)                                                 \
    SET_POINTERS_TEST(NAME##_float_v1, float, false, FMT)                             \
    SET_POINTERS_TEST(NAME##_double_v1, double, false, FMT)                           \
    SET_POINTERS_TEST(NAME##_float_complex_v1, rocsparse_float_complex, false, FMT)   \
    SET_POINTERS_TEST(NAME##_double_complex_v1, rocsparse_double_complex, false, FMT) \
    SET_POINTERS_TEST(NAME##_float_v2, float, true, FMT)                              \
    SET_POINTERS_TEST(NAME##_double_v2, double, true, FMT)                            \
    SET_POINTERS_TEST(NAME##_float_complex_v2, rocsparse_float_complex, true, FMT)    \
    SET_POINTERS_TEST(NAME##_double_complex_v2, rocsparse_double_complex, true, FMT)

    SET_POINTERS_TESTS(csr, fmt::csr)
    SET_POINTERS_TESTS(csc, fmt::csc)
    SET_POINTERS_TESTS(coo, fmt::coo)
    SET_POINTERS_TESTS(bsr, fmt::bsr)
    SET_POINTERS_TESTS(ell, fmt::ell)
#undef SET_POINTERS_TESTS
#undef SET_POINTERS_TEST

    // The reproducer of the report: a 3 x 2 CSR matrix whose offsets and columns are
    // replaced after it was analyzed.
    TEST_F(L2DeviceSpmvSetPointers, replaced_csr_arrays_small_matrix)
    {
        const std::vector<int32_t> old_row = {0, 2, 3, 3};
        const std::vector<int32_t> old_col = {0, 1, 1};
        const std::vector<int32_t> new_row = {0, 1, 3, 3};
        const std::vector<int32_t> new_col = {1, 0, 1};
        const std::vector<float>   val     = {2, 3, 4};

        // nnz-split is left out until #15869 is fixed: on this matrix it computes
        // y[0] = 127 instead of 65 before set_pointers.
        for(const bool v2 : {false, true})
            for(const auto alg : {rocsparse_spmv_alg_default,
                                  rocsparse_spmv_alg_csr_adaptive,
                                  rocsparse_spmv_alg_csr_rowsplit,
                                  rocsparse_spmv_alg_csr_lrb})
            {
                SCOPED_TRACE(::testing::Message() << "v2=" << v2 << " alg=" << alg);

                device_vector<int32_t> drow0(old_row), dcol0(old_col), drow1(new_row),
                    dcol1(new_col);
                device_vector<float> dval(val);

                spmat_guard mat;
                ASSERT_EQ(rocsparse_create_csr_descr(&mat.d,
                                                     3,
                                                     2,
                                                     3,
                                                     drow0.ptr,
                                                     dcol0.ptr,
                                                     dval.ptr,
                                                     rocsparse_indextype_i32,
                                                     rocsparse_indextype_i32,
                                                     rocsparse_index_base_zero,
                                                     rocsparse_datatype_f32_r),
                          rocsparse_status_success);

                config c;
                c.alg = alg;
                c.v2  = v2;
                session<float> s(handle, c, mat.d, 3, 2);
                s.set_data(2.0f, 3.0f, {5, 7}, {1, 2, 3});
                ASSERT_FALSE(HasFatalFailure());
                s.preprocess();
                ASSERT_FALSE(HasFatalFailure());
                ASSERT_EQ(s.compute(), rocsparse_status_success);
                EXPECT_TRUE(expect_near(std::vector<float>{65, 62, 9}, s.y(), "original matrix"));

                ASSERT_EQ(rocsparse_csr_set_pointers(mat.d, drow1.ptr, dcol1.ptr, dval.ptr),
                          rocsparse_status_success);
                s.preprocess();
                ASSERT_FALSE(HasFatalFailure());
                ASSERT_EQ(s.compute(), rocsparse_status_success);
                EXPECT_TRUE(expect_near(std::vector<float>{31, 92, 9}, s.y(), "replaced pointers"));
            }
    }

    // The same pointers are set again after the arrays were rewritten with another
    // pattern, which an analysis recognised by its pointers cannot detect.
    template <typename T>
    void same_pointers_new_contents(rocsparse_handle handle, fmt f, const config& c)
    {
        SCOPED_TRACE(::testing::Message()
                     << fmt_name(f) << " alg=" << c.alg << " trans=" << c.trans << " v2=" << c.v2);

        const scenario<T, int32_t> sc = make_scenario<T, int32_t>(f, 0);
        device_problem<T, int32_t> d(sc.p2);

        spmat_guard mat;
        create(mat, sc.p2, d);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        session<T> s(handle, c, mat.d, sc.p2.rows, sc.p2.cols);
        s.preprocess();
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        ASSERT_EQ(s.compute(), rocsparse_status_success);
        EXPECT_TRUE(expect_near(s.want(sc.p2), s.y(), "product of the first contents"));

        d.overwrite(sc.p1);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        ASSERT_EQ(set_pointers(mat, sc.p1, d), rocsparse_status_success);

        const rocsparse_status st = s.compute();
        if(c.v2)
        {
            EXPECT_EQ(st, rocsparse_status_invalid_value) << "compute without analysis";
            EXPECT_TRUE(expect_near(s.y0(), s.y(), "y after the refused compute"));
        }
        else
        {
            EXPECT_EQ(st, rocsparse_status_success) << "compute without preprocessing";
            EXPECT_TRUE(expect_near(s.want(sc.p1), s.y(), "compute without preprocessing"));
        }

        s.preprocess();
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        ASSERT_EQ(s.compute(), rocsparse_status_success);
        EXPECT_TRUE(expect_near(s.want(sc.p1), s.y(), "product after the new preprocessing"));
    }

    // Formats and algorithms for which both v1 and v2 keep an analysis of the arrays.
    const std::vector<std::pair<fmt, alg_trans>>& cases_with_analysis()
    {
        static const std::vector<std::pair<fmt, alg_trans>> cases
            = {{fmt::csr, {rocsparse_spmv_alg_default, rocsparse_operation_none}},
               {fmt::csr, {rocsparse_spmv_alg_csr_adaptive, rocsparse_operation_none}},
               {fmt::csr, {rocsparse_spmv_alg_csr_lrb, rocsparse_operation_none}},
               {fmt::csc, {rocsparse_spmv_alg_csr_adaptive, rocsparse_operation_transpose}},
               {fmt::csc, {rocsparse_spmv_alg_csr_lrb, rocsparse_operation_transpose}},
               {fmt::bsr, {rocsparse_spmv_alg_default, rocsparse_operation_none}}};
        return cases;
    }

    TEST_F(L2DeviceSpmvSetPointers, same_pointers_new_contents)
    {
        for(const bool v2 : {false, true})
            for(const auto& fc : cases_with_analysis())
            {
                config c;
                c.alg   = fc.second.alg;
                c.trans = fc.second.trans;
                c.v2    = v2;
                same_pointers_new_contents<double>(handle, fc.first, c);
            }
    }

    // The COO analysis stores the longest row, which selects the atomic COO kernel;
    // it has to be the longest row of the new arrays.
    TEST_F(L2DeviceSpmvSetPointers, coo_analysis_of_the_new_arrays)
    {
        using T = double;
        using I = int32_t;
        for(const bool v2 : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << "v2=" << v2);
            const scenario<T, I> sc = make_scenario<T, I>(fmt::coo, 0);
            device_problem<T, I> d1(sc.p1), d2(sc.p2);

            spmat_guard mat;
            create(mat, sc.p1, d1);
            ASSERT_FALSE(HasFatalFailure());
            config c;
            c.alg = rocsparse_spmv_alg_coo_atomic;
            c.v2  = v2;
            session<T> s(handle, c, mat.d, sc.p1.rows, sc.p1.cols);

            auto longest_row = [&]() -> int64_t {
                const rocsparse_coomv_info info = mat.d->info->get_coomv_info();
                return info != nullptr ? info->max_nnz_per_row : -1;
            };

            s.preprocess();
            ASSERT_FALSE(HasFatalFailure());
            EXPECT_EQ(longest_row(), 600);
            ASSERT_EQ(s.compute(), rocsparse_status_success);
            EXPECT_TRUE(expect_near(s.want(sc.p1), s.y(), "initial product"));

            ASSERT_EQ(set_pointers(mat, sc.p2, d2), rocsparse_status_success);
            ASSERT_EQ(s.compute(), rocsparse_status_success);
            EXPECT_TRUE(expect_near(s.want(sc.p2), s.y(), "compute without preprocessing"));

            s.preprocess();
            ASSERT_FALSE(HasFatalFailure());
            EXPECT_EQ(longest_row(), coo_long_row);
            ASSERT_EQ(s.compute(), rocsparse_status_success);
            EXPECT_TRUE(expect_near(s.want(sc.p2), s.y(), "product after the new preprocessing"));
        }
    }

    // set_pointers frees nothing and makes no device call, so it can run while
    // another stream is being captured.
    TEST_F(L2DeviceSpmvSetPointers, set_pointers_during_stream_capture)
    {
        using T                 = double;
        using I                 = int32_t;
        const scenario<T, I> sc = make_scenario<T, I>(fmt::csr, 0);
        device_problem<T, I> d1(sc.p1), d2(sc.p2);

        spmat_guard mat;
        create(mat, sc.p1, d1);
        ASSERT_FALSE(HasFatalFailure());
        config c;
        c.alg = rocsparse_spmv_alg_csr_adaptive;
        session<T> s(handle, c, mat.d, sc.p1.rows, sc.p1.cols);
        s.preprocess();
        ASSERT_FALSE(HasFatalFailure());
        ASSERT_EQ(s.compute(), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        hipStream_t capture = nullptr;
        ASSERT_EQ(hipStreamCreate(&capture), hipSuccess);
        device_vector<char> scratch(size_t(256));
        ASSERT_EQ(hipStreamBeginCapture(capture, hipStreamCaptureModeGlobal), hipSuccess);
        EXPECT_EQ(hipMemsetAsync(scratch.ptr, 0, scratch.n, capture), hipSuccess);

        rocsparse_status st = rocsparse_status_internal_error;
        EXPECT_NO_THROW(
            st = rocsparse_csr_set_pointers(mat.d, d2.first.ptr, d2.second.ptr, d2.val.ptr));
        EXPECT_EQ(st, rocsparse_status_success);

        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        EXPECT_EQ(hipStreamIsCapturing(capture, &status), hipSuccess);
        EXPECT_EQ(status, hipStreamCaptureStatusActive);
        hipGraph_t graph = nullptr;
        EXPECT_EQ(hipStreamEndCapture(capture, &graph), hipSuccess);
        EXPECT_NE(graph, nullptr);
        if(graph != nullptr)
        {
            EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
        }
        EXPECT_EQ(hipStreamDestroy(capture), hipSuccess);

        s.preprocess();
        ASSERT_FALSE(HasFatalFailure());
        ASSERT_EQ(s.compute(), rocsparse_status_success);
        EXPECT_TRUE(expect_near(s.want(sc.p2), s.y(), "product after set_pointers"));
    }

    // set_pointers keeps the row and column arrays and replaces only the values.
    // The v2 analysis is refused all the same; v1 computes the new product.
    template <typename T>
    void values_only(rocsparse_handle handle, fmt f, const config& c)
    {
        SCOPED_TRACE(::testing::Message()
                     << fmt_name(f) << " alg=" << c.alg << " trans=" << c.trans << " v2=" << c.v2);

        const scenario<T, int32_t> sc = make_scenario<T, int32_t>(f, 0);
        device_problem<T, int32_t> d1(sc.p1), d3(sc.p3);
        ASSERT_EQ(sc.p1.first, sc.p3.first);
        ASSERT_EQ(sc.p1.second, sc.p3.second);

        spmat_guard mat;
        create(mat, sc.p1, d1);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        session<T> s(handle, c, mat.d, sc.p1.rows, sc.p1.cols);
        s.preprocess();
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        ASSERT_EQ(s.compute(), rocsparse_status_success);
        EXPECT_TRUE(expect_near(s.want(sc.p1), s.y(), "initial product"));

        ASSERT_EQ(set_arrays(mat, f, d1.first.ptr, d1.second.ptr, d3.val.ptr),
                  rocsparse_status_success);

        const rocsparse_status st = s.compute();
        if(c.v2)
        {
            EXPECT_EQ(st, rocsparse_status_invalid_value) << "compute without analysis";
            EXPECT_TRUE(expect_near(s.y0(), s.y(), "y after the refused compute"));
        }
        else
        {
            EXPECT_EQ(st, rocsparse_status_success) << "compute without preprocessing";
            EXPECT_TRUE(expect_near(s.want(sc.p3), s.y(), "compute without preprocessing"));
        }

        s.preprocess();
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        ASSERT_EQ(s.compute(), rocsparse_status_success);
        EXPECT_TRUE(expect_near(s.want(sc.p3), s.y(), "product after the new preprocessing"));
    }

    TEST_F(L2DeviceSpmvSetPointers, values_only)
    {
        for(const bool v2 : {false, true})
            for(const auto& fc : cases_with_analysis())
            {
                config c;
                c.alg   = fc.second.alg;
                c.trans = fc.second.trans;
                c.v2    = v2;
                values_only<double>(handle, fc.first, c);
            }
    }

    // An analyzed rocsparse_spmv_descr refuses to compute with another matrix
    // descriptor, even one over the same arrays, and still computes with its own.
    // The adaptive, LRB and nnz-split compute stages already refuse a matrix
    // descriptor other than the analyzed one, so this guards the documented
    // contract rather than the set_pointers check.
    template <typename T>
    void v2_another_descriptor(rocsparse_handle handle, fmt f, const config& c)
    {
        SCOPED_TRACE(::testing::Message()
                     << fmt_name(f) << " alg=" << c.alg << " trans=" << c.trans);

        const scenario<T, int32_t> sc = make_scenario<T, int32_t>(f, 0);
        device_problem<T, int32_t> d1(sc.p1);

        spmat_guard mat_a, mat_b;
        create(mat_a, sc.p1, d1);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        create(mat_b, sc.p1, d1);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());

        session<T> s(handle, c, mat_a.d, sc.p1.rows, sc.p1.cols);
        s.preprocess();
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        ASSERT_EQ(s.compute(), rocsparse_status_success);
        EXPECT_TRUE(expect_near(s.want(sc.p1), s.y(), "product of the analyzed descriptor"));

        s.set_matrix(mat_b.d);
        EXPECT_EQ(s.compute(), rocsparse_status_invalid_value) << "another matrix descriptor";
        EXPECT_TRUE(expect_near(s.y0(), s.y(), "y after the refused compute"));

        s.set_matrix(mat_a.d);
        ASSERT_EQ(s.compute(), rocsparse_status_success);
        EXPECT_TRUE(expect_near(s.want(sc.p1), s.y(), "product of the analyzed descriptor again"));
    }

    TEST_F(L2DeviceSpmvSetPointers, v2_another_descriptor_over_the_same_arrays)
    {
        for(const auto& fc : cases_with_analysis())
        {
            config c;
            c.alg   = fc.second.alg;
            c.trans = fc.second.trans;
            c.v2    = true;
            v2_another_descriptor<double>(handle, fc.first, c);
        }
    }

    // Runs stage while the handle stream is being captured in global mode, checks
    // that the capture is still active after it, and returns its status.
    template <typename F>
    rocsparse_status during_handle_stream_capture(rocsparse_handle handle, F&& stage)
    {
        rocsparse_status st    = rocsparse_status_internal_error;
        hipStream_t      saved = nullptr;
        EXPECT_EQ(rocsparse_get_stream(handle, &saved), rocsparse_status_success);
        hipStream_t capture = nullptr;
        EXPECT_EQ(hipStreamCreate(&capture), hipSuccess);
        EXPECT_EQ(rocsparse_set_stream(handle, capture), rocsparse_status_success);
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

        EXPECT_EQ(hipStreamBeginCapture(capture, hipStreamCaptureModeGlobal), hipSuccess);
        EXPECT_NO_THROW(st = stage());
        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        EXPECT_EQ(hipStreamIsCapturing(capture, &status), hipSuccess);
        EXPECT_EQ(status, hipStreamCaptureStatusActive) << "the capture was invalidated";
        hipGraph_t graph = nullptr;
        EXPECT_EQ(hipStreamEndCapture(capture, &graph), hipSuccess);
        if(graph != nullptr)
        {
            EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
        }

        EXPECT_EQ(rocsparse_set_stream(handle, saved), rocsparse_status_success);
        EXPECT_EQ(hipStreamDestroy(capture), hipSuccess);
        return st;
    }

    // The first preprocess after set_pointers frees the analysis of the previous
    // arrays. For CSR, CSC and BSR the free synchronizes the device; for COO only
    // the atomic analysis of a non-transposed matrix that replaces it does. While
    // the handle stream is being captured, a preprocess (v1) or analysis (v2, on a
    // new rocsparse_spmv_descr) that would synchronize returns invalid_value
    // without freeing anything, and the one that follows outside the capture
    // analyzes the new arrays. The others analyze the new arrays during the
    // capture and leave it active.
    template <typename T>
    void preprocess_during_capture(rocsparse_handle handle,
                                   fmt              f,
                                   const config&    c,
                                   rocsparse_status expected)
    {
        SCOPED_TRACE(::testing::Message()
                     << fmt_name(f) << " alg=" << c.alg << " trans=" << c.trans << " v2=" << c.v2);

        const scenario<T, int32_t> sc = make_scenario<T, int32_t>(f, 0);
        device_problem<T, int32_t> d1(sc.p1), d2(sc.p2);

        spmat_guard mat;
        create(mat, sc.p1, d1);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        session<T> s(handle, c, mat.d, sc.p1.rows, sc.p1.cols);
        s.preprocess();
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        ASSERT_EQ(s.compute(), rocsparse_status_success);
        EXPECT_TRUE(expect_near(s.want(sc.p1), s.y(), "initial product"));
        ASSERT_EQ(set_pointers(mat, sc.p2, d2), rocsparse_status_success);

        size_t                               bytes = 0;
        std::unique_ptr<device_vector<char>> buffer;
        if(c.v2)
        {
            ASSERT_EQ(s.new_descr_analysis_buffer_size(&bytes), rocsparse_status_success);
            buffer.reset(new device_vector<char>(std::max<size_t>(bytes, 1)));
        }

        const rocsparse_status st = during_handle_stream_capture(handle, [&] {
            return c.v2 ? s.analysis_stage(bytes, bytes ? buffer->ptr : nullptr)
                        : s.preprocess_stage();
        });
        EXPECT_EQ(st, expected) << "preprocess during the capture";
        ASSERT_FALSE(::testing::Test::HasFailure());

        if(expected != rocsparse_status_success)
        {
            s.preprocess();
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
        }
        ASSERT_EQ(s.compute(), rocsparse_status_success);
        EXPECT_TRUE(expect_near(s.want(sc.p2), s.y(), "product after the capture"));
    }

    TEST_F(L2DeviceSpmvSetPointers, preprocess_after_set_pointers_during_stream_capture)
    {
        for(const auto& fc : cases_with_analysis())
        {
            config c;
            c.alg   = fc.second.alg;
            c.trans = fc.second.trans;
            preprocess_during_capture<double>(handle, fc.first, c, rocsparse_status_invalid_value);
            if(HasFailure())
                return;
        }

        struct coo_case
        {
            rocsparse_spmv_alg  alg;
            rocsparse_operation trans;
            rocsparse_status    expected;
        };
        const coo_case coo_cases[]
            = {{rocsparse_spmv_alg_default, rocsparse_operation_none, rocsparse_status_success},
               {rocsparse_spmv_alg_coo, rocsparse_operation_none, rocsparse_status_success},
               {rocsparse_spmv_alg_coo_atomic,
                rocsparse_operation_transpose,
                rocsparse_status_success},
               {rocsparse_spmv_alg_coo_atomic,
                rocsparse_operation_none,
                rocsparse_status_invalid_value}};
        for(const bool v2 : {false, true})
            for(const coo_case& cc : coo_cases)
            {
                config c;
                c.alg   = cc.alg;
                c.trans = cc.trans;
                c.v2    = v2;
                preprocess_during_capture<double>(handle, fmt::coo, c, cc.expected);
                if(HasFailure())
                    return;
            }
    }

    // With the null stream as handle stream, querying its capture status fails
    // with hipErrorStreamCaptureImplicit while a blocking stream is being captured.
    // The preprocess after set_pointers returns invalid_value, as during a capture
    // of the handle stream, and the preprocess after the capture analyzes the new
    // arrays.
    TEST_F(L2DeviceSpmvSetPointers, preprocess_after_set_pointers_on_the_null_stream_during_capture)
    {
        using T                 = double;
        using I                 = int32_t;
        const scenario<T, I> sc = make_scenario<T, I>(fmt::csr, 0);
        device_problem<T, I> d1(sc.p1), d2(sc.p2);

        hipStream_t saved = nullptr;
        ASSERT_EQ(rocsparse_get_stream(handle, &saved), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_stream(handle, nullptr), rocsparse_status_success);

        spmat_guard mat;
        create(mat, sc.p1, d1);
        ASSERT_FALSE(HasFatalFailure());
        config c;
        c.alg = rocsparse_spmv_alg_csr_adaptive;
        session<T> s(handle, c, mat.d, sc.p1.rows, sc.p1.cols);
        s.preprocess();
        ASSERT_FALSE(HasFatalFailure());
        ASSERT_EQ(s.compute(), rocsparse_status_success);
        EXPECT_TRUE(expect_near(s.want(sc.p1), s.y(), "initial product"));
        ASSERT_EQ(set_pointers(mat, sc.p2, d2), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        hipStream_t capture = nullptr;
        ASSERT_EQ(hipStreamCreate(&capture), hipSuccess);
        device_vector<char> scratch(size_t(256));
        ASSERT_EQ(hipStreamBeginCapture(capture, hipStreamCaptureModeGlobal), hipSuccess);
        EXPECT_EQ(hipMemsetAsync(scratch.ptr, 0, scratch.n, capture), hipSuccess);

        rocsparse_status st = rocsparse_status_success;
        EXPECT_NO_THROW(st = s.preprocess_stage());
        EXPECT_EQ(st, rocsparse_status_invalid_value) << "preprocess during the capture";

        hipGraph_t graph = nullptr;
        EXPECT_EQ(hipStreamEndCapture(capture, &graph), hipSuccess);
        if(graph != nullptr)
        {
            EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
        }
        EXPECT_EQ(hipStreamDestroy(capture), hipSuccess);
        ASSERT_FALSE(HasFailure());

        s.preprocess();
        ASSERT_FALSE(HasFatalFailure());
        ASSERT_EQ(s.compute(), rocsparse_status_success);
        EXPECT_TRUE(expect_near(s.want(sc.p2), s.y(), "product after the capture"));
        EXPECT_EQ(rocsparse_set_stream(handle, saved), rocsparse_status_success);
    }

    // The counterpart of values_only: rocsparse_spmat_set_values replaces only the
    // values and keeps the analysis. The v2 compute stage still succeeds, and the
    // v1 preprocess stage has nothing to free, so it succeeds while the handle
    // stream is being captured, where it returns invalid_value after set_pointers.
    template <typename T>
    void set_values_keeps_analysis(rocsparse_handle handle, fmt f, const config& c)
    {
        SCOPED_TRACE(::testing::Message()
                     << fmt_name(f) << " alg=" << c.alg << " trans=" << c.trans << " v2=" << c.v2);

        const scenario<T, int32_t> sc = make_scenario<T, int32_t>(f, 0);
        device_problem<T, int32_t> d1(sc.p1), d3(sc.p3);

        spmat_guard mat;
        create(mat, sc.p1, d1);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        session<T> s(handle, c, mat.d, sc.p1.rows, sc.p1.cols);
        s.preprocess();
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        ASSERT_EQ(s.compute(), rocsparse_status_success);
        EXPECT_TRUE(expect_near(s.want(sc.p1), s.y(), "initial product"));

        ASSERT_EQ(rocsparse_spmat_set_values(mat.d, d3.val.ptr), rocsparse_status_success);
        if(!c.v2)
        {
            const rocsparse_status st
                = during_handle_stream_capture(handle, [&] { return s.preprocess_stage(); });
            EXPECT_EQ(st, rocsparse_status_success) << "preprocess during the capture";
            ASSERT_FALSE(::testing::Test::HasFailure());
        }
        ASSERT_EQ(s.compute(), rocsparse_status_success);
        EXPECT_TRUE(expect_near(s.want(sc.p3), s.y(), "product after set_values"));
    }

    TEST_F(L2DeviceSpmvSetPointers, set_values_keeps_the_analysis)
    {
        for(const bool v2 : {false, true})
            for(const auto& fc : cases_with_analysis())
            {
                config c;
                c.alg   = fc.second.alg;
                c.trans = fc.second.trans;
                c.v2    = v2;
                set_values_keeps_analysis<double>(handle, fc.first, c);
                if(HasFailure())
                    return;
            }
    }

    // A second analysis on the same v2 descriptor, after set_pointers, must not
    // adopt the CSR, CSC or BSR info built for the previous arrays. That analysis is
    // refused by the existing rule that a rocsparse_spmv_descr is analyzed only
    // once; the compute stage that follows is refused by the set_pointers check.
    template <typename T>
    void v2_reanalysis_same_descr(rocsparse_handle handle, fmt f, const config& c)
    {
        SCOPED_TRACE(::testing::Message()
                     << fmt_name(f) << " alg=" << c.alg << " trans=" << c.trans);

        const scenario<T, int32_t> sc = make_scenario<T, int32_t>(f, 0);
        device_problem<T, int32_t> d1(sc.p1), d2(sc.p2);

        spmat_guard mat;
        create(mat, sc.p1, d1);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        session<T> s(handle, c, mat.d, sc.p1.rows, sc.p1.cols);
        s.preprocess();
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        ASSERT_EQ(s.compute(), rocsparse_status_success);
        EXPECT_TRUE(expect_near(s.want(sc.p1), s.y(), "initial product"));

        ASSERT_EQ(set_pointers(mat, sc.p2, d2), rocsparse_status_success);
        EXPECT_EQ(s.analyze_same_descr(), rocsparse_status_invalid_value)
            << "second analysis of the same descriptor";
        EXPECT_EQ(s.compute(), rocsparse_status_invalid_value) << "compute after the refusal";
        EXPECT_TRUE(expect_near(s.y0(), s.y(), "y after the refused compute"));
    }

    TEST_F(L2DeviceSpmvSetPointers, v2_reanalysis_of_the_same_descriptor)
    {
        for(const auto& fc : cases_with_analysis())
        {
            config c;
            c.alg   = fc.second.alg;
            c.trans = fc.second.trans;
            c.v2    = true;
            v2_reanalysis_same_descr<double>(handle, fc.first, c);
            if(HasFailure())
                return;
        }
    }

    // Destroying an analysis synchronizes the device and frees its buffers, which
    // fails while another stream is being captured in global mode. Neither the
    // rocsparse_spmv_descr of a v2 analysis nor the matrix descriptor of a v1
    // preprocessing may throw from its destructor, for every info that holds
    // device buffers.
    TEST_F(L2DeviceSpmvSetPointers, destroy_analysis_during_stream_capture)
    {
        using T = double;
        using I = int32_t;
        const std::pair<fmt, rocsparse_spmv_alg> cases[]
            = {{fmt::csr, rocsparse_spmv_alg_csr_adaptive},
               {fmt::csr, rocsparse_spmv_alg_csr_lrb},
               {fmt::csr, rocsparse_spmv_alg_csr_nnzsplit},
               {fmt::bsr, rocsparse_spmv_alg_default}};
        for(const auto& fc : cases)
            for(const bool v2 : {false, true})
            {
                SCOPED_TRACE(::testing::Message()
                             << fmt_name(fc.first) << " alg=" << fc.second << " v2=" << v2);
                const scenario<T, I> sc = make_scenario<T, I>(fc.first, 0);
                device_problem<T, I> d1(sc.p1);

                spmat_guard mat;
                create(mat, sc.p1, d1);
                ASSERT_FALSE(HasFatalFailure());
                config c;
                c.alg = fc.second;
                c.v2  = v2;
                session<T> s(handle, c, mat.d, sc.p1.rows, sc.p1.cols);
                s.preprocess();
                ASSERT_FALSE(HasFatalFailure());
                ASSERT_EQ(s.compute(), rocsparse_status_success);
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

                hipStream_t capture = nullptr;
                ASSERT_EQ(hipStreamCreate(&capture), hipSuccess);
                device_vector<char> scratch(size_t(256));
                ASSERT_EQ(hipStreamBeginCapture(capture, hipStreamCaptureModeGlobal), hipSuccess);
                EXPECT_EQ(hipMemsetAsync(scratch.ptr, 0, scratch.n, capture), hipSuccess);

                rocsparse_status st = rocsparse_status_internal_error;
                if(v2)
                {
                    EXPECT_NO_THROW(st = s.destroy_v2_descr());
                }
                else
                {
                    EXPECT_NO_THROW(st = rocsparse_destroy_spmat_descr(mat.d));
                    mat.d = nullptr;
                }
                EXPECT_EQ(st, rocsparse_status_success);

                // The synchronization attempt may have invalidated the capture.
                hipGraph_t graph = nullptr;
                (void)hipStreamEndCapture(capture, &graph);
                if(graph != nullptr)
                {
                    (void)hipGraphDestroy(graph);
                }
                (void)hipGetLastError();
                EXPECT_EQ(hipStreamDestroy(capture), hipSuccess);
                EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
                if(HasFailure())
                    return;
            }
    }
}
