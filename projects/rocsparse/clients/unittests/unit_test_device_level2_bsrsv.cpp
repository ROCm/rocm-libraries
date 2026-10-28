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
// Device unit tests of the block triangular solve rocsparse_Xbsrsv_*: every block
// dimension class, both fill modes, unit and non-unit diagonals, no transpose and
// transpose, both block storage directions, index bases and pointer modes. The
// blocks of the matrices are strictly diagonally dominant (non-unit diagonal) or
// small enough for the triangular recurrence to stay bounded (unit diagonal).
// Entries stored in the other triangle of the matrix, including that of the
// diagonal blocks, hold 37, which the solve must not read and the host reference
// ignores.
//
#include "unit_test_device_level2_common.hpp"

using namespace level2;

namespace
{
    template <typename T>
    struct bsrsv_matrix
    {
        int32_t             mb   = 0;
        int32_t             bd   = 0;
        int                 base = 0;
        rocsparse_direction dir  = rocsparse_direction_row;
        // block structure (with the index base) and values in the block storage order
        std::vector<int32_t> ptr;
        std::vector<int32_t> ind;
        std::vector<T>       val;
        // the same matrix as a scalar CSR matrix (index base 0)
        host_csr<T> scalar;

        int32_t nnzb() const
        {
            return static_cast<int32_t>(ind.size());
        }
    };

    // Block rows with the block columns {i - 3, i - 1, i} (lower) or {i, i + 1, i + 3}
    // (upper), plus the last (lower) or first (upper) block column when it is in the
    // other triangle. With `diag_blocks` false the diagonal block is left out.
    template <typename T>
    bsrsv_matrix<T> make_bsrsv(int32_t             mb,
                               int32_t             bd,
                               rocsparse_fill_mode fill,
                               rocsparse_direction dir,
                               int                 base,
                               bool                diag_blocks = true)
    {
        bsrsv_matrix<T> a;
        a.mb   = mb;
        a.bd   = bd;
        a.base = base;
        a.dir  = dir;

        const bool lower = (fill == rocsparse_fill_mode_lower);

        // Off-diagonal entries are scaled so that a scalar row holds at most 3 * bd of them
        // with a combined magnitude below 1.
        const T scale = rocsparse_ut::scalar<T>(1.0f / (12.0f * static_cast<float>(bd)));

        std::vector<std::vector<int32_t>> cols(mb);
        for(int32_t i = 0; i < mb; ++i)
        {
            std::vector<int32_t> c;
            if(lower)
            {
                if(i >= 3)
                    c.push_back(i - 3);
                if(i >= 1)
                    c.push_back(i - 1);
                if(diag_blocks)
                    c.push_back(i);
            }
            else
            {
                if(diag_blocks)
                    c.push_back(i);
                if(i + 1 < mb)
                    c.push_back(i + 1);
                if(i + 3 < mb)
                    c.push_back(i + 3);
            }
            // A block in the other triangle, which the solve must not read.
            if(lower && i + 1 < mb)
                c.push_back(mb - 1);
            if(!lower && i > 0)
                c.insert(c.begin(), 0);
            cols[i] = c;
        }

        const size_t blk = static_cast<size_t>(bd) * bd;
        a.ptr.push_back(base);
        a.scalar.m    = mb * bd;
        a.scalar.n    = mb * bd;
        a.scalar.base = 0;
        a.scalar.ptr.assign(1, 0);
        int64_t counter = 0;

        // value of every scalar entry, by (scalar row, scalar column)
        std::vector<std::vector<std::pair<int32_t, T>>> rows(static_cast<size_t>(mb) * bd);

        for(int32_t i = 0; i < mb; ++i)
        {
            for(int32_t j : cols[i])
            {
                const size_t first = a.val.size();
                a.val.resize(first + blk, T{});
                for(int32_t r = 0; r < bd; ++r)
                    for(int32_t c = 0; c < bd; ++c)
                    {
                        T v{};
                        if(lower ? j > i : j < i)
                            v = rocsparse_ut::scalar<T>(37.0f);
                        else if(i == j)
                        {
                            // triangular diagonal block; the solve must not read the
                            // other triangle, so it holds 37
                            const bool keep = lower ? (c <= r) : (c >= r);
                            if(keep)
                            {
                                v = value_of<T>(counter++, true);
                                v = (r == c) ? v + rocsparse_ut::scalar<T>(5.0f) : v * scale;
                            }
                            else
                                v = rocsparse_ut::scalar<T>(37.0f);
                        }
                        else
                        {
                            v = value_of<T>(counter++, true) * scale;
                        }
                        const size_t pos
                            = (dir == rocsparse_direction_row) ? r * bd + c : c * bd + r;
                        a.val[first + pos] = v;
                        if(magnitude(v) != 0.0)
                            rows[static_cast<size_t>(i) * bd + r].push_back({j * bd + c, v});
                    }
                a.ind.push_back(j + base);
            }
            a.ptr.push_back(static_cast<int32_t>(a.ind.size()) + base);
        }
        for(auto& r : rows)
        {
            std::sort(
                r.begin(), r.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
            for(const auto& e : r)
            {
                a.scalar.ind.push_back(e.first);
                a.scalar.val.push_back(e.second);
            }
            a.scalar.ptr.push_back(static_cast<int32_t>(a.scalar.ind.size()));
        }
        return a;
    }

    struct solve_case
    {
        int32_t             bd          = 4;
        int32_t             mb          = 12;
        rocsparse_fill_mode fill        = rocsparse_fill_mode_lower;
        rocsparse_diag_type diag        = rocsparse_diag_type_non_unit;
        rocsparse_operation trans       = rocsparse_operation_none;
        rocsparse_direction dir         = rocsparse_direction_row;
        int                 base        = 0;
        bool                device      = false;
        float               alpha       = 1.0f;
        bool                diag_blocks = true;
    };

    template <typename T>
    void run_bsrsv(rocsparse_handle handle, const solve_case& c)
    {
        SCOPED_TRACE(::testing::Message()
                     << "bsrsv bd=" << c.bd << " mb=" << c.mb << " fill=" << c.fill
                     << " diag=" << c.diag << " trans=" << c.trans << " dir=" << c.dir
                     << " base=" << c.base << " device=" << c.device << " alpha=" << c.alpha);

        const bsrsv_matrix<T> A = make_bsrsv<T>(c.mb, c.bd, c.fill, c.dir, c.base, c.diag_blocks);
        const int32_t         m = c.mb * c.bd;

        const T              alpha = scalar_of<T>(c.alpha, 0.25f);
        const std::vector<T> x     = make_vector<T>(m, 3);

        device_vector<rocsparse_int> dptr(convert_vector<rocsparse_int>(A.ptr));
        guarded_indices<>            dind(A.ind);
        device_vector<T>             dval(A.val), dx(x), dy(std::vector<T>(m, T{})),
            dalpha(std::vector<T>(1, alpha));
        ASSERT_TRUE(dptr.ptr && dind.ptr && dval.ptr && dx.ptr && dy.ptr && dalpha.ptr);

        mat_descr descr;
        ASSERT_TRUE(descr.d);
        ASSERT_EQ(rocsparse_set_mat_index_base(descr.d, to_base(c.base)), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_fill_mode(descr.d, c.fill), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_diag_type(descr.d, c.diag), rocsparse_status_success);
        mat_info info;
        ASSERT_TRUE(info.d);

        scoped_pointer_mode mode(handle, c.device);

        size_t bytes = 0;
        ASSERT_EQ(bsrsv_buffer_size<T>(handle,
                                       c.dir,
                                       c.trans,
                                       c.mb,
                                       A.nnzb(),
                                       descr.d,
                                       dval.ptr,
                                       dptr.ptr,
                                       dind.ptr,
                                       c.bd,
                                       info.d,
                                       &bytes),
                  rocsparse_status_success);
        device_vector<char> dbuf(std::max<size_t>(bytes, 1));
        ASSERT_TRUE(dbuf.ptr);
        ASSERT_EQ(bsrsv_analysis<T>(handle,
                                    c.dir,
                                    c.trans,
                                    c.mb,
                                    A.nnzb(),
                                    descr.d,
                                    dval.ptr,
                                    dptr.ptr,
                                    dind.ptr,
                                    c.bd,
                                    info.d,
                                    rocsparse_analysis_policy_reuse,
                                    rocsparse_solve_policy_auto,
                                    dbuf.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(bsrsv_solve<T>(handle,
                                 c.dir,
                                 c.trans,
                                 c.mb,
                                 A.nnzb(),
                                 c.device ? dalpha.ptr : &alpha,
                                 descr.d,
                                 dval.ptr,
                                 dptr.ptr,
                                 dind.ptr,
                                 c.bd,
                                 info.d,
                                 dx.ptr,
                                 dy.ptr,
                                 rocsparse_solve_policy_auto,
                                 dbuf.ptr),
                  rocsparse_status_success);

        // The position lives in host or device memory according to the pointer mode.
        rocsparse_int                position = 0;
        device_vector<rocsparse_int> dposition(std::vector<rocsparse_int>(1, 0));
        ASSERT_TRUE(dposition.ptr);
        // A non-unit diagonal needs the diagonal blocks: without them the solve reports a
        // zero pivot (the matrix of a unit diagonal solve does not need them either).
        const bool singular = !c.diag_blocks && c.diag == rocsparse_diag_type_non_unit;
        ASSERT_EQ(rocsparse_bsrsv_zero_pivot(handle, info.d, c.device ? dposition.ptr : &position),
                  singular ? rocsparse_status_zero_pivot : rocsparse_status_success);
        if(c.device)
            position = to_host(dposition)[0];
        if(singular)
        {
            // Every diagonal block is missing, so the first zero pivot is block row 0.
            EXPECT_EQ(position, c.base);
            ASSERT_EQ(rocsparse_bsrsv_clear(handle, info.d), rocsparse_status_success);
            return;
        }
        EXPECT_EQ(position, -1);

        const std::vector<T> want
            = host_triangular_solve<T>(A.scalar, c.fill, c.diag, c.trans, alpha, x);
        for(const auto& v : want)
            ASSERT_TRUE(std::isfinite(magnitude(v)));
        EXPECT_TRUE(expect_near(want, to_host(dy), "bsrsv y"));

        ASSERT_EQ(rocsparse_bsrsv_clear(handle, info.d), rocsparse_status_success);
    }

    class L2DeviceBsrsv : public rocsparse_ut::HandleTest
    {
    protected:
        template <typename T>
        void sweep_dims(const std::vector<int32_t>& dims, int32_t mb)
        {
            int n = 0;
            for(const int32_t bd : dims)
                for(const auto fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
                    for(const auto diag : {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit})
                        for(const auto trans :
                            {rocsparse_operation_none, rocsparse_operation_transpose})
                        {
                            solve_case c;
                            c.bd     = bd;
                            c.mb     = mb;
                            c.fill   = fill;
                            c.diag   = diag;
                            c.trans  = trans;
                            c.dir    = ((n / 2) % 2) ? rocsparse_direction_column
                                                     : rocsparse_direction_row;
                            c.base   = (n / 3) % 2;
                            c.device = (n % 3) == 1;
                            c.alpha  = (n % 4 == 3) ? -2.0f : 1.0f;
                            ++n;
                            run_bsrsv<T>(handle, c);
                            if(HasFailure())
                                return;
                        }
        }
    };

    TEST_F(L2DeviceBsrsv, large_blocks_float)
    {
        sweep_dims<float>({33, 40}, 6);
    }

    TEST_F(L2DeviceBsrsv, double_and_complex)
    {
        sweep_dims<double>({3, 9, 20}, 8);
        sweep_dims<rocsparse_float_complex>({4, 17}, 8);
        sweep_dims<rocsparse_double_complex>({2, 33}, 6);
    }

    TEST_F(L2DeviceBsrsv, missing_diagonal_block_is_a_zero_pivot)
    {
        for(const int32_t bd : {2, 9})
        {
            solve_case c;
            c.bd          = bd;
            c.mb          = 8;
            c.diag_blocks = false;
            run_bsrsv<float>(handle, c);
        }
    }
}
