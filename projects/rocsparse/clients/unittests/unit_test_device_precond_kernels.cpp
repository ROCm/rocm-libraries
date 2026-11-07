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

// Binary-search kernels of csrilu0 and csrildlt0, which the usual tiny matrices do not
// select:
//
// csrilu0 binsearch:      wavefront 32 and max row nnz >= 512, wavefront 64 and
//                         max row nnz >= 1024
//                         (library/src/precond/csrilu0/rocsparse_csrilu0_kernel_launch.cpp)
// csrildlt0 binsearch:    max row nnz > 512
//                         (library/src/precond/csrildlt0/rocsparse_csrildlt0_kernel_launch.cpp)

#include "unit_test_device_precond_common.hpp"

#include "rocsparse_mat_info.hpp"
#include "rocsparse_spmat_descr.hpp"

#include <string>
#include <vector>

using namespace precond_ut;

namespace
{
    // What the matrix lacks: nothing, the diagonal entry of row 5 (a structural zero
    // pivot) or a non-zero diagonal of row 0 (an exact zero pivot). The values left after a
    // zero pivot are not specified, so they are only compared without one.
    enum class Pivot
    {
        none,
        structural,
        exact
    };

    // Lower-triangular matrix whose last row holds `fat` entries (columns 1..fat). Earlier
    // rows plant a binary-search hit (column 1 is in the fat row) and a miss (column 0 is
    // not). Diagonal 4, off-diagonal 0.1: the factorizations exist without a zero pivot.
    // `upper` mirrors it for csrilu0 (see with_upper_triangle): the fat row then finds
    // column m - 1 in every row it depends on, and the short rows miss it.
    template <typename T>
    csr_matrix<T> make_binsearch_matrix(int32_t fat, Pivot pivot, bool upper = false)
    {
        const int32_t                     m = fat + 1;
        std::vector<std::vector<int32_t>> rows(m);
        rows[0] = {0};
        rows[1] = {0, 1};
        rows[2] = {1, 2};
        rows[3] = {0, 3};
        for(int32_t r = 4; r < m - 1; ++r)
            rows[r] = {r};
        rows[5]
            = (pivot == Pivot::structural) ? std::vector<int32_t>{4} : std::vector<int32_t>{4, 5};
        for(int32_t c = m - fat; c < m; ++c)
            rows[m - 1].push_back(c);

        csr_matrix<T> a;
        a.n = m;
        a.ptr.push_back(0);
        for(int32_t r = 0; r < m; ++r)
        {
            for(const int32_t c : rows[r])
            {
                a.ind.push_back(c);
                const bool zero = (pivot == Pivot::exact && r == 0);
                a.val.push_back(make_value<T>(c == r ? (zero ? 0.0 : 4.0) : 0.1));
            }
            a.ptr.push_back(static_cast<int32_t>(a.ind.size()));
        }
        if(!upper)
            return a;
        csr_matrix<T> full = with_upper_triangle(a);
        if(pivot == Pivot::exact)
            full.val[full.ptr[0]] = make_value<T>(0.0);
        return full;
    }

    int64_t pivot_row(Pivot pivot)
    {
        return pivot == Pivot::none ? -1 : (pivot == Pivot::structural ? 5 : 0);
    }

    // csrilu0 through the legacy interface. singular_tol < 0 keeps the library default.
    template <typename T>
    void run_csrilu0(rocsparse_handle handle, Pivot pivot, double singular_tol, int32_t fat = 513)
    {
        SCOPED_TRACE("fat " + std::to_string(fat));
        const csr_matrix<T>          a = make_binsearch_matrix<T>(fat, pivot, true);
        const rocsparse_int          m = a.n, nnz = a.nnz();
        device_vector<rocsparse_int> drow{as_rocsparse_int(a.ptr)};
        device_vector<rocsparse_int> dcol{as_rocsparse_int(a.ind)};
        device_vector<T>             dval{a.val};
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr);
        descr_info di;
        ASSERT_TRUE(di.descr && di.info);

        size_t buffer_size = 0;
        ASSERT_EQ(PRECOND_CALL(T,
                               csrilu0_buffer_size,
                               handle,
                               m,
                               nnz,
                               di.descr,
                               dval.ptr,
                               drow.ptr,
                               dcol.ptr,
                               di.info,
                               &buffer_size),
                  rocsparse_status_success);
        device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(PRECOND_CALL(T,
                               csrilu0_analysis,
                               handle,
                               m,
                               nnz,
                               di.descr,
                               dval.ptr,
                               drow.ptr,
                               dcol.ptr,
                               di.info,
                               rocsparse_analysis_policy_reuse,
                               rocsparse_solve_policy_auto,
                               buffer.ptr),
                  rocsparse_status_success);
        if(singular_tol >= 0.0)
            ASSERT_EQ(rocsparse_csrilu0_set_tolerance(handle, di.info, singular_tol),
                      rocsparse_status_success);

        const rocsparse_status st = PRECOND_CALL(T,
                                                 csrilu0,
                                                 handle,
                                                 m,
                                                 nnz,
                                                 di.descr,
                                                 dval.ptr,
                                                 drow.ptr,
                                                 dcol.ptr,
                                                 di.info,
                                                 rocsparse_solve_policy_auto,
                                                 buffer.ptr);
        if(pivot == Pivot::none)
            ASSERT_EQ(st, rocsparse_status_success);
        else
            EXPECT_TRUE(st == rocsparse_status_success || st == rocsparse_status_zero_pivot) << st;
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        rocsparse_int position = -2;
        EXPECT_EQ(rocsparse_csrilu0_zero_pivot(handle, di.info, &position),
                  pivot == Pivot::none ? rocsparse_status_success : rocsparse_status_zero_pivot);
        EXPECT_EQ(position, pivot_row(pivot));
        if(pivot != Pivot::none)
            return;
        expect_diagonally_dominant(a);
        const auto got = to_host(dval);
        expect_close(got, host_ilu0(a), tolerance<T>(max_row_length(a)));
        expect_small_residual(ilu_pattern_residual(a, got), a);
    }

    // spildlt0 with I / J indices. `boost` sets the numeric boost with a tolerance that
    // replaces every pivot by one. spildlt0 does not apply the boost set on its
    // descriptor (AISPARSE-909; spic0 has the same gap, AISPARSE-927), so the test also
    // sets it on the matrix info, which the factorization reads. `copy_diag` also requests D
    // in a separate array. The singularity tolerance of the exact-pivot case is above every
    // pivot.
    template <typename T, typename I, typename J>
    void run_csrildlt0(rocsparse_handle handle, Pivot pivot, bool boost, bool copy_diag)
    {
        using real_t          = real_of<T>;
        using A               = api<Kind::ildlt>;
        const csr_matrix<T> a = make_binsearch_matrix<T>(513, pivot);
        dev_csr<T, I, J>    d;
        d.init(a, 1, true);
        ASSERT_NE(d.A, nullptr);
        device_vector<real_t> ddiag{static_cast<size_t>(a.n)};
        ASSERT_TRUE(ddiag.ptr);

        descr_guard<Kind::ildlt> guard(handle);
        auto&                    descr = guard.descr;
        ASSERT_EQ(A::create(handle, &descr), rocsparse_status_success);
        const rocsparse_spildlt0_alg    alg      = rocsparse_spildlt0_alg_default;
        const rocsparse_datatype        dtype    = dt_of<T>();
        const rocsparse_analysis_policy policy   = rocsparse_analysis_policy_reuse;
        const double                    sing_tol = (pivot == Pivot::exact) ? 1.0e6 : 1.0e-12;
        ASSERT_EQ(set_value<Kind::ildlt>(handle, descr, In::alg, &alg, sizeof(alg)),
                  rocsparse_status_success);
        ASSERT_EQ(set_value<Kind::ildlt>(handle, descr, In::dtype, &dtype, sizeof(dtype)),
                  rocsparse_status_success);
        ASSERT_EQ(set_value<Kind::ildlt>(handle, descr, In::policy, &policy, sizeof(policy)),
                  rocsparse_status_success);
        ASSERT_EQ(set_value<Kind::ildlt>(handle, descr, In::sing_tol, &sing_tol, sizeof(void*)),
                  rocsparse_status_success);
        if(copy_diag)
        {
            void* diag_ptr = ddiag.ptr;
            ASSERT_EQ(
                A::set(handle, descr, rocsparse_spildlt0_input_diag, &diag_ptr, sizeof(void*)),
                rocsparse_status_success);
        }
        real_t boost_tol = static_cast<real_t>(1.0e20);
        T      boost_val = make_value<T>(1.0);
        if(boost)
        {
            auto* numeric = d.A->info->get_boost();
            numeric->set_enable(1);
            numeric->set_tol_pointer_mode(rocsparse_pointer_mode_host);
            numeric->set_val_pointer_mode(rocsparse_pointer_mode_host);
            numeric->set_tol_datatype(dt_of<real_t>());
            numeric->set_tol(&boost_tol);
            numeric->set_val(&boost_val);
        }
        const int32_t boost_enable  = 1;
        const double  descr_tol     = 1.0e20;
        const void*   boost_val_ptr = &boost_val;
        if(boost)
        {
            ASSERT_EQ(set_value<Kind::ildlt>(
                          handle, descr, In::boost_enable, &boost_enable, sizeof(int32_t)),
                      rocsparse_status_success);
            ASSERT_EQ(
                set_value<Kind::ildlt>(handle, descr, In::boost_tol, &descr_tol, sizeof(double)),
                rocsparse_status_success);
            ASSERT_EQ(
                set_value<Kind::ildlt>(handle, descr, In::boost_val, boost_val_ptr, sizeof(void*)),
                rocsparse_status_success);
        }

        size_t analysis_size = 0, compute_size = 0;
        ASSERT_EQ(
            A::buffer_size(handle, descr, d.A, d.A, A::stage(Stage::analysis), &analysis_size),
            rocsparse_status_success);
        device_vector<char> analysis_buffer{analysis_size ? analysis_size : size_t{1}};
        ASSERT_TRUE(analysis_buffer.ptr);
        ASSERT_EQ(A::run(handle,
                         descr,
                         d.A,
                         d.A,
                         A::stage(Stage::analysis),
                         analysis_size,
                         analysis_buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(A::buffer_size(handle, descr, d.A, d.A, A::stage(Stage::compute), &compute_size),
                  rocsparse_status_success);
        device_vector<char> compute_buffer{compute_size ? compute_size : size_t{1}};
        ASSERT_TRUE(compute_buffer.ptr);

        ASSERT_EQ(A::run(handle,
                         descr,
                         d.A,
                         d.A,
                         A::stage(Stage::compute),
                         compute_size,
                         compute_buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        rocsparse_singularity singularity = rocsparse_singularity_numeric_near;
        int64_t               position    = -2;
        ASSERT_EQ(A::get(handle, descr, false, &singularity, sizeof(singularity)),
                  rocsparse_status_success);
        ASSERT_EQ(A::get(handle, descr, true, &position, sizeof(position)),
                  rocsparse_status_success);
        ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);
        EXPECT_EQ(position, pivot_row(pivot));
        EXPECT_EQ(singularity == rocsparse_singularity_none, pivot == Pivot::none) << singularity;
        if(pivot != Pivot::none)
            return;

        const auto           got    = to_host(d.val);
        const double         tol    = tolerance<T>(max_row_length(a));
        const std::vector<T> factor = host_ildlt0(a, boost ? 1.0e20 : -1.0, 1.0);
        expect_close(got, factor, tol);
        if(copy_diag)
        {
            std::vector<real_t> want(static_cast<size_t>(a.n));
            for(int32_t r = 0; r < a.n; ++r)
                want[r] = static_cast<real_t>(re(factor[diagonal_position(a, r)]));
            expect_close(to_host(ddiag), want, tol);
        }
    }
}

class PrecondCsrilu0Binsearch : public HandleTest
{
};

TEST_F(PrecondCsrilu0Binsearch, f32_singular_and_zero)
{
    run_csrilu0<float>(handle, Pivot::none, 1.0e6);
    run_csrilu0<float>(handle, Pivot::structural, 1.0e6);
    run_csrilu0<float>(handle, Pivot::exact, 0.0);
}

TEST_F(PrecondCsrilu0Binsearch, value_types_1024)
{
    // max nnz 1024 selects the binary search on both wave sizes (513 does on wave32 only).
    run_csrilu0<float>(handle, Pivot::none, -1.0, 1024);
    run_csrilu0<double>(handle, Pivot::none, -1.0, 1024);
    run_csrilu0<rocsparse_float_complex>(handle, Pivot::none, -1.0, 1024);
    run_csrilu0<rocsparse_double_complex>(handle, Pivot::none, -1.0, 1024);
}

class PrecondCsrildlt0Binsearch : public HandleTest
{
};

TEST_F(PrecondCsrildlt0Binsearch, f32_layouts)
{
    run_csrildlt0<float, int32_t, int32_t>(handle, Pivot::none, false, true);
    run_csrildlt0<float, int64_t, int32_t>(handle, Pivot::none, false, false);
    run_csrildlt0<float, int64_t, int64_t>(handle, Pivot::none, false, false);
    run_csrildlt0<float, int32_t, int32_t>(handle, Pivot::none, true, false);
    run_csrildlt0<float, int32_t, int32_t>(handle, Pivot::structural, false, false);
    run_csrildlt0<float, int32_t, int32_t>(handle, Pivot::exact, false, false);
}

TEST_F(PrecondCsrildlt0Binsearch, value_types)
{
    run_csrildlt0<double, int32_t, int32_t>(handle, Pivot::none, false, false);
    run_csrildlt0<rocsparse_float_complex, int32_t, int32_t>(handle, Pivot::none, false, false);
    run_csrildlt0<rocsparse_double_complex, int32_t, int32_t>(handle, Pivot::none, false, false);
    run_csrildlt0<double, int64_t, int32_t>(handle, Pivot::none, false, false);
    run_csrildlt0<double, int64_t, int64_t>(handle, Pivot::none, false, false);
}
