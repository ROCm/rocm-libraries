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

#pragma once

// What the scenario tests of the CSR and BSR triangular analysis data share: the
// expectations (today's value, the documented one and the known defect between them),
// the host solves, and the zero pivot queries in host or device memory.

#include "unit_test_device_precond_common.hpp"

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace triangular_scenarios
{
    using namespace precond_ut;

    // A zero pivot is expressed as its zero based row; no_pivot when none is reported.
    constexpr int32_t no_pivot = -1;
    // Returned for a report that is not self-consistent (already flagged as a failure).
    constexpr int32_t invalid_report = -2;

    inline std::string describe(int32_t v)
    {
        return std::to_string(v);
    }

    inline std::string describe(rocsparse_status s)
    {
        return rocsparse_get_status_name(s);
    }

    inline std::string describe(bool v)
    {
        return v ? "true" : "false";
    }

    inline std::string describe(double v)
    {
        std::ostringstream s;
        s << v;
        return s.str();
    }

    template <typename V>
    struct expectation
    {
        V           current;
        V           documented;
        const char* defect = nullptr;
    };

    template <typename V>
    expectation<V> is(V v)
    {
        return {v, v, nullptr};
    }

    template <typename V>
    expectation<V> known_defect(V current, V documented, const char* defect)
    {
        return {current, documented, defect};
    }

    template <typename V>
    void check(const V& got, const expectation<V>& want, const std::string& what)
    {
        if(want.defect != nullptr && !(got == want.current) && got == want.documented)
        {
            ADD_FAILURE() << what << ": the known defect \"" << want.defect
                          << "\" no longer reproduces, the documented " << describe(got)
                          << " is returned; make the expectation is(" << describe(got) << ")";
            return;
        }
        EXPECT_TRUE(got == want.current)
            << what << ": got " << describe(got) << ", expected " << describe(want.current)
            << (want.defect ? std::string(" (known defect: ") + want.defect + ", documented "
                                  + describe(want.documented) + ")"
                            : std::string());
    }

    // Whether `got` matches `want` entry by entry, relative to `tol`; for the checks
    // whose documented and current results differ by more than rounding.
    inline bool
        all_close(const std::vector<double>& got, const std::vector<double>& want, double tol)
    {
        if(got.size() != want.size())
            return false;
        for(size_t i = 0; i < got.size(); ++i)
            if(!(std::abs(got[i] - want[i]) <= tol * (1.0 + std::abs(want[i]))))
                return false;
        return true;
    }

    // `what` with the two buffer sizes it compares.
    inline std::string compared(const std::string& what, size_t size, size_t against)
    {
        return what + " (" + std::to_string(size) + " against " + std::to_string(against) + ")";
    }

    inline const char* name(rocsparse_analysis_policy policy)
    {
        return (policy == rocsparse_analysis_policy_reuse) ? "reuse" : "force";
    }

    inline const char* name(rocsparse_index_base base)
    {
        return (base == rocsparse_index_base_one) ? "base one" : "base zero";
    }

    inline const char* name(rocsparse_fill_mode fill)
    {
        return (fill == rocsparse_fill_mode_lower) ? "lower" : "upper";
    }

    inline const char* name(rocsparse_diag_type diag)
    {
        return (diag == rocsparse_diag_type_unit) ? "unit" : "non unit";
    }

    enum class order
    {
        lower_first,
        upper_first
    };

    inline const char* name(order o)
    {
        return (o == order::lower_first) ? "L analysed first" : "U analysed first";
    }

    constexpr rocsparse_analysis_policy policies[]
        = {rocsparse_analysis_policy_reuse, rocsparse_analysis_policy_force};
    constexpr rocsparse_index_base bases[]  = {rocsparse_index_base_zero, rocsparse_index_base_one};
    constexpr order                orders[] = {order::lower_first, order::upper_first};

    inline std::vector<double> right_hand_side(int32_t n, int32_t nrhs = 1)
    {
        std::vector<double> b(static_cast<size_t>(n) * nrhs);
        for(int32_t j = 0; j < nrhs; ++j)
            for(int32_t i = 0; i < n; ++i)
                b[static_cast<size_t>(j) * n + i] = 1.0 + 0.1 * i + j;
        return b;
    }

    // The `cols` x `rows` column major transpose of the `rows` x `cols` column major `v`.
    inline std::vector<double> transposed(const std::vector<double>& v, int32_t rows, int32_t cols)
    {
        std::vector<double> t(v.size());
        for(int32_t j = 0; j < cols; ++j)
            for(int32_t i = 0; i < rows; ++i)
                t[static_cast<size_t>(i) * cols + j] = v[static_cast<size_t>(j) * rows + i];
        return t;
    }

    inline csr_matrix<double> with_zero_diagonal(csr_matrix<double> a, int32_t row)
    {
        for(int32_t p = a.ptr[row]; p < a.ptr[row + 1]; ++p)
            if(a.ind[p] == row)
                a.val[p] = 0.0;
        return a;
    }

    // Solves with the unit lower triangle of `lu` (on the pattern of `a`), then with its
    // upper triangle.
    inline std::vector<double> host_lu_solve(const csr_matrix<double>&  a,
                                             const std::vector<double>& lu,
                                             std::vector<double>        x)
    {
        for(int32_t i = 0; i < a.n; ++i)
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                if(a.ind[p] < i)
                    x[i] -= lu[p] * x[a.ind[p]];
        for(int32_t i = a.n - 1; i >= 0; --i)
        {
            double diagonal = 1.0;
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
            {
                if(a.ind[p] > i)
                    x[i] -= lu[p] * x[a.ind[p]];
                else if(a.ind[p] == i)
                    diagonal = lu[p];
            }
            x[i] /= diagonal;
        }
        return x;
    }

    // Solves with the lower triangle L of `l` (on the pattern of `a`), then with L^T.
    inline std::vector<double> host_llt_solve(const csr_matrix<double>&  a,
                                              const std::vector<double>& l,
                                              std::vector<double>        x)
    {
        std::vector<int32_t> diag(static_cast<size_t>(a.n), -1);
        for(int32_t i = 0; i < a.n; ++i)
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                if(a.ind[p] == i)
                    diag[i] = p;
        for(int32_t i = 0; i < a.n; ++i)
        {
            for(int32_t p = a.ptr[i]; p < diag[i]; ++p)
                x[i] -= l[p] * x[a.ind[p]];
            x[i] /= l[diag[i]];
        }
        for(int32_t i = a.n - 1; i >= 0; --i)
        {
            x[i] /= l[diag[i]];
            for(int32_t p = a.ptr[i]; p < diag[i]; ++p)
                x[a.ind[p]] -= l[p] * x[i];
        }
        return x;
    }

    // Solves with the lower triangle of `a` for the `nrhs` columns of the column major
    // `b`; with a unit diagonal the stored diagonal is ignored.
    inline std::vector<double> host_lower_solve(const csr_matrix<double>& a,
                                                std::vector<double>       b,
                                                int32_t                   nrhs,
                                                rocsparse_diag_type       diag
                                                = rocsparse_diag_type_non_unit)
    {
        for(int32_t j = 0; j < nrhs; ++j)
        {
            double* x = b.data() + static_cast<size_t>(j) * a.n;
            for(int32_t i = 0; i < a.n; ++i)
            {
                double diagonal = 1.0;
                for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                {
                    if(a.ind[p] < i)
                        x[i] -= a.val[p] * x[a.ind[p]];
                    else if(a.ind[p] == i && diag == rocsparse_diag_type_non_unit)
                        diagonal = a.val[p];
                }
                x[i] /= diagonal;
            }
        }
        return b;
    }

    // Solves with the transpose of the lower triangle of `a` (non unit diagonal) for the
    // `nrhs` columns of the column major `b`.
    inline std::vector<double> host_lower_transposed_solve(const csr_matrix<double>& a,
                                                           std::vector<double>       b,
                                                           int32_t                   nrhs)
    {
        for(int32_t j = 0; j < nrhs; ++j)
        {
            double* x = b.data() + static_cast<size_t>(j) * a.n;
            for(int32_t i = a.n - 1; i >= 0; --i)
            {
                for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                    if(a.ind[p] == i)
                        x[i] /= a.val[p];
                for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                    if(a.ind[p] < i)
                        x[a.ind[p]] -= a.val[p] * x[i];
            }
        }
        return b;
    }

    inline rocsparse_mat_descr
        make_descr(rocsparse_index_base base, rocsparse_fill_mode fill, rocsparse_diag_type diag)
    {
        rocsparse_mat_descr descr = nullptr;
        EXPECT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        EXPECT_EQ(rocsparse_set_mat_index_base(descr, base), rocsparse_status_success);
        EXPECT_EQ(rocsparse_set_mat_fill_mode(descr, fill), rocsparse_status_success);
        EXPECT_EQ(rocsparse_set_mat_diag_type(descr, diag), rocsparse_status_success);
        return descr;
    }

    // Runs `call` with the position in host memory, or in device memory when
    // `device_query` is set, and returns the position.
    template <typename F>
    rocsparse_int query_position(rocsparse_handle handle, bool device_query, F&& call)
    {
        rocsparse_int position = -7;
        if(device_query)
        {
            device_vector<rocsparse_int> d(std::vector<rocsparse_int>{-7});
            EXPECT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
                      rocsparse_status_success);
            (void)call(d.ptr);
            EXPECT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
                      rocsparse_status_success);
            position = to_host(d)[0];
        }
        else
        {
            (void)call(&position);
        }
        return position;
    }

    // The zero based row of a position in `base` among `rows` rows.
    inline int32_t row_of(rocsparse_int position, rocsparse_index_base base, int32_t rows)
    {
        if(position == -1)
            return no_pivot;
        for(int32_t r = 0; r < rows; ++r)
            if(position == position_in_base(r, base))
                return r;
        ADD_FAILURE() << "pivot position " << position << " is not a row in " << name(base);
        return invalid_report;
    }

    // The zero based row of a reported zero pivot, after checking that the status and
    // the position agree.
    template <typename F>
    int32_t zero_pivot_row(rocsparse_handle     handle,
                           rocsparse_index_base base,
                           int32_t              rows,
                           bool                 device_query,
                           F&&                  call)
    {
        rocsparse_status status   = rocsparse_status_internal_error;
        const auto       position = query_position(handle, device_query, [&](rocsparse_int* p) {
            status = call(p);
            return status;
        });
        if(status == rocsparse_status_success)
        {
            EXPECT_EQ(position, -1) << "no zero pivot reported, but a position is set";
            return (position == -1) ? no_pivot : invalid_report;
        }
        if(status == rocsparse_status_zero_pivot)
        {
            EXPECT_NE(position, -1) << "zero pivot reported without a position";
            return row_of(position, base, rows);
        }
        ADD_FAILURE() << "zero pivot query returned " << describe(status);
        return invalid_report;
    }
}
