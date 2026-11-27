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

// The block-triangular system shared by the wide bsrsv kernel tests and the
// bsrsv analysis reuse tests: an 8-by-8 block bidiagonal matrix of 2-by-2
// blocks. The diagonal blocks are triangular with diagonal 2 and 4 and an
// in-block entry of 1, and hold 37 in the triangle the solve must not read.
// With alpha = -2 and a varying x, the diagonal division, the in-block
// elimination, alpha and the diagonal type all change the result, which a host
// solve in double checks.

#pragma once

#include "unit_test_utils.hpp"

#include "rocsparse.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace bsrsv_test
{
    using test_T = float;

    constexpr int32_t kMb       = 8;
    constexpr int32_t kBlockDim = 2;
    constexpr int32_t kRows     = kMb * kBlockDim;
    constexpr test_T  kAlpha    = -2;

    // Logical 2-by-2 blocks, stored in the order dir gives.
    using block2              = test_T[2][2];
    constexpr block2 kOffDiag = {{-1, 0}, {0, -1}};
    // kOffAsym is not symmetric, so a kernel that reads it with the wrong
    // direction solves with kOffAsymT instead.
    constexpr block2 kOffAsym  = {{-1, 0}, {1, -1}};
    constexpr block2 kOffAsymT = {{-1, 1}, {0, -1}};

    inline void append_block(std::vector<test_T>& val, const block2& b, rocsparse_direction dir)
    {
        for(int32_t i = 0; i < kBlockDim; ++i)
        {
            for(int32_t j = 0; j < kBlockDim; ++j)
            {
                val.push_back(dir == rocsparse_direction_row ? b[i][j] : b[j][i]);
            }
        }
    }

    // Diagonal block of a lower or upper matrix. A singular block has a zero
    // diagonal, which the kernels report as a zero pivot.
    inline void append_diag_block(std::vector<test_T>& val,
                                  bool                 lower,
                                  bool                 singular,
                                  rocsparse_direction  dir)
    {
        const test_T d0 = singular ? 0 : 2;
        const test_T d1 = singular ? 0 : 4;
        const block2 lower_block{{d0, 37}, {1, d1}};
        const block2 upper_block{{d0, 1}, {37, d1}};
        append_block(val, lower ? lower_block : upper_block, dir);
    }

    template <typename I, typename J>
    struct test_matrix
    {
        bool                 lower = true;
        rocsparse_index_base base  = rocsparse_index_base_zero;
        rocsparse_direction  dir   = rocsparse_direction_row;
        std::vector<I>       row_ptr;
        std::vector<J>       col_ind;
        std::vector<test_T>  val;
        // Block rows in solve order, as the analysis builds it.
        std::vector<J> map;

        I nnzb() const
        {
            return static_cast<I>(col_ind.size());
        }
    };

    template <typename I, typename J>
    test_matrix<I, J> build_matrix(bool                 lower,
                                   rocsparse_index_base base         = rocsparse_index_base_zero,
                                   int32_t              singular_row = -1,
                                   const block2&        off          = kOffDiag,
                                   rocsparse_direction  dir          = rocsparse_direction_row)
    {
        test_matrix<I, J> a;
        a.lower = lower;
        a.base  = base;
        a.dir   = dir;
        a.row_ptr.assign(1, static_cast<I>(0));
        a.map.resize(static_cast<size_t>(kMb));

        for(int32_t row = 0; row < kMb; ++row)
        {
            if(lower && row > 0)
            {
                a.col_ind.push_back(static_cast<J>(row - 1));
                append_block(a.val, off, dir);
            }
            a.col_ind.push_back(static_cast<J>(row));
            append_diag_block(a.val, lower, row == singular_row, dir);
            if(!lower && row + 1 < kMb)
            {
                a.col_ind.push_back(static_cast<J>(row + 1));
                append_block(a.val, off, dir);
            }
            a.map[static_cast<size_t>(row)] = static_cast<J>(lower ? row : kMb - 1 - row);
            a.row_ptr.push_back(static_cast<I>(a.col_ind.size()));
        }

        if(base == rocsparse_index_base_one)
        {
            for(auto& r : a.row_ptr)
            {
                r = static_cast<I>(r + 1);
            }
            for(auto& c : a.col_ind)
            {
                c = static_cast<J>(c + 1);
            }
        }
        return a;
    }

    inline std::vector<test_T> make_x()
    {
        std::vector<test_T> x(static_cast<size_t>(kRows));
        for(int32_t r = 0; r < kRows; ++r)
        {
            x[static_cast<size_t>(r)] = static_cast<test_T>(1 + r % 3);
        }
        return x;
    }

    // Host solve of op(A) y = kAlpha x in double, reading only the triangle of A
    // given by lower. The 37s in the other triangle of the diagonal blocks are
    // ignored, as the solve must ignore them.
    template <typename I, typename J>
    std::vector<double> reference(const test_matrix<I, J>& a,
                                  bool                     lower,
                                  rocsparse_operation      trans,
                                  rocsparse_diag_type      diag)
    {
        constexpr size_t          blk  = static_cast<size_t>(kBlockDim) * kBlockDim;
        const std::vector<test_T> x    = make_x();
        const int64_t             base = static_cast<int64_t>(a.base);

        std::vector<double> op(static_cast<size_t>(kRows) * kRows, 0.0);
        for(int64_t br = 0; br < kMb; ++br)
        {
            for(int64_t k = a.row_ptr[br] - base; k < a.row_ptr[br + 1] - base; ++k)
            {
                const int64_t bc = static_cast<int64_t>(a.col_ind[k]) - base;
                for(int64_t i = 0; i < kBlockDim; ++i)
                {
                    for(int64_t j = 0; j < kBlockDim; ++j)
                    {
                        const int64_t r = br * kBlockDim + i;
                        const int64_t c = bc * kBlockDim + j;
                        if(lower ? c > r : c < r)
                        {
                            continue;
                        }
                        const int64_t pos = (a.dir == rocsparse_direction_row) ? i * kBlockDim + j
                                                                               : j * kBlockDim + i;
                        const double  v
                            = a.val[blk * static_cast<size_t>(k) + static_cast<size_t>(pos)];
                        if(trans == rocsparse_operation_none)
                        {
                            op[r * kRows + c] = v;
                        }
                        else
                        {
                            op[c * kRows + r] = v;
                        }
                    }
                }
            }
        }

        const bool          op_lower = (lower == (trans == rocsparse_operation_none));
        std::vector<double> y(static_cast<size_t>(kRows), 0.0);
        for(int32_t s = 0; s < kRows; ++s)
        {
            const int32_t r   = op_lower ? s : kRows - 1 - s;
            double        sum = static_cast<double>(kAlpha) * x[static_cast<size_t>(r)];
            for(int32_t c = 0; c < kRows; ++c)
            {
                if(c != r)
                {
                    sum -= op[static_cast<size_t>(r) * kRows + c] * y[static_cast<size_t>(c)];
                }
            }
            y[static_cast<size_t>(r)] = (diag == rocsparse_diag_type_unit)
                                            ? sum
                                            : sum / op[static_cast<size_t>(r) * kRows + r];
        }
        return y;
    }

    inline void expect_solution(const std::vector<test_T>& y, const std::vector<double>& ref)
    {
        ASSERT_EQ(y.size(), ref.size());
        for(size_t row = 0; row < ref.size(); ++row)
        {
            EXPECT_NEAR(y[row], ref[row], 1e-5 * std::max(1.0, std::abs(ref[row])))
                << "row " << row;
        }
    }
}
