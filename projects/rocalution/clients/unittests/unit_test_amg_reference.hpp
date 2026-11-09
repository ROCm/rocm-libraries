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

// Test matrices and reference definitions of the AMG setup: strength of connection,
// valid aggregations and C/F splittings, and the smoothed aggregation prolongation

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "unit_test_types.hpp"

namespace rocalution_unit
{
    // 5-point stencil on an nx x ny grid with weights wx and wy. With dirichlet the
    // diagonal is 2 (wx + wy), otherwise it is the sum of the neighbour weights and the
    // row sums are zero.
    template <typename T = double>
    Csr<T> grid(int nx, int ny, double wx, double wy, bool dirichlet)
    {
        Csr<T> A(nx * ny);
        for(int y = 0; y < ny; ++y)
        {
            for(int x = 0; x < nx; ++x)
            {
                std::map<int, double> row;
                double                diagonal = dirichlet ? 2 * (wx + wy) : 0.0;
                auto                  link     = [&](int xx, int yy, double w) {
                    if(xx >= 0 && xx < nx && yy >= 0 && yy < ny)
                    {
                        row[yy * nx + xx] = -w;
                        diagonal += dirichlet ? 0.0 : w;
                    }
                };
                link(x - 1, y, wx);
                link(x + 1, y, wx);
                link(x, y - 1, wy);
                link(x, y + 1, wy);
                row[y * nx + x] = diagonal;
                for(const auto& entry : row)
                {
                    A.add(entry.first, static_cast<T>(entry.second));
                }
                A.end_row();
            }
        }
        return A;
    }

    // Band of the given half bandwidth with couplings -1 and zero row sums. With fan,
    // row 0 couples to every node, so its row of a prolongation has an entry for every
    // aggregate.
    template <typename T = double>
    Csr<T> band(int n, int bandwidth, bool fan)
    {
        Csr<T> A(n);
        for(int i = 0; i < n; ++i)
        {
            std::vector<int> neighbours;
            for(int j = 0; j < n; ++j)
            {
                if(j != i && (std::abs(i - j) <= bandwidth || (fan && i == 0)))
                {
                    neighbours.push_back(j);
                }
            }
            for(int j = 0; j < n; ++j)
            {
                if(j == i)
                {
                    A.add(j, static_cast<T>(neighbours.size()));
                }
                else if(std::binary_search(neighbours.begin(), neighbours.end(), j))
                {
                    A.add(j, static_cast<T>(-1));
                }
            }
            A.end_row();
        }
        return A;
    }

    template <typename T>
    T entry(const Csr<T>& A, int i, int j)
    {
        for(PtrType k = A.row_offset[i]; k < A.row_offset[i + 1]; ++k)
        {
            if(A.col[k] == j)
            {
                return A.val[k];
            }
        }
        return static_cast<T>(0);
    }

    template <typename T>
    int max_row_nnz(const Csr<T>& A)
    {
        PtrType longest = 0;
        for(int i = 0; i < A.n; ++i)
        {
            longest = std::max(longest, A.row_offset[i + 1] - A.row_offset[i]);
        }
        return static_cast<int>(longest);
    }

    // Every off-diagonal entry, as the connections or strong influences
    template <typename T>
    std::vector<bool> off_diagonal(const Csr<T>& A)
    {
        std::vector<bool> strong(A.nnz());
        for(int i = 0; i < A.n; ++i)
        {
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                strong[j] = A.col[j] != i;
            }
        }
        return strong;
    }

    // SA: |a_ij| > eps sqrt(a_ii a_jj), never the diagonal
    template <typename T>
    std::vector<bool> sa_strength(const Csr<T>& A, double eps)
    {
        const T        e = static_cast<T>(eps);
        std::vector<T> diagonal(A.n);
        for(int i = 0; i < A.n; ++i)
        {
            diagonal[i] = entry(A, i, i);
        }

        std::vector<bool> strong(A.nnz());
        for(int i = 0; i < A.n; ++i)
        {
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                int c     = A.col[j];
                strong[j] = c != i && A.val[j] * A.val[j] > e * e * diagonal[i] * diagonal[c];
            }
        }
        return strong;
    }

    // RS: j influences i if a_ij < eps min_k a_ik, for a positive diagonal
    template <typename T>
    std::vector<bool> rs_strength(const Csr<T>& A, double eps)
    {
        const T           e = static_cast<T>(eps);
        std::vector<bool> strong(A.nnz());
        for(int i = 0; i < A.n; ++i)
        {
            T minimum = static_cast<T>(0);
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                if(A.col[j] != i)
                {
                    minimum = std::min(minimum, A.val[j]);
                }
            }
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                strong[j] = A.col[j] != i && A.val[j] < e * minimum;
            }
        }
        return strong;
    }

    // Every node belongs to an aggregate, aggregate k has root number k in index
    // order, and the other nodes are strongly connected to their aggregate
    template <typename T>
    void expect_valid_aggregation(const Csr<T>&               A,
                                  const std::vector<bool>&    strong,
                                  const std::vector<int64_t>& aggregates,
                                  const std::vector<int64_t>& roots)
    {
        ASSERT_EQ(aggregates.size(), static_cast<size_t>(A.n));
        ASSERT_EQ(roots.size(), static_cast<size_t>(A.n));

        std::vector<int> root_list;
        for(int i = 0; i < A.n; ++i)
        {
            ASSERT_GE(roots[i], 0) << "node " << i;
            ASSERT_LT(roots[i], A.n) << "node " << i;
            EXPECT_EQ(roots[roots[i]], roots[i]) << "node " << i;
            EXPECT_EQ(aggregates[i], aggregates[roots[i]]) << "node " << i;
            if(roots[i] == i)
            {
                root_list.push_back(i);
            }
        }
        for(size_t k = 0; k < root_list.size(); ++k)
        {
            EXPECT_EQ(aggregates[root_list[k]], static_cast<int64_t>(k));
        }

        for(int i = 0; i < A.n; ++i)
        {
            if(roots[i] == i)
            {
                continue;
            }
            bool connected = false;
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                connected |= strong[j] && aggregates[A.col[j]] == aggregates[i];
            }
            EXPECT_TRUE(connected) << "node " << i;
        }
    }

    // No two roots are strongly connected. The second PMIS sweep only visits the
    // neighbours of the distance 1 maximum, so roots at distance 2 can occur.
    template <typename T>
    void expect_independent_roots(const Csr<T>&               A,
                                  const std::vector<bool>&    strong,
                                  const std::vector<int64_t>& roots)
    {
        for(int i = 0; i < A.n; ++i)
        {
            if(roots[i] != i)
            {
                continue;
            }
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                const int c = A.col[j];
                EXPECT_FALSE(strong[j] && c != i && roots[c] == c) << "roots " << i << " and " << c;
            }
        }
    }

    // C points (1) are not strongly connected, F points (2) depend on a C point
    template <typename T>
    void expect_valid_splitting(const Csr<T>&            A,
                                const std::vector<bool>& S,
                                const std::vector<int>&  cf)
    {
        ASSERT_EQ(cf.size(), static_cast<size_t>(A.n));
        for(int i = 0; i < A.n; ++i)
        {
            ASSERT_TRUE(cf[i] == 1 || cf[i] == 2) << "node " << i << ": " << cf[i];
            bool has_strong = false;
            bool coarse     = false;
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                if(S[j])
                {
                    has_strong = true;
                    coarse |= cf[A.col[j]] == 1;
                    EXPECT_FALSE(cf[i] == 1 && cf[A.col[j]] == 1)
                        << "C points " << i << " and " << A.col[j];
                }
            }
            if(cf[i] == 2 && has_strong)
            {
                EXPECT_TRUE(coarse) << "F point " << i;
            }
        }
    }

    // C point i interpolates itself, from its index among the C points. For matrices
    // with zero row sums the constants are interpolated exactly.
    template <typename T>
    void expect_interpolation(const Csr<T>& P, const std::vector<int>& cf)
    {
        ASSERT_EQ(cf.size(), static_cast<size_t>(P.n));
        int nc = 0;
        for(int i = 0; i < P.n; ++i)
        {
            if(cf[i] == 1)
            {
                ASSERT_EQ(P.row_offset[i + 1] - P.row_offset[i], 1) << "C point " << i;
                EXPECT_EQ(P.col[P.row_offset[i]], nc) << "C point " << i;
                EXPECT_EQ(P.val[P.row_offset[i]], static_cast<T>(1)) << "C point " << i;
                ++nc;
            }
        }
        EXPECT_EQ(P.m, nc);

        // A sum of k weights has a rounding error of up to (k - 1) u
        std::vector<T> ones = P.apply(std::vector<T>(P.m, static_cast<T>(1)));
        for(int i = 0; i < P.n; ++i)
        {
            const PtrType k     = P.row_offset[i + 1] - P.row_offset[i];
            const double  bound = static_cast<double>(std::max<PtrType>(k - 1, 0))
                                 * std::numeric_limits<T>::epsilon() / 2;
            EXPECT_TRUE(is_near(ones[i], static_cast<T>(1), std::max(tol<T>(), bound)))
                << "row " << i;
        }
    }

    // P = (I - relax D^-1 A_F) P_tent, where A_F drops the weak connections and adds
    // (lumping 0) or subtracts (lumping 1) them on the diagonal
    template <typename T>
    Csr<T> smoothed_prolongation(const Csr<T>&               A,
                                 const std::vector<bool>&    strong,
                                 const std::vector<int64_t>& agg,
                                 double                      relax,
                                 int                         lumping)
    {
        const T   w  = static_cast<T>(relax);
        const int nc = static_cast<int>(*std::max_element(agg.begin(), agg.end()) + 1);
        Csr<T>    P(A.n, nc);
        for(int i = 0; i < A.n; ++i)
        {
            T dia = static_cast<T>(0);
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                if(A.col[j] == i)
                {
                    dia += A.val[j];
                }
                else if(!strong[j])
                {
                    dia += lumping == 0 ? A.val[j] : -A.val[j];
                }
            }

            std::map<int, T> row;
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                int c = A.col[j];
                if(c == i)
                {
                    row[static_cast<int>(agg[c])] += static_cast<T>(1) - w;
                }
                else if(strong[j])
                {
                    row[static_cast<int>(agg[c])] -= w * A.val[j] / dia;
                }
            }
            for(const auto& e : row)
            {
                P.add(e.first, e.second);
            }
            P.end_row();
        }
        return P;
    }

    // Index of entry (i, j) in A
    template <typename T>
    PtrType position(const Csr<T>& A, int i, int j)
    {
        for(PtrType k = A.row_offset[i]; k < A.row_offset[i + 1]; ++k)
        {
            if(A.col[k] == j)
            {
                return k;
            }
        }
        ADD_FAILURE() << "no entry (" << i << ", " << j << ")";
        return 0;
    }

    // Per-entry flags of A (a strength or connection pattern), set at the given entries
    template <typename T>
    std::vector<bool> flags_at(const Csr<T>& A, const std::vector<std::pair<int, int>>& entries)
    {
        std::vector<bool> f(A.nnz(), false);
        for(const auto& e : entries)
        {
            f[position(A, e.first, e.second)] = true;
        }
        return f;
    }
}
