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

// Reference computations on Csr matrices and dense row-major copies of them

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "unit_test_common.hpp"

namespace rocalution_unit
{
    // Row major. Duplicate entries are summed, as in a matrix-vector product.
    template <typename T>
    std::vector<T> dense(const Csr<T>& A)
    {
        std::vector<T> D(static_cast<size_t>(A.n) * A.m, static_cast<T>(0));
        for(int i = 0; i < A.n; ++i)
        {
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                D[static_cast<size_t>(i) * A.m + A.col[j]] += A.val[j];
            }
        }
        return D;
    }

    // Sorted CSR of the non-zeros of a row-major dense n x m matrix
    template <typename T>
    Csr<T> from_dense(int n, int m, const std::vector<T>& D)
    {
        Csr<T> A(n, m);
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < m; ++j)
            {
                if(D[static_cast<size_t>(i) * m + j] != static_cast<T>(0))
                {
                    A.add(j, D[static_cast<size_t>(i) * m + j]);
                }
            }
            A.end_row();
        }
        return A;
    }

    // Row-major product of the n x k matrix A and the k x m matrix B
    template <typename T>
    std::vector<T>
        dense_product(const std::vector<T>& A, const std::vector<T>& B, int n, int k, int m)
    {
        std::vector<T> C(static_cast<size_t>(n) * m, static_cast<T>(0));
        for(int i = 0; i < n; ++i)
        {
            for(int l = 0; l < k; ++l)
            {
                for(int j = 0; j < m; ++j)
                {
                    C[static_cast<size_t>(i) * m + j]
                        += A[static_cast<size_t>(i) * k + l] * B[static_cast<size_t>(l) * m + j];
                }
            }
        }
        return C;
    }

    template <typename T>
    Csr<T> transposed(const Csr<T>& A)
    {
        std::vector<T> D = dense(A);
        std::vector<T> D_t(D.size());
        for(int i = 0; i < A.n; ++i)
        {
            for(int j = 0; j < A.m; ++j)
            {
                D_t[static_cast<size_t>(j) * A.n + i] = D[static_cast<size_t>(i) * A.m + j];
            }
        }
        return from_dense(A.m, A.n, D_t);
    }

    // The entries for which keep(i, j, value) holds
    template <typename T, typename Keep>
    Csr<T> filtered(const Csr<T>& A, Keep keep)
    {
        Csr<T> B(A.n, A.m);
        for(int i = 0; i < A.n; ++i)
        {
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                if(keep(i, A.col[j], A.val[j]))
                {
                    B.add(A.col[j], A.val[j]);
                }
            }
            B.end_row();
        }
        return B;
    }

    // The same pattern, with the values given by f(i, j, value)
    template <typename T, typename F>
    Csr<T> transformed(Csr<T> A, F f)
    {
        for(int i = 0; i < A.n; ++i)
        {
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                A.val[j] = f(i, A.col[j], A.val[j]);
            }
        }
        return A;
    }

    // Row and column i of the square matrix A move to perm[i]. Explicit zeros stay.
    template <typename T>
    Csr<T> permuted(const Csr<T>& A, const std::vector<int>& perm)
    {
        std::vector<T>    D(static_cast<size_t>(A.n) * A.n, static_cast<T>(0));
        std::vector<bool> P(D.size(), false);
        for(int i = 0; i < A.n; ++i)
        {
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                D[static_cast<size_t>(perm[i]) * A.n + perm[A.col[j]]] = A.val[j];
                P[static_cast<size_t>(perm[i]) * A.n + perm[A.col[j]]] = true;
            }
        }

        Csr<T> B(A.n);
        for(int i = 0; i < A.n; ++i)
        {
            for(int j = 0; j < A.n; ++j)
            {
                if(P[static_cast<size_t>(i) * A.n + j])
                {
                    B.add(j, D[static_cast<size_t>(i) * A.n + j]);
                }
            }
            B.end_row();
        }
        return B;
    }

    template <typename T = double>
    std::vector<T> rhs(int n)
    {
        std::vector<T> b(n);
        for(int i = 0; i < n; ++i)
        {
            b[i] = static_cast<T>(i % 4 - 1.5);
        }
        return b;
    }

    // Relative to the reference for magnitudes above 1, absolute below
    template <typename T>
    void expect_near(const std::vector<T>& a, const std::vector<T>& ref, double tol)
    {
        ASSERT_EQ(a.size(), ref.size());
        for(size_t i = 0; i < a.size(); ++i)
        {
            EXPECT_NEAR(a[i], ref[i], tol * std::max(1.0, std::abs(static_cast<double>(ref[i]))))
                << "at index " << i;
        }
    }

    // The same pattern, and values as in the vector version
    template <typename T>
    void expect_near(const Csr<T>& A, const Csr<T>& ref, double tol)
    {
        ASSERT_EQ(A.n, ref.n);
        ASSERT_EQ(A.m, ref.m);
        EXPECT_EQ(A.row_offset, ref.row_offset);
        EXPECT_EQ(A.col, ref.col);
        expect_near(A.val, ref.val, tol);
    }

    // b solved with the lower triangle of the dense n x n matrix D, with a unit
    // diagonal if unit
    template <typename T>
    std::vector<T> lower_solve(const std::vector<T>& D, int n, std::vector<T> b, bool unit)
    {
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < i; ++j)
            {
                b[i] -= D[i * n + j] * b[j];
            }
            if(!unit)
            {
                b[i] /= D[i * n + i];
            }
        }
        return b;
    }

    // b solved with the upper triangle of the dense n x n matrix D, with a unit
    // diagonal if unit
    template <typename T>
    std::vector<T> upper_solve(const std::vector<T>& D, int n, std::vector<T> b, bool unit)
    {
        for(int i = n - 1; i >= 0; --i)
        {
            for(int j = i + 1; j < n; ++j)
            {
                b[i] -= D[i * n + j] * b[j];
            }
            if(!unit)
            {
                b[i] /= D[i * n + i];
            }
        }
        return b;
    }

    // The largest |j - i| of the entries
    template <typename T>
    int bandwidth(const Csr<T>& A)
    {
        int b = 0;
        for(int i = 0; i < A.n; ++i)
        {
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                b = std::max(b, std::abs(A.col[j] - i));
            }
        }
        return b;
    }

    // Whether each entry of the dense n x m matrix is stored
    template <typename T>
    std::vector<bool> pattern(const Csr<T>& A)
    {
        std::vector<bool> P(static_cast<size_t>(A.n) * A.m);
        for(int i = 0; i < A.n; ++i)
        {
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                P[static_cast<size_t>(i) * A.m + A.col[j]] = true;
            }
        }
        return P;
    }

    // ILU(0) on the stored pattern of the square A (explicit zeros included), as one
    // dense matrix: L with unit diagonal below the diagonal, U on and above it.
    // Every pivot must be nonzero.
    template <typename T>
    std::vector<T> ilu0_dense(const Csr<T>& A)
    {
        const int               n = A.n;
        std::vector<T>          D = dense(A);
        const std::vector<bool> P = pattern(A);
        for(int i = 0; i < n; ++i)
        {
            for(int k = 0; k < i; ++k)
            {
                if(!P[i * n + k])
                {
                    continue;
                }
                D[i * n + k] /= D[k * n + k];
                for(int j = k + 1; j < n; ++j)
                {
                    if(P[i * n + j])
                    {
                        D[i * n + j] -= D[i * n + k] * D[k * n + j];
                    }
                }
            }
        }
        return D;
    }

    template <typename T>
    T conj_entry(const T& v)
    {
        return v;
    }

    template <typename R>
    std::complex<R> conj_entry(const std::complex<R>& v)
    {
        return std::conj(v);
    }

    // IC(0) of the Hermitian positive definite A on the stored lower pattern: the
    // dense lower factor L of L L^H. A - sum |l_ik|^2 must stay positive.
    template <typename T>
    std::vector<T> ic0_dense(const Csr<T>& A)
    {
        const int               n = A.n;
        const std::vector<T>    D = dense(A);
        const std::vector<bool> P = pattern(A);
        std::vector<T>          L(D.size(), static_cast<T>(0));
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j <= i; ++j)
            {
                if(!P[i * n + j])
                {
                    continue;
                }
                T sum = D[i * n + j];
                for(int k = 0; k < j; ++k)
                {
                    sum -= L[i * n + k] * conj_entry(L[j * n + k]);
                }
                L[i * n + j] = i == j ? std::sqrt(sum) : sum / L[j * n + j];
            }
        }
        return L;
    }
}
