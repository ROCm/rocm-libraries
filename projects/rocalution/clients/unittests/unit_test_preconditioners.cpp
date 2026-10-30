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

// The preconditioner classes, driven through LocalMatrix and LocalVector. One
// Solve of a built preconditioner is compared with z = M^-1 r of a dense
// reference of its definition, on small matrices with small integer values.

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "unit_test_common.hpp"
#include "unit_test_local.hpp"
#include "unit_test_reference.hpp"
#include "unit_test_types.hpp"

#include "base/local_matrix.hpp"
#include "base/local_vector.hpp"
#include "solvers/direct/inversion.hpp"
#include "solvers/direct/lu.hpp"
#include "solvers/krylov/gmres.hpp"
#include "solvers/preconditioners/preconditioner.hpp"
#include "solvers/preconditioners/preconditioner_ai.hpp"
#include "solvers/preconditioners/preconditioner_as.hpp"
#include "solvers/preconditioners/preconditioner_blockprecond.hpp"
#include "solvers/preconditioners/preconditioner_multicolored_gs.hpp"
#include "solvers/preconditioners/preconditioner_multicolored_ilu.hpp"
#include "solvers/preconditioners/preconditioner_multielimination.hpp"
#include "solvers/preconditioners/preconditioner_saddlepoint.hpp"
#include "utils/allocate_free.hpp"

using namespace rocalution_unit;
using rocalution::LocalMatrix;
using rocalution::LocalVector;

template <typename T>
using Matrix = LocalMatrix<T>;
template <typename T>
using Vector = LocalVector<T>;
template <typename T>
using Solver = rocalution::Solver<Matrix<T>, Vector<T>, T>;

namespace
{
    // One Solve of a built preconditioner, into a vector holding initial
    template <typename T>
    std::vector<T>
        apply(Solver<T>& p, const Matrix<T>& A, const std::vector<T>& r, double initial = 0.0)
    {
        auto in  = make_local_vector(A, r);
        auto out = make_local_vector(A, std::vector<T>(r.size(), static_cast<T>(initial)));
        p.Solve(*in, out.get());
        return values_of(*out);
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
        return 0;
    }

    // Nonsymmetric and diagonally dominant: tridiagonal, and an entry three
    // columns to the right of the diagonal, wrapped around, in every even row
    template <typename T>
    Csr<T> nonsymmetric(int n)
    {
        Csr<T> A(n);
        for(int i = 0; i < n; ++i)
        {
            std::vector<int> cols;
            for(int j = std::max(0, i - 1); j <= std::min(n - 1, i + 1); ++j)
            {
                cols.push_back(j);
            }
            if(i % 2 == 0)
            {
                cols.push_back((i + 3) % n);
            }
            std::sort(cols.begin(), cols.end());
            cols.erase(std::unique(cols.begin(), cols.end()), cols.end());

            for(int j : cols)
            {
                A.add(j,
                      static_cast<T>(j == i ? 8.0 + i % 3 : (j < i ? -1.0 : -2.0) - (i + j) % 2));
            }
            A.end_row();
        }
        return A;
    }

    // Symmetric positive definite by diagonal dominance, with the pattern of
    // nonsymmetric(n) and its transpose
    template <typename T>
    Csr<T> spd(int n)
    {
        std::vector<T> D = dense(nonsymmetric<T>(n));
        std::vector<T> S(D.size(), static_cast<T>(0));
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                if(i != j && (D[i * n + j] != T(0) || D[j * n + i] != T(0)))
                {
                    S[i * n + j] = static_cast<T>(-1.0 - (i + j) % 2);
                }
            }
            S[i * n + i] = static_cast<T>(10.0 + i % 3);
        }
        return from_dense(n, n, S);
    }

    // x = M^-1 b for the dense n x n matrix M, by Gaussian elimination with
    // partial pivoting
    template <typename T>
    std::vector<T> dense_solve(std::vector<T> M, int n, std::vector<T> b)
    {
        for(int k = 0; k < n; ++k)
        {
            int p = k;
            for(int i = k + 1; i < n; ++i)
            {
                if(std::abs(M[i * n + k]) > std::abs(M[p * n + k]))
                {
                    p = i;
                }
            }
            for(int j = 0; j < n; ++j)
            {
                std::swap(M[k * n + j], M[p * n + j]);
            }
            std::swap(b[k], b[p]);

            for(int i = k + 1; i < n; ++i)
            {
                T f = M[i * n + k] / M[k * n + k];
                for(int j = k; j < n; ++j)
                {
                    M[i * n + j] -= f * M[k * n + j];
                }
                b[i] -= f * b[k];
            }
        }
        return upper_solve(M, n, b, false);
    }

    template <typename T>
    std::vector<T> diagonal(const std::vector<T>& D, int n)
    {
        std::vector<T> d(n);
        for(int i = 0; i < n; ++i)
        {
            d[i] = D[i * n + i];
        }
        return d;
    }

    template <typename T>
    std::vector<T> pointwise(const std::vector<T>& a, const std::vector<T>& b)
    {
        std::vector<T> c(a.size());
        for(size_t i = 0; i < a.size(); ++i)
        {
            c[i] = a[i] * b[i];
        }
        return c;
    }

    template <typename T>
    std::vector<T> inverted(std::vector<T> d)
    {
        for(T& v : d)
        {
            v = 1 / v;
        }
        return d;
    }

    template <typename T>
    std::vector<T> dense_apply(const std::vector<T>& M, int n, const std::vector<T>& x)
    {
        std::vector<T> y(n, static_cast<T>(0));
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                y[i] += M[i * n + j] * x[j];
            }
        }
        return y;
    }

    template <typename T>
    std::vector<T> dense_transpose(const std::vector<T>& M, int n)
    {
        std::vector<T> M_t(M.size());
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                M_t[j * n + i] = M[i * n + j];
            }
        }
        return M_t;
    }

    template <typename T>
    std::vector<T> jacobi(const Csr<T>& A, const std::vector<T>& r)
    {
        return pointwise(inverted(diagonal(dense(A), A.n)), r);
    }

    // (D + L)^-1 r
    template <typename T>
    std::vector<T> gauss_seidel(const Csr<T>& A, const std::vector<T>& r)
    {
        return lower_solve(dense(A), A.n, r, false);
    }

    // (D + U)^-1 r
    template <typename T>
    std::vector<T> backward_gauss_seidel(const Csr<T>& A, const std::vector<T>& r)
    {
        return upper_solve(dense(A), A.n, r, false);
    }

    // (D + U)^-1 D (D + L)^-1 r
    template <typename T>
    std::vector<T> symmetric_gauss_seidel(const Csr<T>& A, const std::vector<T>& r)
    {
        const std::vector<T> D = dense(A);
        return upper_solve(
            D, A.n, pointwise(diagonal(D, A.n), lower_solve(D, A.n, r, false)), false);
    }

    // (L U)^-1 r for the factors stored in place: strict lower part of L with a
    // unit diagonal, and U
    template <typename T>
    std::vector<T> lu_solve(const Csr<T>& LU, const std::vector<T>& r)
    {
        const std::vector<T> F = dense(LU);
        return upper_solve(F, LU.n, lower_solve(F, LU.n, r, true), false);
    }

    // Pattern of A^p, without cancellation
    template <typename T>
    std::vector<bool> power_pattern(const Csr<T>& A, int p)
    {
        const int               n      = A.n;
        const std::vector<bool> base   = pattern(A);
        std::vector<bool>       result = base;
        for(int q = 1; q < p; ++q)
        {
            std::vector<bool> next(n * n);
            for(int i = 0; i < n; ++i)
            {
                for(int k = 0; k < n; ++k)
                {
                    if(result[i * n + k])
                    {
                        for(int j = 0; j < n; ++j)
                        {
                            next[i * n + j] = next[i * n + j] || base[k * n + j];
                        }
                    }
                }
            }
            result = next;
        }
        return result;
    }

    // Incomplete LU on the pattern P, with the factors stored in place
    template <typename T>
    Csr<T> ilu_on_pattern(const Csr<T>& A, const std::vector<bool>& P)
    {
        const int      n = A.n;
        std::vector<T> D = dense(A);
        for(int i = 1; i < n; ++i)
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

        Csr<T> F(n);
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                if(P[i * n + j])
                {
                    F.add(j, D[i * n + j]);
                }
            }
            F.end_row();
        }
        return F;
    }

    // ILU(p) with fill levels: an entry is kept if its level is at most p
    template <typename T>
    Csr<T> ilu_levels(const Csr<T>& A, int p)
    {
        const int        n   = A.n;
        const int        inf = 1 << 20;
        std::vector<T>   D   = dense(A);
        std::vector<int> lev(D.size());
        for(size_t k = 0; k < D.size(); ++k)
        {
            lev[k] = D[k] != T(0) ? 0 : inf;
        }

        for(int i = 1; i < n; ++i)
        {
            for(int k = 0; k < i; ++k)
            {
                if(lev[i * n + k] > p)
                {
                    continue;
                }
                D[i * n + k] /= D[k * n + k];
                for(int j = k + 1; j < n; ++j)
                {
                    lev[i * n + j] = std::min(lev[i * n + j], lev[i * n + k] + lev[k * n + j] + 1);
                    D[i * n + j] -= D[i * n + k] * D[k * n + j];
                }
            }
            for(int j = 0; j < n; ++j)
            {
                if(lev[i * n + j] > p)
                {
                    lev[i * n + j] = inf;
                    D[i * n + j]   = 0;
                }
            }
        }

        Csr<T> F(n);
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                if(lev[i * n + j] <= p)
                {
                    F.add(j, D[i * n + j]);
                }
            }
            F.end_row();
        }
        return F;
    }

    // L of the incomplete Cholesky factorization L L^T on the lower pattern of A
    template <typename T>
    std::vector<T> ic0(const Csr<T>& A)
    {
        const int      n = A.n;
        std::vector<T> D = dense(A);
        std::vector<T> L(D.size(), static_cast<T>(0));
        for(int i = 0; i < n; ++i)
        {
            T sum = 0;
            for(int j = 0; j < i; ++j)
            {
                if(D[i * n + j] == T(0))
                {
                    continue;
                }
                T local_sum = 0;
                for(int k = 0; k < j; ++k)
                {
                    local_sum += L[i * n + k] * L[j * n + k];
                }
                L[i * n + j] = (D[i * n + j] - local_sum) / L[j * n + j];
                sum += L[i * n + j] * L[i * n + j];
            }
            L[i * n + i] = std::sqrt(D[i * n + i] - sum);
        }
        return L;
    }

    template <typename T>
    std::vector<T> cholesky_solve(const std::vector<T>& L, int n, const std::vector<T>& r)
    {
        return upper_solve(dense_transpose(L, n), n, lower_solve(L, n, r, false), false);
    }

    template <typename T>
    std::vector<T> exact_solve(const Csr<T>& A, const std::vector<T>& r)
    {
        return dense_solve(dense(A), A.n, r);
    }

    template <typename T>
    using Reference = std::function<std::vector<T>(const Csr<T>&, const std::vector<T>&)>;

    template <typename T>
    double max_difference(const std::vector<T>& a, const std::vector<T>& b)
    {
        double d = 0;
        for(size_t i = 0; i < a.size(); ++i)
        {
            d = std::max(d, static_cast<double>(std::abs(a[i] - b[i])));
        }
        return d;
    }

    Backend other(Backend b)
    {
        return b == Backend::Host ? Backend::Accelerator : Backend::Host;
    }

    template <typename T>
    void move(Backend to, Solver<T>& p, Matrix<T>& A)
    {
        if(to == Backend::Host)
        {
            A.MoveToHost();
            p.MoveToHost();
        }
        else
        {
            A.MoveToAccelerator();
            p.MoveToAccelerator();
        }
    }

    // Rounding keeps the residual of the iterative triangular solves above 1e-15 in float
    template <typename T>
    constexpr double iterative_solve_tolerance()
    {
        return std::is_same_v<T, float> ? 1e-6 : 1e-15;
    }

    // A preconditioner of preconditioner.cpp and z = M^-1 r of rhs(n) for it
    template <typename T>
    struct DirectCase
    {
        std::string                name;
        std::unique_ptr<Solver<T>> p;
        std::vector<T>             reference;
    };

    template <typename T>
    std::vector<DirectCase<T>> direct_cases(const Csr<T>& A)
    {
        const std::vector<T> r = rhs<T>(A.n);

        auto ilu = std::make_unique<rocalution::ILU<Matrix<T>, Vector<T>, T>>();
        ilu->Set(1);

        auto itilu0 = std::make_unique<rocalution::ItILU0<Matrix<T>, Vector<T>, T>>();
        itilu0->SetMaxIter(200);
        itilu0->SetTolerance(1e-14);
        itilu0->SetOptions(rocalution::StoppingCriteria | rocalution::ComputeNrmCorrection);

        auto ilut = std::make_unique<rocalution::ILUT<Matrix<T>, Vector<T>, T>>();
        ilut->Set(0.0, A.n);

        std::vector<DirectCase<T>> cases;
        cases.push_back({"Jacobi",
                         std::make_unique<rocalution::Jacobi<Matrix<T>, Vector<T>, T>>(),
                         jacobi(A, r)});
        cases.push_back({"GS",
                         std::make_unique<rocalution::GS<Matrix<T>, Vector<T>, T>>(),
                         gauss_seidel(A, r)});
        cases.push_back({"SGS",
                         std::make_unique<rocalution::SGS<Matrix<T>, Vector<T>, T>>(),
                         symmetric_gauss_seidel(A, r)});
        cases.push_back({"ILU", std::move(ilu), lu_solve(ilu_levels(A, 1), r)});
        cases.push_back({"ItILU0", std::move(itilu0), lu_solve(ilu_levels(A, 0), r)});
        cases.push_back({"ILUT", std::move(ilut), exact_solve(A, r)});
        cases.push_back({"IC",
                         std::make_unique<rocalution::IC<Matrix<T>, Vector<T>, T>>(),
                         cholesky_solve(ic0(A), A.n, r)});
        return cases;
    }

    const std::vector<Format>& formats()
    {
        static const std::vector<Format> all = {{rocalution::CSR, 1},
                                                {rocalution::MCSR, 1},
                                                {rocalution::BCSR, 2},
                                                {rocalution::BCSR, 3},
                                                {rocalution::COO, 1},
                                                {rocalution::ELL, 1},
                                                {rocalution::DIA, 1},
                                                {rocalution::HYB, 1},
                                                {rocalution::DENSE, 1}};
        return all;
    }

    template <typename T>
    std::vector<T> dense_multiply(const std::vector<T>& A, const std::vector<T>& B, int n)
    {
        std::vector<T> C(A.size(), static_cast<T>(0));
        for(int i = 0; i < n; ++i)
        {
            for(int k = 0; k < n; ++k)
            {
                for(int j = 0; j < n; ++j)
                {
                    C[i * n + j] += A[i * n + k] * B[k * n + j];
                }
            }
        }
        return C;
    }

    // The FSAI factor G on the lower part of the pattern P: row i solves
    // A(J, J) m = e_i on the columns J of row i and is m / sqrt(m_i), so that
    // (G A G^T)_ii = 1. Returns G^T G r.
    template <typename T>
    std::vector<T> fsai(const Csr<T>& A, const std::vector<bool>& P, const std::vector<T>& r)
    {
        const int            n = A.n;
        const std::vector<T> D = dense(A);
        std::vector<T>       G(D.size(), static_cast<T>(0));
        for(int i = 0; i < n; ++i)
        {
            std::vector<int> J;
            for(int j = 0; j <= i; ++j)
            {
                if(P[i * n + j] || j == i)
                {
                    J.push_back(j);
                }
            }
            const int      k = static_cast<int>(J.size());
            std::vector<T> S(k * k);
            for(int a = 0; a < k; ++a)
            {
                for(int b = 0; b < k; ++b)
                {
                    S[a * k + b] = D[J[a] * n + J[b]];
                }
            }
            std::vector<T> e(k, static_cast<T>(0));
            e[k - 1]               = 1;
            const std::vector<T> m = dense_solve(S, k, e);
            for(int a = 0; a < k; ++a)
            {
                G[i * n + J[a]] = m[a] / std::sqrt(std::abs(m[k - 1]));
            }
        }
        return dense_apply(dense_transpose(G, n), n, dense_apply(G, n, r));
    }

    // SPAI on the pattern of A: column i of M minimises |A m - e_i| over the
    // entries of column i of A. Returns M r.
    template <typename T>
    std::vector<T> spai(const Csr<T>& A, const std::vector<T>& r)
    {
        const int            n = A.n;
        const std::vector<T> D = dense(A);
        std::vector<T>       M(D.size(), static_cast<T>(0));
        for(int i = 0; i < n; ++i)
        {
            std::vector<int> J;
            for(int j = 0; j < n; ++j)
            {
                if(D[j * n + i] != T(0))
                {
                    J.push_back(j);
                }
            }
            const int      k = static_cast<int>(J.size());
            std::vector<T> N(k * k, static_cast<T>(0));
            std::vector<T> b(k);
            for(int a = 0; a < k; ++a)
            {
                for(int c = 0; c < k; ++c)
                {
                    for(int row = 0; row < n; ++row)
                    {
                        N[a * k + c] += D[row * n + J[a]] * D[row * n + J[c]];
                    }
                }
                b[a] = D[i * n + J[a]];
            }
            const std::vector<T> m = dense_solve(N, k, b);
            for(int a = 0; a < k; ++a)
            {
                M[J[a] * n + i] = m[a];
            }
        }
        return dense_apply(M, n, r);
    }

    // K^T D^-1 K r with K = I - L + L^2 and L = strict_lower(A) D^-1
    template <typename T>
    std::vector<T> truncated_neumann(const Csr<T>& A, const std::vector<T>& r)
    {
        const int            n    = A.n;
        const std::vector<T> D    = dense(A);
        const std::vector<T> dinv = inverted(diagonal(D, n));
        std::vector<T>       L(D.size(), static_cast<T>(0));
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < i; ++j)
            {
                L[i * n + j] = D[i * n + j] * dinv[j];
            }
        }
        const std::vector<T> L2 = dense_multiply(L, L, n);
        std::vector<T>       K(D.size(), static_cast<T>(0));
        for(int i = 0; i < n * n; ++i)
        {
            K[i] = L2[i] - L[i];
        }
        for(int i = 0; i < n; ++i)
        {
            K[i * n + i] += 1;
        }
        return dense_apply(dense_transpose(K, n), n, pointwise(dinv, dense_apply(K, n, r)));
    }

    // Symmetric tridiagonal Toeplitz matrix with diagonal 4 and off-diagonals -1,
    // whose eigenvalues 4 - 2 cos(k pi / (n + 1)) lie in (2, 6)
    template <typename T>
    Csr<T> toeplitz(int n)
    {
        Csr<T> A(n);
        for(int i = 0; i < n; ++i)
        {
            if(i > 0)
            {
                A.add(i - 1, static_cast<T>(-1));
            }
            A.add(i, static_cast<T>(4));
            if(i + 1 < n)
            {
                A.add(i + 1, static_cast<T>(-1));
            }
            A.end_row();
        }
        return A;
    }

    // 2 x 2 blocks on the diagonal
    template <typename T>
    Csr<T> block_diagonal(int blocks)
    {
        Csr<T> A(2 * blocks);
        for(int b = 0; b < blocks; ++b)
        {
            A.add(2 * b, static_cast<T>(4 + b % 3));
            A.add(2 * b + 1, static_cast<T>(1 + b % 2));
            A.end_row();
            A.add(2 * b, static_cast<T>(-2));
            A.add(2 * b + 1, static_cast<T>(5));
            A.end_row();
        }
        return A;
    }

    template <typename T>
    std::vector<T> submatrix(const std::vector<T>& D, int n, int pos, int size)
    {
        std::vector<T> S(size * size);
        for(int i = 0; i < size; ++i)
        {
            for(int j = 0; j < size; ++j)
            {
                S[i * size + j] = D[(pos + i) * n + pos + j];
            }
        }
        return S;
    }

    // Block i of the Schwarz preconditioners owns the rows [i size, (i + 1) size),
    // the last one up to n, and is extended by the overlap on both sides
    struct SchwarzBlock
    {
        int pos;
        int size;
        int owned;
    };

    std::vector<SchwarzBlock> schwarz_blocks(int n, int nb, int overlap)
    {
        const int                 size = n / nb;
        std::vector<SchwarzBlock> blocks;
        for(int i = 0; i < nb; ++i)
        {
            const int begin = std::max(0, i * size - overlap);
            const int end   = i == nb - 1 ? n : (i + 1) * size + overlap;
            blocks.push_back({begin, end - begin, i * size - begin});
        }
        return blocks;
    }

    // Exact solves on the blocks, added up with the weight 1 / (number of blocks
    // that hold the row)
    template <typename T>
    std::vector<T> additive_schwarz(const Csr<T>& A, int nb, int overlap, const std::vector<T>& r)
    {
        const int            n = A.n;
        const std::vector<T> D = dense(A);
        std::vector<T>       x(n, static_cast<T>(0));
        std::vector<int>     count(n, 0);
        for(const SchwarzBlock& b : schwarz_blocks(n, nb, overlap))
        {
            const std::vector<T> z
                = dense_solve(submatrix(D, n, b.pos, b.size),
                              b.size,
                              std::vector<T>(r.begin() + b.pos, r.begin() + b.pos + b.size));
            for(int k = 0; k < b.size; ++k)
            {
                x[b.pos + k] += z[k];
                ++count[b.pos + k];
            }
        }
        for(int k = 0; k < n; ++k)
        {
            x[k] /= static_cast<T>(count[k]);
        }
        return x;
    }

    // Exact solves on the blocks, each giving the rows it owns
    template <typename T>
    std::vector<T>
        restricted_additive_schwarz(const Csr<T>& A, int nb, int overlap, const std::vector<T>& r)
    {
        const int                       n      = A.n;
        const std::vector<T>            D      = dense(A);
        const std::vector<SchwarzBlock> blocks = schwarz_blocks(n, nb, overlap);
        std::vector<T>                  x(n, static_cast<T>(0));
        for(size_t i = 0; i < blocks.size(); ++i)
        {
            const SchwarzBlock&  b = blocks[i];
            const std::vector<T> z
                = dense_solve(submatrix(D, n, b.pos, b.size),
                              b.size,
                              std::vector<T>(r.begin() + b.pos, r.begin() + b.pos + b.size));
            const int owned_end
                = i + 1 == blocks.size() ? n : blocks[i + 1].pos + blocks[i + 1].owned;
            for(int k = b.pos + b.owned; k < owned_end; ++k)
            {
                x[k] = z[k - b.pos];
            }
        }
        return x;
    }

    // Exact solvers for the blocks of the block preconditioners
    template <typename T>
    struct ExactSolvers
    {
        explicit ExactSolvers(int count)
        {
            for(int i = 0; i < count; ++i)
            {
                lu.push_back(std::make_unique<rocalution::LU<Matrix<T>, Vector<T>, T>>());
                lu.back()->Verbose(0);
                pointers.push_back(lu.back().get());
            }
        }

        Solver<T>** get()
        {
            return pointers.data();
        }

        std::vector<std::unique_ptr<rocalution::LU<Matrix<T>, Vector<T>, T>>> lu;
        std::vector<Solver<T>*>                                               pointers;
    };

    // The colors of the multicolored preconditioners, from the matrix on the
    // backend b, after checking that each color is an independent set
    struct Coloring
    {
        std::vector<int> sizes;
        std::vector<int> perm;
    };

    template <typename T>
    Coloring multicoloring(Backend b, const Csr<T>& structure)
    {
        auto             S           = make_local_matrix(b, structure);
        int              num_colors  = 0;
        int*             size_colors = nullptr;
        LocalVector<int> perm;
        perm.CloneBackend(*S);
        S->MultiColoring(num_colors, &size_colors, &perm);

        Coloring c{std::vector<int>(size_colors, size_colors + num_colors), values_of(perm)};
        rocalution::free_host(&size_colors);

        const int n = structure.n;
        EXPECT_EQ(std::accumulate(c.sizes.begin(), c.sizes.end(), 0), n);
        std::vector<int> sorted = c.perm;
        std::sort(sorted.begin(), sorted.end());
        for(int i = 0; i < n; ++i)
        {
            EXPECT_EQ(sorted[i], i) << "the coloring is not a permutation";
        }

        std::vector<int> color_of_position(n);
        for(int k = 0, offset = 0; k < num_colors; offset += c.sizes[k++])
        {
            std::fill(color_of_position.begin() + offset,
                      color_of_position.begin() + offset + c.sizes[k],
                      k);
        }
        for(int i = 0; i < n; ++i)
        {
            for(PtrType k = structure.row_offset[i]; k < structure.row_offset[i + 1]; ++k)
            {
                const int j = structure.col[k];
                EXPECT_TRUE(i == j || color_of_position[c.perm[i]] != color_of_position[c.perm[j]])
                    << "rows " << i << " and " << j << " are coupled and share a color";
            }
        }
        return c;
    }

    // The independent set of the multi-elimination, from the matrix on the
    // backend b, after checking that it is independent and maximal
    template <typename T>
    std::vector<int> independent_set(Backend b, const Csr<T>& A, int& size)
    {
        auto             M = make_local_matrix(b, A);
        LocalVector<int> perm;
        perm.CloneBackend(*M);
        M->MaximalIndependentSet(size, &perm);
        const std::vector<int> p = values_of(perm);

        for(int i = 0; i < A.n; ++i)
        {
            bool coupled_to_set = false;
            for(PtrType k = A.row_offset[i]; k < A.row_offset[i + 1]; ++k)
            {
                const int j = A.col[k];
                if(i != j && p[j] < size)
                {
                    coupled_to_set = true;
                    EXPECT_GE(p[i], size) << "rows " << i << " and " << j << " are both in the set";
                }
            }
            EXPECT_TRUE(p[i] < size || coupled_to_set) << "row " << i << " could join the set";
        }
        return p;
    }

    // z = solve(B, P r) for B = P A P^T, returned as P^T z
    template <typename T, typename Solve>
    std::vector<T> permuted_solve(const Csr<T>&           A,
                                  const std::vector<int>& perm,
                                  const std::vector<T>&   r,
                                  const Solve&            solve)
    {
        std::vector<T> b(r.size());
        for(size_t i = 0; i < r.size(); ++i)
        {
            b[perm[i]] = r[i];
        }
        const std::vector<T> z = solve(permuted(A, perm), b);
        std::vector<T>       x(r.size());
        for(size_t i = 0; i < r.size(); ++i)
        {
            x[i] = z[perm[i]];
        }
        return x;
    }

    // omega (2 - omega) (D + omega U)^-1 D (D + omega L)^-1 r
    template <typename T>
    std::vector<T> ssor(const Csr<T>& A, T omega, const std::vector<T>& r)
    {
        const int      n = A.n;
        std::vector<T> R = dense(A);
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                if(i != j)
                {
                    R[i * n + j] *= omega;
                }
            }
        }
        std::vector<T> z
            = upper_solve(R, n, pointwise(diagonal(R, n), lower_solve(R, n, r, false)), false);
        for(T& v : z)
        {
            v *= omega * (2 - omega);
        }
        return z;
    }

    // (D / omega + U)^-1 r
    template <typename T>
    std::vector<T> backward_sor(const Csr<T>& A, T omega, const std::vector<T>& r)
    {
        const int      n = A.n;
        std::vector<T> R = dense(A);
        for(int i = 0; i < n; ++i)
        {
            R[i * n + i] /= omega;
        }
        return upper_solve(R, n, r, false);
    }

    template <typename T>
    Csr<T> pattern_matrix(const std::vector<bool>& P, int n)
    {
        Csr<T> S(n);
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                if(P[i * n + j])
                {
                    S.add(j, static_cast<T>(1));
                }
            }
            S.end_row();
        }
        return S;
    }

    // The block lower triangular part of A, or its block diagonal, with the last
    // diagonal block replaced by last if it has rows
    template <typename T>
    std::vector<T> block_lower_solve(const Csr<T>&           A,
                                     const std::vector<int>& sizes,
                                     bool                    diagonal_only,
                                     const std::vector<T>&   last,
                                     const std::vector<T>&   r)
    {
        const int        n = A.n;
        std::vector<T>   L = dense(A);
        std::vector<int> block(n);
        for(int k = 0, offset = 0; k < static_cast<int>(sizes.size()); offset += sizes[k++])
        {
            std::fill(block.begin() + offset, block.begin() + offset + sizes[k], k);
        }
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                if(block[j] > block[i] || (diagonal_only && block[j] != block[i]))
                {
                    L[i * n + j] = 0;
                }
            }
        }
        if(!last.empty())
        {
            const int m      = sizes.back();
            const int offset = n - m;
            for(int i = 0; i < m; ++i)
            {
                for(int j = 0; j < m; ++j)
                {
                    L[(offset + i) * n + offset + j] = last[i * m + j];
                }
            }
        }
        return dense_solve(L, n, r);
    }

    // Saddle point matrix [K B^T; B 0] with K symmetric positive definite, the
    // constraint rows at positions 2, 5, 8 and 11 and no stored zero diagonal
    template <typename T>
    Csr<T> saddle_point()
    {
        const int        n = 12;
        std::vector<T>   D(n * n, static_cast<T>(0));
        std::vector<int> unknowns;
        std::vector<int> constraints = {2, 5, 8, 11};
        for(int i = 0; i < n; ++i)
        {
            if(std::find(constraints.begin(), constraints.end(), i) == constraints.end())
            {
                unknowns.push_back(i);
            }
        }
        for(size_t a = 0; a < unknowns.size(); ++a)
        {
            D[unknowns[a] * n + unknowns[a]] = static_cast<T>(6.0 + a % 3);
            if(a + 1 < unknowns.size())
            {
                D[unknowns[a] * n + unknowns[a + 1]] = -1;
                D[unknowns[a + 1] * n + unknowns[a]] = -1;
            }
        }
        for(size_t c = 0; c < constraints.size(); ++c)
        {
            for(int k = 0; k < 2; ++k)
            {
                const int u               = unknowns[2 * c + k];
                const T   v               = static_cast<T>(k == 0 ? 1.0 : -2.0 + c % 2);
                D[constraints[c] * n + u] = v;
                D[u * n + constraints[c]] = v;
            }
        }
        return from_dense(n, n, D);
    }

    template <typename T>
    std::string printed(const Solver<T>& p)
    {
        testing::internal::CaptureStdout();
        p.Print();
        return testing::internal::GetCapturedStdout();
    }

    // The first lines of s, with their line ends
    std::string head(const std::string& s, int lines)
    {
        size_t end = 0;
        for(int i = 0; i < lines; ++i)
        {
            end = s.find('\n', end);
            if(end == std::string::npos)
            {
                return s;
            }
            ++end;
        }
        return s.substr(0, end);
    }

    template <typename P>
    class preconditioners : public TypedBackendTest<P>
    {
    protected:
        using T = typename P::value_type;

        Backend backend() const
        {
            return P::backend;
        }

        std::unique_ptr<Matrix<T>> make(const Csr<T>& ref) const
        {
            return make_local_matrix(P::backend, ref);
        }
    };
}

TYPED_TEST_SUITE(preconditioners, WithBackends<RealTypes>, TypedName);

TYPED_TEST(preconditioners, jacobi_scales_by_the_inverse_diagonal)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(10);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    rocalution::Jacobi<Matrix<T>, Vector<T>, T> p;
    p.SetOperator(*A);
    p.Build();
    expect_near(apply(p, *A, r), jacobi(ref, r), scaled_tol<T>(1e-15));

    auto in = make_local_vector(*A, r);
    p.Solve(*in, in.get());
    expect_near(values_of(*in), jacobi(ref, r), scaled_tol<T>(1e-15));

    auto out = make_local_vector(*A, std::vector<T>(ref.n, 7.0));
    p.SolveZeroSol(*make_local_vector(*A, r), out.get());
    expect_near(values_of(*out), jacobi(ref, r), scaled_tol<T>(1e-15));
}

// Without entries there is no inverse diagonal, and Solve copies the right-hand side
TYPED_TEST(preconditioners, jacobi_of_a_matrix_without_entries_is_the_identity)
{
    using T = typename TypeParam::value_type;
    const Csr<T>         empty(6);
    const std::vector<T> r = rhs<T>(6);

    auto A = std::make_unique<Matrix<T>>();
    A->AllocateCSR("empty", 0, 6, 6);
    if(this->backend() == Backend::Accelerator)
    {
        A->MoveToAccelerator();
    }

    rocalution::Jacobi<Matrix<T>, Vector<T>, T> p;
    p.SetOperator(*A);
    p.Build();
    expect_near(apply(p, *A, r), r, 0.0);

    auto in = make_local_vector(*A, r);
    p.Solve(*in, in.get());
    expect_near(values_of(*in), r, 0.0);
}

TYPED_TEST(preconditioners, gs_is_the_forward_gauss_seidel_sweep)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(10);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    rocalution::GS<Matrix<T>, Vector<T>, T> p;
    p.SetOperator(*A);
    p.Build();
    expect_near(apply(p, *A, r), gauss_seidel(ref, r), scaled_tol<T>(1e-14));
}

TYPED_TEST(preconditioners, sgs_is_the_symmetric_gauss_seidel_sweep)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(10);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    rocalution::SGS<Matrix<T>, Vector<T>, T> p;
    p.SetOperator(*A);
    p.Build();

    // ResetOperator extracts the diagonal, Build its inverse
    p.ResetOperator(*A);
    expect_near(apply(p, *A, r), symmetric_gauss_seidel(ref, r), scaled_tol<T>(1e-14));

    p.Build();
    const std::vector<T> built = apply(p, *A, r);
    move(other(this->backend()), p, *A);
    const std::vector<T> moved = apply(p, *A, r);
    expect_near(moved, built, scaled_tol<T>(1e-14));

    KNOWN_BUG("Build scales by the inverse diagonal instead of the diagonal");
    expect_near(built, symmetric_gauss_seidel(ref, r), scaled_tol<T>(1e-14));
    expect_near(moved, symmetric_gauss_seidel(ref, r), scaled_tol<T>(1e-14));
}

// The AMG hierarchy calls ResetOperator with the operator the smoother was set
// up with, after changing its values
TYPED_TEST(preconditioners, reset_operator_rebuilds_from_the_new_values)
{
    using T                      = typename TypeParam::value_type;
    const Csr<T>         ref     = nonsymmetric<T>(10);
    const Csr<T>         changed = transformed(ref, [](int, int, T v) { return 2 * v + 1; });
    const std::vector<T> r       = rhs<T>(ref.n);

    rocalution::Jacobi<Matrix<T>, Vector<T>, T> jacobi_p;
    rocalution::GS<Matrix<T>, Vector<T>, T>     gs_p;
    rocalution::SGS<Matrix<T>, Vector<T>, T>    sgs_p;

    const std::vector<std::tuple<std::string, Solver<T>*, Reference<T>>> cases = {
        {"Jacobi", &jacobi_p, jacobi<T>},
        {"GS", &gs_p, gauss_seidel<T>},
        {"SGS", &sgs_p, symmetric_gauss_seidel<T>},
    };

    std::vector<std::unique_ptr<Matrix<T>>> matrices;
    for(const auto& [name, p, reference] : cases)
    {
        SCOPED_TRACE(name);
        matrices.push_back(this->make(ref));
        Matrix<T>& A = *matrices.back();
        p->SetOperator(A);
        p->Build();

        A.CopyFromCSR(changed.row_offset.data(), changed.col.data(), changed.val.data());
        p->ResetOperator(A);
        expect_near(apply(*p, A, r), reference(changed, r), scaled_tol<T>(1e-14));
    }
}

TYPED_TEST(preconditioners, reset_operator_switches_to_another_matrix)
{
    using T                      = typename TypeParam::value_type;
    const Csr<T>         ref     = nonsymmetric<T>(10);
    const Csr<T>         changed = transformed(ref, [](int, int, T v) { return 2 * v + 1; });
    const std::vector<T> r       = rhs<T>(ref.n);
    auto                 A       = this->make(ref);
    auto                 B       = this->make(changed);

    // The base class only replaces the operator
    rocalution::ILU<Matrix<T>, Vector<T>, T> ilu;
    ilu.SetOperator(*A);
    ilu.Build();
    ilu.ResetOperator(*B);
    ilu.Build();
    expect_near(apply(ilu, *B, r), lu_solve(ilu_levels(changed, 0), r), scaled_tol<T>(1e-13));

    rocalution::Jacobi<Matrix<T>, Vector<T>, T> p;
    p.SetOperator(*A);
    p.Build();
    p.ResetOperator(*B);
    const std::vector<T> z = apply(p, *B, r);

    KNOWN_BUG("Jacobi, GS and SGS ResetOperator rebuild from the old operator");
    expect_near(z, jacobi(changed, r), scaled_tol<T>(1e-14));
}

TYPED_TEST(preconditioners, ilu0_of_a_tridiagonal_matrix_solves_exactly)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = Tridiagonal<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    rocalution::ILU<Matrix<T>, Vector<T>, T> p;
    p.SetOperator(*A);
    p.Build();
    expect_near(apply(p, *A, r), exact_solve(ref, r), scaled_tol<T>(1e-13));
}

TYPED_TEST(preconditioners, ilup_keeps_the_fill_up_to_level_p)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    // The two references differ from p = 2 on, so that those cases check which pattern is used
    EXPECT_GT(max_difference(lu_solve(ilu_levels(ref, 2), r),
                             lu_solve(ilu_on_pattern(ref, power_pattern(ref, 3)), r)),
              1e-6);

    for(int p = 0; p <= 3; ++p)
    {
        for(bool level : {true, false})
        {
            SCOPED_TRACE("p = " + std::to_string(p) + (level ? ", levels" : ", power pattern"));

            rocalution::ILU<Matrix<T>, Vector<T>, T> ilu;
            ilu.Set(p, level);
            ilu.SetOperator(*A);
            ilu.Build();

            const Csr<T> factors
                = level ? ilu_levels(ref, p) : ilu_on_pattern(ref, power_pattern(ref, p + 1));
            expect_near(apply(ilu, *A, r), lu_solve(factors, r), tol<T>());
        }
    }
}

TYPED_TEST(preconditioners, itilu0_converges_to_the_ilu0)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    for(auto alg : {rocalution::Default,
                    rocalution::AsyncInPlace,
                    rocalution::SyncSplit,
                    rocalution::SyncSplitFusion})
    {
        SCOPED_TRACE("algorithm " + std::to_string(alg));

        rocalution::ItILU0<Matrix<T>, Vector<T>, T> p;
        p.SetAlgorithm(alg);
        p.SetMaxIter(200);
        p.SetTolerance(1e-14);
        p.SetOptions(rocalution::StoppingCriteria | rocalution::ComputeNrmCorrection);
        p.SetOperator(*A);
        p.Build();
        expect_near(apply(p, *A, r), lu_solve(ilu_levels(ref, 0), r), loose_tol<T>(1e-10));
    }
}

// The host computes the ILU(0) directly and reports no iterations
TYPED_TEST(preconditioners, itilu0_reports_its_convergence_history)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = nonsymmetric<T>(12);
    auto         A   = this->make(ref);

    rocalution::ItILU0<Matrix<T>, Vector<T>, T> p;
    p.SetAlgorithm(rocalution::SyncSplit);
    p.SetMaxIter(100);
    p.SetTolerance(1e-14);
    p.SetOptions(rocalution::StoppingCriteria | rocalution::ComputeNrmResidual
                 | rocalution::ConvergenceHistory);
    p.SetOperator(*A);
    p.Build();

    int           niter   = -1;
    const double* history = p.GetConvergenceHistory(&niter);
    ASSERT_NE(history, nullptr);

    if(this->backend() == Backend::Host)
    {
        EXPECT_EQ(niter, 0);
        return;
    }

    ASSERT_GE(niter, 1);
    ASSERT_LE(niter, 100);
    EXPECT_GT(history[niter], 0.0);
    EXPECT_LE(history[2 * niter - 1], history[niter]);
}

TYPED_TEST(preconditioners, ilut_without_dropping_is_the_complete_lu)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    rocalution::ILUT<Matrix<T>, Vector<T>, T> full;
    full.Set(0.0, ref.n);
    full.SetOperator(*A);
    full.Build();
    expect_near(apply(full, *A, r), exact_solve(ref, r), tol<T>());

    // The default row limit holds every entry of this matrix
    rocalution::ILUT<Matrix<T>, Vector<T>, T> threshold_only;
    threshold_only.Set(0.0);
    threshold_only.SetOperator(*A);
    threshold_only.Build();
    expect_near(apply(threshold_only, *A, r), exact_solve(ref, r), tol<T>());
}

// Every off-diagonal entry is below the threshold, and the diagonal is always kept
TYPED_TEST(preconditioners, ilut_with_a_large_threshold_keeps_the_diagonal)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    rocalution::ILUT<Matrix<T>, Vector<T>, T> p;
    p.Set(1e6, ref.n);
    p.SetOperator(*A);
    p.Build();
    expect_near(apply(p, *A, r), jacobi(ref, r), scaled_tol<T>(1e-14));
}

TYPED_TEST(preconditioners, ic_is_the_incomplete_cholesky_solve)
{
    using T = typename TypeParam::value_type;
    for(const auto& [name, ref] : std::vector<std::pair<std::string, Csr<T>>>{
            {"tridiagonal", Tridiagonal<T>(12)}, {"general", spd<T>(12)}})
    {
        SCOPED_TRACE(name);
        const std::vector<T> r = rhs<T>(ref.n);
        auto                 A = this->make(ref);

        rocalution::IC<Matrix<T>, Vector<T>, T> p;
        p.SetOperator(*A);
        p.Build();
        expect_near(apply(p, *A, r), cholesky_solve(ic0(ref), ref.n, r), tol<T>());
    }

    const Csr<T> tridiagonal = Tridiagonal<T>(12);
    EXPECT_LT(max_difference(cholesky_solve(ic0(tridiagonal), 12, rhs<T>(12)),
                             exact_solve(tridiagonal, rhs<T>(12))),
              tol<T>());
}

// With the iterative triangular solves, enough iterations reach the direct solve
TYPED_TEST(preconditioners, iterative_triangular_solves_reach_the_direct_solve)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = spd<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    rocalution::SolverDescr descr;
    descr.SetTriSolverAlg(rocalution::TriSolverAlg_Iterative);
    descr.SetIterativeSolverMaxIteration(100);
    descr.SetIterativeSolverTolerance(iterative_solve_tolerance<T>());

    for(auto& c : direct_cases(ref))
    {
        if(c.name == "SGS")
        {
            continue;
        }
        SCOPED_TRACE(c.name);
        c.p->SetSolverDescriptor(descr);
        c.p->SetOperator(*A);
        c.p->Build();
        expect_near(apply(*c.p, *A, r), c.reference, loose_tol<T>(1e-11));
    }
}

TYPED_TEST(preconditioners, build_of_a_built_preconditioner_takes_the_new_values)
{
    using T                      = typename TypeParam::value_type;
    const Csr<T>         ref     = spd<T>(12);
    const Csr<T>         changed = transformed(ref, [](int, int, T v) { return 2 * v + 1; });
    const std::vector<T> r       = rhs<T>(ref.n);

    for(auto& c : direct_cases(changed))
    {
        if(c.name == "SGS")
        {
            continue;
        }
        SCOPED_TRACE(c.name);
        auto A = this->make(ref);
        c.p->SetOperator(*A);
        c.p->Build();

        A->CopyFromCSR(changed.row_offset.data(), changed.col.data(), changed.val.data());
        c.p->Build();
        expect_near(apply(*c.p, *A, r), c.reference, loose_tol<T>(1e-11));
    }
}

// ResetOperator, unlike Build, scales by the diagonal
TYPED_TEST(preconditioners, iterative_sgs_reaches_the_direct_solve)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = spd<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    rocalution::SolverDescr descr;
    descr.SetTriSolverAlg(rocalution::TriSolverAlg_Iterative);
    descr.SetIterativeSolverMaxIteration(100);
    descr.SetIterativeSolverTolerance(iterative_solve_tolerance<T>());

    rocalution::SGS<Matrix<T>, Vector<T>, T> p;
    p.SetSolverDescriptor(descr);
    p.SetOperator(*A);
    p.Build();
    p.ResetOperator(*A);

    expect_near(apply(p, *A, r), symmetric_gauss_seidel(ref, r), loose_tol<T>(1e-11));
}

TYPED_TEST(preconditioners, built_preconditioners_move_to_the_other_backend)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = spd<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);

    for(auto& c : direct_cases(ref))
    {
        if(c.name == "SGS" || c.name == "ILUT" || c.name == "IC")
        {
            continue;
        }
        SCOPED_TRACE(c.name);
        auto A = this->make(ref);
        c.p->SetOperator(*A);
        c.p->Build();

        move(other(this->backend()), *c.p, *A);
        expect_near(apply(*c.p, *A, r), c.reference, loose_tol<T>(1e-11));

        move(this->backend(), *c.p, *A);
        expect_near(apply(*c.p, *A, r), c.reference, loose_tol<T>(1e-11));
    }
}

// The moved factors are analysed again, and IC moves its inverse diagonal along
TYPED_TEST(preconditioners, built_ilut_and_ic_move_to_the_other_backend)
{
    using T = typename TypeParam::value_type;
    REQUIRE_BACKEND(Backend::Accelerator);
    const Csr<T>         ref = spd<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);

    for(auto& c : direct_cases(ref))
    {
        if(c.name != "ILUT" && c.name != "IC")
        {
            continue;
        }
        SCOPED_TRACE(c.name);
        auto A = make_local_matrix(this->backend(), ref);
        c.p->SetOperator(*A);
        c.p->Build();
        move(other(this->backend()), *c.p, *A);
        expect_near(apply(*c.p, *A, r), c.reference, loose_tol<T>(1e-11));
    }
}

// With the iterative triangular solves, the move clears the analysis before it
// analyses the factors again
TYPED_TEST(preconditioners, built_ilu_moves_to_its_own_backend)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = spd<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    rocalution::SolverDescr descr;
    descr.SetTriSolverAlg(rocalution::TriSolverAlg_Iterative);
    descr.SetIterativeSolverMaxIteration(100);
    descr.SetIterativeSolverTolerance(iterative_solve_tolerance<T>());

    rocalution::ILU<Matrix<T>, Vector<T>, T> p;
    p.SetSolverDescriptor(descr);
    p.SetOperator(*A);
    p.Build();
    move(this->backend(), p, *A);
    expect_near(apply(p, *A, r), lu_solve(ilu_levels(ref, 0), r), loose_tol<T>(1e-11));
}

// ResetOperator scales by the diagonal, unlike the buggy Build, so it is the
// starting point of the moves
TYPED_TEST(preconditioners, sgs_moves_with_its_factors_and_its_diagonal)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(10);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    rocalution::SGS<Matrix<T>, Vector<T>, T> p;
    p.SetOperator(*A);
    p.Build();
    p.Build();
    p.ResetOperator(*A);
    expect_near(apply(p, *A, r), symmetric_gauss_seidel(ref, r), scaled_tol<T>(1e-14));

    for(Backend to : {this->backend(), other(this->backend()), this->backend()})
    {
        SCOPED_TRACE("moved to the " + backend_name(to));
        move(to, p, *A);
        expect_near(apply(p, *A, r), symmetric_gauss_seidel(ref, r), scaled_tol<T>(1e-14));
    }
}

// A move to the backend the preconditioner is on, and of ILUT from the accelerator to
// the host. SGS has its own test.
TYPED_TEST(preconditioners, built_preconditioners_move_to_their_own_backend)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = spd<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);

    for(auto& c : direct_cases(ref))
    {
        if(c.name == "SGS")
        {
            continue;
        }
        SCOPED_TRACE(c.name);
        auto A = this->make(ref);
        c.p->SetOperator(*A);
        c.p->Build();

        move(this->backend(), *c.p, *A);
        expect_near(apply(*c.p, *A, r), c.reference, loose_tol<T>(1e-11));

        if(c.name == "ILUT" && this->backend() == Backend::Accelerator)
        {
            move(Backend::Host, *c.p, *A);
            expect_near(apply(*c.p, *A, r), c.reference, loose_tol<T>(1e-11));
        }
    }
}

TYPED_TEST(preconditioners, variable_preconditioner_cycles_through_its_sequence)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(10);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    // The sequence outlives p, whose destructor clears it
    rocalution::Jacobi<Matrix<T>, Vector<T>, T>                 first;
    rocalution::GS<Matrix<T>, Vector<T>, T>                     second;
    rocalution::Jacobi<Matrix<T>, Vector<T>, T>                 only;
    rocalution::VariablePreconditioner<Matrix<T>, Vector<T>, T> p;

    Solver<T>* sequence[] = {&first, &second};
    p.SetPreconditioner(2, sequence);
    p.SetOperator(*A);
    p.Build();

    expect_near(apply(p, *A, r), jacobi(ref, r), scaled_tol<T>(1e-14));
    expect_near(apply(p, *A, r), gauss_seidel(ref, r), scaled_tol<T>(1e-14));
    expect_near(apply(p, *A, r), jacobi(ref, r), scaled_tol<T>(1e-14));

    move(other(this->backend()), p, *A);
    expect_near(apply(p, *A, r), gauss_seidel(ref, r), scaled_tol<T>(1e-14));
    expect_near(apply(p, *A, r), jacobi(ref, r), scaled_tol<T>(1e-14));

    p.Clear();
    Solver<T>* single[] = {&only};
    p.SetPreconditioner(1, single);
    p.Build();
    expect_near(apply(p, *A, r), jacobi(ref, r), scaled_tol<T>(1e-14));
    expect_near(apply(p, *A, r), jacobi(ref, r), scaled_tol<T>(1e-14));
}

// Clear forgets the sequence, so a second Build fails its precondition
TYPED_TEST(preconditioners, variable_preconditioner_builds_again)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(10);
    const std::vector<T> r   = rhs<T>(ref.n);

    KNOWN_BUG("Clear discards the configuration that Build and ReBuildNumeric need");

    expect_in_child([&] {
        auto                                                        A = this->make(ref);
        rocalution::Jacobi<Matrix<T>, Vector<T>, T>                 first;
        rocalution::VariablePreconditioner<Matrix<T>, Vector<T>, T> p;
        Solver<T>*                                                  sequence[] = {&first};
        p.SetPreconditioner(1, sequence);
        p.SetOperator(*A);
        p.Build();
        p.Build();
        expect_near(apply(p, *A, r), jacobi(ref, r), scaled_tol<T>(1e-14));
    });
}

// Before Build there is nothing to clear or to move
TYPED_TEST(preconditioners, variable_preconditioner_is_configured_after_a_clear_and_moves)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(10);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    rocalution::Jacobi<Matrix<T>, Vector<T>, T>                 first;
    rocalution::GS<Matrix<T>, Vector<T>, T>                     second;
    rocalution::VariablePreconditioner<Matrix<T>, Vector<T>, T> p;
    p.Clear();
    p.MoveToAccelerator();
    p.MoveToHost();

    Solver<T>* sequence[] = {&first, &second};
    p.SetPreconditioner(2, sequence);
    p.SetOperator(*A);
    p.Build();
    expect_near(apply(p, *A, r), jacobi(ref, r), scaled_tol<T>(1e-14));
    expect_near(apply(p, *A, r), gauss_seidel(ref, r), scaled_tol<T>(1e-14));
}

TYPED_TEST(preconditioners, print_describes_the_configuration)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = nonsymmetric<T>(10);
    auto         A   = this->make(ref);

    rocalution::Jacobi<Matrix<T>, Vector<T>, T> jacobi_p;
    EXPECT_EQ(printed(jacobi_p), "Jacobi preconditioner\n");

    rocalution::GS<Matrix<T>, Vector<T>, T> gs_p;
    EXPECT_EQ(head(printed(gs_p), 1), "Gauss-Seidel (GS) preconditioner\n");

    rocalution::SGS<Matrix<T>, Vector<T>, T> sgs_p;
    EXPECT_EQ(head(printed(sgs_p), 1), "Symmetric Gauss-Seidel (SGS) preconditioner\n");

    rocalution::ILU<Matrix<T>, Vector<T>, T> ilu;
    ilu.Set(1);
    EXPECT_EQ(printed(ilu), "ILU(1) preconditioner\n");
    ilu.SetOperator(*A);
    ilu.Build();
    EXPECT_EQ(head(printed(ilu), 2),
              "ILU(1) preconditioner\nILU nnz = " + std::to_string(ilu_levels(ref, 1).nnz())
                  + "\n");

    rocalution::ItILU0<Matrix<T>, Vector<T>, T> itilu0;
    EXPECT_EQ(printed(itilu0), "ItILU0(AsyncInPlace,10,0.01) preconditioner\n");
    itilu0.SetAlgorithm(rocalution::AsyncInPlace);
    EXPECT_EQ(printed(itilu0), "ItILU0(AsyncInPlace,10,0.01) preconditioner\n");
    itilu0.SetAlgorithm(rocalution::AsyncSplit);
    EXPECT_EQ(printed(itilu0), "ItILU0(AsyncSplit,10,0.01) preconditioner\n");
    itilu0.SetAlgorithm(rocalution::SyncSplitFusion);
    itilu0.SetOptions(rocalution::Verbose | rocalution::StoppingCriteria
                      | rocalution::ComputeNrmCorrection | rocalution::ComputeNrmResidual
                      | rocalution::COOFormat);
    itilu0.SetMaxIter(20);
    itilu0.SetTolerance(0.5);
    EXPECT_EQ(printed(itilu0),
              "ItILU0(SyncSplitFusion,Verbose,StoppingCriteria,ComputeNrmCorrection,"
              "ComputeNrmResidual,COOFormat,20,0.5) preconditioner\n");
    itilu0.SetAlgorithm(rocalution::SyncSplit);
    itilu0.SetOptions(0);
    itilu0.SetOperator(*A);
    itilu0.Build();
    EXPECT_EQ(head(printed(itilu0), 2),
              "ItILU0(SyncSplit,20,0.5) preconditioner\nItILU0 nnz = " + std::to_string(ref.nnz())
                  + "\n");

    rocalution::ILUT<Matrix<T>, Vector<T>, T> ilut;
    ilut.Set(0.25, 3);
    EXPECT_EQ(printed(ilut), "ILUT(0.25,3) preconditioner\n");
    ilut.SetOperator(*A);
    ilut.Build();
    EXPECT_EQ(head(printed(ilut), 1), "ILUT(0.25,3) preconditioner\n");
    EXPECT_EQ(head(printed(ilut), 2).substr(28, 11), "ILUT nnz = ");

    rocalution::IC<Matrix<T>, Vector<T>, T> ic;
    EXPECT_EQ(printed(ic), "IC preconditioner\n");
    const Csr<T> spd_ref = spd<T>(10);
    auto         S       = this->make(spd_ref);
    ic.SetOperator(*S);
    ic.Build();
    EXPECT_EQ(head(printed(ic), 2),
              "IC preconditioner\nIC nnz = " + std::to_string((spd_ref.nnz() + spd_ref.n) / 2)
                  + "\n");

    rocalution::Jacobi<Matrix<T>, Vector<T>, T>                 first;
    rocalution::VariablePreconditioner<Matrix<T>, Vector<T>, T> variable;
    EXPECT_EQ(printed(variable), "VariablePreconditioner preconditioner\n");
    Solver<T>* sequence[] = {&first};
    variable.SetPreconditioner(1, sequence);
    variable.SetOperator(*A);
    variable.Build();
    EXPECT_EQ(printed(variable),
              "VariablePreconditioner with 1 preconditioners:\nJacobi preconditioner\n");
}

TYPED_TEST(preconditioners, spai_of_a_block_diagonal_matrix_is_its_inverse)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = block_diagonal<T>(6);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    rocalution::SPAI<Matrix<T>, Vector<T>, T> p;
    p.SetOperator(*A);
    p.Build();
    expect_near(apply(p, *A, r), exact_solve(ref, r), scaled_tol<T>(1e-14));
}

TYPED_TEST(preconditioners, spai_minimises_the_residual_of_each_column)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    rocalution::SPAI<Matrix<T>, Vector<T>, T> p;
    p.SetOperator(*A);
    p.Build();
    expect_near(apply(p, *A, r), spai(ref, r), scaled_tol<T>(1e-13));
}

TYPED_TEST(preconditioners, fsai_factor_lives_on_the_pattern_of_a_power_of_the_matrix)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = spd<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    for(int power = 1; power <= 3; ++power)
    {
        SCOPED_TRACE("power " + std::to_string(power));
        rocalution::FSAI<Matrix<T>, Vector<T>, T> p;
        p.Set(power);
        p.SetOperator(*A);
        p.Build();
        expect_near(apply(p, *A, r), fsai(ref, power_pattern(ref, power), r), scaled_tol<T>(1e-13));
    }
}

// The diagonal of a tridiagonal matrix is the inverse of a 1 x 1 system
TYPED_TEST(preconditioners, fsai_factor_lives_on_an_external_pattern)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = spd<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    const Csr<T> tridiagonal = Tridiagonal<T>(12);
    auto         P           = this->make(tridiagonal);

    rocalution::FSAI<Matrix<T>, Vector<T>, T> p;
    p.Set(*P);
    p.SetOperator(*A);
    p.Build();
    expect_near(apply(p, *A, r), fsai(ref, pattern(tridiagonal), r), scaled_tol<T>(1e-13));

    KNOWN_BUG("a row of the pattern with only the diagonal keeps the pattern value unless the "
              "row of the matrix starts at the diagonal");

    Csr<T> identity(12);
    for(int i = 0; i < 12; ++i)
    {
        identity.add(i, 1.0);
        identity.end_row();
    }
    auto I = this->make(identity);

    rocalution::FSAI<Matrix<T>, Vector<T>, T> diagonal_p;
    diagonal_p.Set(*I);
    diagonal_p.SetOperator(*A);
    diagonal_p.Build();
    expect_near(apply(diagonal_p, *A, r), jacobi(ref, r), scaled_tol<T>(1e-14));
}

TYPED_TEST(preconditioners, tns_applies_the_truncated_neumann_series)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = spd<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    for(bool implicit : {true, false})
    {
        SCOPED_TRACE(implicit ? "implicit" : "explicit");
        rocalution::TNS<Matrix<T>, Vector<T>, T> p;
        p.Set(implicit);
        p.SetOperator(*A);
        p.Build();
        expect_near(apply(p, *A, r), truncated_neumann(ref, r), scaled_tol<T>(1e-14));
    }
}

TYPED_TEST(preconditioners, approximate_inverses_apply_in_every_format)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = spd<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    for(const Format& f : formats())
    {
        SCOPED_TRACE(format_name(f) + " " + std::to_string(f.blockdim));

        rocalution::FSAI<Matrix<T>, Vector<T>, T> fsai_p;
        fsai_p.SetPrecondMatrixFormat(f.id, f.blockdim);
        fsai_p.SetOperator(*A);
        fsai_p.Build();
        expect_near(apply(fsai_p, *A, r), fsai(ref, pattern(ref), r), scaled_tol<T>(1e-13));

        rocalution::SPAI<Matrix<T>, Vector<T>, T> spai_p;
        spai_p.SetPrecondMatrixFormat(f.id, f.blockdim);
        spai_p.SetOperator(*A);
        spai_p.Build();
        expect_near(apply(spai_p, *A, r), spai(ref, r), scaled_tol<T>(1e-13));

        for(bool implicit : {true, false})
        {
            rocalution::TNS<Matrix<T>, Vector<T>, T> tns_p;
            tns_p.Set(implicit);
            tns_p.SetPrecondMatrixFormat(f.id, f.blockdim);
            tns_p.SetOperator(*A);
            tns_p.Build();
            expect_near(apply(tns_p, *A, r), truncated_neumann(ref, r), scaled_tol<T>(1e-14));
        }
    }
}

TYPED_TEST(preconditioners, approximate_inverses_move_to_the_other_backend)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = spd<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);

    rocalution::FSAI<Matrix<T>, Vector<T>, T> fsai_p;
    rocalution::SPAI<Matrix<T>, Vector<T>, T> spai_p;
    rocalution::TNS<Matrix<T>, Vector<T>, T>  implicit_p;
    rocalution::TNS<Matrix<T>, Vector<T>, T>  explicit_p;
    explicit_p.Set(false);

    const std::vector<std::tuple<std::string, Solver<T>*, std::vector<T>>> cases = {
        {"FSAI", &fsai_p, fsai(ref, pattern(ref), r)},
        {"SPAI", &spai_p, spai(ref, r)},
        {"implicit TNS", &implicit_p, truncated_neumann(ref, r)},
        {"explicit TNS", &explicit_p, truncated_neumann(ref, r)},
    };

    std::vector<std::unique_ptr<Matrix<T>>> matrices;
    for(const auto& [name, p, reference] : cases)
    {
        SCOPED_TRACE(name);
        matrices.push_back(this->make(ref));
        Matrix<T>& A = *matrices.back();
        p->SetOperator(A);
        p->Build();

        move(other(this->backend()), *p, A);
        expect_near(apply(*p, A, r), reference, scaled_tol<T>(1e-13));

        // A second Build clears the first one
        move(this->backend(), *p, A);
        p->Build();
        expect_near(apply(*p, A, r), reference, scaled_tol<T>(1e-13));
    }
}

// The series of degree p + 1 approximates 1 / x on [2, 6] to within
// 2 / sqrt(12) q^(p + 2) / (1 - q) = 2.1e-5 for p = 6, with
// q = (1 - sqrt(1/3)) / (1 + sqrt(1/3)), so |z - A^-1 r| <= 2.1e-5 |r|_2 < 1e-4
// for a symmetric matrix with that spectrum
TYPED_TEST(preconditioners, aichebyshev_approximates_the_inverse)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = toeplitz<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    rocalution::AIChebyshev<Matrix<T>, Vector<T>, T> p;
    p.Set(6, 2.0, 6.0);
    p.SetOperator(*A);
    p.Build();
    const std::vector<T> z = apply(p, *A, r);

    KNOWN_BUG("the series starts from A instead of I, Z scales only the diagonal and the "
              "coefficients are halved");
    expect_near(z, exact_solve(ref, r), 1e-4);
}

// GMRES(30) solves a system of 12 unknowns with any invertible preconditioner, so the
// solves cannot see a stale or unmoved preconditioner. Its action after the build is
// compared after every step instead, which compares the library with itself and holds
// also for the buggy AIChebyshev series.
TYPED_TEST(preconditioners, aichebyshev_preconditions_gmres_after_a_rebuild_and_moves)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = toeplitz<T>(12);
    const std::vector<T> b   = rhs<T>(ref.n);
    const std::vector<T> x   = exact_solve(ref, b);
    auto                 A   = this->make(ref);

    rocalution::AIChebyshev<Matrix<T>, Vector<T>, T> p;
    p.Set(6, 2.0, 6.0);
    rocalution::GMRES<Matrix<T>, Vector<T>, T> solver;
    solver.Verbose(0);
    solver.Init(0.0, std::is_same_v<T, float> ? 1e-6 : 1e-12, 1e8, 1000);
    solver.SetOperator(*A);
    solver.SetPreconditioner(p);
    solver.Build();
    const std::vector<T> z = apply(p, *A, b);

    const auto solve = [&] {
        auto rhs_vector = make_local_vector(*A, b);
        auto result     = make_local_vector(*A, std::vector<T>(ref.n, 0));
        solver.Solve(*rhs_vector, result.get());
        return values_of(*result);
    };
    expect_near(solve(), x, loose_tol<T>(1e-9));

    p.Build();
    expect_near(apply(p, *A, b), z, scaled_tol<T>(1e-13));
    expect_near(solve(), x, loose_tol<T>(1e-9));

    for(Backend to : {other(this->backend()), this->backend()})
    {
        SCOPED_TRACE("moved to the " + backend_name(to));
        move(to, solver, *A);
        expect_near(apply(p, *A, b), z, scaled_tol<T>(1e-13));
        expect_near(solve(), x, loose_tol<T>(1e-9));
    }
}

TYPED_TEST(preconditioners, as_adds_the_weighted_block_solves)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    for(const auto& [nb, overlap] :
        std::vector<std::pair<int, int>>{{1, 0}, {2, 0}, {2, 1}, {2, 3}, {3, 0}, {4, 0}})
    {
        SCOPED_TRACE(std::to_string(nb) + " blocks, overlap " + std::to_string(overlap));
        ExactSolvers<T>                         local(nb);
        rocalution::AS<Matrix<T>, Vector<T>, T> p;
        p.Set(nb, overlap, local.get());
        p.SetOperator(*A);
        p.Build();
        expect_near(apply(p, *A, r), additive_schwarz(ref, nb, overlap, r), scaled_tol<T>(1e-13));
    }

    KNOWN_BUG("a block between two others halves the weights of its left overlap twice and "
              "those of its right overlap not at all");

    ExactSolvers<T>                         local(3);
    rocalution::AS<Matrix<T>, Vector<T>, T> p;
    p.Set(3, 1, local.get());
    p.SetOperator(*A);
    p.Build();
    expect_near(apply(p, *A, r), additive_schwarz(ref, 3, 1, r), scaled_tol<T>(1e-13));
}

TYPED_TEST(preconditioners, ras_takes_each_row_from_the_block_that_owns_it)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    for(const auto& [nb, overlap] : std::vector<std::pair<int, int>>{
            {1, 0}, {2, 0}, {2, 1}, {2, 3}, {3, 0}, {3, 1}, {3, 2}, {4, 1}})
    {
        SCOPED_TRACE(std::to_string(nb) + " blocks, overlap " + std::to_string(overlap));
        ExactSolvers<T>                          local(nb);
        rocalution::RAS<Matrix<T>, Vector<T>, T> p;
        p.Set(nb, overlap, local.get());
        p.SetOperator(*A);
        p.Build();
        expect_near(apply(p, *A, r, 9.0),
                    restricted_additive_schwarz(ref, nb, overlap, r),
                    scaled_tol<T>(1e-13));
    }
}

TYPED_TEST(preconditioners, schwarz_blocks_cover_the_rows_beyond_a_multiple_of_the_block_count)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(13);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    KNOWN_BUG("the last n % nb rows belong to no block");

    ExactSolvers<T>                         as_local(2);
    rocalution::AS<Matrix<T>, Vector<T>, T> as;
    as.Set(2, 1, as_local.get());
    as.SetOperator(*A);
    as.Build();
    expect_near(apply(as, *A, r), additive_schwarz(ref, 2, 1, r), scaled_tol<T>(1e-13));

    ExactSolvers<T>                          ras_local(2);
    rocalution::RAS<Matrix<T>, Vector<T>, T> ras;
    ras.Set(2, 1, ras_local.get());
    ras.SetOperator(*A);
    ras.Build();
    expect_near(
        apply(ras, *A, r, 9.0), restricted_additive_schwarz(ref, 2, 1, r), scaled_tol<T>(1e-13));
}

TYPED_TEST(preconditioners, schwarz_preconditioners_move_and_take_a_new_configuration)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    ExactSolvers<T>                          first(2);
    ExactSolvers<T>                          second(4);
    rocalution::RAS<Matrix<T>, Vector<T>, T> p;
    p.Set(2, 1, first.get());
    p.SetOperator(*A);
    p.Build();

    move(other(this->backend()), p, *A);
    expect_near(apply(p, *A, r), restricted_additive_schwarz(ref, 2, 1, r), scaled_tol<T>(1e-13));
    move(this->backend(), p, *A);
    expect_near(apply(p, *A, r), restricted_additive_schwarz(ref, 2, 1, r), scaled_tol<T>(1e-13));

    // Set clears the built preconditioner
    p.Set(4, 1, second.get());
    p.Build();
    expect_near(apply(p, *A, r), restricted_additive_schwarz(ref, 4, 1, r), scaled_tol<T>(1e-13));
}

TYPED_TEST(preconditioners, schwarz_preconditioners_take_the_last_configuration_before_the_build)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    ExactSolvers<T>                          first(2);
    ExactSolvers<T>                          second(3);
    rocalution::RAS<Matrix<T>, Vector<T>, T> p;
    p.Set(2, 1, first.get());
    p.Set(3, 2, second.get());
    p.SetOperator(*A);
    p.Build();
    expect_near(apply(p, *A, r), restricted_additive_schwarz(ref, 3, 2, r), scaled_tol<T>(1e-13));
}

// Clear forgets the blocks, and a second Build sets the operator of built
// block solvers
TYPED_TEST(preconditioners, schwarz_preconditioners_build_again)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);

    KNOWN_BUG("Clear discards the configuration that Build and ReBuildNumeric need");

    for(bool rebuild_numeric : {false, true})
    {
        SCOPED_TRACE(rebuild_numeric ? "ReBuildNumeric" : "Build");
        expect_in_child([&] {
            auto                                    A = this->make(ref);
            ExactSolvers<T>                         local(2);
            rocalution::AS<Matrix<T>, Vector<T>, T> p;
            p.Set(2, 1, local.get());
            p.SetOperator(*A);
            p.Build();
            if(rebuild_numeric)
            {
                p.ReBuildNumeric();
            }
            else
            {
                p.Build();
            }
            expect_near(apply(p, *A, r), additive_schwarz(ref, 2, 1, r), scaled_tol<T>(1e-13));
        });
    }
}

// The colors are independent sets, so the diagonal blocks of the permuted
// matrix are diagonal and the block sweeps are the pointwise sweeps
TYPED_TEST(preconditioners, multicolored_sgs_is_the_sgs_of_the_color_ordering)
{
    using T                     = typename TypeParam::value_type;
    const Csr<T>         ref    = nonsymmetric<T>(12);
    const std::vector<T> r      = rhs<T>(ref.n);
    auto                 A      = this->make(ref);
    const Coloring       colors = multicoloring(this->backend(), ref);
    EXPECT_GT(colors.sizes.size(), 1u);

    for(bool decomposition : {true, false})
    {
        SCOPED_TRACE(decomposition ? "decomposition" : "triangular solves");
        rocalution::MultiColoredSGS<Matrix<T>, Vector<T>, T> p;
        p.SetDecomposition(decomposition);
        p.SetOperator(*A);
        p.Build();
        expect_near(apply(p, *A, r),
                    permuted_solve(ref, colors.perm, r, symmetric_gauss_seidel<T>),
                    scaled_tol<T>(1e-14));
    }
}

TYPED_TEST(preconditioners, multicolored_gs_is_the_backward_sweep_of_the_color_ordering)
{
    using T                     = typename TypeParam::value_type;
    const Csr<T>         ref    = nonsymmetric<T>(12);
    const std::vector<T> r      = rhs<T>(ref.n);
    auto                 A      = this->make(ref);
    const Coloring       colors = multicoloring(this->backend(), ref);

    rocalution::MultiColoredGS<Matrix<T>, Vector<T>, T> p;
    p.SetOperator(*A);
    p.Build();
    expect_near(apply(p, *A, r),
                permuted_solve(ref, colors.perm, r, backward_gauss_seidel<T>),
                scaled_tol<T>(1e-14));
}

TYPED_TEST(preconditioners, multicolored_gs_without_decomposition_is_not_implemented)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);

    using MultiColoredGS = rocalution::MultiColoredGS<Matrix<T>, Vector<T>, T>;
    EXPECT_FATAL_ERROR(
        {
            auto           A = this->make(ref);
            MultiColoredGS p;
            p.SetDecomposition(false);
            p.SetOperator(*A);
            p.Build();
            apply(p, *A, r);
        },
        "preconditioner_multicolored_gs\\.cpp");
}

TYPED_TEST(preconditioners, multicolored_relaxation_is_ssor)
{
    using T                     = typename TypeParam::value_type;
    const Csr<T>         ref    = nonsymmetric<T>(12);
    const std::vector<T> r      = rhs<T>(ref.n);
    auto                 A      = this->make(ref);
    const Coloring       colors = multicoloring(this->backend(), ref);
    const T              omega  = 1.5;

    std::vector<std::vector<T>> sgs;
    for(bool decomposition : {true, false})
    {
        rocalution::MultiColoredSGS<Matrix<T>, Vector<T>, T> p;
        p.SetRelaxation(omega);
        p.SetDecomposition(decomposition);
        p.SetOperator(*A);
        p.Build();
        sgs.push_back(apply(p, *A, r));
    }

    rocalution::MultiColoredGS<Matrix<T>, Vector<T>, T> gs;
    gs.SetRelaxation(omega);
    gs.SetOperator(*A);
    gs.Build();
    const std::vector<T> gs_z = apply(gs, *A, r);

    KNOWN_BUG("the sweeps divide by omega instead of multiplying, the middle step scales by "
              "omega / (2 - omega) instead of (2 - omega) / omega, and the triangular solves "
              "ignore omega");

    for(size_t k = 0; k < sgs.size(); ++k)
    {
        SCOPED_TRACE(k == 0 ? "decomposition" : "triangular solves");
        expect_near(sgs[k],
                    permuted_solve(ref,
                                   colors.perm,
                                   r,
                                   [&](const Csr<T>& B, const std::vector<T>& b) {
                                       return ssor(B, omega, b);
                                   }),
                    scaled_tol<T>(1e-13));
    }

    expect_near(gs_z,
                permuted_solve(ref,
                               colors.perm,
                               r,
                               [&](const Csr<T>& B, const std::vector<T>& b) {
                                   return backward_sor(B, omega, b);
                               }),
                scaled_tol<T>(1e-13));
}

// GMRES(30) solves a system of 12 unknowns with any invertible preconditioner, so the
// solves cannot see a stale or unmoved preconditioner. Its action after the build is
// compared after the solve, a rebuild and the moves instead, which compares the library
// with itself and holds also for the buggy relaxed sweeps.
TYPED_TEST(preconditioners, relaxed_multicolored_gauss_seidel_preconditions_gmres)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(12);
    const std::vector<T> b   = rhs<T>(ref.n);
    const std::vector<T> x   = exact_solve(ref, b);
    auto                 A   = this->make(ref);

    rocalution::MultiColoredSGS<Matrix<T>, Vector<T>, T> sgs;
    rocalution::MultiColoredGS<Matrix<T>, Vector<T>, T>  gs;
    for(rocalution::MultiColoredSGS<Matrix<T>, Vector<T>, T>* p :
        {&sgs, static_cast<rocalution::MultiColoredSGS<Matrix<T>, Vector<T>, T>*>(&gs)})
    {
        SCOPED_TRACE(p == &sgs ? "MultiColoredSGS" : "MultiColoredGS");
        p->SetRelaxation(1.5);
        rocalution::GMRES<Matrix<T>, Vector<T>, T> solver;
        solver.Verbose(0);
        solver.Init(0.0, std::is_same_v<T, float> ? 1e-6 : 1e-12, 1e8, 1000);
        solver.SetOperator(*A);
        solver.SetPreconditioner(*p);
        solver.Build();
        const std::vector<T> z = apply(*p, *A, b);

        const auto solve = [&] {
            auto rhs_vector = make_local_vector(*A, b);
            auto result     = make_local_vector(*A, std::vector<T>(ref.n, 0));
            solver.Solve(*rhs_vector, result.get());
            return values_of(*result);
        };
        expect_near(solve(), x, loose_tol<T>(1e-9));
        expect_near(apply(*p, *A, b), z, scaled_tol<T>(1e-14));

        p->ReBuildNumeric();
        expect_near(apply(*p, *A, b), z, scaled_tol<T>(1e-14));
        expect_near(solve(), x, loose_tol<T>(1e-9));

        for(Backend to : {other(this->backend()), this->backend()})
        {
            SCOPED_TRACE("moved to the " + backend_name(to));
            move(to, solver, *A);
            expect_near(apply(*p, *A, b), z, scaled_tol<T>(1e-13));
            expect_near(solve(), x, loose_tol<T>(1e-9));
        }
    }
}

// With the colors of the pattern of A^q, q > p, the ILU(p) factors have
// diagonal blocks, so that the block solves are exact
TYPED_TEST(preconditioners, multicolored_ilu_is_the_ilu_of_the_color_ordering)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = nonsymmetric<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    struct Case
    {
        int  p;
        int  q;
        bool level;
        bool set_p_only;
    };
    for(const Case& c : std::vector<Case>{{0, 1, true, true},
                                          {1, 2, true, true},
                                          {1, 3, true, false},
                                          {1, 2, false, false},
                                          {2, 3, true, false}})
    {
        const Coloring colors
            = multicoloring(this->backend(), pattern_matrix<T>(power_pattern(ref, c.q), ref.n));
        const auto ilu = [&](const Csr<T>& B, const std::vector<T>& b) {
            return lu_solve(
                c.level ? ilu_levels(B, c.p) : ilu_on_pattern(B, power_pattern(B, c.p + 1)), b);
        };

        for(bool decomposition : {true, false})
        {
            SCOPED_TRACE("ILU(" + std::to_string(c.p) + ", " + std::to_string(c.q)
                         + (c.level ? ", levels)" : ", power pattern)")
                         + (decomposition ? ", decomposition" : ", triangular solves"));
            rocalution::MultiColoredILU<Matrix<T>, Vector<T>, T> p;
            if(c.set_p_only)
            {
                p.Set(c.p);
            }
            else
            {
                p.Set(c.p, c.q, c.level);
            }
            p.SetDecomposition(decomposition);
            p.SetOperator(*A);
            p.Build();
            expect_near(apply(p, *A, r), permuted_solve(ref, colors.perm, r, ilu), tol<T>());
        }
    }
}

TYPED_TEST(preconditioners, multicolored_blocks_apply_in_every_format)
{
    using T                     = typename TypeParam::value_type;
    const Csr<T>         ref    = nonsymmetric<T>(12);
    const std::vector<T> r      = rhs<T>(ref.n);
    auto                 A      = this->make(ref);
    const Coloring       colors = multicoloring(this->backend(), ref);
    const Coloring       colors2
        = multicoloring(this->backend(), pattern_matrix<T>(power_pattern(ref, 2), ref.n));

    // The off-diagonal color blocks of the ILU(1) factors are rectangular
    const auto check = [&](const Format& f, bool sgs_case, bool ilu_case) {
        SCOPED_TRACE(format_name(f) + " " + std::to_string(f.blockdim));
        if(sgs_case)
        {
            rocalution::MultiColoredSGS<Matrix<T>, Vector<T>, T> sgs;
            sgs.SetPrecondMatrixFormat(f.id, f.blockdim);
            sgs.SetOperator(*A);
            sgs.Build();
            expect_near(apply(sgs, *A, r),
                        permuted_solve(ref, colors.perm, r, symmetric_gauss_seidel<T>),
                        scaled_tol<T>(1e-14));
        }
        if(ilu_case)
        {
            rocalution::MultiColoredILU<Matrix<T>, Vector<T>, T> ilu;
            ilu.Set(1);
            ilu.SetPrecondMatrixFormat(f.id, f.blockdim);
            ilu.SetOperator(*A);
            ilu.Build();
            expect_near(apply(ilu, *A, r),
                        permuted_solve(ref,
                                       colors2.perm,
                                       r,
                                       [](const Csr<T>& B, const std::vector<T>& b) {
                                           return lu_solve(ilu_levels(B, 1), b);
                                       }),
                        tol<T>());
        }
    };

    const bool hip = this->backend() == Backend::Accelerator;
    for(const Format& f : formats())
    {
        const bool ignores_output = hip && (f.id == rocalution::HYB || f.id == rocalution::DENSE);
        check(f, !ignores_output, !ignores_output && f.id != rocalution::DIA);
    }

    if(hip)
    {
        KNOWN_BUG("HIP HYB and DENSE ApplyAdd ignore the output");
        check(Format{rocalution::HYB, 1}, true, true);
        check(Format{rocalution::DENSE, 1}, true, true);
    }

    KNOWN_BUG("csr_to_dia writes past the values of a matrix with more rows than columns; on the "
              "host, DIA Apply also drops the columns from nrow on");
    check(Format{rocalution::DIA, 1}, false, true);
}

TYPED_TEST(preconditioners, multicolored_preconditioners_rebuild_from_new_values)
{
    using T                      = typename TypeParam::value_type;
    const Csr<T>         ref     = nonsymmetric<T>(12);
    const Csr<T>         changed = transformed(ref, [](int, int, T v) { return 2 * v + 1; });
    const std::vector<T> r       = rhs<T>(ref.n);
    const Coloring       colors  = multicoloring(this->backend(), ref);
    const Coloring       colors2
        = multicoloring(this->backend(), pattern_matrix<T>(power_pattern(ref, 2), ref.n));
    const auto ilu1
        = [](const Csr<T>& B, const std::vector<T>& b) { return lu_solve(ilu_levels(B, 1), b); };

    {
        auto                                                 A = this->make(ref);
        rocalution::MultiColoredSGS<Matrix<T>, Vector<T>, T> sgs;
        sgs.SetOperator(*A);
        sgs.Build();
        rocalution::MultiColoredILU<Matrix<T>, Vector<T>, T> ilu;
        ilu.Set(1);
        ilu.SetOperator(*A);
        ilu.Build();

        A->CopyFromCSR(changed.row_offset.data(), changed.col.data(), changed.val.data());
        sgs.ReBuildNumeric();
        ilu.ReBuildNumeric();
        expect_near(apply(sgs, *A, r),
                    permuted_solve(changed, colors.perm, r, symmetric_gauss_seidel<T>),
                    scaled_tol<T>(1e-14));
        expect_near(apply(ilu, *A, r), permuted_solve(changed, colors2.perm, r, ilu1), tol<T>());

        // Clear allows a new Build, also of another matrix
        auto B = this->make(ref);
        sgs.Clear();
        sgs.SetOperator(*B);
        sgs.Build();
        expect_near(apply(sgs, *B, r),
                    permuted_solve(ref, colors.perm, r, symmetric_gauss_seidel<T>),
                    scaled_tol<T>(1e-14));
    }

    KNOWN_BUG("ReBuildNumeric without decomposition calls PermuteBackward, which leaves the COO "
              "entries unsorted, so the conversion back to CSR fails an assert");
    expect_in_child([&] {
        auto                                                 A = this->make(ref);
        rocalution::MultiColoredILU<Matrix<T>, Vector<T>, T> ilu;
        ilu.Set(1);
        ilu.SetDecomposition(false);
        ilu.SetOperator(*A);
        ilu.Build();
        A->CopyFromCSR(changed.row_offset.data(), changed.col.data(), changed.val.data());
        ilu.ReBuildNumeric();
        expect_near(apply(ilu, *A, r), permuted_solve(changed, colors2.perm, r, ilu1), tol<T>());
    });
}

// Nodes 0 to 3 and nodes 4 to 7 are independent sets, so the greedy coloring keeps the
// order: the permutation is the identity, and the PermuteBackward of ReBuildNumeric
// leaves the COO entries sorted, which a known bug of Permute breaks otherwise
TYPED_TEST(preconditioners, multicolored_ilu_without_decomposition_rebuilds_from_new_values)
{
    using T          = typename TypeParam::value_type;
    const int      n = 8;
    std::vector<T> D(n * n, static_cast<T>(0));
    for(int i = 0; i < 4; ++i)
    {
        D[i * n + i]               = static_cast<T>(8 + i % 3);
        D[(i + 4) * n + i + 4]     = static_cast<T>(9 - i % 2);
        D[i * n + 4 + i]           = -1;
        D[i * n + 4 + (i + 1) % 4] = -2;
        D[(i + 4) * n + i]         = -2;
    }
    const Csr<T>         ref     = from_dense(n, n, D);
    const Csr<T>         changed = transformed(ref, [](int, int, T v) { return 2 * v + 1; });
    const std::vector<T> r       = rhs<T>(n);
    const Coloring       colors  = multicoloring(this->backend(), ref);
    std::vector<int>     identity(n);
    std::iota(identity.begin(), identity.end(), 0);
    ASSERT_EQ(colors.perm, identity);
    ASSERT_EQ(colors.sizes, (std::vector<int>{4, 4}));

    auto                                                 A = this->make(ref);
    rocalution::MultiColoredILU<Matrix<T>, Vector<T>, T> ilu;
    ilu.Set(0);
    ilu.SetDecomposition(false);
    ilu.SetOperator(*A);
    ilu.Build();
    expect_near(apply(ilu, *A, r), lu_solve(ilu_levels(ref, 0), r), tol<T>());

    A->CopyFromCSR(changed.row_offset.data(), changed.col.data(), changed.val.data());
    ilu.ReBuildNumeric();
    expect_near(apply(ilu, *A, r), lu_solve(ilu_levels(changed, 0), r), tol<T>());
}

TYPED_TEST(preconditioners, multicolored_preconditioners_move_to_the_other_backend)
{
    using T                     = typename TypeParam::value_type;
    const Csr<T>         ref    = nonsymmetric<T>(12);
    const std::vector<T> r      = rhs<T>(ref.n);
    auto                 A      = this->make(ref);
    const Coloring       colors = multicoloring(this->backend(), ref);
    const Coloring       colors2
        = multicoloring(this->backend(), pattern_matrix<T>(power_pattern(ref, 2), ref.n));
    const std::vector<T> sgs_expected
        = permuted_solve(ref, colors.perm, r, symmetric_gauss_seidel<T>);
    const std::vector<T> gs_expected
        = permuted_solve(ref, colors.perm, r, backward_gauss_seidel<T>);
    const std::vector<T> ilu_expected
        = permuted_solve(ref, colors2.perm, r, [](const Csr<T>& B, const std::vector<T>& b) {
              return lu_solve(ilu_levels(B, 1), b);
          });

    rocalution::MultiColoredSGS<Matrix<T>, Vector<T>, T> sgs;
    sgs.SetOperator(*A);
    sgs.Build();
    rocalution::MultiColoredGS<Matrix<T>, Vector<T>, T> gs;
    gs.SetOperator(*A);
    gs.Build();
    rocalution::MultiColoredILU<Matrix<T>, Vector<T>, T> ilu;
    ilu.Set(1);
    ilu.SetOperator(*A);
    ilu.Build();

    for(Backend to : {other(this->backend()), this->backend()})
    {
        SCOPED_TRACE("moved to the " + backend_name(to));
        move(to, sgs, *A);
        move(to, gs, *A);
        move(to, ilu, *A);
        expect_near(apply(sgs, *A, r), sgs_expected, scaled_tol<T>(1e-14));
        expect_near(apply(gs, *A, r), gs_expected, scaled_tol<T>(1e-14));
        expect_near(apply(ilu, *A, r), ilu_expected, tol<T>());
    }

    const auto moved_triangular_solves = [&](Backend to) {
        SCOPED_TRACE("triangular solves, moved to the " + backend_name(to));
        expect_in_child([&] {
            auto                                                 B = this->make(ref);
            rocalution::MultiColoredSGS<Matrix<T>, Vector<T>, T> sgs_solves;
            sgs_solves.SetDecomposition(false);
            sgs_solves.SetOperator(*B);
            sgs_solves.Build();
            move(to, sgs_solves, *B);
            expect_near(apply(sgs_solves, *B, r), sgs_expected, scaled_tol<T>(1e-14));
        });
        expect_in_child([&] {
            auto                                                 B = this->make(ref);
            rocalution::MultiColoredILU<Matrix<T>, Vector<T>, T> ilu_solves;
            ilu_solves.Set(1);
            ilu_solves.SetDecomposition(false);
            ilu_solves.SetOperator(*B);
            ilu_solves.Build();
            move(to, ilu_solves, *B);
            expect_near(apply(ilu_solves, *B, r), ilu_expected, tol<T>());
        });
    };
    moved_triangular_solves(this->backend());

    KNOWN_BUG("without decomposition, a move neither analyses the factors again nor moves the "
              "diagonal of SGS");
    moved_triangular_solves(other(this->backend()));
}

// The center and the leaves of a star have two colors, and the entries of a
// triangular star lie in one of the two off-diagonal color blocks only
TYPED_TEST(preconditioners, multicolored_gauss_seidel_skips_an_empty_color_block)
{
    using T     = typename TypeParam::value_type;
    const int n = 5;
    for(bool lower : {true, false})
    {
        SCOPED_TRACE(lower ? "lower triangular star" : "upper triangular star");
        Csr<T> ref(n);
        for(int i = 0; i < n; ++i)
        {
            if(lower && i > 0)
            {
                ref.add(0, -1.0);
            }
            ref.add(i, 4.0);
            for(int j = 1; !lower && i == 0 && j < n; ++j)
            {
                ref.add(j, -1.0);
            }
            ref.end_row();
        }
        const std::vector<T> r      = rhs<T>(n);
        const Coloring       colors = multicoloring(this->backend(), ref);
        EXPECT_EQ(colors.sizes.size(), 2u);
        auto A = this->make(ref);

        rocalution::MultiColoredSGS<Matrix<T>, Vector<T>, T> sgs;
        sgs.SetOperator(*A);
        sgs.Build();
        expect_near(apply(sgs, *A, r),
                    permuted_solve(ref, colors.perm, r, symmetric_gauss_seidel<T>),
                    scaled_tol<T>(1e-14));

        rocalution::MultiColoredGS<Matrix<T>, Vector<T>, T> gs;
        gs.SetOperator(*A);
        gs.Build();
        expect_near(apply(gs, *A, r),
                    permuted_solve(ref, colors.perm, r, backward_gauss_seidel<T>),
                    scaled_tol<T>(1e-14));
    }
}

// D is the diagonal of the independent set, so with an exact solve of the Schur
// complement S = C - E D^-1 F the block LU solve is exact
TYPED_TEST(preconditioners, multielimination_with_an_exact_schur_solve_is_exact)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = spd<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    int size = 0;
    independent_set(this->backend(), ref, size);
    EXPECT_GT(size, 0);

    for(int level : {0, 1, 2, 3})
    {
        SCOPED_TRACE("level " + std::to_string(level));
        rocalution::LU<Matrix<T>, Vector<T>, T> lu;
        lu.Verbose(0);
        rocalution::MultiElimination<Matrix<T>, Vector<T>, T> p;
        p.Set(lu, level);
        p.SetOperator(*A);
        p.Build();
        EXPECT_EQ(p.GetSizeDiagBlock(), size);
        EXPECT_EQ(p.GetLevel(), level);
        expect_near(apply(p, *A, r), exact_solve(ref, r), tol<T>());
    }
}

// The Schur complement keeps the entries above the drop-off and its diagonal
TYPED_TEST(preconditioners, multielimination_drops_small_schur_complement_entries)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = spd<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    int                    size = 0;
    const std::vector<int> perm = independent_set(this->backend(), ref, size);

    double     drop_off  = 0.0;
    const auto reference = [&](const Csr<T>& B, const std::vector<T>& b) {
        const int            n = B.n;
        const int            m = n - size;
        const std::vector<T> P = dense(B);
        std::vector<T>       S(m * m);
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < m; ++j)
            {
                T v = P[(size + i) * n + size + j];
                for(int k = 0; k < size; ++k)
                {
                    v -= P[(size + i) * n + k] * P[(size + j) + k * n] / P[k * n + k];
                }
                S[i * m + j] = (std::abs(v) > drop_off || i == j) ? v : T(0);
            }
        }
        std::vector<T> r2(b.begin() + size, b.end());
        for(int i = 0; i < m; ++i)
        {
            for(int k = 0; k < size; ++k)
            {
                r2[i] -= P[(size + i) * n + k] * b[k] / P[k * n + k];
            }
        }
        const std::vector<T> x2 = dense_solve(S, m, r2);
        std::vector<T>       x(n);
        for(int k = 0; k < size; ++k)
        {
            T v = b[k];
            for(int j = 0; j < m; ++j)
            {
                v -= P[k * n + size + j] * x2[j];
            }
            x[k] = v / P[k * n + k];
        }
        std::copy(x2.begin(), x2.end(), x.begin() + size);
        return x;
    };

    for(double d : {0.35, 0.4, 1.0})
    {
        SCOPED_TRACE("drop-off " + std::to_string(d));
        drop_off                      = d;
        const std::vector<T> expected = permuted_solve(ref, perm, r, reference);
        if(d == 1.0)
        {
            EXPECT_GT(max_difference(expected, exact_solve(ref, r)), 1e-3);
        }

        rocalution::LU<Matrix<T>, Vector<T>, T> lu;
        lu.Verbose(0);
        rocalution::MultiElimination<Matrix<T>, Vector<T>, T> p;
        p.Set(lu, 1, drop_off);
        p.SetOperator(*A);
        p.Build();
        expect_near(apply(p, *A, r), expected, tol<T>());
    }
}

TYPED_TEST(preconditioners, multielimination_blocks_apply_in_every_format_and_move)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = spd<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    const auto check = [&](const Format& f) {
        SCOPED_TRACE(format_name(f) + " " + std::to_string(f.blockdim));
        rocalution::LU<Matrix<T>, Vector<T>, T> lu;
        lu.Verbose(0);
        rocalution::MultiElimination<Matrix<T>, Vector<T>, T> p;
        p.Set(lu, 1);
        p.SetPrecondMatrixFormat(f.id, f.blockdim);
        p.SetOperator(*A);
        p.Build();
        expect_near(apply(p, *A, r), exact_solve(ref, r), tol<T>());
    };

    const bool hip = this->backend() == Backend::Accelerator;
    for(const Format& f : formats())
    {
        if(!(hip && (f.id == rocalution::HYB || f.id == rocalution::DENSE)))
        {
            check(f);
        }
    }

    {
        auto                                    B = this->make(ref);
        rocalution::LU<Matrix<T>, Vector<T>, T> lu;
        lu.Verbose(0);
        rocalution::MultiElimination<Matrix<T>, Vector<T>, T> p;
        p.Set(lu, 1);
        p.SetOperator(*B);
        p.Build();

        move(other(this->backend()), p, *B);
        expect_near(apply(p, *B, r), exact_solve(ref, r), tol<T>());
        move(this->backend(), p, *B);
        expect_near(apply(p, *B, r), exact_solve(ref, r), tol<T>());
    }

    if(hip)
    {
        KNOWN_BUG("HIP HYB and DENSE ApplyAdd ignore the output");
        check(Format{rocalution::HYB, 1});
        check(Format{rocalution::DENSE, 1});
    }
}

// A move reaches the lower level twice, through AA_me_ and AA_solver_, so the LU of
// the last level is moved and analysed twice
TYPED_TEST(preconditioners, multielimination_with_two_levels_moves_to_the_other_backend)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = spd<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);

    auto                                    A = this->make(ref);
    rocalution::LU<Matrix<T>, Vector<T>, T> lu;
    lu.Verbose(0);
    rocalution::MultiElimination<Matrix<T>, Vector<T>, T> p;
    p.Set(lu, 2);
    p.SetOperator(*A);
    p.Build();

    move(other(this->backend()), p, *A);
    expect_near(apply(p, *A, r), exact_solve(ref, r), tol<T>());
    move(this->backend(), p, *A);
    expect_near(apply(p, *A, r), exact_solve(ref, r), tol<T>());
}

// Moving the inverse of the last level twice does no harm, unlike the analysis of LU
TYPED_TEST(preconditioners, multielimination_with_two_levels_and_an_inversion_moves)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = spd<T>(12);
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    rocalution::Inversion<Matrix<T>, Vector<T>, T> inversion;
    inversion.Verbose(0);
    rocalution::MultiElimination<Matrix<T>, Vector<T>, T> p;
    p.Set(inversion, 2);
    p.SetOperator(*A);
    p.Build();
    EXPECT_EQ(p.GetLevel(), 2);

    for(Backend to : {other(this->backend()), this->backend()})
    {
        SCOPED_TRACE("moved to the " + backend_name(to));
        move(to, p, *A);
        expect_near(apply(p, *A, r), exact_solve(ref, r), tol<T>());
    }
}

// The saddle point preconditioner solves with K = A(I1, I1) on the rows with a
// stored diagonal I1, and with S = A(I2, I1) diag(K)^-1 A(I1, I2) on the others
TYPED_TEST(preconditioners, saddle_point_solves_the_two_diagonal_blocks)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = saddle_point<T>();
    const std::vector<T> r   = rhs<T>(ref.n);
    auto                 A   = this->make(ref);

    const int            n = ref.n;
    const std::vector<T> D = dense(ref);
    std::vector<int>     I1, I2;
    for(int i = 0; i < n; ++i)
    {
        (D[i * n + i] != 0 ? I1 : I2).push_back(i);
    }
    const int n1 = static_cast<int>(I1.size());
    const int n2 = static_cast<int>(I2.size());

    std::vector<T> K(n1 * n1), S(n2 * n2, 0.0), r1(n1), r2(n2);
    for(int a = 0; a < n1; ++a)
    {
        for(int b = 0; b < n1; ++b)
        {
            K[a * n1 + b] = D[I1[a] * n + I1[b]];
        }
        r1[a] = r[I1[a]];
    }
    for(int a = 0; a < n2; ++a)
    {
        for(int b = 0; b < n2; ++b)
        {
            for(int k = 0; k < n1; ++k)
            {
                S[a * n2 + b] += D[I2[a] * n + I1[k]] * D[I1[k] * n + I2[b]] / K[k * n1 + k];
            }
        }
        r2[a] = r[I2[a]];
    }
    const std::vector<T> x1 = dense_solve(K, n1, r1);
    const std::vector<T> x2 = dense_solve(S, n2, r2);
    std::vector<T>       expected(n);
    for(int a = 0; a < n1; ++a)
    {
        expected[I1[a]] = x1[a];
    }
    for(int a = 0; a < n2; ++a)
    {
        expected[I2[a]] = x2[a];
    }

    rocalution::LU<Matrix<T>, Vector<T>, T> k_solver;
    rocalution::LU<Matrix<T>, Vector<T>, T> s_solver;
    k_solver.Verbose(0);
    s_solver.Verbose(0);
    rocalution::DiagJacobiSaddlePointPrecond<Matrix<T>, Vector<T>, T> p;
    p.Set(k_solver, s_solver);
    p.SetOperator(*A);
    p.Build();
    expect_near(apply(p, *A, r), expected, scaled_tol<T>(1e-13));

    move(other(this->backend()), p, *A);
    expect_near(apply(p, *A, r), expected, scaled_tol<T>(1e-13));
    move(this->backend(), p, *A);
    expect_near(apply(p, *A, r), expected, scaled_tol<T>(1e-13));
}

TYPED_TEST(preconditioners, multielimination_and_saddle_point_rebuild)
{
    using T                = typename TypeParam::value_type;
    const std::vector<T> r = rhs<T>(12);

    KNOWN_BUG("Clear discards the configuration that Build and ReBuildNumeric need");

    SCOPED_TRACE("MultiElimination");
    expect_in_child([&] {
        const Csr<T>                                          ref = spd<T>(12);
        auto                                                  A   = this->make(ref);
        rocalution::LU<Matrix<T>, Vector<T>, T>               lu;
        rocalution::MultiElimination<Matrix<T>, Vector<T>, T> p;
        lu.Verbose(0);
        p.Set(lu, 1);
        p.SetOperator(*A);
        p.Build();
        p.ReBuildNumeric();
        expect_near(apply(p, *A, r), exact_solve(ref, r), tol<T>());
    });

    SCOPED_TRACE("DiagJacobiSaddlePointPrecond");
    expect_in_child([&] {
        const Csr<T>                                                      ref = saddle_point<T>();
        auto                                                              A   = this->make(ref);
        rocalution::LU<Matrix<T>, Vector<T>, T>                           k_solver;
        rocalution::LU<Matrix<T>, Vector<T>, T>                           s_solver;
        rocalution::DiagJacobiSaddlePointPrecond<Matrix<T>, Vector<T>, T> p;
        k_solver.Verbose(0);
        s_solver.Verbose(0);
        p.Set(k_solver, s_solver);
        p.SetOperator(*A);
        p.Build();
        const std::vector<T> before = apply(p, *A, r);
        p.ReBuildNumeric();
        expect_near(apply(p, *A, r), before, scaled_tol<T>(1e-13));
    });
}

TYPED_TEST(preconditioners, block_preconditioner_is_a_block_gauss_seidel_sweep)
{
    using T                      = typename TypeParam::value_type;
    const Csr<T>           ref   = nonsymmetric<T>(12);
    const std::vector<T>   r     = rhs<T>(ref.n);
    auto                   A     = this->make(ref);
    const std::vector<int> sizes = {3, 5, 4};

    for(bool diagonal_only : {false, true})
    {
        SCOPED_TRACE(diagonal_only ? "block diagonal" : "block lower");
        ExactSolvers<T>                                          local(3);
        rocalution::BlockPreconditioner<Matrix<T>, Vector<T>, T> p;
        p.Set(3, sizes.data(), local.get());
        if(diagonal_only)
        {
            p.SetDiagonalSolver();
        }
        else
        {
            p.SetLSolver();
        }
        p.SetOperator(*A);
        p.Build();
        expect_near(apply(p, *A, r),
                    block_lower_solve(ref, sizes, diagonal_only, {}, r),
                    scaled_tol<T>(1e-13));

        move(other(this->backend()), p, *A);
        expect_near(apply(p, *A, r),
                    block_lower_solve(ref, sizes, diagonal_only, {}, r),
                    scaled_tol<T>(1e-13));
        move(this->backend(), p, *A);
        expect_near(apply(p, *A, r),
                    block_lower_solve(ref, sizes, diagonal_only, {}, r),
                    scaled_tol<T>(1e-13));
    }
}

TYPED_TEST(preconditioners, block_preconditioner_takes_a_permutation_and_a_last_block)
{
    using T                      = typename TypeParam::value_type;
    const Csr<T>           ref   = nonsymmetric<T>(12);
    const std::vector<T>   r     = rhs<T>(ref.n);
    auto                   A     = this->make(ref);
    const std::vector<int> sizes = {4, 4, 4};
    const std::vector<int> perm  = {5, 0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10};

    // The permuted matrix, P A P^T, is split into the blocks
    {
        ExactSolvers<T>                                          local(3);
        rocalution::BlockPreconditioner<Matrix<T>, Vector<T>, T> p;
        auto                                                     P = make_local_vector(*A, perm);
        p.Set(3, sizes.data(), local.get());
        p.SetPermutation(*P);
        p.SetOperator(*A);
        p.Build();
        expect_near(apply(p, *A, r),
                    permuted_solve(ref,
                                   perm,
                                   r,
                                   [&](const Csr<T>& B, const std::vector<T>& b) {
                                       return block_lower_solve(B, sizes, false, {}, b);
                                   }),
                    scaled_tol<T>(1e-13));

        for(Backend to : {other(this->backend()), this->backend()})
        {
            SCOPED_TRACE("moved to the " + backend_name(to));
            move(to, p, *A);
            expect_near(apply(p, *A, r),
                        permuted_solve(ref,
                                       perm,
                                       r,
                                       [&](const Csr<T>& B, const std::vector<T>& b) {
                                           return block_lower_solve(B, sizes, false, {}, b);
                                       }),
                        scaled_tol<T>(1e-13));
        }
    }

    // The last diagonal block is replaced by an external matrix
    {
        const Csr<T> last = Tridiagonal<T>(4);
        auto         L    = this->make(last);

        ExactSolvers<T>                                          local(3);
        rocalution::BlockPreconditioner<Matrix<T>, Vector<T>, T> p;
        p.Set(3, sizes.data(), local.get());
        p.SetExternalLastMatrix(*L);
        p.SetOperator(*A);
        p.Build();
        expect_near(apply(p, *A, r),
                    block_lower_solve(ref, sizes, false, dense(last), r),
                    scaled_tol<T>(1e-13));
    }
}

TYPED_TEST(preconditioners, block_preconditioner_rebuilds)
{
    using T                      = typename TypeParam::value_type;
    const Csr<T>           ref   = nonsymmetric<T>(12);
    const std::vector<T>   r     = rhs<T>(ref.n);
    const std::vector<int> sizes = {6, 6};

    KNOWN_BUG("Clear discards the configuration that Build and ReBuildNumeric need");

    expect_in_child([&] {
        auto                                                     A = this->make(ref);
        ExactSolvers<T>                                          local(2);
        rocalution::BlockPreconditioner<Matrix<T>, Vector<T>, T> p;
        p.Set(2, sizes.data(), local.get());
        p.SetOperator(*A);
        p.Build();
        p.ReBuildNumeric();
        expect_near(
            apply(p, *A, r), block_lower_solve(ref, sizes, false, {}, r), scaled_tol<T>(1e-13));
    });
}

// Clear and the moves of a preconditioner that is not built keep its
// configuration, and Build then builds it on the backend of the operator
TYPED_TEST(preconditioners, preconditioners_cleared_and_moved_before_their_build_build_as_usual)
{
    using T                       = typename TypeParam::value_type;
    const Csr<T>           ref    = spd<T>(12);
    const std::vector<T>   r      = rhs<T>(ref.n);
    const Coloring         colors = multicoloring(this->backend(), ref);
    const std::vector<int> sizes  = {3, 5, 4};

    // The preconditioners clear their local solvers when they are destroyed
    std::vector<std::unique_ptr<Matrix<T>>>                  matrices;
    ExactSolvers<T>                                          ras_local(2);
    ExactSolvers<T>                                          block_local(3);
    rocalution::FSAI<Matrix<T>, Vector<T>, T>                fsai_p;
    rocalution::SPAI<Matrix<T>, Vector<T>, T>                spai_p;
    rocalution::TNS<Matrix<T>, Vector<T>, T>                 tns_p;
    rocalution::RAS<Matrix<T>, Vector<T>, T>                 ras_p;
    rocalution::MultiColoredSGS<Matrix<T>, Vector<T>, T>     sgs_p;
    rocalution::BlockPreconditioner<Matrix<T>, Vector<T>, T> block_p;
    ras_p.Set(2, 1, ras_local.get());
    block_p.Set(3, sizes.data(), block_local.get());

    const std::vector<std::tuple<std::string, Solver<T>*, std::vector<T>>> cases = {
        {"FSAI", &fsai_p, fsai(ref, pattern(ref), r)},
        {"SPAI", &spai_p, spai(ref, r)},
        {"TNS", &tns_p, truncated_neumann(ref, r)},
        {"RAS", &ras_p, restricted_additive_schwarz(ref, 2, 1, r)},
        {"MultiColoredSGS", &sgs_p, permuted_solve(ref, colors.perm, r, symmetric_gauss_seidel<T>)},
        {"BlockPreconditioner", &block_p, block_lower_solve(ref, sizes, false, {}, r)},
    };

    for(const auto& [name, p, reference] : cases)
    {
        SCOPED_TRACE(name);
        p->Clear();
        p->MoveToAccelerator();
        p->MoveToHost();

        matrices.push_back(this->make(ref));
        p->SetOperator(*matrices.back());
        p->Build();
        expect_near(apply(*p, *matrices.back(), r), reference, scaled_tol<T>(1e-13));
    }
}
