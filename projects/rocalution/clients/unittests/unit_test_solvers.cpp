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

// The solver classes on LocalMatrix and LocalVector: the Krylov solvers, FixedPoint,
// Chebyshev, the direct solvers and MixedPrecisionDC, with the stopping rules of
// IterationControl. The iterates after a fixed number of iterations are compared with
// their definitions, computed densely on small systems: the Galerkin and minimal
// residual conditions over the Krylov space, and the published recurrences for the
// methods without such a condition.

#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "unit_test_common.hpp"
#include "unit_test_local.hpp"
#include "unit_test_reference.hpp"
#include "unit_test_types.hpp"

#include "base/local_matrix.hpp"
#include "base/local_stencil.hpp"
#include "base/local_vector.hpp"
#include "base/stencil_types.hpp"
#include "solvers/chebyshev.hpp"
#include "solvers/direct/inversion.hpp"
#include "solvers/direct/lu.hpp"
#include "solvers/direct/qr.hpp"
#include "solvers/iter_ctrl.hpp"
#include "solvers/krylov/bicgstab.hpp"
#include "solvers/krylov/bicgstabl.hpp"
#include "solvers/krylov/cg.hpp"
#include "solvers/krylov/cr.hpp"
#include "solvers/krylov/fcg.hpp"
#include "solvers/krylov/fgmres.hpp"
#include "solvers/krylov/gmres.hpp"
#include "solvers/krylov/idr.hpp"
#include "solvers/krylov/qmrcgstab.hpp"
#include "solvers/mixed_precision.hpp"
#include "solvers/preconditioners/preconditioner.hpp"
#include "solvers/solver.hpp"

using namespace rocalution_unit;
using rocalution::LocalMatrix;
using rocalution::LocalVector;

namespace
{
    template <typename T>
    using Matrix = LocalMatrix<T>;
    template <typename T>
    using Vector = LocalVector<T>;
    template <typename T>
    using Solver = rocalution::Solver<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using Iterative = rocalution::IterativeLinearSolver<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using Jacobi = rocalution::Jacobi<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using CG = rocalution::CG<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using CR = rocalution::CR<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using FCG = rocalution::FCG<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using GMRES = rocalution::GMRES<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using FGMRES = rocalution::FGMRES<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using BiCGStab = rocalution::BiCGStab<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using BiCGStabl = rocalution::BiCGStabl<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using IDR = rocalution::IDR<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using QMRCGStab = rocalution::QMRCGStab<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using FixedPoint = rocalution::FixedPoint<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using Chebyshev = rocalution::Chebyshev<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using LU = rocalution::LU<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using QR = rocalution::QR<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using Inversion = rocalution::Inversion<Matrix<T>, Vector<T>, T>;

    // The ValueType of a solver class template <OperatorType, VectorType, ValueType>
    template <typename S>
    struct value_type_of;

    template <template <typename, typename, typename> class S, typename M, typename V, typename T>
    struct value_type_of<S<M, V, T>>
    {
        using type = T;
    };

    // Keeps a parameter out of template argument deduction, so that it takes literals,
    // nullptr, functions and lambdas
    template <typename T>
    struct nondeduced
    {
        using type = T;
    };

    template <typename T>
    using nondeduced_t = typename nondeduced<T>::type;

    template <typename T>
    using Vec = std::vector<T>;
    template <typename T>
    using Op = std::function<Vec<T>(const Vec<T>&)>;

    // Dense reference arithmetic in T. Matrices are row major n x n.

    template <typename T>
    T dot(const Vec<T>& a, const Vec<T>& b)
    {
        T s = 0;
        for(size_t i = 0; i < a.size(); ++i)
        {
            s += a[i] * b[i];
        }
        return s;
    }

    template <typename T>
    T norm(const Vec<T>& a)
    {
        return std::sqrt(dot(a, a));
    }

    // a + s b
    template <typename T>
    Vec<T> plus(const Vec<T>& a, nondeduced_t<T> s, const Vec<T>& b)
    {
        Vec<T> c = a;
        for(size_t i = 0; i < c.size(); ++i)
        {
            c[i] += s * b[i];
        }
        return c;
    }

    template <typename T>
    Vec<T> scaled(nondeduced_t<T> s, const Vec<T>& a)
    {
        return plus(Vec<T>(a.size(), 0), s, a);
    }

    template <typename T>
    Vec<T> multiply(const Vec<T>& A, const Vec<T>& x)
    {
        const size_t n = x.size();
        Vec<T>       y(n, 0);
        for(size_t i = 0; i < n; ++i)
        {
            for(size_t j = 0; j < n; ++j)
            {
                y[i] += A[i * n + j] * x[j];
            }
        }
        return y;
    }

    template <typename T>
    Vec<T> residual(const Vec<T>& A, const Vec<T>& b, const Vec<T>& x)
    {
        return plus(b, -1, multiply(A, x));
    }

    // Gaussian elimination with partial pivoting
    template <typename T>
    Vec<T> dense_solve(Vec<T> A, Vec<T> b)
    {
        const int n = static_cast<int>(b.size());
        for(int k = 0; k < n; ++k)
        {
            int p = k;
            for(int i = k + 1; i < n; ++i)
            {
                if(std::abs(A[i * n + k]) > std::abs(A[p * n + k]))
                {
                    p = i;
                }
            }
            for(int j = 0; j < n; ++j)
            {
                std::swap(A[k * n + j], A[p * n + j]);
            }
            std::swap(b[k], b[p]);

            for(int i = k + 1; i < n; ++i)
            {
                const T l = A[i * n + k] / A[k * n + k];
                for(int j = k; j < n; ++j)
                {
                    A[i * n + j] -= l * A[k * n + j];
                }
                b[i] -= l * b[k];
            }
        }

        Vec<T> x(n);
        for(int i = n - 1; i >= 0; --i)
        {
            T s = b[i];
            for(int j = i + 1; j < n; ++j)
            {
                s -= A[i * n + j] * x[j];
            }
            x[i] = s / A[i * n + i];
        }
        return x;
    }

    // Orthonormal basis of span{v, K v, ..., K^(k-1) v}
    template <typename T>
    std::vector<Vec<T>> krylov_basis(const nondeduced_t<Op<T>>& K, const Vec<T>& v, int k)
    {
        std::vector<Vec<T>> basis;
        Vec<T>              w = v;
        for(int j = 0; j < k; ++j)
        {
            if(j > 0)
            {
                w = K(basis.back());
            }
            for(int pass = 0; pass < 2; ++pass)
            {
                for(const Vec<T>& q : basis)
                {
                    w = plus(w, -dot(w, q), q);
                }
            }
            basis.push_back(scaled(1 / norm(w), w));
        }
        return basis;
    }

    // The y with minimal |B y - c|_2, for the columns B
    template <typename T>
    Vec<T> least_squares(std::vector<Vec<T>> B, const Vec<T>& c)
    {
        const int k = static_cast<int>(B.size());
        Vec<T>    R(k * k, 0);
        for(int j = 0; j < k; ++j)
        {
            for(int pass = 0; pass < 2; ++pass)
            {
                for(int i = 0; i < j; ++i)
                {
                    const T r = dot(B[i], B[j]);
                    R[i * k + j] += r;
                    B[j] = plus(B[j], -r, B[i]);
                }
            }
            R[j * k + j] = norm(B[j]);
            B[j]         = scaled(1 / R[j * k + j], B[j]);
        }

        Vec<T> y(k);
        for(int i = k - 1; i >= 0; --i)
        {
            T s = dot(B[i], c);
            for(int j = i + 1; j < k; ++j)
            {
                s -= R[i * k + j] * y[j];
            }
            y[i] = s / R[i * k + i];
        }
        return y;
    }

    template <typename T>
    Vec<T> combination(const Vec<T>& x0, const std::vector<Vec<T>>& V, const Vec<T>& y)
    {
        Vec<T> x = x0;
        for(size_t j = 0; j < V.size(); ++j)
        {
            x = plus(x, y[j], V[j]);
        }
        return x;
    }

    template <typename T>
    Vec<T> identity(const Vec<T>& v)
    {
        return v;
    }

    // M^-1 for the Jacobi preconditioner M = diag(A), or the identity
    template <typename T>
    Op<T> preconditioner(const Vec<T>& A, bool jacobi)
    {
        if(!jacobi)
        {
            return identity<T>;
        }
        return [A](const Vec<T>& r) {
            const size_t n = r.size();
            Vec<T>       z(n);
            for(size_t i = 0; i < n; ++i)
            {
                z[i] = r[i] / A[i * n + i];
            }
            return z;
        };
    }

    // M^-1/2 for the Jacobi preconditioner, or the identity
    template <typename T>
    Op<T> root_of_preconditioner(const Vec<T>& A, bool jacobi)
    {
        if(!jacobi)
        {
            return identity<T>;
        }
        return [A](const Vec<T>& r) {
            const size_t n = r.size();
            Vec<T>       z(n);
            for(size_t i = 0; i < n; ++i)
            {
                z[i] = r[i] / std::sqrt(A[i * n + i]);
            }
            return z;
        };
    }

    template <typename T>
    std::vector<Vec<T>>
        krylov_space(const Vec<T>& A, const Vec<T>& r0, int k, const nondeduced_t<Op<T>>& M)
    {
        return krylov_basis<T>([&](const Vec<T>& v) { return M(multiply(A, v)); }, M(r0), k);
    }

    // The k-th iterate whose error is A-orthogonal to the Krylov space of M^-1 A and
    // M^-1 r0, so that the A-norm of the error is minimal over x0 plus that space
    template <typename T>
    Vec<T> galerkin_iterate(
        const Vec<T>& A, const Vec<T>& b, const Vec<T>& x0, int k, const nondeduced_t<Op<T>>& M)
    {
        const Vec<T>              r0 = residual(A, b, x0);
        const std::vector<Vec<T>> V  = krylov_space(A, r0, k, M);

        Vec<T> G(k * k), g(k);
        for(int i = 0; i < k; ++i)
        {
            for(int j = 0; j < k; ++j)
            {
                G[i * k + j] = dot(V[i], multiply(A, V[j]));
            }
            g[i] = dot(V[i], r0);
        }
        return combination(x0, V, dense_solve(G, g));
    }

    // The k-th iterate with minimal |W (b - A x)|_2 over x0 plus the Krylov space of
    // M^-1 A and M^-1 r0
    template <typename T>
    Vec<T> min_residual_iterate(const Vec<T>&              A,
                                const Vec<T>&              b,
                                const Vec<T>&              x0,
                                int                        k,
                                const nondeduced_t<Op<T>>& M,
                                const nondeduced_t<Op<T>>& W)
    {
        const Vec<T>              r0 = residual(A, b, x0);
        const std::vector<Vec<T>> V  = krylov_space(A, r0, k, M);

        std::vector<Vec<T>> B;
        for(const Vec<T>& v : V)
        {
            B.push_back(W(multiply(A, v)));
        }
        return combination(x0, V, least_squares(B, W(r0)));
    }

    // The k-th FGMRES iterate with a preconditioner P that may change from step to step:
    // x0 + Z y with z_j = P(v_j) for the Arnoldi vectors v_j of A Z, and the y with
    // minimal |r0 - A Z y|_2
    template <typename T>
    Vec<T> flexible_iterate(
        const Vec<T>& A, const Vec<T>& b, const Vec<T>& x0, int k, const nondeduced_t<Op<T>>& P)
    {
        const Vec<T>        r0 = residual(A, b, x0);
        std::vector<Vec<T>> V  = {scaled(1 / norm(r0), r0)};
        std::vector<Vec<T>> Z, AZ;
        for(int j = 0; j < k; ++j)
        {
            Z.push_back(P(V.back()));
            AZ.push_back(multiply(A, Z.back()));
            Vec<T> w = AZ.back();
            for(int pass = 0; pass < 2; ++pass)
            {
                for(const Vec<T>& q : V)
                {
                    w = plus(w, -dot(w, q), q);
                }
            }
            V.push_back(scaled(1 / norm(w), w));
        }
        return combination(x0, Z, least_squares(AZ, r0));
    }

    // k iterations of a method that restarts every m iterations
    template <typename T>
    Vec<T> restarted(const nondeduced_t<std::function<Vec<T>(const Vec<T>&, int)>>& cycle,
                     Vec<T>                                                         x,
                     int                                                            k,
                     int                                                            m)
    {
        for(int done = 0; done < k; done += m)
        {
            x = cycle(x, std::min(m, k - done));
        }
        return x;
    }

    // Van der Vorst's BiCGStab, right preconditioned, with the shadow residual r0
    template <typename T>
    Vec<T> bicgstab_iterate(
        const Vec<T>& A, const Vec<T>& b, Vec<T> x, int k, const nondeduced_t<Op<T>>& M)
    {
        Vec<T>       r      = residual(A, b, x);
        const Vec<T> shadow = r;
        Vec<T>       p(b.size(), 0), v(b.size(), 0);
        T            rho_old = 1, alpha = 1, omega = 1;
        for(int i = 0; i < k; ++i)
        {
            const T rho        = dot(shadow, r);
            const T beta       = (rho / rho_old) * (alpha / omega);
            p                  = plus(r, beta, plus(p, -omega, v));
            const Vec<T> p_hat = M(p);
            v                  = multiply(A, p_hat);
            alpha              = rho / dot(shadow, v);
            const Vec<T> s     = plus(r, -alpha, v);
            const Vec<T> s_hat = M(s);
            const Vec<T> t     = multiply(A, s_hat);
            omega              = dot(t, s) / dot(t, t);
            x                  = plus(plus(x, alpha, p_hat), omega, s_hat);
            r                  = plus(s, -omega, t);
            rho_old            = rho;
        }
        return x;
    }

    // M^-1 A, column by column
    template <typename T>
    Vec<T> preconditioned_matrix(const Vec<T>& A, const nondeduced_t<Op<T>>& M)
    {
        const size_t n = static_cast<size_t>(std::lround(std::sqrt(static_cast<double>(A.size()))));
        Vec<T>       B(A.size());
        for(size_t j = 0; j < n; ++j)
        {
            Vec<T> column(n);
            for(size_t i = 0; i < n; ++i)
            {
                column[i] = A[i * n + j];
            }
            column = M(column);
            for(size_t i = 0; i < n; ++i)
            {
                B[i * n + j] = column[i];
            }
        }
        return B;
    }

    // QMRCGStab of Chan, Gallopoulos, Simoncini, Szeto and Tong (1994), right
    // preconditioned, with the shadow residual r0 and p = v = d = 0 at the start. With
    // end_with_half_step, the k-th iteration stops after its first quasi-minimization.
    template <typename T>
    Vec<T> qmrcgstab_iterate(const Vec<T>&              A,
                             const Vec<T>&              b,
                             Vec<T>                     x,
                             int                        k,
                             const nondeduced_t<Op<T>>& M,
                             bool                       end_with_half_step = false)
    {
        Vec<T>       r      = residual(A, b, x);
        const Vec<T> shadow = r;
        Vec<T>       p(b.size(), 0), v(b.size(), 0), d(b.size(), 0);
        T            rho_old = 1, alpha = 1, omega = 1;
        T            tau = norm(r), theta = 0, eta = 0;
        for(int i = 0; i < k; ++i)
        {
            const T rho        = dot(shadow, r);
            const T beta       = (rho / rho_old) * (alpha / omega);
            p                  = plus(r, beta, plus(p, -omega, v));
            const Vec<T> p_hat = M(p);
            v                  = multiply(A, p_hat);
            alpha              = rho / dot(shadow, v);
            const Vec<T> s     = plus(r, -alpha, v);

            const T      theta_half = norm(s) / tau;
            T            c          = 1 / std::sqrt(1 + theta_half * theta_half);
            const T      tau_half   = tau * theta_half * c;
            const T      eta_half   = c * c * alpha;
            const Vec<T> d_half     = plus(p_hat, theta * theta * eta / alpha, d);
            const Vec<T> x_half     = plus(x, eta_half, d_half);
            if(end_with_half_step && i + 1 == k)
            {
                return x_half;
            }

            const Vec<T> s_hat = M(s);
            const Vec<T> t     = multiply(A, s_hat);
            omega              = dot(s, t) / dot(t, t);
            r                  = plus(s, -omega, t);

            theta = norm(r) / tau_half;
            c     = 1 / std::sqrt(1 + theta * theta);
            tau   = tau_half * theta * c;
            eta   = c * c * omega;
            d     = plus(s_hat, theta_half * theta_half * eta_half / omega, d_half);
            x     = plus(x_half, eta, d);

            rho_old = rho;
        }
        return x;
    }

    // The Chebyshev iteration for the spectrum of M^-1 A in [lambda_min, lambda_max],
    // as in Saad, Iterative Methods for Sparse Linear Systems, Algorithm 12.1. The error
    // after k iterations is T_k((d - M^-1 A) / c) / T_k(d / c) times the initial error.
    template <typename T>
    Vec<T> chebyshev_iterate(const Vec<T>&              A,
                             const Vec<T>&              b,
                             Vec<T>                     x,
                             int                        k,
                             const nondeduced_t<Op<T>>& M,
                             nondeduced_t<T>            lambda_min,
                             nondeduced_t<T>            lambda_max)
    {
        const T theta = (lambda_max + lambda_min) / 2;
        const T delta = (lambda_max - lambda_min) / 2;
        const T sigma = theta / delta;
        T       rho   = 1 / sigma;
        Vec<T>  r     = residual(A, b, x);
        Vec<T>  d     = scaled(1 / theta, M(r));
        for(int i = 0; i < k; ++i)
        {
            x                = plus(x, 1, d);
            r                = plus(r, -1, multiply(A, d));
            const T rho_next = 1 / (2 * sigma - rho);
            d                = plus(scaled(rho_next * rho, d), 2 * rho_next / delta, M(r));
            rho              = rho_next;
        }
        return x;
    }

    // x_{j+1} = x_j + omega M^-1 (b - A x_j)
    template <typename T>
    Vec<T> richardson_iterate(const Vec<T>&              A,
                              const Vec<T>&              b,
                              Vec<T>                     x,
                              int                        k,
                              const nondeduced_t<Op<T>>& M,
                              nondeduced_t<T>            omega)
    {
        for(int i = 0; i < k; ++i)
        {
            x = plus(x, omega, M(residual(A, b, x)));
        }
        return x;
    }

    template <typename T>
    T norm_of_type(const Vec<T>& r, int type)
    {
        T s = 0;
        for(T e : r)
        {
            s = type == 1 ? s + std::abs(e) : type == 3 ? std::max(s, std::abs(e)) : s + e * e;
        }
        return type == 2 ? std::sqrt(s) : s;
    }

    // Test systems with small integer entries

    template <typename T>
    Csr<T> spd(int n)
    {
        return Tridiagonal<T>(n);
    }

    // A dominant diagonal 6 + i % 3, and an entry three columns right of the diagonal,
    // wrapped around, in the even rows
    template <typename T>
    Csr<T> nonsymmetric(int n)
    {
        Csr<T> A(n);
        for(int i = 0; i < n; ++i)
        {
            std::map<int, T> row;
            row[i] = static_cast<T>(6 + i % 3);
            if(i > 0)
            {
                row[i - 1] = -2;
            }
            if(i + 1 < n)
            {
                row[i + 1] = 1;
            }
            if(i % 2 == 0)
            {
                row[(i + 3) % n] += 1;
            }
            for(const auto& e : row)
            {
                A.add(e.first, e.second);
            }
            A.end_row();
        }
        return A;
    }

    // Krylov iterates do not change when M^-1 = c I, so the checks of preconditioned
    // iterates see Jacobi only on matrices whose diagonal is not constant
    template <typename T>
    bool varying_diagonal(const Vec<T>& A)
    {
        const size_t n = static_cast<size_t>(std::lround(std::sqrt(static_cast<double>(A.size()))));
        for(size_t i = 1; i < n; ++i)
        {
            if(A[i * n + i] != A[0])
            {
                return true;
            }
        }
        return false;
    }

    // I + P_1 + ... + P_(k-1) for n = 8 and k <= 4, with P_m(i, j) = 1 if i = j mod
    // 2^(3 - m). The P_m are the Kronecker products J x I x I, J x J x I and J x J x J
    // of the 2 x 2 matrix of ones J and the identity, which commute. So the matrix is
    // symmetric with the k distinct eigenvalues 2^m - 1, m = 1, ..., k, and its
    // diagonal is k.
    template <typename T>
    Vec<T> with_distinct_eigenvalues(int k)
    {
        const int n = 8;
        Vec<T>    C(n * n, 0);
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                C[i * n + j] = i == j ? T(1) : T(0);
                for(int m = 1; m < k; ++m)
                {
                    if(i % (8 >> m) == j % (8 >> m))
                    {
                        C[i * n + j] += 1;
                    }
                }
            }
        }
        return C;
    }

    template <typename T>
    Csr<T> diagonal(const Vec<T>& d)
    {
        Csr<T> A(static_cast<int>(d.size()));
        for(size_t i = 0; i < d.size(); ++i)
        {
            A.add(static_cast<int>(i), d[i]);
            A.end_row();
        }
        return A;
    }

    // Q diag(eigenvalues) Q^T with the reflection Q = I - 2/n e e^T, exact for n = 8
    template <typename T>
    Csr<T> symmetric_with_eigenvalues(const Vec<T>& eigenvalues)
    {
        const int n = static_cast<int>(eigenvalues.size());
        Vec<T>    Q(n * n), A(n * n, 0);
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                Q[i * n + j] = (i == j ? T(1) : T(0)) - T(2) / n;
            }
        }
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                for(int l = 0; l < n; ++l)
                {
                    A[i * n + j] += Q[i * n + l] * eigenvalues[l] * Q[j * n + l];
                }
            }
        }
        return from_dense(n, n, A);
    }

    // S diag(eigenvalues) S^-1 with S = L U for unit lower and upper bidiagonal L and U
    // of ones, an integer matrix
    template <typename T>
    Vec<T> eigenvector_basis(int n, bool inverse)
    {
        Vec<T> L(n * n, 0), U(n * n, 0);
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                if(inverse)
                {
                    L[i * n + j] = i >= j ? ((i - j) % 2 == 0 ? T(1) : T(-1)) : T(0);
                    U[i * n + j] = j >= i ? ((j - i) % 2 == 0 ? T(1) : T(-1)) : T(0);
                }
                else
                {
                    L[i * n + j] = i == j || i == j + 1 ? T(1) : T(0);
                    U[i * n + j] = i == j || j == i + 1 ? T(1) : T(0);
                }
            }
        }
        return inverse ? dense_product(U, L, n, n, n) : dense_product(L, U, n, n, n);
    }

    template <typename T>
    Csr<T> nonsymmetric_with_eigenvalues(const Vec<T>& eigenvalues)
    {
        const int n = static_cast<int>(eigenvalues.size());
        Vec<T>    S = eigenvector_basis<T>(n, false);
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                S[i * n + j] *= eigenvalues[j];
            }
        }
        return from_dense(n, n, dense_product(S, eigenvector_basis<T>(n, true), n, n, n));
    }

    // Has a component in every eigenvector of nonsymmetric_with_eigenvalues
    template <typename T>
    Vec<T> generic_rhs(int n)
    {
        return multiply(eigenvector_basis<T>(n, false), Vec<T>(n, 1));
    }

    template <typename T>
    Vec<T> start(int n)
    {
        Vec<T> x(n);
        for(int i = 0; i < n; ++i)
        {
            x[i] = static_cast<T>(i % 3 - 1);
        }
        return x;
    }

    // LocalMatrix and LocalVector

    // The solvers by name, with the settings that need the operator

    template <typename T>
    struct Method
    {
        std::string                                    name;
        std::function<std::unique_ptr<Iterative<T>>()> make;
        std::function<void(Iterative<T>&)>             prepare;
        bool                                           needs_preconditioner = false;
    };

    template <typename S, typename T = typename value_type_of<S>::type>
    Method<T> method(const std::string& name)
    {
        return {name, [] { return std::unique_ptr<Iterative<T>>(new S); }, nullptr};
    }

    template <typename S, typename T = typename value_type_of<S>::type>
    Method<T> with_basis(const std::string& name, int basis)
    {
        return {name + "(" + std::to_string(basis) + ")",
                [basis] {
                    auto s = std::make_unique<S>();
                    s->SetBasisSize(basis);
                    return std::unique_ptr<Iterative<T>>(std::move(s));
                },
                nullptr};
    }

    template <typename T>
    Method<T> bicgstabl(int l)
    {
        return {"BiCGStab(" + std::to_string(l) + ")",
                [l] {
                    auto s = std::make_unique<BiCGStabl<T>>();
                    s->SetOrder(l);
                    return std::unique_ptr<Iterative<T>>(std::move(s));
                },
                nullptr};
    }

    template <typename T>
    Method<T> idr(int s)
    {
        return {"IDR(" + std::to_string(s) + ")",
                [] { return std::unique_ptr<Iterative<T>>(new IDR<T>); },
                [s](Iterative<T>& solver) {
                    auto& idr = dynamic_cast<IDR<T>&>(solver);
                    idr.SetShadowSpace(s);
                    idr.SetRandomSeed(12345ULL);
                }};
    }

    template <typename T>
    Method<T> fixed_point()
    {
        Method<T> m            = method<FixedPoint<T>>("FixedPoint");
        m.needs_preconditioner = true;
        return m;
    }

    template <typename T>
    std::vector<Method<T>> spd_methods()
    {
        return {method<CG<T>>("CG"), method<CR<T>>("CR"), method<FCG<T>>("FCG")};
    }

    template <typename T>
    std::vector<Method<T>> general_methods()
    {
        return {with_basis<GMRES<T>>("GMRES", 30),
                with_basis<GMRES<T>>("GMRES", 3),
                with_basis<FGMRES<T>>("FGMRES", 30),
                with_basis<FGMRES<T>>("FGMRES", 3),
                method<BiCGStab<T>>("BiCGStab"),
                bicgstabl<T>(1),
                bicgstabl<T>(2),
                bicgstabl<T>(4),
                idr<T>(1),
                idr<T>(2),
                idr<T>(4),
                method<QMRCGStab<T>>("QMRCGStab"),
                fixed_point<T>()};
    }

    // Gershgorin bounds that hold for spd(n) and for D^-1 spd(n), which lie in [1, n + 3]
    // and [1/3, 5/3]
    template <typename T>
    Method<T> chebyshev(int n)
    {
        return {"Chebyshev",
                [n] {
                    auto s = std::make_unique<Chebyshev<T>>();
                    s->Set(1.0 / 3.0, n + 3.0);
                    return std::unique_ptr<Iterative<T>>(std::move(s));
                },
                nullptr};
    }

    template <typename T>
    struct Problem
    {
        Method<T> method;
        Csr<T>    A;
    };

    // Every method with a system it converges on
    template <typename T>
    std::vector<Problem<T>> problems(int n)
    {
        std::vector<Problem<T>> p;
        for(const Method<T>& m : spd_methods<T>())
        {
            p.push_back({m, spd<T>(n)});
        }
        p.push_back({chebyshev<T>(n), spd<T>(n)});
        for(const Method<T>& m : general_methods<T>())
        {
            p.push_back({m, nonsymmetric<T>(n)});
        }
        for(const Problem<T>& q : p)
        {
            EXPECT_TRUE(varying_diagonal(dense(q.A))) << q.method.name;
        }
        return p;
    }

    template <typename T>
    bool is_bicgstabl(const Method<T>& m)
    {
        return m.name.rfind("BiCGStab(", 0) == 0;
    }

    // Solvers whose ReBuildNumeric of a built solver is broken
    template <typename T>
    bool rebuild_breaks(const Method<T>& m)
    {
        return m.name.rfind("IDR", 0) == 0 || m.name == "Chebyshev";
    }

    template <typename T>
    void set_up(Iterative<T>&            solver,
                const Method<T>&         m,
                const Matrix<T>&         A,
                nondeduced_t<Solver<T>>* precond)
    {
        solver.Verbose(0);
        solver.SetOperator(A);
        if(m.prepare)
        {
            m.prepare(solver);
        }
        if(precond != nullptr)
        {
            solver.SetPreconditioner(*precond);
        }
        solver.Build();
    }

    template <typename T>
    struct Outcome
    {
        Vec<T> x;
        int    iterations;
        int    status;
        double residual;
    };

    template <typename T>
    Outcome<T> solve(Iterative<T>& solver, const Matrix<T>& A, const Vec<T>& b, const Vec<T>& x0)
    {
        auto rhs = make_local_vector(A, b);
        auto x   = make_local_vector(A, x0);
        solver.Solve(*rhs, x.get());
        return {values_of(*x),
                solver.GetIterationCount(),
                solver.GetSolverStatus(),
                solver.GetCurrentResidual()};
    }

    template <typename T>
    std::string trace(const Method<T>& m, bool precond, const std::string& detail = "")
    {
        return m.name + (precond ? " with Jacobi" : "") + (detail.empty() ? "" : ", " + detail);
    }

    // A case whose known bug ends the process runs in a death test subprocess, so that
    // with ROCALUTION_UNIT_TEST_RUN_KNOWN_BUGS=1 it fails instead of ending the tests.
    // check returns whether the result is right; a failed EXPECT or a skip inside check
    // fails the case as well.
    void expect_completes(const std::function<bool()>& check)
    {
        EXPECT_EXIT(
            {
                std::signal(SIGABRT, exit_on_abort);
                const bool right = check();
                std::exit(!right || testing::Test::HasFailure() ? 2
                          : testing::Test::IsSkipped()          ? 3
                                                                : 0);
            },
            testing::ExitedWithCode(0),
            "");
    }

    // Runs f with the messages that the library prints regardless of the solver verbosity
    // dropped. Captures of stdout cannot nest, and a failure inside f would not be shown.
    struct QuietStdout
    {
        QuietStdout()
        {
            testing::internal::CaptureStdout();
        }
        ~QuietStdout()
        {
            testing::internal::GetCapturedStdout();
        }
    };

    template <typename F>
    auto quietly(F&& f) -> decltype(f())
    {
        const QuietStdout quiet;
        return f();
    }

    template <typename T>
    bool near(const Vec<T>& a, const Vec<T>& ref, double tol)
    {
        for(size_t i = 0; i < a.size(); ++i)
        {
            if(!is_near(a[i], ref[i], tol))
            {
                return false;
            }
        }
        return a.size() == ref.size();
    }

    // The relative tolerance of a solve that runs to convergence: float can't reach the
    // tolerances below 1e-5 of double
    template <typename T>
    double solver_tol(double for_double)
    {
        return std::is_same_v<T, float> ? std::max(for_double, 1e-5) : for_double;
    }

    // The accuracy of the solution of a solve of A x = b from x0 that stops once the
    // residual in the norm p is rel times the initial one. As |r|_inf <= |r|_p,
    // |x - x*|_inf <= |A^-1|_inf |r|_inf <= |A^-1|_inf rel |r0|_p. GMRES and BiCGStab(l)
    // are left preconditioned and measure M^-1 r instead, which multiplies the bound by
    // the condition max |a_ii| / min |a_ii| of the Jacobi M; the bound takes that factor
    // for every method with Jacobi. The methods that update the residual instead of
    // recomputing it drift from b - A x by rounding, allowed for as 100 roundings of
    // |A|_inf |x*|_inf + |b|_inf.
    template <typename T>
    double solution_tol(const Vec<T>& A,
                        const Vec<T>& b,
                        const Vec<T>& x0,
                        double        rel,
                        int           norm_type = 2,
                        bool          jacobi    = false)
    {
        const size_t        n = b.size();
        const Vec<T>        x = dense_solve(A, b);
        std::vector<double> inverse_row_sums(n, 0.0);
        for(size_t j = 0; j < n; ++j)
        {
            Vec<T> e(n, 0);
            e[j]                = 1;
            const Vec<T> column = dense_solve(A, e);
            for(size_t i = 0; i < n; ++i)
            {
                inverse_row_sums[i] += std::abs(static_cast<double>(column[i]));
            }
        }
        double a_norm = 0, d_min = std::numeric_limits<double>::infinity(), d_max = 0;
        for(size_t i = 0; i < n; ++i)
        {
            double row_sum = 0;
            for(size_t j = 0; j < n; ++j)
            {
                row_sum += std::abs(static_cast<double>(A[i * n + j]));
            }
            a_norm = std::max(a_norm, row_sum);
            d_min  = std::min(d_min, std::abs(static_cast<double>(A[i * n + i])));
            d_max  = std::max(d_max, std::abs(static_cast<double>(A[i * n + i])));
        }
        const double inverse_norm
            = *std::max_element(inverse_row_sums.begin(), inverse_row_sums.end());
        const double kappa = jacobi ? d_max / d_min : 1.0;
        const double r0    = norm_of_type(residual(A, b, x0), norm_type);
        const double rounding
            = 100.0 * std::numeric_limits<T>::epsilon()
              * (a_norm * norm_of_type(x, 3) + static_cast<double>(norm_of_type(b, 3)));
        return inverse_norm * (kappa * rel * r0 + rounding);
    }

    template <typename P>
    class solvers : public TypedBackendTest<P>
    {
    protected:
        using T = typename P::value_type;

        static Backend backend()
        {
            return P::backend;
        }

        std::unique_ptr<Matrix<T>> make(const Csr<T>& ref) const
        {
            return make_local_matrix(P::backend, ref);
        }
    };
}

TYPED_TEST_SUITE(solvers, WithBackends<RealTypes>, TypedName);

TYPED_TEST(solvers, cg_and_fcg_iterates_minimise_the_energy_norm_of_the_error_over_the_krylov_space)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = spd<T>(8);
    const Vec<T> D   = dense(ref);
    const Vec<T> b   = rhs<T>(8);
    const Vec<T> x0  = start<T>(8);
    auto         A   = this->make(ref);

    for(const Method<T>& m : {method<CG<T>>("CG"), method<FCG<T>>("FCG")})
    {
        for(bool precond : {false, true})
        {
            for(int k = 1; k <= 4; ++k)
            {
                SCOPED_TRACE(trace(m, precond, std::to_string(k) + " iterations"));
                Jacobi<T> jacobi;
                auto      solver = m.make();
                solver->Init(0.0, 0.0, 1e8, k);
                set_up(*solver, m, *A, precond ? &jacobi : nullptr);

                const Outcome<T> out = solve(*solver, *A, b, x0);
                expect_near(out.x,
                            galerkin_iterate(D, b, x0, k, preconditioner(D, precond)),
                            loose_tol<T>(1e-10));
                EXPECT_EQ(out.iterations, k);
                EXPECT_EQ(out.status, 4);
            }
        }
    }
}

// Preconditioned CR minimises the M^-1 norm of the residual
TYPED_TEST(solvers, cr_iterates_minimise_the_residual_over_the_krylov_space)
{
    using T             = typename TypeParam::value_type;
    const Csr<T>    ref = spd<T>(8);
    const Vec<T>    D   = dense(ref);
    const Vec<T>    b   = rhs<T>(8);
    const Vec<T>    x0  = start<T>(8);
    auto            A   = this->make(ref);
    const Method<T> m   = method<CR<T>>("CR");

    for(bool precond : {false, true})
    {
        for(int k = 1; k <= 4; ++k)
        {
            SCOPED_TRACE(trace(m, precond, std::to_string(k) + " iterations"));
            Jacobi<T> jacobi;
            auto      solver = m.make();
            solver->Init(0.0, 0.0, 1e8, k);
            set_up(*solver, m, *A, precond ? &jacobi : nullptr);

            const Outcome<T> out = solve(*solver, *A, b, x0);
            expect_near(
                out.x,
                min_residual_iterate(
                    D, b, x0, k, preconditioner(D, precond), root_of_preconditioner(D, precond)),
                loose_tol<T>(1e-10));
            EXPECT_EQ(out.iterations, k);
            EXPECT_EQ(out.status, 4);
        }
    }
}

// GMRES is left and FGMRES right preconditioned. With a fixed preconditioner both search
// the same space, but GMRES minimises the preconditioned residual.
TYPED_TEST(solvers, gmres_and_fgmres_iterates_minimise_the_residual_over_the_krylov_space)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = nonsymmetric<T>(8);
    const Vec<T> D   = dense(ref);
    const Vec<T> b   = rhs<T>(8);
    const Vec<T> x0  = start<T>(8);
    auto         A   = this->make(ref);
    ASSERT_TRUE(varying_diagonal(D));

    for(int basis : {30, 2})
    {
        for(const Method<T>& m :
            {with_basis<GMRES<T>>("GMRES", basis), with_basis<FGMRES<T>>("FGMRES", basis)})
        {
            const bool flexible = m.name[0] == 'F';
            for(bool precond : {false, true})
            {
                for(int k = 1; k <= 5; ++k)
                {
                    SCOPED_TRACE(trace(m, precond, std::to_string(k) + " iterations"));
                    Jacobi<T> jacobi;
                    auto      solver = m.make();
                    solver->Init(0.0, 0.0, 1e8, k);
                    set_up(*solver, m, *A, precond ? &jacobi : nullptr);

                    const Op<T> M     = preconditioner(D, precond);
                    const Op<T> W     = flexible ? Op<T>(identity<T>) : M;
                    auto        cycle = [&](const Vec<T>& x, int steps) {
                        return min_residual_iterate(D, b, x, steps, M, W);
                    };
                    const Outcome<T> out = solve(*solver, *A, b, x0);
                    expect_near(out.x, restarted<T>(cycle, x0, k, basis), loose_tol<T>(1e-10));
                    EXPECT_EQ(out.iterations, k);
                    EXPECT_EQ(out.status, 4);
                }
            }
        }
    }
}

// c I + S for the cyclic shift S e_j = e_(j+1 mod n). With c = 0 and b = e_0 the residual
// is orthogonal to A times the Krylov space until it is the whole space, so the iterates
// stay zero until iteration n. With c = 1/2 the subdiagonal of the Hessenberg matrix
// dominates its diagonal.
TYPED_TEST(solvers, gmres_and_fgmres_iterates_on_a_shifted_cyclic_permutation)
{
    using T     = typename TypeParam::value_type;
    const int n = 6;
    Vec<T>    b(n, 0);
    b[0] = 1;

    for(double c : {0.0, 0.5})
    {
        Csr<T> ref(n);
        for(int i = 0; i < n; ++i)
        {
            std::map<int, T> row;
            row[(i + n - 1) % n] = 1;
            if(c != 0.0)
            {
                row[i] = static_cast<T>(c);
            }
            for(const auto& e : row)
            {
                ref.add(e.first, e.second);
            }
            ref.end_row();
        }
        const Vec<T> D = dense(ref);
        auto         A = this->make(ref);

        for(const Method<T>& m :
            {with_basis<GMRES<T>>("GMRES", 30), with_basis<FGMRES<T>>("FGMRES", 30)})
        {
            for(int k = 1; k <= n; ++k)
            {
                SCOPED_TRACE(
                    trace(m,
                          false,
                          "c = " + std::to_string(c) + ", " + std::to_string(k) + " iterations"));
                auto solver = m.make();
                solver->Init(0.0, 0.0, 1e8, k);
                set_up(*solver, m, *A, nullptr);

                const Outcome<T> out = solve(*solver, *A, b, Vec<T>(n, 0));
                const Vec<T>     expected
                    = min_residual_iterate(D, b, Vec<T>(n, 0), k, identity<T>, identity<T>);
                expect_near(out.x, expected, tol<T>());
                if(c == 0.0)
                {
                    expect_near(
                        expected, k < n ? Vec<T>(n, 0) : dense_solve(D, b), scaled_tol<T>(1e-14));
                }
                EXPECT_EQ(out.iterations, k);
            }
        }
    }
}

// BiCGStab(1) is BiCGStab, with one BiCGStab iteration per cycle. BiCGStab is right
// and BiCGStab(l) left preconditioned, so BiCGStab(1) with Jacobi M is BiCGStab on
// M^-1 A x = M^-1 b.
TYPED_TEST(solvers, bicgstab_iterates_follow_the_bicgstab_recurrence)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = nonsymmetric<T>(8);
    const Vec<T> D   = dense(ref);
    const Vec<T> b   = rhs<T>(8);
    const Vec<T> x0  = start<T>(8);
    auto         A   = this->make(ref);
    ASSERT_TRUE(varying_diagonal(D));

    for(const Method<T>& m : {method<BiCGStab<T>>("BiCGStab"), bicgstabl<T>(1)})
    {
        for(bool precond : {false, true})
        {
            const Op<T> M    = preconditioner(D, precond);
            const bool  left = precond && is_bicgstabl(m);
            for(int k = 1; k <= 4; ++k)
            {
                SCOPED_TRACE(trace(m, precond, std::to_string(k) + " iterations"));
                Jacobi<T> jacobi;
                auto      solver = m.make();
                solver->Init(0.0, 0.0, 1e8, k);
                set_up(*solver, m, *A, precond ? &jacobi : nullptr);

                const Outcome<T> out = solve(*solver, *A, b, x0);
                expect_near(
                    out.x,
                    left ? bicgstab_iterate(preconditioned_matrix(D, M), M(b), x0, k, identity<T>)
                         : bicgstab_iterate(D, b, x0, k, M),
                    loose_tol<T>(1e-10));
                EXPECT_EQ(out.iterations, k);
                EXPECT_EQ(out.status, 4);
            }
        }
    }
}

TYPED_TEST(solvers, qmrcgstab_iterates_follow_the_qmrcgstab_recurrence)
{
    using T             = typename TypeParam::value_type;
    const Csr<T>    ref = nonsymmetric<T>(8);
    const Vec<T>    D   = dense(ref);
    const Vec<T>    b   = rhs<T>(8);
    const Vec<T>    x0  = start<T>(8);
    auto            A   = this->make(ref);
    const Method<T> m   = method<QMRCGStab<T>>("QMRCGStab");
    ASSERT_TRUE(varying_diagonal(D));

    std::vector<int> iterations;
    for(bool precond : {false, true})
    {
        for(int k = 1; k <= 4; ++k)
        {
            SCOPED_TRACE(trace(m, precond, std::to_string(k) + " iterations"));
            Jacobi<T> jacobi;
            auto      solver = m.make();
            solver->Init(0.0, 0.0, 1e8, k);
            set_up(*solver, m, *A, precond ? &jacobi : nullptr);

            const Outcome<T> out = solve(*solver, *A, b, x0);
            expect_near(out.x,
                        qmrcgstab_iterate(D, b, x0, k, preconditioner(D, precond)),
                        loose_tol<T>(1e-10));
            EXPECT_EQ(out.status, 4);
            iterations.push_back(out.iterations);
        }
    }

    KNOWN_BUG("the residual check after the last iteration is counted as an iteration");
    EXPECT_EQ(iterations, std::vector<int>({1, 2, 3, 4, 1, 2, 3, 4}));
}

TYPED_TEST(solvers, qmrcgstab_starts_every_solve_afresh)
{
    using T             = typename TypeParam::value_type;
    const Csr<T>    ref = nonsymmetric<T>(8);
    const Vec<T>    D   = dense(ref);
    const Vec<T>    b   = rhs<T>(8);
    const Vec<T>    x0  = start<T>(8);
    auto            A   = this->make(ref);
    const Method<T> m   = method<QMRCGStab<T>>("QMRCGStab");

    auto solver = m.make();
    solver->Init(0.0, 0.0, 1e8, 3);
    set_up(*solver, m, *A, nullptr);

    // The bug moves the iterates by up to 1.3e-5, below scaled_tol<float>(1e-10), and
    // float rounds these 3 iterations by less than 1e-6
    const double tolerance = std::is_same_v<T, float> ? 1e-6 : 1e-10;
    const Vec<T> expected  = qmrcgstab_iterate(D, b, x0, 3, identity<T>);
    expect_near(solve(*solver, *A, b, x0).x, expected, tolerance);

    KNOWN_BUG("the first step adds r to the search direction of the previous solve");
    expect_near(solve(*solver, *A, b, x0).x, expected, tolerance);
}

// Exact convergence in the first half step makes the denominators of the second half
// step zero
TYPED_TEST(solvers, qmrcgstab_solves_a_system_that_its_first_half_step_solves)
{
    using T             = typename TypeParam::value_type;
    const Csr<T>    ref = diagonal(Vec<T>(8, 2));
    const Vec<T>    b   = rhs<T>(8);
    auto            A   = this->make(ref);
    const Method<T> m   = method<QMRCGStab<T>>("QMRCGStab");

    auto solver = m.make();
    solver->Init(0.0, 1e-10, 1e8, 10);
    set_up(*solver, m, *A, nullptr);
    const Outcome<T> out = quietly([&] { return solve(*solver, *A, b, Vec<T>(8, 0)); });

    KNOWN_BUG("the first step, before the loop, has no breakdown check and returns NaN");
    expect_near(out.x, scaled(0.5, b), scaled_tol<T>(1e-14));
}

// The orthogonal and bidiagonal similarity transformations keep the matrices exact
TYPED_TEST(solvers, krylov_methods_converge_in_as_many_iterations_as_there_are_distinct_eigenvalues)
{
    using T                           = typename TypeParam::value_type;
    const std::vector<Vec<T>> spectra = {{2, 2, 2, 2, 2, 2, 2, 2},
                                         {1, 1, 1, 1, 3, 3, 3, 3},
                                         {1, 1, 2, 2, 2, 4, 4, 4},
                                         {1, 2, 3, 5, 1, 2, 3, 5}};

    for(bool symmetric : {true, false})
    {
        const std::vector<Method<T>> methods
            = symmetric ? spd_methods<T>()
                        : std::vector<Method<T>>{with_basis<GMRES<T>>("GMRES", 30),
                                                 with_basis<FGMRES<T>>("FGMRES", 30),
                                                 method<BiCGStab<T>>("BiCGStab")};
        for(const Method<T>& m : methods)
        {
            for(size_t k = 1; k <= spectra.size(); ++k)
            {
                SCOPED_TRACE(trace(m, false, std::to_string(k) + " distinct eigenvalues"));
                const Csr<T> ref = symmetric ? symmetric_with_eigenvalues(spectra[k - 1])
                                             : nonsymmetric_with_eigenvalues(spectra[k - 1]);
                const Vec<T> b   = symmetric ? rhs<T>(8) : generic_rhs<T>(8);
                auto         A   = this->make(ref);

                auto solver = m.make();
                solver->Init(0.0, solver_tol<T>(1e-10), 1e8, 50);
                set_up(*solver, m, *A, nullptr);

                // BiCGStab reports the breakdown of 2 I in its first half step
                const Outcome<T> out = quietly([&] { return solve(*solver, *A, b, Vec<T>(8, 0)); });
                EXPECT_EQ(out.iterations, static_cast<int>(k));
                EXPECT_TRUE(out.status == 1 || out.status == 2) << "status " << out.status;
                expect_near(out.x,
                            dense_solve(dense(ref), b),
                            solution_tol(dense(ref), b, Vec<T>(8, 0), solver_tol<T>(1e-10)));
            }
        }
    }
}

// On an operator with k distinct eigenvalues, BiCGStab(l) with l >= k and IDR(s) with
// s >= k solve within their first cycle, and with l < k or s < k they do not. The
// BiCG part of BiCGStab(l) is BiCG, which ends after k steps. The first cycle of IDR(s)
// makes the residual in r0 + A M^-1 K_j(A M^-1, r0) orthogonal to j shadow vectors,
// which at j = k leaves only r = 0. BiCGStab(l) counts completed cycles, so a solution
// in its BiCG part takes 0 iterations; IDR counts its first cycle. With Jacobi the
// operator D C has the diagonal k D, so that M^-1 D C = C / k (BiCGStab(l) is left
// preconditioned) and D C M^-1 = D (C / k) D^-1 (IDR is right preconditioned).
TYPED_TEST(solvers,
           bicgstabl_and_idr_solve_in_their_first_cycle_when_their_order_covers_the_spectrum)
{
    using T     = typename TypeParam::value_type;
    const int n = 8;
    // b and D^-1 b have a component in every eigenspace of C, and D C has 8 distinct
    // eigenvalues
    Vec<T> b(n);
    for(int i = 0; i < n; ++i)
    {
        b[i] = static_cast<T>((i + 1) * (i + 1));
    }

    for(int k = 1; k <= 4; ++k)
    {
        for(bool precond : {false, true})
        {
            Vec<T> D = with_distinct_eigenvalues<T>(k);
            if(precond)
            {
                for(int i = 0; i < n; ++i)
                {
                    for(int j = 0; j < n; ++j)
                    {
                        D[i * n + j] *= static_cast<T>(i + 1);
                    }
                }
                ASSERT_TRUE(varying_diagonal(D));
            }
            auto A = this->make(from_dense(n, n, D));

            for(int order : {1, 2, 4})
            {
                for(const auto& [m, first_cycle] :
                    {std::make_pair(bicgstabl<T>(order), 0), std::make_pair(idr<T>(order), 1)})
                {
                    SCOPED_TRACE(trace(m, precond, std::to_string(k) + " distinct eigenvalues"));
                    Jacobi<T> jacobi;
                    auto      solver = m.make();
                    solver->Init(0.0, solver_tol<T>(1e-10), 1e8, 100);
                    set_up(*solver, m, *A, precond ? &jacobi : nullptr);

                    const Outcome<T> out = solve(*solver, *A, b, Vec<T>(n, 0));
                    EXPECT_TRUE(out.status == 1 || out.status == 2) << "status " << out.status;
                    expect_near(out.x,
                                dense_solve(D, b),
                                solution_tol(D, b, Vec<T>(n, 0), solver_tol<T>(1e-10), 2, precond));
                    if(order >= k)
                    {
                        EXPECT_EQ(out.iterations, first_cycle);
                    }
                    else
                    {
                        EXPECT_GT(out.iterations, first_cycle);
                    }
                }
            }
        }
    }
}

// The preconditioned system is the identity, so the first half step solves it exactly
TYPED_TEST(solvers, bicgstab_solves_a_system_that_its_preconditioned_first_half_step_solves)
{
    using T             = typename TypeParam::value_type;
    const Vec<T>    d   = {2, 4, 8, 16, 2, 4, 8, 16};
    const Csr<T>    ref = diagonal(d);
    const Vec<T>    b   = rhs<T>(8);
    auto            A   = this->make(ref);
    const Method<T> m   = method<BiCGStab<T>>("BiCGStab");

    Vec<T> solution(8);
    for(int i = 0; i < 8; ++i)
    {
        solution[i] = b[i] / d[i];
    }

    {
        SCOPED_TRACE("without preconditioner, on 2 I");
        auto I2     = this->make(diagonal(Vec<T>(8, 2)));
        auto solver = m.make();
        solver->Init(0.0, 1e-10, 1e8, 10);
        set_up(*solver, m, *I2, nullptr);
        testing::internal::CaptureStdout();
        const Outcome<T> out = solve(*solver, *I2, b, Vec<T>(8, 0));
        EXPECT_NE(testing::internal::GetCapturedStdout().find("only in p-direction"),
                  std::string::npos);
        EXPECT_EQ(out.iterations, 1);
        expect_near(out.x, scaled(0.5, b), scaled_tol<T>(1e-14));
    }

    Jacobi<T> jacobi;
    auto      solver = m.make();
    solver->Init(0.0, 1e-10, 1e8, 10);
    set_up(*solver, m, *A, &jacobi);
    testing::internal::CaptureStdout();
    const Outcome<T> out = solve(*solver, *A, b, Vec<T>(8, 0));
    EXPECT_NE(testing::internal::GetCapturedStdout().find("only in p-direction"),
              std::string::npos);
    EXPECT_EQ(out.iterations, 1);

    KNOWN_BUG("the breakdown branch updates x along p instead of the preconditioned M^-1 p");
    expect_near(out.x, solution, scaled_tol<T>(1e-14));
}

namespace
{
    template <typename T>
    Outcome<T> converge(Backend backend, const Problem<T>& p, bool precond, int norm_type, int n)
    {
        auto      A = make_local_matrix(backend, p.A);
        Jacobi<T> jacobi;
        auto      solver = p.method.make();
        solver->Init(0.0, solver_tol<T>(1e-12), 1e8, 1000);
        solver->SetResidualNorm(norm_type);
        set_up(*solver, p.method, *A, precond ? &jacobi : nullptr);
        return solve(*solver, *A, rhs<T>(n), start<T>(n));
    }

    template <typename T>
    double converged_tol(const Problem<T>& p, bool precond, int norm_type, int n)
    {
        return solution_tol(
            dense(p.A), rhs<T>(n), start<T>(n), solver_tol<T>(1e-12), norm_type, precond);
    }
}

TYPED_TEST(solvers, every_solver_reaches_the_solution_with_each_residual_norm)
{
    using T     = typename TypeParam::value_type;
    const int n = 16;

    for(const Problem<T>& p : problems<T>(n))
    {
        const Vec<T> x       = dense_solve(dense(p.A), rhs<T>(n));
        const bool   l2_only = p.method.name.find("GMRES") != std::string::npos;
        for(bool precond : {false, true})
        {
            if(p.method.needs_preconditioner && !precond)
            {
                continue;
            }
            for(int norm_type : {1, 2, 3})
            {
                if(l2_only && norm_type != 2)
                {
                    continue;
                }
                SCOPED_TRACE(trace(p.method, precond, "norm " + std::to_string(norm_type)));
                const Outcome<T> out = converge(this->backend(), p, precond, norm_type, n);
                EXPECT_TRUE(out.status == 1 || out.status == 2) << "status " << out.status;
                expect_near(out.x, x, converged_tol(p, precond, norm_type, n));
            }
        }
    }
}

namespace
{
    // The GMRES residuals decrease strictly, so that a tolerance between two of them
    // stops at a known iteration
    template <typename T>
    struct GmresSequence
    {
        Vec<T>              D;
        Vec<T>              b;
        std::vector<Vec<T>> x;
        Vec<T>              res;
    };

    template <typename T>
    GmresSequence<T> gmres_sequence(const Csr<T>& ref, int k)
    {
        GmresSequence<T> s{dense(ref), rhs<T>(ref.n), {}, {}};
        for(int j = 0; j <= k; ++j)
        {
            s.x.push_back(j == 0 ? Vec<T>(ref.n, 0)
                                 : min_residual_iterate(
                                       s.D, s.b, Vec<T>(ref.n, 0), j, identity<T>, identity<T>));
            s.res.push_back(norm(residual(s.D, s.b, s.x.back())));
        }
        return s;
    }
}

TYPED_TEST(solvers, each_tolerance_stops_at_the_first_iteration_that_meets_it)
{
    using T                    = typename TypeParam::value_type;
    const Csr<T>           ref = nonsymmetric<T>(8);
    const GmresSequence<T> s   = gmres_sequence(ref, 5);
    for(int j = 1; j <= 5; ++j)
    {
        ASSERT_LT(s.res[j], 0.9 * s.res[j - 1]);
    }
    auto            A = this->make(ref);
    const Method<T> m = with_basis<GMRES<T>>("GMRES", 30);

    struct Rule
    {
        std::string name;
        double      abs, rel;
        int         max_iter, status;
        std::string report;
    };
    const double between = std::sqrt(s.res[2] * s.res[3]);
    for(const Rule& rule : {Rule{"absolute", between, 0.0, 100, 1, "ABSOLUTE criteria"},
                            Rule{"relative", 0.0, between / s.res[0], 100, 2, "RELATIVE criteria"},
                            Rule{"maximum iterations", 0.0, 0.0, 3, 4, "MAX ITER criteria"}})
    {
        SCOPED_TRACE(rule.name);
        auto solver = m.make();
        solver->InitTol(rule.abs, rule.rel, 1e8);
        solver->InitMaxIter(rule.max_iter);
        set_up(*solver, m, *A, nullptr);
        solver->Verbose(1);

        testing::internal::CaptureStdout();
        const Outcome<T> out = solve(*solver, *A, s.b, Vec<T>(8, 0));
        EXPECT_NE(testing::internal::GetCapturedStdout().find(rule.report), std::string::npos);
        EXPECT_EQ(out.iterations, 3);
        EXPECT_EQ(out.status, rule.status);
        EXPECT_NEAR(out.residual, s.res[3], loose_tol<T>(1e-10) * s.res[3]);
        expect_near(out.x, s.x[3], loose_tol<T>(1e-10));
    }
}

TYPED_TEST(solvers, divergence_tolerance_stops_a_diverging_iteration)
{
    using T            = typename TypeParam::value_type;
    const Csr<T> ref   = spd<T>(8);
    const Vec<T> D     = dense(ref);
    const Vec<T> b     = rhs<T>(8);
    const double omega = 3.0;
    const Op<T>  M     = preconditioner(D, true);
    const T      r0    = norm(b);

    int    first = -1;
    double below = 0.0;
    for(int j = 1; j < 100 && first < 0; ++j)
    {
        const double ratio
            = norm(residual(D, b, richardson_iterate(D, b, Vec<T>(8, 0), j, M, omega))) / r0;
        if(ratio >= 100.0)
        {
            first = j;
        }
        else
        {
            below = std::max(below, ratio);
        }
    }
    ASSERT_GT(first, 0);
    const double ratio_first
        = norm(residual(D, b, richardson_iterate(D, b, Vec<T>(8, 0), first, M, omega))) / r0;

    auto          A = this->make(ref);
    Jacobi<T>     jacobi;
    FixedPoint<T> solver;
    solver.SetRelaxation(omega);
    solver.Init(0.0, 0.0, std::sqrt(below * ratio_first), 1000);
    set_up(solver, fixed_point<T>(), *A, &jacobi);
    solver.Verbose(1);

    testing::internal::CaptureStdout();
    const Outcome<T> out = solve(solver, *A, b, Vec<T>(8, 0));
    EXPECT_NE(testing::internal::GetCapturedStdout().find("DIVERGENCE criteria"),
              std::string::npos);
    EXPECT_EQ(out.iterations, first);
    EXPECT_EQ(out.status, 3);
    expect_near(
        out.x, richardson_iterate(D, b, Vec<T>(8, 0), first, M, omega), loose_tol<T>(1e-10));
}

// The relative tolerance is met after the first iteration
TYPED_TEST(solvers, minimum_iteration_count_delays_convergence)
{
    using T                    = typename TypeParam::value_type;
    const Csr<T>           ref = nonsymmetric<T>(8);
    const GmresSequence<T> s   = gmres_sequence(ref, 4);
    const double           rel = 0.99 * s.res[1] / s.res[0];
    auto                   A   = this->make(ref);

    struct Case
    {
        Method<T> method;
        Vec<T>    x;
    };
    auto restarted_gmres = [&](int basis) {
        return restarted<T>(
            [&](const Vec<T>& x, int steps) {
                return min_residual_iterate(s.D, s.b, x, steps, identity<T>, identity<T>);
            },
            Vec<T>(8, 0),
            4,
            basis);
    };

    {
        const Method<T> m = with_basis<GMRES<T>>("GMRES", 30);
        SCOPED_TRACE(m.name);
        auto solver = m.make();
        solver->Init(0.0, rel, 1e8, 4, 100);
        set_up(*solver, m, *A, nullptr);
        const Outcome<T> out = solve(*solver, *A, s.b, Vec<T>(8, 0));
        EXPECT_EQ(out.iterations, 4);
        EXPECT_EQ(out.status, 2);
        expect_near(out.x, s.x[4], loose_tol<T>(1e-10));
    }
    {
        const Method<T> m = method<CG<T>>("CG");
        SCOPED_TRACE("CG");
        const Csr<T> sym    = spd<T>(8);
        auto         S      = this->make(sym);
        auto         solver = m.make();
        solver->Init(0.0, 0.999, 1e8, 100);
        solver->InitMinIter(4);
        set_up(*solver, m, *S, nullptr);
        const Outcome<T> out = solve(*solver, *S, s.b, Vec<T>(8, 0));
        EXPECT_EQ(out.iterations, 4);
        expect_near(out.x,
                    galerkin_iterate(dense(sym), s.b, Vec<T>(8, 0), 4, identity<T>),
                    loose_tol<T>(1e-10));
    }

    KNOWN_BUG("the checks at a restart and in the BiCG part ignore the minimum iteration count");
    for(const Case& c : {Case{with_basis<GMRES<T>>("GMRES", 2), restarted_gmres(2)},
                         Case{with_basis<FGMRES<T>>("FGMRES", 2), restarted_gmres(2)},
                         Case{bicgstabl<T>(2), Vec<T>()}})
    {
        SCOPED_TRACE(c.method.name);
        auto solver = c.method.make();
        solver->Init(0.0, rel, 1e8, 4, 100);
        set_up(*solver, c.method, *A, nullptr);
        const Outcome<T> out = solve(*solver, *A, s.b, Vec<T>(8, 0));
        EXPECT_EQ(out.iterations, 4);
        if(!c.x.empty())
        {
            expect_near(out.x, c.x, loose_tol<T>(1e-10));
        }
    }
}

TYPED_TEST(solvers, current_residual_is_measured_in_the_chosen_norm)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = spd<T>(8);
    const Vec<T> D   = dense(ref);
    const Vec<T> b   = rhs<T>(8);
    const Vec<T> x0  = start<T>(8);
    const Vec<T> r   = residual(D, b, galerkin_iterate(D, b, x0, 3, identity<T>));
    auto         A   = this->make(ref);

    const auto largest
        = std::max_element(r.begin(), r.end(), [](T a, T c) { return std::abs(a) < std::abs(c); });
    const int64_t index = largest - r.begin();
    for(size_t i = 0; i < r.size(); ++i)
    {
        ASSERT_TRUE(static_cast<int64_t>(i) == index || std::abs(r[i]) < 0.99 * std::abs(*largest));
    }

    for(int type : {1, 2})
    {
        SCOPED_TRACE("norm " + std::to_string(type));
        CG<T> solver;
        solver.Init(0.0, 0.0, 1e8, 3);
        solver.SetResidualNorm(type);
        set_up(solver, method<CG<T>>("CG"), *A, nullptr);

        const Outcome<T> out = solve(solver, *A, b, x0);
        EXPECT_NEAR(
            out.residual, norm_of_type(r, type), loose_tol<T>(1e-10) * norm_of_type(r, type));
        testing::internal::CaptureStdout();
        EXPECT_EQ(solver.GetAmaxResidualIndex(), -1);
        EXPECT_NE(testing::internal::GetCapturedStdout().find("only available when using Linf"),
                  std::string::npos);
    }

    SCOPED_TRACE("norm 3");
    CG<T> solver;
    solver.Init(0.0, 0.0, 1e8, 3);
    solver.SetResidualNorm(3);
    set_up(solver, method<CG<T>>("CG"), *A, nullptr);
    const Outcome<T> out = solve(solver, *A, b, x0);
    EXPECT_NEAR(out.residual, norm_of_type(r, 3), loose_tol<T>(1e-10) * norm_of_type(r, 3));
    EXPECT_EQ(solver.GetAmaxResidualIndex(), index);
}

TYPED_TEST(solvers, gmres_and_fgmres_switch_to_the_euclidean_norm)
{
    using T                    = typename TypeParam::value_type;
    const Csr<T>           ref = nonsymmetric<T>(8);
    const GmresSequence<T> s   = gmres_sequence(ref, 3);
    auto                   A   = this->make(ref);

    for(const Method<T>& m :
        {with_basis<GMRES<T>>("GMRES", 30), with_basis<FGMRES<T>>("FGMRES", 30)})
    {
        SCOPED_TRACE(m.name);
        auto solver = m.make();
        solver->Init(0.0, 0.0, 1e8, 3);
        solver->SetResidualNorm(3);
        testing::internal::CaptureStdout();
        set_up(*solver, m, *A, nullptr);
        EXPECT_NE(testing::internal::GetCapturedStdout().find("supports only L2 residual norm"),
                  std::string::npos);

        const Outcome<T> out = solve(*solver, *A, s.b, Vec<T>(8, 0));
        EXPECT_NEAR(out.residual, s.res[3], loose_tol<T>(1e-10) * s.res[3]);
    }
}

TYPED_TEST(solvers, residual_history_file_holds_the_residuals_of_the_last_solve)
{
    using T                     = typename TypeParam::value_type;
    const Csr<T>           ref  = nonsymmetric<T>(8);
    const GmresSequence<T> s    = gmres_sequence(ref, 3);
    auto                   A    = this->make(ref);
    const Method<T>        m    = with_basis<GMRES<T>>("GMRES", 30);
    const std::string      file = temp_file("txt");
    // The file holds 7 significant digits, and in float the residuals their rounding
    const double tolerance = std::max(1e-6, loose_tol<T>(1e-10));

    auto read = [&]() {
        std::ifstream       in(file);
        std::vector<double> r;
        double              v;
        while(in >> v)
        {
            r.push_back(v);
        }
        return r;
    };

    auto solver = m.make();
    solver->Init(0.0, 0.0, 1e8, 3);
    solver->RecordResidualHistory();
    set_up(*solver, m, *A, nullptr);
    solve(*solver, *A, s.b, Vec<T>(8, 0));
    testing::internal::CaptureStdout();
    solver->RecordHistory(file);
    EXPECT_NE(testing::internal::GetCapturedStdout().find("Writing residual history"),
              std::string::npos);

    const std::vector<double> written = read();
    std::remove(file.c_str());
    ASSERT_GE(written.size(), 3u);
    for(size_t j = 0; j < 3; ++j)
    {
        EXPECT_NEAR(written[j], s.res[j], tolerance * s.res[j]) << "residual " << j;
    }

    solver->InitMaxIter(1);
    const Outcome<T> out = solve(*solver, *A, s.b, s.x[2]);
    EXPECT_EQ(out.iterations, 1);
    expect_near(out.x,
                min_residual_iterate(s.D, s.b, s.x[2], 1, identity<T>, identity<T>),
                loose_tol<T>(1e-10));
    testing::internal::CaptureStdout();
    solver->RecordHistory(file);
    testing::internal::GetCapturedStdout();
    const std::vector<double> second = read();
    std::remove(file.c_str());
    ASSERT_GE(second.size(), 1u);

    KNOWN_BUG("the file gets the first iteration-count entries of every recorded residual: the "
              "last residual is missing, and a second solve writes the residuals of the first");
    EXPECT_EQ(written.size(), 4u);
    EXPECT_NEAR(second[0], s.res[2], tolerance * s.res[2]);
    EXPECT_EQ(second.size(), 2u);
}

TYPED_TEST(solvers, solvers_return_at_once_when_the_initial_guess_solves_the_system)
{
    using T     = typename TypeParam::value_type;
    const int n = 8;
    Vec<T>    exact(n);
    for(int i = 0; i < n; ++i)
    {
        exact[i] = static_cast<T>(i % 5 - 2);
    }

    for(const Problem<T>& p : problems<T>(n))
    {
        auto         A = this->make(p.A);
        const Vec<T> b = multiply(dense(p.A), exact);
        for(bool precond : {false, true})
        {
            if(p.method.needs_preconditioner && !precond)
            {
                continue;
            }
            for(bool zero : {true, false})
            {
                SCOPED_TRACE(trace(
                    p.method, precond, zero ? "zero right-hand side" : "exact initial guess"));
                Jacobi<T> jacobi;
                auto      solver = p.method.make();
                solver->Init(0.0, 1e-10, 1e8, 100);
                set_up(*solver, p.method, *A, precond ? &jacobi : nullptr);

                const Vec<T> x0 = zero ? Vec<T>(n, 0) : exact;
                // BiCGStab(l) goes on to a breakdown it reports
                const Outcome<T> out
                    = quietly([&] { return solve(*solver, *A, zero ? Vec<T>(n, 0) : b, x0); });
                EXPECT_EQ(out.x, x0);
                EXPECT_EQ(out.iterations, 0);
                EXPECT_EQ(out.status, 1);
            }
        }
    }
}

TYPED_TEST(solvers, solvers_return_at_once_when_the_initial_residual_is_nan_or_within_the_tolerance)
{
    using T     = typename TypeParam::value_type;
    const int n = 8;
    Vec<T>    exact(n);
    for(int i = 0; i < n; ++i)
    {
        exact[i] = static_cast<T>(i % 5 - 2);
    }
    Vec<T> close = exact;
    close[3] += 1e-3;
    Vec<T> nan = exact;
    nan[3]     = std::numeric_limits<T>::quiet_NaN();

    std::vector<Problem<T>> later;
    for(const Problem<T>& p : problems<T>(n))
    {
        auto         A = this->make(p.A);
        const Vec<T> b = multiply(dense(p.A), exact);
        {
            SCOPED_TRACE(
                trace(p.method, p.method.needs_preconditioner, "NaN in the initial guess"));
            Jacobi<T> jacobi;
            auto      solver = p.method.make();
            solver->Init(0.0, 1e-10, 1e8, 100);
            set_up(*solver, p.method, *A, p.method.needs_preconditioner ? &jacobi : nullptr);

            testing::internal::CaptureStdout();
            const Outcome<T> out = solve(*solver, *A, b, nan);
            EXPECT_NE(testing::internal::GetCapturedStdout().find("Residual = nan"),
                      std::string::npos);
            EXPECT_EQ(out.iterations, 0);
            EXPECT_EQ(out.status, 0);
        }
        if(is_bicgstabl(p.method))
        {
            later.push_back(p);
            continue;
        }
        SCOPED_TRACE(trace(p.method, p.method.needs_preconditioner, "within the tolerance"));
        Jacobi<T> jacobi;
        auto      solver = p.method.make();
        solver->Init(1.0, 1e-10, 1e8, 100);
        set_up(*solver, p.method, *A, p.method.needs_preconditioner ? &jacobi : nullptr);

        const Outcome<T> out = solve(*solver, *A, b, close);
        EXPECT_EQ(out.x, close);
        EXPECT_EQ(out.iterations, 0);
        EXPECT_EQ(out.status, 1);
    }

    KNOWN_BUG("BiCGStab(l) ignores that the initial residual already meets the tolerance");
    for(const Problem<T>& p : later)
    {
        SCOPED_TRACE(trace(p.method, false, "within the tolerance"));
        auto A      = this->make(p.A);
        auto solver = p.method.make();
        solver->Init(1.0, 1e-10, 1e8, 100);
        set_up(*solver, p.method, *A, nullptr);
        const Outcome<T> out = solve(*solver, *A, multiply(dense(p.A), exact), close);
        EXPECT_EQ(out.x, close);
    }
}

TYPED_TEST(solvers, cg_stops_with_status_zero_when_the_residual_is_not_finite)
{
    using T = typename TypeParam::value_type;
    // Indefinite: (p, A p) = 0 in the first iteration
    const Csr<T> ref = diagonal<T>({1.0, -1.0});
    auto         A   = this->make(ref);
    CG<T>        solver;
    solver.Init(0.0, 1e-10, 1e8, 10);
    set_up(solver, method<CG<T>>("CG"), *A, nullptr);
    solver.Verbose(1);

    testing::internal::CaptureStdout();
    const Outcome<T>  out    = solve(solver, *A, {1.0, 1.0}, {0.0, 0.0});
    const std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("Residual = "), std::string::npos);
    EXPECT_NE(output.find("NO criteria"), std::string::npos);
    EXPECT_EQ(out.iterations, 1);
    EXPECT_EQ(out.status, 0);
    EXPECT_FALSE(std::isfinite(out.residual));
}

// (A r, r) = 0 for a skew-symmetric A, exactly for integer vectors
TYPED_TEST(solvers, bicgstabl_stops_without_an_update_when_its_first_step_breaks_down)
{
    using T = typename TypeParam::value_type;
    Csr<T>    ref(4);
    const int col[]   = {1, 0, 3, 2};
    const T   value[] = {1.0, -1.0, 2.0, -2.0};
    for(int i = 0; i < 4; ++i)
    {
        ref.add(col[i], value[i]);
        ref.end_row();
    }
    auto A = this->make(ref);

    for(int l : {1, 2})
    {
        SCOPED_TRACE("l = " + std::to_string(l));
        const Method<T> m      = bicgstabl<T>(l);
        auto            solver = m.make();
        solver->Init(0.0, 1e-10, 1e8, 100);
        set_up(*solver, m, *A, nullptr);

        testing::internal::CaptureStdout();
        const Outcome<T> out = solve(*solver, *A, {1.0, 2.0, 3.0, 4.0}, Vec<T>(4, 0));
        EXPECT_NE(testing::internal::GetCapturedStdout().find("sigma == 0"), std::string::npos);
        EXPECT_EQ(out.x, Vec<T>(4, 0));
        EXPECT_EQ(out.iterations, 0);
        EXPECT_EQ(out.status, 0);
    }
}

namespace
{
    // A skew-symmetric matrix with stored zero diagonal entries, so that Jacobi inverts
    // them to one and is the identity
    template <typename T>
    Csr<T> skew_with_zero_diagonal()
    {
        Csr<T>    ref(4);
        const int col[]   = {1, 0, 3, 2};
        const T   value[] = {1.0, -1.0, 2.0, -2.0};
        for(int i = 0; i < 4; ++i)
        {
            ref.add(std::min(i, col[i]), i < col[i] ? T(0) : value[i]);
            ref.add(std::max(i, col[i]), i < col[i] ? value[i] : T(0));
            ref.end_row();
        }
        return ref;
    }

    template <typename T>
    Outcome<T> solve_reporting(Iterative<T>&      solver,
                               const Matrix<T>&   A,
                               const Vec<T>&      b,
                               const std::string& message)
    {
        testing::internal::CaptureStdout();
        const Outcome<T> out = solve(solver, A, b, Vec<T>(b.size(), 0));
        EXPECT_NE(testing::internal::GetCapturedStdout().find(message), std::string::npos)
            << "no \"" << message << "\"";
        return out;
    }
}

TYPED_TEST(solvers,
           preconditioned_bicgstabl_stops_without_an_update_when_its_first_step_breaks_down)
{
    using T = typename TypeParam::value_type;
    auto A  = this->make(skew_with_zero_diagonal<T>());

    for(int l : {1, 2})
    {
        SCOPED_TRACE("l = " + std::to_string(l));
        const Method<T> m = bicgstabl<T>(l);
        Jacobi<T>       jacobi;
        auto            solver = m.make();
        solver->Init(0.0, 1e-10, 1e8, 100);
        quietly([&] { set_up(*solver, m, *A, &jacobi); });

        const Outcome<T> out = solve_reporting(*solver, *A, {1.0, 2.0, 3.0, 4.0}, "sigma == 0");
        EXPECT_EQ(out.x, Vec<T>(4, 0));
        EXPECT_EQ(out.iterations, 0);
        EXPECT_EQ(out.status, 0);
    }
}

// (r0, r1) = 0 after an exact first iteration: alpha = 1, omega = -1/4 without and
// 1/2 with Jacobi, and r1 = (1, 0, -1)
TYPED_TEST(solvers, bicgstab_stops_when_the_residual_is_orthogonal_to_the_shadow_residual)
{
    using T          = typename TypeParam::value_type;
    const Vec<T> D   = {-2, 1, 0, 0, 1, 0, -2, 0, -2};
    const Csr<T> ref = from_dense(3, 3, D);
    const Vec<T> b   = {0, -2, 0};
    auto         A   = this->make(ref);

    for(bool precond : {false, true})
    {
        const Method<T> m = method<BiCGStab<T>>("BiCGStab");
        SCOPED_TRACE(trace(m, precond));
        Jacobi<T> jacobi;
        auto      solver = m.make();
        solver->Init(0.0, 0.0, 1e8, 10);
        set_up(*solver, m, *A, precond ? &jacobi : nullptr);

        const Outcome<T> out = solve_reporting(*solver, *A, b, "rho == 0");
        EXPECT_EQ(out.x, bicgstab_iterate(D, b, Vec<T>(3, 0), 1, preconditioner(D, precond)));
        EXPECT_EQ(out.iterations, 1);
        EXPECT_EQ(out.status, 0);
    }
}

// Jacobi is the identity on these unit diagonals. A known bug counts a residual check as
// an iteration, which makes the iteration counts unreliable.
TYPED_TEST(solvers, qmrcgstab_stops_at_a_breakdown_of_its_second_iteration)
{
    using T = typename TypeParam::value_type;
    // (r0, A p) = 0 at the start of the second iteration
    const Vec<T> shadow_breakdown = {1, 0, 0, 0, 1, -2, 0, -1, 1};
    // Singular: A s = 0 for the second half step residual s
    const Vec<T> step_breakdown = {1, 1, -1, 1, 1, -1, -1, 0, 1};

    struct Case
    {
        std::string name;
        Vec<T>      D;
        Vec<T>      b;
        Vec<T>      x;
    };
    const Case cases[]
        = {{"rho_old == 0",
            shadow_breakdown,
            {-1, 0, -1},
            qmrcgstab_iterate(shadow_breakdown, {-1, 0, -1}, Vec<T>(3, 0), 1, identity<T>)},
           {"omega == 0",
            step_breakdown,
            {2, 0, -2},
            qmrcgstab_iterate(step_breakdown, {2, 0, -2}, Vec<T>(3, 0), 2, identity<T>, true)}};

    for(const Case& c : cases)
    {
        auto A = this->make(from_dense(3, 3, c.D));
        for(bool precond : {false, true})
        {
            const Method<T> m = method<QMRCGStab<T>>("QMRCGStab");
            SCOPED_TRACE(trace(m, precond, c.name));
            Jacobi<T> jacobi;
            auto      solver = m.make();
            solver->Init(0.0, 0.0, 1e8, 10);
            set_up(*solver, m, *A, precond ? &jacobi : nullptr);

            const Outcome<T> out = solve_reporting(*solver, *A, c.b, c.name);
            expect_near(out.x, c.x, loose_tol<T>(1e-12));
            EXPECT_EQ(out.status, 0);
        }
    }
}

// A divergence tolerance below one stops at the first residual above that fraction of the
// initial residual, also at the checks of the BiCG part of BiCGStab(l)
TYPED_TEST(solvers, divergence_tolerance_below_one_stops_at_the_first_check)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = nonsymmetric<T>(8);
    auto         A   = this->make(ref);

    for(const Method<T>& m : {with_basis<GMRES<T>>("GMRES", 30), bicgstabl<T>(2)})
    {
        SCOPED_TRACE(m.name);
        auto solver = m.make();
        solver->Init(0.0, 0.0, 1e-3, 100);
        set_up(*solver, m, *A, nullptr);

        const Outcome<T> out = solve(*solver, *A, rhs<T>(8), Vec<T>(8, 0));
        EXPECT_EQ(out.status, 3);
        // BiCGStab(l) counts completed cycles
        EXPECT_EQ(out.iterations, is_bicgstabl(m) ? 0 : 1);
    }
}

// A breakdown stops IDR and is reported, as a breakdown stops BiCGStab(l)
TYPED_TEST(solvers, idr_stops_when_the_shadow_space_breaks_down)
{
    using T = typename TypeParam::value_type;
    KNOWN_BUG("IDR ends the program on a breakdown instead of stopping");
    expect_in_child([&] {
        // A P^T A U block of zero
        const Csr<T> ref = diagonal(Vec<T>(4, 0));
        auto         A   = this->make(ref);
        IDR<T>       solver;
        solver.Init(0.0, 1e-10, 1e8, 10);
        set_up(solver, idr<T>(2), *A, nullptr);
        solve_reporting(solver, *A, rhs<T>(4), "M(k,k) == 0");
    });
}

// The first step of IDR(1) takes U = r = b and M = (P, A b). On the zero matrix M is
// zero. On the matrix with off-diagonal entries max / 2, the signs of b against those
// of P make M = |P_0| inf or |P_0| inf - |P_1| inf. On the rotation, (A r, r) = 0
// makes omega = 0 * (kappa / 0) in the dimension reduction step.
TYPED_TEST(solvers, idr_reports_the_breakdown_and_stops)
{
    using T = typename TypeParam::value_type;
    Csr<T>  zero(2), huge(2), rotation(2);
    const T h = std::numeric_limits<T>::max() / 2;
    for(int i = 0; i < 2; ++i)
    {
        zero.add(i, 0.0);
        zero.end_row();
        huge.add(0, i == 0 ? T(1) : h);
        huge.add(1, i == 0 ? h : T(1));
        huge.end_row();
        rotation.add(0, i == 0 ? T(0) : T(-1));
        rotation.add(1, i == 0 ? T(1) : T(0));
        rotation.end_row();
    }

    // The shadow vector of IDR(1) before its normalization
    auto P = make_local_vector(this->backend(), Vec<T>(2, 0));
    P->SetRandomNormal(12345ULL, 0.0, 1.0);
    const Vec<T> p     = values_of(*P);
    const T      sign0 = p[0] < 0 ? -1 : 1;
    const T      sign1 = p[1] < 0 ? -1 : 1;

    struct Case
    {
        const Csr<T>* A;
        Vec<T>        b;
        const char*   message;
    };
    const Case cases[] = {{&zero, {1, 2}, "M(k,k) == 0"},
                          {&huge, {0, 10 * sign0}, "M(k,k) == inf"},
                          {&huge, {-10 * sign1, 10 * sign0}, "M(k,k) == NaN"},
                          {&rotation, {1, 2}, "w == NaN"}};

    KNOWN_BUG("IDR ends the program on a breakdown instead of stopping");
    for(const Case& c : cases)
    {
        for(bool precond : {false, true})
        {
            SCOPED_TRACE(std::string(c.message) + (precond ? " with Jacobi" : ""));
            expect_in_child([&] {
                auto      A = this->make(*c.A);
                Jacobi<T> jacobi;
                IDR<T>    solver;
                solver.Init(0.0, 0.0, 1e8, 10);
                quietly([&] { set_up(solver, idr<T>(1), *A, precond ? &jacobi : nullptr); });
                solve_reporting(solver, *A, c.b, c.message);
            });
        }
    }
}

TYPED_TEST(solvers, non_finite_tolerances_are_reported)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = spd<T>(8);
    auto         A   = this->make(ref);
    CG<T>        solver;

    testing::internal::CaptureStdout();
    solver.Init(
        std::numeric_limits<double>::quiet_NaN(), 0.0, std::numeric_limits<double>::quiet_NaN(), 3);
    solver.InitTol(0.0, std::numeric_limits<double>::infinity(), 1e8);
    const std::string out = testing::internal::GetCapturedStdout();
    EXPECT_NE(out.find("Abs tol = nan"), std::string::npos);
    EXPECT_NE(out.find("Rel tol = inf"), std::string::npos);
    EXPECT_NE(out.find("Div tol = nan"), std::string::npos);

    // NaN tolerances are never met
    testing::internal::CaptureStdout();
    solver.InitTol(
        std::numeric_limits<double>::quiet_NaN(), 0.0, std::numeric_limits<double>::quiet_NaN());
    set_up(solver, method<CG<T>>("CG"), *A, nullptr);
    const Outcome<T> result = solve(solver, *A, rhs<T>(8), Vec<T>(8, 0));
    testing::internal::GetCapturedStdout();
    EXPECT_EQ(result.iterations, 3);
    EXPECT_EQ(result.status, 4);
}

TYPED_TEST(solvers, verbose_solvers_report_the_stopping_criterion)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = spd<T>(8);
    const Vec<T> b   = rhs<T>(8);
    auto         A   = this->make(ref);

    struct Case
    {
        std::string              name;
        int                      verbosity;
        std::vector<std::string> expected;
    };
    for(const Case& c : {Case{"verbose 1", 1, {"CG", "initial residual", "RELATIVE criteria"}},
                         Case{"verbose 2", 2, {"IterationControl iter=1; residual="}}})
    {
        SCOPED_TRACE(c.name);
        CG<T> solver;
        solver.Init(0.0, 1e-6, 1e8, 100);
        set_up(solver, method<CG<T>>("CG"), *A, nullptr);
        solver.Verbose(c.verbosity);
        testing::internal::CaptureStdout();
        solve(solver, *A, b, Vec<T>(8, 0));
        const std::string out = testing::internal::GetCapturedStdout();
        for(const std::string& e : c.expected)
        {
            EXPECT_NE(out.find(e), std::string::npos) << e;
        }
    }

    {
        SCOPED_TRACE("FixedPoint");
        Jacobi<T>     jacobi;
        FixedPoint<T> solver;
        solver.Init(0.0, 1e-6, 1e8, 100);
        set_up(solver, fixed_point<T>(), *A, &jacobi);
        solver.Verbose(1);
        auto rhs_vector = make_local_vector(*A, b);
        auto x          = make_local_vector(*A, Vec<T>(8, 0));
        testing::internal::CaptureStdout();
        solver.Solve(*rhs_vector, x.get());
        solver.SolveZeroSol(*rhs_vector, x.get());
        const std::string out = testing::internal::GetCapturedStdout();
        EXPECT_NE(out.find("Fixed Point Iteration solver starts"), std::string::npos);
        EXPECT_NE(out.find("RELATIVE criteria"), std::string::npos);
    }

    {
        SCOPED_TRACE("LU");
        LU<T> solver;
        solver.SetOperator(*A);
        quietly([&] { solver.Build(); });
        solver.Verbose(1);
        auto rhs_vector = make_local_vector(*A, b);
        auto x          = make_local_vector(*A, Vec<T>(8, 0));
        testing::internal::CaptureStdout();
        solver.Solve(*rhs_vector, x.get());
        const std::string out = testing::internal::GetCapturedStdout();
        EXPECT_NE(out.find("LU direct solver starts"), std::string::npos);
        EXPECT_NE(out.find("LU ends"), std::string::npos);
    }
}

TYPED_TEST(solvers, solve_zero_sol_starts_from_zero)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = spd<T>(8);
    const Vec<T> D   = dense(ref);
    const Vec<T> b   = rhs<T>(8);
    auto         A   = this->make(ref);

    Jacobi<T> jacobi;
    CG<T>     solver;
    solver.Init(0.0, 0.0, 1e8, 3);
    set_up(solver, method<CG<T>>("CG"), *A, &jacobi);

    auto rhs_vector = make_local_vector(*A, b);
    auto x          = make_local_vector(*A, Vec<T>(8, 1000));
    solver.SolveZeroSol(*rhs_vector, x.get());
    expect_near(values_of(*x),
                galerkin_iterate(D, b, Vec<T>(8, 0), 3, preconditioner(D, true)),
                loose_tol<T>(1e-10));
}

TYPED_TEST(solvers, a_krylov_solver_serves_as_variable_preconditioner_of_fgmres)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = nonsymmetric<T>(16);
    const Vec<T> b   = rhs<T>(16);
    auto         A   = this->make(ref);

    GMRES<T> inner;
    inner.Verbose(0);
    inner.Init(0.0, 0.0, 1e8, 2);
    FGMRES<T> solver;
    solver.Init(0.0, solver_tol<T>(1e-12), 1e8, 100);
    set_up(solver, method<FGMRES<T>>("FGMRES"), *A, &inner);

    const Outcome<T> out = solve(solver, *A, b, Vec<T>(16, 0));
    EXPECT_EQ(out.status, 2);
    expect_near(out.x,
                dense_solve(dense(ref), b),
                solution_tol(dense(ref), b, Vec<T>(16, 0), solver_tol<T>(1e-12)));

    // The inner solver starts from zero, so it is the GMRES(2) iterate
    const Vec<T> D           = dense(ref);
    const Op<T>  inner_gmres = [&](const Vec<T>& v) {
        return min_residual_iterate(D, v, Vec<T>(16, 0), 2, identity<T>, identity<T>);
    };
    for(int k = 1; k <= 4; ++k)
    {
        SCOPED_TRACE(std::to_string(k) + " iterations");
        GMRES<T> fixed_inner;
        fixed_inner.Verbose(0);
        fixed_inner.Init(0.0, 0.0, 1e8, 2);
        FGMRES<T> fixed;
        fixed.Init(0.0, 0.0, 1e8, k);
        set_up(fixed, method<FGMRES<T>>("FGMRES"), *A, &fixed_inner);
        expect_near(solve(fixed, *A, b, start<T>(16)).x,
                    flexible_iterate(D, b, start<T>(16), k, inner_gmres),
                    loose_tol<T>(1e-10));
    }
}

// A stale preconditioner still converges to the solution of the new values, so the
// iterates after a fixed number of iterations are compared with those of a solver built
// on the new values. Adding 1 + i % 4 to the diagonal changes the Jacobi diagonal by a
// factor that varies from row to row, which the iterates see.
TYPED_TEST(solvers, rebuild_numeric_takes_new_matrix_values)
{
    using T     = typename TypeParam::value_type;
    const int n = 16;
    for(const Problem<T>& p : problems<T>(n))
    {
        if(rebuild_breaks(p.method))
        {
            continue;
        }
        const Csr<T> changed = transformed(
            p.A, [](int i, int j, T v) { return i == j ? v + static_cast<T>(1 + i % 4) : v; });
        for(bool precond : {false, true})
        {
            if(p.method.needs_preconditioner && !precond)
            {
                continue;
            }
            SCOPED_TRACE(trace(p.method, precond));
            const Vec<T> b = rhs<T>(n);

            auto      fresh_A = this->make(changed);
            Jacobi<T> fresh_jacobi;
            auto      fresh = p.method.make();
            fresh->Init(0.0, 0.0, 1e8, 3);
            set_up(*fresh, p.method, *fresh_A, precond ? &fresh_jacobi : nullptr);
            const Vec<T> expected = solve(*fresh, *fresh_A, b, Vec<T>(n, 0)).x;

            auto      A = this->make(p.A);
            Jacobi<T> jacobi;
            auto      solver = p.method.make();
            solver->Init(0.0, 0.0, 1e8, 3);
            set_up(*solver, p.method, *A, precond ? &jacobi : nullptr);
            solve(*solver, *A, b, Vec<T>(n, 0));

            A->CopyFromCSR(changed.row_offset.data(), changed.col.data(), changed.val.data());
            solver->ReBuildNumeric();
            expect_near(solve(*solver, *A, b, Vec<T>(n, 0)).x, expected, tol<T>());
        }
    }

    // ReBuildNumeric of a solver that is not built builds it
    for(const Problem<T>& p : problems<T>(n))
    {
        for(bool precond : {false, true})
        {
            if(p.method.needs_preconditioner && !precond)
            {
                continue;
            }
            SCOPED_TRACE(trace(p.method, precond, "not built before"));
            auto      A = this->make(p.A);
            Jacobi<T> jacobi;
            auto      solver = p.method.make();
            solver->Verbose(0);
            solver->Init(0.0, solver_tol<T>(1e-12), 1e8, 1000);
            solver->SetOperator(*A);
            if(p.method.prepare)
            {
                p.method.prepare(*solver);
            }
            if(precond)
            {
                solver->SetPreconditioner(jacobi);
            }
            solver->ReBuildNumeric();

            const Outcome<T> out = solve(*solver, *A, rhs<T>(n), Vec<T>(n, 0));
            EXPECT_TRUE(out.status == 1 || out.status == 2) << "status " << out.status;
            expect_near(out.x,
                        dense_solve(dense(p.A), rhs<T>(n)),
                        solution_tol(
                            dense(p.A), rhs<T>(n), Vec<T>(n, 0), solver_tol<T>(1e-12), 2, precond));
        }
    }
}

TYPED_TEST(solvers, rebuild_numeric_keeps_idr_and_chebyshev_able_to_solve)
{
    using T          = typename TypeParam::value_type;
    const int    n   = 8;
    const Csr<T> ref = spd<T>(n);
    // Adding 1 to the diagonal keeps the spectra of A and D^-1 A in [2, n + 3] and
    // [1/2, 3/2], within the bounds of chebyshev(n)
    const Csr<T> shifted
        = transformed(ref, [](int i, int j, T v) { return i == j ? v + T(1) : v; });
    const Vec<T> b = rhs<T>(n);
    const Vec<T> x = dense_solve(dense(shifted), b);

    struct Case
    {
        std::string                   name;
        std::unique_ptr<Matrix<T>>    A;
        std::unique_ptr<Jacobi<T>>    jacobi;
        std::unique_ptr<Iterative<T>> solver;
        double                        tolerance;
    };
    std::vector<Case> cases;
    for(const Method<T>& m : {idr<T>(2), chebyshev<T>(n)})
    {
        for(bool precond : {false, true})
        {
            SCOPED_TRACE(trace(m, precond));
            Case c{trace(m, precond),
                   this->make(ref),
                   std::make_unique<Jacobi<T>>(),
                   m.make(),
                   solution_tol(dense(shifted), b, Vec<T>(n, 0), solver_tol<T>(1e-12), 2, precond)};
            c.solver->Init(0.0, solver_tol<T>(1e-12), 1e8, 1000);
            set_up(*c.solver, m, *c.A, precond ? c.jacobi.get() : nullptr);
            expect_near(
                solve(*c.solver, *c.A, b, Vec<T>(n, 0)).x,
                dense_solve(dense(ref), b),
                solution_tol(dense(ref), b, Vec<T>(n, 0), solver_tol<T>(1e-12), 2, precond));

            c.A->AddScalarDiagonal(1.0);
            c.solver->ReBuildNumeric();
            cases.push_back(std::move(c));
        }
    }

    KNOWN_BUG("IDR zeros its shadow space, which ends the next solve, and Chebyshev marks itself "
              "as not built, which fails an assert in the next solve");
    for(Case& c : cases)
    {
        SCOPED_TRACE(c.name);
        expect_completes([&] {
            const Outcome<T> out = solve(*c.solver, *c.A, b, Vec<T>(n, 0));
            return near(out.x, x, c.tolerance);
        });
    }
}

TYPED_TEST(solvers, building_again_keeps_the_preconditioner)
{
    using T = typename TypeParam::value_type;
    for(const Problem<T>& p : problems<T>(16))
    {
        if(p.method.needs_preconditioner || p.method.name == "Chebyshev")
        {
            continue;
        }
        SCOPED_TRACE(trace(p.method, false, "without preconditioner"));
        auto A      = this->make(p.A);
        auto solver = p.method.make();
        solver->Init(0.0, solver_tol<T>(1e-12), 1e8, 1000);
        set_up(*solver, p.method, *A, nullptr);
        solver->Build();
        const Outcome<T> out = solve(*solver, *A, rhs<T>(16), Vec<T>(16, 0));
        EXPECT_TRUE(out.status == 1 || out.status == 2) << "status " << out.status;
        expect_near(out.x,
                    dense_solve(dense(p.A), rhs<T>(16)),
                    solution_tol(dense(p.A), rhs<T>(16), Vec<T>(16, 0), solver_tol<T>(1e-12)));
    }

    const Csr<T> ref      = spd<T>(8);
    const Vec<T> D        = dense(ref);
    const Vec<T> b        = rhs<T>(8);
    const Vec<T> expected = galerkin_iterate(D, b, Vec<T>(8, 0), 3, preconditioner(D, true));
    auto         A        = this->make(ref);

    Jacobi<T> jacobi;
    CG<T>     solver;
    solver.Init(0.0, 0.0, 1e8, 3);
    set_up(solver, method<CG<T>>("CG"), *A, &jacobi);
    expect_near(solve(solver, *A, b, Vec<T>(8, 0)).x, expected, loose_tol<T>(1e-10));

    // The second Build of Chebyshev completes, only its next solve fails; the one of
    // FixedPoint fails an assert itself
    const Method<T> cheb = chebyshev<T>(8);
    Jacobi<T>       cheb_jacobi;
    auto            cheb_solver = cheb.make();
    cheb_solver->Init(0.0, solver_tol<T>(1e-12), 1e8, 1000);
    set_up(*cheb_solver, cheb, *A, &cheb_jacobi);
    cheb_solver->Build();

    solver.Build();
    const Outcome<T> rebuilt = solve(solver, *A, b, Vec<T>(8, 0));
    EXPECT_EQ(rebuilt.iterations, 3);
    EXPECT_EQ(rebuilt.status, 4);

    const double tolerance = solution_tol(D, b, Vec<T>(8, 0), solver_tol<T>(1e-12), 2, true);
    KNOWN_BUG("Build of a built solver calls Clear, which drops the preconditioner and the "
              "Chebyshev bounds; FixedPoint and Chebyshev then fail an assert");
    expect_near(rebuilt.x, expected, loose_tol<T>(1e-10));

    {
        SCOPED_TRACE(cheb.name);
        expect_completes([&] {
            return near(solve(*cheb_solver, *A, b, Vec<T>(8, 0)).x, dense_solve(D, b), tolerance);
        });
    }

    {
        SCOPED_TRACE("FixedPoint");
        expect_completes([&] {
            const Method<T> m = fixed_point<T>();
            Jacobi<T>       p;
            auto            s = m.make();
            s->Init(0.0, solver_tol<T>(1e-12), 1e8, 1000);
            set_up(*s, m, *A, &p);
            s->Build();
            return near(solve(*s, *A, b, Vec<T>(8, 0)).x, dense_solve(D, b), tolerance);
        });
    }
}

TYPED_TEST(solvers, a_cleared_solver_can_be_set_up_for_another_system)
{
    using T            = typename TypeParam::value_type;
    const Csr<T> first = spd<T>(8);
    const Csr<T> second
        = transformed(spd<T>(8), [](int i, int j, T v) { return i == j ? v + 3 : v; });
    const Vec<T> D = dense(second);
    const Vec<T> b = rhs<T>(8);
    auto         A = this->make(first);
    auto         B = this->make(second);

    Jacobi<T> jacobi;
    CG<T>     solver;
    solver.Init(0.0, 0.0, 1e8, 3);
    set_up(solver, method<CG<T>>("CG"), *A, &jacobi);
    solve(solver, *A, b, Vec<T>(8, 0));

    solver.Clear();
    solver.Clear();
    set_up(solver, method<CG<T>>("CG"), *B, &jacobi);
    expect_near(solve(solver, *B, b, Vec<T>(8, 0)).x,
                galerkin_iterate(D, b, Vec<T>(8, 0), 3, preconditioner(D, true)),
                loose_tol<T>(1e-10));
}

TYPED_TEST(solvers, clear_and_moves_before_the_build_leave_a_solver_to_set_up)
{
    using T     = typename TypeParam::value_type;
    const int n = 16;

    for(const Problem<T>& p : problems<T>(n))
    {
        SCOPED_TRACE(p.method.name);
        auto         A = this->make(p.A);
        const Vec<T> b = rhs<T>(n);

        Jacobi<T> jacobi;
        auto      solver = p.method.make();
        solver->Clear();
        solver->MoveToAccelerator();
        solver->MoveToHost();
        solver->Init(0.0, solver_tol<T>(1e-12), 1e8, 1000);
        set_up(*solver, p.method, *A, p.method.needs_preconditioner ? &jacobi : nullptr);

        const Outcome<T> out = solve(*solver, *A, b, Vec<T>(n, 0));
        EXPECT_TRUE(out.status == 1 || out.status == 2) << "status " << out.status;
        expect_near(out.x,
                    dense_solve(dense(p.A), b),
                    solution_tol(dense(p.A),
                                 b,
                                 Vec<T>(n, 0),
                                 solver_tol<T>(1e-12),
                                 2,
                                 p.method.needs_preconditioner));
    }
}

namespace
{
    // The defaults of Solver for everything but Solve, which copies the right-hand side
    template <typename T>
    class CopySolver : public Solver<T>
    {
    public:
        void Print(void) const override {}

        void Solve(const Vector<T>& rhs, Vector<T>* x) override
        {
            x->CopyFrom(rhs);
        }

        // Solver::Clear deletes the preconditioner
        void Own(Solver<T>* precond)
        {
            this->precond_ = precond;
        }

        bool built() const
        {
            return this->build_;
        }

        bool has_preconditioner() const
        {
            return this->precond_ != nullptr;
        }

        bool has_operator() const
        {
            return this->op_ != nullptr;
        }

        int verbosity() const
        {
            return this->verb_;
        }

    protected:
        void PrintStart_(void) const override {}
        void PrintEnd_(void) const override {}
        void MoveToHostLocalData_(void) override {}
        void MoveToAcceleratorLocalData_(void) override {}
    };
}

TYPED_TEST(solvers, the_defaults_of_solver_build_rebuild_and_clear_its_state)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = spd<T>(4);
    const Vec<T> b   = rhs<T>(4);
    auto         A   = this->make(ref);

    CopySolver<T> solver;
    solver.Verbose(2);
    EXPECT_EQ(solver.verbosity(), 2);
    solver.SetOperator(*A);
    solver.Own(new Jacobi<T>);
    solver.Build();
    EXPECT_TRUE(solver.built());
    EXPECT_TRUE(solver.has_preconditioner());

    solver.MoveToHost();
    solver.MoveToAccelerator();
    solver.Sync();
    auto rhs_vector = make_local_vector(*A, b);
    auto x          = make_local_vector(*A, Vec<T>(4, 1));
    solver.SolveZeroSol(*rhs_vector, x.get());
    EXPECT_EQ(values_of(*x), b);

    solver.Build();
    EXPECT_TRUE(solver.built());
    solver.ReBuildNumeric();
    EXPECT_TRUE(solver.built());
    const bool kept = solver.has_preconditioner() && solver.has_operator();

    solver.Clear();
    EXPECT_FALSE(solver.built());
    EXPECT_FALSE(solver.has_preconditioner());

    KNOWN_BUG("Build and ReBuildNumeric of a built solver call Clear, which deletes the "
              "preconditioner and drops the operator");
    EXPECT_TRUE(kept);
}

TYPED_TEST(solvers, reset_operator_solves_with_the_new_operator)
{
    using T            = typename TypeParam::value_type;
    const Csr<T> first = spd<T>(8);
    const Csr<T> second
        = transformed(spd<T>(8), [](int i, int j, T v) { return i == j ? v + 3 : v; });
    const Vec<T> b = rhs<T>(8);
    auto         A = this->make(first);
    auto         B = this->make(second);

    CG<T> solver;
    solver.Init(0.0, solver_tol<T>(1e-12), 1e8, 100);
    set_up(solver, method<CG<T>>("CG"), *A, nullptr);
    solver.ResetOperator(*B);
    expect_near(solve(solver, *B, b, Vec<T>(8, 0)).x,
                dense_solve(dense(second), b),
                solution_tol(dense(second), b, Vec<T>(8, 0), solver_tol<T>(1e-12)));
}

TYPED_TEST(solvers, a_solver_built_on_one_backend_solves_after_moving_to_the_other)
{
    using T = typename TypeParam::value_type;
    if(!available(Backend::Accelerator))
    {
        GTEST_SKIP() << "no accelerator available";
    }
    const Backend other = this->backend() == Backend::Host ? Backend::Accelerator : Backend::Host;
    const int     n     = 16;

    for(const Problem<T>& p : problems<T>(n))
    {
        for(bool precond : {false, true})
        {
            if(p.method.needs_preconditioner && !precond)
            {
                continue;
            }
            SCOPED_TRACE(trace(p.method, precond));
            auto         A = make_local_matrix(other, p.A);
            const Vec<T> b = rhs<T>(n);

            Jacobi<T> jacobi;
            auto      solver = p.method.make();
            solver->Init(0.0, solver_tol<T>(1e-12), 1e8, 1000);
            if(this->backend() == Backend::Host)
            {
                solver->MoveToHost();
            }
            else
            {
                solver->MoveToAccelerator();
            }
            set_up(*solver, p.method, *A, precond ? &jacobi : nullptr);

            if(this->backend() == Backend::Host)
            {
                A->MoveToHost();
                solver->MoveToHost();
            }
            else
            {
                A->MoveToAccelerator();
                solver->MoveToAccelerator();
            }

            const Outcome<T> out = solve(*solver, *A, b, Vec<T>(n, 0));
            EXPECT_TRUE(out.status == 1 || out.status == 2) << "status " << out.status;
            expect_near(
                out.x,
                dense_solve(dense(p.A), b),
                solution_tol(dense(p.A), b, Vec<T>(n, 0), solver_tol<T>(1e-12), 2, precond));
        }
    }
}

TYPED_TEST(solvers, asynchronous_build_solves_on_the_accelerator_after_sync)
{
    using T = typename TypeParam::value_type;
    if(this->backend() != Backend::Accelerator)
    {
        GTEST_SKIP() << "the asynchronous build moves to the accelerator";
    }
    const Csr<T> ref = spd<T>(8);
    const Vec<T> b   = rhs<T>(8);

    for(const Method<T>& m : {method<CG<T>>("CG"), with_basis<GMRES<T>>("GMRES", 30)})
    {
        for(bool precond : {false, true})
        {
            SCOPED_TRACE(trace(m, precond));
            auto      A = make_local_matrix(Backend::Host, ref);
            Jacobi<T> jacobi;
            auto      solver = m.make();
            solver->Verbose(0);
            solver->Init(0.0, solver_tol<T>(1e-12), 1e8, 100);
            solver->SetOperator(*A);
            if(precond)
            {
                solver->SetPreconditioner(jacobi);
            }
            solver->BuildMoveToAcceleratorAsync();
            A->MoveToAccelerator();
            solver->Sync();
            if(!precond)
            {
                solver->BuildMoveToAcceleratorAsync();
                solver->Sync();
            }

            const Outcome<T> out = solve(*solver, *A, b, Vec<T>(8, 0));
            EXPECT_EQ(out.status, 2);
            expect_near(
                out.x,
                dense_solve(dense(ref), b),
                solution_tol(dense(ref), b, Vec<T>(8, 0), solver_tol<T>(1e-12), 2, precond));
        }
    }
}

TYPED_TEST(solvers, fixed_point_iterates_follow_the_relaxed_richardson_recurrence)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = nonsymmetric<T>(8);
    const Vec<T> D   = dense(ref);
    const Vec<T> b   = rhs<T>(8);
    const Vec<T> x0  = start<T>(8);
    const Op<T>  M   = preconditioner(D, true);
    auto         A   = this->make(ref);

    for(double omega : {1.0, 0.5})
    {
        for(bool smoother : {false, true})
        {
            for(int k = 0; k <= 3; ++k)
            {
                SCOPED_TRACE(std::string(smoother ? "smoother" : "solver") + ", omega "
                             + std::to_string(omega) + ", " + std::to_string(k) + " iterations");
                Jacobi<T>     jacobi;
                FixedPoint<T> solver;
                if(smoother)
                {
                    solver.FlagSmoother();
                }
                if(omega != 1.0)
                {
                    solver.SetRelaxation(omega);
                }
                solver.Init(0.0, 0.0, 1e8, k);
                set_up(solver, fixed_point<T>(), *A, &jacobi);

                const Outcome<T> out = solve(solver, *A, b, x0);
                expect_near(out.x, richardson_iterate(D, b, x0, k, M, omega), tol<T>());
                if(k > 0)
                {
                    // A smoother computes no residuals
                    EXPECT_EQ(out.status, smoother ? 0 : 4);
                    auto rhs_vector = make_local_vector(*A, b);
                    auto x          = make_local_vector(*A, Vec<T>(8, 1000));
                    solver.SolveZeroSol(*rhs_vector, x.get());
                    expect_near(values_of(*x),
                                richardson_iterate(D, b, Vec<T>(8, 0), k, M, omega),
                                tol<T>());
                }
            }
        }
    }
}

TYPED_TEST(solvers, fixed_point_solve_zero_sol_returns_zero_when_it_does_not_iterate)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = spd<T>(8);
    auto         A   = this->make(ref);

    struct Case
    {
        std::string name;
        bool        smoother;
        int         max_iter;
        Vec<T>      b;
    };
    std::vector<Vec<T>> results;
    for(const Case& c : {Case{"zero right-hand side", false, 10, Vec<T>(8, 0)},
                         Case{"no iterations", false, 0, rhs<T>(8)},
                         Case{"smoother without sweeps", true, 0, rhs<T>(8)}})
    {
        SCOPED_TRACE(c.name);
        Jacobi<T>     jacobi;
        FixedPoint<T> solver;
        if(c.smoother)
        {
            solver.FlagSmoother();
        }
        solver.Init(0.0, 0.0, 1e8, c.max_iter);
        set_up(solver, fixed_point<T>(), *A, &jacobi);

        auto rhs_vector = make_local_vector(*A, c.b);
        auto x          = make_local_vector(*A, Vec<T>(8, 1000));
        solver.SolveZeroSol(*rhs_vector, x.get());
        results.push_back(values_of(*x));
    }

    KNOWN_BUG("FixedPoint::SolveZeroSol returns before it sets x to zero when the initial residual "
              "meets the tolerance or there are no iterations");
    for(const Vec<T>& x : results)
    {
        EXPECT_EQ(x, Vec<T>(8, 0));
    }
}

TYPED_TEST(solvers, fixed_point_needs_a_preconditioner)
{
    using T         = typename TypeParam::value_type;
    auto          A = this->make(spd<T>(4));
    FixedPoint<T> solver;
    solver.SetOperator(*A);
    EXPECT_ASSERT_DEATH(solver.Build(), "precond_ != NULL");
}

TYPED_TEST(solvers, chebyshev_converges_with_bounds_of_the_spectrum)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = spd<T>(8);
    const Vec<T> b   = rhs<T>(8);
    auto         A   = this->make(ref);

    // Gershgorin bounds of A and of D^-1 A
    for(bool precond : {false, true})
    {
        SCOPED_TRACE(precond ? "with Jacobi" : "without preconditioner");
        Jacobi<T>    jacobi;
        Chebyshev<T> solver;
        solver.Set(precond ? 1.0 / 3.0 : 1.0, precond ? 5.0 / 3.0 : 10.0);
        solver.Init(0.0, solver_tol<T>(1e-12), 1e8, 1000);
        set_up(solver, method<Chebyshev<T>>("Chebyshev"), *A, precond ? &jacobi : nullptr);

        const Outcome<T> out = solve(solver, *A, b, Vec<T>(8, 0));
        EXPECT_EQ(out.status, 2);
        expect_near(out.x,
                    dense_solve(dense(ref), b),
                    solution_tol(dense(ref), b, Vec<T>(8, 0), solver_tol<T>(1e-12), 2, precond));
    }
}

TYPED_TEST(solvers, chebyshev_iterates_follow_the_chebyshev_recurrence)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = spd<T>(8);
    const Vec<T> D   = dense(ref);
    const Vec<T> b   = rhs<T>(8);
    const Vec<T> x0  = start<T>(8);
    auto         A   = this->make(ref);

    struct Case
    {
        std::string name;
        Vec<T>      x;
        Vec<T>      expected;
    };
    std::vector<Case> cases;
    for(bool precond : {false, true})
    {
        const double lambda_min = precond ? 1.0 / 3.0 : 1.0;
        const double lambda_max = precond ? 5.0 / 3.0 : 10.0;
        for(int k = 1; k <= 4; ++k)
        {
            const std::string name
                = std::string(precond ? "with Jacobi, " : "") + std::to_string(k) + " iterations";
            SCOPED_TRACE(name);
            Jacobi<T>    jacobi;
            Chebyshev<T> solver;
            solver.Set(lambda_min, lambda_max);
            solver.Init(0.0, 0.0, 1e8, k);
            set_up(solver, method<Chebyshev<T>>("Chebyshev"), *A, precond ? &jacobi : nullptr);

            const Outcome<T> out = solve(solver, *A, b, x0);
            EXPECT_EQ(out.iterations, k);
            EXPECT_EQ(out.status, 4);
            cases.push_back({name,
                             out.x,
                             chebyshev_iterate(
                                 D, b, x0, k, preconditioner(D, precond), lambda_min, lambda_max)});
        }
    }

    KNOWN_BUG("the recurrence starts with alpha = 2 / d instead of 1 / d and continues with "
              "alpha = 1 / (d - beta) instead of 1 / (d - beta / alpha)");
    for(const Case& c : cases)
    {
        SCOPED_TRACE(c.name);
        expect_near(c.x, c.expected, loose_tol<T>(1e-10));
    }
}

namespace
{
    template <typename T>
    using Direct = rocalution::DirectLinearSolver<Matrix<T>, Vector<T>, T>;

    template <typename T>
    struct DirectCase
    {
        std::string                                 name;
        std::function<std::unique_ptr<Direct<T>>()> make;
        unsigned int                                format;
    };

    template <typename S, typename T = typename value_type_of<S>::type>
    DirectCase<T> direct(const std::string& name, unsigned int format)
    {
        return {name + " " + rocalution::_matrix_format_names[format],
                [] { return std::make_unique<S>(); },
                format};
    }

    template <typename T>
    Vec<T> direct_solve(Direct<T>& solver, const Matrix<T>& A, const Vec<T>& b)
    {
        auto rhs_vector = make_local_vector(A, b);
        auto x          = make_local_vector(A, Vec<T>(b.size(), 0));
        quietly([&] { solver.Solve(*rhs_vector, x.get()); });
        return values_of(*x);
    }
}

TYPED_TEST(solvers, direct_solvers_solve_exactly)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = nonsymmetric<T>(8);
    const Vec<T> b   = rhs<T>(8);
    const Vec<T> x   = dense_solve(dense(ref), b);

    for(const DirectCase<T>& c : {direct<LU<T>>("LU", rocalution::CSR),
                                  direct<QR<T>>("QR", rocalution::CSR),
                                  direct<QR<T>>("QR", rocalution::DENSE),
                                  direct<Inversion<T>>("Inversion", rocalution::CSR),
                                  direct<Inversion<T>>("Inversion", rocalution::DENSE)})
    {
        SCOPED_TRACE(c.name);
        auto A      = make_local_matrix(this->backend(), ref, Format{c.format, 1});
        auto solver = c.make();
        solver->Verbose(0);
        solver->SetOperator(*A);
        quietly([&] { solver->Build(); });
        expect_near(direct_solve(*solver, *A, b), x, tol<T>());
    }

    KNOWN_BUG("LU::Build calls LUAnalyse, which ends the program for a DENSE matrix");
    SCOPED_TRACE("LU DENSE");
    expect_completes([&] {
        auto  A = make_local_matrix(this->backend(), ref, Format{rocalution::DENSE, 1});
        LU<T> solver;
        solver.Verbose(0);
        solver.SetOperator(*A);
        solver.Build();
        return near(direct_solve(solver, *A, b), x, tol<T>());
    });
}

TYPED_TEST(solvers, lu_solves_with_the_iterative_triangular_solves)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = nonsymmetric<T>(8);
    const Vec<T> b   = rhs<T>(8);
    auto         A   = this->make(ref);

    for(bool use_tolerance : {false, true})
    {
        SCOPED_TRACE(use_tolerance ? "with tolerance" : "without tolerance");
        rocalution::SolverDescr descr;
        descr.SetTriSolverAlg(rocalution::TriSolverAlg_Iterative);
        descr.SetIterativeSolverMaxIteration(30);
        descr.SetIterativeSolverTolerance(solver_tol<T>(1e-15));
        if(use_tolerance)
        {
            descr.EnableIterativeSolverTolerance();
        }
        else
        {
            descr.DisableIterativeSolverTolerance();
        }

        LU<T> solver;
        solver.Verbose(0);
        solver.SetSolverDescriptor(descr);
        solver.SetOperator(*A);
        quietly([&] { solver.Build(); });
        expect_near(direct_solve(solver, *A, b), dense_solve(dense(ref), b), tol<T>());
    }
}

TYPED_TEST(solvers, direct_solvers_refactorize_and_move_between_backends)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = nonsymmetric<T>(8);
    const Vec<T> b   = rhs<T>(8);
    const Csr<T> scaled_ref
        = transformed(ref, [](int i, int j, T v) { return i == j ? 2 * v : v; });

    for(const DirectCase<T>& c : {direct<LU<T>>("LU", rocalution::CSR),
                                  direct<QR<T>>("QR", rocalution::CSR),
                                  direct<Inversion<T>>("Inversion", rocalution::CSR)})
    {
        SCOPED_TRACE(c.name);
        auto A      = this->make(ref);
        auto solver = c.make();
        solver->Verbose(0);
        solver->SetOperator(*A);
        quietly([&] { solver->Build(); });

        A->ScaleDiagonal(2.0);
        quietly([&] { solver->ReBuildNumeric(); });
        expect_near(direct_solve(*solver, *A, b), dense_solve(dense(scaled_ref), b), tol<T>());
        quietly([&] { solver->Build(); });
        expect_near(direct_solve(*solver, *A, b), dense_solve(dense(scaled_ref), b), tol<T>());

        if(available(Backend::Accelerator))
        {
            const Backend other
                = this->backend() == Backend::Host ? Backend::Accelerator : Backend::Host;
            for(Backend to : {other, this->backend()})
            {
                SCOPED_TRACE("moved to the " + backend_name(to));
                if(to == Backend::Host)
                {
                    A->MoveToHost();
                    quietly([&] { solver->MoveToHost(); });
                }
                else
                {
                    A->MoveToAccelerator();
                    quietly([&] { solver->MoveToAccelerator(); });
                }
                expect_near(
                    direct_solve(*solver, *A, b), dense_solve(dense(scaled_ref), b), tol<T>());
            }
        }
    }
}

TYPED_TEST(solvers, lu_moved_to_the_backend_it_is_on_still_solves)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = nonsymmetric<T>(8);
    const Vec<T> b   = rhs<T>(8);
    const Vec<T> x   = dense_solve(dense(ref), b);

    for(bool iterative : {false, true})
    {
        SCOPED_TRACE(iterative ? "iterative triangular solves" : "default triangular solves");
        auto  A = this->make(ref);
        LU<T> solver;
        solver.Verbose(0);
        if(iterative)
        {
            rocalution::SolverDescr descr;
            descr.SetTriSolverAlg(rocalution::TriSolverAlg_Iterative);
            descr.SetIterativeSolverMaxIteration(30);
            descr.DisableIterativeSolverTolerance();
            solver.SetSolverDescriptor(descr);
        }
        solver.SetOperator(*A);
        solver.Build();
        if(this->backend() == Backend::Host)
        {
            solver.MoveToHost();
        }
        else
        {
            solver.MoveToAccelerator();
        }
        expect_near(direct_solve(solver, *A, b), x, tol<T>());
    }
}

TYPED_TEST(solvers, direct_solvers_cleared_and_moved_before_the_build_solve_exactly)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = nonsymmetric<T>(8);
    const Vec<T> b   = rhs<T>(8);

    for(const DirectCase<T>& c : {direct<LU<T>>("LU", rocalution::CSR),
                                  direct<QR<T>>("QR", rocalution::CSR),
                                  direct<Inversion<T>>("Inversion", rocalution::CSR)})
    {
        SCOPED_TRACE(c.name);
        auto A      = this->make(ref);
        auto solver = c.make();
        solver->Verbose(0);
        solver->Clear();
        solver->MoveToAccelerator();
        solver->MoveToHost();
        solver->SetOperator(*A);
        quietly([&] { solver->Build(); });
        expect_near(direct_solve(*solver, *A, b), dense_solve(dense(ref), b), tol<T>());
    }
}

// MixedPrecisionDC keeps the operator and the vectors of the outer solver on the host and
// always moves the inner solver to the accelerator when there is one, so it has no
// backend parameter
namespace
{
    using MixedPrecision = rocalution::MixedPrecisionDC<Matrix<double>,
                                                        Vector<double>,
                                                        double,
                                                        LocalMatrix<float>,
                                                        LocalVector<float>,
                                                        float>;

    // MixedPrecisionDC reports its steps regardless of its verbosity
    Outcome<double> quiet_solve(Iterative<double>&    solver,
                                const Matrix<double>& A,
                                const Vec<double>&    b,
                                const Vec<double>&    x0)
    {
        return quietly([&] { return solve(solver, A, b, x0); });
    }
}

TEST(solvers_mixed_precision, defect_correction_reaches_double_accuracy_with_a_float_solver)
{
    const Csr<double> ref = nonsymmetric<double>(16);
    const Vec<double> b   = rhs(16);
    auto              A   = make_local_matrix(Backend::Host, ref);

    rocalution::Jacobi<LocalMatrix<float>, LocalVector<float>, float> jacobi;
    rocalution::GMRES<LocalMatrix<float>, LocalVector<float>, float>  inner;
    inner.Verbose(0);
    inner.Init(0.0, 1e-4, 1e8, 100);
    inner.SetPreconditioner(jacobi);

    MixedPrecision solver;
    solver.Verbose(0);
    solver.Init(0.0, 1e-13, 1e8, 20);
    solver.SetOperator(*A);
    solver.Set(inner);
    quietly([&] { solver.ReBuildNumeric(); });

    const Outcome<double> out = quiet_solve(solver, *A, b, Vec<double>(16, 0.0));
    EXPECT_EQ(out.status, 2);
    expect_near(out.x, dense_solve(dense(ref), b), 1e-11);

    // The moves do nothing: the inner solver stays on the accelerator
    quietly([&] {
        solver.MoveToAccelerator();
        solver.MoveToHost();
    });
    quietly([&] { solver.ReBuildNumeric(); });
    expect_near(
        quiet_solve(solver, *A, b, Vec<double>(16, 0.0)).x, dense_solve(dense(ref), b), 1e-11);

    const Outcome<double> zero
        = quiet_solve(solver, *A, Vec<double>(16, 0.0), Vec<double>(16, 0.0));
    EXPECT_EQ(zero.x, Vec<double>(16, 0.0));
    EXPECT_EQ(zero.iterations, 0);
    EXPECT_EQ(zero.status, 1);
}

TEST(solvers_mixed_precision, build_of_a_built_solver_builds_it_again)
{
    const Csr<double> ref = nonsymmetric<double>(16);
    const Vec<double> b   = rhs(16);

    KNOWN_BUG("Build of a built MixedPrecisionDC clears its inner solver, then asserts it");

    expect_in_child([&] {
        auto A = make_local_matrix(Backend::Host, ref);

        rocalution::LU<LocalMatrix<float>, LocalVector<float>, float> inner;
        inner.Verbose(0);

        MixedPrecision solver;
        solver.Verbose(0);
        solver.Init(0.0, 1e-13, 1e8, 20);
        solver.SetOperator(*A);
        solver.Set(inner);
        quietly([&] { solver.Build(); });
        quietly([&] { solver.Build(); });

        const Outcome<double> out = quiet_solve(solver, *A, b, Vec<double>(16, 0.0));
        EXPECT_EQ(out.status, 2);
        expect_near(out.x, dense_solve(dense(ref), b), 1e-11);
    });
}

TEST(solvers_mixed_precision, rebuild_numeric_takes_new_matrix_values)
{
    const Csr<double> ref = nonsymmetric<double>(16);
    const Vec<double> b   = rhs(16);
    auto              A   = make_local_matrix(Backend::Host, ref);

    rocalution::LU<LocalMatrix<float>, LocalVector<float>, float> inner;
    inner.Verbose(0);

    MixedPrecision solver;
    solver.Verbose(0);
    solver.Init(0.0, 1e-13, 1e8, 20);
    solver.SetOperator(*A);
    solver.Set(inner);
    quietly([&] { solver.Build(); });
    quiet_solve(solver, *A, b, Vec<double>(16, 0.0));

    A->ScaleDiagonal(2.0);
    quietly([&] { solver.ReBuildNumeric(); });
    const Outcome<double> out = quiet_solve(solver, *A, b, Vec<double>(16, 0.0));

    KNOWN_BUG("MixedPrecisionDC::ReBuildNumeric keeps the float copy of the old matrix");
    EXPECT_EQ(out.status, 2);
    expect_near(out.x,
                dense_solve(dense(transformed(
                                ref, [](int i, int j, double v) { return i == j ? 2.0 * v : v; })),
                            b),
                1e-11);
}

TEST(solvers_mixed_precision, one_iteration_is_one_defect_correction)
{
    const Csr<double> ref = nonsymmetric<double>(16);
    const Vec<double> b   = rhs(16);
    const Vec<double> x   = dense_solve(dense(ref), b);
    auto              A   = make_local_matrix(Backend::Host, ref);

    rocalution::LU<LocalMatrix<float>, LocalVector<float>, float> inner;
    inner.Verbose(0);

    MixedPrecision solver;
    solver.Verbose(1);
    solver.Init(0.0, 0.0, 1e8, 1);
    solver.SetOperator(*A);
    solver.Set(inner);
    quietly([&] { solver.Build(); });

    testing::internal::CaptureStdout();
    const Outcome<double> out    = solve(solver, *A, b, Vec<double>(16, 0.0));
    const std::string     output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("MixedPrecisionDC [64bit-32bit] solver starts"), std::string::npos);
    EXPECT_NE(output.find("LU solver"), std::string::npos);
    EXPECT_EQ(out.status, 4);

    KNOWN_BUG("the residual check before the first correction is counted as an iteration");
    EXPECT_EQ(out.iterations, 1);
    expect_near(out.x, x, 1e-5);
}

TEST(solvers_mixed_precision, clear_and_moves_before_the_build_keep_the_inner_solver)
{
    const Csr<double> ref = nonsymmetric<double>(16);
    const Vec<double> b   = rhs(16);
    auto              A   = make_local_matrix(Backend::Host, ref);

    rocalution::LU<LocalMatrix<float>, LocalVector<float>, float> inner;
    inner.Verbose(0);

    MixedPrecision solver;
    solver.Verbose(0);
    solver.Init(0.0, 1e-13, 1e8, 20);
    solver.SetOperator(*A);
    solver.Set(inner);
    solver.Clear();
    solver.MoveToAccelerator();
    solver.MoveToHost();
    quietly([&] { solver.Build(); });

    const Outcome<double> out = quiet_solve(solver, *A, b, Vec<double>(16, 0.0));
    EXPECT_EQ(out.status, 2);
    expect_near(out.x, dense_solve(dense(ref), b), 1e-11);
}

TEST(solver_descriptor, copies_and_assignments_take_every_setting)
{
    rocalution::SolverDescr descr;
    descr.SetTriSolverAlg(rocalution::TriSolverAlg_Iterative);
    descr.SetIterativeSolverMaxIteration(7);
    descr.SetIterativeSolverTolerance(1e-3);
    descr.EnableIterativeSolverTolerance();

    const rocalution::SolverDescr copy(descr);
    rocalution::SolverDescr       assigned;
    assigned                              = descr;
    const rocalution::SolverDescr& itself = assigned;
    assigned                              = itself;

    for(const rocalution::SolverDescr* d : {&copy, &itself})
    {
        EXPECT_EQ(d->GetTriSolverAlg(), rocalution::TriSolverAlg_Iterative);
        EXPECT_EQ(d->GetIterativeSolverMaxIteration(), 7);
        EXPECT_EQ(d->GetIterativeSolverTolerance(), 1e-3);
        EXPECT_TRUE(d->GetIterativeSolverUseTolerance());
    }
}

TEST(iteration_control, non_finite_values_are_reported_and_stop_the_iteration)
{
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    rocalution::IterationControl control;
    control.Verbose(0);
    testing::internal::CaptureStdout();
    control.Init(inf, nan, inf, 2, 5);
    std::string out = testing::internal::GetCapturedStdout();
    EXPECT_NE(out.find("Abs tol = inf"), std::string::npos);
    EXPECT_NE(out.find("Rel tol = nan"), std::string::npos);
    EXPECT_NE(out.find("Div tol = inf"), std::string::npos);
    EXPECT_EQ(control.GetMinimumIterations(), 2);
    EXPECT_EQ(control.GetMaximumIterations(), 5);

    for(double res : {inf, nan})
    {
        SCOPED_TRACE("residual " + std::to_string(res));
        testing::internal::CaptureStdout();
        EXPECT_FALSE(control.InitResidual(res));
        EXPECT_TRUE(control.CheckResidualNoCount(res));
        out = testing::internal::GetCapturedStdout();
        EXPECT_NE(out.find("Residual = "), std::string::npos);
        EXPECT_EQ(control.GetIterationCount(), 0);
    }
}

// LocalStencil exists on the host only
namespace
{
    template <typename P>
    class solvers_stencil : public TypedBackendTest<P>
    {
    };
}

TYPED_TEST_SUITE(solvers_stencil, HostOnly<RealTypes>, TypedName);

TYPED_TEST(solvers_stencil, krylov_solvers_and_chebyshev_solve_the_laplace_stencil)
{
    using T             = typename TypeParam::value_type;
    using Stencil       = rocalution::LocalStencil<T>;
    using StencilSolver = rocalution::IterativeLinearSolver<Stencil, Vector<T>, T>;

    const int s = 3;
    const int n = s * s;
    Vec<T>    D(n * n, 0);
    for(int i = 0; i < s; ++i)
    {
        for(int j = 0; j < s; ++j)
        {
            const int row    = i * s + j;
            D[row * n + row] = 4;
            if(i > 0)
            {
                D[row * n + row - s] = -1;
            }
            if(i + 1 < s)
            {
                D[row * n + row + s] = -1;
            }
            if(j > 0)
            {
                D[row * n + row - 1] = -1;
            }
            if(j + 1 < s)
            {
                D[row * n + row + 1] = -1;
            }
        }
    }
    const Vec<T> b = {1, 3, -2, 5, 0, -1, 4, 2, -3};
    const Vec<T> x = dense_solve(D, b);

    Stencil L(rocalution::Laplace2D);
    L.SetGrid(s);

    // The eigenvalues 4 - 2 cos(i pi / 4) - 2 cos(j pi / 4) take 5 distinct values
    const double c = std::cos(M_PI / (s + 1));
    struct Case
    {
        std::string                                     name;
        std::function<std::unique_ptr<StencilSolver>()> make;
        int                                             iterations;
    };
    const std::vector<Case> cases = {
        {"CG", [] { return std::make_unique<rocalution::CG<Stencil, Vector<T>, T>>(); }, 5},
        {"CR", [] { return std::make_unique<rocalution::CR<Stencil, Vector<T>, T>>(); }, 5},
        {"GMRES", [] { return std::make_unique<rocalution::GMRES<Stencil, Vector<T>, T>>(); }, 5},
        {"FGMRES", [] { return std::make_unique<rocalution::FGMRES<Stencil, Vector<T>, T>>(); }, 5},
        {"BiCGStab",
         [] { return std::make_unique<rocalution::BiCGStab<Stencil, Vector<T>, T>>(); },
         -1},
        {"IDR(2)", [] { return std::make_unique<rocalution::IDR<Stencil, Vector<T>, T>>(); }, -1},
        {"Chebyshev",
         [c] {
             auto solver = std::make_unique<rocalution::Chebyshev<Stencil, Vector<T>, T>>();
             solver->Set(4.0 - 4.0 * c, 4.0 + 4.0 * c);
             return solver;
         },
         -1}};

    for(const Case& k : cases)
    {
        SCOPED_TRACE(k.name);
        auto solver = k.make();
        solver->Verbose(0);
        solver->Init(0.0, solver_tol<T>(1e-12), 1e8, 1000);
        solver->SetOperator(L);
        if(auto idr_solver = dynamic_cast<rocalution::IDR<Stencil, Vector<T>, T>*>(solver.get()))
        {
            idr_solver->SetShadowSpace(2);
            idr_solver->SetRandomSeed(12345ULL);
        }
        solver->Build();

        auto rhs_vector = make_local_vector(L, b);
        auto result     = make_local_vector(L, Vec<T>(n, 0));
        solver->Solve(*rhs_vector, result.get());
        expect_near(values_of(*result), x, solution_tol(D, b, Vec<T>(n, 0), solver_tol<T>(1e-12)));
        if(k.iterations > 0)
        {
            EXPECT_EQ(solver->GetIterationCount(), k.iterations);
        }
    }
}

TYPED_TEST(solvers_stencil, clear_and_set_grid_resize_the_laplace_stencil)
{
    using T = typename TypeParam::value_type;
    rocalution::LocalStencil<T> L(rocalution::Laplace2D);
    L.SetGrid(3);
    EXPECT_EQ(L.GetNDim(), 2);
    EXPECT_EQ(L.GetM(), 9);
    EXPECT_EQ(L.GetN(), 9);
    EXPECT_NE(logged([&] { L.Info(); }).find("size=3"), std::string::npos);

    L.Clear();
    EXPECT_EQ(L.GetM(), 0);
    EXPECT_EQ(L.GetN(), 0);
    EXPECT_NE(logged([&] { L.Info(); }).find("size=0"), std::string::npos);

    // 4 x_ij minus the neighbours on the 2 x 2 grid, each node has two
    L.SetGrid(2);
    EXPECT_EQ(L.GetM(), 4);
    auto x = make_local_vector(L, Vec<T>{1, 2, 3, 4});
    auto y = make_local_vector(L, Vec<T>(4, 0));
    L.Apply(*x, y.get());
    EXPECT_EQ(values_of(*y), (Vec<T>{4 - 2 - 3, 8 - 1 - 4, 12 - 1 - 4, 16 - 2 - 3}));
}
