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

// The preconditioner building blocks of the backend CSR matrices that
// unit_test_matrix_csr.cpp leaves out, checked against dense references: FSAI,
// SPAI, the iterative LL solve, ILUT with dropping and CMK of irregular graphs,
// and the remaining branches of conversions, permutations and checks.

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "unit_test_common.hpp"
#include "unit_test_reference.hpp"
#include "unit_test_types.hpp"

using namespace rocalution_unit;

namespace
{
    const Format csr_format{rocalution::CSR, 1};

    // FSAI and SPAI have no entry in implements(): only the host CSR has them
    bool approximate_inverse_native(Backend b)
    {
        return b == Backend::Host;
    }

#define REQUIRE_APPROXIMATE_INVERSE(backend)                                        \
    if(!approximate_inverse_native(backend))                                        \
    {                                                                               \
        GTEST_SKIP() << "FSAI and SPAI are only implemented for Host CSR, see the " \
                        "fallback test";                                            \
    }

    // Nonsymmetric with a dominant diagonal: tridiagonal, plus entries three and
    // five columns to the right of the diagonal, wrapped around. The magnitudes
    // within a row are distinct.
    template <typename T>
    Csr<T> nonsymmetric(int n)
    {
        std::vector<T> D(static_cast<size_t>(n) * n, static_cast<T>(0));
        for(int i = 0; i < n; ++i)
        {
            for(int j : {i - 1, i + 1, (i + 3) % n, (i + 5) % n})
            {
                if(j >= 0 && j < n && j != i)
                {
                    D[i * n + j] = ((3 * i + 5 * j) % 13 + 1) * ((i + j) % 2 ? T(-1) : T(1));
                }
            }
            D[i * n + i] = static_cast<T>(100 + i);
        }
        return from_dense(n, n, D);
    }

    // N + N^T for N = nonsymmetric(n), positive definite by its dominant diagonal
    template <typename T>
    Csr<T> spd(int n)
    {
        const std::vector<T> D = dense(nonsymmetric<T>(n));
        std::vector<T>       S(D.size());
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                S[i * n + j] = D[i * n + j] + D[j * n + i];
            }
        }
        return from_dense(n, n, S);
    }

    // Ones on the pattern
    template <typename T>
    Csr<T> pattern_of(int n, const std::function<bool(int, int)>& in_pattern)
    {
        Csr<T> P(n);
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                if(in_pattern(i, j))
                {
                    P.add(j, static_cast<T>(1));
                }
            }
            P.end_row();
        }
        return P;
    }

    // Pattern of A^2, without cancellation
    template <typename T>
    Csr<T> squared_pattern(const Csr<T>& A)
    {
        const int            n = A.n;
        const std::vector<T> D = dense(A);
        return pattern_of<T>(n, [&](int i, int j) {
            for(int k = 0; k < n; ++k)
            {
                if(D[i * n + k] != static_cast<T>(0) && D[k * n + j] != static_cast<T>(0))
                {
                    return true;
                }
            }
            return false;
        });
    }

    // The solution of the dense system M x = b, by Gaussian elimination with
    // partial pivoting
    template <typename T>
    std::vector<T> solve(std::vector<T> M, std::vector<T> b)
    {
        const int k = static_cast<int>(b.size());
        for(int c = 0; c < k; ++c)
        {
            int p = c;
            for(int r = c + 1; r < k; ++r)
            {
                if(std::abs(M[r * k + c]) > std::abs(M[p * k + c]))
                {
                    p = r;
                }
            }
            for(int j = 0; j < k; ++j)
            {
                std::swap(M[c * k + j], M[p * k + j]);
            }
            std::swap(b[c], b[p]);

            for(int r = c + 1; r < k; ++r)
            {
                const T f = M[r * k + c] / M[c * k + c];
                for(int j = c; j < k; ++j)
                {
                    M[r * k + j] -= f * M[c * k + j];
                }
                b[r] -= f * b[c];
            }
        }
        for(int c = k - 1; c >= 0; --c)
        {
            for(int j = c + 1; j < k; ++j)
            {
                b[c] -= M[c * k + j] * b[j];
            }
            b[c] /= M[c * k + c];
        }
        return b;
    }

    // Row i of G is A(J, J)^-1 e_last on the columns J <= i of row i of the
    // pattern, scaled by 1 / sqrt(|e_last^T A(J, J)^-1 e_last|), so that
    // diag(G A G^T) = 1
    template <typename T>
    Csr<T> fsai(const Csr<T>& A, const Csr<T>& pattern)
    {
        const int            n = A.n;
        const std::vector<T> D = dense(A);

        Csr<T> G(n);
        for(int i = 0; i < n; ++i)
        {
            std::vector<int> J;
            for(PtrType k = pattern.row_offset[i]; k < pattern.row_offset[i + 1]; ++k)
            {
                if(pattern.col[k] <= i)
                {
                    J.push_back(pattern.col[k]);
                }
            }
            std::sort(J.begin(), J.end());

            const int      k = static_cast<int>(J.size());
            std::vector<T> M(static_cast<size_t>(k) * k);
            for(int r = 0; r < k; ++r)
            {
                for(int c = 0; c < k; ++c)
                {
                    M[r * k + c] = D[J[r] * n + J[c]];
                }
            }
            std::vector<T> e(k, static_cast<T>(0));
            e[k - 1] = static_cast<T>(1);

            const std::vector<T> g = solve(M, e);
            const T              s = static_cast<T>(1) / std::sqrt(std::abs(g[k - 1]));
            for(int c = 0; c < k; ++c)
            {
                G.add(J[c], g[c] * s);
            }
            G.end_row();
        }
        return G;
    }

    // Column i of M minimizes ||A m - e_i|| over the m on the pattern of column i
    // of A, here through the normal equations
    template <typename T>
    Csr<T> spai(const Csr<T>& A)
    {
        const int            n = A.n;
        const std::vector<T> D = dense(A);

        std::vector<bool> P(D.size());
        for(int i = 0; i < n; ++i)
        {
            for(PtrType k = A.row_offset[i]; k < A.row_offset[i + 1]; ++k)
            {
                P[i * n + A.col[k]] = true;
            }
        }

        std::vector<T> M(D.size(), static_cast<T>(0));
        for(int i = 0; i < n; ++i)
        {
            std::vector<int> J;
            for(int j = 0; j < n; ++j)
            {
                if(P[j * n + i])
                {
                    J.push_back(j);
                }
            }

            const int      k = static_cast<int>(J.size());
            std::vector<T> N(static_cast<size_t>(k) * k, static_cast<T>(0));
            std::vector<T> r(k);
            for(int a = 0; a < k; ++a)
            {
                for(int b = 0; b < k; ++b)
                {
                    for(int row = 0; row < n; ++row)
                    {
                        N[a * k + b] += D[row * n + J[a]] * D[row * n + J[b]];
                    }
                }
                r[a] = D[i * n + J[a]];
            }

            const std::vector<T> m = solve(N, r);
            for(int a = 0; a < k; ++a)
            {
                M[J[a] * n + i] = m[a];
            }
        }
        return transformed(A, [&](int i, int j, T) { return M[i * n + j]; });
    }

    // L L^T x = b with the lower triangle L of the dense n x n matrix D
    template <typename T>
    std::vector<T> ll_solve(const std::vector<T>& D, int n, const std::vector<T>& b)
    {
        std::vector<T> Lt(D.size(), static_cast<T>(0));
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j <= i; ++j)
            {
                Lt[j * n + i] = D[i * n + j];
            }
        }
        return upper_solve(Lt, n, lower_solve(D, n, b, false), false);
    }

    // Keeps the p entries of largest magnitude. A tie at the cut would make the
    // reference ambiguous, so the inputs avoid it.
    template <typename T>
    void keep_largest(std::vector<std::pair<int, T>>& v, int p)
    {
        if(static_cast<int>(v.size()) <= p)
        {
            return;
        }
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
            return std::abs(a.second) > std::abs(b.second);
        });
        if(p > 0)
        {
            EXPECT_NE(std::abs(v[p - 1].second), std::abs(v[p].second)) << "tie at the cut";
        }
        v.resize(p);
    }

    // ILUT as computed row by row, in rocALUTION's variant of Saad's ILUT(t, p): a
    // multiplier l_ik = w_k / u_kk is kept if |l_ik| > |t|, an absolute threshold where
    // Saad drops it against t times the norm of the row of A, and then updates the row
    // with row k of U. An entry of U is kept if it exceeds |t| times the mean magnitude
    // of the row of A.
    // Then the maxrow largest multipliers and the maxrow - 1 largest entries of U
    // stay. A missing or zero diagonal becomes (1e-4 + |t|) times that mean. The
    // mean and the thresholds are in double for every T.
    template <typename T>
    Csr<T> ilut(const Csr<T>& A, double t, int maxrow)
    {
        const int    n   = A.n;
        const double tol = std::abs(t);

        std::vector<std::map<int, T>> U(n);
        Csr<T>                        F(n);
        for(int i = 0; i < n; ++i)
        {
            std::map<int, T> w;
            double           norm = 0;
            for(PtrType k = A.row_offset[i]; k < A.row_offset[i + 1]; ++k)
            {
                w[A.col[k]] = A.val[k];
                norm += std::abs(A.val[k]);
            }
            norm /= static_cast<double>(A.row_offset[i + 1] - A.row_offset[i]);

            std::vector<std::pair<int, T>> lower;
            for(auto it = w.begin(); it != w.end() && it->first < i; it = w.upper_bound(it->first))
            {
                const int k      = it->first;
                const T   factor = it->second / U[k].at(k);
                if(std::abs(factor) <= tol)
                {
                    continue;
                }
                lower.emplace_back(k, factor);
                for(const auto& u : U[k])
                {
                    if(u.first > k)
                    {
                        w[u.first] -= factor * u.second;
                    }
                }
            }

            if(w.count(i) == 0 || w[i] == static_cast<T>(0))
            {
                w[i] = static_cast<T>((static_cast<double>(1e-4f) + tol) * norm);
            }

            std::vector<std::pair<int, T>> upper;
            for(const auto& e : w)
            {
                if(e.first > i && std::abs(e.second) > tol * norm)
                {
                    upper.push_back(e);
                }
            }

            keep_largest(lower, maxrow);
            keep_largest(upper, maxrow - 1);

            U[i][i] = w[i];
            U[i].insert(upper.begin(), upper.end());

            std::map<int, T> row(lower.begin(), lower.end());
            row.insert(U[i].begin(), U[i].end());
            for(const auto& e : row)
            {
                F.add(e.first, e.second);
            }
            F.end_row();
        }
        return F;
    }

    // Rows with fill-in and distinct magnitudes. Row 0 has no diagonal, and no
    // other row has an entry in column 0, so that its replacement is never a pivot.
    // The last row has no diagonal either, until the elimination fills it in.
    template <typename T>
    Csr<T> ilut_input()
    {
        const int      n = 8;
        std::vector<T> D(n * n, static_cast<T>(0));
        for(int i = 0; i < n; ++i)
        {
            for(int j = 1; j < n; ++j)
            {
                const int  d       = (j - i + n) % n;
                const bool coupled = std::abs(i - j) == 1 || d == 3 || d == 5;
                if(i == j && i < n - 1)
                {
                    D[i * n + j] = static_cast<T>(20 + i);
                }
                else if(coupled)
                {
                    D[i * n + j] = ((3 * i + 5 * j) % 13 + 1) * ((i + j) % 2 ? T(-0.5) : T(0.5));
                }
            }
        }
        return from_dense(n, n, D);
    }

    // Symmetric matrix of the graph: -1 per edge, the degree plus one on the
    // diagonal
    template <typename T>
    Csr<T> graph(int n, const std::vector<std::pair<int, int>>& edges)
    {
        std::vector<T> D(static_cast<size_t>(n) * n, static_cast<T>(0));
        for(int i = 0; i < n; ++i)
        {
            D[i * n + i] = 1;
        }
        for(const auto& e : edges)
        {
            D[e.first * n + e.second] = -1;
            D[e.second * n + e.first] = -1;
            D[e.first * n + e.first] += 1;
            D[e.second * n + e.second] += 1;
        }
        return from_dense(n, n, D);
    }

    bool is_permutation(std::vector<int> perm)
    {
        std::sort(perm.begin(), perm.end());
        for(size_t i = 0; i < perm.size(); ++i)
        {
            if(perm[i] != static_cast<int>(i))
            {
                return false;
            }
        }
        return true;
    }

    template <typename T>
    std::vector<T> inverse_diagonal(const Csr<T>& A)
    {
        const std::vector<T> D = dense(A);
        std::vector<T>       inv(A.n);
        for(int i = 0; i < A.n; ++i)
        {
            inv[i] = static_cast<T>(1) / D[i * A.n + i];
        }
        return inv;
    }

    bool lower_part(int i, int j, double)
    {
        return j <= i;
    }

    template <typename T>
    T diagonal_entry(const Csr<T>& A, int i)
    {
        for(PtrType k = A.row_offset[i]; k < A.row_offset[i + 1]; ++k)
        {
            if(A.col[k] == i)
            {
                return A.val[k];
            }
        }
        ADD_FAILURE() << "row " << i << " has no diagonal";
        return static_cast<T>(1);
    }

    // b solved with the lower triangle of the sparse F, with a unit diagonal if unit
    template <typename T>
    std::vector<T> sparse_lower_solve(const Csr<T>& F, std::vector<T> b, bool unit)
    {
        for(int i = 0; i < F.n; ++i)
        {
            for(PtrType k = F.row_offset[i]; k < F.row_offset[i + 1]; ++k)
            {
                if(F.col[k] < i)
                {
                    b[i] -= F.val[k] * b[F.col[k]];
                }
            }
            if(!unit)
            {
                b[i] /= diagonal_entry(F, i);
            }
        }
        return b;
    }

    // b solved with the upper triangle of the sparse F
    template <typename T>
    std::vector<T> sparse_upper_solve(const Csr<T>& F, std::vector<T> b)
    {
        for(int i = F.n - 1; i >= 0; --i)
        {
            for(PtrType k = F.row_offset[i]; k < F.row_offset[i + 1]; ++k)
            {
                if(F.col[k] > i)
                {
                    b[i] -= F.val[k] * b[F.col[k]];
                }
            }
            b[i] /= diagonal_entry(F, i);
        }
        return b;
    }

    // b solved with the transpose of the lower triangle of the sparse F
    template <typename T>
    std::vector<T> sparse_lower_transposed_solve(const Csr<T>& F, std::vector<T> b)
    {
        for(int i = F.n - 1; i >= 0; --i)
        {
            b[i] /= diagonal_entry(F, i);
            for(PtrType k = F.row_offset[i]; k < F.row_offset[i + 1]; ++k)
            {
                if(F.col[k] < i)
                {
                    b[F.col[k]] -= F.val[k] * b[i];
                }
            }
        }
        return b;
    }

    // Tridiagonal with the off-diagonals -1 and lower, and a dominant diagonal
    template <typename T>
    Csr<T> large_tridiagonal(int n, T lower)
    {
        Csr<T> A(n);
        for(int i = 0; i < n; ++i)
        {
            if(i > 0)
            {
                A.add(i - 1, lower);
            }
            A.add(i, static_cast<T>(4 + (i % 3) * 0.5));
            if(i + 1 < n)
            {
                A.add(i + 1, static_cast<T>(-1));
            }
            A.end_row();
        }
        return A;
    }
}

template <typename P>
class backend_csr_preconditioners : public TypedBackendTest<P>
{
protected:
    using T = typename P::value_type;

    static Backend backend()
    {
        return P::backend;
    }

    std::unique_ptr<BaseMatrix<T>> make(const Csr<T>& ref)
    {
        return make_matrix(backend(), csr_format, *ref.host_csr());
    }

    std::unique_ptr<BaseMatrix<T>> empty()
    {
        return empty_matrix<T>(backend(), csr_format);
    }
};

TYPED_TEST_SUITE(backend_csr_preconditioners, WithBackends<RealTypes>, TypedName);

// Without a pattern, FSAI works on the lower triangle of A, or of the pattern of
// A^power
TYPED_TEST(backend_csr_preconditioners, fsai_on_the_pattern_of_a_power_of_the_matrix)
{
    using T = typename TypeParam::value_type;
    REQUIRE_APPROXIMATE_INVERSE(this->backend());

    const Csr<T> ref = spd<T>(10);
    for(int power : {1, 2})
    {
        SCOPED_TRACE("power " + std::to_string(power));
        const Csr<T> pattern = power == 1 ? ref : squared_pattern(ref);

        auto A = this->make(ref);
        ASSERT_IN_LOOP(A != nullptr);
        ASSERT_IN_LOOP(A->FSAI(power, nullptr));
        expect_near(csr_of(*A), fsai(ref, pattern), tol<T>());
    }
}

// A given pattern replaces the pattern of A; only its lower triangle is used
TYPED_TEST(backend_csr_preconditioners, fsai_on_a_given_pattern)
{
    using T = typename TypeParam::value_type;
    REQUIRE_APPROXIMATE_INVERSE(this->backend());

    const Csr<T> ref = spd<T>(10);
    const Csr<T> tridiagonal
        = pattern_of<T>(ref.n, [](int i, int j) { return std::abs(i - j) <= 1; });

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);
    auto P = this->make(tridiagonal);
    ASSERT_NE(P, nullptr);
    ASSERT_TRUE(A->FSAI(1, P.get()));
    expect_near(csr_of(*A), fsai(ref, tridiagonal), tol<T>());
}

TYPED_TEST(backend_csr_preconditioners,
           fsai_on_the_diagonal_is_the_inverse_square_root_of_the_diagonal)
{
    using T = typename TypeParam::value_type;
    REQUIRE_APPROXIMATE_INVERSE(this->backend());

    const Csr<T> ref      = spd<T>(10);
    const Csr<T> diagonal = pattern_of<T>(ref.n, [](int i, int j) { return i == j; });

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);
    auto P = this->make(diagonal);
    ASSERT_NE(P, nullptr);
    ASSERT_TRUE(A->FSAI(1, P.get()));

    const Csr<T> G = csr_of(*A);
    ASSERT_EQ(G.row_offset, diagonal.row_offset);
    EXPECT_NEAR(G.val[0], static_cast<T>(1) / std::sqrt(dense(ref)[0]), tol<T>());

    KNOWN_BUG("a row with only the diagonal in the pattern is set to 1 / a_ii only if the "
              "diagonal is the first entry of the row of A");
    expect_near(G, fsai(ref, diagonal), tol<T>());
}

// Column i of the result minimizes ||A m - e_i|| over the pattern of column i of A
TYPED_TEST(backend_csr_preconditioners, spai_solves_the_least_squares_problem_of_each_column)
{
    using T = typename TypeParam::value_type;
    REQUIRE_APPROXIMATE_INVERSE(this->backend());

    const std::vector<std::pair<std::string, Csr<T>>> cases
        = {{"nonsymmetric", nonsymmetric<T>(10)}, {"symmetric", spd<T>(8)}};
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.first);
        auto A = this->make(c.second);
        ASSERT_IN_LOOP(A != nullptr);
        ASSERT_IN_LOOP(A->SPAI());
        expect_near(csr_of(*A), spai(c.second), loose_tol<T>(1e-10));
    }
}

// Without a tolerance, max_iter >= n Jacobi sweeps solve each triangular system,
// from any initial guess. Only the lower triangle of the matrix is read.
TYPED_TEST(backend_csr_preconditioners,
           iterative_ll_solve_without_tolerance_uses_the_lower_triangle)
{
    using T = typename TypeParam::value_type;

    const Csr<T>         ref      = spd<T>(10);
    const int            n        = ref.n;
    const std::vector<T> b        = rhs<T>(n);
    const std::vector<T> expected = ll_solve(dense(ref), n, b);

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);
    auto in  = make_vector(this->backend(), b);
    auto inv = make_vector(this->backend(), inverse_diagonal(ref));

    A->ItLLAnalyse();
    for(T guess : {T(0), T(5)})
    {
        SCOPED_TRACE("initial guess " + std::to_string(guess));

        auto out = make_vector(this->backend(), std::vector<T>(n, guess));
        ASSERT_IN_LOOP(A->ItLLSolve(2 * n, 0.0, false, *in, out.get()));
        expect_near(values_of(*out), expected, loose_tol<T>(1e-10));

        auto out_inv = make_vector(this->backend(), std::vector<T>(n, guess));
        ASSERT_IN_LOOP(A->ItLLSolve(2 * n, 0.0, false, *in, *inv, out_inv.get()));
        expect_near(values_of(*out_inv), expected, loose_tol<T>(1e-10));
    }
    A->ItLLAnalyseClear();

    expect_near(csr_of(*A), ref, 0.0);
}

// With L = [B 0 0; 0 B 0; 0 B 1], the first sweep of the L^T solve leaves the
// residual 1 in row 1 and 2^-10 in row 2, so a tolerance of 2^-9 needs more
// sweeps. The L solve converges in 3 sweeps, which bounds the L^T solve on the host
// (a known bug), but 3 suffice here. B is 2^20, or 2^12 in float, where B + 2^-10 must be
// exact.
TYPED_TEST(backend_csr_preconditioners,
           iterative_ll_solve_stops_once_every_residual_is_below_the_tolerance)
{
    using T = typename TypeParam::value_type;

    const T big = std::ldexp(T(1), std::is_same_v<T, float> ? 12 : 20);
    Csr<T>  L(3);
    L.add(0, big);
    L.end_row();
    L.add(1, big);
    L.end_row();
    L.add(1, big);
    L.add(2, 1.0);
    L.end_row();

    const T              small    = std::ldexp(T(1), -10);
    const std::vector<T> b        = {big, big, big + small};
    const std::vector<T> expected = {T(1) / big, T(1) / big - small, small};
    ASSERT_EQ(ll_solve(dense(L), L.n, b), expected);

    auto A = this->make(L);
    ASSERT_NE(A, nullptr);
    auto in  = make_vector(this->backend(), b);
    auto out = make_vector(this->backend(), std::vector<T>(L.n, 0.0));

    A->ItLLAnalyse();
    ASSERT_TRUE(A->ItLLSolve(10, std::ldexp(1.0, -9), true, *in, out.get()));
    A->ItLLAnalyseClear();

    if(this->backend() == Backend::Host)
    {
        KNOWN_BUG("the stopping test of the transposed solve takes the largest correction of "
                  "the previous rows instead of the largest residual");
    }
    expect_near(values_of(*out), expected, tol<T>());
}

// The iterative solves of a factorized matrix solve with its factors. On the
// accelerator the factorization leaves its buffer to the analysis of the solve.
TYPED_TEST(backend_csr_preconditioners, iterative_solves_after_a_factorization_use_the_factors)
{
    using T = typename TypeParam::value_type;

    const int            n  = 10;
    const std::vector<T> b  = rhs<T>(n);
    auto                 in = make_vector(this->backend(), b);

    {
        SCOPED_TRACE("IC(0), then the LL solve");
        auto A = this->make(spd<T>(n));
        ASSERT_NE(A, nullptr);
        auto inv_diag = make_vector(this->backend(), std::vector<T>(n, 0.0));
        ASSERT_TRUE(A->ICFactorize(inv_diag.get()));
        const std::vector<T> L = dense(filtered(csr_of(*A), lower_part));

        A->ItLLAnalyse();
        auto out = make_vector(this->backend(), std::vector<T>(n, 0.0));
        ASSERT_TRUE(A->ItLLSolve(2 * n, 0.0, false, *in, out.get()));
        A->ItLLAnalyseClear();
        expect_near(values_of(*out), ll_solve(L, n, b), loose_tol<T>(1e-10));
    }

    using Solve = std::function<bool(BaseMatrix<T>&, BaseVector<T>*)>;
    const std::vector<std::tuple<std::string, std::function<void(BaseMatrix<T>&)>, Solve>> solves
        = {
            {"LU",
             [](BaseMatrix<T>& A) { A.ItLUAnalyse(); },
             [&](BaseMatrix<T>& A, BaseVector<T>* out) {
                 bool ok = A.ItLUSolve(2 * n, 0.0, false, *in, out);
                 A.ItLUAnalyseClear();
                 return ok;
             }},
            {"L",
             [](BaseMatrix<T>& A) { A.ItLAnalyse(true); },
             [&](BaseMatrix<T>& A, BaseVector<T>* out) {
                 bool ok = A.ItLSolve(2 * n, 0.0, false, *in, out);
                 A.ItLAnalyseClear();
                 return ok;
             }},
            {"U",
             [](BaseMatrix<T>& A) { A.ItUAnalyse(false); },
             [&](BaseMatrix<T>& A, BaseVector<T>* out) {
                 bool ok = A.ItUSolve(2 * n, 0.0, false, *in, out);
                 A.ItUAnalyseClear();
                 return ok;
             }},
        };

    for(const auto& s : solves)
    {
        SCOPED_TRACE("ILU(0), then the " + std::get<0>(s) + " solve");
        auto A = this->make(nonsymmetric<T>(n));
        ASSERT_IN_LOOP(A != nullptr);
        ASSERT_IN_LOOP(A->ILU0Factorize());
        const std::vector<T> LU = dense(csr_of(*A));

        std::vector<T> expected;
        if(std::get<0>(s) == "LU")
        {
            expected = upper_solve(LU, n, lower_solve(LU, n, b, true), false);
        }
        else if(std::get<0>(s) == "L")
        {
            expected = lower_solve(LU, n, b, true);
        }
        else
        {
            expected = upper_solve(LU, n, b, false);
        }

        std::get<1>(s)(*A);
        auto out = make_vector(this->backend(), std::vector<T>(n, 0.0));
        ASSERT_IN_LOOP(std::get<2>(s)(*A, out.get()));
        expect_near(values_of(*out), expected, loose_tol<T>(1e-10));
    }
}

// From about 200 rows on, the iterative analyses need a larger buffer than the
// factorizations leave on the accelerator
TYPED_TEST(backend_csr_preconditioners, iterative_solves_after_a_factorization_of_a_large_matrix)
{
    using T = typename TypeParam::value_type;

    const int            n      = 1000;
    const int            sweeps = 100;
    const std::vector<T> b      = rhs<T>(n);
    auto                 in     = make_vector(this->backend(), b);

    {
        SCOPED_TRACE("IC(0), then the LL solve");
        auto A = this->make(large_tridiagonal<T>(n, static_cast<T>(-1)));
        ASSERT_NE(A, nullptr);
        auto inv_diag = make_vector(this->backend(), std::vector<T>(n, 0.0));
        ASSERT_TRUE(A->ICFactorize(inv_diag.get()));
        const Csr<T> L = filtered(csr_of(*A), lower_part);

        A->ItLLAnalyse();
        auto out = make_vector(this->backend(), std::vector<T>(n, 0.0));
        ASSERT_TRUE(A->ItLLSolve(sweeps, 0.0, false, *in, out.get()));
        A->ItLLAnalyseClear();
        expect_near(values_of(*out),
                    sparse_lower_transposed_solve(L, sparse_lower_solve(L, b, false)),
                    loose_tol<T>(1e-10));
    }

    using Analyse = std::function<void(BaseMatrix<T>&)>;
    using Solve   = std::function<bool(BaseMatrix<T>&, BaseVector<T>*)>;
    using Expect  = std::function<std::vector<T>(const Csr<T>&)>;
    const std::vector<std::tuple<std::string, Analyse, Solve, Expect>> solves = {
        {"LU",
         [](BaseMatrix<T>& A) { A.ItLUAnalyse(); },
         [&](BaseMatrix<T>& A, BaseVector<T>* out) {
             bool ok = A.ItLUSolve(sweeps, 0.0, false, *in, out);
             A.ItLUAnalyseClear();
             return ok;
         },
         [&](const Csr<T>& F) { return sparse_upper_solve(F, sparse_lower_solve(F, b, true)); }},
        {"L",
         [](BaseMatrix<T>& A) { A.ItLAnalyse(true); },
         [&](BaseMatrix<T>& A, BaseVector<T>* out) {
             bool ok = A.ItLSolve(sweeps, 0.0, false, *in, out);
             A.ItLAnalyseClear();
             return ok;
         },
         [&](const Csr<T>& F) { return sparse_lower_solve(F, b, true); }},
        {"L with its diagonal",
         [](BaseMatrix<T>& A) { A.ItLAnalyse(false); },
         [&](BaseMatrix<T>& A, BaseVector<T>* out) {
             bool ok = A.ItLSolve(sweeps, 0.0, false, *in, out);
             A.ItLAnalyseClear();
             return ok;
         },
         [&](const Csr<T>& F) { return sparse_lower_solve(F, b, false); }},
        {"U",
         [](BaseMatrix<T>& A) { A.ItUAnalyse(false); },
         [&](BaseMatrix<T>& A, BaseVector<T>* out) {
             bool ok = A.ItUSolve(sweeps, 0.0, false, *in, out);
             A.ItUAnalyseClear();
             return ok;
         },
         [&](const Csr<T>& F) { return sparse_upper_solve(F, b); }},
    };

    for(const auto& s : solves)
    {
        SCOPED_TRACE("ILU(0), then the " + std::get<0>(s) + " solve");
        auto A = this->make(large_tridiagonal<T>(n, static_cast<T>(-0.5)));
        ASSERT_IN_LOOP(A != nullptr);
        ASSERT_IN_LOOP(A->ILU0Factorize());
        const Csr<T> F = csr_of(*A);

        std::get<1>(s)(*A);
        auto out = make_vector(this->backend(), std::vector<T>(n, 0.0));
        ASSERT_IN_LOOP(std::get<2>(s)(*A, out.get()));
        expect_near(values_of(*out), std::get<3>(s)(F), loose_tol<T>(1e-10));
    }
}

TYPED_TEST(backend_csr_preconditioners, ilut_factorize_drops_small_and_surplus_entries)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(this->backend(), rocalution::CSR, "ILUTFactorize");

    struct Case
    {
        double t;
        int    maxrow;
    };
    const Csr<T> ref = ilut_input<T>();
    for(const Case& c :
        {Case{0.0, 8}, Case{0.05, 8}, Case{-0.05, 8}, Case{0.0, 2}, Case{0.05, 2}, Case{0.1, 1}})
    {
        SCOPED_TRACE("t " + std::to_string(c.t) + ", maxrow " + std::to_string(c.maxrow));
        const Csr<T> expected = ilut(ref, c.t, c.maxrow);

        auto A = this->make(ref);
        ASSERT_IN_LOOP(A != nullptr);
        ASSERT_IN_LOOP(A->ILUTFactorize(c.t, c.maxrow));
        expect_near(csr_of(*A), expected, tol<T>());
    }
}

// L U for a factor that holds L, with a unit diagonal, below the diagonal and U on and
// above it
template <typename T>
std::vector<T> lu_product(const Csr<T>& F)
{
    const int      n = F.n;
    std::vector<T> L(static_cast<size_t>(n) * n, static_cast<T>(0));
    std::vector<T> U(L.size(), static_cast<T>(0));
    for(int i = 0; i < n; ++i)
    {
        L[static_cast<size_t>(i) * n + i] = 1;
        for(PtrType k = F.row_offset[i]; k < F.row_offset[i + 1]; ++k)
        {
            (F.col[k] < i ? L : U)[static_cast<size_t>(i) * n + F.col[k]] = F.val[k];
        }
    }
    return dense_product(L, U, n, n, n);
}

// No other row of the factors depends on the last row, so scaling the last row of A by
// s scales the last row of L U by s and keeps the others: its drops of U are relative
// to the row, its replacement diagonal scales with the row and the largest entries
// stay the largest. A power of two keeps that exact. rocALUTION drops the multipliers
// against the absolute |t| (see ilut), so the property needs every multiplier of the
// last row on the same side of |t| before and after the scaling. Without drops they
// are 0.028 or more in magnitude, so |t| = 0.002 allows s = 1/4 and |t| = 0.01 only
// s = 4.
TYPED_TEST(backend_csr_preconditioners, ilut_of_a_scaled_last_row_scales_that_row_of_the_product)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(this->backend(), rocalution::CSR, "ILUTFactorize");

    struct Case
    {
        double t;
        int    maxrow;
        T      s;
    };
    const Csr<T> ref  = ilut_input<T>();
    const int    last = ref.n - 1;
    for(const Case& c : {Case{0.0, 8, 4},
                         Case{0.0, 8, 0.25},
                         Case{0.0, 2, 0.25},
                         Case{0.0, 2, 4},
                         Case{0.002, 8, 4},
                         Case{0.002, 8, 0.25},
                         Case{-0.01, 8, 4}})
    {
        SCOPED_TRACE("t " + std::to_string(c.t) + ", maxrow " + std::to_string(c.maxrow) + ", s "
                     + std::to_string(c.s));
        const Csr<T> scaled
            = transformed(ref, [&](int i, int, T v) { return i == last ? c.s * v : v; });

        auto A = this->make(ref);
        auto B = this->make(scaled);
        ASSERT_IN_LOOP(A != nullptr && B != nullptr);
        ASSERT_IN_LOOP(A->ILUTFactorize(c.t, c.maxrow));
        ASSERT_IN_LOOP(B->ILUTFactorize(c.t, c.maxrow));

        std::vector<T> expected = lu_product(csr_of(*A));
        for(int j = 0; j < ref.n; ++j)
        {
            expected[static_cast<size_t>(last) * ref.n + j] *= c.s;
        }
        expect_near(lu_product(csr_of(*B)), expected, tol<T>());
    }
}

// A row limit beyond the number of columns keeps every entry that passes the
// threshold, as a limit of n does. The first estimate of the factor's size is
// min(2 maxrow, n) n entries, and 2 maxrow no longer fits an int from 2^30 on.
TYPED_TEST(backend_csr_preconditioners, ilut_factorize_with_a_row_limit_beyond_the_size)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(this->backend(), rocalution::CSR, "ILUTFactorize");

    const Csr<T> ref      = ilut_input<T>();
    const Csr<T> expected = ilut(ref, 0.05, ref.n);

    auto factorize = [&](int maxrow) {
        SCOPED_TRACE("maxrow " + std::to_string(maxrow));
        auto A = this->make(ref);
        ASSERT_NE(A, nullptr);
        ASSERT_TRUE(A->ILUTFactorize(0.05, maxrow));
        expect_near(csr_of(*A), expected, tol<T>());
    };

    factorize(ref.n + 1);
    factorize((1 << 30) - 1);

    KNOWN_BUG("2 * maxrow overflows for maxrow >= 2^30, and the reallocation ends the program");
    expect_in_child([&] { factorize(1 << 30); });
    expect_in_child([&] { factorize(std::numeric_limits<int>::max()); });
}

// Every connected component is numbered from its first unnumbered node, level by
// level
TYPED_TEST(backend_csr_preconditioners, cmk_numbers_each_component_from_its_first_node)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(this->backend(), rocalution::CSR, "CMK");

    // The paths 0-2-4 and 1-5-3-6
    const Csr<T> ref = graph<T>(7, {{0, 2}, {2, 4}, {1, 5}, {5, 3}, {3, 6}});

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);
    auto perm = make_vector(this->backend(), std::vector<int>{});
    ASSERT_TRUE(A->CMK(perm.get()));

    const std::vector<int> p = values_of(*perm);
    ASSERT_TRUE(is_permutation(p));
    EXPECT_EQ((std::vector<int>{p[0], p[2], p[4]}), (std::vector<int>{0, 1, 2}));

    KNOWN_BUG("the restart for the next component numbers every remaining node in index order");
    EXPECT_EQ(p, (std::vector<int>{0, 3, 1, 5, 2, 4, 6}));
    EXPECT_EQ(bandwidth(permuted(ref, p)), 1);
}

// The nodes of a level are numbered as they are found; the next level is searched
// from the nodes of lower degree first
TYPED_TEST(backend_csr_preconditioners, cmk_searches_every_level_by_increasing_degree)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(this->backend(), rocalution::CSR, "CMK");

    // A tree: 0 has the children 1 and 2, 1 has 3 and 9, 2 has 4, 3 has 8, and 4
    // has 5, 6 and 7
    const Csr<T> ref
        = graph<T>(10, {{0, 1}, {0, 2}, {1, 3}, {1, 9}, {2, 4}, {3, 8}, {4, 5}, {4, 6}, {4, 7}});

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);
    auto perm = make_vector(this->backend(), std::vector<int>{});
    ASSERT_TRUE(A->CMK(perm.get()));

    const std::vector<int> p = values_of(*perm);
    ASSERT_TRUE(is_permutation(p));
    // 2 has a lower degree than 1, so its child 4 comes before 3 and 9
    EXPECT_EQ((std::vector<int>{p[0], p[1], p[2], p[4], p[3], p[9]}),
              (std::vector<int>{0, 1, 2, 3, 4, 5}));

    KNOWN_BUG("the flag of the degree sort is never reset, so only the first level is sorted");
    // 9 (degree 1) before 3 (degree 2) before 4 (degree 4): 8 comes before 5, 6 and 7
    EXPECT_EQ(p, (std::vector<int>{0, 1, 2, 4, 3, 7, 8, 9, 6, 5}));
}

TYPED_TEST(backend_csr_preconditioners, key_depends_on_the_signs_of_the_values)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(this->backend(), rocalution::CSR, "Key");

    const Csr<T> ref = nonsymmetric<T>(8);
    auto         A   = this->make(ref);
    ASSERT_NE(A, nullptr);
    auto B = this->make(transformed(ref, [](int, int, T v) { return -v; }));
    ASSERT_NE(B, nullptr);

    long int row_a = 0, col_a = 0, val_a = 0;
    long int row_b = 0, col_b = 0, val_b = 0;
    ASSERT_TRUE(A->Key(row_a, col_a, val_a));
    ASSERT_TRUE(B->Key(row_b, col_b, val_b));
    EXPECT_EQ(row_b, row_a);
    EXPECT_EQ(col_b, col_a);
    EXPECT_NE(val_b, val_a);
}

// Unassigned nodes after the last assigned one have no entry in either operator
TYPED_TEST(backend_csr_preconditioners, create_from_map_with_unassigned_trailing_nodes)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(this->backend(), rocalution::CSR, "CreateFromMap");

    const std::vector<int> map = {1, 0, 1, -1, -1};
    const int              n = 5, m = 2;
    Csr<T>                 restriction(m, n);
    restriction.add(1, 1.0);
    restriction.end_row();
    restriction.add(0, 1.0);
    restriction.add(2, 1.0);
    restriction.end_row();

    auto v = make_vector(this->backend(), map);
    auto R = this->empty();
    auto P = this->empty();
    ASSERT_TRUE(R->CreateFromMap(*v, n, m, P.get()));
    expect_near(csr_of(*R), restriction, 0.0);
    expect_near(csr_of(*P), transposed(restriction), 0.0);
}

// Check reads the arrays of the matrix, which the accelerator does not
TYPED_TEST(backend_csr_preconditioners, check_accepts_empty_matrices)
{
    if(this->backend() != Backend::Host)
    {
        GTEST_SKIP() << "Check is fatal on the accelerator; LocalMatrix checks on the host";
    }

    auto A = this->empty();
    EXPECT_TRUE(A->Check());

    A->AllocateCSR(0, 4, 3);
    EXPECT_TRUE(A->Check());
}

TYPED_TEST(backend_csr_preconditioners, conversion_of_an_empty_matrix_keeps_its_size)
{
    using T = typename TypeParam::value_type;
    for(unsigned int format : {rocalution::CSR, rocalution::COO})
    {
        SCOPED_TRACE(rocalution::_matrix_format_names[format]);
        auto source = empty_matrix<T>(this->backend(), Format{format, 1});
        if(format == rocalution::CSR)
        {
            source->AllocateCSR(0, 5, 7);
        }
        else
        {
            source->AllocateCOO(0, 5, 7);
        }

        auto A = this->make(nonsymmetric<T>(4));
        ASSERT_IN_LOOP(A != nullptr);
        ASSERT_IN_LOOP(A->ConvertFrom(*source));
        EXPECT_EQ(A->GetM(), 5);
        EXPECT_EQ(A->GetN(), 7);
        EXPECT_EQ(A->GetNnz(), 0);
    }
}

TYPED_TEST(backend_csr_preconditioners, conversion_from_csr_copies_the_matrix)
{
    using T = typename TypeParam::value_type;

    const Csr<T> ref    = nonsymmetric<T>(8);
    auto         source = this->make(ref);
    ASSERT_NE(source, nullptr);

    auto A = this->empty();
    ASSERT_TRUE(A->ConvertFrom(*source));
    expect_near(csr_of(*A), ref, 0.0);
}

// LocalMatrix moves a matrix to the host before it converts it there
TYPED_TEST(backend_csr_preconditioners, conversion_from_another_backend_returns_false)
{
    using T = typename TypeParam::value_type;

    const Backend other = this->backend() == Backend::Host ? Backend::Accelerator : Backend::Host;
    if(!available(other))
    {
        GTEST_SKIP() << backend_name(other) << " is not available";
    }

    auto source = make_matrix(other, csr_format, *nonsymmetric<T>(8).host_csr());
    ASSERT_NE(source, nullptr);

    auto A = this->empty();
    EXPECT_FALSE(A->ConvertFrom(*source));
}

// A block without entries gives an empty matrix of the size of the block
TYPED_TEST(backend_csr_preconditioners, extract_sub_matrix_of_a_block_without_entries)
{
    using T = typename TypeParam::value_type;
    // Two diagonal blocks of size 4
    const Csr<T> ref = filtered(spd<T>(8), [](int i, int j, T) { return i / 4 == j / 4; });

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);
    auto S = this->empty();
    ASSERT_TRUE(A->ExtractSubMatrix(0, 4, 4, 3, S.get()));
    EXPECT_EQ(S->GetM(), 4);
    EXPECT_EQ(S->GetN(), 3);
    EXPECT_EQ(S->GetNnz(), 0);
}

// The accelerator sorts the permuted rows with kernels for rows of up to 4, 8, 16,
// 32 and 64 entries, and a fallback for longer rows
TYPED_TEST(backend_csr_preconditioners, permute_rows_of_every_length)
{
    using T = typename TypeParam::value_type;

    const int        n = 100;
    std::vector<int> perm(n);
    for(int i = 0; i < n; ++i)
    {
        perm[i] = (37 * i + 11) % n;
    }
    ASSERT_TRUE(is_permutation(perm));

    // The longest rows have 7, 13, 25, 49 and 81 entries
    for(int half : {3, 6, 12, 24, 40})
    {
        SCOPED_TRACE("half bandwidth " + std::to_string(half));
        Csr<T> ref(n);
        for(int i = 0; i < n; ++i)
        {
            for(int j = std::max(0, i - half); j <= std::min(n - 1, i + half); ++j)
            {
                ref.add(j, static_cast<T>(1 + (7 * i + 3 * j) % 17));
            }
            ref.end_row();
        }

        auto A = this->make(ref);
        ASSERT_IN_LOOP(A != nullptr);
        auto p = make_vector(this->backend(), perm);
        ASSERT_IN_LOOP(A->Permute(*p));
        expect_near(csr_of(*A), permuted(ref, perm), 0.0);
    }
}

// An operation that the backend does not implement returns false and changes
// nothing, so that LocalMatrix can retry on the host
TYPED_TEST(backend_csr_preconditioners, unimplemented_operations_return_false_and_change_nothing)
{
    using T = typename TypeParam::value_type;
    if(approximate_inverse_native(this->backend()))
    {
        GTEST_SKIP() << "every operation is implemented";
    }

    const Csr<T> ref     = spd<T>(8);
    auto         pattern = this->make(ref);
    ASSERT_NE(pattern, nullptr);

    using Operation = std::function<bool(BaseMatrix<T>&)>;
    const std::vector<std::pair<std::string, Operation>> operations = {
        {"FSAI", [](BaseMatrix<T>& A) { return A.FSAI(1, nullptr); }},
        {"FSAI of a power", [](BaseMatrix<T>& A) { return A.FSAI(2, nullptr); }},
        {"FSAI with a pattern", [&](BaseMatrix<T>& A) { return A.FSAI(1, pattern.get()); }},
        {"SPAI", [](BaseMatrix<T>& A) { return A.SPAI(); }},
    };

    for(const auto& operation : operations)
    {
        SCOPED_TRACE(operation.first);
        auto A = this->make(ref);
        ASSERT_IN_LOOP(A != nullptr);
        EXPECT_FALSE(operation.second(*A));
        expect_near(csr_of(*A), ref, 0.0);
    }
    expect_near(csr_of(*pattern), ref, 0.0);
}
