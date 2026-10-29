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

// CSR matrix operations, checked against dense references on every backend.
// The values are small integers, so that the results are exact.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <utility>

#include "unit_test_amg_reference.hpp"
#include "unit_test_common.hpp"
#include "unit_test_reference.hpp"
#include "unit_test_types.hpp"

#include "solvers/preconditioners/preconditioner.hpp"
#include "utils/allocate_free.hpp"

using namespace rocalution_unit;

namespace
{
    bool is_native(Backend b, const std::string& operation)
    {
        return implements(b, rocalution::CSR, operation);
    }

    template <typename T>
    T value(int i, int j)
    {
        return static_cast<T>(10 * i + j + 1);
    }

    // Nonsymmetric n x n matrix: tridiagonal, plus an entry three columns to the
    // right of the diagonal, wrapped around, in every even row
    template <typename T>
    Csr<T> general(int n)
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
                A.add(j, value<T>(i, j));
            }
            A.end_row();
        }
        return A;
    }

    template <typename T>
    void expect_same(const Csr<T>& A, const Csr<T>& ref)
    {
        ASSERT_EQ(A.n, ref.n);
        ASSERT_EQ(A.m, ref.m);
        EXPECT_EQ(A.row_offset, ref.row_offset);
        EXPECT_EQ(A.col, ref.col);
        EXPECT_EQ(A.val, ref.val);
    }

    template <typename T>
    Csr<T> without_entries(int n, int m)
    {
        Csr<T> A(n, m);
        for(int i = 0; i < n; ++i)
        {
            A.end_row();
        }
        return A;
    }

    // general(n) with a dominant diagonal
    template <typename T>
    Csr<T> dominant(int n)
    {
        return transformed(general<T>(n),
                           [](int i, int j, T v) { return i == j ? 4 * (v + 20) : v; });
    }

    // Symmetric positive definite, with the pattern of general(n) and its transpose
    template <typename T>
    Csr<T> spd(int n)
    {
        std::vector<T> D = dense(general<T>(n));
        std::vector<T> S(D.size());
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                S[i * n + j] = D[i * n + j] + D[j * n + i] + (i == j ? 400 : 0);
            }
        }
        return from_dense(n, n, S);
    }

    // ILU(0) as computed in place: strict lower part of L with unit diagonal and
    // U, both on the pattern of A
    template <typename T>
    Csr<T> ilu0(const Csr<T>& A)
    {
        const std::vector<T> D = ilu0_dense(A);
        return transformed(A, [&](int i, int j, T) { return D[i * A.n + j]; });
    }

    // ilu0 with the host's rule for zero pivots, on the non-zero pattern
    template <typename T>
    Csr<T> ilu0_skipping_zero_pivots(const Csr<T>& A)
    {
        const int         n = A.n;
        std::vector<T>    D = dense(A);
        std::vector<bool> P(D.size());
        for(size_t k = 0; k < D.size(); ++k)
        {
            P[k] = D[k] != T(0);
        }

        for(int i = 0; i < n; ++i)
        {
            for(int k = 0; k < i; ++k)
            {
                // A zero pivot leaves the entry and the rest of the row as they are
                if(!P[i * n + k] || D[k * n + k] == T(0))
                {
                    continue;
                }
                D[i * n + k] /= D[k * n + k];
                for(int j = k + 1; j < n; ++j)
                {
                    if(P[i * n + j] && P[k * n + j])
                    {
                        D[i * n + j] -= D[i * n + k] * D[k * n + j];
                    }
                }
            }
        }

        return transformed(A, [&](int i, int j, T) { return D[i * n + j]; });
    }

    // IC(0): the lower triangle L of L L^T, on the pattern of A
    template <typename T>
    Csr<T> ic0(const Csr<T>& A)
    {
        const std::vector<T> L = ic0_dense(A);
        return filtered(transformed(A, [&](int i, int j, T) { return L[i * A.n + j]; }),
                        [](int i, int j, T) { return j <= i; });
    }

    // The permutation groups the rows by color; no two coupled rows share a color
    template <typename T>
    void expect_valid_coloring(const Csr<T>&           A,
                               int                     num_colors,
                               const std::vector<int>& sizes,
                               const std::vector<int>& p)
    {
        ASSERT_EQ(sizes.size(), static_cast<size_t>(num_colors));
        std::vector<int> first(num_colors + 1, 0);
        for(int c = 0; c < num_colors; ++c)
        {
            first[c + 1] = first[c] + sizes[c];
        }
        ASSERT_EQ(first[num_colors], A.n);

        auto color = [&](int i) {
            return static_cast<int>(std::upper_bound(first.begin(), first.end(), p[i])
                                    - first.begin())
                   - 1;
        };
        for(int i = 0; i < A.n; ++i)
        {
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                if(A.col[j] != i)
                {
                    EXPECT_NE(color(i), color(A.col[j])) << "rows " << i << " and " << A.col[j];
                }
            }
        }
    }

    bool is_permutation(const std::vector<int>& perm)
    {
        std::vector<int> sorted = perm;
        std::sort(sorted.begin(), sorted.end());
        for(size_t i = 0; i < sorted.size(); ++i)
        {
            if(sorted[i] != static_cast<int>(i))
            {
                return false;
            }
        }
        return true;
    }

    // Pattern of A^p, without cancellation
    template <typename T>
    std::vector<bool> power_pattern(const Csr<T>& A, int p)
    {
        const int         n = A.n;
        std::vector<bool> base(n * n), result(n * n);
        for(int i = 0; i < n; ++i)
        {
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                base[i * n + A.col[j]] = true;
            }
        }

        result = base;
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

    // ILU(p) with fill levels: an entry is kept if its level is at most p
    template <typename T>
    Csr<T> ilup(const Csr<T>& A, int p)
    {
        const int               n   = A.n;
        const int               inf = 1 << 20;
        std::vector<T>          D   = dense(A);
        const std::vector<bool> P   = pattern(A);
        std::vector<int>        lev(D.size());
        for(size_t k = 0; k < D.size(); ++k)
        {
            lev[k] = P[k] ? 0 : inf;
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

    // Same values, and sorted columns, but explicit zeros are allowed
    template <typename T>
    void expect_same_values(const Csr<T>& A, const Csr<T>& ref)
    {
        ASSERT_EQ(A.n, ref.n);
        ASSERT_EQ(A.m, ref.m);
        EXPECT_EQ(dense(A), dense(ref));

        for(int i = 0; i < A.n; ++i)
        {
            EXPECT_TRUE(std::is_sorted(A.col.begin() + A.row_offset[i],
                                       A.col.begin() + A.row_offset[i + 1]))
                << "row " << i << " is not sorted";
        }
    }

    // Error bound of the iterative solves and of the iterative ILU(0), which stop at a
    // tolerance of 1e-14. Float cannot reach it, so it runs to the iteration limit and
    // its error is rounding.
    template <typename T>
    double iterative_tol()
    {
        return loose_tol<T>(1e-10);
    }

    template <typename P>
    class matrix_csr : public TypedBackendTest<P>
    {
    protected:
        using T = typename P::value_type;

        std::unique_ptr<BaseMatrix<T>> make(const Csr<T>& ref)
        {
            return make_matrix(P::backend, Format{rocalution::CSR, 1}, *ref.host_csr());
        }

        std::unique_ptr<BaseMatrix<T>> empty()
        {
            return empty_matrix<T>(P::backend, Format{rocalution::CSR, 1});
        }
    };
}

TYPED_TEST_SUITE(matrix_csr, WithBackends<RealTypes>, TypedName);

TYPED_TEST(matrix_csr, value_operations_keep_the_pattern)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = general<T>(8);
    const std::vector<T> diag{1, 2, 3, 4, 5, 6, 7, 8};
    auto                 d = make_vector(TypeParam::backend, diag);

    struct Operation
    {
        std::string                         name;
        std::function<bool(BaseMatrix<T>&)> apply;
        std::function<T(int, int, T)>       expected;
    };

    const std::vector<Operation> operations = {
        {"Zeros", [](auto& A) { return A.Zeros(); }, [](int, int, T) { return T(0); }},
        {"Scale", [](auto& A) { return A.Scale(2.0); }, [](int, int, T v) { return 2 * v; }},
        {"ScaleDiagonal",
         [](auto& A) { return A.ScaleDiagonal(3.0); },
         [](int i, int j, T v) { return i == j ? 3 * v : v; }},
        {"ScaleOffDiagonal",
         [](auto& A) { return A.ScaleOffDiagonal(-1.0); },
         [](int i, int j, T v) { return i == j ? v : -v; }},
        {"AddScalar",
         [](auto& A) { return A.AddScalar(0.5); },
         [](int, int, T v) { return v + T(0.5); }},
        {"AddScalarDiagonal",
         [](auto& A) { return A.AddScalarDiagonal(0.5); },
         [](int i, int j, T v) { return i == j ? v + T(0.5) : v; }},
        {"AddScalarOffDiagonal",
         [](auto& A) { return A.AddScalarOffDiagonal(0.5); },
         [](int i, int j, T v) { return i == j ? v : v + T(0.5); }},
        {"DiagonalMatrixMultR",
         [&](auto& A) { return A.DiagonalMatrixMultR(*d); },
         [&](int, int j, T v) { return v * diag[j]; }},
        {"DiagonalMatrixMultL",
         [&](auto& A) { return A.DiagonalMatrixMultL(*d); },
         [&](int i, int, T v) { return v * diag[i]; }},
    };

    for(const Operation& op : operations)
    {
        SCOPED_TRACE(op.name);

        auto A = this->make(ref);
        ASSERT_IN_LOOP(A != nullptr);
        ASSERT_IN_LOOP(op.apply(*A));
        expect_same(csr_of(*A), transformed(ref, op.expected));
    }
}

// A zero on the diagonal is replaced with one
TYPED_TEST(matrix_csr, extract_inverse_diagonal)
{
    using T = typename TypeParam::value_type;
    const Csr<T> ref
        = transformed(general<T>(8), [](int i, int j, T v) { return i == j && i == 5 ? 0 : v; });
    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);

    auto inv = make_vector(TypeParam::backend, std::vector<T>(ref.n, -7.0));
    ASSERT_TRUE(A->ExtractInverseDiagonal(inv.get()));

    std::vector<T> result = values_of(*inv);
    for(int i = 0; i < ref.n; ++i)
    {
        // Within 4 ulps of T, as EXPECT_FLOAT_EQ and EXPECT_DOUBLE_EQ
        EXPECT_PRED_FORMAT2(testing::internal::CmpHelperFloatingPointEQ<T>,
                            result[i],
                            i == 5 ? T(1) : T(1) / value<T>(i, i))
            << "at row " << i;
    }
}

TYPED_TEST(matrix_csr, extract_triangular_parts)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = general<T>(8);
    auto         A   = this->make(ref);
    ASSERT_NE(A, nullptr);

    struct Part
    {
        std::string                                               name;
        std::function<bool(const BaseMatrix<T>&, BaseMatrix<T>*)> extract;
        std::function<bool(int, int)>                             keep;
    };

    const std::vector<Part> parts = {
        {"ExtractU",
         [](auto& M, auto* out) { return M.ExtractU(out); },
         [](int i, int j) { return j > i; }},
        {"ExtractUDiagonal",
         [](auto& M, auto* out) { return M.ExtractUDiagonal(out); },
         [](int i, int j) { return j >= i; }},
        {"ExtractL",
         [](auto& M, auto* out) { return M.ExtractL(out); },
         [](int i, int j) { return j < i; }},
        {"ExtractLDiagonal",
         [](auto& M, auto* out) { return M.ExtractLDiagonal(out); },
         [](int i, int j) { return j <= i; }},
    };

    for(const Part& part : parts)
    {
        SCOPED_TRACE(part.name);

        auto out = this->empty();
        ASSERT_IN_LOOP(part.extract(*A, out.get()));
        expect_same(csr_of(*out), filtered(ref, [&](int i, int j, T) { return part.keep(i, j); }));
    }
}

TYPED_TEST(matrix_csr, extract_sub_matrix)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = general<T>(8);
    auto         A   = this->make(ref);
    ASSERT_NE(A, nullptr);

    // Rows 2 to 4, columns 1 to 4
    auto sub = this->empty();
    ASSERT_TRUE(A->ExtractSubMatrix(2, 1, 3, 4, sub.get()));

    Csr<T> expected(3, 4);
    for(int i = 2; i < 5; ++i)
    {
        for(PtrType j = ref.row_offset[i]; j < ref.row_offset[i + 1]; ++j)
        {
            if(ref.col[j] >= 1 && ref.col[j] < 5)
            {
                expected.add(ref.col[j] - 1, ref.val[j]);
            }
        }
        expected.end_row();
    }
    expect_same(csr_of(*sub), expected);
}

TYPED_TEST(matrix_csr, transpose)
{
    using T = typename TypeParam::value_type;

    // A 5 x 8 matrix, to tell rows from columns
    Csr<T> ref(5, 8);
    for(int i = 0; i < 5; ++i)
    {
        ref.add(i, value<T>(i, i));
        ref.add(i + 3, value<T>(i, i + 3));
        ref.end_row();
    }

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);

    auto At = this->empty();
    ASSERT_TRUE(A->Transpose(At.get()));
    expect_same(csr_of(*At), transposed(ref));

    ASSERT_TRUE(A->Transpose());
    expect_same(csr_of(*A), transposed(ref));
}

TYPED_TEST(matrix_csr, sort_orders_the_columns_of_each_row)
{
    using T               = typename TypeParam::value_type;
    const Csr<T> ref      = general<T>(8);
    Csr<T>       unsorted = ref;
    for(int i = 0; i < ref.n; ++i)
    {
        std::reverse(unsorted.col.begin() + ref.row_offset[i],
                     unsorted.col.begin() + ref.row_offset[i + 1]);
        std::reverse(unsorted.val.begin() + ref.row_offset[i],
                     unsorted.val.begin() + ref.row_offset[i + 1]);
    }

    auto A = this->make(unsorted);
    ASSERT_NE(A, nullptr);
    ASSERT_TRUE(A->Sort());
    expect_same(csr_of(*A), ref);
}

// Entries at most drop_off in magnitude are removed, except on the diagonal. Rows 3
// and 5 hold entries of exactly +-drop_off.
TYPED_TEST(matrix_csr, compress_drops_small_off_diagonal_entries)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = transformed(general<T>(8), [](int i, int j, T v) {
        return i == 3 ? T(0.25) : i == 5 ? T(-0.25) : j % 2 == 1 ? T(0.125) : v;
    });

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);
    ASSERT_TRUE(A->Compress(0.25));

    expect_same(csr_of(*A),
                filtered(ref, [](int i, int j, T v) { return std::abs(v) > T(0.25) || i == j; }));
}

// Row and column i move to perm[i]
TYPED_TEST(matrix_csr, permute)
{
    using T                     = typename TypeParam::value_type;
    const Csr<T>           ref  = general<T>(8);
    const std::vector<int> perm = {3, 0, 6, 1, 7, 2, 5, 4};

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);
    auto p = make_vector(TypeParam::backend, perm);
    ASSERT_TRUE(A->Permute(*p));

    expect_same(csr_of(*A), permuted(ref, perm));
}

TYPED_TEST(matrix_csr, apply_add)
{
    using T            = typename TypeParam::value_type;
    const Csr<T>   ref = general<T>(8);
    std::vector<T> x(ref.n), y(ref.n);
    for(int i = 0; i < ref.n; ++i)
    {
        x[i] = static_cast<T>(i % 3 + 1);
        y[i] = static_cast<T>(100 + i);
    }

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);
    auto in  = make_vector(TypeParam::backend, x);
    auto out = make_vector(TypeParam::backend, y);

    A->ApplyAdd(*in, 2.0, out.get());

    std::vector<T> Ax = ref.apply(x);
    for(int i = 0; i < ref.n; ++i)
    {
        y[i] += 2 * Ax[i];
    }
    EXPECT_EQ(values_of(*out), y);
}

// Without structure, only the entries of the pattern of the matrix are updated
TYPED_TEST(matrix_csr, matrix_add_on_the_same_pattern)
{
    using T        = typename TypeParam::value_type;
    const Csr<T> a = general<T>(8);
    const Csr<T> b = transformed(a, [](int i, int j, T) { return static_cast<T>(i - j + 0.5); });

    auto A = this->make(a);
    auto B = this->make(b);
    ASSERT_NE(A, nullptr);
    ASSERT_NE(B, nullptr);
    ASSERT_TRUE(A->MatrixAdd(*B, 2.0, -1.0, false));

    expect_same(csr_of(*A), transformed(a, [](int i, int j, T v) {
                    return 2 * v - static_cast<T>(i - j + 0.5);
                }));
}

// With structure, the pattern is the union of both patterns
TYPED_TEST(matrix_csr, matrix_add_with_structure)
{
    using T        = typename TypeParam::value_type;
    const Csr<T> a = general<T>(8);
    const Csr<T> b = transposed(a);

    auto A = this->make(a);
    auto B = this->make(b);
    ASSERT_NE(A, nullptr);
    ASSERT_NE(B, nullptr);
    ASSERT_TRUE(A->MatrixAdd(*B, 2.0, -1.0, true));

    std::vector<T> Da = dense(a), Db = dense(b), Dc(Da.size());
    int64_t        union_nnz = 0;
    for(size_t k = 0; k < Da.size(); ++k)
    {
        Dc[k] = 2 * Da[k] - Db[k];
        union_nnz += (Da[k] != T(0) || Db[k] != T(0));
    }

    Csr<T> result = csr_of(*A);
    EXPECT_EQ(result.nnz(), union_nnz);
    expect_same_values(result, from_dense(a.n, a.n, Dc));
}

TYPED_TEST(matrix_csr, mat_mat_mult)
{
    using T        = typename TypeParam::value_type;
    const Csr<T> a = general<T>(8);
    const Csr<T> b = transposed(a);

    auto A = this->make(a);
    auto B = this->make(b);
    ASSERT_NE(A, nullptr);
    ASSERT_NE(B, nullptr);

    auto C = this->empty();
    ASSERT_TRUE(C->MatMatMult(*A, *B));

    expect_same_values(csr_of(*C),
                       from_dense(a.n, a.n, dense_product(dense(a), dense(b), a.n, a.n, a.n)));
}

// The bounds are the extreme ends of the Gershgorin discs. By the library's
// convention both bounds start at zero, so the interval always contains zero.
TYPED_TEST(matrix_csr, gershgorin)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(TypeParam::backend, rocalution::CSR, "Gershgorin");

    const Csr<T> ref
        = transformed(general<T>(8), [](int i, int j, T v) { return i == j ? v : -v / 4; });

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);

    T lambda_min = -1, lambda_max = -1;
    ASSERT_TRUE(A->Gershgorin(lambda_min, lambda_max));

    T expected_min = 0, expected_max = 0;
    for(int i = 0; i < ref.n; ++i)
    {
        T radius = 0, center = 0;
        for(PtrType j = ref.row_offset[i]; j < ref.row_offset[i + 1]; ++j)
        {
            if(ref.col[j] == i)
            {
                center = ref.val[j];
            }
            else
            {
                radius += std::abs(ref.val[j]);
            }
        }
        expected_min = std::min(expected_min, center - radius);
        expected_max = std::max(expected_max, center + radius);
    }
    EXPECT_EQ(lambda_min, expected_min);
    EXPECT_EQ(lambda_max, expected_max);
}

TYPED_TEST(matrix_csr, extract_row_and_column_vectors)
{
    using T            = typename TypeParam::value_type;
    const Csr<T>   ref = general<T>(8);
    std::vector<T> D   = dense(ref);
    auto           A   = this->make(ref);
    ASSERT_NE(A, nullptr);

    for(int idx : {0, 3, 7})
    {
        SCOPED_TRACE("index " + std::to_string(idx));

        auto row = make_vector(TypeParam::backend, std::vector<T>(ref.n, -7.0));
        ASSERT_IN_LOOP(A->ExtractRowVector(idx, row.get()));
        EXPECT_EQ(values_of(*row),
                  std::vector<T>(D.begin() + idx * ref.n, D.begin() + (idx + 1) * ref.n));

        auto column = make_vector(TypeParam::backend, std::vector<T>(ref.n, -7.0));
        ASSERT_IN_LOOP(A->ExtractColumnVector(idx, column.get()));
        std::vector<T> expected(ref.n);
        for(int i = 0; i < ref.n; ++i)
        {
            expected[i] = D[i * ref.n + idx];
        }
        EXPECT_EQ(values_of(*column), expected);
    }
}

// Zeros in the vector remove entries, the other values replace or insert them
TYPED_TEST(matrix_csr, replace_column_vector)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = general<T>(8);
    const int    idx = 2;

    auto expect_replaced = [&](const std::vector<T>& column) {
        auto A = this->make(ref);
        ASSERT_NE(A, nullptr);
        ASSERT_TRUE(A->ReplaceColumnVector(idx, *make_vector(TypeParam::backend, column)));

        std::vector<T> D = dense(ref);
        for(int i = 0; i < ref.n; ++i)
        {
            D[i * ref.n + idx] = column[i];
        }
        expect_same(csr_of(*A), from_dense(ref.n, ref.n, D));
    };

    auto checks = [&] {
        // Rows 1 and 3 hold column 2 and lose it, row 2 keeps it with a new value
        std::vector<T> column(ref.n, T(0));
        column[2] = 70;
        {
            SCOPED_TRACE("removed and replaced entries");
            expect_replaced(column);
        }

        KNOWN_BUG("inserting an entry drops the next entry of its row");

        // Row 4 does not hold column 2 and gains it before its column 3
        column[4] = 90;
        SCOPED_TRACE("an inserted entry");
        expect_replaced(column);
    };

    if(TypeParam::backend == Backend::Accelerator)
    {
        KNOWN_BUG("the scan does not write the last row offset, so a garbage nnz leads to an "
                  "illegal memory access that ends the process");
        expect_in_child(checks);
        return;
    }
    checks();
}

TYPED_TEST(matrix_csr, replace_row_vector)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(TypeParam::backend, rocalution::CSR, "ReplaceRowVector");

    const Csr<T> ref = general<T>(8);
    const int    idx = 3;

    std::vector<T> row(ref.n, T(0));
    row[0] = 1;
    row[3] = 2;
    row[7] = 3;

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);
    ASSERT_TRUE(A->ReplaceRowVector(idx, *make_vector(TypeParam::backend, row)));

    std::vector<T> D = dense(ref);
    std::copy(row.begin(), row.end(), D.begin() + idx * ref.n);
    expect_same(csr_of(*A), from_dense(ref.n, ref.n, D));
}

// The strict lower part with a unit diagonal is L, the rest is U
TYPED_TEST(matrix_csr, lu_solve)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = dominant<T>(8);
    const std::vector<T> b   = rhs<T>(ref.n);
    const std::vector<T> D   = dense(ref);

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);
    auto in  = make_vector(TypeParam::backend, b);
    auto out = make_vector(TypeParam::backend, std::vector<T>(ref.n, T(0)));

    A->LUAnalyse();
    ASSERT_TRUE(A->LUSolve(*in, out.get()));
    A->LUAnalyseClear();

    expect_near(
        values_of(*out), upper_solve(D, ref.n, lower_solve(D, ref.n, b, true), false), tol<T>());
}

// The matrix is the lower triangle L of L L^T, with the diagonal last in each row
TYPED_TEST(matrix_csr, ll_solve)
{
    using T                = typename TypeParam::value_type;
    const Csr<T>         L = filtered(spd<T>(8), [](int i, int j, T) { return j <= i; });
    const std::vector<T> b = rhs<T>(L.n);
    const std::vector<T> D = dense(L);

    std::vector<T> Lt(D.size());
    std::vector<T> inv_diag(L.n);
    for(int i = 0; i < L.n; ++i)
    {
        inv_diag[i] = T(1) / D[i * L.n + i];
        for(int j = 0; j < L.n; ++j)
        {
            Lt[j * L.n + i] = D[i * L.n + j];
        }
    }
    const std::vector<T> expected = upper_solve(Lt, L.n, lower_solve(D, L.n, b, false), false);

    auto A = this->make(L);
    ASSERT_NE(A, nullptr);
    auto in  = make_vector(TypeParam::backend, b);
    auto inv = make_vector(TypeParam::backend, inv_diag);

    A->LLAnalyse();

    auto out = make_vector(TypeParam::backend, std::vector<T>(L.n, T(0)));
    ASSERT_TRUE(A->LLSolve(*in, out.get()));
    expect_near(values_of(*out), expected, tol<T>());

    auto out_inv = make_vector(TypeParam::backend, std::vector<T>(L.n, T(0)));
    ASSERT_TRUE(A->LLSolve(*in, *inv, out_inv.get()));
    expect_near(values_of(*out_inv), expected, tol<T>());

    A->LLAnalyseClear();
}

// Only the triangle of the solve is read from the full matrix
TYPED_TEST(matrix_csr, triangular_solves)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = dominant<T>(8);
    const std::vector<T> b   = rhs<T>(ref.n);
    const std::vector<T> D   = dense(ref);

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);
    auto in = make_vector(TypeParam::backend, b);

    for(bool unit : {false, true})
    {
        SCOPED_TRACE(unit ? "unit diagonal" : "diagonal of the matrix");

        auto out = make_vector(TypeParam::backend, std::vector<T>(ref.n, T(0)));
        A->LAnalyse(unit);
        ASSERT_IN_LOOP(A->LSolve(*in, out.get()));
        A->LAnalyseClear();
        expect_near(values_of(*out), lower_solve(D, ref.n, b, unit), tol<T>());

        A->UAnalyse(unit);
        ASSERT_IN_LOOP(A->USolve(*in, out.get()));
        A->UAnalyseClear();
        expect_near(values_of(*out), upper_solve(D, ref.n, b, unit), tol<T>());
    }
}

// The iterative solves converge to the solutions of the direct ones
TYPED_TEST(matrix_csr, iterative_triangular_solves)
{
    using T               = typename TypeParam::value_type;
    const int    max_iter = 100;
    const double tol      = 1e-14;

    const Csr<T>         ref = dominant<T>(8);
    const std::vector<T> b   = rhs<T>(ref.n);
    const std::vector<T> D   = dense(ref);

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);
    auto in  = make_vector(TypeParam::backend, b);
    auto out = make_vector(TypeParam::backend, std::vector<T>(ref.n, T(0)));

    A->ItLUAnalyse();
    ASSERT_TRUE(A->ItLUSolve(max_iter, tol, true, *in, out.get()));
    A->ItLUAnalyseClear();
    expect_near(values_of(*out),
                upper_solve(D, ref.n, lower_solve(D, ref.n, b, true), false),
                iterative_tol<T>());

    for(bool unit : {false, true})
    {
        SCOPED_TRACE(unit ? "unit diagonal" : "diagonal of the matrix");

        out = make_vector(TypeParam::backend, std::vector<T>(ref.n, T(0)));
        A->ItLAnalyse(unit);
        ASSERT_IN_LOOP(A->ItLSolve(max_iter, tol, true, *in, out.get()));
        A->ItLAnalyseClear();
        expect_near(values_of(*out), lower_solve(D, ref.n, b, unit), iterative_tol<T>());

        out = make_vector(TypeParam::backend, std::vector<T>(ref.n, T(0)));
        A->ItUAnalyse(unit);
        ASSERT_IN_LOOP(A->ItUSolve(max_iter, tol, true, *in, out.get()));
        A->ItUAnalyseClear();
        expect_near(values_of(*out), upper_solve(D, ref.n, b, unit), iterative_tol<T>());
    }
}

// The unit lower solve converges in one iteration, the upper solve needs more
TYPED_TEST(matrix_csr, iterative_lu_solve_of_an_upper_triangular_matrix)
{
    using T                = typename TypeParam::value_type;
    const Csr<T>         U = filtered(dominant<T>(8), [](int i, int j, T) { return j >= i; });
    const std::vector<T> b = rhs<T>(U.n);

    auto A = this->make(U);
    ASSERT_NE(A, nullptr);
    auto in  = make_vector(TypeParam::backend, b);
    auto out = make_vector(TypeParam::backend, std::vector<T>(U.n, T(0)));

    if(TypeParam::backend == Backend::Host)
    {
        KNOWN_BUG("the iteration count of the L solve becomes the limit of the U solve");
    }

    A->ItLUAnalyse();
    ASSERT_TRUE(A->ItLUSolve(100, 1e-14, true, *in, out.get()));
    A->ItLUAnalyseClear();
    expect_near(values_of(*out), upper_solve(dense(U), U.n, b, false), iterative_tol<T>());
}

TYPED_TEST(matrix_csr, iterative_ll_solve)
{
    using T                = typename TypeParam::value_type;
    const Csr<T>         L = filtered(spd<T>(8), [](int i, int j, T) { return j <= i; });
    const std::vector<T> b = rhs<T>(L.n);
    const std::vector<T> D = dense(L);

    std::vector<T> Lt(D.size());
    std::vector<T> inv_diag(L.n);
    for(int i = 0; i < L.n; ++i)
    {
        inv_diag[i] = T(1) / D[i * L.n + i];
        for(int j = 0; j < L.n; ++j)
        {
            Lt[j * L.n + i] = D[i * L.n + j];
        }
    }
    const std::vector<T> expected = upper_solve(Lt, L.n, lower_solve(D, L.n, b, false), false);

    auto A = this->make(L);
    ASSERT_NE(A, nullptr);
    auto in  = make_vector(TypeParam::backend, b);
    auto inv = make_vector(TypeParam::backend, inv_diag);

    // Float does not reach 1e-14, so both solves run to the limit and the bug does not show
    if(TypeParam::backend == Backend::Host && !std::is_same_v<T, float>)
    {
        KNOWN_BUG("the iteration count of the L solve becomes the limit of the L^T solve");
    }

    A->ItLLAnalyse();

    auto out = make_vector(TypeParam::backend, std::vector<T>(L.n, T(0)));
    ASSERT_TRUE(A->ItLLSolve(100, 1e-14, true, *in, out.get()));
    expect_near(values_of(*out), expected, iterative_tol<T>());

    auto out_inv = make_vector(TypeParam::backend, std::vector<T>(L.n, T(0)));
    ASSERT_TRUE(A->ItLLSolve(100, 1e-14, true, *in, *inv, out_inv.get()));
    expect_near(values_of(*out_inv), expected, iterative_tol<T>());

    A->ItLLAnalyseClear();
}

TYPED_TEST(matrix_csr, ilu0_factorize)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = dominant<T>(8);
    auto         A   = this->make(ref);
    ASSERT_NE(A, nullptr);

    ASSERT_TRUE(A->ILU0Factorize());
    expect_near(csr_of(*A), ilu0(ref), tol<T>());
}

// The lower triangle becomes L of L L^T
TYPED_TEST(matrix_csr, ic_factorize)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = spd<T>(8);
    const Csr<T> L   = ic0(ref);

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);
    auto inv_diag = make_vector(TypeParam::backend, std::vector<T>(ref.n, T(0)));

    ASSERT_TRUE(A->ICFactorize(inv_diag.get()));

    expect_near(filtered(csr_of(*A), [](int i, int j, T) { return j <= i; }), L, tol<T>());

    std::vector<T> D = dense(L);
    std::vector<T> expected(ref.n);
    for(int i = 0; i < ref.n; ++i)
    {
        expected[i] = T(1) / D[i * ref.n + i];
    }
    expect_near(values_of(*inv_diag), expected, tol<T>());
}

// Without dropping, ILUT is the complete LU factorization, with fill-in
TYPED_TEST(matrix_csr, ilut_factorize_without_dropping)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(TypeParam::backend, rocalution::CSR, "ILUTFactorize");

    const Csr<T> ref = dominant<T>(8);
    auto         A   = this->make(ref);
    ASSERT_NE(A, nullptr);
    ASSERT_TRUE(A->ILUTFactorize(0.0, ref.n));

    std::vector<T> LU = dense(csr_of(*A));
    std::vector<T> product(LU.size(), T(0));
    for(int i = 0; i < ref.n; ++i)
    {
        for(int j = 0; j < ref.n; ++j)
        {
            for(int k = 0; k <= std::min(i, j); ++k)
            {
                T l = (k == i) ? T(1) : LU[i * ref.n + k];
                product[i * ref.n + j] += l * LU[k * ref.n + j];
            }
        }
    }
    expect_near(product, dense(ref), tol<T>());
}

// ILU(p) needs the pattern of A^(p+1), which contains every fill-in of level p
TYPED_TEST(matrix_csr, ilup_factorize_numeric)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(TypeParam::backend, rocalution::CSR, "ILUpFactorizeNumeric");

    const Csr<T> ref = dominant<T>(8);

    for(int p : {1, 2, 8})
    {
        SCOPED_TRACE("level " + std::to_string(p));

        HostMatrixCSR<T> host_structure(descriptor());
        host_structure.CopyFrom(*ref.host_csr());
        ASSERT_IN_LOOP(host_structure.SymbolicPower(p + 1));
        auto structure
            = make_matrix(TypeParam::backend, Format{rocalution::CSR, 1}, host_structure);
        ASSERT_IN_LOOP(structure != nullptr);

        auto A = this->make(ref);
        ASSERT_IN_LOOP(A != nullptr);

        ASSERT_IN_LOOP(A->ILUpFactorizeNumeric(p, *structure));
        expect_near(csr_of(*A), ilup(ref, p), tol<T>());
    }
}

// A scrambled tridiagonal matrix gets its band back. The search starts at row
// 0, which has to be an end of the path for a bandwidth of 1.
TYPED_TEST(matrix_csr, cmk_and_rcmk_reduce_the_bandwidth)
{
    using T                         = typename TypeParam::value_type;
    const std::vector<int> scramble = {0, 11, 5, 7, 3, 9, 1, 10, 6, 2, 8, 4};
    const Csr<T>           ref      = permuted(Csr<T>(Tridiagonal<T>(12)), scramble);
    ASSERT_GT(bandwidth(ref), 1);

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);

    for(const std::string name : {"CMK", "RCMK"})
    {
        SCOPED_TRACE(name);
        REQUIRE_NATIVE(TypeParam::backend, rocalution::CSR, name);

        auto perm = make_vector(TypeParam::backend, std::vector<int>());
        ASSERT_IN_LOOP(name == "CMK" ? A->CMK(perm.get()) : A->RCMK(perm.get()));

        std::vector<int> p = values_of(*perm);
        ASSERT_IN_LOOP(is_permutation(p));
        EXPECT_EQ(bandwidth(permuted(ref, p)), 1);
    }
}

// Used as documented, mat.ConnectivityOrder(&conn); mat.Permute(conn);
TYPED_TEST(matrix_csr, connectivity_order_sorts_the_rows_by_their_non_zeros)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(TypeParam::backend, rocalution::CSR, "ConnectivityOrder");

    const Csr<T> ref = general<T>(8);
    auto         A   = this->make(ref);
    ASSERT_NE(A, nullptr);

    auto perm = make_vector(TypeParam::backend, std::vector<int>());
    ASSERT_TRUE(A->ConnectivityOrder(perm.get()));

    std::vector<int> p = values_of(*perm);
    ASSERT_TRUE(is_permutation(p));

    KNOWN_BUG("it returns perm[new] = old instead of the perm[old] = new that Permute takes");

    Csr<T> sorted = permuted(ref, p);
    for(int i = 1; i < sorted.n; ++i)
    {
        EXPECT_LE(sorted.row_offset[i] - sorted.row_offset[i - 1],
                  sorted.row_offset[i + 1] - sorted.row_offset[i])
            << "rows " << i - 1 << " and " << i;
    }
}

// The permutation groups the rows by color; no two coupled rows share a color
TYPED_TEST(matrix_csr, multi_coloring)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = general<T>(8);
    auto         A   = this->make(ref);
    ASSERT_NE(A, nullptr);

    int  num_colors  = 0;
    int* size_colors = nullptr;
    auto perm        = make_vector(TypeParam::backend, std::vector<int>());
    ASSERT_TRUE(A->MultiColoring(num_colors, &size_colors, perm.get()));

    std::vector<int> sizes(size_colors, size_colors + num_colors);
    rocalution::free_host(&size_colors);

    std::vector<int> p = values_of(*perm);
    ASSERT_TRUE(is_permutation(p));
    expect_valid_coloring(ref, num_colors, sizes, p);
}

// The host gives a row and column without entries the first color
TYPED_TEST(matrix_csr, multi_coloring_of_a_matrix_with_an_isolated_row)
{
    using T               = typename TypeParam::value_type;
    const int    isolated = 3;
    const Csr<T> ref
        = filtered(general<T>(8), [&](int i, int j, T) { return i != isolated && j != isolated; });
    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);

    int  num_colors  = 0;
    int* size_colors = nullptr;
    auto perm        = make_vector(TypeParam::backend, std::vector<int>());
    ASSERT_TRUE(A->MultiColoring(num_colors, &size_colors, perm.get()));

    std::vector<int> sizes(size_colors, size_colors + num_colors);
    rocalution::free_host(&size_colors);

    std::vector<int> p = values_of(*perm);
    ASSERT_TRUE(is_permutation(p));
    expect_valid_coloring(ref, num_colors, sizes, p);
    if(TypeParam::backend == Backend::Host)
    {
        ASSERT_GE(num_colors, 1);
        EXPECT_LT(p[isolated], sizes[0]);
    }
}

// The first size rows of the permutation form an independent set, and every
// other row is coupled to one of them
TYPED_TEST(matrix_csr, maximal_independent_set)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = spd<T>(8);
    auto         A   = this->make(ref);
    ASSERT_NE(A, nullptr);

    int  size = 0;
    auto perm = make_vector(TypeParam::backend, std::vector<int>());
    ASSERT_TRUE(A->MaximalIndependentSet(size, perm.get()));

    std::vector<int> p = values_of(*perm);
    ASSERT_TRUE(is_permutation(p));

    for(int i = 0; i < ref.n; ++i)
    {
        bool in_set  = p[i] < size;
        bool coupled = false;
        for(PtrType j = ref.row_offset[i]; j < ref.row_offset[i + 1]; ++j)
        {
            if(ref.col[j] != i && p[ref.col[j]] < size)
            {
                coupled = true;
            }
        }
        EXPECT_NE(in_set, coupled) << "row " << i;
    }
}

// Rows without a diagonal entry move to the end, both groups keep their order
TYPED_TEST(matrix_csr, zero_block_permutation)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(TypeParam::backend, rocalution::CSR, "ZeroBlockPermutation");

    const Csr<T> ref
        = filtered(general<T>(8), [](int i, int j, T) { return i != j || (i != 2 && i != 5); });
    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);

    int  size = -1;
    auto perm = make_vector(TypeParam::backend, std::vector<int>(ref.n, -1));
    ASSERT_TRUE(A->ZeroBlockPermutation(size, perm.get()));

    EXPECT_EQ(size, 6);
    EXPECT_EQ(values_of(*perm), (std::vector<int>{0, 1, 6, 2, 3, 7, 4, 5}));
}

TYPED_TEST(matrix_csr, symbolic_power)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(TypeParam::backend, rocalution::CSR, "SymbolicPower");

    const Csr<T> ref = Tridiagonal<T>(24);

    for(int p = 2; p <= 10; ++p)
    {
        SCOPED_TRACE("power " + std::to_string(p));

        if(p > 8 && TypeParam::backend == Backend::Host)
        {
            KNOWN_BUG("above 8 it gives the pattern of A^(p+1)");
        }

        auto A = this->make(ref);
        ASSERT_IN_LOOP(A != nullptr);

        ASSERT_IN_LOOP(A->SymbolicPower(p));
        EXPECT_EQ(pattern(csr_of(*A)), power_pattern(ref, p));
    }
}

TYPED_TEST(matrix_csr, symbolic_and_numeric_mat_mat_mult)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(TypeParam::backend, rocalution::CSR, "SymbolicMatMatMult");

    const Csr<T> a = general<T>(8);
    const Csr<T> b = transposed(a);

    auto A = this->make(a);
    auto B = this->make(b);
    ASSERT_NE(A, nullptr);
    ASSERT_NE(B, nullptr);

    auto C = this->empty();
    ASSERT_TRUE(C->SymbolicMatMatMult(*A, *B));
    ASSERT_TRUE(C->NumericMatMatMult(*A, *B));

    expect_same_values(csr_of(*C),
                       from_dense(a.n, a.n, dense_product(dense(a), dense(b), a.n, a.n, a.n)));
}

// Row map[i] of the restriction holds a one in column i; unassigned entries
// are negative. The prolongation is its transpose.
TYPED_TEST(matrix_csr, create_from_map)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(TypeParam::backend, rocalution::CSR, "CreateFromMap");

    const int m = 3;

    auto restriction_of = [&](const std::vector<int>& map) {
        Csr<T> restriction(m, static_cast<int>(map.size()));
        for(int r = 0; r < m; ++r)
        {
            for(size_t i = 0; i < map.size(); ++i)
            {
                if(map[i] == r)
                {
                    restriction.add(static_cast<int>(i), T(1));
                }
            }
            restriction.end_row();
        }
        return restriction;
    };

    auto expect_created = [&](const std::vector<int>& map) {
        const int    n           = static_cast<int>(map.size());
        const Csr<T> restriction = restriction_of(map);
        auto         v           = make_vector(TypeParam::backend, map);

        auto R = this->empty();
        ASSERT_TRUE(R->CreateFromMap(*v, n, m));
        expect_same(csr_of(*R), restriction);

        auto R2 = this->empty();
        auto P  = this->empty();
        ASSERT_TRUE(R2->CreateFromMap(*v, n, m, P.get()));
        expect_same(csr_of(*R2), restriction);
        expect_same(csr_of(*P), transposed(restriction));
    };

    {
        SCOPED_TRACE("every node mapped");
        expect_created({0, 2, 1, 1, 0, 2, 1, 2, 0});
    }

    SCOPED_TRACE("unassigned nodes");
    const std::vector<int> map         = {0, 2, -1, 1, 0, 2, 1, -1, 0};
    const int              n           = static_cast<int>(map.size());
    const Csr<T>           restriction = restriction_of(map);

    // The rows are counted over all nodes, so the sizes and the row offsets are right
    auto R = this->empty();
    ASSERT_TRUE(R->CreateFromMap(*make_vector(TypeParam::backend, map), n, m));
    const Csr<T> created = csr_of(*R);
    EXPECT_EQ(created.n, m);
    EXPECT_EQ(created.m, n);
    EXPECT_EQ(created.row_offset, restriction.row_offset);

    KNOWN_BUG("with unassigned nodes the fill loop stops at nnz and indexes by node");

    expect_same(created, restriction);
    expect_created(map);
}

// Into an empty matrix and into one of the same size
TYPED_TEST(matrix_csr, copies_between_matrices_of_the_backend)
{
    using T            = typename TypeParam::value_type;
    const Csr<T> ref   = general<T>(8);
    const Csr<T> other = transformed(ref, [](int, int, T v) { return -v; });

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);

    using Copy = std::function<void(BaseMatrix<T>&)>;
    const std::vector<std::pair<std::string, Copy>> copies
        = {{"CopyFrom", [&](BaseMatrix<T>& B) { B.CopyFrom(*A); }},
           {"CopyFromAsync", [&](BaseMatrix<T>& B) { B.CopyFromAsync(*A); }},
           {"CopyTo", [&](BaseMatrix<T>& B) { A->CopyTo(&B); }},
           {"CopyToAsync", [&](BaseMatrix<T>& B) { A->CopyToAsync(&B); }}};

    for(const auto& copy : copies)
    {
        SCOPED_TRACE(copy.first);

        auto B = this->empty();
        copy.second(*B);
        rocalution::_rocalution_sync();
        expect_same(csr_of(*B), ref);

        auto C = this->make(other);
        ASSERT_IN_LOOP(C != nullptr);
        copy.second(*C);
        rocalution::_rocalution_sync();
        expect_same(csr_of(*C), ref);
    }
}

// The arrays of a backend matrix are left to the caller, copied and handed back
TYPED_TEST(matrix_csr, leave_copy_and_set_the_backend_arrays)
{
    using T              = typename TypeParam::value_type;
    const Csr<T> ref     = general<T>(8);
    const Csr<T> negated = transformed(ref, [](int, int, T v) { return -v; });

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);

    PtrType* row_offset = nullptr;
    int*     col        = nullptr;
    T*       val        = nullptr;
    A->LeaveDataPtrCSR(&row_offset, &col, &val);
    ASSERT_NE(row_offset, nullptr);
    ASSERT_NE(col, nullptr);
    ASSERT_NE(val, nullptr);
    EXPECT_EQ(A->GetM(), 0);
    EXPECT_EQ(A->GetN(), 0);
    EXPECT_EQ(A->GetNnz(), 0);

    auto B = this->empty();
    B->AllocateCSR(ref.nnz(), ref.n, ref.m);
    B->CopyFromCSR(row_offset, col, val);
    expect_same(csr_of(*B), ref);

    auto N = this->make(negated);
    ASSERT_NE(N, nullptr);
    N->CopyToCSR(row_offset, col, val);

    auto C = this->empty();
    C->SetDataPtrCSR(&row_offset, &col, &val, ref.nnz(), ref.n, ref.m);
    expect_same(csr_of(*C), negated);
}

TYPED_TEST(matrix_csr, copy_from_host_arrays)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = general<T>(8);

    auto A = this->empty();
    A->CopyFromHostCSR(
        ref.row_offset.data(), ref.col.data(), ref.val.data(), ref.nnz(), ref.n, ref.m);
    expect_same(csr_of(*A), ref);
}

TYPED_TEST(matrix_csr, key)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(TypeParam::backend, rocalution::CSR, "Key");

    const Csr<T> ref = general<T>(8);
    auto         A   = this->make(ref);
    ASSERT_NE(A, nullptr);

    long int row_key = 0, col_key = 0, val_key = 0;
    ASSERT_TRUE(A->Key(row_key, col_key, val_key));

    long int row_again = 0, col_again = 0, val_again = 0;
    ASSERT_TRUE(A->Key(row_again, col_again, val_again));
    EXPECT_EQ(row_again, row_key);
    EXPECT_EQ(col_again, col_key);
    EXPECT_EQ(val_again, val_key);

    ASSERT_TRUE(A->Scale(2.0));
    ASSERT_TRUE(A->Key(row_again, col_again, val_again));
    EXPECT_EQ(row_again, row_key);
    EXPECT_EQ(col_again, col_key);
    EXPECT_NE(val_again, val_key);

    auto At = this->make(transposed(ref));
    ASSERT_NE(At, nullptr);
    ASSERT_TRUE(At->Key(row_again, col_again, val_again));
    EXPECT_NE(col_again, col_key);
}

TYPED_TEST(matrix_csr, write_and_read_files)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = Csr<T>(banded_with_gaps<T>(9));

    for(const std::string format : {"CSR", "RSIO"})
    {
        SCOPED_TRACE(format);
        REQUIRE_NATIVE(TypeParam::backend, rocalution::CSR, "WriteFile" + format);
        REQUIRE_NATIVE(TypeParam::backend, rocalution::CSR, "ReadFile" + format);

        const std::string file = temp_file(format);
        const bool        csr  = format == "CSR";

        auto A = this->make(ref);
        ASSERT_IN_LOOP(A != nullptr);
        ASSERT_IN_LOOP(csr ? A->WriteFileCSR(file) : A->WriteFileRSIO(file));

        auto B = this->empty();
        EXPECT_TRUE(csr ? B->ReadFileCSR(file) : B->ReadFileRSIO(file));
        expect_same(csr_of(*B), ref);
        std::remove(file.c_str());
    }
}

// A missing or malformed file and an unwritable path return false
TYPED_TEST(matrix_csr, file_errors_return_false)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = Csr<T>(banded_with_gaps<T>(9));

    for(const std::string format : {"CSR", "RSIO"})
    {
        SCOPED_TRACE(format);
        REQUIRE_NATIVE(TypeParam::backend, rocalution::CSR, "WriteFile" + format);
        REQUIRE_NATIVE(TypeParam::backend, rocalution::CSR, "ReadFile" + format);

        const std::string file = temp_file(format);
        const bool        csr  = format == "CSR";

        auto A = this->make(ref);
        ASSERT_IN_LOOP(A != nullptr);
        auto B = this->empty();

        std::remove(file.c_str());
        EXPECT_FALSE(csr ? B->ReadFileCSR(file) : B->ReadFileRSIO(file)) << "missing file";

        const std::string unwritable = testing::TempDir() + "no/such/directory/file";
        EXPECT_FALSE(csr ? A->WriteFileCSR(unwritable) : A->WriteFileRSIO(unwritable))
            << "unwritable path";

        std::ofstream(file) << "not a matrix\n";
        EXPECT_FALSE(csr ? B->ReadFileCSR(file) : B->ReadFileRSIO(file)) << "malformed file";
        std::remove(file.c_str());
    }
}

// An operation that the backend does not implement returns false and changes
// neither the matrix nor its outputs, so that LocalMatrix can retry on the host
TYPED_TEST(matrix_csr, unimplemented_operations_return_false_and_change_nothing)
{
    using T                = typename TypeParam::value_type;
    const Csr<T>      ref  = general<T>(8);
    const std::string file = temp_file("matrix");
    ASSERT_TRUE(ref.host_csr()->WriteFileCSR(file + ".CSR"));
    ASSERT_TRUE(ref.host_csr()->WriteFileRSIO(file + ".RSIO"));

    auto other = this->make(ref);
    ASSERT_NE(other, nullptr);

    auto untouched_perm = [&](const std::function<bool(BaseVector<int>*)>& f) {
        auto perm   = make_vector(TypeParam::backend, std::vector<int>{-7});
        bool native = f(perm.get());
        EXPECT_EQ(values_of(*perm), std::vector<int>{-7});
        return native;
    };
    auto no_file = [](const std::string& name, bool native) {
        EXPECT_FALSE(std::ifstream(name).good()) << name << " was written";
        std::remove(name.c_str());
        return native;
    };

    using Operation = std::function<bool(BaseMatrix<T>&)>;
    const std::vector<std::pair<std::string, Operation>> operations = {
        {"Gershgorin",
         [](BaseMatrix<T>& A) {
             T    lambda_min = -7, lambda_max = -7;
             bool native = A.Gershgorin(lambda_min, lambda_max);
             EXPECT_EQ(lambda_min, -7);
             EXPECT_EQ(lambda_max, -7);
             return native;
         }},
        {"ReplaceRowVector",
         [&](BaseMatrix<T>& A) {
             return A.ReplaceRowVector(
                 3, *make_vector(TypeParam::backend, std::vector<T>(ref.n, T(1))));
         }},
        {"ILUTFactorize", [&](BaseMatrix<T>& A) { return A.ILUTFactorize(0.0, ref.n); }},
        {"ILUpFactorizeNumeric",
         [&](BaseMatrix<T>& A) { return A.ILUpFactorizeNumeric(1, *other); }},
        {"CMK",
         [&](BaseMatrix<T>& A) {
             return untouched_perm([&](BaseVector<int>* p) { return A.CMK(p); });
         }},
        {"RCMK",
         [&](BaseMatrix<T>& A) {
             return untouched_perm([&](BaseVector<int>* p) { return A.RCMK(p); });
         }},
        {"ConnectivityOrder",
         [&](BaseMatrix<T>& A) {
             return untouched_perm([&](BaseVector<int>* p) { return A.ConnectivityOrder(p); });
         }},
        {"ZeroBlockPermutation",
         [&](BaseMatrix<T>& A) {
             int  size   = -7;
             bool native = untouched_perm(
                 [&](BaseVector<int>* p) { return A.ZeroBlockPermutation(size, p); });
             EXPECT_EQ(size, -7);
             return native;
         }},
        {"SymbolicPower", [](BaseMatrix<T>& A) { return A.SymbolicPower(2); }},
        {"SymbolicMatMatMult",
         [&](BaseMatrix<T>& A) { return A.SymbolicMatMatMult(*other, *other); }},
        {"CreateFromMap",
         [&](BaseMatrix<T>& A) {
             return A.CreateFromMap(
                 *make_vector(TypeParam::backend, std::vector<int>(ref.n, 0)), ref.n, 1);
         }},
        {"Key",
         [](BaseMatrix<T>& A) {
             long int row_key = -7, col_key = -7, val_key = -7;
             bool     native = A.Key(row_key, col_key, val_key);
             EXPECT_EQ(row_key, -7);
             EXPECT_EQ(col_key, -7);
             EXPECT_EQ(val_key, -7);
             return native;
         }},
        {"WriteFileCSR",
         [&](BaseMatrix<T>& A) { return no_file(file + ".out", A.WriteFileCSR(file + ".out")); }},
        {"WriteFileRSIO",
         [&](BaseMatrix<T>& A) { return no_file(file + ".out", A.WriteFileRSIO(file + ".out")); }},
        {"ReadFileCSR", [&](BaseMatrix<T>& A) { return A.ReadFileCSR(file + ".CSR"); }},
        {"ReadFileRSIO", [&](BaseMatrix<T>& A) { return A.ReadFileRSIO(file + ".RSIO"); }},
    };

    int checked = 0;
    for(const auto& operation : operations)
    {
        if(is_native(TypeParam::backend, operation.first))
        {
            continue;
        }
        SCOPED_TRACE(operation.first);

        auto A = this->make(ref);
        ASSERT_IN_LOOP(A != nullptr);
        EXPECT_FALSE(operation.second(*A));
        expect_same(csr_of(*A), ref);
        ++checked;
    }

    std::remove((file + ".CSR").c_str());
    std::remove((file + ".RSIO").c_str());
    if(checked == 0)
    {
        GTEST_SKIP() << "every operation is implemented";
    }
}

std::string itilu0_algorithm_name(rocalution::ItILU0Algorithm alg)
{
    static const char* names[]
        = {"Default", "AsyncInPlace", "AsyncSplit", "SyncSplit", "SyncSplitFusion"};
    return names[alg];
}

// The iterative ILU(0) converges to the ILU(0). The host computes the ILU(0)
// directly and ignores the algorithm.
class matrix_csr_itilu0
    : public testing::TestWithParam<std::tuple<Backend, rocalution::ItILU0Algorithm>>
{
protected:
    void SetUp() override
    {
        REQUIRE_BACKEND(std::get<0>(GetParam()));
    }
};

TEST_P(matrix_csr_itilu0, converges_to_the_ilu0)
{
    auto checks = [&] {
        for_each_type<RealTypes>([&](auto type) {
            using T          = typename decltype(type)::type;
            const Csr<T> ref = dominant<T>(8);
            auto         A
                = make_matrix(std::get<0>(GetParam()), Format{rocalution::CSR, 1}, *ref.host_csr());
            ASSERT_NE(A, nullptr);

            int niter = -1;
            ASSERT_TRUE(
                A->ItILU0Factorize(std::get<1>(GetParam()), 0, 200, 1e-14, &niter, nullptr));
            expect_near(csr_of(*A), ilu0(ref), iterative_tol<T>());
        });
    };

    if(std::get<0>(GetParam()) == Backend::Accelerator
       && std::get<1>(GetParam()) == rocalution::AsyncSplit)
    {
        KNOWN_BUG("rocsparse_csritilu0_compute with async_split and no options does not return "
                  "for 20 or more iterations");
        expect_in_child(checks, 60);
        return;
    }
    checks();
}

// A convergence history needs the stopping criteria. The options are those of
// the rocsparse_csritilu0_compute example. rocSPARSE stores the correction norms
// in history[0, niter) and the residual norms in history[niter, 2 niter).
TEST_P(matrix_csr_itilu0, records_the_convergence_history)
{
    for_each_type<RealTypes>([&](auto type) {
        using T          = typename decltype(type)::type;
        const Csr<T> ref = dominant<T>(8);
        auto A = make_matrix(std::get<0>(GetParam()), Format{rocalution::CSR, 1}, *ref.host_csr());
        ASSERT_NE(A, nullptr);

        const int option = rocalution::StoppingCriteria | rocalution::ComputeNrmResidual
                           | rocalution::ConvergenceHistory;
        const int    max_iter  = 200;
        const double untouched = -1.0;

        int                 niter = -1;
        std::vector<double> history(2 * max_iter, untouched);
        ASSERT_TRUE(A->ItILU0Factorize(
            std::get<1>(GetParam()), option, max_iter, 1e-14, &niter, history.data()));
        expect_near(csr_of(*A), ilu0(ref), iterative_tol<T>());

        // The host computes the ILU(0) directly, so there are no iterations to report
        if(std::get<0>(GetParam()) == Backend::Host)
        {
            EXPECT_EQ(niter, -1);
            EXPECT_EQ(history, std::vector<double>(2 * max_iter, untouched));
            return;
        }

        ASSERT_GE(niter, 1);
        ASSERT_LE(niter, max_iter);

        for(int i = niter; i < 2 * niter; ++i)
        {
            EXPECT_TRUE(std::isfinite(history[i])) << "residual norm " << i - niter;
            EXPECT_GE(history[i], 0.0) << "residual norm " << i - niter;
        }
        EXPECT_GT(history[niter], 0.0) << "the first residual norm";
        EXPECT_LE(history[2 * niter - 1], history[niter]) << "the last residual norm";

        for(int i = 2 * niter; i < 2 * max_iter; ++i)
        {
            EXPECT_EQ(history[i], untouched) << "at " << i;
        }
    });
}

INSTANTIATE_TEST_SUITE_P(
    backends,
    matrix_csr_itilu0,
    testing::Combine(testing::Values(Backend::Host, Backend::Accelerator),
                     testing::Values(rocalution::Default,
                                     rocalution::AsyncInPlace,
                                     rocalution::AsyncSplit,
                                     rocalution::SyncSplit,
                                     rocalution::SyncSplitFusion)),
    [](const testing::TestParamInfo<std::tuple<Backend, rocalution::ItILU0Algorithm>>& info) {
        return backend_name(std::get<0>(info.param)) + "_"
               + itilu0_algorithm_name(std::get<1>(info.param));
    });

namespace
{
    template <typename P>
    class host_matrix_csr : public TypedBackendTest<P>
    {
    };
}

// HostMatrixCSR itself: Check, which only the host implements, and host paths
// that the accelerator does not share
TYPED_TEST_SUITE(host_matrix_csr, HostOnly<RealTypes>, TypedName);

TYPED_TEST(host_matrix_csr, check_accepts_valid_and_unsorted_matrices)
{
    using T    = typename TypeParam::value_type;
    Csr<T> ref = general<T>(8);
    EXPECT_TRUE(ref.host_csr()->Check());

    std::swap(ref.col[0], ref.col[1]);
    std::swap(ref.val[0], ref.val[1]);
    EXPECT_TRUE(ref.host_csr()->Check());
}

TYPED_TEST(host_matrix_csr, check_rejects_invalid_column_indices)
{
    using T          = typename TypeParam::value_type;
    Csr<T> duplicate = general<T>(8);
    duplicate.col[1] = duplicate.col[0];
    EXPECT_FALSE(duplicate.host_csr()->Check());

    Csr<T> negative = general<T>(8);
    negative.col[0] = -1;
    EXPECT_FALSE(negative.host_csr()->Check());

    KNOWN_BUG("a column index equal to the number of columns passes");

    Csr<T> outside                      = general<T>(8);
    outside.col[outside.col.size() - 1] = outside.m;
    EXPECT_FALSE(outside.host_csr()->Check());
}

TYPED_TEST(host_matrix_csr, check_rejects_invalid_row_offsets_and_values)
{
    using T = typename TypeParam::value_type;

    Csr<T> negative_offset        = general<T>(8);
    negative_offset.row_offset[1] = -1;
    EXPECT_FALSE(negative_offset.host_csr()->Check());

    Csr<T> offset_past_the_end        = general<T>(8);
    offset_past_the_end.row_offset[1] = offset_past_the_end.nnz() + 1;
    EXPECT_FALSE(offset_past_the_end.host_csr()->Check());

    for(T bad : {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity()})
    {
        Csr<T> A = general<T>(8);
        A.val[3] = bad;
        EXPECT_FALSE(A.host_csr()->Check()) << bad;
    }

    KNOWN_BUG("Check compares the values with +infinity only");

    Csr<T> A = general<T>(8);
    A.val[3] = -std::numeric_limits<T>::infinity();
    EXPECT_FALSE(A.host_csr()->Check());
}

TYPED_TEST(host_matrix_csr, check_accepts_matrices_without_entries)
{
    using T = typename TypeParam::value_type;

    for(const auto& size : std::vector<std::pair<int, int>>{{0, 0}, {4, 4}, {0, 3}, {3, 0}})
    {
        SCOPED_TRACE(std::to_string(size.first) + " x " + std::to_string(size.second));
        HostMatrixCSR<T> A(descriptor());
        A.AllocateCSR(0, size.first, size.second);
        EXPECT_TRUE(A.Check());
    }

    HostMatrixCSR<T> unallocated(descriptor());
    EXPECT_TRUE(unallocated.Check());
}

TYPED_TEST(host_matrix_csr, info_names_the_class)
{
    using T = typename TypeParam::value_type;
    HostMatrixCSR<T> A(descriptor());
    EXPECT_NE(logged([&] { A.Info(); }).find("HostMatrixCSR"), std::string::npos);
}

TYPED_TEST(host_matrix_csr, copies_of_matrices_without_entries)
{
    using T = typename TypeParam::value_type;

    HostMatrixCSR<T> A(descriptor());
    A.AllocateCSR(0, 4, 4);
    std::vector<PtrType> row_offset(5, -1);
    A.CopyToCSR(row_offset.data(), nullptr, nullptr);
    EXPECT_EQ(row_offset, std::vector<PtrType>(5, 0));

    HostMatrixCSR<T> unallocated(descriptor());
    HostMatrixCSR<T> B(descriptor());
    B.CopyFrom(unallocated);
    EXPECT_EQ(B.GetM(), 0);
    EXPECT_EQ(B.GetN(), 0);
    EXPECT_EQ(B.GetNnz(), 0);
    EXPECT_TRUE(B.Check());
}

// The solves of a matrix without entries need no analysis and leave their output
TYPED_TEST(host_matrix_csr, operations_on_a_matrix_without_entries_change_nothing)
{
    using T                   = typename TypeParam::value_type;
    const int            n    = 4;
    const std::vector<T> kept = {7, -2, 5, 3};

    HostMatrixCSR<T> A(descriptor());
    A.AllocateCSR(0, n, n);
    auto in   = host_vector(std::vector<T>(n, T(1)));
    auto out  = host_vector(kept);
    auto perm = host_vector(std::vector<int>{3, 1, 0, 2});

    A.ApplyAdd(*in, T(2), out.get());
    EXPECT_TRUE(A.ItLSolve(10, 0.0, false, *in, out.get()));
    EXPECT_TRUE(A.ItUSolve(10, 0.0, false, *in, out.get()));
    EXPECT_TRUE(A.ItLUSolve(10, 0.0, false, *in, out.get()));
    EXPECT_TRUE(A.ItLLSolve(10, 0.0, false, *in, out.get()));
    EXPECT_EQ(values_of(*out), kept);

    EXPECT_TRUE(A.Compress(0.5));
    EXPECT_TRUE(A.Transpose());
    EXPECT_TRUE(A.Sort());
    EXPECT_TRUE(A.Permute(*perm));
    expect_same(csr_of(A), without_entries<T>(n, n));
}

TYPED_TEST(host_matrix_csr, transpose_of_a_matrix_without_entries_swaps_the_sizes)
{
    using T = typename TypeParam::value_type;

    KNOWN_BUG("the transpose of a matrix without entries keeps the sizes");

    HostMatrixCSR<T> A(descriptor());
    A.AllocateCSR(0, 3, 5);
    HostMatrixCSR<T> At(descriptor());
    ASSERT_TRUE(A.Transpose(&At));
    expect_same(csr_of(At), without_entries<T>(5, 3));

    ASSERT_TRUE(A.Transpose());
    expect_same(csr_of(A), without_entries<T>(5, 3));
}

// The rows and columns of a matrix without entries are zero
TYPED_TEST(host_matrix_csr, rows_and_columns_of_a_matrix_without_entries)
{
    using T     = typename TypeParam::value_type;
    const int n = 4;

    KNOWN_BUG("rows and columns of a matrix without entries are neither extracted nor replaced");

    HostMatrixCSR<T> A(descriptor());
    A.AllocateCSR(0, n, n);
    auto v = host_vector(std::vector<T>(n, T(7)));
    ASSERT_TRUE(A.ExtractColumnVector(1, v.get()));
    EXPECT_EQ(values_of(*v), std::vector<T>(n, T(0)));

    v = host_vector(std::vector<T>(n, T(7)));
    ASSERT_TRUE(A.ExtractRowVector(1, v.get()));
    EXPECT_EQ(values_of(*v), std::vector<T>(n, T(0)));

    const std::vector<T> values = {1, 0, 2, 3};
    ASSERT_TRUE(A.ReplaceColumnVector(1, *host_vector(values)));
    std::vector<T> D(n * n, T(0));
    for(int i = 0; i < n; ++i)
    {
        D[i * n + 1] = values[i];
    }
    expect_same(csr_of(A), from_dense(n, n, D));

    HostMatrixCSR<T> B(descriptor());
    B.AllocateCSR(0, n, n);
    ASSERT_TRUE(B.ReplaceRowVector(2, *host_vector(values)));
    std::fill(D.begin(), D.end(), T(0));
    std::copy(values.begin(), values.end(), D.begin() + 2 * n);
    expect_same(csr_of(B), from_dense(n, n, D));
}

// The iterative analyses share one buffer. A unit diagonal needs less of it, so an
// analysis after one with a unit diagonal grows the buffer, and one after an
// analysis with the same need reuses it.
// The iterative L, U, LU and LL analyses share one buffer, which is not visible from
// outside. Interleaved analyses and clears must keep every solve right.
TYPED_TEST(host_matrix_csr, iterative_analyses_in_any_order_keep_their_solves)
{
    using T                  = typename TypeParam::value_type;
    const int            it  = 100;
    const Csr<T>         ref = dominant<T>(8);
    const std::vector<T> b   = rhs<T>(ref.n);
    const std::vector<T> D   = dense(ref);
    const std::vector<T> l   = lower_solve(D, ref.n, b, false);
    const std::vector<T> u   = upper_solve(D, ref.n, b, false);
    const std::vector<T> lu  = upper_solve(D, ref.n, lower_solve(D, ref.n, b, true), false);

    auto A  = ref.host_csr();
    auto in = host_vector(b);

    auto expect_solves = [&](bool lower, const std::vector<T>& expected) {
        auto out = host_vector(std::vector<T>(ref.n, T(0)));
        ASSERT_TRUE(lower ? A->ItLSolve(it, 0.0, false, *in, out.get())
                          : A->ItUSolve(it, 0.0, false, *in, out.get()));
        expect_near(values_of(*out), expected, iterative_tol<T>());
    };

    {
        SCOPED_TRACE("U after L");
        A->ItLAnalyse(true);
        A->ItUAnalyse(false);
        A->ItLAnalyse(false);
        expect_solves(true, l);
        expect_solves(false, u);
        A->ItLAnalyseClear();
    }
    {
        SCOPED_TRACE("L after U");
        A->ItUAnalyse(true);
        A->ItLAnalyse(false);
        A->ItUAnalyse(false);
        expect_solves(true, l);
        expect_solves(false, u);
        A->ItUAnalyseClear();
    }
    for(bool unit : {true, false})
    {
        SCOPED_TRACE(unit ? "LU after a unit L" : "LU after L");
        A->ItLAnalyse(unit);
        A->ItLUAnalyse();
        auto out = host_vector(std::vector<T>(ref.n, T(0)));
        ASSERT_IN_LOOP(A->ItLUSolve(it, 0.0, false, *in, out.get()));
        expect_near(values_of(*out), lu, iterative_tol<T>());
        A->ItLUAnalyseClear();
    }

    const Csr<T>   L  = filtered(spd<T>(8), [](int i, int j, T) { return j <= i; });
    std::vector<T> DL = dense(L);
    std::vector<T> Lt(DL.size());
    for(int i = 0; i < L.n; ++i)
    {
        for(int j = 0; j < L.n; ++j)
        {
            Lt[j * L.n + i] = DL[i * L.n + j];
        }
    }
    const std::vector<T> ll = upper_solve(Lt, L.n, lower_solve(DL, L.n, b, false), false);

    auto M = L.host_csr();
    for(bool unit : {true, false})
    {
        SCOPED_TRACE(unit ? "LL after a unit L" : "LL after L");
        M->ItLAnalyse(unit);
        M->ItLLAnalyse();
        auto out = host_vector(std::vector<T>(L.n, T(0)));
        ASSERT_IN_LOOP(M->ItLLSolve(it, 0.0, false, *in, out.get()));
        expect_near(values_of(*out), ll, iterative_tol<T>());
        M->ItLLAnalyseClear();
    }
}

// A zero pivot leaves the entries below it, and a last row without an upper part
// is only eliminated
TYPED_TEST(host_matrix_csr, ilu0_factorize_skips_zero_pivots)
{
    using T          = typename TypeParam::value_type;
    const int    n   = 8;
    const Csr<T> ref = filtered(
        transformed(dominant<T>(n), [](int i, int j, T v) { return i == 0 && j == 0 ? T(0) : v; }),
        [&](int i, int j, T) { return i != n - 1 || j < i; });
    ASSERT_EQ(ref.row_offset[n] - ref.row_offset[n - 1], 1);

    auto A = ref.host_csr();
    ASSERT_TRUE(A->ILU0Factorize());
    expect_near(csr_of(*A), ilu0_skipping_zero_pivots(ref), tol<T>());
}

// Row i of the prolongation holds a one in column aggregates[i], none for a negative
// entry; the columns are the aggregates up to the largest
TYPED_TEST(host_matrix_csr, unsmoothed_aggregation_from_an_aggregate_vector)
{
    using T                               = typename TypeParam::value_type;
    const std::vector<int64_t> aggregates = {0, 2, -1, 1, 0, 2, 1, -1};
    const int                  n          = static_cast<int>(aggregates.size());

    Csr<T> expected(n, 3);
    for(int i = 0; i < n; ++i)
    {
        if(aggregates[i] >= 0)
        {
            expected.add(static_cast<int>(aggregates[i]), T(1));
        }
        expected.end_row();
    }

    auto             A = general<T>(n).host_csr();
    HostMatrixCSR<T> P(descriptor());
    ASSERT_TRUE(A->AMGUnsmoothedAggregation(*host_vector(aggregates), &P));
    expect_same(csr_of(P), expected);
}

TYPED_TEST(host_matrix_csr, rs_interpolation_truncation_without_limits_keeps_the_matrix)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = general<T>(8);

    for(const auto& limits : std::vector<std::pair<float, int>>{{0.0f, 0}, {-1.0f, -1}})
    {
        auto A = ref.host_csr();
        ASSERT_IN_LOOP(A->RSInterpolationTruncation(limits.first, limits.second));
        expect_same(csr_of(*A), ref);
    }
}

// Row i depends only on row i + 1, so the first pass decides the chain from its
// end and leaves undecided points that nothing depends on. The last row has no
// negative coupling and is independent of the others.
TYPED_TEST(host_matrix_csr, rs_coarsening_of_a_directed_chain)
{
    using T     = typename TypeParam::value_type;
    const int n = 9;

    Csr<T> ref(n);
    for(int i = 0; i < n; ++i)
    {
        if(i == n - 1)
        {
            ref.add(0, T(1));
        }
        ref.add(i, T(2));
        if(i < n - 1)
        {
            ref.add(i + 1, T(-1));
        }
        ref.end_row();
    }

    auto             A = ref.host_csr();
    HostVector<int>  cf(descriptor());
    HostVector<bool> S(descriptor());
    ASSERT_TRUE(A->RSCoarsening(0.25f, &cf, &S));

    const std::vector<bool> strong = rs_strength(ref, 0.25);
    EXPECT_EQ(values_of(S), strong);

    const std::vector<int> c = values_of(cf);
    ASSERT_EQ(c.size(), static_cast<size_t>(n));
    EXPECT_EQ(c[n - 1], 2);
    for(int i = 0; i < n; ++i)
    {
        ASSERT_TRUE(c[i] == 1 || c[i] == 2) << "node " << i << ": " << c[i];
        if(c[i] == 2 && i < n - 1)
        {
            EXPECT_EQ(c[i + 1], 1) << "F point " << i << " has no C point";
        }
    }
}

// IC(0) needs the diagonal of every row. The search for the diagonal of row 1
// stops at its first entry above the diagonal, or, without one, at the end of
// the row.
TEST(host_matrix_csr_DeathTest, ic_factorize_of_a_row_without_a_diagonal_ends_the_program)
{
    for(double upper : {-1.0, 0.0})
    {
        SCOPED_TRACE(upper != 0.0 ? "an entry above the diagonal" : "only an entry below it");
        // clang-format off
        const std::vector<double> D = {
             4, -1,     0,
            -1,  0, upper,
             0, upper,  4,
        };
        // clang-format on
        auto A        = from_dense(3, 3, D).host_csr();
        auto inv_diag = host_vector(std::vector<double>(3, 0.0));
        EXPECT_FATAL_ERROR(A->ICFactorize(inv_diag.get()), "host_matrix_csr\\.cpp");
    }
}
