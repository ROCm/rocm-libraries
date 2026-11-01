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

// The AMG setup of LocalMatrix: aggregation, coarsening, interpolation and truncation.
// These are multi-step protocols over the backend matrices, so they are tested through
// the LocalMatrix calls that drive them, on small grids with checked properties. The
// single steps are tested in unit_test_backend_amg.cpp.
//
// Aggregation and coarsening are deterministic (the random measures are hashes of the node
// index), and both backends compute the same result on the matrices here, which
// unit_test_multigrid.cpp relies on for whole hierarchies. Here each backend is checked for
// a valid result, and the prolongations are compared between the backends for the same
// aggregates and C/F splitting, computed once on the host, so that a difference in the
// prolongation is not hidden behind one in its inputs.

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "unit_test_amg_reference.hpp"
#include "unit_test_common.hpp"
#include "unit_test_local.hpp"
#include "unit_test_reference.hpp"
#include "unit_test_types.hpp"

#include "base/local_matrix.hpp"
#include "base/local_vector.hpp"
#include "utils/allocate_free.hpp"

using namespace rocalution_unit;
using rocalution::LocalMatrix;
using rocalution::LocalVector;

namespace
{
    // P(i, agg[i]) = 1, an empty row for unaggregated nodes
    template <typename T>
    Csr<T> tentative_prolongation(const std::vector<int64_t>& agg)
    {
        const int n  = static_cast<int>(agg.size());
        const int nc = static_cast<int>(*std::max_element(agg.begin(), agg.end()) + 1);
        Csr<T>    P(n, nc);
        for(int i = 0; i < n; ++i)
        {
            if(agg[i] >= 0)
            {
                P.add(static_cast<int>(agg[i]), 1.0);
            }
            P.end_row();
        }
        return P;
    }

    // Truncation keeps the entries of a row with |p_ij| >= trunc_factor max_k |p_ik|, and
    // of those at most max_elmts of the largest magnitude. The kept entries are scaled by
    // one factor, so that the row sum is preserved.
    template <typename T>
    void expect_valid_truncation(const Csr<T>& P,
                                 const Csr<T>& truncated,
                                 float         trunc_factor,
                                 int           max_elmts)
    {
        ASSERT_EQ(truncated.n, P.n);
        ASSERT_EQ(truncated.m, P.m);
        for(int i = 0; i < P.n; ++i)
        {
            std::map<int, T> original;
            T                largest = 0;
            T                row_sum = 0;
            for(PtrType j = P.row_offset[i]; j < P.row_offset[i + 1]; ++j)
            {
                original[P.col[j]] = P.val[j];
                largest            = std::max(largest, std::abs(P.val[j]));
                row_sum += P.val[j];
            }
            const T threshold = trunc_factor > 0.0f ? static_cast<T>(trunc_factor) * largest : T(0);

            std::map<int, T> kept;
            int              previous = -1;
            for(PtrType j = truncated.row_offset[i]; j < truncated.row_offset[i + 1]; ++j)
            {
                EXPECT_GT(truncated.col[j], previous) << "row " << i << " is not sorted";
                previous = truncated.col[j];
                ASSERT_EQ(original.count(truncated.col[j]), 1u)
                    << "row " << i << " gained column " << truncated.col[j];
                kept[truncated.col[j]] = truncated.val[j];
            }
            if(max_elmts > 0)
            {
                EXPECT_LE(kept.size(), static_cast<size_t>(max_elmts)) << "row " << i;
            }

            T smallest_kept = std::numeric_limits<T>::infinity();
            T kept_sum      = 0;
            for(const auto& e : kept)
            {
                const T v = original[e.first];
                EXPECT_GE(std::abs(v), threshold) << "row " << i << ", column " << e.first;
                smallest_kept = std::min(smallest_kept, std::abs(v));
                kept_sum += v;
            }

            const bool full = max_elmts > 0 && kept.size() == static_cast<size_t>(max_elmts);
            for(const auto& e : original)
            {
                if(kept.count(e.first) == 0)
                {
                    EXPECT_TRUE(std::abs(e.second) < threshold
                                || (full && std::abs(e.second) <= smallest_kept))
                        << "row " << i << " dropped column " << e.first << " with " << e.second;
                }
            }

            if(kept_sum != T(0))
            {
                const T factor = row_sum / kept_sum;
                T       sum    = 0;
                for(const auto& e : kept)
                {
                    const T expected = original[e.first] * factor;
                    EXPECT_TRUE(is_near(e.second, expected, scaled_tol<T>(1e-13)))
                        << "row " << i << ", column " << e.first;
                    sum += e.second;
                }
                EXPECT_TRUE(is_near(sum, row_sum, scaled_tol<T>(1e-13))) << "row " << i;
            }
        }
    }

    template <typename P>
    class local_matrix_amg : public TypedBackendTest<P>
    {
    protected:
        using T = typename P::value_type;

        std::unique_ptr<LocalMatrix<T>> make(const Csr<T>& ref) const
        {
            return make_local_matrix(P::backend, ref);
        }

        template <typename V>
        std::unique_ptr<LocalVector<V>> vector_like(const LocalMatrix<T>& A) const
        {
            auto v = std::make_unique<LocalVector<V>>();
            v->CloneBackend(A);
            return v;
        }
    };

    // Strong in x, weak in y for the SA threshold 0.08
    template <typename T>
    Csr<T> anisotropic()
    {
        return grid<T>(8, 6, 1.0, 0.01, true);
    }

    template <typename T>
    std::vector<std::pair<std::string, Csr<T>>> sa_grids()
    {
        return {{"isotropic", grid<T>(8, 8, 1.0, 1.0, true)}, {"anisotropic", anisotropic<T>()}};
    }
}

TYPED_TEST_SUITE(local_matrix_amg, WithBackends<RealTypes>, TypedName);

TYPED_TEST(local_matrix_amg, pmis_aggregation)
{
    using T = typename TypeParam::value_type;
    for(const auto& g : sa_grids<T>())
    {
        SCOPED_TRACE(g.first);
        const Csr<T>& ref = g.second;

        auto A           = this->make(ref);
        auto connections = this->template vector_like<bool>(*A);
        auto aggregates  = this->template vector_like<int64_t>(*A);
        auto roots       = this->template vector_like<int64_t>(*A);
        A->AMGPMISAggregate(0.08, connections.get(), aggregates.get(), roots.get());

        const std::vector<bool> strong = sa_strength(ref, 0.08);
        EXPECT_EQ(values_of(*connections), strong);

        const std::vector<int64_t> rt = values_of(*roots);
        expect_valid_aggregation(ref, strong, values_of(*aggregates), rt);
        expect_independent_roots(ref, strong, rt);
    }
}

TYPED_TEST(local_matrix_amg, greedy_aggregation)
{
    using T = typename TypeParam::value_type;
    for(const auto& g : sa_grids<T>())
    {
        SCOPED_TRACE(g.first);
        const Csr<T>& ref = g.second;

        auto A           = this->make(ref);
        auto connections = this->template vector_like<bool>(*A);
        auto aggregates  = this->template vector_like<int64_t>(*A);
        auto roots       = this->template vector_like<int64_t>(*A);
        A->AMGGreedyAggregate(0.08, connections.get(), aggregates.get(), roots.get());

        const std::vector<bool> strong = sa_strength(ref, 0.08);
        EXPECT_EQ(values_of(*connections), strong);
        expect_valid_aggregation(ref, strong, values_of(*aggregates), values_of(*roots));
    }
}

TYPED_TEST(local_matrix_amg, unsmoothed_aggregation)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T> ref         = anisotropic<T>();
    auto         A           = this->make(ref);
    auto         connections = this->template vector_like<bool>(*A);
    auto         aggregates  = this->template vector_like<int64_t>(*A);
    auto         roots       = this->template vector_like<int64_t>(*A);
    A->AMGPMISAggregate(0.08, connections.get(), aggregates.get(), roots.get());

    LocalMatrix<T> P;
    P.CloneBackend(*A);
    A->AMGUnsmoothedAggregation(*aggregates, *roots, &P);

    expect_near(csr_of(P), tentative_prolongation<T>(values_of(*aggregates)), 0.0);
}

TYPED_TEST(local_matrix_amg, smoothed_aggregation)
{
    using T            = typename TypeParam::value_type;
    const Csr<T> ref   = anisotropic<T>();
    const double relax = 2.0 / 3.0;

    for(int lumping : {0, 1})
    {
        SCOPED_TRACE("lumping " + std::to_string(lumping));

        auto A           = this->make(ref);
        auto connections = this->template vector_like<bool>(*A);
        auto aggregates  = this->template vector_like<int64_t>(*A);
        auto roots       = this->template vector_like<int64_t>(*A);
        A->AMGPMISAggregate(0.08, connections.get(), aggregates.get(), roots.get());

        LocalMatrix<T> P;
        P.CloneBackend(*A);
        A->AMGSmoothedAggregation(relax, *connections, *aggregates, *roots, &P, lumping);

        expect_near(csr_of(P),
                    smoothed_prolongation(
                        ref, values_of(*connections), values_of(*aggregates), relax, lumping),
                    scaled_tol<T>(1e-14));
    }
}

TYPED_TEST(local_matrix_amg, rs_coarsening)
{
    using T          = typename TypeParam::value_type;
    using Coarsening = void (LocalMatrix<T>::*)(float, LocalVector<int>*, LocalVector<bool>*) const;
    const std::vector<std::pair<std::string, Coarsening>> coarsenings
        = {{"RSCoarsening", &LocalMatrix<T>::RSCoarsening},
           {"RSPMISCoarsening", &LocalMatrix<T>::RSPMISCoarsening}};

    for(const auto& coarsening : coarsenings)
    {
        for(const Csr<T>& ref : {grid<T>(8, 8, 1.0, 1.0, true), grid<T>(8, 6, 1.0, 0.1, true)})
        {
            SCOPED_TRACE(coarsening.first);

            auto A  = this->make(ref);
            auto cf = this->template vector_like<int>(*A);
            auto S  = this->template vector_like<bool>(*A);
            ((*A).*(coarsening.second))(0.25f, cf.get(), S.get());

            const std::vector<bool> strong = rs_strength(ref, 0.25);
            EXPECT_EQ(values_of(*S), strong);
            expect_valid_splitting(ref, strong, values_of(*cf));
        }
    }
}

namespace
{
    template <typename T>
    Csr<T> from_rows(int m, const std::vector<std::vector<std::pair<int, double>>>& rows)
    {
        Csr<T> A(static_cast<int>(rows.size()), m);
        for(const auto& row : rows)
        {
            for(const auto& e : row)
            {
                A.add(e.first, static_cast<T>(e.second));
            }
            A.end_row();
        }
        return A;
    }

    template <typename T>
    std::pair<std::vector<int>, std::vector<bool>> rs_coarsening_on(Backend b, const Csr<T>& ref)
    {
        auto              A = make_local_matrix(b, ref);
        LocalVector<int>  cf;
        LocalVector<bool> S;
        cf.CloneBackend(*A);
        S.CloneBackend(*A);
        A->RSCoarsening(0.25f, &cf, &S);
        return {values_of(cf), values_of(S)};
    }
}

// The first pass of Ruge-Stueben by lambda, the number of points that depend strongly on a
// point. The C point of the largest lambda makes its dependants F points; an undecided
// point that a new C point depends on loses one from lambda, and one that a new F point
// depends on gains one. When no undecided point with lambda > 0 is left, the rest are C.
//
// Node 0 has the dependants 4, 5, 6 and depends on 2; 1 has the dependants 7, 8 and
// depends on 2; 2 has the dependants 0, 1 and depends on 3, which depends on nothing (an F
// point from the start). After 0 becomes C, lambda_2 drops to 1, so 1 comes before 2 and
// becomes C; lambda_2 drops to 0 and 2 becomes C at the end. Without the update 2 would
// come first and make 1 an F point.
TYPED_TEST(local_matrix_amg, rs_coarsening_updates_lambda_of_the_points_a_c_point_depends_on)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = from_rows<T>(9,
                                    {{{0, 3.0}, {2, -1.0}, {4, -0.1}, {5, -0.1}, {6, -0.1}},
                                     {{1, 2.0}, {2, -1.0}, {7, -0.1}, {8, -0.1}},
                                     {{0, -0.1}, {1, -0.1}, {2, 2.0}, {3, -1.0}},
                                     {{2, 0.5}, {3, 1.0}},
                                     {{0, -1.0}, {4, 1.0}},
                                     {{0, -1.0}, {5, 1.0}},
                                     {{0, -1.0}, {6, 1.0}},
                                     {{1, -1.0}, {7, 1.0}},
                                     {{1, -1.0}, {8, 1.0}}});

    const auto result = rs_coarsening_on(TypeParam::backend, ref);
    EXPECT_EQ(result.second, rs_strength(ref, 0.25));
    EXPECT_EQ(result.first, (std::vector<int>{1, 1, 1, 2, 2, 2, 2, 2, 2}));
}

// lambda is at most n - 1, the bound of its buckets. In the complete graph of 3 nodes each
// has lambda 2. Every pair is strongly connected, so a valid splitting has exactly one C
// point; which one depends on the tie-breaking.
TYPED_TEST(local_matrix_amg, rs_coarsening_of_a_complete_graph)
{
    using T             = typename TypeParam::value_type;
    const Csr<T> ref    = band<T>(3, 2, false);
    const auto   result = rs_coarsening_on(TypeParam::backend, ref);
    EXPECT_EQ(result.second, off_diagonal(ref));
    expect_valid_splitting(ref, result.second, result.first);
    EXPECT_EQ(std::count(result.first.begin(), result.first.end(), 1), 1);
}

// Strength does not change when a row is scaled by -1
TYPED_TEST(local_matrix_amg, rs_coarsening_of_a_negated_grid)
{
    using T               = typename TypeParam::value_type;
    const Csr<T> ref      = grid<T>(8, 6, 1.0, 0.1, true);
    const auto   original = rs_coarsening_on(TypeParam::backend, ref);
    EXPECT_EQ(original.second, rs_strength(ref, 0.25));
    expect_valid_splitting(ref, original.second, original.first);

    const auto negated
        = rs_coarsening_on(TypeParam::backend, transformed(ref, [](int, int, T v) { return -v; }));
    KNOWN_BUG("for a negative diagonal, a_ij < eps max_k a_ik is taken as strong");
    EXPECT_EQ(negated.second, original.second);
    EXPECT_EQ(negated.first, original.first);
}

TYPED_TEST(local_matrix_amg, rs_interpolation)
{
    using T             = typename TypeParam::value_type;
    using Interpolation = std::function<void(
        const LocalMatrix<T>&, const LocalVector<int>&, const LocalVector<bool>&, LocalMatrix<T>*)>;
    const std::vector<std::pair<std::string, Interpolation>> interpolations
        = {{"direct",
            [](const LocalMatrix<T>&    A,
               const LocalVector<int>&  cf,
               const LocalVector<bool>& S,
               LocalMatrix<T>*          P) { A.RSDirectInterpolation(cf, S, P); }},
           {"ext+i",
            [](const LocalMatrix<T>&    A,
               const LocalVector<int>&  cf,
               const LocalVector<bool>& S,
               LocalMatrix<T>*          P) { A.RSExtPIInterpolation(cf, S, false, P); }},
           {"ext+i FF1",
            [](const LocalMatrix<T>&    A,
               const LocalVector<int>&  cf,
               const LocalVector<bool>& S,
               LocalMatrix<T>*          P) { A.RSExtPIInterpolation(cf, S, true, P); }},
           {"matrix-matrix ext+i",
            [](const LocalMatrix<T>&    A,
               const LocalVector<int>&  cf,
               const LocalVector<bool>& S,
               LocalMatrix<T>*          P) { A.RSMMExtPIInterpolation(cf, S, P); }},
           {"matrix-matrix ext+e",
            [](const LocalMatrix<T>&    A,
               const LocalVector<int>&  cf,
               const LocalVector<bool>& S,
               LocalMatrix<T>*          P) { A.RSMMExtPEInterpolation(cf, S, P); }}};

    for(const auto& interpolation : interpolations)
    {
        for(const Csr<T>& ref : {grid<T>(8, 8, 1.0, 1.0, false), grid<T>(8, 6, 1.0, 0.1, false)})
        {
            SCOPED_TRACE(interpolation.first);

            auto A  = this->make(ref);
            auto cf = this->template vector_like<int>(*A);
            auto S  = this->template vector_like<bool>(*A);
            A->RSPMISCoarsening(0.25f, cf.get(), S.get());

            LocalMatrix<T> P;
            P.CloneBackend(*A);
            interpolation.second(*A, *cf, *S, &P);
            expect_interpolation(csr_of(P), values_of(*cf));
        }
    }
}

TYPED_TEST(local_matrix_amg, rs_interpolation_truncation)
{
    using T = typename TypeParam::value_type;
    // F rows with one entry below the threshold and one more survivor than max_elmts, a
    // C row, a row within both limits and an empty row
    Csr<T> P(5);
    for(const auto& row :
        std::vector<std::map<int, T>>{{{0, 0.4}, {1, -0.05}, {2, 0.3}, {3, 0.2}, {4, 0.15}},
                                      {{1, 1.0}},
                                      {{0, -0.2}, {1, 0.6}, {3, 0.5}, {4, 0.1}},
                                      {{0, 0.7}, {1, 0.3}},
                                      {}})
    {
        for(const auto& e : row)
        {
            P.add(e.first, e.second);
        }
        P.end_row();
    }

    // Thresholds 0.1 and 0.15 drop -0.05 and 0.1, keeping 2 of the rest rescales by
    // 1 / 0.7 and 1 / 1.1
    Csr<T> expected(5);
    expected.add(0, 4.0 / 7.0);
    expected.add(2, 3.0 / 7.0);
    expected.end_row();
    expected.add(1, 1.0);
    expected.end_row();
    expected.add(1, 6.0 / 11.0);
    expected.add(3, 5.0 / 11.0);
    expected.end_row();
    expected.add(0, 0.7);
    expected.add(1, 0.3);
    expected.end_row();
    expected.end_row();

    auto M = this->make(P);
    M->RSInterpolationTruncation(0.25f, 2);
    expect_near(csr_of(*M), expected, scaled_tol<T>(1e-15));
}

TYPED_TEST(local_matrix_amg, rs_interpolation_truncation_invariants)
{
    using T = typename TypeParam::value_type;
    // Rows of mixed sign, and ext+i interpolations with C rows and ties, computed once on
    // the host so that both backends truncate the same matrix
    std::vector<std::pair<std::string, Csr<T>>> matrices;

    Csr<T> mixed(3, 6);
    for(const auto& row : std::vector<std::vector<T>>{
            {0.5, -0.04, 0.3, 0.2, -0.1, 0.07}, {0.9, 0.05}, {0.25, -0.35, 0.15, 0.6, -0.08}})
    {
        for(size_t j = 0; j < row.size(); ++j)
        {
            mixed.add(static_cast<int>(j), row[j]);
        }
        mixed.end_row();
    }
    matrices.push_back({"mixed signs", mixed});

    // Every row holds 1 to 6, so which entries are kept is unambiguous
    Csr<T> magnitudes(8, 6);
    for(int i = 0; i < magnitudes.n; ++i)
    {
        for(int j = 0; j < 6; ++j)
        {
            magnitudes.add(j, static_cast<T>(j + 1));
        }
        magnitudes.end_row();
    }
    matrices.push_back({"rows of 1 to 6", magnitudes});

    for(const auto& g : std::vector<std::pair<std::string, Csr<T>>>{
            {"isotropic ext+i", grid<T>(8, 8, 1.0, 1.0, false)},
            {"anisotropic ext+i", grid<T>(8, 6, 1.0, 0.1, false)},
            {"ext+i of many blocks", grid<T>(40, 40, 1.0, 0.5, false)}})
    {
        auto              A = make_local_matrix(Backend::Host, g.second);
        LocalVector<int>  cf;
        LocalVector<bool> S;
        A->RSPMISCoarsening(0.25f, &cf, &S);
        LocalMatrix<T> E;
        A->RSExtPIInterpolation(cf, S, false, &E);
        matrices.push_back({g.first, csr_of(E)});
    }

    std::vector<std::pair<float, int>> limits;
    for(float trunc_factor : {0.0f, 0.2f, 0.5f})
    {
        for(int max_elmts : {0, 1, 2, 3, 4})
        {
            limits.push_back({trunc_factor, max_elmts});
        }
    }

    for(const auto& m : matrices)
    {
        for(const auto& limit : limits)
        {
            SCOPED_TRACE(m.first + ", trunc_factor " + std::to_string(limit.first) + ", max_elmts "
                         + std::to_string(limit.second));

            auto M = this->make(m.second);
            M->RSInterpolationTruncation(limit.first, limit.second);
            const Csr<T> T = csr_of(*M);
            expect_valid_truncation(m.second, T, limit.first, limit.second);
            if(limit.first <= 0.0f && limit.second <= 0)
            {
                expect_near(T, m.second, 0.0);
            }
        }
    }
}

namespace
{
    template <typename T>
    struct Prolongations
    {
        Csr<T> smoothed;
        Csr<T> unsmoothed;
        Csr<T> direct;
        Csr<T> ext;
    };

    // Aggregates and C/F splitting of the host, the inputs of the prolongations
    struct SetupInputs
    {
        std::vector<bool>    connections;
        std::vector<int64_t> aggregates;
        std::vector<int64_t> roots;
        std::vector<int>     cf;
        std::vector<bool>    S;
    };

    // Checks the aggregation and splitting of backend b and returns them
    template <typename T>
    SetupInputs checked_setup(Backend b, const Csr<T>& ref, double sa_eps, float rs_eps)
    {
        SCOPED_TRACE(backend_name(b));
        auto        A = make_local_matrix(b, ref);
        SetupInputs in;

        LocalVector<bool>    connections;
        LocalVector<int64_t> agg, roots;
        connections.CloneBackend(*A);
        agg.CloneBackend(*A);
        roots.CloneBackend(*A);
        A->AMGPMISAggregate(sa_eps, &connections, &agg, &roots);
        in.connections = values_of(connections);
        in.aggregates  = values_of(agg);
        in.roots       = values_of(roots);
        EXPECT_EQ(in.connections, sa_strength(ref, sa_eps));
        expect_valid_aggregation(ref, in.connections, in.aggregates, in.roots);
        expect_independent_roots(ref, in.connections, in.roots);

        LocalVector<int>  cf;
        LocalVector<bool> S;
        cf.CloneBackend(*A);
        S.CloneBackend(*A);
        A->RSPMISCoarsening(rs_eps, &cf, &S);
        in.cf = values_of(cf);
        in.S  = values_of(S);
        EXPECT_EQ(in.S, rs_strength(ref, rs_eps));
        expect_valid_splitting(ref, in.S, in.cf);

        return in;
    }

    template <typename T>
    Csr<T> smoothed_on(Backend b, const Csr<T>& ref, const SetupInputs& in)
    {
        auto           A           = make_local_matrix(b, ref);
        auto           connections = make_local_vector(*A, in.connections);
        auto           agg         = make_local_vector(*A, in.aggregates);
        auto           roots       = make_local_vector(*A, in.roots);
        LocalMatrix<T> P;
        P.CloneBackend(*A);
        A->AMGSmoothedAggregation(2.0 / 3.0, *connections, *agg, *roots, &P, 0);
        return csr_of(P);
    }

    template <typename T>
    Csr<T> ext_on(Backend b, const Csr<T>& ref, const SetupInputs& in)
    {
        auto           A  = make_local_matrix(b, ref);
        auto           cf = make_local_vector(*A, in.cf);
        auto           S  = make_local_vector(*A, in.S);
        LocalMatrix<T> P;
        P.CloneBackend(*A);
        A->RSExtPIInterpolation(*cf, *S, false, &P);
        return csr_of(P);
    }

    template <typename T>
    Prolongations<T> prolongations(Backend b, const Csr<T>& ref, const SetupInputs& in)
    {
        auto A     = make_local_matrix(b, ref);
        auto agg   = make_local_vector(*A, in.aggregates);
        auto roots = make_local_vector(*A, in.roots);
        auto cf    = make_local_vector(*A, in.cf);
        auto S     = make_local_vector(*A, in.S);

        LocalMatrix<T> UA, D;
        UA.CloneBackend(*A);
        D.CloneBackend(*A);
        A->AMGUnsmoothedAggregation(*agg, *roots, &UA);
        A->RSDirectInterpolation(*cf, *S, &D);
        return {smoothed_on(b, ref, in), csr_of(UA), csr_of(D), ext_on(b, ref, in)};
    }

    // The interpolations are deterministic functions of their inputs, so both backends
    // compute them up to the order of summation
    template <typename T>
    void expect_prolongations(const Csr<T>& ref, const SetupInputs& in, double tol)
    {
        const Prolongations<T> host = prolongations(Backend::Host, ref, in);
        const Prolongations<T> acc  = prolongations(Backend::Accelerator, ref, in);

        expect_near(host.smoothed,
                    smoothed_prolongation(ref, in.connections, in.aggregates, 2.0 / 3.0, 0),
                    tol);
        expect_near(acc.smoothed, host.smoothed, tol);

        expect_near(host.unsmoothed, tentative_prolongation<T>(in.aggregates), 0.0);
        expect_near(acc.unsmoothed, host.unsmoothed, 0.0);

        expect_interpolation(host.direct, in.cf);
        expect_near(acc.direct, host.direct, tol);

        expect_interpolation(host.ext, in.cf);
        expect_near(acc.ext, host.ext, tol);
    }

    template <typename P>
    class local_matrix_amg_backends : public TypedBackendTest<P>
    {
    };
}

TYPED_TEST_SUITE(local_matrix_amg_backends, AcceleratorOnly<RealTypes>, TypedName);

TYPED_TEST(local_matrix_amg_backends, agree_for_the_same_inputs)
{
    using T = typename TypeParam::value_type;

    // Zero row sums, so that the RS interpolations reproduce the constants
    for(const auto& g : std::vector<std::pair<std::string, Csr<T>>>{
            {"isotropic", grid<T>(8, 8, 1.0, 1.0, false)},
            {"anisotropic", grid<T>(8, 6, 1.0, 0.1, false)}})
    {
        SCOPED_TRACE(g.first);
        checked_setup(Backend::Accelerator, g.second, 0.08, 0.25f);
        expect_prolongations(
            g.second, checked_setup(Backend::Host, g.second, 0.08, 0.25f), scaled_tol<T>(1e-14));
    }
}

namespace
{
    // Node 0 coupled to every other node. Ray diagonal 1 gives zero row sums, 2 the
    // diagonally dominant rays of the former SA and ext+i host fallback tests.
    template <typename T>
    Csr<T> star(int rays, int ray_diagonal = 1)
    {
        Csr<T> A(rays + 1);
        A.add(0, static_cast<T>(rays));
        for(int j = 1; j <= rays; ++j)
        {
            A.add(j, -1.0);
        }
        A.end_row();
        for(int i = 1; i <= rays; ++i)
        {
            A.add(0, -1.0);
            A.add(i, static_cast<T>(ray_diagonal));
            A.end_row();
        }
        return A;
    }
}

// With every node its own aggregate, row 0 of P has an entry for every node. The HIP
// ProlongNnz bins the longest row of A up to 1024 and ProlongFill the longest row of P
// up to 512, above which LocalMatrix computes P on the host.
TYPED_TEST(local_matrix_amg_backends, smoothed_aggregation_of_long_rows)
{
    using T = typename TypeParam::value_type;

    // Longest bins of both steps, the ProlongFill fallback, the ProlongNnz fallback; a
    // middle bin and the ProlongFill fallback with dominant rays
    for(const auto& c :
        std::vector<std::pair<int, int>>{{300, 1}, {600, 1}, {1100, 1}, {200, 2}, {600, 2}})
    {
        const int rays = c.first;
        SCOPED_TRACE(std::to_string(rays) + " rays of diagonal " + std::to_string(c.second));
        const Csr<T> ref = star<T>(rays, c.second);

        SetupInputs in;
        in.connections = off_diagonal(ref);
        for(int i = 0; i < ref.n; ++i)
        {
            in.aggregates.push_back(i);
            in.roots.push_back(i);
        }

        const Csr<T> host = smoothed_on(Backend::Host, ref, in);
        EXPECT_EQ(max_row_nnz(host), rays + 1);
        expect_near(host,
                    smoothed_prolongation(ref, in.connections, in.aggregates, 2.0 / 3.0, 0),
                    scaled_tol<T>(1e-14));
        expect_near(smoothed_on(Backend::Accelerator, ref, in), host, scaled_tol<T>(1e-14));
    }
}

// The F centre interpolates from every C ray. The HIP ProlongNnz bins the estimated row
// length up to 8192 and ProlongFill the row length of P up to 4096, above which
// LocalMatrix computes P on the host.
TYPED_TEST(local_matrix_amg_backends, ext_interpolation_of_long_rows)
{
    using T = typename TypeParam::value_type;

    // Bins below 2048 and 4096 of both steps, the ProlongFill fallback, the ProlongNnz
    // fallback; a middle bin and the ProlongFill fallback with dominant rays
    for(const auto& c : std::vector<std::pair<int, int>>{
            {1500, 1}, {3000, 1}, {4200, 1}, {8200, 1}, {512, 2}, {4200, 2}})
    {
        const int rays = c.first;
        SCOPED_TRACE(std::to_string(rays) + " rays of diagonal " + std::to_string(c.second));
        const Csr<T> ref = star<T>(rays, c.second);

        SetupInputs in;
        in.S     = off_diagonal(ref);
        in.cf    = std::vector<int>(ref.n, 1);
        in.cf[0] = 2;
        expect_valid_splitting(ref, in.S, in.cf);

        const Csr<T> host = ext_on(Backend::Host, ref, in);
        EXPECT_EQ(max_row_nnz(host), rays);
        expect_interpolation(host, in.cf);
        expect_near(ext_on(Backend::Accelerator, ref, in), host, scaled_tol<T>(1e-14));
    }
}

// Ext+i (De Sterck et al. 2008): w_ij = -(a_ij + sum_k a_ik abar_kj / D_k) / atilde_ii,
// D_k = sum_{l in C^_i + {i}} abar_kl, where abar_kl drops the a_kl of the sign of a_kk.
// The formula does not change when a row is scaled by -1, and the zero row sums make
// it reproduce the constants.
//   F row 1: strong C 0, strong F 2 through which 3 joins C^_1, weak a_13 in C^_1
//   F row 2: strong C 0 and 3, weak F 1 lumped into the diagonal, positive a_20
//   F row 4: strong C 3 and 5, positive a_45
TYPED_TEST(local_matrix_amg, rs_ext_interpolation_of_both_signs)
{
    using T                                    = typename TypeParam::value_type;
    const Csr<T>                           ref = from_rows<T>(6,
                                                              {{{0, 1.5}, {1, -1.0}, {2, -0.5}},
                                                               {{0, -1.0}, {1, 1.5}, {2, -1.0}, {3, 0.5}},
                                                               {{0, 0.5}, {1, 0.25}, {2, 1.25}, {3, -2.0}},
                                                               {{2, -1.0}, {3, 2.0}, {4, -1.0}},
                                                               {{3, -2.0}, {4, 1.0}, {5, 1.0}},
                                                               {{4, -1.0}, {5, 1.0}}});
    const std::vector<std::pair<int, int>> strong
        = {{1, 0}, {1, 2}, {2, 0}, {2, 3}, {4, 3}, {4, 5}};
    const Csr<T> expected = from_rows<T>(3,
                                         {{{0, 1.0}},
                                          {{0, 2.0 / 3.0}, {1, 1.0 / 3.0}},
                                          {{0, -1.0 / 3.0}, {1, 4.0 / 3.0}},
                                          {{1, 1.0}},
                                          {{1, 2.0}, {2, -1.0}},
                                          {{2, 1.0}}});

    SetupInputs in;
    in.cf = {1, 2, 2, 1, 2, 1};
    in.S  = flags_at(ref, strong);

    const auto expect_expected = [&](const Csr<T>& A, const SetupInputs& inputs) {
        const Csr<T> P = ext_on(TypeParam::backend, A, inputs);
        expect_interpolation(P, inputs.cf);
        expect_near(dense(P), dense(expected), tol<T>());
    };
    {
        SCOPED_TRACE("A");
        expect_expected(ref, in);
    }
    {
        SCOPED_TRACE("diagonal flagged strong");
        std::vector<std::pair<int, int>> with_diagonal = strong;
        for(int i = 0; i < ref.n; ++i)
        {
            with_diagonal.emplace_back(i, i);
        }
        SetupInputs diagonal = in;
        diagonal.S           = flags_at(ref, with_diagonal);
        expect_expected(ref, diagonal);
    }
    {
        SCOPED_TRACE("-A");
        expect_expected(transformed(ref, [](int, int, T v) { return -v; }), in);
    }

    const Csr<T> row_2_negated = transformed(ref, [](int i, int, T v) { return i == 2 ? -v : v; });
    const Csr<T> P             = ext_on(TypeParam::backend, row_2_negated, in);
    KNOWN_BUG("D_k drops the a_kl of the sign of a_ii instead of a_kk, so w_13 = -19/9");
    expect_near(dense(P), dense(expected), tol<T>());
}

namespace
{
    template <typename T>
    using RsInterpolation = std::function<void(
        const LocalMatrix<T>&, const LocalVector<int>&, const LocalVector<bool>&, LocalMatrix<T>*)>;

    template <typename T>
    std::vector<std::pair<std::string, RsInterpolation<T>>> ext_interpolations()
    {
        return {{"ext+i",
                 [](const LocalMatrix<T>&    A,
                    const LocalVector<int>&  cf,
                    const LocalVector<bool>& S,
                    LocalMatrix<T>*          P) { A.RSExtPIInterpolation(cf, S, false, P); }},
                {"ext+i FF1",
                 [](const LocalMatrix<T>&    A,
                    const LocalVector<int>&  cf,
                    const LocalVector<bool>& S,
                    LocalMatrix<T>*          P) { A.RSExtPIInterpolation(cf, S, true, P); }},
                {"matrix-matrix ext+i",
                 [](const LocalMatrix<T>&    A,
                    const LocalVector<int>&  cf,
                    const LocalVector<bool>& S,
                    LocalMatrix<T>*          P) { A.RSMMExtPIInterpolation(cf, S, P); }},
                {"matrix-matrix ext+e",
                 [](const LocalMatrix<T>&    A,
                    const LocalVector<int>&  cf,
                    const LocalVector<bool>& S,
                    LocalMatrix<T>*          P) { A.RSMMExtPEInterpolation(cf, S, P); }}};
    }

    template <typename T>
    Csr<T> interpolated_on(Backend                   b,
                           const Csr<T>&             ref,
                           const SetupInputs&        in,
                           const RsInterpolation<T>& interpolation)
    {
        auto           A  = make_local_matrix(b, ref);
        auto           cf = make_local_vector(*A, in.cf);
        auto           S  = make_local_vector(*A, in.S);
        LocalMatrix<T> P;
        P.CloneBackend(*A);
        interpolation(*A, *cf, *S, &P);
        return csr_of(P);
    }

    // Each interpolation of ext_interpolations, by name, against its expected P
    template <typename T>
    void expect_ext_interpolations(Backend                              b,
                                   const Csr<T>&                        A,
                                   const SetupInputs&                   in,
                                   const std::map<std::string, Csr<T>>& expected)
    {
        for(const auto& interpolation : ext_interpolations<T>())
        {
            const auto e = expected.find(interpolation.first);
            if(e == expected.end())
            {
                continue;
            }
            SCOPED_TRACE(interpolation.first);
            const Csr<T> P = interpolated_on(b, A, in, interpolation.second);
            expect_interpolation(P, in.cf);
            expect_near(dense(P), dense(e->second), tol<T>());
        }
    }
}

// Matrix-matrix ext+i (Li, Sjoegreen and Yang 2021; hypre's ModExtPI) and ext+e (ModExtPE),
// over the strong F neighbours k of i and the strong C neighbours j of i and k:
//   ext+i: w_ij = -(a_ij + sum_k a_ik a_kj / (q_k + s_ki)) / (d_i + sum_k a_ik s_ki / (q_k + s_ki))
//   ext+e: the same with s_ki replaced by l_k
// q_k is the sum of the strong C couplings of k, s_ki = a_ki if i strongly influences k and
// 0 otherwise, l_k the mean of the strong F couplings of k, and d_i the diagonal plus the
// weak couplings of i. No coupling is dropped for its sign, so scaling a row by -1 changes
// no weight. The matrix of rs_ext_interpolation_of_both_signs, with (2, 1) also strong:
//   row 1: k = 2, q_2 + s_21 = 0.5 - 2 + 0.25 = -1.25, d_1 = 1.5 + 0.5 = 2,
//          denominator 2 + (-1)(0.25) / -1.25 = 2.2,
//          w_10 = -(-1 + (-1)(0.5) / -1.25) / 2.2 = 3/11, w_13 = -((-1)(-2) / -1.25) / 2.2 = 8/11
//   row 2: k = 1, q_1 + s_12 = -1 - 1 = -2, d_2 = 1.25,
//          denominator 1.25 + (0.25)(-1) / -2 = 1.375,
//          w_20 = -(0.5 + (0.25)(-1) / -2) / 1.375 = -5/11, w_23 = 2 / 1.375 = 16/11
//   row 4: no strong F neighbour, d_4 = 1, w_43 = 2, w_45 = -1
// The only strong F coupling of 1 and of 2 is the one to the other, so l_k = s_ki and ext+e
// gives the same P. Classical ext+i differs (rs_ext_interpolation_of_both_signs): it keeps
// the weak a_13 in the interpolation, and drops a_20 for its sign.
TYPED_TEST(local_matrix_amg, rs_matrix_matrix_ext_interpolations_of_both_signs)
{
    using T                                    = typename TypeParam::value_type;
    const Csr<T>                           ref = from_rows<T>(6,
                                                              {{{0, 1.5}, {1, -1.0}, {2, -0.5}},
                                                               {{0, -1.0}, {1, 1.5}, {2, -1.0}, {3, 0.5}},
                                                               {{0, 0.5}, {1, 0.25}, {2, 1.25}, {3, -2.0}},
                                                               {{2, -1.0}, {3, 2.0}, {4, -1.0}},
                                                               {{3, -2.0}, {4, 1.0}, {5, 1.0}},
                                                               {{4, -1.0}, {5, 1.0}}});
    const std::vector<std::pair<int, int>> strong
        = {{1, 0}, {1, 2}, {2, 0}, {2, 1}, {2, 3}, {4, 3}, {4, 5}};
    const Csr<T>                        expected = from_rows<T>(3,
                                                                {{{0, 1.0}},
                                                                 {{0, 3.0 / 11.0}, {1, 8.0 / 11.0}},
                                                                 {{0, -5.0 / 11.0}, {1, 16.0 / 11.0}},
                                                                 {{1, 1.0}},
                                                                 {{1, 2.0}, {2, -1.0}},
                                                                 {{2, 1.0}}});
    const std::map<std::string, Csr<T>> both
        = {{"matrix-matrix ext+i", expected}, {"matrix-matrix ext+e", expected}};

    SetupInputs in;
    in.cf = {1, 2, 2, 1, 2, 1};
    in.S  = flags_at(ref, strong);
    {
        SCOPED_TRACE("A");
        expect_ext_interpolations(TypeParam::backend, ref, in, both);
    }
    {
        SCOPED_TRACE("diagonal flagged strong");
        std::vector<std::pair<int, int>> with_diagonal = strong;
        for(int i = 0; i < ref.n; ++i)
        {
            with_diagonal.emplace_back(i, i);
        }
        SetupInputs diagonal = in;
        diagonal.S           = flags_at(ref, with_diagonal);
        expect_ext_interpolations(TypeParam::backend, ref, diagonal, both);
    }
    {
        SCOPED_TRACE("-A");
        expect_ext_interpolations(
            TypeParam::backend, transformed(ref, [](int, int, T v) { return -v; }), in, both);
    }
    {
        SCOPED_TRACE("row 2 negated");
        expect_ext_interpolations(TypeParam::backend,
                                  transformed(ref, [](int i, int, T v) { return i == 2 ? -v : v; }),
                                  in,
                                  both);
    }
}

// The chain 0 - 1 - 2 - 3 - 4 with C points 0 and 4, and the C point 5 coupled to 2. Every
// coupling of an F row is strong and negative, and no F row has a weak coupling, so the
// classical and the matrix-matrix ext+i agree (C^_i is C^s_i and the C^s_k, and D_k is
// q_k + a_ki), and FF1 changes nothing because no F point has two strong C points. With the
// formulas of rs_matrix_matrix_ext_interpolations_of_both_signs:
//   row 1: k = 2, q_2 + a_21 = -3, w_1 = -(-2, 0, (-1)(-2) / -3) / (3 + (-1)(-1) / -3)
//          = (3/4, 0, 1/4)
//   row 2: k = 1 with q_1 + a_12 = -3, k = 3 with q_3 + a_32 = -2,
//          denominator 6 + (-1)(-1) / -3 + (-3)(-1) / -2 = 25/6,
//          w_2 = (6/25) (2/3, 3/2, 2) = (4/25, 9/25, 12/25)
//   row 3: k = 2, q_2 + a_23 = -5, w_3 = -(0, -1, (-1)(-2) / -5) / (2 + (-1)(-3) / -5)
//          = (0, 5/7, 2/7)
// ext+e takes the mean l_2 = (a_21 + a_23) / 2 = -2 instead of a_21 and a_23, so
// q_2 + l_2 = -4 for rows 1 and 3; l_1 = a_12 and l_3 = a_32, so row 2 does not change:
//   row 1: w_1 = -(-2, 0, (-1)(-2) / -4) / (3 + (-1)(-2) / -4) = (4/5, 0, 1/5)
//   row 3: w_3 = -(0, -1, (-1)(-2) / -4) / (2 + (-1)(-2) / -4) = (0, 2/3, 1/3)
TYPED_TEST(local_matrix_amg, rs_ext_e_interpolation_takes_the_mean_of_the_f_couplings)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = from_rows<T>(6,
                                    {{{0, 1.0}, {1, -1.0}},
                                     {{0, -2.0}, {1, 3.0}, {2, -1.0}},
                                     {{1, -1.0}, {2, 6.0}, {3, -3.0}, {5, -2.0}},
                                     {{2, -1.0}, {3, 2.0}, {4, -1.0}},
                                     {{3, -1.0}, {4, 1.0}},
                                     {{2, -1.0}, {5, 1.0}}});
    SetupInputs  in;
    in.cf = {1, 2, 2, 2, 1, 1};
    in.S  = flags_at(ref, {{1, 0}, {1, 2}, {2, 1}, {2, 3}, {2, 5}, {3, 2}, {3, 4}});

    const auto with_rows = [](const std::vector<std::pair<int, double>>& row_1,
                              const std::vector<std::pair<int, double>>& row_3) {
        return from_rows<T>(3,
                            {{{0, 1.0}},
                             row_1,
                             {{0, 4.0 / 25.0}, {1, 9.0 / 25.0}, {2, 12.0 / 25.0}},
                             row_3,
                             {{1, 1.0}},
                             {{2, 1.0}}});
    };
    const Csr<T> ext_i
        = with_rows({{0, 3.0 / 4.0}, {2, 1.0 / 4.0}}, {{1, 5.0 / 7.0}, {2, 2.0 / 7.0}});
    const Csr<T> ext_e
        = with_rows({{0, 4.0 / 5.0}, {2, 1.0 / 5.0}}, {{1, 2.0 / 3.0}, {2, 1.0 / 3.0}});
    expect_ext_interpolations(TypeParam::backend,
                              ref,
                              in,
                              {{"ext+i", ext_i},
                               {"ext+i FF1", ext_i},
                               {"matrix-matrix ext+i", ext_i},
                               {"matrix-matrix ext+e", ext_e}});
}

// The F point 1 interpolates from its strong C point 0 and its strong F neighbour 2, which
// shares no C point with 1 and has the strong C points 3 and 4. Ext+i adds both to C^_1; with
// the FF1 limit only the first strong C point of 2 joins, 3. Every coupling of an F row is
// strong and negative, so the abar of De Sterck et al. (2008) are the a_kl.
//   ext+i, row 1: C^_1 = {0, 3, 4}, D_2 = a_21 + a_23 + a_24 = -4,
//          denominator 3 + (-1)(-1) / -4 = 11/4,
//          w_1 = -(-2, (-1)(-2) / -4, (-1)(-1) / -4) / (11/4) = (8/11, 2/11, 1/11)
//   FF1, row 1: C^_1 = {0, 3}, D_2 = a_21 + a_23 = -3, denominator 3 + (-1)(-1) / -3 = 8/3,
//          w_1 = -(-2, (-1)(-2) / -3, 0) / (8/3) = (3/4, 1/4, 0)
//   row 2: C^_2 = {0, 3, 4} with either, D_1 = a_10 + a_12 = -3,
//          denominator 4 + (-1)(-1) / -3 = 11/3,
//          w_2 = -((-1)(-2) / -3, -2, -1) / (11/3) = (2/11, 6/11, 3/11)
// The matrix-matrix ext+i gives the ext+i weights (q_2 + a_21 = -4, q_1 + a_12 = -3), and
// ext+e the same, because 1 and 2 have one strong F coupling each.
TYPED_TEST(local_matrix_amg, rs_ext_interpolation_with_the_ff1_limit)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = from_rows<T>(5,
                                    {{{0, 1.0}, {1, -1.0}},
                                     {{0, -2.0}, {1, 3.0}, {2, -1.0}},
                                     {{1, -1.0}, {2, 4.0}, {3, -2.0}, {4, -1.0}},
                                     {{2, -2.0}, {3, 2.0}},
                                     {{2, -1.0}, {4, 1.0}}});
    SetupInputs  in;
    in.cf = {1, 2, 2, 1, 1};
    in.S  = flags_at(ref, {{1, 0}, {1, 2}, {2, 1}, {2, 3}, {2, 4}});

    const auto with_row_1 = [](const std::vector<std::pair<int, double>>& row_1) {
        return from_rows<T>(3,
                            {{{0, 1.0}},
                             row_1,
                             {{0, 2.0 / 11.0}, {1, 6.0 / 11.0}, {2, 3.0 / 11.0}},
                             {{1, 1.0}},
                             {{2, 1.0}}});
    };
    const Csr<T> ext_i = with_row_1({{0, 8.0 / 11.0}, {1, 2.0 / 11.0}, {2, 1.0 / 11.0}});
    expect_ext_interpolations(TypeParam::backend,
                              ref,
                              in,
                              {{"ext+i", ext_i},
                               {"ext+i FF1", with_row_1({{0, 3.0 / 4.0}, {1, 1.0 / 4.0}})},
                               {"matrix-matrix ext+i", ext_i},
                               {"matrix-matrix ext+e", ext_i}});
}

// The F point 1 has the strong C point 0 and the strong F neighbour 2, whose only strong C
// point 3 joins C^_1 = {0, 3}. Every diagonal is positive, and 2 has no negative coupling
// to C^_1 + {1} (a_21 = 0, a_23 = 1), so D_2 = 0 in ext+i (De Sterck et al. 2008), with or
// without FF1. hypre's BuildExtPIInterp (par_lr_interp.c) then cannot distribute a_12 and
// adds it to the diagonal, which keeps the constants for the zero row sums:
//   row 1: atilde_11 = a_11 + a_12 = 1, w_1 = -(a_10, 0) / 1 = (1, 0)
//   row 2: C^_2 = {3}, weak a_24 lumped, w_23 = -a_23 / (a_22 + a_24) = 1
// The matrix-matrix ext+i and ext+e have no D_k: q_2 + s_21 = q_2 + l_2 = 1, so
//   row 1: w_1 = -(a_10, a_12 a_23 / 1) / (a_11 + 0) = (1/2, 1/2)
TYPED_TEST(local_matrix_amg, rs_ext_interpolation_without_distribution_through_a_strong_f_point)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = from_rows<T>(5,
                                    {{{0, 1.0}, {1, -1.0}},
                                     {{0, -1.0}, {1, 2.0}, {2, -1.0}},
                                     {{2, 1.0}, {3, 1.0}, {4, -2.0}},
                                     {{2, -1.0}, {3, 1.0}},
                                     {{2, -1.0}, {4, 1.0}}});
    SetupInputs  in;
    in.cf = {1, 2, 2, 1, 1};
    in.S  = flags_at(ref, {{1, 0}, {1, 2}, {2, 3}});

    const auto with_row_1 = [](const std::vector<std::pair<int, double>>& row_1) {
        return from_rows<T>(3, {{{0, 1.0}}, row_1, {{1, 1.0}}, {{1, 1.0}}, {{2, 1.0}}});
    };
    const Csr<T> mm = with_row_1({{0, 0.5}, {1, 0.5}});
    expect_ext_interpolations(
        TypeParam::backend, ref, in, {{"matrix-matrix ext+i", mm}, {"matrix-matrix ext+e", mm}});

    const Csr<T>                                expected = with_row_1({{0, 1.0}, {1, 0.0}});
    const std::vector<T>                        D        = dense(expected);
    std::vector<std::pair<std::string, Csr<T>>> ext;
    for(const auto& interpolation : ext_interpolations<T>())
    {
        if(interpolation.first != "ext+i" && interpolation.first != "ext+i FF1")
        {
            continue;
        }
        SCOPED_TRACE(interpolation.first);
        ext.emplace_back(interpolation.first,
                         interpolated_on(TypeParam::backend, ref, in, interpolation.second));
        const Csr<T>&        P  = ext.back().second;
        const std::vector<T> DP = dense(P);
        ASSERT_EQ(DP.size(), D.size());
        EXPECT_EQ(P.row_offset, expected.row_offset);
        EXPECT_EQ(P.col, expected.col);
        for(int i = 0; i < expected.n; ++i)
        {
            if(i == 1)
            {
                continue;
            }
            for(int j = 0; j < expected.m; ++j)
            {
                EXPECT_NEAR(DP[i * expected.m + j], D[i * expected.m + j], tol<T>())
                    << "row " << i << ", column " << j;
            }
        }
    }

    KNOWN_BUG("with D_k = 0 the a_ik / D_k is infinite and a_ik is dropped instead of lumped "
              "into the diagonal, so w_1 = (1/2, 0)");
    for(const auto& P : ext)
    {
        SCOPED_TRACE(P.first);
        expect_interpolation(P.second, in.cf);
        expect_near(dense(P.second), D, tol<T>());
    }
}

// The overloads with the ghost part of a distributed matrix count its couplings. The
// accelerator has no pairwise aggregation and falls back to the host.
TYPED_TEST(local_matrix_amg, pairwise_aggregation_with_a_ghost_part)
{
    using T = typename TypeParam::value_type;

    struct Pairs
    {
        int  nc     = -1;
        int  Gsize  = -1;
        int  rGsize = -1;
        int* rG     = nullptr;

        ~Pairs()
        {
            rocalution::free_host(&rG);
        }
    };

    // Row 2 of the path is strongly diagonally dominant on its own (11 > 5 * 2), but not
    // with a ghost coupling of -1 (11 < 5 * 3), so the initial pass pairs it
    const Csr<T> dominant = transformed(
        band<T>(6, 1, false), [](int i, int j, T v) { return i == 2 && j == 2 ? T(11) : v; });
    Csr<T> ghost(6, 2);
    for(int i = 0; i < 6; ++i)
    {
        if(i == 2)
        {
            ghost.add(1, -1.0);
        }
        ghost.end_row();
    }
    {
        auto  A = this->make(dominant);
        auto  G = make_local_vector(*A, std::vector<int>(6, 7));
        Pairs local, with_ghost;
        A->InitialPairwiseAggregation(
            T(0.25), local.nc, G.get(), local.Gsize, &local.rG, local.rGsize, 0);
        EXPECT_EQ(values_of(*G), (std::vector<int>{0, 0, -1, 1, 1, 2}));

        A->InitialPairwiseAggregation(*this->make(ghost),
                                      T(0.25),
                                      with_ghost.nc,
                                      G.get(),
                                      with_ghost.Gsize,
                                      &with_ghost.rG,
                                      with_ghost.rGsize,
                                      0);
        EXPECT_EQ(with_ghost.nc, 3);
        EXPECT_EQ(values_of(*G), (std::vector<int>{0, 0, 1, 1, 2, 2}));
        EXPECT_EQ(backend_of(*G), backend_name(TypeParam::backend));
    }

    // Ghost couplings weaker than the local ones change no pairing in either pass. The
    // coarse matrix of the pairs {0, 1} and {2, 3} of the path is the path of 2 nodes.
    Csr<T> weak(4, 1), coarse_weak(2, 1);
    for(Csr<T>* g : {&weak, &coarse_weak})
    {
        for(int i = 0; i < g->n; ++i)
        {
            g->add(0, -0.1);
            g->end_row();
        }
    }
    auto  A = this->make(band<T>(4, 1, false));
    auto  G = make_local_vector(*A, std::vector<int>(4, 7));
    Pairs agg;
    A->InitialPairwiseAggregation(
        *this->make(weak), T(0.25), agg.nc, G.get(), agg.Gsize, &agg.rG, agg.rGsize, 0);
    ASSERT_EQ(values_of(*G), (std::vector<int>{0, 0, 1, 1}));

    auto Ac = this->make(band<T>(2, 1, false));
    Ac->FurtherPairwiseAggregation(
        *this->make(coarse_weak), T(0.25), agg.nc, G.get(), agg.Gsize, &agg.rG, agg.rGsize, 0);
    EXPECT_EQ(agg.nc, 1);
    EXPECT_EQ(values_of(*G), (std::vector<int>{0, 0, 0, 0}));
    EXPECT_EQ(backend_of(*G), backend_name(TypeParam::backend));
}

namespace
{
    // The HIP setup picks its kernels by the size of the rows:
    //   strong connections and PMIS sweeps   nnz(A) / n:   <= 8, 16, ..., 256, above
    //   SA ProlongNnz                        longest row of A:  < 8, 16, ..., 1024, host
    //   SA ProlongFill                       longest row of P:  < 8, 16, ..., 512, host
    //   ext+i ProlongNnz                     row bound:         < 16, 32, ..., 8192, host
    //   ext+i ProlongFill                    longest row of P:  < 16, 32, ..., 4096, host
    // The row bound of an F point counts its strong C neighbours and the strong C
    // neighbours of each of its strong F neighbours.
    // The bins above the sweep are covered by the *_of_long_rows tests.
    struct SizeCase
    {
        std::string name;
        int         n;
        int         bandwidth;
        bool        fan;
    };

    // Every coupling is strong for SA
    template <typename T>
    double sa_eps(const Csr<T>& A)
    {
        T largest = 0;
        for(int i = 0; i < A.n; ++i)
        {
            largest = std::max(largest, entry(A, i, i));
        }
        return 0.5 / static_cast<double>(largest);
    }

    class local_matrix_amg_sizes : public testing::TestWithParam<SizeCase>
    {
    protected:
        void SetUp() override
        {
            REQUIRE_BACKEND(Backend::Accelerator);
        }
    };

    template <typename T>
    void expect_backends_agree(const SizeCase& c)
    {
        SCOPED_TRACE(type_name<T>());
        const Csr<T> ref = band<T>(c.n, c.bandwidth, c.fan);
        const double eps = sa_eps(ref);

        checked_setup(Backend::Accelerator, ref, eps, 0.25f);
        const SetupInputs in = checked_setup(Backend::Host, ref, eps, 0.25f);

        // The rows sum up to a few hundred couplings in different orders
        expect_prolongations(ref, in, tol<T>());
    }
}

// gtest has no typed value-parameterized tests, so each case runs both value types
TEST_P(local_matrix_amg_sizes, backends_agree)
{
    expect_backends_agree<float>(GetParam());
    expect_backends_agree<double>(GetParam());
}

// The bands step through the bins of nnz(A) / n, of the longest row of A and of the ext+i
// row bound, the fans through the bins of the longest rows of P. As measured, with the
// bins of the table above:
//
//            nnz(A)/n   row of A     row of SA P   ext+i bound   row of ext+i P
//   band2       4 <=8      5 <8         3 <8           6 <16          3 <16
//   band5      10 <=16    11 <16        3 <8          17 <32          4 <16
//   band12     24 <=32    25 <32        3 <8          46 <64          4 <16
//   band25     50 <=64    51 <64        3 <8          98 <128         4 <16
//   band50     98 <=128  101 <128       3 <8         197 <256         4 <16
//   band100   190 <=256  201 <256       3 <8         355 <512         4 <16
//   band160   295 >256   321 <512       2 <8         613 <1024        3 <16
//   fan12       3 <=8     12 <16        4 <8          12 <16          4 <16
//   fan40       3 <=8     40 <64       14 <16         42 <64         14 <16
//   fan80       3 <=8     80 <128      29 <32         84 <128        28 <32
//   fan160      3 <=8    160 <256      60 <64        171 <256        57 <64
//   fan330      3 <=8    330 <512     120 <128       354 <512       118 <128
//   fan660      3 <=8    660 <1024    233 <256       710 <1024      237 <256
//   fan1300     3 <=8   1300 host     461 host      1413 <2048      471 <512
//   fan2600     3 <=8   2600 host     917 host      3039 <4096     1013 <1024
INSTANTIATE_TEST_SUITE_P(rows,
                         local_matrix_amg_sizes,
                         testing::Values(SizeCase{"band2", 1000, 2, false},
                                         SizeCase{"band5", 1000, 5, false},
                                         SizeCase{"band12", 1000, 12, false},
                                         SizeCase{"band25", 1000, 25, false},
                                         SizeCase{"band50", 1000, 50, false},
                                         SizeCase{"band100", 1000, 100, false},
                                         SizeCase{"band160", 1000, 160, false},
                                         SizeCase{"fan12", 12, 1, true},
                                         SizeCase{"fan40", 40, 1, true},
                                         SizeCase{"fan80", 80, 1, true},
                                         SizeCase{"fan160", 160, 1, true},
                                         SizeCase{"fan330", 330, 1, true},
                                         SizeCase{"fan660", 660, 1, true},
                                         SizeCase{"fan1300", 1300, 1, true},
                                         SizeCase{"fan2600", 2600, 1, true}),
                         [](const testing::TestParamInfo<SizeCase>& info) {
                             return info.param.name;
                         });
