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

// The steps of the pairwise aggregation AMG setup (Notay, ETNA 37, 2010) on the backend
// matrices. G maps a fine node to its aggregate, or to -1 if it is in none; aggregate c holds
// the fine nodes rG[r * rGsize + c], r < Gsize, padded with -1. The initial pass pairs the
// nodes of A, the further pass pairs the aggregates on the coarse matrix and doubles Gsize,
// and CoarsenOperator computes P^T A P of the fine matrix for P(i, G[i]) = 1. The overloads
// with a second matrix take the ghost part of a distributed matrix, whose couplings enter
// the tests of a row, but which is never paired.

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "unit_test_amg_reference.hpp"
#include "unit_test_common.hpp"
#include "unit_test_reference.hpp"
#include "unit_test_types.hpp"

#include "utils/allocate_free.hpp"

using namespace rocalution_unit;

namespace
{
    const Format csr_format{rocalution::CSR, 1};

    struct Aggregation
    {
        int              nc     = -1;
        int              Gsize  = -1;
        int              rGsize = -1;
        int*             rG     = nullptr;
        std::vector<int> G;

        Aggregation() = default;

        Aggregation(const Aggregation&)            = delete;
        Aggregation& operator=(const Aggregation&) = delete;

        ~Aggregation()
        {
            rocalution::free_host(&rG);
        }

        std::vector<int> members(int c) const
        {
            std::vector<int> m;
            for(int r = 0; r < Gsize; ++r)
            {
                if(rG[r * rGsize + c] >= 0)
                {
                    m.push_back(rG[r * rGsize + c]);
                }
            }
            return m;
        }

        std::vector<bool> unaggregated() const
        {
            std::vector<bool> none(G.size());
            for(size_t i = 0; i < G.size(); ++i)
            {
                none[i] = G[i] == -1;
            }
            return none;
        }
    };

    // Rows excluded from the initial pass: a_ii > 5 sum_j!=i |a_ij|, with the couplings of
    // the ghost part
    template <typename T>
    std::vector<bool> dominant_rows(const Csr<T>& A, const Csr<T>* ghost = nullptr)
    {
        std::vector<bool> dominant(A.n);
        for(int i = 0; i < A.n; ++i)
        {
            T sum = static_cast<T>(0);
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                sum += A.col[j] == i ? static_cast<T>(0) : std::abs(A.val[j]);
            }
            if(ghost != nullptr)
            {
                for(PtrType j = ghost->row_offset[i]; j < ghost->row_offset[i + 1]; ++j)
                {
                    sum += std::abs(ghost->val[j]);
                }
            }
            dominant[i] = entry(A, i, i) > 5 * sum;
        }
        return dominant;
    }

    // The nodes induce a connected subgraph of the pattern of A + A^T
    template <typename T>
    bool connected(const Csr<T>& A, const std::vector<int>& nodes)
    {
        std::vector<int> reached = {nodes[0]};
        for(size_t k = 0; k < reached.size(); ++k)
        {
            for(int other : nodes)
            {
                const int r = reached[k];
                if(std::find(reached.begin(), reached.end(), other) == reached.end()
                   && (entry(A, r, other) != static_cast<T>(0)
                       || entry(A, other, r) != static_cast<T>(0)))
                {
                    reached.push_back(other);
                }
            }
        }
        return reached.size() == nodes.size();
    }

    // Every aggregate holds 1 to Gsize nodes and is connected in A, G and rG describe the
    // same aggregates, and the nodes of no aggregate have G = -1
    template <typename T>
    void expect_valid_pairwise(const Csr<T>& A, const Aggregation& agg, int Gsize)
    {
        ASSERT_EQ(agg.G.size(), static_cast<size_t>(A.n));
        ASSERT_EQ(agg.Gsize, Gsize);
        ASSERT_GE(agg.nc, 0);
        ASSERT_GE(agg.rGsize, agg.nc);
        ASSERT_TRUE(agg.nc == 0 || agg.rG != nullptr);

        std::vector<int> owner(A.n, -1);
        for(int c = 0; c < agg.nc; ++c)
        {
            const std::vector<int> m = agg.members(c);
            ASSERT_GE(m.size(), 1u) << "aggregate " << c;
            for(int i : m)
            {
                ASSERT_LT(i, A.n) << "aggregate " << c;
                EXPECT_EQ(owner[i], -1)
                    << "node " << i << " is in aggregates " << owner[i] << " and " << c;
                EXPECT_EQ(agg.G[i], c) << "node " << i;
                owner[i] = c;
            }
            EXPECT_TRUE(connected(A, m)) << "aggregate " << c;
        }
        for(int c = agg.nc; c < agg.rGsize; ++c)
        {
            EXPECT_TRUE(agg.members(c).empty()) << "column " << c << " of rG";
        }
        for(int i = 0; i < A.n; ++i)
        {
            if(owner[i] == -1)
            {
                EXPECT_EQ(agg.G[i], -1) << "node " << i;
            }
        }
    }

    // P^T A P for P(i, G[i]) = 1, with an entry for every pair of coupled aggregates
    template <typename T>
    Csr<T> galerkin(const Csr<T>& A, const std::vector<int>& G, int nc)
    {
        std::vector<std::map<int, T>> rows(nc);
        for(int i = 0; i < A.n; ++i)
        {
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                if(G[i] >= 0 && G[A.col[j]] >= 0)
                {
                    rows[G[i]][G[A.col[j]]] += A.val[j];
                }
            }
        }
        Csr<T> Ac(nc);
        for(const auto& row : rows)
        {
            for(const auto& e : row)
            {
                Ac.add(e.first, e.second);
            }
            Ac.end_row();
        }
        return Ac;
    }

    template <typename T>
    Csr<T> with_sorted_rows(const Csr<T>& A)
    {
        Csr<T> B(A.n, A.m);
        for(int i = 0; i < A.n; ++i)
        {
            std::map<int, T> row;
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                row[A.col[j]] = A.val[j];
            }
            for(const auto& e : row)
            {
                B.add(e.first, e.second);
            }
            B.end_row();
        }
        return B;
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

    template <typename T>
    Csr<T> negated(Csr<T> A)
    {
        for(T& v : A.val)
        {
            v = -v;
        }
        return A;
    }

    // An n x 1 ghost part with the coupling v in every row
    template <typename T>
    Csr<T> column_ghost(int n, T v)
    {
        Csr<T> ghost(n, 1);
        for(int i = 0; i < n; ++i)
        {
            ghost.add(0, v);
            ghost.end_row();
        }
        return ghost;
    }

    // Path 0 - ... - (n - 1) with couplings -1 and zero row sums
    template <typename T>
    Csr<T> path(int n)
    {
        return band<T>(n, 1, false);
    }

    // Each coarse node c becomes the fine nodes 2c and 2c + 1, coupled by -100, and a coupling
    // (k, l), k < l, of the symmetric coarse matrix couples 2k + 1 and 2l. Then the initial
    // pass pairs 2c with 2c + 1, and the coarse matrix of these pairs is coarse.
    template <typename T>
    Csr<T> doubled(const Csr<T>& coarse)
    {
        const T        s = static_cast<T>(100);
        std::vector<T> D(4 * coarse.n * coarse.n, static_cast<T>(0));
        const int      n = 2 * coarse.n;
        for(int k = 0; k < coarse.n; ++k)
        {
            D[(2 * k) * n + 2 * k + 1]     = -s;
            D[(2 * k + 1) * n + 2 * k]     = -s;
            D[(2 * k) * n + 2 * k]         = s;
            D[(2 * k + 1) * n + 2 * k + 1] = s + entry(coarse, k, k);
            for(int l = k + 1; l < coarse.n; ++l)
            {
                D[(2 * k + 1) * n + 2 * l] = entry(coarse, k, l);
                D[(2 * l) * n + 2 * k + 1] = entry(coarse, k, l);
            }
        }
        return from_dense(n, n, D);
    }

    template <typename P>
    class backend_pairwise_amg : public TypedBackendTest<P>
    {
    protected:
        using T = typename P::value_type;

        std::unique_ptr<BaseMatrix<T>> make(const Csr<T>& ref) const
        {
            return make_matrix(P::backend, csr_format, *ref.host_csr());
        }

        bool initial(const Csr<T>& A,
                     T             beta,
                     int           ordering,
                     Aggregation*  agg,
                     const Csr<T>* ghost = nullptr) const
        {
            auto M = make(A);
            auto G = make_vector(P::backend, std::vector<int>(A.n, 7));
            bool native
                = ghost == nullptr
                      ? M->InitialPairwiseAggregation(
                            beta, agg->nc, G.get(), agg->Gsize, &agg->rG, agg->rGsize, ordering)
                      : M->InitialPairwiseAggregation(*make(*ghost),
                                                      beta,
                                                      agg->nc,
                                                      G.get(),
                                                      agg->Gsize,
                                                      &agg->rG,
                                                      agg->rGsize,
                                                      ordering);
            agg->G = values_of(*G);
            return native;
        }

        // On the coarse matrix Ac of the aggregation
        bool further(const Csr<T>& Ac,
                     T             beta,
                     int           ordering,
                     Aggregation*  agg,
                     const Csr<T>* ghost = nullptr) const
        {
            auto M = make(Ac);
            auto G = make_vector(P::backend, agg->G);
            bool native
                = ghost == nullptr
                      ? M->FurtherPairwiseAggregation(
                            beta, agg->nc, G.get(), agg->Gsize, &agg->rG, agg->rGsize, ordering)
                      : M->FurtherPairwiseAggregation(*make(*ghost),
                                                      beta,
                                                      agg->nc,
                                                      G.get(),
                                                      agg->Gsize,
                                                      &agg->rG,
                                                      agg->rGsize,
                                                      ordering);
            agg->G = values_of(*G);
            return native;
        }

        Csr<T> coarsen(const Csr<T>& A, const Aggregation& agg) const
        {
            auto M  = make(A);
            auto Ac = empty_matrix<T>(P::backend, csr_format);
            EXPECT_TRUE(M->CoarsenOperator(Ac.get(),
                                           agg.nc,
                                           agg.nc,
                                           *make_vector(P::backend, agg.G),
                                           agg.Gsize,
                                           agg.rG,
                                           agg.rGsize));
            return csr_of(*Ac);
        }
    };
}

// The pairwise steps have no entry in implements(): only the host CSR has them. The
// accelerator matrices only return false, see backend_pairwise_amg_fallback.
TYPED_TEST_SUITE(backend_pairwise_amg, HostOnly<RealTypes>, TypedName);

// Visiting the nodes in index order, each unaggregated node takes its most negatively
// coupled unaggregated neighbour
TYPED_TEST(backend_pairwise_amg, initial_pass_pairs_the_nodes_of_a_path)
{
    using T = typename TypeParam::value_type;
    Aggregation agg;
    ASSERT_TRUE(this->initial(path<T>(6), 0.25, 0, &agg));

    EXPECT_EQ(agg.nc, 3);
    EXPECT_EQ(agg.G, (std::vector<int>{0, 0, 1, 1, 2, 2}));
    expect_valid_pairwise(path<T>(6), agg, 2);
}

TYPED_TEST(backend_pairwise_amg, initial_pass_takes_the_most_negative_coupling)
{
    using T = typename TypeParam::value_type;
    Csr<T> star(4);
    for(T v : {T(6), T(-1), T(-3), T(-2)})
    {
        star.add(static_cast<int>(star.col.size()), v);
    }
    star.end_row();
    for(int i = 1; i < 4; ++i)
    {
        star.add(0, star.val[i]);
        star.add(i, 4.0);
        star.end_row();
    }

    Aggregation agg;
    ASSERT_TRUE(this->initial(star, 0.25, 0, &agg));
    EXPECT_EQ(agg.nc, 3);
    EXPECT_EQ(agg.G, (std::vector<int>{0, 1, 0, 2}));
}

// a_ii > 5 sum |a_ij| leaves node i out of every aggregate; equality does not
TYPED_TEST(backend_pairwise_amg, initial_pass_leaves_out_strongly_diagonally_dominant_rows)
{
    using T = typename TypeParam::value_type;
    struct Case
    {
        T                diagonal;
        std::vector<int> G;
    };
    const std::vector<Case> cases = {{11.0, {0, 0, -1, 1, 1, 2}}, {10.0, {0, 0, 1, 1, 2, 2}}};

    for(const Case& c : cases)
    {
        SCOPED_TRACE("diagonal " + std::to_string(c.diagonal));
        const Csr<T> A = transformed(
            path<T>(6), [&](int i, int j, T v) { return i == 2 && j == 2 ? c.diagonal : v; });

        Aggregation agg;
        ASSERT_IN_LOOP(this->initial(A, 0.25, 0, &agg));
        EXPECT_EQ(agg.G, c.G);
        expect_valid_pairwise(A, agg, 2);
        EXPECT_EQ(agg.unaggregated(), dominant_rows(A));
    }
}

TYPED_TEST(backend_pairwise_amg, every_row_dominant_gives_no_aggregate)
{
    using T        = typename TypeParam::value_type;
    const Csr<T> A = transformed(path<T>(4), [](int i, int j, T v) { return i == j ? T(11) : v; });
    Aggregation  agg;
    ASSERT_TRUE(this->initial(A, 0.25, 0, &agg));
    EXPECT_EQ(agg.nc, 0);
    EXPECT_EQ(agg.rGsize, 0);
    EXPECT_EQ(agg.G, std::vector<int>(4, -1));
}

// Negating a row flips the signs of the couplings, so it does not change the pairing
TYPED_TEST(backend_pairwise_amg, initial_pass_of_a_negated_matrix)
{
    using T        = typename TypeParam::value_type;
    const Csr<T> A = grid<T>(5, 4, 1.0, 0.5, true);
    Aggregation  agg, neg;
    ASSERT_TRUE(this->initial(A, 0.25, 0, &agg));
    ASSERT_TRUE(this->initial(negated(A), 0.25, 0, &neg));
    EXPECT_EQ(neg.G, agg.G);

    const Csr<T> D
        = transformed(path<T>(6), [&](int i, int j, T v) { return i == 2 && j == 2 ? T(11) : v; });
    Aggregation dominant;
    ASSERT_TRUE(this->initial(negated(D), 0.25, 0, &dominant));
    expect_valid_pairwise(D, dominant, 2);

    KNOWN_BUG("the dominance test compares the signed diagonal, so a negative diagonal "
              "is never dominant");
    EXPECT_EQ(dominant.unaggregated(), dominant_rows(D));
}

// Node 2 is coupled by -4 to the aggregated node 1 and by -0.5 to its only unaggregated
// neighbour 3. -0.5 is not below -beta max_k |a_2k| = -1, so 2 stays alone.
TYPED_TEST(backend_pairwise_amg, initial_pass_pairs_only_strong_negative_couplings)
{
    using T = typename TypeParam::value_type;
    Csr<T> A(4);
    A.add(0, 4.0);
    A.add(1, -4.0);
    A.end_row();
    A.add(0, -4.0);
    A.add(1, 8.0);
    A.add(2, -4.0);
    A.end_row();
    A.add(1, -4.0);
    A.add(2, 8.0);
    A.add(3, -0.5);
    A.end_row();
    A.add(2, -0.5);
    A.add(3, 1.0);
    A.end_row();

    Aggregation agg;
    ASSERT_TRUE(this->initial(A, 0.25, 0, &agg));
    expect_valid_pairwise(A, agg, 2);
    EXPECT_EQ(agg.G[0], 0);
    EXPECT_EQ(agg.G[1], 0);

    KNOWN_BUG("the threshold is -beta times the largest signed coupling, which is "
              "positive when all couplings are negative, so every coupling passes");
    EXPECT_EQ(agg.nc, 3);
    EXPECT_NE(agg.G[2], agg.G[3]);
}

TYPED_TEST(backend_pairwise_amg, initial_pass_of_grids_in_every_ordering)
{
    using T = typename TypeParam::value_type;
    // An anisotropic grid, and an isotropic one with some dominant rows
    const Csr<T> grids[] = {grid<T>(9, 7, 1.0, 0.1, true),
                            transformed(grid<T>(8, 8, 1.0, 1.0, false), [](int i, int j, T v) {
                                return i == j && i % 7 == 3 ? T(25) : v;
                            })};

    for(const Csr<T>& A : grids)
    {
        for(int ordering = 0; ordering <= 5; ++ordering)
        {
            SCOPED_TRACE("n " + std::to_string(A.n) + ", ordering " + std::to_string(ordering));

            Aggregation agg;
            ASSERT_IN_LOOP(this->initial(A, 0.25, ordering, &agg));
            expect_valid_pairwise(A, agg, 2);
            EXPECT_EQ(agg.unaggregated(), dominant_rows(A));
            EXPECT_GE(agg.nc,
                      (A.n - static_cast<int>(std::count(agg.G.begin(), agg.G.end(), -1))) / 2);
        }
    }
}

// The connectivity ordering visits the rows by increasing number of non-zeros, and the first
// free node it visits starts the next aggregate. A member with a shorter row would have been
// visited, and started an aggregate, earlier. So the shortest row of every aggregate is at
// least as long as the one of the previous aggregate.
TYPED_TEST(backend_pairwise_amg, connectivity_ordering_visits_short_rows_first)
{
    using T        = typename TypeParam::value_type;
    const Csr<T> A = grid<T>(6, 5, 1.0, 1.0, true);
    Aggregation  agg;
    ASSERT_TRUE(this->initial(A, 0.25, 1, &agg));
    expect_valid_pairwise(A, agg, 2);

    std::vector<PtrType> shortest(agg.nc, std::numeric_limits<PtrType>::max());
    for(int i = 0; i < A.n; ++i)
    {
        if(agg.G[i] >= 0)
        {
            shortest[agg.G[i]]
                = std::min(shortest[agg.G[i]], A.row_offset[i + 1] - A.row_offset[i]);
        }
    }
    for(int c = 1; c < agg.nc; ++c)
    {
        EXPECT_LE(shortest[c - 1], shortest[c]) << "aggregate " << c;
    }
}

// The path 0 - 11 - 5 - 7 - 3 - 9 - 1 - 10 - 6 - 2 - 8 - 4. CMK numbers it along the path
// from node 0, and RCMK from node 4, so visiting the nodes in either order pairs
// neighbours along the path without leaving a node alone.
TYPED_TEST(backend_pairwise_amg, cmk_orderings_pair_a_scrambled_path_along_the_path)
{
    using T                         = typename TypeParam::value_type;
    const std::vector<int> scramble = {0, 11, 5, 7, 3, 9, 1, 10, 6, 2, 8, 4};
    const Csr<T>           A        = permuted(path<T>(12), scramble);

    for(int ordering : {2, 3})
    {
        SCOPED_TRACE(ordering == 2 ? "CMK" : "RCMK");

        Aggregation agg;
        ASSERT_IN_LOOP(this->initial(A, 0.25, ordering, &agg));
        expect_valid_pairwise(A, agg, 2);
    }

    KNOWN_BUG("the CMK, RCMK, MIS and multicoloring permutations map a node to its "
              "position, but the node visited at step k is taken as perm[k]");

    for(int ordering : {2, 3})
    {
        SCOPED_TRACE(ordering == 2 ? "CMK" : "RCMK");

        Aggregation agg;
        ASSERT_IN_LOOP(this->initial(A, 0.25, ordering, &agg));
        EXPECT_EQ(agg.nc, 6);
    }
}

// The pairs of path(6) form path(3) with the diagonal 1, 2, 1. Its nodes 0 and 1 are
// paired, and node 2 stays alone, so the aggregates of the fine nodes are {0, 1, 2, 3} and
// {4, 5}.
TYPED_TEST(backend_pairwise_amg, further_pass_pairs_the_aggregates_of_a_path)
{
    using T        = typename TypeParam::value_type;
    const Csr<T> A = path<T>(6);
    Aggregation  agg;
    ASSERT_TRUE(this->initial(A, 0.25, 0, &agg));
    const Csr<T> Ac = galerkin(A, agg.G, agg.nc);
    expect_near(Ac, path<T>(3), 0.0);

    ASSERT_TRUE(this->further(Ac, 0.25, 0, &agg));
    EXPECT_EQ(agg.nc, 2);
    EXPECT_EQ(agg.G, (std::vector<int>{0, 0, 0, 0, 1, 1}));
    expect_valid_pairwise(A, agg, 4);
}

// Nodes left out by the initial pass stay out
TYPED_TEST(backend_pairwise_amg, further_pass_keeps_dominant_rows_out)
{
    using T = typename TypeParam::value_type;
    const Csr<T> A
        = transformed(path<T>(8), [](int i, int j, T v) { return i == j && i == 3 ? T(20) : v; });
    Aggregation agg;
    ASSERT_TRUE(this->initial(A, 0.25, 0, &agg));
    ASSERT_TRUE(this->further(galerkin(A, agg.G, agg.nc), 0.25, 0, &agg));
    expect_valid_pairwise(A, agg, 4);
    EXPECT_EQ(agg.unaggregated(), dominant_rows(A));
}

// In the coarse path 0 - 1 - 2 - 3 with couplings -4, -4 and -0.5, node 2 is weakly coupled
// to its only unaggregated neighbour 3
TYPED_TEST(backend_pairwise_amg, further_pass_pairs_only_strong_negative_couplings)
{
    using T = typename TypeParam::value_type;
    Csr<T> coarse(4);
    coarse.add(0, 4.0);
    coarse.add(1, -4.0);
    coarse.end_row();
    coarse.add(0, -4.0);
    coarse.add(1, 8.0);
    coarse.add(2, -4.0);
    coarse.end_row();
    coarse.add(1, -4.0);
    coarse.add(2, 8.0);
    coarse.add(3, -0.5);
    coarse.end_row();
    coarse.add(2, -0.5);
    coarse.add(3, 1.0);
    coarse.end_row();

    const Csr<T> A = doubled(coarse);
    Aggregation  agg;
    ASSERT_TRUE(this->initial(A, 0.25, 0, &agg));
    ASSERT_EQ(agg.G, (std::vector<int>{0, 0, 1, 1, 2, 2, 3, 3}));
    ASSERT_EQ(dense(galerkin(A, agg.G, agg.nc)), dense(coarse));

    ASSERT_TRUE(this->further(coarse, 0.25, 0, &agg));
    expect_valid_pairwise(A, agg, 4);
    EXPECT_EQ(agg.G[0], agg.G[2]);

    KNOWN_BUG("the threshold is -beta times the smallest coupling, which is positive "
              "when it is negative, so every coupling passes");
    EXPECT_EQ(agg.nc, 3);
    EXPECT_NE(agg.G[4], agg.G[6]);
}

// Up to four fine nodes per aggregate after the first further pass, eight after the second
TYPED_TEST(backend_pairwise_amg, further_passes_of_grids_in_every_ordering)
{
    using T        = typename TypeParam::value_type;
    const Csr<T> A = transformed(grid<T>(9, 8, 1.0, 0.3, true), [](int i, int j, T v) {
        return i == j && i % 11 == 5 ? T(40) : v;
    });

    for(int ordering = 0; ordering <= 5; ++ordering)
    {
        SCOPED_TRACE("ordering " + std::to_string(ordering));

        Aggregation agg;
        ASSERT_IN_LOOP(this->initial(A, 0.25, ordering, &agg));
        for(int Gsize : {4, 8})
        {
            const int nc = agg.nc;
            ASSERT_IN_LOOP(this->further(galerkin(A, agg.G, agg.nc), 0.25, ordering, &agg));
            expect_valid_pairwise(A, agg, Gsize);
            EXPECT_EQ(agg.unaggregated(), dominant_rows(A));
            EXPECT_LT(agg.nc, nc);
            EXPECT_GE(2 * agg.nc, nc);
        }
    }
}

// (Ac)_kl is the sum of a_ij over i in aggregate k and j in aggregate l
TYPED_TEST(backend_pairwise_amg, coarse_operator_of_the_path_pairs)
{
    using T        = typename TypeParam::value_type;
    const Csr<T> A = path<T>(6);
    Aggregation  agg;
    ASSERT_TRUE(this->initial(A, 0.25, 0, &agg));
    expect_near(with_sorted_rows(this->coarsen(A, agg)), path<T>(3), 0.0);
}

// Excluded nodes contribute nothing
TYPED_TEST(backend_pairwise_amg, coarse_operator_drops_unaggregated_nodes)
{
    using T = typename TypeParam::value_type;
    const Csr<T> A
        = transformed(path<T>(6), [](int i, int j, T v) { return i == j && i == 2 ? T(11) : v; });
    Aggregation agg;
    ASSERT_TRUE(this->initial(A, 0.25, 0, &agg));
    ASSERT_EQ(agg.G, (std::vector<int>{0, 0, -1, 1, 1, 2}));

    Csr<T> expected(3);
    expected.add(0, 1.0);
    expected.end_row();
    expected.add(1, 2.0);
    expected.add(2, -1.0);
    expected.end_row();
    expected.add(1, -1.0);
    expected.add(2, 1.0);
    expected.end_row();
    expect_near(with_sorted_rows(this->coarsen(A, agg)), expected, 0.0);
}

TYPED_TEST(backend_pairwise_amg, coarse_operators_of_grids_are_galerkin_products)
{
    using T        = typename TypeParam::value_type;
    const Csr<T> A = transformed(grid<T>(9, 7, 1.0, 0.2, false), [](int i, int j, T v) {
        return i == j && i % 13 == 6 ? T(30) : v + (i + j) % 3;
    });

    Aggregation agg;
    ASSERT_TRUE(this->initial(A, 0.25, 0, &agg));
    for(int pass = 0; pass < 3; ++pass)
    {
        SCOPED_TRACE("pass " + std::to_string(pass));
        if(pass > 0)
        {
            ASSERT_IN_LOOP(this->further(galerkin(A, agg.G, agg.nc), 0.25, 0, &agg));
        }
        const Csr<T> Ac = this->coarsen(A, agg);
        EXPECT_EQ(Ac.nnz(), galerkin(A, agg.G, agg.nc).nnz());
        expect_near(with_sorted_rows(Ac), galerkin(A, agg.G, agg.nc), tol<T>());
    }
}

// Without ghost couplings, the overloads for the interior of a distributed matrix pair as
// the local ones
TYPED_TEST(backend_pairwise_amg, empty_ghost_part_changes_nothing)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T> grid_matrix = transformed(grid<T>(8, 7, 1.0, 0.3, true), [](int i, int j, T v) {
        return i == j && i % 9 == 4 ? T(30) : v;
    });
    const Csr<T> ghost       = without_entries<T>(grid_matrix.n, 3);

    for(int ordering = 0; ordering <= 5; ++ordering)
    {
        for(bool negative : {false, true})
        {
            SCOPED_TRACE("ordering " + std::to_string(ordering) + (negative ? ", negated" : ""));
            const Csr<T> A = negative ? negated(grid_matrix) : grid_matrix;

            Aggregation local, interior;
            ASSERT_IN_LOOP(this->initial(A, 0.25, ordering, &local));
            ASSERT_IN_LOOP(this->initial(A, 0.25, ordering, &interior, &ghost));
            EXPECT_EQ(interior.nc, local.nc);
            EXPECT_EQ(interior.G, local.G);
            EXPECT_EQ(interior.rGsize, local.rGsize);

            const Csr<T> Ac           = galerkin(A, local.G, local.nc);
            const Csr<T> coarse_ghost = without_entries<T>(Ac.n, 3);
            ASSERT_IN_LOOP(this->further(Ac, 0.25, ordering, &local));
            ASSERT_IN_LOOP(this->further(Ac, 0.25, ordering, &interior, &coarse_ghost));
            EXPECT_EQ(interior.nc, local.nc);
            EXPECT_EQ(interior.G, local.G);
            EXPECT_EQ(std::vector<int>(interior.rG, interior.rG + 4 * interior.rGsize),
                      std::vector<int>(local.rG, local.rG + 4 * local.rGsize));
        }
    }
}

// Ghost couplings count in the dominance test of their row
TYPED_TEST(backend_pairwise_amg, ghost_couplings_count_in_the_dominance_test)
{
    using T = typename TypeParam::value_type;
    const Csr<T> A
        = transformed(path<T>(6), [](int i, int j, T v) { return i == j && i == 2 ? T(11) : v; });
    Csr<T> ghost(6, 2);
    for(int i = 0; i < 6; ++i)
    {
        if(i == 2)
        {
            ghost.add(1, -1.0);
        }
        ghost.end_row();
    }

    Aggregation agg;
    ASSERT_TRUE(this->initial(A, 0.25, 0, &agg, &ghost));
    EXPECT_EQ(agg.G, (std::vector<int>{0, 0, 1, 1, 2, 2}));
    EXPECT_EQ(agg.unaggregated(), dominant_rows(A, &ghost));
}

// A ghost coupling weaker than the local ones changes no pairing, a stronger one makes the
// local couplings weak
TYPED_TEST(backend_pairwise_amg, ghost_couplings_count_in_the_strength_threshold)
{
    using T        = typename TypeParam::value_type;
    const Csr<T> A = path<T>(4);

    const Csr<T> weak = column_ghost<T>(4, -0.1);
    Aggregation  agg;
    ASSERT_TRUE(this->initial(A, 0.25, 0, &agg, &weak));
    EXPECT_EQ(agg.G, (std::vector<int>{0, 0, 1, 1}));

    const Csr<T> coarse_weak = column_ghost<T>(2, -0.1);
    ASSERT_TRUE(this->further(galerkin(A, agg.G, agg.nc), 0.25, 0, &agg, &coarse_weak));
    EXPECT_EQ(agg.G, (std::vector<int>{0, 0, 0, 0}));

    KNOWN_BUG("the ghost couplings enter the threshold with their sign, as a maximum");

    const Csr<T> strong = column_ghost<T>(4, -8.0);
    Aggregation  alone;
    ASSERT_TRUE(this->initial(A, 0.25, 0, &alone, &strong));
    EXPECT_EQ(alone.G, (std::vector<int>{0, 1, 2, 3}));
}

// For a negative diagonal the signs of the row, its ghost couplings included, are flipped
TYPED_TEST(backend_pairwise_amg, ghost_couplings_of_a_negated_matrix)
{
    using T = typename TypeParam::value_type;
    for(bool negative : {false, true})
    {
        SCOPED_TRACE(negative ? "negated" : "original");
        const auto   sign  = [&](const Csr<T>& M) { return negative ? negated(M) : M; };
        const Csr<T> A     = sign(path<T>(4));
        const Csr<T> ghost = sign(column_ghost<T>(4, -0.1));
        Aggregation  agg;
        ASSERT_TRUE(this->initial(A, 0.25, 0, &agg, &ghost));
        EXPECT_EQ(agg.G, (std::vector<int>{0, 0, 1, 1}));

        const Csr<T> coarse_ghost = sign(column_ghost<T>(2, -0.1));
        ASSERT_TRUE(this->further(galerkin(A, agg.G, agg.nc), 0.25, 0, &agg, &coarse_ghost));
        EXPECT_EQ(agg.G, (std::vector<int>{0, 0, 0, 0}));
        EXPECT_EQ(agg.nc, 1);
    }
}

// Node 2 of M couples positively to 3 and 3 only positively to 2, so neither pairs with
// the other, locally or with ghost couplings
TYPED_TEST(backend_pairwise_amg, positive_couplings_are_never_strong)
{
    using T = typename TypeParam::value_type;
    Csr<T> M(4);
    M.add(0, 1.0);
    M.add(1, -1.0);
    M.end_row();
    M.add(0, -1.0);
    M.add(1, 2.0);
    M.add(2, -1.0);
    M.end_row();
    M.add(1, -1.0);
    M.add(2, 2.0);
    M.add(3, 0.5);
    M.end_row();
    M.add(2, 0.5);
    M.add(3, 1.0);
    M.end_row();
    const Csr<T> ghost         = column_ghost<T>(4, -0.1);
    const Csr<T> A             = doubled(M);
    const Csr<T> doubled_ghost = column_ghost<T>(A.n, -0.1);

    for(bool with_ghost : {false, true})
    {
        SCOPED_TRACE(with_ghost ? "with ghost couplings" : "local");

        Aggregation initial;
        ASSERT_TRUE(this->initial(M, 0.25, 0, &initial, with_ghost ? &ghost : nullptr));
        EXPECT_EQ(initial.nc, 3);
        EXPECT_EQ(initial.G, (std::vector<int>{0, 0, 1, 2}));

        Aggregation agg;
        ASSERT_TRUE(this->initial(A, 0.25, 0, &agg, with_ghost ? &doubled_ghost : nullptr));
        ASSERT_EQ(agg.G, (std::vector<int>{0, 0, 1, 1, 2, 2, 3, 3}));
        ASSERT_EQ(dense(galerkin(A, agg.G, agg.nc)), dense(M));

        ASSERT_TRUE(this->further(M, 0.25, 0, &agg, with_ghost ? &ghost : nullptr));
        expect_valid_pairwise(A, agg, 4);
        EXPECT_EQ(agg.nc, 3);
        EXPECT_EQ(agg.G, (std::vector<int>{0, 0, 0, 0, 1, 1, 2, 2}));
    }
}

// Coarse node 0 stores no diagonal, as an aggregate of a zero row sum can
TYPED_TEST(backend_pairwise_amg, further_pass_of_a_row_without_diagonal)
{
    using T = typename TypeParam::value_type;
    Csr<T> coarse(3);
    coarse.add(1, -1.0);
    coarse.end_row();
    coarse.add(0, -1.0);
    coarse.add(1, 2.0);
    coarse.add(2, -1.0);
    coarse.end_row();
    coarse.add(1, -1.0);
    coarse.add(2, 1.0);
    coarse.end_row();
    const Csr<T> A     = doubled(coarse);
    const Csr<T> ghost = without_entries<T>(coarse.n, 3);

    for(bool with_ghost : {false, true})
    {
        SCOPED_TRACE(with_ghost ? "with ghost part" : "local");
        Aggregation agg;
        ASSERT_TRUE(this->initial(A, 0.25, 0, &agg));
        ASSERT_EQ(agg.G, (std::vector<int>{0, 0, 1, 1, 2, 2}));

        ASSERT_TRUE(this->further(coarse, 0.25, 0, &agg, with_ghost ? &ghost : nullptr));
        expect_valid_pairwise(A, agg, 4);
        EXPECT_EQ(agg.nc, 2);
        EXPECT_EQ(agg.G, (std::vector<int>{0, 0, 0, 0, 1, 1}));
    }
}

// Fine node 0 stores no diagonal, so a_00 = 0: the row is not dominant, its couplings keep
// their signs, and 0 pairs with 1 (-1 < -0.25 max_k |a_0k|)
TYPED_TEST(backend_pairwise_amg, initial_pass_of_a_row_without_diagonal)
{
    using T        = typename TypeParam::value_type;
    const Csr<T> P = path<T>(6);
    Csr<T>       A(P.n);
    for(int i = 0; i < P.n; ++i)
    {
        for(PtrType j = P.row_offset[i]; j < P.row_offset[i + 1]; ++j)
        {
            if(i != 0 || P.col[j] != 0)
            {
                A.add(P.col[j], P.val[j]);
            }
        }
        A.end_row();
    }
    const Csr<T> ghost = without_entries<T>(A.n, 3);

    KNOWN_BUG("the initial pass reads a_ii at a diagonal index that a row without diagonal "
              "never sets, which crashes");
    expect_in_child([&] {
        for(bool with_ghost : {false, true})
        {
            SCOPED_TRACE(with_ghost ? "with ghost part" : "local");
            Aggregation agg;
            ASSERT_TRUE(this->initial(A, 0.25, 0, &agg, with_ghost ? &ghost : nullptr));
            expect_valid_pairwise(A, agg, 2);
            EXPECT_EQ(agg.G, (std::vector<int>{0, 0, 1, 1, 2, 2}));
        }
    });
}

namespace
{
    template <typename P>
    class backend_pairwise_amg_fallback : public backend_pairwise_amg<P>
    {
    };
}

TYPED_TEST_SUITE(backend_pairwise_amg_fallback, AcceleratorOnly<RealTypes>, TypedName);

// An operation that the backend does not implement returns false and changes nothing, so
// that LocalMatrix can retry on the host
TYPED_TEST(backend_pairwise_amg_fallback, unimplemented_operations_return_false_and_change_nothing)
{
    using T = typename TypeParam::value_type;

    const Csr<T> A = path<T>(6);
    auto         M = this->make(A);
    ASSERT_NE(M, nullptr);
    auto ghost = empty_matrix<T>(TypeParam::backend, csr_format);

    for(bool with_ghost : {false, true})
    {
        SCOPED_TRACE(with_ghost ? "with a ghost part" : "local");

        int  nc = -7, Gsize = -7, rGsize = -7;
        int* rG = nullptr;
        auto G  = make_vector(TypeParam::backend, std::vector<int>(A.n, 7));
        EXPECT_FALSE(
            with_ghost
                ? M->InitialPairwiseAggregation(*ghost, 0.25, nc, G.get(), Gsize, &rG, rGsize, 0)
                : M->InitialPairwiseAggregation(0.25, nc, G.get(), Gsize, &rG, rGsize, 0));
        EXPECT_EQ(nc, -7);
        EXPECT_EQ(Gsize, -7);
        EXPECT_EQ(rGsize, -7);
        EXPECT_EQ(rG, nullptr);
        EXPECT_EQ(values_of(*G), std::vector<int>(A.n, 7));

        Gsize  = 2;
        rGsize = 3;
        rocalution::allocate_host(6, &rG);
        const std::vector<int> pairs = {0, 2, 4, 1, 3, 5};
        std::copy(pairs.begin(), pairs.end(), rG);
        int* const before = rG;
        auto       Gc     = make_vector(TypeParam::backend, std::vector<int>{0, 0, 1, 1, 2, 2});
        EXPECT_FALSE(
            with_ghost
                ? M->FurtherPairwiseAggregation(*ghost, 0.25, nc, Gc.get(), Gsize, &rG, rGsize, 0)
                : M->FurtherPairwiseAggregation(0.25, nc, Gc.get(), Gsize, &rG, rGsize, 0));
        EXPECT_EQ(nc, -7);
        EXPECT_EQ(Gsize, 2);
        EXPECT_EQ(rGsize, 3);
        EXPECT_EQ(rG, before);
        EXPECT_EQ(std::vector<int>(rG, rG + 6), pairs);
        EXPECT_EQ(values_of(*Gc), (std::vector<int>{0, 0, 1, 1, 2, 2}));

        auto Ac = this->make(path<T>(3));
        EXPECT_FALSE(M->CoarsenOperator(Ac.get(), 3, 3, *Gc, Gsize, rG, rGsize));
        expect_near(csr_of(*Ac), path<T>(3), 0.0);
        rocalution::free_host(&rG);
    }
    expect_near(csr_of(*M), A, 0.0);
}
