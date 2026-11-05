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

// The single steps of the AMG setup on the backend matrices, whose results are defined
// by their inputs alone: strength of connection, the PMIS aggregation and coarsening
// sweeps, and the parts of the interpolations that are not the interpolation formula.
// The arguments for the distributed setup are passed as LocalMatrix passes them: an
// empty ghost matrix, unallocated boundary and l2g vectors, and no ghost prolongation.

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "unit_test_amg_reference.hpp"
#include "unit_test_common.hpp"
#include "unit_test_reference.hpp"
#include "unit_test_types.hpp"

using namespace rocalution_unit;

namespace
{
    const Format csr_format{rocalution::CSR, 1};

    // make_vector and values_of hold their values in a std::vector, which has no data()
    // for bool
    std::unique_ptr<BaseVector<bool>> make_flags(Backend b, const std::vector<bool>& values)
    {
        std::unique_ptr<bool[]> data(new bool[values.size()]);
        std::copy(values.begin(), values.end(), data.get());

        auto host = std::make_unique<HostVector<bool>>(descriptor());
        host->Allocate(values.size());
        if(!values.empty())
        {
            host->CopyFromData(data.get());
        }
        if(b == Backend::Host)
        {
            return host;
        }

        std::unique_ptr<rocalution::AcceleratorVector<bool>> accel(
            rocalution::_rocalution_init_base_backend_vector<bool>(descriptor()));
        accel->Allocate(values.size());
        accel->CopyFromHost(*host);
        return accel;
    }

    std::vector<bool> flags_of(const BaseVector<bool>& v)
    {
        HostVector<bool> host(descriptor());
        host.Allocate(v.GetSize());
        if(auto accel = dynamic_cast<const rocalution::AcceleratorVector<bool>*>(&v))
        {
            accel->CopyToHost(&host);
        }
        else
        {
            host.CopyFrom(v);
        }

        std::unique_ptr<bool[]> data(new bool[v.GetSize()]);
        if(v.GetSize() > 0)
        {
            host.CopyToData(data.get());
        }
        return std::vector<bool>(data.get(), data.get() + v.GetSize());
    }

    // Path 0 - 1 - ... - (n - 1) with couplings -1 and zero row sums, followed by
    // isolated nodes that store only a diagonal
    template <typename T>
    Csr<T> path(int n, int isolated = 0)
    {
        Csr<T> A(n + isolated);
        for(int i = 0; i < n; ++i)
        {
            for(int j = i - 1; j <= i + 1; ++j)
            {
                if(j >= 0 && j < n)
                {
                    A.add(j, j == i ? (i == 0 || i == n - 1 ? 1.0 : 2.0) : -1.0);
                }
            }
            A.end_row();
        }
        for(int i = n; i < n + isolated; ++i)
        {
            A.add(i, 1.0);
            A.end_row();
        }
        return A;
    }

    template <typename T>
    Csr<T> diagonal_matrix(int n)
    {
        return path<T>(0, n);
    }

    template <typename T>
    std::vector<T> diagonal_of(const Csr<T>& A)
    {
        std::vector<T> d(A.n);
        for(int i = 0; i < A.n; ++i)
        {
            d[i] = entry(A, i, i);
        }
        return d;
    }

    template <typename P>
    class backend_amg : public TypedBackendTest<P>
    {
    protected:
        using T = typename P::value_type;

        static Backend backend()
        {
            return P::backend;
        }

        std::unique_ptr<BaseMatrix<T>> make(const Csr<T>& ref) const
        {
            return make_matrix(backend(), csr_format, *ref.host_csr());
        }

        // The ghost part of a matrix without neighbours
        std::unique_ptr<BaseMatrix<T>> ghost() const
        {
            return empty_matrix<T>(backend(), csr_format);
        }

        template <typename V>
        std::unique_ptr<BaseVector<V>> vec(const std::vector<V>& values) const
        {
            return make_vector(backend(), values);
        }

        std::unique_ptr<BaseVector<bool>> flags(const std::vector<bool>& values) const
        {
            return make_flags(backend(), values);
        }

        // Unallocated, for the arguments of the distributed setup
        template <typename V>
        std::unique_ptr<BaseVector<V>> none() const
        {
            if(backend() == Backend::Host)
            {
                return std::make_unique<HostVector<V>>(descriptor());
            }
            return std::unique_ptr<BaseVector<V>>(
                rocalution::_rocalution_init_base_backend_vector<V>(descriptor()));
        }

        // Connections of the SA setup with the diagonal of A
        std::vector<bool> strong_connections(const Csr<T>& ref, double eps) const
        {
            auto A    = make(ref);
            auto diag = vec(diagonal_of(ref));
            auto conn = flags(std::vector<bool>(ref.nnz(), true));
            EXPECT_TRUE(
                A->AMGComputeStrongConnections(eps, *diag, *none<int64_t>(), conn.get(), *ghost()));
            return flags_of(*conn);
        }

        // One PMIS sweep from state with max_state = state, as LocalMatrix calls it
        bool find_max_neighbour_node(const Csr<T>&            ref,
                                     const std::vector<bool>& conn,
                                     const std::vector<int>&  state,
                                     const std::vector<int>&  hash,
                                     std::vector<int>*        max_state,
                                     std::vector<int64_t>*    aggregates) const
        {
            auto A   = make(ref);
            auto c   = flags(conn);
            auto s   = vec(state);
            auto h   = vec(hash);
            auto ms  = vec(state);
            auto agg = vec(*aggregates);

            bool undecided = false;
            EXPECT_TRUE(A->AMGPMISFindMaxNeighbourNode(0,
                                                       ref.n,
                                                       undecided,
                                                       *c,
                                                       *s,
                                                       *h,
                                                       *none<PtrType>(),
                                                       *none<int64_t>(),
                                                       *none<int>(),
                                                       *none<int>(),
                                                       ms.get(),
                                                       agg.get(),
                                                       *ghost()));
            *max_state  = values_of(*ms);
            *aggregates = values_of(*agg);
            return undecided;
        }

        // The ext+i style steps index the C/F map by column, so A is square
        std::vector<bool> strong_influences(const Csr<T>&       ref,
                                            float               eps,
                                            int64_t             offset,
                                            std::vector<float>* omega) const
        {
            auto A = make(ref);
            auto S = flags(std::vector<bool>(ref.nnz(), true));
            auto w = vec(std::vector<float>(ref.n, -1.0f));
            EXPECT_TRUE(A->RSPMISStrongInfluences(eps, S.get(), w.get(), offset, *ghost()));
            if(omega != nullptr)
            {
                *omega = values_of(*w);
            }
            return flags_of(*S);
        }
    };
}

TYPED_TEST_SUITE(backend_amg, WithBackends<RealTypes>, TypedName);

// |a_ij| > eps sqrt(d_i d_j) for the given diagonal d, never on the diagonal
TYPED_TEST(backend_amg, strong_connections_compare_couplings_with_the_diagonal)
{
    using T = typename TypeParam::value_type;
    Csr<T> ref(3);
    ref.add(0, 4.0);
    ref.add(1, -1.0);
    ref.add(2, -0.5);
    ref.end_row();
    ref.add(0, -1.0);
    ref.add(1, 4.0);
    ref.add(2, -2.0);
    ref.end_row();
    ref.add(0, -0.5);
    ref.add(1, -2.0);
    ref.add(2, 1.0);
    ref.end_row();

    // With eps = 0.25 the couplings of 0 lie exactly on the threshold, |-1| = 0.25 * 4
    // and |-0.5| = 0.25 * 2
    EXPECT_EQ(this->strong_connections(ref, 0.25),
              (std::vector<bool>{false, false, false, false, false, true, false, true, false}));
    EXPECT_EQ(this->strong_connections(ref, 0.2),
              (std::vector<bool>{false, true, true, true, false, true, true, true, false}));
}

TYPED_TEST(backend_amg, strong_connections_of_an_anisotropic_grid)
{
    using T           = typename TypeParam::value_type;
    const Csr<T> ref  = grid<T>(8, 6, 1.0, 0.01, true);
    const auto   conn = this->strong_connections(ref, 0.08);
    EXPECT_EQ(conn, sa_strength(ref, 0.08));
    EXPECT_NE(conn, off_diagonal(ref));
}

namespace
{
    // Couplings -1 to -4 with zero row sums, so that about half of them are strong
    template <typename T>
    Csr<T> weighted_band(int n, int bandwidth)
    {
        Csr<T> A(n);
        for(int i = 0; i < n; ++i)
        {
            double sum = 0.0;
            for(int j = std::max(0, i - bandwidth); j <= std::min(n - 1, i + bandwidth); ++j)
            {
                sum += j == i ? 0.0 : 1.0 + (i + j) % 4;
            }
            for(int j = std::max(0, i - bandwidth); j <= std::min(n - 1, i + bandwidth); ++j)
            {
                A.add(j, j == i ? sum : -(1.0 + (i + j) % 4));
            }
            A.end_row();
        }
        return A;
    }

    struct RowBin
    {
        int bandwidth;
        int above;
        int up_to;
    };

    // The HIP strong connections and PMIS sweeps pick their kernel by nnz(A) / n
    const std::vector<RowBin> row_bins = {{2, 0, 8},
                                          {5, 8, 16},
                                          {12, 16, 32},
                                          {25, 32, 64},
                                          {50, 64, 128},
                                          {100, 128, 256},
                                          {160, 256, 500}};
}

TYPED_TEST(backend_amg, strong_connections_in_every_row_length_bin)
{
    using T = typename TypeParam::value_type;
    for(const RowBin& bin : row_bins)
    {
        SCOPED_TRACE("bandwidth " + std::to_string(bin.bandwidth));
        const Csr<T> ref = weighted_band<T>(500, bin.bandwidth);
        ASSERT_GT(ref.nnz() / ref.n, bin.above);
        ASSERT_LE(ref.nnz() / ref.n, bin.up_to);

        const double eps  = 2.45 / (5.0 * bin.bandwidth);
        const auto   conn = this->strong_connections(ref, eps);
        EXPECT_EQ(conn, sa_strength(ref, eps));
        EXPECT_NE(std::count(conn.begin(), conn.end(), true), 0);
        EXPECT_NE(conn, off_diagonal(ref));
    }
}

// State 0 for nodes with a strong connection, -2 for the others, and a hash of the global
// index
TYPED_TEST(backend_amg, pmis_initial_state)
{
    using T                     = typename TypeParam::value_type;
    const Csr<T>            ref = path<T>(4, 1);
    const std::vector<bool> conn
        = {false, true, false, false, false, false, false, true, true, false, false};

    auto A = this->make(ref);
    auto c = this->flags(conn);
    auto s = this->vec(std::vector<int>(ref.n, 7));
    auto h = this->vec(std::vector<int>(ref.n, 7));
    ASSERT_TRUE(A->AMGPMISInitializeState(0, *c, s.get(), h.get(), *this->ghost()));
    EXPECT_EQ(values_of(*s), (std::vector<int>{0, -2, 0, 0, -2}));
    const std::vector<int> hash = values_of(*h);

    auto shifted = this->vec(std::vector<int>(ref.n, 7));
    ASSERT_TRUE(A->AMGPMISInitializeState(2, *c, s.get(), shifted.get(), *this->ghost()));
    const std::vector<int> hash2 = values_of(*shifted);
    for(int i = 0; i + 2 < ref.n; ++i)
    {
        EXPECT_EQ(hash2[i], hash[i + 2]) << "node " << i;
    }
}

namespace
{
    // Path 0 - ... - 7 and the isolated node 8, with distinct hashes. The local maxima
    // 1 and 5 are also the maxima of their distance 2 neighbourhoods.
    const std::vector<int> path_hash = {50, 90, 30, 10, 20, 70, 40, 0, 60};
}

// A node becomes a root (state 1, aggregate 1) if its (state, hash) is the largest within
// distance 2. Nodes with a decided state are left alone.
TYPED_TEST(backend_amg, pmis_sweep_selects_local_maxima)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = path<T>(8, 1);
    std::vector<int>     max_state;
    std::vector<int64_t> aggregates(ref.n, -7);
    const bool           undecided = this->find_max_neighbour_node(
        ref, off_diagonal(ref), {0, 0, 0, 0, 0, 0, 0, 0, -2}, path_hash, &max_state, &aggregates);

    EXPECT_TRUE(undecided);
    EXPECT_EQ(max_state, (std::vector<int>{0, 1, 0, 0, 0, 1, 0, 0, -2}));
    EXPECT_EQ(aggregates[1], 1);
    EXPECT_EQ(aggregates[5], 1);
    EXPECT_EQ(aggregates[8], -7);
}

// A node within distance 2 of a root is excluded (state -1, aggregate 0)
TYPED_TEST(backend_amg, pmis_sweep_excludes_the_neighbourhood_of_roots)
{
    using T                      = typename TypeParam::value_type;
    const Csr<T>           ref   = path<T>(8, 1);
    const std::vector<int> state = {0, 1, 0, 0, 0, 1, 0, 0, -2};
    std::vector<int>       max_state;
    std::vector<int64_t>   aggregates(ref.n, -7);
    const bool             undecided = this->find_max_neighbour_node(
        ref, off_diagonal(ref), state, path_hash, &max_state, &aggregates);

    EXPECT_FALSE(undecided);
    EXPECT_EQ(max_state, (std::vector<int>{-1, 1, -1, -1, -1, 1, -1, -1, -2}));
    EXPECT_EQ(aggregates, (std::vector<int64_t>{0, -7, 0, 0, 0, -7, 0, 0, -7}));
}

// Every sweep from the initial state lies between the PMIS of distance 1 and of
// distance 2: a node that is the maximum within distance 2 becomes a root, and a root is
// the maximum within distance 1
TYPED_TEST(backend_amg, pmis_sweep_in_every_row_length_bin)
{
    using T = typename TypeParam::value_type;
    for(const RowBin& bin : row_bins)
    {
        SCOPED_TRACE("bandwidth " + std::to_string(bin.bandwidth));
        const Csr<T>            ref  = weighted_band<T>(500, bin.bandwidth);
        const std::vector<bool> conn = sa_strength(ref, 2.45 / (5.0 * bin.bandwidth));

        std::vector<int> state(ref.n, -2);
        std::vector<int> hash(ref.n);
        for(int i = 0; i < ref.n; ++i)
        {
            for(PtrType j = ref.row_offset[i]; j < ref.row_offset[i + 1]; ++j)
            {
                state[i] = conn[j] ? 0 : state[i];
            }
            hash[i] = (i * 389) % ref.n;
        }

        std::vector<int>     max_state;
        std::vector<int64_t> aggregates(ref.n, -7);
        const bool           undecided
            = this->find_max_neighbour_node(ref, conn, state, hash, &max_state, &aggregates);

        // The largest hash of the undecided nodes within the given distance of i
        auto largest = [&](int i, int depth) {
            std::vector<bool> seen(ref.n, false);
            std::vector<int>  front = {i};
            int               best  = hash[i];
            seen[i]                 = true;
            for(int d = 0; d < depth; ++d)
            {
                std::vector<int> next;
                for(int k : front)
                {
                    for(PtrType j = ref.row_offset[k]; j < ref.row_offset[k + 1]; ++j)
                    {
                        const int c = ref.col[j];
                        if(conn[j] && !seen[c])
                        {
                            seen[c] = true;
                            next.push_back(c);
                            best = state[c] == 0 ? std::max(best, hash[c]) : best;
                        }
                    }
                }
                front = next;
            }
            return best;
        };

        bool any_undecided = false;
        int  roots         = 0;
        for(int i = 0; i < ref.n; ++i)
        {
            if(state[i] != 0)
            {
                EXPECT_EQ(max_state[i], state[i]) << "node " << i;
                continue;
            }
            ASSERT_TRUE(max_state[i] == 0 || max_state[i] == 1) << "node " << i;
            any_undecided |= max_state[i] == 0;
            if(max_state[i] == 1)
            {
                ++roots;
                EXPECT_EQ(aggregates[i], 1) << "node " << i;
                EXPECT_EQ(largest(i, 1), hash[i]) << "node " << i;
            }
            else
            {
                EXPECT_NE(largest(i, 2), hash[i]) << "node " << i;
            }
        }
        EXPECT_GT(roots, 0);
        EXPECT_EQ(undecided, any_undecided);
    }
}

// Two strongly connected nodes of equal state and hash cannot both be roots. hash1 maps
// two indices to every hash value, so ties occur.
TYPED_TEST(backend_amg, pmis_sweep_breaks_ties)
{
    using T                  = typename TypeParam::value_type;
    const Csr<T>         ref = path<T>(2);
    std::vector<int>     max_state;
    std::vector<int64_t> aggregates(ref.n, -7);
    this->find_max_neighbour_node(ref, off_diagonal(ref), {0, 0}, {7, 7}, &max_state, &aggregates);

    EXPECT_NE(max_state, (std::vector<int>{0, 0}));
    KNOWN_BUG("lexographical_max does not compare the node index");
    EXPECT_NE(max_state, (std::vector<int>{1, 1}));
}

// Roots hold their global index, all other nodes -1
TYPED_TEST(backend_amg, pmis_root_indices)
{
    using T                               = typename TypeParam::value_type;
    const Csr<T>               ref        = path<T>(6);
    const std::vector<int64_t> aggregates = {1, 0, 0, 1, 0, 1};
    for(int64_t begin : {0, 100})
    {
        SCOPED_TRACE("begin " + std::to_string(begin));
        auto A     = this->make(ref);
        auto agg   = this->vec(aggregates);
        auto roots = this->vec(std::vector<int64_t>(ref.n, 7));
        ASSERT_TRUE(A->AMGPMISInitializeAggregateGlobalIndices(begin, *agg, roots.get()));
        EXPECT_EQ(values_of(*roots),
                  (std::vector<int64_t>{begin, -1, -1, begin + 3, -1, begin + 5}));
    }
}

// An excluded node (state -1) joins the aggregate of a strongly connected node with state
// 1 and takes state 1, nodes without strong connections (state -2) get aggregate -2
TYPED_TEST(backend_amg, pmis_unassigned_nodes_join_a_neighbouring_aggregate)
{
    using T                      = typename TypeParam::value_type;
    const Csr<T>            ref  = path<T>(8, 1);
    const std::vector<bool> conn = off_diagonal(ref);

    auto A     = this->make(ref);
    auto c     = this->flags(conn);
    auto agg   = this->vec(std::vector<int64_t>{0, 0, 1, 1, 1, 1, 2, 2, 2});
    auto roots = this->vec(std::vector<int64_t>{-1, 1, -1, -1, -1, 5, -1, -1, -1});

    std::vector<int> state = {-1, 1, -1, -1, -1, 1, -1, -1, -2};
    auto             ms    = this->vec(state);
    ASSERT_TRUE(A->AMGPMISAddUnassignedNodesToAggregations(0,
                                                           *c,
                                                           *this->vec(state),
                                                           *this->template none<int64_t>(),
                                                           ms.get(),
                                                           agg.get(),
                                                           roots.get(),
                                                           *this->ghost()));

    // 3 and 7 have no neighbour of state 1 yet
    state = values_of(*ms);
    EXPECT_EQ(state, (std::vector<int>{1, 1, 1, -1, 1, 1, 1, -1, -2}));
    std::vector<int64_t> a = values_of(*agg);
    std::vector<int64_t> r = values_of(*roots);
    for(int i : {0, 1, 2})
    {
        EXPECT_EQ(a[i], 0) << "node " << i;
        EXPECT_EQ(r[i], 1) << "node " << i;
    }
    for(int i : {4, 5, 6})
    {
        EXPECT_EQ(a[i], 1) << "node " << i;
        EXPECT_EQ(r[i], 5) << "node " << i;
    }
    EXPECT_EQ(a[8], -2);

    ASSERT_TRUE(A->AMGPMISAddUnassignedNodesToAggregations(0,
                                                           *c,
                                                           *this->vec(state),
                                                           *this->template none<int64_t>(),
                                                           ms.get(),
                                                           agg.get(),
                                                           roots.get(),
                                                           *this->ghost()));
    EXPECT_EQ(values_of(*ms), (std::vector<int>{1, 1, 1, 1, 1, 1, 1, 1, -2}));
    a = values_of(*agg);
    r = values_of(*roots);
    EXPECT_TRUE((a[3] == 0 && r[3] == 1) || (a[3] == 1 && r[3] == 5))
        << "node 3 in aggregate " << a[3] << " with root " << r[3];
    EXPECT_EQ(a[7], 1);
    EXPECT_EQ(r[7], 5);
    EXPECT_EQ(a[8], -2);
}

// j influences i if a_ij < eps min_k a_ik, never on the diagonal. omega counts the rows
// that i influences, plus a random number in [0, 1).
TYPED_TEST(backend_amg, rs_strong_influences_compare_with_the_most_negative_coupling)
{
    using T = typename TypeParam::value_type;
    Csr<T> ref(4);
    ref.add(0, 4.0);
    ref.add(1, -1.0);
    ref.add(2, -0.2);
    ref.add(3, 0.5);
    ref.end_row();
    ref.add(0, -1.0);
    ref.add(1, 3.0);
    ref.add(2, -1.0);
    ref.end_row();
    ref.add(0, -0.2);
    ref.add(1, -1.0);
    ref.add(2, 2.0);
    ref.add(3, -0.25);
    ref.end_row();
    ref.add(0, 0.5);
    ref.add(2, -0.25);
    ref.add(3, 1.0);
    ref.end_row();

    // a_23 = -0.25 lies exactly on the threshold 0.25 * -1
    std::vector<float> omega;
    EXPECT_EQ(this->strong_influences(ref, 0.25f, 0, &omega),
              (std::vector<bool>{false,
                                 true,
                                 false,
                                 false,
                                 true,
                                 false,
                                 true,
                                 false,
                                 true,
                                 false,
                                 false,
                                 false,
                                 true,
                                 false}));
    EXPECT_EQ(this->strong_influences(ref, 0.25f, 0, nullptr), rs_strength(ref, 0.25));

    ASSERT_EQ(omega.size(), 4u);
    const std::vector<int> influenced = {1, 2, 2, 0};
    for(int i = 0; i < 4; ++i)
    {
        EXPECT_GE(omega[i], static_cast<float>(influenced[i])) << "node " << i;
        EXPECT_LT(omega[i], static_cast<float>(influenced[i] + 1)) << "node " << i;
    }
}

// The random part of omega is a function of the global row
TYPED_TEST(backend_amg, rs_strong_influences_random_part)
{
    using T                = typename TypeParam::value_type;
    const Csr<T>       ref = diagonal_matrix<T>(6);
    std::vector<float> omega, shifted;
    EXPECT_EQ(this->strong_influences(ref, 0.25f, 0, &omega), std::vector<bool>(ref.nnz(), false));
    this->strong_influences(ref, 0.25f, 2, &shifted);
    for(int i = 0; i < ref.n; ++i)
    {
        EXPECT_GE(omega[i], 0.0f) << "node " << i;
        EXPECT_LT(omega[i], 1.0f) << "node " << i;
        if(i + 2 < ref.n)
        {
            EXPECT_EQ(shifted[i], omega[i + 2]) << "node " << i;
        }
    }
}

// Strength is invariant under scaling a row by -1, so it treats a negative diagonal
// with the sign of the couplings reversed
TYPED_TEST(backend_amg, rs_strong_influences_of_a_grid_and_its_negation)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = grid<T>(8, 6, 1.0, 0.1, true);
    const auto   S   = this->strong_influences(ref, 0.25f, 0, nullptr);
    EXPECT_EQ(S, rs_strength(ref, 0.25));
    EXPECT_NE(S, off_diagonal(ref));

    Csr<T> negated = ref;
    for(T& v : negated.val)
    {
        v = -v;
    }
    const auto S_negated = this->strong_influences(negated, 0.25f, 0, nullptr);
    KNOWN_BUG("for a negative diagonal, a_ij < eps max_k a_ik is taken as strong");
    EXPECT_EQ(S_negated, S);
}

// Undecided points (0) become C points (1) and are marked if they influence another
// point, omega >= 1, and F points (2) otherwise
TYPED_TEST(backend_amg, rs_unassigned_to_coarse)
{
    using T             = typename TypeParam::value_type;
    const Csr<T> ref    = diagonal_matrix<T>(6);
    auto         A      = this->make(ref);
    auto         cf     = this->vec(std::vector<int>{0, 0, 0, 1, 2, 0});
    auto         marked = this->flags(std::vector<bool>(ref.n, true));
    auto         omega  = this->vec(std::vector<float>{1.5f, 0.3f, 2.0f, 5.0f, 5.0f, 0.999f});

    ASSERT_TRUE(A->RSPMISUnassignedToCoarse(cf.get(), marked.get(), *omega));
    EXPECT_EQ(values_of(*cf), (std::vector<int>{1, 2, 1, 1, 2, 2}));
    EXPECT_EQ(flags_of(*marked), (std::vector<bool>{true, false, true, false, false, false}));
}

// Of two marked points connected in either direction, the one with the smaller omega
// becomes undecided again
TYPED_TEST(backend_amg, rs_correct_coarse_keeps_the_larger_measure)
{
    using T                = typename TypeParam::value_type;
    const Csr<T>      ref  = path<T>(7);
    std::vector<bool> S    = off_diagonal(ref);
    S[position(ref, 4, 3)] = false;
    S[position(ref, 4, 5)] = false;

    auto A      = this->make(ref);
    auto cf     = this->vec(std::vector<int>{1, 1, 1, 1, 1, 1, 2});
    auto marked = this->flags({true, true, false, true, true, true, false});
    auto omega  = this->vec(std::vector<float>{3.1f, 2.2f, 0.5f, 2.7f, 1.9f, 1.4f, 9.0f});

    ASSERT_TRUE(A->RSPMISCorrectCoarse(cf.get(), *this->flags(S), *marked, *omega, *this->ghost()));
    EXPECT_EQ(values_of(*cf), (std::vector<int>{1, 0, 1, 1, 0, 0, 2}));
}

// An undecided point that depends strongly on a C point becomes an F point
TYPED_TEST(backend_amg, rs_coarse_edges_to_fine)
{
    using T                = typename TypeParam::value_type;
    const Csr<T>      ref  = path<T>(7);
    std::vector<bool> S    = off_diagonal(ref);
    S[position(ref, 4, 5)] = false;

    auto A  = this->make(ref);
    auto cf = this->vec(std::vector<int>{1, 0, 0, 2, 0, 1, 0});
    ASSERT_TRUE(A->RSPMISCoarseEdgesToFine(cf.get(), *this->flags(S), *this->ghost()));
    EXPECT_EQ(values_of(*cf), (std::vector<int>{1, 2, 0, 2, 0, 1, 2}));
}

// LocalMatrix passes the flag uninitialized
TYPED_TEST(backend_amg, rs_check_undecided)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = diagonal_matrix<T>(3);
    auto         A   = this->make(ref);
    for(bool initial : {false, true})
    {
        bool undecided = initial;
        ASSERT_TRUE(A->RSPMISCheckUndecided(undecided, *this->vec(std::vector<int>{1, 0, 2})));
        EXPECT_TRUE(undecided);

        undecided = initial;
        ASSERT_TRUE(A->RSPMISCheckUndecided(undecided, *this->vec(std::vector<int>{1, 2, 2})));
        EXPECT_FALSE(undecided);
    }
}

// Linear interpolation from the C neighbours on a path. An F neighbour adds its
// coupling to the C neighbour.
TYPED_TEST(backend_amg, rs_direct_interpolation_of_a_path)
{
    using T = typename TypeParam::value_type;
    struct Case
    {
        std::vector<int> cf;
        int              nc;
        std::vector<T>   P;
    };
    const std::vector<Case> cases
        = {{{1, 2, 1, 2, 1}, 3, {1, 0, 0, 0.5, 0.5, 0, 0, 1, 0, 0, 0.5, 0.5, 0, 0, 1}},
           {{1, 2, 2, 1}, 2, {1, 0, 1, 0, 0, 1, 0, 1}}};

    for(const Case& c : cases)
    {
        const Csr<T> ref  = path<T>(static_cast<int>(c.cf.size()));
        auto         A    = this->make(ref);
        auto         cf   = this->vec(c.cf);
        auto         S    = this->flags(off_diagonal(ref));
        auto         f2c  = this->vec(std::vector<int>(ref.n + 1, 0));
        auto         amin = this->vec(std::vector<T>(ref.n, 0.0));
        auto         amax = this->vec(std::vector<T>(ref.n, 0.0));
        auto         P    = empty_matrix<T>(TypeParam::backend, csr_format);

        ASSERT_TRUE(A->RSDirectProlongNnz(
            *cf, *S, *this->ghost(), amin.get(), amax.get(), f2c.get(), P.get(), nullptr));
        ASSERT_TRUE(A->RSDirectProlongFill(*this->template none<int64_t>(),
                                           *f2c,
                                           *cf,
                                           *S,
                                           *this->ghost(),
                                           *amin,
                                           *amax,
                                           P.get(),
                                           nullptr,
                                           nullptr));

        const Csr<T> result = csr_of(*P);
        ASSERT_EQ(result.n, ref.n);
        ASSERT_EQ(result.m, c.nc);
        expect_near(dense(result), c.P, tol<T>());
    }
}

// Strong C couplings within 0.2 times the most negative or the most positive one are
// dropped, and the kept ones of each sign are scaled up by the sum they lost. Positive
// couplings without a positive strong C coupling are added to the diagonal.
//   F row 1: -2 kept, -0.2 dropped, weak -0.8
//   F row 3: -2 and +1 kept, +0.1 dropped, strong +0.3 to the F point 5
//   F row 5: -2 kept, weak +0.5 lumped
TYPED_TEST(backend_amg, rs_direct_interpolation_of_both_signs)
{
    using T = typename TypeParam::value_type;
    Csr<T> ref(7);
    for(const auto& row : std::vector<std::vector<std::pair<int, double>>>{
            {{0, 1.0}},
            {{0, -2.0}, {1, 3.0}, {2, -0.2}, {3, -0.8}},
            {{2, 1.0}},
            {{2, -2.0}, {3, 0.9}, {4, 1.0}, {5, 0.3}, {6, 0.1}},
            {{4, 1.0}},
            {{4, -2.0}, {5, 1.5}, {6, 0.5}},
            {{6, 1.0}}})
    {
        for(const auto& e : row)
        {
            ref.add(e.first, e.second);
        }
        ref.end_row();
    }
    const std::vector<bool> S
        = flags_at(ref, {{1, 0}, {1, 2}, {3, 2}, {3, 4}, {3, 5}, {3, 6}, {5, 4}});

    auto A    = this->make(ref);
    auto cf   = this->vec(std::vector<int>{1, 2, 1, 2, 1, 2, 1});
    auto s    = this->flags(S);
    auto f2c  = this->vec(std::vector<int>(ref.n + 1, 0));
    auto amin = this->vec(std::vector<T>(ref.n, 0.0));
    auto amax = this->vec(std::vector<T>(ref.n, 0.0));
    auto P    = empty_matrix<T>(TypeParam::backend, csr_format);
    ASSERT_TRUE(A->RSDirectProlongNnz(
        *cf, *s, *this->ghost(), amin.get(), amax.get(), f2c.get(), P.get(), nullptr));
    ASSERT_TRUE(A->RSDirectProlongFill(*this->template none<int64_t>(),
                                       *f2c,
                                       *cf,
                                       *s,
                                       *this->ghost(),
                                       *amin,
                                       *amax,
                                       P.get(),
                                       nullptr,
                                       nullptr));

    Csr<T> expected(7, 4);
    for(const auto& row :
        std::vector<std::vector<std::pair<int, double>>>{{{0, 1.0}},
                                                         {{0, 1.0}},
                                                         {{1, 1.0}},
                                                         {{1, 20.0 / 9.0}, {2, -14.0 / 9.0}},
                                                         {{2, 1.0}},
                                                         {{2, 1.0}},
                                                         {{3, 1.0}}})
    {
        for(const auto& e : row)
        {
            expected.add(e.first, e.second);
        }
        expected.end_row();
    }
    expect_near(csr_of(*P), expected, tol<T>());
}

// P(i, j) = 1 where j numbers the root of i among the roots in index order, an empty
// row for nodes without aggregate
TYPED_TEST(backend_amg, unsmoothed_aggregation_prolongation)
{
    using T            = typename TypeParam::value_type;
    const Csr<T> ref   = path<T>(6);
    auto         A     = this->make(ref);
    auto         agg   = this->vec(std::vector<int64_t>{0, 0, 1, -2, 1, 0});
    auto         roots = this->vec(std::vector<int64_t>{1, 1, 4, -1, 4, 1});
    auto         f2c   = this->vec(std::vector<int>(ref.n + 1, 0));
    auto         P     = empty_matrix<T>(TypeParam::backend, csr_format);

    ASSERT_TRUE(A->AMGUnsmoothedAggregationProlongNnz(
        0, ref.n, *agg, *roots, *this->ghost(), f2c.get(), P.get(), nullptr));
    ASSERT_TRUE(A->AMGUnsmoothedAggregationProlongFill(
        0, ref.n, *agg, *roots, *f2c, *this->ghost(), P.get(), nullptr, nullptr));

    Csr<T> expected(6, 2);
    for(int j : {0, 0, 1, -1, 1, 0})
    {
        if(j >= 0)
        {
            expected.add(j, 1.0);
        }
        expected.end_row();
    }
    expect_near(csr_of(*P), expected, 0.0);
}

namespace
{
    // F, C, F, C, F; the F point 4 stores no diagonal and 1 has a weak F coupling
    template <typename T>
    Csr<T> split_matrix()
    {
        Csr<T> A(5);
        A.add(0, 4.0);
        A.add(1, -1.0);
        A.add(2, -2.0);
        A.add(4, -0.5);
        A.end_row();
        A.add(0, -1.0);
        A.add(1, 3.0);
        A.add(3, -1.0);
        A.end_row();
        A.add(0, -2.0);
        A.add(2, 5.0);
        A.add(3, -1.0);
        A.end_row();
        A.add(1, -1.0);
        A.add(2, -1.0);
        A.add(3, 2.0);
        A.end_row();
        A.add(0, -0.5);
        A.add(1, -3.0);
        A.add(2, -0.4);
        A.end_row();
        return A;
    }

    const std::vector<int> split_cf  = {2, 1, 2, 1, 2};
    const std::vector<int> split_f2c = {0, 0, 1, 1, 2, 2};
    const std::vector<int> split_f2f = {0, 1, 1, 2, 2, 3};
}

// f2c and f2f number the C and the F points, with their counts at the end. A_FF holds
// the diagonal, a zero where A stores none, and the strong F couplings; A_FC the strong
// C couplings, both with sorted columns.
TYPED_TEST(backend_amg, rs_matrix_matrix_split)
{
    using T                = typename TypeParam::value_type;
    const Csr<T>      ref  = split_matrix<T>();
    std::vector<bool> S    = off_diagonal(ref);
    S[position(ref, 0, 4)] = false;
    S[position(ref, 4, 2)] = false;

    auto A    = this->make(ref);
    auto f2c  = this->vec(std::vector<int>(ref.n + 1, 7));
    auto f2f  = this->vec(std::vector<int>(ref.n + 1, 7));
    auto A_FF = empty_matrix<T>(TypeParam::backend, csr_format);
    auto A_FC = empty_matrix<T>(TypeParam::backend, csr_format);
    ASSERT_TRUE(A->RSMMExtPISplit(
        *this->vec(split_cf), *this->flags(S), f2c.get(), f2f.get(), A_FF.get(), A_FC.get()));

    EXPECT_EQ(values_of(*f2c), split_f2c);
    EXPECT_EQ(values_of(*f2f), split_f2f);

    Csr<T> ff(3);
    ff.add(0, 4.0);
    ff.add(1, -2.0);
    ff.end_row();
    ff.add(0, -2.0);
    ff.add(1, 5.0);
    ff.end_row();
    ff.add(0, -0.5);
    ff.add(2, 0.0);
    ff.end_row();
    expect_near(csr_of(*A_FF), ff, 0.0);

    Csr<T> fc(3, 2);
    fc.add(0, -1.0);
    fc.end_row();
    fc.add(1, -1.0);
    fc.end_row();
    fc.add(0, -3.0);
    fc.end_row();
    expect_near(csr_of(*A_FC), fc, 0.0);
}

// F, F, C, C. The F point 0 stores no diagonal, so its zero diagonal goes ahead of its
// strong F neighbour 1. The F point 1 has neither a strong C coupling nor a strong
// coupling back to 0, so the denominator of the coupling (0, 1), the row sum of A_FC
// plus (A_FF)_10, vanishes. That coupling multiplies the empty row 1 of A_FC, so the
// weights A_FF A_FC do not depend on it.
TYPED_TEST(backend_amg, rs_matrix_matrix_steps_of_a_row_without_diagonal)
{
    using T = typename TypeParam::value_type;
    Csr<T> ref(4);
    ref.add(1, -1.0);
    ref.add(2, -1.0);
    ref.add(3, 0.5);
    ref.end_row();
    ref.add(0, -0.5);
    ref.add(1, 1.0);
    ref.end_row();
    ref.add(2, 1.0);
    ref.end_row();
    ref.add(3, 1.0);
    ref.end_row();
    const std::vector<int> cf = {2, 2, 1, 1};

    auto A    = this->make(ref);
    auto f2c  = this->vec(std::vector<int>(ref.n + 1, 7));
    auto f2f  = this->vec(std::vector<int>(ref.n + 1, 7));
    auto A_FF = empty_matrix<T>(TypeParam::backend, csr_format);
    auto A_FC = empty_matrix<T>(TypeParam::backend, csr_format);
    ASSERT_TRUE(A->RSMMExtPISplit(*this->vec(cf),
                                  *this->flags(flags_at(ref, {{0, 1}, {0, 2}})),
                                  f2c.get(),
                                  f2f.get(),
                                  A_FF.get(),
                                  A_FC.get()));
    EXPECT_EQ(values_of(*f2c), (std::vector<int>{0, 0, 0, 1, 2}));
    EXPECT_EQ(values_of(*f2f), (std::vector<int>{0, 1, 2, 2, 2}));

    Csr<T> ff(2);
    ff.add(0, 0.0);
    ff.add(1, -1.0);
    ff.end_row();
    ff.add(1, 1.0);
    ff.end_row();
    expect_near(csr_of(*A_FF), ff, 0.0);

    Csr<T> fc(2);
    fc.add(0, -1.0);
    fc.end_row();
    fc.end_row();
    expect_near(csr_of(*A_FC), fc, 0.0);

    // Both rows have the weak part D_w = 0.5 and no correction, so they are scaled by
    // -1 / 0.5 after the diagonal is set to 1
    ASSERT_TRUE(A->RSMMExtPIScale(*this->vec(cf), *f2f, *A_FC, A_FF.get()));
    const Csr<T> scaled = csr_of(*A_FF);
    EXPECT_NEAR(entry(scaled, 0, 0), -2.0, tol<T>());
    EXPECT_NEAR(entry(scaled, 1, 1), -2.0, tol<T>());
    EXPECT_EQ(entry(scaled, 1, 0), T(0));
    expect_near(scaled.apply(std::vector<T>{-1.0, 0.0}), std::vector<T>{2.0, 0.0}, tol<T>());
}

// C rows of P are the identity, F rows the rows of the weights W
TYPED_TEST(backend_amg, rs_matrix_matrix_assemble)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = split_matrix<T>();
    Csr<T>       W(3, 2);
    W.add(0, 0.7);
    W.add(1, 0.3);
    W.end_row();
    W.add(1, 1.0);
    W.end_row();
    W.end_row();

    auto A = this->make(ref);
    auto P = empty_matrix<T>(TypeParam::backend, csr_format);
    ASSERT_TRUE(A->RSMMExtPIAssembleP(*this->vec(split_cf),
                                      *this->vec(split_f2c),
                                      *this->vec(split_f2f),
                                      *this->make(W),
                                      P.get()));

    Csr<T> expected(5, 2);
    expected.add(0, 0.7);
    expected.add(1, 0.3);
    expected.end_row();
    expected.add(0, 1.0);
    expected.end_row();
    expected.add(1, 1.0);
    expected.end_row();
    expected.add(1, 1.0);
    expected.end_row();
    expected.end_row();
    expect_near(csr_of(*P), expected, 0.0);
}

// Without F points both blocks are empty, scaling them does nothing, and P is the
// identity
TYPED_TEST(backend_amg, rs_matrix_matrix_steps_without_f_points)
{
    using T                    = typename TypeParam::value_type;
    const Csr<T>           ref = path<T>(4);
    const std::vector<int> cf(ref.n, 1);

    auto A    = this->make(ref);
    auto f2c  = this->vec(std::vector<int>(ref.n + 1, 7));
    auto f2f  = this->vec(std::vector<int>(ref.n + 1, 7));
    auto A_FF = empty_matrix<T>(TypeParam::backend, csr_format);
    auto A_FC = empty_matrix<T>(TypeParam::backend, csr_format);
    ASSERT_TRUE(A->RSMMExtPISplit(*this->vec(cf),
                                  *this->flags(off_diagonal(ref)),
                                  f2c.get(),
                                  f2f.get(),
                                  A_FF.get(),
                                  A_FC.get()));
    EXPECT_EQ(values_of(*f2c), (std::vector<int>{0, 1, 2, 3, 4}));
    EXPECT_EQ(values_of(*f2f), (std::vector<int>{0, 0, 0, 0, 0}));
    EXPECT_EQ(A_FF->GetM(), 0);
    EXPECT_EQ(A_FC->GetM(), 0);

    ASSERT_TRUE(A->RSMMExtPIScale(*this->vec(cf), *f2f, *A_FC, A_FF.get()));
    EXPECT_EQ(A_FF->GetM(), 0);
    ASSERT_TRUE(A->RSMMExtPEScale(*this->vec(cf), *f2f, A_FC.get(), A_FF.get()));
    EXPECT_EQ(A_FF->GetM(), 0);
    EXPECT_EQ(A_FC->GetM(), 0);

    auto P = empty_matrix<T>(TypeParam::backend, csr_format);
    ASSERT_TRUE(
        A->RSMMExtPIAssembleP(*this->vec(cf), *f2c, *f2f, *this->make(Csr<T>(0, ref.n)), P.get()));
    Csr<T> identity(ref.n);
    for(int i = 0; i < ref.n; ++i)
    {
        identity.add(i, 1.0);
        identity.end_row();
    }
    expect_near(csr_of(*P), identity, 0.0);
}

namespace
{
    template <typename P>
    class backend_amg_host : public backend_amg<P>
    {
    };
}

// The greedy aggregation has no accelerator version
TYPED_TEST_SUITE(backend_amg_host, HostOnly<RealTypes>, TypedName);

// A node without connections of its own is in no aggregate (-2), also when a seed is
// connected to it. The seed 1 takes its other neighbour 2.
TYPED_TEST(backend_amg_host, greedy_aggregation_leaves_out_nodes_without_connections)
{
    using T            = typename TypeParam::value_type;
    const Csr<T> ref   = path<T>(3);
    auto         A     = this->make(ref);
    auto         agg   = this->vec(std::vector<int64_t>(ref.n, 7));
    auto         roots = this->vec(std::vector<int64_t>(ref.n, 7));
    ASSERT_TRUE(A->AMGGreedyAggregate(
        *this->flags(flags_at(ref, {{1, 0}, {1, 2}, {2, 1}})), agg.get(), roots.get()));

    EXPECT_EQ(values_of(*agg), (std::vector<int64_t>{-2, 0, 0}));
    const std::vector<int64_t> root = values_of(*roots);
    EXPECT_EQ(std::vector<int64_t>(root.begin() + 1, root.end()), (std::vector<int64_t>{1, 1}));
}
