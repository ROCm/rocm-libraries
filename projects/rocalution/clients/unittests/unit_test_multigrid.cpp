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

// The multigrid solver classes: the cycles of BaseMultiGrid on a user-supplied
// hierarchy (MultiGrid), and the hierarchies the AMG classes build. Each cycle is
// compared with a dense reference of the same cycle, each hierarchy with the Galerkin
// products of its transfer operators. The AMG building blocks themselves are tested in
// unit_test_local_matrix_amg.cpp; here they only provide the aggregations and
// splittings from which the expected transfer operators are computed.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "unit_test_amg_reference.hpp"
#include "unit_test_common.hpp"
#include "unit_test_local.hpp"
#include "unit_test_reference.hpp"
#include "unit_test_types.hpp"

#include "base/local_matrix.hpp"
#include "base/local_vector.hpp"
#include "solvers/krylov/cg.hpp"
#include "solvers/multigrid/multigrid.hpp"
#include "solvers/multigrid/pairwise_amg.hpp"
#include "solvers/multigrid/ruge_stueben_amg.hpp"
#include "solvers/multigrid/smoothed_amg.hpp"
#include "solvers/multigrid/unsmoothed_amg.hpp"
#include "solvers/preconditioners/preconditioner.hpp"
#include "solvers/solver.hpp"

// PairwiseAMG is deprecated
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

using namespace rocalution_unit;
using rocalution::AddWeakConnections;
using rocalution::CoarseningStrategy;
using rocalution::Direct;
using rocalution::ExtPI;
using rocalution::Greedy;
using rocalution::LocalMatrix;
using rocalution::LocalVector;
using rocalution::LumpingStrategy;
using rocalution::MMExtPE;
using rocalution::MMExtPI;
using rocalution::PMIS;
using rocalution::SubtractWeakConnections;

namespace
{
    template <typename T>
    using Matrix = LocalMatrix<T>;
    template <typename T>
    using Vector = LocalVector<T>;
    template <typename T>
    using Vec = std::vector<T>;
    template <typename T>
    using MG = rocalution::BaseMultiGrid<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using AMG = rocalution::BaseAMG<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using MultiGrid = rocalution::MultiGrid<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using SAAMG = rocalution::SAAMG<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using UAAMG = rocalution::UAAMG<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using RSAMG = rocalution::RugeStuebenAMG<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using PWAMG = rocalution::PairwiseAMG<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using CG = rocalution::CG<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using FixedPoint = rocalution::FixedPoint<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using Jacobi = rocalution::Jacobi<Matrix<T>, Vector<T>, T>;
    template <typename T>
    using Smoother = rocalution::IterativeLinearSolver<Matrix<T>, Vector<T>, T>;

    // LocalMatrix has no public query of its backend
    template <typename T>
    struct MatrixPeek : Matrix<T>
    {
        static bool on_host(const Matrix<T>& A)
        {
            return (A.*(&MatrixPeek::is_host_))();
        }
    };

    // The levels of the multigrid classes are protected
    template <typename T>
    struct Peek : MG<T>
    {
        static int levels(const MG<T>& mg)
        {
            return mg.*(&Peek::levels_);
        }

        // Operator of a coarse level, level >= 1
        static Matrix<T>& op(const MG<T>& mg, int level)
        {
            return *(mg.*(&Peek::op_level_))[level - 1];
        }

        // From level to level + 1
        static Matrix<T>& restriction(const MG<T>& mg, int level)
        {
            return *(mg.*(&Peek::restrict_op_level_))[level];
        }

        // From level + 1 to level
        static Matrix<T>& prolongation(const MG<T>& mg, int level)
        {
            return *(mg.*(&Peek::prolong_op_level_))[level];
        }
    };

    template <typename T>
    T dot(const Vec<T>& a, const Vec<T>& b)
    {
        T s = static_cast<T>(0);
        for(size_t i = 0; i < a.size(); ++i)
        {
            s += a[i] * b[i];
        }
        return s;
    }

    template <typename T>
    void axpy(T alpha, const Vec<T>& x, Vec<T>& y)
    {
        for(size_t i = 0; i < x.size(); ++i)
        {
            y[i] += alpha * x[i];
        }
    }

    template <typename T>
    Vec<T> residual(const Csr<T>& A, const Vec<T>& b, const Vec<T>& x)
    {
        Vec<T> r = A.apply(x);
        for(size_t i = 0; i < r.size(); ++i)
        {
            r[i] = b[i] - r[i];
        }
        return r;
    }

    // Gaussian elimination with partial pivoting
    template <typename T>
    Vec<T> solve_dense(const Csr<T>& A, Vec<T> b)
    {
        const int n = A.n;
        Vec<T>    D = dense(A);
        for(int k = 0; k < n; ++k)
        {
            int p = k;
            for(int i = k + 1; i < n; ++i)
            {
                if(std::abs(D[i * n + k]) > std::abs(D[p * n + k]))
                {
                    p = i;
                }
            }
            for(int j = 0; j < n; ++j)
            {
                std::swap(D[k * n + j], D[p * n + j]);
            }
            std::swap(b[k], b[p]);
            for(int i = k + 1; i < n; ++i)
            {
                const T f = D[i * n + k] / D[k * n + k];
                for(int j = k; j < n; ++j)
                {
                    D[i * n + j] -= f * D[k * n + j];
                }
                b[i] -= f * b[k];
            }
        }
        Vec<T> x(n);
        for(int i = n - 1; i >= 0; --i)
        {
            T s = b[i];
            for(int j = i + 1; j < n; ++j)
            {
                s -= D[i * n + j] * x[j];
            }
            x[i] = s / D[i * n + i];
        }
        return x;
    }

    template <typename T>
    Csr<T> galerkin(const Csr<T>& R, const Csr<T>& A, const Csr<T>& P, double scale)
    {
        Vec<T> RAP = dense_product(
            dense_product(dense(R), dense(A), R.n, R.m, A.m), dense(P), R.n, A.m, P.m);
        for(T& v : RAP)
        {
            v *= static_cast<T>(scale);
        }
        return from_dense(R.n, P.m, RAP);
    }

    // The same values, ignoring stored zeros
    template <typename T>
    void expect_dense_near(const Csr<T>& A, const Csr<T>& ref, double tol)
    {
        ASSERT_EQ(A.n, ref.n);
        ASSERT_EQ(A.m, ref.m);
        expect_near(dense(A), dense(ref), tol);
    }

    // A, and for each coarser level its operator and the transfer operators from the
    // level above
    template <typename T>
    struct Hierarchy
    {
        std::vector<Csr<T>> A;
        std::vector<Csr<T>> R;
        std::vector<Csr<T>> P;

        int levels() const
        {
            return static_cast<int>(A.size());
        }
    };

    template <typename T>
    Hierarchy<T> hierarchy_of(const MG<T>& mg, const Csr<T>& A0)
    {
        Hierarchy<T> h;
        h.A.push_back(A0);
        for(int l = 0; l < Peek<T>::levels(mg) - 1; ++l)
        {
            h.R.push_back(csr_of(Peek<T>::restriction(mg, l)));
            h.P.push_back(csr_of(Peek<T>::prolongation(mg, l)));
            h.A.push_back(csr_of(Peek<T>::op(mg, l + 1)));
        }
        return h;
    }

    // The same transfer operators, with the Galerkin products of the new finest operator
    template <typename T>
    Hierarchy<T> with_finest(const Hierarchy<T>& h, const Csr<T>& A0)
    {
        Hierarchy<T> n = h;
        n.A[0]         = A0;
        for(int l = 0; l + 1 < h.levels(); ++l)
        {
            n.A[l + 1] = galerkin(h.R[l], n.A[l], h.P[l], 1.0);
        }
        return n;
    }

    // Linear interpolation from the odd nodes of a path of 2m + 1 nodes
    template <typename T>
    Csr<T> linear_interpolation(int n)
    {
        const int m = (n - 1) / 2;
        Csr<T>    P(n, m);
        for(int i = 0; i < n; ++i)
        {
            if(i % 2 == 1)
            {
                P.add(i / 2, static_cast<T>(1));
            }
            else
            {
                if(i / 2 - 1 >= 0)
                {
                    P.add(i / 2 - 1, static_cast<T>(0.5));
                }
                if(i / 2 < m)
                {
                    P.add(i / 2, static_cast<T>(0.5));
                }
            }
            P.end_row();
        }
        return P;
    }

    // 15, 7, 3, 1 unknowns on the levels
    template <typename T>
    Hierarchy<T> geometric_hierarchy(int levels)
    {
        Hierarchy<T> h;
        h.A.push_back(Tridiagonal<T>(15));
        for(int l = 0; l + 1 < levels; ++l)
        {
            h.P.push_back(linear_interpolation<T>(h.A[l].n));
            h.R.push_back(transposed(h.P[l]));
            h.A.push_back(galerkin(h.R[l], h.A[l], h.P[l], 1.0));
        }
        return h;
    }

    // The geometric hierarchy with a 16th, uncoupled unknown on the finest level that
    // the prolongation does not reach
    template <typename T>
    Hierarchy<T> with_uncoupled_unknown(int levels)
    {
        Hierarchy<T> h = geometric_hierarchy<T>(levels);
        Csr<T>       A(16);
        Csr<T>       P(16, h.P[0].m);
        for(int i = 0; i < 15; ++i)
        {
            for(PtrType j = h.A[0].row_offset[i]; j < h.A[0].row_offset[i + 1]; ++j)
            {
                A.add(h.A[0].col[j], h.A[0].val[j]);
            }
            A.end_row();
            for(PtrType j = h.P[0].row_offset[i]; j < h.P[0].row_offset[i + 1]; ++j)
            {
                P.add(h.P[0].col[j], h.P[0].val[j]);
            }
            P.end_row();
        }
        A.add(15, static_cast<T>(4));
        A.end_row();
        P.end_row();
        h.A[0] = A;
        h.P[0] = P;
        h.R[0] = transposed(P);
        return h;
    }

    struct CycleOptions
    {
        unsigned int cycle       = rocalution::Vcycle;
        int          pre         = 1;
        int          post        = 1;
        double       omega       = 0.5;
        bool         scaling     = false;
        bool         kcycle_full = true;
    };

    // The cycles of BaseMultiGrid with damped Jacobi smoothers and an exact coarse solve.
    // The coarse levels start from zero, except for the second cycle of a W-cycle.
    template <typename T>
    class DenseCycle
    {
    public:
        DenseCycle(const Hierarchy<T>& h, const CycleOptions& o)
            : h_(h)
            , o_(o)
        {
        }

        // One iteration of the solver, from x
        void iterate(const Vec<T>& b, Vec<T>& x) const
        {
            cycle(0, b, x, false);
        }

        // The preconditioner: one cycle from zero
        Vec<T> apply(const Vec<T>& b) const
        {
            Vec<T> x(b.size(), static_cast<T>(0));
            cycle(0, b, x, true);
            return x;
        }

    private:
        int last() const
        {
            return h_.levels() - 1;
        }

        static T ratio(T f, T d)
        {
            return d == static_cast<T>(0) ? static_cast<T>(1) : f / d;
        }

        void smooth(int l, const Vec<T>& b, Vec<T>& x, int steps) const
        {
            const Csr<T>& A     = h_.A[l];
            const T       omega = static_cast<T>(o_.omega);
            for(int k = 0; k < steps; ++k)
            {
                const Vec<T> r = residual(A, b, x);
                for(int i = 0; i < A.n; ++i)
                {
                    x[i] += omega * r[i] / entry(A, i, i);
                }
            }
        }

        void cycle(int l, const Vec<T>& b, Vec<T>& x, bool zero_guess) const
        {
            const Csr<T>& A = h_.A[l];
            if(l == last())
            {
                x = solve_dense(A, b);
                return;
            }

            if(zero_guess)
            {
                std::fill(x.begin(), x.end(), static_cast<T>(0));
            }
            smooth(l, b, x, o_.pre);
            if(o_.scaling && l > 0 && l < last() - 1 && o_.pre > 0)
            {
                const T f = ratio(dot(b, x), dot(A.apply(x), x));
                for(T& v : x)
                {
                    v *= f;
                }
            }

            const Vec<T> r = residual(A, b, x);
            Vec<T>       xc(h_.A[l + 1].n, static_cast<T>(0));
            correction(l + 1, h_.R[l].apply(r), xc);
            const Vec<T> e = h_.P[l].apply(xc);

            // The factor that minimizes the A-norm error of x + factor * e
            T factor = static_cast<T>(1);
            if(o_.scaling && l < last() - 1)
            {
                factor = ratio(dot(r, e), dot(A.apply(e), e));
            }
            axpy(factor, e, x);
            smooth(l, b, x, o_.post);
        }

        void correction(int l, const Vec<T>& b, Vec<T>& x) const
        {
            switch(o_.cycle)
            {
            case rocalution::Wcycle:
                cycle(l, b, x, true);
                cycle(l, b, x, false);
                break;
            case rocalution::Kcycle:
                kcycle(l, b, x);
                break;
            default:
                cycle(l, b, x, true);
                break;
            }
        }

        // Two steps of preconditioned CG with the cycle as preconditioner
        void kcycle(int l, const Vec<T>& b, Vec<T>& x) const
        {
            if((l != 1 && !o_.kcycle_full) || l == last())
            {
                cycle(l, b, x, true);
                return;
            }

            const Csr<T>& A = h_.A[l];
            cycle(l, b, x, true);
            Vec<T> r   = b;
            T      rho = dot(r, x);
            Vec<T> q   = A.apply(x);
            if(dot(x, q) == static_cast<T>(0))
            {
                // A zero residual has the zero correction
                return;
            }
            T alpha = rho / dot(x, q);
            axpy(-alpha, q, r);

            cycle(l, r, q, true);
            const T rho_old = rho;
            rho             = dot(r, q);
            Vec<T> p        = q;
            axpy(rho / rho_old, x, p);
            q = A.apply(p);
            for(T& v : x)
            {
                v *= alpha;
            }
            alpha = rho / dot(p, q);
            axpy(alpha, p, x);
        }

        const Hierarchy<T>& h_;
        CycleOptions        o_;
    };

    template <typename T>
    Vec<T> cycled(const DenseCycle<T>& ref, const Vec<T>& b, Vec<T> x, int cycles)
    {
        for(int k = 0; k < cycles; ++k)
        {
            ref.iterate(b, x);
        }
        return x;
    }

    template <typename T>
    T norm(const Vec<T>& v)
    {
        return std::sqrt(dot(v, v));
    }

    template <typename T>
    Vec<T> initial_guess(int n)
    {
        Vec<T> x(n);
        for(int i = 0; i < n; ++i)
        {
            x[i] = static_cast<T>(i % 3 - 1);
        }
        return x;
    }

    // Runs the multigrid solver for the given number of cycles
    template <typename T>
    Vec<T> solve(MG<T>& mg, const Matrix<T>& A, const Vec<T>& b, const Vec<T>& x0, int cycles)
    {
        mg.Init(0.0, 0.0, 1e+8, cycles);
        auto rhs = make_local_vector(A, b);
        auto x   = make_local_vector(A, x0);
        mg.Solve(*rhs, x.get());
        return values_of(*x);
    }

    // Damped Jacobi smoothers for the levels above the coarsest
    template <typename T>
    struct Smoothers
    {
        Smoothers(int count, double omega)
        {
            for(int i = 0; i < count; ++i)
            {
                jacobi.push_back(std::make_unique<Jacobi<T>>());
                fixed_point.push_back(std::make_unique<FixedPoint<T>>());
                fixed_point.back()->SetRelaxation(static_cast<T>(omega));
                fixed_point.back()->SetPreconditioner(*jacobi.back());
                fixed_point.back()->Verbose(0);
                levels.push_back(fixed_point.back().get());
            }
        }

        std::vector<std::unique_ptr<Jacobi<T>>>     jacobi;
        std::vector<std::unique_ptr<FixedPoint<T>>> fixed_point;
        std::vector<Smoother<T>*>                   levels;
    };

    // Converges to rounding on the small coarse levels
    template <typename T>
    void init_exact(CG<T>& cg)
    {
        // A relative residual of 1e-14 is below the rounding of float
        cg.Init(0.0, std::is_same_v<T, float> ? 1e-6 : 1e-14, 1e+8, 200);
        cg.Verbose(0);
    }

    // MultiGrid does not allocate the transfer maps that Finalize deletes, a known bug
    template <typename T>
    class UserMultiGrid : public MultiGrid<T>
    {
    public:
        void Build(void) override
        {
            MultiGrid<T>::Build();
            this->trans_level_ = new LocalVector<int>*[this->levels_ - 1]();
        }
    };

    // A hierarchy on a backend, with the pointer arrays that MultiGrid takes
    template <typename T>
    struct UserLevels
    {
        UserLevels(Backend b, const Hierarchy<T>& h)
            : levels(h.levels())
        {
            A = make_local_matrix(b, h.A[0]);
            for(int l = 0; l + 1 < levels; ++l)
            {
                ops.push_back(make_local_matrix(b, h.A[l + 1]));
                restriction.push_back(make_local_matrix(b, h.R[l]));
                prolongation.push_back(make_local_matrix(b, h.P[l]));
                op_ptr.push_back(ops.back().get());
                r_ptr.push_back(restriction.back().get());
                p_ptr.push_back(prolongation.back().get());
            }
        }

        void attach(MultiGrid<T>& mg)
        {
            mg.InitLevels(levels);
            mg.SetOperator(*A);
            mg.SetOperatorHierarchy(op_ptr.data());
            mg.SetRestrictOperator(r_ptr.data());
            mg.SetProlongOperator(p_ptr.data());
        }

        int                                     levels;
        std::unique_ptr<Matrix<T>>              A;
        std::vector<std::unique_ptr<Matrix<T>>> ops;
        std::vector<std::unique_ptr<Matrix<T>>> restriction;
        std::vector<std::unique_ptr<Matrix<T>>> prolongation;
        std::vector<Matrix<T>*>                 op_ptr;
        std::vector<Matrix<T>*>                 r_ptr;
        std::vector<Matrix<T>*>                 p_ptr;
    };

    template <typename T>
    void set_cycle(MG<T>& mg, const CycleOptions& o)
    {
        mg.SetCycle(o.cycle);
        mg.SetKcycleFull(o.kcycle_full);
        mg.SetSmootherPreIter(o.pre);
        mg.SetSmootherPostIter(o.post);
        mg.SetScaling(o.scaling);
        mg.Verbose(0);
    }

    // A built MultiGrid on a geometric hierarchy, with Jacobi smoothers and an exact
    // coarse solve. The members are declared in the order that destroys the solver
    // before the objects it uses.
    template <typename T>
    struct UserSetup
    {
        UserSetup(Backend b, int levels, const CycleOptions& o, bool set_options = true)
            : UserSetup(b, geometric_hierarchy<T>(levels), o, set_options)
        {
        }

        UserSetup(Backend b, Hierarchy<T> hierarchy, const CycleOptions& o, bool set_options = true)
            : h(std::move(hierarchy))
            , user(b, h)
            , smoothers(h.levels() - 1, o.omega)
        {
            init_exact(coarse);
            user.attach(mg);
            mg.SetSmoother(smoothers.levels.data());
            mg.SetSolver(coarse);
            if(set_options)
            {
                set_cycle(mg, o);
            }
            mg.Verbose(0);
        }

        Hierarchy<T>     h;
        UserLevels<T>    user;
        Smoothers<T>     smoothers;
        CG<T>            coarse;
        UserMultiGrid<T> mg;
    };

    template <typename P>
    class multigrid : public TypedBackendTest<P>
    {
    };

    struct CycleCase
    {
        std::string  name;
        int          levels;
        CycleOptions options;
    };

    CycleOptions
        options(unsigned int cycle, int pre, int post, bool scaling, bool kcycle_full = true)
    {
        CycleOptions o;
        o.cycle       = cycle;
        o.pre         = pre;
        o.post        = post;
        o.scaling     = scaling;
        o.kcycle_full = kcycle_full;
        return o;
    }
}

TYPED_TEST_SUITE(multigrid, WithBackends<RealTypes>, TypedName);

TYPED_TEST(multigrid, cycles_match_the_dense_cycle)
{
    using T = typename TypeParam::value_type;
    using rocalution::Kcycle;
    using rocalution::Vcycle;
    const std::vector<CycleCase> cases
        = {{"V", 4, options(Vcycle, 1, 1, false)},
           {"V of two levels", 2, options(Vcycle, 1, 1, false)},
           {"V with 2 pre- and 3 post-smoothing steps", 4, options(Vcycle, 2, 3, false)},
           {"V without post-smoothing", 4, options(Vcycle, 1, 0, false)},
           {"V of three levels with scaling", 3, options(Vcycle, 2, 1, true)},
           {"K", 4, options(Kcycle, 1, 1, false)},
           {"K on the first coarse level only", 4, options(Kcycle, 1, 1, false, false)},
           {"K of two levels", 2, options(Kcycle, 1, 1, false)}};
    // With four levels the first coarse level scales its correction too
    const std::vector<CycleCase> coarse_scaling
        = {{"V with scaling", 4, options(Vcycle, 1, 1, true)},
           {"K with scaling", 4, options(Kcycle, 1, 2, true)}};

    const auto expect = [](const CycleCase& c) {
        SCOPED_TRACE(c.name);
        UserSetup<T> s(TypeParam::backend, c.levels, c.options);
        s.mg.Build();

        const Vec<T>        b  = rhs<T>(15);
        const Vec<T>        x0 = initial_guess<T>(15);
        const DenseCycle<T> ref(s.h, c.options);
        const Vec<T>        expected = cycled(ref, b, x0, 2);

        const Vec<T> x = solve(s.mg, *s.user.A, b, x0, 2);
        expect_near(x, expected, tol<T>());
        EXPECT_EQ(s.mg.GetIterationCount(), 2);
        const T res = norm(residual(s.h.A[0], b, expected));
        EXPECT_TRUE(is_near(static_cast<T>(s.mg.GetCurrentResidual()), res, tol<T>()));
    };

    for(const CycleCase& c : cases)
    {
        expect(c);
    }
    KNOWN_BUG("coarse levels scale the correction by (b, e) / (Ae, e) instead of (r, e) / (Ae, e)");
    for(const CycleCase& c : coarse_scaling)
    {
        expect(c);
    }
}

TYPED_TEST(multigrid, w_cycle_continues_from_the_first_coarse_cycle)
{
    using T              = typename TypeParam::value_type;
    const CycleOptions o = options(rocalution::Wcycle, 1, 1, false);
    UserSetup<T>       s(TypeParam::backend, 3, o);
    s.mg.Build();

    const Vec<T> b  = rhs<T>(15);
    const Vec<T> x0 = initial_guess<T>(15);
    const Vec<T> x  = solve(s.mg, *s.user.A, b, x0, 1);

    // The second V-cycle of the W-cycle starts from zero again, so the W-cycle
    // computes the V-cycle
    KNOWN_BUG("the W-cycle repeats the same V-cycle on the coarse levels");
    expect_near(x, cycled(DenseCycle<T>(s.h, o), b, x0, 1), tol<T>());
}

TYPED_TEST(multigrid, coarse_levels_start_from_zero_without_pre_smoothing)
{
    using T              = typename TypeParam::value_type;
    const CycleOptions o = options(rocalution::Vcycle, 0, 1, false);
    UserSetup<T>       s(TypeParam::backend, 3, o);
    s.mg.Build();

    const Vec<T>        b  = rhs<T>(15);
    const Vec<T>        x0 = initial_guess<T>(15);
    const DenseCycle<T> ref(s.h, o);

    // The coarse vectors are zero after Build, so the first cycle is right
    expect_near(solve(s.mg, *s.user.A, b, x0, 1), cycled(ref, b, x0, 1), tol<T>());

    KNOWN_BUG("without pre-smoothing a coarse level starts from its previous correction");
    expect_near(solve(s.mg, *s.user.A, b, x0, 2), cycled(ref, b, x0, 2), tol<T>());
}

// The residual after the first smoothing step lives on the uncoupled unknown only, so
// every coarse level gets a zero right-hand side and the scaling factors divide by zero
TYPED_TEST(multigrid, zero_coarse_residual_gives_zero_coarse_correction)
{
    using T = typename TypeParam::value_type;
    Vec<T> b(16, static_cast<T>(0));
    b[15]           = static_cast<T>(4);
    const Vec<T> x0 = Vec<T>(16, static_cast<T>(0));

    const auto expect = [&](const CycleOptions& o) {
        UserSetup<T> s(TypeParam::backend, with_uncoupled_unknown<T>(4), o);
        s.mg.Build();
        const Vec<T> x = solve(s.mg, *s.user.A, b, x0, 1);
        for(T v : x)
        {
            ASSERT_TRUE(std::isfinite(v));
        }
        expect_near(x, cycled(DenseCycle<T>(s.h, o), b, x0, 1), scaled_tol<T>(1e-14));
    };

    {
        SCOPED_TRACE("V");
        expect(options(rocalution::Vcycle, 1, 1, true));
    }
    KNOWN_BUG("the K-cycle divides by (x, Ax) = 0 for a zero coarse residual and returns NaN");
    for(bool full : {true, false})
    {
        SCOPED_TRACE(full ? "K, all levels" : "K, level 1 only");
        expect(options(rocalution::Kcycle, 1, 1, true, full));
    }
}

TYPED_TEST(multigrid, scaling_is_on_by_default_and_fixed_at_build)
{
    using T = typename TypeParam::value_type;
    // Three levels scale only the finest correction, which the coarse-level scaling bug
    // leaves alone
    UserSetup<T> s(TypeParam::backend, 3, CycleOptions(), false);
    s.mg.Build();

    CycleOptions scaled;
    scaled.scaling        = true;
    const Vec<T> b        = rhs<T>(15);
    const Vec<T> x0       = initial_guess<T>(15);
    const Vec<T> expected = cycled(DenseCycle<T>(s.h, scaled), b, x0, 1);
    Vec<T>       change   = cycled(DenseCycle<T>(s.h, CycleOptions()), b, x0, 1);
    axpy(static_cast<T>(-1), expected, change);
    ASSERT_GT(norm(change), static_cast<T>(100 * tol<T>()) * norm(expected));
    expect_near(solve(s.mg, *s.user.A, b, x0, 1), expected, tol<T>());

    testing::internal::CaptureStdout();
    s.mg.SetScaling(false);
    EXPECT_NE(testing::internal::GetCapturedStdout().find("Scaling must be set before building"),
              std::string::npos);
    expect_near(solve(s.mg, *s.user.A, b, x0, 1), expected, tol<T>());
}

TYPED_TEST(multigrid, preconditioner_applies_one_cycle_from_zero)
{
    using T = typename TypeParam::value_type;
    for(unsigned int cycle : {rocalution::Vcycle, rocalution::Kcycle})
    {
        SCOPED_TRACE(cycle == rocalution::Vcycle ? "V" : "K");
        // Three levels scale only the finest correction, which the coarse-level scaling
        // bug leaves alone
        const CycleOptions o = options(cycle, 1, 1, true);
        UserSetup<T>       s(TypeParam::backend, 3, o);
        s.mg.InitMaxIter(5);

        FixedPoint<T> outer;
        outer.SetOperator(*s.user.A);
        outer.SetPreconditioner(s.mg);
        outer.Init(0.0, 0.0, 1e+8, 1);
        outer.Verbose(0);
        outer.Build();

        const Vec<T> b  = rhs<T>(15);
        const Vec<T> x0 = initial_guess<T>(15);
        auto         bv = make_local_vector(*s.user.A, b);
        auto         xv = make_local_vector(*s.user.A, x0);
        outer.Solve(*bv, xv.get());

        Vec<T> expected = x0;
        axpy(static_cast<T>(1), DenseCycle<T>(s.h, o).apply(residual(s.h.A[0], b, x0)), expected);
        expect_near(values_of(*xv), expected, tol<T>());
        outer.Clear();
    }
}

TYPED_TEST(multigrid, solve_returns_at_once_from_the_solution)
{
    using T = typename TypeParam::value_type;
    UserSetup<T> s(TypeParam::backend, 3, CycleOptions());
    s.mg.Build();

    const Vec<T> x0 = initial_guess<T>(15);
    auto         b  = make_local_vector(*s.user.A, s.h.A[0].apply(x0));
    auto         x  = make_local_vector(*s.user.A, x0);
    s.mg.Solve(*b, x.get());

    EXPECT_EQ(values_of(*x), x0);
    EXPECT_EQ(s.mg.GetIterationCount(), 0);
}

TYPED_TEST(multigrid, solve_stops_at_the_relative_tolerance)
{
    using T = typename TypeParam::value_type;
    UserSetup<T> s(TypeParam::backend, 3, CycleOptions());
    s.mg.Build();

    const Vec<T>        b  = rhs<T>(15);
    const Vec<T>        x0 = initial_guess<T>(15);
    const DenseCycle<T> ref(s.h, CycleOptions());
    const T             r0 = norm(residual(s.h.A[0], b, x0));
    const T             r2 = norm(residual(s.h.A[0], b, cycled(ref, b, x0, 2))) / r0;
    const T             r3 = norm(residual(s.h.A[0], b, cycled(ref, b, x0, 3))) / r0;
    ASSERT_LT(r3, static_cast<T>(0.5) * r2);

    s.mg.Init(0.0, std::sqrt(r2 * r3), 1e+8, 100);
    auto bv = make_local_vector(*s.user.A, b);
    auto xv = make_local_vector(*s.user.A, x0);
    s.mg.Solve(*bv, xv.get());

    EXPECT_EQ(s.mg.GetIterationCount(), 3);
    expect_near(values_of(*xv), cycled(ref, b, x0, 3), tol<T>());
}

TYPED_TEST(multigrid, verbose_solve_prints_the_cycle)
{
    using T = typename TypeParam::value_type;
    UserSetup<T> s(TypeParam::backend, 3, CycleOptions());
    s.mg.Build();
    s.mg.Verbose(1);

    testing::internal::CaptureStdout();
    solve(s.mg, *s.user.A, rhs<T>(15), initial_guess<T>(15), 1);
    const std::string out = testing::internal::GetCapturedStdout();
    EXPECT_NE(out.find("MultiGrid solver starts"), std::string::npos) << out;
    EXPECT_NE(out.find("MultiGrid ends"), std::string::npos) << out;
}

namespace
{
    // Level placement of the operators, from level 1
    template <typename T>
    std::vector<bool> coarse_on_host(const MG<T>& mg)
    {
        std::vector<bool> host;
        for(int l = 1; l < Peek<T>::levels(mg); ++l)
        {
            host.push_back(MatrixPeek<T>::on_host(Peek<T>::op(mg, l)));
        }
        return host;
    }

    template <typename T>
    std::vector<bool> transfers_on_host(const MG<T>& mg)
    {
        std::vector<bool> host;
        for(int l = 0; l + 1 < Peek<T>::levels(mg); ++l)
        {
            host.push_back(MatrixPeek<T>::on_host(Peek<T>::restriction(mg, l))
                           && MatrixPeek<T>::on_host(Peek<T>::prolongation(mg, l)));
        }
        return host;
    }

    template <typename P>
    class multigrid_backends : public TypedBackendTest<P>
    {
    };

    template <typename P>
    class multigrid_lifetime : public TypedBackendTest<P>
    {
    };

    template <typename P>
    class multigrid_errors : public TypedBackendTest<P>
    {
    };
}

TYPED_TEST_SUITE(multigrid_backends, AcceleratorOnly<RealTypes>, TypedName);
TYPED_TEST_SUITE(multigrid_lifetime, HostOnly<RealTypes>, TypedName);
TYPED_TEST_SUITE(multigrid_errors, HostOnly<RealTypes>, TypedName);

TYPED_TEST(multigrid_backends, host_levels_run_the_coarsest_levels_on_the_host)
{
    using T              = typename TypeParam::value_type;
    const CycleOptions o = options(rocalution::Kcycle, 1, 1, true);
    UserSetup<T>       s(Backend::Accelerator, 4, o);
    s.mg.Build();
    const Vec<T> b        = rhs<T>(15);
    const Vec<T> x0       = initial_guess<T>(15);
    const Vec<T> expected = cycled(DenseCycle<T>(s.h, o), b, x0, 2);

    s.mg.SetHostLevels(2);
    EXPECT_EQ(coarse_on_host(s.mg), (std::vector<bool>{false, true, true}));
    EXPECT_EQ(transfers_on_host(s.mg), (std::vector<bool>{false, true, true}));
    const Vec<T> x = solve(s.mg, *s.user.A, b, x0, 2);

    // Every placement computes the same cycle
    s.user.A->MoveToHost();
    s.mg.MoveToHost();
    EXPECT_EQ(coarse_on_host(s.mg), (std::vector<bool>{true, true, true}));
    EXPECT_EQ(transfers_on_host(s.mg), (std::vector<bool>{true, true, true}));
    expect_near(solve(s.mg, *s.user.A, b, x0, 2), x, tol<T>());

    s.user.A->MoveToAccelerator();
    s.mg.MoveToAccelerator();
    EXPECT_EQ(coarse_on_host(s.mg), (std::vector<bool>{false, true, true}));
    EXPECT_EQ(transfers_on_host(s.mg), (std::vector<bool>{false, true, true}));
    expect_near(solve(s.mg, *s.user.A, b, x0, 2), x, tol<T>());

    testing::internal::CaptureStdout();
    s.mg.SetHostLevels(10);
    EXPECT_NE(testing::internal::GetCapturedStdout().find("larger than the total number of levels"),
              std::string::npos);
    EXPECT_EQ(coarse_on_host(s.mg), (std::vector<bool>{true, true, true}));
    EXPECT_EQ(transfers_on_host(s.mg), (std::vector<bool>{true, true, true}));
    EXPECT_FALSE(MatrixPeek<T>::on_host(*s.user.A));
    expect_near(solve(s.mg, *s.user.A, b, x0, 2), x, tol<T>());

    KNOWN_BUG("coarse levels scale the correction by (b, e) / (Ae, e) instead of (r, e) / (Ae, e)");
    expect_near(x, expected, tol<T>());
}

// Clear, Finalize and the moves do nothing before Build
TYPED_TEST(multigrid_backends, an_unscaled_v_cycle_moves_before_and_after_its_build)
{
    using T              = typename TypeParam::value_type;
    const CycleOptions o = options(rocalution::Vcycle, 1, 1, false);
    UserSetup<T>       s(Backend::Accelerator, 4, o);
    s.mg.Clear();
    s.mg.Finalize();
    s.mg.MoveToHost();
    s.mg.MoveToAccelerator();
    s.mg.Build();
    const Vec<T> b        = rhs<T>(15);
    const Vec<T> x0       = initial_guess<T>(15);
    const Vec<T> expected = cycled(DenseCycle<T>(s.h, o), b, x0, 2);
    expect_near(solve(s.mg, *s.user.A, b, x0, 2), expected, tol<T>());

    s.user.A->MoveToHost();
    s.mg.MoveToHost();
    EXPECT_EQ(coarse_on_host(s.mg), (std::vector<bool>{true, true, true}));
    EXPECT_EQ(transfers_on_host(s.mg), (std::vector<bool>{true, true, true}));
    expect_near(solve(s.mg, *s.user.A, b, x0, 2), expected, tol<T>());

    s.user.A->MoveToAccelerator();
    s.mg.MoveToAccelerator();
    EXPECT_EQ(coarse_on_host(s.mg), (std::vector<bool>{false, false, false}));
    expect_near(solve(s.mg, *s.user.A, b, x0, 2), expected, tol<T>());

    s.mg.SetHostLevels(2);
    EXPECT_EQ(coarse_on_host(s.mg), (std::vector<bool>{false, true, true}));
    EXPECT_EQ(transfers_on_host(s.mg), (std::vector<bool>{false, true, true}));
    expect_near(solve(s.mg, *s.user.A, b, x0, 2), expected, tol<T>());
}

TYPED_TEST(multigrid_lifetime, clear_after_build_releases_the_levels)
{
    using T = typename TypeParam::value_type;
    KNOWN_BUG("MultiGrid deletes transfer maps that it never allocated");
    EXPECT_EXIT(
        {
            const Hierarchy<T> h = geometric_hierarchy<T>(3);
            UserLevels<T>      user(Backend::Host, h);
            Smoothers<T>       smoothers(2, 0.5);
            CG<T>              coarse;
            init_exact(coarse);
            {
                MultiGrid<T> mg;
                user.attach(mg);
                mg.SetSmoother(smoothers.levels.data());
                mg.SetSolver(coarse);
                mg.Verbose(0);
                mg.Build();
                mg.Clear();
            }
            std::exit(0);
        },
        testing::ExitedWithCode(0),
        "");
}

TYPED_TEST(multigrid_lifetime, build_of_a_built_multigrid_rebuilds_it)
{
    using T = typename TypeParam::value_type;
    KNOWN_BUG("Build of a built MultiGrid calls Clear, which sets the number of levels to "
              "-1, so Build fails its assert levels_ > 0");
    EXPECT_EXIT(
        {
            UserSetup<T> s(Backend::Host, 3, CycleOptions());
            s.mg.Build();
            s.mg.Build();
            const Vec<T> b     = rhs<T>(15);
            const Vec<T> x0    = initial_guess<T>(15);
            const Vec<T> x     = solve(s.mg, *s.user.A, b, x0, 1);
            const Vec<T> ref   = cycled(DenseCycle<T>(s.h, CycleOptions()), b, x0, 1);
            double       error = 0.0;
            for(size_t i = 0; i < x.size(); ++i)
            {
                error = std::max(error, static_cast<double>(std::abs(x[i] - ref[i])));
            }
            std::exit(error < tol<T>() ? 0 : 2);
        },
        testing::ExitedWithCode(0),
        "");
}

TYPED_TEST(multigrid_errors, f_cycle_is_a_fatal_error)
{
    using T = typename TypeParam::value_type;
    EXPECT_FATAL_ERROR(
        {
            UserSetup<T> s(Backend::Host, 3, options(rocalution::Fcycle, 1, 1, false));
            s.mg.Build();
            solve(s.mg, *s.user.A, rhs<T>(15), initial_guess<T>(15), 1);
        },
        "base_multigrid\\.cpp");
}

namespace
{
    // A configured AMG, and the cycle its defaults run
    template <typename T>
    struct AmgCase
    {
        std::string                              name;
        std::function<std::unique_ptr<AMG<T>>()> make;
        CycleOptions                             cycle;
    };

    CycleOptions pairwise_cycle()
    {
        return options(rocalution::Kcycle, 1, 2, false);
    }

    template <typename T>
    std::unique_ptr<AMG<T>> make_sa(rocalution::CoarseningStrategy strat,
                                    rocalution::LumpingStrategy    lumping)
    {
        auto amg = std::make_unique<SAAMG<T>>();
        amg->SetCouplingStrength(static_cast<T>(0.08));
        amg->SetCoarseningStrategy(strat);
        amg->SetLumpingStrategy(lumping);
        return amg;
    }

    template <typename T>
    std::unique_ptr<AMG<T>> make_ua(rocalution::CoarseningStrategy strat, double over_interp)
    {
        auto amg = std::make_unique<UAAMG<T>>();
        amg->SetCouplingStrength(static_cast<T>(0.08));
        amg->SetCoarseningStrategy(strat);
        amg->SetOverInterp(static_cast<T>(over_interp));
        return amg;
    }

    template <typename T>
    std::unique_ptr<AMG<T>> make_rs(rocalution::CoarseningStrategy strat,
                                    rocalution::InterpolationType  interp)
    {
        auto amg = std::make_unique<RSAMG<T>>();
        amg->SetCoarseningStrategy(strat);
        amg->SetInterpolationType(interp);
        return amg;
    }

    template <typename T>
    std::unique_ptr<AMG<T>> make_pw()
    {
        return std::unique_ptr<AMG<T>>(new PWAMG<T>);
    }

    // One of each class, with the exact transfer operators that the tests below check
    template <typename T>
    std::vector<AmgCase<T>> amg_classes()
    {
        return {
            {"SA", [] { return make_sa<T>(PMIS, AddWeakConnections); }, CycleOptions()},
            {"UA without over-interpolation", [] { return make_ua<T>(PMIS, 1.0); }, CycleOptions()},
            {"RS", [] { return make_rs<T>(PMIS, ExtPI); }, CycleOptions()},
            {"pairwise", make_pw<T>, pairwise_cycle()}};
    }

    template <typename T>
    void set_manual(AMG<T>& amg, Smoothers<T>& smoothers, CG<T>& coarse)
    {
        init_exact(coarse);
        amg.SetManualSmoothers(true);
        amg.SetManualSolver(true);
        amg.SetSmoother(smoothers.levels.data());
        amg.SetSolver(coarse);
    }

    // An AMG with its hierarchy built, and Jacobi smoothers and an exact coarse solve
    // set by the user. The members are declared in the order that destroys the solver
    // before the objects it uses.
    template <typename T>
    struct ManualAmg
    {
        ManualAmg(Backend b, const Csr<T>& A0, const AmgCase<T>& c, int coarse_size)
            : A(make_local_matrix(b, A0))
        {
            amg = c.make();
            amg->SetOperator(*A);
            amg->SetCoarsestLevel(coarse_size);
            amg->BuildHierarchy();
            smoothers = std::make_unique<Smoothers<T>>(amg->GetNumLevels() - 1, 0.5);
            set_manual(*amg, *smoothers, coarse);
            set_cycle(*amg, c.cycle);
        }

        std::unique_ptr<Matrix<T>>    A;
        std::unique_ptr<Smoothers<T>> smoothers;
        CG<T>                         coarse;
        std::unique_ptr<AMG<T>>       amg;
    };

    // Each coarse operator is R A P of the operators above, R = P^T, the sizes decrease,
    // and only the coarsest level is at most coarse_size
    template <typename T>
    void expect_galerkin_hierarchy(const Hierarchy<T>& h, int coarse_size, double coarse_scale)
    {
        ASSERT_GE(h.levels(), 2);
        for(int l = 0; l + 1 < h.levels(); ++l)
        {
            SCOPED_TRACE("level " + std::to_string(l + 1));
            ASSERT_EQ(h.P[l].n, h.A[l].n);
            ASSERT_EQ(h.P[l].m, h.A[l + 1].n);
            expect_dense_near(h.R[l], transposed(h.P[l]), 0.0);
            expect_dense_near(h.A[l + 1], galerkin(h.R[l], h.A[l], h.P[l], coarse_scale), tol<T>());
            expect_dense_near(h.A[l + 1], transposed(h.A[l + 1]), tol<T>());
            EXPECT_LT(h.A[l + 1].n, h.A[l].n);
            if(l + 2 < h.levels())
            {
                EXPECT_GT(h.A[l + 1].n, coarse_size);
            }
        }
    }

    // Every row holds a single 1: the aggregates of a tentative prolongation
    template <typename T>
    std::vector<int64_t> aggregates_of(const Csr<T>& P)
    {
        std::vector<int64_t> agg;
        for(int i = 0; i < P.n; ++i)
        {
            EXPECT_EQ(P.row_offset[i + 1] - P.row_offset[i], 1) << "row " << i;
            if(P.row_offset[i + 1] - P.row_offset[i] == 1)
            {
                EXPECT_EQ(P.val[P.row_offset[i]], static_cast<T>(1)) << "row " << i;
                agg.push_back(P.col[P.row_offset[i]]);
            }
        }
        return agg;
    }

    // The level operator on the backend of the AMG
    template <typename T>
    std::unique_ptr<Matrix<T>> level_operator(const AMG<T>& amg, const Matrix<T>& A, int level)
    {
        auto copy = std::make_unique<Matrix<T>>();
        copy->CloneFrom(level == 0 ? A : Peek<T>::op(amg, level));
        return copy;
    }

    struct Aggregation
    {
        std::vector<bool>    connections;
        std::vector<int64_t> aggregates;
        std::vector<int64_t> roots;
    };

    template <typename T>
    Aggregation aggregate(const Matrix<T>& A, rocalution::CoarseningStrategy strat, double eps)
    {
        LocalVector<bool>    connections;
        LocalVector<int64_t> agg, roots;
        connections.CloneBackend(A);
        agg.CloneBackend(A);
        roots.CloneBackend(A);
        if(strat == rocalution::Greedy)
        {
            A.AMGGreedyAggregate(static_cast<T>(eps), &connections, &agg, &roots);
        }
        else
        {
            A.AMGPMISAggregate(static_cast<T>(eps), &connections, &agg, &roots);
        }
        return {values_of(connections), values_of(agg), values_of(roots)};
    }

    // Nodes without strong connections stay outside the aggregates (-2), the others
    // form a valid aggregation
    template <typename T>
    void expect_aggregation(const Csr<T>& A, const Aggregation& in)
    {
        ASSERT_EQ(in.aggregates.size(), static_cast<size_t>(A.n));
        ASSERT_EQ(in.roots.size(), static_cast<size_t>(A.n));
        int64_t next = 0;
        for(int i = 0; i < A.n; ++i)
        {
            bool strong    = false;
            bool connected = false;
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                strong |= in.connections[j];
                connected |= in.connections[j] && in.aggregates[A.col[j]] == in.aggregates[i];
            }
            if(in.aggregates[i] < 0)
            {
                EXPECT_EQ(in.aggregates[i], -2) << "node " << i;
                EXPECT_FALSE(strong) << "node " << i;
                continue;
            }
            const int64_t root = in.roots[i];
            ASSERT_GE(root, 0) << "node " << i;
            ASSERT_LT(root, A.n) << "node " << i;
            EXPECT_EQ(in.roots[root], root) << "node " << i;
            EXPECT_EQ(in.aggregates[root], in.aggregates[i]) << "node " << i;
            if(root == i)
            {
                EXPECT_EQ(in.aggregates[i], next++) << "node " << i;
            }
            else
            {
                EXPECT_TRUE(connected) << "node " << i;
            }
        }
        EXPECT_EQ(*std::max_element(in.aggregates.begin(), in.aggregates.end()) + 1, next);
    }

    // smoothed_prolongation gives the nodes outside the aggregates a column -2
    template <typename T>
    Csr<T> without_unaggregated(const Csr<T>& P)
    {
        Csr<T> kept(P.n, P.m);
        for(int i = 0; i < P.n; ++i)
        {
            for(PtrType j = P.row_offset[i]; j < P.row_offset[i + 1]; ++j)
            {
                if(P.col[j] >= 0)
                {
                    kept.add(P.col[j], P.val[j]);
                }
            }
            kept.end_row();
        }
        return kept;
    }

    template <typename P>
    class multigrid_amg : public TypedBackendTest<P>
    {
    };

    // Strong in x, weak in y for the coupling strength 0.08
    template <typename T>
    Csr<T> anisotropic()
    {
        return grid<T>(8, 8, 1.0, 0.05, true);
    }

    template <typename T>
    std::vector<T> csr_values(const Csr<T>& A)
    {
        return A.val;
    }
}

TYPED_TEST_SUITE(multigrid_amg, WithBackends<RealTypes>, TypedName);

TYPED_TEST(multigrid_amg, smoothed_aggregation_halves_the_coupling_strength_on_each_level)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = anisotropic<T>();
    for(CoarseningStrategy strat : {Greedy, PMIS})
    {
        for(LumpingStrategy lumping : {AddWeakConnections, SubtractWeakConnections})
        {
            for(double relax : {2.0 / 3.0, 0.5})
            {
                SCOPED_TRACE(std::string(strat == Greedy ? "greedy" : "PMIS") + ", lumping "
                             + std::to_string(lumping) + ", relax " + std::to_string(relax));
                auto     A = make_local_matrix(TypeParam::backend, ref);
                SAAMG<T> amg;
                amg.SetOperator(*A);
                amg.SetCoarsestLevel(4);
                amg.SetCouplingStrength(static_cast<T>(0.08));
                amg.SetCoarseningStrategy(strat);
                amg.SetLumpingStrategy(lumping);
                amg.SetInterpRelax(static_cast<T>(relax));
                amg.Verbose(0);
                amg.Build();

                const Hierarchy<T> h = hierarchy_of(amg, ref);
                expect_galerkin_hierarchy(h, 4, 1.0);
                EXPECT_LE(h.A.back().n, 4);

                T eps = static_cast<T>(0.08);
                for(int l = 0; l + 1 < h.levels(); ++l, eps *= static_cast<T>(0.5))
                {
                    SCOPED_TRACE("level " + std::to_string(l));
                    const Aggregation in = aggregate(*level_operator(amg, *A, l), strat, eps);
                    ASSERT_IN_LOOP(in.connections == sa_strength(h.A[l], eps));
                    expect_aggregation(h.A[l], in);
                    expect_near(h.P[l],
                                without_unaggregated(smoothed_prolongation(
                                    h.A[l], in.connections, in.aggregates, relax, lumping)),
                                scaled_tol<T>(1e-13));
                }
            }
        }
    }
}

TYPED_TEST(multigrid_amg,
           unsmoothed_aggregation_scales_the_coarse_operators_by_the_over_interpolation)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = anisotropic<T>();
    for(CoarseningStrategy strat : {Greedy, PMIS})
    {
        // Factors up to 1 leave the coarse operators unscaled
        for(double over_interp : {1.5, 2.0, 1.0, 0.5})
        {
            SCOPED_TRACE(std::string(strat == Greedy ? "greedy" : "PMIS") + ", over-interpolation "
                         + std::to_string(over_interp));
            auto     A = make_local_matrix(TypeParam::backend, ref);
            UAAMG<T> amg;
            amg.SetOperator(*A);
            amg.SetCoarsestLevel(4);
            amg.SetCouplingStrength(static_cast<T>(0.08));
            amg.SetCoarseningStrategy(strat);
            amg.SetOverInterp(static_cast<T>(over_interp));
            amg.Verbose(0);
            amg.Build();

            const Hierarchy<T> h = hierarchy_of(amg, ref);
            const T scale = over_interp > 1.0 ? static_cast<T>(1) / static_cast<T>(over_interp)
                                              : static_cast<T>(1);
            expect_galerkin_hierarchy(h, 4, scale);
            EXPECT_LE(h.A.back().n, 4);

            T eps = static_cast<T>(0.08);
            for(int l = 0; l + 1 < h.levels(); ++l, eps *= static_cast<T>(0.5))
            {
                SCOPED_TRACE("level " + std::to_string(l));
                const Aggregation in = aggregate(*level_operator(amg, *A, l), strat, eps);
                expect_aggregation(h.A[l], in);
                EXPECT_EQ(aggregates_of(h.P[l]), in.aggregates);
            }
        }
    }
}

namespace
{
    template <typename T>
    void interpolate(const Matrix<T>&              A,
                     rocalution::InterpolationType type,
                     bool                          ff1,
                     const LocalVector<int>&       cf,
                     const LocalVector<bool>&      S,
                     Matrix<T>*                    P)
    {
        switch(type)
        {
        case rocalution::Direct:
            A.RSDirectInterpolation(cf, S, P);
            break;
        case rocalution::ExtPI:
            A.RSExtPIInterpolation(cf, S, ff1, P);
            break;
        case rocalution::MMExtPI:
            A.RSMMExtPIInterpolation(cf, S, P);
            break;
        case rocalution::MMExtPE:
            A.RSMMExtPEInterpolation(cf, S, P);
            break;
        }
    }

    // The truncated rows keep their largest entries, at most max_elmts of them and none
    // below trunc_factor times the largest, scaled to the row sums of the full rows
    template <typename T>
    void expect_truncation_of(const Csr<T>& P,
                              const Csr<T>& truncated,
                              float         trunc_factor,
                              int           max_elmts)
    {
        ASSERT_EQ(truncated.n, P.n);
        ASSERT_EQ(truncated.m, P.m);
        for(int i = 0; i < P.n; ++i)
        {
            std::map<int, T> full;
            T                largest = static_cast<T>(0);
            T                sum     = static_cast<T>(0);
            for(PtrType j = P.row_offset[i]; j < P.row_offset[i + 1]; ++j)
            {
                full[P.col[j]] = P.val[j];
                largest        = std::max(largest, std::abs(P.val[j]));
                sum += P.val[j];
            }

            std::map<int, T> kept;
            for(PtrType j = truncated.row_offset[i]; j < truncated.row_offset[i + 1]; ++j)
            {
                ASSERT_EQ(full.count(truncated.col[j]), 1u) << "row " << i;
                kept[truncated.col[j]] = truncated.val[j];
            }
            if(max_elmts > 0)
            {
                EXPECT_LE(kept.size(), static_cast<size_t>(max_elmts)) << "row " << i;
            }

            T smallest_kept = std::numeric_limits<T>::infinity();
            T kept_sum      = static_cast<T>(0);
            for(const auto& e : kept)
            {
                const T v = full[e.first];
                EXPECT_GE(std::abs(v), trunc_factor * largest) << "row " << i;
                smallest_kept = std::min(smallest_kept, std::abs(v));
                kept_sum += v;
            }
            for(const auto& e : full)
            {
                if(kept.count(e.first) == 0)
                {
                    EXPECT_LE(std::abs(e.second), smallest_kept) << "row " << i;
                }
            }
            for(const auto& e : kept)
            {
                EXPECT_NEAR(e.second, full[e.first] * sum / kept_sum, scaled_tol<T>(1e-13))
                    << "row " << i;
            }
        }
    }

    // PMIS forces only the points that depend on a new C point to F, so for a
    // non-symmetric strength a C point may depend on a later C point
    template <typename T>
    void expect_splitting(const Csr<T>& A, const std::vector<bool>& S, const std::vector<int>& cf)
    {
        bool symmetric = true;
        for(int i = 0; i < A.n; ++i)
        {
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                const int c    = A.col[j];
                bool      back = false;
                for(PtrType k = A.row_offset[c]; k < A.row_offset[c + 1]; ++k)
                {
                    back |= A.col[k] == i && S[k];
                }
                symmetric &= S[j] == back || c == i;
            }
        }
        if(symmetric)
        {
            expect_valid_splitting(A, S, cf);
            return;
        }

        ASSERT_EQ(cf.size(), static_cast<size_t>(A.n));
        for(int i = 0; i < A.n; ++i)
        {
            ASSERT_TRUE(cf[i] == 1 || cf[i] == 2) << "node " << i << ": " << cf[i];
            bool has_strong = false;
            bool coarse     = false;
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                has_strong |= S[j];
                coarse |= S[j] && cf[A.col[j]] == 1;
            }
            if(cf[i] == 2 && has_strong)
            {
                EXPECT_TRUE(coarse) << "F point " << i;
            }
        }
    }

    struct RsCase
    {
        rocalution::CoarseningStrategy strat;
        rocalution::InterpolationType  type;
        bool                           ff1;
        float                          trunc_factor;
        int                            max_elmts;
    };

    std::string rs_name(const RsCase& c)
    {
        const char* types[] = {"direct", "ext+i", "matrix-matrix ext+i", "matrix-matrix ext+e"};
        return std::string(c.strat == rocalution::Greedy ? "greedy, " : "PMIS, ") + types[c.type]
               + (c.ff1 ? " FF1" : "") + ", trunc_factor " + std::to_string(c.trunc_factor)
               + ", max_elmts " + std::to_string(c.max_elmts);
    }
}

TYPED_TEST(multigrid_amg, ruge_stueben_interpolates_with_the_selected_type)
{
    using T = typename TypeParam::value_type;
    // Zero row sums, so that the interpolations reproduce the constants
    const Csr<T> ref = grid<T>(8, 8, 1.0, 0.1, false);

    const std::vector<RsCase> cases = {{Greedy, Direct, false, 0.0f, 0},
                                       {PMIS, Direct, false, 0.0f, 0},
                                       {PMIS, ExtPI, false, 0.0f, 0},
                                       {PMIS, ExtPI, true, 0.0f, 0},
                                       {Greedy, MMExtPI, false, 0.0f, 0},
                                       {PMIS, MMExtPE, false, 0.0f, 0},
                                       {PMIS, ExtPI, false, 0.3f, 0},
                                       {PMIS, ExtPI, false, 0.0f, 2},
                                       {PMIS, MMExtPI, false, 0.2f, 3}};

    for(const RsCase& c : cases)
    {
        SCOPED_TRACE(rs_name(c));
        auto     A = make_local_matrix(TypeParam::backend, ref);
        RSAMG<T> amg;
        amg.SetOperator(*A);
        amg.SetCoarsestLevel(6);
        amg.SetStrengthThreshold(0.3f);
        amg.SetCoarseningStrategy(c.strat);
        amg.SetInterpolationType(c.type);
        amg.SetInterpolationFF1Limit(c.ff1);
        amg.SetInterpolationTruncationFactor(c.trunc_factor);
        amg.SetInterpolationMaxElmts(c.max_elmts);
        amg.Verbose(0);
        amg.Build();

        const Hierarchy<T> h = hierarchy_of(amg, ref);
        expect_galerkin_hierarchy(h, 6, 1.0);
        EXPECT_LE(h.A.back().n, 6);

        const bool truncated = c.trunc_factor > 0.0f || c.max_elmts > 0;
        for(int l = 0; l + 1 < h.levels(); ++l)
        {
            SCOPED_TRACE("level " + std::to_string(l));
            auto              Al = level_operator(amg, *A, l);
            LocalVector<int>  cf;
            LocalVector<bool> S;
            cf.CloneBackend(*Al);
            S.CloneBackend(*Al);
            if(c.strat == Greedy)
            {
                Al->RSCoarsening(0.3f, &cf, &S);
            }
            else
            {
                Al->RSPMISCoarsening(0.3f, &cf, &S);
            }
            ASSERT_IN_LOOP(values_of(S) == rs_strength(h.A[l], static_cast<double>(0.3f)));
            expect_splitting(h.A[l], values_of(S), values_of(cf));

            // The library's own interpolation of the level, so this checks that the
            // hierarchy uses the selected type and truncation, not the weights; those are
            // checked by hand in unit_test_local_matrix_amg.cpp
            Matrix<T> full;
            full.CloneBackend(*Al);
            interpolate(*Al, c.type, c.ff1, cf, S, &full);
            if(truncated)
            {
                expect_truncation_of(csr_of(full), h.P[l], c.trunc_factor, c.max_elmts);
            }
            else
            {
                expect_interpolation(h.P[l], values_of(cf));
                expect_near(h.P[l], csr_of(full), 0.0);
            }
        }
    }
}

TYPED_TEST(multigrid_amg, pairwise_aggregation_reaches_the_coarsening_factor)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = grid<T>(8, 8, 1.0, 1.0, true);
    struct PairwiseCase
    {
        unsigned int ordering;
        double       factor;
        double       beta;
    };
    const std::vector<PairwiseCase> cases = {{rocalution::NoOrdering, 4.0, 0.25},
                                             {rocalution::Connectivity, 4.0, 0.25},
                                             {rocalution::CMK, 4.0, 0.25},
                                             {rocalution::RCMK, 4.0, 0.25},
                                             {rocalution::MIS, 4.0, 0.25},
                                             {rocalution::MultiColoring, 4.0, 0.25},
                                             {rocalution::NoOrdering, 2.0, 0.25},
                                             {rocalution::NoOrdering, 6.0, 0.5}};

    for(const PairwiseCase& c : cases)
    {
        SCOPED_TRACE("ordering " + std::to_string(c.ordering) + ", factor "
                     + std::to_string(c.factor) + ", beta " + std::to_string(c.beta));
        auto     A = make_local_matrix(TypeParam::backend, ref);
        PWAMG<T> amg;
        amg.SetOperator(*A);
        amg.SetCoarsestLevel(4);
        amg.SetOrdering(c.ordering);
        amg.SetCoarseningFactor(c.factor);
        amg.SetBeta(static_cast<T>(c.beta));
        amg.Verbose(0);
        amg.Build();

        const Hierarchy<T> h = hierarchy_of(amg, ref);
        expect_galerkin_hierarchy(h, 4, 1.0);
        EXPECT_LE(h.A.back().n, 4);
        for(int l = 0; l + 1 < h.levels(); ++l)
        {
            SCOPED_TRACE("level " + std::to_string(l));
            const std::vector<int64_t> agg = aggregates_of(h.P[l]);
            for(int64_t k = 0; k < h.P[l].m; ++k)
            {
                EXPECT_NE(std::find(agg.begin(), agg.end(), k), agg.end())
                    << "empty aggregate " << k;
            }
            EXPECT_GE(static_cast<double>(h.A[l].n) / h.A[l + 1].n, c.factor);
        }
    }
}

TYPED_TEST(multigrid_amg, max_levels_limits_the_hierarchy)
{
    using T = typename TypeParam::value_type;
    for(const AmgCase<T>& c : amg_classes<T>())
    {
        for(int max_levels : {2, 3})
        {
            SCOPED_TRACE(c.name + ", " + std::to_string(max_levels) + " levels");
            const Csr<T> ref = grid<T>(8, 8, 1.0, 1.0, true);
            auto         A   = make_local_matrix(TypeParam::backend, ref);
            auto         amg = c.make();
            amg->SetOperator(*A);
            amg->SetCoarsestLevel(2);
            amg->SetMaxLevels(max_levels);
            amg->Verbose(0);
            amg->Build();

            EXPECT_EQ(amg->GetNumLevels(), max_levels);
            const Hierarchy<T> h = hierarchy_of(*amg, ref);
            expect_galerkin_hierarchy(h, 2, 1.0);
            EXPECT_GT(h.A.back().n, 2);
        }
    }
}

namespace
{
    // Disjoint paths of three nodes. Each path is one aggregate, so the first coarse
    // operator is diagonal and has no strong connections left.
    template <typename T>
    Csr<T> paths(int count)
    {
        Csr<T> A(3 * count);
        for(int p = 0; p < count; ++p)
        {
            for(int i = 0; i < 3; ++i)
            {
                for(int j = 0; j < 3; ++j)
                {
                    if(std::abs(i - j) <= 1)
                    {
                        A.add(3 * p + j, static_cast<T>(i == j ? 2 : -1));
                    }
                }
                A.end_row();
            }
        }
        return A;
    }

    template <typename T>
    bool diagonal(const Csr<T>& A)
    {
        for(int i = 0; i < A.n; ++i)
        {
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                if(A.col[j] != i && A.val[j] != static_cast<T>(0))
                {
                    return false;
                }
            }
        }
        return true;
    }
}

TYPED_TEST(multigrid_amg, coarsening_stops_at_a_level_without_strong_connections)
{
    using T             = typename TypeParam::value_type;
    const Csr<T> ref    = paths<T>(4);
    const auto   expect = [&](const AmgCase<T>& c) {
        SCOPED_TRACE(c.name);
        auto A   = make_local_matrix(TypeParam::backend, ref);
        auto amg = c.make();
        amg->SetOperator(*A);
        amg->SetCoarsestLevel(2);
        amg->Verbose(0);

        testing::internal::CaptureStdout();
        amg->Build();
        const std::string out = testing::internal::GetCapturedStdout();
        EXPECT_NE(out.find("Requested coarse operator size cannot be reached"), std::string::npos);

        const Hierarchy<T> h = hierarchy_of(*amg, ref);
        expect_galerkin_hierarchy(h, 2, 1.0);
        EXPECT_EQ(h.levels(), 2);
        EXPECT_TRUE(diagonal(h.A.back()));
        EXPECT_EQ(h.A.back().n, 4);
    };

    expect({"SA greedy", [] { return make_sa<T>(Greedy, AddWeakConnections); }, CycleOptions()});
    expect({"RS", [] { return make_rs<T>(PMIS, Direct); }, CycleOptions()});

    const auto        A  = make_local_matrix(TypeParam::backend, ref);
    const Aggregation in = aggregate(*A, PMIS, 0.01);
    expect_valid_aggregation(ref, in.connections, in.aggregates, in.roots);
    KNOWN_BUG("PMIS makes two strongly connected nodes of a path roots");
    expect_independent_roots(ref, in.connections, in.roots);
    expect({"SA PMIS", [] { return make_sa<T>(PMIS, AddWeakConnections); }, CycleOptions()});
}

TYPED_TEST(multigrid_amg, unsmoothed_coarsening_stops_at_a_level_without_strong_connections)
{
    using T               = typename TypeParam::value_type;
    const Csr<T>      ref = paths<T>(4);
    const Aggregation in  = aggregate(*make_local_matrix(TypeParam::backend, ref), Greedy, 0.01);
    expect_aggregation(ref, in);
    EXPECT_EQ(*std::max_element(in.aggregates.begin(), in.aggregates.end()), 3);

    KNOWN_BUG("UAAMG::Aggregate_ does not stop at a level without aggregates, so the Galerkin "
              "product of the empty prolongation fails an assert");
    const Backend backend = TypeParam::backend;
    EXPECT_EXIT(
        {
            auto A   = make_local_matrix(backend, ref);
            auto amg = make_ua<T>(Greedy, 1.0);
            amg->SetOperator(*A);
            amg->SetCoarsestLevel(2);
            amg->Verbose(0);
            amg->Build();
            const Hierarchy<T> h = hierarchy_of(*amg, ref);
            std::exit(h.levels() == 2 && h.A.back().n == 4 && diagonal(h.A.back()) ? 0 : 2);
        },
        testing::ExitedWithCode(0),
        "");
}

TYPED_TEST(multigrid_amg, cycle_matches_the_dense_cycle_on_the_built_hierarchy)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = grid<T>(8, 8, 1.0, 1.0, true);
    for(const AmgCase<T>& c : amg_classes<T>())
    {
        for(int pre : {1, 2})
        {
            SCOPED_TRACE(c.name + ", " + std::to_string(pre) + " pre-smoothing steps");
            CycleOptions o = c.cycle;
            o.pre          = pre;
            ManualAmg<T> m(TypeParam::backend, ref, c, 4);
            set_cycle(*m.amg, o);
            m.amg->Build();

            const Hierarchy<T> h = hierarchy_of(*m.amg, ref);
            ASSERT_IN_LOOP(h.levels() >= 3);
            const Vec<T> b  = rhs<T>(ref.n);
            const Vec<T> x0 = initial_guess<T>(ref.n);
            expect_near(
                solve(*m.amg, *m.A, b, x0, 2), cycled(DenseCycle<T>(h, o), b, x0, 2), tol<T>());
        }
    }
}

TYPED_TEST(multigrid_amg, default_smoothers_and_coarse_solver)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = grid<T>(8, 8, 1.0, 1.0, true);
    for(const AmgCase<T>& c : amg_classes<T>())
    {
        for(unsigned int format : {rocalution::CSR, rocalution::ELL})
        {
            SCOPED_TRACE(c.name + ", smoother format " + rocalution::_matrix_format_names[format]);
            auto A   = make_local_matrix(TypeParam::backend, ref);
            auto amg = c.make();
            amg->SetOperator(*A);
            // CG solves the coarsest level of at most 2 unknowns to rounding
            amg->SetCoarsestLevel(2);
            amg->SetDefaultSmootherFormat(format);
            amg->Verbose(0);
            amg->Build();

            const Hierarchy<T> h = hierarchy_of(*amg, ref);
            ASSERT_IN_LOOP(h.A.back().n <= 2);
            for(int l = 1; l < h.levels(); ++l)
            {
                EXPECT_EQ(Peek<T>::op(*amg, l).GetFormat(), rocalution::CSR);
            }

            CycleOptions o  = c.cycle;
            o.omega         = static_cast<double>(2.f / 3.f);
            const Vec<T> b  = rhs<T>(ref.n);
            const Vec<T> x0 = initial_guess<T>(ref.n);
            expect_near(solve(*amg, *A, b, x0, 1),
                        cycled(DenseCycle<T>(h, o), b, x0, 1),
                        loose_tol<T>(1e-10));
        }
    }
}

TYPED_TEST(multigrid_amg, scaling_and_formats_of_the_levels)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = grid<T>(8, 8, 1.0, 1.0, true);
    struct Result
    {
        std::string name;
        Vec<T>      x;
        Vec<T>      expected;
    };
    std::vector<Result> results;
    for(const AmgCase<T>& c : amg_classes<T>())
    {
        for(unsigned int format : {rocalution::CSR, rocalution::ELL})
        {
            const std::string name = c.name + ", " + rocalution::_matrix_format_names[format];
            SCOPED_TRACE(name);
            CycleOptions o = c.cycle;
            o.scaling      = true;
            ManualAmg<T> m(TypeParam::backend, ref, c, 4);
            set_cycle(*m.amg, o);
            m.amg->SetOperatorFormat(format, 1);
            m.amg->Build();

            for(int l = 1; l < m.amg->GetNumLevels(); ++l)
            {
                EXPECT_EQ(Peek<T>::op(*m.amg, l).GetFormat(), format) << "level " << l;
            }
            const Hierarchy<T> h  = hierarchy_of(*m.amg, ref);
            const Vec<T>       b  = rhs<T>(ref.n);
            const Vec<T>       x0 = initial_guess<T>(ref.n);
            results.push_back(
                {name, solve(*m.amg, *m.A, b, x0, 2), cycled(DenseCycle<T>(h, o), b, x0, 2)});

            // The format of the levels does not change the cycle
            if(format == rocalution::ELL)
            {
                expect_near(results.back().x, results[results.size() - 2].x, tol<T>());
            }
        }
    }

    KNOWN_BUG("coarse levels scale the correction by (b, e) / (Ae, e) instead of (r, e) / (Ae, e)");
    for(const Result& r : results)
    {
        SCOPED_TRACE(r.name);
        expect_near(r.x, r.expected, tol<T>());
    }
}

TYPED_TEST(multigrid_amg, rebuild_numeric_keeps_the_transfer_operators)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = grid<T>(8, 8, 1.0, 1.0, true);
    // The same pattern, with other symmetric values
    const Csr<T> next = transformed(ref, [](int i, int j, T v) {
        return i == j ? v + static_cast<T>(2)
                      : v * (static_cast<T>(1) + static_cast<T>(0.5) * static_cast<T>((i + j) % 2));
    });

    struct Variant
    {
        std::string  name;
        unsigned int operator_format;
        unsigned int level_format;
    };
    const std::vector<Variant> variants = {{"CSR", rocalution::CSR, rocalution::CSR},
                                           {"operator in ELL", rocalution::ELL, rocalution::CSR},
                                           {"levels in ELL", rocalution::CSR, rocalution::ELL}};

    for(const AmgCase<T>& c : amg_classes<T>())
    {
        for(const Variant& v : variants)
        {
            SCOPED_TRACE(c.name + ", " + v.name);
            ManualAmg<T> m(TypeParam::backend, ref, c, 4);
            m.amg->SetOperatorFormat(v.level_format, 1);
            m.amg->Build();
            const Hierarchy<T> before = hierarchy_of(*m.amg, ref);

            Vec<T> val = csr_values(next);
            m.A->UpdateValuesCSR(val.data());
            m.A->ConvertTo(v.operator_format, 1);
            m.amg->ReBuildNumeric();

            const Hierarchy<T> after    = hierarchy_of(*m.amg, next);
            const Hierarchy<T> expected = with_finest(before, next);
            // Pairwise coarsens each operator in the format of the finer one, and
            // converts only to a format other than CSR
            const unsigned int format = c.name == "pairwise" && v.level_format == rocalution::CSR
                                            ? v.operator_format
                                            : v.level_format;
            ASSERT_IN_LOOP(after.levels() == before.levels());
            for(int l = 0; l + 1 < after.levels(); ++l)
            {
                SCOPED_TRACE("level " + std::to_string(l + 1));
                expect_near(after.R[l], before.R[l], 0.0);
                expect_near(after.P[l], before.P[l], 0.0);
                expect_dense_near(after.A[l + 1], expected.A[l + 1], tol<T>());
                EXPECT_EQ(Peek<T>::op(*m.amg, l + 1).GetFormat(), format);
            }

            const Vec<T> b  = rhs<T>(ref.n);
            const Vec<T> x0 = initial_guess<T>(ref.n);
            expect_near(solve(*m.amg, *m.A, b, x0, 1),
                        cycled(DenseCycle<T>(expected, c.cycle), b, x0, 1),
                        tol<T>());
        }
    }
}

TYPED_TEST(multigrid_amg, unsmoothed_rebuild_numeric_keeps_the_over_interpolation)
{
    using T              = typename TypeParam::value_type;
    const Csr<T>     ref = grid<T>(8, 8, 1.0, 1.0, true);
    const AmgCase<T> c   = {"UA", [] { return make_ua<T>(rocalution::PMIS, 1.5); }, CycleOptions()};
    ManualAmg<T>     m(TypeParam::backend, ref, c, 4);
    m.amg->Build();

    const Hierarchy<T> built = hierarchy_of(*m.amg, ref);
    expect_galerkin_hierarchy(built, 4, static_cast<T>(1) / static_cast<T>(1.5));

    KNOWN_BUG("UAAMG::ReBuildNumeric does not scale the coarse operators by the "
              "over-interpolation");
    m.amg->ReBuildNumeric();
    const Hierarchy<T> rebuilt = hierarchy_of(*m.amg, ref);
    for(int l = 1; l < built.levels(); ++l)
    {
        SCOPED_TRACE("level " + std::to_string(l));
        expect_dense_near(rebuilt.A[l], built.A[l], tol<T>());
    }
}

TYPED_TEST(multigrid_amg, build_after_clear_builds_the_same_hierarchy)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = grid<T>(8, 8, 1.0, 1.0, true);
    for(const AmgCase<T>& c : amg_classes<T>())
    {
        SCOPED_TRACE(c.name);
        const std::unique_ptr<Matrix<T>> A   = make_local_matrix(TypeParam::backend, ref);
        const std::unique_ptr<AMG<T>>    amg = c.make();
        amg->SetOperator(*A);
        amg->SetCoarsestLevel(4);
        amg->Verbose(0);

        amg->Build();
        const Hierarchy<T> first = hierarchy_of(*amg, ref);

        amg->Build();
        const Hierarchy<T> second = hierarchy_of(*amg, ref);

        amg->Clear();
        amg->Build();
        const Hierarchy<T> third = hierarchy_of(*amg, ref);

        for(const Hierarchy<T>* h : {&second, &third})
        {
            ASSERT_IN_LOOP(h->levels() == first.levels());
            for(int l = 0; l + 1 < first.levels(); ++l)
            {
                expect_near(h->P[l], first.P[l], 0.0);
                expect_near(h->A[l + 1], first.A[l + 1], 0.0);
            }
        }
    }
}

TYPED_TEST(multigrid_lifetime, rebuild_of_an_amg_keeps_the_user_smoothers)
{
    using T              = typename TypeParam::value_type;
    const Csr<T>     ref = grid<T>(8, 8, 1.0, 1.0, true);
    const AmgCase<T> c   = amg_classes<T>()[0];
    {
        ManualAmg<T> m(Backend::Host, ref, c, 4);
        m.amg->Build();
        const Hierarchy<T> h  = hierarchy_of(*m.amg, ref);
        const Vec<T>       b  = rhs<T>(ref.n);
        const Vec<T>       x0 = initial_guess<T>(ref.n);
        expect_near(solve(*m.amg, *m.A, b, x0, 1),
                    cycled(DenseCycle<T>(h, c.cycle), b, x0, 1),
                    loose_tol<T>(1e-10));
    }

    KNOWN_BUG("BaseMultiGrid::Finalize clears the user's FixedPoint smoothers, which drops their "
              "preconditioners, so the next Build asserts in FixedPoint::Build");
    EXPECT_EXIT(
        {
            ManualAmg<T> m(Backend::Host, ref, c, 4);
            m.amg->Build();
            m.amg->Build();
            const Hierarchy<T> h     = hierarchy_of(*m.amg, ref);
            const Vec<T>       b     = rhs<T>(ref.n);
            const Vec<T>       x0    = initial_guess<T>(ref.n);
            const Vec<T>       x     = solve(*m.amg, *m.A, b, x0, 1);
            const Vec<T>       e     = cycled(DenseCycle<T>(h, c.cycle), b, x0, 1);
            double             error = 0.0;
            for(size_t i = 0; i < x.size(); ++i)
            {
                error = std::max(error, static_cast<double>(std::abs(x[i] - e[i])));
            }
            std::exit(error < loose_tol<T>(1e-10) ? 0 : 2);
        },
        testing::ExitedWithCode(0),
        "");
}

TYPED_TEST(multigrid_amg, verbose_solve_prints_the_class)
{
    using T                                                   = typename TypeParam::value_type;
    const Csr<T>                                          ref = grid<T>(8, 8, 1.0, 1.0, true);
    const std::vector<std::pair<AmgCase<T>, std::string>> cases
        = {{amg_classes<T>()[0], "SAAMG solver starts"},
           {{"UA", [] { return make_ua<T>(rocalution::PMIS, 1.5); }, CycleOptions()},
            "UAAMG solver starts"},
           {amg_classes<T>()[2], "AMG Ruge-Stuben using PMIS coarsening with Ext+i interpolation"},
           {amg_classes<T>()[3], "AMG using pairwise aggregation"}};

    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.first.name);
        ManualAmg<T> m(TypeParam::backend, ref, c.first, 4);
        m.amg->Build();
        m.amg->Verbose(1);

        testing::internal::CaptureStdout();
        solve(*m.amg, *m.A, rhs<T>(ref.n), initial_guess<T>(ref.n), 1);
        const std::string out = testing::internal::GetCapturedStdout();
        EXPECT_NE(out.find(c.second), std::string::npos) << out;
    }
}

// The setup steps are deterministic, so both backends build the same hierarchy; the steps
// themselves are checked in unit_test_local_matrix_amg.cpp
TYPED_TEST(multigrid_backends, hierarchies_agree_on_host_and_accelerator)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = grid<T>(8, 8, 1.0, 0.1, false);

    std::vector<AmgCase<T>> cases = amg_classes<T>();
    cases.push_back({"SA greedy", [] { return make_sa<T>(Greedy, SubtractWeakConnections); }, {}});
    cases.push_back({"UA greedy", [] { return make_ua<T>(Greedy, 1.5); }, {}});
    for(rocalution::InterpolationType type : {Direct, ExtPI, MMExtPI, MMExtPE})
    {
        for(float factor : {0.0f, 0.2f, 0.5f})
        {
            for(int max_elmts : {0, 2, 4})
            {
                const RsCase rs = {PMIS, type, false, factor, max_elmts};
                cases.push_back({"RS " + rs_name(rs),
                                 [rs] {
                                     auto amg = std::make_unique<RSAMG<T>>();
                                     amg->SetCoarseningStrategy(rs.strat);
                                     amg->SetInterpolationType(rs.type);
                                     amg->SetInterpolationTruncationFactor(rs.trunc_factor);
                                     amg->SetInterpolationMaxElmts(rs.max_elmts);
                                     return std::unique_ptr<AMG<T>>(std::move(amg));
                                 },
                                 {}});
            }
        }
    }

    for(const AmgCase<T>& c : cases)
    {
        SCOPED_TRACE(c.name);
        std::vector<Hierarchy<T>> built;
        for(Backend b : {Backend::Host, Backend::Accelerator})
        {
            auto A   = make_local_matrix(b, ref);
            auto amg = c.make();
            amg->SetOperator(*A);
            amg->SetCoarsestLevel(6);
            amg->Verbose(0);
            amg->Build();
            built.push_back(hierarchy_of(*amg, ref));
        }
        ASSERT_IN_LOOP(built[0].levels() == built[1].levels());
        for(int l = 0; l + 1 < built[0].levels(); ++l)
        {
            SCOPED_TRACE("level " + std::to_string(l + 1));
            expect_near(built[1].P[l], built[0].P[l], tol<T>());
            expect_dense_near(built[1].A[l + 1], built[0].A[l + 1], tol<T>());
        }
    }
}

TYPED_TEST(multigrid_backends, amg_levels_follow_the_moves_of_the_solver)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = grid<T>(8, 8, 1.0, 1.0, true);
    struct Result
    {
        std::string name;
        Vec<T>      x;
        Vec<T>      expected;
    };
    std::vector<Result> results;
    for(const AmgCase<T>& c : amg_classes<T>())
    {
        SCOPED_TRACE(c.name);
        CycleOptions o = options(rocalution::Kcycle, 1, 2, true);
        ManualAmg<T> m(Backend::Host, ref, c, 4);
        set_cycle(*m.amg, o);

        // The hierarchy is built on the backend of the operator
        m.amg->MoveToAccelerator();
        m.amg->Build();
        const int levels = m.amg->GetNumLevels();
        EXPECT_EQ(coarse_on_host(*m.amg), std::vector<bool>(levels - 1, true));

        const Hierarchy<T> h        = hierarchy_of(*m.amg, ref);
        const Vec<T>       b        = rhs<T>(ref.n);
        const Vec<T>       x0       = initial_guess<T>(ref.n);
        const Vec<T>       expected = cycled(DenseCycle<T>(h, o), b, x0, 2);

        m.A->MoveToAccelerator();
        m.amg->MoveToAccelerator();
        EXPECT_EQ(coarse_on_host(*m.amg), std::vector<bool>(levels - 1, false));
        EXPECT_EQ(transfers_on_host(*m.amg), std::vector<bool>(levels - 1, false));
        results.push_back({c.name, solve(*m.amg, *m.A, b, x0, 2), expected});

        // Both placements compute the same cycle
        m.A->MoveToHost();
        m.amg->MoveToHost();
        EXPECT_EQ(coarse_on_host(*m.amg), std::vector<bool>(levels - 1, true));
        EXPECT_EQ(transfers_on_host(*m.amg), std::vector<bool>(levels - 1, true));
        expect_near(solve(*m.amg, *m.A, b, x0, 2), results.back().x, tol<T>());
    }

    KNOWN_BUG("coarse levels scale the correction by (b, e) / (Ae, e) instead of (r, e) / (Ae, e)");
    for(const Result& r : results)
    {
        SCOPED_TRACE(r.name);
        expect_near(r.x, r.expected, tol<T>());
    }
}

TYPED_TEST(multigrid_backends, rebuild_numeric_keeps_the_host_levels)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = grid<T>(8, 8, 1.0, 1.0, true);
    const Csr<T> next
        = transformed(ref, [](int i, int j, T v) { return i == j ? v + static_cast<T>(2) : v; });

    // Pairwise with the levels below the finest on the accelerator
    {
        ManualAmg<T> m(Backend::Accelerator, ref, amg_classes<T>()[3], 4);
        m.amg->Build();
        const int levels = m.amg->GetNumLevels();
        ASSERT_GE(levels, 3);
        m.amg->SetHostLevels(levels - 2);
        const Hierarchy<T> before = hierarchy_of(*m.amg, ref);

        Vec<T> val = csr_values(next);
        m.A->UpdateValuesCSR(val.data());
        m.amg->ReBuildNumeric();

        std::vector<bool> host(levels - 1, true);
        host[0] = false;
        EXPECT_EQ(coarse_on_host(*m.amg), host);
        const Hierarchy<T> expected = with_finest(before, next);
        const Hierarchy<T> after    = hierarchy_of(*m.amg, next);
        for(int l = 1; l < levels; ++l)
        {
            expect_dense_near(after.A[l], expected.A[l], tol<T>());
        }
        const Vec<T> b  = rhs<T>(ref.n);
        const Vec<T> x0 = initial_guess<T>(ref.n);
        expect_near(solve(*m.amg, *m.A, b, x0, 1),
                    cycled(DenseCycle<T>(expected, pairwise_cycle()), b, x0, 1),
                    tol<T>());
    }

    // The coarse operators are recomputed on the accelerator, from transfer operators on
    // the host
    KNOWN_BUG("ReBuildNumeric with host levels clones the coarse operators to the backend of "
              "the finest operator, so SA, UA and RS fail the backend assert of "
              "TripleMatrixProduct, and pairwise that of CoarsenOperator when all coarse levels "
              "are on the host");
    for(const AmgCase<T>& c : amg_classes<T>())
    {
        SCOPED_TRACE(c.name);
        for(int host_levels : {1, 2})
        {
            SCOPED_TRACE(std::to_string(host_levels) + " host levels");
            EXPECT_EXIT(
                {
                    ManualAmg<T> m(Backend::Accelerator, ref, c, 4);
                    m.amg->Build();
                    m.amg->SetHostLevels(host_levels);
                    Vec<T> val = csr_values(next);
                    m.A->UpdateValuesCSR(val.data());
                    m.amg->ReBuildNumeric();
                    solve(*m.amg, *m.A, rhs<T>(ref.n), initial_guess<T>(ref.n), 1);
                    std::exit(0);
                },
                testing::ExitedWithCode(0),
                "");
        }
    }
}

namespace
{
    template <typename P>
    class multigrid_hierarchy : public TypedBackendTest<P>
    {
    };
}

TYPED_TEST_SUITE(multigrid_hierarchy, HostOnly<RealTypes>, TypedName);

// Ruge-Stueben about halves the 1D Laplacian on every level, so 3 * 2^18 unknowns need
// at least 20 levels to reach 2
TYPED_TEST(multigrid_hierarchy, deep_hierarchy_warns_about_its_number_of_levels)
{
    using T     = typename TypeParam::value_type;
    const int n = 3 << 18;
    Csr<T>    ref(n);
    for(int i = 0; i < n; ++i)
    {
        if(i > 0)
        {
            ref.add(i - 1, static_cast<T>(-1));
        }
        ref.add(i, static_cast<T>(2));
        if(i + 1 < n)
        {
            ref.add(i + 1, static_cast<T>(-1));
        }
        ref.end_row();
    }
    auto     A = make_local_matrix(Backend::Host, ref);
    RSAMG<T> amg;
    amg.SetOperator(*A);
    amg.SetCoarsestLevel(2);
    amg.Verbose(0);

    testing::internal::CaptureStdout();
    amg.BuildHierarchy();
    const std::string out = testing::internal::GetCapturedStdout();

    const int levels = amg.GetNumLevels();
    ASSERT_GE(levels, 20);
    int64_t previous = n;
    for(int l = 1; l < levels; ++l)
    {
        EXPECT_LT(Peek<T>::op(amg, l).GetM(), previous);
        EXPECT_GE(Peek<T>::op(amg, l).GetM() * 3, previous);
        previous = Peek<T>::op(amg, l).GetM();
    }
    EXPECT_LE(previous, 2);
    EXPECT_NE(out.find("Current number of levels: " + std::to_string(levels)), std::string::npos);
    EXPECT_NE(out.find("Current number of levels: 20"), std::string::npos);
    EXPECT_EQ(out.find("Current number of levels: 19"), std::string::npos);
    amg.Build();
}

TYPED_TEST(multigrid_errors, building_an_amg_without_an_operator_asserts)
{
    using T = typename TypeParam::value_type;
    SAAMG<T> amg;
    EXPECT_ASSERT_DEATH(amg.Build(), "this->op_ != NULL");
}

TYPED_TEST(multigrid_errors, amg_of_a_problem_within_the_coarsest_size_is_a_fatal_error)
{
    using T = typename TypeParam::value_type;
    EXPECT_FATAL_ERROR(
        {
            auto     A = make_local_matrix(Backend::Host, grid<T>(4, 4, 1.0, 1.0, true));
            SAAMG<T> amg;
            amg.SetOperator(*A);
            amg.SetCoarsestLevel(16);
            amg.Build();
        },
        "base_amg\\.cpp");
}
