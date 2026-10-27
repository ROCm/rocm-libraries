/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <typeinfo>
#include <vector>

#include <Tensile/Tensile.hpp>
#include <Tensile/Task.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    /**
 * \ingroup Tensile
 * \defgroup SolutionLibrary Solution Library Classes
 *
 * \copydoc TensileLite::SolutionLibrary
 */

    enum class SolutionLibrarySearchType
    {
        DEFAULT        = 1,
        GEMM_TYPE_ONLY = 2,
        HARDWARE_ONLY  = 3,
        COUNT          = 4
    };

    template <typename MySolution, typename MyProblem>
    inline bool isGemmTypeSame(const MySolution& solutions, const MyProblem& problem)
    {
        if(solutions.problemType.transA == problem.transA()
           && solutions.problemType.transB == problem.transB()
           && solutions.problemType.aType == problem.a().dataType()
           && solutions.problemType.bType == problem.b().dataType()
           && solutions.problemType.cType == problem.c().dataType()
           && solutions.problemType.dType == problem.d().dataType()
           && solutions.problemType.computeType == problem.computeType()
           && solutions.problemType.groupedGemm == problem.groupedGemm()
           && solutions.problemType.mxBlockA == problem.mxBlockA()
           && solutions.problemType.mxBlockB == problem.mxBlockB())
            return true;
        return false;
    }

    template <typename MySolution, typename MyProblem>
    inline bool softwarePredicate(const SolutionLibrarySearchType& searchType,
                                  Task&                            task,
                                  Hardware const&                  hardware,
                                  const MySolution&                solutions,
                                  const MyProblem&                 problem)
    {
        switch(searchType)
        {
        case SolutionLibrarySearchType::DEFAULT:
            // streamKDynamicQueueSupported() excludes StreamK dynamic-queue /
            // work-stealing solutions (SK4 and the dynamic sub-path of SK5) on
            // devices whose XCD count is not a power of two, warning the user
            // once. This is reject-and-continue: selection falls through to
            // another (SK3-static / non-StreamK) solution for the GEMM.
            // uniformSummationOrderSupported() is the same kind of filter:
            // under USO it admits only kernels this problem can launch
            // (Synchronizer allocation is the remaining solve()-only clause).
            // Sites that gained this conjunction with the per-tile split mapping
            // reach it through selectionPredicate(), which with USO off falls back
            // to the narrower problemPredicate && taskPredicate it subsumes.
            return (*solutions.problemPredicate)(problem) && (*solutions.taskPredicate)(task)
                   && solutions.streamKDynamicQueueSupported(problem, hardware)
                   && solutions.uniformSummationOrderSupported(problem, hardware);
            break;
        case SolutionLibrarySearchType::GEMM_TYPE_ONLY:
            return isGemmTypeSame(solutions, problem);
            break;
        case SolutionLibrarySearchType::HARDWARE_ONLY:
            return true;
            break;
        default:
            break;
        }
        return false;
    }

    // With USO off this applies only problemPredicate && taskPredicate &&
    // hardwarePredicate. With USO on, softwarePredicate() adds
    // streamKDynamicQueueSupported() and uniformSummationOrderSupported(), both
    // live filters on that arm. hardwarePredicate keeps each arm's original
    // position -- last off, first on -- because taskPredicate can warn via
    // requiredWorkspaceSize(), so short-circuit order is user-visible.
    template <typename MySolution, typename MyProblem>
    inline bool selectionPredicate(Task&             task,
                                   Hardware const&   hardware,
                                   const MySolution& solution,
                                   const MyProblem&  problem)
    {
        if(problem.getParams().uniformSummationOrder())
            return (*solution.hardwarePredicate)(hardware)
                   && softwarePredicate(
                       SolutionLibrarySearchType::DEFAULT, task, hardware, solution, problem);

        return (*solution.problemPredicate)(problem) && (*solution.taskPredicate)(task)
               && (*solution.hardwarePredicate)(hardware);
    }

    template <typename MySolution>
    using SolutionSet = std::set<std::shared_ptr<MySolution>>;
    template <typename MySolution>
    using SolutionVector = std::vector<std::shared_ptr<MySolution>>;

    /**
 * The route of a findTopSolutions call: which row kinds
 * ExactLogicLibrary::findTopRows drops before their predicate, as bits.
 */
    namespace SelectionRoute
    {
        /// Equality and Range rows (TENSILE_PREDICTION_LIB, or dynamic Stream-K
        /// scheduling).
        constexpr unsigned DropsEqualityAndRange = 1;
        /// GridBased and FreeSize rows (TENSILE_PREDICTION_LIB only).
        constexpr unsigned DropsGridBasedAndFreeSize = 2;
        constexpr unsigned Count                     = 4;
    }

    /// What the findTopSolutions calls of a library can return on a route,
    /// whatever the problem (SolutionLibrary::routeYield).
    enum class RouteYield : std::uint8_t
    {
        /// Solutions, possibly.
        Some,
        /// No solution; no row is asked about of a TilewrightRowOrder.
        Nothing,
        /// Only the entries of Equality tables whose key is the problem's;
        /// only the rows of those tables are asked about of a
        /// TilewrightRowOrder. returnsNothing tells, per problem.
        ExactKeys,
    };

    /**
 * Makes the solutions a findTopSolutions call has collected so far visible to
 * the libraries it calls on the same thread, for as long as it lives. Scopes
 * nest; contains() searches all of them.
 */
    class TENSILELITEHOST_EXPORT SelectedSolutionsScope
    {
    public:
        template <typename MySolution>
        explicit SelectedSolutionsScope(SolutionVector<MySolution> const& selected)
            : m_selected(&selected)
            , m_contains(&containsIndex<MySolution>)
        {
            auto& head = innermost();
            m_outer    = head;
            head       = this;
        }

        ~SelectedSolutionsScope()
        {
            innermost() = m_outer;
        }

        SelectedSolutionsScope(SelectedSolutionsScope const&)            = delete;
        SelectedSolutionsScope& operator=(SelectedSolutionsScope const&) = delete;

        /// Whether an enclosing scope holds a solution with this index.
        static bool contains(int index)
        {
            for(auto const* scope = innermost(); scope != nullptr; scope = scope->m_outer)
            {
                if(scope->m_contains(scope->m_selected, index))
                    return true;
            }
            return false;
        }

    private:
        template <typename MySolution>
        static bool containsIndex(void const* selected, int index)
        {
            for(auto const& solution : *static_cast<SolutionVector<MySolution> const*>(selected))
            {
                if(solution && solution->index == index)
                    return true;
            }
            return false;
        }

        static SelectedSolutionsScope const*& innermost();

        void const* m_selected;
        bool (*m_contains)(void const*, int);
        SelectedSolutionsScope const* m_outer;
    };

    /**
 * Makes the predicate of a row that matched the problem visible to the
 * libraries the row's library calls on the same thread, for as long as it
 * lives. Scopes nest.
 */
    class TENSILELITEHOST_EXPORT MatchedPredicateScope
    {
    public:
        template <typename Predicate>
        explicit MatchedPredicateScope(Predicate const* predicate)
            : m_predicate(predicate)
            , m_type(&typeid(Predicate))
        {
            auto& head = innermost();
            m_outer    = head;
            head       = this;
        }

        ~MatchedPredicateScope()
        {
            innermost() = m_outer;
        }

        MatchedPredicateScope(MatchedPredicateScope const&)            = delete;
        MatchedPredicateScope& operator=(MatchedPredicateScope const&) = delete;

        /// The predicate of the innermost scope when that scope was made from
        /// a Predicate, or null.
        template <typename Predicate>
        static Predicate const* innermostOf()
        {
            auto const* scope = innermost();
            if(scope == nullptr || *scope->m_type != typeid(Predicate))
                return nullptr;
            return static_cast<Predicate const*>(scope->m_predicate);
        }

    private:
        static MatchedPredicateScope const*& innermost();

        void const*                  m_predicate;
        std::type_info const*        m_type;
        MatchedPredicateScope const* m_outer;
    };

    /**
 * \ingroup SolutionLibrary
 *
 * @brief Abstract base class for Library objects that can provide a
 * mapping from `Problem` and `Hardware` objects to Solution objects.
 *
 * A complete SolutionLibrary is a tree of objects which each handles a
 * single aspect of selecting a solution for a given problem. Each node in
 * the tree will handle an aspect such as:
 * - Compatibility with a particular model of GPU
 * - Selecting kernels that solve a particular type of problem (transpose,
 *   data type, etc.)
 * - Selecting the fastest kernel based on benchmark results or other logic
 * - Ensuring that a problem is compatible with any assumptions made by a
 *   particular kernel (e.g. size or stride requirements)
 *
 * A particular complete library might look like:
 * - Master library which manages serialization
 *   - GPU selection
 *     - Problem type selection
 *        - Predicated logic for specific sizes
 *          - Matching library based on benchmarks
 *            - Individual kernels
 *
 */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct TENSILELITEHOST_EXPORT SolutionLibrary
    {
        virtual ~SolutionLibrary() = default;

        /**
   * Returns the single `Solution` object that best solves this
   * particular `problem` on this particular piece of `hardware`.
   *
   * May return `nullptr` if no such object exists.
   */
        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int        index) const
            = 0;

        virtual std::shared_ptr<MySolution> getSolutionByIndex(const int index) const
        {
            throw std::runtime_error("[getSolutionByIndex] You should not reach here.");
            return std::shared_ptr<MySolution>();
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(Hardware const&  hardware, const int index) const
        {
            throw std::runtime_error("[getSolutionByIndex] You should not reach here.");
            return std::shared_ptr<MySolution>();
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double* fitness = nullptr) const
            = 0;

        virtual std::shared_ptr<MySolution> findBestSolution(std::vector<MyProblem> const& problems,
                                                             Hardware const&               hardware,
                                                             double* fitness = nullptr) const
        {
            return std::shared_ptr<MySolution>();
        }

        /**
   * Returns all `Solution` objects that are capable of correctly solving this
   * `problem` on this `hardware`.
   *
   * May return an empty set if no such object exists.
   */
        virtual SolutionSet<MySolution> findAllSolutions(MyProblem const&          problem,
                                                         Hardware const&           hardware,
                                                         SolutionLibrarySearchType searchType
                                                         = SolutionLibrarySearchType::DEFAULT) const
            = 0;

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const
            = 0;

        virtual std::string type() const        = 0;
        virtual std::string description() const = 0;

        /**
   * Returns the multiple `Solution` object that best solves this
   * particular `problem` on this particular piece of `hardware`.
   *
   * May return `nullptr` if no such object exists.
   */
        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int              numSolutions) const
        {
            return SolutionVector<MySolution>();
        }

        virtual bool lastFindTopAlreadyRetAll() const
        {
            return true;
        }

        virtual SolutionVector<MySolution>
            findTopSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        int                           numSolutions) const
        {
            return SolutionVector<MySolution>();
        }

        // Get the library filename for debug output (if available)
        virtual std::string getLibraryFileName() const
        {
            return "";
        }

        /// Whether a tilewright model ranks this library's kernels, which only
        /// a Prediction library loaded with TENSILE_USE_TILEWRIGHT can have.
        virtual bool hasTilewrightModel() const
        {
            return false;
        }

        /// What findTopSolutions can return on `route` (SelectionRoute bits),
        /// whatever the problem: Some unless the library can tell more. A
        /// library whose rows are not loaded yet answers Some.
        virtual RouteYield routeYield(unsigned route) const
        {
            return RouteYield::Some;
        }

        /// Whether findTopSolutions(problem, ...) on `route` returns no
        /// solution, asking a TilewrightRowOrder at most about Equality rows:
        /// then an ExactLogicLibrary row that leads here need not be searched
        /// (ExactLogicLibrary::findTopRows). False unless the library can tell
        /// cheaply. It is asked before the predicate of that row has accepted
        /// the problem, so it may read only what is defined for every problem
        /// that can reach this library.
        virtual bool returnsNothing(MyProblem const& problem, unsigned route) const
        {
            return false;
        }
    };

} // namespace TensileLite

