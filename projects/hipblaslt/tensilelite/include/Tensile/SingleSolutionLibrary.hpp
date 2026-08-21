/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/Debug.hpp>
#include <Tensile/PredicateDebugger.hpp>
#include <Tensile/SolutionBlobCache.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    /**
 * \ingroup SolutionLibrary
 *
 * Leaf of the tree. Represents a single `Solution` object. Can eliminate
 * itself from consideration based on restrictions of that particular
 * `Solution`.
 */
    template <typename MyProblem, typename MySolution>
    struct SingleSolutionLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        static std::string Type()
        {
            return "Single";
        }
        std::string type() const override
        {
            return Type();
        }
        std::string description() const override
        {
            std::string rv = type();
            if(solution != nullptr)
            {
                rv += ": ";
                rv += solution->name();
            }
            else if(m_cache)
            {
                // Deliberately does not resolve: this is called from debug
                // printing on selection paths, and forcing a parse just to
                // build a log line would defeat lazy loading.
                rv += concatenate(" (index ", m_index, ", unresolved)");
            }
            else
            {
                rv += " (nullptr)";
            }

            return rv;
        }

        std::shared_ptr<MySolution> solution;

        SingleSolutionLibrary() = default;
        SingleSolutionLibrary(std::shared_ptr<MySolution> s)
            : solution(s)
        {
        }

        /// Lazy form: remembers which solution it stands for and asks the cache
        /// for it when a query actually needs it.
        SingleSolutionLibrary(int index, std::shared_ptr<SolutionBlobCache<MySolution>> cache)
            : m_index(index)
            , m_cache(std::move(cache))
        {
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            // Compare before resolving. A tree walk looking for one index would
            // otherwise parse every leaf it passes.
            if(m_cache)
            {
                if(m_index != index)
                    return std::shared_ptr<MySolution>();
                return resolved();
            }

            if(solution && solution->index == index)
            {
                return solution;
            }
            return std::shared_ptr<MySolution>();
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            bool debug = Debug::Instance().printPredicateEvaluation();

            auto solution = resolved();
            if(solution)
            {
                Task task(hardware, problem, *(solution));
                if(debug)
                {
                    PredicateDebugger::printHeader(std::cout, "Single: " + solution->name());
                    solution->hardwarePredicate->debugEval(hardware, std::cout);
                    solution->problemPredicate->debugEval(problem, std::cout);
                    solution->taskPredicate->debugEval(task, std::cout);
                }

                bool hwMatch = (*solution->hardwarePredicate)(hardware);
                bool swMatch = softwarePredicate(
                    SolutionLibrarySearchType::DEFAULT, task, hardware, (*solution), problem);

                if(debug)
                    PredicateDebugger::printFooter(std::cout, hwMatch && swMatch);

                if(hwMatch && swMatch)
                    return solution;
            }
            else if(debug)
            {
                std::cout << " (empty library)";
            }

            return std::shared_ptr<MySolution>();
        }

        virtual std::shared_ptr<MySolution> findBestSolution(std::vector<MyProblem> const& problems,
                                                             Hardware const&               hardware,
                                                             double*                       fitness
                                                             = nullptr) const override
        {
            bool debug = Debug::Instance().printPredicateEvaluation();

            auto solution = resolved();
            if(solution)
            {
                if(debug)
                {
                    PredicateDebugger::printHeader(std::cout, "Single (Grouped): " + solution->name());
                    solution->hardwarePredicate->debugEval(hardware, std::cout);
                    for(size_t idx = 0; idx < problems.size(); idx++)
                    {
                        auto problem = problems[idx];
                        Task task(hardware, problem, *(solution));
                        solution->problemPredicate->debugEval(problem, std::cout);
                        solution->taskPredicate->debugEval(task, std::cout);
                    }
                }

                if(!(*solution->hardwarePredicate)(hardware))
                {
                    if(debug)
                        PredicateDebugger::printFooter(std::cout, false);
                    return std::shared_ptr<MySolution>();
                }

                size_t ws = (*solution).requiredWorkspaceSizeGroupedGemm(problems, hardware);

                for(size_t idx = 0; idx < problems.size(); idx++)
                {
                    auto problem = problems[idx];
                    Task task(hardware, problem, *(solution));
                    problem.setWorkspaceSizeGroupedGemm(ws);
                    problem.setGroupedGemmCount(problems.size());
                    if(!(*solution->problemPredicate)(problem) || !(*solution->taskPredicate)(task))
                    {
                        if(debug)
                            PredicateDebugger::printFooter(std::cout, false);
                        return std::shared_ptr<MySolution>();
                    }
                }

                if(solution->requiredHostWorkspaceSizePerProblem == static_cast<size_t>(-1))
                {
                    solution->requiredHostWorkspaceSizePerProblem
                        = solution->requiredHostSizeGroupedGemmSingle(problems[0], hardware);
                }

                if(debug)
                    PredicateDebugger::printFooter(std::cout, true);
                return solution;
            }
            else if(debug)
            {
                std::cout << " (empty library)" << std::endl;
            }

            return std::shared_ptr<MySolution>();
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            bool debug = Debug::Instance().printPredicateEvaluation();

            auto solution    = resolved();
            bool useSolution = false;
            if(solution)
            {
                Task task(hardware, problem, (*solution));
                if(debug)
                {
                    PredicateDebugger::printHeader(std::cout, "Single: " + solution->name());
                    solution->hardwarePredicate->debugEval(hardware, std::cout);
                    if(searchType == SolutionLibrarySearchType::DEFAULT)
                    {
                        solution->problemPredicate->debugEval(problem, std::cout);
                        solution->taskPredicate->debugEval(task, std::cout);
                    }
                }

                if((*solution->hardwarePredicate)(hardware)
                   && softwarePredicate(searchType, task, hardware, (*solution), problem))
                    useSolution = true;

                if(debug)
                    PredicateDebugger::printFooter(std::cout, useSolution);
            }
            else if(debug)
            {
                std::cout << " (empty library)" << std::endl;
            }

            if(useSolution)
                return SolutionSet<MySolution>({solution});

            return SolutionSet<MySolution>();
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            bool debug = Debug::Instance().printPredicateEvaluation();

            auto solution    = resolved();
            bool useSolution = false;
            if(solution)
            {
                if(debug)
                    PredicateDebugger::printHeader(std::cout, "Single (Grouped): " + solution->name());

                if((*solution->hardwarePredicate)(hardware))
                    useSolution = true;

                if(searchType == SolutionLibrarySearchType::DEFAULT)
                {
                    size_t ws = (*solution).requiredWorkspaceSizeGroupedGemm(problems, hardware);

                    for(size_t idx = 0; idx < problems.size(); idx++)
                    {
                        auto problem = problems[idx];
                        Task task(hardware, problem, (*solution));
                        problem.setWorkspaceSizeGroupedGemm(ws);
                        problem.setGroupedGemmCount(problems.size());
                        if(!(*solution->problemPredicate)(problem)
                           || !(*solution->taskPredicate)(task))
                            useSolution = false;
                    }
                }
                else if(searchType == SolutionLibrarySearchType::GEMM_TYPE_ONLY)
                {
                    if(!isGemmTypeSame((*solution), problems[0]))
                        useSolution = false;
                }

                if(debug)
                {
                    solution->hardwarePredicate->debugEval(hardware, std::cout);
                    if(searchType == SolutionLibrarySearchType::DEFAULT)
                        for(size_t idx = 0; idx < problems.size(); idx++)
                        {
                            auto problem = problems[idx];
                            Task task(hardware, problem, (*solution));
                            solution->problemPredicate->debugEval(problem, std::cout);
                            solution->taskPredicate->debugEval(task, std::cout);
                        }
                    PredicateDebugger::printFooter(std::cout, useSolution);
                }
            }
            else if(debug)
            {
                std::cout << " (empty library)" << std::endl;
            }

            if(useSolution)
            {
                if(solution->requiredHostWorkspaceSizePerProblem == static_cast<size_t>(-1))
                {
                    solution->requiredHostWorkspaceSizePerProblem
                        = solution->requiredHostSizeGroupedGemmSingle(problems[0], hardware);
                }
                return SolutionSet<MySolution>({solution});
            }

            return SolutionSet<MySolution>();
        }

    private:
        /// The solution this leaf stands for, parsing it from the blob on first
        /// use when the library was loaded in indexed form.
        ///
        /// Deliberately does not memoize into `solution`: several matching table
        /// rows can wrap the same index, and concurrent queries would then race
        /// writing that shared_ptr. The cache is already the memo, and its hit
        /// path is a lookup under a shared lock.
        std::shared_ptr<MySolution> resolved() const
        {
            if(solution || !m_cache)
                return solution;

            return m_cache->get(m_index);
        }

        int                                           m_index = -1;
        std::shared_ptr<SolutionBlobCache<MySolution>> m_cache;
    };
} // namespace TensileLite

