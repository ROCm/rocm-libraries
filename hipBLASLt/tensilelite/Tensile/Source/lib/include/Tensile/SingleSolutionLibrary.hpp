/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
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

            if(solution)
            {
                if(debug)
                {
                    std::cout << "hardwarePredicate:" << std::endl;
                    solution->hardwarePredicate->debugEval(hardware, std::cout);
                    std::cout << std::endl;
                    std::cout << "problemPredicate:" << std::endl;
                    solution->problemPredicate->debugEval(problem, std::cout);
                    std::cout << std::endl;
                }

                if((*solution->hardwarePredicate)(hardware)
                   && (*solution->problemPredicate)(problem))
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

            if(solution)
            {
                if(debug)
                {
                    solution->hardwarePredicate->debugEval(hardware, std::cout);
                    for(int idx = 0; idx < problems.size(); idx++)
                    {
                        auto problem = problems[idx];
                        solution->problemPredicate->debugEval(problem, std::cout);
                    }
                }

                if(!(*solution->hardwarePredicate)(hardware))
                    return std::shared_ptr<MySolution>();

                size_t ws = (*solution).requiredWorkspaceSizeGroupedGemm(problems, hardware);

                for(int idx = 0; idx < problems.size(); idx++)
                {
                    auto problem = problems[idx];
                    problem.setWorkspaceSizeGroupedGemm(ws);
                    problem.setGroupedGemmCount(problems.size());
                    if(!(*solution->problemPredicate)(problem))
                        return std::shared_ptr<MySolution>();
                }

                if(solution->requiredHostWorkspaceSizePerProblem == static_cast<size_t>(-1))
                {
                    solution->requiredHostWorkspaceSizePerProblem
                        = solution->requiredHostSizeGroupedGemmSingle(problems[0],hardware);
                }

                return solution;
            }
            else if(debug)
            {
                std::cout << " (empty library)";
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

            bool useSolution = false;
            if(solution)
            {
                if(debug)
                {
                    solution->hardwarePredicate->debugEval(hardware, std::cout);
                    if(searchType == SolutionLibrarySearchType::DEFAULT)
                        solution->problemPredicate->debugEval(problem, std::cout);
                }

                if((*solution->hardwarePredicate)(hardware)
                   && softwarePredicate(searchType, (*solution), problem))
                    useSolution = true;
            }
            else if(debug)
            {
                std::cout << " (empty library)";
            }

            if(debug)
            {
                if(useSolution)
                    std::cout << " (match)";
                else
                    std::cout << " (no match)";
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

            bool useSolution = false;
            if(solution)
            {
                if((*solution->hardwarePredicate)(hardware))
                    useSolution = true;

                if(searchType == SolutionLibrarySearchType::DEFAULT)
                {
                    size_t ws = (*solution).requiredWorkspaceSizeGroupedGemm(problems, hardware);

                    for(int idx = 0; idx < problems.size(); idx++)
                    {
                        auto problem = problems[idx];
                        problem.setWorkspaceSizeGroupedGemm(ws);
                        problem.setGroupedGemmCount(problems.size());
                        if(!(*solution->problemPredicate)(problem))
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
                        for(int idx = 0; idx < problems.size(); idx++)
                        {
                            auto problem = problems[idx];
                            solution->problemPredicate->debugEval(problem, std::cout);
                        }
                }
            }
            else if(debug)
            {
                std::cout << " (empty library)";
            }

            if(debug)
            {
                if(useSolution)
                    std::cout << " (match)";
                else
                    std::cout << " (no match)";
            }

            if(useSolution)
            {
                if(solution->requiredHostWorkspaceSizePerProblem == static_cast<size_t>(-1))
                {
                    solution->requiredHostWorkspaceSizePerProblem
                        = solution->requiredHostSizeGroupedGemmSingle(problems[0],hardware);
                }
                return SolutionSet<MySolution>({solution});
            }

            return SolutionSet<MySolution>();
        }
    };
} // namespace TensileLite
