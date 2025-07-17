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

#include <set>
#include <vector>

#include <Tensile/PropertyMatching.hpp>

namespace TensileLite
{
    /**
     * \ingroup SolutionLibrary
     *
     * Uses a distance function to select solutions based on benchmarks.
     * Benchmarks are performed to determine the optimal solution at a number of
     * specific sizes. At runtime, we find the benchmarked size that is closest
     * to the size asked for.
     */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct ProblemMatchingLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        using Element = std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>;
        using Table   = Matching::MatchingTable<MyProblem, Element, std::shared_ptr<MySolution>>;
        std::shared_ptr<Table> table;

        static std::string Type()
        {
            return "Matching";
        }
        virtual std::string type() const override
        {
            return Type();
        }
        virtual std::string description() const override
        {
            if(table == nullptr)
                return concatenate(type(), ", table: nullptr");
            else
                return concatenate(type(), ": ", table->description());
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            typename Table::Transform transform
                = [&](Element library) -> std::shared_ptr<MySolution> {
                return library->getSolutionByIndex(problem, hardware, index);
            };
            double                      fitness;
            std::shared_ptr<MySolution> solution;
            std::tie(solution, fitness) = table->findBestMatch(problem, transform);
            return solution;
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            bool useDebugSelection = Debug::Instance().enableDebugSelection();

            typename Table::Transform transform
                = [&](Element library) -> std::shared_ptr<MySolution> {
                return library->findBestSolution(problem, hardware);
            };

            if(useDebugSelection)
            {
                std::shared_ptr<MySolution> evaluationSolution
                    = table->findBestEvaluationSolution(problem, hardware, transform);
                return evaluationSolution;
            }
            else
            {
                double localFitness = std::numeric_limits<double>::max();
                fitness             = (fitness) ? fitness : &localFitness;
                std::shared_ptr<MySolution> solution;
                std::tie(solution, *fitness) = table->findBestMatch(problem, transform);
                return solution;
            }
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            bool debug = Debug::Instance().printPropertyEvaluation();

            SolutionSet<MySolution> rv;

            auto matches = searchType != SolutionLibrarySearchType::DEFAULT
                               ? table->GetAll()
                               : table->matchesInOrder(problem);

            for(auto const& row : matches)
            {
                if(debug)
                    std::cout << row->description() << std::endl;

                auto rowSolutions = row->findAllSolutions(problem, hardware, searchType);
                rv.insert(rowSolutions.begin(), rowSolutions.end());

                if(debug)
                    std::cout << std::endl;
            }

            return rv;
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            bool debug = Debug::Instance().printPropertyEvaluation();

            SolutionSet<MySolution> rv;

            auto matches = searchType != SolutionLibrarySearchType::DEFAULT
                               ? table->GetAll()
                               : table->matchesInOrder(problems[0]);

            for(auto const& row : matches)
            {
                if(debug)
                    std::cout << row->description() << std::endl;

                auto rowSolutions
                    = row->findAllSolutionsGroupedGemm(problems, hardware, searchType);
                rv.insert(rowSolutions.begin(), rowSolutions.end());

                if(debug)
                    std::cout << std::endl;
            }

            return rv;
        }

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            typename Table::Transform transform
                = [&](Element library) -> std::shared_ptr<MySolution> {
                return library->findBestSolution(problem, hardware);
            };
            SolutionVector<MySolution> solutions
                = table->findTopMatch(problem, transform, numSolutions);

            if(Debug::Instance().printLibraryLogicIndex())
            {
                if(!solutions.empty()) {
                    std::cout << "Library logic index of top solutions: ";
                    for (auto &rv : solutions)
                        std::cout << rv->libraryLogicIndex << ", ";
                    std::cout << std::endl;
                }
                else
                    std::cout << "No solution found" << std::endl;
            }
            return solutions;
        }

        virtual SolutionVector<MySolution>
            findTopSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        int                           numSolutions) const override
        {
            typename Table::Transform transform
                = [&](Element library) -> std::shared_ptr<MySolution> {
                return library->findBestSolution(problems, hardware);
            };
            SolutionVector<MySolution> solutions
                = table->findTopMatch(problems[0], transform, numSolutions);
            return solutions;
        }
    };
} // namespace TensileLite
