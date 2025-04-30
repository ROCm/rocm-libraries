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

#include <Tensile/Debug.hpp>
#include <Tensile/DecisionTree.hpp>
#include <Tensile/ProblemKey.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Utils.hpp>

namespace TensileLite
{
    /**
     * \ingroup SolutionLibrary
     *
     * Uses a set of decision trees to select a solution. Each decision tree manages
     * a single solution and decides if said solution will perform well for the size
     * asked for.
     */

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct DecisionTreeLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        using Element = std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>;
        using Forest  = DecisionTree::Forest<MyProblem, Element, std::shared_ptr<MySolution>>;
        using MySolutionVector = SolutionVector<MySolution>;

        std::shared_ptr<Forest> forest;

        static std::string Type()
        {
            return "DecisionTree";
        }
        virtual std::string type() const override
        {
            return Type();
        }
        virtual std::string description() const override
        {
            if(forest == nullptr)
                return concatenate(type(), ", forest: nullptr");
            else
                return concatenate(type(), ": ", forest->description());
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            const bool experimental = Debug::Instance().useExperimentalSelection();
            if(! experimental)
            {
                // If the experimental library mode is not on treat it like it asserted out
                return nullptr;
            }

            typename Forest::Transform transform
                = [&](Element library) -> std::shared_ptr<MySolution> {
                return library->getSolutionByIndex(problem, hardware, index);
            };
            return forest->findBestMatch(problem, transform);
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            typename Forest::Transform transform
                = [&](Element library) -> std::shared_ptr<MySolution> {
                return library->findBestSolution(problem, hardware);
            };
            return forest->findBestMatch(problem, transform);
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            if(searchType != SolutionLibrarySearchType::DEFAULT)
            {
                // if the solution library search is not default then return an empty
                // set of solutions.
                SolutionSet<MySolution> rv;
                return rv;
	        }

            const bool experimental = Debug::Instance().useExperimentalSelection();
            if(! experimental)
            {
                // Skip the search for solutions if the environment variable
                // that enables the experimental method is not set
                SolutionSet<MySolution> rv;
                return rv;
            }

            typename Forest::Transform transform
                = [&](Element library) -> std::shared_ptr<MySolution> {
                return library->findBestSolution(problem, hardware);
            };
            return forest->matchesInOrder(problem, transform);
        }
        
        virtual MySolutionVector findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            typename Forest::Transform transform
                = [&](Element library) -> std::shared_ptr<MySolution> {
                return library->findBestSolution(problem, hardware);
            };

            return forest->topMatches(problem, transform, numSolutions);
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            if(searchType != SolutionLibrarySearchType::DEFAULT)
            {
                // if the solution library search is notSolutionSet default then return an empty
                // set of solutions
                SolutionSet<MySolution> rv;
                return rv;
            }

            const bool experimental = Debug::Instance().useExperimentalSelection();
            if(! experimental)
            {
                // Skip the search for solutions if the environment variable
                // that enables the experimental method is not set
                SolutionSet<MySolution> rv;
                return rv;
            }

            typename Forest::Transform transform
                = [&](Element library) -> std::shared_ptr<MySolution> {
                return library->findBestSolution(problems, hardware);
            };
            return forest->matchesInOrder(problems[0], transform);
        }
    };

} // namespace TensileLite
