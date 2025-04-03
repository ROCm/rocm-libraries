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

#include <queue>
#include <set>
#include <vector>

#include <Tensile/Debug.hpp>
#include <Tensile/MLFeatures.hpp>
#include <Tensile/MLPClassification.hpp>
#include <Tensile/ProblemKey.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Utils.hpp>

namespace TensileLite
{
    /**
     * \ingroup SolutionLibrary
     *
     * Uses a small neural network to rank solutions for a given size.
     */

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct MLPClassificationLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        using MLPNet           = MLPClassification::MLPNet;
        using SolutionFeatures = std::vector<std::shared_ptr<MLFeatures::MLFeature<MySolution>>>;
        using ProblemFeatures  = std::vector<std::shared_ptr<MLFeatures::MLFeature<MyProblem>>>;

        std::map<int, std::shared_ptr<MySolution>> solutionmap;
        std::shared_ptr<MLPNet>                   model;
        SolutionFeatures                           solFeatures;
        ProblemFeatures                            probFeatures;

        static std::string Type()
        {
            return "MLPClassification";
        }
        virtual std::string type() const override
        {
            return Type();
        }
        virtual std::string description() const override
        {
            if(model == nullptr)
                return concatenate(type(), ", MLPNet: nullptr");
            else
                return concatenate(type(), ": ", model->description());
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            const bool experimental = Debug::Instance().useExperimentalSelection();
            if(!experimental)
            {
                // If the experimental library mode is not on treat it like it asserted out
                return nullptr;
            }
            // ;
            auto indexMatch = solutionmap.find(index);
            if(indexMatch != solutionmap.end())
                return indexMatch->second;
            return nullptr;
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            SolutionVector<MySolution>  solutions = findTopSolutions(problem, hardware, 1);
            std::shared_ptr<MySolution> solution  = nullptr;
            if(solutions.size() > 0)
                solution = solutions[0];
            return solution;
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            const bool experimental = Debug::Instance().useExperimentalSelection();
            if(!experimental)
            {
                // Skip the search for solutions if the environment variable
                // that enables the experimental method is not set
                SolutionSet<MySolution> rv;
                return rv;
            }
            SolutionSet<MySolution> rv;
            for(auto const& row : solutionmap)
                rv.insert(row.second);

            return rv;
        }

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            std::vector<float> problemkey
                = ProblemKey::keyForProblem<std::vector<float>, MyProblem, float>(
                    problem, this->probFeatures);

            auto logits = model->predict(problemkey);
            assert(logits.size() == solutionmap.size());

            std::vector<std::pair<decltype(logits)::value_type,
                                  std::shared_ptr<MySolution>*>> solution_ranking;
            solution_ranking.reserve(solutionmap.size());
            for(auto& s : solutionmap)
                solution_ranking.emplace_back(logits[s.second->libraryLogicIndex],
                    (std::shared_ptr<MySolution>*)(&s.second));

            SolutionVector<MySolution> rv;
            int numToSort = std::min(numSolutions, int(solution_ranking.size()));
            rv.reserve(numToSort);
            auto it = solution_ranking.begin(), it_end = solution_ranking.end();
            while(it != it_end && numToSort)
            {
                std::partial_sort(it, it + numToSort, it_end, std::greater{});
                for(; it != it + numToSort; it++)
                    if((*((*it->second)->problemPredicate))(problem))
                    {
                        rv.emplace_back(*it->second);
                        numToSort--;
                    }
            }
            return rv;
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            const bool experimental = Debug::Instance().useExperimentalSelection();
            if(!experimental)
            {
                // Skip the search for solutions if the environment variable
                // that enables the experimental method is not set
                SolutionSet<MySolution> rv;
                return rv;
            }

            SolutionSet<MySolution> rv;
            for(auto const& row : solutionmap)
                rv.insert(row.second);

            return rv;
        }
    };

} // namespace TensileLite
