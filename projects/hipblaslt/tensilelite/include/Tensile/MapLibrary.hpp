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

#include <Tensile/SolutionLibrary.hpp>

#include <Tensile/PropertyMatching.hpp>

namespace TensileLite
{
    /**
 * \ingroup SolutionLibrary
 */
    template <typename MyProblem, typename MySolution, typename MyKey>
    using LibraryMap = std::unordered_map<MyKey, LibraryEntry<MyProblem, MySolution>>;

    /**
 * \ingroup SolutionLibrary
 *
 * Represents a set of solutions where each problem can map to only one
 * sub-library, as opposed to an `ExactLogicLibrary` where fall-through can
 * occur.
 *
 * Example: Problem type: Number of dimensions, transposes, etc.
 */
    template <typename MyProblem, typename MySolution, typename Key = std::string>
    struct ProblemMapLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        static std::string Type()
        {
            return "ProblemMap";
        }
        virtual std::string type() const override
        {
            return Type();
        }
        virtual std::string description() const override
        {
            if(property == nullptr)
            {
                return concatenate(type(), " (property: nullptr, ", map.size(), " rows)");
            }
            else
            {
                return concatenate(
                    type(), " (property: ", property->toString(), ", ", map.size(), " rows)");
            }
        }

        LibraryEntry<MyProblem, MySolution> lookup(MyProblem const& problem,
                                                   Hardware const&  hardware) const
        {
            auto key  = (*property)(problem);
            auto iter = map.find(key);

            bool debug = Debug::Instance().printPropertyEvaluation();
            if(debug)
            {
                std::cout << type() << " Searching for " << key;

                if(iter == map.end())
                {
                    std::cout << " (not found).  Available keys:" << std::endl;
                    for(auto const& pair : map)
                    {
                        std::cout << "  " << pair.first << std::endl;
                    }
                }
                else
                {
                    std::cout << " found " << iter->second->description();
                }

                std::cout << std::endl;
            }

            if(iter == map.end())
                return nullptr;

            return iter->second;
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            auto library = lookup(problem, hardware);

            if(library == nullptr)
                return std::shared_ptr<MySolution>();

            return library->getSolutionByIndex(problem, hardware, index);
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            auto library = lookup(problem, hardware);

            if(library == nullptr)
                return std::shared_ptr<MySolution>();

            return library->findBestSolution(problem, hardware, fitness);
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            if(searchType == SolutionLibrarySearchType::HARDWARE_ONLY)
            {
                auto result = SolutionSet<MySolution>();
                for(auto& pair : map)
                {
                    auto library = pair.second;
                    if(library == nullptr)
                        continue;
                    auto solutions = library->findAllSolutions(problem, hardware, searchType);
                    result.insert(solutions.begin(), solutions.end());
                }
                return result;
            }
            auto library = lookup(problem, hardware);

            if(library == nullptr)
                return SolutionSet<MySolution>();

            return library->findAllSolutions(problem, hardware, searchType);
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            if(searchType == SolutionLibrarySearchType::HARDWARE_ONLY)
            {
                auto result = SolutionSet<MySolution>();
                for(auto& pair : map)
                {
                    auto library = pair.second;
                    if(library == nullptr)
                        continue;
                    auto solutions
                        = library->findAllSolutionsGroupedGemm(problems, hardware, searchType);
                    result.insert(solutions.begin(), solutions.end());
                }
                return result;
            }
            auto library = lookup(problems[0], hardware);

            if(library == nullptr)
                return SolutionSet<MySolution>();

            return library->findAllSolutionsGroupedGemm(problems, hardware, searchType);
        }

        std::shared_ptr<Property<MyProblem, Key>> property;
        LibraryMap<MyProblem, MySolution, Key>    map;

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            auto library = lookup(problem, hardware);

            if(library == nullptr)
                return SolutionVector<MySolution>();

            return library->findTopSolutions(problem, hardware, numSolutions);
        }

        virtual SolutionVector<MySolution>
            findTopSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        int                           numSolutions) const override
        {
            auto library = lookup(problems[0], hardware);

            if(library == nullptr)
                return SolutionVector<MySolution>();

            return library->findTopSolutionsGroupedGemm(problems, hardware, numSolutions);
        }
    };
} // namespace TensileLite
