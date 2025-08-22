/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/Predicates.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <type_traits>

namespace TensileLite
{
    /**
 * \addtogroup SolutionLibrary
 * @{
 */
    template <typename MyProblem, typename MySolution>
    using LibraryEntry = std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>;
    template <typename MyProblem, typename MySolution, typename MyPredicate>
    using LibraryRow = std::pair<MyPredicate, LibraryEntry<MyProblem, MySolution>>;

    /**
 * Represents a set of sub-libraries, each with associated predicates. It
 * should be placed in order of best to worst solutions. We assume the best
 * solution is the first one where we match the predicates.
 *
 * Examples: Picking solutions written for a particular GPU, solutions that
 * assume that a particular size is a multiple of something, etc.
 */
    template <typename MyProblem, typename MySolution, typename MyPredicate>
    struct ExactLogicLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        using Row = LibraryRow<MyProblem, MySolution, MyPredicate>;
        std::vector<Row> rows;

        ExactLogicLibrary() = default;
        ExactLogicLibrary(std::initializer_list<Row> init)
            : rows(init)
        {
        }

        ExactLogicLibrary(std::vector<Row> const& init)
            : rows(init)
        {
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            std::shared_ptr<MySolution> rv;
            const bool                  streamK = Debug::Instance().useExperimentalSelection() == 2;

            for(auto const& row : rows)
            {
                if(row.first.value->type() == "ExperimentalStreamK" && !streamK)
                    continue;

                if(row.first(problem, hardware))
                {
                    rv = row.second->getSolutionByIndex(problem, hardware, index);
                    if(rv)
                        return rv;
                }
            }

            return rv;
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            std::shared_ptr<MySolution> rv;
            const bool                  streamK = Debug::Instance().useExperimentalSelection() == 2;

            for(auto const& row : rows)
            {
                if(row.first.value->type() == "ExperimentalStreamK" && !streamK)
                    continue;

                if(row.first(problem, hardware))
                {
                    rv = row.second->findBestSolution(problem, hardware, fitness);

                    if(rv
                       && dynamic_cast<Predicates::Contraction::EqualityMatching*>(
                           row.first.value.get()))
                        rv->tag = MySolution::MatchingTag::Equal;

                    if(rv)
                        return rv;
                }
            }

            return rv;
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            SolutionSet<MySolution> rv;
            const bool              streamK = Debug::Instance().useExperimentalSelection() == 2;

            for(auto const& row : rows)
            {
                if(row.first.value->type() == "ExperimentalStreamK" && !streamK)
                    continue;

                if(row.first.value->type() == "AMDGPU" && !row.first(problem, hardware))
                    continue;

                auto rowSolutions = row.second->findAllSolutions(problem, hardware, searchType);
                rv.insert(rowSolutions.begin(), rowSolutions.end());
            }

            return rv;
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            SolutionSet<MySolution> rv;

            for(auto const& row : rows)
            {
                if(row.first.value->type() == "AMDGPU" && !row.first(problems[0], hardware))
                    continue;

                auto rowSolutions
                    = row.second->findAllSolutionsGroupedGemm(problems, hardware, searchType);
                rv.insert(rowSolutions.begin(), rowSolutions.end());
            }

            return rv;
        }

        virtual std::string description() const override
        {
            return concatenate(this->type(), " library (", rows.size(), " rows)");
        }

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            SolutionVector<MySolution> rv, solutions;
            const bool                 streamK = Debug::Instance().useExperimentalSelection() == 2;

            for(auto const& row : rows)
            {
                if(row.first.value->type() == "ExperimentalStreamK" && !streamK)
                    continue;

                if(row.first(problem, hardware))
                {
                    solutions
                        = row.second->findTopSolutions(problem, hardware, numSolutions - rv.size());

                    if(dynamic_cast<Predicates::Contraction::EqualityMatching*>(
                           row.first.value.get()))
                        for(auto& sol : solutions)
                            sol->tag = MySolution::MatchingTag::Equal;

                    rv.insert(std::end(rv), std::begin(solutions), std::end(solutions));
                    if(rv.size() == numSolutions)
                        return rv;
                }
            }

            return rv;
        }

        virtual SolutionVector<MySolution>
            findTopSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        int                           numSolutions) const override
        {
            SolutionVector<MySolution> rv, solutions;

            for(auto const& row : rows)
            {
                if(row.first(problems[0], hardware))
                {
                    solutions = row.second->findTopSolutionsGroupedGemm(
                        problems, hardware, numSolutions - rv.size());
                    rv.insert(std::end(rv), std::begin(solutions), std::end(solutions));
                    if(rv.size() == numSolutions)
                        return rv;
                }
            }

            return rv;
        }
    };

    struct HardwarePredicate
    {
        std::shared_ptr<Predicates::Predicate<Hardware>> value;

        HardwarePredicate() = default;
        HardwarePredicate(std::shared_ptr<Predicates::Predicate<Hardware>> init)
            : value(init)
        {
        }

        template <typename Any>
        bool operator()(Any const& problem, Hardware const& hardware) const
        {
            bool debug = Debug::Instance().printDeviceSelection();

            if(debug)
            {
                value->debugEval(hardware, std::cout);
                std::cout << std::endl;
            }

            return (*value)(hardware);
        }
    };

    template <typename MyProblem, typename MySolution>
    struct HardwareSelectionLibrary
        : public ExactLogicLibrary<MyProblem, MySolution, HardwarePredicate>
    {
        using Base = ExactLogicLibrary<MyProblem, MySolution, HardwarePredicate>;

        HardwareSelectionLibrary() = default;
        HardwareSelectionLibrary(std::initializer_list<typename Base::Row> init)
            : Base(init)
        {
        }

        HardwareSelectionLibrary(std::vector<typename Base::Row> const& init)
            : Base(init)
        {
        }

        static std::string Type()
        {
            return "Hardware";
        }
        virtual std::string type() const override
        {
            return Type();
        }
    };

    template <typename MyProblem>
    struct ProblemPredicate
    {
        std::shared_ptr<Predicates::Predicate<MyProblem>> value;

        ProblemPredicate() = default;
        ProblemPredicate(std::shared_ptr<Predicates::Predicate<MyProblem>> init)
            : value(init)
        {
        }

        bool operator()(MyProblem const& problem, Hardware const& hardware) const
        {
            bool debug = Debug::Instance().printPredicateEvaluation();

            if(debug)
            {
                value->debugEval(problem, std::cout);
                std::cout << std::endl;
            }

            return (*value)(problem);
        }
    };

    template <typename MyProblem, typename MySolution>
    struct ProblemSelectionLibrary
        : public ExactLogicLibrary<MyProblem, MySolution, ProblemPredicate<MyProblem>>
    {
        using Base = ExactLogicLibrary<MyProblem, MySolution, ProblemPredicate<MyProblem>>;

        ProblemSelectionLibrary() = default;
        ProblemSelectionLibrary(std::initializer_list<typename Base::Row> init)
            : Base(init)
        {
        }

        ProblemSelectionLibrary(std::vector<typename Base::Row> const& init)
            : Base(init)
        {
        }

        static std::string Type()
        {
            return "Problem";
        }
        virtual std::string type() const
        {
            return Type();
        }
    };

    /**
 * @}
 */

} // namespace TensileLite
