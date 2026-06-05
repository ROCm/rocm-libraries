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

#include <atomic>
#include <optional>
#include <set>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/PredicateDebugger.hpp>
#include <Tensile/Predicates.hpp>
#include <Tensile/SolutionLibrary.hpp>

#include <Tensile/Macros.hpp>

TENSILE_HIDDEN_BEGIN

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
        std::vector<Row>          rows;
        mutable std::atomic<bool> lastFindTopRetAll = false;

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
            std::shared_ptr<MySolution> fallbackRv;
            const bool                  streamK = Debug::Instance().useExperimentalSelection() == 2;

            // Return exact match immediately; otherwise keep first successful fallback.
            for(auto const& row : rows)
            {
                if(row.first.value->type() == "ExperimentalStreamK" && !streamK)
                    continue;

                if(!row.first(problem, hardware))
                    continue;

                if(!row.first.isFallbackMatch(hardware))
                {
                    auto rv = row.second->findBestSolution(problem, hardware, fitness);

                    if(rv
                       && dynamic_cast<Predicates::Contraction::EqualityMatching*>(
                           row.first.value.get()))
                        rv->tag = MySolution::MatchingTag::Equal;

                    if(rv)
                    {
                        if(Debug::Instance().printDeviceSelection())
                        {
                            std::cout << "  Solution found (exact): " << rv->name()
                                      << " [MatchingTag: " << rv->matchingTag() << "]" << std::endl;
                        }
                        return rv;
                    }
                }
                else if(!fallbackRv)
                {
                    fallbackRv = row.second->findBestSolution(problem, hardware, fitness);

                    if(fallbackRv
                       && dynamic_cast<Predicates::Contraction::EqualityMatching*>(
                           row.first.value.get()))
                        fallbackRv->tag = MySolution::MatchingTag::Equal;
                }
            }

            if(fallbackRv && Debug::Instance().printDeviceSelection())
            {
                std::cout << "  Solution found (fallback): " << fallbackRv->name()
                          << " [MatchingTag: " << fallbackRv->matchingTag() << "]" << std::endl;
            }

            return fallbackRv;
        }

    public:

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            SolutionSet<MySolution> rv;
            const bool              debug       = Debug::Instance().printPropertyEvaluation();
            const bool              streamK     = Debug::Instance().useExperimentalSelection() == 2;
            const auto&             excludedLib = Debug::Instance().excludedLibFromGetAll();

            auto amdGPU             = static_cast<AMDGPU const*>(&hardware);
            bool isStandardCUDevice = amdGPU->isStandardCU();

            for(auto const& row : rows)
            {
                // we want to exclude this lib from getAll. If the Set is not empty,
                // it means we want to skip searched GridBased, Prediction.
                // But if this is a non-standard-CU GPU, we still need to search all for CU-Fallback solutions.
                // In this case, we don't skip the excludedLib.
                if(!excludedLib.empty() && isStandardCUDevice)
                {
                    if(excludedLib.count(row.first.value->type()))
                        continue;
                }

                if(row.first.value->type() == "ExperimentalStreamK" && !streamK)
                    continue;

                if(row.first.value->type() == "AMDGPU" && !row.first(problem, hardware))
                    continue;

                auto rowSolutions = row.second->findAllSolutions(problem, hardware, searchType);

                // hipblaslt_ext::matmulIsTuned() -> rocblaslt_matmul_is_tuned() needs this Equal test
                if(dynamic_cast<Predicates::Contraction::EqualityMatching*>(row.first.value.get()))
                {
                    for(auto& sol : rowSolutions)
                        sol->tag = MySolution::MatchingTag::Equal;
                }
                // except for Equal, we test others only when debug mode.
                else if(debug)
                {
                    if(dynamic_cast<Predicates::Contraction::GridBasedMatching*>(
                           row.first.value.get()))
                    {
                        for(auto& sol : rowSolutions)
                            sol->tag = MySolution::MatchingTag::GridBased;
                    }
                    else if(dynamic_cast<Predicates::Contraction::RangeMatching*>(
                                row.first.value.get()))
                    {
                        for(auto& sol : rowSolutions)
                            sol->tag = MySolution::MatchingTag::Range;
                    }
                    else if(dynamic_cast<Predicates::Contraction::FreeSizeMatching*>(
                                row.first.value.get()))
                    {
                        for(auto& sol : rowSolutions)
                            sol->tag = MySolution::MatchingTag::FreeSize;
                    }
                    else if(dynamic_cast<Predicates::Contraction::PredictionMatching*>(
                                row.first.value.get()))
                    {
                        for(auto& sol : rowSolutions)
                            sol->tag = MySolution::MatchingTag::Prediction;
                    }
                    // TODO- Experimental?
                }

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
            const bool                 debug   = Debug::Instance().printPropertyEvaluation();
            const bool                 streamK = Debug::Instance().useExperimentalSelection() == 2;
            const bool                 predictionLib = Debug::Instance().usePredictionLibrary();

            // false in case of early return;
            lastFindTopRetAll = false;

            for(auto const& row : rows)
            {
                if(row.first.value->type() == "ExperimentalStreamK" && !streamK)
                    continue;

                if(predictionLib
                   && ((row.first.value->type() == "EqualityMatching")
                       || (row.first.value->type() == "RangeMatching")))
                    continue;

                if(row.first(problem, hardware))
                {
                    solutions
                        = row.second->findTopSolutions(problem, hardware, numSolutions - rv.size());

                    // hipblaslt_ext::matmulIsTuned() -> rocblaslt_matmul_is_tuned() needs this Equal test
                    if(dynamic_cast<Predicates::Contraction::EqualityMatching*>(
                           row.first.value.get()))
                    {
                        for(auto& sol : solutions)
                            sol->tag = MySolution::MatchingTag::Equal;
                    }
                    // except for Equal, we test others only when debug mode.
                    else if(debug)
                    {
                        if(dynamic_cast<Predicates::Contraction::GridBasedMatching*>(
                               row.first.value.get()))
                        {
                            for(auto& sol : solutions)
                                sol->tag = MySolution::MatchingTag::GridBased;
                        }
                        else if(dynamic_cast<Predicates::Contraction::RangeMatching*>(
                                    row.first.value.get()))
                        {
                            for(auto& sol : solutions)
                                sol->tag = MySolution::MatchingTag::Range;
                        }
                        else if(dynamic_cast<Predicates::Contraction::FreeSizeMatching*>(
                                    row.first.value.get()))
                        {
                            for(auto& sol : solutions)
                                sol->tag = MySolution::MatchingTag::FreeSize;
                        }
                        else if(dynamic_cast<Predicates::Contraction::PredictionMatching*>(
                                    row.first.value.get()))
                        {
                            for(auto& sol : solutions)
                                sol->tag = MySolution::MatchingTag::Prediction;
                        }
                        // TODO- Experimental
                    }

                    rv.insert(std::end(rv), std::begin(solutions), std::end(solutions));
                    if(rv.size() == numSolutions)
                        return rv;
                }
            }

            // can't reach the requested number, means findTop already done its best
            lastFindTopRetAll = (rv.size() < numSolutions);
            return rv;
        }

        virtual bool lastFindTopAlreadyRetAll() const override
        {
            return lastFindTopRetAll;
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

        // The chip IDs this predicate targets, if any. Set by callers that
        // know the chip IDs (e.g. makeHwPred, deserialization). When empty
        // the predicate has no chip-ID constraint and every match is exact.
        std::set<int> targetPciChipIds;

        HardwarePredicate() = default;
        HardwarePredicate(std::shared_ptr<Predicates::Predicate<Hardware>> init,
                          std::optional<int> chipId = std::nullopt)
            : value(init)
        {
            if(chipId)
                targetPciChipIds.insert(chipId.value());
        }

        HardwarePredicate(std::shared_ptr<Predicates::Predicate<Hardware>> init,
                          std::set<int>                                 chipIds)
            : value(init)
            , targetPciChipIds(chipIds)
        {
        }

        template <typename Any>
        bool operator()(Any const& problem, Hardware const& hardware) const
        {
            bool debug = Debug::Instance().printDeviceSelection();
            bool rv    = (*value)(hardware);

            if(debug)
            {
                PredicateDebugger::printHeader(std::cout, "ExactLogic: Hardware");
                value->debugEval(hardware, std::cout);
                PredicateDebugger::printFooter(std::cout, rv);
            }

            return rv;
        }

        // A match is a fallback when the predicate targets a specific chip ID
        // and the GPU's chip ID is different (operator() already confirmed
        // compatibility via ChipIdRegistry::canUseSolution).
        bool isFallbackMatch(Hardware const& hardware) const
        {
            if(targetPciChipIds.empty())
                return false;

            auto gpuChipId = hardware.pciChipId();
            if(!gpuChipId)
                return false;

            return targetPciChipIds.count(gpuChipId.value()) == 0;
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
            bool rv    = (*value)(problem);

            if(debug)
            {
                PredicateDebugger::printHeader(std::cout, "ExactLogic: Problem");
                value->debugEval(problem, std::cout);
                PredicateDebugger::printFooter(std::cout, rv);
            }

            return rv;
        }

        // Problem predicates never involve hardware fallback.
        bool isFallbackMatch(Hardware const&) const
        {
            return false;
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

TENSILE_HIDDEN_END
