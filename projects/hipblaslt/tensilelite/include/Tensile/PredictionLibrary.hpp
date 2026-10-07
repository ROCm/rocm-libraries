/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include <Tensile/Debug.hpp>
#include <Tensile/PredicateDebugger.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/UtilsOrigami.hpp>

#ifdef TENSILELITE_HAS_TILEWRIGHT
#include <Tensile/TilewrightRanker.hpp>
#endif

#include <tensilelitehost/export.h>

namespace TensileLite
{
    // Declared unconditionally so ProblemPredictionLibrary has the same layout in
    // translation units built with and without TENSILELITE_HAS_TILEWRIGHT.
    class TilewrightRanker;
    template <typename MySolution>
    class TilewrightBank;

    /**
     * \ingroup SolutionLibrary
     *
     * Uses a distance function to select solutions based on benchmarks.
     * Benchmarks are performed to determine the optimal solution at a number of
     * specific sizes. At runtime, we find the benchmarked size that is closest
     * to the size asked for.
     */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct ProblemPredictionLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        std::vector<std::pair<int, std::shared_ptr<MySolution>>> solution_list;
        std::vector<origami::config_t>                           origami_config_list;
        // Pool position i of the ranker is solution_list[i].
        std::shared_ptr<const TilewrightRanker> tilewright_ranker;
        // Set when tilewright also ranks the problem type's Equality kernels; the
        // bank pools follow solution_list with the solutions of these libraries
        // (TilewrightBank). Without sources the bank is empty.
        std::shared_ptr<const TilewrightBank<MySolution>> tilewright_bank;
        std::function<std::vector<std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>>()>
            tilewright_bank_sources;

        mutable std::atomic<bool> lastFindTopRetAll = false;

        static std::string Type()
        {
            return "Prediction";
        }
        virtual std::string type() const override
        {
            return Type();
        }
        virtual std::string description() const override
        {
            if(solution_list.empty())
                return concatenate(type(), ", solution_list: empty");
            return concatenate(type(), solution_list.size());
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            auto indexMatch =
                std::find_if(solution_list.begin(), solution_list.end(),
                             [&index](auto& s){ return s.first == index; });
            if(indexMatch != solution_list.end())
                return indexMatch->second;
            return nullptr;
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            auto                        topSolutions = findTopSolutions(problem, hardware, 1);
            std::shared_ptr<MySolution> solution;
            if(!topSolutions.empty())
            {
                solution = topSolutions[0];
            }
            return solution;
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            bool                    debug = Debug::Instance().printPropertyEvaluation();
            SolutionSet<MySolution> rv;
            if(searchType == SolutionLibrarySearchType::DEFAULT)
                return rv;

            for(auto const& row : this->solution_list)
            {
                if(debug)
                    std::cout << row.second->description() << std::endl;
                rv.insert(row.second);
            }

            return rv;
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            bool                    debug = Debug::Instance().printPropertyEvaluation();
            SolutionSet<MySolution> rv;
            if(searchType == SolutionLibrarySearchType::DEFAULT)
                return rv;

            for(auto const& row : this->solution_list)
            {
                if(debug)
                    std::cout << row.second->description() << std::endl;
                rv.insert(row.second);
            }

            return rv;
        }

#ifdef TENSILELITE_HAS_TILEWRIGHT
        // Looks up the stem of `logicFile` in the tilewright_index of its directory.
        void loadTilewright(std::string const& logicFile)
        {
            auto model        = TilewrightRanker::loadModel(logicFile);
            tilewright_ranker = TilewrightRanker::load(model, logicFile, solution_list);
            if(tilewright_ranker && TilewrightRanker::ranksBank(*model))
                tilewright_bank = std::make_shared<const TilewrightBank<MySolution>>(model);
        }
#endif

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            SolutionVector<MySolution> rv;
            if(numSolutions == 0)
            {
                lastFindTopRetAll = false;
                return rv;
            }
            hip::HipAMDGPU const* pAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);

            const bool debug = Debug::Instance().printPropertyEvaluation();

            auto considerSolution = [&](std::shared_ptr<MySolution> const& solution) {
                Task task(hardware, problem, *solution);
                const bool hwMatch = (*(solution->hardwarePredicate))(hardware);
                // With uniform summation order off, the filter stays
                // hardwarePredicate && problemPredicate. The extra conjuncts in
                // softwarePredicate() (taskPredicate, StreamK dynamic queue) are
                // real filters, so they stay behind the check.
                const bool swMatch = problem.getParams().uniformSummationOrder()
                                         ? softwarePredicate(SolutionLibrarySearchType::DEFAULT,
                                                             task,
                                                             hardware,
                                                             *solution,
                                                             problem)
                                         : (*(solution->problemPredicate))(problem);
                const bool predicateMatch = hwMatch && swMatch;

                if(debug)
                {
                    PredicateDebugger::printHeader(
                        std::cout, "Prediction: " + solution->name());
                    solution->hardwarePredicate->debugEval(hardware, std::cout);
                    solution->problemPredicate->debugEval(problem, std::cout);
                    solution->taskPredicate->debugEval(task, std::cout);
                    PredicateDebugger::printFooter(std::cout, predicateMatch);
                }

                if(predicateMatch)
                {
                    rv.emplace_back(solution);
                }
            };

            if(pAMDGPU && pAMDGPU->analyticalHardware)
            {
#ifdef TENSILELITE_HAS_TILEWRIGHT
                TilewrightRanker const*                         ranker = tilewright_ranker.get();
                std::vector<std::shared_ptr<MySolution>> const* bank   = nullptr;
                if(ranker && tilewright_bank)
                {
                    auto const& pool = tilewright_bank->pool(
                        *pAMDGPU, [&]() { return makeBankPool(problem, hardware); });
                    if(pool.ranker)
                    {
                        ranker = pool.ranker.get();
                        bank   = &pool.bank;
                    }
                }

                // Pool positions tilewright offered, in order; origami ranks the rest.
                std::vector<size_t> ranked;
                size_t              offeredCount = 0;
                if(ranker)
                {
                    ranked = ranker->rank(problem, *(pAMDGPU->analyticalHardware), numSolutions);
                    for(; offeredCount < ranked.size() && rv.size() != numSolutions; offeredCount++)
                    {
                        // In a bank pool, a kernel can be the exact match an
                        // Equality row already returned.
                        size_t const position = ranked[offeredCount];
                        if(position < solution_list.size())
                        {
                            if(!bank
                               || !SelectedSolutionsScope::contains(solution_list[position].first))
                                considerSolution(solution_list[position].second);
                        }
                        else if(bank && position - solution_list.size() < bank->size())
                        {
                            auto const& solution = (*bank)[position - solution_list.size()];
                            if(!SelectedSolutionsScope::contains(solution->index))
                                considerSolution(solution);
                        }
                    }
                }
#endif

                if(rv.size() != numSolutions)
                {
                    size_t m     = 1;
                    size_t n     = 1;
                    size_t k     = 1;
                    size_t batch = 1;
                    for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
                    {
                        m *= problem.freeSizeA(i);
                    }
                    for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
                    {
                        n *= problem.freeSizeB(i);
                    }
                    for(size_t i = 0; i < problem.boundIndices().size(); ++i)
                    {
                        k *= problem.boundSize(i);
                    }
                    for(size_t i = 0; i < problem.batchIndices().size(); ++i)
                    {
                        batch *= problem.batchSize(i);
                    }

                    auto miDataType = datatypeToAnalyticalDatatype(problem.computeInputTypeA());

                    if(problem.f32XdlMathOp()
                       == rocisa::DataType::XFloat32) // Check F32 compute type
                        miDataType = origami::data_type_t::XFloat32;
                    origami::problem_t origami_problem = {
                        .size  = {m, n, k},
                        .batch = batch,
                        // CU budget hint; 0 = use all CUs.
                        .num_cus = static_cast<size_t>(problem.getParams().smCountTarget()),
                        .a_transpose
                        = problem.transA() ? origami::transpose_t::T : origami::transpose_t::N,
                        .b_transpose
                        = problem.transB() ? origami::transpose_t::T : origami::transpose_t::N,
                        .a_dtype         = datatypeToAnalyticalDatatype(problem.a().dataType()),
                        .b_dtype         = datatypeToAnalyticalDatatype(problem.b().dataType()),
                        .c_dtype         = datatypeToAnalyticalDatatype(problem.c().dataType()),
                        .d_dtype         = datatypeToAnalyticalDatatype(problem.d().dataType()),
                        .mi_dtype        = miDataType,
                        .a_mx_block_size = 0, // MX Data types come from rocroller
                        .b_mx_block_size = 0, // MX Data types come from rocroller
                    };

#ifdef TENSILELITE_HAS_TILEWRIGHT
                    std::vector<bool> offered;
                    if(ranker)
                    {
                        offered.assign(solution_list.size(), false);
                        for(size_t i = 0; i < offeredCount; i++)
                        {
                            if(ranked[i] < solution_list.size())
                                offered[ranked[i]] = true;
                        }
                    }
#endif

                    auto prediction_result = origami::rank_configs(
                        origami_problem, *(pAMDGPU->analyticalHardware), origami_config_list);

                    for(const auto& r : prediction_result)
                    {
                        if(r.config.index >= solution_list.size())
                        {
                            continue;
                        }
#ifdef TENSILELITE_HAS_TILEWRIGHT
                        if(!offered.empty() && offered[r.config.index])
                        {
                            continue;
                        }
#endif
                        considerSolution(solution_list[r.config.index].second);
                        if(rv.size() == numSolutions)
                        {
                            break;
                        }
                    }
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
            SolutionVector<MySolution> solutions;
            return solutions;
        }

#ifdef TENSILELITE_HAS_TILEWRIGHT
    private:
        // The bank pool for the device of `hardware`. Failing to build one
        // leaves tilewright ranking solution_list alone on that device.
        typename TilewrightBank<MySolution>::Pool makeBankPool(MyProblem const& problem,
                                                               Hardware const&  hardware) const
        {
            typename TilewrightBank<MySolution>::Pool pool;
            try
            {
                std::map<int, std::shared_ptr<MySolution>> bank;
                if(tilewright_bank_sources)
                {
                    for(auto const& source : tilewright_bank_sources())
                    {
                        for(auto const& solution : source->findAllSolutions(
                                problem, hardware, SolutionLibrarySearchType::HARDWARE_ONLY))
                            bank.emplace(solution->index, solution);
                    }
                }
                for(auto const& entry : solution_list)
                    bank.erase(entry.first);

                if(Debug::Instance().printDataInit())
                    std::cout << "tilewright bank: " << bank.size() << " kernels after "
                              << solution_list.size() << " Prediction kernels" << std::endl;
                if(bank.empty())
                    return pool;

                pool.bank.reserve(bank.size());
                for(auto& entry : bank)
                    pool.bank.push_back(std::move(entry.second));
                size_t const own = solution_list.size();
                pool.ranker
                    = TilewrightRanker::make(tilewright_bank->model(),
                                             own + pool.bank.size(),
                                             [&](size_t i) -> SizeMapping const& {
                                                 return i < own
                                                            ? solution_list[i].second->sizeMapping
                                                            : pool.bank[i - own]->sizeMapping;
                                             });
            }
            catch(std::exception const& e)
            {
                std::cerr << "hipBLASLt Warning: the tilewright bank could not be ranked ("
                          << e.what() << "); ranking the Prediction kernels alone.\n";
                pool = {};
            }
            return pool;
        }
#endif
    };
} // namespace TensileLite
