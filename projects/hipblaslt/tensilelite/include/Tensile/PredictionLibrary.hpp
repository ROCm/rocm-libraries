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

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <set>
#include <vector>

#include <Tensile/Debug.hpp>
#include <Tensile/OrigamiBoost.hpp>
#include <Tensile/PredicateDebugger.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/UtilsOrigami.hpp>

#if ORIGAMI_ENABLE_NN
#  include <origami/nn/nn.hpp>
#endif

#include <tensilelitehost/export.h>

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
    struct ProblemPredictionLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        std::vector<std::pair<int, std::shared_ptr<MySolution>>> solution_list;
        std::vector<origami::config_t>                           origami_config_list;

        origami::nn::library_models_t nn_models;

        /// With a Boost model: per solution_list entry, the position of the first entry with
        /// the same solutionName (the re-rank skips repeated kernels by name; set with the model).
        std::vector<std::uint32_t> solution_name_ids;

        mutable std::atomic<bool> lastFindTopRetAll = false;

        /// When set, called with each prediction row as it is deserialized on this thread
        /// (OrigamiBoost builds the library manifest of a model from it).
        static inline thread_local std::function<void(ProblemPredictionLibrary const&)> const*
            deserializeObserver = nullptr;

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

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            SolutionVector<MySolution> rv;
            if(numSolutions == 0)
                return rv;
            size_t                     m     = 1;
            size_t                     n     = 1;
            size_t                     k     = 1;
            size_t                     batch = 1;
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

            hip::HipAMDGPU const* pAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);

            const bool debug = Debug::Instance().printPropertyEvaluation();

            auto matches = [&](std::shared_ptr<MySolution> const& solution) {
                Task task(hardware, problem, *solution);
                const bool hwMatch = (*(solution->hardwarePredicate))(hardware);
                // With uniform summation order off, the filter stays
                // hardwarePredicate && problemPredicate. The extra conjuncts in
                // softwarePredicate() (taskPredicate, StreamK dynamic queue) are
                // real filters, so they stay behind the check.
                const bool swMatch = problem.getParams().uniformSummationOrder() != 0
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

                return predicateMatch;
            };

            // Origami currently models individual output tiles. A spatial
            // StreamK block also launches its padded edge peers; charge for
            // that footprint before truncating the analytical candidate list.
            // Index lookup and measured/exact libraries do not use this path.
            auto chargeStreamKClusters = [&](std::vector<origami::prediction_result_t>& results) {
                for(auto& r : results)
                {
                    if(r.config.index < solution_list.size())
                        r.latency /= solution_list[r.config.index].second
                                         ->streamKClusterUtilization(problem);
                }
                std::stable_sort(results.begin(), results.end(),
                                 [](auto const& a, auto const& b) { return a.latency < b.latency; });
            };

            // Origami Boost's re-rank: it re-orders the charged order and takes the predicate
            // verdicts of the configs it offers.
            OrigamiBoost::RerankSource<MyProblem,
                                       MySolution,
                                       decltype(matches),
                                       decltype(chargeStreamKClusters)>
                rerankSource(problem,
                             hardware,
                             solution_list,
                             solution_name_ids,
                             matches,
                             chargeStreamKClusters);

            auto considerSolution = [&](std::shared_ptr<MySolution> const& solution) {
                const int known = rerankSource.verdict(solution.get());
                if(known < 0 ? matches(solution) : known == 1)
                {
                    rv.emplace_back(solution);
                }
            };

            if(pAMDGPU && pAMDGPU->analyticalHardware)
            {
                auto miDataType = datatypeToAnalyticalDatatype(problem.computeInputTypeA());

                if(problem.f32XdlMathOp() == rocisa::DataType::XFloat32) // Check F32 compute type
                    miDataType = origami::data_type_t::XFloat32;
                origami::problem_t origami_problem = {
                    .size        = {m, n, k},
                    .batch       = batch,
                    // CU budget hint; 0 = use all CUs.
                    .num_cus     = static_cast<size_t>(problem.getParams().smCountTarget()),
                    .a_transpose = problem.transA() ? origami::transpose_t::T : origami::transpose_t::N,
                    .b_transpose = problem.transB() ? origami::transpose_t::T : origami::transpose_t::N,
                    .a_dtype     = datatypeToAnalyticalDatatype(problem.a().dataType()),
                    .b_dtype     = datatypeToAnalyticalDatatype(problem.b().dataType()),
                    .c_dtype     = datatypeToAnalyticalDatatype(problem.c().dataType()),
                    .d_dtype     = datatypeToAnalyticalDatatype(problem.d().dataType()),
                    .mi_dtype    = miDataType,
                    .a_mx_block_size = 0, // MX Data types come from rocroller
                    .b_mx_block_size = 0, // MX Data types come from rocroller
                };

                origami::rank_options_t rank_options;
#if ORIGAMI_ENABLE_NN
                rank_options.library_models = &nn_models;
                const std::size_t nnDepth   = numSolutions < 0
                                                  ? std::numeric_limits<std::size_t>::max()
                                                  : static_cast<std::size_t>(numSolutions);
                rank_options.nn.min_scored  = nnDepth;
#endif
                // nn_models.boost is set only with HIPBLASLT_ORIGAMI_BOOST=1 (deserialization).
                if(nn_models.boost >= 0 && OrigamiBoost::problemClass(problem) != nullptr)
                {
                    rerankSource.activate(origami_problem);
                    rank_options.inference        = origami::inference_mode_t::analytical_rerank;
                    rank_options.nn_backend       = origami::nn_backend_t::boost;
                    rank_options.library_models   = &nn_models;
                    rank_options.nn.rerank_source = &rerankSource;
                }
                auto prediction_result = origami::rank_configs(
                    origami_problem, *(pAMDGPU->analyticalHardware), origami_config_list, rank_options);
                if(!rerankSource.prepared())
                    chargeStreamKClusters(prediction_result);

                for(const auto& r : prediction_result)
                {
                    if(r.config.index >= solution_list.size())
                    {
                        continue;
                    }
#if ORIGAMI_ENABLE_NN
                    if(std::isnan(r.latency))
                        continue;
#endif
                    considerSolution(solution_list[r.config.index].second);
                    if(rv.size() == static_cast<std::size_t>(numSolutions))
                    {
                        break;
                    }
                }

#if ORIGAMI_ENABLE_NN
                if(rv.empty())
                {
                    for(const auto& r : prediction_result)
                    {
                        if(r.config.index >= solution_list.size())
                        {
                            continue;
                        }
                        considerSolution(solution_list[r.config.index].second);
                        if(rv.size() == static_cast<std::size_t>(numSolutions))
                        {
                            break;
                        }
                    }
                }
#endif
            }
            // can't reach the requested number, means findTop already done its best
            lastFindTopRetAll = (rv.size() < static_cast<std::size_t>(numSolutions));
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
    };
} // namespace TensileLite

