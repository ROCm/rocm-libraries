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
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <vector>

#include <Tensile/Debug.hpp>
#include <Tensile/PredicateDebugger.hpp>
#include <Tensile/UtilsOrigami.hpp>

#include "tilewright/hipblaslt/adapter.hpp"

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
        // Pre-built tilewright configs (base params + ML features), index-aligned
        // with origami_config_list. Built once at deserialize (the kernel name +
        // SizeMapping are available there); see Serialization/PredictionLibrary.hpp.
        std::vector<tilewright::Config> tilewright_config_list;
        // Per-library tilewright model handle, resolved at deserialize from this
        // logic file's stem via the colocated tilewright_index (see
        // Serialization/PredictionLibrary.hpp). -1 => no tilewright model for this
        // library; the code below uses the base analytical (origami) path instead.
        int tilewright_model = -1;

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
            auto indexMatch = std::find_if(solution_list.begin(),
                                           solution_list.end(),
                                           [&index](auto& s) { return s.first == index; });
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
            // numSolutions == 0 requests nothing; return before ranking/selection
            // (the selection loop below only breaks AFTER a push, so 0 would
            // otherwise return at least one solution).
            if(numSolutions == 0)
                return rv;
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

            hip::HipAMDGPU const* pAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);

            const bool debug = Debug::Instance().printPropertyEvaluation();

            auto considerSolution = [&](std::shared_ptr<MySolution> const& solution) {
                const bool hwMatch   = (*(solution->hardwarePredicate))(hardware);
                const bool probMatch = (*(solution->problemPredicate))(problem);
                const bool predicateMatch = hwMatch && probMatch;

                if(debug)
                {
                    PredicateDebugger::printHeader(
                        std::cout, "Prediction: " + solution->name());
                    solution->hardwarePredicate->debugEval(hardware, std::cout);
                    solution->problemPredicate->debugEval(problem, std::cout);
                    PredicateDebugger::printFooter(std::cout, predicateMatch);
                }

                if(predicateMatch)
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

                // Use the tilewright recommender only when this library actually has a
                // per-library model (loaded via tilewright_index at deserialize).
                // Otherwise fall back to plain analytical origami -- NOT to some
                // other dtype/layout's model.
                //
                // Pass the requested solution count as tilewright's ranking depth
                // (min_scored), covering every request shape:
                //   numSolutions <  0  (e.g. -1 "all"): score EVERY feasible config
                //                       (LDS + kernel-feasibility filtered) beyond the
                //                       smart_K whitelist -> SIZE_MAX depth.
                //   numSolutions <= smart_K size: whitelist only, in score order
                //                       (numSolutions == 1 == the common top-1 path,
                //                       identical picks, no extra cost).
                //   numSolutions >  smart_K size: extend past the whitelist. The engine
                //                       clamps to the feasible set, so e.g. 999 with 257
                //                       feasible kernels ranks all 257 (never overruns).
                const std::size_t tilewrightDepth
                    = numSolutions < 0 ? std::numeric_limits<std::size_t>::max()
                                       : static_cast<std::size_t>(numSolutions);
                auto              prediction_result
                    = (Debug::Instance().useTilewright() && tilewright_model >= 0)
                          ? tilewright::hipblaslt::rank_configs(tilewright_model,
                                                                origami_problem,
                                                                *(pAMDGPU->analyticalHardware),
                                                                tilewright_config_list,
                                                                origami_config_list,
                                                                tilewrightDepth)
                          : origami::rank_configs(origami_problem,
                                                  *(pAMDGPU->analyticalHardware),
                                                  origami_config_list);

                for(const auto& r : prediction_result)
                {
                    if(r.config.index >= solution_list.size())
                    {
                        continue;
                    }
                    // Only return configs the recommender actually scored (i.e. that
                    // survived the LDS + kernel-feasibility pre-filters). Filtered-out
                    // configs come back with a NaN "latency" sentinel; never select them,
                    // even when many solutions are requested -- "all" (rsn < 0) means all
                    // FEASIBLE, not the full padded pool.
                    if(std::isnan(r.latency))
                    {
                        continue;
                    }
                    considerSolution(solution_list[r.config.index].second);
                    if(rv.size() == numSolutions)
                    {
                        break;
                    }
                }

                // Safety net: if the recommender scored nothing (e.g. no per-cell
                // model matched and every config came back unscored), fall back to
                // predicate-passing configs so a kernel is still returned.
                if(rv.empty())
                {
                    for(const auto& r : prediction_result)
                    {
                        if(r.config.index >= solution_list.size())
                        {
                            continue;
                        }
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
    };
} // namespace TensileLite

