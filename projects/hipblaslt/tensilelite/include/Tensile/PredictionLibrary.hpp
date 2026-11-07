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
#include <iostream>
#include <mutex>
#include <set>
#include <vector>

#include <Tensile/Debug.hpp>
#include <Tensile/ParamScorer.hpp>
#include <Tensile/PredicateDebugger.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/UtilsOrigami.hpp>

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

        mutable std::atomic<bool> lastFindTopRetAll = false;

        // Optional param scorer (TENSILE_PARAM_MODEL, see ParamScorer.hpp): bound
        // once per library. paramPosOfRow[row] = solution_list position; empty
        // when the model does not belong to this library.
        mutable std::once_flag    paramBindOnce;
        mutable std::vector<int>  paramPosOfRow;

        bool bindParamModel(ParamScorer::Model const& model) const
        {
            std::call_once(paramBindOnce, [&]() {
                std::vector<ParamScorer::SolutionInfo> infos;
                infos.reserve(solution_list.size());
                for(auto const& entry : solution_list)
                {
                    auto const&               sol = *entry.second;
                    ParamScorer::SolutionInfo info;
                    info.libraryLogicIndex = sol.libraryLogicIndex;
                    info.name              = sol.name();
                    info.mt0               = sol.sizeMapping.macroTile.x;
                    info.mt1               = sol.sizeMapping.macroTile.y;
                    info.depthU            = sol.sizeMapping.depthU;
                    info.mi0               = sol.sizeMapping.matrixInstruction[0];
                    info.mi1               = sol.sizeMapping.matrixInstruction[1];
                    info.wg0               = sol.sizeMapping.workGroupSize.x;
                    info.wg1               = sol.sizeMapping.workGroupSize.y;
                    info.wg2               = sol.sizeMapping.workGroupSize.z;
                    info.wgm               = sol.sizeMapping.workGroupMapping;
                    info.ldsb              = sol.sizeMapping.OneLDSBuffer ? 1 : 0;
                    info.pgr               = sol.sizeMapping.PrefetchGlobalRead;
                    infos.push_back(std::move(info));
                }
                auto rowOfPos = ParamScorer::bindLibrary(model, infos);
                if(!rowOfPos.empty())
                {
                    std::vector<int> posOfRow(rowOfPos.size(), -1);
                    for(size_t pos = 0; pos < rowOfPos.size(); ++pos)
                        posOfRow[rowOfPos[pos]] = static_cast<int>(pos);
                    paramPosOfRow.swap(posOfRow);
                }
            });
            return !paramPosOfRow.empty();
        }

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

            auto considerSolution = [&](std::shared_ptr<MySolution> const& solution) {
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

                if(predicateMatch)
                {
                    rv.emplace_back(solution);
                }
            };

            auto makeOrigamiProblem = [&]() {
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
                return origami_problem;
            };

            ParamScorer::Model const* paramModel = ParamScorer::activeModel();
            if(paramModel && bindParamModel(*paramModel))
            {
                // Param-scorer order (model rows -> solution_list positions). Only the
                // first `sortedPrefix` rows are sorted up front (the walk almost always
                // stops there); the tail is sorted when the walk reaches it, so the
                // visiting order is exactly the full score order.
                const int           topk         = ParamScorer::topK(*paramModel);
                const size_t        sortedPrefix = ParamScorer::sortedPrefix(*paramModel);
                std::vector<double> scores;
                auto                rows
                    = ParamScorer::rankRows(*paramModel, m, n, k, batch, &scores, sortedPrefix);
                std::vector<size_t> order;
                order.reserve(rows.size());
                for(int row : rows)
                    order.push_back(static_cast<size_t>(paramPosOfRow[row]));

                // Optional: rerank the top k by Origami; the rest keep param order.
                size_t    reranked = 0;
                if(topk > 1 && pAMDGPU && pAMDGPU->analyticalHardware)
                {
                    const size_t kk = std::min(static_cast<size_t>(topk), order.size());
                    std::vector<origami::config_t> subset;
                    subset.reserve(kk);
                    for(size_t i = 0; i < kk; ++i)
                        if(order[i] < origami_config_list.size())
                            subset.push_back(origami_config_list[order[i]]);
                    if(!subset.empty())
                    {
                        auto ranked = origami::rank_configs(
                            makeOrigamiProblem(), *(pAMDGPU->analyticalHardware), subset);
                        std::vector<char>   used(solution_list.size(), 0);
                        std::vector<size_t> head;
                        head.reserve(order.size());
                        for(const auto& r : ranked)
                        {
                            if(r.config.index < solution_list.size() && !used[r.config.index])
                            {
                                head.push_back(r.config.index);
                                used[r.config.index] = 1;
                            }
                        }
                        reranked = head.size();
                        // Top-k members Origami rejected, then the rest, in param order.
                        for(size_t pos : order)
                        {
                            if(!used[pos])
                            {
                                head.push_back(pos);
                                used[pos] = 1;
                            }
                        }
                        order.swap(head);
                    }
                }

                // The rerank only permutes positions < sortedPrefix, so order[i] for
                // i >= sortedPrefix is still paramPosOfRow[rows[i]] (the unsorted tail).
                for(size_t i = 0; i < order.size(); ++i)
                {
                    if(i == sortedPrefix)
                    {
                        ParamScorer::sortRowsTail(scores, rows, sortedPrefix);
                        for(size_t j = sortedPrefix; j < rows.size(); ++j)
                            order[j] = static_cast<size_t>(paramPosOfRow[rows[j]]);
                    }
                    considerSolution(solution_list[order[i]].second);
                    if(rv.size() == numSolutions)
                    {
                        break;
                    }
                }

                if(ParamScorer::debug())
                {
                    std::cerr << "[TENSILE_PARAM] m=" << m << " n=" << n << " k=" << k
                              << " b=" << batch << " topk=" << topk << " reranked=" << reranked
                              << " chosen=";
                    if(rv.empty())
                        std::cerr << "none";
                    else
                        std::cerr << rv[0]->libraryLogicIndex;
                    std::cerr << " | param top5:";
                    for(size_t i = 0; i < std::min<size_t>(5, rows.size()); ++i)
                        std::cerr << " " << rows[i] << ":" << scores[rows[i]];
                    std::cerr << " | walk head:";
                    for(size_t i = 0; i < std::min<size_t>(5, order.size()); ++i)
                        std::cerr << " " << solution_list[order[i]].second->libraryLogicIndex;
                    std::cerr << std::endl;
                }
            }
            else if(pAMDGPU && pAMDGPU->analyticalHardware)
            {
                auto prediction_result = origami::rank_configs(
                    makeOrigamiProblem(), *(pAMDGPU->analyticalHardware), origami_config_list);

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

