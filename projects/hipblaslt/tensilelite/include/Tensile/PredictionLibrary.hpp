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
#include <iostream>
#include <limits>
#include <set>
#include <vector>

#include <Tensile/Debug.hpp>
#include <Tensile/PredicateDebugger.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/UtilsOrigami.hpp>

#ifdef TENSILELITE_HAS_TILEWRIGHT
#include <tilewright/model.hpp>
#include <tilewright/types.hpp>
#endif

#include <tensilelitehost/export.h>

// Declared unconditionally so ProblemPredictionLibrary has the same layout in
// translation units built with and without TENSILELITE_HAS_TILEWRIGHT.
namespace tilewright
{
    class CandidateSet;
}

namespace TensileLite
{
#ifdef TENSILELITE_HAS_TILEWRIGHT
    inline tilewright::DataType datatypeToTilewrightDatatype(rocisa::DataType type)
    {
        switch(type)
        {
        case rocisa::DataType::Float:
            return tilewright::DataType::Float;
        case rocisa::DataType::Double:
            return tilewright::DataType::Double;
        case rocisa::DataType::ComplexFloat:
            return tilewright::DataType::ComplexFloat;
        case rocisa::DataType::ComplexDouble:
            return tilewright::DataType::ComplexDouble;
        case rocisa::DataType::Half:
            return tilewright::DataType::Half;
        case rocisa::DataType::Int8x4:
            return tilewright::DataType::Int8x4;
        case rocisa::DataType::Int32:
            return tilewright::DataType::Int32;
        case rocisa::DataType::BFloat16:
            return tilewright::DataType::BFloat16;
        case rocisa::DataType::Int8:
            return tilewright::DataType::Int8;
        case rocisa::DataType::Int64:
            return tilewright::DataType::Int64;
        case rocisa::DataType::XFloat32:
            return tilewright::DataType::XFloat32;
        case rocisa::DataType::Float8_fnuz:
            return tilewright::DataType::Float8_fnuz;
        case rocisa::DataType::BFloat8_fnuz:
            return tilewright::DataType::BFloat8_fnuz;
        case rocisa::DataType::Float8BFloat8_fnuz:
            return tilewright::DataType::Float8BFloat8_fnuz;
        case rocisa::DataType::BFloat8Float8_fnuz:
            return tilewright::DataType::BFloat8Float8_fnuz;
        case rocisa::DataType::Float8:
            return tilewright::DataType::Float8;
        case rocisa::DataType::BFloat8:
            return tilewright::DataType::BFloat8;
        case rocisa::DataType::Float8BFloat8:
            return tilewright::DataType::Float8BFloat8;
        case rocisa::DataType::BFloat8Float8:
            return tilewright::DataType::BFloat8Float8;
        case rocisa::DataType::Float6:
            return tilewright::DataType::Float6;
        case rocisa::DataType::BFloat6:
            return tilewright::DataType::BFloat6;
        case rocisa::DataType::Float4:
            return tilewright::DataType::Float4;
        default:
            return tilewright::DataType::None;
        }
    }
#endif

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
        // Config i of the set is solution_list[i].
        std::shared_ptr<const tilewright::CandidateSet> tilewright_candidates;

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
            if(logicFile.empty())
                return;

            std::string dir          = ".";
            std::string stem         = logicFile;
            size_t      directoryPos = stem.rfind('/');
#ifdef _WIN32
            if(directoryPos == std::string::npos)
                directoryPos = stem.rfind('\\');
#endif
            if(directoryPos != std::string::npos)
            {
                dir  = stem.substr(0, directoryPos);
                stem = stem.substr(directoryPos + 1);
            }
            // Logic stems contain no '.', and the file may carry two extensions.
            stem = stem.substr(0, stem.find('.'));

            std::string error;
            try
            {
                auto model = tilewright::load_model_by_index(stem, dir, &error);
                if(model)
                {
                    std::vector<tilewright::Config> configs;
                    configs.reserve(solution_list.size());
                    for(size_t i = 0; i < solution_list.size(); i++)
                    {
                        auto const& sizeMapping = solution_list[i].second->sizeMapping;
                        auto const& mi          = sizeMapping.matrixInstruction;

                        // Same stand-in for dot2 kernels as the origami config.
                        tilewright::Dim3 miSize{1, 1, 64};
                        if(mi[0] != 0 || mi[1] != 0 || mi[2] != 0)
                            miSize = {static_cast<size_t>(mi[0]),
                                      static_cast<size_t>(mi[1]),
                                      static_cast<size_t>(mi[2])};

                        configs.push_back({
                            .mt            = {sizeMapping.macroTile.x,
                                              sizeMapping.macroTile.y,
                                              sizeMapping.depthU},
                            .mi            = miSize,
                            .occupancy     = std::max(sizeMapping.CUOccupancy, 1),
                            .cache_hints_a = sizeMapping.cacheHintA(),
                            .cache_hints_b = sizeMapping.cacheHintB(),
                            .grvw_a        = sizeMapping.grvwA,
                            .grvw_b        = sizeMapping.grvwB,
                            .gwvw_d        = sizeMapping.gwvwD,
                            .index         = i,
                        });
                    }
                    tilewright_candidates = std::make_shared<const tilewright::CandidateSet>(
                        std::move(model), std::move(configs));
                }
            }
            catch(std::exception const& e)
            {
                error = e.what();
            }
            catch(...)
            {
                error = "unknown exception";
            }

            if(!tilewright_candidates && !error.empty())
                std::cerr << "hipBLASLt Warning: TENSILE_USE_TILEWRIGHT is set but the tilewright "
                             "model for "
                          << stem << " could not be used (" << error
                          << "); ranking with origami instead.\n";
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

#ifdef TENSILELITE_HAS_TILEWRIGHT
                // Solutions tilewright already offered; origami ranks the rest.
                std::vector<bool> offered;
                if(tilewright_candidates)
                {
                    const tilewright::Problem tilewright_problem = {
                        .size  = {m, n, k},
                        .batch = batch,
                        .a_transpose
                        = problem.transA() ? tilewright::Transpose::T : tilewright::Transpose::N,
                        .b_transpose
                        = problem.transB() ? tilewright::Transpose::T : tilewright::Transpose::N,
                        .a_dtype  = datatypeToTilewrightDatatype(problem.a().dataType()),
                        .b_dtype  = datatypeToTilewrightDatatype(problem.b().dataType()),
                        .c_dtype  = datatypeToTilewrightDatatype(problem.c().dataType()),
                        .d_dtype  = datatypeToTilewrightDatatype(problem.d().dataType()),
                        .mi_dtype = problem.f32XdlMathOp() == rocisa::DataType::XFloat32
                                        ? tilewright::DataType::XFloat32
                                        : datatypeToTilewrightDatatype(problem.computeInputTypeA()),
                    };
                    const tilewright::Hardware tilewright_hardware = {
                        .N_CU         = pAMDGPU->analyticalHardware->N_CU,
                        .lds_capacity = pAMDGPU->analyticalHardware->lds_capacity,
                        .L2_capacity  = pAMDGPU->analyticalHardware->L2_capacity,
                    };

                    const size_t depth = numSolutions < 0 ? std::numeric_limits<size_t>::max()
                                                          : static_cast<size_t>(numSolutions);
                    std::vector<tilewright::Result> ranked;
                    try
                    {
                        ranked = tilewright_candidates->rank(
                            tilewright_problem, tilewright_hardware, depth);
                    }
                    catch(std::exception const&)
                    {
                        // Ranking only fails to allocate; origami ranks every kernel then.
                        ranked.clear();
                    }
                    offered.assign(solution_list.size(), false);
                    for(const auto& r : ranked)
                    {
                        if(!r.scored)
                        {
                            break;
                        }
                        if(r.config_index >= solution_list.size())
                        {
                            continue;
                        }
                        offered[r.config_index] = true;
                        considerSolution(solution_list[r.config_index].second);
                        if(rv.size() == numSolutions)
                        {
                            break;
                        }
                    }
                }
#endif

                if(rv.size() != numSolutions)
                {
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
    };
} // namespace TensileLite

