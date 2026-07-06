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
#include <cstdlib>
#include <set>
#include <vector>

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

            const origami::hardware_t& analytical_hardware = *(pAMDGPU->analyticalHardware);
            auto miDataType = datatypeToAnalyticalDatatype(problem.computeInputTypeA());

            if(problem.f32XdlMathOp() == rocisa::DataType::XFloat32) // Check F32 compute type
                miDataType = origami::data_type_t::XFloat32;
            origami::problem_t origami_problem = {
                .size        = {m, n, k},
                .batch       = batch,
                // Number of CUs the GEMM will run on (CU budget hint); 0 = use all CUs.
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

            // When ORIGAMI_LEVELED_ESTIMATION is set, route selection through the
            // leveled coarse-to-fine cascade (a single estimation phase makes
            // rank_configs use GemmModel::score_candidates -> score_estimation_leveled)
            // instead of the flat per-config path.
            static const bool use_leveled_estimation = [] {
                const char* e = std::getenv("ORIGAMI_LEVELED_ESTIMATION");
                return e != nullptr && std::strtol(e, nullptr, 0) != 0;
            }();

            // Split-K problems regress under the leveled cascade: StreamK splits the
            // K loop across CUs and the winner is decided by memory reuse, which the
            // context-free coarse levels can't see -- so they prune it. Route those
            // problems to the accurate flat per-config path; the fast leveled path
            // handles the non-split-K majority.
            //
            // Detect "split-K-prone" with StreamK's own select_reduction criterion,
            // evaluated on a large reference tile: a problem needs K-splitting when
            // even a big (256x256) tile underfills the GPU (output tiles < N_CU) and K
            // is deep enough to split (iters/tile >= 64, i.e. K >= 64*64). This is
            // shape-based (not a raw K cutoff), so it catches split-K at moderate K
            // too. On the workload it cuts leveled divergence to ~0.4% (routing ~21%
            // of problems to flat) vs ~1.4% for a plain K>=10k threshold.
            const size_t ref_tile      = 256;
            const size_t ref_tiles     = ((m + ref_tile - 1) / ref_tile)
                                     * ((n + ref_tile - 1) / ref_tile) * batch;
            const size_t ref_k_iters   = k / 64;  // K-iterations for a 64-deep k-tile
            const bool   split_k_prone = ref_tiles < analytical_hardware.N_CU && ref_k_iters >= 64;
            const bool   use_leveled   = use_leveled_estimation && !split_k_prone;

            std::vector<origami::prediction_result_t> prediction_result;
            if(use_leveled)
            {
                origami::ranking_phase_t phase;
                phase.model    = origami::model_t::gemm;
                phase.target   = origami::target_t::tensilelite;
                phase.fidelity = origami::prediction_modes_t::estimation;
                origami::ranking_pipeline_t pipeline;
                pipeline.phases.push_back(phase);
                prediction_result = origami::rank_configs(origami_problem,
                                                          analytical_hardware,
                                                          origami_config_list,
                                                          origami::model_t::gemm,
                                                          pipeline);
            }
            else
            {
                prediction_result = origami::rank_configs(
                    origami_problem, *(pAMDGPU->analyticalHardware), origami_config_list);
            }

            for(const auto& r : prediction_result)
            {
                auto& solution = solution_list[r.config.index].second;
                if((*(solution->hardwarePredicate))(hardware)
                   && (*(solution->problemPredicate))(problem))
                {
                    rv.emplace_back(solution);
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

