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

#include <set>
#include <vector>

#include <origami/utils.hpp>

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
        std::unordered_map<int, std::shared_ptr<MySolution>>        solutionmap;
        std::vector<origami::tile_tuple>             tile_list;
        std::unordered_map<origami::tile_tuple, int> tile_map;

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
            if(solutionmap.empty())
                return concatenate(type(), ", solutionmap: empty");
            return concatenate(type(), solutionmap.size());
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
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

            for(auto const& row : this->solutionmap)
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

            for(auto const& row : this->solutionmap)
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

            bool                  debug   = Debug::Instance().printPropertyEvaluation();
            hip::HipAMDGPU const* pAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
            size_t elementSizeA_bits
                = problem.a().elementBytes() * 8;
            size_t elementSizeB_bits
                = problem.b().elementBytes() * 8;
            size_t elementSizeC_bits
                = problem.c().elementBytes() * 8;
            const origami::hardware_t& analaytical_hardware = *(pAMDGPU->analyticalHardware);
            if(origami::hardware_t::is_debug_enabled())
            {
                analaytical_hardware.print();
            }
            int defaultWGM = std::ceil(std::sqrt(analaytical_hardware.N_CU / analaytical_hardware.NUM_XCD));
            origami::data_type_t miDataType = static_cast<origami::data_type_t>(problem.computeInputType());
            if(problem.f32XdlMathOp() == rocisa::DataType::XFloat32) // Check F32 compute type
                miDataType = origami::data_type_t::XFloat32;
            auto selected_tiles = origami::select_best_macro_tile_size(
                m,
                n,
                k,
                batch,
                problem.transA(),
                problem.transB(),
                *(pAMDGPU->analyticalHardware),
                tile_list,
                elementSizeA_bits,
                elementSizeB_bits,
                elementSizeC_bits,
                miDataType,
                0,   // mx_block_size -> MX Data types come from rocroller.
                0.8, // L2 hit-rate (not used anymore -- should be removed)
                false,
                defaultWGM,
                pAMDGPU->skMaxCUs);
            for(const auto& tile : selected_tiles)
            {
                auto mapiter  = tile_map.find(std::make_tuple(std::get<1>(tile),
                                                              std::get<2>(tile),
                                                              std::get<3>(tile),
                                                              std::get<4>(tile),
                                                              std::get<5>(tile),
                                                              std::get<6>(tile),
                                                              std::get<7>(tile),
                                                              std::get<8>(tile),
                                                              std::get<9>(tile),
                                                              std::get<10>(tile)
                                                            ));
                auto smapiter = solutionmap.find(mapiter->second);
                if(mapiter != tile_map.end() && smapiter != solutionmap.end())
                {
                    auto solution = smapiter->second;
                    if((*solution->hardwarePredicate)(hardware)
                       && (*solution->problemPredicate)(problem))
                    {
                        rv.emplace_back(solution);
                        if(rv.size() == numSolutions)
                        {
                            break;
                        }
                    }
                }
            }
            return rv;
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
