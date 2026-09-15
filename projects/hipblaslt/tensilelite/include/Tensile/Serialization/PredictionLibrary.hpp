/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/SingleSolutionLibrary.hpp>

#include <Tensile/PredictionLibrary.hpp>

#include <Tensile/Debug.hpp>
#if ORIGAMI_ENABLE_NN
#include <origami/nn/nn.hpp>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <tensilelitehost/export.h>

namespace TensileLite
{
    namespace Serialization
    {

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<ProblemPredictionLibrary<MyProblem, MySolution>, IO>
        {
            using Library = ProblemPredictionLibrary<MyProblem, MySolution>;
            using iot     = IOTraits<IO>;

#if ORIGAMI_ENABLE_NN
            static std::string logic_stem_from_path(const std::string& path)
            {
                std::string stem     = path;
                const auto  slashPos = stem.find_last_of("/\\");
                if(slashPos != std::string::npos)
                    stem = stem.substr(slashPos + 1);
                const auto periodPos = stem.find('.');
                if(periodPos != std::string::npos)
                    stem = stem.substr(0, periodPos);
                return stem;
            }

            static std::string directory_from_path(const std::string& path)
            {
                const auto slashPos = path.find_last_of("/\\");
                if(slashPos == std::string::npos)
                    return ".";
                return path.substr(0, slashPos);
            }
#endif

            static void mapping(IO& io, Library& lib)
            {
                auto ctx = static_cast<LibraryIOContext<MySolution>*>(iot::getContext(io));
                if(ctx == nullptr)
                {
                    iot::setError(io,
                                  "ProblemPredictionLibrary requires that context be "
                                  "set to a SolutionMap.");
                }
                std::vector<int> mappingIndices;
                if(iot::outputting(io))
                {
                    mappingIndices.reserve(lib.solution_list.size());

                    for(auto const& pair : lib.solution_list)
                        mappingIndices.push_back(pair.first);

                    iot::mapRequired(io, "table", mappingIndices);
                }
                else
                {
                    iot::mapRequired(io, "table", mappingIndices);
                    if(mappingIndices.empty())
                        iot::setError(io,
                                      "ProblemPredictionLibrary requires non empty "
                                      "mapping index set.");

                    for(std::size_t local_index = 0; local_index < mappingIndices.size();
                        local_index++)
                    {
                        int  index   = mappingIndices[local_index];
                        auto slnIter = ctx->solutions->find(index);
                        if(slnIter == ctx->solutions->end())
                        {
                            iot::setError(
                                io,
                                concatenate("[ProblemPredictionLibrary] Invalid solution index: ",
                                            index));
                        }
                        else
                        {
                            auto solution = slnIter->second;
                            lib.solution_list.emplace_back(index, solution);

                            origami::dim3_t origami_mi;
                            if(solution->sizeMapping.matrixInstruction[0] == 0
                               && solution->sizeMapping.matrixInstruction[1] == 0
                               && solution->sizeMapping.matrixInstruction[2] == 0)
                            {
                                // Override dot2 instruction with vector lane widths
                                origami_mi = {1, 1, 64};
                            }
                            else
                            {
                                origami_mi = {
                                    static_cast<size_t>(solution->sizeMapping.matrixInstruction[0]),
                                    static_cast<size_t>(solution->sizeMapping.matrixInstruction[1]),
                                    static_cast<size_t>(
                                        solution->sizeMapping.matrixInstruction[2])};
                            }

                            if(Debug::Instance().printPropertyEvaluation()
                               && solution->sizeMapping.CUOccupancy <= 0)
                            {
                                std::cerr << "TensileLite::DEBUG: sizeMapping.CUOccupancy="
                                          << solution->sizeMapping.CUOccupancy
                                          << " (<=0) for solution '" << solution->kernelName
                                          << "'; clamping to 1 in origami config.\n";
                            }
                            origami::config_t origami_config = {
                                .mt = {solution->sizeMapping.macroTile.x,
                                       solution->sizeMapping.macroTile.y,
                                       solution->sizeMapping.depthU},
                                .mi = origami_mi,
                                .hand_optimized_main_loop
                                = (solution->sizeMapping.customMainLoopScheduling > 0) ? true
                                                                                       : false,
                                .subtile = solution->sizeMapping.useSubtileImpl,
                                .occupancy
                                = std::max(solution->sizeMapping.CUOccupancy, static_cast<int>(1)),
                                .workgroup_mapping         = solution->sizeMapping.workGroupMapping,
                                .cache_hints_a             = solution->sizeMapping.nonTemporalA,
                                .cache_hints_b             = solution->sizeMapping.nonTemporalB,
                                .workspace_size            = std::numeric_limits<size_t>::max(),
                                .workspace_size_per_elem_c = std::numeric_limits<size_t>::max(),
                                .stream_k                  = solution->sizeMapping.streamK,
                                .index                     = local_index,
                            };

                            lib.origami_config_list.emplace_back(origami_config);
                        }
                    }

#if ORIGAMI_ENABLE_NN
                    const std::string logicStem = logic_stem_from_path(ctx->filename);
                    const std::string dataDir   = directory_from_path(ctx->filename);
                    lib.nn_models = origami::nn::load_models_for_logic(logicStem, dataDir);
                    if(const char* diag = std::getenv("ORIGAMI_NN_DIAG"))
                    {
                        if(diag[0] != '\0' && std::strcmp(diag, "0") != 0)
                        {
                            std::fprintf(stderr,
                                         "[ORIGAMI_NN_DIAG] PredictionLibrary stem=%s dir=%s "
                                         "tilewright_handle=%d\n",
                                         logicStem.c_str(),
                                         dataDir.c_str(),
                                         lib.nn_models.tilewright);
                            std::fflush(stderr);
                        }
                    }
#endif
                }
            }
            const static bool flow = false;
        };
    } // namespace Serialization
} // namespace TensileLite
