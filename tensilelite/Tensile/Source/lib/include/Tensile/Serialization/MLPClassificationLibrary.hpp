/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/Debug.hpp>
#include <Tensile/MLPClassificationLibrary.hpp>

#include <cstddef>
#include <unordered_set>

namespace TensileLite
{
    namespace Serialization
    {

        template <typename IO>
        struct MappingTraits<MLPClassification::StandardScaler, IO>
        {
            using Scaler = MLPClassification::StandardScaler;
            using iot    = IOTraits<IO>;

            static void mapping(IO& io, Scaler& scaler)
            {
                std::vector<float> mean, scale;
                iot::mapRequired(io, "mean", mean);
                iot::mapRequired(io, "scale", scale);
                scaler.mean.assign(mean.begin(), mean.end());
                scaler.scale.assign(scale.begin(), scale.end());
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<MLPClassification::MLPNet, IO>
        {
            using MLPNet = MLPClassification::MLPNet;
            using iot     = IOTraits<IO>;

            static void mapping(IO& io, MLPNet& mlp)
            {
                iot::mapRequired(io, "scaler", mlp.scaler);
                iot::mapRequired(io, "res_blocks", mlp.res_blocks);
                iot::mapRequired(io, "dense", mlp.dense);
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<MLPClassification::DenseLayer, IO>
        {
            using DenseLayer = MLPClassification::DenseLayer;
            using iot        = IOTraits<IO>;

            static void mapping(IO& io, DenseLayer& l)
            {
                std::vector<float> W, B;
                iot::mapRequired(io, "weight", W);
                iot::mapRequired(io, "bias", B);
                l = DenseLayer(W, B);
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<MLPClassification::ResBlock, IO>
        {
            using ResBlock = MLPClassification::ResBlock;
            using iot      = IOTraits<IO>;

            static void mapping(IO& io, ResBlock& block)
            {
                iot::mapRequired(io, "linear1", block.linear1);
                iot::mapRequired(io, "linear2", block.linear2);
                iot::mapRequired(io, "res", block.res);
            }

            const static bool flow = false;
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<MLPClassificationLibrary<MyProblem, MySolution>, IO>
        {
            using Library = MLPClassificationLibrary<MyProblem, MySolution>;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                auto ctx = static_cast<LibraryIOContext<MySolution>*>(iot::getContext(io));
                if(ctx == nullptr)
                {
                    iot::setError(io,
                                  "MLPClassificationLibrary requires that context be "
                                  "set to a SolutionMap.");
                }
                std::vector<int> mappingIndices;
                if(iot::outputting(io))
                {
                    mappingIndices.reserve(lib.solutionmap.size());

                    for(auto const& pair : lib.solutionmap)
                        mappingIndices.push_back(pair.first);

                    iot::mapRequired(io, "table", mappingIndices);
                }
                else
                {
                    iot::mapRequired(io, "table", mappingIndices);
                    if(mappingIndices.empty())
                        iot::setError(io,
                                      "MLPClassificationLibrary requires non empty "
                                      "mapping index set.");

                    for(int index : mappingIndices)
                    {
                        auto slnIter = ctx->solutions->find(index);
                        if(slnIter == ctx->solutions->end())
                        {
                            iot::setError(
                                io,
                                concatenate("[MLPClassificationLibrary] Invalid solution index: ",
                                            index));
                        }
                        else
                        {
                            auto solution = slnIter->second;
                            lib.solutionmap.insert(std::make_pair(index, solution));
                        }
                    }
                }

                using MLPNet = MLPClassification::MLPNet;
                std::shared_ptr<MLPNet> model;
                if(iot::outputting(io))
                {
                    model = std::dynamic_pointer_cast<MLPNet>(lib.model);
                }
                else
                {
                    model     = std::make_shared<MLPNet>();
                    lib.model = model;
                }
                iot::mapRequired(io, "mlp", *model);
                if(!model->valid(true))
                    throw std::runtime_error("ERROR: MLP library not in a valid state.");

                using ProblemFeatures
                    = std::vector<std::shared_ptr<MLFeatures::MLFeature<MyProblem>>>;
                ProblemFeatures probFeatures;
                if(iot::outputting(io))
                {
                    probFeatures = lib.probFeatures;
                }
                iot::mapOptional(io, "problemFeatures", probFeatures);
                lib.probFeatures = probFeatures;
            }
            const static bool flow = false;
        };

    } // namespace Serialization
} // namespace TensileLite
