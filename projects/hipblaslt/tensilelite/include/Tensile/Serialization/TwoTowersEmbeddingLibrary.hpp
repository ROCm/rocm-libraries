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
#include <Tensile/TwoTowersEmbeddingLibrary.hpp>

#include <cstddef>
#include <unordered_set>

namespace TensileLite
{
    namespace Serialization
    {

        template <typename IO>
        struct MappingTraits<TwoTowersEmbedding::StandardScaler, IO>
        {
            using Scaler = TwoTowersEmbedding::StandardScaler;
            using iot    = IOTraits<IO>;

            static void mapping(IO& io, Scaler& scaler)
            {
                std::vector<float> mean, scale;
                iot::mapRequired(io, "mean", mean);
                iot::mapRequired(io, "scale", scale);
                scaler.mean.assign(mean.begin(), mean.end()); // TODO check
                scaler.scale.assign(scale.begin(), scale.end());
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<TwoTowersEmbedding::TwoTowersEmbedding, IO>
        {
            using TwoTowersEmbedding = TwoTowersEmbedding::TwoTowersEmbedding;
            using iot                = IOTraits<IO>;

            static void mapping(IO& io, TwoTowersEmbedding& two_towers)
            {
                iot::mapRequired(io, "state_dict", two_towers.query_encoder);
                iot::mapRequired(io, "scaler", two_towers.scaler);
                iot::mapRequired(io, "apply_log", two_towers.apply_log);
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<TensileLite::TwoTowersEmbedding::QueryEncoder, IO>
        {
            using QueryEncoder = TensileLite::TwoTowersEmbedding::QueryEncoder;
            using iot          = IOTraits<IO>;

            static void mapping(IO& io, QueryEncoder& encoder)
            {
                iot::mapRequired(io, "proj_weights", encoder.proj_weights);
                iot::mapRequired(io, "proj_bias", encoder.proj_bias);

                iot::mapRequired(io, "weights", encoder.weights);
                iot::mapRequired(io, "bias", encoder.bias);
            }
            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<TensileLite::TwoTowersEmbedding::DocEmbeddings, IO>
        {
            using DocEmbeddings = TensileLite::TwoTowersEmbedding::DocEmbeddings;
            using iot           = IOTraits<IO>;

            static void mapping(IO& io, DocEmbeddings& docEmbeddings)
            {
                iot::mapRequired(io, "embeddings", docEmbeddings.embeddings);
                iot::mapRequired(io, "cluster_sols", docEmbeddings.cluster_sols);
                iot::mapRequired(io, "centroids", docEmbeddings.centroids);
            }

            const static bool flow = false;
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<TwoTowersEmbeddingLibrary<MyProblem, MySolution>, IO>
        {
            using Library = TwoTowersEmbeddingLibrary<MyProblem, MySolution>;
            using iot     = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {

                auto ctx = static_cast<LibraryIOContext<MySolution>*>(iot::getContext(io));
                if(ctx == nullptr)
                {
                    iot::setError(io,
                                  "TwoTowersEmbeddingLibrary requires that context be "
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
                                      "TwoTowersEmbeddingLibrary requires non empty "
                                      "mapping index set.");

                    for(int index : mappingIndices)
                    {
                        auto slnIter = ctx->solutions->find(index);
                        if(slnIter == ctx->solutions->end())
                        {
                            iot::setError(
                                io,
                                concatenate("[TwoTowersEmbeddingLibrary] Invalid solution index: ",
                                            index));
                        }
                        else
                        {
                            auto solution = slnIter->second;
                            lib.solutionmap.insert(std::make_pair(index, solution));
                            lib.solutions.push_back(solution);
                        }
                    }
                }

                std::shared_ptr<TwoTowersEmbedding::TwoTowersEmbedding> model;
                if(iot::outputting(io))
                {
                    model = std::dynamic_pointer_cast<TwoTowersEmbedding::TwoTowersEmbedding>(
                        lib.model);
                }
                else
                {
                    model     = std::make_shared<TwoTowersEmbedding::TwoTowersEmbedding>();
                    lib.model = model;
                }
                iot::mapRequired(io, "query_tower", *model);

                std::shared_ptr<TensileLite::TwoTowersEmbedding::DocEmbeddings> embeddings;
                if(iot::outputting(io))
                {
                    embeddings
                        = std::dynamic_pointer_cast<TensileLite::TwoTowersEmbedding::DocEmbeddings>(
                            lib.embeddings);
                }
                else
                {
                    embeddings = std::make_shared<TensileLite::TwoTowersEmbedding::DocEmbeddings>();
                    lib.embeddings = embeddings;
                }
                iot::mapRequired(io, "sol_embeddings", *embeddings);

                // Checks
                if(embeddings->size() != lib.solutions.size())
                    throw std::runtime_error(
                        "ERROR: TwoTowersEmbedding library solution embeddings amount "
                        "does not match solution map size.");
                if(lib.solutions.size() == 0)
                    throw std::runtime_error(
                        "ERROR: TwoTowersEmbedding library solution embeddings amount equals 0");

                if(lib.model->query_encoder.proj_bias.size() != embeddings->embeddings[0][0].size())
                    throw std::runtime_error(
                        "ERROR: TwoTowersEmbedding library solution embeddings size "
                        "does not match model output size.");
            }

            const static bool flow = false;
        };

    } // namespace Serialization
} // namespace TensileLite
