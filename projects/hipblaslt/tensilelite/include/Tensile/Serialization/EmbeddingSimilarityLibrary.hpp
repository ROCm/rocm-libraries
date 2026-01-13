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
#include <Tensile/EmbeddingSimilarityLibrary.hpp>

#include <cstddef>
#include <unordered_set>

namespace TensileLite
{
    namespace Serialization
    {

        template <typename IO>
        struct MappingTraits<EmbeddingSimilarity::StandardScaler, IO>
        {
            using Scaler = EmbeddingSimilarity::StandardScaler;
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
        struct MappingTraits<EmbeddingSimilarity::Encoder, IO>
        {
            using Encoder = EmbeddingSimilarity::Encoder;
            using iot     = IOTraits<IO>;

            static void mapping(IO& io, Encoder& encoder)
            {
                iot::mapRequired(io, "state_dict", encoder.network);
                iot::mapRequired(io, "scaler", encoder.scaler);
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<EmbeddingSimilarity::Network, IO>
        {
            using Network = EmbeddingSimilarity::Network;
            using iot     = IOTraits<IO>;

            static void mapping(IO& io, Network& net)
            {
                iot::mapRequired(io, "proj_weights", net.proj_weights);
                iot::mapRequired(io, "proj_bias", net.proj_bias);

                iot::mapRequired(io, "weights", net.weights);
                iot::mapRequired(io, "bias", net.bias);
            }
            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<EmbeddingSimilarity::SolutionEmbeddings, IO>
        {
            using SolutionEmbeddings = EmbeddingSimilarity::SolutionEmbeddings;
            using iot                = IOTraits<IO>;

            static void mapping(IO& io, SolutionEmbeddings& data)
            {
                iot::mapRequired(io, "embeddings", data.embeddings);
                iot::mapRequired(io, "cluster_indices", data.cluster_indices);
                iot::mapRequired(io, "centroids", data.centroids);
            }

            const static bool flow = false;
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<EmbeddingSimilarityLibrary<MyProblem, MySolution>, IO>
        {
            using Library = EmbeddingSimilarityLibrary<MyProblem, MySolution>;
            using iot     = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {

                auto ctx = static_cast<LibraryIOContext<MySolution>*>(iot::getContext(io));
                if(ctx == nullptr)
                {
                    iot::setError(io,
                                  "EmbeddingSimilarityLibrary requires that context be "
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
                                      "EmbeddingSimilarityLibrary requires non empty "
                                      "mapping index set.");

                    for(int index : mappingIndices)
                    {
                        auto slnIter = ctx->solutions->find(index);
                        if(slnIter == ctx->solutions->end())
                        {
                            iot::setError(
                                io,
                                concatenate("[EmbeddingSimilarityLibrary] Invalid solution index: ",
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

                std::shared_ptr<EmbeddingSimilarity::Encoder> encoder;
                if(iot::outputting(io))
                {
                    encoder = std::dynamic_pointer_cast<EmbeddingSimilarity::Encoder>(lib.encoder);
                }
                else
                {
                    encoder     = std::make_shared<EmbeddingSimilarity::Encoder>();
                    lib.encoder = encoder;
                }
                iot::mapRequired(io, "encoder", *encoder);

                std::shared_ptr<TensileLite::EmbeddingSimilarity::SolutionEmbeddings> embeddings;
                if(iot::outputting(io))
                {
                    embeddings = std::dynamic_pointer_cast<
                        TensileLite::EmbeddingSimilarity::SolutionEmbeddings>(lib.embeddings);
                }
                else
                {
                    embeddings
                        = std::make_shared<TensileLite::EmbeddingSimilarity::SolutionEmbeddings>();
                    lib.embeddings = embeddings;
                }
                iot::mapRequired(io, "solution_embeddings", *embeddings);

                // Checks
                if(embeddings->size() != lib.solutions.size())
                    throw std::runtime_error(
                        "ERROR: EmbeddingSimilarity library solution embeddings amount "
                        "does not match solution map size.");

                if(lib.solutions.size() == 0)
                    throw std::runtime_error(
                        "ERROR: EmbeddingSimilarity library solution embeddings amount equals 0");

                if(lib.encoder->network.proj_bias.size() != embeddings->embeddings[0][0].size())
                    throw std::runtime_error(
                        "ERROR: EmbeddingSimilarity library solution embeddings size "
                        "does not match the network output size.");

                if(lib.encoder->network.weights[0].size() != (91 * lib.encoder->network.bias[0].size()))
                {
                    throw std::runtime_error(
                        "ERROR: EmbeddingSimilarity network input size ("
                        + std::to_string((int) (lib.encoder->network.weights[0].size() / lib.encoder->network.bias[0].size()))
                        + ") does not match the input vector size (91)");
                }
            }

            const static bool flow = false;
        };

    } // namespace Serialization
} // namespace TensileLite
