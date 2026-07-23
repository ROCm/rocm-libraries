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
                iot::mapRequired(io, "proj_weights", net.proj_weights_);
                iot::mapRequired(io, "proj_bias", net.proj_bias_);

                iot::mapRequired(io, "weights", net.weights_);
                iot::mapRequired(io, "bias", net.bias_);
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

        template <typename IO>
        struct MappingTraits<EmbeddingSimilarity::HardwareConstants, IO>
        {
            using HWConstants = EmbeddingSimilarity::HardwareConstants;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, HWConstants& hw)
            {
                iot::mapRequired(io, "n_cu", hw.n_cu);
                iot::mapRequired(io, "peak_flops", hw.peak_flops);
                iot::mapRequired(io, "mem_bw", hw.mem_bw);
                iot::mapRequired(io, "l1_size", hw.l1_size);
                iot::mapRequired(io, "l2_size", hw.l2_size);
                iot::mapRequired(io, "l3_size", hw.l3_size);
                iot::mapRequired(io, "wave_size", hw.wave_size);
                iot::mapRequired(io, "dtype_size", hw.dtype_size);
                iot::mapRequired(io, "acc_size", hw.acc_size);
            }

            const static bool flow = false;
        };


        template <typename IO>
        struct MappingTraits<EmbeddingSimilarity::FallbackRule, IO>
        {
            using PreRule = EmbeddingSimilarity::FallbackRule;
            using iot  = IOTraits<IO>;

            static void mapping(IO& io, PreRule& rule)
            {
                if(iot::outputting(io))
                {
                    int                ruleId      = rule.ruleId();
                    std::vector<float> mRanges     = rule.mRanges();
                    std::vector<float> nRanges     = rule.nRanges();
                    std::vector<float> kRanges     = rule.kRanges();
                    std::vector<int>   categories  = rule.categories();

                    iot::mapRequired(io, "rule_id", ruleId);
                    iot::mapRequired(io, "m", mRanges);
                    iot::mapRequired(io, "n", nRanges);
                    iot::mapRequired(io, "k", kRanges);
                    iot::mapRequired(io, "cats", categories);
                }
                else
                {
                    int                ruleId = 0;
                    std::vector<float> mRanges;
                    std::vector<float> nRanges;
                    std::vector<float> kRanges;
                    std::vector<int>   categories;

                    iot::mapRequired(io, "rule_id", ruleId);
                    iot::mapRequired(io, "m", mRanges);
                    iot::mapRequired(io, "n", nRanges);
                    iot::mapRequired(io, "k", kRanges);
                    iot::mapRequired(io, "cats", categories);

                    rule = PreRule(ruleId,
                                   std::move(mRanges),
                                   std::move(nRanges),
                                   std::move(kRanges),
                                   std::move(categories));
                }
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<EmbeddingSimilarity::FallbackPostRule, IO>
        {
            using PostRule = EmbeddingSimilarity::FallbackPostRule;
            using iot  = IOTraits<IO>;

            static void mapping(IO& io, PostRule& rule)
            {
                if(iot::outputting(io))
                {
                    int                ruleId      = rule.ruleId();
                    std::vector<float> mRanges     = rule.mRanges();
                    std::vector<float> nRanges     = rule.nRanges();
                    std::vector<float> kRanges     = rule.kRanges();
                    std::vector<float> scoreRanges = rule.scoreRanges();
                    std::vector<int>   categories  = rule.categories();

                    iot::mapRequired(io, "rule_id", ruleId);
                    iot::mapRequired(io, "m", mRanges);
                    iot::mapRequired(io, "n", nRanges);
                    iot::mapRequired(io, "k", kRanges);
                    iot::mapOptional(io, "score", scoreRanges);
                    iot::mapRequired(io, "cats", categories);
                }
                else
                {
                    int                ruleId = 0;
                    std::vector<float> mRanges;
                    std::vector<float> nRanges;
                    std::vector<float> kRanges;
                    std::vector<float> scoreRanges;
                    std::vector<int>   categories;

                    iot::mapRequired(io, "rule_id", ruleId);
                    iot::mapRequired(io, "m", mRanges);
                    iot::mapRequired(io, "n", nRanges);
                    iot::mapRequired(io, "k", kRanges);
                    iot::mapOptional(io, "score", scoreRanges);
                    iot::mapRequired(io, "cats", categories);

                    rule = PostRule(ruleId,
                                    std::move(mRanges),
                                    std::move(nRanges),
                                    std::move(kRanges),
                                    std::move(scoreRanges),
                                    std::move(categories));
                }
            }

            const static bool flow = false;
        };

       
        template <typename IO>
        struct MappingTraits<EmbeddingSimilarity::FallbackRules, IO>
        {
            using FallbackRules = EmbeddingSimilarity::FallbackRules;
            using iot           = IOTraits<IO>;

            static void mapping(IO& io, FallbackRules& fallback)
            {            
                iot::mapRequired(io, "interval_semantics", fallback.interval_semantics);
                iot::mapOptional(io, "notes", fallback.notes);
                iot::mapRequired(io, "all_cats", fallback.all_cats);
               
                iot::mapOptional(io, "pre_model_features", fallback.pre_model_features);
                iot::mapOptional(io, "post_model_features", fallback.post_model_features);
            
   
                for (size_t i = 0; i < fallback.pre_model_features.size(); ++i)
                {
                    const auto& rule = fallback.pre_model_features[i];
                }

                for (size_t i = 0; i < fallback.post_model_features.size(); ++i)
                {
                    const auto& rule = fallback.post_model_features[i];
                }

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
                    return;
                }

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

                auto encoder     = std::make_shared<EmbeddingSimilarity::Encoder>();
                lib.encoder      = encoder;
                iot::mapOptional(io, "encoder", *encoder);

                auto embeddings = std::make_shared<TensileLite::EmbeddingSimilarity::SolutionEmbeddings>();
                lib.embeddings  = embeddings;
                iot::mapOptional(io, "solution_embeddings", *embeddings);

                auto hw_constants = std::make_shared<EmbeddingSimilarity::HardwareConstants>();
                lib.hw_constants  = hw_constants;
                iot::mapOptional(io, "hardware_constants", *hw_constants);

                auto fallback_rules = std::make_shared<EmbeddingSimilarity::FallbackRules>();
                lib.fallback_rules  = fallback_rules;
                iot::mapOptional(io, "fallback", *fallback_rules);

                if(fallback_rules->hasData())
                {
                    if(!fallback_rules->valid())
                    {
                        throw std::runtime_error(
                            "ERROR: EmbeddingSimilarity fallback_rules are invalid.");
                    }
                }

                bool quantize = false;
                iot::mapOptional(io, "quantize", quantize);

                if(quantize)
                {
                    lib.quantize();
                }

                const bool model_loaded = !encoder->network.proj_bias_.empty()
                                          && !embeddings->embeddings.empty();

                if(!model_loaded)
                    return;

                if(!hw_constants->valid())
                    throw std::runtime_error(
                        "ERROR: EmbeddingSimilarity hardware constants are invalid.");

                if(embeddings->size() != lib.solutions.size())
                    throw std::runtime_error(
                        "ERROR: EmbeddingSimilarity library solution embeddings amount "
                        "does not match solution map size.");

                if(lib.solutions.size() == 0)
                    throw std::runtime_error(
                        "ERROR: EmbeddingSimilarity library solution embeddings amount equals 0");

                if(encoder->network.proj_bias_.size() != embeddings->embeddings[0][0].size())
                    throw std::runtime_error(
                        "ERROR: EmbeddingSimilarity library solution embeddings size "
                        "does not match the network output size.");
            }

            const static bool flow = false;
        };

    } // namespace Serialization
} // namespace TensileLite
