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

#include <cstring>
#include <tuple>
#include <utility>

#include "DataTypes_Half.hpp"
#include "Fallback.hpp"
#include <array>
#include <map>
#include <memory>
#include <set>
#include <vector>

#ifdef __AVX2__
float avx_dot(int N, const float* __restrict__ A, const float* __restrict__ B);
#if defined(__AVX512F__) && defined(__AVX512BF16__)
float avx_dot_bf16(int N, const uint16_t* __restrict__ A, const uint16_t* __restrict__ B);
#else
float avx_dot_bf16(int N, const float* __restrict__ A, const uint16_t* __restrict__ B);
#endif
#endif

    static inline uint16_t float_to_bf16_rne(float x)
{
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    const uint32_t lsb           = (u >> 16) & 1u;
    const uint32_t rounding_bias = 0x7FFFu + lsb;
    u += rounding_bias;
    return static_cast<uint16_t>(u >> 16);
}

static inline float bf16_to_float(uint16_t b)
{
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float    out;
    std::memcpy(&out, &u, sizeof(out));
    return out;
}

namespace TensileLite
{
    /**
     * \ingroup Tensile
     * \defgroup EmbeddingSimilarity
     *
     * @brief EmbeddingSimilarity model
     *
     * Encoder used to estimate embedding values for problems in the
     * library. Used for EmbeddingSimilarityLibrary.
     *
     * See EmbeddingSimilarity.cpp
     */

    /**
     * \ingroup EmbeddingSimilarity
     */
    namespace EmbeddingSimilarity
    {

        using dtype = float;

        struct StandardScaler
        {
            void operator()(std::vector<dtype>& F) const;
            bool valid(bool verbose = false) const;

            std::vector<dtype> mean, scale;
        };

        struct Network
        {
            using Matrix      = std::vector<std::vector<dtype>>;
            using Vector      = std::vector<dtype>;
            using ForwardImpl = std::vector<dtype> (Network::*)(const std::vector<dtype>&) const;

            void               quantize();
            std::vector<dtype> operator()(const std::vector<dtype>& F) const;
            bool               valid(bool verbose) const;

            std::vector<dtype> forward_fp32_(const std::vector<dtype>& F) const;
            std::vector<dtype> forward_bf16_(const std::vector<dtype>& F) const;

            Matrix                             weights_;
            Matrix                             bias_;
            Vector                             proj_weights_;
            Vector                             proj_bias_;
            std::vector<std::vector<uint16_t>> weights_bf16_;
            std::vector<uint16_t>              proj_weights_bf16_;

            ForwardImpl forward_impl_ = &Network::forward_fp32_;
        };

        struct Encoder
        {
            Encoder() = default;

            std::vector<dtype> forward(std::vector<float>& probkey) const;

            bool valid(bool verbose = false) const;

            std::string description() const
            {
                return "Encoder";
            }

            StandardScaler scaler;
            Network        network;
        };

        struct SolutionEmbeddings
        {
            SolutionEmbeddings() = default;

            std::string description() const
            {
                return "SolutionEmbeddings";
            }

            std::vector<std::vector<float>>                 centroids;
            std::vector<std::vector<std::vector<float>>>    embeddings;
            std::vector<std::vector<uint16_t>>              centroids_bf16;
            std::vector<std::vector<std::vector<uint16_t>>> embeddings_bf16;
            std::vector<std::vector<int>>                   cluster_indices;
            std::size_t                                     size() const
            {
                std::set<int> unique_values;
                for(const auto& cluster : cluster_indices)
                {
                    unique_values.insert(cluster.begin(), cluster.end());
                }
                return unique_values.size();
            }
            void quantize();
        };

        struct HardwareConstants
        {
            HardwareConstants() = default;

            std::string description() const
            {
                return "HardwareConstants";
            }

            bool valid(bool verbose = false) const
            {
                bool rv = true;
                if(n_cu <= 0)
                {
                    if(verbose)
                        std::cout << "Invalid n_cu: " << n_cu << std::endl;
                    rv = false;
                }
                if(peak_flops <= 0.0f || mem_bw <= 0.0f)
                {
                    if(verbose)
                        std::cout << "Invalid peak_flops or mem_bw" << std::endl;
                    rv = false;
                }
                return rv;
            }

            int   n_cu       = 256;
            float peak_flops = 2.3e15f;
            float mem_bw     = 8e12f;
            float l1_size    = 32.0f * 1024.0f;
            float l2_size    = 4.0f * 1024.0f * 1024.0f;
            float l3_size    = 256.0f * 1024.0f * 1024.0f;
            float wave_size  = 64.0f;
            float dtype_size = 2.0f;
            float acc_size   = 4.0f;
        };

        template <int KeyId>
        struct FallbackFloatTag
        {
            using Type = float;
            static constexpr int Id = KeyId;
        };

        template <int KeyId>
        struct FallbackIntTag
        {
            using Type = int;
            static constexpr int Id = KeyId;
        };

        using FallbackMTag        = FallbackFloatTag<0>;
        using FallbackNTag        = FallbackFloatTag<1>;
        using FallbackKTag        = FallbackFloatTag<2>;
        using FallbackScoreTag    = FallbackFloatTag<3>;
        using FallbackCategoryTag = FallbackIntTag<0>;

        struct FallbackBaseRule
        {
            using CategoryRule = TensileLite::Fallback::Category<FallbackCategoryTag, int>;
            using MRule        = TensileLite::Fallback::Range<FallbackMTag, float>;
            using NRule        = TensileLite::Fallback::Range<FallbackNTag, float>;
            using KRule        = TensileLite::Fallback::Range<FallbackKTag, float>;
            using PreRuleSet   = TensileLite::Fallback::RuleSet<CategoryRule, MRule, NRule, KRule>;

            FallbackBaseRule()
                : FallbackBaseRule(0, {}, {}, {}, {})
            {
            }

            FallbackBaseRule(int                ruleId,
                             std::vector<float> mRanges,
                             std::vector<float> nRanges,
                             std::vector<float> kRanges,
                             std::vector<int>   categories)
                : rule_id_(ruleId)
                , m_ranges_(std::move(mRanges))
                , n_ranges_(std::move(nRanges))
                , k_ranges_(std::move(kRanges))
                , cats_(std::move(categories))
                , pre_rule_set_(CategoryRule(cats_),
                                MRule(MRule::fromPairs(m_ranges_)),
                                NRule(NRule::fromPairs(n_ranges_)),
                                KRule(KRule::fromPairs(k_ranges_)))
            {
            }

            std::string description() const
            {
                return "FallbackRule";
            }

            bool valid(bool verbose = false) const
            {
                bool rv = true;
                rv       = MRule::validPairs(m_ranges_, verbose) && rv;
                rv       = NRule::validPairs(n_ranges_, verbose) && rv;
                rv       = KRule::validPairs(k_ranges_, verbose) && rv;
                rv       = CategoryRule(cats_).valid(verbose) && rv;
                rv       = MRule(MRule::fromPairs(m_ranges_)).valid(verbose) && rv;
                rv       = NRule(NRule::fromPairs(n_ranges_)).valid(verbose) && rv;
                rv       = KRule(KRule::fromPairs(k_ranges_)).valid(verbose) && rv;
                return rv;
            }

            int ruleId() const
            {
                return rule_id_;
            }

            const std::vector<float>& mRanges() const
            {
                return m_ranges_;
            }

            const std::vector<float>& nRanges() const
            {
                return n_ranges_;
            }

            const std::vector<float>& kRanges() const
            {
                return k_ranges_;
            }

            const std::vector<int>& categories() const
            {
                return cats_;
            }

        protected:
            int                rule_id_ = 0;
            std::vector<float> m_ranges_;
            std::vector<float> n_ranges_;
            std::vector<float> k_ranges_;
            std::vector<int>   cats_;

            PreRuleSet pre_rule_set_;
        };

        struct FallbackRule : FallbackBaseRule
        {
            using FallbackBaseRule::FallbackBaseRule;

            FallbackRule() = default;

            FallbackRule(int                ruleId,
                         std::vector<float> mRanges,
                         std::vector<float> nRanges,
                         std::vector<float> kRanges,
                         std::vector<int>   categories)
                : FallbackBaseRule(ruleId,
                                   std::move(mRanges),
                                   std::move(nRanges),
                                   std::move(kRanges),
                                   std::move(categories))
            {
            }

            bool matches(float m, float n, float k, int cat) const
            {
                const auto context = TensileLite::Fallback::Context(
                    TensileLite::Fallback::bind<FallbackMTag>(m),
                    TensileLite::Fallback::bind<FallbackNTag>(n),
                    TensileLite::Fallback::bind<FallbackKTag>(k),
                    TensileLite::Fallback::bind<FallbackCategoryTag>(cat));

                return pre_rule_set_.matches(context);
            }
        };

        struct FallbackPostRule : FallbackBaseRule
        {
            using ScoreRule   = TensileLite::Fallback::Range<FallbackScoreTag, float>;
            using PostRuleSet = TensileLite::Fallback::RuleSet<CategoryRule,
                                                               MRule,
                                                               NRule,
                                                               KRule,
                                                               ScoreRule>;

            FallbackPostRule()
                : FallbackPostRule(0, {}, {}, {}, {}, {})
            {
            }

            FallbackPostRule(int                ruleId,
                             std::vector<float> mRanges,
                             std::vector<float> nRanges,
                             std::vector<float> kRanges,
                             std::vector<float> scoreRanges,
                             std::vector<int>   categories)
                : FallbackBaseRule(ruleId,
                                   std::move(mRanges),
                                   std::move(nRanges),
                                   std::move(kRanges),
                                   std::move(categories))
                , score_ranges_(std::move(scoreRanges))
                , post_rule_set_(CategoryRule(cats_),
                                 MRule(MRule::fromPairs(m_ranges_)),
                                 NRule(NRule::fromPairs(n_ranges_)),
                                 KRule(KRule::fromPairs(k_ranges_)),
                                 ScoreRule(ScoreRule::fromPairs(score_ranges_)))
            {
            }

            bool valid(bool verbose = false) const
            {
                bool rv = FallbackBaseRule::valid(verbose);
                rv      = ScoreRule::validPairs(score_ranges_, verbose) && rv;
                rv      = ScoreRule(ScoreRule::fromPairs(score_ranges_)).valid(verbose) && rv;
                return rv;
            }

            const std::vector<float>& scoreRanges() const
            {
                return score_ranges_;
            }

            bool matches(float m, float n, float k, int cat, float score) const
            {
                const auto context = TensileLite::Fallback::Context(
                    TensileLite::Fallback::bind<FallbackMTag>(m),
                    TensileLite::Fallback::bind<FallbackNTag>(n),
                    TensileLite::Fallback::bind<FallbackKTag>(k),
                    TensileLite::Fallback::bind<FallbackCategoryTag>(cat),
                    TensileLite::Fallback::bind<FallbackScoreTag>(score));

                return post_rule_set_.matches(context);
            }

        private:
            std::vector<float> score_ranges_;
            PostRuleSet        post_rule_set_;
        };

        struct FallbackRules
        {
            FallbackRules() = default;

            std::string description() const
            {
                return "FallbackRules";
            }

            bool valid(bool verbose = false) const
            {
                bool rv = true;
                if(all_cats.empty())
                {
                    if(verbose) std::cout << "FallbackRules: all_cats is empty" << std::endl;
                    rv = false;
                }

                for(const auto& rule : pre_model_features)
                {
                    rv = rule.valid(verbose) && rv;
                }

                for(const auto& rule : post_model_features)
                {
                    rv = rule.valid(verbose) && rv;
                }

                return rv;
            }

            bool isEmpty() const
            {
                return all_cats.empty() &&
                        pre_model_features.empty() &&
                        post_model_features.empty();
            }

            bool hasData() const
            {
                return !isEmpty();
            }

            bool matchesPreModel(float m, float n, float k, int cat, bool debug = false) const
            {
                return matches(pre_model_features, m, n, k, cat, debug);
            }

            bool matchesPostModel(float m, float n, float k, int cat, float score, bool debug = false) const
            {
                return matches(post_model_features, m, n, k, cat, score, debug);
            }

            std::string interval_semantics = "open_open"; // TODO
            std::string notes;
            std::vector<int> all_cats;                     
            std::vector<FallbackRule> pre_model_features;     // [m,n,k,cat]
            std::vector<FallbackPostRule> post_model_features; // [m,n,k,cat,score]

        private:
            static bool matches(const std::vector<FallbackRule>& rules,
                                float                            m,
                                float                            n,
                                float                            k,
                                int                              cat,
                                bool                             debug)
            {
                for(const auto& rule : rules)
                {
                    if(rule.matches(m, n, k, cat))
                    {
                        if(debug)
                        {
                            std::cout << "FALLBACK triggered by pre-model rule_id=" << rule.ruleId()
                                      << "\n";
                            std::cout << "RULE_INPUT: M=" << m << ", N=" << n << ", K=" << k
                                      << ", CAT=" << cat << "\n";
                        }
                        return true;
                    }
                }
                return false;
            }

            static bool matches(const std::vector<FallbackPostRule>& rules,
                                float                            m,
                                float                            n,
                                float                            k,
                                int                              cat,
                                float                            score,
                                bool                             debug)
            {
                for(const auto& rule : rules)
                {
                    if(rule.matches(m, n, k, cat, score))
                    {
                        if(debug)
                        {
                            std::cout << "FALLBACK triggered by post-model rule_id=" << rule.ruleId()
                                      << "\n";
                            std::cout << "RULE_INPUT: M=" << m << ", N=" << n << ", K=" << k
                                      << ", CAT=" << cat << ", SCORE=" << score << "\n";
                        }
                        return true;
                    }
                }
                return false;
            }
        };

    } // namespace EmbeddingSimilarity
} // namespace TensileLite