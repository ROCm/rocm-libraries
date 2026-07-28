/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iostream>
#include <queue>
#include <set>
#include <vector>

#include <Tensile/Debug.hpp>
#include <Tensile/EmbeddingSimilarity.hpp>
#include <Tensile/MLFeatures.hpp>
#include <Tensile/ProblemKey.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Utils.hpp>


namespace TensileLite
{

    /**
     * \ingroup SolutionLibrary
     *
     * Uses a EmbeddingSimilarity network to create embeddings for rank solutions for a given size.
     */

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct EmbeddingSimilarityLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        using Encoder            = EmbeddingSimilarity::Encoder;
        using SolutionEmbeddings = TensileLite::EmbeddingSimilarity::SolutionEmbeddings;
        using HardwareConstants  = EmbeddingSimilarity::HardwareConstants;
        using FallbackRules      = EmbeddingSimilarity::FallbackRules;


        std::map<int, std::shared_ptr<MySolution>> solutionmap;
        std::vector<std::shared_ptr<MySolution>>   solutions;
        std::shared_ptr<Encoder>                   encoder;
        std::shared_ptr<SolutionEmbeddings>        embeddings;
        std::shared_ptr<HardwareConstants>         hw_constants;
        std::shared_ptr<FallbackRules>             fallback_rules;
        bool                                       is_quantized_ = false;

        void quantize()
        {
            if(encoder == nullptr || embeddings == nullptr)
            {
                return;
            }
            encoder->network.quantize();
            embeddings->quantize();
            is_quantized_ = true;
        }

        static std::string Type()
        {
            return "EmbeddingSimilarity";
        }
        virtual std::string type() const override
        {
            return Type();
        }
        virtual std::string description() const override
        {
            if(encoder == nullptr)
                return concatenate(type(), ", EmbeddingSimilarity: nullptr");
            else
                return concatenate(type(), ": ", encoder->description());
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

            SolutionVector<MySolution> solutions = findTopSolutions(problem, hardware, 1);
            return solutions.size() ? solutions[0] : nullptr;
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            SolutionSet<MySolution> rv;
            for(auto const& row : solutionmap)
                rv.insert(row.second);

            return rv;
        }

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            bool debug = Debug::Instance().printPropertyEvaluation();
            float batch_count = problem.batchSize(0);
            
            if(batch_count > 1) // TODO batch_count !=  1 not supported
                return {};

            float m = problem.freeSizeA(0);
            float n = problem.freeSizeB(0);
            float k = problem.boundSize(0);
            int gemm_category = -1;

            if (fallback_rules && fallback_rules->hasData())
            {
                gemm_category = classifyGEMM(m, n, k, batch_count); 
                if (fallback_rules->matchesPreModel(m, n, k, gemm_category, debug))
                {
                    return {};
                }
            }
            
            std::vector<float> gemm_embedding = computeGEMMEmbeddings(problem);

            std::vector<int> centroid_indexes(embeddings->centroids.size());
            std::iota(centroid_indexes.begin(), centroid_indexes.end(), 0);
            std::vector<float> centroid_similarities(embeddings->centroids.size());

            int max_sim_idx = 0;
            if (embeddings->centroids.size() > 1)
            {
                if (is_quantized_)
                    inner_product_bf16(embeddings->centroids_bf16, gemm_embedding, centroid_similarities);
                else
                    inner_product(embeddings->centroids, gemm_embedding, centroid_similarities);
            }
    
            std::vector<std::pair<float, std::shared_ptr<MySolution>>> rankedSolutions;
            rankedSolutions.reserve(numSolutions);

            int remSolutions = numSolutions;
            remSolutions     = check_cluster(
                numSolutions, gemm_embedding, max_sim_idx, problem, hardware, rankedSolutions);

            if(remSolutions > 0)
            {
                auto cent_begin = centroid_indexes.begin(), cent_end = centroid_indexes.end();

                size_t currentIndex = 1;
                while(remSolutions > 0 && currentIndex < centroid_indexes.size())
                {
                    std::partial_sort(cent_begin + currentIndex - 1,
                                      cent_begin + currentIndex + 1,
                                      cent_end,
                                      [&centroid_similarities](int i0, int i1) {
                                          return centroid_similarities[i0]
                                                 > centroid_similarities[i1];
                                      });
                    auto cidx    = centroid_indexes[currentIndex];
                    remSolutions = check_cluster(
                        remSolutions, gemm_embedding, cidx, problem, hardware, rankedSolutions);
                    ++currentIndex;
                }
            }

            int numToSort = std::min(numSolutions, int(rankedSolutions.size()));
            if(numToSort > 1)
            {
                std::partial_sort(rankedSolutions.begin(),
                                  rankedSolutions.begin() + numToSort,
                                  rankedSolutions.end(),
                                  [](const std::pair<float, std::shared_ptr<MySolution>>& a,
                                     const std::pair<float, std::shared_ptr<MySolution>>& b) {
                                      return a.first > b.first;
                                  });
            }

            // Check post-model fallback rules with top score
            if (gemm_category != -1 && !rankedSolutions.empty())
            {
                float top_score = rankedSolutions[0].first; 
                if (fallback_rules->matchesPostModel(m, n, k, gemm_category, top_score, debug))
                {
                    return {}; 
                }
            }

            SolutionVector<MySolution> rv;
            rv.reserve(numToSort);
            std::transform(rankedSolutions.begin(),
                           rankedSolutions.begin() + numToSort,
                           std::back_inserter(rv),
                           [](std::pair<float, std::shared_ptr<MySolution>>& p) {
                               return std::move(p.second);
                           });
            return rv;
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            SolutionSet<MySolution> rv;
            for(auto const& row : solutionmap)
                rv.insert(row.second);

            return rv;
        }

    protected:
        static constexpr float EPSILON = 1e-8f;

        int classifyGEMM(float m, float n, float k, float batch_count) const
        {
            // Categories checked in order (first match wins)
            struct CategoryRule {
                int cat;
                float m_min, m_max;
                float n_min, n_max;
                float k_min, k_max;
                float b_min, b_max;
            };

            const float INF = std::numeric_limits<float>::infinity();
            const CategoryRule categories[] = {
                {1, 2.0f, 1024.0f, 2.0f, 1024.0f, 2.0f, 1024.0f, 1.0f, 1.0f}, // 1. Small GEMMs
                {3, 4094.0f, 8192.0f, 4096.0f, 8192.0f, 4096.0f, 8192.0f, 1.0f, 1.0f}, // 3. Large GEMMs (checked before 2. Medium)
                {2, 2.0f, 8192.0f, 2.0f, 8192.0f, 2.0f, 8192.0f, 1.0f, 1.0f}, // 2. Medium GEMMs                
                {5, 8193.0f, INF, 2.0f, 128.0f, 2.0f, 128.0f, 1.0f, 1.0f}, // 5. Large M, very small N and K
                {4, 8193.0f, INF, 2.0f, 8192.0f, 2.0f, 8192.0f, 1.0f, 1.0f}, // 4. Large M, smaller N and K
                {7, 2.0f, 128.0f, 8193.0f, INF, 2.0f, 128.0f, 1.0f, 1.0f}, // 7. Large N, very small M and K
                {6, 2.0f, 8192.0f, 8193.0f, INF, 2.0f, 8192.0f, 1.0f, 1.0f}, // 6. Large N, smaller M and K
                {9, 2.0f, 128.0f, 2.0f, 128.0f, 8193.0f, INF, 1.0f, 1.0f}, // 9. Large K, very small M and N
                {8, 2.0f, 8192.0f, 2.0f, 8192.0f, 8193.0f, INF, 1.0f, 1.0f}, // 8. Large K, smaller M and N
                {10, 8193.0f, INF, 8193.0f, INF, 2.0f, 8192.0f, 1.0f, 1.0f}, // 10. Large M and N
                {11, 2.0f, 8192.0f, 8193.0f, INF, 8193.0f, INF, 1.0f, 1.0f}, // 11. Large N and K
                {12, 8193.0f, INF, 2.0f, 8192.0f, 8193.0f, INF, 1.0f, 1.0f}, // 12. Large M and K
                {13, 8193.0f, INF, 8193.0f, INF, 8193.0f, INF, 1.0f, 1.0f}, // 13. Very Large GEMMs
                {14, 1.0f, 1.0f, 1.0f, INF, 1.0f, INF, 1.0f, 1.0f}, // 14. M = 1
                {15, 1.0f, INF, 1.0f, 1.0f, 1.0f, INF, 1.0f, 1.0f}, // 15. N = 1                
                {16, 1.0f, INF, 1.0f, INF, 1.0f, 1.0f, 1.0f, 1.0f}, // 16. K = 1
                {17, 1.0f, INF, 1.0f, INF, 1.0f, INF, 2.0f, 128.0f}, // 17. Small batch
                {18, 1.0f, INF, 1.0f, INF, 1.0f, INF, 129.0f, 1024.0f}, // 18. Medium batch
                {19, 1.0f, INF, 1.0f, INF, 1.0f, INF, 1025.0f, 8192.0f}, // 19. Large batch
                {20, 1.0f, INF, 1.0f, INF, 1.0f, INF, 8193.0f, INF} // 20. Very large batch
            };
            for (const auto& rule : categories)
            {
                if (m >= rule.m_min && m <= rule.m_max &&
                    n >= rule.n_min && n <= rule.n_max &&
                    k >= rule.k_min && k <= rule.k_max &&
                    batch_count >= rule.b_min && batch_count <= rule.b_max)
                {
                    return rule.cat;
                }
            }
            return -1;  
        }

        /* Helper functions */
        inline float bucket_dimension(float x) const
        {
            // Buckets: [0,16,32,64,128,192,256,512,1024,2048,4096,8192,inf]
            if(x <= 16.0f)
                return 0.0f;
            if(x <= 32.0f)
                return 1.0f;
            if(x <= 64.0f)
                return 2.0f;
            if(x <= 128.0f)
                return 3.0f;
            if(x <= 192.0f)
                return 4.0f;
            if(x <= 256.0f)
                return 5.0f;
            if(x <= 512.0f)
                return 6.0f;
            if(x <= 1024.0f)
                return 7.0f;
            if(x <= 2048.0f)
                return 8.0f;
            if(x <= 4096.0f)
                return 9.0f;
            if(x <= 8192.0f)
                return 10.0f;
            return 11.0f;
        }

        inline float bucket_aspect_ratio(float aspect) const
        {
            // Buckets: [0, 0.5, 1.5, 2.5, inf]
            if(aspect <= 0.5f)
                return 0.0f;
            if(aspect <= 1.5f)
                return 1.0f;
            if(aspect <= 2.5f)
                return 2.0f;
            return 3.0f;
        }

        inline float compute_wastage(float m, float n, float tile) const
        {
            float tiles_m = std::ceil(m / tile);
            float tiles_n = std::ceil(n / tile);
            float work    = tiles_m * tile * tiles_n * tile;
            return (work - m * n) / work;
        }

        inline float best_fit_tile(float x) const
        {
            float tiles[]   = {128.0f, 192.0f, 224.0f, 256.0f};
            float best      = tiles[0];
            float min_waste = std::ceil(x / tiles[0]) * tiles[0] - x;

            for(int i = 1; i < 4; i++)
            {
                float waste = std::ceil(x / tiles[i]) * tiles[i] - x;
                if(waste < min_waste)
                {
                    min_waste = waste;
                    best      = tiles[i];
                }
            }
            return best;
        }

        /* Main feature computation function */
        std::vector<float> computeGEMMEmbeddings(const MyProblem& problem) const
        {
            // Extract basic problem dimensions
            float m = problem.freeSizeA(0);
            float n = problem.freeSizeB(0);
            float k = problem.boundSize(0);

            bool transA = problem.transA();
            bool transB = problem.transB();

            bool is_NT = (!transA && transB); // N-T

            float lda = problem.a().strides()[1];
            float ldb = problem.b().strides()[1];
            float ldc = problem.c().strides()[1];
            float ldd = problem.d().strides()[1];

            float batch_count = problem.batchSize(0);

            float stride_a = transA ? lda * m : lda * k;
            float stride_b = transB ? ldb * k : ldb * n;
            float stride_c = ldc * n;
            float stride_d = ldd * n;

            // Basic computations
            float flops                = 2.0f * m * n * k * batch_count;
            float bytes_moved          = (m * k + k * n + m * n) * hw_constants->dtype_size;
            float arithmetic_intensity = flops / bytes_moved;
            float output_size          = m * n;

            // Memory and compute characteristics
            float balance_ai = hw_constants->peak_flops / hw_constants->mem_bw;
            float ai_vs_balance = arithmetic_intensity / balance_ai;
            float memory_peak = hw_constants->mem_bw * arithmetic_intensity;
            float compute_peak = hw_constants->peak_flops;
            float is_compute_bound = (memory_peak > compute_peak) ? 1.0f : 0.0f;
            float memory_headroom = memory_peak / compute_peak;
            float memory_headroom_clipped = std::min(std::max(memory_headroom, 0.0f), 2.0f);

            // Cache pressure
            float ws_l1_ratio = bytes_moved / hw_constants->l1_size;
            float ws_l2_ratio = bytes_moved / hw_constants->l2_size;
            float ws_l3_ratio = bytes_moved / hw_constants->l3_size;

            float fits_in_l1 = (bytes_moved <= hw_constants->l1_size) ? 1.0f : 0.0f;
            float fits_in_l2 = (bytes_moved <= hw_constants->l2_size) ? 1.0f : 0.0f;
            float fits_in_l3 = (bytes_moved <= hw_constants->l3_size) ? 1.0f : 0.0f;


            constexpr float SWEET_SPOT_LOWER = 0.5f;
            float in_l2_sweet_spot = (bytes_moved > SWEET_SPOT_LOWER * hw_constants->l2_size && bytes_moved <= hw_constants->l2_size) ? 1.0f : 0.0f;
            float in_l3_sweet_spot = (bytes_moved > SWEET_SPOT_LOWER * hw_constants->l3_size && bytes_moved <= hw_constants->l3_size) ? 1.0f : 0.0f;

            float fits_in_l3_not_l2 = (bytes_moved <= hw_constants->l3_size && bytes_moved > hw_constants->l2_size) ? 1.0f : 0.0f;

            // K-dimension pressure
            float k_underutilizes_wave = (k < hw_constants->wave_size) ? 1.0f : 0.0f;
            float k_saturates_waves    = (k >= 4.0f * hw_constants->wave_size) ? 1.0f : 0.0f;

            // Accumulator pressure
            float acc_bytes = hw_constants->acc_size * output_size;

            // Wave alignment
            float m_wave_misalignment
                = std::fmod(m, hw_constants->wave_size) / hw_constants->wave_size;
            float n_wave_misalignment
                = std::fmod(n, hw_constants->wave_size) / hw_constants->wave_size;
            float wave_misalignment_total = m_wave_misalignment + n_wave_misalignment;
            float m_wave_aligned
                = (static_cast<int>(m) % static_cast<int>(hw_constants->wave_size) == 0) ? 1.0f
                                                                                         : 0.0f;
            float n_wave_aligned
                = (static_cast<int>(n) % static_cast<int>(hw_constants->wave_size) == 0) ? 1.0f
                                                                                         : 0.0f;
            float both_wave_aligned = m_wave_aligned * n_wave_aligned;

            // Stream-K hints
            float streamk_favorable = ((k > 1024.0f) && (output_size < 4096.0f)) ? 1.0f : 0.0f;

            // Reuse factors
            float low_reuse  = ((n < 64.0f) || (m < 64.0f)) ? 1.0f : 0.0f;
            float high_reuse = ((n >= 256.0f) && (m >= 256.0f)) ? 1.0f : 0.0f;

            // Tile preferences
            float prefer_small_tile = (ws_l1_ratio > 2.0f) ? 1.0f : 0.0f;

            // Aspect ratios
            float aspect_m_n     = m / (n + EPSILON);
            float sqrt_aspect_nm = std::sqrt(n / (m + EPSILON));

            // Occupancy
            float est_tiles = std::ceil(m / 256.0f) * std::ceil(n / 256.0f);
            float is_saturating
                = (est_tiles >= static_cast<float>(hw_constants->n_cu)) ? 1.0f : 0.0f;
            float est_waves = est_tiles / static_cast<float>(hw_constants->n_cu);

            // Tile counts
            float tiles_64x48 = std::ceil(m / 64.0f) * std::ceil(n / 48.0f);
            float tiles_64x96 = std::ceil(m / 64.0f) * std::ceil(n / 96.0f);
            float tiles_128 = std::ceil(m / 128.0f) * std::ceil(n / 128.0f);
            float tiles_192 = std::ceil(m / 192.0f) * std::ceil(n / 192.0f);
            float tiles_224 = std::ceil(m / 224.0f) * std::ceil(n / 224.0f);

            // Wastage
            float wastage_32  = compute_wastage(m, n, 32.0f);
            float wastage_64  = compute_wastage(m, n, 64.0f);
            float wastage_128 = compute_wastage(m, n, 128.0f);
            float wastage_192 = compute_wastage(m, n, 192.0f);
            float wastage_224 = compute_wastage(m, n, 224.0f);
            float wastage_256 = compute_wastage(m, n, 256.0f);

            // Best fit
            float best_fit_n = best_fit_tile(n);

            // Edge case features
            float is_tiny_m = (m <= 32.0f) ? 1.0f : 0.0f;
            float is_tiny_n = (n <= 32.0f) ? 1.0f : 0.0f;
            float is_small_m = ((m > 32.0f) && (m <= 128.0f)) ? 1.0f : 0.0f;
            float is_small_n = ((n > 32.0f) && (n <= 128.0f)) ? 1.0f : 0.0f;
            float is_gemv_n = (n == 1.0f) ? 1.0f : 0.0f;
            float is_all_tiny = ((m <= 64.0f) && (n <= 64.0f) && (k <= 64.0f)) ? 1.0f : 0.0f;

            // K-dimension features
            float k_ultra_tiny    = (k <= 8.0f) ? 1.0f : 0.0f;
            float is_tiny_k       = (k <= 16.0f) ? 1.0f : 0.0f;
            float is_very_small_k = (k <= 64.0f) ? 1.0f : 0.0f;
            float is_small_k      = (k < 128.0f) ? 1.0f : 0.0f;
            float k_small_problem = ((k <= 128.0f) && (m < 4096.0f) && (n < 4096.0f)) ? 1.0f : 0.0f;
            float is_large_k = (k > 4096.0f) ? 1.0f : 0.0f;

            // General features
            float n_small_misaligned
                = ((n < 300.0f) && (static_cast<int>(n) % 16 != 0)) ? 1.0f : 0.0f;
            float k_small_misaligned
                = ((k < 300.0f) && (static_cast<int>(k) % 16 != 0)) ? 1.0f : 0.0f;
            float n_small_wastage_ratio = (n < 300.0f) ? (std::fmod(n, 16.0f) / 16.0f) : 0.0f;

            float extreme_aspect_ratio = ((n > 3.0f * m) || (m > 3.0f * n)) ? 1.0f : 0.0f;
            float n_vector = (n <= 2.0f) ? 1.0f : 0.0f;
            float very_extreme_aspect = ((m > 10.0f * n) || (n > 10.0f * m)) ? 1.0f : 0.0f;

            // Output size features
            float is_tiny_output       = (output_size < 1000.0f) ? 1.0f : 0.0f;
            float is_very_tiny_output  = (output_size < 100.0f) ? 1.0f : 0.0f;
            float is_ultra_tiny_output = (output_size < 50.0f) ? 1.0f : 0.0f;

            // Parallelization
            float insufficient_parallelism
                = (output_size < static_cast<float>(hw_constants->n_cu)) ? 1.0f : 0.0f;
            float severe_underutilization
                = (output_size < static_cast<float>(hw_constants->n_cu) / 2.0f) ? 1.0f : 0.0f;

            // K reuse
            float k_reuse_per_output = k / (output_size + 1.0f);

            // K memory
            float k_memory_bytes = k * hw_constants->dtype_size * (m + n);
            float k_memory_vs_l3 = k_memory_bytes / hw_constants->l3_size;

            // Dimension dominance
            float max_dim = std::max({m, n, k});
            float k_is_max_dim = (k == max_dim) ? 1.0f : 0.0f;
            float k_dominates_both = ((k > 10.0f * m) && (k > 10.0f * n)) ? 1.0f : 0.0f;

            // NT-specific features (if IS_NT is true)
            float k_dominates_n_10x = 0.0f, k_dominates_n_100x = 0.0f, k_dominates_n_1000x = 0.0f;
            float m_dominates_n_10x = 0.0f;
            float k_dominates_m_10x = 0.0f, k_dominates_m_1000x = 0.0f;
            float n_dominates_m_10x = 0.0f, n_dominates_m_100x = 0.0f;
            float n_dominates_k_10x = 0.0f, n_dominates_k_100x = 0.0f;
            float m_dominates_k_10x = 0.0f, m_dominates_k_100x = 0.0f;

            float min_dim                     = std::min({m, n, k});
            float extreme_dimension_ratio_10x = 0.0f, extreme_dimension_ratio_100x = 0.0f;

            float m_ultra_tiny = 0.0f, n_ultra_tiny = 0.0f, k_ultra_tiny_v2 = 0.0f;
            float any_dim_ultra_tiny = 0.0f, multiple_dims_tiny = 0.0f;

            float large_output_small_k = 0.0f;

            float extreme_ratio_and_tiny_dim = 0.0f, k_dominates_and_small_output = 0.0f;

            float n_div_k_ratio = 0.0f, m_div_k_ratio = 0.0f;
            float n_div_k_very_small = 0.0f, m_div_k_very_small = 0.0f;
            float n_div_k_ultra_small = 0.0f;

            float likely_needs_small_tile = 0.0f;

            float work_elements = m * n * k;
            float is_micro_gemm = 0.0f, is_nano_gemm = 0.0f;

            float m_not_vec4_aligned = 0.0f, n_not_vec4_aligned = 0.0f, k_not_vec4_aligned = 0.0f;
            float m_not_vec8_aligned = 0.0f, n_not_vec8_aligned = 0.0f;

            float pathological_case_type1 = 0.0f;
            float problem_severity_count = 0.0f, multiple_problems = 0.0f;

            if (is_NT) {

                // Aspect ratio patterns
                k_dominates_n_10x   = (k / (n + 1.0f) > 10.0f) ? 1.0f : 0.0f;
                k_dominates_n_100x  = (k / (n + 1.0f) > 100.0f) ? 1.0f : 0.0f;
                k_dominates_n_1000x = (k / (n + 1.0f) > 1000.0f) ? 1.0f : 0.0f;

                m_dominates_n_10x = (m / (n + 1.0f) > 10.0f) ? 1.0f : 0.0f;

                k_dominates_m_10x = (k / (m + 1.0f) > 10.0f) ? 1.0f : 0.0f;
                k_dominates_m_1000x = (k / (m + 1.0f) > 1000.0f) ? 1.0f : 0.0f;

                n_dominates_m_10x  = (n / (m + 1.0f) > 10.0f) ? 1.0f : 0.0f;
                n_dominates_m_100x = (n / (m + 1.0f) > 100.0f) ? 1.0f : 0.0f;

                n_dominates_k_10x  = (n / (k + 1.0f) > 10.0f) ? 1.0f : 0.0f;
                n_dominates_k_100x = (n / (k + 1.0f) > 100.0f) ? 1.0f : 0.0f;

                m_dominates_k_10x  = (m / (k + 1.0f) > 10.0f) ? 1.0f : 0.0f;
                m_dominates_k_100x = (m / (k + 1.0f) > 100.0f) ? 1.0f : 0.0f;

                float extreme_dim_ratio      = max_dim / (min_dim + 1.0f);
                extreme_dimension_ratio_10x  = (extreme_dim_ratio > 10.0f) ? 1.0f : 0.0f;
                extreme_dimension_ratio_100x = (extreme_dim_ratio > 100.0f) ? 1.0f : 0.0f;

                // Ultra tiny dimensions
                m_ultra_tiny    = (m <= 10.0f) ? 1.0f : 0.0f;
                n_ultra_tiny    = (n <= 10.0f) ? 1.0f : 0.0f;
                k_ultra_tiny_v2 = (k <= 10.0f) ? 1.0f : 0.0f;

                any_dim_ultra_tiny = ((m <= 10.0f) || (n <= 10.0f) || (k <= 10.0f)) ? 1.0f : 0.0f;

                int tiny_count = (m <= 32.0f ? 1 : 0) + (n <= 32.0f ? 1 : 0) + (k <= 32.0f ? 1 : 0);
                multiple_dims_tiny = (tiny_count >= 2) ? 1.0f : 0.0f;

                // Problematic configurations
                large_output_small_k = ((output_size > 1000000.0f) && (k < 256.0f)) ? 1.0f : 0.0f;

                // Combined problematic patterns
                extreme_ratio_and_tiny_dim = extreme_dimension_ratio_10x * any_dim_ultra_tiny;
                k_dominates_and_small_output
                    = k_dominates_both * ((output_size < 10000.0f) ? 1.0f : 0.0f);

                // Specific ratios
                n_div_k_ratio = n / (k + 1.0f);
                m_div_k_ratio = m / (k + 1.0f);

                n_div_k_very_small = (n_div_k_ratio < 0.1f) ? 1.0f : 0.0f;
                m_div_k_very_small = (m_div_k_ratio < 0.1f) ? 1.0f : 0.0f;
                n_div_k_ultra_small = (n_div_k_ratio < 0.01f) ? 1.0f : 0.0f;

                // Tile needs
                float est_tile_m_16     = (m < 128.0f) ? 1.0f : 0.0f;
                float est_tile_n_16     = (n < 128.0f) ? 1.0f : 0.0f;
                likely_needs_small_tile = (est_tile_m_16 + est_tile_n_16 >= 1.0f) ? 1.0f : 0.0f;
            

                // Work elements
                is_micro_gemm = (work_elements < 100000.0f) ? 1.0f : 0.0f;
                is_nano_gemm  = (work_elements < 10000.0f) ? 1.0f : 0.0f;

                // Vectorization alignment
                m_not_vec4_aligned = (static_cast<int>(m) % 4 != 0) ? 1.0f : 0.0f;
                n_not_vec4_aligned = (static_cast<int>(n) % 4 != 0) ? 1.0f : 0.0f;
                k_not_vec4_aligned = (static_cast<int>(k) % 4 != 0) ? 1.0f : 0.0f;

                m_not_vec8_aligned = (static_cast<int>(m) % 8 != 0) ? 1.0f : 0.0f;
                n_not_vec8_aligned = (static_cast<int>(n) % 8 != 0) ? 1.0f : 0.0f;

                // Pathological cases
                pathological_case_type1 = ((n / (m + 1.0f) > 50.0f) && (k < 100.0f)) ? 1.0f : 0.0f;
 
                // Problem severity
                problem_severity_count = extreme_dimension_ratio_10x + any_dim_ultra_tiny + k_dominates_both;
                multiple_problems = (problem_severity_count >= 2.0f) ? 1.0f : 0.0f;
            }

            // Build feature vector (matching Python order exactly)
            std::vector<float> features;
            features.reserve(400); // Pre-allocate for efficiency

            // Log-transformed inputs (order matching Python after preprocessing)
            features.push_back(std::log1p(m)); 
            features.push_back(std::log1p(n));
            features.push_back(std::log1p(k));
            features.push_back(std::log1p(lda));
            features.push_back(std::log1p(stride_a));
            features.push_back(std::log1p(ldb));
            features.push_back(std::log1p(stride_b));
            features.push_back(std::log1p(ldc));
            features.push_back(std::log1p(stride_c));
            features.push_back(std::log1p(ldd));
            features.push_back(std::log1p(stride_d));
            features.push_back(std::log1p(batch_count));

            // Computed features
            features.push_back(std::log1p(flops)); // log_flops
            features.push_back(std::log1p(bytes_moved)); // log_bytes
            features.push_back(arithmetic_intensity);
            features.push_back(std::log1p(arithmetic_intensity)); // log_ai

            // Roofline model
            features.push_back(is_compute_bound); 
            features.push_back(ai_vs_balance);
            features.push_back(std::log1p(ai_vs_balance));  // log_ai_vs_balance
            features.push_back(memory_headroom_clipped);

            // Cache pressure
            features.push_back(std::log1p(ws_l1_ratio)); // log_ws_l1_ratio
            features.push_back(fits_in_l1);
            features.push_back(std::log1p(ws_l2_ratio)); // log_ws_l2_ratio
            features.push_back(fits_in_l2);
            features.push_back(std::log1p(ws_l3_ratio)); // log_ws_l3_ratio
            features.push_back(fits_in_l3);
            features.push_back(in_l2_sweet_spot);
            features.push_back(in_l3_sweet_spot);
            features.push_back(fits_in_l3_not_l2);

            // K-dimension pressure
            features.push_back(std::log1p((k * hw_constants->dtype_size)
                                          / hw_constants->l1_size)); // log_k_l1_pressure
            features.push_back(std::log1p(k / hw_constants->wave_size)); // log_k_parallelism
            features.push_back(k_underutilizes_wave);
            features.push_back(k_saturates_waves);

            // Bandwidth pressure
            features.push_back(
                std::log1p(bytes_moved / hw_constants->mem_bw)); // log_bandwidth_pressure

            // Accumulator pressure
            features.push_back(std::log1p(acc_bytes)); // log_acc_bytes
            features.push_back(std::log1p(acc_bytes / hw_constants->l2_size)); // log_acc_pressure
            features.push_back(
                std::log1p(acc_bytes / hw_constants->l3_size)); // log_acc_pressure_l3

            // Wave alignment
            features.push_back(m_wave_misalignment);
            features.push_back(n_wave_misalignment);
            features.push_back(wave_misalignment_total);
            features.push_back(m_wave_aligned);
            features.push_back(n_wave_aligned);
            features.push_back(both_wave_aligned);

            // Stream-K hints
            features.push_back(std::log1p(k / (output_size + EPSILON))); // log_k_vs_mn
            features.push_back(std::log1p(k / (m + n + EPSILON))); // log_streamk_imbalance
            features.push_back(streamk_favorable);

            // Reuse factors
            features.push_back(low_reuse);
            features.push_back(high_reuse);

            // Tile preferences
            features.push_back(prefer_small_tile);

            // Problem size buckets (categorical)
            features.push_back(bucket_dimension(m)); // m_bucket
            features.push_back(bucket_dimension(n)); // n_bucket
            features.push_back(bucket_dimension(k)); // k_bucket

            // Aspect ratios
            features.push_back(sqrt_aspect_nm);
            features.push_back(std::log1p(n / (m + EPSILON))); // log_n_to_m_ratio
            features.push_back(std::log1p(aspect_m_n)); // log_aspect_m_n
            features.push_back(std::log1p(m / (k + EPSILON))); // log_aspect_m_k
            features.push_back(std::log1p(n / (k + EPSILON))); // log_aspect_n_k
            features.push_back(bucket_aspect_ratio(aspect_m_n)); // shape_category

            // Memory access patterns
            features.push_back(std::max(1.0f, std::log1p(ldc / (n + EPSILON)))); // ldc_efficiency

            // Tile alignment (M)
            features.push_back(static_cast<float>(static_cast<int>(m) % 128 == 0));
            features.push_back(static_cast<float>(static_cast<int>(m) % 160 == 0));
            features.push_back(static_cast<float>(static_cast<int>(m) % 192 == 0));
            features.push_back(static_cast<float>(static_cast<int>(m) % 224 == 0));
            features.push_back(static_cast<float>(static_cast<int>(m) % 256 == 0));

            // Tile alignment (N)
            features.push_back(static_cast<float>(static_cast<int>(n) % 128 == 0));
            features.push_back(static_cast<float>(static_cast<int>(n) % 160 == 0));
            features.push_back(static_cast<float>(static_cast<int>(n) % 192 == 0));
            features.push_back(static_cast<float>(static_cast<int>(n) % 224 == 0));
            features.push_back(static_cast<float>(static_cast<int>(n) % 256 == 0));

            // Tile alignment (K)
            features.push_back(static_cast<float>(static_cast<int>(k) % 32 == 0));
            features.push_back(static_cast<float>(static_cast<int>(k) % 128 == 0));

            // Size ratios
            features.push_back(std::log1p(n / 128.0f)); // n_div_tile128
            features.push_back(std::log1p(n / 256.0f)); // n_div_tile256

            // Problem scale
            features.push_back((m >= 8192.0f || n >= 8192.0f || k >= 8192.0f) ? 1.0f
                                                                              : 0.0f); // is_large

            // Shape flags
            features.push_back((m > n) ? 1.0f : 0.0f);  // is_tall
            features.push_back((n > m) ? 1.0f : 0.0f);  // is_wide
            features.push_back(((m > 4.0f * n) && (m > 4.0f * k)) ? 1.0f : 0.0f);  // is_tall_skinny
            features.push_back(((n > 4.0f * m) && (n > 4.0f * k)) ? 1.0f : 0.0f);  // is_short_wide
            features.push_back(((k > 4.0f * m) && (k > 4.0f * n)) ? 1.0f : 0.0f);  // is_deep_k

            // K-dimension features
            features.push_back(k_ultra_tiny);
            features.push_back(is_tiny_k);
            features.push_back(is_very_small_k);
            features.push_back(is_small_k);
            features.push_back(k_small_problem);
            features.push_back(is_large_k);

            features.push_back(std::log1p(k / 32.0f)); // k_div_32
            features.push_back(std::log1p(k / 64.0f)); // k_div_64

            // Occupancy proxy
            features.push_back(std::log1p(est_tiles)); // log_est_tiles
            features.push_back(is_saturating);
            features.push_back(std::log1p(est_waves)); // log_est_waves

            // Modulo features
            features.push_back(static_cast<float>(static_cast<int>(m) % 64 == 0)); // m_mod_64
            features.push_back(static_cast<float>(static_cast<int>(n) % 64 == 0)); // n_mod_64
            features.push_back(static_cast<float>(static_cast<int>(k) % 64 == 0)); // k_mod_64

            // Tile counts (log-transformed for specific sizes)
            features.push_back(std::log1p(tiles_64x48)); // log_tiles_64x48
            features.push_back(std::log1p(tiles_64x96)); // log_tiles_64x96
            features.push_back(std::log1p(tiles_128)); // log_tiles_128x128
            features.push_back(std::log1p(tiles_192)); // log_tiles_192x192
            features.push_back(std::log1p(tiles_224)); // log_tiles_224x224

            // Wastage features
            features.push_back(wastage_32);
            features.push_back(wastage_64);
            features.push_back(wastage_128);
            features.push_back(wastage_192);
            features.push_back(wastage_224);
            features.push_back(wastage_256);

            // Best fit
            features.push_back(best_fit_n);

            // Underfill flags
            features.push_back((m < 256.0f) ? 1.0f : 0.0f); // m_underfills_256
            features.push_back((n < 256.0f) ? 1.0f : 0.0f); // n_underfills_256
            features.push_back((m < 192.0f) ? 1.0f : 0.0f); // m_underfills_192
            features.push_back((n < 192.0f) ? 1.0f : 0.0f); // n_underfills_192

            // Partial tiles (M & N)
            features.push_back(std::fmod(m, 32.0f) / 32.0f);  // m_partial_32
            features.push_back(std::fmod(n, 32.0f) / 32.0f);  // n_partial_32
            features.push_back(std::fmod(m, 64.0f) / 64.0f);  // m_partial_64
            features.push_back(std::fmod(n, 64.0f) / 64.0f);  // n_partial_64

            features.push_back(std::fmod(m, 128.0f) / 128.0f);  // m_partial_128
            features.push_back(std::fmod(n, 128.0f) / 128.0f);  // n_partial_128
            features.push_back(std::fmod(m, 160.0f) / 160.0f);  // m_partial_160
            features.push_back(std::fmod(n, 160.0f) / 160.0f);  // n_partial_160
            features.push_back(std::fmod(m, 192.0f) / 192.0f);  // m_partial_192
            features.push_back(std::fmod(n, 192.0f) / 192.0f);  // n_partial_192
            features.push_back(std::fmod(m, 224.0f) / 224.0f);  // m_partial_224
            features.push_back(std::fmod(n, 224.0f) / 224.0f);  // n_partial_224
            features.push_back(std::fmod(m, 256.0f) / 256.0f);  // m_partial_256
            features.push_back(std::fmod(n, 256.0f) / 256.0f);  // n_partial_256
       
            // Wastage comparisons
            features.push_back(wastage_256 - wastage_224);
            features.push_back(wastage_256 - wastage_192); // wastage_256_vs_192
            features.push_back(wastage_256 - wastage_128);

            // Raw remainders
            features.push_back(std::fmod(m, 224.0f));  // m_mod_224
            features.push_back(std::fmod(n, 224.0f));  // n_mod_224
            features.push_back(std::fmod(n, 256.0f));  // n_mod_256

            // Edge case features
            features.push_back(is_tiny_m);
            features.push_back(is_tiny_n);
            features.push_back(is_small_m);
            features.push_back(is_small_n);
            features.push_back(is_gemv_n);
            features.push_back(is_all_tiny);

            // General features
            features.push_back(n_small_misaligned);
            features.push_back(k_small_misaligned);
            features.push_back(n_small_wastage_ratio);
            features.push_back(extreme_aspect_ratio);
            features.push_back(n_vector);
            features.push_back(very_extreme_aspect);

            // NT-specific features
            if (is_NT) {
                // Output size features
                features.push_back(std::log1p(output_size)); // log_output_size
                features.push_back(is_tiny_output);
                features.push_back(is_very_tiny_output);
                features.push_back(is_ultra_tiny_output);

        
                features.push_back(std::log1p(k / (m + 1.0f)));  // log_k_vs_m
                features.push_back(std::log1p(k / (n + 1.0f)));  // log_k_vs_n

                // Parallelization
                features.push_back(std::log1p(
                    output_size / static_cast<float>(hw_constants->n_cu))); // log_output_vs_cu
                features.push_back(insufficient_parallelism);
                features.push_back(severe_underutilization);
                features.push_back(std::log1p(k_reuse_per_output));  // log_k_reuse

                // K memory
                features.push_back(std::log1p(k_memory_bytes));  // log_k_memory
                features.push_back(std::log1p(k_memory_vs_l3)); 

                // Workload distribution
                features.push_back(std::log1p(k * 2.0f));  // log_work_per_output

                // Dimension dominance
                features.push_back(k_is_max_dim);
                features.push_back(k_dominates_both);
                features.push_back(k_dominates_n_10x);
                features.push_back(k_dominates_n_100x);
                features.push_back(k_dominates_n_1000x);
                features.push_back(m_dominates_n_10x);
                features.push_back(k_dominates_m_10x);
                features.push_back(n_dominates_m_10x);
                features.push_back(n_dominates_m_100x);
                features.push_back(n_dominates_k_10x);
                features.push_back(n_dominates_k_100x);
                features.push_back(m_dominates_k_10x); 
                features.push_back(m_dominates_k_100x);

                // Severity
                features.push_back(std::log1p(std::max({
                    m / (n + 1.0f), n / (m + 1.0f),
                    k / (m + 1.0f), m / (k + 1.0f),
                    k / (n + 1.0f), n / (k + 1.0f)
                })));   // log_max_aspect_ratio

                features.push_back(extreme_dimension_ratio_10x);
                features.push_back(extreme_dimension_ratio_100x);

                // Ultra tiny
                features.push_back(m_ultra_tiny);
                features.push_back(n_ultra_tiny);
                features.push_back(k_ultra_tiny_v2);
                features.push_back(any_dim_ultra_tiny);
                features.push_back(multiple_dims_tiny);

                // Problematic configs
                features.push_back(large_output_small_k);

                // Combined patterns
                features.push_back(extreme_ratio_and_tiny_dim);
                features.push_back(k_dominates_and_small_output);

                // Specific ratios
                features.push_back(n_div_k_very_small);
                features.push_back(m_div_k_very_small);
                features.push_back(n_div_k_ultra_small);

                // Tile needs
                features.push_back(likely_needs_small_tile);

                // Work elements
                features.push_back(is_micro_gemm);
                features.push_back(is_nano_gemm);

                // Vectorization
                features.push_back(m_not_vec4_aligned);
                features.push_back(n_not_vec4_aligned);
                features.push_back(k_not_vec4_aligned);
                features.push_back(m_not_vec8_aligned);
                features.push_back(n_not_vec8_aligned);

                // Pathological cases
                features.push_back(pathological_case_type1);
                features.push_back(problem_severity_count);
                features.push_back(multiple_problems);
            }
            return encoder->forward(features);
        }

        int check_cluster(
            int                                                         remSolutions,
            const std::vector<float>&                                   gemm_embedding,
            const int                                                   cidx,
            const MyProblem&                                            problem,
            const Hardware&                                             hardware,
            std::vector<std::pair<float, std::shared_ptr<MySolution>>>& rankedSolutions) const
        {
            const int          n_solutions = static_cast<int>(embeddings->embeddings[cidx].size());
            std::vector<float> solution_similarities(n_solutions);

            int max_sim_idx = 0;
            if(is_quantized_)
            {
                max_sim_idx = inner_product_bf16(
                    embeddings->embeddings_bf16[cidx], gemm_embedding, solution_similarities);
            }
            else
            {
                max_sim_idx = inner_product(
                    embeddings->embeddings[cidx], gemm_embedding, solution_similarities);
            }

            auto sol = solutions[embeddings->cluster_indices[cidx][max_sim_idx]];
            if((*(sol->problemPredicate))(problem))
            {

                Task task(hardware, problem, *(sol));
                if((*sol->taskPredicate)(task))
                {
                    rankedSolutions.emplace_back(solution_similarities[max_sim_idx], sol);
                    if(remSolutions == 1)
                    {
                        return 0;
                    }
                }
            }

            std::vector<int> solution_indices(n_solutions);
            std::iota(solution_indices.begin(), solution_indices.end(), 0);

            size_t currentIndex = 1;
            while(remSolutions > 0 && currentIndex < solution_indices.size())
            {
                std::partial_sort(solution_indices.begin() + currentIndex - 1,
                                  solution_indices.begin() + currentIndex + 1,
                                  solution_indices.end(),
                                  [&solution_similarities](int i0, int i1) {
                                      return solution_similarities[i0] > solution_similarities[i1];
                                  });

                auto kidx = solution_indices[currentIndex];
                auto sol  = solutions[embeddings->cluster_indices[cidx][kidx]];

                if((*(sol->problemPredicate))(problem))
                {

                    Task task(hardware, problem, *(sol));
                    if((*sol->taskPredicate)(task))
                    {
                        rankedSolutions.emplace_back(solution_similarities[kidx], sol);
                        remSolutions--;
                    }
                }

                ++currentIndex;
            }
            return remSolutions;
        }

        int inner_product(const std::vector<std::vector<float>>& solution_embeddings,
                          const std::vector<float>&              gemm_embedding,
                          std::vector<float>&                    scores) const
        {

            const int n_solutions              = static_cast<int>(solution_embeddings.size());
            const int embedding_dim            = static_cast<int>(gemm_embedding.size());
            const float* __restrict__ gemm_ptr = gemm_embedding.data();

            short amax = 0;
            float vmax = std::numeric_limits<float>::lowest();

            for(int i = 0; i < n_solutions; ++i)
            {
                const float* __restrict__ sol_ptr = solution_embeddings[i].data();
#ifdef __AVX2__
                scores[i] = avx_dot(embedding_dim, sol_ptr, gemm_ptr);
#else
                scores[i] = 0.0f;
                for(int j = 0; j < embedding_dim; j += 4)
                {
                    float out0 = sol_ptr[j] * gemm_ptr[j];
                    float out1 = sol_ptr[j + 1] * gemm_ptr[j + 1];
                    float out2 = sol_ptr[j + 2] * gemm_ptr[j + 2];
                    float out3 = sol_ptr[j + 3] * gemm_ptr[j + 3];
                    scores[i] += out0 + out1 + out2 + out3;
                }
#endif
                if(scores[i] > vmax)
                {
                    vmax = scores[i];
                    amax = i;
                }
            }

            return amax;
        }

        int inner_product_bf16(const std::vector<std::vector<uint16_t>>& solution_embeddings_bf16,
                               const std::vector<float>&                 gemm_embedding,
                               std::vector<float>&                       scores) const
        {
            short     amax          = 0;
            float     vmax          = std::numeric_limits<float>::lowest();
            const int embedding_dim = static_cast<int>(gemm_embedding.size());

#if defined(__AVX512F__) && defined(__AVX512BF16__)
            std::vector<uint16_t> gemm_bf16(static_cast<std::size_t>(embedding_dim));
            const float* __restrict__ gemm_ptr_f32 = gemm_embedding.data();
            for(int j = 0; j < embedding_dim; ++j)
            {
                gemm_bf16[static_cast<std::size_t>(j)] = float_to_bf16_rne(gemm_ptr_f32[j]);
            }
            const uint16_t* __restrict__ gemm_ptr = gemm_bf16.data();
#else
            const float* __restrict__ gemm_ptr = gemm_embedding.data();
#endif

            for(int i = 0; i < static_cast<int>(solution_embeddings_bf16.size()); ++i)
            {
                const uint16_t* __restrict__ sol_ptr = solution_embeddings_bf16[i].data();
#ifdef __AVX2__
                scores[i] = avx_dot_bf16(embedding_dim, gemm_ptr, sol_ptr);
#else
                scores[i] = 0.0f;
                for(int j = 0; j < embedding_dim; j += 4)
                {
                    float out0 = bf16_to_float(sol_ptr[j]) * gemm_ptr[j];
                    float out1 = bf16_to_float(sol_ptr[j + 1]) * gemm_ptr[j + 1];
                    float out2 = bf16_to_float(sol_ptr[j + 2]) * gemm_ptr[j + 2];
                    float out3 = bf16_to_float(sol_ptr[j + 3]) * gemm_ptr[j + 3];
                    scores[i] += out0 + out1 + out2 + out3;
                }
#endif
                if(scores[i] > vmax)
                {
                    vmax = scores[i];
                    amax = i;
                }
            }

            return amax;
        }
    };

} // namespace TensileLite
