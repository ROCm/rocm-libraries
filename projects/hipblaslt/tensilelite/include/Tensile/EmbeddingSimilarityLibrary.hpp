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

        std::map<int, std::shared_ptr<MySolution>> solutionmap;
        std::vector<std::shared_ptr<MySolution>>   solutions;
        std::shared_ptr<Encoder>                   encoder;
        std::shared_ptr<SolutionEmbeddings>        embeddings;

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
            if(problem.batchSize(0) > 1) // TODO Temporary patch until we have the logic for it
                return {};

            std::vector<float> gemm_embedding = computeGEMMEmbeddings(problem);

            std::vector<int> centroid_indexes(embeddings->centroids.size());
            std::iota(centroid_indexes.begin(), centroid_indexes.end(), 0);
            std::vector<float> centroid_similarities(embeddings->centroids.size());

            int max_sim_idx
                = (embeddings->centroids.size()) > 1
                      ? inner_product(embeddings->centroids, gemm_embedding, centroid_similarities)
                      : 0;

            std::vector<std::pair<float, std::shared_ptr<MySolution>>> rankedSolutions;
            rankedSolutions.reserve(numSolutions);

            int remSolutions = checkCluster(numSolutions,
                                            gemm_embedding,
                                            embeddings->embeddings[max_sim_idx],
                                            embeddings->cluster_indices[max_sim_idx],
                                            problem,
                                            hardware,
                                            rankedSolutions);
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
                    remSolutions = checkCluster(remSolutions,
                                                gemm_embedding,
                                                embeddings->embeddings[cidx],
                                                embeddings->cluster_indices[cidx],
                                                problem,
                                                hardware,
                                                rankedSolutions);
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
        // Constants
        static constexpr float EPSILON = 1e-6f;
        static constexpr int N_CU = 256; // TODO properties.multiProcessorCount ?
 
        // Bucket function for dimension binning
        // Matches pd.cut with bins=[0, 128, 256, 512, 1024, 2048, 4096, 8192, inf]
        // Using (left, right] intervals (exclusive left, inclusive right)
        float bucket_dimension(float value) const {
            if (value <= 0.0f) return -1.0f;  // NaN equivalent
            if (value <= 128.0f) return 0.0f;
            if (value <= 256.0f) return 1.0f;
            if (value <= 512.0f) return 2.0f;
            if (value <= 1024.0f) return 3.0f;
            if (value <= 2048.0f) return 4.0f;
            if (value <= 4096.0f) return 5.0f;
            if (value <= 8192.0f) return 6.0f;
            return 7.0f;
        }

        // Bucket function for aspect ratio (shape_category)
        // Matches pd.cut with bins=[0, 0.5, 1.5, 2.5, inf], labels=[0, 1, 2, 3]
        float bucket_aspect_ratio(float aspect) const {
            if (aspect <= 0.0f) return -1.0f;  // NaN equivalent
            if (aspect <= 0.5f) return 0.0f;   // wide
            if (aspect <= 1.5f) return 1.0f;   // square
            if (aspect <= 2.5f) return 2.0f;   // tall
            return 3.0f;                        // very_tall
        }

        // Find best fitting tile size (minimum wastage)
        // Matches Python: min(tiles, key=lambda t: ceil(x/t)*t - x)
        float best_fit_tile(float x) const {
            constexpr float tiles[] = {128.0f, 192.0f, 224.0f, 256.0f};
            float best = tiles[0];
            float min_waste = std::ceil(x / tiles[0]) * tiles[0] - x;

            for (int i = 1; i < 4; ++i) {
                float waste = std::ceil(x / tiles[i]) * tiles[i] - x;
                if (waste < min_waste) {
                    min_waste = waste;
                    best = tiles[i];
                }
            }
            return best;
        }

        // Compute wastage for a given tile size
        float compute_wastage(float m, float n, float tile) const {
            float tiles_m = std::ceil(m / tile);
            float tiles_n = std::ceil(n / tile);
            float work = tiles_m * tile * tiles_n * tile;
            return (work - m * n) / work;
        }

        /* Helper function to compute GEMM embeddings. */
        std::vector<float> computeGEMMEmbeddings(const MyProblem& problem) const
        {
            // TODO check if sizes > 1
            float m = problem.freeSizeA(0);
            float n = problem.freeSizeB(0);
            float k = problem.boundSize(0);

            bool transA = problem.transA();
            bool transB = problem.transB();

            float lda = problem.a().strides()[1];
            float ldb = problem.b().strides()[1];
            float ldc = problem.c().strides()[1];
            float ldd = problem.d().strides()[1];

            float batch_count = problem.batchSize(0);

            float stride_a = transA ? lda * m : lda * k;
            float stride_b = transB ? ldb * k : ldb * n;

            float stride_c = ldc * n;
            float stride_d = ldd * n;

            float flops = 2 * m * n * k * batch_count;
            float sqrt_mn = std::sqrt(m * n);
            float bytes = (m * k + k * n + m * n) * 2;

            // Aspect ratios
            float aspect_m_n = m / (n + EPSILON);
            float aspect_m_k = m / (k + EPSILON);
            float aspect_n_k = n / (k + EPSILON);

            // N div tile (before log)
            float n_div_tile128 = n / 128.0f;
            float n_div_tile256 = n / 256.0f;

            // K div (before log)
            float k_div_32 = k / 32.0f;
            float k_div_64 = k / 64.0f;

            // Occupancy
            float est_tiles = std::ceil(m / 256.0f) * std::ceil(n / 256.0f);

            // Tile counts
            float tiles_128 = std::ceil(m / 128.0f) * std::ceil(n / 128.0f);
            float tiles_192 = std::ceil(m / 192.0f) * std::ceil(n / 192.0f);
            float tiles_224 = std::ceil(m / 224.0f) * std::ceil(n / 224.0f);
            float tiles_256 = est_tiles;  // Same as est_tiles

            // Wastage
            float wastage_128 = compute_wastage(m, n, 128.0f);
            float wastage_192 = compute_wastage(m, n, 192.0f);
            float wastage_224 = compute_wastage(m, n, 224.0f);
            float wastage_256 = compute_wastage(m, n, 256.0f);

            std::vector<float> features = { std::log1p(m),
                                            std::log1p(n),
                                            std::log1p(k),
                                            std::log1p(lda),
                                            std::log1p(stride_a),
                                            std::log1p(ldb),
                                            std::log1p(stride_b),
                                            std::log1p(ldc),
                                            std::log1p(stride_c),
                                            std::log1p(ldd),
                                            std::log1p(stride_d),
                                            std::log1p(batch_count),
                                            std::log1p(flops),
                                            bucket_dimension(m),
                                            bucket_dimension(n),
                                            bucket_dimension(k),
                                            aspect_m_n,
                                            aspect_m_k,
                                            aspect_n_k,
                                            bucket_aspect_ratio(aspect_m_n),
                                            std::max(1.0f, ldc / n), // ldc_efficiency
                                            std::log1p(bytes),
                                            flops / (bytes + EPSILON),
                                            // 23-27: M tile alignment
                                            static_cast<float>(static_cast<int>(m) % 128 == 0),
                                            static_cast<float>(static_cast<int>(m) % 160 == 0),
                                            static_cast<float>(static_cast<int>(m) % 192 == 0),
                                            static_cast<float>(static_cast<int>(m) % 224 == 0),
                                            static_cast<float>(static_cast<int>(m) % 256 == 0),
                                            
                                            // 28-32: N tile alignment
                                            static_cast<float>(static_cast<int>(n) % 128 == 0),
                                            static_cast<float>(static_cast<int>(n) % 160 == 0),
                                            static_cast<float>(static_cast<int>(n) % 192 == 0),
                                            static_cast<float>(static_cast<int>(n) % 224 == 0),
                                            static_cast<float>(static_cast<int>(n) % 256 == 0),
                                            // 33-35: K tile alignment
                                            static_cast<float>(static_cast<int>(k) % 32 == 0),
                                            static_cast<float>(static_cast<int>(k) % 64 == 0),
                                            static_cast<float>(static_cast<int>(k) % 128 == 0),
                                            // 36-37: n_div_tile (log transformed)
                                            std::log1p(n_div_tile128),
                                            std::log1p(n_div_tile256),
                                            // 38: is_large
                                            static_cast<float>(m >= 8192.0f || n >= 8192.0f || k >= 8192.0f),
                                            // 39-41: Shape flags
                                            static_cast<float>(m > n),   // is_tall
                                            static_cast<float>(n > m),   // is_wide
                                            static_cast<float>(m == n),  // is_square
                                            // 42-44: Extreme shapes
                                            static_cast<float>(m > 4.0f * n && m > 4.0f * k),  // is_tall_skinny
                                            static_cast<float>(n > 4.0f * m && n > 4.0f * k),  // is_short_wide
                                            static_cast<float>(k > 4.0f * m && k > 4.0f * n),  // is_deep_k
                                            // 45-48: K size flags
                                            static_cast<float>(k < 128.0f),   // is_small_k
                                            static_cast<float>(k > 4096.0f),  // is_large_k
                                            static_cast<float>(k <= 16.0f),   // is_tiny_k
                                            static_cast<float>(k <= 64.0f),   // is_very_small_k
                                            // 49-50: k_div (log transformed)
                                            std::log1p(k_div_32),
                                            std::log1p(k_div_64),
                                            // 51-53: Occupancy
                                            std::log1p(est_tiles),                              // log_est_tiles
                                            static_cast<float>(est_tiles >= static_cast<float>(N_CU)),  // is_saturating
                                            est_tiles / static_cast<float>(N_CU),               // est_waves
                                            // 54-55: Mod 64 flags
                                            static_cast<float>(static_cast<int>(m) % 64 == 0),
                                            static_cast<float>(static_cast<int>(n)% 64 == 0),
                                            // 56-58: Tile counts
                                            tiles_128,
                                            std::log1p(tiles_192),  // log_tiles_192
                                            tiles_224,
                                            // 59-61: K ratios
                                            k / (sqrt_mn + 1.0f),  // k_to_output_ratio
                                            std::log1p(k / (m * n + 1.0f) * 1e6f),  // log_k_to_mn
                                            std::sqrt(n / (m + 1.0f)),  // sqrt_aspect_nm
                                            // 62-65: Wastage
                                            wastage_128,
                                            wastage_192,
                                            wastage_224,
                                            wastage_256,
                                            // 66-67: Best fit
                                            best_fit_tile(m),
                                            best_fit_tile(n),
                                            // 68-71: Underfill flags
                                            static_cast<float>(m < 256.0f),  // m_underfills_256
                                            static_cast<float>(n < 256.0f),  // n_underfills_256
                                            static_cast<float>(m < 192.0f),  // m_underfills_192
                                            static_cast<float>(n < 192.0f),  // n_underfills_192
                                            // 72-76: Partial M
                                            std::fmod(m, 128.0f) / 128.0f,
                                            std::fmod(m, 160.0f) / 160.0f,
                                            std::fmod(m, 192.0f) / 192.0f,
                                            std::fmod(m, 224.0f) / 224.0f,
                                            std::fmod(m, 256.0f) / 256.0f,
                                            // 77-81: Partial N
                                            std::fmod(n, 128.0f) / 128.0f,
                                            std::fmod(n, 160.0f) / 160.0f,
                                            std::fmod(n, 192.0f) / 192.0f,
                                            std::fmod(n, 224.0f) / 224.0f,
                                            std::fmod(n, 256.0f) / 256.0f,
                                            // 82-84: Wastage comparisons
                                            wastage_256 - wastage_224,
                                            wastage_256 - wastage_192,
                                            wastage_256 - wastage_128,
                                            // 85-88: Raw remainders
                                            std::fmod(m, 224.0f),
                                            std::fmod(m, 256.0f),
                                            std::fmod(n, 224.0f),
                                            std::fmod(n, 256.0f),
                                            // 89-90: Tile count differences
                                            tiles_256 - tiles_224,
                                            tiles_256 - tiles_192
                                        };
            return encoder->forward(features);
        }

        int checkCluster(
            int                                                         remSolutions,
            const std::vector<float>&                                   gemm_embedding,
            const std::vector<std::vector<float>>&                      solution_embeddings,
            const std::vector<int>&                                     cluster_indices,
            const MyProblem&                                            problem,
            const Hardware&                                             hardware,
            std::vector<std::pair<float, std::shared_ptr<MySolution>>>& rankedSolutions) const
        {
            std::vector<float> solution_similarities(solution_embeddings.size());

            int max_sim_idx
                = inner_product(solution_embeddings, gemm_embedding, solution_similarities);

            auto sol = solutions[cluster_indices[max_sim_idx]];
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

            std::vector<int> solution_indices(solution_embeddings.size());
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
                auto sol  = solutions[cluster_indices[kidx]];

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
            short amax = 0;
            float vmax = 0;
            for(int i = 0; i < solution_embeddings.size(); i++)
            {
                auto& solution_embedding = solution_embeddings[i];
#ifdef __AVX2__
                scores[i] = avx_dot(
                    gemm_embedding.size(), solution_embedding.data(), gemm_embedding.data());
#else
                scores[i] = 0.0f;
                for(int j = 0; j < gemm_embedding.size(); j += 4)
                {
                    float out0 = solution_embedding[j] * gemm_embedding[j];
                    float out1 = solution_embedding[j + 1] * gemm_embedding[j + 1];
                    float out2 = solution_embedding[j + 2] * gemm_embedding[j + 2];
                    float out3 = solution_embedding[j + 3] * gemm_embedding[j + 3];
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
