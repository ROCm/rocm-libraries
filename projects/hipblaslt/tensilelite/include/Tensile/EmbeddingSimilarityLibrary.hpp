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
#include <Tensile/MLFeatures.hpp>
#include <Tensile/ProblemKey.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/EmbeddingSimilarity.hpp>
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
                                          return centroid_similarities[i0] > centroid_similarities[i1]; 
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
        /* Helper function to compute GEMM embeddings. */
        std::vector<float> computeGEMMEmbeddings(const MyProblem& problem) const
        {
            float m = problem.freeSizeA(0);
            float n = problem.freeSizeB(0);
            float k = problem.boundSize(0);

            bool transA = problem.transA();
            bool transB = problem.transB();

            float lda = problem.a().strides()[1];
            float ldb = problem.b().strides()[1];
            float ldc = problem.c().strides()[1];
            float ldd = problem.d().strides()[1];

            float stride_a = transA ? lda * m : lda * k;
            float stride_b = transB ? ldb * k : ldb * n;

            float stride_c = ldc * n;
            float stride_d = ldd * n;

            float flops = 2 * m * n * k;

            std::vector<float> features
                = {m, n, k, lda, stride_a, ldb, stride_b, ldc, stride_c, ldd, stride_d, flops};

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
                scores[i]
                    = avx_dot(gemm_embedding.size(), solution_embedding.data(), gemm_embedding.data());
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
