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
#include <Tensile/TwoTowersEmbedding.hpp>
#include <Tensile/Utils.hpp>

namespace TensileLite
{

    /**
     * \ingroup SolutionLibrary
     *
     * Uses a TwoTowersEmbedding network to create embeddings for rank solutions for a given size.
     */

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct TwoTowersEmbeddingLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        using TwoTowersEmbedding = TwoTowersEmbedding::TwoTowersEmbedding;
        using DocEmbeddings      = TensileLite::TwoTowersEmbedding::DocEmbeddings;

        std::map<int, std::shared_ptr<MySolution>> solutionmap;
        std::vector<std::shared_ptr<MySolution>>   solutions;
        std::shared_ptr<TwoTowersEmbedding>        model;
        std::shared_ptr<DocEmbeddings>             embeddings;

        static std::string Type()
        {
            return "TwoTowersEmbedding";
        }
        virtual std::string type() const override
        {
            return Type();
        }
        virtual std::string description() const override
        {
            if(model == nullptr)
                return concatenate(type(), ", TwoTowersEmbedding: nullptr");
            else
                return concatenate(type(), ": ", model->description());
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            const bool experimental = Debug::Instance().useExperimentalSelection();
            if(!experimental)
            {
                // If the experimental library mode is not on treat it like it asserted out
                return nullptr;
            }

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
            const bool experimental = Debug::Instance().useExperimentalSelection();
            if(!experimental)
            {
                // Skip the search for solutions if the environment variable
                // that enables the experimental method is not set
                SolutionSet<MySolution> rv;
                return rv;
            }
            SolutionSet<MySolution> rv;
            for(auto const& row : solutionmap)
                rv.insert(row.second);

            return rv;
        }

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            std::vector<float> queryEmb = computeQueryEmbs(problem);

            std::vector<int> cent_indexes(embeddings->centroids.size());
            std::iota(cent_indexes.begin(), cent_indexes.end(), 0);
            std::vector<float> cent_similarities(embeddings->centroids.size());

            int max_sim_idx
                = (embeddings->centroids.size()) > 1
                      ? inner_product(embeddings->centroids, queryEmb, cent_similarities)
                      : 0;

            std::vector<std::pair<float, std::shared_ptr<MySolution>>> rankedSolutions;
            rankedSolutions.reserve(numSolutions);

            int remSolutions = checkCluster(numSolutions,
                                            queryEmb,
                                            embeddings->embeddings[max_sim_idx],
                                            embeddings->cluster_sols[max_sim_idx],
                                            problem,
                                            hardware,
                                            rankedSolutions);
            if(remSolutions > 0)
            {
                auto cent_begin = cent_indexes.begin(), cent_end = cent_indexes.end();

                size_t currentIndex = 1;
                while(remSolutions > 0 && currentIndex < cent_indexes.size())
                {
                    std::partial_sort(cent_begin + currentIndex - 1,
                                      cent_begin + currentIndex + 1,
                                      cent_end,
                                      [&cent_similarities](int i0, int i1) {
                                          return cent_similarities[i0] > cent_similarities[i1];
                                      });
                    auto cidx    = cent_indexes[currentIndex];
                    remSolutions = checkCluster(remSolutions,
                                                queryEmb,
                                                embeddings->embeddings[cidx],
                                                embeddings->cluster_sols[cidx],
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
            const bool experimental = Debug::Instance().useExperimentalSelection();
            if(!experimental)
            {
                // Skip the search for solutions if the environment variable
                // that enables the experimental method is not set
                SolutionSet<MySolution> rv;
                return rv;
            }

            SolutionSet<MySolution> rv;
            for(auto const& row : solutionmap)
                rv.insert(row.second);

            return rv;
        }

    protected:
        /* Helper function to compute query embeddings. */
        std::vector<float> computeQueryEmbs(const MyProblem& problem) const
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

            return model->forward(features);
        }

        int checkCluster(
            int                                                         remSolutions,
            const std::vector<float>&                                   query_embedding,
            const std::vector<std::vector<float>>&                      kernel_embeddings,
            const std::vector<int>&                                     cluster_solutions,
            const MyProblem&                                            problem,
            const Hardware&                                             hardware,
            std::vector<std::pair<float, std::shared_ptr<MySolution>>>& rankedSolutions) const
        {
            std::vector<float> kernel_similarities(kernel_embeddings.size());

            int max_sim_idx
                = inner_product(kernel_embeddings, query_embedding, kernel_similarities);

            auto sol = solutions[cluster_solutions[max_sim_idx]];
            if((*(sol->problemPredicate))(problem))
            {
                Task task(hardware, problem, *(sol));
                if((*sol->taskPredicate)(task))
                {
                    rankedSolutions.emplace_back(kernel_similarities[max_sim_idx], sol);
                    if(remSolutions == 1)
                    {
                        return 0;
                    }
                }
            }

            std::vector<int> kernel_indices(kernel_embeddings.size());
            std::iota(kernel_indices.begin(), kernel_indices.end(), 0);

            size_t currentIndex = 1;
            while(remSolutions > 0 && currentIndex < kernel_indices.size())
            {
                std::partial_sort(kernel_indices.begin() + currentIndex - 1,
                                  kernel_indices.begin() + currentIndex + 1,
                                  kernel_indices.end(),
                                  [&kernel_similarities](int i0, int i1) {
                                      return kernel_similarities[i0] > kernel_similarities[i1];
                                  });

                auto kidx = kernel_indices[currentIndex];
                auto sol  = solutions[cluster_solutions[kidx]];

                if((*(sol->problemPredicate))(problem))
                {
                    Task task(hardware, problem, *(sol));
                    if((*sol->taskPredicate)(task))
                    { // (*sol->hardwarePredicate)(hardware)
                        rankedSolutions.emplace_back(kernel_similarities[kidx], sol);
                        remSolutions--;
                    }
                }
                ++currentIndex;
            }
            return remSolutions;
        }

        int inner_product(const std::vector<std::vector<float>>& doc_embeddings,
                          const std::vector<float>&              query_embedding,
                          std::vector<float>&                    scores) const
        {
            short amax = 0;
            float vmax = 0;
            for(int i = 0; i < doc_embeddings.size(); i++)
            {
                auto& doc_embedding = doc_embeddings[i];
#ifdef __AVX2__
                scores[i]
                    = avx_dot(query_embedding.size(), doc_embedding.data(), query_embedding.data());
#else
                scores[i] = 0.0f;
                for(int j = 0; j < query_embedding.size(); j += 4)
                {
                    float a = doc_embedding[j] * query_embedding[j];
                    float b = doc_embedding[j + 1] * query_embedding[j + 1];
                    float c = doc_embedding[j + 2] * query_embedding[j + 2];
                    float d = doc_embedding[j + 3] * query_embedding[j + 3];
                    scores[i] += a + b + c + d;
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
