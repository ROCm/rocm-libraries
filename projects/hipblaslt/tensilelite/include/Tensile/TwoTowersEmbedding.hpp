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

#include "DataTypes_Half.hpp"
#include <array>
#include <map>
#include <memory>
#include <set>
#include <vector>

#ifdef __AVX2__
float avx_dot(int N, const float* A, const float* B);
#endif

namespace TensileLite
{
    /**
     * \ingroup Tensile
     * \defgroup TwoTowersEmbedding
     *
     * @brief TwoTowersEmbedding model
     *
     * Encoder used to estimate embedding values for problems in the
     * library. Used for TwoTowersEmbeddingLibrary.
     *
     * See TwoTowersEmbedding.cpp
     */

    /**
     * \ingroup TwoTowersEmbedding
     */
    namespace TwoTowersEmbedding
    {

        using dtype = float;

        struct StandardScaler
        {
            void operator()(std::vector<dtype>& F) const;
            bool valid(bool verbose = false) const;

            std::vector<dtype> mean, scale;
        };

        struct QueryEncoder
        {
            QueryEncoder() = default;

            std::vector<dtype> operator()(const std::vector<dtype>& F) const;

            bool valid(bool verbose = false) const;

            std::string description() const
            {
                return "QueryEncoder";
            }

            std::vector<std::vector<TensileLite::TwoTowersEmbedding::dtype>> weights;
            std::vector<std::vector<TensileLite::TwoTowersEmbedding::dtype>> bias;
            std::vector<TensileLite::TwoTowersEmbedding::dtype>              proj_weights;
            std::vector<TensileLite::TwoTowersEmbedding::dtype>              proj_bias;
        };

        struct TwoTowersEmbedding
        {
            TwoTowersEmbedding() = default;

            std::vector<dtype> forward(std::vector<float>& probkey) const;

            bool valid(bool verbose = false) const;

            std::string description() const
            {
                return "TwoTowersEmbedding";
            }

            bool           apply_log;
            StandardScaler scaler;
            QueryEncoder   query_encoder;
        };

        struct DocEmbeddings
        {
            DocEmbeddings() = default;

            std::string description() const
            {
                return "DocEmbeddings";
            }

            std::vector<std::vector<float>>              centroids;
            std::vector<std::vector<std::vector<float>>> embeddings;
            std::vector<std::vector<int>>                cluster_sols;

            std::size_t size() const
            {
                std::set<int> unique_values;
                for(const auto& cluster : cluster_sols)
                {
                    unique_values.insert(cluster.begin(), cluster.end());
                }
                return unique_values.size();
            }
        };

    } // namespace TwoTowersEmbedding
} // namespace TensileLite