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

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include <Tensile/EmbeddingSimilarity.hpp>

#ifdef __AVX2__
#include <immintrin.h> // For AVX intrinsics

float avx_dot(int N, const float* A, const float* B)
{
    float  dot_product = 0.0f;
    __m256 sum_vec     = _mm256_setzero_ps(); // Initialize a 256-bit vector of zeros
    int    i           = 0;

    // Process 8 floats at a time (for AVX)
    for(; i < N - N % 8; i += 8)
    {
        __m256 a_vec    = _mm256_loadu_ps(&A[i]); // Load 8 floats from A
        __m256 b_vec    = _mm256_loadu_ps(&B[i]); // Load 8 floats from B
        __m256 prod_vec = _mm256_mul_ps(a_vec, b_vec); // Multiply element-wise
        sum_vec         = _mm256_add_ps(sum_vec, prod_vec); // Add to accumulator
    }
    // Horizontal sum of the accumulator vector
    float temp_array[8];
    _mm256_storeu_ps(temp_array, sum_vec);
    for(int j = 0; j < 8; ++j)
    {
        dot_product += temp_array[j];
    }
    // Ramainder
    for(; i < N; ++i)
    {
        dot_product += A[i] * B[i];
    }
    return dot_product;
}
#endif

namespace TensileLite
{
    namespace EmbeddingSimilarity
    {
        void StandardScaler::operator()(std::vector<dtype>& F) const
        {
            assert(mean.size() == F.size() && scale.size() == F.size());
            std::transform(F.begin(), F.end(), mean.begin(), F.begin(), std::minus{});
            std::transform(F.begin(), F.end(), scale.begin(), F.begin(), std::divides{});
        }

        bool StandardScaler::valid(bool verbose) const
        {
            bool is_valid = true;
            if(mean.size() != scale.size())
            {
                if(verbose)
                {
                    std::cerr << "StandardScaler mean and scale do not match." << std::endl;
                }
                is_valid = false;
            }
            if(std::find(scale.begin(), scale.end(), 0.) != scale.end())
            {
                if(verbose)
                {
                    std::cerr << "StandardScaler scale contains zero." << std::endl;
                }
                is_valid = false;
            }
            return is_valid;
        }

        std::vector<dtype> relu_activation(std::vector<dtype>&& F)
        {
            for(auto& f : F)
                f = f > 0.0f ? f : 0.0f; // std::max(f, 0.f);
            return F;
        }

        std::vector<float> dense_forward(const std::vector<float>& input,
                                         const std::vector<float>& weights,
                                         const std::vector<float>& bias)
        {
            size_t             input_dim  = input.size();
            size_t             output_dim = bias.size();
            std::vector<float> output     = bias;

            for(size_t j = 0; j < output_dim; ++j)
            {
#ifdef __AVX2__
                output[j] += avx_dot(input_dim, input.data(), weights.data() + j * input_dim);
#else
                int k = 0;
                int offset = j * input_dim;
                for(; k < input_dim - input_dim % 8; k += 8, offset += 8)
                {
                    float out0 = input[k] * weights[offset];
                    float out1 = input[k + 1] * weights[offset + 1];
                    float out2 = input[k + 2] * weights[offset + 2];
                    float out3 = input[k + 3] * weights[offset + 3];
                    float out4 = input[k + 4] * weights[offset + 4];
                    float out5 = input[k + 5] * weights[offset + 5];
                    float out6 = input[k + 6] * weights[offset + 6];
                    float out7 = input[k + 7] * weights[offset + 7];
                    output[j] += out0 + out1 + out2 + out3 + out4 + out5 + out6 + out7;
                }
                for(; k < input_dim; k++, offset++)
                {
                    output[j] += input[k] * weights[offset];
                }
#endif
            }
            return output;
        }

        std::vector<dtype> Network::operator()(const std::vector<dtype>& F) const
        {
            std::vector<dtype> output = F;
            for(int i = 0; i < weights.size(); i++)
            {
                output = relu_activation(std::move(dense_forward(output, weights[i], bias[i])));
            }

            return dense_forward(output, proj_weights, proj_bias);
        }

        bool Network::valid(bool verbose) const
        {
            // Check dense layers
            for(size_t i = 0; i < weights.size(); ++i)
            {
                size_t output_dim = bias[i].size();
                size_t input_dim  = weights[i].size() / output_dim;
                if(weights[i].size() != input_dim * output_dim || bias[i].size() != output_dim)
                {
                    if(verbose)
                    {
                        std::cerr << "Dense layer " << i << " dimensions do not match: "
                                  << "weights.size() = " << weights[i].size()
                                  << ", bias.size() = " << bias[i].size() << std::endl;
                    }
                    return false;
                }
            }
            // Check projection layer
            size_t proj_out_dim = proj_bias.size();
            if(proj_out_dim % 4 != 0)
            {
                if(verbose)
                {
                    std::cerr << "Projection layer output dimensions must be divisible by 4"
                              << std::endl;
                }
                return false;
            }

            if(!weights.empty())
            {
                size_t proj_in_dim = bias.back().size();
                if(proj_weights.size() != proj_in_dim * proj_out_dim)
                {
                    if(verbose)
                    {
                        std::cerr << "Projection layer dimensions do not match: "
                                  << "proj_weights.size() = " << proj_weights.size()
                                  << ", expected = " << proj_in_dim * proj_out_dim
                                  << ", proj_bias.size() = " << proj_bias.size() << std::endl;
                    }
                    return false;
                }
            }
            return true;
        }

        std::vector<dtype> Encoder::forward(std::vector<float>& gemm_features) const

        {
            scaler(gemm_features);

            std::vector<dtype> encoded_gemm= network(gemm_features);

            return encoded_gemm;
        }

        bool Encoder::valid(bool verbose) const
        {
            bool is_valid = scaler.valid(verbose) && network.valid(verbose);

            size_t input_size = 0;
            if(network.weights.empty())
            {
                input_size = network.proj_weights.size() / network.proj_bias.size();
            }
            else
            {
                input_size = network.weights[0].size() / network.bias[0].size();
            }

            if(scaler.mean.size() != input_size)
            {
                if(verbose)
                {
                    std::cerr << "StandardScaler size (" << scaler.mean.size()
                              << ") does not match EmbeddingSimilarity network input size ("
                              << input_size << ")." << std::endl;
                }
                is_valid = false;
            }

            return is_valid;
        }

    }
}
