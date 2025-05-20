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

#include <Tensile/MLPClassification.hpp>

namespace TensileLite
{
    namespace MLPClassification
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

        struct WeightMatrix
        {
            WeightMatrix() = default;
            WeightMatrix(const std::vector<float>& W)
                : weight(W.begin(), W.end())
            {
            }
            virtual ~WeightMatrix() = default;

            virtual void operator()(const std::vector<dtype>& F, std::vector<dtype>& Fout) const;

            std::vector<dtype> weight;
        };

        /*
         * Specifying matrix dimensions at compile time for better unrolling etc.?
         */
        template <int N_IN>
        struct WeightMatrixFixed : public WeightMatrix
        {
            WeightMatrixFixed() = default;
            WeightMatrixFixed(const std::vector<float>& W)
                : WeightMatrix(W)
            {
            }

            void operator()(const std::vector<dtype>& F, std::vector<dtype>& Fout) const override;
        };

        void WeightMatrix::operator()(const std::vector<dtype>& F, std::vector<dtype>& Fout) const
        {
            for(int i = 0; i < Fout.size(); i++)
                Fout[i] += std::inner_product(
                    F.begin(), F.end(), weight.begin() + i * F.size(), dtype(0.));
        }

        template <int N_IN>
        void WeightMatrixFixed<N_IN>::operator()(const std::vector<dtype>& F,
                                                 std::vector<dtype>&       Fout) const
        {
            assert(F.size() == N_IN);
            auto W = weight.data();
            for(int i = 0; i < Fout.size(); i++)
            {
                dtype fi(0.);
                auto  Fptr = F.data();
#pragma clang loop unroll_count(N_IN)
                for(int j = 0; j < N_IN; j++)
                    fi += (*W++) * (*Fptr++);
                Fout[i] += fi;
            }
        }

        DenseLayer::DenseLayer(const std::vector<float>& weights, const std::vector<float>& bias)
        {
            size_out = bias.size();
            size_in  = weights.size() / size_out;
            if(size_in * size_out != weights.size())
                throw std::runtime_error("Error: weights and bias dimensions do not match.");
            switch(size_in)
            {
            case 10:
                W = std::make_shared<WeightMatrixFixed<10>>(weights);
                break;
            case 16:
                W = std::make_shared<WeightMatrixFixed<16>>(weights);
                break;
            case 32:
                W = std::make_shared<WeightMatrixFixed<32>>(weights);
                break;
            case 64:
                W = std::make_shared<WeightMatrixFixed<64>>(weights);
                break;
            case 128:
                W = std::make_shared<WeightMatrixFixed<128>>(weights);
                break;
            case 256:
                W = std::make_shared<WeightMatrixFixed<256>>(weights);
                break;
            default:
                W = std::make_shared<WeightMatrix>(weights);
            }
            B.assign(bias.begin(), bias.end());
        }

        std::vector<dtype> DenseLayer::operator()(const std::vector<dtype>& F) const
        {
            auto Fout = B;
            (*W)(F, Fout);
            return Fout;
        }

        bool DenseLayer::valid(bool verbose) const
        {
            if(B.size() != size_out || W->weight.size() != size_in * size_out)
            {
                if(verbose)
                {
                    std::cerr << "Bias and weight dimensions do not match." << std::endl;
                }
                return false;
            }
            return true;
        }

        std::vector<dtype>& activation(std::vector<dtype>& F)
        {
            for(auto& f : F) // relu
                f = f > 0. ? f : 0.; // std::max(f, 0.f);
            return F;
        }

        std::vector<dtype> activation(std::vector<dtype>&& F)
        {
            return activation(F);
        }

        std::vector<dtype> ResBlock::operator()(const std::vector<dtype>& F) const
        {
            auto Fout = linear2(activation(linear1(F)));
            auto Fres = res(F);
            std::transform(Fout.begin(), Fout.end(), Fres.begin(), Fout.begin(), std::plus{});
            return activation(Fout);
        }

        bool ResBlock::valid(bool verbose) const
        {
            bool is_valid = linear1.valid(verbose) && linear2.valid(verbose) && res.valid(verbose);
            if(linear1.size_out != linear2.size_in)
            {
                if(verbose)
                {
                    std::cerr << "Linear layer 2 does not match output of linear layer 1."
                              << std::endl;
                }
                is_valid = false;
            }
            if(linear1.size_in != res.size_in || linear2.size_out != res.size_out)
            {
                if(verbose)
                {
                    std::cerr
                        << "Residual connection layer size does not match other linear layers."
                        << std::endl;
                }
                is_valid = false;
            }
            return is_valid;
        }

        std::vector<dtype> MLPNet::predict(std::vector<float> const& probkey) const
        {
            dtype M = probkey[0], N = probkey[1], B = probkey[2], K = probkey[3];
            M = std::min(dtype(16384), M);
            N = std::min(dtype(16384), N);
            K = std::min(dtype(16384), K);
            B = std::min(dtype(32), B);
            dtype gflops = M * N * K / 1.e9, reads = (M * N + M * K + K * N) / 1.e6;
            std::vector<dtype> F = {M,
                                    N,
                                    K,
                                    B,
                                    dtype(std::log(M * N)),
                                    dtype(int(M) % 256),
                                    dtype(int(N) % 256),
                                    dtype(int(K) % 256),
                                    dtype(int(B) % 256),
                                    gflops,
                                    reads,
                                    gflops / reads};
            scaler(F);
            for(auto& res : res_blocks)
                F = res(F);
            return dense(F);
        }

        bool MLPNet::valid(bool verbose) const
        {
            bool is_valid
                = scaler.valid(verbose) && dense.valid(verbose)
                  && std::all_of(res_blocks.cbegin(),
                                 res_blocks.cend(),
                                 [&verbose](const ResBlock& r) { return r.valid(verbose); });
            if(!(res_blocks.empty() || dense.size_in == res_blocks.back().res.size_out))
            {
                if(verbose)
                {
                    std::cerr << "MLPNet dense layer input size not correct." << std::endl;
                }
                is_valid = false;
            }
            if(scaler.mean.size()
               != (res_blocks.empty() ? dense.size_in : res_blocks[0].linear1.size_in))
            {
                if(verbose)
                {
                    std::cerr << "StandardScaler size does not match MLPNet network input size."
                              << std::endl;
                }
                is_valid = false;
            }
            return is_valid;
        }

    }
}
