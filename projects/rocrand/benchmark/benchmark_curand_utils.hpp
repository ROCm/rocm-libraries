// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCRAND_BENCHMARK_CURAND_UTILS_HPP_
#define ROCRAND_BENCHMARK_CURAND_UTILS_HPP_

#include "benchmark_utils.hpp"

#include <benchmark/benchmark.h>

#include <curand.h>

#include <iostream>

#define CURAND_CALL(condition)                                                                \
    do                                                                                        \
    {                                                                                         \
        curandStatus_t status_ = condition;                                                   \
        if(status_ != CURAND_STATUS_SUCCESS)                                                  \
        {                                                                                     \
            std::cout << "CURAND error: " << status_ << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                           \
            exit(status_);                                                                    \
        }                                                                                     \
    }                                                                                         \
    while(0)

#define CUDA_CALL(condition)                                                               \
    do                                                                                     \
    {                                                                                      \
        cudaError_t error_ = condition;                                                    \
        if(error_ != cudaSuccess)                                                          \
        {                                                                                  \
            std::cout << "CUDA error: " << error_ << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                        \
            exit(error_);                                                                  \
        }                                                                                  \
    }                                                                                      \
    while(0)

inline void add_common_benchmark_curand_info()
{
    int version;
    CURAND_CALL(curandGetVersion(&version));
    benchmark::AddCustomContext("curand_version", std::to_string(version));

    add_common_benchmark_info();
}

inline std::string engine_name(const curandRngType rng_type)
{
    // The returned names have to be able to reproduce the curandRngType by prepending
    // `CUCRAND_RNG_{PSEUDO|QUASI}_` to the name written in all capital letters. The scripts in
    // scripts/config-tuning/ rely on this.
    // clang-format off
    switch(rng_type)
    {
        case CURAND_RNG_PSEUDO_XORWOW:           return "xorwow";
        case CURAND_RNG_PSEUDO_MRG32K3A:         return "mrg32k3a";
        case CURAND_RNG_PSEUDO_MTGP32:           return "mtgp32";
        case CURAND_RNG_PSEUDO_PHILOX4_32_10:    return "philox4_32_10";
        case CURAND_RNG_PSEUDO_MT19937:          return "mt19937";
        case CURAND_RNG_QUASI_SOBOL32:           return "sobol32";
        case CURAND_RNG_QUASI_SCRAMBLED_SOBOL32: return "scrambled_sobol32";
        case CURAND_RNG_QUASI_SOBOL64:           return "sobol64";
        case CURAND_RNG_QUASI_SCRAMBLED_SOBOL64: return "scrambled_sobol64";
        case CURAND_RNG_TEST:                    return "test";
        case CURAND_RNG_PSEUDO_DEFAULT:          return "pseudo_default";
        case CURAND_RNG_QUASI_DEFAULT:           return "quasi_default";
    }
    // clang-format on
}

#endif // ROCRAND_BENCHMARK_CURAND_UTILS_HPP_
