// MIT License
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "parity.hpp"

#include <cuda_runtime.h>
#include <curand.h>

#include <iostream>

#define CUDA_CHECK(condition)                                                                \
    {                                                                                        \
        cudaError_t error = condition;                                                       \
        if(error != cudaSuccess)                                                             \
        {                                                                                    \
            std::cout << "cuda error at " __FILE__ << ":" << __LINE__ << error << std::endl; \
            exit(EXIT_FAILURE);                                                              \
        }                                                                                    \
    }

#define CURAND_CHECK(condition)                                                          \
    {                                                                                    \
        curandStatus_t error = condition;                                                \
        if(error != CURAND_STATUS_SUCCESS)                                               \
        {                                                                                \
            std::cout << "curand error at " __FILE__ << ":" << __LINE__ << ": " << error \
                      << std::endl;                                                      \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }

static curandRngType rng_type_to_curand(const generator_type rng_type)
{
    switch(rng_type)
    {
        case generator_type::XORWOW: return CURAND_RNG_PSEUDO_XORWOW;
        case generator_type::MRG32K3A: return CURAND_RNG_PSEUDO_MRG32K3A;
        case generator_type::MTGP32: return CURAND_RNG_PSEUDO_MTGP32;
        case generator_type::PHILOX4_32_10: return CURAND_RNG_PSEUDO_PHILOX4_32_10;
        case generator_type::MT19937: return CURAND_RNG_PSEUDO_MT19937;
        case generator_type::SOBOL32: return CURAND_RNG_QUASI_SOBOL32;
        case generator_type::SCRAMBLED_SOBOL32: return CURAND_RNG_QUASI_SCRAMBLED_SOBOL32;
        case generator_type::SOBOL64: return CURAND_RNG_QUASI_SOBOL64;
        case generator_type::SCRAMBLED_SOBOL64: return CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
    }
}

template<typename T, typename F>
static std::vector<T> generate(const test_case& test_case, F callback)
{
    T* data;
    CUDA_CHECK(cudaMalloc(&data, test_case.size * sizeof(T)));

    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, rng_type_to_curand(test_case.rng_type)));
    if(generator_is_pseudo(test_case.rng_type))
    {
        CURAND_CHECK(curandSetGeneratorOrdering(generator, CURAND_ORDERING_PSEUDO_LEGACY));
        if(test_case.prng_seed >= 0)
        {
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, test_case.prng_seed));
        }
    }
    else if(test_case.qrng_dimensions >= 0)
    {
        CURAND_CHECK(curandSetQuasiRandomGeneratorDimensions(generator, test_case.qrng_dimensions));
    }

    if(test_case.offset >= 0)
    {
        CURAND_CHECK(curandSetGeneratorOffset(generator, test_case.offset));
    }

    CURAND_CHECK(callback(generator, data, test_case.size));

    CURAND_CHECK(curandDestroyGenerator(generator));

    std::vector<T> results(test_case.size);
    CUDA_CHECK(
        cudaMemcpy(results.data(), data, test_case.size * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(data));

    return results;
}

std::vector<unsigned int> test_curand_generate(const test_case& test_case)
{
    return generate<unsigned int>(test_case, curandGenerate);
}

std::vector<unsigned long long> test_curand_generate_long_long(const test_case& test_case)
{
    return generate<unsigned long long>(test_case, curandGenerateLongLong);
}
