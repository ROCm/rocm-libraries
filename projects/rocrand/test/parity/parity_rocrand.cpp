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

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>

#include <iostream>

#define HIP_CHECK(condition)                                                                     \
    {                                                                                            \
        hipError_t error = condition;                                                            \
        if(error != hipSuccess)                                                                  \
        {                                                                                        \
            std::cout << "hip error at " __FILE__ ":" << __LINE__ << ": " << error << std::endl; \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }

#define ROCRAND_CHECK(condition)                                                        \
    {                                                                                   \
        rocrand_status status = condition;                                              \
        if(status != ROCRAND_STATUS_SUCCESS)                                            \
        {                                                                               \
            std::cout << "rocrand error at " __FILE__ ":" << __LINE__ << ": " << status \
                      << std::endl;                                                     \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    }

static rocrand_rng_type rng_type_to_rocrand(const generator_type rng_type)
{
    switch(rng_type)
    {
        case generator_type::XORWOW: return ROCRAND_RNG_PSEUDO_XORWOW;
        case generator_type::MRG32K3A: return ROCRAND_RNG_PSEUDO_MRG32K3A;
        case generator_type::MTGP32: return ROCRAND_RNG_PSEUDO_MTGP32;
        case generator_type::PHILOX4_32_10: return ROCRAND_RNG_PSEUDO_PHILOX4_32_10;
        case generator_type::MT19937: return ROCRAND_RNG_PSEUDO_MT19937;
        case generator_type::SOBOL32: return ROCRAND_RNG_QUASI_SOBOL32;
        case generator_type::SCRAMBLED_SOBOL32: return ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32;
        case generator_type::SOBOL64: return ROCRAND_RNG_QUASI_SOBOL64;
        case generator_type::SCRAMBLED_SOBOL64: return ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64;
    }
}

template<typename T, typename F>
static std::vector<T> generate(const test_case& test_case, F callback)
{
    T* data;
    HIP_CHECK(hipMalloc(&data, test_case.size * sizeof(T)));

    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type_to_rocrand(test_case.rng_type)));
    if(generator_is_pseudo(test_case.rng_type))
    {
        ROCRAND_CHECK(rocrand_set_ordering(generator, ROCRAND_ORDERING_PSEUDO_LEGACY));
        if(test_case.prng_seed >= 0)
        {
            ROCRAND_CHECK(rocrand_set_seed(generator, test_case.prng_seed));
        }
    }
    else if(test_case.qrng_dimensions >= 0)
    {
        ROCRAND_CHECK(
            rocrand_set_quasi_random_generator_dimensions(generator, test_case.qrng_dimensions));
    }

    if(test_case.offset >= 0)
    {
        ROCRAND_CHECK(rocrand_set_offset(generator, test_case.offset));
    }

    ROCRAND_CHECK(callback(generator, data, test_case.size));

    ROCRAND_CHECK(rocrand_destroy_generator(generator));

    std::vector<T> results(test_case.size);
    HIP_CHECK(hipMemcpy(results.data(), data, test_case.size * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(data));

    return results;
}

std::vector<unsigned int> test_rocrand_generate(const test_case& test_case)
{
    return generate<unsigned int>(test_case, rocrand_generate);
}

std::vector<unsigned long long> test_rocrand_generate_long_long(const test_case& test_case)
{
    return generate<unsigned long long>(test_case, rocrand_generate_long_long);
}
