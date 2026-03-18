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

#pragma once

#include "primbench.hpp"

#include <curand.h>

#include <iostream>
#include <string>

#define CURAND_CHECK(condition)                                                                \
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

#define CUDA_CHECK(condition)                                                               \
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
    return "unknown";
}

PRIMBENCH_REGISTER_TYPE(uint32_t, "u32")
PRIMBENCH_REGISTER_TYPE(unsigned long long, "u64")
PRIMBENCH_REGISTER_TYPE(float, "f32")
PRIMBENCH_REGISTER_TYPE(double, "f64")
