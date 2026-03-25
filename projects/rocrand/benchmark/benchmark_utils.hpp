// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifdef __HIP__
#include <rocrand/rocrand_kernel.h>
#else // __HIP__
#include <curand_kernel.h>
#endif // __HIP__

#ifdef __HIP__
    #define CHECK(condition)                                                               \
        do                                                                                         \
        {                                                                                          \
            rocrand_status status_ = condition;                                                    \
            if(status_ != ROCRAND_STATUS_SUCCESS)                                                  \
            {                                                                                      \
                std::cout << "ROCRAND error: " << status_ << " at " << __FILE__ << ":" << __LINE__ \
                        << std::endl;                                                            \
                exit(status_);                                                                     \
            }                                                                                      \
        }                                                                                          \
        while(0)
#else // __HIP__
    #define CHECK(condition)                                                                \
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
#endif // __HIP__

#ifdef __HIP__
using gpu_error_t = hipError_t;
using stream_t = hipStream_t;
using rng_type_t = rocrand_rng_type;
using ordering_t = rocrand_ordering;
using generator_t = rocrand_generator;
#else // __HIP__
using gpu_error_t = cudaError_t;
using stream_t = cudaStream_t;
using rng_type_t = curandRngType;
using ordering_t = curandOrdering;
using generator_t = curandGenerator_t;
#endif // __HIP__

inline std::string engine_name(const rng_type_t rng_type)
{
    // The returned names have to be able to reproduce the rocrand_rng_type by prepending
    // `ROCRAND_RNG_{PSEUDO|QUASI}_` to the name written in all capital letters. The scripts in
    // scripts/config-tuning/ rely on this.
    // clang-format off
    switch(rng_type)
    {
#ifdef __HIP__
        case ROCRAND_RNG_PSEUDO_XORWOW:           return "xorwow";
        case ROCRAND_RNG_PSEUDO_MRG32K3A:         return "mrg32k3a";
        case ROCRAND_RNG_PSEUDO_MTGP32:           return "mtgp32";
        case ROCRAND_RNG_PSEUDO_PHILOX4_32_10:    return "philox4_32_10";
        case ROCRAND_RNG_PSEUDO_MRG31K3P:         return "mrg31k3p";
        case ROCRAND_RNG_PSEUDO_LFSR113:          return "lfsr113";
        case ROCRAND_RNG_PSEUDO_MT19937:          return "mt19937";
        case ROCRAND_RNG_PSEUDO_THREEFRY2_32_20:  return "threefry2_32_20";
        case ROCRAND_RNG_PSEUDO_THREEFRY2_64_20:  return "threefry2_64_20";
        case ROCRAND_RNG_PSEUDO_THREEFRY4_32_20:  return "threefry4_32_20";
        case ROCRAND_RNG_PSEUDO_THREEFRY4_64_20:  return "threefry4_64_20";
        case ROCRAND_RNG_QUASI_SOBOL32:           return "sobol32";
        case ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32: return "scrambled_sobol32";
        case ROCRAND_RNG_QUASI_SOBOL64:           return "sobol64";
        case ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64: return "scrambled_sobol64";
        case ROCRAND_RNG_PSEUDO_DEFAULT:          return "pseudo_default";
        case ROCRAND_RNG_QUASI_DEFAULT:           return "quasi_default";
#else // __HIP__
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
#endif // __HIP__
    }
    // clang-format on
    return "unknown";
}

enum distribution
{
    DISTRIBUTION_UNIFORM,
    DISTRIBUTION_NORMAL,
    DISTRIBUTION_LOG_NORMAL,
    DISTRIBUTION_POISSON,
    DISTRIBUTION_DISCRETE_POISSON,
    DISTRIBUTION_DISCRETE_CUSTOM,
};

constexpr const char* distribution_name(distribution d)
{
    switch(d)
    {
        case DISTRIBUTION_UNIFORM: return "uniform";
        case DISTRIBUTION_NORMAL: return "normal";
        case DISTRIBUTION_LOG_NORMAL: return "log_normal";
        case DISTRIBUTION_POISSON: return "poisson";
        case DISTRIBUTION_DISCRETE_POISSON: return "discrete_poisson";
        case DISTRIBUTION_DISCRETE_CUSTOM: return "discrete_custom";
    }
    return "unknown";
}

/// Backend-agnostic wrappers for rocRAND and cuRAND.
/// The functions in this namespace are sorted alphabetically.
namespace wrappers
{

#ifdef __HIP__
    #define DISPATCH(rocrand_fn, curand_fn) rocrand_fn
#else
    #define DISPATCH(rocrand_fn, curand_fn) curand_fn
#endif

inline auto create_generator(generator_t* generator, rng_type_t rng_type) {
    return DISPATCH(rocrand_create_generator, curandCreateGenerator)(generator, rng_type);
}

inline auto create_generator_host(generator_t* generator, rng_type_t rng_type) {
    return DISPATCH(rocrand_create_generator_host, curandCreateGeneratorHost)(generator, rng_type);
}

inline auto destroy_generator(generator_t generator) {
    return DISPATCH(rocrand_destroy_generator, curandDestroyGenerator)(generator);
}

inline auto gpu_free(void* device) {
    return DISPATCH(hipFree, cudaFree)(device);
}

template <typename T>
inline auto gpu_malloc(T** device, size_t size) {
    return DISPATCH(hipMalloc, cudaMalloc)(device, size);
}

inline auto set_offset(generator_t generator, unsigned long long offset) {
    return DISPATCH(rocrand_set_offset, curandSetGeneratorOffset)(generator, offset);
}

inline auto set_ordering(generator_t generator, ordering_t order) {
    return DISPATCH(rocrand_set_ordering, curandSetGeneratorOrdering)(generator, order);
}

inline auto set_quasi_random_generator_dimensions(generator_t generator, unsigned int dimensions) {
    return DISPATCH(rocrand_set_quasi_random_generator_dimensions, curandSetQuasiRandomGeneratorDimensions)(generator, dimensions);
}

inline auto set_stream(generator_t generator, stream_t stream) {
    return DISPATCH(rocrand_set_stream, curandSetStream)(generator, stream);
}

#undef DISPATCH

} // namespace wrappers

/// Bring wrappers into this namespace, so we can call
/// functions like gpu_malloc() without the wrappers:: prefix.
using namespace wrappers;

PRIMBENCH_REGISTER_TYPE(uint8_t, "u8")
PRIMBENCH_REGISTER_TYPE(uint16_t, "u16")
PRIMBENCH_REGISTER_TYPE(uint32_t, "u32")
PRIMBENCH_REGISTER_TYPE(unsigned long long, "u64")
PRIMBENCH_REGISTER_TYPE(float, "f32")
PRIMBENCH_REGISTER_TYPE(double, "f64")
PRIMBENCH_REGISTER_TYPE(__half, "half")
