// Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#elif defined(__CUDACC__)
    #include <curand_kernel.h>
#endif

#ifdef __HIP__
    #define RAND_CHECK(condition)                                                                  \
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
#elif defined(__CUDACC__)
    #define RAND_CHECK(condition)                                                                 \
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
#endif

#ifdef __HIP__
using stream_t                     = hipStream_t;
using rng_type_t                   = rocrand_rng_type;
using ordering_t                   = rocrand_ordering;
using generator_t                  = rocrand_generator;
using memcpy_kind_t                = hipMemcpyKind;
using rand_discrete_distribution_t = rocrand_discrete_distribution;
using rand_direction_vector_set_t  = rocrand_direction_vector_set;
using direction_vectors32_t        = const unsigned int;
using direction_vectors64_t        = const unsigned long long;
#elif defined(__CUDACC__)
using stream_t                     = cudaStream_t;
using rng_type_t                   = curandRngType;
using ordering_t                   = curandOrdering;
using generator_t                  = curandGenerator_t;
using memcpy_kind_t                = cudaMemcpyKind;
using rand_discrete_distribution_t = curandDiscreteDistribution_t;
using rand_direction_vector_set_t  = curandDirectionVectorSet_t;
using direction_vectors32_t        = curandDirectionVectors32_t;
using direction_vectors64_t        = curandDirectionVectors64_t;
#endif

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
#elif defined(__CUDACC__)
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
#endif
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

inline auto create_generator(generator_t* generator, rng_type_t rng_type)
{
    return DISPATCH(rocrand_create_generator, curandCreateGenerator)(generator, rng_type);
}

inline auto create_generator_host(generator_t* generator, rng_type_t rng_type)
{
    return DISPATCH(rocrand_create_generator_host, curandCreateGeneratorHost)(generator, rng_type);
}

inline auto destroy_generator(generator_t generator)
{
    return DISPATCH(rocrand_destroy_generator, curandDestroyGenerator)(generator);
}

inline auto gpu_device_synchronize()
{
    return DISPATCH(hipDeviceSynchronize, cudaDeviceSynchronize)();
}

inline auto gpu_free(void* device)
{
    return DISPATCH(hipFree, cudaFree)(device);
}

inline auto gpu_get_last_error()
{
    return DISPATCH(hipGetLastError, cudaGetLastError)();
}

template<typename T>
inline auto gpu_malloc(T** device, size_t size)
{
    return DISPATCH(hipMalloc, cudaMalloc)(device, size);
}

inline auto gpu_memcpy(void* dst, const void* src, size_t count, memcpy_kind_t kind)
{
    return DISPATCH(hipMemcpy, cudaMemcpy)(dst, src, count, kind);
}

template<typename Engine>
__forceinline__ __device__ __host__
auto gpu_rand(Engine* state)
{
    return DISPATCH(rocrand, curand)(state);
}

template<typename Engine>
__forceinline__ __device__ __host__
auto gpu_rand_log_normal(Engine* state, float mean, float stddev)
{
    return DISPATCH(rocrand_log_normal, curand_log_normal)(state, mean, stddev);
}

template<typename Engine>
__forceinline__ __device__ __host__
auto gpu_rand_log_normal_double(Engine* state, double mean, double stddev)
{
    return DISPATCH(rocrand_log_normal_double, curand_log_normal_double)(state, mean, stddev);
}

template<typename Engine>
__forceinline__ __device__ __host__
auto gpu_rand_normal(Engine* state)
{
    return DISPATCH(rocrand_normal, curand_normal)(state);
}

template<typename Engine>
__forceinline__ __device__ __host__
auto gpu_rand_normal_double(Engine* state)
{
    return DISPATCH(rocrand_normal_double, curand_normal_double)(state);
}

template<typename Engine>
__forceinline__ __device__ __host__
auto gpu_rand_poisson(Engine* state, double lambda)
{
    return DISPATCH(rocrand_poisson, curand_poisson)(state, lambda);
}

template<typename Engine>
__forceinline__ __device__ __host__
auto gpu_rand_uniform(Engine* state)
{
    return DISPATCH(rocrand_uniform, curand_uniform)(state);
}

template<typename Engine>
__forceinline__ __device__ __host__
auto gpu_rand_uniform_double(Engine* state)
{
    return DISPATCH(rocrand_uniform_double, curand_uniform_double)(state);
}

inline auto rand_create_poisson_distribution(double                        lambda,
                                             rand_discrete_distribution_t* discrete_distribution)
{
    return DISPATCH(rocrand_create_poisson_distribution,
                    curandCreatePoissonDistribution)(lambda, discrete_distribution);
}

inline auto rand_destroy_discrete_distribution(rand_discrete_distribution_t discrete_distribution)
{
    return DISPATCH(rocrand_destroy_discrete_distribution,
                    curandDestroyDistribution)(discrete_distribution);
}

template<typename Engine>
__forceinline__ __device__ __host__
auto rand_discrete(Engine* state, const rand_discrete_distribution_t discrete_distribution)
{
    return DISPATCH(rocrand_discrete, curand_discrete)(state, discrete_distribution);
}

inline auto rand_get_direction_vectors32(direction_vectors32_t**     vectors,
                                         rand_direction_vector_set_t set)
{
    return DISPATCH(rocrand_get_direction_vectors32, curandGetDirectionVectors32)(vectors, set);
}

inline auto rand_get_direction_vectors64(direction_vectors64_t**     vectors,
                                         rand_direction_vector_set_t set)
{
    return DISPATCH(rocrand_get_direction_vectors64, curandGetDirectionVectors64)(vectors, set);
}

inline auto rand_get_scramble_constants32(const unsigned int** constants)
{
#ifdef __HIP__
    return rocrand_get_scramble_constants32(constants);
#else
    return curandGetScrambleConstants32(const_cast<unsigned int**>(constants));
#endif
}

inline auto rand_get_scramble_constants64(const unsigned long long** constants)
{
#ifdef __HIP__
    return rocrand_get_scramble_constants64(constants);
#else
    return curandGetScrambleConstants64(const_cast<unsigned long long**>(constants));
#endif
}

// Perfect forwarding with variadic templates
// handles overloads of rocrand_init() and curand_init().
template<typename... Args>
__forceinline__ __device__ __host__
void rand_init(Args&&... args)
{
    DISPATCH(rocrand_init, curand_init)(std::forward<Args>(args)...);
}

inline auto set_offset(generator_t generator, unsigned long long offset)
{
    return DISPATCH(rocrand_set_offset, curandSetGeneratorOffset)(generator, offset);
}

inline auto set_ordering(generator_t generator, ordering_t order)
{
    return DISPATCH(rocrand_set_ordering, curandSetGeneratorOrdering)(generator, order);
}

inline auto set_quasi_random_generator_dimensions(generator_t generator, unsigned int dimensions)
{
    return DISPATCH(rocrand_set_quasi_random_generator_dimensions,
                    curandSetQuasiRandomGeneratorDimensions)(generator, dimensions);
}

inline auto set_stream(generator_t generator, stream_t stream)
{
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
