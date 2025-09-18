// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <limits>
#include <type_traits>
#include <vector>

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>
#include <stdio.h>

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>
#include <hiprand/hiprand_mtgp32_host.h>

#if defined(__HIP_PLATFORM_AMD__)
    #include <rocrand/rocrand_mtgp32_11213.h>
#else // for HIP NVCC platfrom
    #include <curand_mtgp32dc_p_11213.h>
#endif // __HIP_PLATFORM_AMD__

#include "test_common.hpp"

template<class T>
struct is_sobol_scrambled
    : std::integral_constant<bool,
                             std::is_same<T, hiprandStateScrambledSobol32>::value
                                 || std::is_same<T, hiprandStateScrambledSobol64>::value>
{};

template<class T>
struct is_sobol_32
    : std::integral_constant<bool,
                             std::is_same<T, hiprandStateSobol32>::value
                                 || std::is_same<T, hiprandStateScrambledSobol32>::value>
{};

template<class S>
hiprandStatus_t get_sobol_directions(hiprandDirectionVectors32_t** vector)
{
    constexpr auto set = is_sobol_scrambled<S>::value
                             ? HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6
                             : HIPRAND_DIRECTION_VECTORS_32_JOEKUO6;
    return hiprandGetDirectionVectors32(vector, set);
}

// Utility type to easily convert from
//   unsigned int       => hiprandDirectionVectors32_t (aka. unsigned int[32])
//   unsigned long long => hiprandDirectionVectors64_t (aka. unsigned long long[64])
template<typename T>
using DirectionVectors_t = T[sizeof(T) * 8];

template<class S>
hiprandStatus_t get_sobol_directions(hiprandDirectionVectors64_t** vector)
{
    constexpr auto set = is_sobol_scrambled<S>::value
                             ? HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6
                             : HIPRAND_DIRECTION_VECTORS_64_JOEKUO6;
    return hiprandGetDirectionVectors64(vector, set);
}

hiprandStatus_t get_sobol_constants(const unsigned int** vector)
{
    return hiprandGetScrambleConstants32(vector);
}

hiprandStatus_t get_sobol_constants(const unsigned long long** vector)
{
    return hiprandGetScrambleConstants64(vector);
}

template<class S, class T>
void load_sobol_vectors_to_gpu(const unsigned int      dimensions,
                               DirectionVectors_t<T>** direction_vectors)
{
    DirectionVectors_t<T>* h_directions;

    constexpr unsigned int vec_size = sizeof(T) * 8;

    HIPRAND_CHECK(get_sobol_directions<S>(&h_directions));

    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(direction_vectors),
                              sizeof(T) * dimensions * vec_size));
    HIP_CHECK(hipMemcpy(*direction_vectors,
                        h_directions,
                        sizeof(T) * dimensions * vec_size,
                        hipMemcpyHostToDevice));
}

template<class S, class T>
void load_scrambled_sobol_constants_and_vectors_to_gpu(const unsigned int      dimensions,
                                                       DirectionVectors_t<T>** direction_vectors,
                                                       T**                     scramble_constants)
{
    load_sobol_vectors_to_gpu<S, T>(dimensions, direction_vectors);

    const T* h_constants;

    HIPRAND_CHECK(get_sobol_constants(&h_constants));

    HIP_CHECK(
        hipMallocHelper(reinterpret_cast<void**>(scramble_constants), sizeof(T) * dimensions));
    HIP_CHECK(
        hipMemcpy(*scramble_constants, h_constants, sizeof(T) * dimensions, hipMemcpyHostToDevice));
}

template <class GeneratorState>
__global__
__launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU)
void hiprand_init_kernel(GeneratorState * states,
                         const size_t states_size,
                         unsigned long long seed,
                         unsigned long long offset)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int subsequence = state_id;
    if(state_id < states_size)
    {
        GeneratorState state;
        hiprand_init(seed, subsequence, offset, &state);
        states[state_id] = state;
    }
}

template<class GeneratorState, typename direction_vectors>
__global__ __launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU) void hiprand_init_sobol_kernel(
    GeneratorState*    states,
    const size_t       states_size,
    unsigned long long offset,
    direction_vectors* dir_vecs)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(state_id < states_size)
    {
        GeneratorState state;
        hiprand_init(*dir_vecs, offset, &state);
        states[state_id] = state;
    }
}

template <class GeneratorState>
__global__
__launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU)
void hiprand_skip_kernel(GeneratorState * states,
                         const size_t states_size,
                         unsigned long long seed,
                         unsigned long long offset)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int subsequence = state_id;
    if(state_id < states_size)
    {
        GeneratorState state;
        hiprand_init(seed, 0, offset, &state);
        skipahead(1ULL, &state);
        skipahead_sequence(subsequence, &state);
        states[state_id] = state;
    }
}

template <class GeneratorState>
__global__
__launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU)
void hiprand_kernel(unsigned int * output, const size_t size)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState state;
    const unsigned int subsequence = state_id;
    hiprand_init(12345, subsequence, 0, &state);

    unsigned int index = state_id;
    const size_t r = size%hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = hiprand(&state);
        if(index < size)
            output[index] = value;
        index += global_size;
    }
}

template<class GeneratorState, typename direction_vectors, typename T>
__global__ __launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU) void hiprand_sobol_kernel(
    T* output, const size_t size, direction_vectors* dir_vecs)
{
    const unsigned int state_id    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState state;
    hiprand_init(*dir_vecs, state_id, &state);

    unsigned int index           = state_id;
    const size_t r               = size % hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = sizeof(T) == 4 ? hiprand(&state) : hiprand_long_long(&state);
        if(index < size)
            output[index] = value;
        index += global_size;
        skipahead(global_size - 1, &state);
    }
}

template<class GeneratorState>
__global__ __launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU) void hiprand_mtgp_kernel(
    GeneratorState* states, unsigned int* output, const size_t size)
{
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    __shared__ GeneratorState state;
    if(threadIdx.x == 0)
    {
        state = states[blockIdx.x];
    }
    __syncthreads();

    unsigned int index           = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const size_t r               = size % hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = hiprand(&state);
        if(index < size)
            output[index] = value;
        index += global_size;
    }
}

template<class T, class StateType>
__forceinline__ __device__ auto hiprand_auto(StateType* state) ->
    typename std::enable_if<std::is_same<T, unsigned int>::value, unsigned int>::type
{
    return hiprand(state);
}

template<class T, class StateType>
__forceinline__ __device__ auto hiprand_auto(StateType* state) ->
    typename std::enable_if<std::is_same<T, unsigned long long>::value, unsigned long long>::type
{
    return hiprand_long_long(state);
}

template<class GeneratorState, class IntType>
__global__ __launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU) void hiprand_sobol_kernel(
    IntType* output, DirectionVectors_t<IntType>* vectors, const size_t items_per_dimension)
{
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int dimension = blockIdx.y;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    const unsigned int threads_per_dim = gridDim.x * blockDim.x;
    const unsigned int items_per_thread
        = (items_per_dimension + threads_per_dim - 1) / threads_per_dim;

    const unsigned int block_offset  = state_id * items_per_thread;
    const unsigned int global_offset = dimension * items_per_dimension;

    GeneratorState state;
    hiprand_init(vectors[dimension], block_offset, &state);

    for(unsigned int i = 0; block_offset + i < items_per_dimension; i++)
    {
        output[global_offset + block_offset + i] = hiprand_auto<IntType>(&state);
        skipahead(global_size - 1, &state);
    }
}

template<class GeneratorState, class IntType>
__global__
    __launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU) void hiprand_scrambled_sobol_kernel(
        IntType*                     output,
        DirectionVectors_t<IntType>* vectors,
        IntType*                     constants,
        const size_t                 items_per_dimension)
{
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int dimension = blockIdx.y;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    const unsigned int threads_per_dim = gridDim.x * blockDim.x;
    const unsigned int items_per_thread
        = (items_per_dimension + threads_per_dim - 1) / threads_per_dim;

    const unsigned int block_offset  = state_id * items_per_thread;
    const unsigned int global_offset = dimension * items_per_dimension;
    GeneratorState     state;
    hiprand_init(vectors[dimension], constants[dimension], block_offset, &state);

    for(unsigned int i = 0; block_offset + i < items_per_dimension; i++)
    {
        output[global_offset + block_offset + i] = hiprand_auto<IntType>(&state);
        skipahead(global_size - 1, &state);
    }
}

template <class GeneratorState>
__global__
__launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU)
void hiprand_uniform_kernel(float * output, const size_t size)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState state;
    const unsigned int subsequence = state_id;
    hiprand_init(12345, subsequence, 0, &state);

    unsigned int index = state_id;
    const size_t r = size%hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = hiprand_uniform(&state);
        if(index < size)
            output[index] = value;
        index += global_size;
    }
}

template<class GeneratorState, typename direction_vectors>
__global__
    __launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU) void hiprand_uniform_sobol_kernel(
        float* output, const size_t size, direction_vectors* dir_vecs)
{
    const unsigned int state_id    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState state;
    hiprand_init(*dir_vecs, state_id, &state);

    unsigned int index           = state_id;
    const size_t r               = size % hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = hiprand_uniform(&state);
        if(index < size)
            output[index] = value;
        index += global_size;
        skipahead(global_size - 1, &state);
    }
}

template<class GeneratorState>
__global__ __launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU) void hiprand_uniform_mtgp_kernel(
    GeneratorState* states, float* output, const size_t size)
{
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    __shared__ GeneratorState state;
    if(threadIdx.x == 0)
    {
        state = states[blockIdx.x];
    }
    __syncthreads();

    unsigned int index           = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const size_t r               = size % hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = hiprand_uniform(&state);
        if(index < size)
            output[index] = value;
        index += global_size;
    }
}

template <class GeneratorState>
__global__
__launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU)
void hiprand_normal_kernel(float * output, const size_t size)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState state;
    const unsigned int subsequence = state_id;
    hiprand_init(12345, subsequence, 0, &state);

    unsigned int index = state_id;
    const size_t r = size%hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        float value;
        if(hipBlockIdx_x % 2 == 0)
            value = hiprand_normal2(&state).x;
        else
            value = hiprand_normal(&state);

        if(index < size)
            output[index] = value;
        index += global_size;
    }
}

template<class GeneratorState, typename direction_vectors>
__global__ __launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU) void hiprand_normal_sobol_kernel(
    float* output, const size_t size, direction_vectors* dir_vecs)
{
    const unsigned int state_id    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState state;
    hiprand_init(*dir_vecs, state_id, &state);

    unsigned int index           = state_id;
    const size_t r               = size % hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        float value = hiprand_normal(&state);

        if(index < size)
            output[index] = value;
        index += global_size;
        skipahead(global_size - 1, &state);
    }
}

template<class GeneratorState>
__global__ __launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU) void hiprand_normal_mtgp_kernel(
    GeneratorState* states, float* output, const size_t size)
{
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    __shared__ GeneratorState state;
    if(threadIdx.x == 0)
    {
        state = states[blockIdx.x];
    }
    __syncthreads();

    unsigned int index           = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const size_t r               = size % hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        float value = hiprand_normal(&state);

        if(index < size)
            output[index] = value;
        index += global_size;
    }
}

template <class GeneratorState>
__global__
__launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU)
void hiprand_log_normal_kernel(float * output, const size_t size)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState state;
    const unsigned int subsequence = state_id;
    hiprand_init(12345, subsequence, 0, &state);

    unsigned int index = state_id;
    const size_t r = size%hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        float value;
        if(hipBlockIdx_x % 2 == 0)
            value = hiprand_log_normal2(&state, 1.6f, 0.25f).x;
        else
            value = hiprand_log_normal(&state, 1.6f, 0.25f);

        if(index < size)
            output[index] = value;
        index += global_size;
    }
}

template<class GeneratorState, typename direction_vectors>
__global__
    __launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU) void hiprand_log_normal_sobol_kernel(
        float* output, const size_t size, direction_vectors* dir_vecs)
{
    const unsigned int state_id    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState state;
    hiprand_init(*dir_vecs, state_id, &state);

    unsigned int index           = state_id;
    const size_t r               = size % hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        float value = hiprand_log_normal(&state, 1.6f, 0.25f);

        if(index < size)
            output[index] = value;
        index += global_size;
        skipahead(global_size - 1, &state);
    }
}

template<class GeneratorState>
__global__
    __launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU) void hiprand_log_normal_mtgp_kernel(
        GeneratorState* states, float* output, const size_t size)
{
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    __shared__ GeneratorState state;
    if(threadIdx.x == 0)
    {
        state = states[blockIdx.x];
    }
    __syncthreads();

    unsigned int index           = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const size_t r               = size % hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        float value = hiprand_log_normal(&state, 1.6f, 0.25f);

        if(index < size)
            output[index] = value;
        index += global_size;
    }
}

template<class GeneratorState>
__global__ __launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU) void hiprand_poisson_kernel(
    unsigned int* output, const size_t size, double lambda)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState state;
    const unsigned int subsequence = state_id;
    hiprand_init(12345, subsequence, 0, &state);

    unsigned int index = state_id;
    const size_t r = size%hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = hiprand_poisson(&state, lambda);
        if(index < size)
            output[index] = value;
        index += global_size;
    }
}

template<class GeneratorState, typename direction_vectors>
__global__
    __launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU) void hiprand_poisson_sobol_kernel(
        unsigned int* output, const size_t size, double lambda, direction_vectors* dir_vecs)
{
    const unsigned int state_id    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState state;
    hiprand_init(*dir_vecs, state_id, &state);

    unsigned int index           = state_id;
    const size_t r               = size % hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = hiprand_poisson(&state, lambda);
        if(index < size)
            output[index] = value;
        index += global_size;
        skipahead(global_size - 1, &state);
    }
}

template<class GeneratorState>
__global__ __launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU) void hiprand_poisson_mtgp_kernel(
    GeneratorState* states, unsigned int* output, const size_t size, double lambda)
{
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    __shared__ GeneratorState state;
    if(threadIdx.x == 0)
    {
        state = states[blockIdx.x];
    }
    __syncthreads();

    unsigned int index           = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const size_t r               = size % hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = hiprand_poisson(&state, lambda);
        if(index < size)
            output[index] = value;
        index += global_size;
    }
}

template<class GeneratorState>
__global__ __launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU) void hiprand_discrete_kernel(
    unsigned int* output, const size_t size, hiprandDiscreteDistribution_t discrete_distribution)
{
    const unsigned int state_id    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState     state;
    const unsigned int subsequence = state_id;
    hiprand_init(12345, subsequence, 0, &state);

    unsigned int index           = state_id;
    const size_t r               = size % hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = hiprand_discrete(&state, discrete_distribution);
        if(index < size)
            output[index] = value;
        index += global_size;
    }
}

template<class GeneratorState, typename direction_vectors>
__global__
    __launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU) void hiprand_discrete_sobol_kernel(
        unsigned int*                 output,
        const size_t                  size,
        hiprandDiscreteDistribution_t discrete_distribution,
        direction_vectors*            dir_vecs)
{
    const unsigned int state_id    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState state;
    hiprand_init(*dir_vecs, state_id, &state);

    unsigned int index           = state_id;
    const size_t r               = size % hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = hiprand_discrete(&state, discrete_distribution);
        if(index < size)
            output[index] = value;
        index += global_size;
        skipahead(global_size - 1, &state);
    }
}

template<class GeneratorState>
__global__
    __launch_bounds__(64, HIPRAND_DEFAULT_MIN_WARPS_PER_EU) void hiprand_discrete_mtgp_kernel(
        GeneratorState*               states,
        unsigned int*                 output,
        const size_t                  size,
        hiprandDiscreteDistribution_t discrete_distribution)
{
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    __shared__ GeneratorState state;
    if(threadIdx.x == 0)
    {
        state = states[blockIdx.x];
    }
    __syncthreads();

    unsigned int index           = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const size_t r               = size % hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = hiprand_discrete(&state, discrete_distribution);
        if(index < size)
            output[index] = value;
        index += global_size;
    }
}

template<class T>
void hiprand_kernel_h_hiprand_init_test()
{
    typedef T state_type;

    unsigned long long seed   = 0xdeadbeefbeefdeadULL;
    unsigned long long offset = 4;

    const size_t states_size = 256;
    state_type*  states;
    HIP_CHECK(hipMallocHelper((void**)&states, states_size * sizeof(state_type)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_init_kernel<<<dim3(4), dim3(64), 0, 0>>>(states, states_size, seed, offset);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(states));
}

template<class state_type>
void hiprand_kernel_h_hiprand_sobol_init_test()
{
    using int_type = typename std::
        conditional<is_sobol_32<state_type>::value, unsigned int, unsigned long long int>::type;

    const unsigned int dimensions = 8;

    DirectionVectors_t<int_type>* d_vector;
    int_type*                     d_constants;

    if(is_sobol_scrambled<state_type>::value)
    {
        load_scrambled_sobol_constants_and_vectors_to_gpu<state_type, int_type>(dimensions,
                                                                                &d_vector,
                                                                                &d_constants);
    }
    else
    {
        load_sobol_vectors_to_gpu<state_type, int_type>(dimensions, &d_vector);
        d_constants = NULL;
    }

    unsigned long long offset = 4;

    const size_t states_size = 256;
    state_type*  states;
    HIP_CHECK(hipMallocHelper((void**)&states, states_size * sizeof(state_type)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_init_sobol_kernel<<<dim3(4), dim3(64), 0, 0>>>(states, states_size, offset, d_vector);

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(states));
}

template<class state_type>
void hiprand_kernel_h_hiprand_mtgp_init_test()
{
    const size_t states_size = 128;
    state_type*  states;
    HIP_CHECK(hipMallocHelper((void**)&states, states_size * sizeof(state_type)));
    mtgp32_kernel_params_t* k;
    HIP_CHECK(hipMallocHelper((void**)&k, states_size * sizeof(mtgp32_kernel_params_t)));

    HIPRAND_CHECK(hiprandMakeMTGP32Constants(mtgp32dc_params_fast_11213, k));

    unsigned long long seed = 0;
    HIPRAND_CHECK(
        hiprandMakeMTGP32KernelState(states, mtgp32dc_params_fast_11213, k, states_size, seed));

    HIP_CHECK(hipFree(states));
}

TEST(hiprand_kernel_h_philox4x32_10, hiprand_init)
{
    typedef hiprandStatePhilox4_32_10_t state_type;
    hiprand_kernel_h_hiprand_init_test<state_type>();
}

TEST(hiprand_kernel_h_mrg32k3a, hiprand_init)
{
    typedef hiprandStateMRG32k3a_t state_type;
    hiprand_kernel_h_hiprand_init_test<state_type>();
}

TEST(hiprand_kernel_h_xorwow, hiprand_init)
{
    typedef hiprandStateXORWOW_t state_type;
    hiprand_kernel_h_hiprand_init_test<state_type>();
}

TEST(hiprand_kernel_h_default, hiprand_init)
{
    typedef hiprandState_t state_type;
    hiprand_kernel_h_hiprand_init_test<state_type>();
}

TEST(hiprand_kernel_h_sobol_32, hiprand_init)
{
    typedef hiprandStateSobol32_t state_type;
    hiprand_kernel_h_hiprand_sobol_init_test<state_type>();
}

TEST(hiprand_kernel_h_sobol_64, hiprand_init)
{
    typedef hiprandStateSobol64_t state_type;
    hiprand_kernel_h_hiprand_sobol_init_test<state_type>();
}

TEST(hiprand_kernel_h_mtgp_32, hiprand_init)
{
    typedef hiprandStateMtgp32_t state_type;
    hiprand_kernel_h_hiprand_mtgp_init_test<state_type>();
}

#ifdef __HIP_PLATFORM_NVIDIA__
TEST(hiprand_kernel_h_philox4x32_10, hiprand_init_nvcc)
{
    typedef hiprandStatePhilox4_32_10_t state_type;

    unsigned long long seed = 0xdeadbeefbeefdeadULL;
    unsigned long long offset = 4 * ((UINT_MAX * 17ULL) + 17);

    const size_t states_size = 256;
    state_type * states;
    HIP_CHECK(hipMallocHelper((void **)&states, states_size * sizeof(state_type)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_init_kernel<<<dim3(4), dim3(64), 0, 0>>>(states, states_size, seed, offset);
    HIP_CHECK(hipGetLastError());

    std::vector<state_type> states_host(states_size);
    HIP_CHECK(
        hipMemcpy(
            states_host.data(), states,
            states_size * sizeof(state_type),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(states));

    unsigned int subsequence = 0;
    for(auto& s : states_host)
    {
        EXPECT_EQ(s.key.x, 0xbeefdeadU);
        EXPECT_EQ(s.key.y, 0xdeadbeefU);

        EXPECT_EQ(s.ctr.x, 0U);
        EXPECT_EQ(s.ctr.y, 17U);
        EXPECT_EQ(s.ctr.z, subsequence);
        EXPECT_EQ(s.ctr.w, 0U);

        EXPECT_TRUE(
            s.output.x != 0U
            || s.output.y != 0U
            || s.output.z != 0U
            || s.output.w
        );

        EXPECT_EQ(s.STATE, 0U);

        subsequence++;
    }
}

TEST(hiprand_kernel_h_philox4x32_10, hiprand_skip_nvcc)
{
    typedef hiprandStatePhilox4_32_10_t state_type;

    unsigned long long seed = 0xdeadbeefbeefdeadULL;
    unsigned long long offset = 4 * ((UINT_MAX * 17ULL) + 17);

    const size_t states_size = 256;
    state_type * states;
    HIP_CHECK(hipMallocHelper((void **)&states, states_size * sizeof(state_type)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_skip_kernel<<<dim3(4), dim3(64), 0, 0>>>(states, states_size, seed, offset);
    HIP_CHECK(hipGetLastError());

    std::vector<state_type> states_host(states_size);
    HIP_CHECK(
        hipMemcpy(
            states_host.data(), states,
            states_size * sizeof(state_type),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(states));

    unsigned int subsequence = 0;
    for(auto& s : states_host)
    {
        EXPECT_EQ(s.ctr.x, 0U);
        EXPECT_EQ(s.ctr.y, 17U);
        EXPECT_EQ(s.ctr.z, subsequence);
        EXPECT_EQ(s.ctr.w, 0U);
        EXPECT_EQ(s.STATE, 1U);
        subsequence++;
    }
}
#endif

template<class T>
void hiprand_kernel_h_hiprand_test()
{
    typedef T state_type;

    const size_t output_size = 8192;
    unsigned int * output;
    HIP_CHECK(hipMallocHelper((void **)&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_kernel<state_type><<<dim3(4), dim3(64), 0, 0>>>(output, output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(unsigned int),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v) / UINT_MAX;
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

template<class state_type>
void hiprand_kernel_h_hiprand_sobol_test()
{
    using int_type = typename std::
        conditional<is_sobol_32<state_type>::value, unsigned int, unsigned long long int>::type;

    const unsigned int dimensions = 8;

    DirectionVectors_t<int_type>* d_vector;
    int_type*                     d_constants;

    if(is_sobol_scrambled<state_type>::value)
    {
        load_scrambled_sobol_constants_and_vectors_to_gpu<state_type, int_type>(dimensions,
                                                                                &d_vector,
                                                                                &d_constants);
    }
    else
    {
        load_sobol_vectors_to_gpu<state_type, int_type>(dimensions, &d_vector);
        d_constants = NULL;
    }

    const size_t output_size = 8192;
    int_type*    output;
    HIP_CHECK(hipMallocHelper((void**)&output, output_size * sizeof(int_type)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_sobol_kernel<state_type><<<dim3(4), dim3(64), 0, 0>>>(output, output_size, d_vector);
    HIP_CHECK(hipGetLastError());

    std::vector<int_type> output_host(output_size);
    HIP_CHECK(hipMemcpy(output_host.data(),
                        output,
                        output_size * sizeof(int_type),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v)
                / static_cast<double>(is_sobol_32<state_type>::value ? UINT_MAX : ULONG_LONG_MAX);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

template<class state_type>
void hiprand_kernel_h_hiprand_mtgp_test()
{
    const size_t states_size = 128;
    state_type*  states;
    HIP_CHECK(hipMallocHelper((void**)&states, states_size * sizeof(state_type)));
    mtgp32_kernel_params_t* k;
    HIP_CHECK(hipMallocHelper((void**)&k, states_size * sizeof(mtgp32_kernel_params_t)));

    HIPRAND_CHECK(hiprandMakeMTGP32Constants(mtgp32dc_params_fast_11213, k));

    unsigned long long seed = 0;
    HIPRAND_CHECK(
        hiprandMakeMTGP32KernelState(states, mtgp32dc_params_fast_11213, k, states_size, seed));

    const size_t  output_size = 8192;
    unsigned int* output;
    HIP_CHECK(hipMallocHelper((void**)&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_mtgp_kernel<state_type><<<dim3(4), dim3(64), 0, 0>>>(states, output, output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(hipMemcpy(output_host.data(),
                        output,
                        output_size * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(states));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v) / UINT_MAX;
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

TEST(hiprand_kernel_h_philox4x32_10, hiprand)
{
    typedef hiprandStatePhilox4_32_10_t state_type;
    hiprand_kernel_h_hiprand_test<state_type>();
}

TEST(hiprand_kernel_h_mrg32k3a, hiprand)
{
    typedef hiprandStateMRG32k3a_t state_type;
    hiprand_kernel_h_hiprand_test<state_type>();
}

TEST(hiprand_kernel_h_xorwow, hiprand)
{
    typedef hiprandStateXORWOW_t state_type;
    hiprand_kernel_h_hiprand_test<state_type>();
}

TEST(hiprand_kernel_h_default, hiprand)
{
    typedef hiprandState_t state_type;
    hiprand_kernel_h_hiprand_test<state_type>();
}

TEST(hiprand_kernel_h_sobol_32, hiprand)
{
    typedef hiprandStateSobol32_t state_type;
    hiprand_kernel_h_hiprand_sobol_test<state_type>();
}

TEST(hiprand_kernel_h_sobol_64, hiprand)
{
    typedef hiprandStateSobol64_t state_type;
    hiprand_kernel_h_hiprand_sobol_test<state_type>();
}

TEST(hiprand_kernel_h_mtgp_32, hiprand)
{
    typedef hiprandStateMtgp32_t state_type;
    hiprand_kernel_h_hiprand_mtgp_test<state_type>();
}

template<class T>
void hiprand_kernel_h_hiprand_uniform_test()
{
    typedef T state_type;

    const size_t output_size = 8192;
    float * output;
    HIP_CHECK(hipMallocHelper((void **)&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_uniform_kernel<state_type><<<dim3(4), dim3(64), 0, 0>>>(output, output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(output_host.data(), output, output_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

template<class state_type>
void hiprand_kernel_h_hiprand_uniform_sobol_test()
{
    using int_type = typename std::
        conditional<is_sobol_32<state_type>::value, unsigned int, unsigned long long int>::type;

    const unsigned int dimensions = 8;

    DirectionVectors_t<int_type>* d_vector;
    int_type*                     d_constants;

    if(is_sobol_scrambled<state_type>::value)
    {
        load_scrambled_sobol_constants_and_vectors_to_gpu<state_type, int_type>(dimensions,
                                                                                &d_vector,
                                                                                &d_constants);
    }
    else
    {
        load_sobol_vectors_to_gpu<state_type, int_type>(dimensions, &d_vector);
        d_constants = NULL;
    }

    const size_t output_size = 8192;
    float*       output;
    HIP_CHECK(hipMallocHelper((void**)&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_uniform_sobol_kernel<state_type>
        <<<dim3(4), dim3(64), 0, 0>>>(output, output_size, d_vector);
    HIP_CHECK(hipGetLastError());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(output_host.data(), output, output_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

template<class state_type>
void hiprand_kernel_h_hiprand_uniform_mtgp_test()
{
    const size_t states_size = 128;
    state_type*  states;
    HIP_CHECK(hipMallocHelper((void**)&states, states_size * sizeof(state_type)));
    mtgp32_kernel_params_t* k;
    HIP_CHECK(hipMallocHelper((void**)&k, states_size * sizeof(mtgp32_kernel_params_t)));

    HIPRAND_CHECK(hiprandMakeMTGP32Constants(mtgp32dc_params_fast_11213, k));

    unsigned long long seed = 0;
    HIPRAND_CHECK(
        hiprandMakeMTGP32KernelState(states, mtgp32dc_params_fast_11213, k, states_size, seed));

    const size_t output_size = 8192;
    float*       output;
    HIP_CHECK(hipMallocHelper((void**)&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_uniform_mtgp_kernel<state_type>
        <<<dim3(4), dim3(64), 0, 0>>>(states, output, output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(output_host.data(), output, output_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(states));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

TEST(hiprand_kernel_h_philox4x32_10, hiprand_uniform)
{
    typedef hiprandStatePhilox4_32_10_t state_type;
    hiprand_kernel_h_hiprand_uniform_test<state_type>();
}

TEST(hiprand_kernel_h_mrg32k3a, hiprand_uniform)
{
    typedef hiprandStateMRG32k3a_t state_type;
    hiprand_kernel_h_hiprand_uniform_test<state_type>();
}

TEST(hiprand_kernel_h_xorwow, hiprand_uniform)
{
    typedef hiprandStateXORWOW_t state_type;
    hiprand_kernel_h_hiprand_uniform_test<state_type>();
}

TEST(hiprand_kernel_h_default, hiprand_uniform)
{
    typedef hiprandState_t state_type;
    hiprand_kernel_h_hiprand_uniform_test<state_type>();
}

TEST(hiprand_kernel_h_sobol_32, hiprand_uniform)
{
    typedef hiprandStateSobol32_t state_type;
    hiprand_kernel_h_hiprand_uniform_sobol_test<state_type>();
}

TEST(hiprand_kernel_h_sobol_64, hiprand_uniform)
{
    typedef hiprandStateSobol64_t state_type;
    hiprand_kernel_h_hiprand_uniform_sobol_test<state_type>();
}

TEST(hiprand_kernel_h_mtgp_32, hiprand_uniform)
{
    typedef hiprandStateMtgp32_t state_type;
    hiprand_kernel_h_hiprand_uniform_mtgp_test<state_type>();
}

template<class T>
void hiprand_kernel_h_hiprand_normal_test()
{
    typedef T state_type;

    const size_t output_size = 8192;
    float*       output;
    HIP_CHECK(hipMallocHelper((void**)&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_normal_kernel<state_type><<<dim3(4), dim3(64), 0, 0>>>(output, output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(output_host.data(), output, output_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.0, 0.2);

    double stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(static_cast<double>(v) - mean, 2);
    }
    stddev = stddev / output_size;
    EXPECT_NEAR(stddev, 1.0, 0.2);
}

template<class state_type>
void hiprand_kernel_h_hiprand_normal_sobol_test()
{
    using int_type = typename std::
        conditional<is_sobol_32<state_type>::value, unsigned int, unsigned long long int>::type;

    const unsigned int dimensions = 8;

    DirectionVectors_t<int_type>* d_vector;
    int_type*                     d_constants;

    if(is_sobol_scrambled<state_type>::value)
    {
        load_scrambled_sobol_constants_and_vectors_to_gpu<state_type, int_type>(dimensions,
                                                                                &d_vector,
                                                                                &d_constants);
    }
    else
    {
        load_sobol_vectors_to_gpu<state_type, int_type>(dimensions, &d_vector);
        d_constants = NULL;
    }

    const size_t output_size = 8192;
    float*       output;
    HIP_CHECK(hipMallocHelper((void**)&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_normal_sobol_kernel<state_type>
        <<<dim3(4), dim3(64), 0, 0>>>(output, output_size, d_vector);
    HIP_CHECK(hipGetLastError());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(output_host.data(), output, output_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.0, 0.2);

    double stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(static_cast<double>(v) - mean, 2);
    }
    stddev = stddev / output_size;
    EXPECT_NEAR(stddev, 1.0, 0.2);
}

template<class state_type>
void hiprand_kernel_h_hiprand_normal_mtgp_test()
{
    const size_t states_size = 128;
    state_type*  states;
    HIP_CHECK(hipMallocHelper((void**)&states, states_size * sizeof(state_type)));
    mtgp32_kernel_params_t* k;
    HIP_CHECK(hipMallocHelper((void**)&k, states_size * sizeof(mtgp32_kernel_params_t)));

    HIPRAND_CHECK(hiprandMakeMTGP32Constants(mtgp32dc_params_fast_11213, k));

    unsigned long long seed = 0;
    HIPRAND_CHECK(
        hiprandMakeMTGP32KernelState(states, mtgp32dc_params_fast_11213, k, states_size, seed));

    const size_t output_size = 8192;
    float*       output;
    HIP_CHECK(hipMallocHelper((void**)&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_normal_mtgp_kernel<state_type>
        <<<dim3(4), dim3(64), 0, 0>>>(states, output, output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(float),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(states));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.0, 0.2);

    double stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(static_cast<double>(v) - mean, 2);
    }
    stddev = stddev / output_size;
    EXPECT_NEAR(stddev, 1.0, 0.2);
}

TEST(hiprand_kernel_h_philox4x32_10, hiprand_normal)
{
    typedef hiprandStatePhilox4_32_10_t state_type;
    hiprand_kernel_h_hiprand_normal_test<state_type>();
}

TEST(hiprand_kernel_h_mrg32k3a, hiprand_normal)
{
    typedef hiprandStateMRG32k3a_t state_type;
    hiprand_kernel_h_hiprand_normal_test<state_type>();
}

TEST(hiprand_kernel_h_xorwow, hiprand_normal)
{
    typedef hiprandStateXORWOW_t state_type;
    hiprand_kernel_h_hiprand_normal_test<state_type>();
}

TEST(hiprand_kernel_h_default, hiprand_normal)
{
    typedef hiprandState_t state_type;
    hiprand_kernel_h_hiprand_normal_test<state_type>();
}

TEST(hiprand_kernel_h_sobol_32, hiprand_normal)
{
    typedef hiprandStateSobol32_t state_type;
    hiprand_kernel_h_hiprand_normal_sobol_test<state_type>();
}

TEST(hiprand_kernel_h_sobol_64, hiprand_normal)
{
    typedef hiprandStateSobol64_t state_type;
    hiprand_kernel_h_hiprand_normal_sobol_test<state_type>();
}

TEST(hiprand_kernel_h_mtgp_32, hiprand_normal)
{
    typedef hiprandStateMtgp32_t state_type;
    hiprand_kernel_h_hiprand_normal_mtgp_test<state_type>();
}

template<class T>
void hiprand_kernel_h_hiprand_log_normal_test()
{
    typedef T state_type;

    const size_t output_size = 8192;
    float*       output;
    HIP_CHECK(hipMallocHelper((void**)&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_log_normal_kernel<state_type><<<dim3(4), dim3(64), 0, 0>>>(output, output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(output_host.data(), output, output_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(v - mean, 2);
    }
    stddev = std::sqrt(stddev / output_size);

    double logmean = std::log(mean * mean / std::sqrt(stddev + mean * mean));
    double logstd  = std::sqrt(std::log(1.0f + stddev / (mean * mean)));

    EXPECT_NEAR(1.6, logmean, 1.6 * 0.2);
    EXPECT_NEAR(0.25, logstd, 0.25 * 0.2);
}

template<class state_type>
void hiprand_kernel_h_hiprand_log_normal_sobol_test()
{
    using int_type = typename std::
        conditional<is_sobol_32<state_type>::value, unsigned int, unsigned long long int>::type;

    const unsigned int dimensions = 8;

    DirectionVectors_t<int_type>* d_vector;
    int_type*                     d_constants;

    if(is_sobol_scrambled<state_type>::value)
    {
        load_scrambled_sobol_constants_and_vectors_to_gpu<state_type, int_type>(dimensions,
                                                                                &d_vector,
                                                                                &d_constants);
    }
    else
    {
        load_sobol_vectors_to_gpu<state_type, int_type>(dimensions, &d_vector);
        d_constants = NULL;
    }

    const size_t output_size = 8192;
    float*       output;
    HIP_CHECK(hipMallocHelper((void**)&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_log_normal_sobol_kernel<state_type>
        <<<dim3(4), dim3(64), 0, 0>>>(output, output_size, d_vector);
    HIP_CHECK(hipGetLastError());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(output_host.data(), output, output_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(v - mean, 2);
    }
    stddev = std::sqrt(stddev / output_size);

    double logmean = std::log(mean * mean / std::sqrt(stddev + mean * mean));
    double logstd  = std::sqrt(std::log(1.0f + stddev / (mean * mean)));

    EXPECT_NEAR(1.6, logmean, 1.6 * 0.2);
    EXPECT_NEAR(0.25, logstd, 0.25 * 0.2);
}

template<class state_type>
void hiprand_kernel_h_hiprand_log_normal_mtgp_test()
{
    const size_t states_size = 128;
    state_type*  states;
    HIP_CHECK(hipMallocHelper((void**)&states, states_size * sizeof(state_type)));
    mtgp32_kernel_params_t* k;
    HIP_CHECK(hipMallocHelper((void**)&k, states_size * sizeof(mtgp32_kernel_params_t)));

    HIPRAND_CHECK(hiprandMakeMTGP32Constants(mtgp32dc_params_fast_11213, k));

    unsigned long long seed = 0;
    HIPRAND_CHECK(
        hiprandMakeMTGP32KernelState(states, mtgp32dc_params_fast_11213, k, states_size, seed));

    const size_t output_size = 8192;
    float * output;
    HIP_CHECK(hipMallocHelper((void **)&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_log_normal_mtgp_kernel<state_type>
        <<<dim3(4), dim3(64), 0, 0>>>(states, output, output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(float),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(states));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(v - mean, 2);
    }
    stddev = std::sqrt(stddev / output_size);

    double logmean = std::log(mean * mean / std::sqrt(stddev + mean * mean));
    double logstd = std::sqrt(std::log(1.0f + stddev/(mean * mean)));

    EXPECT_NEAR(1.6, logmean, 1.6 * 0.2);
    EXPECT_NEAR(0.25, logstd, 0.25 * 0.2);
}

TEST(hiprand_kernel_h_philox4x32_10, hiprand_log_normal)
{
    typedef hiprandStatePhilox4_32_10_t state_type;
    hiprand_kernel_h_hiprand_log_normal_test<state_type>();
}

TEST(hiprand_kernel_h_mrg32k3a, hiprand_log_normal)
{
    typedef hiprandStateMRG32k3a_t state_type;
    hiprand_kernel_h_hiprand_log_normal_test<state_type>();
}

TEST(hiprand_kernel_h_xorwow, hiprand_log_normal)
{
    typedef hiprandStateXORWOW_t state_type;
    hiprand_kernel_h_hiprand_log_normal_test<state_type>();
}

TEST(hiprand_kernel_h_default, hiprand_log_normal)
{
    typedef hiprandState_t state_type;
    hiprand_kernel_h_hiprand_log_normal_test<state_type>();
}

TEST(hiprand_kernel_h_sobol_32, hiprand_log_normal)
{
    typedef hiprandStateSobol32_t state_type;
    hiprand_kernel_h_hiprand_log_normal_sobol_test<state_type>();
}

TEST(hiprand_kernel_h_sobol_64, hiprand_log_normal)
{
    typedef hiprandStateSobol64_t state_type;
    hiprand_kernel_h_hiprand_log_normal_sobol_test<state_type>();
}

TEST(hiprand_kernel_h_mtgp_32, hiprand_log_normal)
{
    typedef hiprandStateMtgp32_t state_type;
    hiprand_kernel_h_hiprand_log_normal_mtgp_test<state_type>();
}

template<class S, class T>
auto launch_sobol_kernel(T*                     output,
                         DirectionVectors_t<T>* d_vector,
                         T*                     d_constants,
                         const unsigned int     dimensions,
                         const size_t           items_per_dimension) ->
    typename std::enable_if<is_sobol_scrambled<S>::value>::type
{
    hiprand_scrambled_sobol_kernel<S, T>
        <<<dim3(8, dimensions), dim3(32), 0, 0>>>(output,
                                                  d_vector,
                                                  d_constants,
                                                  items_per_dimension);
}
template<class S, class T>
auto launch_sobol_kernel(T*                     output,
                         DirectionVectors_t<T>* d_vector,
                         T*,
                         const unsigned int dimensions,
                         const size_t       items_per_dimension) ->
    typename std::enable_if<!is_sobol_scrambled<S>::value>::type
{
    hiprand_sobol_kernel<S, T>
        <<<dim3(8, dimensions), dim3(32), 0, 0>>>(output, d_vector, items_per_dimension);
}

template<class S, class T>
void hiprand_kernel_h_sobol()
{
    const unsigned int dimensions          = 8;
    const size_t       items_per_dimension = 8192;
    const size_t       output_size         = dimensions * items_per_dimension;

    DirectionVectors_t<T>* d_vector;
    T*                     d_constants;
    T*                     d_output;

    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&d_output), output_size * sizeof(T)));
    HIP_CHECK(hipDeviceSynchronize());

    if(is_sobol_scrambled<S>::value)
    {
        load_scrambled_sobol_constants_and_vectors_to_gpu<S, T>(dimensions,
                                                                &d_vector,
                                                                &d_constants);
    }
    else
    {
        load_sobol_vectors_to_gpu<S, T>(dimensions, &d_vector);
        d_constants = NULL;
    }
    HIP_CHECK(hipDeviceSynchronize());

    launch_sobol_kernel<S, T>(d_output, d_vector, d_constants, dimensions, items_per_dimension);
    HIP_CHECK(hipGetLastError());

    std::vector<T> output(output_size);
    HIP_CHECK(hipMemcpy(output.data(), d_output, output_size * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_vector));

    if(is_sobol_scrambled<S>::value)
    {
        HIP_CHECK(hipFree(d_constants));
    }

    const double int_type_max = (double)std::numeric_limits<T>::max();

    double mean = 0;
    for(auto v : output)
    {
        mean += static_cast<double>(v) / int_type_max;
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

template<typename S, typename T>
struct sobol_test_type
{
    typedef S state_type;
    typedef T int_type;
};

template<typename test_type>
struct sobol_tests : public ::testing::Test
{
    typedef typename test_type::state_type state_type;
    typedef typename test_type::int_type   int_type;
};

typedef ::testing::Types<sobol_test_type<hiprandStateSobol32, unsigned int>,
                         sobol_test_type<hiprandStateSobol64, unsigned long long int>,
                         sobol_test_type<hiprandStateScrambledSobol32, unsigned int>,
                         sobol_test_type<hiprandStateScrambledSobol64, unsigned long long int>>
    sobol_test_types;
TYPED_TEST_SUITE(sobol_tests, sobol_test_types);

TYPED_TEST(sobol_tests, sobol_tests)
{
    using state_type = typename TestFixture::state_type;
    using int_type   = typename TestFixture::int_type;

    hiprand_kernel_h_sobol<state_type, int_type>();
}

template<class T>
void hiprand_kernel_h_hiprand_poisson_test(double lambda)
{
    typedef T state_type;

    const size_t output_size = 8192;
    unsigned int * output;
    HIP_CHECK(hipMallocHelper((void **)&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_poisson_kernel<state_type><<<dim3(4), dim3(64), 0, 0>>>(output, output_size, lambda);
    HIP_CHECK(hipGetLastError());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(unsigned int),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double variance = 0;
    for(auto v : output_host)
    {
        variance += std::pow(v - mean, 2);
    }
    variance = variance / output_size;

    EXPECT_NEAR(mean, lambda, std::max(1.0, lambda * 1e-1));
    EXPECT_NEAR(variance, lambda, std::max(1.0, lambda * 1e-1));
}

template<class state_type>
void hiprand_kernel_h_hiprand_poisson_sobol_test(double lambda)
{
    using int_type = typename std::
        conditional<is_sobol_32<state_type>::value, unsigned int, unsigned long long int>::type;

    const unsigned int dimensions = 8;

    DirectionVectors_t<int_type>* d_vector;
    int_type*                     d_constants;

    if(is_sobol_scrambled<state_type>::value)
    {
        load_scrambled_sobol_constants_and_vectors_to_gpu<state_type, int_type>(dimensions,
                                                                                &d_vector,
                                                                                &d_constants);
    }
    else
    {
        load_sobol_vectors_to_gpu<state_type, int_type>(dimensions, &d_vector);
        d_constants = NULL;
    }

    const size_t  output_size = 8192;
    unsigned int* output;
    HIP_CHECK(hipMallocHelper((void**)&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_poisson_sobol_kernel<state_type>
        <<<dim3(4), dim3(64), 0, 0>>>(output, output_size, lambda, d_vector);
    HIP_CHECK(hipGetLastError());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(hipMemcpy(output_host.data(),
                        output,
                        output_size * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double variance = 0;
    for(auto v : output_host)
    {
        variance += std::pow(v - mean, 2);
    }
    variance = variance / output_size;

    EXPECT_NEAR(mean, lambda, std::max(1.0, lambda * 1e-1));
    EXPECT_NEAR(variance, lambda, std::max(1.0, lambda * 1e-1));
}

template<class state_type>
void hiprand_kernel_h_hiprand_poisson_mtgp_test(double lambda)
{
    const size_t states_size = 128;
    state_type*  states;
    HIP_CHECK(hipMallocHelper((void**)&states, states_size * sizeof(state_type)));
    mtgp32_kernel_params_t* k;
    HIP_CHECK(hipMallocHelper((void**)&k, states_size * sizeof(mtgp32_kernel_params_t)));

    HIPRAND_CHECK(hiprandMakeMTGP32Constants(mtgp32dc_params_fast_11213, k));

    unsigned long long seed = 0;
    HIPRAND_CHECK(
        hiprandMakeMTGP32KernelState(states, mtgp32dc_params_fast_11213, k, states_size, seed));

    const size_t  output_size = 8192;
    unsigned int* output;
    HIP_CHECK(hipMallocHelper((void**)&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprand_poisson_mtgp_kernel<state_type>
        <<<dim3(4), dim3(64), 0, 0>>>(states, output, output_size, lambda);
    HIP_CHECK(hipGetLastError());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(hipMemcpy(output_host.data(),
                        output,
                        output_size * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(states));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double variance = 0;
    for(auto v : output_host)
    {
        variance += std::pow(v - mean, 2);
    }
    variance = variance / output_size;

    EXPECT_NEAR(mean, lambda, std::max(1.0, lambda * 1e-1));
    EXPECT_NEAR(variance, lambda, std::max(1.0, lambda * 1e-1));
}

template<class T>
void hiprand_kernel_h_hiprand_discrete_test(double lambda)
{
    typedef T state_type;

    const size_t output_size = 8192;
    unsigned int * output;
    HIP_CHECK(hipMallocHelper((void **)&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprandDiscreteDistribution_t discrete_distribution;
    ASSERT_EQ(hiprandCreatePoissonDistribution(lambda, &discrete_distribution), HIPRAND_STATUS_SUCCESS);

    hiprand_discrete_kernel<state_type>
        <<<dim3(4), dim3(64), 0, 0>>>(output, output_size, discrete_distribution);
    HIP_CHECK(hipGetLastError());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(unsigned int),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    ASSERT_EQ(hiprandDestroyDistribution(discrete_distribution), HIPRAND_STATUS_SUCCESS);

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double variance = 0;
    for(auto v : output_host)
    {
        variance += std::pow(v - mean, 2);
    }
    variance = variance / output_size;

    EXPECT_NEAR(mean, lambda, std::max(1.0, lambda * 1e-1));
    EXPECT_NEAR(variance, lambda, std::max(1.0, lambda * 1e-1));
}

template<class state_type>
void hiprand_kernel_h_hiprand_discrete_sobol_test(double lambda)
{
    using int_type = typename std::
        conditional<is_sobol_32<state_type>::value, unsigned int, unsigned long long int>::type;

    const unsigned int dimensions = 8;

    DirectionVectors_t<int_type>* d_vector;
    int_type*                     d_constants;

    if(is_sobol_scrambled<state_type>::value)
    {
        load_scrambled_sobol_constants_and_vectors_to_gpu<state_type, int_type>(dimensions,
                                                                                &d_vector,
                                                                                &d_constants);
    }
    else
    {
        load_sobol_vectors_to_gpu<state_type, int_type>(dimensions, &d_vector);
        d_constants = NULL;
    }

    const size_t  output_size = 8192;
    unsigned int* output;
    HIP_CHECK(hipMallocHelper((void**)&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprandDiscreteDistribution_t discrete_distribution;
    ASSERT_EQ(hiprandCreatePoissonDistribution(lambda, &discrete_distribution),
              HIPRAND_STATUS_SUCCESS);

    hiprand_discrete_sobol_kernel<state_type>
        <<<dim3(4), dim3(64), 0, 0>>>(output, output_size, discrete_distribution, d_vector);
    HIP_CHECK(hipGetLastError());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(hipMemcpy(output_host.data(),
                        output,
                        output_size * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    ASSERT_EQ(hiprandDestroyDistribution(discrete_distribution), HIPRAND_STATUS_SUCCESS);

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double variance = 0;
    for(auto v : output_host)
    {
        variance += std::pow(v - mean, 2);
    }
    variance = variance / output_size;

    EXPECT_NEAR(mean, lambda, std::max(1.0, lambda * 1e-1));
    EXPECT_NEAR(variance, lambda, std::max(1.0, lambda * 1e-1));
}

template<class state_type>
void hiprand_kernel_h_hiprand_discrete_mtgp_test(double lambda)
{
    const size_t states_size = 128;
    state_type*  states;
    HIP_CHECK(hipMallocHelper((void**)&states, states_size * sizeof(state_type)));
    mtgp32_kernel_params_t* k;
    HIP_CHECK(hipMallocHelper((void**)&k, states_size * sizeof(mtgp32_kernel_params_t)));

    HIPRAND_CHECK(hiprandMakeMTGP32Constants(mtgp32dc_params_fast_11213, k));

    unsigned long long seed = 0;
    HIPRAND_CHECK(
        hiprandMakeMTGP32KernelState(states, mtgp32dc_params_fast_11213, k, states_size, seed));

    const size_t  output_size = 8192;
    unsigned int* output;
    HIP_CHECK(hipMallocHelper((void**)&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprandDiscreteDistribution_t discrete_distribution;
    ASSERT_EQ(hiprandCreatePoissonDistribution(lambda, &discrete_distribution),
              HIPRAND_STATUS_SUCCESS);

    hiprand_discrete_mtgp_kernel<state_type>
        <<<dim3(4), dim3(64), 0, 0>>>(states, output, output_size, discrete_distribution);
    HIP_CHECK(hipGetLastError());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(hipMemcpy(output_host.data(),
                        output,
                        output_size * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(states));
    ASSERT_EQ(hiprandDestroyDistribution(discrete_distribution), HIPRAND_STATUS_SUCCESS);

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double variance = 0;
    for(auto v : output_host)
    {
        variance += std::pow(v - mean, 2);
    }
    variance = variance / output_size;

    EXPECT_NEAR(mean, lambda, std::max(1.0, lambda * 1e-1));
    EXPECT_NEAR(variance, lambda, std::max(1.0, lambda * 1e-1));
}

const double lambdas[] = { 1.0, 5.5, 20.0, 100.0, 1234.5, 5000.0 };

class hiprand_kernel_h_philox4x32_10_poisson : public ::testing::TestWithParam<double> { };

TEST_P(hiprand_kernel_h_philox4x32_10_poisson, hiprand_poisson)
{
    typedef hiprandStatePhilox4_32_10_t state_type;
    hiprand_kernel_h_hiprand_poisson_test<state_type>(GetParam());
}

TEST_P(hiprand_kernel_h_philox4x32_10_poisson, hiprand_discrete)
{
    typedef hiprandStatePhilox4_32_10_t state_type;
    hiprand_kernel_h_hiprand_discrete_test<state_type>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(hiprand_kernel_h_philox4x32_10_poisson,
                        hiprand_kernel_h_philox4x32_10_poisson,
                        ::testing::ValuesIn(lambdas));

class hiprand_kernel_h_mrg32k3a_poisson : public ::testing::TestWithParam<double> { };

TEST_P(hiprand_kernel_h_mrg32k3a_poisson, hiprand_poisson)
{
    typedef hiprandStateMRG32k3a_t state_type;
    hiprand_kernel_h_hiprand_poisson_test<state_type>(GetParam());
}

TEST_P(hiprand_kernel_h_mrg32k3a_poisson, hiprand_discrete)
{
    typedef hiprandStateMRG32k3a_t state_type;
    hiprand_kernel_h_hiprand_discrete_test<state_type>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(hiprand_kernel_h_mrg32k3a_poisson,
                        hiprand_kernel_h_mrg32k3a_poisson,
                        ::testing::ValuesIn(lambdas));

class hiprand_kernel_h_xorwow_poisson : public ::testing::TestWithParam<double> { };

TEST_P(hiprand_kernel_h_xorwow_poisson, hiprand_poisson)
{
    typedef hiprandStateXORWOW_t state_type;
    hiprand_kernel_h_hiprand_poisson_test<state_type>(GetParam());
}

TEST_P(hiprand_kernel_h_xorwow_poisson, hiprand_discrete)
{
    typedef hiprandStateXORWOW_t state_type;
    hiprand_kernel_h_hiprand_discrete_test<state_type>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(hiprand_kernel_h_xorwow_poisson,
                        hiprand_kernel_h_xorwow_poisson,
                        ::testing::ValuesIn(lambdas));

class hiprand_kernel_h_default_poisson : public ::testing::TestWithParam<double> { };

TEST_P(hiprand_kernel_h_default_poisson, hiprand_poisson)
{
    typedef hiprandState_t state_type;
    hiprand_kernel_h_hiprand_poisson_test<state_type>(GetParam());
}

TEST_P(hiprand_kernel_h_default_poisson, hiprand_discrete)
{
    typedef hiprandState_t state_type;
    hiprand_kernel_h_hiprand_discrete_test<state_type>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(hiprand_kernel_h_default_poisson,
                        hiprand_kernel_h_default_poisson,
                        ::testing::ValuesIn(lambdas));

class hiprand_kernel_h_sobol_32_poisson : public ::testing::TestWithParam<double>
{};

TEST_P(hiprand_kernel_h_sobol_32_poisson, hiprand_poisson)
{
    typedef hiprandStateSobol32_t state_type;
    hiprand_kernel_h_hiprand_poisson_sobol_test<state_type>(GetParam());
}

TEST_P(hiprand_kernel_h_sobol_32_poisson, hiprand_discrete)
{
    typedef hiprandStateSobol32_t state_type;
    hiprand_kernel_h_hiprand_discrete_sobol_test<state_type>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(hiprand_kernel_h_sobol_32_poisson,
                         hiprand_kernel_h_sobol_32_poisson,
                         ::testing::ValuesIn(lambdas));

class hiprand_kernel_h_sobol_64_poisson : public ::testing::TestWithParam<double>
{};

TEST_P(hiprand_kernel_h_sobol_64_poisson, hiprand_poisson)
{
    typedef hiprandStateSobol64_t state_type;
    hiprand_kernel_h_hiprand_poisson_sobol_test<state_type>(GetParam());
}

TEST_P(hiprand_kernel_h_sobol_64_poisson, hiprand_discrete)
{
    typedef hiprandStateSobol64_t state_type;
    hiprand_kernel_h_hiprand_discrete_sobol_test<state_type>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(hiprand_kernel_h_sobol_64_poisson,
                         hiprand_kernel_h_sobol_64_poisson,
                         ::testing::ValuesIn(lambdas));

class hiprand_kernel_h_mtgp_32_poisson : public ::testing::TestWithParam<double>
{};

TEST_P(hiprand_kernel_h_mtgp_32_poisson, hiprand_poisson)
{
    typedef hiprandStateMtgp32_t state_type;
    hiprand_kernel_h_hiprand_poisson_mtgp_test<state_type>(GetParam());
}

TEST_P(hiprand_kernel_h_mtgp_32_poisson, hiprand_discrete)
{
    typedef hiprandStateMtgp32_t state_type;
    hiprand_kernel_h_hiprand_discrete_mtgp_test<state_type>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(hiprand_kernel_h_mtgp_32_poisson,
                         hiprand_kernel_h_mtgp_32_poisson,
                         ::testing::ValuesIn(lambdas));
