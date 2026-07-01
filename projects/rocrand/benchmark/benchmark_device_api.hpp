// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_utils.hpp"

#ifdef __HIP__
    #include <rocrand/rocrand_mtgp32_11213.h>
#elif defined(__CUDACC__)
    #include <curand_mtgp32_host.h>
#endif

#include <optional>

// This file provides benchmark runners for device API-based benchmarks.
//
// It is organized into multiple layers of abstraction:
// - device_api_benchmark: provides the primbench infrastructure.
// - runner: provides infrastructure to set up an RNG engine and launch a generator.
// - generator: executes the sampling logic and draws values from the requested distribution.
// - gpu_rand: provides a shared interface to access the C-style device API.
// - generate_kernel: the actual device kernel being benchmarked.
//
// A benchmark is triggered by an API call as follows:
// 1. `device_api_benchmark<...>` is instantiated in `benchmark_device_api.cpp`.
// 2. `device_api_benchmark::run` creates a `runner` and a `generator`.
// 3. The runner uses the generator to produce values.
//    (This step primarily handles host-side orchestration.)
// 4. The generator uses `gpu_rand` to invoke the device API.
//    The call may be vectorized (e.g., `rocrand4`).
//    Vectorization size is determined by `get_vectorization`.
// 5. The vectorized output is unrolled via the `unrolled` utility type.

/// The default maximum number of threads per block.
#define RAND_DEFAULT_MAX_BLOCK_SIZE 256

#ifdef __HIP__
using rand_state_mrg32k3a_t          = rocrand_state_mrg32k3a;
using rand_state_philox4x32_10_t     = rocrand_state_philox4x32_10;
using rand_state_xorwow_t            = rocrand_state_xorwow;
using rand_state_mtgp32_t            = rocrand_state_mtgp32;
using rand_state_sobol32_t           = rocrand_state_sobol32;
using rand_state_scrambled_sobol32_t = rocrand_state_scrambled_sobol32;
using rand_state_sobol64_t           = rocrand_state_sobol64;
using rand_state_scrambled_sobol64_t = rocrand_state_scrambled_sobol64;
#elif defined(__CUDACC__)
using rand_state_mrg32k3a_t          = curandStateMRG32k3a_t;
using rand_state_philox4x32_10_t     = curandStatePhilox4_32_10_t;
using rand_state_xorwow_t            = curandStateXORWOW_t;
using rand_state_mtgp32_t            = curandStateMtgp32_t;
using rand_state_sobol32_t           = curandStateSobol32_t;
using rand_state_scrambled_sobol32_t = curandStateScrambledSobol32_t;
using rand_state_sobol64_t           = curandStateSobol64_t;
using rand_state_scrambled_sobol64_t = curandStateScrambledSobol64_t;
#endif

#ifdef __HIP__
constexpr rand_direction_vector_set_t RAND_DIRECTION_VECTORS_32_JOEKUO6
    = ROCRAND_DIRECTION_VECTORS_32_JOEKUO6;
constexpr rand_direction_vector_set_t RAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6
    = ROCRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6;
constexpr rand_direction_vector_set_t RAND_DIRECTION_VECTORS_64_JOEKUO6
    = ROCRAND_DIRECTION_VECTORS_64_JOEKUO6;
constexpr rand_direction_vector_set_t RAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6
    = ROCRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6;
#elif defined(__CUDACC__)
constexpr rand_direction_vector_set_t RAND_DIRECTION_VECTORS_32_JOEKUO6
    = CURAND_DIRECTION_VECTORS_32_JOEKUO6;
constexpr rand_direction_vector_set_t RAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6
    = CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6;
constexpr rand_direction_vector_set_t RAND_DIRECTION_VECTORS_64_JOEKUO6
    = CURAND_DIRECTION_VECTORS_64_JOEKUO6;
constexpr rand_direction_vector_set_t RAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6
    = CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6;
#endif

constexpr size_t next_power2(size_t x)
{
    size_t power = 1;
    while(power < x)
        power *= 2;
    return power;
}

template<typename EngineState>
__global__ __launch_bounds__(RAND_DEFAULT_MAX_BLOCK_SIZE)
void init_kernel(EngineState*             states,
                 const unsigned long long seed,
                 const unsigned long long offset)
{
    const unsigned int state_id = blockIdx.x * blockDim.x + threadIdx.x;
    EngineState        state;
    rand_init(seed, state_id, offset, &state);
    states[state_id] = state;
}

/// Return how many elements an engine can emit in one iteration.
template<typename EngineState, distribution D, typename T>
constexpr int get_vectorization()
{
    // Sobol doesn't support vectorization.
    // Mtgp32 does support box-muller, but is not implemented in benchmarks.
    constexpr bool is_sobol = std::is_same_v<EngineState, rand_state_sobol32_t>
                              || std::is_same_v<EngineState, rand_state_sobol64_t>
                              || std::is_same_v<EngineState, rand_state_scrambled_sobol32_t>
                              || std::is_same_v<EngineState, rand_state_scrambled_sobol64_t>;
    constexpr bool is_mtgp32 = std::is_same_v<EngineState, rand_state_mtgp32_t>;
    if constexpr(is_sobol || is_mtgp32)
    {
        return 1;
    }

    // Philox4x supports all vectorized distributions.
    constexpr bool is_philox4x32_10 = std::is_same_v<EngineState, rand_state_philox4x32_10_t>;
    if constexpr(is_philox4x32_10)
    {
#ifdef __HIP__
        // rocrand_poisson4 is slow!
        if constexpr(D == DISTRIBUTION_POISSON)
        {
            return 1;
        }
#else
        // curand_uniform4_double(philox4x32) is slow!
        if constexpr(D == DISTRIBUTION_UNIFORM && std::is_same_v<T, double>)
        {
            return 2;
        }
#endif
        return 4;
    }

    // We can use box-muller for vectorization.
    if constexpr(D == DISTRIBUTION_NORMAL || D == DISTRIBUTION_LOG_NORMAL)
    {
        return 2;
    }

#ifdef __HIP__
    constexpr bool is_threefry2x32_20 = std::is_same_v<EngineState, rocrand_state_threefry2x32_20>;
    constexpr bool is_threefry2x64_20 = std::is_same_v<EngineState, rocrand_state_threefry2x64_20>;
    constexpr bool is_threefry4x32_20 = std::is_same_v<EngineState, rocrand_state_threefry4x32_20>;
    constexpr bool is_threefry4x64_20 = std::is_same_v<EngineState, rocrand_state_threefry4x64_20>;

    // Threefry generators can generate more raw values.
    if constexpr(D == DISTRIBUTION_UNIFORM && std::is_integral_v<T>)
    {
        if constexpr(is_threefry4x32_20 || is_threefry4x64_20)
        {
            return 4;
        }

        if constexpr(is_threefry2x32_20 || is_threefry2x64_20)
        {
            return 2;
        }
    }
#endif

    // All other generators and distributions have no vectorization!
    return 1;
}

template<typename EngineState, distribution D, typename T>
constexpr int vectorization = get_vectorization<EngineState, D, T>();

/// This struct unrolls a device generator. It automatically
/// checks if the generator can be vectorized. The operator
/// will write `n` items to a given `ptr`.
template<typename Generator, typename T, typename EngineState>
struct unrolled
{
    /// Number of elements in generated vector.
    static constexpr int n = Generator::n;

    __device__
    unrolled(Generator& gen)
        : gen(gen)
    {}
    Generator& gen;

    __device__
    void       operator()(EngineState* state, T* ptr) const
    {
        static_assert(n >= 1, "Generator must produce at least 1 element!");
        const auto v = gen(state);
        if constexpr(n == 1)
        {
            ptr[0] = v;
        }
        else if constexpr(n == 2)
        {
            // HIP supports indexing vectorized types via operator[].
            // CUDA 12.x does not.
            ptr[0] = v.x;
            ptr[1] = v.y;
        }
        else if constexpr(n == 4)
        {
            ptr[0] = v.x;
            ptr[1] = v.y;
            ptr[2] = v.z;
            ptr[3] = v.w;
        }
    }
};

template<typename EngineState, typename T, typename Generator>
__global__ __launch_bounds__(RAND_DEFAULT_MAX_BLOCK_SIZE)
void generate_kernel(EngineState* states, T* data, const size_t size, Generator generator)
{
    const auto         f        = unrolled<Generator, T, EngineState>(generator);
    const unsigned int state_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int stride   = (gridDim.x * blockDim.x) * f.n;

    EngineState  state = states[state_id];
    unsigned int index = state_id * f.n;
    while(index < size)
    {
        f(&state, data + index);
        index += stride;
    }
    states[state_id] = state;
}

template<typename EngineState>
struct runner
{
    EngineState* states;

    runner(const size_t /* dimensions */,
           const size_t             blocks,
           const size_t             threads,
           const unsigned long long seed,
           const unsigned long long offset)
    {
        const size_t states_size = blocks * threads;
        PRIMBENCH_CHECK(gpu_malloc(&states, states_size * sizeof(EngineState)));

        init_kernel<<<dim3(blocks), dim3(threads)>>>(states, seed, offset);

        PRIMBENCH_CHECK(gpu_get_last_error());
        PRIMBENCH_CHECK(gpu_device_synchronize());
    }

    ~runner()
    {
        PRIMBENCH_CHECK(gpu_free(states));
    }

    template<typename T, typename Generator>
    void generate(const size_t     blocks,
                  const size_t     threads,
                  stream_t         stream,
                  T*               data,
                  const size_t     size,
                  const Generator& generator)
    {
        generate_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(states, data, size, generator);
    }
};

template<typename T, typename Generator>
__global__ __launch_bounds__(RAND_DEFAULT_MAX_BLOCK_SIZE)
void generate_kernel(rand_state_mtgp32_t* states, T* data, const size_t size, Generator generator)
{
    const auto f = unrolled<Generator, T, rand_state_mtgp32_t>(generator);
    static_assert(f.n == 1, "mtgp32 does not support vectorized generation!");
    const unsigned int  state_id  = blockIdx.x;
    const unsigned int  thread_id = threadIdx.x;
    unsigned int        index     = blockIdx.x * blockDim.x + thread_id;
    unsigned int        stride    = gridDim.x * blockDim.x;

    __shared__
    rand_state_mtgp32_t state;

#ifdef __HIP__
    rocrand_mtgp32_block_copy(&states[state_id], &state);
#else
    if(thread_id == 0)
        state = states[state_id];
    __syncthreads();
#endif

    const size_t r                 = size % blockDim.x;
    const size_t size_rounded_down = size - r;
    const size_t size_rounded_up   = r == 0 ? size : size_rounded_down + blockDim.x;
    while(index < size_rounded_down)
    {
        f(&state, data + index);
        index += stride;
    }
    while(index < size_rounded_up)
    {
        T value;
        f(&state, &value);
        if(index < size)
        {
            data[index] = value;
        }
        index += stride;
    }

#ifdef __HIP__
    rocrand_mtgp32_block_copy(&state, &states[state_id]);
#else
    __syncthreads();
    if(thread_id == 0)
        states[state_id] = state;
#endif
}

template<>
struct runner<rand_state_mtgp32_t>
{
    rand_state_mtgp32_t* states;

#ifndef __HIP__
    mtgp32_kernel_params_t* d_param;
#endif

    runner(const size_t /* dimensions */,
           const size_t blocks,
           const size_t /* threads */,
           const unsigned long long seed,
           const unsigned long long /* offset */)
    {
        const size_t states_size = std::min((size_t)200, blocks);
        PRIMBENCH_CHECK(gpu_malloc(&states, states_size * sizeof(rand_state_mtgp32_t)));

#ifdef __HIP__
        RAND_CHECK(
            rocrand_make_state_mtgp32(states, mtgp32dc_params_fast_11213, states_size, seed));
#else
        PRIMBENCH_CHECK(gpu_malloc(&d_param, sizeof(mtgp32_kernel_params)));
        RAND_CHECK(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, d_param));
        RAND_CHECK(curandMakeMTGP32KernelState(states,
                                               mtgp32dc_params_fast_11213,
                                               d_param,
                                               states_size,
                                               seed));
#endif
    }

    ~runner()
    {
        PRIMBENCH_CHECK(gpu_free(states));

#ifndef __HIP__
        PRIMBENCH_CHECK(gpu_free(d_param));
#endif
    }

    template<typename T, typename Generator>
    void generate(const size_t blocks,
                  const size_t /* threads */,
                  stream_t         stream,
                  T*               data,
                  const size_t     size,
                  const Generator& generator)
    {
        generate_kernel<<<dim3(std::min((size_t)200, blocks)), dim3(256), 0, stream>>>(states,
                                                                                       data,
                                                                                       size,
                                                                                       generator);
    }
};

#ifdef __HIP__
__global__ __launch_bounds__(RAND_DEFAULT_MAX_BLOCK_SIZE)
void init_kernel(rocrand_state_lfsr113* states, const uint4 seed)
{
    const unsigned int    state_id = blockIdx.x * blockDim.x + threadIdx.x;
    rocrand_state_lfsr113 state;
    rand_init(seed, state_id, &state);
    states[state_id] = state;
}

template<>
struct runner<rocrand_state_lfsr113>
{
    rocrand_state_lfsr113* states;

    runner(const size_t /* dimensions */,
           const size_t blocks,
           const size_t threads,
           const unsigned long long /* seed */,
           const unsigned long long /* offset */)
    {
        const size_t states_size = blocks * threads;
        PRIMBENCH_CHECK(gpu_malloc(&states, states_size * sizeof(rocrand_state_lfsr113)));

        init_kernel<<<dim3(blocks), dim3(threads), 0, 0>>>(states,
                                                           uint4{ROCRAND_LFSR113_DEFAULT_SEED_X,
                                                                 ROCRAND_LFSR113_DEFAULT_SEED_Y,
                                                                 ROCRAND_LFSR113_DEFAULT_SEED_Z,
                                                                 ROCRAND_LFSR113_DEFAULT_SEED_W});

        PRIMBENCH_CHECK(gpu_get_last_error());
        PRIMBENCH_CHECK(gpu_device_synchronize());
    }

    ~runner()
    {
        PRIMBENCH_CHECK(gpu_free(states));
    }

    template<typename T, typename Generator>
    void generate(const size_t     blocks,
                  const size_t     threads,
                  stream_t         stream,
                  T*               data,
                  const size_t     size,
                  const Generator& generator)
    {
        generate_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(states, data, size, generator);
    }
};
#endif

template<typename EngineState, typename SobolType>
__global__ __launch_bounds__(RAND_DEFAULT_MAX_BLOCK_SIZE)
void init_sobol_kernel(EngineState* states, SobolType* directions, SobolType offset)
{
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;
    EngineState        state;
    rand_init(&directions[dimension * sizeof(SobolType) * 8], offset + state_id, &state);
    states[gridDim.x * blockDim.x * dimension + state_id] = state;
}

template<typename EngineState, typename SobolType>
__global__ __launch_bounds__(RAND_DEFAULT_MAX_BLOCK_SIZE)
void init_scrambled_sobol_kernel(EngineState* states,
                                 SobolType*   directions,
                                 SobolType*   scramble_constants,
                                 SobolType    offset)
{
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;
    EngineState        state;
    rand_init(&directions[dimension * sizeof(SobolType) * 8],
              scramble_constants[dimension],
              offset + state_id,
              &state);
    states[gridDim.x * blockDim.x * dimension + state_id] = state;
}

// generate_kernel for the normal and scrambled sobol generators
template<typename EngineState, typename T, typename Generator>
__global__ __launch_bounds__(RAND_DEFAULT_MAX_BLOCK_SIZE)
void generate_sobol_kernel(EngineState* states, T* data, const size_t size, Generator generator)
{
    const auto f = unrolled<Generator, T, EngineState>(generator);
    static_assert(f.n == 1, "sobol does not support vectorized generation!");
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride    = gridDim.x * blockDim.x;

    EngineState  state  = states[gridDim.x * blockDim.x * dimension + state_id];
    const size_t offset = dimension * size;
    unsigned int index  = state_id;
    while(index < size)
    {
        f(&state, data + offset + index);
        skipahead(stride - 1, &state);
        index += stride;
    }
    state = states[gridDim.x * blockDim.x * dimension + state_id];
    skipahead(static_cast<unsigned int>(size), &state);
    states[gridDim.x * blockDim.x * dimension + state_id] = state;
}

template<>
struct runner<rand_state_sobol32_t>
{
    rand_state_sobol32_t* states;
    size_t                dimensions;

    runner(const size_t dimensions,
           const size_t blocks,
           const size_t threads,
           const unsigned long long /* seed */,
           const unsigned long long offset)
    {
        this->dimensions = dimensions;

        direction_vectors32_t* h_directions;
        RAND_CHECK(rand_get_direction_vectors32(&h_directions, RAND_DIRECTION_VECTORS_32_JOEKUO6));

        const size_t states_size = blocks * threads * dimensions;
        PRIMBENCH_CHECK(gpu_malloc(&states, states_size * sizeof(rand_state_sobol32_t)));

        unsigned int* directions;
        const size_t  size = dimensions * 32 * sizeof(unsigned int);
        PRIMBENCH_CHECK(gpu_malloc(&directions, size));
        PRIMBENCH_CHECK(gpu_memcpy(directions, h_directions, size, MEMCPY_HOST_TO_DEVICE));

        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        init_sobol_kernel<<<dim3(blocks_x, dimensions), dim3(threads)>>>(
            states,
            directions,
            static_cast<unsigned int>(offset));

        PRIMBENCH_CHECK(gpu_get_last_error());
        PRIMBENCH_CHECK(gpu_device_synchronize());

        PRIMBENCH_CHECK(gpu_free(directions));
    }

    ~runner()
    {
        PRIMBENCH_CHECK(gpu_free(states));
    }

    template<typename T, typename Generator>
    void generate(const size_t     blocks,
                  const size_t     threads,
                  stream_t         stream,
                  T*               data,
                  const size_t     size,
                  const Generator& generator)
    {
        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        generate_sobol_kernel<<<dim3(blocks_x, dimensions), dim3(threads), 0, stream>>>(
            states,
            data,
            size / dimensions,
            generator);
    }
};

template<>
struct runner<rand_state_scrambled_sobol32_t>
{
    rand_state_scrambled_sobol32_t* states;
    size_t                          dimensions;

    runner(const size_t dimensions,
           const size_t blocks,
           const size_t threads,
           const unsigned long long /* seed */,
           const unsigned long long offset)
    {
        this->dimensions = dimensions;

        direction_vectors32_t* h_directions;
        RAND_CHECK(rand_get_direction_vectors32(&h_directions,
                                                RAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));

        const unsigned int* h_constants;
        RAND_CHECK(rand_get_scramble_constants32(&h_constants));

        const size_t states_size = blocks * threads * dimensions;
        PRIMBENCH_CHECK(gpu_malloc(&states, states_size * sizeof(rand_state_scrambled_sobol32_t)));

        unsigned int* directions;
        const size_t  directions_size = dimensions * 32 * sizeof(unsigned int);
        PRIMBENCH_CHECK(gpu_malloc(&directions, directions_size));
        PRIMBENCH_CHECK(
            gpu_memcpy(directions, h_directions, directions_size, MEMCPY_HOST_TO_DEVICE));

        unsigned int* scramble_constants;
        const size_t  constants_size = dimensions * sizeof(unsigned int);
        PRIMBENCH_CHECK(gpu_malloc(&scramble_constants, constants_size));
        PRIMBENCH_CHECK(
            gpu_memcpy(scramble_constants, h_constants, constants_size, MEMCPY_HOST_TO_DEVICE));

        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        init_scrambled_sobol_kernel<<<dim3(blocks_x, dimensions), dim3(threads)>>>(
            states,
            directions,
            scramble_constants,
            static_cast<unsigned int>(offset));

        PRIMBENCH_CHECK(gpu_get_last_error());
        PRIMBENCH_CHECK(gpu_device_synchronize());

        PRIMBENCH_CHECK(gpu_free(directions));
        PRIMBENCH_CHECK(gpu_free(scramble_constants));
    }

    ~runner()
    {
        PRIMBENCH_CHECK(gpu_free(states));
    }

    template<typename T, typename Generator>
    void generate(const size_t     blocks,
                  const size_t     threads,
                  stream_t         stream,
                  T*               data,
                  const size_t     size,
                  const Generator& generator)
    {
        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        generate_sobol_kernel<<<dim3(blocks_x, dimensions), dim3(threads), 0, stream>>>(
            states,
            data,
            size / dimensions,
            generator);
    }
};

template<>
struct runner<rand_state_sobol64_t>
{
    rand_state_sobol64_t* states;
    size_t                dimensions;

    runner(const size_t dimensions,
           const size_t blocks,
           const size_t threads,
           const unsigned long long /* seed */,
           const unsigned long long offset)
    {
        this->dimensions = dimensions;

        direction_vectors64_t* h_directions;
        RAND_CHECK(rand_get_direction_vectors64(&h_directions, RAND_DIRECTION_VECTORS_64_JOEKUO6));

        const size_t states_size = blocks * threads * dimensions;
        PRIMBENCH_CHECK(gpu_malloc(&states, states_size * sizeof(rand_state_sobol64_t)));

        unsigned long long int* directions;
        const size_t            size = dimensions * 64 * sizeof(unsigned long long int);
        PRIMBENCH_CHECK(gpu_malloc(&directions, size));
        PRIMBENCH_CHECK(gpu_memcpy(directions, h_directions, size, MEMCPY_HOST_TO_DEVICE));

        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        init_sobol_kernel<<<dim3(blocks_x, dimensions), dim3(threads)>>>(states,
                                                                         directions,
                                                                         offset);

        PRIMBENCH_CHECK(gpu_get_last_error());
        PRIMBENCH_CHECK(gpu_device_synchronize());

        PRIMBENCH_CHECK(gpu_free(directions));
    }

    ~runner()
    {
        PRIMBENCH_CHECK(gpu_free(states));
    }

    template<typename T, typename Generator>
    void generate(const size_t     blocks,
                  const size_t     threads,
                  stream_t         stream,
                  T*               data,
                  const size_t     size,
                  const Generator& generator)
    {
        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        generate_sobol_kernel<<<dim3(blocks_x, dimensions), dim3(threads), 0, stream>>>(
            states,
            data,
            size / dimensions,
            generator);
    }
};

template<>
struct runner<rand_state_scrambled_sobol64_t>
{
    rand_state_scrambled_sobol64_t* states;
    size_t                          dimensions;

    runner(const size_t dimensions,
           const size_t blocks,
           const size_t threads,
           const unsigned long long /* seed */,
           const unsigned long long offset)
    {
        this->dimensions = dimensions;

        direction_vectors64_t* h_directions;
        RAND_CHECK(rand_get_direction_vectors64(&h_directions,
                                                RAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6));

        const unsigned long long* h_constants;
        RAND_CHECK(rand_get_scramble_constants64(&h_constants));

        const size_t states_size = blocks * threads * dimensions;
        PRIMBENCH_CHECK(gpu_malloc(&states, states_size * sizeof(rand_state_scrambled_sobol64_t)));

        unsigned long long int* directions;
        const size_t            directions_size = dimensions * 64 * sizeof(unsigned long long int);
        PRIMBENCH_CHECK(gpu_malloc(&directions, directions_size));
        PRIMBENCH_CHECK(
            gpu_memcpy(directions, h_directions, directions_size, MEMCPY_HOST_TO_DEVICE));

        unsigned long long int* scramble_constants;
        const size_t            constants_size = dimensions * sizeof(unsigned long long int);
        PRIMBENCH_CHECK(gpu_malloc(&scramble_constants, constants_size));
        PRIMBENCH_CHECK(
            gpu_memcpy(scramble_constants, h_constants, constants_size, MEMCPY_HOST_TO_DEVICE));

        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        init_scrambled_sobol_kernel<<<dim3(blocks_x, dimensions), dim3(threads)>>>(
            states,
            directions,
            scramble_constants,
            offset);

        PRIMBENCH_CHECK(gpu_get_last_error());
        PRIMBENCH_CHECK(gpu_device_synchronize());

        PRIMBENCH_CHECK(gpu_free(directions));
        PRIMBENCH_CHECK(gpu_free(scramble_constants));
    }

    ~runner()
    {
        PRIMBENCH_CHECK(gpu_free(states));
    }

    template<typename T, typename Generator>
    void generate(const size_t     blocks,
                  const size_t     threads,
                  stream_t         stream,
                  T*               data,
                  const size_t     size,
                  const Generator& generator)
    {
        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        generate_sobol_kernel<<<dim3(blocks_x, dimensions), dim3(threads), 0, stream>>>(
            states,
            data,
            size / dimensions,
            generator);
    }
};

// Provide optional create and destroy functions for the generators.
struct generator_type
{
    static void create() {}

    static void destroy() {}
};

/// Benchmarkable generators.
template<distribution Distribution, typename T, typename Engine>
struct generator : public generator_type
{
    static constexpr int n = vectorization<Engine, Distribution, T>;
    using device_api       = wrappers::gpu_rand<Distribution, n, T>;

    __device__
    auto operator()(Engine* state) const
    {
        return device_api{}(state);
    }
};

template<typename T, typename Engine>
struct generator<DISTRIBUTION_LOG_NORMAL, T, Engine> : public generator_type
{
    static constexpr int n = vectorization<Engine, DISTRIBUTION_LOG_NORMAL, T>;
    using device_api       = wrappers::gpu_rand<DISTRIBUTION_LOG_NORMAL, n, T>;

    __device__
    auto operator()(Engine* state) const
    {
        return device_api{}(state, 0.f, 1.f);
    }
};

template<typename T, typename Engine>
struct generator<DISTRIBUTION_POISSON, T, Engine> : public generator_type
{
    generator(double l) : lambda(l) {}
    double lambda;

    static constexpr int n = vectorization<Engine, DISTRIBUTION_POISSON, T>;
    using device_api       = wrappers::gpu_rand<DISTRIBUTION_POISSON, n, T>;

    __device__
    auto operator()(Engine* state) const
    {
        return device_api{}(state, lambda);
    }
};

template<typename T, typename Engine>
struct generator<DISTRIBUTION_DISCRETE_POISSON, T, Engine> : public generator_type
{
    generator(double l) : lambda(l) {}
    rand_discrete_distribution_t discrete_distribution;
    double                       lambda;

    void create()
    {
        RAND_CHECK(rand_create_poisson_distribution(lambda, &discrete_distribution));
    }

    void destroy()
    {
        RAND_CHECK(rand_destroy_discrete_distribution(discrete_distribution));
    }

    static constexpr int n = vectorization<Engine, DISTRIBUTION_DISCRETE_POISSON, T>;
    using device_api       = wrappers::gpu_rand<DISTRIBUTION_DISCRETE_POISSON, n, T>;

    __device__
    auto operator()(Engine* state) const
    {
        return device_api{}(state, discrete_distribution);
    }
};

#ifdef __HIP__
template<typename T, typename Engine>
struct generator<DISTRIBUTION_DISCRETE_CUSTOM, T, Engine> : public generator_type
{
    rand_discrete_distribution_t discrete_distribution;

    void create()
    {
        const unsigned int  offset        = 1234;
        std::vector<double> probabilities = {10, 10, 1, 120, 8, 6, 140, 2, 150, 150, 10, 80};

        double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.);
        std::transform(probabilities.begin(),
                       probabilities.end(),
                       probabilities.begin(),
                       [=](double p) { return p / sum; });
        RAND_CHECK(rocrand_create_discrete_distribution(probabilities.data(),
                                                        probabilities.size(),
                                                        offset,
                                                        &discrete_distribution));
    }

    void destroy()
    {
        RAND_CHECK(rand_destroy_discrete_distribution(discrete_distribution));
    }

    static constexpr int n = vectorization<Engine, DISTRIBUTION_DISCRETE_CUSTOM, T>;
    using device_api       = wrappers::gpu_rand<DISTRIBUTION_DISCRETE_CUSTOM, n, T>;

    __device__
    auto operator()(Engine* state) const
    {
        return device_api{}(state, discrete_distribution);
    }
};
#endif

template<typename Generator, typename State, typename T, distribution Distribution>
struct device_api_benchmark : public primbench::benchmark_interface
{
    device_api_benchmark(Generator             generator,
                         rng_type_t            engine,
                         size_t                blocks,
                         size_t                threads,
                         size_t                dimensions,
                         size_t                offset,
                         std::optional<double> poisson_lambda = std::nullopt)
        : m_generator(generator)
        , m_engine(engine)
        , m_blocks(blocks)
        , m_threads(threads)
        , m_dimensions(dimensions)
        , m_offset(offset)
        , m_poisson_lambda(poisson_lambda)
    {}

    primbench::json meta() const override
    {
        auto json
            = primbench::json{}
                  .add("algo", "device_api")
                  .add("engine", engine_name(m_engine))
                  .add("type", primbench::name<T>())
                  .add("distribution", distribution_name(Distribution))
                  .add("cfg", primbench::json{}.add("blocks", m_blocks).add("threads", m_threads));

        if constexpr(Distribution == DISTRIBUTION_POISSON
                     || Distribution == DISTRIBUTION_DISCRETE_POISSON)
            json.add("poisson_lambda", *m_poisson_lambda);

        return json;
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;

        // Pad items s.t. it is always divisible by 4 to account for vectorization.
        const size_t items = (state.size + 0b11) & ~size_t(0b11);

        primbench::log("Creating generator");
        m_generator.create();

        primbench::log("Allocating data");
        T* data;
        PRIMBENCH_CHECK(gpu_malloc(&data, items * sizeof(T)));

        primbench::log("Creating runner");
        runner<State> r(m_dimensions, m_blocks, m_threads, state.seed, m_offset);

        state.set_items(items);
        state.add_writes<T>(items);

        state.test([&] { test(data, items); });

        state.run([&] { r.generate(m_blocks, m_threads, stream, data, items, m_generator); });

        m_generator.destroy();

        PRIMBENCH_CHECK(gpu_free(data));
    }

private:
    void test(T* data, size_t items)
    {
        // Early return if the sample size is too small to guarantee statistical stability
        if(items < 10000)
        {
            return;
        }

        std::vector<T> h_data(items);
        PRIMBENCH_CHECK(gpu_memcpy(h_data.data(), data, items * sizeof(T), MEMCPY_DEVICE_TO_HOST));

        // Initializing to impossible values ensures the
        // test fails if a distribution branch is omitted.
        double expected_mean    = -9999.0;
        double expected_std_dev = -9999.0;

        // Calculate expected Mean and StdDev based on the distribution
        if constexpr(Distribution == DISTRIBUTION_UNIFORM)
        {
            expected_mean = 0.5;
            expected_std_dev
                = 1.0 / std::sqrt(12.0); // std::sqrt(1/12) for continuous uniform [0, 1]
        }
        else if constexpr(Distribution == DISTRIBUTION_NORMAL)
        {
            expected_mean    = 0.0;
            expected_std_dev = 1.0;
        }
        else if constexpr(Distribution == DISTRIBUTION_LOG_NORMAL)
        {
            // Log-normal mean and variance given parameters mu=0.0 and sigma=1.0
            expected_mean    = std::exp(0.5);
            expected_std_dev = std::sqrt((std::exp(1.0) - 1.0) * std::exp(1.0));
        }
        else if constexpr(Distribution == DISTRIBUTION_POISSON
                          || Distribution == DISTRIBUTION_DISCRETE_POISSON)
        {
            expected_mean    = *m_poisson_lambda;
            expected_std_dev = std::sqrt(*m_poisson_lambda);
        }
        else if constexpr(Distribution == DISTRIBUTION_DISCRETE_CUSTOM)
        {
            const double discrete_offset = 1234.0;
            const double weights[]       = {10, 10, 1, 120, 8, 6, 140, 2, 150, 150, 10, 80};

            double sum_of_weights              = 0.0;
            double sum_of_weighted_indices     = 0.0;
            double sum_of_weighted_squared_idx = 0.0;

            for(size_t i = 0; i < std::size(weights); ++i)
            {
                const double w = weights[i];
                sum_of_weights += w;
                sum_of_weighted_indices += static_cast<double>(i) * w;
                sum_of_weighted_squared_idx += static_cast<double>(i * i) * w;
            }

            const double expected_index_mean         = sum_of_weighted_indices / sum_of_weights;
            const double expected_index_squared_mean = sum_of_weighted_squared_idx / sum_of_weights;
            const double expected_index_variance
                = expected_index_squared_mean - std::pow(expected_index_mean, 2);

            expected_mean    = discrete_offset + expected_index_mean;
            expected_std_dev = std::sqrt(expected_index_variance);
        }

        auto normalize = [](T x) -> double
        {
            if constexpr(Distribution == DISTRIBUTION_UNIFORM && std::is_integral<T>::value)
            {
                double mini = static_cast<double>(std::numeric_limits<T>::min());
                double maxi = static_cast<double>(std::numeric_limits<T>::max());
                return (x - mini) / (maxi - mini);
            }
            else
            {
                return x;
            }
        };

        double actual_mean = std::accumulate(h_data.begin(),
                                             h_data.end(),
                                             0.0,
                                             [&](double acc, T x) { return acc + normalize(x); })
                             / static_cast<double>(items);

        double actual_std_dev = std::accumulate(h_data.begin(),
                                                h_data.end(),
                                                0.0,
                                                [&](double acc, T x)
                                                {
                                                    double diff = normalize(x) - actual_mean;
                                                    return acc + diff * diff;
                                                });
        actual_std_dev        = std::sqrt(actual_std_dev / static_cast<double>(items));

        // Use a 5% relative error tolerance for validation
        // If expected mean is close to 0, fall back to an absolute tolerance of 0.05
        double mean_tol = std::abs(expected_mean) > 1e-6 ? std::abs(expected_mean * 0.05) : 0.05;
        double std_dev_tol
            = std::abs(expected_std_dev) > 1e-6 ? std::abs(expected_std_dev * 0.05) : 0.05;

        if(std::abs(actual_mean - expected_mean) > mean_tol
           || std::abs(actual_std_dev - expected_std_dev) > std_dev_tol)
        {
            std::cerr << "\nError: Statistical mismatch for (" << engine_name(m_engine) << ", "
                      << distribution_name(Distribution) << ", " << primbench::name<T>() << ")\n"
                      << "  Expected Mean: " << expected_mean << ", Actual: " << actual_mean << "\n"
                      << "  Expected StdDev: " << expected_std_dev << ", Actual: " << actual_std_dev
                      << "\n";
            exit(EXIT_FAILURE);
        }
    }

    Generator             m_generator;
    rng_type_t            m_engine;
    size_t                m_blocks;
    size_t                m_threads;
    size_t                m_dimensions;
    size_t                m_offset;
    std::optional<double> m_poisson_lambda;
};
