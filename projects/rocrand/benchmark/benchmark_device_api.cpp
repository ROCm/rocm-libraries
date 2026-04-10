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

#include "benchmark_utils.hpp"

#ifdef __HIP__
    #include <rocrand/rocrand_mtgp32_11213.h>
#elif defined(__CUDACC__)
    #include <curand_mtgp32_host.h>
#endif

#include <optional>

/// The default maximum number of threads per block.
#define RAND_DEFAULT_MAX_BLOCK_SIZE 256

#ifdef __HIP__
constexpr memcpy_kind_t MEMCPY_HOST_TO_DEVICE = hipMemcpyHostToDevice;
#elif defined(__CUDACC__)
constexpr memcpy_kind_t MEMCPY_HOST_TO_DEVICE = cudaMemcpyHostToDevice;
#endif

#ifdef __HIP__
constexpr rng_type_t RAND_RNG_PSEUDO_MRG32K3A         = ROCRAND_RNG_PSEUDO_MRG32K3A;
constexpr rng_type_t RAND_RNG_PSEUDO_XORWOW           = ROCRAND_RNG_PSEUDO_XORWOW;
constexpr rng_type_t RAND_RNG_PSEUDO_PHILOX4_32_10    = ROCRAND_RNG_PSEUDO_PHILOX4_32_10;
constexpr rng_type_t RAND_RNG_PSEUDO_MTGP32           = ROCRAND_RNG_PSEUDO_MTGP32;
constexpr rng_type_t RAND_RNG_QUASI_SOBOL32           = ROCRAND_RNG_QUASI_SOBOL32;
constexpr rng_type_t RAND_RNG_QUASI_SCRAMBLED_SOBOL32 = ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32;
constexpr rng_type_t RAND_RNG_QUASI_SOBOL64           = ROCRAND_RNG_QUASI_SOBOL64;
constexpr rng_type_t RAND_RNG_QUASI_SCRAMBLED_SOBOL64 = ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64;
#elif defined(__CUDACC__)
constexpr rng_type_t RAND_RNG_PSEUDO_MRG32K3A         = CURAND_RNG_PSEUDO_MRG32K3A;
constexpr rng_type_t RAND_RNG_PSEUDO_XORWOW           = CURAND_RNG_PSEUDO_XORWOW;
constexpr rng_type_t RAND_RNG_PSEUDO_PHILOX4_32_10    = CURAND_RNG_PSEUDO_PHILOX4_32_10;
constexpr rng_type_t RAND_RNG_PSEUDO_MTGP32           = CURAND_RNG_PSEUDO_MTGP32;
constexpr rng_type_t RAND_RNG_QUASI_SOBOL32           = CURAND_RNG_QUASI_SOBOL32;
constexpr rng_type_t RAND_RNG_QUASI_SCRAMBLED_SOBOL32 = CURAND_RNG_QUASI_SCRAMBLED_SOBOL32;
constexpr rng_type_t RAND_RNG_QUASI_SOBOL64           = CURAND_RNG_QUASI_SOBOL64;
constexpr rng_type_t RAND_RNG_QUASI_SCRAMBLED_SOBOL64 = CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
#endif

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

template<typename EngineState, typename T, typename Generator>
__global__ __launch_bounds__(RAND_DEFAULT_MAX_BLOCK_SIZE)
void generate_kernel(EngineState* states, T* data, const size_t size, Generator generator)
{
    const unsigned int state_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride   = gridDim.x * blockDim.x;

    EngineState  state = states[state_id];
    unsigned int index = state_id;
    while(index < size)
    {
        data[index] = generator(&state);
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
        data[index] = generator(&state);
        index += stride;
    }
    while(index < size_rounded_up)
    {
        auto value = generator(&state);
        if(index < size)
            data[index] = value;
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
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride    = gridDim.x * blockDim.x;

    EngineState  state  = states[gridDim.x * blockDim.x * dimension + state_id];
    const size_t offset = dimension * size;
    unsigned int index  = state_id;
    while(index < size)
    {
        data[offset + index] = generator(&state);
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

template<typename Engine>
struct generator_uint : public generator_type
{
    typedef unsigned int data_type;

    __device__
    data_type            operator()(Engine* state) const
    {
        return gpu_rand(state);
    }
};

template<typename Engine>
struct generator_ullong : public generator_type
{
    typedef unsigned long long int data_type;

    __device__
    data_type                      operator()(Engine* state) const
    {
        return gpu_rand(state);
    }
};

template<typename Engine>
struct generator_uniform : public generator_type
{
    typedef float data_type;

    __device__
    data_type     operator()(Engine* state) const
    {
        return gpu_rand_uniform(state);
    }
};

template<typename Engine>
struct generator_uniform_double : public generator_type
{
    typedef double data_type;

    __device__
    data_type      operator()(Engine* state) const
    {
        return gpu_rand_uniform_double(state);
    }
};

template<typename Engine>
struct generator_normal : public generator_type
{
    typedef float data_type;

    __device__
    data_type     operator()(Engine* state) const
    {
        return gpu_rand_normal(state);
    }
};

template<typename Engine>
struct generator_normal_double : public generator_type
{
    typedef double data_type;

    __device__
    data_type      operator()(Engine* state) const
    {
        return gpu_rand_normal_double(state);
    }
};

template<typename Engine>
struct generator_log_normal : public generator_type
{
    typedef float data_type;

    __device__
    data_type     operator()(Engine* state) const
    {
        return gpu_rand_log_normal(state, 0.f, 1.f);
    }
};

template<typename Engine>
struct generator_log_normal_double : public generator_type
{
    typedef double data_type;

    __device__
    data_type      operator()(Engine* state) const
    {
        return gpu_rand_log_normal_double(state, 0., 1.);
    }
};

template<typename Engine>
struct generator_poisson : public generator_type
{
    generator_poisson(double l) : lambda(l) {}

    typedef unsigned int data_type;

    __device__
    data_type            operator()(Engine* state) const
    {
        return gpu_rand_poisson(state, lambda);
    }

    double lambda;
};

template<typename Engine>
struct generator_discrete_poisson : public generator_type
{
    generator_discrete_poisson(double l) : lambda(l) {}

    typedef unsigned int data_type;

    void create()
    {
        RAND_CHECK(rand_create_poisson_distribution(lambda, &discrete_distribution));
    }

    void destroy()
    {
        RAND_CHECK(rand_destroy_discrete_distribution(discrete_distribution));
    }

    __device__
    data_type operator()(Engine* state) const
    {
        return rand_discrete(state, discrete_distribution);
    }

    rand_discrete_distribution_t discrete_distribution;
    double                       lambda;
};

#ifdef __HIP__
template<typename Engine>
struct generator_discrete_custom : public generator_type
{
    typedef unsigned int data_type;

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

    __device__
    data_type operator()(Engine* state) const
    {
        return rand_discrete(state, discrete_distribution);
    }

    rand_discrete_distribution_t discrete_distribution;
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

        const size_t items = state.size;

        primbench::log("Creating generator");
        m_generator.create();

        primbench::log("Allocating data");
        T* data;
        PRIMBENCH_CHECK(gpu_malloc(&data, items * sizeof(T)));

        primbench::log("Creating runner");
        runner<State> r(m_dimensions, m_blocks, m_threads, state.seed, m_offset);

        state.set_items(items);
        state.add_writes<T>(items);

        state.run([&] { r.generate(m_blocks, m_threads, stream, data, items, m_generator); });

        m_generator.destroy();

        PRIMBENCH_CHECK(gpu_free(data));
    }

private:
    Generator             m_generator;
    rng_type_t            m_engine;
    size_t                m_blocks;
    size_t                m_threads;
    size_t                m_dimensions;
    size_t                m_offset;
    std::optional<double> m_poisson_lambda;
};

#define QUEUE(generator, T, State, engine, Distribution, ...)                \
    executor.queue<device_api_benchmark<generator, State, T, Distribution>>( \
        generator(__VA_ARGS__),                                              \
        engine,                                                              \
        blocks,                                                              \
        threads,                                                             \
        dimensions,                                                          \
        offset,                                                              \
        ##__VA_ARGS__)

#ifdef __HIP__
    #define QUEUE_DISTRIBUTIONS(State, engine)                                                   \
        do                                                                                       \
        {                                                                                        \
            if constexpr(std::is_same_v<State, rand_state_sobol64_t>                             \
                         || std::is_same_v<State, rand_state_scrambled_sobol64_t>                \
                         || std::is_same_v<State, rocrand_state_threefry2x64_20>                 \
                         || std::is_same_v<State, rocrand_state_threefry4x64_20>)                \
            {                                                                                    \
                QUEUE(generator_ullong<State>,                                                   \
                      unsigned long long,                                                        \
                      State,                                                                     \
                      engine,                                                                    \
                      DISTRIBUTION_UNIFORM);                                                     \
            }                                                                                    \
            else                                                                                 \
            {                                                                                    \
                QUEUE(generator_uint<State>, uint32_t, State, engine, DISTRIBUTION_UNIFORM);     \
            }                                                                                    \
                                                                                                 \
            QUEUE(generator_uniform<State>, float, State, engine, DISTRIBUTION_UNIFORM);         \
            QUEUE(generator_uniform_double<State>, double, State, engine, DISTRIBUTION_UNIFORM); \
            QUEUE(generator_normal<State>, float, State, engine, DISTRIBUTION_NORMAL);           \
            QUEUE(generator_normal_double<State>, double, State, engine, DISTRIBUTION_NORMAL);   \
            QUEUE(generator_log_normal<State>, float, State, engine, DISTRIBUTION_LOG_NORMAL);   \
            QUEUE(generator_log_normal_double<State>,                                            \
                  double,                                                                        \
                  State,                                                                         \
                  engine,                                                                        \
                  DISTRIBUTION_LOG_NORMAL);                                                      \
                                                                                                 \
            for(double lambda : poisson_lambdas)                                                 \
            {                                                                                    \
                QUEUE(generator_poisson<State>,                                                  \
                      uint32_t,                                                                  \
                      State,                                                                     \
                      engine,                                                                    \
                      DISTRIBUTION_POISSON,                                                      \
                      lambda);                                                                   \
                QUEUE(generator_discrete_poisson<State>,                                         \
                      uint32_t,                                                                  \
                      State,                                                                     \
                      engine,                                                                    \
                      DISTRIBUTION_DISCRETE_POISSON,                                             \
                      lambda);                                                                   \
            }                                                                                    \
                                                                                                 \
            QUEUE(generator_discrete_custom<State>,                                              \
                  uint32_t,                                                                      \
                  State,                                                                         \
                  engine,                                                                        \
                  DISTRIBUTION_DISCRETE_CUSTOM);                                                 \
        }                                                                                        \
        while(0)
#elif defined(__CUDACC__)
    #define QUEUE_DISTRIBUTIONS(State, engine)                                                   \
        do                                                                                       \
        {                                                                                        \
            if constexpr(std::is_same_v<State, rand_state_sobol64_t>                             \
                         || std::is_same_v<State, rand_state_scrambled_sobol64_t>)               \
            {                                                                                    \
                QUEUE(generator_ullong<State>,                                                   \
                      unsigned long long,                                                        \
                      State,                                                                     \
                      engine,                                                                    \
                      DISTRIBUTION_UNIFORM);                                                     \
            }                                                                                    \
            else                                                                                 \
            {                                                                                    \
                QUEUE(generator_uint<State>, uint32_t, State, engine, DISTRIBUTION_UNIFORM);     \
            }                                                                                    \
                                                                                                 \
            QUEUE(generator_uniform<State>, float, State, engine, DISTRIBUTION_UNIFORM);         \
            QUEUE(generator_uniform_double<State>, double, State, engine, DISTRIBUTION_UNIFORM); \
            QUEUE(generator_normal<State>, float, State, engine, DISTRIBUTION_NORMAL);           \
            QUEUE(generator_normal_double<State>, double, State, engine, DISTRIBUTION_NORMAL);   \
            QUEUE(generator_log_normal<State>, float, State, engine, DISTRIBUTION_LOG_NORMAL);   \
            QUEUE(generator_log_normal_double<State>,                                            \
                  double,                                                                        \
                  State,                                                                         \
                  engine,                                                                        \
                  DISTRIBUTION_LOG_NORMAL);                                                      \
                                                                                                 \
            for(double lambda : poisson_lambdas)                                                 \
            {                                                                                    \
                QUEUE(generator_poisson<State>,                                                  \
                      uint32_t,                                                                  \
                      State,                                                                     \
                      engine,                                                                    \
                      DISTRIBUTION_POISSON,                                                      \
                      lambda);                                                                   \
                QUEUE(generator_discrete_poisson<State>,                                         \
                      uint32_t,                                                                  \
                      State,                                                                     \
                      engine,                                                                    \
                      DISTRIBUTION_DISCRETE_POISSON,                                             \
                      lambda);                                                                   \
            }                                                                                    \
        }                                                                                        \
        while(0)
#endif

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 128 * 1024 * 1024; // In items
    settings.min_gpu_ms_per_batch = 100;
    settings.hot                  = true;
    primbench::executor executor(argc, argv, settings);

    auto blocks     = executor.get<size_t>("blocks", 256, "Number of blocks");
    auto threads    = executor.get<size_t>("threads", 256, "Threads per block");
    auto dimensions = executor.get<size_t>("dimensions", 1, "Number of quasi-random dimensions");
    auto offset     = executor.get<size_t>("offset", 0, "Offset of generated pseudo-random values");
    auto poisson_lambdas
        = executor.get<std::vector<double>>("lambda",
                                            {10.0},
                                            "Space-separated list of Poisson lambdas");

    QUEUE_DISTRIBUTIONS(rand_state_mrg32k3a_t, RAND_RNG_PSEUDO_MRG32K3A);
    QUEUE_DISTRIBUTIONS(rand_state_philox4x32_10_t, RAND_RNG_PSEUDO_PHILOX4_32_10);
    QUEUE_DISTRIBUTIONS(rand_state_xorwow_t, RAND_RNG_PSEUDO_XORWOW);
    QUEUE_DISTRIBUTIONS(rand_state_mtgp32_t, RAND_RNG_PSEUDO_MTGP32);
    QUEUE_DISTRIBUTIONS(rand_state_sobol32_t, RAND_RNG_QUASI_SOBOL32);
    QUEUE_DISTRIBUTIONS(rand_state_scrambled_sobol32_t, RAND_RNG_QUASI_SCRAMBLED_SOBOL32);
    QUEUE_DISTRIBUTIONS(rand_state_sobol64_t, RAND_RNG_QUASI_SOBOL64);
    QUEUE_DISTRIBUTIONS(rand_state_scrambled_sobol64_t, RAND_RNG_QUASI_SCRAMBLED_SOBOL64);

#ifdef __HIP__
    QUEUE_DISTRIBUTIONS(rocrand_state_lfsr113, ROCRAND_RNG_PSEUDO_LFSR113);
    QUEUE_DISTRIBUTIONS(rocrand_state_mrg31k3p, ROCRAND_RNG_PSEUDO_MRG31K3P);
    QUEUE_DISTRIBUTIONS(rocrand_state_threefry2x32_20, ROCRAND_RNG_PSEUDO_THREEFRY2_32_20);
    QUEUE_DISTRIBUTIONS(rocrand_state_threefry4x32_20, ROCRAND_RNG_PSEUDO_THREEFRY4_32_20);
    QUEUE_DISTRIBUTIONS(rocrand_state_threefry2x64_20, ROCRAND_RNG_PSEUDO_THREEFRY2_64_20);
    QUEUE_DISTRIBUTIONS(rocrand_state_threefry4x64_20, ROCRAND_RNG_PSEUDO_THREEFRY4_64_20);
#endif

    executor.run();
}
