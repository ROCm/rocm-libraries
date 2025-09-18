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

#include "benchmark_rocrand_utils.hpp"
#include "cmdparser.hpp"

#include <benchmark/benchmark.h>

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>
#include <rocrand/rocrand_kernel.h>
#include <rocrand/rocrand_mtgp32_11213.h>

#include "custom_csv_formater.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifndef DEFAULT_RAND_N
    #define DEFAULT_RAND_N (1024 * 1024 * 128)
#endif

template<typename EngineState>
__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void init_kernel(EngineState*             states,
                 const unsigned long long seed,
                 const unsigned long long offset)
{
    const unsigned int state_id = blockIdx.x * blockDim.x + threadIdx.x;
    EngineState        state;
    rocrand_init(seed, state_id, offset, &state);
    states[state_id] = state;
}

template<typename EngineState, typename T, typename Generator>
__global__
__launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
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
        HIP_CHECK(hipMalloc(&states, states_size * sizeof(EngineState)));

        init_kernel<<<dim3(blocks), dim3(threads)>>>(states, seed, offset);

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
    }

    ~runner()
    {
        HIP_CHECK(hipFree(states));
    }

    template<typename T, typename Generator>
    void generate(const size_t     blocks,
                  const size_t     threads,
                  hipStream_t      stream,
                  T*               data,
                  const size_t     size,
                  const Generator& generator)
    {
        generate_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(states, data, size, generator);
    }
};

template<typename T, typename Generator>
__global__
__launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void generate_kernel(rocrand_state_mtgp32* states, T* data, const size_t size, Generator generator)
{
    const unsigned int state_id = blockIdx.x;
    unsigned int       index    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride         = gridDim.x * blockDim.x;

    __shared__ rocrand_state_mtgp32 state;
    rocrand_mtgp32_block_copy(&states[state_id], &state);

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

    rocrand_mtgp32_block_copy(&state, &states[state_id]);
}

template<>
struct runner<rocrand_state_mtgp32>
{
    rocrand_state_mtgp32* states;

    runner(const size_t /* dimensions */,
           const size_t blocks,
           const size_t /* threads */,
           const unsigned long long seed,
           const unsigned long long /* offset */)
    {
        const size_t states_size = std::min((size_t)200, blocks);
        HIP_CHECK(hipMalloc(&states, states_size * sizeof(rocrand_state_mtgp32)));

        ROCRAND_CHECK(
            rocrand_make_state_mtgp32(states, mtgp32dc_params_fast_11213, states_size, seed));
    }

    ~runner()
    {
        HIP_CHECK(hipFree(states));
    }

    template<typename T, typename Generator>
    void generate(const size_t blocks,
                  const size_t /* threads */,
                  hipStream_t      stream,
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

__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void init_kernel(rocrand_state_lfsr113* states, const uint4 seed)
{
    const unsigned int    state_id = blockIdx.x * blockDim.x + threadIdx.x;
    rocrand_state_lfsr113 state;
    rocrand_init(seed, state_id, &state);
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
        HIP_CHECK(hipMalloc(&states, states_size * sizeof(rocrand_state_lfsr113)));

        hipLaunchKernelGGL(HIP_KERNEL_NAME(init_kernel),
                           dim3(blocks),
                           dim3(threads),
                           0,
                           0,
                           states,
                           uint4{ROCRAND_LFSR113_DEFAULT_SEED_X,
                                 ROCRAND_LFSR113_DEFAULT_SEED_Y,
                                 ROCRAND_LFSR113_DEFAULT_SEED_Z,
                                 ROCRAND_LFSR113_DEFAULT_SEED_W});

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
    }

    ~runner()
    {
        HIP_CHECK(hipFree(states));
    }

    template<typename T, typename Generator>
    void generate(const size_t     blocks,
                  const size_t     threads,
                  hipStream_t      stream,
                  T*               data,
                  const size_t     size,
                  const Generator& generator)
    {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(generate_kernel),
                           dim3(blocks),
                           dim3(threads),
                           0,
                           stream,
                           states,
                           data,
                           size,
                           generator);
    }
};

template<typename EngineState, typename SobolType>
__global__
__launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void init_sobol_kernel(EngineState* states, SobolType* directions, SobolType offset)
{
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;
    EngineState        state;
    rocrand_init(&directions[dimension * sizeof(SobolType) * 8], offset + state_id, &state);
    states[gridDim.x * blockDim.x * dimension + state_id] = state;
}

template<typename EngineState, typename SobolType>
__global__
__launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void init_scrambled_sobol_kernel(EngineState* states,
                                 SobolType*   directions,
                                 SobolType*   scramble_constants,
                                 SobolType    offset)
{
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;
    EngineState        state;
    rocrand_init(&directions[dimension * sizeof(SobolType) * 8],
                 scramble_constants[dimension],
                 offset + state_id,
                 &state);
    states[gridDim.x * blockDim.x * dimension + state_id] = state;
}

// generate_kernel for the normal and scrambled sobol generators
template<typename EngineState, typename T, typename Generator>
__global__
__launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
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
struct runner<rocrand_state_sobol32>
{
    rocrand_state_sobol32* states;
    size_t                 dimensions;

    runner(const size_t dimensions,
           const size_t blocks,
           const size_t threads,
           const unsigned long long /* seed */,
           const unsigned long long offset)
    {
        this->dimensions = dimensions;

        const unsigned int* h_directions;
        ROCRAND_CHECK(
            rocrand_get_direction_vectors32(&h_directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));

        const size_t states_size = blocks * threads * dimensions;
        HIP_CHECK(hipMalloc(&states, states_size * sizeof(rocrand_state_sobol32)));

        unsigned int* directions;
        const size_t  size = dimensions * 32 * sizeof(unsigned int);
        HIP_CHECK(hipMalloc(&directions, size));
        HIP_CHECK(hipMemcpy(directions, h_directions, size, hipMemcpyHostToDevice));

        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        init_sobol_kernel<<<dim3(blocks_x, dimensions), dim3(threads)>>>(
            states,
            directions,
            static_cast<unsigned int>(offset));

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipFree(directions));
    }

    ~runner()
    {
        HIP_CHECK(hipFree(states));
    }

    template<typename T, typename Generator>
    void generate(const size_t     blocks,
                  const size_t     threads,
                  hipStream_t      stream,
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
struct runner<rocrand_state_scrambled_sobol32>
{
    rocrand_state_scrambled_sobol32* states;
    size_t                           dimensions;

    runner(const size_t dimensions,
           const size_t blocks,
           const size_t threads,
           const unsigned long long /* seed */,
           const unsigned long long offset)
    {
        this->dimensions = dimensions;

        const unsigned int* h_directions;
        const unsigned int* h_constants;

        ROCRAND_CHECK(
            rocrand_get_direction_vectors32(&h_directions,
                                            ROCRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));
        ROCRAND_CHECK(rocrand_get_scramble_constants32(&h_constants));

        const size_t states_size = blocks * threads * dimensions;
        HIP_CHECK(hipMalloc(&states, states_size * sizeof(rocrand_state_scrambled_sobol32)));

        unsigned int* directions;
        const size_t  directions_size = dimensions * 32 * sizeof(unsigned int);
        HIP_CHECK(hipMalloc(&directions, directions_size));
        HIP_CHECK(hipMemcpy(directions, h_directions, directions_size, hipMemcpyHostToDevice));

        unsigned int* scramble_constants;
        const size_t  constants_size = dimensions * sizeof(unsigned int);
        HIP_CHECK(hipMalloc(&scramble_constants, constants_size));
        HIP_CHECK(
            hipMemcpy(scramble_constants, h_constants, constants_size, hipMemcpyHostToDevice));

        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        init_scrambled_sobol_kernel<<<dim3(blocks_x, dimensions), dim3(threads)>>>(
            states,
            directions,
            scramble_constants,
            static_cast<unsigned int>(offset));

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipFree(directions));
        HIP_CHECK(hipFree(scramble_constants));
    }

    ~runner()
    {
        HIP_CHECK(hipFree(states));
    }

    template<typename T, typename Generator>
    void generate(const size_t     blocks,
                  const size_t     threads,
                  hipStream_t      stream,
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
struct runner<rocrand_state_sobol64>
{
    rocrand_state_sobol64* states;
    size_t                 dimensions;

    runner(const size_t dimensions,
           const size_t blocks,
           const size_t threads,
           const unsigned long long /* seed */,
           const unsigned long long offset)
    {
        this->dimensions = dimensions;

        const unsigned long long* h_directions;
        ROCRAND_CHECK(
            rocrand_get_direction_vectors64(&h_directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));

        const size_t states_size = blocks * threads * dimensions;
        HIP_CHECK(hipMalloc(&states, states_size * sizeof(rocrand_state_sobol64)));

        unsigned long long int* directions;
        const size_t            size = dimensions * 64 * sizeof(unsigned long long int);
        HIP_CHECK(hipMalloc(&directions, size));
        HIP_CHECK(hipMemcpy(directions, h_directions, size, hipMemcpyHostToDevice));

        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        init_sobol_kernel<<<dim3(blocks_x, dimensions), dim3(threads)>>>(states,
                                                                         directions,
                                                                         offset);

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipFree(directions));
    }

    ~runner()
    {
        HIP_CHECK(hipFree(states));
    }

    template<typename T, typename Generator>
    void generate(const size_t     blocks,
                  const size_t     threads,
                  hipStream_t      stream,
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
struct runner<rocrand_state_scrambled_sobol64>
{
    rocrand_state_scrambled_sobol64* states;
    size_t                           dimensions;

    runner(const size_t dimensions,
           const size_t blocks,
           const size_t threads,
           const unsigned long long /* seed */,
           const unsigned long long offset)
    {
        this->dimensions = dimensions;

        const unsigned long long* h_directions;
        const unsigned long long* h_constants;

        ROCRAND_CHECK(
            rocrand_get_direction_vectors64(&h_directions,
                                            ROCRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6));
        ROCRAND_CHECK(rocrand_get_scramble_constants64(&h_constants));

        const size_t states_size = blocks * threads * dimensions;
        HIP_CHECK(hipMalloc(&states, states_size * sizeof(rocrand_state_scrambled_sobol64)));

        unsigned long long int* directions;
        const size_t            directions_size = dimensions * 64 * sizeof(unsigned long long int);
        HIP_CHECK(hipMalloc(&directions, directions_size));
        HIP_CHECK(hipMemcpy(directions, h_directions, directions_size, hipMemcpyHostToDevice));

        unsigned long long int* scramble_constants;
        const size_t            constants_size = dimensions * sizeof(unsigned long long int);
        HIP_CHECK(hipMalloc(&scramble_constants, constants_size));
        HIP_CHECK(
            hipMemcpy(scramble_constants, h_constants, constants_size, hipMemcpyHostToDevice));

        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        init_scrambled_sobol_kernel<<<dim3(blocks_x, dimensions), dim3(threads)>>>(
            states,
            directions,
            scramble_constants,
            offset);

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipFree(directions));
        HIP_CHECK(hipFree(scramble_constants));
    }

    ~runner()
    {
        HIP_CHECK(hipFree(states));
    }

    template<typename T, typename Generator>
    void generate(const size_t     blocks,
                  const size_t     threads,
                  hipStream_t      stream,
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

    std::string name()
    {
        return "uniform-uint";
    }

    __device__
    data_type
        operator()(Engine* state) const
    {
        return rocrand(state);
    }
};

template<typename Engine>
struct generator_ullong : public generator_type
{
    typedef unsigned long long int data_type;

    std::string name()
    {
        return "uniform-ullong";
    }

    __device__
    data_type
        operator()(Engine* state) const
    {
        return rocrand(state);
    }
};

template<typename Engine>
struct generator_uniform : public generator_type
{
    typedef float data_type;

    std::string name()
    {
        return "uniform-float";
    }

    __device__
    data_type
        operator()(Engine* state) const
    {
        return rocrand_uniform(state);
    }
};

template<typename Engine>
struct generator_uniform_double : public generator_type
{
    typedef double data_type;

    std::string name()
    {
        return "uniform-double";
    }

    __device__
    data_type
        operator()(Engine* state) const
    {
        return rocrand_uniform_double(state);
    }
};

template<typename Engine>
struct generator_normal : public generator_type
{
    typedef float data_type;

    std::string name()
    {
        return "normal-float";
    }

    __device__
    data_type
        operator()(Engine* state) const
    {
        return rocrand_normal(state);
    }
};

template<typename Engine>
struct generator_normal_double : public generator_type
{
    typedef double data_type;

    std::string name()
    {
        return "normal-double";
    }

    __device__
    data_type
        operator()(Engine* state) const
    {
        return rocrand_normal_double(state);
    }
};

template<typename Engine>
struct generator_log_normal : public generator_type
{
    typedef float data_type;

    std::string name()
    {
        return "log-normal-float";
    }

    __device__
    data_type
        operator()(Engine* state) const
    {
        return rocrand_log_normal(state, 0.f, 1.f);
    }
};

template<typename Engine>
struct generator_log_normal_double : public generator_type
{
    typedef double data_type;

    std::string name()
    {
        return "log-normal-double";
    }

    __device__
    data_type
        operator()(Engine* state) const
    {
        return rocrand_log_normal_double(state, 0., 1.);
    }
};

template<typename Engine>
struct generator_poisson : public generator_type
{
    typedef unsigned int data_type;

    std::string name()
    {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(1) << lambda;
        return "poisson(lambda=" + stream.str() + ")";
    }

    __device__
    data_type
        operator()(Engine* state)
    {
        return rocrand_poisson(state, lambda);
    }

    double lambda;
};

template<typename Engine>
struct generator_discrete_poisson : public generator_type
{
    typedef unsigned int data_type;

    std::string name()
    {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(1) << lambda;
        return "discrete-poisson(lambda=" + stream.str() + ")";
    }

    void create()
    {
        ROCRAND_CHECK(rocrand_create_poisson_distribution(lambda, &discrete_distribution));
    }

    void destroy()
    {
        ROCRAND_CHECK(rocrand_destroy_discrete_distribution(discrete_distribution));
    }

    __device__
    data_type
        operator()(Engine* state)
    {
        return rocrand_discrete(state, discrete_distribution);
    }

    rocrand_discrete_distribution discrete_distribution;
    double                        lambda;
};

template<typename Engine>
struct generator_discrete_custom : public generator_type
{
    typedef unsigned int data_type;

    std::string name()
    {
        return "discrete-custom";
    }

    void create()
    {
        const unsigned int  offset        = 1234;
        std::vector<double> probabilities = {10, 10, 1, 120, 8, 6, 140, 2, 150, 150, 10, 80};

        double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.);
        std::transform(probabilities.begin(),
                       probabilities.end(),
                       probabilities.begin(),
                       [=](double p) { return p / sum; });
        ROCRAND_CHECK(rocrand_create_discrete_distribution(probabilities.data(),
                                                           probabilities.size(),
                                                           offset,
                                                           &discrete_distribution));
    }

    void destroy()
    {
        ROCRAND_CHECK(rocrand_destroy_discrete_distribution(discrete_distribution));
    }

    __device__
    data_type
        operator()(Engine* state)
    {
        return rocrand_discrete(state, discrete_distribution);
    }

    rocrand_discrete_distribution discrete_distribution;
};

struct benchmark_context
{
    size_t              size;
    size_t              dimensions;
    size_t              trials;
    size_t              blocks;
    size_t              threads;
    std::vector<double> lambdas;
};

template<typename Engine, typename Generator>
void run_benchmark(benchmark::State&        state,
                   const hipStream_t        stream,
                   const benchmark_context& context,
                   Generator                generator)
{
    typedef typename Generator::data_type data_type;

    const size_t size       = context.size;
    const size_t dimensions = context.dimensions;
    const size_t trials     = context.trials;
    const size_t blocks     = context.blocks;
    const size_t threads    = context.threads;

    // Optional initialization of the generator
    generator.create();

    data_type* data;
    HIP_CHECK(hipMalloc(&data, size * sizeof(data_type)));

    constexpr unsigned long long int seed   = 12345ULL;
    constexpr unsigned long long int offset = 6789ULL;

    runner<Engine> r(dimensions, blocks, threads, seed, offset);

    // Warm-up
    for(size_t i = 0; i < 5; i++)
    {
        r.generate(blocks, threads, stream, data, size, generator);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
    }

    // Measurement
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    for(auto _ : state)
    {
        HIP_CHECK(hipEventRecord(start, stream));
        for(size_t i = 0; i < trials; i++)
        {
            r.generate(blocks, threads, stream, data, size, generator);
        }
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));

        state.SetIterationTime(elapsed / 1000.f);
    }
    state.SetBytesProcessed(trials * state.iterations() * size * sizeof(data_type));
    state.SetItemsProcessed(trials * state.iterations() * size);

    // Optional de-initialization of the generator
    generator.destroy();

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(data));
}

template<typename Engine, typename Generator>
void add_benchmark(const benchmark_context&                      context,
                   const hipStream_t                             stream,
                   std::vector<benchmark::internal::Benchmark*>& benchmarks,
                   const std::string&                            name,
                   Generator                                     generator)
{
    static_assert(std::is_trivially_copyable<Generator>::value
                      && std::is_trivially_destructible<Generator>::value,
                  "Generator gets copied to device at kernel launch.");
    const std::string benchmark_name = "device_kernel<" + name + "," + generator.name() + ">";
    benchmarks.emplace_back(benchmark::RegisterBenchmark(benchmark_name.c_str(),
                                                         &run_benchmark<Engine, Generator>,
                                                         stream,
                                                         context,
                                                         generator));
}

template<typename Engine>
void add_benchmarks(const benchmark_context&                      ctx,
                    const hipStream_t                             stream,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const rocrand_rng_type                        engine_type)
{
    constexpr bool is_64_bits = std::is_same<Engine, rocrand_state_scrambled_sobol64>::value
                                || std::is_same<Engine, rocrand_state_sobol64>::value
                                || std::is_same<Engine, rocrand_state_threefry2x64_20>::value
                                || std::is_same<Engine, rocrand_state_threefry4x64_20>::value;

    const std::string name = engine_name(engine_type);

    if(is_64_bits)
    {
        add_benchmark<Engine>(ctx, stream, benchmarks, name, generator_ullong<Engine>());
    }
    else
    {
        add_benchmark<Engine>(ctx, stream, benchmarks, name, generator_uint<Engine>());
    }

    add_benchmark<Engine>(ctx, stream, benchmarks, name, generator_uniform<Engine>());
    add_benchmark<Engine>(ctx, stream, benchmarks, name, generator_uniform_double<Engine>());
    add_benchmark<Engine>(ctx, stream, benchmarks, name, generator_normal<Engine>());
    add_benchmark<Engine>(ctx, stream, benchmarks, name, generator_normal_double<Engine>());
    add_benchmark<Engine>(ctx, stream, benchmarks, name, generator_log_normal<Engine>());
    add_benchmark<Engine>(ctx, stream, benchmarks, name, generator_log_normal_double<Engine>());

    for(size_t i = 0; i < ctx.lambdas.size(); i++)
    {
        generator_poisson<Engine> gen_poisson;
        gen_poisson.lambda = ctx.lambdas[i];
        add_benchmark<Engine>(ctx, stream, benchmarks, name, gen_poisson);
    }

    for(size_t i = 0; i < ctx.lambdas.size(); i++)
    {
        generator_discrete_poisson<Engine> gen_discrete_poisson;
        gen_discrete_poisson.lambda = ctx.lambdas[i];
        add_benchmark<Engine>(ctx, stream, benchmarks, name, gen_discrete_poisson);
    }

    add_benchmark<Engine>(ctx, stream, benchmarks, name, generator_discrete_custom<Engine>());
}

int main(int argc, char* argv[])
{
    // get paramaters before they are passed into
    // benchmark::Initialize()
    std::string outFormat     = "";
    std::string filter        = "";
    std::string consoleFormat = "";

    getFormats(argc, argv, outFormat, filter, consoleFormat);

    benchmark::Initialize(&argc, argv);

    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_RAND_N, "number of values");
    parser.set_optional<size_t>("dimensions",
                                "dimensions",
                                1,
                                "number of dimensions of quasi-random values");
    parser.set_optional<size_t>("trials", "trials", 20, "number of trials");
    parser.set_optional<size_t>("blocks", "blocks", 256, "number of blocks");
    parser.set_optional<size_t>("threads", "threads", 256, "number of threads in each block");
    parser.set_optional<std::vector<double>>(
        "lambda",
        "lambda",
        {10.0},
        "space-separated list of lambdas of Poisson distribution");
    parser.run_and_exit_if_error();

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    add_common_benchmark_rocrand_info();

    benchmark_context ctx{};

    ctx.size       = parser.get<size_t>("size");
    ctx.dimensions = parser.get<size_t>("dimensions");
    ctx.trials     = parser.get<size_t>("trials");
    ctx.blocks     = parser.get<size_t>("blocks");
    ctx.threads    = parser.get<size_t>("threads");
    ctx.lambdas    = parser.get<std::vector<double>>("lambda");

    benchmark::AddCustomContext("size", std::to_string(ctx.size));
    benchmark::AddCustomContext("dimensions", std::to_string(ctx.dimensions));
    benchmark::AddCustomContext("trials", std::to_string(ctx.trials));
    benchmark::AddCustomContext("blocks", std::to_string(ctx.blocks));
    benchmark::AddCustomContext("threads", std::to_string(ctx.threads));

    std::vector<benchmark::internal::Benchmark*> benchmarks = {};

    // MT19937 has no kernel implementation
    add_benchmarks<rocrand_state_lfsr113>(ctx, stream, benchmarks, ROCRAND_RNG_PSEUDO_LFSR113);
    add_benchmarks<rocrand_state_mrg31k3p>(ctx, stream, benchmarks, ROCRAND_RNG_PSEUDO_MRG31K3P);
    add_benchmarks<rocrand_state_mrg32k3a>(ctx, stream, benchmarks, ROCRAND_RNG_PSEUDO_MRG32K3A);
    add_benchmarks<rocrand_state_mtgp32>(ctx, stream, benchmarks, ROCRAND_RNG_PSEUDO_MTGP32);
    add_benchmarks<rocrand_state_philox4x32_10>(ctx,
                                                stream,
                                                benchmarks,
                                                ROCRAND_RNG_PSEUDO_PHILOX4_32_10);
    add_benchmarks<rocrand_state_scrambled_sobol32>(ctx,
                                                    stream,
                                                    benchmarks,
                                                    ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32);
    add_benchmarks<rocrand_state_scrambled_sobol64>(ctx,
                                                    stream,
                                                    benchmarks,
                                                    ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64);
    add_benchmarks<rocrand_state_sobol32>(ctx, stream, benchmarks, ROCRAND_RNG_QUASI_SOBOL32);
    add_benchmarks<rocrand_state_sobol64>(ctx, stream, benchmarks, ROCRAND_RNG_QUASI_SOBOL64);
    add_benchmarks<rocrand_state_threefry2x32_20>(ctx,
                                                  stream,
                                                  benchmarks,
                                                  ROCRAND_RNG_PSEUDO_THREEFRY2_32_20);
    add_benchmarks<rocrand_state_threefry4x32_20>(ctx,
                                                  stream,
                                                  benchmarks,
                                                  ROCRAND_RNG_PSEUDO_THREEFRY4_32_20);
    add_benchmarks<rocrand_state_threefry2x64_20>(ctx,
                                                  stream,
                                                  benchmarks,
                                                  ROCRAND_RNG_PSEUDO_THREEFRY2_64_20);
    add_benchmarks<rocrand_state_threefry4x64_20>(ctx,
                                                  stream,
                                                  benchmarks,
                                                  ROCRAND_RNG_PSEUDO_THREEFRY4_64_20);
    add_benchmarks<rocrand_state_xorwow>(ctx, stream, benchmarks, ROCRAND_RNG_PSEUDO_XORWOW);

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    benchmark::BenchmarkReporter* console_reporter  = getConsoleReporter(consoleFormat);
    benchmark::BenchmarkReporter* out_file_reporter = getOutFileReporter(outFormat);

    std::string spec = (filter == "" || filter == "all") ? "." : filter;

    // Run benchmarks
    if(outFormat == "") // default case
        benchmark::RunSpecifiedBenchmarks(console_reporter, spec);
    else
        benchmark::RunSpecifiedBenchmarks(console_reporter, out_file_reporter, spec);
    HIP_CHECK(hipStreamDestroy(stream));

    return 0;
}
