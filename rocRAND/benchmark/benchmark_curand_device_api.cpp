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

#include "benchmark_curand_utils.hpp"
#include "cmdparser.hpp"

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define CURAND_DEFAULT_MAX_BLOCK_SIZE 256

#ifndef DEFAULT_RAND_N
    #define DEFAULT_RAND_N (1024 * 1024 * 128)
#endif

template<typename EngineState>
__global__ __launch_bounds__(CURAND_DEFAULT_MAX_BLOCK_SIZE) void init_kernel(
    EngineState* states, const unsigned long long seed, const unsigned long long offset)
{
    const unsigned int state_id = blockIdx.x * blockDim.x + threadIdx.x;
    EngineState        state;
    curand_init(seed, state_id, offset, &state);
    states[state_id] = state;
}

template<typename EngineState, typename T, typename Generator>
__global__ __launch_bounds__(CURAND_DEFAULT_MAX_BLOCK_SIZE) void generate_kernel(
    EngineState* states, T* data, const size_t size, Generator generator)
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
        CUDA_CALL(cudaMalloc(&states, states_size * sizeof(EngineState)));

        init_kernel<<<blocks, threads>>>(states, seed, offset);

        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());
    }

    ~runner()
    {
        CUDA_CALL(cudaFree(states));
    }

    template<typename T, typename Generator>
    void generate(const size_t     blocks,
                  const size_t     threads,
                  cudaStream_t     stream,
                  T*               data,
                  const size_t     size,
                  const Generator& generator)
    {
        generate_kernel<<<blocks, threads, 0, stream>>>(states, data, size, generator);
    }
};

template<typename T, typename Generator>
__global__ __launch_bounds__(CURAND_DEFAULT_MAX_BLOCK_SIZE) void generate_kernel(
    curandStateMtgp32_t* states, T* data, const size_t size, Generator generator)
{
    const unsigned int state_id  = blockIdx.x;
    const unsigned int thread_id = threadIdx.x;
    unsigned int       index     = blockIdx.x * blockDim.x + thread_id;
    unsigned int       stride    = gridDim.x * blockDim.x;

    __shared__ curandStateMtgp32_t state;

    if(thread_id == 0)
        state = states[state_id];
    __syncthreads();

    const size_t r               = size % blockDim.x;
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
    __syncthreads();

    if(thread_id == 0)
        states[state_id] = state;
}

template<>
struct runner<curandStateMtgp32_t>
{
    curandStateMtgp32_t*    states;
    mtgp32_kernel_params_t* d_param;

    runner(const size_t /* dimensions */,
           const size_t blocks,
           const size_t /* threads */,
           const unsigned long long seed,
           const unsigned long long /* offset */)
    {
        const size_t states_size = std::min((size_t)200, blocks);
        CUDA_CALL(cudaMalloc(&states, states_size * sizeof(curandStateMtgp32_t)));

        CUDA_CALL(cudaMalloc(&d_param, sizeof(mtgp32_kernel_params)));
        CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, d_param));
        CURAND_CALL(curandMakeMTGP32KernelState(states,
                                                mtgp32dc_params_fast_11213,
                                                d_param,
                                                states_size,
                                                seed));
    }

    ~runner()
    {
        CUDA_CALL(cudaFree(states));
        CUDA_CALL(cudaFree(d_param));
    }

    template<typename T, typename Generator>
    void generate(const size_t blocks,
                  const size_t /* threads */,
                  cudaStream_t     stream,
                  T*               data,
                  const size_t     size,
                  const Generator& generator)
    {
        generate_kernel<<<std::min((size_t)200, blocks), 256, 0, stream>>>(states,
                                                                           data,
                                                                           size,
                                                                           generator);
    }
};

template<typename EngineState, typename SobolType>
__global__ __launch_bounds__(CURAND_DEFAULT_MAX_BLOCK_SIZE) void init_sobol_kernel(
    EngineState* states, SobolType* directions, SobolType offset)
{
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;
    EngineState        state;
    curand_init(&directions[dimension * sizeof(SobolType) * 8], offset + state_id, &state);
    states[gridDim.x * blockDim.x * dimension + state_id] = state;
}

template<typename EngineState, typename SobolType>
__global__ __launch_bounds__(CURAND_DEFAULT_MAX_BLOCK_SIZE) void init_scrambled_sobol_kernel(
    EngineState* states, SobolType* directions, SobolType* scramble_constants, SobolType offset)
{
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;
    EngineState        state;
    curand_init(&directions[dimension * sizeof(SobolType) * 8],
                scramble_constants[dimension],
                offset + state_id,
                &state);
    states[gridDim.x * blockDim.x * dimension + state_id] = state;
}

// generate_kernel for the sobol generators
template<typename EngineState, typename T, typename Generator>
__global__ __launch_bounds__(CURAND_DEFAULT_MAX_BLOCK_SIZE) void generate_sobol_kernel(
    EngineState* states, T* data, const size_t size, Generator generator)
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
    skipahead(size, &state);
    states[gridDim.x * blockDim.x * dimension + state_id] = state;
}

template<>
struct runner<curandStateSobol32_t>
{
    curandStateSobol32_t* states;
    size_t                dimensions;

    runner(const size_t dimensions,
           const size_t blocks,
           const size_t threads,
           const unsigned long long /* seed */,
           const unsigned long long offset)
    {
        this->dimensions = dimensions;

        curandDirectionVectors32_t* h_directions;
        CURAND_CALL(
            curandGetDirectionVectors32(&h_directions, CURAND_DIRECTION_VECTORS_32_JOEKUO6));

        const size_t states_size = blocks * threads * dimensions;
        CUDA_CALL(cudaMalloc(&states, states_size * sizeof(curandStateSobol32_t)));

        unsigned int* directions;
        const size_t  size = dimensions * sizeof(unsigned int) * 32;
        CUDA_CALL(cudaMalloc(&directions, size));
        CUDA_CALL(cudaMemcpy(directions, h_directions, size, cudaMemcpyHostToDevice));

        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        init_sobol_kernel<<<dim3(blocks_x, dimensions), threads>>>(
            states,
            directions,
            static_cast<unsigned int>(offset));

        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaFree(directions));
    }

    ~runner()
    {
        CUDA_CALL(cudaFree(states));
    }

    template<typename T, typename Generator>
    void generate(const size_t     blocks,
                  const size_t     threads,
                  cudaStream_t     stream,
                  T*               data,
                  const size_t     size,
                  const Generator& generator)
    {
        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        generate_sobol_kernel<<<dim3(blocks_x, dimensions), threads, 0, stream>>>(states,
                                                                                  data,
                                                                                  size / dimensions,
                                                                                  generator);
    }
};

template<>
struct runner<curandStateScrambledSobol32_t>
{
    curandStateScrambledSobol32_t* states;
    size_t                         dimensions;

    runner(const size_t dimensions,
           const size_t blocks,
           const size_t threads,
           const unsigned long long /* seed */,
           const unsigned long long offset)
    {
        this->dimensions = dimensions;

        curandDirectionVectors32_t* h_directions;
        unsigned int*               h_constants;

        CURAND_CALL(
            curandGetDirectionVectors32(&h_directions, CURAND_DIRECTION_VECTORS_32_JOEKUO6));
        CURAND_CALL(curandGetScrambleConstants32(&h_constants));

        const size_t states_size = blocks * threads * dimensions;
        CUDA_CALL(cudaMalloc(&states, states_size * sizeof(curandStateScrambledSobol32_t)));

        unsigned int* directions;
        const size_t  directions_size = dimensions * sizeof(unsigned int) * 32;
        CUDA_CALL(cudaMalloc(&directions, directions_size));
        CUDA_CALL(cudaMemcpy(directions, h_directions, directions_size, cudaMemcpyHostToDevice));

        unsigned int* scramble_constants;
        const size_t  constants_size = dimensions * sizeof(unsigned int);
        CUDA_CALL(cudaMalloc(&scramble_constants, constants_size));
        CUDA_CALL(
            cudaMemcpy(scramble_constants, h_constants, constants_size, cudaMemcpyHostToDevice));

        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        init_scrambled_sobol_kernel<<<dim3(blocks_x, dimensions), threads>>>(
            states,
            directions,
            scramble_constants,
            static_cast<unsigned int>(offset));

        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaFree(directions));
        CUDA_CALL(cudaFree(scramble_constants));
    }

    ~runner()
    {
        CUDA_CALL(cudaFree(states));
    }

    template<typename T, typename Generator>
    void generate(const size_t     blocks,
                  const size_t     threads,
                  cudaStream_t     stream,
                  T*               data,
                  const size_t     size,
                  const Generator& generator)
    {
        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        generate_sobol_kernel<<<dim3(blocks_x, dimensions), threads, 0, stream>>>(states,
                                                                                  data,
                                                                                  size / dimensions,
                                                                                  generator);
    }
};

template<>
struct runner<curandStateSobol64_t>
{
    curandStateSobol64_t* states;
    size_t                dimensions;

    runner(const size_t dimensions,
           const size_t blocks,
           const size_t threads,
           const unsigned long long /* seed */,
           const unsigned long long offset)
    {
        this->dimensions = dimensions;

        curandDirectionVectors64_t* h_directions;
        CURAND_CALL(
            curandGetDirectionVectors64(&h_directions, CURAND_DIRECTION_VECTORS_64_JOEKUO6));

        const size_t states_size = blocks * threads * dimensions;
        CUDA_CALL(cudaMalloc(&states, states_size * sizeof(curandStateSobol64_t)));

        unsigned long long int* directions;
        const size_t            size = dimensions * sizeof(unsigned long long) * 64;
        CUDA_CALL(cudaMalloc(&directions, size));
        CUDA_CALL(cudaMemcpy(directions, h_directions, size, cudaMemcpyHostToDevice));

        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        init_sobol_kernel<<<dim3(blocks_x, dimensions), threads>>>(states, directions, offset);

        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaFree(directions));
    }

    ~runner()
    {
        CUDA_CALL(cudaFree(states));
    }

    template<typename T, typename Generator>
    void generate(const size_t     blocks,
                  const size_t     threads,
                  cudaStream_t     stream,
                  T*               data,
                  const size_t     size,
                  const Generator& generator)
    {
        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        generate_sobol_kernel<<<dim3(blocks_x, dimensions), threads, 0, stream>>>(states,
                                                                                  data,
                                                                                  size / dimensions,
                                                                                  generator);
    }
};

template<>
struct runner<curandStateScrambledSobol64_t>
{
    curandStateScrambledSobol64_t* states;
    size_t                         dimensions;

    runner(const size_t dimensions,
           const size_t blocks,
           const size_t threads,
           const unsigned long long /* seed */,
           const unsigned long long offset)
    {
        this->dimensions = dimensions;

        curandDirectionVectors64_t* h_directions;
        unsigned long long*         h_constants;

        CURAND_CALL(
            curandGetDirectionVectors64(&h_directions, CURAND_DIRECTION_VECTORS_64_JOEKUO6));
        CURAND_CALL(curandGetScrambleConstants64(&h_constants));

        const size_t states_size = blocks * threads * dimensions;
        CUDA_CALL(cudaMalloc(&states, states_size * sizeof(curandStateScrambledSobol64_t)));

        unsigned long long* directions;
        const size_t        directions_size = dimensions * sizeof(unsigned long long) * 64;
        CUDA_CALL(cudaMalloc(&directions, directions_size));
        CUDA_CALL(cudaMemcpy(directions, h_directions, directions_size, cudaMemcpyHostToDevice));

        unsigned long long* scramble_constants;
        const size_t        constants_size = dimensions * sizeof(unsigned long long);
        CUDA_CALL(cudaMalloc(&scramble_constants, constants_size));
        CUDA_CALL(
            cudaMemcpy(scramble_constants, h_constants, constants_size, cudaMemcpyHostToDevice));

        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        init_scrambled_sobol_kernel<<<dim3(blocks_x, dimensions), threads>>>(states,
                                                                             directions,
                                                                             scramble_constants,
                                                                             offset);

        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaFree(directions));
        CUDA_CALL(cudaFree(scramble_constants));
    }

    ~runner()
    {
        CUDA_CALL(cudaFree(states));
    }

    template<typename T, typename Generator>
    void generate(const size_t     blocks,
                  const size_t     threads,
                  cudaStream_t     stream,
                  T*               data,
                  const size_t     size,
                  const Generator& generator)
    {
        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        generate_sobol_kernel<<<dim3(blocks_x, dimensions), threads, 0, stream>>>(states,
                                                                                  data,
                                                                                  size / dimensions,
                                                                                  generator);
    }
};

// Provide optional create and destroy functions for the generators.
struct generator_type
{
    void create() {}

    void destroy() {}
};

template<typename Engine>
struct generator_uint : public generator_type
{
    typedef unsigned int data_type;

    std::string name()
    {
        return "uniform-uint";
    }

    __device__ data_type operator()(Engine* state)
    {
        return curand(state);
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

    __device__ data_type operator()(Engine* state)
    {
        return curand(state);
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

    __device__ data_type operator()(Engine* state)
    {
        return curand_uniform(state);
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

    __device__ data_type operator()(Engine* state)
    {
        return curand_uniform_double(state);
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

    __device__ data_type operator()(Engine* state)
    {
        return curand_normal(state);
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

    __device__ data_type operator()(Engine* state)
    {
        return curand_normal_double(state);
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

    __device__ data_type operator()(Engine* state)
    {
        return curand_log_normal(state, 0.f, 1.f);
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

    __device__ data_type operator()(Engine* state)
    {
        return curand_log_normal_double(state, 0., 1.);
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

    __device__ data_type operator()(Engine* state)
    {
        return curand_poisson(state, lambda);
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
        CURAND_CALL(curandCreatePoissonDistribution(lambda, &discrete_distribution));
    }

    void destroy()
    {
        CURAND_CALL(curandDestroyDistribution(discrete_distribution));
    }

    __device__ data_type operator()(Engine* state)
    {
        return curand_discrete(state, discrete_distribution);
    }

    curandDiscreteDistribution_t discrete_distribution;
    double                       lambda;
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
                   const cudaStream_t       stream,
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
    CUDA_CALL(cudaMalloc(&data, size * sizeof(data_type)));

    constexpr unsigned long long int seed   = 12345ULL;
    constexpr unsigned long long int offset = 6789ULL;

    runner<Engine> r(dimensions, blocks, threads, seed, offset);

    // Warm-up
    for(size_t i = 0; i < 5; i++)
    {
        r.generate(blocks, threads, stream, data, size, generator);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());
    }

    // Measurement
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    for(auto _ : state)
    {
        CUDA_CALL(cudaEventRecord(start, stream));
        for(size_t i = 0; i < trials; i++)
        {
            r.generate(blocks, threads, stream, data, size, generator);
        }
        CUDA_CALL(cudaEventRecord(stop, stream));
        CUDA_CALL(cudaEventSynchronize(stop));

        float elapsed;
        CUDA_CALL(cudaEventElapsedTime(&elapsed, start, stop));

        state.SetIterationTime(elapsed / 1000.f);
    }
    state.SetBytesProcessed(trials * state.iterations() * size * sizeof(data_type));
    state.SetItemsProcessed(trials * state.iterations() * size);

    // Optional de-initialization of the generator
    generator.destroy();

    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    CUDA_CALL(cudaFree(data));
}

template<typename Engine, typename Generator>
void add_benchmark(const benchmark_context&                      context,
                   const cudaStream_t                            stream,
                   std::vector<benchmark::internal::Benchmark*>& benchmarks,
                   const std::string&                            engine_name,
                   Generator                                     generator)
{
    static_assert(std::is_trivially_copyable<Generator>::value
                      && std::is_trivially_destructible<Generator>::value,
                  "Generator gets copied to device at kernel launch.");
    const std::string benchmark_name
        = "device_kernel<" + engine_name + "," + generator.name() + ">";
    benchmarks.emplace_back(benchmark::RegisterBenchmark(benchmark_name.c_str(),
                                                         &run_benchmark<Engine, Generator>,
                                                         stream,
                                                         context,
                                                         generator));
}

template<typename Engine>
void add_benchmarks(const benchmark_context&                      ctx,
                    const cudaStream_t                            stream,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const curandRngType                           engine_type)
{
    constexpr bool is_64_bits = std::is_same<Engine, curandStateSobol64_t>::value
                                || std::is_same<Engine, curandStateScrambledSobol64_t>::value;

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

    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));

    add_common_benchmark_curand_info();

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

    add_benchmarks<curandStateMRG32k3a_t>(ctx, stream, benchmarks, CURAND_RNG_PSEUDO_MRG32K3A);
    add_benchmarks<curandStateMtgp32_t>(ctx, stream, benchmarks, CURAND_RNG_PSEUDO_MTGP32);
    add_benchmarks<curandStatePhilox4_32_10_t>(ctx,
                                               stream,
                                               benchmarks,
                                               CURAND_RNG_PSEUDO_PHILOX4_32_10);
    add_benchmarks<curandStateScrambledSobol32_t>(ctx,
                                                  stream,
                                                  benchmarks,
                                                  CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
    add_benchmarks<curandStateScrambledSobol64_t>(ctx,
                                                  stream,
                                                  benchmarks,
                                                  CURAND_RNG_QUASI_SCRAMBLED_SOBOL64);
    add_benchmarks<curandStateSobol32_t>(ctx, stream, benchmarks, CURAND_RNG_QUASI_SOBOL32);
    add_benchmarks<curandStateSobol64_t>(ctx, stream, benchmarks, CURAND_RNG_QUASI_SOBOL64);
    add_benchmarks<curandStateXORWOW_t>(ctx, stream, benchmarks, CURAND_RNG_PSEUDO_XORWOW);

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
    {
        benchmark::RunSpecifiedBenchmarks(console_reporter, spec);
    }
    else
    {
        benchmark::RunSpecifiedBenchmarks(console_reporter, out_file_reporter, spec);
    }
    CUDA_CALL(cudaStreamDestroy(stream));

    return 0;
}
