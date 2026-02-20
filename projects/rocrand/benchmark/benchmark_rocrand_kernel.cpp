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

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <numeric>
#include <utility>
#include <algorithm>

#include "cmdparser.hpp"

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>
#include <rocrand/rocrand_kernel.h>
#include <rocrand/rocrand_mtgp32_11213.h>

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

#define ROCRAND_CHECK(condition)                 \
  {                                              \
    rocrand_status status = condition;           \
    if(status != ROCRAND_STATUS_SUCCESS) {       \
        std::cout << "ROCRAND error: " << status << " line: " << __LINE__ << std::endl; \
        exit(status); \
    } \
  }

#ifndef DEFAULT_RAND_N
const size_t DEFAULT_RAND_N = 1024 * 1024 * 128;
#endif

size_t next_power2(size_t x)
{
    size_t power = 1;
    while (power < x)
    {
        power *= 2;
    }
    return power;
}

template<typename GeneratorState>
__global__
__launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void init_kernel(GeneratorState * states,
                 const unsigned long long seed,
                 const unsigned long long offset)
{
    const unsigned int state_id = blockIdx.x * blockDim.x + threadIdx.x;
    GeneratorState state;
    rocrand_init(seed, state_id, offset, &state);
    states[state_id] = state;
}

template<typename T, typename GeneratorState, typename GenerateFunc, typename Extra>
__global__
__launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void generate_kernel(GeneratorState * states,
                     T * data,
                     const size_t size,
                     GenerateFunc generate_func,
                     const Extra extra)
{
    const unsigned int state_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride   = gridDim.x * blockDim.x;

    GeneratorState state = states[state_id];
    unsigned int index = state_id;
    while(index < size)
    {
        data[index] = generate_func(&state, extra);
        index += stride;
    }
    states[state_id] = state;
}

template<typename GeneratorState>
struct runner
{
    GeneratorState * states;

    runner(const size_t /* dimensions */,
           const size_t blocks,
           const size_t threads,
           const unsigned long long seed,
           const unsigned long long offset)
    {
        const size_t states_size = blocks * threads;
        HIP_CHECK(hipMalloc(&states, states_size * sizeof(GeneratorState)));

        init_kernel<<<dim3(blocks), dim3(threads)>>>(states, seed, offset);

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
    }

    runner(const runner&)            = delete;
    runner(runner&&)                 = delete;
    runner& operator=(const runner&) = delete;
    runner& operator=(runner&&)      = delete;

    ~runner()
    {
        HIP_CHECK(hipFree(states));
    }

    template<typename T, typename GenerateFunc, typename Extra>
    void generate(const size_t blocks,
                  const size_t threads,
                  hipStream_t stream,
                  T * data,
                  const size_t size,
                  const GenerateFunc& generate_func,
                  const Extra extra)
    {
        generate_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(states,
                                                                    data,
                                                                    size,
                                                                    generate_func,
                                                                    extra);
    }
};

template<typename T, typename GenerateFunc, typename Extra>
__global__
__launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void generate_kernel(rocrand_state_mtgp32 * states,
                     T * data,
                     const size_t size,
                     GenerateFunc generate_func,
                     const Extra extra)
{
    const unsigned int state_id = blockIdx.x;
    unsigned int       index    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int       stride   = gridDim.x * blockDim.x;

    __shared__ rocrand_state_mtgp32 state;
    rocrand_mtgp32_block_copy(&states[state_id], &state);

    const size_t r                 = size % blockDim.x;
    const size_t size_rounded_down = size - r;
    const size_t size_rounded_up   = r == 0 ? size : size_rounded_down + blockDim.x;
    while(index < size_rounded_down)
    {
        data[index] = generate_func(&state, extra);
        index += stride;
    }
    while(index < size_rounded_up)
    {
        auto value = generate_func(&state, extra);
        if(index < size)
            data[index] = value;
        index += stride;
    }

    rocrand_mtgp32_block_copy(&state, &states[state_id]);
}

template<>
struct runner<rocrand_state_mtgp32>
{
    rocrand_state_mtgp32 * states;

    runner(const size_t /* dimensions */,
           const size_t blocks,
           const size_t /* threads */,
           const unsigned long long seed,
           const unsigned long long /* offset */)
    {
        const size_t states_size = std::min((size_t)200, blocks);
        HIP_CHECK(hipMalloc(&states, states_size * sizeof(rocrand_state_mtgp32)));

        ROCRAND_CHECK(rocrand_make_state_mtgp32(states, mtgp32dc_params_fast_11213, states_size, seed));
    }

    runner(const runner&)            = delete;
    runner(runner&&)                 = delete;
    runner& operator=(const runner&) = delete;
    runner& operator=(runner&&)      = delete;

    ~runner()
    {
        HIP_CHECK(hipFree(states));
    }

    template<typename T, typename GenerateFunc, typename Extra>
    void generate(const size_t blocks,
                  const size_t /* threads */,
                  hipStream_t stream,
                  T * data,
                  const size_t size,
                  const GenerateFunc& generate_func,
                  const Extra extra)
    {
        generate_kernel<<<dim3(std::min((size_t)200, blocks)), dim3(256), 0, stream>>>(
            states,
            data,
            size,
            generate_func,
            extra);
    }
};

__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE) void init_kernel(
    rocrand_state_lfsr113* states, const uint4 seed)
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
    size_t                 dimensions;

    runner(const size_t /* dimensions */,
           const size_t blocks,
           const size_t threads,
           const unsigned long long /* seed */,
           const unsigned long long /* offset */)
    {
        this->dimensions = 0;
        const size_t states_size = blocks * threads;
        HIP_CHECK(hipMalloc(&states, states_size * sizeof(rocrand_state_lfsr113)));

        init_kernel<<<dim3(blocks), dim3(threads)>>>(states,
                                                     uint4{ROCRAND_LFSR113_DEFAULT_SEED_X,
                                                           ROCRAND_LFSR113_DEFAULT_SEED_Y,
                                                           ROCRAND_LFSR113_DEFAULT_SEED_Z,
                                                           ROCRAND_LFSR113_DEFAULT_SEED_W});

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
    }

    runner(const runner&)            = delete;
    runner(runner&&)                 = delete;
    runner& operator=(const runner&) = delete;
    runner& operator=(runner&&)      = delete;

    ~runner()
    {
        HIP_CHECK(hipFree(states));
    }

    template<typename T, typename GenerateFunc, typename Extra>
    void generate(const size_t        blocks,
                  const size_t        threads,
                  hipStream_t         stream,
                  T*                  data,
                  const size_t        size,
                  const GenerateFunc& generate_func,
                  const Extra         extra)
    {
        generate_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(states,
                                                                    data,
                                                                    size,
                                                                    generate_func,
                                                                    extra);
    }
};

template<typename GeneratorState, typename SobolType>
__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE) void init_sobol_kernel(
    GeneratorState* states, SobolType* directions, SobolType offset)
{
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;
    GeneratorState     state;
    rocrand_init(&directions[dimension * sizeof(SobolType) * 8], offset + state_id, &state);
    states[gridDim.x * blockDim.x * dimension + state_id] = state;
}

template<typename GeneratorState, typename SobolType>
__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE) void init_scrambled_sobol_kernel(
    GeneratorState* states, SobolType* directions, SobolType* scramble_constants, SobolType offset)
{
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;
    GeneratorState     state;
    rocrand_init(&directions[dimension * sizeof(SobolType) * 8],
                 scramble_constants[dimension],
                 offset + state_id,
                 &state);
    states[gridDim.x * blockDim.x * dimension + state_id] = state;
}

// generate_kernel for the normal and scrambled sobol generators
template<typename GeneratorState, typename T, typename GenerateFunc, typename Extra>
__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE) void generate_sobol_kernel(
    GeneratorState* states,
    T*              data,
    const size_t    size,
    GenerateFunc    generate_func,
    const Extra     extra)
{
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride    = gridDim.x * blockDim.x;

    GeneratorState state  = states[gridDim.x * blockDim.x * dimension + state_id];
    const size_t   offset = dimension * size;
    unsigned int index = state_id;
    while(index < size)
    {
        data[offset + index] = generate_func(&state, extra);
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
    rocrand_state_sobol32 * states;
    size_t dimensions;

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

        unsigned int * directions;
        const size_t size = dimensions * 32 * sizeof(unsigned int);
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

    runner(const runner&)            = delete;
    runner(runner&&)                 = delete;
    runner& operator=(const runner&) = delete;
    runner& operator=(runner&&)      = delete;

    ~runner()
    {
        HIP_CHECK(hipFree(states));
    }

    template<typename T, typename GenerateFunc, typename Extra>
    void generate(const size_t blocks,
                  const size_t threads,
                  hipStream_t stream,
                  T * data,
                  const size_t size,
                  const GenerateFunc& generate_func,
                  const Extra extra)
    {
        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        generate_sobol_kernel<<<dim3(blocks_x, dimensions), dim3(threads), 0, stream>>>(
            states,
            data,
            size / dimensions,
            generate_func,
            extra);
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

    runner(const runner&)            = delete;
    runner(runner&&)                 = delete;
    runner& operator=(const runner&) = delete;
    runner& operator=(runner&&)      = delete;

    ~runner()
    {
        HIP_CHECK(hipFree(states));
    }

    template<typename T, typename GenerateFunc, typename Extra>
    void generate(const size_t        blocks,
                  const size_t        threads,
                  hipStream_t         stream,
                  T*                  data,
                  const size_t        size,
                  const GenerateFunc& generate_func,
                  const Extra         extra)
    {
        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        generate_sobol_kernel<<<dim3(blocks_x, dimensions), dim3(threads), 0, stream>>>(
            states,
            data,
            size / dimensions,
            generate_func,
            extra);
    }
};

template<>
struct runner<rocrand_state_sobol64>
{
    rocrand_state_sobol64 * states;
    size_t dimensions;

    runner(const size_t dimensions,
           const size_t blocks,
           const size_t threads,
           const unsigned long long /* seed */,
           const unsigned long long offset)
    {
        this->dimensions = dimensions;

        const unsigned long long* h_directions;
        rocrand_get_direction_vectors64(&h_directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6);

        const size_t states_size = blocks * threads * dimensions;
        HIP_CHECK(hipMalloc(&states, states_size * sizeof(rocrand_state_sobol64)));

        unsigned long long int * directions;
        const size_t size = dimensions * 64 * sizeof(unsigned long long int);
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

    runner(const runner&)            = delete;
    runner(runner&&)                 = delete;
    runner& operator=(const runner&) = delete;
    runner& operator=(runner&&)      = delete;

    ~runner()
    {
        HIP_CHECK(hipFree(states));
    }

    template<typename T, typename GenerateFunc, typename Extra>
    void generate(const size_t blocks,
                  const size_t threads,
                  hipStream_t stream,
                  T * data,
                  const size_t size,
                  const GenerateFunc& generate_func,
                  const Extra extra)
    {
        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        generate_sobol_kernel<<<dim3(blocks_x, dimensions), dim3(threads), 0, stream>>>(
            states,
            data,
            size / dimensions,
            generate_func,
            extra);
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

        rocrand_get_direction_vectors64(&h_directions,
                                        ROCRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6);
        rocrand_get_scramble_constants64(&h_constants);

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

    runner(const runner&)            = delete;
    runner(runner&&)                 = delete;
    runner& operator=(const runner&) = delete;
    runner& operator=(runner&&)      = delete;

    ~runner()
    {
        HIP_CHECK(hipFree(states));
    }

    template<typename T, typename GenerateFunc, typename Extra>
    void generate(const size_t        blocks,
                  const size_t        threads,
                  hipStream_t         stream,
                  T*                  data,
                  const size_t        size,
                  const GenerateFunc& generate_func,
                  const Extra         extra)
    {
        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        generate_sobol_kernel<<<dim3(blocks_x, dimensions), dim3(threads), 0, stream>>>(
            states,
            data,
            size / dimensions,
            generate_func,
            extra);
    }
};

template<typename T, typename GeneratorState, typename GenerateFunc, typename Extra>
void run_benchmark(const cli::Parser& parser,
                   hipStream_t stream,
                   const GenerateFunc& generate_func,
                   const Extra extra,
                   const std::string& distribution,
                   const std::string& engine,
                   const double lambda = 0.f)
{
    const size_t size = parser.get<size_t>("size");
    const size_t dimensions = parser.get<size_t>("dimensions");
    const size_t trials = parser.get<size_t>("trials");

    const size_t blocks = parser.get<size_t>("blocks");
    const size_t threads = parser.get<size_t>("threads");

    const std::string format = parser.get<std::string>("format");

    T * data;
    HIP_CHECK(hipMalloc(&data, size * sizeof(T)));

    runner<GeneratorState> r(dimensions, blocks, threads, 12345ULL, 6789ULL);

    // Warm-up
    for (size_t i = 0; i < 5; i++)
    {
        r.generate(blocks, threads, stream, data, size, generate_func, extra);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Measurement
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    for (size_t i = 0; i < trials; i++)
    {
        r.generate(blocks, threads, stream, data, size, generate_func, extra);
    }
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed;
    HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    if (format.compare("csv") == 0)
    {
        std::cout << std::fixed << std::setprecision(3)
                  << engine << ","
                  << distribution << ","
                  << (trials * size * sizeof(T)) /
                        (elapsed / 1e3 * (1 << 30)) << ","
                  << (trials * size) /
                        (elapsed / 1e3 * (1 << 30)) << ","
                  << elapsed / trials << ","
                  << elapsed << ","
                  << size << ",";
        if (distribution.compare("poisson") == 0 || distribution.compare("discrete-poisson") == 0)
        {
             std::cout <<  lambda;
        }
        std::cout << std::endl;
    }
    else
    {
        if (format.compare("console") != 0)
        {
            std::cout << "Unknown format specified (must be either console or csv).  Defaulting to console output." << std::endl;
        }
        std::cout << std::fixed << std::setprecision(3)
                  << "      "
                  << "Throughput = "
                  << std::setw(8) << (trials * size * sizeof(T)) /
                        (elapsed / 1e3 * (1 << 30))
                  << " GB/s, Samples = "
                  << std::setw(8) << (trials * size) /
                        (elapsed / 1e3 * (1 << 30))
                  << " GSample/s, AvgTime (1 trial) = "
                  << std::setw(8) << elapsed / trials
                  << " ms, Time (all) = "
                  << std::setw(8) << elapsed
                  << " ms, Size = " << size
                  << std::endl;
    }
    HIP_CHECK(hipFree(data));
}

template<typename GeneratorState>
void run_benchmarks(const cli::Parser& parser,
                    const std::string& distribution,
                    const std::string& engine,
                    hipStream_t stream)
{
    const std::string format = parser.get<std::string>("format");
    if (distribution == "uniform-uint")
    {
        run_benchmark<unsigned int, GeneratorState>(parser, stream,
            [] __device__ (GeneratorState * state, int) {
                return rocrand(state);
            }, 0,
            distribution, engine
        );
    }
    if (distribution == "uniform-float")
    {
        run_benchmark<float, GeneratorState>(parser, stream,
            [] __device__ (GeneratorState * state, int) {
                return rocrand_uniform(state);
            }, 0,
            distribution, engine
        );
    }
    if (distribution == "uniform-double")
    {
        run_benchmark<double, GeneratorState>(parser, stream,
            [] __device__ (GeneratorState * state, int) {
                return rocrand_uniform_double(state);
            }, 0,
            distribution, engine
        );
    }
    if (distribution == "normal-float")
    {
        run_benchmark<float, GeneratorState>(parser, stream,
            [] __device__ (GeneratorState * state, int) {
                return rocrand_normal(state);
            }, 0,
            distribution, engine
        );
    }
    if (distribution == "normal-double")
    {
        run_benchmark<double, GeneratorState>(parser, stream,
            [] __device__ (GeneratorState * state, int) {
                return rocrand_normal_double(state);
            }, 0,
            distribution, engine
        );
    }
    if (distribution == "log-normal-float")
    {
        run_benchmark<float, GeneratorState>(parser, stream,
            [] __device__ (GeneratorState * state, int) {
                return rocrand_log_normal(state, 0.0f, 1.0f);
            }, 0,
            distribution, engine
        );
    }
    if (distribution == "log-normal-double")
    {
        run_benchmark<double, GeneratorState>(parser, stream,
            [] __device__ (GeneratorState * state, int) {
                return rocrand_log_normal_double(state, 0.0, 1.0);
            }, 0,
            distribution, engine
        );
    }
    if (distribution == "poisson")
    {
        const auto lambdas = parser.get<std::vector<double>>("lambda");
        for (double lambda : lambdas)
        {
            if (format.compare("console") == 0)
            {
                std::cout << "    " << "lambda "
                    << std::fixed << std::setprecision(1) << lambda << std::endl;
            }
            run_benchmark<unsigned int, GeneratorState>(parser, stream,
                [] __device__ (GeneratorState * state, double lambda) {
                    return rocrand_poisson(state, lambda);
                }, lambda,
                distribution, engine, lambda
            );
        }
    }
    if (distribution == "discrete-poisson")
    {
        const auto lambdas = parser.get<std::vector<double>>("lambda");
        for (double lambda : lambdas)
        {
            if (format.compare("console") == 0)
            {
                std::cout << "    " << "lambda "
                    << std::fixed << std::setprecision(1) << lambda << std::endl;
            }
            rocrand_discrete_distribution discrete_distribution;
            ROCRAND_CHECK(rocrand_create_poisson_distribution(lambda, &discrete_distribution));
            run_benchmark<unsigned int, GeneratorState>(parser, stream,
                [] __device__ (GeneratorState * state, rocrand_discrete_distribution discrete_distribution) {
                    return rocrand_discrete(state, discrete_distribution);
                }, discrete_distribution,
                distribution, engine, lambda
            );
            ROCRAND_CHECK(rocrand_destroy_discrete_distribution(discrete_distribution));
        }
    }
    if (distribution == "discrete-custom")
    {
        const unsigned int offset = 1234;
        std::vector<double> probabilities = { 10, 10, 1, 120, 8, 6, 140, 2, 150, 150, 10, 80 };
        const int size = probabilities.size();
        double sum = 0.0;
        for (int i = 0; i < size; i++)
        {
            sum += probabilities[i];
        }
        for (int i = 0; i < size; i++)
        {
            probabilities[i] /= sum;
        }

        rocrand_discrete_distribution discrete_distribution;
        ROCRAND_CHECK(rocrand_create_discrete_distribution(probabilities.data(), probabilities.size(), offset, &discrete_distribution));
        run_benchmark<unsigned int, GeneratorState>(parser, stream,
            [] __device__ (GeneratorState * state, rocrand_discrete_distribution discrete_distribution) {
                return rocrand_discrete(state, discrete_distribution);
            }, discrete_distribution,
            distribution, engine
        );
        ROCRAND_CHECK(rocrand_destroy_discrete_distribution(discrete_distribution));
    }
}

const std::vector<std::string> all_engines = {"xorwow",
                                              "mrg31k3p",
                                              "mrg32k3a",
                                              "mtgp32",
                                              // "mt19937",
                                              "philox",
                                              "threefry2x32",
                                              "threefry2x64",
                                              "threefry4x32",
                                              "threefry4x64",
                                              "sobol32",
                                              "scrambled_sobol32",
                                              "sobol64",
                                              "scrambled_sobol64",
                                              "lfsr113"};

const std::vector<std::string> all_distributions = {
    "uniform-uint",
    // "uniform-long-long",
    "uniform-float",
    "uniform-double",
    "normal-float",
    "normal-double",
    "log-normal-float",
    "log-normal-double",
    "poisson",
    "discrete-poisson",
    "discrete-custom",
};

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);

    const std::string distribution_desc =
        "space-separated list of distributions:" +
        std::accumulate(all_distributions.begin(), all_distributions.end(), std::string(),
            [](const std::string& a, const std::string& b) {
                return a + "\n      " + b;
            }
        ) +
        "\n      or all";
    const std::string engine_desc =
        "space-separated list of random number engines:" +
        std::accumulate(all_engines.begin(), all_engines.end(), std::string(),
            [](const std::string& a, const std::string& b) {
                return a + "\n      " + b;
            }
        ) +
        "\n      or all";

    parser.set_optional<size_t>("size", "size", DEFAULT_RAND_N, "number of values");
    parser.set_optional<size_t>("dimensions", "dimensions", 1, "number of dimensions of quasi-random values");
    parser.set_optional<size_t>("trials", "trials", 20, "number of trials");
    parser.set_optional<size_t>("blocks", "blocks", 256, "number of blocks");
    parser.set_optional<size_t>("threads", "threads", 256, "number of threads in each block");
    parser.set_optional<std::vector<std::string>>("dis", "dis", {"uniform-uint"}, distribution_desc);
    parser.set_optional<std::vector<std::string>>("engine", "engine", {"philox"}, engine_desc);
    parser.set_optional<std::vector<double>>("lambda", "lambda", {10.0}, "space-separated list of lambdas of Poisson distribution");
    parser.set_optional<std::string>("format", "format", {"console"}, "output format: console or csv");
    parser.run_and_exit_if_error();

    std::vector<std::string> engines;
    {
        auto es = parser.get<std::vector<std::string>>("engine");
        if (std::find(es.begin(), es.end(), "all") != es.end())
        {
            engines = all_engines;
        }
        else
        {
            for (auto e : all_engines)
            {
                if (std::find(es.begin(), es.end(), e) != es.end())
                    engines.push_back(e);
            }
        }
    }

    std::vector<std::string> distributions;
    {
        auto ds = parser.get<std::vector<std::string>>("dis");
        if (std::find(ds.begin(), ds.end(), "all") != ds.end())
        {
            distributions = all_distributions;
        }
        else
        {
            for (auto d : all_distributions)
            {
                if (std::find(ds.begin(), ds.end(), d) != ds.end())
                    distributions.push_back(d);
            }
        }
    }

    int version;
    ROCRAND_CHECK(rocrand_get_version(&version));
    int runtime_version;
    HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
    int device_id;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));

    std::cout << "benchmark_rocrand_kernel" << std::endl;
    std::cout << "rocRAND: " << version << " ";
    std::cout << "Runtime: " << runtime_version << " ";
    std::cout << "Device: " << props.name;
    std::cout << std::endl << std::endl;

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    std::string format = parser.get<std::string>("format");
    bool console_output = format.compare("console") == 0 ? true : false;

    if (!console_output)
    {
        std::cout << "Engine,Distribution,Throughput,Samples,AvgTime (1 Trial),Time(all),Size,Lambda"
                  << std::endl;
        std::cout << ",,GB/s,GSample/s,ms),ms),values,"
                  << std::endl;
    }
    for (auto engine : engines)
    {
        if (console_output) std::cout << engine << ":" << std::endl;
        for (auto distribution : distributions)
        {
            if (console_output) std::cout << "  " << distribution << ":" << std::endl;
            if (engine == "xorwow")
            {
                run_benchmarks<rocrand_state_xorwow>(parser, distribution, engine, stream);
            }
            else if(engine == "mrg31k3p")
            {
                run_benchmarks<rocrand_state_mrg31k3p>(parser, distribution, engine, stream);
            }
            else if (engine == "mrg32k3a")
            {
                run_benchmarks<rocrand_state_mrg32k3a>(parser, distribution, engine, stream);
            }
            else if (engine == "philox")
            {
                run_benchmarks<rocrand_state_philox4x32_10>(parser, distribution, engine, stream);
            }
            else if (engine == "sobol32")
            {
                run_benchmarks<rocrand_state_sobol32>(parser, distribution, engine, stream);
            }
            else if(engine == "scrambled_sobol32")
            {
                run_benchmarks<rocrand_state_scrambled_sobol32>(parser, distribution, engine, stream);
            }
            else if (engine == "sobol64")
            {
                run_benchmarks<rocrand_state_sobol64>(parser, distribution, engine, stream);
            }
            else if(engine == "scrambled_sobol64")
            {
                run_benchmarks<rocrand_state_scrambled_sobol64>(parser, distribution, engine, stream);
            }
            else if (engine == "mtgp32")
            {
                run_benchmarks<rocrand_state_mtgp32>(parser, distribution, engine, stream);
            }
            else if(engine == "lfsr113")
            {
                run_benchmarks<rocrand_state_lfsr113>(parser, distribution, engine, stream);
            }
            else if(engine == "threefry2x32")
            {
                run_benchmarks<rocrand_state_threefry2x32_20>(parser, distribution, engine, stream);
            }
            else if(engine == "threefry2x64")
            {
                run_benchmarks<rocrand_state_threefry2x64_20>(parser, distribution, engine, stream);
            }
            else if(engine == "threefry4x32")
            {
                run_benchmarks<rocrand_state_threefry4x32_20>(parser, distribution, engine, stream);
            }
            else if(engine == "threefry4x64")
            {
                run_benchmarks<rocrand_state_threefry4x64_20>(parser, distribution, engine, stream);
            }
        }
        std::cout << std::endl;
    }

    HIP_CHECK(hipStreamDestroy(stream));

    return 0;
}
