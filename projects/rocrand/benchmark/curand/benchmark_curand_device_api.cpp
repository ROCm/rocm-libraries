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

#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>

#include <optional>

#define CURAND_DEFAULT_MAX_BLOCK_SIZE 256

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

constexpr size_t next_power2(size_t x)
{
    size_t power = 1;
    while(power < x)
        power *= 2;
    return power;
}

template<typename EngineState>
__global__ __launch_bounds__(CURAND_DEFAULT_MAX_BLOCK_SIZE)
void init_kernel(EngineState*             states,
                 const unsigned long long seed,
                 const unsigned long long offset)
{
    const unsigned int state_id = blockIdx.x * blockDim.x + threadIdx.x;
    EngineState        state;
    curand_init(seed, state_id, offset, &state);
    states[state_id] = state;
}

template<typename EngineState, typename T, typename Generator>
__global__ __launch_bounds__(CURAND_DEFAULT_MAX_BLOCK_SIZE)
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
        CUDA_CHECK(cudaMalloc(&states, states_size * sizeof(EngineState)));

        init_kernel<<<blocks, threads>>>(states, seed, offset);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~runner()
    {
        CUDA_CHECK(cudaFree(states));
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
__global__ __launch_bounds__(CURAND_DEFAULT_MAX_BLOCK_SIZE)
void generate_kernel(curandStateMtgp32_t* states, T* data, const size_t size, Generator generator)
{
    const unsigned int  state_id  = blockIdx.x;
    const unsigned int  thread_id = threadIdx.x;
    unsigned int        index     = blockIdx.x * blockDim.x + thread_id;
    unsigned int        stride    = gridDim.x * blockDim.x;

    __shared__
    curandStateMtgp32_t state;

    if(thread_id == 0)
        state = states[state_id];
    __syncthreads();

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
        CUDA_CHECK(cudaMalloc(&states, states_size * sizeof(curandStateMtgp32_t)));

        CUDA_CHECK(cudaMalloc(&d_param, sizeof(mtgp32_kernel_params)));
        CURAND_CHECK(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, d_param));
        CURAND_CHECK(curandMakeMTGP32KernelState(states,
                                                mtgp32dc_params_fast_11213,
                                                d_param,
                                                states_size,
                                                seed));
    }

    ~runner()
    {
        CUDA_CHECK(cudaFree(states));
        CUDA_CHECK(cudaFree(d_param));
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
__global__ __launch_bounds__(CURAND_DEFAULT_MAX_BLOCK_SIZE)
void init_sobol_kernel(EngineState* states, SobolType* directions, SobolType offset)
{
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;
    EngineState        state;
    curand_init(&directions[dimension * sizeof(SobolType) * 8], offset + state_id, &state);
    states[gridDim.x * blockDim.x * dimension + state_id] = state;
}

template<typename EngineState, typename SobolType>
__global__ __launch_bounds__(CURAND_DEFAULT_MAX_BLOCK_SIZE)
void init_scrambled_sobol_kernel(EngineState* states,
                                 SobolType*   directions,
                                 SobolType*   scramble_constants,
                                 SobolType    offset)
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
__global__ __launch_bounds__(CURAND_DEFAULT_MAX_BLOCK_SIZE)
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
        CURAND_CHECK(
            curandGetDirectionVectors32(&h_directions, CURAND_DIRECTION_VECTORS_32_JOEKUO6));

        const size_t states_size = blocks * threads * dimensions;
        CUDA_CHECK(cudaMalloc(&states, states_size * sizeof(curandStateSobol32_t)));

        unsigned int* directions;
        const size_t  size = dimensions * sizeof(unsigned int) * 32;
        CUDA_CHECK(cudaMalloc(&directions, size));
        CUDA_CHECK(cudaMemcpy(directions, h_directions, size, cudaMemcpyHostToDevice));

        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        init_sobol_kernel<<<dim3(blocks_x, dimensions), threads>>>(
            states,
            directions,
            static_cast<unsigned int>(offset));

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(directions));
    }

    ~runner()
    {
        CUDA_CHECK(cudaFree(states));
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

        CURAND_CHECK(
            curandGetDirectionVectors32(&h_directions, CURAND_DIRECTION_VECTORS_32_JOEKUO6));
        CURAND_CHECK(curandGetScrambleConstants32(&h_constants));

        const size_t states_size = blocks * threads * dimensions;
        CUDA_CHECK(cudaMalloc(&states, states_size * sizeof(curandStateScrambledSobol32_t)));

        unsigned int* directions;
        const size_t  directions_size = dimensions * sizeof(unsigned int) * 32;
        CUDA_CHECK(cudaMalloc(&directions, directions_size));
        CUDA_CHECK(cudaMemcpy(directions, h_directions, directions_size, cudaMemcpyHostToDevice));

        unsigned int* scramble_constants;
        const size_t  constants_size = dimensions * sizeof(unsigned int);
        CUDA_CHECK(cudaMalloc(&scramble_constants, constants_size));
        CUDA_CHECK(
            cudaMemcpy(scramble_constants, h_constants, constants_size, cudaMemcpyHostToDevice));

        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        init_scrambled_sobol_kernel<<<dim3(blocks_x, dimensions), threads>>>(
            states,
            directions,
            scramble_constants,
            static_cast<unsigned int>(offset));

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(directions));
        CUDA_CHECK(cudaFree(scramble_constants));
    }

    ~runner()
    {
        CUDA_CHECK(cudaFree(states));
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
        CURAND_CHECK(
            curandGetDirectionVectors64(&h_directions, CURAND_DIRECTION_VECTORS_64_JOEKUO6));

        const size_t states_size = blocks * threads * dimensions;
        CUDA_CHECK(cudaMalloc(&states, states_size * sizeof(curandStateSobol64_t)));

        unsigned long long int* directions;
        const size_t            size = dimensions * sizeof(unsigned long long) * 64;
        CUDA_CHECK(cudaMalloc(&directions, size));
        CUDA_CHECK(cudaMemcpy(directions, h_directions, size, cudaMemcpyHostToDevice));

        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        init_sobol_kernel<<<dim3(blocks_x, dimensions), threads>>>(states, directions, offset);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(directions));
    }

    ~runner()
    {
        CUDA_CHECK(cudaFree(states));
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

        CURAND_CHECK(
            curandGetDirectionVectors64(&h_directions, CURAND_DIRECTION_VECTORS_64_JOEKUO6));
        CURAND_CHECK(curandGetScrambleConstants64(&h_constants));

        const size_t states_size = blocks * threads * dimensions;
        CUDA_CHECK(cudaMalloc(&states, states_size * sizeof(curandStateScrambledSobol64_t)));

        unsigned long long* directions;
        const size_t        directions_size = dimensions * sizeof(unsigned long long) * 64;
        CUDA_CHECK(cudaMalloc(&directions, directions_size));
        CUDA_CHECK(cudaMemcpy(directions, h_directions, directions_size, cudaMemcpyHostToDevice));

        unsigned long long* scramble_constants;
        const size_t        constants_size = dimensions * sizeof(unsigned long long);
        CUDA_CHECK(cudaMalloc(&scramble_constants, constants_size));
        CUDA_CHECK(
            cudaMemcpy(scramble_constants, h_constants, constants_size, cudaMemcpyHostToDevice));

        const size_t blocks_x = next_power2((blocks + dimensions - 1) / dimensions);
        init_scrambled_sobol_kernel<<<dim3(blocks_x, dimensions), threads>>>(states,
                                                                             directions,
                                                                             scramble_constants,
                                                                             offset);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(directions));
        CUDA_CHECK(cudaFree(scramble_constants));
    }

    ~runner()
    {
        CUDA_CHECK(cudaFree(states));
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
    static void create() {}

    static void destroy() {}
};

template<typename Engine>
struct generator_uint : public generator_type
{
    typedef unsigned int data_type;

    __device__
    data_type operator()(Engine* state) const
    {
        return curand(state);
    }
};

template<typename Engine>
struct generator_ullong : public generator_type
{
    typedef unsigned long long int data_type;

    __device__
    data_type operator()(Engine* state) const
    {
        return curand(state);
    }
};

template<typename Engine>
struct generator_uniform : public generator_type
{
    typedef float data_type;

    __device__
    data_type operator()(Engine* state) const
    {
        return curand_uniform(state);
    }
};

template<typename Engine>
struct generator_uniform_double : public generator_type
{
    typedef double data_type;

    __device__
    data_type operator()(Engine* state) const
    {
        return curand_uniform_double(state);
    }
};

template<typename Engine>
struct generator_normal : public generator_type
{
    typedef float data_type;

    __device__
    data_type operator()(Engine* state) const
    {
        return curand_normal(state);
    }
};

template<typename Engine>
struct generator_normal_double : public generator_type
{
    typedef double data_type;

    __device__
    data_type operator()(Engine* state) const
    {
        return curand_normal_double(state);
    }
};

template<typename Engine>
struct generator_log_normal : public generator_type
{
    typedef float data_type;

    __device__
    data_type operator()(Engine* state) const
    {
        return curand_log_normal(state, 0.f, 1.f);
    }
};

template<typename Engine>
struct generator_log_normal_double : public generator_type
{
    typedef double data_type;

    __device__
    data_type operator()(Engine* state) const
    {
        return curand_log_normal_double(state, 0., 1.);
    }
};

template<typename Engine>
struct generator_poisson : public generator_type
{
    // TODO: REMOVE!
    generator_poisson(double l) : lambda(l) {}

    typedef unsigned int data_type;

    __device__
    data_type operator()(Engine* state) const
    {
        return curand_poisson(state, lambda);
    }

    double lambda;
};

template<typename Engine>
struct generator_discrete_poisson : public generator_type
{
    // TODO: REMOVE!
    generator_discrete_poisson(double l) : lambda(l) {}

    typedef unsigned int data_type;

    void create()
    {
        CURAND_CHECK(curandCreatePoissonDistribution(lambda, &discrete_distribution));
    }

    void destroy()
    {
        CURAND_CHECK(curandDestroyDistribution(discrete_distribution));
    }

    __device__
    data_type operator()(Engine* state) const
    {
        return curand_discrete(state, discrete_distribution);
    }

    curandDiscreteDistribution_t discrete_distribution;
    double                       lambda;
};

template<typename Generator, typename State, typename T, distribution Distribution>
struct curand_device_api_benchmark : public primbench::benchmark_interface
{
    curand_device_api_benchmark(Generator             generator, // TODO: REMOVE!
                                curandRngType         engine,
                                size_t                blocks,
                                size_t                threads,
                                size_t                dimensions,
                                size_t                offset,
                                std::optional<double> poisson_lambda = std::nullopt)
        : m_generator(generator) // TODO: REMOVE!
        , m_engine(engine)
        , m_blocks(blocks)
        , m_threads(threads)
        , m_dimensions(dimensions)
        , m_offset(offset)
        , m_poisson_lambda(poisson_lambda)
        // MTGP32 supports a maximum of 200 states.
        , m_mtgp32_states(std::min((size_t)200, blocks))
    {
    }

    primbench::json meta() const override
    {
        auto json = primbench::json{}
                        .add("algo", "curand_device_api")
                        .add("engine", engine_name(m_engine))
                        .add("type", primbench::name<T>())
                        .add("distribution", distribution_name(Distribution))
                        .add("cfg",
                            primbench::json{}
                                .add("blocks", m_blocks)
                                .add("threads", m_threads));

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
        CUDA_CHECK(cudaMalloc(&data, items * sizeof(T)));

        constexpr unsigned long long int seed   = 12345ULL; // TODO: Use state.seed
        constexpr unsigned long long int offset = 6789ULL; // TODO: Use m_offset

        primbench::log("Creating runner");
        runner<State> r(m_dimensions, m_blocks, m_threads, seed, offset);

        state.set_items(items);
        state.add_writes<T>(items);

        state.run([&] { r.generate(m_blocks, m_threads, stream, data, items, m_generator); });

        m_generator.destroy();

        CUDA_CHECK(cudaFree(data));
    }

private:
    Generator             m_generator; // TODO: REMOVE!
    curandRngType      m_engine;
    size_t                m_blocks;
    size_t                m_threads;
    size_t                m_dimensions;
    size_t                m_offset;
    std::optional<double> m_poisson_lambda; // TODO: USE!
    size_t                m_mtgp32_states; // TODO: USE!
};

#define QUEUE(generator, T, State, engine, Distribution, ...)                        \
    executor.queue<curand_device_api_benchmark<generator, State, T, Distribution>>( \
        generator(__VA_ARGS__),                                                      \
        engine,                                                                      \
        blocks,                                                                      \
        threads,                                                                     \
        dimensions,                                                                  \
        offset,                                                                      \
        ##__VA_ARGS__)

#define QUEUE_DISTRIBUTIONS(State, engine)                                                         \
    do                                                                                             \
    {                                                                                              \
        if constexpr(std::is_same_v<State, curandStateSobol64_t>                                  \
                     || std::is_same_v<State, curandStateScrambledSobol64_t>)                    \
        {                                                                                          \
            QUEUE(generator_ullong<State>,                                                         \
                  unsigned long long,                                                              \
                  State,                                                                           \
                  engine,                                                                          \
                  DISTRIBUTION_UNIFORM);                                                           \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            QUEUE(generator_uint<State>, uint32_t, State, engine, DISTRIBUTION_UNIFORM);           \
        }                                                                                          \
                                                                                                   \
        QUEUE(generator_uniform<State>, float, State, engine, DISTRIBUTION_UNIFORM);               \
        QUEUE(generator_uniform_double<State>, double, State, engine, DISTRIBUTION_UNIFORM);       \
        QUEUE(generator_normal<State>, float, State, engine, DISTRIBUTION_NORMAL);                 \
        QUEUE(generator_normal_double<State>, double, State, engine, DISTRIBUTION_NORMAL);         \
        QUEUE(generator_log_normal<State>, float, State, engine, DISTRIBUTION_LOG_NORMAL);         \
        QUEUE(generator_log_normal_double<State>, double, State, engine, DISTRIBUTION_LOG_NORMAL); \
                                                                                                   \
        for(double lambda : poisson_lambdas)                                                       \
        {                                                                                          \
            QUEUE(generator_poisson<State>,                                                        \
                  uint32_t,                                                                        \
                  State,                                                                           \
                  engine,                                                                          \
                  DISTRIBUTION_POISSON,                                                            \
                  lambda);                                                                         \
            QUEUE(generator_discrete_poisson<State>,                                               \
                  uint32_t,                                                                        \
                  State,                                                                           \
                  engine,                                                                          \
                  DISTRIBUTION_DISCRETE_POISSON,                                                   \
                  lambda);                                                                         \
        }                                                                                          \
    }                                                                                              \
    while(0)

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 128 * 1024 * 1024; // In items
    primbench::executor executor(argc, argv, settings);

    auto blocks     = executor.get<size_t>("blocks", 256, "Number of blocks");
    auto threads    = executor.get<size_t>("threads", 256, "Threads per block");
    auto dimensions = executor.get<size_t>("dimensions", 1, "Number of quasi-random dimensions");
    auto offset     = executor.get<size_t>("offset", 0, "Offset of generated pseudo-random values");
    auto poisson_lambdas
        = executor.get<std::vector<double>>("lambda",
                                            {10.0},
                                            "Space-separated list of Poisson lambdas");

    QUEUE_DISTRIBUTIONS(curandStateMRG32k3a_t, CURAND_RNG_PSEUDO_MRG32K3A);
    QUEUE_DISTRIBUTIONS(curandStatePhilox4_32_10_t, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    QUEUE_DISTRIBUTIONS(curandStateXORWOW_t, CURAND_RNG_PSEUDO_XORWOW);
    QUEUE_DISTRIBUTIONS(curandStateMtgp32_t, CURAND_RNG_PSEUDO_MTGP32);
    QUEUE_DISTRIBUTIONS(curandStateSobol32_t, CURAND_RNG_QUASI_SOBOL32);
    QUEUE_DISTRIBUTIONS(curandStateScrambledSobol32_t, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
    QUEUE_DISTRIBUTIONS(curandStateSobol64_t, CURAND_RNG_QUASI_SOBOL64);
    QUEUE_DISTRIBUTIONS(curandStateScrambledSobol64_t, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64);

    executor.run();
}
