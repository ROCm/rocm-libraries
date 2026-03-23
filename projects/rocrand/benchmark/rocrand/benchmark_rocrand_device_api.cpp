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

#include <hip/hip_runtime.h>

#include <rocrand/rocrand.h>
#include <rocrand/rocrand_kernel.h>
#include <rocrand/rocrand_mtgp32_11213.h>

#include <optional>

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
__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void generate_kernel(rocrand_state_mtgp32* states, T* data, const size_t size, Generator generator)
{
    const unsigned int state_id = blockIdx.x;
    unsigned int       index    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int       stride   = gridDim.x * blockDim.x;

    __shared__
    rocrand_state_mtgp32 state;
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
__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void init_sobol_kernel(EngineState* states, SobolType* directions, SobolType offset)
{
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;
    EngineState        state;
    rocrand_init(&directions[dimension * sizeof(SobolType) * 8], offset + state_id, &state);
    states[gridDim.x * blockDim.x * dimension + state_id] = state;
}

template<typename EngineState, typename SobolType>
__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
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
__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
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

    __device__
    data_type operator()(Engine* state) const
    {
        return rocrand(state);
    }
};

template<typename Engine>
struct generator_ullong : public generator_type
{
    typedef unsigned long long int data_type;

    __device__
    data_type operator()(Engine* state) const
    {
        return rocrand(state);
    }
};

template<typename Engine>
struct generator_uniform : public generator_type
{
    typedef float data_type;

    __device__
    data_type operator()(Engine* state) const
    {
        return rocrand_uniform(state);
    }
};

template<typename Engine>
struct generator_uniform_double : public generator_type
{
    typedef double data_type;

    __device__
    data_type operator()(Engine* state) const
    {
        return rocrand_uniform_double(state);
    }
};

template<typename Engine>
struct generator_normal : public generator_type
{
    typedef float data_type;

    __device__
    data_type operator()(Engine* state) const
    {
        return rocrand_normal(state);
    }
};

template<typename Engine>
struct generator_normal_double : public generator_type
{
    typedef double data_type;

    __device__
    data_type operator()(Engine* state) const
    {
        return rocrand_normal_double(state);
    }
};

template<typename Engine>
struct generator_log_normal : public generator_type
{
    typedef float data_type;

    __device__
    data_type operator()(Engine* state) const
    {
        return rocrand_log_normal(state, 0.f, 1.f);
    }
};

template<typename Engine>
struct generator_log_normal_double : public generator_type
{
    typedef double data_type;

    __device__
    data_type operator()(Engine* state) const
    {
        return rocrand_log_normal_double(state, 0., 1.);
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
        return rocrand_poisson(state, lambda);
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
        ROCRAND_CHECK(rocrand_create_poisson_distribution(lambda, &discrete_distribution));
    }

    void destroy()
    {
        ROCRAND_CHECK(rocrand_destroy_discrete_distribution(discrete_distribution));
    }

    __device__
    data_type operator()(Engine* state) const
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
    data_type operator()(Engine* state) const
    {
        return rocrand_discrete(state, discrete_distribution);
    }

    rocrand_discrete_distribution discrete_distribution;
};

template<typename Generator, typename State, typename T, distribution Distribution>
struct rocrand_device_api_benchmark : public primbench::benchmark_interface
{
    rocrand_device_api_benchmark(Generator             generator, // TODO: REMOVE!
                                 rocrand_rng_type      engine,
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
                        .add("algo", "rocrand_device_api")
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
        HIP_CHECK(hipMalloc(&data, items * sizeof(T)));

        constexpr unsigned long long int seed   = 12345ULL; // TODO: Use state.seed
        constexpr unsigned long long int offset = 6789ULL; // TODO: Use m_offset

        primbench::log("Creating runner");
        runner<State> r(m_dimensions, m_blocks, m_threads, seed, offset);

        state.set_items(items);
        state.add_writes<T>(items);

        state.run([&] { r.generate(m_blocks, m_threads, stream, data, items, m_generator); });

        m_generator.destroy();

        HIP_CHECK(hipFree(data));
    }

private:
    Generator             m_generator; // TODO: REMOVE!
    rocrand_rng_type      m_engine;
    size_t                m_blocks;
    size_t                m_threads;
    size_t                m_dimensions;
    size_t                m_offset;
    std::optional<double> m_poisson_lambda; // TODO: USE!
    size_t                m_mtgp32_states; // TODO: USE!
};

#define QUEUE(generator, T, State, engine, Distribution, ...)                        \
    executor.queue<rocrand_device_api_benchmark<generator, State, T, Distribution>>( \
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
        if constexpr(std::is_same_v<State, rocrand_state_sobol64>                                  \
                     || std::is_same_v<State, rocrand_state_scrambled_sobol64>                     \
                     || std::is_same_v<State, rocrand_state_threefry2x64_20>                       \
                     || std::is_same_v<State, rocrand_state_threefry4x64_20>)                      \
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
                                                                                                   \
        QUEUE(generator_discrete_custom<State>,                                                    \
              uint32_t,                                                                            \
              State,                                                                               \
              engine,                                                                              \
              DISTRIBUTION_DISCRETE_CUSTOM);                                                       \
    }                                                                                              \
    while(0)

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 128 * 1024 * 1024; // In items
    settings.min_gpu_ms_per_batch = 100;
    settings.hot = true;
    primbench::executor executor(argc, argv, settings);

    auto blocks     = executor.get<size_t>("blocks", 256, "Number of blocks");
    auto threads    = executor.get<size_t>("threads", 256, "Threads per block");
    auto dimensions = executor.get<size_t>("dimensions", 1, "Number of quasi-random dimensions");
    auto offset     = executor.get<size_t>("offset", 0, "Offset of generated pseudo-random values");
    auto poisson_lambdas
        = executor.get<std::vector<double>>("lambda",
                                            {10.0},
                                            "Space-separated list of Poisson lambdas");

    QUEUE_DISTRIBUTIONS(rocrand_state_lfsr113, ROCRAND_RNG_PSEUDO_LFSR113);
    QUEUE_DISTRIBUTIONS(rocrand_state_mrg31k3p, ROCRAND_RNG_PSEUDO_MRG31K3P);
    QUEUE_DISTRIBUTIONS(rocrand_state_mrg32k3a, ROCRAND_RNG_PSEUDO_MRG32K3A);
    QUEUE_DISTRIBUTIONS(rocrand_state_philox4x32_10, ROCRAND_RNG_PSEUDO_PHILOX4_32_10);
    QUEUE_DISTRIBUTIONS(rocrand_state_threefry2x32_20, ROCRAND_RNG_PSEUDO_THREEFRY2_32_20);
    QUEUE_DISTRIBUTIONS(rocrand_state_threefry4x32_20, ROCRAND_RNG_PSEUDO_THREEFRY4_32_20);
    QUEUE_DISTRIBUTIONS(rocrand_state_threefry2x64_20, ROCRAND_RNG_PSEUDO_THREEFRY2_64_20);
    QUEUE_DISTRIBUTIONS(rocrand_state_threefry4x64_20, ROCRAND_RNG_PSEUDO_THREEFRY4_64_20);
    QUEUE_DISTRIBUTIONS(rocrand_state_xorwow, ROCRAND_RNG_PSEUDO_XORWOW);
    QUEUE_DISTRIBUTIONS(rocrand_state_mtgp32, ROCRAND_RNG_PSEUDO_MTGP32);
    QUEUE_DISTRIBUTIONS(rocrand_state_sobol32, ROCRAND_RNG_QUASI_SOBOL32);
    QUEUE_DISTRIBUTIONS(rocrand_state_scrambled_sobol32, ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32);
    QUEUE_DISTRIBUTIONS(rocrand_state_sobol64, ROCRAND_RNG_QUASI_SOBOL64);
    QUEUE_DISTRIBUTIONS(rocrand_state_scrambled_sobol64, ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64);

    executor.run();
}
