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

#include "benchmark_curand_utils.hpp"

#include <cuda_runtime.h>

#include <curand.h>

#include <optional>

enum distribution
{
    DISTRIBUTION_UNIFORM,
    DISTRIBUTION_NORMAL,
    DISTRIBUTION_LOG_NORMAL,
    DISTRIBUTION_POISSON,
};

constexpr const char* distribution_name(distribution distribution)
{
    switch(distribution)
    {
        case DISTRIBUTION_UNIFORM: return "uniform";
        case DISTRIBUTION_NORMAL: return "normal";
        case DISTRIBUTION_LOG_NORMAL: return "log_normal";
        case DISTRIBUTION_POISSON: return "poisson";
    }
    return "unknown";
}

constexpr const char* ordering_name(curandOrdering order)
{
    switch(order)
    {
        case CURAND_ORDERING_PSEUDO_DEFAULT: return "default";
        case CURAND_ORDERING_PSEUDO_LEGACY: return "legacy";
        case CURAND_ORDERING_PSEUDO_BEST: return "best";
        case CURAND_ORDERING_PSEUDO_DYNAMIC: return "dynamic";
        case CURAND_ORDERING_PSEUDO_SEEDED: return "seeded";
        case CURAND_ORDERING_QUASI_DEFAULT: return "quasi_default";
    }
    return "unknown";
}

template<typename T, distribution Distribution>
struct curand_host_api_benchmark : public primbench::benchmark_interface
{
    curand_host_api_benchmark(curandRngType         engine,
                              curandOrdering        ordering,
                              size_t                dimensions,
                              size_t                offset,
                              bool                  benchmark_host,
                              std::optional<double> poisson_lambda = std::nullopt)
        : m_engine(engine)
        , m_ordering(ordering)
        , m_dimensions(dimensions)
        , m_offset(offset)
        , m_benchmark_host(benchmark_host)
        , m_poisson_lambda(poisson_lambda)
    {}

    primbench::json meta() const override
    {
        auto json = primbench::json{}
                        .add("algo", "curand_host_api")
                        .add("type", primbench::name<T>())
                        .add("engine", engine_name(m_engine))
                        .add("ordering", ordering_name(m_ordering))
                        .add("distribution", distribution_name(Distribution));

        if constexpr(Distribution == DISTRIBUTION_POISSON)
        {
            json.add("poisson_lambda", *m_poisson_lambda);
        }

        return json;
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& input_items = state.size;

        const size_t items = (input_items / m_dimensions) * m_dimensions;

        T*                data;
        curandGenerator_t generator;

        if(m_benchmark_host)
        {
            primbench::log("Creating host generator");
            data = new T[items];
            CURAND_CHECK(curandCreateGeneratorHost(&generator, m_engine));
        }
        else
        {
            primbench::log("Creating device generator");
            CUDA_CHECK(cudaMalloc(&data, items * sizeof(T)));
            CURAND_CHECK(curandCreateGenerator(&generator, m_engine));
        }

        primbench::log("Setting ordering");
        CURAND_CHECK(curandSetGeneratorOrdering(generator, m_ordering));

        primbench::log("Setting dimensions");
        curandStatus_t status = curandSetQuasiRandomGeneratorDimensions(generator, m_dimensions);
        if(status != CURAND_STATUS_TYPE_ERROR) // If the RNG is not quasi-random
        {
            CURAND_CHECK(status);
        }

        primbench::log("Setting stream");
        CURAND_CHECK(curandSetStream(generator, stream));

        primbench::log("Setting offset");
        status = curandSetGeneratorOffset(generator, m_offset);
        if(status != CURAND_STATUS_TYPE_ERROR) // If the RNG is not pseudo-random
        {
            CURAND_CHECK(status);
        }

        const auto launch = [&]
        {
            if constexpr(Distribution == DISTRIBUTION_UNIFORM && std::is_same_v<T, unsigned int>)
                return curandGenerate(generator, data, items);
            else if constexpr(Distribution == DISTRIBUTION_UNIFORM && std::is_same_v<T, unsigned long long>)
                return curandGenerateLongLong(generator, data, items);
            else if constexpr(Distribution == DISTRIBUTION_UNIFORM && std::is_same_v<T, float>)
                return curandGenerateUniform(generator, data, items);
            else if constexpr(Distribution == DISTRIBUTION_UNIFORM && std::is_same_v<T, double>)
                return curandGenerateUniformDouble(generator, data, items);
            else if constexpr(Distribution == DISTRIBUTION_NORMAL && std::is_same_v<T, float>)
                return curandGenerateNormal(generator, data, items, 0.0f, 1.0f);
            else if constexpr(Distribution == DISTRIBUTION_NORMAL && std::is_same_v<T, double>)
                return curandGenerateNormalDouble(generator, data, items, 0.0, 1.0);
            else if constexpr(Distribution == DISTRIBUTION_LOG_NORMAL && std::is_same_v<T, float>)
                return curandGenerateLogNormal(generator, data, items, 0.0f, 1.0f);
            else if constexpr(Distribution == DISTRIBUTION_LOG_NORMAL && std::is_same_v<T, double>)
                return curandGenerateLogNormalDouble(generator, data, items, 0.0, 1.0);
            else if constexpr(Distribution == DISTRIBUTION_POISSON)
                return curandGeneratePoisson(generator, data, items, *m_poisson_lambda);
            else
                static_assert(sizeof(T) == 0, "Missing a constexpr elif.");
        };

        state.set_items(items);
        state.add_writes<T>(items);

        state.run([&] { CURAND_CHECK(launch()); });

        CURAND_CHECK(curandDestroyGenerator(generator));

        if(m_benchmark_host)
        {
            delete[] data;
        }
        else
        {
            CUDA_CHECK(cudaFree(data));
        }
    }

private:
    curandRngType         m_engine;
    curandOrdering        m_ordering;
    size_t                m_dimensions;
    size_t                m_offset;
    bool                  m_benchmark_host;
    std::optional<double> m_poisson_lambda;
};

#define QUEUE_POISSON(engine, ordering, poisson_lambda)                                            \
    executor.queue<curand_host_api_benchmark<unsigned int, DISTRIBUTION_POISSON>>(engine,         \
                                                                                   ordering,       \
                                                                                   dimensions,     \
                                                                                   offset,         \
                                                                                   benchmark_host, \
                                                                                   poisson_lambda)

#define QUEUE(T, engine, ordering, Distribution)                            \
    executor.queue<curand_host_api_benchmark<T, Distribution>>(engine,      \
                                                                ordering,   \
                                                                dimensions, \
                                                                offset,     \
                                                                benchmark_host)

#define QUEUE_DISTRIBUTIONS(engine, ordering)                              \
    do                                                                     \
    {                                                                      \
        if(engine != CURAND_RNG_QUASI_SOBOL64 && engine != CURAND_RNG_QUASI_SCRAMBLED_SOBOL64) \
        { \
            QUEUE(unsigned int, engine, ordering, DISTRIBUTION_UNIFORM);       \
        } else {                                                               \
            QUEUE(unsigned long long, engine, ordering, DISTRIBUTION_UNIFORM); \
        }                                                                      \
                                                                               \
        QUEUE(float, engine, ordering, DISTRIBUTION_UNIFORM);                  \
        QUEUE(double, engine, ordering, DISTRIBUTION_UNIFORM);                 \
                                                                               \
        QUEUE(float, engine, ordering, DISTRIBUTION_NORMAL);                   \
        QUEUE(double, engine, ordering, DISTRIBUTION_NORMAL);                  \
                                                                               \
        QUEUE(float, engine, ordering, DISTRIBUTION_LOG_NORMAL);               \
        QUEUE(double, engine, ordering, DISTRIBUTION_LOG_NORMAL);              \
                                                                               \
        for(auto poisson_lambda : poisson_lambdas)                             \
        {                                                                      \
            QUEUE_POISSON(engine, ordering, poisson_lambda);                   \
        }                                                                      \
    }                                                                          \
    while(0)

// Quoting programmers-guide.rst:
// ``ROCRAND_ORDERING_PSEUDO_DYNAMIC`` is not supported for generators
// created with ``rocrand_create_generator_host``.
#define QUEUE_PSEUDO(engine)                                              \
    do                                                                    \
    {                                                                     \
        QUEUE_DISTRIBUTIONS(engine, CURAND_ORDERING_PSEUDO_DEFAULT);      \
        if(!benchmark_host)                                               \
        {                                                                 \
            QUEUE_DISTRIBUTIONS(engine, CURAND_ORDERING_PSEUDO_DYNAMIC);  \
        }                                                                 \
    }                                                                     \
    while(0)

#define QUEUE_QUASI(engine) QUEUE_DISTRIBUTIONS(engine, CURAND_ORDERING_QUASI_DEFAULT)

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 128 * 1024 * 1024; // In items
    settings.min_gpu_ms_per_batch = 100;
    primbench::executor executor(argc, argv, settings, primbench::flags::sync);

    auto dimensions
        = executor.get<size_t>("dimensions", 1, "Number of dimensions of quasi-random values");

    auto offset = executor.get<size_t>("offset", 0, "Offset of generated pseudo-random values");

    auto poisson_lambdas = executor.get<std::vector<double>>(
        "lambda",
        {10.0},
        "Space-separated list of lambdas of Poisson distribution");

    auto benchmark_host
        = executor.get<bool>("host", false, "Run benchmarks on the host instead of on the device");

    QUEUE_PSEUDO(CURAND_RNG_PSEUDO_MRG32K3A);
    QUEUE_PSEUDO(CURAND_RNG_PSEUDO_MTGP32);
    QUEUE_DISTRIBUTIONS(CURAND_RNG_PSEUDO_MT19937, CURAND_ORDERING_PSEUDO_DEFAULT);
    QUEUE_PSEUDO(CURAND_RNG_PSEUDO_PHILOX4_32_10);
    QUEUE_PSEUDO(CURAND_RNG_PSEUDO_XORWOW);
    QUEUE_QUASI(CURAND_RNG_QUASI_SOBOL32);
    QUEUE_QUASI(CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
    QUEUE_QUASI(CURAND_RNG_QUASI_SOBOL64);
    QUEUE_QUASI(CURAND_RNG_QUASI_SCRAMBLED_SOBOL64);

    executor.run();
}
