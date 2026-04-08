// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <optional>

#ifdef __HIP__
constexpr rocrand_status RAND_STATUS_TYPE_ERROR = ROCRAND_STATUS_TYPE_ERROR;
#elif defined(__CUDACC__)
constexpr curandStatus_t RAND_STATUS_TYPE_ERROR = CURAND_STATUS_TYPE_ERROR;
#endif

#ifdef __HIP__
using rand_ordering_t = rocrand_ordering;
constexpr rand_ordering_t RAND_ORDERING_PSEUDO_DEFAULT = ROCRAND_ORDERING_PSEUDO_DEFAULT;
constexpr rand_ordering_t RAND_ORDERING_PSEUDO_LEGACY = ROCRAND_ORDERING_PSEUDO_LEGACY;
constexpr rand_ordering_t RAND_ORDERING_PSEUDO_BEST = ROCRAND_ORDERING_PSEUDO_BEST;
constexpr rand_ordering_t RAND_ORDERING_PSEUDO_DYNAMIC = ROCRAND_ORDERING_PSEUDO_DYNAMIC;
constexpr rand_ordering_t RAND_ORDERING_PSEUDO_SEEDED = ROCRAND_ORDERING_PSEUDO_SEEDED;
constexpr rand_ordering_t RAND_ORDERING_QUASI_DEFAULT = ROCRAND_ORDERING_QUASI_DEFAULT;
#elif defined(__CUDACC__)
using rand_ordering_t = curandOrdering_t;
constexpr rand_ordering_t RAND_ORDERING_PSEUDO_DEFAULT = CURAND_ORDERING_PSEUDO_DEFAULT;
constexpr rand_ordering_t RAND_ORDERING_PSEUDO_LEGACY = CURAND_ORDERING_PSEUDO_LEGACY;
constexpr rand_ordering_t RAND_ORDERING_PSEUDO_BEST = CURAND_ORDERING_PSEUDO_BEST;
constexpr rand_ordering_t RAND_ORDERING_PSEUDO_DYNAMIC = CURAND_ORDERING_PSEUDO_DYNAMIC;
constexpr rand_ordering_t RAND_ORDERING_PSEUDO_SEEDED = CURAND_ORDERING_PSEUDO_SEEDED;
constexpr rand_ordering_t RAND_ORDERING_QUASI_DEFAULT = CURAND_ORDERING_QUASI_DEFAULT;
#endif

constexpr const char* ordering_name(ordering_t order)
{
    switch(order)
    {
        case RAND_ORDERING_PSEUDO_DEFAULT: return "default";
        case RAND_ORDERING_PSEUDO_LEGACY: return "legacy";
        case RAND_ORDERING_PSEUDO_BEST: return "best";
        case RAND_ORDERING_PSEUDO_DYNAMIC: return "dynamic";
        case RAND_ORDERING_PSEUDO_SEEDED: return "seeded";
        case RAND_ORDERING_QUASI_DEFAULT: return "quasi_default";
    }
    return "unknown";
}

template<typename T, distribution Distribution>
struct host_api_benchmark : public primbench::benchmark_interface
{
    host_api_benchmark(rng_type_t engine, ordering_t ordering, size_t dimensions, size_t offset, bool benchmark_host, std::optional<double> poisson_lambda = std::nullopt)
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
                        .add("algo", "host_api")
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
        const auto& stream      = state.stream;
        const auto& input_items = state.size;

        const size_t items = (input_items / m_dimensions) * m_dimensions;

        T*          data;
        generator_t generator;

        if(m_benchmark_host)
        {
            primbench::log("Creating host generator");
            data = new T[items];
            RAND_CHECK(create_generator_host(&generator, m_engine));
        }
        else
        {
            primbench::log("Creating device generator");
            PRIMBENCH_CHECK(gpu_malloc(&data, items * sizeof(T)));
            RAND_CHECK(create_generator(&generator, m_engine));
        }

        primbench::log("Setting ordering");
        RAND_CHECK(set_ordering(generator, m_ordering));

        primbench::log("Setting dimensions");
        auto status = set_quasi_random_generator_dimensions(generator, m_dimensions);
        if(status != RAND_STATUS_TYPE_ERROR) // If the RNG is not quasi-random
        {
            RAND_CHECK(status);
        }

        primbench::log("Setting stream");
        RAND_CHECK(set_stream(generator, stream));

        primbench::log("Setting offset");
        status = set_offset(generator, m_offset);
        if(status != RAND_STATUS_TYPE_ERROR) // If the RNG is not pseudo-random
        {
            RAND_CHECK(status);
        }

        // cuRAND doesn't have generators for:
        // - char
        // - short
        // - uniform half
        // - normal half
        // - log normal half
#ifdef __HIP__
        const auto launch = [&]
        {
            if constexpr(Distribution == DISTRIBUTION_UNIFORM && std::is_same_v<T, unsigned int>)
                return rocrand_generate(generator, data, items);
            else if constexpr(Distribution == DISTRIBUTION_UNIFORM && std::is_same_v<T, unsigned char>)
                return rocrand_generate_char(generator, data, items);
            else if constexpr(Distribution == DISTRIBUTION_UNIFORM && std::is_same_v<T, unsigned short>)
                return rocrand_generate_short(generator, data, items);
            else if constexpr(Distribution == DISTRIBUTION_UNIFORM && std::is_same_v<T, __half>)
                return rocrand_generate_uniform_half(generator, data, items);
            else if constexpr(Distribution == DISTRIBUTION_UNIFORM && std::is_same_v<T, float>)
                return rocrand_generate_uniform(generator, data, items);
            else if constexpr(Distribution == DISTRIBUTION_UNIFORM && std::is_same_v<T, double>)
                return rocrand_generate_uniform_double(generator, data, items);
            else if constexpr(Distribution == DISTRIBUTION_NORMAL && std::is_same_v<T, __half>)
                return rocrand_generate_normal_half(generator,
                                                    data,
                                                    items,
                                                    __float2half(0.0f),
                                                    __float2half(1.0f));
            else if constexpr(Distribution == DISTRIBUTION_NORMAL && std::is_same_v<T, float>)
                return rocrand_generate_normal(generator, data, items, 0.0f, 1.0f);
            else if constexpr(Distribution == DISTRIBUTION_NORMAL && std::is_same_v<T, double>)
                return rocrand_generate_normal_double(generator, data, items, 0.0, 1.0);
            else if constexpr(Distribution == DISTRIBUTION_LOG_NORMAL && std::is_same_v<T, __half>)
                return rocrand_generate_log_normal_half(generator,
                                                        data,
                                                        items,
                                                        __float2half(0.0f),
                                                        __float2half(1.0f));
            else if constexpr(Distribution == DISTRIBUTION_LOG_NORMAL && std::is_same_v<T, float>)
                return rocrand_generate_log_normal(generator, data, items, 0.0f, 1.0f);
            else if constexpr(Distribution == DISTRIBUTION_LOG_NORMAL && std::is_same_v<T, double>)
                return rocrand_generate_log_normal_double(generator, data, items, 0.0, 1.0);
            else if constexpr(Distribution == DISTRIBUTION_POISSON)
                return rocrand_generate_poisson(generator, data, items, *m_poisson_lambda);
            else
                static_assert(sizeof(T) == 0, "Missing a constexpr elif.");
        };
#elif defined(__CUDACC__)
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
#endif

        state.set_items(items);
        state.add_writes<T>(items);

        state.run([&] { RAND_CHECK(launch()); });

        RAND_CHECK(destroy_generator(generator));

        if(m_benchmark_host)
        {
            delete[] data;
        }
        else
        {
            PRIMBENCH_CHECK(gpu_free(data));
        }
    }

private:
    rng_type_t            m_engine;
    ordering_t            m_ordering;
    size_t                m_dimensions;
    size_t                m_offset;
    bool                  m_benchmark_host;
    std::optional<double> m_poisson_lambda;
};

#define QUEUE_POISSON(engine, ordering, poisson_lambda)                                            \
    executor.queue<host_api_benchmark<unsigned int, DISTRIBUTION_POISSON>>(engine,         \
                                                                                   ordering,       \
                                                                                   dimensions,     \
                                                                                   offset,         \
                                                                                   benchmark_host, \
                                                                                   poisson_lambda)

#define QUEUE(T, engine, ordering, Distribution)                            \
    executor.queue<host_api_benchmark<T, Distribution>>(engine,     \
                                                                ordering,   \
                                                                dimensions, \
                                                                offset,     \
                                                                benchmark_host)

#ifdef __HIP__
#define QUEUE_DISTRIBUTIONS(engine, ordering)                          \
    do                                                                 \
    {                                                                  \
        QUEUE(unsigned int, engine, ordering, DISTRIBUTION_UNIFORM);   \
        QUEUE(unsigned char, engine, ordering, DISTRIBUTION_UNIFORM);  \
        QUEUE(unsigned short, engine, ordering, DISTRIBUTION_UNIFORM); \
                                                                       \
        QUEUE(__half, engine, ordering, DISTRIBUTION_UNIFORM);         \
        QUEUE(float, engine, ordering, DISTRIBUTION_UNIFORM);          \
        QUEUE(double, engine, ordering, DISTRIBUTION_UNIFORM);         \
                                                                       \
        QUEUE(__half, engine, ordering, DISTRIBUTION_NORMAL);          \
        QUEUE(float, engine, ordering, DISTRIBUTION_NORMAL);           \
        QUEUE(double, engine, ordering, DISTRIBUTION_NORMAL);          \
                                                                       \
        QUEUE(__half, engine, ordering, DISTRIBUTION_LOG_NORMAL);      \
        QUEUE(float, engine, ordering, DISTRIBUTION_LOG_NORMAL);       \
        QUEUE(double, engine, ordering, DISTRIBUTION_LOG_NORMAL);      \
                                                                       \
        for(auto poisson_lambda : poisson_lambdas)                     \
        {                                                              \
            QUEUE_POISSON(engine, ordering, poisson_lambda);           \
        }                                                              \
    }                                                                  \
    while(0)
#elif defined(__CUDACC__)
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
#endif

// Quoting programmers-guide.rst:
// ``ROCRAND_ORDERING_PSEUDO_DYNAMIC`` is not supported for generators
// created with ``rocrand_create_generator_host``.
#define QUEUE_PSEUDO(engine)                                              \
    do                                                                    \
    {                                                                     \
        QUEUE_DISTRIBUTIONS(engine, RAND_ORDERING_PSEUDO_DEFAULT);     \
        if(!benchmark_host)                                               \
        {                                                                 \
            QUEUE_DISTRIBUTIONS(engine, RAND_ORDERING_PSEUDO_DYNAMIC); \
        }                                                                 \
    }                                                                     \
    while(0)

#define QUEUE_QUASI(engine) QUEUE_DISTRIBUTIONS(engine, RAND_ORDERING_QUASI_DEFAULT)

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 128 * 1024 * 1024; // In items
    settings.min_gpu_ms_per_batch = 1000;
    settings.batch_window_size = 3;
    settings.noise_tolerance_percent = 3;
    settings.hot = true;
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

#ifdef __HIP__
    QUEUE_PSEUDO(ROCRAND_RNG_PSEUDO_LFSR113);
    QUEUE_PSEUDO(ROCRAND_RNG_PSEUDO_MRG31K3P);
    QUEUE_PSEUDO(ROCRAND_RNG_PSEUDO_MRG32K3A);
    QUEUE_PSEUDO(ROCRAND_RNG_PSEUDO_MTGP32);
    QUEUE_DISTRIBUTIONS(ROCRAND_RNG_PSEUDO_MT19937, ROCRAND_ORDERING_PSEUDO_DEFAULT);
    QUEUE_PSEUDO(ROCRAND_RNG_PSEUDO_PHILOX4_32_10);
    QUEUE_PSEUDO(ROCRAND_RNG_PSEUDO_THREEFRY2_32_20);
    QUEUE_PSEUDO(ROCRAND_RNG_PSEUDO_THREEFRY2_64_20);
    QUEUE_PSEUDO(ROCRAND_RNG_PSEUDO_THREEFRY4_32_20);
    QUEUE_PSEUDO(ROCRAND_RNG_PSEUDO_THREEFRY4_64_20);
    QUEUE_PSEUDO(ROCRAND_RNG_PSEUDO_XORWOW);
    QUEUE_QUASI(ROCRAND_RNG_QUASI_SOBOL32);
    QUEUE_QUASI(ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32);
    QUEUE_QUASI(ROCRAND_RNG_QUASI_SOBOL64);
    QUEUE_QUASI(ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64);
#elif defined(__CUDACC__)
    QUEUE_PSEUDO(CURAND_RNG_PSEUDO_MRG32K3A);
    QUEUE_PSEUDO(CURAND_RNG_PSEUDO_MTGP32);
    QUEUE_DISTRIBUTIONS(CURAND_RNG_PSEUDO_MT19937, CURAND_ORDERING_PSEUDO_DEFAULT);
    QUEUE_PSEUDO(CURAND_RNG_PSEUDO_PHILOX4_32_10);
    QUEUE_PSEUDO(CURAND_RNG_PSEUDO_XORWOW);
    QUEUE_QUASI(CURAND_RNG_QUASI_SOBOL32);
    QUEUE_QUASI(CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
    QUEUE_QUASI(CURAND_RNG_QUASI_SOBOL64);
    QUEUE_QUASI(CURAND_RNG_QUASI_SCRAMBLED_SOBOL64);
#endif

    executor.run();
}
