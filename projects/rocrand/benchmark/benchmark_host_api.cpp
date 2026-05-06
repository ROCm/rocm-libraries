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

#include "benchmark_host_api.hpp"

#define QUEUE_POISSON(engine, ordering, poisson_lambda)                                    \
    executor.queue<host_api_benchmark<unsigned int, DISTRIBUTION_POISSON>>(engine,         \
                                                                           ordering,       \
                                                                           dimensions,     \
                                                                           offset,         \
                                                                           benchmark_host, \
                                                                           poisson_lambda)

#define QUEUE(T, engine, ordering, Distribution)                    \
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
    #define QUEUE_DISTRIBUTIONS(engine, ordering)                                                  \
        do                                                                                         \
        {                                                                                          \
            if(engine != CURAND_RNG_QUASI_SOBOL64 && engine != CURAND_RNG_QUASI_SCRAMBLED_SOBOL64) \
            {                                                                                      \
                QUEUE(unsigned int, engine, ordering, DISTRIBUTION_UNIFORM);                       \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                QUEUE(unsigned long long, engine, ordering, DISTRIBUTION_UNIFORM);                 \
            }                                                                                      \
                                                                                                   \
            QUEUE(float, engine, ordering, DISTRIBUTION_UNIFORM);                                  \
            QUEUE(double, engine, ordering, DISTRIBUTION_UNIFORM);                                 \
                                                                                                   \
            QUEUE(float, engine, ordering, DISTRIBUTION_NORMAL);                                   \
            QUEUE(double, engine, ordering, DISTRIBUTION_NORMAL);                                  \
                                                                                                   \
            QUEUE(float, engine, ordering, DISTRIBUTION_LOG_NORMAL);                               \
            QUEUE(double, engine, ordering, DISTRIBUTION_LOG_NORMAL);                              \
                                                                                                   \
            for(auto poisson_lambda : poisson_lambdas)                                             \
            {                                                                                      \
                QUEUE_POISSON(engine, ordering, poisson_lambda);                                   \
            }                                                                                      \
        }                                                                                          \
        while(0)
#endif

// Quoting programmers-guide.rst:
// ``ROCRAND_ORDERING_PSEUDO_DYNAMIC`` is not supported for generators
// created with ``rocrand_create_generator_host``.
#define QUEUE_PSEUDO(engine)                                           \
    do                                                                 \
    {                                                                  \
        QUEUE_DISTRIBUTIONS(engine, RAND_ORDERING_PSEUDO_DEFAULT);     \
        if(!benchmark_host)                                            \
        {                                                              \
            QUEUE_DISTRIBUTIONS(engine, RAND_ORDERING_PSEUDO_DYNAMIC); \
        }                                                              \
    }                                                                  \
    while(0)

#define QUEUE_QUASI(engine) QUEUE_DISTRIBUTIONS(engine, RAND_ORDERING_QUASI_DEFAULT)

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                    = 128 * 1024 * 1024; // In items
    settings.min_gpu_ms_per_batch    = 1000;
    settings.batch_window_size       = 3;
    settings.noise_tolerance_percent = 3;
    settings.hot                     = true;
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
