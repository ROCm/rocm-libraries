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

#include "benchmark_device_api.hpp"

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
