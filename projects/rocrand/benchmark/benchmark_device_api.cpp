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

#include <vector>

template<template<typename> class Gen,
         typename T,
         typename State,
         auto Dist,
         typename EngineType,
         typename... Args>
void queue_device_bench(primbench::executor& executor,
                        EngineType           engine,
                        size_t               blocks,
                        size_t               threads,
                        size_t               dimensions,
                        size_t               offset,
                        Args... args)
{
    executor.queue<device_api_benchmark<Gen<State>, State, T, Dist>>(Gen<State>(args...),
                                                                     engine,
                                                                     blocks,
                                                                     threads,
                                                                     dimensions,
                                                                     offset,
                                                                     args...);
}

template<typename State>
inline constexpr bool uses_64bit_int_v = std::is_same_v<State, rand_state_sobol64_t>
                                         || std::is_same_v<State, rand_state_scrambled_sobol64_t>
#ifdef __HIP__
                                         || std::is_same_v<State, rocrand_state_threefry2x64_20>
                                         || std::is_same_v<State, rocrand_state_threefry4x64_20>
#endif
    ;

template<typename StateType, auto EngineVal>
struct engine_state
{
    using State                  = StateType;
    static constexpr auto Engine = EngineVal;
};

template<typename State, auto Engine>
void queue_device_permutations(primbench::executor&       executor,
                               size_t                     blocks,
                               size_t                     threads,
                               size_t                     dimensions,
                               size_t                     offset,
                               const std::vector<double>& poisson_lambdas)
{
    if constexpr(uses_64bit_int_v<State>)
    {
        queue_device_bench<generator_ullong, unsigned long long, State, DISTRIBUTION_UNIFORM>(
            executor,
            Engine,
            blocks,
            threads,
            dimensions,
            offset);
    }
    else
    {
        queue_device_bench<generator_uint, uint32_t, State, DISTRIBUTION_UNIFORM>(executor,
                                                                                  Engine,
                                                                                  blocks,
                                                                                  threads,
                                                                                  dimensions,
                                                                                  offset);
    }

    queue_device_bench<generator_uniform, float, State, DISTRIBUTION_UNIFORM>(executor,
                                                                              Engine,
                                                                              blocks,
                                                                              threads,
                                                                              dimensions,
                                                                              offset);
    queue_device_bench<generator_uniform_double, double, State, DISTRIBUTION_UNIFORM>(executor,
                                                                                      Engine,
                                                                                      blocks,
                                                                                      threads,
                                                                                      dimensions,
                                                                                      offset);
    queue_device_bench<generator_normal, float, State, DISTRIBUTION_NORMAL>(executor,
                                                                            Engine,
                                                                            blocks,
                                                                            threads,
                                                                            dimensions,
                                                                            offset);
    queue_device_bench<generator_normal_double, double, State, DISTRIBUTION_NORMAL>(executor,
                                                                                    Engine,
                                                                                    blocks,
                                                                                    threads,
                                                                                    dimensions,
                                                                                    offset);
    queue_device_bench<generator_log_normal, float, State, DISTRIBUTION_LOG_NORMAL>(executor,
                                                                                    Engine,
                                                                                    blocks,
                                                                                    threads,
                                                                                    dimensions,
                                                                                    offset);
    queue_device_bench<generator_log_normal_double, double, State, DISTRIBUTION_LOG_NORMAL>(
        executor,
        Engine,
        blocks,
        threads,
        dimensions,
        offset);

    for(double lambda : poisson_lambdas)
    {
        queue_device_bench<generator_poisson, uint32_t, State, DISTRIBUTION_POISSON>(executor,
                                                                                     Engine,
                                                                                     blocks,
                                                                                     threads,
                                                                                     dimensions,
                                                                                     offset,
                                                                                     lambda);
        queue_device_bench<generator_discrete_poisson,
                           uint32_t,
                           State,
                           DISTRIBUTION_DISCRETE_POISSON>(executor,
                                                          Engine,
                                                          blocks,
                                                          threads,
                                                          dimensions,
                                                          offset,
                                                          lambda);
    }

#ifdef __HIP__
    queue_device_bench<generator_discrete_custom, uint32_t, State, DISTRIBUTION_DISCRETE_CUSTOM>(
        executor,
        Engine,
        blocks,
        threads,
        dimensions,
        offset);
#endif
}

template<typename... Configs>
void queue_all_permutations(primbench::executor&       executor,
                            size_t                     blocks,
                            size_t                     threads,
                            size_t                     dimensions,
                            size_t                     offset,
                            const std::vector<double>& poisson_lambdas)
{
    (queue_device_permutations<typename Configs::State, Configs::Engine>(executor,
                                                                         blocks,
                                                                         threads,
                                                                         dimensions,
                                                                         offset,
                                                                         poisson_lambdas),
     ...);
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 128 * primbench::MiB; // In items
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

    queue_all_permutations<
        engine_state<rand_state_mrg32k3a_t, RAND_RNG_PSEUDO_MRG32K3A>,
        engine_state<rand_state_philox4x32_10_t, RAND_RNG_PSEUDO_PHILOX4_32_10>,
        engine_state<rand_state_xorwow_t, RAND_RNG_PSEUDO_XORWOW>,
        engine_state<rand_state_mtgp32_t, RAND_RNG_PSEUDO_MTGP32>,
        engine_state<rand_state_sobol32_t, RAND_RNG_QUASI_SOBOL32>,
        engine_state<rand_state_scrambled_sobol32_t, RAND_RNG_QUASI_SCRAMBLED_SOBOL32>,
        engine_state<rand_state_sobol64_t, RAND_RNG_QUASI_SOBOL64>,
        engine_state<rand_state_scrambled_sobol64_t, RAND_RNG_QUASI_SCRAMBLED_SOBOL64>
#ifdef __HIP__
        ,
        engine_state<rocrand_state_lfsr113, ROCRAND_RNG_PSEUDO_LFSR113>,
        engine_state<rocrand_state_mrg31k3p, ROCRAND_RNG_PSEUDO_MRG31K3P>,
        engine_state<rocrand_state_threefry2x32_20, ROCRAND_RNG_PSEUDO_THREEFRY2_32_20>,
        engine_state<rocrand_state_threefry4x32_20, ROCRAND_RNG_PSEUDO_THREEFRY4_32_20>,
        engine_state<rocrand_state_threefry2x64_20, ROCRAND_RNG_PSEUDO_THREEFRY2_64_20>,
        engine_state<rocrand_state_threefry4x64_20, ROCRAND_RNG_PSEUDO_THREEFRY4_64_20>
#endif
        >(executor, blocks, threads, dimensions, offset, poisson_lambdas);

    executor.run();
}
