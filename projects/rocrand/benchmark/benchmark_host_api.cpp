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
#include <vector>

template<typename T, auto Dist, typename EngineType, typename OrderingType, typename... Args>
void queue_host_bench(primbench::executor& executor,
                      EngineType           engine,
                      OrderingType         ordering,
                      size_t               dimensions,
                      size_t               offset,
                      bool                 benchmark_host,
                      Args... args)
{
    executor.queue<host_api_benchmark<T, Dist>>(engine,
                                                ordering,
                                                dimensions,
                                                offset,
                                                benchmark_host,
                                                args...);
}

template<auto Engine, auto Ordering>
void queue_host_permutations(primbench::executor&       executor,
                             size_t                     dimensions,
                             size_t                     offset,
                             bool                       benchmark_host,
                             const std::vector<double>& poisson_lambdas)
{
#ifdef __HIP__
    queue_host_bench<uint32_t, DISTRIBUTION_UNIFORM>(executor,
                                                     Engine,
                                                     Ordering,
                                                     dimensions,
                                                     offset,
                                                     benchmark_host);
    queue_host_bench<uint8_t, DISTRIBUTION_UNIFORM>(executor,
                                                    Engine,
                                                    Ordering,
                                                    dimensions,
                                                    offset,
                                                    benchmark_host);
    queue_host_bench<uint16_t, DISTRIBUTION_UNIFORM>(executor,
                                                     Engine,
                                                     Ordering,
                                                     dimensions,
                                                     offset,
                                                     benchmark_host);

    queue_host_bench<__half, DISTRIBUTION_UNIFORM>(executor,
                                                   Engine,
                                                   Ordering,
                                                   dimensions,
                                                   offset,
                                                   benchmark_host);
    queue_host_bench<__half, DISTRIBUTION_NORMAL>(executor,
                                                  Engine,
                                                  Ordering,
                                                  dimensions,
                                                  offset,
                                                  benchmark_host);
    queue_host_bench<__half, DISTRIBUTION_LOG_NORMAL>(executor,
                                                      Engine,
                                                      Ordering,
                                                      dimensions,
                                                      offset,
                                                      benchmark_host);
#elif defined(__CUDACC__)
    if constexpr(Engine == CURAND_RNG_QUASI_SOBOL64 || Engine == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64)
    {
        queue_host_bench<unsigned long long, DISTRIBUTION_UNIFORM>(executor,
                                                                   Engine,
                                                                   Ordering,
                                                                   dimensions,
                                                                   offset,
                                                                   benchmark_host);
    }
    else
    {
        queue_host_bench<uint32_t, DISTRIBUTION_UNIFORM>(executor,
                                                         Engine,
                                                         Ordering,
                                                         dimensions,
                                                         offset,
                                                         benchmark_host);
    }
#endif

    queue_host_bench<float, DISTRIBUTION_UNIFORM>(executor,
                                                  Engine,
                                                  Ordering,
                                                  dimensions,
                                                  offset,
                                                  benchmark_host);
    queue_host_bench<double, DISTRIBUTION_UNIFORM>(executor,
                                                   Engine,
                                                   Ordering,
                                                   dimensions,
                                                   offset,
                                                   benchmark_host);

    queue_host_bench<float, DISTRIBUTION_NORMAL>(executor,
                                                 Engine,
                                                 Ordering,
                                                 dimensions,
                                                 offset,
                                                 benchmark_host);
    queue_host_bench<double, DISTRIBUTION_NORMAL>(executor,
                                                  Engine,
                                                  Ordering,
                                                  dimensions,
                                                  offset,
                                                  benchmark_host);

    queue_host_bench<float, DISTRIBUTION_LOG_NORMAL>(executor,
                                                     Engine,
                                                     Ordering,
                                                     dimensions,
                                                     offset,
                                                     benchmark_host);
    queue_host_bench<double, DISTRIBUTION_LOG_NORMAL>(executor,
                                                      Engine,
                                                      Ordering,
                                                      dimensions,
                                                      offset,
                                                      benchmark_host);

    for(double lambda : poisson_lambdas)
    {
        queue_host_bench<uint32_t, DISTRIBUTION_POISSON>(executor,
                                                         Engine,
                                                         Ordering,
                                                         dimensions,
                                                         offset,
                                                         benchmark_host,
                                                         lambda);
    }
}

template<auto Engine>
void queue_host_pseudo_permutations(primbench::executor&       executor,
                                    size_t                     dimensions,
                                    size_t                     offset,
                                    bool                       benchmark_host,
                                    const std::vector<double>& poisson_lambdas)
{
    queue_host_permutations<Engine, RAND_ORDERING_PSEUDO_DEFAULT>(executor,
                                                                  dimensions,
                                                                  offset,
                                                                  benchmark_host,
                                                                  poisson_lambdas);

    if(!benchmark_host)
    {
        queue_host_permutations<Engine, RAND_ORDERING_PSEUDO_DYNAMIC>(executor,
                                                                      dimensions,
                                                                      offset,
                                                                      benchmark_host,
                                                                      poisson_lambdas);
    }
}

template<auto EngineVal, auto OrderingVal>
struct engine_ordering
{
    static constexpr auto Engine   = EngineVal;
    static constexpr auto Ordering = OrderingVal;
};

template<auto... Engines>
void queue_all_pseudo_permutations(primbench::executor&       executor,
                                   size_t                     dimensions,
                                   size_t                     offset,
                                   bool                       benchmark_host,
                                   const std::vector<double>& poisson_lambdas)
{
    (queue_host_pseudo_permutations<Engines>(executor,
                                             dimensions,
                                             offset,
                                             benchmark_host,
                                             poisson_lambdas),
     ...);
}

template<typename... Configs>
void queue_all_permutations(primbench::executor&       executor,
                            size_t                     dimensions,
                            size_t                     offset,
                            bool                       benchmark_host,
                            const std::vector<double>& poisson_lambdas)
{
    (queue_host_permutations<Configs::Engine, Configs::Ordering>(executor,
                                                                 dimensions,
                                                                 offset,
                                                                 benchmark_host,
                                                                 poisson_lambdas),
     ...);
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                    = 128 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch    = 1000;
    settings.batch_window_size       = 3;
    settings.noise_tolerance_percent = 3;
    settings.hot                     = true;
    primbench::executor executor(argc, argv, settings, primbench::flags::sync);

    auto dimensions
        = executor.get<size_t>("dimensions", 1, "Number of dimensions of quasi-random values");
    auto offset = executor.get<size_t>("offset", 0, "Offset of generated pseudo-random values");
    auto benchmark_host
        = executor.get<bool>("host", false, "Run benchmarks on the host instead of on the device");
    auto poisson_lambdas = executor.get<std::vector<double>>(
        "lambda",
        {10.0},
        "Space-separated list of lambdas of Poisson distribution");

#ifdef __HIP__
    queue_all_pseudo_permutations<ROCRAND_RNG_PSEUDO_LFSR113,
                                  ROCRAND_RNG_PSEUDO_MRG31K3P,
                                  ROCRAND_RNG_PSEUDO_MRG32K3A,
                                  ROCRAND_RNG_PSEUDO_MTGP32,
                                  ROCRAND_RNG_PSEUDO_PHILOX4_32_10,
                                  ROCRAND_RNG_PSEUDO_THREEFRY2_32_20,
                                  ROCRAND_RNG_PSEUDO_THREEFRY2_64_20,
                                  ROCRAND_RNG_PSEUDO_THREEFRY4_32_20,
                                  ROCRAND_RNG_PSEUDO_THREEFRY4_64_20,
                                  ROCRAND_RNG_PSEUDO_XORWOW>(executor,
                                                             dimensions,
                                                             offset,
                                                             benchmark_host,
                                                             poisson_lambdas);

    queue_all_permutations<
        engine_ordering<ROCRAND_RNG_PSEUDO_MT19937, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
        engine_ordering<ROCRAND_RNG_QUASI_SOBOL32, RAND_ORDERING_QUASI_DEFAULT>,
        engine_ordering<ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32, RAND_ORDERING_QUASI_DEFAULT>,
        engine_ordering<ROCRAND_RNG_QUASI_SOBOL64, RAND_ORDERING_QUASI_DEFAULT>,
        engine_ordering<ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64, RAND_ORDERING_QUASI_DEFAULT>>(
        executor,
        dimensions,
        offset,
        benchmark_host,
        poisson_lambdas);
#elif defined(__CUDACC__)
    queue_all_pseudo_permutations<CURAND_RNG_PSEUDO_MRG32K3A,
                                  CURAND_RNG_PSEUDO_MTGP32,
                                  CURAND_RNG_PSEUDO_PHILOX4_32_10,
                                  CURAND_RNG_PSEUDO_XORWOW>(executor,
                                                            dimensions,
                                                            offset,
                                                            benchmark_host,
                                                            poisson_lambdas);

    queue_all_permutations<
        engine_ordering<CURAND_RNG_PSEUDO_MT19937, CURAND_ORDERING_PSEUDO_DEFAULT>,
        engine_ordering<CURAND_RNG_QUASI_SOBOL32, RAND_ORDERING_QUASI_DEFAULT>,
        engine_ordering<CURAND_RNG_QUASI_SCRAMBLED_SOBOL32, RAND_ORDERING_QUASI_DEFAULT>,
        engine_ordering<CURAND_RNG_QUASI_SOBOL64, RAND_ORDERING_QUASI_DEFAULT>,
        engine_ordering<CURAND_RNG_QUASI_SCRAMBLED_SOBOL64, RAND_ORDERING_QUASI_DEFAULT>>(
        executor,
        dimensions,
        offset,
        benchmark_host,
        poisson_lambdas);
#endif

    executor.run();
}
