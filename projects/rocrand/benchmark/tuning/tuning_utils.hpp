// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "benchmark_host_api.hpp"

#include "rng/lfsr113.hpp"
#include "rng/mrg.hpp"
#include "rng/mt19937.hpp"
#include "rng/mtgp32.hpp"
#include "rng/philox4x32_10.hpp"
#include "rng/threefry.hpp"
#include "rng/xorwow.hpp"

#include "rocrand/rocrand_mt19937_precomputed.h"

#include <array>
#include <tuple>
#include <type_traits>
#include <vector>

template<class... Ts>
constexpr auto unsigned_array(const Ts... args)
{
    return std::array{static_cast<uint32_t>(args)...};
}

constexpr inline uint32_t min_benchmarked_grid_size = BENCHMARK_TUNING_MIN_GRID_SIZE;
constexpr inline auto     thread_options = unsigned_array(BENCHMARK_TUNING_THREAD_OPTIONS);
constexpr inline auto     block_options  = unsigned_array(BENCHMARK_TUNING_BLOCK_OPTIONS);

static constexpr inline auto permutations
    = rocrand_impl::cpp_utils::numeric_combinations(thread_options, block_options);

namespace benchmark_tuning
{

// --------------- Generator template aliases ---------------

template<class ConfigProvider>
using lfsr113_generator_template
    = rocrand_impl::host::lfsr113_generator_template<rocrand_impl::system::device_system,
                                                     ConfigProvider>;

template<class ConfigProvider>
using mrg31k3p_generator_template
    = rocrand_impl::host::mrg_generator_template<rocrand_impl::system::device_system,
                                                 rocrand_device::mrg31k3p_engine,
                                                 ConfigProvider>;

template<class ConfigProvider>
using mrg32k3a_generator_template
    = rocrand_impl::host::mrg_generator_template<rocrand_impl::system::device_system,
                                                 rocrand_device::mrg32k3a_engine,
                                                 ConfigProvider>;

template<class ConfigProvider>
using mtgp32_generator_template
    = rocrand_impl::host::mtgp32_generator_template<rocrand_impl::system::device_system,
                                                    ConfigProvider>;

template<class ConfigProvider>
using mt19937_generator_template
    = rocrand_impl::host::mt19937_generator_template<rocrand_impl::system::device_system,
                                                     ConfigProvider>;

template<class ConfigProvider>
using philox4x32_10_generator_template
    = rocrand_impl::host::philox4x32_10_generator_template<rocrand_impl::system::device_system,
                                                           ConfigProvider>;

template<class ConfigProvider>
using threefry2x32_20_generator_template = rocrand_impl::host::threefry_generator_template<
    rocrand_impl::system::device_system,
    rocrand_impl::host::threefry_device_engine<rocrand_device::threefry2x32_20_engine>,
    ConfigProvider>;

template<class ConfigProvider>
using threefry2x64_20_generator_template = rocrand_impl::host::threefry_generator_template<
    rocrand_impl::system::device_system,
    rocrand_impl::host::threefry_device_engine<rocrand_device::threefry2x64_20_engine>,
    ConfigProvider>;

template<class ConfigProvider>
using threefry4x32_20_generator_template = rocrand_impl::host::threefry_generator_template<
    rocrand_impl::system::device_system,
    rocrand_impl::host::threefry_device_engine<rocrand_device::threefry4x32_20_engine>,
    ConfigProvider>;

template<class ConfigProvider>
using threefry4x64_20_generator_template = rocrand_impl::host::threefry_generator_template<
    rocrand_impl::system::device_system,
    rocrand_impl::host::threefry_device_engine<rocrand_device::threefry4x64_20_engine>,
    ConfigProvider>;

template<class ConfigProvider>
using xorwow_generator_template
    = rocrand_impl::host::xorwow_generator_template<rocrand_impl::system::device_system,
                                                    ConfigProvider>;

// --------------- output_type_supported ---------------

template<typename T, template<class> class GeneratorTemplate>
struct output_type_supported : std::true_type
{};

#define DISABLE_ULL(Generator)                                                    \
    template<>                                                                    \
    struct output_type_supported<unsigned long long, Generator> : std::false_type \
    {}

DISABLE_ULL(lfsr113_generator_template);
DISABLE_ULL(mrg31k3p_generator_template);
DISABLE_ULL(mrg32k3a_generator_template);
DISABLE_ULL(mtgp32_generator_template);
DISABLE_ULL(mt19937_generator_template);
DISABLE_ULL(philox4x32_10_generator_template);
DISABLE_ULL(threefry2x32_20_generator_template);
DISABLE_ULL(threefry4x32_20_generator_template);
DISABLE_ULL(xorwow_generator_template);

#undef DISABLE_ULL

// --------------- config_filter ---------------

template<template<class> class GeneratorTemplate, class T>
struct config_filter
{
    static constexpr bool is_enabled(rocrand_impl::host::generator_config)
    {
        return true;
    }
};

template<class T>
struct config_filter<mtgp32_generator_template, T>
{
    static constexpr bool is_enabled(rocrand_impl::host::generator_config config)
    {
        return config.blocks <= 512 && config.threads == 256;
    }
};

template<class T>
struct config_filter<mt19937_generator_template, T>
{
    static constexpr bool is_enabled(rocrand_impl::host::generator_config config)
    {
        return (config.blocks * config.threads
                / rocrand_impl::host::mt19937_octo_engine::threads_per_generator)
               <= mt19937_jumps_radix * mt19937_jumps_radix;
    }
};

// --------------- Queuing templates ---------------

template<class T, distribution Dist, template<class> class GeneratorTemplate, class ConfigProvider>
void queue_one(primbench::executor& executor,
               size_t               dimensions,
               size_t               offset,
               bool                 benchmark_host,
               double               poisson_lambda)
{
    using generator_t       = GeneratorTemplate<ConfigProvider>;
    constexpr auto rng_type = generator_t::type();

    if constexpr(Dist == DISTRIBUTION_POISSON)
    {
        executor.queue<host_api_benchmark<T, Dist, generator_t>>(rng_type,
                                                                 RAND_ORDERING_PSEUDO_DYNAMIC,
                                                                 dimensions,
                                                                 offset,
                                                                 benchmark_host,
                                                                 poisson_lambda);
    }
    else
    {
        executor.queue<host_api_benchmark<T, Dist, generator_t>>(rng_type,
                                                                 RAND_ORDERING_PSEUDO_DEFAULT,
                                                                 dimensions,
                                                                 offset,
                                                                 benchmark_host);
    }
}

template<typename T, template<class> class GeneratorTemplate, class ConfigProvider>
void queue_distributions(primbench::executor&       executor,
                         size_t                     dimensions,
                         size_t                     offset,
                         bool                       benchmark_host,
                         const std::vector<double>& poisson_lambdas)
{
    if constexpr(!output_type_supported<T, GeneratorTemplate>::value)
    {
        return;
    }
    else if constexpr(!config_filter<GeneratorTemplate, T>::is_enabled(
                          ConfigProvider::static_config))
    {
        return;
    }
    else if constexpr(std::is_integral_v<T>)
    {
        queue_one<T, DISTRIBUTION_UNIFORM, GeneratorTemplate, ConfigProvider>(executor,
                                                                              dimensions,
                                                                              offset,
                                                                              benchmark_host,
                                                                              0.0);

        if constexpr(std::is_same_v<T, uint32_t>)
        {
            for(double lambda : poisson_lambdas)
            {
                queue_one<T, DISTRIBUTION_POISSON, GeneratorTemplate, ConfigProvider>(
                    executor,
                    dimensions,
                    offset,
                    benchmark_host,
                    lambda);
            }
        }
    }
    else
    {
        queue_one<T, DISTRIBUTION_UNIFORM, GeneratorTemplate, ConfigProvider>(executor,
                                                                              dimensions,
                                                                              offset,
                                                                              benchmark_host,
                                                                              0.0);
        queue_one<T, DISTRIBUTION_NORMAL, GeneratorTemplate, ConfigProvider>(executor,
                                                                             dimensions,
                                                                             offset,
                                                                             benchmark_host,
                                                                             0.0);
        queue_one<T, DISTRIBUTION_LOG_NORMAL, GeneratorTemplate, ConfigProvider>(executor,
                                                                                 dimensions,
                                                                                 offset,
                                                                                 benchmark_host,
                                                                                 0.0);
    }
}

template<template<class> class GeneratorTemplate, class ConfigProvider>
void queue_all_types(primbench::executor&       executor,
                     size_t                     dimensions,
                     size_t                     offset,
                     bool                       benchmark_host,
                     const std::vector<double>& poisson_lambdas)
{
    queue_distributions<uint32_t, GeneratorTemplate, ConfigProvider>(executor,
                                                                     dimensions,
                                                                     offset,
                                                                     benchmark_host,
                                                                     poisson_lambdas);
    queue_distributions<uint8_t, GeneratorTemplate, ConfigProvider>(executor,
                                                                    dimensions,
                                                                    offset,
                                                                    benchmark_host,
                                                                    poisson_lambdas);
    queue_distributions<uint16_t, GeneratorTemplate, ConfigProvider>(executor,
                                                                     dimensions,
                                                                     offset,
                                                                     benchmark_host,
                                                                     poisson_lambdas);
    queue_distributions<unsigned long long, GeneratorTemplate, ConfigProvider>(executor,
                                                                               dimensions,
                                                                               offset,
                                                                               benchmark_host,
                                                                               poisson_lambdas);
    queue_distributions<float, GeneratorTemplate, ConfigProvider>(executor,
                                                                  dimensions,
                                                                  offset,
                                                                  benchmark_host,
                                                                  poisson_lambdas);
    queue_distributions<half, GeneratorTemplate, ConfigProvider>(executor,
                                                                 dimensions,
                                                                 offset,
                                                                 benchmark_host,
                                                                 poisson_lambdas);
    queue_distributions<double, GeneratorTemplate, ConfigProvider>(executor,
                                                                   dimensions,
                                                                   offset,
                                                                   benchmark_host,
                                                                   poisson_lambdas);
}

template<template<class> class GeneratorTemplate, std::size_t... Indices>
void queue_generator_impl(std::index_sequence<Indices...>,
                          primbench::executor&       executor,
                          size_t                     dimensions,
                          size_t                     offset,
                          bool                       benchmark_host,
                          const std::vector<double>& poisson_lambdas)
{
    (
        [&]
        {
            constexpr auto combo     = permutations[Indices];
            constexpr auto threads   = std::get<0>(combo);
            constexpr auto blocks    = std::get<1>(combo);
            constexpr auto grid_size = threads * blocks;

            if constexpr(grid_size < min_benchmarked_grid_size)
            {
                return;
            }

            using config_provider = rocrand_impl::host::static_config_provider<threads, blocks>;

            queue_all_types<GeneratorTemplate, config_provider>(executor,
                                                                dimensions,
                                                                offset,
                                                                benchmark_host,
                                                                poisson_lambdas);
        }(),
        ...);
}

template<template<class> class GeneratorTemplate>
void queue_generator(primbench::executor&       executor,
                     size_t                     dimensions,
                     size_t                     offset,
                     bool                       benchmark_host,
                     const std::vector<double>& poisson_lambdas)
{
    queue_generator_impl<GeneratorTemplate>(std::make_index_sequence<permutations.size()>{},
                                            executor,
                                            dimensions,
                                            offset,
                                            benchmark_host,
                                            poisson_lambdas);
}

} // namespace benchmark_tuning
