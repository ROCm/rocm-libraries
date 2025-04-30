// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_BENCHMARK_TUNING_HPP_
#define ROCRAND_BENCHMARK_TUNING_HPP_

#include <tuple>
#include <type_traits>
#include <utility>

#include "benchmark/benchmark.h"
#include "hip/hip_runtime.h"
#include "rocrand/rocrand.h"

#include "rng/config_types.hpp"
#include "rng/distributions.hpp"
#include "rng/utils/cpp_utils.hpp"

#include "benchmark_rocrand_utils.hpp"
#include "benchmark_tuning_setup.hpp"
#include "distribution_traits.hpp"

namespace benchmark_tuning
{

/// @brief Controls whether the specified type \ref T can be generated
/// by the specified \ref GeneratorTemplate.
/// @tparam T Type of the generated values.
template<class T, template<class> class GeneratorTemplate>
struct output_type_supported : public std::true_type
{};

template<template<class> class GeneratorTemplate>
struct distribution_input
{
    using type = unsigned int;
};

template<template<class> class GeneratorTemplate>
using distribution_input_t = typename distribution_input<GeneratorTemplate>::type;

using rocrand_impl::host::generator_config;

/// @brief Provides a way to opt out from benchmarking certain configs for certain generators and types
/// @note See benchmarked_generators.hpp for specializations
template<template<class> class GeneratorTemplate, class T>
struct config_filter
{
    static constexpr bool is_enabled(generator_config /*config*/)
    {
        return true;
    }
};

/// @brief Runs the googlebenchmark for the specified generator, output type and distribution.
/// @tparam T The generated value type.
/// @tparam Generator The type rocRAND generator to use for the RNG.
/// @tparam Distribution The rocRAND distribution to generate.
/// @param state Benchmarking state.
/// @param config Benchmark config, controlling e.g. the size of the generated random array.
template<class T, class Generator, class Distribution>
void run_benchmark(benchmark::State& state, const benchmark_config& config)
{
    const hipStream_t stream = 0;
    const std::size_t size   = config.bytes / sizeof(T);

    T* data;
    HIP_CHECK(hipMalloc(&data, size * sizeof(T)));

    Generator generator;
    generator.set_stream(stream);

    const auto generate_func = [&]
    {
        default_distribution<Distribution> default_distribution_provider;
        return generator.generate(data, size, default_distribution_provider(config));
    };

    // Warm-up
    ROCRAND_CHECK(generate_func());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    for(auto _ : state)
    {
        HIP_CHECK(hipEventRecord(start, stream));
        ROCRAND_CHECK(generate_func());
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));

        state.SetIterationTime(elapsed / 1000.f);
    }
    state.SetBytesProcessed(state.iterations() * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * size);

    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipFree(data));
}

/// @brief Helper class to instantiate all benchmarks with the specified \ref GeneratorTemplate.
template<template<class ConfigProvider> class GeneratorTemplate>
class generator_benchmark_factory
{
public:
    generator_benchmark_factory(const benchmark_config&                       config,
                                std::vector<benchmark::internal::Benchmark*>& benchmarks)
        : m_config(config), m_benchmarks(benchmarks)
    {}

    /// @brief Instantiate benchmarks with all supported distributions for the specified value type.
    /// @tparam T The generated value type.
    template<class T>
    void add_benchmarks()
    {
        if constexpr(!output_type_supported<T, GeneratorTemplate>::value)
        {
            // If the generator doesn't support the requested type, just return.
            return;
        }
        else if constexpr(std::is_integral_v<T>)
        {
            using uniform_distribution_t
                = rocrand_impl::host::uniform_distribution<T,
                                                           distribution_input_t<GeneratorTemplate>>;
            add_benchmarks_impl<T, uniform_distribution_t>();

            if constexpr(std::is_same_v<T, unsigned int>)
            {
                // The poisson distribution is only supported for unsigned int.
                add_benchmarks_impl<T, select_poisson_distribution_t<GeneratorTemplate>>();
            }
        }
        else if constexpr(std::is_floating_point_v<T> || std::is_same_v<T, half>)
        {
            // float, double and half support these distributions only.
            using uniform_distribution_t
                = rocrand_impl::host::uniform_distribution<T,
                                                           distribution_input_t<GeneratorTemplate>>;
            add_benchmarks_impl<T, uniform_distribution_t>();

            constexpr rocrand_rng_type rng_type
                = rocrand_impl::host::gen_template_type_v<GeneratorTemplate>;

            using normal_distribution_t = rocrand_impl::host::normal_distribution<
                T,
                distribution_input_t<GeneratorTemplate>,
                rocrand_impl::host::normal_distribution_max_input_width<rng_type, T>>;
            add_benchmarks_impl<T, normal_distribution_t>();

            using log_normal_distribution_t = rocrand_impl::host::log_normal_distribution<
                T,
                distribution_input_t<GeneratorTemplate>,
                rocrand_impl::host::log_normal_distribution_max_input_width<rng_type, T>>;
            add_benchmarks_impl<T, log_normal_distribution_t>();
        }
    }

private:
    benchmark_config                              m_config;
    std::vector<benchmark::internal::Benchmark*>& m_benchmarks;

    // This is an array of arrays, listing all {threads,blocks} pairs that run for the benchmark tuning.
    // The elements of the arrays can be controlled with CMake cache variables
    // BENCHMARK_TUNING_THREAD_OPTIONS and BENCHMARK_TUNING_BLOCK_OPTIONS
    static constexpr inline auto s_param_combinations
        = rocrand_impl::cpp_utils::numeric_combinations(thread_options, block_options);

    template<class Distribution, class StaticConfigProvider>
    static std::string get_benchmark_name()
    {
        using Generator                 = GeneratorTemplate<StaticConfigProvider>;
        const rocrand_rng_type rng_type = Generator::type();
        return engine_name(rng_type) + "_" + distribution_name<Distribution>{}() + "_t"
               + std::to_string(StaticConfigProvider::static_config.threads) + "_b"
               + std::to_string(StaticConfigProvider::static_config.blocks);
    }

    template<class T, class Distribution>
    void add_benchmarks_impl()
    {
        add_benchmarks_impl<T, Distribution>(
            std::make_index_sequence<s_param_combinations.size()>());
    }

    template<class T, class Distribution, std::size_t... Indices>
    void add_benchmarks_impl(std::index_sequence<Indices...>)
    {
        // Execute the following lambda for all configuration combinations
        ((
             [&]
             {
                 constexpr auto combination_idx     = Indices;
                 constexpr auto current_combination = s_param_combinations[combination_idx];
                 constexpr auto threads             = std::get<0>(current_combination);
                 constexpr auto blocks              = std::get<1>(current_combination);
                 constexpr auto grid_size           = threads * blocks;

                 // If the grid size is very small, it wouldn't make sense to run the benchmarks for it
                 // The threshold is controlled by CMake cache variable BENCHMARK_TUNING_MIN_GRID_SIZE
                 if constexpr(grid_size < min_benchmarked_grid_size)
                     return;

                 using ConfigProvider = rocrand_impl::host::static_config_provider<threads, blocks>;

                 if constexpr(config_filter<GeneratorTemplate, T>::is_enabled(
                                  ConfigProvider::static_config))
                 {
                     const auto benchmark_name = get_benchmark_name<Distribution, ConfigProvider>();

                     // Append the benchmark to the list using the appropriate ConfigProvider.
                     // Note that captures must be by-value. This class instance won't live to see
                     // the execution of the benchmarks.
                     m_benchmarks.push_back(benchmark::RegisterBenchmark(
                         benchmark_name.c_str(),
                         [*this](auto& state) {
                             run_benchmark<T, GeneratorTemplate<ConfigProvider>, Distribution>(
                                 state,
                                 m_config);
                         }));
                 }
             }()),
         ...);
    }
};

/// @brief Instantiate all benchmarks for the specified \ref GeneratorTemplate.
/// @param benchmarks The list of benchmarks the new benchmarks are appended to.
/// @param config Benchmark config, controlling e.g. the size of the generated random array.
template<template<class ConfigProvider> class GeneratorTemplate>
void add_all_benchmarks_for_generator(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                      const benchmark_config&                       config)
{
    generator_benchmark_factory<GeneratorTemplate> benchmark_factory(config, benchmarks);

    benchmark_factory.template add_benchmarks<unsigned int>();
    benchmark_factory.template add_benchmarks<unsigned char>();
    benchmark_factory.template add_benchmarks<unsigned short>();
    benchmark_factory.template add_benchmarks<unsigned long long>();
    benchmark_factory.template add_benchmarks<float>();
    benchmark_factory.template add_benchmarks<half>();
    benchmark_factory.template add_benchmarks<double>();
}

} // namespace benchmark_tuning

#endif // ROCRAND_BENCHMARK_TUNING_HPP_
