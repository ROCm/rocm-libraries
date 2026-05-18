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

// TODO: REMOVE THIS INCLUDE AND FILE
#include "cmdparser.hpp"

// TODO: REMOVE
#include "benchmark/benchmark.h"

#include "rng/xorwow.hpp"

#include "rng/lfsr113.hpp"
#include "rng/mrg.hpp"
#include "rng/mt19937.hpp"
#include "rng/mtgp32.hpp"
#include "rng/philox4x32_10.hpp"
#include "rng/threefry.hpp"
#include "rng/xorwow.hpp"

#include "rng/system.hpp"

#include "rocrand/rocrand_mt19937_precomputed.h"

// Forward declaring generator templates, so includes can be omitted.
// The tuning benchmarks are instantiated in the respective benchmark_tuning_*.cpp
// source files.

// mt19937 needs to be included, as access to threads_per_generator is needed
#include "rng/mt19937.hpp"

// TODO: Remove
#include "benchmark_tuning_setup.hpp"

#include "benchmark_utils.hpp"

// TODO: REMOVE!
struct benchmark_config
{
    std::size_t bytes{};
    double      lambda{};
};

namespace benchmark_tuning
{

template<class T>
struct type_name
{
    // always fails
    static_assert(sizeof(T) == 0,
                  "A specialization of type_name is not implemented for this type.");
};

template<>
struct type_name<unsigned int>
{
    std::string operator()()
    {
        return "unsigned_int";
    }
};

template<>
struct type_name<unsigned char>
{
    std::string operator()()
    {
        return "unsigned_char";
    }
};

template<>
struct type_name<unsigned short>
{
    std::string operator()()
    {
        return "unsigned_short";
    }
};

template<>
struct type_name<unsigned long long>
{
    std::string operator()()
    {
        return "unsigned_long_long";
    }
};

template<>
struct type_name<float>
{
    std::string operator()()
    {
        return "float";
    }
};

template<>
struct type_name<half>
{
    std::string operator()()
    {
        return "half";
    }
};

template<>
struct type_name<double>
{
    std::string operator()()
    {
        return "double";
    }
};

template<class Distribution>
struct distribution_name
{
    // always fails
    static_assert(sizeof(Distribution) == 0,
                  "A specialization of distribution_name is not implemented for this type.");
};

template<class T, class U>
struct distribution_name<rocrand_impl::host::uniform_distribution<T, U>>
{
    std::string operator()()
    {
        return "uniform_" + type_name<T>{}();
    }
};

template<class T, class U, unsigned int I>
struct distribution_name<rocrand_impl::host::normal_distribution<T, U, I>>
{
    std::string operator()()
    {
        return "normal_" + type_name<T>{}();
    }
};

template<class T, class U, unsigned int I>
struct distribution_name<rocrand_impl::host::log_normal_distribution<T, U, I>>
{
    std::string operator()()
    {
        return "log_normal_" + type_name<T>{}();
    }
};

template<>
struct distribution_name<
    rocrand_impl::host::poisson_distribution<rocrand_impl::host::DISCRETE_METHOD_ALIAS>>
{
    std::string operator()()
    {
        return "poisson_unsigned_int";
    }
};

template<>
struct distribution_name<rocrand_impl::host::mrg_poisson_distribution>
{
    std::string operator()()
    {
        return "poisson_unsigned_int";
    }
};

template<class Distribution>
struct default_distribution
{
    // always fails
    static_assert(sizeof(Distribution) == 0,
                  "A specialization of default_distribution is not implemented for this type.");
};

template<class T, class U>
struct default_distribution<rocrand_impl::host::uniform_distribution<T, U>>
{
    auto operator()(const benchmark_config& /*config*/)
    {
        return rocrand_impl::host::uniform_distribution<T, U>{};
    }
};

template<class T, class U, unsigned int I>
struct default_distribution<rocrand_impl::host::normal_distribution<T, U, I>>
{
    auto operator()(const benchmark_config& /*config*/)
    {
        const T mean   = 0;
        const T stddev = 1;
        return rocrand_impl::host::normal_distribution<T, U, I>(mean, stddev);
    }
};

template<class T, class U, unsigned int I>
struct default_distribution<rocrand_impl::host::log_normal_distribution<T, U, I>>
{
    auto operator()(const benchmark_config& /*config*/)
    {
        const T mean   = 0;
        const T stddev = 1;
        return rocrand_impl::host::log_normal_distribution<T, U, I>(mean, stddev);
    }
};

template<>
struct default_distribution<
    rocrand_impl::host::poisson_distribution<rocrand_impl::host::DISCRETE_METHOD_ALIAS>>
{
    auto operator()(const benchmark_config& config)
    {
        return std::get<
            rocrand_impl::host::poisson_distribution<rocrand_impl::host::DISCRETE_METHOD_ALIAS>>(
            m_poisson_manager.get_distribution(config.lambda));
    }

private:
    rocrand_impl::host::poisson_distribution_manager<rocrand_impl::host::DISCRETE_METHOD_ALIAS>
        m_poisson_manager;
};

template<>
struct default_distribution<rocrand_impl::host::mrg_poisson_distribution>
{
    auto operator()(const benchmark_config& config)
    {
        auto poisson_distribution = std::get<
            rocrand_impl::host::poisson_distribution<rocrand_impl::host::DISCRETE_METHOD_ALIAS>>(
            m_poisson_manager.get_distribution(config.lambda));
        return rocrand_impl::host::mrg_poisson_distribution(poisson_distribution);
    }

private:
    rocrand_impl::host::poisson_distribution_manager<rocrand_impl::host::DISCRETE_METHOD_ALIAS>
        m_poisson_manager;
};

template<template<class> class GeneratorTemplate>
struct select_poisson_distribution
{
    using dummy_generator_t = GeneratorTemplate<rocrand_impl::host::static_config_provider<0, 0>>;
    static constexpr inline rocrand_rng_type rng_type = dummy_generator_t::type();
    static constexpr inline bool             is_mrg
        = rng_type == ROCRAND_RNG_PSEUDO_MRG31K3P || rng_type == ROCRAND_RNG_PSEUDO_MRG32K3A;

    using type = std::conditional_t<
        is_mrg,
        rocrand_impl::host::mrg_poisson_distribution,
        rocrand_impl::host::poisson_distribution<rocrand_impl::host::DISCRETE_METHOD_ALIAS>>;
};

template<template<class> class GeneratorTemplate>
using select_poisson_distribution_t = typename select_poisson_distribution<GeneratorTemplate>::type;

} // namespace benchmark_tuning

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
    PRIMBENCH_CHECK(hipMalloc(&data, size * sizeof(T)));

    Generator generator;
    generator.set_stream(stream);

    const auto generate_func = [&]
    {
        default_distribution<Distribution> default_distribution_provider;
        return generator.generate(data, size, default_distribution_provider(config));
    };

    // Warm-up
    RAND_CHECK(generate_func());
    PRIMBENCH_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    PRIMBENCH_CHECK(hipEventCreate(&start));
    PRIMBENCH_CHECK(hipEventCreate(&stop));
    for(auto _ : state)
    {
        PRIMBENCH_CHECK(hipEventRecord(start, stream));
        RAND_CHECK(generate_func());
        PRIMBENCH_CHECK(hipEventRecord(stop, stream));
        PRIMBENCH_CHECK(hipEventSynchronize(stop));

        float elapsed = 0.0f;
        PRIMBENCH_CHECK(hipEventElapsedTime(&elapsed, start, stop));

        state.SetIterationTime(elapsed / 1000.f);
    }
    state.SetBytesProcessed(state.iterations() * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * size);

    PRIMBENCH_CHECK(hipEventDestroy(stop));
    PRIMBENCH_CHECK(hipEventDestroy(start));
    PRIMBENCH_CHECK(hipFree(data));
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
    // TODO: ADD BACK!
    // benchmark_factory.template add_benchmarks<unsigned char>();
    // benchmark_factory.template add_benchmarks<unsigned short>();
    // benchmark_factory.template add_benchmarks<unsigned long long>();
    // benchmark_factory.template add_benchmarks<float>();
    // benchmark_factory.template add_benchmarks<half>();
    // benchmark_factory.template add_benchmarks<double>();
}

} // namespace benchmark_tuning

namespace rocrand_impl::host
{

template<class System, class ConfigProvider>
class lfsr113_generator_template;

template<class System, class Engine, class ConfigProvider>
class mrg_generator_template;

template<class System, class ConfigProvider>
class mtgp32_generator_template;

template<class System, class ConfigProvider>
class philox4x32_10_generator_template;

template<class System, class Engine, class ConfigProvider>
class threefry_generator_template;

template<class System, class ConfigProvider>
class xorwow_generator_template;

template<class DeviceEngine>
struct threefry_device_engine;

} // namespace rocrand_impl::host

// Further forward declarations
namespace rocrand_device
{
class mrg31k3p_engine;
class mrg32k3a_engine;
class threefry2x32_20_engine;
class threefry2x64_20_engine;
class threefry4x32_20_engine;
class threefry4x64_20_engine;
} // namespace rocrand_device

namespace benchmark_tuning
{

// Defining aliases for all generator templates in the benchmark_tuning namespace,
// so the device system can be selected, for the generators that already implement
// both host and device systems

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

template<>
struct output_type_supported<unsigned long long, lfsr113_generator_template>
    : public std::false_type
{};

template<>
struct output_type_supported<unsigned long long, mrg31k3p_generator_template>
    : public std::false_type
{};

template<>
struct output_type_supported<unsigned long long, mrg32k3a_generator_template>
    : public std::false_type
{};

template<>
struct output_type_supported<unsigned long long, mtgp32_generator_template> : public std::false_type
{};

template<>
struct output_type_supported<unsigned long long, mt19937_generator_template>
    : public std::false_type
{};

template<>
struct output_type_supported<unsigned long long, philox4x32_10_generator_template>
    : public std::false_type
{};

template<>
struct output_type_supported<unsigned long long, threefry2x32_20_generator_template>
    : public std::false_type
{};

template<>
struct output_type_supported<unsigned long long, threefry4x32_20_generator_template>
    : public std::false_type
{};

template<>
struct output_type_supported<unsigned long long, xorwow_generator_template> : public std::false_type
{};

template<class T>
struct config_filter<mtgp32_generator_template, T>
{
    static constexpr bool is_enabled(rocrand_impl::host::generator_config config)
    {
        // The current implementation of MTGP32 requires a fixed block size,
        // and the grid size is also limited.
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

template<>
struct distribution_input<threefry2x64_20_generator_template>
{
    using type = unsigned long long;
};

template<>
struct distribution_input<threefry4x64_20_generator_template>
{
    using type = unsigned long long;
};

extern template void add_all_benchmarks_for_generator<lfsr113_generator_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<mrg31k3p_generator_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<mrg32k3a_generator_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<mt19937_generator_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<mtgp32_generator_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<philox4x32_10_generator_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<threefry2x32_20_generator_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<threefry2x64_20_generator_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<threefry4x32_20_generator_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<threefry4x64_20_generator_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<xorwow_generator_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

} // namespace benchmark_tuning

namespace benchmark_tuning
{

template void add_all_benchmarks_for_generator<lfsr113_generator_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

// TODO: ADD ALL OF THESE BACK!
// template void add_all_benchmarks_for_generator<mrg31k3p_generator_template>(
//     std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

// template void add_all_benchmarks_for_generator<mrg32k3a_generator_template>(
//     std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

// template void add_all_benchmarks_for_generator<mt19937_generator_template>(
//     std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

// template void add_all_benchmarks_for_generator<mtgp32_generator_template>(
//     std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

// template void add_all_benchmarks_for_generator<philox4x32_10_generator_template>(
//     std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

// template void add_all_benchmarks_for_generator<threefry2x32_20_generator_template>(
//     std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

// template void add_all_benchmarks_for_generator<threefry2x64_20_generator_template>(
//     std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

// template void add_all_benchmarks_for_generator<threefry4x32_20_generator_template>(
//     std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

// template void add_all_benchmarks_for_generator<threefry4x64_20_generator_template>(
//     std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

// template void add_all_benchmarks_for_generator<xorwow_generator_template>(
//     std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

} // namespace benchmark_tuning

int main(int argc, char** argv)
{
    constexpr std::size_t default_bytes  = 1024 * 1024 * 512;
    constexpr double      default_lambda = 10;

    benchmark::Initialize(&argc, argv);
    cli::Parser parser(argc, argv);
    parser.set_optional<std::size_t>("bytes",
                                     "bytes",
                                     default_bytes,
                                     "number of bytes to generate");
    parser.set_optional<double>("lambda",
                                "lambda",
                                default_lambda,
                                "lambda value to be used in the Poisson distribution");
    parser.run_and_exit_if_error();

    const benchmark_config config{
        parser.get<std::size_t>("bytes"),
        parser.get<double>("lambda"),
    };

    std::vector<benchmark::internal::Benchmark*> benchmarks;
    benchmark_tuning::add_all_benchmarks_for_generator<
        benchmark_tuning::lfsr113_generator_template>(benchmarks, config);

    // TODO: ADD BACK!
    // benchmark_tuning::add_all_benchmarks_for_generator<
    //     benchmark_tuning::mrg31k3p_generator_template>(benchmarks, config);
    // benchmark_tuning::add_all_benchmarks_for_generator<
    //     benchmark_tuning::mrg32k3a_generator_template>(benchmarks, config);
    // benchmark_tuning::add_all_benchmarks_for_generator<
    //     benchmark_tuning::mt19937_generator_template>(benchmarks, config);
    // benchmark_tuning::add_all_benchmarks_for_generator<benchmark_tuning::mtgp32_generator_template>(
    //     benchmarks,
    //     config);
    // benchmark_tuning::add_all_benchmarks_for_generator<
    //     benchmark_tuning::philox4x32_10_generator_template>(benchmarks, config);
    // benchmark_tuning::add_all_benchmarks_for_generator<
    //     benchmark_tuning::threefry2x32_20_generator_template>(benchmarks, config);
    // benchmark_tuning::add_all_benchmarks_for_generator<
    //     benchmark_tuning::threefry2x64_20_generator_template>(benchmarks, config);
    // benchmark_tuning::add_all_benchmarks_for_generator<
    //     benchmark_tuning::threefry4x32_20_generator_template>(benchmarks, config);
    // benchmark_tuning::add_all_benchmarks_for_generator<
    //     benchmark_tuning::threefry4x64_20_generator_template>(benchmarks, config);
    // benchmark_tuning::add_all_benchmarks_for_generator<benchmark_tuning::xorwow_generator_template>(
    //     benchmarks,
    //     config);

    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }
    benchmark::RunSpecifiedBenchmarks();
}
