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

/// \file benchmarked_generators.hpp
/// This header should contain the benchmarking setup for specific rocRAND generators.
/// E.g. specialization of the \c output_type_supported class template.
/// Additionally, an \c extern \c template declaration could be present for each
/// benchmarked generator, to offload compilation of the benchmarks to multiple translation
/// units.

#ifndef ROCRAND_BENCHMARK_TUNING_BENCHMARKED_GENERATORS_HPP_
#define ROCRAND_BENCHMARK_TUNING_BENCHMARKED_GENERATORS_HPP_

#include "benchmark_tuning.hpp"
#include "rng/system.hpp"

#include "rocrand/rocrand_mt19937_precomputed.h"

// Forward declaring generator templates, so includes can be omitted.
// The tuning benchmarks are instantiated in the respective benchmark_tuning_*.cpp
// source files.

// mt19937 needs to be included, as access to threads_per_generator is needed
#include "rng/mt19937.hpp"

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

#endif
