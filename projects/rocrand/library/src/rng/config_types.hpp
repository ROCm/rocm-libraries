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

#ifndef ROCRAND_RNG_CONFIG_TYPES_H_
#define ROCRAND_RNG_CONFIG_TYPES_H_

#include "common.hpp"
#include "rocrand/rocrand.h"
#include "utils/cpp_utils.hpp"

#include <hip/hip_runtime.h>

#include <atomic>
#include <cassert>
#include <limits>
#include <numeric>
#include <string>

namespace rocrand_impl::host
{

/// \brief Represents a device target's processor architecture.
enum class target_arch : unsigned int
{
    // This must be zero, to initialize the device -> architecture cache
    invalid = 0,
    gfx900  = 900,
    gfx902  = 902,
    gfx904  = 904,
    gfx906  = 906,
    gfx908  = 908,
    gfx909  = 909,
    gfx90a  = 910,
    gfx942  = 942,
    gfx1030 = 1030,
    gfx1100 = 1100,
    gfx1101 = 1101,
    gfx1102 = 1102,
    gfx1201 = 1201,
    unknown = std::numeric_limits<unsigned int>::max(),
};

/// @brief Returns the detected processor architecture of the device that is currently compiled against.
__host__ __device__ constexpr target_arch get_device_arch()
{
#if !USE_DEVICE_DISPATCH
    return target_arch::unknown;
#elif defined(__gfx900__)
    return target_arch::gfx900;
#elif defined(__gfx902__)
    return target_arch::gfx902;
#elif defined(__gfx904__)
    return target_arch::gfx904;
#elif defined(__gfx906__)
    return target_arch::gfx906;
#elif defined(__gfx908__)
    return target_arch::gfx908;
#elif defined(__gfx909__)
    return target_arch::gfx909;
#elif defined(__gfx90a__)
    return target_arch::gfx90a;
#elif defined(__gfx942__)
    return target_arch::gfx942;
#elif defined(__gfx1030__)
    return target_arch::gfx1030;
#elif defined(__gfx1100__)
    return target_arch::gfx1100;
#elif defined(__gfx1101__)
    return target_arch::gfx1101;
#elif defined(__gfx1102__)
    return target_arch::gfx1102;
#elif defined(__gfx1201__)
    return target_arch::gfx1201;
#else
    return target_arch::unknown;
#endif
}

/// @brief Maps the device architecture name to the \ref target_arch enum.
/// @param arch_name The name of the architecture. Normally queried from the HIP runtime.
/// @return The detected element of the enum, of \ref target_arch::unknown, if not found.
inline target_arch parse_gcn_arch(const std::string& arch_name)
{
    const std::string target_names[]         = {"gfx900",
                                                "gfx902",
                                                "gfx904",
                                                "gfx906",
                                                "gfx908",
                                                "gfx909",
                                                "gfx90a",
                                                "gfx942",
                                                "gfx1030",
                                                "gfx1100",
                                                "gfx1101",
                                                "gfx1102",
                                                "gfx1201"};
    const target_arch target_architectures[] = {
        target_arch::gfx900,
        target_arch::gfx902,
        target_arch::gfx904,
        target_arch::gfx906,
        target_arch::gfx908,
        target_arch::gfx909,
        target_arch::gfx90a,
        target_arch::gfx942,
        target_arch::gfx1030,
        target_arch::gfx1100,
        target_arch::gfx1101,
        target_arch::gfx1102,
        target_arch::gfx1201,
    };
    static_assert(sizeof(target_names) / sizeof(target_names[0])
                      == sizeof(target_architectures) / sizeof(target_architectures[0]),
                  "target_names and target_architectures should have the same number of elements");
    constexpr auto num_architectures = sizeof(target_names) / sizeof(target_names[0]);

    for(unsigned int i = 0; i < num_architectures; ++i)
    {
        if(arch_name.find(target_names[i]) == 0)
        {
            return target_architectures[i];
        }
    }
    return target_arch::unknown;
}

/// @brief Queries the device architecture of a HIP device.
/// @param device_id The ID of the HIP device.
/// @param arch Out parameter. The detected device architecture is set here.
/// @return \ref hipSuccess if the querying was successful, a different error code otherwise.
inline hipError_t get_device_arch([[maybe_unused]] int device_id, target_arch& arch)
{
#if USE_DEVICE_DISPATCH
    constexpr unsigned int          device_arch_cache_size             = 512;
    static std::atomic<target_arch> arch_cache[device_arch_cache_size] = {};

    assert(device_id >= 0);
    if(static_cast<unsigned int>(device_id) >= device_arch_cache_size)
    {
        // Device architecture cache is too small.
        return hipErrorUnknown;
    }

    arch = arch_cache[device_id].load(std::memory_order_relaxed);
    if(arch != target_arch::invalid)
    {
        return hipSuccess;
    }

    hipDeviceProp_t  device_props;
    const hipError_t result = hipGetDeviceProperties(&device_props, device_id);
    if(result != hipSuccess)
    {
        return result;
    }

    arch = parse_gcn_arch(device_props.gcnArchName);
    arch_cache[device_id].exchange(arch, std::memory_order_relaxed);
#else
    arch = target_arch::unknown;
#endif

    return hipSuccess;
}

#if USE_DEVICE_DISPATCH
/// @brief Queries a device corresponding to a stream using the HIP API.
/// @param stream The stream in question.
/// @param device_id Out parameter. The result of the device query is written here.
/// @return \ref hipSuccess if the querying was successful, a different error code otherwise.
inline hipError_t get_device_from_stream(const hipStream_t stream, int& device_id)
{
    constexpr hipStream_t default_stream = 0;
    if(stream == default_stream || stream == hipStreamPerThread)
    {
        const hipError_t result = hipGetDevice(&device_id);
        if(result != hipSuccess)
        {
            return result;
        }
        return hipSuccess;
    }
    device_id = hipGetStreamDeviceId(stream);
    if(device_id < 0)
    {
        return hipErrorInvalidHandle;
    }
    return hipSuccess;
}
#endif

/// @brief Queries the device architecture of a device corresponding to a HIP stream.
/// @param stream The ID of the HIP stream.
/// @param arch Out parameter. The detected device architecture is set here.
/// @return \ref hipSuccess if the querying was successful, a different error code otherwise.
inline hipError_t get_device_arch(const hipStream_t stream, target_arch& arch)
{
#if USE_DEVICE_DISPATCH
    int              device_id;
    const hipError_t result = get_device_from_stream(stream, device_id);
    if(result != hipSuccess)
    {
        return result;
    }
    return get_device_arch(device_id, arch);
#else
    (void)stream;
    arch = target_arch::unknown;
    return hipSuccess;
#endif
}

/// @brief Default kernel launch configuration for the random engines.
/// Partial or full template specializations are implemented for the engines
/// to select different parameterizations.
/// Architecture-dependent (dynamic) configurations can be implemented by
/// specializing \ref generator_config_selector.
/// @tparam T The type of the generated random values.
/// @tparam GeneratorType The kind of the random engine.
template<rocrand_rng_type GeneratorType, class T>
struct generator_config_defaults
{
    // The predicate is always false, but it must depend on the type parameters.
    static_assert(sizeof(T) == 0,
                  "generator_config_defaults must be specialized for each generator type.");
};

/// @brief Provides architecture-dependent (dynamic) kernel launch
/// configuration to the random engines.
/// Partial or full template specializations are implemented for the engines
/// to select different parameterizations.
/// @tparam T The datatype of the generated random values.
/// @tparam GeneratorType The kind of the random engine.
template<rocrand_rng_type GeneratorType, class T>
struct generator_config_selector
{
    __host__ __device__ static constexpr unsigned int get_threads(const target_arch /*arch*/)
    {
        return generator_config_defaults<GeneratorType, T>::threads;
    }

    __host__ __device__ static constexpr unsigned int get_blocks(const target_arch /*arch*/)
    {
        return generator_config_defaults<GeneratorType, T>::blocks;
    }
};

/// @brief POD struct to store kernel launch configurations for the random engines.
/// This struct can be used both at runtime on the host and at compile time on the device.
struct generator_config
{
    unsigned int threads;
    unsigned int blocks;
    // When adding a new member variable, consider updating the operator< with that.

    __host__ __device__ constexpr bool operator<(const generator_config& other) const
    {
        // In order to store the configs in a \ref std::map, we must define an ordering.
        return (blocks != other.blocks) ? (blocks < other.blocks) : (threads < other.threads);
    }
};

/// @brief Returns whether the provided ordering allows the architecture-dependent
/// selection of kernel launch parameters.
__host__ __device__ constexpr bool is_ordering_dynamic(const rocrand_ordering ordering)
{
    return ordering == ROCRAND_ORDERING_PSEUDO_DYNAMIC
           || ordering == ROCRAND_ORDERING_QUASI_DEFAULT;
}

/// @brief Returns whether this ordering is applicable to pseudo-random number generators.
__host__ __device__ constexpr bool is_ordering_pseudo(const rocrand_ordering ordering)
{
    return ordering != ROCRAND_ORDERING_QUASI_DEFAULT;
}

/// @brief Returns whether this ordering is applicable to quasi-random number generators.
__host__ __device__ constexpr bool is_ordering_quasi(const rocrand_ordering ordering)
{
    return ordering == ROCRAND_ORDERING_QUASI_DEFAULT;
}

template<typename Function>
auto dynamic_dispatch(rocrand_ordering order, Function&& func)
{
    bool is_dynamic = is_ordering_dynamic(order);
    if(is_dynamic)
    {
        return std::forward<Function>(func)(std::true_type{});
    }
    return std::forward<Function>(func)(std::false_type{});
}

/// @brief Selects the preset kernel launch config for the given random engine and
/// generated value type.
/// @tparam T The datatype of the generated random values.
/// @tparam GeneratorType The kind of the random engine.
/// @param stream The HIP stream on which the random generation is executed.
/// @param ordering The ordering of the random engine.
/// @param config Out parameter. The selected launch config is written here.
/// @return \ref hipSuccess if the querying was successful, a different error code otherwise.
template<rocrand_rng_type GeneratorType, class T>
hipError_t get_generator_config(const hipStream_t      stream,
                                const rocrand_ordering ordering,
                                generator_config&      config)
{
    if(is_ordering_dynamic(ordering))
    {
        target_arch      current_arch;
        const hipError_t error = get_device_arch(stream, current_arch);
        if(error != hipSuccess)
        {
            return error;
        }
        config.threads = generator_config_selector<GeneratorType, T>::get_threads(current_arch);
        config.blocks  = generator_config_selector<GeneratorType, T>::get_blocks(current_arch);
    }
    else
    {
        config.threads = generator_config_defaults<GeneratorType, T>::threads;
        config.blocks  = generator_config_defaults<GeneratorType, T>::blocks;
    }
    return hipSuccess;
}

/// @brief Selects the preset kernel launch config for the given random engine and
/// generated value type.
/// @tparam T The datatype of the generated random values.
/// @tparam GeneratorType The kind of the random engine.
/// @param dynamic_config Whether architecture-specific launch config can be selected or not.
/// @return The selected launch config.
template<rocrand_rng_type GeneratorType, class T>
__host__ __device__ constexpr generator_config get_generator_config_device(bool dynamic_config)
{
    return generator_config{generator_config_selector<GeneratorType, T>::get_threads(
                                dynamic_config ? get_device_arch() : target_arch::unknown),
                            generator_config_selector<GeneratorType, T>::get_blocks(
                                dynamic_config ? get_device_arch() : target_arch::unknown)};
}

/// @brief Loads the appropriate kernel launch config for both host and kernel code.
/// The purpose of this class is to aggregate all configs (host, device dynamic, device static)
/// in a single type, which can be passed to the generators as the \c ConfigProvider template
/// parameter. This class gets the config based on the architecture and generated type \c T,
/// but other \c ConfigProviders (e.g. used in benchmarking) can just return the same value
/// regardless of type/architecture.
/// @tparam GeneratorType The type of the generator (e.g. XORWOW, MT19937) to provide config for.
template<rocrand_rng_type GeneratorType>
struct default_config_provider
{
    /// @brief Returns the appropriate kernel config for the provided generated type in device code.
    /// @tparam T The type of the generated random values.
    /// @param is_dynamic Controls if the returned config belongs to the static or the dynamic ordering.
    /// @return The kernel config struct.
    template<class T>
    __host__ __device__ static constexpr generator_config device_config(const bool is_dynamic)
    {
        return get_generator_config_device<GeneratorType, T>(is_dynamic);
    }

    /// @brief Load the config on the host for a specific architecture and ordering.
    /// @param stream The HIP stream to detect the device architecture from.
    /// @param ordering The currently used ordering.
    /// @param config The result of the config query is returned here.
    /// @return \c hipSuccess if the query was successful, otherwise the error code from the
    /// first failing HIP runtime function invocation.
    template<class T>
    static hipError_t host_config(const hipStream_t      stream,
                                  const rocrand_ordering ordering,
                                  generator_config&      config)
    {
        return get_generator_config<GeneratorType, T>(stream, ordering, config);
    }
};

/// @brief ConfigProvider that always returns a config with the specified \ref Blocks and \ref Threads.
/// This can be used in place of \ref rocrand_impl::host::default_config_provider, which bases the
/// returned configuration on the current architecture.
/// @tparam Threads The number of threads in the kernel block.
/// @tparam Blocks The number of blocks in the kernel grid.
template<unsigned int Threads, unsigned int Blocks>
struct static_config_provider
{
    static constexpr inline generator_config static_config = {Threads, Blocks};

    template<class>
    __host__ __device__ static constexpr generator_config device_config(const bool /*is_dynamic*/)
    {
        return static_config;
    }

    template<class>
    static hipError_t host_config(const hipStream_t /*stream*/,
                                  const rocrand_ordering /*ordering*/,
                                  generator_config& config)
    {
        config = static_config;
        return hipSuccess;
    }
};

template<rocrand_rng_type RngType>
using static_default_config_provider_t
    = static_config_provider<generator_config_defaults<RngType, void>::threads,
                             generator_config_defaults<RngType, void>::blocks>;

/// @brief ConfigProvider that does not specify the grid size. This can be passed directly to a kernel's
/// template argument list, when only the block size (number of threads) is needed in compile time.
/// @tparam Threads The number of threads in the kernel block.
template<unsigned int Threads>
struct static_block_size_config_provider
{
    struct block_size_generator_config
    {
        unsigned int threads;
    };

    static constexpr inline block_size_generator_config static_config = {Threads};

    template<class>
    __host__ __device__ static constexpr block_size_generator_config
        device_config(const bool /*is_dynamic*/)
    {
        return static_config;
    }

    template<class>
    static hipError_t host_config(const hipStream_t /*stream*/,
                                  const rocrand_ordering /*ordering*/,
                                  block_size_generator_config& config)
    {
        config = static_config;
        return hipSuccess;
    }
};

/// @brief Returns the maximum grid size (blocks * threads) of all kernel configurations
/// that are possibly selected on the device corresponding to the provided stream.
/// @param stream Stream object to select the corresponding device (GPU).
/// @param ordering Ordering that determines the potential configurations selected.
/// @param least_common_grid_size Returns the least common multiple of all grid sizes across configurations.
/// The reference may be modified, even if the function doesn't return \c hipSuccess.
/// @return \c hipSuccess if the operation completed successfully, otherwise the error code
/// from the first failing HIP runtime function invocation.
/// @tparam ConfigProvider Provider of the kernel launch configs.
template<class ConfigProvider>
hipError_t get_least_common_grid_size(const hipStream_t      stream,
                                      const rocrand_ordering ordering,
                                      unsigned int&          least_common_grid_size)
{
    least_common_grid_size = 1;

    const auto get_grid_lcm = [&](const auto tag) -> hipError_t
    {
        generator_config config{};
        const hipError_t error
            = ConfigProvider::template host_config<std::decay_t<decltype(tag)>>(stream,
                                                                                ordering,
                                                                                config);
        if(error != hipSuccess)
            return error;
        least_common_grid_size = std::lcm(least_common_grid_size, config.blocks * config.threads);
        return hipSuccess;
    };

    constexpr std::
        tuple<unsigned int, unsigned short, unsigned char, unsigned long long, float, double, half>
            all_generated_types{};

    hipError_t error = hipSuccess;
    cpp_utils::visit_tuple(
        [&](const auto tag)
        {
            if(error == hipSuccess)
            {
                error = get_grid_lcm(tag);
            }
        },
        all_generated_types);

    return error;
}

/// @brief Returns the maximum grid size (blocks * threads) of all kernel configurations
/// that are possibly selected on the device currently compiled to.
/// @param is_dynamic Whether the current kernel uses dynamic ordering or not.
/// @return The least common multiple of all grid sizes across configurations.
/// @tparam ConfigProvider Provider of the kernel launch configs.
template<class ConfigProvider>
__host__ __device__ constexpr unsigned int get_least_common_grid_size(const bool is_dynamic)
{
    generator_config type_configs[6]{};
    type_configs[0] = ConfigProvider::template device_config<unsigned int>(is_dynamic);
    type_configs[1] = ConfigProvider::template device_config<unsigned short>(is_dynamic);
    type_configs[2] = ConfigProvider::template device_config<unsigned char>(is_dynamic);
    type_configs[3] = ConfigProvider::template device_config<half>(is_dynamic);
    type_configs[4] = ConfigProvider::template device_config<float>(is_dynamic);
    type_configs[5] = ConfigProvider::template device_config<double>(is_dynamic);

    unsigned int least_common_grid_size = 1;
    for(const auto config : type_configs)
    {
        const unsigned int grid_size = config.blocks * config.threads;
        least_common_grid_size       = cpp_utils::lcm(least_common_grid_size, grid_size);
    }

    return least_common_grid_size;
}

/// @brief Returns if the total number of threads for the current config
/// matches the least common multiple grid size of all kernel configs on the
/// current device.
/// @param is_dynamic Whether the current kernel uses dynamic ordering or not.
/// @tparam ConfigProvider Provider of the kernel launch configs.
/// @tparam T The generated value type to load the config for.
template<class ConfigProvider, class T>
__host__ __device__ constexpr bool is_single_tile_config(const bool is_dynamic)
{
    const auto         config        = ConfigProvider::template device_config<T>(is_dynamic);
    const unsigned int grid_size     = config.blocks * config.threads;
    const unsigned int lcm_grid_size = get_least_common_grid_size<ConfigProvider>(is_dynamic);

    return grid_size == lcm_grid_size;
}

/// @brief Helper function that is needed because the enclosed expression cannot be
/// passed to \c __launch_bounds__ on NVCC. It queries the kernel config corresponding with
/// the passed \c ConfigProvider and \c T generated value type, and extracts the number of
/// threads per block.
/// @tparam ConfigProvider The \c ConfigProvider which is queried for configs.
/// @tparam T The current generated value type.
/// @param is_dynamic Whether the current kernel uses dynamic ordering or not.
/// @returns The number of threads per block for the current config.
template<class ConfigProvider, class T>
__host__ __device__ constexpr unsigned int get_block_size(const bool is_dynamic)
{
    return ConfigProvider::template device_config<T>(is_dynamic).threads;
}

/// @brief Extracts the `rocrand_rng_type` from a generator template.
/// @tparam GeneratorTemplate The generator template type.
template<template<class> class GeneratorTemplate>
constexpr inline rocrand_rng_type gen_template_type_v = GeneratorTemplate<void>::type();

} // end namespace rocrand_impl::host

#endif // ROCRAND_RNG_CONFIG_TYPES_H_
