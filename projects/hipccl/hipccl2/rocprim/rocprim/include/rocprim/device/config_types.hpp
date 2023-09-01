// Copyright (c) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_CONFIG_TYPES_HPP_
#define ROCPRIM_DEVICE_CONFIG_TYPES_HPP_

#include <algorithm>
#include <atomic>
#include <limits>
#include <type_traits>

#include <cassert>

#include "../config.hpp"
#include "../intrinsics/thread.hpp"
#include "../detail/various.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Special type used to show that the given device-level operation
/// will be executed with optimal configuration dependent on types of the function's parameters
/// and the target device architecture specified by ROCPRIM_TARGET_ARCH.
/// Algorithms supporting dynamic dispatch will ignore ROCPRIM_TARGET_ARCH and
/// launch using optimal configuration based on the target architecture derived from the stream.
struct default_config
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    // default_config should be able to act as if any other config, members from those configs are provided here
    // merge_sort_config
    using block_sort_config  = default_config;
    using block_merge_config = default_config;
    // radix_sort_config_v2
    using single_sort_config = default_config;
    using merge_sort_config  = default_config;
    using onesweep_config    = default_config;
    // merge_sort_block_sort_config
    using sort_config = default_config;
#endif
};

namespace detail
{

// Non-templated kernel_config for dynamic dispatch
struct kernel_config_params
{
    /// \brief Number of threads in a block.
    unsigned int block_size = 64;
    /// \brief Number of items processed by each thread.
    unsigned int items_per_thread = 1;
    /// \brief Number of items processed by a single kernel launch.
    unsigned int size_limit = ROCPRIM_GRID_SIZE_LIMIT;
};

} // namespace detail

/// \brief Configuration of particular kernels launched by device-level operation
///
/// \tparam BlockSize - number of threads in a block.
/// \tparam ItemsPerThread - number of items processed by each thread.
template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int SizeLimit = ROCPRIM_GRID_SIZE_LIMIT>
struct kernel_config : detail::kernel_config_params
{
    constexpr kernel_config() : detail::kernel_config_params{BlockSize, ItemsPerThread, SizeLimit}
    {}
    /// \brief Number of threads in a block.
    static constexpr unsigned int block_size = BlockSize;
    /// \brief Number of items processed by each thread.
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    /// \brief Number of items processed by a single kernel launch.
    static constexpr unsigned int size_limit = SizeLimit;
};

namespace detail
{

template<
    unsigned int MaxBlockSize,
    unsigned int SharedMemoryPerThread,
    // Most kernels require block sizes not smaller than warp
    unsigned int MinBlockSize, 
    // Can fit in shared memory?
    // Although GPUs have 64KiB, 32KiB is used here as a "soft" limit,
    // because some additional memory may be required in kernels
    bool = (MaxBlockSize * SharedMemoryPerThread <= (1u << 15))
>
struct limit_block_size
{
    // No, then try to decrease block size
    static constexpr unsigned int value =
        limit_block_size<
            detail::next_power_of_two(MaxBlockSize) / 2,
            SharedMemoryPerThread,
            MinBlockSize
        >::value;
};

template<
    unsigned int MaxBlockSize,
    unsigned int SharedMemoryPerThread,
    unsigned int MinBlockSize
>
struct limit_block_size<MaxBlockSize, SharedMemoryPerThread, MinBlockSize, true>
{
    static_assert(MaxBlockSize >= MinBlockSize, "Data is too large, it cannot fit in shared memory");

    static constexpr unsigned int value = MaxBlockSize;
};

template<unsigned int Arch, class T>
struct select_arch_case
{
    static constexpr unsigned int arch = Arch;
    using type = T;
};

template<unsigned int TargetArch, class Case, class... OtherCases>
struct select_arch
    : std::conditional<
        Case::arch == TargetArch,
        extract_type<typename Case::type>,
        select_arch<TargetArch, OtherCases...>
    >::type { };

template<unsigned int TargetArch, class Universal>
struct select_arch<TargetArch, Universal> : extract_type<Universal> { };

template<class Config, class Default>
using default_or_custom_config =
    typename std::conditional<
        std::is_same<Config, default_config>::value,
        Default,
        Config
    >::type;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
enum class target_arch : unsigned int
{
    // This must be zero, to initialize the device -> architecture cache
    invalid = 0,
    gfx803  = 803,
    gfx900  = 900,
    gfx906  = 906,
    gfx908  = 908,
    gfx90a  = 910,
    gfx1030 = 1030,
    gfx1102 = 1102,
    unknown = std::numeric_limits<unsigned int>::max(),
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

/**
 * \brief Checks if the first `n` characters of `rhs` are equal to `lhs`
 * 
 * \param lhs the string to compare against
 * \param rhs the string to compare with
 * \param n length of the substring of `rhs` to chceck
 * \return true if the `n` character long prefix of `rhs` is equal to `lhs`
 */
constexpr bool prefix_equals(const char* lhs, const char* rhs, std::size_t n)
{
    std::size_t i = 0;
    for(; i < n; ++i)
    {
        if(*lhs != *rhs || *lhs == '\0')
        {
            break;
        }
        ++lhs;
        ++rhs;
    }

    // All characters of the prefix of `rhs` was consumed and `lhs` "has run out"
    return i == n && *lhs == '\0';
}

constexpr target_arch get_target_arch_from_name(const char* const arch_name, const std::size_t n)
{
    constexpr const char* target_names[]
        = {"gfx803", "gfx900", "gfx906", "gfx908", "gfx90a", "gfx1030", "gfx1102"};
    constexpr target_arch target_architectures[] = {
        target_arch::gfx803,
        target_arch::gfx900,
        target_arch::gfx906,
        target_arch::gfx908,
        target_arch::gfx90a,
        target_arch::gfx1030,
        target_arch::gfx1102,
    };
    static_assert(sizeof(target_names) / sizeof(target_names[0])
                      == sizeof(target_architectures) / sizeof(target_architectures[0]),
                  "target_names and target_architectures should have the same number of elements");
    constexpr auto num_architectures = sizeof(target_names) / sizeof(target_names[0]);

    for(unsigned int i = 0; i < num_architectures; ++i)
    {
        if(prefix_equals(target_names[i], arch_name, n))
        {
            return target_architectures[i];
        }
    }
    return target_arch::unknown;
}

/**
 * \brief Get the current architecture in device compilation.
 * 
 * This function will always return `unkown` when called from the host, host could should instead
 * call host_target_arch to query the current device from the HIP API.
 * 
 * \return target_arch the architecture currently being compiled for on the device.
 */
constexpr target_arch device_target_arch()
{
#if defined(__amdgcn_processor__)
    // The terminating zero is not counted in the length of the string
    return get_target_arch_from_name(__amdgcn_processor__,
                                     sizeof(__amdgcn_processor__) - sizeof('\0'));
#else
    return target_arch::unknown;
#endif
}

template<class Config>
auto dispatch_target_arch(const target_arch target_arch)
{
    switch(target_arch)
    {
        case target_arch::unknown:
            return Config::template architecture_config<target_arch::unknown>::params;
        case target_arch::gfx803:
            return Config::template architecture_config<target_arch::gfx803>::params;
        case target_arch::gfx900:
            return Config::template architecture_config<target_arch::gfx900>::params;
        case target_arch::gfx906:
            return Config::template architecture_config<target_arch::gfx906>::params;
        case target_arch::gfx908:
            return Config::template architecture_config<target_arch::gfx908>::params;
        case target_arch::gfx90a:
            return Config::template architecture_config<target_arch::gfx90a>::params;
        case target_arch::gfx1030:
            return Config::template architecture_config<target_arch::gfx1030>::params;
        case target_arch::gfx1102:
            return Config::template architecture_config<target_arch::gfx1102>::params;
        case target_arch::invalid:
            assert(false && "Invalid target architecture selected at runtime.");
    }
    return Config::template architecture_config<target_arch::unknown>::params;
}

template<typename Config>
constexpr auto device_params()
{
    return Config::template architecture_config<device_target_arch()>::params;
}

inline target_arch parse_gcn_arch(const char* arch_name)
{
    static constexpr auto length = sizeof(hipDeviceProp_t::gcnArchName);

    const char* arch_end = std::find_if(arch_name,
                                        arch_name + length,
                                        [](const char& val) { return val == ':' || val == '\0'; });

    return get_target_arch_from_name(arch_name, arch_end - arch_name);
}

inline hipError_t get_device_arch(int device_id, target_arch& arch)
{
    static constexpr unsigned int   device_arch_cache_size             = 512;
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

    return hipSuccess;
}

#ifndef _WIN32
inline hipError_t get_device_from_stream(const hipStream_t stream, int& device_id)
{
    static constexpr hipStream_t default_stream = 0;
    if(stream == default_stream || stream == hipStreamPerThread)
    {
        const hipError_t result = hipGetDevice(&device_id);
        if(result != hipSuccess)
        {
            return result;
        }
        return hipSuccess;
    }

#ifdef __HIP_PLATFORM_AMD__
    device_id = hipGetStreamDeviceId(stream);
    if(device_id < 0)
    {
        return hipErrorInvalidHandle;
    }
#else
    #error("Getting the current device from a stream is not implemented for this platform");
#endif
    return hipSuccess;
}
#endif

inline hipError_t host_target_arch(const hipStream_t stream, target_arch& arch)
{
#ifdef _WIN32
    (void)stream;
    arch = target_arch::unknown;
    return hipSuccess;
#else
    int              device_id;
    const hipError_t result = get_device_from_stream(stream, device_id);
    if(result != hipSuccess)
    {
        return result;
    }

    return get_device_arch(device_id, arch);
#endif
}

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_CONFIG_TYPES_HPP_
