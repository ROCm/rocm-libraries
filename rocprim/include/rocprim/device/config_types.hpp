// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "../detail/various.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Special type used to show that the given device-level operation
/// will be executed with optimal configuration dependent on types of the function's parameters.
/// With dynamic dispatch algorithms will launch using optimal configuration based on the target
/// architecture derived from the stream.
struct default_config
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    // default_config should be able to act as if any other config, members from those configs are provided here
    // merge_sort_config
    using block_sort_config  = default_config;
    using block_merge_config = default_config;
    // radix_sort_config
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
/// \tparam BlockSize number of threads in a block.
/// \tparam ItemsPerThread number of items processed by each thread.
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

template<unsigned int MaxBlockSize,
         unsigned int SharedMemoryPerThread,
         // Most kernels require block sizes not smaller than warp
         unsigned int MinBlockSize,
         // If kernels require more than MaxBlockSize * SharedMemoryPerThread bytes
         // (eg. to store some kind of block-wide state), that size can be specified here
         unsigned int ExtraSharedMemory = 0,
         // virtual shared memory support
         bool VsmemSupport = false,
         // Can fit in shared memory?
         // Although GPUs have 64KiB, 32KiB is used here as a "soft" limit,
         // because some additional memory may be required in kernels
         bool = (MaxBlockSize * SharedMemoryPerThread + ExtraSharedMemory <= (1u << 15))>
struct limit_block_size
{
    // No, then try to decrease block size
    static constexpr unsigned int value
        = limit_block_size<detail::next_power_of_two(MaxBlockSize) / 2,
                           SharedMemoryPerThread,
                           MinBlockSize,
                           ExtraSharedMemory,
                           VsmemSupport>::value;
};

template<unsigned int MaxBlockSize,
         unsigned int SharedMemoryPerThread,
         unsigned int MinBlockSize,
         unsigned int ExtraSharedMemory,
         bool         VsmemSupport>
struct limit_block_size<MaxBlockSize,
                        SharedMemoryPerThread,
                        MinBlockSize,
                        ExtraSharedMemory,
                        VsmemSupport,
                        true>
{
    static_assert(MaxBlockSize >= MinBlockSize || VsmemSupport,
                  "Data is too large, it cannot fit in shared memory");

    static constexpr unsigned int value = MaxBlockSize;
};

template<unsigned int MaxBlockSize,
         unsigned int SharedMemoryPerThread,
         unsigned int MinBlockSize,
         unsigned int ExtraSharedMemory = 0>
struct fallback_block_size
{

    static constexpr unsigned int fallback_bs = limit_block_size<MaxBlockSize,
                                                                 SharedMemoryPerThread,
                                                                 MinBlockSize,
                                                                 ExtraSharedMemory,
                                                                 true>::value;
    static constexpr unsigned int value = fallback_bs >= MinBlockSize ? fallback_bs : MaxBlockSize;
};

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
    gfx942  = 942,
    gfx1030 = 1030,
    gfx1100 = 1100,
    gfx1102 = 1102,
    gfx1200 = 1200,
    gfx1201 = 1201,
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
    constexpr const char* target_names[]         = {"gfx803",
                                                    "gfx900",
                                                    "gfx906",
                                                    "gfx908",
                                                    "gfx90a",
                                                    "gfx942",
                                                    "gfx1030",
                                                    "gfx1100",
                                                    "gfx1102",
                                                    "gfx1200",
                                                    "gfx1201"};
    constexpr target_arch target_architectures[] = {
        target_arch::gfx803,
        target_arch::gfx900,
        target_arch::gfx906,
        target_arch::gfx908,
        target_arch::gfx90a,
        target_arch::gfx942,
        target_arch::gfx1030,
        target_arch::gfx1100,
        target_arch::gfx1102,
        target_arch::gfx1200,
        target_arch::gfx1201,
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
 * This function will always return `unknown` when called from the host, host could should instead
 * call host_target_arch to query the current device from the HIP API.
 * 
 * \return target_arch the architecture currently being compiled for on the device.
 */
constexpr target_arch device_target_arch()
{
#if defined(__amdgcn_processor__) && !defined(ROCPRIM_EXPERIMENTAL_SPIRV)
    // The terminating zero is not counted in the length of the string
    return get_target_arch_from_name(__amdgcn_processor__,
                                     sizeof(__amdgcn_processor__) - sizeof('\0'));
#else
    return target_arch::unknown;
#endif
}

template<class Config>
auto dispatch_target_arch([[maybe_unused]] const target_arch target_arch)
{
#if !defined(ROCPRIM_EXPERIMENTAL_SPIRV)
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
        case target_arch::gfx942:
            return Config::template architecture_config<target_arch::gfx942>::params;
        case target_arch::gfx1030:
            return Config::template architecture_config<target_arch::gfx1030>::params;
        case target_arch::gfx1100:
            return Config::template architecture_config<target_arch::gfx1100>::params;
        case target_arch::gfx1102:
            return Config::template architecture_config<target_arch::gfx1102>::params;
        case target_arch::gfx1200:
            return Config::template architecture_config<target_arch::gfx1200>::params;
        case target_arch::gfx1201:
            return Config::template architecture_config<target_arch::gfx1201>::params;
        case target_arch::invalid:
            assert(false && "Invalid target architecture selected at runtime.");
    }
#endif
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

inline hipError_t get_device_from_stream(const hipStream_t stream, int& device_id)
{
    static constexpr hipStream_t default_stream = 0;

    // hipStreamLegacy is supported in HIP >= 6.1.0
#if (HIP_VERSION_MAJOR >= 6 && HIP_VERSION_MINOR >= 1)
    const bool is_legacy_stream = (stream == hipStreamLegacy);
#else
    const bool is_legacy_stream = false;
#endif

    if (stream == default_stream || stream == hipStreamPerThread || is_legacy_stream)
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

inline hipError_t host_target_arch(const hipStream_t stream, target_arch& arch)
{
    int              device_id;
    const hipError_t result = get_device_from_stream(stream, device_id);
    if(result != hipSuccess)
    {
        return result;
    }

    return get_device_arch(device_id, arch);
}

} // end namespace detail

/// \brief Returns a number of threads in a hardware warp for the actual device.
/// At host side this constant is available at runtime only.
/// \param device_id the device that should be queried.
/// \param warp_size out parameter for the warp size.
/// \return hipError_t any error that might occur.
///
/// It is constant for a device.
ROCPRIM_HOST
inline hipError_t host_warp_size(const int device_id, unsigned int& warp_size)
{
    warp_size = -1;
    int        warp_size_attribute{};
    hipError_t success
        = hipDeviceGetAttribute(&warp_size_attribute, hipDeviceAttributeWarpSize, device_id);

    if(success == hipSuccess)
    {
        warp_size = static_cast<unsigned int>(warp_size_attribute);
    }
    return success;
};

/// \brief Returns the number of threads in a hardware warp for the device associated with the stream.
/// At host side this constant is available at runtime only.
/// \param stream the stream, whose device should be queried.
/// \param warp_size out parameter for the warp size.
/// \return hipError_t any error that might occur.
///
/// It is constant for a device.
ROCPRIM_HOST inline hipError_t host_warp_size(const hipStream_t stream, unsigned int& warp_size)
{
    int        hip_device;
    hipError_t success = detail::get_device_from_stream(stream, hip_device);
    if(success == hipSuccess)
    {
        return host_warp_size(hip_device, warp_size);
    }
    return success;
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_CONFIG_TYPES_HPP_
