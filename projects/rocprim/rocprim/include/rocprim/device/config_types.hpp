// Copyright (c) 2018-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <array>
#include <atomic>
#include <limits>
#include <optional>
#include <string_view>
#include <tuple>
#include <type_traits>

#include <cassert>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../intrinsics/arch.hpp"

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
    typename std::conditional<std::is_same<Config, default_config>::value, Default, Config>::type;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// NOTE: When adding a new target_arch also add it to gen_from_target_arch and get_target_arch_from_name
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
    gfx950  = 950,
    gfx1010 = 1010,
    gfx1011 = 1011,
    gfx1012 = 1012,
    gfx1030 = 1030,
    gfx1100 = 1100,
    gfx1101 = 1101,
    gfx1102 = 1102,
    gfx1103 = 1103,
    gfx1150 = 1150,
    gfx1151 = 1151,
    gfx1152 = 1152,
    gfx1153 = 1153,
    gfx1200 = 1200,
    gfx1201 = 1201,
    unknown = std::numeric_limits<unsigned int>::max(),
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

enum class rep
{
    amdgcn,
    spirv,
};

enum class gen
{
    unknown,
    gcn3,
    gcn5,
    cdna1,
    cdna2,
    cdna3,
    cdna4,
    rdna1,
    rdna2,
    rdna3,
    rdna4,
};

enum class gpu
{
    generic,
    v620,
    rx6900,
    rx7900,
    rx9060,
    rx9070,
    mi50,
    mi100,
    mi210,
    mi300x,
    mi300a,
    mi308x,
    mi325x,
    mi350x
};

constexpr gen gen_from_target_arch(target_arch i)
{
    switch(i)
    {
        case target_arch::gfx803: return gen::gcn3;
        case target_arch::gfx900:
        case target_arch::gfx906: return gen::gcn5;
        case target_arch::gfx908: return gen::cdna1;
        case target_arch::gfx90a: return gen::cdna2;
        case target_arch::gfx942: return gen::cdna3;
        case target_arch::gfx950: return gen::cdna4;
        case target_arch::gfx1010:
        case target_arch::gfx1011:
        case target_arch::gfx1012: return gen::rdna1;
        case target_arch::gfx1030: return gen::rdna2;
        case target_arch::gfx1100:
        case target_arch::gfx1101:
        case target_arch::gfx1102:
        case target_arch::gfx1103:
        case target_arch::gfx1150:
        case target_arch::gfx1151:
        case target_arch::gfx1152:
        case target_arch::gfx1153: return gen::rdna3;
        case target_arch::gfx1200:
        case target_arch::gfx1201: return gen::rdna4;
        case target_arch::unknown:
        case target_arch::invalid: return gen::unknown;
    }
}

constexpr std::tuple<std::string_view, gpu> target_gpu_names[] = {
    std::make_tuple<std::string_view, gpu>("MI350X", gpu::mi350x),
    std::make_tuple<std::string_view, gpu>("MI325X", gpu::mi325x),
    std::make_tuple<std::string_view, gpu>("MI308X", gpu::mi308x),
    std::make_tuple<std::string_view, gpu>("MI300A", gpu::mi300a),
    std::make_tuple<std::string_view, gpu>("MI300X", gpu::mi300x),
    std::make_tuple<std::string_view, gpu>("MI210", gpu::mi210),
    std::make_tuple<std::string_view, gpu>("MI100", gpu::mi100),
    std::make_tuple<std::string_view, gpu>("RX 9060", gpu::rx9060),
    std::make_tuple<std::string_view, gpu>("RX 9070", gpu::rx9070),
    std::make_tuple<std::string_view, gpu>("V620", gpu::v620),
    std::make_tuple<std::string_view, gpu>("RX 7900", gpu::rx7900),
    std::make_tuple<std::string_view, gpu>("RX 6900", gpu::rx6900),
};

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

#define ROCPRIM_RETURN_IF_ARCH(ID)       \
    if(prefix_equals(#ID, arch_name, n)) \
    {                                    \
        return target_arch::ID;          \
    }
constexpr target_arch get_target_arch_from_name(const char* const arch_name, const std::size_t n)
{
    ROCPRIM_RETURN_IF_ARCH(gfx803);
    ROCPRIM_RETURN_IF_ARCH(gfx900);
    ROCPRIM_RETURN_IF_ARCH(gfx906);
    ROCPRIM_RETURN_IF_ARCH(gfx908);
    ROCPRIM_RETURN_IF_ARCH(gfx90a);
    ROCPRIM_RETURN_IF_ARCH(gfx942);
    ROCPRIM_RETURN_IF_ARCH(gfx950);
    ROCPRIM_RETURN_IF_ARCH(gfx1010);
    ROCPRIM_RETURN_IF_ARCH(gfx1011);
    ROCPRIM_RETURN_IF_ARCH(gfx1012);
    ROCPRIM_RETURN_IF_ARCH(gfx1030);
    ROCPRIM_RETURN_IF_ARCH(gfx1100);
    ROCPRIM_RETURN_IF_ARCH(gfx1101);
    ROCPRIM_RETURN_IF_ARCH(gfx1102);
    ROCPRIM_RETURN_IF_ARCH(gfx1103);
    ROCPRIM_RETURN_IF_ARCH(gfx1150);
    ROCPRIM_RETURN_IF_ARCH(gfx1151);
    ROCPRIM_RETURN_IF_ARCH(gfx1152);
    ROCPRIM_RETURN_IF_ARCH(gfx1153);
    ROCPRIM_RETURN_IF_ARCH(gfx1200);
    ROCPRIM_RETURN_IF_ARCH(gfx1201);

    return target_arch::unknown;
}
#undef ROCPRIM_RETURN_IF_ARCH

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

    // hipStreamLegacy is supported in HIP >= 6.2.0
#if(HIP_VERSION_MAJOR > 6 || (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR >= 2))
    const bool is_legacy_stream = (stream == hipStreamLegacy);
#else
    const bool is_legacy_stream = false;
#endif

    if(stream == default_stream || stream == hipStreamPerThread || is_legacy_stream)
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

constexpr gpu get_target_gpu_from_name(std::string_view name)
{
    for(const auto& each : target_gpu_names)
    {
        if(name.find(std::get<0>(each)) != name.npos)
        {
            return std::get<1>(each);
        }
    }
    return gpu::generic;
}

inline hipError_t host_target_gpu(const hipStream_t stream, gpu& gpu)
{
    int device_id;
    ROCPRIM_RETURN_ON_ERROR(get_device_from_stream(stream, device_id));

    hipDeviceProp_t prop;
    ROCPRIM_RETURN_ON_ERROR(hipGetDeviceProperties(&prop, device_id));

    gpu = get_target_gpu_from_name(prop.name);

    return hipSuccess;
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
ROCPRIM_HOST
inline hipError_t host_warp_size(const hipStream_t stream, unsigned int& warp_size)
{
    int        hip_device;
    hipError_t success = detail::get_device_from_stream(stream, hip_device);
    if(success == hipSuccess)
    {
        return host_warp_size(hip_device, warp_size);
    }
    return success;
};

namespace detail
{

// TODO: Remove comp_target when adopting C++20 and dropping C++17 support.
// comp_target exists, because target can not be passed as a template variable before C++20.
template<gen g_, target_arch i_ = target_arch::unknown, gpu s_ = gpu::generic, rep r_ = rep::amdgcn>
struct comp_target
{
    static constexpr gen         g = g_;
    static constexpr target_arch i = i_;
    static constexpr gpu         s = s_;
    static constexpr rep         r = r_;
};

// Macro to have a singular place for conversion, limited by C++17.
#define TARGET_TO_COMP_TARGET(CT) comp_target<(CT).g, (CT).i, (CT).s, (CT).r>

struct target
{
    gen          g;
    target_arch  i;
    gpu          s;
    rep          r;
    unsigned int warp_size;

    constexpr target(target_arch  i,
                     gpu          s         = gpu::generic,
                     rep          r         = rep::amdgcn,
                     unsigned int warp_size = arch::wavefront::min_size())
        : g(gen_from_target_arch(i)), i(i), s(s), r(r), warp_size(warp_size){};

    constexpr target(gen          g         = gen::unknown,
                     target_arch  i         = target_arch::unknown,
                     gpu          s         = gpu::generic,
                     rep          r         = rep::amdgcn,
                     unsigned int warp_size = arch::wavefront::min_size())
        : g(g), i(i), s(s), r(r), warp_size(warp_size){};

    // Host runtime constructor
    target(const hipStream_t stream)
    {
        if(host_target_arch(stream, i) != hipSuccess)
        {
            i = target_arch::unknown;
        }

        if(host_target_gpu(stream, s) != hipSuccess)
        {
            s = gpu::generic;
        }

        if(host_warp_size(stream, warp_size) != hipSuccess)
        {
            warp_size = arch::wavefront::min_size();
        }

        g = gen_from_target_arch(i);
        r = rep::amdgcn;
    }

    template<class CompTarget>
    constexpr target(CompTarget)
        : g(CompTarget::g)
        , i(CompTarget::i)
        , s(CompTarget::s)
        , r(CompTarget::r)
        , warp_size(arch::wavefront::min_size())
    {}

    constexpr bool operator==(target other) const
    {
        return g == other.g && i == other.i && s == other.s && r == other.r;
    }
};

template<typename... Ts>
struct comp_targets
{
    template<typename F>
    static constexpr void for_each(F f)
    {
        (f(Ts{}), ...);
    }
};

constexpr arch::wavefront::target get_wavefront_size(const gen gen = gen::unknown)
{
    // If they are the same we already know the wavefront size at compile time.
    if constexpr(arch::wavefront::max_size() == arch::wavefront::min_size())
    {
        return arch::wavefront::max_size() == 32u ? arch::wavefront::target::size32
                                                  : arch::wavefront::target::size64;
    }

    // Otherwise we determine it based on what generation of gpu it was compiled for.
    switch(gen)
    {
        case gen::unknown: return arch::wavefront::target::dynamic;
        case gen::gcn3:
        case gen::gcn5:
        case gen::cdna1:
        case gen::cdna2:
        case gen::cdna3:
        case gen::cdna4: return arch::wavefront::target::size64;
        case gen::rdna1:
        case gen::rdna2:
        case gen::rdna3:
        case gen::rdna4: return arch::wavefront::target::size32;
    }
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

template<typename Kernel>
struct launch_plan
{
    using kernel_type = void (*)(Kernel);
    kernel_type kernel;
    Kernel      device_callback;

    void launch(dim3 grid_size, dim3 block_size, size_t shared_mem, hipStream_t stream) const
    {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel),
                           grid_size,
                           block_size,
                           shared_mem,
                           stream,
                           device_callback);
    }
};

template<class Targets>
constexpr target most_common_config(target target_current)
{
    // Takes unknown as default.
    target ret{};
    Targets::for_each(
        [&](auto t)
        {
            // Skip unknown target for picking.
            if(!(target{} == t))
            {
                constexpr target_arch Arch = t.i;
                constexpr gpu         GPU  = t.s;
                constexpr gen         Gen  = t.g;

                // Update `ret` if the candidate `t` matches more specifically than the current `ret`.
                // Priority order: prefer exact GPU match first; otherwise allow an Arch match (if GPU differs);
                // finally allow a Gen match (if both Arch and GPU differ). This ensures we progressively
                // refine the fallback from generic -> generation -> arch -> exact GPU.
                if((GPU == target_current.s)
                   || (Arch == target_current.i
                       && (target_current.s != ret.s || ret.s == gpu::generic))
                   || (Gen == target_current.g
                       && ((target_current.s != ret.s || ret.s == gpu::generic)
                           && (target_current.i != ret.i || ret.i == target_arch::unknown))))
                {
                    ret = target{t};
                }
            }
        });

    return ret;
}

template<class Selector>
constexpr typename Selector::param_type default_select_config(target t)
{
    using Targets = typename Selector::targets;
    using Params  = typename Selector::param_type;

    const target target_config = most_common_config<Targets>(t);

    Params params{};

    Targets::for_each(
        [&](auto candidate)
        {
            if(target{candidate} == target_config)
            {
                params = Selector{candidate}.params;
            }
        });

    return params;
}

template<class Selector, class Config>
constexpr typename Selector::param_type get_config(Config config, target t)
{
    if constexpr(std::is_same_v<Config, default_config>)
    {
        return default_select_config<Selector>(t);
    }
    else
    {
        return config;
    }
};

template<class Config,
         class Selector,
         class Target,
         arch::wavefront::target TargetWaveSize = get_wavefront_size(Target::g)>
struct target_config
{
    constexpr static target config_target = target{Target{}};
    constexpr static auto   params        = get_config<Selector>(Config{}, config_target);
    constexpr static auto   wavefront     = TargetWaveSize;
};

template<class Config, class Selector, class Target>
struct default_config_static_selector
{
    static constexpr auto block_size
        = target_config<Config, Selector, Target>::params.kernel_config.block_size;
};

// trampoline_kernel that is fully specialized at compile-time for a single GPU architecture.
// By instantiating this template once per supported `target_arch`, the correct tuned config
// will be derived from the template.
template<class Config,
         class Selector,
         class Kernel,
         class Target,
         template<class, class, class>
         class LaunchSelector,
         arch::wavefront::target TargetWaveSize = get_wavefront_size(Target::g)>
ROCPRIM_KERNEL ROCPRIM_LAUNCH_BOUNDS((LaunchSelector<Config, Selector, Target>::block_size)) void
    trampoline_kernel(Kernel kernel)
{
    using TargetConfig = target_config<Config, Selector, Target, TargetWaveSize>;

#if !defined(ROCPRIM_TARGET_SPIRV) || ROCPRIM_TARGET_SPIRV == 0
    using Targets = typename Selector::targets;
    // If the arch does not exist in the Targets it should run the arch for the most_common_config.
    constexpr target device_arch_target = most_common_config<Targets>(target(device_target_arch()));
    // If the build time arch from device_target_arch is a generic arch it is not the same as the runtime arch.
    if constexpr(Target::i == device_arch_target.i)
    {
        kernel(TargetConfig{});
    }
    else if constexpr(ROCPRIM_IS_GENERIC())
    {
        kernel(TargetConfig{});
    }
    else
    {
        __builtin_unreachable();
    }
#else
    kernel(TargetConfig{});
#endif
}

template<class Config,
         class ConfigSelector,
         template<class, class, class> class LaunchSelector = default_config_static_selector,
         class Kernel>
auto make_launch_plan(target target_current, Kernel kernel) -> launch_plan<Kernel>
{
    using Targets = typename ConfigSelector::targets;

    std::optional<void (*)(Kernel)> tuned_kernel = std::nullopt;

    const target target_config = most_common_config<Targets>(target_current);

    // The target config is always in Targets.
    Targets::for_each(
        [&](auto t)
        {
            if(target{t} == target_config)
            {
                using Target = decltype(t);
                if constexpr(get_wavefront_size(Target::g) != arch::wavefront::target::dynamic)
                {

                    tuned_kernel
                        = trampoline_kernel<Config, ConfigSelector, Kernel, Target, LaunchSelector>;
                }
                else
                {
                    if(target_current.warp_size == ROCPRIM_WARP_SIZE_64)
                    {
                        tuned_kernel = trampoline_kernel<Config,
                                                         ConfigSelector,
                                                         Kernel,
                                                         Target,
                                                         LaunchSelector,
                                                         arch::wavefront::target::size64>;
                    }
                    else
                    {
                        tuned_kernel = trampoline_kernel<Config,
                                                         ConfigSelector,
                                                         Kernel,
                                                         Target,
                                                         LaunchSelector,
                                                         arch::wavefront::target::size32>;
                    }
                }
            }
        });

    return {tuned_kernel.value(), kernel};
}

template<class Config,
         class ConfigSelector,
         template<class, class, class> class LaunchSelector = default_config_static_selector,
         class Kernel>
hipError_t execute_launch_plan(
    target t, Kernel kernel, dim3 grid_size, dim3 block_size, size_t shmem, hipStream_t stream)
{
    const auto launch_plan = make_launch_plan<Config, ConfigSelector, LaunchSelector>(t, kernel);
    launch_plan.launch(grid_size, block_size, shmem, stream);
    return hipGetLastError();
}

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_CONFIG_TYPES_HPP_
