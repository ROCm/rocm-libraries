// Copyright (c) 2018-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#include <type_traits>

#include "../config.hpp"
#include "../intrinsics/thread.hpp"
#include "../detail/various.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Special type used to show that the given device-level operation
/// will be executed with optimal configuration dependent on types of the function's parameters
/// and the target device architecture specified by ROCPRIM_TARGET_ARCH.
struct default_config { };

/// \brief Configuration of particular kernels launched by device-level operation
///
/// \tparam BlockSize - number of threads in a block.
/// \tparam ItemsPerThread - number of items processed by each thread.
template<unsigned int BlockSize, unsigned int ItemsPerThread>
struct kernel_config
{
    /// \brief Number of threads in a block.
    static constexpr unsigned int block_size = BlockSize;
    /// \brief Number of items processed by each thread.
    static constexpr unsigned int items_per_thread = ItemsPerThread;
};

namespace detail
{

template<
    unsigned int MaxBlockSize,
    unsigned int SharedMemoryPerThread,
    // Most kernels require block sizes not smaller than warp
    unsigned int MinBlockSize = ::rocprim::warp_size(),
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

template<class...>
using void_t = void;

template<class T, class = void>
struct extract_type : T { };

template<class T>
struct extract_type<T, void_t<typename T::type> > : extract_type<typename T::type> { };

template<bool Value, class T>
struct select_type_case
{
    static constexpr bool value = Value;
    using type = T;
};

template<class Case, class... OtherCases>
struct select_type
    : std::conditional<
        Case::value,
        extract_type<typename Case::type>,
        select_type<OtherCases...>
    >::type { };

template<class T>
struct select_type<select_type_case<true, T>> : extract_type<T> { };

template<class T>
struct select_type<select_type_case<false, T>>
{
    static_assert(
        sizeof(T) == 0,
        "Cannot select any case. "
        "The last case must have true condition or be a fallback type."
    );
};

template<class Fallback>
struct select_type<Fallback> : extract_type<Fallback> { };

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

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_CONFIG_TYPES_HPP_
