// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_RADIX_SORT_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_RADIX_SORT_CONFIG_HPP_

#include "config_types.hpp"
#include "detail/config/device_radix_sort_block_sort.hpp"
#include "detail/device_config_helper.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Configuration of device-level radix sort operation.
///
/// One of three algorithms is used: single sort (launches only a single block),
/// merge sort, or Onesweep.
///
/// \tparam SortSingleConfig - Configuration for the single kernel subalgorithm.
///         must be \p kernel_config or \p default_config.
/// \tparam MergeSortConfig - Configuration for the merge sort subalgorithm.
///         must be \p merge_sort_config or \p default_config. If \p merge_sort_config, the sorted
///         items per block must be a power of two.
/// \tparam OnesweepConfig - Configuration for the Onesweep subalgorithm.
///         must be \p radix_sort_onesweep_config or \p default_config.
/// \tparam MergeSortLimit - The largest number of items for which the merge sort algorithm will be
///         used. Note that below this limit, a different algorithm may be used.
template<class SingleSortConfig = default_config,
         class MergeSortConfig  = default_config,
         class OnesweepConfig   = default_config,
         size_t MergeSortLimit  = 1024 * 1024>
struct radix_sort_config_v2
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    /// \brief Configuration of radix sort single kernel.
    using single_sort_config = SingleSortConfig;
    /// \brief Configuration of merge sort algorithm.
    using merge_sort_config = MergeSortConfig;
    /// \brief Configuration of radix sort onesweep.
    using onesweep_config = OnesweepConfig;
    /// \brief Maximum number of items to use merge sort algorithm.
    static constexpr size_t merge_sort_limit = MergeSortLimit;
#endif
};

/// \brief Legacy configuration of device-level radix sort operation.
///
/// \deprecated Due to a new implementation the configuration options no longer match the algorithm
/// parameters. Use `radix_sort_config_v2` for the new parameters of the algorithm. Only a best
/// effort mapping is provided for these options, parameters not applicable to the new algorithm
/// are ignored.
///
/// Radix sort is executed in a single tile (at size < BlocksPerItem) or few iterations (passes)
/// depending on total number of bits to be sorted (\p begin_bit and \p end_bit), each iteration
/// sorts either \p LongRadixBits or \p ShortRadixBits bits, chosen to cover whole bit range in
/// optimal way.
///
/// For example, if \p LongRadixBits is 7, \p ShortRadixBits is 6, \p begin_bit is 0 and \p end_bit
/// is 32 there will be 5 iterations: 7 + 7 + 6 + 6 + 6 = 32 bits.
///
/// \tparam LongRadixBits - number of bits in long iterations.
/// \tparam ShortRadixBits - number of bits in short iterations, must be equal to or less than \p LongRadixBits.
/// \tparam ScanConfig - configuration of digits scan kernel. Must be \p kernel_config.
/// \tparam SortConfig - configuration of radix sort kernel. Must be \p kernel_config.
template<unsigned int LongRadixBits,
         unsigned int ShortRadixBits,
         class ScanConfig,
         class SortConfig,
         class SortSingleConfig               = kernel_config<256, 10>,
         class SortMergeConfig                = kernel_config<1024, 1>,
         unsigned int MergeSizeLimitBlocks    = 1024U,
         bool         ForceSingleKernelConfig = false,
         class OnesweepHistogramConfig        = kernel_config<256, 8>,
         class OnesweepSortConfig             = kernel_config<256, 15>,
         unsigned int OnesweepRadixBits       = 4>
struct [[deprecated("use radix_sort_config_v2")]] radix_sort_config
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    /// \brief Configuration of radix sort single kernel.
    using single_sort_config = SortSingleConfig;
    /// \brief Configuration of merge sort algorithm.
    using merge_sort_config = default_config;
    /// \brief Configuration of radix sort onesweep.
    using onesweep_config = radix_sort_onesweep_config<OnesweepHistogramConfig,
                                                       OnesweepSortConfig,
                                                       OnesweepRadixBits>;
    /// \brief Maximum number of items to use merge sort algorithm.
    static constexpr size_t merge_sort_limit = 1024 * MergeSizeLimitBlocks;
#endif
};

namespace detail
{

// sub-algorithm onesweep:
template<typename RadixSortOnesweepConfig, typename, typename>
struct wrapped_radix_sort_onesweep_config
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr radix_sort_onesweep_config_params params = RadixSortOnesweepConfig();
    };
};

template<typename Key, typename Value>
struct wrapped_radix_sort_onesweep_config<default_config, Key, Value>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr radix_sort_onesweep_config_params params
            = default_radix_sort_onesweep_config<static_cast<unsigned int>(Arch), Key, Value>();
    };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<typename RadixSortOnesweepConfig, typename Key, typename Value>
template<target_arch Arch>
constexpr radix_sort_onesweep_config_params
    wrapped_radix_sort_onesweep_config<RadixSortOnesweepConfig, Key, Value>::architecture_config<
        Arch>::params;

template<typename Key, typename Value>
template<target_arch Arch>
constexpr radix_sort_onesweep_config_params
    wrapped_radix_sort_onesweep_config<default_config, Key, Value>::architecture_config<
        Arch>::params;
#endif // DOXYGEN_SHOULD_SKIP_THIS

// Sub-algorithm block_sort:
template<typename RadixSortBlockSortConfig, typename, typename>
struct wrapped_radix_sort_block_sort_config
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr kernel_config_params params = RadixSortBlockSortConfig();
    };
};

template<typename Key, typename Value>
struct wrapped_radix_sort_block_sort_config<default_config, Key, Value>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr kernel_config_params params
            = default_radix_sort_block_sort_config<static_cast<unsigned int>(Arch), Key, Value>();
    };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<typename RadixSortBlockSortConfig, typename Key, typename Value>
template<target_arch Arch>
constexpr kernel_config_params
    wrapped_radix_sort_block_sort_config<RadixSortBlockSortConfig, Key, Value>::architecture_config<
        Arch>::params;

template<typename Key, typename Value>
template<target_arch Arch>
constexpr kernel_config_params wrapped_radix_sort_block_sort_config<default_config, Key, Value>::
    architecture_config<Arch>::params;
#endif // DOXYGEN_SHOULD_SKIP_THIS

} // namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_RADIX_SORT_CONFIG_HPP_
