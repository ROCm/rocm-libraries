// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_CONFIG_HELPER_HPP_
#define ROCPRIM_DEVICE_DETAIL_CONFIG_HELPER_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../block/block_load.hpp"
#include "../../block/block_reduce.hpp"
#include "../../block/block_scan.hpp"
#include "../../block/block_store.hpp"

#include "../config_types.hpp"
#include "rocprim/block/block_radix_rank.hpp"
#include "rocprim/block/block_sort.hpp"

#include "lookback_scan_state.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

/// \brief Default values are provided by \p merge_sort_block_sort_config_base.
struct merge_sort_block_sort_config_params
{
    kernel_config_params block_sort_config = {0, 0};
};

// Necessary to construct a parameterized type of `merge_sort_block_sort_config_params`.
// Used in passing to host-side sub-algorithms and GPU kernels so non-default parameters can be available during compile-time.
template<unsigned int BlockSize, unsigned int ItemsPerThread, rocprim::block_sort_algorithm Algo>
struct merge_sort_block_sort_config : rocprim::detail::merge_sort_block_sort_config_params
{
    using sort_config = kernel_config<BlockSize, ItemsPerThread>;
    constexpr merge_sort_block_sort_config()
        : rocprim::detail::merge_sort_block_sort_config_params{sort_config()} {};
};

constexpr unsigned int merge_sort_items_per_thread(const unsigned int item_scale)
{
    if(item_scale <= 4)
    {
        return 8;
    }
    else if(item_scale <= 64)
    {
        return 4;
    }
    else if(item_scale <= 256)
    {
        return 2;
    }
    return 1;
}

constexpr unsigned int merge_sort_block_size(const unsigned int item_scale)
{
    if(item_scale <= 32)
    {
        return 128;
    }
    else if(item_scale <= 128)
    {
        return 64;
    }
    return 32;
}

// Calculate kernel configurations, such that it will not exceed shared memory maximum
template<class Key, class Value>
struct merge_sort_block_sort_config_base
{
    static constexpr unsigned int item_scale = ::rocprim::max(sizeof(Key), sizeof(Value));
    // multiply by 2 to ensure block_sort's items_per_block >= block_merge's items_per_block
    static constexpr unsigned int block_size       = merge_sort_block_size(item_scale) * 2;
    static constexpr unsigned int items_per_thread = merge_sort_items_per_thread(item_scale);
    using type                                     = merge_sort_block_sort_config<block_size,
                                              items_per_thread,
                                              block_sort_algorithm::stable_merge_sort>;
};

// Calculate kernel configurations, such that it will not exceed shared memory maximum
// No radix_sort_block_sort_params and radix_sort_block_sort_config exist since the only
// configuration member is a kernel_config.
template<class Key, class Value>
struct radix_sort_block_sort_config_base
{
    static constexpr unsigned int item_scale = ::rocprim::max(sizeof(Key), sizeof(Value));

    // multiply by 2 to ensure block_sort's items_per_block >= block_merge's items_per_block
    static constexpr unsigned int block_size = merge_sort_block_size(item_scale) * 2;
    static constexpr unsigned int items_per_thread
        = rocprim::min(4u, merge_sort_items_per_thread(item_scale));
    using type = kernel_config<block_size, items_per_thread>;

    // The items per block should be a power of two, as this is a requirement for the
    // radix sort merge sort.
    static_assert(is_power_of_two(block_size * items_per_thread),
                  "Sorted items per block should be a power of two.");
};

/// \brief Default values are provided by \p merge_sort_block_merge_config_base.
struct merge_sort_block_merge_config_params
{
    kernel_config_params merge_oddeven_config             = {0, 0, 0};
    kernel_config_params merge_mergepath_partition_config = {0, 0};
    kernel_config_params merge_mergepath_config           = {0, 0};
};

// Necessary to construct a parameterized type of `merge_sort_block_merge_config_params`.
// Used in passing to host-side sub-algorithms and GPU kernels so non-default parameters can be available during compile-time.
template<unsigned int OddEvenBlockSize        = 256,
         unsigned int OddEvenItemsPerThread   = 1,
         unsigned int OddEvenSizeLimit        = (1 << 17) + 70000,
         unsigned int PartitionBlockSize      = 128,
         unsigned int MergePathBlockSize      = 128,
         unsigned int MergePathItemsPerThread = 4>
struct merge_sort_block_merge_config : rocprim::detail::merge_sort_block_merge_config_params
{
    constexpr merge_sort_block_merge_config()
        : rocprim::detail::merge_sort_block_merge_config_params{
            {OddEvenBlockSize, OddEvenItemsPerThread, OddEvenSizeLimit},
            {PartitionBlockSize, 1},
            {MergePathBlockSize, MergePathItemsPerThread}
    } {};
};

template<class Key, class Value>
struct merge_sort_block_merge_config_base
{
    static constexpr unsigned int item_scale = ::rocprim::max(sizeof(Key), sizeof(Value));

    static constexpr unsigned int block_size       = merge_sort_block_size(item_scale);
    static constexpr unsigned int items_per_thread = merge_sort_items_per_thread(item_scale);
    using type                                     = merge_sort_block_merge_config<block_size,
                                               1,
                                               (1 << 17) + 70000,
                                               128,
                                               block_size,
                                               items_per_thread>;
};

/// \brief Default values are provided by \p radix_sort_onesweep_config_base.
struct radix_sort_onesweep_config_params
{
    kernel_config_params histogram = {0, 0};
    kernel_config_params sort      = {0, 0};

    /// \brief The number of bits to sort in one onesweep iteration.
    unsigned int radix_bits_per_place = 1;

    /// \brief The internal block radix rank algorithm to use during the onesweep iteration.
    block_radix_rank_algorithm radix_rank_algorithm = block_radix_rank_algorithm::default_algorithm;
};

} // namespace detail

/// \brief Configuration of subalgorithm Onesweep.
///
/// \tparam HistogramConfig - configuration of histogram kernel.
/// \tparam SortConfig - configuration of sort kernel.
/// \tparam RadixBits - number of bits per iteration.
/// \tparam RadixRankAlgorithm - algorithm used for radix rank.
template<class HistogramConfig                = kernel_config<256, 12>,
         class SortConfig                     = kernel_config<256, 12>,
         unsigned int               RadixBits = 4,
         block_radix_rank_algorithm RadixRankAlgorithm
         = block_radix_rank_algorithm::default_algorithm>
struct radix_sort_onesweep_config : detail::radix_sort_onesweep_config_params
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    /// \brief Configration of radix sort onesweep histogram kernel.
    using histogram = HistogramConfig;
    /// \brief Configration of radix sort onesweep sort kernel.
    using sort = SortConfig;

    constexpr radix_sort_onesweep_config()
        : radix_sort_onesweep_config_params{
            {HistogramConfig::block_size, HistogramConfig::items_per_thread},
            {     SortConfig::block_size,      SortConfig::items_per_thread},
            RadixBits,
            RadixRankAlgorithm,
    } {};
#endif
};

namespace detail
{

struct reduce_config_tag
{};

// Calculate kernel configurations, such that it will not exceed shared memory maximum
template<class Key, class Value>
struct radix_sort_onesweep_config_base
{
    static constexpr unsigned int item_scale = ::rocprim::max(sizeof(Key), sizeof(Value));

    static constexpr unsigned int block_size = merge_sort_block_size(item_scale) * 4;
    using type                               = radix_sort_onesweep_config<
        kernel_config<256, 12>,
        kernel_config<block_size, ::rocprim::max(1u, 65000u / block_size / item_scale)>,
        4>;
};

struct reduce_config_params
{
    kernel_config_params   reduce_config;
    block_reduce_algorithm block_reduce_method;
};

} // namespace detail

/// \brief Configuration of device-level reduce primitives.
///
/// \tparam BlockSize - number of threads in a block.
/// \tparam ItemsPerThread - number of items processed by each thread.
/// \tparam BlockReduceMethod - algorithm for block reduce.
/// \tparam SizeLimit - limit on the number of items reduced by a single launch
template<unsigned int                      BlockSize      = 256,
         unsigned int                      ItemsPerThread = 8,
         ::rocprim::block_reduce_algorithm BlockReduceMethod
         = ::rocprim::block_reduce_algorithm::default_algorithm,
         unsigned int SizeLimit = ROCPRIM_GRID_SIZE_LIMIT>
struct reduce_config : rocprim::detail::reduce_config_params
{
    /// \brief Identifies the algorithm associated to the config.
    using tag = detail::reduce_config_tag;
    constexpr reduce_config()
        : rocprim::detail::reduce_config_params{
            {BlockSize, ItemsPerThread, SizeLimit},
            BlockReduceMethod
    } {};
};

namespace detail
{

template<class Value>
struct default_reduce_config_base
{
    static constexpr unsigned int item_scale
        = ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = reduce_config<limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                               ::rocprim::max(1u, 16u / item_scale),
                               ::rocprim::block_reduce_algorithm::using_warp_reduce>;
};

struct scan_config_tag
{};

/// \brief Provides the kernel parameters for exclusive_scan and inclusive_scan based
///        on autotuned configurations or user-provided configurations.
struct scan_config_params
{
    kernel_config_params            kernel_config{};
    ::rocprim::block_load_method    block_load_method{};
    ::rocprim::block_store_method   block_store_method{};
    ::rocprim::block_scan_algorithm block_scan_method{};
};

} // namespace detail

/// \brief Configuration of device-level scan primitives.
///
/// \tparam BlockSize - number of threads in a block.
/// \tparam ItemsPerThread - number of items processed by each thread.
/// \tparam BlockLoadMethod - method for loading input values.
/// \tparam StoreLoadMethod - method for storing values.
/// \tparam BlockScanMethod - algorithm for block scan.
/// \tparam SizeLimit - limit on the number of items for a single scan kernel launch.
template<unsigned int                    BlockSize,
         unsigned int                    ItemsPerThread,
         ::rocprim::block_load_method    BlockLoadMethod,
         ::rocprim::block_store_method   BlockStoreMethod,
         ::rocprim::block_scan_algorithm BlockScanMethod,
         unsigned int                    SizeLimit = ROCPRIM_GRID_SIZE_LIMIT>
struct scan_config : ::rocprim::detail::scan_config_params
{
    /// \brief Identifies the algorithm associated to the config.
    using tag = detail::scan_config_tag;
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    // Requirement dictated by init_lookback_scan_state_kernel.
    static_assert(BlockSize <= ROCPRIM_DEFAULT_MAX_BLOCK_SIZE,
                  "Block size should at most be ROCPRIM_DEFAULT_MAX_BLOCK_SIZE.");

    /// \brief Number of threads in a block.
    static constexpr unsigned int block_size = BlockSize;
    /// \brief Number of items processed by each thread.
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    /// \brief Method for loading input values.
    static constexpr ::rocprim::block_load_method block_load_method = BlockLoadMethod;
    /// \brief Method for storing values.
    static constexpr ::rocprim::block_store_method block_store_method = BlockStoreMethod;
    /// \brief Algorithm for block scan.
    static constexpr ::rocprim::block_scan_algorithm block_scan_method = BlockScanMethod;
    /// \brief Limit on the number of items for a single scan kernel launch.
    static constexpr unsigned int size_limit = SizeLimit;

    constexpr scan_config()
        : ::rocprim::detail::scan_config_params{
            {BlockSize, ItemsPerThread, SizeLimit},
            BlockLoadMethod,
            BlockStoreMethod,
            BlockScanMethod
    } {};
#endif
};

namespace detail
{

struct scan_by_key_config_tag
{};

template<class Value>
struct default_scan_config_base
{
    static constexpr unsigned int item_scale
        = ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = scan_config<limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                             ::rocprim::max(1u, 16u / item_scale),
                             ::rocprim::block_load_method::block_load_transpose,
                             ::rocprim::block_store_method::block_store_transpose,
                             ::rocprim::block_scan_algorithm::using_warp_scan>;
};

/// \brief Provides the kernel parameters for exclusive_scan_by_key and inclusive_scan_by_key based
///        on autotuned configurations or user-provided configurations.
struct scan_by_key_config_params
{
    kernel_config_params            kernel_config;
    ::rocprim::block_load_method    block_load_method;
    ::rocprim::block_store_method   block_store_method;
    ::rocprim::block_scan_algorithm block_scan_method;
};

} // namespace detail

/// \brief Configuration of device-level scan-by-key operation.
///
/// \tparam BlockSize - number of threads in a block.
/// \tparam ItemsPerThread - number of items processed by each thread.
/// \tparam BlockLoadMethod - method for loading input values.
/// \tparam StoreLoadMethod - method for storing values.
/// \tparam BlockScanMethod - algorithm for block scan.
/// \tparam SizeLimit - limit on the number of items for a single scan kernel launch.
template<unsigned int                    BlockSize,
         unsigned int                    ItemsPerThread,
         ::rocprim::block_load_method    BlockLoadMethod,
         ::rocprim::block_store_method   BlockStoreMethod,
         ::rocprim::block_scan_algorithm BlockScanMethod,
         unsigned int                    SizeLimit = ROCPRIM_GRID_SIZE_LIMIT>
struct scan_by_key_config : ::rocprim::detail::scan_by_key_config_params
{
    /// \brief Identifies the algorithm associated to the config.
    using tag = detail::scan_by_key_config_tag;
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    // Requirement dictated by init_lookback_scan_state_kernel.
    static_assert(BlockSize <= ROCPRIM_DEFAULT_MAX_BLOCK_SIZE,
                  "Block size should at most be ROCPRIM_DEFAULT_MAX_BLOCK_SIZE.");

    /// \brief Number of threads in a block.
    static constexpr unsigned int block_size = BlockSize;
    /// \brief Number of items processed by each thread.
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    /// \brief Method for loading input values.
    static constexpr ::rocprim::block_load_method block_load_method = BlockLoadMethod;
    /// \brief Method for storing values.
    static constexpr ::rocprim::block_store_method block_store_method = BlockStoreMethod;
    /// \brief Algorithm for block scan.
    static constexpr ::rocprim::block_scan_algorithm block_scan_method = BlockScanMethod;
    /// \brief Limit on the number of items for a single scan kernel launch.
    static constexpr unsigned int size_limit = SizeLimit;

    constexpr scan_by_key_config()
        : ::rocprim::detail::scan_by_key_config_params{
            {BlockSize, ItemsPerThread, SizeLimit},
            BlockLoadMethod,
            BlockStoreMethod,
            BlockScanMethod
    } {};
#endif
};

namespace detail
{

template<class Key, class Value>
struct default_scan_by_key_config_base
{
    static constexpr unsigned int item_scale = ::rocprim::detail::ceiling_div<unsigned int>(
        sizeof(Key) + sizeof(Value), 2 * sizeof(int));

    using type = scan_by_key_config<
        limit_block_size<256U, sizeof(Key) + sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 16u / item_scale),
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_store_method::block_store_transpose,
        ::rocprim::block_scan_algorithm::using_warp_scan>;
};

struct transform_config_tag
{};

struct transform_config_params
{
    kernel_config_params kernel_config{};
};

} // namespace detail

namespace detail
{
struct segmented_radix_sort_config_tag
{};

struct warp_sort_config_params
{
    /// \brief Allow the partitioning of batches by size for processing via size-optimized kernels.
    bool partitioning_allowed = false;
    /// \brief The number of threads in the logical warp in the small segment processing kernel.
    unsigned int logical_warp_size_small = 0;
    /// \brief The number of items processed by a thread in the small segment processing kernel.
    unsigned int items_per_thread_small = 0;
    /// \brief The number of threads per block in the small segment processing kernel.
    unsigned int block_size_small = 0;
    /// \brief If the number of segments is at least \p partitioning_threshold, then the segments are partitioned into
    /// small and large segment groups, and each group is handled by a different, specialized kernel.
    unsigned int partitioning_threshold = 0;
    /// \brief The number of threads in the logical warp in the medium segment processing kernel.
    unsigned int logical_warp_size_medium = 0;
    /// \brief The number of items processed by a thread in the medium segment processing kernel.
    unsigned int items_per_thread_medium = 0;
    /// \brief The number of threads per block in the medium segment processing kernel.
    unsigned int block_size_medium = 0;
};

struct segmented_radix_sort_config_params
{
    /// \brief Kernel start parameters.
    kernel_config_params kernel_config{};
    /// \brief Number of bits in long iterations.
    unsigned int long_radix_bits = 0;
    /// \brief Number of bits in short iterations.
    unsigned int short_radix_bits = 0;
    /// \brief If set to \p true, warp sort can be used to sort the small segments, even if no partitioning happens.
    bool enable_unpartitioned_warp_sort = true;
    /// \brief Warp sort config params
    warp_sort_config_params warp_sort_config{};
};

} // namespace detail

/// \brief Configuration of the warp sort part of the device segmented radix sort operation.
/// Short enough segments are processed on warp level.
///
/// \tparam LogicalWarpSizeSmall - number of threads in the logical warp of the kernel
/// that processes small segments.
/// \tparam ItemsPerThreadSmall - number of items processed by a thread in the kernel that processes
/// small segments.
/// \tparam BlockSizeSmall - number of threads per block in the kernel which processes the small segments.
/// \tparam PartitioningThreshold - if the number of segments is at least this threshold, the
/// segments are partitioned to a small, a medium and a large segment collection. Both collections
/// are sorted by different kernels. Otherwise, all segments are sorted by a single kernel.
/// \tparam EnableUnpartitionedWarpSort - If set to \p true, warp sort can be used to sort
/// the small segments, even if the total number of segments is below \p PartitioningThreshold.
/// \tparam LogicalWarpSizeMedium - number of threads in the logical warp of the kernel
/// that processes medium segments.
/// \tparam ItemsPerThreadMedium - number of items processed by a thread in the kernel that processes
/// medium segments.
/// \tparam BlockSizeMedium - number of threads per block in the kernel which processes the medium segments.
template<unsigned int LogicalWarpSizeSmall,
         unsigned int ItemsPerThreadSmall,
         unsigned int BlockSizeSmall        = 256,
         unsigned int PartitioningThreshold = 3000,
         unsigned int LogicalWarpSizeMedium = std::max(32u, LogicalWarpSizeSmall),
         unsigned int ItemsPerThreadMedium  = std::max(4u, ItemsPerThreadSmall),
         unsigned int BlockSizeMedium       = 256>
struct WarpSortConfig
{
    static_assert(LogicalWarpSizeSmall * ItemsPerThreadSmall
                      <= LogicalWarpSizeMedium * ItemsPerThreadMedium,
                  "The number of items processed by a small warp cannot be larger than the number "
                  "of items processed by a medium warp");

    /// \brief Allow the partitioning of batches by size for processing via size-optimized kernels.
    static constexpr bool partitioning_allowed = true;
    /// \brief The number of threads in the logical warp in the small segment processing kernel.
    static constexpr unsigned int logical_warp_size_small = LogicalWarpSizeSmall;
    /// \brief The number of items processed by a thread in the small segment processing kernel.
    static constexpr unsigned int items_per_thread_small = ItemsPerThreadSmall;
    /// \brief The number of threads per block in the small segment processing kernel.
    static constexpr unsigned int block_size_small = BlockSizeSmall;
    /// \brief If the number of segments is at least \p partitioning_threshold, then the segments are partitioned into
    /// small and large segment groups, and each group is handled by a different, specialized kernel.
    static constexpr unsigned int partitioning_threshold = PartitioningThreshold;
    /// \brief The number of threads in the logical warp in the medium segment processing kernel.
    static constexpr unsigned int logical_warp_size_medium = LogicalWarpSizeMedium;
    /// \brief The number of items processed by a thread in the medium segment processing kernel.
    static constexpr unsigned int items_per_thread_medium = ItemsPerThreadMedium;
    /// \brief The number of threads per block in the medium segment processing kernel.
    static constexpr unsigned int block_size_medium = BlockSizeMedium;
};

/// \brief Indicates if the warp level sorting is disabled in the
/// device segmented radix sort configuration.
struct DisabledWarpSortConfig
{
    /// \brief Allow the partitioning of batches by size for processing via size-optimized kernels.
    static constexpr bool partitioning_allowed = false;
    /// \brief The number of threads in the logical warp in the small segment processing kernel.
    static constexpr unsigned int logical_warp_size_small = 1;
    /// \brief The number of items processed by a thread in the small segment processing kernel.
    static constexpr unsigned int items_per_thread_small = 1;
    /// \brief The number of threads per block in the small segment processing kernel.
    static constexpr unsigned int block_size_small = 1;
    /// \brief If the number of segments is at least \p partitioning_threshold, then the segments are partitioned into
    /// small and large segment groups, and each group is handled by a different, specialized kernel.
    static constexpr unsigned int partitioning_threshold = 0;
    /// \brief The number of threads in the logical warp in the medium segment processing kernel.
    static constexpr unsigned int logical_warp_size_medium = 1;
    /// \brief The number of items processed by a thread in the medium segment processing kernel.
    static constexpr unsigned int items_per_thread_medium = 1;
    /// \brief The number of threads per block in the medium segment processing kernel.
    static constexpr unsigned int block_size_medium = 1;
};

//// \brief Configuration of device-level segmented radix sort operation.
///
/// Radix sort is excecuted in a few iterations (passes) depending on total number of bits to be sorted
/// (`begin_bit` and `end_bit`), each iteration sorts either `LongRadixBits` or `ShortRadixBits` bits
/// chosen to cover whole bit range in optimal way.
///
/// For example, if `LongRadixBits` is 7, `ShortRadixBits` is 6, `begin_bit` is 0 and `end_bit` is 32
/// there will be 5 iterations: 7 + 7 + 6 + 6 + 6 = 32 bits.
///
/// If a segment's element count is low ( <= warp_sort_config::items_per_thread * warp_sort_config::logical_warp_size ),
/// it is sorted by a special warp-level sorting method.
///
/// \tparam LongRadixBits - number of bits in long iterations.
/// \tparam ShortRadixBits - number of bits in short iterations, must be equal to or less than `LongRadixBits`.
/// \tparam SortConfig - configuration of radix sort kernel. Must be `kernel_config`.
/// \tparam WarpSortConfig - configuration of the warp sort that is used on the short segments.
template<unsigned int LongRadixBits,
         unsigned int ShortRadixBits,
         class SortConfig,
         class WarpSortConfig             = DisabledWarpSortConfig,
         bool EnableUnpartitionedWarpSort = true>
struct segmented_radix_sort_config : public detail::segmented_radix_sort_config_params
{
    /// \brief Identifies the algorithm associated to the config.
    using tag = detail::segmented_radix_sort_config_tag;
#ifndef DOXYGEN_SHOULD_SKIP_THIS

    /// \brief Number of bits in long iterations.
    static constexpr unsigned int long_radix_bits = LongRadixBits;

    /// \brief Number of bits in short iterations.
    static constexpr unsigned int short_radix_bits = ShortRadixBits;

    /// \brief Number of threads in a block.
    static constexpr unsigned int block_size = SortConfig::block_size;

    /// \brief Number of items processed by each thread.
    static constexpr unsigned int items_per_thread = SortConfig::items_per_thread;

    /// \brief If set to \p true, warp sort can be used to sort the small segments, even if no partitioning happens.
    static constexpr bool enable_unpartitioned_warp_sort = EnableUnpartitionedWarpSort;

    /// \brief Limit on the number of items for a single kernel launch.
    static constexpr unsigned int size_limit = SortConfig::size_limit;

    using warp_sort_config = WarpSortConfig;

    constexpr segmented_radix_sort_config()
        : detail::segmented_radix_sort_config_params{
            SortConfig(),
            LongRadixBits,
            ShortRadixBits,
            EnableUnpartitionedWarpSort,
            {warp_sort_config::partitioning_allowed,
              warp_sort_config::logical_warp_size_small,
              warp_sort_config::items_per_thread_small,
              warp_sort_config::block_size_small,
              warp_sort_config::partitioning_threshold,
              warp_sort_config::logical_warp_size_medium,
              warp_sort_config::items_per_thread_medium,
              warp_sort_config::block_size_medium}
    }
    {}
#endif
};

namespace detail
{
/// \brief Default segmented_radix_sort kernel configurations, such that the maximum shared memory is not exceeded.
///
/// \tparam LongRadixBits - Long bits used during the sorting.
/// \tparam ShortRadixBits - Short bits used during the sorting.
/// \tparam ItemsPerThread - Items per thread when type Key has size 1.
template<unsigned int LongRadixBits, unsigned int ShortRadixBits>
struct default_segmented_radix_sort_config_base
{
    static constexpr unsigned int item_scale = ::rocprim::detail::ceiling_div<unsigned int>(
        sizeof(unsigned int) + sizeof(unsigned int), sizeof(int));
    using type = segmented_radix_sort_config<LongRadixBits,
                                             ShortRadixBits,
                                             kernel_config<128, 17u>,
                                             WarpSortConfig<32, 4, 256, 3000, 32, 4, 256>,
                                             true>;
};

} // namespace detail

/// \brief Configuration for the device-level transform operation.
/// \tparam BlockSize Number of threads in a block.
/// \tparam ItemsPerThread Number of items processed by each thread.
/// \tparam SizeLimit Limit on the number of items for a single kernel launch.
template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int SizeLimit = ROCPRIM_GRID_SIZE_LIMIT>
struct transform_config : public detail::transform_config_params
{
    /// \brief Identifies the algorithm associated to the config.
    using tag = detail::transform_config_tag;
#ifndef DOXYGEN_SHOULD_SKIP_THIS

    /// \brief Number of threads in a block.
    static constexpr unsigned int block_size = BlockSize;

    /// \brief Number of items processed by each thread.
    static constexpr unsigned int items_per_thread = ItemsPerThread;

    /// \brief Limit on the number of items for a single kernel launch.
    static constexpr unsigned int size_limit = SizeLimit;

    constexpr transform_config()
        : detail::transform_config_params{
            {BlockSize, ItemsPerThread, SizeLimit}
    }
    {}
#endif
};

namespace detail
{

template<class Value>
struct default_transform_config_base
{
    static constexpr unsigned int item_scale
        = ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = transform_config<256, ::rocprim::max(1u, 16u / item_scale)>;
};

struct binary_search_config_tag : public transform_config_tag
{};
struct upper_bound_config_tag : public transform_config_tag
{};
struct lower_bound_config_tag : public transform_config_tag
{};

} // namespace detail

/// \brief Configuration for the device-level binary search operation.
/// \tparam BlockSize Number of threads in a block.
/// \tparam ItemsPerThread Number of items processed by each thread.
/// \tparam SizeLimit Limit on the number of items for a single kernel launch.
template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int SizeLimit = ROCPRIM_GRID_SIZE_LIMIT>
struct binary_search_config : transform_config<BlockSize, ItemsPerThread, SizeLimit>
{
    /// \brief Identifies the algorithm associated to the config.
    using tag = detail::binary_search_config_tag;
};

/// \brief Configuration for the device-level upper bound operation.
/// \tparam BlockSize Number of threads in a block.
/// \tparam ItemsPerThread Number of items processed by each thread.
/// \tparam SizeLimit Limit on the number of items for a single kernel launch.
template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int SizeLimit = ROCPRIM_GRID_SIZE_LIMIT>
struct upper_bound_config : transform_config<BlockSize, ItemsPerThread, SizeLimit>
{
    /// \brief Identifies the algorithm associated to the config.
    using tag = detail::upper_bound_config_tag;
};

/// \brief Configuration for the device-level lower bound operation.
/// \tparam BlockSize Number of threads in a block.
/// \tparam ItemsPerThread Number of items processed by each thread.
/// \tparam SizeLimit Limit on the number of items for a single kernel launch.
template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int SizeLimit = ROCPRIM_GRID_SIZE_LIMIT>
struct lower_bound_config : transform_config<BlockSize, ItemsPerThread, SizeLimit>
{
    /// \brief Identifies the algorithm associated to the config.
    using tag = detail::lower_bound_config_tag;
};

namespace detail
{

struct histogram_config_tag
{};

template<class Value, class Output>
struct default_binary_search_config_base
    : binary_search_config<
          limit_block_size<256U, sizeof(Value) + sizeof(Output), ROCPRIM_WARP_SIZE_64>::value,
          1>
{};

/// \brief Provides the kernel parameters for histogram_even, multi_histogram_even,
///        histogram_range, and multi_histogram_range based on autotuned configurations or
///        user-provided configurations.
struct histogram_config_params
{
    kernel_config_params histogram_config = {0, 0};

    unsigned int max_grid_size          = 0;
    unsigned int shared_impl_max_bins   = 0;
    unsigned int shared_impl_histograms = 0;
};

} // namespace detail

/// \brief Configuration of device-level histogram operation.
///
/// \tparam HistogramConfig - configuration of histogram kernel. Must be \p kernel_config.
/// \tparam MaxGridSize - maximum number of blocks to launch.
/// \tparam SharedImplMaxBins - maximum total number of bins for all active channels
/// for the shared memory histogram implementation (samples -> shared memory bins -> global memory bins),
/// when exceeded the global memory implementation is used (samples -> global memory bins).
/// \tparam SharedImplHistograms - number of histograms in the shared memory to reduce bank conflicts
/// for atomic operations with narrow sample distributions. Sweetspot for 9xx and 10xx is 3.
template<class HistogramConfig,
         unsigned int MaxGridSize          = 1024,
         unsigned int SharedImplMaxBins    = 2048,
         unsigned int SharedImplHistograms = 3>
struct histogram_config : detail::histogram_config_params
{
    /// \brief Identifies the algorithm associated to the config.
    using tag = detail::histogram_config_tag;
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using histogram = HistogramConfig;

    static constexpr unsigned int max_grid_size          = MaxGridSize;
    static constexpr unsigned int shared_impl_max_bins   = SharedImplMaxBins;
    static constexpr unsigned int shared_impl_histograms = SharedImplHistograms;

    constexpr histogram_config()
        : detail::histogram_config_params{
            HistogramConfig{}, MaxGridSize, SharedImplMaxBins, SharedImplHistograms} {};
#endif
};

namespace detail
{

template<class Sample, unsigned int Channels, unsigned int ActiveChannels>
struct default_histogram_config_base
{
    static constexpr unsigned int item_scale
        = ::rocprim::detail::ceiling_div(sizeof(Sample), sizeof(int));

    using type
        = histogram_config<kernel_config<256, ::rocprim::max(8u / Channels / item_scale, 1u)>>;
};

struct adjacent_difference_config_tag
{};

struct adjacent_difference_config_params
{
    kernel_config_params          adjacent_difference_kernel_config;
    ::rocprim::block_load_method  block_load_method;
    ::rocprim::block_store_method block_store_method;
};
} // namespace detail

/// \brief Configuration of device-level adjacent difference primitives.
///
/// \tparam BlockSize - number of threads in a block.
/// \tparam ItemsPerThread - number of items processed by each thread.
/// \tparam BlockLoadMethod - method for loading input values.
/// \tparam BlockStoreMethod - method for storing values.
/// \tparam SizeLimit - limit on the number of items for a single adjacent difference kernel launch.
template<unsigned int       BlockSize,
         unsigned int       ItemsPerThread,
         block_load_method  BlockLoadMethod  = block_load_method::block_load_transpose,
         block_store_method BlockStoreMethod = block_store_method::block_store_transpose,
         unsigned int       SizeLimit        = ROCPRIM_GRID_SIZE_LIMIT>
struct adjacent_difference_config : public detail::adjacent_difference_config_params
{
    /// \brief Identifies the algorithm associated to the config.
    using tag = detail::adjacent_difference_config_tag;
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    static constexpr ::rocprim::block_load_method  block_load_method  = BlockLoadMethod;
    static constexpr ::rocprim::block_store_method block_store_method = BlockStoreMethod;
    static constexpr unsigned int                  block_size         = BlockSize;
    static constexpr unsigned int                  items_per_thread   = ItemsPerThread;
    static constexpr unsigned int                  size_limit         = SizeLimit;

    constexpr adjacent_difference_config()
        : detail::adjacent_difference_config_params{
            {BlockSize, ItemsPerThread, SizeLimit},
            BlockLoadMethod, BlockStoreMethod
    } {};
#endif
};

namespace detail
{

template<class Value>
struct default_adjacent_difference_config_base
{
    static constexpr unsigned int item_scale
        = ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = adjacent_difference_config<
        limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 16u / item_scale),
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_store_method::block_store_transpose>;
};

} // namespace detail

namespace detail
{

struct partition_config_params
{
    kernel_config_params kernel_config;
    block_load_method    key_block_load_method;
    block_load_method    value_block_load_method;
    block_load_method    flag_block_load_method;
    block_scan_algorithm block_scan_method;
};

} // namespace detail

/// \brief Configuration of device-level partition and select operation.
///
/// \tparam BlockSize - number of threads in a block.
/// \tparam ItemsPerThread - number of items processed by each thread.
/// \tparam KeyBlockLoadMethod - method for loading input keys.
/// \tparam ValueBlockLoadMethod - method for loading input values.
/// \tparam FlagBlockLoadMethod - method for loading flag values.
/// \tparam BlockScanMethod - algorithm for block scan.
/// \tparam SizeLimit - limit on the number of items for a single select kernel launch.
template<unsigned int                 BlockSize,
         unsigned int                 ItemsPerThread,
         ::rocprim::block_load_method KeyBlockLoadMethod
         = ::rocprim::block_load_method::block_load_transpose,
         ::rocprim::block_load_method ValueBlockLoadMethod
         = ::rocprim::block_load_method::block_load_transpose,
         ::rocprim::block_load_method FlagBlockLoadMethod
         = ::rocprim::block_load_method::block_load_transpose,
         ::rocprim::block_scan_algorithm BlockScanMethod
         = ::rocprim::block_scan_algorithm::using_warp_scan,
         unsigned int SizeLimit = ROCPRIM_GRID_SIZE_LIMIT>
struct select_config : public detail::partition_config_params
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    /// \brief Number of threads in a block.
    static constexpr unsigned int block_size = BlockSize;
    /// \brief Number of items processed by each thread.
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    /// \brief Method for loading input keys.
    static constexpr block_load_method key_block_load_method = KeyBlockLoadMethod;
    /// \brief Method for loading input values.
    static constexpr block_load_method value_block_load_method = ValueBlockLoadMethod;
    /// \brief Method for loading flag values.
    static constexpr block_load_method flag_block_load_method = FlagBlockLoadMethod;
    /// \brief Algorithm for block scan.
    static constexpr block_scan_algorithm block_scan_method = BlockScanMethod;
    /// \brief Limit on the number of items for a single select kernel launch.
    static constexpr unsigned int size_limit = SizeLimit;

    constexpr select_config()
        : detail::partition_config_params{
            {BlockSize, ItemsPerThread, SizeLimit},
            KeyBlockLoadMethod,
            ValueBlockLoadMethod,
            FlagBlockLoadMethod,
            BlockScanMethod
    } {};
#endif
};

namespace detail
{

template<typename Key, bool IsThreeway, int ItemScaleBase = 13>
struct default_partition_config_base
{
    static constexpr unsigned int item_scale
        = ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Key), sizeof(int));

    using offset_t = std::conditional_t<IsThreeway, uint2, unsigned int>;

    // Additional shared memory is required by the lookback scan state.
    static constexpr unsigned int shared_mem_offset = sizeof(
        typename offset_lookback_scan_prefix_op<offset_t,
                                                lookback_scan_state<offset_t>>::storage_type);

    using type = select_config<
        limit_block_size<256U, sizeof(Key), ROCPRIM_WARP_SIZE_64, shared_mem_offset>::value,
        ::rocprim::max(1u, ItemScaleBase / item_scale),
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_scan_algorithm::using_warp_scan>;
};

struct reduce_by_key_config_params
{
    kernel_config_params kernel_config;
    unsigned int         tiles_per_block;
    block_load_method    load_keys_method;
    block_load_method    load_values_method;
    block_scan_algorithm scan_algorithm;
};

} // namespace detail

/**
 * \brief Configuration of device-level reduce-by-key operation.
 * 
 * \tparam BlockSize number of threads in a block.
 * \tparam ItemsPerThread number of items processed by each thread per tile. 
 * \tparam LoadKeysMethod method of loading keys
 * \tparam LoadValuesMethod method of loading values
 * \tparam ScanAlgorithm block level scan algorithm to use
 * \tparam TilesPerBlock number of tiles (`BlockSize` * `ItemsPerThread` items) to process per block
 * \tparam SizeLimit limit on the number of items for a single reduce_by_key kernel launch.
 */
template<unsigned int         BlockSize,
         unsigned int         ItemsPerThread,
         block_load_method    LoadKeysMethod   = block_load_method::block_load_transpose,
         block_load_method    LoadValuesMethod = block_load_method::block_load_transpose,
         block_scan_algorithm ScanAlgorithm    = block_scan_algorithm::using_warp_scan,
         unsigned int         TilesPerBlock    = 1,
         unsigned int         SizeLimit        = ROCPRIM_GRID_SIZE_LIMIT>
struct reduce_by_key_config : public detail::reduce_by_key_config_params
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    /// Number of threads in a block.
    static constexpr unsigned int block_size = BlockSize;
    /// Number of tiles (`BlockSize` * `ItemsPerThread` items) to process per block
    static constexpr unsigned int tiles_per_block = TilesPerBlock;
    /// Number of items processed by each thread per tile.
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    /// A rocprim::block_load_method emum value indicating how the keys should be loaded.
    /// Defaults to block_load_method::block_load_transpose
    static constexpr block_load_method load_keys_method = LoadKeysMethod;
    /// A rocprim::block_load_method emum value indicating how the values should be loaded.
    /// Defaults to block_load_method::block_load_transpose
    static constexpr block_load_method load_values_method = LoadValuesMethod;
    /// A rocprim::block_scan_algorithm enum value indicating how the reduction should
    /// be done. Defaults to block_scan_algorithm::using_warp_scan
    static constexpr block_scan_algorithm scan_algorithm = ScanAlgorithm;
    /// Maximum possible number of values. Defaults to ROCPRIM_GRID_SIZE_LIMIT.
    static constexpr unsigned int size_limit = SizeLimit;

    constexpr reduce_by_key_config()
        : detail::reduce_by_key_config_params{
            {BlockSize, ItemsPerThread, SizeLimit},
            TilesPerBlock,
            LoadKeysMethod,
            LoadValuesMethod,
            ScanAlgorithm
    } {};
#endif
};

namespace detail
{

template<class Key, class Value>
struct default_reduce_by_key_config_base
{
    using small_config = reduce_by_key_config<256,
                                              15,
                                              block_load_method::block_load_transpose,
                                              block_load_method::block_load_transpose,
                                              block_scan_algorithm::using_warp_scan,
                                              sizeof(Value) < 16 ? 1 : 2>;

    static constexpr unsigned int size_memory_per_item = std::max(sizeof(Key), sizeof(Value));
    static constexpr unsigned int item_scale
        = static_cast<unsigned int>(ceiling_div(size_memory_per_item, 2 * sizeof(int)));
    static constexpr unsigned int items_per_thread = std::max(1u, 15u / item_scale);

    using large_config
        = reduce_by_key_config<limit_block_size<256U,
                                                items_per_thread * size_memory_per_item,
                                                ROCPRIM_WARP_SIZE_64>::value,
                               items_per_thread,
                               block_load_method::block_load_transpose,
                               block_load_method::block_load_transpose,
                               block_scan_algorithm::using_warp_scan,
                               2>;

    using type = std::
        conditional_t<std::max(sizeof(Key), sizeof(Value)) <= 16, small_config, large_config>;
};

} // namespace detail

namespace detail
{

struct nth_element_config_params
{
    unsigned int               stop_recursion_size;
    unsigned int               number_of_buckets;
    block_radix_rank_algorithm radix_rank_algorithm;
    kernel_config_params       kernel_config;
};

} // namespace detail

/// \brief Configuration of device-level nth_element
///
/// \tparam BlockSize number of threads in a block.
/// \tparam ItemsPerThread number of items processed by each thread.
/// \tparam StopRecursionSize the size from where recursion is stopped to do a block sort
/// \tparam NumberOfBuckets the number of buckets that are used in the algorithm
/// \tparam RadixRankAlgorithm algorithm for radix rank
template<unsigned int               BlockSize,
         unsigned int               ItemsPerThread,
         unsigned int               StopRecursionSize,
         unsigned int               NumberOfBuckets,
         block_radix_rank_algorithm RadixRankAlgorithm>
struct nth_element_config : public detail::nth_element_config_params
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    constexpr nth_element_config()
        : detail::nth_element_config_params{
            StopRecursionSize,
            NumberOfBuckets,
            RadixRankAlgorithm,
            {BlockSize, ItemsPerThread, ROCPRIM_GRID_SIZE_LIMIT}
    }
    {}
#endif
};

namespace detail
{

struct find_first_of_config_params
{
    kernel_config_params kernel_config{};
};

} // namespace detail

/// \brief Configuration of device-level find_first_of
///
/// \tparam BlockSize number of threads in a block.
/// \tparam ItemsPerThread number of items processed by each thread.
template<unsigned int BlockSize, unsigned int ItemsPerThread>
struct find_first_of_config : public detail::find_first_of_config_params
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    constexpr find_first_of_config()
        : detail::find_first_of_config_params{
            {BlockSize, ItemsPerThread, 0}
    }
    {}
#endif
};

namespace detail
{

template<class Value>
struct default_find_first_of_config_base
{
    static constexpr unsigned int item_scale
        = ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = find_first_of_config<256, ::rocprim::max(1u, 16u / item_scale)>;
};

} // namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif //ROCPRIM_DEVICE_DETAIL_CONFIG_HELPER_HPP_
