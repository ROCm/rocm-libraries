// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

struct merge_sort_block_sort_config_params
{
    kernel_config_params block_sort_config = {512, 4};
    block_sort_algorithm block_sort_method = block_sort_algorithm::merge_sort;
};

// Necessary to construct a parameterized type of `merge_sort_block_sort_config_params`.
// Used in passing to host-side sub-algorithms and GPU kernels so non-default parameters can be available during compile-time.
template<unsigned int BlockSize, unsigned int ItemsPerThread, rocprim::block_sort_algorithm Algo>
struct merge_sort_block_sort_config : rocprim::detail::merge_sort_block_sort_config_params
{
    constexpr merge_sort_block_sort_config()
        : rocprim::detail::merge_sort_block_sort_config_params{
            {BlockSize, ItemsPerThread},
            Algo
    } {};
};

constexpr unsigned int merge_sort_items_per_thread(const unsigned int item_scale)
{
    if(item_scale < 32)
    {
        return 8;
    }
    else if(item_scale < 64)
    {
        return 4;
    }
    else if(item_scale < 128)
    {
        return 2;
    }
    return 1;
}
constexpr unsigned int merge_sort_block_size(const unsigned int item_scale)
{
    if(item_scale < 16)
    {
        return 128;
    }
    else if(item_scale < 32)
    {
        return 64;
    }
    return 32;
}

// Calculate kernel configurations, such that it will not exceed shared memory maximum
template<class Key, class Value>
struct merge_sort_block_sort_config_base
{
    static constexpr unsigned int item_scale
        = ::rocprim::max(sizeof(Key) + sizeof(unsigned int), sizeof(Value));
    // multiply by 2 to ensure block_sort's items_per_block >= block_merge's items_per_block
    static constexpr unsigned int block_size       = merge_sort_block_size(item_scale) * 2;
    static constexpr unsigned int items_per_thread = merge_sort_items_per_thread(item_scale);
    using type                                     = merge_sort_block_sort_config<block_size,
                                              items_per_thread,
                                              block_sort_algorithm::merge_sort>;
};

// Calculate kernel configurations, such that it will not exceed shared memory maximum
template<class Key, class Value>
struct radix_sort_block_sort_config_base
{
    static constexpr unsigned int item_scale = ::rocprim::max(sizeof(Key), sizeof(Value));

    // multiply by 2 to ensure block_sort's items_per_block >= block_merge's items_per_block
    static constexpr unsigned int block_size = merge_sort_block_size(item_scale) * 2;
    static constexpr unsigned int items_per_thread
        = rocprim::min(4u, merge_sort_items_per_thread(item_scale));
    using type = kernel_config<block_size, items_per_thread>;
};

struct merge_sort_block_merge_config_params
{
    kernel_config_params merge_oddeven_config             = {256, 1, (1 << 17) + 70000};
    kernel_config_params merge_mergepath_partition_config = {128, 1};
    kernel_config_params merge_mergepath_config           = {128, 4};
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

struct merge_sort_config_params
{
    merge_sort_block_sort_config_params  block_sort_config;
    merge_sort_block_merge_config_params block_merge_config;
};

} // namespace detail

/// \brief Configuration of device-level merge primitives.
///
/// \tparam SortBlockSize - block size in the block-sort step
/// \tparam SortItemsPerThread - ItemsPerThread in the block-sort step
/// \tparam MergeOddevenBlockSize - block size in the block merge step using oddeven impl (used when input_size < MinInputSizeMergepath)
/// \tparam MergeMergepathPartitionBlockSize - block size of the partition kernel in the block merge step using mergepath impl
/// \tparam MergeMergepathBlockSize - block size in the block merge step using mergepath impl
/// \tparam MergeMergepathItemsPerThread - ItemsPerThread in the block merge step using mergepath impl
/// \tparam MinInputSizeMergepath - breakpoint of input-size to use mergepath impl for block merge step
template<unsigned int MergeOddevenBlockSize            = 512,
         unsigned int SortBlockSize                    = MergeOddevenBlockSize,
         unsigned int SortItemsPerThread               = 1,
         unsigned int MergeMergepathPartitionBlockSize = 128,
         unsigned int MergeMergepathBlockSize          = 128,
         unsigned int MergeMergepathItemsPerThread     = 4,
         unsigned int MinInputSizeMergepath            = (1 << 17) + 70000>
struct merge_sort_config : detail::merge_sort_config_params
{
    /// \remark Here we map the public parameters to our internal structure.
    using block_sort_config
        = detail::merge_sort_block_sort_config<SortBlockSize,
                                               SortItemsPerThread,
                                               block_sort_algorithm::default_algorithm>;
    using block_merge_config = detail::merge_sort_block_merge_config<MergeOddevenBlockSize,
                                                                     1,
                                                                     MinInputSizeMergepath,
                                                                     MergeMergepathBlockSize,
                                                                     MergeMergepathBlockSize,
                                                                     MergeMergepathItemsPerThread>;
    constexpr merge_sort_config()
        : detail::merge_sort_config_params{block_sort_config(), block_merge_config()} {};
};

namespace detail
{

template<class Key, class Value, bool = is_scalar<Key>::value>
struct default_merge_sort_config_base_helper
{
    using type = select_type<
        // clang-format off
            select_type_case<(sizeof(Key) == 1 && sizeof(Value) <= 16), merge_sort_config<512U, 512U, 2U>>,
            select_type_case<(sizeof(Key) == 2 && sizeof(Value) <= 16), merge_sort_config<512U, 256U, 4U>>,
            select_type_case<(sizeof(Key) == 4 && sizeof(Value) <= 16), merge_sort_config<512U, 256U, 4U>>,
            select_type_case<(sizeof(Key) == 8 && sizeof(Value) <= 16), merge_sort_config<256U, 256U, 4U>>,
        // clang-format on
        merge_sort_config<
            limit_block_size<1024U,
                             ::rocprim::max(sizeof(Key) + sizeof(unsigned int), sizeof(Value)),
                             ROCPRIM_WARP_SIZE_64>::value>>;
};

template<class Key, class Value>
struct default_merge_sort_config_base_helper<Key, Value, false>
{
    using type = select_type<
        // clang-format off
            select_type_case<(sizeof(Key) == 8  && sizeof(Value) <= 16), merge_sort_config<512U, 512U, 2U>>,
            select_type_case<(sizeof(Key) == 16 && sizeof(Value) <= 16), merge_sort_config<512U, 512U, 2U>>,
        // clang-format on
        merge_sort_config<
            limit_block_size<512U,
                             ::rocprim::max(sizeof(Key) + sizeof(unsigned int), sizeof(Value)),
                             ROCPRIM_WARP_SIZE_64>::value>>;
};

template<class Key, class Value>
struct default_merge_sort_config_base : default_merge_sort_config_base_helper<Key, Value>::type
{};

struct radix_sort_onesweep_config_params
{
    kernel_config_params histogram = {256, 12};
    kernel_config_params sort      = {256, 12};

    /// \brief The number of bits to sort in one onesweep iteration.
    unsigned int radix_bits_per_place = 4;

    /// \brief The internal block radix rank algorithm to use during the onesweep iteration.
    block_radix_rank_algorithm radix_rank_algorithm = block_radix_rank_algorithm::default_algorithm;
};

template<class HistogramConfig                = kernel_config<256, 12>,
         class SortConfig                     = kernel_config<256, 12>,
         unsigned int               RadixBits = 4,
         block_radix_rank_algorithm RadixRankAlgorithm
         = block_radix_rank_algorithm::default_algorithm>
struct radix_sort_onesweep_config : radix_sort_onesweep_config_params
{
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
};

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

} // namespace detail

/// \brief Configuration of device-level radix sort operation.
///
/// Radix sort is excecuted in a single tile (at size < BlocksPerItem) or
/// few iterations (passes) depending on total number of bits to be sorted
/// (\p begin_bit and \p end_bit), each iteration sorts either \p LongRadixBits or \p ShortRadixBits bits
/// choosen to cover whole bit range in optimal way.
///
/// For example, if \p LongRadixBits is 7, \p ShortRadixBits is 6, \p begin_bit is 0 and \p end_bit is 32
/// there will be 5 iterations: 7 + 7 + 6 + 6 + 6 = 32 bits.
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
struct radix_sort_config
{
    /// \remark Here we map the public parameters to our internal structure.
    /// \brief Limit number of blocks to use merge kernel.
    static constexpr unsigned int merge_size_limit_blocks = MergeSizeLimitBlocks;

    /// \brief Configuration of radix sort single kernel.
    using block_sort_config = SortSingleConfig;
    /// \brief Configuration of merge sort algorithm.
    using merge_sort_config = default_config;
    /// \brief Configration of radix sort onesweep.
    using onesweep = detail::
        radix_sort_onesweep_config<OnesweepHistogramConfig, OnesweepSortConfig, OnesweepRadixBits>;

    /// \brief Force use radix sort single kernel configuration.
    static constexpr bool force_single_kernel_config = ForceSingleKernelConfig;
};

namespace detail
{

template<class Key, class Value>
struct default_radix_sort_config_base_helper
{
    static constexpr unsigned int item_scale = ::rocprim::detail::ceiling_div<unsigned int>(
        ::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using scan = kernel_config<256, 2>;

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            radix_sort_config<4, 4, scan, kernel_config<256, 10>, kernel_config<256, 19>>>,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            radix_sort_config<6, 5, scan, kernel_config<256, 10>, kernel_config<256, 17>>>,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            radix_sort_config<7, 6, scan, kernel_config<256, 15>, kernel_config<256, 15>>>,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            radix_sort_config<7, 6, scan, kernel_config<256, 15>, kernel_config<256, 12>>>,
        radix_sort_config<
            6,
            4,
            scan,
            kernel_config<limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                          ::rocprim::max(1u, 15u / item_scale)>,
            kernel_config<limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                          ::rocprim::max(1u, 10u / item_scale)>,
            kernel_config<limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                          ::rocprim::max(1u, 10u / item_scale)>>>;
};

template<class Key>
struct default_radix_sort_config_base_helper<Key, empty_type>
    : select_type<select_type_case<sizeof(Key) == 1,
                                   radix_sort_config<4,
                                                     3,
                                                     kernel_config<256, 2>,
                                                     kernel_config<256, 10>,
                                                     kernel_config<256, 19>>>,
                  select_type_case<sizeof(Key) == 2,
                                   radix_sort_config<6,
                                                     5,
                                                     kernel_config<256, 2>,
                                                     kernel_config<256, 10>,
                                                     kernel_config<256, 16>>>,
                  select_type_case<sizeof(Key) == 4,
                                   radix_sort_config<7,
                                                     6,
                                                     kernel_config<256, 2>,
                                                     kernel_config<256, 17>,
                                                     kernel_config<256, 15>>>,
                  select_type_case<sizeof(Key) == 8,
                                   radix_sort_config<7,
                                                     6,
                                                     kernel_config<256, 2>,
                                                     kernel_config<256, 15>,
                                                     kernel_config<256, 12>>>>
{};

template<class Value, class Key>
struct default_radix_sort_config_base : default_radix_sort_config_base_helper<Key, Value>::type
{};

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
    constexpr reduce_config()
        : rocprim::detail::reduce_config_params{
            {BlockSize, ItemsPerThread, SizeLimit},
            BlockReduceMethod
    } {};
};

namespace detail
{

template<class Value>
struct default_reduce_config_base_helper
{
    static constexpr unsigned int item_scale
        = ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = reduce_config<limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                               ::rocprim::max(1u, 16u / item_scale),
                               ::rocprim::block_reduce_algorithm::using_warp_reduce>;
};

template<class Value>
struct default_reduce_config_base : default_reduce_config_base_helper<Value>::type
{};

} // namespace detail

/// \brief Configuration of device-level scan primitives.
///
/// \tparam BlockSize - number of threads in a block.
/// \tparam ItemsPerThread - number of items processed by each thread.
/// \tparam UseLookback - whether to use lookback scan or reduce-then-scan algorithm.
/// \tparam BlockLoadMethod - method for loading input values.
/// \tparam StoreLoadMethod - method for storing values.
/// \tparam BlockScanMethod - algorithm for block scan.
/// \tparam SizeLimit - limit on the number of items for a single scan kernel launch.
template<unsigned int                    BlockSize,
         unsigned int                    ItemsPerThread,
         bool                            UseLookback,
         ::rocprim::block_load_method    BlockLoadMethod,
         ::rocprim::block_store_method   BlockStoreMethod,
         ::rocprim::block_scan_algorithm BlockScanMethod,
         unsigned int                    SizeLimit = ROCPRIM_GRID_SIZE_LIMIT>
struct scan_config
{
    /// \brief Number of threads in a block.
    static constexpr unsigned int block_size = BlockSize;
    /// \brief Number of items processed by each thread.
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    /// \brief Whether to use lookback scan or reduce-then-scan algorithm.
    static constexpr bool use_lookback = UseLookback;
    /// \brief Method for loading input values.
    static constexpr ::rocprim::block_load_method block_load_method = BlockLoadMethod;
    /// \brief Method for storing values.
    static constexpr ::rocprim::block_store_method block_store_method = BlockStoreMethod;
    /// \brief Algorithm for block scan.
    static constexpr ::rocprim::block_scan_algorithm block_scan_method = BlockScanMethod;
    /// \brief Limit on the number of items for a single scan kernel launch.
    static constexpr unsigned int size_limit = SizeLimit;
};

namespace detail
{

template<class Value>
struct default_scan_config_base_helper
{
    static constexpr unsigned int item_scale
        = ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = scan_config<limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                             ::rocprim::max(1u, 16u / item_scale),
                             ROCPRIM_DETAIL_USE_LOOKBACK_SCAN,
                             ::rocprim::block_load_method::block_load_transpose,
                             ::rocprim::block_store_method::block_store_transpose,
                             ::rocprim::block_scan_algorithm::using_warp_scan>;
};

template<class Value>
struct default_scan_config_base : default_scan_config_base_helper<Value>::type
{};

} // namespace detail

/// \brief Configuration of device-level scan-by-key operation.
///
/// \tparam BlockSize - number of threads in a block.
/// \tparam ItemsPerThread - number of items processed by each thread.
/// \tparam UseLookback - whether to use lookback scan or reduce-then-scan algorithm.
/// \tparam BlockLoadMethod - method for loading input values.
/// \tparam StoreLoadMethod - method for storing values.
/// \tparam BlockScanMethod - algorithm for block scan.
/// \tparam SizeLimit - limit on the number of items for a single scan kernel launch.
template<unsigned int                    BlockSize,
         unsigned int                    ItemsPerThread,
         bool                            UseLookback,
         ::rocprim::block_load_method    BlockLoadMethod,
         ::rocprim::block_store_method   BlockStoreMethod,
         ::rocprim::block_scan_algorithm BlockScanMethod,
         unsigned int                    SizeLimit = ROCPRIM_GRID_SIZE_LIMIT>
struct scan_by_key_config
{
    /// \brief Number of threads in a block.
    static constexpr unsigned int block_size = BlockSize;
    /// \brief Number of items processed by each thread.
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    /// \brief Whether to use lookback scan or reduce-then-scan algorithm.
    static constexpr bool use_lookback = UseLookback;
    /// \brief Method for loading input values.
    static constexpr ::rocprim::block_load_method block_load_method = BlockLoadMethod;
    /// \brief Method for storing values.
    static constexpr ::rocprim::block_store_method block_store_method = BlockStoreMethod;
    /// \brief Algorithm for block scan.
    static constexpr ::rocprim::block_scan_algorithm block_scan_method = BlockScanMethod;
    /// \brief Limit on the number of items for a single scan kernel launch.
    static constexpr unsigned int size_limit = SizeLimit;
};

namespace detail
{

template<class Key, class Value>
struct default_scan_by_key_config_base_helper
{
    static constexpr unsigned int item_scale = ::rocprim::detail::ceiling_div<unsigned int>(
        sizeof(Key) + sizeof(Value), 2 * sizeof(int));

    using type = scan_config<
        limit_block_size<256U, sizeof(Key) + sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 16u / item_scale),
        ROCPRIM_DETAIL_USE_LOOKBACK_SCAN,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_store_method::block_store_transpose,
        ::rocprim::block_scan_algorithm::using_warp_scan>;
};

template<class Key, class Value>
struct default_scan_by_key_config_base : default_scan_by_key_config_base_helper<Key, Value>::type
{};

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif //ROCPRIM_DEVICE_DETAIL_CONFIG_HELPER_HPP_
