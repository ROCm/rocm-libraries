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

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<unsigned int SortBlockSize,
         unsigned int SortItemsPerThread,
         unsigned int MergeImpl1BlockSize,
         unsigned int MergeImplMPPartitionBlockSize,
         unsigned int MergeImplMPBlockSize,
         unsigned int MergeImplMPItemsPerThread,
         unsigned int MinInputSizeMergepath>
struct merge_sort_config_impl
{
    using sort_config                      = kernel_config<SortBlockSize, SortItemsPerThread>;
    using merge_impl1_config               = kernel_config<MergeImpl1BlockSize, 1>;
    using merge_mergepath_partition_config = kernel_config<MergeImplMPPartitionBlockSize, 1>;
    using merge_mergepath_config = kernel_config<MergeImplMPBlockSize, MergeImplMPItemsPerThread>;
    static constexpr unsigned int min_input_size_mergepath = MinInputSizeMergepath;
};

} // namespace detail

/// \brief Configuration of device-level merge primitives.
///
/// \tparam SortBlockSize - block size in the block-sort step
/// \tparam SortItemsPerThread - ItemsPerThread in the block-sort step
/// \tparam MergeImpl1BlockSize - block size in the block merge step using impl1 (used when input_size < MinInputSizeMergepath)
/// \tparam MergeImplMPPartitionBlockSize - block size of the partition kernel in the block merge step using mergepath impl
/// \tparam MergeImplMPBlockSize - block size in the block merge step using mergepath impl
/// \tparam MergeImplMPItemsPerThread - ItemsPerThread in the block merge step using mergepath impl
/// \tparam MinInputSizeMergepath - breakpoint of input-size to use mergepath impl for block merge step
template<unsigned int     MergeImpl1BlockSize           = 512,
         unsigned int     SortBlockSize                 = MergeImpl1BlockSize,
         unsigned int     SortItemsPerThread            = 1,
         unsigned int     MergeImplMPPartitionBlockSize = 128,
         unsigned int     MergeImplMPBlockSize          = std::min(SortBlockSize, 128u),
         unsigned int     MergeImplMPItemsPerThread
         = SortBlockSize* SortItemsPerThread / MergeImplMPBlockSize,
         unsigned int     MinInputSizeMergepath = 200000>
using merge_sort_config = detail::merge_sort_config_impl<SortBlockSize,
                                                         SortItemsPerThread,
                                                         MergeImpl1BlockSize,
                                                         MergeImplMPPartitionBlockSize,
                                                         MergeImplMPBlockSize,
                                                         MergeImplMPItemsPerThread,
                                                         MinInputSizeMergepath>;

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
         bool         ForceSingleKernelConfig = false>
struct radix_sort_config
{
    /// \brief Number of bits in long iterations.
    static constexpr unsigned int long_radix_bits = LongRadixBits;
    /// \brief Number of bits in short iterations.
    static constexpr unsigned int short_radix_bits = ShortRadixBits;
    /// \brief Limit number of blocks to use merge kernel.
    static constexpr unsigned int merge_size_limit_blocks = MergeSizeLimitBlocks;

    /// \brief Configuration of digits scan kernel.
    using scan = ScanConfig;
    /// \brief Configuration of radix sort kernel.
    using sort = SortConfig;
    /// \brief Configuration of radix sort single kernel.
    using sort_single = SortSingleConfig;
    /// \brief Configuration of radix sort merge kernel.
    using sort_merge = SortMergeConfig;
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

} // namespace detail

/// \brief Configuration of device-level reduce primitives.
///
/// \tparam BlockSize - number of threads in a block.
/// \tparam ItemsPerThread - number of items processed by each thread.
/// \tparam BlockReduceMethod - algorithm for block reduce.
/// \tparam SizeLimit - limit on the number of items reduced by a single launch
template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    ::rocprim::block_reduce_algorithm BlockReduceMethod,
    unsigned int SizeLimit = ROCPRIM_GRID_SIZE_LIMIT
>
struct reduce_config
{
    /// \brief Number of threads in a block.
    static constexpr unsigned int block_size = BlockSize;
    /// \brief Number of items processed by each thread.
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    /// \brief Algorithm for block reduce.
    static constexpr block_reduce_algorithm block_reduce_method = BlockReduceMethod;
    /// \brief Limit on the number of items reduced by a single launch
    static constexpr unsigned int size_limit = SizeLimit;
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
