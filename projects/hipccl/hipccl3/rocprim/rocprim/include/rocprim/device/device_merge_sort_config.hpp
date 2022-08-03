// Copyright (c) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_MERGE_SORT_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_MERGE_SORT_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"

#include "config_types.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
    template <unsigned int SortBlockSize,
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
        using merge_mergepath_config
            = kernel_config<MergeImplMPBlockSize, MergeImplMPItemsPerThread>;
        static constexpr unsigned int min_input_size_mergepath = MinInputSizeMergepath;
    };
}

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

template<class Key, class Value>
struct merge_sort_config_803
{
    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            merge_sort_config<64U>
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            merge_sort_config<256U>
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            merge_sort_config<512U>
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            merge_sort_config<1024U>
        >,
        merge_sort_config<limit_block_size<1024U, sizeof(Key) + sizeof(Value), ROCPRIM_WARP_SIZE_64>::value>
    >;
};

template<class Value>
struct merge_sort_config_803<rocprim::half, Value>
{
    using type = merge_sort_config<limit_block_size<256U, sizeof(rocprim::half) + sizeof(Value), ROCPRIM_WARP_SIZE_64>::value>;
};

template<class Key>
struct merge_sort_config_803<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, merge_sort_config<64U> >,
        select_type_case<sizeof(Key) == 2, merge_sort_config<256U> >,
        select_type_case<sizeof(Key) == 4, merge_sort_config<256U> >,
        select_type_case<sizeof(Key) >= 8, merge_sort_config<limit_block_size<512U, sizeof(Key), ROCPRIM_WARP_SIZE_64>::value> >
    > { };

template<>
struct merge_sort_config_803<rocprim::half, empty_type>
{
    using type = merge_sort_config<256U>;
};

template<class Key, class Value, bool = is_scalar<Key>::value>
struct merge_sort_config_900
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
struct merge_sort_config_900<Key, Value, false>
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

// TODO: We need to update these parameters
template<class Key, class Value>
struct merge_sort_config_1030
{
    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            merge_sort_config<64U>
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            merge_sort_config<256U>
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            merge_sort_config<512U>
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            merge_sort_config<1024U>
        >,
        merge_sort_config<limit_block_size<1024U, sizeof(Key) + sizeof(Value), ROCPRIM_WARP_SIZE_64>::value>
    >;
};

template<class Value>
struct merge_sort_config_1030<rocprim::half, Value>
{
    using type = merge_sort_config<limit_block_size<256U, sizeof(rocprim::half) + sizeof(Value), ROCPRIM_WARP_SIZE_32>::value>;
};

template<class Key>
struct merge_sort_config_1030<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, merge_sort_config<64U> >,
        select_type_case<sizeof(Key) == 2, merge_sort_config<256U> >,
        select_type_case<sizeof(Key) == 4, merge_sort_config<256U> >,
        select_type_case<sizeof(Key) >= 8, merge_sort_config<limit_block_size<512U, sizeof(Key), ROCPRIM_WARP_SIZE_32>::value> >
    > { };

template<>
struct merge_sort_config_1030<rocprim::half, empty_type>
{
    using type = merge_sort_config<256U>;
};

template<unsigned int TargetArch, class Key, class Value>
struct default_merge_sort_config
    : select_arch<
        TargetArch,
        select_arch_case<803, merge_sort_config_803<Key, Value>>,
        select_arch_case<900, merge_sort_config_900<Key, Value>>,
        select_arch_case<1030, merge_sort_config_1030<Key, Value>>,
        merge_sort_config_900<Key, Value>
    > { };

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_MERGE_SORT_CONFIG_HPP_
