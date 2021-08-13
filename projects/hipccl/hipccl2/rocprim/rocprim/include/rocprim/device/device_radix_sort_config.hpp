// Copyright (c) 2018-2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "config_types.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

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
template<
    unsigned int LongRadixBits,
    unsigned int ShortRadixBits,
    class ScanConfig,
    class SortConfig,
    class SortSingleConfig = kernel_config<256, 10>,
    class SortMergeConfig = kernel_config<1024, 1>,
    unsigned int MergeSizeLimitBlocks = 1024U,
    bool ForceSingleKernelConfig = false
>
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
struct radix_sort_config_803
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using scan = kernel_config<256, 2>;

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            radix_sort_config<
                8, 7, scan,
                kernel_config<256, 10>, kernel_config<256, 19>
            >
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            radix_sort_config<
                8, 7, scan,
                kernel_config<256, 10>, kernel_config<256, 17>
            >
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            radix_sort_config<
                7, 6, scan,
                kernel_config<256, 15>, kernel_config<256, 13>
            >
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            radix_sort_config<
                7, 6, scan,
                kernel_config<256, 13>, kernel_config<256, 10>
            >
        >,
        radix_sort_config<
            6, 4, scan,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 15u / item_scale)
            >,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 10u / item_scale)
            >,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 10u / item_scale)
            >
        >
    >;
};

template<class Key>
struct radix_sort_config_803<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, radix_sort_config<8, 7, kernel_config<256, 2>, kernel_config<256, 10>, kernel_config<256, 19> > >,
        select_type_case<sizeof(Key) == 2, radix_sort_config<8, 7, kernel_config<256, 2>, kernel_config<256, 10>, kernel_config<256, 16> > >,
        select_type_case<sizeof(Key) == 4, radix_sort_config<7, 6, kernel_config<256, 2>, kernel_config<256, 9>, kernel_config<256, 15> > >,
        select_type_case<sizeof(Key) == 8, radix_sort_config<7, 6, kernel_config<256, 2>, kernel_config<256, 7>, kernel_config<256, 12> > >
    > { };

template<class Key, class Value>
struct radix_sort_config_900
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using scan = kernel_config<256, 2>;

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            radix_sort_config<4, 4, scan,
            kernel_config<256, 10>, kernel_config<256, 19> >
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            radix_sort_config<6, 5, scan,
            kernel_config<256, 10>, kernel_config<256, 17> >
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            radix_sort_config<7, 6, scan,
            kernel_config<256, 15>, kernel_config<256, 15> >
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            radix_sort_config<7, 6, scan,
            kernel_config<256, 15>, kernel_config<256, 12> >
        >,
        radix_sort_config<
            6, 4, scan,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 15u / item_scale)
            >,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 10u / item_scale)
            >,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 10u / item_scale)
            >
        >
    >;
};

template<class Key>
struct radix_sort_config_900<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, radix_sort_config<4, 3, kernel_config<256, 2>, kernel_config<256, 10>, kernel_config<256, 19> > >,
        select_type_case<sizeof(Key) == 2, radix_sort_config<6, 5, kernel_config<256, 2>, kernel_config<256, 10>, kernel_config<256, 16> > >,
        select_type_case<sizeof(Key) == 4, radix_sort_config<7, 6, kernel_config<256, 2>, kernel_config<256, 17>, kernel_config<256, 15> > >,
        select_type_case<sizeof(Key) == 8, radix_sort_config<7, 6, kernel_config<256, 2>, kernel_config<256, 15>, kernel_config<256, 12> > >
    > { };


template<class Key, class Value>
struct radix_sort_config_908
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using scan = kernel_config<256, 2>;

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            radix_sort_config<4, 4, scan,
            kernel_config<256, 10>, kernel_config<256, 19> >
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            radix_sort_config<6, 5, scan,
            kernel_config<256, 10>, kernel_config<256, 17> >
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            radix_sort_config<7, 6, kernel_config<256, 4>,
            kernel_config<256, 15>, kernel_config<256, 15> >
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            radix_sort_config<7, 6, kernel_config<256, 4>,
            kernel_config<256, 14>, kernel_config<256, 12> >
        >,
        radix_sort_config<
            6, 4, scan,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 15u / item_scale)
            >,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 10u / item_scale)
            >,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 10u / item_scale)
            >
        >
    >;
};

template<class Key>
struct radix_sort_config_908<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, radix_sort_config<4, 3, kernel_config<256, 2>, kernel_config<256, 10>, kernel_config<256, 19> > >,
        select_type_case<sizeof(Key) == 2, radix_sort_config<6, 5, kernel_config<256, 2>, kernel_config<256, 10>, kernel_config<256, 17> > >,
        select_type_case<sizeof(Key) == 4, radix_sort_config<7, 6, kernel_config<256, 4>, kernel_config<256, 17>, kernel_config<256, 15> > >,
        select_type_case<sizeof(Key) == 8, radix_sort_config<7, 6, kernel_config<256, 4>, kernel_config<256, 15>, kernel_config<256, 12> > >
    > { };

// TODO: We need to update these parameters
template<class Key, class Value>
struct radix_sort_config_90a
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using scan = kernel_config<256, 1>;

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            radix_sort_config<4, 4, scan,
            kernel_config<256, 5>, kernel_config<256, 19> >
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            radix_sort_config<6, 5, scan,
            kernel_config<256, 5>, kernel_config<256, 17> >
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            radix_sort_config<7, 6, scan,
            kernel_config<256, 7>, kernel_config<256, 15> >
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            radix_sort_config<7, 6, scan,
            kernel_config<256, 7>, kernel_config<256, 14> >
        >,
        radix_sort_config<
            6, 4, scan,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 15u / item_scale)
            >,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 10u / item_scale)
            >
        >
    >;
};

template<class Key>
struct radix_sort_config_90a<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, radix_sort_config<4, 3, kernel_config<256, 1>, kernel_config<256, 5>, kernel_config<256, 19> > >,
        select_type_case<sizeof(Key) == 2, radix_sort_config<6, 5, kernel_config<256, 1>, kernel_config<256, 5>, kernel_config<256, 17> > >,
        select_type_case<sizeof(Key) == 4, radix_sort_config<7, 6, kernel_config<256, 1>, kernel_config<256, 8>, kernel_config<256, 15> > >,
        select_type_case<sizeof(Key) == 8, radix_sort_config<7, 6, kernel_config<256, 1>, kernel_config<256, 7>, kernel_config<256, 14> > >
    > { };

// TODO: We need to update these parameters
template<class Key, class Value>
struct radix_sort_config_1030
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using scan = kernel_config<256, 2>;

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            radix_sort_config<4, 4, scan,
            kernel_config<256, 10>, kernel_config<256, 19> >
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            radix_sort_config<6, 5, scan,
            kernel_config<256, 10>, kernel_config<256, 17> >
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            radix_sort_config<7, 6, scan,
            kernel_config<256, 15>, kernel_config<256, 15> >
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            radix_sort_config<7, 6, scan,
            kernel_config<256, 15>, kernel_config<256, 14> >
        >,
        radix_sort_config<
            6, 4, scan,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_32>::value,
                ::rocprim::max(1u, 15u / item_scale)
            >,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 10u / item_scale)
            >,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 10u / item_scale)
            >
        >
    >;
};

template<class Key>
struct radix_sort_config_1030<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, radix_sort_config<4, 3, kernel_config<256, 2>, kernel_config<256, 10>, kernel_config<256, 19> > >,
        select_type_case<sizeof(Key) == 2, radix_sort_config<6, 5, kernel_config<256, 2>, kernel_config<256, 10>, kernel_config<256, 19> > >,
        select_type_case<sizeof(Key) == 4, radix_sort_config<7, 6, kernel_config<256, 2>, kernel_config<256, 17>, kernel_config<256, 17> > >,
        select_type_case<sizeof(Key) == 8, radix_sort_config<7, 6, kernel_config<256, 2>, kernel_config<256, 15>, kernel_config<256, 15> > >
    > { };

template<unsigned int TargetArch, class Key, class Value>
struct default_radix_sort_config
    : select_arch<
        TargetArch,
        select_arch_case<803, radix_sort_config_803<Key, Value> >,
        select_arch_case<900, radix_sort_config_900<Key, Value> >,
        select_arch_case<908, radix_sort_config_908<Key, Value> >,
        select_arch_case<ROCPRIM_ARCH_90a, radix_sort_config_90a<Key, Value> >,
        select_arch_case<1030, radix_sort_config_1030<Key, Value> >,
        radix_sort_config_900<Key, Value>
    > { };

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_RADIX_SORT_CONFIG_HPP_
