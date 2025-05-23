/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2025, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_

#include "../../../config.hpp"

#include "../tuple.hpp"
#include "../util_type.hpp"

#include <rocprim/block/block_radix_sort.hpp> // IWYU pragma: export
#include <rocprim/functional.hpp> // IWYU pragma: export
#include <rocprim/types/tuple.hpp> // IWYU pragma: export

#include <initializer_list>
#include <type_traits>
#include <utility>

#include "block_scan.hpp"

BEGIN_HIPCUB_NAMESPACE

namespace detail
{

template<class Tuple>
struct tuple_bit_size
{
    // Always false
    static_assert(sizeof(Tuple) == 0, "tuple_bit_size can only be used with hipcub::tuple");
};

template<class Tuple, size_t Index>
constexpr auto tuple_bit_size_impl()
    -> std::enable_if_t<Index == ::hipcub::tuple_size<Tuple>::value, size_t>
{
    return 0;
}

template<class Tuple, size_t Index>
constexpr auto tuple_bit_size_impl()
    -> std::enable_if_t<Index != ::hipcub::tuple_size<Tuple>::value, size_t>
{
    using element_t = std::decay_t<::hipcub::tuple_element_t<Index, Tuple>>;
    return 8 * sizeof(element_t) + tuple_bit_size_impl<Tuple, Index + 1>();
}

template<class... Args>
struct tuple_bit_size<::hipcub::tuple<Args...>>
    : public std::integral_constant<std::size_t, tuple_bit_size_impl<::hipcub::tuple<Args...>, 0>()>
{};

} // namespace detail

template<
    typename KeyT,
    int BLOCK_DIM_X,
    int ITEMS_PER_THREAD,
    typename ValueT = NullType,
    int RADIX_BITS = 4, /* ignored */
    bool MEMOIZE_OUTER_SCAN = true, /* ignored */
    BlockScanAlgorithm INNER_SCAN_ALGORITHM = BLOCK_SCAN_WARP_SCANS, /* ignored */
    hipSharedMemConfig SMEM_CONFIG = hipSharedMemBankSizeFourByte, /* ignored */
    int BLOCK_DIM_Y = 1,
    int BLOCK_DIM_Z = 1,
    int PTX_ARCH = HIPCUB_ARCH /* ignored */
>
class BlockRadixSort
    : private ::rocprim::block_radix_sort<
        KeyT,
        BLOCK_DIM_X,
        ITEMS_PER_THREAD,
        ValueT,
        BLOCK_DIM_Y,
        BLOCK_DIM_Z
      >
{
    static_assert(
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
        "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0"
    );

    using base_type =
        typename ::rocprim::block_radix_sort<
            KeyT,
            BLOCK_DIM_X,
            ITEMS_PER_THREAD,
            ValueT,
            BLOCK_DIM_Y,
            BLOCK_DIM_Z
        >;

    // Reference to temporary storage (usually shared memory)
    typename base_type::storage_type& temp_storage_;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE BlockRadixSort() : temp_storage_(private_storage()) {}

    HIPCUB_DEVICE BlockRadixSort(TempStorage& temp_storage) : temp_storage_(temp_storage) {}

    HIPCUB_DEVICE void
        Sort(KeyT (&keys)[ITEMS_PER_THREAD], int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
    {
        base_type::sort(keys, temp_storage_, begin_bit, end_bit);
    }

    template<class Decomposer>
    HIPCUB_DEVICE std::enable_if_t<!std::is_convertible<Decomposer, int>::value>
        Sort(KeyT (&keys)[ITEMS_PER_THREAD], Decomposer decomposer, int begin_bit, int end_bit)
    {
        base_type::sort(keys, temp_storage_, begin_bit, end_bit, decomposer);
    }

    template<class Decomposer>
    HIPCUB_DEVICE std::enable_if_t<!std::is_convertible<Decomposer, int>::value>
                  Sort(KeyT (&keys)[ITEMS_PER_THREAD], Decomposer decomposer)
    {
        static constexpr int end_bit = detail::tuple_bit_size<decltype(decomposer(keys[0]))>::value;
        base_type::sort(keys, temp_storage_, 0, end_bit, decomposer);
    }

    HIPCUB_DEVICE void Sort(KeyT (&keys)[ITEMS_PER_THREAD],
                            ValueT (&values)[ITEMS_PER_THREAD],
                            int begin_bit = 0,
                            int end_bit   = sizeof(KeyT) * 8)
    {
        base_type::sort(keys, values, temp_storage_, begin_bit, end_bit);
    }

    template<class Decomposer>
    HIPCUB_DEVICE std::enable_if_t<!std::is_convertible<Decomposer, int>::value>
                  Sort(KeyT (&keys)[ITEMS_PER_THREAD],
                       ValueT (&values)[ITEMS_PER_THREAD],
                       Decomposer decomposer,
                       int        begin_bit,
                       int        end_bit)
    {
        base_type::sort(keys, values, temp_storage_, begin_bit, end_bit, decomposer);
    }

    template<class Decomposer>
    HIPCUB_DEVICE std::enable_if_t<!std::is_convertible<Decomposer, int>::value> Sort(
        KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&values)[ITEMS_PER_THREAD], Decomposer decomposer)
    {
        static constexpr int end_bit = detail::tuple_bit_size<decltype(decomposer(keys[0]))>::value;
        base_type::sort(keys, values, temp_storage_, 0, end_bit, decomposer);
    }

    HIPCUB_DEVICE void SortDescending(KeyT (&keys)[ITEMS_PER_THREAD],
                                      int begin_bit = 0,
                                      int end_bit   = sizeof(KeyT) * 8)
    {
        base_type::sort_desc(keys, temp_storage_, begin_bit, end_bit);
    }

    template<class Decomposer>
    HIPCUB_DEVICE std::enable_if_t<!std::is_convertible<Decomposer, int>::value> SortDescending(
        KeyT (&keys)[ITEMS_PER_THREAD], Decomposer decomposer, int begin_bit, int end_bit)
    {
        base_type::sort_desc(keys, temp_storage_, begin_bit, end_bit, decomposer);
    }

    template<class Decomposer>
    HIPCUB_DEVICE std::enable_if_t<!std::is_convertible<Decomposer, int>::value>
                  SortDescending(KeyT (&keys)[ITEMS_PER_THREAD], Decomposer decomposer)
    {
        static constexpr int end_bit = detail::tuple_bit_size<decltype(decomposer(keys[0]))>::value;
        base_type::sort_desc(keys, temp_storage_, 0, end_bit, decomposer);
    }

    HIPCUB_DEVICE void SortDescending(KeyT (&keys)[ITEMS_PER_THREAD],
                                      ValueT (&values)[ITEMS_PER_THREAD],
                                      int begin_bit = 0,
                                      int end_bit   = sizeof(KeyT) * 8)
    {
        base_type::sort_desc(keys, values, temp_storage_, begin_bit, end_bit);
    }

    template<class Decomposer>
    HIPCUB_DEVICE std::enable_if_t<!std::is_convertible<Decomposer, int>::value>
                  SortDescending(KeyT (&keys)[ITEMS_PER_THREAD],
                                 ValueT (&values)[ITEMS_PER_THREAD],
                                 Decomposer decomposer,
                                 int        begin_bit,
                                 int        end_bit)
    {
        base_type::sort_desc(keys, values, temp_storage_, begin_bit, end_bit, decomposer);
    }

    template<class Decomposer>
    HIPCUB_DEVICE std::enable_if_t<!std::is_convertible<Decomposer, int>::value> SortDescending(
        KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&values)[ITEMS_PER_THREAD], Decomposer decomposer)
    {
        static constexpr int end_bit = detail::tuple_bit_size<decltype(decomposer(keys[0]))>::value;
        base_type::sort_desc(keys, values, temp_storage_, 0, end_bit, decomposer);
    }

    HIPCUB_DEVICE void SortBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                                            int begin_bit = 0,
                                            int end_bit   = sizeof(KeyT) * 8)
    {
        base_type::sort_to_striped(keys, temp_storage_, begin_bit, end_bit);
    }

    template<class Decomposer>
    HIPCUB_DEVICE std::enable_if_t<!std::is_convertible<Decomposer, int>::value>
                  SortBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                                       Decomposer decomposer,
                                       int        begin_bit,
                                       int        end_bit)
    {
        base_type::sort_to_striped(keys, temp_storage_, begin_bit, end_bit, decomposer);
    }

    template<class Decomposer>
    HIPCUB_DEVICE std::enable_if_t<!std::is_convertible<Decomposer, int>::value>
                  SortBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD], Decomposer decomposer)
    {
        static constexpr int end_bit = detail::tuple_bit_size<decltype(decomposer(keys[0]))>::value;
        base_type::sort_to_striped(keys, temp_storage_, 0, end_bit, decomposer);
    }

    HIPCUB_DEVICE void SortBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                                            ValueT (&values)[ITEMS_PER_THREAD],
                                            int begin_bit = 0,
                                            int end_bit   = sizeof(KeyT) * 8)
    {
        base_type::sort_to_striped(keys, values, temp_storage_, begin_bit, end_bit);
    }

    template<class Decomposer>
    HIPCUB_DEVICE std::enable_if_t<!std::is_convertible<Decomposer, int>::value>
                  SortBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                                       ValueT (&values)[ITEMS_PER_THREAD],
                                       Decomposer decomposer,
                                       int        begin_bit,
                                       int        end_bit)
    {
        base_type::sort_to_striped(keys, values, temp_storage_, begin_bit, end_bit, decomposer);
    }

    template<class Decomposer>
    HIPCUB_DEVICE std::enable_if_t<!std::is_convertible<Decomposer, int>::value>
                  SortBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                                       ValueT (&values)[ITEMS_PER_THREAD],
                                       Decomposer decomposer)
    {
        static constexpr int end_bit = detail::tuple_bit_size<decltype(decomposer(keys[0]))>::value;
        base_type::sort_to_striped(keys, values, temp_storage_, 0, end_bit, decomposer);
    }

    HIPCUB_DEVICE void SortDescendingBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                                                      int begin_bit = 0,
                                                      int end_bit   = sizeof(KeyT) * 8)
    {
        base_type::sort_desc_to_striped(keys, temp_storage_, begin_bit, end_bit);
    }

    template<class Decomposer>
    HIPCUB_DEVICE std::enable_if_t<!std::is_convertible<Decomposer, int>::value>
                  SortDescendingBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                                                 Decomposer decomposer,
                                                 int        begin_bit,
                                                 int        end_bit)
    {
        base_type::sort_desc_to_striped(keys, temp_storage_, begin_bit, end_bit, decomposer);
    }

    template<class Decomposer>
    HIPCUB_DEVICE std::enable_if_t<!std::is_convertible<Decomposer, int>::value>
        SortDescendingBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD], Decomposer decomposer)
    {
        static constexpr int end_bit = detail::tuple_bit_size<decltype(decomposer(keys[0]))>::value;
        base_type::sort_desc_to_striped(keys, temp_storage_, 0, end_bit, decomposer);
    }

    HIPCUB_DEVICE void SortDescendingBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                                                      ValueT (&values)[ITEMS_PER_THREAD],
                                                      int begin_bit = 0,
                                                      int end_bit   = sizeof(KeyT) * 8)
    {
        base_type::sort_desc_to_striped(keys, values, temp_storage_, begin_bit, end_bit);
    }

    template<class Decomposer>
    HIPCUB_DEVICE std::enable_if_t<!std::is_convertible<Decomposer, int>::value>
                  SortDescendingBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                                                 ValueT (&values)[ITEMS_PER_THREAD],
                                                 Decomposer decomposer,
                                                 int        begin_bit,
                                                 int        end_bit)
    {
        base_type::sort_desc_to_striped(keys,
                                        values,
                                        temp_storage_,
                                        begin_bit,
                                        end_bit,
                                        decomposer);
    }

    template<class Decomposer>
    HIPCUB_DEVICE std::enable_if_t<!std::is_convertible<Decomposer, int>::value>
                  SortDescendingBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                                                 ValueT (&values)[ITEMS_PER_THREAD],
                                                 Decomposer decomposer)
    {
        static constexpr int end_bit = detail::tuple_bit_size<decltype(decomposer(keys[0]))>::value;
        base_type::sort_desc_to_striped(keys, values, temp_storage_, 0, end_bit, decomposer);
    }

private:
    HIPCUB_DEVICE TempStorage& private_storage()
    {
        HIPCUB_SHARED_MEMORY TempStorage private_storage;
        return private_storage;
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_
