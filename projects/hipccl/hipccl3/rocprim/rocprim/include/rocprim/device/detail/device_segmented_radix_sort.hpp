// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_RADIX_SORT_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_RADIX_SORT_HPP_

#include <type_traits>
#include <iterator>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"
#include "../../types.hpp"

#include "../../block/block_scan.hpp"

#include "device_radix_sort.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int RadixBits,
    bool Descending,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class OffsetIterator
>
ROCPRIM_DEVICE inline
void segmented_sort(KeysInputIterator keys_input,
                    KeysOutputIterator keys_output,
                    ValuesInputIterator values_input,
                    ValuesOutputIterator values_output,
                    OffsetIterator begin_offsets,
                    OffsetIterator end_offsets,
                    unsigned int bit,
                    unsigned int current_radix_bits)
{
    constexpr unsigned int radix_size = 1 << RadixBits;

    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    using count_helper_type = radix_digit_count_helper<BlockSize, ItemsPerThread, RadixBits, Descending>;
    using scan_type = typename ::rocprim::block_scan<unsigned int, radix_size>;
    using sort_and_scatter_helper = radix_sort_and_scatter_helper<
        BlockSize, ItemsPerThread, RadixBits, Descending,
        key_type, value_type
    >;

    ROCPRIM_SHARED_MEMORY union
    {
        typename count_helper_type::storage_type count_helper;
        typename sort_and_scatter_helper::storage_type sort_and_scatter;
    } storage;

    const unsigned int segment_id = ::rocprim::detail::block_id<0>();

    const unsigned int begin_offset = begin_offsets[segment_id];
    const unsigned int end_offset = end_offsets[segment_id];

    // Empty segment
    if(end_offset <= begin_offset)
    {
        return;
    }

    unsigned int digit_count;
    count_helper_type().count_digits(
        keys_input,
        begin_offset, end_offset,
        bit, current_radix_bits,
        storage.count_helper,
        digit_count
    );

    unsigned int digit_start;
    scan_type().exclusive_scan(digit_count, digit_start, 0);
    digit_start += begin_offset;

    ::rocprim::syncthreads();

    sort_and_scatter_helper().sort_and_scatter(
        keys_input, keys_output, values_input, values_output,
        begin_offset, end_offset,
        bit, current_radix_bits,
        digit_start,
        storage.sort_and_scatter
    );
}

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_RADIX_SORT_HPP_
