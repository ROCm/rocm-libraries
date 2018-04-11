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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_SCAN_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_SCAN_HPP_

#include <type_traits>
#include <iterator>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../types.hpp"

#include "../../block/block_load.hpp"
#include "../../block/block_store.hpp"
#include "../../block/block_scan.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Replaces first value of given range with given value
// Important: it does not dereference the first item in given range, so
// it does not matter if it's an invalid pointer.
template<class InputIterator>
class replace_first_iterator
{
private:
    using input_category = typename std::iterator_traits<InputIterator>::iterator_category;
    static_assert(
        std::is_same<input_category, std::random_access_iterator_tag>::value,
        "InputIterator must be a random-access iterator"
    );

public:
    using value_type = typename std::iterator_traits<InputIterator>::value_type;
    using reference = value_type;
    using pointer = const value_type*;
    using difference_type = typename std::iterator_traits<InputIterator>::difference_type;
    using iterator_category = std::random_access_iterator_tag;

    ROCPRIM_HOST_DEVICE inline
    ~replace_first_iterator() = default;

    ROCPRIM_HOST_DEVICE inline
    replace_first_iterator(InputIterator iterator, value_type value, size_t index = 0)
        : iterator_(iterator), value_(value), index_(index)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    replace_first_iterator& operator++()
    {
        iterator_++;
        index_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    replace_first_iterator operator++(int)
    {
        replace_first_iterator old = *this;
        iterator_++;
        index_++;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline
    value_type operator*() const
    {
        if(index_ == 0)
        {
            return value_;
        }
        return *iterator_;
    }

    ROCPRIM_HOST_DEVICE inline
    value_type operator[](difference_type distance) const
    {
        replace_first_iterator i = (*this) + distance;
        return *i;
    }

    ROCPRIM_HOST_DEVICE inline
    replace_first_iterator operator+(difference_type distance) const
    {
        return replace_first_iterator(iterator_ + distance, value_, index_ + distance);
    }

    ROCPRIM_HOST_DEVICE inline
    replace_first_iterator& operator+=(difference_type distance)
    {
        iterator_ += distance;
        index_ += distance;
        return *this;
    }

private:
    InputIterator iterator_;
    value_type value_;
    size_t index_;
};

template<class V, class F, class BinaryFunction>
struct segmented_scan_flag_wrapper_op
{
    #ifdef __cpp_lib_is_invocable
    using result_type = typename std::invoke_result<BinaryFunction, V, V>::type;
    #else
    using result_type = typename std::result_of<BinaryFunction(V, V)>::type;
    #endif

    ROCPRIM_HOST_DEVICE inline
    segmented_scan_flag_wrapper_op() = default;

    ROCPRIM_HOST_DEVICE inline
    segmented_scan_flag_wrapper_op(BinaryFunction scan_op)
        : scan_op_(scan_op)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    ~segmented_scan_flag_wrapper_op() = default;

    ROCPRIM_HOST_DEVICE inline
    rocprim::tuple<result_type, F> operator()(const rocprim::tuple<result_type, F>& t1,
                                              const rocprim::tuple<result_type, F>& t2) const
    {
        if(!rocprim::get<1>(t2))
        {
            return rocprim::make_tuple(
                scan_op_(rocprim::get<0>(t1), rocprim::get<0>(t2)),
                static_cast<F>(rocprim::get<1>(t1) || rocprim::get<1>(t2))
            );
        }
        return t2;
    }

private:
    BinaryFunction scan_op_;
};

template<
    bool Exclusive,
    bool UsePrefix,
    class BlockScanType,
    class T,
    unsigned int ItemsPerThread,
    class BinaryFunction
>
ROCPRIM_DEVICE inline
auto segmented_scan_block_scan(T (&input)[ItemsPerThread],
                               T (&output)[ItemsPerThread],
                               T& prefix,
                               typename BlockScanType::storage_type& storage,
                               BinaryFunction scan_op)
    -> typename std::enable_if<Exclusive>::type
{
    auto prefix_op =
        [&prefix, scan_op](const T& reduction)
        {
            auto saved_prefix = prefix;
            prefix = scan_op(prefix, reduction);
            return saved_prefix;
        };
    BlockScanType()
        .exclusive_scan(
            input, output,
            storage, prefix_op, scan_op
        );
}

template<
    bool Exclusive,
    bool UsePrefix,
    class BlockScanType,
    class T,
    unsigned int ItemsPerThread,
    class BinaryFunction
>
ROCPRIM_DEVICE inline
auto segmented_scan_block_scan(T (&input)[ItemsPerThread],
                               T (&output)[ItemsPerThread],
                               T& prefix,
                               typename BlockScanType::storage_type& storage,
                               BinaryFunction scan_op)
    -> typename std::enable_if<!Exclusive>::type
{
    if(UsePrefix)
    {
        auto prefix_op =
            [&prefix, scan_op](const T& reduction)
            {
                auto saved_prefix = prefix;
                prefix = scan_op(prefix, reduction);
                return saved_prefix;
            };
        BlockScanType()
            .inclusive_scan(
                input, output,
                storage, prefix_op, scan_op
            );
        return;
    }
    BlockScanType()
        .inclusive_scan(
            input, output, prefix,
            storage, scan_op
        );
}

template<
    bool Exclusive,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class ResultType,
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class InitValueType,
    class BinaryFunction
>
ROCPRIM_DEVICE inline
void segmented_scan(InputIterator input,
                    OutputIterator output,
                    OffsetIterator begin_offsets,
                    OffsetIterator end_offsets,
                    InitValueType initial_value,
                    BinaryFunction scan_op)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    using result_type = ResultType;
    using block_load_type = ::rocprim::block_load<
        result_type, BlockSize, ItemsPerThread,
        ::rocprim::block_load_method::block_load_transpose
    >;
    using block_store_type = ::rocprim::block_store<
        result_type, BlockSize, ItemsPerThread,
        ::rocprim::block_store_method::block_store_transpose
    >;
    using block_scan_type = ::rocprim::block_scan<
        result_type, BlockSize,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_load_type::storage_type load;
        typename block_store_type::storage_type store;
        typename block_scan_type::storage_type scan;
    } storage;

    const unsigned int segment_id = ::rocprim::detail::block_id<0>();
    const unsigned int begin_offset = begin_offsets[segment_id];
    const unsigned int end_offset = end_offsets[segment_id];

    // Empty segment
    if(end_offset <= begin_offset)
    {
        return;
    }

    // Input values
    result_type values[ItemsPerThread];
    result_type prefix = initial_value;

    unsigned int block_offset = begin_offset;
    if(block_offset + items_per_block > end_offset)
    {
        // Segment is shorter than items_per_block

        // Load the partial block
        const unsigned int valid_count = end_offset - block_offset;
        block_load_type().load(input + block_offset, values, valid_count, storage.load);
        ::rocprim::syncthreads();
        // Perform scan operation
        segmented_scan_block_scan<Exclusive, false, block_scan_type>(
            values, values, prefix, storage.scan, scan_op
        );
        ::rocprim::syncthreads();
        // Store the partial block
        block_store_type().store(output + block_offset, values, valid_count, storage.store);
    }
    else
    {
        // Long segments

        // Load the first block of input values
        block_load_type().load(input + block_offset, values, storage.load);
        ::rocprim::syncthreads();
        // Perform scan operation
        segmented_scan_block_scan<Exclusive, false, block_scan_type>(
            values, values, prefix, storage.scan, scan_op
        );
        ::rocprim::syncthreads();
        // Store
        block_store_type().store(output + block_offset, values, storage.store);
        ::rocprim::syncthreads();
        block_offset += items_per_block;

        // Load next full blocks and continue scanning
        while(block_offset + items_per_block < end_offset)
        {
            block_load_type().load(input + block_offset, values, storage.load);
            ::rocprim::syncthreads();
            // Perform scan operation
            segmented_scan_block_scan<Exclusive, true, block_scan_type>(
                values, values, prefix, storage.scan, scan_op
            );
            ::rocprim::syncthreads();
            block_store_type().store(output + block_offset, values, storage.store);
            ::rocprim::syncthreads();
            block_offset += items_per_block;
        }

        // Load the last (probably partial) block and continue scanning
        const unsigned int valid_count = end_offset - block_offset;
        block_load_type().load(input + block_offset, values, valid_count, storage.load);
        ::rocprim::syncthreads();
        // Perform scan operation
        segmented_scan_block_scan<Exclusive, true, block_scan_type>(
            values, values, prefix, storage.scan, scan_op
        );
        ::rocprim::syncthreads();
        // Store the partial block
        block_store_type().store(output + block_offset, values, valid_count, storage.store);
    }
}

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_REDUCE_HPP_
