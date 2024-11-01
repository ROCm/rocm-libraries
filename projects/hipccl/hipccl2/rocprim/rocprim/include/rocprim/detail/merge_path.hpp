// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DETAIL_MERGE_PATH_HPP_
#define ROCPRIM_DETAIL_MERGE_PATH_HPP_

#include "../config.hpp"

#include <cassert>
#include <iterator>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class OffsetT = unsigned int>
struct range_t
{
    OffsetT begin1;
    OffsetT end1;
    OffsetT begin2;
    OffsetT end2;

    /// \brief Number of elements in first range.
    ROCPRIM_DEVICE ROCPRIM_INLINE
    constexpr OffsetT count1() const
    {
        return end1 - begin1;
    }

    /// \brief Number of elements in second range.
    ROCPRIM_DEVICE ROCPRIM_INLINE
    constexpr OffsetT count2() const
    {
        return end2 - begin2;
    }
};

template<class KeysInputIterator1, class KeysInputIterator2, class OffsetT, class BinaryFunction>
ROCPRIM_HOST_DEVICE ROCPRIM_INLINE OffsetT merge_path(KeysInputIterator1 keys_input1,
                                                      KeysInputIterator2 keys_input2,
                                                      const OffsetT      input1_size,
                                                      const OffsetT      input2_size,
                                                      const OffsetT      diag,
                                                      BinaryFunction     compare_function)
{
    using key_type_1 = typename std::iterator_traits<KeysInputIterator1>::value_type;
    using key_type_2 = typename std::iterator_traits<KeysInputIterator2>::value_type;

    OffsetT begin = diag < input2_size ? 0u : diag - input2_size;
    OffsetT end   = min(diag, input1_size);

    while(begin < end)
    {
        OffsetT    a       = (begin + end) / 2;
        OffsetT    b       = diag - 1 - a;
        key_type_1 input_a = keys_input1[a];
        key_type_2 input_b = keys_input2[b];
        if(!compare_function(input_b, input_a))
        {
            begin = a + 1;
        }
        else
        {
            end = a;
        }
    }

    return begin;
}

template<unsigned int ItemsPerThread,
         bool         AllowUnsafe,
         class KeyType,
         class BinaryFunction,
         class OutputFunction,
         class OffsetT>
ROCPRIM_DEVICE ROCPRIM_INLINE
void serial_merge(KeyType*                keys_shared,
                  const range_t<OffsetT>& range,
                  BinaryFunction          compare_function,
                  OutputFunction          output_function)
{
    // Pre condition, we're including some edge cases too.
    if (!AllowUnsafe && range.begin1 > range.end1 && range.begin2 > range.end2)
        return; // don't do anything, we have invalid inputs  

    // More descriptive names for ranges:
    auto       idx_a = range.begin1;
    auto       idx_b = range.begin2;
    const auto end_a = range.end1;
    const auto end_b = range.end2;

    // Pre-loaded keys so we don't have to re-fetch multiple times from memory.
    // These will be updated every iteration.
    KeyType key_a;
    KeyType key_b;

    // Only load valid keys, otherwise might be out of bounds!
    // If we allow unsafe, this check is not done.
    if(AllowUnsafe || idx_a < end_a)
    {
        key_a = keys_shared[idx_a];
    }
    if(AllowUnsafe || idx_b < end_b)
    {
        key_b = keys_shared[idx_b];
    }

    ROCPRIM_UNROLL
    for(unsigned int i = 0; i < ItemsPerThread; ++i)
    {
        // If we don't have any in b, we always take from a. Then, if we don't
        // have any in a, we take from b. Otherwise we take the smallest item.
        const bool take_a
            = (idx_b >= end_b) || ((idx_a < end_a) && !compare_function(key_b, key_a));

        // Retrieve info about the smallest key.
        const auto idx = take_a ? idx_a : idx_b;
        const auto end = take_a ? end_a : end_b;
        const auto key = take_a ? key_a : key_b;

        // Output results.
        output_function(i, key, idx);

        // Get the next idx, if we allow unsafe we may access out-of-bounds elements.
        const auto next_idx = idx + 1;

        // Load the next item. The compiler *should* be smart enough to optimize
        // away the case where we don't have any items to read.
        const auto next_key = keys_shared[AllowUnsafe ? next_idx : min(next_idx, end - 1)];

        // Store the info about the next key.
        if(take_a)
        {
            idx_a = next_idx;
            key_a = next_key;
        }
        else
        {
            idx_b = next_idx;
            key_b = next_key;
        }
    }

    // We don't finish with a block sync since this may be used on thread or
    // warp granularity!
}

template<bool AllowUnsafe = false,
         class KeyType,
         unsigned int ItemsPerThread,
         class BinaryFunction,
         class OffsetT>
ROCPRIM_DEVICE ROCPRIM_INLINE
void serial_merge(KeyType* keys_shared,
                  KeyType (&outputs)[ItemsPerThread],
                  unsigned int (&indices)[ItemsPerThread],
                  const range_t<OffsetT>& range,
                  BinaryFunction          compare_function)
{
    serial_merge<ItemsPerThread, AllowUnsafe>(
        keys_shared,
        range,
        compare_function,
        [&](const unsigned int& i, const KeyType& key, const OffsetT& index)
        {
            outputs[i] = key;
            indices[i] = index;
        });
}

template<bool AllowUnsafe = false,
         class KeyType,
         unsigned int ItemsPerThread,
         class BinaryFunction,
         class OffsetT>
ROCPRIM_DEVICE ROCPRIM_INLINE
void serial_merge(KeyType* keys_shared,
                  KeyType (&outputs)[ItemsPerThread],
                  const range_t<OffsetT>& range,
                  BinaryFunction          compare_function)
{
    serial_merge<ItemsPerThread, AllowUnsafe>(
        keys_shared,
        range,
        compare_function,
        [&](const unsigned int& i, const KeyType& key, const OffsetT&) { outputs[i] = key; });
}

template<bool AllowUnsafe = false,
         class KeyType,
         class ValueType,
         unsigned int ItemsPerThread,
         class BinaryFunction,
         class OffsetT>
ROCPRIM_DEVICE ROCPRIM_INLINE
void serial_merge(KeyType* keys_shared,
                  KeyType (&outputs)[ItemsPerThread],
                  ValueType* values_shared,
                  ValueType (&values)[ItemsPerThread],
                  const range_t<OffsetT>& range,
                  BinaryFunction          compare_function)
{
    serial_merge<ItemsPerThread, AllowUnsafe>(
        keys_shared,
        range,
        compare_function,
        [&](const unsigned int& i, const KeyType& key, const OffsetT& index)
        {
            outputs[i] = key;
            values[i]  = values_shared[index];
        });
}

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DETAIL_MERGE_PATH_HPP_
