// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
// THE SOFTWARE

#ifndef ROCPRIM_WARP_DETAIL_WARP_SORT_STABLE_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_SORT_STABLE_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../functional.hpp"
#include "../../intrinsics.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<typename Key,
         unsigned int BlockSize,
         unsigned int WarpSize,
         unsigned int ItemsPerThread,
         typename Value>
class warp_sort_stable
{
private:
    constexpr static unsigned int items_per_block = BlockSize * ItemsPerThread;
    constexpr static bool         with_values = !std::is_same<Value, rocprim::empty_type>::value;

    struct storage_type_keys
    {
        Key keys[items_per_block];
    };

    struct storage_type_keys_values
    {
        Key   keys[items_per_block];
        Value values[items_per_block];
    };

    using storage_type_
        = std::conditional_t<with_values, storage_type_keys_values, storage_type_keys>;

    /// Sort the keys and values of each thread separately.
    template<bool is_incomplete, typename CompareFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void thread_sort(Key (&thread_keys)[ItemsPerThread],
                                                   CompareFunction    compare_function,
                                                   const unsigned int input_size = items_per_block)
    {
        const auto thread_offset     = rocprim::flat_block_thread_id() * ItemsPerThread;
        const auto thread_input_size = thread_offset > input_size ? 0 : input_size - thread_offset;

        ROCPRIM_UNROLL
        for(auto i = 0u; i < ItemsPerThread; ++i)
        {
            ROCPRIM_UNROLL
            for(auto j = i & 1u; j < ItemsPerThread - 1u; j += 2u)
            {
                if(j + 1 < thread_input_size
                   && compare_function(thread_keys[j + 1], thread_keys[j]))
                {
                    ::rocprim::swap(thread_keys[j + 1], thread_keys[j]);
                }
            }
        }
    }

    /// Sort the keys and values of each thread separately.
    template<bool is_incomplete, typename CompareFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void thread_sort(Key (&thread_keys)[ItemsPerThread],
                                                   Value (&thread_values)[ItemsPerThread],
                                                   CompareFunction    compare_function,
                                                   const unsigned int input_size = items_per_block)
    {
        const auto thread_offset     = rocprim::flat_block_thread_id() * ItemsPerThread;
        const auto thread_input_size = thread_offset > input_size ? 0 : input_size - thread_offset;

        ROCPRIM_UNROLL
        for(auto i = 0u; i < ItemsPerThread; ++i)
        {
            ROCPRIM_UNROLL
            for(auto j = i & 1u; j < ItemsPerThread - 1u; j += 2u)
            {
                if(j + 1 < thread_input_size
                   && compare_function(thread_keys[j + 1], thread_keys[j]))
                {
                    ::rocprim::swap(thread_keys[j + 1], thread_keys[j]);
                    ::rocprim::swap(thread_values[j + 1], thread_values[j]);
                }
            }
        }
    }

    template<bool is_incomplete, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void merge_path_merge(Key (&thread_keys)[ItemsPerThread],
                                                        storage_type_&     storage,
                                                        BinaryFunction     compare_function,
                                                        const unsigned int input_size
                                                        = items_per_block)
    {
        const auto lane = lane_id();
        const auto warp = warp_id();

        const auto warp_offset     = warp * ItemsPerThread * device_warp_size();
        const auto warp_input_size = warp_offset > input_size ? 0 : input_size - warp_offset;
        const auto shared_keys     = &storage.keys[warp_offset];

        ROCPRIM_UNROLL
        for(auto partition_size = 1u; partition_size < WarpSize; partition_size <<= 1u)
        {
            ROCPRIM_UNROLL
            for(auto i = 0u; i < ItemsPerThread; ++i)
            {
                shared_keys[ItemsPerThread * lane + i] = thread_keys[i];
            }

            wave_barrier();

            const auto size = partition_size * ItemsPerThread;
            const auto mask = (partition_size * 2) - 1;

            const auto start       = lane & ~mask;
            const auto keys1_begin = start * ItemsPerThread;
            const auto keys1_end   = std::min(keys1_begin + size, warp_input_size);
            const auto keys2_begin = keys1_end;
            const auto keys2_end   = std::min(keys2_begin + size, warp_input_size);

            const auto diag      = std::min(ItemsPerThread * (mask & lane), warp_input_size);
            const auto partition = merge_path(&shared_keys[keys1_begin],
                                              &shared_keys[keys2_begin],
                                              keys1_end - keys1_begin,
                                              keys2_end - keys2_begin,
                                              diag,
                                              compare_function);

            const auto keys1_merge_begin = keys1_begin + partition;
            const auto keys2_merge_begin = keys2_begin + diag - partition;

            const range_t range = {
                keys1_merge_begin,
                keys1_end,
                keys2_merge_begin,
                keys2_end,
            };

            serial_merge(shared_keys, thread_keys, range, compare_function);

            wave_barrier();
        }
    }

    template<bool is_incomplete, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void merge_path_merge(Key (&thread_keys)[ItemsPerThread],
                                                        Value (&thread_values)[ItemsPerThread],
                                                        storage_type_&     storage,
                                                        BinaryFunction     compare_function,
                                                        const unsigned int input_size
                                                        = items_per_block)
    {
        const auto lane = lane_id();
        const auto warp = warp_id();

        const auto warp_offset     = warp * ItemsPerThread * device_warp_size();
        const auto warp_input_size = warp_offset > input_size ? 0 : input_size - warp_offset;
        const auto shared_keys     = &storage.keys[warp_offset];
        const auto shared_values   = &storage.values[warp_offset];

        ROCPRIM_UNROLL
        for(auto partition_size = 1u; partition_size < WarpSize; partition_size <<= 1u)
        {
            ROCPRIM_UNROLL
            for(auto i = 0u; i < ItemsPerThread; ++i)
            {
                shared_keys[ItemsPerThread * lane + i]   = thread_keys[i];
                shared_values[ItemsPerThread * lane + i] = thread_values[i];
            }

            wave_barrier();

            const auto size = partition_size * ItemsPerThread;
            const auto mask = (partition_size * 2) - 1;

            const auto start       = lane & ~mask;
            const auto keys1_begin = start * ItemsPerThread;
            const auto keys1_end   = std::min(keys1_begin + size, warp_input_size);
            const auto keys2_begin = keys1_end;
            const auto keys2_end   = std::min(keys2_begin + size, warp_input_size);

            const auto diag      = std::min(ItemsPerThread * (mask & lane), warp_input_size);
            const auto partition = merge_path(&shared_keys[keys1_begin],
                                              &shared_keys[keys2_begin],
                                              keys1_end - keys1_begin,
                                              keys2_end - keys2_begin,
                                              diag,
                                              compare_function);

            const auto keys1_merge_begin = keys1_begin + partition;
            const auto keys2_merge_begin = keys2_begin + diag - partition;

            const range_t range = {
                keys1_merge_begin,
                keys1_end,
                keys2_merge_begin,
                keys2_end,
            };

            serial_merge(shared_keys,
                         thread_keys,
                         shared_values,
                         thread_values,
                         range,
                         compare_function);

            wave_barrier();
        }
    }

public:
    static_assert(detail::is_power_of_two(WarpSize), "WarpSize must be power of 2");

    using storage_type = raw_storage<storage_type_>;

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key& thread_key, BinaryFunction compare_function)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort(thread_key, storage, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void
        sort(Key& thread_key, storage_type& storage, BinaryFunction compare_function)
    {
        Key thread_keys[] = {thread_key};
        sort(thread_keys, storage, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_keys)[ItemsPerThread],
                                            BinaryFunction compare_function)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort(thread_keys, storage, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_keys)[ItemsPerThread],
                                            storage_type&  storage,
                                            BinaryFunction compare_function)
    {
        thread_sort<false>(thread_keys, compare_function);

        merge_path_merge<false>(thread_keys, storage.get(), compare_function);
        syncthreads();
    }

    template<class BinaryFunction, class V = Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE void
        sort(Key& thread_key, Value& thread_value, BinaryFunction compare_function)
    {
        Key   thread_keys[]   = {thread_key};
        Value thread_values[] = {thread_value};
        sort(thread_keys, thread_values, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key&           thread_key,
                                            Value&         thread_value,
                                            storage_type&  storage,
                                            BinaryFunction compare_function)
    {
        Key   thread_keys[]   = {thread_key};
        Value thread_values[] = {thread_value};
        sort(thread_keys, thread_values, storage, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_keys)[ItemsPerThread],
                                            Value (&thread_values)[ItemsPerThread],
                                            BinaryFunction compare_function)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort(thread_keys, thread_values, storage, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_keys)[ItemsPerThread],
                                            storage_type&      storage,
                                            const unsigned int input_size,
                                            BinaryFunction     compare_function)
    {
        thread_sort<true>(thread_keys, compare_function, input_size);

        merge_path_merge<true>(thread_keys, storage.get(), compare_function, input_size);

        syncthreads();
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_keys)[ItemsPerThread],
                                            Value (&thread_values)[ItemsPerThread],
                                            storage_type&  storage,
                                            BinaryFunction compare_function)
    {
        thread_sort<false>(thread_keys, thread_values, compare_function);

        merge_path_merge<false>(thread_keys, thread_values, storage.get(), compare_function);
        syncthreads();
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_keys)[ItemsPerThread],
                                            Value (&thread_values)[ItemsPerThread],
                                            storage_type&      storage,
                                            const unsigned int input_size,
                                            BinaryFunction     compare_function)
    {
        thread_sort<true>(thread_keys, thread_values, compare_function, input_size);

        merge_path_merge<true>(thread_keys,
                               thread_values,
                               storage.get(),
                               compare_function,
                               input_size);

        syncthreads();
    }
};

template<typename Key, unsigned int BlockSize, unsigned int WarpSize, typename Value>
class warp_sort_stable<Key, BlockSize, WarpSize, 1, Value>
{
private:
    constexpr static unsigned items_per_thread = 1;
    /// Merge consecutive pairs of ranges by computing for each item a rank in the new output.
    ///
    /// Given an input like
    /// | 1 7 | 3 6 | 4 5 | 0 2 |
    /// The output will be
    /// | 0 3 | 1 2 | 6 7 | 4 5 |
    /// Permuting the input by the output gives the merged ranges:
    /// | 1 3 6 7 | 0 2 4 5 |
    ///
    /// \param m - The size of each subsequence to merge. The output consists of indices
    /// for sorted ranges of 2 * m elements.
    template<bool is_incomplete, typename BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE int merge_rank(const unsigned int m,
                                                 Key&               thread_key,
                                                 BinaryFunction     compare_function,
                                                 const unsigned int valid_items = BlockSize)
    {
        // The thread's index in the current warp.
        const auto lane = lane_id();
        // The size of each merged subsequence.
        const auto n = m * 2;
        // The thread's index in its (merged) subsequence.
        const auto index = lane % n;
        // Whether the thread is in the lower- or upper-half of the merged range.
        const auto is_lower = index < m;
        // The starting offset of the (merged) subsequence that the thread is in.
        const auto base = lane - index;

        // The starting index of the to-be-searched subsequence of elements. If in the lower
        // half, this points to the first element of the upper half, and vice versa.
        auto begin = base + (is_lower ? m : 0);
        // The past-ending index of the to-be-searched subsequence of elements.
        auto end = begin + m;

        // Note: we cannot use a while loop here because all threads need to be active during the
        // shuffle.
        ROCPRIM_UNROLL
        for(auto i = 1u; i <= m; i <<= 1u)
        {
            const auto mid = (begin + end) / 2;
            // Swap keys if in the lower half to eliminate a more expensive divergent branch in the comparator.
            // Note: this needs to be done in order to achieve stability, in the left subsequence we want the index
            // to be before any equal elements, but in the right subsequence it must be after.
            auto key_a = thread_key;
            auto key_b = warp_shuffle(thread_key, mid);
            if(is_lower)
                ::rocprim::swap(key_a, key_b);

            const auto mid_smaller = ((!is_incomplete || (lane < valid_items && mid < valid_items))
                                      && compare_function(key_a, key_b))
                                     == is_lower;

            if(mid_smaller && begin != end)
                begin = mid + 1;
            else
                end = mid;
        }

        // The rank of an item in the merged sequence is given by
        // rank(merged) = rank(left) + rank(right).
        // The rank in one of the subsequences is given by `begin`, and the other is given by `index`.
        // Note that for the left subsequence `begin` is offset by `m`, and for the right subsequence
        // `index` is offset by `m`. Subtracting `m` for the result this gives the correct final rank.
        return index + begin - m;
    }

public:
    static_assert(detail::is_power_of_two(WarpSize), "WarpSize must be power of 2");

    using storage_type = empty_storage_type;

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key& thread_key, BinaryFunction compare_function)
    {
        ROCPRIM_UNROLL
        for(auto i = 1u; i < WarpSize; i <<= 1u)
        {
            const auto thread_rank = merge_rank<false>(i, thread_key, compare_function);
            thread_key             = warp_permute(thread_key, thread_rank);
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void
        sort(Key& thread_key, storage_type& storage, BinaryFunction compare_function)
    {
        (void)storage;
        sort(thread_key, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_keys)[items_per_thread],
                                            BinaryFunction compare_function)
    {
        sort(thread_keys[0], compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_keys)[items_per_thread],
                                            storage_type&  storage,
                                            BinaryFunction compare_function)
    {
        sort(thread_keys[0], storage, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_keys)[items_per_thread],
                                            storage_type&      storage,
                                            const unsigned int input_size,
                                            BinaryFunction     compare_function)
    {
        sort(thread_keys[0], storage, input_size, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key&               thread_key,
                                            storage_type&      storage,
                                            const unsigned int input_size,
                                            BinaryFunction     compare_function)
    {
        (void)storage;

        const auto warp_offset     = warp_id() * device_warp_size();
        const auto warp_input_size = warp_offset > input_size ? 0 : input_size - warp_offset;

        ROCPRIM_UNROLL
        for(auto i = 1u; i < WarpSize; i <<= 1u)
        {
            const auto thread_rank
                = merge_rank<true>(i, thread_key, compare_function, warp_input_size);
            thread_key = warp_permute(thread_key, thread_rank);
        }
    }

    template<class BinaryFunction, class V = Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE typename std::enable_if<(sizeof(V) <= sizeof(int))>::type
        sort(Key& thread_key, V& thread_value, BinaryFunction compare_function)
    {
        ROCPRIM_UNROLL
        for(auto i = 1u; i < WarpSize; i <<= 1u)
        {
            const auto thread_rank = merge_rank<false>(i, thread_key, compare_function);
            thread_key             = warp_permute(thread_key, thread_rank);
            thread_value           = warp_permute(thread_value, thread_rank);
        }
    }

    template<class BinaryFunction, class V = Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE typename std::enable_if<!(sizeof(V) <= sizeof(int))>::type
        sort(Key& thread_key, V& thread_value, BinaryFunction compare_function)
    {
        // Use indices to reduce the amount of permutations.
        auto value_index = lane_id();
        sort(thread_key, value_index, compare_function);
        // Perform a shuffle to get the final value.
        thread_value = warp_shuffle(thread_value, value_index);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key&           thread_key,
                                            Value&         thread_value,
                                            storage_type&  storage,
                                            BinaryFunction compare_function)
    {
        (void)storage;
        sort(compare_function, thread_key, thread_value);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_keys)[items_per_thread],
                                            Value (&thread_values)[items_per_thread],
                                            BinaryFunction compare_function)
    {
        sort(thread_keys[0], thread_values[0], compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_keys)[items_per_thread],
                                            Value (&thread_values)[items_per_thread],
                                            storage_type&  storage,
                                            BinaryFunction compare_function)
    {
        (void)storage;
        sort(thread_keys[0], thread_values[0], compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_keys)[items_per_thread],
                                            Value (&thread_values)[items_per_thread],
                                            storage_type&  storage,
                                            unsigned int   input_size,
                                            BinaryFunction compare_function)
    {
        (void)storage;
        sort(thread_keys[0], thread_values[0], storage, input_size, compare_function);
    }

    template<class BinaryFunction, typename V = Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE typename std::enable_if<(sizeof(V) <= sizeof(int))>::type
        sort(Key&           thread_key,
             V&             thread_value,
             storage_type&  storage,
             unsigned int   input_size,
             BinaryFunction compare_function)
    {
        (void)storage;

        const auto warp_offset     = warp_id() * device_warp_size();
        const auto warp_input_size = warp_offset > input_size ? 0 : input_size - warp_offset;

        ROCPRIM_UNROLL
        for(auto i = 1u; i < WarpSize; i <<= 1u)
        {
            const auto thread_rank
                = merge_rank<true>(i, thread_key, compare_function, warp_input_size);
            thread_key   = warp_permute(thread_key, thread_rank);
            thread_value = warp_permute(thread_value, thread_rank);
        }
    }

    template<class BinaryFunction, typename V = Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE typename std::enable_if<!(sizeof(V) <= sizeof(int))>::type
        sort(Key&           thread_key,
             V&             thread_value,
             storage_type&  storage,
             unsigned int   input_size,
             BinaryFunction compare_function)
    {
        // Use indices to reduce the amount of permutations.
        auto value_index = lane_id();
        sort(thread_key, value_index, storage, input_size, compare_function);
        // Perform a shuffle to get the final value.
        thread_value = warp_shuffle(thread_value, value_index);
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_SORT_SHUFFLE_HPP_
