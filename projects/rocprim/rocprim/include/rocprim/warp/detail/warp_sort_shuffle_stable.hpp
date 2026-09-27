// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_WARP_DETAIL_WARP_SORT_SHUFFLE_STABLE_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_SORT_SHUFFLE_STABLE_HPP_

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../functional.hpp"
#include "../../intrinsics/thread.hpp"
#include "../../intrinsics/warp_shuffle.hpp"

#include "warp_sort_shuffle.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Key,
         unsigned int BlockSize,
         unsigned int VirtualWaveSize,
         unsigned int ItemsPerThread,
         class Value>
struct warp_sort_shuffle_stable
{
public:
    static_assert(detail::is_power_of_two(VirtualWaveSize), "VirtualWaveSize must be power of 2");

    using storage_type = ::rocprim::detail::empty_storage_type;

private:
    // Wrapper for key and original index.
    struct stable_key_t
    {
        Key          key;
        unsigned int index; // original index: lane_id * IPT + item_idx
    };

    // Wrapper for compare function for stability.
    template<class BinaryFunction>
    struct stable_comparator
    {
        BinaryFunction user_compare;

        ROCPRIM_DEVICE ROCPRIM_INLINE
        stable_comparator(BinaryFunction func)
            : user_compare(func)
        {}

        ROCPRIM_DEVICE ROCPRIM_INLINE
        bool operator()(const stable_key_t& a, const stable_key_t& b) const
        {
            if(user_compare(a.key, b.key))
            {
                return true;
            }
            if(user_compare(b.key, a.key))
            {
                return false;
            }

            // if two elements are equal (a == b), compare the original index.
            return a.index < b.index;
        }
    };

public:
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_values)[ItemsPerThread], BinaryFunction compare_function)
    {
        // Get data in stable wrapper.
        stable_key_t       stable_items[ItemsPerThread];
        const unsigned int flat_id = detail::logical_lane_id<VirtualWaveSize>() * ItemsPerThread;

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            stable_items[i].key   = thread_values[i];
            stable_items[i].index = flat_id + i;
        }

        // Stable sort with wrapped data and comparator.
        warp_shuffle_sort_impl<VirtualWaveSize, ItemsPerThread>::bitonic_sort(
            stable_comparator<BinaryFunction>(compare_function),
            stable_items);

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            thread_values[i] = stable_items[i].key;
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key& thread_value, BinaryFunction compare_function)
    {
        stable_key_t item;
        item.key   = thread_value;
        item.index = detail::logical_lane_id<VirtualWaveSize>();

        warp_shuffle_sort_impl<VirtualWaveSize, 1>::bitonic_sort(
            stable_comparator<BinaryFunction>(compare_function),
            item);
        thread_value = item.key;
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key& thread_value, storage_type& storage, BinaryFunction compare_function)
    {
        (void)storage;
        sort(thread_value, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_values)[ItemsPerThread],
              storage_type&  storage,
              BinaryFunction compare_function)
    {
        (void)storage;
        sort(thread_values, compare_function);
    }

    // Do not allow input_size since this function uses bitonic sort.
    // Throw compilation error by deleting the implementation.
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_values)[ItemsPerThread],
              storage_type&      storage,
              const unsigned int input_size,
              BinaryFunction     compare_function) = delete;

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key&               thread_value,
              storage_type&      storage,
              const unsigned int input_size,
              BinaryFunction     compare_function) = delete;

    template<class BinaryFunction, class V = Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              BinaryFunction compare_function)
    {
        // Instead of passing wrapped data between lanes we pass indices and gather values after sorting.
        stable_key_t       stable_items[ItemsPerThread];
        const unsigned int flat_id = detail::logical_lane_id<VirtualWaveSize>() * ItemsPerThread;

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            stable_items[i].key   = thread_keys[i];
            stable_items[i].index = flat_id + i;
        }

        warp_shuffle_sort_impl<VirtualWaveSize, ItemsPerThread>::bitonic_sort(
            stable_comparator<BinaryFunction>(compare_function),
            stable_items);

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            thread_keys[i] = stable_items[i].key;
        }

        warp_shuffle_sort_impl<VirtualWaveSize, ItemsPerThread>::apply_permutation(
            thread_values,
            [&](unsigned int i) { return stable_items[i].index; });
    }

    template<class BinaryFunction, class V = Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key& thread_key, Value& thread_value, BinaryFunction compare_function)
    {
        stable_key_t item;
        item.key   = thread_key;
        item.index = detail::logical_lane_id<VirtualWaveSize>();

        warp_shuffle_sort_impl<VirtualWaveSize, 1>::bitonic_sort(
            stable_comparator<BinaryFunction>(compare_function),
            item);
        thread_key = item.key;

        // Shuffle value
        unsigned int src_lane = item.index; // index is just lane_id here
        thread_value          = warp_shuffle(thread_value, src_lane, VirtualWaveSize);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key&           thread_key,
              Value&         thread_value,
              storage_type&  storage,
              BinaryFunction compare_function)
    {
        (void)storage;
        sort(thread_key, thread_value, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              storage_type&  storage,
              BinaryFunction compare_function)
    {
        (void)storage;
        sort(thread_keys, thread_values, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              storage_type&      storage,
              const unsigned int input_size,
              BinaryFunction     compare_function) = delete;

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key&               thread_key,
              Value&             thread_value,
              storage_type&      storage,
              const unsigned int input_size,
              BinaryFunction     compare_function) = delete;
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_SORT_SHUFFLE_STABLE_HPP_
