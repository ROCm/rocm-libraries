// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

template<class Key, unsigned int VirtualWaveSize, class Value>
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
        unsigned int index; // original index：lane_id * IPT + item_idx
    };

    // Wrapper for compare function for stability.
    template<class BinaryFunction>
    struct stable_comparator
    {
        BinaryFunction user_compare;

        ROCPRIM_DEVICE ROCPRIM_INLINE
        stable_comparator(BinaryFunction func) : user_compare(func) {}

        ROCPRIM_DEVICE ROCPRIM_INLINE
        bool operator()(const stable_key_t& a, const stable_key_t& b) const
        {
            if (user_compare(a.key, b.key))
            {
                return true;
            }
            if (user_compare(b.key, a.key))
            {
                return false;
            }

            // if two elements are equal (a == b)，compare the original index.
            return a.index < b.index;
        }
    };

public:

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_values)[ItemsPerThread], BinaryFunction compare_function)
    {
        // Get data in stable wrapper.
        stable_key_t stable_items[ItemsPerThread];
        const unsigned int flat_id = detail::logical_lane_id<VirtualWaveSize>() * ItemsPerThread;

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            stable_items[i].key = thread_values[i];
            stable_items[i].index = flat_id + i;
        }

        // Stable sort with wrapped data and comparator.
        warp_shuffle_sort_impl<VirtualWaveSize, ItemsPerThread>::bitonic_sort(
            stable_comparator<BinaryFunction>(compare_function),
            stable_items
        );

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
        Key temp_arr[1] = { thread_value };
        sort<1>(temp_arr, compare_function);
        thread_value = temp_arr[0];
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key& thread_value, storage_type& storage, BinaryFunction compare_function)
    {
        (void)storage;
        sort(thread_value, compare_function);
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_values)[ItemsPerThread],
              storage_type&  storage,
              BinaryFunction compare_function)
    {
        (void)storage;
        sort(thread_values, compare_function);
    }

    template<unsigned int ItemsPerThread, class BinaryFunction, class V = Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              BinaryFunction compare_function)
    {
        // Instead of passing wrapped data between lanes we pass indices and gather values after sorting.
        stable_key_t stable_items[ItemsPerThread];
        const unsigned int flat_id = detail::logical_lane_id<VirtualWaveSize>() * ItemsPerThread;

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            stable_items[i].key = thread_keys[i];
            stable_items[i].index = flat_id + i;
        }

        warp_shuffle_sort_impl<VirtualWaveSize, ItemsPerThread>::bitonic_sort(
            stable_comparator<BinaryFunction>(compare_function),
            stable_items
        );

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            thread_keys[i] = stable_items[i].key;
        }
        
        // Create a copy of 'thread_values' so we can swizzle them around without overwriting.
        V source_values[ItemsPerThread];
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            source_values[i] = thread_values[i];
        }

        // We will now write into 'thread_values' from 'copy'. We do this by checking for
        // the matrix between destination and source index, since we cannot dynamically
        // index registers.
        //
        // This requires IPT^2 shuffles because both need index lane and item offset.
        ROCPRIM_UNROLL
        for(unsigned int dst_item = 0; dst_item < ItemsPerThread; ++dst_item)
        {
            unsigned int src_idx = stable_items[dst_item].index;
            
            unsigned int src_lane = src_idx / ItemsPerThread;
            unsigned int src_item_offset = src_idx % ItemsPerThread;

            ROCPRIM_UNROLL
            for(unsigned int k = 0; k < ItemsPerThread; ++k)
            {
                // This shuffle can potentially be moved into the branch. We can then
                // trade the extra masking for in-place shuffle which may potentially
                // be faster. This may require an extra memory fence since the previous
                // duplication into 'copy' must be finalized and we can't reuse
                // registers as freely.
                V val = warp_shuffle(source_values[k], src_lane, VirtualWaveSize);

                if(k == src_item_offset)
                {
                    thread_values[dst_item] = val;
                }
            }
        }
    }

    template<class BinaryFunction, class V = Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key& thread_key, Value& thread_value, BinaryFunction compare_function)
    {
        Key k_arr[1] = { thread_key };
        Value v_arr[1] = { thread_value };
        sort<1>(k_arr, v_arr, compare_function);
        thread_key = k_arr[0];
        thread_value = v_arr[0];
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

    template<unsigned int ItemsPerThread, class BinaryFunction, class V = Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              storage_type&  storage,
              BinaryFunction compare_function)
    {
        (void)storage;
        sort(thread_keys, thread_values, compare_function);
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_SORT_SHUFFLE_STABLE_HPP_
