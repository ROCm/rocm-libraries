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

#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_SORT_SHARED_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_SORT_SHARED_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class Key,
    unsigned int BlockSize,
    class Value
>
class block_sort_shared
{
public:
    static_assert(detail::is_power_of_two(BlockSize), "BlockSize must be power of 2");

    struct storage_type
    {
        Key key[BlockSize];
        Value value[BlockSize];
    };

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              storage_type& storage,
              BinaryFunction compare_function)
    {
        this->sort_impl(
            ::rocprim::flat_block_thread_id(),
            storage, compare_function,
            thread_key
        );
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              BinaryFunction compare_function)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->sort(thread_key, storage, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              Value& thread_value,
              storage_type& storage,
              BinaryFunction compare_function)
    {
        this->sort_impl(
            ::rocprim::flat_block_thread_id(),
            storage, compare_function,
            thread_key, thread_value
        );
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              Value& thread_value,
              BinaryFunction compare_function)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->sort(thread_key, thread_value, storage, compare_function);
    }

private:
    ROCPRIM_DEVICE inline
    void load(Key& k, const unsigned int flat_tid, storage_type& storage)
    {
        storage.key[flat_tid] = k;
        ::rocprim::syncthreads();
    }

    ROCPRIM_DEVICE inline
    void load(Key& k, Value& v, const unsigned int flat_tid, storage_type& storage)
    {
        storage.key[flat_tid] = k;
        storage.value[flat_tid] = v;
        ::rocprim::syncthreads();
    }

    ROCPRIM_DEVICE inline
    void store(Key& k, const unsigned int flat_tid, storage_type& storage)
    {
        k = storage.key[flat_tid];
    }

    ROCPRIM_DEVICE inline
    void store(Key& k, Value& v, const unsigned int flat_tid, storage_type& storage)
    {
        k = storage.key[flat_tid];
        v = storage.value[flat_tid];
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void swap(Key& key,
              const unsigned int flat_tid,
              const unsigned int k,
              const bool dir,
              storage_type& storage,
              BinaryFunction compare_function)
    {
        unsigned int sibling_idx = flat_tid ^ k;
        Key sibling_key = storage.key[sibling_idx];
        bool compare = compare_function(sibling_key, key);
        bool swap = compare ^ (sibling_idx < flat_tid) ^ dir;
        key = swap ? sibling_key : key;
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void swap(Key& key,
              Value& value,
              const unsigned int flat_tid,
              const unsigned int k,
              const bool dir,
              storage_type& storage,
              BinaryFunction compare_function)
    {
        unsigned int sibling_idx = flat_tid ^ k;
        Key sibling_key = storage.key[sibling_idx];
        Value sibling_value = storage.value[sibling_idx];
        bool compare = compare_function(sibling_key, key);
        bool swap = compare ^ (sibling_idx < flat_tid) ^ dir;
        key = swap ? sibling_key : key;
        value = swap ? sibling_value : value;
    }

    template<class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE inline
    void sort_impl(const unsigned int flat_tid,
                   storage_type& storage,
                   BinaryFunction compare_function,
                   KeyValue&... kv)
    {
        static constexpr unsigned int PairSize =  sizeof...(KeyValue);
        static_assert(
            PairSize < 3,
            "KeyValue parameter pack can 1 or 2 elements (key, or key and value)"
        );

        load(kv..., flat_tid, storage);

        for(unsigned int length = 1; length < BlockSize; length <<= 1)
        {
            bool dir = (flat_tid & (length << 1)) != 0;
            for(unsigned int k = length; k > 0; k >>= 1)
            {
                swap(kv..., flat_tid, k, dir, storage, compare_function);
                ::rocprim::syncthreads();
                load(kv..., flat_tid, storage);
            }
        }

        store(kv..., flat_tid, storage);
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_SORT_SHARED_HPP_
