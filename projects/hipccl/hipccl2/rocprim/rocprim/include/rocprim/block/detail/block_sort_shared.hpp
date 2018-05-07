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
        this->sort_impl<BlockSize>(
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
        this->sort_impl<BlockSize>(
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
        unsigned int next_id = k;
        Key next_key = storage.key[next_id];
        bool compare = compare_function(next_key, key);
        bool swap = compare ^ (next_id < flat_tid) ^ dir;
        key = swap ? next_key : key;
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
        unsigned int next_id = k;
        Key next_key = storage.key[next_id];
        Value next_value = storage.value[next_id];
        bool compare = compare_function(next_key, key);
        bool swap = compare ^ (next_id < flat_tid) ^ dir;
        key = swap ? next_key : key;
        value = swap ? next_value : value;
    }

    template<
        unsigned int Size,
        class BinaryFunction,
        class... KeyValue
    >
    ROCPRIM_DEVICE inline
    typename std::enable_if<detail::is_power_of_two(Size)>::type
    sort_impl(const unsigned int flat_tid,
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

        for(unsigned int length = 1; length < Size; length <<= 1)
        {
            bool dir = (flat_tid & (length << 1)) != 0;
            for(unsigned int k = length; k > 0; k >>= 1)
            {
                swap(kv..., flat_tid, flat_tid ^ k, dir, storage, compare_function);
                ::rocprim::syncthreads();
                load(kv..., flat_tid, storage);
            }
        }

        store(kv..., flat_tid, storage);
    }
    
    template<
        unsigned int Size,
        class BinaryFunction,
        class... KeyValue
    >
    ROCPRIM_DEVICE inline
    typename std::enable_if<!detail::is_power_of_two(Size)>::type
    sort_impl(const unsigned int flat_tid,
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

        bool is_even = (flat_tid % 2 == 0);
        unsigned int odd_id = (is_even) ? std::max(flat_tid, (unsigned int) 1) - 1 : std::min(flat_tid + 1, Size - 1);
        unsigned int even_id = (is_even) ? std::min(flat_tid + 1, Size - 1) : std::max(flat_tid, (unsigned int) 1) - 1;
        
        for(unsigned int i = 0; i < Size; i++)
        {
            unsigned int next_id = (i % 2 == 0) ? even_id : odd_id;
            swap(kv..., flat_tid, next_id, 0, storage, compare_function);
            ::rocprim::syncthreads();
            load(kv..., flat_tid, storage);
        }

        store(kv..., flat_tid, storage);
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_SORT_SHARED_HPP_
