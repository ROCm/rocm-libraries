// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../warp/warp_sort.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class Key,
    unsigned int BlockSizeX,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ,
    class Value
>
class block_sort_bitonic
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;

    template<class KeyType, class ValueType>
    struct storage_type_
    {
        KeyType key[BlockSize];
        ValueType value[BlockSize];
    };

    template<class KeyType>
    struct storage_type_<KeyType, empty_type>
    {
        KeyType key[BlockSize];
    };

public:
    using storage_type = detail::raw_storage<storage_type_<Key, Value>>;

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              storage_type& storage,
              BinaryFunction compare_function)
    {
        this->sort_impl<BlockSize>(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
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
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
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

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              storage_type& storage,
              const unsigned int size,
              BinaryFunction compare_function)
    {
        this->sort_impl(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(), size,
            storage, compare_function,
            thread_key
        );
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              Value& thread_value,
              storage_type& storage,
              const unsigned int size,
              BinaryFunction compare_function)
    {
        this->sort_impl(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(), size,
            storage, compare_function,
            thread_key, thread_value
        );
    }

private:
    ROCPRIM_DEVICE inline
    void copy_to_shared(Key& k, const unsigned int flat_tid, storage_type& storage)
    {
        storage_type_<Key, Value>& storage_ = storage.get();
        storage_.key[flat_tid] = k;
        ::rocprim::syncthreads();
    }

    ROCPRIM_DEVICE inline
    void copy_to_shared(Key& k, Value& v, const unsigned int flat_tid, storage_type& storage)
    {
        storage_type_<Key, Value>& storage_ = storage.get();
        storage_.key[flat_tid] = k;
        storage_.value[flat_tid] = v;
        ::rocprim::syncthreads();
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void swap(Key& key,
              const unsigned int flat_tid,
              const unsigned int next_id,
              const bool dir,
              storage_type& storage,
              BinaryFunction compare_function)
    {
        storage_type_<Key, Value>& storage_ = storage.get();
        Key next_key = storage_.key[next_id];
        bool compare = (next_id < flat_tid) ? compare_function(key, next_key) : compare_function(next_key, key);
        bool swap = compare ^ dir;
        if(swap)
        {
            key = next_key;
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void swap(Key& key,
              Value& value,
              const unsigned int flat_tid,
              const unsigned int next_id,
              const bool dir,
              storage_type& storage,
              BinaryFunction compare_function)
    {
        storage_type_<Key, Value>& storage_ = storage.get();
        Key next_key = storage_.key[next_id];
        Value next_value = storage_.value[next_id];
        bool compare = (next_id < flat_tid) ? compare_function(key, next_key) : compare_function(next_key, key);
        bool swap = compare ^ dir;
        if(swap)
        {
            key = next_key;
            value = next_value;
        }
    }

    template<
        unsigned int Size,
        class BinaryFunction,
        class... KeyValue
    >
    ROCPRIM_DEVICE inline
    typename std::enable_if<(Size <= ::rocprim::warp_size())>::type
    sort_power_two(const unsigned int flat_tid,
                   storage_type& storage,
                   BinaryFunction compare_function,
                   KeyValue&... kv)
    {
        (void) flat_tid;
        (void) storage;

        ::rocprim::warp_sort<Key, Size, Value> wsort;
        wsort.sort(kv..., compare_function);
    }

    template<
        unsigned int Size,
        class BinaryFunction,
        class... KeyValue
    >
    ROCPRIM_DEVICE inline
    typename std::enable_if<(Size > ::rocprim::warp_size())>::type
    sort_power_two(const unsigned int flat_tid,
                   storage_type& storage,
                   BinaryFunction compare_function,
                   KeyValue&... kv)
    {
        const auto warp_id_is_even = ((flat_tid / ::rocprim::warp_size()) % 2) == 0;
        ::rocprim::warp_sort<Key, ::rocprim::warp_size(), Value> wsort;
        auto compare_function2 =
            [compare_function, warp_id_is_even](const Key& a, const Key& b) mutable -> bool
            {
                auto r = compare_function(a, b);
                if(warp_id_is_even)
                    return r;
                return !r;
            };
        wsort.sort(kv..., compare_function2);

        #pragma unroll
        for(unsigned int length = ::rocprim::warp_size(); length < Size; length *= 2)
        {
            bool dir = (flat_tid & (length * 2)) != 0;
            #pragma unroll
            for(unsigned int k = length; k > 0; k /= 2)
            {
                copy_to_shared(kv..., flat_tid, storage);
                swap(kv..., flat_tid, flat_tid ^ k, dir, storage, compare_function);
                ::rocprim::syncthreads();
            }
        }
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

        sort_power_two<Size, BinaryFunction>(flat_tid, storage, compare_function, kv...);
    }

    // In case BlockSize is not a power-of-two, the slower odd-even mergesort function is used
    // instead of the bitonic sort function
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

        copy_to_shared(kv..., flat_tid, storage);

        bool is_even = (flat_tid % 2) == 0;
        unsigned int odd_id = (is_even) ? ::rocprim::max(flat_tid, 1u) - 1 : ::rocprim::min(flat_tid + 1, Size - 1);
        unsigned int even_id = (is_even) ? ::rocprim::min(flat_tid + 1, Size - 1) : ::rocprim::max(flat_tid, 1u) - 1;

        #pragma unroll
        for(unsigned int length = 0; length < Size; length++)
        {
            unsigned int next_id = (length % 2) == 0 ? even_id : odd_id;
            swap(kv..., flat_tid, next_id, 0, storage, compare_function);
            ::rocprim::syncthreads();
            copy_to_shared(kv..., flat_tid, storage);
        }
    }

    template<
        class BinaryFunction,
        class... KeyValue
    >
    ROCPRIM_DEVICE inline
    void sort_impl(const unsigned int flat_tid,
                   const unsigned int size,
                   storage_type& storage,
                   BinaryFunction compare_function,
                   KeyValue&... kv)
    {
        static constexpr unsigned int PairSize =  sizeof...(KeyValue);
        static_assert(
            PairSize < 3,
            "KeyValue parameter pack can 1 or 2 elements (key, or key and value)"
        );

        if(size > BlockSize)
        {
            return;
        }

        copy_to_shared(kv..., flat_tid, storage);

        bool is_even = (flat_tid % 2 == 0);
        unsigned int odd_id = (is_even) ? ::rocprim::max(flat_tid, 1u) - 1 : ::rocprim::min(flat_tid + 1, size - 1);
        unsigned int even_id = (is_even) ? ::rocprim::min(flat_tid + 1, size - 1) : ::rocprim::max(flat_tid, 1u) - 1;

        for(unsigned int length = 0; length < size; length++)
        {
            unsigned int next_id = (length % 2 == 0) ? even_id : odd_id;
            // Use only "valid" keys to ensure that compare_function will not use garbage keys
            // for example, as indices of an array (a lookup table)
            if(flat_tid < size)
            {
                swap(kv..., flat_tid, next_id, 0, storage, compare_function);
            }
            ::rocprim::syncthreads();
            copy_to_shared(kv..., flat_tid, storage);
        }
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_SORT_SHARED_HPP_
