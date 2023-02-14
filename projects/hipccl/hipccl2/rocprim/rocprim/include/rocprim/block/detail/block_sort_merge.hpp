// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_SORT_MERGE_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_SORT_MERGE_HPP_

#include "../../config.hpp"
#include "../../detail/merge_path.hpp"
#include "../../detail/various.hpp"
#include "../../warp/detail/warp_sort_stable.hpp"
#include "../../warp/warp_sort.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Key,
         unsigned int BlockSizeX,
         unsigned int BlockSizeY,
         unsigned int BlockSizeZ,
         unsigned int ItemsPerThread,
         class Value,
         bool Stable = false>
class block_sort_merge
{
    static constexpr const unsigned int BlockSize     = BlockSizeX * BlockSizeY * BlockSizeZ;
    static constexpr const unsigned int ItemsPerBlock = BlockSize * ItemsPerThread;
    static constexpr const unsigned int WarpSortSize  = std::min(BlockSize, 16u);
    static constexpr const bool with_values = !std::is_same<Value, rocprim::empty_type>::value;
    using warp_sort_type                            = std::conditional_t<
        Stable,
        rocprim::detail::warp_sort_stable<Key, BlockSize, WarpSortSize, ItemsPerThread, Value>,
        rocprim::warp_sort<Key, WarpSortSize, Value>>;

    static_assert(rocprim::detail::is_power_of_two(BlockSize),
                  "BlockSize must be a power of two for block_sort_merge!");

    static_assert(rocprim::detail::is_power_of_two(ItemsPerThread),
                  "ItemsPerThread must be a power of two for block_sort_merge!");

    template<bool with_values>
    union storage_type_
    {
        typename warp_sort_type::storage_type   warp_sort;
        detail::raw_storage<Key[ItemsPerBlock]> keys;
    };

    template<>
    union storage_type_<true>
    {
        typename warp_sort_type::storage_type warp_sort;
        struct
        {
            detail::raw_storage<Key[ItemsPerBlock]>   keys;
            detail::raw_storage<Value[ItemsPerBlock]> values;
        };
    };

public:
    using storage_type = storage_type_<with_values>;

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void
        sort(Key& thread_key, storage_type& storage, BinaryFunction compare_function)
    {
        Key thread_keys[] = {thread_key};
        this->sort_impl<ItemsPerBlock>(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            storage,
            compare_function,
            thread_keys);
        thread_key = thread_keys[0];
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_keys)[ItemsPerThread],
                                            storage_type&  storage,
                                            BinaryFunction compare_function)
    {
        this->sort_impl<ItemsPerBlock>(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            storage,
            compare_function,
            thread_keys);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void sort(Key& thread_key, BinaryFunction compare_function)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->sort(thread_key, storage, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void sort(Key (&thread_keys)[ItemsPerThread],
                                                  BinaryFunction compare_function)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->sort(thread_keys, storage, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key&           thread_key,
                                            Value&         thread_value,
                                            storage_type&  storage,
                                            BinaryFunction compare_function)
    {
        Key   thread_keys[]   = {thread_key};
        Value thread_values[] = {thread_value};
        this->sort_impl<ItemsPerBlock>(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            storage,
            compare_function,
            thread_keys,
            thread_values);
        thread_key   = thread_keys[0];
        thread_value = thread_values[0];
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline void sort(Key (&thread_keys)[ItemsPerThread],
                                    Value (&thread_values)[ItemsPerThread],
                                    storage_type&  storage,
                                    BinaryFunction compare_function)
    {
        this->sort_impl<ItemsPerBlock>(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            storage,
            compare_function,
            thread_keys,
            thread_values);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
        sort(Key& thread_key, Value& thread_value, BinaryFunction compare_function)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->sort(thread_key, thread_value, storage, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void sort(Key (&thread_keys)[ItemsPerThread],
                                                  Value (&thread_values)[ItemsPerThread],
                                                  BinaryFunction compare_function)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->sort(thread_keys, thread_values, storage, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void sort(Key (&thread_keys)[ItemsPerThread],
                                                  storage_type&  storage,
                                                  unsigned int   size,
                                                  BinaryFunction compare_function)
    {
        this->sort_impl(::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
                        size,
                        storage,
                        compare_function,
                        thread_keys);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void sort(Key (&thread_keys)[ItemsPerThread],
                                                  Value (&thread_values)[ItemsPerThread],
                                                  storage_type&  storage,
                                                  unsigned int   size,
                                                  BinaryFunction compare_function)
    {
        this->sort_impl(::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
                        size,
                        storage,
                        compare_function,
                        thread_keys,
                        thread_values);
    }

private:
    ROCPRIM_DEVICE ROCPRIM_INLINE void
        copy_to_shared(Key& k, const unsigned int flat_tid, Key* keys_shared)
    {
        keys_shared[flat_tid] = k;
        ::rocprim::syncthreads();
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void
        copy_to_shared(Key (&k)[ItemsPerThread], const unsigned int flat_tid, Key* keys_shared)
    {
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; ++item)
        {
            keys_shared[ItemsPerThread * flat_tid + item] = k[item];
        }
        ::rocprim::syncthreads();
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void copy_to_shared(
        Key& k, Value& v, const unsigned int flat_tid, Key* keys_shared, Value* values_shared)
    {
        keys_shared[flat_tid]   = k;
        values_shared[flat_tid] = v;
        ::rocprim::syncthreads();
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void copy_to_shared(Key (&k)[ItemsPerThread],
                                                      Value (&v)[ItemsPerThread],
                                                      const unsigned int flat_tid,
                                                      Key*               keys_shared,
                                                      Value*             values_shared)
    {
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; ++item)
        {
            keys_shared[ItemsPerThread * flat_tid + item]   = k[item];
            values_shared[ItemsPerThread * flat_tid + item] = v[item];
        }
        ::rocprim::syncthreads();
    }

    template<unsigned int Size, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort_impl(const unsigned int flat_tid,
                                                 storage_type&      storage,
                                                 BinaryFunction     compare_function,
                                                 Key (&keys)[ItemsPerThread])
    {
        if(Size > ItemsPerBlock)
        {
            return;
        }
        warp_sort_type ws;
        ws.sort(keys, storage.warp_sort, compare_function);
        sort_merge_impl(flat_tid,
                        Size,
                        ItemsPerThread * WarpSortSize,
                        storage,
                        compare_function,
                        keys);
    }

    template<unsigned int Size, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort_impl(const unsigned int flat_tid,
                                                 storage_type&      storage,
                                                 BinaryFunction     compare_function,
                                                 Key (&keys)[ItemsPerThread],
                                                 Value (&values)[ItemsPerThread])
    {
        if(Size > ItemsPerBlock)
        {
            return;
        }
        warp_sort_type ws;
        ws.sort(keys, values, storage.warp_sort, compare_function);
        sort_merge_impl(flat_tid,
                        Size,
                        ItemsPerThread * WarpSortSize,
                        storage,
                        compare_function,
                        keys,
                        values);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort_impl(const unsigned int flat_tid,
                                                 const unsigned int input_size,
                                                 storage_type&      storage,
                                                 BinaryFunction     compare_function,
                                                 Key (&keys)[ItemsPerThread])
    {
        warp_sort_type ws;
        ws.sort(keys, storage.warp_sort, input_size, compare_function);
        sort_merge_impl(flat_tid,
                        input_size,
                        ItemsPerThread * WarpSortSize,
                        storage,
                        compare_function,
                        keys);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort_impl(const unsigned int flat_tid,
                                                 const unsigned int input_size,
                                                 storage_type&      storage,
                                                 BinaryFunction     compare_function,
                                                 Key (&keys)[ItemsPerThread],
                                                 Value (&values)[ItemsPerThread])
    {
        warp_sort_type ws;
        ws.sort(keys, values, storage.warp_sort, input_size, compare_function);
        sort_merge_impl(flat_tid,
                        input_size,
                        ItemsPerThread * WarpSortSize,
                        storage,
                        compare_function,
                        keys,
                        values);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort_merge_impl(const unsigned int flat_tid,
                                                       const unsigned int input_size,
                                                       unsigned int       sorted_block_size,
                                                       storage_type&      storage,
                                                       BinaryFunction     compare_function,
                                                       Key (&thread_keys)[ItemsPerThread])
    {
        const unsigned int thread_offset = flat_tid * ItemsPerThread;
        auto&              keys_shared   = storage.keys.get();

        if(ItemsPerThread == 1 && thread_offset > input_size)
            return;
        // loop as long as sorted_block_size < input_size
        while(sorted_block_size < input_size)
        {
            copy_to_shared(thread_keys, flat_tid, keys_shared);
            const unsigned int target_sorted_block_size = sorted_block_size * 2;
            const unsigned int mask                     = target_sorted_block_size - 1;
            const unsigned int keys1_beg                = ~mask & thread_offset;
            const unsigned int keys1_end   = std::min(input_size, keys1_beg + sorted_block_size);
            const unsigned int keys2_end   = std::min(input_size, keys1_end + sorted_block_size);
            sorted_block_size              = target_sorted_block_size;
            const unsigned int diag0_local = std::min(input_size, mask & thread_offset);

            const unsigned int num_keys1 = keys1_end - keys1_beg;
            const unsigned int num_keys2 = keys2_end - keys1_end;

            const unsigned int keys1_beg_local = merge_path(&keys_shared[keys1_beg],
                                                            &keys_shared[keys1_end],
                                                            num_keys1,
                                                            num_keys2,
                                                            diag0_local,
                                                            compare_function);
            const unsigned int keys2_beg_local = diag0_local - keys1_beg_local;
            range_t            range_local
                = {keys1_beg_local + keys1_beg, keys1_end, keys2_beg_local + keys1_end, keys2_end};

            serial_merge(keys_shared, thread_keys, range_local, compare_function);
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort_merge_impl(const unsigned int flat_tid,
                                                       const unsigned int input_size,
                                                       unsigned int       sorted_block_size,
                                                       storage_type&      storage,
                                                       BinaryFunction     compare_function,
                                                       Key (&thread_keys)[ItemsPerThread],
                                                       Value (&thread_values)[ItemsPerThread])
    {
        const unsigned int thread_offset = flat_tid * ItemsPerThread;
        auto&              keys_shared   = storage.keys.get();
        auto&              values_shared = storage.values.get();
        // loop as long as sorted_block_size < input_size
        while(sorted_block_size < input_size)
        {
            copy_to_shared(thread_keys, thread_values, flat_tid, keys_shared, values_shared);
            const unsigned int target_sorted_block_size = sorted_block_size * 2;
            const unsigned int mask                     = target_sorted_block_size - 1;
            const unsigned int keys1_beg                = ~mask & thread_offset;
            const unsigned int keys1_end   = std::min(input_size, keys1_beg + sorted_block_size);
            const unsigned int keys2_end   = std::min(input_size, keys1_end + sorted_block_size);
            sorted_block_size              = target_sorted_block_size;
            const unsigned int diag0_local = std::min(input_size, mask & thread_offset);

            const unsigned int num_keys1 = keys1_end - keys1_beg;
            const unsigned int num_keys2 = keys2_end - keys1_end;

            const unsigned int keys1_beg_local = merge_path(&keys_shared[keys1_beg],
                                                            &keys_shared[keys1_end],
                                                            num_keys1,
                                                            num_keys2,
                                                            diag0_local,
                                                            compare_function);
            const unsigned int keys2_beg_local = diag0_local - keys1_beg_local;
            range_t            range_local
                = {keys1_beg_local + keys1_beg, keys1_end, keys2_beg_local + keys1_end, keys2_end};

            serial_merge(keys_shared,
                         thread_keys,
                         values_shared,
                         thread_values,
                         range_local,
                         compare_function);
        }
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_SORT_MERGE_HPP_
