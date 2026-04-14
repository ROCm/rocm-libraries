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

#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_SORT_SHARED_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_SORT_SHARED_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"
#include "../../functional.hpp"
#include "../../intrinsics/arch.hpp"
#include "../../intrinsics/thread.hpp"
#include "../../warp/detail/warp_sort_shuffle.hpp"
#include "rocprim/types.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<int BlockSize, int VirtualWaveSize, int ItemsPerThread>
struct block_sort_bitonic_impl
{
    // Actual bitonic sorting networking implementation.

    static_assert(::rocprim::detail::is_power_of_two(BlockSize));
    static_assert(::rocprim::detail::is_power_of_two(VirtualWaveSize));
    static_assert(::rocprim::detail::is_power_of_two(ItemsPerThread));

    // Use warp-level bitonic sort for its warp compare-and-swap,
    // and thread sort implementation.
    using wlev = ::rocprim::detail::warp_shuffle_sort_impl<VirtualWaveSize, ItemsPerThread>;

    static constexpr int num_wave_bits = Log2<VirtualWaveSize>::VALUE;
    static constexpr int num_id_bits   = Log2<BlockSize>::VALUE;

    // Define block-level compare-and-swap. This family of functions compares
    // keys between threads and swaps the smallest key (and value) in the specified
    // direction.
    template<class Storage, class K, class V, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void blev_cas(unsigned int   id,
                         Storage&       storage,
                         bool           dir,
                         BinaryFunction compare_function,
                         unsigned int   xor_mask,
                         K (&k)[ItemsPerThread],
                         V (&v)[ItemsPerThread])
    {
        ROCPRIM_UNROLL
        for(unsigned int i = 0u; i < ItemsPerThread; ++i)
        {
            storage.key[id + (i * BlockSize)]   = k[i];
            storage.value[id + (i * BlockSize)] = v[i];
        }

        ::rocprim::syncthreads();

        ROCPRIM_UNROLL
        for(unsigned int i = 0u; i < ItemsPerThread; ++i)
        {
            const auto i1   = (id ^ xor_mask) + (i * BlockSize);
            const K&   k0   = k[i];
            const K    k1   = storage.key[i1];
            const bool swap = compare_function(dir ? k0 : k1, dir ? k1 : k0);
            if(swap)
            {
                k[i] = k1;
                v[i] = storage.value[i1];
            }
        }
    }

    template<class Storage, class K, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void blev_cas(unsigned int   id,
                         Storage&       storage,
                         bool           dir,
                         BinaryFunction compare_function,
                         unsigned int   xor_mask,
                         K (&k)[ItemsPerThread])
    {
        ROCPRIM_UNROLL
        for(unsigned int i = 0u; i < ItemsPerThread; ++i)
        {
            storage.key[id + (i * BlockSize)] = k[i];
        }

        ::rocprim::syncthreads();

        ROCPRIM_UNROLL
        for(unsigned int i = 0u; i < ItemsPerThread; ++i)
        {
            const K&   k0   = k[i];
            const auto i1   = (id ^ xor_mask) + (i * BlockSize);
            const K    k1   = storage.key[i1];
            const bool swap = compare_function(dir ? k0 : k1, dir ? k1 : k0);
            k[i]            = swap ? k1 : k0;
        }
    }

    template<class Storage, class K, class V, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static auto blev_cas(unsigned int   id,
                         Storage&       storage,
                         bool           dir,
                         BinaryFunction compare_function,
                         unsigned int   xor_mask,
                         K&             k,
                         V&             v)
    {
        storage.key[id]   = k;
        storage.value[id] = v;

        ::rocprim::syncthreads();

        const auto i1   = id ^ xor_mask;
        const K    k1   = storage.key[i1];
        const bool swap = compare_function(dir ? k : k1, dir ? k1 : k);
        if(swap)
        {
            k = k1;
            v = storage.value[i1];
        }
    }

    template<class Storage, class K, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void blev_cas(unsigned int   id,
                         Storage&       storage,
                         bool           dir,
                         BinaryFunction compare_function,
                         unsigned int   xor_mask,
                         K&             k)
    {
        storage.key[id] = k;

        ::rocprim::syncthreads();

        const auto i1   = id ^ xor_mask;
        const K    k1   = storage.key[i1];
        const bool swap = compare_function(dir ? k : k1, dir ? k1 : k);
        k               = swap ? k1 : k;
    }

    template<class Storage, class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void bitonic_sort(Storage&           storage,
                             const unsigned int flat_tid,
                             BinaryFunction     compare_function,
                             KeyValue&... kv)
    {
        // Unpack flat_tid into bits to promote better unrolling.
        unsigned int id_bits[num_id_bits + 1];

        // for(int i = 0; i < num_id_bits; ++i)
        ::rocprim::detail::constexpr_for_lt<0, num_id_bits, 1>(
            [&](auto i) { id_bits[i] = __builtin_amdgcn_ubfe(flat_tid, i, 1u); });
        id_bits[num_id_bits] = 0u;

        // We have folded the implementation of:
        //   wlev::bitonic_sort(compare_function, kv...);
        // into the rest of the block-level algorithm. We cannot simply
        // call the wave-level sort because for the bitonic property to
        // presist, we need to reverse the sorting direction of all odd
        // warps.
        //
        // The trick is that the wave-level implementation already exists
        // in the block-level implementation. We simply must disable the
        // block-level logic, which already happens implicitely due to the
        // loop conditions.
        //
        // For a more in-depth explenation of this algorithm, please refer
        // to 'warp_shuffle_sort_impl::bitonic_sort(...)'.

        // Thread level sort.
        if constexpr(ItemsPerThread > 1)
        {
            wlev::tlev_sort(id_bits[0], compare_function, kv...);
        }

        // Warp & block level sort.
        // for (int group_bit = num_wave_bits + 1; i <= num_id_bits; ++group_bit)
        ::rocprim::detail::constexpr_for_lte<1, num_id_bits, 1>(
            [&](auto group_bit)
            {
                // The first group bit that requires block-level cooperation.
                constexpr bool is_first_block_group_bit = group_bit == (num_wave_bits + 1);

                // Apply block-level compare and swaps.
                // for (int offset_bit = group_bit - 1; group_bit >= num_wave_bits; --offset_bit)
                ::rocprim::detail::constexpr_for_gte<group_bit - 1, num_wave_bits, -1>(
                    [&](auto offset_bit)
                    {
                        constexpr bool         is_first_offset_bit = offset_bit == group_bit - 1;
                        constexpr unsigned int offset              = 1u << offset_bit;
                        const unsigned int     local_dir = id_bits[group_bit] ^ id_bits[offset_bit];

                        // Assume that shared storage is ready to write on the very first
                        // 'blev_cas'-invocation. So, we skip if we're on the first group
                        // and first offset.
                        constexpr bool skip_block_sync
                            = is_first_block_group_bit && is_first_offset_bit;
                        if constexpr(!skip_block_sync)
                        {
                            // Ensure that all threads have consumed data from shared
                            // storage!
                            ::rocprim::syncthreads();
                        }

                        static_assert(offset >= VirtualWaveSize);
                        blev_cas(flat_tid, storage, local_dir, compare_function, offset, kv...);
                    });

                // 'group_bit' may be smaller than 'num_wave_bits'. We must make sure
                // that the next loop starts at which ever is smaller.
                constexpr int wave_group_bit = std::min(static_cast<int>(group_bit), num_wave_bits);

                // Apply wavefront-level compare and swaps.
                // for (int offset_bit = wave_group_bit - 1; group_bit >= 0; --offset_bit)
                ::rocprim::detail::constexpr_for_gte<wave_group_bit - 1, 0, -1>(
                    [&](auto offset_bit)
                    {
                        constexpr unsigned int offset    = 1u << offset_bit;
                        const unsigned int     local_dir = id_bits[group_bit] ^ id_bits[offset_bit];

                        // Enable packing since we have enough work to hide hazards from
                        // utilizing vcc. Packing allows the compiler to pack multiple items
                        // per thread into a singular abstraction, and then decide by it self
                        // how the data depedencies should resolve.
                        //
                        // In practice this results in more freedom for the compiler with more
                        // relaxed local data sharing.
                        constexpr bool try_pack = true;

                        static_assert(offset < VirtualWaveSize);
                        wlev::template wlev_cas<try_pack>(local_dir,
                                                          compare_function,
                                                          offset,
                                                          kv...);
                    });

                // Apply thread-level compare and swaps.
                if constexpr(ItemsPerThread > 1)
                {
                    wlev::tlev_pass(id_bits[group_bit], compare_function, kv...);
                }
            });
    }
};

template<bool SizeCheck, int BlockSize, int VirtualWaveSize, int ItemsPerThread>
struct block_sort_oddeven_impl
{
    // Odd-even sorting network implementation. This is
    // used as a fallback when BS and IPT are not powers
    // of two.

    static constexpr int ItemsPerBlock = BlockSize * ItemsPerThread;

    template<class Storage, class Key, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void swap_oddeven(Key&               key,
                             const unsigned int next_id,
                             const unsigned int /* item */,
                             const unsigned int next_item_id,
                             bool               dir,
                             Storage&           storage,
                             BinaryFunction     compare_function)
    {
        Key next_key = storage.key[(next_item_id * BlockSize) + next_id];

        bool swap = compare_function(dir ? key : next_key, dir ? next_key : key);
        key       = swap ? next_key : key;
    }

    template<class Storage, class Key, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void swap_oddeven(Key (&keys)[ItemsPerThread],
                             const unsigned int next_id,
                             const unsigned int item,
                             const unsigned int next_item_id,
                             bool               dir,
                             Storage&           storage,
                             BinaryFunction     compare_function)
    {
        Key next_key = storage.key[(next_item_id * BlockSize) + next_id];

        bool swap  = compare_function(dir ? keys[item] : next_key, dir ? next_key : keys[item]);
        keys[item] = swap ? next_key : keys[item];
    }

    template<class Storage, class Key, class Value, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void swap_oddeven(Key&               key,
                             Value&             value,
                             const unsigned int next_id,
                             const unsigned int /* item */,
                             const unsigned int next_item_id,
                             bool               dir,
                             Storage&           storage,
                             BinaryFunction     compare_function)
    {

        Key next_key = storage.key[(next_item_id * BlockSize) + next_id];

        bool swap = compare_function(dir ? key : next_key, dir ? next_key : key);
        key       = swap ? next_key : key;
        value     = swap ? storage.value[(next_item_id * BlockSize) + next_id] : value;
    }

    template<class Storage, class Key, class Value, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void swap_oddeven(Key (&keys)[ItemsPerThread],
                             Value (&values)[ItemsPerThread],
                             const unsigned int next_id,
                             const unsigned int item,
                             const unsigned int next_item_id,
                             bool               dir,
                             Storage&           storage,
                             BinaryFunction     compare_function)
    {
        Key next_key = storage.key[(next_item_id * BlockSize) + next_id];

        bool swap    = compare_function(dir ? keys[item] : next_key, dir ? next_key : keys[item]);
        keys[item]   = swap ? next_key : keys[item];
        values[item] = swap ? storage.value[(next_item_id * BlockSize) + next_id] : values[item];
    }

    template<class Storage, class Key>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void copy_to_shared(Storage& storage, const unsigned int flat_tid, Key& k)
    {
        storage.key[flat_tid] = k;
        ::rocprim::syncthreads();
    }

    template<class Storage, class Key>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void
        copy_to_shared(Storage& storage, const unsigned int flat_tid, Key (&k)[ItemsPerThread])
    {
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; ++item)
        {
            storage.key[(item * BlockSize) + flat_tid] = k[item];
        }
        ::rocprim::syncthreads();
    }

    template<class Storage, class Key, class Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void copy_to_shared(Storage& storage, const unsigned int flat_tid, Key& k, Value& v)
    {
        storage.key[flat_tid]   = k;
        storage.value[flat_tid] = v;
        ::rocprim::syncthreads();
    }

    template<class Storage, class Key, class Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void copy_to_shared(Storage&           storage,
                               const unsigned int flat_tid,
                               Key (&k)[ItemsPerThread],
                               Value (&v)[ItemsPerThread])
    {
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; ++item)
        {
            storage.key[(item * BlockSize) + flat_tid]   = k[item];
            storage.value[(item * BlockSize) + flat_tid] = v[item];
        }
        ::rocprim::syncthreads();
    }

    template<class Storage, class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void oddeven_sort(Storage&           storage,
                             const unsigned int flat_tid,
                             BinaryFunction     compare_function,
                             const unsigned int size,
                             KeyValue&... kv)
    {
        static constexpr unsigned int PairSize = sizeof...(KeyValue);
        static_assert(PairSize < 3,
                      "KeyValue parameter pack can be 1 or 2 elements (key, or key and value)");

        if(SizeCheck && size > ItemsPerBlock)
        {
            return;
        }

        copy_to_shared(storage, flat_tid, kv...);

        for(unsigned int i = 0; i < size; i++)
        {
            bool is_even_iter = i % 2 == 0;
            for(unsigned int item = 0; item < ItemsPerThread; ++item)
            {
                // the element in the original array that key[item] represents
                unsigned int linear_id   = (flat_tid * ItemsPerThread) + item;
                bool         is_even_lid = linear_id % 2 == 0;

                // one up/down from the linear_id
                unsigned int odd_lid  = is_even_lid ? ::rocprim::max(linear_id, 1u) - 1
                                                    : ::rocprim::min(linear_id + 1, size - 1);
                unsigned int even_lid = is_even_lid ? ::rocprim::min(linear_id + 1, size - 1)
                                                    : ::rocprim::max(linear_id, 1u) - 1;

                // determine if the odd or even index must be used
                unsigned int next_lid = is_even_iter ? even_lid : odd_lid;

                // map the linear_id back to item and thread id for indexing shared memory
                unsigned int next_id      = next_lid / ItemsPerThread;
                unsigned int next_item_id = next_lid % ItemsPerThread;

                // prevent calling the compare function with out-of-bounds items
                if(!SizeCheck || (linear_id < size && next_lid < size))
                {
                    swap_oddeven(kv...,
                                 next_id,
                                 item,
                                 next_item_id,
                                 next_lid < linear_id,
                                 storage,
                                 compare_function);
                }
            }
            ::rocprim::syncthreads();
            copy_to_shared(storage, flat_tid, kv...);
        }
    }
};

template<class Key,
         unsigned int BlockSizeX,
         unsigned int BlockSizeY,
         unsigned int BlockSizeZ,
         unsigned int ItemsPerThread,
         class Value>
class block_sort_bitonic
{
    static constexpr unsigned int BlockSize     = BlockSizeX * BlockSizeY * BlockSizeZ;
    static constexpr unsigned int ItemsPerBlock = BlockSize * ItemsPerThread;

    // If the value type is large, we're better of using indices
    // for this and gather and scatter through LDS to retrieve them.
    static constexpr bool use_index = sizeof(Value) > 4;
    using IndexType                 = std::conditional_t<use_index, unsigned int, empty_type>;

    template<class KeyType, class ValueType, class IndexType = IndexType>
    union storage_type_
    {
        // Instead of sorting keys and values, we'll sort keys and.
        // indices. We'll use the same shared memory buffer to gather
        // the values once we know at what index we need to fetch.
        struct
        {
            KeyType   key[BlockSize * ItemsPerThread];
            IndexType value[BlockSize * ItemsPerThread];
        } kv;

        ValueType value[BlockSize * ItemsPerThread];
    };

    template<class KeyType, class ValueType>
    union storage_type_<KeyType, ValueType, empty_type>
    {
        struct
        {
            KeyType   key[BlockSize * ItemsPerThread];
            ValueType value[BlockSize * ItemsPerThread];
        } kv;
    };

    template<class KeyType>
    union storage_type_<KeyType, empty_type, empty_type>
    {
        struct
        {
            KeyType key[BlockSize * ItemsPerThread];
        } kv;
    };

public:
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH
    using storage_type = detail::raw_storage<storage_type_<Key, Value>>;
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP

private:
    template<class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void sort_impl(const unsigned int flat_tid,
                          storage_type&      storage,
                          BinaryFunction     compare_function,
                          KeyValue&... kv)
    {
        static constexpr unsigned int PairSize = sizeof...(KeyValue);
        static_assert(PairSize < 3,
                      "KeyValue parameter pack can be 1 or 2 elements (key, or key and value)");

        static constexpr bool use_bitonic
            = is_power_of_two(BlockSize) && is_power_of_two(ItemsPerThread);
        static constexpr bool use_wlev = BlockSize <= ::rocprim::arch::wavefront::min_size();

        if constexpr(use_bitonic)
        {
            if constexpr(use_wlev)
            {
                // On small enough problems we can just invoke warp sort!
                using wlev = warp_shuffle_sort_impl<BlockSize, ItemsPerThread>;
                wlev::bitonic_sort(compare_function, kv...);
            }
            else
            {
                // On bigger problems, we must use block level bitonic sort!
                using blev
                    = block_sort_bitonic_impl<BlockSize,
                                              std::min(BlockSize,
                                                       ::rocprim::arch::wavefront::min_size()),
                                              ItemsPerThread>;
                blev::bitonic_sort(storage.get().kv, flat_tid, compare_function, kv...);
            }
        }
        else
        {
            // Fallback to odd-even sort.
            using blev = block_sort_oddeven_impl<false,
                                                 BlockSize,
                                                 ::rocprim::arch::wavefront::min_size(),
                                                 ItemsPerThread>;

            constexpr int size = ItemsPerBlock;
            blev::oddeven_sort(storage.get().kv, flat_tid, compare_function, size, kv...);
        }
    }

    template<class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void sort_impl(const unsigned int flat_tid,
                          const unsigned int size,
                          storage_type&      storage,
                          BinaryFunction     compare_function,
                          KeyValue&... kv)
    {
        using blev = block_sort_oddeven_impl<false,
                                             BlockSize,
                                             ::rocprim::arch::wavefront::min_size(),
                                             ItemsPerThread>;
        blev::oddeven_sort(storage.get().kv, flat_tid, compare_function, size, kv...);
    }

public:
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key& thread_key, storage_type& storage, BinaryFunction compare_function)
    {
        sort_impl(::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
                  storage,
                  compare_function,
                  thread_key);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              storage_type&  storage,
              BinaryFunction compare_function)
    {
        sort_impl(::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
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
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key&           thread_key,
              Value&         thread_value,
              storage_type&  storage,
              BinaryFunction compare_function)
    {
        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        if constexpr(use_index)
        {
            // Sort value indices instead of the values.
            IndexType thread_index = flat_tid * ItemsPerThread;
            sort_impl(flat_tid, storage, compare_function, thread_key, thread_index);
            syncthreads();

            // Store values in LDS.
            storage.get().value[flat_tid * ItemsPerThread] = thread_value;
            syncthreads();

            // Use the indices to gather values from LDS.
            thread_value = storage.get().value[thread_index];
        }
        else
        {
            sort_impl(flat_tid, storage, compare_function, thread_key, thread_value);
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              storage_type&  storage,
              BinaryFunction compare_function)
    {
        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        if constexpr(use_index)
        {
            // Sort value indices instead of the values.
            IndexType thread_indices[ItemsPerThread];
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                thread_indices[i] = (flat_tid * ItemsPerThread) + i;
            }
            sort_impl(flat_tid, storage, compare_function, thread_keys, thread_indices);
            syncthreads();

            // Store values in LDS.
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                storage.get().value[(flat_tid * ItemsPerThread) + i] = thread_values[i];
            }
            syncthreads();

            // Use the indices to gather values from LDS.
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                thread_values[i] = storage.get().value[thread_indices[i]];
            }
        }
        else
        {
            sort_impl(flat_tid, storage, compare_function, thread_keys, thread_values);
        }
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
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key&               thread_key,
              storage_type&      storage,
              const unsigned int size,
              BinaryFunction     compare_function)
    {
        this->sort_impl(::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
                        size,
                        storage,
                        compare_function,
                        thread_key);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              storage_type&      storage,
              const unsigned int size,
              BinaryFunction     compare_function)
    {
        this->sort_impl(::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
                        size,
                        storage,
                        compare_function,
                        thread_keys);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key&               thread_key,
              Value&             thread_value,
              storage_type&      storage,
              const unsigned int size,
              BinaryFunction     compare_function)
    {
        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        if constexpr(use_index)
        {
            // Sort value indices instead of the values.
            IndexType thread_index = flat_tid * ItemsPerThread;
            sort_impl(flat_tid, size, storage, compare_function, thread_key, thread_index);
            syncthreads();

            // Store values in LDS.
            storage.get().value[flat_tid * ItemsPerThread] = thread_value;
            syncthreads();

            // Use the indices to gather values from LDS.
            thread_value = storage.get().value[thread_index];
        }
        else
        {
            this->sort_impl(flat_tid, size, storage, compare_function, thread_key, thread_value);
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              storage_type&      storage,
              const unsigned int size,
              BinaryFunction     compare_function)
    {
        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        if constexpr(use_index)
        {
            // Sort value indices instead of the values.
            IndexType thread_indices[ItemsPerThread];
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                thread_indices[i] = (flat_tid * ItemsPerThread) + i;
            }
            sort_impl(flat_tid, size, storage, compare_function, thread_keys, thread_indices);
            syncthreads();

            // Store values in LDS.
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                storage.get().value[(flat_tid * ItemsPerThread) + i] = thread_values[i];
            }
            syncthreads();

            // Use the indices to gather values from LDS.
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                thread_values[i] = storage.get().value[thread_indices[i]];
            }
        }
        else
        {
            this->sort_impl(flat_tid, size, storage, compare_function, thread_keys, thread_values);
        }
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_SORT_SHARED_HPP_
