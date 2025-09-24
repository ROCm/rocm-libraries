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

#ifndef ROCPRIM_WARP_DETAIL_WARP_SORT_SHUFFLE_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_SORT_SHUFFLE_HPP_

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../functional.hpp"
#include "../../intrinsics/thread.hpp"
#include "../../intrinsics/warp_shuffle.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
// This is actually bitonic sort.
//
// We have two layers here:
// * warp_shuffle_sort_impl,
// * warp_sort_shuffle.
//
// 'warp_shuffle_sort_impl' contains the main bitonic implementation.
// It handles the various compare and swaps, and also handles cross-
// lane operations. Micro-optimization (i.e. hardware intrinsics)
// should be done here.
//
// 'warp_sort_shuffle' exposes the higher level API. It handles key-
// value pairs. Algorithmic optimizations (i.e. shuffling indexes for
// values) should be done here.

template<unsigned int VirtualWaveSize>
struct warp_shuffle_sort_impl
{
    // Since we have to apply bitonic sort of potentially two levels
    // we use the following nomenclature:
    // * 'tlev' for thread-level operations,
    // * 'wlev' for warp/wavefront-level operations.
    //
    // Please look at 'bitonic_sort(...)' for the main algorithm.

    /// Swaps `v` between lanes according to a `xor_mask`.
    template<class V>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static auto lane_xor_swap(V& v, const int xor_mask)
    {
        // This function is here so we can more easily experiment and add DPP in the future.
        // The related DPP invocations are:
        //   warp_move_dpp<V, 0b10'11'00'01>(v); // if xor_mask == 0b01
        //   warp_move_dpp<V, 0b01'00'11'10>(v); // if xor_mask == 0b10
        // However, more experimentation is required to get this to work well since from
        // preliminary testing using DPP is slower.
        return warp_swizzle_shuffle(v, xor_mask, VirtualWaveSize);
    }

    /// Compares the value between pairs of lanes, where the pairs are selected via
    /// a XOR-masks. Then sorts the keys and values according to the passed direction.
    template<class K, class V, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void
        wlev_cas(bool dir, BinaryFunction compare_function, unsigned int xor_mask, K& k, V& v)
    {
        const K    k1   = lane_xor_swap(k, xor_mask);
        const bool swap = compare_function(dir ? k : k1, dir ? k1 : k);
        if(swap)
        {
            k = k1;
            v = lane_xor_swap(v, xor_mask);
        }
    }

    /// Compares multiple values between pairs of lanes, where the pairs are selected via
    /// a XOR-masks. Then sorts the keys and values according to the passed direction.
    template<class K, class V, class BinaryFunction, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void wlev_cas(bool           dir,
                         BinaryFunction compare_function,
                         unsigned int   xor_mask,
                         K (&k)[ItemsPerThread],
                         V (&v)[ItemsPerThread])
    {
        // for(unsigned int item = 0; item < ItemsPerThread; item++)
        ::rocprim::detail::constexpr_for_lt<0, ItemsPerThread, 1>(
            [&](auto item)
            {
                const K&   k0   = k[item];
                const V&   v0   = v[item];
                const K    k1   = lane_xor_swap(k0, xor_mask);
                const bool swap = compare_function(dir ? k0 : k1, dir ? k1 : k0);

                // There are two main ways of conditionally storing the result of 'k1'
                // and 'v1'. Semanticly, these are the same, but the compiler will treat
                // them differently.
                //
                // Using ternary will push the compiler to use 'v_cndmask' instructions
                // which are branchless. This comes at the cost of having a guaranteed
                // invocation of 'lane_xor_swap'.
                //
                // Moving the 'lane_xor_swap' into a branch will make allows us to saves
                // us the extra condition moves since we can swizzle in-place for values.
                if constexpr(sizeof(K) >= 16 || (ItemsPerThread > 1 && sizeof(K) <= 1))
                {
                    const V v1 = lane_xor_swap(v0, xor_mask);

                    k[item] = swap ? k1 : k0;
                    v[item] = swap ? v1 : v0;
                }
                else
                {
                    if(swap)
                    {
                        k[item] = k1;
                        v[item] = lane_xor_swap(v0, xor_mask);
                    }
                }
            });
    }

    /// Compares the value between pairs of lanes, where the pairs are selected via
    /// a XOR-masks. Then sorts the keys according to the passed direction.
    template<class K, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void wlev_cas(bool dir, BinaryFunction compare_function, unsigned int xor_mask, K& k)
    {
        const K    k1   = lane_xor_swap(k, xor_mask);
        const bool swap = compare_function(dir ? k : k1, dir ? k1 : k);
        k               = swap ? k1 : k;
    }

    /// Compares multiple values between pairs of lanes, where the pairs are selected via
    /// a XOR-masks. Then sorts the keys according to the passed direction.
    template<class K, class BinaryFunction, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void wlev_cas(bool           dir,
                         BinaryFunction compare_function,
                         unsigned int   xor_mask,
                         K (&k)[ItemsPerThread])
    {
        // for(unsigned int item = 0; item < ItemsPerThread; item++)
        ::rocprim::detail::constexpr_for_lt<0, ItemsPerThread, 1>(
            [&](auto item)
            {
                const K    k1   = lane_xor_swap(k[item], xor_mask);
                const bool swap = compare_function(dir ? k[item] : k1, dir ? k1 : k[item]);
                k[item]         = swap ? k1 : k[item];
            });
    }

    template<int i_l, int i_r, int ItemsPerThread, class BinaryFunction, class K>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void tlev_cas_single(const bool&    local_dir,
                                BinaryFunction compare_function,
                                K (&k)[ItemsPerThread])
    {
        const bool swap = compare_function(k[i_l], k[i_r]) == local_dir;
        // We need copies if we're going to use ternaries.
        const K k_l = k[i_l];
        const K k_r = k[i_r];
        // Using ternary is much faster than 'if(cond) rocprim::swap(...)'.
        k[i_l] = swap ? k_r : k_l;
        k[i_r] = swap ? k_l : k_r;
    }

    template<int i_l, int i_r, int ItemsPerThread, class BinaryFunction, class K, class V>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void tlev_cas_single(const bool&    local_dir,
                                BinaryFunction compare_function,
                                K (&k)[ItemsPerThread],
                                V (&v)[ItemsPerThread])
    {
        const bool swap = compare_function(k[i_l], k[i_r]) == local_dir;
        // We need copies if we're going to use ternaries.
        const K k_l = k[i_l];
        const K k_r = k[i_r];
        const V v_l = v[i_l];
        const V v_r = v[i_r];
        // Using ternary is much faster than 'if(cond) rocprim::swap(...)'.
        k[i_l] = swap ? k_r : k_l;
        k[i_r] = swap ? k_l : k_r;
        v[i_l] = swap ? v_r : v_l;
        v[i_r] = swap ? v_l : v_r;
    }

    /// Applies the thread-level compare and swaps.
    template<unsigned int ItemsPerThread,
             unsigned int group_size,
             unsigned int offset,
             class BinaryFunction,
             class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void tlev_cas(unsigned int group_dir, BinaryFunction compare_function, KeyValue&... kv)
    {
        // Note: we're storing 'group_dir' as unsigned int s.t. the compiler is more
        // inclined to re-use the results from '__builtin_amdgcn_ubfe'.

        // for(unsigned int base = 0; base < ItemsPerThread; base += 2 * offset)
        ::rocprim::detail::constexpr_for_lt<0, ItemsPerThread, 2 * offset>(
            [&](auto base)
            {
                // The local direction must change every group_size items
                // and is flipped if dir is true
                const bool local_dir = (base & group_size) > 0 != group_dir;

                // for(unsigned i = 0; i < offset; ++i)
                ::rocprim::detail::constexpr_for_lt<0, offset, 1>(
                    [&](auto i)
                    {
                        constexpr unsigned int i_l = base + i;
                        constexpr unsigned int i_r = base + i + offset;
                        tlev_cas_single<i_l, i_r>(local_dir, compare_function, kv...);
                    });
            });
    }

    template<unsigned int ItemsPerThread,
             class BinaryFunction,
             int group_size = ItemsPerThread,
             int offset     = group_size / 2,
             class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void tlev_pass(unsigned int group_dir, BinaryFunction compare_function, KeyValue&... kv)
    {
        // Note: we're storing 'group_dir' as unsigned int s.t. the compiler is more
        // inclined to re-use the results from '__builtin_amdgcn_ubfe'.

        // Implement the following loop using recursion:
        //   for(unsigned int offset = group_size / 2; offset > 0; offset /= 2)
        if constexpr(offset > 0)
        {
            tlev_cas<ItemsPerThread, group_size, offset>(group_dir, compare_function, kv...);
            // Recurse...
            tlev_pass<ItemsPerThread, BinaryFunction, group_size, offset / 2>(group_dir,
                                                                              compare_function,
                                                                              kv...);
        }
    }

    template<unsigned int ItemsPerThread,
             class BinaryFunction,
             int group_size = 2,
             class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void tlev_sort(unsigned int group_dir, BinaryFunction compare_function, KeyValue&... kv)
    {
        // Note: we're storing 'group_dir' as unsigned int s.t. the compiler is more
        // inclined to re-use the results from '__builtin_amdgcn_ubfe'.

        // Implement the following loop using recursion:
        //   for(unsigned int group_size = 2; group_size <= ItemsPerThread; group_size *= 2)
        if constexpr(group_size <= ItemsPerThread)
        {
            tlev_pass<ItemsPerThread, BinaryFunction, group_size>(group_dir,
                                                                  compare_function,
                                                                  kv...);
            // Recurse...
            tlev_sort<ItemsPerThread, BinaryFunction, group_size * 2, KeyValue...>(group_dir,
                                                                                   compare_function,
                                                                                   kv...);
        }
    }

    /// High level bitonic sort. Handles both single item per threads and multiple.
    template<unsigned int ItemsPerThread, class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void bitonic_sort(BinaryFunction compare_function, KeyValue&... kv)
    {
        // !!! Preamble and boiler plate !!!

        static_assert(sizeof...(KeyValue) < 3,
                      "KeyValue parameter pack can 1 or 2 elements (key, or key and value)");
        static_assert(detail::is_power_of_two(ItemsPerThread), "ItemsPerThread must be power of 2");

        const unsigned int id = detail::logical_lane_id<VirtualWaveSize>();

        // We need to invoke '(id >> i) & 1u' a lot, so let's just nudge the
        // compiler that we can really easily precompute this using bit-field
        // extract.
        constexpr int num_id_bits = Log2<VirtualWaveSize>::VALUE;
        unsigned int  id_bits[num_id_bits + 1];

        // for(int i = 0; i < num_id_bits; ++i)
        ::rocprim::detail::constexpr_for_lt<0, num_id_bits, 1>(
            [&](auto i)
            {
                // Use unsigned bitfield extract to select the i-th bit from id.
                id_bits[i] = __builtin_amdgcn_ubfe(id, i, 1u);
            });
        // This will get optimized out, just needed to make the loop work below.
        id_bits[num_id_bits] = 0u;

        // !!! Actually start sorting now !!!
        //
        // Bitonic works by recursively getting larger and larger bitonic sequences.
        // We have to therefore start with the smallest sequence: thread-level items!
        //
        // We will use the following nomenclature:
        // * pass: sorts a bitonic sequence into a monotonic sequence
        // * sort: sorts any sequence into a monotonic sequence
        // E.g. tlev_sort: ensures all thread level items are sorted.
        //
        // First we need to make sure that we have bitonic lane pairs. Only need to 
        // invoke thread-level algorithms if we have multiple items per thread.
        if constexpr(ItemsPerThread > 1)
        {
            tlev_sort<ItemsPerThread>(id_bits[0], compare_function, kv...);
        }

        // Now we have bitonic sequences over our thread-level items, we need to
        // cooperatively create bigger sequences over more and more lanes.
        //
        // for(int group_bit = 1; (1 << group_bit) < VirtualWaveSize; ++group_bit)
        ::rocprim::detail::constexpr_for_lte<1, num_id_bits, 1>(
            [&](auto group_bit)
            {
                // Each iteration here combines a bitonic sequence of '1 << group_bit'
                // elements. I.e. the first iteration (group_bit = 1) we have 2 elements
                // in our bitonic sequence. This bit also indicate the sort direction s.t.
                // we produce a bitonic sequence of double the size (i.e. group_bit = 2
                // has 4 elements in the bitonic sequence).
                //
                // Example:
                //   /\/\ (1) 2 bitonic sequences
                //     -- This part is sorted in reverse!
                //   //\\ (2) 1 bitonic sequence
                //        No part is reversed! This is using 'id_bits[num_id_bits] = 0u'
                //   //// (3) 1 monotonic sequence

                // for(int pass_bit = group_bit - 1; pass_bit >= 0; --pass_bit)
                ::rocprim::detail::constexpr_for_gte<group_bit - 1, 0, -1>(
                    [&](auto pass_bit)
                    {
                        // The pass_bit indicates which direction this lane should sort. For
                        // example if lane 0 and 1 need to exchange items, then lane 1 needs
                        // to use the reverse direction of self vs other comparison.
                        // E.g. lane 0: v[0] (self ) < v[1] (other)
                        //      lane 1: v[0] (other) < v[1] (self )
                        const unsigned int pass_mask = 1u << pass_bit;
                        const unsigned int pass_dir  = id_bits[group_bit] ^ id_bits[pass_bit];
                        wlev_cas(pass_dir, compare_function, pass_mask, kv...);
                    });

                // Don't forget that we need to also do a pass (not a full sort) over
                // thread level items. Only need to invoke thread-level algorithms if we have multiple
                // items per thread.
                if constexpr(ItemsPerThread > 1)
                {
                    tlev_pass<ItemsPerThread>(id_bits[group_bit], compare_function, kv...);
                }
            });
    }
};

template<class Key, unsigned int VirtualWaveSize, class Value>
struct warp_sort_shuffle
{
    using impl = warp_shuffle_sort_impl<VirtualWaveSize>;

public:
    static_assert(detail::is_power_of_two(VirtualWaveSize), "VirtualWaveSize must be power of 2");

    using storage_type = ::rocprim::detail::empty_storage_type;

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key& thread_value, BinaryFunction compare_function)
    {
        // sort by value only
        impl::template bitonic_sort<1, BinaryFunction, Key&>(compare_function, thread_value);
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
    void sort(Key (&thread_values)[ItemsPerThread], BinaryFunction compare_function)
    {
        // sort by value only
        impl::template bitonic_sort<ItemsPerThread>(compare_function, thread_values);
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

    template<class BinaryFunction, class V = Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key& thread_key, Value& thread_value, BinaryFunction compare_function)
    {
        if(sizeof(V) <= sizeof(int))
        {
            impl::template bitonic_sort<1>(compare_function, thread_key, thread_value);
        }
        else
        {
            // Instead of passing large values between lanes we pass indices and gather values after sorting.
            unsigned int v = detail::logical_lane_id<VirtualWaveSize>();
            impl::template bitonic_sort<1>(compare_function, thread_key, v);
            thread_value = warp_shuffle(thread_value, v, VirtualWaveSize);
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key&           thread_key,
              Value&         thread_value,
              storage_type&  storage,
              BinaryFunction compare_function)
    {
        (void)storage;
        sort(compare_function, thread_key, thread_value);
    }

    template<unsigned int ItemsPerThread, class BinaryFunction, class V = Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              BinaryFunction compare_function)
    {
        if(sizeof(V) <= sizeof(int))
        {
            impl::template bitonic_sort<ItemsPerThread>(compare_function,
                                                        thread_keys,
                                                        thread_values);
        }
        else
        {
            // Instead of passing large values between lanes we pass indices and gather values after sorting.
            unsigned int index[ItemsPerThread];
            ROCPRIM_UNROLL
            for(unsigned int item = 0; item < ItemsPerThread; item++)
            {
                index[item] = ItemsPerThread * detail::logical_lane_id<VirtualWaveSize>() + item;
            }

            impl::template bitonic_sort<ItemsPerThread>(compare_function, thread_keys, index);

            // Create a copy of 'thread_values' so we can swizzle them around without overwriting.
            V copy[ItemsPerThread];
            ROCPRIM_UNROLL
            for(unsigned item = 0; item < ItemsPerThread; ++item)
            {
                copy[item] = thread_values[item];
            }

            // We will now write into 'thread_values' from 'copy'. We do this by checking for
            // the matrix between destination and source index, since we cannot dynamically
            // index registers.
            //
            // This requires IPT^2 shuffles because both need index lane and item offset.
            ROCPRIM_UNROLL
            for(unsigned int dst_item = 0; dst_item < ItemsPerThread; ++dst_item)
            {
                ROCPRIM_UNROLL
                for(unsigned src_item = 0; src_item < ItemsPerThread; ++src_item)
                {
                    // This shuffle can potentially be moved into the branch. We can then
                    // trade the extra masking for in-place shuffle which may potentially
                    // be faster. This may require an extra memory fence since the previous
                    // duplication into 'copy' must be finalized and we can't reuse
                    // registers as freely.
                    V temp = warp_shuffle(copy[src_item],
                                          index[dst_item] / ItemsPerThread,
                                          VirtualWaveSize);
                    if(index[dst_item] % ItemsPerThread == src_item)
                    {
                        thread_values[dst_item] = temp;
                    }
                }
            }
        }
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
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

#endif // ROCPRIM_WARP_DETAIL_WARP_SORT_SHUFFLE_HPP_
