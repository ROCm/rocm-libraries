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

template<int VirtualWaveSize, int ItemsPerThread>
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
    template<bool = false, bool = false, class K, class V, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void
        wlev_cas(bool is_upper, BinaryFunction compare_function, unsigned int xor_mask, K& k, V& v)
    {
        K k1 = lane_xor_swap(k, xor_mask);

        const bool swap = compare_function(is_upper ? k : k1, is_upper ? k1 : k);

        if(swap)
        {
            k = k1;
            v = lane_xor_swap(v, xor_mask);
        }
    }

    /// Compares multiple values between pairs of lanes, where the pairs are selected via
    /// a XOR-masks. Then sorts the keys and values according to the passed direction.
    template<bool is_first_step = false,
             bool try_pack      = false,
             class K,
             class V,
             class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void wlev_cas(bool           is_upper,
                         BinaryFunction compare_function,
                         unsigned int   xor_mask,
                         K (&k)[ItemsPerThread],
                         V (&v)[ItemsPerThread])
    {
        if constexpr(is_first_step && ItemsPerThread > 1)
        {
            ::rocprim::detail::constexpr_for_lt<0, ItemsPerThread / 2, 1>(
                [&](auto item)
                {
                    constexpr auto other_item = ItemsPerThread - 1 - item;

                    // To avoid read-after-write conditions, we perform a round-trip exchange.
                    // First, fetch the paired lane's other_item. By reading the values before
                    // any local modifications, the original data will be preserved across lanes.
                    K other_k = lane_xor_swap(k[other_item], xor_mask);
                    V other_v = lane_xor_swap(v[other_item], xor_mask);

                    const bool swap = compare_function(is_upper ? k[item] : other_k,
                                                       is_upper ? other_k : k[item]);

                    swap_if<swap_method::ternary>(swap, k[item], other_k);
                    swap_if<swap_method::ternary>(swap, v[item], other_v);

                    // Return the updated value back to the paired lane.
                    k[other_item] = lane_xor_swap(other_k, xor_mask);
                    v[other_item] = lane_xor_swap(other_v, xor_mask);
                });
        }
        else
        {
            ::rocprim::detail::constexpr_for_lt<0, ItemsPerThread, 1>(
                [&](auto item)
                {
                    K k1 = lane_xor_swap(k[item], xor_mask);

                    const bool swap
                        = compare_function(is_upper ? k[item] : k1, is_upper ? k1 : k[item]);

                    if(swap)
                    {
                        k[item] = k1;
                        v[item] = lane_xor_swap(v[item], xor_mask);
                    }
                });
        }
    }

    /// Compares the value between pairs of lanes, where the pairs are selected via
    /// a XOR-masks. Then sorts the keys according to the passed direction.
    template<bool = false, bool = false, class K, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void
        wlev_cas(bool is_upper, BinaryFunction compare_function, unsigned int xor_mask, K& k)
    {
        const K    k1   = lane_xor_swap(k, xor_mask);
        const bool swap = compare_function(is_upper ? k : k1, is_upper ? k1 : k);
        k               = swap ? k1 : k;
    }

    /// Compares multiple values between pairs of lanes, where the pairs are selected via
    /// a XOR-masks. Then sorts the keys according to the passed direction.
    template<bool is_first_step = false, bool try_pack = false, class K, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void wlev_cas(bool           is_upper,
                         BinaryFunction compare_function,
                         unsigned int   xor_mask,
                         K (&k)[ItemsPerThread])
    {
        constexpr bool is_packable = ItemsPerThread > 1;
        if constexpr(try_pack && VirtualWaveSize >= ::rocprim::arch::wavefront::min_size()
                     && is_packable)
        {
            using pack        = rocprim::detail::thread_items_pack<K, ItemsPerThread>;
            auto packed_items = pack::create(k);
            auto other_items  = lane_xor_swap(packed_items, xor_mask);

            // for(unsigned int item = 0; item < ItemsPerThread; item++)
            ::rocprim::detail::constexpr_for_lt<0, ItemsPerThread, 1>(
                [&](auto item)
                {
                    constexpr auto other_item = is_first_step ? (ItemsPerThread - 1 - item) : item;

                    K          k1 = other_items[other_item];
                    const bool swap
                        = compare_function(is_upper ? k[item] : k1, is_upper ? k1 : k[item]);
                    k[item] = swap ? k1 : k[item];
                });
        }
        else if constexpr(is_first_step)
        {
            ::rocprim::detail::constexpr_for_lt<0, ItemsPerThread / 2, 1>(
                [&](auto item)
                {
                    constexpr auto other_item = ItemsPerThread - 1 - item;

                    K other_k = lane_xor_swap(k[other_item], xor_mask);

                    const bool swap = compare_function(is_upper ? k[item] : other_k,
                                                       is_upper ? other_k : k[item]);

                    swap_if<swap_method::ternary>(swap, k[item], other_k);

                    k[other_item] = lane_xor_swap(other_k, xor_mask);
                });
        }
        else
        {
            // for(unsigned int item = 0; item < ItemsPerThread; item++)
            ::rocprim::detail::constexpr_for_lt<0, ItemsPerThread, 1>(
                [&](auto item)
                {
                    K          k1   = lane_xor_swap(k[item], xor_mask);
                    const bool swap
                        = compare_function(is_upper ? k[item] : k1, is_upper ? k1 : k[item]);
                    k[item] = swap ? k1 : k[item];
                });
        }
    }

    template<int i_l, int i_r, class BinaryFunction, class K>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void tlev_cas_single(BinaryFunction compare_function, K (&k)[ItemsPerThread])
    {
        // Using v_cndmask (ternary) is faster than v_mov (branched).
        //
        // However the compiler does not want to emit 'v_dual_cndmask'
        // for some unknown reason. We require swap to be in VCC, but there
        // isn't really a way to enforce it right now. However, in branched
        // we can use `v_dual_mov`, so it's preferred for large types.
        // constexpr swap_method method
        //     = sizeof(K) <= 4 ? swap_method::ternary : swap_method::branched;
        constexpr swap_method method = swap_method::ternary;

        const bool swap = compare_function(k[i_r], k[i_l]);
        swap_if<method>(swap, k[i_l], k[i_r]);
    }

    template<int i_l, int i_r, class BinaryFunction, class K, class V>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void tlev_cas_single(BinaryFunction compare_function,
                                K (&k)[ItemsPerThread],
                                V (&v)[ItemsPerThread])
    {
        // See previous 'tlev_cas_single' implementation.
        constexpr swap_method method
            = (sizeof(K) <= 4 || sizeof(V) <= 4) ? swap_method::ternary : swap_method::branched;

        const bool swap = compare_function(k[i_r], k[i_l]);
        swap_if<method>(swap, k[i_l], k[i_r]);
        swap_if<method>(swap, v[i_l], v[i_r]);
    }

    // Applies the thread-level compare and swaps.
    template<unsigned int group_size,
             unsigned int offset,
             bool         use_mirror_pairing,
             class BinaryFunction,
             class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void tlev_cas(BinaryFunction compare_function, KeyValue&... kv)
    {
        // for(unsigned int base = 0; base < ItemsPerThread; base += 2 * offset)
        ::rocprim::detail::constexpr_for_lt<0, ItemsPerThread, 2 * offset>(
            [&](auto base)
            {
                // for(unsigned i = 0; i < offset; ++i)
                ::rocprim::detail::constexpr_for_lt<0, offset, 1>(
                    [&](auto i)
                    {
                        constexpr unsigned int i_l = base + i;

                        // To support forward-only comparisons, the first step of building a bitonic
                        // sequence pairs the lower half with the reversed (mirrored) upper half.
                        // Subsequent passes use a standard linear offset.
                        constexpr unsigned int i_r = use_mirror_pairing
                                                         ? (base + group_size - 1 - i)
                                                         : (base + i + offset);

                        tlev_cas_single<i_l, i_r>(compare_function, kv...);
                    });
            });
    }

    template<class BinaryFunction,
             int  group_size    = ItemsPerThread,
             int  offset        = group_size / 2,
             bool is_first_pass = false,
             class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void tlev_pass(BinaryFunction compare_function, KeyValue&... kv)
    {
        // Implement the following loop using recursion:
        //   for(unsigned int offset = group_size / 2; offset > 0; offset /= 2)
        if constexpr(offset > 0)
        {
            // Mirror pairing is only required on the first step of the group to
            // establish the correct bitonic topology.
            constexpr bool use_mirror = is_first_pass && (offset == group_size / 2);
            tlev_cas<group_size, offset, use_mirror>(compare_function, kv...);

            tlev_pass<BinaryFunction, group_size, offset / 2, false>(compare_function, kv...);
        }
    }

    template<class BinaryFunction, int group_size = 2, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void tlev_sort(BinaryFunction compare_function, KeyValue&... kv)
    {
        // Recursively double the group size to build increasingly larger *bitonic* sequences.
        //   for(unsigned int group_size = 2; group_size <= ItemsPerThread; group_size *= 2)
        if constexpr(group_size <= ItemsPerThread)
        {
            // The very first pass for a new group size must use mirror pairing to properly
            // pair the two monotonic sequences into a bitonic one.
            tlev_pass<BinaryFunction, group_size, group_size / 2, true>(compare_function, kv...);
            // Recurse...
            tlev_sort<BinaryFunction, group_size * 2, KeyValue...>(compare_function, kv...);
        }
    }

    /// High level bitonic sort. Handles both single item per threads and multiple.
    template<class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void bitonic_sort(BinaryFunction compare_function, KeyValue&... kv)
    {
        static_assert(sizeof...(KeyValue) < 3,
                      "KeyValue parameter pack can 1 or 2 elements (key, or key and value)");
        static_assert(detail::is_power_of_two(ItemsPerThread), "ItemsPerThread must be power of 2");

        constexpr int num_id_bits = Log2<VirtualWaveSize>::VALUE;

        // We need to invoke '(id >> i) & 1u' a lot, so let's just nudge the
        // compiler that we can really easily precompute this using bit-field
        // extract.
        unsigned int id_bits[num_id_bits + 1];

        const unsigned int id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        // for(int i = 0; i < num_id_bits; ++i)
        ::rocprim::detail::constexpr_for_lt<0, num_id_bits, 1>(
            [&](auto i)
            {
                // Use unsigned bitfield extract to select the i-th bit from id.
                id_bits[i] = __builtin_amdgcn_ubfe(id, i, 1u);
            });
        // This will get optimized out, just needed to make the loop work below.
        id_bits[num_id_bits] = 0u;

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
            tlev_sort(compare_function, kv...);
        }

        // Now we have bitonic sequences over our thread-level items, we need to
        // cooperatively create bigger sequences over more and more lanes.
        //
        // for(int group_bit = 1; (1 << group_bit) <= VirtualWaveSize; ++group_bit)
        ::rocprim::detail::constexpr_for_lte<1, num_id_bits, 1>(
            [&](auto group_bit)
            {
                // Each iteration builds and merges a bitonic sequence of '1 << group_bit'
                // elements across lanes.
                //
                // Unlike traditional bitonic sort which alternates sorting directions
                // (ascending/descending) to build the bitonic topology (/\/\), this is a
                // FORWARD-ONLY bitonic sort. All lanes sort in the same monotonic direction (////).
                //
                // We simulate the bitonic sequence structurally: during the very first step
                // of a new group size, we pair lanes using a mirrored XOR mask. This folds
                // the previously sorted monotonic sequences into a bitonic shape, which is
                // then merged into a larger monotonic sequence by following linear steps.
                ::rocprim::detail::constexpr_for_gte<group_bit - 1, 0, -1>(
                    [&](auto offset_bit)
                    {
                        // The offset_bit (via is_upper) denotes the role of the current lane
                        // within the comparison pair. It ensures the smaller element goes to
                        // the lane with the lower ID, and the larger to the higher ID.
                        //
                        // is_upper == 0: This lane is the lower half. It wants the minimum.
                        // is_upper == 1: This lane is the upper half. It wants the maximum.
                        constexpr bool         is_first_step = (offset_bit == group_bit - 1);
                        constexpr unsigned int xor_mask
                            = is_first_step ? ((1u << group_bit) - 1u) : (1u << offset_bit);
                        const unsigned int is_upper = id_bits[offset_bit];

                        wlev_cas<is_first_step>(is_upper, compare_function, xor_mask, kv...);
                    });

                // Don't forget that we need to also do a pass (not a full sort) over
                // thread level items. Only need to invoke thread-level algorithms if we have multiple
                // items per thread.
                if constexpr(ItemsPerThread > 1)
                {
                    tlev_pass(compare_function, kv...);
                }
            });
    }
};

template<class Key, unsigned int VirtualWaveSize, class Value>
struct warp_sort_shuffle
{
public:
    static_assert(detail::is_power_of_two(VirtualWaveSize), "VirtualWaveSize must be power of 2");

    using storage_type = ::rocprim::detail::empty_storage_type;

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key& thread_value, BinaryFunction compare_function)
    {
        // sort by value only
        warp_shuffle_sort_impl<VirtualWaveSize, 1>::bitonic_sort(compare_function, thread_value);
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
        warp_shuffle_sort_impl<VirtualWaveSize, ItemsPerThread>::bitonic_sort(compare_function,
                                                                              thread_values);
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
            warp_shuffle_sort_impl<VirtualWaveSize, 1>::bitonic_sort(compare_function,
                                                                     thread_key,
                                                                     thread_value);
        }
        else
        {
            // Instead of passing large values between lanes we pass indices and gather values after sorting.
            unsigned int v = detail::logical_lane_id<VirtualWaveSize>();
            warp_shuffle_sort_impl<VirtualWaveSize, 1>::bitonic_sort(compare_function,
                                                                     thread_key,
                                                                     v);
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
            warp_shuffle_sort_impl<VirtualWaveSize, ItemsPerThread>::bitonic_sort(compare_function,
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

            warp_shuffle_sort_impl<VirtualWaveSize, ItemsPerThread>::bitonic_sort(compare_function,
                                                                                  thread_keys,
                                                                                  index);

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
