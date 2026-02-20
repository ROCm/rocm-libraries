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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_RADIX_SORT_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_RADIX_SORT_HPP_

#include <iterator>
#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../functional.hpp"
#include "../../intrinsics.hpp"
#include "../../type_traits.hpp"
#include "../../types.hpp"

#include "../../block/block_load.hpp"
#include "../../block/block_scan.hpp"
#include "../../block/block_store.hpp"

#include "../../warp/detail/warp_sort_stable.hpp"
#include "../../warp/warp_load.hpp"
#include "../../warp/warp_store.hpp"

#include "../device_segmented_radix_sort_config.hpp"
#include "device_radix_sort.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Key,
         class Value,
         unsigned int WarpSize,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int RadixBits,
         bool         Descending>
class segmented_radix_sort_helper
{
    static constexpr unsigned int radix_size = 1 << RadixBits;

    using key_type   = Key;
    using value_type = Value;

    using count_helper_type
        = radix_digit_count_helper<WarpSize, BlockSize, ItemsPerThread, RadixBits, Descending>;
    using scan_type               = typename ::rocprim::block_scan<unsigned int, radix_size>;
    using sort_and_scatter_helper = radix_sort_and_scatter_helper<BlockSize,
                                                                  ItemsPerThread,
                                                                  RadixBits,
                                                                  Descending,
                                                                  key_type,
                                                                  value_type,
                                                                  unsigned int>;

public:
    union storage_type
    {
        typename count_helper_type::storage_type       count_helper;
        typename sort_and_scatter_helper::storage_type sort_and_scatter_helper;
    };

    template<
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort(KeysInputIterator keys_input,
              key_type * keys_tmp,
              KeysOutputIterator keys_output,
              ValuesInputIterator values_input,
              value_type * values_tmp,
              ValuesOutputIterator values_output,
              bool to_output,
              unsigned int begin_offset,
              unsigned int end_offset,
              unsigned int bit,
              unsigned int begin_bit,
              unsigned int end_bit,
              storage_type& storage)
    {
        // Handle cases when (end_bit - bit) is not divisible by radix_bits, i.e. the last
        // iteration has a shorter mask.
        const unsigned int current_radix_bits = ::rocprim::min(RadixBits, end_bit - bit);

        const bool is_first_iteration = (bit == begin_bit);

        if(is_first_iteration)
        {
            if(to_output)
            {
                sort(keys_input,
                     keys_output,
                     values_input,
                     values_output,
                     begin_offset,
                     end_offset,
                     bit,
                     current_radix_bits,
                     storage);
            }
            else
            {
                sort(keys_input,
                     keys_tmp,
                     values_input,
                     values_tmp,
                     begin_offset,
                     end_offset,
                     bit,
                     current_radix_bits,
                     storage);
            }
        }
        else
        {
            if(to_output)
            {
                sort(keys_tmp,
                     keys_output,
                     values_tmp,
                     values_output,
                     begin_offset,
                     end_offset,
                     bit,
                     current_radix_bits,
                     storage);
            }
            else
            {
                sort(keys_output,
                     keys_tmp,
                     values_output,
                     values_tmp,
                     begin_offset,
                     end_offset,
                     bit,
                     current_radix_bits,
                     storage);
            }
        }
    }

    // When all iterators are raw pointers, this overload is used to minimize code duplication in the kernel
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort(key_type * keys_input,
              key_type * keys_tmp,
              key_type * keys_output,
              value_type * values_input,
              value_type * values_tmp,
              value_type * values_output,
              bool to_output,
              unsigned int begin_offset,
              unsigned int end_offset,
              unsigned int bit,
              unsigned int begin_bit,
              unsigned int end_bit,
              storage_type& storage)
    {
        // Handle cases when (end_bit - bit) is not divisible by radix_bits, i.e. the last
        // iteration has a shorter mask.
        const unsigned int current_radix_bits = ::rocprim::min(RadixBits, end_bit - bit);

        const bool is_first_iteration = (bit == begin_bit);

        key_type*   current_keys_input;
        key_type*   current_keys_output;
        value_type* current_values_input;
        value_type* current_values_output;
        if(is_first_iteration)
        {
            if(to_output)
            {
                current_keys_input    = keys_input;
                current_keys_output   = keys_output;
                current_values_input  = values_input;
                current_values_output = values_output;
            }
            else
            {
                current_keys_input    = keys_input;
                current_keys_output   = keys_tmp;
                current_values_input  = values_input;
                current_values_output = values_tmp;
            }
        }
        else
        {
            if(to_output)
            {
                current_keys_input    = keys_tmp;
                current_keys_output   = keys_output;
                current_values_input  = values_tmp;
                current_values_output = values_output;
            }
            else
            {
                current_keys_input    = keys_output;
                current_keys_output   = keys_tmp;
                current_values_input  = values_output;
                current_values_output = values_tmp;
            }
        }
        sort(current_keys_input,
             current_keys_output,
             current_values_input,
             current_values_output,
             begin_offset,
             end_offset,
             bit,
             current_radix_bits,
             storage);
    }

private:
    template<
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort(KeysInputIterator keys_input,
              KeysOutputIterator keys_output,
              ValuesInputIterator values_input,
              ValuesOutputIterator values_output,
              unsigned int begin_offset,
              unsigned int end_offset,
              unsigned int bit,
              unsigned int current_radix_bits,
              storage_type& storage)
    {
        unsigned int digit_count;
        count_helper_type().count_digits(keys_input,
                                         begin_offset,
                                         end_offset,
                                         bit,
                                         current_radix_bits,
                                         storage.count_helper,
                                         digit_count);

        unsigned int digit_start;
        scan_type().exclusive_scan(digit_count, digit_start, 0);
        digit_start += begin_offset;

        ::rocprim::syncthreads();

        sort_and_scatter_helper().sort_and_scatter(keys_input,
                                                   keys_output,
                                                   values_input,
                                                   values_output,
                                                   begin_offset,
                                                   end_offset,
                                                   bit,
                                                   current_radix_bits,
                                                   digit_start,
                                                   storage.sort_and_scatter_helper);

        ::rocprim::syncthreads();
    }
};

template<class Key,
         class Value,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         bool         Descending>
class segmented_radix_sort_single_block_helper
{
    using key_codec
        = decltype(::rocprim::traits::get<Key>().template radix_key_codec<Descending>());
    using bit_key_type = typename key_codec::bit_key_type;
    using sort_type    = ::rocprim::block_radix_sort<Key,
                                                  BlockSize,
                                                  ItemsPerThread,
                                                  Value,
                                                  1,
                                                  1,
                                                  8,
                                                  block_radix_rank_algorithm::match>;

    static constexpr bool with_values = !std::is_same<Value, ::rocprim::empty_type>::value;

public:
    union storage_type
    {
        typename sort_type::storage_type sort;
    };

    template<class KeysInputIterator,
             class KeysOutputIterator,
             class ValuesInputIterator,
             class ValuesOutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(KeysInputIterator    keys_input,
              Key*                 keys_tmp,
              KeysOutputIterator   keys_output,
              ValuesInputIterator  values_input,
              Value*               values_tmp,
              ValuesOutputIterator values_output,
              bool                 to_output,
              unsigned int         begin_offset,
              unsigned int         end_offset,
              unsigned int         begin_bit,
              unsigned int         end_bit,
              storage_type&        storage)
    {
        if(to_output)
        {
            sort(keys_input,
                 keys_output,
                 values_input,
                 values_output,
                 begin_offset,
                 end_offset,
                 begin_bit,
                 end_bit,
                 storage);
        }
        else
        {
            sort(keys_input,
                 keys_tmp,
                 values_input,
                 values_tmp,
                 begin_offset,
                 end_offset,
                 begin_bit,
                 end_bit,
                 storage);
        }
    }

    // When all iterators are raw pointers, this overload is used to minimize code duplication in the kernel
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key*          keys_input,
              Key*          keys_tmp,
              Key*          keys_output,
              Value*        values_input,
              Value*        values_tmp,
              Value*        values_output,
              bool          to_output,
              unsigned int  begin_offset,
              unsigned int  end_offset,
              unsigned int  begin_bit,
              unsigned int  end_bit,
              storage_type& storage)
    {
        sort(keys_input,
             (to_output ? keys_output : keys_tmp),
             values_input,
             (to_output ? values_output : values_tmp),
             begin_offset,
             end_offset,
             begin_bit,
             end_bit,
             storage);
    }

    template<class KeysInputIterator,
             class KeysOutputIterator,
             class ValuesInputIterator,
             class ValuesOutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    bool sort(KeysInputIterator    keys_input,
              KeysOutputIterator   keys_output,
              ValuesInputIterator  values_input,
              ValuesOutputIterator values_output,
              unsigned int         begin_offset,
              unsigned int         end_offset,
              unsigned int         begin_bit,
              unsigned int         end_bit,
              storage_type&        storage)
    {
        constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

        using shorter_single_block_helper
            = segmented_radix_sort_single_block_helper<Key,
                                                       Value,
                                                       BlockSize,
                                                       ItemsPerThread / 2,
                                                       Descending>;

        // Segment is longer than supported by this function
        if(end_offset - begin_offset > items_per_block)
        {
            return false;
        }

        // Recursively check if it is possible to sort the segment using fewer items per thread
        const bool processed_by_shorter = shorter_single_block_helper().sort(
            keys_input,
            keys_output,
            values_input,
            values_output,
            begin_offset,
            end_offset,
            begin_bit,
            end_bit,
            reinterpret_cast<typename shorter_single_block_helper::storage_type&>(storage));
        if(processed_by_shorter)
        {
            return true;
        }

        const unsigned int flat_id = ::rocprim::flat_block_thread_id();

        Key                keys[ItemsPerThread];
        Value              values[ItemsPerThread];
        const unsigned int valid_count = end_offset - begin_offset;
        // Sort will leave "invalid" (out of size) items at the end of the sorted sequence
        const Key out_of_bounds = key_codec::get_out_of_bounds_key();
        block_load_direct_warp_striped(flat_id,
                                       keys_input + begin_offset,
                                       keys,
                                       valid_count,
                                       out_of_bounds);
        if(with_values)
        {
            block_load_direct_warp_striped(flat_id,
                                           values_input + begin_offset,
                                           values,
                                           valid_count);
        }

        ::rocprim::syncthreads();
        sort_warp_striped_to_striped<Descending>(sort_type(),
                                                 keys,
                                                 values,
                                                 storage.sort,
                                                 identity_decomposer{},
                                                 begin_bit,
                                                 end_bit);

        ::rocprim::syncthreads();
        block_store_direct_striped<BlockSize>(flat_id,
                                              keys_output + begin_offset,
                                              keys,
                                              valid_count);
        if constexpr(with_values)
        {
            block_store_direct_striped<BlockSize>(flat_id,
                                                  values_output + begin_offset,
                                                  values,
                                                  valid_count);
        }

        return true;
    }
};

template<class Key, class Value, unsigned int BlockSize, bool Descending>
class segmented_radix_sort_single_block_helper<Key, Value, BlockSize, 0, Descending>
{
public:
    struct storage_type
    {};

    template<class KeysInputIterator,
             class KeysOutputIterator,
             class ValuesInputIterator,
             class ValuesOutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    bool sort(KeysInputIterator,
              KeysOutputIterator,
              ValuesInputIterator,
              ValuesOutputIterator,
              unsigned int,
              unsigned int,
              unsigned int,
              unsigned int,
              storage_type&)
    {
        // It can't sort anything because ItemsPerThread is 0.
        // The segment will be sorted by the calles (i.e. using ItemsPerThread = 1)
        return false;
    }
};

template<unsigned int LogicalWarpSize, unsigned int ItemsPerThread, unsigned int BlockSize>
struct WarpSortHelperConfig
{
    static constexpr unsigned int logical_warp_size = LogicalWarpSize;
    static constexpr unsigned int items_per_thread  = ItemsPerThread;
    static constexpr unsigned int block_size        = BlockSize;
};

struct DisabledWarpSortHelperConfig
{
    static constexpr unsigned int logical_warp_size = 1;
    static constexpr unsigned int items_per_thread  = 1;
    static constexpr unsigned int block_size        = 1;
};

template<bool         partitioning_allowed,
         unsigned int logical_warp_size,
         unsigned int items_per_thread,
         unsigned int block_size>
using select_warp_sort_helper_config_t
    = std::conditional_t<partitioning_allowed,
                         WarpSortHelperConfig<logical_warp_size, items_per_thread, block_size>,
                         DisabledWarpSortHelperConfig>;

template<class Config, class Key, class Value, int BlockSize, bool Descending, class Enable = void>
struct segmented_warp_sort_helper
{
    static constexpr unsigned int items_per_warp = 0;
    using storage_type                           = ::rocprim::empty_type;

    template<class... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Args&&...)
    {}
};

template<class Config, class Key, class Value, int BlockSize, bool Descending>
class segmented_warp_sort_helper<
    Config,
    Key,
    Value,
    BlockSize,
    Descending,
    std::enable_if_t<!std::is_same<DisabledWarpSortHelperConfig, Config>::value>>
{
    static constexpr unsigned int logical_warp_size = Config::logical_warp_size;
    static constexpr unsigned int items_per_thread  = Config::items_per_thread;

    using key_codec
        = decltype(::rocprim::traits::get<Key>().template radix_key_codec<Descending>());

    using bit_key_type = typename key_codec::bit_key_type;

    using keys_load_type    = ::rocprim::warp_load<Key, items_per_thread, logical_warp_size>;
    using values_load_type  = ::rocprim::warp_load<Value, items_per_thread, logical_warp_size>;
    using keys_store_type   = ::rocprim::warp_store<Key, items_per_thread, logical_warp_size>;
    using values_store_type = ::rocprim::warp_store<Value, items_per_thread, logical_warp_size>;
    template<bool UseRadixMask>
    using radix_comparator_type
        = ::rocprim::detail::radix_merge_compare<Descending, UseRadixMask, Key>;
    using sort_type = ::rocprim::detail::
        warp_sort_stable<Key, BlockSize, logical_warp_size, items_per_thread, Value>;

    static constexpr bool with_values = !std::is_same<Value, ::rocprim::empty_type>::value;

public:
    static constexpr unsigned int items_per_warp = items_per_thread * logical_warp_size;

    union storage_type
    {
        typename keys_load_type::storage_type    keys_load;
        typename values_load_type::storage_type  values_load;
        typename keys_store_type::storage_type   keys_store;
        typename values_store_type::storage_type values_store;
        typename sort_type::storage_type         sort;
    };

private:
    template<typename V = Value, typename F>
    ROCPRIM_DEVICE
    auto invoke_warp_sort(Key (&keys)[items_per_thread],
                          Value (&values)[items_per_thread],
                          storage_type& storage,
                          F             comparator)
        -> std::enable_if_t<std::is_same<V, ::rocprim::empty_type>::value>
    {
        (void)values;
        sort_type().sort(keys, storage.sort, comparator);
    }

    template<typename V = Value, typename F>
    ROCPRIM_DEVICE
    auto invoke_warp_sort(Key (&keys)[items_per_thread],
                          Value (&values)[items_per_thread],
                          storage_type& storage,
                          F             comparator)
        -> std::enable_if_t<!std::is_same<V, ::rocprim::empty_type>::value>
    {
        sort_type().sort(keys, values, storage.sort, comparator);
    }

    template<class K = Key>
    ROCPRIM_DEVICE
    auto invoke_warp_sort(Key (&keys)[items_per_thread],
                          Value (&values)[items_per_thread],
                          storage_type& storage,
                          unsigned int  begin_bit,
                          unsigned int  end_bit) -> std::enable_if_t<!is_integral<K>::value>
    {
        (void)begin_bit;
        (void)end_bit;
        invoke_warp_sort(keys, values, storage, radix_comparator_type<false>{});
    }

    template<class K = Key>
    ROCPRIM_DEVICE
    auto invoke_warp_sort(Key (&keys)[items_per_thread],
                          Value (&values)[items_per_thread],
                          storage_type& storage,
                          unsigned int  begin_bit,
                          unsigned int  end_bit) -> std::enable_if_t<is_integral<K>::value>
    {
        if(begin_bit == 0 && end_bit == 8 * sizeof(Key))
        {
            invoke_warp_sort(keys, values, storage, radix_comparator_type<false>{});
        }
        else
        {
            radix_comparator_type<true> comparator(begin_bit, end_bit - begin_bit);
            invoke_warp_sort(keys, values, storage, comparator);
        }
    }

public:
    template<class KeysInputIterator,
             class KeysOutputIterator,
             class ValuesInputIterator,
             class ValuesOutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(KeysInputIterator    keys_input,
              KeysOutputIterator   keys_output,
              ValuesInputIterator  values_input,
              ValuesOutputIterator values_output,
              unsigned int         begin_offset,
              unsigned int         end_offset,
              unsigned int         begin_bit,
              unsigned int         end_bit,
              storage_type&        storage)
    {
        const unsigned int num_items     = end_offset - begin_offset;
        const Key          out_of_bounds = key_codec::get_out_of_bounds_key();

        Key   keys[items_per_thread];
        Value values[items_per_thread];
        keys_load_type().load(keys_input + begin_offset,
                              keys,
                              num_items,
                              out_of_bounds,
                              storage.keys_load);

        if constexpr(with_values)
        {
            ::rocprim::wave_barrier();
            values_load_type().load(values_input + begin_offset,
                                    values,
                                    num_items,
                                    storage.values_load);
        }

        ::rocprim::wave_barrier();
        invoke_warp_sort(keys, values, storage, begin_bit, end_bit);

        ::rocprim::wave_barrier();
        keys_store_type().store(keys_output + begin_offset, keys, num_items, storage.keys_store);

        if constexpr(with_values)
        {
            ::rocprim::wave_barrier();
            values_store_type().store(values_output + begin_offset,
                                      values,
                                      num_items,
                                      storage.values_store);
        }
    }

    template<class KeysInputIterator,
             class KeysOutputIterator,
             class ValuesInputIterator,
             class ValuesOutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(KeysInputIterator    keys_input,
              Key*                 keys_tmp,
              KeysOutputIterator   keys_output,
              ValuesInputIterator  values_input,
              Value*               values_tmp,
              ValuesOutputIterator values_output,
              bool                 to_output,
              unsigned int         begin_offset,
              unsigned int         end_offset,
              unsigned int         begin_bit,
              unsigned int         end_bit,
              storage_type&        storage)
    {
        if(to_output)
        {
            sort(keys_input,
                 keys_output,
                 values_input,
                 values_output,
                 begin_offset,
                 end_offset,
                 begin_bit,
                 end_bit,
                 storage);
        }
        else
        {
            sort(keys_input,
                 keys_tmp,
                 values_input,
                 values_tmp,
                 begin_offset,
                 end_offset,
                 begin_bit,
                 end_bit,
                 storage);
        }
    }
};

template<
    class ArchConfig,
    bool Descending,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class OffsetIterator
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void segmented_sort(KeysInputIterator keys_input,
                    typename std::iterator_traits<KeysInputIterator>::value_type * keys_tmp,
                    KeysOutputIterator keys_output,
                    ValuesInputIterator values_input,
                    typename std::iterator_traits<ValuesInputIterator>::value_type * values_tmp,
                    ValuesOutputIterator values_output,
                    bool to_output,
                    OffsetIterator begin_offsets,
                    OffsetIterator end_offsets,
                    unsigned int iterations,
                    unsigned int begin_bit,
                    unsigned int end_bit)
{
    static constexpr segmented_radix_sort_config_params params = ArchConfig::params;

    static constexpr unsigned int radix_bits        = params.radix_bits;
    static constexpr unsigned int block_size        = params.kernel_config.block_size;
    static constexpr unsigned int items_per_thread  = params.kernel_config.items_per_thread;
    static constexpr unsigned int items_per_block   = block_size * items_per_thread;
    static constexpr bool         warp_sort_enabled = params.enable_unpartitioned_warp_sort;

    using key_type   = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    using single_block_helper_type = segmented_radix_sort_single_block_helper<key_type,
                                                                              value_type,
                                                                              block_size,
                                                                              items_per_thread,
                                                                              Descending>;
    using long_radix_helper_type
        = segmented_radix_sort_helper<key_type,
                                      value_type,
                                      ::rocprim::arch::wavefront::min_size(),
                                      block_size,
                                      items_per_thread,
                                      radix_bits,
                                      Descending>;
    using warp_sort_helper_type = segmented_warp_sort_helper<
        select_warp_sort_helper_config_t<params.warp_sort_config.partitioning_allowed,
                                         params.warp_sort_config.logical_warp_size_small,
                                         params.warp_sort_config.items_per_thread_small,
                                         params.warp_sort_config.block_size_small>,
        key_type,
        value_type,
        block_size,
        Descending>;

    static constexpr unsigned int items_per_warp = warp_sort_helper_type::items_per_warp;

    ROCPRIM_SHARED_MEMORY union
    {
        typename single_block_helper_type::storage_type single_block_helper;
        typename long_radix_helper_type::storage_type   long_radix_helper;
        typename warp_sort_helper_type::storage_type    warp_sort_helper;
    } storage;

    const unsigned int segment_id = ::rocprim::detail::block_id<0>();

    const unsigned int begin_offset = begin_offsets[segment_id];
    const unsigned int end_offset   = end_offsets[segment_id];

    // Empty segment
    if(end_offset <= begin_offset)
    {
        return;
    }

    if(end_offset - begin_offset > items_per_block)
    {
        // Large segment
        for(unsigned int bit = begin_bit; bit < end_bit; bit += radix_bits)
        {
            long_radix_helper_type().sort(keys_input,
                                          keys_tmp,
                                          keys_output,
                                          values_input,
                                          values_tmp,
                                          values_output,
                                          to_output,
                                          begin_offset,
                                          end_offset,
                                          bit,
                                          begin_bit,
                                          end_bit,
                                          storage.long_radix_helper);

            to_output = !to_output;
        }
    }
    else if(!warp_sort_enabled || end_offset - begin_offset > items_per_warp)
    {
        // Small segment
        single_block_helper_type().sort(keys_input,
                                        keys_tmp,
                                        keys_output,
                                        values_input,
                                        values_tmp,
                                        values_output,
                                        (iterations % 2 == 0) != to_output,
                                        begin_offset,
                                        end_offset,
                                        begin_bit,
                                        end_bit,
                                        storage.single_block_helper);
    }
    else if(::rocprim::flat_block_thread_id() < params.warp_sort_config.logical_warp_size_small)
    {
        // Single warp segment
        warp_sort_helper_type().sort(keys_input,
                                     keys_tmp,
                                     keys_output,
                                     values_input,
                                     values_tmp,
                                     values_output,
                                     (iterations % 2 == 0) != to_output,
                                     begin_offset,
                                     end_offset,
                                     begin_bit,
                                     end_bit,
                                     storage.warp_sort_helper);
    }
}

template<
    class ArchConfig,
    bool Descending,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class SegmentIndexIterator,
    class OffsetIterator
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void segmented_sort_large(KeysInputIterator keys_input,
                          typename std::iterator_traits<KeysInputIterator>::value_type * keys_tmp,
                          KeysOutputIterator keys_output,
                          ValuesInputIterator values_input,
                          typename std::iterator_traits<ValuesInputIterator>::value_type * values_tmp,
                          ValuesOutputIterator values_output,
                          bool to_output,
                          SegmentIndexIterator segment_indices,
                          OffsetIterator begin_offsets,
                          OffsetIterator end_offsets,
                          unsigned int iterations,
                          unsigned int begin_bit,
                          unsigned int end_bit)
{
    static constexpr segmented_radix_sort_config_params params = ArchConfig::params;

    static constexpr unsigned int radix_bits       = params.radix_bits;
    static constexpr unsigned int block_size       = params.kernel_config.block_size;
    static constexpr unsigned int items_per_thread = params.kernel_config.items_per_thread;
    static constexpr unsigned int items_per_block  = block_size * items_per_thread;

    using key_type   = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    using single_block_helper_type = segmented_radix_sort_single_block_helper<key_type,
                                                                              value_type,
                                                                              block_size,
                                                                              items_per_thread,
                                                                              Descending>;
    using long_radix_helper_type
        = segmented_radix_sort_helper<key_type,
                                      value_type,
                                      ::rocprim::arch::wavefront::min_size(),
                                      block_size,
                                      items_per_thread,
                                      radix_bits,
                                      Descending>;

    ROCPRIM_SHARED_MEMORY union
    {
        typename single_block_helper_type::storage_type single_block_helper;
        typename long_radix_helper_type::storage_type   long_radix_helper;
    } storage;

    const unsigned int block_id     = ::rocprim::detail::block_id<0>();
    const unsigned int segment_id   = segment_indices[block_id];
    const unsigned int begin_offset = begin_offsets[segment_id];
    const unsigned int end_offset   = end_offsets[segment_id];

    if(end_offset <= begin_offset)
    {
        return;
    }

    if(end_offset - begin_offset > items_per_block)
    {
        for(unsigned int bit = begin_bit; bit < end_bit; bit += radix_bits)
        {
            long_radix_helper_type().sort(keys_input,
                                          keys_tmp,
                                          keys_output,
                                          values_input,
                                          values_tmp,
                                          values_output,
                                          to_output,
                                          begin_offset,
                                          end_offset,
                                          bit,
                                          begin_bit,
                                          end_bit,
                                          storage.long_radix_helper);

            to_output = !to_output;
        }
    }
    else
    {
        single_block_helper_type().sort(keys_input,
                                        keys_tmp,
                                        keys_output,
                                        values_input,
                                        values_tmp,
                                        values_output,
                                        (iterations % 2 == 0) != to_output,
                                        begin_offset,
                                        end_offset,
                                        begin_bit,
                                        end_bit,
                                        storage.single_block_helper);
    }
}

template<
    class ArchConfig,
    bool Descending,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class SegmentIndexIterator,
    class OffsetIterator
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void segmented_sort_small(KeysInputIterator keys_input,
                          typename std::iterator_traits<KeysInputIterator>::value_type * keys_tmp,
                          KeysOutputIterator keys_output,
                          ValuesInputIterator values_input,
                          typename std::iterator_traits<ValuesInputIterator>::value_type * values_tmp,
                          ValuesOutputIterator values_output,
                          bool to_output,
                          unsigned int num_segments,
                          SegmentIndexIterator segment_indices,
                          OffsetIterator begin_offsets,
                          OffsetIterator end_offsets,
                          unsigned int begin_bit,
                          unsigned int end_bit)
{
    static constexpr segmented_radix_sort_config_params params = ArchConfig::params;

    static constexpr unsigned int block_size = params.warp_sort_config.block_size_small;
    static constexpr unsigned int logical_warp_size
        = params.warp_sort_config.logical_warp_size_small;
    static_assert(block_size % logical_warp_size == 0,
                  "logical_warp_size must be a divisor of block_size");
    static constexpr unsigned int warps_per_block = block_size / logical_warp_size;

    using key_type   = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    using warp_sort_helper_type = segmented_warp_sort_helper<
        select_warp_sort_helper_config_t<params.warp_sort_config.partitioning_allowed,
                                         params.warp_sort_config.logical_warp_size_small,
                                         params.warp_sort_config.items_per_thread_small,
                                         params.warp_sort_config.block_size_small>,
        key_type,
        value_type,
        block_size,
        Descending>;

    ROCPRIM_SHARED_MEMORY typename warp_sort_helper_type::storage_type storage;

    const unsigned int block_id        = ::rocprim::detail::block_id<0>();
    const unsigned int logical_warp_id = ::rocprim::detail::logical_warp_id<logical_warp_size>();
    const unsigned int segment_index   = block_id * warps_per_block + logical_warp_id;
    if(segment_index >= num_segments)
    {
        return;
    }

    const unsigned int segment_id   = segment_indices[segment_index];
    const unsigned int begin_offset = begin_offsets[segment_id];
    const unsigned int end_offset   = end_offsets[segment_id];

    if(end_offset <= begin_offset)
    {
        return;
    }

    warp_sort_helper_type().sort(keys_input,
                                 keys_tmp,
                                 keys_output,
                                 values_input,
                                 values_tmp,
                                 values_output,
                                 to_output,
                                 begin_offset,
                                 end_offset,
                                 begin_bit,
                                 end_bit,
                                 storage);
}

template<class ArchConfig,
         bool Descending,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class SegmentIndexIterator,
         class OffsetIterator>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void segmented_sort_medium(
    KeysInputIterator                                               keys_input,
    typename std::iterator_traits<KeysInputIterator>::value_type*   keys_tmp,
    KeysOutputIterator                                              keys_output,
    ValuesInputIterator                                             values_input,
    typename std::iterator_traits<ValuesInputIterator>::value_type* values_tmp,
    ValuesOutputIterator                                            values_output,
    bool                                                            to_output,
    unsigned int                                                    num_segments,
    SegmentIndexIterator                                            segment_indices,
    OffsetIterator                                                  begin_offsets,
    OffsetIterator                                                  end_offsets,
    unsigned int                                                    begin_bit,
    unsigned int                                                    end_bit)
{
    static constexpr segmented_radix_sort_config_params params = ArchConfig::params;

    static constexpr unsigned int block_size = params.warp_sort_config.block_size_medium;
    static constexpr unsigned int logical_warp_size
        = params.warp_sort_config.logical_warp_size_medium;
    static_assert(block_size % logical_warp_size == 0,
                  "logical_warp_size must be a divisor of block_size");
    static constexpr unsigned int warps_per_block = block_size / logical_warp_size;

    using key_type   = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    using warp_sort_helper_type = segmented_warp_sort_helper<
        select_warp_sort_helper_config_t<params.warp_sort_config.partitioning_allowed,
                                         params.warp_sort_config.logical_warp_size_medium,
                                         params.warp_sort_config.items_per_thread_medium,
                                         params.warp_sort_config.block_size_medium>,
        key_type,
        value_type,
        block_size,
        Descending>;

    ROCPRIM_SHARED_MEMORY typename warp_sort_helper_type::storage_type storage;

    const unsigned int block_id        = ::rocprim::detail::block_id<0>();
    const unsigned int logical_warp_id = ::rocprim::detail::logical_warp_id<logical_warp_size>();
    const unsigned int segment_index   = block_id * warps_per_block + logical_warp_id;
    if(segment_index >= num_segments)
    {
        return;
    }

    const unsigned int segment_id   = segment_indices[segment_index];
    const unsigned int begin_offset = begin_offsets[segment_id];
    const unsigned int end_offset   = end_offsets[segment_id];
    if(end_offset <= begin_offset)
    {
        return;
    }

    warp_sort_helper_type().sort(keys_input,
                                 keys_tmp,
                                 keys_output,
                                 values_input,
                                 values_tmp,
                                 values_output,
                                 to_output,
                                 begin_offset,
                                 end_offset,
                                 begin_bit,
                                 end_bit,
                                 storage);
}

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_RADIX_SORT_HPP_
