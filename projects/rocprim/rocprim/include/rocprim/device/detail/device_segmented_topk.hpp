// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_TOPK_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_TOPK_HPP_

#include "../../block/block_load_func.hpp"
#include "../../block/block_scan.hpp"
#include "../../block/block_store_func.hpp"
#include "../../detail/temp_storage.hpp"
#include "../device_segmented_radix_sort.hpp"
#include "../device_transform.hpp"
#include "rocprim/config.hpp"
#include "rocprim/device/config_types.hpp"
#include "rocprim/device/device_select.hpp"
#include "rocprim/functional.hpp"
#include <iostream>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

/// \brief TODO: This is a naive implementation of a segmented topk algorithm.
/// It uses radix sort to sort each segment and then selects the top K elements from each segment.
/// This implementation is not optimized for performance and is only intended to be a reference implementation
/// for testing and validation purposes. A more efficient implementation will be added in the future.
///
template<typename KeysInputIterator,
         typename KeysOutputIterator,
         typename ValuesInputIterator,
         typename ValuesOutputIterator,
         class OffsetIterator,
         typename SizeIn,
         typename SizeOut,
         bool Descending,
         class Decomposer>
struct device_segmented_topk_impl
{
    // Constant member variables
    using key_in_t    = typename std::iterator_traits<KeysInputIterator>::value_type;
    using key_out_t   = typename std::iterator_traits<KeysOutputIterator>::value_type;
    using value_in_t  = typename std::iterator_traits<ValuesInputIterator>::value_type;
    using value_out_t = typename std::iterator_traits<ValuesOutputIterator>::value_type;

    static_assert(std::is_same_v<key_in_t, key_out_t>,
                  "KeysInputIterator and KeysOutputIterator must have the same value_type");
    static_assert(std::is_same_v<value_in_t, value_out_t>,
                  "ValuesInputIterator and ValuesOutputIterator must have the same value_type");
    static_assert(rocprim::is_integral<SizeIn>::value, "SizeIn must be integral");
    static_assert(rocprim::is_integral<SizeOut>::value, "SizeOut must be integral");
    // key type must be a fundamental/integral type that supports radix sort without custom decomposer
    static_assert(!std::is_same_v<key_in_t, ::rocprim::empty_type>, "key_in_t empty!");
    // key type must be a fundamental/integral type that supports radix sort without custom decomposer
    static_assert(!std::is_same_v<key_out_t, ::rocprim::empty_type>, "key_out_t empty!");

public:
    static hipError_t impl(void*                             temporary_storage,
                           size_t&                           storage_size,
                           const KeysInputIterator           keys_input,
                           const KeysOutputIterator          keys_output,
                           const ValuesInputIterator         values_input,
                           const ValuesOutputIterator        values_output,
                           const SizeIn                      size,
                           const SizeOut                     K,
                           const size_t                      segments,
                           const OffsetIterator              begin_offsets,
                           const OffsetIterator              end_offsets,
                           [[maybe_unused]] const Decomposer decomposer        = {},
                           const hipStream_t                 stream            = 0,
                           const bool                        debug_synchronous = false)
    {

        ValuesInputIterator temp_values              = nullptr;
        KeysInputIterator   temp_keys                = nullptr;
        void*               temporary_storage_radix  = nullptr;
        size_t              radix_storage_size_bytes = 0;

        if(temporary_storage != nullptr)
        {
            constexpr bool with_values = !std::is_same_v<ValuesInputIterator, rocprim::empty_type>;
            // Partition temporary storage for segmented_radix_sort and temporary results

            radix_storage_size_bytes = storage_size - size * sizeof(key_in_t) - (with_values ? size : 0) * sizeof(value_in_t);
            // When keys and values are sorted we need temporary storage for both
            ROCPRIM_RETURN_ON_ERROR(detail::temp_storage::partition(
                temporary_storage,
                storage_size,
                detail::temp_storage::make_linear_partition(
                    detail::temp_storage::ptr_aligned_array(&temp_keys, size),
                    detail::temp_storage::ptr_aligned_array(&temp_values, with_values ? size : 0),
                    detail::temp_storage::make_partition(&temporary_storage_radix, 1))));
        }

        bool ignored = false;

        ROCPRIM_RETURN_ON_ERROR(
            detail::segmented_radix_sort_impl<default_config, Descending>(temporary_storage_radix,
                                                                          radix_storage_size_bytes,
                                                                          keys_input,
                                                                          nullptr,
                                                                          temp_keys,
                                                                          values_input,
                                                                          nullptr,
                                                                          temp_values,
                                                                          size,
                                                                          ignored,
                                                                          segments,
                                                                          begin_offsets,
                                                                          end_offsets,
                                                                          0,
                                                                          8 * sizeof(key_in_t),
                                                                          stream,
                                                                          debug_synchronous));

        if(temporary_storage == nullptr)
        {
            storage_size
                = radix_storage_size_bytes + size * sizeof(key_in_t) + size * sizeof(value_in_t);

            ROCPRIM_RETURN_ON_ERROR(detail::temp_storage::partition(
                temporary_storage,
                storage_size,
                detail::temp_storage::make_linear_partition(
                    detail::temp_storage::ptr_aligned_array(&temp_keys, size),
                    detail::temp_storage::ptr_aligned_array(&temp_values, size),
                    detail::temp_storage::make_partition(&temporary_storage_radix,
                                                         radix_storage_size_bytes))));
            // Return temp memory size required by segmented_radix_sort and temporary results
            return hipSuccess;
        }

        for(size_t segment = 0; segment < segments; segment++)
        {

            // Move first K keys from sorted temporary buffer to output
            ROCPRIM_RETURN_ON_ERROR(rocprim::transform(temp_keys + begin_offsets[segment],
                                                       keys_output + segment * K,
                                                       K,
                                                       rocprim::identity<>{}));
            if constexpr(!std::is_same_v<ValuesInputIterator, rocprim::empty_type>)
            {
                // If values are provided, also move first K values to the output
                ROCPRIM_RETURN_ON_ERROR(rocprim::transform(temp_values + begin_offsets[segment],
                                                           values_output + segment * K,
                                                           K,
                                                           rocprim::identity<>{}));
            }
        }

        return hipSuccess;
    }
};

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_TOPK_HPP_
