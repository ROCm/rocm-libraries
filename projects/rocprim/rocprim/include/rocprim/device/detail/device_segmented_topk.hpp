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

#include "../../config.hpp"
#include "../../detail/temp_storage.hpp"
#include "../../iterator/constant_iterator.hpp"
#include "../../iterator/counting_iterator.hpp"
#include "../../iterator/transform_iterator.hpp"
#include "../../type_traits.hpp"
#include "../config_types.hpp"
#include "../device_copy.hpp"
#include "../device_segmented_radix_sort.hpp"

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

    static constexpr bool with_values = !std::is_same_v<ValuesInputIterator, rocprim::empty_type>;

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

        KeysInputIterator   temp_keys       = nullptr;
        ValuesInputIterator temp_values     = nullptr;
        void*               scratch_storage = nullptr;

        size_t segmented_radix_sort_size = 0;
        bool   ignored                   = false;
        auto   do_segmented_radix_sort   = [&]()
        {
            return detail::segmented_radix_sort_impl<default_config, Descending>(
                scratch_storage,
                segmented_radix_sort_size,
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
                debug_synchronous);
        };

        size_t copy_keys_size = 0;
        auto   do_copy_keys   = [&]()
        {
            return rocprim::batch_copy(
                scratch_storage,
                copy_keys_size,
                rocprim::make_transform_iterator(begin_offsets,
                                                 [=](auto offset) { return temp_keys + offset; }),
                rocprim::make_transform_iterator(rocprim::make_counting_iterator(size_t{0}),
                                                 [=](auto i) { return keys_output + i * K; }),
                rocprim::make_constant_iterator(K),
                segments,
                stream,
                debug_synchronous);
        };

        size_t copy_vals_size = 0;
        auto   do_copy_vals   = [&]()
        {
            return rocprim::batch_copy(
                scratch_storage,
                copy_vals_size,
                rocprim::make_transform_iterator(begin_offsets,
                                                 [=](auto offset) { return temp_values + offset; }),
                rocprim::make_transform_iterator(rocprim::make_counting_iterator(size_t{0}),
                                                 [=](auto i) { return values_output + i * K; }),
                rocprim::make_constant_iterator(K),
                segments,
                stream,
                debug_synchronous);
        };

        // Compute required scratch storage for other passes.
        ROCPRIM_RETURN_ON_ERROR(do_segmented_radix_sort());
        ROCPRIM_RETURN_ON_ERROR(do_copy_keys());
        if constexpr(with_values)
        {
            ROCPRIM_RETURN_ON_ERROR(do_copy_vals());
        }

        const size_t scratch_storage_size
            = std::max(segmented_radix_sort_size, std::max(copy_keys_size, copy_vals_size));
        ROCPRIM_RETURN_ON_ERROR(detail::temp_storage::partition(
            temporary_storage,
            storage_size,
            detail::temp_storage::make_linear_partition(
                detail::temp_storage::ptr_aligned_array(&temp_keys, size),
                detail::temp_storage::ptr_aligned_array(&temp_values, with_values ? size : 0),
                detail::temp_storage::make_partition(&scratch_storage, scratch_storage_size))));
        // Return temporary storage and early exit on no-ops.
        if(temporary_storage == nullptr || segments == 0 || size == 0 || K == 0)
        {
            return hipSuccess;
        }

        // Execute segmented radix sort.
        ROCPRIM_RETURN_ON_ERROR(do_segmented_radix_sort());

        // Copy the relevant results from the sorted buffer to the output.
        // We use batch copy instead of normal copy through identity transform.
        ROCPRIM_RETURN_ON_ERROR(do_copy_keys());
        if constexpr(with_values)
        {
            ROCPRIM_RETURN_ON_ERROR(do_copy_vals());
        }

        return hipSuccess;
    }
};

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_TOPK_HPP_
