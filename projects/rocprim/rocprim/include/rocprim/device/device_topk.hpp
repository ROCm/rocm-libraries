// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_TOPK_HPP_
#define ROCPRIM_DEVICE_DEVICE_TOPK_HPP_

#include "../detail/temp_storage.hpp"
#include "detail/device_topk_air_topk.hpp"

#include "device_merge_sort.hpp"
#include "device_radix_sort.hpp"
#include "device_transform.hpp"

#include <iterator>
#include <type_traits>

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

template<typename KeysInputIterator, typename BinaryFunction, typename Decomposer>
struct radix_topk_condition_checker
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;

    static constexpr bool is_custom_decomposer
        = !std::is_same<Decomposer, rocprim::identity_decomposer>::value;
    static constexpr bool descending
        = std::is_same<BinaryFunction, rocprim::greater<key_type>>::value
          || std::is_same<BinaryFunction, rocprim::greater<void>>::value;
    static constexpr bool ascending = std::is_same<BinaryFunction, rocprim::less<key_type>>::value
                                      || std::is_same<BinaryFunction, rocprim::less<void>>::value;
    static constexpr bool is_radix_key_fundamental
        = rocprim::traits::radix_key_codec::radix_key_fundamental<key_type>::value;
    static constexpr bool use_radix
        = (is_radix_key_fundamental || is_custom_decomposer) && (descending || ascending);
};

// Primary template for TopKImpl, assumes default topk_impl_algorithm
template<bool UseRadix,
         class config,
         bool Ordered,
         bool Deterministic,
         bool Stable,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class SizeIn,
         class SizeOut,
         class BinaryFunction,
         class Decomposer>
struct TopKImpl
{
    static ROCPRIM_INLINE
    hipError_t algo_impl(void*                      temporary_storage,
                         size_t&                    storage_size,
                         const KeysInputIterator    keys_input,
                         const KeysOutputIterator   keys_output,
                         const ValuesInputIterator  values_input,
                         const ValuesOutputIterator values_output,
                         const SizeIn               size,
                         const SizeOut              K,
                         const hipStream_t          stream,
                         const bool                 debug_synchronous,
                         const BinaryFunction /*compare_function*/,
                         const Decomposer decomposer = {})
    {
        // Default is radix_topk, check we can actually use it
        using radix_checker
            = radix_topk_condition_checker<KeysInputIterator, BinaryFunction, Decomposer>;
        static_assert(UseRadix && radix_checker::use_radix,
                      "Parameters for TopK implementation RadixTopK are not valid!");

        // Check implementation properties
        static_assert(!radix_checker::is_custom_decomposer,
                      "RadixTopK does not support custom keys");
        static_assert(Ordered == false, "Radix TopK does not support ordered output");
        static_assert(Deterministic == false, "Radix TopK does not support determinism");

        if constexpr(Stable)
        {
            bool ignored;
            // Radix sort needs keys inplace
            auto ret = detail::radix_sort_impl<config, radix_checker::descending>(
                temporary_storage,
                storage_size,
                keys_input,
                nullptr,
                keys_input,
                values_input,
                nullptr,
                values_input,
                size,
                ignored,
                decomposer,
                0,
                sizeof(typename std::iterator_traits<KeysInputIterator>::value_type) * 8,
                stream,
                false,
                debug_synchronous);
            if(ret != hipSuccess)
            {
                return ret;
            }
            ret              = transform(keys_input,
                            keys_output,
                            K,
                            ::rocprim::identity<>(),
                            stream,
                            debug_synchronous);
            using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
            // TODO: need also check if input is nullptr, this can be done in the api function
            // Only pass empty type into this function
            static constexpr bool with_values
                = !std::is_same<value_type, rocprim::empty_type>::value;
            if constexpr(with_values)
            {
                if(ret != hipSuccess)
                {
                    return ret;
                }
                return transform(values_input,
                                 values_output,
                                 K,
                                 ::rocprim::identity<>(),
                                 stream,
                                 debug_synchronous);
            }
            else
            {
                return ret;
            }
        }
        else
        {
            // TODO: Launch plan need to be added
            using topk = rocprim::detail::device_air_topk_impl<256,
                                                               4,
                                                               8,
                                                               radix_checker::ascending,
                                                               KeysInputIterator,
                                                               KeysOutputIterator,
                                                               ValuesInputIterator,
                                                               ValuesOutputIterator,
                                                               SizeIn,
                                                               SizeOut,
                                                               Decomposer>;
            return topk{}(temporary_storage,
                          storage_size,
                          keys_input,
                          keys_output,
                          values_input,
                          values_output,
                          size,
                          K,
                          decomposer,
                          stream,
                          debug_synchronous);
        }
    }
};

template<bool UseRadix,
         class Config,
         bool Ordered,
         bool Deterministic,
         bool Stable,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class SizeIn,
         class SizeOut,
         class BinaryFunction,
         class Decomposer>
ROCPRIM_INLINE
hipError_t topk_impl(void*                      temporary_storage,
                     size_t&                    storage_size,
                     const KeysInputIterator    keys_input,
                     const KeysOutputIterator   keys_output,
                     const ValuesInputIterator  values_input,
                     const ValuesOutputIterator values_output,
                     const SizeIn               size,
                     SizeOut                    K,
                     const BinaryFunction       compare_function  = BinaryFunction(),
                     const Decomposer           decomposer        = {},
                     const hipStream_t          stream            = 0,
                     const bool                 debug_synchronous = false)
{
    using key_type      = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type    = typename std::iterator_traits<ValuesInputIterator>::value_type;
    using common_size_t = typename std::common_type<decltype(size), decltype(K)>::type;
    static_assert(std::is_integral<common_size_t>::value, "Size and K must be integral types.");
    static_assert(
        std::is_same<key_type,
                     typename std::iterator_traits<KeysOutputIterator>::value_type>::value,
        "KeysInputIterator and KeysOutputIterator must have the same value_type");
    static_assert(
        std::is_same<value_type,
                     typename std::iterator_traits<ValuesOutputIterator>::value_type>::value,
        "ValuesInputIterator and ValuesOutputIterator must have the same value_type");

    // Limit K to size
    if(K < 0)
    {
        return hipErrorInvalidValue;
    }
    K = static_cast<SizeOut>(std::min(common_size_t{K}, static_cast<common_size_t>(size)));

    if(temporary_storage == nullptr)
    {
        return detail::TopKImpl<UseRadix,
                                Config,
                                Ordered,
                                Deterministic,
                                Stable,
                                KeysInputIterator,
                                KeysOutputIterator,
                                ValuesInputIterator,
                                ValuesOutputIterator,
                                SizeIn,
                                SizeOut,
                                BinaryFunction,
                                Decomposer>::algo_impl(temporary_storage,
                                                       storage_size,
                                                       keys_input,
                                                       keys_output,
                                                       values_input,
                                                       values_output,
                                                       size,
                                                       K,
                                                       stream,
                                                       debug_synchronous,
                                                       compare_function,
                                                       decomposer);
    }

    // Start point for time measurements
    std::chrono::steady_clock::time_point start;
    if(debug_synchronous)
    {
        start = std::chrono::steady_clock::now();
    }

    ROCPRIM_RETURN_ON_ERROR(detail::TopKImpl<UseRadix,
                                             Config,
                                             Ordered,
                                             Deterministic,
                                             Stable,
                                             KeysInputIterator,
                                             KeysOutputIterator,
                                             ValuesInputIterator,
                                             ValuesOutputIterator,
                                             SizeIn,
                                             SizeOut,
                                             BinaryFunction,
                                             Decomposer>::algo_impl(temporary_storage,
                                                                    storage_size,
                                                                    keys_input,
                                                                    keys_output,
                                                                    values_input,
                                                                    values_output,
                                                                    size,
                                                                    K,
                                                                    stream,
                                                                    debug_synchronous,
                                                                    compare_function,
                                                                    decomposer));

    return hipSuccess;
}

} // namespace detail

/// \brief Find the largest/smallest K elements from an input array of keys.
///
/// The K elements are returned within the K first positions of the output array and in a non specific order.
template<class Config       = default_config,
         bool Descending    = false,
         bool Ordered       = false,
         bool Deterministic = false,
         bool Stable        = false,
         class Decomposer   = ::rocprim::identity_decomposer,
         class KeysInputIterator,
         class KeysOutputIterator,
         class SizeIn,
         class SizeOut>
ROCPRIM_INLINE
hipError_t topk(void*                    temporary_storage,
                size_t&                  storage_size,
                const KeysInputIterator  keys_input,
                const KeysOutputIterator keys_output,
                const SizeIn             size,
                const SizeOut            K,
                Decomposer               decomposer        = {},
                const hipStream_t        stream            = 0,
                const bool               debug_synchronous = false)
{
    using compare_function = std::conditional_t<
        Descending,
        rocprim::greater<typename std::iterator_traits<KeysInputIterator>::value_type>,
        rocprim::less<typename std::iterator_traits<KeysInputIterator>::value_type>>;
    return detail::topk_impl<true, Config, Ordered, Deterministic, Stable>(
        temporary_storage,
        storage_size,
        keys_input,
        keys_output,
        static_cast<empty_type*>(nullptr),
        static_cast<empty_type*>(nullptr),
        size,
        K,
        compare_function(),
        decomposer,
        stream,
        debug_synchronous);
}

/// \brief Find the largest/smallest K elements from an input array of values based on their correspondent keys.
///
/// The K pairs (key, value) are returned within the K first positions of the output keys and values arrays,
/// and in a non specific order.
template<class Config       = default_config,
         bool Descending    = false,
         bool Ordered       = false,
         bool Deterministic = false,
         bool Stable        = false,
         class Decomposer   = rocprim::identity_decomposer,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class SizeIn,
         class SizeOut>
ROCPRIM_INLINE
hipError_t topk_pairs(void*                      temporary_storage,
                      size_t&                    storage_size,
                      const KeysInputIterator    keys_input,
                      const KeysOutputIterator   keys_output,
                      const ValuesInputIterator  values_input,
                      const ValuesOutputIterator values_output,
                      const SizeIn               size,
                      const SizeOut              K,
                      const Decomposer           decomposer        = {},
                      const hipStream_t          stream            = 0,
                      const bool                 debug_synchronous = false)
{
    using compare_function = std::conditional_t<
        Descending,
        rocprim::greater<typename std::iterator_traits<KeysInputIterator>::value_type>,
        rocprim::less<typename std::iterator_traits<KeysInputIterator>::value_type>>;
    return detail::topk_impl<true, Config, Ordered, Deterministic, Stable>(temporary_storage,
                                                                           storage_size,
                                                                           keys_input,
                                                                           keys_output,
                                                                           values_input,
                                                                           values_output,
                                                                           size,
                                                                           K,
                                                                           compare_function(),
                                                                           decomposer,
                                                                           stream,
                                                                           debug_synchronous);
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group devicemodule

#endif // ROCPRIM_DEVICE_DEVICE_TOPK_HPP_
