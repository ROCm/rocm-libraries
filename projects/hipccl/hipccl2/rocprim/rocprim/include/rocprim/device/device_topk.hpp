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

#include "../common.hpp"
#include "../config.hpp"
#include "../detail/temp_storage.hpp"
#include "detail/device_topk_air.hpp"

#include "device_merge_sort.hpp"
#include "device_radix_sort.hpp"
#include "device_transform.hpp"

#include <iterator>
#include <type_traits>

/// \addtogroup devicemodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

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
            return rocprim::detail::device_topk_air<config, radix_checker::ascending>(
                temporary_storage,
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

#endif // DOXYGEN_SHOULD_SKIP_THIS

/// \brief Find the largest or smallest K elements from an input array of keys.
///
/// Returns the K largest or smallest elements. These maybe be in any order.
///
/// A similar algorithm is nth_element. The main differences are:
/// * topk returns arrays of size `k`, whereas nth_element returns an array matching the input size.
/// * The element at index `n` is set to the element that would be at the n-th position in nth_element, while
///   topk does not guarantee that.
/// * Elements which are smaller than the n-th element will be placed in the front of the output, while topk ignores
///   all elements which are smaller or larger than the k-th element (depending on descending or ascending).
/// * `In-place` operation can be done by nth_element, while topk does not support that.
/// * topk supports hipGraph, but nth_element does not.
///
/// Here are some tips for choosing between nth_element and topk:
/// * If you need the n-th largest or smallest element to be placed at index `n`, please use
///   nth_element.
/// * If you want to care about both smaller elements and larger elements, please use nth_element.
/// * If you want to use hipGraph, please use topk.
///
/// \tparam Config [optional] configuration of the primitive, must be `default_config` or `topk_config`.
/// \tparam Descending [optional] determines the starting direction. If \p true, select the largest K elements, and vice versa.
/// \tparam Ordered [optional] determines whether the output results are sorted by size.
/// \note Ordered output is not supported yet. This feature will be added in the future.
/// \tparam Deterministic [optional] to ensure that the results are exactly the same every time.
/// \note Deterministic output is not supported yet. This feature will be added in the future.
/// \tparam Stable [optional] determines whether elements in the output are arranged according to their relative position in the input.
/// \note Stable output is not supported yet. This feature will be added in the future.
/// \tparam Decomposer [optional] the type of the decomposer functor.
/// \tparam KeysInputIterator [optional] random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator [optional] random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam SizeIn [optional] integral type that represents the problem size.
/// \tparam SizeOut [optional] integral type that counts the number of output elements.
///
/// \param [in] temporary_storage pointer to device-accessible temporary storage.
/// If a null pointer is provided, the required allocation size (in bytes) is written
/// to \p storage_size, and the function returns without performing the sort operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input pointer to the first element in the input range.
/// \param [out] keys_output pointer to the first element in the output range.
/// \param [in] size number of elements in the input range.
/// \param [in] K number of elements to be selected from the input.
/// \param [in] decomposer decomposer functor that produces a tuple of references from the
/// input key type.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] if \p true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after a successful top-k operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending top-k is performed on an array.
///
/// The full example is [on GitHub](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocprim/example/rocprim/device/example_device_topk.cpp).
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
/// // Prepare input and output (declare pointers, allocate device memory, etc.)
/// size_t input_size;      // e.g., 8
/// size_t k;               // e.g., 3
/// int * input;            // e.g., [2, 3, 4, -1, -2, -3, 0, 5]
/// int * output;           // empty array of 3 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::topk(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size, k
/// );
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
/// // perform topk
/// rocprim::topk(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size, k
/// );
/// // output should be [5, 4, 3] (order is not guaranteed)
/// \endcode
/// \endparblock
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

/// \brief Find the largest or smallest K elements from an input array of values based on their corresponding keys.
///
/// Returns the K largest or smallest (key, value) pairs. These maybe be in any order.
///
/// A similar algorithm is nth_element. The main differences are:
/// * topk returns arrays of size `k`, whereas nth_element returns an array matching the input size.
/// * The element at index `n` is set to the element that would be at the n-th position in nth_element, while
///   topk does not guarantee that.
/// * Elements which are smaller than the n-th element will be placed in the front of the output, while topk ignores
///   all elements which are smaller or larger than the k-th element (depending on descending or ascending).
/// * `In-place` operation can be done by nth_element, while topk does not support that.
/// * topk supports hipGraph, but nth_element does not.
///
/// Here are some tips for choosing between nth_element and topk:
/// * If you need the n-th largest or smallest element to be placed at index `n`, please use
///   nth_element.
/// * If you want to care about both smaller elements and larger elements, please use nth_element.
/// * If you want to use hipGraph, please use topk.
///
/// \tparam Config [optional] configuration of the primitive, must be `default_config` or `topk_config`.
/// \tparam Descending [optional] determines the starting direction. If \p true, select the largest K elements, and vice versa.
/// \tparam Ordered [optional] determines whether the output results are sorted by size.
/// \note Ordered output is not supported yet. This feature will be added in the future.
/// \tparam Deterministic [optional] to ensure that the results are exactly the same every time.
/// \note Deterministic output is not supported yet. This feature will be added in the future.
/// \tparam Stable [optional] determines whether elements in the output are arranged according to their relative position in the input.
/// \note Stable output is not supported yet. This feature will be added in the future.
/// \tparam Decomposer [optional] the type of the decomposer functor.
/// \tparam KeysInputIterator [optional] random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator [optional] random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam ValuesInputIterator [optional] random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam ValuesOutputIterator [optional] random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam SizeIn [optional] integral type that represents the problem size.
/// \tparam SizeOut [optional] integral type that counts the number of output elements.
///
/// \param [in] temporary_storage pointer to device-accessible temporary storage.
/// If a null pointer is provided, the required allocation size (in bytes) is written
/// to \p storage_size, and the function returns without performing the sort operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input pointer to the first key in the input range.
/// \param [out] keys_output pointer to the first key in the output range.
/// \param [in] values_input pointer to the first value in the output range.
/// \param [out] values_output pointer to the first value in the output range.
/// \param [in] size number of elements in the input range.
/// \param [in] K number of elements to be selected from the input.
/// \param [in] decomposer decomposer functor that produces a tuple of references from the
/// input key type.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] if \p true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after a successful top-k operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending top-k is performed on an array.
///
/// The full example is [on GitHub](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocprim/example/rocprim/device/example_device_topk.cpp).
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
/// // Prepare input and output (declare pointers, allocate device memory, etc.)
/// size_t input_size;      // e.g., 8
/// size_t k;               // e.g., 3
/// int * input_keys;       // e.g., [2, 3, 4, -1, -2, -3, 0, 5]
/// int * output_keys;      // empty array of 3 elements
/// int * input_vals;       // e.g., [0, 1, 2, 3, 4, 5, 6, 7]
/// int * output_vals;      // empty array of 3 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::topk_pairs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input_keys, output_keys, input_vals, output_vals, input_size, k
/// );
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
/// // perform topk_pairs
/// rocprim::topk_pairs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input_keys, output_keys, input_vals, output_vals, input_size, k
/// );
/// // output_keys should be [5, 4, 3] (order is not guaranteed)
/// // output_vals should be [7, 2, 1] (order matches output_keys)
/// \endcode
/// \endparblock
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
