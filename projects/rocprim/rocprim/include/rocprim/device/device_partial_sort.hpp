// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_PARTIAL_SORT_HPP_
#define ROCPRIM_DEVICE_DEVICE_PARTIAL_SORT_HPP_

#include "detail/device_nth_element.hpp"

#include "../detail/temp_storage.hpp"

#include "../config.hpp"

#include "config_types.hpp"
#include "device_merge_sort.hpp"
#include "device_nth_element.hpp"
#include "device_partial_sort_config.hpp"
#include "device_radix_sort.hpp"
#include "device_transform.hpp"

#include <iostream>
#include <iterator>

#include <cstddef>
#include <cstdio>
#include <type_traits>

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

template<typename KeysInputIterator, typename BinaryFunction, typename Decomposer>
struct radix_sort_condition_checker
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;

    static constexpr bool is_custom_decomposer
        = !std::is_same<Decomposer, rocprim::identity_decomposer>::value;
    static constexpr bool descending
        = std::is_same<BinaryFunction, rocprim::greater<key_type>>::value;
    static constexpr bool ascending = std::is_same<BinaryFunction, rocprim::less<key_type>>::value;
    static constexpr bool is_radix_key_fundamental
        = rocprim::traits::radix_key_codec::radix_key_fundamental<key_type>::value;
    static constexpr bool use_radix_sort
        = (is_radix_key_fundamental || is_custom_decomposer) && (descending || ascending);
};

// Primary template for SortImpl
template<bool use_radix_sort,
         typename config,
         typename KeysInputIterator,
         typename KeysOutputIterator,
         typename BinaryFunction,
         typename Decomposer = rocprim::identity_decomposer,
         typename checker
         = radix_sort_condition_checker<KeysInputIterator, BinaryFunction, Decomposer>>
struct SortImpl
{
    static ROCPRIM_INLINE
    hipError_t algo_sort(void*              temporary_storage,
                         size_t&            storage_size,
                         KeysInputIterator  keys_input,
                         KeysOutputIterator keys_output,
                         const size_t       size,
                         BinaryFunction     compare_function,
                         const hipStream_t  stream,
                         bool               debug_synchronous,
                         typename std::iterator_traits<KeysInputIterator>::value_type* keys_buffer,
                         Decomposer /*decomposer*/)
    {
        // Merge sort implementation
        return detail::merge_sort_impl<typename config::merge_sort>(
            temporary_storage,
            storage_size,
            keys_input,
            keys_output,
            static_cast<empty_type*>(nullptr),
            static_cast<empty_type*>(nullptr),
            size,
            compare_function,
            stream,
            debug_synchronous,
            keys_buffer,
            static_cast<empty_type*>(nullptr),
            true);
    }
};

// Specialization for radix sort
template<typename config,
         typename KeysInputIterator,
         typename KeysOutputIterator,
         typename BinaryFunction,
         typename Decomposer,
         typename checker>
struct SortImpl<true,
                config,
                KeysInputIterator,
                KeysOutputIterator,
                BinaryFunction,
                Decomposer,
                checker>
{
    static ROCPRIM_INLINE
    hipError_t
        algo_sort(void*              temporary_storage,
                  size_t&            storage_size,
                  KeysInputIterator  keys_input,
                  KeysOutputIterator keys_output,
                  const size_t       size,
                  BinaryFunction /*compare_function*/,
                  const hipStream_t stream,
                  bool              debug_synchronous,
                  typename std::iterator_traits<KeysInputIterator>::value_type* /*keys_buffer*/,
                  Decomposer decomposer = {})
    {
        // Radix sort implementation
        bool ignored;
        return detail::radix_sort_impl<typename config::radix_sort, checker::descending>(
            temporary_storage,
            storage_size,
            keys_input,
            nullptr,
            keys_output,
            static_cast<empty_type*>(nullptr),
            static_cast<empty_type*>(nullptr),
            static_cast<empty_type*>(nullptr),
            size,
            ignored,
            decomposer,
            0,
            sizeof(typename std::iterator_traits<KeysInputIterator>::value_type) * 8,
            stream,
            false,
            debug_synchronous);
    }
};

template<bool inplace>
struct partial_sort_nth_element_helper
{
    template<class Config,
             class KeysInputIterator,
             class KeysInputIteratorNthElement,
             class KeysOutputIterator,
             class BinaryFunction,
             class Decomposer = rocprim::identity_decomposer>
    ROCPRIM_INLINE
    hipError_t
        algo_sort_impl(void*             temporary_storage,
                       size_t&           storage_size,
                       KeysInputIterator keys_input,
                       KeysInputIteratorNthElement /*keys_input_nth_element*/,
                       KeysOutputIterator keys_output,
                       const size_t       size,
                       BinaryFunction     compare_function,
                       const hipStream_t  stream,
                       bool               debug_synchronous,
                       typename std::iterator_traits<KeysInputIterator>::value_type* keys_buffer
                       = nullptr,
                       Decomposer decomposer = {})
    {
        using checker = radix_sort_condition_checker<KeysInputIterator, BinaryFunction, Decomposer>;
        return SortImpl<checker::use_radix_sort,
                        Config,
                        KeysInputIterator,
                        KeysOutputIterator,
                        BinaryFunction,
                        Decomposer,
                        checker>::algo_sort(temporary_storage,
                                            storage_size,
                                            keys_input,
                                            keys_output,
                                            size,
                                            compare_function,
                                            stream,
                                            debug_synchronous,
                                            keys_buffer,
                                            decomposer);
    }

    template<class Config, class KeysIterator, class KeysIteratorNthElement, class BinaryFunction>
    ROCPRIM_INLINE
    hipError_t nth_element_impl(
        void*        temporary_storage,
        size_t&      storage_size,
        KeysIterator keys,
        KeysIteratorNthElement /*keys_nth_element*/,
        size_t                                                   nth,
        size_t                                                   size,
        BinaryFunction                                           compare_function,
        hipStream_t                                              stream,
        bool                                                     debug_synchronous,
        typename std::iterator_traits<KeysIterator>::value_type* keys_double_buffer)
    {
        return detail::nth_element_impl<Config>(temporary_storage,
                                                storage_size,
                                                keys,
                                                nth,
                                                size,
                                                compare_function,
                                                stream,
                                                debug_synchronous,
                                                keys_double_buffer);
    }
};

template<>
struct partial_sort_nth_element_helper<false>
{
    template<typename config,
             class KeysInputIterator,
             class KeysInputIteratorNthElement,
             class KeysOutputIterator,
             class BinaryFunction,
             class Decomposer = rocprim::identity_decomposer>
    ROCPRIM_INLINE
    hipError_t
        algo_sort_impl(void*   temporary_storage,
                       size_t& storage_size,
                       KeysInputIterator /*keys_input*/,
                       KeysInputIteratorNthElement keys_input_nth_element,
                       KeysOutputIterator          keys_output,
                       const size_t                size,
                       BinaryFunction              compare_function,
                       const hipStream_t           stream,
                       bool                        debug_synchronous,
                       typename std::iterator_traits<KeysInputIterator>::value_type* keys_buffer
                       = nullptr,
                       Decomposer decomposer = {})
    {
        using checker = radix_sort_condition_checker<KeysInputIterator, BinaryFunction, Decomposer>;
        return SortImpl<checker::use_radix_sort,
                        config,
                        KeysInputIteratorNthElement,
                        KeysOutputIterator,
                        BinaryFunction,
                        Decomposer>::algo_sort(temporary_storage,
                                               storage_size,
                                               keys_input_nth_element,
                                               keys_output,
                                               size,
                                               compare_function,
                                               stream,
                                               debug_synchronous,
                                               keys_buffer,
                                               decomposer);
    }

    template<class Config, class KeysIterator, class KeysIteratorNthElement, class BinaryFunction>
    ROCPRIM_INLINE
    hipError_t nth_element_impl(
        void*   temporary_storage,
        size_t& storage_size,
        KeysIterator /*keys*/,
        KeysIteratorNthElement                                   keys_nth_element,
        size_t                                                   nth,
        size_t                                                   size,
        BinaryFunction                                           compare_function,
        hipStream_t                                              stream,
        bool                                                     debug_synchronous,
        typename std::iterator_traits<KeysIterator>::value_type* keys_double_buffer)
    {
        return detail::nth_element_impl<Config>(temporary_storage,
                                                storage_size,
                                                keys_nth_element,
                                                nth,
                                                size,
                                                compare_function,
                                                stream,
                                                debug_synchronous,
                                                keys_double_buffer);
    }
};

template<class Config,
         class KeysInputIterator,
         class KeysOutputIterator,
         class BinaryFunction,
         class Decomposer = rocprim::identity_decomposer,
         bool inplace     = true>
hipError_t partial_sort_impl(void*              temporary_storage,
                             size_t&            storage_size,
                             KeysInputIterator  keys_in,
                             KeysOutputIterator keys_out,
                             size_t             middle,
                             size_t             size,
                             BinaryFunction     compare_function,
                             hipStream_t        stream,
                             bool               debug_synchronous,
                             Decomposer         decomposer = {})
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using input_reference_type = typename std::iterator_traits<KeysInputIterator>::reference;
    using config = default_or_custom_config<Config, detail::default_partial_sort_config<key_type>>;
    using config_nth_element = typename config::nth_element;

    static_assert(!std::is_const<std::remove_reference_t<input_reference_type>>::value || !inplace,
                  "Key input iterator must be mutable with in-place partial sort");

    if(size != 0 && middle >= size)
    {
        return hipErrorInvalidValue;
    }

    partial_sort_nth_element_helper<inplace> helper;

    size_t storage_size_nth_element{};
    // non-null placeholder so that no buffer is allocated for keys
    key_type* keys_buffer_placeholder = reinterpret_cast<key_type*>(1);

    void*     temporary_storage_nth_element = nullptr;
    void*     temporary_storage_algo_sort   = nullptr;
    key_type* keys_buffer                   = nullptr;
    key_type* keys_output_nth_element       = nullptr;

    const bool full_sort = middle + 1 == size;
    if(!full_sort)
    {
        ROCPRIM_RETURN_ON_ERROR(
            helper.template nth_element_impl<config_nth_element>(nullptr,
                                                                 storage_size_nth_element,
                                                                 keys_in,
                                                                 keys_output_nth_element,
                                                                 middle,
                                                                 size,
                                                                 compare_function,
                                                                 stream,
                                                                 debug_synchronous,
                                                                 keys_buffer_placeholder));
    }
    size_t storage_size_algo_sort{};

    ROCPRIM_RETURN_ON_ERROR(
        helper.template algo_sort_impl<config>(nullptr,
                                               storage_size_algo_sort,
                                               keys_in,
                                               keys_output_nth_element,
                                               keys_out,
                                               (!inplace || full_sort) ? middle + 1 : middle,
                                               compare_function,
                                               stream,
                                               debug_synchronous,
                                               keys_buffer_placeholder, // keys_buffer
                                               decomposer));

    const hipError_t partition_result = temp_storage::partition(
        temporary_storage,
        storage_size,
        temp_storage::make_linear_partition(
            temp_storage::ptr_aligned_array(&keys_buffer, size),
            temp_storage::ptr_aligned_array(&keys_output_nth_element, inplace ? 0 : size),
            temp_storage::make_partition(&temporary_storage_nth_element, storage_size_nth_element),
            temp_storage::make_partition(&temporary_storage_algo_sort, storage_size_algo_sort)));

    if(partition_result != hipSuccess || temporary_storage == nullptr)
    {
        return partition_result;
    }

    if(size == 0)
    {
        return hipSuccess;
    }

    if(!inplace)
    {
        ROCPRIM_RETURN_ON_ERROR(transform(keys_in,
                                          keys_output_nth_element,
                                          size,
                                          rocprim::identity<key_type>(),
                                          stream,
                                          debug_synchronous));
    }

    if(!full_sort)
    {
        ROCPRIM_RETURN_ON_ERROR(
            helper.template nth_element_impl<config_nth_element>(temporary_storage_nth_element,
                                                                 storage_size_nth_element,
                                                                 keys_in,
                                                                 keys_output_nth_element,
                                                                 middle,
                                                                 size,
                                                                 compare_function,
                                                                 stream,
                                                                 debug_synchronous,
                                                                 keys_buffer));
    }

    if(middle == 0)
    {
        if(!inplace)
        {
            ROCPRIM_RETURN_ON_ERROR(transform(keys_output_nth_element,
                                              keys_out,
                                              1,
                                              rocprim::identity<key_type>(),
                                              stream,
                                              debug_synchronous));
        }
        return hipSuccess;
    }

    return helper.template algo_sort_impl<config>(temporary_storage_algo_sort,
                                                  storage_size_algo_sort,
                                                  keys_in,
                                                  keys_output_nth_element,
                                                  keys_out,
                                                  (!inplace || full_sort) ? middle + 1 : middle,
                                                  compare_function,
                                                  stream,
                                                  debug_synchronous,
                                                  keys_buffer, // keys_buffer
                                                  decomposer);
}

} // namespace detail

/// \brief Rearranges elements such that the range [0, middle) contains the sorted middle smallest elements in the range [0, size).
///
/// \par Overview
/// * The contents of the inputs are not altered by the function.
/// * Returns the required size of `temporary_storage` in `storage_size`
/// if `temporary_storage` is a null pointer.
/// * Accepts custom compare_functions for partial_sort_copy across the device.
/// * Streams in graph capture mode are not supported
/// * When possible, partial_sort_copy will use radix_sort as the sorting algorithm. If radix sort is not applicable, it will fall back to merge_sort.
/// If a custom decomposer is provided, partial_sort_copy will use radix_sort.
///
/// \par Stability
/// \p partial_sort_copy is <b>not stable</b>: it doesn't necessarily preserve the relative ordering
/// of equivalent keys.
/// That is, given two keys \p a and \p b and a binary boolean operation \p op such that:
///   * \p a precedes \p b in the input keys, and
///   * op(a, b) and op(b, a) are both false,
/// then it is <b>not guaranteed</b> that \p a will precede \p b as well in the output
/// (ordered) keys.
///
/// \tparam Config [optional] configuration of the primitive. It has to be `partial_sort_config`.
/// \tparam KeysInputIterator [inferred] random-access iterator type of the input range. Must meet the
///   requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator [inferred] random-access iterator type of the output range. Must meet the
///   requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam CompareFunction [inferred] Type of binary function that accepts two arguments of the
///   type `KeysIterator` and returns a value convertible to bool. Default type is `::rocprim::less<>.`
/// \tparam Decomposer The type of the decomposer functor. Default is ::rocprim::identity_decomposer.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
///   a null pointer is passed, the required allocation size (in bytes) is written to
/// `storage_size` and function returns without performing the partial_sort_copy rearrangement.
/// \param [in,out] storage_size reference to a size (in bytes) of `temporary_storage`.
/// \param [in] keys_input iterator to the input range.
/// \param [out] keys_output iterator to the output range. No overlap at all is allowed between `keys_input` and `keys_output`.
///   `keys_output` should be able to be write to at least `middle` + 1 elements.
/// \param [in] middle The index of the point till where it is sorted in the input range.
/// \param [in] size number of element in the input range.
/// \param [in] compare_function binary operation function object that will be used for comparison.
///   The signature of the function should be equivalent to the following:
///   <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
///   <tt>const &</tt>, but function object must not modify the objects passed to it.
///   The comparator must meet the C++ named requirement Compare.
///   The default value is `BinaryFunction()`.
/// \param [in] stream [optional] HIP stream object. Default is `0` (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
///   launch is forced in order to check for errors. Default value is `false`.
/// \param [in] decomposer decomposer functor that produces a tuple of references from the
/// input key type.
///
/// \returns `hipSuccess` (`0`) after successful rearrangement; otherwise a HIP runtime error of
///   type `hipError_t`.
///
/// \par Example
/// \parblock
/// In this example a device-level partial_sort_copy is performed where input keys are
///   represented by an array of unsigned integers.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// size_t middle;              // e.g., 4
/// unsigned int * keys_input;  // e.g., [ 6, 3, 5, 4, 1, 8, 2, 7 ]
/// unsigned int * keys_output; // e.g., [ 9, 9, 9, 9, 9, 9, 9, 9 ]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::partial_sort_copy(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, middle, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform partial_sort
/// rocprim::partial_sort_copy(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, middle, input_size
/// );
/// // possible keys_output:   [ 1, 2, 3, 4, 5, 9, 9, 9 ]
/// \endcode
/// \endparblock
template<class Config = default_config,
         class KeysInputIterator,
         class KeysOutputIterator,
         class BinaryFunction
         = ::rocprim::less<typename std::iterator_traits<KeysInputIterator>::value_type>,
         class Decomposer = ::rocprim::identity_decomposer>
hipError_t partial_sort_copy(void*              temporary_storage,
                             size_t&            storage_size,
                             KeysInputIterator  keys_input,
                             KeysOutputIterator keys_output,
                             size_t             middle,
                             size_t             size,
                             BinaryFunction     compare_function  = BinaryFunction(),
                             hipStream_t        stream            = 0,
                             bool               debug_synchronous = false,
                             Decomposer         decomposer        = Decomposer())
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    static_assert(
        std::is_same<key_type,
                     typename std::iterator_traits<KeysOutputIterator>::value_type>::value,
        "KeysInputIterator and KeysOutputIterator must have the same value_type");

    return detail::partial_sort_impl<Config,
                                     KeysInputIterator,
                                     KeysOutputIterator,
                                     BinaryFunction,
                                     Decomposer,
                                     false>(temporary_storage,
                                            storage_size,
                                            keys_input,
                                            keys_output,
                                            middle,
                                            size,
                                            compare_function,
                                            stream,
                                            debug_synchronous,
                                            decomposer);
}

/// \brief Rearranges elements such that the range [0, middle) contains the sorted middle smallest elements in the range [0, size).
///
/// \par Overview
/// * The contents of the inputs are not altered by the function.
/// * Returns the required size of `temporary_storage` in `storage_size`
/// if `temporary_storage` is a null pointer.
/// * Accepts custom compare_functions for partial_sort across the device.
/// * Streams in graph capture mode are not supported
/// * When possible, partial_sort will use radix_sort as the sorting algorithm. If radix sort is not applicable, it will fall back to merge_sort.
/// If a custom decomposer is provided, partial_sort will use radix_sort.
///
/// \par Stability
/// \p partial_sort is <b>not stable</b>: it doesn't necessarily preserve the relative ordering
/// of equivalent keys.
/// That is, given two keys \p a and \p b and a binary boolean operation \p op such that:
///   * \p a precedes \p b in the input keys, and
///   * op(a, b) and op(b, a) are both false,
/// then it is <b>not guaranteed</b> that \p a will precede \p b as well in the output
/// (ordered) keys.
///
/// \tparam Config [optional] configuration of the primitive. It has to be `partial_sort_config`.
/// \tparam KeysIterator [inferred] random-access iterator type of the input range. Must meet the
///   requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam CompareFunction [inferred] Type of binary function that accepts two arguments of the
///   type `KeysIterator` and returns a value convertible to bool. Default type is `::rocprim::less<>.`
/// \tparam Decomposer The type of the decomposer functor. Default is ::rocprim::identity_decomposer.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
///   a null pointer is passed, the required allocation size (in bytes) is written to
/// `storage_size` and function returns without performing the partial_sort rearrangement.
/// \param [in,out] storage_size reference to a size (in bytes) of `temporary_storage`.
/// \param [in,out] keys iterator to the input range.
/// \param [in] middle The index of the point till where it is sorted in the input range.
/// \param [in] size number of element in the input range.
/// \param [in] compare_function binary operation function object that will be used for comparison.
///   The signature of the function should be equivalent to the following:
///   <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
///   <tt>const &</tt>, but function object must not modify the objects passed to it.
///   The comparator must meet the C++ named requirement Compare.
///   The default value is `BinaryFunction()`.
/// \param [in] stream [optional] HIP stream object. Default is `0` (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
///   launch is forced in order to check for errors. Default value is `false`.
/// \param [in] decomposer decomposer functor that produces a tuple of references from the
/// input key type.
///
/// \returns `hipSuccess` (`0`) after successful rearrangement; otherwise a HIP runtime error of
///   type `hipError_t`.
///
/// \par Example
/// \parblock
/// In this example a device-level partial_sort is performed where input keys are
///   represented by an array of unsigned integers.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// size_t middle;              // e.g., 4
/// unsigned int * keys;  // e.g., [ 6, 3, 5, 4, 1, 8, 2, 7 ]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::partial_sort(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, middle, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform partial_sort
/// rocprim::partial_sort(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, middle, input_size
/// );
/// // possible keys:   [ 1, 2, 3, 4, 5, 8, 7, 6 ]
/// \endcode
/// \endparblock
template<class Config = default_config,
         class KeysIterator,
         class BinaryFunction
         = ::rocprim::less<typename std::iterator_traits<KeysIterator>::value_type>,
         class Decomposer = ::rocprim::identity_decomposer>
hipError_t partial_sort(void*          temporary_storage,
                        size_t&        storage_size,
                        KeysIterator   keys,
                        size_t         middle,
                        size_t         size,
                        BinaryFunction compare_function  = BinaryFunction(),
                        hipStream_t    stream            = 0,
                        bool           debug_synchronous = false,
                        Decomposer     decomposer        = {})
{
    return detail::
        partial_sort_impl<Config, KeysIterator, KeysIterator, BinaryFunction, Decomposer, true>(
            temporary_storage,
            storage_size,
            keys,
            keys,
            middle,
            size,
            compare_function,
            stream,
            debug_synchronous,
            decomposer);
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_PARTIAL_SORT_HPP_
