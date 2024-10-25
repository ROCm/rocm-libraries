// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_NTH_ELEMENT_HPP_
#define ROCPRIM_DEVICE_DEVICE_NTH_ELEMENT_HPP_

#include "detail/device_nth_element.hpp"

#include "../detail/temp_storage.hpp"

#include "../config.hpp"

#include "config_types.hpp"
#include "device_nth_element_config.hpp"
#include "device_transform.hpp"

#include <iostream>
#include <iterator>

#include <cstddef>
#include <cstdio>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Config, class KeysIterator, class BinaryFunction>
ROCPRIM_INLINE
hipError_t
    nth_element_impl(void*                                                    temporary_storage,
                     size_t&                                                  storage_size,
                     KeysIterator                                             keys,
                     size_t                                                   nth,
                     size_t                                                   size,
                     BinaryFunction                                           compare_function,
                     hipStream_t                                              stream,
                     bool                                                     debug_synchronous,
                     typename std::iterator_traits<KeysIterator>::value_type* keys_double_buffer
                     = nullptr)
{
    using key_type = typename std::iterator_traits<KeysIterator>::value_type;
    using config   = wrapped_nth_element_config<Config, key_type>;

    target_arch target_arch;
    hipError_t  result = host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }
    const nth_element_config_params params = dispatch_target_arch<config>(target_arch);

    constexpr unsigned int num_partitions        = 3;
    const unsigned int     num_buckets           = params.number_of_buckets;
    const unsigned int     num_splitters         = num_buckets - 1;
    const unsigned int     stop_recursion_size   = params.stop_recursion_size;
    const unsigned int     num_items_per_threads = params.kernel_config.items_per_thread;
    const unsigned int     num_threads_per_block = params.kernel_config.block_size;
    const unsigned int     num_items_per_block   = num_threads_per_block * num_items_per_threads;
    const unsigned int     num_blocks            = ceiling_div(size, num_items_per_block);

    key_type*                            tree             = nullptr;
    size_t*                              buckets          = nullptr;
    n_th_element_iteration_data*         nth_element_data = nullptr;
    bool*                                equality_buckets = nullptr;
    nth_element_onesweep_lookback_state* lookback_states  = nullptr;

    key_type* keys_buffer = nullptr;

    {
        using namespace temp_storage;

        hipError_t partition_result;
        if(keys_double_buffer == nullptr)
        {
            partition_result
                = partition(temporary_storage,
                            storage_size,
                            make_linear_partition(
                                ptr_aligned_array(&tree, num_splitters),
                                ptr_aligned_array(&equality_buckets, num_buckets),
                                ptr_aligned_array(&buckets, num_buckets),
                                ptr_aligned_array(&keys_buffer, size),
                                ptr_aligned_array(&nth_element_data, 1),
                                ptr_aligned_array(&lookback_states, num_partitions * num_blocks)));
        }
        else
        {
            partition_result
                = partition(temporary_storage,
                            storage_size,
                            make_linear_partition(
                                ptr_aligned_array(&tree, num_splitters),
                                ptr_aligned_array(&equality_buckets, num_buckets),
                                ptr_aligned_array(&buckets, num_buckets),
                                ptr_aligned_array(&nth_element_data, 1),
                                ptr_aligned_array(&lookback_states, num_partitions * num_blocks)));
            keys_buffer = keys_double_buffer;
        }

        if(partition_result != hipSuccess || temporary_storage == nullptr)
        {
            return partition_result;
        }
    }

    if((size == 0) || (size == 1 && nth == 0))
    {
        return hipSuccess;
    }

    if(nth >= size)
    {
        return hipErrorInvalidValue;
    }

    if(debug_synchronous)
    {
        std::cout << "-----" << '\n';
        std::cout << "size: " << size << '\n';
        std::cout << "num_buckets: " << num_buckets << '\n';
        std::cout << "num_threads_per_block: " << num_threads_per_block << '\n';
        std::cout << "num_blocks: " << num_blocks << '\n';
        std::cout << "storage_size: " << storage_size << '\n';
    }

    return nth_element_keys_impl<config, num_partitions>(keys,
                                                         keys_buffer,
                                                         tree,
                                                         nth,
                                                         size,
                                                         buckets,
                                                         equality_buckets,
                                                         lookback_states,
                                                         num_buckets,
                                                         stop_recursion_size,
                                                         num_threads_per_block,
                                                         num_items_per_threads,
                                                         nth_element_data,
                                                         compare_function,
                                                         stream,
                                                         debug_synchronous);
}

} // namespace detail

/// \addtogroup devicemodule
/// @{

/// \brief Rearrange elements smaller than the n-th before and bigger than n-th after the n-th element.
///
/// The element at index `n` is set to the element that would be at the n-th position if the input was sorted.
///   Additionally the other elements are rearranged such that for all values of `i` in [keys_output, keys_output + n)
///   and all values of `j` in [keys_output + n, keys_output + size): `comp(*i, *j)` is false.
///   Smaller elements than the n-th will be arranged before, and bigger ones after the n-th element.
///
/// \par Overview
/// * The contents of the inputs are not altered by the function.
/// * Returns the required size of `temporary_storage` in `storage_size`
/// if `temporary_storage` is a null pointer.
/// * Accepts custom compare_functions for nth_element across the device.
/// * Streams in graph capture mode are not supported
///
/// \par Stability
/// \p nth_element is <b>not stable</b>: it doesn't necessarily preserve the relative ordering
/// of equivalent keys.
/// That is, given two keys \p a and \p b and a binary boolean operation \p op such that:
///   * \p a precedes \p b in the input keys, and
///   * op(a, b) and op(b, a) are both false,
/// then it is <b>not guaranteed</b> that \p a will precede \p b as well in the output keys.
///
/// \tparam Config [optional] configuration of the primitive. It has to be `nth_element_config`.
/// \tparam KeysIterator [inferred] random-access iterator type of the input range. Must meet the
///   requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam CompareFunction [inferred] Type of binary function that accepts two arguments of the
///   type `KeysIterator` and returns a value convertible to bool. Default type is `::rocprim::less<>.`
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
///   a null pointer is passed, the required allocation size (in bytes) is written to
/// `storage_size` and function returns without performing the nth_element rearrangement.
/// \param [in,out] storage_size reference to a size (in bytes) of `temporary_storage`.
/// \param [in,out] keys iterator to the input range.
/// \param [in] nth The index of the nth_element in the input range.
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
///
/// \returns `hipSuccess` (`0`) after successful rearrangement; otherwise a HIP runtime error of
///   type `hipError_t`.
///
/// \par Example
/// \parblock
/// In this example a device-level nth_element is performed where input keys are
///   represented by an array of unsigned integers.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// size_t nth;                 // e.g., 4
/// unsigned int * keys;  // e.g., [ 6, 3, 5, 4, 1, 8, 2, 7 ]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::nth_element(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, nth, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform nth_element
/// rocprim::nth_element(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, nth, input_size
/// );
/// // possible keys:   [ 1, 3, 4, 2, 5, 8, 7, 6 ]
/// \endcode
/// \endparblock
template<class Config = default_config,
         class KeysIterator,
         class BinaryFunction
         = ::rocprim::less<typename std::iterator_traits<KeysIterator>::value_type>>
ROCPRIM_INLINE hipError_t nth_element(void*          temporary_storage,
                                      size_t&        storage_size,
                                      KeysIterator   keys,
                                      size_t         nth,
                                      size_t         size,
                                      BinaryFunction compare_function  = BinaryFunction(),
                                      hipStream_t    stream            = 0,
                                      bool           debug_synchronous = false)
{
    return detail::nth_element_impl<Config>(temporary_storage,
                                            storage_size,
                                            keys,
                                            nth,
                                            size,
                                            compare_function,
                                            stream,
                                            debug_synchronous,
                                            nullptr);
}

/// \brief Rearrange elements smaller than the n-th before and bigger than n-th after the n-th element.
///
/// The element at index `n` is set to the element that would be at the n-th position if the input was sorted.
///   Additionally the other elements are rearranged such that for all values of `i` in [keys_output, keys_output + n)
///   and all values of `j` in [keys_output + n, keys_output + size): `comp(*i, *j)` is false.
///   Smaller elements than the n-th will be arranged before, and bigger ones after the n-th element.
///
/// \par Overview
/// * The contents of the inputs are not altered by the function.
/// * Returns the required size of `temporary_storage` in `storage_size`
/// if `temporary_storage` is a null pointer.
/// * Accepts custom compare_functions for nth_element across the device.
/// * Streams in graph capture mode are not supported
///
/// \par Stability
/// \p nth_element is <b>not stable</b>: it doesn't necessarily preserve the relative ordering
/// of equivalent keys.
/// That is, given two keys \p a and \p b and a binary boolean operation \p op such that:
///   * \p a precedes \p b in the input keys, and
///   * op(a, b) and op(b, a) are both false,
/// then it is <b>not guaranteed</b> that \p a will precede \p b as well in the output keys.
///
/// \tparam Config [optional] configuration of the primitive. It has to be `nth_element_config`.
/// \tparam KeysInputIterator [inferred] random-access iterator type of the input range. Must meet the
///   requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator [inferred] random-access iterator type of the output range. Must meet the
///   requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam CompareFunction [inferred] Type of binary function that accepts two arguments of the
///   type `KeysIterator` and returns a value convertible to bool. Default type is `::rocprim::less<>.`
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
///   a null pointer is passed, the required allocation size (in bytes) is written to
/// `storage_size` and function returns without performing the nth_element rearrangement.
/// \param [in,out] storage_size reference to a size (in bytes) of `temporary_storage`.
/// \param [in] keys_input iterator to the input range.
/// \param [out] keys_output iterator to the output range. No overlap at all is allowed between `keys_input` and `keys_output`.
///   `keys_output` should be able to be written and read from for `size` elements.
/// \param [in] nth The index of the nth_element in the input range.
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
///
/// \returns `hipSuccess` (`0`) after successful rearrangement; otherwise a HIP runtime error of
///   type `hipError_t`.
///
/// \par Example
/// \parblock
/// In this example a device-level nth_element is performed where input keys are
///   represented by an array of unsigned integers.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// size_t nth;                 // e.g., 4
/// unsigned int * keys_input;  // e.g., [ 6, 3, 5, 4, 1, 8, 2, 7 ]
/// unsigned int * keys_output; // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::nth_element(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, nth, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform nth_element
/// rocprim::nth_element(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, nth, input_size
/// );
/// // possible keys_output:   [ 1, 3, 4, 2, 5, 8, 7, 6 ]
/// \endcode
/// \endparblock
template<class Config = default_config,
         class KeysInputIterator,
         class KeysOutputIterator,
         class BinaryFunction
         = ::rocprim::less<typename std::iterator_traits<KeysInputIterator>::value_type>>
ROCPRIM_INLINE hipError_t nth_element(void*              temporary_storage,
                                      size_t&            storage_size,
                                      KeysInputIterator  keys_input,
                                      KeysOutputIterator keys_output,
                                      size_t             nth,
                                      size_t             size,
                                      BinaryFunction     compare_function  = BinaryFunction(),
                                      hipStream_t        stream            = 0,
                                      bool               debug_synchronous = false)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    static_assert(
        std::is_same<key_type,
                     typename std::iterator_traits<KeysOutputIterator>::value_type>::value,
        "KeysInputIterator and KeysOutputIterator must have the same value_type");

    if(temporary_storage != nullptr)
    {
        hipError_t error = transform(keys_input,
                                     keys_output,
                                     size,
                                     ::rocprim::identity<key_type>(),
                                     stream,
                                     debug_synchronous);
        if(error != hipSuccess)
        {
            return error;
        }
    }
    return nth_element<Config>(temporary_storage,
                               storage_size,
                               keys_output,
                               nth,
                               size,
                               compare_function,
                               stream,
                               debug_synchronous);
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_NTH_ELEMENT_HPP_
