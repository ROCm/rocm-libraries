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

#ifndef ROCPRIM_DEVICE_DEVICE_SEARCH_HPP_
#define ROCPRIM_DEVICE_DEVICE_SEARCH_HPP_

#include "../config.hpp"

#include "config_types.hpp"
#include "detail/device_search.hpp"

#include <iterator>

#include <cstddef>
#include <cstdio>

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

/// \brief Searches for the first occurrence of the sequence.
///
/// Searches the input for the first occurence of a sequence, according to a particular
///   comparison function. If found, the index of the first item of the found sequence
///   in the input is returned. Otherwise, returns the size of the input.
///
/// \par Overview
/// * The contents of the inputs are not altered by the function.
/// * Returns the required size of `temporary_storage` in `storage_size`
/// if `temporary_storage` is a null pointer.
/// * Accepts custom compare_functions for search across the device.
/// * Streams in graph capture mode are supported
///
/// \tparam Config [optional] configuration of the primitive, must be `default_config` or `search_config`.
/// \tparam InputIterator1 [inferred] random-access iterator type of the input range. Must meet the
///   requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam InputIterator2 [inferred] random-access iterator type of the input range. Must meet the
///   requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator [inferred] random-access iterator type of the input range. Must meet the
///   requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction [inferred] Type of binary function that accepts two arguments of the
///   type `InputIterator1` and returns a value convertible to bool.
///   Default type is `rocprim::less<>.`
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
///   a null pointer is passed, the required allocation size (in bytes) is written to
/// `storage_size` and function returns without performing the search.
/// \param [in,out] storage_size reference to a size (in bytes) of `temporary_storage`.
/// \param [in] input iterator to the input range.
/// \param [in] keys iterator to the key range.
/// \param [out] output iterator to the output range. The output is one element.
/// \param [in] size number of elements in the input range.
/// \param [in] keys_size number of elements in the key range.
/// \param [in] compare_function binary operation function object that will be used for comparison.
///   The signature of the function should be equivalent to the following:
///   <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
///   <tt>const &</tt>, but function object must not modify the objects passed to it.
///   The comparator must meet the C++ named requirement BinaryPredicate.
///   The default value is `BinaryFunction()`.
/// \param [in] stream [optional] HIP stream object. Default is `0` (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
///   launch is forced in order to check for errors. Default value is `false`.
///
/// \returns `hipSuccess` (`0`) after successful search; otherwise a HIP runtime error of
///   type `hipError_t`.
///
/// \par Example
/// \parblock
/// In this example a device-level search is performed where input values are
///   represented by an array of unsigned integers and the key is also an array
///   of unsigned integers.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t size;           // e.g., 10
/// size_t key_size;       // e.g., 3
/// unsigned int * input;  // e.g., [ 6, 3, 5, 4, 1, 8, 2, 5, 4, 1 ]
/// unsigned int * key;    // e.g., [ 5, 4, 1 ]
/// unsigned int * output; // e.g., empty array of size 1
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::search(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, key, output, size, key_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform search
/// rocprim::search(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, key, output, size, key_size
/// );
/// // output:   [ 2 ]
/// \endcode
/// \endparblock
template<class Config = default_config,
         class InputIterator1,
         class InputIterator2,
         class OutputIterator,
         class BinaryFunction
         = rocprim::equal_to<typename std::iterator_traits<InputIterator1>::value_type>>
ROCPRIM_INLINE
hipError_t search(void*          temporary_storage,
                  size_t&        storage_size,
                  InputIterator1 input,
                  InputIterator2 keys,
                  OutputIterator output,
                  size_t         size,
                  size_t         keys_size,
                  BinaryFunction compare_function  = BinaryFunction(),
                  hipStream_t    stream            = 0,
                  bool           debug_synchronous = false)
{
    return detail::search_impl<Config, true>(temporary_storage,
                                             storage_size,
                                             input,
                                             keys,
                                             output,
                                             size,
                                             keys_size,
                                             compare_function,
                                             stream,
                                             debug_synchronous);
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SEARCH_HPP_
