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

#ifndef ROCPRIM_DEVICE_DEVICE_SEARCH_N_HPP_
#define ROCPRIM_DEVICE_DEVICE_SEARCH_N_HPP_

#include "../config.hpp"
#include "config_types.hpp"
#include "detail/device_search_n.hpp"
#include "device_search_n_config.hpp"

#include <cstddef>
#include <cstdio>
#include <iterator>

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

/// \brief Searches for the first occurrence of a sequence of \p count elements all equal to \p value.
///
/// The equality of the elements of the sequence and the given value is determined according to a
/// given comparison function. If found, the index of the first item of the found sequence
/// in the input is returned. Otherwise, returns the size of the input.
///
/// \par Overview
/// * The contents of the inputs are not altered by the function.
/// * Returns the required size of `temporary_storage` in `storage_size`
/// if `temporary_storage` is a null pointer.
/// * Accepts custom compare_functions for search across the device.
/// * Streams in graph capture mode are supported
///
/// \tparam Config [optional] configuration of the primitive. It must be `default_config` or `search_n_config`.
/// \tparam InputIterator [inferred] random-access iterator type of the input range. Must meet the
///   requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator [inferred] random-access iterator type of the input range. Must meet the
///   requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam BinaryPredicate [inferred] Type of binary function that accepts two arguments of
///   type `InputIterator` and returns a value convertible to bool. Default type is `rocprim::equal_to<>.`
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
///   a null pointer is passed, the required allocation size (in bytes) is written to
/// `storage_size` and function returns without performing the search.
/// \param [in,out] storage_size reference to a size (in bytes) of `temporary_storage`.
/// \param [in] input iterator to the input range.
/// \param [out] output iterator to the output range. The output is one element.
/// \param [in] size number of elements in the input range.
/// \param [in] count number of elements in the sequence. Must be less or equal than \p size, otherwise `hipErrorInvalidValue` will be returned.
/// \param [in] value value of the elements to search for.
/// \param [in] binary_predicate binary operation function object that will be used for comparison.
///   The signature of the function should be equivalent to the following:
///   <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
///   <tt>const &</tt>, but function object must not modify the objects passed to it.
///   The comparator must meet the C++ requirements of BinaryPredicate.
///   The default value is `BinaryPredicate()`.
/// \param [in] stream [optional] HIP stream object. Default is `0` (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
///   launch is forced in order to check for errors. Default value is `false`.
///
/// \returns `hipSuccess` (`0`) after successful search; otherwise a HIP runtime error of
///   type `hipError_t`.
template<class Config = default_config,
         class InputIterator,
         class OutputIterator,
         class BinaryPredicate
         = rocprim::equal_to<typename std::iterator_traits<InputIterator>::value_type>>
ROCPRIM_INLINE
hipError_t search_n(void*          temporary_storage,
                    size_t&        storage_size,
                    InputIterator  input,
                    OutputIterator output,
                    const size_t   size,
                    const size_t   count,
                    const typename std::iterator_traits<InputIterator>::value_type* value,
                    const BinaryPredicate binary_predicate  = BinaryPredicate(),
                    const hipStream_t     stream            = static_cast<hipStream_t>(0),
                    const bool            debug_synchronous = false)
{
    return detail::search_n_impl<Config>(temporary_storage,
                                         storage_size,
                                         input,
                                         output,
                                         size,
                                         count,
                                         value,
                                         binary_predicate,
                                         stream,
                                         debug_synchronous);
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SEARCH_N_HPP_
