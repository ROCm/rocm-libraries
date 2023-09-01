// Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_HPP_
#define ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "detail/device_binary_search.hpp"
#include "device_binary_search_config.hpp"
#include "device_transform.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

template<
    class Config,
    class HaystackIterator,
    class NeedlesIterator,
    class OutputIterator,
    class SearchFunction,
    class CompareFunction
>
inline
hipError_t binary_search(void * temporary_storage,
                         size_t& storage_size,
                         HaystackIterator haystack,
                         NeedlesIterator needles,
                         OutputIterator output,
                         size_t haystack_size,
                         size_t needles_size,
                         SearchFunction search_op,
                         CompareFunction compare_op,
                         hipStream_t stream,
                         bool debug_synchronous)
{
    using value_type = typename std::iterator_traits<NeedlesIterator>::value_type;

    if(temporary_storage == nullptr)
    {
        // Make sure user won't try to allocate 0 bytes memory, otherwise
        // user may again pass nullptr as temporary_storage
        storage_size = 4;
        return hipSuccess;
    }

    return transform<Config>(
        needles, output,
        needles_size,
        [haystack, haystack_size, search_op, compare_op]
        ROCPRIM_DEVICE
        (const value_type& value)
        {
            return search_op(haystack, haystack_size, value, compare_op);
        },
        stream, debug_synchronous
    );
}

} // end of detail namespace

/// \brief Performs a device-level lower bound check.
///
/// \par Overview
/// Runs multiple lower bound checks in parallel (one for each \p needle in <tt>needles</tt>).
/// A lower bound check returns the index of the first element in \p haystack that
/// causes \p compare_op(element,needle) to return false. If no item in \p haystack satisfies
/// this criteria, then \p haystack_size is returned.
/// Results are written by \p output.
///
/// \tparam Config - [optional] configuration information for the primitive. This can be 
/// \p lower_bound_config or a custom class with the same members.
/// \tparam HaystackIterator - Iterator type for items we'll be searching through (values).
/// \tparam NeedlesIterator - Iterator type for items we are performing lower bound checks
/// for (keys).
/// \tparam OutputIterator - Iterator type for the output indices.
/// \tparam CompareFunction [optional] A callable that can be used to compare two values.
/// defaults to rocprim::less.
///
/// \param [in] temporary_storage - pointer to device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and the function returns without performing the search operation.
/// \param [in,out] storage_size - reference to the size (in bytes) of \p temporary_storage.
/// \param haystack [in] - iterator pointing to the beginning of the range to search through.
/// \param needles [in] - iterator pointing to the first of the elements to perform lower
/// bound checks on.
/// \param output [out] - Iterator pointing to the beginning of the range where the results
/// are to be stored.
/// \param haystack_size [in] - the total number of values to search through.
/// \param needles_size [in] - the total number of keys to perform lower bound checks for.
/// \param compare_op [in] - binary operation function that will be used for comparison.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but the function object must not modify the objects passed to it.
/// The default value is \p CompareFunction().
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
template<
    class Config = default_config,
    class HaystackIterator,
    class NeedlesIterator,
    class OutputIterator,
    class CompareFunction = ::rocprim::less<>
>
inline
hipError_t lower_bound(void * temporary_storage,
                       size_t& storage_size,
                       HaystackIterator haystack,
                       NeedlesIterator needles,
                       OutputIterator output,
                       size_t haystack_size,
                       size_t needles_size,
                       CompareFunction compare_op = CompareFunction(),
                       hipStream_t stream = 0,
                       bool debug_synchronous = false)
{
    using value_type  = typename std::iterator_traits<NeedlesIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using config
        = std::conditional_t<std::is_same<default_config, Config>::value,
                             detail::default_config_for_lower_bound<value_type, output_type>,
                             Config>;

    return detail::binary_search<config>(temporary_storage,
                                         storage_size,
                                         haystack,
                                         needles,
                                         output,
                                         haystack_size,
                                         needles_size,
                                         detail::lower_bound_search_op(),
                                         compare_op,
                                         stream,
                                         debug_synchronous);
}

/// \brief Performs a device-level upper bound check.
///
/// \par Overview
/// Runs multiple upper bound checks in parallel (one for each \p needle in <tt>needles</tt>).
/// An upper bound check returns the index of the first element in \p haystack that
/// causes \p compare_op(needle,element) to return true. If no item in \p haystack satisfies
/// this criteria, then \p haystack_size is returned.
/// Results are written by \p output.
///
/// \tparam Config - [optional] configuration information for the primitive. This can be 
/// \p upper_bound_config or a custom class with the same members.
/// \tparam HaystackIterator - Iterator type for items we'll be searching through (values).
/// \tparam NeedlesIterator - Iterator type for items we are performing upper bound checks
/// for (keys).
/// \tparam OutputIterator - Iterator type for the output indices.
/// \tparam CompareFunction [optional] A callable that can be used to compare two values.
/// defaults to rocprim::less.
///
/// \param [in] temporary_storage - pointer to device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and the function returns without performing the search operation.
/// \param [in,out] storage_size - reference to the size (in bytes) of \p temporary_storage.
/// \param haystack [in] - iterator pointing to the beginning of the range to search through.
/// \param needles [in] - iterator pointing to the first of the elements to perform upper
/// bound checks on.
/// \param output [out] - Iterator pointing to the beginning of the range where the results
/// are to be stored.
/// \param haystack_size [in] - the total number of values to search through.
/// \param needles_size [in] - the total number of keys to perform upper bound checks for.
/// \param compare_op [in] - binary operation function that will be used for comparison.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but the function object must not modify the objects passed to it.
/// The default value is \p CompareFunction().
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
template<
    class Config = default_config,
    class HaystackIterator,
    class NeedlesIterator,
    class OutputIterator,
    class CompareFunction = ::rocprim::less<>
>
inline
hipError_t upper_bound(void * temporary_storage,
                       size_t& storage_size,
                       HaystackIterator haystack,
                       NeedlesIterator needles,
                       OutputIterator output,
                       size_t haystack_size,
                       size_t needles_size,
                       CompareFunction compare_op = CompareFunction(),
                       hipStream_t stream = 0,
                       bool debug_synchronous = false)
{
    using value_type  = typename std::iterator_traits<NeedlesIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using config
        = std::conditional_t<std::is_same<default_config, Config>::value,
                             detail::default_config_for_upper_bound<value_type, output_type>,
                             Config>;

    return detail::binary_search<config>(temporary_storage,
                                         storage_size,
                                         haystack,
                                         needles,
                                         output,
                                         haystack_size,
                                         needles_size,
                                         detail::upper_bound_search_op(),
                                         compare_op,
                                         stream,
                                         debug_synchronous);
}

/// \brief Performs a device-level parallel binary search.
///
/// \par Overview
/// Runs multiple binary searches in parallel. The result is a sequence of bools,
/// where each bool indicates if the corresponding search succeeded (the key was found)
/// or not. Results are written by \p output.
///
/// \tparam Config - [optional] configuration information for the primitive. This can be 
/// \p binary_search_config or a custom class with the same members.
/// \tparam HaystackIterator - Iterator type for items we'll be searching through (values).
/// \tparam NeedlesIterator - Iterator type for item we are searching for (keys).
/// \tparam OutputIterator - Iterator type for the output bools.
/// \tparam CompareFunction [optional] A callable that can be used to compare two values.
/// defaults to rocprim::less.
///
/// \param [in] temporary_storage - pointer to device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and the function returns without performing the search operation.
/// \param [in,out] storage_size - reference to the size (in bytes) of \p temporary_storage.
/// \param haystack [in] - iterator pointing to the beginning of the range to search through.
/// \param needles [in] - iterator pointing to the first of the elements to find.
/// \param output [out] - Iterator pointing to the beginning of the range where the results
/// are to be stored.
/// \param haystack_size [in] - the total number of values to search through.
/// \param needles_size [in] - the total number of keys to search for.
/// \param compare_op [in] - binary operation function that will be used for comparison.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but the function object must not modify the objects passed to it.
/// The default value is \p CompareFunction().
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
template<
    class Config = default_config,
    class HaystackIterator,
    class NeedlesIterator,
    class OutputIterator,
    class CompareFunction = ::rocprim::less<>
>
inline
hipError_t binary_search(void * temporary_storage,
                         size_t& storage_size,
                         HaystackIterator haystack,
                         NeedlesIterator needles,
                         OutputIterator output,
                         size_t haystack_size,
                         size_t needles_size,
                         CompareFunction compare_op = CompareFunction(),
                         hipStream_t stream = 0,
                         bool debug_synchronous = false)
{
    using value_type  = typename std::iterator_traits<NeedlesIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using config
        = std::conditional_t<std::is_same<default_config, Config>::value,
                             detail::default_config_for_binary_search<value_type, output_type>,
                             Config>;

    return detail::binary_search<config>(temporary_storage,
                                         storage_size,
                                         haystack,
                                         needles,
                                         output,
                                         haystack_size,
                                         needles_size,
                                         detail::binary_search_op(),
                                         compare_op,
                                         stream,
                                         debug_synchronous);
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_HPP_
