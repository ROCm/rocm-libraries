// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_SELECT_HPP_
#define ROCPRIM_DEVICE_DEVICE_SELECT_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../detail/binary_op_wrappers.hpp"

#include "../iterator/transform_iterator.hpp"

#include "device_partition.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

} // end detail namespace

/// \brief Parallel select primitive for device level using range of flags.
///
/// Performs a device-wide selection based on input \p flags. If a value from \p input
/// should be selected and copied into \p output range the corresponding item from
/// \p flags range should be set to such value that can be implicitly converted to
/// \p true (\p bool type).
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p input and \p flags must have at least \p size elements.
/// * Range specified by \p output must have at least so many elements, that all positively
/// flagged values can be copied into it.
/// * Range specified by \p selected_count_output must have at least 1 element.
/// * Values of \p flag range should be implicitly convertible to `bool` type.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `select_config`.
/// \tparam InputIterator random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam FlagIterator random-access iterator type of the flag range. It can be
/// a simple pointer type.
/// \tparam OutputIterator random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam SelectedCountOutputIterator random-access iterator type of the selected_count_output
/// value. It can be a simple pointer type.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the select operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input iterator to the first element in the range to select values from.
/// \param [in] flags iterator to the selection flag corresponding to the first element from \p input range.
/// \param [out] output iterator to the first element in the output range.
/// \param [out] selected_count_output iterator to the total number of selected values (length of \p output).
/// \param [in] size number of element in the input range.
/// \param [in] stream [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level select operation is performed on an array of
/// integer values with array of <tt>char</tt>s used as flags.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;     // e.g., 8
/// int * input;           // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// char * flags;          // e.g., [0, 1, 1, 0, 0, 1, 0, 1]
/// int * output;          // empty array of 8 elements
/// size_t * output_count; // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::select(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, flags,
///     output, output_count,
///     input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform selection
/// rocprim::select(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, flags,
///     output, output_count,
///     input_size
/// );
/// // output: [2, 3, 6, 8]
/// // output_count: 4
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class FlagIterator,
    class OutputIterator,
    class SelectedCountOutputIterator
>
inline
hipError_t select(void * temporary_storage,
                  size_t& storage_size,
                  InputIterator input,
                  FlagIterator flags,
                  OutputIterator output,
                  SelectedCountOutputIterator selected_count_output,
                  const size_t size,
                  const hipStream_t stream = 0,
                  const bool debug_synchronous = false)
{
    // Dummy unary predicate
    using unary_predicate_type = ::rocprim::empty_type;
    // Dummy inequality operation
    using inequality_op_type = ::rocprim::empty_type;
    using offset_type = unsigned int;
    rocprim::empty_type* const no_values = nullptr; // key only

    using output_key_iterator_tuple = tuple<OutputIterator, ::rocprim::empty_type>;
    output_key_iterator_tuple output_tuple{output, ::rocprim::empty_type()};

    using output_value_iterator_tuple = tuple<::rocprim::empty_type*, ::rocprim::empty_type*>;
    const output_value_iterator_tuple no_output_values{nullptr, nullptr}; // key only

    return detail::partition_impl<detail::partition_subalgo::select_flag, Config, offset_type>(
        temporary_storage,
        storage_size,
        input,
        no_values,
        flags,
        output_tuple,
        no_output_values,
        selected_count_output,
        size,
        inequality_op_type(),
        stream,
        debug_synchronous,
        unary_predicate_type());
}

/// \brief Parallel select primitive for device level using selection operator.
///
/// Performs a device-wide selection using selection operator. If a value \p x from \p input
/// should be selected and copied into \p output range, then <tt>predicate(x)</tt> has to
/// return \p true.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Range specified by \p input must have at least \p size elements.
/// * Range specified by \p output must have at least so many elements, that all selected
/// values can be copied into it.
/// * Range specified by \p selected_count_output must have at least 1 element.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `select_config`.
/// \tparam InputIterator random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam OutputIterator random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam SelectedCountOutputIterator random-access iterator type of the selected_count_output
/// value. It can be a simple pointer type.
/// \tparam UnaryPredicate type of a unary selection predicate.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the select operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input iterator to the first element in the range to select values from.
/// \param [out] output iterator to the first element in the output range.
/// \param [out] selected_count_output iterator to the total number of selected values (length of \p output).
/// \param [in] size number of element in the input range.
/// \param [in] predicate unary function object that will be used for selecting values.
/// The predicate must meet the C++ named requirement \p BinaryPredicate :
///   - The result of applying the predicate must be convertible to bool
///   - The predicate must accept const object arguments, with the same behavior regardless of
///     whether its arguments are const or non-const.
/// In practice, the signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a);</tt>. The signature does not need to have
/// <tt>const &</tt>, but the function object must not modify the object passed to it.
/// \param [in] stream [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level select operation is performed on an array of
/// integer values. Only even values are selected.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// auto predicate =
///     [] __device__ (int a) -> bool
///     {
///         return (a % 2) == 0;
///     };
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;     // e.g., 8
/// int * input;           // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int * output;          // empty array of 8 elements
/// size_t * output_count; // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::select(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, output_count,
///     predicate, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform selection
/// rocprim::select(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, output_count,
///     predicate, input_size
/// );
/// // output: [2, 4, 6, 8]
/// // output_count: 4
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class SelectedCountOutputIterator,
    class UnaryPredicate
>
inline
hipError_t select(void * temporary_storage,
                  size_t& storage_size,
                  InputIterator input,
                  OutputIterator output,
                  SelectedCountOutputIterator selected_count_output,
                  const size_t size,
                  UnaryPredicate predicate,
                  const hipStream_t stream = 0,
                  const bool debug_synchronous = false)
{
    // Dummy flag type
    using flag_type = ::rocprim::empty_type;
    using offset_type = unsigned int;
    flag_type * flags = nullptr;
    // Dummy inequality operation
    using inequality_op_type = ::rocprim::empty_type;
    rocprim::empty_type* const no_values = nullptr; // key only

    using output_key_iterator_tuple = tuple<OutputIterator, ::rocprim::empty_type>;
    output_key_iterator_tuple output_tuple{output, ::rocprim::empty_type()};

    using output_value_iterator_tuple = tuple<::rocprim::empty_type*, ::rocprim::empty_type*>;
    const output_value_iterator_tuple no_output_values{nullptr, nullptr}; // key only

    return detail::partition_impl<detail::partition_subalgo::select_predicate, Config, offset_type>(
        temporary_storage,
        storage_size,
        input,
        no_values,
        flags,
        output_tuple,
        no_output_values,
        selected_count_output,
        size,
        inequality_op_type(),
        stream,
        debug_synchronous,
        predicate);
}

/// \brief Parallel select primitive for device level using a range of pre-selected flags.
///
/// Performs a device-wide selection based on input \p flags to which a selection operator is
/// applied before doing the selection. If a value \p x from \p input should be selected and
/// copied into \p output range the corresponding item from the \p flags range should be set
/// to such value that, after applying the <tt>predicate(x)</tt> to it, it can be implicitly
/// converted to \p true (\p bool type).
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Range specified by \p input and \p flags must have at least \p size elements.
/// * Range specified by \p output must have at least so many elements, that all selected
/// values can be copied into it.
/// * Range specified by \p selected_count_output must have at least 1 element.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `select_config`.
/// \tparam InputIterator random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam FlagIterator random-access iterator type of the flag range. It can be
/// a simple pointer type.
/// \tparam OutputIterator random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam SelectedCountOutputIterator random-access iterator type of the selected_count_output
/// value. It can be a simple pointer type.
/// \tparam UnaryPredicate type of a unary selection predicate.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the select operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input iterator to the first element in the range to select values from.
/// \param [in] flags iterator to the selection flag corresponding to the first element from \p input range.
/// \param [out] output iterator to the first element in the output range.
/// \param [out] selected_count_output iterator to the total number of selected values (length of \p output).
/// \param [in] size number of element in the input range.
/// \param [in] predicate unary function object that will be used for selecting flags.
/// The predicate must meet the C++ named requirement \p BinaryPredicate:
///   - The result of applying the predicate must be convertible to bool.
///   - The predicate must accept const object arguments, with the same behavior regardless of
///     whether its arguments are const or non-const.
/// In practice, the signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a);</tt>. The signature does not need to have
/// <tt>const &</tt>, but the function object must not modify the object passed to it.
/// \param [in] stream [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level select operation is performed on an array of
/// integer values. Only values with even flags are selected.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// auto predicate =
///     [] __device__ (int a) -> bool
///     {
///         return (a % 2) == 0;
///     };
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;     // e.g., 8
/// int * input;           // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int * flags;           // e.g., [0, 1, 2, 3, 4, 5, 6, 7]
/// int * output;          // empty array of 8 elements
/// size_t * output_count; // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::select(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, flags, output, output_count,
///     predicate, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform selection
/// rocprim::select(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, flags, output, output_count,
///     predicate, input_size
/// );
/// // output: [1, 3, 5, 7]
/// // output_count: 4
/// \endcode
/// \endparblock
template<class Config = default_config,
         class InputIterator,
         class FlagIterator,
         class OutputIterator,
         class SelectedCountOutputIterator,
         class UnaryPredicate>
inline hipError_t select(void*                       temporary_storage,
                         size_t&                     storage_size,
                         InputIterator               input,
                         FlagIterator                flags,
                         OutputIterator              output,
                         SelectedCountOutputIterator selected_count_output,
                         const size_t                size,
                         UnaryPredicate              predicate,
                         const hipStream_t           stream            = 0,
                         const bool                  debug_synchronous = false)
{
    // Dummy inequality operation
    using inequality_op_type             = ::rocprim::empty_type;
    using offset_type                    = unsigned int;
    rocprim::empty_type* const no_values = nullptr; // key only

    using output_key_iterator_tuple = tuple<OutputIterator, ::rocprim::empty_type>;
    output_key_iterator_tuple output_tuple{output, ::rocprim::empty_type()};

    using output_value_iterator_tuple = tuple<::rocprim::empty_type*, ::rocprim::empty_type*>;
    const output_value_iterator_tuple no_output_values{nullptr, nullptr}; // key only

    return detail::partition_impl<detail::partition_subalgo::select_predicated_flag,
                                  Config,
                                  offset_type>(temporary_storage,
                                               storage_size,
                                               input,
                                               no_values,
                                               flags,
                                               output_tuple,
                                               no_output_values,
                                               selected_count_output,
                                               size,
                                               inequality_op_type(),
                                               stream,
                                               debug_synchronous,
                                               predicate);
}

/// \brief Device-level parallel unique primitive.
///
/// From given \p input range unique primitive eliminates all but the first element from every
/// consecutive group of equivalent elements and copies them into \p output.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage is a null pointer.
/// * Range specified by \p input must have at least \p size elements.
/// * Range specified by \p output must have at least so many elements, that all selected
/// values can be copied into it.
/// * Range specified by \p unique_count_output must have at least 1 element.
/// * By default <tt>InputIterator::value_type</tt>'s equality operator is used to check
/// if elements are equivalent.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `select_config`.
/// \tparam InputIterator random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam OutputIterator random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam UniqueCountOutputIterator random-access iterator type of the unique_count_output
/// value used to return number of unique values. It can be a simple pointer type.
/// \tparam EqualityOp type of an binary operator used to compare values for equality.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the unique operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input iterator to the first element in the range to select values from.
/// \param [out] output iterator to the first element in the output range.
/// \param [out] unique_count_output iterator to the total number of selected values (length of \p output).
/// \param [in] size number of element in the input range.
/// \param [in] equality_op [optional] binary function object used to compare input values for equality.
/// The signature of the function should be equivalent to the following:
/// <tt>bool equal_to(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the object passed to it.
/// \param [in] stream [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level unique operation is performed on an array of integer values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;     // e.g., 8
/// int * input;           // e.g., [1, 4, 2, 4, 4, 7, 7, 7]
/// int * output;          // empty array of 8 elements
/// size_t * output_count; // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::unique(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, output_count,
///     input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform unique operation
/// rocprim::unique(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, output_count,
///     input_size
/// );
/// // output: [1, 4, 2, 4, 7]
/// // output_count: 5
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class UniqueCountOutputIterator,
    class EqualityOp = ::rocprim::equal_to<typename std::iterator_traits<InputIterator>::value_type>
>
inline
hipError_t unique(void * temporary_storage,
                  size_t& storage_size,
                  InputIterator input,
                  OutputIterator output,
                  UniqueCountOutputIterator unique_count_output,
                  const size_t size,
                  EqualityOp equality_op = EqualityOp(),
                  const hipStream_t stream = 0,
                  const bool debug_synchronous = false)
{
    // Dummy unary predicate
    using unary_predicate_type = ::rocprim::empty_type;
    using offset_type = unsigned int;
    // Dummy flag type
    using flag_type = ::rocprim::empty_type;
    const flag_type * flags = nullptr;
    rocprim::empty_type* const no_values = nullptr; // key only

    // Convert equality operator to inequality operator
    auto inequality_op = detail::inequality_wrapper<EqualityOp>(equality_op);

    using output_key_iterator_tuple = tuple<OutputIterator, ::rocprim::empty_type>;
    output_key_iterator_tuple output_tuple{output, ::rocprim::empty_type()};

    using output_value_iterator_tuple = tuple<::rocprim::empty_type*, ::rocprim::empty_type*>;
    const output_value_iterator_tuple no_output_values{nullptr, nullptr}; // key only

    return detail::partition_impl<detail::partition_subalgo::select_unique, Config, offset_type>(
        temporary_storage,
        storage_size,
        input,
        no_values,
        flags,
        output_tuple,
        no_output_values,
        unique_count_output,
        size,
        inequality_op,
        stream,
        debug_synchronous,
        unary_predicate_type());
}

/// \brief Device-level parallel unique by key primitive.
///
/// From given \p input range unique primitive eliminates all but the first element from every
/// consecutive group of equivalent elements and copies them and their corresponding keys into
/// \p output.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage is a null pointer.
/// * Ranges specified by \p keys_input and value_input must have at least \p size elements each.
/// * Ranges specified by \p keys_output and values_output each must have at least so many elements,
/// that all selected values can be copied into them.
/// * Range specified by \p unique_count_output must have at least 1 element.
/// * By default <tt>InputIterator::value_type</tt>'s equality operator is used to check
/// if elements are equivalent.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `select_config`.
/// \tparam KeyIterator random-access iterator type of the input key range. It can be
/// a simple pointer type.
/// \tparam ValueIterator random-access iterator type of the input value range. It can be
/// a simple pointer type.
/// \tparam OutputKeyIterator random-access iterator type of the output key range. It can be
/// a simple pointer type.
/// \tparam OutputValueIterator random-access iterator type of the output value range. It can be
/// a simple pointer type.
/// \tparam UniqueCountOutputIterator random-access iterator type of the unique_count_output
/// value used to return number of unique keys and values. It can be a simple pointer type.
/// \tparam EqualityOp type of an binary operator used to compare keys for equality.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the unique operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input iterator to the first element in the range to select keys from.
/// \param [in] values_input iterator to the first element in the range of values corresponding to keys
/// \param [out] keys_output iterator to the first element in the output key range.
/// \param [out] values_output iterator to the first element in the output value range.
/// \param [out] unique_count_output iterator to the total number of selected values (length of \p output).
/// \param [in] size number of element in the input range.
/// \param [in] equality_op [optional] binary function object used to compare input values for equality.
/// The signature of the function should be equivalent to the following:
/// <tt>bool equal_to(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the object passed to it.
/// \param [in] stream [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
template <typename Config = default_config,
          typename KeyIterator,
          typename ValueIterator,
          typename OutputKeyIterator,
          typename OutputValueIterator,
          typename UniqueCountOutputIterator,
          typename EqualityOp
          = ::rocprim::equal_to<typename std::iterator_traits<KeyIterator>::value_type>>
inline hipError_t unique_by_key(void*                           temporary_storage,
                                size_t&                         storage_size,
                                const KeyIterator               keys_input,
                                const ValueIterator             values_input,
                                const OutputKeyIterator         keys_output,
                                const OutputValueIterator       values_output,
                                const UniqueCountOutputIterator unique_count_output,
                                const size_t                    size,
                                const EqualityOp                equality_op       = EqualityOp(),
                                const hipStream_t               stream            = 0,
                                const bool                      debug_synchronous = false)
{
    using offset_type = unsigned int;
    // Dummy flag
    ::rocprim::empty_type* const no_flags = nullptr;
    // Dummy predicate
    const auto no_predicate = ::rocprim::empty_type{};

    // Convert equality operator to inequality operator
    const auto inequality_op = detail::inequality_wrapper<EqualityOp>(equality_op);

    using output_key_iterator_tuple = tuple<OutputKeyIterator, ::rocprim::empty_type>;
    output_key_iterator_tuple output_key_tuple{keys_output, ::rocprim::empty_type()};

    using output_value_iterator_tuple = tuple<OutputValueIterator, ::rocprim::empty_type*>;
    const output_value_iterator_tuple output_value_tuple{values_output, nullptr};

    return detail::partition_impl<detail::partition_subalgo::select_unique_by_key,
                                  Config,
                                  offset_type>(temporary_storage,
                                               storage_size,
                                               keys_input,
                                               values_input,
                                               no_flags,
                                               output_key_tuple,
                                               output_value_tuple,
                                               unique_count_output,
                                               size,
                                               inequality_op,
                                               stream,
                                               debug_synchronous,
                                               no_predicate);
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SELECT_HPP_
