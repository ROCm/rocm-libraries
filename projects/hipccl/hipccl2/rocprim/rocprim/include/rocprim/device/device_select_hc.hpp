// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_SELECT_HC_HPP_
#define ROCPRIM_DEVICE_DEVICE_SELECT_HC_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../detail/binary_op_wrappers.hpp"

#include "../iterator/transform_iterator.hpp"

#include "device_scan_hc.hpp"
#include "device_partition_hc.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule_hc
/// @{

namespace detail
{

#define ROCPRIM_DETAIL_HC_SYNC(name, size, start) \
    { \
        if(debug_synchronous) \
        { \
            std::cout << name << "(" << size << ")"; \
            acc_view.wait(); \
            auto end = std::chrono::high_resolution_clock::now(); \
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); \
            std::cout << " " << d.count() * 1000 << " ms" << '\n'; \
        } \
    }

} // end detail namespace

/// \brief HC parallel select primitive for device level using range of flags.
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
/// \tparam Config - [optional] configuration of the primitive. It can be \p select_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam FlagIterator - random-access iterator type of the flag range. It can be
/// a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam SelectedCountOutputIterator - random-access iterator type of the selected_count_output
/// value. It can be a simple pointer type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the select operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to select values from.
/// \param [in] flags - iterator to the selection flag corresponding to the first element from \p input range.
/// \param [out] output - iterator to the first element in the output range.
/// \param [out] selected_count_output - iterator to the total number of selected values (length of \p output).
/// \param [in] size - number of element in the input range.
/// \param [in] acc_view - [optional] \p hc::accelerator_view object. The default value
/// is \p hc::accelerator().get_default_view() (default view of the default accelerator).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced. Default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level select operation is performed on an array of
/// integer values with array of <tt>char</tt>s used as flags.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// hc::accelerator_view acc_view = ...;
///
/// // Prepare input and output (declare arrays, allocate device memory etc.)
/// size_t size;                                           // e.g., 8
/// hc::array<int> input(hc::extent<1>(size), ...);        // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// hc::array<char> flags(hc::extent<1>(size), ...);       // e.g., [0, 1, 1, 0, 0, 1, 0, 1]
/// hc::array<int> output(hc::extent<1>(size), ...);       // empty array of 8 elements
/// hc::array<size_t> output_count(hc::extent<1>(1), ...); // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// // Get required size of the temporary storage
/// rocprim::select(
///     nullptr, temporary_storage_size_bytes,
///     input.accelerator_pointer(), flags.accelerator_pointer(),
///     output.accelerator_pointer(), output_count.accelerator_pointer(),
///     size, acc_view, false
/// );
///
/// // allocate temporary storage
/// hc::array<char> temporary_storage(temporary_storage_size_bytes, acc_view);
///
/// // perform selection
/// rocprim::select(
///     temporary_storage.accelerator_pointer(), temporary_storage_size_bytes,
///     input.accelerator_pointer(), flags.accelerator_pointer(),
///     output.accelerator_pointer(), output_count.accelerator_pointer(),
///     size, acc_view, false
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
void select(void * temporary_storage,
            size_t& storage_size,
            InputIterator input,
            FlagIterator flags,
            OutputIterator output,
            SelectedCountOutputIterator selected_count_output,
            const size_t size,
            hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
            const bool debug_synchronous = false)
{
    // Dummy unary predicate
    using unary_predicate_type = ::rocprim::empty_type;
    // Dummy inequality operation
    using inequality_op_type = ::rocprim::empty_type;

    detail::partition_impl<detail::select_method::flag, true, Config>(
        temporary_storage, storage_size, input, flags, output, selected_count_output,
        size, unary_predicate_type(), inequality_op_type(), acc_view, debug_synchronous
    );
}

/// \brief HC parallel select primitive for device level using selection operator.
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
/// \tparam Config - [optional] configuration of the primitive. It can be \p select_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam SelectedCountOutputIterator - random-access iterator type of the selected_count_output
/// value. It can be a simple pointer type.
/// \tparam UnaryPredicate - type of a unary selection operator.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the select operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to select values from.
/// \param [out] output - iterator to the first element in the output range.
/// \param [out] selected_count_output - iterator to the total number of selected values (length of \p output).
/// \param [in] size - number of element in the input range.
/// \param [in] predicate - unary function object that will be used for selecting values.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the object passed to it.
/// \param [in] acc_view - [optional] \p hc::accelerator_view object. The default value
/// is \p hc::accelerator().get_default_view() (default view of the default accelerator).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced. Default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level select operation is performed on an array of
/// integer values, only even values are selected.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// auto predicate =
///     [](int a) [[hcc]] -> bool
///     {
///         return (a%2) == 0;
///     };
///
/// hc::accelerator_view acc_view = ...;
///
/// // Prepare input and output (declare arrays, allocate device memory etc.)
/// size_t size;                                           // e.g., 8
/// hc::array<int> input(hc::extent<1>(size), ...);        // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// hc::array<int> output(hc::extent<1>(size), ...);       // empty array of 8 elements
/// hc::array<size_t> output_count(hc::extent<1>(1), ...); // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// // Get required size of the temporary storage
/// rocprim::select(
///     nullptr, temporary_storage_size_bytes,
///     input.accelerator_pointer(), output.accelerator_pointer(),
//      output_count.accelerator_pointer(),
///     predicate, size, acc_view, false
/// );
///
/// // allocate temporary storage
/// hc::array<char> temporary_storage(temporary_storage_size_bytes, acc_view);
///
/// // perform selection
/// rocprim::select(
///     temporary_storage.accelerator_pointer(), temporary_storage_size_bytes,
///     input.accelerator_pointer(), output.accelerator_pointer(),
//      output_count.accelerator_pointer(),
///     predicate, size, acc_view, false
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
void select(void * temporary_storage,
            size_t& storage_size,
            InputIterator input,
            OutputIterator output,
            SelectedCountOutputIterator selected_count_output,
            const size_t size,
            UnaryPredicate predicate,
            hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
            const bool debug_synchronous = false)
{
    // Dummy flag type
    using flag_type = ::rocprim::empty_type;
    flag_type * flags = nullptr;
    // Dummy inequality operation
    using inequality_op_type = ::rocprim::empty_type;

    detail::partition_impl<detail::select_method::predicate, true, Config>(
        temporary_storage, storage_size, input, flags, output, selected_count_output,
        size, predicate, inequality_op_type(), acc_view, debug_synchronous
    );
}

/// \brief HC device-level parallel unique primitive.
///
/// From given \p input range unique primitive eliminates all but the first element from every
/// consecutive group of equivalent elements and copies them into \p output.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Range specified by \p input must have at least \p size elements.
/// * Range specified by \p output must have at least so many elements, that all selected
/// values can be copied into it.
/// * Range specified by \p unique_count_output must have at least 1 element.
/// * By default <tt>InputIterator::value_type</tt>'s equality operator is used to check
/// if elements are equivalent.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p select_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam UniqueCountOutputIterator - random-access iterator type of the unique_count_output
/// value used to return number of unique values. It can be a simple pointer type.
/// \tparam EqualityOp - type of an binary operator used to compare values for equality.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the unique operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to select values from.
/// \param [out] output - iterator to the first element in the output range.
/// \param [out] unique_count_output - iterator to the total number of selected values (length of \p output).
/// \param [in] size - number of element in the input range.
/// \param [in] equality_op - [optional] binary function object used to compare input values for equality.
/// The signature of the function should be equivalent to the following:
/// <tt>bool equal_to(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the object passed to it.
/// \param [in] acc_view - [optional] \p hc::accelerator_view object. The default value
/// is \p hc::accelerator().get_default_view() (default view of the default accelerator).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced. The default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level unique operation is performed on an array of integer values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// hc::accelerator_view acc_view = ...;
///
/// // Prepare input and output (declare arrays, allocate device memory etc.)
/// size_t size;     // e.g., 8
/// hc::array<int> input(hc::extent<1>(size), ...);        // e.g., [1, 4, 2, 4, 4, 7, 7, 7]
/// hc::array<int> output(hc::extent<1>(size), ...);       // empty array of 8 elements
/// hc::array<size_t> output_count(hc::extent<1>(1), ...); // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// // Get required size of the temporary storage
/// rocprim::unique(
///     nullptr, temporary_storage_size_bytes
///     input.accelerator_pointer(),
///     output.accelerator_pointer(), output_count.accelerator_pointer(),
///     size, rocprim::equal_to<int>(), acc_view, false
/// );
///
/// // allocate temporary storage
/// hc::array<char> temporary_storage(temporary_storage_size_bytes, acc_view);
///
/// // perform unique operation
/// rocprim::unique(
///     temporary_storage.accelerator_pointer(), temporary_storage_size_bytes
///     input.accelerator_pointer(),
///     output.accelerator_pointer(), output_count.accelerator_pointer(),
///     size, rocprim::equal_to<int>(), acc_view, false
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
void unique(void * temporary_storage,
            size_t& storage_size,
            InputIterator input,
            OutputIterator output,
            UniqueCountOutputIterator unique_count_output,
            const size_t size,
            EqualityOp equality_op = EqualityOp(),
            hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
            const bool debug_synchronous = false)
{
    // Dummy unary predicate
    using unary_predicate_type = ::rocprim::empty_type;

    // Dummy flag type
    using flag_type = ::rocprim::empty_type;
    flag_type * flags = nullptr;

    // Convert equality operator to inequality operator
    auto inequality_op = detail::inequality_wrapper<EqualityOp>(equality_op);

    return detail::partition_impl<detail::select_method::unique, true, Config>(
        temporary_storage, storage_size, input, flags, output, unique_count_output,
        size, unary_predicate_type(), inequality_op, acc_view, debug_synchronous
    );
}

#undef ROCPRIM_DETAIL_HC_SYNC

/// @}
// end of group devicemodule_hc

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SELECT_HC_HPP_
