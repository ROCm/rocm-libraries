// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_SCAN_BY_KEY_HIP_HPP_
#define ROCPRIM_DEVICE_DEVICE_SCAN_BY_KEY_HIP_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../iterator/zip_iterator.hpp"
#include "../iterator/discard_iterator.hpp"
#include "../iterator/detail/replace_first_iterator.hpp"
#include "../types/tuple.hpp"

#include "../detail/various.hpp"
#include "../detail/binary_op_wrappers.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule_hip
/// @{

/// \brief HIP parallel inclusive scan-by-key primitive for device level.
///
/// inclusive_scan_by_key function performs a device-wide inclusive prefix scan-by-key
/// operation using binary \p scan_op operator.
///
/// \par Overview
/// * Supports non-commutative scan operators. However, a scan operator should be
/// associative. When used with non-associative functions the results may be non-deterministic
/// and/or vary in precision.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p keys_input, \p values_input, and \p values_output must have
/// at least \p size elements.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p scan_config or
/// a custom class with the same members.
/// \tparam KeysInputIterator - random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam ValuesInputIterator - random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam ValuesOutputIterator - random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam BinaryFunction - type of binary function used for scan. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
/// \tparam KeyCompareFunction - type of binary function used to determine keys equality. Default type
/// is \p rocprim::equal_to<T>, where \p T is a \p value_type of \p KeysInputIterator.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the scan operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - iterator to the first element in the range of keys.
/// \param [in] values_input - iterator to the first element in the range of values to scan.
/// \param [out] values_output - iterator to the first element in the output value range.
/// \param [in] size - number of element in the input range.
/// \param [in] scan_op - binary operation function object that will be used for scanning
/// input values.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is BinaryFunction().
/// \param [in] key_compare_op - binary operation function object that will be used to determine keys equality.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is KeyCompareFunction().
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful scan; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level inclusive sum-by-key operation is performed on an array of
/// integer values (<tt>short</tt>s are scanned into <tt>int</tt>s).
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t size;           // e.g., 8
/// int *   keys_input;    // e.g., [1, 1, 2, 2, 3, 3, 3, 5]
/// short * values_input;  // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int *   values_output; // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::inclusive_scan_by_key(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, values_input,
///     values_output, size,
///     rocprim::plus<int>()
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform scan-by-key
/// rocprim::inclusive_scan_by_key(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, values_input,
///     values_output, size,
///     rocprim::plus<int>()
/// );
/// // values_output: [1, 2, 3, 7, 5, 11, 18, 8]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class KeysInputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>,
    class KeyCompareFunction = ::rocprim::equal_to<typename std::iterator_traits<KeysInputIterator>::value_type>
>
inline
hipError_t inclusive_scan_by_key(void * temporary_storage,
                                 size_t& storage_size,
                                 KeysInputIterator keys_input,
                                 ValuesInputIterator values_input,
                                 ValuesOutputIterator values_output,
                                 const size_t size,
                                 BinaryFunction scan_op = BinaryFunction(),
                                 KeyCompareFunction key_compare_op = KeyCompareFunction(),
                                 const hipStream_t stream = 0,
                                 bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using scan_by_key_operator = detail::scan_by_key_op_wrapper<
        input_type, key_type, BinaryFunction, KeyCompareFunction
    >;

    return inclusive_scan<Config>(
        temporary_storage, storage_size,
        make_zip_iterator(
            make_tuple(values_input, keys_input)
        ),
        make_zip_iterator(
            make_tuple(values_output, make_discard_iterator())
        ),
        size,
        scan_by_key_operator(scan_op, key_compare_op),
        stream,
        debug_synchronous
    );
}

/// \brief HIP parallel exclusive scan-by-key primitive for device level.
///
/// inclusive_scan_by_key function performs a device-wide exclusive prefix scan-by-key
/// operation using binary \p scan_op operator.
///
/// \par Overview
/// * Supports non-commutative scan operators. However, a scan operator should be
/// associative. When used with non-associative functions the results may be non-deterministic
/// and/or vary in precision.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p keys_input, \p values_input, and \p values_output must have
/// at least \p size elements.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p scan_config or
/// a custom class with the same members.
/// \tparam KeysInputIterator - random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam ValuesInputIterator - random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam ValuesOutputIterator - random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam InitValueType - type of the initial value.
/// \tparam BinaryFunction - type of binary function used for scan. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
/// \tparam KeyCompareFunction - type of binary function used to determine keys equality. Default type
/// is \p rocprim::equal_to<T>, where \p T is a \p value_type of \p KeysInputIterator.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the scan operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - iterator to the first element in the range of keys.
/// \param [in] values_input - iterator to the first element in the range of values to scan.
/// \param [out] values_output - iterator to the first element in the output value range.
/// \param [in] initial_value - initial value to start the scan.
/// \param [in] size - number of element in the input range.
/// \param [in] scan_op - binary operation function object that will be used for scanning
/// input values.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is BinaryFunction().
/// \param [in] key_compare_op - binary operation function object that will be used to determine keys equality.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is KeyCompareFunction().
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful scan; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level inclusive sum-by-key operation is performed on an array of
/// integer values (<tt>short</tt>s are scanned into <tt>int</tt>s).
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t size;           // e.g., 8
/// int *   keys_input;    // e.g., [1, 1, 1, 2, 2, 3, 3, 4]
/// short * values_input;  // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int start_value;       // e.g., 9
/// int *   values_output; // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::exclusive_scan_by_key(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, values_input,
///     values_output, start_value,
///     size,rocprim::plus<int>()
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform scan-by-key
/// rocprim::exclusive_scan_by_key(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, values_input,
///     values_output, start_value,
///     size,rocprim::plus<int>()
/// );
/// // values_output: [9, 10, 12, 9, 13, 9, 15, 9]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class KeysInputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class InitialValueType,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>,
    class KeyCompareFunction = ::rocprim::equal_to<typename std::iterator_traits<KeysInputIterator>::value_type>
>
inline
hipError_t exclusive_scan_by_key(void * temporary_storage,
                                 size_t& storage_size,
                                 KeysInputIterator keys_input,
                                 ValuesInputIterator values_input,
                                 ValuesOutputIterator values_output,
                                 const InitialValueType initial_value,
                                 const size_t size,
                                 BinaryFunction scan_op = BinaryFunction(),
                                 KeyCompareFunction key_compare_op = KeyCompareFunction(),
                                 const hipStream_t stream = 0,
                                 bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using scan_by_key_operator = detail::scan_by_key_op_wrapper<
        input_type, key_type, BinaryFunction, KeyCompareFunction
    >;

    return inclusive_scan<Config>(
        temporary_storage, storage_size,
        make_transform_iterator(
            make_zip_iterator(
                make_tuple(
                    detail::replace_first_iterator<ValuesInputIterator>(
                        values_input - 1, initial_value
                    ),
                    keys_input,
                    detail::replace_first_iterator<KeysInputIterator>(
                        keys_input - 1, key_type()
                    )
                )
            ),
            [initial_value, key_compare_op] ROCPRIM_HOST_DEVICE
            (const ::rocprim::tuple<input_type, key_type, key_type>& t)
                -> ::rocprim::tuple<input_type, key_type>
            {
                if(!key_compare_op(::rocprim::get<1>(t), ::rocprim::get<2>(t)))
                {
                    return ::rocprim::make_tuple(
                        static_cast<input_type>(initial_value),
                        ::rocprim::get<1>(t)
                    );
                }
                return ::rocprim::make_tuple(
                    ::rocprim::get<0>(t), ::rocprim::get<1>(t)
                );
            }
        ),
        make_zip_iterator(make_tuple(values_output, make_discard_iterator())),
        size,
        scan_by_key_operator(scan_op, key_compare_op),
        stream,
        debug_synchronous
    );
}

/// @}
// end of group devicemodule_hip

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SCAN_HIP_HPP_
