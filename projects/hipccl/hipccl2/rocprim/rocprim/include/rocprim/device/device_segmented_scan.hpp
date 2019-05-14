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

#ifndef ROCPRIM_DEVICE_DEVICE_SEGMENTED_SCAN_HPP_
#define ROCPRIM_DEVICE_DEVICE_SEGMENTED_SCAN_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../detail/match_result_type.hpp"

#include "../iterator/zip_iterator.hpp"
#include "../iterator/discard_iterator.hpp"
#include "../iterator/transform_iterator.hpp"
#include "../types/tuple.hpp"

#include "device_scan_config.hpp"
#include "detail/device_segmented_scan.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

template<
    bool Exclusive,
    class Config,
    class ResultType,
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class InitValueType,
    class BinaryFunction
>
__global__
void segmented_scan_kernel(InputIterator input,
                           OutputIterator output,
                           OffsetIterator begin_offsets,
                           OffsetIterator end_offsets,
                           InitValueType initial_value,
                           BinaryFunction scan_op)
{
    segmented_scan<Exclusive, Config, ResultType>(
        input, output, begin_offsets, end_offsets,
        static_cast<ResultType>(initial_value), scan_op
    );
}

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start) \
    { \
        auto error = hipPeekAtLastError(); \
        if(error != hipSuccess) return error; \
        if(debug_synchronous) \
        { \
            std::cout << name << "(" << size << ")"; \
            auto error = hipStreamSynchronize(stream); \
            if(error != hipSuccess) return error; \
            auto end = std::chrono::high_resolution_clock::now(); \
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); \
            std::cout << " " << d.count() * 1000 << " ms" << '\n'; \
        } \
    }

template<
    bool Exclusive,
    class Config,
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class InitValueType,
    class BinaryFunction
>
inline
hipError_t segmented_scan_impl(void * temporary_storage,
                               size_t& storage_size,
                               InputIterator input,
                               OutputIterator output,
                               unsigned int segments,
                               OffsetIterator begin_offsets,
                               OffsetIterator end_offsets,
                               const InitValueType initial_value,
                               BinaryFunction scan_op,
                               hipStream_t stream,
                               bool debug_synchronous)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using result_type = typename ::rocprim::detail::match_result_type<
        input_type, BinaryFunction
    >::type;

    // Get default config if Config is default_config
    using config = default_or_custom_config<
        Config,
        default_scan_config<ROCPRIM_TARGET_ARCH, result_type>
    >;

    constexpr unsigned int block_size = config::block_size;

    if(temporary_storage == nullptr)
    {
        // Make sure user won't try to allocate 0 bytes memory, because
        // hipMalloc will return nullptr when size is zero.
        storage_size = 4;
        return hipSuccess;
    }

    std::chrono::high_resolution_clock::time_point start;
    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(segmented_scan_kernel<Exclusive, config, result_type>),
        dim3(segments), dim3(block_size), 0, stream,
        input, output,
        begin_offsets, end_offsets,
        initial_value, scan_op
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("segmented_scan", segments, start);
    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

} // end of detail namespace

/// \brief Parallel segmented inclusive scan primitive for device level.
///
/// segmented_inclusive_scan function performs a device-wide inclusive scan operation
/// across multiple sequences from \p input using binary \p scan_op operator.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p input and \p output must have at least \p size elements.
/// * Ranges specified by \p begin_offsets and \p end_offsets must have
/// at least \p segments elements. They may use the same sequence <tt>offsets</tt> of at least
/// <tt>segments + 1</tt> elements: <tt>offsets</tt> for \p begin_offsets and
/// <tt>offsets + 1</tt> for \p end_offsets.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p scan_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ RandomAccessIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ RandomAccessIterator concept. It can be a simple pointer type.
/// \tparam OffsetIterator - random-access iterator type of segment offsets. Must meet the
/// requirements of a C++ RandomAccessIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction - type of binary function used for scan operation. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the scan operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to scan.
/// \param [out] output - iterator to the first element in the output range.
/// \param [in] segments - number of segments in the input range.
/// \param [in] begin_offsets - iterator to the first element in the range of beginning offsets.
/// \param [in] end_offsets - iterator to the first element in the range of ending offsets.
/// \param [in] scan_op - binary operation function object that will be used for scan.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// The default value is \p BinaryFunction().
/// \param [in] stream - [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful scan; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level segmented inclusive min-scan operation is performed on
/// an array of integer values (<tt>short</tt>s are scanned into <tt>int</tt>s) using custom operator.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // custom scan function
/// auto min_op =
///     [] __device__ (int a, int b) -> int
///     {
///         return a < b ? a : b;
///     };
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// short * input;        // e.g., [4, 7, 6, 2, 5, 1, 3, 8]
/// int   * output;       // empty array of 8 elements
/// size_t segments;      // e.g., 3
/// int * offsets;        // e.g. [0, 2, 4, 8]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::segmented_inclusive_scan(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, segments, offsets, offsets + 1, min_op
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform scan
/// rocprim::inclusive_scan(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, segments, offsets, offsets + 1, min_op
/// );
/// // output: [4, 4, 6, 2, 5, 1, 1, 1]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
hipError_t segmented_inclusive_scan(void * temporary_storage,
                                    size_t& storage_size,
                                    InputIterator input,
                                    OutputIterator output,
                                    unsigned int segments,
                                    OffsetIterator begin_offsets,
                                    OffsetIterator end_offsets,
                                    BinaryFunction scan_op = BinaryFunction(),
                                    hipStream_t stream = 0,
                                    bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using result_type = typename ::rocprim::detail::match_result_type<
        input_type, BinaryFunction
    >::type;

    return detail::segmented_scan_impl<false, Config>(
        temporary_storage, storage_size,
        input, output, segments, begin_offsets, end_offsets, result_type(),
        scan_op, stream, debug_synchronous
    );
}

/// \brief Parallel segmented exclusive scan primitive for device level.
///
/// segmented_exclusive_scan function performs a device-wide exclusive scan operation
/// across multiple sequences from \p input using binary \p scan_op operator.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p input and \p output must have at least \p size elements.
/// * Ranges specified by \p begin_offsets and \p end_offsets must have
/// at least \p segments elements. They may use the same sequence <tt>offsets</tt> of at least
/// <tt>segments + 1</tt> elements: <tt>offsets</tt> for \p begin_offsets and
/// <tt>offsets + 1</tt> for \p end_offsets.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p scan_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ RandomAccessIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ RandomAccessIterator concept. It can be a simple pointer type.
/// \tparam OffsetIterator - random-access iterator type of segment offsets. Must meet the
/// requirements of a C++ RandomAccessIterator concept. It can be a simple pointer type.
/// \tparam InitValueType - type of the initial value.
/// \tparam BinaryFunction - type of binary function used for scan operation. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the scan operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to scan.
/// \param [out] output - iterator to the first element in the output range.
/// \param [in] segments - number of segments in the input range.
/// \param [in] begin_offsets - iterator to the first element in the range of beginning offsets.
/// \param [in] end_offsets - iterator to the first element in the range of ending offsets.
/// \param [in] initial_value - initial value to start the scan.
/// \param [in] scan_op - binary operation function object that will be used for scan.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// The default value is \p BinaryFunction().
/// \param [in] stream - [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful scan; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level segmented exclusive min-scan operation is performed on
/// an array of integer values (<tt>short</tt>s are scanned into <tt>int</tt>s) using custom operator.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // custom scan function
/// auto min_op =
///     [] __device__ (int a, int b) -> int
///     {
///         return a < b ? a : b;
///     };
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// int start_value;      // e.g., 9
/// short * input;        // e.g., [4, 7, 6, 2, 5, 1, 3, 8]
/// int   * output;       // empty array of 8 elements
/// size_t segments;      // e.g., 3
/// int * offsets;        // e.g. [0, 2, 4, 8]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::segmented_exclusive_scan(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, segments, offsets, offsets + 1
///     start_value, min_op
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform scan
/// rocprim::exclusive_scan(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, segments, offsets, offsets + 1
///     start_value, min_op
/// );
/// // output: [9, 4, 9, 6, 9, 5, 1, 1]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class InitValueType,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
hipError_t segmented_exclusive_scan(void * temporary_storage,
                                    size_t& storage_size,
                                    InputIterator input,
                                    OutputIterator output,
                                    unsigned int segments,
                                    OffsetIterator begin_offsets,
                                    OffsetIterator end_offsets,
                                    const InitValueType initial_value,
                                    BinaryFunction scan_op = BinaryFunction(),
                                    hipStream_t stream = 0,
                                    bool debug_synchronous = false)
{
    return detail::segmented_scan_impl<true, Config>(
        temporary_storage, storage_size,
        input, output, segments, begin_offsets, end_offsets, initial_value,
        scan_op, stream, debug_synchronous
    );
}

/// \brief Parallel segmented inclusive scan primitive for device level.
///
/// segmented_inclusive_scan function performs a device-wide inclusive scan operation
/// across multiple sequences from \p input using binary \p scan_op operator. Beginnings
/// of the segments should be marked by value convertible to \p true at corresponding
/// position in \p flags range.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p input, \p output, and \p flags must have at least \p size elements.
/// * \p value_type of \p HeadFlagIterator iterator should be convertible to \p bool type.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p scan_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ RandomAccessIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ RandomAccessIterator concept. It can be a simple pointer type.
/// \tparam HeadFlagIterator - random-access iterator type of flags. Must meet the
/// requirements of a C++ RandomAccessIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction - type of binary function used for scan operation. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the scan operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to scan.
/// \param [out] output - iterator to the first element in the output range.
/// \param [in] head_flags - iterator to the first element in the range of head flags marking
/// beginnings of each segment in the input range.
/// \param [in] size - number of element in the input range.
/// \param [in] scan_op - binary operation function object that will be used for scan.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// The default value is \p BinaryFunction().
/// \param [in] stream - [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful scan; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level segmented inclusive sum operation is performed on
/// an array of integer values (<tt>short</tt>s are added into <tt>int</tt>s).
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t size;      // e.g., 8
/// short * input;    // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int * flags;      // e.g., [1, 0, 0, 1, 0, 1, 0, 0]
/// int * output;     // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::segmented_inclusive_scan(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, flags, size, ::rocprim::plus<int>()
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform scan
/// rocprim::inclusive_scan(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, flags, size, ::rocprim::plus<int>()
/// );
/// // output: [1, 3, 6, 4, 9, 6, 13, 21]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class HeadFlagIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
hipError_t segmented_inclusive_scan(void * temporary_storage,
                                    size_t& storage_size,
                                    InputIterator input,
                                    OutputIterator output,
                                    HeadFlagIterator head_flags,
                                    size_t size,
                                    BinaryFunction scan_op = BinaryFunction(),
                                    hipStream_t stream = 0,
                                    bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using result_type = typename ::rocprim::detail::match_result_type<
        input_type, BinaryFunction
    >::type;
    using flag_type = typename std::iterator_traits<HeadFlagIterator>::value_type;
    using headflag_scan_op_wrapper_type =
        detail::headflag_scan_op_wrapper<
            result_type, flag_type, BinaryFunction
        >;

    return inclusive_scan<Config>(
        temporary_storage, storage_size,
        rocprim::make_zip_iterator(rocprim::make_tuple(input, head_flags)),
        rocprim::make_zip_iterator(rocprim::make_tuple(output, rocprim::make_discard_iterator())),
        size, headflag_scan_op_wrapper_type(scan_op),
        stream, debug_synchronous
    );
}

/// \brief Parallel segmented exclusive scan primitive for device level.
///
/// segmented_exclusive_scan function performs a device-wide exclusive scan operation
/// across multiple sequences from \p input using binary \p scan_op operator. Beginnings
/// of the segments should be marked by value convertible to \p true at corresponding
/// position in \p flags range.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p input, \p output, and \p flags must have at least \p size elements.
/// * \p value_type of \p HeadFlagIterator iterator should be convertible to \p bool type.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p scan_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ RandomAccessIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ RandomAccessIterator concept. It can be a simple pointer type.
/// \tparam HeadFlagIterator - random-access iterator type of flags. Must meet the
/// requirements of a C++ RandomAccessIterator concept. It can be a simple pointer type.
/// \tparam InitValueType - type of the initial value.
/// \tparam BinaryFunction - type of binary function used for scan operation. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the scan operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to scan.
/// \param [out] output - iterator to the first element in the output range.
/// \param [in] head_flags - iterator to the first element in the range of head flags marking
/// beginnings of each segment in the input range.
/// \param [in] initial_value - initial value to start the scan.
/// \param [in] size - number of element in the input range.
/// \param [in] scan_op - binary operation function object that will be used for scan.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// The default value is \p BinaryFunction().
/// \param [in] stream - [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful scan; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level segmented exclusive sum operation is performed on
/// an array of integer values (<tt>short</tt>s are added into <tt>int</tt>s).
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t size;      // e.g., 8
/// short * input;    // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int * flags;      // e.g., [1, 0, 0, 1, 0, 1, 0, 0]
/// int init;         // e.g., 9
/// int * output;     // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::segmented_exclusive_scan(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, flags, init, size, ::rocprim::plus<int>()
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform scan
/// rocprim::exclusive_scan(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, flags, init, size, ::rocprim::plus<int>()
/// );
/// // output: [9, 10, 12, 9, 13, 9, 15, 22]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class HeadFlagIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
hipError_t segmented_exclusive_scan(void * temporary_storage,
                                    size_t& storage_size,
                                    InputIterator input,
                                    OutputIterator output,
                                    HeadFlagIterator head_flags,
                                    const InitValueType initial_value,
                                    size_t size,
                                    BinaryFunction scan_op = BinaryFunction(),
                                    hipStream_t stream = 0,
                                    bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using result_type = typename ::rocprim::detail::match_result_type<
        input_type, BinaryFunction
    >::type;
    using flag_type = typename std::iterator_traits<HeadFlagIterator>::value_type;
    using headflag_scan_op_wrapper_type =
        detail::headflag_scan_op_wrapper<
            result_type, flag_type, BinaryFunction
        >;

    const result_type initial_value_converted = static_cast<result_type>(initial_value);

    // Flag the last item of each segment as the next segment's head, use initial_value as its value,
    // then run exclusive scan
    return exclusive_scan<Config>(
        temporary_storage, storage_size,
        rocprim::make_transform_iterator(
            rocprim::make_counting_iterator<size_t>(0),
            [input, head_flags, initial_value_converted, size]
            ROCPRIM_DEVICE
            (const size_t i)
            {
                flag_type flag(false);
                if(i + 1 < size)
                {
                    flag = head_flags[i + 1];
                }
                result_type value = initial_value_converted;
                if(!flag)
                {
                    value = input[i];
                }
                return rocprim::make_tuple(value, flag);
            }
        ),
        rocprim::make_zip_iterator(rocprim::make_tuple(output, rocprim::make_discard_iterator())),
        rocprim::make_tuple(initial_value_converted, flag_type(true)), // init value is a head of the first segment
        size,
        headflag_scan_op_wrapper_type(scan_op),
        stream,
        debug_synchronous
    );
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SEGMENTED_SCAN_HPP_
