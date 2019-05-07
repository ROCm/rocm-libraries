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

#ifndef ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HPP_
#define ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../iterator/constant_iterator.hpp"
#include "../iterator/counting_iterator.hpp"
#include "../iterator/discard_iterator.hpp"
#include "../iterator/zip_iterator.hpp"

#include "device_run_length_encode_config.hpp"
#include "device_reduce_by_key.hpp"
#include "device_select.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start) \
    { \
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

} // end detail namespace

/// \brief Parallel run-length encoding for device level.
///
/// run_length_encode function performs a device-wide run-length encoding of runs (groups)
/// of consecutive values. The first value of each run is copied to \p unique_output and
/// the length of the run is written to \p counts_output.
/// The total number of runs is written to \p runs_count_output.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Range specified by \p input must have at least \p size elements.
/// * Range specified by \p runs_count_output must have at least 1 element.
/// * Ranges specified by \p unique_output and \p counts_output must have at least
/// <tt>*runs_count_output</tt> (i.e. the number of runs) elements.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p run_length_encode_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam UniqueOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam CountsOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam RunsCountOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range of values.
/// \param [in] size - number of element in the input range.
/// \param [out] unique_output - iterator to the first element in the output range of unique values.
/// \param [out] counts_output - iterator to the first element in the output range of lenghts.
/// \param [out] runs_count_output - iterator to total number of runs.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level run-length encoding operation is performed on an array of
/// integer values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// int * input;                // e.g., [1, 1, 1, 2, 10, 10, 10, 88]
/// int * unique_output;        // empty array of at least 4 elements
/// int * counts_output;        // empty array of at least 4 elements
/// int * runs_count_output;    // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::run_length_encode(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, input_size,
///     unique_output, counts_output, runs_count_output
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform encoding
/// rocprim::run_length_encode(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, input_size,
///     unique_output, counts_output, runs_count_output
/// );
/// // unique_output:     [1, 2, 10, 88]
/// // counts_output:     [3, 1,  3,  1]
/// // runs_count_output: [4]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class UniqueOutputIterator,
    class CountsOutputIterator,
    class RunsCountOutputIterator
>
inline
hipError_t run_length_encode(void * temporary_storage,
                             size_t& storage_size,
                             InputIterator input,
                             unsigned int size,
                             UniqueOutputIterator unique_output,
                             CountsOutputIterator counts_output,
                             RunsCountOutputIterator runs_count_output,
                             hipStream_t stream = 0,
                             bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using count_type = unsigned int;

    using config = detail::default_or_custom_config<
        Config,
        detail::default_run_length_encode_config
    >;

    return ::rocprim::reduce_by_key<typename config::reduce_by_key>(
        temporary_storage, storage_size,
        input, make_constant_iterator<count_type>(1), size,
        unique_output, counts_output, runs_count_output,
        ::rocprim::plus<count_type>(), ::rocprim::equal_to<input_type>(),
        stream, debug_synchronous
    );
}

/// \brief Parallel run-length encoding of non-trivial runs for device level.
///
/// run_length_encode_non_trivial_runs function performs a device-wide run-length encoding of
/// non-trivial runs (groups) of consecutive values (groups of more than one element).
/// The offset of the first value of each non-trivial run is copied to \p offsets_output and
/// the length of the run (the count of elements) is written to \p counts_output.
/// The total number of non-trivial runs is written to \p runs_count_output.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Range specified by \p input must have at least \p size elements.
/// * Range specified by \p runs_count_output must have at least 1 element.
/// * Ranges specified by \p offsets_output and \p counts_output must have at least
/// <tt>*runs_count_output</tt> (i.e. the number of non-trivial runs) elements.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p run_length_encode_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OffsetsOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam CountsOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam RunsCountOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range of values.
/// \param [in] size - number of element in the input range.
/// \param [out] offsets_output - iterator to the first element in the output range of offsets.
/// \param [out] counts_output - iterator to the first element in the output range of lenghts.
/// \param [out] runs_count_output - iterator to total number of runs.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level run-length encoding of non-trivial runs is performed on an array of
/// integer values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// int * input;                // e.g., [1, 1, 1, 2, 10, 10, 10, 88]
/// int * offsets_output;       // empty array of at least 2 elements
/// int * counts_output;        // empty array of at least 2 elements
/// int * runs_count_output;    // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::run_length_encode_non_trivial_runs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, input_size,
///     offsets_output, counts_output, runs_count_output
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform encoding
/// rocprim::run_length_encode_non_trivial_runs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, input_size,
///     offsets_output, counts_output, runs_count_output
/// );
/// // offsets_output:    [0, 4]
/// // counts_output:     [3, 3]
/// // runs_count_output: [2]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class OffsetsOutputIterator,
    class CountsOutputIterator,
    class RunsCountOutputIterator
>
inline
hipError_t run_length_encode_non_trivial_runs(void * temporary_storage,
                                              size_t& storage_size,
                                              InputIterator input,
                                              unsigned int size,
                                              OffsetsOutputIterator offsets_output,
                                              CountsOutputIterator counts_output,
                                              RunsCountOutputIterator runs_count_output,
                                              hipStream_t stream = 0,
                                              bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using offset_type = unsigned int;
    using count_type = unsigned int;
    using offset_count_pair = typename ::rocprim::tuple<offset_type, count_type>;

    using config = detail::default_or_custom_config<
        Config,
        detail::default_run_length_encode_config
    >;

    hipError_t error;

    auto reduce_op = [] __device__ (const offset_count_pair& a, const offset_count_pair& b)
    {
        return offset_count_pair(
            ::rocprim::get<0>(a), // Always use offset of the first item of the run
            ::rocprim::get<1>(a) + ::rocprim::get<1>(b) // Number of items in the run
        );
    };
    auto non_trivial_runs_select_op = [] __device__ (const offset_count_pair& a)
    {
        return ::rocprim::get<1>(a) > 1;
    };

    offset_type * offsets_tmp = nullptr;
    count_type * counts_tmp = nullptr;
    count_type * all_runs_count_tmp = nullptr;

    // Calculate size of temporary storage for reduce_by_key operation
    size_t reduce_by_key_bytes;
    error = ::rocprim::reduce_by_key<typename config::reduce_by_key>(
        nullptr, reduce_by_key_bytes,
        input,
        ::rocprim::make_zip_iterator(
            ::rocprim::make_tuple(
                ::rocprim::make_counting_iterator<offset_type>(0),
                ::rocprim::make_constant_iterator<count_type>(1)
            )
        ),
        size,
        ::rocprim::make_discard_iterator(),
        ::rocprim::make_zip_iterator(::rocprim::make_tuple(offsets_tmp, counts_tmp)),
        all_runs_count_tmp,
        reduce_op, ::rocprim::equal_to<input_type>(),
        stream, debug_synchronous
    );
    if(error != hipSuccess) return error;
    reduce_by_key_bytes = ::rocprim::detail::align_size(reduce_by_key_bytes);

    // Calculate size of temporary storage for select operation
    size_t select_bytes;
    error = ::rocprim::select<typename config::select>(
        nullptr, select_bytes,
        ::rocprim::make_zip_iterator(::rocprim::make_tuple(offsets_tmp, counts_tmp)),
        ::rocprim::make_zip_iterator(::rocprim::make_tuple(offsets_output, counts_output)),
        runs_count_output,
        size,
        non_trivial_runs_select_op,
        stream, debug_synchronous
    );
    if(error != hipSuccess) return error;
    select_bytes = ::rocprim::detail::align_size(select_bytes);

    const size_t offsets_tmp_bytes = ::rocprim::detail::align_size(size * sizeof(offset_type));
    const size_t counts_tmp_bytes = ::rocprim::detail::align_size(size * sizeof(count_type));
    const size_t all_runs_count_tmp_bytes = sizeof(count_type);
    if(temporary_storage == nullptr)
    {
        storage_size = ::rocprim::max(reduce_by_key_bytes, select_bytes) +
            offsets_tmp_bytes + counts_tmp_bytes + all_runs_count_tmp_bytes;
        return hipSuccess;
    }

    char * ptr = reinterpret_cast<char *>(temporary_storage);
    ptr += ::rocprim::max(reduce_by_key_bytes, select_bytes);
    offsets_tmp = reinterpret_cast<offset_type *>(ptr);
    ptr += offsets_tmp_bytes;
    counts_tmp = reinterpret_cast<count_type *>(ptr);
    ptr += counts_tmp_bytes;
    all_runs_count_tmp = reinterpret_cast<count_type *>(ptr);

    std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    error = ::rocprim::reduce_by_key<typename config::reduce_by_key>(
        temporary_storage, reduce_by_key_bytes,
        input,
        ::rocprim::make_zip_iterator(
            ::rocprim::make_tuple(
                ::rocprim::make_counting_iterator<offset_type>(0),
                ::rocprim::make_constant_iterator<count_type>(1)
            )
        ),
        size,
        ::rocprim::make_discard_iterator(), // Ignore unique output
        ::rocprim::make_zip_iterator(rocprim::make_tuple(offsets_tmp, counts_tmp)),
        all_runs_count_tmp,
        reduce_op, ::rocprim::equal_to<input_type>(),
        stream, debug_synchronous
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("rocprim::reduce_by_key", size, start)

    // Read count of all runs (including trivial runs)
    count_type all_runs_count;
    error = hipMemcpyAsync(&all_runs_count, all_runs_count_tmp, sizeof(count_type), hipMemcpyDeviceToHost, stream);
    if(error != hipSuccess) return error;
    error = hipStreamSynchronize(stream);
    if(error != hipSuccess) return error;

    // Select non-trivial runs
    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    error = ::rocprim::select<typename config::select>(
        temporary_storage, select_bytes,
        ::rocprim::make_zip_iterator(::rocprim::make_tuple(offsets_tmp, counts_tmp)),
        ::rocprim::make_zip_iterator(::rocprim::make_tuple(offsets_output, counts_output)),
        runs_count_output,
        all_runs_count,
        non_trivial_runs_select_op,
        stream, debug_synchronous
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("rocprim::select", all_runs_count, start)

    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HPP_
