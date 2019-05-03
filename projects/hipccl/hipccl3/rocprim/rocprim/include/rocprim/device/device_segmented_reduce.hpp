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

#ifndef ROCPRIM_DEVICE_DEVICE_SEGMENTED_REDUCE_HPP_
#define ROCPRIM_DEVICE_DEVICE_SEGMENTED_REDUCE_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../functional.hpp"
#include "../detail/various.hpp"
#include "../detail/match_result_type.hpp"

#include "detail/device_segmented_reduce.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

template<
    class Config,
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class ResultType,
    class BinaryFunction
>
__global__
void segmented_reduce_kernel(InputIterator input,
                             OutputIterator output,
                             OffsetIterator begin_offsets,
                             OffsetIterator end_offsets,
                             BinaryFunction reduce_op,
                             ResultType initial_value)
{
    segmented_reduce<Config>(
        input, output,
        begin_offsets, end_offsets,
        reduce_op, initial_value
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
    class Config,
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class InitValueType,
    class BinaryFunction
>
inline
hipError_t segmented_reduce_impl(void * temporary_storage,
                                 size_t& storage_size,
                                 InputIterator input,
                                 OutputIterator output,
                                 unsigned int segments,
                                 OffsetIterator begin_offsets,
                                 OffsetIterator end_offsets,
                                 BinaryFunction reduce_op,
                                 InitValueType initial_value,
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
        default_reduce_config<ROCPRIM_TARGET_ARCH, result_type>
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
        HIP_KERNEL_NAME(segmented_reduce_kernel<config>),
        dim3(segments), dim3(block_size), 0, stream,
        input, output,
        begin_offsets, end_offsets,
        reduce_op, static_cast<result_type>(initial_value)
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("segmented_reduce", segments, start);

    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

} // end of detail namespace

/// \brief Parallel segmented reduction primitive for device level.
///
/// segmented_reduce function performs a device-wide reduction operation across multiple sequences
/// using binary \p reduce_op operator.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p input must have at least \p size elements, \p output must have
/// \p segments elements.
/// * Ranges specified by \p begin_offsets and \p end_offsets must have
/// at least \p segments elements. They may use the same sequence <tt>offsets</tt> of at least
/// <tt>segments + 1</tt> elements: <tt>offsets</tt> for \p begin_offsets and
/// <tt>offsets + 1</tt> for \p end_offsets.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p reduce_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam OffsetIterator - random-access iterator type of segment offsets. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction - type of binary function used for reduction. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
/// \tparam InitValueType - type of the initial value.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to reduce.
/// \param [out] output - iterator to the first element in the output range.
/// \param [in] segments - number of segments in the input range.
/// \param [in] begin_offsets - iterator to the first element in the range of beginning offsets.
/// \param [in] end_offsets - iterator to the first element in the range of ending offsets.
/// \param [in] initial_value - initial value to start the reduction.
/// \param [in] reduce_op - binary operation function object that will be used for reduction.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// The default value is \p BinaryFunction().
/// \param [in] stream - [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful reduction; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level segmented min-reduction operation is performed on an array of
/// integer values (<tt>short</tt>s are reduced into <tt>int</tt>s) using custom operator.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // custom reduce function
/// auto min_op =
///     [] __device__ (int a, int b) -> int
///     {
///         return a < b ? a : b;
///     };
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// unsigned int segments;   // e.g., 3
/// short * input;           // e.g., [4, 7, 6, 2, 5, 1, 3, 8]
/// int * output;            // empty array of 3 elements
/// int * offsets;           // e.g. [0, 2, 3, 8]
/// int init_value;          // e.g., 9
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::segmented_reduce(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output,
///     segments, offsets, offsets + 1,
///     min_op, init_value
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform segmented reduction
/// rocprim::segmented_reduce(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output,
///     segments, offsets, offsets + 1,
///     min_op, init_value
/// );
/// // output: [4, 6, 1]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>,
    class InitValueType = typename std::iterator_traits<InputIterator>::value_type
>
inline
hipError_t segmented_reduce(void * temporary_storage,
                            size_t& storage_size,
                            InputIterator input,
                            OutputIterator output,
                            unsigned int segments,
                            OffsetIterator begin_offsets,
                            OffsetIterator end_offsets,
                            BinaryFunction reduce_op = BinaryFunction(),
                            InitValueType initial_value = InitValueType(),
                            hipStream_t stream = 0,
                            bool debug_synchronous = false)
{
    return detail::segmented_reduce_impl<Config>(
        temporary_storage, storage_size,
        input, output,
        segments, begin_offsets, end_offsets,
        reduce_op, initial_value,
        stream, debug_synchronous
    );
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SEGMENTED_REDUCE_HPP_
