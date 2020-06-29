// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_REDUCE_HPP_
#define ROCPRIM_DEVICE_DEVICE_REDUCE_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../detail/match_result_type.hpp"

#include "device_reduce_config.hpp"
#include "detail/device_reduce.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

template<
    bool WithInitialValue,
    class Config,
    class ResultType,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction
>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void block_reduce_kernel(InputIterator input,
                         const size_t size,
                         OutputIterator output,
                         InitValueType initial_value,
                         BinaryFunction reduce_op)
{
    block_reduce_kernel_impl<WithInitialValue, Config, ResultType>(
        input, size, output, initial_value, reduce_op
    );
}

#define ROCPRIM_DETAIL_HIP_SYNC(name, size, start) \
    if(debug_synchronous) \
    { \
        std::cout << name << "(" << size << ")"; \
        auto error = hipStreamSynchronize(stream); \
        if(error != hipSuccess) return error; \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); \
        std::cout << " " << d.count() * 1000 << " ms" << '\n'; \
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
    bool WithInitialValue, // true when inital_value should be used in reduction
    class Config,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction
>
inline
hipError_t reduce_impl(void * temporary_storage,
                       size_t& storage_size,
                       InputIterator input,
                       OutputIterator output,
                       const InitValueType initial_value,
                       const size_t size,
                       BinaryFunction reduce_op,
                       const hipStream_t stream,
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
    constexpr unsigned int items_per_thread = config::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;

    if(temporary_storage == nullptr)
    {
        storage_size = reduce_get_temporary_storage_bytes<result_type>(size, items_per_block);
        // Make sure user won't try to allocate 0 bytes memory
        storage_size = storage_size == 0 ? 4 : storage_size;
        return hipSuccess;
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    auto number_of_blocks = (size + items_per_block - 1)/items_per_block;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
    }

    if(number_of_blocks > 1)
    {
        // Pointer to array with block_prefixes
        result_type * block_prefixes = static_cast<result_type*>(temporary_storage);

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::block_reduce_kernel<false, config, result_type>),
            dim3(number_of_blocks), dim3(block_size), 0, stream,
            input, size, block_prefixes, initial_value, reduce_op
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("block_reduce_kernel", size, start);

        void * nested_temp_storage = static_cast<void*>(block_prefixes + number_of_blocks);
        auto nested_temp_storage_size = storage_size - (number_of_blocks * sizeof(result_type));

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        auto error = reduce_impl<WithInitialValue, config>(
            nested_temp_storage,
            nested_temp_storage_size,
            block_prefixes, // input
            output, // output
            initial_value,
            number_of_blocks, // input size
            reduce_op,
            stream,
            debug_synchronous
        );
        if(error != hipSuccess) return error;
        ROCPRIM_DETAIL_HIP_SYNC("nested_device_reduce", number_of_blocks, start);
    }
    else
    {
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::block_reduce_kernel<WithInitialValue, config, result_type>),
            dim3(1), dim3(block_size), 0, stream,
            input, size, output, initial_value, reduce_op
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("block_reduce_kernel", size, start);
    }

    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR
#undef ROCPRIM_DETAIL_HIP_SYNC

} // end of detail namespace

/// \brief Parallel reduction primitive for device level.
///
/// reduce function performs a device-wide reduction operation
/// using binary \p reduce_op operator.
///
/// \par Overview
/// * Does not support non-commutative reduction operators. Reduction operator should also be
/// associative. When used with non-associative functions the results may be non-deterministic
/// and/or vary in precision.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p input must have at least \p size elements, while \p output
/// only needs one element.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p reduce_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam InitValueType - type of the initial value.
/// \tparam BinaryFunction - type of binary function used for reduction. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to reduce.
/// \param [out] output - iterator to the first element in the output range. It can be
/// same as \p input.
/// \param [in] initial_value - initial value to start the reduction.
/// \param [in] size - number of element in the input range.
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
/// In this example a device-level min-reduction operation is performed on an array of
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
/// size_t input_size;    // e.g., 8
/// short * input;        // e.g., [4, 7, 6, 2, 5, 1, 3, 8]
/// int * output;         // empty array of 1 element
/// int start_value;      // e.g., 9
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::reduce(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, start_value, input_size, min_op
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform reduce
/// rocprim::reduce(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, start_value, input_size, min_op
/// );
/// // output: [1]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
hipError_t reduce(void * temporary_storage,
                 size_t& storage_size,
                 InputIterator input,
                 OutputIterator output,
                 const InitValueType initial_value,
                 const size_t size,
                 BinaryFunction reduce_op = BinaryFunction(),
                 const hipStream_t stream = 0,
                 bool debug_synchronous = false)
{
    return detail::reduce_impl<true, Config>(
        temporary_storage, storage_size,
        input, output, initial_value, size,
        reduce_op, stream, debug_synchronous
    );
}

/// \brief Parallel reduce primitive for device level.
///
/// reduce function performs a device-wide reduction operation
/// using binary \p reduce_op operator.
///
/// \par Overview
/// * Does not support non-commutative reduction operators. Reduction operator should also be
/// associative. When used with non-associative functions the results may be non-deterministic
/// and/or vary in precision.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p input must have at least \p size elements, while \p output
/// only needs one element.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p reduce_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction - type of binary function used for reduction. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to reduce.
/// \param [out] output - iterator to the first element in the output range. It can be
/// same as \p input.
/// \param [in] size - number of element in the input range.
/// \param [in] reduce_op - binary operation function object that will be used for reduction.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is BinaryFunction().
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful reduction; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level sum operation is performed on an array of
/// integer values (<tt>short</tt>s are reduced into <tt>int</tt>s).
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;    // e.g., 8
/// short * input;        // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int * output;         // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::reduce(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size, rocprim::plus<int>()
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform reduce
/// rocprim::reduce(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size, rocprim::plus<int>()
/// );
/// // output: [36]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
hipError_t reduce(void * temporary_storage,
                  size_t& storage_size,
                  InputIterator input,
                  OutputIterator output,
                  const size_t size,
                  BinaryFunction reduce_op = BinaryFunction(),
                  const hipStream_t stream = 0,
                  bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;

    return detail::reduce_impl<false, Config>(
        temporary_storage, storage_size,
        input, output, input_type(), size,
        reduce_op, stream, debug_synchronous
    );
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_REDUCE_HPP_
