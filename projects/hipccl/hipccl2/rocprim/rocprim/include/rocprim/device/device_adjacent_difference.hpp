// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_ADJACENT_DIFFERENCE_HPP_
#define ROCPRIM_DEVICE_DEVICE_ADJACENT_DIFFERENCE_HPP_

#include "detail/device_adjacent_difference.hpp"

#include "device_adjacent_difference_config.hpp"

#include "config_types.hpp"
#include "device_transform.hpp"

#include "../config.hpp"
#include "../functional.hpp"

#include "../detail/various.hpp"
#include "../iterator/counting_iterator.hpp"
#include "../iterator/transform_iterator.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <iostream>
#include <iterator>

#include <cstddef>

/// \file
///
/// Device level adjacent_difference parallel primitives

BEGIN_ROCPRIM_NAMESPACE

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start)                           \
    {                                                                                            \
        auto _error = hipGetLastError();                                                         \
        if(_error != hipSuccess)                                                                 \
            return _error;                                                                       \
        if(debug_synchronous)                                                                    \
        {                                                                                        \
            std::cout << name << "(" << size << ")";                                             \
            auto __error = hipStreamSynchronize(stream);                                         \
            if(__error != hipSuccess)                                                            \
                return __error;                                                                  \
            auto _end = std::chrono::high_resolution_clock::now();                               \
            auto _d   = std::chrono::duration_cast<std::chrono::duration<double>>(_end - start); \
            std::cout << " " << _d.count() * 1000 << " ms" << '\n';                              \
        }                                                                                        \
    }

namespace detail
{
template <typename Config,
          bool InPlace,
          bool Right,
          typename InputIt,
          typename OutputIt,
          typename BinaryFunction>
void ROCPRIM_KERNEL __launch_bounds__(Config::block_size) adjacent_difference_kernel(
    const InputIt                                             input,
    const OutputIt                                            output,
    const std::size_t                                         size,
    const BinaryFunction                                      op,
    const typename std::iterator_traits<InputIt>::value_type* previous_values,
    const std::size_t                                         starting_block)
{
    adjacent_difference_kernel_impl<Config, InPlace, Right>(
        input, output, size, op, previous_values, starting_block);
}

template <typename Config,
          bool InPlace,
          bool Right,
          typename InputIt,
          typename OutputIt,
          typename BinaryFunction>
hipError_t adjacent_difference_impl(void* const          temporary_storage,
                                    std::size_t&         storage_size,
                                    const InputIt        input,
                                    const OutputIt       output,
                                    const std::size_t    size,
                                    const BinaryFunction op,
                                    const hipStream_t    stream,
                                    const bool           debug_synchronous)
{
    using value_type = typename std::iterator_traits<InputIt>::value_type;

    using config = detail::default_or_custom_config<
        Config,
        detail::default_adjacent_difference_config<ROCPRIM_TARGET_ARCH, value_type>>;

    static constexpr unsigned int block_size       = config::block_size;
    static constexpr unsigned int items_per_thread = config::items_per_thread;
    static constexpr unsigned int items_per_block  = block_size * items_per_thread;

    const std::size_t num_blocks = ceiling_div(size, items_per_block);

    if(temporary_storage == nullptr)
    {
        if(InPlace && num_blocks >= 2)
        {
            storage_size = align_size((num_blocks - 1) * sizeof(value_type));
        }
        else
        {
            // Make sure user won't try to allocate 0 bytes memory, because
            // hipMalloc will return nullptr when size is zero.
            storage_size = 4;
        }

        return hipSuccess;
    }

    if(num_blocks == 0)
    {
        return hipSuccess;
    }

    // Copy values before they are overwritten to use as tile predecessors/successors
    // this is not dereferenced when the operation is not in place
    auto* const previous_values = static_cast<value_type*>(temporary_storage);
    if ROCPRIM_IF_CONSTEXPR(InPlace)
    {
        // If doing left adjacent diff then the last item of each block is needed for the
        // next block, otherwise the first item is needed for the previous block
        static constexpr auto offset = items_per_block - (Right ? 0 : 1);

        const auto block_starts_iter = make_transform_iterator(
            rocprim::make_counting_iterator(std::size_t {0}),
            [base = input + offset](std::size_t i) { return base[i * items_per_block]; });

        const hipError_t error = ::rocprim::transform(block_starts_iter,
                                                      previous_values,
                                                      num_blocks - 1,
                                                      rocprim::identity<> {},
                                                      stream,
                                                      debug_synchronous);
        if(error != hipSuccess)
        {
            return error;
        }
    }

    static constexpr unsigned int size_limit     = config::size_limit;
    static constexpr auto number_of_blocks_limit = std::max(size_limit / items_per_block, 1u);
    static constexpr auto aligned_size_limit     = number_of_blocks_limit * items_per_block;

    // Launch number_of_blocks_limit blocks while there is still at least as many blocks
    // left as the limit
    const auto number_of_launch = ceiling_div(size, aligned_size_limit);

    if(debug_synchronous)
    {
        std::cout << "----------------------------------\n";
        std::cout << "size:               " << size << '\n';
        std::cout << "aligned_size_limit: " << aligned_size_limit << '\n';
        std::cout << "number_of_launch:   " << number_of_launch << '\n';
        std::cout << "block_size:         " << block_size << '\n';
        std::cout << "items_per_block:    " << items_per_block << '\n';
        std::cout << "----------------------------------\n";
    }

    for(std::size_t i = 0, offset = 0; i < number_of_launch; ++i, offset += aligned_size_limit)
    {
        const auto current_size
            = static_cast<unsigned int>(std::min<std::size_t>(size - offset, aligned_size_limit));
        const auto current_blocks = ceiling_div(current_size, items_per_block);
        const auto starting_block = i * number_of_blocks_limit;

        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        if(debug_synchronous)
        {
            std::cout << "index:            " << i << '\n';
            std::cout << "current_size:     " << current_size << '\n';
            std::cout << "number of blocks: " << current_blocks << '\n';

            start = std::chrono::high_resolution_clock::now();
        }
        hipLaunchKernelGGL(HIP_KERNEL_NAME(adjacent_difference_kernel<config, InPlace, Right>),
                           dim3(current_blocks),
                           dim3(block_size),
                           0,
                           stream,
                           input + offset,
                           output + offset,
                           size,
                           op,
                           previous_values + starting_block,
                           starting_block);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(
            "adjacent_difference_kernel", current_size, start);
    }
    return hipSuccess;
}
} // namespace detail

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

#endif // DOXYGEN_SHOULD_SKIP_THIS

/// \addtogroup devicemodule
/// @{

/// \brief Parallel primitive for applying a binary operation across pairs of consecutive elements
/// in device accessible memory. Writes the output to the position of the left item.
///
/// Copies the first item to the output then performs calls the supplied operator with each pair
/// of neighboring elements and writes its result to the location of the second element.
/// Equivalent to the following code
/// \code{.cpp}
/// output[0] = input[0];
/// for(std::size_t int i = 1; i < size; ++i)
/// {
///     output[i] = op(input[i], input[i - 1]);
/// }
/// \endcode
///
/// \tparam Config - [optional] configuration of the primitive. It can be
/// `adjacent_difference_config` or a class with the same members.
/// \tparam InputIt - [inferred] random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIt - [inferred] random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction - [inferred] binary operation function object that will be applied to
/// consecutive items. The signature of the function should be equivalent to the following:
/// `U f(const T1& a, const T2& b)`. The signature does not need to have
/// `const &`, but function object must not modify the object passed to it
/// \param temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// `storage_size` and function returns without performing the scan operation
/// \param storage_size - reference to a size (in bytes) of `temporary_storage`
/// \param input - iterator to the input range
/// \param output - iterator to the output range, must have any overlap with input
/// \param size - number of items in the input
/// \param op - [optional] the binary operation to apply
/// \param stream - [optional] HIP stream object. Default is `0` (the default stream)
/// \param debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors and extra debugging info is printed to the
/// standard output. Default value is `false`
///
/// \return `hipSuccess` (0) after successful scan, otherwise the HIP runtime error of
/// type `hipError_t`
///
/// \par Example
/// \parblock
/// In this example a device-level adjacent_difference operation is performed on integer values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp> //or <rocprim/device/device_adjacent_difference.hpp>
///
/// // custom binary function
/// auto binary_op =
///     [] __device__ (int a, int b) -> int
///     {
///         return a - b;
///     };
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// std::size_t size; // e.g., 8
/// int* input1; // e.g., [8, 7, 6, 5, 4, 3, 2, 1]
/// int* output; // empty array of 8 elements
///
/// std::size_t temporary_storage_size_bytes;
/// void* temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::adjacent_difference(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, size, binary_op
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform adjacent difference
/// rocprim::adjacent_difference(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, size, binary_op
/// );
/// // output: [8, 1, 1, 1, 1, 1, 1, 1]
/// \endcode
/// \endparblock
template <typename Config = default_config,
          typename InputIt,
          typename OutputIt,
          typename BinaryFunction = ::rocprim::minus<>>
hipError_t adjacent_difference(void* const          temporary_storage,
                               std::size_t&         storage_size,
                               const InputIt        input,
                               const OutputIt       output,
                               const std::size_t    size,
                               const BinaryFunction op                = BinaryFunction {},
                               const hipStream_t    stream            = 0,
                               const bool           debug_synchronous = false)
{
    static constexpr bool in_place = false;
    static constexpr bool right    = false;
    return detail::adjacent_difference_impl<Config, in_place, right>(
        temporary_storage, storage_size, input, output, size, op, stream, debug_synchronous);
}

/// \brief Parallel primitive for applying a binary operation across pairs of consecutive elements
/// in device accessible memory. Writes the output to the position of the left item in place.
///
/// Copies the first item to the output then performs calls the supplied operator with each pair
/// of neighboring elements and writes its result to the location of the second element.
/// Equivalent to the following code
/// \code{.cpp}
/// for(std::size_t int i = size - 1; i > 0; --i)
/// {
///     input[i] = op(input[i], input[i - 1]);
/// }
/// \endcode
///
/// \tparam Config - [optional] configuration of the primitive. It can be
/// `adjacent_difference_config` or a class with the same members.
/// \tparam InputIt - [inferred] random-access iterator type of the value range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction - [inferred] binary operation function object that will be applied to
/// consecutive items. The signature of the function should be equivalent to the following:
/// `U f(const T1& a, const T2& b)`. The signature does not need to have
/// `const &`, but function object must not modify the object passed to it
/// \param temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// `storage_size` and function returns without performing the scan operation
/// \param storage_size - reference to a size (in bytes) of `temporary_storage`
/// \param values - iterator to the range values, will be overwritten with the results
/// \param size - number of items in the input
/// \param op - [optional] the binary operation to apply
/// \param stream - [optional] HIP stream object. Default is `0` (the default stream)
/// \param debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors and extra debugging info is printed to the
/// standard output. Default value is `false`
///
/// \return `hipSuccess` (0) after successful scan, otherwise the HIP runtime error of
/// type `hipError_t`
template <typename Config = default_config,
          typename InputIt,
          typename BinaryFunction = ::rocprim::minus<>>
hipError_t adjacent_difference_inplace(void* const          temporary_storage,
                                       std::size_t&         storage_size,
                                       const InputIt        values,
                                       const std::size_t    size,
                                       const BinaryFunction op                = BinaryFunction {},
                                       const hipStream_t    stream            = 0,
                                       const bool           debug_synchronous = false)
{
    static constexpr bool in_place = true;
    static constexpr bool right    = false;
    return detail::adjacent_difference_impl<Config, in_place, right>(
        temporary_storage, storage_size, values, values, size, op, stream, debug_synchronous);
}

/// \brief Parallel primitive for applying a binary operation across pairs of consecutive elements
/// in device accessible memory. Writes the output to the position of the right item.
///
/// Copies the last item to the output then performs calls the supplied operator with each pair
/// of neighboring elements and writes its result to the location of the first element.
/// Equivalent to the following code
/// \code{.cpp}
/// output[size - 1] = input[size - 1];
/// for(std::size_t int i = 0; i < size - 1; ++i)
/// {
///     output[i] = op(input[i], input[i + 1]);
/// }
/// \endcode
///
/// \tparam Config - [optional] configuration of the primitive. It can be
/// `adjacent_difference_config` or a class with the same members.
/// \tparam InputIt - [inferred] random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIt - [inferred] random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction - [inferred] binary operation function object that will be applied to
/// consecutive items. The signature of the function should be equivalent to the following:
/// `U f(const T1& a, const T2& b)`. The signature does not need to have
/// `const &`, but function object must not modify the object passed to it
/// \param temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// `storage_size` and function returns without performing the scan operation
/// \param storage_size - reference to a size (in bytes) of `temporary_storage`
/// \param input - iterator to the input range
/// \param output - iterator to the output range, must have any overlap with input
/// \param size - number of items in the input
/// \param op - [optional] the binary operation to apply
/// \param stream - [optional] HIP stream object. Default is `0` (the default stream)
/// \param debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors and extra debugging info is printed to the
/// standard output. Default value is `false`
///
/// \return `hipSuccess` (0) after successful scan, otherwise the HIP runtime error of
/// type `hipError_t`
///
/// \par Example
/// \parblock
/// In this example a device-level adjacent_difference operation is performed on integer values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp> //or <rocprim/device/device_adjacent_difference.hpp>
///
/// // custom binary function
/// auto binary_op =
///     [] __device__ (int a, int b) -> int
///     {
///         return a - b;
///     };
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// std::size_t size; // e.g., 8
/// int* input1; // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int* output; // empty array of 8 elements
///
/// std::size_t temporary_storage_size_bytes;
/// void* temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::adjacent_difference_right(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, size, binary_op
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform adjacent difference
/// rocprim::adjacent_difference_right(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, size, binary_op
/// );
/// // output: [1, 1, 1, 1, 1, 1, 1, 8]
/// \endcode
/// \endparblock
template <typename Config = default_config,
          typename InputIt,
          typename OutputIt,
          typename BinaryFunction = ::rocprim::minus<>>
hipError_t adjacent_difference_right(void* const          temporary_storage,
                                     std::size_t&         storage_size,
                                     const InputIt        input,
                                     const OutputIt       output,
                                     const std::size_t    size,
                                     const BinaryFunction op                = BinaryFunction {},
                                     const hipStream_t    stream            = 0,
                                     const bool           debug_synchronous = false)
{
    static constexpr bool in_place = false;
    static constexpr bool right    = true;
    return detail::adjacent_difference_impl<Config, in_place, right>(
        temporary_storage, storage_size, input, output, size, op, stream, debug_synchronous);
}

/// \brief Parallel primitive for applying a binary operation across pairs of consecutive elements
/// in device accessible memory. Writes the output to the position of the right item in place.
///
/// Copies the last item to the output then performs calls the supplied operator with each pair
/// of neighboring elements and writes its result to the location of the first element.
/// Equivalent to the following code
/// \code{.cpp}
/// for(std::size_t int i = 0; i < size - 1; --i)
/// {
///     input[i] = op(input[i], input[i + 1]);
/// }
/// \endcode
///
/// \tparam Config - [optional] configuration of the primitive. It can be
/// `adjacent_difference_config` or a class with the same members.
/// \tparam InputIt - [inferred] random-access iterator type of the value range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction - [inferred] binary operation function object that will be applied to
/// consecutive items. The signature of the function should be equivalent to the following:
/// `U f(const T1& a, const T2& b)`. The signature does not need to have
/// `const &`, but function object must not modify the object passed to it
/// \param temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// `storage_size` and function returns without performing the scan operation
/// \param storage_size - reference to a size (in bytes) of `temporary_storage`
/// \param values - iterator to the range values, will be overwritten with the results
/// \param size - number of items in the input
/// \param op - [optional] the binary operation to apply
/// \param stream - [optional] HIP stream object. Default is `0` (the default stream)
/// \param debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors and extra debugging info is printed to the
/// standard output. Default value is `false`
///
/// \return `hipSuccess` (0) after successful scan, otherwise the HIP runtime error of
/// type `hipError_t`
template <typename Config = default_config,
          typename InputIt,
          typename BinaryFunction = ::rocprim::minus<>>
hipError_t adjacent_difference_right_inplace(void* const          temporary_storage,
                                             std::size_t&         storage_size,
                                             const InputIt        values,
                                             const std::size_t    size,
                                             const BinaryFunction op     = BinaryFunction {},
                                             const hipStream_t    stream = 0,
                                             const bool           debug_synchronous = false)
{
    static constexpr bool in_place = true;
    static constexpr bool right    = true;
    return detail::adjacent_difference_impl<Config, in_place, right>(
        temporary_storage, storage_size, values, values, size, op, stream, debug_synchronous);
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_ADJACENT_DIFFERENCE_HPP_
