// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_TRANSFORM_HPP_
#define ROCPRIM_DEVICE_DEVICE_TRANSFORM_HPP_

#include "../common.hpp"
#include "../config.hpp"
#include "../detail/various.hpp"
#include "../iterator/zip_iterator.hpp"
#include "../types/tuple.hpp"

#include "config_types.hpp"
#include "detail/device_transform.hpp"
#include "device_transform_config.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <tuple>
#include <type_traits>

/// \addtogroup devicemodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Config,
         bool IsPointer,
         class ResultType,
         class InputIt,
         class OutputIt,
         class UnaryOp>
inline hipError_t launch_transform(detail::target_arch arch,
                                   InputIt             in,
                                   OutputIt            out,
                                   size_t              n,
                                   UnaryOp             op,
                                   dim3                grid,
                                   dim3                block,
                                   size_t              shmem,
                                   hipStream_t         stream)
{
    auto kernel = [=](auto arch_config)
    {
        constexpr auto params = decltype(arch_config)::params;

        detail::transform_kernel_impl<IsPointer,
                                      params.kernel_config.block_size,
                                      params.kernel_config.items_per_thread,
                                      params.load_type,
                                      ResultType>(in, n, out, op);
    };

    return execute_launch_plan<Config>(arch, kernel, grid, block, shmem, stream);
}

template<bool IsPointer,
         class Config,
         class InputIterator,
         class OutputIterator,
         class UnaryFunction>
inline hipError_t transform_impl(InputIterator     input,
                                 OutputIterator    output,
                                 const size_t      size,
                                 UnaryFunction     transform_op,
                                 const hipStream_t stream,
                                 bool              debug_synchronous)
{
    if(size == size_t(0))
    {
        return hipSuccess;
    }

    using input_type  = typename std::iterator_traits<InputIterator>::value_type;
    using result_type = typename ::rocprim::invoke_result<UnaryFunction, input_type>::type;

    using config = detail::wrapped_transform_config<Config, input_type, IsPointer>;

    detail::target_arch target_arch;
    hipError_t          result = detail::host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }
    const detail::transform_config_params params
        = detail::dispatch_target_arch<config, false>(target_arch);

    const unsigned int block_size       = params.kernel_config.block_size;
    const unsigned int items_per_thread = params.kernel_config.items_per_thread;
    const auto         items_per_block  = block_size * items_per_thread;

    // Start point for time measurements
    std::chrono::steady_clock::time_point start;

    const auto size_limit             = params.kernel_config.size_limit;
    const auto number_of_blocks_limit = ::rocprim::max<size_t>(size_limit / items_per_block, 1);

    auto number_of_blocks = (size + items_per_block - 1) / items_per_block;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "number of blocks limit " << number_of_blocks_limit << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
    }

    const auto aligned_size_limit = number_of_blocks_limit * items_per_block;

    // Launch number_of_blocks_limit blocks while there is still at least as many blocks left as the limit
    const auto number_of_launch = (size + aligned_size_limit - 1) / aligned_size_limit;
    for(size_t i = 0, offset = 0; i < number_of_launch; ++i, offset += aligned_size_limit)
    {
        const auto current_size   = std::min(size - offset, aligned_size_limit);
        const auto current_blocks = (current_size + items_per_block - 1) / items_per_block;

        if(debug_synchronous)
        {
            start = std::chrono::steady_clock::now();
        }

        ROCPRIM_RETURN_ON_ERROR(launch_transform<config, IsPointer, result_type>(target_arch,
                                                                                 input + offset,
                                                                                 output + offset,
                                                                                 current_size,
                                                                                 transform_op,
                                                                                 current_blocks,
                                                                                 block_size,
                                                                                 0,
                                                                                 stream));
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("transform_kernel", current_size, start);
    }

    return hipSuccess;
}

} // namespace detail

/// \brief Parallel transform primitive for device level.
///
/// transform function performs a device-wide transformation operation
/// using unary \p transform_op operator.
///
/// \par Overview
/// * Ranges specified by \p input and \p output must have at least \p size elements.
///
/// \tparam Config [optional] configuration of the primitive. It has to be \p transform_config or a class derived from it.
/// \tparam InputIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam UnaryFunction type of unary function used for transform.
///
/// \param [in] input iterator to the first element in the range to transform.
/// \param [out] output iterator to the first element in the output range.
/// \param [in] size number of element in the input range.
/// \param [in] transform_op unary operation function object that will be used for transform.
/// The signature of the function should be equivalent to the following:
/// <tt>U f(const T &a);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the object passed to it.
/// \param [in] stream [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level transform operation is performed on an array of
/// integer values (<tt>short</tt>s are transformed into <tt>int</tt>s).
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // custom transform function
/// auto transform_op =
///     [] (int a) -> int
///     {
///         return a + 5;
///     };
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;    // e.g., 8
/// short * input;        // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int * output;         // empty array of 8 elements
///
/// // perform transform
/// rocprim::transform(
///     input, output, input_size, transform_op
/// );
/// // output: [6, 7, 8, 9, 10, 11, 12, 13]
/// \endcode
/// \endparblock
template<class Config = default_config,
         class InputIterator,
         class OutputIterator,
         class UnaryFunction>
inline hipError_t transform(InputIterator     input,
                            OutputIterator    output,
                            const size_t      size,
                            UnaryFunction     transform_op,
                            const hipStream_t stream            = 0,
                            bool              debug_synchronous = false)
{
    constexpr bool is_pointer
        = std::is_pointer<InputIterator>::value && std::is_pointer<OutputIterator>::value;

    return detail::transform_impl<is_pointer, Config, InputIterator, OutputIterator, UnaryFunction>(
        input,
        output,
        size,
        transform_op,
        stream,
        debug_synchronous);
}

/// \brief Parallel device-level transform primitive for two inputs.
///
/// transform function performs a device-wide transformation operation
/// on two input ranges using binary \p transform_op operator.
///
/// \par Overview
/// * Ranges specified by \p input1, \p input2, and \p output must have at least \p size elements.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `transform_config`.
/// \tparam InputIterator1 random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam InputIterator2 random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction type of binary function used for transform.
///
/// \param [in] input1 iterator to the first element in the 1st range to transform.
/// \param [in] input2 iterator to the first element in the 2nd range to transform.
/// \param [out] output iterator to the first element in the output range.
/// \param [in] size number of element in the input range.
/// \param [in] transform_op binary operation function object that will be used for transform.
/// The signature of the function should be equivalent to the following:
/// <tt>U f(const T1& a, const T2& b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the object passed to it.
/// \param [in] stream [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced. Default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level transform operation is performed on two arrays of
/// integer values (element-wise sum is performed).
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // custom transform function
/// auto transform_op =
///     [] (int a, int b) -> int
///     {
///         return a + b;
///     };
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t size;   // e.g., 8
/// int* input1;   // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int* input2;   // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int* output;   // empty array of 8 elements
///
/// // perform transform
/// rocprim::transform(
///     input1, input2, output, input1.size(), transform_op
/// );
/// // output: [2, 4, 6, 8, 10, 12, 14, 16]
/// \endcode
/// \endparblock
template<class Config = default_config,
         class InputIterator1,
         class InputIterator2,
         class OutputIterator,
         class BinaryFunction>
inline hipError_t transform(InputIterator1    input1,
                            InputIterator2    input2,
                            OutputIterator    output,
                            const size_t      size,
                            BinaryFunction    transform_op,
                            const hipStream_t stream            = 0,
                            bool              debug_synchronous = false)
{
    using value_type1 = typename std::iterator_traits<InputIterator1>::value_type;
    using value_type2 = typename std::iterator_traits<InputIterator2>::value_type;
    return transform<Config>(
        ::rocprim::make_zip_iterator(::rocprim::make_tuple(input1, input2)),
        output,
        size,
        detail::unpack_binary_op<value_type1, value_type2, BinaryFunction>(transform_op),
        stream,
        debug_synchronous);
}

/// \brief Parallel device-level transform primitive for an arbitrary amount of inputs.
///
/// transform function performs a device-wide transformation operation
/// on n input ranges using binary \p transform_op operator.
///
/// \par Overview
/// * Ranges specified by \p output and all iterators in \p input_iters must have at least \p size elements.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `transform_config`.
/// \tparam InputIterators all the random-access iterator types of the input range. These types must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction type of binary function used for transform.
///
/// \param [in] input_iters a tuple of iterators to the input sequences where num_items elements are read from each.
/// \param [out] output iterator to the first element in the output range.
/// \param [in] size number of element in the input range.
/// \param [in] transform_op an n-ary function object used for the transform, where n is the number of input sequences.
/// The function object must not modify the object passed to it.
/// \param [in] stream [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced. Default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level transform operation is performed on three arrays of
/// integer values (element-wise sum is performed).
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // custom transform function
/// auto transform_op =
///     [] (int a, int b, int c) -> int
///     {
///         return a + b + c;
///     };
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t size;   // e.g., 8
/// int* input1;   // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int* input2;   // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int* input3;   // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int* output;   // empty array of 8 elements
///
/// // perform transform
/// rocprim::transform(
///     rocprim::tuple(input1, input2, input3), output, input1.size(), transform_op
/// );
/// // output: [3, 6, 9, 12, 15, 18, 21, 24]
/// \endcode
/// \endparblock
template<class Config = default_config,
         class... InputIterators,
         class OutputIterator,
         class TransformOp>
inline hipError_t transform(rocprim::tuple<InputIterators...> input_iters,
                            OutputIterator                    output,
                            const size_t                      size,
                            TransformOp                       transform_op,
                            const hipStream_t                 stream            = 0,
                            bool                              debug_synchronous = false)
{
    return transform<Config>(
        ::rocprim::make_zip_iterator(input_iters),
        output,
        size,
        detail::unpack_nary_op<TransformOp,
                               typename std::iterator_traits<InputIterators>::value_type...>(
            transform_op),
        stream,
        debug_synchronous);
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group devicemodule

#endif // ROCPRIM_DEVICE_DEVICE_TRANSFORM_HPP_
