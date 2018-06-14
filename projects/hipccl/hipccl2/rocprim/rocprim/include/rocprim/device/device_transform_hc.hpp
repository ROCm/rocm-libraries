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

#ifndef ROCPRIM_DEVICE_DEVICE_TRANSFORM_HC_HPP_
#define ROCPRIM_DEVICE_DEVICE_TRANSFORM_HC_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../types/tuple.hpp"
#include "../iterator/zip_iterator.hpp"

#include "device_transform_config.hpp"
#include "detail/device_transform.hpp"

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

} // end of detail namespace

/// \brief HC parallel transform primitive for device level.
///
/// transform function performs a device-wide transformation operation
/// using unary \p transform_op operator.
///
/// \par Overview
/// * Ranges specified by \p input and \p output must have at least \p size elements.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p transform_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam UnaryFunction - type of unary function used for transform.
///
/// \param [in] input - iterator to the first element in the range to transform.
/// \param [out] output - iterator to the first element in the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] transform_op - unary operation function object that will be used for transform.
/// The signature of the function should be equivalent to the following:
/// <tt>U f(const T &a);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the object passed to it.
/// \param [in] acc_view - [optional] \p hc::accelerator_view object. The default value
/// is \p hc::accelerator().get_default_view() (default view of the default accelerator).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced. Default value is \p false.
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
///     [](int a) [[hc]]
///     {
///         return a + 5;
///     };
///
/// hc::accelerator_view acc_view = ...;
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t size;                                      // e.g., 8
/// hc::array<short> input(hc::extent<1>(size), ...); // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// hc::array<int> output(input.get_extent(), ...);   // empty array of 8 elements
///
/// // perform transform
/// rocprim::transform(
///     input.accelerator_pointer(), output.accelerator_pointer(), size,
///     transform_op, acc_view, false
/// );
/// // output: [6, 7, 8, 9, 10, 11, 12, 13]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class UnaryFunction
>
inline
void transform(InputIterator input,
               OutputIterator output,
               const size_t size,
               UnaryFunction transform_op,
               hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
               bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    #ifdef __cpp_lib_is_invocable
    using result_type = typename std::invoke_result<UnaryFunction, input_type>::type;
    #else
    using result_type = typename std::result_of<UnaryFunction(input_type)>::type;
    #endif

    // Get default config if Config is default_config
    using config = detail::default_or_custom_config<
        Config,
        detail::default_transform_config<ROCPRIM_TARGET_ARCH, result_type>
    >;

    constexpr unsigned int block_size = config::block_size;
    constexpr unsigned int items_per_thread = config::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    auto number_of_blocks = (size + items_per_block - 1)/items_per_block;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
    }

    auto grid_size = number_of_blocks * block_size;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hc::parallel_for_each(
        acc_view,
        hc::tiled_extent<1>(grid_size, block_size),
        [=](hc::tiled_index<1>) [[hc]]
        {
            detail::transform_kernel_impl<block_size, items_per_thread, result_type>(
                input, size, output, transform_op
            );
        }
    );
    ROCPRIM_DETAIL_HC_SYNC("transform_kernel", size, start)
}

/// \brief HC parallel device-level transform primitive for two inputs.
///
/// transform function performs a device-wide transformation operation
/// on two input ranges using binary \p transform_op operator.
///
/// \par Overview
/// * Ranges specified by \p input1, \p input2, and \p output must have at least \p size elements.
///
/// \tparam InputIterator1 - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam InputIterator2 - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction - type of binary function used for transform.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p transform_config or
/// a custom class with the same members.
/// \param [in] input1 - iterator to the first element in the 1st range to transform.
/// \param [in] input2 - iterator to the first element in the 2nd range to transform.
/// \param [out] output - iterator to the first element in the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] transform_op - binary operation function object that will be used for transform.
/// The signature of the function should be equivalent to the following:
/// <tt>U f(const T1& a, const T2& b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the object passed to it.
/// \param [in] acc_view - [optional] \p hc::accelerator_view object. The default value
/// is \p hc::accelerator().get_default_view() (default view of the default accelerator).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
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
///     [](int a, int b) [[hc]]
///     {
///         return a + b;
///     };
///
/// hc::accelerator_view acc_view = ...;
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t size;                                      // e.g., 8
/// hc::array<int> input1(hc::extent<1>(size), ...); // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// hc::array<int> input2(hc::extent<1>(size), ...); // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// hc::array<int> output(input.get_extent(), ...);  // empty array of 8 elements
///
/// // perform transform
/// rocprim::transform(
///     input1.accelerator_pointer(), input2.accelerator_pointer(),
///     output.accelerator_pointer(), input1.size(),
///     transform_op, acc_view, false
/// );
/// // output: [2, 4, 6, 8, 10, 12, 14, 16]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator1,
    class InputIterator2,
    class OutputIterator,
    class BinaryFunction
>
inline
void transform(InputIterator1 input1,
               InputIterator2 input2,
               OutputIterator output,
               const size_t size,
               BinaryFunction transform_op,
               hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
               bool debug_synchronous = false)
{
    using value_type1 = typename std::iterator_traits<InputIterator1>::value_type;
    using value_type2 = typename std::iterator_traits<InputIterator2>::value_type;
    transform<Config>(
        ::rocprim::make_zip_iterator(::rocprim::make_tuple(input1, input2)), output,
        size, detail::unpack_binary_op<value_type1, value_type2, BinaryFunction>(transform_op),
        acc_view, debug_synchronous
    );
}

#undef ROCPRIM_DETAIL_HC_SYNC

/// @}
// end of group devicemodule_hc

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_TRANSFORM_HC_HPP_
