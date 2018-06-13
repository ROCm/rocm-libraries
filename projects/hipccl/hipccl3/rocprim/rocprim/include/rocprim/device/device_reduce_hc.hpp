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

#ifndef ROCPRIM_DEVICE_DEVICE_REDUCE_HC_HPP_
#define ROCPRIM_DEVICE_DEVICE_REDUCE_HC_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../detail/match_result_type.hpp"

#include "device_reduce_config.hpp"
#include "detail/device_reduce.hpp"

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

template<
    bool WithInitialValue, // true when inital_value should be used in reduction
    class Config,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction
>
inline
void reduce_impl(void * temporary_storage,
                 size_t& storage_size,
                 InputIterator input,
                 OutputIterator output,
                 const InitValueType initial_value,
                 const size_t size,
                 BinaryFunction reduce_op,
                 hc::accelerator_view acc_view,
                 const bool debug_synchronous)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using result_type = typename ::rocprim::detail::match_result_type<
        input_type, output_type, BinaryFunction
    >::type;

    // Get default config if Config is default_config
    using config = default_or_custom_config<
        Config,
        default_reduce_config<ROCPRIM_TARGET_ARCH, result_type>
    >;

    constexpr unsigned int block_size = config::block_size;
    constexpr unsigned int items_per_thread = config::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;

    // Calculate required temporary storage
    if(temporary_storage == nullptr)
    {
        storage_size = reduce_get_temporary_storage_bytes<result_type>(size, items_per_block);
        // Make sure user won't try to allocate 0 bytes memory, otherwise
        // user may again pass nullptr as temporary_storage
        storage_size = storage_size == 0 ? 4 : storage_size;
        return;
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    auto number_of_blocks = (size + items_per_block - 1)/items_per_block;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
        std::cout << "temporary storage size " << storage_size << '\n';
    }

    if(number_of_blocks > 1)
    {
        // Pointer to array with block_prefixes
        result_type * block_prefixes = static_cast<result_type*>(temporary_storage);

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<1>(number_of_blocks * block_size, block_size),
            [=](hc::tiled_index<1>) [[hc]]
            {
                block_reduce_kernel_impl<false, config, result_type>(
                    input, size, block_prefixes, initial_value, reduce_op
                );
            }
        );
        ROCPRIM_DETAIL_HC_SYNC("block_reduce_kernel", size, start)

        void * nested_temp_storage = static_cast<void*>(block_prefixes + number_of_blocks);
        auto nested_temp_storage_size = storage_size - (number_of_blocks * sizeof(result_type));

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        reduce_impl<WithInitialValue, config>(
            nested_temp_storage,
            nested_temp_storage_size,
            block_prefixes, // input
            output, // output
            initial_value,
            number_of_blocks, // input size
            reduce_op,
            acc_view,
            debug_synchronous
        );
        ROCPRIM_DETAIL_HC_SYNC("nested_device_reduce", number_of_blocks, start);
    }
    else
    {
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<1>(block_size, block_size),
            [=](hc::tiled_index<1>) [[hc]]
            {
                block_reduce_kernel_impl<WithInitialValue, config, result_type>(
                    input, size, output, initial_value, reduce_op
                );
            }
        );
        ROCPRIM_DETAIL_HC_SYNC("block_reduce_kernel", size, start);
    }
}

#undef ROCPRIM_DETAIL_HC_SYNC

} // end of detail namespace

/// \brief HC parallel reduction primitive for device level.
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
/// \param [out] output - iterator to the first element in the output range.
/// \param [in] initial_value - initial value to start the reduction.
/// \param [in] size - number of element in the input range.
/// \param [in] reduce_op - binary operation function object that will be used for reduction.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// The default value is \p BinaryFunction().
/// \param [in] acc_view - [optional] \p hc::accelerator_view object. The default value
/// is \p hc::accelerator().get_default_view() (default view of the default accelerator).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced. The default value is \p false.
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
///     [](int a, int b) [[hc]]
///     {
///         return a < b ? a : b;
///     };
///
/// hc::accelerator_view acc_view = ...;
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t size;                                      // e.g., 8
/// hc::array<short> input(hc::extent<1>(size), ...); // e.g., [4, 7, 6, 2, 5, 1, 3, 8]
/// hc::array<int> output(1, ...);                    // empty array of 1 element
/// int start_value;                                  // e.g., 9
///
/// size_t temporary_storage_size_bytes;
/// // Get required size of the temporary storage
/// rocprim::reduce(
///     nullptr, temporary_storage_size_bytes,
///     input.accelerator_pointer(), output.accelerator_pointer(), start_value,
///     size, min_op, acc_view, false
/// );
///
/// // allocate temporary storage
/// hc::array<char> temporary_storage(temporary_storage_size_bytes, acc_view);
///
/// // perform reduce
/// rocprim::reduce(
///     temporary_storage.accelerator_pointer(), temporary_storage_size_bytes,
///     input.accelerator_pointer(), output.accelerator_pointer(), start_value,
///     size, min_op, acc_view, false
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
void reduce(void * temporary_storage,
            size_t& storage_size,
            InputIterator input,
            OutputIterator output,
            const InitValueType initial_value,
            const size_t size,
            BinaryFunction reduce_op = BinaryFunction(),
            hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
            bool debug_synchronous = false)
{
    return detail::reduce_impl<true, Config>(
        temporary_storage, storage_size,
        input, output, initial_value, size,
        reduce_op, acc_view, debug_synchronous
    );
}

/// \brief HC parallel reduction primitive for device level.
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
/// \param [out] output - iterator to the first element in the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] reduce_op - binary operation function object that will be used for reduction.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is BinaryFunction().
/// \param [in] acc_view - [optional] \p hc::accelerator_view object. The default value
/// is \p hc::accelerator().get_default_view() (default view of the default accelerator).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced. Default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level sum operation is performed on an array of
/// integer values (<tt>short</tt>s are reduced into <tt>int</tt>s).
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// hc::accelerator_view acc_view = ...;
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t size;                                      // e.g., 8
/// hc::array<short> input(hc::extent<1>(size), ...); // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// hc::array<int> output(1, ...);                    // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// // Get required size of the temporary storage
/// rocprim::reduce(
///     nullptr, temporary_storage_size_bytes,
///     input.accelerator_pointer(), output.accelerator_pointer(), size,
///     rocprim::plus<int>(), acc_view, false
/// );
///
/// // allocate temporary storage
/// hc::array<char> temporary_storage(temporary_storage_size_bytes, acc_view);
///
/// // perform reduce
/// rocprim::reduce(
///     temporary_storage.accelerator_pointer(), temporary_storage_size_bytes,
///     input.accelerator_pointer(), output.accelerator_pointer(), size,
///     rocprim::plus<int>(), acc_view, false
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
void reduce(void * temporary_storage,
            size_t& storage_size,
            InputIterator input,
            OutputIterator output,
            const size_t size,
            BinaryFunction reduce_op = BinaryFunction(),
            hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
            bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;

    return detail::reduce_impl<false, Config>(
        temporary_storage, storage_size,
        input, output, input_type(), size,
        reduce_op, acc_view, debug_synchronous
    );
}

/// @}
// end of group devicemodule_hc

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_REDUCE_HC_HPP_
