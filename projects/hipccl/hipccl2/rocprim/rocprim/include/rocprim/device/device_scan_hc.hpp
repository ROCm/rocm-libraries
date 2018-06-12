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

#ifndef ROCPRIM_DEVICE_DEVICE_SCAN_HC_HPP_
#define ROCPRIM_DEVICE_DEVICE_SCAN_HC_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../functional.hpp"
#include "../type_traits.hpp"
#include "../detail/various.hpp"
#include "../detail/match_result_type.hpp"

#include "device_scan_config.hpp"
#include "detail/device_scan_reduce_then_scan.hpp"
#include "detail/device_scan_lookback.hpp"

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
    bool Exclusive,
    class Config,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction
>
inline
void scan_impl(void * temporary_storage,
               size_t& storage_size,
               InputIterator input,
               OutputIterator output,
               const InitValueType initial_value,
               const size_t size,
               BinaryFunction scan_op,
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
        default_scan_config<ROCPRIM_TARGET_ARCH, result_type>
    >;

    constexpr unsigned int block_size = config::block_size;
    constexpr unsigned int items_per_thread = config::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;

    // Calculate required temporary storage
    if(temporary_storage == nullptr)
    {
        storage_size = scan_get_temporary_storage_bytes<result_type>(size, items_per_block);
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

        // Grid size for block_reduce_kernel, we don't need to calculate reduction
        // of the last block as it will never be used as prefix for other blocks
        auto grid_size = (number_of_blocks - 1) * block_size;
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<1>(grid_size, block_size),
            [=](hc::tiled_index<1>) [[hc]]
            {
                block_reduce_kernel_impl<config>(
                    input, scan_op, block_prefixes
                );
            }
        );
        ROCPRIM_DETAIL_HC_SYNC("block_reduce_kernel", size, start)

        // TODO: Performance may increase if for (number_of_blocks < 8192) (or some other
        // threshold) we would just use CPU to calculate prefixes.

        // Calculate size of temporary storage for nested device scan operation
        void * nested_temp_storage = static_cast<void*>(block_prefixes + number_of_blocks);
        auto nested_temp_storage_size = storage_size - (number_of_blocks * sizeof(result_type));

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        scan_impl<false, config>(
            nested_temp_storage,
            nested_temp_storage_size,
            block_prefixes, // input
            block_prefixes, // output
            result_type(), // dummy initial value
            number_of_blocks, // size
            scan_op,
            acc_view,
            debug_synchronous
        );
        ROCPRIM_DETAIL_HC_SYNC("nested_device_scan", number_of_blocks, start)

        // Grid size for final_scan_kernel
        grid_size = number_of_blocks * block_size;
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<1>(grid_size, block_size),
            [=](hc::tiled_index<1>) [[hc]]
            {
                final_scan_kernel_impl<Exclusive, config>(
                    input, size, output, static_cast<result_type>(initial_value),
                    scan_op, block_prefixes
                );
            }
        );
        ROCPRIM_DETAIL_HC_SYNC("final_scan_kernel", size, start)
    }
    else
    {
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<1>(block_size, block_size),
            [=](hc::tiled_index<1>) [[hc]]
            {
                single_scan_kernel_impl<Exclusive, config>(
                    input, size, static_cast<result_type>(initial_value), output, scan_op
                );
            }
        );
        ROCPRIM_DETAIL_HC_SYNC("single_scan_kernel", size, start);
    }
}

template<
    bool Exclusive,
    class Config,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction
>
inline
void lookback_scan_impl(void * temporary_storage,
                        size_t& storage_size,
                        InputIterator input,
                        OutputIterator output,
                        const InitValueType initial_value,
                        const size_t size,
                        BinaryFunction scan_op,
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
        default_scan_config<ROCPRIM_TARGET_ARCH, result_type>
    >;

    using scan_state_type = detail::lookback_scan_state<result_type>;
    using ordered_block_id_type = detail::ordered_block_id<unsigned int>;

    constexpr unsigned int block_size = config::block_size;
    constexpr unsigned int items_per_thread = config::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;
    const unsigned int number_of_blocks = (size + items_per_block - 1)/items_per_block;

    // Calculate required temporary storage
    size_t scan_state_bytes = ::rocprim::detail::align_size(
        scan_state_type::get_storage_size(number_of_blocks)
    );
    size_t ordered_block_id_bytes = ordered_block_id_type::get_storage_size();
    if(temporary_storage == nullptr)
    {
        // storage_size is never zero
        storage_size = scan_state_bytes + ordered_block_id_bytes;
        return;
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;
    if(debug_synchronous)
    {
        std::cout << "size " << size << '\n';
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
    }

    if(number_of_blocks > 1)
    {
        // Create and initialize lookback_scan_state obj
        auto scan_state = scan_state_type::create(temporary_storage, number_of_blocks);
        // Create ad initialize ordered_block_id obj
        auto ptr = reinterpret_cast<char*>(temporary_storage);
        auto ordered_bid = ordered_block_id_type::create(
            reinterpret_cast<ordered_block_id_type::id_type*>(ptr + scan_state_bytes)
        );

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        auto grid_size = block_size * ((number_of_blocks + block_size - 1) / block_size);
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<1>(grid_size, block_size),
            [=](hc::tiled_index<1>) [[hc]]
            {
                init_lookback_scan_state_kernel_impl(
                    scan_state, number_of_blocks, ordered_bid
                );
            }
        );
        ROCPRIM_DETAIL_HC_SYNC("init_lookback_scan_state_kernel", size, start)

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        grid_size = block_size * number_of_blocks;
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<1>(grid_size, block_size),
            [=](hc::tiled_index<1>) [[hc]]
            {
                lookback_scan_kernel_impl<Exclusive, config>(
                    input, output, size, static_cast<result_type>(initial_value),
                    scan_op, scan_state, number_of_blocks, ordered_bid
                );
            }
        );
        ROCPRIM_DETAIL_HC_SYNC("lookback_scan_kernel", size, start)
    }
    else
    {
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<1>(block_size, block_size),
            [=](hc::tiled_index<1>) [[hc]]
            {
                single_scan_kernel_impl<Exclusive, config>(
                    input, size, static_cast<result_type>(initial_value), output, scan_op
                );
            }
        );
        ROCPRIM_DETAIL_HC_SYNC("single_scan_kernel", size, start);
    }
}

#undef ROCPRIM_DETAIL_HC_SYNC

} // end of detail namespace

/// \brief HC parallel inclusive scan primitive for device level.
///
/// inclusive_scan function performs a device-wide inclusive prefix scan operation
/// using binary \p scan_op operator.
///
/// \par Overview
/// * Supports non-commutative scan operators. However, a scan operator should be
/// associative. When used with non-associative functions the results may be non-deterministic
/// and/or vary in precision.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p input and \p output must have at least \p size elements.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p scan_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction - type of binary function used for scan. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the scan operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to scan.
/// \param [out] output - iterator to the first element in the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] scan_op - binary operation function object that will be used for scan.
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
/// In this example a device-level inclusive sum operation is performed on an array of
/// integer values (<tt>short</tt>s are scanned into <tt>int</tt>s).
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// hc::accelerator_view acc_view = ...;
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t size;                                      // e.g., 8
/// hc::array<short> input(hc::extent<1>(size), ...); // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// hc::array<int> output(input.get_extent(), ...);   // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// // Get required size of the temporary storage
/// rocprim::inclusive_scan(
///     nullptr, temporary_storage_size_bytes,
///     input.accelerator_pointer(), output.accelerator_pointer(), size,
///     rocprim::plus<int>(), acc_view, false
/// );
///
/// // allocate temporary storage
/// hc::array<char> temporary_storage(temporary_storage_size_bytes, acc_view);
///
/// // perform scan
/// rocprim::inclusive_scan(
///     temporary_storage.accelerator_pointer(), temporary_storage_size_bytes,
///     input.accelerator_pointer(), output.accelerator_pointer(), size,
///     rocprim::plus<int>(), acc_view, false
/// );
/// // output: [1, 3, 6, 10, 15, 21, 28, 36]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
void inclusive_scan(void * temporary_storage,
                    size_t& storage_size,
                    InputIterator input,
                    OutputIterator output,
                    const size_t size,
                    BinaryFunction scan_op = BinaryFunction(),
                    hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                    const bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using result_type = typename ::rocprim::detail::match_result_type<
        input_type, output_type, BinaryFunction
    >::type;

    // Lookback scan has problems with types that are not arithmetic
    if(::rocprim::is_arithmetic<result_type>::value)
    {
        return detail::lookback_scan_impl<false, Config>(
            temporary_storage, storage_size,
            // result_type() is a dummy initial value (not used)
            input, output, result_type(), size,
            scan_op, acc_view, debug_synchronous
        );
    }
    else
    {
        return detail::scan_impl<false, Config>(
            temporary_storage, storage_size,
            // result_type() is a dummy initial value (not used)
            input, output, result_type(), size,
            scan_op, acc_view, debug_synchronous
        );
    }
}

/// \brief HC parallel exclusive scan primitive for device level.
///
/// exclusive_scan function performs a device-wide exclusive prefix scan operation
/// using binary \p scan_op operator.
///
/// \par Overview
/// * Supports non-commutative scan operators. However, a scan operator should be
/// associative. When used with non-associative functions the results may be non-deterministic
/// and/or vary in precision.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p input and \p output must have at least \p size elements.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p scan_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam InitValueType - type of the initial value.
/// \tparam BinaryFunction - type of binary function used for scan. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the scan operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to scan.
/// \param [out] output - iterator to the first element in the output range.
/// \param [in] initial_value - initial value to start the scan.
/// \param [in] size - number of element in the input range.
/// \param [in] scan_op - binary operation function object that will be used for scan.
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
/// In this example a device-level inclusive min-scan operation is performed on an array of
/// integer values (<tt>short</tt>s are scanned into <tt>int</tt>s) using custom operator.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // custom scan function
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
/// hc::array<int> output(input.get_extent(), ...);   // empty array of 8 elements
/// int start_value;                                  // e.g., 9
///
/// size_t temporary_storage_size_bytes;
/// // Get required size of the temporary storage
/// rocprim::inclusive_scan(
///     nullptr, temporary_storage_size_bytes,
///     input.accelerator_pointer(), output.accelerator_pointer(), start_value,
///     size, min_op, acc_view, false
/// );
///
/// // allocate temporary storage
/// hc::array<char> temporary_storage(temporary_storage_size_bytes, acc_view);
///
/// // perform scan
/// rocprim::inclusive_scan(
///     temporary_storage.accelerator_pointer(), temporary_storage_size_bytes,
///     input.accelerator_pointer(), output.accelerator_pointer(), start_value,
///     size, min_op, acc_view, false
/// );
/// // output: [9, 4, 7, 6, 2, 2, 1, 1]
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
void exclusive_scan(void * temporary_storage,
                    size_t& storage_size,
                    InputIterator input,
                    OutputIterator output,
                    const InitValueType initial_value,
                    const size_t size,
                    BinaryFunction scan_op = BinaryFunction(),
                    hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                    const bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using result_type = typename ::rocprim::detail::match_result_type<
        input_type, output_type, BinaryFunction
    >::type;

    // Lookback scan has problems with types that are not arithmetic
    if(::rocprim::is_arithmetic<result_type>::value)
    {
        return detail::lookback_scan_impl<true, Config>(
            temporary_storage, storage_size,
            input, output, initial_value, size,
            scan_op, acc_view, debug_synchronous
        );
    }
    else
    {
        return detail::scan_impl<true, Config>(
            temporary_storage, storage_size,
            input, output, initial_value, size,
            scan_op, acc_view, debug_synchronous
        );
    }
}

/// @}
// end of group devicemodule_hc

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SCAN_HC_HPP_
