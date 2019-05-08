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

#ifndef ROCPRIM_DEVICE_DEVICE_PARTITION_HPP_
#define ROCPRIM_DEVICE_DEVICE_PARTITION_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../functional.hpp"
#include "../type_traits.hpp"
#include "../detail/various.hpp"

#include "device_select_config.hpp"
#include "detail/device_partition.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

template<
    select_method SelectMethod,
    bool OnlySelected,
    class Config,
    class InputIterator,
    class FlagIterator,
    class OutputIterator,
    class SelectedCountOutputIterator,
    class UnaryPredicate,
    class InequalityOp,
    class OffsetLookbackScanState
>
__global__
void partition_kernel(InputIterator input,
                      FlagIterator flags,
                      OutputIterator output,
                      SelectedCountOutputIterator selected_count_output,
                      const size_t size,
                      UnaryPredicate predicate,
                      InequalityOp inequality_op,
                      OffsetLookbackScanState offset_scan_state,
                      const unsigned int number_of_blocks,
                      ordered_block_id<unsigned int> ordered_bid)
{
    partition_kernel_impl<SelectMethod, OnlySelected, Config>(
        input, flags, output, selected_count_output, size, predicate,
        inequality_op, offset_scan_state, number_of_blocks, ordered_bid
    );
}

template<class OffsetLookBackScanState>
__global__
void init_offset_scan_state_kernel(OffsetLookBackScanState offset_scan_state,
                                   const unsigned int number_of_blocks,
                                   ordered_block_id<unsigned int> ordered_bid)
{
    init_lookback_scan_state_kernel_impl(
        offset_scan_state, number_of_blocks, ordered_bid
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
    // Method of selection: flag, predicate, unique
    select_method SelectMethod,
     // if true, it doesn't copy rejected values to output
    bool OnlySelected,
    class Config,
    class InputIterator,
    class FlagIterator,
    class OutputIterator,
    class UnaryPredicate,
    class InequalityOp,
    class SelectedCountOutputIterator
>
inline
hipError_t partition_impl(void * temporary_storage,
                          size_t& storage_size,
                          InputIterator input,
                          FlagIterator flags,
                          OutputIterator output,
                          SelectedCountOutputIterator selected_count_output,
                          const size_t size,
                          UnaryPredicate predicate,
                          InequalityOp inequality_op,
                          const hipStream_t stream,
                          bool debug_synchronous)
{
    using offset_type = unsigned int;
    using input_type = typename std::iterator_traits<InputIterator>::value_type;

    // Get default config if Config is default_config
    using config = default_or_custom_config<
        Config,
        default_select_config<ROCPRIM_TARGET_ARCH, input_type>
    >;

    using offset_scan_state_type = detail::lookback_scan_state<offset_type>;
    using ordered_block_id_type = detail::ordered_block_id<unsigned int>;

    constexpr unsigned int block_size = config::block_size;
    constexpr unsigned int items_per_thread = config::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;
    const unsigned int number_of_blocks =
        std::max(1u, static_cast<unsigned int>((size + items_per_block - 1)/items_per_block));

    // Calculate required temporary storage
    size_t offset_scan_state_bytes = ::rocprim::detail::align_size(
        offset_scan_state_type::get_storage_size(number_of_blocks)
    );
    size_t ordered_block_id_bytes = ordered_block_id_type::get_storage_size();
    if(temporary_storage == nullptr)
    {
        // storage_size is never zero
        storage_size = offset_scan_state_bytes + ordered_block_id_bytes;
        return hipSuccess;
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

    // Create and initialize lookback_scan_state obj
    auto offset_scan_state = offset_scan_state_type::create(
        temporary_storage, number_of_blocks
    );
    // Create ad initialize ordered_block_id obj
    auto ptr = reinterpret_cast<char*>(temporary_storage);
    auto ordered_bid = ordered_block_id_type::create(
        reinterpret_cast<ordered_block_id_type::id_type*>(ptr + offset_scan_state_bytes)
    );

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    auto grid_size = (number_of_blocks + block_size - 1)/block_size;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(init_offset_scan_state_kernel<offset_scan_state_type>),
        dim3(grid_size), dim3(block_size), 0, stream,
        offset_scan_state, number_of_blocks, ordered_bid
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init_offset_scan_state_kernel", size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    grid_size = number_of_blocks;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(partition_kernel<
            SelectMethod, OnlySelected, config,
            InputIterator, FlagIterator, OutputIterator, SelectedCountOutputIterator,
            UnaryPredicate, decltype(inequality_op), offset_scan_state_type
        >),
        dim3(grid_size), dim3(block_size), 0, stream,
        input, flags, output, selected_count_output, size, predicate,
        inequality_op, offset_scan_state, number_of_blocks, ordered_bid
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("partition_kernel", size, start)

    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR
#undef ROCPRIM_DETAIL_HIP_SYNC

} // end of detail namespace

/// \brief Parallel select primitive for device level using range of flags.
///
/// Performs a device-wide partition based on input \p flags. Partition copies
/// the values from \p input to \p output in such a way that all values for which the corresponding
/// items from /p flags are \p true (or can be implicitly converted to \p true) precede
/// the elements for which the corresponding items from /p flags are \p false.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p input, \p flags and \p output must have at least \p size elements.
/// * Range specified by \p selected_count_output must have at least 1 element.
/// * Values of \p flag range should be implicitly convertible to `bool` type.
/// * Relative order is preserved for the elements for which the corresponding values from \p flags
/// are \p true. Other elements are copied in reverse order.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p select_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam FlagIterator - random-access iterator type of the flag range. It can be
/// a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam SelectedCountOutputIterator - random-access iterator type of the selected_count_output
/// value. It can be a simple pointer type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the select operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to select values from.
/// \param [in] flags - iterator to the selection flag corresponding to the first element from \p input range.
/// \param [out] output - iterator to the first element in the output range.
/// \param [out] selected_count_output - iterator to the total number of selected values (length of \p output).
/// \param [in] size - number of element in the input range.
/// \param [in] stream - [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level partition operation is performed on an array of
/// integer values with array of <tt>char</tt>s used as flags.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;     // e.g., 8
/// int * input;           // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// char * flags;          // e.g., [0, 1, 1, 0, 0, 1, 0, 1]
/// int * output;          // empty array of 8 elements
/// size_t * output_count; // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::partition(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, flags,
///     output, output_count,
///     input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform partition
/// rocprim::partition(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, flags,
///     output, output_count,
///     input_size
/// );
/// // output: [2, 3, 6, 8, 7, 5, 4, 1]
/// // output_count: 4
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class FlagIterator,
    class OutputIterator,
    class SelectedCountOutputIterator
>
inline
hipError_t partition(void * temporary_storage,
                     size_t& storage_size,
                     InputIterator input,
                     FlagIterator flags,
                     OutputIterator output,
                     SelectedCountOutputIterator selected_count_output,
                     const size_t size,
                     const hipStream_t stream = 0,
                     const bool debug_synchronous = false)
{
    // Dummy unary predicate
    using unary_predicate_type = ::rocprim::empty_type;
    // Dummy inequality operation
    using inequality_op_type = ::rocprim::empty_type;

    return detail::partition_impl<detail::select_method::flag, false, Config>(
        temporary_storage, storage_size, input, flags, output, selected_count_output,
        size, unary_predicate_type(), inequality_op_type(), stream, debug_synchronous
    );
}

/// \brief Parallel select primitive for device level using selection predicate.
///
/// Performs a device-wide partition using selection predicate. Partition copies
/// the values from \p input to \p output  in such a way that all values for which
/// the \p predicate returns \p true precede the elements for which it returns \p false.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p input, \p flags and \p output must have at least \p size elements.
/// * Range specified by \p selected_count_output must have at least 1 element.
/// * Relative order is preserved for the elements for which the \p predicate returns \p true. Other
/// elements are copied in reverse order.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p select_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam SelectedCountOutputIterator - random-access iterator type of the selected_count_output
/// value. It can be a simple pointer type.
/// \tparam UnaryPredicate - type of a unary selection predicate.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the select operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to select values from.
/// \param [out] output - iterator to the first element in the output range.
/// \param [out] selected_count_output - iterator to the total number of selected values (length of \p output).
/// \param [in] size - number of element in the input range.
/// \param [in] predicate - unary function object which returns /p true if the element should be
/// ordered before other elements.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the object passed to it.
/// \param [in] stream - [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level partition operation is performed on an array of
/// integer values, even values are copied before odd values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>///
///
/// auto predicate =
///     [] __device__ (int a) -> bool
///     {
///         return (a%2) == 0;
///     };
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;     // e.g., 8
/// int * input;           // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int * output;          // empty array of 8 elements
/// size_t * output_count; // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::partition(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input,
///     output, output_count,
///     input_size,
///     predicate
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform partition
/// rocprim::partition(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input,
///     output, output_count,
///     input_size,
///     predicate
/// );
/// // output: [2, 4, 6, 8, 7, 5, 3, 1]
/// // output_count: 4
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class SelectedCountOutputIterator,
    class UnaryPredicate
>
inline
hipError_t partition(void * temporary_storage,
                     size_t& storage_size,
                     InputIterator input,
                     OutputIterator output,
                     SelectedCountOutputIterator selected_count_output,
                     const size_t size,
                     UnaryPredicate predicate,
                     const hipStream_t stream = 0,
                     const bool debug_synchronous = false)
{
    // Dummy flag type
    using flag_type = ::rocprim::empty_type;
    flag_type * flags = nullptr;
    // Dummy inequality operation
    using inequality_op_type = ::rocprim::empty_type;

    return detail::partition_impl<detail::select_method::predicate, false, Config>(
        temporary_storage, storage_size, input, flags, output, selected_count_output,
        size, predicate, inequality_op_type(), stream, debug_synchronous
    );
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_PARTITION_HPP_
