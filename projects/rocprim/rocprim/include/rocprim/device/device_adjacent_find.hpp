// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_ADJACENT_FIND_HPP_
#define ROCPRIM_DEVICE_DEVICE_ADJACENT_FIND_HPP_

#include "config_types.hpp"
#include "detail/device_adjacent_find.hpp"
#include "detail/device_config_helper.hpp"
#include "device_adjacent_find_config.hpp"
#include "device_reduce.hpp"
#include "device_transform.hpp"

#include "../common.hpp"
#include "../functional.hpp"
#include "../iterator/counting_iterator.hpp"
#include "../iterator/transform_iterator.hpp"
#include "../iterator/zip_iterator.hpp"
#include "../types/tuple.hpp"

#include <cstring>

BEGIN_ROCPRIM_NAMESPACE

#ifndef DOXYGEN_DOCUMENTATION_BUILD // Do not document

namespace detail
{
template<typename Config = default_config,
         typename InputIterator,
         typename OutputIterator,
         typename BinaryPred>
ROCPRIM_INLINE
hipError_t adjacent_find_impl(void* const       temporary_storage,
                              std::size_t&      storage_size,
                              InputIterator     input,
                              OutputIterator    output,
                              const std::size_t size,
                              BinaryPred        op,
                              const hipStream_t stream,
                              const bool        debug_synchronous)
{
    // Data types
    using input_type         = typename std::iterator_traits<InputIterator>::value_type;
    using op_result_type     = bool;
    using index_type         = std::size_t;
    using wrapped_input_type = ::rocprim::tuple<input_type, input_type, index_type>;

    // Operations types
    using reduce_op_type = ::rocprim::minimum<index_type>;

    // Use dynamic tile id
    using ordered_tile_id_type = detail::ordered_block_id<unsigned long long>;

    // Kernel launch config
    using config = wrapped_adjacent_find_config<Config, input_type>;

    // Transform input
    auto wrapped_equal_op = [op, size](const wrapped_input_type& a) -> index_type
    {
        if(op_result_type(op(::rocprim::get<0>(a), ::rocprim::get<1>(a))))
        {
            return ::rocprim::get<2>(a);
        }
        return size;
    };

    // Kernel wrapper
    using counting_it_t  = rocprim::counting_iterator<index_type>;
    using tuple_t        = rocprim::tuple<InputIterator, InputIterator, counting_it_t>;
    using zip_it_t       = rocprim::zip_iterator<tuple_t>;
    using transform_it_t = rocprim::transform_iterator<zip_it_t, decltype(wrapped_equal_op)>;

    using adjacent_find_kernels = adjacent_find_impl_kernels<transform_it_t,
                                                             index_type*,
                                                             reduce_op_type,
                                                             ordered_tile_id_type>;

    // Calculate required temporary storage
    ordered_tile_id_type::id_type* ordered_tile_id_storage;
    index_type*                    reduce_output = nullptr;

    hipError_t result = detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            detail::temp_storage::make_partition(&ordered_tile_id_storage,
                                                 ordered_tile_id_type::get_temp_storage_layout()),
            detail::temp_storage::ptr_aligned_array(&reduce_output, sizeof(*reduce_output))));

    if(result != hipSuccess || temporary_storage == nullptr)
    {
        return result;
    }

    std::chrono::steady_clock::time_point start;
    if(debug_synchronous)
    {
        start = std::chrono::steady_clock::now();
    }

    // Launch adjacent_find_impl_kernels::init_adjacent_find
    auto ordered_tile_id = ordered_tile_id_type::create(ordered_tile_id_storage);
    adjacent_find_kernels::init_adjacent_find<<<1, 1, 0, stream>>>(reduce_output,
                                                                   ordered_tile_id,
                                                                   size);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(
        "rocprim::detail::adjacent_find::init_adjacent_find",
        size,
        start);

    if(size > 1)
    {
        // Wrap adjacent input in zip iterator with idx values
        auto iota = ::rocprim::make_counting_iterator<index_type>(0);
        auto wrapped_input
            = ::rocprim::make_zip_iterator(::rocprim::make_tuple(input, input + 1, iota));

        auto transformed_input
            = ::rocprim::make_transform_iterator(wrapped_input, wrapped_equal_op);

        target_arch target_arch;
        ROCPRIM_RETURN_ON_ERROR(host_target_arch(stream, target_arch));
        const adjacent_find_config_params params = dispatch_target_arch<config, false>(target_arch);
        const unsigned int                block_size       = params.kernel_config.block_size;
        const unsigned int                items_per_thread = params.kernel_config.items_per_thread;
        const unsigned int                items_per_block  = block_size * items_per_thread;
        const unsigned int grid_size        = (size + items_per_block - 1) / items_per_block;
        const unsigned int shared_mem_bytes = 0; /*no dynamic shared mem*/

        auto kernel = [=](auto arch_config)
        {
            adjacent_find_kernels::template block_reduce_kernel<decltype(arch_config)>(
                transformed_input,
                reduce_output,
                size,
                reduce_op_type{},
                ordered_tile_id);
        };

        auto adjacent_find_block_reduce_kernel = make_launch_plan<config>(target_arch, kernel);

        // Get grid size for maximum occupancy, as we may not be able to schedule all the blocks
        // at the same time
        int min_grid_size      = 0;
        int optimal_block_size = 0;
        ROCPRIM_RETURN_ON_ERROR(
            hipOccupancyMaxPotentialBlockSize(&min_grid_size,
                                              &optimal_block_size,
                                              adjacent_find_block_reduce_kernel.kernel,
                                              shared_mem_bytes,
                                              int(block_size)));
        min_grid_size = std::min(static_cast<unsigned int>(min_grid_size), grid_size);

        if(debug_synchronous)
        {
            start = std::chrono::steady_clock::now();
        }

        // Launch adjacent_find_impl_kernels::block_reduce_kernel
        adjacent_find_block_reduce_kernel.launch(min_grid_size,
                                                 block_size,
                                                 shared_mem_bytes,
                                                 stream);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(
            "rocprim::detail::adjacent_find::block_reduce_kernel",
            size,
            start);
    }

    ROCPRIM_RETURN_ON_ERROR(::rocprim::transform(reduce_output,
                                                 output,
                                                 1,
                                                 ::rocprim::identity<void>(),
                                                 stream,
                                                 debug_synchronous));

    return hipSuccess;
}

} // namespace detail

#endif // DOXYGEN_DOCUMENTATION_BUILD

/// \addtogroup devicemodule
/// @{

/// \brief Searches the input sequence for the first appearance of a consecutive pair of equal elements.
///
/// The returned index is either: the index within the input array of the first element of the first
/// pair of consecutive equal elements found or the size of the input array if no such pair is found.
/// Equivalent to the following code
/// \code{.cpp}
/// if(size > 1)
/// {
///     for(std::size_t i = 0; i < size - 1 ; ++i)
///         if (op(input[i], input[i + 1]))
///             return i;
/// }
/// return size;
/// \endcode
///
/// \par Overview
/// * The contents of the inputs are not altered by the function.
/// * Returns the required size of `temporary_storage` in `storage_size` if `temporary_storage` is a null pointer.
/// * Accepts custom \p op.
/// * Streams in graph capture mode are supported.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `adjacent_find_config`.
/// \tparam InputIterator [inferred] Random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator [inferred] Random-access iterator type of the output index. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam BinaryPred [inferred] Boolean binary operation function object that will be applied to
/// consecutive items to check whether they are equal or not. The signature of the function should be equivalent
/// to the following:
/// <tt>bool f(const T& a, const T& b)</tt>. The signature does not need to have
/// <tt>const &</tt>, but the function object must not modify the object passed to it.
/// The operator must meet the C++ named requirement \p BinaryPredicate.
/// The default operation used is <tt>rocprim::equal_to<T></tt>, where \p T is the type of the elements
/// in the input range obtained with <tt>std::iterator_traits<InputIterator>::value_type</tt>>.
///
/// \param [in] temporary_storage Pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// `storage_size` and the function returns without performing any device computation.
/// \param [in,out] storage_size Reference to a size (in bytes) of `temporary_storage`
/// \param [in] input Iterator to the input range.
/// \param [out] output iterator to the output index.
/// \param [in] size Number of items in the input.
/// \param [in] op [optional] The boolean binary operation to be used by the algorithm. Default is
/// \p ::rocprim::equal_to specialized for the type of the input elements.
/// \param [in] stream [optional] HIP stream object. Default is `0` (the default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors and extra debugging info is printed to the
/// standard output. Default value is `false`.
///
/// \return `hipSuccess` (0) after a successful search, otherwise the HIP runtime error of
/// type `hipError_t`.
///
/// \par Example
/// \parblock
/// In this example a device-level adjacent_find operation is performed on integer values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp> //or <rocprim/device/device_adjacent_find.hpp>
///
/// // Custom boolean binary function
/// auto equal_op = [](int a, int b) -> bool { return (a - b == 2); };
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// std::size_t  size;   // e.g., 8
/// int*         input;  // e.g., [8, 7, 5, 4, 3, 2, 1, 0]
/// std::size_t* output; // output index
/// auto         custom_op = equal_op{};
///
/// std::size_t  temporary_storage_size_bytes;
/// void*        temporary_storage_ptr = nullptr;
///
/// // Get required size of the temporary storage
/// rocprim::adjacent_find(
///     temporary_storage_ptr, temporary_storage_size_bytes, input, output, size, custom_op);
///
/// // Allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // Perform adjacent find
/// rocprim::adjacent_find(
///     temporary_storage_ptr, temporary_storage_size_bytes, input, output, size, custom_op);
/// // output: 1
/// \endcode
/// \endparblock
template<typename Config = default_config,
         typename InputIterator,
         typename OutputIterator,
         typename BinaryPred
         = ::rocprim::equal_to<typename std::iterator_traits<InputIterator>::value_type>>
ROCPRIM_INLINE
hipError_t adjacent_find(void* const       temporary_storage,
                         std::size_t&      storage_size,
                         InputIterator     input,
                         OutputIterator    output,
                         const std::size_t size,
                         BinaryPred        op                = BinaryPred{},
                         const hipStream_t stream            = 0,
                         const bool        debug_synchronous = false)
{
    return detail::adjacent_find_impl<Config>(temporary_storage,
                                              storage_size,
                                              input,
                                              output,
                                              size,
                                              op,
                                              stream,
                                              debug_synchronous);
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_ADJACENT_FIND_HPP_
