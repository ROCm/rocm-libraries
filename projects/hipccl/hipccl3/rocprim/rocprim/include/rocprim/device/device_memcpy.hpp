// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_MEMCPY_HPP_
#define ROCPRIM_DEVICE_DEVICE_MEMCPY_HPP_

#include "../config.hpp"
#include "../functional.hpp"

#include "config_types.hpp"

#include "detail/device_batch_memcpy.hpp"
#include "device_memcpy_config.hpp"
#include "rocprim/device/detail/device_config_helper.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \brief Copy `sizes[i]` bytes from `sources[i]` to `destinations[i]` for all `i` in the range [0, `num_copies`].
///
/// \tparam Config [optional] configuration of  the primitive. It has to be \p batch_memcpy_config .
/// \tparam InputBufferItType type of iterator to source pointers.
/// \tparam OutputBufferItType type of iterator to desetination pointers.
/// \tparam BufferSizeItType type of iterator to sizes.
///
/// \param [in] temporary_storage pointer to device-accessible temporary storage.
/// When a null pointer is passed, the required allocation size in bytes is written to
/// `storage_size` and the function returns without performing the copy.
/// \param [in, out] storage_size reference to the size in bytes of `temporary_storage`.
/// \param [in] sources iterator of source pointers.
/// \param [in] destinations iterator of destination pointers.
/// \param [in] sizes iterator of range sizes to copy.
/// \param [in] num_copies number of ranges to copy
/// \param [in] stream [optional] HIP stream object to enqueue the copy on. Default is `hipStreamDefault`.
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is `false`.
///
/// Performs multiple device to device memory copies as a single batched operation.
/// Roughly equivalent to
/// \code{.cpp}
/// for (auto i = 0; i < num_copies; ++i) {
///     char* src = sources[i];
///     char* dst = destinations[i];
///     auto size = sizes[i];
///     hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToDevice, stream);
/// }
/// \endcode
/// except executed on the device in parallel.
/// Note that sources and destinations do not have to be part of the same array. I.e. you can copy
/// from both array A and B to array C and D with a single call to this function.
/// Source ranges are allowed to overlap,
/// however, destinations overlapping with either other destinations or with sources is not allowed,
/// and will result in undefined behaviour.
///
/// \par Example
/// \parblock
/// In this example multiple sections of data are copied from \p a to \p b .
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp
///
/// // Device allocated data:
/// int* a;             // e.g, [9, 1, 2, 3, 4, 5, 6, 7, 8]
/// int* b;             // e.g, [0, 0, 0, 0, 0, 0, 0, 0, 0]
///
/// // Batch memcpy parameters:
/// int   num_copies;   // Number of buffers to copy.
///                      // e.g, 4.
/// int** sources;       // Pointer to source pointers.
///                      // e.g, [&a[0], &a[4] &a[7]]
/// int** destinations;  // Pointer to destination pointers.
///                      // e.g, [&b[5], &b[2] &b[0]]
/// int*  sizes;         // Size of buffers to copy.
///                      // e.g., [3 * sizeof(int), 2 * sizeof(int), 2 * sizeof(int)]
///
/// // Calculate the required temporary storage.
/// size_t temporary_storage_size_bytes;
/// void* temporary_storage_ptr = nullptr;
/// rocprim::batch_memcpy(
///     temporary_storage_ptr,
///     temporary_storage_size_bytes,
///     sources,
///     destinations,
///     sizes,
///     num_buffers);
///
/// // Allocate temporary storage.
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // Copy buffers.
/// rocprim::batch_memcpy(
///     temporary_storage_ptr,
///     temporary_storage_size_bytes,
///     sources,
///     destinations,
///     sizes,
///     num_copies);
///
/// // b is now: [7, 8, 4, 5, 0, 9, 1, 2, 0]
/// //   3rd copy ^--^  ^--^     ^--^--^ 1st copy
/// //                2nd copy
/// \endcode
/// \endparblock
template<class Config_ = default_config,
         class InputBufferItType,
         class OutputBufferItType,
         class BufferSizeItType>
ROCPRIM_INLINE static hipError_t batch_memcpy(void*              temporary_storage,
                                              size_t&            storage_size,
                                              InputBufferItType  sources,
                                              OutputBufferItType destinations,
                                              BufferSizeItType   sizes,
                                              uint32_t           num_copies,
                                              hipStream_t        stream = hipStreamDefault,
                                              bool               debug_synchronous = false)
{
    using Config = detail::default_or_custom_config<Config_, batch_memcpy_config<>>;

    static_assert(Config::wlev_size_threshold < Config::blev_size_threshold,
                  "wlev_size_threshold should be smaller than blev_size_threshold");

    using BufferOffsetType = unsigned int;
    using BlockOffsetType  = unsigned int;

    hipError_t error = hipSuccess;

    using batch_memcpy_impl_type = detail::
        batch_memcpy_impl<Config, InputBufferItType, OutputBufferItType, BufferSizeItType>;

    static constexpr uint32_t non_blev_block_size         = Config::non_blev_block_size;
    static constexpr uint32_t non_blev_buffers_per_thread = Config::non_blev_buffers_per_thread;
    static constexpr uint32_t blev_block_size             = Config::blev_block_size;

    constexpr uint32_t buffers_per_block = non_blev_block_size * non_blev_buffers_per_thread;
    const uint32_t     num_blocks = rocprim::detail::ceiling_div(num_copies, buffers_per_block);

    using scan_state_buffer_type = rocprim::detail::lookback_scan_state<BufferOffsetType>;
    using scan_state_block_type  = rocprim::detail::lookback_scan_state<BlockOffsetType>;

    // Pack buffers
    typename batch_memcpy_impl_type::copyable_buffers const buffers{
        sources,
        destinations,
        sizes,
    };

    detail::temp_storage::layout scan_state_buffer_layout{};
    error = scan_state_buffer_type::get_temp_storage_layout(num_blocks,
                                                            stream,
                                                            scan_state_buffer_layout);
    if(error != hipSuccess)
    {
        return error;
    }

    detail::temp_storage::layout blev_block_scan_state_layout{};
    error = scan_state_block_type::get_temp_storage_layout(num_blocks,
                                                           stream,
                                                           blev_block_scan_state_layout);
    if(error != hipSuccess)
    {
        return error;
    }

    uint8_t* blev_buffer_scan_data;
    uint8_t* blev_block_scan_state_data;

    // The non-blev kernel will prepare blev copy. Communication between the two
    // kernels is done via `blev_buffers`.
    typename batch_memcpy_impl_type::copyable_blev_buffers blev_buffers{};

    // Partition `d_temp_storage`.
    // If `d_temp_storage` is null, calculate the allocation size instead.
    error = detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            detail::temp_storage::ptr_aligned_array(&blev_buffers.srcs, num_copies),
            detail::temp_storage::ptr_aligned_array(&blev_buffers.dsts, num_copies),
            detail::temp_storage::ptr_aligned_array(&blev_buffers.sizes, num_copies),
            detail::temp_storage::ptr_aligned_array(&blev_buffers.offsets, num_copies),
            detail::temp_storage::make_partition(&blev_buffer_scan_data, scan_state_buffer_layout),
            detail::temp_storage::make_partition(&blev_block_scan_state_data,
                                                 blev_block_scan_state_layout)));

    // If allocation failed, return error.
    if(error != hipSuccess)
    {
        return error;
    }

    // Return the storage size.
    if(temporary_storage == nullptr)
    {
        return hipSuccess;
    }

    // Compute launch parameters.

    int device_id = hipGetStreamDeviceId(stream);

    // Get the number of multiprocessors
    int multiprocessor_count{};
    error = hipDeviceGetAttribute(&multiprocessor_count,
                                  hipDeviceAttributeMultiprocessorCount,
                                  device_id);
    if(error != hipSuccess)
    {
        return error;
    }

    // `hipOccupancyMaxActiveBlocksPerMultiprocessor` uses the default device.
    // We need to perserve the current default device id while we change it temporarily
    // to get the max occupancy on this stream.
    int previous_device;
    error = hipGetDevice(&previous_device);
    if(error != hipSuccess)
    {
        return error;
    }

    error = hipSetDevice(device_id);
    if(error != hipSuccess)
    {
        return error;
    }

    int blev_occupancy{};
    error = hipOccupancyMaxActiveBlocksPerMultiprocessor(&blev_occupancy,
                                                         batch_memcpy_impl_type::blev_memcpy_kernel,
                                                         blev_block_size,
                                                         0 /* dynSharedMemPerBlk */);
    if(error != hipSuccess)
    {
        return error;
    }

    // Restore the default device id to initial state
    error = hipSetDevice(previous_device);
    if(error != hipSuccess)
    {
        return error;
    }

    constexpr BlockOffsetType init_kernel_threads = 128;
    const BlockOffsetType     init_kernel_grid_size
        = rocprim::detail::ceiling_div(num_blocks, init_kernel_threads);

    auto batch_memcpy_blev_grid_size
        = multiprocessor_count * blev_occupancy * 1 /* subscription factor */;

    BlockOffsetType batch_memcpy_grid_size = num_blocks;

    // Prepare init_scan_states_kernel.
    scan_state_buffer_type scan_state_buffer{};
    error = scan_state_buffer_type::create(scan_state_buffer,
                                           blev_buffer_scan_data,
                                           num_blocks,
                                           stream);
    if(error != hipSuccess)
    {
        return error;
    }

    scan_state_block_type scan_state_block{};
    error = scan_state_block_type::create(scan_state_block,
                                          blev_block_scan_state_data,
                                          num_blocks,
                                          stream);
    if(error != hipSuccess)
    {
        return error;
    }

    // Launch init_scan_states_kernel.
    batch_memcpy_impl_type::
        init_tile_state_kernel<<<init_kernel_grid_size, init_kernel_threads, 0, stream>>>(
            scan_state_buffer,
            scan_state_block,
            num_blocks);
    error = hipGetLastError();
    if(error != hipSuccess)
    {
        return error;
    }
    if(debug_synchronous)
    {
        hipStreamSynchronize(stream);
    }

    // Launch batch_memcpy_non_blev_kernel.
    batch_memcpy_impl_type::
        non_blev_memcpy_kernel<<<batch_memcpy_grid_size, non_blev_block_size, 0, stream>>>(
            buffers,
            num_copies,
            blev_buffers,
            scan_state_buffer,
            scan_state_block);
    error = hipGetLastError();
    if(error != hipSuccess)
    {
        return error;
    }
    if(debug_synchronous)
    {
        hipStreamSynchronize(stream);
    }

    // Launch batch_memcpy_blev_kernel.
    batch_memcpy_impl_type::
        blev_memcpy_kernel<<<batch_memcpy_blev_grid_size, blev_block_size, 0, stream>>>(
            blev_buffers,
            scan_state_buffer,
            batch_memcpy_grid_size - 1);
    error = hipGetLastError();
    if(error != hipSuccess)
    {
        return error;
    }
    if(debug_synchronous)
    {
        hipStreamSynchronize(stream);
    }

    return hipSuccess;
}

END_ROCPRIM_NAMESPACE

#endif
