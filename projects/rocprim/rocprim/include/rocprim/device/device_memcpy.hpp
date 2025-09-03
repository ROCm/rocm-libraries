// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
/// \tparam Config [optional] Configuration of  the primitive, must be `default_config` or `batch_memcpy_config`.
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
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
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
template<class Config = default_config,
         class InputBufferItType,
         class OutputBufferItType,
         class BufferSizeItType>
ROCPRIM_INLINE
static hipError_t batch_memcpy(void*              temporary_storage,
                               size_t&            storage_size,
                               InputBufferItType  sources,
                               OutputBufferItType destinations,
                               BufferSizeItType   sizes,
                               uint32_t           num_copies,
                               hipStream_t        stream            = hipStreamDefault,
                               bool               debug_synchronous = false)
{
    return detail::
        batch_memcpy_func<Config, InputBufferItType, OutputBufferItType, BufferSizeItType, true>(
            temporary_storage,
            storage_size,
            sources,
            destinations,
            sizes,
            num_copies,
            stream,
            debug_synchronous);
}

END_ROCPRIM_NAMESPACE

#endif
