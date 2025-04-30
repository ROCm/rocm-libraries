// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_INTRINSICS_ARCH_HPP_
#define ROCPRIM_INTRINSICS_ARCH_HPP_

#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \brief Utilities to query architecture details.
namespace arch
{

/// \brief Utilities to query wavefront details.
namespace wavefront
{

/// \brief Return the number of threads in the wavefront.
///
/// This function is not `constexpr`.
    ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int size()
{
    // This function is **not** constexpr because it will
    // be using '__builtin_amdgcn_wavefrontsize()'.
    return ROCPRIM_WAVEFRONT_SIZE;
}

/// \brief Return the minimum number of threads in the wavefront.
///
/// This function can be used to setup compile time allocation of
/// global or shared memory.
///
/// \par Example
/// \parblock
/// The example below shows how shared memory can be allocated
/// to collect per warp results.
/// \code{.cpp}
/// constexpr auto total_items = 1024;
/// constexpr auto max_warps   = total_items / arch::min_size();
///
/// // If we want to use shared memory to exchange data
/// // between warps, we can allocate it as:
/// __shared int per_warp_results[max_warps];
/// \endcode
/// \endparblock
    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE
constexpr unsigned int min_size()
{
#if __HIP_DEVICE_COMPILE__
    return ROCPRIM_WAVEFRONT_SIZE;
#else
    return ROCPRIM_WARP_SIZE_32;
#endif
}

/// \brief Return the maximum number of threads in the wavefront.
///
/// This function can be used to setup compile time allocation of
/// global or shared memory.
///
/// \par Example
/// \parblock
/// The example below shows how an array can be allocated
/// to collect a single warp's results.
/// \code{.cpp}
/// constexpr auto items_per_thread = 2;
///
/// // If we want to collect all the elements in a single array
/// // on a single thread, we can allocate it as:
/// int single_warp[items_per_thread * arch::max_size()];
/// \endcode
/// \endparblock
    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE
constexpr unsigned int max_size()
{
#if __HIP_DEVICE_COMPILE__
    return ROCPRIM_WAVEFRONT_SIZE;
#else
    return ROCPRIM_WARP_SIZE_64;
#endif
}
}; // namespace wavefront

} // namespace arch

END_ROCPRIM_NAMESPACE

#endif
