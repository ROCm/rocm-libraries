/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights meserved.
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/nv_target.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/hip/detail/cdp_dispatch.h>
#include <thrust/system/hip/detail/execution_policy.h>

#if !THRUST_COMPILER(NVRTC)
#  include <thrust/system/hip/error.h>
#  include <thrust/system_error.h>

#  include <cstdio>
#endif // !THRUST_COMPILER(NVRTC)

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <iterator>
#endif

// Define the value to 0, if you want to disable printf on device side.
#ifndef THRUST_HIP_PRINTF_ENABLED
#  define THRUST_HIP_PRINTF_ENABLED 1
#endif

#if THRUST_HIP_PRINTF_ENABLED == 1
#  define THRUST_HIP_PRINTF(text, ...) printf(text, ##__VA_ARGS__)
#else
#  define THRUST_HIP_PRINTF(text, ...)
#endif

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

inline THRUST_HOST_DEVICE hipStream_t default_stream()
{
#ifdef HIP_API_PER_THREAD_DEFAULT_STREAM
  return hipStreamPerThread;
#else
  return hipStreamDefault; // There's not hipStreamLegacy
#endif
}

// Fallback implementation of the customization point.
template <class Derived>
THRUST_HOST_DEVICE hipStream_t get_stream(execution_policy<Derived>&)
{
  return default_stream();
}

// Entry point/interface.
template <class Derived>
THRUST_HOST_DEVICE hipStream_t stream(execution_policy<Derived>& policy)
{
  return get_stream(derived_cast(policy));
}

// Fallback implementation of the customization point.
template <class Derived>
THRUST_HOST_DEVICE bool must_perform_optional_stream_synchronization(execution_policy<Derived>&)
{
  return true;
}

// Entry point/interface.
template <class Derived>
THRUST_HOST_DEVICE bool must_perform_optional_synchronization(execution_policy<Derived>& policy)
{
  return must_perform_optional_stream_synchronization(derived_cast(policy));
}

template <class Derived>
THRUST_HOST_DEVICE integral_constant<bool, true> allows_nondeterminism(execution_policy<Derived>&)
{
  return {};
}

template <class Derived>
THRUST_HOST_DEVICE auto nondeterministic(execution_policy<Derived>& policy)
  -> decltype(allows_nondeterminism(derived_cast(policy)))
{
  return {};
}

// Fallback implementation of the customization point.
THRUST_EXEC_CHECK_DISABLE
template <class Derived>
THRUST_HOST_DEVICE hipError_t synchronize_stream(execution_policy<Derived>& policy)
{
  hipError_t result;

  THRUST_CDP_DISPATCH((result = hipStreamSynchronize(stream(policy));), ((void) policy; result = hipSuccess;));

  return result;
}

// Entry point/interface.
template <class Policy>
THRUST_HOST_DEVICE hipError_t synchronize(Policy& policy)
{
  THRUST_CDP_DISPATCH((return synchronize_stream(derived_cast(policy));), ((void) policy; return hipSuccess;));
}

// Fallback implementation of the customization point.
THRUST_EXEC_CHECK_DISABLE
template <class Derived>
THRUST_HOST_DEVICE hipError_t synchronize_stream_optional(execution_policy<Derived>& policy)
{
  hipError_t result;

  if (must_perform_optional_synchronization(policy))
  {
    result = synchronize_stream(policy);
  }
  else
  {
    result = hipSuccess;
  }

  return result;
}

// Entry point/interface.
template <class Policy>
THRUST_HOST_DEVICE hipError_t synchronize_optional(Policy& policy)
{
  return synchronize_stream_optional(derived_cast(policy));
}

#if !THRUST_COMPILER(NVRTC)
template <class Type>
THRUST_HIP_HOST_FUNCTION hipError_t
trivial_copy_from_device(Type* dst, Type const* src, size_t count, hipStream_t stream)
{
  hipError_t status = hipSuccess;
  if (count == 0)
  {
    return status;
  }

// hipMemcpyWithStream is only supported on rocm 3.1 and above
#if HIP_VERSION_MAJOR >= 3
#  if HIP_VERSION_MINOR >= 1 || HIP_VERSION_MAJOR >= 4
  status = ::hipMemcpyWithStream(dst, src, sizeof(Type) * count, hipMemcpyDeviceToHost, stream);
#  else
  status = ::hipMemcpyAsync(dst, src, sizeof(Type) * count, hipMemcpyDeviceToHost, stream);
  if (status != hipSuccess)
  {
    return status;
  }
  status = hipStreamSynchronize(stream);
#  endif
#endif
  return status;
}

template <class Type>
THRUST_HIP_HOST_FUNCTION hipError_t trivial_copy_to_device(Type* dst, Type const* src, size_t count, hipStream_t stream)
{
  hipError_t status = hipSuccess;
  if (count == 0)
  {
    return status;
  }

// hipMemcpyWithStream is only supported on rocm 3.1 and above
#if HIP_VERSION_MAJOR >= 3
#  if HIP_VERSION_MINOR >= 1 || HIP_VERSION_MAJOR >= 4
  status = ::hipMemcpyWithStream(dst, src, sizeof(Type) * count, hipMemcpyHostToDevice, stream);
#  else
  status = ::hipMemcpyAsync(dst, src, sizeof(Type) * count, hipMemcpyHostToDevice, stream);
  if (status != hipSuccess)
  {
    return status;
  }
  status = hipStreamSynchronize(stream);
#  endif
#endif
  return status;
}

template <class Policy, class Type>
THRUST_HIP_RUNTIME_FUNCTION hipError_t
trivial_copy_device_to_device(Policy& policy, Type* dst, Type const* src, size_t count)
{
  hipError_t status = hipSuccess;
  if (count == 0)
  {
    return status;
  }

  hipStream_t stream = hip_rocprim::stream(policy);
  //
  status = ::hipMemcpyAsync(dst, src, sizeof(Type) * count, hipMemcpyDeviceToDevice, stream);
  if (status != hipSuccess)
  {
    return status;
  }
  status = hip_rocprim::synchronize_optional(policy);
  return status;
}
#endif // !THRUST_COMPILER(NVRTC)

THRUST_HOST_DEVICE inline void throw_on_error(hipError_t status)
{
  // Clear the global HIP error state which may have been set by the last
  // call. Otherwise, errors may "leak" to unrelated kernel launches.
#ifdef THRUST_RDC_ENABLED
  (void) hipGetLastError();
#else
  NV_IF_TARGET(NV_IS_HOST, ((void) hipGetLastError();));
#endif

  if (hipSuccess != status)
  {
    // Can't use #if inside NV_IF_TARGET, use a temp macro to hoist the device
    // instructions out of the target logic.
#if defined(THRUST_RDC_ENABLED) || THRUST_HIP_PRINTF_ENABLED == 0

#  define THRUST_TEMP_DEVICE_CODE \
    THRUST_HIP_PRINTF("Thrust HIP backend error: %s: %s\n", hipGetErrorName(status), hipGetErrorString(status))

#else

#  define THRUST_TEMP_DEVICE_CODE THRUST_HIP_PRINTF("Thrust HIP backend error: %d\n", static_cast<int>(status))

#endif

    NV_IF_TARGET(NV_IS_HOST,
                 (throw thrust::system_error(status, thrust::hip_category());),
                 (THRUST_TEMP_DEVICE_CODE;
#if _THRUST_HAS_DEVICE_SYSTEM_STD
                  _THRUST_STD_NOVERSION::terminate();
#else
                  __builtin_trap();
                  __builtin_unreachable();
#endif
                  ));

#undef THRUST_TEMP_DEVICE_CODE
  }
}

THRUST_HOST_DEVICE inline void throw_on_error(hipError_t status, char const* msg)
{
  // Clear the global HIP error state which may have been set by the last
  // call. Otherwise, errors may "leak" to unrelated kernel launches.
#ifdef THRUST_RDC_ENABLED
  (void) hipGetLastError();
#else
  NV_IF_TARGET(NV_IS_HOST, ((void) hipGetLastError();));
#endif

  if (hipSuccess != status)
  {
    // Can't use #if inside NV_IF_TARGET, use a temp macro to hoist the device
    // instructions out of the target logic.
#if defined(THRUST_RDC_ENABLED) || THRUST_HIP_PRINTF_ENABLED == 0

#  define THRUST_TEMP_DEVICE_CODE \
    THRUST_HIP_PRINTF("Thrust HIP backend error: %s: %s: %s\n", hipGetErrorName(status), hipGetErrorString(status), msg)

#else

#  define THRUST_TEMP_DEVICE_CODE THRUST_HIP_PRINTF("Thrust HIP backend error: %d: %s\n", static_cast<int>(status), msg)

#endif

    NV_IF_TARGET(NV_IS_HOST,
                 (throw thrust::system_error(status, thrust::hip_category(), msg);),
                 (THRUST_TEMP_DEVICE_CODE;
#if _THRUST_HAS_DEVICE_SYSTEM_STD
                  _THRUST_STD_NOVERSION::terminate();
#else
                  __builtin_trap();
                  __builtin_unreachable();
#endif
                  ));

#undef THRUST_TEMP_DEVICE_CODE
  }
}
} // namespace hip_rocprim

THRUST_NAMESPACE_END
