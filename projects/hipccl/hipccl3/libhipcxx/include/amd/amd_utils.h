// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#pragma once
#if defined(__HIP__)
#include <hip/hip_runtime.h>
#include <rocm-core/rocm_version.h>
#endif
#ifndef __HIP_DEVICE_COMPILE__
#ifndef __host__ 
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

namespace libhipcxx
{
  __host__ __device__ inline void __trap(){
    #ifndef NDEBUG
    // #warning "__trap(): the call of __builtin_trap() will abort the host process. \
    // This deviates from the CUDA implementation where __trap() terminates a kernel \
    // and the host process is notified via an error status."
      __builtin_trap();
    #endif
  }
  #ifdef __HIP_DEVICE_COMPILE__
  /**
   * \brief Find First Set
   * \return index of first set bit of lowest significance.
   * \note Return value type matches that of the underlying device builtin.
   * \note While `uint64_t` is defined as `unsigned long int` on x86_64,
   *       the HIP `__ffsll` device function provides `__ffsll` with `unsigned long long int`
   *       argument, which is also an 64-bit integer type on x86_64.
   *       However, the compilers typically see both as different types.
   *       We work with `uint64t` and `uint32t` here, so explicit instantiations
   *       for both are added here.
   */
  template <typename T>
  __device__ inline int __FFS(T v);
  
  template <>
  __device__ inline int __FFS<int32_t>(int32_t v) {
    return __ffs(v);
  }
  // NOTE(HIP/AMD): hiprtc defines long long as int64_t, 
  // therefore, this specialization is not needed.
  #if !defined(_CCCL_COMPILER_HIPRTC)
  template <>
  __device__ inline int __FFS<int64_t>(int64_t v) {
    return __ffsll(static_cast<unsigned long long int>(v));
  }
  #endif
  template <>
  __device__ inline int __FFS<uint32_t>(uint32_t v) {
    return __ffs(v);
  }
  
  template <>
  __device__ inline int __FFS<unsigned long long>(unsigned long long v) {
    return __ffsll(static_cast<unsigned long long int>(v));
  }
  #if !defined(_CCCL_COMPILER_HIPRTC)
  template <>
  __device__ inline int __FFS<uint64_t>(uint64_t v) {
    return __ffsll(static_cast<unsigned long long int>(v));
  }
  #endif
  #endif
}

// Returns true if the current ROCm version is at least major.minor.patch
#define LIBHIPCXX_ROCM_VERSION_GE(major, minor, patch) \
    (defined(ROCM_VERSION_MAJOR) && \
     defined(ROCM_VERSION_MINOR) && \
     defined(ROCM_VERSION_PATCH) && \
     (ROCM_VERSION_MAJOR > (major)) || \
     (ROCM_VERSION_MAJOR == (major) && ROCM_VERSION_MINOR > (minor)) || \
     (ROCM_VERSION_MAJOR == (major) && ROCM_VERSION_MINOR == (minor) && ROCM_VERSION_PATCH >= (patch)))

// Returns true if the current ROCm version is at least major.minor the patch value is ignored
#define LIBHIPCXX_ROCM_VERSION_GE(major, minor) \
    (defined(ROCM_VERSION_MAJOR) && \
     defined(ROCM_VERSION_MINOR) && \
     (ROCM_VERSION_MAJOR > (major)) || \
     (ROCM_VERSION_MAJOR == (major) && ROCM_VERSION_MINOR >= (minor)))

// Returns true if the current ROCm version is at least major the minor and patch value is ignored
#define LIBHIPCXX_ROCM_VERSION_GE(major) \
    (defined(ROCM_VERSION_MAJOR) && \
     (ROCM_VERSION_MAJOR >= (major)))

// Returns true if the current ROCm version is at most major.minor.patch (i.e. >=)
#define LIBHIPCXX_ROCM_VERSION_LE(major, minor, patch) \
    (defined(ROCM_VERSION_MAJOR) && \
     defined(ROCM_VERSION_MINOR) && \
     defined(ROCM_VERSION_PATCH) && \
     (ROCM_VERSION_MAJOR < (major)) || \
     (ROCM_VERSION_MAJOR == (major) && ROCM_VERSION_MINOR < (minor)) || \
     (ROCM_VERSION_MAJOR == (major) && ROCM_VERSION_MINOR == (minor) && ROCM_VERSION_PATCH <= (patch)))

// Returns true if the current ROCm version is at most major.minor (i.e. >=) the patch value is ignored
#define LIBHIPCXX_ROCM_VERSION_LE(major, minor) \
    (defined(ROCM_VERSION_MAJOR) && \
     defined(ROCM_VERSION_MINOR) && \
     (ROCM_VERSION_MAJOR < (major)) || \
     (ROCM_VERSION_MAJOR == (major) && ROCM_VERSION_MINOR <= (minor)))

// Returns true if the current ROCm version is at most major (i.e. >=) the patch and minor value is ignored
#define LIBHIPCXX_ROCM_VERSION_LE(major) \
    (defined(ROCM_VERSION_MAJOR) && \
     (ROCM_VERSION_MAJOR <= (major)))

// Returns true if the current ROCm version matches exactly major.minor.patch
#define LIBHIPCXX_ROCM_VERSION_EQ(major, minor, patch) \
     ROCM_VERSION_MAJOR == (major) && ROCM_VERSION_MINOR == (minor) && ROCM_VERSION_PATCH == (patch)

// Returns true if the current ROCm version matches exactly major.minor (ignoring patch release)
#define LIBHIPCXX_ROCM_VERSION_EQ(major, minor) \
     ROCM_VERSION_MAJOR == (major) && ROCM_VERSION_MINOR == (minor)

// Returns true if the current ROCm version matches exactly major (ignoring minor and patch release)
#define LIBHIPCXX_ROCM_VERSION_EQ(major) \
     ROCM_VERSION_MAJOR == (major) && ROCM_VERSION_MINOR == (minor)
