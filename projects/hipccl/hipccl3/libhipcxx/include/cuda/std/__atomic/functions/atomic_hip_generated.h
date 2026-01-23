//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <hip/hip_runtime.h>

//#include <cuda/std/cassert>
//#include <cuda/std/cstdint>

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>

#include <cuda/std/__atomic/scopes.h>
//#include <cuda/std/__atomic/order.h>
//#include <cuda/std/__atomic/functions/common.h>
//#include <cuda/std/__atomic/functions/cuda_ptx_generated_helper.h>
_LIBCUDACXX_BEGIN_NAMESPACE_STD

static inline __device__ void __atomic_thread_fence_cuda(int __memorder, __thread_scope_block_tag) {
    __threadfence_block();
}

template<class _Type>
__device__ void __atomic_load_cuda(const volatile _Type *__ptr, _Type& __dst, int __memorder, __thread_scope_block_tag) {
    __dst = __hip_atomic_load(__ptr, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type>
__device__ void __atomic_store_cuda(volatile _Type *__ptr, _Type& __val, int __memorder, __thread_scope_block_tag) {
    __hip_atomic_store(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type __desired, bool, int __success_memorder, int __failure_memorder, __thread_scope_block_tag) {
    return __hip_atomic_compare_exchange_weak(__ptr, __expected, __desired, __success_memorder, __failure_memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type>
__device__ void __atomic_exchange_cuda(volatile _Type* __ptr, _Type& __old, _Type __new, int __memorder, __thread_scope_block_tag) {
    __old = __hip_atomic_exchange(__ptr, __new, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type>
__device__ _Type __atomic_fetch_and_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_and(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type>
__device__ _Type __atomic_fetch_or_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_or(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type>
__device__ _Type __atomic_fetch_xor_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_xor(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_max(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_min(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

template<class _Type>
__device__ _Type* __atomic_fetch_add_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, __thread_scope_block_tag) {
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_WORKGROUP);
}

static inline __device__ void __atomic_thread_fence_cuda(int __memorder, __thread_scope_device_tag) {
    __threadfence();
}

template<class _Type>
__device__ void __atomic_load_cuda(const volatile _Type *__ptr, _Type& __dst, int __memorder, __thread_scope_device_tag) {
    __dst = __hip_atomic_load(__ptr, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type>
__device__ void __atomic_store_cuda(volatile _Type *__ptr, _Type& __val, int __memorder, __thread_scope_device_tag) {
    __hip_atomic_store(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type __desired, bool, int __success_memorder, int __failure_memorder, __thread_scope_device_tag) {
    return __hip_atomic_compare_exchange_weak(__ptr, __expected, __desired, __success_memorder, __failure_memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type>
__device__ void __atomic_exchange_cuda(volatile _Type* __ptr, _Type& __old, _Type __new, int __memorder, __thread_scope_device_tag) {
    __old = __hip_atomic_exchange(__ptr, __new, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type>
__device__ _Type __atomic_fetch_and_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_and(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type>
__device__ _Type __atomic_fetch_or_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_or(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type>
__device__ _Type __atomic_fetch_xor_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_xor(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_max(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_min(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

template<class _Type>
__device__ _Type* __atomic_fetch_add_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}
template<class _Type>
__device__ _Type* __atomic_fetch_sub_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, __thread_scope_device_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_AGENT);
}

static inline __device__ void __atomic_thread_fence_cuda(int __memorder, __thread_scope_system_tag) {
    __threadfence_system();
}

template<class _Type>
__device__ void __atomic_load_cuda(const volatile _Type *__ptr, _Type& __dst, int __memorder, __thread_scope_system_tag) {
    __dst = __hip_atomic_load(__ptr, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type>
__device__ void __atomic_store_cuda(volatile _Type *__ptr, _Type& __val, int __memorder, __thread_scope_system_tag) {
    __hip_atomic_store(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type __desired, bool __is_weak, int __success_memorder, int __failure_memorder, __thread_scope_system_tag) {
    if(__is_weak)
        return __hip_atomic_compare_exchange_weak(__ptr, __expected, __desired, __success_memorder, __failure_memorder, __HIP_MEMORY_SCOPE_SYSTEM);
    else 
        return __hip_atomic_compare_exchange_strong(__ptr, __expected, __desired, __success_memorder, __failure_memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type>
__device__ void __atomic_exchange_cuda(volatile _Type* __ptr, _Type& __old, _Type __new, int __memorder, __thread_scope_system_tag) {
    __old = __hip_atomic_exchange(__ptr, __new, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type>
__device__ _Type __atomic_fetch_and_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_and(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type>
__device__ _Type __atomic_fetch_or_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_or(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type>
__device__ _Type __atomic_fetch_xor_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_xor(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
   return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_max(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_min(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

template<class _Type>
__device__ _Type* __atomic_fetch_add_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}
template<class _Type>
__device__ _Type* __atomic_fetch_sub_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, __thread_scope_system_tag) {
    return __hip_atomic_fetch_add(__ptr, -__val, __memorder, __HIP_MEMORY_SCOPE_SYSTEM);
}

_LIBCUDACXX_END_NAMESPACE_STD
