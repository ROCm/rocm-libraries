//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#ifndef _CUDA__MEMORY_RESOURCE_PROPERTIES_H
#define _CUDA__MEMORY_RESOURCE_PROPERTIES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/type_set.h>
#include <cuda/std/cstddef>

#if defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR

//! @brief The default alignment by a cudaMalloc{...} call
_CCCL_INLINE_VAR constexpr size_t default_cuda_malloc_alignment = 256;

//! @brief The default alignment by a cudaMallocHost{...} call
_CCCL_INLINE_VAR constexpr size_t default_cuda_malloc_host_alignment = alignof(_CUDA_VSTD::max_align_t);

//! @brief The device_accessible property signals that the allocated memory is device accessible
struct device_accessible
{};

//! @brief The device_accessible property signals that the allocated memory is host accessible
struct host_accessible
{};

//! @brief determines whether a set of properties signals host accessible memory.
template <class... _Properties>
_CCCL_INLINE_VAR constexpr bool __is_host_accessible =
  _CUDA_VSTD::__type_set_contains_v<_CUDA_VSTD::__make_type_set<_Properties...>, host_accessible>;

//! @brief determines whether a set of properties signals device accessible memory.
template <class... _Properties>
_CCCL_INLINE_VAR constexpr bool __is_device_accessible =
  _CUDA_VSTD::__type_set_contains_v<_CUDA_VSTD::__make_type_set<_Properties...>, device_accessible>;

//! @brief determines whether a set of properties signals host device accessible memory.
template <class... _Properties>
_CCCL_INLINE_VAR constexpr bool __is_host_device_accessible =
  _CUDA_VSTD::__type_set_contains_v<_CUDA_VSTD::__make_type_set<_Properties...>, host_accessible, device_accessible>;

//! @brief verifies that a set of properties contains at least one execution space property
template <class... _Properties>
_CCCL_INLINE_VAR constexpr bool __contains_execution_space_property =
  __is_host_accessible<_Properties...> || __is_device_accessible<_Properties...>;

_LIBCUDACXX_END_NAMESPACE_CUDA_MR

#endif // LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif //_CUDA__MEMORY_RESOURCE_PROPERTIES_H
