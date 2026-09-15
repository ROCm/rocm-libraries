//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef _LIBCUDACXX___INTERNAL_NAMESPACES_H
#define _LIBCUDACXX___INTERNAL_NAMESPACES_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#define _LIBCUDACXX_CONCAT1(_LIBCUDACXX_X, _LIBCUDACXX_Y) _LIBCUDACXX_X##_LIBCUDACXX_Y
#define _LIBCUDACXX_CONCAT(_LIBCUDACXX_X, _LIBCUDACXX_Y)  _LIBCUDACXX_CONCAT1(_LIBCUDACXX_X, _LIBCUDACXX_Y)

#ifndef _LIBCUDACXX_ABI_NAMESPACE
#  define _LIBCUDACXX_ABI_NAMESPACE _LIBCUDACXX_CONCAT(__, _LIBCUDACXX_CUDA_ABI_VERSION)
#endif // _LIBCUDACXX_ABI_NAMESPACE

// clang-format off

// Standard namespaces with or without versioning
#  define _LIBCUDACXX_BEGIN_NAMESPACE_STD_NOVERSION namespace cuda { namespace std {
#  define _LIBCUDACXX_END_NAMESPACE_STD_NOVERSION } } \
    namespace hip = cuda;
#  define _LIBCUDACXX_BEGIN_NAMESPACE_STD namespace cuda { namespace std { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_STD } } } \
    namespace hip = cuda;

// cuda specific namespaces
#  define _LIBCUDACXX_BEGIN_NAMESPACE_CUDA namespace cuda { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_CUDA } } \
    namespace hip = cuda;
#  define _LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR namespace cuda { namespace mr { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_CUDA_MR } } } \
    namespace hip = cuda;
#  define _LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE namespace cuda { namespace device { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE } } } \
    namespace hip = cuda;
#  define _LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX namespace cuda { namespace ptx { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_CUDA_PTX } } } \
    namespace hip = cuda;
#  define _LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE_EXPERIMENTAL namespace cuda { namespace device { namespace experimental { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE_EXPERIMENTAL } } } } \
    namespace hip = cuda;

// Namespaces related to <ranges>
#  define _LIBCUDACXX_BEGIN_NAMESPACE_RANGES namespace cuda { namespace std { namespace ranges { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_RANGES } } } } \
    namespace hip = cuda;
#  define _LIBCUDACXX_BEGIN_NAMESPACE_VIEWS namespace cuda { namespace std { namespace ranges { namespace views { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_VIEWS } } } } } \
    namespace hip = cuda;

#  if _CCCL_STD_VER >= 2020
#    define _LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI inline namespace __cxx20 {
#  else
#    define _LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI inline namespace __cxx17 {
#  endif
#  define _LIBCUDACXX_END_NAMESPACE_RANGES_ABI }

#  define _LIBCUDACXX_BEGIN_NAMESPACE_CPO(_CPO) namespace _CPO { _LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI
#  define _LIBCUDACXX_END_NAMESPACE_CPO } _LIBCUDACXX_END_NAMESPACE_RANGES_ABI

// Namespaces related to chrono / filesystem
#  define _LIBCUDACXX_BEGIN_NAMESPACE_FILESYSTEM namespace cuda { namespace std { inline namespace __fs { namespace filesystem { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_FILESYSTEM } } } } }

// Shorthands for different qualifiers
#  define _CUDA_VSTD_NOVERSION ::cuda::std
#  define _CUDA_VSTD           ::cuda::std::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VRANGES        ::cuda::std::ranges::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VIEWS          ::cuda::std::ranges::views::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VMR            ::cuda::mr::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VPTX           ::cuda::ptx::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VSTD_FS        ::cuda::std::__fs::filesystem::_LIBCUDACXX_ABI_NAMESPACE

// clang-format on

#endif // _LIBCUDACXX___INTERNAL_NAMESPACES_H
