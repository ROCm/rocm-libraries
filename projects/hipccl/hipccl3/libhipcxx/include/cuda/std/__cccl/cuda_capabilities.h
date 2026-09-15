//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

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

#ifndef __CCCL_CUDA_CAPABILITIES
#define __CCCL_CUDA_CAPABILITIES

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// CUDA headers might not be present when using NVRTC, see NVIDIA/cccl#2095 for detail
#if !_CCCL_COMPILER(NVRTC) && not defined(__HIP_PLATFORM_AMD__)
#  include <cuda_runtime_api.h>
#endif // !_CCCL_COMPILER(NVRTC)

#include <nv/target>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#ifdef _CCCL_DOXYGEN_INVOKED // Only parse this during doxygen passes:
//! When this macro is defined, Programmatic Dependent Launch (PDL) is disabled across CCCL
#  define CCCL_DISABLE_PDL
#endif // _CCCL_DOXYGEN_INVOKED

#ifdef CCCL_DISABLE_PDL
#  define _CCCL_HAS_PDL() 0
#else // CCCL_DISABLE_PDL
#  define _CCCL_HAS_PDL() _CCCL_CTK_AT_LEAST(12, 0)
#endif // CCCL_DISABLE_PDL

#if _CCCL_HAS_PDL()
// Waits for the previous kernel to complete (when it reaches its final membar). Should be put before the first global
// memory access in a kernel.
#  define _CCCL_PDL_GRID_DEPENDENCY_SYNC() NV_IF_TARGET_LIBHIPCXX(NV_PROVIDES_SM_90, cudaGridDependencySynchronize();)
// Allows the subsequent kernel in the same stream to launch. Can be put anywhere in a kernel.
// Heuristic(ahendriksen): put it after the last load.
#  define _CCCL_PDL_TRIGGER_NEXT_LAUNCH() NV_IF_TARGET_LIBHIPCXX(NV_PROVIDES_SM_90, cudaTriggerProgrammaticLaunchCompletion();)
#else // _CCCL_HAS_PDL()
#  define _CCCL_PDL_GRID_DEPENDENCY_SYNC()
#  define _CCCL_PDL_TRIGGER_NEXT_LAUNCH()
#endif // _CCCL_HAS_PDL()

#endif // __CCCL_CUDA_CAPABILITIES
