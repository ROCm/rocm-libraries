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

// NOTE(HIP/AMD): AMD does not have a PTX equivalent
// UNSUPPORTED: hipcc, hiprtc

// UNSUPPORTED: libcpp-has-no-threads

// <cuda/ptx>

#include <cuda/ptx>
#include <cuda/std/utility>

/*
 * We use a special strategy to force the generation of the PTX. This is mainly
 * a fight against dead-code-elimination in the NVVM layer.
 *
 * The reason we need this strategy is because certain older versions of ptxas
 * segfault when a non-sensical sequence of PTX is generated. So instead, we try
 * to force the instantiation and compilation to PTX of all the overloads of the
 * PTX wrapping functions.
 *
 * We do this by writing a function pointer of each overload to the kernel
 * parameter `fn_ptr`.
 *
 * Because `fn_ptr` is possibly visible outside this translation unit, the
 * compiler must compile all the functions which are stored.
 *
 */

__global__ void test_tensormap_replace(void** fn_ptr)
{
#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.global_address.global.b1024.b64    [tm_addr], new_val;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, void*, int64_t)>(
          cuda::ptx::tensormap_replace_global_address));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.global_address.shared::cta.b1024.b64    [tm_addr], new_val;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_shared_t, void*, int64_t)>(
          cuda::ptx::tensormap_replace_global_address));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.rank.global.b1024.b32              [tm_addr], new_val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, void*, int32_t)>(cuda::ptx::tensormap_replace_rank));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.rank.shared::cta.b1024.b32              [tm_addr], new_val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, void*, int32_t)>(cuda::ptx::tensormap_replace_rank));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.box_dim.global.b1024.b32           [tm_addr], ord, new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, void*, cuda::ptx::n32_t<0>, int32_t)>(
            cuda::ptx::tensormap_replace_box_dim));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.box_dim.shared::cta.b1024.b32           [tm_addr], ord, new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_shared_t, void*, cuda::ptx::n32_t<0>, int32_t)>(
            cuda::ptx::tensormap_replace_box_dim));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.global_dim.global.b1024.b32        [tm_addr], ord, new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, void*, cuda::ptx::n32_t<0>, int32_t)>(
            cuda::ptx::tensormap_replace_global_dim));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.global_dim.shared::cta.b1024.b32        [tm_addr], ord, new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_shared_t, void*, cuda::ptx::n32_t<0>, int32_t)>(
            cuda::ptx::tensormap_replace_global_dim));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.global_stride.global.b1024.b64     [tm_addr], ord, new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, void*, cuda::ptx::n32_t<0>, int64_t)>(
            cuda::ptx::tensormap_replace_global_stride));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.global_stride.shared::cta.b1024.b64     [tm_addr], ord, new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_shared_t, void*, cuda::ptx::n32_t<0>, int64_t)>(
            cuda::ptx::tensormap_replace_global_stride));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.element_stride.global.b1024.b32    [tm_addr], ord, new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, void*, cuda::ptx::n32_t<0>, int32_t)>(
            cuda::ptx::tensormap_replace_element_size));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.element_stride.shared::cta.b1024.b32    [tm_addr], ord, new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_shared_t, void*, cuda::ptx::n32_t<0>, int32_t)>(
            cuda::ptx::tensormap_replace_element_size));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.elemtype.global.b1024.b32          [tm_addr], new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, void*, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tensormap_replace_elemtype));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.elemtype.shared::cta.b1024.b32          [tm_addr], new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_shared_t, void*, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tensormap_replace_elemtype));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.interleave_layout.global.b1024.b32 [tm_addr], new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, void*, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tensormap_replace_interleave_layout));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [tm_addr], new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_shared_t, void*, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tensormap_replace_interleave_layout));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.swizzle_mode.global.b1024.b32      [tm_addr], new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, void*, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tensormap_replace_swizzle_mode));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32      [tm_addr], new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_shared_t, void*, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tensormap_replace_swizzle_mode));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.fill_mode.global.b1024.b32         [tm_addr], new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, void*, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tensormap_replace_fill_mode));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.fill_mode.shared::cta.b1024.b32         [tm_addr], new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_shared_t, void*, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tensormap_replace_fill_mode));));
#endif // __cccl_ptx_isa >= 830
}

int main(int, char**)
{
  return 0;
}
