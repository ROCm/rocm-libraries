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

#ifndef FENCE_H
#define FENCE_H

#include <string>

#include "definitions.h"
#include <fmt/format.h>

inline std::string membar_scope(Scope sco)
{
  static std::map scope_map{
    std::pair{Scope::GPU, ".gl"},
    std::pair{Scope::System, ".sys"},
    std::pair{Scope::CTA, ".cta"},
  };

  return scope_map[sco];
}

inline void FormatFence(std::ostream& out)
{
  // Argument ID Reference
  // 0 - Membar scope tag
  // 1 - Membar scope
  const std::string intrinsic_membar = R"XXX(
static inline _CCCL_DEVICE void __cuda_atomic_membar({0})
{{ asm volatile("membar{1};" ::: "memory"); }})XXX";

  const std::map membar_scopes{
    std::pair{Scope::GPU, ".gl"},
    std::pair{Scope::System, ".sys"},
    std::pair{Scope::CTA, ".cta"},
  };

  for (const auto& sco : membar_scopes)
  {
    out << fmt::format(intrinsic_membar, scope_tag(sco.first), sco.second);
  }

  // Argument ID Reference
  // 0 - Fence scope tag
  // 1 - Fence scope
  // 2 - Fence order tag
  // 3 - Fence order
  const std::string intrinsic_fence = R"XXX(
static inline _CCCL_DEVICE void __cuda_atomic_fence({0}, {2})
{{ asm volatile("fence{1}{3};" ::: "memory"); }})XXX";

  const Scope fence_scopes[] = {
    Scope::CTA,
    Scope::Cluster,
    Scope::GPU,
    Scope::System,
  };

  const Semantic fence_semantics[] = {
    Semantic::Acq_Rel,
    Semantic::Seq_Cst,
  };

  for (const auto& sco : fence_scopes)
  {
    for (const auto& sem : fence_semantics)
    {
      out << fmt::format(intrinsic_fence, scope_tag(sco), semantic(sem), semantic_tag(sem), scope(sco));
    }
  }
  out << "\n"
      << R"XXX(
template <typename _Sco>
static inline _CCCL_DEVICE void __atomic_thread_fence_cuda(int __memorder, _Sco) {
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_70, (
      switch (__memorder) {
        case __ATOMIC_SEQ_CST: __cuda_atomic_fence(_Sco{}, __atomic_cuda_seq_cst{}); break;
        case __ATOMIC_CONSUME: _CCCL_FALLTHROUGH();
        case __ATOMIC_ACQUIRE: _CCCL_FALLTHROUGH();
        case __ATOMIC_ACQ_REL: _CCCL_FALLTHROUGH();
        case __ATOMIC_RELEASE: __cuda_atomic_fence(_Sco{}, __atomic_cuda_acq_rel{}); break;
        case __ATOMIC_RELAXED: break;
        default: assert(0);
      }
    ),
    NV_IS_DEVICE_LIBHIPCXX, (
      switch (__memorder) {
        case __ATOMIC_SEQ_CST: _CCCL_FALLTHROUGH();
        case __ATOMIC_CONSUME: _CCCL_FALLTHROUGH();
        case __ATOMIC_ACQUIRE: _CCCL_FALLTHROUGH();
        case __ATOMIC_ACQ_REL: _CCCL_FALLTHROUGH();
        case __ATOMIC_RELEASE: __cuda_atomic_membar(_Sco{}); break;
        case __ATOMIC_RELAXED: break;
        default: assert(0);
      }
    )
  )
}
)XXX";
}

#endif // FENCE_H
