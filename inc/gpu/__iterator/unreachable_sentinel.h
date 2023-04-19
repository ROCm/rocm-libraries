// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ITERATOR_UNREACHABLE_SENTINEL_H
#define __GPU___ITERATOR_UNREACHABLE_SENTINEL_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 20

struct unreachable_sentinel_t {
  template<weakly_incrementable _Iter>
  __device__ _LIBGPU_HIDE_FROM_ABI
  friend constexpr bool operator==(unreachable_sentinel_t, const _Iter&) noexcept {
    return false;
  }
};

inline constexpr unreachable_sentinel_t unreachable_sentinel{};

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___ITERATOR_UNREACHABLE_SENTINEL_H
