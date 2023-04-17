// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ITERATOR_DEFAULT_SENTINEL_H
#define __GPU___ITERATOR_DEFAULT_SENTINEL_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 20

struct default_sentinel_t { };
inline constexpr default_sentinel_t default_sentinel{};

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___ITERATOR_DEFAULT_SENTINEL_H
