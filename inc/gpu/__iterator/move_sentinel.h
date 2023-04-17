//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ITERATOR_MOVE_SENTINEL_H
#define __GPU___ITERATOR_MOVE_SENTINEL_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 20

template <semiregular _Sent>
class _LIBGPU_TEMPLATE_VIS move_sentinel
{
public:
  _LIBGPU_HIDE_FROM_ABI
  move_sentinel() = default;

  _LIBGPU_HIDE_FROM_ABI constexpr
  explicit move_sentinel(_Sent __s) : __last_(std::move(__s)) {}

  template <class _S2>
    requires std::convertible_to<const _S2&, _Sent>
  _LIBGPU_HIDE_FROM_ABI constexpr
  move_sentinel(const move_sentinel<_S2>& __s) : __last_(__s.base()) {}

  template <class _S2>
    requires assignable_from<_Sent&, const _S2&>
  _LIBGPU_HIDE_FROM_ABI constexpr
  move_sentinel& operator=(const move_sentinel<_S2>& __s)
    { __last_ = __s.base(); return *this; }

  constexpr _Sent base() const { return __last_; }

private:
    _Sent __last_ = _Sent();
};

_LIBGPU_CTAD_SUPPORTED_FOR_TYPE(move_sentinel);

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___ITERATOR_MOVE_SENTINEL_H
