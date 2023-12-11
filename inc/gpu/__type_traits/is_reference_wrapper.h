//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___TYPE_TRAITS_IS_REFERENCE_WRAPPER_H
#define __GPU___TYPE_TRAITS_IS_REFERENCE_WRAPPER_H

#include "gpu/__config"
#include <functional>

namespace gpu {

template <class _Tp> struct __is_reference_wrapper_impl : public std::false_type {};
template <class _Tp> struct __is_reference_wrapper_impl<std::reference_wrapper<_Tp> > : public std::true_type {};
template <class _Tp> struct __is_reference_wrapper
    : public __is_reference_wrapper_impl<std::remove_cv_t<_Tp> > {};

} // namespace gpu

#endif // __GPU___TYPE_TRAITS_IS_REFERENCE_WRAPPER_H
