// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ITERATOR_FRONT_INSERT_ITERATOR_H
#define __GPU___ITERATOR_FRONT_INSERT_ITERATOR_H

#include "gpu/__config"

namespace gpu {

_LIBGPU_SUPPRESS_DEPRECATED_PUSH
template <class _Container>
class _LIBGPU_TEMPLATE_VIS front_insert_iterator
#if _LIBGPU_STD_VER <= 14 || !defined(_LIBGPU_ABI_NO_ITERATOR_BASES)
    : public iterator<std::output_iterator_tag, void, void, void, void>
#endif
{
_LIBGPU_SUPPRESS_DEPRECATED_POP
protected:
    _Container* container;
public:
    typedef std::output_iterator_tag iterator_category;
    typedef void value_type;
#if _LIBGPU_STD_VER >= 20
    typedef std::ptrdiff_t difference_type;
#else
    typedef void difference_type;
#endif
    typedef void pointer;
    typedef void reference;
    typedef _Container container_type;

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 explicit front_insert_iterator(_Container& __x) : container(std::addressof(__x)) {}
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 front_insert_iterator& operator=(const typename _Container::value_type& __value)
        {container->push_front(__value); return *this;}
#ifndef _LIBGPU_CXX03_LANG
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 front_insert_iterator& operator=(typename _Container::value_type&& __value)
        {container->push_front(std::move(__value)); return *this;}
#endif // _LIBGPU_CXX03_LANG
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 front_insert_iterator& operator*()     {return *this;}
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 front_insert_iterator& operator++()    {return *this;}
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 front_insert_iterator  operator++(int) {return *this;}
};
_LIBGPU_CTAD_SUPPORTED_FOR_TYPE(front_insert_iterator);

template <class _Container>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
front_insert_iterator<_Container>
front_inserter(_Container& __x)
{
    return front_insert_iterator<_Container>(__x);
}

} // namespace gpu

#endif // __GPU___ITERATOR_FRONT_INSERT_ITERATOR_H
