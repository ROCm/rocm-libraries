// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file distance.h
 *  \brief Computes the size of a range
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include _THRUST_STD_INCLUDE(__iterator/distance.h)
#include _THRUST_STD_INCLUDE(__iterator/iterator_traits.h)

THRUST_NAMESPACE_BEGIN

//! deprecated [since 10.1]

template <class InputIter>
[[nodiscard]] CCCL_DEPRECATED_BECAUSE("Use hip::std::distance instead")
_CCCL_API constexpr typename _THRUST_STD::iterator_traits<InputIter>::difference_type
distance(InputIter first, InputIter last)
{
  return _THRUST_STD::distance(first, last);
}

THRUST_NAMESPACE_END
