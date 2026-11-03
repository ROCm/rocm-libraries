// SPDX-FileCopyrightText: Copyright (c) 2008-2021, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include _THRUST_STD_INCLUDE(__type_traits/conjunction.h)
#include _THRUST_STD_INCLUDE(__type_traits/disjunction.h)
#include _THRUST_STD_INCLUDE(__type_traits/negation.h)

THRUST_NAMESPACE_BEGIN

using _THRUST_STD::conjunction;
using _THRUST_STD::conjunction_v;
using _THRUST_STD::disjunction;
using _THRUST_STD::disjunction_v;
using _THRUST_STD::negation;
using _THRUST_STD::negation_v;

THRUST_NAMESPACE_END
