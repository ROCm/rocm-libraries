// Copyright (c) 2018 NVIDIA Corporation
// Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/detail/cpp_version_check.h>

#if THRUST_CPP_DIALECT >= 2017

#  include <thrust/system/hip/detail/execution_policy.h>
#  include <thrust/system/hip/pointer.h>

THRUST_NAMESPACE_BEGIN

namespace system
{
namespace hip
{

struct ready_event;

template <typename T>
struct ready_future;

struct unique_eager_event;

template <typename T>
struct unique_eager_future;

template <typename... Events>
THRUST_HOST unique_eager_event when_all(Events&&... evs);

} // namespace hip
} // namespace system

namespace hip
{

using thrust::system::hip::ready_event;

using thrust::system::hip::ready_future;

using thrust::system::hip::unique_eager_event;
using event = unique_eager_event;

using thrust::system::hip::unique_eager_future;
template <typename T>
using future = unique_eager_future<T>;

using thrust::system::hip::when_all;

} // namespace hip

THRUST_SUPPRESS_DEPRECATED_PUSH
template <typename DerivedPolicy>
THRUST_HOST thrust::hip::unique_eager_event
unique_eager_event_type(thrust::hip::execution_policy<DerivedPolicy> const&) noexcept;

template <typename T, typename DerivedPolicy>
THRUST_HOST thrust::hip::unique_eager_future<T>
unique_eager_future_type(thrust::hip::execution_policy<DerivedPolicy> const&) noexcept;
THRUST_SUPPRESS_DEPRECATED_POP

THRUST_NAMESPACE_END

#  include <thrust/system/hip/detail/future.inl>

#endif // C++17
