// Copyright (c) 2018 NVIDIA Corporation
// Modifications Copyright© 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/hip/pointer.h>
#include <thrust/system/hip/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

namespace system { namespace hip
{

struct ready_event;

template <typename T>
struct ready_future;

struct unique_eager_event;

template <typename T>
struct unique_eager_future;

template <typename... Events>
THRUST_HOST
unique_eager_event when_all(Events&&... evs);

}} // namespace system::hip

namespace hip
{

using thrust::system::hip::ready_event;

using thrust::system::hip::ready_future;

using thrust::system::hip::unique_eager_event;
using event = unique_eager_event;

using thrust::system::hip::unique_eager_future;
template <typename T> using future = unique_eager_future<T>;

using thrust::system::hip::when_all;

} // namespace hip

template <typename DerivedPolicy>
THRUST_HOST
thrust::hip::unique_eager_event
unique_eager_event_type(
  thrust::hip::execution_policy<DerivedPolicy> const&
) noexcept;

template <typename T, typename DerivedPolicy>
THRUST_HOST
thrust::hip::unique_eager_future<T>
unique_eager_future_type(
  thrust::hip::execution_policy<DerivedPolicy> const&
) noexcept;

THRUST_NAMESPACE_END

#include <thrust/system/hip/detail/future.inl>
