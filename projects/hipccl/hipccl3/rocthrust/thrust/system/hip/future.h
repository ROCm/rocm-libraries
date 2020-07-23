// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/system/hip/pointer.h>
#include <thrust/system/hip/detail/execution_policy.h>

#include <thrust/future.h>

THRUST_BEGIN_NS

namespace system { namespace hip
{

template <typename T>
struct ready_future;

template <typename T, typename Pointer = pointer<T>>
struct unique_eager_future;

}} // namespace system::hip

namespace hip
{

template <typename T>
using ready_future = thrust::system::hip::ready_future<T>;

template <typename T, typename Pointer = thrust::system::hip::pointer<T>>
using unique_eager_future = thrust::system::hip::unique_eager_future<T, Pointer>;

} // namespace hip

template <typename T, typename Pointer, typename DerivedPolicy>
__host__ __device__
thrust::system::hip::unique_eager_future<T, Pointer>
unique_eager_future_type(thrust::hip_rocprim::execution_policy<DerivedPolicy> const&);

THRUST_END_NS

#include <thrust/system/hip/detail/future.inl>

#endif // THRUST_CPP_DIALECT >= 2011
