/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#include <thrust/system/detail/generic/tag.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename ForwardIterator>
THRUST_HOST_DEVICE void
sequence(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last);

template <typename DerivedPolicy, typename ForwardIterator, typename T>
THRUST_HOST_DEVICE void
sequence(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, T init);

template <typename DerivedPolicy, typename ForwardIterator, typename T>
THRUST_HOST_DEVICE void
sequence(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, T init, T step);

} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/sequence.inl>
