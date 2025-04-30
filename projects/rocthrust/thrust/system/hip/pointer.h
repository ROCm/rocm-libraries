/******************************************************************************
 *  Copyright 2008-2020 NVIDIA Corporation
 *  Modifications Copyright (c) 2019-2021, Advanced Micro Devices, Inc.  All rights reserved.
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

/*! \file thrust/system/hip/memory.h
 *  \brief Managing memory associated with Thrust's Standard C++ system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/hip/detail/execution_policy.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/pointer.h>
#include <thrust/detail/reference.h>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

/*! \p hip::pointer stores a pointer to an object allocated in memory
 *  accessible by the \p hip system. This type provides type safety when
 *  dispatching algorithms on ranges resident in \p hip memory.
 *
 *  \p hip::pointer has pointer semantics: it may be dereferenced and
 *  manipulated with pointer arithmetic.
 *
 *  \p hip::pointer can be created with the function \p hip::malloc, or by
 *  explicitly calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p hip::pointer may be obtained by eiter
 *  its <tt>get</tt> member function or the \p raw_pointer_cast function.
 *
 *  \note \p hip::pointer is not a "smart" pointer; it is the programmer's
 *        responsibility to deallocate memory pointed to by \p hip::pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see hip::malloc
 *  \see hip::free
 *  \see raw_pointer_cast
 */
template <typename T>
using pointer = thrust::pointer<
  T,
  thrust::hip_rocprim::tag,
  thrust::tagged_reference<T, thrust::hip_rocprim::tag>
>;

/*! \p hip::universal_pointer stores a pointer to an object allocated in
 *  memory accessible by the \p hip system and host systems.
 *
 *  \p hip::universal_pointer has pointer semantics: it may be dereferenced
 *  and manipulated with pointer arithmetic.
 *
 *  \p hip::universal_pointer can be created with \p hip::universal_allocator
 *  or by explicitly calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p hip::universal_pointer may be
 *  obtained by eiter its <tt>get</tt> member function or the \p
 *  raw_pointer_cast function.
 *
 *  \note \p hip::universal_pointer is not a "smart" pointer; it is the
 *        programmer's responsibility to deallocate memory pointed to by
 *        \p hip::universal_pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see hip::universal_allocator
 *  \see raw_pointer_cast
 */
template <typename T>
using universal_pointer = thrust::pointer<
  T,
  thrust::hip_rocprim::tag,
  typename std::add_lvalue_reference<T>::type
>;

/*! \p hip::reference is a wrapped reference to an object stored in memory
 *  accessible by the \p hip system. \p hip::reference is the type of the
 *  result of dereferencing a \p hip::pointer.
 *
 *  \tparam T Specifies the type of the referenced object.
 *
 *  \see hip::pointer
 */
template <typename T>
using reference = thrust::tagged_reference<T, thrust::hip_rocprim::tag>;

} // end hip_rocprim
/*! \addtogroup system_backends Systems
 *  \ingroup system
 *  \{
 */

/*! \namespace thrust::system::hip
 *  \brief \p thrust::system::hip is the namespace containing functionality
 *  for allocating, manipulating, and deallocating memory available to Thrust's
 *  HIP backend system. The identifiers are provided in a separate namespace
 *  underneath <tt>thrust::system</tt> for import convenience but are also
 *  aliased in the top-level <tt>thrust::hip</tt> namespace for easy access.
 *
 */
namespace system { namespace hip
{
using thrust::hip_rocprim::pointer;
using thrust::hip_rocprim::universal_pointer;
using thrust::hip_rocprim::reference;
}} // namespace system::hip
/*! \}
 */

/*! \namespace thrust::hip
 *  \brief \p thrust::hip is a top-level alias for \p thrust::system::hip.
 */
namespace hip
{
using thrust::hip_rocprim::pointer;
using thrust::hip_rocprim::universal_pointer;
using thrust::hip_rocprim::reference;
} // namespace hip

THRUST_NAMESPACE_END
