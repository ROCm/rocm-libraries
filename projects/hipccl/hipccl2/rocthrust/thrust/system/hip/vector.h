/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
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

/*! \file thrust/system/hip/vector.h
 *  \brief A dynamically-sizable array of elements which reside in memory available to
 *         Thrust's hip system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/hip/memory.h>
#include <thrust/detail/vector_base.h>
#include <vector>

namespace thrust
{

// forward declaration of host_vector
template<typename T, typename Allocator> class host_vector;

namespace hip_rocprim
{

// XXX upon c++11
// template<typename T, typename Allocator = allocator<T> > using vector = thrust::detail::vector_base<T,Allocator>;

/*! \p hip_rocprim::vector is a container that supports random access to elements,
 *  constant time removal of elements at the end, and linear time insertion
 *  and removal of elements at the beginning or in the middle. The number of
 *  elements in a \p hip_rocprim::vector may vary dynamically; memory management is
 *  automatic. The elements contained in a \p hip_rocprim::vector reside in memory
 *  available to the \p hip_rocprim system.
 *
 *  \tparam T The element type of the \p hip_rocprim::vector.
 *  \tparam Allocator The allocator type of the \p hip_rocprim::vector. Defaults to \p hip_rocprim::allocator.
 *
 *  \see http://www.sgi.com/tech/stl/Vector.html
 *  \see host_vector For the documentation of the complete interface which is
 *                   shared by \p hip_rocprim::vector
 *  \see device_vector
 */
template<typename T, typename Allocator = allocator<T> >
  class vector
    : public thrust::detail::vector_base<T,Allocator>
{
  /*! \cond
   */
  private:
    typedef thrust::detail::vector_base<T,Allocator> super_t;
  /*! \endcond
   */

  public:

  /*! \cond
   */
    typedef typename super_t::size_type  size_type;
    typedef typename super_t::value_type value_type;
  /*! \endcond
   */

    /*! This constructor creates an empty \p hip_rocprim::vector.
     */
    vector();

    /*! This constructor creates a \p hip_rocprim::vector with \p n default-constructed elements.
     *  \param n The size of the \p hip_rocprim::vector to create.
     */
    explicit vector(size_type n);

    /*! This constructor creates a \p hip_rocprim::vector with \p n copies of \p value.
     *  \param n The size of the \p hip_rocprim::vector to create.
     *  \param value An element to copy.
     */
    explicit vector(size_type n, const value_type &value);

    /*! Copy constructor copies from another \p hip_rocprim::vector.
     *  \param x The other \p hip_rocprim::vector to copy.
     */
    vector(const vector &x);

#if __cplusplus >= 201103L
    /*! Move constructor moves from over another \p cuda::vector.
     *  \param x The other \p cuda::vector to move from.
     */
    vector(vector &&x);
#endif

    /*! This constructor copies from another Thrust vector-like object.
     *  \param x The other object to copy from.
     */
    template<typename OtherT, typename OtherAllocator>
    vector(const thrust::detail::vector_base<OtherT,OtherAllocator> &x);

    /*! This constructor copies from a \c std::vector.
     *  \param x The \c std::vector to copy from.
     */
    template<typename OtherT, typename OtherAllocator>
    vector(const std::vector<OtherT,OtherAllocator> &x);

    /*! This constructor creates a \p hip_rocprim::vector by copying from a range.
     *  \param first The beginning of the range.
     *  \param last The end of the range.
     */
    template<typename InputIterator>
    vector(InputIterator first, InputIterator last);

    /*! Assignment operator assigns from another \p cuda::vector.
     *  \param x The other object to assign from.
     *  \return <tt>*this</tt>
     */
    vector &operator=(const vector &x);

  #if __cplusplus >= 201103L
    /*! Move assignment operator moves from another \p cuda::vector.
     *  \param x The other \p cuda::vector to move from.
     *  \return <tt>*this</tt>
     */
     vector &operator=(vector &&x);
  #endif

    // XXX vector_base should take a Derived type so we don't have to define these superfluous assigns
    //
    /*! Assignment operator assigns from a \c std::vector.
     *  \param x The \c std::vector to assign from.
     *  \return <tt>*this</tt>
     */
    template<typename OtherT, typename OtherAllocator>
    vector &operator=(const std::vector<OtherT,OtherAllocator> &x);

    /*! Assignment operator assigns from another Thrust vector-like object.
     *  \param x The other object to assign from.
     *  \return <tt>*this</tt>
     */
    template<typename OtherT, typename OtherAllocator>
    vector &operator=(const thrust::detail::vector_base<OtherT,OtherAllocator> &x);
}; // end vector

} // end hip_rocprim

// alias system::hip_rocprim names at top-level
namespace hip_rocprim
{

using thrust::hip_rocprim::vector;

} // end hip_rocprim

} // end thrust

#include <thrust/system/hip/detail/vector.inl>
