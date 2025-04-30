/*
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

/*! \file thrust/system/hip/hipstdpar/impl/search.hpp
 *  \brief <tt>Search operations</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#include "hipstd.hpp"

#include <thrust/count.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/logical.h>
#include <thrust/mismatch.h>

#include <rocprim/rocprim.hpp>

#include <algorithm>
#include <execution>
#include <utility>

// rocThrust is currently missing some API entries, forward calls to rocPRIM until they are added.
namespace thrust
{
// BEGIN FIND_FIRST_OF
template<class InputIt1, class InputIt2, class BinaryPred>
InputIt1 THRUST_HIP_FUNCTION
find_first_of(InputIt1 first, InputIt1 last,
              InputIt2 s_first, InputIt2 s_last,
              BinaryPred p)
{
    if (s_first == s_last)
    {
        return last;
    }

    thrust::device_system_tag dev_tag;
    size_t* d_output;
    d_output = thrust::malloc<size_t>(dev_tag, sizeof(*d_output)).get();

    size_t size = last - first;
    size_t s_size = s_last - s_first;

    // temp storage
    size_t temp_storage_size_bytes;
    void*  d_temp_storage = nullptr;
    // Get size of d_temp_storage
    hipError_t error = ::rocprim::find_first_of(d_temp_storage,
                                              temp_storage_size_bytes,
                                              first,
                                              s_first,
                                              d_output,
                                              size,
                                              s_size,
                                              p);

    if (error != hipSuccess)
    {
        return last;
    }

    d_temp_storage = thrust::malloc(dev_tag, temp_storage_size_bytes).get();

    error = ::rocprim::find_first_of(d_temp_storage,
                                   temp_storage_size_bytes,
                                   first,
                                   s_first,
                                   d_output,
                                   size,
                                   s_size,
                                   p);

    if (error != hipSuccess)
    {
        thrust::free(dev_tag, d_temp_storage);
        thrust::free(dev_tag, d_output);
        return last;
    }

    error = hipDeviceSynchronize();
    if (error != hipSuccess)
    {
        thrust::free(dev_tag, d_temp_storage);
        thrust::free(dev_tag, d_output);
        return last;
    }

    size_t offset;
    hipMemcpy(&offset, d_output, sizeof(offset), hipMemcpyDeviceToHost);

    thrust::free(dev_tag, d_temp_storage);
    thrust::free(dev_tag, d_output);

    return first + offset;
}
// END FIND_FIRST_OF

// BEGIN SEARCH
template<class InputIt1, class InputIt2, class BinaryPred>
InputIt1 THRUST_HIP_FUNCTION
search(InputIt1 first, InputIt1 last,
              InputIt2 s_first, InputIt2 s_last,
              BinaryPred p)
{
    if (s_first == s_last)
    {
        return first;
    }

    thrust::device_system_tag dev_tag;
    size_t* d_output;
    d_output = thrust::malloc<size_t>(dev_tag, sizeof(*d_output)).get();

    size_t size = last - first;
    size_t s_size = s_last - s_first;

    // temp storage
    size_t temp_storage_size_bytes;
    void*  d_temp_storage = nullptr;
    // Get size of d_temp_storage
    hipError_t error = ::rocprim::search(d_temp_storage,
                                        temp_storage_size_bytes,
                                        first,
                                        s_first,
                                        d_output,
                                        size,
                                        s_size,
                                        p);

    if (error != hipSuccess)
    {
        return last;
    }

    d_temp_storage = thrust::malloc(dev_tag, temp_storage_size_bytes).get();

    error = ::rocprim::search(d_temp_storage,
                            temp_storage_size_bytes,
                            first,
                            s_first,
                            d_output,
                            size,
                            s_size,
                            p);

    if (error != hipSuccess)
    {
        thrust::free(dev_tag, d_temp_storage);
        thrust::free(dev_tag, d_output);
        return last;
    }

    error = hipDeviceSynchronize();
    if (error != hipSuccess)
    {
        thrust::free(dev_tag, d_temp_storage);
        thrust::free(dev_tag, d_output);
        return last;
    }

    size_t offset;
    hipMemcpy(&offset, d_output, sizeof(offset), hipMemcpyDeviceToHost);

    thrust::free(dev_tag, d_temp_storage);
    thrust::free(dev_tag, d_output);

    return first + offset;
}
// END SEARCH

// BEGIN SEARCH_N
template <class InputIt, class BinaryPred>
InputIt THRUST_HIP_FUNCTION search_n(
  InputIt                                                   first,
  InputIt                                                   last,
  size_t                                                    count,
  typename std::iterator_traits<InputIt>::value_type const& value,
  BinaryPred                                                p)
{
  using input_type = typename std::iterator_traits<InputIt>::value_type;
  thrust::device_system_tag                        dev_tag;
  size_t*                                          d_output;
  input_type*                                      d_value;
  d_output = thrust::malloc<size_t>(dev_tag, sizeof(*d_output)).get();
  d_value  = thrust::malloc<input_type>(dev_tag, sizeof(*d_value)).get();

  hipMemcpy(d_value, &value, sizeof(*d_value), hipMemcpyHostToDevice);

  size_t size = last - first;

  // temp storage
  size_t temp_storage_size_bytes;
  void*  d_temp_storage = nullptr;
  // Get size of d_temp_storage
  hipError_t error =
    ::rocprim::search_n(d_temp_storage, temp_storage_size_bytes, first, d_output, size, count, d_value, p);

  if (error != hipSuccess)
  {
    return last;
  }

  d_temp_storage = thrust::malloc(dev_tag, temp_storage_size_bytes).get();

  error = ::rocprim::search_n(d_temp_storage, temp_storage_size_bytes, first, d_output, size, count, d_value, p);

  if (error != hipSuccess)
  {
    thrust::free(dev_tag, d_temp_storage);
    thrust::free(dev_tag, d_output);
    thrust::free(dev_tag, d_value);
    return last;
  }

  error = hipDeviceSynchronize();
  if (error != hipSuccess)
  {
    thrust::free(dev_tag, d_temp_storage);
    thrust::free(dev_tag, d_output);
    thrust::free(dev_tag, d_value);
    return last;
  }

  size_t offset;
  hipMemcpy(&offset, d_output, sizeof(offset), hipMemcpyDeviceToHost);

  thrust::free(dev_tag, d_temp_storage);
  thrust::free(dev_tag, d_output);
  thrust::free(dev_tag, d_value);

  return first + offset;
}
// END SEARCH_N

// BEGIN FIND_END
template<class InputIt1, class InputIt2, class BinaryPred>
InputIt1 THRUST_HIP_FUNCTION
find_end(InputIt1 first, InputIt1 last,
              InputIt2 s_first, InputIt2 s_last,
              BinaryPred p)
{
    if (s_first == s_last)
    {
        return last;
    }

    thrust::device_system_tag dev_tag;
    size_t* d_output;
    d_output = thrust::malloc<size_t>(dev_tag, sizeof(*d_output)).get();

    size_t size = last - first;
    size_t s_size = s_last - s_first;

    // temp storage
    size_t temp_storage_size_bytes;
    void*  d_temp_storage = nullptr;
    // Get size of d_temp_storage
    hipError_t error = ::rocprim::find_end(d_temp_storage,
                                        temp_storage_size_bytes,
                                        first,
                                        s_first,
                                        d_output,
                                        size,
                                        s_size,
                                        p);

    if (error != hipSuccess)
    {
        return last;
    }

    d_temp_storage = thrust::malloc(dev_tag, temp_storage_size_bytes).get();

    error = ::rocprim::find_end(d_temp_storage,
                            temp_storage_size_bytes,
                            first,
                            s_first,
                            d_output,
                            size,
                            s_size,
                            p);

    if (error != hipSuccess)
    {
        thrust::free(dev_tag, d_temp_storage);
        thrust::free(dev_tag, d_output);
        return last;
    }

    error = hipDeviceSynchronize();
    if (error != hipSuccess)
    {
        thrust::free(dev_tag, d_temp_storage);
        thrust::free(dev_tag, d_output);
        return last;
    }

    size_t offset;
    hipMemcpy(&offset, d_output, sizeof(offset), hipMemcpyDeviceToHost);

    thrust::free(dev_tag, d_temp_storage);
    thrust::free(dev_tag, d_output);

    return first + offset;
}
// END FIND_END

// BEGIN ADJACENT_FIND
template <class InputIt, class BinaryPred>
InputIt THRUST_HIP_FUNCTION adjacent_find(InputIt first, InputIt last, BinaryPred p)
{
  if (first == last)
  {
    return last;
  }

  thrust::device_system_tag dev_tag;
  size_t*                   d_output;
  d_output = thrust::malloc<size_t>(dev_tag, sizeof(*d_output)).get();

  size_t size   = last - first;

  // temp storage
  size_t temp_storage_size_bytes;
  void*  d_temp_storage = nullptr;
  // Get size of d_temp_storage
  hipError_t error =
    ::rocprim::adjacent_find(d_temp_storage, temp_storage_size_bytes, first, d_output, size, p);

  if (error != hipSuccess)
  {
    return last;
  }

  d_temp_storage = thrust::malloc(dev_tag, temp_storage_size_bytes).get();

  error = ::rocprim::adjacent_find(d_temp_storage, temp_storage_size_bytes, first, d_output, size, p);

  if (error != hipSuccess)
  {
    thrust::free(dev_tag, d_temp_storage);
    thrust::free(dev_tag, d_output);
    return last;
  }

  error = hipDeviceSynchronize();
  if (error != hipSuccess)
  {
    thrust::free(dev_tag, d_temp_storage);
    thrust::free(dev_tag, d_output);
    return last;
  }

  size_t offset;
  hipMemcpy(&offset, d_output, sizeof(offset), hipMemcpyDeviceToHost);

  thrust::free(dev_tag, d_temp_storage);
  thrust::free(dev_tag, d_output);

  return first + offset;
}
// END ADJACENT_FIND
}

namespace std
{
    // BEGIN ALL_OF
    template<
        typename I,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    bool all_of(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        return ::thrust::all_of(::thrust::device, f, l, ::std::move(p));
    }

    template<
        typename I,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    bool all_of(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::all_of(::std::execution::par, f, l, ::std::move(p));
    }
    // END ALL_OF

    // BEGIN ANY_OF
    template<
        typename I,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    bool any_of(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        return ::thrust::any_of(::thrust::device, f, l, ::std::move(p));
    }

    template<
        typename I,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    bool any_of(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::any_of(::std::execution::par, f, l, ::std::move(p));
    }
    // END ANY_OF

    // BEGIN NONE_OF
    template<
        typename I,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    bool none_of(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        return ::thrust::none_of(::thrust::device, f, l, ::std::move(p));
    }

    template<
        typename I,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    bool none_of(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::none_of(::std::execution::par, f, l, ::std::move(p));
    }
    // END NONE_OF

    // BEGIN FIND
    template<
        typename I,
        typename T,
        enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    I find(execution::parallel_unsequenced_policy, I f, I l, const T& x)
    {
        return ::thrust::find(::thrust::device, f, l, x);
    }

    template<
        typename I,
        typename T,
        enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    I find(execution::parallel_unsequenced_policy, I f, I l, const T& x)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category>();

        return ::std::find(::std::execution::par, f, l, x);
    }
    // END FIND

    // BEGIN FIND_IF
    template<
        typename I,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    I find_if(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        return ::thrust::find_if(::thrust::device, f, l, ::std::move(p));
    }

    template<
        typename I,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    I find_if(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::find_if(::std::execution::par, f, l, ::std::move(p));
    }
    // END FIND_IF

    // BEGIN FIND_IF_NOT
    template<
        typename I,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    I find_if_not(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        return
            ::thrust::find_if_not(::thrust::device, f, l, ::std::move(p));
    }

    template<
        typename I,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    I find_if_not(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return
            ::std::find_if_not(::std::execution::par, f, l, ::std::move(p));
    }
    // END FIND_IF_NOT

    // BEGIN FIND_END
        template< class ForwardIt1,
              class ForwardIt2,
              enable_if_t<
                ::hipstd::is_offloadable_iterator<ForwardIt1>() &&
                ::hipstd::is_offloadable_iterator<ForwardIt2>()>* = nullptr>
        inline
        ForwardIt1 find_end(execution::parallel_unsequenced_policy, 
                                 ForwardIt1 first, ForwardIt1 last,
                                 ForwardIt2 s_first, ForwardIt2 s_last)
    {
        return ::thrust::find_end(first, last, s_first, s_last, thrust::equal_to<> {});
    }

    template< class ForwardIt1,
              class ForwardIt2,
              enable_if_t<
                !::hipstd::is_offloadable_iterator<ForwardIt1>() ||
                !::hipstd::is_offloadable_iterator<ForwardIt2>()>* = nullptr>
        inline
        ForwardIt1 find_end(execution::parallel_unsequenced_policy, 
                                 ForwardIt1 first, ForwardIt1 last,
                                 ForwardIt2 s_first, ForwardIt2 s_last )
    {
        return ::std::find_end(::std::execution::par, first, last, s_first, s_last);
    }

    template< class ForwardIt1,
              class ForwardIt2,
              class BinaryPred,
              enable_if_t<
                ::hipstd::is_offloadable_iterator<ForwardIt1>() &&
                ::hipstd::is_offloadable_iterator<ForwardIt2>() &&
                ::hipstd::is_offloadable_callable<BinaryPred>()>* = nullptr>
        inline
        ForwardIt1 find_end(execution::parallel_unsequenced_policy, 
                                 ForwardIt1 first, ForwardIt1 last,
                                 ForwardIt2 s_first, ForwardIt2 s_last, BinaryPred p)
    {
        return ::thrust::find_end(first, last, s_first, s_last, p);
    }

    template< class ForwardIt1,
              class ForwardIt2,
              class BinaryPred,
              enable_if_t<
                !::hipstd::is_offloadable_iterator<ForwardIt1>() ||
                !::hipstd::is_offloadable_iterator<ForwardIt2>() ||
                !::hipstd::is_offloadable_callable<BinaryPred>()>* = nullptr>
        inline
        ForwardIt1 find_end(execution::parallel_unsequenced_policy, 
                                 ForwardIt1 first, ForwardIt1 last,
                                 ForwardIt2 s_first, ForwardIt2 s_last, BinaryPred p)
    {
        return ::std::find_end(::std::execution::par, first, last, s_first, s_last, p);
    }
    // END FIND_END

    // BEGIN FIND_FIRST_OF
    template< class ForwardIt1,
              class ForwardIt2,
              enable_if_t<
                ::hipstd::is_offloadable_iterator<ForwardIt1>() &&
                ::hipstd::is_offloadable_iterator<ForwardIt2>()>* = nullptr>
        inline
        ForwardIt1 find_first_of(execution::parallel_unsequenced_policy, 
                                 ForwardIt1 first, ForwardIt1 last,
                                 ForwardIt2 s_first, ForwardIt2 s_last)
    {
        return ::thrust::find_first_of(first, last, s_first, s_last, thrust::equal_to<> {});
    }

    template< class ForwardIt1,
              class ForwardIt2,
              enable_if_t<
                !::hipstd::is_offloadable_iterator<ForwardIt1>() ||
                !::hipstd::is_offloadable_iterator<ForwardIt2>()>* = nullptr>
        inline
        ForwardIt1 find_first_of(execution::parallel_unsequenced_policy, 
                                 ForwardIt1 first, ForwardIt1 last,
                                 ForwardIt2 s_first, ForwardIt2 s_last )
    {
        return ::std::find_first_of(::std::execution::par, first, last, s_first, s_last);
    }

    template< class ForwardIt1,
              class ForwardIt2,
              class BinaryPred,
              enable_if_t<
                ::hipstd::is_offloadable_iterator<ForwardIt1>() &&
                ::hipstd::is_offloadable_iterator<ForwardIt2>() &&
                ::hipstd::is_offloadable_callable<BinaryPred>()>* = nullptr>
        inline
        ForwardIt1 find_first_of(execution::parallel_unsequenced_policy, 
                                 ForwardIt1 first, ForwardIt1 last,
                                 ForwardIt2 s_first, ForwardIt2 s_last, BinaryPred p)
    {
        return ::thrust::find_first_of(first, last, s_first, s_last, p);
    }

    template< class ForwardIt1,
              class ForwardIt2,
              class BinaryPred,
              enable_if_t<
                !::hipstd::is_offloadable_iterator<ForwardIt1>() ||
                !::hipstd::is_offloadable_iterator<ForwardIt2>() ||
                !::hipstd::is_offloadable_callable<BinaryPred>()>* = nullptr>
        inline
        ForwardIt1 find_first_of(execution::parallel_unsequenced_policy, 
                                 ForwardIt1 first, ForwardIt1 last,
                                 ForwardIt2 s_first, ForwardIt2 s_last, BinaryPred p)
    {
        return ::std::find_first_of(::std::execution::par, first, last, s_first, s_last, p);
    }
    // END FIND_FIRST_OF

    // BEGIN ADJACENT_FIND
    template<
        typename I,
        enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    I adjacent_find(execution::parallel_unsequenced_policy, I f, I l)
    {
      return ::thrust::adjacent_find(f, l, thrust::equal_to<> {});
    }

    template<
        typename I,
        typename P,
        enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    I adjacent_find(execution::parallel_unsequenced_policy, I f, I l)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category>();

        return ::std::adjacent_find(::std::execution::par, f, l);
    }

    template<
        typename I,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    I adjacent_find(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
      return ::thrust::adjacent_find(f, l, p);
    }

    template<
        typename I,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    I adjacent_find(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::adjacent_find(
            ::std::execution::par, f, l, ::std::move(p));
    }
    // END ADJACENT_FIND

    // BEGIN COUNT
    template<
        typename I,
        typename T,
        enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    typename iterator_traits<I>::difference_type count(
        execution::parallel_unsequenced_policy, I f, I l, const T& x)
    {
        return ::thrust::count(::thrust::device, f, l, x);
    }

    template<
        typename I,
        typename T,
        enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    typename iterator_traits<I>::difference_type count(
        execution::parallel_unsequenced_policy, I f, I l, const T& x)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category>();

        return ::std::count(::std::execution::par, f, l, x);
    }
    // END COUNT

    // BEGIN COUNT_IF
    template<
        typename I,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    typename iterator_traits<I>::difference_type count_if(
        execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        return ::thrust::count_if(::thrust::device, f, l, ::std::move(p));
    }

        template<
        typename I,
        typename O,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    typename iterator_traits<I>::difference_type count_if(
        execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::count_if(::std::execution::par, f, l, ::std::move(p));
    }
    // END COUNT_IF

    // BEGIN MISMATCH
    template<
        typename I0,
        typename I1,
        enable_if_t<::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    pair<I0, I1> mismatch(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1)
    {
        auto [m0, m1] = ::thrust::mismatch(::thrust::device, f0, l0, f1);

        return {::std::move(m0), ::std::move(m1)};
    }

    template<
        typename I0,
        typename I1,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    pair<I0, I1> mismatch(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I0>::iterator_category,
            typename iterator_traits<I1>::iterator_category>();

        return ::std::mismatch(::std::execution::par, f0, l0, f1);
    }

    template<
        typename I0,
        typename I1,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I0, I1>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    pair<I0, I1> mismatch(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, P p)
    {
        auto [m0, m1] = ::thrust::mismatch(
            ::thrust::device, f0, l0, f1, ::std::move(p));

        return {::std::move(m0), ::std::move(m1)};
    }

    template<
        typename I0,
        typename I1,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    pair<I0, I1> mismatch(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I0, I1>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I0>::iterator_category,
                typename iterator_traits<I1>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::mismatch(
            ::std::execution::par, f0, l0, f1, ::std::move(p));
    }

    template<
        typename I0,
        typename I1,
        enable_if_t<::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    pair<I0, I1> mismatch(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, I1 l1)
    {
        const auto n = ::std::min(l0 - f0, l1 - f1);

        auto [m0, m1] =
            ::thrust::mismatch(::thrust::device, f0, f0 + n, f1);

        return {::std::move(m0), ::std::move(m1)};
    }

    template<
        typename I0,
        typename I1,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    pair<I0, I1> mismatch(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, I1 l1)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I0>::iterator_category,
            typename iterator_traits<I1>::iterator_category>();

        return ::std::mismatch(::std::execution::par, f0, l0, f1, l1);
    }

    template<
        typename I0,
        typename I1,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I0, I1>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    pair<I0, I1> mismatch(
        execution::parallel_unsequenced_policy,
        I0 f0,
        I0 l0,
        I1 f1,
        I1 l1,
        P p)
    {
        const auto n = ::std::min(l0 - f0, l1 - f1);

        auto [m0, m1] = ::thrust::mismatch(
            ::thrust::device, f0, f0 + n, f1, ::std::move(p));

        return {::std::move(m0), ::std::move(m1)};
    }

    template<
        typename I0,
        typename I1,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    pair<I0, I1> mismatch(
        execution::parallel_unsequenced_policy,
        I0 f0,
        I0 l0,
        I1 f1,
        I1 l1,
        P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I0, I1>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I0>::iterator_category,
                typename iterator_traits<I1>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::mismatch(
            ::std::execution::par, f0, l0, f1, l1, ::std::move(p));
    }
    // END MISMATCH

    // BEGIN EQUAL
    template<
        typename I0,
        typename I1,
        enable_if_t<::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    bool equal(execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1)
    {
        return ::thrust::equal(::thrust::device, f0, l0, f1);
    }

    template<
        typename I0,
        typename I1,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    bool equal(execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I0>::iterator_category,
            typename iterator_traits<I1>::iterator_category>();

        return ::std::equal(::std::execution::par, f0, l0, f1);
    }

    template<
        typename I0,
        typename I1,
        typename R,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I0, I1>() &&
            ::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    bool equal(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, R r)
    {
        return
            ::thrust::equal(::thrust::device, f0, l0, f1, ::std::move(r));
    }

    template<
        typename I0,
        typename I1,
        typename R,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>() ||
            !::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    bool equal(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, R r)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I0, I1>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I0>::iterator_category,
                typename iterator_traits<I1>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<R>()) {
            ::hipstd::unsupported_callable_type<R>();
        }
        return
            ::std::equal(::std::execution::par, f0, l0, f1, ::std::move(r));
    }

    template<
        typename I0,
        typename I1,
        enable_if_t<::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    bool equal(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, I1 l1)
    {
        if (l0 - f0 != l1 - f1) return false;

        return ::thrust::equal(::thrust::device, f0, l0, f1);
    }

    template<
        typename I0,
        typename I1,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    bool equal(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, I1 l1)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I0>::iterator_category,
            typename iterator_traits<I1>::iterator_category>();

        return ::std::equal(::std::execution::par, f0, l0, f1, l1);
    }

    template<
        typename I0,
        typename I1,
        typename R,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I0, I1>() &&
            ::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    bool equal(
        execution::parallel_unsequenced_policy,
        I0 f0,
        I0 l0,
        I1 f1,
        I1 l1,
        R r)
    {
        if (l0 - f0 != l1 - f1) return false;

        return ::thrust::equal(
            ::thrust::device, f0, l0, f1, ::std::move(r));
    }

    template<
        typename I0,
        typename I1,
        typename R,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>() ||
            !::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    bool equal(
        execution::parallel_unsequenced_policy,
        I0 f0,
        I0 l0,
        I1 f1,
        I1 l1,
        R r)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I0, I1>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I0>::iterator_category,
                typename iterator_traits<I1>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<R>()) {
            ::hipstd::unsupported_callable_type<R>();
        }
        return ::std::equal(
            ::std::execution::par, f0, l0, f1, l1, ::std::move(r));
    }
    // END EQUAL

    // BEGIN SEARCH
    template< class ForwardIt1,
              class ForwardIt2,
              enable_if_t<
                ::hipstd::is_offloadable_iterator<ForwardIt1>() &&
                ::hipstd::is_offloadable_iterator<ForwardIt2>()>* = nullptr>
        inline
        ForwardIt1 search(execution::parallel_unsequenced_policy, 
                                 ForwardIt1 first, ForwardIt1 last,
                                 ForwardIt2 s_first, ForwardIt2 s_last)
    {
        return ::thrust::search(first, last, s_first, s_last, thrust::equal_to<> {});
    }

    template< class ForwardIt1,
              class ForwardIt2,
              enable_if_t<
                !::hipstd::is_offloadable_iterator<ForwardIt1>() ||
                !::hipstd::is_offloadable_iterator<ForwardIt2>()>* = nullptr>
        inline
        ForwardIt1 search(execution::parallel_unsequenced_policy, 
                                 ForwardIt1 first, ForwardIt1 last,
                                 ForwardIt2 s_first, ForwardIt2 s_last )
    {
        return ::std::search(::std::execution::par, first, last, s_first, s_last);
    }

    template< class ForwardIt1,
              class ForwardIt2,
              class BinaryPred,
              enable_if_t<
                ::hipstd::is_offloadable_iterator<ForwardIt1>() &&
                ::hipstd::is_offloadable_iterator<ForwardIt2>() &&
                ::hipstd::is_offloadable_callable<BinaryPred>()>* = nullptr>
        inline
        ForwardIt1 search(execution::parallel_unsequenced_policy, 
                                 ForwardIt1 first, ForwardIt1 last,
                                 ForwardIt2 s_first, ForwardIt2 s_last, BinaryPred p)
    {
        return ::thrust::search(first, last, s_first, s_last, p);
    }

    template< class ForwardIt1,
              class ForwardIt2,
              class BinaryPred,
              enable_if_t<
                !::hipstd::is_offloadable_iterator<ForwardIt1>() ||
                !::hipstd::is_offloadable_iterator<ForwardIt2>() ||
                !::hipstd::is_offloadable_callable<BinaryPred>()>* = nullptr>
        inline
        ForwardIt1 search(execution::parallel_unsequenced_policy, 
                                 ForwardIt1 first, ForwardIt1 last,
                                 ForwardIt2 s_first, ForwardIt2 s_last, BinaryPred p)
    {
        return ::std::search(::std::execution::par, first, last, s_first, s_last, p);
    }
    // END SEARCH

    // BEGIN SEARCH_N
    template <class ForwardIt, enable_if_t<::hipstd::is_offloadable_iterator<ForwardIt>()>* = nullptr>
    inline ForwardIt search_n(
      execution::parallel_unsequenced_policy,
      ForwardIt                                                   first,
      ForwardIt                                                   last,
      size_t                                                      count,
      typename std::iterator_traits<ForwardIt>::value_type const& value)
    {
      return ::thrust::search_n(first, last, count, value, thrust::equal_to<>{});
    }

    template <class ForwardIt, enable_if_t<!::hipstd::is_offloadable_iterator<ForwardIt>()>* = nullptr>
    inline ForwardIt search_n(
      execution::parallel_unsequenced_policy,
      ForwardIt                                                   first,
      ForwardIt                                                   last,
      size_t                                                      count,
      typename std::iterator_traits<ForwardIt>::value_type const& value)
    {
      return ::std::search_n(::std::execution::par, first, last, count, value);
    }

    template <class ForwardIt,
              class BinaryPred,
              enable_if_t<::hipstd::is_offloadable_iterator<ForwardIt>()
                          && ::hipstd::is_offloadable_callable<BinaryPred>()>* = nullptr>
    inline ForwardIt search_n(
      execution::parallel_unsequenced_policy,
      ForwardIt                                                   first,
      ForwardIt                                                   last,
      size_t                                                      count,
      typename std::iterator_traits<ForwardIt>::value_type const& value,
      BinaryPred                                                  p)
    {
      return ::thrust::search_n(first, last, count, value, p);
    }

    template <class ForwardIt,
              class BinaryPred,
              enable_if_t<!::hipstd::is_offloadable_iterator<ForwardIt>()
                          || !::hipstd::is_offloadable_callable<BinaryPred>()>* = nullptr>
    inline ForwardIt search_n(
      execution::parallel_unsequenced_policy,
      ForwardIt                                                   first,
      ForwardIt                                                   last,
      size_t                                                      count,
      typename std::iterator_traits<ForwardIt>::value_type const& value,
      BinaryPred                                                  p)
    {
      return ::std::search_n(::std::execution::par, first, last, count, value, p);
    }
    // END SEARCH_N
}
#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
