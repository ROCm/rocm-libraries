/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
#include <thrust/detail/config.h>

#include <thrust/detail/cstdint.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/distance.h>
#include <thrust/pair.h>
#include <thrust/partition.h>
#include <thrust/system/hip/detail/find.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/detail/reverse.h>
#include <thrust/system/hip/detail/uninitialized_copy.h>
#include <thrust/system/hip/detail/util.h>

#include <thrust/detail/alignment.h>


// rocprim include
#include <rocprim/rocprim.hpp>

THRUST_BEGIN_NS
namespace hip_rocprim
{
namespace __partition
{
    template <class SINGLE_OUTPUT,
              class SelectedOutIt,
              class RejectedOutIt,
              class Size,
              class Value>
    struct partition_fill;

    template <class SelectedOutIt, class RejectedOutIt, class Size, class Value>
    struct partition_fill<detail::true_type, SelectedOutIt, RejectedOutIt, Size, Value>
    {
        SelectedOutIt selected_out_first;
        RejectedOutIt rejected_out_first;

        Value* ptr;
        Size   num_selected;
        Size   num_items;

        THRUST_HIP_FUNCTION
        partition_fill(SelectedOutIt selected_out_first_,
                       RejectedOutIt rejected_out_first_,
                       Value*        ptr_,
                       Size          num_selected_,
                       Size          num_items_)
            : selected_out_first(selected_out_first_)
            , rejected_out_first(rejected_out_first_)
            , ptr(ptr_)
            , num_selected(num_selected_)
            , num_items(num_items_)
        {
        }

        void THRUST_HIP_DEVICE_FUNCTION operator()(Size idx)
        {
            selected_out_first[idx] = ptr[idx];
        }
    };

    template <class SelectedOutIt, class RejectedOutIt, class Size, class Value>
    struct partition_fill<detail::false_type, SelectedOutIt, RejectedOutIt, Size, Value>
    {
        SelectedOutIt selected_out_first;
        RejectedOutIt rejected_out_first;

        Value* ptr;
        Size   num_selected;
        Size   num_items;

        THRUST_HIP_FUNCTION
        partition_fill(SelectedOutIt selected_out_first_,
                       RejectedOutIt rejected_out_first_,
                       Value*        ptr_,
                       Size          num_selected_,
                       Size          num_items_)
            : selected_out_first(selected_out_first_)
            , rejected_out_first(rejected_out_first_)
            , ptr(ptr_)
            , num_selected(num_selected_)
            , num_items(num_items_)
        {
        }

        void THRUST_HIP_DEVICE_FUNCTION operator()(Size idx)
        {
            if(num_selected > idx)
            {
                selected_out_first[idx] = ptr[idx];
            }
            else
            {
                rejected_out_first[num_items - idx - 1] = ptr[idx];
            }
        }
    };

    template <class SINGLE_OUTPUT>
    struct rejected_last;

    template <>
    struct rejected_last<detail::true_type>
    {
        template <class RejectedOutIt, class Size>
        static RejectedOutIt THRUST_HIP_RUNTIME_FUNCTION
        rejected(RejectedOutIt rejected_result, Size)
        {
            return rejected_result;
        }
    };

    template <>
    struct rejected_last<detail::false_type>
    {
        template <class RejectedOutIt, class Size>
        static RejectedOutIt THRUST_HIP_RUNTIME_FUNCTION
        rejected(RejectedOutIt rejected_result, Size num_unselected)
        {
            return rejected_result + num_unselected;
        }
    };

    template <class SINGLE_OUTPUT,
              class Derived,
              class InputIt,
              class SelectedOutIt,
              class RejectedOutIt,
              class Predicate>
    THRUST_HIP_RUNTIME_FUNCTION
    pair<SelectedOutIt, RejectedOutIt>
    partition(execution_policy<Derived>& policy,
              InputIt                    first,
              InputIt                    last,
              SelectedOutIt              selected_result,
              RejectedOutIt              rejected_result,
              Predicate                  predicate)
    {
        typedef typename iterator_traits<InputIt>::difference_type size_type;
        typedef typename iterator_traits<InputIt>::value_type      value_type;

        size_type   num_items          = static_cast<size_type>(thrust::distance(first, last));
        size_t      temp_storage_bytes = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::partition(NULL,
                                                       temp_storage_bytes,
                                                       first,
                                                       reinterpret_cast<value_type*>(NULL),
                                                       reinterpret_cast<size_type*>(NULL),
                                                       num_items,
                                                       predicate,
                                                       stream,
                                                       debug_sync),
                                    "partition failed on 1st step");

        size_t storage_size = sizeof(size_type) + temp_storage_bytes + sizeof(value_type) * num_items;

        // Allocate temporary storage.
        thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
            tmp(policy, storage_size);
        void *ptr = static_cast<void*>(tmp.data().get());

        size_type* d_num_selected_out = reinterpret_cast<size_type*>(
            reinterpret_cast<char*>(ptr) + temp_storage_bytes);

        value_type* d_partition_out = reinterpret_cast<value_type*>(
            reinterpret_cast<char*>(d_num_selected_out) + sizeof(size_type));

        hip_rocprim::throw_on_error(rocprim::partition(ptr,
                                                       temp_storage_bytes,
                                                       first,
                                                       d_partition_out,
                                                       d_num_selected_out,
                                                       num_items,
                                                       predicate,
                                                       stream,
                                                       debug_sync),
                                    "partition failed on 2nd step");

        size_type num_selected = 0;
        if(num_items > 0)
        {
            num_selected = get_value(policy, d_num_selected_out);
        }

        // fill the values
        hip_rocprim::parallel_for(
            policy,
            partition_fill<SINGLE_OUTPUT, SelectedOutIt, RejectedOutIt, size_type, value_type>(
                selected_result, rejected_result, d_partition_out, num_selected, num_items),
            num_items);


        return thrust::make_pair(
            selected_result + num_selected,
            rejected_last<SINGLE_OUTPUT>::rejected(rejected_result, num_items - num_selected));
    }

    template <class SINGLE_OUTPUT,
              class Derived,
              class InputIt,
              class StencilIt,
              class SelectedOutIt,
              class RejectedOutIt,
              class Predicate>
    pair<SelectedOutIt, RejectedOutIt> THRUST_HIP_RUNTIME_FUNCTION
    partition(execution_policy<Derived>& policy,
              InputIt                    first,
              InputIt                    last,
              StencilIt                  stencil,
              SelectedOutIt              selected_result,
              RejectedOutIt              rejected_result,
              Predicate                  predicate)
    {
        typedef typename iterator_traits<InputIt>::difference_type size_type;
        typedef typename iterator_traits<InputIt>::value_type      value_type;

        size_type   num_items          = static_cast<size_type>(thrust::distance(first, last));
        size_t      temp_storage_bytes = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::partition(NULL,
                                                       temp_storage_bytes,
                                                       first,
                                                       reinterpret_cast<bool*>(NULL),
                                                       reinterpret_cast<value_type*>(NULL),
                                                       reinterpret_cast<size_type*>(NULL),
                                                       num_items,
                                                       stream,
                                                       debug_sync),
                                    "partition failed on 1st step");

        size_t storage_size = sizeof(size_type) + temp_storage_bytes + sizeof(value_type) * num_items + sizeof(bool) * num_items;

        // Allocate temporary storage.
        thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
            tmp(policy, storage_size);
        void *ptr = static_cast<void*>(tmp.data().get());

        size_type* d_num_selected_out = reinterpret_cast<size_type*>(
            reinterpret_cast<char*>(ptr) + temp_storage_bytes);

        value_type* d_partition_out = reinterpret_cast<value_type*>(
            reinterpret_cast<char*>(d_num_selected_out) + sizeof(size_type));

        bool* d_flags = reinterpret_cast<bool*>(reinterpret_cast<char*>(d_partition_out)
                                          + sizeof(value_type) * num_items);

        hip_rocprim::transform(policy, stencil, stencil + num_items, d_flags, predicate);

        hip_rocprim::throw_on_error(rocprim::partition(ptr,
                                                       temp_storage_bytes,
                                                       first,
                                                       d_flags,
                                                       d_partition_out,
                                                       d_num_selected_out,
                                                       num_items,
                                                       stream,
                                                       debug_sync),
                                    "partition failed on 2nd step");

        size_type num_selected = 0;
        if(num_items > 0)
        {
            num_selected = get_value(policy, d_num_selected_out);
        }

        hip_rocprim::parallel_for(
            policy,
            partition_fill<SINGLE_OUTPUT, SelectedOutIt, RejectedOutIt, size_type, value_type>(
                selected_result, rejected_result, d_partition_out, num_selected, num_items),
            num_items);

        return thrust::make_pair(
            selected_result + num_selected,
            rejected_last<SINGLE_OUTPUT>::rejected(rejected_result, num_items - num_selected));
    }

    template <typename Derived, typename Iterator, typename Predicate>
    Iterator THRUST_HIP_RUNTIME_FUNCTION partition_inplace(execution_policy<Derived>& policy,
                                                           Iterator                   first,
                                                           Iterator                   last,
                                                           Predicate                  predicate)
    {
        typedef typename iterator_traits<Iterator>::difference_type size_type;
        typedef typename iterator_traits<Iterator>::value_type      value_type;

        size_type num_items = thrust::distance(first, last);

        // Allocate temporary storage.
        thrust::detail::temporary_array<value_type, Derived> tmp(policy, num_items);

        hip_rocprim::uninitialized_copy(policy, first, last,  tmp.begin());

        pair<Iterator, Iterator> result = partition<detail::true_type>(
            policy,  tmp.begin(),  tmp.begin() + num_items, first, first, predicate);

        size_type num_selected = result.first - first;

        return first + num_selected;
    }

    template <typename Derived, typename Iterator, typename StencilIt, typename Predicate>
    Iterator THRUST_HIP_RUNTIME_FUNCTION
    partition_inplace(execution_policy<Derived>& policy,
                      Iterator                   first,
                      Iterator                   last,
                      StencilIt                  stencil,
                      Predicate                  predicate)
    {
        typedef typename iterator_traits<Iterator>::difference_type size_type;
        typedef typename iterator_traits<Iterator>::value_type      value_type;

        size_type num_items = thrust::distance(first, last);

        // Allocate temporary storage.
        thrust::detail::temporary_array<value_type, Derived> tmp(policy, num_items);

        hip_rocprim::uninitialized_copy(policy, first, last, tmp.begin());

        pair<Iterator, Iterator> result = partition<detail::true_type>(
            policy, tmp.begin(), tmp.begin() + num_items, stencil, first, first, predicate);

        size_type num_selected = result.first - first;

        return first + num_selected;
    }
} // namespace __partition

//-------------------------
// Thrust API entry points
//-------------------------

__thrust_exec_check_disable__ template <class Derived,
                                        class InputIt,
                                        class StencilIt,
                                        class SelectedOutIt,
                                        class RejectedOutIt,
                                        class Predicate>
pair<SelectedOutIt, RejectedOutIt> THRUST_HIP_FUNCTION
partition_copy(execution_policy<Derived>& policy,
               InputIt                    first,
               InputIt                    last,
               StencilIt                  stencil,
               SelectedOutIt              selected_result,
               RejectedOutIt              rejected_result,
               Predicate                  predicate)
{
  struct workaround
  {
      __host__
      static pair<SelectedOutIt, RejectedOutIt> par(
        execution_policy<Derived>& policy,
        InputIt                    first,
        InputIt                    last,
        StencilIt                  stencil,
        SelectedOutIt              selected_result,
        RejectedOutIt              rejected_result,
        Predicate                  predicate)
      {
      #if __HCC__ && __HIP_DEVICE_COMPILE__
      THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
          (__partition::partition<detail::false_type,
                                  Derived,
                                  InputIt,
                                  StencilIt,
                                  SelectedOutIt,
                                  RejectedOutIt,
                                  Predicate>)
      );
      #else

      return __partition::partition<detail::false_type>(
          policy, first, last, stencil, selected_result, rejected_result, predicate
        );
      #endif
      }
      __device__
      static pair<SelectedOutIt, RejectedOutIt> seq(
        execution_policy<Derived>& policy,
        InputIt                    first,
        InputIt                    last,
        StencilIt                  stencil,
        SelectedOutIt              selected_result,
        RejectedOutIt              rejected_result,
        Predicate                  predicate)
      {
        return thrust::partition_copy(
          cvt_to_seq(derived_cast(policy)),
          first,
          last,
          stencil,
          selected_result,
          rejected_result,
          predicate
        );
      }
  };
  #if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, stencil, selected_result, rejected_result, predicate);
  #else
  return workaround::seq(policy, first, last, stencil, selected_result, rejected_result, predicate);
  #endif
}

__thrust_exec_check_disable__ template <class Derived,
                                        class InputIt,
                                        class SelectedOutIt,
                                        class RejectedOutIt,
                                        class Predicate>
pair<SelectedOutIt, RejectedOutIt> THRUST_HIP_FUNCTION
partition_copy(execution_policy<Derived>& policy,
               InputIt                    first,
               InputIt                    last,
               SelectedOutIt              selected_result,
               RejectedOutIt              rejected_result,
               Predicate                  predicate)
{
    struct workaround
    {
        __host__
        static pair<SelectedOutIt, RejectedOutIt> par(
          execution_policy<Derived>& policy,
          InputIt                    first,
          InputIt                    last,
          SelectedOutIt              selected_result,
          RejectedOutIt              rejected_result,
          Predicate                  predicate)
        {
        #if __HCC__ && __HIP_DEVICE_COMPILE__
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
          (__partition::partition<detail::false_type,
                                          Derived,
                                          InputIt,
                                          SelectedOutIt,
                                          RejectedOutIt,
                                          Predicate>
        );
        #else
        return __partition::partition<detail::false_type>(
          policy, first, last, selected_result, rejected_result, predicate
        );
        #endif
        }
        __device__
        static pair<SelectedOutIt, RejectedOutIt> seq(
          execution_policy<Derived>& policy,
          InputIt                    first,
          InputIt                    last,
          SelectedOutIt              selected_result,
          RejectedOutIt              rejected_result,
          Predicate                  predicate)
        {
          return thrust::partition_copy(cvt_to_seq(derived_cast(policy)),
                                        first,
                                        last,
                                        selected_result,
                                        rejected_result,
                                        predicate);
        }
    };
    #if __THRUST_HAS_HIPRT__
    return workaround::par(policy, first, last, selected_result, rejected_result, predicate);
    #else
    return workaround::seq(policy, first, last, selected_result, rejected_result, predicate);
    #endif
}

__thrust_exec_check_disable__ template <class Derived,
                                        class InputIt,
                                        class SelectedOutIt,
                                        class RejectedOutIt,
                                        class Predicate>
pair<SelectedOutIt, RejectedOutIt> THRUST_HIP_FUNCTION
stable_partition_copy(execution_policy<Derived>& policy,
                      InputIt                    first,
                      InputIt                    last,
                      SelectedOutIt              selected_result,
                      RejectedOutIt              rejected_result,
                      Predicate                  predicate)
{
  struct workaround
  {
      __host__
      static pair<SelectedOutIt, RejectedOutIt> par(
        execution_policy<Derived>& policy,
        InputIt                    first,
        InputIt                    last,
        SelectedOutIt              selected_result,
        RejectedOutIt              rejected_result,
        Predicate                  predicate)
      {
        #if __HCC__ && __HIP_DEVICE_COMPILE__
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
            (__partition::partition<detail::false_type,
                                    Derived,
                                    InputIt,
                                    SelectedOutIt,
                                    RejectedOutIt,
                                    Predicate>)
        );
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((hip_rocprim::reverse<Derived, InputIt>));
        #else
        return __partition::partition<detail::false_type>(
            policy, first, last, selected_result, rejected_result, predicate);
        #endif
      }
      __device__
      static pair<SelectedOutIt, RejectedOutIt> seq(
        execution_policy<Derived>& policy,
        InputIt                    first,
        InputIt                    last,
        SelectedOutIt              selected_result,
        RejectedOutIt              rejected_result,
        Predicate                  predicate)
      {
        return thrust::stable_partition_copy(
          cvt_to_seq(derived_cast(policy)),
          first,
          last,
          selected_result,
          rejected_result,
          predicate
        );
      }
  };
  #if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, selected_result, rejected_result, predicate);
  #else
  return workaround::seq(policy, first, last, selected_result, rejected_result, predicate);
  #endif
}

__thrust_exec_check_disable__ template <class Derived,
                                        class InputIt,
                                        class StencilIt,
                                        class SelectedOutIt,
                                        class RejectedOutIt,
                                        class Predicate>
pair<SelectedOutIt, RejectedOutIt> THRUST_HIP_FUNCTION
stable_partition_copy(execution_policy<Derived>& policy,
                      InputIt                    first,
                      InputIt                    last,
                      StencilIt                  stencil,
                      SelectedOutIt              selected_result,
                      RejectedOutIt              rejected_result,
                      Predicate                  predicate)
{
  struct workaround
  {
      __host__
      static pair<SelectedOutIt, RejectedOutIt> par(
        execution_policy<Derived>& policy,
        InputIt                    first,
        InputIt                    last,
        StencilIt                  stencil,
        SelectedOutIt              selected_result,
        RejectedOutIt              rejected_result,
        Predicate                  predicate)
      {
        #if __HCC__ && __HIP_DEVICE_COMPILE__
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
            (__partition::partition<detail::false_type,
                                    Derived,
                                    InputIt,
                                    StencilIt,
                                    SelectedOutIt,
                                    RejectedOutIt,
                                    Predicate>)
        );
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((hip_rocprim::reverse<Derived, InputIt>));
        #else
        return __partition::partition<detail::false_type>(
            policy, first, last, stencil, selected_result, rejected_result, predicate);
        #endif
      }
      __device__
      static pair<SelectedOutIt, RejectedOutIt> seq(
        execution_policy<Derived>& policy,
        InputIt                    first,
        InputIt                    last,
        StencilIt                  stencil,
        SelectedOutIt              selected_result,
        RejectedOutIt              rejected_result,
        Predicate                  predicate)
      {
        return thrust::stable_partition_copy(
          cvt_to_seq(derived_cast(policy)),
          first,
          last,
          stencil,
          selected_result,
          rejected_result,
          predicate
        );
      }
  };
  #if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, stencil, selected_result, rejected_result, predicate);
  #else
  return workaround::seq(policy, first, last, stencil, selected_result, rejected_result, predicate);
  #endif
}

/// inplace
__thrust_exec_check_disable__ template <class Derived,
                                        class Iterator,
                                        class StencilIt,
                                        class Predicate>
Iterator THRUST_HIP_FUNCTION
partition(execution_policy<Derived>& policy,
          Iterator                   first,
          Iterator                   last,
          StencilIt                  stencil,
          Predicate                  predicate)
{
  struct workaround
  {
      __host__
      static Iterator par(execution_policy<Derived>& policy,
                          Iterator                   first,
                          Iterator                   last,
                          StencilIt                  stencil,
                          Predicate                  predicate)
      {
        #if __HCC__ && __HIP_DEVICE_COMPILE__
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
            (__partition::partition_inplace<Derived, Iterator, StencilIt, Predicate>)
        );
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
            (hip_rocprim::reverse<Derived, Iterator>)
        );
        #else
        Iterator result =  __partition::partition_inplace(
          policy,
          first,
          last,
          stencil,
          predicate
        );
        hip_rocprim::reverse<Derived,Iterator>(policy, result, last);
        return result;
        #endif
      }
      __device__
      static Iterator seq(execution_policy<Derived>& policy,
                          Iterator                   first,
                          Iterator                   last,
                          StencilIt                  stencil,
                          Predicate                  predicate)
      {
          return thrust::partition(
             cvt_to_seq(derived_cast(policy)),
             first,
             last,
             stencil,
             predicate
          );
      }
  };
  #if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, stencil, predicate);
  #else
  return workaround::seq(policy, first, last, stencil, predicate);
  #endif
}

__thrust_exec_check_disable__ template <class Derived, class Iterator, class Predicate>
Iterator THRUST_HIP_FUNCTION
partition(execution_policy<Derived>& policy,
          Iterator                   first,
          Iterator                   last,
          Predicate                  predicate)
{

  struct workaround
  {
      __host__
      static Iterator par(execution_policy<Derived>& policy,
                          Iterator                   first,
                          Iterator                   last,
                          Predicate                  predicate)
      {
      #if __HCC__ && __HIP_DEVICE_COMPILE__
      THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
          (__partition::partition<Derived, Iterator, Iterator, StencilIt, Predicate>)
      );
      #else
      return __partition::partition_inplace(
        policy,
        first,
        last,
        predicate
      );
      #endif
      }
      __device__
      static Iterator seq(execution_policy<Derived>& policy,
                          Iterator                   first,
                          Iterator                   last,
                          Predicate                  predicate)
      {
          return thrust::partition(
             cvt_to_seq(derived_cast(policy)),
             first,
             last,
             predicate
          );
      }
  };
  #if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, predicate);
  #else
  return workaround::seq(policy, first, last, predicate);
  #endif

}

__thrust_exec_check_disable__ template <class Derived,
                                        class Iterator,
                                        class StencilIt,
                                        class Predicate>
Iterator THRUST_HIP_FUNCTION
stable_partition(execution_policy<Derived>& policy,
                 Iterator                   first,
                 Iterator                   last,
                 StencilIt                  stencil,
                 Predicate                  predicate)
{
  struct workaround
  {
      __host__
      static Iterator par(execution_policy<Derived>& policy,
                          Iterator                   first,
                          Iterator                   last,
                          StencilIt                  stencil,
                          Predicate                  predicate)
      {
        #if __HCC__ && __HIP_DEVICE_COMPILE__
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
            (__partition::partition_inplace<Derived, Iterator, StencilIt, Predicate>)
        );
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
            (hip_rocprim::reverse<Derived, Iterator>)
        );
        #else
        Iterator result =  __partition::partition_inplace(
          policy,
          first,
          last,
          stencil,
          predicate
        );
        hip_rocprim::reverse<Derived,Iterator>(policy, result, last);
        return result;
        #endif
      }
      __device__
      static Iterator seq(execution_policy<Derived>& policy,
                          Iterator                   first,
                          Iterator                   last,
                          StencilIt                  stencil,
                          Predicate                  predicate)
      {
          return thrust::stable_partition(
             cvt_to_seq(derived_cast(policy)),
             first,
             last,
             stencil,
             predicate
          );
      }
  };
  #if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, stencil, predicate);
  #else
  return workaround::seq(policy, first, last, stencil, predicate);
  #endif
}

__thrust_exec_check_disable__ template <class Derived, class Iterator, class Predicate>
Iterator THRUST_HIP_FUNCTION
stable_partition(execution_policy<Derived>& policy,
                 Iterator                   first,
                 Iterator                   last,
                 Predicate                  predicate)
{
  struct workaround
  {
      __host__
      static Iterator par(execution_policy<Derived>& policy,
                          Iterator                   first,
                          Iterator                   last,
                          Predicate                  predicate)
      {
        #if __HCC__ && __HIP_DEVICE_COMPILE__
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
            (__partition::partition_inplace<Derived, Iterator, Predicate>)
        );
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
            (hip_rocprim::reverse<Derived, Iterator>)
        );
        #else
        Iterator result =  __partition::partition_inplace(
          policy,
          first,
          last,
          predicate
        );
        hip_rocprim::reverse<Derived,Iterator>(policy, result, last);
        return result;
        #endif
      }
      __device__
      static Iterator seq(execution_policy<Derived>& policy,
                          Iterator                   first,
                          Iterator                   last,
                          Predicate                  predicate)
      {
          return thrust::stable_partition(
             cvt_to_seq(derived_cast(policy)),
             first,
             last,
             predicate
          );
      }
  };
  #if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, predicate);
  #else
  return workaround::seq(policy, first, last, predicate);
  #endif
}

template <class Derived, class ItemsIt, class Predicate>
bool THRUST_HIP_FUNCTION
is_partitioned(execution_policy<Derived>& policy,
               ItemsIt                    first,
               ItemsIt                    last,
               Predicate                  predicate)
{
    ItemsIt boundary = hip_rocprim::find_if_not(policy, first, last, predicate);
    ItemsIt end      = hip_rocprim::find_if(policy, boundary, last, predicate);
    return end == last;
}

} // namespace hip_rocprim
THRUST_END_NS
#endif
