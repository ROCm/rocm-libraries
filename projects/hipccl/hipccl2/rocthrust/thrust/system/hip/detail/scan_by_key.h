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
#include <thrust/detail/cstdint.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/hip/detail/util.h>

#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/execution_policy.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/detail/minmax.h>
#include <thrust/distance.h>


// rocprim include
#include <rocprim/rocprim.hpp>

THRUST_BEGIN_NS
namespace hip_rocprim
{
namespace __scan_by_key
{
    template <typename Derived,
              typename KeysInputIterator,
              typename ValuesInputIterator,
              typename ValuesOutputIterator,
              typename KeyCompareFunction = ::rocprim::equal_to<
                  typename std::iterator_traits<KeysInputIterator>::value_type>,
              typename BinaryFunction
              = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>>
    THRUST_HIP_RUNTIME_FUNCTION ValuesOutputIterator
    inclusive_scan_by_key(execution_policy<Derived>& policy,
                          KeysInputIterator          key_first,
                          KeysInputIterator          key_last,
                          ValuesInputIterator        value_first,
                          ValuesOutputIterator       value_result,
                          KeyCompareFunction         key_compare_op,
                          BinaryFunction             scan_op)
    {
        size_t      num_items    = static_cast<size_t>(thrust::distance(key_first, key_last));
        size_t      storage_size = 0;
        hipStream_t stream       = hip_rocprim::stream(policy);
        bool        debug_sync   = THRUST_HIP_DEBUG_SYNC_FLAG;

        if(num_items == 0)
            return value_result;

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::inclusive_scan_by_key(NULL,
                                                                   storage_size,
                                                                   key_first,
                                                                   value_first,
                                                                   value_result,
                                                                   num_items,
                                                                   scan_op,
                                                                   key_compare_op,
                                                                   stream,
                                                                   debug_sync),
                                    "scan_by_key failed on 1st step");

        // Allocate temporary storage.
        thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
            tmp(policy, storage_size);
        void *ptr = static_cast<void*>(tmp.data().get());

        // Run scan.
        hip_rocprim::throw_on_error(rocprim::inclusive_scan_by_key(ptr,
                                                                   storage_size,
                                                                   key_first,
                                                                   value_first,
                                                                   value_result,
                                                                   num_items,
                                                                   scan_op,
                                                                   key_compare_op,
                                                                   stream,
                                                                   debug_sync),
                                    "scan_by_key failed on 2nd step");

        return value_result + num_items;
    }

    template <typename Derived,
              typename KeysInputIterator,
              typename ValuesInputIterator,
              typename ValuesOutputIterator,
              typename InitialValueType,
              typename KeyCompareFunction = ::rocprim::equal_to<
                  typename std::iterator_traits<KeysInputIterator>::value_type>,
              typename BinaryFunction
              = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>>
    THRUST_HIP_RUNTIME_FUNCTION ValuesOutputIterator
    exclusive_scan_by_key(execution_policy<Derived>& policy,
                          KeysInputIterator          key_first,
                          KeysInputIterator          key_last,
                          ValuesInputIterator        value_first,
                          ValuesOutputIterator       value_result,
                          InitialValueType           init,
                          KeyCompareFunction         key_compare_op,
                          BinaryFunction             scan_op)
    {
        size_t      num_items    = static_cast<size_t>(thrust::distance(key_first, key_last));
        size_t      storage_size = 0;
        hipStream_t stream       = hip_rocprim::stream(policy);
        bool        debug_sync   = THRUST_HIP_DEBUG_SYNC_FLAG;

        if(num_items == 0)
            return value_result;

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::exclusive_scan_by_key(NULL,
                                                                   storage_size,
                                                                   key_first,
                                                                   value_first,
                                                                   value_result,
                                                                   init,
                                                                   num_items,
                                                                   scan_op,
                                                                   key_compare_op,
                                                                   stream,
                                                                   debug_sync),
                                    "scan_by_key failed on 1st step");


        // Allocate temporary storage.
        thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
            tmp(policy, storage_size);
        void *ptr = static_cast<void*>(tmp.data().get());

        // Run scan.
        hip_rocprim::throw_on_error(rocprim::exclusive_scan_by_key(ptr,
                                                                   storage_size,
                                                                   key_first,
                                                                   value_first,
                                                                   value_result,
                                                                   init,
                                                                   num_items,
                                                                   scan_op,
                                                                   key_compare_op,
                                                                   stream,
                                                                   debug_sync),
                                    "scan_by_key failed on 2nd step");

        return value_result + num_items;
    }
} // namspace scan_by_key

//-------------------------
// Thrust API entry points
//-------------------------

//---------------------------
//   Inclusive scan
//---------------------------

__thrust_exec_check_disable__ template <class Derived,
                                        class KeyInputIt,
                                        class ValInputIt,
                                        class ValOutputIt,
                                        class BinaryPred,
                                        class ScanOp>
THRUST_HIP_FUNCTION ValOutputIt
inclusive_scan_by_key(execution_policy<Derived>& policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result,
                      BinaryPred                 binary_pred,
                      ScanOp                     scan_op)
{
  struct workaround
  {
      __host__
      static ValOutputIt par(execution_policy<Derived>& policy,
                          KeyInputIt                 key_first,
                          KeyInputIt                 key_last,
                          ValInputIt                 value_first,
                          ValOutputIt                value_result,
                          BinaryPred                 binary_pred,
                          ScanOp                     scan_op)
      {
        #if __HCC__ && __HIP_DEVICE_COMPILE__
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
               (__scan_by_key::inclusive_scan_by_key<Derived,KeyInputIt,ValInputIt,ValOutputIt,BinaryPred,ScanOp>)
        );
        #else
        return __scan_by_key::inclusive_scan_by_key(policy,key_first, key_last, value_first, value_result, binary_pred, scan_op);
        #endif
      }

      __device__
      static ValOutputIt seq(execution_policy<Derived>& policy,
                          KeyInputIt                 key_first,
                          KeyInputIt                 key_last,
                          ValInputIt                 value_first,
                          ValOutputIt                value_result,
                          BinaryPred                 binary_pred,
                          ScanOp                     scan_op)
      {
        return thrust::inclusive_scan_by_key( cvt_to_seq(derived_cast(policy)), key_first, key_last, value_first, value_result, binary_pred, scan_op);
      }
  };

  #if __THRUST_HAS_HIPRT__
    return workaround::par(policy, key_first, key_last, value_first, value_result, binary_pred, scan_op);
  #else
    return workaround::seq(policy, key_first, key_last, value_first, value_result, binary_pred, scan_op);
  #endif



}

template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class ValOutputIt,
          class BinaryPred>
THRUST_HIP_FUNCTION ValOutputIt
inclusive_scan_by_key(execution_policy<Derived>& policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result,
                      BinaryPred                 binary_pred)
{
    typedef typename thrust::iterator_traits<ValOutputIt>::value_type value_type;
    return hip_rocprim::inclusive_scan_by_key(policy,
                                              key_first,
                                              key_last,
                                              value_first,
                                              value_result,
                                              binary_pred,
                                              plus<value_type>());
}

template <class Derived, class KeyInputIt, class ValInputIt, class ValOutputIt>
THRUST_HIP_FUNCTION ValOutputIt
inclusive_scan_by_key(execution_policy<Derived>& policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result)
{
    typedef typename thrust::iterator_traits<KeyInputIt>::value_type key_type;
    return hip_rocprim::inclusive_scan_by_key(
        policy, key_first, key_last, value_first, value_result, equal_to<key_type>()
    );
}

//---------------------------
//   Exclusive scan
//---------------------------

__thrust_exec_check_disable__ template <class Derived,
                                        class KeyInputIt,
                                        class ValInputIt,
                                        class ValOutputIt,
                                        class Init,
                                        class BinaryPred,
                                        class ScanOp>
THRUST_HIP_FUNCTION ValOutputIt
exclusive_scan_by_key(execution_policy<Derived>& policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result,
                      Init                       init,
                      BinaryPred                 binary_pred,
                      ScanOp                     scan_op)
{

  struct workaround
  {
      __host__
      static ValOutputIt par(execution_policy<Derived>& policy,
                            KeyInputIt                 key_first,
                            KeyInputIt                 key_last,
                            ValInputIt                 value_first,
                            ValOutputIt                value_result,
                            Init                       init,
                            BinaryPred                 binary_pred,
                            ScanOp                     scan_op)
      {
        #if __HCC__ && __HIP_DEVICE_COMPILE__
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
               (__scan_by_key::inclusive_scan_by_key<Derived,KeyInputIt,ValInputIt,ValOutputIt,Init,BinaryPred,ScanOp>)
        );
        #else
        return __scan_by_key::exclusive_scan_by_key(policy,key_first, key_last, value_first, value_result, init, binary_pred, scan_op);
        #endif
      }

      __device__
      static ValOutputIt seq(execution_policy<Derived>& policy,
                            KeyInputIt                 key_first,
                            KeyInputIt                 key_last,
                            ValInputIt                 value_first,
                            ValOutputIt                value_result,
                            Init                       init,
                            BinaryPred                 binary_pred,
                            ScanOp                     scan_op)
      {
        return thrust::exclusive_scan_by_key( cvt_to_seq(derived_cast(policy)), key_first, key_last, value_first, value_result, init, binary_pred, scan_op);
      }
  };
  
  #if __THRUST_HAS_HIPRT__
    return workaround::par(policy, key_first, key_last, value_first, value_result, init, binary_pred, scan_op);
  #else
    return workaround::seq(policy, key_first, key_last, value_first, value_result, init, binary_pred, scan_op);
  #endif




}

template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class ValOutputIt,
          class Init,
          class BinaryPred>
THRUST_HIP_FUNCTION ValOutputIt
exclusive_scan_by_key(execution_policy<Derived>& policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result,
                      Init                       init,
                      BinaryPred                 binary_pred)
{
    return hip_rocprim::exclusive_scan_by_key(policy,
                                              key_first,
                                              key_last,
                                              value_first,
                                              value_result,
                                              init,
                                              binary_pred,
                                              plus<Init>());
}

template <class Derived, class KeyInputIt, class ValInputIt, class ValOutputIt, class Init>
THRUST_HIP_FUNCTION ValOutputIt
exclusive_scan_by_key(execution_policy<Derived>& policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result,
                      Init                       init)
{
    typedef typename iterator_traits<KeyInputIt>::value_type key_type;
    return hip_rocprim::exclusive_scan_by_key(
        policy, key_first, key_last, value_first, value_result, init, equal_to<key_type>()
    );
}

template <class Derived, class KeyInputIt, class ValInputIt, class ValOutputIt>
THRUST_HIP_FUNCTION ValOutputIt
exclusive_scan_by_key(execution_policy<Derived>& policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result)
{
    typedef typename iterator_traits<ValOutputIt>::value_type value_type;
    return hip_rocprim::exclusive_scan_by_key(
        policy, key_first, key_last, value_first, value_result, value_type(0)
    );
}

} // namespace hip_rocprim
THRUST_END_NS

#include <thrust/scan.h>

#endif
