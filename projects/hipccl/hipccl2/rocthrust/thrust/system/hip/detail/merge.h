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

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <thrust/system/hip/config.h>

#include <thrust/detail/minmax.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/detail/range/head_flags.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/merge.h>
#include <thrust/pair.h>
#include <thrust/system/hip/detail/get_value.h>
#include <thrust/system/hip/detail/memory_buffer.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/detail/util.h>

// rocPRIM includes
#include <rocprim/rocprim.hpp>

BEGIN_NS_THRUST
namespace hip_rocprim
{
namespace __merge
{
    template <class KeyType, class ValueType, class Predicate>
    struct predicate_wrapper
    {
        Predicate                                  predicate;
        typedef rocprim::tuple<KeyType, ValueType> pair_type;

        THRUST_HIP_FUNCTION
        predicate_wrapper(Predicate p)
            : predicate(p)
        {
        }

        bool THRUST_HIP_DEVICE_FUNCTION operator()(pair_type const& lhs,
                                                   pair_type const& rhs) const
        {
            return predicate(rocprim::get<0>(lhs), rocprim::get<0>(rhs));
        }
    }; // struct predicate_wrapper

    template <class Derived, class KeysIt1, class KeysIt2, class ResultIt, class CompareOp>
    ResultIt THRUST_HIP_RUNTIME_FUNCTION
    merge(execution_policy<Derived>& policy,
          KeysIt1                    keys1_first,
          KeysIt1                    keys1_last,
          KeysIt2                    keys2_first,
          KeysIt2                    keys2_last,
          ResultIt                   result,
          CompareOp                  compare_op)

    {
        typedef size_t size_type;

        size_type input1_size
            = static_cast<size_type>(thrust::distance(keys1_first, keys1_last));
        size_type input2_size
            = static_cast<size_type>(thrust::distance(keys2_first, keys2_last));

        void*       d_temp_storage     = NULL;
        size_t      temp_storage_bytes = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::merge(d_temp_storage,
                                                   temp_storage_bytes,
                                                   keys1_first,
                                                   keys2_first,
                                                   result,
                                                   input1_size,
                                                   input2_size,
                                                   compare_op,
                                                   stream,
                                                   debug_sync),
                                    "merge failed on 1st step");

        // Allocate temporary storage.
        d_temp_storage = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes);
        hip_rocprim::throw_on_error(hipGetLastError(), "merge failed to get memory buffer");

        hip_rocprim::throw_on_error(rocprim::merge(d_temp_storage,
                                                   temp_storage_bytes,
                                                   keys1_first,
                                                   keys2_first,
                                                   result,
                                                   input1_size,
                                                   input2_size,
                                                   compare_op,
                                                   stream,
                                                   debug_sync),
                                    "merge failed on 2nd step");

        hip_rocprim::return_memory_buffer(policy, d_temp_storage);
        hip_rocprim::throw_on_error(hipGetLastError(), "merge failed to return memory buffer");

        ResultIt result_end = result + input1_size + input2_size;
        return result_end;
    }

    template <class Policy,
              class KeysIt1,
              class KeysIt2,
              class ItemsIt1,
              class ItemsIt2,
              class KeysOutputIt,
              class ItemsOutputIt,
              class CompareOp>
    pair<KeysOutputIt, ItemsOutputIt> THRUST_HIP_RUNTIME_FUNCTION
    merge(Policy&       policy,
          KeysIt1       keys1_first,
          KeysIt1       keys1_last,
          KeysIt2       keys2_first,
          KeysIt2       keys2_last,
          ItemsIt1      items1_first,
          ItemsIt2      items2_first,
          KeysOutputIt  keys_result,
          ItemsOutputIt items_result,
          CompareOp     compare_op)
    {
        typedef size_t size_type;

        typedef typename iterator_traits<KeysIt1>::value_type  KeyType;
        typedef typename iterator_traits<ItemsIt1>::value_type ValueType;

        predicate_wrapper<KeyType, ValueType, CompareOp> wrapped_binary_pred(compare_op);

        size_type input1_size
            = static_cast<size_type>(thrust::distance(keys1_first, keys1_last));
        size_type input2_size
            = static_cast<size_type>(thrust::distance(keys2_first, keys2_last));

        void*       d_temp_storage     = NULL;
        size_t      temp_storage_bytes = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(
            rocprim::merge(
                d_temp_storage,
                temp_storage_bytes,
                rocprim::make_zip_iterator(rocprim::make_tuple(keys1_first, items1_first)),
                rocprim::make_zip_iterator(rocprim::make_tuple(keys2_first, items2_first)),
                rocprim::make_zip_iterator(rocprim::make_tuple(keys_result, items_result)),
                input1_size,
                input2_size,
                wrapped_binary_pred,
                stream,
                debug_sync),
            "merge_by_key failed on 1st step");

        // Allocate temporary storage.
        d_temp_storage = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes);
        hip_rocprim::throw_on_error(hipGetLastError(),
                                    "merge_by_key failed to get memory buffer");

        hip_rocprim::throw_on_error(
            rocprim::merge(
                d_temp_storage,
                temp_storage_bytes,
                rocprim::make_zip_iterator(rocprim::make_tuple(keys1_first, items1_first)),
                rocprim::make_zip_iterator(rocprim::make_tuple(keys2_first, items2_first)),
                rocprim::make_zip_iterator(rocprim::make_tuple(keys_result, items_result)),
                input1_size,
                input2_size,
                wrapped_binary_pred,
                stream,
                debug_sync),
            "merge_by_key failed on 2nd step");

        hip_rocprim::return_memory_buffer(policy, d_temp_storage);
        hip_rocprim::throw_on_error(hipGetLastError(),
                                    "merge_by_key failed to return memory buffer");

        size_t count = input1_size + input2_size;
        return thrust::make_pair(keys_result + count, items_result + count);
    }

} //namespace __merge

//-------------------------
// Thrust API entry points
//-------------------------
__thrust_exec_check_disable__ template <class Derived,
                                        class KeysIt1,
                                        class KeysIt2,
                                        class ResultIt,
                                        class CompareOp>
ResultIt THRUST_HIP_FUNCTION
merge(execution_policy<Derived>& policy,
      KeysIt1                    keys1_first,
      KeysIt1                    keys1_last,
      KeysIt2                    keys2_first,
      KeysIt2                    keys2_last,
      ResultIt                   result,
      CompareOp                  compare_op)

{
    ResultIt ret = result;
    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
        (__merge::merge<Derived, KeysIt1, KeysIt2, ResultIt, CompareOp>)
    );
#if __THRUST_HAS_HIPRT__
    ret = __merge::merge(
        policy, keys1_first, keys1_last, keys2_first, keys2_last, result, compare_op
    );
#else
    ret = thrust::merge(cvt_to_seq(derived_cast(policy)),
                        keys1_first,
                        keys1_last,
                        keys2_first,
                        keys2_last,
                        result,
                        compare_op);

#endif
    return ret;
}

__thrust_exec_check_disable__ template <class Derived,
                                        class KeysIt1,
                                        class KeysIt2,
                                        class ItemsIt1,
                                        class ItemsIt2,
                                        class KeysOutputIt,
                                        class ItemsOutputIt,
                                        class CompareOp>
pair<KeysOutputIt, ItemsOutputIt> THRUST_HIP_FUNCTION
merge_by_key(execution_policy<Derived>& policy,
             KeysIt1                    keys1_first,
             KeysIt1                    keys1_last,
             KeysIt2                    keys2_first,
             KeysIt2                    keys2_last,
             ItemsIt1                   items1_first,
             ItemsIt2                   items2_first,
             KeysOutputIt               keys_result,
             ItemsOutputIt              items_result,
             CompareOp                  compare_op)
{
    pair<KeysOutputIt, ItemsOutputIt> ret = thrust::make_pair(keys_result, items_result);
    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
      (__merge::merge<Derived,
                      KeysIt1,
                      KeysIt2,
                      ItemsIt1,
                      ItemsIt2,
                      KeysOutputIt,
                      ItemsOutputIt,
                      CompareOp>)
    );
#if __THRUST_HAS_HIPRT__
    return __merge::merge(policy,
                          keys1_first,
                          keys1_last,
                          keys2_first,
                          keys2_last,
                          items1_first,
                          items2_first,
                          keys_result,
                          items_result,
                          compare_op);

#else
    ret = thrust::merge_by_key(cvt_to_seq(derived_cast(policy)),
                               keys1_first,
                               keys1_last,
                               keys2_first,
                               keys2_last,
                               items1_first,
                               items2_first,
                               keys_result,
                               items_result,
                               compare_op);
#endif
    return ret;
}

__thrust_exec_check_disable__ template <class Derived,
                                        class KeysIt1,
                                        class KeysIt2,
                                        class ResultIt>
ResultIt THRUST_HIP_FUNCTION
merge(execution_policy<Derived>& policy,
      KeysIt1                    keys1_first,
      KeysIt1                    keys1_last,
      KeysIt2                    keys2_first,
      KeysIt2                    keys2_last,
      ResultIt                   result)
{
    typedef typename thrust::iterator_value<KeysIt1>::type keys_type;
    return hip_rocprim::merge(
        policy, keys1_first, keys1_last, keys2_first, keys2_last, result, less<keys_type>());
}

template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ItemsIt1,
          class ItemsIt2,
          class KeysOutputIt,
          class ItemsOutputIt>
pair<KeysOutputIt, ItemsOutputIt> THRUST_HIP_FUNCTION
merge_by_key(execution_policy<Derived>& policy,
             KeysIt1                    keys1_first,
             KeysIt1                    keys1_last,
             KeysIt2                    keys2_first,
             KeysIt2                    keys2_last,
             ItemsIt1                   items1_first,
             ItemsIt2                   items2_first,
             KeysOutputIt               keys_result,
             ItemsOutputIt              items_result)
{
    typedef typename thrust::iterator_value<ItemsIt1>::type items_type;
    return hip_rocprim::merge_by_key(policy,
                                     keys1_first,
                                     keys1_last,
                                     keys2_first,
                                     keys2_last,
                                     items1_first,
                                     items2_first,
                                     keys_result,
                                     items_result,
                                     thrust::less<items_type>());
}

} // namespace hip_rocprim

END_NS_THRUST
#endif
