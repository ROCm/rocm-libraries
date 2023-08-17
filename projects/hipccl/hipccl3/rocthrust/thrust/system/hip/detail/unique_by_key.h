 /******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *  Modifications Copyright© 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <thrust/detail/minmax.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/detail/range/head_flags.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/pair.h>
#include <thrust/system/hip/config.h>
#include <thrust/system/hip/detail/get_value.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/detail/util.h>

// rocPRIM includes
#include <rocprim/rocprim.hpp>

THRUST_NAMESPACE_BEGIN

template <typename DerivedPolicy, typename ForwardIterator1, typename ForwardIterator2>
thrust::pair<ForwardIterator1, ForwardIterator2> __host__ __device__
unique_by_key(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
              ForwardIterator1                                            keys_first,
              ForwardIterator1                                            keys_last,
              ForwardIterator2                                            values_first);
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2>
 __host__ __device__ thrust::pair<OutputIterator1, OutputIterator2>
unique_by_key_copy(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                   InputIterator1                                              keys_first,
                   InputIterator1                                              keys_last,
                   InputIterator2                                              values_first,
                   OutputIterator1                                             keys_result,
                   OutputIterator2                                             values_result);

namespace hip_rocprim
{
namespace __unique_by_key
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

    template <typename Derived,
              typename KeyInputIt,
              typename ValInputIt,
              typename KeyOutputIt,
              typename ValOutputIt,
              typename BinaryPred>
    THRUST_HIP_RUNTIME_FUNCTION
    pair<KeyOutputIt, ValOutputIt>
    unique_by_key(execution_policy<Derived>& policy,
                  KeyInputIt                 keys_first,
                  KeyInputIt                 keys_last,
                  ValInputIt                 values_first,
                  KeyOutputIt                keys_result,
                  ValOutputIt                values_result,
                  BinaryPred                 binary_pred)
    {
        typedef size_t size_type;

        typedef typename iterator_traits<KeyInputIt>::value_type KeyType;
        typedef typename iterator_traits<ValInputIt>::value_type ValueType;

        predicate_wrapper<KeyType, ValueType, BinaryPred> wrapped_binary_pred(binary_pred);

        size_type   num_items = static_cast<size_type>(thrust::distance(keys_first, keys_last));
        size_t      temp_storage_bytes = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        if(num_items == 0)
            return thrust::make_pair(keys_result, values_result);

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(
            rocprim::unique(
                NULL,
                temp_storage_bytes,
                rocprim::make_zip_iterator(rocprim::make_tuple(keys_first, values_first)),
                rocprim::make_zip_iterator(rocprim::make_tuple(keys_result, values_result)),
                reinterpret_cast<size_type*>(NULL),
                num_items,
                wrapped_binary_pred,
                stream,
                debug_sync),
            "unique_by_key failed on 1st step");

        // Allocate temporary storage.
        thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
            tmp(policy, temp_storage_bytes + sizeof(size_type));
        void *ptr = static_cast<void*>(tmp.data().get());

        size_type* d_num_selected_out = reinterpret_cast<size_type*>(
            reinterpret_cast<char*>(ptr) + temp_storage_bytes);

        hip_rocprim::throw_on_error(
            rocprim::unique(
                ptr,
                temp_storage_bytes,
                rocprim::make_zip_iterator(rocprim::make_tuple(keys_first, values_first)),
                rocprim::make_zip_iterator(rocprim::make_tuple(keys_result, values_result)),
                d_num_selected_out,
                num_items,
                wrapped_binary_pred,
                stream,
                debug_sync),
            "unique_by_key failed on 2nd step");

        size_type num_selected = get_value(policy, d_num_selected_out);

        return thrust::make_pair(keys_result + num_selected, values_result + num_selected);
    }
} // namespace __unique_by_key

//-------------------------
// Thrust API entry points
//-------------------------

__thrust_exec_check_disable__ template <class Derived,
                                        class KeyInputIt,
                                        class ValInputIt,
                                        class KeyOutputIt,
                                        class ValOutputIt,
                                        class BinaryPred>
pair<KeyOutputIt, ValOutputIt>
    THRUST_HIP_FUNCTION unique_by_key_copy(execution_policy<Derived>& policy,
                                           KeyInputIt                 keys_first,
                                           KeyInputIt                 keys_last,
                                           ValInputIt                 values_first,
                                           KeyOutputIt                keys_result,
                                           ValOutputIt                values_result,
                                           BinaryPred                 binary_pred)
{
    // struct workaround is required for HIP-clang
    struct workaround
    {
        __host__
        static pair<KeyOutputIt, ValOutputIt> par(execution_policy<Derived>& policy,
                                                  KeyInputIt                 keys_first,
                                                  KeyInputIt                 keys_last,
                                                  ValInputIt                 values_first,
                                                  KeyOutputIt                keys_result,
                                                  ValOutputIt                values_result,
                                                  BinaryPred                 binary_pred)
        {
            return __unique_by_key::unique_by_key(policy,
                                                  keys_first,
                                                  keys_last,
                                                  values_first,
                                                  keys_result,
                                                  values_result,
                                                  binary_pred);
        }
        __device__
        static pair<KeyOutputIt, ValOutputIt> seq(execution_policy<Derived>& policy,
                                                  KeyInputIt                 keys_first,
                                                  KeyInputIt                 keys_last,
                                                  ValInputIt                 values_first,
                                                  KeyOutputIt                keys_result,
                                                  ValOutputIt                values_result,
                                                  BinaryPred                 binary_pred)
        {
            return thrust::unique_by_key_copy(
                cvt_to_seq(derived_cast(policy)),
                keys_first,
                keys_last,
                values_first,
                keys_result,
                values_result,
                binary_pred
            );
        }
    };
    #if __THRUST_HAS_HIPRT__
        return workaround::par(policy, keys_first, keys_last, values_first, keys_result, values_result, binary_pred);
    #else
        return workaround::seq(policy, keys_first, keys_last, values_first, keys_result, values_result, binary_pred);
    #endif
}

template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class KeyOutputIt,
          class ValOutputIt>
pair<KeyOutputIt, ValOutputIt> THRUST_HIP_FUNCTION
unique_by_key_copy(execution_policy<Derived>& policy,
                   KeyInputIt                 keys_first,
                   KeyInputIt                 keys_last,
                   ValInputIt                 values_first,
                   KeyOutputIt                keys_result,
                   ValOutputIt                values_result)
{
    typedef typename iterator_traits<KeyInputIt>::value_type key_type;
    return hip_rocprim::unique_by_key_copy(policy,
                                           keys_first,
                                           keys_last,
                                           values_first,
                                           keys_result,
                                           values_result,
                                           equal_to<key_type>());
}

template <class Derived, class KeyInputIt, class ValInputIt, class BinaryPred>
pair<KeyInputIt, ValInputIt> THRUST_HIP_FUNCTION
unique_by_key(execution_policy<Derived>& policy,
              KeyInputIt                 keys_first,
              KeyInputIt                 keys_last,
              ValInputIt                 values_first,
              BinaryPred                 binary_pred)
{
    // struct workaround is required for HIP-clang
    struct workaround
    {
        __host__ static pair<KeyInputIt, ValInputIt> par(execution_policy<Derived>& policy,
                                                         KeyInputIt                 keys_first,
                                                         KeyInputIt                 keys_last,
                                                         ValInputIt                 values_first,
                                                         BinaryPred                 binary_pred)
        {
            return hip_rocprim::unique_by_key_copy(
                policy, keys_first, keys_last, values_first, keys_first, values_first, binary_pred);
        }
        __device__ static pair<KeyInputIt, ValInputIt> seq(execution_policy<Derived>& policy,
                                                           KeyInputIt                 keys_first,
                                                           KeyInputIt                 keys_last,
                                                           ValInputIt                 values_first,
                                                           BinaryPred                 binary_pred)
        {
            return thrust::unique_by_key(
                cvt_to_seq(derived_cast(policy)), keys_first, keys_last, values_first, binary_pred);
        }
  };
  #if __THRUST_HAS_HIPRT__
      return workaround::par(policy, keys_first, keys_last, values_first, binary_pred);
  #else
      return workaround::seq(policy, keys_first, keys_last, values_first, binary_pred);
  #endif
}

template <class Derived, class KeyInputIt, class ValInputIt>
pair<KeyInputIt, ValInputIt> THRUST_HIP_FUNCTION
unique_by_key(execution_policy<Derived>& policy,
              KeyInputIt                 keys_first,
              KeyInputIt                 keys_last,
              ValInputIt                 values_first)
{
    typedef typename iterator_traits<KeyInputIt>::value_type key_type;
    return hip_rocprim::unique_by_key(
        policy, keys_first, keys_last, values_first, equal_to<key_type>()
    );
}

} // namespace hip_rocprim
THRUST_NAMESPACE_END

#include <thrust/memory.h>
#include <thrust/unique.h>

#endif
