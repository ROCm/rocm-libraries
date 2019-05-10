/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/distance.h>
#include <thrust/sort.h>
#include <thrust/system/hip/detail/execution_policy.h>
#include <thrust/system/hip/detail/memory_buffer.h>
#include <thrust/system/hip/detail/par_to_seq.h>

// rocPRIM includes
#include <rocprim/rocprim.hpp>

BEGIN_NS_THRUST
namespace hip_rocprim
{
namespace __merge_sort
{
    template <class SORT_ITEMS>
    struct dispatch;

    // sort keys
    template <>
    struct dispatch<detail::false_type>
    {
        template <class KeysIt, class ItemsIt, class Size, class CompareOp>
        static hipError_t THRUST_HIP_RUNTIME_FUNCTION
        doit(void*       d_temp_storage,
             size_t&     temp_storage_bytes,
             KeysIt      keys,
             ItemsIt     /*items*/,
             Size        count,
             CompareOp   compare_op,
             hipStream_t stream,
             bool        debug_sync)
        {
            return rocprim::merge_sort(d_temp_storage,
                                       temp_storage_bytes,
                                       keys,
                                       keys,
                                       static_cast<unsigned int>(count),
                                       compare_op,
                                       stream,
                                       debug_sync);
        }
    };

    // sort pairs
    template <>
    struct dispatch<detail::true_type>
    {
        template <class KeysIt, class ItemsIt, class Size, class CompareOp>
        static hipError_t THRUST_HIP_RUNTIME_FUNCTION
        doit(void*       d_temp_storage,
             size_t&     temp_storage_bytes,
             KeysIt      keys,
             ItemsIt     items,
             Size        count,
             CompareOp   compare_op,
             hipStream_t stream,
             bool        debug_sync)
        {
            return rocprim::merge_sort(d_temp_storage,
                                       temp_storage_bytes,
                                       keys,
                                       keys,
                                       items,
                                       items,
                                       static_cast<unsigned int>(count),
                                       compare_op,
                                       stream,
                                       debug_sync);
        }
    };

    template <class SORT_ITEMS, class Policy, class KeysIt, class ItemsIt, class CompareOp>
    void THRUST_HIP_RUNTIME_FUNCTION
    merge_sort(Policy&   policy,
               KeysIt    keys_first,
               KeysIt    keys_last,
               ItemsIt   items_first,
               CompareOp compare_op)
    {
        typedef typename iterator_traits<KeysIt>::difference_type size_type;

        const size_type count = static_cast<size_type>(thrust::distance(keys_first, keys_last));

        void*       d_temp_storage     = NULL;
        size_t      temp_storage_bytes = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        hipError_t status;

        status = dispatch<SORT_ITEMS>::doit(d_temp_storage,
                                            temp_storage_bytes,
                                            keys_first,
                                            items_first,
                                            count,
                                            compare_op,
                                            stream,
                                            debug_sync);
        hip_rocprim::throw_on_error(status, "merge_sort: failed on 1st step");

        d_temp_storage = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes);
        hip_rocprim::throw_on_error(hipGetLastError(),
                                    "merge_sort: failed to get memory buffer");

        status = dispatch<SORT_ITEMS>::doit(d_temp_storage,
                                            temp_storage_bytes,
                                            keys_first,
                                            items_first,
                                            count,
                                            compare_op,
                                            stream,
                                            debug_sync);
        hip_rocprim::throw_on_error(status, "merge_sort: failed on 2nd step");

        hip_rocprim::return_memory_buffer(policy, d_temp_storage);
        hip_rocprim::throw_on_error(hipGetLastError(),
                                    "merge_sort: failed to return memory buffer");
    }
} // namespace __merge_sort

namespace __radix_sort
{

    template <class SORT_ITEMS, class Comparator>
    struct dispatch;

    // sort keys in ascending order
    template <class K>
    struct dispatch<detail::false_type, thrust::less<K>>
    {
        template <class KeysIt, class ItemsIt, class Size>
        static hipError_t THRUST_HIP_RUNTIME_FUNCTION
        doit(void*   d_temp_storage,
             size_t& temp_storage_bytes,
             KeysIt  keys,
             ItemsIt /*items*/,
             Size        count,
             hipStream_t stream,
             bool        debug_sync)
        {
            return rocprim::radix_sort_keys(d_temp_storage,
                                            temp_storage_bytes,
                                            keys,
                                            keys,
                                            static_cast<unsigned int>(count),
                                            0,
                                            sizeof(K) * 8,
                                            stream,
                                            debug_sync);
        }
    }; // struct dispatch -- sort keys in ascending order;

    // sort keys in descending order
    template <class K>
    struct dispatch<detail::false_type, thrust::greater<K>>
    {
        template <class KeysIt, class ItemsIt, class Size>
        static hipError_t THRUST_HIP_RUNTIME_FUNCTION
        doit(void*   d_temp_storage,
             size_t& temp_storage_bytes,
             KeysIt  keys,
             ItemsIt /*items*/,
             Size        count,
             hipStream_t stream,
             bool        debug_sync)
        {
            return rocprim::radix_sort_keys_desc(d_temp_storage,
                                                 temp_storage_bytes,
                                                 keys,
                                                 keys,
                                                 static_cast<unsigned int>(count),
                                                 0,
                                                 sizeof(K) * 8,
                                                 stream,
                                                 debug_sync);
        }
    }; // struct dispatch -- sort keys in descending order;

    // sort pairs in ascending order
    template <class K>
    struct dispatch<detail::true_type, thrust::less<K>>
    {
        template <class KeysIt, class ItemsIt, class Size>
        static hipError_t THRUST_HIP_RUNTIME_FUNCTION
        doit(void*       d_temp_storage,
             size_t&     temp_storage_bytes,
             KeysIt      keys,
             ItemsIt     items,
             Size        count,
             hipStream_t stream,
             bool        debug_sync)
        {
            return rocprim::radix_sort_pairs(d_temp_storage,
                                             temp_storage_bytes,
                                             keys,
                                             keys,
                                             items,
                                             items,
                                             static_cast<unsigned int>(count),
                                             0,
                                             sizeof(K) * 8,
                                             stream,
                                             debug_sync);
        }
    }; // struct dispatch -- sort pairs in ascending order;

    // sort pairs in descending order
    template <class K>
    struct dispatch<detail::true_type, thrust::greater<K>>
    {
        template <class KeysIt, class ItemsIt, class Size>
         static hipError_t THRUST_HIP_RUNTIME_FUNCTION
         doit(void*       d_temp_storage,
              size_t&     temp_storage_bytes,
              KeysIt      keys,
              ItemsIt     items,
              Size        count,
              hipStream_t stream,
              bool        debug_sync)
        {
            return rocprim::radix_sort_pairs_desc(d_temp_storage,
                                                  temp_storage_bytes,
                                                  keys,
                                                  keys,
                                                  items,
                                                  items,
                                                  static_cast<unsigned int>(count),
                                                  0,
                                                  sizeof(K) * 8,
                                                  stream,
                                                  debug_sync);
        }
    }; // struct dispatch -- sort pairs in descending order;

    template <class SORT_ITEMS, class Policy, class KeysIt, class ItemsIt, class CompareOp>
    void THRUST_HIP_RUNTIME_FUNCTION
    radix_sort(Policy&   policy,
               KeysIt    keys_first,
               KeysIt    keys_last,
               ItemsIt   items_first,
               CompareOp )
    {
        typedef typename iterator_traits<KeysIt>::difference_type size_type;

        const size_type count = static_cast<size_type>(thrust::distance(keys_first, keys_last));

        void*       d_temp_storage     = NULL;
        size_t      temp_storage_bytes = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        hipError_t status;

        status = dispatch<SORT_ITEMS, CompareOp>::doit(d_temp_storage,
                                                       temp_storage_bytes,
                                                       keys_first,
                                                       items_first,
                                                       count,
                                                       stream,
                                                       debug_sync);
        hip_rocprim::throw_on_error(status, "radix_sort: failed on 1st step");

        d_temp_storage = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes);
        hip_rocprim::throw_on_error(hipGetLastError(),
                                    "radix_sort: failed to get memory buffer");

        status = dispatch<SORT_ITEMS, CompareOp>::doit(d_temp_storage,
                                                       temp_storage_bytes,
                                                       keys_first,
                                                       items_first,
                                                       count,
                                                       stream,
                                                       debug_sync);
        hip_rocprim::throw_on_error(status, "radix_sort: failed on 2nd step");

        hip_rocprim::return_memory_buffer(policy, d_temp_storage);
        hip_rocprim::throw_on_error(hipGetLastError(),
                                    "radix_sort: failed to return memory buffer");
    }
} // __radix_sort

//---------------------------------------------------------------------
// Smart sort picks at runtime whether to dispatch radix or merge sort
//---------------------------------------------------------------------

namespace __smart_sort
{
    template <class Key, class CompareOp>
    struct can_use_primitive_sort
        : thrust::detail::and_<
              thrust::detail::is_arithmetic<Key>,
              thrust::detail::or_<thrust::detail::is_same<CompareOp, thrust::less<Key>>,
                                  thrust::detail::is_same<CompareOp, thrust::greater<Key>>>>
    {
    };

    template <class Iterator, class CompareOp>
    struct enable_if_primitive_sort
        : thrust::detail::enable_if<
              can_use_primitive_sort<typename iterator_value<Iterator>::type, CompareOp>::value>
    {
    };

    template <class Iterator, class CompareOp>
    struct enable_if_comparison_sort
        : thrust::detail::disable_if<
              can_use_primitive_sort<typename iterator_value<Iterator>::type, CompareOp>::value>
    {
    };

    template <class SORT_ITEMS, class Derived, class KeysIt, class ItemsIt, class CompareOp>
    typename enable_if_comparison_sort<KeysIt, CompareOp>::type
    THRUST_HIP_RUNTIME_FUNCTION
    smart_sort(execution_policy<Derived>& policy,
               KeysIt                     keys_first,
               KeysIt                     keys_last,
               ItemsIt                    items_first,
               CompareOp                  compare_op)
    {
        __merge_sort::merge_sort<SORT_ITEMS>(
            policy, keys_first, keys_last, items_first, compare_op
        );
    }

    template <class SORT_ITEMS, class Derived, class KeysIt, class ItemsIt, class CompareOp>
    typename enable_if_primitive_sort<KeysIt, CompareOp>::type
    THRUST_HIP_RUNTIME_FUNCTION
    smart_sort(execution_policy<Derived>& policy,
               KeysIt                     keys_first,
               KeysIt                     keys_last,
               ItemsIt                    items_first,
               CompareOp                  compare_op)
    {
        __radix_sort::radix_sort<SORT_ITEMS>(
            policy, keys_first, keys_last, items_first, compare_op
        );
    }
}; // namespace __smart_sort

//-------------------------
// Thrust API entry points
//-------------------------

__thrust_exec_check_disable__ template <class Derived, class ItemsIt, class CompareOp>
void THRUST_HIP_FUNCTION
stable_sort(execution_policy<Derived>& policy,
            ItemsIt                    first,
            ItemsIt                    last,
            CompareOp                  compare_op)
{
    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
        (__smart_sort::smart_sort<detail::false_type, Derived, ItemsIt, ItemsIt, CompareOp>)
    );
#if __THRUST_HAS_HIPRT__
    typedef typename thrust::iterator_value<ItemsIt>::type item_type;
    __smart_sort::smart_sort<detail::false_type>(
        policy, first, last, (item_type*)NULL, compare_op
    );
#else
    thrust::stable_sort(cvt_to_seq(derived_cast(policy)), first, last, compare_op);
#endif
}

__thrust_exec_check_disable__ template <class Derived, class ItemsIt, class CompareOp>
void THRUST_HIP_FUNCTION
sort(execution_policy<Derived>& policy,
     ItemsIt                    first,
     ItemsIt                    last,
     CompareOp                  compare_op)
{
    hip_rocprim::stable_sort(policy, first, last, compare_op);
}

__thrust_exec_check_disable__ template <class Derived,
                                        class KeysIt,
                                        class ValuesIt,
                                        class CompareOp>
void THRUST_HIP_FUNCTION
stable_sort_by_key(execution_policy<Derived>& policy,
                   KeysIt                     keys_first,
                   KeysIt                     keys_last,
                   ValuesIt                   values,
                   CompareOp                  compare_op)
{
    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
        (__smart_sort::smart_sort<detail::true_type, Derived, KeysIt, ValuesIt, CompareOp>)
    );
#if __THRUST_HAS_HIPRT__
    __smart_sort::smart_sort<detail::true_type>(
        policy, keys_first, keys_last, values, compare_op
    );
#else
    thrust::stable_sort_by_key(
        cvt_to_seq(derived_cast(policy)), keys_first, keys_last, values, compare_op
    );
#endif
}

__thrust_exec_check_disable__ template <class Derived,
                                        class KeysIt,
                                        class ValuesIt,
                                        class CompareOp>
void THRUST_HIP_FUNCTION
sort_by_key(execution_policy<Derived>& policy,
            KeysIt                     keys_first,
            KeysIt                     keys_last,
            ValuesIt                   values,
            CompareOp                  compare_op)
{
    hip_rocprim::stable_sort_by_key(policy, keys_first, keys_last, values, compare_op);
}

// API with default comparator

template <class Derived, class ItemsIt>
void THRUST_HIP_FUNCTION
sort(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last)
{
    typedef typename thrust::iterator_value<ItemsIt>::type item_type;
    hip_rocprim::sort(policy, first, last, less<item_type>());
}

template <class Derived, class ItemsIt>
void THRUST_HIP_FUNCTION
stable_sort(execution_policy<Derived>& policy,
            ItemsIt                    first,
            ItemsIt                    last)
{
    typedef typename thrust::iterator_value<ItemsIt>::type item_type;
    hip_rocprim::stable_sort(policy, first, last, less<item_type>());
}

template <class Derived, class KeysIt, class ValuesIt>
void THRUST_HIP_FUNCTION
sort_by_key(execution_policy<Derived>& policy,
            KeysIt                     keys_first,
            KeysIt                     keys_last,
            ValuesIt                   values)
{
    typedef typename thrust::iterator_value<KeysIt>::type key_type;
    hip_rocprim::sort_by_key(policy, keys_first, keys_last, values, less<key_type>());
}

template <class Derived, class KeysIt, class ValuesIt>
void THRUST_HIP_FUNCTION
stable_sort_by_key(execution_policy<Derived>& policy,
                   KeysIt                     keys_first,
                   KeysIt                     keys_last,
                   ValuesIt                   values)
{
    typedef typename thrust::iterator_value<KeysIt>::type key_type;
    hip_rocprim::stable_sort_by_key(policy, keys_first, keys_last, values, less<key_type>());
}

} // namespace hip_rocprim
END_NS_THRUST
#endif
