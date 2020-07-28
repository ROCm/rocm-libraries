/*******************************************************************************
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
#include <thrust/system/hip/config.h>
#include <thrust/system/hip/detail/reduce.h>
#include <thrust/system/hip/detail/get_value.h>

#include <thrust/detail/cstdint.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/distance.h>


// rocprim include
#include <rocprim/rocprim.hpp>

THRUST_BEGIN_NS
namespace hip_rocprim
{
namespace __extrema
{
    template <class InputType, class IndexType, class Predicate>
    struct arg_min_f
    {
        Predicate                           predicate;
        typedef tuple<InputType, IndexType> pair_type;

        THRUST_HIP_FUNCTION
        arg_min_f(Predicate p)
            : predicate(p)
        {
        }

        pair_type THRUST_HIP_DEVICE_FUNCTION
        operator()(pair_type const& lhs,
                   pair_type const& rhs)
        {
            InputType const& rhs_value = get<0>(rhs);
            InputType const& lhs_value = get<0>(lhs);
            IndexType const& rhs_key   = get<1>(rhs);
            IndexType const& lhs_key   = get<1>(lhs);

            // check values first
            if(predicate(lhs_value, rhs_value))
                return lhs;
            else if(predicate(rhs_value, lhs_value))
                return rhs;

            // values are equivalent, prefer smaller index
            if(lhs_key < rhs_key)
                return lhs;
            else
                return rhs;
        }
    }; // struct arg_min_f

    template <class InputType, class IndexType, class Predicate>
    struct arg_max_f
    {
        Predicate                           predicate;
        typedef tuple<InputType, IndexType> pair_type;

        THRUST_HIP_FUNCTION
        arg_max_f(Predicate p)
            : predicate(p)
        {
        }

        pair_type THRUST_HIP_DEVICE_FUNCTION
        operator()(pair_type const& lhs,
                   pair_type const& rhs)
        {
            InputType const& rhs_value = get<0>(rhs);
            InputType const& lhs_value = get<0>(lhs);
            IndexType const& rhs_key   = get<1>(rhs);
            IndexType const& lhs_key   = get<1>(lhs);

            // check values first
            if(predicate(lhs_value, rhs_value))
                return rhs;
            else if(predicate(rhs_value, lhs_value))
                return lhs;

            // values are equivalent, prefer smaller index
            if(lhs_key < rhs_key)
                return lhs;
            else
                return rhs;
        }
    }; // struct arg_max_f

    template <class InputType, class IndexType, class Predicate>
    struct arg_minmax_f
    {
        Predicate predicate;

        typedef tuple<InputType, IndexType> pair_type;
        typedef tuple<pair_type, pair_type> two_pairs_type;

        typedef arg_min_f<InputType, IndexType, Predicate> arg_min_t;
        typedef arg_max_f<InputType, IndexType, Predicate> arg_max_t;

        THRUST_HIP_FUNCTION
        arg_minmax_f(Predicate p)
            : predicate(p)
        {
        }

        two_pairs_type THRUST_HIP_DEVICE_FUNCTION
        operator()(two_pairs_type const& lhs,
                   two_pairs_type const& rhs)
        {
            pair_type const& rhs_min = get<0>(rhs);
            pair_type const& lhs_min = get<0>(lhs);
            pair_type const& rhs_max = get<1>(rhs);
            pair_type const& lhs_max = get<1>(lhs);
            return make_tuple(arg_min_t(predicate)(lhs_min, rhs_min),
                              arg_max_t(predicate)(lhs_max, rhs_max));
        }

        struct duplicate_tuple
        {
            two_pairs_type THRUST_HIP_DEVICE_FUNCTION
            operator()(pair_type const& t)
            {
                return thrust::make_tuple(t, t);
            }
        };
    }; // struct arg_minmax_f

    template <class Derived, class InputIt, class Size, class BinaryOp, class T>
    T THRUST_HIP_RUNTIME_FUNCTION
    extrema(execution_policy<Derived>& policy,
            InputIt                    first,
            Size                       num_items,
            BinaryOp                   binary_op,
            T*                         )
    {
        if(num_items == 0)
            hip_rocprim::throw_on_error(hipErrorInvalidValue,
                                        "extrema number of items is zero");

        size_t      temp_storage_bytes = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::reduce(NULL,
                                                    temp_storage_bytes,
                                                    first,
                                                    reinterpret_cast<T*>(NULL),
                                                    num_items,
                                                    binary_op,
                                                    stream,
                                                    debug_sync),
                                    "extrema failed on 1st step");

        size_t storage_size = temp_storage_bytes + sizeof(T);

        // Allocate temporary storage.
        thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
            tmp(policy, storage_size);
        void *ptr = static_cast<void*>(tmp.data().get());

        T* d_result = reinterpret_cast<T*>(
	    reinterpret_cast<char*>(ptr) + temp_storage_bytes);

        hip_rocprim::throw_on_error(rocprim::reduce(ptr,
                                                    temp_storage_bytes,
                                                    first,
                                                    d_result,
                                                    num_items,
                                                    binary_op,
                                                    stream,
                                                    debug_sync),
                                    "extrema failed on 2nd step");

        T return_value = hip_rocprim::get_value(policy, d_result);


        return return_value;
    }

    template <template <class, class, class> class ArgFunctor,
              class Derived,
              class ItemsIt,
              class BinaryPred>
    ItemsIt THRUST_HIP_RUNTIME_FUNCTION
    element(execution_policy<Derived>& policy,
            ItemsIt                    first,
            ItemsIt                    last,
            BinaryPred                 binary_pred)
    {
        if(first == last)
            return last;

        typedef typename iterator_traits<ItemsIt>::value_type      InputType;
        typedef typename iterator_traits<ItemsIt>::difference_type IndexType;

        IndexType num_items = static_cast<IndexType>(thrust::distance(first, last));

        typedef tuple<ItemsIt, counting_iterator_t<IndexType>> iterator_tuple;
        typedef zip_iterator<iterator_tuple>                   zip_iterator;

        iterator_tuple iter_tuple = make_tuple(first, counting_iterator_t<IndexType>(0));

        typedef ArgFunctor<InputType, IndexType, BinaryPred> arg_min_t;
        typedef tuple<InputType, IndexType>                  T;

        zip_iterator begin = make_zip_iterator(iter_tuple);

        T result = extrema(policy, begin, num_items, arg_min_t(binary_pred), (T*)(NULL));
        return first + thrust::get<1>(result);
    }

} // namespace __extrema

/// min element
__thrust_exec_check_disable__ template <class Derived, class ItemsIt, class BinaryPred>
ItemsIt THRUST_HIP_FUNCTION
min_element(execution_policy<Derived>& policy,
            ItemsIt                    first,
            ItemsIt                    last,
            BinaryPred                 binary_pred)
{
    ItemsIt ret = first;

    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
        (__extrema::element<__extrema::arg_min_f, Derived, ItemsIt, BinaryPred>)
    );
#if __THRUST_HAS_HIPRT__
    ret = __extrema::element<__extrema::arg_min_f>(policy, first, last, binary_pred);
#else // __THRUST_HAS_HIPRT__
    ret = thrust::min_element(cvt_to_seq(derived_cast(policy)), first, last, binary_pred);
#endif // __THRUST_HAS_HIPRT__
    return ret;
}

template <class Derived, class ItemsIt>
ItemsIt THRUST_HIP_FUNCTION
min_element(execution_policy<Derived>& policy,
            ItemsIt                    first,
            ItemsIt                    last)
{
    typedef typename iterator_value<ItemsIt>::type value_type;
    return hip_rocprim::min_element(policy, first, last, less<value_type>());
}

/// max element
__thrust_exec_check_disable__ template <class Derived, class ItemsIt, class BinaryPred>
ItemsIt THRUST_HIP_FUNCTION
max_element(execution_policy<Derived>& policy,
            ItemsIt                    first,
            ItemsIt                    last,
            BinaryPred                 binary_pred)
{
    ItemsIt ret = first;

    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
        (__extrema::element<__extrema::arg_max_f, Derived, ItemsIt, BinaryPred>)
    );
#if __THRUST_HAS_HIPRT__
    ret = __extrema::element<__extrema::arg_max_f>(policy, first, last, binary_pred);
#else // __THRUST_HAS_HIPRT__
    ret = thrust::max_element(cvt_to_seq(derived_cast(policy)), first, last, binary_pred);
#endif // __THRUST_HAS_HIPRT__
    return ret;
}

template <class Derived, class ItemsIt>
ItemsIt THRUST_HIP_FUNCTION
max_element(execution_policy<Derived>& policy,
            ItemsIt                    first,
            ItemsIt                    last)
{
    typedef typename iterator_value<ItemsIt>::type value_type;
    return hip_rocprim::max_element(policy, first, last, less<value_type>());
}

/// minmax element
__thrust_exec_check_disable__ template <class Derived, class ItemsIt, class BinaryPred>
pair<ItemsIt, ItemsIt> THRUST_HIP_FUNCTION
minmax_element(execution_policy<Derived>& policy,
               ItemsIt                    first,
               ItemsIt                    last,
               BinaryPred                 binary_pred)
{
    pair<ItemsIt, ItemsIt> ret = thrust::make_pair(first, first);

    typedef typename iterator_traits<ItemsIt>::value_type      InputType;
    typedef typename iterator_traits<ItemsIt>::difference_type IndexType;

    typedef tuple<ItemsIt, hip_rocprim::counting_iterator_t<IndexType>> iterator_tuple;
    typedef zip_iterator<iterator_tuple>                                zip_iterator;

    typedef __extrema::arg_minmax_f<InputType, IndexType, BinaryPred> arg_minmax_t;

    typedef typename arg_minmax_t::two_pairs_type  two_pairs_type;
    typedef typename arg_minmax_t::duplicate_tuple duplicate_t;
    typedef transform_input_iterator_t<two_pairs_type, zip_iterator, duplicate_t> transform_t;

    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
        (__extrema::extrema<Derived, transform_t, IndexType, arg_minmax_t, two_pairs_type>)
    );
#if __THRUST_HAS_HIPRT__
    if(first == last)
        return thrust::make_pair(last, last);

    IndexType num_items = static_cast<IndexType>(thrust::distance(first, last));

    iterator_tuple iter_tuple = make_tuple(first, counting_iterator_t<IndexType>(0));

    zip_iterator   begin  = make_zip_iterator(iter_tuple);
    two_pairs_type result = __extrema::extrema(policy,
                                               transform_t(begin, duplicate_t()),
                                               num_items,
                                               arg_minmax_t(binary_pred),
                                               (two_pairs_type*)(NULL));

    ret = thrust::make_pair(first + get<1>(get<0>(result)), first + get<1>(get<1>(result)));
#else // __THRUST_HAS_HIPRT__
    ret = thrust::minmax_element(cvt_to_seq(derived_cast(policy)), first, last, binary_pred);
#endif // __THRUST_HAS_HIPRT__
    return ret;
}

template <class Derived, class ItemsIt>
pair<ItemsIt, ItemsIt> THRUST_HIP_FUNCTION
minmax_element(execution_policy<Derived>& policy,
               ItemsIt                    first,
               ItemsIt                    last)
{
    typedef typename iterator_value<ItemsIt>::type value_type;
    return hip_rocprim::minmax_element(policy, first, last, less<value_type>());
}

} // namespace hip_rocprim
THRUST_END_NS
#endif
