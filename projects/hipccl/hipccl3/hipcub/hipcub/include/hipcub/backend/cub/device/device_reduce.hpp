/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2026, Advanced Micro Devices, Inc.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_CUB_DEVICE_DEVICE_REDUCE_HPP_
#define HIPCUB_CUB_DEVICE_DEVICE_REDUCE_HPP_

#include "../../../config.hpp"
#include "../../../util_deprecated.hpp"
#include "../../../util_type.hpp"

#include <cub/device/device_reduce.cuh> // IWYU pragma: export

#include <cuda/std/limits>

#include <iterator>

BEGIN_HIPCUB_NAMESPACE

class DeviceReduce
{
public:
    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename ReduceOpT,
             typename T,
             typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t Reduce(void*           d_temp_storage,
                                                     size_t&         temp_storage_bytes,
                                                     InputIteratorT  d_in,
                                                     OutputIteratorT d_out,
                                                     NumItemsT       num_items,
                                                     ReduceOpT       reduction_op,
                                                     T               init,
                                                     hipStream_t     stream = 0)
    {
        return hipCUDAErrorTohipError(::cub::DeviceReduce::Reduce(d_temp_storage,
                                                                  temp_storage_bytes,
                                                                  d_in,
                                                                  d_out,
                                                                  num_items,
                                                                  reduction_op,
                                                                  init,
                                                                  stream));
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t Sum(void*           d_temp_storage,
                                                  size_t&         temp_storage_bytes,
                                                  InputIteratorT  d_in,
                                                  OutputIteratorT d_out,
                                                  NumItemsT       num_items,
                                                  hipStream_t     stream = 0)
    {
        return hipCUDAErrorTohipError(::cub::DeviceReduce::Sum(d_temp_storage,
                                                               temp_storage_bytes,
                                                               d_in,
                                                               d_out,
                                                               num_items,
                                                               stream));
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t Min(void*           d_temp_storage,
                                                  size_t&         temp_storage_bytes,
                                                  InputIteratorT  d_in,
                                                  OutputIteratorT d_out,
                                                  NumItemsT       num_items,
                                                  hipStream_t     stream = 0)
    {
        return hipCUDAErrorTohipError(::cub::DeviceReduce::Min(d_temp_storage,
                                                               temp_storage_bytes,
                                                               d_in,
                                                               d_out,
                                                               num_items,
                                                               stream));
    }

    template<typename InputIteratorT,
             typename ExtremumOutIteratorT,
             typename IndexOutIteratorT,
             typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ArgMin(void*                d_temp_storage,
                             size_t&              temp_storage_bytes,
                             InputIteratorT       d_in,
                             ExtremumOutIteratorT d_min_out,
                             IndexOutIteratorT    d_index_out,
                             NumItemsT            num_items,
                             hipStream_t          stream = 0)
    {
        using value_type = ::hipcub::detail::it_value_t<InputIteratorT>;
        using index_type = ::hipcub::detail::it_value_t<IndexOutIteratorT>;

        // CUB handles zero-length inputs in its internal dispatch layer.
        // That behavior must be reproduced manually, so this case is handled here.
        if(num_items == 0)
        {
            if(d_temp_storage == nullptr)
            {
                temp_storage_bytes = sizeof(int);
                return hipSuccess;
            }

            value_type init_value = ::cuda::std::numeric_limits<value_type>::max();
            index_type init_index = 1;

            hipError_t e1 = hipMemcpyAsync(d_min_out,
                                           &init_value,
                                           sizeof(value_type),
                                           hipMemcpyHostToDevice,
                                           stream);

            hipError_t e2 = hipMemcpyAsync(d_index_out,
                                           &init_index,
                                           sizeof(index_type),
                                           hipMemcpyHostToDevice,
                                           stream);

            return (e1 != hipSuccess ? e1 : e2);
        }

        return hipCUDAErrorTohipError(::cub::DeviceReduce::ArgMin(d_temp_storage,
                                                                  temp_storage_bytes,
                                                                  d_in,
                                                                  d_min_out,
                                                                  d_index_out,
                                                                  num_items,
                                                                  stream));
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
    HIPCUB_DEPRECATED_BECAUSE(
        "CUB has superseded this interface in favor of the ArgMin interface "
        "that takes two separate "
        "iterators: one iterator to which the extremum is written and another "
        "iterator to which the "
        "index of the found extremum is written. ")
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ArgMin(void*           d_temp_storage,
                             size_t&         temp_storage_bytes,
                             InputIteratorT  d_in,
                             OutputIteratorT d_out,
                             NumItemsT       num_items,
                             hipStream_t     stream = 0)
    {
        using pair_type  = ::hipcub::detail::it_value_t<OutputIteratorT>;
        using value_type = decltype(pair_type::value);
        using index_type = decltype(pair_type::key);

        if(num_items == 0)
        {
            if(d_temp_storage == nullptr)
            {
                temp_storage_bytes = sizeof(int);
                return hipSuccess;
            }

            pair_type init;
            init.key   = static_cast<index_type>(1);
            init.value = ::cuda::std::numeric_limits<value_type>::max();

            return hipMemcpyAsync(d_out, &init, sizeof(pair_type), hipMemcpyHostToDevice, stream);
        }

        pair_type* out_pair = reinterpret_cast<pair_type*>(d_out);

        value_type* d_min_out   = &(out_pair->value);
        index_type* d_index_out = &(out_pair->key);

        _CCCL_SUPPRESS_DEPRECATED_PUSH
        auto status = ArgMin(d_temp_storage,
                             temp_storage_bytes,
                             d_in,
                             d_min_out,
                             d_index_out,
                             static_cast<index_type>(num_items),
                             stream);
        _CCCL_SUPPRESS_DEPRECATED_POP
        return status;
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t Max(void*           d_temp_storage,
                                                  size_t&         temp_storage_bytes,
                                                  InputIteratorT  d_in,
                                                  OutputIteratorT d_out,
                                                  NumItemsT       num_items,
                                                  hipStream_t     stream = 0)
    {
        return hipCUDAErrorTohipError(::cub::DeviceReduce::Max(d_temp_storage,
                                                               temp_storage_bytes,
                                                               d_in,
                                                               d_out,
                                                               num_items,
                                                               stream));
    }

    template<typename InputIteratorT,
             typename ExtremumOutIteratorT,
             typename IndexOutIteratorT,
             typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ArgMax(void*                d_temp_storage,
                             size_t&              temp_storage_bytes,
                             InputIteratorT       d_in,
                             ExtremumOutIteratorT d_max_out,
                             IndexOutIteratorT    d_index_out,
                             NumItemsT            num_items,
                             hipStream_t          stream = 0)
    {
        using value_type = ::hipcub::detail::it_value_t<InputIteratorT>;
        using index_type = ::hipcub::detail::it_value_t<IndexOutIteratorT>;

        // CUB documentation claims zero-length inputs initialize with numeric_limits<T>::max(),
        // but the actual CUB implementation uses numeric_limits<T>::lowest().
        // hipCUB matches the implementation.

        if(num_items == 0)
        {
            if(d_temp_storage == nullptr)
            {
                temp_storage_bytes = sizeof(int);
                return hipSuccess;
            }

            value_type init_value = ::cuda::std::numeric_limits<value_type>::lowest();
            index_type init_index = 1; // hipCUB 1-based index

            hipError_t e1 = hipMemcpyAsync(d_max_out,
                                           &init_value,
                                           sizeof(value_type),
                                           hipMemcpyHostToDevice,
                                           stream);

            hipError_t e2 = hipMemcpyAsync(d_index_out,
                                           &init_index,
                                           sizeof(index_type),
                                           hipMemcpyHostToDevice,
                                           stream);

            return (e1 != hipSuccess ? e1 : e2);
        }

        return hipCUDAErrorTohipError(::cub::DeviceReduce::ArgMax(d_temp_storage,
                                                                  temp_storage_bytes,
                                                                  d_in,
                                                                  d_max_out,
                                                                  d_index_out,
                                                                  num_items,
                                                                  stream));
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
    HIPCUB_DEPRECATED_BECAUSE("CUB has superseded this interface in favor of the ArgMax interface "
                              "that takes two separate iterators: one iterator to which the "
                              "extremum is written and another "
                              "iterator to which the index of the found extremum is written. ")
HIPCUB_RUNTIME_FUNCTION
    static hipError_t ArgMax(void*           d_temp_storage,
                             size_t&         temp_storage_bytes,
                             InputIteratorT  d_in,
                             OutputIteratorT d_out,
                             NumItemsT       num_items,
                             hipStream_t     stream = 0)
    {
        using pair_type  = ::hipcub::detail::it_value_t<OutputIteratorT>;
        using value_type = decltype(pair_type::value);
        using index_type = decltype(pair_type::key);

        if(num_items == 0)
        {
            if(d_temp_storage == nullptr)
            {
                temp_storage_bytes = sizeof(int);
                return hipSuccess;
            }

            pair_type init;
            init.key   = static_cast<index_type>(1);
            init.value = ::cuda::std::numeric_limits<value_type>::lowest();

            return hipMemcpyAsync(d_out, &init, sizeof(pair_type), hipMemcpyHostToDevice, stream);
        }

        pair_type* out_pair = reinterpret_cast<pair_type*>(d_out);

        value_type* d_max_out   = &(out_pair->value);
        index_type* d_index_out = &(out_pair->key);

        _CCCL_SUPPRESS_DEPRECATED_PUSH
        auto status = ArgMax(d_temp_storage,
                             temp_storage_bytes,
                             d_in,
                             d_max_out,
                             d_index_out,
                             static_cast<index_type>(num_items),
                             stream);
        _CCCL_SUPPRESS_DEPRECATED_POP
        return status;
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename ReductionOpT,
             typename TransformOpT,
             typename T,
             typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t TransformReduce(void*           d_temp_storage,
                                      size_t&         temp_storage_bytes,
                                      InputIteratorT  d_in,
                                      OutputIteratorT d_out,
                                      NumItemsT       num_items,
                                      ReductionOpT    reduction_op,
                                      TransformOpT    transform_op,
                                      T               init,
                                      hipStream_t     stream = 0)
    {
        return hipCUDAErrorTohipError(::cub::DeviceReduce::TransformReduce(d_temp_storage,
                                                                           temp_storage_bytes,
                                                                           d_in,
                                                                           d_out,
                                                                           num_items,
                                                                           reduction_op,
                                                                           transform_op,
                                                                           init,
                                                                           stream));
    }

    template<typename KeysInputIteratorT,
             typename UniqueOutputIteratorT,
             typename ValuesInputIteratorT,
             typename AggregatesOutputIteratorT,
             typename NumRunsOutputIteratorT,
             typename ReductionOpT,
             typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t
        ReduceByKey(void*                     d_temp_storage,
                    size_t&                   temp_storage_bytes,
                    KeysInputIteratorT        d_keys_in,
                    UniqueOutputIteratorT     d_unique_out,
                    ValuesInputIteratorT      d_values_in,
                    AggregatesOutputIteratorT d_aggregates_out,
                    NumRunsOutputIteratorT    d_num_runs_out,
                    ReductionOpT              reduction_op,
                    NumItemsT                 num_items,
                    hipStream_t               stream = 0)
    {
        return hipCUDAErrorTohipError(::cub::DeviceReduce::ReduceByKey(d_temp_storage,
                                                                       temp_storage_bytes,
                                                                       d_keys_in,
                                                                       d_unique_out,
                                                                       d_values_in,
                                                                       d_aggregates_out,
                                                                       d_num_runs_out,
                                                                       reduction_op,
                                                                       num_items,
                                                                       stream));
    }

private:
    template<typename ScalarOutputIt, typename T>
    struct value_only_pair_output_iterator
    {
        ScalarOutputIt out;
        using value_type = ::cub::KeyValuePair<int, T>;
        HIPCUB_HOST_DEVICE
        value_only_pair_output_iterator(ScalarOutputIt o)
            : out(o)
        {}
        HIPCUB_HOST_DEVICE
        value_only_pair_output_iterator& operator*()
        {
            return *this;
        }
        HIPCUB_HOST_DEVICE
        value_only_pair_output_iterator& operator=(value_type const& p)
        {
            *out = p.value;
            return *this;
        }
        HIPCUB_HOST_DEVICE
        value_only_pair_output_iterator& operator++()
        {
            ++out;
            return *this;
        }
        HIPCUB_HOST_DEVICE
        value_only_pair_output_iterator operator++(int)
        {
            value_only_pair_output_iterator tmp = *this;
            ++out;
            return tmp;
        }
    };
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_DEVICE_DEVICE_REDUCE_HPP_
