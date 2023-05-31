/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2020, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_REDUCE_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_REDUCE_HPP_

#include <limits>
#include <iterator>

#include <hip/hip_fp16.h> // __half
#include <hip/hip_bfloat16.h> // hip_bfloat16

#include "../../../config.hpp"
#include "../iterator/arg_index_input_iterator.hpp"
#include "../thread/thread_operators.hpp"

#include <rocprim/device/device_reduce.hpp>
#include <rocprim/device/device_reduce_by_key.hpp>

BEGIN_HIPCUB_NAMESPACE
namespace detail
{

template<typename T>
HIPCUB_HOST_DEVICE T set_half_bits(uint16_t value)
{
    T              half_value{};
    unsigned char* char_representation = reinterpret_cast<unsigned char*>(&half_value);
    char_representation[0]             = value;
    char_representation[1]             = value >> 8;
    return half_value;
}

template<class T>
HIPCUB_HOST_DEVICE inline T get_lowest_value()
{
    return std::numeric_limits<T>::lowest();
}

template<>
HIPCUB_HOST_DEVICE inline __half get_lowest_value<__half>()
{
    // smallest normal value (not subnormal): 1 11110 1111111111
    return set_half_bits<__half>(0xfbff);
}

template<>
HIPCUB_HOST_DEVICE inline hip_bfloat16 get_lowest_value<hip_bfloat16>()
{
    // smallest normal value (not subnormal): 1 11111110 1111111
    return set_half_bits<hip_bfloat16>(0xff7f);
}

template<class T>
HIPCUB_HOST_DEVICE inline T get_max_value()
{
    return std::numeric_limits<T>::max();
}

template<>
HIPCUB_HOST_DEVICE inline __half get_max_value<__half>()
{
    // largest normal value (not subnormal): 0 11110 1111111111
    return set_half_bits<__half>(0x7bff);
}

template<>
HIPCUB_HOST_DEVICE inline hip_bfloat16 get_max_value<hip_bfloat16>()
{
    // largest normal value (not subnormal): 0 11111110 1111111
    return set_half_bits<hip_bfloat16>(0x7f7f);
}

/// Same as \p get_lowest_value, but includes negative infinity for floating-point types.
template<class T>
inline auto get_lowest_special_value() ->
    typename std::enable_if_t<!rocprim::is_floating_point<T>::value, T>
{
    return get_lowest_value<T>();
}

/// Same as \p get_lowest_value, but includes negative infinity for floating-point types.
template<class T>
inline auto get_lowest_special_value() ->
    typename std::enable_if_t<rocprim::is_floating_point<T>::value, T>
{
    return -std::numeric_limits<T>::infinity();
}

template<>
inline __half get_lowest_special_value<__half>()
{
    // negative infinity: 1 11111 0000000000
    return set_half_bits<__half>(0xfc00);
}

template<>
inline hip_bfloat16 get_lowest_special_value<hip_bfloat16>()
{
    // negative infinity: 1 11111111 0000000
    return set_half_bits<hip_bfloat16>(0xff80);
}

/// Same as \p get_max_value, but includes positive infinity for floating-point types.
template<typename T>
inline auto get_max_special_value() ->
    typename std::enable_if_t<!rocprim::is_floating_point<T>::value, T>
{
    return get_max_value<T>();
}

/// Same as \p get_max_value, but includes positive infinity for floating-point types.
template<typename T>
inline auto get_max_special_value() ->
    typename std::enable_if_t<rocprim::is_floating_point<T>::value, T>
{
    return std::numeric_limits<T>::infinity();
}

template<>
inline __half get_max_special_value<__half>()
{
    // positive infinity: 0 11111 0000000000
    return set_half_bits<__half>(0x7c00);
}

template<>
inline hip_bfloat16 get_max_special_value<hip_bfloat16>()
{
    // positive infinity: 0 11111111 0000000
    return set_half_bits<hip_bfloat16>(0x7f80);
}

} // end detail namespace

class DeviceReduce
{
public:
    template <
        typename InputIteratorT,
        typename OutputIteratorT,
        typename ReduceOpT,
        typename T
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t Reduce(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      InputIteratorT d_in,
                      OutputIteratorT d_out,
                      int num_items,
                      ReduceOpT reduction_op,
                      T init,
                      hipStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        return ::rocprim::reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, init, num_items,
            ::hipcub::detail::convert_result_type<InputIteratorT, OutputIteratorT>(reduction_op),
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t Sum(void *d_temp_storage,
                   size_t &temp_storage_bytes,
                   InputIteratorT d_in,
                   OutputIteratorT d_out,
                   int num_items,
                   hipStream_t stream = 0,
                   bool debug_synchronous = false)
    {
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, num_items, ::hipcub::Sum(), T(0),
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t Min(void *d_temp_storage,
                   size_t &temp_storage_bytes,
                   InputIteratorT d_in,
                   OutputIteratorT d_out,
                   int num_items,
                   hipStream_t stream = 0,
                   bool debug_synchronous = false)
    {
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, num_items, ::hipcub::Min(), detail::get_max_value<T>(),
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t ArgMin(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      InputIteratorT d_in,
                      OutputIteratorT d_out,
                      int num_items,
                      hipStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        using OffsetT = int;
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        using O = typename std::iterator_traits<OutputIteratorT>::value_type;
        using OutputTupleT =
            typename std::conditional<
                std::is_same<O, void>::value,
                KeyValuePair<OffsetT, T>,
                O
            >::type;

        using OutputValueT = typename OutputTupleT::Value;
        using IteratorT = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

        IteratorT d_indexed_in(d_in);
        // Empty inputs produce a specific value dictated by CUB's API: numeric_limits::max.
        // When not empty, using this value as initial is invalid and +infinity is used instead.
        OutputTupleT init(1,
                          num_items > 0 ? detail::get_max_special_value<T>()
                                        : detail::get_max_value<T>());

        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_indexed_in, d_out, num_items, ::hipcub::ArgMin(), init,
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t Max(void *d_temp_storage,
                   size_t &temp_storage_bytes,
                   InputIteratorT d_in,
                   OutputIteratorT d_out,
                   int num_items,
                   hipStream_t stream = 0,
                   bool debug_synchronous = false)
    {
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, num_items, ::hipcub::Max(), detail::get_lowest_value<T>(),
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t ArgMax(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      InputIteratorT d_in,
                      OutputIteratorT d_out,
                      int num_items,
                      hipStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        using OffsetT = int;
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        using O = typename std::iterator_traits<OutputIteratorT>::value_type;
        using OutputTupleT =
            typename std::conditional<
                std::is_same<O, void>::value,
                KeyValuePair<OffsetT, T>,
                O
            >::type;

        using OutputValueT = typename OutputTupleT::Value;
        using IteratorT = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

        IteratorT d_indexed_in(d_in);
        // Empty inputs produce a specific value dictated by CUB's API: numeric_limits::lowest.
        // When not empty, using this value as initial is invalid and -infinity is used instead.
        const OutputTupleT init(1,
                                num_items > 0 ? detail::get_lowest_special_value<T>()
                                              : detail::get_lowest_value<T>());

        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_indexed_in, d_out, num_items, ::hipcub::ArgMax(), init,
            stream, debug_synchronous
        );
    }

    template<
        typename KeysInputIteratorT,
        typename UniqueOutputIteratorT,
        typename ValuesInputIteratorT,
        typename AggregatesOutputIteratorT,
        typename NumRunsOutputIteratorT,
        typename ReductionOpT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t ReduceByKey(void * d_temp_storage,
                           size_t& temp_storage_bytes,
                           KeysInputIteratorT d_keys_in,
                           UniqueOutputIteratorT d_unique_out,
                           ValuesInputIteratorT d_values_in,
                           AggregatesOutputIteratorT d_aggregates_out,
                           NumRunsOutputIteratorT d_num_runs_out,
                           ReductionOpT reduction_op,
                           int num_items,
                           hipStream_t stream = 0,
                           bool debug_synchronous = false)
    {
        using key_compare_op =
            ::rocprim::equal_to<typename std::iterator_traits<KeysInputIteratorT>::value_type>;
        return ::rocprim::reduce_by_key(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_values_in, num_items,
            d_unique_out, d_aggregates_out, d_num_runs_out,
            ::hipcub::detail::convert_result_type<ValuesInputIteratorT, AggregatesOutputIteratorT>(reduction_op),
            key_compare_op(),
            stream, debug_synchronous
        );
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_REDUCE_HPP_
