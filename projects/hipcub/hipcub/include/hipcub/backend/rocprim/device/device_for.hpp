/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2024-2025, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_FOR_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_FOR_HPP_

#include "../../../config.hpp"

#include "../iterator/counting_input_iterator.hpp"
#include "../iterator/discard_output_iterator.hpp"
#include "../thread/thread_operators.hpp"
#include "../util_mdspan.hpp"

#include <rocprim/device/device_transform.hpp> // IWYU pragma: export

#include <type_traits>

BEGIN_HIPCUB_NAMESPACE

namespace detail
{
namespace bulk
{
template<class T, class OpT>
struct OpWrapper
{
    OpT op;
    HIPCUB_HOST_DEVICE __forceinline__
    T   operator()(T const& a) const
    {
        // Make copies of operator and variable
        OpT op2 = op;
        T   b   = a;

        (void)op2(b);
        return b;
    }
};
} // namespace bulk

namespace for_each_in_extents
{
template<class OpT, class IndexType, class ExtentsIndexType, size_t... Extents>
struct OpWrapper
{
    using extents_type = ::hipcub::extents<ExtentsIndexType, Extents...>;
    OpT op;

    template<class IndexSeqType>
    struct OpDispatcher
    {};

    template<size_t... Rank>
    struct OpDispatcher<std::index_sequence<Rank...>>
    {
        __device__ __host__ __forceinline__
        void operator()(IndexType idx, OpT op) const
        {
            op(idx, OpWrapper::coordinate_at<Rank>(idx)...);
        }
    };

    template<size_t Rank>
        __device__ __host__ __forceinline__
    static auto coordinate_at(typename extents_type::index_type index)
    {
        using extent_index_type = typename extents_type::index_type;
        return static_cast<extent_index_type>(
            (index / extents_sub_size<Rank + 1, extents_type>::value)
            % extents_type::static_extent(Rank));
    }

    __device__ __host__ __forceinline__
    IndexType
        operator()(IndexType idx)
    {
        using seq_type = std::make_index_sequence<extents_type::rank()>;
        OpDispatcher<seq_type>{}(idx, op);
        return 0;
    }
};
} // namespace for_each_in_extents

} // namespace detail

struct DeviceFor
{
    template<class RandomAccessIteratorT, class OffsetT, class OpT>
    HIPCUB_RUNTIME_FUNCTION
    static auto ForEachN(RandomAccessIteratorT first,
                         OffsetT               num_items,
                         OpT                   op,
                         hipStream_t           stream = 0)
        -> std::enable_if_t<!std::is_assignable<decltype(*std::declval<RandomAccessIteratorT>()),
                                                typename std::iterator_traits<
                                                    RandomAccessIteratorT>::value_type>::value,
                            hipError_t>
    {
        using T              = typename std::iterator_traits<RandomAccessIteratorT>::value_type;
        using OutputIterator = typename rocprim::discard_iterator;
        detail::bulk::OpWrapper<T, OpT> wrapper_op = {op};

        OutputIterator output;

        return rocprim::transform(first,
                                  output,
                                  num_items,
                                  wrapper_op,
                                  stream,
                                  HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<class RandomAccessIteratorT, class OffsetT, class OpT>
    HIPCUB_RUNTIME_FUNCTION
    static auto
        ForEachN(RandomAccessIteratorT first, OffsetT num_items, OpT op, hipStream_t stream = 0)
            -> std::enable_if_t<std::is_assignable<decltype(*std::declval<RandomAccessIteratorT>()),
                                                   typename std::iterator_traits<
                                                       RandomAccessIteratorT>::value_type>::value,
                                hipError_t>
    {
        using T = typename std::iterator_traits<RandomAccessIteratorT>::value_type;

        detail::bulk::OpWrapper<T, OpT> wrapper_op = {op};

        return rocprim::transform(first,
                                  first,
                                  num_items,
                                  wrapper_op,
                                  stream,
                                  HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<class RandomAccessIteratorT, class OffsetT, class OpT>
HIPCUB_RUNTIME_FUNCTION
    static hipError_t ForEachN(void*                 d_temp_storage,
                               size_t&               temp_storage_bytes,
                               RandomAccessIteratorT first,
                               OffsetT               num_items,
                               OpT                   op,
                               hipStream_t           stream = 0)
    {
        if(d_temp_storage == nullptr)
        {
            temp_storage_bytes = 1;
            return hipSuccess;
        }

        return ForEachN(first, num_items, op, stream);
    }

    template<class RandomAccessIteratorT, class OffsetT, class OpT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t
        ForEachCopyN(RandomAccessIteratorT first, OffsetT num_items, OpT op, hipStream_t stream = 0)
    {
        return ForEachN(first, num_items, op, stream);
    }

    template<class RandomAccessIteratorT, class OffsetT, class OpT>
HIPCUB_RUNTIME_FUNCTION
    static hipError_t ForEachCopyN(void*                 d_temp_storage,
                                   size_t&               temp_storage_bytes,
                                   RandomAccessIteratorT first,
                                   OffsetT               num_items,
                                   OpT                   op,
                                   hipStream_t           stream = 0)
    {
        if(d_temp_storage == nullptr)
        {
            temp_storage_bytes = 1;
            return hipSuccess;
        }

        return ForEachCopyN(first, num_items, op, stream);
    }

    template<class RandomAccessIteratorT, class OpT>
HIPCUB_RUNTIME_FUNCTION
    static hipError_t ForEach(RandomAccessIteratorT first,
                              RandomAccessIteratorT last,
                              OpT                   op,
                              hipStream_t           stream = 0)
    {
        using offset_t = typename std::iterator_traits<RandomAccessIteratorT>::difference_type;
        const offset_t num_items = static_cast<offset_t>(std::distance(first, last));

        return ForEachN(first, num_items, op, stream);
    }

    template<class RandomAccessIteratorT, class OpT>
HIPCUB_RUNTIME_FUNCTION
    static hipError_t ForEach(void*                 d_temp_storage,
                              size_t&               temp_storage_bytes,
                              RandomAccessIteratorT first,
                              RandomAccessIteratorT last,
                              OpT                   op,
                              hipStream_t           stream = 0)
    {
        if(d_temp_storage == nullptr)
        {
            temp_storage_bytes = 1;
            return hipSuccess;
        }

        return ForEach(first, last, op, stream);
    }

    template<class RandomAccessIteratorT, class OpT>
HIPCUB_RUNTIME_FUNCTION
    static hipError_t ForEachCopy(void*                 d_temp_storage,
                                  size_t&               temp_storage_bytes,
                                  RandomAccessIteratorT first,
                                  RandomAccessIteratorT last,
                                  OpT                   op,
                                  hipStream_t           stream = 0)
    {
        if(d_temp_storage == nullptr)
        {
            temp_storage_bytes = 1;
            return hipSuccess;
        }

        return ForEachCopy(first, last, op, stream);
    }

    template<class RandomAccessIteratorT, class OpT>
HIPCUB_RUNTIME_FUNCTION
    static hipError_t ForEachCopy(RandomAccessIteratorT first,
                                  RandomAccessIteratorT last,
                                  OpT                   op,
                                  hipStream_t           stream = 0)
    {
        return ForEach(first, last, op, stream);
    }

    template<class ShapeT, class OpT>
HIPCUB_RUNTIME_FUNCTION
    static hipError_t Bulk(ShapeT shape, OpT op, hipStream_t stream = 0)
    {
        static_assert(std::is_integral<ShapeT>::value, "ShapeT must be an integral type");
        using InputIterator  = typename rocprim::counting_iterator<ShapeT>;
        using OutputIterator = typename rocprim::discard_iterator;
        detail::bulk::OpWrapper<ShapeT, OpT> wrapper_op = {op};

        InputIterator  input(ShapeT(0));
        OutputIterator output;

        return rocprim::transform(input,
                                  output,
                                  shape,
                                  wrapper_op,
                                  stream,
                                  HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<class ShapeT, class OpT>
HIPCUB_RUNTIME_FUNCTION
    static hipError_t Bulk(void*       d_temp_storage,
                           size_t&     temp_storage_bytes,
                           ShapeT      shape,
                           OpT         op,
                           hipStream_t stream = 0)
    {
        static_assert(std::is_integral<ShapeT>::value, "ShapeT must be an integral type");

        if(d_temp_storage == nullptr)
        {
            temp_storage_bytes = 1;
            return hipSuccess;
        }

        return Bulk(shape, op, stream);
    }

    /**
     * \brief Iterate through a multi-dimensional extents.
     * 
     * \tparam IndexType The index type to create the extents.
     * \tparam Extents The sizes of each extent.
     * \tparam OpT The function that will be applied for each iteration. The function should be callable
     * with arguments (IndexType, ExtentsIndexType...extents_idx). The size of extents_idx... should be 
     * the same as the rank of the the multi-dimensional extents.
     */
    template<class IndexType, size_t... Extents, typename OpT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ForEachInExtents(void*   d_temp_storage,
                                       size_t& temp_storage_bytes,
                                       const ::hipcub::extents<IndexType, Extents...>& extents,
                                       OpT                                             op,
                                       hipStream_t                                     stream = {})
    {
        // Index overlapping is checked by the `in_range` in the implementation in `::hipcub::extents`.
        // TODO: it's better to add a check here:
        // `static_assert(std::is_invocable::<OpT, IndexType, decltype(static_cast<extents_index_type>(Extents))...>::value,"check invocable.")`
        // But the `std::is_invocalble` does not work well here. Checked on a host environment with clang 19.1.0, and it works there. Since this
        // could be a compiler bug and invocability is also checked by the compiler. So, a "TODO" is left here.
        if(d_temp_storage == nullptr)
        {
            temp_storage_bytes = 1;
            return hipSuccess;
        }
        return ForEachInExtents(extents, op, stream);
    }

    /**
     * \brief Iterate through a multi-dimensional extents.
     * 
     * \tparam IndexType The index type to create the extents.
     * \tparam Extents The sizes of each extent.
     * \tparam OpT The function that will be applied for each iteration. The function should be callable
     * with arguments (IndexType, ExtentsIndexType...extents_idx). The size of extents_idx... should be 
     * the same as the rank of the the multi-dimensional extents.
     */
    template<class IndexType, size_t... Extents, typename OpT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ForEachInExtents(const ::hipcub::extents<IndexType, Extents...>&,
                                       OpT         op,
                                       hipStream_t stream = {})
    {
        using ext_type       = ::hipcub::extents<IndexType, Extents...>;
        using ext_index_type = typename ext_type::index_type;
        using wrapped_op_type
            = detail::for_each_in_extents::OpWrapper<OpT, IndexType, ext_index_type, Extents...>;

        // rocprim::counting_iterator only holds the index, not the data.
        using InputIterator = typename rocprim::counting_iterator<IndexType>;
        // We don't actually need the output, so we use rocprim::discard_iterator here as a placeholder.
        using OutputIterator = typename rocprim::discard_iterator;

        // How many times rocprim::transform will iterate.
        constexpr auto ext_size = ::hipcub::extents_size<ext_type>::value;

        InputIterator  input(IndexType(0)); // Initialize the input iterator, starting from 0.
        OutputIterator output;

        // `ForEachInExtents` only iterates over the extents on device and does not guarantee ordering.
        // We only need to invoke `$op` `$ext_size` times. Therefore, `rocprim::transform` is suitable.
        // In `ForEachInExtents`, the data isn’t necessarily needed, but even if it's needed, it may be embedded
        // in `OpT`. We only need to iterate exactly `$ext_size` times. Therefore, we use a `rocprim::counting_iterator`,
        // which provides only the index, and a `rocprim::discard_iterator` for output.
        return rocprim::transform(input,
                                  output,
                                  ext_size,
                                  wrapped_op_type{op},
                                  stream,
                                  HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_FOR_HPP_
