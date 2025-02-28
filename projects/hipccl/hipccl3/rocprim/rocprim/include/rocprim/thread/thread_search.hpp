/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021-2024, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef ROCPRIM_THREAD_THREAD_SEARCH_HPP_
#define ROCPRIM_THREAD_THREAD_SEARCH_HPP_

#include "../detail/merge_path.hpp"

#include "../config.hpp"
#include "../functional.hpp"

#include <iterator>

BEGIN_ROCPRIM_NAMESPACE

/// \defgroup thread_search Thread Search Functions
/// \ingroup threadmodule

/// \addtogroup thread_search
/// @{

/// \brief Computes the begin offsets into A and B for the specific \p diagonal. If the \p diagonal
/// is larger than the sum of the two lengths the first component of the result will be clamped to
/// be no larger than \p a_len and the sum of the two components will be equal to \p diagonal.
template<class AIteratorT,
         class BIteratorT,
         class OffsetT,
         class CoordinateT,
         class BinaryFunction
         = rocprim::less<typename std::iterator_traits<AIteratorT>::value_type>>
ROCPRIM_HOST_DEVICE inline void merge_path_search(OffsetT        diagonal,
                                                  AIteratorT     a,
                                                  BIteratorT     b,
                                                  OffsetT        a_len,
                                                  OffsetT        b_len,
                                                  CoordinateT&   path_coordinate,
                                                  BinaryFunction compare_function
                                                  = BinaryFunction())
{
    OffsetT split_min = rocprim::detail::merge_path(a, b, a_len, b_len, diagonal, compare_function);
    path_coordinate.x = ::rocprim::min(split_min, a_len);
    path_coordinate.y = diagonal - split_min;
}

/// \brief Returns the offset of the first value within \p input which does not compare less than \p val
/// \tparam InputIteratorT   <b>[inferred]</b> Type of iterator for the input data to be searched
/// \tparam OffsetT          <b>[inferred]</b> The data type of num_items
/// \tparam T                <b>[inferred]</b> The data type of the input sequence elements
/// \param input     [in]    Input sequence
/// \param num_items [in]    Input sequence length
/// \param val       [in]    Search Key
/// \return                  Offset at which val was found
template<typename InputIteratorT, typename OffsetT, typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE OffsetT lower_bound(InputIteratorT input, OffsetT num_items, T val)
{
    OffsetT retval = 0;
    while(num_items > 0)
    {
        OffsetT half = num_items >> 1;
        if(input[retval + half] < val)
        {
            retval    = retval + (half + 1);
            num_items = num_items - (half + 1);
        }
        else
        {
            num_items = half;
        }
    }

    return retval;
}

/// \brief Returns the offset of the first value within \p input which compares greater than \p val
/// \tparam InputIteratorT   <b>[inferred]</b> Type of iterator for the input data to be searched
/// \tparam OffsetT          <b>[inferred]</b> The data type of num_items
/// \tparam T                <b>[inferred]</b> The data type of the input sequence elements
/// \param input     [in]    Input sequence
/// \param num_items [in]    Input sequence length
/// \param val       [in]    Search Key
/// \return                  Offset at which val was found
template<typename InputIteratorT, typename OffsetT, typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE OffsetT upper_bound(InputIteratorT input, OffsetT num_items, T val)
{
    OffsetT retval = 0;
    while(num_items > 0)
    {
        OffsetT half = num_items >> 1;
        if(val < input[retval + half])
        {
            num_items = half;
        }
        else
        {
            retval    = retval + (half + 1);
            num_items = num_items - (half + 1);
        }
    }

    return retval;
}

/// \brief Returns the offset of the first value within \p input which compares greater than \p val
/// computed as a statically unrolled loop
/// \tparam MaxNumItems      The maximum number of items.
/// \tparam InputIteratorT   <b>[inferred]</b> Type of iterator for the input data to be searched
/// \tparam OffsetT          <b>[inferred]</b> The data type of num_items
/// \tparam T                <b>[inferred]</b> The data type of the input sequence elements
/// \param input     [in]    Input sequence
/// \param num_items [in]    Input sequence length
/// \param val       [in]    Search Key
/// \return                  Offset at which val was found
template<int MaxNumItems, typename InputIteratorT, typename OffsetT, typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE OffsetT static_upper_bound(InputIteratorT input,
                                                         OffsetT        num_items,
                                                         T              val)
{
    OffsetT lower_bound = 0;
    OffsetT upper_bound = num_items;
#pragma unroll
    for(int i = 0; i <= Log2<MaxNumItems>::VALUE; i++)
    {
        OffsetT mid = lower_bound + (upper_bound - lower_bound) / 2;
        mid         = rocprim::min(mid, num_items - 1);

        if(val < input[mid])
        {
            upper_bound = mid;
        }
        else
        {
            lower_bound = mid + 1;
        }
    }

    return lower_bound;
}

/// @}
// end of group thread_search

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_THREAD_THREAD_SEARCH_HPP_
