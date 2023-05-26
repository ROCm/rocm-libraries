/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021-2023, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef ROCPRIM_THREAD_THREAD_SCAN_HPP_
#define ROCPRIM_THREAD_THREAD_SCAN_HPP_

#include "../config.hpp"
#include "../functional.hpp"

#include <iterator>

BEGIN_ROCPRIM_NAMESPACE

/**
 * Computes the begin offsets into A and B for the specific diagonal
 */
template <
    typename AIteratorT,
    typename BIteratorT,
    typename OffsetT,
    typename CoordinateT>
ROCPRIM_HOST_DEVICE inline void merge_path_search(
    OffsetT         diagonal,
    AIteratorT      a,
    BIteratorT      b,
    OffsetT         a_len,
    OffsetT         b_len,
    CoordinateT&    path_coordinate)
{
    OffsetT split_min = ::rocprim::max(diagonal - b_len, static_cast<OffsetT>(0));
    OffsetT split_max = ::rocprim::min(diagonal, a_len);

    while (split_min < split_max)
    {
        OffsetT split_pivot = (split_min + split_max) >> 1;
        if (a[split_pivot] <= b[diagonal - split_pivot - 1])
        {
            // Move candidate split range up A, down B
            split_min = split_pivot + 1;
        }
        else
        {
            // Move candidate split range up B, down A
            split_max = split_pivot;
        }
    }

    path_coordinate.x = ::rocprim::min(split_min, a_len);
    path_coordinate.y = diagonal - split_min;
}




/// \brief Returns the offset of the first value within \p input which does not compare less than \p val
/// \tparam InputIteratorT   - <b>[inferred]</b> Type of iterator for the input data to be searched
/// \tparam OffsetT          - <b>[inferred]</b> The data type of num_items
/// \tparam T                - <b>[inferred]</b> The data type of the input sequence elements
/// \param input     [in]    - Input sequence
/// \param num_items [out]   - Input sequence length
/// \param val       [in]    - Search Key
/// \return                  - Offset at which val was found
template <
    typename InputIteratorT,
    typename OffsetT,
    typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE OffsetT lower_bound(
    InputIteratorT      input,
    OffsetT             num_items,
    T                   val)
{
    OffsetT retval = 0;
    while (num_items > 0)
    {
        OffsetT half = num_items >> 1;
        if (input[retval + half] < val)
        {
            retval = retval + (half + 1);
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
/// \tparam InputIteratorT   - <b>[inferred]</b> Type of iterator for the input data to be searched
/// \tparam OffsetT          - <b>[inferred]</b> The data type of num_items
/// \tparam T                - <b>[inferred]</b> The data type of the input sequence elements
/// \param input     [in]    - Input sequence
/// \param num_items [out]   - Input sequence length
/// \param val       [in]    - Search Key
/// \return                  - Offset at which val was found
template <
    typename InputIteratorT,
    typename OffsetT,
    typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE OffsetT upper_bound(
    InputIteratorT      input,              ///< [in] Input sequence
    OffsetT             num_items,          ///< [in] Input sequence length
    T                   val)                ///< [in] Search key
{
    OffsetT retval = 0;
    while (num_items > 0)
    {
        OffsetT half = num_items >> 1;
        if (val < input[retval + half])
        {
            num_items = half;
        }
        else
        {
            retval = retval + (half + 1);
            num_items = num_items - (half + 1);
        }
    }

    return retval;
}

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_THREAD_THREAD_SCAN_HPP_
