/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2022-2025, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_ADJACENT_DIFFERENCE_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_ADJACENT_DIFFERENCE_HPP_

#include "../../../config.hpp"
#include "../../../util_deprecated.hpp"

#include <hipcub/thread/thread_operators.hpp>
#include <rocprim/device/device_adjacent_difference.hpp> // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE

struct DeviceAdjacentDifference
{
    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename DifferenceOpT = ::hipcub::Difference,
             typename NumItemsT     = std::uint32_t>
    static HIPCUB_RUNTIME_FUNCTION hipError_t SubtractLeftCopy(void*           d_temp_storage,
                                                               std::size_t&    temp_storage_bytes,
                                                               InputIteratorT  d_input,
                                                               OutputIteratorT d_output,
                                                               NumItemsT       num_items,
                                                               DifferenceOpT   difference_op = {},
                                                               hipStream_t     stream        = 0)
    {
        return ::rocprim::adjacent_difference(d_temp_storage,
                                              temp_storage_bytes,
                                              d_input,
                                              d_output,
                                              num_items,
                                              difference_op,
                                              stream,
                                              HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename DifferenceOpT = ::hipcub::Difference,
             typename NumItemsT     = std::uint32_t>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS static HIPCUB_RUNTIME_FUNCTION hipError_t
        SubtractLeftCopy(void*           d_temp_storage,
                         std::size_t&    temp_storage_bytes,
                         InputIteratorT  d_input,
                         OutputIteratorT d_output,
                         NumItemsT       num_items,
                         DifferenceOpT   difference_op,
                         hipStream_t     stream,
                         bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return SubtractLeftCopy(d_temp_storage,
                                temp_storage_bytes,
                                d_input,
                                d_output,
                                num_items,
                                difference_op,
                                stream);
    }

    template<typename RandomAccessIteratorT,
             typename DifferenceOpT = ::hipcub::Difference,
             typename NumItemsT     = std::uint32_t>
    static HIPCUB_RUNTIME_FUNCTION hipError_t SubtractLeft(void*                 d_temp_storage,
                                                           std::size_t&          temp_storage_bytes,
                                                           RandomAccessIteratorT d_input,
                                                           NumItemsT             num_items,
                                                           DifferenceOpT         difference_op = {},
                                                           hipStream_t           stream        = 0)
    {
        return ::rocprim::adjacent_difference_inplace(d_temp_storage,
                                                      temp_storage_bytes,
                                                      d_input,
                                                      num_items,
                                                      difference_op,
                                                      stream,
                                                      HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename RandomAccessIteratorT,
             typename DifferenceOpT = ::hipcub::Difference,
             typename NumItemsT     = std::uint32_t>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS static HIPCUB_RUNTIME_FUNCTION hipError_t
        SubtractLeft(void*                 d_temp_storage,
                     std::size_t&          temp_storage_bytes,
                     RandomAccessIteratorT d_input,
                     NumItemsT             num_items,
                     DifferenceOpT         difference_op,
                     hipStream_t           stream,
                     bool                  debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return SubtractLeft(d_temp_storage,
                            temp_storage_bytes,
                            d_input,
                            num_items,
                            difference_op,
                            stream);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename DifferenceOpT = ::hipcub::Difference,
             typename NumItemsT     = std::uint32_t>
    static HIPCUB_RUNTIME_FUNCTION hipError_t SubtractRightCopy(void*           d_temp_storage,
                                                                std::size_t&    temp_storage_bytes,
                                                                InputIteratorT  d_input,
                                                                OutputIteratorT d_output,
                                                                NumItemsT       num_items,
                                                                DifferenceOpT   difference_op = {},
                                                                hipStream_t     stream        = 0)
    {
        return ::rocprim::adjacent_difference_right(d_temp_storage,
                                                    temp_storage_bytes,
                                                    d_input,
                                                    d_output,
                                                    num_items,
                                                    difference_op,
                                                    stream,
                                                    HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename DifferenceOpT = ::hipcub::Difference,
             typename NumItemsT     = std::uint32_t>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS static HIPCUB_RUNTIME_FUNCTION hipError_t
        SubtractRightCopy(void*           d_temp_storage,
                          std::size_t&    temp_storage_bytes,
                          InputIteratorT  d_input,
                          OutputIteratorT d_output,
                          NumItemsT       num_items,
                          DifferenceOpT   difference_op,
                          hipStream_t     stream,
                          bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return SubtractRightCopy(d_temp_storage,
                                 temp_storage_bytes,
                                 d_input,
                                 d_output,
                                 num_items,
                                 difference_op,
                                 stream);
    }

    template<typename RandomAccessIteratorT,
             typename DifferenceOpT = ::hipcub::Difference,
             typename NumItemsT     = std::uint32_t>
    static HIPCUB_RUNTIME_FUNCTION hipError_t SubtractRight(void*        d_temp_storage,
                                                            std::size_t& temp_storage_bytes,
                                                            RandomAccessIteratorT d_input,
                                                            NumItemsT             num_items,
                                                            DifferenceOpT difference_op = {},
                                                            hipStream_t   stream        = 0)
    {
        return ::rocprim::adjacent_difference_right_inplace(d_temp_storage,
                                                            temp_storage_bytes,
                                                            d_input,
                                                            num_items,
                                                            difference_op,
                                                            stream,
                                                            HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename RandomAccessIteratorT,
             typename DifferenceOpT = ::hipcub::Difference,
             typename NumItemsT     = std::uint32_t>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS static HIPCUB_RUNTIME_FUNCTION hipError_t
        SubtractRight(void*                 d_temp_storage,
                      std::size_t&          temp_storage_bytes,
                      RandomAccessIteratorT d_input,
                      NumItemsT             num_items,
                      DifferenceOpT         difference_op,
                      hipStream_t           stream,
                      bool                  debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return SubtractRight(d_temp_storage,
                             temp_storage_bytes,
                             d_input,
                             num_items,
                             difference_op,
                             stream);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_ADJACENT_DIFFERENCE_HPP_
