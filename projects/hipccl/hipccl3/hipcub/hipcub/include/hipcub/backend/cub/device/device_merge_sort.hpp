/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2023, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_CUB_DEVICE_DEVICE_MERGE_SORT_HPP_
#define HIPCUB_CUB_DEVICE_DEVICE_MERGE_SORT_HPP_

#include "../../../config.hpp"

#include <cub/device/device_merge_sort.cuh>

BEGIN_HIPCUB_NAMESPACE

struct DeviceMergeSort
{
    template<typename KeyIteratorT, typename ValueIteratorT, typename OffsetT, typename CompareOpT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t SortPairs(void *         d_temp_storage,
                                                      std::size_t &  temp_storage_bytes,
                                                      KeyIteratorT   d_keys,
                                                      ValueIteratorT d_items,
                                                      OffsetT        num_items,
                                                      CompareOpT     compare_op,
                                                      hipStream_t    stream            = 0,
                                                      bool           debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(::cub::DeviceMergeSort::SortPairs(d_temp_storage,
                                                                        temp_storage_bytes,
                                                                        d_keys,
                                                                        d_items,
                                                                        num_items,
                                                                        compare_op,
                                                                        stream));
    }

    template<typename KeyInputIteratorT,
             typename ValueInputIteratorT,
             typename KeyIteratorT,
             typename ValueIteratorT,
             typename OffsetT,
             typename CompareOpT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t SortPairsCopy(void *              d_temp_storage,
                                                          std::size_t &       temp_storage_bytes,
                                                          KeyInputIteratorT   d_input_keys,
                                                          ValueInputIteratorT d_input_items,
                                                          KeyIteratorT        d_output_keys,
                                                          ValueIteratorT      d_output_items,
                                                          OffsetT             num_items,
                                                          CompareOpT          compare_op,
                                                          hipStream_t         stream = 0,
                                                          bool debug_synchronous     = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(::cub::DeviceMergeSort::SortPairsCopy(d_temp_storage,
                                                                            temp_storage_bytes,
                                                                            d_input_keys,
                                                                            d_input_items,
                                                                            d_output_keys,
                                                                            d_output_items,
                                                                            num_items,
                                                                            compare_op,
                                                                            stream));
    }

    template<typename KeyIteratorT, typename OffsetT, typename CompareOpT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t SortKeys(void *        d_temp_storage,
                                                     std::size_t & temp_storage_bytes,
                                                     KeyIteratorT  d_keys,
                                                     OffsetT       num_items,
                                                     CompareOpT    compare_op,
                                                     hipStream_t   stream            = 0,
                                                     bool          debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(::cub::DeviceMergeSort::SortKeys(d_temp_storage,
                                                                       temp_storage_bytes,
                                                                       d_keys,
                                                                       num_items,
                                                                       compare_op,
                                                                       stream));
    }

    template<typename KeyInputIteratorT,
             typename KeyIteratorT,
             typename OffsetT,
             typename CompareOpT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t SortKeysCopy(void * d_temp_storage,
                                                         std::size_t & temp_storage_bytes,
                                                         KeyInputIteratorT d_input_keys,
                                                         KeyIteratorT      d_output_keys,
                                                         OffsetT           num_items,
                                                         CompareOpT        compare_op,
                                                         hipStream_t stream = 0,
                                                         bool debug_synchronous = false)

    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(::cub::DeviceMergeSort::SortKeysCopy(d_temp_storage,
                                                                           temp_storage_bytes,
                                                                           d_input_keys,
                                                                           d_output_keys,
                                                                           num_items,
                                                                           compare_op,
                                                                           stream));
    }

    template<typename KeyIteratorT, typename ValueIteratorT, typename OffsetT, typename CompareOpT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t StableSortPairs(void*          d_temp_storage,
                                                              std::size_t&   temp_storage_bytes,
                                                              KeyIteratorT   d_keys,
                                                              ValueIteratorT d_items,
                                                              OffsetT        num_items,
                                                              CompareOpT     compare_op,
                                                              hipStream_t    stream  = 0,
                                                              bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(::cub::DeviceMergeSort::StableSortPairs(d_temp_storage,
                                                                              temp_storage_bytes,
                                                                              d_keys,
                                                                              d_items,
                                                                              num_items,
                                                                              compare_op,
                                                                              stream));
    }

    template<typename KeyIteratorT, typename OffsetT, typename CompareOpT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t StableSortKeys(void *        d_temp_storage,
                                                             std::size_t & temp_storage_bytes,
                                                             KeyIteratorT  d_keys,
                                                             OffsetT       num_items,
                                                             CompareOpT    compare_op,
                                                             hipStream_t   stream   = 0,
                                                             bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(::cub::DeviceMergeSort::StableSortKeys(d_temp_storage,
                                                                             temp_storage_bytes,
                                                                             d_keys,
                                                                             num_items,
                                                                             compare_op,
                                                                             stream));
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_DEVICE_DEVICE_MERGE_SORT_HPP_
