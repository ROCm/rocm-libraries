/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef ROCWMMA_REFERENCE_HPP
#define ROCWMMA_REFERENCE_HPP

#include <type_traits>

#include <rocwmma/internal/cross_lane_ops.hpp>
#include <rocwmma/internal/types.hpp>

namespace rocwmma
{

    template <typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    void gemm_CPU(uint32_t       m,
                  uint32_t       n,
                  uint32_t       k,
                  InputT const*  a,
                  InputT const*  b,
                  OutputT const* c,
                  OutputT*       d,
                  ComputeT       alpha,
                  ComputeT       beta);

    template <typename DataT>
    void
        dlrm_fwd_CPU(DataT const* input, DataT* output, uint32_t m, uint32_t k, uint32_t batchSize);

    template <typename DataT>
    void dlrm_bwd_CPU(DataT const* input,
                      DataT const* upstreamGrad,
                      DataT*       bottomMlpGrad,
                      DataT*       output,
                      uint32_t     m,
                      uint32_t     k,
                      uint32_t     batchSize);

    template <uint32_t ElementIdx,
              uint32_t GroupSize,
              uint32_t RowMask   = 0xF,
              uint32_t BankMask  = 0xF,
              bool     BoundCtrl = false>
    void cross_lane_bcast_CPU(uint32_t*       dataOut,
                              uint32_t const* dataIn,
                              uint32_t        elementCount,
                              uint32_t        fillVal = 0u);

    template <uint32_t BlockIdx,
              uint32_t GroupSize,
              uint32_t RowMask   = 0xF,
              uint32_t BankMask  = 0xF,
              bool     BoundCtrl = false>
    void cross_lane_block_bcast_CPU(uint32_t*       dataOut,
                                    uint32_t const* dataIn,
                                    uint32_t        elementCount,
                                    uint32_t        fillVal = 0u);

    template <uint32_t Select0,
              uint32_t Select1,
              uint32_t Select2,
              uint32_t Select3,
              uint32_t GroupSize,
              uint32_t RowMask   = 0xF,
              uint32_t BankMask  = 0xF,
              bool     BoundCtrl = false>
    void cross_lane_byte_blend_CPU(uint32_t*       dataOut,
                                   uint32_t const* src0,
                                   uint32_t const* src1,
                                   uint32_t        elementCount,
                                   uint32_t        fillVal = 0u);

    template <uint32_t GroupSize,
              uint32_t RowMask   = 0xF,
              uint32_t BankMask  = 0xF,
              bool     BoundCtrl = false>
    void cross_lane_wfall_bcast_CPU(uint32_t*       dataOut,
                                    uint32_t const* dataIn,
                                    uint32_t        elementCount,
                                    uint32_t        fillVal = 0u);

    template <uint32_t GroupSize,
              uint32_t RowMask   = 0xF,
              uint32_t BankMask  = 0xF,
              bool     BoundCtrl = false>
    void cross_lane_reverse_CPU(uint32_t*       dataOut,
                                uint32_t const* dataIn,
                                uint32_t        elementCount,
                                uint32_t        fillVal = 0u);

    template <uint32_t RotateDir,
              uint32_t RotateDist,
              uint32_t GroupSize,
              uint32_t RowMask   = 0xF,
              uint32_t BankMask  = 0xF,
              bool     BoundCtrl = false>
    void cross_lane_rotate_CPU(uint32_t*       dataOut,
                               uint32_t const* dataIn,
                               uint32_t        elementCount,
                               uint32_t        fillVal = 0u);

    template <uint32_t ShiftDir,
              uint32_t ShiftDist,
              uint32_t GroupSize,
              uint32_t RowMask   = 0xF,
              uint32_t BankMask  = 0xF,
              bool     BoundCtrl = false>
    void cross_lane_shift_CPU(uint32_t*       dataOut,
                              uint32_t const* dataIn,
                              uint32_t        elementCount,
                              uint32_t        fillVal = 0u);

    template <uint32_t Select0,
              uint32_t Select1,
              uint32_t Select2,
              uint32_t Select3,
              uint32_t GroupSize,
              uint32_t RowMask   = 0xF,
              uint32_t BankMask  = 0xF,
              bool     BoundCtrl = false>
    void cross_lane_shuffle_CPU(uint32_t*       dataOut,
                                uint32_t const* dataIn,
                                uint32_t        elementCount,
                                uint32_t        fillVal = 0u);

    template <uint32_t GroupSize,
              uint32_t RowMask   = 0xF,
              uint32_t BankMask  = 0xF,
              bool     BoundCtrl = false>
    void cross_lane_swap_CPU(uint32_t*       dataOut,
                             uint32_t const* dataIn,
                             uint32_t        elementCount,
                             uint32_t        fillVal = 0u);

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t RowMask   = 0xF,
              uint32_t BankMask  = 0xF,
              bool     BoundCtrl = false,
              typename DispatchEnabler>
    void cross_lane_ref_dispatch_CPU(DataT*       dataOut,
                                     DataT const* dataIn,
                                     uint32_t     elementCount,
                                     DataT        fillVal = DataT(0.0f));

} // namespace rocwmma

#include "reference_impl.hpp"

#endif // ROCWMMA_REFERENCE_HPP
