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
#ifndef ROCWMMA_REFERENCE_IMPL_HPP
#define ROCWMMA_REFERENCE_IMPL_HPP

#include "hip_device.hpp"
#include "reference.hpp"
#include <rocwmma/internal/pack_util.hpp>

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
                  ComputeT       beta)
    {
        int lda = std::is_same<LayoutA, row_major>::value ? k : m;
        int ldb = std::is_same<LayoutB, row_major>::value ? n : k;
        int ldc = std::is_same<LayoutC, row_major>::value ? n : m;
        int ldd = std::is_same<LayoutD, row_major>::value ? n : m;

        auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
        auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

        auto aIndex = std::is_same<LayoutA, row_major>::value ? rowMjr : colMjr;
        auto bIndex = std::is_same<LayoutB, row_major>::value ? rowMjr : colMjr;
        auto cIndex = std::is_same<LayoutC, row_major>::value ? rowMjr : colMjr;
        auto dIndex = std::is_same<LayoutD, row_major>::value ? rowMjr : colMjr;

#pragma omp parallel for
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                ComputeT accum = static_cast<ComputeT>(0);
                for(int h = 0; h < k; ++h)
                {
                    accum += static_cast<ComputeT>(a[aIndex(i, h, lda)])
                             * static_cast<ComputeT>(b[bIndex(h, j, ldb)]);
                }
                d[dIndex(i, j, ldd)] = static_cast<OutputT>(
                    alpha * accum + beta * static_cast<ComputeT>(c[cIndex(i, j, ldc)]));
            }
        }
    }

    template <typename DataT>
    void dlrm_fwd_CPU(DataT const* input, DataT* output, uint32_t m, uint32_t k, uint32_t batchSize)
    {
        auto batchOffset       = m * k;
        uint outputBatchOffset = ((m * (m - 1)) / 2) + k;
#pragma omp parallel for
        for(int b = 0; b < batchSize; b++)
        {
            uint outputIdx = b * outputBatchOffset;

            // Copy MLP to output
            for(int i = 0; i < k; i++)
            {

                output[outputIdx] = input[b * batchOffset + i];
                outputIdx++;
            }
            for(int i = 0; i < m; i++)
            {
                for(int j = 0; j < m; j++)
                {
                    float accum = static_cast<float>(0);
                    for(int h = 0; h < k; h++)
                    {
                        accum += static_cast<float>(input[b * batchOffset + i * k + h])
                                 * static_cast<float>(input[b * batchOffset + j * k + h]);
                    }

                    if(j < i)
                    {
                        output[outputIdx] = static_cast<DataT>(accum);
                        outputIdx++;
                    }
                }
            }
        }
    }

    template <typename DataT>
    void dlrm_bwd_CPU(DataT const* input,
                      DataT const* upstreamGrad,
                      DataT*       bottomMlpGrad,
                      DataT*       output,
                      uint32_t     m,
                      uint32_t     k,
                      uint32_t     batchSize)
    {
        auto batchOffset = m * k;
        auto accOffset   = m * m;
        auto trilSize    = ((m * (m - 1)) / 2) + k;
        auto acc         = new DataT[batchSize * m * m];

#pragma omp parallel for
        for(int b = 0; b < batchSize; b++)
        {
            // Copy bottom MLP grad
            for(int j = 0; j < k; j++)
            {
                bottomMlpGrad[b * k + j] = upstreamGrad[b * trilSize + j];
            }

            // Remake tril
            uint32_t upstreamIdx = b * trilSize + k;
            for(int i = 0; i < m; i++)
            {
                for(int j = 0; j <= i; j++)
                {
                    if(i == j)
                    {
                        acc[b * accOffset + i * m + j] = 0;
                    }
                    else
                    {
                        acc[b * accOffset + i * m + j] = upstreamGrad[upstreamIdx];
                        acc[b * accOffset + j * m + i] = upstreamGrad[upstreamIdx];
                        upstreamIdx++;
                    }
                }
            }

            // Perform reverse bmm
            for(int i = 0; i < m; i++)
            {
                for(int j = 0; j < k; j++)
                {
                    float accum = 0.0f;
                    for(int h = 0; h < m; h++)
                    {
                        accum += static_cast<float>(acc[b * accOffset + i * m + h])
                                 * static_cast<float>(input[b * batchOffset + h * k + j]);
                    }
                    output[b * batchOffset + i * k + j] = static_cast<DataT>(accum);
                }
            }
        }
        delete[] acc;
    }

    template <typename PackedT,
              uint32_t ElementIdx,
              uint32_t GroupSize,
              uint32_t RowMask /* = 0xF */,
              uint32_t BankMask /* = 0xF */,
              bool     BoundCtrl /* = false */>
    void cross_lane_bcast_CPU(PackedT*       dataOut,
                              PackedT const* dataIn,
                              uint32_t       elementCount,
                              uint32_t       fillVal /* = 0u */)
    {
        auto waveSize = HipDevice::instance()->warpSize();
        auto groupSize
            = (GroupSize == CrossLaneOps::Properties::OP_GROUP_SIZE_WARP) ? waveSize : GroupSize;

        auto const loopCnt = elementCount / waveSize;

        for(uint32_t i = 0u; i < loopCnt; ++i)
        {
            // setup the base ptr (each WaveSize elements)
            auto const baseOffset = i * waveSize;

            // For each wave group
            auto const groupCnt = waveSize / groupSize;

            for(uint32_t j = 0u; j < groupCnt; j++)
            {
                auto const groupOffset = j * groupSize;

                for(uint32_t k = 0u; k < groupSize; k++)
                {
                    auto const writeOffset = groupOffset + k;
                    auto const readOffset  = groupOffset + ElementIdx;

                    if(((0x1 << (writeOffset % waveSize / 16u)) & RowMask)
                       && ((0x1 << (writeOffset % 16u / 4u)) & BankMask))
                    {
                        if(ElementIdx >= groupSize)
                        {
                            dataOut[baseOffset + writeOffset] = BoundCtrl ? 0u : fillVal;
                        }
                        else
                        {
                            dataOut[baseOffset + writeOffset] = dataIn[baseOffset + readOffset];
                        }
                    }
                    else
                    {
                        dataOut[baseOffset + writeOffset] = fillVal;
                    }
                }
            }
        }
    }

    template <typename PackedT,
              uint32_t BlockIdx,
              uint32_t BlockSize,
              uint32_t RowMask /* = 0xF */,
              uint32_t BankMask /* = 0xF */,
              bool     BoundCtrl /* = false */>
    void cross_lane_block_bcast_CPU(PackedT*       dataOut,
                                    PackedT const* dataIn,
                                    uint32_t       elementCount,
                                    uint32_t       fillVal /* = 0u */)
    {
        auto waveSize = HipDevice::instance()->warpSize();
        auto groupSize
            = (BlockSize == CrossLaneOps::Properties::OP_GROUP_SIZE_WARP) ? waveSize : BlockSize;

        auto const loopCnt = elementCount / waveSize;

        for(uint32_t i = 0u; i < loopCnt; ++i)
        {
            // setup the base ptr (each WaveSize elements)
            auto const baseOffset = i * waveSize;

            // For each wave group
            auto const groupCnt = waveSize / groupSize;
            for(uint32_t j = 0u; j < groupCnt; j++)
            {
                auto const groupOffset = j * groupSize;

                for(uint32_t k = 0u; k < groupSize; k++)
                {
                    auto const writeOffset = groupOffset + k;
                    auto const readOffset  = BlockIdx * groupSize + k;

                    if(((0x1 << (writeOffset % waveSize / 16u)) & RowMask)
                       && ((0x1 << (writeOffset % 16u / 4u)) & BankMask))
                    {
                        if(BlockIdx >= groupCnt)
                        {
                            dataOut[baseOffset + writeOffset] = BoundCtrl ? 0u : fillVal;
                        }
                        else
                        {
                            dataOut[baseOffset + writeOffset] = dataIn[baseOffset + readOffset];
                        }
                    }
                    else
                    {
                        dataOut[baseOffset + writeOffset] = fillVal;
                    }
                }
            }
        }
    }

    template <typename PackedT,
              uint32_t Select0,
              uint32_t Select1,
              uint32_t Select2,
              uint32_t Select3,
              uint32_t GroupSize,
              uint32_t RowMask /* = 0xF */,
              uint32_t BankMask /* = 0xF */,
              bool     BoundCtrl /* = false */>
    void cross_lane_byte_blend_CPU(PackedT*       dataOut,
                                   PackedT const* src0,
                                   PackedT const* src1,
                                   uint32_t       elementCount,
                                   uint32_t       fillVal /* = 0u */)
    {
        auto waveSize = HipDevice::instance()->warpSize();
        auto groupSize
            = (GroupSize == CrossLaneOps::Properties::OP_GROUP_SIZE_WARP) ? waveSize : GroupSize;

        auto const loopCnt = elementCount / waveSize;

        for(uint32_t i = 0u; i < loopCnt; ++i)
        {
            // setup the base ptr (each WaveSize elements)
            auto const baseOffset = i * waveSize;

            // For each wave group
            auto const groupCnt = waveSize / groupSize;

            for(uint32_t j = 0u; j < groupCnt; j++)
            {
                auto const groupOffset = j * groupSize;

                for(uint32_t k = 0u; k < groupSize; k++)
                {
                    // For each 32b element, blend the bytes from Select values
                    auto const writeOffset = groupOffset + k;
                    auto const readOffset  = groupOffset + k;

                    auto    blendSrc0   = *(PackedT*)&src0[baseOffset + readOffset];
                    auto    blendSrc1   = *(PackedT*)&src1[baseOffset + readOffset];
                    PackedT blendResult = 0;

                    for(int index = 0; index < sizeof(PackedT) / sizeof(uint32_t); index++)
                    {
                        uint32_t ele0 = ((uint32_t*)&blendSrc0)[index];
                        uint32_t ele1 = ((uint32_t*)&blendSrc1)[index];
                        // 0 <= Select < 4  element 0
                        // 4 <= Select < 8  element 1
                        uint32_t bytes0 = ((Select0 < 4u) ? ele0 : ele1);
                        uint32_t bytes1 = ((Select1 < 4u) ? ele0 : ele1);
                        uint32_t bytes2 = ((Select2 < 4u) ? ele0 : ele1);
                        uint32_t bytes3 = ((Select3 < 4u) ? ele0 : ele1);

                        // Byte mask and bit shifts needed
                        uint32_t byteMask  = 0xFF;
                        uint32_t bitShift0 = Select0 % 4u * 8;
                        uint32_t bitShift1 = Select1 % 4u * 8;
                        uint32_t bitShift2 = Select2 % 4u * 8;
                        uint32_t bitShift3 = Select3 % 4u * 8;

                        // Shift byte mask to selected byte, and copy this selected byte into position
                        uint32_t result = 0x0;
                        result |= (((byteMask << bitShift0) & bytes0) >> bitShift0);
                        result |= (((byteMask << bitShift1) & bytes1) >> bitShift1 << 8u);
                        result |= (((byteMask << bitShift2) & bytes2) >> bitShift2 << 16u);
                        result |= (((byteMask << bitShift3) & bytes3) >> bitShift3 << 24u);

                        ((uint32_t*)&blendResult)[index] = result;
                    }

                    // Check the row / bank masking
                    if(((0x1 << (writeOffset % waveSize / 16u)) & RowMask)
                       && ((0x1 << (writeOffset % 16u / 4u)) & BankMask))
                    {
                        *(PackedT*)&dataOut[baseOffset + writeOffset] = blendResult;
                    }
                    // BoundCtrl does not affect shuffle 4 as the indices are & with 0x3
                    else
                    {
                        *(PackedT*)&dataOut[baseOffset + writeOffset] = fillVal;
                    }
                }
            }
        }
    }

    template <typename PackedT,
              uint32_t GroupSize,
              uint32_t RowMask /* = 0xF */,
              uint32_t BankMask /* = 0xF */,
              bool     BoundCtrl /* = false */>
    void cross_lane_wfall_bcast_CPU(PackedT*       dataOut,
                                    PackedT const* dataIn,
                                    uint32_t       elementCount,
                                    uint32_t       fillVal /* = 0u */)
    {
        auto waveSize = HipDevice::instance()->warpSize();
        auto groupSize
            = (GroupSize == CrossLaneOps::Properties::OP_GROUP_SIZE_WARP) ? waveSize : GroupSize;

        auto const loopCnt = elementCount / waveSize;

        for(uint32_t i = 0u; i < loopCnt; ++i)
        {
            // setup the base ptr (each WaveSize elements)
            auto const baseOffset = i * waveSize;

            // For each wave group
            auto const groupCnt = waveSize / groupSize;
            for(uint32_t j = 0u; j < groupCnt; j++)
            {
                auto const groupOffset = j * groupSize;

                // Read offset is last element of prev group
                auto const readOffset = groupOffset - 1u;

                for(uint32_t k = 0u; k < groupSize; k++)
                {
                    auto const writeOffset = groupOffset + k;

                    if(((0x1 << (writeOffset % waveSize / 16u)) & RowMask)
                       && ((0x1 << (writeOffset % 16u / 4u)) & BankMask))
                    {
                        // OOB, covers the case of first group is read-thru
                        if(readOffset >= writeOffset)
                        {
                            dataOut[baseOffset + writeOffset] = dataIn[baseOffset + writeOffset];
                        }
                        else
                        {
                            dataOut[baseOffset + writeOffset] = dataIn[baseOffset + readOffset];
                        }
                    }
                    else
                    {
                        dataOut[baseOffset + writeOffset] = fillVal;
                    }
                }
            }
        }
    }

    template <typename PackedT,
              uint32_t GroupSize,
              uint32_t RowMask /* = 0xF */,
              uint32_t BankMask /* = 0xF */,
              bool     BoundCtrl /* = false */>
    void cross_lane_reverse_CPU(PackedT*       dataOut,
                                PackedT const* dataIn,
                                uint32_t       elementCount,
                                uint32_t       fillVal /* = 0u */)
    {
        auto waveSize = HipDevice::instance()->warpSize();
        auto groupSize
            = (GroupSize == CrossLaneOps::Properties::OP_GROUP_SIZE_WARP) ? waveSize : GroupSize;

        auto const loopCnt = elementCount / waveSize;

        for(uint32_t i = 0u; i < loopCnt; ++i)
        {
            // setup the base ptr (each WaveSize elements)
            auto const baseOffset = i * waveSize;

            // For each wave group
            auto const groupCnt = waveSize / groupSize;
            for(uint32_t j = 0u; j < groupCnt; j++)
            {
                auto const groupOffset = j * groupSize;

                for(uint32_t k = 0u; k < groupSize; k++)
                {
                    auto const writeOffset = groupOffset + k;
                    auto const readOffset  = groupOffset + (groupSize - k - 1u);

                    // Check the row / bank masking
                    if(((0x1 << (writeOffset % waveSize / 16u)) & RowMask)
                       && ((0x1 << (writeOffset % 16u / 4u)) & BankMask))
                    {
                        dataOut[baseOffset + writeOffset] = dataIn[baseOffset + readOffset];
                    }
                    // BoundCtrl has no effect on rotate - no indices are ever OOB
                    else
                    {
                        dataOut[baseOffset + writeOffset] = fillVal;
                    }
                }
            }
        }
    }

    template <typename PackedT,
              uint32_t RotateDir,
              uint32_t RotateDist,
              uint32_t GroupSize,
              uint32_t RowMask /* = 0xF */,
              uint32_t BankMask /* = 0xF */,
              bool     BoundCtrl /* = false */>
    void cross_lane_rotate_CPU(PackedT*       dataOut,
                               PackedT const* dataIn,
                               uint32_t       elementCount,
                               uint32_t       fillVal /* = 0u */)
    {
        auto waveSize = HipDevice::instance()->warpSize();
        auto groupSize
            = (GroupSize == CrossLaneOps::Properties::OP_GROUP_SIZE_WARP) ? waveSize : GroupSize;

        auto const loopCnt = elementCount / waveSize;

        for(uint32_t i = 0u; i < loopCnt; ++i)
        {
            // setup the base ptr (each WaveSize elements)
            auto const baseOffset = i * waveSize;

            // For each wave group
            auto const groupCnt = waveSize / groupSize;
            for(uint32_t j = 0u; j < groupCnt; j++)
            {
                auto const groupOffset = j * groupSize;

                for(uint32_t k = 0u; k < groupSize; k++)
                {
                    auto const writeOffset = groupOffset + k;
                    auto const readOffset
                        = groupOffset
                          + (k + (RotateDir ? (groupSize - RotateDist) : RotateDist)) % groupSize;

                    // Check the row / bank masking
                    if(((0x1 << (writeOffset % waveSize / 16u)) & RowMask)
                       && ((0x1 << (writeOffset % 16u / 4u)) & BankMask))
                    {
                        dataOut[baseOffset + writeOffset] = dataIn[baseOffset + readOffset];
                    }
                    // BoundCtrl has no effect on rotate - no indices are ever OOB
                    else
                    {
                        dataOut[baseOffset + writeOffset] = fillVal;
                    }
                }
            }
        }
    }

    template <typename PackedT,
              uint32_t ShiftDir,
              uint32_t ShiftDist,
              uint32_t GroupSize,
              uint32_t RowMask /* = 0xF */,
              uint32_t BankMask /* = 0xF */,
              bool     BoundCtrl /* = false */>
    void cross_lane_shift_CPU(PackedT*       dataOut,
                              PackedT const* dataIn,
                              uint32_t       elementCount,
                              uint32_t       fillVal /* = 0u */)
    {
        auto waveSize = HipDevice::instance()->warpSize();
        auto groupSize
            = (GroupSize == CrossLaneOps::Properties::OP_GROUP_SIZE_WARP) ? waveSize : GroupSize;

        auto const loopCnt = elementCount / waveSize;

        for(uint32_t i = 0u; i < loopCnt; ++i)
        {
            // setup the base ptr (each WaveSize elements)
            auto const baseOffset = i * waveSize;

            // For each wave group
            auto const groupCnt = waveSize / groupSize;
            for(uint32_t j = 0u; j < groupCnt; j++)
            {
                auto const groupOffset = j * groupSize;

                for(uint32_t k = 0u; k < groupSize; k++)
                {
                    auto const writeOffset = groupOffset + k;
                    auto const shiftOffset = (ShiftDir == CrossLaneOps::Properties::OP_DIR_R)
                                                 ? (k - ShiftDist)
                                                 : (k + ShiftDist);
                    auto const readOffset  = groupOffset + shiftOffset;

                    // Check the row / bank masking
                    if(((0x1 << (writeOffset % waveSize / 16u)) & RowMask)
                       && ((0x1 << (writeOffset % 16u / 4u)) & BankMask))
                    {
                        // OOB would be > groupsize, as uint32_t type
                        if(shiftOffset >= groupSize)
                        {
                            dataOut[baseOffset + writeOffset] = BoundCtrl ? 0u : fillVal;
                        }
                        else
                        {
                            dataOut[baseOffset + writeOffset] = dataIn[baseOffset + readOffset];
                        }
                    }
                    // If outside the mask, then use prev val to fill
                    else
                    {
                        dataOut[baseOffset + writeOffset] = fillVal;
                    }
                }
            }
        }
    }

    template <typename PackedT,
              uint32_t Select0,
              uint32_t Select1,
              uint32_t Select2,
              uint32_t Select3,
              uint32_t GroupSize,
              uint32_t RowMask /* = 0xF */,
              uint32_t BankMask /* = 0xF */,
              bool     BoundCtrl /* = false */>
    void cross_lane_shuffle_CPU(PackedT*       dataOut,
                                PackedT const* dataIn,
                                uint32_t       elementCount,
                                uint32_t       fillVal /* = 0u */)
    {
        auto waveSize = HipDevice::instance()->warpSize();
        auto groupSize
            = (GroupSize == CrossLaneOps::Properties::OP_GROUP_SIZE_WARP) ? waveSize : GroupSize;

        auto const loopCnt = elementCount / waveSize;

        for(uint32_t i = 0u; i < loopCnt; ++i)
        {
            // setup the base ptr (each WaveSize elements)
            auto const baseOffset = i * waveSize;

            // For each wave group
            auto const groupCnt = waveSize / groupSize;
            for(uint32_t j = 0u; j < groupCnt; j++)
            {
                auto const groupOffset = j * groupSize;

                for(uint32_t k = 0u; k < groupSize; k++)
                {
                    auto const writeOffset = groupOffset + k;

                    auto readIndex = 0u;
                    switch(k)
                    {
                    case 0u:
                        readIndex = Select0;
                        break;
                    case 1u:
                        readIndex = Select1;
                        break;
                    case 2u:
                        readIndex = Select2;
                        break;
                    case 3u:
                        readIndex = Select3;
                        break;
                    default:;
                    }

                    auto const readOffset = groupOffset + readIndex;

                    // Check the row / bank masking
                    if(((0x1 << (writeOffset % waveSize / 16u)) & RowMask)
                       && ((0x1 << (writeOffset % 16u / 4u)) & BankMask))
                    {
                        dataOut[baseOffset + writeOffset] = dataIn[baseOffset + readOffset];
                    }
                    // BoundCtrl does not affect shuffle 4 as the indices are & with 0x3
                    else
                    {
                        dataOut[baseOffset + writeOffset] = fillVal;
                    }
                }
            }
        }
    }

    template <typename PackedT,
              uint32_t GroupSize,
              uint32_t RowMask /* = 0xF */,
              uint32_t BankMask /* = 0xF */,
              bool     BoundCtrl /* = false */>
    void cross_lane_swap_CPU(PackedT*       dataOut,
                             PackedT const* dataIn,
                             uint32_t       elementCount,
                             uint32_t       fillVal /* = 0u */)
    {
        auto waveSize = HipDevice::instance()->warpSize();
        auto groupSize
            = (GroupSize == CrossLaneOps::Properties::OP_GROUP_SIZE_WARP) ? waveSize : GroupSize;

        auto const loopCnt = elementCount / waveSize;

        for(uint32_t i = 0u; i < loopCnt; ++i)
        {
            // setup the base ptr (each WaveSize elements)
            auto const baseOffset = i * waveSize;

            // For each wave group
            auto const groupCnt = waveSize / groupSize;
            for(uint32_t j = 0u; j < groupCnt; j++)
            {
                auto const groupOffset = j * groupSize;

                for(uint32_t k = 0u; k < groupSize; k++)
                {
                    auto const writeOffset = groupOffset + k;
                    auto const readOffset  = groupOffset + k + (j % 2 ? -groupSize : groupSize);

                    // Check the row / bank masking
                    if(((0x1 << (writeOffset % waveSize / 16u)) & RowMask)
                       && ((0x1 << (writeOffset % 16u / 4u)) & BankMask))
                    {
                        dataOut[baseOffset + writeOffset] = dataIn[baseOffset + readOffset];
                    }
                    // BoundCtrl has no effect on rotate - no indices are ever OOB
                    else
                    {
                        dataOut[baseOffset + writeOffset] = fillVal;
                    }
                }
            }
        }
    }

    // Dispatcher for CPU references with single input sources.
    // Select reference using cross lane op meta data.
    template <typename DataT,
              typename CrossLaneOp,
              uint32_t RowMask   = 0xF,
              uint32_t BankMask  = 0xF,
              bool     BoundCtrl = false>
    void cross_lane_ref_dispatch_CPU(DataT*       dataOut,
                                     DataT const* dataIn,
                                     uint32_t     elementCount,
                                     DataT        fillVal = DataT(0))
    {
        using PackedT = typename PackTraits<DataT>::PackedT;
        // Interface to device kernel
        using RefFunc = void (*)(PackedT*, // dataOut
                                 PackedT const*, // dataIn
                                 uint32_t, // elementCount
                                 uint32_t); // fillVal

        RefFunc dispatcher = nullptr;

        // Select reference function
        if constexpr(CrossLaneOp::opId() == rocwmma::CrossLaneOps::Properties::OP_ID_BCAST)
        {
            dispatcher = cross_lane_bcast_CPU<PackedT,
                                              CrossLaneOp::elementIdx(),
                                              CrossLaneOp::groupSize(),
                                              RowMask,
                                              BankMask,
                                              BoundCtrl>;
        }
        else if constexpr(CrossLaneOp::opId()
                          == rocwmma::CrossLaneOps::Properties::OP_ID_BLOCK_BCAST)
        {
            dispatcher = cross_lane_block_bcast_CPU<PackedT,
                                                    CrossLaneOp::elementIdx(),
                                                    CrossLaneOp::groupSize(),
                                                    RowMask,
                                                    BankMask,
                                                    BoundCtrl>;
        }
        else if constexpr(CrossLaneOp::opId() == rocwmma::CrossLaneOps::Properties::OP_ID_REVERSE)
        {
            dispatcher = cross_lane_reverse_CPU<PackedT,
                                                CrossLaneOp::groupSize(),
                                                RowMask,
                                                BankMask,
                                                BoundCtrl>;
        }
        else if constexpr(CrossLaneOp::opId() == rocwmma::CrossLaneOps::Properties::OP_ID_SWAP)
        {
            dispatcher = cross_lane_swap_CPU<PackedT,
                                             CrossLaneOp::groupSize(),
                                             RowMask,
                                             BankMask,
                                             BoundCtrl>;
        }
        else if constexpr(CrossLaneOp::opId()
                          == rocwmma::CrossLaneOps::Properties::OP_ID_WFALL_BCAST)
        {
            dispatcher = cross_lane_wfall_bcast_CPU<PackedT,
                                                    CrossLaneOp::groupSize(),
                                                    RowMask,
                                                    BankMask,
                                                    BoundCtrl>;
        }
        else if constexpr(CrossLaneOp::opId() == rocwmma::CrossLaneOps::Properties::OP_ID_ROTATE)
        {
            dispatcher = cross_lane_rotate_CPU<PackedT,
                                               CrossLaneOp::opDir(),
                                               CrossLaneOp::opDist(),
                                               CrossLaneOp::groupSize(),
                                               RowMask,
                                               BankMask,
                                               BoundCtrl>;
        }
        else if constexpr(CrossLaneOp::opId() == rocwmma::CrossLaneOps::Properties::OP_ID_SHIFT)
        {
            dispatcher = cross_lane_shift_CPU<PackedT,
                                              CrossLaneOp::opDir(),
                                              CrossLaneOp::opDist(),
                                              CrossLaneOp::groupSize(),
                                              RowMask,
                                              BankMask,
                                              BoundCtrl>;
        }
        else if constexpr((CrossLaneOp::opId() == rocwmma::CrossLaneOps::Properties::OP_ID_SHUFFLE))
        {
            if constexpr(CrossLaneOp::groupSize() == 4u)
            {
                dispatcher = cross_lane_shuffle_CPU<PackedT,
                                                    CrossLaneOp::select0(),
                                                    CrossLaneOp::select1(),
                                                    CrossLaneOp::select2(),
                                                    CrossLaneOp::select3(),
                                                    CrossLaneOp::groupSize(),
                                                    RowMask,
                                                    BankMask,
                                                    BoundCtrl>;
            }
            else if constexpr(CrossLaneOp::groupSize() == 2u)
            {
                dispatcher = cross_lane_shuffle_CPU<PackedT,
                                                    CrossLaneOp::select0(),
                                                    CrossLaneOp::select1(),
                                                    CrossLaneOp::select0() + 2u,
                                                    CrossLaneOp::select1() + 2u,
                                                    CrossLaneOp::groupSize(),
                                                    RowMask,
                                                    BankMask,
                                                    BoundCtrl>;
            }
        }

        // Determine function params
        // Must scale in 32b chunks
        PackedT*       writeOut = reinterpret_cast<PackedT*>(dataOut);
        PackedT const* readIn   = reinterpret_cast<PackedT const*>(dataIn);
        elementCount            = elementCount / PackTraits<DataT>::PackRatio;

        // From here forth is in 32b land...

        // Finally, run the reference function
        if(dispatcher != nullptr)
        {
            dispatcher(writeOut, readIn, elementCount, static_cast<uint32_t>(fillVal));
        }
    }

    // Dispatcher for CPU references with dual input sources.
    // Select reference using cross lane op meta data.
    template <typename DataT,
              typename CrossLaneOp,
              uint32_t RowMask   = 0xF,
              uint32_t BankMask  = 0xF,
              bool     BoundCtrl = false>
    void cross_lane_ref_dispatch_CPU(DataT*       dataOut,
                                     DataT const* dataIn0,
                                     DataT const* dataIn1,
                                     uint32_t     elementCount,
                                     DataT        fillVal = DataT(0))
    {
        using PackedT = typename PackTraits<DataT>::PackedT;
        // Interface to cpu reference kernel
        using RefFunc = void (*)(PackedT*, // dataOut
                                 PackedT const*, // dataIn0
                                 PackedT const*, // dataIn1
                                 uint32_t, // elementCount
                                 uint32_t); // fillVal

        RefFunc dispatcher = nullptr;

        // Select reference function
        if constexpr(CrossLaneOp::opId() == rocwmma::CrossLaneOps::Properties::OP_ID_PERM_BYTE)
        {
            if constexpr(CrossLaneOp::groupSize() == 1u)
            {
                dispatcher = cross_lane_byte_blend_CPU<PackedT,
                                                       CrossLaneOp::select0(),
                                                       CrossLaneOp::select1(),
                                                       CrossLaneOp::select2(),
                                                       CrossLaneOp::select3(),
                                                       CrossLaneOp::groupSize(),
                                                       RowMask,
                                                       BankMask,
                                                       BoundCtrl>;
            }
        }

        // Determine function params
        // Must scale in 32b chunks
        PackedT*       writeOut = reinterpret_cast<PackedT*>(dataOut);
        PackedT const* src0In   = reinterpret_cast<PackedT const*>(dataIn0);
        PackedT const* src1In   = reinterpret_cast<PackedT const*>(dataIn1);
        elementCount            = elementCount / PackTraits<DataT>::PackRatio;

        // From here forth is in 32b land...

        // Finally, run the reference function
        if(dispatcher != nullptr)
        {
            dispatcher(writeOut, src0In, src1In, elementCount, static_cast<uint32_t>(fillVal));
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_REFERENCE_IMPL_HPP
