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

#ifndef ROCWMMA_DEVICE_DPP_OPS_HPP
#define ROCWMMA_DEVICE_DPP_OPS_HPP

#include "cross_lane_ops_util.hpp"

namespace rocwmma
{
    ROCWMMA_DEVICE inline bool
        isDppMasked(uint32_t id, uint32_t WriteRowMask, uint32_t WriteBankMask)
    {
        return (WriteRowMask & (1 << ((id >> 4) & 0x3)))
               && (WriteBankMask & (1 << ((id >> 2) & 0x3)));
    }

    /**
	 * @addtogroup cross_lane_op_gen_ref_value_funcs
	 * @{
	 */
    template <uint32_t SubgroupSize, uint32_t Element>
    ROCWMMA_DEVICE inline uint32_t getDppBCastExpect(uint32_t input)
    {
        return (input & (~(SubgroupSize - 1))) + Element;
    }

    template <uint32_t SubgroupSize>
    ROCWMMA_DEVICE inline uint32_t getDppReverseExpect(uint32_t input)
    {
        uint32_t maxInGroup = SubgroupSize - 1;
        return ((input & (~maxInGroup) | (maxInGroup - (input & maxInGroup))));
    }

    template <uint32_t SubgroupSize, uint32_t Direction, uint32_t Distance>
    ROCWMMA_DEVICE inline uint32_t getDppRotateExpect(uint32_t input)
    {
        auto afterRotate = (input & (SubgroupSize - 1));
        afterRotate += Direction == CrossLaneOps::OP_DIR_L ? Distance : -Distance;
        afterRotate += SubgroupSize;
        afterRotate &= (SubgroupSize - 1);
        return (input & (~(SubgroupSize - 1))) | afterRotate;
    }

    template <uint32_t SubGroupSize, uint32_t ShiftDir, uint32_t ShiftDist, bool BoundCtrl>
    ROCWMMA_DEVICE inline uint32_t getDppShiftExpect(uint32_t input, uint32_t prev)
    {
        input += ShiftDir == CrossLaneOps::OP_DIR_L ? ShiftDist : -ShiftDist;
        uint32_t fillValue  = BoundCtrl ? 0 : prev;
        auto     afterShift = (input & (SubGroupSize - 1));
        afterShift += ShiftDir == CrossLaneOps::OP_DIR_L ? -ShiftDist : ShiftDist;
        return (afterShift < 0 || afterShift >= SubGroupSize) ? fillValue : input;
    }

    template <uint32_t Select0, uint32_t Select1>
    ROCWMMA_DEVICE inline uint32_t getDppShuffle2Expect(uint32_t input)
    {
        auto id = input & 0b1;
        input -= id;
        input += id == 0 ? Select0 : Select1;
        return input;
    }

    template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
    ROCWMMA_DEVICE inline uint32_t getDppShuffle4Expect(uint32_t input)
    {
        auto id = input & 0b11;
        input -= id;
        input += id == 0 ? Select0 : id == 1 ? Select1 : id == 2 ? Select2 : Select3;
        return input;
    }

    template <uint32_t SubGroupSize>
    ROCWMMA_DEVICE inline uint32_t getDppSwapExpect(uint32_t input)
    {
        return input ^ SubGroupSize;
    }

    template <uint32_t SubGroupSize>
    ROCWMMA_DEVICE inline uint32_t getDppWFallBCastExpect(uint32_t input)
    {
        if constexpr(SubGroupSize == 16)
        {
            auto firstInRow = input & 0b110000;
            input           = firstInRow > 0 ? firstInRow - 1 : input;
        }
        else
        {
            auto firstInRow = input & 0b100000;
            input           = firstInRow > 0 ? firstInRow - 1 : input;
        }
        return input;
    }
    /** @} */

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask,
              uint32_t WriteBankMask,
              bool     BoundCtrl>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_BCAST
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_DPP,
                                    bool>
                   dppOpsTestCase()
    {
        uint32_t id       = threadIdx.x;
        uint32_t prev     = VALUE_OUT_OF_RANGE;
        DataT    input    = makeValueFromU32<DataT>(id);
        bool     isMasked = isDppMasked(id, WriteRowMask, WriteBankMask);
        DataT    expect   = makeValueFromU32<DataT>(
            isMasked ? getDppBCastExpect<CrossLaneOp::GROUP_SIZE, CrossLaneOp::ELEMENT_IDX>(id)
                          : prev);

        auto output
            = rocwmma::Dpp::Driver<CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>::exec(
                input, makeValueFromU32<DataT>(prev));

        return output != expect;
    }

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask,
              uint32_t WriteBankMask,
              bool     BoundCtrl>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_REVERSE
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_DPP,
                                    bool>
                   dppOpsTestCase()
    {
        uint32_t id       = threadIdx.x;
        uint32_t prev     = VALUE_OUT_OF_RANGE;
        DataT    input    = makeValueFromU32<DataT>(id);
        bool     isMasked = isDppMasked(id, WriteRowMask, WriteBankMask);
        DataT    expect   = makeValueFromU32<DataT>(
            isMasked ? getDppReverseExpect<CrossLaneOp::GROUP_SIZE>(id) : prev);
        DataT output
            = rocwmma::Dpp::Driver<CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>::exec(
                input, makeValueFromU32<DataT>(prev));

        return output != expect;
    }

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask,
              uint32_t WriteBankMask,
              bool     BoundCtrl>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_ROTATE
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_DPP,
                                    bool>
                   dppOpsTestCase()
    {
        uint32_t id       = threadIdx.x;
        uint32_t prev     = VALUE_OUT_OF_RANGE;
        DataT    input    = makeValueFromU32<DataT>(id);
        bool     isMasked = isDppMasked(id, WriteRowMask, WriteBankMask);
        DataT    expect
            = makeValueFromU32<DataT>(isMasked ? getDppRotateExpect<CrossLaneOp::GROUP_SIZE,
                                                                    CrossLaneOp::OP_DIR,
                                                                    CrossLaneOp::OP_DIST>(id)
                                               : prev);
        DataT output
            = rocwmma::Dpp::Driver<CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>::exec(
                input, makeValueFromU32<DataT>(prev));

        return output != expect;
    }

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask,
              uint32_t WriteBankMask,
              bool     BoundCtrl>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_SHIFT
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_DPP,
                                    bool>
                   dppOpsTestCase()
    {
        uint32_t id       = threadIdx.x;
        uint32_t prev     = VALUE_OUT_OF_RANGE;
        DataT    input    = makeValueFromU32<DataT>(id);
        bool     isMasked = isDppMasked(id, WriteRowMask, WriteBankMask);
        DataT expect = makeValueFromU32<DataT>(isMasked ? getDppShiftExpect<CrossLaneOp::GROUP_SIZE,
                                                                            CrossLaneOp::OP_DIR,
                                                                            CrossLaneOp::OP_DIST,
                                                                            BoundCtrl>(input, prev)
                                                        : prev);
        DataT output
            = rocwmma::Dpp::Driver<CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>::exec(
                input, makeValueFromU32<DataT>(prev));

        return output != expect;
    }

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask,
              uint32_t WriteBankMask,
              bool     BoundCtrl>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_SHUFFLE
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_DPP,
                                    bool>
                   dppOpsTestCase()
    {
        uint32_t id       = threadIdx.x;
        uint32_t prev     = VALUE_OUT_OF_RANGE;
        DataT    input    = makeValueFromU32<DataT>(id);
        bool     isMasked = isDppMasked(id, WriteRowMask, WriteBankMask);
        DataT    expect   = -1;
        if constexpr(CrossLaneOp::groupSize() == 2)
        {
            expect = makeValueFromU32<DataT>(
                isMasked ? getDppShuffle2Expect<CrossLaneOp::SELECT_0, CrossLaneOp::SELECT_1>(id)
                         : prev);
        }
        else if constexpr(CrossLaneOp::groupSize() == 4)
        {
            expect
                = makeValueFromU32<DataT>(isMasked ? getDppShuffle4Expect<CrossLaneOp::SELECT_0,
                                                                          CrossLaneOp::SELECT_1,
                                                                          CrossLaneOp::SELECT_2,
                                                                          CrossLaneOp::SELECT_3>(id)
                                                   : prev);
        }
        DataT output
            = rocwmma::Dpp::Driver<CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>::exec(
                input, makeValueFromU32<DataT>(prev));

        return output != expect;
    }

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask,
              uint32_t WriteBankMask,
              bool     BoundCtrl>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_SWAP
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_DPP,
                                    bool>
                   dppOpsTestCase()
    {
        uint32_t id       = threadIdx.x;
        uint32_t prev     = VALUE_OUT_OF_RANGE;
        DataT    input    = makeValueFromU32<DataT>(id);
        bool     isMasked = isDppMasked(id, WriteRowMask, WriteBankMask);
        DataT    expect   = makeValueFromU32<DataT>(
            isMasked ? getDppSwapExpect<CrossLaneOp::GROUP_SIZE>(id) : prev);
        DataT output
            = rocwmma::Dpp::Driver<CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>::exec(
                input, makeValueFromU32<DataT>(prev));

        return output != expect;
    }

    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask,
              uint32_t WriteBankMask,
              bool     BoundCtrl>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_WFALL_BCAST
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_DPP,
                                    bool>
                   dppOpsTestCase()
    {
        uint32_t id       = threadIdx.x;
        uint32_t prev     = VALUE_OUT_OF_RANGE;
        DataT    input    = makeValueFromU32<DataT>(id);
        bool     isMasked = isDppMasked(id, WriteRowMask, WriteBankMask);
        DataT    expect   = makeValueFromU32<DataT>(
            isMasked ? getDppWFallBCastExpect<CrossLaneOp::GROUP_SIZE>(id) : prev);
        DataT output
            = rocwmma::Dpp::Driver<CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>::exec(
                input, makeValueFromU32<DataT>(prev));

        return output != expect;
    }
} // namespace rocwmma

#endif // ROCWMMA_DEVICE_DPP_OPS_HPP
