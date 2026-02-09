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

#ifndef ROCWMMA_DEVICE_SWIZZLE_OPS_HPP
#define ROCWMMA_DEVICE_SWIZZLE_OPS_HPP

#include "cross_lane_ops_util.hpp"

namespace rocwmma
{
    /**
	 * @addtogroup cross_lane_op_gen_ref_value_funcs
	 * @{
	 */
    template <uint32_t ElementIdx, uint32_t SubGroupSize>
    ROCWMMA_DEVICE inline uint32_t getSwizzleBCastExpect(uint32_t input)
    {
        return (input & (~(SubGroupSize - 1))) + ElementIdx;
    }

    template <uint32_t SubGroupSize>
    ROCWMMA_DEVICE inline uint32_t getSwizzleReverseExpect(uint32_t input)
    {
        uint32_t maxInGroup = SubGroupSize - 1;
        return ((input & (~maxInGroup) | (maxInGroup - (input & maxInGroup))));
    }

    template <uint32_t SubGroupSize, uint32_t Direction, uint32_t Distance>
    ROCWMMA_DEVICE inline uint32_t getSwizzleRotateExpect(uint32_t input)
    {
        auto afterRotate = (input & (SubGroupSize - 1));
        afterRotate += Direction == CrossLaneOps::OP_DIR_L ? Distance : -Distance;
        afterRotate += SubGroupSize;
        afterRotate &= (SubGroupSize - 1);
        return (input & (~(SubGroupSize - 1))) | afterRotate;
    }

    template <uint32_t Select0, uint32_t Select1>
    ROCWMMA_DEVICE inline uint32_t getSwizzleShuffle2Expect(uint32_t input)
    {
        auto id = input & 0b1;
        input -= id;
        input += id == 0 ? Select0 : Select1;
        return input;
    }

    template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
    ROCWMMA_DEVICE inline uint32_t getSwizzleShuffle4Expect(uint32_t input)
    {
        auto id = input & 0b11;
        input -= id;
        input += id == 0 ? Select0 : id == 1 ? Select1 : id == 2 ? Select2 : Select3;
        return input;
    }

    template <uint32_t SubGroupSize>
    ROCWMMA_DEVICE inline uint32_t getSwizzleSwapExpect(uint32_t input)
    {
        return input ^ SubGroupSize;
    }
    /** @} */

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_BCAST
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_SWIZZLE,
                                    bool>
                   swizzleOpsTestCase()
    {
        uint32_t id     = threadIdx.x;
        DataT    input  = makeValueFromU32<DataT>(id);
        DataT    expect = makeValueFromU32<DataT>(
            getSwizzleBCastExpect<CrossLaneOp::elementIdx(), CrossLaneOp::groupSize()>(id));
        DataT output = rocwmma::Swizzle::Driver<CrossLaneOp>::exec(input);

        return output != expect;
    }

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_REVERSE
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_SWIZZLE,
                                    bool>
                   swizzleOpsTestCase()
    {
        uint32_t id    = threadIdx.x;
        DataT    input = makeValueFromU32<DataT>(id);
        DataT    expect
            = makeValueFromU32<DataT>(getSwizzleReverseExpect<CrossLaneOp::groupSize()>(id));
        DataT output = rocwmma::Swizzle::Driver<CrossLaneOp>::exec(input);

        return output != expect;
    }

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_ROTATE
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_SWIZZLE,
                                    bool>
                   swizzleOpsTestCase()
    {
        uint32_t id     = threadIdx.x;
        DataT    input  = makeValueFromU32<DataT>(id);
        DataT    expect = makeValueFromU32<DataT>(getSwizzleRotateExpect<CrossLaneOp::groupSize(),
                                                                      CrossLaneOp::opDir(),
                                                                      CrossLaneOp::opDist()>(id));
        DataT    output = rocwmma::Swizzle::Driver<CrossLaneOp>::exec(input);

        return output != expect;
    }

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_SHUFFLE
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_SWIZZLE,
                                    bool>
                   swizzleOpsTestCase()
    {
        uint32_t id     = threadIdx.x;
        DataT    input  = makeValueFromU32<DataT>(id);
        DataT    expect = -1;
        if constexpr(CrossLaneOp::groupSize() == 2)
        {
            expect = makeValueFromU32<DataT>(
                getSwizzleShuffle2Expect<CrossLaneOp::select0(), CrossLaneOp::select1()>(id));
        }
        else if constexpr(CrossLaneOp::groupSize() == 4)
        {
            expect = makeValueFromU32<DataT>(getSwizzleShuffle4Expect<CrossLaneOp::select0(),
                                                                      CrossLaneOp::select1(),
                                                                      CrossLaneOp::select2(),
                                                                      CrossLaneOp::select3()>(id));
        }
        DataT output = rocwmma::Swizzle::Driver<CrossLaneOp>::exec(input);

        return output != expect;
    }

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_SWAP
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_SWIZZLE,
                                    bool>
                   swizzleOpsTestCase()
    {
        uint32_t id    = threadIdx.x;
        DataT    input = makeValueFromU32<DataT>(id);
        DataT expect = makeValueFromU32<DataT>(getSwizzleSwapExpect<CrossLaneOp::groupSize()>(id));
        DataT output = rocwmma::Swizzle::Driver<CrossLaneOp>::exec(input);

        return output != expect;
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_SWIZZLE_OPS_HPP
