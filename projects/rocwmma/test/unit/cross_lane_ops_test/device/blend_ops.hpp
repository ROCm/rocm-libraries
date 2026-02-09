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

#ifndef ROCWMMA_DEVICE_BLEND_OPS_HPP
#define ROCWMMA_DEVICE_BLEND_OPS_HPP

#include "cross_lane_ops_util.hpp"

namespace rocwmma
{
    /**
	 * @addtogroup cross_lane_op_gen_ref_value_funcs
	 * @{
	 */
    ROCWMMA_DEVICE inline uint32_t
        getBlendPermByte(uint32_t input0, uint32_t input1, uint32_t select)
    {
        // check the invokation of __builtin_amdgcn_perm in blend_impl.hpp
        // "src0 and src1 are flipped". So do same here.
        uint64_t inputRegs = input1;
        inputRegs          = inputRegs << 32 | input0;

        return (inputRegs >> (select * 8)) & 0xFF;
    }

    template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
    ROCWMMA_DEVICE inline uint32_t getBlendPermByteExpect(uint32_t input0, uint32_t input1)
    {
        uint32_t output = 0;
        output |= getBlendPermByte(input0, input1, Select3);
        output <<= 8;
        output |= getBlendPermByte(input0, input1, Select2);
        output <<= 8;
        output |= getBlendPermByte(input0, input1, Select1);
        output <<= 8;
        output |= getBlendPermByte(input0, input1, Select0);
        return output;
    }

    template <uint32_t SubGroupSize>
    ROCWMMA_DEVICE inline uint32_t getBlendZipExpect(uint32_t input0, uint32_t input1, uint32_t idx)
    {
        return (idx / SubGroupSize) & 0x1 ? input1 : input0;
    }
    /** @} */

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_PERM_BYTE
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_VPERM,
                                    bool>
                   blendOpsTestCase()
    {
        DataT input0 = makeValueFromU32<DataT>(0x05060708);
        DataT input1 = makeValueFromU32<DataT>(0x01020304);
        DataT expect = makeValueFromU32<DataT>(
            getBlendPermByteExpect<CrossLaneOp::select0(),
                                   CrossLaneOp::select1(),
                                   CrossLaneOp::select2(),
                                   CrossLaneOp::select3()>(input0, input1));
        DataT output = rocwmma::Blend::Driver<CrossLaneOp>::exec(input0, input1);

        return output != expect;
    }

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_BLEND
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_VBLEND,
                                    bool>
                   blendOpsTestCase()
    {
        DataT input0 = makeValueFromU32<DataT>(1);
        DataT input1 = makeValueFromU32<DataT>(2);
        DataT expect = makeValueFromU32<DataT>(
            getBlendZipExpect<CrossLaneOp::groupSize()>(input0, input1, threadIdx.x));
        DataT output = rocwmma::Blend::Driver<CrossLaneOp>::exec(input0, input1);

        return output != expect;
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_BLEND_OPS_HPP
