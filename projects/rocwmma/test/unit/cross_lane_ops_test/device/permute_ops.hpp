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

#ifndef ROCWMMA_DEVICE_PERMUTE_OPS_HPP
#define ROCWMMA_DEVICE_PERMUTE_OPS_HPP

#include "cross_lane_ops_util.hpp"

namespace rocwmma
{
    /**
	 * @addtogroup cross_lane_op_gen_ref_value_funcs
	 * @{
	 */
    template <uint32_t BlockIdx, uint32_t BlockSize>
    ROCWMMA_DEVICE inline uint32_t getPermuteBlockBCastExpect(uint32_t input)
    {
        auto idxInBlock = input % BlockSize;
        input           = BlockIdx * BlockSize + idxInBlock;
        return input;
    }

    template <uint32_t SubgroupSize, uint32_t Direction, uint32_t Distance>
    ROCWMMA_DEVICE inline uint32_t getPermuteRotateExpect(uint32_t input)
    {
        auto afterRotate = (input & (SubgroupSize - 1));
        afterRotate += Direction == CrossLaneOps::OP_DIR_L ? Distance : -Distance;
        afterRotate += SubgroupSize;
        afterRotate &= (SubgroupSize - 1);
        return (input & (~(SubgroupSize - 1))) | afterRotate;
    }

    template <uint32_t SubGroupSize, uint32_t VW, uint32_t Shift>
    ROCWMMA_DEVICE inline uint32_t getPermuteGatherExpect(uint32_t input)
    {
        const uint32_t offset0 = (input * SubGroupSize / VW + Shift) % SubGroupSize;
        const uint32_t offset1 = input / VW % (SubGroupSize / VW);
        const uint32_t offset2 = (input / SubGroupSize) * SubGroupSize;
        return (offset0 + offset1 + offset2) % Constants::AMDGCN_WAVE_SIZE;
    }

    template <uint32_t SubGroupSize, uint32_t VW, uint32_t Shift>
    ROCWMMA_DEVICE inline uint32_t getPermuteScatterExpect(uint32_t input, uint32_t* srcValues)
    {
        const uint32_t offset0 = (input * SubGroupSize / VW + Shift) % SubGroupSize;
        const uint32_t offset1 = input / VW % (SubGroupSize / VW);
        const uint32_t offset2 = (input / SubGroupSize) * SubGroupSize;
        auto           srcIdx  = (offset0 + offset1 + offset2) % Constants::AMDGCN_WAVE_SIZE;
        srcValues[srcIdx]      = input;
        __syncthreads();
        return srcValues[threadIdx.x];
    }
    /** @} */

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_BLOCK_BCAST
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_BPERMUTE,
                                    bool>
                   permuteOpsTestCase(uint32_t* /*srcValues*/)
    {
        uint32_t id     = threadIdx.x;
        DataT    input  = makeValueFromU32<DataT>(id);
        DataT    expect = makeValueFromU32<DataT>(
            getPermuteBlockBCastExpect<CrossLaneOp::ELEMENT_IDX, CrossLaneOp::GROUP_SIZE>(id));
        DataT output = rocwmma::Permute::Driver<CrossLaneOp>::exec(input);

        return output != expect;
    }

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_ROTATE
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_BPERMUTE,
                                    bool>
                   permuteOpsTestCase(uint32_t* /*srcValues*/)
    {
        uint32_t id     = threadIdx.x;
        DataT    input  = makeValueFromU32<DataT>(id);
        DataT    expect = makeValueFromU32<DataT>(getPermuteRotateExpect<CrossLaneOp::GROUP_SIZE,
                                                                      CrossLaneOps::OP_DIR_L,
                                                                      CrossLaneOp::opDist()>(id));
        DataT    output = rocwmma::Permute::Driver<CrossLaneOp>::exec(input);

        return output != expect;
    }

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_ROTATE
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_PERMUTE,
                                    bool>
                   permuteOpsTestCase(uint32_t* /*srcValues*/)
    {
        uint32_t id     = threadIdx.x;
        DataT    input  = makeValueFromU32<DataT>(id);
        DataT    expect = makeValueFromU32<DataT>(getPermuteRotateExpect<CrossLaneOp::GROUP_SIZE,
                                                                      CrossLaneOps::OP_DIR_R,
                                                                      CrossLaneOp::opDist()>(id));
        DataT    output = rocwmma::Permute::Driver<CrossLaneOp>::exec(input);

        return output != expect;
    }

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_GATHER
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_BPERMUTE,
                                    bool>
                   permuteOpsTestCase(uint32_t* /*srcValues*/)
    {
        uint32_t id     = threadIdx.x;
        DataT    input  = makeValueFromU32<DataT>(id);
        DataT    expect = makeValueFromU32<DataT>(getPermuteGatherExpect<CrossLaneOp::GROUP_SIZE,
                                                                      CrossLaneOp::vw(),
                                                                      CrossLaneOp::shift()>(id));
        DataT    output = rocwmma::Permute::Driver<CrossLaneOp>::exec(input);

        return output != expect;
    }

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_DEVICE std::enable_if_t<CrossLaneOp::opId() == CrossLaneOps::OP_ID_SCATTER
                                        && CrossLaneOp::opImpl() == CrossLaneOps::OP_IMPL_PERMUTE,
                                    bool>
                   permuteOpsTestCase(uint32_t* srcValues)
    {
        uint32_t id = threadIdx.x;
        if(id < Constants::AMDGCN_WAVE_SIZE)
        {
            srcValues[id] = id;
        }

        DataT input = makeValueFromU32<DataT>(id);
        DataT expect
            = makeValueFromU32<DataT>(getPermuteScatterExpect<CrossLaneOp::GROUP_SIZE,
                                                              CrossLaneOp::vw(),
                                                              CrossLaneOp::shift()>(id, srcValues));
        DataT output = rocwmma::Permute::Driver<CrossLaneOp>::exec(input);

        return output != expect;
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_PERMUTE_OPS_HPP
