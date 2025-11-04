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

#ifndef ROCWMMA_DEVICE_CROSS_LANE_OPS_HPP
#define ROCWMMA_DEVICE_CROSS_LANE_OPS_HPP

#include "common.hpp"
#include "device/blend_ops.hpp"
#include "device/dpp_ops.hpp"
#include "device/permute_ops.hpp"
#include "device/swizzle_ops.hpp"
#include <rocwmma/rocwmma.hpp>

namespace rocwmma
{
    template <typename DataT, typename Func>
    ROCWMMA_DEVICE void crossLaneOpsTest(Func         op,
                                         uint32_t     m,
                                         uint32_t     n,
                                         DataT const* in,
                                         DataT*       out,
                                         uint32_t     ld,
                                         DataT        param1,
                                         DataT        param2)
    {
        __shared__ uint32_t result;
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0
           && blockIdx.y == 0 && blockIdx.z == 0)
        {
            result = 0;
        }
        synchronize_workgroup();

        bool err = false;

        // Add tests here
        err = err ? err : op();

        // Reduce error count
        atomicAdd(&result, err ? 1 : 0);

        // Wait for all threads
        synchronize_workgroup();

        // Just need one thread to update output
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0
           && blockIdx.y == 0 && blockIdx.z == 0)
        {
            out[0] = static_cast<DataT>(result == 0 ? SUCCESS_VALUE : ERROR_VALUE);
        }
    }
    template <typename DataT,
              typename CrossLaneOp,
              uint32_t WriteRowMask,
              uint32_t WriteBankMask,
              bool     BoundCtrl>
    ROCWMMA_KERNEL void dppOpsTest(uint32_t     m,
                                   uint32_t     n,
                                   DataT const* in,
                                   DataT*       out,
                                   uint32_t     ld,
                                   DataT        param1,
                                   DataT        param2)
    {
        crossLaneOpsTest<DataT>(
            dppOpsTestCase<DataT, CrossLaneOp, WriteRowMask, WriteBankMask, BoundCtrl>,
            m,
            n,
            in,
            out,
            ld,
            param1,
            param2);
    }

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_KERNEL void swizzleOpsTest(uint32_t     m,
                                       uint32_t     n,
                                       DataT const* in,
                                       DataT*       out,
                                       uint32_t     ld,
                                       DataT        param1,
                                       DataT        param2)
    {
        crossLaneOpsTest<DataT>(
            swizzleOpsTestCase<DataT, CrossLaneOp>, m, n, in, out, ld, param1, param2);
    }

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_KERNEL void permuteOpsTest(uint32_t     m,
                                       uint32_t     n,
                                       DataT const* in,
                                       DataT*       out,
                                       uint32_t     ld,
                                       DataT        param1,
                                       DataT        param2)
    {
        __shared__ uint32_t sharedSrcValues[Constants::AMDGCN_WAVE_SIZE];
        uint32_t*           srcValues = sharedSrcValues;
        crossLaneOpsTest<DataT>(
            [srcValues]() { return permuteOpsTestCase<DataT, CrossLaneOp>(srcValues); },
            m,
            n,
            in,
            out,
            ld,
            param1,
            param2);
    }

    template <typename DataT, typename CrossLaneOp>
    ROCWMMA_KERNEL void blendOpsTest(uint32_t     m,
                                     uint32_t     n,
                                     DataT const* in,
                                     DataT*       out,
                                     uint32_t     ld,
                                     DataT        param1,
                                     DataT        param2)
    {
        crossLaneOpsTest<DataT>(
            blendOpsTestCase<DataT, CrossLaneOp>, m, n, in, out, ld, param1, param2);
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_CROSS_LANE_OPS_HPP
