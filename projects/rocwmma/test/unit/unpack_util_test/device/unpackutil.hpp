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

#ifndef ROCWMMA_DEVICE_TRANSFORMS_TEST_HPP
#define ROCWMMA_DEVICE_TRANSFORMS_TEST_HPP

#include <rocwmma/rocwmma.hpp>

#include "hip_device.hpp"

namespace rocwmma
{
    /*************************************************************
     *     unpackLo
     *************************************************************/

    template <typename DataT, uint32_t VW>
    ROCWMMA_KERNEL void unpackLo2Test(uint32_t     m,
                                      uint32_t     n,
                                      DataT const* in,
                                      DataT*       out,
                                      uint32_t     ld,
                                      DataT        param1,
                                      DataT        param2)
    {
        using InVecT  = VecT<DataT, VW>;
        using OutVecT = VecT<DataT, VW / 2>;
        InVecT v      = *(reinterpret_cast<InVecT const*>(in + (uint32_t)threadIdx.x * VW));
        auto   result = unpackLo2(v);
        *(reinterpret_cast<OutVecT*>(out + (uint32_t)threadIdx.x * VW / 2)) = result;
    }

    template <typename DataT, uint32_t VW>
    ROCWMMA_KERNEL void unpackLo4Test(uint32_t     m,
                                      uint32_t     n,
                                      DataT const* in,
                                      DataT*       out,
                                      uint32_t     ld,
                                      DataT        param1,
                                      DataT        param2)
    {
        using InVecT  = VecT<DataT, VW>;
        using OutVecT = VecT<DataT, VW / 2>;
        InVecT v      = *(reinterpret_cast<InVecT const*>(in + (uint32_t)threadIdx.x * VW));
        auto   result = unpackLo4(v);
        *(reinterpret_cast<OutVecT*>(out + (uint32_t)threadIdx.x * VW / 2)) = result;
    }

    template <typename DataT, uint32_t VW>
    ROCWMMA_KERNEL void unpackLo8Test(uint32_t     m,
                                      uint32_t     n,
                                      DataT const* in,
                                      DataT*       out,
                                      uint32_t     ld,
                                      DataT        param1,
                                      DataT        param2)
    {
        using InVecT  = VecT<DataT, VW>;
        using OutVecT = VecT<DataT, VW / 2>;
        InVecT v      = *(reinterpret_cast<InVecT const*>(in + (uint32_t)threadIdx.x * VW));
        auto   result = unpackLo8(v);
        *(reinterpret_cast<OutVecT*>(out + (uint32_t)threadIdx.x * VW / 2)) = result;
    }

    /*************************************************************
     *     unpackHi
     *************************************************************/

    template <typename DataT, uint32_t VW>
    ROCWMMA_KERNEL void unpackHi2Test(uint32_t     m,
                                      uint32_t     n,
                                      DataT const* in,
                                      DataT*       out,
                                      uint32_t     ld,
                                      DataT        param1,
                                      DataT        param2)
    {
        using InVecT  = VecT<DataT, VW>;
        using OutVecT = VecT<DataT, VW / 2>;
        InVecT v      = *(reinterpret_cast<InVecT const*>(in + (uint32_t)threadIdx.x * VW));
        auto   result = unpackHi2(v);
        *(reinterpret_cast<OutVecT*>(out + (uint32_t)threadIdx.x * VW / 2)) = result;
    }

    template <typename DataT, uint32_t VW>
    ROCWMMA_KERNEL void unpackHi4Test(uint32_t     m,
                                      uint32_t     n,
                                      DataT const* in,
                                      DataT*       out,
                                      uint32_t     ld,
                                      DataT        param1,
                                      DataT        param2)
    {
        using InVecT  = VecT<DataT, VW>;
        using OutVecT = VecT<DataT, VW / 2>;
        InVecT v      = *(reinterpret_cast<InVecT const*>(in + (uint32_t)threadIdx.x * VW));
        auto   result = unpackHi4(v);
        *(reinterpret_cast<OutVecT*>(out + (uint32_t)threadIdx.x * VW / 2)) = result;
    }

    template <typename DataT, uint32_t VW>
    ROCWMMA_KERNEL void unpackHi8Test(uint32_t     m,
                                      uint32_t     n,
                                      DataT const* in,
                                      DataT*       out,
                                      uint32_t     ld,
                                      DataT        param1,
                                      DataT        param2)
    {
        using InVecT  = VecT<DataT, VW>;
        using OutVecT = VecT<DataT, VW / 2>;
        InVecT v      = *(reinterpret_cast<InVecT const*>(in + (uint32_t)threadIdx.x * VW));
        auto   result = unpackHi8(v);
        *(reinterpret_cast<OutVecT*>(out + (uint32_t)threadIdx.x * VW / 2)) = result;
    }

    /*************************************************************
     *     unpackLoHi
     *************************************************************/

    template <typename DataT, uint32_t VW>
    ROCWMMA_KERNEL void unpackLoHi1Test(uint32_t     m,
                                        uint32_t     n,
                                        DataT const* in,
                                        DataT*       out,
                                        uint32_t     ld,
                                        DataT        param1,
                                        DataT        param2)
    {
        using InVecT  = VecT<DataT, VW>;
        using OutVecT = VecT<DataT, VW>;
        InVecT v      = *(reinterpret_cast<InVecT const*>(in + (uint32_t)threadIdx.x * VW));
        auto   result = unpackLoHi1(v);
        *(reinterpret_cast<OutVecT*>(out + (uint32_t)threadIdx.x * VW)) = result;
    }

    template <typename DataT, uint32_t VW>
    ROCWMMA_KERNEL void unpackLoHi2Test(uint32_t     m,
                                        uint32_t     n,
                                        DataT const* in,
                                        DataT*       out,
                                        uint32_t     ld,
                                        DataT        param1,
                                        DataT        param2)
    {
        using InVecT  = VecT<DataT, VW>;
        using OutVecT = VecT<DataT, VW>;
        InVecT v      = *(reinterpret_cast<InVecT const*>(in + (uint32_t)threadIdx.x * VW));
        auto   result = unpackLoHi2(v);
        *(reinterpret_cast<OutVecT*>(out + (uint32_t)threadIdx.x * VW)) = result;
    }

    template <typename DataT, uint32_t VW>
    ROCWMMA_KERNEL void unpackLoHi4Test(uint32_t     m,
                                        uint32_t     n,
                                        DataT const* in,
                                        DataT*       out,
                                        uint32_t     ld,
                                        DataT        param1,
                                        DataT        param2)
    {
        using InVecT  = VecT<DataT, VW>;
        using OutVecT = VecT<DataT, VW>;
        InVecT v      = *(reinterpret_cast<InVecT const*>(in + (uint32_t)threadIdx.x * VW));
        auto   result = unpackLoHi4(v);
        *(reinterpret_cast<OutVecT*>(out + (uint32_t)threadIdx.x * VW)) = result;
    }

    template <typename DataT, uint32_t VW>
    ROCWMMA_KERNEL void unpackLoHi8Test(uint32_t     m,
                                        uint32_t     n,
                                        DataT const* in,
                                        DataT*       out,
                                        uint32_t     ld,
                                        DataT        param1,
                                        DataT        param2)
    {
        using InVecT  = VecT<DataT, VW>;
        using OutVecT = VecT<DataT, VW>;
        InVecT v      = *(reinterpret_cast<InVecT const*>(in + (uint32_t)threadIdx.x * VW));
        auto   result = unpackLoHi8(v);
        *(reinterpret_cast<OutVecT*>(out + (uint32_t)threadIdx.x * VW)) = result;
    }

    template <typename DataT, uint32_t VW>
    ROCWMMA_KERNEL void unpackLoHi16Test(uint32_t     m,
                                         uint32_t     n,
                                         DataT const* in,
                                         DataT*       out,
                                         uint32_t     ld,
                                         DataT        param1,
                                         DataT        param2)
    {
        using InVecT  = VecT<DataT, VW>;
        using OutVecT = VecT<DataT, VW>;
        InVecT v      = *(reinterpret_cast<InVecT const*>(in + (uint32_t)threadIdx.x * VW));
        auto   result = unpackLoHi16(v);
        *(reinterpret_cast<OutVecT*>(out + (uint32_t)threadIdx.x * VW)) = result;
    }

    template <typename DataT, uint32_t VW>
    ROCWMMA_KERNEL void unpackLoHi32Test(uint32_t     m,
                                         uint32_t     n,
                                         DataT const* in,
                                         DataT*       out,
                                         uint32_t     ld,
                                         DataT        param1,
                                         DataT        param2)
    {
        using InVecT  = VecT<DataT, VW>;
        using OutVecT = VecT<DataT, VW>;
        InVecT v      = *(reinterpret_cast<InVecT const*>(in + (uint32_t)threadIdx.x * VW));
        auto   result = unpackLoHi32(v);
        // printf("result: %03d, %f, %f, %f, %f\n", (int)threadIdx.x, (double)result.data[0], (double)result.data[1], (double)result.data[2], (double)result.data[3]);
        *(reinterpret_cast<OutVecT*>(out + (uint32_t)threadIdx.x * VW)) = result;
    }
} // namespace rocwmma

#endif // ROCWMMA_DEVICE_TRANSFORMS_TEST_HPP
