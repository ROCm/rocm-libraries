/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPTENSOR_REDUCTION_RESOURCE_IMPL_HPP
#define HIPTENSOR_REDUCTION_RESOURCE_IMPL_HPP

#include "reduction_resource.hpp"
#include "data_types.hpp"
#include "utils.hpp"

namespace hiptensor
{

    ReductionResource::ReductionResource()
        : HipResource()
        , mDeviceA(Base::allocDevice(0))
        , mDeviceC(Base::allocDevice(0))
        , mDeviceD(Base::allocDevice(0))
        , mHostA(Base::allocHost(0))
        , mHostC(Base::allocHost(0))
        , mHostD(Base::allocHost(0))
        , mCurrentDataType(HIP_R_32F)
        , mCurrentInputElementCount(0)
        , mCurrentInputAllocByte(0)
        , mCurrentOutputElementCount(0)
        , mCurrentOutputAllocByte(0)
    {
    }

    ReductionResource::ReductionResource(ReductionResource&& rhs)
        : HipResource()
        , mDeviceA(std::move(rhs.mDeviceA))
        , mDeviceC(std::move(rhs.mDeviceC))
        , mDeviceD(std::move(rhs.mDeviceD))
        , mHostA(std::move(rhs.mHostA))
        , mHostC(std::move(rhs.mHostC))
        , mHostD(std::move(rhs.mHostD))
        , mCurrentDataType(rhs.mCurrentDataType)
        , mCurrentInputElementCount(rhs.mCurrentInputElementCount)
        , mCurrentInputAllocByte(rhs.mCurrentInputAllocByte)
        , mCurrentOutputElementCount(rhs.mCurrentOutputElementCount)
        , mCurrentOutputAllocByte(rhs.mCurrentOutputAllocByte)
    {
    }

    void ReductionResource::setupStorage(ProblemDims const& dimSizes,
                                         ProblemDims const& outputSizes,
                                         hipDataType        dataType)
    {
        // check buffer for A
        auto requiredInputElementCount = getProduct(dimSizes);
        auto requiredInputMemorySize   = requiredInputElementCount * hipDataTypeSize(dataType);

        bool needFillDataA = false;
        if(requiredInputMemorySize > mCurrentInputAllocByte)
        {
            Base::reallocDeviceHostPair(mDeviceA, mHostA, requiredInputMemorySize);
            mCurrentInputAllocByte = requiredInputMemorySize;
            needFillDataA          = true;
        }
        if(mCurrentDataType != dataType || mCurrentInputElementCount < requiredInputElementCount)
        {
            needFillDataA = true;
        }
        mCurrentInputElementCount = requiredInputElementCount;

        // check buffer for C
        bool needFillDataC = false;
        auto requiredOutputElementCount
            = getProduct(outputSizes); // returns 1 for empty container which is correct
        auto requiredOutputMemorySize = requiredOutputElementCount * hipDataTypeSize(dataType);
        if(requiredOutputMemorySize > mCurrentOutputAllocByte)
        {
            Base::reallocDeviceHostPair(mDeviceC, mHostC, requiredOutputMemorySize);
            Base::reallocDeviceHostPair(mDeviceD, mHostD, requiredOutputMemorySize);
            Base::reallocDeviceHostPair(mDeviceReference, mHostReference, requiredOutputMemorySize);
            mCurrentOutputAllocByte = requiredOutputMemorySize;
            needFillDataC           = true;
        }
        if(mCurrentDataType != dataType || mCurrentOutputElementCount < requiredOutputElementCount)
        {
            needFillDataC = true;
        }
        mCurrentOutputElementCount = requiredOutputElementCount;

        mCurrentDataType = dataType;

        const uint32_t seedA = 17;
        const uint32_t seedC = 19;
        if(needFillDataA)
        {
            fillRand(hostA(), deviceA(), dataType, getCurrentInputElementCount(), seedA);
        }

        if(needFillDataC)
        {
            fillRand(hostC(), deviceC(), dataType, getCurrentOutputElementCount(), seedC);
        }
        fillConstant(hostD(), deviceD(), dataType, getCurrentOutputElementCount(), 0);
        copyData(hostReference(), hostD(), getCurrentOutputMemorySize());
    }

    void ReductionResource::setupWorkspace(uint64_t workspaceSize)
    {
        reallocDevice(mDeviceWorkspace, workspaceSize);
    }

    void ReductionResource::reset()
    {
        Base::reallocDeviceHostPair(mDeviceA, mHostA, 0);
        Base::reallocDeviceHostPair(mDeviceC, mHostC, 0);
        Base::reallocDeviceHostPair(mDeviceD, mHostD, 0);
        Base::reallocDeviceHostPair(mDeviceReference, mHostReference, 0);
        mCurrentDataType           = HIP_R_32F;
        mCurrentInputElementCount  = 0;
        mCurrentInputAllocByte     = 0;
        mCurrentOutputElementCount = 0;
        mCurrentOutputAllocByte    = 0;
    }

    void ReductionResource::fillRand(HostPtrT&   hostBuf,
                                     DevicePtrT& deviceBuf,
                                     hipDataType dataType,
                                     size_t      elementCount,
                                     uint32_t    seed)
    {
        if(dataType == HIP_R_16F)
        {
            fillLaunchKernel<float16_t>((float16_t*)deviceBuf.get(), elementCount, seed);
        }
        else if(dataType == HIP_R_16BF)
        {
            fillLaunchKernel<bfloat16_t>((bfloat16_t*)deviceBuf.get(), elementCount, seed);
        }
        else if(dataType == HIP_R_32F)
        {
            fillLaunchKernel<float32_t>((float32_t*)deviceBuf.get(), elementCount, seed);
        }
        else if(dataType == HIP_R_64F)
        {
            fillLaunchKernel<float64_t>((float64_t*)deviceBuf.get(), elementCount, seed);
        }
        Base::copyData(hostBuf, deviceBuf, elementCount * hipDataTypeSize(dataType));
    }

    void ReductionResource::fillConstant(HostPtrT&   hostBuf,
                                         DevicePtrT& deviceBuf,
                                         hipDataType dataType,
                                         size_t      elementCount,
                                         double      value)
    {
        if(dataType == HIP_R_16F)
        {
            fillValLaunchKernel<float16_t>(
                (float16_t*)deviceBuf.get(), elementCount, (float16_t)value);
        }
        else if(dataType == HIP_R_16BF)
        {
            fillValLaunchKernel<bfloat16_t>(
                (bfloat16_t*)deviceBuf.get(), elementCount, (bfloat16_t)value);
        }
        else if(dataType == HIP_R_32F)
        {
            fillValLaunchKernel<float32_t>(
                (float32_t*)deviceBuf.get(), elementCount, (float32_t)value);
        }
        else if(dataType == HIP_R_64F)
        {
            fillValLaunchKernel<float64_t>(
                (float64_t*)deviceBuf.get(), elementCount, (float64_t)value);
        }
        Base::copyData(hostBuf, deviceBuf, elementCount * hipDataTypeSize(dataType));
    }

    void ReductionResource::copyOutputToHost()
    {
        Base::copyData(hostD(), deviceD(), getCurrentOutputMemorySize());
    }

    void ReductionResource::copyReferenceToDevice()
    {
        Base::copyData(deviceReference(), hostReference(), getCurrentOutputMemorySize());
    }

    size_t ReductionResource::getCurrentInputElementCount() const
    {
        return mCurrentInputElementCount;
    }

    size_t ReductionResource::getCurrentInputMemorySize() const
    {
        return mCurrentInputElementCount * hipDataTypeSize(mCurrentDataType);
    }

    size_t ReductionResource::getCurrentOutputElementCount() const
    {
        return mCurrentOutputElementCount;
    }

    size_t ReductionResource::getCurrentOutputMemorySize() const
    {
        return mCurrentOutputElementCount * hipDataTypeSize(mCurrentDataType);
    }

    auto ReductionResource::hostA() -> HostPtrT&
    {
        return mHostA;
    }

    auto ReductionResource::hostC() -> HostPtrT&
    {
        return mHostC;
    }

    auto ReductionResource::hostD() -> HostPtrT&
    {
        return mHostD;
    }

    auto ReductionResource::hostReference() -> HostPtrT&
    {
        return mHostReference;
    }

    auto ReductionResource::deviceA() -> DevicePtrT&
    {
        return mDeviceA;
    }

    auto ReductionResource::deviceC() -> DevicePtrT&
    {
        return mDeviceC;
    }

    auto ReductionResource::deviceD() -> DevicePtrT&
    {
        return mDeviceD;
    }

    auto ReductionResource::deviceReference() -> DevicePtrT&
    {
        return mDeviceReference;
    }

    auto ReductionResource::deviceWorkspace() -> DevicePtrT&
    {
        return mDeviceWorkspace;
    }
} // namespace hiptensor

#endif // HIPTENSOR_REDUCTION_RESOURCE_IMPL_HPP
