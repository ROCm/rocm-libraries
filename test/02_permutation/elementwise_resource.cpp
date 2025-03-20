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

#ifndef HIPTENSOR_PERMUTATION_RESOURCE_IMPL_HPP
#define HIPTENSOR_PERMUTATION_RESOURCE_IMPL_HPP

#include "elementwise_resource.hpp"
#include "data_types.hpp"
#include "utils.hpp"

namespace hiptensor
{

    ElementwiseResource::ElementwiseResource()
        : HipResource()
        , mDeviceInput1(Base::allocDevice(0))
        , mDeviceInput2(Base::allocDevice(0))
        , mDeviceInput3(Base::allocDevice(0))
        , mDeviceOutput(Base::allocDevice(0))
        , mHostInput1(Base::allocHost(0))
        , mHostInput2(Base::allocHost(0))
        , mHostInput3(Base::allocHost(0))
        , mHostOutput(Base::allocHost(0))
        , mCurrentMatrixElement(0)
        , mOpType(ElementwiseOp::PERMUTATION)
        , mCurrentDataType(HIP_R_32F)
        , mCurrentAllocByte(0)
    {
    }

    ElementwiseResource::ElementwiseResource(ElementwiseResource&& rhs)
        : HipResource()
        , mDeviceInput1(std::move(rhs.mDeviceInput1))
        , mDeviceInput2(std::move(rhs.mDeviceInput2))
        , mDeviceInput3(std::move(rhs.mDeviceInput3))
        , mDeviceOutput(std::move(rhs.mDeviceOutput))
        , mHostInput1(std::move(rhs.mHostInput1))
        , mHostInput2(std::move(rhs.mHostInput2))
        , mHostInput3(std::move(rhs.mHostInput3))
        , mHostOutput(std::move(rhs.mHostOutput))
        , mCurrentMatrixElement(rhs.mCurrentMatrixElement)
        , mOpType(rhs.mOpType)
        , mCurrentDataType(rhs.mCurrentDataType)
        , mCurrentAllocByte(rhs.mCurrentAllocByte)
    {
    }

    // suppose that all input and output data types are same.
    void ElementwiseResource::setupStorage(ProblemDims const& dimSizes,
                                           hipDataType        dataType,
                                           ElementwiseOp      opType)
    {
        mOpType                   = opType;
        auto requiredElementCount = getProduct(dimSizes);
        auto requiredMemorySize   = requiredElementCount * hipDataTypeSize(dataType);

        bool needFillData = false;
        if(requiredMemorySize > mCurrentAllocByte)
        {
            switch(mOpType)
            {
            case ElementwiseOp::TRINARY_OP:
                Base::reallocDeviceHostPair(mDeviceInput3, mHostInput3, requiredMemorySize);
                // no break;
            case ElementwiseOp::BINARY_OP:
                Base::reallocDeviceHostPair(mDeviceInput2, mHostInput2, requiredMemorySize);
                // no break;
            case ElementwiseOp::PERMUTATION:
                Base::reallocDeviceHostPair(mDeviceInput1, mHostInput1, requiredMemorySize);
                break;
            default:
                break;
            }
            Base::reallocDeviceHostPair(mDeviceOutput, mHostOutput, requiredMemorySize);
            Base::reallocDeviceHostPair(mDeviceReference, mHostReference, requiredMemorySize);
            mCurrentAllocByte = requiredMemorySize;
            needFillData      = true;
        }
        if(mCurrentDataType != dataType || mCurrentMatrixElement < requiredElementCount)
        {
            needFillData = true;
        }
        mCurrentMatrixElement = requiredElementCount;
        mCurrentDataType      = dataType;
        if(needFillData)
        {
            switch(mOpType)
            {
            case ElementwiseOp::TRINARY_OP:
                fillRandToInput3();
                // no break;
            case ElementwiseOp::BINARY_OP:
                fillRandToInput2();
                // no break;
            case ElementwiseOp::PERMUTATION:
                fillRandToInput1();
                break;
            default:
                break;
            }
        }
    }

    void ElementwiseResource::reset()
    {
        Base::reallocDeviceHostPair(mDeviceInput1, mHostInput1, 0);
        Base::reallocDeviceHostPair(mDeviceInput2, mHostInput2, 0);
        Base::reallocDeviceHostPair(mDeviceInput3, mHostInput3, 0);
        Base::reallocDeviceHostPair(mDeviceOutput, mHostOutput, 0);
        Base::reallocDeviceHostPair(mDeviceReference, mHostReference, 0);
        mCurrentMatrixElement = 0;
        mOpType               = ElementwiseOp::PERMUTATION;
        mCurrentDataType      = HIP_R_32F;
        mCurrentAllocByte     = 0;
    }

    void ElementwiseResource::fillRandToInput(HostPtrT& hostPtr, DevicePtrT& devicePtr)
    {
        uint32_t seed = static_cast<uint32_t>(256);

        if(mCurrentDataType == HIP_R_64F)
        {
            fillLaunchKernel<double>((double*)devicePtr.get(), mCurrentMatrixElement, seed);
        }
        else if(mCurrentDataType == HIP_R_32F)
        {
            fillLaunchKernel<float>((float*)devicePtr.get(), mCurrentMatrixElement, seed);
        }
        else
        {
            fillLaunchKernel<_Float16>((_Float16*)devicePtr.get(), mCurrentMatrixElement, seed);
        }
        Base::copyData(hostPtr, devicePtr, getCurrentMatrixMemorySize());
    }

    void ElementwiseResource::fillRandToInput1()
    {
        fillRandToInput(hostInput1(), deviceInput1());
    }

    void ElementwiseResource::fillRandToInput2()
    {
        fillRandToInput(hostInput2(), deviceInput2());
    }

    void ElementwiseResource::fillRandToInput3()
    {
        fillRandToInput(hostInput3(), deviceInput3());
    }

    void ElementwiseResource::copyOutputToHost()
    {
        Base::copyData(hostOutput(), deviceOutput(), getCurrentMatrixMemorySize());
    }

    void ElementwiseResource::copyReferenceToDevice()
    {
        Base::copyData(deviceReference(), hostReference(), getCurrentMatrixMemorySize());
    }

    size_t ElementwiseResource::getCurrentMatrixElement() const
    {
        return mCurrentMatrixElement;
    }

    size_t ElementwiseResource::getCurrentMatrixMemorySize() const
    {
        return mCurrentMatrixElement * hipDataTypeSize(mCurrentDataType);
    }

    auto ElementwiseResource::hostInput1() -> HostPtrT&
    {
        return mHostInput1;
    }

    auto ElementwiseResource::hostInput2() -> HostPtrT&
    {
        return mHostInput2;
    }

    auto ElementwiseResource::hostInput3() -> HostPtrT&
    {
        return mHostInput3;
    }

    auto ElementwiseResource::hostOutput() -> HostPtrT&
    {
        return mHostOutput;
    }

    auto ElementwiseResource::hostReference() -> HostPtrT&
    {
        return mHostReference;
    }

    auto ElementwiseResource::deviceInput1() -> DevicePtrT&
    {
        return mDeviceInput1;
    }

    auto ElementwiseResource::deviceInput2() -> DevicePtrT&
    {
        return mDeviceInput2;
    }

    auto ElementwiseResource::deviceInput3() -> DevicePtrT&
    {
        return mDeviceInput3;
    }

    auto ElementwiseResource::deviceOutput() -> DevicePtrT&
    {
        return mDeviceOutput;
    }

    auto ElementwiseResource::deviceReference() -> DevicePtrT&
    {
        return mDeviceReference;
    }
} // namespace hiptensor

#endif // HIPTENSOR_PERMUTATION_RESOURCE_IMPL_HPP
