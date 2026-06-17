/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <numeric>

#include "trinary_contraction_resource.hpp"

namespace hiptensor
{

    TrinaryContractionResource::TrinaryContractionResource()
        : HipResource()
        , mDeviceA(Base::allocDevice(0))
        , mDeviceB(Base::allocDevice(0))
        , mDeviceC(Base::allocDevice(0))
        , mDeviceD(Base::allocDevice(0))
        , mDeviceE(Base::allocDevice(0))
        , mHostA(Base::allocHost(0))
        , mHostB(Base::allocHost(0))
        , mHostC(Base::allocHost(0))
        , mHostD(Base::allocHost(0))
        , mHostE(Base::allocHost(0))
        , mCurrentElementBytes({0, 0, 0, 0, 0})
        , mCurrentAllocBytes({0, 0, 0, 0, 0})
        , mCurrentMatrixElements({0, 0, 0, 0, 0})
        , mCurrentAllocElements({0, 0, 0, 0, 0})
    {
    }

    TrinaryContractionResource::TrinaryContractionResource(TrinaryContractionResource&& rhs)
        : HipResource()
        , mDeviceA(std::move(rhs.mDeviceA))
        , mDeviceB(std::move(rhs.mDeviceB))
        , mDeviceC(std::move(rhs.mDeviceC))
        , mDeviceD(std::move(rhs.mDeviceD))
        , mDeviceE(std::move(rhs.mDeviceE))
        , mHostA(std::move(rhs.mHostA))
        , mHostB(std::move(rhs.mHostB))
        , mHostC(std::move(rhs.mHostC))
        , mHostD(std::move(rhs.mHostD))
        , mHostE(std::move(rhs.mHostE))
        , mCurrentElementBytes(rhs.mCurrentElementBytes)
        , mCurrentAllocBytes(rhs.mCurrentAllocBytes)
        , mCurrentMatrixElements(rhs.mCurrentMatrixElements)
        , mCurrentAllocElements(rhs.mCurrentAllocElements)
    {
    }

    void TrinaryContractionResource::copyHostToDeviceAll(ElementBytes const& bytesPerElement)
    {
        Base::copyData(mDeviceA, mHostA,
                       std::get<TensorA>(mCurrentMatrixElements) * std::get<TensorA>(bytesPerElement));
        Base::copyData(mDeviceB, mHostB,
                       std::get<TensorB>(mCurrentMatrixElements) * std::get<TensorB>(bytesPerElement));
        Base::copyData(mDeviceC, mHostC,
                       std::get<TensorC>(mCurrentMatrixElements) * std::get<TensorC>(bytesPerElement));
        Base::copyData(mDeviceD, mHostD,
                       std::get<TensorD>(mCurrentMatrixElements) * std::get<TensorD>(bytesPerElement));
        Base::copyData(mDeviceE, mHostE,
                       std::get<TensorE>(mCurrentMatrixElements) * std::get<TensorE>(bytesPerElement));
    }

    void TrinaryContractionResource::copyDeviceToHostAll(ElementBytes const& bytesPerElement)
    {
        Base::copyData(mHostA, mDeviceA,
                       std::get<TensorA>(mCurrentMatrixElements) * std::get<TensorA>(bytesPerElement));
        Base::copyData(mHostB, mDeviceB,
                       std::get<TensorB>(mCurrentMatrixElements) * std::get<TensorB>(bytesPerElement));
        Base::copyData(mHostC, mDeviceC,
                       std::get<TensorC>(mCurrentMatrixElements) * std::get<TensorC>(bytesPerElement));
        Base::copyData(mHostD, mDeviceD,
                       std::get<TensorD>(mCurrentMatrixElements) * std::get<TensorD>(bytesPerElement));
        Base::copyData(mHostE, mDeviceE,
                       std::get<TensorE>(mCurrentMatrixElements) * std::get<TensorE>(bytesPerElement));
    }

    void TrinaryContractionResource::resizeStorage(ProblemDims const& dimSizes,
                                                   ElementBytes       bytesPerElement)
    {
        // dimSizes: [A_lengths, B_lengths, C_lengths, DE_lengths]
        int aSize = std::accumulate(
            dimSizes[0].begin(), dimSizes[0].end(), int{1}, std::multiplies<int>());
        int bSize = std::accumulate(
            dimSizes[1].begin(), dimSizes[1].end(), int{1}, std::multiplies<int>());
        int cSize = std::accumulate(
            dimSizes[2].begin(), dimSizes[2].end(), int{1}, std::multiplies<int>());
        int deSize = std::accumulate(
            dimSizes[3].begin(), dimSizes[3].end(), int{1}, std::multiplies<int>());

        resizeStorage(std::make_tuple(aSize, bSize, cSize, deSize, deSize), bytesPerElement);
    }

    void TrinaryContractionResource::resizeStorage(MatrixElements const& newMatrixElements,
                                                   ElementBytes          bytesPerElement)
    {
        auto conditionalReallocDeviceHostPair = [](auto&    devicePtr,
                                                   auto&    hostPtr,
                                                   int64_t& currentAllocElements,
                                                   int64_t  newAllocElements,
                                                   int32_t& currentElementBytes,
                                                   int32_t  newElementBytes) {
            if((currentAllocElements * currentElementBytes) < (newAllocElements * newElementBytes))
            {
                Base::reallocDeviceHostPair(devicePtr, hostPtr, newAllocElements * newElementBytes);
                currentAllocElements = newAllocElements;
                currentElementBytes  = newElementBytes;
            }
        };

        conditionalReallocDeviceHostPair(mDeviceA, mHostA,
                                         std::get<TensorA>(mCurrentAllocElements),
                                         std::get<TensorA>(newMatrixElements),
                                         std::get<TensorA>(mCurrentElementBytes),
                                         std::get<TensorA>(bytesPerElement));
        conditionalReallocDeviceHostPair(mDeviceB, mHostB,
                                         std::get<TensorB>(mCurrentAllocElements),
                                         std::get<TensorB>(newMatrixElements),
                                         std::get<TensorB>(mCurrentElementBytes),
                                         std::get<TensorB>(bytesPerElement));
        conditionalReallocDeviceHostPair(mDeviceC, mHostC,
                                         std::get<TensorC>(mCurrentAllocElements),
                                         std::get<TensorC>(newMatrixElements),
                                         std::get<TensorC>(mCurrentElementBytes),
                                         std::get<TensorC>(bytesPerElement));
        conditionalReallocDeviceHostPair(mDeviceD, mHostD,
                                         std::get<TensorD>(mCurrentAllocElements),
                                         std::get<TensorD>(newMatrixElements),
                                         std::get<TensorD>(mCurrentElementBytes),
                                         std::get<TensorD>(bytesPerElement));
        conditionalReallocDeviceHostPair(mDeviceE, mHostE,
                                         std::get<TensorE>(mCurrentAllocElements),
                                         std::get<TensorE>(newMatrixElements),
                                         std::get<TensorE>(mCurrentElementBytes),
                                         std::get<TensorE>(bytesPerElement));

        mCurrentMatrixElements = newMatrixElements;
        mCurrentAllocBytes     = bytesPerElement;
    }

    void TrinaryContractionResource::reset()
    {
        Base::reallocDeviceHostPair(mDeviceA, mHostA, 0);
        Base::reallocDeviceHostPair(mDeviceB, mHostB, 0);
        Base::reallocDeviceHostPair(mDeviceC, mHostC, 0);
        Base::reallocDeviceHostPair(mDeviceD, mHostD, 0);
        Base::reallocDeviceHostPair(mDeviceE, mHostE, 0);
        mCurrentAllocElements  = {0, 0, 0, 0, 0};
        mCurrentMatrixElements = {0, 0, 0, 0, 0};
    }

    auto TrinaryContractionResource::hostA() -> HostPtrT& { return mHostA; }
    auto TrinaryContractionResource::hostB() -> HostPtrT& { return mHostB; }
    auto TrinaryContractionResource::hostC() -> HostPtrT& { return mHostC; }
    auto TrinaryContractionResource::hostD() -> HostPtrT& { return mHostD; }
    auto TrinaryContractionResource::hostE() -> HostPtrT& { return mHostE; }

    auto TrinaryContractionResource::deviceA() -> DevicePtrT& { return mDeviceA; }
    auto TrinaryContractionResource::deviceB() -> DevicePtrT& { return mDeviceB; }
    auto TrinaryContractionResource::deviceC() -> DevicePtrT& { return mDeviceC; }
    auto TrinaryContractionResource::deviceD() -> DevicePtrT& { return mDeviceD; }
    auto TrinaryContractionResource::deviceE() -> DevicePtrT& { return mDeviceE; }

} // namespace hiptensor
