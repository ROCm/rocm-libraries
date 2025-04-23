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

#ifndef ROCWMMA_GEMM_RESOURCE_IMPL_HPP
#define ROCWMMA_GEMM_RESOURCE_IMPL_HPP

#include "gemm_resource.hpp"

namespace rocwmma
{

    template <typename InputT, typename OutputT>
    GemmResource<InputT, OutputT>::GemmResource()
        : HipResource()
        , mDeviceA(Base::template allocDevice<InputT>(0))
        , mDeviceB(Base::template allocDevice<InputT>(0))
        , mDeviceC(Base::template allocDevice<OutputT>(0))
        , mDeviceD(Base::template allocDevice<OutputT>(0))
        , mHostA(Base::template allocHost<InputT>(0))
        , mHostB(Base::template allocHost<InputT>(0))
        , mHostC(Base::template allocHost<OutputT>(0))
        , mHostD(Base::template allocHost<OutputT>(0))
        , mCurrentMatrixElements({0, 0, 0, 0})
        , mCurrentAllocElements({0, 0, 0, 0})
    {
    }

    template <typename InputT, typename OutputT>
    GemmResource<InputT, OutputT>::GemmResource(GemmResource<InputT, OutputT>&& rhs)
        : HipResource()
        , mDeviceA(std::move(rhs.mDeviceA))
        , mDeviceB(std::move(rhs.mDeviceB))
        , mDeviceC(std::move(rhs.mDeviceC))
        , mDeviceD(std::move(rhs.mDeviceD))
        , mHostA(std::move(rhs.mHostA))
        , mHostB(std::move(rhs.mHostB))
        , mHostC(std::move(rhs.mHostC))
        , mHostD(std::move(rhs.mHostD))
        , mCurrentMatrixElements(rhs.mCurrentMatrixElements)
        , mCurrentAllocElements(rhs.mCurrentAllocElements)
    {
    }

    template <typename InputT, typename OutputT>
    void GemmResource<InputT, OutputT>::copyHostToDeviceAll()
    {
        Base::copyData(mDeviceA, mHostA, std::get<MatrixA>(mCurrentMatrixElements));
        Base::copyData(mDeviceB, mHostB, std::get<MatrixB>(mCurrentMatrixElements));
        Base::copyData(mDeviceC, mHostC, std::get<MatrixC>(mCurrentMatrixElements));
        Base::copyData(mDeviceD, mHostD, std::get<MatrixD>(mCurrentMatrixElements));
    }

    template <typename InputT, typename OutputT>
    void GemmResource<InputT, OutputT>::copyDeviceToHostAll()
    {
        Base::copyData(mHostA, mDeviceA, std::get<MatrixA>(mCurrentMatrixElements));
        Base::copyData(mHostB, mDeviceB, std::get<MatrixB>(mCurrentMatrixElements));
        Base::copyData(mHostC, mDeviceC, std::get<MatrixC>(mCurrentMatrixElements));
        Base::copyData(mHostD, mDeviceD, std::get<MatrixD>(mCurrentMatrixElements));
    }

    template <typename InputT, typename OutputT>
    void GemmResource<InputT, OutputT>::resizeStorage(ProblemDims const& size)
    {
        resizeStorage(
            std::make_tuple(std::get<M>(size) * std::get<K>(size), // elements MatrixA = M * K
                            std::get<K>(size) * std::get<N>(size), // elements MatrixB = K * N
                            std::get<M>(size) * std::get<N>(size), // elements MatrixC = M * N
                            std::get<M>(size) * std::get<N>(size))); // elements MatrixD = M * N)
    }

    template <typename InputT, typename OutputT>
    void GemmResource<InputT, OutputT>::resizeStorage(MatrixElements const& newMatrixElements)
    {
        auto conditionalReallocDeviceHostPair = [](auto&    devicePtr,
                                                   auto&    hostPtr,
                                                   int64_t& currentAllocElements,
                                                   int64_t  newAllocElements) {
            // Only realloc if required (e.g. current allocation won't fit new sizes)
            if(currentAllocElements < newAllocElements)
            {
                Base::reallocDeviceHostPair(devicePtr, hostPtr, newAllocElements);
                currentAllocElements = newAllocElements;
            }
        };

        conditionalReallocDeviceHostPair(mDeviceA,
                                         mHostA,
                                         std::get<MatrixA>(mCurrentAllocElements),
                                         std::get<MatrixA>(newMatrixElements));
        conditionalReallocDeviceHostPair(mDeviceB,
                                         mHostB,
                                         std::get<MatrixB>(mCurrentAllocElements),
                                         std::get<MatrixB>(newMatrixElements));
        conditionalReallocDeviceHostPair(mDeviceC,
                                         mHostC,
                                         std::get<MatrixC>(mCurrentAllocElements),
                                         std::get<MatrixC>(newMatrixElements));
        conditionalReallocDeviceHostPair(mDeviceD,
                                         mHostD,
                                         std::get<MatrixD>(mCurrentAllocElements),
                                         std::get<MatrixD>(newMatrixElements));

        // Always update the current matrix element count
        mCurrentMatrixElements = newMatrixElements;
    }

    template <typename InputT, typename OutputT>
    void GemmResource<InputT, OutputT>::reset()
    {
        Base::reallocDeviceHostPair(mDeviceA, mHostA, 0);
        Base::reallocDeviceHostPair(mDeviceB, mHostB, 0);
        Base::reallocDeviceHostPair(mDeviceC, mHostC, 0);
        Base::reallocDeviceHostPair(mDeviceD, mHostD, 0);
        mCurrentAllocElements  = {0, 0, 0, 0};
        mCurrentMatrixElements = {0, 0, 0, 0};
    }

    template <typename InputT, typename OutputT>
    auto GemmResource<InputT, OutputT>::hostA() -> HostPtrT<InputT>&
    {
        return mHostA;
    }

    template <typename InputT, typename OutputT>
    auto GemmResource<InputT, OutputT>::hostB() -> HostPtrT<InputT>&
    {
        return mHostB;
    }

    template <typename InputT, typename OutputT>
    auto GemmResource<InputT, OutputT>::hostC() -> HostPtrT<OutputT>&
    {
        return mHostC;
    }

    template <typename InputT, typename OutputT>
    auto GemmResource<InputT, OutputT>::hostD() -> HostPtrT<OutputT>&
    {
        return mHostD;
    }

    template <typename InputT, typename OutputT>
    auto GemmResource<InputT, OutputT>::deviceA() -> DevicePtrT<InputT>&
    {
        return mDeviceA;
    }

    template <typename InputT, typename OutputT>
    auto GemmResource<InputT, OutputT>::deviceB() -> DevicePtrT<InputT>&
    {
        return mDeviceB;
    }

    template <typename InputT, typename OutputT>
    auto GemmResource<InputT, OutputT>::deviceC() -> DevicePtrT<OutputT>&
    {
        return mDeviceC;
    }

    template <typename InputT, typename OutputT>
    auto GemmResource<InputT, OutputT>::deviceD() -> DevicePtrT<OutputT>&
    {
        return mDeviceD;
    }

} // namespace rocwmma

#endif // ROCWMMA_GEMM_RESOURCE_IMPL_HPP
