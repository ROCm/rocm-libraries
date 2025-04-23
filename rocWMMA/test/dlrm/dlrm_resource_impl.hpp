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

#ifndef DLRM_GEMM_RESOURCE_IMPL_HPP
#define DLRM_GEMM_RESOURCE_IMPL_HPP

#include "dlrm_resource.hpp"

namespace rocwmma
{

    template <typename DataT>
    DlrmResource<DataT>::DlrmResource()
        : mDeviceInput(Base::template allocDevice<DataT>(0))
        , mDeviceOutput(Base::template allocDevice<DataT>(0))
        , mDeviceAccFwd(Base::template allocDevice<float>(0))
        , mDeviceUpstreamGrad(Base::template allocDevice<DataT>(0))
        , mDeviceGrad(Base::template allocDevice<DataT>(0))
        , mDeviceBottomMlpGrad(Base::template allocDevice<DataT>(0))
        , mDeviceAccBwd(Base::template allocDevice<DataT>(0))
        , mHostInput(Base::template allocHost<DataT>(0))
        , mHostOutput(Base::template allocHost<DataT>(0))
        , mHostOutputRef(Base::template allocHost<DataT>(0))
        , mHostAccFwd(Base::template allocHost<float>(0))
        , mHostUpstreamGrad(Base::template allocHost<DataT>(0))
        , mHostGrad(Base::template allocHost<DataT>(0))
        , mHostGradRef(Base::template allocHost<DataT>(0))
        , mHostBottomMlpGrad(Base::template allocHost<DataT>(0))
        , mHostBottomMlpGradRef(Base::template allocHost<DataT>(0))
        , mHostAccBwd(Base::template allocHost<DataT>(0))
        , mCurrentElementCountFwd({0, 0, 0, DummyT()})
        , mCurrentElementCountBwd({0, 0, 0, 0, 0})
        , mMaxFwdCapacity({0, 0, 0, DummyT()})
        , mMaxBwdCapacity({0, 0, 0, 0, 0})
    {
    }

    template <typename DataT>
    DlrmResource<DataT>::DlrmResource(DlrmResource<DataT>&& rhs)
        : HipResource()
        , mDeviceInput(std::move(rhs.mDeviceInput))
        , mDeviceOutput(std::move(rhs.mDeviceOutput))
        , mDeviceAccFwd(std::move(rhs.mDeviceAccFwd))
        , mDeviceUpstreamGrad(std::move(rhs.mDeviceUpstreamGrad))
        , mDeviceGrad(std::move(rhs.mDeviceGrad))
        , mDeviceBottomMlpGrad(std::move(rhs.mDeviceBottomMlpGrad))
        , mDeviceAccBwd(std::move(rhs.mDeviceAccBwd))
        , mHostInput(std::move(rhs.mHostInput))
        , mHostOutput(std::move(rhs.mHostOutput))
        , mHostOutputRef(std::move(rhs.mHostOutputRef))
        , mHostAccFwd(std::move(rhs.mHostAccFwd))
        , mHostUpstreamGrad(std::move(rhs.mHostUpstreamGrad))
        , mHostGrad(std::move(rhs.mHostGrad))
        , mHostGradRef(std::move(rhs.mHostGradRef))
        , mHostBottomMlpGrad(std::move(rhs.mHostBottomMlpGrad))
        , mHostBottomMlpGradRef(std::move(rhs.mHostBottomMlpGradRef))
        , mHostAccBwd(std::move(rhs.mHostAccBwd))
        , mCurrentElementCountFwd(rhs.mCurrentElementCountFwd)
        , mCurrentElementCountBwd(rhs.mCurrentElementCountBwd)
        , mMaxFwdCapacity(rhs.mMaxFwdCapacity)
        , mMaxBwdCapacity(rhs.mMaxBwdCapacity)
    {
    }

    template <typename DataT>
    template <typename T>
    inline void DlrmResource<DataT>::conditionalReallocDeviceHostPair(DevicePtrT<T>& devicePtr,
                                                                      HostPtrT<T>&   hostPtr,
                                                                      int64_t&       currentMax,
                                                                      int64_t        newSize)
    {
        if(currentMax < newSize)
        {
            Base::reallocDeviceHostPair(devicePtr, hostPtr, newSize);
            currentMax = newSize;
        }
    }

    template <typename DataT>
    inline int64_t DlrmResource<DataT>::calcTrilSize(ProblemSize const& size)
    {
        return ((std::get<M>(size) * (std::get<M>(size) - 1)) / 2) + std::get<K>(size);
    };

    template <typename DataT>
    void DlrmResource<DataT>::copyHostToDeviceFwdAll()
    {
        Base::copyData(mDeviceInput, mHostInput, std::get<Input>(mCurrentElementCountFwd));
    }

    template <typename DataT>
    void DlrmResource<DataT>::copyHostToDeviceBwdAll()
    {
        Base::copyData(mDeviceInput, mHostInput, std::get<Input>(mCurrentElementCountBwd));
        Base::copyData(mDeviceUpstreamGrad,
                       mHostUpstreamGrad,
                       std::get<UpstreamGrad>(mCurrentElementCountBwd));
    }

    template <typename DataT>
    void DlrmResource<DataT>::copyDeviceToHostFwdInput()
    {
        Base::copyData(mHostInput, mDeviceInput, std::get<Input>(mCurrentElementCountFwd));
    }

    template <typename DataT>
    void DlrmResource<DataT>::copyDeviceToHostFwdOutput()
    {
        Base::copyData(mHostOutput, mDeviceOutput, std::get<Output>(mCurrentElementCountFwd));
    }

    template <typename DataT>
    void DlrmResource<DataT>::copyDeviceToHostBwdInput()
    {
        Base::copyData(mHostInput, mDeviceInput, std::get<Input>(mCurrentElementCountBwd));
        Base::copyData(mHostUpstreamGrad,
                       mDeviceUpstreamGrad,
                       std::get<UpstreamGrad>(mCurrentElementCountBwd));
    }

    template <typename DataT>
    void DlrmResource<DataT>::copyDeviceToHostBwdOutput()
    {
        Base::copyData(mHostGrad, mDeviceGrad, std::get<Grad>(mCurrentElementCountBwd));
        Base::copyData(mHostBottomMlpGrad,
                       mDeviceBottomMlpGrad,
                       std::get<BottomMlpGrad>(mCurrentElementCountBwd));
        Base::copyData(mHostAccBwd, mDeviceAccBwd, std::get<Acc>(mCurrentElementCountBwd));
    }

    template <typename DataT>
    void DlrmResource<DataT>::resizeFwdStorage(ProblemSize const& size)
    {
        resizeFwdStorage(
            std::make_tuple(std::get<M>(size) * std::get<K>(size) * std::get<B>(size), // Input
                            calcTrilSize(size) * std::get<B>(size), // Output
                            std::get<M>(size) * std::get<M>(size) * std::get<B>(size),
                            DummyT()));
    }

    template <typename DataT>
    void DlrmResource<DataT>::resizeFwdStorage(ElementCountFwd const& newElementCounts)
    {
        conditionalReallocDeviceHostPair(mDeviceInput,
                                         mHostInput,
                                         std::get<Input>(mMaxFwdCapacity),
                                         std::get<Input>(newElementCounts));
        conditionalReallocDeviceHostPair(mDeviceOutput,
                                         mHostOutput,
                                         std::get<Output>(mMaxFwdCapacity),
                                         std::get<Output>(newElementCounts));
        conditionalReallocDeviceHostPair(mDeviceAccFwd,
                                         mHostAccFwd,
                                         std::get<Acc>(mMaxFwdCapacity),
                                         std::get<Acc>(newElementCounts));

        Base::reallocHost(mHostOutputRef, std::get<Output>(newElementCounts));

        mCurrentElementCountFwd = newElementCounts;
    }

    template <typename DataT>
    void DlrmResource<DataT>::resizeBwdStorage(ProblemSize const& size)
    {
        resizeBwdStorage(
            std::make_tuple(std::get<M>(size) * std::get<K>(size) * std::get<B>(size), // Input
                            calcTrilSize(size) * std::get<B>(size), // UpstreamGrad
                            std::get<M>(size) * std::get<M>(size) * std::get<B>(size), // Acc
                            std::get<M>(size) * std::get<K>(size) * std::get<B>(size), // Grad
                            std::get<K>(size) * std::get<B>(size)));
    }

    template <typename DataT>
    void DlrmResource<DataT>::resizeBwdStorage(ElementCountBwd const& newElementCounts)
    {
        conditionalReallocDeviceHostPair(mDeviceInput,
                                         mHostInput,
                                         std::get<Input>(mMaxBwdCapacity),
                                         std::get<Input>(newElementCounts));
        conditionalReallocDeviceHostPair(mDeviceUpstreamGrad,
                                         mHostUpstreamGrad,
                                         std::get<UpstreamGrad>(mMaxBwdCapacity),
                                         std::get<UpstreamGrad>(newElementCounts));
        conditionalReallocDeviceHostPair(mDeviceGrad,
                                         mHostGrad,
                                         std::get<Grad>(mMaxBwdCapacity),
                                         std::get<Grad>(newElementCounts));
        conditionalReallocDeviceHostPair(mDeviceBottomMlpGrad,
                                         mHostBottomMlpGrad,
                                         std::get<BottomMlpGrad>(mMaxBwdCapacity),
                                         std::get<BottomMlpGrad>(newElementCounts));
        conditionalReallocDeviceHostPair(mDeviceAccBwd,
                                         mHostAccBwd,
                                         std::get<Acc>(mMaxBwdCapacity),
                                         std::get<Acc>(newElementCounts));

        Base::reallocHost(mHostGradRef, std::get<Grad>(newElementCounts));
        Base::reallocHost(mHostBottomMlpGradRef, std::get<BottomMlpGrad>(newElementCounts));

        mCurrentElementCountBwd = newElementCounts;
    }

    template <typename DataT>
    void DlrmResource<DataT>::reset()
    {
        Base::reallocDeviceHostPair(mDeviceInput, mHostInput, 0);
        Base::reallocDeviceHostPair(mDeviceOutput, mHostOutput, 0);
        Base::reallocDeviceHostPair(mDeviceAccFwd, mHostAccFwd, 0);
        Base::reallocDeviceHostPair(mDeviceUpstreamGrad, mHostUpstreamGrad, 0);
        Base::reallocDeviceHostPair(mDeviceGrad, mHostGrad, 0);
        Base::reallocDeviceHostPair(mDeviceBottomMlpGrad, mHostBottomMlpGrad, 0);
        Base::reallocDeviceHostPair(mDeviceAccBwd, mHostAccBwd, 0);
        mCurrentElementCountFwd = {0, 0, 0, DummyT()};
        mCurrentElementCountBwd = {0, 0, 0, 0, 0};
        mMaxFwdCapacity         = {0, 0, 0, DummyT()};
        mMaxBwdCapacity         = {0, 0, 0, 0, 0};
    }

    template <typename DataT>
    auto DlrmResource<DataT>::hostInput() -> HostPtrT<DataT>&
    {
        return mHostInput;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::hostOutput() -> HostPtrT<DataT>&
    {
        return mHostOutput;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::hostOutputRef() -> HostPtrT<DataT>&
    {
        return mHostOutputRef;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::hostAccFwd() -> HostPtrT<float>&
    {
        return mHostAccFwd;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::hostUpstreamGrad() -> HostPtrT<DataT>&
    {
        return mHostUpstreamGrad;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::hostGrad() -> HostPtrT<DataT>&
    {
        return mHostGrad;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::hostGradRef() -> HostPtrT<DataT>&
    {
        return mHostGradRef;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::hostBottomMlpGrad() -> HostPtrT<DataT>&
    {
        return mHostBottomMlpGrad;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::hostBottomMlpGradRef() -> HostPtrT<DataT>&
    {
        return mHostBottomMlpGradRef;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::hostAccBwd() -> HostPtrT<DataT>&
    {
        return mHostAccBwd;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::deviceInput() -> DevicePtrT<DataT>&
    {
        return mDeviceInput;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::deviceOutput() -> DevicePtrT<DataT>&
    {
        return mDeviceOutput;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::deviceAccFwd() -> DevicePtrT<float>&
    {
        return mDeviceAccFwd;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::deviceUpstreamGrad() -> DevicePtrT<DataT>&
    {
        return mDeviceUpstreamGrad;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::deviceGrad() -> DevicePtrT<DataT>&
    {
        return mDeviceGrad;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::deviceBottomMlpGrad() -> DevicePtrT<DataT>&
    {
        return mDeviceBottomMlpGrad;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::deviceAccBwd() -> DevicePtrT<DataT>&
    {
        return mDeviceAccBwd;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::currentElementCountFwd() const -> ElementCountFwd
    {
        return mCurrentElementCountFwd;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::currentElementCountBwd() const -> ElementCountBwd
    {
        return mCurrentElementCountBwd;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::maxFwdCapacity() const -> ElementCountFwd
    {
        return mMaxFwdCapacity;
    }

    template <typename DataT>
    auto DlrmResource<DataT>::maxBwdCapacity() const -> ElementCountBwd
    {
        return mMaxBwdCapacity;
    }

} // namespace rocwmma

#endif // DLRM_GEMM_RESOURCE_IMPL_HPP
