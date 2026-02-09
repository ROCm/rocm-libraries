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

#ifndef ROCWMMA_UNIT_RESOURCE_IMPL_HPP
#define ROCWMMA_UNIT_RESOURCE_IMPL_HPP

#include <cstring> // for std::memcpy

#include "common.hpp"
#include "unit_resource.hpp"

namespace rocwmma
{

    template <typename DataT>
    UnitResource<DataT>::UnitResource()
        : mDeviceIn(nullptr, [](DataT*) {})
        , mDeviceOut(nullptr, [](DataT*) {})
        , mHostIn(nullptr)
        , mHostOut(nullptr)
        , mCurrentProblemSize({0, 0})
        , mMaxCapacity(0)
    {
    }

    template <typename DataT>
    void UnitResource<DataT>::resizeStorage(ProblemSize const& size)
    {
        auto newSize = std::get<M>(size) * std::get<N>(size); // M * N = C, D

        if(mMaxCapacity < newSize)
        {
            mMaxCapacity = newSize;
            mHostIn      = std::move(Base::template allocHost<DataT>(mMaxCapacity));
            mHostOut     = std::move(Base::template allocHost<DataT>(mMaxCapacity));
            mDeviceIn    = std::move(Base::template allocDevice<DataT>(mMaxCapacity));
            mDeviceOut   = std::move(Base::template allocDevice<DataT>(mMaxCapacity));
        }
        mCurrentProblemSize = size;
    }

    template <typename DataT>
    auto UnitResource<DataT>::hostIn() -> HostPtrT&
    {
        return mHostIn;
    }

    template <typename DataT>
    auto UnitResource<DataT>::hostOut() -> HostPtrT&
    {
        return mHostOut;
    }

    template <typename DataT>
    auto UnitResource<DataT>::deviceIn() -> DevicePtrT&
    {
        return mDeviceIn;
    }

    template <typename DataT>
    auto UnitResource<DataT>::deviceOut() -> DevicePtrT&
    {
        return mDeviceOut;
    }

    template <typename DataT>
    auto UnitResource<DataT>::problemSize() const -> ProblemSize
    {
        return mCurrentProblemSize;
    }

    template <typename DataT>
    int64_t UnitResource<DataT>::maxCapacity() const
    {
        return mMaxCapacity;
    }

    template <typename DataT>
    void UnitResource<DataT>::reset()
    {
    }

} // namespace rocwmma

#endif // ROCWMMA_UNIT_RESOURCE_IMPL_HPP
