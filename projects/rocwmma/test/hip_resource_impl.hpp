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

#ifndef ROCWMMA_HIP_RESOURCE_IMPL_HPP
#define ROCWMMA_HIP_RESOURCE_IMPL_HPP

#include <hip/hip_runtime_api.h>

#include "common.hpp"
#include "hip_resource.hpp"

namespace rocwmma
{

    template <typename DataT>
    auto inline HipResource::allocDevice(int64_t numElements) -> DevicePtrT<DataT>
    {
        DataT* data;
        CHECK_HIP_ERROR(hipMalloc(&data, numElements * sizeof(DataT)));
        return DevicePtrT<DataT>(data, [](DataT* d) { CHECK_HIP_ERROR(hipFree(d)); });
    }

    template <typename DataT>
    inline void HipResource::reallocDevice(DevicePtrT<DataT>& devicePtr, int64_t numElements)
    {
        // Free existing ptr first before alloc in case of big sizes.
        devicePtr.reset(nullptr);
        devicePtr = std::move(allocDevice<DataT>(numElements));
    }

    template <typename DataT>
    auto HipResource::allocHost(int64_t numElements) -> HostPtrT<DataT>
    {
        return HostPtrT<DataT>(new DataT[numElements]);
    }

    template <typename DataT>
    inline void HipResource::reallocHost(HostPtrT<DataT>& hostPtr, int64_t numElements)
    {
        // Free existing ptr first before alloc in case of big sizes.
        hostPtr.reset(nullptr);
        hostPtr = std::move(allocHost<DataT>(numElements));
    }

    template <typename DataT>
    inline void HipResource::reallocDeviceHostPair(DevicePtrT<DataT>& devicePtr,
                                                   HostPtrT<DataT>&   hostPtr,
                                                   int64_t            numElements)
    {
        reallocDevice(devicePtr, numElements);
        reallocHost(hostPtr, numElements);
    }

    template <typename DataT>
    void HipResource::copyData(HostPtrT<DataT>&         dst,
                               DevicePtrT<DataT> const& src,
                               int64_t                  numElements)
    {
        CHECK_HIP_ERROR(
            hipMemcpy(dst.get(), src.get(), numElements * sizeof(DataT), hipMemcpyDeviceToHost));
    }

    template <typename DataT>
    void HipResource::copyData(DevicePtrT<DataT>&     dst,
                               HostPtrT<DataT> const& src,
                               int64_t                numElements)
    {
        CHECK_HIP_ERROR(
            hipMemcpy(dst.get(), src.get(), numElements * sizeof(DataT), hipMemcpyHostToDevice));
    }

    template <typename DataT>
    void
        HipResource::copyData(HostPtrT<DataT>& dst, HostPtrT<DataT> const& src, int64_t numElements)
    {
        CHECK_HIP_ERROR(
            hipMemcpy(dst.get(), src.get(), numElements * sizeof(DataT), hipMemcpyHostToHost));
    }

    template <typename DataT>
    void HipResource::copyData(DevicePtrT<DataT>&       dst,
                               DevicePtrT<DataT> const& src,
                               int64_t                  numElements)
    {
        CHECK_HIP_ERROR(
            hipMemcpy(dst.get(), src.get(), numElements * sizeof(DataT), hipMemcpyDeviceToDevice));
    }

} // namespace rocwmma

#endif //ROCWMMA_HIP_RESOURCE_IMPL_HPP
