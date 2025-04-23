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

#ifndef HIPTENSOR_HIP_RESOURCE_IMPL_HPP
#define HIPTENSOR_HIP_RESOURCE_IMPL_HPP

#include <hip/hip_runtime_api.h>

#include "hip_resource.hpp"
#include <hiptensor/internal/hiptensor_utility.hpp>

namespace hiptensor
{

    auto HipResource::allocDevice(int64_t numBytes) -> DevicePtrT
    {
        char* data;
        CHECK_HIP_ERROR(hipMalloc(&data, numBytes));
        return DevicePtrT(data, DeviceDeleter());
    }

    void HipResource::reallocDevice(DevicePtrT& devicePtr, int64_t numBytes)
    {
        // Free existing ptr first before alloc in case of big sizes.
        devicePtr.reset(nullptr);
        devicePtr = std::move(allocDevice(numBytes));
    }

    auto HipResource::allocHost(int64_t numBytes) -> HostPtrT
    {
        void* data;
        data = operator new(numBytes);
        return HostPtrT(data, HostDeleter());
    }

    void HipResource::reallocHost(HostPtrT& hostPtr, int64_t numBytes)
    {
        // Free existing ptr first before alloc in case of big sizes.
        hostPtr.reset(nullptr);
        hostPtr = std::move(allocHost(numBytes));
    }

    void HipResource::reallocDeviceHostPair(DevicePtrT& devicePtr,
                                            HostPtrT&   hostPtr,
                                            int64_t     numBytes)
    {
        reallocDevice(devicePtr, numBytes);
        reallocHost(hostPtr, numBytes);
    }

    void HipResource::copyData(HostPtrT& dst, DevicePtrT const& src, int64_t numBytes)
    {
        CHECK_HIP_ERROR(hipMemcpy(dst.get(), src.get(), numBytes, hipMemcpyDeviceToHost));
    }

    void HipResource::copyData(DevicePtrT& dst, HostPtrT const& src, int64_t numBytes)
    {
        CHECK_HIP_ERROR(hipMemcpy(dst.get(), src.get(), numBytes, hipMemcpyHostToDevice));
    }

    void HipResource::copyData(HostPtrT& dst, HostPtrT const& src, int64_t numBytes)
    {
        CHECK_HIP_ERROR(hipMemcpy(dst.get(), src.get(), numBytes, hipMemcpyHostToHost));
    }

    void HipResource::copyData(DevicePtrT& dst, DevicePtrT const& src, int64_t numBytes)
    {
        CHECK_HIP_ERROR(hipMemcpy(dst.get(), src.get(), numBytes, hipMemcpyDeviceToDevice));
    }

} // namespace hiptensor

#endif //HIPTENSOR_HIP_RESOURCE_IMPL_HPP
