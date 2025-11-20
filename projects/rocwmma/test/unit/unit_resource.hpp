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

#ifndef ROCWMMA_UNIT_RESOURCE_HPP
#define ROCWMMA_UNIT_RESOURCE_HPP

#include <memory>
#include <tuple>

#include <hip/hip_runtime_api.h>

#include "hip_resource.hpp"
#include "singleton.hpp"

namespace rocwmma
{

    // UnitResource class is intended to manage a shared pool of resources for
    // testing various kernels on the GPU.
    //
    // It minimizes the memory handling overhead for launching thousands of GPU
    // kernels by allowing re-use of existing memory allocations. Memory is only
    // re-allocated as necessary to satisfy minimum size requirements.
    //
    // The interface indicates memory ownership by this class and shall only be
    // used to access for read/write purposes.
    //
    // Currently uses HIP as the backend for device allocation.

    template <typename DataT>
    struct UnitResource : public HipResource, public LazySingleton<UnitResource<DataT>>
    {
        // For static initialization
        friend std::unique_ptr<UnitResource<DataT>> std::make_unique<UnitResource<DataT>>();

        using Base = HipResource;

        using DevicePtrT = Base::DevicePtrT<DataT>;

        using HostPtrT = Base::HostPtrT<DataT>;

        // M, N
        using ProblemSize = std::tuple<int64_t, int64_t>;

        enum : uint32_t
        {
            // Problem size indices
            M = 0,
            N = 1
        };

    protected:
        // Singleton instantiation
        UnitResource();
        UnitResource(UnitResource const&)            = delete;
        UnitResource& operator=(UnitResource const&) = delete;

    public:
        ~UnitResource() = default;
        void resizeStorage(ProblemSize const& size);

        HostPtrT&   hostIn();
        HostPtrT&   hostOut();
        DevicePtrT& deviceIn();
        DevicePtrT& deviceOut();

        ProblemSize problemSize() const;
        int64_t     maxCapacity() const;

        // Reset sizes
        void reset() final;

    protected:
        DevicePtrT  mDeviceIn, mDeviceOut;
        HostPtrT    mHostIn, mHostOut;
        ProblemSize mCurrentProblemSize;
        int64_t     mMaxCapacity;
    };

} // namespace rocwmma

#include "unit_resource_impl.hpp"

#endif // ROCWMMA_UNIT_RESOURCE_HPP
