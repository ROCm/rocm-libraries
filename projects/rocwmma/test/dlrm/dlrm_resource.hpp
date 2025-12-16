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

#ifndef DLRM_RESOURCE_HPP
#define DLRM_RESOURCE_HPP

#include <memory>
#include <tuple>

#include "hip_resource.hpp"
#include "singleton.hpp"

namespace rocwmma
{

    // DlrmResource class is intended to manage a shared pool of resources for
    // testing DLRM kernels on the GPU.
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
    struct DlrmResource : public HipResource, public LazySingleton<DlrmResource<DataT>>
    {
        // For static initialization
        friend std::unique_ptr<DlrmResource<DataT>> std::make_unique<DlrmResource<DataT>>();

        using Base = HipResource;

        template <typename T>
        using DevicePtrT = Base::template DevicePtrT<T>;

        template <typename T>
        using HostPtrT = Base::template HostPtrT<T>;

        // M, K, BatchSize
        using ProblemSize = std::tuple<int64_t, int64_t, int64_t>;

        // Forward pass data sizes
        // Input, Output, Acc
        // Differentiate from ProblemSize
        using DummyT          = DataT;
        using ElementCountFwd = std::tuple<int64_t, int64_t, int64_t, DummyT>;

        // Backward pass data sizes
        // Input, UpstreamGrad, Acc, Grad, BottomMlpGrad
        using ElementCountBwd = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>;

        enum : uint32_t
        {
            // Forward pass data size indices
            Input  = 0,
            Output = 1,
            Acc    = 2,

            // Backward pass data size indices
            UpstreamGrad  = 1,
            Grad          = 3,
            BottomMlpGrad = 4,

            // Problem size indices
            M = 0,
            K = 1,
            B = 2
        };

    protected: // No public instantiation except make_unique.
               // No copy
        DlrmResource();
        DlrmResource(DlrmResource const&)            = delete;
        DlrmResource& operator=(DlrmResource const&) = delete;

        // Helpers
        template <typename T>
        static inline void    conditionalReallocDeviceHostPair(DevicePtrT<T>& devicePtr,
                                                               HostPtrT<T>&   hostPtr,
                                                               int64_t&       currentMax,
                                                               int64_t        newSize);
        static inline int64_t calcTrilSize(ProblemSize const& size);

    public:
        DlrmResource(DlrmResource&&);
        ~DlrmResource() = default;

        void copyHostToDeviceFwdAll();
        void copyHostToDeviceBwdAll();
        void copyDeviceToHostFwdInput();
        void copyDeviceToHostFwdOutput();
        void copyDeviceToHostBwdInput();
        void copyDeviceToHostBwdOutput();
        void resizeFwdStorage(ProblemSize const& size);
        void resizeFwdStorage(ElementCountFwd const& size);

        void resizeBwdStorage(ProblemSize const& size);
        void resizeBwdStorage(ElementCountBwd const& size);

        // Forward pass data
        HostPtrT<DataT>&     hostInput();
        HostPtrT<DataT>&     hostOutput();
        HostPtrT<DataT>&     hostOutputRef();
        HostPtrT<float32_t>& hostAccFwd();

        DevicePtrT<DataT>&     deviceInput();
        DevicePtrT<DataT>&     deviceOutput();
        DevicePtrT<float32_t>& deviceAccFwd();

        // Backward pass data
        HostPtrT<DataT>& hostUpstreamGrad();
        HostPtrT<DataT>& hostGrad();
        HostPtrT<DataT>& hostGradRef();
        HostPtrT<DataT>& hostBottomMlpGrad();
        HostPtrT<DataT>& hostBottomMlpGradRef();
        HostPtrT<DataT>& hostAccBwd();

        DevicePtrT<DataT>& deviceUpstreamGrad();
        DevicePtrT<DataT>& deviceGrad();
        DevicePtrT<DataT>& deviceBottomMlpGrad();
        DevicePtrT<DataT>& deviceAccBwd();

        // Data sizes
        ElementCountFwd currentElementCountFwd() const;
        ElementCountBwd currentElementCountBwd() const;
        ElementCountFwd maxFwdCapacity() const;
        ElementCountBwd maxBwdCapacity() const;

        // Reset sizes
        void reset() final;

    protected:
        // Forward pass data
        DevicePtrT<DataT>     mDeviceInput, mDeviceOutput;
        DevicePtrT<float32_t> mDeviceAccFwd;
        HostPtrT<DataT>       mHostInput, mHostOutput, mHostOutputRef;
        HostPtrT<float32_t>   mHostAccFwd;

        // Backward pass data
        DevicePtrT<DataT> mDeviceUpstreamGrad, mDeviceGrad, mDeviceBottomMlpGrad, mDeviceAccBwd;
        HostPtrT<DataT>   mHostUpstreamGrad, mHostGrad, mHostGradRef, mHostBottomMlpGrad,
            mHostBottomMlpGradRef, mHostAccBwd;

        ElementCountFwd mCurrentElementCountFwd;
        ElementCountBwd mCurrentElementCountBwd;

        ElementCountFwd mMaxFwdCapacity;
        ElementCountBwd mMaxBwdCapacity;
    };

} // namespace rocwmma

#include "dlrm_resource_impl.hpp"

#endif // DLRM_RESOURCE_HPP
