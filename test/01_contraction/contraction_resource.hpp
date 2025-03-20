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

#ifndef HIPTENSOR_CONTRACTION_RESOURCE_HPP
#define HIPTENSOR_CONTRACTION_RESOURCE_HPP

#include <memory>
#include <tuple>

#include "hip_resource.hpp"
#include "singleton.hpp"

// ContractionResource class is intended to manage a shared pool of resources for
// testing hiptensor contraction kernels on the GPU.
//
// It minimizes the memory handling overhead for launching thousands of GPU
// kernels by allowing re-use of existing memory allocations. Memory is only
// re-allocated as necessary to satisfy minimum size requirements.
//
// The interface indicates memory ownership by this class and shall only be
// used to access for read/write purposes.
//
// Currently uses HIP as the backend for device allocation.

namespace hiptensor
{

    struct ContractionResource : public HipResource, public LazySingleton<ContractionResource>
    {
        // For static initialization
        friend std::unique_ptr<ContractionResource> std::make_unique<ContractionResource>();

        using Base = HipResource;

    public:
        using DevicePtrT = Base::DevicePtrT;
        using HostPtrT   = Base::HostPtrT;

        // M, N, U, V, H, K
        using ProblemDims = std::vector<std::vector<std::size_t>>;
        ;

        // MatrixA, MatrixB, MatrixC, MatrixD (# of elements)
        using MatrixElements = std::tuple<int64_t, int64_t, int64_t, int64_t>;

        // Bytes per element for matrices A/B/C/D
        using ElementBytes = std::tuple<int32_t, int32_t, int32_t, int32_t>;

        enum : uint32_t
        {
            // Matrix size indices
            MatrixA = 0,
            MatrixB = 1,
            MatrixC = 2,
            MatrixD = 3,
        };

    private: // No public instantiation except make_unique.
             // No copy
        ContractionResource();
        ContractionResource(const ContractionResource&)            = delete;
        ContractionResource& operator=(const ContractionResource&) = delete;

    public:
        ContractionResource(ContractionResource&&);
        ~ContractionResource() = default;

        void copyHostToDeviceAll(ElementBytes const& bytesPerElement);
        void copyDeviceToHostAll(ElementBytes const& bytesPerElement);
        void resizeStorage(ProblemDims const& size, ElementBytes bytesPerElement);
        void resizeStorage(MatrixElements const& size, ElementBytes bytesPerElement);

        HostPtrT& hostA();
        HostPtrT& hostB();
        HostPtrT& hostC();
        HostPtrT& hostD();

        DevicePtrT& deviceA();
        DevicePtrT& deviceB();
        DevicePtrT& deviceC();
        DevicePtrT& deviceD();

        void reset() final;

    protected:
        DevicePtrT     mDeviceA, mDeviceB;
        DevicePtrT     mDeviceC, mDeviceD;
        HostPtrT       mHostA, mHostB;
        HostPtrT       mHostC, mHostD;
        ElementBytes   mCurrentElementBytes;
        ElementBytes   mCurrentAllocBytes;
        MatrixElements mCurrentMatrixElements;
        MatrixElements mCurrentAllocElements;
    };

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_RESOURCE_HPP
