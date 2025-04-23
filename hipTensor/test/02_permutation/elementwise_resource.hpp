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

#ifndef HIPTENSOR_PERMUTATION_RESOURCE_HPP
#define HIPTENSOR_PERMUTATION_RESOURCE_HPP

#include <memory>
#include <tuple>

#include "hip_resource.hpp"
#include "singleton.hpp"

// ElementwiseResource class is intended to manage a shared pool of resources for
// testing hiptensor permutation kernels on the GPU.
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

    struct ElementwiseResource : public HipResource, public LazySingleton<ElementwiseResource>
    {
        enum class ElementwiseOp
        {
            PERMUTATION,
            BINARY_OP,
            TRINARY_OP,
        };
        // For static initialization
        friend std::unique_ptr<ElementwiseResource> std::make_unique<ElementwiseResource>();

        using Base = HipResource;

    public:
        using DevicePtrT = Base::DevicePtrT;
        using HostPtrT   = Base::HostPtrT;

        using ProblemDims = std::vector<std::size_t>;

    private: // No public instantiation except make_unique.
             // No copy
        ElementwiseResource();
        ElementwiseResource(const ElementwiseResource&)            = delete;
        ElementwiseResource& operator=(const ElementwiseResource&) = delete;

        void   fillRandToInput(HostPtrT& hostPtr, DevicePtrT& devicePtr);
        void   fillRandToInput1();
        void   fillRandToInput2();
        void   fillRandToInput3();
        size_t getCurrentMatrixMemorySize() const;
        void   reset() final;

    public:
        ElementwiseResource(ElementwiseResource&&);
        virtual ~ElementwiseResource() = default;

        void setupStorage(ProblemDims const& dimSizes, hipDataType dataType, ElementwiseOp opType);
        void copyOutputToHost();
        void copyReferenceToDevice();

        HostPtrT& hostInput1();
        HostPtrT& hostInput2();
        HostPtrT& hostInput3();
        HostPtrT& hostOutput();
        HostPtrT& hostReference();

        DevicePtrT& deviceInput1();
        DevicePtrT& deviceInput2();
        DevicePtrT& deviceInput3();
        DevicePtrT& deviceOutput();
        DevicePtrT& deviceReference();

        size_t getCurrentMatrixElement() const;

    protected:
        DevicePtrT mDeviceInput1;
        DevicePtrT mDeviceInput2;
        DevicePtrT mDeviceInput3;
        DevicePtrT mDeviceOutput;
        DevicePtrT mDeviceReference;
        HostPtrT   mHostInput1;
        HostPtrT   mHostInput2;
        HostPtrT   mHostInput3;
        HostPtrT   mHostOutput;
        HostPtrT   mHostReference;

        ElementwiseOp mOpType;
        size_t        mCurrentMatrixElement; /**< Element count of Input[1,2,3]/Output */
        hipDataType
            mCurrentDataType; /**< Type size of element of Input[1,2,3]/Output, only support HIP_R_16F, HIP_R_32F, HIP_R_64F */
        size_t mCurrentAllocByte; /**< Allocated size of memory */
    };

} // namespace hiptensor

#endif // HIPTENSOR_PERMUTATION_RESOURCE_HPP
