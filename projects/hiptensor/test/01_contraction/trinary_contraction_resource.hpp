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

#pragma once

#include <memory>
#include <tuple>

#include "hip_resource.hpp"
#include "singleton.hpp"

namespace hiptensor
{

    struct TrinaryContractionResource : public HipResource,
                                        public LazySingleton<TrinaryContractionResource>
    {
        friend std::unique_ptr<TrinaryContractionResource>
            std::make_unique<TrinaryContractionResource>();

        using Base = HipResource;

    public:
        using DevicePtrT = Base::DevicePtrT;
        using HostPtrT   = Base::HostPtrT;

        using ProblemDims = std::vector<std::vector<std::size_t>>;

        // A, B, C, D, E
        using MatrixElements = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>;
        using ElementBytes   = std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t>;

        enum : uint32_t
        {
            TensorA = 0,
            TensorB = 1,
            TensorC = 2,
            TensorD = 3,
            TensorE = 4,
        };

    private:
        TrinaryContractionResource();
        TrinaryContractionResource(const TrinaryContractionResource&)            = delete;
        TrinaryContractionResource& operator=(const TrinaryContractionResource&) = delete;

    public:
        TrinaryContractionResource(TrinaryContractionResource&&);
        ~TrinaryContractionResource() = default;

        void copyHostToDeviceAll(ElementBytes const& bytesPerElement);
        void copyDeviceToHostAll(ElementBytes const& bytesPerElement);
        void resizeStorage(ProblemDims const& size, ElementBytes bytesPerElement);
        void resizeStorage(MatrixElements const& size, ElementBytes bytesPerElement);

        HostPtrT& hostA();
        HostPtrT& hostB();
        HostPtrT& hostC();
        HostPtrT& hostD();
        HostPtrT& hostE();

        DevicePtrT& deviceA();
        DevicePtrT& deviceB();
        DevicePtrT& deviceC();
        DevicePtrT& deviceD();
        DevicePtrT& deviceE();

        void reset() final;

    protected:
        DevicePtrT     mDeviceA, mDeviceB, mDeviceC, mDeviceD, mDeviceE;
        HostPtrT       mHostA, mHostB, mHostC, mHostD, mHostE;
        ElementBytes   mCurrentElementBytes;
        ElementBytes   mCurrentAllocBytes;
        MatrixElements mCurrentMatrixElements;
        MatrixElements mCurrentAllocElements;
    };

} // namespace hiptensor
