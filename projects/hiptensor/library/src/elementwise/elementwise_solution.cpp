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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#include "elementwise_solution.hpp"

namespace hiptensor
{

    ElementwiseSolution::ElementwiseSolution(
        std::unique_ptr<ck::tensor_operation::device::BaseOperator>&& deviceOp)
        : mDim(0)
        , mSize(0)
        , mValid(false)
        , mDeviceOp(std::move(deviceOp))
    {
    }

    float ElementwiseSolution::operator()(StreamConfig const& streamConfig /*= StreamConfig{}*/)
    {
        if(!mInvokerArgPtr || !mInvokerPtr)
        {
#if !NDEBUG
            std::cout << mDeviceOp->GetTypeString() << " is not initialized" << std::endl;
#endif // !NDEBUG
            return -1.0f;
        }

        if(!mValid)
        {
#if !NDEBUG
            std::cout << kernelName() << " does not support this problem" << std::endl;
#endif // !NDEBUG
            return -1.0f;
        }

        return mInvokerPtr->Run(mInvokerArgPtr.get(), streamConfig);
    }

    size_t ElementwiseSolution::uid() const
    {
        // Convert CK uid string into binary.
        std::istringstream converter(mDeviceOp->GetTypeIdHashCode());
        size_t             value;
        converter >> std::hex >> value;
        return value;
    }

    ck::index_t ElementwiseSolution::problemSize() const
    {
        return mSize;
    }

    std::string ElementwiseSolution::kernelName() const
    {
        return mDeviceOp->GetTypeString();
    }

    void ElementwiseSolution::resetArgs()
    {
        mDim  = 0;
        mSize = 0;

        mInvokerArgPtr.reset(nullptr);
        mInvokerPtr.reset(nullptr);

        mValid = false;
    }

} // namespace hiptensor
