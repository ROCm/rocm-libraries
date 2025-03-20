/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "permutation_solution.hpp"

namespace hiptensor
{

    PermutationSolution::PermutationSolution(
        std::unique_ptr<ck::tensor_operation::device::BaseOperator>&& deviceOp)
        : mDim(0)
        , mSize(0)
        , mValid(false)
        , mDeviceOp(std::move(deviceOp))
    {
    }

    PermutationSolution::PermutationSolution(PermutationSolution&& other)
        : mDim(other.mDim)
        , mSize(other.mSize)
        , mValid(other.mValid)
        , mDeviceOp(std::move(other.mDeviceOp))
        , mInvokerArgPtr(std::move(other.mInvokerArgPtr))
        , mInvokerPtr(std::move(other.mInvokerPtr))
    {
    }

    PermutationSolution& PermutationSolution::operator=(PermutationSolution&& other)
    {
        if(this != &other)
        {
            mDim = other.mDim;

            mSize  = other.mSize;
            mValid = other.mValid;

            mDeviceOp      = std::move(other.mDeviceOp);
            mInvokerArgPtr = std::move(other.mInvokerArgPtr);
            mInvokerPtr    = std::move(other.mInvokerPtr);
        }
        return *this;
    }

    float PermutationSolution::operator()(StreamConfig const& streamConfig /*= StreamConfig{}*/)
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

    bool PermutationSolution::isValid() const
    {
        return mValid;
    }

    size_t PermutationSolution::uid() const
    {
        // Convert CK uid string into binary.
        std::istringstream converter(mDeviceOp->GetTypeIdHashCode());
        size_t             value;
        converter >> std::hex >> value;
        return value;
    }

    uint32_t PermutationSolution::threadDim() const
    {
        return mThreadDim;
    }

    ck::index_t PermutationSolution::problemDim() const
    {
        return mDim;
    }

    ck::index_t PermutationSolution::problemSize() const
    {
        return mSize;
    }

    std::string PermutationSolution::kernelName() const
    {
        return mDeviceOp->GetTypeString();
    }

    size_t PermutationSolution::workspaceSize() const
    {
        if(mValid)
        {
            return mDeviceOp->GetWorkSpaceSize(mInvokerArgPtr.get());
        }
        else
        {
            return 0;
        }
    }

    void PermutationSolution::resetArgs()
    {
        mDim  = 0;
        mSize = 0;

        mInvokerArgPtr.reset(nullptr);
        mInvokerPtr.reset(nullptr);

        mValid = false;
    }

} // namespace hiptensor
