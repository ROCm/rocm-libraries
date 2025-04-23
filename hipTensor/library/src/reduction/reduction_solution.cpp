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

#include "reduction_solution.hpp"

namespace hiptensor
{

    ReductionSolution::ReductionSolution(
        std::unique_ptr<ck::tensor_operation::device::BaseOperator>&& deviceOp,
        std::unique_ptr<ReductionSolutionParams>&&                    params)
        : mDim(0)
        , mBytes(0)
        , mValid(false)
        , mDeviceOp(std::move(deviceOp))
        , mParams(std::move(params))
    {
    }

    ReductionSolution::ReductionSolution(ReductionSolution&& other)
        : mDim(other.mDim)
        , mBytes(other.mBytes)
        , mValid(other.mValid)
        , mDeviceOp(std::move(other.mDeviceOp))
        , mParams(std::move(other.mParams))
        , mInvokerArgPtr(std::move(other.mInvokerArgPtr))
        , mInvokerPtr(std::move(other.mInvokerPtr))
    {
    }

    ReductionSolution& ReductionSolution::operator=(ReductionSolution&& other)
    {
        if(this != &other)
        {
            mDim = other.mDim;

            mBytes = other.mBytes;
            mValid = other.mValid;

            mParams        = std::move(other.mParams);
            mDeviceOp      = std::move(other.mDeviceOp);
            mInvokerArgPtr = std::move(other.mInvokerArgPtr);
            mInvokerPtr    = std::move(other.mInvokerPtr);
        }
        return *this;
    }

    std::pair<bool, float> ReductionSolution::operator()(std::vector<std::size_t> const& a_lengths,
                                                         std::vector<std::size_t> const& a_strides,
                                                         std::vector<int32_t> const&     a_modes,
                                                         std::vector<std::size_t> const& c_lengths,
                                                         std::vector<std::size_t> const& c_strides,
                                                         std::vector<int32_t> const&     c_modes,
                                                         double                          alpha,
                                                         double                          beta,
                                                         void const*                     A,
                                                         void*                           C,
                                                         hiptensorOperator_t             opReduce,
                                                         StreamConfig const& streamConfig)
    {
        if(!initArgs(a_lengths,
                     a_strides,
                     a_modes,
                     c_lengths,
                     c_strides,
                     c_modes,
                     alpha,
                     beta,
                     A,
                     C,
                     opReduce))
        {
#if !NDEBUG
            std::cout << kernelName() << " does not support this problem" << std::endl;
#endif // !NDEBUG
            return {false, 0.0f};
        }

        return {true, mInvokerPtr->Run(mInvokerArgPtr.get(), streamConfig)};
    }

    bool ReductionSolution::isValid() const
    {
        return mValid;
    }

    std::unique_ptr<ReductionSolutionParams> const& ReductionSolution::params() const
    {
        return mParams;
    }

    size_t ReductionSolution::uid() const
    {
        // Convert CK uid string into binary.
        std::istringstream converter(mDeviceOp->GetTypeIdHashCode());
        size_t             value;
        converter >> std::hex >> value;
        return value;
    }

    uint32_t ReductionSolution::threadDim() const
    {
        return mThreadDim;
    }

    ck::index_t ReductionSolution::problemDim() const
    {
        return mDim;
    }

    ck::index_t ReductionSolution::problemBytes() const
    {
        return mBytes;
    }

    std::string ReductionSolution::kernelName() const
    {
        return mDeviceOp->GetTypeString();
    }

    size_t ReductionSolution::workspaceSize() const
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

    void ReductionSolution::resetArgs()
    {
        mDim   = 0;
        mBytes = 0;

        mInvokerArgPtr.reset(nullptr);
        mInvokerPtr.reset(nullptr);

        mValid = false;
    }

} // namespace hiptensor
