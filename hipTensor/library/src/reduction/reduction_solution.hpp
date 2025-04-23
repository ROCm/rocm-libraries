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

#ifndef HIPTENSOR_REDUCTION_SOLUTION_HPP
#define HIPTENSOR_REDUCTION_SOLUTION_HPP

#include <functional>
#include <memory>
#include <tuple>
#include <vector>

#include "performance.hpp"
#include "reduction_meta_traits.hpp"
#include "reduction_solution_params.hpp"
#include "util.hpp"

namespace hiptensor
{
    class ReductionSolution
    {
    public:
        // Due to unique_ptr ownership of members,
        // ReductionSolutions should also be considered unique.
        // This means disabling default and copy ctor
        ReductionSolution()                                    = delete;
        ReductionSolution(ReductionSolution const&)            = delete;
        virtual ~ReductionSolution()                           = default;
        ReductionSolution& operator=(ReductionSolution const&) = delete;

        // This class is intended to receive DeviceOp kernel pointers from
        // the CK generator and take ownership.
        ReductionSolution(std::unique_ptr<ck::tensor_operation::device::BaseOperator>&& deviceOp,
                          std::unique_ptr<ReductionSolutionParams>&&                    params);
        ReductionSolution(ReductionSolution&& other);
        ReductionSolution& operator=(ReductionSolution&& other);

        // Must specialize incoming arg handling
        virtual bool initArgs(std::vector<std::size_t> const& a_lengths,
                              std::vector<std::size_t> const& a_strides,
                              std::vector<int32_t> const&     a_modes,
                              std::vector<std::size_t> const& c_lengths,
                              std::vector<std::size_t> const& c_strides,
                              std::vector<int32_t> const&     c_modes,
                              double                          alpha,
                              double                          beta,
                              void const*                     A,
                              void*                           C,
                              hiptensorOperator_t             opReduce)
            = 0;

        std::pair<bool, float> operator()(std::vector<std::size_t> const& a_lengths,
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
                                          StreamConfig const& streamConfig = StreamConfig{});

        /// Accessors

        // Problem can be solved with this kernel
        bool isValid() const;

        // Run-time solution parameters
        std::unique_ptr<ReductionSolutionParams> const& params() const;

        // Unique ID for the kernel
        size_t uid() const;

        // Get Number of threads across dimension
        uint32_t threadDim() const;

        // Problem dimension
        ck::index_t problemDim() const;

        // Byte count
        ck::index_t problemBytes() const;

        // Kernel's name encoding
        std::string kernelName() const;

        // Kernel's required workspace size
        size_t workspaceSize() const;

        // Reset all arguments
        void resetArgs();

    protected:
        // Derived runtime arguments
        ck::index_t mDim;
        ck::index_t mBytes;
        bool        mValid;
        uint32_t    mThreadDim;

        // Kernel Params
        std::unique_ptr<ReductionSolutionParams>                    mParams;
        std::unique_ptr<ck::tensor_operation::device::BaseOperator> mDeviceOp;
        std::unique_ptr<ck::tensor_operation::device::BaseArgument> mInvokerArgPtr;
        std::unique_ptr<ck::tensor_operation::device::BaseInvoker>  mInvokerPtr;
    };

    template <typename InDataType,
              typename AccDataType,
              typename OutDataType,
              int                 Rank,
              int                 NumReduceDim,
              hiptensorOperator_t opReduce,
              bool                PropagateNan,
              bool                OutputIndex>
    auto enumerateReductionSolutions();

} // namespace hiptensor

#include "reduction_solution_impl.hpp"

#endif // HIPTENSOR_REDUCTION_SOLUTION_HPP
