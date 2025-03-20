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

#ifndef HIPTENSOR_PERMUTATION_SOLUTION_HPP
#define HIPTENSOR_PERMUTATION_SOLUTION_HPP

#include <functional>
#include <memory>
#include <tuple>
#include <vector>

// CK includes
#include <combined_element_wise_operation.hpp>
#include <device_elementwise_dynamic_vector_dims_impl.hpp>

#include "performance.hpp"
#include "permutation_meta_traits.hpp"
#include "util.hpp"

namespace hiptensor
{
    class PermutationSolution
    {
    public:
        // Due to unique_ptr ownership of members,
        // PermutationSolutions should also be considered unique.
        // This means disabling default and copy ctor
        PermutationSolution()                           = delete;
        PermutationSolution(PermutationSolution const&) = delete;
        virtual ~PermutationSolution()                  = default;
        PermutationSolution& operator=(PermutationSolution const&) = delete;

        // This class is intended to receive DeviceOp kernel pointers from
        // the CK generator and take ownership.
        explicit PermutationSolution(
            std::unique_ptr<ck::tensor_operation::device::BaseOperator>&& deviceOp);
        PermutationSolution(PermutationSolution&& other);
        PermutationSolution& operator=(PermutationSolution&& other);

        // Must specialize incoming arg handling
        virtual bool initArgs(std::vector<float> const&                    scalarValues,
                              std::vector<std::vector<std::size_t>> const& inLengthsArray,
                              std::vector<std::vector<std::size_t>> const& inStridesArray,
                              std::vector<std::vector<int32_t>> const&     inModesArray,
                              std::vector<std::vector<std::size_t>> const& outLengthsArray,
                              std::vector<std::vector<std::size_t>> const& outStridesArray,
                              std::vector<std::vector<int32_t>> const&     outModesArray,
                              std::vector<hiptensorOperator_t> const&      operators,
                              std::vector<const void*> const&              inBuffers,
                              std::vector<void*> const&                    outBuffers)
            = 0;

        float operator()(StreamConfig const& streamConfig = StreamConfig{});

        /// Accessors

        // Problem can be solved with this kernel
        bool isValid() const;

        // Unique ID for the kernel
        size_t uid() const;

        // Get Number of threads across dimension
        uint32_t threadDim() const;

        // Problem dimension
        ck::index_t problemDim() const;

        // Problem size
        ck::index_t problemSize() const;

        // Kernel's name encoding
        std::string kernelName() const;

        // Kernel's required workspace size
        size_t workspaceSize() const;

        // Reset all arguments
        void resetArgs();

    protected:
        // Derived runtime arguments
        ck::index_t mDim;
        ck::index_t mSize;
        bool        mValid;
        uint32_t    mThreadDim;

        // Kernel Params
        std::unique_ptr<ck::tensor_operation::device::BaseOperator> mDeviceOp;
        std::unique_ptr<ck::tensor_operation::device::BaseArgument> mInvokerArgPtr;
        std::unique_ptr<ck::tensor_operation::device::BaseInvoker>  mInvokerPtr;
    };

} // namespace hiptensor

#include "permutation_solution_impl.hpp"

#endif // HIPTENSOR_PERMUTATION_SOLUTION_HPP
