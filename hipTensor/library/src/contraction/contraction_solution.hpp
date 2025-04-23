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

#ifndef HIPTENSOR_CONTRACTION_SOLUTION_HPP
#define HIPTENSOR_CONTRACTION_SOLUTION_HPP

#include <functional>
#include <memory>
#include <tuple>
#include <vector>

// CK includes
#include <contraction_bilinear.hpp>
#include <contraction_scale.hpp>
#include <device_contraction_multiple_d.hpp>
#include <element_wise_operation.hpp>

#include "device/device_element_wise_operation_complex.hpp"

#include "contraction_meta_traits.hpp"
#include "contraction_solution_params.hpp"
#include "performance.hpp"

namespace hiptensor
{
    class ContractionSolution
    {
    public:
        // Due to unique_ptr ownership of members,
        // ContractionSolutions should also be considered unique.
        // This means disabling default and copy ctor
        ContractionSolution()                                      = delete;
        ContractionSolution(ContractionSolution const&)            = delete;
        virtual ~ContractionSolution()                             = default;
        ContractionSolution& operator=(ContractionSolution const&) = delete;

        // This class is intended to receive DeviceOp kernel pointers from
        // the CK generator and take ownership.
        ContractionSolution(std::unique_ptr<ck::tensor_operation::device::BaseOperator>&& deviceOp,
                            std::unique_ptr<ContractionSolutionParams>&&                  params);
        ContractionSolution(ContractionSolution&& other);
        ContractionSolution& operator=(ContractionSolution&& other);

        // Must specialize incoming arg handling
        virtual bool initArgs(void const*              alpha,
                              void const*              A,
                              void const*              B,
                              void const*              beta,
                              void const*              D,
                              void*                    E,
                              std::vector<std::size_t> a_ms_ns_lengths,
                              std::vector<std::size_t> a_ms_ks_strides,
                              std::vector<int32_t>     a_ms_ks_modes,
                              std::vector<std::size_t> b_ns_ks_lengths,
                              std::vector<std::size_t> b_ns_ks_strides,
                              std::vector<int32_t>     b_ns_ks_modes,
                              std::vector<std::size_t> ds_ms_ns_lengths,
                              std::vector<std::size_t> ds_ms_ns_strides,
                              std::vector<int32_t>     ds_ms_ns_modes,
                              std::vector<std::size_t> e_ms_ns_lengths,
                              std::vector<std::size_t> e_ms_ns_strides,
                              std::vector<int32_t>     e_ms_ns_modes,
                              void*                    workspacePtr)
            = 0;

        std::tuple<hiptensorStatus_t, float> operator()(void const*              alpha,
                                                        void const*              A,
                                                        void const*              B,
                                                        void const*              beta,
                                                        void const*              D,
                                                        void*                    E,
                                                        std::vector<std::size_t> a_ms_ns_lengths,
                                                        std::vector<std::size_t> a_ms_ks_strides,
                                                        std::vector<int32_t>     a_ms_ks_modes,
                                                        std::vector<std::size_t> b_ns_ks_lengths,
                                                        std::vector<std::size_t> b_ns_ks_strides,
                                                        std::vector<int32_t>     b_ns_ks_modes,
                                                        std::vector<std::size_t> ds_ms_ns_lengths,
                                                        std::vector<std::size_t> ds_ms_ns_strides,
                                                        std::vector<int32_t>     ds_ms_ns_modes,
                                                        std::vector<std::size_t> e_ms_ns_lengths,
                                                        std::vector<std::size_t> e_ms_ns_strides,
                                                        std::vector<int32_t>     e_ms_ns_modes,
                                                        void*                    workspacePtr,
                                                        unsigned long            workspaceSize,
                                                        StreamConfig const&      streamConfig
                                                        = StreamConfig{});

        /// Accessors

        // Problem can be solved with this kernel
        bool isValid() const;

        // Run-time solution parameters
        std::unique_ptr<ContractionSolutionParams> const& params() const;

        // Unique ID for the kernel
        size_t uid() const;

        // Problem dimensions
        std::tuple<ck::index_t, ck::index_t, ck::index_t> problemDims() const;

        // Byte count
        ck::index_t problemBytes() const;

        // Kernel's name encoding
        std::string kernelName() const;

        // Kernel's required workspace size
        size_t workspaceSize() const;

        // Reset all arguments
        void resetArgs();

    protected:
        void resetInvokerArgs();

        // Derived runtime arguments
        ck::index_t mM, mN, mK;
        ck::index_t mBytes;
        bool        mValid;

        // Kernel Params
        std::unique_ptr<ContractionSolutionParams>                  mParams;
        std::unique_ptr<ck::tensor_operation::device::BaseOperator> mDeviceOp;
        std::unique_ptr<ck::tensor_operation::device::BaseArgument> mInvokerArgPtr;
        std::unique_ptr<ck::tensor_operation::device::BaseInvoker>  mInvokerPtr;
    };

    template <ck::index_t NumDimM,
              ck::index_t NumDimN,
              ck::index_t NumDimK,
              typename ADataType,
              typename BDataType,
              typename DsDataType,
              typename EDataType,
              typename AElementwiseOperation,
              typename BElementwiseOperation,
              typename CDEElementwiseOperation,
              typename ComputeDataType>
    std::vector<std::unique_ptr<hiptensor::ContractionSolution>> enumerateContractionSolutions();

} // namespace hiptensor

#include "contraction_solution_impl.hpp"

#endif // HIPTENSOR_CONTRACTION_SOLUTION_HPP
