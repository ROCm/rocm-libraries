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

#ifndef HIPTENSOR_PERMUTATION_SOLUTION_IMPL_HPP
#define HIPTENSOR_PERMUTATION_SOLUTION_IMPL_HPP

#include <map>
#include <numeric>

#include "hash.hpp"
#include "hiptensor_options.hpp"
#include "permutation_solution.hpp"
#include <hiptensor_unary_element_wise_operation.hpp>

namespace hiptensor
{
    template <typename DeviceOp, typename Enabler = void>
    class PermutationSolutionImpl;

    template <typename DeviceOp>
    class PermutationSolutionImpl<DeviceOp> : public PermutationSolution
    {
    public:
        PermutationSolutionImpl(std::unique_ptr<DeviceOp>&& deviceOp)
            : PermutationSolution(std::move(deviceOp))
        {
        }

        bool initArgs(std::vector<float> const&                    scalarValues,
                      std::vector<std::vector<std::size_t>> const& inLengthsArray,
                      std::vector<std::vector<std::size_t>> const& inStridesArray,
                      std::vector<std::vector<int32_t>> const&     inModesArray,
                      std::vector<std::vector<std::size_t>> const& outLengthsArray,
                      std::vector<std::vector<std::size_t>> const& outStridesArray,
                      std::vector<std::vector<int32_t>> const&     outModesArray,
                      std::vector<hiptensorOperator_t> const&      operators,
                      std::vector<const void*> const&              inBuffers,
                      std::vector<void*> const&                    outBuffers) override
        {
            using Base   = PermutationSolution;
            using Traits = MetaTraits<DeviceOp>;

            // Clear out the previous arguments
            resetArgs();

            // Promote to derived class for necessary functions such as
            // MakeArgumentPointer and MakeInvokerPointer.
            auto* deviceOp = dynamic_cast<DeviceOp*>(Base::mDeviceOp.get());
            if(deviceOp == nullptr)
            {
                return false;
            }

            auto findThreadDim = [](std::string argValues) -> uint32_t {
                if(!argValues.empty())
                {
                    std::string kernelName = argValues.substr(0, argValues.find('<'));
                    if(kernelName == "DeviceElementwiseImpl"
                       || kernelName == "ReferencePermutation")
                    {
                        int beg = argValues.find(',');
                        int end = argValues.find(',', beg + 1);
                        return std::stoi(argValues.substr(beg + 1, end - beg));
                    }
                }
                return 1;
            };

            auto isColMajorStrides = HiptensorOptions::instance()->isColMajorStrides();

            std::array<index_t, Traits::NDim> deviceInputLengths;
            convertVectorToCkArray(inLengthsArray[0], deviceInputLengths);

            std::array<std::array<index_t, Traits::NDim>, Traits::InDataT::Size()>
                deviceInputStrides;
            for(int i = 0; i < deviceInputStrides.size(); i++)
            {
                if(inStridesArray.empty() || inStridesArray[i].empty())
                {
                    convertVectorToCkArray(
                        hiptensor::stridesFromLengths(inLengthsArray[i], isColMajorStrides),
                        deviceInputStrides[i]);
                }
                else
                {
                    convertVectorToCkArray(inStridesArray[i], deviceInputStrides[i]);
                }
            }

            std::array<std::array<index_t, Traits::NDim>, Traits::OutDataT::Size()>
                deviceOutputStrides;
            for(int i = 0; i < deviceOutputStrides.size(); i++)
            {
                auto stides = hiptensor::stridesFromLengths(outLengthsArray[i], isColMajorStrides);
                std::map<int32_t, int> modeToIndex;
                for(int j = 0; j < Traits::NDim; j++)
                {
                    modeToIndex[outModesArray[i][j]] = j;
                }
                for(int j = 0; j < Traits::NDim; j++)
                {
                    deviceOutputStrides[i][j] = stides[modeToIndex[inModesArray[i][j]]];
                }
            }

            std::array<const void*, Traits::InDataT::Size()> deviceInBuffers;
            std::array<void*, Traits::OutDataT::Size()>      deviceOutBuffers;
            convertVectorToCkArray(inBuffers, deviceInBuffers);
            convertVectorToCkArray(outBuffers, deviceOutBuffers);

            // Initialize the argument pointer
            if constexpr(Traits::InstanceType == ElementwiseInstanceType_t::PERMUTATION)
            {
                if constexpr(std::is_same_v<typename Traits::ScaleOp,
                                            ck::tensor_operation::element_wise::PassThrough>)
                {
                    Base::mInvokerArgPtr = std::move(deviceOp->MakeArgumentPointer(
                        deviceInputLengths,
                        deviceInputStrides,
                        deviceOutputStrides,
                        deviceInBuffers,
                        deviceOutBuffers,
                        typename Traits::CombinedOp{
                            ck::tensor_operation::element_wise::PassThrough{},
                            ck::tensor_operation::element_wise::PassThrough{}}));
                }
                else
                {

                    // According to the definition of permutation \f$B_{\Pi^B(i_0,i_1,...,i_n)} = \alpha \Psi(A_{\Pi^A(i_0,i_1,...,i_n)}))\f$
                    // No operations can be applied to B so that the `opB` which is from descriptor B should be ignored.
                    Base::mInvokerArgPtr = std::move(deviceOp->MakeArgumentPointer(
                        deviceInputLengths,
                        deviceInputStrides,
                        deviceOutputStrides,
                        deviceInBuffers,
                        deviceOutBuffers,
                        // ck::tensor_operation::element_wise::PassThrough{}));
                        typename Traits::CombinedOp{typename Traits::AOp{operators[0]},
                                                    typename Traits::ScaleOp{scalarValues[0]}}));
                }
            }
            else if constexpr(Traits::InstanceType
                              == ElementwiseInstanceType_t::ELEMENTWISE_BINARY_OP)
            {
                Base::mInvokerArgPtr = std::move(deviceOp->MakeArgumentPointer(
                    deviceInputLengths,
                    deviceInputStrides,
                    deviceOutputStrides,
                    deviceInBuffers,
                    deviceOutBuffers,
                    typename Traits::CombinedOp{
                        typename Traits::ACOp{operators[0]},
                        typename Traits::AOp{CkHiptensorUnaryOp{operators[1]},
                                             CkScale{scalarValues[0]}},
                        typename Traits::COp{CkHiptensorUnaryOp{operators[2]},
                                             CkScale{scalarValues[1]}},
                    })); // ignore opB since none operation should be applied on output
            }
            else if constexpr(Traits::InstanceType
                              == ElementwiseInstanceType_t::ELEMENTWISE_TRINARY_OP)
            {
                Base::mInvokerArgPtr = std::move(deviceOp->MakeArgumentPointer(
                    deviceInputLengths,
                    deviceInputStrides,
                    deviceOutputStrides,
                    deviceInBuffers,
                    deviceOutBuffers,
                    typename Traits::CombinedOp{
                        typename Traits::ABCOp{operators[0]},
                        typename Traits::ABOp{operators[1]},
                        typename Traits::AOp{CkHiptensorUnaryOp{operators[2]},
                                             CkScale{scalarValues[0]}},
                        typename Traits::AOp{CkHiptensorUnaryOp{operators[3]},
                                             CkScale{scalarValues[1]}},
                        typename Traits::COp{CkHiptensorUnaryOp{operators[4]},
                                             CkScale{scalarValues[2]}},
                    })); // ignore opB since none operation should be applied on output
            }
            else
            {
                static_assert(false, "InstanceType of the solution intance is invalid");
            }

            // Initialize the invoker
            Base::mInvokerPtr = std::move(deviceOp->MakeInvokerPointer());

            // Fill problem metrics
            Base::mDim = Traits::NDim;

            // Size count
            Base::mSize = std::accumulate(
                inLengthsArray[0].cbegin(), inLengthsArray[0].cend(), 1, std::multiplies{});

            // Arg test
            Base::mValid = deviceOp->IsSupportedArgument(Base::mInvokerArgPtr.get());

            Base::mThreadDim = findThreadDim(deviceOp->GetTypeString());

            return mValid;
        }
    };

    template <typename InDataTypeTuple,
              typename OutDataTypeTuple,
              typename ElementwiseOperation,
              ck::index_t NumDim>
    auto enumeratePermutationSolutions()
    {
        using PermutationOp = ck::tensor_operation::device::
            DeviceElementwise<InDataTypeTuple, OutDataTypeTuple, ElementwiseOperation, NumDim>;

        using Factory
            = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<PermutationOp>;

        std::unordered_map<Uid, std::unique_ptr<PermutationSolution>> result;
        for(auto& item : Factory::GetInstances())
        {
            result.insert(
                {item.first,
                 std::make_unique<PermutationSolutionImpl<PermutationOp>>(std::move(item.second))});
        }
        return result;
    }

} // namespace hiptensor

#endif // HIPTENSOR_PERMUTATION_SOLUTION_IMPL_HPP
