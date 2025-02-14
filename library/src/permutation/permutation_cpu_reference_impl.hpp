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

#ifndef HIPTENSOR_PERMUTATION_CPU_REFERENCE_IMPL_HPP
#define HIPTENSOR_PERMUTATION_CPU_REFERENCE_IMPL_HPP

// Std includes
#include <array>
#include <list>
#include <numeric>
#include <vector>

// CK includes
#include <combined_element_wise_operation.hpp>
#include <device_elementwise_dynamic_vector_dims_impl.hpp>
#include <host_tensor.hpp>

#include "permutation_meta_traits.hpp"
#include "permutation_solution.hpp"

namespace hiptensor
{
    template <typename InDataTypeTuple,
              typename OutDataTypeTuple,
              typename ElementOp,
              ck::index_t NumDim>
    struct ReferencePermutation
        : public ck::tensor_operation::device::DeviceElementwiseImpl<InDataTypeTuple,
                                                                     OutDataTypeTuple,
                                                                     ElementOp,
                                                                     NumDim,
                                                                     32,
                                                                     16,
                                                                     32,
                                                                     4,
                                                                     4,
                                                                     ck::Sequence<1, 0>,
                                                                     ck::Sequence<1>,
                                                                     ck::Sequence<1>>
    {
        using BaseArgument = ck::tensor_operation::device::BaseArgument;
        using BaseInvoker  = ck::tensor_operation::device::BaseInvoker;
        using index_t      = ck::index_t;

        using mInDataType  = ck::tuple_element_t<0, InDataTypeTuple>;
        using mOutDataType = ck::tuple_element_t<0, OutDataTypeTuple>;

        static constexpr int NumInput  = InDataTypeTuple::Size();
        static constexpr int NumOutput = OutDataTypeTuple::Size();

        // Argument
        struct Argument : public BaseArgument
        {
            Argument(const std::array<index_t, NumDim>                        lengths,
                     const std::array<std::array<index_t, NumDim>, NumInput>  inStridesArray,
                     const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                     const void*                                              in_dev_buffers,
                     void*                                                    out_dev_buffers,
                     ElementOp                                                elementwise_op)
                : BaseArgument()
                , mLengths(lengths)
                , mInStrides(inStridesArray)
                , mOutStrides(outStridesArray)
                , mElementOp(elementwise_op)
            {
                mInput  = (mInDataType*)in_dev_buffers;
                mOutput = (mOutDataType*)out_dev_buffers;
            }

            Argument(Argument const&)            = default;
            Argument& operator=(Argument const&) = default;
            ~Argument()                          = default;

            const mInDataType* mInput;
            mOutDataType*      mOutput;

            std::array<index_t, NumDim>                        mLengths;
            std::array<std::array<index_t, NumDim>, NumInput>  mInStrides;
            std::array<std::array<index_t, NumDim>, NumOutput> mOutStrides;

            ElementOp mElementOp;
        };

        // Invoker
        struct Invoker : public BaseInvoker
        {
            using Argument = ReferencePermutation::Argument;

            float Run(const Argument& arg)
            {
                int  modeSize     = arg.mLengths.size();
                auto elementCount = hiptensor::elementsFromLengths(
                    std::vector<index_t>(std::begin(arg.mLengths), std::end(arg.mLengths)));

                // Find the write offset and index in output for every input element
                auto indices = std::vector<int32_t>(modeSize, 0);
                for(int elementIndex = 0; elementIndex < elementCount; elementIndex++)
                {
                    auto nextIndex = [&indices, &arg]() -> bool {
                        int N = indices.size();
                        for(int i = N - 1; i >= 0; --i)
                        {
                            if(indices[i] < arg.mLengths[i] - 1)
                            {
                                ++indices[i];
                                return true;
                            }
                            else
                            {
                                indices[i] = 0;
                            }
                        }
                        return false;
                    };

                    auto bOffset = std::inner_product(
                        indices.rbegin(), indices.rend(), std::rbegin(arg.mOutStrides[0]), 0);
                    auto aOffset = std::inner_product(
                        indices.rbegin(), indices.rend(), std::rbegin(arg.mInStrides[0]), 0);
                    nextIndex();

                    // Perform sequence of unary, scale operations on input
                    arg.mElementOp(arg.mOutput[bOffset], arg.mInput[aOffset]);
                }
                return 0;
            }

            float Run(const BaseArgument* p_arg,
                      const StreamConfig& /* stream_config */ = StreamConfig{}) override
            {
                return Run(*dynamic_cast<const Argument*>(p_arg));
            }
        };

        static constexpr bool IsValidCompilationParameter()
        {
            // TODO: properly implement this check
            return true;
        }

        bool IsSupportedArgument(const BaseArgument*) override
        {
            return true;
        }

        static auto
            MakeArgument(const std::array<index_t, NumDim>                        lengths,
                         const std::array<std::array<index_t, NumDim>, NumInput>  inStridesArray,
                         const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                         const void*                                              in_dev_buffers,
                         void*                                                    out_dev_buffers,
                         ElementOp                                                elementwise_op)
        {
            return Argument{lengths,
                            inStridesArray,
                            outStridesArray,
                            in_dev_buffers,
                            out_dev_buffers,
                            elementwise_op};
        }

        std::unique_ptr<BaseArgument> MakeArgumentPointer(
            const std::array<index_t, NumDim>                        lengths,
            const std::array<std::array<index_t, NumDim>, NumInput>  inStridesArray,
            const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
            const void*                                              in_dev_buffers,
            void*                                                    out_dev_buffers,
            ElementOp                                                elementwise_op)
        {
            return std::make_unique<Argument>(Argument{lengths,
                                                       inStridesArray,
                                                       outStridesArray,
                                                       in_dev_buffers,
                                                       out_dev_buffers,
                                                       elementwise_op});
        }

        static auto MakeInvoker()
        {
            return Invoker{};
        }

        std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
        {
            return std::make_unique<Invoker>(Invoker{});
        }

        std::string GetTypeString() const override
        {
            auto str = std::stringstream();

            // clang-format off
            str << "ReferencePermutation<";
            str << NumDim << ", ";
            str << 1 << ">";
            // clang-format on

            return str.str();
        }
    };

    // Partial specialize for reference permutation
    template <typename InDataTypeTuple,
              typename OutDataTypeTuple,
              typename Aop,
              typename Bop,
              typename Scale,
              ck::index_t NumDim>
    struct MetaTraits<
        ReferencePermutation<InDataTypeTuple,
                             OutDataTypeTuple,
                             ck::tensor_operation::element_wise::UnaryCombinedOp<Aop, Scale, Bop>,
                             NumDim>>
        : public MetaTraits<ck::tensor_operation::device::DeviceElementwise<
              InDataTypeTuple,
              OutDataTypeTuple,
              ck::tensor_operation::element_wise::UnaryCombinedOp<Aop, Scale, Bop>,
              NumDim>>
    {
    };

    template <typename InDataTypeTuple,
              typename OutDataTypeTuple,
              typename Aop,
              typename Bop,
              typename Scale,
              ck::index_t NumDim>
    auto enumerateReferenceSolutions()
    {
        using ReferenceOp = ReferencePermutation<
            InDataTypeTuple,
            OutDataTypeTuple,
            ck::tensor_operation::element_wise::UnaryCombinedOp<Aop, Scale, Bop>,
            NumDim>;

        auto solution = std::make_unique<PermutationSolutionImpl<ReferenceOp>>(
            std::make_unique<ReferenceOp>());

        auto hashCode = ck::tensor_operation::device::instance::
            DeviceElementwiseParams<InDataTypeTuple, OutDataTypeTuple, Scale, NumDim>::hashCode();
        auto result = std::unordered_map<Uid, std::unique_ptr<PermutationSolution>>();
        result.insert({hashCode, std::move(solution)});

        return result;
    }

} // namespace hiptensor

#endif // HIPTENSOR_PERMUTATION_CPU_REFERENCE_IMPL_HPP
