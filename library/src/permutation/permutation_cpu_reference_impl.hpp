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
    // MakeTupleOfPointers
    template <typename Tuple>
    struct MakeTupleOfPointers;

    template <typename... Types>
    struct MakeTupleOfPointers<ck::Tuple<Types...>>
    {
        using type = ck::Tuple<std::add_pointer_t<Types>...>;
    };

    template <typename Tuple>
    using MakeTupleOfPointers_t = typename MakeTupleOfPointers<Tuple>::type;

    template <typename Tuple>
    struct MakeTupleOfConstPointers;

    template <typename... Types>
    struct MakeTupleOfConstPointers<ck::Tuple<Types...>>
    {
        using type = ck::Tuple<std::add_pointer_t<std::add_const_t<Types>>...>;
    };

    template <typename Tuple>
    using MakeTupleOfConstPointers_t = typename MakeTupleOfConstPointers<Tuple>::type;

    template <typename InDataTypeTuple,
              typename OutDataTypeTuple,
              typename ElementOp,
              ck::index_t NumDim>
    struct ReferencePermutation
        : public ck::tensor_operation::device::
              DeviceElementwise<InDataTypeTuple, OutDataTypeTuple, ElementOp, NumDim>
    {
        using Base = ck::tensor_operation::device::
            DeviceElementwise<InDataTypeTuple, OutDataTypeTuple, ElementOp, NumDim>;
        using BaseArgument = ck::tensor_operation::device::BaseArgument;
        using BaseInvoker  = ck::tensor_operation::device::BaseInvoker;
        using index_t      = ck::index_t;

        static constexpr int NumInput  = InDataTypeTuple::Size();
        static constexpr int NumOutput = OutDataTypeTuple::Size();

        // Argument
        struct Argument : public BaseArgument
        {
            Argument(const std::array<index_t, NumDim>                        lengths,
                     const std::array<std::array<index_t, NumDim>, NumInput>  inStridesArray,
                     const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                     const std::array<const void*, NumInput>                  in_dev_buffers,
                     const std::array<void*, NumOutput>                       out_dev_buffers,
                     ElementOp                                                elementwise_op)
                : BaseArgument()
                , mLengths(lengths)
                , mInStrides(inStridesArray)
                , mOutStrides(outStridesArray)
                , mElementOp(elementwise_op)
            {
                static_for<NumInput>([this, &in_dev_buffers](auto index) {
                    mInput.At(ck::Number<index>{})
                        = static_cast<ck::tuple_element_t<index, decltype(mInput)>>(
                            in_dev_buffers[index]);
                });
                static_for<NumOutput>([this, &out_dev_buffers](auto index) {
                    mOutput.At(ck::Number<index>{})
                        = static_cast<ck::tuple_element_t<index, decltype(mOutput)>>(
                            out_dev_buffers[index]);
                });
            }

            Argument(Argument const&)            = default;
            Argument& operator=(Argument const&) = default;
            ~Argument()                          = default;

            MakeTupleOfConstPointers_t<InDataTypeTuple> mInput;
            MakeTupleOfPointers_t<OutDataTypeTuple>     mOutput;

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

                    auto outOffset = std::inner_product(
                        indices.rbegin(), indices.rend(), std::rbegin(arg.mOutStrides[0]), 0);
                    auto inOffset = std::inner_product(
                        indices.rbegin(), indices.rend(), std::rbegin(arg.mInStrides[0]), 0);
                    nextIndex();

                    // Perform sequence of unary, scale operations on input
                    if constexpr(NumInput == 1)
                    {
                        arg.mElementOp(arg.mOutput.At(ck::Number<0>{})[outOffset],
                                       arg.mInput.At(ck::Number<0>{})[inOffset]);
                    }
                    else if constexpr(NumInput == 2)
                    {
                        arg.mElementOp(arg.mOutput.At(ck::Number<0>{})[outOffset],
                                       arg.mInput.At(ck::Number<0>{})[inOffset],
                                       arg.mInput.At(ck::Number<1>{})[inOffset]);
                    }
                    else if constexpr(NumInput == 3)
                    {
                        arg.mElementOp(arg.mOutput.At(ck::Number<0>{})[outOffset],
                                       arg.mInput.At(ck::Number<0>{})[inOffset],
                                       arg.mInput.At(ck::Number<1>{})[inOffset],
                                       arg.mInput.At(ck::Number<2>{})[inOffset]);
                    }
                    else
                    {
                        static_assert(false, "Invalid lengths of InDataTypeTuple");
                    }
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
                         const std::array<const void*, NumInput>                  in_dev_buffers,
                         const std::array<void*, NumOutput>                       out_dev_buffers,
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
            const std::array<const void*, NumInput>                  in_dev_buffers,
            const std::array<void*, NumOutput>                       out_dev_buffers,
            ElementOp                                                elementwise_op) override
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

    template <typename InDataTypeTuple,
              typename OutDataTypeTuple,
              typename ElementwiseOperation,
              ck::index_t NumDim>
    auto enumerateReferenceSolutions()
    {
        using ReferenceOp
            = ReferencePermutation<InDataTypeTuple, OutDataTypeTuple, ElementwiseOperation, NumDim>;
        using BaseOp = typename ReferenceOp::Base;

        auto solution
            = std::make_unique<PermutationSolutionImpl<BaseOp>>(std::make_unique<ReferenceOp>());

        constexpr hiptensor::PermutationOpId_t opType
            = std::is_same_v<ElementwiseOperation, CkPermutationPassThroughCombinedOp>
                  ? hiptensor::PermutationOpId_t::PASS_THROUGH
                  : hiptensor::PermutationOpId_t::SCALE;
        auto params = ck::tensor_operation::device::instance::DeviceElementwiseParams::
            Gen<InDataTypeTuple, OutDataTypeTuple, opType, NumDim>();
        auto hashCode = hiptensor::Hash{}(params);
        auto result   = std::unordered_map<Uid, std::unique_ptr<PermutationSolution>>();
        result.insert({hashCode, std::move(solution)});

        return result;
    }

} // namespace hiptensor

#endif // HIPTENSOR_PERMUTATION_CPU_REFERENCE_IMPL_HPP
