// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cmath>
#include <cstdlib>
#include <numeric>
#include <type_traits>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

//
// @brief      Reference implementation for forward transposed convolution.
//
// @paragraph
//             Tensor descriptor in GNCHW/GKCXY/GNKHW dimensional order
//             Supports both GNCHW/NGCHW as well as GNHWC/NHWGC physical layout
//             as long as dimensions in tensor descriptor is in GNCHW order
//
// @tparam     NDimSpatial  Number of spatial dimensions.
// @tparam     InDataType               Input tensor data type.
// @tparam     WeiDataType              Weights tensor data type.
// @tparam     OutDataType              Output tensor data type.
// @tparam     InElementwiseOperation   Functor for input tensor elementwise
//                                      operation.
// @tparam     WeiElementwiseOperation  Functor for weights tensor elementwise
//                                      operation.
// @tparam     NumAElementwiseTensor  Number of A elementwise tensors.
// @tparam     NumBElementwiseTensor  Number of B elementwise tensors.
// @tparam     NumDElementwiseTensor  Number of D elementwise tensors.
//
// input descriptor in [G, N, C, Do, Ho, Wo] order
// weight descriptor in [G, K, C, Z, Y, X] order
// output descriptor in [G, N, K, Di, Hi, Wi] order
// phyiscal layout is irrelavent
template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ck::index_t NumAElementwiseTensor                                         = 0,
          ck::index_t NumBElementwiseTensor                                         = 0,
          ck::index_t NumDElementwiseTensor                                         = 0,
          typename std::enable_if<NDimSpatial >= 1 && NDimSpatial <= 3, bool>::type = false>
struct ReferenceConvFwdTransposed : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(
            const Tensor<InDataType>& input,
            const Tensor<WeiDataType>& weight,
            Tensor<OutDataType>& output,
            std::vector<ck::long_index_t> conv_filter_strides,
            std::vector<ck::long_index_t> input_left_pads,
            std::vector<ck::long_index_t> input_right_pads,
            InElementwiseOperation in_element_op,
            WeiElementwiseOperation wei_element_op,
            OutElementwiseOperation out_element_op,
            const std::array<Tensor<InDataType>, NumAElementwiseTensor>& elementwise_a_tensors,
            const std::array<Tensor<WeiDataType>, NumBElementwiseTensor>& elementwise_b_tensors,
            const std::array<Tensor<OutDataType>, NumDElementwiseTensor>& elementwise_d_tensors)
            : input_{input},
              weight_{weight},
              output_{output},
              elementwise_a_tensors_{elementwise_a_tensors},
              elementwise_b_tensors_{elementwise_b_tensors},
              elementwise_d_tensors_{elementwise_d_tensors},
              conv_strides_{conv_filter_strides},
              in_left_pads_{input_left_pads},
              in_right_pads_{input_right_pads},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              out_element_op_{out_element_op}
        {
        }

        const Tensor<InDataType>& input_;
        const Tensor<WeiDataType>& weight_;
        Tensor<OutDataType>& output_;

        const std::array<Tensor<InDataType>, NumAElementwiseTensor>& elementwise_a_tensors_;
        const std::array<Tensor<WeiDataType>, NumBElementwiseTensor>& elementwise_b_tensors_;
        const std::array<Tensor<OutDataType>, NumDElementwiseTensor>& elementwise_d_tensors_;

        std::vector<long_index_t> conv_strides_;
        std::vector<long_index_t> in_left_pads_;
        std::vector<long_index_t> in_right_pads_;

        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        OutElementwiseOperation out_element_op_;
    };

    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceConvFwdTransposed::Argument;

        float Run(const Argument& arg)
        {
            if(!(arg.input_.GetNumOfDimension() == NDimSpatial + 3 &&
                 arg.weight_.GetNumOfDimension() == NDimSpatial + 3 &&
                 arg.output_.GetNumOfDimension() == NDimSpatial + 3))
            {
                throw std::runtime_error("wrong! inconsistent dimension");
            }

            if constexpr(NDimSpatial == 2)
            {
                auto func = [&](auto g, auto n, auto k, auto hi, auto wi) {
                    for(std::size_t y = 0; y < arg.weight_.GetLengths()[3]; ++y)
                    {
                        auto ho = static_cast<ck::long_index_t>(hi * arg.conv_strides_[0] + y -
                                                                arg.in_left_pads_[0]);

                        for(std::size_t x = 0; x < arg.weight_.GetLengths()[4]; ++x)
                        {
                            auto wo = static_cast<ck::long_index_t>(wi * arg.conv_strides_[1] + x -
                                                                    arg.in_left_pads_[1]);
                            float v_acc = 0;
                            if(ho >= 0 &&
                               ck::type_convert<std::size_t>(ho) < arg.output_.GetLengths()[3] &&
                               wo >= 0 &&
                               ck::type_convert<std::size_t>(wo) < arg.output_.GetLengths()[4])
                            {

                                for(std::size_t c = 0; c < arg.weight_.GetLengths()[2]; ++c)
                                {
                                    InDataType v_in;
                                    WeiDataType v_wei;

                                    ExecuteElementwiseOp(arg.in_element_op_,
                                                         arg.elementwise_a_tensors_,
                                                         Number<NumAElementwiseTensor>{},
                                                         v_in,
                                                         arg.input_(g, n, c, hi, wi),
                                                         g,
                                                         n,
                                                         c,
                                                         hi,
                                                         wi);
                                    ExecuteElementwiseOp(arg.wei_element_op_,
                                                         arg.elementwise_b_tensors_,
                                                         Number<NumBElementwiseTensor>{},
                                                         v_wei,
                                                         arg.weight_(g, k, c, y, x),
                                                         g,
                                                         k,
                                                         c,
                                                         y,
                                                         x);
                                    v_acc += ck::type_convert<float>(v_in) *
                                             ck::type_convert<float>(v_wei);
                                }
                            }

                            OutDataType v_acc_converted = ck::type_convert<OutDataType>(v_acc);
                            OutDataType& v_out          = arg.output_(g, n, k, ho, wo);
                            ExecuteElementwiseOp(arg.out_element_op_,
                                                 arg.elementwise_d_tensors_,
                                                 Number<NumDElementwiseTensor>{},
                                                 v_out,
                                                 v_acc_converted,
                                                 g,
                                                 n,
                                                 k,
                                                 ho,
                                                 wo);
                        }
                    }
                };

                make_ParallelTensorFunctor(func,
                                           arg.output_.GetLengths()[0],
                                           arg.output_.GetLengths()[1],
                                           arg.output_.GetLengths()[2],
                                           arg.input_.GetLengths()[3],
                                           arg.input_.GetLengths()[4])(
                    std::thread::hardware_concurrency());

                return 0;
            }
            throw std::runtime_error("Conv_fwd_transposed: only dimension 2 is supported!.");
            return 1;
        }

        float Run(const device::BaseArgument* p_arg,
                  const StreamConfig& /*stream_config*/ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        }
    };

    template <typename... Args,
              typename ElementwiseOp,
              typename ElementwiseTensor,
              typename NumTensor,
              typename T>
    static void ExecuteElementwiseOp(ElementwiseOp& elementwise_op,
                                     ElementwiseTensor& elementwise_tensors,
                                     NumTensor,
                                     T& y,
                                     const T& x,
                                     Args... dims)
    {
        if constexpr(NumTensor::value == 0)
        {
            elementwise_op(y, x);
        }
        else if constexpr(NumTensor::value == 1)
        {
            elementwise_op(y, x, elementwise_tensors[0](dims...));
        }
        else if constexpr(NumTensor::value == 2)
        {
            elementwise_op(y, x, elementwise_tensors[0](dims...), elementwise_tensors[1](dims...));
        }
        else
        {
            throw std::runtime_error("ElementOp not supported in reference.");
        }
    }

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    bool IsSupportedArgument(const device::BaseArgument*) override
    {
        return NDimSpatial >= 1 && NDimSpatial <= 3;
    }

    static auto MakeArgument(
        const Tensor<InDataType>& input,
        const Tensor<WeiDataType>& weight,
        Tensor<OutDataType>& output,
        std::vector<ck::long_index_t> conv_filter_strides,
        std::vector<ck::long_index_t> input_left_pads,
        std::vector<ck::long_index_t> input_right_pads,
        InElementwiseOperation in_element_op,
        WeiElementwiseOperation wei_element_op,
        OutElementwiseOperation out_element_op,
        const std::array<Tensor<InDataType>, NumAElementwiseTensor>& elementwise_a_tensors  = {},
        const std::array<Tensor<WeiDataType>, NumBElementwiseTensor>& elementwise_b_tensors = {},
        const std::array<Tensor<OutDataType>, NumDElementwiseTensor>& elementwise_d_tensors = {})
    {
        return Argument{input,
                        weight,
                        output,
                        conv_filter_strides,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        wei_element_op,
                        out_element_op,
                        elementwise_a_tensors,
                        elementwise_b_tensors,
                        elementwise_d_tensors};
    }

    static auto MakeInvoker() { return Invoker{}; }

    virtual std::unique_ptr<device::BaseInvoker> MakeInvokerPointer()
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "ReferenceConvFwdTransposed"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
