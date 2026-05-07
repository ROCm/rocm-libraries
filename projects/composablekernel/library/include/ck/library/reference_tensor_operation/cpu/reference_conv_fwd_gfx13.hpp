// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

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
// @brief      Reference implementation for forward convolution for GFX13 architecture
//             It uses `acc_convert_interval_` to chunk the input channels (C) and perform
//             staged conversion of the FP32 accumulator back to the output type (half/bhalf),
//             3x3-specific tap reordering via `x_lookup`/`y_lookup` to align with FFM/CSIM
//             reference ordering, mimicking hardware paths.
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
struct ReferenceConvFwd_GFX13 : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(
            const Tensor<InDataType>& input,
            const Tensor<WeiDataType>& weight,
            Tensor<OutDataType>& output,
            std::vector<ck::long_index_t> conv_filter_strides,
            std::vector<ck::long_index_t> conv_filter_dilations,
            std::vector<ck::long_index_t> input_left_pads,
            std::vector<ck::long_index_t> input_right_pads,
            InElementwiseOperation in_element_op,
            WeiElementwiseOperation wei_element_op,
            OutElementwiseOperation out_element_op,
            const std::array<Tensor<InDataType>, NumAElementwiseTensor>& elementwise_a_tensors,
            const std::array<Tensor<WeiDataType>, NumBElementwiseTensor>& elementwise_b_tensors,
            const std::array<Tensor<OutDataType>, NumDElementwiseTensor>& elementwise_d_tensors,
            size_t acc_convert_interval,
            bool disable_bhalf_rne)
            : input_{input},
              weight_{weight},
              output_{output},
              elementwise_a_tensors_{elementwise_a_tensors},
              elementwise_b_tensors_{elementwise_b_tensors},
              elementwise_d_tensors_{elementwise_d_tensors},
              conv_strides_{conv_filter_strides},
              conv_dilations_{conv_filter_dilations},
              in_left_pads_{input_left_pads},
              in_right_pads_{input_right_pads},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              out_element_op_{out_element_op},
              acc_convert_interval_(acc_convert_interval),
              disable_bhalf_rne_(disable_bhalf_rne)
        {
        }

        const Tensor<InDataType>& input_;
        const Tensor<WeiDataType>& weight_;
        Tensor<OutDataType>& output_;

        const std::array<Tensor<InDataType>, NumAElementwiseTensor>& elementwise_a_tensors_;
        const std::array<Tensor<WeiDataType>, NumBElementwiseTensor>& elementwise_b_tensors_;
        const std::array<Tensor<OutDataType>, NumDElementwiseTensor>& elementwise_d_tensors_;

        std::vector<ck::long_index_t> conv_strides_;
        std::vector<ck::long_index_t> conv_dilations_;
        std::vector<ck::long_index_t> in_left_pads_;
        std::vector<ck::long_index_t> in_right_pads_;

        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        OutElementwiseOperation out_element_op_;
        size_t acc_convert_interval_;
        bool disable_bhalf_rne_;
    };

    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceConvFwd_GFX13::Argument;

        float Run(const Argument& arg)
        {
            if(!(arg.input_.GetNumOfDimension() == NDimSpatial + 3 &&
                 arg.weight_.GetNumOfDimension() == NDimSpatial + 3 &&
                 arg.output_.GetNumOfDimension() == NDimSpatial + 3))
            {
                throw std::runtime_error("wrong! inconsistent dimension");
            }

            static_assert(NDimSpatial == 2);
            static_assert(std::is_same<OutDataType, ck::bhalf_t>::value ||
                          std::is_same<OutDataType, ck::half_t>::value);
            if constexpr(NDimSpatial == 1)
            {
                static_assert(false, "Conv_fwd_gfx13_3x3: NDimSpatial 1 is not supported yet.");
                return 0;
            }
            else if constexpr(NDimSpatial == 2)
            {
                auto func = [&](auto g, auto n, auto k, auto ho, auto wo) {
                    float v_acc = 0;
                    // 3x3 filter using the same lookup as FFM and CSIM
                    std::array<uint32_t, 9> x_lookup = {1, 1, 0, 0, 0, 1, 2, 2, 2};
                    std::array<uint32_t, 9> y_lookup = {1, 0, 0, 1, 2, 2, 2, 1, 0};
                    auto is3x3 =
                        (arg.weight_.GetLengths()[3] == 3) && (arg.weight_.GetLengths()[4] == 3);

                    // loop over input channels in chunks defined by acc_convert_interval_ as FFM
                    // and CSIM
                    const std::size_t C = arg.weight_.GetLengths()[2];
                    const std::size_t interval =
                        (arg.acc_convert_interval_ == 0) ? C : arg.acc_convert_interval_;

                    for(std::size_t c_begin = 0; c_begin < C; c_begin += interval)
                    {
                        const std::size_t chunk_size =
                            std::min(interval, static_cast<std::size_t>(C - c_begin));

                        for(std::size_t yi = 0; yi < arg.weight_.GetLengths()[3]; ++yi)
                        {
                            for(std::size_t xi = 0; xi < arg.weight_.GetLengths()[4]; ++xi)
                            {
                                auto y = yi;
                                auto x = xi;
                                if(is3x3)
                                {
                                    // 3x3 specific logic
                                    auto filterWtPos = yi * 3 + xi;
                                    y                = y_lookup[filterWtPos];
                                    x                = x_lookup[filterWtPos];
                                }

                                auto hi =
                                    static_cast<ck::long_index_t>(ho * arg.conv_strides_[0]) +
                                    static_cast<ck::long_index_t>(y * arg.conv_dilations_[0]) -
                                    static_cast<ck::long_index_t>(arg.in_left_pads_[0]);

                                auto wi =
                                    static_cast<ck::long_index_t>(wo * arg.conv_strides_[1]) +
                                    static_cast<ck::long_index_t>(x * arg.conv_dilations_[1]) -
                                    static_cast<ck::long_index_t>(arg.in_left_pads_[1]);

                                if(hi >= 0 &&
                                   ck::type_convert<std::size_t>(hi) < arg.input_.GetLengths()[3] &&
                                   wi >= 0 &&
                                   ck::type_convert<std::size_t>(wi) < arg.input_.GetLengths()[4])
                                {
                                    for(std::size_t ci = 0; ci < chunk_size; ++ci)
                                    {
                                        std::size_t c = c_begin + ci;
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

                                    if(arg.disable_bhalf_rne_)
                                    {
                                        union
                                        {
                                            float fp32;
                                            uint32_t int32;
                                        } u     = {v_acc};
                                        u.int32 = (u.int32 >> 16) << 16;
                                        v_acc   = u.fp32;
                                    }
                                    else
                                    {
                                        OutDataType v_acc_converted =
                                            ck::type_convert<OutDataType>(v_acc);
                                        v_acc = ck::type_convert<float>(v_acc_converted);
                                    }
                                }
                            }
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
                };

                if((arg.weight_.GetLengths()[3] == 3) && (arg.weight_.GetLengths()[4] == 3))
                {
                    // we don't support stride = 2 in gfx13 3x3 convolution
                    if((arg.conv_strides_[0] != 1) || (arg.conv_strides_[1] != 1))
                    {
                        throw std::runtime_error(
                            "Conv_fwd_gfx13_3x3: stride must be 1 for 3x3 convolution.");
                    }
                }

                make_ParallelTensorFunctor(func,
                                           arg.output_.GetLengths()[0],
                                           arg.output_.GetLengths()[1],
                                           arg.output_.GetLengths()[2],
                                           arg.output_.GetLengths()[3],
                                           arg.output_.GetLengths()[4])(
                    std::thread::hardware_concurrency());

                return 0;
            }
            else if constexpr(NDimSpatial == 3)
            {
                static_assert(false, "Conv_fwd_gfx13_3x3: NDimSpatial 3 is not supported yet.");
                return 0;
            }
            throw std::runtime_error("Conv_fwd: number of dimensions must be between 1 and 3.");
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
        float y_f32;
        if constexpr(NumTensor::value == 0)
        {
            elementwise_op(y_f32, ck::type_convert<float>(x));
        }
        else if constexpr(NumTensor::value == 1)
        {
            elementwise_op(y_f32,
                           ck::type_convert<float>(x),
                           ck::type_convert<float>(elementwise_tensors[0](dims...)));
        }
        else if constexpr(NumTensor::value == 2)
        {
            elementwise_op(y_f32,
                           ck::type_convert<float>(x),
                           ck::type_convert<float>(elementwise_tensors[0](dims...)),
                           ck::type_convert<float>(elementwise_tensors[1](dims...)));
        }
        else
        {
            throw std::runtime_error("ElementOp not supported in reference.");
        }
        y = ck::type_convert<T>(y_f32);
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
        std::vector<ck::long_index_t> conv_filter_dilations,
        std::vector<ck::long_index_t> input_left_pads,
        std::vector<ck::long_index_t> input_right_pads,
        InElementwiseOperation in_element_op,
        WeiElementwiseOperation wei_element_op,
        OutElementwiseOperation out_element_op,
        const std::array<Tensor<InDataType>, NumAElementwiseTensor>& elementwise_a_tensors  = {},
        const std::array<Tensor<WeiDataType>, NumBElementwiseTensor>& elementwise_b_tensors = {},
        const std::array<Tensor<OutDataType>, NumDElementwiseTensor>& elementwise_d_tensors = {},
        size_t acc_convert_interval                                                         = 0,
        bool disable_bhalf_rne                                                              = false)
    {
        return Argument{input,
                        weight,
                        output,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        wei_element_op,
                        out_element_op,
                        elementwise_a_tensors,
                        elementwise_b_tensors,
                        elementwise_d_tensors,
                        acc_convert_interval,
                        disable_bhalf_rne};
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
        str << "ReferenceConvFwd_GFX13"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
