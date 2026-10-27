#pragma once

#include "conv_kernel.h"
#include "hipconv/conv_params.hpp"

namespace hipconv
{

// Shared base for the depthwise convolution kernel family.
//
// Only checks the data invariants every depthwise kernel requires: a supported dtype
// combination and NHWC layout. The depthwise-shape gate (groups == C == K) lives in the
// backend's is_applicable; everything else is checked by the concrete kernel that overrides
// this -- including whether it implements tf32, which the CDNA4 and CDNA5 1D Toeplitz ones do.
class DepthwiseConvKernel : public ConvKernel
{
public:
    using ConvKernel::ConvKernel;

    std::string_view name() const override { return "depthwise"; }

    hipconv::Algorithm algorithm() const override { return hipconv::Algorithm::Depthwise; }

    bool is_applicable(const hipconv::ConvParams& par) const override
    {
        using namespace hipconv;
        const bool ok_fp16bf16 =
            (par.input_type == DataType::fp16 || par.input_type == DataType::bf16) &&
            par.weight_type == par.input_type && par.output_type == par.input_type;
        const bool ok_tf32 = par.input_type == DataType::tf32 &&
                             par.weight_type == DataType::tf32 && par.output_type == DataType::fp32;
        if(!ok_fp16bf16 && !ok_tf32)
            return false;
        if(par.order != TensorOrder::NHWC)
            return false;
        return true;
    }

    // A valid depthwise config is the dedicated 1c path for this shape.
    float get_weighted_throughput_index(const hipconv::ConvParams& /*par*/) const override
    {
        return 1.0f;
    }
};

// Shared base for depthwise weights-gradient kernels. The two contraction
// operands remain fp16/bf16, while dW is accumulated and returned as fp32.
class DepthwiseWgradConvKernel : public ConvKernel
{
public:
    using ConvKernel::ConvKernel;

    std::string_view name() const override { return "depthwise_wgrad_hankel"; }

    hipconv::Algorithm algorithm() const override { return hipconv::Algorithm::Depthwise; }

    bool is_applicable(const hipconv::ConvParams& par) const override
    {
        using namespace hipconv;
        if(par.direction != Direction::Wgrad)
            return false;
        if(par.input_type != DataType::fp16 && par.input_type != DataType::bf16)
            return false;
        if(par.output_grad_type() != par.input_type)
            return false;
        if(par.weight_grad_type != DataType::fp32)
            return false;
        if(par.order != TensorOrder::NHWC)
            return false;
        if(par.stride_h != par.stride_w || (par.stride_h != 1 && par.stride_h != 2))
            return false;
        if(par.dilation_h != 1 || par.dilation_w != 1)
            return false;
        if(par.pad_h > par.kh - 1 || par.pad_w > par.kw - 1)
            return false;
        ConvSize sz(par);
        return sz.input_bytes() <= INT32_MAX && sz.output_grad_bytes() <= INT32_MAX;
    }

    float get_weighted_throughput_index(const hipconv::ConvParams&) const override { return 1.0f; }
};

} // namespace hipconv
