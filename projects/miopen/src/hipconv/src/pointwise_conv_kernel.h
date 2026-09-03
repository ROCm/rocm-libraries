#pragma once

#include "conv_kernel.h"
#include "hipconv/conv2d_params.hpp"

namespace hipconv
{

class PointwiseConvKernel : public ConvKernel
{
public:
    using ConvKernel::ConvKernel;

    std::string_view name() const override { return "pointwise"; }

    hipconv::Algorithm algorithm() const override { return hipconv::Algorithm::Pointwise; }

    bool is_applicable(const hipconv::Conv2dParams& par) const override
    {
        using namespace hipconv;

        const bool ok_fp16bf16 =
            (par.input_type == DataType::fp16 || par.input_type == DataType::bf16) &&
            par.weight_type == par.input_type &&
            (par.direction == Direction::Wgrad ? par.weight_grad_type == DataType::fp32
                                               : par.output_type == par.input_type);
        if(!ok_fp16bf16)
            return false;
        if(par.order != TensorOrder::NHWC)
            return false;
        if(par.groups != 1)
            return false;
        if(par.stride_h != 1 || par.stride_w != 1)
            return false;
        if(par.dilation_h != 1 || par.dilation_w != 1)
            return false;
        if(par.pad_h != 0 || par.pad_w != 0)
            return false;
        if(par.p != par.h || par.q != par.w)
            return false;

        return true;
    }

    // The hipBLASLt-backed pointwise path is the tuned choice for 1x1.
    float get_weighted_throughput_index(const hipconv::Conv2dParams& /*par*/) const override
    {
        return 1.0f;
    }
};

} // namespace hipconv
