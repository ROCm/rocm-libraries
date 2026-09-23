#pragma once

#include "conv_kernel.h"
#include "hipconv/conv2d_params.hpp"

namespace hipconv
{

// Shared base for the depthwise convolution kernel family.
//
// Only checks the data invariants every depthwise kernel requires: fp16/bf16 with
// input == weight == output, and NHWC layout. The depthwise-shape gate
// (groups == C == K) lives in the backend's is_applicable; everything else is
// checked by the concrete kernel that overrides this.
class DepthwiseConvKernel : public ConvKernel
{
public:
    using ConvKernel::ConvKernel;

    std::string_view name() const override { return "depthwise"; }

    hipconv::Algorithm algorithm() const override { return hipconv::Algorithm::Depthwise; }

    bool is_applicable(const hipconv::Conv2dParams& par) const override
    {
        using namespace hipconv;
        if(par.input_type != DataType::fp16 && par.input_type != DataType::bf16)
            return false;
        if(par.weight_type != par.input_type || par.output_type != par.input_type)
            return false;
        if(par.order != TensorOrder::NHWC)
            return false;
        return true;
    }

    // A valid depthwise config is the dedicated 1c path for this shape.
    float get_weighted_throughput_index(const hipconv::Conv2dParams& /*par*/) const override
    {
        return 1.0f;
    }
};

} // namespace hipconv
