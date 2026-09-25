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
// this -- including whether it implements tf32, which only the CDNA5 1D Toeplitz one does.
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

} // namespace hipconv
