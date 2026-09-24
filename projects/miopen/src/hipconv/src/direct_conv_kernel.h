#pragma once

#include "conv_kernel.h"
#include "hipconv/conv2d_params.hpp"
#include "types.h"

#include <cmath>

namespace hipconv
{

class DirectConvKernel : public ConvKernel
{
public:
    using ConvKernel::ConvKernel;

    // Default family name; leaves that want a distinct name (e.g. direct_l1)
    // override this.
    std::string_view name() const override { return "direct"; }

    hipconv::Algorithm algorithm() const override { return hipconv::Algorithm::Direct; }

    bool is_applicable(const hipconv::Conv2dParams& par) const override
    {
        using namespace hipconv;

        if(par.input_type != DataType::fp16 && par.input_type != DataType::bf16)
            return false;
        if(par.weight_type != DataType::fp16 && par.weight_type != DataType::bf16)
            return false;
        if(par.output_type != DataType::fp16 && par.output_type != DataType::bf16)
            return false;
        if(par.input_type != par.weight_type || par.weight_type != par.output_type)
            return false;
        if(par.order != TensorOrder::NHWC)
            return false;
        if(par.direction == Direction::Wgrad)
            return false;
        if(par.stride_h != 1 || par.stride_w != 1)
            return false;
        return true;
    }

    // Full utilization at 128+ channels per group, discounted below.
    //
    // Direct is almost certainly the fastest kernel at 128+ channels per group;
    // below that it is a coverage fallback whose speed we can't vouch for, so a
    // different provider may beat it.
    float get_weighted_throughput_index(const hipconv::Conv2dParams& par) const override
    {
        constexpr int confident_channels = 128;
        return par.channels_per_group() >= confident_channels ? 1.0f : 0.5f;
    }
};

} // namespace hipconv
