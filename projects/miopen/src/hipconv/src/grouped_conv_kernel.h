#pragma once

#include "conv_kernel.h"
#include "hipconv/conv_params.hpp"

#include <cstdint>

namespace hipconv
{

// Whether one image of the tensor a grouped fprop or dgrad kernel reads, X or dY, passes 2 GiB.
inline bool grouped_image_past_2gib(const ConvParams& par)
{
    const ConvSize sz(par);
    const size_t bytes =
        par.direction == Direction::Dgrad ? sz.output_grad_bytes() : sz.input_bytes();
    return bytes / par.n > INT32_MAX;
}

class GroupedConvKernel : public ConvKernel
{
public:
    using ConvKernel::ConvKernel;

    std::string_view name() const override { return "grouped"; }

    hipconv::Algorithm algorithm() const override { return hipconv::Algorithm::Grouped; }

    bool is_applicable(const hipconv::Device& /*dev*/,
                       const hipconv::ConvParams& par) const override
    {
        using namespace hipconv;

        // Supported dtype combinations:
        //   fp16/bf16: input == weight == output (same type)
        //   tf32:      input == weight == tf32, output == fp32
        // Arches without tf32 support narrow this by deriving from an
        // intermediate base that rejects non-fp16 input.
        const bool ok_fp16bf16 =
            (par.input_type == DataType::fp16 || par.input_type == DataType::bf16) &&
            par.weight_type == par.input_type && par.output_type == par.input_type;
        const bool ok_tf32 = par.input_type == DataType::tf32 &&
                             par.weight_type == DataType::tf32 && par.output_type == DataType::fp32;
        if(!ok_fp16bf16 && !ok_tf32)
            return false;
        if(par.order != TensorOrder::NHWC)
            return false;
        if(par.direction != Direction::Fprop && par.direction != Direction::Dgrad)
            return false;
        if(par.kh != 3 || par.kw != 3)
            return false;
        if(par.k != par.c)
            return false;
        if(par.channels_per_group() != group_channels())
            return false;
        if(par.c % group_channels() != 0)
            return false;
        if((par.stride_h != 1 && par.stride_h != 2) || (par.stride_w != 1 && par.stride_w != 2))
            return false;
        if(par.dilation_h != 1 || par.dilation_w != 1)
            return false;
        if(par.pad_h > par.kh - 1 || par.pad_w > par.kw - 1)
            return false;
        // grouped_image_past_2gib, which every config checks, divides by the batch size.
        if(par.n <= 0)
            return false;
        // Descriptors span an image of X, or of dY in dgrad, or past 2 GiB a row of one, and the
        // weights one descriptor. Row, row-band and plane counts are ints; a dgrad upsample and
        // the bands' rounding take rows past the image's, so rows stop at 2^29.
        const bool fprop     = par.direction == Direction::Fprop;
        const int64_t width  = fprop ? par.w : par.q;
        const int64_t rows   = fprop ? int64_t(par.d) * par.h : int64_t(par.e) * par.p;
        const int64_t planes = int64_t(par.n) * (fprop ? par.e : par.d);
        return width * par.c * sizeof_data_type(par.input_type) <= INT32_MAX &&
               rows <= (int64_t(1) << 29) && planes <= INT32_MAX &&
               ConvSize(par).weight_bytes() <= INT32_MAX;
    }

    // A valid grouped config is almost certainly the fastest kernel.
    float get_weighted_throughput_index(const hipconv::Device& /*dev*/,
                                        const hipconv::ConvParams& /*par*/) const override
    {
        return 1.0f;
    }

protected:
    virtual int group_channels() const = 0;
};

class GroupedWgradConvKernel : public ConvKernel
{
public:
    using ConvKernel::ConvKernel;

    std::string_view name() const override { return "grouped_wgrad"; }

    hipconv::Algorithm algorithm() const override { return hipconv::Algorithm::Grouped; }

    bool is_applicable(const hipconv::Device& /*dev*/,
                       const hipconv::ConvParams& par) const override
    {
        using namespace hipconv;

        // Supported dtype combinations for wgrad:
        //   fp16/bf16: input == delta == dtype, dW == fp32 (or dtype if supports_narrow_dw())
        //   tf32:      input == tf32, delta == fp32, dW == fp32 (delta stores fp32
        //              because tf32 has fp32 storage). Arches without tf32 wgrad
        //              narrow this per-config via is_valid_config.
        const bool ok_fp16bf16 =
            (par.input_type == DataType::fp16 || par.input_type == DataType::bf16) &&
            par.weight_type == par.input_type && par.output_grad_type() == par.input_type;
        const bool ok_tf32 =
            par.input_type == DataType::tf32 && par.output_grad_type() == DataType::fp32;
        if(!ok_fp16bf16 && !ok_tf32)
            return false;
        if(par.direction != Direction::Wgrad)
            return false;
        if(par.order != TensorOrder::NHWC)
            return false;
        // dW is fp32, or for a family that sums in fp32 and rounds once, the input's 16-bit type.
        const bool narrow_dw =
            supports_narrow_dw() && ok_fp16bf16 && par.weight_grad_type == par.input_type;
        if(par.weight_grad_type != DataType::fp32 && !narrow_dw)
            return false;
        if(par.kh != 3 || par.kw != 3)
            return false;
        if(par.k != par.c)
            return false;
        if(par.channels_per_group() != group_channels())
            return false;
        if(par.c % group_channels() != 0)
            return false;
        const bool stride_ok = (par.stride_h == 1 && par.stride_w == 1) ||
                               (supports_stride2() && par.stride_h == 2 && par.stride_w == 2);
        if(!stride_ok)
            return false;
        if(par.dilation_h != 1 || par.dilation_w != 1)
            return false;
        if(par.pad_h > par.kh - 1 || par.pad_w > par.kw - 1)
            return false;
        // Buffer load instructions use 32-bit offsets and the buffer descriptor's
        // NUM_RECORDS field is 32 bits. Reject tensors >= 4 GB.
        ConvSize sz(par);
        if(sz.input_bytes() > INT32_MAX)
            return false;
        if(sz.output_grad_bytes() > INT32_MAX)
            return false;
        return true;
    }

    // A valid grouped wgrad config is almost certainly the fastest kernel.
    float get_weighted_throughput_index(const hipconv::Device& /*dev*/,
                                        const hipconv::ConvParams& /*par*/) const override
    {
        return 1.0f;
    }

    // Workgroups add dW tiles atomically; a family that reduces them in a fixed order overrides
    // this.
    bool is_deterministic(const hipconv::Device& /*dev*/,
                          const hipconv::ConvParams& /*par*/) const override
    {
        return false;
    }

protected:
    virtual bool supports_narrow_dw() const { return false; }
    virtual int group_channels() const = 0;
    virtual bool supports_stride2() const { return false; }
};

} // namespace hipconv
