// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile::direct_conv
{

enum class DataType
{
    fp16,
    bf16,
    fp32,
    fp8,
    bf8,
};

// The direction of the operation (forward (Fprop) or gradient type).
enum class Direction
{
    // Forward
    Fprop = 1,

    // Input gradient
    Dgrad = 2,

    // Weights gradient
    Wgrad = 4
};

inline auto to_string(Direction dir) -> char const*
{
    switch(dir)
    {
    case Direction::Fprop:
        return "Fprop";
    case Direction::Dgrad:
        return "Dgrad";
    case Direction::Wgrad:
        return "Wgrad";
    }
}

// The tensor order (i.e. layout).
enum TensorOrder
{
    // Channels first
    NCHW,

    // Channels last
    NHWC
};

// Define the parameters for a conv2d layer.
struct Conv2dParams
{
    Direction direction = Direction::Fprop;
    int n;                              // batch size
    int h, w;                           // input size
    int c_tot;                          // input channels (over all groups)
    int k_tot;                          // output channels (over all groups)
    int kh, kw;                         // filter size
    int pad_h = 1, pad_w = 1;           // padding
    int stride_h = 1, stride_w = 1;     // stride
    int dilation_h = 1, dilation_w = 1; // dilation
    int p, q;                           // output size
    int groups        = 1;              // number of channel groups
    DataType in_type  = DataType::fp16;
    DataType wei_type = DataType::fp16;
    DataType out_type = DataType::fp16;
    TensorOrder order = TensorOrder::NHWC;

    // Overwrite (p, q) with output size derived from input size, padding, and stride.
    void compute_output_size()
    {
        p = (h + 2 * pad_h - kh) / stride_h + 1;
        q = (w + 2 * pad_w - kw) / stride_w + 1;
    }

    bool is_valid() const
    {
        if(n <= 0 || h <= 0 || w <= 0 || c_tot <= 0 || k_tot <= 0 || kh <= 0 || kw <= 0)
        {
            return false;
        }
        if(groups <= 0 || c_tot % groups != 0 || k_tot % groups != 0)
        {
            return false;
        }
        return true;
    }

    int channels_per_group() const { return c_tot / groups; }
    int filters_per_group() const { return k_tot / groups; }

    // True for standard (non-grouped) convolution: G=1 with C > 32.
    // These cases require C-reduction across multiple MFMA iterations.
    bool is_non_grouped() const { return groups == 1 && channels_per_group() > 32; }
};


template <Direction D>
struct SizeView
{
    static_assert(D == Direction::Fprop || D == Direction::Dgrad,
                  "SizeView does not support Wgrad");

    const Conv2dParams& par;

    SizeView(const Conv2dParams& par_in) : par(par_in) {}

    int h() const
    {
        if constexpr(D == Direction::Fprop)
        {
            return par.h;
        }
        else
        {
            return par.p;
        }
    }

    int w() const
    {
        if constexpr(D == Direction::Fprop)
        {
            return par.w;
        }
        else
        {
            return par.q;
        }
    }

    int p() const
    {
        if constexpr(D == Direction::Fprop)
        {
            return par.p;
        }
        else
        {
            return par.h;
        }
    }

    int q() const
    {
        if constexpr(D == Direction::Fprop)
        {
            return par.q;
        }
        else
        {
            return par.w;
        }
    }

    int pad_h() const
    {
        if constexpr(D == Direction::Fprop)
            return par.pad_h;
        else
            return (par.kh - 1) * par.dilation_h - par.pad_h;
    }

    int pad_w() const
    {
        if constexpr(D == Direction::Fprop)
            return par.pad_w;
        else
            return (par.kw - 1) * par.dilation_w - par.pad_w;
    }

    int stride_h() const
    {
        if constexpr(D == Direction::Fprop)
            return par.stride_h;
        else
            return par.dilation_h;
    }

    int stride_w() const
    {
        if constexpr(D == Direction::Fprop)
            return par.stride_w;
        else
            return par.dilation_w;
    }

    int dilation_h() const
    {
        if constexpr(D == Direction::Fprop)
            return par.dilation_h;
        else
            return par.stride_h;
    }

    int dilation_w() const
    {
        if constexpr(D == Direction::Fprop)
            return par.dilation_w;
        else
            return par.stride_w;
    }
};


} // namespace ck_tile::direct_conv
