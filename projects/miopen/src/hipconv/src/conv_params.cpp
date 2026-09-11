#include "hipconv/conv_params.hpp"

#include "unreachable.h"

#include <cstdint>
#include <limits>

namespace hipconv
{

namespace
{
// Does a * b still fit the int extents ConvParams stores?
bool product_fits_int(int a, int b)
{
    return static_cast<std::int64_t>(a) * b <= std::numeric_limits<int>::max();
}
} // namespace

ConvParams ConvParams::unfolded() const
{
    ConvParams out = *this;
    // Normalize before dims moves, since normalize_unused_axes keys off it.
    //
    // is_valid admits a conv1d carrying the default pad_h of 1 -- an unused
    // axis's padding is the one field it does not check -- and at dims 2 that
    // axis is live, so folding first would derive p = 3 from a unit height.
    out.normalize_unused_axes();
    out.p = out.q = out.e = -1;

    // A normalized conv1d is already a conv2d of unit height.
    if(dims == 1)
    {
        out.dims = 2;
        out.compute_output_size();
        return out;
    }
    if(dims != 3)
        return *this;

    // Case A: depth is not convolved, so each depth plane is an independent
    // image and folds into the batch. Preferred when case B also applies, since
    // it leaves the h/w geometry the 2D kernels are tuned for.
    if(kd == 1 && pad_d == 0 && stride_d == 1 && dilation_d == 1 && product_fits_int(n, d))
    {
        out.dims = 2;
        out.n    = n * d;
        out.compute_output_size();
        return out;
    }

    // Case B: spatially pointwise, so depth becomes the convolved axis and the
    // pointwise plane flattens into width.
    if(kh == 1 && kw == 1 && pad_h == 0 && pad_w == 0 && stride_h == 1 && stride_w == 1 &&
       dilation_h == 1 && dilation_w == 1 && product_fits_int(h, w))
    {
        out.dims       = 2;
        out.h          = d;
        out.w          = h * w;
        out.kh         = kd;
        out.kw         = 1;
        out.pad_h      = pad_d;
        out.pad_w      = 0;
        out.stride_h   = stride_d;
        out.stride_w   = 1;
        out.dilation_h = dilation_d;
        out.dilation_w = 1;
        out.compute_output_size();
        return out;
    }

    return *this;
}

size_t sizeof_data_type(DataType dtype)
{
    switch(dtype)
    {
    case DataType::fp8:
    case DataType::bf8:
        return 1;
    case DataType::fp16:
    case DataType::bf16:
        return 2;
    case DataType::fp32:
    case DataType::tf32:
        // TF32 is stored as fp32 (the BF16 decomposition happens inside the kernel).
        return 4;
    }
    HIPCONV_UNREACHABLE();
}

auto to_string(Direction dir) -> char const*
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
    HIPCONV_UNREACHABLE();
}

auto to_string(DataType dtype) -> char const*
{
    switch(dtype)
    {
    case DataType::fp16:
        return "fp16";
    case DataType::bf16:
        return "bf16";
    case DataType::fp32:
        return "fp32";
    case DataType::fp8:
        return "fp8";
    case DataType::bf8:
        return "bf8";
    case DataType::tf32:
        return "tf32";
    }
    HIPCONV_UNREACHABLE();
}

auto to_string(Algorithm algo) -> char const*
{
    switch(algo)
    {
    case Algorithm::Grouped:
        return "grouped";
    case Algorithm::Depthwise:
        return "depthwise";
    case Algorithm::Direct:
        return "direct";
    case Algorithm::Pointwise:
        return "pointwise";
    }
    HIPCONV_UNREACHABLE();
}

std::optional<Algorithm> parse_algorithm(std::string_view name)
{
    for(auto algo : all_algorithms)
        if(name == to_string(algo))
            return algo;
    return std::nullopt;
}

std::optional<DataType> parse_data_type(std::string_view name)
{
    for(auto dtype : input_data_types)
        if(name == to_string(dtype))
            return dtype;
    return std::nullopt;
}

std::optional<int> parse_direction_mask(std::string_view name)
{
    if(name == "fprop")
        return static_cast<int>(Direction::Fprop);
    if(name == "dgrad")
        return static_cast<int>(Direction::Dgrad);
    if(name == "wgrad")
        return static_cast<int>(Direction::Wgrad);
    if(name == "all")
    {
        int mask = 0;
        for(auto dir : all_directions)
            mask |= static_cast<int>(dir);
        return mask;
    }
    return std::nullopt;
}

} // namespace hipconv
