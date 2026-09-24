#include "hipconv/conv2d_params.hpp"

#include "unreachable.h"

namespace hipconv
{

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
