#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>

namespace hipconv
{
// Enumeration of data-types.
enum class DataType
{
    fp16,
    bf16,
    fp32,
    fp8,
    bf8,
    // TF32: stored as fp32, computed via 3x BF16 MFMA decomposition.
    tf32,
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

// All supported algorithms.
enum class Algorithm
{
    Grouped,
    // Depthwise (1 channel per group, groups == C == K).
    Depthwise,
    Direct,
    Pointwise
};

// Every Direction value, in declaration order.
inline constexpr std::array all_directions{Direction::Fprop, Direction::Dgrad, Direction::Wgrad};

// Every Algorithm value, in declaration order.
inline constexpr std::array all_algorithms{Algorithm::Grouped,
                                           Algorithm::Depthwise,
                                           Algorithm::Direct,
                                           Algorithm::Pointwise};

// DataType values the driver accepts as input tags, in declaration order.
inline constexpr std::array input_data_types{DataType::fp16, DataType::bf16, DataType::tf32};

// Return the size in bytes of the given data-type.
size_t sizeof_data_type(DataType dtype);

auto to_string(Direction dir) -> char const*;

auto to_string(DataType dtype) -> char const*;

auto to_string(Algorithm algo) -> char const*;

// Parse an algorithm name (the inverse of to_string(Algorithm)).
//
// Returns nullopt for any string that is not one of grouped|depthwise|direct|pointwise.
std::optional<Algorithm> parse_algorithm(std::string_view name);

// Parse a supported input data-type name (fp16|bf16|tf32).
//
// This is the set of dtypes the driver accepts for a layer's input, a subset of
// DataType: fp32, fp8, and bf8 are valid DataType values but not input tags, so
// they return nullopt. Returns nullopt for any unrecognized string.
std::optional<DataType> parse_data_type(std::string_view name);

// Parse a direction name into a mask of Direction bits.
//
// Accepts fprop, dgrad, and wgrad, plus "all" for every direction at once. A mask
// rather than a Direction is returned because "all" names three of them; callers
// that want a single direction test the bits. Returns nullopt for any other string.
//
// The accepted spelling is lowercase, which is what the command line and the spec
// files use; to_string(Direction) is capitalized, so this is not its strict inverse.
std::optional<int> parse_direction_mask(std::string_view name);

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
    int c;                              // input channels
    int k;                              // output channels
    int kh, kw;                         // filter size
    int pad_h = 1, pad_w = 1;           // padding
    int stride_h = 1, stride_w = 1;     // stride
    int dilation_h = 1, dilation_w = 1; // dilation
    int p = -1, q = -1;                 // output size
    int groups = 1;                     // number of channel groups

    DataType input_type       = DataType::fp16; // X
    DataType weight_type      = DataType::fp16; // W
    DataType output_type      = DataType::fp16; // Y
    DataType weight_grad_type = DataType::fp32; // dW

    TensorOrder order = TensorOrder::NHWC;

    // Overwrite (p, q) with output size derived from input size, padding, and stride.
    void compute_output_size()
    {
        p = (h + 2 * pad_h - kh) / stride_h + 1;
        q = (w + 2 * pad_w - kw) / stride_w + 1;
    }

    // Same as compute_output_size with additionally returning true if both p and q were originally
    // either
    // 1) unset (== -1)
    // 2) or match the computed value
    auto compute_output_size_checked() -> bool
    {
        const int p_original = p, q_original = q;
        compute_output_size();
        return (p_original == -1 || p == p_original) && (q_original == -1 || q == q_original);
    }

    bool is_valid() const
    {
        if(n <= 0 || h <= 0 || w <= 0 || c <= 0 || k <= 0 || kh <= 0 || kw <= 0)
        {
            return false;
        }
        // stride and dilation are divisors in compute_output_size; a zero or
        // negative value is not a valid layer and would divide by zero.
        if(stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0)
        {
            return false;
        }
        if(groups <= 0 || c % groups != 0 || k % groups != 0)
        {
            return false;
        }
        return true;
    }

    int channels_per_group() const { return c / groups; }
    int filters_per_group() const { return k / groups; }

    DataType input_grad_type() const { return input_type; }   // dX
    DataType output_grad_type() const { return output_type; } // dY

    // The datatype of the tensor that the conv2d operation writes.
    DataType result_type() const
    {
        switch(direction)
        {
        case Direction::Fprop:
            return output_type;
        case Direction::Dgrad:
            return input_grad_type();
        case Direction::Wgrad:
            return weight_grad_type;
        }
        return output_type;
    }
};

// Helper class that reports the size of Conv2d tensors.
//
// All methods return the number of elements in the tensor. Multiply sizeof_data_type
// to convert the size to the number of bytes.
class Conv2dSize
{
public:
    Conv2dSize(const Conv2dParams& par) : par_(par) {}

    size_t input_size() const { return static_cast<size_t>(par_.n) * par_.h * par_.w * par_.c; }

    size_t output_size() const { return static_cast<size_t>(par_.n) * par_.p * par_.q * par_.k; }

    size_t weight_size() const
    {
        return static_cast<size_t>(par_.k) * par_.channels_per_group() * par_.kh * par_.kw;
    }

    // dX has the same shape as X (n * h * w * c).
    size_t input_grad_size() const { return input_size(); }

    size_t input_bytes() const { return input_size() * sizeof_data_type(par_.input_type); }

    size_t output_bytes() const { return output_size() * sizeof_data_type(par_.output_type); }

    size_t weight_bytes() const { return weight_size() * sizeof_data_type(par_.weight_type); }

    size_t input_grad_bytes() const
    {
        return input_grad_size() * sizeof_data_type(par_.input_grad_type());
    }

    size_t output_grad_bytes() const
    {
        return output_size() * sizeof_data_type(par_.output_grad_type());
    }

    size_t weight_grad_bytes() const
    {
        return weight_size() * sizeof_data_type(par_.weight_grad_type);
    }

private:
    const Conv2dParams& par_;
};


template <hipconv::Direction D>
struct SizeView
{
    static_assert(D == Direction::Fprop || D == Direction::Dgrad,
                  "SizeView does not support Wgrad");

    const hipconv::Conv2dParams& par;

    SizeView(const hipconv::Conv2dParams& par_in) : par(par_in) {}

    int h() const
    {
        if constexpr(D == hipconv::Direction::Fprop)
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
        if constexpr(D == hipconv::Direction::Fprop)
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
        if constexpr(D == hipconv::Direction::Fprop)
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
        if constexpr(D == hipconv::Direction::Fprop)
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
        if constexpr(D == hipconv::Direction::Fprop)
            return par.pad_h;
        else
            return (par.kh - 1) * par.dilation_h - par.pad_h;
    }

    int pad_w() const
    {
        if constexpr(D == hipconv::Direction::Fprop)
            return par.pad_w;
        else
            return (par.kw - 1) * par.dilation_w - par.pad_w;
    }

    int stride_h() const
    {
        if constexpr(D == hipconv::Direction::Fprop)
            return par.stride_h;
        else
            return par.dilation_h;
    }

    int stride_w() const
    {
        if constexpr(D == hipconv::Direction::Fprop)
            return par.stride_w;
        else
            return par.dilation_w;
    }

    int dilation_h() const
    {
        if constexpr(D == hipconv::Direction::Fprop)
            return par.dilation_h;
        else
            return par.stride_h;
    }

    int dilation_w() const
    {
        if constexpr(D == hipconv::Direction::Fprop)
            return par.dilation_w;
        else
            return par.stride_w;
    }
};


} // namespace hipconv
