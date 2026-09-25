#include "tolerance.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace hipconv
{

float get_unit_roundoff(DataType dtype)
{
    switch(dtype)
    {
    case DataType::fp16:
    case DataType::tf32:
        return 0x1p-11f; // 2^{-(10+1)}, 10 mantissa bits
    case DataType::bf16:
        return 0x1p-8f; // 2^{-(7+1)},  7 mantissa bits
    case DataType::fp32:
        return 0x1p-24f; // 2^{-(23+1)}, 23 mantissa bits
    default:
        throw std::invalid_argument("unsupported data type");
    }
}

size_t get_accumulation_depth(const Conv2dParams& par)
{
    if(par.direction == Direction::Wgrad)
        return static_cast<size_t>(par.n) * par.p * par.q;
    if(par.direction == Direction::Fprop)
        return static_cast<size_t>(par.channels_per_group()) * par.kh * par.kw;
    if(par.direction == Direction::Dgrad)
        return static_cast<size_t>(par.filters_per_group()) * par.kh * par.kw;
    throw std::invalid_argument("unsupported convolution direction");
}

// Error bound for mixed-precision matrix multiply (low-precision inputs, fp32 accumulation).
//
// From Blanchard, Higham, Lopez, Mary, Pranesh, "Mixed Precision Block Fused Multiply-Add:
// Error Analysis and Application to GPU Tensor Cores", SIAM J. Sci. Comput., 2020,
// doi:10.1137/19M1289546.
//
// FP16/BF16
// =========
// Modeled scenario: The inputs are given in low precision (no rounding), the accumulation
// is in-register in high precision (e.g. u_high = 2^{-24} for fp32), and the output
// is rounded to low precision before storing.
// For the resulting tensor C_high we have before rounding
//
// |C-C_high| <= gamma(n, u_high) conv(|A|,|B|)
//
// where n is the accumulation depth, and conv(|A|,|B|) is the convolution
// of the componentwise absolute values.
//
// n is the caller's `depth`, which is the number of fp32 roundings on the longest path from a
// product to the result. Recursive summation makes that the whole contraction, but a blocked
// accumulation makes it far smaller, and the analysis below is indifferent to which.
// (Cf. Theorem 3.1 with u_bar = u_fma = u = u_high; here identical to Higham's classical estimate
// of the error of the dot product given in "Accuracy and Stability of Numerical Algorithms".)
//
// If u_high != u_low, final rounding introduces an error of order u_low.
//
// |C-C_low| <= |C-C_high| + |C_high-C_low|
//           <= gamma(n, u_high) conv(|A|,|B|) + u_low |C_high|
//
// With |C_high| <= |C| + |C_high-C| <= (1 + gamma(n, u_high)) conv(|A|,|B|)
// we arrive at the final estimate
//
// |C-C_low| <= (u_low + gamma(n, u_fp32) * (1 + u_low)) conv(|A|,|B|)
//
// TF32
// ====
// Modeled scenario: The inputs are given in FP32, the accumulation is in-register in high
// precision, and the output is stored FP32. FP32 is emulated via 3xBF16 MFMA/WMMA instructions,
// giving lower precision than FP32 but higher precision than TF32.
//
// Tensors are split into A = A_b + A_s and B = B_b + B_s, where
// A_b = toBF16(A), A_s = A - toFP32(A_b)
// B_b = toBF16(B), B_s = B - toFP32(B_b)
// BF16 giving us 7 + 1 bits of precision we have
// |A_s| <= u_low |A_b|
// |B_s| <= u_low |B_b|
// u_low = 2^{-8}
//
// Using the exact split, we have
// C = conv(A,B) = conv(A_b + A_s, B_b + B_s)
//   = conv(A_b, B_b) + conv(A_s, B_b) + conv(A_b, B_s) + conv(A_s, B_s)
//
// Assuming that A_s and B_s can be represented exactly in BF16, and dropping the small-small term,
// we obtain
// Ctilde = conv(A_b, B_b) + conv(A_s, B_b) + conv(A_b, B_s) + dC,
// |dC| <= gamma(3*n, u_high) (conv(|A_b|, |B_b|) + conv(|A_s|, |B_b|) + conv(|A_b|, |B_s|))
//      <= (1 + 2*u_low) * gamma(3*n, u_high) conv(|A_b|, |B_b|)
// and get the error bound
// |C-Ctilde| = |conv(A_s, B_s) - dC|
//            <= (u_low^2 + (1 + 2*u_low) * gamma(3*n, u_high)) conv(|A_b|, |B_b|)
//
// Inputs are passed unrounded to the oracle, thus we approximate A_s and B_s by Ahat_s and Bhat_s:
// Ahat_s = A_s + dA_s, |dA_s| <= u_low |A_s|, |Ahat_s| <= u_low |A_b|,
// Bhat_s = B_s + dB_s, |dB_s| <= u_low |B_s|, |Bhat_s| <= u_low |B_b|.
// Plugging in the hat-tensors we have
// Chat = conv(A_b, B_b) + conv(Ahat_s, B_b) + conv(A_b, Bhat_s) + dChat,
// |dChat| <= (1 + 2*u_low) * gamma(3*n, u_high) conv(|A_b|, |B_b|)
// and get the error bound
// |C-Chat| = |conv(A_s, B_s) - conv(dA_s, B_b) - conv(A_b, dB_s) - dChat|
//          <= conv(|A_s|, |B_s|) + conv(|dA_s|, |B_b|) + conv(|A_b, |dB_s|) + |dChat|
//          <= (3*u_low^2 + (1 + 2*u_low) gamma(3*n, u_high)) conv(|A_b|, |B_b|)
//
// Lastly, we have
// A_b = A + dA_b, |dA_b| <= u_low |A|,
// B_b = B + dB_b, |dB_b| <= u_low |B|.
// Therefore,
// conv(|A_b|, |B_b|) <= conv(|A|, |B|) + conv(|A|, |dB_b|) + conv(|dA_b|, |B|)
//                    <= (1 + 2*u_low) * conv(|A|, |B|)
// giving the final error bound
// |C-Ctilde| = (3*u_low^2 + (1 + 2*u_low) gamma(3*n, u_high)) (1 + 2*u_low) conv(|A|, |B|)
//
void get_mixed_precision_tolerance(const Conv2dParams& par, size_t depth, float& atol, float& rtol)
{
    // u_high is the per-multiply error of the MFMA/WMMA pipeline. This is the fp32 accumulation
    // roundoff (2^{-24}).
    auto u_high = get_unit_roundoff(DataType::fp32);
    auto n      = depth;
    // gamma(n, u) = nu/(1 - nu) holds only under nu < 1, assumed throughout the paper cited above.
    //
    // Past that the denominator crosses zero and the expression turns negative. Infinity stands in,
    // and the check after the branches turns it into TOLERANCE_UNAVAILABLE.
    const auto gamma = [](auto n, double u) {
        const double nu = static_cast<double>(n) * u;
        return nu < 1.0 ? nu / (1.0 - nu) : std::numeric_limits<double>::infinity();
    };
    atol = 0.0f;

    if(par.input_type == DataType::tf32)
    {
        const auto u_low  = get_unit_roundoff(DataType::bf16);
        const auto u_drop = u_low * u_low;
        rtol              = 3 * u_drop + (1 + 2 * u_low) * gamma(3 * n, u_high);
        rtol              = rtol * (1 + 2 * u_low);
    }
    else
    {
        const auto u_low = get_unit_roundoff(par.result_type());
        rtol             = gamma(n, u_high);
        if(u_low != u_high)
        {
            rtol = u_low + rtol * (1 + u_low);
        }
    }

    // Past the hypothesis there is no bound to report.
    //
    // A bound that survives it can still be too loose to check anything against, which is the
    // caller's judgement to make; see docs/algorithms/direct/direct-wgrad-tolerance.md.
    if(!std::isfinite(rtol))
    {
        rtol = TOLERANCE_UNAVAILABLE;
    }
}

void get_mixed_precision_tolerance(const Conv2dParams& par, float& atol, float& rtol)
{
    // Recursive summation over the whole contraction, which is the depth a kernel has unless it
    // says otherwise.
    get_mixed_precision_tolerance(par, get_accumulation_depth(par), atol, rtol);
}

} // namespace hipconv
