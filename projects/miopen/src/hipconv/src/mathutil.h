#pragma once

#include <bit>
#include <cstdint>

// builtin max is not constexpr, so we define our own. These are plain constexpr
// (no __device__ __host__): HIP treats constexpr functions as host+device, and
// dropping the qualifiers lets the non-HIP mathutil_test compile under a plain
// C++ compiler.
inline constexpr int maximum(int a, int b)
{
    return (a > b) ? a : b;
}

inline constexpr int minimum(int a, int b)
{
    return (a < b) ? a : b;
}

inline constexpr int divup(int x, int y)
{
    return (x + y - 1) / y;
}

// Return the least multiple of divisor greater than or equal to x.
inline constexpr int make_divisible(int x, int divisor)
{
    return divup(x, divisor) * divisor;
}

// Factor n = pow2 * odd, where pow2 is the largest power of two dividing n.
//
// Requires n > 0. Used to size the direct_l1 K-partition: the power-of-two
// factor of the K-block count decides how evenly the blocks split across the
// power-of-two XCD count.
struct SplitPow2
{
    uint32_t pow2;
    uint32_t odd;
};

inline constexpr SplitPow2 split_pow2(uint32_t n)
{
    // countr_zero(0) is 32 and a shift by 32 is undefined; n == 0 has no
    // largest-power-of-two factor, so report the trivial factoring 1 * 0.
    if(n == 0)
        return {1, 0};
    const int k = std::countr_zero(n);
    return {uint32_t(1) << k, n >> k};
}
