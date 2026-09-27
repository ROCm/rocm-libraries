#pragma once

#include <cstdint>

// builtin max is not constexpr, so we define our own. These are plain constexpr
// (no __device__ __host__): HIP treats constexpr functions as host+device, and
// dropping the qualifiers lets the non-HIP mathutil_test compile under a plain
// C++ compiler.
//
// The same host+device treatment is why <bit> and <numeric> are not used here.
// MSVC implements std::countr_zero with runtime ISA dispatch through the host
// global __isa_available, and std::gcd on top of it, so a device call to either
// fails to compile on Windows. See docs/windows-device-compilation.md. Where a
// replacement uses a clang builtin it carries a plain-C++ fallback, because
// mathutil_test builds as CXX and need not be compiled by clang.
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

// Greatest common divisor of two non-negative values.
inline constexpr int gcd(int a, int b)
{
    while(b != 0)
    {
        const int remainder = a % b;
        a                   = b;
        b                   = remainder;
    }
    return a;
}

// Least common multiple of two non-negative values.
//
// Zero when either argument is zero, as std::lcm is, which also keeps gcd's zero out of
// the divisor. Divides before multiplying so the intermediate cannot overflow where the
// result itself fits.
inline constexpr int lcm(int a, int b)
{
    if(a == 0 || b == 0)
        return 0;
    return (a / gcd(a, b)) * b;
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
    // n == 0 has no largest-power-of-two factor, so report the trivial factoring 1 * 0.
    // __builtin_ctz(0) is undefined and the fallback scan would not terminate either.
    if(n == 0)
        return {1, 0};
#if defined(__clang__) || defined(__GNUC__)
    const int k = __builtin_ctz(n);
#else
    // Portable fallback for mathutil_test under a compiler without the builtin. Clang
    // recognises this idiom at only a few of the call sites, so it is a fallback rather
    // than the implementation.
    int k = 0;
    while(((n >> k) & 1u) == 0u)
        ++k;
#endif
    return {uint32_t(1) << k, n >> k};
}
