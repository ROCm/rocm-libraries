// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

/// @file DeterministicRandom.hpp
/// @brief Draws that give the same values on every standard library for a given seed.
///
/// `std::mt19937_64`'s raw output is specified exactly by the standard, but the `<random>`
/// distributions are not: libstdc++, libc++ and MSVC turn the same raw output into different
/// integers and reals. A corpus must be reproducible from its seed (RFC 0019.13 §5.8), so every
/// draw is built here from the raw output with fixed arithmetic. The normal draw uses
/// `std::log` and `std::cos`, which a libm may round differently in the last place; that
/// perturbs a direction by about 1e-16, far below the rounding to whole shape values.
namespace hipdnn_corpus_gen
{
namespace detail
{

/// An integer in [0, @p bound) with no modulo bias: raw outputs below `2^64 mod bound` are
/// refused so every residue is equally likely. Returns 0 for a zero bound.
inline uint64_t uniformBelow(std::mt19937_64& rng, uint64_t bound)
{
    if(bound == 0)
    {
        return 0;
    }
    // (2^64 - bound) % bound == 2^64 % bound, in unsigned arithmetic.
    const uint64_t refused = (uint64_t{0} - bound) % bound;
    uint64_t raw = rng();
    while(raw < refused)
    {
        raw = rng();
    }
    return raw % bound;
}

/// An index in [0, @p count). @p count must be positive.
inline size_t uniformIndex(std::mt19937_64& rng, size_t count)
{
    return static_cast<size_t>(uniformBelow(rng, static_cast<uint64_t>(count)));
}

/// A real in [0, 1) from the top 53 bits of one raw output, the width of a double's mantissa.
inline double uniformUnit(std::mt19937_64& rng)
{
    constexpr double K_SCALE = 1.0 / 9007199254740992.0; // 2^-53
    return static_cast<double>(rng() >> 11) * K_SCALE;
}

/// A real in [@p low, @p high); @p low when the interval is empty.
inline double uniformReal(std::mt19937_64& rng, double low, double high)
{
    return low + ((high - low) * uniformUnit(rng));
}

/// True with probability @p probability.
inline bool bernoulli(std::mt19937_64& rng, double probability)
{
    return uniformUnit(rng) < probability;
}

/// A standard normal draw by the Box-Muller transform, using one of the pair it yields.
inline double standardNormal(std::mt19937_64& rng)
{
    constexpr double K_TWO_PI = 6.283185307179586476925286766559;
    // 1 - u lies in (0, 1], so the logarithm is finite.
    const double radius = std::sqrt(-2.0 * std::log(1.0 - uniformUnit(rng)));
    return radius * std::cos(K_TWO_PI * uniformUnit(rng));
}

} // namespace detail
} // namespace hipdnn_corpus_gen
