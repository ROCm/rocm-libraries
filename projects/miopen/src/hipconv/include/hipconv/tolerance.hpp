#pragma once

namespace hipconv
{

// The rtol reported when no error model applies to a layer.
//
// The bound rests on gamma(n, u) = nu/(1 - nu), which holds only under nu < 1; past that none is
// available from it. A bound that is available can still be too loose to check against, which the
// caller judges; see docs/algorithms/direct/direct-wgrad-tolerance.md.
inline constexpr float TOLERANCE_UNAVAILABLE = -1.0f;

// Whether a reported rtol is a bound.
inline constexpr bool has_tolerance(float rtol)
{
    return rtol >= 0.0f;
}

} // namespace hipconv
