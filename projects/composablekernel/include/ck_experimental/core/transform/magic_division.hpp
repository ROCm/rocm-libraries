// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file magic_division.hpp
 *  @brief Value-based magic division constants for structural NTTP usage.
 *
 *  Provides constexpr computation and CK_TILE_HOST_DEVICE execution of magic
 *  number division. Unlike ck_tile::magic_division which returns ck_tile::tuple
 *  (incompatible with structured bindings), this returns a plain aggregate
 *  suitable for storage in structural NTTP types.
 *
 *  Magic division replaces integer division (expensive on GPU) with a
 *  multiply-shift sequence using pre-computed constants. Valid for divisors
 *  in range [1, INT32_MAX] and dividends in range [0, INT32_MAX].
 */

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include <stdint.h>

namespace ck_tile::core::transform::detail {

/** @brief Pre-computed magic division constants for one divisor.
 *
 *  Structural type (aggregate, defaulted ==) — safe for use in NTTPs.
 *  Compute via computeMagicDiv(). Execute via doMagicDiv().
 */
struct MagicDivConstants
{
    uint32_t multiplier = 0;
    uint32_t shift      = 0;

    constexpr bool operator==(const MagicDivConstants&) const = default;
};

/** @brief Compute magic division constants for a given divisor at compile time.
 *
 *  @param divisor  The divisor (must be >= 1 and <= INT32_MAX)
 *  @return Pre-computed multiplier and shift for use with doMagicDiv()
 *
 *  @pre divisor >= 1 && divisor <= INT32_MAX
 */
constexpr MagicDivConstants computeMagicDiv(uint32_t divisor)
{
    // Defensive guard for out-of-precondition divisor 0. Without this,
    // `tmp / divisor` below is division-by-zero (constexpr hard error /
    // runtime UB). The shift-loop cap below ALSO defends, but the divide
    // is the more dangerous failure mode. Returns a zero-init MagicDivConstants
    // — the same value the rest of the code's default-init produces, so
    // downstream `doMagicDiv` calls return well-defined garbage rather than
    // hanging the GPU. Callers are still expected to honor the precondition
    // (`@pre divisor >= 1`); this guard only stops the worst failure mode.
    // (Plan: Critical 2 layer 1.)
    if(divisor == 0)
    {
        return {};
    }

    uint32_t shift_val = 0;

    // Cap at 32 — `1U << 32` is UB (shift width >= type width). With a valid
    // divisor in [1, INT32_MAX] this loop terminates well before 32; the cap
    // is a defensive guard against an out-of-precondition divisor producing
    // a hang or UB. (Plan: Critical 2 layer 1.)
    while(shift_val < 32 && (1U << shift_val) < divisor)
    {
        shift_val++;
    }

    uint64_t tmp            = static_cast<uint64_t>((1UL << shift_val) - divisor) << 32;
    uint32_t multiplier_val = static_cast<uint32_t>(tmp / divisor + 1);

    return {multiplier_val, shift_val};
}

/** @brief Perform magic division at runtime.
 *
 *  Computes dividend / divisor using pre-computed magic constants.
 *  On device, uses __umulhi intrinsic when not in constexpr context.
 *  In constexpr context, uses portable 64-bit arithmetic.
 *
 *  @param dividend  The value to divide (must be non-negative)
 *  @param md        Pre-computed magic constants from computeMagicDiv()
 *  @return dividend / original_divisor
 */
CK_TILE_HOST_DEVICE constexpr uint32_t doMagicDiv(uint32_t dividend, MagicDivConstants md)
{
#ifdef __HIP_DEVICE_COMPILE__
    if(!__builtin_is_constant_evaluated())
    {
        uint32_t tmp = __umulhi(dividend, md.multiplier);
        return (tmp + dividend) >> md.shift;
    }
#endif
    uint32_t tmp = static_cast<uint32_t>((static_cast<uint64_t>(dividend) * md.multiplier) >> 32);
    return (tmp + dividend) >> md.shift;
}

/** @brief 64-bit counterpart of MagicDivConstants. Structural aggregate
 *  (defaulted ==) -- safe for NTTP / per-transform State storage.
 */
struct MagicDivConstants64
{
    uint64_t multiplier = 0;
    uint32_t shift      = 0;

    constexpr bool operator==(const MagicDivConstants64&) const = default;
};

/** @brief 64-bit magic division constants. Mirrors computeMagicDiv widened to a
 *  64-bit divisor with a 128-bit intermediate (the same Hacker's-Delight /
 *  libdivide round-up form). The precomputation runs at compile time, so the
 *  __int128 arithmetic never reaches device codegen.
 *
 *  @pre divisor >= 1 && divisor <= INT64_MAX
 */
constexpr MagicDivConstants64 computeMagicDiv64(uint64_t divisor)
{
    // Defensive guard for divisor 0 (would be a divide-by-zero below); mirrors
    // the 32-bit path. Callers must still honor `@pre divisor >= 1`.
    if(divisor == 0)
    {
        return {};
    }

    uint32_t shift_val = 0;

    // Cap at 64 -- `1ULL << 64` is UB. A valid divisor in [1, INT64_MAX]
    // terminates the loop by shift 63; the cap defends an out-of-precondition
    // divisor against a hang / UB.
    while(shift_val < 64 && (uint64_t{1} << shift_val) < divisor)
    {
        shift_val++;
    }

    unsigned __int128 tmp =
        static_cast<unsigned __int128>((uint64_t{1} << shift_val) - divisor) << 64;
    uint64_t multiplier_val = static_cast<uint64_t>(tmp / divisor + 1);

    return {multiplier_val, shift_val};
}

/** @brief 64-bit counterpart of doMagicDiv. On device uses the __umul64hi
 *  intrinsic for the high 64 bits of the 64x64 product; in constexpr / host
 *  context uses a 128-bit product.
 *
 *  Same simplified add-dividend form as the 32-bit doMagicDiv: valid for
 *  dividend in [0, INT64_MAX]. Transform coordinates are non-negative signed
 *  values (<= INT64_MAX), so the restriction is satisfied by construction.
 *
 *  @param dividend  The value to divide (must be in [0, INT64_MAX])
 *  @param md        Pre-computed constants from computeMagicDiv64()
 *  @return dividend / original_divisor
 */
CK_TILE_HOST_DEVICE constexpr uint64_t doMagicDiv64(uint64_t dividend, MagicDivConstants64 md)
{
#ifdef __HIP_DEVICE_COMPILE__
    if(!__builtin_is_constant_evaluated())
    {
        uint64_t tmp = __umul64hi(dividend, md.multiplier);
        return (tmp + dividend) >> md.shift;
    }
#endif
    uint64_t tmp = static_cast<uint64_t>(
        (static_cast<unsigned __int128>(dividend) * md.multiplier) >> 64);
    return (tmp + dividend) >> md.shift;
}

/** @brief Maps an unsigned index/length type to its magic-division constant
 *  type: 32-bit indices use MagicDivConstants, 64-bit use MagicDivConstants64.
 *  The primary template is intentionally left undefined so an unsupported width
 *  is a compile error (a new precision must add a specialization here).
 */
template <typename UIntT>
struct MagicDivFor;
template <>
struct MagicDivFor<uint32_t>
{
    using type = MagicDivConstants;
};
template <>
struct MagicDivFor<uint64_t>
{
    using type = MagicDivConstants64;
};
template <typename UIntT>
using magic_div_t = typename MagicDivFor<UIntT>::type;

/** @brief Width-dispatched magic-division entry points: select the 32- or
 *  64-bit algorithm by the unsigned index width via overload resolution, so a
 *  caller (currently only MERGE) need not spell out an
 *  `if constexpr(sizeof(Unsigned)==8)`. The dispatch lives here, next to the
 *  algorithm and the `magic_div_t` type-level selector, rather than on the
 *  precision policy (which stays a pure type/constant fact-bundle). The returned
 *  constants type is `magic_div_t<UIntT>` by construction, so it matches the
 *  `md` parameter of the corresponding `doMagicDivFor`.
 *
 *  computeMagicDivFor is host/constexpr (graph build); doMagicDivFor is
 *  CK_TILE_HOST_DEVICE (it is on MERGE's device mapCoord path via __umulhi /
 *  __umul64hi).
 */
constexpr MagicDivConstants computeMagicDivFor(uint32_t divisor) noexcept
{
    return computeMagicDiv(divisor);
}
constexpr MagicDivConstants64 computeMagicDivFor(uint64_t divisor) noexcept
{
    return computeMagicDiv64(divisor);
}

CK_TILE_HOST_DEVICE constexpr uint32_t
doMagicDivFor(uint32_t dividend, MagicDivConstants md) noexcept
{
    return doMagicDiv(dividend, md);
}
CK_TILE_HOST_DEVICE constexpr uint64_t
doMagicDivFor(uint64_t dividend, MagicDivConstants64 md) noexcept
{
    return doMagicDiv64(dividend, md);
}

} // namespace ck_tile::core::transform::detail
