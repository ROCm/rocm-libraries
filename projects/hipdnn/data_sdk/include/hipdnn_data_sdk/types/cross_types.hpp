// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file cross_types.hpp
 * @brief Cross-type constructor definitions for hipDNN portable types.
 *
 * This header provides explicit constructor definitions that enable
 * conversion between different custom floating-point types (bfloat16,
 * half, fp4_e2m1, fp6_e2m3, fp6_e3m2, fp8_e4m3, fp8_e5m2, fp8_e8m0).
 * All conversions go through float as an intermediate representation.
 *
 * @note This header must be included after all type headers are included,
 * as it defines out-of-line constructors that were declared in the
 * individual type headers.
 */

#include "Bfloat16.hpp"
#include "Fp4E2M1.hpp"
#include "Fp6E2M3.hpp"
#include "Fp6E3M2.hpp"
#include "Fp8E4M3.hpp"
#include "Fp8E4M3Fnuz.hpp"
#include "Fp8E5M2.hpp"
#include "Fp8E5M2Fnuz.hpp"
#include "Fp8E8M0.hpp"
#include "Half.hpp"

namespace hipdnn_data_sdk::types
{

// ============================================================================
// bfloat16_t cross-type constructors (templated for any rounding mode)
// ============================================================================

template <Bfloat16RoundingMode RoundMode>
inline bfloat16_t<RoundMode>::bfloat16_t(half h) noexcept
    : bfloat16_t(static_cast<float>(h))
{
}

template <Bfloat16RoundingMode RoundMode>
inline bfloat16_t<RoundMode>::bfloat16_t(fp4_e2m1 f) noexcept
    : bfloat16_t(static_cast<float>(f))
{
}

template <Bfloat16RoundingMode RoundMode>
inline bfloat16_t<RoundMode>::bfloat16_t(fp6_e2m3 f) noexcept
    : bfloat16_t(static_cast<float>(f))
{
}

template <Bfloat16RoundingMode RoundMode>
inline bfloat16_t<RoundMode>::bfloat16_t(fp6_e3m2 f) noexcept
    : bfloat16_t(static_cast<float>(f))
{
}

template <Bfloat16RoundingMode RoundMode>
inline bfloat16_t<RoundMode>::bfloat16_t(fp8_e4m3 f) noexcept
    : bfloat16_t(static_cast<float>(f))
{
}

template <Bfloat16RoundingMode RoundMode>
inline bfloat16_t<RoundMode>::bfloat16_t(fp8_e4m3_fnuz f) noexcept
    : bfloat16_t(static_cast<float>(f))
{
}

template <Bfloat16RoundingMode RoundMode>
inline bfloat16_t<RoundMode>::bfloat16_t(fp8_e5m2 f) noexcept
    : bfloat16_t(static_cast<float>(f))
{
}

template <Bfloat16RoundingMode RoundMode>
inline bfloat16_t<RoundMode>::bfloat16_t(fp8_e5m2_fnuz f) noexcept
    : bfloat16_t(static_cast<float>(f))
{
}

template <Bfloat16RoundingMode RoundMode>
inline bfloat16_t<RoundMode>::bfloat16_t(fp8_e8m0 f) noexcept
    : bfloat16_t(static_cast<float>(f))
{
}

// ============================================================================
// half cross-type constructors
// ============================================================================

template <Bfloat16RoundingMode M>
inline half::half(bfloat16_t<M> b) noexcept
    : half(static_cast<float>(b))
{
}

inline half::half(fp4_e2m1 f) noexcept
    : half(static_cast<float>(f))
{
}

inline half::half(fp6_e2m3 f) noexcept
    : half(static_cast<float>(f))
{
}

inline half::half(fp6_e3m2 f) noexcept
    : half(static_cast<float>(f))
{
}

inline half::half(fp8_e4m3 f) noexcept
    : half(static_cast<float>(f))
{
}

inline half::half(fp8_e4m3_fnuz f) noexcept
    : half(static_cast<float>(f))
{
}

inline half::half(fp8_e5m2 f) noexcept
    : half(static_cast<float>(f))
{
}

inline half::half(fp8_e5m2_fnuz f) noexcept
    : half(static_cast<float>(f))
{
}

inline half::half(fp8_e8m0 f) noexcept
    : half(static_cast<float>(f))
{
}

// ============================================================================
// fp4_e2m1 cross-type constructors
// ============================================================================

template <Bfloat16RoundingMode M>
inline fp4_e2m1::fp4_e2m1(bfloat16_t<M> b) noexcept
    : fp4_e2m1(static_cast<float>(b))
{
}

inline fp4_e2m1::fp4_e2m1(half h) noexcept
    : fp4_e2m1(static_cast<float>(h))
{
}

// ============================================================================
// fp6_e2m3 cross-type constructors
// ============================================================================

template <Bfloat16RoundingMode M>
inline fp6_e2m3::fp6_e2m3(bfloat16_t<M> b) noexcept
    : fp6_e2m3(static_cast<float>(b))
{
}

inline fp6_e2m3::fp6_e2m3(half h) noexcept
    : fp6_e2m3(static_cast<float>(h))
{
}

// ============================================================================
// fp6_e3m2 cross-type constructors
// ============================================================================

template <Bfloat16RoundingMode M>
inline fp6_e3m2::fp6_e3m2(bfloat16_t<M> b) noexcept
    : fp6_e3m2(static_cast<float>(b))
{
}

inline fp6_e3m2::fp6_e3m2(half h) noexcept
    : fp6_e3m2(static_cast<float>(h))
{
}

// ============================================================================
// fp8_e4m3 cross-type constructors
// ============================================================================

template <Bfloat16RoundingMode M>
inline fp8_e4m3::fp8_e4m3(bfloat16_t<M> b) noexcept
    : fp8_e4m3(static_cast<float>(b))
{
}

inline fp8_e4m3::fp8_e4m3(half h) noexcept
    : fp8_e4m3(static_cast<float>(h))
{
}

inline fp8_e4m3::fp8_e4m3(fp8_e5m2 f) noexcept
    : fp8_e4m3(static_cast<float>(f))
{
}

// ============================================================================
// fp8_e4m3_fnuz cross-type constructors
// ============================================================================

template <Bfloat16RoundingMode M>
inline fp8_e4m3_fnuz::fp8_e4m3_fnuz(bfloat16_t<M> b) noexcept
    : fp8_e4m3_fnuz(static_cast<float>(b))
{
}

inline fp8_e4m3_fnuz::fp8_e4m3_fnuz(half h) noexcept
    : fp8_e4m3_fnuz(static_cast<float>(h))
{
}

inline fp8_e4m3_fnuz::fp8_e4m3_fnuz(fp8_e5m2_fnuz f) noexcept
    : fp8_e4m3_fnuz(static_cast<float>(f))
{
}

// ============================================================================
// fp8_e5m2 cross-type constructors
// ============================================================================

template <Bfloat16RoundingMode M>
inline fp8_e5m2::fp8_e5m2(bfloat16_t<M> b) noexcept
    : fp8_e5m2(static_cast<float>(b))
{
}

inline fp8_e5m2::fp8_e5m2(half h) noexcept
    : fp8_e5m2(static_cast<float>(h))
{
}

inline fp8_e5m2::fp8_e5m2(fp8_e4m3 f) noexcept
    : fp8_e5m2(static_cast<float>(f))
{
}

// ============================================================================
// fp8_e5m2_fnuz cross-type constructors
// ============================================================================

template <Bfloat16RoundingMode M>
inline fp8_e5m2_fnuz::fp8_e5m2_fnuz(bfloat16_t<M> b) noexcept
    : fp8_e5m2_fnuz(static_cast<float>(b))
{
}

inline fp8_e5m2_fnuz::fp8_e5m2_fnuz(half h) noexcept
    : fp8_e5m2_fnuz(static_cast<float>(h))
{
}

inline fp8_e5m2_fnuz::fp8_e5m2_fnuz(fp8_e4m3_fnuz f) noexcept
    : fp8_e5m2_fnuz(static_cast<float>(f))
{
}

// ============================================================================
// fp8_e8m0 cross-type constructors
// ============================================================================

template <Bfloat16RoundingMode M>
inline fp8_e8m0::fp8_e8m0(bfloat16_t<M> b) noexcept
    : fp8_e8m0(static_cast<float>(b))
{
}

inline fp8_e8m0::fp8_e8m0(half h) noexcept
    : fp8_e8m0(static_cast<float>(h))
{
}

} // namespace hipdnn_data_sdk::types
