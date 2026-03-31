// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file types.hpp
 * @brief Main header for all hipDNN portable data types.
 *
 * This header provides access to all portable floating-point types.
 * Custom types were added instead of using hip versions since we want to limit dependencies,
 * and we need control over math implementations to prevent implicit conversions or accuracy issues.
 *
 * **Custom Types:**
 * - bfloat16: 16-bit brain floating point (1 sign, 8 exponent, 7 mantissa)
 * - half: 16-bit IEEE 754 half precision (1 sign, 5 exponent, 10 mantissa)
 * - fp8_e4m3: 8-bit floating point (1 sign, 4 exponent, 3 mantissa) - OCP format
 * - fp8_e5m2: 8-bit floating point (1 sign, 5 exponent, 2 mantissa) - OCP format
 * - fp8_e8m0: 8-bit floating point (0 sign, 8 exponent, 0 mantissa) - OCP MX scale format
 * - fp6_e2m3: 6-bit floating point (1 sign, 2 exponent, 3 mantissa) - OCP MX format
 * - fp6_e3m2: 6-bit floating point (1 sign, 3 exponent, 2 mantissa) - OCP MX format
 * - fp4_e2m1: 4-bit floating point (1 sign, 2 exponent, 1 mantissa) - OCP MX format
 *
 * **Forwarding Functions:**
 * Also includes forwarding functions for built-in types (float, double,
 * int8_t, uint8_t, int32_t) that enable uniform unqualified calls for
 * math functions across all types via ADL (Argument Dependent Lookup).
 *
 * **Design Principles:**
 * - All types use EXPLICIT constructors and conversion operators to prevent
 *   silent precision loss and eliminate overload ambiguity issues.
 * - All custom types are trivially copyable and standard layout for binary
 *   compatibility with HIP native types.
 * - Cross-type conversions go through float as an intermediate representation.
 *
 * @example
 * @code
 * #include <hipdnn_data_sdk/types.hpp>
 *
 * using namespace hipdnn_data_sdk::types;
 *
 * half h(3.14f);
 * bfloat16 b(h);  // Cross-type conversion
 * float f = static_cast<float>(b);
 *
 * // Math functions work uniformly via ADL
 * auto absVal = abs(h);
 * bool isNan = isnan(b);
 * @endcode
 */

// Individual type headers
#include "types/Bfloat16.hpp"
#include "types/Double.hpp"
#include "types/Float.hpp"
#include "types/Fp4E2M1.hpp"
#include "types/Fp6E2M3.hpp"
#include "types/Fp6E3M2.hpp"
#include "types/Fp8E4M3.hpp"
#include "types/Fp8E5M2.hpp"
#include "types/Fp8E8M0.hpp"
#include "types/Half.hpp"
#include "types/Int32.hpp"
#include "types/Int8.hpp"
#include "types/Uint8.hpp"

// Cross-type constructor definitions (must come after all types are declared)
#include "types/cross_types.hpp"
