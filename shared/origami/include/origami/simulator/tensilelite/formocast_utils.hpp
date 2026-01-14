// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace origami
{
    namespace Utils
    {
        /**
         * @brief Ceiling division for unsigned integers
         * @param numerator The dividend
         * @param denominator The divisor
         * @return The result of ceiling(numerator / denominator)
         * @throws std::invalid_argument if denominator is zero
         */
        inline uint32_t ceilDivide(uint32_t numerator, uint32_t denominator)
        {
            if (denominator == 0) {
                throw std::invalid_argument("Denominator cannot be zero");
            }
            return (numerator + denominator - 1) / denominator;
        }

        /**
         * @brief Ceiling math function for floating point numbers
         * @param value The value to round up
         * @param significance The multiple to round up to (default: 1)
         * @return The smallest multiple of significance that is >= value
         */
        inline double ceiling_math(double value, double significance = 1.0)
        {
            return std::ceil(value / significance) * significance;
        }

    } // namespace Utils
} // namespace Tensilelite

