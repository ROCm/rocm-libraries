/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace Tensilelite
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

