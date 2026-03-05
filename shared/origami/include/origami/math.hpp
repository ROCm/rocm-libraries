/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

namespace origami {
namespace math {

/**
 * @brief Performs `(n + d - 1) / d`, but is robust against the case where
 * `(n + d - 1)` would overflow.
 *
 */
template <typename N, typename D>
inline constexpr N safe_ceil_div(N n, D d) {
  // Static cast to undo integral promotion.
  return static_cast<N>(d == 0 ? 0 : (n / d + (n % d != 0 ? 1 : 0)));
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

}  // namespace math
}  // namespace origami
