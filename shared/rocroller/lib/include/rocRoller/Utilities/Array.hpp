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

#include <array>
#include <concepts>
#include <memory>
#include <ranges>

#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    /**
     * A collection of helper functions to deal with the sparsely populated,
     * statically allocated std::array of shared_ptr and optional values used
     * in rocRoller.
     */

    /**
     * Returns an iterator to the first empty slot in `array`, or `end` if the
     * array is full.
     */
    template <typename T, size_t N>
    auto emptySlot(std::array<T, N> const& array);

    /**
     * Returns an iterator to the first empty slot in `array`, or `end` if the
     * array is full.
     */
    template <typename T, size_t N>
    auto emptySlot(std::array<T, N>& array);

    /**
     * Places `value` in the first empty slot in `array`, or throws if the
     * array is full.
     */
    template <typename T, size_t N>
    void append(std::array<T, N>& array, T value);

    /**
     * Places `values` in the first empty slots in `array`, or throws if the
     * array is full.
     */
    template <typename T, size_t N, typename C>
    requires CInputRangeOf<C, T>
    void append(std::array<T, N>& array, C values);
}

#include "Array_impl.hpp"
