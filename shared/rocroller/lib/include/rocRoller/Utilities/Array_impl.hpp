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

#include "Array.hpp"

namespace rocRoller
{
    template <typename T, size_t N>
    auto emptySlot(std::array<T, N> const& array)
    {
        return std::ranges::find_if(array, [](auto const& slot) { return !slot; });
    }

    template <typename T, size_t N>
    auto emptySlot(std::array<T, N>& array)
    {
        return std::ranges::find_if(array, [](auto const& slot) { return !slot; });
    }

    template <typename T, size_t N>
    void append(std::array<T, N>& array, T value)
    {
        auto iter = emptySlot(array);
        AssertFatal(iter != array.end(), "Array is full!");
        *iter = std::move(value);
    }

    template <typename T, size_t N, typename C>
    requires CInputRangeOf<C, T>
    void append(std::array<T, N>& array, C values)
    {
        auto arrayIter = emptySlot(array);

        if constexpr(std::ranges::sized_range<C>)
        {
            auto remainingSlots = array.end() - arrayIter;
            auto numValues      = std::ranges::size(values);
            AssertFatal(remainingSlots >= numValues,
                        "Array is full!",
                        ShowValue(remainingSlots),
                        ShowValue(numValues));
        }

        auto valueIter = values.begin();
        for(; valueIter != values.end() && arrayIter != array.end(); ++arrayIter, ++valueIter)
        {
            AssertFatal(!*arrayIter, "Array values not contiguous!");
            *arrayIter = std::move(*valueIter);
        }

        AssertFatal(valueIter == values.end(), "Array is full!");
    }

}
