/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <rocRoller/Utilities/Random.hpp>

namespace rocRoller
{
    template <typename T, typename R>
    std::vector<typename UnsegmentedTypeOf<T>::type> RandomGenerator::vector(uint nx, R min, R max)
    {
        using U = typename UnsegmentedTypeOf<T>::type;

        std::vector<T>                   x(nx);
        std::uniform_real_distribution<> udist(min, max);

        for(unsigned i = 0; i < nx; i++)
        {
            x[i] = static_cast<T>(udist(m_gen));
        }

        if constexpr(std::is_same_v<T, U>)
        {
            return x;
        }

        if constexpr(std::is_same_v<T, FP6> || std::is_same_v<T, BF6>)
        {
            using F6x16 = std::conditional_t<std::is_same_v<T, FP6>, FP6x16, BF6x16>;
            std::vector<F6x16> y(nx / 16);
            packF6x16((uint32_t*)y.data(), (uint8_t const*)x.data(), nx);
            return y;
        }

        if constexpr(std::is_same_v<T, FP4>)
        {
            std::vector<FP4x8> y(nx / 8);
            packFP4x8((uint32_t*)y.data(), (uint8_t const*)x.data(), nx);
            return y;
        }

        Throw<FatalError>("Unhandled packing/segmentation.");
    }

    template <std::integral T>
    T RandomGenerator::next(T min, T max)
    {
        std::uniform_int_distribution<T> dist(min, max);
        return dist(m_gen);
    }
}
