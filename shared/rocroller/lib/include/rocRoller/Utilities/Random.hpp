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

#include <cstdlib>
#include <random>
#include <vector>

#include <rocRoller/DataTypes/DataTypes_Utils.hpp>

/*
 * Random vector generator.
 */

namespace rocRoller
{
    /**
     * Random vector generator.
     *
     * A seed must be passed to the constructor.  If the environment
     * variable specified by `ROCROLLER_RANDOM_SEED` is present, it
     * supercedes the seed passed to the constructor.
     *
     * A seed may be set programmatically (at any time) by calling
     * seed().
     */
    class RandomGenerator
    {
    public:
        explicit RandomGenerator(int seedNumber);

        /**
         * Set a new seed.
         */
        void seed(int seedNumber);

        /**
         * Generate a random vector of length `nx`, with values
         * between `min` and `max`.
         */
        template <typename T, typename R>
        std::vector<typename PackedTypeOf<T>::type> vector(uint nx, R min, R max);

        template <std::integral T>
        T next(T min, T max);

    private:
        std::mt19937 m_gen;
    };

    inline uint32_t constexpr LFSRRandomNumberGenerator(uint32_t const seed)
    {
        return ((seed << 1) ^ (((seed >> 31) & 1) ? 0xc5 : 0x00));
    }
}

#include <rocRoller/Utilities/Random_impl.hpp>
