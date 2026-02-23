// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
