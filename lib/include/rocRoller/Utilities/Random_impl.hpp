#pragma once

#include "Random.hpp"

namespace rocRoller
{
    template <typename T, typename R>
    std::vector<T> RandomGenerator::vector(uint nx, R min, R max)
    {
        std::vector<T>                   x(nx);
        std::uniform_real_distribution<> udist(min, max);

        for(unsigned i = 0; i < nx; i++)
        {
            x[i] = static_cast<T>(udist(m_gen));
        }

        return x;
    }

    template <std::integral T>
    T RandomGenerator::next(T min, T max)
    {
        std::uniform_int_distribution<T> dist(min, max);
        return dist(m_gen);
    }
}
