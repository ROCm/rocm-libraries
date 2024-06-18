#pragma once

#include <rocRoller/DataTypes/DataTypes_FP6_Utils.hpp>
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

        if constexpr(std::is_same_v<T, FP6>)
        {
            std::vector<FP6x16> y(nx / 16);
            packFP6x16((uint32_t*)y.data(), (uint8_t const*)x.data(), nx);
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
