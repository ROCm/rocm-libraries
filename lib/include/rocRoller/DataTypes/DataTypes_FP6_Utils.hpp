#pragma once

#include <vector>

#include <rocRoller/DataTypes/DataTypes.hpp>

namespace rocRoller
{
    void                  packFP6x16(uint32_t*, uint8_t const*, int);
    std::vector<uint32_t> packFP6x16(std::vector<uint8_t> const&);

    std::vector<uint8_t> unpackFP6x16(std::vector<uint32_t> const&);
    std::vector<float>   unpackFP6x16(std::vector<FP6x16> const&);
}
