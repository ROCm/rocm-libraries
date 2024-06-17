#pragma once

#include <vector>

#include <rocRoller/DataTypes/DataTypes.hpp>

namespace rocRoller
{
    std::vector<uint8_t> unpackFP4x8(uint32_t* x, size_t n);
    void                 packFP4x8(uint32_t* out, uint8_t const* data, int n);

    std::vector<uint32_t> f32_to_fp4x8(std::vector<float> f32);
    std::vector<float>    fp4x8_to_f32(std::vector<uint32_t> f4x8);
}
