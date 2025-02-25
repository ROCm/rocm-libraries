#pragma once

#include <vector>

#include <rocRoller/DataTypes/DataTypes.hpp>

namespace rocRoller
{
    void                  packFP4x8(uint32_t* out, uint8_t const* data, int n);
    std::vector<uint32_t> packFP4x8(std::vector<uint8_t> const&);
    std::vector<uint8_t>  unpackFP4x8(std::vector<uint32_t> const&);

    std::vector<uint32_t> f32_to_fp4x8(std::vector<float> f32);
    std::vector<float>    fp4x8_to_f32(std::vector<uint32_t> f4x8);

    void                  packF6x16(uint32_t*, uint8_t const*, int);
    std::vector<uint32_t> packF6x16(std::vector<uint8_t> const&);
    std::vector<uint8_t>  unpackF6x16(std::vector<uint32_t> const&);

    std::vector<float> unpackToFloat(std::vector<Half> const&);
    std::vector<float> unpackToFloat(std::vector<BFloat16> const&);
    std::vector<float> unpackToFloat(std::vector<FP8> const&);
    std::vector<float> unpackToFloat(std::vector<BF8> const&);
    std::vector<float> unpackToFloat(std::vector<FP8x4> const&);
    std::vector<float> unpackToFloat(std::vector<BF8x4> const&);
    std::vector<float> unpackToFloat(std::vector<FP6x16> const&);
    std::vector<float> unpackToFloat(std::vector<BF6x16> const&);
    std::vector<float> unpackToFloat(std::vector<FP4x8> const&);
}
