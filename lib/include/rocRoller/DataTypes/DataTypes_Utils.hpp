#pragma once

#include <vector>

#include <rocRoller/DataTypes/DataTypes.hpp>

namespace rocRoller
{
    template <typename T>
    struct UnsegmentedTypeOf
    {
        typedef T type;
    };

    template <>
    struct UnsegmentedTypeOf<FP6>
    {
        typedef FP6x16 type;
    };

    template <>
    struct UnsegmentedTypeOf<BF6>
    {
        typedef BF6x16 type;
    };

    template <>
    struct UnsegmentedTypeOf<FP4>
    {
        typedef FP4x8 type;
    };

    template <typename T>
    struct SegmentedTypeOf
    {
        typedef T type;
    };

    template <>
    struct SegmentedTypeOf<FP6x16>
    {
        typedef FP6 type;
    };

    template <>
    struct SegmentedTypeOf<BF6x16>
    {
        typedef BF6 type;
    };

    template <>
    struct SegmentedTypeOf<FP4x8>
    {
        typedef FP4 type;
    };

    void                  packFP4x8(uint32_t* out, uint8_t const* data, size_t n);
    std::vector<uint32_t> packFP4x8(std::vector<uint8_t> const&);
    std::vector<uint8_t>  unpackFP4x8(std::vector<uint32_t> const&);

    std::vector<uint32_t> f32_to_fp4x8(std::vector<float> f32);
    std::vector<float>    fp4x8_to_f32(std::vector<uint32_t> f4x8);

    void                  packF6x16(uint32_t*, uint8_t const*, size_t);
    std::vector<uint32_t> packF6x16(std::vector<uint8_t> const&);
    std::vector<uint8_t>  unpackF6x16(std::vector<uint32_t> const&);

    inline std::vector<float> unpackToFloat(std::vector<float> const& x)
    {
        return x;
    };
    std::vector<float> unpackToFloat(std::vector<Half> const&);
    std::vector<float> unpackToFloat(std::vector<Halfx2> const&);
    std::vector<float> unpackToFloat(std::vector<BFloat16> const&);
    std::vector<float> unpackToFloat(std::vector<FP8> const&);
    std::vector<float> unpackToFloat(std::vector<BF8> const&);
    std::vector<float> unpackToFloat(std::vector<FP8x4> const&);
    std::vector<float> unpackToFloat(std::vector<BF8x4> const&);
    std::vector<float> unpackToFloat(std::vector<FP6x16> const&);
    std::vector<float> unpackToFloat(std::vector<BF6x16> const&);
    std::vector<float> unpackToFloat(std::vector<FP4x8> const&);
}
