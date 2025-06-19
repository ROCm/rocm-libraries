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

#include <vector>

#include <rocRoller/DataTypes/DataTypes.hpp>

namespace rocRoller
{
    template <typename T>
    struct PackedTypeOf
    {
        typedef T type;
    };

    template <>
    struct PackedTypeOf<FP6>
    {
        typedef FP6x16 type;
    };

    template <>
    struct PackedTypeOf<BF6>
    {
        typedef BF6x16 type;
    };

    template <>
    struct PackedTypeOf<FP4>
    {
        typedef FP4x8 type;
    };

    template <>
    struct PackedTypeOf<E8M0>
    {
        typedef E8M0x4 type;
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

    bool isF16(DataType type);
    bool isUnpackedF16(DataType type);
    bool isF8(DataType type);
    bool isUnpackedF8(DataType type);
    bool isF6(DataType type);
    bool isUnpackedF6(DataType type);
    bool isF4(DataType type);
    bool isUnpackedF4(DataType type);

    uint packingFactorForDataType(DataType type);

    uint8_t floatToScale(DataType scaleType, float value);
    float   scaleToFloat(DataType scaleType, uint8_t scale);

    inline constexpr bool isScaleType(DataType type)
    {
        switch(type)
        {
        case DataType::E8M0:
        case DataType::E8M0x4:
            return true;
        default:
            return false;
        };
    }
}
