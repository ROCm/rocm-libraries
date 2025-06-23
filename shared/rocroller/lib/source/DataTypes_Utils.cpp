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

#include <rocRoller/DataTypes/DataTypes_Utils.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    uint8_t getLow(uint8_t twoFp4)
    {
        uint32_t ret = twoFp4 & 0xf;
        uint8_t  fp4 = ret;
        return fp4;
    }

    uint8_t getHigh(uint8_t twoFp4)
    {
        uint32_t ret = (twoFp4 >> 4) & 0xf;
        uint8_t  fp4 = ret;
        return fp4;
    }

    uint8_t getFp4(uint8_t twoFp4, int high)
    {
        if(high == 1)
        {
            return getHigh(twoFp4);
        }
        else
        {
            return getLow(twoFp4);
        }
    }

    void setLow(uint8_t* twoFp4, uint8_t fp4)
    {
        uint32_t value = fp4;
        *twoFp4        = *twoFp4 & 0xf0;
        value          = value & 0xf;
        *twoFp4        = *twoFp4 | value;
    }

    void setHigh(uint8_t* twoFp4, uint8_t fp4)
    {
        uint32_t value = fp4;
        *twoFp4        = *twoFp4 & 0x0f;
        value          = value & 0xf;
        value          = value << 4;
        *twoFp4        = *twoFp4 | value;
    }

    uint8_t getFp4(const uint8_t* const buffer, int index)
    {
        int     high   = index % 2;
        uint8_t twoFp4 = buffer[index / 2];
        uint8_t ret    = getFp4(twoFp4, high);
        return ret;
    }

    void setFp4(uint8_t* buffer, uint8_t value, int index)
    {
        int high = index % 2;
        if(high == 1)
        {
            setHigh(buffer + index / 2, value);
        }
        else
        {
            setLow(buffer + index / 2, value);
        }
    }

    template <typename T>
    std::vector<T> unpackFP4x8(uint32_t const* x, size_t n)
    {
        auto rv = std::vector<T>(n * 8);

        for(int i = 0; i < n * 8; ++i)
        {
            uint8_t value = getFp4(reinterpret_cast<uint8_t const*>(x), i);
            if constexpr(std::is_same_v<T, uint8_t>)
                rv[i] = value;
            else if constexpr(std::is_same_v<T, float>)
            {
                uint4_t in;
                in.val    = value;
                float f32 = fp4_to_f32<float>(in);
                rv[i]     = f32;
            }
            else
                Throw<FatalError>("Unable to unpack FP4x8: unhandled data type.");
        }
        return rv;
    }

    std::vector<float> unpackFP4x8(std::vector<FP4x8> const& f4x8)
    {
        return unpackFP4x8<float>(reinterpret_cast<uint32_t const*>(f4x8.data()), f4x8.size());
    }

    std::vector<uint8_t> unpackFP4x8(std::vector<uint32_t> const& f4x8regs)
    {
        return unpackFP4x8<uint8_t>(f4x8regs.data(), f4x8regs.size());
    }

    void packFP4x8(uint32_t* out, uint8_t const* data, size_t n)
    {
        AssertFatal(n % 8 == 0, "Number of F4 values must be a multiple of 8.");
        std::memset(out, 0, 4 * n / 8);
        for(int i = 0; i < n; ++i)
            setFp4(reinterpret_cast<uint8_t*>(out), data[i], i);
        return;
    }

    std::vector<uint32_t> packFP4x8(std::vector<uint8_t> const& f4bytes)
    {
        std::vector<uint32_t> f4x8regs(f4bytes.size() / 8);
        packFP4x8(f4x8regs.data(), f4bytes.data(), f4bytes.size());
        return f4x8regs;
    }

    std::vector<uint32_t> f32_to_fp4x8(std::vector<float> f32)
    {
        AssertFatal(f32.size() % 8 == 0, "Invalid FP32 size");
        std::vector<uint8_t> data;
        for(auto const& value : f32)
        {
            FP4 fp4 = FP4(value);
            data.push_back(reinterpret_cast<uint8_t&>(fp4));
        }
        return packFP4x8(data);
    }

    std::vector<float> fp4x8_to_f32(std::vector<uint32_t> in)
    {
        return unpackFP4x8<float>(reinterpret_cast<uint32_t const*>(in.data()), in.size());
    }

    uint8_t getF6(uint8_t const* buffer, int index)
    {
        int p1, p2, cp1;
        p1  = index / 4;
        p2  = index % 4;
        cp1 = p1 * 3;

        uint8_t temp1 = 0;
        uint8_t temp2 = 0;

        uint8_t ret = 0;
        switch(p2)
        {
        case 0:
            temp1 = buffer[cp1];
            ret   = temp1 & 0x3f;
            break;
        case 1:
            temp1 = buffer[cp1];
            temp2 = buffer[cp1 + 1];
            ret   = ((temp1 & 0xc0) >> 6) | ((temp2 & 0xf) << 2);
            break;
        case 2:
            temp1 = buffer[cp1 + 1];
            temp2 = buffer[cp1 + 2];
            ret   = ((temp1 & 0xf0) >> 4) | ((temp2 & 0x3) << 4);
            break;
        case 3:
            temp1 = buffer[cp1 + 2];
            ret   = (temp1 & 0xfc) >> 2;
            break;
        }

        return ret;
    }

    void setF6(uint8_t* buffer, uint8_t value, int index)
    {
        int p1, p2, cp1;
        p1  = index / 4;
        p2  = index % 4;
        cp1 = p1 * 3;

        uint8_t temp1 = 0;
        uint8_t temp2 = 0;
        uint8_t save  = value;
        switch(p2)
        {
        case 0:
            temp1       = buffer[cp1];
            buffer[cp1] = (temp1 & 0xc0) | save;
            break;
        case 1:
            temp1           = buffer[cp1];
            temp2           = buffer[cp1 + 1];
            buffer[cp1]     = ((save & 0x3) << 6) | (temp1 & 0x3f);
            buffer[cp1 + 1] = (temp2 & 0xf) | ((save & 0x3c) >> 2);
            break;
        case 2:
            temp1           = buffer[cp1 + 1];
            temp2           = buffer[cp1 + 2];
            buffer[cp1 + 1] = ((save & 0xf) << 4) | (temp1 & 0xf);
            buffer[cp1 + 2] = ((save & 0x30) >> 4) | (temp2 & 0x3);
            break;
        case 3:
            temp1           = buffer[cp1 + 2];
            buffer[cp1 + 2] = (save << 2) | (temp1 & 0x3);
            break;
        }
    }

    template <typename DstType, typename SrcType>
    std::vector<DstType> unpackF6x16(uint32_t const* x, size_t n)
    {
        AssertFatal(n % 3 == 0, "Number of F6x16 registers must be a multiple of 3.");
        auto rv = std::vector<DstType>(n / 3 * 16);
        for(int i = 0; i < n / 3 * 16; ++i)
        {
            auto v = getF6(reinterpret_cast<uint8_t const*>(x), i);
            if constexpr(std::is_same_v<DstType, uint8_t>)
                rv[i] = v;
            else if constexpr(std::is_same_v<SrcType, FP6x16>)
                rv[i] = cast_from_f6<DstType>(v, DataTypes::FP6_FMT);
            else
                rv[i] = cast_from_f6<DstType>(v, DataTypes::BF6_FMT);
        }
        return rv;
    }

    std::vector<float> unpackFP6x16(std::vector<FP6x16> const& f6x16)
    {
        return unpackF6x16<float, FP6x16>(reinterpret_cast<uint32_t const*>(f6x16.data()),
                                          3 * f6x16.size());
    }

    std::vector<float> unpackBF6x16(std::vector<BF6x16> const& f6x16)
    {
        return unpackF6x16<float, BF6x16>(reinterpret_cast<uint32_t const*>(f6x16.data()),
                                          3 * f6x16.size());
    }

    std::vector<uint8_t> unpackF6x16(std::vector<uint32_t> const& f6x16regs)
    {
        return unpackF6x16<uint8_t, uint32_t>(f6x16regs.data(), f6x16regs.size());
    }

    void packF6x16(uint32_t* out, uint8_t const* data, size_t n)
    {
        AssertFatal(n % 16 == 0, "Number of F6 values must be a multiple of 16.");
        std::memset(out, 0, 6 * n / 8);
        for(int i = 0; i < n; ++i)
            setF6(reinterpret_cast<uint8_t*>(out), data[i], i);
    }

    std::vector<uint32_t> packF6x16(std::vector<uint8_t> const& f6bytes)
    {
        std::vector<uint32_t> f6x16regs(3 * f6bytes.size() / 16);
        packF6x16(f6x16regs.data(), f6bytes.data(), f6bytes.size());
        return f6x16regs;
    }

    std::vector<float> unpackToFloat(std::vector<Half> const& x)
    {
        auto n = x.size();

        std::vector<float> rv(n);
        for(auto i = 0; i < n; ++i)
            rv[i] = float(x[i]);
        return rv;
    }

    std::vector<float> unpackToFloat(std::vector<Halfx2> const& halfx2)
    {
        auto n       = halfx2.size() * 2;
        auto halfptr = reinterpret_cast<Half const*>(halfx2.data());

        std::vector<float> rv(n);
        for(auto i = 0; i < n; ++i)
            rv[i] = float(halfptr[i]);
        return rv;
    }

    std::vector<float> unpackToFloat(std::vector<BFloat16> const& x)
    {
        auto n = x.size();

        std::vector<float> rv(n);
        for(auto i = 0; i < n; ++i)
            rv[i] = float(x[i]);
        return rv;
    }

    std::vector<float> unpackToFloat(std::vector<FP8> const& f8)
    {
        auto n = f8.size();

        std::vector<float> rv(n);
        for(auto i = 0; i < n; ++i)
            rv[i] = float(f8[i]);
        return rv;
    }

    std::vector<float> unpackToFloat(std::vector<BF8> const& f8)
    {
        auto n = f8.size();

        std::vector<float> rv(n);
        for(auto i = 0; i < n; ++i)
            rv[i] = float(f8[i]);
        return rv;
    }

    std::vector<float> unpackToFloat(std::vector<FP8x4> const& f8x4)
    {
        auto n     = f8x4.size() * 4;
        auto f8ptr = reinterpret_cast<FP8 const*>(f8x4.data());

        std::vector<float> rv(n);
        for(auto i = 0; i < n; ++i)
            rv[i] = float(f8ptr[i]);
        return rv;
    }

    std::vector<float> unpackToFloat(std::vector<BF8x4> const& f8x4)
    {
        auto n     = f8x4.size() * 4;
        auto f8ptr = reinterpret_cast<BF8 const*>(f8x4.data());

        std::vector<float> rv(n);
        for(auto i = 0; i < n; ++i)
            rv[i] = float(f8ptr[i]);
        return rv;
    }

    std::vector<float> unpackToFloat(std::vector<FP6x16> const& f6x16)
    {
        return unpackF6x16<float, FP6x16>(reinterpret_cast<uint32_t const*>(f6x16.data()),
                                          3 * f6x16.size());
    }

    std::vector<float> unpackToFloat(std::vector<BF6x16> const& f6x16)
    {
        return unpackF6x16<float, BF6x16>(reinterpret_cast<uint32_t const*>(f6x16.data()),
                                          3 * f6x16.size());
    }

    std::vector<float> unpackToFloat(std::vector<FP4x8> const& f4x8)
    {
        return unpackFP4x8<float>(reinterpret_cast<uint32_t const*>(f4x8.data()), f4x8.size());
    }

    bool isF16(DataType type)
    {
        switch(type)
        {
        case DataType::Half:
        case DataType::Halfx2:
        case DataType::BFloat16:
        case DataType::BFloat16x2:
            return true;
        default:
            return false;
        };
    }

    bool isUnpackedF16(DataType type)
    {
        switch(type)
        {
        case DataType::Half:
        case DataType::BFloat16:
            return true;
        default:
            return false;
        };
    }

    bool isF8(DataType type)
    {
        switch(type)
        {
        case DataType::FP8:
        case DataType::FP8x4:
        case DataType::BF8:
        case DataType::BF8x4:
            return true;
        default:
            return false;
        };
    }

    bool isUnpackedF8(DataType type)
    {
        switch(type)
        {
        case DataType::FP8:
        case DataType::BF8:
            return true;
        default:
            return false;
        };
    }

    bool isF6(DataType type)
    {
        switch(type)
        {
        case DataType::FP6:
        case DataType::FP6x16:
        case DataType::BF6:
        case DataType::BF6x16:
            return true;
        default:
            return false;
        };
    }

    bool isUnpackedF6(DataType type)
    {
        switch(type)
        {
        case DataType::FP6:
        case DataType::BF6:
            return true;
        default:
            return false;
        };
    }

    bool isF4(DataType type)
    {
        switch(type)
        {
        case DataType::FP4:
        case DataType::FP4x8:
            return true;
        default:
            return false;
        };
    }

    bool isUnpackedF4(DataType type)
    {
        return type == DataType::FP4;
    }

    uint packingFactorForDataType(DataType type)
    {
        auto packing = DataTypeInfo::Get(type).packing;
        if(packing == 1)
        {
            auto maybePackedType = DataTypeInfo::Get(type).packedVariableType();
            if(maybePackedType)
            {
                return DataTypeInfo::Get(maybePackedType->dataType).packing;
            }
        }
        return packing;
    }

    uint8_t floatToScale(DataType scaleType, float value)
    {
        AssertFatal(isScaleType(scaleType));

        uint8_t scale = 0;
        switch(scaleType)
        {
        case DataType::E8M0:
            scale = static_cast<uint8_t>(floatToScale<E8M0>(value));
            break;
        default:
            AssertFatal(
                false, "floatToScale is unimplemented for scale type: ", ShowValue(scaleType));
        }

        return scale;
    }

    float scaleToFloat(DataType scaleType, uint8_t scale)
    {
        AssertFatal(isScaleType(scaleType));

        float value = 0;
        switch(scaleType)
        {
        case DataType::E8M0:
            value = scaleToFloat<E8M0>(static_cast<E8M0>(scale));
            break;
        default:
            AssertFatal(
                false, "scaleToFloat is unimplemented for scale type: ", ShowValue(scaleType));
        }

        return value;
    }

};
