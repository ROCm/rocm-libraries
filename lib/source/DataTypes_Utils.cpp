#include <rocRoller/DataTypes/DataTypes_Utils.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    static uint8_t getlow(uint8_t twoFp4)
    {
        uint32_t ret = twoFp4 & 0xf;
        uint8_t  fp4 = ret;
        return fp4;
    }

    static uint8_t gethigh(uint8_t twoFp4)
    {
        uint32_t ret = (twoFp4 >> 4) & 0xf;
        uint8_t  fp4 = ret;
        return fp4;
    }

    static uint8_t getFp4(uint8_t twoFp4, int high)
    {
        if(high == 1)
        {
            return gethigh(twoFp4);
        }
        else
        {
            return getlow(twoFp4);
        }
    }

    static void setlow(uint8_t* twoFp4, uint8_t fp4)
    {
        uint32_t value = fp4;
        *twoFp4        = *twoFp4 & 0xf0;
        value          = value & 0xf;
        *twoFp4        = *twoFp4 | value;
    }

    static void sethigh(uint8_t* twoFp4, uint8_t fp4)
    {
        uint32_t value = fp4;
        *twoFp4        = *twoFp4 & 0x0f;
        value          = value & 0xf;
        value          = value << 4;
        *twoFp4        = *twoFp4 | value;
    }

    static void setFp4(uint8_t* twoFp4, uint8_t value, int high)
    {
        if(high == 1)
        {
            sethigh(twoFp4, value);
        }
        else
        {
            setlow(twoFp4, value);
        }
    }

    // Library function that gets fp4 from matrix
    static uint8_t getFp4(
        const uint8_t* const buffer, int m, int n, int i, int j, int rowmajor, bool debug = false)
    {
        int index = 0;
        if(rowmajor == 1)
        {
            index = i * n + j;
        }
        else
        {
            index = j * m + i;
        }
        int high = index % 2;

        uint8_t twoFp4 = buffer[index / 2];
        uint8_t ret    = getFp4(twoFp4, high);
        if(debug)
        {
            printf("m:%d, n:%d, i:%d, j:%d, index:%d, rowmajor:%d, ret:%01x\n",
                   m,
                   n,
                   i,
                   j,
                   index,
                   rowmajor,
                   ret);
        }
        return ret;
    }

    // Library function that sets fp4 to matrix
    static void setFp4(uint8_t* buffer, uint8_t value, int m, int n, int i, int j, int rowmajor)
    {
        int index = 0;
        if(rowmajor == 1)
        {
            index = i * n + j;
        }
        else
        {
            index = j * m + i;
        }
        int high = index % 2;

        setFp4(buffer + index / 2, value, high);
    }

    template <typename T>
    std::vector<T> unpackFP4x8(uint32_t const* x, size_t n)
    {
        auto rv = std::vector<T>(n * 8);

        for(int i = 0; i < n * 8; ++i)
        {
            uint8_t value = getFp4(reinterpret_cast<uint8_t const*>(x), 0, 0, i, 0, 0);
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

    void packFP4x8(uint32_t* out, uint8_t const* data, int n)
    {
        for(int i = 0; i < n; ++i)
            setFp4(reinterpret_cast<uint8_t*>(out), data[i], 0, 0, i, 0, 0);
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
        AssertFatal(n % 3 == 0, "Number of F6x16 registers must be a multiple 3.");
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

    void packF6x16(uint32_t* out, uint8_t const* data, int n)
    {
        AssertFatal(n % 16 == 0, "Number of F6 values must be a multiple 16.");

        for(int i = 0; i < n; ++i)
            setF6(reinterpret_cast<uint8_t*>(out), data[i], i);
    }

    std::vector<uint32_t> packF6x16(std::vector<uint8_t> const& f6bytes)
    {
        std::vector<uint32_t> f6x16regs(3 * f6bytes.size() / 16);
        packF6x16(f6x16regs.data(), f6bytes.data(), f6bytes.size());
        return f6x16regs;
    }
};
