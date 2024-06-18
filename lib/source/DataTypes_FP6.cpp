#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/DataTypes/DataTypes_FP6.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{

    uint8_t getFp6(uint8_t const* buffer, int index)
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

    void setFp6(uint8_t* buffer, uint8_t value, int index)
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

    template <typename T>
    std::vector<T> unpackFP6x16(uint32_t const* x, size_t n)
    {
        AssertFatal(n % 3 == 0, "Number of FP6x16 registers must be a multiple 3.");
        auto rv = std::vector<T>(n / 3 * 16);
        for(int i = 0; i < n / 3 * 16; ++i)
        {
            auto v = getFp6(reinterpret_cast<uint8_t const*>(x), i);
            if constexpr(std::is_same_v<T, uint8_t>)
                rv[i] = v;
            else
                rv[i] = cast_from_fp6<T>(v, FP6_FMT);
        }
        return rv;
    }

    std::vector<float> unpackFP6x16(std::vector<FP6x16> const& f6x16)
    {
        return unpackFP6x16<float>(reinterpret_cast<uint32_t const*>(f6x16.data()),
                                   3 * f6x16.size());
    }

    std::vector<uint8_t> unpackFP6x16(std::vector<uint32_t> const& f6x16regs)
    {
        return unpackFP6x16<uint8_t>(f6x16regs.data(), f6x16regs.size());
    }

    void packFP6x16(uint32_t* out, uint8_t const* data, int n)
    {
        AssertFatal(n % 16 == 0, "Number of F6 values must be a multiple 16.");

        for(int i = 0; i < n; ++i)
            setFp6(reinterpret_cast<uint8_t*>(out), data[i], i);
    }

    std::vector<uint32_t> packFP6x16(std::vector<uint8_t> const& f6bytes)
    {
        std::vector<uint32_t> f6x16regs(3 * f6bytes.size() / 16);
        packFP6x16(f6x16regs.data(), f6bytes.data(), f6bytes.size());
        return f6x16regs;
    }
};
